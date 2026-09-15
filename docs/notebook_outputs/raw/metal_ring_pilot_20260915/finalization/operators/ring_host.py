"""One fresh owned G4 continuation, preserving all previous campaign evidence.

Prepare is local-only. Allocation, uploads and teardown are separate explicit
actions. No authentication flow is initiated by this helper.
"""
import argparse
import contextlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import time

from ring_common import PROFILE, BUDGET_PROFILE, PRIOR_SECONDS, TOTAL_CAP, TRAINING_CAP, MAIN_CAP, allocation_deadline, numeric, read, require, save, sha, validate_handoff

OPS = Path(__file__).resolve().parent
ROOT = OPS.parents[4]
OLD = ROOT/"DeepMzyme_Data/notebook_outputs/campaigns/metal_architecture_pilot_10h_v1_20260915"
OLDOPS = ROOT/"DeepMzyme_Data/notebook_outputs/plans/metal_architecture_pilot_10h_v1/colab"
PREVIOUS_OPS = ROOT/"DeepMzyme_Data/notebook_outputs/plans/metal_coordination_geometry_pilot_v1/colab"
OPERATORS = ("ring_common.py", "ring_host.py", "ring_bootstrap.py", "ring_control.py", "ring_finalization.py",
             "ring_archive_verify.py", "refresh_owned_proxy.py", "watchdog.py", "test_ring_operations.py")


def config():
    value = read(OPS/"session_config.json")
    require(value and value.get("profile") == PROFILE, "Prepare a reviewed continuation configuration first")
    return value


def sanitize(value):
    return re.sub(r"(https?://[^\s?#]+)\?[^\s]+", r"\1?[REDACTED]", value)


def auth_stop(status):
    """Inspect existing authentication only; forbid further workload calls."""
    result = dict(http_status=status, stop_required=True, epoch=time.time())
    try:
        check = subprocess.run(["colab", "whoami"], capture_output=True, text=True, timeout=60)
        result.update(whoami_returncode=check.returncode, whoami_output=sanitize(check.stdout+check.stderr))
    except Exception as exc:
        result["whoami_error_type"] = type(exc).__name__
    save(OPS/"auth_stop_required.json", result)


def cli(*arguments, timeout=180):
    cfg = config()
    require(not (OPS/"auth_stop_required.json").exists() or arguments[0] in ("stop", "sessions", "whoami"),
            "Authentication failure was recorded; only inspection and owned teardown are allowed")
    if arguments[0] in ("new", "status", "exec", "upload", "download", "stop"):
        require("-s" in arguments and arguments[arguments.index("-s")+1] == cfg["session_name"],
                "Every runtime command must target this exact owned name")
    refresh = read(OPS/"runtime_proxy_refresh.json", {})
    if arguments[0] in ("status", "exec", "upload", "download") and (OPS/"owned_session.json").is_file():
        if not refresh or float(refresh["conservative_expiry_epoch"])-time.time() < 600:
            import refresh_owned_proxy
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    refresh_owned_proxy.main()
            except Exception as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status in (401, 403):
                    auth_stop(status)
                raise RuntimeError(f"Owned proxy synchronization failed: {type(exc).__name__}; HTTP {status}. Stop required.") from None
    process = subprocess.run(["colab", *map(str, arguments)], capture_output=True, text=True, timeout=timeout)
    output = sanitize(process.stdout+process.stderr)
    if process.returncode:
        save(OPS/"last_cli_error.json", dict(command=list(map(str, arguments)), returncode=process.returncode,
                                           output=output, epoch=time.time()))
        auth_error = re.search(r"(?:HTTP(?:Error)?[^\n]{0,16}\b(401|403)\b|\b(401|403)\s+(?:Client Error|Unauthorized|Forbidden))", output, re.I)
        if auth_error:
            auth_stop(int(next(group for group in auth_error.groups() if group)))
        raise RuntimeError(output)
    return output


def build_handoff():
    ledger = read(OLD/"allocation_ledger.json")
    stop = read(PREVIOUS_OPS/"session_stopped.json")
    require(stop and stop["status"] == "stopped_verified" and math.isclose(stop["cumulative_allocated_seconds"], PRIOR_SECONDS, abs_tol=1e-6, rel_tol=0),
            "Verified previous stop receipt is required")
    prior = read(OLD/"finalization/campaign_attempt_ledger.json")
    geometry = read(OLD/"finalization/coordination_geometry_budget_usage.json")
    analysis = read(OLD/"finalization/coordination_geometry_analysis_budget_usage.json")
    main = sum(float(a.get("elapsed_seconds") or 0) for a in prior if a["block"] not in ("P", "S") and not a.get("retry_of"))
    retry = sum(float(a.get("elapsed_seconds") or 0) for a in prior if a.get("retry_of"))
    source_paths = [OLD/"allocation_ledger.json", PREVIOUS_OPS/"session_stopped.json",
                   OLD/"finalization/campaign_attempt_ledger.json", OLD/"finalization/coordination_geometry_budget_usage.json",
                   OLD/"finalization/coordination_geometry_analysis_budget_usage.json", OLD/"finalization/campaign_manifest.json",
                   OLD/"finalization/expected_split.json"]
    value = dict(profile=BUDGET_PROFILE, total_cap_seconds=TOTAL_CAP, training_cap_seconds=TRAINING_CAP, main_cap_seconds=MAIN_CAP,
        prior_intervals=ledger["intervals"], prior_allocated_seconds=stop["cumulative_allocated_seconds"],
        prior_main_seconds=main+numeric(geometry["normal_elapsed_seconds"], "geometry cost")+numeric(analysis["normal_elapsed_seconds"], "analysis cost"),
        prior_retry_seconds=retry+numeric(geometry["retry_elapsed_seconds"], "geometry retry cost"),
        source_artifact_hashes={str(path): sha(path) for path in source_paths},
        parent_manifest_sha256=sha(OLD/"finalization/campaign_manifest.json"),
        expected_split_sha256=sha(OLD/"finalization/expected_split.json"),
        previous_closed_receipts_preserved=True)
    return validate_handoff(value)


def prepare(source_archive, session_name="deepmzyme-metal-ring-20260915"):
    require(not (OPS/"session_config.json").exists(), "An existing prepared configuration is immutable")
    require(re.fullmatch(r"deepmzyme-metal-ring-[a-z0-9-]+", session_name), "Use a fresh RING-specific owned session name")
    source_archive = Path(source_archive).resolve()
    with tarfile.open(source_archive) as stream:
        require(stream.getmember("src/run_metal_ring_pilot.py").isfile(), "Source archive has no RING runner")
        snapshot = json.load(stream.extractfile("code_snapshot_manifest.json"))
        for name, expected in snapshot["files"].items():
            import hashlib
            require(hashlib.sha256(stream.extractfile(name).read()).hexdigest() == expected, "Source snapshot checksum mismatch: "+name)
    handoff = build_handoff()
    local_audit = OPS.parent/"local_audit"
    audit_report = read(local_audit/"ring_input_audit.json", {})
    require(audit_report.get("status") == "passed" and audit_report.get("all_node_and_site_tensors_identical") is True,
            "The local whole-cohort RING input audit must pass before preparation or allocation")
    require(sha(local_audit/"training_cache_audit.json") == audit_report["cache_audit_sha256"], "Local audited cache proof changed")
    for name in ("ring_input_audit.json", "training_cache_audit.json", "audit_provenance.json"):
        handoff["source_artifact_hashes"][str(local_audit/name)] = sha(local_audit/name)
    receipt = read(PREVIOUS_OPS/"persistence_receipt.json")
    require(receipt.get("method") == "verified_archive_transfer" and receipt.get("drive_probe_verified")
            and receipt.get("local_write_verified") and receipt.get("drive_folder_id"), "Verified Drive/local persistence receipt required")
    reference = OPS/"parent_reference"
    reference.mkdir()
    for name in ("campaign_manifest.json", "expected_split.json", "training_cache_audit.json", "readiness.json"):
        shutil.copyfile(OLD/"finalization"/name, reference/name)
    for name in ("expected_ring_files.json", "expected_feature_files.json", "expected_cache_inventory_provenance.json"):
        require((local_audit/name).is_file(), "Missing reviewed feature inventory: "+name)
        shutil.copyfile(local_audit/name, reference/name)
        handoff["source_artifact_hashes"][str(local_audit/name)] = sha(local_audit/name)
    archive = OPS/"parent_reference.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=3) as stream:
        for path in sorted(reference.iterdir()):
            stream.add(path, arcname=path.name, recursive=False)
    save(OPS/"budget_handoff.json", handoff)
    save(OPS/"persistence_receipt.json", receipt)
    previous = read(PREVIOUS_OPS/"pilot_setup.json")
    local = ROOT/"DeepMzyme_Data/notebook_outputs/campaigns"/(PROFILE+"_20260915")
    require(not local.exists(), "New RING campaign output already exists")
    cfg = dict(profile=PROFILE, session_name=session_name, repo_root="/content/DeepMzyme_ring_v1",
        output_dir="/content/metal_ring_pilot_v1", budget_root="/content/metal_ring_continuation_budget_v1",
        parent_reference_dir="/content/metal_ring_parent_reference", local_output_dir=str(local),
        local_budget_root=str(local/"budget"), source_archive="/content/metal_ring_source.tar.gz",
        source_local_path=str(source_archive), source_sha256=sha(source_archive), source_commit=snapshot["base_commit"]+":snapshot:"+sha(source_archive),
        parent_reference_archive="/content/metal_ring_parent_reference.tar.gz", parent_reference_sha256=sha(archive),
        budget_handoff_sha256=sha(OPS/"budget_handoff.json"), prior_allocated_seconds=PRIOR_SECONDS,
        setup_limit_seconds=1800, operator_files={name: sha(OPS/name) for name in OPERATORS},
        overlay_archive="/content/metal_pilot_feature_overlay.tar.gz", overlay_local_path=str(OLDOPS/"metal-pilot-feature-overlay.tar.gz"),
        overlay_sha256=previous["overlay_sha256"], data_root=previous["data_root"], bundle_url=previous["bundle_url"],
        bundle_sha256=previous["bundle_sha256"], external_features_root_dir=previous["external_features_root_dir"],
        feature_overlay_manifest=previous["feature_overlay_manifest"], drive_folder_id=receipt["drive_folder_id"])
    operators_archive = OPS/"ring_operators.tar.gz"
    with tarfile.open(operators_archive, "w:gz", compresslevel=3) as stream:
        for name in OPERATORS:
            stream.add(OPS/name, arcname=name, recursive=False)
    cfg["operator_archive_sha256"] = sha(operators_archive)
    require(sha(Path(cfg["overlay_local_path"])) == cfg["overlay_sha256"], "Repaired feature overlay checksum mismatch")
    local.mkdir(parents=True)
    save(Path(cfg["local_budget_root"])/"allocation_ledger.json", dict(intervals=handoff["prior_intervals"]))
    save(Path(cfg["local_budget_root"])/"budget_handoff.json", handoff)
    save(OPS/"session_config.json", cfg)
    return dict(status="prepared_not_allocated", session_name=session_name, source_sha256=cfg["source_sha256"],
                budget_handoff=handoff, operators=cfg["operator_files"])


def ownership_from_history(cfg):
    history = Path("/home/mechti/.config/colab-cli/history")/(cfg["session_name"]+".jsonl")
    entries = [json.loads(line) for line in history.read_text().splitlines()] if history.is_file() else []
    created = [row for row in entries if row.get("event_type") == "session_created"]
    require(created, "No proven session_created ownership; do not target an unrelated runtime")
    row = created[-1]
    return dict(session_name=cfg["session_name"], endpoint=row["endpoint"], accelerator=row.get("accelerator"), variant=row.get("variant"),
                history_file=str(history), started_epoch=cfg["allocation_started_epoch"])


def validate_prepared_inputs(cfg):
    for name, expected in cfg["operator_files"].items():
        require(sha(OPS/name) == expected, "Prepared operator source changed: "+name)
    for path, expected in ((Path(cfg["source_local_path"]), cfg["source_sha256"]),
                           (Path(cfg["overlay_local_path"]), cfg["overlay_sha256"]),
                           (OPS/"parent_reference.tar.gz", cfg["parent_reference_sha256"]),
                           (OPS/"ring_operators.tar.gz", cfg["operator_archive_sha256"]),
                           (OPS/"budget_handoff.json", cfg["budget_handoff_sha256"])):
        require(sha(path) == expected, "Prepared input changed: "+path.name)
    handoff = validate_handoff(read(OPS/"budget_handoff.json"))
    for name, expected in handoff["source_artifact_hashes"].items():
        require(sha(Path(name)) == expected, "Prior closed evidence changed before allocation")


def allocate():
    cfg = config()
    require(not (OPS/"allocation_requested.json").exists() and not (OPS/"owned_session.json").exists(), "This namespace already requested its one allocation")
    require(not (Path("/home/mechti/.config/colab-cli/history")/(cfg["session_name"]+".jsonl")).exists(), "Owned session name has prior history; choose a fresh namespace before allocation")
    validate_prepared_inputs(cfg)
    save(OPS/"session_preallocation_config.json", cfg)
    cfg["allocation_started_epoch"] = time.time()
    save(OPS/"session_config.json", cfg)
    save(OPS/"allocation_requested.json", dict(session_name=cfg["session_name"], started_epoch=cfg["allocation_started_epoch"], gpu="G4"))
    try:
        output = cli("new", "--gpu", "G4", "-s", cfg["session_name"], timeout=180)
    except BaseException:
        # A lost response can follow allocation; keep the deadline guard alive
        # while the operator reconciles creation history and stops the owned VM.
        start_watchdog()
        raise
    start_watchdog()
    owned = ownership_from_history(cfg)
    save(OPS/"owned_session.json", owned)
    require(owned["accelerator"] == "G4" and owned["variant"] == "GPU", "Assigned hardware differs from requested G4 GPU; owned teardown required")
    ledger = read(Path(cfg["local_budget_root"])/"allocation_ledger.json")
    ledger["intervals"].append(dict(started_epoch=cfg["allocation_started_epoch"], observed_epoch=time.time()))
    save(Path(cfg["local_budget_root"])/"allocation_ledger.json", ledger)
    return dict(owned=owned, allocation_output=output, status_output=cli("status", "-s", cfg["session_name"]))


def start_watchdog():
    with (OPS/"watchdog.log").open("a") as log:
        process = subprocess.Popen([sys.executable, str(OPS/"watchdog.py")], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    save(OPS/"watchdog_process.json", dict(pid=process.pid))


def invoke(source, name="invoke_ring_remote.py"):
    path = OPS/name
    path.write_text(source)
    return cli("exec", "-s", config()["session_name"], "--timeout", "180", "-f", path, timeout=200)


def upload_inputs():
    cfg = config()
    require(read(OPS/"owned_session.json", {}).get("session_name") == cfg["session_name"], "Allocate and verify this named session first")
    require(time.time() < allocation_deadline(cfg, training=True), "Remaining training allocation time exhausted")
    validate_prepared_inputs(cfg)
    pairs = [(Path(cfg["source_local_path"]), cfg["source_archive"]), (Path(cfg["overlay_local_path"]), cfg["overlay_archive"]),
             (OPS/"parent_reference.tar.gz", cfg["parent_reference_archive"]), (OPS/"budget_handoff.json", "/content/ring_budget_handoff.json"),
             (OPS/"session_config.json", "/content/ring_session_config.json"), (OPS/"persistence_receipt.json", "/content/metal_ring_persistence_receipt.json"),
             (OPS/"ring_operators.tar.gz", "/content/ring_operators.tar.gz")]
    for source, destination in pairs:
        cli("upload", "-s", cfg["session_name"], source, destination)
    return invoke("import hashlib,json,subprocess,sys,tarfile\nfrom pathlib import Path\n"
        "cfg=json.loads(Path('/content/ring_session_config.json').read_text())\nout=Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True)\n"
        "assert hashlib.sha256(Path('/content/ring_operators.tar.gz').read_bytes()).hexdigest()==cfg['operator_archive_sha256']\n"
        "with tarfile.open('/content/ring_operators.tar.gz') as archive: archive.extractall('/content',filter='data')\n"
        "from ring_common import require_idle\nrequire_idle(out)\n"
        "with (out/'bootstrap.log').open('w') as log:\n p=subprocess.Popen([sys.executable,'/content/ring_bootstrap.py'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\n"
        "(out/'bootstrap_process.json').write_text(json.dumps({'pid':p.pid}))\nprint(json.dumps({'bootstrap_pid':p.pid}))\n")


def control(action, phase="execute"):
    command = ["/content/ring_control.py", action, "--phase", phase]
    output = invoke("import subprocess,sys\np=subprocess.run([sys.executable]+"+repr(command)+",capture_output=True,text=True)\nprint(p.stdout)\nprint(p.stderr)\np.check_returncode()\n")
    lines = [line.split("RING_CONTROL=", 1)[1] for line in output.splitlines() if "RING_CONTROL=" in line]
    require(lines, "Remote controller returned no structured result: "+output[-1800:])
    value = json.loads(lines[-1])
    save(Path(config()["local_output_dir"])/("last_"+action+"_host_result.json"), value)
    return value


def archive(action="archive"):
    cfg = config()
    value = control(action)
    require((value["attempt_id"] == "finalization") == (action == "finalize"), "Archive action and identifier differ")
    local = Path(cfg["local_output_dir"])
    if action == "finalize":
        require(value["attempt_id"] == "finalization", "Terminal archive identifier differs")
        local = local/"finalization"
    archives = local/"archives"
    archives.mkdir(parents=True, exist_ok=True)
    save(archives/(value["attempt_id"]+"_archive.json"), value)
    for part in value["parts"]:
        target = archives/Path(part["path"]).name
        if not (target.is_file() and sha(target) == part["sha256"] and target.stat().st_size == part["bytes"]):
            cli("download", "-s", cfg["session_name"], part["path"], target)
    from ring_archive_verify import verify_and_extract
    result = verify_and_extract(value, local)
    save(local/"last_download_receipt.json", result)
    return result


def receipt(drive_id, *, final=False):
    require(drive_id and drive_id.strip(), "Supply the archive Drive ID only after connector readback")
    cfg = config()
    local = Path(cfg["local_output_dir"])
    if final:
        local = local/"finalization"
    value = read(local/"last_download_receipt.json")
    require(value and value.get("local_sha256_verified") and sha(Path(value["archive"])) == value["archive_sha256"], "Verify the local archive before recording Drive transfer")
    require((value["attempt_id"] == "finalization") == final, "Transfer receipt targets the wrong archive kind")
    value.update(drive_verified=True, drive_file_id=drive_id.strip(), drive_folder_id=cfg["drive_folder_id"],
                 drive_verification="Operator supplied ID after connector readback of file, parent and byte size")
    path = local/"transfer_receipts"/(value["attempt_id"]+".json")
    save(path, value)
    invoke("from pathlib import Path\nPath("+repr(cfg["output_dir"]+"/transfer_receipts")+").mkdir(parents=True,exist_ok=True)\n")
    cli("upload", "-s", cfg["session_name"], path, cfg["output_dir"]+"/transfer_receipts/"+path.name)
    return value


def closeout():
    """Package actual verified stop and final receipts locally; never stops a VM."""
    cfg = config()
    stopped = read(OPS/"session_stopped.json", {})
    require(stopped.get("status") == "stopped_verified" and stopped.get("session_name") == cfg["session_name"], "Actual verified owned stop is required")
    owned = read(OPS/"owned_session.json", {})
    require(stopped["endpoint"] == owned.get("endpoint") and owned.get("session_name") == cfg["session_name"]
            and stopped["started_epoch"] == cfg["allocation_started_epoch"], "Stop receipt does not match the owned endpoint and allocation start")
    require(stopped["endpoint"] not in stopped["sessions_output"] and cfg["session_name"] not in stopped["sessions_output"], "Owned session absence was not verified")
    handoff = validate_handoff(read(OPS/"budget_handoff.json"))
    ledger = read(Path(cfg["local_budget_root"])/"allocation_ledger.json")
    require(ledger["intervals"][:2] == handoff["prior_intervals"] and len(ledger["intervals"]) == 3, "Closed budget intervals differ")
    require(ledger["intervals"][-1]["started_epoch"] == cfg["allocation_started_epoch"] and ledger["intervals"][-1]["ended_epoch"] == stopped["stopped_epoch"], "Actual stop interval differs")
    total = sum(row["ended_epoch"]-row["started_epoch"] for row in ledger["intervals"])
    require(math.isclose(total, stopped["cumulative_allocated_seconds"], rel_tol=0, abs_tol=1e-6), "Closed allocation total differs")
    local = Path(cfg["local_output_dir"])
    final = local/"finalization"
    receipt = read(final/"transfer_receipts/finalization.json", {})
    require(receipt.get("drive_verified") and receipt.get("local_sha256_verified") and receipt.get("attempt_id") == "finalization", "Verify final archive locally and on Drive before normal closeout packaging")
    require(sha(Path(receipt["archive"])) == receipt["archive_sha256"] == read(final/"archives/finalization_archive.json")["archive_sha256"], "Final archive proof changed")
    require(sha(final/"campaign_manifest.json") == receipt["manifest_sha256"], "Final campaign identity differs")
    download = read(final/"last_download_receipt.json", {})
    descriptor = read(final/"archives/finalization_archive.json")
    for key in ("attempt_id", "archive_sha256", "manifest_sha256", "bytes"):
        require(receipt.get(key) == download.get(key) == descriptor.get(key), "Final transfer/download/descriptor identity differs: "+key)
    require(Path(receipt["archive"]).stat().st_size == receipt["bytes"], "Final archive byte size differs")
    import hashlib
    with tarfile.open(receipt["archive"]) as stream:
        for name in ("campaign_manifest.json", "campaign_state.json", "ring_validation_results.json", "final_capture_receipt.json"):
            member = stream.getmember(name)
            require(member.isfile() and hashlib.sha256(stream.extractfile(member).read()).hexdigest() == sha(final/name),
                    "Finalization evidence differs from verified archive: "+name)
    destination = local/"host_closeout_allocation3"
    require(not destination.exists(), "Existing post-stop package is immutable")
    destination.mkdir()
    paths = {name: OPS/name for name in ("session_stopped.json", "owned_session.json", "session_config.json", "session_preallocation_config.json", "budget_handoff.json")}
    paths.update({"operators/"+name: OPS/name for name in OPERATORS})
    paths.update({"allocation_ledger.json": Path(cfg["local_budget_root"])/"allocation_ledger.json",
                  "final_transfer_receipt.json": final/"transfer_receipts/finalization.json",
                  "final_archive_descriptor.json": final/"archives/finalization_archive.json",
                  "final_capture_receipt.json": final/"final_capture_receipt.json"})
    hashes = {}
    for name, source in paths.items():
        target = destination/name
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = sha(source)
        shutil.copyfile(source, target)
        require(sha(target) == expected, "Closeout evidence changed while copying")
        hashes[name] = expected
    result = dict(status="stopped_verified_and_final_archive_preserved", cumulative_allocated_seconds=total,
                  remaining_seconds=TOTAL_CAP-total, files=hashes, final_archive_sha256=receipt["archive_sha256"],
                  final_drive_file_id=receipt["drive_file_id"], prior_allocation_evidence_preserved=True)
    save(destination/"post_stop_receipt.json", result)
    archive_path = local/"host_closeout_allocation3.tar.gz"
    require(not archive_path.exists(), "Existing post-stop archive is immutable")
    with tarfile.open(archive_path, "w:gz", compresslevel=3) as stream:
        stream.add(destination, arcname=destination.name)
    return dict(**result, archive=str(archive_path), archive_sha256=sha(archive_path))


def stop():
    cfg = config()
    existing = read(OPS/"session_stopped.json")
    if existing:
        require(existing["status"] == "stopped_verified" and existing["session_name"] == cfg["session_name"], "Existing stop receipt differs")
        return existing
    owned = read(OPS/"owned_session.json") or ownership_from_history(cfg)
    require(owned["session_name"] == cfg["session_name"], "Owned stop name differs")
    proof = ownership_from_history(cfg)
    require(owned["endpoint"] == proof["endpoint"], "Owned endpoint differs from creation history")
    save(OPS/"owned_session.json", owned)
    sys.path.append("/home/mechti/.local/share/uv/tools/google-colab-cli/lib/python3.12/site-packages")
    from colab_cli.state import SessionState, StateStore
    store = StateStore()
    entry = store.get(cfg["session_name"])
    if entry is None:
        store.add(SessionState(name=cfg["session_name"], endpoint=owned["endpoint"], token="", url="",
                               accelerator=owned.get("accelerator", "G4"), variant=owned.get("variant", "GPU")))
    else:
        require(entry.endpoint == owned["endpoint"], "Local stop mapping belongs to another endpoint")
    try:
        output = cli("stop", "-s", cfg["session_name"], timeout=60)
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        output = "Named stop response unavailable: "+sanitize(str(exc))
    sessions = cli("sessions", timeout=60)
    require(owned["endpoint"] not in sessions and cfg["session_name"] not in sessions, "Owned session is still present")
    ended = time.time()
    handoff = validate_handoff(read(OPS/"budget_handoff.json"))
    ledger = read(Path(cfg["local_budget_root"])/"allocation_ledger.json", {"intervals": handoff["prior_intervals"]})
    require(ledger["intervals"][:2] == handoff["prior_intervals"], "Prior closed intervals changed")
    current = dict(started_epoch=cfg["allocation_started_epoch"], observed_epoch=ended, ended_epoch=ended)
    if len(ledger["intervals"]) == 2:
        ledger["intervals"].append(current)
    else:
        require(len(ledger["intervals"]) == 3 and ledger["intervals"][-1]["started_epoch"] == cfg["allocation_started_epoch"], "Unexpected allocation ledger")
        require("ended_epoch" not in ledger["intervals"][-1], "This allocation already has an immutable close")
        ledger["intervals"][-1].update(current)
    save(Path(cfg["local_budget_root"])/"allocation_ledger.json", ledger)
    total = sum(row["ended_epoch"]-row["started_epoch"] for row in ledger["intervals"])
    result = dict(status="stopped_verified", session_name=cfg["session_name"], endpoint=owned["endpoint"],
                  started_epoch=cfg["allocation_started_epoch"], stopped_epoch=ended, stop_output=output, sessions_output=sessions,
                  prior_allocated_seconds=PRIOR_SECONDS, cumulative_allocated_seconds=total, remaining_seconds=TOTAL_CAP-total)
    save(OPS/"session_stopped.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "allocate", "upload_inputs", "status", "launch", "archive", "finalize", "receipt", "stop", "closeout"))
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--session-name", default="deepmzyme-metal-ring-20260915")
    parser.add_argument("--phase", choices=("plan", "preflight", "execute"), default="execute")
    parser.add_argument("--drive-id")
    parser.add_argument("--final", action="store_true", help="Apply a Drive transfer receipt to the separate terminal archive")
    args = parser.parse_args()
    require(Path(sys.executable).resolve() == Path("/home/mechti/miniconda3/envs/DeepMzyme/bin/python").resolve(), "Discover the SDK, then use the AGENTS.md Conda interpreter")
    if args.action == "prepare":
        result = prepare(args.source_archive, args.session_name)
    elif args.action in ("status", "launch"):
        result = control(args.action, args.phase)
    elif args.action == "receipt":
        result = receipt(args.drive_id, final=args.final)
    elif args.action == "finalize":
        result = archive("finalize")
    else:
        result = globals()[args.action]()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
