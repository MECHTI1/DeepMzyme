"""CPU fixtures for the bounded workstation queue; no controller/network/GPU use."""
from datetime import datetime, timezone
from contextlib import contextmanager, redirect_stdout
import hashlib
import io
import os
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from ec_clean30 import workstation as ws  # noqa: E402


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class LocalTransport:
    """A separate fixture VM tree with real file copying and host readback."""

    def __init__(self, *, fail=False, lost_launch=False, extra_ack=False, failed_downloads=0):
        self.events, self.fail, self.lost_launch = [], fail, lost_launch
        self.extra_ack, self.failed_downloads = extra_ack, failed_downloads
        self.started = 0

    def read_json(self, path, *, optional=False):
        return json.loads(Path(path).read_text()) if Path(path).exists() else None

    def stat_file(self, path):
        p = Path(path)
        return {"sha256": ws._sha(p), "size_bytes": p.stat().st_size} if p.is_file() else None

    def upload(self, local, remote, **kwargs):
        self.events.append(("upload", str(remote)))
        Path(remote).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, remote)

    def download(self, remote, local, **kwargs):
        self.events.append(("download", str(remote)))
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(remote, local)

    def download_files(self, remote_root, local_root, files, **kwargs):
        self.events.append(("payload", tuple(files)))
        if self.failed_downloads:
            self.failed_downloads -= 1
            raise subprocess.CalledProcessError(23, ["synthetic-rsync"])
        for name in files:
            target = local_root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(remote_root) / name, target)

    def transfer(self, root, request, name, paths):
        relative, ack = f"persistence_receipts/{name}.json", f"persistence_receipts/{name}.ack.json"
        manifest = {"schema_version": 1, "persistence_mode": "host_pull", "destination_root": request["durable_root"],
                    "ack_path": ack, "files": [{"path": str(path.relative_to(root)), "size_bytes": path.stat().st_size,
                                               "sha256": ws._sha(path)} for path in paths]}
        write(root / relative, manifest)
        write(root / "execution_state.json", {"pending_transfer": {"manifest_path": relative, "ack_path": ack,
              "manifest_sha256": ws._sha(root / relative)}, "pending_unit": {"run_name": "fixture"}})

    def launch(self, argv, *, cwd, job_dir):
        job = Path(job_dir)
        self.events.append(("launch", job.suffix))
        if job.exists():
            return
        write(job / "command.json", {"argv": argv, "cwd": cwd})
        request = json.loads(Path(argv[-1]).read_text())
        root = Path(request["execution_root"])
        if job.suffix == ".job":
            self.started += 1
            artifact = root / "runs/fixture/model.bin"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(b"synthetic model bytes")
            self.transfer(root, request, "first", [artifact])
        else:
            state = json.loads((root / "execution_state.json").read_text())
            pending = state.get("pending_transfer")
            assert pending and (root / pending["ack_path"]).is_file(), "finalized before verified host ack"
            acknowledgment = json.loads((root / pending["ack_path"]).read_text())
            assert acknowledgment["manifest_sha256"] == pending["manifest_sha256"]
            write(root / "execution_state.json", {"pending_transfer": None})
            complete = root / "runs/fixture/complete.json"
            write(complete, {"status": "passed", "host_transfer": acknowledgment})
            if self.extra_ack and job.suffix == ".finalize":
                self.transfer(root, request, "second", [complete])
        write(job / "result.json", {"returncode": 1 if self.fail and job.suffix == ".job" else 0})
        if self.lost_launch and job.suffix == ".job":
            raise ws.WorkstationError("synthetic lost SSH response")


@pytest.fixture
def cycle(tmp_path):
    root, durable, state = tmp_path / "vm/campaign/runs/fold0/g2", tmp_path / "host/backup", tmp_path / "host/state"
    request = {"execution_root": str(root), "durable_root": str(durable), "python": sys.executable}
    kwargs = {"runner": str(tmp_path / "vm/repo/scripts/runner.py"), "action": "timing-unit",
              "remote_request": str(tmp_path / "vm/requests/fixture.json"),
              "completed_relative": "runs/fixture/complete.json", "timeout": 3}
    return request, durable, state, kwargs


@pytest.mark.parametrize("lost,extra", [(False, False), (True, False), (False, True)])
def test_remote_cycle_copies_reads_acknowledges_then_finalizes(cycle, lost, extra):
    request, durable, state, kwargs = cycle
    transport = LocalTransport(lost_launch=lost, extra_ack=extra)
    host = ws.HostPull(transport, durable, state, max_download_bytes=100000, deadline_unix=time.time() + 600)
    complete = ws.complete_remote_unit(transport, host, request, **kwargs)
    assert complete["status"] == "passed" and transport.started == 1
    events = transport.events
    payload = next(i for i, item in enumerate(events) if item[0] == "payload")
    ack = next(i for i, item in enumerate(events) if item[0] == "upload" and item[1].endswith(".ack.json"))
    finalize = events.index(("launch", ".finalize"))
    assert payload < ack < finalize
    assert (durable / "runs/fixture/model.bin").read_bytes() == b"synthetic model bytes"
    assert (durable / "runs/fixture/complete.json").is_file()
    assert ("launch", ".seal") in events if extra else ("launch", ".seal") not in events


def test_host_pull_mirrors_verified_bytes_before_releasing_ack(cycle, tmp_path):
    request, durable, state, kwargs = cycle
    transport = LocalTransport()
    mirror = tmp_path / "nvme"
    upload = transport.upload

    def checked_upload(local, remote, **options):
        if remote.endswith(".ack.json"):
            assert (mirror / "runs/fixture/model.bin").read_bytes() == b"synthetic model bytes"
            assert (mirror / "persistence_receipts/first.json").is_file()
            assert (mirror / "persistence_receipts/first.ack.json").read_bytes() == Path(local).read_bytes()
        upload(local, remote, **options)

    transport.upload = checked_upload
    host = ws.HostPull(transport, durable, state, max_download_bytes=100000,
                       deadline_unix=time.time() + 600, mirror_root=mirror)
    ws.complete_remote_unit(transport, host, request, **kwargs)
    assert (mirror / "runs/fixture/complete.json").read_bytes() == (durable / "runs/fixture/complete.json").read_bytes()


def test_mirror_failure_withholds_remote_ack_and_next_launch(cycle, tmp_path, monkeypatch):
    request, durable, state, kwargs = cycle
    transport = LocalTransport()
    host = ws.HostPull(transport, durable, state, max_download_bytes=100000,
                       deadline_unix=time.time() + 600, mirror_root=tmp_path / "nvme")

    def failed(_):
        raise OSError("synthetic mirror disk failure")

    monkeypatch.setattr(host, "_mirror", failed)
    with pytest.raises(OSError, match="mirror disk"):
        ws.complete_remote_unit(transport, host, request, **kwargs)
    assert not any(kind == "upload" and value.endswith(".ack.json") for kind, value in transport.events)
    assert ("launch", ".finalize") not in transport.events


def test_runpod_closeout_preserves_bookkeeping_without_mistaking_it_for_manifests(cycle, tmp_path, monkeypatch):
    request, durable, state, _ = cycle
    root = Path(request["execution_root"])
    payloads = {"runs/model.bin": b"checkpoint",
                "persistence_receipts/unit.json": json.dumps({"schema_version": 1, "persistence_mode": "host_pull",
                    "ack_path": "persistence_receipts/unit.ack.json", "files": []}).encode(),
                "persistence_receipts/unit.ack.json": b'{"manifest_sha256":"fixture"}',
                "persistence_receipts/unit.state.json": b'{"status":"persisted"}',
                "persistence_receipts/unit.events.jsonl": b'{"event":"first"}\n{"event":"second"}\n'}
    entries = []
    for name, data in payloads.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        entries.append({"path": name, "size_bytes": len(data), "sha256": ws._sha(target)})
    transport = LocalTransport()

    def remote(argv, **kwargs):
        # The real work-estimate and inventory programs run here; their control file is read by download().
        monkeypatch.setattr(sys, "argv", ["-c", *argv[3:]])
        output = io.StringIO()
        with redirect_stdout(output):
            exec(compile(argv[2], "<remote-inventory>", "exec"), {})
        return output.getvalue()

    transport.remote = remote
    config = {"remote_campaign": str(root.parents[2]), "bootstrap_python": sys.executable,
              "durable_root": str(durable), "mirror_root": str(tmp_path / "nvme"), "state_root": str(state)}
    controller = type("Controller", (), {"config_sha": "a" * 64})()
    receipt = ws._preserve_runpod_outputs(config, transport, controller,
        {"pod_id": "pod", "session_id": "session", "deadline_unix": time.time() + 600})
    proof = json.loads(receipt.read_text())
    assert proof["host_pull_manifests"] == ["persistence_receipts/unit.json"]
    assert {row["path"] for row in proof["files"]} == set(payloads) | {"closeout/remote_inventory.json"}
    assert proof["files"][:-1] == sorted(entries, key=lambda row: row["path"])
    for name, data in payloads.items():
        assert (durable / name).read_bytes() == (tmp_path / "nvme" / name).read_bytes() == data


def test_runpod_transport_propagates_fixed_precision_to_remote_launcher(tmp_path, monkeypatch):
    calls = []
    controller = type("Controller", (), {"heartbeat": lambda self: None})()
    config = {"state_root": str(tmp_path), "bootstrap_python": sys.executable,
              "runpod": {"ssh_alias": "fixture"}}
    transport = ws.RunpodTransport(config, controller)

    def execute(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "ready", "")

    monkeypatch.setattr(subprocess, "run", execute)
    assert transport.remote(["python", "runner.py"], cwd="/workspace/repo") == "ready"
    assert calls[0][-1].startswith("cd /workspace/repo && ")
    assert calls[0][-1].endswith("env NVIDIA_TF32_OVERRIDE=0 PYTHONNOUSERSITE=1 python runner.py")


@pytest.mark.parametrize("returncode", [0, 2])
def test_detached_launch_script_runs_once_and_keeps_terminal_code(tmp_path, monkeypatch, returncode):
    transport = ws.ControllerTransport(tmp_path, "fixture", bootstrap_python=sys.executable)

    def local(argv, **kwargs):
        return subprocess.run(argv, check=True, capture_output=True, text=True, timeout=5).stdout

    monkeypatch.setattr(transport, "remote", local)
    counter, job = tmp_path / "counter", tmp_path / "job"
    argv = [sys.executable, "-c", "import pathlib,sys; p=pathlib.Path(sys.argv[1]); "
            "p.write_text((p.read_text() if p.exists() else '')+'x'); sys.exit(int(sys.argv[2]))", str(counter), str(returncode)]
    transport.launch(argv, cwd=str(tmp_path), job_dir=str(job))
    transport.launch(argv, cwd=str(tmp_path), job_dir=str(job))
    until = time.monotonic() + 5
    while not (job / "result.json").exists() and time.monotonic() < until:
        time.sleep(.01)
    assert json.loads((job / "result.json").read_text())["returncode"] == returncode
    assert counter.read_text() == "x"


def test_failed_remote_unit_preserves_and_acknowledges_without_next_launch(cycle):
    request, durable, state, kwargs = cycle
    transport = LocalTransport(fail=True)
    host = ws.HostPull(transport, durable, state, max_download_bytes=100000, deadline_unix=time.time() + 600)
    with pytest.raises(ws.WorkstationError, match="failed.*preserved"):
        ws.complete_remote_unit(transport, host, request, **kwargs)
    assert (durable / "persistence_receipts/first.ack.json").is_file()
    assert ("launch", ".finalize") not in transport.events


def test_transfer_budget_counts_failed_attempt_before_retry(cycle):
    request, durable, state, kwargs = cycle
    transport = LocalTransport(failed_downloads=1)
    host = ws.HostPull(transport, durable, state, max_download_bytes=30, deadline_unix=time.time() + 600)
    with pytest.raises(ws.WorkstationError, match="payload ceiling"):
        ws.complete_remote_unit(transport, host, request, **kwargs)
    assert json.loads((state / "transfer_budget.json").read_text())["reserved_payload_bytes"] == len(b"synthetic model bytes")
    assert not (durable / "persistence_receipts/first.ack.json").exists()


@pytest.mark.parametrize("problem", ["tampered", "traversal", "foreign_destination"])
def test_host_pull_refuses_invalid_transfer_before_ack(cycle, problem):
    request, durable, state, kwargs = cycle
    transport = LocalTransport()
    root = Path(request["execution_root"])
    file = root / "file.bin"
    file.parent.mkdir(parents=True)
    file.write_bytes(b"data")
    transport.transfer(root, request, "first", [file])
    manifest_path = root / "persistence_receipts/first.json"
    manifest = json.loads(manifest_path.read_text())
    if problem == "tampered":
        file.write_bytes(b"wrong bytes")
    elif problem == "traversal":
        manifest["files"][0]["path"] = "../escaped.bin"
    else:
        manifest["destination_root"] = str(durable.parent / "other")
    write(manifest_path, manifest)
    s = json.loads((root / "execution_state.json").read_text())
    s["pending_transfer"]["manifest_sha256"] = ws._sha(manifest_path)
    write(root / "execution_state.json", s)
    host = ws.HostPull(transport, durable, state, max_download_bytes=10000, deadline_unix=time.time() + 600)
    with pytest.raises(RuntimeError):
        host.sync_pending(str(root))
    assert not (durable / "persistence_receipts/first.ack.json").exists()


def stamp(value=None):
    return datetime.fromtimestamp(value or time.time(), timezone.utc).isoformat()


@pytest.fixture
def config(tmp_path, monkeypatch):
    secondary, controller = tmp_path / "data1", tmp_path / "controller"
    secondary.mkdir()
    identity = {"project": "fixture-project", "zone": "us-central1-a", "vm_name": "fixture-l4", "instance_id": "123"}
    controller.mkdir()
    (controller / "config.env").write_text("PROJECT_ID=fixture-project\nREGION=us-central1\nZONE=us-central1-a\nVM_NAME=fixture-l4\nSELECTED_INSTANCE_ID=123\nMACHINE_TYPE=g2-standard-8\nPROVISIONING_MODEL=STANDARD\nSSH_HOST_ALIAS=fixture-host\nREPO_URL=https://github.com/fixture/repository.git\n")
    for name in ("status", "report", "start", "stop", "pycharm-info"):
        path = controller / "bin" / f"vm-{name}"
        path.parent.mkdir(exist_ok=True)
        path.write_text("#!/bin/sh\nexit 99\n")
        path.chmod(0o755)
    write(controller / "state/price_cache.json", {"verified_at": stamp(), "gross_usd_per_hour": .879172,
          "machine_type": "g2-standard-8", "region": "us-central1"})
    write(controller / "state/current_session.json", {"active": False})
    remote = str(tmp_path / "vm")
    source = secondary / "inputs"
    source.mkdir()
    stage = []
    files = ["view.tar", "clean/bin/python", "clean/harness.py", "clean/extract.py", "weights.pt", "weights-contact-regression.pt",
             "hf_home/weights.pth", "timing/timing_inputs.json", "timing/timing.fasta", "timing/sequence_map.tsv"]
    for name in files:
        local = source / name
        local.parent.mkdir(parents=True, exist_ok=True)
        if name == "view.tar":
            payload = b'{"fold": 0}\n'
            header = {"format": ws.phase3.TRANSFER_FORMAT, "campaign_id": ws.phase3.CAMPAIGN_ID, "fold": 0,
                      "smoke": False, "files": [{"path": "view_manifest.json", "size_bytes": len(payload),
                                                 "sha256": hashlib.sha256(payload).hexdigest(), "mode": 0o444}]}
            with tarfile.open(local, "w") as archive:
                for member_name, data in ((ws.phase3.TRANSFER_MANIFEST, json.dumps(header).encode()),
                                          ("view_manifest.json", payload)):
                    member = tarfile.TarInfo(member_name)
                    member.size, member.mode = len(data), 0o444
                    archive.addfile(member, io.BytesIO(data))
        else:
            local.write_bytes(name.encode())
        stage.append({"local": str(local), "remote": f"{remote}/inputs/{name}", "sha256": ws._sha(local), "kind": "file"})
    cfg = {"controller_root": str(controller), "expected_identity": identity, "ssh_alias": "fixture-host",
           "secondary_drive": str(secondary), "durable_root": str(secondary / "backup"), "state_root": str(secondary / "state"),
           "remote_repo": f"{remote}/repo", "remote_campaign": f"{remote}/{ws.phase3.CAMPAIGN_ID}", "remote_python": f"{remote}/python",
           "bootstrap_python": "/usr/bin/python3", "execution_manifest": str(ROOT / "fixture_manifest.json"), "stage": stage,
           "view_archive": f"{remote}/inputs/view.tar", "view_archive_sha256": stage[0]["sha256"], "timing_inputs": f"{remote}/inputs/timing",
           "bounds": {name: {"forecast_seconds": 60, "unit_timeout_seconds": 60} for _, name in ws.QUEUE},
           "profiles": {name: {"memory_limit_bytes": gib * ws.GIB, "cap_prefix": ["systemd-run", "--user", "--scope", "-p",
                        f"CPUQuota={cpu}%", "-p", f"MemoryMax={gib}G", "-p", "MemorySwapMax=0"]}
                        for name, gib, cpu in (("fit", 8, 200), ("esm1b", 24, 400))},
           "lane_limits": {"cpu_percent": 600, "ram_bytes": 24 * ws.GIB, "vram_bytes": 21 * ws.GIB},
           "embedding": {"esmc": {"hf_home": f"{remote}/inputs/hf_home", "esmc_weights": f"{remote}/inputs/hf_home/weights.pth"},
                         "esm1b": {"clean_python": f"{remote}/inputs/clean/bin/python", "harness": f"{remote}/inputs/clean/harness.py",
                                   "extract_script": f"{remote}/inputs/clean/extract.py", "weights": f"{remote}/inputs/weights.pt",
                                   "weights_sha256": stage[4]["sha256"], "contact_weights_sha256": stage[5]["sha256"], "threads": 4}},
           "pricing": {"snapshot_usd_per_hour": .021, "egress_usd_per_gib": .15, "verified_at": stamp()}}
    monkeypatch.setattr(Path, "is_mount", lambda path: path == secondary)
    monkeypatch.setattr(shutil, "disk_usage", lambda path: shutil._ntuple_diskusage(100 * ws.GIB, 0, 100 * ws.GIB))
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: "git@github.com:fixture/repository.git\n")
    monkeypatch.setattr(ws.phase3, "verify_execution_source", lambda *a, **kw: {"published": kw.get("require_published", True),
        "execution_manifest_sha256": "a" * 64, "git_head": "b" * 40})
    path = secondary / "config.json"
    write(path, cfg)
    return cfg, path


@pytest.mark.parametrize("entrypoint", ["preflight", "_prepared_preflight"])
@pytest.mark.parametrize("provider", ["gcp", "runpod"])
def test_preflight_refuses_noncanonical_campaign_before_hashing_or_cloud(config, monkeypatch, entrypoint, provider):
    cfg, path = config
    cfg["provider"] = provider
    cfg["remote_campaign"] = "/workspace/campaign"
    if provider == "runpod":
        cfg["artifact_cache"] = {"fixture": True}
        cfg["stage"][0]["conda_unpack"] = True
        cfg["runtime_environments"] = {key: {"archive": cfg["stage"][0]["remote"],
            "archive_sha256": cfg["stage"][0]["sha256"], "packages": {"torch": "fixture"},
            "python_version": "fixture", "torch_cuda": "fixture"} for key in ("fit", "esm1b")}
    monkeypatch.setattr(ws, "_sha", lambda *_: pytest.fail("campaign location must be checked before hashing"))
    monkeypatch.setattr(ws.phase3, "verify_execution_source", lambda *a, **kw: pytest.fail("source read before location check"))
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("cloud call before location check"))
    with pytest.raises(ws.WorkstationError, match="remote_campaign must end with"):
        getattr(ws, entrypoint)(cfg, path)
    assert not (Path(cfg["state_root"]) / "driver_state.json").exists()


def test_preview_is_local_only_and_names_exact_launch(config, monkeypatch):
    cfg, path = config
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("preview made an external call"))
    preview = ws.preflight(cfg, path)
    assert ws.phase3.views.view_location_problem(Path(cfg["remote_campaign"]) / "views/fold0",
                                                smoke=False, fold=0) is None
    assert preview["status"] == "draft_unpublished"
    assert len(preview["queue"]) == 12 and preview["cost"]["conservative_max_usd"] < 3
    assert preview["launch_command"].endswith(f"--config {path} --run")
    archive = Path(cfg["stage"][0]["local"])
    with tarfile.open(archive) as view:
        unpacked = sum(member.size for member in view if member.isreg())
    assert unpacked > 0
    assert preview["staging_peak_bytes"] == sum(Path(item["local"]).stat().st_size for item in cfg["stage"]) + unpacked
    assert not (Path(cfg["state_root"]) / "driver_state.json").exists()


@pytest.mark.parametrize("problem", ["invalid_tar", "unsafe_member", "foreign_header"])
def test_view_capacity_refuses_invalid_pinned_archive(config, problem):
    cfg, path = config
    item = cfg["stage"][0]
    archive_path = Path(item["local"])
    if problem == "invalid_tar":
        archive_path.write_bytes(b"not a transfer archive")
    else:
        with tarfile.open(archive_path) as archive:
            members = [(member, archive.extractfile(member).read()) for member in archive.getmembers()]
        with tarfile.open(archive_path, "w") as archive:
            for member, data in members:
                if problem == "unsafe_member" and member.name == "view_manifest.json":
                    member.name = "../outside.json"
                elif problem == "foreign_header" and member.name == ws.phase3.TRANSFER_MANIFEST:
                    header = json.loads(data)
                    header["fold"] = 1
                    data = json.dumps(header).encode()
                    member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
    item["sha256"] = cfg["view_archive_sha256"] = ws._sha(archive_path)
    with pytest.raises(ws.WorkstationError, match="Fold-0"):
        ws.preflight(cfg, path)


@pytest.mark.parametrize("problem", ["changed_file", "missing_stage", "cap", "deadline", "queue", "cost", "stale_price", "identity", "disk"])
def test_preflight_refuses_missing_inputs_or_widened_bounds(config, monkeypatch, problem):
    cfg, path = config
    if problem == "changed_file":
        Path(cfg["stage"][0]["local"]).write_text("changed")
    elif problem == "missing_stage":
        cfg["stage"].pop()
    elif problem == "cap":
        cfg["profiles"]["fit"]["cap_prefix"] = ["env"]
    elif problem == "deadline":
        cfg["bounds"]["F1"]["unit_timeout_seconds"] = 61
    elif problem == "queue":
        cfg["bounds"].pop("F4")
    elif problem == "cost":
        cfg["pricing"]["egress_usd_per_gib"] = 1
    elif problem == "stale_price":
        cfg["pricing"]["verified_at"] = stamp(time.time() - 90000)
    elif problem == "identity":
        cfg["expected_identity"]["instance_id"] = "other"
    else:
        monkeypatch.setattr(shutil, "disk_usage", lambda p: shutil._ntuple_diskusage(ws.GIB, 0, ws.GIB))
    with pytest.raises(ws.WorkstationError):
        ws.preflight(cfg, path)


@pytest.fixture
def weight_stage(config, monkeypatch):
    cfg, path = config
    pins = {}
    for url, original in ws.OFFICIAL_ESM1B_WEIGHTS.items():
        index = 4 if original["field"] == "weights" else 5
        item = cfg["stage"][index]
        item["remote_url"] = url
        pins[url] = {**original, "sha256": item["sha256"], "size_bytes": Path(item["local"]).stat().st_size}
    monkeypatch.setattr(ws, "OFFICIAL_ESM1B_WEIGHTS", pins)
    return cfg, path, cfg["stage"][4]


class WeightTransport(LocalTransport):
    def __init__(self, item, outcome="verified"):
        super().__init__()
        self.item, self.outcome = item, outcome
        self.readbacks = []
        self.deadline_unix, self.reserve_seconds = time.time() + 1200, 900

    def stat_file(self, path):
        self.readbacks.append(path)
        return super().stat_file(path)

    def fetch_official_weight(self, url, partial):
        self.events.append(("fetch", url))
        if self.outcome == "ambiguous":
            raise ws.WorkstationError("lost SSH download response")
        target = Path(partial)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(self.item["local"]).read_bytes() if self.outcome == "verified" else b"bad download")
        if self.outcome == "expired":
            self.deadline_unix = time.time() + 899
        return {"terminal": True, "status": "verified" if self.outcome in {"verified", "wrong_digest"} else "failed"}

    def upload(self, local, remote, **kwargs):
        ws.ControllerTransport._bounded(self, kwargs["timeout"])
        super().upload(local, remote, **kwargs)
        if self.outcome == "bad_fallback":
            Path(remote).write_bytes(b"bad upload")

    def remote(self, argv, **kwargs):
        assert argv[:2] == ["mv", "--"]
        self.events.append(("promote", argv[-1]))
        Path(argv[-2]).replace(argv[-1])
        return ""


@pytest.mark.parametrize("index", [4, 5])
def test_official_weight_direct_success_avoids_upload(weight_stage, index):
    cfg, _, _ = weight_stage
    item = cfg["stage"][index]
    transport = WeightTransport(item)
    ws._stage_file(cfg, transport, item)
    assert [kind for kind, _ in transport.events] == ["fetch", "promote"]
    assert transport.readbacks.count(item["remote"] + ".partial") == 1
    assert ws._sha(Path(item["remote"])) == item["sha256"]


@pytest.mark.parametrize("outcome", ["http_failure", "wrong_digest"])
def test_terminal_public_download_failure_uses_verified_upload_fallback(weight_stage, outcome):
    cfg, _, item = weight_stage
    transport = WeightTransport(item, outcome)
    ws._stage_file(cfg, transport, item)
    assert [kind for kind, _ in transport.events] == ["fetch", "upload", "promote"]
    assert ws._sha(Path(item["remote"])) == item["sha256"]


@pytest.mark.parametrize("outcome", ["ambiguous", "expired", "bad_fallback"])
def test_unconfirmed_or_invalid_weight_is_never_promoted(weight_stage, outcome):
    cfg, _, item = weight_stage
    transport = WeightTransport(item, outcome)
    with pytest.raises(ws.WorkstationError):
        ws._stage_file(cfg, transport, item)
    assert not Path(item["remote"]).exists()
    assert not any(kind == "promote" for kind, _ in transport.events)
    if outcome in {"ambiguous", "expired"}:
        assert not any(kind == "upload" for kind, _ in transport.events)


def test_verified_remote_weight_is_reused_without_fetch_or_upload(weight_stage):
    cfg, _, item = weight_stage
    target = Path(item["remote"])
    target.parent.mkdir(parents=True)
    shutil.copyfile(item["local"], target)
    transport = WeightTransport(item)
    ws._stage_file(cfg, transport, item)
    assert transport.events == []


@pytest.mark.parametrize("problem", ["tampered_local", "unapproved_url", "malformed_url", "wrong_destination", "wrong_pin"])
def test_public_url_does_not_weaken_local_or_exact_source_pins(weight_stage, problem):
    cfg, path, item = weight_stage
    if problem == "tampered_local":
        Path(item["local"]).write_bytes(b"changed local bytes")
    elif problem == "unapproved_url":
        item["remote_url"] = "https://example.invalid/weights.pt"
    elif problem == "malformed_url":
        item["remote_url"] = [item["remote_url"]]
    elif problem == "wrong_destination":
        item["remote"] += ".different"
    else:
        cfg["embedding"]["esm1b"]["weights_sha256"] = "0" * 64
    with pytest.raises(ws.WorkstationError):
        ws.preflight(cfg, path)


@pytest.mark.parametrize("outcome", ["verified", "http_failure", "wrong_digest"])
def test_remote_download_has_total_alarm_and_terminal_writer_result(weight_stage, tmp_path, monkeypatch, outcome):
    import contextlib
    import signal
    import urllib.error
    import urllib.request

    _, _, item = weight_stage
    transport = ws.ControllerTransport(tmp_path, "fixture", bootstrap_python=sys.executable)
    transport.deadline_unix, transport.reserve_seconds = time.time() + 925, 900
    alarms, calls, responses = [], [], []
    monkeypatch.setattr(signal, "signal", lambda *args: None)
    monkeypatch.setattr(signal, "setitimer", lambda timer, duration: alarms.append(duration))

    class Response(io.BytesIO):
        def geturl(self):
            return item["remote_url"]

    def open_url(url, *, timeout, context=None):
        assert url == item["remote_url"] and timeout < 25 and context is None
        if outcome == "http_failure":
            raise urllib.error.URLError("synthetic HTTP failure")
        response = Response(Path(item["local"]).read_bytes() if outcome == "verified" else b"bad digest")
        responses.append(response)
        return response

    monkeypatch.setattr(urllib.request, "urlopen", open_url)

    def local_program(argv, **kwargs):
        calls.append((argv, kwargs))
        monkeypatch.setattr(sys, "argv", ["fixture", *argv[3:]])
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            exec(compile(argv[2], "<bounded-weight-download>", "exec"), {})
        return captured.getvalue()

    monkeypatch.setattr(transport, "remote", local_program)
    result = transport.fetch_official_weight(item["remote_url"], str(tmp_path / "weight.partial"))
    assert result["terminal"] is True and result["status"] == ("verified" if outcome == "verified" else "failed")
    assert 0 < alarms[0] < calls[0][1]["timeout"] <= 25 and alarms[-1] == 0
    assert all(response.closed for response in responses)


class LifecycleTransport:
    def __init__(self, cfg, *, fail_start=False, initial_state="TERMINATED"):
        self.cfg, self.controller_root, self.calls = cfg, Path(cfg["controller_root"]), []
        self.state, self.fail_start = initial_state, fail_start

    def controller(self, name, *args, **kwargs):
        self.calls.append((name, args))
        identity = self.cfg["expected_identity"]
        if name == "status":
            return f"VM: {identity['vm_name']} project={identity['project']} zone={identity['zone']}\nState: {self.state}\nGoogle-side auto-STOP: VERIFIED\n"
        if name == "start":
            if "--dry-run" in args:
                return "maxRunDuration=7020s"
            if self.fail_start:
                raise ws.WorkstationError("start response lost")
            self.state = "RUNNING"
            now = time.time()
            write(self.controller_root / "state/current_session.json", {**identity, "active": True, "session_id": "session-1",
                  "session_start": stamp(now), "termination_ts": stamp(now + 7020), "max_run_duration_s": 7020,
                  "gross_usd_per_hour": .879172})
        elif name == "stop":
            self.state = "TERMINATED"
            write(self.controller_root / "state/current_session.json", {"active": False})
        return "ok"

    def remote(self, *args, **kwargs):
        return str(100 * ws.GIB)

    def read_json(self, path, **kwargs):
        return {"wall_seconds": 10}


@pytest.fixture
def lifecycle(config, monkeypatch):
    cfg, path = config
    monkeypatch.setattr(ws, "_stage_runtime", lambda *a: None)
    monkeypatch.setattr(ws, "_runtime_preflight", lambda *a: ("/fixture/manifest.json", "/fixture/probe.json"))
    measured = []

    def complete(transport, host, request, **kwargs):
        measured.append((kwargs["action"], request.get("recipe", request.get("kind", "lane_batch"))))
        return {"status": "passed", "execution_manifest_sha256": "a" * 64}

    monkeypatch.setattr(ws, "complete_remote_unit", complete)
    return cfg, path, measured


def test_driver_runs_exact_fixed_queue_once_then_stops(lifecycle):
    cfg, path, measured = lifecycle
    transport = LifecycleTransport(cfg)
    state = ws.run_g2(cfg, path, transport=transport)
    assert measured == list(ws.QUEUE)
    assert state["provider_stop_verified"] is True and state["closed"] is True
    assert transport.calls.count(("start", ("--confirm", "--hours", "2"))) == 1
    with pytest.raises(ws.WorkstationError, match="no automatic restart"):
        ws.run_g2(cfg, path, transport=transport)


@pytest.mark.parametrize("failure", [None, "caps", "unit", "start"])
def test_runpod_uses_shared_queue_and_always_closes_owned_attempt(lifecycle, monkeypatch, failure):
    cfg, path, measured = lifecycle
    cfg.update(provider="runpod", hardware_profile="runpod_rtx4090", gcp_controller_root=cfg["controller_root"],
               gcp_stopped_identity=cfg["expected_identity"], mirror_root=str(path.parent / "nvme"))
    write(path, cfg)
    transport = LifecycleTransport(cfg)
    now = time.time()

    class Controller:
        def __init__(self):
            self.starts, self.stops = 0, 0

        def start(self):
            self.starts += 1
            if failure == "start":
                raise ws.WorkstationError("synthetic start failure")
            return self.session()

        def session(self, previous=None):
            assert previous in (None, "runpod-session")
            return {"session_id": "runpod-session", "allocation_started_unix": now,
                    "deadline_unix": now + 6960, "max_total_seconds": 6960}

        def heartbeat(self):
            pass

        @contextmanager
        def maintain_heartbeat(self, *, interval=30):
            assert interval == 30
            yield type("Heartbeat", (), {"check": lambda self: None})()

        def admit_workload(self):
            pass

        def write_ssh_config(self):
            return path.parent / "ssh_config"

        def stop(self):
            self.stops += 1
            return {"desiredStatus": "EXITED", "runtimeStatus": "stopped"}

    controller = Controller()
    monkeypatch.setattr(ws, "preflight", lambda *a, **kw: {"source": {"execution_manifest_sha256": "a" * 64},
                                                         "staging_peak_bytes": 1})
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 0, transport.controller("status"), ""))
    monkeypatch.setattr(ws, "_stage_checkout", lambda *a: None)

    def caps(*args, **kwargs):
        if failure == "caps":
            raise ws.WorkstationError("synthetic missing cgroup delegation")
        return "probe.json"

    monkeypatch.setattr(ws, "_container_probe", caps)
    monkeypatch.setattr(ws, "_preserve_runpod_outputs", lambda *a: None)
    write(Path(cfg["state_root"]) / "esm1b_model_probe.json", {"fixture": True})
    write(Path(cfg["state_root"]) / "main_model_probe.json", {"fixture": True})
    if failure == "unit":
        monkeypatch.setattr(ws, "complete_remote_unit", lambda *a, **kw: (_ for _ in ()).throw(ws.WorkstationError("synthetic unit failure")))
    if failure:
        with pytest.raises(ws.WorkstationError):
            ws._run_runpod_g2(cfg, path, transport=transport, controller=controller)
    else:
        result = ws._run_runpod_g2(cfg, path, transport=transport, controller=controller)
        assert measured == list(ws.QUEUE) and result["status"] == "measurements_complete"
    assert controller.starts == controller.stops == 1
    state = json.loads((Path(cfg["state_root"]) / "driver_state.json").read_text())
    assert state["closed"] and state["provider_stop_verified"]
    if failure in {"caps", "start"}:
        assert measured == []


@pytest.mark.parametrize("problem", ["unit", "start_response"])
def test_owned_allocation_is_stopped_on_failure(lifecycle, monkeypatch, problem):
    cfg, path, measured = lifecycle
    transport = LifecycleTransport(cfg, fail_start=problem == "start_response")
    if problem == "unit":
        monkeypatch.setattr(ws, "complete_remote_unit", lambda *a, **kw: (_ for _ in ()).throw(ws.WorkstationError("unit failed")))
    with pytest.raises(ws.WorkstationError):
        ws.run_g2(cfg, path, transport=transport)
    assert ("stop", ("--now",)) in transport.calls
    state = json.loads((Path(cfg["state_root"]) / "driver_state.json").read_text())
    assert state["closed"] and state["provider_stop_verified"]


def test_foreign_running_vm_is_neither_started_nor_stopped(lifecycle):
    cfg, path, _ = lifecycle
    transport = LifecycleTransport(cfg, initial_state="RUNNING")
    with pytest.raises(ws.WorkstationError, match="positively stopped"):
        ws.run_g2(cfg, path, transport=transport)
    assert not any(name in {"start", "stop"} for name, _ in transport.calls)


def test_recovery_attaches_same_live_session_without_starting_again(lifecycle):
    cfg, path, measured = lifecycle
    transport = LifecycleTransport(cfg)
    transport.controller("start", "--confirm", "--hours", "2")
    transport.calls.clear()
    write(Path(cfg["state_root"]) / "driver_state.json", {"config_sha256": ws._sha(path), "completed": [],
          "start_attempted": True, "session_id": "session-1"})
    result = ws.run_g2(cfg, path, transport=transport)
    assert result["closed"] and measured == list(ws.QUEUE)
    assert not any(name == "start" for name, _ in transport.calls)


def test_recovery_refuses_another_allocation_without_stopping_it(lifecycle):
    cfg, path, measured = lifecycle
    transport = LifecycleTransport(cfg)
    transport.controller("start", "--confirm", "--hours", "2")
    transport.calls.clear()
    write(Path(cfg["state_root"]) / "driver_state.json", {"config_sha256": ws._sha(path), "completed": [],
          "start_attempted": True, "session_id": "older-session"})
    with pytest.raises(ws.WorkstationError, match="Another allocation"):
        ws.run_g2(cfg, path, transport=transport)
    assert not measured and not any(name in {"start", "stop"} for name, _ in transport.calls)


def test_ambiguous_start_cannot_stop_an_unbound_active_session(lifecycle):
    cfg, path, measured = lifecycle

    class RacedTransport(LifecycleTransport):
        def controller(self, name, *args, **kwargs):
            result = super().controller(name, *args, **kwargs)
            if name == "start" and "--confirm" in args:
                # Another allocation appears during a lost/refused start response; even
                # its matching instance and fresh timestamp do not prove this call owns it.
                current = self.controller_root / "state/current_session.json"
                candidate = json.loads(current.read_text())
                candidate["session_id"] = "foreign-session"
                write(current, candidate)
                raise ws.WorkstationError("ambiguous start response")
            return result

    transport = RacedTransport(cfg)
    with pytest.raises(ws.WorkstationError, match="unbound active session"):
        ws.run_g2(cfg, path, transport=transport)
    state = json.loads((Path(cfg["state_root"]) / "driver_state.json").read_text())
    assert state["start_attempted"] and not state.get("session_id")
    assert not measured and not any(name == "stop" for name, _ in transport.calls)


def test_successful_start_binds_session_before_later_validation_failure(lifecycle, monkeypatch):
    cfg, path, measured = lifecycle
    transport = LifecycleTransport(cfg)
    monkeypatch.setattr(ws, "_session", lambda *a, **kw: (_ for _ in ()).throw(ws.WorkstationError("runtime validation failed")))
    with pytest.raises(ws.WorkstationError, match="runtime validation failed"):
        ws.run_g2(cfg, path, transport=transport)
    state = json.loads((Path(cfg["state_root"]) / "driver_state.json").read_text())
    assert state["session_id"] == "session-1" and state["provider_stop_verified"]
    assert not measured and ("stop", ("--now",)) in transport.calls


@pytest.mark.parametrize("text", ["State: UNKNOWN", "VM: fixture-l4 project=fixture-project zone=us-central1-a\nState: RUNNING\n",
    "VM: fixture-l4 project=fixture-project zone=us-central1-a\nState: TERMINATED\nduplicate scan failed"])
def test_unknown_provider_state_fails_closed(config, text):
    cfg, _ = config
    with pytest.raises(ws.WorkstationError):
        ws._status(text, cfg["expected_identity"])


def allow_fallback(cfg, path):
    cfg["fallback"] = {
        "allowed_zones": ["us-central1-b", "us-central1-c"], "storage_reserve_usd": .30,
        "storage_window_seconds": 14400,
        "storage_cost_evidence": {"verified_at": stamp(), "disk_usd_per_hour": 15 / 730,
                                  "snapshot_usd_per_hour": 7.5 / 730, "recycle_minimum_usd": 7.5 / 730},
    }
    executable = Path(cfg["controller_root"]) / "bin/vm-fallback"
    executable.write_text("#!/bin/sh\nexit 99\n")
    executable.chmod(0o755)
    write(path, cfg)


def allow_cross_region(cfg, path):
    allow_fallback(cfg, path)
    regions = {"us-central1": ("10.42.0.0/24", "A88A-5A60-E821:1,D18E-3563-415F:8,F0A4-E3D3-33BD:32", "6AE1-525F-8B80", ".853624"),
               "us-west1": ("10.44.0.0/24", "A88A-5A60-E821:1,D18E-3563-415F:8,F0A4-E3D3-33BD:32", "6AE1-525F-8B80", ".853624"),
               "us-east4": ("10.45.0.0/24", "1D4C-3419-0297:1,CF07-A027-C390:8,A15F-B56F-97FE:32", "7D52-6D58-14FF", ".850829")}
    budget = {"schema_version": 1, "scope": "cross-region-v1", "allowed_zones": list(ws.CROSS_REGION_ZONES),
              "gross_ceiling_usd": 3.5, "host_egress_reserve_usd": .6, "retained_snapshot_usd_per_hour": .021,
              "max_storage_seconds": 14400, "transfer_usd_per_gib": .02, "transfer_bytes_per_restore": 150 * ws.GIB,
              "transfer_source_url": "https://cloud.google.com/compute/disks-image-pricing", "verified_at": stamp(),
              "snapshot_usd_per_hour": 7.5 / 730, "recycle_usd_per_hour": 7.5 / 730,
              "region_profiles": {region: {"REGION": region, "SUBNET": "deepmzyme-subnet", "SUBNET_RANGE": row[0],
                  "PRICE_COMPUTE_SKUS": row[1], "PRICE_DISK_CAPACITY_SKU_ID": row[2],
                  "PRICE_EXPECTED_COMPUTE_USD_PER_HOUR": row[3]} for region, row in regions.items()},
              "region_prices": {region: {"verified_at": stamp(), "gross_usd_per_hour": .879172,
                  "disk_usd_per_hour": .023} for region in regions}}
    budget_path = path.with_name("recovery_budget.json")
    write(budget_path, budget)
    cfg["fallback"].update(scope="cross-region-v1", allowed_zones=list(ws.CROSS_REGION_ZONES), storage_reserve_usd=.61,
                           recovery_budget={"path": str(budget_path), "sha256": ws._sha(budget_path)})
    cfg["fallback"]["storage_cost_evidence"]["disk_usd_per_hour"] = .023
    controller_path = Path(cfg["controller_root"]) / "config.env"
    controller = ws._controller_config(controller_path.parent)
    controller.update(budget["region_profiles"]["us-central1"])
    controller_path.write_text("".join(f"{key}={value}\n" for key, value in controller.items()))
    write(path, cfg)
    return budget, budget_path


@pytest.mark.parametrize("zone", ws.CROSS_REGION_ZONES)
def test_cross_region_identity_requires_matching_subnet_and_pricing(config, zone):
    cfg, path = config
    budget, _ = allow_cross_region(cfg, path)
    region = zone.rsplit("-", 1)[0]
    cfg["expected_identity"]["zone"] = zone
    controller = ws._controller_config(Path(cfg["controller_root"]))
    controller.update(budget["region_profiles"][region], ZONE=zone)
    ws._identity(cfg, controller)
    for key in ("REGION", "SUBNET", "PRICE_COMPUTE_SKUS", "PRICE_DISK_CAPACITY_SKU_ID"):
        changed = {**controller, key: "wrong-region-value"}
        with pytest.raises(ws.WorkstationError):
            ws._identity(cfg, changed)


def test_cross_region_profile_accepts_equivalent_price_rounding(config):
    cfg, path = config
    allow_cross_region(cfg, path)
    controller = ws._controller_config(Path(cfg["controller_root"]))
    controller["PRICE_EXPECTED_COMPUTE_USD_PER_HOUR"] = "0.853624271"
    ws._identity(cfg, controller)
    controller["PRICE_EXPECTED_COMPUTE_USD_PER_HOUR"] = "0.9"
    with pytest.raises(ws.WorkstationError):
        ws._identity(cfg, controller)


def test_full_cross_region_pass_refuses_three_fifty_before_any_cloud_call(config, monkeypatch):
    cfg, path = config
    allow_cross_region(cfg, path)
    monkeypatch.setattr(ws.ControllerTransport, "controller", lambda *a, **kw: pytest.fail("budget refusal must precede cloud calls"))
    with pytest.raises(ws.WorkstationError, match=r"require \$18.*exceeding the \$3.5 gross ceiling"):
        ws.run_g2(cfg, path)
    assert not (Path(cfg["state_root"]) / "driver_state.json").exists()


@pytest.mark.parametrize("ceiling", [19.0, 19.01])
def test_cross_region_budget_accepts_approved_nineteen_only(config, monkeypatch, ceiling):
    cfg, path = config
    budget, budget_path = allow_cross_region(cfg, path)
    budget["gross_ceiling_usd"] = ceiling
    write(budget_path, budget)
    cfg["fallback"]["recovery_budget"]["sha256"] = ws._sha(budget_path)
    write(path, cfg)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("preview made an external call"))
    if ceiling > 19:
        with pytest.raises(ws.WorkstationError, match=r"authorized \$19 ceiling"):
            ws.preflight(cfg, path, published=True)
    else:
        preview = ws.preflight(cfg, path, published=True)
        assert preview["status"] == "ready" and len(preview["queue"]) == 12
        assert preview["cost"]["gross_ceiling_usd"] == 19
        assert preview["cost"]["conservative_max_usd"] < 19
        assert preview["cost"]["recovery_transfer_reserve_usd"] == 15
        assert preview["max_provider_seconds"] == 117 * 60


@pytest.mark.parametrize("problem", ["checksum", "scope", "order", "ceiling", "transfer_rate", "transfer_bytes", "stale", "profile", "missing_region"])
def test_cross_region_budget_cannot_omit_authority_or_charge(config, problem):
    cfg, path = config
    budget, budget_path = allow_cross_region(cfg, path)
    if problem == "checksum":
        budget["gross_ceiling_usd"] = 3.4
    elif problem == "scope":
        cfg["fallback"].pop("scope")
    elif problem == "order":
        cfg["fallback"]["allowed_zones"].reverse()
    elif problem == "ceiling":
        budget["gross_ceiling_usd"] = 20
    elif problem == "transfer_rate":
        budget["transfer_usd_per_gib"] = 0
    elif problem == "transfer_bytes":
        budget["transfer_bytes_per_restore"] = 20 * ws.GIB
    elif problem == "stale":
        budget["region_prices"]["us-east4"]["verified_at"] = stamp(time.time() - 90000)
    elif problem == "profile":
        budget["region_profiles"]["us-east4"].pop("PRICE_COMPUTE_SKUS")
    else:
        budget["region_prices"].pop("us-west1")
    write(budget_path, budget)
    if problem != "checksum":
        cfg["fallback"]["recovery_budget"]["sha256"] = ws._sha(budget_path)
    with pytest.raises(ws.WorkstationError):
        ws._cross_region_budget(cfg)


@pytest.mark.parametrize("zone", ["us-central1-b", "us-central1-c"])
def test_replacement_identity_requires_exact_authorized_controller_endpoint(config, zone):
    cfg, path = config
    allow_fallback(cfg, path)
    cfg["expected_identity"] = {**cfg["expected_identity"], "zone": zone, "instance_id": "456"}
    controller = ws._controller_config(Path(cfg["controller_root"]))
    controller.update(ZONE=zone, SELECTED_INSTANCE_ID="456")
    ws._identity(cfg, controller)
    controller["SELECTED_INSTANCE_ID"] = "789"
    with pytest.raises(ws.WorkstationError):
        ws._identity(cfg, controller)


@pytest.mark.parametrize("problem", ["unapproved", "cross_region", "region_mismatch", "hardware", "project", "name"])
def test_fallback_does_not_expand_endpoint_authority(config, problem):
    cfg, path = config
    allow_fallback(cfg, path)
    controller = ws._controller_config(Path(cfg["controller_root"]))
    zone = "us-central1-b"
    cfg["expected_identity"] = {**cfg["expected_identity"], "zone": zone}
    controller["ZONE"] = zone
    if problem == "unapproved":
        cfg.pop("fallback")
    elif problem == "cross_region":
        controller.update(REGION="us-east4", ZONE="us-east4-a")
        cfg["expected_identity"]["zone"] = "us-east4-a"
    elif problem == "region_mismatch":
        controller["REGION"] = "us-east4"
    elif problem == "hardware":
        controller["MACHINE_TYPE"] = "g2-standard-12"
    elif problem == "project":
        controller["PROJECT_ID"] = "another-project"
    else:
        controller["VM_NAME"] = "another-vm"
    with pytest.raises(ws.WorkstationError):
        ws._identity(cfg, controller)


def test_fallback_cost_reserves_storage_without_widening_three_dollars(config):
    cfg, path = config
    baseline = ws._cost(cfg)
    allow_fallback(cfg, path)
    recovered = ws._cost(cfg)
    assert recovered["conservative_max_usd"] == pytest.approx(baseline["conservative_max_usd"] + .30)
    assert recovered["conservative_max_usd"] < 3
    cfg["fallback"]["storage_reserve_usd"] = .40
    with pytest.raises(ws.WorkstationError):
        ws._cost(cfg)


@pytest.mark.parametrize("problem", ["missing_reserve", "insufficient_reserve", "stale_price", "unapproved_zone"])
def test_recovery_storage_and_zone_evidence_cannot_be_omitted(config, problem):
    cfg, path = config
    allow_fallback(cfg, path)
    if problem == "missing_reserve":
        cfg["fallback"].pop("storage_reserve_usd")
    elif problem == "insufficient_reserve":
        cfg["fallback"]["storage_reserve_usd"] = .05
    elif problem == "stale_price":
        cfg["fallback"]["storage_cost_evidence"]["verified_at"] = stamp(time.time() - 90000)
    else:
        cfg["fallback"]["allowed_zones"] = ["us-central1-b", "us-east4-a"]
    with pytest.raises((ws.WorkstationError, KeyError)):
        ws._cost(cfg)


class FallbackTransport(LifecycleTransport):
    """The controller's selected endpoint changes only after a completed recovery."""

    def __init__(self, cfg, *, zone="us-central1-b", start_error="ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS",
                 preview_zones="us-central1-b, us-central1-c", replacement_problem=None, preview_problem=None):
        super().__init__(cfg)
        self.source = dict(cfg["expected_identity"])
        self.zone, self.start_error = zone, start_error
        self.preview_zones, self.replacement_problem = preview_zones, replacement_problem
        self.preview_problem = preview_problem
        self.replacement_session = None

    def controller(self, name, *args, **kwargs):
        if name == "status":
            self.calls.append((name, args))
            live = ws._controller_config(self.controller_root)
            return (f"VM: {live['VM_NAME']} project={live['PROJECT_ID']} zone={live['ZONE']}\n"
                    f"State: {self.state}\nGoogle-side auto-STOP: VERIFIED\n")
        if name == "start" and "--confirm" in args and self.start_error:
            self.calls.append((name, args))
            raise ws.WorkstationError(f"Controller start failed: {self.start_error}")
        if name != "fallback":
            return super().controller(name, *args, **kwargs)
        self.calls.append((name, args))
        if "--dry-run" in args:
            rates = {"verified_at": stamp(), "disk_usd_per_hour": 15 / 730,
                     "snapshot_usd_per_hour": 7.5 / 730, "recycle_usd_per_hour": 7.5 / 730,
                     "snapshot_minimum_seconds": 3600, "recycle_minimum_seconds": 3600}
            if self.preview_problem == "storage_price":
                rates["disk_usd_per_hour"] *= 2
            elif self.preview_problem == "minimum_charge":
                rates["recycle_minimum_seconds"] = 7200
            elif self.preview_problem == "stale_price":
                rates["verified_at"] = stamp(time.time() - 90000)
            return ("Google-side hard STOP after:  117 min (maxRunDuration=7020s, instanceTerminationAction=STOP)\n"
                    f"Candidate offerings (capacity not guaranteed): {self.preview_zones}\n"
                    f"Verified storage rates: {json.dumps(rates)}\n")
        assert "--confirm" in args
        if self.replacement_problem == "all_stockout":
            raise ws.WorkstationError("ZONE_RESOURCE_POOL_EXHAUSTED: bounded recovery exhausted")
        identity = {**self.source, "zone": self.zone, "instance_id": "456"}
        now = time.time()
        current = {**identity, "active": True, "session_id": "replacement-session", "session_start": stamp(now),
                   "termination_ts": stamp(now + 7020), "max_run_duration_s": 7020,
                   "gross_usd_per_hour": .879172}
        receipt = {"schema_version": 1, "project": identity["project"], "pass_id": "f01234567890",
                   "source": self.source, "source_disk": {"sizeGb": "150"}, "replacement": identity,
                   "phase": "selected", "requested_hours": 2, "limit": {"effective_seconds": 7020},
                   "allowed_zones": ["us-central1-b", "us-central1-c"],
                   "candidates": ["us-central1-b", "us-central1-c"],
                   "attempts": [{"zone": self.zone, "status": "create_requested"}],
                   "session_id": current["session_id"], "session_start": current["session_start"],
                   "termination_ts": current["termination_ts"], "updated_at": stamp(),
                   "price": {"disk_usd_per_hour": 15 / 730, "gross_usd_per_hour": .879172},
                   "snapshot_price": {"usd_per_gib_month": .05}, "recycle_price": {"usd_per_gib_month": .05}}
        if self.replacement_problem == "source":
            receipt["source"] = {**self.source, "instance_id": "wrong-source"}
        elif self.replacement_problem == "session_id":
            receipt["session_id"] = "unrelated-session"
        elif self.replacement_problem == "instance_id":
            current["instance_id"] = "unrelated-instance"
        elif self.replacement_problem == "duration":
            current.update(termination_ts=stamp(time.time() + 8000), max_run_duration_s=8000)
        elif self.replacement_problem == "rate":
            current["gross_usd_per_hour"] = 1.1
        elif self.replacement_problem == "stale_session":
            current["session_start"] = receipt["session_start"] = stamp(time.time() - 100)
        elif self.replacement_problem == "limit":
            receipt["limit"]["effective_seconds"] = 8000
        elif self.replacement_problem == "storage_price":
            receipt["price"]["disk_usd_per_hour"] *= 2
        live_path = self.controller_root / "config.env"
        live = ws._controller_config(self.controller_root)
        live.update(ZONE=identity["zone"], SELECTED_INSTANCE_ID=identity["instance_id"])
        live_path.write_text("".join(f"{key}={value}\n" for key, value in live.items()))
        write(self.controller_root / "state/fallback.json", receipt)
        write(self.controller_root / "state/current_session.json", current)
        self.state, self.replacement_session = "RUNNING", current
        return "Recovery selected fixture-l4; source and snapshot preserved."


@pytest.fixture
def fallback_lifecycle(lifecycle):
    cfg, path, measured = lifecycle
    allow_fallback(cfg, path)
    return cfg, path, measured


@pytest.mark.parametrize("zone", ["us-central1-b", "us-central1-c"])
def test_confirmed_stockout_runs_one_bounded_pass_then_same_measurement_queue(fallback_lifecycle, zone):
    cfg, path, measured = fallback_lifecycle
    transport = FallbackTransport(cfg, zone=zone)
    state = ws.run_g2(cfg, path, transport=transport)
    assert measured == list(ws.QUEUE)
    assert state["provider_stop_verified"] and state["closed"]
    assert state["effective_identity"]["zone"] == zone
    assert state["fallback_pass_id"] == "f01234567890"
    assert state["session_id"] == "replacement-session"
    assert cfg["expected_identity"]["zone"] == "us-central1-a"
    assert len([call for call in transport.calls if call[0] == "start" and "--confirm" in call[1]]) == 1
    assert len([call for call in transport.calls if call[0] == "fallback" and "--dry-run" in call[1]]) == 1
    assert len([call for call in transport.calls if call[0] == "fallback" and "--confirm" in call[1]]) == 1
    assert not any(call[0] == "fallback" and "--finalize" in call[1] for call in transport.calls)


@pytest.mark.parametrize("error", ["permission denied", "QUOTA_EXCEEDED", "start response lost"])
def test_noncapacity_or_ambiguous_start_failure_never_falls_back(fallback_lifecycle, error):
    cfg, path, measured = fallback_lifecycle
    transport = FallbackTransport(cfg, start_error=error)
    with pytest.raises(ws.WorkstationError):
        ws.run_g2(cfg, path, transport=transport)
    assert not measured and not any(call[0] == "fallback" for call in transport.calls)


@pytest.mark.parametrize("provider_state", ["RUNNING", "TERMINATED", "UNKNOWN"])
def test_stockout_text_cannot_override_an_active_foreign_session(fallback_lifecycle, provider_state):
    cfg, path, measured = fallback_lifecycle

    class ForeignSessionTransport(FallbackTransport):
        def controller(self, name, *args, **kwargs):
            try:
                return super().controller(name, *args, **kwargs)
            except ws.WorkstationError:
                if name == "start" and "--confirm" in args:
                    self.state = provider_state
                    write(self.controller_root / "state/current_session.json",
                          {**self.source, "active": True, "session_id": "foreign-session"})
                raise

    transport = ForeignSessionTransport(cfg)
    with pytest.raises(ws.WorkstationError):
        ws.run_g2(cfg, path, transport=transport)
    assert not measured
    assert not any(call[0] in {"fallback", "stop"} for call in transport.calls)


@pytest.mark.parametrize("zones", ["us-central1-b, us-central1-f", "us-east4-a", ""])
def test_fallback_preview_refuses_candidates_outside_authorized_pass(fallback_lifecycle, zones):
    cfg, path, measured = fallback_lifecycle
    transport = FallbackTransport(cfg, preview_zones=zones)
    with pytest.raises(ws.WorkstationError):
        ws.run_g2(cfg, path, transport=transport)
    assert not measured
    assert not any(call[0] == "fallback" and "--confirm" in call[1] for call in transport.calls)


@pytest.mark.parametrize("problem", ["storage_price", "minimum_charge", "stale_price"])
def test_fallback_preview_rechecks_live_storage_prices_before_spending(fallback_lifecycle, problem):
    cfg, path, measured = fallback_lifecycle
    transport = FallbackTransport(cfg, preview_problem=problem)
    with pytest.raises(ws.WorkstationError):
        ws.run_g2(cfg, path, transport=transport)
    assert not measured
    assert not any(call[0] == "fallback" and "--confirm" in call[1] for call in transport.calls)


@pytest.mark.parametrize("problem", ["source", "session_id", "instance_id", "duration", "rate", "stale_session", "limit", "storage_price"])
def test_replacement_requires_valid_session_before_any_measurement(fallback_lifecycle, problem):
    cfg, path, measured = fallback_lifecycle
    transport = FallbackTransport(cfg, replacement_problem=problem)
    with pytest.raises(ws.WorkstationError):
        ws.run_g2(cfg, path, transport=transport)
    assert not measured
    assert len([call for call in transport.calls if call[0] == "fallback" and "--confirm" in call[1]]) == 1
    if problem in {"source", "session_id", "instance_id", "stale_session"}:
        assert not any(call[0] == "stop" for call in transport.calls), "Unbound replacement must not be blindly stopped"


def test_exhausted_fallback_is_not_repeated(fallback_lifecycle):
    cfg, path, measured = fallback_lifecycle
    transport = FallbackTransport(cfg, replacement_problem="all_stockout")
    with pytest.raises(ws.WorkstationError):
        ws.run_g2(cfg, path, transport=transport)
    assert not measured
    assert len([call for call in transport.calls if call[0] == "fallback" and "--confirm" in call[1]]) == 1
    with pytest.raises(ws.WorkstationError):
        ws.run_g2(cfg, path, transport=transport)
    assert len([call for call in transport.calls if call[0] == "fallback" and "--confirm" in call[1]]) == 1


def test_recovery_reattaches_only_its_live_replacement_without_another_start(fallback_lifecycle):
    cfg, path, measured = fallback_lifecycle
    transport = FallbackTransport(cfg)
    started = time.time()
    transport.controller("fallback", "--confirm", "--hours", "2")
    identity = {key: transport.replacement_session[key] for key in cfg["expected_identity"]}
    write(Path(cfg["state_root"]) / "driver_state.json", {
        "config_sha256": ws._sha(path), "completed": [], "start_attempted": True,
        "session_id": "replacement-session", "effective_identity": identity,
        "fallback_attempted": True, "fallback_started_unix": started, "fallback_pass_id": "f01234567890",
        "fallback_source_identity": dict(cfg["expected_identity"]),
    })
    transport.calls.clear()
    result = ws.run_g2(cfg, path, transport=transport)
    assert measured == list(ws.QUEUE) and result["closed"] and result["provider_stop_verified"]
    assert not any(call[0] in {"start", "fallback"} for call in transport.calls)


def test_lost_fallback_reply_uses_selected_ownership_only_to_close_safely(fallback_lifecycle):
    cfg, path, measured = fallback_lifecycle

    class LostReplyTransport(FallbackTransport):
        def controller(self, name, *args, **kwargs):
            response = super().controller(name, *args, **kwargs)
            if name == "fallback" and "--confirm" in args:
                raise ws.WorkstationError("fallback reply lost")
            return response

    transport = LostReplyTransport(cfg)
    with pytest.raises(ws.WorkstationError, match="reply lost"):
        ws.run_g2(cfg, path, transport=transport)
    state = json.loads((Path(cfg["state_root"]) / "driver_state.json").read_text())
    assert not measured and state["closed"] and state["provider_stop_verified"]
    assert state["session_id"] == "replacement-session"
    assert len([call for call in transport.calls if call[0] == "fallback" and "--confirm" in call[1]]) == 1
    with pytest.raises(ws.WorkstationError):
        ws.run_g2(cfg, path, transport=transport)
    assert len([call for call in transport.calls if call[0] == "fallback" and "--confirm" in call[1]]) == 1


@pytest.fixture
def tls_weight_server(tmp_path):
    """A local HTTPS server with one self-signed certificate; expired or current."""
    import http.server, ssl, threading
    payload = b"pinned public weight bytes " * 4096
    (tmp_path / "www").mkdir()
    (tmp_path / "www" / "weight.pt").write_bytes(payload)
    servers = []

    def start(kind):
        key, cert = tmp_path / f"{kind}.key", tmp_path / f"{kind}.pem"
        validity = (["-not_before", "20200101000000Z", "-not_after", "20200102000000Z"] if kind == "expired" else ["-days", "2"])
        made = subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", str(key), "-out", str(cert),
                               "-subj", "/CN=localhost", "-addext", "subjectAltName=DNS:localhost", *validity],
                              capture_output=True, text=True)
        if made.returncode:
            pytest.skip("This OpenSSL cannot issue a certificate with explicit validity dates.")
        handler = lambda *a, **kw: http.server.SimpleHTTPRequestHandler(*a, directory=str(tmp_path / "www"), **kw)
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert, key)
        server.socket = context.wrap_socket(server.socket, server_side=True)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"https://localhost:{server.server_address[1]}/weight.pt", cert

    yield start, payload
    for server in servers:
        server.shutdown()
        server.server_close()


def _official_download(url, target, payload, *, trust=None, digest=None):
    env = {key: value for key, value in os.environ.items() if key not in {"SSL_CERT_FILE", "SSL_CERT_DIR"}}
    if trust is not None:
        env["SSL_CERT_FILE"] = str(trust)
    done = subprocess.run([sys.executable, "-c", ws.OFFICIAL_WEIGHT_PROGRAM, url, str(target),
                           digest or hashlib.sha256(payload).hexdigest(), str(len(payload)), "20"],
                          capture_output=True, text=True, timeout=60, env=env)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_official_weight_accepts_only_pinned_bytes_through_an_expired_certificate(tls_weight_server, tmp_path):
    start, payload = tls_weight_server
    url, cert = start("expired")
    target = tmp_path / "out" / "weight.partial"
    result = _official_download(url, target, payload, trust=cert)
    assert result == {"terminal": True, "status": "verified", "sha256": hashlib.sha256(payload).hexdigest(),
                      "size_bytes": len(payload), "tls": "expired_certificate_content_pinned"}
    assert target.read_bytes() == payload
    target.unlink()
    refused = _official_download(url, target, payload, trust=cert, digest="0" * 64)
    assert refused["status"] == "failed" and "checksum or size differs" in refused["error"] and not target.exists()


def test_official_weight_never_relaxes_tls_for_any_other_certificate_failure(tls_weight_server, tmp_path):
    start, payload = tls_weight_server
    target = tmp_path / "out" / "weight.partial"
    url, cert = start("current")
    trusted = _official_download(url, target, payload, trust=cert)
    assert trusted["status"] == "verified" and trusted["tls"] == "verified"
    target.unlink()
    untrusted = _official_download(url, target, payload)
    assert untrusted["status"] == "failed" and "CERTIFICATE_VERIFY_FAILED" in untrusted["error"] and not target.exists()
    expired_url, _ = start("expired")
    unknown_expired = _official_download(expired_url, target, payload)
    assert unknown_expired["status"] == "failed" and not target.exists()
