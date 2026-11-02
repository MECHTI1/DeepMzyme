"""CPU fixtures for the bounded workstation queue; no controller/network/GPU use."""
from datetime import datetime, timezone
import hashlib
import io
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
    (controller / "config.env").write_text("PROJECT_ID=fixture-project\nZONE=us-central1-a\nVM_NAME=fixture-l4\nSELECTED_INSTANCE_ID=123\nMACHINE_TYPE=g2-standard-8\nSSH_HOST_ALIAS=fixture-host\nREPO_URL=https://github.com/fixture/repository.git\n")
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
           "remote_repo": f"{remote}/repo", "remote_campaign": f"{remote}/campaign", "remote_python": f"{remote}/python",
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


def test_preview_is_local_only_and_names_exact_launch(config, monkeypatch):
    cfg, path = config
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("preview made an external call"))
    preview = ws.preflight(cfg, path)
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

    def open_url(url, *, timeout):
        assert url == item["remote_url"] and timeout < 25
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
