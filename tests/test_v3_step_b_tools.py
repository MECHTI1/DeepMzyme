"""Workstation step-B tools: per-lane host pull (real worker manifests) and the launcher (no VM contact)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import pmm_v3_host_pull as host_pull  # noqa: E402
import pmm_v3_probes as probes  # noqa: E402
import pmm_v3_step_b as launcher  # noqa: E402
from benchmarking import pmm_execution as execution  # noqa: E402


# ---------------------------------------------------------------------------
# Host pull against genuine worker manifests (CampaignExecution in host_pull mode)
# ---------------------------------------------------------------------------

def finished_lane(tmp_path, lane=0, content='{"fit_status":"completed"}\n'):
    """A lane whose worker persisted one unit in host_pull mode and exited (awaiting the host)."""
    remote = tmp_path / "vm" / "lanes" / f"lane{lane}"
    durable = tmp_path / "host"
    worker = execution.CampaignExecution(remote, durable_root=durable / f"lane{lane}", session_id="gcp",
                                         persistence_mode="host_pull",
                                         policy=execution.ExecutionPolicy(4_000_000_000, 4000))
    with worker as owner:
        owner.admit("unit", 10)
        path = owner.root / "runs" / "unit" / "selected_checkpoint.json"
        path.parent.mkdir(parents=True)
        path.write_text(content)
        owner.record_result("unit", "completed", 10)
        assert owner.persist([path.parent])["status"] == "awaiting_host_ack"
    return remote, durable


def reopen(remote, durable, lane=0):
    return execution.CampaignExecution(remote, durable_root=durable / f"lane{lane}", session_id="gcp",
                                       persistence_mode="host_pull",
                                       policy=execution.ExecutionPolicy(4_000_000_000, 4000))


def verifier():
    return host_pull.load_verifier(ROOT)


def test_pull_acknowledges_and_the_worker_accepts(tmp_path):
    remote, durable = finished_lane(tmp_path)
    with pytest.raises(execution.ExecutionBlocked, match="Host must"):
        with reopen(remote, durable):
            pass
    result = host_pull.pull_lane(host_pull.LocalTransport(), str(remote), durable / "lane0", verifier())
    assert result["status"] == "acknowledged" and result["file_count"] >= 1
    assert (durable / "lane0" / "runs" / "unit" / "selected_checkpoint.json").is_file()
    with reopen(remote, durable) as successor:
        assert successor.state["pending_unit"]["persisted"] and successor.state["pending_transfer"] is None


def test_repeated_pull_never_restamps_and_reuploads_a_lost_ack_unchanged(tmp_path):
    remote, durable = finished_lane(tmp_path)
    transport, verify = host_pull.LocalTransport(), verifier()
    first = host_pull.pull_lane(transport, str(remote), durable / "lane0", verify)
    again = host_pull.pull_lane(transport, str(remote), durable / "lane0", verify)
    assert again["status"] == "already_acknowledged" and again["verified_unix"] == first["verified_unix"]
    transfer = json.loads((remote / "execution_state.json").read_text())["pending_transfer"]
    remote_ack = remote / transfer["ack_path"]
    original = remote_ack.read_bytes()
    remote_ack.unlink()
    restored = host_pull.pull_lane(transport, str(remote), durable / "lane0", verify)
    assert restored["how"] == "re-uploaded unchanged" and remote_ack.read_bytes() == original


def test_corrupted_download_is_never_acknowledged(tmp_path):
    remote, durable = finished_lane(tmp_path)

    class Corrupting(host_pull.LocalTransport):
        def fetch(self, remote_root, members, local_root):
            super().fetch(remote_root, members, local_root)
            target = next(Path(local_root) / m for m in members if m.endswith("selected_checkpoint.json"))
            target.write_text("corrupt")

    with pytest.raises(RuntimeError, match="checksum") as raised:  # the verifier's own PersistenceError class
        host_pull.pull_lane(Corrupting(), str(remote), durable / "lane0", verifier())
    assert type(raised.value).__name__ == "PersistenceError"
    transfer = json.loads((remote / "execution_state.json").read_text())["pending_transfer"]
    assert not (remote / transfer["ack_path"]).exists()


@pytest.mark.parametrize("case", ["destination", "running", "manifest", "foreign_ack", "symlink"])
def test_pull_refusals(tmp_path, case):
    remote, durable = finished_lane(tmp_path)
    state_path = remote / "execution_state.json"
    state = json.loads(state_path.read_text())
    transfer = state["pending_transfer"]
    local = durable / "lane0"
    if case == "destination":
        local = tmp_path / "elsewhere" / "lane0"
    elif case == "running":
        state["pending_unit"]["status"] = "running"
        state_path.write_text(json.dumps(state))
    elif case == "manifest":
        (remote / transfer["manifest_path"]).write_text("{}")
    elif case == "foreign_ack":
        (remote / transfer["ack_path"]).write_text("{}")
    elif case == "symlink":
        real = tmp_path / "real_host"
        real.mkdir()
        durable.rename(tmp_path / "unused") if durable.exists() else None
        os.symlink(real, durable)
    with pytest.raises(host_pull.HostPullError):
        host_pull.pull_lane(host_pull.LocalTransport(), str(remote), local, verifier())
    assert case == "foreign_ack" or not (remote / transfer["ack_path"]).exists()


def test_pull_over_lanes_reports_each_lane(tmp_path):
    remote0, durable = finished_lane(tmp_path, lane=0)
    finished_lane(tmp_path, lane=1, content='{"fit_status":"completed","x":1}\n')
    result = host_pull.pull(host_pull.LocalTransport(), str(tmp_path / "vm"), durable, [0, 1, 2])
    assert result["lanes"]["lane0"]["status"] == result["lanes"]["lane1"]["status"] == "acknowledged"
    assert result["lanes"]["lane2"]["status"] == "no_lane_state" and result["ok"]


def test_host_pull_tool_and_launcher_never_import_torch():
    code = ("import sys; sys.path.insert(0, %r); import pmm_v3_host_pull as h, pmm_v3_step_b; "
            "h.load_verifier(h.ROOT); print('torch' in sys.modules)" % str(ROOT))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "False"


# ---------------------------------------------------------------------------
# Launcher: controller session, SSH config, commands (no VM contact)
# ---------------------------------------------------------------------------

def controller(tmp_path, *, active=True, instance="111", start="2026-10-06T08:00:00Z",
               stop="2026-10-06T11:57:00Z", alias_instance="111"):
    root = tmp_path / "ctl"
    (root / "state").mkdir(parents=True)
    (root / "config.env").write_text("PROJECT_ID=p\nZONE=z\nVM_NAME=v\nSELECTED_INSTANCE_ID=111\n# c\n")
    session = {"active": active, "project": "p", "zone": "z", "vm_name": "v", "instance_id": instance,
               "session_id": "s-1", "session_start": start, "termination_ts": stop, "max_run_duration_s": 14220}
    (root / "state" / "current_session.json").write_text(json.dumps(session))
    (root / "state" / "ssh_config").write_text(f"Host deepmzyme-vm\n    HostKeyAlias deepmzyme-vm-p-z-{alias_instance}\n")
    return root


NOW = launcher._stamp("2026-10-06T09:00:00Z")


def test_session_is_read_as_unix_allocation_fields(tmp_path):
    session = launcher.read_session(controller(tmp_path), now=NOW)
    assert session["session_id"] == "s-1" and session["allocation_started"] == launcher._stamp("2026-10-06T08:00:00Z")
    assert session["deadline"] - session["allocation_started"] == 14220
    assert session["remaining_seconds"] == pytest.approx(2 * 3600 + 57 * 60)


@pytest.mark.parametrize("kwargs, message", [({"active": False}, "No active"), ({"instance": "999"}, "instance_id"),
                                             ({"stop": "2026-10-06T08:30:00Z"}, "hard stop")])
def test_session_refusals(tmp_path, kwargs, message):
    with pytest.raises(launcher.LauncherError, match=message):
        launcher.read_session(controller(tmp_path, **kwargs), now=NOW)


def test_stale_ssh_config_is_refused(tmp_path):
    assert launcher.check_ssh_config(controller(tmp_path)).name == "ssh_config"
    with pytest.raises(launcher.LauncherError, match="vm-setup --stages ssh"):
        launcher.check_ssh_config(controller(tmp_path / "x", alias_instance="6465074116022266445"))


def session_fields():
    return {"session_id": "s-1", "allocation_started": 1000.0, "deadline": 15220.0, "max_seconds": 14220.0,
            "remaining_seconds": 10000.0}


def test_commands_follow_the_probe_manifest_and_flag_conventions():
    short = launcher.run_command("w2-fp32-b", session_fields(), fit_seconds=900)
    assert short[short.index("--probe") + 1] == "w2-fp32-b" and short[short.index("--lane") + 1] == "1"
    assert short[short.index("--amp") + 1] == "off" and short[short.index("--epochs") + 1] == str(probes.SHORT_EPOCHS)
    assert short[short.index("--durable-root") + 1] == str(launcher.DURABLE)
    assert short[short.index("--persistence-mode") + 1] == "host_pull"
    full = launcher.run_command("full-amp", session_fields(), fit_seconds=2100)
    assert "--epochs" not in full and full[full.index("--amp") + 1] == "on"
    with pytest.raises(ValueError, match="not a step-B probe"):
        launcher.run_command("w2-fp32-z", session_fields(), fit_seconds=900)


def test_launch_script_starts_every_member_together_detached():
    script = launcher.launch_script("w3-fp32", probes.BATCHES[("fp32", 3)], session_fields(), fit_seconds=900)
    assert script.count("setsid nohup") == 1 and "echo launched" in script
    for tag in probes.BATCHES[("fp32", 3)]:
        assert f"--probe {tag}" in script and f"{tag}.rc" in script
    assert script.index("setsid") > script.index("mv \"$f\"")  # earlier attempt outputs are kept, then launch


def test_steps_cover_the_manifest_and_retries():
    table = launcher.steps()
    assert table["w2-fp32"] == probes.BATCHES[("fp32", 2)]
    assert table["w2-fp32-retry"] == tuple(t + "-retry" for t in probes.BATCHES[("fp32", 2)])
    assert "w1-fp32-r1-retry" not in table and table["full-fp32"] == ("full-fp32",)
    assert launcher.default_fit_seconds("w1-fp32-r1") > launcher.default_fit_seconds("w1-fp32-r2")


class Recorder:
    """A fake Remote: records scripts and answers the few queries the launcher makes."""

    def __init__(self, *, sampler_age=3, alive="yes", codes=None, report=None, lanes=None, earlier=None):
        self.scripts, self.sampler_age, self.alive = [], sampler_age, alive
        self.codes, self.report = codes or {}, report or {}
        self.lanes, self.earlier = lanes or {}, earlier or {}

    def run(self, script, *, check=True, timeout=None):
        import re

        self.scripts.append(script)
        if "--sample-host" in script and "age=" in script:
            out = f"{self.alive} {self.sampler_age}\n"
        elif "execution_state.json" in script and "fcntl" in script:
            out = json.dumps({f"lane{k}": self.lanes.get(f"lane{k}", {"exists": False, "lock_free": True})
                              for k in script.split()[-3:] if k.isdigit()})
        elif ".rc 2>/dev/null" in script:
            tags = re.findall(r"echo (\S+) \$\(cat", script)
            source = self.earlier if "[ 0 = 1 ]" in script else self.codes  # launched_only: earlier launches
            out = "".join(f"{tag} {source[tag] or '-'}\n" for tag in tags if tag in source) if source is self.earlier \
                else "".join(f"{tag} {self.codes.get(tag, '-')}\n" for tag in tags)
        elif "speed_report.json" in script and "cat" in script:
            out = json.dumps(self.report)
        else:
            out = "launched\n"
        return subprocess.CompletedProcess([], 0, stdout=out, stderr="")


def test_lane_blockers_name_every_reason_a_member_would_be_refused():
    ready = {"lane0": {"exists": True, "lock_free": True, "pending_unit": {"run_name": "u", "status": "completed",
                                                                            "persisted": False},
                       "pending_transfer": True, "ack_uploaded": True, "active_child_alive": False},
             "lane1": {"exists": False, "lock_free": True}}
    assert launcher.lane_blockers(ready) == []
    blocked = {"lane0": {**ready["lane0"], "ack_uploaded": False},
               "lane1": {"exists": True, "lock_free": False, "pending_unit": {"run_name": "v", "status": "running",
                                                                               "persisted": False},
                         "pending_transfer": False, "active_child_alive": True}}
    problems = "\n".join(launcher.lane_blockers(blocked))
    for text in ("awaits its host pull", "lane lock", "training child", "running or was interrupted"):
        assert text in problems


def test_step_refuses_blocked_lanes_and_unfinished_earlier_launches(monkeypatch):
    monkeypatch.setattr(launcher, "cmd_pull", lambda remote, lanes: {"lanes": lanes, "ok": True})
    lanes = {"lane1": {"exists": True, "lock_free": True, "pending_unit": None, "pending_transfer": True,
                       "ack_uploaded": False, "active_child_alive": False}}
    with pytest.raises(launcher.LauncherError, match="awaits its host pull"):
        launcher.cmd_step(Recorder(lanes=lanes), session_fields(), "w2-fp32", fit_seconds=900)
    with pytest.raises(launcher.LauncherError, match="no exit code yet"):
        launcher.cmd_step(Recorder(earlier={"w2-fp32-a": None}), session_fields(), "w2-fp32", fit_seconds=900)


def test_a_failed_pull_fails_the_command(monkeypatch):
    monkeypatch.setattr(launcher.host_pull, "pull", lambda *a, **k: {"ok": False, "lanes": {
        "lane0": {"status": "error", "error": "rsync failed"}}})

    class Bare:
        transport = None

    with pytest.raises(launcher.LauncherError, match="rsync failed"):
        launcher.cmd_pull(Bare(), [0])


def test_step_refuses_without_time_sampler_or_retry_permission(monkeypatch):
    monkeypatch.setattr(launcher, "cmd_pull", lambda remote, lanes: {"lanes": lanes})
    late = {**session_fields(), "remaining_seconds": 1000.0}
    with pytest.raises(launcher.LauncherError, match="hard stop"):
        launcher.cmd_step(Recorder(), late, "w2-fp32", fit_seconds=900)
    with pytest.raises(launcher.LauncherError, match="sampler"):
        launcher.cmd_step(Recorder(sampler_age=60), session_fields(), "w2-fp32", fit_seconds=900)
    with pytest.raises(launcher.LauncherError, match="does not require a retry"):
        launcher.cmd_step(Recorder(report={"retry_required": []}), session_fields(), "w2-fp32-retry", fit_seconds=900)
    tags = probes.BATCHES[("fp32", 2)]
    remote = Recorder(codes={t: "0" for t in tags})
    result = launcher.cmd_step(remote, {**session_fields(), "deadline": 4e9}, "w2-fp32", fit_seconds=900,
                               sleep=lambda s: None)
    assert result["exit_codes"] == {t: "0" for t in tags} and result["pull"] == {"lanes": [0, 1]}
    assert any("setsid nohup" in s for s in remote.scripts)
    retry = Recorder(report={"retry_required": ["fp32x2"]}, codes={t + "-retry": "0" for t in tags})
    assert launcher.cmd_step(retry, {**session_fields(), "deadline": 4e9}, "w2-fp32-retry", fit_seconds=900,
                             sleep=lambda s: None)["exit_codes"] == {t + "-retry": "0" for t in tags}


def test_evidence_copy_is_checked_on_both_ends(tmp_path):
    vm = tmp_path / "vm_v3"
    (vm / "lanes" / "lane0").mkdir(parents=True)
    (vm / "step_b").mkdir()
    (vm / "campaign_manifest.json").write_text("{}")
    (vm / "lanes" / "lane0" / "execution_events.jsonl").write_text("{}\n")
    (vm / "step_b" / "host.jsonl").write_text("{}\n")
    code = tmp_path / "vm_code"
    code.mkdir()
    (code / "v3_bundle_apply_receipt.json").write_text("{}")

    class LocalRemote:
        transport = host_pull.LocalTransport()

        def run(self, script, *, check=True, timeout=None):
            script = script.replace(launcher.REMOTE_V3, str(vm)).replace(launcher.REMOTE_CODE, str(code))
            script = script.replace(launcher.REMOTE_PY, sys.executable)
            result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
            assert result.returncode == 0, result.stderr
            return result

    import pmm_v3_step_b as module
    original = (module.REMOTE_V3, module.REMOTE_CODE)
    try:
        module.REMOTE_V3, module.REMOTE_CODE = str(vm), str(code)
        out = launcher.cmd_evidence(LocalRemote(), destination=tmp_path / "evidence")
    finally:
        module.REMOTE_V3, module.REMOTE_CODE = original
    manifest = json.loads((tmp_path / "evidence" / "evidence_manifest.json").read_text())
    assert out["files"] == len(manifest["files"]) and not manifest["mismatched"]
    assert "campaign/step_b/host.jsonl" not in manifest["files"]  # live file copied as a frozen snapshot
    assert any(name.startswith("campaign/step_b/host_snapshots/") for name in manifest["files"])
    assert "code/v3_bundle_apply_receipt.json" in manifest["files"]
    assert "campaign/lanes/lane0/execution_events.jsonl" in manifest["files"]


def test_a_rerun_never_overwrites_the_acknowledged_first_attempt(tmp_path):
    remote, durable = finished_lane(tmp_path)
    transport, verify = host_pull.LocalTransport(), verifier()
    host_pull.pull_lane(transport, str(remote), durable / "lane0", verify)
    first_copy = (durable / "lane0" / "runs" / "unit" / "selected_checkpoint.json").read_text()
    # archive-failed moved attempt 1 on the VM; the rerun of the same name writes different bytes
    with reopen(remote, durable) as owner:
        shutil.rmtree(owner.root / "runs" / "unit")
        owner.admit("unit", 10)
        path = owner.root / "runs" / "unit" / "selected_checkpoint.json"
        path.parent.mkdir(parents=True)
        path.write_text('{"fit_status":"completed","attempt":2}\n')
        owner.record_result("unit", "completed", 10)
        owner.persist([path.parent])
    result = host_pull.pull_lane(transport, str(remote), durable / "lane0", verify)
    assert result["status"] == "acknowledged" and result["superseded_local_copies"] == ["runs/unit"]
    kept = list((durable / "lane0" / "superseded").glob("*/runs/unit/selected_checkpoint.json"))
    assert len(kept) == 1 and kept[0].read_text() == first_copy
    assert "attempt" in (durable / "lane0" / "runs" / "unit" / "selected_checkpoint.json").read_text()
