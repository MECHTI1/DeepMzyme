"""Closeout inventory through the real Runpod transport: ssh, scp and rsync are replaced by local copies.

The remote programs run unchanged in a local shell, so the real inventory program, its control file, the
digest check and the full two-copy preservation run against a synthetic G2 root; nothing reaches a provider.
"""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from ec_clean30 import workstation as ws  # noqa: E402

SSH_OUTPUT_CAP = 4 * 1024 * 1024


def _world(base, n_sequences):
    """A G2 root laid out like F6 preparation: two sequence-mean cache files per union sequence."""
    campaign = base / "remote" / "campaign"
    root = campaign / "runs/fold0/g2"
    attempt = root / "runs/ecc30_pretrain_fold0_attempt1"
    cache = attempt / "gpu_cache"
    cache.mkdir(parents=True)
    for index in range(n_sequences):
        digest = hashlib.sha256(f"union-{index}".encode()).hexdigest()
        for layer in ("final", "block26"):
            (cache / f"{digest}__esmc300m_{layer}_seqmean.pt").write_bytes(b"%d\n" % index)
    (attempt / "pretrain_attachment.json").write_text(json.dumps({"status": "restored"}))
    (root / "execution_state.json").write_text("{}")
    deadline = time.time() + 3600
    config = {"state_root": str(base / "state"), "runpod": {"ssh_alias": "synthetic-pod"},
              "bootstrap_python": sys.executable, "remote_campaign": str(campaign),
              "durable_root": str(base / "primary"), "mirror_root": str(base / "mirror")}
    controller = SimpleNamespace(heartbeat=lambda: None, config_sha="a" * 64)
    transport = ws.RunpodTransport(config, controller)
    transport.deadline_unix, transport.reserve_seconds = deadline, 15
    session = {"pod_id": "pod", "session_id": "session", "deadline_unix": deadline}
    return SimpleNamespace(config=config, controller=controller, transport=transport, session=session, root=root,
                           campaign=campaign)


@pytest.fixture
def local_provider(monkeypatch):
    """Run every ssh command in a local shell; scp and rsync copy local files."""
    real_run = subprocess.run
    log = SimpleNamespace(ssh=[], tamper=False)

    def run(argv, *args, **kwargs):
        tool = argv[0]
        if tool == "ssh":
            result = real_run(["bash", "-c", argv[-1]], *args, **kwargs)
            log.ssh.append({"timeout": kwargs["timeout"], "stdout_chars": len(result.stdout or "")})
            return result
        if tool == "scp":
            remote, local = argv[-2].split(":", 1)[1], Path(argv[-1])
            data = Path(remote).read_bytes()
            if log.tamper:
                # Same length, still JSON: one recorded digest changes.
                match = re.search(rb'"sha256": "([0-9a-f])', data)
                flipped = b"1" if match.group(1) != b"1" else b"2"
                data = data[:match.start(1)] + flipped + data[match.end(1):]
            local.write_bytes(data)
            return subprocess.CompletedProcess(argv, 0)
        if tool == "rsync":
            listing = next(item for item in argv if item.startswith("--files-from=")).split("=", 1)[1]
            source, target = Path(argv[-2].split(":", 1)[1]), Path(argv[-1])
            for name in Path(listing).read_text().splitlines():
                (target / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / name, target / name)
            return subprocess.CompletedProcess(argv, 0)
        raise AssertionError(f"Unexpected provider process: {tool}")

    monkeypatch.setattr(ws.subprocess, "run", run)
    # Thousands of synthetic files: durability syncs add nothing to this transfer check.
    monkeypatch.setattr(ws.os, "fsync", lambda descriptor: None)
    return log


def test_inventory_over_the_ssh_output_cap_is_preserved_through_a_verified_file(tmp_path, local_provider):
    world = _world(tmp_path, 12500)
    path = ws._preserve_runpod_outputs(world.config, world.transport, world.controller, world.session)
    receipt = json.loads(path.read_text())
    downloaded = list((Path(world.config["state_root"]) / "closeout_inventories").iterdir())
    assert len(downloaded) == 1 and downloaded[0].stat().st_size > SSH_OUTPUT_CAP
    inventory = json.loads(downloaded[0].read_text())
    assert len(inventory["files"]) == 25000 + 2
    assert [row["path"] for row in inventory["files"]] == sorted(row["path"] for row in inventory["files"])
    # Only the work estimate and the file digest travel as SSH output.
    assert len(local_provider.ssh) == 2 and max(row["stdout_chars"] for row in local_provider.ssh) < 1024
    control = world.campaign / "g2_requests/closeout" / downloaded[0].name
    assert control.is_file() and world.root not in control.parents
    assert ws._sha(control) == ws._sha(downloaded[0])
    assert receipt["files"][:-1] == inventory["files"]
    assert receipt["files"][-1]["path"] == receipt["remote_inventory_path"] == "closeout/remote_inventory.json"
    local = json.loads((tmp_path / "primary/closeout/remote_inventory.json").read_text())
    assert local["files"] == inventory["files"] and local["session_id"] == "session"
    for entry in inventory["files"][::997]:
        for side in ("primary", "mirror"):
            assert ws._sha(tmp_path / side / entry["path"]) == entry["sha256"]
    # Hashing time scales with the counted work and stays inside the session's stop reserve.
    scan, hashing = (row["timeout"] for row in local_provider.ssh)
    assert hashing > ws.INVENTORY_MIN_SECONDS + 25000 * ws.INVENTORY_FILE_SECONDS - 1
    assert max(scan, hashing) <= world.session["deadline_unix"] - time.time() - world.transport.reserve_seconds + 5


def test_tampered_inventory_download_is_refused(tmp_path, local_provider):
    world = _world(tmp_path, 20)
    local_provider.tamper = True
    with pytest.raises(ws.WorkstationError, match="differs from its remote digest"):
        ws._preserve_runpod_outputs(world.config, world.transport, world.controller, world.session)
    assert not (tmp_path / "primary/closeout").exists()


def test_inventory_beyond_its_size_bound_is_refused_on_the_pod(tmp_path, local_provider, monkeypatch):
    world = _world(tmp_path, 20)
    monkeypatch.setattr(ws, "INVENTORY_LIMIT_BYTES", 256)
    with pytest.raises(ws.WorkstationError, match="Remote command failed"):
        ws._preserve_runpod_outputs(world.config, world.transport, world.controller, world.session)
    assert not list((world.campaign / "g2_requests/closeout").glob("*.json"))


def test_hashing_timeout_never_passes_the_stop_reserve(tmp_path, local_provider, monkeypatch):
    world = _world(tmp_path, 20)
    world.transport.deadline_unix = world.session["deadline_unix"] = time.time() + 200
    monkeypatch.setattr(ws, "INVENTORY_MIN_SECONDS", 10_000)
    ws._preserve_runpod_outputs(world.config, world.transport, world.controller, world.session)
    assert len(local_provider.ssh) == 2
    assert all(row["timeout"] <= 200 - world.transport.reserve_seconds for row in local_provider.ssh)


def test_pending_marker_still_blocks_the_inventory(tmp_path, local_provider):
    world = _world(tmp_path, 3)
    marker = world.root / "runs/ecc30_pretrain_fold0_attempt1/pretrain_attachment.json"
    marker.write_text(json.dumps({"status": "pending"}))
    with pytest.raises(ws.WorkstationError, match="Remote command failed"):
        ws._preserve_runpod_outputs(world.config, world.transport, world.controller, world.session)
    assert not (world.campaign / "g2_requests/closeout").exists()
    assert not (tmp_path / "state" / "closeout_inventories").exists()
