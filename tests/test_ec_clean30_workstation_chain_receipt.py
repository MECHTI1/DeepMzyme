"""CPU-only session receipt plumbing; no full verification, cloud, SSH or fitting."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from ec_clean30 import workstation as ws  # noqa: E402


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


@pytest.fixture
def receipt_case(tmp_path, monkeypatch):
    import ec_clean30

    chain = [digest("first-link"), digest("second-link")]
    document = {"format": ws.phase3.EXECUTION_V5_FORMAT, "amendment_chain": chain,
                "fit_chain": chain, "scientific_identity_sha256": digest("science"),
                "freeze1_sha256": digest("freeze"), "amendment_path": "docs/amendment2/amendment.json"}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(document))
    config = {"execution_manifest": str(manifest), "execution_scope": "g2_then_fold0_v5",
              "state_root": str(tmp_path / "state"), "remote_campaign": str(tmp_path / "remote")}
    verified = {"published": True, "tip_sequence": 2, "chain": chain,
                "fit_chain": chain, "scientific_identity_sha256": document["scientific_identity_sha256"]}
    source = {**document, "execution_manifest_sha256": digest("execution")}
    events = []
    builder, checker = ModuleType("ec_clean30.amendment2"), ModuleType("ec_clean30.amendment_chain")

    def write(tip, freeze, out, **kwargs):
        events.append(("write", tip, freeze, kwargs))
        out.write_text(json.dumps({"require_published": True, "verification": verified}))

    def check(path, **kwargs):
        events.append(("check", path, kwargs))
        assert path.is_file()
        return dict(verified)

    def source_check(path, **kwargs):
        events.append(("source", path, kwargs))
        return dict(source)

    builder.write_chain_receipt = write
    checker.check_chain_receipt = check
    for name, module in (("amendment2", builder), ("amendment_chain", checker)):
        monkeypatch.setitem(sys.modules, f"ec_clean30.{name}", module)
        monkeypatch.setattr(ec_clean30, name, module, raising=False)
    monkeypatch.setattr(ws.phase3, "verify_execution_source", source_check)
    return config, document, verified, source, events


def test_session_writes_once_and_rechecks_recovery(receipt_case):
    config, document, _, source, events = receipt_case
    first = ws.prepare_session_chain_receipt(config)
    second = ws.prepare_session_chain_receipt(config)
    assert first == second and first["expected_chain"] == document["amendment_chain"]
    assert [event[0] for event in events] == ["write", "check", "source", "check", "source"]
    write = events[0]
    assert write[3] == {"draft": False, "expected_chain": document["amendment_chain"], "repo": ws.phase3.REPO_ROOT}
    for event in (events[2], events[4]):
        assert event[2]["require_published"] is True
        assert event[2]["chain_receipt_path"] == Path(first["path"])
        assert event[2]["expected_chain"] == document["amendment_chain"]
    assert first["scientific_identity_sha256"] == source["scientific_identity_sha256"]


@pytest.mark.parametrize("change", ["short", "draft", "identity", "fit_chain"])
def test_receipt_refuses_short_or_foreign_source(receipt_case, change):
    config, document, verified, _, events = receipt_case
    if change == "short":
        document["amendment_chain"] = document["amendment_chain"][:1]
        Path(config["execution_manifest"]).write_text(json.dumps(document))
    elif change == "draft":
        verified["published"] = False
    elif change == "identity":
        verified["scientific_identity_sha256"] = digest("foreign")
    else:
        verified["fit_chain"] = [digest("foreign")]
    with pytest.raises(ws.WorkstationError):
        ws.prepare_session_chain_receipt(config)
    assert not any(event[0] == "source" for event in events)
    if change == "short":
        assert not events


def test_stale_receipt_is_not_overwritten(receipt_case, monkeypatch):
    config, _, _, _, events = receipt_case
    path = ws.session_chain_receipt_path(config)
    path.parent.mkdir()
    path.write_text("stale")

    def refuse(*args, **kwargs):
        raise ValueError("receipt source changed")

    monkeypatch.setattr(sys.modules["ec_clean30.amendment_chain"], "check_chain_receipt", refuse)
    with pytest.raises(ValueError, match="source changed"):
        ws.prepare_session_chain_receipt(config)
    assert path.read_text() == "stale" and not events


def test_preflight_source_only_checks_receipt(receipt_case):
    config, document, _, _, events = receipt_case
    ws._execution_source(config, published=True)
    assert [event[0] for event in events] == ["source"]
    assert events[0][2]["expected_chain"] == document["amendment_chain"]
    assert events[0][2]["chain_receipt_path"] == ws.session_chain_receipt_path(config)


class FileTransport:
    def __init__(self):
        self.uploads = 0

    def stat_file(self, path):
        path = Path(path)
        return {"sha256": ws._sha(path), "size_bytes": path.stat().st_size} if path.is_file() else None

    def upload(self, local, remote):
        self.uploads += 1
        path = Path(remote)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(Path(local).read_bytes())

    def remote(self, argv):
        assert argv[:2] == ["mv", "--"]
        Path(argv[2]).rename(argv[3])


def test_ship_receipt_and_refuse_foreign_remote_bytes(receipt_case):
    config, document, _, source, _ = receipt_case
    ws.prepare_session_chain_receipt(config)
    transport = FileTransport()
    fields = ws.stage_session_chain_receipt(config, transport, source)
    assert fields["expected_chain"] == document["amendment_chain"]
    assert Path(fields["amendment_chain_receipt"]).read_bytes() == ws.session_chain_receipt_path(config).read_bytes()
    assert ws.stage_session_chain_receipt(config, transport, source) == fields
    assert transport.uploads == 1
    Path(fields["amendment_chain_receipt"]).write_text("foreign")
    with pytest.raises(ws.WorkstationError, match="foreign or changed"):
        ws.stage_session_chain_receipt(config, transport, source)
    assert transport.uploads == 1


def test_legacy_source_and_transport_need_no_receipt(monkeypatch):
    calls = []
    monkeypatch.setattr(ws.phase3, "verify_execution_source", lambda *a, **kw: calls.append((a, kw)) or {})
    config = {"execution_manifest": "/legacy.json"}
    ws._execution_source(config, published=True)
    assert calls == [((Path("/legacy.json"),), {"require_published": True})]
    assert ws.stage_session_chain_receipt(config, object(), {"format": "legacy"}) == {}


def test_remote_environment_scrubs_local_and_remote_git_variables(tmp_path, monkeypatch):
    actual_run = subprocess.run
    monkeypatch.setenv("GIT_DIR", "/foreign")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.worktree")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "/foreign")

    def fake_ssh(argv, **kwargs):
        assert "GIT_DIR" not in kwargs["env"] and "GIT_CONFIG_COUNT" not in kwargs["env"]
        command = shlex.split(argv[-1])
        # This child deliberately inherits the hostile remote environment; its exec wrapper must scrub it.
        return actual_run(command, capture_output=True, text=True, check=True)

    monkeypatch.setattr(ws.subprocess, "run", fake_ssh)
    transport = ws.ControllerTransport(tmp_path, "fixture", bootstrap_python=sys.executable)
    output = transport.remote([sys.executable, "-c",
                               "import json,os; print(json.dumps([k for k in os.environ if k.startswith('GIT_')]))"])
    assert json.loads(output) == []


def test_checkout_git_commands_receive_scrubbed_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_WORK_TREE", "/foreign")
    calls = []
    monkeypatch.setattr(ws.subprocess, "run", lambda argv, **kw: calls.append((argv, kw)))
    monkeypatch.setattr(ws.subprocess, "check_output", lambda argv, **kw: calls.append((argv, kw)) or "")

    class CheckoutTransport:
        def remote(self, argv, **kwargs):
            monkeypatch.setattr(sys, "argv", ["-c", *argv[3:]])
            exec(argv[2], {})

    ws._stage_checkout({"provider": "runpod", "repo_url": "https://github.com/example/repo.git",
                        "bootstrap_python": sys.executable, "remote_repo": str(tmp_path)},
                       CheckoutTransport(), {"git_head": "fixture-head"})
    assert len(calls) == 3
    assert all(not any(key.startswith("GIT_") for key in options["env"]) for _, options in calls)
