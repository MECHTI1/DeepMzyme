"""CPU fixtures for the bounded SSH reconnect of the workstation transport; no network, provider or GPU use."""
from pathlib import Path
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from ec_clean30 import workstation as ws  # noqa: E402

REFUSED = "ssh: connect to host 203.0.113.7 port 14975: Connection refused\n"


class Clock:
    """A monotonic and wall clock that advances only when the transport sleeps."""

    def __init__(self):
        self.now, self.sleeps = 1000.0, []

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(time, "sleep", clock.sleep)
    monkeypatch.setattr(time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(time, "time", lambda: clock.now)
    return clock


class Controller:
    def __init__(self, tmp_path, fail=False):
        self.tmp_path, self.fail, self.writes = tmp_path, fail, 0

    def heartbeat(self):
        pass

    def write_ssh_config(self):
        self.writes += 1
        if self.fail:
            raise RuntimeError("Provider did not return a direct SSH IP address.")
        return self.tmp_path / f"ssh_config_{self.writes}"


def runpod_transport(tmp_path, controller):
    config = {"state_root": str(tmp_path), "bootstrap_python": sys.executable, "runpod": {"ssh_alias": "fixture"}}
    return ws.RunpodTransport(config, controller)


def ssh_results(monkeypatch, results):
    """Each ssh call consumes one (returncode, stderr); the calls are recorded."""
    calls, pending = [], list(results)

    def execute(argv, **kwargs):
        calls.append(list(argv))
        returncode, stderr = pending.pop(0)
        return subprocess.CompletedProcess(argv, returncode, "ready" if returncode == 0 else "", stderr)

    monkeypatch.setattr(subprocess, "run", execute)
    return calls


def test_refused_connect_is_repeated_with_backoff_and_a_renewed_endpoint(tmp_path, monkeypatch, clock):
    controller = Controller(tmp_path)
    transport = runpod_transport(tmp_path, controller)
    calls = ssh_results(monkeypatch, [(255, REFUSED)] * 4 + [(0, "")])
    assert transport.remote(["python", "runner.py"]) == "ready"
    assert clock.sleeps == [5, 10, 15, 15]
    assert len(calls) == 5 and len({call[-1] for call in calls}) == 1  # The identical command, never a second job.
    assert controller.writes == 4
    assert [call[2] for call in calls[1:]] == [str(tmp_path / f"ssh_config_{index}") for index in range(1, 5)]


def test_reconnect_stops_at_its_bound_and_reports_the_ssh_failure(tmp_path, monkeypatch, clock):
    controller = Controller(tmp_path, fail=True)  # A provider that reports no endpoint keeps the previous one.
    transport = runpod_transport(tmp_path, controller)
    previous = transport.ssh_config
    calls = ssh_results(monkeypatch, [(255, REFUSED)] * 100)
    with pytest.raises(ws.WorkstationError, match=r"Remote command failed \(255\): ssh: connect to host"):
        transport.remote(["python", "runner.py"])
    assert sum(clock.sleeps) <= ws.SSH_RECONNECT_SECONDS < sum(clock.sleeps) + max(ws.SSH_RECONNECT_DELAYS)
    assert len(calls) == len(clock.sleeps) + 1 == controller.writes + 1
    assert transport.ssh_config == previous


def test_reconnect_never_enters_the_reserve_before_the_session_deadline(tmp_path, monkeypatch, clock):
    transport = runpod_transport(tmp_path, Controller(tmp_path))
    transport.deadline_unix, transport.reserve_seconds = clock.now + 60, 30
    calls = ssh_results(monkeypatch, [(255, REFUSED)] * 100)
    with pytest.raises(ws.WorkstationError, match="Connection refused"):
        transport.remote(["python", "runner.py"])
    assert clock.sleeps == [5, 10] and len(calls) == 3  # A third wait of 15 s would reach the 30 s reserve.


@pytest.mark.parametrize("returncode,stderr", [
    (255, "client_loop: send disconnect: Broken pipe\n"),  # Lost after delivery: the command may have run.
    (255, "Timeout, server fixture not responding.\n"),
    (255, "Host key verification failed.\n"),
    (2, REFUSED),  # A remote command's own exit status is never a transport failure.
    (1, ""),
])
def test_only_an_undelivered_command_is_repeated(tmp_path, monkeypatch, clock, returncode, stderr):
    controller = Controller(tmp_path)
    transport = runpod_transport(tmp_path, controller)
    calls = ssh_results(monkeypatch, [(returncode, stderr), (0, "")])
    with pytest.raises(ws.WorkstationError, match=rf"Remote command failed \({returncode}\)"):
        transport.remote(["python", "runner.py"])
    assert len(calls) == 1 and clock.sleeps == [] and controller.writes == 0


def test_base_transport_reconnects_without_a_provider(tmp_path, monkeypatch, clock):
    transport = ws.ControllerTransport(tmp_path, "fixture", bootstrap_python=sys.executable)
    calls = ssh_results(monkeypatch, [(255, REFUSED), (0, "")])
    assert transport.remote(["true"]) == "ready"
    assert clock.sleeps == [5] and len(calls) == 2 and transport.ssh_config == tmp_path / "state/ssh_config"


def test_single_file_copy_is_repeated_once_after_a_transport_failure(tmp_path, monkeypatch, clock):
    transport = runpod_transport(tmp_path, Controller(tmp_path))
    calls, outcomes = [], {"scp": [255, 0], "ssh": [0]}

    def execute(argv, **kwargs):
        calls.append(argv[0])
        returncode = outcomes[argv[0]].pop(0)
        if returncode and kwargs.get("check"):
            raise subprocess.CalledProcessError(returncode, argv)
        return subprocess.CompletedProcess(argv, returncode, "ready", "")

    monkeypatch.setattr(subprocess, "run", execute)
    transport.download("/workspace/receipt.json", tmp_path / "local/receipt.json")
    assert calls == ["scp", "ssh", "scp"]  # The reachability read sits between the two attempts.

    calls.clear()
    outcomes.update(scp=[255, 255], ssh=[0])
    with pytest.raises(subprocess.CalledProcessError):
        transport.download("/workspace/receipt.json", tmp_path / "local/receipt.json")
    assert calls == ["scp", "ssh", "scp"]

    calls.clear()
    outcomes.update(scp=[1], ssh=[])
    with pytest.raises(subprocess.CalledProcessError):
        transport.download("/workspace/receipt.json", tmp_path / "local/receipt.json")
    assert calls == ["scp"]  # A copy error that is not the transport's is never repeated.


def test_unit_poll_survives_a_refused_connect_without_a_second_launch(tmp_path, monkeypatch, clock):
    """The polling read of a running unit returns after the endpoint is back; the job is launched once."""
    transport = runpod_transport(tmp_path, Controller(tmp_path))
    launches, reads = [], []
    states = iter([(255, REFUSED), (255, REFUSED), (0, "")])

    def execute(argv, **kwargs):
        reads.append(argv[-1])
        returncode, stderr = next(states)
        return subprocess.CompletedProcess(argv, returncode, '{"returncode": 0}' if returncode == 0 else "", stderr)

    monkeypatch.setattr(subprocess, "run", execute)
    monkeypatch.setattr(transport, "launch", lambda *args, **kwargs: launches.append(args))
    assert transport.read_json("/workspace/unit.job/result.json", optional=True) == {"returncode": 0}
    assert clock.sleeps == [5, 10] and len(reads) == 3 and len(set(reads)) == 1 and launches == []
