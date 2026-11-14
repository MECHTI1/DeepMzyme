"""CPU fixtures for the fold-0 session after G2: the device profile of each unit and the lane batch telemetry."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from ec_clean30 import amendment_rules, container_caps, phase3, phase4, workstation as ws  # noqa: E402

GIB = 1024 ** 3
VERIFICATION = ROOT / "docs/campaigns/ec_class_vs_clean30_2026-10/execution_v5/cpu_verification.json"


class Stop(Exception):
    """The continuation is observed up to its first queue; nothing is run."""


def continuation(tmp_path, monkeypatch, *, reviewed_profile=None):
    """Drive the coordinator's fold-0 continuation to its first queue and return that queue's request."""
    session = {"pod_id": "pod", "session_id": "session", "allocation_started_unix": 1000,
               "deadline_unix": 20000, "max_total_seconds": 20000}
    owner = SimpleNamespace(config_sha="a" * 64, register_output_roots=lambda roots: None)
    transport = SimpleNamespace(stop_controller=owner)
    config = {"execution_manifest": str(tmp_path / "manifest.json"), "hardware_profile": "runpod_rtx4090",
              "remote_campaign": "/workspace/campaign", "durable_root": str(tmp_path / "g2-primary"),
              "mirror_root": "/dev/shm/deepmzyme-test-g2-mirror"}
    template = {"execution_root": "/workspace/campaign/runs/fold0/phase4", "durable_root": str(tmp_path / "fold-primary")}
    if reviewed_profile:
        template["hardware_profile"] = reviewed_profile
    prepared = {"execution_manifest": config["execution_manifest"], "queue_actions": ["s1"],
                "durable_root": template["durable_root"], "mirror_root": "/dev/shm/deepmzyme-test-fold-mirror",
                "host_state_root": str(tmp_path / "fold-state"), "max_download_bytes": 1000,
                "remote_request_template": template}
    seen = {}

    def queue(request, **options):
        seen.update(request)
        raise Stop

    monkeypatch.setattr(phase4, "run_s1_queue", queue)
    state = {"completed": [name for _, name in ws.QUEUE]}
    if Path("/dev/shm").stat().st_dev == tmp_path.stat().st_dev:
        pytest.skip("The primary and mirror roots need two devices.")
    with pytest.raises(Stop):
        ws._continue_runpod_phase4(config, transport, state, tmp_path / "state.json", session, {}, "remote_manifest",
                                   "remote_probe", lambda: session, lambda *args: dict(prepared))
    return seen


def test_every_fold_unit_request_names_the_sessions_hardware_profile(tmp_path, monkeypatch):
    request = continuation(tmp_path, monkeypatch)
    assert request["hardware_profile"] == "runpod_rtx4090"
    # The pod's worker reads the template: without the profile its GPU check assumed another provider's device.
    assert request["remote_request_template"]["hardware_profile"] == "runpod_rtx4090"
    unit = {**request["remote_request_template"], "unit_id": "ecc30_clean_embed_base"}
    expected = phase3.hardware_profile(unit["hardware_profile"])
    assert expected["device_name"] == "NVIDIA GeForce RTX 4090" and repr(expected) in phase3.cuda_probe_program(
        unit.get("hardware_profile", "gcp_l4"))


def test_a_reviewed_request_cannot_name_another_device(tmp_path, monkeypatch):
    with pytest.raises(ws.WorkstationError, match="changes the owned allocation"):
        continuation(tmp_path, monkeypatch, reviewed_profile="runpod_l4")


def gpu_request(tmp_path, monkeypatch, *, request_profile, probe_profile):
    """One remote GPU unit as the pod's runtime gate sees it, with its authority checks replaced."""
    source = {"format": phase3.EXECUTION_V5_FORMAT, "hardware_profiles": ["runpod_l4", "runpod_rtx4090"]}
    root = tmp_path / "runs/fold0/phase4"
    view = SimpleNamespace(fold=0, default_runs_dir=tmp_path / "runs/fold0", manifest_sha256="5" * 64)
    monkeypatch.setattr(phase4, "_source_view", lambda request: (source, view))
    monkeypatch.setattr(phase4, "verify_authority", lambda request, source, resource=None: ({"allocations": {}}, {}))
    monkeypatch.setattr(phase3, "require_same_scientific_identity", lambda left, right: None)
    probe = tmp_path / "probe.json"
    record = {"status": "passed", "view_manifest_sha256": "5" * 64, "source": {}}
    if probe_profile:
        record["hardware_profile"] = probe_profile
    probe.write_text(json.dumps(record))
    request = {"provider": "runpod", "cap_prefix": ["cap", "--"], "unit_timeout_seconds": 90, "forecast_seconds": 100,
               "memory_limit_bytes": 1000, "device": "cuda", "deadline_unix": 20000.0, "max_total_seconds": 20000.0,
               "allocation_started_unix": 1000.0, "execution_root": str(root), "persistence_mode": "host_pull",
               "vm_probe": str(probe)}
    if request_profile:
        request["hardware_profile"] = request_profile
    return request


def test_the_pods_gate_admits_a_unit_with_the_qualified_profile(tmp_path, monkeypatch):
    request = gpu_request(tmp_path, monkeypatch, request_profile="runpod_rtx4090", probe_profile="runpod_rtx4090")
    assert phase4._runtime(request, resource="gpu")[2].max_total_seconds == 20000.0


@pytest.mark.parametrize("request_profile,probe_profile", [
    (None, "runpod_rtx4090"),  # The defect of the tenth window: the unit would assume another provider's device.
    ("gcp_l4", "runpod_rtx4090"), ("runpod_l4", "runpod_rtx4090"), ("runpod_rtx4090", None),
])
def test_the_pods_gate_refuses_a_unit_without_the_qualified_profile(tmp_path, monkeypatch, request_profile,
                                                                   probe_profile):
    request = gpu_request(tmp_path, monkeypatch, request_profile=request_profile, probe_profile=probe_profile)
    with pytest.raises(phase4.Phase4Error, match="hardware profile that its session's probe qualified"):
        phase4._runtime(request, resource="gpu")


def no_gpu_tool(monkeypatch, megabytes="1790"):
    monkeypatch.setattr(phase3.subprocess, "check_output", lambda argv, **options: megabytes + "\n")


def write(root, files):
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)


def test_container_sample_reads_the_unified_hierarchy(tmp_path, monkeypatch):
    no_gpu_tool(monkeypatch)
    write(tmp_path, {"memory.current": f"{9 * GIB}\n", "memory.stat": f"anon {5 * GIB}\ninactive_file {3 * GIB}\n",
                     "memory.swap.current": "0\n", "cpu.stat": "usage_usec 4000000\nuser_usec 3000000\n"})
    meter = {"version": 2, "memory": tmp_path, "cpu": tmp_path}
    clock = iter([100.0, 102.0])
    monkeypatch.setattr(phase3.time, "monotonic", lambda: next(clock))
    first = phase3._host_sample(meter)
    assert first["ram_bytes"] == 6 * GIB and first["swap_bytes"] == 0 and first["vram_bytes"] == 1790 * 1024 * 1024
    write(tmp_path, {"cpu.stat": "usage_usec 10000000\n"})
    second = phase3._host_sample(meter)
    # Six CPU-seconds in two seconds of wall time: three cores, as a percentage of one core.
    assert 100 * (second["cpu_busy"] - first["cpu_busy"]) / (second["cpu_span"] - first["cpu_span"]) == 300.0


def test_container_sample_reads_the_split_hierarchy(tmp_path, monkeypatch):
    no_gpu_tool(monkeypatch)
    memory, cpu = tmp_path / "memory", tmp_path / "cpu,cpuacct"
    write(memory, {"memory.usage_in_bytes": f"{12 * GIB}\n",
                   "memory.stat": f"cache {8 * GIB}\ntotal_inactive_file {7 * GIB}\ntotal_swap 0\n"})
    write(cpu, {"cpuacct.usage": "5000000000\n"})
    monkeypatch.setattr(container_caps, "validate_root", lambda root: Path(root))
    monkeypatch.setattr(container_caps, "_v1_roots", lambda root, writable=True: {"cpu": cpu, "memory": memory})
    meter = phase3._container_meter(tmp_path)
    assert meter == {"version": 1, "memory": memory, "cpu": cpu}
    sample = phase3._host_sample(meter)
    assert sample["ram_bytes"] == 5 * GIB and sample["swap_bytes"] == 0 and sample["cpu_busy"] == 5.0


def test_container_without_swap_accounting_keeps_the_machines_swap_figure(tmp_path, monkeypatch):
    no_gpu_tool(monkeypatch)
    write(tmp_path, {"memory.usage_in_bytes": "100\n", "memory.stat": "total_inactive_file 40\n",
                     "cpuacct.usage": "0\n"})
    sample = phase3._host_sample({"version": 1, "memory": tmp_path, "cpu": tmp_path})
    machine = phase3._host_sample()
    assert sample["ram_bytes"] == 60 and sample["swap_bytes"] == machine["swap_bytes"]


def test_a_container_without_accounting_files_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(container_caps, "validate_root", lambda root: Path(root))
    write(tmp_path, {"cgroup.type": "domain\n", "memory.current": "1\n"})
    with pytest.raises(phase3.Phase3Error, match="no memory or CPU accounting"):
        phase3._container_meter(tmp_path)


def test_whole_machine_sample_is_kept_for_a_dedicated_machine(monkeypatch):
    no_gpu_tool(monkeypatch)
    sample = phase3._host_sample()
    assert set(sample) == {"ram_bytes", "swap_bytes", "vram_bytes", "cpu_busy", "cpu_span"}
    assert sample["ram_bytes"] > 0 and sample["cpu_span"] > 0 and 0 <= sample["cpu_busy"]
    assert phase3._lane_meter({"hardware_profile": "gcp_l4"}) is None and phase3._lane_meter({}) is None


def test_runpod_lane_batch_measures_the_probed_container(tmp_path, monkeypatch):
    probe = tmp_path / "probe.json"
    probe.write_text(json.dumps({"container_caps_probe": {"cgroup_root": "/sys/fs/cgroup"}}))
    monkeypatch.setattr(phase3, "_container_meter", lambda root: {"root": str(root)})
    assert phase3._lane_meter({"hardware_profile": "runpod_rtx4090", "vm_probe": str(probe)}) == {
        "root": "/sys/fs/cgroup"}


def lane_batch(tmp_path, monkeypatch, meter):
    """The real lane worker with its runtime gate, children and samples replaced; returns its batch record."""
    no_gpu_tool(monkeypatch)
    samples = iter([{"ram_bytes": 2 * GIB, "swap_bytes": 0, "vram_bytes": 1.0, "cpu_busy": 10.0, "cpu_span": 100.0},
                    {"ram_bytes": 9 * GIB, "swap_bytes": 0, "vram_bytes": 2.0, "cpu_busy": 16.0, "cpu_span": 102.0}])
    seen = []
    monkeypatch.setattr(phase3, "_runtime", lambda request: None)
    monkeypatch.setattr(phase3, "_lane_meter", meter)
    monkeypatch.setattr(phase3, "_host_sample", lambda meter=None: seen.append(meter) or next(samples))
    monkeypatch.setattr(phase3, "_child_command",
                        lambda path, action, python: [sys.executable, "-c", "import time; time.sleep(0.2)"])
    wait = phase3.time.sleep
    monkeypatch.setattr(phase3.time, "sleep", lambda seconds: wait(0.6))  # One sample after the children ended.
    monkeypatch.setenv("DEEPMZYME_EXECUTION_LOCK_FD", "0")
    record = phase3.lane_worker({"execution_root": str(tmp_path), "batch_name": "batch", "python": sys.executable})
    return record, seen


def test_lane_worker_records_the_containers_peaks_and_scope(tmp_path, monkeypatch):
    record, seen = lane_batch(tmp_path, monkeypatch, lambda request: {"version": 2})
    assert seen == [{"version": 2}, {"version": 2}]
    assert record["observed"] == {"cpu_percent": 300.0, "ram_bytes": 9 * GIB, "swap_bytes": 0, "vram_bytes": 2.0,
                                  "oom": False, "read_guard_breach": False, "scope": "container"}
    assert record["status"] == "failed" and record["return_codes"] == [0, 0, 0]  # No timing receipts in this fixture.


def test_lane_worker_measures_the_machine_when_the_container_cannot_be_metered(tmp_path, monkeypatch):
    def unavailable(request):
        raise container_caps.CapError("The v1 mount must expose only this process's own container cgroup.")

    record, seen = lane_batch(tmp_path, monkeypatch, unavailable)
    assert seen == [None, None] and record["observed"]["scope"] == "host"
    assert record["observed"]["container_meter_error"].startswith("CapError: The v1 mount")
    assert record["return_codes"] == [0, 0, 0]  # The three lanes still ran and were measured.


def test_the_tenth_windows_lane_batch_passes_once_its_own_container_is_measured():
    """The frozen gate is unchanged: only the two machine-wide peaks kept three passing lanes from being counted."""
    truth = [index % 7 for index in range(140)]
    identity = {"scientific_identity_sha256": "a" * 64, "fit_chain": ["b" * 64, "c" * 64]}
    runs = [{"recipe": name, "ids": [f"q{index}" for index in range(140)], "true": truth, "predicted": truth,
             "wall_seconds": seconds, **identity}
            for name, seconds in (("lane_serial_1", 246.0), ("lane_serial_2", 247.0), ("lane_parallel_1", 290.0),
                                  ("lane_parallel_2", 295.0), ("lane_parallel_3", 300.0))]
    limits = {"cpu_percent": 600, "ram_bytes": 24 * GIB, "vram_bytes": 21 * GIB}
    peaks = {"swap_bytes": 0, "oom": False, "read_guard_breach": False, "vram_bytes": 1876951040.0}
    machine = phase3.lane_gate(runs[:2], runs[2:], batch_wall_seconds=301.2, limits=limits,
                               observed={**peaks, "cpu_percent": 624.6, "ram_bytes": 47242424320, "scope": "host"})
    assert machine["status"] == "failed" and machine["lanes"] == 1 and machine["speedup"] > 2.4
    assert all(row["passed"] for row in machine["comparisons"]) and not machine["resource_passed"]
    own = phase3.lane_gate(runs[:2], runs[2:], batch_wall_seconds=301.2, limits=limits,
                           observed={**peaks, "cpu_percent": 310.0, "ram_bytes": 14 * GIB, "scope": "container"})
    assert own["status"] == "passed" and own["lanes"] == 3 and own["observed"]["scope"] == "container"
    over = phase3.lane_gate(runs[:2], runs[2:], batch_wall_seconds=301.2, limits=limits,
                            observed={**peaks, "cpu_percent": 310.0, "ram_bytes": 25 * GIB, "scope": "container"})
    assert over["status"] == "failed" and over["lanes"] == 1  # The limits still bind the container's own use.


def test_the_frozen_gates_keep_their_recorded_source_pins():
    recorded = json.loads(VERIFICATION.read_text())["checks"]["pinned_functions"]
    live = amendment_rules.pinned_function_hashes(ROOT, amendment_rules.PINNED_EXECUTION_FUNCTIONS)
    assert live == {key: value for key, value in recorded.items() if key != "status"}
