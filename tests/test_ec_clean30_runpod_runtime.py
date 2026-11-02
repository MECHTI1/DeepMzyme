"""CPU-only checks binding the provider profiles to measured containment and full-model qualification."""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from ec_clean30 import container_caps, phase3
from test_ec_clean30_container_caps import kernel
from test_ec_clean30_phase3 import transfer_view, _sha, _json_write, _source_identity

REAL_POPEN = subprocess.Popen


def model_evidence(profile="runpod_rtx4090"):
    return {"format": "deepmzyme_ec_clean30_esm1b_model_probe_v1", "status": "passed",
            "hardware_profile": profile, "device_name": phase3.hardware_profile(profile)["device_name"],
            "device_count": 1, "total_memory_bytes": 24 * 1024 ** 3, "torch": "fixture", "cuda": "fixture",
            "kernel_value": 8.0, "tf32_override": "0", "weights_sha256": phase3.ESM1B_WEIGHTS_SHA256,
            "contact_weights_sha256": phase3.ESM1B_CONTACT_SHA256, "model_on_cuda": True,
            "model_layers": 33, "model_parameter_count": 650_000_000, "synthetic_forward_finite": True,
            "cpu_gpu_replay": {"sequence_sha256": hashlib.sha256(b"MKT").hexdigest(), "representation_layer": 33,
                               "representation_shape": [1, 5, 1280], "atol": 1e-4, "rtol": 1e-4,
                               "max_abs_difference": 0.0, "max_reference_abs": 1.0, "max_scaled_error": 0.0,
                               "passed": True, "cpu_finite": True, "gpu_finite": True},
            "hostname": socket.gethostname(), "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "python_executable": sys.executable, "created_unix": time.time()}


@pytest.mark.parametrize("case", ["weights", "companion", "cpu_model", "partial_model", "nonfinite", "other_gpu"])
def test_full_model_evidence_refuses_partial_or_unpinned_qualification(case):
    evidence = model_evidence()
    key, value = {"weights": ("weights_sha256", "0" * 64), "companion": ("contact_weights_sha256", "0" * 64),
                  "cpu_model": ("model_on_cuda", False), "partial_model": ("model_parameter_count", 1_000),
                  "nonfinite": ("synthetic_forward_finite", False), "other_gpu": ("device_name", "NVIDIA A100")}[case]
    evidence[key] = value
    with pytest.raises(phase3.Phase3Error):
        phase3.validate_esm1b_model_evidence(evidence, "runpod_rtx4090")


@pytest.mark.parametrize("difference", [0.0, 5e-5, 2e-4])
def test_full_model_probe_loads_pinned_files_and_compares_cpu_gpu_forward(tmp_path, monkeypatch, capsys, difference):
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    weights = tmp_path / "esm1b_t33_650M_UR50S.pt"
    companion = tmp_path / "esm1b_t33_650M_UR50S-contact-regression.pt"
    weights.write_bytes(b"model fixture")
    companion.write_bytes(b"companion fixture")
    monkeypatch.setattr(phase3, "ESM1B_WEIGHTS_SHA256", _sha(weights))
    monkeypatch.setattr(phase3, "ESM1B_CONTACT_SHA256", _sha(companion))
    calls = []

    class Tensor:
        shape = (1, 5, 1280)

        def __init__(self, value=8.0, is_cuda=False):
            self.value = value
            self.is_cuda = is_cuda

        def __mul__(self, value):
            return self

        def __rmul__(self, value):
            return Tensor(value*self.value)

        def __radd__(self, value):
            return Tensor(value+self.value)

        def __truediv__(self, other):
            return Tensor(self.value/other.value)

        def sum(self):
            return self

        def item(self):
            return self.value

        def numel(self):
            return 650_000_000

        def cuda(self):
            return Tensor(self.value, True)

        def cpu(self):
            return Tensor(self.value, False)

        def __sub__(self, other):
            return Tensor(self.value-other.value)

        def abs(self):
            return Tensor(abs(self.value))

        def max(self):
            return self

    class Model:
        num_layers = 33
        is_cuda = False

        def eval(self):
            calls.append("eval")

        def cuda(self):
            calls.append("full_model_cuda")
            self.is_cuda = True
            return self

        def parameters(self):
            return iter([Tensor(is_cuda=self.is_cuda)])

        def __call__(self, tokens, *, repr_layers, return_contacts):
            assert tokens.is_cuda == self.is_cuda and repr_layers == [33] and return_contacts is False
            calls.append("gpu_forward" if self.is_cuda else "cpu_forward")
            return {"representations": {33: Tensor(difference if self.is_cuda else 0.0, self.is_cuda)}}

    def load(path):
        assert path == str(weights)
        calls.append("load_local")
        alphabet = SimpleNamespace(get_batch_converter=lambda: lambda rows: ([rows[0][0]], [rows[0][1]], Tensor()))
        return Model(), alphabet

    cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                           get_device_name=lambda index: "NVIDIA GeForce RTX 4090",
                           get_device_properties=lambda index: SimpleNamespace(total_memory=24 * 1024 ** 3),
                           synchronize=lambda: None)
    torch = SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda="fixture"), __version__="fixture",
                            ones=lambda *args, **kwargs: Tensor(), no_grad=nullcontext,
                            allclose=lambda left, right, *, atol, rtol: abs(left.value-right.value) <= atol + rtol*abs(right.value),
                            isfinite=lambda value: SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: True)))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "esm", SimpleNamespace(pretrained=SimpleNamespace(load_model_and_alphabet_local=load)))
    program = phase3.esm1b_model_probe_program("runpod_rtx4090", str(weights), _sha(weights), _sha(companion))
    if difference > phase3.ESM1B_REPLAY_ATOL:
        with pytest.raises(AssertionError, match="CPU/GPU replay differs"):
            exec(program, {})
        assert capsys.readouterr().out == ""
        return
    exec(program, {})
    result = json.loads(capsys.readouterr().out)
    assert phase3.validate_esm1b_model_evidence(result, "runpod_rtx4090")["model_on_cuda"]
    assert result["cpu_gpu_replay"]["max_abs_difference"] == difference
    assert calls == ["load_local", "eval", "cpu_forward", "full_model_cuda", "gpu_forward"]
    weights.write_bytes(b"changed model")
    with pytest.raises(AssertionError, match="weight hash"):
        exec(program, {})
    assert calls == ["load_local", "eval", "cpu_forward", "full_model_cuda", "gpu_forward"]


@pytest.mark.parametrize("key,value", [("atol", 1e-3), ("rtol", 1e-3), ("rtol", False),
                                     ("max_abs_difference", 2e-3), ("max_abs_difference", float("nan")),
                                     ("max_scaled_error", 1.01), ("max_reference_abs", float("nan")),
                                     ("max_abs_difference", True), ("passed", False), ("cpu_finite", False),
                                     ("gpu_finite", False), ("representation_shape", [1, 5, 1])])
def test_full_model_evidence_refuses_loose_or_failed_cpu_gpu_replay(key, value):
    evidence = model_evidence()
    evidence["cpu_gpu_replay"][key] = value
    with pytest.raises(phase3.Phase3Error, match="CPU/GPU replay"):
        phase3.validate_esm1b_model_evidence(evidence, "runpod_rtx4090")


def test_model_probe_refuses_remote_or_unpinned_weights():
    with pytest.raises(phase3.Phase3Error, match="absolute local"):
        phase3.esm1b_model_probe_program("runpod_l4", "https://example.invalid/weights", "0" * 64, "1" * 64)
    with pytest.raises(phase3.Phase3Error, match="pinned"):
        phase3.esm1b_model_probe_program("runpod_l4", "/weights.pt", "0" * 64, "1" * 64)


@pytest.fixture
def runpod_probe(kernel, transfer_view, tmp_path, monkeypatch):
    source = {**_source_identity(), "format": phase3.RUNPOD_FORMAT, "hardware_profiles": sorted(phase3.GPU_PROFILES),
              "source_sha256": {"src/ec_clean30/container_caps.py": _sha(Path(container_caps.__file__))}}
    monkeypatch.setattr(phase3, "verify_execution_source", lambda *args, **kwargs: source)
    monkeypatch.setattr(phase3.commands, "dry_run", lambda *args, **kwargs: {"status": "passed"})
    evidence = container_caps.probe(kernel[0])
    monkeypatch.setattr(container_caps.subprocess, "Popen", REAL_POPEN)
    evidence_path = tmp_path / "caps.json"
    _json_write(evidence_path, evidence)
    prefix = container_caps.cap_prefix(sys.executable, phase3.REPO_ROOT, evidence["cgroup_root"],
                                        cpu_cores=2, memory_bytes=8 * 1024 ** 3)
    return source, evidence_path, prefix


@pytest.mark.parametrize("case", ["passed", "different_host", "different_boot", "different_helper", "relaxed_cap", "missing_probe"])
def test_vm_probe_binds_measured_caps_to_current_host_and_authenticated_source(runpod_probe, transfer_view, tmp_path, case):
    source, evidence_path, prefix = runpod_probe
    evidence = json.loads(evidence_path.read_text())
    if case == "different_host":
        evidence["hostname"] = "another-host"
    elif case == "different_boot":
        evidence["boot_id"] = "0" * 8 + "-" + "0" * 4 + "-" + "0" * 4 + "-" + "0" * 4 + "-" + "0" * 12
    elif case == "different_helper":
        evidence["helper_sha256"] = "0" * 64
    elif case == "relaxed_cap":
        prefix = prefix[:-4] + ["--memory-bytes", str(32 * 1024 ** 3), "--"]
    _json_write(evidence_path, evidence)
    out = tmp_path / "vm.json"
    args = {"cap_prefix": prefix, "python": sys.executable, "smoke": True,
            "hardware_profile": "runpod_rtx4090", "cap_probe_path": None if case == "missing_probe" else evidence_path}
    if case == "passed":
        result = phase3.vm_probe(transfer_view.root, tmp_path / "manifest.json", out, **args)
        assert result["container_caps_probe_sha256"] == _sha(evidence_path)
        assert result["hardware_profile"] == "runpod_rtx4090"
    else:
        with pytest.raises((phase3.Phase3Error, container_caps.CapError)):
            phase3.vm_probe(transfer_view.root, tmp_path / "manifest.json", out, **args)
        assert not out.exists()


@pytest.mark.parametrize("case", ["passed", "bytes", "host", "boot", "older_allocation", "interpreter", "missing"])
def test_model_qualification_is_bound_to_host_allocation_and_actual_clean_interpreter(transfer_view, tmp_path, case):
    evidence = model_evidence()
    path = tmp_path / "model.json"
    if case == "host":
        evidence["hostname"] = "another-host"
    elif case == "boot":
        evidence["boot_id"] = "00000000-0000-0000-0000-000000000000"
    elif case == "older_allocation":
        evidence["created_unix"] -= 3600
    _json_write(path, evidence)
    request = {"esm1b_model_probe": str(path), "esm1b_model_probe_sha256": _sha(path),
               "kind": "esm1b", "clean_python": sys.executable,
               "allocation_started_unix": time.time() - 1, "deadline_unix": time.time() + 600}
    if case == "bytes":
        path.write_text(path.read_text() + "\n")
    elif case == "interpreter":
        request["clean_python"] = "/other/bin/python"
    elif case == "missing":
        request.pop("esm1b_model_probe")
    if case == "passed":
        assert phase3._verify_esm1b_model_probe(request, transfer_view, "runpod_rtx4090")["status"] == "passed"
    else:
        with pytest.raises(phase3.Phase3Error):
            phase3._verify_esm1b_model_probe(request, transfer_view, "runpod_rtx4090")


def main_evidence():
    return {**model_evidence(), "format": "deepmzyme_ec_clean30_main_model_probe_v1",
            "synthetic_only": True, "model_family": "F4", "fixture_version": 1,
            "seed": phase3.MAIN_PROBE_SEED, "optimizer_steps": 1,
            "checkpoint_sha256": "a" * 64, "cpu_predictions_sha256": "b" * 64,
            "gpu_predictions_sha256": "c" * 64, "pocket_count": 2, "protein_count": 2, "classes": 7,
            "logit_atol": 1e-6, "logit_rtol": 0.0, "max_abs_difference": 0.0,
            "near_tie_class_changes": 0, "cpu_gpu_replay_passed": True,
            "phase3_sha256": _sha(Path(phase3.__file__))}


def test_main_qualification_checkpoint_reconstructs_production_f4_on_cpu(tmp_path, monkeypatch):
    """Exercise real PyG, fusion, optimization, checkpoint reload, and row replay without CUDA."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    import torch
    from training import ec_prediction_export as export

    path, config, pockets = phase3._main_model_checkpoint(tmp_path / "synthetic")
    assert len(pockets) == 2 and config["model_architecture"] == "gvp"
    assert config["use_esm_branch"] and config["use_global_esm_branch"]
    identity = {"run_id": "runtime_qualification", "mode": "synthetic_qualification", "fold_index": "0",
                "seed": str(phase3.MAIN_PROBE_SEED)}
    rows = []
    for _ in range(2):
        model, _ = export.rebuild_model_from_checkpoint(path, device="cpu")
        assert all(parameter.device.type == "cpu" for parameter in model.parameters())
        prediction = export._predict(model, export._prediction_loader(pockets, config, None), pockets, "cpu")
        assert prediction["ec_logits"].shape == (2, 7)
        assert torch.isfinite(prediction["ec_logits"]).all()
        pocket_rows, protein_rows, conflicts = export.build_prediction_rows(pockets, prediction["ec_logits"], identity)
        assert conflicts == 0
        rows.append((pocket_rows, protein_rows))
    for index, keys in ((0, ("pocket_id",)), (1, ("ec_group_key",))):
        difference, problems, _ = export.compare_prediction_rows(rows[0][index], rows[1][index],
            key_columns=keys, logit_atol=phase3.MAIN_REPLAY_ATOL)
        assert difference == 0.0 and problems == []


@pytest.mark.parametrize("key,value", [("model_family", "F1"), ("synthetic_only", False),
    ("optimizer_steps", 2), ("classes", 6), ("checkpoint_sha256", "unbound"),
    ("phase3_sha256", "0" * 64), ("cpu_gpu_replay_passed", False),
    ("logit_atol", 1e-5), ("logit_rtol", 1e-6), ("max_abs_difference", 2e-6),
    ("max_abs_difference", float("nan")), ("max_abs_difference", True), ("tf32_override", "1")])
def test_main_qualification_refuses_drift_or_relaxed_replay(key, value):
    evidence = main_evidence()
    assert phase3.validate_main_model_evidence(evidence, "runpod_rtx4090")["status"] == "passed"
    evidence[key] = value
    with pytest.raises(phase3.Phase3Error):
        phase3.validate_main_model_evidence(evidence, "runpod_rtx4090")


@pytest.mark.parametrize("case", ["passed", "bytes", "host", "boot", "older_allocation", "interpreter", "missing", "source"])
def test_main_qualification_binds_current_source_host_boot_allocation_and_interpreter(transfer_view, tmp_path, case):
    evidence = main_evidence()
    if case == "host":
        evidence["hostname"] = "another-host"
    elif case == "boot":
        evidence["boot_id"] = "00000000-0000-0000-0000-000000000000"
    elif case == "older_allocation":
        evidence["created_unix"] -= 3600
    elif case == "source":
        evidence["phase3_sha256"] = "0" * 64
    path = tmp_path / "main_model.json"
    _json_write(path, evidence)
    request = {"main_model_probe": str(path), "main_model_probe_sha256": _sha(path), "python": sys.executable,
               "allocation_started_unix": time.time() - 1, "deadline_unix": time.time() + 600}
    if case == "bytes":
        path.write_text(path.read_text() + "\n")
    elif case == "interpreter":
        request["python"] = "/other/bin/python"
    elif case == "missing":
        request.pop("main_model_probe")
    if case == "passed":
        assert phase3._verify_main_model_probe(request, transfer_view, "runpod_rtx4090")["status"] == "passed"
    else:
        with pytest.raises(phase3.Phase3Error):
            phase3._verify_main_model_probe(request, transfer_view, "runpod_rtx4090")
