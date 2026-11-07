"""Exercise the real guarded child adapters with a synthetic torch package, never a GPU."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ec_clean30 import runtime_precision as precision

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_clean_split30_comparator as clean


def policy(device="cuda"):
    return {"device": device, "nvidia_tf32_override": "0", "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False, "torch_version": "synthetic", "cuda_device_count": int(device == "cuda"),
            "cuda_device_name": "Synthetic GPU" if device == "cuda" else None,
            "helper_sha256": hashlib.sha256(Path(precision.__file__).read_bytes()).hexdigest()}


@pytest.fixture
def fake_torch(monkeypatch):
    backend = SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                              cudnn=SimpleNamespace(allow_tf32=True))
    cuda = SimpleNamespace(device_count=lambda: 1, is_available=lambda: True,
                           get_device_name=lambda index: "Synthetic GPU")
    module = SimpleNamespace(backends=backend, cuda=cuda, __version__="synthetic")
    monkeypatch.setitem(sys.modules, "torch", module)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "1")
    return module


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_child_policy_sets_and_records_both_actual_backend_flags(fake_torch, device, capsys):
    assert precision.configure_and_record(device) == policy(device)
    assert os.environ["NVIDIA_TF32_OVERRIDE"] == "0"
    assert fake_torch.backends.cuda.matmul.allow_tf32 is False
    assert fake_torch.backends.cudnn.allow_tf32 is False
    assert json.loads(capsys.readouterr().out.removeprefix(precision.POLICY_PREFIX)) == policy(device)


@pytest.mark.parametrize("count,available", [(0, False), (1, False), (2, True)])
def test_cuda_child_refuses_fallback_or_multiple_devices(fake_torch, monkeypatch, count, available):
    monkeypatch.setattr(fake_torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(fake_torch.cuda, "is_available", lambda: available)
    with pytest.raises(RuntimeError, match="exactly one available GPU"):
        precision.configure_and_record("cuda")


@pytest.mark.parametrize("visibility", ["", "  "])
def test_cuda_visibility_suppression_is_never_restored(fake_torch, monkeypatch, visibility):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visibility)
    with pytest.raises(RuntimeError, match="explicitly disabled"):
        precision.configure_and_record("cuda")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == visibility


@pytest.mark.parametrize("change", [
    {"device": "cpu"}, {"nvidia_tf32_override": "1"}, {"cuda_matmul_allow_tf32": True},
    {"cudnn_allow_tf32": True}, {"cuda_device_count": 0}, {"cuda_device_count": True},
    {"cuda_device_name": None}, {"helper_sha256": "a" * 64}, {"torch_version": ""},
])
def test_policy_log_rejects_wrong_observed_state(tmp_path, change):
    log = tmp_path / "child.log"
    log.write_text(precision.POLICY_PREFIX + json.dumps({**policy(), **change}) + "\n")
    with pytest.raises(RuntimeError, match="requested device and TF32-off"):
        precision.read_policy_log(log, device="cuda")


@pytest.mark.parametrize("content", ["no record\n", "not json", "[]", "duplicate"])
def test_policy_log_requires_one_valid_object(tmp_path, content):
    log = tmp_path / "child.log"
    if content == "duplicate":
        content = (precision.POLICY_PREFIX + json.dumps(policy()) + "\n") * 2
    elif content != "no record\n":
        content = precision.POLICY_PREFIX + content + "\n"
    log.write_text(content)
    with pytest.raises(RuntimeError):
        precision.read_policy_log(log, device="cuda")


def write_fake_torch(directory):
    package = directory / "torch"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "import os\nfrom types import SimpleNamespace as S\n"
        "assert os.environ['NVIDIA_TF32_OVERRIDE'] == '0'\n"
        "__version__ = 'synthetic'\n"
        "backends = S(cuda=S(matmul=S(allow_tf32=True)), cudnn=S(allow_tf32=True))\n"
        "cuda = S(is_available=lambda: True, device_count=lambda: 1, get_device_name=lambda i: 'Synthetic GPU')\n")
    (package / "hub.py").write_text("")


def test_clean_guarded_child_enforces_policy_before_original_script(tmp_path):
    clone = tmp_path / "clone"
    write_fake_torch(clone / "app/src")
    forbidden = tmp_path / "held_out"
    forbidden.mkdir()
    sentinel = forbidden / "sentinel"
    sentinel.write_text("forbidden")
    script = tmp_path / "train_fixture.py"
    script.write_text(
        "import torch\nfrom pathlib import Path\n"
        "assert torch.backends.cuda.matmul.allow_tf32 is False\n"
        "assert torch.backends.cudnn.allow_tf32 is False\n"
        f"try:\n    Path({str(sentinel)!r}).read_text()\n"
        "except PermissionError:\n    pass\nelse:\n    raise AssertionError('guard missing')\n"
        "print('==> device used: cuda:0 | dtype used: torch.float32')\n")
    env = clean.build_clean_env(clean_clone=clone, hash_seed=42, threads=1, forbidden_roots=[forbidden],
        base_env={**os.environ, "NVIDIA_TF32_OVERRIDE": "1", "CUDA_VISIBLE_DEVICES": "0"}, device="cuda")
    log = tmp_path / "train.log"
    command = clean.guarded_command(Path(sys.executable), script, [], device="cuda")
    with log.open("w") as handle:
        result = subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT, timeout=30)
    assert result.returncode == 0, log.read_text()
    assert clean.cuda_step_evidence(log, training=True) == {
        "runtime_policy": policy(), "log_sha256": clean.sha256_file(log)}


def test_pinned_farm_adapter_preserves_source_and_repairs_filtered_child_env(tmp_path, monkeypatch):
    write_fake_torch(tmp_path / "packages")
    log = tmp_path / "extract.log"
    forbidden = tmp_path / "held_out"
    forbidden.mkdir()
    launch = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('guard', {str(clean.ACCESS_GUARD_FILE)!r})\n"
        "guard = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(guard)\n"
        "assert guard.install_guard_from_environment()\nimport torch.hub\n"
        "assert torch.backends.cuda.matmul.allow_tf32 is False\n"
        "assert torch.backends.cudnn.allow_tf32 is False\nprint('Transferred model to GPU')\n")
    harness = tmp_path / "synthetic_farm.py"
    harness.write_text(
        "import subprocess, sys\n"
        f"EXTRACT_LAUNCH = {launch!r}\n"
        "def extraction_env(runner, roots, device, threads):\n"
        f"    return {{'PYTHONPATH': {str(tmp_path / 'packages')!r}, 'CUDA_VISIBLE_DEVICES': '0', "
        f"{clean.FORBIDDEN_READ_ROOTS_ENV!r}: {str(forbidden)!r}}}\n"
        "def main(args):\n"
        "    env = extraction_env(None, [], args[args.index('--device')+1], 1)\n"
        f"    with open({str(log)!r}, 'w') as handle:\n"
        "        return subprocess.run([sys.executable, '-c', EXTRACT_LAUNCH], env=env, stdout=handle, "
        "stderr=subprocess.STDOUT, timeout=30).returncode\n")
    original = harness.read_bytes()
    monkeypatch.setattr(precision, "PRIVATE_FARM_SHA256", hashlib.sha256(original).hexdigest())
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "1")
    assert precision.run_private_farm(harness, ["--device", "cuda"]) == 0, log.read_text()
    assert precision.read_policy_log(log, device="cuda") == policy()
    assert harness.read_bytes() == original
    harness.write_bytes(original + b"# changed\n")
    with pytest.raises(RuntimeError, match="frozen source pin"):
        precision.run_private_farm(harness, ["--device", "cuda"])


@pytest.mark.parametrize("device,visibility", [("cpu", "0"), ("cuda", "0"), ("cuda", None)])
def test_comparator_environment_preserves_legacy_cpu_and_explicit_cuda(tmp_path, device, visibility):
    base = {"NVIDIA_TF32_OVERRIDE": "1"}
    if visibility is not None:
        base["CUDA_VISIBLE_DEVICES"] = visibility
    env = clean.build_clean_env(clean_clone=tmp_path, hash_seed=42, threads=1,
                               forbidden_roots=[tmp_path / "held_out"], base_env=base, device=device)
    if device == "cpu":
        assert env["CUDA_VISIBLE_DEVICES"] == ""
    else:
        assert env.get("CUDA_VISIBLE_DEVICES") == visibility
        assert env["NVIDIA_TF32_OVERRIDE"] == "0"


def test_comparator_refuses_suppressed_cuda_and_unsupported_export(tmp_path):
    with pytest.raises(clean.ComparatorError, match="explicitly disabled"):
        clean.build_clean_env(clean_clone=tmp_path, hash_seed=42, threads=1,
            forbidden_roots=[tmp_path / "held_out"], base_env={"CUDA_VISIBLE_DEVICES": ""}, device="cuda")
    with pytest.raises(clean.ComparatorError, match="only explicit CPU"):
        clean.run_export(work_dir=tmp_path, query_ids_file=tmp_path, query_esm_dir=tmp_path, out=tmp_path,
            mode="validation", clean_python=Path(sys.executable), clean_clone=tmp_path, export_script=tmp_path,
            device="cuda")
