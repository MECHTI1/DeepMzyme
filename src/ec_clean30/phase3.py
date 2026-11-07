"""Small transfer, measurement and persistence tools for the frozen CLEAN30 campaign.

This module owns no cloud lifecycle. Measurements have separate G2 identities and
cannot be selected as S1 fits. Existing views, command builders, replay and execution
hooks remain authoritative; no preparation or scientific policy is regenerated.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import shutil
import socket
import stat
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import time
from typing import Any, Mapping, Sequence

from ec_clean30 import commands, freeze, provenance, runtime_precision, views
from ec_clean30.campaign import CAMPAIGN_ID, FAMILIES, KIND_INNER, default_smoke_recipe, recipe_with_lever
from benchmarking.pmm_execution import CampaignExecution, ExecutionPolicy, verify_host_pull

REPO_ROOT = Path(__file__).resolve().parents[2]
EXECUTION_FILES = ("src/ec_clean30/phase3.py", "scripts/run_ec_clean30_phase3.py")
TRANSFER_MANIFEST = "phase3_transfer.json"
RECEIPT_NAMES = (views.DRY_RUN_RECEIPT_NAME, views.VERIFY_RECEIPT_NAME,
                 views.FILE_MODES_RECEIPT_NAME, views.FILE_MODES_APPLIED_RECEIPT_NAME)
TIMING_ROWS = 1000
FREEZE1B_SHA256 = "fc6f032805f084d7aef4f704a42149e92fb7894af3699b2ad5c12e86c87bcbdb"
OFFICIAL_FOLD0_SHA256 = "ab8a8a21219aec19ba03dd7205dec8d8c1a156ff5cfad23aad344c7d7fbdd9c1"
EXECUTION_FORMAT = "deepmzyme_ec_clean30_execution_v1"
AUTOMATION_FORMAT = "deepmzyme_ec_clean30_execution_v2"
AUTOMATION_FILES = (*EXECUTION_FILES, "src/ec_clean30/phase4.py", "scripts/run_ec_clean30_phase4.py",
                    "src/ec_clean30/workstation.py", "scripts/run_ec_clean30_workstation.py")
RUNPOD_FORMAT = "deepmzyme_ec_clean30_execution_v3"
RUNPOD_FILES = (*AUTOMATION_FILES, "src/ec_clean30/runpod_controller.py", "src/ec_clean30/container_caps.py")
BOOTSTRAP_FORMAT = "deepmzyme_ec_clean30_execution_v4"
BOOTSTRAP_FILES = (*RUNPOD_FILES, "src/ec_clean30/runpod_bootstrap.py")
EXECUTION_V5_FORMAT = "deepmzyme_ec_clean30_execution_v5"
TRANSFER_FORMAT = "deepmzyme_ec_clean30_fold0_transfer_v1"
FOLD_TRANSFER_FORMAT = "deepmzyme_ec_clean30_fold_transfer_v2"
RUNTIME_LOCKS_PATH = "docs/campaigns/ec_class_vs_clean30_2026-10/execution_v3/runtime_locks_v2.json"
PRETRAIN_ATTACHMENT_MARKER = "pretrain_attachment.json"
PRETRAIN_ATTACHMENT_FORMAT = "deepmzyme_ec_clean30_pretrain_attachment_snapshot_v1"
GPU_PROFILES = {"gcp_l4": "NVIDIA L4", "runpod_l4": "NVIDIA L4",
                "runpod_rtx4090": "NVIDIA GeForce RTX 4090"}
GPU_MEMORY_MIN_BYTES = 22 * 1024 ** 3
GPU_MEMORY_MAX_BYTES = 25 * 1024 ** 3
ESM1B_WEIGHTS_SHA256 = "0569754efaff7dcb7e068c27367bc73f10afb4b450ea30aac30d9bc60783a8b1"
ESM1B_CONTACT_SHA256 = "77193a8814f0db0b36a03aebb1a311adc6b4745f463c04839defc15407bbb28a"
# Synthetic qualification retains its separate tolerances; fit replay follows Freeze-1 G1.
ESM1B_REPLAY_ATOL = 1.0e-4
ESM1B_REPLAY_RTOL = 1.0e-4
MAIN_REPLAY_ATOL = 1.0e-6
MAIN_PROBE_SEED = 42017


class Phase3Error(ValueError):
    """The frozen inputs or runtime evidence do not permit the requested action."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise Phase3Error(message)


def hardware_profile(name: str = "gcp_l4") -> dict[str, Any]:
    """Resolve only the reviewed nominal-24-GB, single-device provider profiles."""
    _require(isinstance(name, str) and name in GPU_PROFILES, "Unknown approved GPU hardware profile.")
    return {"name": name, "device_name": GPU_PROFILES[name], "device_count": 1,
            "minimum_memory_bytes": GPU_MEMORY_MIN_BYTES, "maximum_memory_bytes": GPU_MEMORY_MAX_BYTES,
            "runtime_environment": {"NVIDIA_TF32_OVERRIDE": "0"} if name.startswith("runpod_") else {}}


def cuda_probe_program(profile: str = "gcp_l4") -> str:
    """Run in the selected interpreter so a working host driver alone cannot pass."""
    expected = hardware_profile(profile)
    return (
        "import json,os,torch; "
        f"expected={expected!r}; "
        "assert all(os.environ.get(key)==value for key,value in expected['runtime_environment'].items()), 'Runtime precision environment differs'; "
        "assert torch.version.cuda and torch.cuda.is_available(), 'Interpreter has no usable CUDA'; "
        "count=torch.cuda.device_count(); assert count==expected['device_count'], 'Expected one visible GPU'; "
        "name=torch.cuda.get_device_name(0); assert name==expected['device_name'], 'GPU profile differs'; "
        "memory=int(torch.cuda.get_device_properties(0).total_memory); "
        "assert expected['minimum_memory_bytes']<=memory<=expected['maximum_memory_bytes'], 'Expected nominal 24-GB GPU'; "
        "value=(torch.ones(4,device='cuda')*2).sum().item(); torch.cuda.synchronize(); "
        "assert value==8.0, 'CUDA kernel check failed'; "
        "print(json.dumps({'hardware_profile':expected['name'],'device_name':name,'device_count':count,"
        "'total_memory_bytes':memory,'torch':torch.__version__,'cuda':torch.version.cuda,'kernel_value':value,"
        "'tf32_override':os.environ.get('NVIDIA_TF32_OVERRIDE')}))"
    )


def validate_cuda_evidence(evidence: Mapping[str, Any], profile: str = "gcp_l4") -> dict[str, Any]:
    expected = hardware_profile(profile)
    memory = evidence.get("total_memory_bytes")
    _require(evidence.get("hardware_profile") == profile
             and evidence.get("device_name") == expected["device_name"]
             and type(evidence.get("device_count")) is int and evidence["device_count"] == 1
             and type(memory) is int and expected["minimum_memory_bytes"] <= memory <= expected["maximum_memory_bytes"]
             and bool(evidence.get("torch")) and bool(evidence.get("cuda")) and evidence.get("kernel_value") == 8.0,
             "CUDA preflight did not prove the approved device, memory class and working interpreter.")
    if profile.startswith("runpod_"):
        _require(evidence.get("tf32_override") == "0", "Runpod CUDA preflight requires its fixed full-FP32 environment.")
    return dict(evidence)


def esm1b_model_probe_program(profile: str, weights: str, weights_sha256: str,
                              contact_weights_sha256: str) -> str:
    """Qualify the complete pinned local ESM-1b model in the separate CLEAN interpreter."""
    hardware_profile(profile)
    _require(Path(weights).is_absolute() and ".." not in Path(weights).parts,
             "The model probe requires an absolute local weights path.")
    _require(weights_sha256 == ESM1B_WEIGHTS_SHA256 and contact_weights_sha256 == ESM1B_CONTACT_SHA256,
             "The model probe must use the existing pinned ESM-1b weights and companion.")
    return textwrap.dedent(f"""\
        import contextlib,hashlib,io,json,pathlib,socket,sys,time,torch
        from esm import pretrained
        captured=io.StringIO()
        with contextlib.redirect_stdout(captured):
            exec({cuda_probe_program(profile)!r}, {{}})
        evidence=json.loads(captured.getvalue())
        weights=pathlib.Path({weights!r})
        contact=pathlib.Path(str(weights.with_suffix(''))+'-contact-regression.pt')
        def digest(path):
            assert path.is_file() and not path.is_symlink(), 'Expected a regular local weight file'
            value=hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda:handle.read(8*1024*1024),b''):
                    value.update(chunk)
            return value.hexdigest()
        assert digest(weights)=={weights_sha256!r}, 'ESM-1b weight hash differs'
        assert digest(contact)=={contact_weights_sha256!r}, 'ESM-1b companion hash differs'
        with contextlib.redirect_stdout(io.StringIO()):
            model,alphabet=pretrained.load_model_and_alphabet_local(str(weights))
            model.eval()
            assert model.num_layers==33, 'Expected the complete ESM-1b model'
            _,_,tokens=alphabet.get_batch_converter()([('runtime_qualification','MKT')])
            with torch.no_grad():
                cpu_representation=model(tokens,repr_layers=[33],return_contacts=False)['representations'][33]
            assert not cpu_representation.is_cuda and torch.isfinite(cpu_representation).all().item(), 'ESM-1b CPU forward failed'
            model=model.cuda()
            parameters=list(model.parameters())
            assert parameters and all(value.is_cuda for value in parameters), 'Full model did not transfer to CUDA'
            with torch.no_grad():
                result=model(tokens.cuda(),repr_layers=[33],return_contacts=False)
            representation=result['representations'][33]
            assert representation.is_cuda and torch.isfinite(representation).all().item(), 'ESM-1b CUDA forward failed'
            torch.cuda.synchronize()
            gpu_representation=representation.cpu()
            assert list(cpu_representation.shape)==list(gpu_representation.shape)==[1,5,1280], 'ESM-1b representation shape differs'
            difference=(cpu_representation-gpu_representation).abs()
            max_difference=float(difference.max().item())
            max_reference=float(cpu_representation.abs().max().item())
            max_scaled_error=float((difference/({ESM1B_REPLAY_ATOL!r}+{ESM1B_REPLAY_RTOL!r}*cpu_representation.abs())).max().item())
            replay_passed=bool(torch.allclose(gpu_representation,cpu_representation,atol={ESM1B_REPLAY_ATOL!r},rtol={ESM1B_REPLAY_RTOL!r}))
            assert replay_passed, 'ESM-1b CPU/GPU replay differs at the fixed feature tolerance'
            replay={{'sequence_sha256':hashlib.sha256(b'MKT').hexdigest(),'representation_layer':33,
                    'representation_shape':[1,5,1280],'atol':{ESM1B_REPLAY_ATOL!r},'rtol':{ESM1B_REPLAY_RTOL!r},
                    'max_abs_difference':max_difference,'max_reference_abs':max_reference,
                    'max_scaled_error':max_scaled_error,'passed':replay_passed,
                    'cpu_finite':True,'gpu_finite':True}}
        evidence.update(format='deepmzyme_ec_clean30_esm1b_model_probe_v1',status='passed',
                        weights_sha256={weights_sha256!r},contact_weights_sha256={contact_weights_sha256!r},
                        model_on_cuda=True,model_layers=model.num_layers,
                        model_parameter_count=sum(value.numel() for value in parameters),
                        synthetic_forward_finite=True,cpu_gpu_replay=replay,hostname=socket.gethostname(),
                        boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                        python_executable=sys.executable,created_unix=time.time())
        print(json.dumps(evidence,sort_keys=True))
    """)


def validate_esm1b_model_evidence(evidence: Mapping[str, Any], profile: str) -> dict[str, Any]:
    validate_cuda_evidence(evidence, profile)
    count = evidence.get("model_parameter_count")
    replay = evidence.get("cpu_gpu_replay", {})
    _require(isinstance(replay, dict), "ESM-1b CPU/GPU replay evidence is missing.")
    difference = replay.get("max_abs_difference")
    reference, scaled = replay.get("max_reference_abs"), replay.get("max_scaled_error")
    _require(replay.get("sequence_sha256") == hashlib.sha256(b"MKT").hexdigest()
             and replay.get("representation_layer") == 33 and replay.get("representation_shape") == [1, 5, 1280]
             and type(replay.get("atol")) in (int, float) and replay["atol"] == ESM1B_REPLAY_ATOL
             and type(replay.get("rtol")) in (int, float) and replay["rtol"] == ESM1B_REPLAY_RTOL
             and type(reference) in (int, float) and math.isfinite(reference) and reference >= 0
             and type(scaled) in (int, float) and math.isfinite(scaled) and 0 <= scaled <= 1
             and type(difference) in (int, float) and math.isfinite(difference)
             and 0 <= difference <= ESM1B_REPLAY_ATOL + ESM1B_REPLAY_RTOL * reference
             and replay.get("passed") is True and replay.get("cpu_finite") is True and replay.get("gpu_finite") is True,
             "ESM-1b CPU/GPU replay must pass the fixed feature tolerance with finite representations.")
    _require(evidence.get("format") == "deepmzyme_ec_clean30_esm1b_model_probe_v1"
             and evidence.get("status") == "passed" and evidence.get("weights_sha256") == ESM1B_WEIGHTS_SHA256
             and evidence.get("contact_weights_sha256") == ESM1B_CONTACT_SHA256
             and evidence.get("model_on_cuda") is True and evidence.get("model_layers") == 33
             and type(count) is int and 600_000_000 < count < 700_000_000
             and evidence.get("synthetic_forward_finite") is True and bool(evidence.get("hostname"))
             and re.fullmatch(r"[0-9a-f-]{36}", str(evidence.get("boot_id", "")))
             and Path(str(evidence.get("python_executable", ""))).is_absolute()
             and type(evidence.get("created_unix")) in (int, float)
             and math.isfinite(evidence["created_unix"]) and evidence["created_unix"] > 0,
             "ESM-1b qualification requires the complete pinned model on CUDA and a finite forward pass.")
    return dict(evidence)


def main_model_probe_program(profile: str, output_directory: str) -> str:
    """Invoke the authenticated production-model probe from the clean repository root."""
    hardware_profile(profile)
    path = Path(output_directory)
    _require(path.is_absolute() and ".." not in path.parts, "An absolute main-model probe directory is required.")
    return ("import json,pathlib,sys; sys.path.insert(0,str(pathlib.Path.cwd()/'src')); "
            "from ec_clean30 import phase3; "
            f"print(json.dumps(phase3.main_model_probe({profile!r},pathlib.Path({str(path)!r})),sort_keys=True))")


def _main_model_fixture():
    """Two deterministic, synthetic pockets; never consult campaign or held-out inputs."""
    from dataclasses import asdict
    import torch
    from data_structures import PocketRecord, ResidueRecord
    from training.config import TrainConfig

    config = asdict(TrainConfig(task="ec", ec_class_vocabulary="fixed_seven", model_architecture="gvp",
        esm_dim=8, hidden_s=16, hidden_v=4, edge_hidden=8, gvp_layers=1, esm_fusion_dim=16,
        use_esm_branch=True, fusion_mode="late_fusion", use_global_esm_branch=True, global_esm_dim=8,
        head_mlp_dropout=0.0, esm_graph_encoder_dropout=0.0, global_esm_dropout=0.0, batch_size=2,
        shell_role_source="geometry"))
    pockets = []
    for index in range(2):
        residues = []
        for number in range(1, 4):
            residues.append(ResidueRecord(chain_id="A", resseq=number, icode="", resname="ASP",
                atoms={name: torch.tensor([float(number + index), y, z]) for name, y, z in
                       (("N", .3, 0.), ("CA", 0., 0.), ("C", -.3, 0.), ("O", -.5, 0.),
                        ("CB", 0., .8), ("CG", 0., .3), ("OD1", 0., 0.), ("OD2", .4, 0.))},
                esm_embedding=torch.arange(8, dtype=torch.float32) / 8 + number / 10,
                has_esm_embedding=True))
        structure = f"qualification{index}__chain_A__EC_{index + 1}.1.1.1"
        pockets.append(PocketRecord(structure_id=structure, pocket_id=f"qualification_pocket_{index}",
            metal_element="ZN", metal_coords=[torch.zeros(3)], residues=residues, y_ec=index,
            metadata={"ec_group_key": structure, "ec_group_id": index,
                      "global_esm_vector": torch.arange(8, dtype=torch.float32) / 8 + index / 10}))
    return config, pockets


def _main_model_checkpoint(output_directory: Path) -> tuple[Path, dict[str, Any], list[Any]]:
    """Exercise the real graph/factory/optimizer/checkpoint path using synthetic data only."""
    import torch
    from torch_geometric.data import Batch
    from graph.construction import pocket_to_pyg_data
    from label_schemes import METAL_TARGET_LABELS, configure_active_metal_label_scheme
    from model_variants import build_pocket_classifier
    from training import ec_prediction_export as export

    torch.manual_seed(MAIN_PROBE_SEED)
    torch.set_num_threads(2)
    config, pockets = _main_model_fixture()
    configure_active_metal_label_scheme(config["metal_label_scheme"])
    labels = {index: str(index + 1) for index in range(7)}
    checkpoint = {"config": config, "ec_labels": labels, "metal_labels": dict(METAL_TARGET_LABELS),
                  "model_state_dict": {}, "checkpoint_rule": "terminal"}
    kwargs = export.factory_kwargs(REPO_ROOT / "src/training/run.py", config, checkpoint, METAL_TARGET_LABELS, labels)
    model = build_pocket_classifier(**kwargs)
    graphs = [pocket_to_pyg_data(pocket, **export.graph_options(config)) for pocket in pockets]
    batch = Batch.from_data_list(graphs)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    loss = model(batch)["loss"]
    _require(bool(torch.isfinite(loss)), "Synthetic qualification optimizer loss is not finite.")
    loss.backward()
    optimizer.step()
    checkpoint["model_state_dict"] = model.state_dict()
    output_directory.mkdir(parents=True, exist_ok=False)
    path = output_directory / "synthetic_checkpoint.pt"
    torch.save(checkpoint, path)
    return path, config, pockets


def main_model_probe(profile: str, output_directory: Path) -> dict[str, Any]:
    """Reconstruct a small production F4 model independently on CPU and CUDA."""
    import contextlib
    import torch
    from training import ec_prediction_export as export

    _require(Path(output_directory).is_absolute(), "An absolute main-model probe directory is required.")
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        exec(cuda_probe_program(profile), {})
    evidence = json.loads(captured.getvalue())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with contextlib.redirect_stdout(io.StringIO()):
        path, config, pockets = _main_model_checkpoint(Path(output_directory))
        exported = {}
        identity = {"run_id": "runtime_qualification", "mode": "synthetic_qualification", "fold_index": "0",
                    "seed": str(MAIN_PROBE_SEED)}
        for device in ("cpu", "cuda"):
            model, _ = export.rebuild_model_from_checkpoint(path, device=device)
            _require(all(parameter.device.type == device for parameter in model.parameters()),
                     "Reconstructed qualification model is on the wrong device.")
            loader = export.prediction_loader(pockets, config, None)
            prediction = export.predict(model, loader, pockets, device)
            pocket_rows, protein_rows, conflicts = export.build_prediction_rows(pockets, prediction["ec_logits"], identity)
            _require(conflicts == 0, "Synthetic qualification identities conflict.")
            exported[device] = {"pockets": pocket_rows, "proteins": protein_rows}
            _write(Path(output_directory) / f"{device}_predictions.json", exported[device])
            del model, loader, prediction
        torch.cuda.synchronize()
        differences, ties = [], 0
        for kind, keys in (("pockets", ("pocket_id",)), ("proteins", ("ec_group_key",))):
            difference, problems, near_ties = export.compare_prediction_rows(exported["cpu"][kind],
                exported["cuda"][kind], key_columns=keys, logit_atol=MAIN_REPLAY_ATOL)
            _require(not problems, "Main-model CPU/GPU replay failed: " + "; ".join(problems))
            differences.append(difference)
            ties += near_ties
    evidence.update(format="deepmzyme_ec_clean30_main_model_probe_v1", status="passed", synthetic_only=True,
        model_family="F4", fixture_version=1, seed=MAIN_PROBE_SEED, optimizer_steps=1,
        checkpoint_sha256=_sha(path), cpu_predictions_sha256=_sha(Path(output_directory) / "cpu_predictions.json"),
        gpu_predictions_sha256=_sha(Path(output_directory) / "cuda_predictions.json"),
        pocket_count=2, protein_count=2, classes=7, logit_atol=MAIN_REPLAY_ATOL, logit_rtol=0.0,
        max_abs_difference=max(differences), near_tie_class_changes=ties, cpu_gpu_replay_passed=True,
        phase3_sha256=_sha(Path(__file__)), hostname=socket.gethostname(),
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        python_executable=sys.executable, created_unix=time.time())
    validate_main_model_evidence(evidence, profile)
    return evidence


def validate_main_model_evidence(evidence: Mapping[str, Any], profile: str) -> dict[str, Any]:
    validate_cuda_evidence(evidence, profile)
    difference = evidence.get("max_abs_difference")
    _require(evidence.get("format") == "deepmzyme_ec_clean30_main_model_probe_v1"
             and evidence.get("status") == "passed" and evidence.get("synthetic_only") is True
             and evidence.get("model_family") == "F4" and evidence.get("fixture_version") == 1
             and evidence.get("seed") == MAIN_PROBE_SEED and evidence.get("optimizer_steps") == 1
             and evidence.get("pocket_count") == evidence.get("protein_count") == 2 and evidence.get("classes") == 7
             and evidence.get("logit_atol") == MAIN_REPLAY_ATOL and evidence.get("logit_rtol") == 0.0
             and type(difference) in (int, float) and math.isfinite(difference) and 0 <= difference <= MAIN_REPLAY_ATOL
             and type(evidence.get("near_tie_class_changes")) is int and evidence["near_tie_class_changes"] >= 0
             and evidence.get("cpu_gpu_replay_passed") is True
             and all(re.fullmatch(r"[0-9a-f]{64}", str(evidence.get(key, "")))
                     for key in ("checkpoint_sha256", "cpu_predictions_sha256", "gpu_predictions_sha256"))
             and evidence.get("phase3_sha256") == _sha(Path(__file__)) and bool(evidence.get("hostname"))
             and re.fullmatch(r"[0-9a-f-]{36}", str(evidence.get("boot_id", "")))
             and Path(str(evidence.get("python_executable", ""))).is_absolute()
             and type(evidence.get("created_unix")) in (int, float) and math.isfinite(evidence["created_unix"])
             and evidence["created_unix"] > 0, "Main-model qualification requires the fixed synthetic CPU/GPU checkpoint replay.")
    return dict(evidence)


def _sha(path: Path) -> str:
    return provenance.sha256_file(Path(path))


def _json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def _bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _write(path: Path, value: Any) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_bytes(value))
    return _sha(path)


def _relative(name: str) -> str:
    path = PurePosixPath(name)
    _require(bool(name) and not path.is_absolute() and ".." not in path.parts
             and "." not in path.parts and "\\" not in name and str(path) == name,
             "Unsafe relative member name.")
    return name


def _safe_file(root: Path, name: str) -> Path:
    name = _relative(name)
    path = root / name
    current = root
    for part in PurePosixPath(name).parts:
        current /= part
        _require(not current.is_symlink(), "Input contains a symlink.")
    _require(path.is_file(), f"Required regular file is absent: {name}")
    return path


def _no_restricted_path(path: Path) -> None:
    for form in (Path(path).absolute(), Path(path).resolve()):
        parts = {part.lower() for part in form.parts}
        _require(not parts.intersection({"identity", "test", "held-out", "held_out"})
                 and not any(part.startswith("identity_after_") or "_test_split_" in part for part in parts),
                 "Restricted identity/test paths cannot be used by Phase 3.")


def _outside_guard(path: Path, roots: Sequence[str]) -> None:
    _no_restricted_path(path)
    for form in (Path(path).absolute(), Path(path).resolve()):
        for root in roots:
            for forbidden in (Path(root).absolute(), Path(root).resolve()):
                _require(form != forbidden and forbidden not in form.parents,
                         "Timing input, output or weights lie within a frozen forbidden-read root.")


def _transfer_names(root: Path, manifest: Mapping[str, Any]) -> list[str]:
    names = {views.VIEW_MANIFEST_NAME}
    names.update(_relative(row["path"]) for row in manifest["files"])
    _require(not any(PurePosixPath(name).parts[0] in views.WRITABLE_DIRS for name in names),
             "The immutable view manifest lists a writable/cache location.")
    for directory in views.CHAIN_MEAN_DIRS.values():
        cache = _json(_safe_file(root, f"{directory}/manifest.json"))
        names.add(f"{directory}/manifest.json")
        for row in cache["files"]:
            _require(PurePosixPath(row["file"]).name == row["file"], "Unsafe chain-cache file.")
            names.add(f"{directory}/{_relative(row['file'])}")
    for name in RECEIPT_NAMES:
        relative = f"{views.RECEIPTS_DIR}/{name}"
        if (root / relative).exists():
            names.add(relative)
    _require(f"{views.RECEIPTS_DIR}/{views.DRY_RUN_RECEIPT_NAME}" in names,
             "Transfer requires the original completed dry-run receipt.")
    return sorted(names)


def _member_mode(name: str) -> int:
    return 0o644 if name.startswith(views.RECEIPTS_DIR + "/") else 0o444


def pack_view(view_root: Path, archive_path: Path, *, smoke: bool = False, fold: int = 0) -> dict[str, Any]:
    """Write the exact declared fold closure; attached F6 inputs and stale probes never travel."""
    _no_restricted_path(view_root)
    view = views.View.load(view_root)
    _require(type(fold) is int and fold in range(5) and view.fold == fold and view.smoke == smoke,
             "Only the declared fold view may travel.")
    views.verify_view(view.root, require_chain_means=("final", "block26"), write_receipt=False)
    receipt = views.read_dry_run_receipt(view)
    _require(receipt is not None and receipt.get("status") == "passed"
             and receipt.get("view_manifest_sha256") == view.manifest_sha256,
             "The original dry-run receipt does not certify this view.")
    names = _transfer_names(view.root, view.manifest)
    files = [{"path": name, "sha256": _sha(_safe_file(view.root, name)),
              "size_bytes": (view.root / name).stat().st_size, "mode": _member_mode(name)} for name in names]
    manifest = {"format": TRANSFER_FORMAT if fold == 0 else FOLD_TRANSFER_FORMAT,
                "campaign_id": CAMPAIGN_ID, "fold": fold, "smoke": smoke,
                "view_manifest_sha256": view.manifest_sha256, "files": files,
                "original_dry_run_sha256": _sha(view.receipts / views.DRY_RUN_RECEIPT_NAME)}
    archive_path = Path(archive_path)
    _no_restricted_path(archive_path)
    _require(not archive_path.exists(), "Transfer archive already exists.")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    partial = archive_path.with_name(archive_path.name + ".partial")
    _require(not partial.exists(), "A previous partial transfer archive needs reconciliation.")
    try:
        with tarfile.open(partial, "w", format=tarfile.PAX_FORMAT) as archive:
            payload = _bytes(manifest)
            info = tarfile.TarInfo(TRANSFER_MANIFEST)
            info.size, info.mode, info.mtime = len(payload), 0o444, 0
            archive.addfile(info, io.BytesIO(payload))
            for row in files:
                path = _safe_file(view.root, row["path"])
                info = tarfile.TarInfo(row["path"])
                info.size, info.mode, info.mtime = row["size_bytes"], row["mode"], 0
                with path.open("rb") as handle:
                    archive.addfile(info, handle)
                _require(_sha(path) == row["sha256"], "View changed while it was being archived.")
        partial.rename(archive_path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return {"status": "passed", "archive_sha256": _sha(archive_path), "files": len(files),
            "archive_bytes": archive_path.stat().st_size, "unpacked_bytes": sum(row["size_bytes"] for row in files),
            "view_manifest_sha256": view.manifest_sha256,
            "original_dry_run_sha256": manifest["original_dry_run_sha256"]}


def unpack_view(archive_path: Path, destination: Path, *, expected_sha256: str,
                smoke: bool = False, fold: int = 0) -> dict[str, Any]:
    """Validate every tar member before extracting into an isolated staging view."""
    archive_path, destination = Path(archive_path), Path(destination)
    _no_restricted_path(archive_path)
    _no_restricted_path(destination)
    _require(_sha(archive_path) == expected_sha256, "Transfer archive checksum differs.")
    _require(not destination.exists(), "Transfer destination must not exist.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:*") as archive:
        members = archive.getmembers()
        names = [_relative(member.name) for member in members]
        _require(len(names) == len(set(names)), "Duplicate archive member.")
        _require(all(member.isreg() and not member.linkname for member in members),
                 "Only regular files are accepted; links and special members are forbidden.")
        _require(TRANSFER_MANIFEST in names, "Transfer manifest is missing.")
        descriptor = archive.getmember(TRANSFER_MANIFEST)
        _require(descriptor.size <= 16 * 1024 * 1024 and descriptor.mode == 0o444,
                 "Invalid transfer manifest size or permissions.")
        manifest = json.load(archive.extractfile(descriptor))
        _require(type(fold) is int and fold in range(5)
                 and manifest.get("format") == (TRANSFER_FORMAT if fold == 0 else FOLD_TRANSFER_FORMAT)
                 and manifest.get("campaign_id") == CAMPAIGN_ID
                 and manifest.get("fold") == fold and manifest.get("smoke") is smoke,
                 "Transfer is not the declared fold view.")
        records = manifest.get("files", [])
        expected = {row["path"]: row for row in records}
        _require(len(expected) == len(records) and set(names) == {TRANSFER_MANIFEST, *expected},
                 "Archive members differ from its exact transfer manifest.")
        for name, row in expected.items():
            _relative(name)
            member = archive.getmember(name)
            _require(row["mode"] == _member_mode(name) == member.mode
                     and row["size_bytes"] == member.size, "Archive size or permission differs.")
        with tempfile.TemporaryDirectory(prefix="ec30-unpack-", dir=destination.parent) as temporary:
            stage = Path(temporary) / CAMPAIGN_ID / ("smoke_views" if smoke else "views") / destination.name
            stage.mkdir(parents=True)
            for name, row in expected.items():
                path = stage / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(name) as source, path.open("xb") as output:
                    shutil.copyfileobj(source, output)
                _require(_sha(path) == row["sha256"], "Extracted member checksum differs.")
                path.chmod(row["mode"])
            embedded = _json(stage / views.VIEW_MANIFEST_NAME)
            _require(set(_transfer_names(stage, embedded)) == set(expected),
                     "Transfer includes files outside the view and cache manifests.")
            _require(_sha(stage / views.VIEW_MANIFEST_NAME) == manifest["view_manifest_sha256"],
                     "View manifest checksum differs.")
            for name in (*views.WRITABLE_DIRS, views.RING_DIR):
                (stage / name).mkdir(exist_ok=True)
            for name in getattr(views, "ATTACHED_DIRS", ()):
                attached = stage / name
                _require(not attached.exists(), "Attached inputs must not occur in the transferred view manifest.")
                attached.mkdir(mode=0o755)
            for name in views.CACHE_SUBDIRS:
                (stage / views.CACHE_DIR / name).mkdir(exist_ok=True)
            # Existing permission and verification hooks enforce the relocated view contract.
            try:
                modes = views.set_permissions(stage)
                _require(modes["outcome"] == views.FILE_MODES_ENFORCED,
                         "The extraction filesystem does not enforce required view permissions.")
                views.verify_view(stage, require_chain_means=("final", "block26"), write_receipt=False)
            except BaseException:
                views.remove_tree(stage)
                raise
            _require(_sha(stage / views.RECEIPTS_DIR / views.DRY_RUN_RECEIPT_NAME)
                     == manifest["original_dry_run_sha256"], "Original dry-run receipt changed.")
            # Moving a directory between parents updates its '..' entry on POSIX.
            stage.chmod(0o755)
            stage.rename(destination)
            destination.chmod(views.DATA_DIR_MODE)
    return {"status": "passed", "archive_sha256": expected_sha256,
            "view_manifest_sha256": manifest["view_manifest_sha256"], "files": len(expected),
            "original_dry_run_sha256": manifest["original_dry_run_sha256"]}


def timing_inputs(training_file: Path, out: Path, *, expected_sha256: str) -> dict[str, Any]:
    """The deterministic 1,000-row fold-0 sequence timing set; no other fold is read."""
    from scripts.run_clean_split30_comparator import SEQUENCE_MAP_COLUMNS, read_official_training_rows

    training_file, out = Path(training_file), Path(out)
    _no_restricted_path(training_file)
    _no_restricted_path(out)
    _require(_sha(training_file) == expected_sha256, "Official training input checksum differs.")
    rows, _ = read_official_training_rows(training_file, 0, [])
    _require(len(rows) >= TIMING_ROWS, "Timing input needs at least 1,000 official training rows.")
    chosen = sorted(rows, key=lambda row: (hashlib.sha256(row.sequence.encode()).hexdigest(), row.accession))[:TIMING_ROWS]
    _require(not out.exists(), "Timing input output already exists.")
    out.mkdir(parents=True)
    with (out / "timing.fasta").open("x") as fasta, (out / "sequence_map.tsv").open("x", newline="") as table:
        writer = csv.writer(table, delimiter="\t", lineterminator="\n")
        writer.writerow(SEQUENCE_MAP_COLUMNS)
        for row in chosen:
            fasta.write(f">{row.accession}\n{row.sequence}\n")
            writer.writerow((row.accession, hashlib.sha256(row.sequence.encode()).hexdigest(), len(row.sequence), 1, row.sequence))
    receipt = {"format": "deepmzyme_ec_clean30_timing_inputs_v1", "fold": 0, "n_rows": TIMING_ROWS,
               "source_name": training_file.name, "source_sha256": expected_sha256,
               "rule": "ascending sequence SHA-256, then accession; first 1000 rows",
               "files": {name: _sha(out / name) for name in ("timing.fasta", "sequence_map.tsv")}}
    _write(out / "timing_inputs.json", receipt)
    return receipt


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True,
                                   stderr=subprocess.PIPE, env=git_environment()).strip()


def git_environment() -> dict[str, str]:
    """Repository subprocesses never inherit another checkout's Git environment."""
    return clean_repository_environment()


def clean_repository_environment(base: Mapping[str, str] | None = None) -> dict[str, str]:
    return {key: value for key, value in (os.environ if base is None else base).items()
            if not key.startswith("GIT_")}


def source_identity(source: Mapping[str, Any]) -> dict[str, Any]:
    """Record the scientific binding; retain the historical execution binding for replay."""
    if source.get("format") == EXECUTION_V5_FORMAT or "scientific_identity_sha256" in source:
        identity, chain = source.get("scientific_identity_sha256"), source.get("fit_chain")
        _require(isinstance(identity, str) and re.fullmatch(r"[0-9a-f]{64}", identity)
                 and isinstance(chain, list) and len(chain) >= 2
                 and all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in chain),
                 "Execution v5 requires its scientific identity and amendment-2 fit chain.")
        return {"scientific_identity_sha256": identity, "fit_chain": list(chain)}
    _require(bool(source.get("execution_manifest_sha256")), "Historical execution binding is missing.")
    return {"execution_manifest_sha256": source["execution_manifest_sha256"]}


def require_same_scientific_identity(record: Mapping[str, Any], source: Mapping[str, Any]) -> None:
    expected = source_identity(source)
    _require(all(record.get(key) == value for key, value in expected.items()),
             "Record belongs to another scientific identity or fit chain.")


def authenticated_chain(source: Mapping[str, Any], *, repo: Path = REPO_ROOT) -> dict[str, Any]:
    """Check the workstation receipt cheaply; never run full chain verification on a pod."""
    from ec_clean30 import amendment_chain

    _require(source.get("format") == EXECUTION_V5_FORMAT, "An execution-v5 source is required.")
    receipt = source.get("amendment_chain_receipt")
    _require(isinstance(receipt, str) and receipt, "The session amendment-chain receipt is missing.")
    chain = amendment_chain.check_chain_receipt(Path(receipt), repo=repo,
                                                expected_chain=source.get("amendment_chain"))
    _require(chain.get("tip_sequence", 0) >= 2 and chain.get("published") is True,
             "Real work requires a published amendment chain at sequence 2 or later.")
    require_same_scientific_identity(chain, source)
    return chain


def require_record_precedence(record: Mapping[str, Any], source: Mapping[str, Any], *,
                              repo: Path = REPO_ROOT) -> None:
    require_same_scientific_identity(record, source)
    if source.get("format") == EXECUTION_V5_FORMAT:
        from ec_clean30 import amendment_chain

        amendment_chain.require_record_precedence(authenticated_chain(source, repo=repo), [record], repo=repo)


def _source_pins(repo: Path) -> dict[str, str]:
    names = _git(repo, "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "src", "scripts")
    return {name: _sha(_safe_file(repo, name)) for name in sorted(set(filter(None, names.split("\0"))))}


def _baseline_authentication(repo: Path, commit: str, amendment_name: str, base_digest: str) -> dict[str, Any]:
    """Run the unchanged verifier in a sparse checkout of its published source version."""
    _require(re.fullmatch(r"[0-9a-f]{40}", commit), "Amendment commit must be a full Git object ID.")
    _require(_git(repo, "merge-base", commit, "refs/remotes/origin/main") == commit,
             "The amendment source commit is not published.")
    with tempfile.TemporaryDirectory(prefix="ec30-amendment-source-") as temporary:
        baseline = Path(temporary) / "source"
        subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(repo), str(baseline)],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=git_environment())
        _git(baseline, "sparse-checkout", "set", "--no-cone", "/src/", "/scripts/",
             "/" + freeze.CAMPAIGN_DOCS_DIR + "/")
        _git(baseline, "checkout", "--quiet", "--detach", commit)
        return freeze.verify_s1_amendment(baseline / amendment_name, expected_freeze1_sha256=base_digest,
                                         repo=baseline, require_published=True)


def make_execution_manifest(amendment_path: Path, out: Path, *, amendment_commit: str,
                            freeze1b_sha256: str, timing_inputs_sha256: str | None = None,
                            repo: Path = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo).resolve()
    amendment_path = Path(amendment_path).resolve()
    amendment_name = amendment_path.relative_to(repo).as_posix()
    amendment = _json(amendment_path)
    base_digest = amendment["base_freeze1"]["sha256"]
    verified = _baseline_authentication(repo, amendment_commit, amendment_name, base_digest)
    _require(_sha(amendment_path) == verified["document_sha256"], "Live amendment differs from published bytes.")
    _require(freeze1b_sha256 == FREEZE1B_SHA256, "Freeze-1b binding differs from the published audit.")
    document = {"format": EXECUTION_FORMAT, "campaign_id": CAMPAIGN_ID,
                "freeze1_sha256": base_digest, "freeze1b_sha256": freeze1b_sha256,
                "amendment_path": amendment_name, "amendment_sha256": verified["document_sha256"],
                "amendment_commit": amendment_commit, "amendment_chain": verified["amendment_chain"],
                "timing_inputs_sha256": timing_inputs_sha256,
                "official_fold0_training_sha256": OFFICIAL_FOLD0_SHA256,
                "source_sha256": {name: _sha(_safe_file(repo, name)) for name in EXECUTION_FILES},
                "purpose": "G2 measurements only; separate from S1 scientific fits"}
    _write(out, document)
    verify_execution_source(out, repo=repo, require_published=False)
    return document


def make_execution_v5_manifest(chain_receipt_path: Path, out: Path, *, timing_inputs_sha256: str,
                               expected_chain: Sequence[str], repo: Path = REPO_ROOT,
                               runtime_locks_path: str = RUNTIME_LOCKS_PATH) -> dict[str, Any]:
    """Build a draft from the published chain receipt; publication is a separate maintainer action."""
    from ec_clean30 import amendment_chain, amendment_rules

    repo = Path(repo).resolve()
    verified = amendment_chain.check_chain_receipt(chain_receipt_path, repo=repo, expected_chain=expected_chain)
    _require(verified.get("published") is True and verified.get("tip_sequence", 0) >= 2,
             "Manifest v5 requires the published amendment-2 chain.")
    _require(isinstance(timing_inputs_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", timing_inputs_sha256),
             "Timing inputs need their SHA-256 binding.")
    execution_files = verified["execution_file_set"]
    amendment_rules.partition(_git(repo, "ls-files", "-z", "--", "src", "scripts").split("\0"), execution_files)
    _require(set(BOOTSTRAP_FILES).issubset(execution_files), "The closed execution set lacks its Runpod transport.")
    runtime_path = _safe_file(repo, runtime_locks_path)
    _require(_json(runtime_path).get("format") == "deepmzyme_ec_clean30_runtime_locks_v1",
             "Manifest v5 needs the complete reviewed runtime lock document.")
    document = {"format": EXECUTION_V5_FORMAT, "campaign_id": CAMPAIGN_ID,
                "freeze1_sha256": amendment_rules.FREEZE1_SHA256, "freeze1b_sha256": FREEZE1B_SHA256,
                "amendment_chain": list(verified["chain"]), **source_identity(verified),
                "amendment_path": amendment_rules.link_path(verified["tip_sequence"]),
                "source_sha256": {name: _sha(_safe_file(repo, name)) for name in execution_files},
                "execution_file_set": list(execution_files),
                "runtime_locks": {"path": runtime_locks_path, "sha256": _sha(runtime_path)},
                "timing_inputs_sha256": timing_inputs_sha256,
                "official_fold0_training_sha256": OFFICIAL_FOLD0_SHA256,
                "hardware_profiles": sorted(name for name in GPU_PROFILES if name.startswith("runpod_")),
                "purpose": "Runpod G2 v2 and fold execution under a published scientific identity; separate launch authority required"}
    _write(out, document)
    return document


def _verify_execution_v5(document: Mapping[str, Any], manifest_path: Path, *, repo: Path,
                         require_published: bool, chain_receipt_path: Path | None,
                         expected_chain: Sequence[str] | None) -> dict[str, Any]:
    from ec_clean30 import amendment_chain, amendment_rules

    _require(chain_receipt_path is not None, "Execution v5 requires the session's workstation chain receipt.")
    declared_chain = document.get("amendment_chain")
    _require(isinstance(declared_chain, list) and len(declared_chain) >= 2,
             "Real work requires an amendment-chain tip at sequence 2 or later.")
    if expected_chain is not None:
        _require(declared_chain == list(expected_chain), "Manifest amendment chain differs from the expected chain.")
    verified = amendment_chain.check_chain_receipt(chain_receipt_path, repo=repo, expected_chain=declared_chain)
    _require(verified.get("tip_sequence", 0) >= 2 and verified.get("published") is True,
             "Real work requires a published amendment-chain tip at sequence 2 or later.")
    require_same_scientific_identity(document, verified)
    _require(document.get("amendment_path") == amendment_rules.link_path(verified["tip_sequence"]),
             "Manifest v5 must name its authenticated chain tip.")
    _require(document.get("freeze1_sha256") == amendment_rules.FREEZE1_SHA256,
             "Manifest v5 must retain the original Freeze-1 binding.")
    allowed = verified["execution_file_set"]
    _require(document.get("execution_file_set") == allowed
             and set(document.get("source_sha256", {})) == set(allowed),
             "Manifest v5 must pin exactly the chain's closed execution file set.")
    amendment_rules.partition(_git(repo, "ls-files", "-z", "--", "src", "scripts").split("\0"), allowed)
    _require(all(_sha(_safe_file(repo, name)) == digest for name, digest in document["source_sha256"].items()),
             "Live execution source differs from manifest v5.")
    _require(document.get("hardware_profiles") == sorted(name for name in GPU_PROFILES if name.startswith("runpod_")),
             "Execution v5 permits only Runpod hardware profiles.")
    locks = document.get("runtime_locks", {})
    _require(isinstance(locks, dict) and _sha(_safe_file(repo, locks.get("path", ""))) == locks.get("sha256")
             and _json(_safe_file(repo, locks["path"])).get("format") == "deepmzyme_ec_clean30_runtime_locks_v1",
             "Manifest v5 runtime-lock binding is missing or changed.")
    head = _git(repo, "rev-parse", "HEAD")
    published = (_git(repo, "merge-base", head, "refs/remotes/origin/main") == head
                 and not _git(repo, "status", "--porcelain", "--untracked-files=all"))
    if require_published:
        _require(published, "Real work requires a clean, published execution-v5 checkout.")
        content = subprocess.check_output(["git", "-C", str(repo), "show", f"HEAD:{manifest_path.relative_to(repo)}"],
                                          env=git_environment())
        _require(content == manifest_path.read_bytes(), "Execution manifest v5 is not published with these bytes.")
    base_path = _safe_file(repo, f"{freeze.CAMPAIGN_DOCS_DIR}/freeze1.json")
    _require(_sha(base_path) == document["freeze1_sha256"], "Original Freeze 1 changed.")
    base = _json(base_path)
    return {**document, "execution_manifest_sha256": _sha(manifest_path), "published": published,
            "git_head": head, "source_tree_sha256": provenance.source_tree_sha256(repo),
            "amendment_chain_receipt": str(Path(chain_receipt_path).resolve()), "chain_verification": verified,
            "frozen_views": base.get("views", {}), "frozen_fold0_view": base.get("views", {}).get("0"),
            "frozen_inputs_manifest": base.get("inputs_manifest"),
            "frozen_esmc_weight_sha256": base.get("pins", {}).get("esmc_300m_weights")}


def verify_execution_source(manifest_path: Path, *, repo: Path = REPO_ROOT,
                            require_published: bool = True, chain_receipt_path: Path | None = None,
                            expected_chain: Sequence[str] | None = None) -> dict[str, Any]:
    repo, manifest_path = Path(repo).resolve(), Path(manifest_path).absolute()
    _require(repo in manifest_path.parents and not manifest_path.is_symlink()
             and manifest_path.resolve() == manifest_path,
             "Execution manifest must be a regular repository document.")
    document = _json(manifest_path)
    _require(document.get("format") in {EXECUTION_FORMAT, AUTOMATION_FORMAT, RUNPOD_FORMAT, BOOTSTRAP_FORMAT,
                                       EXECUTION_V5_FORMAT}
             and document.get("campaign_id") == CAMPAIGN_ID,
             "Unknown execution manifest.")
    _require(document.get("freeze1b_sha256") == FREEZE1B_SHA256, "Freeze-1b binding is missing or changed.")
    if document["format"] == EXECUTION_V5_FORMAT:
        return _verify_execution_v5(document, manifest_path, repo=repo, require_published=require_published,
                                    chain_receipt_path=chain_receipt_path, expected_chain=expected_chain)
    verified = _baseline_authentication(repo, document["amendment_commit"], document["amendment_path"],
                                        document["freeze1_sha256"])
    _require(document.get("amendment_sha256") == verified["document_sha256"]
             and document.get("amendment_chain") == verified["amendment_chain"], "Amendment binding differs.")
    amendment = verified["document"]
    base_path = _safe_file(repo, amendment["base_freeze1"]["path"])
    _require(_sha(base_path) == document["freeze1_sha256"] == amendment["base_freeze1"]["sha256"],
             "Live original Freeze 1 changed.")
    base = _json(base_path)
    _require(_sha(repo / document["amendment_path"]) == verified["document_sha256"], "Live amendment changed.")
    expected = dict(base["code"]["tracked_file_sha256"])
    expected.update({name: row["after"] for name, row in amendment["source_changes"].items()})
    pins = document.get("source_sha256", {})
    if document["format"] == BOOTSTRAP_FORMAT:
        _verify_runpod_base(repo, document)
        allowed = BOOTSTRAP_FILES
    elif document["format"] == RUNPOD_FORMAT:
        _verify_execution_base(repo, document)
        _verify_automation_base(repo, document)
        allowed = RUNPOD_FILES
    elif document["format"] == AUTOMATION_FORMAT:
        _verify_execution_base(repo, document)
        allowed = AUTOMATION_FILES
    else:
        allowed = EXECUTION_FILES
    _require(set(pins) == set(allowed), "Execution pins must name exactly the versioned execution files.")
    expected.update(pins)
    _require(_source_pins(repo) == expected, "Live source differs from frozen and execution pins.")
    head = _git(repo, "rev-parse", "HEAD")
    published = (_git(repo, "merge-base", head, "refs/remotes/origin/main") == head
                 and not _git(repo, "status", "--porcelain"))
    if require_published:
        _require(published, "Real G2 work requires a clean, published execution checkout.")
        content = subprocess.check_output(["git", "-C", str(repo), "show", f"HEAD:{manifest_path.relative_to(repo)}"],
                                          env=git_environment())
        _require(content == manifest_path.read_bytes(), "Execution manifest is not published with these bytes.")
    return {**document, "execution_manifest_sha256": _sha(manifest_path), "published": published,
            "git_head": head, "source_tree_sha256": provenance.source_tree_sha256(repo),
            "frozen_fold0_view": base.get("views", {}).get("0"),
            "frozen_esmc_weight_sha256": base.get("pins", {}).get("esmc_300m_weights")}


def _verify_execution_base(repo: Path, document: Mapping[str, Any]) -> dict[str, Any]:
    """The automation extension preserves its published WP-13 manifest and scientific bindings."""
    reference = document.get("base_execution", {})
    commit = reference.get("commit", "")
    _require(isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit),
             "Automation requires a full published WP-13 commit.")
    _require(_git(repo, "merge-base", commit, "refs/remotes/origin/main") == commit
             and _git(repo, "merge-base", commit, "HEAD") == commit,
             "WP-13 base must be published and an ancestor of the execution checkout.")
    name = _relative(reference.get("path", ""))
    path = _safe_file(repo, name)
    published = subprocess.check_output(["git", "-C", str(repo), "show", f"{commit}:{name}"], env=git_environment())
    _require(_sha(path) == reference.get("sha256") == hashlib.sha256(published).hexdigest(),
             "Original published WP-13 manifest changed.")
    base = json.loads(published)
    _require(base.get("format") == EXECUTION_FORMAT and set(base.get("source_sha256", {})) == set(EXECUTION_FILES),
             "Automation base must be the original two-file WP-13 execution manifest.")
    for key in ("campaign_id", "freeze1_sha256", "freeze1b_sha256", "amendment_path", "amendment_sha256",
                "amendment_commit", "amendment_chain", "official_fold0_training_sha256", "timing_inputs_sha256"):
        _require(document.get(key) == base.get(key), f"Automation changed the WP-13 {key} binding.")
    for name, digest in base["source_sha256"].items():
        content = subprocess.check_output(["git", "-C", str(repo), "show", f"{commit}:{name}"], env=git_environment())
        _require(hashlib.sha256(content).hexdigest() == digest, "Published WP-13 source pin differs.")
    return base


def make_automation_manifest(base_execution: Path, out: Path, *, base_commit: str,
                             repo: Path = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo).resolve()
    base_execution = Path(base_execution).resolve()
    document = {**_json(base_execution), "format": AUTOMATION_FORMAT,
                "base_execution": {"path": base_execution.relative_to(repo).as_posix(),
                                   "sha256": _sha(base_execution), "commit": base_commit},
                "source_sha256": {name: _sha(_safe_file(repo, name)) for name in AUTOMATION_FILES},
                "purpose": "G2 automation and Fold-0 S1/CLEAN dispatch; Phase 4 requires numerical G3 authority"}
    _verify_execution_base(repo, document)
    _write(out, document)
    verify_execution_source(out, repo=repo, require_published=False)
    return document


def _verify_automation_base(repo: Path, document: Mapping[str, Any]) -> dict[str, Any]:
    """Authenticate the unchanged published v2 before allowing only the named provider additions."""
    reference = document.get("base_automation", {})
    commit = reference.get("commit", "")
    _require(isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit),
             "Runpod extension requires a full published automation commit.")
    _require(_git(repo, "merge-base", commit, "refs/remotes/origin/main") == commit
             and _git(repo, "merge-base", commit, "HEAD") == commit,
             "Automation base must be published and an ancestor of the execution checkout.")
    name = _relative(reference.get("path", ""))
    path = _safe_file(repo, name)
    published = subprocess.check_output(["git", "-C", str(repo), "show", f"{commit}:{name}"], env=git_environment())
    _require(_sha(path) == reference.get("sha256") == hashlib.sha256(published).hexdigest(),
             "Original published automation manifest changed.")
    base = json.loads(published)
    _require(base.get("format") == AUTOMATION_FORMAT
             and set(base.get("source_sha256", {})) == set(AUTOMATION_FILES),
             "Runpod base must be the original six-file automation manifest.")
    _verify_execution_base(repo, base)
    for key in ("campaign_id", "freeze1_sha256", "freeze1b_sha256", "amendment_path", "amendment_sha256",
                "amendment_commit", "amendment_chain", "official_fold0_training_sha256", "timing_inputs_sha256",
                "base_execution"):
        _require(document.get(key) == base.get(key), f"Runpod extension changed the automation {key} binding.")
    for name, digest in base["source_sha256"].items():
        content = subprocess.check_output(["git", "-C", str(repo), "show", f"{commit}:{name}"], env=git_environment())
        _require(hashlib.sha256(content).hexdigest() == digest, "Published automation source pin differs.")
    _require(document.get("hardware_profiles") == sorted(GPU_PROFILES), "Approved hardware profiles differ.")
    return base


def make_runpod_manifest(base_automation: Path, out: Path, *, base_commit: str,
                         repo: Path = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo).resolve()
    base_automation = Path(base_automation).resolve()
    document = {**_json(base_automation), "format": RUNPOD_FORMAT,
                "base_automation": {"path": base_automation.relative_to(repo).as_posix(),
                                    "sha256": _sha(base_automation), "commit": base_commit},
                "source_sha256": {name: _sha(_safe_file(repo, name)) for name in RUNPOD_FILES},
                "hardware_profiles": sorted(GPU_PROFILES),
                "purpose": "Bounded G2 provider qualification; Phase 4 remains gated by numerical G3 authority"}
    _verify_automation_base(repo, document)
    _write(out, document)
    verify_execution_source(out, repo=repo, require_published=False)
    return document


def _verify_runpod_base(repo: Path, document: Mapping[str, Any]) -> dict[str, Any]:
    """Bind bootstrap additions to the unchanged published provider extension."""
    reference = document.get("base_runpod", {})
    commit = reference.get("commit", "")
    _require(isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit),
             "Bootstrap extension requires a full published Runpod commit.")
    _require(_git(repo, "merge-base", commit, "refs/remotes/origin/main") == commit
             and _git(repo, "merge-base", commit, "HEAD") == commit,
             "Runpod base must be published and an ancestor of the execution checkout.")
    name = _relative(reference.get("path", ""))
    path = _safe_file(repo, name)
    published = subprocess.check_output(["git", "-C", str(repo), "show", f"{commit}:{name}"], env=git_environment())
    _require(_sha(path) == reference.get("sha256") == hashlib.sha256(published).hexdigest(),
             "Original published Runpod manifest changed.")
    base = json.loads(published)
    _require(base.get("format") == RUNPOD_FORMAT
             and set(base.get("source_sha256", {})) == set(RUNPOD_FILES),
             "Bootstrap base must be the original eight-file Runpod execution manifest.")
    _verify_execution_base(repo, base)
    _verify_automation_base(repo, base)
    for key in ("campaign_id", "freeze1_sha256", "freeze1b_sha256", "amendment_path", "amendment_sha256",
                "amendment_commit", "amendment_chain", "official_fold0_training_sha256", "timing_inputs_sha256",
                "base_execution", "base_automation", "hardware_profiles"):
        _require(document.get(key) == base.get(key), f"Bootstrap extension changed the Runpod {key} binding.")
    for name, digest in base["source_sha256"].items():
        content = subprocess.check_output(["git", "-C", str(repo), "show", f"{commit}:{name}"], env=git_environment())
        _require(hashlib.sha256(content).hexdigest() == digest, "Published Runpod source pin differs.")
    return base


def make_bootstrap_manifest(base_runpod: Path, out: Path, *, base_commit: str,
                            repo: Path = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo).resolve()
    base_runpod = Path(base_runpod).resolve()
    document = {**_json(base_runpod), "format": BOOTSTRAP_FORMAT,
                "base_runpod": {"path": base_runpod.relative_to(repo).as_posix(),
                                "sha256": _sha(base_runpod), "commit": base_commit},
                "source_sha256": {name: _sha(_safe_file(repo, name)) for name in BOOTSTRAP_FILES},
                "purpose": "Verified Runpod bootstrap and cached G2 staging; Phase 4 remains gated by numerical G3 authority"}
    _verify_runpod_base(repo, document)
    _write(out, document)
    verify_execution_source(out, repo=repo, require_published=False)
    return document


def _approved_runtime_profile(source: Mapping[str, Any], profile: str) -> dict[str, Any]:
    expected = hardware_profile(profile)
    if profile.startswith("runpod_"):
        _require(source.get("format") in {RUNPOD_FORMAT, BOOTSTRAP_FORMAT, EXECUTION_V5_FORMAT}
                 and profile in source.get("hardware_profiles", []),
                 "Runpod hardware requires its authenticated provider extension.")
    if source.get("format") == EXECUTION_V5_FORMAT:
        _require(profile.startswith("runpod_"), "Execution v5 permits only Runpod sessions.")
    return expected


def _verify_container_caps(source: Mapping[str, Any], evidence: dict[str, Any], cap_prefix: Sequence[str],
                           *, python: str, esm1b: bool = False) -> None:
    from ec_clean30 import container_caps

    container_caps.validate_probe_evidence(evidence)
    _require(evidence["helper_sha256"] == source.get("source_sha256", {}).get("src/ec_clean30/container_caps.py")
             == _sha(REPO_ROOT / "src/ec_clean30/container_caps.py"),
             "Container-cap probe does not bind the authenticated helper.")
    _require(evidence["hostname"] == socket.gethostname()
             and evidence["boot_id"] == Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
             "Container-cap probe belongs to another host or boot.")
    _require(Path(evidence["python_executable"]).resolve() == Path(python).resolve(),
             "Container-cap probe used a different interpreter.")
    cores, gib = (4, 24) if esm1b else (2, 8)
    expected = container_caps.cap_prefix(python, REPO_ROOT, evidence["cgroup_root"],
                                         cpu_cores=cores, memory_bytes=gib * 1024 ** 3)
    _require(list(cap_prefix) == expected, "Runtime cap differs from its validated container profile.")


def _verify_esm1b_model_probe(request: Mapping[str, Any], view: views.View, profile: str) -> dict[str, Any]:
    _require(isinstance(request.get("esm1b_model_probe"), str) and request["esm1b_model_probe"],
             "Runpod G2 requires the completed full-model ESM-1b qualification.")
    path = Path(request["esm1b_model_probe"])
    _outside_guard(path, view.forbidden_read_roots)
    _require(not path.is_symlink() and re.fullmatch(r"[0-9a-f]{64}", str(request.get("esm1b_model_probe_sha256", "")))
             and _sha(path) == request["esm1b_model_probe_sha256"], "ESM-1b qualification bytes differ.")
    evidence = validate_esm1b_model_evidence(_json(path), profile)
    _require(evidence["hostname"] == socket.gethostname()
             and evidence["boot_id"] == Path("/proc/sys/kernel/random/boot_id").read_text().strip()
             and float(request["allocation_started_unix"]) <= evidence["created_unix"] < float(request["deadline_unix"]),
             "ESM-1b qualification belongs to another host, boot or allocation.")
    if request.get("kind") == "esm1b":
        _require(Path(evidence["python_executable"]).resolve() == Path(request["clean_python"]).resolve(),
                 "ESM-1b timing interpreter differs from its full-model qualification.")
    return evidence


def _verify_main_model_probe(request: Mapping[str, Any], view: views.View, profile: str) -> dict[str, Any]:
    _require(isinstance(request.get("main_model_probe"), str) and request["main_model_probe"],
             "Runpod G2 requires the completed main-model CPU/GPU qualification.")
    path = Path(request["main_model_probe"])
    _outside_guard(path, view.forbidden_read_roots)
    _require(not path.is_symlink() and re.fullmatch(r"[0-9a-f]{64}", str(request.get("main_model_probe_sha256", "")))
             and _sha(path) == request["main_model_probe_sha256"], "Main-model qualification bytes differ.")
    evidence = validate_main_model_evidence(_json(path), profile)
    _require(evidence["hostname"] == socket.gethostname()
             and evidence["boot_id"] == Path("/proc/sys/kernel/random/boot_id").read_text().strip()
             and Path(evidence["python_executable"]).resolve() == Path(request["python"]).resolve()
             and float(request["allocation_started_unix"]) <= evidence["created_unix"] < float(request["deadline_unix"]),
             "Main-model qualification belongs to another host, boot, interpreter or allocation.")
    return evidence


def _bind_view(view: views.View, source: Mapping[str, Any]) -> None:
    if view.smoke:
        return
    frozen = (source.get("frozen_views") or {}).get(str(view.fold))
    if frozen is None:
        frozen = source.get("frozen_fold0_view") if view.fold == 0 else {}
    frozen = frozen or {}
    _require(frozen.get("view_manifest_sha256") == view.manifest_sha256
             and frozen.get("dry_run_receipt", {}).get("sha256") == _sha(view.receipts / views.DRY_RUN_RECEIPT_NAME)
             and all(frozen.get("chain_means", {}).get(layer) == view.chain_means_manifest_sha256(layer)
                     for layer in ("final", "block26")), "Fold view differs from original Freeze 1.")
    if source.get("format") == EXECUTION_V5_FORMAT or source.get("scientific_identity_sha256"):
        from ec_clean30 import amendment_chain

        verified = authenticated_chain(source)
        evidence = amendment_chain.fold_dry_run_evidence(verified, view.fold)
        # The dry-run-v2 receipt is private (never on a pod): the chain verification checked its bytes, status,
        # fold, plan version and clean source commit, and carries its view manifest SHA-256 (C11).
        _require(evidence["receipt"].get("view_manifest_sha256") == view.manifest_sha256
                 and evidence["in_view_receipt_sha256"] == _sha(view.receipts / views.DRY_RUN_RECEIPT_NAME),
                 "Amendment-2 dry-run evidence does not bind this fold view (C11).")


def vm_probe(view_root: Path, execution_manifest_path: Path, out: Path, *,
             cap_prefix: Sequence[str], python: str = sys.executable, smoke: bool = False,
             hardware_profile: str = "gcp_l4", cap_probe_path: Path | None = None,
             chain_receipt_path: Path | None = None, expected_chain: Sequence[str] | None = None) -> dict[str, Any]:
    source = verify_execution_source(execution_manifest_path, require_published=not smoke,
                                    chain_receipt_path=chain_receipt_path, expected_chain=expected_chain)
    _approved_runtime_profile(source, hardware_profile)
    view = views.View.load(view_root)
    accepted_folds = range(5) if source.get("format") == EXECUTION_V5_FORMAT and not smoke else (0,)
    _require(view.fold in accepted_folds and view.smoke == smoke, "Probe view differs from its declared role.")
    _bind_view(view, source)
    cap_evidence = {}
    if hardware_profile.startswith("runpod_"):
        _require(cap_probe_path is not None, "Runpod requires a measured container-cap probe.")
        _outside_guard(cap_probe_path, view.forbidden_read_roots)
        measured = _json(cap_probe_path)
        _verify_container_caps(source, measured, cap_prefix, python=python)
        cap_evidence = {"container_caps_probe": measured, "container_caps_probe_sha256": _sha(cap_probe_path),
                        "container_caps_evidence_sha256": hashlib.sha256(_bytes(measured)).hexdigest()}
    original = _sha(view.receipts / views.DRY_RUN_RECEIPT_NAME)
    verification = views.verify_view(view.root, require_chain_means=("final", "block26"), write_receipt=False)
    dry_options = {"plan_version": 2, "pretrain_stub": True} if source.get("format") == EXECUTION_V5_FORMAT else {}
    report = commands.dry_run(view, cap_prefix=cap_prefix, python=python, write=False, **dry_options)
    _require(_sha(view.receipts / views.DRY_RUN_RECEIPT_NAME) == original, "Probe rewrote the original dry run.")
    _require(report.get("status") == "passed", "Relocated VM dry-run probe failed.")
    receipt = {"format": "deepmzyme_ec_clean30_vm_probe_v1", "status": "passed", "source": source,
               "hardware_profile": hardware_profile, **cap_evidence,
               "view_manifest_sha256": view.manifest_sha256, "original_dry_run_sha256": original,
               "verification": verification, "dry_run": report}
    if source.get("format") == EXECUTION_V5_FORMAT:
        receipt.update(source_identity(source), git_head=source["git_head"], created_utc=provenance.utc_now())
    _write(out, receipt)
    return receipt


def timing_plan(*, version: int = 1) -> dict[str, dict[str, Any]]:
    if version == 2:
        from ec_clean30.campaign import timing_recipes_v2

        result = {name: {"family": recipe.family, "epochs": recipe.epochs, "lever": None,
                         "recipe": recipe} for name, recipe in timing_recipes_v2()}
        lane_recipe = dict(timing_recipes_v2())["F4_raw"]
        result.update({name: {"family": lane_recipe.family, "epochs": lane_recipe.epochs,
                             "lever": None, "recipe": lane_recipe}
                       for name in ("lane_serial_1", "lane_serial_2", "lane_parallel_1",
                                    "lane_parallel_2", "lane_parallel_3")})
        return result
    _require(version == 1, "Unknown G2 timing plan version.")
    result = {name: {"family": name, "epochs": 5, "lever": None} for name in FAMILIES}
    result["F4_L3"] = {"family": "F4", "epochs": 5, "lever": "L3"}
    for name in ("lane_serial_1", "lane_serial_2", "lane_parallel_1", "lane_parallel_2", "lane_parallel_3"):
        result[name] = {"family": "F4", "epochs": 30, "lever": None}
    return result


def _runtime(request: Mapping[str, Any], *, pretraining: bool = False) -> tuple[dict[str, Any], views.View, ExecutionPolicy]:
    smoke = request.get("smoke", False)
    _require(type(smoke) is bool, "smoke must be explicit boolean.")
    source = verify_execution_source(Path(request["execution_manifest"]), require_published=not smoke,
                                    chain_receipt_path=request.get("amendment_chain_receipt"),
                                    expected_chain=request.get("expected_chain"))
    if source.get("format") == EXECUTION_V5_FORMAT:
        _require(request.get("plan_version") == 2, "Execution v5 requires G2 v2.")
    approved = _approved_runtime_profile(source, request.get("hardware_profile", "gcp_l4"))
    _require(all(os.environ.get(key) == value for key, value in approved["runtime_environment"].items()),
             "Runtime precision environment differs from its approved hardware profile.")
    view = views.View.load(request["view"])
    _require(view.fold == 0 and view.smoke == smoke, "Runtime view role differs.")
    _bind_view(view, source)
    probe = _json(Path(request["vm_probe"]))
    _require(probe.get("status") == "passed" and probe.get("view_manifest_sha256") == view.manifest_sha256,
             "Runtime requires its matching completed VM probe.")
    require_same_scientific_identity(probe.get("source", {}), source)
    if source.get("format") == EXECUTION_V5_FORMAT and not pretraining:
        verify_pretrain_evidence(Path(request["pretrain_evidence"]), source, view=view)
    profile = request.get("hardware_profile", "gcp_l4")
    if profile.startswith("runpod_"):
        _require(probe.get("hardware_profile") == profile, "Runtime hardware profile differs from its VM probe.")
        _require(isinstance(probe.get("container_caps_probe"), dict)
                 and re.fullmatch(r"[0-9a-f]{64}", str(probe.get("container_caps_probe_sha256", "")))
                 and probe.get("container_caps_evidence_sha256")
                 == hashlib.sha256(_bytes(probe["container_caps_probe"])).hexdigest(),
                 "Runtime requires its hash-bound container-cap probe.")
        _verify_container_caps(source, probe["container_caps_probe"], request["cap_prefix"],
                               python=request["python"], esm1b=request.get("kind") == "esm1b")
        _verify_esm1b_model_probe(request, view, profile)
        _verify_main_model_probe(request, view, profile)
        cap_gib = 24 if request.get("kind") == "esm1b" else 8
        _require(request.get("memory_limit_bytes") == cap_gib * 1024 ** 3,
                 "Runtime memory verification limit differs from its declared container cap.")
    _require(isinstance(request.get("cap_prefix"), list) and request["cap_prefix"], "A nonempty runtime cap is required.")
    _require(request.get("device") in {"cpu", "cuda"} and (smoke or request["device"] == "cuda"),
             "Real G2 timing requires the declared CUDA device.")
    _require(int(request["memory_limit_bytes"]) > 0, "Memory limit must be positive.")
    _require(0 < float(request["unit_timeout_seconds"]) <= float(request["forecast_seconds"]),
             "The bounded probe timeout must fit its full-unit admission forecast.")
    policy = ExecutionPolicy(deadline_unix=float(request["deadline_unix"]),
                             max_total_seconds=float(request["max_total_seconds"]),
                             allocation_started_unix=float(request["allocation_started_unix"]),
                             reserve_seconds=900, forecast_margin=1.25)
    if source.get("format") == EXECUTION_V5_FORMAT:
        _require(math.isfinite(policy.max_total_seconds) and policy.max_total_seconds > policy.reserve_seconds
                 and math.isclose(policy.deadline_unix - policy.allocation_started_unix,
                                  policy.max_total_seconds, rel_tol=0, abs_tol=1)
                 and re.fullmatch(r"[0-9a-f]{64}", str(request.get("runpod_config_sha256", ""))),
                 "Execution v5 requires the configured session duration, deadline and Runpod configuration binding.")
    else:
        _require(policy.max_total_seconds <= 7200, "This G2 package permits at most a two-hour allocation.")
        if profile.startswith("runpod_"):
            _require(policy.max_total_seconds <= 7020, "Runpod G2 permits at most the reviewed 117-minute allocation.")
    execution_root = Path(request["execution_root"]).resolve()
    _require(execution_root == (view.default_runs_dir / "g2").resolve(), "G2 uses its dedicated fold-0 execution root.")
    _require(request.get("persistence_mode") == "host_pull", "G2 uses independent host-pull acknowledgment.")
    return source, view, policy


def _unit_name(request: Mapping[str, Any]) -> str:
    name, attempt = request["recipe"], request.get("attempt", 1)
    _require(name in timing_plan(version=request.get("plan_version", 1))
             and type(attempt) is int and attempt > 0, "Unknown timing recipe or attempt.")
    return f"ecc30timing_{name}_attempt{attempt}"


def _build_timing_command(request: Mapping[str, Any], source: Mapping[str, Any], view: views.View):
    from dataclasses import replace

    version = 2 if source.get("format") == EXECUTION_V5_FORMAT else 1
    spec = timing_plan(version=version)[request["recipe"]]
    recipe = spec.get("recipe") or default_smoke_recipe(spec["family"], epochs=spec["epochs"], lr_index=1)
    if spec["lever"]:
        recipe = recipe_with_lever(recipe, spec["lever"])
    command = commands.build_unit_command(view, recipe, unit_id=_unit_name(request), kind=KIND_INNER,
        seed=42, runs_dir=Path(request["execution_root"]) / "runs", step="G2_TIMING", inner_fold=0,
        device=request["device"], freeze1_sha256=source["freeze1_sha256"],
        freeze1b_sha256=source["freeze1b_sha256"],
        freeze1_amendment_chain=source.get("fit_chain", source["amendment_chain"]),
        git_head=source["git_head"], tree_sha256=source["source_tree_sha256"])
    return replace(command, unset_env=tuple(sorted(set(command.unset_env)
                   | {key for key in os.environ if key.startswith("GIT_")})))


def _prediction_table(run_dir: Path) -> dict[str, list[Any]]:
    prediction = _json(run_dir / "ec_prediction_receipt.json")
    path = run_dir / prediction["protein_predictions"]["path"]
    _require(_sha(path) == prediction["protein_predictions"]["sha256"], "Prediction bytes changed.")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    # The exporter owns these column names and the protein-level averaging.
    ids = [row["ec_group_key"] for row in rows]
    _require(bool(ids) and len(ids) == len(set(ids)), "Protein export has duplicate or empty identities.")
    return {"ids": ids, "true": [int(row["y"]) for row in rows],
            "predicted": [int(row["pred"]) for row in rows]}


def timing_worker(request: Mapping[str, Any]) -> dict[str, Any]:
    """One owned child does fit -> verify -> independent replay, with a fixed timing identity."""
    started = time.monotonic()
    source, view, _ = _runtime(request)
    command = _build_timing_command(request, source, view)
    prepared = time.monotonic()
    runner = commands.run_unit(command, view=view, cap_prefix=request["cap_prefix"], python=request["python"])
    _require(runner.get("status") == "completed", "Timing fit failed; preserve its attempt and restart from seed.")
    fitted = time.monotonic()
    verified = commands.verify_unit(Path(command.runs_dir), command.unit_id, view=view,
                                   memory_limit_bytes=int(request["memory_limit_bytes"]))
    _require(verified.get("passed"), "Timing unit verification failed.")
    checked = time.monotonic()
    run_dir = Path(command.runs_dir) / command.unit_id
    replay_atol = "1e-5" if request["device"] == "cuda" else "1e-6"
    replay_argv = [request["python"], "-m", "training.ec_prediction_export", "replay",
                   "--run-dir", str(run_dir), "--device", "cpu", "--logit-atol", replay_atol]
    for root in view.forbidden_read_roots:
        replay_argv += ["--forbidden-read-root", root]
    env = clean_repository_environment(command.environment())
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    with (run_dir / "g2_replay.log").open("x") as log:
        subprocess.run([*request["cap_prefix"], *replay_argv], cwd=command.runs_dir, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    replay = run_dir / "ec_prediction_replay" / "ec_prediction_replay_receipt.json"
    _require(replay.is_file(), "Independent replay receipt is missing.")
    replay_record = _json(replay)
    _require(replay_record.get("device") == "cpu" and replay_record.get("logit_atol") == float(replay_atol)
             and replay_record.get("replay_status") == "match", "Independent replay policy or result differs.")
    if source.get("format") == EXECUTION_V5_FORMAT:
        from ec_clean30.phase4 import verify_cpu_replay

        verify_cpu_replay(run_dir)
    finished = time.monotonic()
    record = {"format": "deepmzyme_ec_clean30_timing_unit_v1", "status": "passed", "recipe": request["recipe"],
              "unit_id": command.unit_id, "source": source, "view_manifest_sha256": view.manifest_sha256,
              "execution_manifest_sha256": source["execution_manifest_sha256"],
              **source_identity(source), "git_head": source.get("git_head"), "created_utc": provenance.utc_now(),
              "runner_sha256": _sha(commands.runner_receipt_path(Path(command.runs_dir), command.unit_id)),
              "replay_sha256": _sha(replay), "verification": verified,
              "seconds": {"preparation": prepared-started, "fit_and_export": fitted-prepared,
                          "verification": checked-fitted, "independent_replay": finished-checked},
              "trainer_runtime_profile": _json(run_dir / "runtime_profile.json"),
              "wall_seconds": finished-started, **_prediction_table(run_dir)}
    _write(run_dir / "g2_timing.json", record)
    return record


def _execution(request: Mapping[str, Any], policy: ExecutionPolicy) -> CampaignExecution:
    require_restored_pretrain_attempts(Path(request["execution_root"]))
    return CampaignExecution(Path(request["execution_root"]), policy=policy,
                             session_id=request["session_id"], durable_root=Path(request["durable_root"]),
                             persistence_mode="host_pull")


def _child_command(request_file: Path, action: str, python: str) -> list[str]:
    return [python, str(REPO_ROOT / "scripts/run_ec_clean30_phase3.py"), action, "--request", str(request_file)]


def run_timing_unit(request: Mapping[str, Any]) -> dict[str, Any]:
    source, view, policy = _runtime(request)
    name = _unit_name(request)
    root = Path(request["execution_root"])
    output = root / "runs" / name / "g2_timing.json"
    with _execution(request, policy) as execution:
        if output.exists():
            record = _json(output)
            _require(record.get("status") == "passed",
                     "Existing timing result has another source or failed.")
            if source.get("format") == EXECUTION_V5_FORMAT:
                require_record_precedence(record, source)
            else:
                _require(record.get("source") == source, "Existing timing result has another source or failed.")
            verification = commands.verify_unit(root / "runs", name, view=view,
                                                memory_limit_bytes=int(request["memory_limit_bytes"]))
            _require(verification["passed"], "Existing timing result no longer verifies.")
            _require(execution.state.get("pending_unit", {}).get("persisted"),
                     "Existing timing result awaits persistence acknowledgment.")
            complete = _complete_timing(record, execution, output)
            return {"status": "reused_verified", "unit_id": name, "timing_sha256": _sha(output),
                    "complete": complete}
        request_file = root / "commands" / f"{name}.json"
        _write(request_file, dict(request))
        execution.admit(name, float(request["forecast_seconds"]))
        started = time.monotonic()
        with (root / "commands" / f"{name}.log").open("x") as log:
            result = execution.run_subprocess(_child_command(request_file, "_timing-worker", request["python"]),
                                              cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                              env=clean_repository_environment(),
                                              timeout=float(request["unit_timeout_seconds"]))
        elapsed = time.monotonic()-started
        passed = result.returncode == 0 and output.is_file() and _json(output).get("status") == "passed"
        execution.record_result(name, "completed" if passed else "failed_timing", elapsed)
        artifacts = [request_file, root / "commands" / f"{name}.log"]
        artifacts += [path for path in (root / "runs" / name, root / "runs" / f"{name}.log",
                      commands.runner_receipt_path(root / "runs", name)) if path.exists()]
        persisted = execution.persist(artifacts)
        _require(passed, "Timing failed; saved artifacts must be acknowledged before an original-seed retry.")
        return {"status": "awaiting_host_ack", "unit_id": name, "timing_sha256": _sha(output),
                "wall_seconds": elapsed, "persistence": persisted}


def host_ack(manifest_path: Path, downloaded_root: Path, ack_path: Path, *, transfer_seconds: float) -> dict[str, Any]:
    _require(math.isfinite(transfer_seconds) and transfer_seconds >= 0, "Record actual completed transfer seconds.")
    _require(not Path(ack_path).exists(), "Host acknowledgment already exists.")
    started = time.monotonic()
    record = verify_host_pull(Path(manifest_path), Path(downloaded_root), Path(ack_path))
    record.update(transfer_seconds=transfer_seconds, verification_seconds=time.monotonic()-started)
    Path(ack_path).write_bytes(_bytes(record))
    return record


def _persistence_seconds(execution: CampaignExecution) -> tuple[float, dict[str, Any]]:
    transfer = execution.state.get("last_host_transfer") or {}
    _require(bool(transfer.get("ack_path")), "A measured independent host acknowledgment is required.")
    ack = _json(execution.root / transfer["ack_path"])
    values = [ack.get(name) for name in ("transfer_seconds", "verification_seconds")]
    _require(all(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 for value in values),
             "Host acknowledgment lacks actual transfer/readback timing.")
    return float(sum(values)), transfer


def _complete_timing(record: Mapping[str, Any], execution: CampaignExecution, output: Path) -> dict[str, Any]:
    destination = output.with_name("g2_complete.json")
    if destination.exists():
        result = _json(destination)
        _require(result.get("timing_sha256") == _sha(output), "Completed timing receipt changed.")
        return result
    seconds, transfer = _persistence_seconds(execution)
    _require(execution.state.get("pending_unit", {}).get("run_name") == record["unit_id"],
             "Acknowledgment belongs to a different timing unit.")
    complete = {**record, "format": "deepmzyme_ec_clean30_completed_timing_v1",
                "timing_sha256": _sha(output), "persistence_seconds": seconds,
                "wall_seconds": float(record["wall_seconds"])+seconds,
                "host_transfer": transfer}
    _write(destination, complete)
    return complete


def lane_gate(serial: Sequence[Mapping[str, Any]], parallel: Sequence[Mapping[str, Any]], *,
              batch_wall_seconds: float, limits: Mapping[str, Any], observed: Mapping[str, Any]) -> dict[str, Any]:
    """The fixed operational lane diagnostic; it never selects an EC model."""
    import numpy as np

    _require(len(serial) == 2 and len(parallel) == 3, "The lane gate needs exactly two serial and three parallel fits.")
    all_runs = [*serial, *parallel]
    _require([row["recipe"] for row in serial] == ["lane_serial_1", "lane_serial_2"]
             and [row["recipe"] for row in parallel] == [f"lane_parallel_{i}" for i in (1, 2, 3)],
             "Lane recipe identities differ from the fixed plan.")
    ids, truth = list(serial[0]["ids"]), np.asarray(serial[0]["true"], dtype=int)
    _require(ids and len(set(ids)) == len(ids) and len(truth) == len(ids), "Lane query identities are invalid.")
    if serial[0].get("scientific_identity_sha256"):
        binding = {key: serial[0].get(key) for key in ("scientific_identity_sha256", "fit_chain")}
        _require(re.fullmatch(r"[0-9a-f]{64}", str(binding["scientific_identity_sha256"]))
                 and isinstance(binding["fit_chain"], list) and len(binding["fit_chain"]) >= 2
                 and all(re.fullmatch(r"[0-9a-f]{64}", str(value)) for value in binding["fit_chain"])
                 and all(all(row.get(key) == value for key, value in binding.items()) for row in all_runs),
                 "Lane scientific identities or fit chains differ.")
    else:
        binding = {"execution_manifest_sha256": serial[0].get("execution_manifest_sha256")}
        _require(bool(binding["execution_manifest_sha256"])
                 and all(not row.get("scientific_identity_sha256")
                         and row.get("execution_manifest_sha256") == binding["execution_manifest_sha256"]
                         for row in all_runs), "Lane source identities differ.")
    for row in all_runs:
        _require(list(row["ids"]) == ids and np.array_equal(row["true"], truth)
                 and len(row["predicted"]) == len(ids), "Lane query/label membership differs.")
        _require(math.isfinite(float(row["wall_seconds"])) and float(row["wall_seconds"]) > 0,
                 "Lane timing must be finite and positive.")
    _require(math.isfinite(batch_wall_seconds) and batch_wall_seconds > 0, "Batch wall time must be positive.")
    supported = [label for label in sorted(set(truth)) if int(np.sum(truth == label)) >= 10]
    comparisons = []
    for lane in parallel:
        for reference in serial:
            a, b = np.asarray(reference["predicted"]), np.asarray(lane["predicted"])
            recall_a = [float(np.mean(a[truth == label] == label)) for label in supported]
            recall_b = [float(np.mean(b[truth == label] == label)) for label in supported]
            delta_ba = abs(float(np.mean(recall_a))-float(np.mean(recall_b)))*100 if supported else None
            delta_recall = max((abs(a-b)*100 for a, b in zip(recall_a, recall_b)), default=None)
            agreement = float(np.mean(a == b))
            comparisons.append({"lane": lane["recipe"], "reference": reference["recipe"],
                                "ba_difference_points": delta_ba, "max_recall_difference_points": delta_recall,
                                "top1_agreement": agreement, "passed": bool(supported) and delta_ba <= 1.0+1e-12
                                and delta_recall <= 3.0+1e-12 and agreement >= .95-1e-12})
    resource_pass = True
    for name in ("cpu_percent", "ram_bytes", "vram_bytes"):
        _require(name in limits and name in observed and math.isfinite(float(limits[name]))
                 and math.isfinite(float(observed[name])) and float(limits[name]) > 0 and float(observed[name]) >= 0,
                 "The lane gate requires measured numeric host limits and peaks.")
        resource_pass &= float(observed[name]) <= float(limits[name])
    resource_pass &= observed.get("swap_bytes") == 0 and observed.get("oom") is False and observed.get("read_guard_breach") is False
    speedup = 3 * sum(float(row["wall_seconds"]) for row in serial) / 2 / batch_wall_seconds
    passed = resource_pass and all(row["passed"] for row in comparisons) and speedup >= 1.2
    return {"format": "deepmzyme_ec_clean30_lane_gate_v1", "status": "passed" if passed else "failed",
            **binding, "lanes": 3 if passed else 1, "speedup": speedup,
            "supported_classes": [int(label) for label in supported], "minimum_validation_support": 10,
            "comparisons": comparisons, "resource_passed": resource_pass, "limits": dict(limits),
            "observed": dict(observed), "batch_wall_seconds": batch_wall_seconds,
            "timing_scope": "fit/export, verification, independent replay and active host transfer/readback"}


def _host_sample() -> dict[str, float]:
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        memory[key] = int(value.split()[0]) * 1024
    cpu = [int(value) for value in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                  text=True, timeout=5).splitlines()
    _require(len(gpu) == 1, "The lane gate expects one GPU.")
    return {"ram_bytes": memory["MemTotal"]-memory["MemAvailable"],
            "swap_bytes": memory["SwapTotal"]-memory["SwapFree"], "vram_bytes": float(gpu[0])*1024*1024,
            "cpu_total": sum(cpu), "cpu_idle": cpu[3]+cpu[4]}


def lane_worker(request: Mapping[str, Any]) -> dict[str, Any]:
    """One owned process starts precisely the three declared parallel children."""
    _runtime(request)
    root = Path(request["execution_root"])
    batch = root / "runs" / request["batch_name"]
    batch.mkdir(parents=True)
    lock_fd = int(os.environ["DEEPMZYME_EXECUTION_LOCK_FD"])
    os.fstat(lock_fd)
    processes, logs = [], []
    started = time.monotonic()
    previous = _host_sample()
    peaks = {"cpu_percent": 0.0, **{key: previous[key] for key in ("ram_bytes", "swap_bytes", "vram_bytes")},
             "oom": False, "read_guard_breach": False}
    try:
        for index in (1, 2, 3):
            child = {**request, "recipe": f"lane_parallel_{index}"}
            path = batch / f"lane{index}.json"
            _write(path, child)
            log = (batch / f"lane{index}.log").open("x")
            logs.append(log)
            processes.append(subprocess.Popen(_child_command(path, "_timing-worker", request["python"]),
                                               cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                               env=clean_repository_environment(),
                                               pass_fds=(lock_fd,)))
        while any(process.poll() is None for process in processes):
            time.sleep(.5)
            current = _host_sample()
            total = current["cpu_total"]-previous["cpu_total"]
            if total > 0:
                busy = total-(current["cpu_idle"]-previous["cpu_idle"])
                peaks["cpu_percent"] = max(peaks["cpu_percent"], 100*(os.cpu_count() or 1)*busy/total)
            for name in ("ram_bytes", "swap_bytes", "vram_bytes"):
                peaks[name] = max(peaks[name], current[name])
            previous = current
        codes = [process.wait() for process in processes]
        records = []
        for index in (1, 2, 3):
            path = root / "runs" / _unit_name({**request, "recipe": f"lane_parallel_{index}"}) / "g2_timing.json"
            if path.is_file():
                records.append(_json(path))
        record = {"status": "passed" if all(code == 0 for code in codes) and len(records) == 3 else "failed",
                  "records": records, "return_codes": codes, "observed": peaks,
                  "batch_wall_seconds": time.monotonic()-started}
        _write(batch / "batch_timing.json", record)
        return record
    finally:
        active = [process for process in processes if process.poll() is None]
        for process in active:
            process.terminate()
        for process in active:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
        for log in logs:
            log.close()


def run_lane_gate(request: Mapping[str, Any]) -> dict[str, Any]:
    source, _view, policy = _runtime(request)
    serial_paths = [Path(path) for path in request["serial_receipts"]]
    serial = [_json(path) for path in serial_paths]
    _require(len(serial) == 2 and all(row.get("format") == "deepmzyme_ec_clean30_completed_timing_v1"
             and row.get("status") == "passed"
             for row in serial), "Lane gate requires two verified and persisted serial repeats.")
    for row in serial:
        require_record_precedence(row, source)
    _require(float(request["forecast_seconds"]) >= 3*max(row["wall_seconds"] for row in serial),
             "Batch admission may not assume the speedup under test.")
    root = Path(request["execution_root"])
    attempt = request.get("attempt", 1)
    _require(type(attempt) is int and attempt > 0, "Invalid batch attempt.")
    name = f"ecc30timing_lane_batch_attempt{attempt}"
    batch = root / "runs" / name
    measured = batch / "batch_timing.json"
    with _execution(request, policy) as execution:
        if measured.exists():
            result = _json(measured)
            persistence_seconds, transfer = _persistence_seconds(execution)
            _require(execution.state.get("pending_unit", {}).get("run_name") == name
                     and execution.state["pending_unit"].get("persisted"), "Batch acknowledgment is missing.")
            gate_path = batch / "lane_gate.json"
            if gate_path.exists():
                return _json(gate_path)
            if result.get("status") == "passed":
                # Both serial units and the parallel batch include exactly their own transfer/readback work.
                gate = lane_gate(serial, result["records"], batch_wall_seconds=result["batch_wall_seconds"]+persistence_seconds,
                                 limits=request["limits"], observed=result["observed"])
            else:
                gate = {"format": "deepmzyme_ec_clean30_lane_gate_v1", "status": "failed", "lanes": 1,
                        "reason": "parallel_batch_failed", **source_identity(source)}
            gate.update(host_transfer=transfer, serial_receipts=[{"path": str(path), "sha256": _sha(path)} for path in serial_paths],
                        batch_timing_sha256=_sha(measured), persistence_seconds=persistence_seconds,
                        git_head=source.get("git_head"), created_utc=provenance.utc_now())
            _write(gate_path, gate)
            return gate
        local_request = {**request, "batch_name": name}
        request_file = root / "commands" / f"{name}.json"
        _write(request_file, local_request)
        execution.admit(name, float(request["forecast_seconds"]))
        env = clean_repository_environment()
        env["DEEPMZYME_EXECUTION_LOCK_FD"] = str(execution._lock.fileno())
        started = time.monotonic()
        with (root / "commands" / f"{name}.log").open("x") as log:
            result = execution.run_subprocess(_child_command(request_file, "_lane-worker", request["python"]),
                cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                timeout=float(request["unit_timeout_seconds"]))
        passed = result.returncode == 0 and measured.is_file() and _json(measured).get("status") == "passed"
        if not measured.exists():
            _write(measured, {"status": "failed", "reason": "batch_worker_failed", "records": []})
        execution.record_result(name, "completed" if passed else "failed_lane", time.monotonic()-started)
        artifacts = [batch, request_file, root / "commands" / f"{name}.log"]
        for index in (1, 2, 3):
            unit = _unit_name({**request, "recipe": f"lane_parallel_{index}"})
            artifacts += [path for path in (root / "runs" / unit, root / "runs" / f"{unit}.log",
                          commands.runner_receipt_path(root / "runs", unit)) if path.exists()]
        persistence = execution.persist(artifacts)
        return {"status": "awaiting_host_ack", "batch_passed": passed, "persistence": persistence}


def build_freeze1c(evidence: Sequence[Path], out: Path, *, execution_manifest_path: Path,
                   l3_vm_hours_per_fold: Mapping[str, Any] | None = None,
                   chain_receipt_path: Path | None = None,
                   expected_chain: Sequence[str] | None = None,
                   pretrain_evidence: Path | None = None) -> dict[str, Any]:
    source = verify_execution_source(execution_manifest_path, chain_receipt_path=chain_receipt_path,
                                    expected_chain=expected_chain)
    v2 = source.get("format") == EXECUTION_V5_FORMAT
    if not v2:
        _require(isinstance(l3_vm_hours_per_fold, Mapping)
                 and set(l3_vm_hours_per_fold) == {str(index) for index in range(5)}
                 and all(not isinstance(value, bool) and math.isfinite(float(value)) and float(value) > 0
                         for value in l3_vm_hours_per_fold.values()), "Freeze 1c needs five positive numerical L3 ceilings.")
    else:
        _require(l3_vm_hours_per_fold is None, "Freeze 1c v2 has no inactive L3 ceilings.")
        _require(pretrain_evidence is not None, "Freeze 1c v2 requires persisted F6 cache and parity preparation.")
        verify_pretrain_evidence(Path(pretrain_evidence), source)
    records, bindings = [], []
    for path in evidence:
        path = Path(path)
        _no_restricted_path(path)
        row = _json(path)
        require_record_precedence(row, source)
        transfer = row.get("host_transfer") or {}
        _require(all(re.fullmatch(r"[0-9a-f]{64}", str(transfer.get(key, "")))
                     for key in ("manifest_sha256", "ack_sha256")), "Measurement lacks verified host persistence.")
        records.append(row)
        bindings.append({"path": str(path), "sha256": _sha(path)})
    timings = [row for row in records if row.get("format") == "deepmzyme_ec_clean30_completed_timing_v1"]
    if v2:
        from ec_clean30.campaign import timing_recipes_v2

        required_timings = {name for name, _recipe in timing_recipes_v2()}
    else:
        required_timings = {*FAMILIES, "F4_L3"}
    _require({row.get("recipe") for row in timings} == required_timings
             and len(timings) == 7 and all(row.get("status") == "passed" for row in timings),
             "Freeze 1c needs every timing recipe from its versioned G2 plan.")
    embedding = [row for row in records if row.get("format") == "deepmzyme_ec_clean30_completed_embedding_v1"]
    _require(len(embedding) == 2 and {row.get("kind") for row in embedding} == {"esmc", "esm1b"}
             and all(row.get("status") == "passed" for row in embedding), "Both embedding timings are required.")
    gates = [row for row in records if row.get("format") == "deepmzyme_ec_clean30_lane_gate_v1"]
    _require(len(records) == 10 and len(gates) == 1 and gates[0].get("status") in {"passed", "failed"}
             and gates[0].get("lanes") == (3 if gates[0]["status"] == "passed" else 1),
             "Freeze 1c needs one complete lane decision.")
    document = {"format": "deepmzyme_ec_clean30_freeze1c_v2" if v2 else "deepmzyme_ec_clean30_freeze1c_v1",
                "campaign_id": CAMPAIGN_ID,
                "freeze1_sha256": source["freeze1_sha256"], "freeze1b_sha256": source["freeze1b_sha256"],
                "execution_manifest_sha256": source["execution_manifest_sha256"],
                **source_identity(source), "git_head": source.get("git_head"), "created_utc": provenance.utc_now(),
                "lanes": gates[0]["lanes"],
                "evidence": bindings, "budget_authority": "separate_G3_approval_required"}
    if v2:
        document["timing_recipes"] = [name for name, _recipe in timing_recipes_v2()]
        document["pretrain_evidence"] = {"path": str(pretrain_evidence), "sha256": _sha(Path(pretrain_evidence))}
    else:
        document["l3_vm_hours_per_fold"] = dict(l3_vm_hours_per_fold)
    _write(out, document)
    return document


def _esmc_cached_weight(cache: Path) -> Path:
    """Resolve the same default snapshot/file used by the installed ESMC-300m SDK, offline."""
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(repo_id="EvolutionaryScale/esmc-300m-2024-12",
                                 cache_dir=str(cache / "hub"), local_files_only=True)
    return (Path(snapshot) / "data/weights/esmc_300m_2024_12_v0.pth").resolve()


def _bound_pretrain_file(source: Mapping[str, Any], role: str, path: Path) -> dict[str, str]:
    from ec_clean30 import amendment_rules

    tip = _json(_safe_file(REPO_ROOT, source["amendment_path"]))
    bindings = amendment_rules.file_bindings(amendment_rules.binding_at(tip, role), role)
    _no_restricted_path(path)
    _require(path.is_file() and not path.is_symlink(), "A bound pretraining input must be a regular file.")
    digest = _sha(path)
    matched = [name for _where, name, expected in bindings if expected == digest]
    _require(len(matched) == 1, f"Pretraining input does not match the authenticated {role} binding.")
    return {"path": matched[0], "sha256": digest}


def pretrain_sequence_inputs(union_fasta: Path, out: Path) -> dict[str, Any]:
    """Materialize the producer's sequence map and its public, predeclared CPU parity sample."""
    from ec_clean30 import pretrain_corpus
    from embed_helpers.esmc_chain_means import read_fasta

    records = read_fasta(union_fasta)
    sequences = dict(records)
    _require(bool(records) and len(records) == len(sequences), "Union FASTA has duplicate or empty accession keys.")
    keys = pretrain_corpus.parity_sample_keys(sequences)
    out.mkdir(parents=True, exist_ok=False)
    mapping = out / "sequence_map.tsv"
    with mapping.open("x", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("accession", "sha256", "sequence"))
        for accession, sequence in sorted(sequences.items()):
            writer.writerow((accession, pretrain_corpus.sequence_key(sequence), sequence))
    sample = {}
    for accession, sequence in sorted(sequences.items()):
        key = pretrain_corpus.sequence_key(sequence)
        if key in keys and key not in sample:
            sample[key] = (accession, sequence)
    sample_fasta = out / "cpu_sample.fasta"
    with sample_fasta.open("x") as handle:
        handle.write("".join(f">{sample[key][0]}\n{sample[key][1]}\n" for key in keys))
    result = {"sequence_map": str(mapping), "sequence_map_sha256": _sha(mapping),
              "cpu_sample_fasta": str(sample_fasta), "cpu_sample_fasta_sha256": _sha(sample_fasta),
              "sample_keys": keys, "union_fasta_sha256": _sha(union_fasta)}
    _write(out / "sequence_inputs.json", result)
    return result


def pretrain_embedding_command(request: Mapping[str, Any], source: Mapping[str, Any], view: views.View, *,
                               fasta: Path, sequence_map: Path, sequence_map_sha256: str,
                               output: Path, device: str) -> tuple[list[str], dict[str, str]]:
    from training.access_guard import FORBIDDEN_READ_ROOTS_ENV, forbidden_roots_environment

    _require(device in {"cpu", "cuda"}, "F6 embedding device must be CPU or CUDA.")
    roots = view.forbidden_read_roots
    cache, weights = Path(request["hf_home"]).resolve(), Path(request["esmc_weights"]).resolve()
    for path in (fasta, sequence_map, output, cache, weights):
        _outside_guard(path, roots)
    _require(_sha(sequence_map) == sequence_map_sha256 and cache in weights.parents
             and _esmc_cached_weight(cache) == weights and _sha(weights) == source["frozen_esmc_weight_sha256"],
             "F6 embedding input or frozen offline ESMC weight differs.")
    env = clean_repository_environment()
    env.update(PYTHONPATH=os.pathsep.join((str(REPO_ROOT / "src"), str(REPO_ROOT))), HF_HOME=str(cache),
               HF_HUB_CACHE=str(cache / "hub"), HUGGINGFACE_HUB_CACHE=str(cache / "hub"),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    env.pop("INFRA_PROVIDER", None)
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    env[FORBIDDEN_READ_ROOTS_ENV] = forbidden_roots_environment(roots)
    env = runtime_precision.execution_environment(env, device)
    # The fp32 producer itself records the precision policy after converting the loaded model.
    wrapper = ("import runpy,sys\nfrom training.access_guard import install_guard_from_environment\n"
               "if not install_guard_from_environment():\n    raise SystemExit('forbidden-read roots are required')\n"
               "sys.argv=sys.argv[1:]\nrunpy.run_module('embed_helpers.esmc_chain_means',run_name='__main__')")
    argv = [request["python"], "-c", wrapper, "esmc_chain_means", "sequences", "--fasta", str(fasta),
            "--sequence-map", str(sequence_map), "--sequence-map-sha256", sequence_map_sha256,
            "--out", str(output), "--layer-index", "26", "--device", device, "--forward-precision", "fp32"]
    return argv, env


def attach_pretrain(request: Mapping[str, Any]) -> dict[str, Any]:
    """Attach a fold's bound corpus to the verified parent cache; no fit or model build is performed."""
    from ec_clean30 import pretrain_corpus

    source = verify_execution_source(Path(request["execution_manifest"]),
        chain_receipt_path=request.get("amendment_chain_receipt"), expected_chain=request.get("expected_chain"))
    _require(source.get("format") == EXECUTION_V5_FORMAT, "F6 attachment requires execution manifest v5.")
    authenticated_chain(source)
    view = views.View.load(request["view"])
    _require(not view.smoke and view.fold in range(5), "F6 attachment needs its real frozen fold view.")
    if view.fold > 0:
        from ec_clean30 import futility

        decision = futility.verify_futility_decision(Path(request["fold0_step_chain"]),
            freeze1_sha256=source["freeze1_sha256"], amendment_tip=REPO_ROOT / source["amendment_path"],
            campaign_root=Path(request["campaign_root"]), chain_receipt=Path(source["amendment_chain_receipt"]),
            expected_chain=source["amendment_chain"])
        require_same_scientific_identity(decision, source)
        _require(decision.get("decision") == "continue", "Fold 1-4 F6 preparation is blocked by the futility decision.")
    _bind_view(view, source)
    corpus = Path(request["corpus_table"])
    corpus_binding = _bound_pretrain_file(source, f"inputs.f6_corpus.tables.{view.fold}", corpus)
    holdout = Path(request["holdout_receipt"])
    holdout_binding = _bound_pretrain_file(source, f"inputs.f6_holdout.{view.fold}", holdout)
    identity_work = Path(request["identity_work"])
    _require(holdout == identity_work.parent / pretrain_corpus.HOLDOUT_RECEIPT_NAME
             and _json(holdout).get("fold") == view.fold, "F6 hold-out receipt belongs to another identity folder or fold.")
    cache = Path(request["seq_cache"])
    digest = request["seq_manifest_sha256"]
    _require(_sha(cache / "manifest.json") == digest, "F6 parent cache manifest changed.")
    root = pretrain_corpus.pretrain_dir(view)
    if not root.exists() or not any(root.iterdir()):
        if view.fold == 0:
            owned_attempt = Path(request["execution_root"]) / "runs" / _pretrain_name(request)
            base_request = {key: value for key, value in request.items() if key not in {"seq_cache", "seq_manifest_sha256"}}
            _require_pretrain_worker_marker(owned_attempt, base_request, source, view)
            _require(cache.absolute() == (owned_attempt / "gpu_cache").absolute(),
                     "Fold-0 F6 attachment must use its owned preparation cache.")
        pretrain_corpus.build_view_pretrain_corpus(view.root, official_split_dir=request["official_split_dir"],
            seq_cache_dir=cache, seq_manifest_sha256=digest, identity_work=identity_work,
            provenance=source, bound_corpus=(corpus, corpus_binding["sha256"]))
    verification = pretrain_corpus.verify_whole_protein_pretrain(view, required=True)
    inputs = pretrain_corpus.view_pretrain_inputs(view)
    receipt = _json(root / pretrain_corpus.CORPUS_RECEIPT_NAME)
    _require(inputs.corpus_sha256 == corpus_binding["sha256"]
             and receipt.get("inputs", {}).get("parent_sequence_means", {}).get("manifest_sha256") == digest,
             "An existing F6 attachment belongs to another bound corpus or parent cache.")
    return {"status": "passed", "fold": view.fold, "view_manifest_sha256": view.manifest_sha256,
            "corpus_binding": corpus_binding, "holdout_binding": holdout_binding,
            "parent_manifest_sha256": digest, "corpus_sha256": inputs.corpus_sha256,
            "manifest_sha256": inputs.manifest_sha256, "receipt_sha256": inputs.receipt_sha256,
            "verification": verification, **source_identity(source), "git_head": source["git_head"],
            "created_utc": provenance.utc_now()}


def verify_pretrain_evidence(path: Path, source: Mapping[str, Any], *, view: views.View | None = None) -> dict[str, Any]:
    """Require the persisted CUDA cache, CPU sample parity and fold-0 attachment before any G2 v2 unit."""
    _no_restricted_path(path)
    record = _json(path)
    require_record_precedence(record, source)
    _require(record.get("format") == "deepmzyme_ec_clean30_pretrain_complete_v1" and record.get("status") == "passed"
             and record.get("fold") == 0 and record.get("sequence") == ["gpu_fp32_cache", "attach", "cpu_sample_parity", "verify"],
             "G2 v2 requires completed F6 cache, parity and attachment preparation.")
    _require(record.get("parity", {}).get("passed") is True and record["parity"].get("n_compared", 0) > 0
             and record.get("attachment", {}).get("status") == "passed"
             and all(re.fullmatch(r"[0-9a-f]{64}", str(record.get("host_transfer", {}).get(key, "")))
                     for key in ("manifest_sha256", "ack_sha256")),
             "F6 preparation lacks passed parity, attachment or verified host persistence.")
    attachment, caches = record["attachment"], record.get("caches", {})
    require_same_scientific_identity(attachment, source)
    _require(all(re.fullmatch(r"[0-9a-f]{64}", str(caches.get(device, {}).get("sha256", "")))
                 for device in ("cuda", "cpu"))
             and caches["cuda"]["sha256"] == attachment.get("parent_manifest_sha256")
             and re.fullmatch(r"[0-9a-f]{64}", str(record.get("attachment_snapshot_sha256", ""))),
             "F6 preparation cache, attachment or snapshot binding differs.")
    marker = path.parent / PRETRAIN_ATTACHMENT_MARKER
    if marker.exists():
        preserved = _pretrain_marker(marker)
        _require(_sha(marker) == record["attachment_snapshot_sha256"], "The F6 snapshot marker changed after preparation.")
        require_same_scientific_identity(preserved["identity"], source)
        verify_preserved_pretrain_attachment(marker, original_attempt=preserved["identity"]["attempt_root"])
    for key, device in (("gpu_precision", "cuda"), ("cpu_precision", "cpu")):
        runtime_precision.validate_policy(record[key], device=device)
    if view is not None:
        from ec_clean30 import pretrain_corpus

        inputs = pretrain_corpus.view_pretrain_inputs(view)
        _require(view.fold == 0 and record["attachment"]["view_manifest_sha256"] == view.manifest_sha256
                 and inputs.receipt_sha256 == record["attachment"]["receipt_sha256"]
                 and inputs.manifest_sha256 == record["attachment"]["manifest_sha256"]
                 and inputs.corpus_sha256 == record["attachment"]["corpus_sha256"], "F6 attachment changed before G2.")
    return record


def _pretrain_name(request: Mapping[str, Any]) -> str:
    attempt = request.get("attempt", 1)
    _require(type(attempt) is int and attempt > 0, "F6 preparation requires a positive attempt number.")
    return f"ecc30_pretrain_fold0_attempt{attempt}"


def _pretrain_marker(path: Path) -> dict[str, Any]:
    _require(path.name == PRETRAIN_ATTACHMENT_MARKER and path.is_file() and not path.is_symlink(),
             "The F6 attachment ownership marker is missing or linked.")
    record = _json(path)
    identity = record.get("identity")
    _require(record.get("format") == PRETRAIN_ATTACHMENT_FORMAT and record.get("status") in {"pending", "restored"}
             and isinstance(identity, dict) and record.get("identity_sha256") == hashlib.sha256(_bytes(identity)).hexdigest(),
             "The F6 attachment ownership marker is invalid.")
    for name in ("attempt_root", "view_root"):
        value = identity.get(name, "")
        _require(isinstance(value, str) and PurePosixPath(value).is_absolute()
                 and ".." not in PurePosixPath(value).parts and str(PurePosixPath(value)) == value,
                 "The F6 attachment identity has an unsafe original path.")
    _require(identity.get("unit_id") == PurePosixPath(identity["attempt_root"]).name
             and isinstance(identity.get("session_id"), str) and bool(identity["session_id"])
             and all(re.fullmatch(r"[0-9a-f]{64}", str(identity.get(name, "")))
                     for name in ("view_manifest_sha256", "request_sha256")), "The F6 snapshot identity is incomplete.")
    source_identity(identity)
    return record


def _write_pretrain_marker(path: Path, record: Mapping[str, Any], *, replace: bool = False) -> None:
    target = path.with_name(path.name + ".partial") if replace else path
    if replace and target.exists():
        _require(target.is_file() and not target.is_symlink()
                 and _json(target).get("identity_sha256") == record["identity_sha256"],
                 "An interrupted F6 marker replacement has another identity.")
        target.unlink()
    with target.open("xb") as handle:
        handle.write(_bytes(record))
        handle.flush()
        os.fsync(handle.fileno())
    if replace:
        os.replace(target, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def prepare_pretrain_attachment(out: Path, *, request: Mapping[str, Any], source: Mapping[str, Any],
                                view: views.View, request_path: Path) -> Path:
    """Persist pending ownership before a child can write into the view's attached folder."""
    out.mkdir(parents=True, exist_ok=False)
    identity = {"attempt_root": str(out.absolute()), "view_root": str(view.root.absolute()),
                "view_manifest_sha256": view.manifest_sha256, "session_id": request["session_id"],
                "pod_id": request.get("pod_id"), "unit_id": _pretrain_name(request),
                "request_sha256": _sha(request_path), **source_identity(source)}
    marker = out / PRETRAIN_ATTACHMENT_MARKER
    _write_pretrain_marker(marker, {"format": PRETRAIN_ATTACHMENT_FORMAT, "status": "pending", "identity": identity,
                                  "identity_sha256": hashlib.sha256(_bytes(identity)).hexdigest()})
    descriptor = os.open(out.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return marker


def _require_pretrain_worker_marker(out: Path, request: Mapping[str, Any], source: Mapping[str, Any],
                                   view: views.View) -> dict[str, Any]:
    marker = _pretrain_marker(out / PRETRAIN_ATTACHMENT_MARKER)
    identity = marker["identity"]
    request_path = Path(request["execution_root"]) / "commands" / f"{_pretrain_name(request)}.json"
    _require(marker["status"] == "pending" and identity["attempt_root"] == str(out.absolute())
             and identity["view_root"] == str(view.root.absolute())
             and identity["view_manifest_sha256"] == view.manifest_sha256
             and identity["session_id"] == request["session_id"]
             and identity["pod_id"] == request.get("pod_id")
             and request_path.is_file() and identity["request_sha256"] == _sha(request_path)
             and _json(request_path) == dict(request), "F6 worker requires its parent's pending attachment marker.")
    require_same_scientific_identity(identity, source)
    return marker


def _attachment_inventory(root: Path) -> list[dict[str, Any]]:
    _require(root.is_dir() and not root.is_symlink(), "The F6 attachment snapshot is not a regular directory.")
    files = []
    for path in sorted(root.rglob("*")):
        mode = path.lstat().st_mode
        _require(stat.S_ISREG(mode) or stat.S_ISDIR(mode), "F6 attachment snapshots refuse links and special files.")
        if stat.S_ISREG(mode):
            files.append({"path": path.relative_to(root).as_posix(), "sha256": _sha(path), "size_bytes": path.stat().st_size})
    return files


def _fsync_attachment_snapshot(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    directories = [root, *(path for path in root.rglob("*") if path.is_dir())]
    for path in reversed(directories):
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def verify_preserved_pretrain_attachment(local_marker: Path, *, original_attempt: str | Path) -> dict[str, Any]:
    """Verify a restored attachment from an independent copy, without opening any original remote path."""
    local_marker = Path(local_marker)
    record = _pretrain_marker(local_marker)
    _require(record["status"] == "restored" and record["identity"]["attempt_root"] == str(PurePosixPath(original_attempt)),
             "F6 attachment is pending or belongs to another original attempt.")
    inventory = record.get("files")
    _require(isinstance(inventory, list) and record.get("inventory_sha256") == hashlib.sha256(_bytes(inventory)).hexdigest()
             and inventory == _attachment_inventory(local_marker.parent / "attachment_snapshot"),
             "F6 attachment snapshot inventory or bytes differ.")
    proof = record.get("termination_proof")
    _require(isinstance(proof, dict) and proof.get("session_id") == record["identity"]["session_id"]
             and proof.get("request_sha256") == record["identity"]["request_sha256"]
             and proof.get("kind") in {"owned_process_group_reaped", "provider_terminal"},
             "F6 attachment snapshot lacks its termination proof.")
    return record


def restore_pretrain_attachment(marker_path: Path, *, worker_terminated: bool,
                                termination_proof: Mapping[str, Any],
                                execution: CampaignExecution | None = None) -> dict[str, Any]:
    """Snapshot only after the owning parent reaped its child, or explicit verified provider termination.

    A pending marker is never recovered automatically. An external recovery supplies a controller receipt,
    not an assertion that a former worker probably finished.
    """
    marker_path = Path(marker_path)
    record = _pretrain_marker(marker_path)
    identity = record["identity"]
    _require(str(marker_path.parent.absolute()) == identity["attempt_root"], "Restore must run at the owned attempt.")
    if record["status"] == "restored":
        return verify_preserved_pretrain_attachment(marker_path, original_attempt=identity["attempt_root"])
    proof = dict(termination_proof)
    _require(worker_terminated is True and proof.get("session_id") == identity["session_id"]
             and proof.get("request_sha256") == identity["request_sha256"], "Verified worker termination is required before snapshotting.")
    if proof.get("kind") == "provider_terminal":
        binding = proof.get("receipt", {})
        path = Path(binding.get("path", ""))
        _require(path.is_file() and not path.is_symlink() and _sha(path) == binding.get("sha256"),
                 "F6 recovery needs the immutable controller termination receipt.")
        terminal = _json(path)
        _require(terminal.get("session_id") == identity["session_id"] and terminal.get("pod_id") == identity.get("pod_id")
                 and isinstance(identity.get("pod_id"), str)
                 and ((terminal.get("status") == "EXITED" and terminal.get("runtime_status") == "stopped")
                      or (terminal.get("status") == "DELETED" and terminal.get("provider_absence_verified") is True)),
                 "The controller receipt does not prove termination of this F6 worker's pod.")
    else:
        _require(proof.get("kind") == "owned_process_group_reaped" and proof.get("parent_pid") == os.getpid()
                 and isinstance(execution, CampaignExecution) and execution._lock is not None
                 and execution.root == marker_path.parent.parent.parent.resolve()
                 and execution.session_id == identity["session_id"]
                 and execution.state.get("owner", {}).get("pid") == os.getpid()
                 and execution.state.get("pending_unit", {}).get("run_name") == identity["unit_id"]
                 and not execution.state.get("active_child"),
                 "Only the owning parent may supply a process-group reaping proof.")
    view = views.View.load(identity["view_root"])
    _require(view.manifest_sha256 == identity["view_manifest_sha256"], "The F6 attachment view changed before snapshotting.")
    from ec_clean30 import pretrain_corpus

    attached = pretrain_corpus.pretrain_dir(view)
    before = _attachment_inventory(attached)
    staging, destination = marker_path.parent / "attachment_snapshot.partial", marker_path.parent / "attachment_snapshot"
    if staging.exists():
        _require(not staging.is_symlink(), "An interrupted F6 snapshot is linked.")
        shutil.rmtree(staging)
    if not destination.exists():
        shutil.copytree(attached, staging)
        _require(before == _attachment_inventory(staging) == _attachment_inventory(attached),
                 "F6 attachment changed while its snapshot was copied.")
        staging.rename(destination)
    _require(before == _attachment_inventory(destination), "The existing F6 snapshot differs from its attachment.")
    _fsync_attachment_snapshot(destination)
    restored = {**record, "status": "restored", "files": before,
                "inventory_sha256": hashlib.sha256(_bytes(before)).hexdigest(), "termination_proof": proof}
    _write_pretrain_marker(marker_path, restored, replace=True)
    return verify_preserved_pretrain_attachment(marker_path, original_attempt=identity["attempt_root"])


def require_restored_pretrain_attempts(root: Path) -> None:
    """Run before CampaignExecution can auto-persist an interrupted attempt or accept an acknowledgment."""
    runs = Path(root) / "runs"
    if not runs.exists():
        return
    for attempt in runs.iterdir():
        if not re.fullmatch(r"ecc30_pretrain_fold0_attempt[1-9][0-9]*", attempt.name):
            continue
        _require(attempt.is_dir() and not attempt.is_symlink(), "F6 attempt root is linked or irregular.")
        verify_preserved_pretrain_attachment(attempt / PRETRAIN_ATTACHMENT_MARKER, original_attempt=attempt.absolute())


def pretrain_worker(request: Mapping[str, Any]) -> dict[str, Any]:
    from ec_clean30 import pretrain_corpus

    source, view, _policy = _runtime(request, pretraining=True)
    _require(source.get("format") == EXECUTION_V5_FORMAT and not view.smoke,
             "Real F6 cache preparation requires the published execution-v5 source.")
    union = Path(request["union_fasta"])
    union_binding = _bound_pretrain_file(source, "inputs.f6_corpus.union_fasta", union)
    out = Path(request["execution_root"]) / "runs" / _pretrain_name(request)
    _require_pretrain_worker_marker(out, request, source, view)
    mapping = pretrain_sequence_inputs(union, out / "sequence_inputs")
    cpu_cap = request["cpu_cap_prefix"]
    _verify_container_caps(source, _json(Path(request["vm_probe"]))["container_caps_probe"], cpu_cap,
                           python=request["python"])
    caches, precision = {}, {}
    for device, fasta in (("cuda", union), ("cpu", Path(mapping["cpu_sample_fasta"]))):
        cache = out / ("gpu_cache" if device == "cuda" else "cpu_sample_cache")
        argv, env = pretrain_embedding_command(request, source, view, fasta=fasta,
            sequence_map=Path(mapping["sequence_map"]), sequence_map_sha256=mapping["sequence_map_sha256"],
            output=cache, device=device)
        log_path = out / f"{device}_embedding.log"
        with log_path.open("x") as log:
            subprocess.run([*(request["cap_prefix"] if device == "cuda" else cpu_cap), *argv], cwd=REPO_ROOT,
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        manifest = _json(cache / "manifest.json")
        pretrain_corpus.parent_precision_record(manifest, smoke=False)
        precision[device] = runtime_precision.read_policy_log(log_path, device=device)
        caches[device] = {"path": str(cache), "sha256": _sha(cache / "manifest.json")}
        if device == "cuda":
            attachment = attach_pretrain({**request, "seq_cache": caches["cuda"]["path"],
                                         "seq_manifest_sha256": caches["cuda"]["sha256"]})
    parity = pretrain_corpus.compare_sequence_mean_samples(caches["cuda"]["path"], caches["cuda"]["sha256"],
        caches["cpu"]["path"], caches["cpu"]["sha256"], mapping["sample_keys"])
    _require(parity.get("passed") is True and parity.get("n_compared") == len(mapping["sample_keys"]),
             "F6 CUDA/CPU sequence-mean parity failed.")
    pretrain_corpus.verify_whole_protein_pretrain(view, required=True)
    record = {"format": "deepmzyme_ec_clean30_pretrain_preparation_v1", "status": "passed", "fold": 0,
              "unit_id": _pretrain_name(request), "sequence": ["gpu_fp32_cache", "attach", "cpu_sample_parity", "verify"],
              "union_binding": union_binding, "caches": caches, "parity": parity, "attachment": attachment,
              "gpu_precision": precision["cuda"], "cpu_precision": precision["cpu"],
              **source_identity(source), "git_head": source["git_head"], "created_utc": provenance.utc_now()}
    _write(out / "pretrain_preparation.json", record)
    return record


def run_pretrain_preparation(request: Mapping[str, Any]) -> dict[str, Any]:
    source, view, policy = _runtime(request, pretraining=True)
    _require(source.get("format") == EXECUTION_V5_FORMAT, "F6 preparation requires execution manifest v5.")
    root, name = Path(request["execution_root"]), _pretrain_name(request)
    out = root / "runs" / name
    result_path, complete_path = out / "pretrain_preparation.json", out / "pretrain_complete.json"
    with _execution(request, policy) as execution:
        if complete_path.exists():
            return verify_pretrain_evidence(complete_path, source, view=view)
        if result_path.exists():
            record = _json(result_path)
            require_record_precedence(record, source)
            _require(execution.state.get("pending_unit", {}).get("run_name") == name
                     and execution.state["pending_unit"].get("persisted"), "F6 preparation awaits independent host acknowledgment.")
            seconds, transfer = _persistence_seconds(execution)
            _write(complete_path, {**record, "format": "deepmzyme_ec_clean30_pretrain_complete_v1",
                                  "host_transfer": transfer, "persistence_seconds": seconds,
                                  "preparation_sha256": _sha(result_path),
                                  "attachment_snapshot_sha256": _sha(out / PRETRAIN_ATTACHMENT_MARKER)})
            return verify_pretrain_evidence(complete_path, source, view=view)
        _require(not out.exists(), "An incomplete F6 preparation must be preserved before a new explicit attempt.")
        request_path = root / "commands" / f"{name}.json"
        _write(request_path, dict(request))
        execution.admit(name, float(request["forecast_seconds"]))
        marker = prepare_pretrain_attachment(out, request=request, source=source, view=view, request_path=request_path)
        started = time.monotonic()
        failure = None
        with request_path.with_suffix(".log").open("x") as log:
            try:
                result = execution.run_subprocess([*request["cap_prefix"],
                    *_child_command(request_path, "_pretrain-worker", request["python"])],
                    cwd=REPO_ROOT, env=clean_repository_environment(), stdout=log, stderr=subprocess.STDOUT,
                    timeout=float(request["unit_timeout_seconds"]))
            except Exception as exc:
                result, failure = None, exc
        passed = result is not None and result.returncode == 0 and result_path.is_file() and _json(result_path).get("status") == "passed"
        # run_subprocess has reaped its owned child on success, failure and timeout. Only now can
        # the out-of-root attached data become a stable, independently preservable snapshot.
        restore_pretrain_attachment(marker, worker_terminated=True, termination_proof={
            "kind": "owned_process_group_reaped", "parent_pid": os.getpid(),
            "session_id": request["session_id"], "request_sha256": _sha(request_path)}, execution=execution)
        execution.record_result(name, "completed" if passed else "failed_preparation", time.monotonic()-started)
        persistence = execution.persist([request_path, request_path.with_suffix(".log"), *([out] if out.exists() else [])])
        _require(passed, "F6 preparation failed; preserve its terminal attempt before retrying."
                 + (f" {type(failure).__name__}: {failure}" if failure else ""))
        return {"status": "awaiting_host_ack", "unit_id": name, "persistence": persistence}


def embedding_command(request: Mapping[str, Any], source: Mapping[str, Any], view: views.View,
                      output: Path) -> tuple[list[str], dict[str, str]]:
    """Only the two existing embedding producers, on the pinned timing subset."""
    inputs = Path(request["timing_inputs"])
    roots = view.forbidden_read_roots
    _outside_guard(inputs, roots)
    _outside_guard(output, roots)
    receipt_path = inputs / "timing_inputs.json"
    _require(source.get("timing_inputs_sha256") == _sha(receipt_path), "Timing subset is not bound by the execution manifest.")
    receipt = _json(receipt_path)
    _require(receipt.get("n_rows") == TIMING_ROWS and receipt.get("fold") == 0
             and receipt.get("source_sha256") == source.get("official_fold0_training_sha256") == OFFICIAL_FOLD0_SHA256,
             "Embedding timing is not the exact official fold-0 subset.")
    for name in ("timing.fasta", "sequence_map.tsv"):
        _require(receipt.get("files", {}).get(name) == _sha(_safe_file(inputs, name)), "Timing input changed.")
    env = clean_repository_environment()
    env.update(PYTHONPATH=os.pathsep.join((str(REPO_ROOT / "src"), str(REPO_ROOT))),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1", NVIDIA_TF32_OVERRIDE="0")
    if request["kind"] == "esmc":
        from training.access_guard import FORBIDDEN_READ_ROOTS_ENV, forbidden_roots_environment
        cache, weights = Path(request["hf_home"]).resolve(), Path(request["esmc_weights"]).resolve()
        _outside_guard(Path(request["hf_home"]), roots)
        _outside_guard(Path(request["esmc_weights"]), roots)
        _require(cache in weights.parents and weights.is_file()
                 and _esmc_cached_weight(cache) == weights
                 and re.fullmatch(r"[0-9a-f]{64}", str(source.get("frozen_esmc_weight_sha256", "")))
                 and _sha(weights) == source["frozen_esmc_weight_sha256"],
                 "ESMC weights must match Freeze 1 inside the explicit offline cache.")
        env.update(HF_HOME=str(cache), HF_HUB_CACHE=str(cache / "hub"), HUGGINGFACE_HUB_CACHE=str(cache / "hub"))
        env.pop("INFRA_PROVIDER", None)  # The SDK otherwise bypasses its HF cache and loads relative data/weights.
        env[FORBIDDEN_READ_ROOTS_ENV] = forbidden_roots_environment(roots)
        fp32 = source.get("format") == EXECUTION_V5_FORMAT
        wrapper = ("import runpy,sys\nfrom training.access_guard import install_guard_from_environment\n"
                   "if not install_guard_from_environment():\n    raise SystemExit('forbidden-read roots are required')\n"
                   + ("" if fp32 else runtime_precision.bootstrap_code(request["device"]))
                   + "sys.argv=sys.argv[1:]\nrunpy.run_module('embed_helpers.esmc_chain_means',run_name='__main__')")
        argv = [request["python"], "-c", wrapper, "esmc_chain_means", "sequences", "--fasta", str(inputs / "timing.fasta"),
                "--sequence-map", str(inputs / "sequence_map.tsv"), "--sequence-map-sha256", receipt["files"]["sequence_map.tsv"],
                "--out", str(output), "--layer-index", "26", "--device", request["device"]]
        if fp32:
            argv += ["--forward-precision", "fp32"]
    elif request["kind"] == "esm1b":
        pins = {"harness": "73c25c5396f89a603a2ca97cf2adf82bb6b220bccf4d9f45801d1bdcf60bede4",
                "extract_script": "902a69cffa0b35ab16798ba655d7b0e357d0e8efd5a268558cc4b064d2e7719d"}
        for name, expected in pins.items():
            path = Path(request[name]); _outside_guard(path, roots)
            _require(path.is_file() and _sha(path) == expected, "Private embedding helper differs from its existing pin.")
        weights = Path(request["weights"])
        _outside_guard(weights, roots)
        contact = Path(str(weights.with_suffix("")) + "-contact-regression.pt")
        _outside_guard(contact, roots)
        _require(weights.is_file() and contact.is_file(), "Local ESM-1b weights and contact-regression companion are required.")
        for path, key in ((weights, "weights_sha256"), (contact, "contact_weights_sha256")):
            _require(re.fullmatch(r"[0-9a-f]{64}", str(request.get(key, ""))) and _sha(path) == request[key],
                     "Private local weight checksum differs.")
        argv = [request["clean_python"], str(REPO_ROOT / "src/ec_clean30/runtime_precision.py"), "esm1b-farm",
                "--harness", str(request["harness"]), "--", "--runner",
                str(REPO_ROOT / "scripts/run_clean_split30_comparator.py"), "--extract-script", str(request["extract_script"]),
                "--extract-sha256", pins["extract_script"], "--weights", str(weights), "--weights-sha256", request["weights_sha256"],
                "--sequence-map", str(inputs / "sequence_map.tsv"), "--sequence-map-sha256", receipt["files"]["sequence_map.tsv"],
                "--kind", "base", "--out-dir", str(output), "--device", request["device"], "--threads", str(request["threads"])]
        for root in roots:
            argv += ["--forbidden-root", root]
    else:
        raise Phase3Error("Embedding kind must be esmc or esm1b.")
    return argv, env


def _esm1b_cuda_preflight(request: Mapping[str, Any], env: Mapping[str, str], directory: Path) -> dict[str, str]:
    """The private extractor can silently use CPU, so qualify its actual interpreter."""
    profile = request.get("hardware_profile", "gcp_l4")
    program = cuda_probe_program(profile)
    path = directory / "cuda_preflight.json"
    log = directory / "cuda_preflight.stderr.log"
    with path.open("x") as handle, log.open("x") as errors:
        subprocess.run([*request["cap_prefix"], request["clean_python"], "-c", program],
                       cwd=REPO_ROOT, env=dict(env), stdout=handle, stderr=errors,
                       check=True, timeout=min(60.0, float(request["unit_timeout_seconds"])))
    validate_cuda_evidence(_json(path), profile)
    return {"cuda_preflight_sha256": _sha(path), "cuda_preflight_log_sha256": _sha(log)}


def _esm1b_gpu_evidence(output: Path) -> dict[str, Any]:
    extraction = _json(output / "farm_receipt.json").get("extraction", {})
    path = output / "extract.log"
    _require(extraction.get("device") == "cuda" and extraction.get("returncode") == 0
             and extraction.get("log") == "extract.log" and path.is_file()
             and extraction.get("log_sha256") == _sha(path)
             and "Transferred model to GPU" in path.read_text().splitlines(),
             "ESM-1b extraction lacks hash-bound evidence that its model actually used CUDA.")
    try:
        policy = runtime_precision.read_policy_log(path, device="cuda")
    except RuntimeError as exc:
        raise Phase3Error(f"ESM-1b child precision policy is invalid: {exc}") from exc
    return {"extraction_log_sha256": _sha(path), "precision_policy": policy}


def embedding_worker(request: Mapping[str, Any]) -> dict[str, Any]:
    source, view, _ = _runtime(request)
    name = f"ecc30timing_embed_{request['kind']}_attempt{request.get('attempt', 1)}"
    directory = Path(request["execution_root"]) / "runs" / name
    directory.mkdir(parents=True)
    started = time.monotonic()
    argv, env = embedding_command(request, source, view, directory / "embeddings")
    env = clean_repository_environment(env)
    device_evidence = {}
    if request["kind"] == "esm1b" and request["device"] == "cuda":
        device_evidence.update(_esm1b_cuda_preflight(request, env, directory))
    with (directory / "embedding.log").open("x") as log:
        subprocess.run([*request["cap_prefix"], *argv], cwd=REPO_ROOT, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    receipt = directory / "embeddings" / ("build_receipt.json" if request["kind"] == "esmc" else "farm_receipt.json")
    _require(receipt.is_file(), "Embedding producer did not leave its completed receipt.")
    if request["kind"] == "esm1b" and request["device"] == "cuda":
        device_evidence.update(_esm1b_gpu_evidence(directory / "embeddings"))
    elif request["kind"] == "esmc":
        path = directory / "embedding.log"
        device_evidence.update(precision_policy=runtime_precision.read_policy_log(path, device=request["device"]),
                               child_log_sha256=_sha(path))
    result = {"format": "deepmzyme_ec_clean30_embedding_timing_v1", "status": "passed", "kind": request["kind"],
              "unit_id": name, "source": source, "execution_manifest_sha256": source["execution_manifest_sha256"],
              **source_identity(source), "git_head": source.get("git_head"), "created_utc": provenance.utc_now(),
              "producer_receipt_sha256": _sha(receipt), "wall_seconds": time.monotonic()-started,
              "n_input_sequences": TIMING_ROWS, "device_evidence": device_evidence}
    _write(directory / "embedding_timing.json", result)
    return result


def embed_timing(request: Mapping[str, Any]) -> dict[str, Any]:
    source, _view, policy = _runtime(request)
    _require(request.get("kind") in {"esmc", "esm1b"} and type(request.get("attempt", 1)) is int
             and request.get("attempt", 1) > 0, "Unknown embedding timing or attempt.")
    name = f"ecc30timing_embed_{request['kind']}_attempt{request.get('attempt', 1)}"
    root = Path(request["execution_root"])
    directory = root / "runs" / name
    timing = directory / "embedding_timing.json"
    with _execution(request, policy) as execution:
        if timing.exists():
            row = _json(timing)
            _require(row.get("status") == "passed", "Embedding source or status differs.")
            if source.get("format") == EXECUTION_V5_FORMAT:
                require_record_precedence(row, source)
            else:
                _require(row.get("source") == source, "Embedding source or status differs.")
            persistence_seconds, transfer = _persistence_seconds(execution)
            _require(execution.state.get("pending_unit", {}).get("run_name") == name
                     and execution.state["pending_unit"].get("persisted"), "Embedding persistence is not acknowledged.")
            complete = directory / "embedding_complete.json"
            if not complete.exists():
                _write(complete, {**row, "format": "deepmzyme_ec_clean30_completed_embedding_v1",
                                 "persistence_seconds": persistence_seconds, "host_transfer": transfer,
                                 "wall_seconds": float(row["wall_seconds"])+persistence_seconds,
                                 "timing_sha256": _sha(timing)})
            return _json(complete)
        request_file = root / "commands" / f"{name}.json"
        _write(request_file, dict(request))
        execution.admit(name, float(request["forecast_seconds"]))
        started = time.monotonic()
        with (root / "commands" / f"{name}.log").open("x") as log:
            result = execution.run_subprocess(_child_command(request_file, "_embedding-worker", request["python"]),
                cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, env=clean_repository_environment(),
                timeout=float(request["unit_timeout_seconds"]))
        passed = result.returncode == 0 and timing.is_file() and _json(timing).get("status") == "passed"
        execution.record_result(name, "completed" if passed else "failed_embedding", time.monotonic()-started)
        paths = [request_file, root / "commands" / f"{name}.log"] + ([directory] if directory.exists() else [])
        persistence = execution.persist(paths)
        _require(passed, "Embedding timing failed; acknowledge its saved attempt before retrying.")
        return {"status": "awaiting_host_ack", "kind": request["kind"], "persistence": persistence}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    pack = actions.add_parser("pack-view")
    pack.add_argument("--view", type=Path, required=True)
    pack.add_argument("--archive", type=Path, required=True)
    pack.add_argument("--smoke", action="store_true")
    pack.add_argument("--fold", type=int, choices=range(5), default=0)
    unpack = actions.add_parser("unpack-view")
    unpack.add_argument("--archive", type=Path, required=True)
    unpack.add_argument("--destination", type=Path, required=True)
    unpack.add_argument("--sha256", required=True)
    unpack.add_argument("--smoke", action="store_true")
    unpack.add_argument("--fold", type=int, choices=range(5), default=0)
    inputs = actions.add_parser("timing-inputs")
    inputs.add_argument("--training-file", type=Path, required=True)
    inputs.add_argument("--sha256", required=True)
    inputs.add_argument("--out", type=Path, required=True)
    source = actions.add_parser("execution-manifest")
    source.add_argument("--amendment", type=Path, required=True)
    source.add_argument("--amendment-commit", required=True)
    source.add_argument("--freeze1b-sha256", required=True)
    source.add_argument("--timing-inputs-sha256", required=True)
    source.add_argument("--out", type=Path, required=True)
    automation = actions.add_parser("automation-manifest")
    automation.add_argument("--base-execution", type=Path, required=True)
    automation.add_argument("--base-commit", required=True)
    automation.add_argument("--out", type=Path, required=True)
    runpod = actions.add_parser("runpod-manifest")
    runpod.add_argument("--base-automation", type=Path, required=True)
    runpod.add_argument("--base-commit", required=True)
    runpod.add_argument("--out", type=Path, required=True)
    bootstrap = actions.add_parser("bootstrap-execution-manifest")
    bootstrap.add_argument("--base-runpod", type=Path, required=True)
    bootstrap.add_argument("--base-commit", required=True)
    bootstrap.add_argument("--out", type=Path, required=True)
    v5 = actions.add_parser("execution-v5-manifest")
    v5.add_argument("--amendment-chain-receipt", type=Path, required=True)
    v5.add_argument("--expected-chain", action="append", required=True)
    v5.add_argument("--timing-inputs-sha256", required=True)
    v5.add_argument("--out", type=Path, required=True)
    v5.add_argument("--runtime-locks-path", default=RUNTIME_LOCKS_PATH)
    verify = actions.add_parser("verify-execution")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--draft", action="store_true")
    verify.add_argument("--amendment-chain-receipt", type=Path)
    verify.add_argument("--expected-chain", action="append")
    probe = actions.add_parser("vm-probe")
    probe.add_argument("--view", type=Path, required=True)
    probe.add_argument("--execution-manifest", type=Path, required=True)
    probe.add_argument("--out", type=Path, required=True)
    probe.add_argument("--cap-json", type=Path, required=True, help="JSON array of the approved runtime cap argv.")
    probe.add_argument("--python", default=sys.executable)
    probe.add_argument("--hardware-profile", choices=sorted(GPU_PROFILES), default="gcp_l4")
    probe.add_argument("--cap-probe", type=Path)
    probe.add_argument("--smoke", action="store_true")
    probe.add_argument("--amendment-chain-receipt", type=Path)
    probe.add_argument("--expected-chain", action="append")
    for name in ("timing-unit", "lane-gate", "embed-timing", "_timing-worker", "_lane-worker", "_embedding-worker",
                 "pretrain-prepare", "_pretrain-worker", "pretrain-attach"):
        action = actions.add_parser(name)
        action.add_argument("--request", type=Path, required=True)
    ack = actions.add_parser("ack-host")
    ack.add_argument("--manifest", type=Path, required=True)
    ack.add_argument("--downloaded-root", type=Path, required=True)
    ack.add_argument("--ack", type=Path, required=True)
    ack.add_argument("--transfer-seconds", type=float, required=True)
    final = actions.add_parser("freeze1c")
    final.add_argument("--execution-manifest", type=Path, required=True)
    final.add_argument("--evidence", type=Path, action="append", required=True)
    final.add_argument("--l3-ceilings", type=Path, help="Legacy v1 only: JSON fold keys 0..4 and proposed hours.")
    final.add_argument("--amendment-chain-receipt", type=Path)
    final.add_argument("--expected-chain", action="append")
    final.add_argument("--pretrain-evidence", type=Path)
    final.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    for key in tuple(os.environ):
        if key.startswith("GIT_"):
            del os.environ[key]
    args = build_parser().parse_args(argv)
    try:
        if args.action == "pack-view":
            result = pack_view(args.view, args.archive, smoke=args.smoke, fold=args.fold)
        elif args.action == "unpack-view":
            result = unpack_view(args.archive, args.destination, expected_sha256=args.sha256, smoke=args.smoke, fold=args.fold)
        elif args.action == "timing-inputs":
            result = timing_inputs(args.training_file, args.out, expected_sha256=args.sha256)
        elif args.action == "execution-manifest":
            result = make_execution_manifest(args.amendment, args.out, amendment_commit=args.amendment_commit,
                freeze1b_sha256=args.freeze1b_sha256, timing_inputs_sha256=args.timing_inputs_sha256)
        elif args.action == "automation-manifest":
            result = make_automation_manifest(args.base_execution, args.out, base_commit=args.base_commit)
        elif args.action == "runpod-manifest":
            result = make_runpod_manifest(args.base_automation, args.out, base_commit=args.base_commit)
        elif args.action == "bootstrap-execution-manifest":
            result = make_bootstrap_manifest(args.base_runpod, args.out, base_commit=args.base_commit)
        elif args.action == "execution-v5-manifest":
            result = make_execution_v5_manifest(args.amendment_chain_receipt, args.out,
                timing_inputs_sha256=args.timing_inputs_sha256, expected_chain=args.expected_chain,
                runtime_locks_path=args.runtime_locks_path)
        elif args.action == "verify-execution":
            result = verify_execution_source(args.manifest, require_published=not args.draft,
                chain_receipt_path=args.amendment_chain_receipt, expected_chain=args.expected_chain)
        elif args.action == "vm-probe":
            result = vm_probe(args.view, args.execution_manifest, args.out, cap_prefix=_json(args.cap_json),
                              python=args.python, smoke=args.smoke, hardware_profile=args.hardware_profile,
                              cap_probe_path=args.cap_probe, chain_receipt_path=args.amendment_chain_receipt,
                              expected_chain=args.expected_chain)
        elif args.action == "ack-host":
            result = host_ack(args.manifest, args.downloaded_root, args.ack, transfer_seconds=args.transfer_seconds)
        elif args.action == "freeze1c":
            result = build_freeze1c(args.evidence, args.out, execution_manifest_path=args.execution_manifest,
                l3_vm_hours_per_fold=_json(args.l3_ceilings) if args.l3_ceilings else None,
                chain_receipt_path=args.amendment_chain_receipt, expected_chain=args.expected_chain,
                pretrain_evidence=args.pretrain_evidence)
        else:
            _no_restricted_path(args.request)
            action = {"timing-unit": run_timing_unit, "lane-gate": run_lane_gate, "embed-timing": embed_timing,
                      "_timing-worker": timing_worker, "_lane-worker": lane_worker, "_embedding-worker": embedding_worker,
                      "pretrain-prepare": run_pretrain_preparation, "_pretrain-worker": pretrain_worker,
                      "pretrain-attach": attach_pretrain}[args.action]
            result = action(_json(args.request))
        public = {key: result[key] for key in ("status", "format", "kind", "unit_id", "recipe", "archive_sha256",
                  "archive_bytes", "unpacked_bytes", "files", "n_rows", "published", "execution_manifest_sha256",
                  "lanes", "speedup") if key in result and (key != "files" or isinstance(result[key], int))}
        print(json.dumps(public, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(f"Phase 3 refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
