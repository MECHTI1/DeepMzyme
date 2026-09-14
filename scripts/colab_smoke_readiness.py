"""Run from the existing notebook after dataset detection; never trains a model."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import torch
from training.runtime_preparation import (
    discover_missing_esm_embeddings,
    discover_missing_updated_external_features,
)
from training.esm_feature_loading import summarize_esm_embedding_metadata
from training.labels import parse_structure_identity
from structure_store import resolve_structure_files


assert CONFIG["basic"]["task"] == "metal"
assert not CONFIG["basic"]["run_held_out_test_eval"]
assert STANDALONE_PHASE == "smoke" and EPOCHS == 1 and MAX_CONFIGURATION_RUNS == 1
assert MODEL_PRESET == "Only-GVP" and METAL_LABEL_SCHEME == "four_class"
assert DEVICE == "cuda" and torch.cuda.is_available()
assert not LAUNCH_PLANNED_MAIN_TRAINING_RUNS
probe = torch.ones(32, device="cuda", requires_grad=True)
probe.square().sum().backward()
torch.cuda.synchronize()
runtime = {
    "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
    "cuda": torch.version.cuda, "capability": list(torch.cuda.get_device_capability(0)),
    "cuda_forward_backward": bool(probe.grad.is_cuda),
    "nvidia_smi": subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], text=True).strip(),
}
print(json.dumps(runtime, indent=2))
drive_root = Path(DRIVE_ROOT)
assert Path("/content/drive/MyDrive").is_dir(), "Authorize Google Drive in data setup first."
assert Path(RUNS_DIR) == drive_root / "notebook_outputs/runs", "Outputs must persist directly to Drive."
persist = Path(RUNS_DIR) / RUN_BATCH_ID
persist.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile(dir=persist, mode="w+") as handle:
    handle.write("DeepMzyme persistence check")
    handle.flush()
    handle.seek(0)
    assert handle.read() == "DeepMzyme persistence check"

expected = json.loads((CHAT4_SNAPSHOT_ROOT / "colab_expected_metal_inputs.json").read_text())
assert DATASET_ROOT.name == expected["dataset"]
csv_digest = hashlib.sha256(TRAIN_SITE_SUMMARY_CSV.read_bytes()).hexdigest()
assert csv_digest == expected["summary_sha256"], "Colab train CSV differs from prepared local inputs."
observed = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in TRAIN_STRUCTURES}
assert observed == expected["train_structures"], "Colab training structures differ from the prepared cohort."
train_groups = {parse_structure_identity(path.stem)[0] for path in TRAIN_STRUCTURES}
test_groups = {parse_structure_identity(path.stem)[0] for path in TEST_STRUCTURES}
assert not train_groups.intersection(test_groups), "Train/test group overlap."
esm_root = Path(ESM_EMBEDDINGS_DIR)
external_root = Path(EXTERNAL_FEATURES_ROOT_DIR)
missing_esm = discover_missing_esm_embeddings(TRAIN_STRUCTURES, esm_root)
missing_external = discover_missing_updated_external_features(
    TRAIN_STRUCTURES, structure_root=TRAIN_DIR, external_features_root_dir=external_root)
metadata = summarize_esm_embedding_metadata(TRAIN_STRUCTURES, esm_root)
report = {
    "kind": "Colab_input_and_CUDA_preflight_no_training_or_test_inference",
    "runtime": runtime, "snapshot_sha256": CHAT4_SNAPSHOT_DIGEST,
    "train_csv_sha256": csv_digest, "train_structures": len(TRAIN_STRUCTURES),
    "train_groups": len(train_groups), "group_overlap": [],
    "missing_esm": [p.name for p in missing_esm],
    "missing_external": [p.name for p in missing_external], "esm_metadata": metadata,
    "held_out_evaluation_enabled": False, "persistent_output_dir": str(persist),
}
(persist / "colab_smoke_readiness.json").write_text(json.dumps(report, indent=2) + "\n")
assert not missing_esm and not missing_external, "Complete ESM/external coverage is required for all families."
assert metadata["metadata_sidecars_missing"] == 0
assert metadata["esm_model_names"] == ["esmc_300m"] and metadata["embedding_dims"] == [960]

# Check existing EC train caches before any proposal to regenerate features.
# Only the configured project storage is searched; no EC training or test evaluation.
care = DATA_ROOT / "CARE_task1_30_clusterRes30_train_test_metallo/train"
ec_report = {"training_enabled": False, "dataset_train_dir": str(care), "storage_candidates": []}
if care.is_dir():
    ec_structures = resolve_structure_files(care, recursive_legacy_scan=False)
    storage_roots = [DATA_ROOT, drive_root / "DeepMzyme_Data", drive_root]
    for storage in storage_roots:
        eroot, froot = storage / "esm_embeddings", storage / "updated_feature_extraction"
        ec_report["storage_candidates"].append({
            "root": str(storage), "exists": storage.exists(),
            "esm_dir_exists": eroot.is_dir(), "external_dir_exists": froot.is_dir(),
            "train_structures": len(ec_structures),
            "missing_esm": [p.name for p in discover_missing_esm_embeddings(ec_structures, eroot)],
            "missing_external": [p.name for p in discover_missing_updated_external_features(
                ec_structures, structure_root=care, external_features_root_dir=froot)],
        })
(persist / "ec_storage_inventory.json").write_text(json.dumps(ec_report, indent=2) + "\n")
print("EC remains blocked. Existing storage counts:", [
    (r["root"], len(r["missing_esm"]), len(r["missing_external"]))
    for r in ec_report["storage_candidates"]])
for filename in ("code_snapshot_manifest.json", "deepmzyme-chat4-code.tar.gz"):
    shutil.copy2(CHAT4_SNAPSHOT_ROOT / filename, persist / filename)
shutil.copy2(CHAT4_SNAPSHOT_ROOT / "notebooks/DeepMzyme_training_colab.ipynb", persist / "prepared_notebook.ipynb")
print("Verified metal inputs, CUDA and persistent output:", persist)
