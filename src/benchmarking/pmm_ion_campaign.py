"""Frozen profile, fold membership, commands and run identity for the PMM ion campaign.

Plan Step 3. The generalized fivefold runner (``scripts/run_metal_5fold_cv.py``)
delegates here when given ``--campaign-dir``; there is no second orchestrator.

Grid (one model seed, PDB-grouped folds, validation only):
three baseline families x {four_class, six_class} x 5 folds, plus the
``first_shell_bias`` readout for each family's direct-four arm x 5 folds = 45 fits.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch  # noqa: E402

from data_structures import PocketRecord  # noqa: E402
from training.access_guard import FORBIDDEN_READ_ROOTS_ENV, forbidden_roots_environment  # noqa: E402
from training.campaign_runtime import FOLD_MEMBERSHIP_COLUMNS, membership_identity_sha256  # noqa: E402
from training.source_cohort import CohortBinding, read_cohort_csv, sha256_file  # noqa: E402
from training.splits import split_pockets_k_fold  # noqa: E402

CAMPAIGN_ID = "pmm_ion_metal_v2_context"
PROFILE_SCHEMA_VERSION = 1
N_FOLDS = 5
SPLIT_SEED = 42
MODEL_SEEDS = (42,)
NATIVE_ELEMENTS = ("MN", "CU", "ZN", "FE", "CO", "NI")
COMMON_FOUR = {"MN": "Mn", "CU": "Cu", "ZN": "Zn", "FE": "Class VIII", "CO": "Class VIII", "NI": "Class VIII"}
OMITTED_EXTERNAL_FEATURES = ("biotite_residue_sasa", "custom_charge_distance_proxy", "dpka_titr")
ESM_MODEL_NAME = "esmc_600m"
ESM_DIM = 1152

# Frozen campaign settings (plan Step 3 tables). Changing any value is a new campaign.
PROFILE: dict[str, Any] = {
    "campaign_id": CAMPAIGN_ID,
    "schema_version": PROFILE_SCHEMA_VERSION,
    "task": "metal",
    "metal_example_unit": "ion",
    "n_folds": N_FOLDS,
    "train_val_split_by": "pdbid",
    "split_seed": SPLIT_SEED,
    "split_stratify_by": "metal_site",
    "model_seeds": list(MODEL_SEEDS),
    "seed_design": "one-seed grouped-fold comparison (not seed-repeat confirmation)",
    "epochs": 50,
    "batch_size": 16,
    "optimizer": "AdamW",
    "weight_decay": 1e-4,
    "lr_schedule": "fixed",
    "selection_metric": "val_metal_balanced_acc",
    "tie_rule": "earliest epoch",
    "train_metrics_every_n_epochs": 10,
    "pocket_radius": 10.0,
    "edge_radius": 8.0,
    "metal_node_mode": "none",
    "structural_readout_scope": "residue_only",
    "shell_role_source": "geometry",
    "use_ring_edges": False,
    "hidden_s": 128,
    "hidden_v": 16,
    "edge_hidden": 64,
    "gvp_layers": 4,
    "esm_fusion_dim": 128,
    "head_mlp_layers": 2,
    "head_mlp_dropout": 0.2,
    "esm_graph_encoder_dropout": 0.1,
    "node_feature_set": "conservative",
    "omit_node_features": list(OMITTED_EXTERNAL_FEATURES),
    "esm_model_name": ESM_MODEL_NAME,
    "esm_dim": ESM_DIM,
    "metal_class_weight_mode": "manual_common_four_equalized",
    "metal_loss_function": "cross_entropy",
    "metal_label_smoothing": 0.0,
    "metal_collapsed_loss_weight": 0.0,
    "sampler": "ordinary shuffled",
    "evaluate_test": False,
}

FAMILIES: dict[str, dict[str, Any]] = {
    "only_esm": {"architecture": "only_esm", "fusion_mode": None, "uses_esm": True, "lr": "3e-5",
                 "gvp_lr": None, "rbf_raw": False},
    "only_gvp": {"architecture": "only_gvp", "fusion_mode": None, "uses_esm": False, "lr": "3e-4",
                 "gvp_lr": None, "rbf_raw": True},
    "gvp_late_fusion": {"architecture": "gvp", "fusion_mode": "late_fusion", "uses_esm": True, "lr": "3e-5",
                        "gvp_lr": "3e-4", "rbf_raw": True},
}
TARGET_SCHEMES = {"four_class": "merge_fe_class_viii", "six_class": "split_all_metals"}
READOUTS = ("none", "first_shell_bias")


@dataclass(frozen=True)
class GridConfig:
    family: str
    target: str
    readout: str

    @property
    def config_id(self) -> str:
        return f"{self.family}__{self.target}__{self.readout}"


def grid_configs() -> list[GridConfig]:
    configs = [GridConfig(family, target, "none") for family in FAMILIES for target in TARGET_SCHEMES]
    configs += [GridConfig(family, "four_class", "first_shell_bias") for family in FAMILIES]
    return configs


PREDECLARED_CONTRASTS = tuple(
    [(f"{family}: six_class vs four_class", GridConfig(family, "four_class", "none"),
      GridConfig(family, "six_class", "none")) for family in FAMILIES]
    + [(f"{family}: first_shell_bias vs ordinary readout", GridConfig(family, "four_class", "none"),
        GridConfig(family, "four_class", "first_shell_bias")) for family in FAMILIES]
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Campaign layout and identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CampaignPaths:
    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    @property
    def manifest(self) -> Path:
        return self.root / "campaign_manifest.json"

    @property
    def cohort(self) -> Path:
        return self.root / "train_cohort.csv"

    @property
    def fold_membership(self) -> Path:
        return self.root / "fold_membership.csv"

    @property
    def fold_class_weights(self) -> Path:
        return self.root / "fold_class_weights.json"

    @property
    def feature_inventory(self) -> Path:
        return self.root / "feature_inventory.json"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def commands(self) -> Path:
        return self.root / "commands"

    @property
    def empty_external_features(self) -> Path:
        return self.root / "inputs" / "external_features_intentionally_empty"

    @property
    def empty_esm(self) -> Path:
        return self.root / "inputs" / "esm_not_used_by_only_gvp"

    @property
    def parse_cache(self) -> Path:
        return self.root / "parse_cache"


def load_campaign(paths: CampaignPaths) -> dict[str, Any]:
    manifest = read_json(paths.manifest)
    if manifest.get("campaign_id") not in {CAMPAIGN_ID, "pmm_ion_metal_v1"}:
        raise ValueError(f"{paths.manifest} is not a {CAMPAIGN_ID} campaign")
    if sha256_file(paths.cohort) != manifest["cohort"]["sha256"]:
        raise ValueError("train_cohort.csv changed after the campaign was frozen")
    return manifest


def forbidden_read_roots(train_dir: Path) -> list[Path]:
    """Held-out and mixed-side paths a development process must never read."""
    dataset_root = Path(train_dir).resolve().parent
    return [dataset_root / "test", dataset_root / "site_crosswalk.csv",
            # PinMyMetal's held-out source rows are tracked in the repository.
            REPO_ROOT / "prepare_training_and_test_set" / "pinmymetal_files" / "classmodel_test_set"]


# ---------------------------------------------------------------------------
# Fold membership and class weights
# ---------------------------------------------------------------------------


def _split_proxy(binding: CohortBinding) -> PocketRecord:
    """Label-free proxy with exactly the fields ``split_pockets_k_fold`` reads."""
    return PocketRecord(
        structure_id=binding.structure_stem,
        pocket_id=binding.example_id(),
        metal_element=binding.native_element,
        metal_coords=[torch.tensor(binding.coord)],
        residues=[],
        metadata={
            "source_uid": binding.source_uid,
            "matched_summary_site_metal_types": [binding.native_element],
            "metal_symbols_observed": [binding.native_element],
        },
    )


def compute_fold_membership(bindings: list[CohortBinding]) -> list[dict[str, Any]]:
    """Reuse the training split function on the frozen cohort (target-scheme independent)."""
    proxies = [_split_proxy(binding) for binding in bindings]
    fold_of: dict[str, int] = {}
    for fold in range(N_FOLDS):
        split = split_pockets_k_fold(
            proxies, n_folds=N_FOLDS, fold_index=fold, split_by=PROFILE["train_val_split_by"],
            seed=SPLIT_SEED, task="metal", stratify_by=PROFILE["split_stratify_by"],
        )
        for pocket in split.val_pockets:
            uid = pocket.metadata["source_uid"]
            if uid in fold_of:
                raise AssertionError(f"{uid} assigned to two validation folds")
            fold_of[uid] = fold
    if len(fold_of) != len(bindings):
        raise AssertionError("Some cohort rows received no validation fold")
    rows = [{"source_uid": b.source_uid, "physical_ion_id": b.physical_ion_id, "group_id": b.group_id,
             "native_element": b.native_element, "fold": fold_of[b.source_uid]} for b in bindings]
    groups: dict[str, set[int]] = {}
    for row in rows:
        groups.setdefault(row["group_id"], set()).add(row["fold"])
    split_groups = [group for group, folds in groups.items() if len(folds) > 1]
    if split_groups:
        raise AssertionError(f"PDB groups split across folds: {split_groups[:5]}")
    return rows


def fold_class_weights(rows: list[dict[str, Any]], *, strict: bool = True) -> dict[str, Any]:
    """Common-four equalized weights per training fold, applied identically to both schemes.

    ``w_c = N_train / (4 * n_c)`` over the common four classes. The six-class arm
    assigns ``w_VIII`` to Fe, Co and Ni separately, so Class VIII's aggregate
    training weight is unchanged by splitting its output.
    """
    payload: dict[str, Any] = {"rule": "w_c = N_train / (4 * n_c) on the common-four endpoint", "folds": {}}
    for fold in range(N_FOLDS):
        train = [row for row in rows if int(row["fold"]) != fold]
        val = [row for row in rows if int(row["fold"]) == fold]
        native_train = Counter(row["native_element"] for row in train)
        native_val = Counter(row["native_element"] for row in val)
        absent = [element for element in NATIVE_ELEMENTS if not native_train[element] or not native_val[element]]
        if absent:
            if strict:
                raise ValueError(f"Fold {fold}: native classes absent from training or validation: {absent}")
            # A small smoke subset may leave a class out of some folds; such folds are not runnable.
            payload["folds"][str(fold)] = {"n_train": len(train), "n_val": len(val), "runnable": False,
                                           "absent_native_classes": absent}
            continue
        four_counts = Counter(COMMON_FOUR[row["native_element"]] for row in train)
        weights = {name: len(train) / (4.0 * four_counts[name]) for name in ("Mn", "Cu", "Zn", "Class VIII")}
        payload["folds"][str(fold)] = {
            "n_train": len(train), "n_val": len(val), "runnable": True, "absent_native_classes": [],
            "train_native_counts": dict(sorted(native_train.items())),
            "val_native_counts": dict(sorted(native_val.items())),
            "train_common_four_counts": dict(sorted(four_counts.items())),
            "common_four_weights": weights,
            "four_class_multipliers": {"mn": weights["Mn"], "cu": weights["Cu"], "zn": weights["Zn"],
                                       "class_viii": weights["Class VIII"]},
            "six_class_multipliers": {"mn": weights["Mn"], "cu": weights["Cu"], "zn": weights["Zn"],
                                      "fe": weights["Class VIII"], "co": weights["Class VIII"],
                                      "ni": weights["Class VIII"]},
        }
    return payload


def runnable_folds(paths: CampaignPaths) -> list[int]:
    weights = read_json(paths.fold_class_weights)["folds"]
    return [int(fold) for fold, entry in sorted(weights.items()) if entry.get("runnable", True)]


def freeze_folds(paths: CampaignPaths, *, strict: bool = True) -> dict[str, Any]:
    manifest = load_campaign(paths)
    bindings = read_cohort_csv(paths.cohort, expected_sha256=manifest["cohort"]["sha256"])
    rows = compute_fold_membership(bindings)
    if paths.fold_membership.exists():
        with paths.fold_membership.open(encoding="utf-8", newline="") as handle:
            existing = list(csv.DictReader(handle))
        if membership_identity_sha256(existing) != membership_identity_sha256(rows):
            raise ValueError("Recomputed folds differ from the frozen fold_membership.csv")
    else:
        with paths.fold_membership.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(FOLD_MEMBERSHIP_COLUMNS))
            writer.writeheader()
            writer.writerows(rows)
    weights = fold_class_weights(rows, strict=strict)
    weights["fold_membership_sha256"] = sha256_file(paths.fold_membership)
    weights["membership_identity_sha256"] = membership_identity_sha256(rows)
    write_json(paths.fold_class_weights, weights)
    return weights


# ---------------------------------------------------------------------------
# Commands and run identity
# ---------------------------------------------------------------------------


def run_name(config: GridConfig, fold: int, seed: int) -> str:
    return f"{config.config_id}__fold{fold}__seed{seed}"


# Resolved-config keys that do not change the scientific fit: runtime placement,
# relocatable paths (their contents are bound by the cohort/fold/inventory hashes)
# and I/O-only switches.
NON_IDENTITY_CONFIG_KEYS = frozenset({
    "device", "runs_dir", "run_name", "load_workers", "num_workers", "pin_memory", "campaign_run_identity",
    "save_epoch_checkpoints", "structure_dir", "summary_csv", "source_cohort_csv", "fold_membership_csv",
    "esm_embeddings_dir", "external_features_root_dir", "ring_features_dir", "test_structure_dir",
    "test_summary_csv", "log_per_class_metrics",
})
# Inventory fields that change on every certification without changing its content.
VOLATILE_INVENTORY_KEYS = frozenset({"certified_at", "load_seconds"})
# Identity fields a completed-grid assessment may not require to equal the current tree.
ASSESSMENT_IGNORED_IDENTITY_KEYS = ("source_tree_sha256",)


def inventory_identity_sha256(paths: CampaignPaths) -> str | None:
    """Content identity of the feature inventory, stable across re-certification timestamps."""
    if not paths.feature_inventory.exists():
        return None
    inventory = read_json(paths.feature_inventory)
    return stable_hash({key: value for key, value in inventory.items() if key not in VOLATILE_INVENTORY_KEYS})


def source_tree_sha256() -> str:
    """sha256 over sorted src/**/*.py and scripts/*.py (path + content); binds reuse to the code."""
    digest = hashlib.sha256()
    for path in sorted([*SRC_ROOT.rglob("*.py"), *(REPO_ROOT / "scripts").glob("*.py")]):
        if "__pycache__" in path.parts:
            continue
        digest.update(str(path.relative_to(REPO_ROOT)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def identity_matches(recorded: dict[str, Any] | None, expected: dict[str, Any], ignore: tuple[str, ...] = ()) -> bool:
    if not isinstance(recorded, dict):
        return False
    strip = lambda item: {key: value for key, value in item.items() if key not in ignore}  # noqa: E731
    return strip(recorded) == strip(expected)


def build_train_command(
    paths: CampaignPaths,
    *,
    python_bin: str,
    train_dir: Path,
    config: GridConfig,
    fold: int,
    seed: int,
    device: str,
    runs_dir: Path,
    epochs: int | None = None,
    load_workers: int | None = None,
    save_epoch_checkpoints: bool = True,
    smoke: bool = False,
) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    from training.config import config_to_payload, parse_args

    family = FAMILIES[config.family]
    manifest = read_json(paths.manifest)
    weights = read_json(paths.fold_class_weights)["folds"][str(fold)]
    if not weights.get("runnable", True):
        raise ValueError(f"Fold {fold} is not runnable: absent native classes {weights['absent_native_classes']}")
    multipliers = weights["four_class_multipliers" if config.target == "four_class" else "six_class_multipliers"]
    esm_dir = None
    inventory = read_json(paths.feature_inventory)
    if not inventory.get("certified"):
        raise ValueError("Fitting requires a certified feature_inventory.json")
    if not smoke and inventory.get("certified_scope") != "structure_and_esmc600m":
        # Recertifying later would change every identity; grid fits wait for the complete inventory.
        raise ValueError("Grid fits require the complete ESMC-600M-certified feature inventory")
    identity = {
        "campaign_id": manifest["campaign_id"],
        "profile_sha256": stable_hash({**PROFILE, "campaign_id": manifest["campaign_id"]}),
        "cohort_sha256": manifest["cohort"]["sha256"],
        "fold_membership_sha256": sha256_file(paths.fold_membership),
        "feature_inventory_sha256": inventory_identity_sha256(paths),
        "source_tree_sha256": source_tree_sha256(),
        "family": config.family,
        "target_scheme": config.target,
        "readout": config.readout,
        "fold": int(fold),
        "model_seed": int(seed),
        "epochs": int(epochs if epochs is not None else PROFILE["epochs"]),
        "smoke": bool(smoke),
    }
    if family["uses_esm"]:
        if inventory.get("esm") is None:
            raise ValueError("ESM families require an ESMC-600M-certified feature_inventory.json")
        esm_dir = inventory["esm"]["embeddings_dir"]
    command = [
        python_bin, "-u", str(REPO_ROOT / "src" / "train.py"),
        "--task", "metal",
        "--metal-example-unit", "ion",
        "--metal-label-scheme", TARGET_SCHEMES[config.target],
        "--metal-eligibility-scheme", "six_class",
        "--structure-dir", str(train_dir),
        # Recorded for provenance only: cohort mode never matches sites through the summary table.
        "--summary-csv", str(Path(train_dir) / "final_data_summarazing_table.csv"),
        "--source-cohort-csv", str(paths.cohort),
        "--source-cohort-sha256", manifest["cohort"]["sha256"],
        "--fold-membership-csv", str(paths.fold_membership),
        "--fold-membership-sha256", identity["fold_membership_sha256"],
        "--n-folds", str(N_FOLDS),
        "--fold-index", str(fold),
        "--train-val-split-by", PROFILE["train_val_split_by"],
        "--split-seed", str(SPLIT_SEED),
        "--split-stratify-by", PROFILE["split_stratify_by"],
        "--seed", str(seed),
        "--epochs", str(identity["epochs"]),
        "--batch-size", str(PROFILE["batch_size"]),
        "--weight-decay", str(PROFILE["weight_decay"]),
        "--lr-schedule", PROFILE["lr_schedule"],
        "--learning-rate", family["lr"],
        "--selection-metric", PROFILE["selection_metric"],
        "--model-architecture", family["architecture"],
        "--edge-radius", str(PROFILE["edge_radius"]),
        "--metal-node-mode", PROFILE["metal_node_mode"],
        "--structural-readout-scope", PROFILE["structural_readout_scope"],
        "--shell-role-source", PROFILE["shell_role_source"],
        "--no-prepare-missing-ring-edges",
        "--hidden-s", str(PROFILE["hidden_s"]),
        "--hidden-v", str(PROFILE["hidden_v"]),
        "--edge-hidden", str(PROFILE["edge_hidden"]),
        "--gvp-layers", str(PROFILE["gvp_layers"]),
        "--esm-fusion-dim", str(PROFILE["esm_fusion_dim"]),
        "--head-mlp-layers", str(PROFILE["head_mlp_layers"]),
        "--head-mlp-dropout", str(PROFILE["head_mlp_dropout"]),
        "--esm-graph-encoder-dropout", str(PROFILE["esm_graph_encoder_dropout"]),
        "--node-feature-set", PROFILE["node_feature_set"],
        "--omit-node-features", ",".join(OMITTED_EXTERNAL_FEATURES),
        "--external-feature-source", "updated",
        "--external-features-root-dir", str(paths.empty_external_features),
        "--allow-missing-external-features",
        "--esm-dim", str(ESM_DIM),
        "--esm-embeddings-dir", str(esm_dir if esm_dir is not None else paths.empty_esm),
        "--no-prepare-missing-esm-embeddings",
        "--metal-class-weight-mode", "manual",
        "--metal-loss-function", PROFILE["metal_loss_function"],
        "--metal-label-smoothing", "0.0",
        "--metal-collapsed-loss-weight", "0.0",
        "--unsupported-metal-policy", "error",
        "--invalid-structure-policy", "error",
        "--require-all-task-classes",
        "--binding-residue-pooling", config.readout,
        "--export-validation-predictions",
        "--train-metrics-every-n-epochs", str(PROFILE["train_metrics_every_n_epochs"]),
        "--feature-inventory-sha256", str(identity["feature_inventory_sha256"]),
        "--device", device,
        "--runs-dir", str(runs_dir),
        "--run-name", run_name(config, fold, seed),
    ]
    for key, value in multipliers.items():
        command += [f"--{key.replace('_', '-')}-loss-multiplier", repr(float(value))]
    if family["fusion_mode"]:
        command += ["--fusion-mode", family["fusion_mode"]]
    if family["gvp_lr"]:
        command += ["--gvp-learning-rate", family["gvp_lr"]]
    if family["rbf_raw"]:
        command.append("--rbf-use-raw-distances")
    if save_epoch_checkpoints:
        command.append("--save-epoch-checkpoints")
    if load_workers is not None:
        command += ["--load-workers", str(load_workers)]
    # Bind the complete resolved training configuration, not only the profile fields.
    resolved = config_to_payload(parse_args(command[3:]))
    identity["resolved_config_sha256"] = stable_hash(
        {key: value for key, value in resolved.items() if key not in NON_IDENTITY_CONFIG_KEYS}
    )
    command += ["--campaign-run-identity", json.dumps(identity, sort_keys=True)]
    env = {
        FORBIDDEN_READ_ROOTS_ENV: forbidden_roots_environment(forbidden_read_roots(train_dir)),
        "DEEPMZYME_PARSE_CACHE_DIR": str(paths.parse_cache),
        "MKL_THREADING_LAYER": "GNU",
    }
    return command, env, identity


# ---------------------------------------------------------------------------
# Execution with identity-checked reuse
# ---------------------------------------------------------------------------


def completed_run_receipt(run_dir: Path, identity: dict[str, Any], *,
                          ignore: tuple[str, ...] = (), require_independent: bool = True) -> dict[str, Any] | None:
    """Return the selected-checkpoint receipt only for a verified, identical, completed fit."""
    receipt_path = run_dir / "selected_checkpoint.json"
    metadata_path = run_dir / "run_metadata.json"
    checkpoint_path = run_dir / "best_model_checkpoint.pt"
    if not all(path.is_file() for path in (receipt_path, metadata_path, checkpoint_path, run_dir / "run_config.json")):
        return None
    receipt = read_json(receipt_path)
    metadata = read_json(metadata_path)
    if receipt.get("fit_status") != "completed" or metadata.get("fit_status") != "completed":
        return None
    if receipt.get("reconciliation_status") != "match":
        return None
    if not (identity_matches(receipt.get("campaign_run_identity"), identity, ignore)
            and identity_matches(metadata.get("campaign_run_identity"), identity, ignore)):
        return None
    if sha256_file(checkpoint_path) != receipt.get("selected_checkpoint_sha256"):
        return None
    predictions = run_dir / receipt["validation_predictions"]["path"]
    if not predictions.is_file() or sha256_file(predictions) != receipt["validation_predictions"]["sha256"]:
        return None
    if metadata.get("test_report") is not None:
        raise ValueError(f"{run_dir} contains a held-out report; it cannot be a validation-chain unit")
    saved_config = read_json(run_dir / "run_config.json")
    epochs = identity.get("epochs", saved_config.get("config", {}).get("epochs"))
    if epochs is None or len(saved_config.get("history", [])) != int(epochs):
        return None
    if require_independent and identity.get("campaign_id") == CAMPAIGN_ID:
        directory = run_dir / "independent_validation_replay"
        replay_path = directory / "replay_receipt.json"
        if not replay_path.is_file():
            return None
        replay = read_json(replay_path)
        if (replay.get("independent_replay") is not True or replay.get("prediction_rows_verified") is not True
                or replay.get("fit_status") != "completed" or replay.get("reconciliation_status") != "match"
                or replay.get("selected_checkpoint_sha256") != receipt["selected_checkpoint_sha256"]
                or not identity_matches(replay.get("campaign_run_identity"), identity, ignore)):
            return None
        replay_predictions = directory / replay["validation_predictions"]["path"]
        if not replay_predictions.is_file() or sha256_file(replay_predictions) != replay["validation_predictions"]["sha256"]:
            return None
    return receipt


def execute_unit(
    paths: CampaignPaths,
    command: list[str],
    env_extra: dict[str, str],
    identity: dict[str, Any],
    runs_dir: Path,
    name: str,
    *,
    dry_run: bool = False,
    execution=None,
) -> dict[str, Any]:
    from training.config import config_to_payload, parse_args

    run_dir = runs_dir / name
    receipt = completed_run_receipt(run_dir, identity)
    # Resolve through the real training parser before fitting, so the saved record is
    # the configuration the child will run, not only its argv.
    resolved = config_to_payload(parse_args(command[3:]))
    command_record = {"run_name": name, "argv": command, "env": env_extra, "identity": identity,
                      "profile": PROFILE, "resolved_config": resolved,
                      "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    write_json(paths.commands / f"{name}.json", command_record)
    if receipt is not None:
        return {"run_name": name, "status": "reused_verified", "receipt": receipt}
    if dry_run:
        return {"run_name": name, "status": "planned"}
    fitted = completed_run_receipt(run_dir, identity, require_independent=False)
    if run_dir.exists() and fitted is None:
        # A restart, never an optimizer-state resume: keep the partial attempt for audit.
        attempts = runs_dir / "_incomplete_attempts"
        attempts.mkdir(parents=True, exist_ok=True)
        shutil.move(str(run_dir), str(attempts / f"{name}__{time.strftime('%Y%m%dT%H%M%S')}"))
    env = {**os.environ, **env_extra}
    started = time.time()
    log_path = runs_dir / f"{name}.log"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_process = execution.run_subprocess if execution is not None else subprocess.run
    if fitted is None:
        with log_path.open("w", encoding="utf-8") as log:
            process = run_process(command, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=REPO_ROOT)
        elapsed = time.time() - started
        if process.returncode != 0:
            return {"run_name": name, "status": "failed", "return_code": process.returncode,
                    "elapsed_seconds": elapsed, "log": str(log_path)}
    elapsed = time.time() - started
    receipt = completed_run_receipt(run_dir, identity, require_independent=False)
    if receipt is None:
        return {"run_name": name, "status": "failed_verification", "elapsed_seconds": elapsed, "log": str(log_path)}
    if identity.get("campaign_id") == CAMPAIGN_ID:
        replay_dir = run_dir / "independent_validation_replay"
        if replay_dir.exists():
            attempts = run_dir / "_incomplete_replays"
            attempts.mkdir(exist_ok=True)
            shutil.move(str(replay_dir), str(attempts / f"attempt_{time.time_ns()}"))
        replay_command = [command[0], str(SRC_ROOT / "export_validation_predictions.py"),
                          "--campaign-run-dir", str(run_dir), "--validation-only", "--device", resolved["device"]]
        with log_path.open("a", encoding="utf-8") as log:
            replay_result = run_process(replay_command, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=REPO_ROOT)
        elapsed = time.time() - started
        if replay_result.returncode or completed_run_receipt(run_dir, identity) is None:
            return {"run_name": name, "status": "failed_independent_replay", "elapsed_seconds": elapsed,
                    "log": str(log_path)}
    return {"run_name": name, "status": "completed", "elapsed_seconds": elapsed, "receipt": receipt,
            "fit_reused_for_replay": fitted is not None}


def selected_units(families: list[str] | None, targets: list[str] | None, readouts: list[str] | None,
                   folds: list[int] | None) -> list[tuple[GridConfig, int, int]]:
    for values, allowed, label in ((families, FAMILIES, "families"), (targets, TARGET_SCHEMES, "targets"),
                                   (readouts, READOUTS, "readouts"), (folds, range(N_FOLDS), "folds")):
        if values is not None and (not values or len(set(values)) != len(values) or set(values) - set(allowed)):
            raise ValueError(f"Invalid or repeated campaign {label}: {values}")
    units = []
    for config in grid_configs():
        if families and config.family not in families:
            continue
        if targets and config.target not in targets:
            continue
        if readouts and config.readout not in readouts:
            continue
        for fold in folds if folds is not None else range(N_FOLDS):
            for seed in MODEL_SEEDS:
                units.append((config, fold, seed))
    if not units:
        raise ValueError("The requested filters select no evaluated campaign configuration")
    return units


# ---------------------------------------------------------------------------
# Guards and the training-only smoke campaign
# ---------------------------------------------------------------------------


def campaign_manifest_guard(paths: CampaignPaths, *, require_context: bool = False) -> None:
    """Refuse fitting unless cohort, folds, weights and feature inventory are frozen and consistent."""
    manifest = load_campaign(paths)
    if manifest["campaign_id"] != CAMPAIGN_ID:
        raise ValueError("Historical v1 has unresolved structure context; use the certified v2 context cohort")
    if require_context:
        context_paths = paths
        if "smoke_of" in manifest:
            context_paths = CampaignPaths(paths.root.parent)
            if sha256_file(context_paths.manifest) != manifest["smoke_of"]["campaign_manifest_sha256"]:
                raise ValueError("Smoke parent campaign changed after derivation")
        context_manifest = load_campaign(context_paths)
        certificate = read_json(context_paths.root / "train_context_audit.json")
        if (certificate.get("input_contract_certified") is not True
                or certificate["cohort_sha256"] != context_manifest["cohort"]["sha256"]
                or certificate["structure_content_sha256"] != context_manifest["structure_content_sha256"]
                or certificate.get("blockers")):
            raise ValueError("Structure context is not certified for this frozen cohort")
    for required in (paths.fold_membership, paths.fold_class_weights, paths.feature_inventory):
        if not required.is_file():
            raise ValueError(f"Missing frozen campaign file {required}; run the preceding step first")
    weights = read_json(paths.fold_class_weights)
    if weights["fold_membership_sha256"] != sha256_file(paths.fold_membership):
        raise ValueError("fold_class_weights.json does not describe the current fold_membership.csv")
    inventory = read_json(paths.feature_inventory)
    if inventory.get("cohort_sha256") != manifest["cohort"]["sha256"]:
        raise ValueError("feature_inventory.json certifies a different cohort")
    paths.empty_external_features.mkdir(parents=True, exist_ok=True)
    paths.empty_esm.mkdir(parents=True, exist_ok=True)
    for directory in (paths.empty_external_features, paths.empty_esm):
        if any(directory.iterdir()):
            raise ValueError(f"{directory} must remain empty")


def prepare_smoke_campaign(paths: CampaignPaths, per_element_groups: int = 12) -> CampaignPaths:
    """Derive a small certified-training-example campaign under ``<campaign>/smoke``.

    Groups are chosen deterministically (sorted PDB IDs) so every native element
    reaches every fold; the smoke never touches the parent's runs or folds.
    """
    manifest = load_campaign(paths)
    smoke = CampaignPaths(paths.root / "smoke")
    smoke.root.mkdir(exist_ok=True)
    bindings = read_cohort_csv(paths.cohort, expected_sha256=manifest["cohort"]["sha256"])
    groups_by_element: dict[str, list[str]] = {}
    for binding in bindings:
        groups_by_element.setdefault(binding.native_element, [])
        if binding.group_id not in groups_by_element[binding.native_element]:
            groups_by_element[binding.native_element].append(binding.group_id)
    chosen = {group for element in NATIVE_ELEMENTS
              for group in sorted(groups_by_element.get(element, []))[:per_element_groups]}
    with paths.cohort.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [row for row in reader if row["group_id"] in chosen]
    smoke_cohort = smoke.cohort
    with smoke_cohort.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    smoke_manifest = dict(manifest)
    smoke_manifest["smoke_of"] = {"campaign_manifest_sha256": sha256_file(paths.manifest),
                                  "per_element_groups": per_element_groups}
    smoke_manifest["cohort"] = {"path": "train_cohort.csv", "sha256": sha256_file(smoke_cohort), "n_rows": len(rows)}
    write_json(smoke.manifest, smoke_manifest)
    if smoke.fold_membership.exists():
        smoke.fold_membership.unlink()
    freeze_folds(smoke, strict=False)
    if not runnable_folds(smoke):
        raise ValueError("No smoke fold contains every native class; increase per_element_groups")
    if paths.feature_inventory.exists():
        inventory = read_json(paths.feature_inventory)
        if inventory.get("cohort_sha256") != manifest["cohort"]["sha256"]:
            raise ValueError("Parent feature inventory certifies a different cohort")
        inventory["cohort_sha256"] = smoke_manifest["cohort"]["sha256"]
        inventory["smoke_subset_of_inventory_sha256"] = inventory_identity_sha256(paths)
        inventory["certification_method"] = "exact subset of the parent's certified examples and feature files"
        inventory["n_examples_loaded"] = len(rows)
        inventory["n_graphs_built"] = len(rows)
        inventory["n_empty_first_shell_graphs"] = sum(int(row["n_first_shell_residues"]) == 0 for row in rows)
        inventory["total_context_residues"] = sum(int(row["n_context_residues"]) for row in rows)
        inventory.pop("feature_fallbacks", None)  # the parent's aggregate is not a subset count
        smoke.empty_external_features.mkdir(parents=True, exist_ok=True)
        smoke.empty_esm.mkdir(parents=True, exist_ok=True)
        if "external_features" in inventory:
            inventory["external_features"]["external_features_root_dir"] = str(smoke.empty_external_features)
        if "parse_cache" in inventory:
            inventory["parse_cache"]["root"] = str(smoke.parse_cache)
        esm = inventory.get("esm")
        parent_plan = paths.root / "esm_generation_plan.csv"
        parent_summary = paths.root / "esm_generation_plan.json"
        if esm is not None and inventory.get("schema_version") == 2 and esm.get("files"):
            if not parent_plan.is_file() or not parent_summary.is_file():
                raise ValueError("Certified ESM smoke requires its parent's frozen generation plan")
            plan_summary = read_json(parent_summary)
            if (sha256_file(parent_plan) != plan_summary["plan_csv_sha256"]
                    or esm.get("plan_csv_sha256") != plan_summary["plan_csv_sha256"]):
                raise ValueError("Parent ESM generation plan changed after certification")
            required = Counter((row["structure_name"], chain) for row in rows
                               for chain in row["context_chains"].split(";"))
            with parent_plan.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                plan_fields = list(reader.fieldnames or [])
                subset_plan = [dict(row, n_ions_using_chain=required[(row["structure_name"], row["chain"])])
                               for row in reader if (row["structure_name"], row["chain"]) in required]
            if (len(subset_plan) != len(required)
                    or {(row["structure_name"], row["chain"]) for row in subset_plan} != set(required)):
                raise ValueError("Parent ESM plan does not cover every smoke context chain exactly once")
            subset_plan.sort(key=lambda row: (row["structure_name"], row["chain"]))
            smoke_plan = smoke.root / "esm_generation_plan.csv"
            with smoke_plan.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=plan_fields)
                writer.writeheader()
                writer.writerows(subset_plan)
            unique = {row["sequence_sha256"]: int(row["sequence_length"]) for row in subset_plan}
            residues = sum(int(row["sequence_length"]) for row in subset_plan)
            write_json(smoke.root / "esm_generation_plan.json", {
                "n_chain_files": len(subset_plan),
                "n_structures": len({row["structure_name"] for row in subset_plan}),
                "n_unique_sequences": len(unique), "unique_sequence_residues": sum(unique.values()),
                "stored_residues": residues,
                "estimated_float32_storage_gb": residues * int(esm["embedding_dim"]) * 4 / 1e9,
                "max_sequence_length": max(unique.values()), "plan_csv_sha256": sha256_file(smoke_plan),
                "parent_plan_csv_sha256": plan_summary["plan_csv_sha256"],
            })
            expected_names = {f"{Path(row['structure_name']).stem}_chain_{row['chain']}_esmc.pt"
                              for row in subset_plan}
            records = [record for record in esm["files"] if record["path"] in expected_names]
            if len(records) != len(expected_names) or {record["path"] for record in records} != expected_names:
                raise ValueError("Parent ESM inventory does not certify every smoke context chain")
            digest = hashlib.sha256()
            for record in records:
                digest.update(f"{record['path']}\t{record['sha256']}\n".encode())
            esm.update(files=records, n_files=len(records), files_sha256=digest.hexdigest(),
                       plan_csv_sha256=sha256_file(smoke_plan),
                       residue_coverage={"retained_residues": inventory["total_context_residues"], "missing": 0})
        elif esm is not None:
            # Folds can still be planned from a legacy/minimal inventory. Such a
            # descriptor is never sufficient evidence to admit smoke execution.
            inventory["certified"] = False
            inventory["certification_blocker"] = "Parent needs schema-v2 per-file ESM certification and a frozen plan"
        write_json(smoke.feature_inventory, inventory)
    return smoke
