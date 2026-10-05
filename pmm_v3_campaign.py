"""PMM ion metal v3 campaign: frozen inputs, recipes, units, commands and lane execution.

Plan: docs/campaigns/pmm_ion_metal_v3/plan.md (steps A2 and C-E). This module lives
at the repository root, outside the hashed ``src/`` tree, so defining a step-D
combination recipe never invalidates graph caches or the frozen source identity.

Layout of a v3 campaign root (relocatable; nothing outside it is written):
  campaign_manifest.json   frozen identities, profile, recipes, source tree
  frozen_inputs/           hash-checked copies of the v2 cohort, ESMC plan and inventory
  fold_membership.csv      copy of the frozen v3 fold file
  fold_class_weights.json  common-four equalized weights per v3 training fold
  parse_cache/, raw_graph_cache/   shared caches (atomic writes; safe across lanes)
  lanes/laneK/             one execution lock, runs/, commands/ and state per lane

Lanes are independent single-child executions (benchmarking.pmm_execution); running
two or three lanes at once is the concurrency option tested in step B. Every unit is
admitted once: an existing artifact for the same unit in any lane refuses the run.
"""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from benchmarking import pmm_ion_campaign as v2  # noqa: E402  (generic helpers only)
from training.access_guard import FORBIDDEN_READ_ROOTS_ENV, forbidden_roots_environment  # noqa: E402
from training.source_cohort import read_cohort_csv, sha256_file  # noqa: E402

CAMPAIGN_ID = "pmm_ion_metal_v3"
SCHEMA_VERSION = 1
N_FOLDS = 5
FAMILIES = ("only_esm", "only_gvp", "gvp_late_fusion")
TARGETS = ("four_class", "five_class", "six_class")
TARGET_SCHEMES = {"four_class": "merge_fe_class_viii", "five_class": "five_class", "six_class": "split_all_metals"}
SEEDS = (42, 43)
GVP_FAMILIES = ("only_gvp", "gvp_late_fusion")
FROZEN_INPUTS = ("train_cohort.csv", "esm_generation_plan.csv", "feature_inventory.json",
                 "campaign_manifest.json", "train_context_audit.json")

# The v3 baseline recipe. Values equal the v2 profile except the fold source,
# schedule, checkpoint rule, seeds and checkpoint retention (plan "Fixed decisions").
PROFILE: dict[str, Any] = {
    "campaign_id": CAMPAIGN_ID, "schema_version": SCHEMA_VERSION, "task": "metal",
    "metal_example_unit": "ion", "n_folds": N_FOLDS, "fold_split_source": "membership",
    "model_seeds": list(SEEDS), "epochs": 50, "batch_size": 16, "optimizer": "AdamW",
    "weight_decay": 1e-4, "lr_schedule": "cosine", "checkpoint_rule": "terminal",
    "descriptive_selection_metric": "val_metal_balanced_acc", "train_metrics_every_n_epochs": 10,
    "pocket_radius": 10.0, "edge_radius": 8.0, "metal_node_mode": "none",
    "structural_readout_scope": "residue_only", "shell_role_source": "geometry", "use_ring_edges": False,
    "hidden_s": 128, "hidden_v": 16, "edge_hidden": 64, "gvp_layers": 4, "esm_fusion_dim": 128,
    "head_mlp_layers": 2, "head_mlp_dropout": 0.2, "esm_graph_encoder_dropout": 0.1,
    "node_feature_set": "conservative", "omit_node_features": list(v2.OMITTED_EXTERNAL_FEATURES),
    "esm_model_name": v2.ESM_MODEL_NAME, "esm_dim": v2.ESM_DIM,
    "metal_class_weight_mode": "manual_common_four_equalized", "metal_loss_function": "cross_entropy",
    "metal_label_smoothing": 0.0, "metal_collapsed_loss_weight": 0.0, "sampler": "ordinary shuffled",
    "save_epoch_checkpoints": False, "evaluate_test": False,
}

# Single-change recipes (plan step D table). Unlisted keys inherit the baseline.
RECIPES: dict[str, dict[str, Any]] = {
    "baseline": {"families": FAMILIES, "flags": []},
    "v2recipe": {"families": FAMILIES, "flags": [], "lr_schedule": "fixed",
                 "checkpoint_rule": "best_validation", "targets": ("four_class",)},
    "meanagg": {"families": GVP_FAMILIES, "flags": ["--gvp-normalize-message-aggregation"]},
    "resdrop01": {"families": GVP_FAMILIES, "flags": ["--gvp-residual-dropout", "0.1"]},
    "structlr": {"families": ("gvp_late_fusion",), "flags": ["--gvp-lr-scope", "structural"]},
    "gvpaux03": {"families": ("gvp_late_fusion",), "flags": ["--gvp-auxiliary-loss-weight", "0.3"]},
    "esmdrop02": {"families": ("gvp_late_fusion",), "flags": ["--esm-modality-dropout", "0.2"]},
    "sitenone": {"families": GVP_FAMILIES, "flags": ["--site-geometry-features", "none"]},
    "sitecountsangles": {"families": GVP_FAMILIES, "flags": ["--site-geometry-features", "counts_angles"]},
    "posnoise01": {"families": ("only_gvp",), "flags": ["--position-noise-std", "0.1"]},
    "outerdrop01": {"families": ("only_gvp",), "flags": ["--outer-residue-dropout", "0.1"]},
    "vecnorm": {"families": ("only_gvp",), "flags": ["--gvp-vector-norm"]},
    "invsqrtw": {"families": GVP_FAMILIES, "flags": [], "weight_mode": "inverse_sqrt_frequency"},
}
# Alternatives that may not be combined (plan step D combination rule).
EXCLUSIVE_PAIRS = ({"gvpaux03", "esmdrop02"}, {"posnoise01", "outerdrop01"}, {"sitenone", "sitecountsangles"})
NON_COMBINABLE = {"baseline", "v2recipe", "sitenone"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def stable_hash(payload: Any) -> str:
    return v2.stable_hash(payload)


def file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Recipes and units
# ---------------------------------------------------------------------------

def resolve_recipe(name: str) -> dict[str, Any]:
    """A single recipe, or ``combo-a+b`` joining two or more combinable single recipes."""
    if name in RECIPES:
        recipe = dict(RECIPES[name])
    else:
        require(name.startswith("combo-"), f"Unknown recipe {name!r}")
        parts = name[len("combo-"):].split("+")
        require(len(parts) >= 2 and len(set(parts)) == len(parts) and parts == sorted(parts),
                "A combination lists two or more distinct recipes in sorted order")
        for part in parts:
            require(part in RECIPES and part not in NON_COMBINABLE, f"{part!r} cannot be combined")
        for pair in EXCLUSIVE_PAIRS:
            require(len(pair & set(parts)) < 2, f"Alternatives {sorted(pair)} cannot be combined")
        families = set(FAMILIES)
        flags: list[str] = []
        weight_mode = None
        for part in parts:
            families &= set(RECIPES[part]["families"])
            flags += RECIPES[part]["flags"]
            weight_mode = RECIPES[part].get("weight_mode", weight_mode)
        recipe = {"families": tuple(f for f in FAMILIES if f in families), "flags": flags}
        if weight_mode:
            recipe["weight_mode"] = weight_mode
        recipe["components"] = parts
    recipe.setdefault("lr_schedule", PROFILE["lr_schedule"])
    recipe.setdefault("checkpoint_rule", PROFILE["checkpoint_rule"])
    recipe.setdefault("targets", TARGETS)
    recipe.setdefault("weight_mode", "manual")
    return recipe


@dataclass(frozen=True)
class Unit:
    family: str
    target: str
    recipe: str
    fold: int
    seed: int

    def __post_init__(self) -> None:
        recipe = resolve_recipe(self.recipe)
        require(self.family in FAMILIES and self.target in TARGETS, "Unknown family or target")
        require(self.family in recipe["families"], f"{self.recipe} does not apply to {self.family}")
        require(self.target in recipe["targets"], f"{self.recipe} does not apply to {self.target}")
        require(type(self.fold) is int and 0 <= self.fold < N_FOLDS, "fold must be 0..4")
        require(self.seed in SEEDS, f"seed must be one of {SEEDS}")

    @property
    def name(self) -> str:
        return f"{self.family}__{self.target}__{self.recipe}__fold{self.fold}__seed{self.seed}"

    @classmethod
    def parse(cls, name: str) -> "Unit":
        parts = name.split("__")
        require(len(parts) == 5 and parts[3].startswith("fold") and parts[4].startswith("seed"),
                f"Not a v3 unit name: {name!r}")
        return cls(parts[0], parts[1], parts[2], int(parts[3][4:]), int(parts[4][4:]))


def step_units(step: str, *, final_recipes: dict[str, str] | None = None,
               combo: str | None = None, family: str | None = None) -> list[Unit]:
    """The frozen unit list of a plan step (C, D-A, D-B, D-combo, E-neutral, E-improvement)."""
    if step == "C":
        units = [Unit(f, t, "baseline", 0, 42) for f in FAMILIES for t in TARGETS]
        return units + [Unit(f, "four_class", "v2recipe", 0, 42) for f in FAMILIES]
    if step == "D-A":
        units = [Unit(f, "four_class", "baseline", 0, 43) for f in GVP_FAMILIES]
        for recipe in ("meanagg", "resdrop01", "structlr", "gvpaux03", "esmdrop02"):
            units += [Unit(f, "four_class", recipe, 0, s) for f in RECIPES[recipe]["families"] for s in SEEDS]
        return units
    if step == "D-B":
        units = []
        for recipe in ("sitenone", "sitecountsangles", "posnoise01", "outerdrop01", "vecnorm", "invsqrtw"):
            units += [Unit(f, "four_class", recipe, 0, s) for f in RECIPES[recipe]["families"] for s in SEEDS]
        return units
    if step == "D-combo":
        require(combo is not None and family is not None, "D-combo needs --recipe combo-... and --family")
        return [Unit(family, "four_class", combo, 0, s) for s in SEEDS]
    if step == "E-neutral":
        return [Unit(f, t, "baseline", fold, 42) for f in FAMILIES for t in TARGETS for fold in (1, 2, 3, 4)]
    if step == "E-improvement":
        require(bool(final_recipes), "E-improvement needs the final recipe per improved family")
        units = []
        for fam, recipe in sorted(final_recipes.items()):
            require(fam in GVP_FAMILIES and recipe != "baseline", f"No improvement recipe for {fam}")
            units += [Unit(fam, "four_class", recipe, fold, 42) for fold in (1, 2, 3, 4)]
        return units
    raise ValueError(f"Unknown step {step!r}")


# ---------------------------------------------------------------------------
# Campaign root
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class V3Paths:
    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).resolve())

    manifest = property(lambda self: self.root / "campaign_manifest.json")
    frozen = property(lambda self: self.root / "frozen_inputs")
    cohort = property(lambda self: self.root / "frozen_inputs" / "train_cohort.csv")
    fold_membership = property(lambda self: self.root / "fold_membership.csv")
    fold_class_weights = property(lambda self: self.root / "fold_class_weights.json")
    inventory = property(lambda self: self.root / "frozen_inputs" / "feature_inventory.json")
    esm_plan = property(lambda self: self.root / "frozen_inputs" / "esm_generation_plan.csv")
    empty_external_features = property(lambda self: self.root / "inputs" / "external_features_intentionally_empty")
    empty_esm = property(lambda self: self.root / "inputs" / "esm_not_used_by_only_gvp")
    parse_cache = property(lambda self: self.root / "parse_cache")
    graph_cache = property(lambda self: self.root / "raw_graph_cache")
    lanes = property(lambda self: self.root / "lanes")
    execution_settings = property(lambda self: self.root / "execution_settings.json")
    claims = property(lambda self: self.root / "claims")

    def lane(self, index: int) -> Path:
        require(type(index) is int and 0 <= index < 8, "lane must be 0..7")
        return self.lanes / f"lane{index}"


def runner_sha256() -> dict[str, str]:
    return {name: file_sha(ROOT / name) for name in ("pmm_v3_campaign.py", "run_pmm_v3_campaign.py")
            if (ROOT / name).is_file()}


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "--no-optional-locks", "-C", str(ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def prepare_campaign(v3_root: Path, *, v2_root: Path, fold_dir: Path) -> dict[str, Any]:
    """Freeze a new v3 campaign root from the v2 cohort and a frozen v3 fold set (CPU only)."""
    paths = V3Paths(v3_root)
    require(not paths.manifest.exists(), f"{paths.manifest} exists; a campaign is never re-prepared")
    v2_paths = v2.CampaignPaths(Path(v2_root).resolve())
    v2_manifest = v2.load_campaign(v2_paths)
    require(v2_manifest["campaign_id"] == "pmm_ion_metal_v2_context", "Source must be the v2 context campaign")
    inventory = json.loads(v2_paths.feature_inventory.read_text())
    require(inventory.get("certified") and inventory.get("certified_scope") == "structure_and_esmc600m",
            "The v2 feature inventory must be ESMC-600M certified")
    fold_dir = Path(fold_dir).resolve()
    receipt = json.loads((fold_dir / "fold_receipt.json").read_text())
    require(receipt.get("accepted") is True and not (fold_dir / "SUPERSEDED.md").exists(),
            "Fold set is not accepted or is superseded")
    fold_sha = sha256_file(fold_dir / "fold_membership.csv")
    require(receipt["outputs_sha256"]["fold_membership.csv"] == fold_sha, "Fold file differs from its receipt")
    require(receipt["inputs"]["cohort_sha256"] == v2_manifest["cohort"]["sha256"], "Folds belong to another cohort")
    bindings = read_cohort_csv(v2_paths.cohort, expected_sha256=v2_manifest["cohort"]["sha256"])
    with (fold_dir / "fold_membership.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(sorted(r["source_uid"] for r in rows) == sorted(b.source_uid for b in bindings),
            "Fold file does not cover the cohort exactly once")
    weights = v2.fold_class_weights(rows, strict=True)
    paths.frozen.mkdir(parents=True, exist_ok=False)
    copied = {}
    for name in FROZEN_INPUTS:
        source = v2_paths.root / name
        shutil.copy2(source, paths.frozen / name)
        require(file_sha(paths.frozen / name) == file_sha(source), f"Copy of {name} changed")
        copied[name] = file_sha(source)
    shutil.copy2(fold_dir / "fold_membership.csv", paths.fold_membership)
    require(sha256_file(paths.fold_membership) == fold_sha, "Fold file copy changed")
    weights["fold_membership_sha256"] = fold_sha
    v2.write_json(paths.fold_class_weights, weights)
    for directory in (paths.empty_external_features, paths.empty_esm, paths.parse_cache,
                      paths.graph_cache, paths.lanes):
        directory.mkdir(parents=True, exist_ok=True)
    recipes = {name: resolve_recipe(name) for name in RECIPES}
    manifest = {
        "schema_version": SCHEMA_VERSION, "campaign_id": CAMPAIGN_ID,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "profile": PROFILE, "profile_sha256": stable_hash(PROFILE),
        "recipes": recipes, "recipes_sha256": stable_hash(recipes),
        "cohort": {"sha256": v2_manifest["cohort"]["sha256"], "n_rows": len(bindings)},
        "structure_content_sha256": v2_manifest["structure_content_sha256"],
        "frozen_inputs_sha256": copied,
        "fold_set": {"id": fold_dir.name, "fold_membership_sha256": fold_sha,
                     "receipt_sha256": file_sha(fold_dir / "fold_receipt.json"),
                     "builder_version": receipt["builder_version"]},
        "fold_class_weights_sha256": sha256_file(paths.fold_class_weights),
        "feature_inventory_identity_sha256": v2.inventory_identity_sha256(v2_paths),
        "frozen_source_tree_sha256": v2.source_tree_sha256(), "git_commit": git_commit(),
        "runner_sha256": runner_sha256(), "parent_campaign": v2_manifest["campaign_id"],
        "held_out_access": False,
    }
    v2.write_json(paths.manifest, manifest)
    return manifest


def verify_frozen_esm(paths: V3Paths, esm_dir: Path) -> dict[str, Any]:
    """Rehash every planned ESMC payload and sidecar in a relocatable directory."""
    inventory = json.loads(paths.inventory.read_text())
    esm = inventory["esm"]
    require(esm.get("plan_csv_sha256") == sha256_file(paths.esm_plan), "ESMC plan differs from the inventory")
    for record in esm["files"]:
        require(Path(record["path"]).name == record["path"], "Inventory paths must be plain names")
        path = Path(esm_dir) / record["path"]
        require(sha256_file(path) == record["sha256"]
                and sha256_file(path.with_name(path.name + ".json")) == record["sidecar_sha256"],
                f"ESMC payload changed or missing: {path}")
    return {"verified": True, "n_files": len(esm["files"]), "esm_dir": str(esm_dir)}


def verify_runner_identity(manifest: dict[str, Any]) -> None:
    """The runner files and every recipe definition must be the ones recorded at preparation."""
    require(runner_sha256() == manifest["runner_sha256"],
            "pmm_v3_campaign.py or run_pmm_v3_campaign.py changed since preparation")
    recipes = {name: resolve_recipe(name) for name in RECIPES}
    require(stable_hash(recipes) == manifest["recipes_sha256"], "Recipe definitions changed since preparation")


def require_frozen_spec(paths: V3Paths) -> dict[str, Any]:
    """Plan A4: no fit before the assessment specification is frozen and describes this campaign."""
    import pmm_v3_assessment as assessment  # imported lazily: the assessor imports this module

    spec = assessment.load_spec()
    assessment.check_spec_against_campaign(spec, json.loads(paths.manifest.read_text()))
    return spec


def verify_campaign(paths: V3Paths, *, esm_dir: Path | None) -> dict[str, Any]:
    """Refuse fitting unless every frozen identity still matches (run-time guard)."""
    manifest = json.loads(paths.manifest.read_text())
    require(manifest.get("campaign_id") == CAMPAIGN_ID, "Not a v3 campaign root")
    verify_runner_identity(manifest)
    for name, digest in manifest["frozen_inputs_sha256"].items():
        require(file_sha(paths.frozen / name) == digest, f"Frozen input changed: {name}")
    require(sha256_file(paths.cohort) == manifest["cohort"]["sha256"], "Cohort changed")
    require(sha256_file(paths.fold_membership) == manifest["fold_set"]["fold_membership_sha256"], "Fold file changed")
    require(sha256_file(paths.fold_class_weights) == manifest["fold_class_weights_sha256"], "Class weights changed")
    require(stable_hash(manifest["profile"]) == manifest["profile_sha256"] and manifest["profile"] == PROFILE,
            "Profile changed since preparation")
    require(v2.source_tree_sha256() == manifest["frozen_source_tree_sha256"],
            "src/ or scripts/ changed since preparation; all v3 fits share one frozen source tree")
    for directory in (paths.empty_external_features, paths.empty_esm):
        require(directory.is_dir() and not any(directory.iterdir()), f"{directory} must exist and stay empty")
    if esm_dir is not None:
        verify_frozen_esm(paths, esm_dir)
    return manifest


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def build_command(paths: V3Paths, unit: Unit, *, python_bin: str, train_dir: Path, esm_dir: Path | None,
                  device: str, lane: int, load_workers: int | None = None, epochs: int | None = None,
                  amp: bool = False, run_name: str | None = None) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    """Training argv, environment and run identity for one admitted unit (AMP from the step-B setting)."""
    from training.config import config_to_payload, parse_args

    manifest = json.loads(paths.manifest.read_text())
    recipe = resolve_recipe(unit.recipe)
    family = v2.FAMILIES[unit.family]
    weights = json.loads(paths.fold_class_weights.read_text())["folds"][str(unit.fold)]
    require(weights.get("runnable", True), f"Fold {unit.fold} is not runnable")
    if family["uses_esm"]:
        require(esm_dir is not None, f"{unit.family} needs the ESMC directory")
    epochs = int(epochs if epochs is not None else PROFILE["epochs"])
    runs_dir = paths.lane(lane) / "runs"
    command = [
        python_bin, "-u", str(ROOT / "src" / "train.py"),
        "--task", "metal", "--metal-example-unit", "ion",
        "--metal-label-scheme", TARGET_SCHEMES[unit.target], "--metal-eligibility-scheme", "six_class",
        "--structure-dir", str(train_dir), "--summary-csv", str(Path(train_dir) / "final_data_summarazing_table.csv"),
        "--source-cohort-csv", str(paths.cohort), "--source-cohort-sha256", manifest["cohort"]["sha256"],
        "--fold-membership-csv", str(paths.fold_membership),
        "--fold-membership-sha256", manifest["fold_set"]["fold_membership_sha256"],
        "--fold-split-source", "membership", "--n-folds", str(N_FOLDS), "--fold-index", str(unit.fold),
        "--seed", str(unit.seed), "--epochs", str(epochs), "--batch-size", str(PROFILE["batch_size"]),
        "--weight-decay", str(PROFILE["weight_decay"]), "--lr-schedule", recipe["lr_schedule"],
        "--checkpoint-rule", recipe["checkpoint_rule"], "--learning-rate", family["lr"],
        "--selection-metric", PROFILE["descriptive_selection_metric"],
        "--model-architecture", family["architecture"], "--edge-radius", str(PROFILE["edge_radius"]),
        "--metal-node-mode", PROFILE["metal_node_mode"],
        "--structural-readout-scope", PROFILE["structural_readout_scope"],
        "--shell-role-source", PROFILE["shell_role_source"], "--no-prepare-missing-ring-edges",
        "--hidden-s", str(PROFILE["hidden_s"]), "--hidden-v", str(PROFILE["hidden_v"]),
        "--edge-hidden", str(PROFILE["edge_hidden"]), "--gvp-layers", str(PROFILE["gvp_layers"]),
        "--esm-fusion-dim", str(PROFILE["esm_fusion_dim"]), "--head-mlp-layers", str(PROFILE["head_mlp_layers"]),
        "--head-mlp-dropout", str(PROFILE["head_mlp_dropout"]),
        "--esm-graph-encoder-dropout", str(PROFILE["esm_graph_encoder_dropout"]),
        "--node-feature-set", PROFILE["node_feature_set"], "--omit-node-features", ",".join(PROFILE["omit_node_features"]),
        "--external-feature-source", "updated", "--external-features-root-dir", str(paths.empty_external_features),
        "--allow-missing-external-features", "--esm-dim", str(PROFILE["esm_dim"]),
        "--esm-embeddings-dir", str(esm_dir if family["uses_esm"] else paths.empty_esm),
        "--no-prepare-missing-esm-embeddings",
        "--metal-loss-function", PROFILE["metal_loss_function"], "--metal-label-smoothing", "0.0",
        "--metal-collapsed-loss-weight", "0.0", "--unsupported-metal-policy", "error",
        "--invalid-structure-policy", "error", "--require-all-task-classes", "--binding-residue-pooling", "none",
        "--export-validation-predictions",
        "--train-metrics-every-n-epochs", str(PROFILE["train_metrics_every_n_epochs"]),
        "--feature-inventory-sha256", manifest["feature_inventory_identity_sha256"],
        "--device", device, "--runs-dir", str(runs_dir), "--run-name", run_name or unit.name,
    ]
    if recipe["weight_mode"] == "manual":
        command += ["--metal-class-weight-mode", "manual"]
        multipliers = weights["six_class_multipliers" if unit.target == "six_class" else "four_class_multipliers"]
        for key, value in multipliers.items():
            command += [f"--{key.replace('_', '-')}-loss-multiplier", repr(float(value))]
        if unit.target == "five_class":  # native Fe keeps the Class VIII common-four weight
            command += ["--fe-loss-multiplier", repr(float(multipliers["class_viii"]))]
    else:
        command += ["--metal-class-weight-mode", recipe["weight_mode"]]
    if family["fusion_mode"]:
        command += ["--fusion-mode", family["fusion_mode"]]
    if family["gvp_lr"]:
        command += ["--gvp-learning-rate", family["gvp_lr"]]
    if family["rbf_raw"]:
        command.append("--rbf-use-raw-distances")
    command += list(recipe["flags"])
    if amp:
        command.append("--amp")
    if load_workers is not None:
        command += ["--load-workers", str(load_workers)]
    resolved = config_to_payload(parse_args(command[3:]))
    identity = {
        "campaign_id": CAMPAIGN_ID, "profile_sha256": manifest["profile_sha256"],
        "cohort_sha256": manifest["cohort"]["sha256"], "fold_set_id": manifest["fold_set"]["id"],
        "fold_membership_sha256": manifest["fold_set"]["fold_membership_sha256"],
        "fold_class_weights_sha256": manifest["fold_class_weights_sha256"],
        "feature_inventory_sha256": manifest["feature_inventory_identity_sha256"],
        "source_tree_sha256": manifest["frozen_source_tree_sha256"], "runner_sha256": runner_sha256(),
        "family": unit.family, "target_scheme": unit.target, "recipe": unit.recipe,
        "recipe_definition": recipe, "fold": unit.fold, "model_seed": unit.seed, "epochs": epochs, "amp": bool(amp),
        "resolved_config_sha256": stable_hash({k: v for k, v in resolved.items()
                                               if k not in v2.NON_IDENTITY_CONFIG_KEYS}),
    }
    identity = json.loads(json.dumps(identity, sort_keys=True))  # the form the run records
    command += ["--campaign-run-identity", json.dumps(identity, sort_keys=True)]
    env = {FORBIDDEN_READ_ROOTS_ENV: forbidden_roots_environment(v2.forbidden_read_roots(train_dir)),
           "DEEPMZYME_PARSE_CACHE_DIR": str(paths.parse_cache),
           "DEEPMZYME_GRAPH_CACHE_DIR": str(paths.graph_cache), "MKL_THREADING_LAYER": "GNU"}
    return command, env, identity


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def unit_artifacts(paths: V3Paths, name: str) -> list[Path]:
    found = [paths.claims / f"{name}.json"] if (paths.claims / f"{name}.json").exists() else []
    if paths.lanes.is_dir():
        for lane in sorted(paths.lanes.iterdir()):
            for path in (lane / "runs" / name, lane / "runs" / f"{name}.log", lane / "commands" / f"{name}.json",
                         lane / f"run_status_{name}.json"):
                if path.exists() or path.is_symlink():
                    found.append(path)
            found += sorted((lane / "runs" / "_incomplete_attempts").glob(name + "__*"))
    return found


def verify_completed_unit(run_dir: Path, identity: dict[str, Any], recipe: dict[str, Any]) -> dict[str, Any]:
    """The selected-checkpoint receipt of a complete, identical, reconciled fit (never a test report)."""
    receipt = v2.read_json(run_dir / "selected_checkpoint.json")
    metadata = v2.read_json(run_dir / "run_metadata.json")
    saved = v2.read_json(run_dir / "run_config.json")
    require(receipt.get("fit_status") == "completed" and metadata.get("fit_status") == "completed",
            "Fit did not complete")
    require(receipt.get("reconciliation_status") == "match", "Selected checkpoint does not reconcile")
    require(receipt.get("campaign_run_identity") == identity and metadata.get("campaign_run_identity") == identity,
            "Run identity differs from the admitted unit")
    require(metadata.get("test_report") is None, "A held-out report exists; not a validation unit")
    require(len(saved.get("history", [])) == identity["epochs"], "History length differs from planned epochs")
    expected_name = ("terminal_model_checkpoint.pt" if recipe["checkpoint_rule"] == "terminal"
                     else "best_model_checkpoint.pt")
    require(receipt.get("selected_checkpoint") == expected_name, "Selected checkpoint violates the recipe rule")
    if recipe["checkpoint_rule"] == "terminal":
        require(receipt.get("selected_epoch") == identity["epochs"], "Terminal rule must select the last epoch")
    require(sha256_file(run_dir / expected_name) == receipt["selected_checkpoint_sha256"], "Checkpoint changed")
    predictions = run_dir / receipt["validation_predictions"]["path"]
    require(sha256_file(predictions) == receipt["validation_predictions"]["sha256"], "Predictions changed")
    return receipt


def verify_independent_replay(run_dir: Path, receipt: dict[str, Any]) -> dict[str, Any]:
    replay = v2.read_json(run_dir / "independent_validation_replay" / "replay_receipt.json")
    require(replay.get("independent_replay") is True and replay.get("prediction_rows_verified") is True
            and replay.get("reconciliation_status") == "match"
            and replay.get("selected_checkpoint_sha256") == receipt["selected_checkpoint_sha256"]
            and replay.get("campaign_run_identity") == receipt["campaign_run_identity"],
            "Independent replay does not confirm the selected checkpoint")
    return replay


def execute_unit(paths: V3Paths, lane: int, unit: Unit, command: list[str], env_extra: dict[str, str],
                 identity: dict[str, Any], *, execution, python_bin: str, device: str,
                 run_name: str | None = None) -> dict[str, Any]:
    """Train, verify and independently replay one admitted unit inside a lane's execution."""
    from training.config import config_to_payload, parse_args

    name = run_name or unit.name
    lane_root = paths.lane(lane)
    runs_dir, commands_dir = lane_root / "runs", lane_root / "commands"
    runs_dir.mkdir(parents=True, exist_ok=True)
    commands_dir.mkdir(parents=True, exist_ok=True)
    resolved = config_to_payload(parse_args(command[3:]))
    v2.write_json(commands_dir / f"{name}.json", {
        "run_name": name, "argv": command, "env": env_extra, "identity": identity,
        "resolved_config": resolved, "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    env = {**os.environ, **env_extra}
    log_path = runs_dir / f"{name}.log"
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        process = execution.run_subprocess(command, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
    if process.returncode != 0:
        return {"run_name": name, "status": "failed", "return_code": process.returncode,
                "elapsed_seconds": time.time() - started}
    run_dir = runs_dir / name
    try:
        receipt = verify_completed_unit(run_dir, identity, resolve_recipe(unit.recipe))
    except (ValueError, KeyError, OSError) as exc:
        return {"run_name": name, "status": "failed_verification", "error": str(exc),
                "elapsed_seconds": time.time() - started}
    replay_command = [python_bin, str(ROOT / "run_pmm_v3_campaign.py"), "--action", "replay",
                      "--run-dir", str(run_dir), "--device", device]
    with log_path.open("a", encoding="utf-8") as log:
        replayed = execution.run_subprocess(replay_command, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
    try:
        require(replayed.returncode == 0, "Independent replay process failed")
        replay = verify_independent_replay(run_dir, receipt)
    except (ValueError, KeyError, OSError) as exc:
        return {"run_name": name, "status": "failed_independent_replay", "error": str(exc),
                "elapsed_seconds": time.time() - started}
    return {"run_name": name, "status": "completed", "elapsed_seconds": time.time() - started,
            "selected_epoch": receipt["selected_epoch"], "selected_checkpoint": receipt["selected_checkpoint"],
            "selected_checkpoint_sha256": receipt["selected_checkpoint_sha256"],
            "replay_selected_epoch": replay["selected_epoch"]}


PROBE_TAG = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")


def run_name_for(unit: Unit, probe: str | None) -> str:
    """Campaign units keep their unit name; step-B probes live in their own ``probe-<tag>__`` namespace."""
    if probe is None:
        return unit.name
    require(bool(PROBE_TAG.match(probe)), "A probe tag is 1-40 lowercase letters, digits or hyphens")
    return f"probe-{probe}__{unit.name}"


def read_execution_settings(paths: V3Paths) -> dict[str, Any] | None:
    return json.loads(paths.execution_settings.read_text()) if paths.execution_settings.exists() else None


def resolve_amp(paths: V3Paths, *, probe: str | None, amp: bool | None) -> bool:
    """Probes choose AMP explicitly (step B); every campaign unit follows the recorded step-B setting."""
    settings = read_execution_settings(paths)
    if probe is not None:
        require(settings is None, "Step-B probes run only before the execution setting is recorded")
        require(amp is not None, "A probe must state AMP on or off")
        return bool(amp)
    require(settings is not None, "Record the step-B execution setting (set-execution) before campaign units")
    require(amp is None or bool(amp) == settings["amp"], "AMP differs from the recorded step-B setting")
    return bool(settings["amp"])


def run_unit(paths: V3Paths, unit: Unit, *, lane: int, train_dir: Path, esm_dir: Path | None, python_bin: str,
             device: str, execution_policy: dict[str, Any], load_workers: int | None = None,
             epochs: int | None = None, probe: str | None = None, amp: bool | None = None) -> dict[str, Any]:
    """Admit one unit (or one step-B probe of it) into a lane, run it once, and persist its artifacts."""
    from benchmarking.pmm_execution import CampaignExecution, ExecutionPolicy
    from training.access_guard import install_forbidden_read_guard

    install_forbidden_read_guard(v2.forbidden_read_roots(train_dir))
    require(Path(train_dir).name == "train", "Only the training-side directory named train is allowed")
    name = run_name_for(unit, probe)
    amp = resolve_amp(paths, probe=probe, amp=amp)
    existing = unit_artifacts(paths, name)
    if existing:
        raise ValueError(f"Unit artifacts exist ({existing[0]}); runs are never retried automatically")
    verify_campaign(paths, esm_dir=esm_dir)
    require_frozen_spec(paths)
    command, env, identity = build_command(paths, unit, python_bin=python_bin, train_dir=train_dir,
                                           esm_dir=esm_dir, device=device, lane=lane,
                                           load_workers=load_workers, epochs=epochs, amp=amp, run_name=name)
    require_retry_identity(paths, name, identity)
    lane_root = paths.lane(lane)
    policy = ExecutionPolicy(deadline_unix=execution_policy["deadline_unix"],
                             max_total_seconds=execution_policy["max_total_seconds"],
                             allocation_started_unix=execution_policy.get("allocation_started_unix"))
    execution = CampaignExecution(lane_root, policy=policy, session_id=execution_policy["session_id"],
                                  durable_root=Path(execution_policy["durable_root"]) / lane_root.name,
                                  persistence_mode=execution_policy["persistence_mode"])
    claim = claim_run(paths, name, lane=lane, identity=identity, session_id=execution_policy["session_id"])
    admitted = False
    try:
        with execution:
            execution.admit(name, float(execution_policy["estimated_fit_seconds"]))
            admitted = True
            result = _run_admitted(paths, lane, unit, name, probe, command, env, identity, execution,
                                   python_bin=python_bin, device=device)
    finally:
        if not admitted:  # nothing ran: the lane or the admission refused, so the name is released
            claim.unlink(missing_ok=True)
    return result


def _run_admitted(paths: V3Paths, lane: int, unit: Unit, name: str, probe: str | None, command: list[str],
                  env: dict[str, str], identity: dict[str, Any], execution, *, python_bin: str,
                  device: str) -> dict[str, Any]:
    """Run, record and persist one admitted unit inside its lane's execution."""
    lane_root = paths.lane(lane)
    result = execute_unit(paths, lane, unit, command, env, identity, execution=execution,
                          python_bin=python_bin, device=device, run_name=name)
    status_path = lane_root / f"run_status_{name}.json"
    v2.write_json(status_path, {**result, "identity": identity, "lane": lane, "unit": unit.name, "probe": probe})
    execution.record_result(name, result["status"], result["elapsed_seconds"])
    execution.persist([path for path in (lane_root / "runs" / name, lane_root / "runs" / f"{name}.log",
                                         lane_root / "commands" / f"{name}.json", status_path)
                       if path.exists()])
    return result


def set_execution_settings(paths: V3Paths, *, amp: bool, lanes: int, evidence: str,
                           reuse: dict[str, str] | None = None, epochs: int | None = None) -> dict[str, Any]:
    """Record the step-B choice once. ``reuse`` maps a step-C unit to the completed full-length probe
    that matches the chosen setting (plan: that run becomes the cell's step C baseline)."""
    require(not paths.execution_settings.exists(), "The step-B execution setting is recorded once")
    require(type(lanes) is int and 1 <= lanes <= 3, "Concurrent lanes must be 1, 2 or 3")
    require(bool(str(evidence).strip()), "Name the step-B evidence (report path)")
    statuses = completed_units(paths)
    checked = {}
    for unit_name, run_name in sorted((reuse or {}).items()):
        unit = Unit.parse(unit_name)
        require(unit in step_units("C"), f"{unit_name} is not a step C unit")
        require(run_name.startswith("probe-") and run_name.endswith("__" + unit_name),
                f"{run_name} is not a probe of {unit_name}")
        record = statuses.get(run_name)
        require(record is not None and record["status"] == "completed", f"{run_name} did not complete")
        expected = build_command(paths, unit, python_bin="python", train_dir=paths.root / "train",
                                 esm_dir=paths.root / "esm", device="cuda", lane=0, epochs=epochs, amp=amp)[2]
        differing = sorted(k for k in set(expected) | set(record["identity"])
                           if k != "runner_sha256" and expected.get(k) != record["identity"].get(k))
        require(not differing, f"{run_name} does not match {unit_name} under this setting: {differing}")
        checked[unit_name] = run_name
    settings = {"amp": bool(amp), "concurrent_lanes": lanes, "evidence": str(evidence), "reuse": checked,
                "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    v2.write_json(paths.execution_settings, settings)
    return settings


MAX_RERUNS = 1  # plan: a failed run is repeated once, unchanged (same seed and identity)


def claim_run(paths: V3Paths, name: str, *, lane: int, identity: dict[str, Any], session_id: str) -> Path:
    """Atomically claim a run name for every lane (O_CREAT | O_EXCL); a claim is only ever archived."""
    from benchmarking.pmm_execution import _process_identity

    paths.claims.mkdir(parents=True, exist_ok=True)
    path = paths.claims / f"{name}.json"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        raise ValueError(f"{name} is already claimed ({path}); a run name is launched once across lanes") from None
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"run_name": name, "lane": lane, "session_id": session_id, "identity": identity,
                   "worker": _process_identity(os.getpid()),
                   "claimed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    return path


def _process_alive(identity: dict[str, Any] | None) -> bool:
    from benchmarking.pmm_execution import _process_identity

    if not identity:
        return False
    try:
        return _process_identity(int(identity["pid"])) == identity
    except (FileNotFoundError, ProcessLookupError, ValueError, KeyError):
        return False


def verify_worker_stopped(paths: V3Paths, claim: dict[str, Any]) -> dict[str, Any]:
    """A missing terminal status is not enough: the claiming worker, its training child and the lane
    lock must all be gone (checked on the host that ran the unit)."""
    worker = claim.get("worker") or {}
    require(worker.get("hostname") == socket.gethostname(),
            "Archive on the host that ran the unit; its worker cannot be checked from here")
    require(not _process_alive(worker), f"Worker PID {worker.get('pid')} is still running")
    lane_root = paths.lane(int(claim["lane"]))
    state_path = lane_root / "execution_state.json"
    child = json.loads(state_path.read_text()).get("active_child") if state_path.exists() else None
    require(not _process_alive(child), f"Training child PID {child and child.get('pid')} is still running")
    lock_path = lane_root / "execution.lock"
    if lock_path.exists():
        with lock_path.open("a+") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError(f"Lane {claim['lane']} is still owned by a running worker") from None
            fcntl.flock(handle, fcntl.LOCK_UN)
    return {"worker_alive": False, "training_child_alive": False, "lane_lock_free": True,
            "checked_on": socket.gethostname(), "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def require_retry_identity(paths: V3Paths, name: str, identity: dict[str, Any]) -> None:
    """A rerun after an archived attempt must be unchanged: the same identity, seed included."""
    attempts = archived_attempts(paths, name)
    if attempts:
        archived = json.loads((attempts[-1] / "archive_receipt.json").read_text()).get("identity")
        require(archived == identity, f"{name}: the rerun identity differs from the archived attempt")


def archived_attempts(paths: V3Paths, name: str) -> list[Path]:
    root = paths.root / "failed_attempts" / name
    return sorted(root.glob("attempt*")) if root.is_dir() else []


def archive_failed_attempt(paths: V3Paths, name: str) -> Path:
    """Move a failed or interrupted unit's artifacts aside so the same unit can run once more."""
    record = completed_units(paths).get(name)
    require(record is None or record["status"] != "completed", f"{name} completed; a completed unit is never rerun")
    previous = archived_attempts(paths, name)
    require(len(previous) < MAX_RERUNS, f"{name} was already rerun {len(previous)} time(s); no further reruns")
    claim_path = paths.claims / f"{name}.json"
    require(claim_path.is_file(), f"{name} has no claim, so its worker cannot be shown to have stopped")
    try:
        claim = json.loads(claim_path.read_text())
    except json.JSONDecodeError:
        raise ValueError(f"{claim_path} is unreadable; inspect the lane before archiving") from None
    stopped = verify_worker_stopped(paths, claim)
    existing = unit_artifacts(paths, name)
    target = paths.root / "failed_attempts" / name / f"attempt{len(previous) + 1}"
    moved = []
    for path in existing:
        destination = target / path.relative_to(paths.root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        path.rename(destination)
        moved.append(str(destination.relative_to(paths.root)))
    v2.write_json(target / "archive_receipt.json", {
        "run_name": name, "status": None if record is None else record["status"],
        "status_meaning": "interrupted (no status record)" if record is None else "terminal failure",
        "identity": claim["identity"], "worker_stopped": stopped,
        "moved": moved, "archived_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "rerun_rule": "same seed and identity; at most one rerun"})
    return target


def completed_units(paths: V3Paths) -> dict[str, dict[str, Any]]:
    """Run name -> terminal status record across every lane (for planning and assessment)."""
    statuses = {}
    if paths.lanes.is_dir():
        for path in sorted(paths.lanes.glob("lane*/run_status_*.json")):
            record = v2.read_json(path)
            require(record["run_name"] not in statuses, f"Unit {record['run_name']} ran in two lanes")
            statuses[record["run_name"]] = {**record, "status_path": str(path)}
    return statuses


# ---------------------------------------------------------------------------
# Step C regression run: new code, v2 recipe, old v2 fold 0, v2 execution settings
# ---------------------------------------------------------------------------

REGRESSION_NAME = "regression__gvp_late_fusion__four_class__v2recipe__v2fold0__seed42"
REGRESSION_REFERENCE = {"best_epoch_common_four_ba": 0.894653, "last10_mean_common_four_ba": 0.857794,
                        "tolerance": 0.03,
                        "source": "pmm_ion_metal_v2_context gvp_late_fusion__four_class__none__fold0__seed42"}


def build_regression_command(paths: V3Paths, *, v2_root: Path, python_bin: str, train_dir: Path, esm_dir: Path,
                             device: str, lane: int, load_workers: int | None = None,
                             epochs: int | None = None) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    """The v2 late-fusion four-class fold-0 command, unchanged except for relocatable paths."""
    v2_paths = v2.CampaignPaths(Path(v2_root).resolve())
    command, env, identity = v2.build_train_command(
        v2_paths, python_bin=python_bin, train_dir=train_dir,
        config=v2.GridConfig("gvp_late_fusion", "four_class", "none"), fold=0, seed=42, device=device,
        runs_dir=paths.lane(lane) / "runs", epochs=epochs, load_workers=load_workers, save_epoch_checkpoints=False)
    command[command.index("--esm-embeddings-dir") + 1] = str(esm_dir)
    command[command.index("--run-name") + 1] = REGRESSION_NAME
    env.update(DEEPMZYME_PARSE_CACHE_DIR=str(paths.parse_cache), DEEPMZYME_GRAPH_CACHE_DIR=str(paths.graph_cache))
    return command, env, identity


def regression_gate(run_dir: Path, *, epochs: int = 50) -> dict[str, Any]:
    """Pre-declared step C gate: complete, replayed, and within 3.0 BA points of the v2 values."""
    receipt = v2.read_json(run_dir / "selected_checkpoint.json")
    replay = v2.read_json(run_dir / "independent_validation_replay" / "replay_receipt.json")
    with (run_dir / "epoch_metrics.csv").open(encoding="utf-8", newline="") as handle:
        history = [float(row["val_metal_collapsed4_balanced_acc"]) for row in csv.DictReader(handle)]
    best = float(receipt["metrics"]["val_metal_collapsed4_balanced_acc"])
    last10 = sum(history[-10:]) / len(history[-10:]) if history else float("nan")
    reference = REGRESSION_REFERENCE
    checks = {
        "all_epochs_completed": len(history) == epochs and receipt.get("fit_status") == "completed",
        "replay_confirmed": replay.get("prediction_rows_verified") is True
        and replay.get("selected_checkpoint_sha256") == receipt["selected_checkpoint_sha256"],
        "best_epoch_within_band": abs(best - reference["best_epoch_common_four_ba"]) <= reference["tolerance"],
        "last10_within_band": abs(last10 - reference["last10_mean_common_four_ba"]) <= reference["tolerance"],
    }
    return {"passed": all(checks.values()), "checks": checks, "best_epoch_common_four_ba": best,
            "last10_mean_common_four_ba": last10, "reference": reference,
            "note": "Engineering band, not a statistical bound; a failure stops v3 for diagnosis, no repeat."}


def run_regression(paths: V3Paths, *, v2_root: Path, lane: int, train_dir: Path, esm_dir: Path, python_bin: str,
                   device: str, execution_policy: dict[str, Any], load_workers: int | None = None,
                   epochs: int | None = None) -> dict[str, Any]:
    """Run the step C regression unit once in a lane, replay it and apply the pre-declared gate."""
    from benchmarking.pmm_execution import CampaignExecution, ExecutionPolicy
    from training.access_guard import install_forbidden_read_guard

    install_forbidden_read_guard(v2.forbidden_read_roots(train_dir))
    existing = unit_artifacts(paths, REGRESSION_NAME)
    if existing:
        raise ValueError(f"Regression artifacts exist ({existing[0]}); it is never repeated automatically")
    verify_campaign(paths, esm_dir=esm_dir)
    require_frozen_spec(paths)
    command, env_extra, identity = build_regression_command(
        paths, v2_root=v2_root, python_bin=python_bin, train_dir=train_dir, esm_dir=esm_dir, device=device,
        lane=lane, load_workers=load_workers, epochs=epochs)
    identity = json.loads(json.dumps(identity, sort_keys=True))
    require_retry_identity(paths, REGRESSION_NAME, identity)
    claim = claim_run(paths, REGRESSION_NAME, lane=lane, identity=identity, session_id=execution_policy["session_id"])
    lane_root = paths.lane(lane)
    (lane_root / "runs").mkdir(parents=True, exist_ok=True)
    (lane_root / "commands").mkdir(parents=True, exist_ok=True)
    v2.write_json(lane_root / "commands" / f"{REGRESSION_NAME}.json",
                  {"run_name": REGRESSION_NAME, "argv": command, "env": env_extra, "identity": identity})
    policy = ExecutionPolicy(deadline_unix=execution_policy["deadline_unix"],
                             max_total_seconds=execution_policy["max_total_seconds"],
                             allocation_started_unix=execution_policy.get("allocation_started_unix"))
    execution = CampaignExecution(lane_root, policy=policy, session_id=execution_policy["session_id"],
                                  durable_root=Path(execution_policy["durable_root"]) / lane_root.name,
                                  persistence_mode=execution_policy["persistence_mode"])
    run_dir = lane_root / "runs" / REGRESSION_NAME
    log_path = lane_root / "runs" / f"{REGRESSION_NAME}.log"
    env = {**os.environ, **env_extra}
    admitted = False
    try:
        with execution:
            execution.admit(REGRESSION_NAME, float(execution_policy["estimated_fit_seconds"]))
            admitted = True
            started = time.time()
            with log_path.open("w", encoding="utf-8") as log:
                trained = execution.run_subprocess(command, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
            result: dict[str, Any] = {"run_name": REGRESSION_NAME}
            if trained.returncode != 0:
                result.update(status="failed", return_code=trained.returncode)
            else:
                replay_command = [python_bin, str(ROOT / "run_pmm_v3_campaign.py"), "--action", "replay",
                                  "--run-dir", str(run_dir), "--device", device]
                with log_path.open("a", encoding="utf-8") as log:
                    replayed = execution.run_subprocess(replay_command, stdout=log, stderr=subprocess.STDOUT,
                                                        env=env, cwd=ROOT)
                if replayed.returncode != 0:
                    result.update(status="failed_independent_replay")
                else:
                    gate = regression_gate(run_dir, epochs=int(epochs if epochs is not None else PROFILE["epochs"]))
                    result.update(status="completed" if gate["passed"] else "failed_regression_gate", gate=gate)
            result["elapsed_seconds"] = time.time() - started
            status_path = lane_root / f"run_status_{REGRESSION_NAME}.json"
            v2.write_json(status_path, {**result, "identity": identity, "lane": lane})
            execution.record_result(REGRESSION_NAME, result["status"], result["elapsed_seconds"])
            execution.persist([path for path in (run_dir, log_path, lane_root / "commands" / f"{REGRESSION_NAME}.json",
                                                 status_path) if path.exists()])
    finally:
        if not admitted:  # nothing ran: the lane or the admission refused, so the name is released
            claim.unlink(missing_ok=True)
            (lane_root / "commands" / f"{REGRESSION_NAME}.json").unlink(missing_ok=True)
    return result
