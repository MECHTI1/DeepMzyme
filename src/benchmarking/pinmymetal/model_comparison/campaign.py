"""Scientific units and commands for the additive metal-completion comparison.

The v3 trainer, source tree, input records and runner modules remain immutable.
This module never allocates compute and its command builder never writes files.
The scientific grid is not execution authorization.
"""

from __future__ import annotations


import sys as _sys
from pathlib import Path as _Path
for _p in (str(_Path(__file__).resolve().parents[4]), str(_Path(__file__).resolve().parents[4] / "src")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
from project_paths import RELOCATED_FILES, install_relocated_module_aliases, resolve_existing_path, resolve_repo_file
install_relocated_module_aliases()
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

CAMPAIGN_ID = "metal_completion_2026-10"
FAMILY = "gvp_late_fusion"
TARGETS = ("four_class", "five_class", "six_class")
RECIPES = ("B", "P", "S")
FOLDS = tuple(range(5))
ROOT = Path(__file__).resolve().parents[4]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with resolve_existing_path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Unit:
    phase: str
    recipe: str
    target: str
    fold: int
    seed: int

    def __post_init__(self) -> None:
        require(self.phase in ("discovery", "confirmation"), "Unknown phase")
        require(self.recipe in RECIPES and self.target in TARGETS, "Unknown recipe or target")
        require(type(self.fold) is int and self.fold in FOLDS, "fold must be 0..4")
        seeds = (42,) if self.phase == "discovery" else (43, 44)
        require(type(self.seed) is int and self.seed in seeds, f"{self.phase} seeds must be {seeds}")

    @property
    def name(self) -> str:
        return (f"metal_completion__{self.phase}__{self.recipe}__{self.target}"
                f"__fold{self.fold}__seed{self.seed}")

    @property
    def family(self) -> str:
        return FAMILY

    @classmethod
    def parse(cls, name: str) -> "Unit":
        match = re.fullmatch(r"metal_completion__(discovery|confirmation)__([BPS])__"
                             r"(four_class|five_class|six_class)__fold([0-4])__seed(42|43|44)", name)
        require(match is not None, f"Invalid completion unit name: {name!r}")
        phase, recipe, target, fold, seed = match.groups()
        return cls(phase, recipe, target, int(fold), int(seed))


def discovery_units() -> list[Unit]:
    return [Unit("discovery", recipe, target, fold, 42)
            for recipe in RECIPES for target in TARGETS for fold in FOLDS]


def confirmation_units(shared_recipe: str) -> list[Unit]:
    require(shared_recipe in RECIPES, "Unknown shared recipe")
    cells = [(shared_recipe, target) for target in TARGETS]
    if shared_recipe != "B":
        cells.append(("B", "four_class"))
    return [Unit("confirmation", recipe, target, fold, seed)
            for recipe, target in cells for fold in FOLDS for seed in (43, 44)]


def recipe_config(recipe: str) -> dict[str, Any]:
    require(recipe in RECIPES, "Unknown recipe")
    return {"metal_loss_function": "focal" if recipe == "P" else "cross_entropy",
            "metal_focal_gamma": 0.0 if recipe == "P" else 2.0,
            "metal_label_smoothing": 0.1 if recipe == "S" else 0.0}


def effective_loss(recipe: str) -> dict[str, Any]:
    config = recipe_config(recipe)
    return {**config, "name": {"B": "batch_weight_normalized_cross_entropy",
                                "P": "population_normalized_weighted_cross_entropy",
                                "S": "native_label_smoothed_weighted_cross_entropy"}[recipe],
            "formula": ("sum_i(w[y_i] * -log(p[y_i])) / n_supervised" if recipe == "P" else
                        "torch.cross_entropy(weight=w, reduction=mean, label_smoothing="
                        f"{config['metal_label_smoothing']})"),
            "class_weight_rule": "manual common-four N_train / (4 * n_common_four)",
            "population_mean_example_weight": 1.0,
            "smoothing_space": "native target classes; coarse smoothing mass differs across targets"}


def validate_recipe_config(recipe: str, config: Mapping[str, Any]) -> None:
    for key, expected in recipe_config(recipe).items():
        require(config.get(key) == expected, f"{recipe}: {key} must equal {expected!r}")
    require(config.get("metal_class_weight_mode") == "manual", "Use the frozen manual common-four weights")
    require(config.get("metal_collapsed_loss_weight") == 0.0, "Collapsed auxiliary loss must remain off")
    require(config.get("gvp_auxiliary_loss_weight") == 0.0, "GVP auxiliary loss must remain off")


def _replace_flag(command: list[str], flag: str, value: str) -> None:
    positions = [index for index, token in enumerate(command) if token == flag]
    require(len(positions) <= 1, f"Duplicate trainer flag: {flag}")
    if positions:
        require(positions[0] + 1 < len(command), f"Missing flag value: {flag}")
        command[positions[0] + 1] = value
    else:
        command.extend((flag, value))


def isolated_paths(parent_root: Path, successor_root: Path):
    import pmm_v3_campaign as v3

    parent, successor = Path(parent_root).resolve(), Path(successor_root).resolve()
    require(parent != successor and parent not in successor.parents and successor not in parent.parents,
            "Successor outputs must be separate from, and not nested with, the frozen parent")
    require(successor != ROOT and ROOT not in successor.parents,
            "Successor outputs must be outside the immutable executable package")
    paths = v3.V3Paths(successor)
    for child in (paths.lanes, paths.claims, paths.parse_cache, paths.graph_cache):
        require(not child.is_symlink() and child.resolve().is_relative_to(successor),
                f"Unsafe successor output path: {child}")
    return paths


def resolved_config_sha256(command: list[str]) -> str:
    import pmm_v3_campaign as v3
    from training.config import config_to_payload, parse_args

    resolved = config_to_payload(parse_args(command[3:]))
    return v3.stable_hash({key: value for key, value in resolved.items()
                           if key not in v3.v2.NON_IDENTITY_CONFIG_KEYS})


def build_command(parent_paths, unit: Unit, *, successor_root: Path, manifest_sha256: str,
                  python_bin: str, train_dir: Path, esm_dir: Path, device: str, lane: int,
                  amp: bool, code_sha256: Mapping[str, str] | None = None,
                  load_workers: int | None = None) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    """Build through v3 without mutating its seed allowlist, recipe table or source guard.

    Seed 42 is a builder template only. The successor identity and parsed command
    record the actual 42/43/44 seed; no seed-44 fit is labeled as seed 42.
    Runtime source verification belongs to ``verify_manifest`` and the worker.
    """
    import pmm_v3_campaign as v3
    from training.config import config_to_payload, parse_args

    require(re.fullmatch(r"[0-9a-f]{64}", manifest_sha256) is not None, "Invalid manifest SHA-256")
    paths = isolated_paths(parent_paths.root, successor_root)
    template = v3.Unit(FAMILY, unit.target, "baseline", unit.fold, 42)
    command, env, original_identity = v3.build_command(
        parent_paths, template, python_bin=python_bin, train_dir=train_dir, esm_dir=esm_dir,
        device=device, lane=lane, amp=amp, load_workers=load_workers, run_name=unit.name)
    require(command[-2] == "--campaign-run-identity" and json.loads(command[-1]) == original_identity,
            "The frozen builder's identity layout changed")
    command = command[:-2]
    require(resolved_config_sha256(command) == original_identity["resolved_config_sha256"],
            "Frozen configuration hash does not reproduce")
    before = config_to_payload(parse_args(command[3:]))
    _replace_flag(command, "--seed", str(unit.seed))
    _replace_flag(command, "--runs-dir", str(paths.lane(lane) / "runs"))
    for key, value in recipe_config(unit.recipe).items():
        _replace_flag(command, "--" + key.replace("_", "-"), str(value))
    resolved = config_to_payload(parse_args(command[3:]))
    validate_recipe_config(unit.recipe, resolved)
    allowed = {"seed", "runs_dir", *recipe_config(unit.recipe)}
    differences = {key for key in set(before) | set(resolved) if before.get(key) != resolved.get(key)}
    require(differences <= allowed, f"Unexpected trainer differences: {sorted(differences - allowed)}")
    identity = {**original_identity, "campaign_id": CAMPAIGN_ID, "phase": unit.phase,
                "recipe": unit.recipe, "recipe_definition": effective_loss(unit.recipe),
                "model_seed": unit.seed, "resolved_config_sha256": resolved_config_sha256(command),
                "completion_manifest_sha256": manifest_sha256,
                "completion_code_sha256": dict(code_sha256 or {}),
                "parent_v3_template_identity": original_identity}
    identity = json.loads(json.dumps(identity, sort_keys=True))
    env = {**env, "DEEPMZYME_PARSE_CACHE_DIR": str(paths.parse_cache),
           "DEEPMZYME_GRAPH_CACHE_DIR": str(paths.graph_cache), "PYTHONDONTWRITEBYTECODE": "1"}
    return command + ["--campaign-run-identity", json.dumps(identity, sort_keys=True)], env, identity


def verify_training_inputs(parent, train_dir: Path) -> dict[str, Any]:
    """Check actual training bytes against v3's frozen source-cohort manifest."""
    import pmm_v3_campaign as v3
    from benchmarking import pmm_ion_cohort as cohort
    from training.access_guard import install_forbidden_read_guard

    train_dir = Path(train_dir).resolve()
    require(train_dir.name == "train", "Only the training-side directory named train is allowed")
    forbidden = [path.resolve() for path in v3.v2.forbidden_read_roots(train_dir)]
    install_forbidden_read_guard(forbidden)

    def guarded(path: Path) -> Path:
        resolved = path.resolve()
        require(not any(resolved == root or root in resolved.parents for root in forbidden),
                f"Training input resolves to a held-out path: {path}")
        return path

    frozen_path = guarded(parent.frozen / "campaign_manifest.json")
    frozen = json.loads(frozen_path.read_text())
    expected_metadata = frozen["train_metadata_sha256"]
    require(isinstance(expected_metadata, dict) and set(expected_metadata) == set(cohort.TRAIN_FILES),
            "Frozen training metadata must name the exact four input files")
    actual_metadata = {}
    for name, expected in expected_metadata.items():
        actual_metadata[name] = sha256_file(guarded(train_dir / name))
        require(actual_metadata[name] == expected, f"Training metadata changed: {name}")
    sha_by_name: dict[str, str] = {}
    for row in cohort.read_csv_rows(guarded(train_dir / "structure_manifest.csv")):
        name, digest = row.get("structure_name", ""), row.get("sha256", "")
        require(bool(name) and name == name.strip() and name not in (".", "..")
                and Path(name).name == name and "\\" not in name,
                "Structure names must be basenames")
        require(row.get("relative_path") == f"structures/{name}",
                f"Structure relative path does not match its name: {name}")
        require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
                f"Invalid structure SHA-256: {name}")
        require(name not in sha_by_name or sha_by_name[name] == digest,
                f"Conflicting duplicate structure identity: {name}")
        sha_by_name[name] = digest
    require(type(frozen["n_structure_files"]) is int and len(sha_by_name) == frozen["n_structure_files"]
            and bool(sha_by_name), "Frozen structure count does not match the manifest")
    aggregate = cohort._aggregate_structure_hash(sha_by_name)
    require(aggregate == frozen["structure_content_sha256"], "Frozen structure manifest identity changed")
    for name in sha_by_name:
        guarded(train_dir / "structures" / name)
    problems = cohort.verify_structure_hashes(train_dir, sha_by_name, set(sha_by_name), workers=1)
    require(not problems, f"Training structure bytes changed: {len(problems)} file(s)")
    cohort_path = guarded(parent.cohort)
    cohort_sha = sha256_file(cohort_path)
    require(cohort_sha == frozen["cohort"]["sha256"], "Frozen cohort changed")
    for row in cohort.read_csv_rows(cohort_path):
        require(sha_by_name.get(row["structure_name"]) == row["structure_sha256"],
                f"Cohort structure identity disagrees with training manifest: {row['structure_name']}")
    return {"schema_version": 1, "train_dir": str(train_dir),
            "frozen_input_manifest_sha256": sha256_file(frozen_path),
            "train_metadata_sha256": actual_metadata, "structure_content_sha256": aggregate,
            "n_structure_files": len(sha_by_name), "cohort_sha256": cohort_sha}


def verify_manifest(manifest_path: Path, expected_sha256: str, *, esm_dir: Path) -> dict[str, Any]:
    """Read-only verification; scientific preparation cannot grant launch permission."""
    import pmm_v3_campaign as v3
    from training.access_guard import install_forbidden_read_guard

    require(sha256_file(manifest_path) == expected_sha256, "Completion manifest changed")
    manifest = json.loads(Path(manifest_path).read_text())
    require(manifest.get("schema_version") == 1 and manifest.get("campaign_id") == CAMPAIGN_ID,
            "Wrong completion manifest schema or campaign")
    require(isinstance(manifest.get("esm_dir"), str) and bool(manifest["esm_dir"].strip())
            and Path(esm_dir).resolve() == Path(manifest["esm_dir"]).resolve(),
            "Supplied esm_dir does not match the manifest")
    require(isinstance(manifest.get("train_dir"), str) and bool(manifest["train_dir"].strip()),
            "Manifest must bind train_dir")
    install_forbidden_read_guard(v3.v2.forbidden_read_roots(Path(manifest["train_dir"])))
    require(Path(manifest["code_root"]).resolve() == ROOT == v3.ROOT.resolve(),
            "Worker and immutable v3 modules must execute from the manifest's code package")
    codes = manifest["code_sha256"]
    require({"metal_completion_campaign.py", "run_metal_completion.py", "metal_completion_safety.py"}
            <= set(codes), "Manifest lacks mandatory worker/safety code hashes")
    for name, digest in codes.items():
        path = ROOT / name
        require(not Path(name).is_absolute() and path.resolve().is_relative_to(ROOT)
                and path.is_file() and sha256_file(path) == digest, f"Executable code changed: {name}")
    parent = v3.V3Paths(Path(manifest["parent_campaign_dir"]))
    isolated_paths(parent.root, Path(manifest["successor_root"]))
    require(sha256_file(parent.manifest) == manifest["parent_manifest_sha256"], "Parent manifest changed")
    extension = sha256_file(parent.extension) if parent.extension.exists() else None
    require(extension == manifest.get("parent_extension_sha256"), "Parent extension changed")
    verified = v3.verify_campaign(parent, esm_dir=esm_dir)
    v3.require_frozen_spec(parent)
    require(verified["frozen_source_tree_sha256"] == manifest["source_tree_sha256"],
            "Source tree does not match the immutable parent")
    require(verify_training_inputs(parent, Path(manifest["train_dir"]))
            == manifest.get("training_input_verification"), "Training input verification receipt changed")
    for key, actual in (("parent_cohort_sha256", verified["cohort"]["sha256"]),
                        ("parent_fold_membership_sha256", verified["fold_set"]["fold_membership_sha256"])):
        if key in manifest:
            require(manifest[key] == actual, f"{key} changed")
    units = manifest["authorized_grid_units"]
    require(isinstance(units, list) and units and len(units) == len(set(units)), "Malformed scientific unit grid")
    parsed = [Unit.parse(name) for name in units]
    phase = parsed[0].phase
    require(all(unit.phase == phase for unit in parsed), "One manifest must name one phase")
    if phase == "discovery":
        expected = discovery_units()
    else:
        expected = confirmation_units(manifest["shared_recipe"])
        receipt = Path(manifest["selection_receipt_path"])
        require(sha256_file(receipt) == manifest["selection_receipt_sha256"], "Frozen recipe selection changed")
        selection = json.loads(receipt.read_text())
        selection = selection.get("result", selection.get("assessment", selection))
        require(selection.get("phase") == "discovery" and selection.get("complete") is True
                and selection.get("shared_recipe") == manifest["shared_recipe"],
                "Confirmation recipe does not match the complete discovery selection")
    require(set(units) == {unit.name for unit in expected}, "Incomplete or expanded scientific grid")
    require(sha256_file(Path(manifest["assessment_spec_path"])) == manifest["assessment_spec_sha256"],
            "Assessment specification changed")
    return manifest
