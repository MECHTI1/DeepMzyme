"""Inner-fold selection steps and the development gap gate of the EC class versus CLEAN30 campaign (WP-11).

``select`` turns the verified inner fits of one outer fold into a hash-chained step record:

    select --fold K --step {S1,S2,S3} --plan <campaign root>/plans/fold<K>.json --runs R [--runs R2 ...]
           --components C --freeze1-sha256 F --out STEP_RECORDS [--smoke]
           [--freeze1c FILE --l3-cost-disposition FILE]

- The frozen plan ``--plan`` must hash to its own ``plan_sha256`` (``campaign.plan_units``) and equal the
  live ``campaign.build_fold_plan(K)``; every required fit comes from it.
- A fit is a runner record ``<runs>/<unit>.runner.json`` with its run folder ``<runs>/<unit>/``. Every runner
  record in the runs folders must belong to fold K; a unit in two runs folders, a missing required fit, two
  fits at the same inner fold and seed, or a record of the step that the plan does not hold (or whose identity
  disagrees with its plan unit and recipe) refuses the step. A used fit must be covered by a passed
  ``verify_unit_receipt.json`` of its runs folder, and its ``ec_prediction_receipt.json`` must bind the
  terminal checkpoint, ``run_config.json``, the protein predictions and the runner identity by SHA-256.
- Out-of-fold rows are read from ``ec_val_protein_predictions.csv`` (protein ID: the accession; true class:
  ``y``; training priors for L9 from the receipt's ``training_protein_class_counts``, add-one smoothed) and
  pooled per configuration with ``selection.pool_inner_oof``. Training class counts for the recall guard are
  the pooled true classes (every eligible protein once). Inner identity components come from the data
  overlay's copies (``--components``: ``<overlay>/identity/fold<K>`` or its ``components_inner.tsv``);
  anything under ``<campaign root>/identity`` is refused. The pooled proteins must be exactly the certified
  membership's, each in its own inner fold, and every fit's identity must carry that membership's SHA-256.
- S1: learning rate per family (``s1_choose_learning_rate``) and the family escalation
  (``s1_family_escalation``) at ``promotion_config_for(None)``.
- S2: each lever alone against the S1 winner (re-pooled from its fits and matched to the S1 record's hash).
  L5 counts as tested only through ``certify_l5`` (below) and then ``screen_lever(positive_step_fraction=...)``;
  L7 through ``screen_epoch_lever`` and L9 through ``screen_logit_adjustment``; every promotion uses
  ``promotion_config_for(lever)`` (D1: alpha 0.04, L9 0.01). Inapplicable levers are logged "not applicable".
- S3: ``s3_combination`` at ``promotion_config_for(None)``; passing challengers are re-pooled and matched to
  the S2 record's hashes, and a passed L5 is certified again from its eight S2 fits at the recorded fraction,
  each fit bound to the S2 record by its SHA-256 values. S3 coverage is descriptive, not another 90% gate.
  L9 plus one fitted lever reuses that lever's S2 fits with the fixed S2 tau; no fits or tau search are added. Each S1
  and S3 record names the selected configuration (its units, recipe, tau and predictions hash), which the gap
  gate reads.
- Records are appended with ``selection.append_step_record`` anchored on the Freeze-1 SHA-256; one record per
  fold and step. A chain that does not verify refuses the step. S2 and S3 require the inner tables of
  ``--components`` to be the ones the earlier records of the fold used (all fields but the folder). ``--out``
  may not lie under ``<campaign root>/identity``, inside the ``--components`` folder or inside a data overlay
  (``simulation.check_out_folder``): step and gate records hold per-fold class numbers.

L5 certification (maintainer's requirements of 2026-11-01). ``certify_l5`` derives the eight required fits
(``N_INNER_FOLDS`` x ``SCREENING_SEEDS``) from the frozen plan and refuses a missing, duplicate or foreign fit
(any runner record whose recipe applies L5 outside the eight, or whose identity disagrees with its plan unit).
For each fit it checks the hash bindings, the plan's L5 settings in ``run_config["config"]``, the per-epoch
counters of ``run_config["history"]`` against the terminal checkpoint's history (loaded only after its SHA-256
matched), the history's length against the receipt's completed epochs (the exporter writes ``len(history)``)
and the receipt's ``ec_supcon_coverage`` against totals recomputed from that history. Integrity problems
refuse the step. Counters that are absent or invalid on an otherwise intact fit (no counter keys,
zero batches, more positives than batches, booleans, non-contiguous epochs) leave L5 uncertified: fraction
None, L5 logged "not tested" with the reason, never assumed to pass. The certified fraction is the minimum
share over the eight fits (``selection.supcon_positive_step_fraction``) under the 90% rule.

``gap-gate`` (fold 0 only; validation only; can only stop or continue):

    gap-gate --fold 0 --after {S1,S3} --step-chain STEP_RECORDS --freeze1-sha256 F --plan P --runs R [...]
             --components C --clean-r1-inner NPZ (x4) --clean-r2-inner NPZ (x4) --out G [--smoke]

DeepMzyme's selected configuration (re-pooled and matched to the record's hash) and the CLEAN inner arms
(contract C4 exports, one per inner fold and arm) are scored on the inner-validation proteins whose full EC
numbers all occur among the R1-inner training centres of their inner fold. DeepMzyme's score is the seed mean
of its per-seed EC class 1-6 balanced accuracy; a CLEAN tie counts as wrong. The gap is DeepMzyme minus
CLEAN-R1 (after S3, minus the 1.0-point optimism discount first); at or above -3.0 points the decision is
``continue``, otherwise ``maintainer_decision_required``. Denominators, the seven-class gap and EC class 7
outcomes are reported; seven-class and R2-inner gaps are undiscounted and never gated on. CLEAN query IDs
must exactly equal the certified validation membership of each inner fold before closure filtering. No output names a protein; printed summaries
hold totals only (per-class counts of the fold stay in the written records).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from ec_clean30.campaign import (
    CAMPAIGN_ID, KIND_INNER, LEVER_CONFIGS, STEP_S1, STEP_S2, STEP_S3, UNIT_PREFIX, CampaignPlanError, PlannedUnit,
    Recipe, build_fold_plan, plan_units, recipe_with_lever, unit_id as planned_unit_id,
)
from ec_clean30.commands import VERIFY_UNIT_RECEIPT_NAME
from ec_clean30.provenance import (
    provenance_record, real_action_git_state, refuse_test_paths, sha256_file, write_new_json,
)
from ec_clean30.selection import (
    EPOCH_LEVER, EPOCH_LOOKS, FLOAT_TOLERANCE, LEVER_PASSED, LEVERS, LOGIT_ADJUSTMENT_LEVER,
    N_INNER_FOLDS, POINTS, SCREENING_SEEDS, SUPCON_LEVER, SUPCON_MIN_POSITIVE_STEP_FRACTION, InnerFoldPredictions,
    LeverOutcome, LeverScreen, PooledOof, PromotionConfig, apply_tau, append_step_record, learning_rate_label,
    lever_applies, lever_not_applicable, lever_not_tested, pool_inner_oof, predictions_sha256, promotion_config_for,
    read_step_chain, record_sha256, s1_choose_learning_rate, s1_family_escalation, s2_lever_screen, s3_combination,
    screen_epoch_lever, screen_lever, screen_logit_adjustment, supcon_lever_testable,
    supcon_positive_step_fraction, verify_step_chain,
)
from training.ec_class_metrics import N_EC_CLASSES, PRIMARY_CLASS_INDICES
from training.ec_classes import (
    EC_CLASS_TOKENS, STRUCTURE_ID_RE, accession_from_structure_id, ec_numbers_from_structure_id,
)


STEP_RECORD_FORMAT = "deepmzyme_ec_clean30_step_record_v1"
GAP_GATE_FORMAT = "deepmzyme_ec_clean30_gap_gate_v1"
SCHEMA_VERSION = 1
SELECT_STEPS: tuple[str, ...] = (STEP_S1, STEP_S2, STEP_S3)
GAP_GATE_AFTER: tuple[str, ...] = (STEP_S1, STEP_S3)
GAP_GATE_FOLD = 0
GAP_CONTINUE_POINTS = -3.0
S3_OPTIMISM_DISCOUNT_POINTS = 1.0
DECISION_CONTINUE = "continue"
DECISION_MAINTAINER = "maintainer_decision_required"
CLEAN_ARMS: tuple[str, ...] = ("r1_inner", "r2_inner")
# The gate denominators a printed summary may show (totals; never per-class or per-inner-fold counts).
GAP_GATE_PRINTED_TOTALS: tuple[str, ...] = (
    "n_inner_validation_proteins", "n_in_r1_inner_closure", "n_outside_closure", "ec_classes_1_6_averaged",
)

# File names of a finished inner fit (training.ec_prediction_export and training.run, which load torch; a test
# keeps these equal to the originals).
RUNNER_SUFFIX = ".runner.json"
TERMINAL_CHECKPOINT_NAME = "terminal_model_checkpoint.pt"
EC_PREDICTION_RECEIPT_NAME = "ec_prediction_receipt.json"
RUN_CONFIG_NAME = "run_config.json"
PROTEIN_PREDICTIONS_NAME = "ec_val_protein_predictions.csv"
LOGIT_COLUMNS: tuple[str, ...] = tuple(f"logit_{token}" for token in EC_CLASS_TOKENS)
PROTEIN_PREDICTION_COLUMNS: tuple[str, ...] = (
    "run_id", "mode", "fold_index", "n_folds", "seed", "accession", "ec_group_key", "structure_ids", "n_pockets",
    "y", "pred", *LOGIT_COLUMNS,
)
# training.loop counter keys (per epoch, in the fit history).
SUPCON_BATCHES_KEY = "ec_supcon_train_batches"
SUPCON_BATCHES_WITH_POSITIVE_KEY = "ec_supcon_batches_with_positive"
PRIOR_SMOOTHING = 1.0  # logit_adjustment.PRIOR_SMOOTHING: (n_c + 1) / (N + 7)

PLANS_FOLDER = "plans"
PLAN_FILE_NAME = "fold{fold}.json"
COMPONENTS_FILE_NAME = "components_inner.tsv"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

SELECT_HELP = ("Inner-fold selection step S1, S2 or S3 of one outer fold from verified fits; appends a hash-chained "
               "step record (L5 certified from hash-bound per-epoch counters).")
GAP_GATE_HELP = ("Fold-0 development gap gate after S1 or S3: DeepMzyme's selected inner OOF against the CLEAN inner "
                 "arms on the R1-inner EC closure; continue at a gap of -3.0 points or more.")


class SelectionInputError(ValueError):
    """``select`` or ``gap-gate`` refuses its inputs; no step record or gate record is written."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise SelectionInputError(message)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _read_json(path: Path, what: str) -> Any:
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), f"{what}: {path.name} is missing or a link.")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise SelectionInputError(f"{what}: {path.name} is not valid JSON ({exc.__class__.__name__}).") from exc


def _check_sha256(value: Any, what: str) -> str:
    _require(isinstance(value, str) and bool(_SHA256_RE.match(value)), f"{what} must be a lower-case SHA-256.")
    return value


def _recipe_key(recipe: Mapping[str, Any] | Recipe) -> dict[str, Any]:
    """A recipe as a comparable record (learning-rate items and lever configurations in sorted order)."""
    record = recipe.to_dict() if isinstance(recipe, Recipe) else dict(recipe or {})
    learning_rate = record.get("learning_rate") or {}
    return {"family": record.get("family"),
            "learning_rate": sorted([str(key), value] for key, value in dict(learning_rate).items()),
            "epochs": record.get("epochs"),
            "lever_configs": sorted(str(item) for item in record.get("lever_configs") or [])}


def recipe_from_record(record: Mapping[str, Any]) -> Recipe:
    """The ``Recipe`` of a ``Recipe.to_dict()`` record (a step record or a runner identity)."""
    try:
        return Recipe(record["family"], tuple(dict(record["learning_rate"]).items()), int(record["epochs"]),
                      tuple(record.get("lever_configs") or ()))
    except (KeyError, TypeError, ValueError) as exc:
        raise SelectionInputError(f"Recipe record is not a campaign recipe ({exc.__class__.__name__}).") from exc


# ---------------------------------------------------------------------------
# Frozen plan


def load_frozen_plan(path: Path, fold: int) -> dict[str, Any]:
    """``<campaign root>/plans/fold<K>.json``: its own hash rechecked and equal to the live plan of fold K."""
    path = Path(path)
    _require(path.name == PLAN_FILE_NAME.format(fold=fold) and path.parent.name == PLANS_FOLDER,
             f"--plan must be <campaign root>/{PLANS_FOLDER}/{PLAN_FILE_NAME.format(fold=fold)}, got {path.name}.")
    plan = _read_json(path, "--plan")
    _require(isinstance(plan, dict) and plan.get("fold") == fold,
             f"--plan holds fold {plan.get('fold') if isinstance(plan, dict) else None!r}, not {fold}.")
    try:
        plan_units(plan)
        live = build_fold_plan(fold)["plan_sha256"]
    except (CampaignPlanError, KeyError, TypeError) as exc:
        raise SelectionInputError(f"--plan does not verify: {exc}") from exc
    _require(plan["plan_sha256"] == live, f"--plan's plan_sha256 differs from the live plan of fold {fold}.")
    return plan


def _plan_unit_map(plan: Mapping[str, Any]) -> dict[str, PlannedUnit]:
    return {unit.unit_id: unit for unit in plan_units(plan)}


def _grid_ids(fold: int, step: str, config: str) -> list[str]:
    return [planned_unit_id(UNIT_PREFIX, fold, step, config, inner_fold=inner, seed=seed)
            for inner in range(N_INNER_FOLDS) for seed in SCREENING_SEEDS]


def required_lever_units(plan: Mapping[str, Any], config_id: str = SUPCON_LEVER) -> tuple[PlannedUnit, ...]:
    """The S2 units of one lever configuration in the frozen plan: exactly inner folds x screening seeds.

    For L5 these are the eight fits ``ecc30__fold<K>__S2__L5__inner<i>__seed<s>`` that certification needs.
    """
    fold = int(plan["fold"])
    units = [unit for unit in plan_units(plan) if unit.step == STEP_S2 and unit.lever_configs == (config_id,)]
    expected = _grid_ids(fold, STEP_S2, config_id)
    _require(sorted(unit.unit_id for unit in units) == sorted(expected),
             f"The plan holds {len(units)} S2 units of {config_id}; expected the {len(expected)} of "
             f"{N_INNER_FOLDS} inner folds x seeds {list(SCREENING_SEEDS)}.")
    for unit in units:
        _require(unit.fold == fold and unit.kind == KIND_INNER and unit.inner_fold in range(N_INNER_FOLDS)
                 and unit.seed in SCREENING_SEEDS, f"Plan unit {unit.unit_id} is not an inner screening fit.")
    order = {unit_id: position for position, unit_id in enumerate(expected)}
    return tuple(sorted(units, key=lambda unit: order[unit.unit_id]))


def _lever_options(plan: Mapping[str, Any], config_id: str) -> dict[str, Any]:
    """The plan's trainer options of one configuration as run-configuration keys (``--ec-supcon-weight`` ->
    ``ec_supcon_weight``)."""
    config = plan["constants"]["lever_configs"][config_id]
    return {str(flag).lstrip("-").replace("-", "_"): value for flag, value in config["options"]}


# ---------------------------------------------------------------------------
# Runner records and fit collection


@dataclass(frozen=True)
class FitRecord:
    """One runner record ``<runs>/<unit>.runner.json`` and its identity."""

    unit_id: str
    runs_dir: Path
    runner: Mapping[str, Any] = field(repr=False)
    identity: Mapping[str, Any] = field(repr=False)
    runner_sha256: str = ""

    @property
    def run_dir(self) -> Path:
        return self.runs_dir / self.unit_id

    @property
    def step(self) -> Any:
        return self.identity.get("step")

    @property
    def lever_configs(self) -> tuple[str, ...]:
        return tuple(str(item) for item in (self.identity.get("recipe") or {}).get("lever_configs") or ())

    @property
    def position(self) -> tuple[Any, Any]:
        return self.identity.get("inner_fold"), self.identity.get("seed")


def read_runner_records(runs_dirs: Sequence[Path], *, fold: int) -> dict[str, FitRecord]:
    """Every runner record of the runs folders; refuses a unit in two folders and any record of another fold."""
    _require(len(runs_dirs) > 0, "At least one --runs folder is needed.")
    resolved = [Path(path).resolve() for path in runs_dirs]
    _require(len(set(resolved)) == len(resolved), "A --runs folder is given more than once.")
    records: dict[str, FitRecord] = {}
    prefix = f"{UNIT_PREFIX}__fold{fold}__"
    for runs in (Path(path) for path in runs_dirs):
        _require(runs.is_dir(), f"--runs {runs} is not a folder.")
        for path in sorted(runs.iterdir()):
            if not path.name.endswith(RUNNER_SUFFIX):
                continue
            unit_id = path.name[: -len(RUNNER_SUFFIX)]
            _require(unit_id not in records, f"Duplicate fit: unit {unit_id} has runner records in two runs folders.")
            runner = _read_json(path, unit_id)
            unit = runner.get("unit") if isinstance(runner, dict) else None
            identity = unit.get("identity") if isinstance(unit, dict) else None
            _require(isinstance(identity, dict), f"{unit_id}: the runner record holds no unit identity.")
            _require(runner.get("unit_id") == unit.get("unit_id") == identity.get("unit_id") == unit_id,
                     f"{unit_id}: the runner record names another unit.")
            _require(identity.get("campaign_id") == CAMPAIGN_ID, f"{unit_id}: the identity names another campaign.")
            _require(unit_id.startswith(prefix) and identity.get("fold") == fold,
                     f"Foreign fit: {unit_id} (identity fold {identity.get('fold')!r}) is not a fit of fold {fold}.")
            for key in ("step", "kind", "seed", "inner_fold"):
                _require(unit.get(key) == identity.get(key), f"{unit_id}: the runner record's {key} differs from its "
                                                             "identity.")
            _require(_recipe_key(unit.get("recipe") or {}) == _recipe_key(identity.get("recipe") or {}),
                     f"{unit_id}: the runner record's recipe differs from its identity.")
            records[unit_id] = FitRecord(unit_id, runs, runner, identity, sha256_file(path))
    return records


def _identity_problems(record: FitRecord, unit: PlannedUnit, recipe: Recipe, *, freeze1_sha256: Optional[str],
                       smoke: bool) -> list[str]:
    identity = record.identity
    expected = {"unit_id": unit.unit_id, "fold": unit.fold, "step": unit.step, "kind": unit.kind,
                "inner_fold": unit.inner_fold, "seed": unit.seed}
    problems = [key for key, value in expected.items()
                if identity.get(key) != value or isinstance(identity.get(key), bool)]
    if _recipe_key(identity.get("recipe") or {}) != _recipe_key(recipe):
        problems.append("recipe")
    if identity.get("smoke") is not bool(smoke):
        problems.append("smoke")
    if freeze1_sha256 is not None and identity.get("freeze1_sha256") != freeze1_sha256:
        problems.append("freeze1_sha256")
    return problems


def collect_fits(
    records: Mapping[str, FitRecord],
    required: Sequence[tuple[PlannedUnit, Recipe]],
    *,
    owns: Callable[[FitRecord], bool],
    what: str,
    freeze1_sha256: Optional[str] = None,
    smoke: bool = False,
) -> dict[str, FitRecord]:
    """The runner records of ``required`` (plan unit, expected recipe), in that order.

    ``owns`` marks the records that belong to this group of fits (for L5: every recipe that applies L5).
    Refused: two records of one recipe at the same inner fold and seed, a missing required unit, an owned record that is
    not required (foreign), and a required record whose identity disagrees with its plan unit or recipe.
    """
    wanted = {unit.unit_id: (unit, recipe) for unit, recipe in required}
    _require(len(wanted) == len(required), f"{what}: a required unit is listed twice.")
    involved = {record.unit_id: record for record in records.values() if owns(record)}
    involved.update({unit_id: records[unit_id] for unit_id in wanted if unit_id in records})
    by_position: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for record in involved.values():
        recipe_key = json.dumps(_recipe_key(record.identity.get("recipe") or {}), sort_keys=True, default=str)
        by_position[(recipe_key, *record.position)].append(record.unit_id)
    repeated = sorted((key[1:], sorted(ids)) for key, ids in by_position.items() if len(ids) > 1)
    _require(not repeated, f"{what}: duplicate fits of one recipe at the same inner fold and seed: "
                           + "; ".join(f"inner {key[0]} seed {key[1]}: {ids}" for key, ids in repeated[:4]))
    missing = [unit_id for unit_id in wanted if unit_id not in records]
    _require(not missing, f"{what}: {len(missing)} required fits are missing: {missing[:8]}.")
    foreign = sorted(unit_id for unit_id in involved if unit_id not in wanted)
    _require(not foreign, f"{what}: foreign fits not in the frozen plan's grid: {foreign[:8]}.")
    for unit_id, (unit, recipe) in wanted.items():
        problems = _identity_problems(records[unit_id], unit, recipe, freeze1_sha256=freeze1_sha256, smoke=smoke)
        _require(not problems,
                 f"{what}: foreign fit {unit_id}: its identity disagrees with the plan unit in {problems}.")
    return {unit_id: records[unit_id] for unit_id in wanted}


def collect_lever_fits(
    runs_dirs: Sequence[Path],
    plan: Mapping[str, Any],
    *,
    s1_winner: Recipe,
    config_id: str = SUPCON_LEVER,
    freeze1_sha256: Optional[str] = None,
    smoke: bool = False,
    records: Optional[Mapping[str, FitRecord]] = None,
) -> dict[str, FitRecord]:
    """The S2 fits of one lever configuration on the S1 winner, from the frozen plan's grid.

    Every runner record whose recipe applies the configuration counts (so a stray L5 fit of another step is
    foreign), and every record must belong to the plan's fold.
    """
    fold = int(plan["fold"])
    if records is None:
        records = read_runner_records(runs_dirs, fold=fold)
    try:
        recipe = recipe_with_lever(s1_winner, config_id)
    except CampaignPlanError as exc:
        raise SelectionInputError(f"{config_id} cannot be applied to the S1 winner: {exc}") from exc
    required = [(unit, recipe) for unit in required_lever_units(plan, config_id)]
    return collect_fits(records, required, owns=lambda record: config_id in record.lever_configs,
                        what=f"{config_id} fits", freeze1_sha256=freeze1_sha256, smoke=smoke)


# ---------------------------------------------------------------------------
# Hash-bound fit evidence


@dataclass(frozen=True)
class FitEvidence:
    """A fit whose receipts bind each other: checkpoint, run configuration, predictions and identity."""

    unit_id: str
    inner_fold: int
    seed: int
    receipt: Mapping[str, Any] = field(repr=False)
    run_config: Mapping[str, Any] = field(repr=False)
    checkpoint_sha256: str = ""
    run_config_sha256: str = ""
    receipt_sha256: str = ""
    protein_predictions_sha256: str = ""
    runner_sha256: str = ""

    def record(self) -> dict[str, Any]:
        return {"unit_id": self.unit_id, "inner_fold": self.inner_fold, "seed": self.seed,
                "checkpoint_sha256": self.checkpoint_sha256, "run_config_sha256": self.run_config_sha256,
                "prediction_receipt_sha256": self.receipt_sha256,
                "protein_predictions_sha256": self.protein_predictions_sha256, "runner_sha256": self.runner_sha256}


class FitVerifier:
    """Reads each runs folder's ``verify_unit_receipt.json`` once."""

    def __init__(self) -> None:
        self._receipts: dict[Path, tuple[str, dict[str, Mapping[str, Any]]]] = {}

    def verified_entry(self, runs_dir: Path, unit_id: str) -> Mapping[str, Any]:
        key = Path(runs_dir).resolve()
        if key not in self._receipts:
            receipt = _read_json(Path(runs_dir) / VERIFY_UNIT_RECEIPT_NAME, f"runs folder {Path(runs_dir).name}")
            units = receipt.get("units") if isinstance(receipt, dict) else None
            _require(isinstance(units, list), f"{VERIFY_UNIT_RECEIPT_NAME} lists no units.")
            entries: dict[str, Mapping[str, Any]] = {}
            for entry in units:
                if isinstance(entry, dict) and isinstance(entry.get("unit_id"), str):
                    _require(entry["unit_id"] not in entries,
                             f"{VERIFY_UNIT_RECEIPT_NAME} lists {entry['unit_id']} twice.")
                    entries[entry["unit_id"]] = entry
            self._receipts[key] = (str(receipt.get("status")), entries)
        status, entries = self._receipts[key]
        _require(status == "passed", f"{VERIFY_UNIT_RECEIPT_NAME} of {Path(runs_dir).name} has status {status!r}, "
                                     "not 'passed'.")
        entry = entries.get(unit_id)
        _require(entry is not None and entry.get("passed") is True,
                 f"{unit_id} is not covered as passed by {VERIFY_UNIT_RECEIPT_NAME}.")
        return entry


def verify_fit_evidence(record: FitRecord, *, verifier: Optional[FitVerifier] = None) -> FitEvidence:
    """Check the hash bindings of one finished inner fit (integrity problems raise ``SelectionInputError``)."""
    verifier = verifier or FitVerifier()
    unit_id, identity = record.unit_id, record.identity
    run_dir = record.run_dir
    _require(run_dir.is_dir() and not run_dir.is_symlink(), f"{unit_id}: the run folder is missing.")
    _require(record.runner.get("returncode") == 0 and record.runner.get("status") == "completed",
             f"{unit_id}: the runner record is not a completed run.")
    entry = verifier.verified_entry(record.runs_dir, unit_id)
    checkpoint = run_dir / TERMINAL_CHECKPOINT_NAME
    _require(checkpoint.is_file() and not checkpoint.is_symlink(), f"{unit_id}: {TERMINAL_CHECKPOINT_NAME} is missing.")
    checkpoint_sha256 = sha256_file(checkpoint)
    _require(entry.get("terminal_checkpoint_sha256") == checkpoint_sha256,
             f"{unit_id}: the terminal checkpoint SHA-256 differs from {VERIFY_UNIT_RECEIPT_NAME}.")
    receipt_path = run_dir / EC_PREDICTION_RECEIPT_NAME
    receipt = _read_json(receipt_path, unit_id)
    config_path = run_dir / RUN_CONFIG_NAME
    run_config = _read_json(config_path, unit_id)
    _require(isinstance(receipt, dict) and isinstance(run_config, dict), f"{unit_id}: receipts must be JSON objects.")
    bound = receipt.get("checkpoint") or {}
    _require(bound.get("path") == TERMINAL_CHECKPOINT_NAME and bound.get("sha256") == checkpoint_sha256
             and bound.get("checkpoint_rule") == "terminal",
             f"{unit_id}: the prediction receipt's checkpoint SHA-256 differs from {TERMINAL_CHECKPOINT_NAME}.")
    run_config_sha256 = sha256_file(config_path)
    _require(receipt.get("run_config_sha256") == run_config_sha256,
             f"{unit_id}: the prediction receipt's run_config_sha256 differs from {RUN_CONFIG_NAME}.")
    _require(receipt.get("campaign_run_identity") == dict(identity),
             f"{unit_id}: the prediction receipt's campaign_run_identity differs from the runner identity.")
    config = run_config.get("config") or {}
    try:
        config_identity = json.loads(config.get("campaign_run_identity") or "null")
    except (TypeError, ValueError):
        config_identity = None
    _require(config_identity == dict(identity), f"{unit_id}: {RUN_CONFIG_NAME} carries another campaign identity.")
    epochs = (identity.get("recipe") or {}).get("epochs")
    recorded = receipt.get("epochs") or {}
    _require(_is_int(epochs) and recorded.get("planned") == epochs and recorded.get("completed") == epochs
             and bound.get("epoch") == epochs, f"{unit_id}: the receipt's epochs differ from the recipe's {epochs}.")
    inner_fold, seed = identity.get("inner_fold"), identity.get("seed")
    _require(receipt.get("mode") == "validation" and receipt.get("reconciliation_status") == "match"
             and receipt.get("fold_index") == inner_fold and receipt.get("n_folds") == N_INNER_FOLDS
             and receipt.get("seed") == seed,
             f"{unit_id}: the prediction receipt is not a reconciled validation export of inner fold {inner_fold}, "
             f"seed {seed}.")
    predictions = receipt.get("protein_predictions") or {}
    csv_path = run_dir / PROTEIN_PREDICTIONS_NAME
    _require(predictions.get("path") == PROTEIN_PREDICTIONS_NAME and csv_path.is_file() and not csv_path.is_symlink(),
             f"{unit_id}: {PROTEIN_PREDICTIONS_NAME} is missing.")
    predictions_sha256 = sha256_file(csv_path)
    _require(predictions.get("sha256") == predictions_sha256,
             f"{unit_id}: {PROTEIN_PREDICTIONS_NAME} differs from its prediction receipt.")
    return FitEvidence(unit_id, int(inner_fold), int(seed), receipt, run_config, checkpoint_sha256, run_config_sha256,
                       sha256_file(receipt_path), predictions_sha256, record.runner_sha256)


# ---------------------------------------------------------------------------
# L5: SupCon coverage certification


def _counter_view(history: Any) -> Optional[list[tuple[Any, ...]]]:
    """Per epoch: the epoch value and both counters, each with its type, so ``True`` never equals ``1``."""
    if not isinstance(history, list):
        return None
    view = []
    for record in history:
        if not isinstance(record, dict):
            return None
        items = []
        for key in ("epoch", SUPCON_BATCHES_KEY, SUPCON_BATCHES_WITH_POSITIVE_KEY):
            items.append((type(record[key]).__name__, record[key]) if key in record else ("absent", None))
        view.append(tuple(items))
    return view


def _exporter_totals(history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The receipt's ``ec_supcon_coverage`` counts as ``training.ec_prediction_export.supcon_coverage`` sums them."""
    complete = bool(history) and all(SUPCON_BATCHES_KEY in record and SUPCON_BATCHES_WITH_POSITIVE_KEY in record
                                     for record in history)
    try:
        batches = sum(int(record[SUPCON_BATCHES_KEY]) for record in history) if complete else None
        positive = sum(int(record[SUPCON_BATCHES_WITH_POSITIVE_KEY]) for record in history) if complete else None
    except (TypeError, ValueError) as exc:
        raise SelectionInputError("SupCon counters cannot be summed as the exporter sums them; the receipt cannot "
                                  "come from this history.") from exc
    return {"train_batches": batches, "batches_with_positive": positive,
            "epochs_counted": sum(SUPCON_BATCHES_KEY in record for record in history)}


def counter_problems(history: Sequence[Mapping[str, Any]], epochs: int) -> list[str]:
    """Why per-epoch SupCon counters do not certify a fit (empty: complete and valid).

    Valid: epochs 1..E in order with E = ``epochs``, and every epoch holds both counters as non-boolean
    integers with at least one training minibatch and 0 <= minibatches with a positive <= minibatches.
    """
    problems = []
    if len(history) != epochs:
        problems.append(f"the history holds {len(history)} epochs, not {epochs}")
    numbers = [record.get("epoch") for record in history]
    if not all(_is_int(value) for value in numbers) or numbers != list(range(1, len(history) + 1)):
        problems.append("epochs are not 1..E in order")
    for position, record in enumerate(history, start=1):
        batches = record.get(SUPCON_BATCHES_KEY)
        positive = record.get(SUPCON_BATCHES_WITH_POSITIVE_KEY)
        if SUPCON_BATCHES_KEY not in record or SUPCON_BATCHES_WITH_POSITIVE_KEY not in record:
            problems.append(f"epoch record {position} lacks the SupCon counters")
        elif not (_is_int(batches) and _is_int(positive)):
            problems.append(f"epoch record {position} holds counters that are not integers (booleans refused)")
        elif batches < 1:
            problems.append(f"epoch record {position} counts {batches} training minibatches")
        elif not 0 <= positive <= batches:
            problems.append(f"epoch record {position} counts {positive} minibatches with a positive of {batches}")
    return problems


@dataclass(frozen=True)
class SupconEvidence:
    """One L5 fit's verified counters: ``coverage`` is None when the counters do not certify it."""

    fit: FitEvidence
    coverage: Optional[Mapping[str, int]]
    problems: tuple[str, ...]
    recomputed: Mapping[str, Any]

    @property
    def share(self) -> Optional[float]:
        if self.coverage is None:
            return None
        return self.coverage["batches_with_positive"] / self.coverage["train_batches"]

    def record(self) -> dict[str, Any]:
        return {**self.fit.record(), "recomputed_totals": dict(self.recomputed),
                "certified_coverage": None if self.coverage is None else dict(self.coverage),
                "share": self.share, "problems": list(self.problems)}


def _load_checkpoint_history(path: Path, expected_sha256: str, unit_id: str) -> tuple[Any, Any, Any]:
    """The terminal checkpoint's history, epoch and configuration; loaded only if its SHA-256 still matches."""
    _require(sha256_file(path) == expected_sha256, f"{unit_id}: the terminal checkpoint changed after it was hashed.")
    import torch  # lazy: the module stays torch-free at import

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - any unpickling failure is an integrity problem
        raise SelectionInputError(
            f"{unit_id}: the terminal checkpoint does not load ({exc.__class__.__name__}).") from exc
    _require(isinstance(payload, dict), f"{unit_id}: the terminal checkpoint is not a checkpoint payload.")
    return payload.get("history"), payload.get("epoch"), payload.get("config")


def verify_supcon_evidence(record: FitRecord, plan: Mapping[str, Any], *,
                           verifier: Optional[FitVerifier] = None) -> SupconEvidence:
    """One L5 fit: hash bindings, the plan's L5 settings, the history's length versus the receipt's completed
    epochs, checkpoint versus run-configuration counters and the receipt's totals versus the recomputed ones (all
    refuse); then the counters' validity (uncertified)."""
    evidence = verify_fit_evidence(record, verifier=verifier)
    unit_id = record.unit_id
    config = evidence.run_config.get("config") or {}
    expected = _lever_options(plan, SUPCON_LEVER)
    wrong = sorted(key for key, value in expected.items()
                   if key not in config or config[key] != value or isinstance(config[key], bool))
    _require(not wrong, f"{unit_id}: {RUN_CONFIG_NAME} does not carry the plan's L5 settings ({wrong}).")
    weight = config.get("ec_supcon_weight")
    _require(isinstance(weight, (int, float)) and not isinstance(weight, bool) and weight > 0,
             f"{unit_id}: {RUN_CONFIG_NAME} has no positive ec_supcon_weight.")
    history = evidence.run_config.get("history")
    _require(isinstance(history, list) and all(isinstance(item, dict) for item in history),
             f"{unit_id}: {RUN_CONFIG_NAME} holds no epoch history.")
    completed = evidence.receipt["epochs"]["completed"]
    _require(len(history) == completed,
             f"{unit_id}: {RUN_CONFIG_NAME} holds {len(history)} epoch records, but the prediction receipt counts "
             f"{completed} completed epochs (the exporter writes the history's length).")
    saved_history, saved_epoch, saved_config = _load_checkpoint_history(
        record.run_dir / TERMINAL_CHECKPOINT_NAME, evidence.checkpoint_sha256, unit_id)
    _require(_counter_view(saved_history) == _counter_view(history),
             f"{unit_id}: the terminal checkpoint's per-epoch counters disagree with {RUN_CONFIG_NAME}.")
    _require(saved_epoch == evidence.receipt["checkpoint"]["epoch"],
             f"{unit_id}: the terminal checkpoint holds epoch {saved_epoch!r}, not the receipt's terminal epoch.")
    _require(isinstance(saved_config, dict), f"{unit_id}: the terminal checkpoint holds no run configuration.")
    differing = sorted(key for key in (*expected, "campaign_run_identity")
                       if saved_config.get(key) != config.get(key))
    _require(not differing, f"{unit_id}: the terminal checkpoint's configuration differs from {RUN_CONFIG_NAME} in "
                            f"{differing}.")
    recomputed = _exporter_totals(history)
    coverage = evidence.receipt.get("ec_supcon_coverage")
    _require(isinstance(coverage, dict), f"{unit_id}: the prediction receipt holds no ec_supcon_coverage.")
    disagree = sorted(key for key, value in recomputed.items()
                      if coverage.get(key) != value or type(coverage.get(key)) is not type(value))
    _require(not disagree, f"{unit_id}: the receipt's ec_supcon_coverage disagrees with the totals recomputed from "
                           f"the hash-bound history in {disagree}.")
    problems = counter_problems(history, int(completed))
    certified = None
    if not problems:
        _require(recomputed["epochs_counted"] == len(history), f"{unit_id}: epochs_counted differs from the history.")
        certified = {"train_batches": recomputed["train_batches"],
                     "batches_with_positive": recomputed["batches_with_positive"], "epochs_counted": len(history)}
    return SupconEvidence(evidence, certified, tuple(problems), recomputed)


@dataclass(frozen=True)
class L5Certification:
    """L5's certified positive-step fraction (None: uncertified) over its eight required fits."""

    fraction: Optional[float]
    reason: str
    fits: tuple[SupconEvidence, ...]
    aggregation: Mapping[str, Any]

    @property
    def testable(self) -> bool:
        return self.fraction is not None and supcon_lever_testable(self.fraction)

    def record(self) -> dict[str, Any]:
        return {"positive_step_fraction": self.fraction, "testable": self.testable, "reason": self.reason,
                "rule": f"minimum share over the {len(self.fits)} required fits, at least "
                        f"{SUPCON_MIN_POSITIVE_STEP_FRACTION:.0%}",
                "fits": [fit.record() for fit in self.fits], "aggregation": dict(self.aggregation)}


def certify_l5(
    plan: Mapping[str, Any],
    runs_dirs: Sequence[Path],
    *,
    s1_winner: Recipe,
    freeze1_sha256: Optional[str] = None,
    smoke: bool = False,
    records: Optional[Mapping[str, FitRecord]] = None,
    verifier: Optional[FitVerifier] = None,
) -> L5Certification:
    """Certify L5 from the frozen plan's eight fits (see the module docstring); integrity problems raise."""
    fits = collect_lever_fits(runs_dirs, plan, s1_winner=s1_winner, config_id=SUPCON_LEVER,
                              freeze1_sha256=freeze1_sha256, smoke=smoke, records=records)
    verifier = verifier or FitVerifier()
    evidence = tuple(verify_supcon_evidence(record, plan, verifier=verifier) for record in fits.values())
    fraction, aggregation = supcon_positive_step_fraction({item.fit.unit_id: item.coverage for item in evidence})
    if fraction is None:
        failing = [item for item in evidence if item.coverage is None]
        reason = (f"L5 uncertified: {len(failing)} of {len(evidence)} required fits lack complete, valid SupCon "
                  f"counters ({failing[0].fit.unit_id}: {failing[0].problems[0]})")
    elif supcon_lever_testable(fraction):
        reason = f"L5 certified: minimum share {fraction:.4f} of training minibatches with a SupCon positive"
    else:
        reason = (f"SupCon positives on {fraction:.1%} of training minibatches (minimum over the required fits), "
                  f"below {SUPCON_MIN_POSITIVE_STEP_FRACTION:.0%}")
    return L5Certification(fraction, reason, evidence, aggregation)


# ---------------------------------------------------------------------------
# Out-of-fold predictions


@dataclass(frozen=True, eq=False)
class FitPredictions:
    """One fit's protein rows: ``InnerFoldPredictions`` plus the true classes and full EC numbers per protein."""

    part: InnerFoldPredictions
    true_class: Mapping[str, int] = field(repr=False)
    ec_numbers: Mapping[str, tuple[str, ...]] = field(repr=False)


def training_priors(counts: Any, unit_id: str) -> np.ndarray:
    """Add-one smoothed priors ``(n_c + 1) / (N + 7)`` from ``training_protein_class_counts``."""
    _require(isinstance(counts, dict) and sorted(counts) == sorted(EC_CLASS_TOKENS)
             and all(_is_int(counts[token]) and counts[token] >= 0 for token in EC_CLASS_TOKENS),
             f"{unit_id}: training_protein_class_counts must hold one count per EC class 1-7.")
    values = np.array([counts[token] for token in EC_CLASS_TOKENS], dtype=np.float64)
    return (values + PRIOR_SMOOTHING) / (values.sum() + PRIOR_SMOOTHING * N_EC_CLASSES)


def split_structure_ids(value: str) -> tuple[str, ...]:
    """The structure IDs of a protein row's ``structure_ids`` (joined with ";" by the exporter).

    An EC list inside one structure ID may itself hold ";" (``ACC__chain_A__EC_1.1.1.1;2.7.1.1``), so a
    piece without ``__chain_`` continues the EC list of the structure ID before it.
    """
    structures: list[str] = []
    for piece in (part.strip() for part in str(value).split(";")):
        if not piece:
            continue
        if STRUCTURE_ID_RE.match(piece):
            structures.append(piece)
        elif structures:
            structures[-1] = f"{structures[-1]};{piece}"
        else:
            raise ValueError("structure_ids does not start with a structure ID")
    return tuple(structures)


def read_inner_fold_predictions(evidence: FitEvidence, run_dir: Path) -> FitPredictions:
    """The fit's ``ec_val_protein_predictions.csv`` (already matched to its receipt) as inner-fold predictions."""
    unit_id = evidence.unit_id
    with (Path(run_dir) / PROTEIN_PREDICTIONS_NAME).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require(tuple(reader.fieldnames or ()) == PROTEIN_PREDICTION_COLUMNS,
                 f"{unit_id}: {PROTEIN_PREDICTIONS_NAME} has unexpected columns.")
        rows = list(reader)
    _require(rows and len(rows) == (evidence.receipt.get("protein_predictions") or {}).get("n_rows"),
             f"{unit_id}: {PROTEIN_PREDICTIONS_NAME} row count differs from its receipt.")
    expected = {"run_id": unit_id, "mode": "validation", "fold_index": str(evidence.inner_fold),
                "n_folds": str(N_INNER_FOLDS), "seed": str(evidence.seed)}
    ids, logits, true_class, ec_numbers = [], [], {}, {}
    for row in rows:
        _require(all(row[key] == value for key, value in expected.items()),
                 f"{unit_id}: a protein row names another run, fold or seed.")
        accession = row["accession"].strip()
        _require(bool(accession) and accession not in true_class, f"{unit_id}: protein rows repeat or lack an ID.")
        try:
            label = int(row["y"])
            values = [float(row[column]) for column in LOGIT_COLUMNS]
            structures = split_structure_ids(row["structure_ids"])
            numbers = sorted({number for sid in structures for number in ec_numbers_from_structure_id(sid)})
            owners = {accession_from_structure_id(sid) for sid in structures}
        except ValueError as exc:
            raise SelectionInputError(f"{unit_id}: a protein row does not parse ({exc.__class__.__name__}).") from exc
        _require(0 <= label < N_EC_CLASSES and all(math.isfinite(value) for value in values) and numbers,
                 f"{unit_id}: a protein row holds an invalid class, logit or EC number.")
        _require(owners == {accession}, f"{unit_id}: a protein row lists structures of another accession.")
        ids.append(accession)
        logits.append(values)
        true_class[accession] = label
        ec_numbers[accession] = tuple(numbers)
    priors = training_priors(evidence.receipt.get("training_protein_class_counts"), unit_id)
    part = InnerFoldPredictions(evidence.seed, evidence.inner_fold, tuple(ids), np.array(logits, dtype=np.float64),
                                priors)
    return FitPredictions(part, true_class, ec_numbers)


@dataclass(frozen=True)
class InnerTables:
    """Inner identity components (bootstrap units) and, from the certified membership, each protein's inner fold.

    ``membership_sha256`` is the certified membership file's SHA-256; every pooled fit's identity must carry it
    (a fold view holds a byte copy of that file).
    """

    component_of: Mapping[str, str] = field(repr=False)
    inner_fold_of: Optional[Mapping[str, int]] = field(default=None, repr=False)
    record: Mapping[str, Any] = field(default_factory=dict)
    membership_sha256: Optional[str] = None


@dataclass(frozen=True, eq=False)
class PooledConfiguration:
    """A pooled configuration with its fits' evidence and the proteins' full EC numbers."""

    pooled: PooledOof
    fits: tuple[FitEvidence, ...]
    ec_numbers: Mapping[str, tuple[str, ...]] = field(repr=False)

    def record(self) -> dict[str, Any]:
        return {"name": self.pooled.name, "predictions_sha256": predictions_sha256(self.pooled),
                "n_proteins": len(self.pooled.protein_ids), "fits": [fit.record() for fit in self.fits]}


def pool_fits(name: str, fits: Mapping[str, FitRecord], *, tables: Optional[InnerTables] = None,
              verifier: Optional[FitVerifier] = None) -> PooledConfiguration:
    """Verify every fit, read its protein rows and pool them with ``selection.pool_inner_oof``.

    The eligible proteins are the predicted ones; their true classes must agree across fits and, with a
    certified membership, they must be exactly its proteins in its inner folds. Messages give counts only.
    """
    verifier = verifier or FitVerifier()
    if tables is not None and tables.membership_sha256 is not None:
        other = sorted(uid for uid, record in fits.items()
                       if record.identity.get("membership_sha256") != tables.membership_sha256)
        _require(not other, f"{name}: {len(other)} fits were trained on another inner membership than the "
                            f"certified one of --components (e.g. {other[:2]}).")
    evidence, parts, true_class, ec_numbers = [], [], {}, {}
    for record in fits.values():
        item = verify_fit_evidence(record, verifier=verifier)
        predictions = read_inner_fold_predictions(item, record.run_dir)
        evidence.append(item)
        parts.append(predictions.part)
        for accession, label in predictions.true_class.items():
            _require(true_class.setdefault(accession, label) == label,
                     f"{name}: true classes of the same protein disagree between fits.")
            _require(ec_numbers.setdefault(accession, predictions.ec_numbers[accession])
                     == predictions.ec_numbers[accession], f"{name}: EC numbers of a protein disagree between fits.")
    if tables is not None and tables.inner_fold_of is not None:
        membership = tables.inner_fold_of
        _require(set(true_class) == set(membership),
                 f"{name}: {len(set(membership) - set(true_class))} membership proteins lack predictions and "
                 f"{len(set(true_class) - set(membership))} predicted proteins are not in the membership.")
        misplaced = sum(1 for part in parts for pid in part.protein_ids if membership[pid] != part.inner_fold)
        _require(misplaced == 0,
                 f"{name}: {misplaced} protein rows come from another inner fold than the membership's.")
    by_seed: dict[int, list[str]] = defaultdict(list)
    for part in parts:
        by_seed[part.seed].extend(part.protein_ids)
    for seed, ids in sorted(by_seed.items()):
        _require(len(ids) == len(set(ids)) == len(true_class),
                 f"{name}: seed {seed} predicts {len(set(ids))} distinct proteins in {len(ids)} rows; "
                 f"{len(true_class)} are eligible.")
    try:
        pooled = pool_inner_oof(name, parts, true_class)
    except ValueError as exc:
        raise SelectionInputError(f"{name}: the inner-fold predictions do not pool (seeds, inner folds or priors; "
                                  "protein details withheld).") from exc
    return PooledConfiguration(pooled, tuple(evidence), ec_numbers)


# ---------------------------------------------------------------------------
# Inner tables from the data overlay


def check_components_path(fold: int, path: Path, *, smoke: bool) -> Path:
    """The overlay identity folder of ``--components`` (a folder or its ``components_inner.tsv``); refuses
    anything under ``<campaign root>/identity`` (``simulation.check_calibration_folder``)."""
    from ec_clean30.simulation import SimulationError, check_calibration_folder, is_campaign_identity_path

    path = Path(path)
    refuse_test_paths([path], what="--components")
    _require(not is_campaign_identity_path(path),
             f"--components {path} lies under the campaign's identity folder, which holds test-side content after the "
             "identity audit; read the data overlay's copy (<overlay>/identity/fold<K>) instead.")
    folder = path.parent if path.name == COMPONENTS_FILE_NAME else path
    try:
        check_calibration_folder(fold, folder, require_overlay=not smoke)
    except SimulationError as exc:
        raise SelectionInputError(str(exc).replace("--calibrate-from", "--components")) from exc
    return folder


def load_inner_tables(fold: int, path: Path, *, smoke: bool = False) -> InnerTables:
    """Inner components and the certified membership's inner folds, through the identity loaders."""
    from ec_clean30 import identity

    folder = check_components_path(fold, path, smoke=smoke)
    try:
        chain = identity.load_inner_chain(folder, fold)
        components, components_receipt_sha256 = identity._load_inner_components(folder, fold, chain)
        membership, certificate_sha256 = identity._load_certified_membership(folder, fold, chain)
    except (ValueError, KeyError, OSError) as exc:
        raise SelectionInputError(f"--components {fold}: the identity copies do not verify ({exc}).") from exc
    inner_fold_of: dict[str, int] = {}
    for row in membership.rows:
        _require(inner_fold_of.setdefault(row["accession"], int(row["fold"])) == int(row["fold"]),
                 "The inner membership places one protein in two inner folds.")
    record = {"folder": str(folder), "components_receipt_sha256": components_receipt_sha256,
              "membership_sha256": membership.file_sha256, "certificate_sha256": certificate_sha256,
              "n_components": len(set(components.values())), "n_membership_proteins": len(inner_fold_of)}
    return InnerTables(components, inner_fold_of, record, membership.file_sha256)


# ---------------------------------------------------------------------------
# Step chain helpers


def _fold_records(records: Sequence[Mapping[str, Any]], fold: int) -> dict[str, tuple[Mapping[str, Any], str]]:
    """This fold's step records by step, each with its record SHA-256 (one record per fold and step)."""
    found: dict[str, tuple[Mapping[str, Any], str]] = {}
    for record in records:
        payload = record.get("payload") or {}
        if payload.get("format") != STEP_RECORD_FORMAT or payload.get("fold") != fold:
            continue
        _require(record["step"] not in found, f"The step chain holds two {record['step']} records of fold {fold}.")
        found[record["step"]] = (payload, record_sha256(record))
    return found


def _check_chain_inputs(payload: Mapping[str, Any], *, plan_sha256: str, freeze1_sha256: str) -> None:
    _require(payload.get("plan_sha256") == plan_sha256 and payload.get("freeze1_sha256") == freeze1_sha256,
             f"The {payload.get('step')} record was built from another plan or Freeze 1.")


def _selected(step: str, config: str, recipe: Recipe, units: Sequence[str], tau: float,
              pooled: PooledOof) -> dict[str, Any]:
    return {"step": step, "config": config, "recipe": recipe.to_dict(), "unit_ids": list(units), "tau": float(tau),
            "predictions_sha256": predictions_sha256(pooled)}


def pool_selected(descriptor: Mapping[str, Any], plan: Mapping[str, Any], records: Mapping[str, FitRecord], *,
                  tables: Optional[InnerTables] = None, freeze1_sha256: Optional[str] = None, smoke: bool = False,
                  verifier: Optional[FitVerifier] = None) -> PooledConfiguration:
    """Re-pool a selected configuration named by a step record and require the recorded predictions hash."""
    units = _plan_unit_map(plan)
    unit_ids = list(descriptor.get("unit_ids") or [])
    _require(len(unit_ids) == N_INNER_FOLDS * len(SCREENING_SEEDS) and all(uid in units for uid in unit_ids),
             f"The selected configuration {descriptor.get('config')!r} does not name a grid of plan units.")
    recipe = recipe_from_record(descriptor.get("recipe") or {})
    fits = collect_fits(records, [(units[uid], recipe) for uid in unit_ids], owns=lambda record: False,
                        what=f"selected {descriptor.get('config')}", freeze1_sha256=freeze1_sha256, smoke=smoke)
    configuration = pool_fits(str(descriptor.get("config")), fits, tables=tables, verifier=verifier)
    tau = float(descriptor.get("tau") or 0.0)
    if tau != 0.0:
        configuration = replace(configuration, pooled=apply_tau(configuration.pooled, tau))
    _require(predictions_sha256(configuration.pooled) == descriptor.get("predictions_sha256"),
             f"The re-pooled {descriptor.get('config')} predictions differ from the step record's hash.")
    return configuration


# ---------------------------------------------------------------------------
# select


def _component_check(tables: InnerTables, pooled: PooledOof) -> None:
    missing = sum(1 for pid in pooled.protein_ids if pid not in tables.component_of)
    _require(missing == 0, f"{missing} pooled proteins have no inner identity component in --components.")


def _training_counts(pooled: PooledOof) -> np.ndarray:
    return np.bincount(pooled.true_idx, minlength=N_EC_CLASSES)


def _s1(plan, records, tables, *, freeze1_sha256, smoke, base_config, verifier) -> dict[str, Any]:
    s1_units = [unit for unit in plan_units(plan) if unit.step == STEP_S1]
    required = [(unit, Recipe(unit.family, tuple(unit.learning_rate), unit.epochs, tuple(unit.lever_configs)))
                for unit in s1_units]
    fits = collect_fits(records, required, owns=lambda record: record.step == STEP_S1, what="S1 fits",
                        freeze1_sha256=freeze1_sha256, smoke=smoke)
    families = list(dict.fromkeys(unit.family for unit in s1_units))
    config = promotion_config_for(None, base_config)
    choices, winners, configurations, recipes = {}, {}, {}, {}
    for family in families:
        settings = {}
        for unit in s1_units:
            if unit.family != family:
                continue
            label = learning_rate_label(unit.learning_rate)
            name = unit.unit_id.split("__")[3]
            settings.setdefault(label, (name, unit.learning_rate, []))[2].append(unit.unit_id)
        pooled = {}
        for label, (name, setting, unit_ids) in settings.items():
            configurations[name] = pool_fits(name, {uid: fits[uid] for uid in unit_ids}, tables=tables,
                                             verifier=verifier)
            _component_check(tables, configurations[name].pooled)
            pooled[label] = configurations[name].pooled
            recipes[(family, label)] = (name, Recipe(family, tuple(setting)), unit_ids)
        try:
            choices[family] = s1_choose_learning_rate(family, pooled)
        except ValueError as exc:
            raise SelectionInputError(f"S1 {family}: {exc}") from exc
        winners[family] = choices[family].winner
    reference = next(iter(winners.values()))
    try:
        escalation = s1_family_escalation(winners, component_of=tables.component_of,
                                          training_class_counts=_training_counts(reference), config=config)
    except ValueError as exc:
        raise SelectionInputError(f"S1 family escalation: {exc}") from exc
    winner_family = escalation.winner_family
    name, recipe, unit_ids = recipes[(winner_family, choices[winner_family].chosen)]
    return {
        "learning_rate_choices": {family: choice.to_record() for family, choice in choices.items()},
        "family_escalation": escalation.to_record(),
        "promotion_config": config.to_record(),
        "configurations": {key: value.record() for key, value in configurations.items()},
        "s1_winner": recipe.to_dict(),
        "selected": _selected(STEP_S1, name, recipe, unit_ids, 0.0, escalation.winner),
    }


def _lever_configuration(plan, records, tables, config_id, s1_recipe, *, freeze1_sha256, smoke, verifier):
    """One S2 configuration pooled; S3 records (whose recipes repeat the lever) are not part of this group."""
    grid = set(_grid_ids(int(plan["fold"]), STEP_S2, config_id))
    candidates = {uid: record for uid, record in records.items() if record.step == STEP_S2 or uid in grid}
    fits = collect_lever_fits((), plan, s1_winner=s1_recipe, config_id=config_id, freeze1_sha256=freeze1_sha256,
                              smoke=smoke, records=candidates)
    configuration = pool_fits(config_id, fits, tables=tables, verifier=verifier)
    _component_check(tables, configuration.pooled)
    return configuration


def _s2_config_ids(family: str) -> list[str]:
    return [config_id for config_id, config in LEVER_CONFIGS.items() if lever_applies(config.lever, family)]



def _bound_cost_file(parent: Path, reference: Any, what: str) -> Path:
    """A local evidence file referenced by name and SHA-256; never follow a file link."""
    _require(isinstance(reference, dict) and isinstance(reference.get("path"), str)
             and bool(reference["path"]), f"{what}: a path and SHA-256 are required.")
    path = Path(reference["path"])
    _require(not path.is_absolute(), f"{what}: evidence paths must be relative to their record.")
    path = parent / path
    refuse_test_paths([path], what=what)
    _require(path.is_file() and not path.is_symlink(), f"{what}: evidence is missing or a link.")
    _require(sha256_file(path) == _check_sha256(reference.get("sha256"), what),
             f"{what}: evidence SHA-256 disagrees.")
    return path


def _hours(value: Any, what: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool)
             and math.isfinite(value) and value >= 0, f"{what} must be finite nonnegative hours.")
    return float(value)


def verify_l3_cost_disposition(plan, records, s1_recipe, *, freeze1c_path: Path, disposition_path: Path,
                               freeze1_sha256: str, smoke: bool = False,
                               verifier: Optional[FitVerifier] = None) -> dict[str, Any]:
    """Validate a cost close-out, without confusing a frozen ceiling with a runtime decision.

    Freeze 1c supplies ``campaign_id``, ``freeze1_sha256`` and ``l3_vm_hours_per_fold`` (fold keys as
    strings). The disposition supplies the same campaign/Freeze-1 identity, fold, plan and Freeze-1c hashes,
    ``disposition: not_tested_cost``, ``reason`` (ceiling_reached or projected_completion_exceeds_ceiling),
    and relative ``{path, sha256}`` references to ``accounting`` and ``authority`` JSON records.

    Both referenced records repeat those five identity bindings. Accounting holds disjoint charged VM wall-time
    intervals ``[start_unix_seconds, end_unix_seconds]``, ``used_vm_hours``, ``remaining_vm_hours_estimate``, a
    hash-bound ``forecast_evidence`` file, and ``attempted_fits`` (unit_id, status, runner_sha256). VM intervals
    count allocation time once, not the sum of concurrent fit durations. Authority names the accounting SHA-256,
    ``decision: stop_and_close_out`` and a nonempty ``decision_reference`` to the maintainer's recorded decision.
    No ordinary failed/running fit may masquerade as a cost stop. Eight completed fits are always screened.
    """
    freeze1c_path, disposition_path = Path(freeze1c_path), Path(disposition_path)
    refuse_test_paths([freeze1c_path, disposition_path], what="L3 cost inputs")
    frozen = _read_json(freeze1c_path, "Freeze-1c")
    disposition = _read_json(disposition_path, "L3 cost disposition")
    _require(isinstance(frozen, dict) and isinstance(disposition, dict), "L3 cost inputs must be JSON objects.")
    _require(frozen.get("campaign_id") == CAMPAIGN_ID and frozen.get("freeze1_sha256") == freeze1_sha256,
             "Freeze-1c names another campaign or Freeze-1.")
    limits = frozen.get("l3_vm_hours_per_fold")
    _require(isinstance(limits, dict), "Freeze-1c holds no L3 per-fold ceilings.")
    ceiling = _hours(limits.get(str(plan["fold"])), "Freeze-1c L3 ceiling")
    _require(ceiling > 0, "Freeze-1c L3 ceiling must be positive.")
    bindings = {"campaign_id": CAMPAIGN_ID, "fold": plan["fold"], "plan_sha256": plan["plan_sha256"],
                "freeze1_sha256": freeze1_sha256, "freeze1c_sha256": sha256_file(freeze1c_path)}
    _require(disposition.get("disposition") == "not_tested_cost", "L3 disposition is not not_tested_cost.")
    bound = {}
    for key in ("accounting", "authority"):
        path = _bound_cost_file(disposition_path.parent, disposition.get(key), f"L3 {key}")
        record = _read_json(path, f"L3 {key}")
        _require(isinstance(record, dict), f"L3 {key} must be a JSON object.")
        bound[key] = (path, record)
    for what, record in (("disposition", disposition), *((key, item[1]) for key, item in bound.items())):
        _require(all(record.get(key) == value and type(record.get(key)) is type(value)
                     for key, value in bindings.items()), f"L3 {what}: campaign/fold/plan/Freeze hashes disagree.")
    accounting_path, accounting = bound["accounting"]
    authority = bound["authority"][1]
    _require(authority.get("accounting_sha256") == sha256_file(accounting_path)
             and authority.get("decision") == "stop_and_close_out"
             and isinstance(authority.get("decision_reference"), str)
             and bool(authority["decision_reference"].strip()),
             "L3 cost close-out needs a recorded stop decision bound to this accounting.")
    intervals = accounting.get("vm_intervals")
    _require(isinstance(intervals, list), "L3 accounting needs charged VM wall-time intervals.")
    measured_seconds, previous_end = 0.0, None
    for interval in intervals:
        _require(isinstance(interval, list) and len(interval) == 2, "L3 VM interval must have start and end.")
        start, end = (_hours(value, "L3 interval timestamp") for value in interval)
        _require(end > start and (previous_end is None or start >= previous_end),
                 "L3 VM intervals must be positive, ordered and non-overlapping.")
        measured_seconds += end - start
        previous_end = end
    used = _hours(accounting.get("used_vm_hours"), "L3 used_vm_hours")
    remaining = _hours(accounting.get("remaining_vm_hours_estimate"), "L3 remaining_vm_hours_estimate")
    _require(math.isclose(used, measured_seconds / 3600, rel_tol=1e-9, abs_tol=1e-9),
             "L3 used_vm_hours disagrees with charged VM intervals.")
    forecast = _bound_cost_file(accounting_path.parent, accounting.get("forecast_evidence"), "L3 forecast")
    reason = disposition.get("reason")
    _require((reason == "ceiling_reached" and used >= ceiling)
             or (reason == "projected_completion_exceeds_ceiling" and used + remaining > ceiling),
             "L3 cost reason is not supported by the measured use, forecast and frozen ceiling.")
    attempts = accounting.get("attempted_fits")
    _require(isinstance(attempts, list) and all(isinstance(item, dict) for item in attempts),
             "L3 accounting needs an attempted_fits list.")
    units = {unit.unit_id: unit for unit in required_lever_units(plan, "L3")}
    owned = {uid: record for uid, record in records.items()
             if record.step == STEP_S2 and (uid in units or "L3" in record.lever_configs)}
    ids = [item.get("unit_id") for item in attempts]
    _require(all(isinstance(uid, str) for uid in ids) and len(ids) == len(set(ids))
             and set(ids) == set(owned) and set(ids) <= set(units),
             "L3 accounting must name exactly the attempted fits of the frozen grid, with no duplicates.")
    recipe = recipe_with_lever(s1_recipe, "L3")
    completed, evidence = 0, []
    verifier = verifier or FitVerifier()
    for item in attempts:
        uid = item["unit_id"]
        record = owned[uid]
        problems = _identity_problems(record, units[uid], recipe,
                                      freeze1_sha256=None if smoke else freeze1_sha256, smoke=smoke)
        _require(not problems, f"L3 cost fit {uid} has a foreign identity: {problems}.")
        status = item.get("status")
        _require(item.get("runner_sha256") == record.runner_sha256
                 and status in ("completed", "stopped_cost") and record.runner.get("status") == status,
                 f"L3 cost fit {uid}: unresolved/failed status or runner hash mismatch.")
        if status == "completed":
            evidence.append(verify_fit_evidence(record, verifier=verifier).record())
            completed += 1
        else:
            _require(record.runner.get("stop_reason") == "cost" and record.runner.get("returncode") is not None,
                     f"L3 cost fit {uid}: no terminal cost-stop evidence.")
            evidence.append(dict(item))
    _require(not attempts or used > 0, "Attempted L3 fits require nonzero measured VM use.")
    return {**bindings,
            "disposition_sha256": sha256_file(disposition_path), "accounting_sha256": sha256_file(accounting_path),
            "authority_sha256": sha256_file(bound["authority"][0]), "forecast_sha256": sha256_file(forecast),
            "ceiling_vm_hours": ceiling, "used_vm_hours": used, "remaining_vm_hours_estimate": remaining,
            "reason": reason, "attempted_fits": evidence, "skip": completed < len(units)}

def _s2(plan, records, tables, chain, *, freeze1_sha256, smoke, base_config, verifier,
        cost_disposition=None) -> dict[str, Any]:
    fold = int(plan["fold"])
    s1_payload, s1_sha256 = chain[STEP_S1]
    _check_chain_inputs(s1_payload, plan_sha256=plan["plan_sha256"], freeze1_sha256=freeze1_sha256)
    s1_recipe = recipe_from_record(s1_payload["s1_winner"])
    family = s1_recipe.family
    s1 = pool_selected(s1_payload["selected"], plan, records, tables=tables, freeze1_sha256=freeze1_sha256,
                       smoke=smoke, verifier=verifier)
    _component_check(tables, s1.pooled)
    expected_s2 = {uid for config_id in _s2_config_ids(family) for uid in _grid_ids(fold, STEP_S2, config_id)}
    stray = sorted(uid for uid, record in records.items() if record.step == STEP_S2 and uid not in expected_s2)
    _require(not stray, f"S2: foreign fits not in the applicable levers' grid for {family}: {stray[:8]}.")
    kwargs = {"component_of": tables.component_of, "training_class_counts": _training_counts(s1.pooled)}
    outcomes, configurations, certification = [], {}, None
    l3_ids = set(_grid_ids(fold, STEP_S2, "L3"))
    for lever in LEVERS:
        if not lever_applies(lever, family):
            outcomes.append(lever_not_applicable(lever, family))
            continue
        config = promotion_config_for(lever, base_config)
        try:
            if lever == LOGIT_ADJUSTMENT_LEVER:
                outcomes.append(screen_logit_adjustment(family, s1.pooled, config=config, **kwargs))
            elif lever == EPOCH_LEVER:
                looks = {}
                for epochs in EPOCH_LOOKS:
                    config_id = f"L7e{epochs}"
                    configurations[config_id] = _lever_configuration(plan, records, tables, config_id, s1_recipe,
                                                                     freeze1_sha256=freeze1_sha256, smoke=smoke,
                                                                     verifier=verifier)
                    looks[epochs] = configurations[config_id].pooled
                outcomes.append(screen_epoch_lever(family, s1.pooled, looks, config=config, **kwargs))
            elif lever == SUPCON_LEVER:
                certification = certify_l5(plan, (), s1_winner=s1_recipe, freeze1_sha256=freeze1_sha256, smoke=smoke,
                                           records=records, verifier=verifier)
                if certification.fraction is None:
                    outcome = lever_not_tested(lever, family, certification.reason)
                else:
                    configurations[lever] = _lever_configuration(plan, records, tables, lever, s1_recipe,
                                                                 freeze1_sha256=freeze1_sha256, smoke=smoke,
                                                                 verifier=verifier)
                    outcome = screen_lever(lever, family, s1.pooled, configurations[lever].pooled, config=config,
                                           positive_step_fraction=certification.fraction, **kwargs)
                outcomes.append(replace(outcome, details={**outcome.details,
                                                          "certification": certification.record()}))
            else:
                if lever == "L3" and cost_disposition is not None and cost_disposition["skip"]:
                    outcomes.append(lever_not_tested(lever, family, "not tested (cost): verified cost disposition"))
                    continue
                if lever == "L3" and not any(uid in records for uid in l3_ids):
                    raise SelectionInputError(
                        "L3 has no fits: a verified Freeze-1c and cost disposition are required for a cost skip.")
                configurations[lever] = _lever_configuration(plan, records, tables, lever, s1_recipe,
                                                             freeze1_sha256=freeze1_sha256, smoke=smoke,
                                                             verifier=verifier)
                outcomes.append(screen_lever(lever, family, s1.pooled, configurations[lever].pooled, config=config,
                                             **kwargs))
        except SelectionInputError:
            raise
        except ValueError as exc:
            raise SelectionInputError(f"S2 {lever}: {exc}") from exc
    try:
        screen = s2_lever_screen(family, s1.pooled, outcomes)
    except ValueError as exc:
        raise SelectionInputError(f"S2 lever screen: {exc}") from exc
    return {
        "s1_record_sha256": s1_sha256,
        "s1_winner": s1_recipe.to_dict(),
        "s1_winner_predictions_sha256": predictions_sha256(s1.pooled),
        "screen": screen.to_record(),
        "promotion_alpha_by_lever": {lever: promotion_config_for(lever, base_config).alpha for lever in LEVERS},
        "l3_cost_disposition": cost_disposition,
        "l5_certification": None if certification is None else certification.record(),
        "configurations": {key: value.record() for key, value in configurations.items()},
    }


_FIT_BINDING_KEYS: tuple[str, ...] = tuple(FitEvidence("", 0, 0, {}, {}).record())


def _fit_bindings(recorded: Any) -> Optional[list[dict[str, Any]]]:
    """The ``FitEvidence.record()`` fields of each recorded certification fit (None: no list of fits)."""
    if not isinstance(recorded, list) or not all(isinstance(item, dict) for item in recorded):
        return None
    return [{key: item.get(key) for key in _FIT_BINDING_KEYS} for item in recorded]


def _s3(plan, records, tables, chain, *, freeze1_sha256, smoke, base_config, verifier) -> dict[str, Any]:
    fold = int(plan["fold"])
    _require(STEP_S2 in chain, f"S3 needs the S2 record of fold {fold}.")
    s1_payload, _ = chain[STEP_S1]
    s2_payload, s2_sha256 = chain[STEP_S2]
    for payload in (s1_payload, s2_payload):
        _check_chain_inputs(payload, plan_sha256=plan["plan_sha256"], freeze1_sha256=freeze1_sha256)
    s1_recipe = recipe_from_record(s1_payload["s1_winner"])
    _require(_recipe_key(s2_payload["s1_winner"]) == _recipe_key(s1_recipe), "The S2 record names another S1 winner.")
    family = s1_recipe.family
    s1 = pool_selected(s1_payload["selected"], plan, records, tables=tables, freeze1_sha256=freeze1_sha256,
                       smoke=smoke, verifier=verifier)
    _component_check(tables, s1.pooled)
    _require(predictions_sha256(s1.pooled) == s2_payload["s1_winner_predictions_sha256"],
             "The S1 winner's predictions differ from the ones the S2 record screened against.")
    outcomes, sources, tau = [], {}, 0.0
    for item in s2_payload["screen"]["outcomes"]:
        lever, status = item["lever"], item["status"]
        if status != LEVER_PASSED:
            outcomes.append(LeverOutcome(lever, status, reason=item.get("reason", "")))
            continue
        details = item.get("details") or {}
        if lever == SUPCON_LEVER:
            # S3 fits may repeat L5 in their recipe; the certification covers the eight S2 fits only.
            s2_records = {uid: record for uid, record in records.items() if record.step != STEP_S3}
            certification = certify_l5(plan, (), s1_winner=s1_recipe, freeze1_sha256=freeze1_sha256, smoke=smoke,
                                       records=s2_records, verifier=verifier)
            recorded = details.get("positive_step_fraction")
            _require(certification.testable and isinstance(recorded, (int, float))
                     and abs(certification.fraction - float(recorded)) <= FLOAT_TOLERANCE,
                     "S3: L5 passed in S2 but its eight fits no longer certify at the recorded fraction "
                     f"({certification.reason}).")
            certified = (s2_payload.get("l5_certification") or {}).get("fits")
            _require(_fit_bindings(certified) == [item.fit.record() for item in certification.fits],
                     "S3: L5's fits differ from the ones the S2 record certified (unit, checkpoint, run "
                     "configuration, receipt, predictions or runner SHA-256).")
        if lever == LOGIT_ADJUSTMENT_LEVER:
            tau = float(details["chosen_tau"])
            challenger, recipe = apply_tau(s1.pooled, tau), s1_recipe
            config_id, units = s1_payload["selected"]["config"], s1_payload["selected"]["unit_ids"]
        else:
            config_id = f"L7e{int(details['challenger_epochs'])}" if lever == EPOCH_LEVER else lever
            configuration = _lever_configuration(plan, records, tables, config_id, s1_recipe,
                                                 freeze1_sha256=freeze1_sha256, smoke=smoke, verifier=verifier)
            challenger, units = configuration.pooled, _grid_ids(fold, STEP_S2, config_id)
            recipe = recipe_with_lever(s1_recipe, config_id)
        _require(predictions_sha256(challenger) == item["decision"]["inputs"]["challenger_predictions_sha256"],
                 f"S3: lever {lever}'s re-pooled predictions differ from the S2 record.")
        sources[lever] = (config_id, recipe, units, tau if lever == LOGIT_ADJUSTMENT_LEVER else 0.0, challenger)
        outcomes.append(LeverOutcome(lever, LEVER_PASSED, challenger=challenger))
    screen = LeverScreen(family, predictions_sha256(s1.pooled), tuple(outcomes))
    _require(screen.s1_winner_sha256 == s2_payload["screen"]["s1_winner_predictions_sha256"],
             "The S2 screen was built against another S1 winner.")
    passing = [outcome.lever for outcome in screen.passing]
    s3_ids = _grid_ids(fold, STEP_S3, "combination")
    combination, combined_recipe = None, None
    combination_units, combination_source, combined_coverage = [], None, None
    if len(passing) >= 2:
        fit_configs = [sources[lever][0] for lever in passing if lever != LOGIT_ADJUSTMENT_LEVER]
        epochs = next((LEVER_CONFIGS[config_id].epochs for config_id in fit_configs
                       if LEVER_CONFIGS[config_id].epochs is not None), s1_recipe.epochs)
        try:
            combined_recipe = Recipe(family, s1_recipe.learning_rate, epochs, tuple(fit_configs))
        except CampaignPlanError as exc:
            raise SelectionInputError(f"S3: the combined recipe is not valid: {exc}") from exc
        units = _plan_unit_map(plan)
        reuse_s2 = len(fit_configs) == 1 and LOGIT_ADJUSTMENT_LEVER in passing
        combination_units = sources[next(lever for lever in passing if lever != LOGIT_ADJUSTMENT_LEVER)][2] \
            if reuse_s2 else s3_ids
        if reuse_s2:
            _require(not any(record.step == STEP_S3 for record in records.values()),
                     "S3: L9 plus one fitted lever reuses S2; extra S3 fits are refused.")
        fits = collect_fits(records, [(units[uid], combined_recipe) for uid in combination_units],
                            owns=(lambda record: False) if reuse_s2 else (lambda record: record.step == STEP_S3),
                            what="S3 combination source", freeze1_sha256=freeze1_sha256, smoke=smoke)
        configuration = pool_fits("combination", fits, tables=tables, verifier=verifier)
        combination_source = {"step": STEP_S2 if reuse_s2 else STEP_S3, **configuration.record()}
        combination = configuration.pooled if tau == 0.0 else apply_tau(configuration.pooled, tau)
        if SUPCON_LEVER in passing:
            coverage = [verify_supcon_evidence(fit, plan, verifier=verifier) for fit in fits.values()]
            fraction, aggregation = supcon_positive_step_fraction({item.fit.unit_id: item.coverage for item in coverage})
            combined_coverage = {"role": "descriptive_only", "positive_step_fraction": fraction,
                                 "fits": [item.record() for item in coverage], "aggregation": aggregation}
        _component_check(tables, combination)
    else:
        stray = sorted(uid for uid, record in records.items() if record.step == STEP_S3)
        _require(not stray,
                 f"S3: {len(passing)} lever(s) passed, so no combination is fitted; foreign fits {stray[:8]}.")
    config = promotion_config_for(None, base_config)
    try:
        outcome = s3_combination(s1.pooled, screen, combination, component_of=tables.component_of,
                                 training_class_counts=_training_counts(s1.pooled), config=config)
    except ValueError as exc:
        raise SelectionInputError(f"S3: {exc}") from exc
    if outcome.selected == "s1_winner":
        selected = _selected(STEP_S3, s1_payload["selected"]["config"], s1_recipe, s1_payload["selected"]["unit_ids"],
                             0.0, s1.pooled)
    elif outcome.selected == "combination":
        selected = _selected(STEP_S3, "combination", combined_recipe, combination_units, tau, combination)
    else:
        config_id, recipe, units, lever_tau, challenger = sources[outcome.selected]
        selected = _selected(STEP_S3, config_id, recipe, units, lever_tau, challenger)
    return {"s2_record_sha256": s2_sha256, "s1_winner": s1_recipe.to_dict(), "passing_levers": passing,
            "combination": outcome.to_record(), "combined_recipe": None if combined_recipe is None
            else combined_recipe.to_dict(), "combination_source": combination_source,
            "combination_supcon_coverage": combined_coverage,
            "promotion_config": config.to_record(), "selected": selected}


def run_select(
    *,
    fold: int,
    step: str,
    plan_path: Path,
    runs_dirs: Sequence[Path],
    tables: InnerTables,
    freeze1_sha256: str,
    out: Path,
    smoke: bool = False,
    provenance: Optional[Mapping[str, Any]] = None,
    base_config: Optional[PromotionConfig] = None,
    freeze1c_path: Optional[Path] = None,
    l3_cost_disposition_path: Optional[Path] = None,
) -> dict[str, Any]:
    """One selection step of one outer fold; appends its record to ``out`` and returns it.

    ``base_config`` replaces the frozen promotion settings except alpha, which ``promotion_config_for`` always
    sets (engineering tests only; the runner never passes it).
    """
    _require(step in SELECT_STEPS, f"--step must be one of {SELECT_STEPS}, got {step!r}.")
    freeze1_sha256 = _check_sha256(freeze1_sha256, "--freeze1-sha256")
    _require((freeze1c_path is None) == (l3_cost_disposition_path is None),
             "--freeze1c and --l3-cost-disposition must be supplied together.")
    _require(freeze1c_path is None or step == STEP_S2, "L3 cost inputs are accepted only for S2.")
    out = Path(out)
    folder = tables.record.get("folder")
    _refuse_record_output(out, {fold: Path(folder)} if folder else {})
    plan = load_frozen_plan(plan_path, fold)
    chain_records = _read_chain(out, freeze1_sha256, "--out") if out.exists() else []
    chain = _fold_records(chain_records, fold)
    _require(step not in chain, f"The step chain already holds the {step} record of fold {fold}.")
    previous = {STEP_S1: None, STEP_S2: STEP_S1, STEP_S3: STEP_S2}[step]
    _require(previous is None or previous in chain, f"{step} needs the {previous} record of fold {fold} in --out.")
    _require(not any(later in chain for later in SELECT_STEPS[SELECT_STEPS.index(step) + 1:]),
             f"The step chain already holds a later step of fold {fold}.")
    for earlier in SELECT_STEPS[:SELECT_STEPS.index(step)]:
        _require(_components_binding(chain[earlier][0].get("components")) == _components_binding(tables.record),
                 f"--components differs from the inner tables the {earlier} record of fold {fold} used "
                 "(component, membership or certificate SHA-256, or their counts).")
    records = read_runner_records(runs_dirs, fold=fold)
    verifier = FitVerifier()
    options = {"freeze1_sha256": None if smoke else freeze1_sha256, "smoke": smoke, "base_config": base_config,
               "verifier": verifier}
    if step == STEP_S1:
        result = _s1(plan, records, tables, **options)
    elif step == STEP_S2:
        cost = None if freeze1c_path is None else verify_l3_cost_disposition(
            plan, records, recipe_from_record(chain[STEP_S1][0]["s1_winner"]),
            freeze1c_path=freeze1c_path, disposition_path=l3_cost_disposition_path,
            freeze1_sha256=freeze1_sha256, smoke=smoke, verifier=verifier)
        result = _s2(plan, records, tables, chain, cost_disposition=cost, **options)
    else:
        result = _s3(plan, records, tables, chain, **options)
    payload = {
        "format": STEP_RECORD_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "fold": fold,
        "step": step,
        "plan_sha256": plan["plan_sha256"],
        "freeze1_sha256": freeze1_sha256,
        "smoke": bool(smoke),
        "components": dict(tables.record),
        "n_runner_records": len(records),
        **result,
        "provenance": dict(provenance) if provenance is not None else provenance_record(smoke=smoke),
    }
    try:
        return append_step_record(out, step, payload, anchor_sha256=freeze1_sha256)
    except (ValueError, OSError) as exc:
        raise SelectionInputError(f"--out {out.name}: the step chain cannot be appended ({exc}).") from exc


def _components_binding(record: Any) -> dict[str, Any]:
    """An inner-tables record without its folder, as JSON values (a copy elsewhere with equal hashes agrees)."""
    record = dict(record) if isinstance(record, Mapping) else {}
    return json.loads(json.dumps({key: value for key, value in record.items() if key != "folder"},
                                 sort_keys=True, default=str))


def _read_chain(path: Path, freeze1_sha256: str, what: str) -> list[dict[str, Any]]:
    """``selection.read_step_chain``; a missing, unreadable, corrupt or unlinked chain refuses the action."""
    try:
        return read_step_chain(path, anchor_sha256=freeze1_sha256)
    except (ValueError, OSError) as exc:
        raise SelectionInputError(f"{what} {Path(path).name}: the step chain does not verify against "
                                  f"--freeze1-sha256 ({exc}).") from exc


def _refuse_record_output(path: Path, components_folders: Mapping[int, Path]) -> None:
    """Refuse a test-named ``--out``, one under ``<campaign root>/identity``, one inside a ``--components``
    folder and one inside a data overlay (``simulation.check_out_folder``): the records hold per-fold class
    numbers, and the overlay is checked file by file against its manifest."""
    from ec_clean30.simulation import SimulationError, check_out_folder, is_campaign_identity_path

    refuse_test_paths([path], what="--out")
    _require(not is_campaign_identity_path(path), f"--out {path} lies under the campaign's identity folder.")
    try:
        check_out_folder(path, components_folders)
    except SimulationError as exc:
        message = str(exc).replace("calibration folder", "--components folder")
        raise SelectionInputError(message.replace("simulation outputs", "step and gate records")) from exc


# ---------------------------------------------------------------------------
# Gap gate


def correctness_balanced_accuracy(true_idx: Sequence[int] | np.ndarray, correct: Sequence[bool] | np.ndarray,
                                  classes: Sequence[int] = PRIMARY_CLASS_INDICES) -> tuple[float, int]:
    """Mean per-class recall over the listed classes present among ``true_idx``, from per-protein correctness
    (so a CLEAN tie, never correct, needs no class); returns (points, classes averaged)."""
    true = np.asarray(true_idx, dtype=np.int64)
    hits = np.asarray(correct, dtype=bool)
    _require(true.shape == hits.shape and true.size > 0, "Balanced accuracy needs one correctness per protein.")
    recalls = [hits[true == index].mean() for index in classes if np.any(true == index)]
    return (POINTS * float(np.mean(recalls)) if recalls else float("nan")), len(recalls)


def gap_gate_decision(deepmzyme_points: float, clean_r1_points: float, *, after: str) -> dict[str, Any]:
    """The gate rule: gap = DeepMzyme - CLEAN-R1 (after S3 minus the optimism discount first); continue at
    a gap of -3.0 points or more, else the maintainer decides (stop or continue as frozen)."""
    _require(after in GAP_GATE_AFTER, f"--after must be one of {GAP_GATE_AFTER}, got {after!r}.")
    _require(math.isfinite(deepmzyme_points) and math.isfinite(clean_r1_points), "Gate scores must be finite.")
    discount = S3_OPTIMISM_DISCOUNT_POINTS if after == STEP_S3 else 0.0
    gap = float(deepmzyme_points) - discount - float(clean_r1_points)
    decision = DECISION_CONTINUE if gap >= GAP_CONTINUE_POINTS - FLOAT_TOLERANCE else DECISION_MAINTAINER
    return {"deepmzyme_points": float(deepmzyme_points), "optimism_discount_points": discount,
            "clean_r1_points": float(clean_r1_points), "gap_points": gap, "threshold_points": GAP_CONTINUE_POINTS,
            "decision": decision}


def _read_clean_arm(paths: Sequence[Path], arm: str) -> dict[int, Any]:
    from ec_clean30 import scoring

    comparator = scoring.comparator
    exports: dict[int, Any] = {}
    for path in paths:
        path = Path(path)
        refuse_test_paths([path], what=f"--clean-{arm.replace('_', '-')}")
        try:
            export = comparator.read_distance_export(path)
        except (comparator.ComparatorError, OSError, ValueError) as exc:
            raise SelectionInputError(f"CLEAN {arm} export {path.name} does not verify: {exc}") from exc
        receipt = export.receipt
        inner = receipt.get("inner_fold")
        _require(receipt.get("fold") == GAP_GATE_FOLD and receipt.get("regime") == arm
                 and receipt.get("mode") == "validation" and _is_int(inner) and 0 <= inner < N_INNER_FOLDS,
                 f"CLEAN export {path.name} is not a fold-{GAP_GATE_FOLD} {arm} validation export of an inner fold.")
        in_sample = receipt.get("n_in_sample_queries")
        _require(not (_is_int(in_sample) and in_sample > 0),
                 f"CLEAN export {path.name} scores {in_sample} queries that were in its own training table.")
        _require(inner not in exports, f"CLEAN {arm}: inner fold {inner} is given more than once (one seed each).")
        exports[inner] = export
    _require(sorted(exports) == list(range(N_INNER_FOLDS)),
             f"CLEAN {arm}: one export per inner fold 0-{N_INNER_FOLDS - 1} is needed, got {sorted(exports)}.")
    return exports


def score_gap_gate(configuration: PooledConfiguration, r1_exports: Mapping[int, Any], r2_exports: Mapping[int, Any],
                   *, after: str) -> dict[str, Any]:
    """Scores on the R1-inner EC closure (see the module docstring); returns counts and points only."""
    from ec_clean30 import scoring

    comparator = scoring.comparator
    pooled = configuration.pooled
    _require(pooled.inner_fold is not None, "The pooled predictions carry no inner folds.")
    for arm, exports in (("r1_inner", r1_exports), ("r2_inner", r2_exports)):
        _require(set(exports) == set(range(N_INNER_FOLDS)), f"CLEAN {arm} needs exactly four inner folds.")
        for inner, export in exports.items():
            expected = {pid for pid, fold in zip(pooled.protein_ids, pooled.inner_fold) if int(fold) == inner}
            queries = list(export.query_ids)
            _require(len(queries) == len(set(queries)) and set(queries) == expected,
                     f"CLEAN {arm} inner fold {inner}: query IDs must exactly equal the certified validation "
                     "membership before closure filtering (no extra, missing or duplicate queries).")
    closure = {inner: set(export.ec_labels) for inner, export in r1_exports.items()}
    keep = np.array([all(number in closure[int(fold)] for number in configuration.ec_numbers[pid])
                     for pid, fold in zip(pooled.protein_ids, pooled.inner_fold)], dtype=bool)
    _require(keep.any(), "No inner-validation protein has all its EC numbers among the R1-inner training centres.")
    ids = [pid for pid, kept in zip(pooled.protein_ids, keep) if kept]
    true = pooled.true_idx[keep]
    folds = pooled.inner_fold[keep]
    systems: dict[str, dict[str, Any]] = {}
    dm_correct = pooled.predictions[:, keep] == true[None, :]
    by_seed = [correctness_balanced_accuracy(true, row)[0] for row in dm_correct]
    seven_by_seed = [correctness_balanced_accuracy(true, row, range(N_EC_CLASSES))[0] for row in dm_correct]
    systems["deepmzyme"] = {"ec_classes_1_6_ba_points": float(np.mean(by_seed)), "ba_points_by_seed": by_seed,
                            "seven_class_ba_points": float(np.mean(seven_by_seed)),
                            "ec7_correct_by_seed": [int(row[true == 6].sum()) for row in dm_correct]}
    for arm, exports in (("clean_r1_inner", r1_exports), ("clean_r2_inner", r2_exports)):
        correct = np.zeros(len(ids), dtype=bool)
        ties = 0
        for inner, export in exports.items():
            projection = comparator.project_ec_classes(export.distances, export.query_ids, export.ec_labels)
            row_of = {query: position for position, query in enumerate(projection.query_ids)}
            members = [position for position, fold in enumerate(folds) if int(fold) == inner]
            missing = sum(1 for position in members if ids[position] not in row_of)
            _require(missing == 0, f"CLEAN {arm} inner fold {inner} lacks predictions for {missing} scored proteins.")
            for position in members:
                row = row_of[ids[position]]
                tied = bool(projection.tie[row])
                ties += int(tied)
                correct[position] = (not tied) and projection.predicted_class[row] == EC_CLASS_TOKENS[true[position]]
        systems[arm] = {"ec_classes_1_6_ba_points": correctness_balanced_accuracy(true, correct)[0],
                        "seven_class_ba_points": correctness_balanced_accuracy(true, correct, range(N_EC_CLASSES))[0],
                        "n_ties_counted_wrong": ties, "ec7_correct": int(correct[true == 6].sum())}
    support = np.bincount(true, minlength=N_EC_CLASSES)
    denominators = {"n_inner_validation_proteins": len(pooled.protein_ids), "n_in_r1_inner_closure": len(ids),
                    "n_outside_closure": int((~keep).sum()),
                    "ec_classes_1_6_averaged": int(sum(1 for index in PRIMARY_CLASS_INDICES if support[index] > 0)),
                    "support_by_ec_class": {EC_CLASS_TOKENS[index]: int(support[index])
                                            for index in range(N_EC_CLASSES)},
                    "n_by_inner_fold": {str(inner): int((folds == inner).sum()) for inner in range(N_INNER_FOLDS)}}
    gate = gap_gate_decision(systems["deepmzyme"]["ec_classes_1_6_ba_points"],
                             systems["clean_r1_inner"]["ec_classes_1_6_ba_points"], after=after)
    seven_gap = (systems["deepmzyme"]["seven_class_ba_points"]
                 - systems["clean_r1_inner"]["seven_class_ba_points"])
    return {"gate": gate, "systems": systems, "denominators": denominators,
            "seven_class_gap_points": seven_gap,
            "r2_inner_gap_points": (systems["deepmzyme"]["ec_classes_1_6_ba_points"]
                                    - systems["clean_r2_inner"]["ec_classes_1_6_ba_points"])}


def _chain_head(records: Sequence[Mapping[str, Any]], freeze1_sha256: str) -> str:
    try:
        return verify_step_chain(records, anchor_sha256=freeze1_sha256)
    except ValueError as exc:
        raise SelectionInputError(f"The step chain does not verify against --freeze1-sha256 ({exc}).") from exc


def run_gap_gate(
    *,
    fold: int,
    after: str,
    step_chain: Path,
    freeze1_sha256: str,
    plan_path: Path,
    runs_dirs: Sequence[Path],
    clean_r1_inner: Sequence[Path],
    clean_r2_inner: Sequence[Path],
    tables: InnerTables,
    out: Path,
    smoke: bool = False,
    provenance: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """The fold-0 gap gate after S1 or S3; writes the gate record ``out`` once and returns it."""
    _require(fold == GAP_GATE_FOLD, f"The gap gate runs on fold {GAP_GATE_FOLD} only, not fold {fold}.")
    _require(after in GAP_GATE_AFTER, f"--after must be one of {GAP_GATE_AFTER}, got {after!r}.")
    freeze1_sha256 = _check_sha256(freeze1_sha256, "--freeze1-sha256")
    out = Path(out)
    folder = tables.record.get("folder")
    _refuse_record_output(out, {fold: Path(folder)} if folder else {})
    _require(not out.exists(), f"--out {out.name} already exists (written once).")
    plan = load_frozen_plan(plan_path, fold)
    records_chain = _read_chain(Path(step_chain), freeze1_sha256, "--step-chain")
    chain = _fold_records(records_chain, fold)
    _require(after in chain, f"The step chain holds no {after} record of fold {fold}.")
    payload, record_hash = chain[after]
    _check_chain_inputs(payload, plan_sha256=plan["plan_sha256"], freeze1_sha256=freeze1_sha256)
    _require(_components_binding(payload.get("components")) == _components_binding(tables.record),
             "--components differs from the inner tables of the selected step record.")
    records = read_runner_records(runs_dirs, fold=fold)
    configuration = pool_selected(payload["selected"], plan, records, tables=tables,
                                  freeze1_sha256=None if smoke else freeze1_sha256,
                                  smoke=smoke)
    _component_check(tables, configuration.pooled)
    r1 = _read_clean_arm(clean_r1_inner, "r1_inner")
    r2 = _read_clean_arm(clean_r2_inner, "r2_inner")
    scores = score_gap_gate(configuration, r1, r2, after=after)
    record = {
        "format": GAP_GATE_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "fold": fold,
        "after": after,
        "freeze1_sha256": freeze1_sha256,
        "plan_sha256": plan["plan_sha256"],
        "components": dict(tables.record),
        "step_record_sha256": record_hash,
        "step_chain_head_sha256": _chain_head(records_chain, freeze1_sha256),
        "selected": dict(payload["selected"]),
        "clean_exports": {arm: {str(inner): {"npz_sha256": export.receipt.get("npz_sha256"),
                                             "seed": export.receipt.get("seed"),
                                             "checkpoint_sha256": export.receipt.get("checkpoint_sha256")}
                                for inner, export in sorted(exports.items())}
                          for arm, exports in (("r1_inner", r1), ("r2_inner", r2))},
        "scoring_rule": ("inner-validation proteins whose full EC numbers all occur among the R1-inner training "
                         "centres of their inner fold; DeepMzyme: seed mean of per-seed EC class 1-6 balanced "
                         "accuracy; CLEAN: one seed per inner fold, a tie counts as wrong"),
        **scores,
        "smoke": bool(smoke),
        "provenance": dict(provenance) if provenance is not None else provenance_record(smoke=smoke),
    }
    record["gate_record_sha256"] = write_new_json(out, record)
    return record


# ---------------------------------------------------------------------------
# Runner actions


def add_select_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The ``select`` action's arguments (see the module docstring); the handler is ``action_select``."""
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--step", choices=SELECT_STEPS, required=True)
    parser.add_argument("--plan", required=True, help="The frozen plan <campaign root>/plans/fold<K>.json.")
    parser.add_argument("--runs", action="append", required=True, help="Runs folder of the fold's fits (repeatable).")
    parser.add_argument("--components", required=True,
                        help="The data overlay's identity copy <overlay>/identity/fold<K> "
                             "(or its components_inner.tsv).")
    parser.add_argument("--freeze1-sha256", required=True, help="Freeze-1 SHA-256, the step chain's anchor.")
    parser.add_argument("--freeze1c", help="Freeze-1c JSON with the approved L3 ceiling (S2 cost skip only).")
    parser.add_argument("--l3-cost-disposition", help="Hash-bound L3 accounting and close-out decision (S2 only).")
    parser.add_argument("--out", required=True, help="Step-record chain (JSON lines; appended).")
    parser.add_argument("--smoke", action="store_true", help="Engineering use: allow a dirty tree and smoke fits.")
    return parser


def add_gap_gate_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The ``gap-gate`` action's arguments (see the module docstring); the handler is ``action_gap_gate``."""
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--after", choices=GAP_GATE_AFTER, required=True)
    parser.add_argument("--step-chain", required=True, help="The fold's step-record chain.")
    parser.add_argument("--components", required=True, help="The data overlay's certified inner identity copy.")
    parser.add_argument("--freeze1-sha256", required=True, help="Freeze-1 SHA-256, the step chain's anchor.")
    parser.add_argument("--plan", required=True, help="The frozen plan <campaign root>/plans/fold0.json.")
    parser.add_argument("--runs", action="append", required=True, help="Runs folder of the fold's fits (repeatable).")
    parser.add_argument("--clean-r1-inner", action="append", required=True,
                        help="CLEAN R1-inner C4 validation export (.npz), one per inner fold (repeatable).")
    parser.add_argument("--clean-r2-inner", action="append", required=True,
                        help="CLEAN R2-inner C4 validation export (.npz), one per inner fold (repeatable).")
    parser.add_argument("--out", required=True, help="Gate record (JSON, written once).")
    parser.add_argument("--smoke", action="store_true", help="Engineering use: allow a dirty tree and smoke fits.")
    return parser


def _print_summary(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def action_select(args: argparse.Namespace) -> int:
    """The ``select`` action: refusals, the clean-tree check (real runs), the step and a count-only summary.

    Returns 0 after appending a record and 2 when the inputs are refused (``SelectionInputError``, printed as
    ``error: ...``); ``HeldOutInputError`` and ``DirtyTreeError`` propagate to the runner.
    """
    for name in ("plan", "components", "out", "freeze1c", "l3_cost_disposition"):
        if getattr(args, name, None) is not None:
            refuse_test_paths([getattr(args, name)], what=f"--{name.replace('_', '-')}")
    refuse_test_paths(args.runs, what="--runs")
    try:
        git = real_action_git_state(smoke=args.smoke)
        tables = load_inner_tables(args.fold, Path(args.components), smoke=args.smoke)
        record = run_select(fold=args.fold, step=args.step, plan_path=Path(args.plan),
                            runs_dirs=[Path(path) for path in args.runs], tables=tables,
                            freeze1_sha256=args.freeze1_sha256, out=Path(args.out), smoke=args.smoke,
                            provenance=provenance_record(smoke=args.smoke, git=git),
                            freeze1c_path=Path(args.freeze1c) if args.freeze1c else None,
                            l3_cost_disposition_path=Path(args.l3_cost_disposition) if args.l3_cost_disposition else None)
    except SelectionInputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    payload = record["payload"]
    summary = {"step": record["step"], "index": record["index"], "fold": payload["fold"],
               "record_sha256": record_sha256(record), "n_runner_records": payload["n_runner_records"]}
    if record["step"] == STEP_S2:
        summary["levers"] = {item["lever"]: item["status"] for item in payload["screen"]["outcomes"]}
    if "selected" in payload:
        summary["selected"] = payload["selected"]["config"]
    _print_summary(summary)
    return 0


def action_gap_gate(args: argparse.Namespace) -> int:
    """The ``gap-gate`` action: refusals, the clean-tree check (real runs), the gate and a summary of totals
    (``GAP_GATE_PRINTED_TOTALS``, the decision and the gap; no per-class counts)."""
    for name in ("step_chain", "plan", "out", "components"):
        refuse_test_paths([getattr(args, name)], what=f"--{name.replace('_', '-')}")
    refuse_test_paths(args.runs, what="--runs")
    try:
        git = real_action_git_state(smoke=args.smoke)
        tables = load_inner_tables(args.fold, Path(args.components), smoke=args.smoke)
        record = run_gap_gate(fold=args.fold, after=args.after, step_chain=Path(args.step_chain),
                              freeze1_sha256=args.freeze1_sha256, plan_path=Path(args.plan),
                              runs_dirs=[Path(path) for path in args.runs],
                              clean_r1_inner=[Path(path) for path in args.clean_r1_inner],
                              clean_r2_inner=[Path(path) for path in args.clean_r2_inner], tables=tables, out=Path(args.out),
                              smoke=args.smoke, provenance=provenance_record(smoke=args.smoke, git=git))
    except SelectionInputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    # Totals only: per-EC-class counts of the fold stay in the written gate record.
    totals = {key: record["denominators"][key] for key in GAP_GATE_PRINTED_TOTALS}
    _print_summary({"after": record["after"], "decision": record["gate"]["decision"],
                    "gap_points": record["gate"]["gap_points"], "totals": totals,
                    "gate_record_sha256": record["gate_record_sha256"]})
    return 0
