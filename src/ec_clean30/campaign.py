"""Per-fold unit matrix of the EC class versus CLEAN30 campaign (torch-free).

Every outer fold runs the same pre-registered steps on its own training proteins
(campaign design, selection algorithm and fit count):

- S1: six families, each at its two learning-rate settings, on 4 inner folds with
  2 screening seeds: 6 x 2 x 4 x 2 = 96 fits.
- S2: each fit-producing lever configuration applied alone to the S1 winner on the
  same inner folds and seeds. L1-L6 and L8 are one configuration each, L7 is two
  (12 and 20 epochs), L9 is post-hoc logit adjustment without fits: at most
  9 x 8 = 72 fits. Which levers apply depends on the S1 winner, so without a winner
  the plan lists the upper bound with each configuration's applicability. L3
  (augmentation, which rebuilds every graph each epoch) runs last within S2 under its
  own sub-ceiling (campaign plan, lever table); ``S2_EXECUTION_ORDER`` and
  ``S2_SUB_CEILINGS`` record both in every plan.
- S3: one combination on the same inner folds and seeds: 8 fits.
- S4: the selected recipe refitted on the fold's full training view, seeds 42-46:
  5 fits.

So a fold has at most 181 fits; ``build_fold_plan`` asserts it. Units of S2-S4 are
templates: their recipe is fixed later by the selection step from the fold's own
inner predictions, and the plan names the recipe source instead.

Unit IDs are single path components and stable across runs, e.g.
``ecc30__fold0__S1__F3_lr3e-4-1e-4_e30__inner2__seed43`` or
``ecc30__fold0__S4__selected__refit__seed44``.

Lever settings that the campaign plan leaves to Freeze 1 (L3 strengths, L5
weight, the L1 scope of decision D5, the L8 layer) carry provisional values and
are listed in ``PROVISIONAL_LEVER_SETTINGS``; the Freeze-1 manifest records them.

Amendment 2 adds plan format v2 (``build_fold_plan_v2``, written to
``<campaign root>/plans_v2/fold<K>.json``) and two recipe-level items, without changing
any historical table (``FAMILY_SPECS``, the lever table, ``build_fold_plan`` and
``campaign_constants`` stay byte-identical, so every Freeze-1 plan hash replays):

- candidate family F6 (``F6_FAMILY_SPEC``; looked up through ``FAMILY_SPECS_V2``): the F5
  architecture whose whole-protein branch is first pretrained on the fold's full official
  training file and then frozen. The pretraining runs inside each F6 fit (``F6_PRETRAINING``
  holds the fixed rule); its classifier gives the control C-GF
  (``WHOLE_PROTEIN_PRETRAIN_CONTROL``) at no extra fit, reported and never selectable;
- the raw-distance RBF encoding of the GVP families (``Recipe.rbf_distances`` and
  ``PlannedUnit.rbf_distances``; ``"raw"`` only for families with a GVP trunk, emitted by
  ``to_dict`` only when raw, so historical recipe dictionaries and identities are unchanged).
  ``AMENDMENT2_RBF_DISTANCES`` gives the amendment-2 value per family, and v2 raw units carry
  the tag ``_rbfraw`` in their configuration name.

Plan v2 per fold:

- S1: every family at its two learning-rate settings on 4 inner folds with 2 screening
  seeds, 16 fits per family, 30 epochs: F1 (fold 0 only), F2, F3, F4, F5, F6 and C-G, so
  112 fits in fold 0 and 96 in folds 1-4. C-GF has no fits of its own.
- S2 and S3: no fits (L9 needs none; L1-L8 and S3 are not run).
- S4: the selected recipe refitted, seeds 42-46: 5.

So a v2 fold has at most 117 fits; ``build_fold_plan_v2`` asserts the caps of
``STEP_FIT_CAPS_V2``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional, Sequence

from ec_clean30.selection import (
    ANCHOR_FAMILY, CONTROL_FAMILIES, DEFAULT_EPOCHS, EPOCH_LOOKS, FAMILIES, GLOBAL_BRANCH_FAMILIES, GVP_FAMILIES,
    LEVERS, N_INNER_FOLDS, REFIT_SEEDS, S1_LEARNING_RATE_GRID, SCREENING_SEEDS, lever_applies,
)
from ec_clean30 import selection as _selection


CAMPAIGN_ID = "ec_class_vs_clean30_2026-10"
PLAN_FORMAT = "deepmzyme_ec_clean30_fold_plan_v1"
PLAN_SCHEMA_VERSION = 1
UNIT_PREFIX = "ecc30"
SMOKE_UNIT_PREFIX = "ecc30smoke"
N_OUTER_FOLDS = 5
MAX_FITS_PER_FOLD = 181

STEP_S1, STEP_S2, STEP_S3, STEP_S4 = "S1", "S2", "S3", "S4"
STEPS: tuple[str, ...] = (STEP_S1, STEP_S2, STEP_S3, STEP_S4)
KIND_INNER = "inner"
KIND_REFIT = "refit"
STEP_FIT_CAPS = {STEP_S1: 96, STEP_S2: 72, STEP_S3: 8, STEP_S4: 5}

# Families that S1 may select (the controls are reported, never selected).
CANDIDATE_FAMILIES: tuple[str, ...] = tuple(family for family in FAMILIES if family not in CONTROL_FAMILIES)

RECIPE_FIXED = "fixed"
RECIPE_S1_WINNER = "s1_winner"
RECIPE_S3_COMBINATION = "s3_combination"
RECIPE_SELECTED = "selected_recipe"

# Amendment 2: plan format v2 (the v1 constants above are unchanged). F6 pretrains inside each F6 fit, so the
# plan has no pretraining units; C-GF is read from the F6 inner fits' control files and adds no fit.
PLAN_FORMAT_V2 = "deepmzyme_ec_clean30_fold_plan_v2"
PLAN_SCHEMA_VERSION_V2 = 2
STEPS_V2: tuple[str, ...] = (STEP_S1, STEP_S2, STEP_S3, STEP_S4)
STEP_FIT_CAPS_V2 = {STEP_S1: 112, STEP_S2: 0, STEP_S3: 0, STEP_S4: 5}
MAX_FITS_PER_FOLD_V2 = 117
CANDIDATE_FAMILIES_V2: tuple[str, ...] = _selection.S1_V2_CANDIDATES
CONTROLS_BY_FOLD_V2: Mapping[int, tuple[str, ...]] = {
    fold: (_selection.S1_V2_CONTROLS_FOLD0 if fold in _selection.F1_CONTROL_FOLDS else _selection.S1_V2_CONTROLS)
    for fold in range(N_OUTER_FOLDS)
}
FAMILIES_BY_FOLD_V2: Mapping[int, tuple[str, ...]] = {
    fold: tuple(family for family in _selection.FAMILIES_V2
                if family in CANDIDATE_FAMILIES_V2 or family in CONTROLS_BY_FOLD_V2[fold])
    for fold in range(N_OUTER_FOLDS)
}
# Families without S1 fits of their own (C-GF is scored from the F6 inner fits' pretraining control files).
NO_FINE_TUNING_FAMILIES_V2: tuple[str, ...] = (_selection.WHOLE_PROTEIN_CONTROL,)
RBFRAW_CONFIG_TAG = "_rbfraw"
SEARCH_RULE_V2 = "l9_only_no_s3_v1"
CONTROLS_RULE_V2 = "controls_v2"

_UNIT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class CampaignPlanError(ValueError):
    """A plan request names an unknown family, lever or fold, or breaks the fit cap."""


# ---------------------------------------------------------------------------
# Families and levers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FamilySpec:
    """Architecture of one family; ``global_branch`` adds the whole-protein ESMC branch."""

    family: str
    model_architecture: str
    fusion_mode: Optional[str]
    global_branch: bool
    has_gvp_trunk: bool
    role: str


FAMILY_SPECS: Mapping[str, FamilySpec] = {
    "F1": FamilySpec("F1", "only_gvp", None, False, True, "control"),
    "F2": FamilySpec("F2", "only_esm", None, False, False, "anchor"),
    "F3": FamilySpec("F3", "gvp", "late_fusion", False, True, "candidate"),
    "F4": FamilySpec("F4", "gvp", "late_fusion", True, True, "candidate"),
    "F5": FamilySpec("F5", "only_esm", None, True, False, "candidate"),
    "C-G": FamilySpec("C-G", "only_global", None, True, False, "control"),
}

# ---------------------------------------------------------------------------
# Amendment 2: family F6 (whole-protein branch pretrained on the full training file) and C-GF
#
# ``FAMILY_SPECS`` stays the Freeze-1 table: ``campaign_constants`` hashes every spec into every
# historical plan, so F6 lives in a separate, versioned table. Recipes look families up there.

F6_FAMILY = "F6"
F6_FAMILY_SPEC = FamilySpec(F6_FAMILY, "only_esm", None, True, False, "candidate")
FAMILY_SPECS_V2: Mapping[str, FamilySpec] = MappingProxyType({**FAMILY_SPECS, F6_FAMILY: F6_FAMILY_SPEC})
# Families whose whole-protein branch is pretrained on the fold's full official training file.
WHOLE_PROTEIN_PRETRAIN_FAMILIES = frozenset({F6_FAMILY})
# F6 fine-tunes its metallo model on F5's learning-rate grid (the pretraining stage has its own fixed rule).
F6_LEARNING_RATE_GRID: tuple[tuple[tuple[str, float], ...], ...] = tuple(
    tuple(setting) for setting in S1_LEARNING_RATE_GRID["F5"])
# The fixed pretraining rule (amendment 2, F6-5); identical in selection and refit, never tuned.
F6_PRETRAINING: Mapping[str, Any] = MappingProxyType({
    "epochs": 30,
    "learning_rate": 1e-3,
    "batch_size": 64,
    "weight_decay": 1e-4,
    "grad_clip_norm": 1.0,
    "schedule": "cosine_per_epoch",
    "loss": "cross_entropy_inverse_frequency_class_weights",
    "head": "two_linear_layers_128_128_7_dropout_0.2",
    "branch": "frozen_after_pretraining",
    "device": "cpu",
    "dtype": "float32",
})
WP_PRETRAIN_SPLIT_REFIT = "refit"
WP_PRETRAIN_INNER_SPLIT_PREFIX = "inner"
# The by-product control: F6's pretraining classifier scored on its inner-validation proteins.
WHOLE_PROTEIN_PRETRAIN_CONTROL_ID = "C-GF"
WHOLE_PROTEIN_PRETRAIN_CONTROL: Mapping[str, Any] = MappingProxyType({
    "control": WHOLE_PROTEIN_PRETRAIN_CONTROL_ID,
    "source_family": F6_FAMILY,
    "role": "control",
    "selectable": False,
    "steps": ["S1"],
    "kinds": [KIND_INNER],
    "refit": False,
    "test_pass": False,
    "pooled_from": "the first F6 learning-rate setting of every inner fold and screening seed",
    "determinism_check": "the second F6 learning-rate setting's control predictions must be identical",
    "prediction_file": "wp_pretrain_val_protein_predictions.csv",
})


def wp_pretrain_split(kind: str, inner_fold: Optional[int]) -> str:
    """Trainer pretraining split of a unit: ``inner<j>`` for inner units, ``refit`` for refit units."""
    if kind == KIND_INNER:
        if isinstance(inner_fold, bool) or not isinstance(inner_fold, int) or inner_fold < 0:
            raise CampaignPlanError(f"Inner units need a non-negative inner fold, got {inner_fold!r}.")
        return f"{WP_PRETRAIN_INNER_SPLIT_PREFIX}{inner_fold}"
    if kind == KIND_REFIT:
        if inner_fold is not None:
            raise CampaignPlanError("Refit units have no inner fold.")
        return WP_PRETRAIN_SPLIT_REFIT
    raise CampaignPlanError(f"Unit kind must be {KIND_INNER!r} or {KIND_REFIT!r}, got {kind!r}.")


# ---------------------------------------------------------------------------
# Amendment 2: raw-distance RBFs for the GVP families (P14)

RBF_DISTANCES_NORMALIZED = "normalized"
RBF_DISTANCES_RAW = "raw"
RBF_DISTANCE_MODES: tuple[str, ...] = (RBF_DISTANCES_NORMALIZED, RBF_DISTANCES_RAW)
# The amendment-2 encoding: every family with a GVP trunk reads raw-distance RBFs; the others have none.
AMENDMENT2_RBF_DISTANCES: Mapping[str, str] = MappingProxyType({
    family: RBF_DISTANCES_RAW if spec.has_gvp_trunk else RBF_DISTANCES_NORMALIZED
    for family, spec in FAMILY_SPECS_V2.items()
})


def family_spec(family: str) -> FamilySpec:
    """The spec of a Freeze-1 family or of an amendment-2 family (F6)."""
    if family not in FAMILY_SPECS_V2:
        raise CampaignPlanError(f"Unknown family {family!r}; expected one of {list(FAMILY_SPECS_V2)}.")
    return FAMILY_SPECS_V2[family]


def recipe_learning_rate_grid(family: str) -> tuple[tuple[tuple[str, float], ...], ...]:
    """The learning-rate grid a recipe of ``family`` must use (F6 uses F5's grid)."""
    family_spec(family)
    if family == F6_FAMILY:
        return F6_LEARNING_RATE_GRID
    return tuple(tuple(setting) for setting in S1_LEARNING_RATE_GRID[family])


@dataclass(frozen=True)
class LeverConfig:
    """One fit-producing lever configuration: trainer option values set on top of a recipe.

    ``options`` maps trainer flags to values (``True`` for a bare flag). ``f1_options``
    replaces them for Only-GVP (L6 has no ESM encoder there). ``chain_mean_layer`` names
    the whole-protein cache layer the configuration reads.
    """

    config_id: str
    lever: str
    options: tuple[tuple[str, Any], ...]
    epochs: Optional[int] = None
    f1_options: Optional[tuple[tuple[str, Any], ...]] = None
    chain_mean_layer: Optional[str] = None
    provisional: tuple[str, ...] = ()

    def options_for(self, family: str) -> tuple[tuple[str, Any], ...]:
        if family == "F1" and self.f1_options is not None:
            return self.f1_options
        return self.options


L1_TARGET_SHRINK = 0.059
L8_LAYER = "block26"
LEVER_CONFIGS: Mapping[str, LeverConfig] = {
    "L1": LeverConfig("L1", "L1", (("--gvp-target-shrink", L1_TARGET_SHRINK), ("--gvp-target-shrink-scope", "trunk")),
                      provisional=("--gvp-target-shrink-scope",)),
    "L2": LeverConfig("L2", "L2", (("--rbf-use-raw-distances", True), ("--edge-radius", 8.0))),
    "L3": LeverConfig("L3", "L3", (("--position-noise-std", 0.1), ("--outer-residue-dropout", 0.1),
                                   ("--augmentation-rng", "per_fit")),
                      provisional=("--position-noise-std", "--outer-residue-dropout")),
    "L4": LeverConfig("L4", "L4", (("--ec-aux-levels", "2,3"), ("--ec-aux-weight", 0.3))),
    "L5": LeverConfig("L5", "L5", (("--ec-supcon-weight", 0.1), ("--ec-supcon-temperature", 0.1),
                                   ("--ec-supcon-memory", 256), ("--ec-supcon-projection-dim", 128)),
                      provisional=("--ec-supcon-weight", "--ec-supcon-temperature", "--ec-supcon-projection-dim")),
    "L6": LeverConfig("L6", "L6", (("--head-mlp-dropout", 0.3), ("--esm-graph-encoder-dropout", 0.2)),
                      f1_options=(("--head-mlp-dropout", 0.3),)),
    "L7e12": LeverConfig("L7e12", "L7", (), epochs=EPOCH_LOOKS[0]),
    "L7e20": LeverConfig("L7e20", "L7", (), epochs=EPOCH_LOOKS[1]),
    "L8": LeverConfig("L8", "L8", (("--global-esm-layer", L8_LAYER), ("--global-esm-standardize", "train_proteins")),
                      chain_mean_layer=L8_LAYER, provisional=("--global-esm-layer",)),
}
# L9 (post-hoc logit adjustment on the out-of-fold logits) needs no fits.
LEVERS_WITHOUT_FITS: tuple[str, ...] = ("L9",)
S2_CONFIG_IDS: tuple[str, ...] = tuple(LEVER_CONFIGS)
# Execution order of the S2 screen: L3 last, under its own sub-ceiling.
S2_LAST_CONFIG_IDS: tuple[str, ...] = ("L3",)
S2_EXECUTION_ORDER: tuple[str, ...] = (
    *(config_id for config_id in S2_CONFIG_IDS if config_id not in S2_LAST_CONFIG_IDS), *S2_LAST_CONFIG_IDS,
)
S2_SUB_CEILINGS: Mapping[str, Mapping[str, Any]] = {
    "L3": {
        "config_ids": ["L3"],
        "max_fits_per_fold": 8,
        "vm_hours_per_fold": None,
        "status": "provisional until Freeze 1 (the VM-hour value comes from the timing smoke)",
        "when_exceeded": "record L3 as 'not tested (cost)'",
    },
}
PROVISIONAL_LEVER_SETTINGS: Mapping[str, tuple[str, ...]] = {
    config_id: config.provisional for config_id, config in LEVER_CONFIGS.items() if config.provisional
}

assert set(config.lever for config in LEVER_CONFIGS.values()) | set(LEVERS_WITHOUT_FITS) == set(LEVERS)


def check_family(family: str) -> str:
    if family not in FAMILY_SPECS:
        raise CampaignPlanError(f"Unknown family {family!r}; expected one of {list(FAMILY_SPECS)}.")
    return family


def check_lever(lever: str) -> str:
    if lever not in LEVERS:
        raise CampaignPlanError(f"Unknown lever {lever!r}; expected one of {list(LEVERS)}.")
    return lever


def lever_config_ids(levers: Iterable[str]) -> tuple[str, ...]:
    """Fit-producing configurations of the given levers, in the S2 execution order (L3 last)."""
    wanted = {check_lever(lever) for lever in levers}
    return tuple(config_id for config_id in S2_EXECUTION_ORDER if LEVER_CONFIGS[config_id].lever in wanted)


def config_applies(config_id: str, family: str) -> bool:
    return lever_applies(LEVER_CONFIGS[config_id].lever, check_family(family))


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

def format_learning_rate(value: float) -> str:
    """``1e-4`` style for single-digit mantissas (the grid values), ``repr`` otherwise."""
    value = float(value)
    mantissa, exponent = f"{value:.0e}".split("e")
    if float(f"{mantissa}e{exponent}") == value:
        return f"{int(mantissa)}e{int(exponent)}"
    return repr(value)


def learning_rate_tag(setting: Sequence[tuple[str, float]]) -> str:
    """``lr1e-4`` or ``lr3e-4-1e-4`` (GVP trunk first, then the rest)."""
    values = dict(setting)
    if "gvp_learning_rate" in values:
        return f"lr{format_learning_rate(values['gvp_learning_rate'])}-{format_learning_rate(values['learning_rate'])}"
    return f"lr{format_learning_rate(values['learning_rate'])}"


@dataclass(frozen=True)
class PlannedUnit:
    """One fit of the matrix. ``family``/``learning_rate`` are ``None`` for template units."""

    unit_id: str
    fold: int
    step: str
    kind: str
    recipe_source: str
    family: Optional[str]
    learning_rate: Optional[tuple[tuple[str, float], ...]]
    epochs: Optional[int]
    lever_configs: tuple[str, ...]
    inner_fold: Optional[int]
    seed: int
    applies_to: tuple[str, ...] = ()
    sub_ceiling: Optional[str] = None  # key of S2_SUB_CEILINGS whose budget this unit is charged to
    # Amendment 2 (P14): "raw" for a fixed GVP unit of an amendment-2 plan; None keeps the historical encoding.
    rbf_distances: Optional[str] = None

    def __post_init__(self) -> None:
        if self.rbf_distances not in (None, *RBF_DISTANCE_MODES):
            raise CampaignPlanError(f"rbf_distances must be one of {RBF_DISTANCE_MODES} or None, "
                                    f"got {self.rbf_distances!r}.")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["learning_rate"] = None if self.learning_rate is None else dict(self.learning_rate)
        payload["lever_configs"] = list(self.lever_configs)
        payload["applies_to"] = list(self.applies_to)
        # Emitted only when raw, so every historical plan dictionary (and plan_sha256) is unchanged.
        if self.rbf_distances in (None, RBF_DISTANCES_NORMALIZED):
            payload.pop("rbf_distances")
        return payload


def unit_id(prefix: str, fold: int, step: str, config: str, *, inner_fold: Optional[int], seed: int) -> str:
    position = f"inner{inner_fold}" if inner_fold is not None else KIND_REFIT
    value = f"{prefix}__fold{fold}__{step}__{config}__{position}__seed{seed}"
    if not _UNIT_ID_RE.match(value) or "/" in value:
        raise CampaignPlanError(f"Unit ID {value!r} is not a single path component.")
    return value


def s1_config_name(family: str, setting: Sequence[tuple[str, float]], epochs: int) -> str:
    return f"{family}_{learning_rate_tag(setting)}_e{epochs}"


def _check_fold(fold: int) -> int:
    if isinstance(fold, bool) or not isinstance(fold, int) or not 0 <= fold < N_OUTER_FOLDS:
        raise CampaignPlanError(f"Outer fold must be an integer in 0..{N_OUTER_FOLDS - 1}, got {fold!r}.")
    return fold


def _inner_positions() -> list[tuple[int, int]]:
    return [(inner, seed) for inner in range(N_INNER_FOLDS) for seed in SCREENING_SEEDS]


def _s1_units(fold: int, families: Sequence[str]) -> list[PlannedUnit]:
    units = []
    for family in families:
        for setting in S1_LEARNING_RATE_GRID[family]:
            config = s1_config_name(family, setting, DEFAULT_EPOCHS)
            for inner, seed in _inner_positions():
                units.append(PlannedUnit(
                    unit_id=unit_id(UNIT_PREFIX, fold, STEP_S1, config, inner_fold=inner, seed=seed),
                    fold=fold, step=STEP_S1, kind=KIND_INNER, recipe_source=RECIPE_FIXED, family=family,
                    learning_rate=tuple(setting), epochs=DEFAULT_EPOCHS, lever_configs=(), inner_fold=inner,
                    seed=seed, applies_to=(family,),
                ))
    return units


def _s2_units(fold: int, config_ids: Sequence[str], s1_winner: Optional[str]) -> list[PlannedUnit]:
    units = []
    for config_id in config_ids:
        candidates = (s1_winner,) if s1_winner is not None else CANDIDATE_FAMILIES
        applies = tuple(family for family in candidates if config_applies(config_id, family))
        if not applies:
            continue
        config = LEVER_CONFIGS[config_id]
        for inner, seed in _inner_positions():
            units.append(PlannedUnit(
                unit_id=unit_id(UNIT_PREFIX, fold, STEP_S2, config_id, inner_fold=inner, seed=seed),
                fold=fold, step=STEP_S2, kind=KIND_INNER, recipe_source=RECIPE_S1_WINNER, family=s1_winner,
                learning_rate=None, epochs=config.epochs, lever_configs=(config_id,), inner_fold=inner, seed=seed,
                applies_to=applies,
                sub_ceiling=next((key for key, ceiling in S2_SUB_CEILINGS.items()
                                  if config_id in ceiling["config_ids"]), None),
            ))
    return units


def _s3_units(fold: int) -> list[PlannedUnit]:
    return [PlannedUnit(
        unit_id=unit_id(UNIT_PREFIX, fold, STEP_S3, "combination", inner_fold=inner, seed=seed),
        fold=fold, step=STEP_S3, kind=KIND_INNER, recipe_source=RECIPE_S3_COMBINATION, family=None,
        learning_rate=None, epochs=None, lever_configs=(), inner_fold=inner, seed=seed, applies_to=CANDIDATE_FAMILIES,
    ) for inner, seed in _inner_positions()]


def _s4_units(fold: int) -> list[PlannedUnit]:
    return [PlannedUnit(
        unit_id=unit_id(UNIT_PREFIX, fold, STEP_S4, "selected", inner_fold=None, seed=seed),
        fold=fold, step=STEP_S4, kind=KIND_REFIT, recipe_source=RECIPE_SELECTED, family=None, learning_rate=None,
        epochs=None, lever_configs=(), inner_fold=None, seed=seed, applies_to=CANDIDATE_FAMILIES,
    ) for seed in REFIT_SEEDS]


# Plan v2 (amendment 2)


def s1_config_name_v2(family: str, setting: Sequence[tuple[str, float]], epochs: int,
                      rbf_distances: Optional[str] = None) -> str:
    """The v2 S1 configuration name: the v1 name plus ``_rbfraw`` for raw-distance units (rule A2-TRN-2)."""
    name = s1_config_name(family, setting, epochs)
    return name + RBFRAW_CONFIG_TAG if rbf_distances == RBF_DISTANCES_RAW else name


def _unit_rbf_distances_v2(family: str) -> Optional[str]:
    """``"raw"`` for the GVP families under amendment 2, ``None`` (historical encoding, not emitted) otherwise."""
    value = AMENDMENT2_RBF_DISTANCES.get(family, RBF_DISTANCES_NORMALIZED)
    return RBF_DISTANCES_RAW if value == RBF_DISTANCES_RAW else None


def _s1_units_v2(fold: int) -> list[PlannedUnit]:
    units = []
    for family in FAMILIES_BY_FOLD_V2[fold]:
        if family in NO_FINE_TUNING_FAMILIES_V2:
            continue
        rbf = _unit_rbf_distances_v2(family)
        for setting in _selection.S1_LEARNING_RATE_GRID_V2[family]:
            config = s1_config_name_v2(family, setting, _selection.EPOCHS_V2, rbf)
            for inner, seed in _inner_positions():
                units.append(PlannedUnit(
                    unit_id=unit_id(UNIT_PREFIX, fold, STEP_S1, config, inner_fold=inner, seed=seed),
                    fold=fold, step=STEP_S1, kind=KIND_INNER, recipe_source=RECIPE_FIXED, family=family,
                    learning_rate=tuple(setting), epochs=_selection.EPOCHS_V2, lever_configs=(), inner_fold=inner,
                    seed=seed, applies_to=(family,), rbf_distances=rbf,
                ))
    return units


def _s4_units_v2(fold: int) -> list[PlannedUnit]:
    return [PlannedUnit(
        unit_id=unit_id(UNIT_PREFIX, fold, STEP_S4, "selected", inner_fold=None, seed=seed),
        fold=fold, step=STEP_S4, kind=KIND_REFIT, recipe_source=RECIPE_SELECTED, family=None, learning_rate=None,
        epochs=_selection.EPOCHS_V2, lever_configs=(), inner_fold=None, seed=seed, applies_to=CANDIDATE_FAMILIES_V2,
    ) for seed in REFIT_SEEDS]


def campaign_constants() -> dict[str, Any]:
    """Every constant the plan depends on (hashed into each plan)."""
    return {
        "campaign_id": CAMPAIGN_ID,
        "families": {name: asdict(spec) for name, spec in FAMILY_SPECS.items()},
        "anchor_family": ANCHOR_FAMILY,
        "control_families": list(CONTROL_FAMILIES),
        "candidate_families": list(CANDIDATE_FAMILIES),
        "gvp_families": sorted(GVP_FAMILIES),
        "global_branch_families": sorted(GLOBAL_BRANCH_FAMILIES),
        "s1_learning_rate_grid": {family: [dict(setting) for setting in grid]
                                  for family, grid in S1_LEARNING_RATE_GRID.items()},
        "screening_seeds": list(SCREENING_SEEDS),
        "refit_seeds": list(REFIT_SEEDS),
        "n_inner_folds": N_INNER_FOLDS,
        "default_epochs": DEFAULT_EPOCHS,
        "epoch_looks": list(EPOCH_LOOKS),
        "lever_configs": {config_id: {"lever": config.lever, "options": [[flag, value] for flag, value in config.options],
                                      "f1_options": (None if config.f1_options is None
                                                     else [[flag, value] for flag, value in config.f1_options]),
                                      "epochs": config.epochs, "chain_mean_layer": config.chain_mean_layer,
                                      "provisional": list(config.provisional)}
                          for config_id, config in LEVER_CONFIGS.items()},
        "levers_without_fits": list(LEVERS_WITHOUT_FITS),
        "s2_execution_order": list(S2_EXECUTION_ORDER),
        "s2_sub_ceilings": {key: dict(value) for key, value in S2_SUB_CEILINGS.items()},
        "step_fit_caps": dict(STEP_FIT_CAPS),
        "max_fits_per_fold": MAX_FITS_PER_FOLD,
    }


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def build_fold_plan(
    fold: int,
    *,
    families: Optional[Sequence[str]] = None,
    levers: Optional[Sequence[str]] = None,
    s1_winner: Optional[str] = None,
) -> dict[str, Any]:
    """The deterministic unit matrix of one outer fold (no timestamps; same inputs, same bytes)."""
    fold = _check_fold(fold)
    family_list = [check_family(family) for family in (families if families is not None else FAMILIES)]
    if len(set(family_list)) != len(family_list):
        raise CampaignPlanError(f"Families repeat: {family_list}")
    family_list = [family for family in FAMILIES if family in family_list]
    lever_list = [check_lever(lever) for lever in (levers if levers is not None else LEVERS)]
    if len(set(lever_list)) != len(lever_list):
        raise CampaignPlanError(f"Levers repeat: {lever_list}")
    if s1_winner is not None:
        check_family(s1_winner)
        if s1_winner not in CANDIDATE_FAMILIES:
            raise CampaignPlanError(f"{s1_winner} is a control family and is never selected in S1.")
    config_ids = lever_config_ids(lever_list)

    units = [*_s1_units(fold, family_list), *_s2_units(fold, config_ids, s1_winner), *_s3_units(fold),
             *_s4_units(fold)]
    ids = [unit.unit_id for unit in units]
    if len(ids) != len(set(ids)):
        raise CampaignPlanError("Unit IDs repeat within the plan.")
    counts = {step: sum(1 for unit in units if unit.step == step) for step in STEPS}
    for step, cap in STEP_FIT_CAPS.items():
        if counts[step] > cap:
            raise CampaignPlanError(f"{step} plans {counts[step]} fits, above its cap of {cap}.")
    counts["total"] = len(units)
    if counts["total"] > MAX_FITS_PER_FOLD:
        raise CampaignPlanError(f"Fold {fold} plans {counts['total']} fits, above the cap of {MAX_FITS_PER_FOLD}.")
    plan = {
        "format": PLAN_FORMAT,
        "schema_version": PLAN_SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "fold": fold,
        "filters": {"families": family_list, "levers": lever_list, "s1_winner": s1_winner},
        "constants": campaign_constants(),
        "counts": counts,
        "levers_not_applicable": ({config_id: "not applicable" for config_id in config_ids
                                   if not config_applies(config_id, s1_winner)} if s1_winner is not None else {}),
        "units": [unit.to_dict() for unit in units],
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan


def search_policy_record_v2() -> dict[str, Any]:
    """Amendment-2 search scope (rule A2-TRN-3): S2 is L9 only, S3 is not run, 30 epochs fixed."""
    return {
        "rule_id": SEARCH_RULE_V2,
        "screened_levers": list(_selection.SCREENED_LEVERS_V2),
        "dropped_levers": list(_selection.DROPPED_LEVERS_V2),
        "s2_fits": 0,
        "s3": "not run",
        "epochs": _selection.EPOCHS_V2,
        "schedule": _selection.TRAINING_SCHEDULE_V2,
        "checkpoint_rule": _selection.CHECKPOINT_RULE_V2,
        "epoch_selection": "none (L7 is not screened)",
        "historical_lever_configs": "unchanged and replayable (LEVER_CONFIGS); not used by plan v2",
        "rbf_distances": dict(AMENDMENT2_RBF_DISTANCES),
        "rbf_raw_unit_tag": RBFRAW_CONFIG_TAG,
    }


def controls_policy_record_v2() -> dict[str, Any]:
    """Amendment-2 control scope (rule A2-SEL-6): controls are scored and recorded, never selected."""
    return {
        "rule_id": CONTROLS_RULE_V2,
        "controls_by_fold": {str(fold): list(controls) for fold, controls in CONTROLS_BY_FOLD_V2.items()},
        "f1_folds": list(_selection.F1_CONTROL_FOLDS),
        "selectable": False,
        "refit": False,
        "whole_protein_control": {
            **dict(WHOLE_PROTEIN_PRETRAIN_CONTROL),
            "steps": list(WHOLE_PROTEIN_PRETRAIN_CONTROL["steps"]),
            "kinds": list(WHOLE_PROTEIN_PRETRAIN_CONTROL["kinds"]),
            "definition": ("the pretraining classifier of every F6 inner fit (whole-protein branch pretrained on the "
                           "fold's full official training file, inner exclusions held out), scored on the "
                           "sequence-mean vectors of that fit's inner-validation proteins; no metallo fine-tuning"),
            "extra_fits": 0,
        },
    }


def _family_specs_v2() -> dict[str, dict[str, Any]]:
    """Architecture records of the amendment-2 families (C-GF, which has no fits, gets its control record)."""
    table = globals().get("FAMILY_SPECS_V2")
    if not isinstance(table, Mapping):
        raise CampaignPlanError("Plan v2 needs the amendment-2 family table FAMILY_SPECS_V2 (with F6).")
    trained = [family for family in _selection.FAMILIES_V2 if family not in NO_FINE_TUNING_FAMILIES_V2]
    missing = [family for family in trained if family not in table]
    if missing:
        raise CampaignPlanError(f"FAMILY_SPECS_V2 lacks the amendment-2 families {missing}.")
    for family in FAMILY_SPECS:
        if asdict(table[family]) != asdict(FAMILY_SPECS[family]):
            raise CampaignPlanError(f"FAMILY_SPECS_V2 changes the Freeze-1 family {family}.")
    records = {family: asdict(table[family]) for family in trained}
    for family in NO_FINE_TUNING_FAMILIES_V2:
        records[family] = {"family": family, "role": "control", "fits": 0,
                           "source_family": WHOLE_PROTEIN_PRETRAIN_CONTROL["source_family"]}
    return {family: records[family] for family in _selection.FAMILIES_V2}


def check_family_v2(family: str) -> str:
    """A family of the amendment-2 table (``check_family`` stays the Freeze-1 check)."""
    if family not in FAMILY_SPECS_V2:
        raise CampaignPlanError(f"Unknown family {family!r}; expected one of {list(FAMILY_SPECS_V2)}.")
    return family


def config_applies_v2(config_id: str, family: str) -> bool:
    """Amendment 2 runs no S2 or S3 fits: no lever configuration applies to any v2 unit (L9 needs no fit)."""
    check_family_v2(family)
    if config_id not in LEVER_CONFIGS:
        raise CampaignPlanError(f"Unknown lever configuration {config_id!r}.")
    return False


def campaign_constants_v2() -> dict[str, Any]:
    """Every constant plan v2 depends on (hashed into each v2 plan). ``campaign_constants`` is unchanged."""
    return {
        "campaign_id": CAMPAIGN_ID,
        "plan_format": PLAN_FORMAT_V2,
        "selection_pipeline": _selection.SELECTION_PIPELINE_V2,
        "selection_policy_sha256": _selection.selection_policy_sha256_v2(),
        "freeze1_campaign_constants_sha256": canonical_sha256(campaign_constants()),
        "families": _family_specs_v2(),
        "families_by_fold": {str(fold): list(families) for fold, families in FAMILIES_BY_FOLD_V2.items()},
        "anchor_family": ANCHOR_FAMILY,
        "candidate_families": list(CANDIDATE_FAMILIES_V2),
        "controls": controls_policy_record_v2(),
        "no_fine_tuning_families": list(NO_FINE_TUNING_FAMILIES_V2),
        "gvp_families": sorted(_selection.GVP_FAMILIES_V2),
        "global_branch_families": sorted(_selection.GLOBAL_BRANCH_FAMILIES_V2),
        "full_file_pretrained_families": list(_selection.FULL_FILE_PRETRAINED_FAMILIES),
        "s1_learning_rate_grid": {family: [dict(setting) for setting in grid]
                                  for family, grid in _selection.S1_LEARNING_RATE_GRID_V2.items()},
        "screening_seeds": list(SCREENING_SEEDS),
        "refit_seeds": list(REFIT_SEEDS),
        "n_inner_folds": N_INNER_FOLDS,
        "epochs": _selection.EPOCHS_V2,
        "rbf_distances": dict(AMENDMENT2_RBF_DISTANCES),
        "f6_pretraining": dict(F6_PRETRAINING),
        "f6_pretraining_location": ("inside each F6 fit, before the F6 model is built; inner fits hold out the "
                                    "inner exclusions of their inner fold, refits use the whole corpus"),
        "search": search_policy_record_v2(),
        "step_fit_caps": dict(STEP_FIT_CAPS_V2),
        "max_fits_per_fold": MAX_FITS_PER_FOLD_V2,
    }


def build_fold_plan_v2(fold: int) -> dict[str, Any]:
    """The deterministic amendment-2 unit matrix of one outer fold (no filters, no timestamps)."""
    fold = _check_fold(fold)
    units = [*_s1_units_v2(fold), *_s4_units_v2(fold)]
    ids = [unit.unit_id for unit in units]
    if len(ids) != len(set(ids)):
        raise CampaignPlanError("Unit IDs repeat within the plan.")
    counts = {step: sum(1 for unit in units if unit.step == step) for step in STEPS_V2}
    for step, cap in STEP_FIT_CAPS_V2.items():
        if counts[step] > cap:
            raise CampaignPlanError(f"{step} plans {counts[step]} fits, above its cap of {cap}.")
    counts["total"] = len(units)
    if counts["total"] > MAX_FITS_PER_FOLD_V2:
        raise CampaignPlanError(f"Fold {fold} plans {counts['total']} fits, above the cap of {MAX_FITS_PER_FOLD_V2}.")
    plan = {
        "format": PLAN_FORMAT_V2,
        "schema_version": PLAN_SCHEMA_VERSION_V2,
        "campaign_id": CAMPAIGN_ID,
        "fold": fold,
        "families": list(FAMILIES_BY_FOLD_V2[fold]),
        "constants": campaign_constants_v2(),
        "counts": counts,
        "v1_plan_sha256": build_fold_plan(fold)["plan_sha256"],
        "units": [unit.to_dict() for unit in units],
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan


def plan_version_for_chain(chain: Sequence[str]) -> int:
    """The plan version a Freeze-1 amendment fit chain uses: 1 for the empty chain and ``[A1]``, 2 for
    ``[A1, A2]`` (contract X4); any other length is refused."""
    chain = list(chain)
    if len(chain) in (0, 1):
        return 1
    if len(chain) == 2:
        return 2
    raise CampaignPlanError(f"No plan version is defined for a fit chain of {len(chain)} links.")


def s1_units_for_plan(plan: Mapping[str, Any]) -> tuple[PlannedUnit, ...]:
    """The S1 units of a v1 or v2 plan (112 in a v2 fold-0 plan, 96 otherwise), from its verified bytes."""
    return tuple(unit for unit in plan_units(plan) if unit.step == STEP_S1)


def fold_units_v2(plan: Mapping[str, Any], step: str) -> tuple[PlannedUnit, ...]:
    """The S1 units or the S4 refit templates of a v2 plan (contract X15)."""
    if plan.get("format") != PLAN_FORMAT_V2:
        raise CampaignPlanError(f"fold_units_v2 needs a {PLAN_FORMAT_V2} plan, got {plan.get('format')!r}.")
    if step not in (STEP_S1, STEP_S4):
        raise CampaignPlanError(f"A v2 plan has fits only in S1 and S4, not {step!r}.")
    return tuple(unit for unit in plan_units(plan) if unit.step == step)


def timing_recipes_v2() -> tuple[tuple[str, Recipe], ...]:
    """The G2 v2 timing set in order (contract X11), each on inner fold 0, seed 42: the anchor F2 at 1e-4, F5,
    F6, F3 raw with the split rate, F4 raw at 3e-4/1e-4, C-G and F1 raw."""
    def grid(family: str, index: int) -> tuple[tuple[str, float], ...]:
        return tuple(_selection.S1_LEARNING_RATE_GRID_V2[family][index])

    return (
        ("anchor", Recipe("F2", grid("F2", 0), _selection.EPOCHS_V2)),
        ("F5", Recipe("F5", grid("F5", 0), _selection.EPOCHS_V2)),
        ("F6", amendment2_recipe("F6", grid("F6", 0), epochs=_selection.EPOCHS_V2)),
        ("F3_raw", amendment2_recipe("F3", grid("F3", 1), epochs=_selection.EPOCHS_V2)),
        ("F4_raw", amendment2_recipe("F4", grid("F4", 1), epochs=_selection.EPOCHS_V2)),
        ("C-G", Recipe("C-G", grid("C-G", 0), _selection.EPOCHS_V2)),
        ("F1_raw", amendment2_recipe("F1", grid("F1", 0), epochs=_selection.EPOCHS_V2)),
    )


def plan_units(plan: Mapping[str, Any]) -> list[PlannedUnit]:
    """Rebuild ``PlannedUnit`` objects from a plan dict (v1 or v2), checking its hash."""
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if canonical_sha256(body) != plan.get("plan_sha256"):
        raise CampaignPlanError("Plan content does not match its plan_sha256.")
    units = []
    for item in plan["units"]:
        lr = item["learning_rate"]
        units.append(PlannedUnit(
            unit_id=item["unit_id"], fold=item["fold"], step=item["step"], kind=item["kind"],
            recipe_source=item["recipe_source"], family=item["family"],
            learning_rate=None if lr is None else tuple(lr.items()), epochs=item["epochs"],
            lever_configs=tuple(item["lever_configs"]), inner_fold=item["inner_fold"], seed=item["seed"],
            applies_to=tuple(item["applies_to"]), sub_ceiling=item.get("sub_ceiling"),
            rbf_distances=item.get("rbf_distances"),
        ))
    return units


# ---------------------------------------------------------------------------
# Concrete recipes (what a command is built from)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Recipe:
    """A concrete training recipe: family, learning-rate setting, epoch count and lever configurations.

    ``rbf_distances`` (amendment 2, P14) selects the RBF input of the GVP families: ``"normalized"``
    (the Freeze-1 encoding, z-scored distances) or ``"raw"`` (distances in angstrom, rendered as
    ``--rbf-use-raw-distances``). ``"raw"`` is refused for families without a GVP trunk and next to
    L2 (which already sets raw distances together with 8 A edges and stays defined for replay only).
    """

    family: str
    learning_rate: tuple[tuple[str, float], ...]
    epochs: int = DEFAULT_EPOCHS
    lever_configs: tuple[str, ...] = ()
    rbf_distances: str = RBF_DISTANCES_NORMALIZED

    def __post_init__(self) -> None:
        spec = family_spec(self.family)
        grid = list(recipe_learning_rate_grid(self.family))
        if tuple(self.learning_rate) not in grid:
            raise CampaignPlanError(f"{self.family}: learning-rate setting {dict(self.learning_rate)} is not in the S1 grid.")
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, int) or self.epochs < 1:
            raise CampaignPlanError(f"Epochs must be a positive integer, got {self.epochs!r}.")
        if self.rbf_distances not in RBF_DISTANCE_MODES:
            raise CampaignPlanError(f"rbf_distances must be one of {RBF_DISTANCE_MODES}, got {self.rbf_distances!r}.")
        if self.rbf_distances == RBF_DISTANCES_RAW and not spec.has_gvp_trunk:
            raise CampaignPlanError(f"{self.family} has no GVP trunk; raw-distance RBFs apply to GVP families only.")
        if self.family in WHOLE_PROTEIN_PRETRAIN_FAMILIES and self.lever_configs:
            raise CampaignPlanError(f"{self.family} is an amendment-2 family; it takes no S2 lever configurations "
                                    "(L9 is post-hoc and needs no fits).")
        seen_levers = set()
        for config_id in self.lever_configs:
            if config_id not in LEVER_CONFIGS:
                raise CampaignPlanError(f"Unknown lever configuration {config_id!r}; expected one of {list(LEVER_CONFIGS)}.")
            lever = LEVER_CONFIGS[config_id].lever
            if lever in seen_levers:
                raise CampaignPlanError(f"Lever {lever} appears twice in one recipe.")
            seen_levers.add(lever)
            if not config_applies(config_id, self.family):
                raise CampaignPlanError(f"Lever configuration {config_id} does not apply to {self.family}.")
            if self.rbf_distances == RBF_DISTANCES_RAW and lever == "L2":
                raise CampaignPlanError("L2 already sets raw-distance RBFs (with 8 A edges); it is not combined with "
                                        "the amendment-2 raw-distance encoding.")

    @property
    def spec(self) -> FamilySpec:
        return family_spec(self.family)

    @property
    def pretrains_global_branch(self) -> bool:
        """Whether the recipe's whole-protein branch is pretrained on the full training file (F6)."""
        return self.family in WHOLE_PROTEIN_PRETRAIN_FAMILIES

    def chain_mean_layer(self) -> Optional[str]:
        if not self.spec.global_branch:
            return None
        for config_id in self.lever_configs:
            layer = LEVER_CONFIGS[config_id].chain_mean_layer
            if layer is not None:
                return layer
        return "final"

    def to_dict(self) -> dict[str, Any]:
        payload = {"family": self.family, "learning_rate": dict(self.learning_rate), "epochs": self.epochs,
                   "lever_configs": list(self.lever_configs)}
        # Emitted only when raw, so the historical recipe dictionaries (and unit identities) are unchanged.
        if self.rbf_distances == RBF_DISTANCES_RAW:
            payload["rbf_distances"] = self.rbf_distances
        return payload

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "Recipe":
        """The recipe a ``to_dict`` record describes (a record without ``rbf_distances`` is the historical form)."""
        return cls(str(record["family"]), tuple(dict(record["learning_rate"]).items()), int(record["epochs"]),
                   tuple(record.get("lever_configs") or ()),
                   str(record.get("rbf_distances") or RBF_DISTANCES_NORMALIZED))


def recipe_with_lever(base: Recipe, config_id: str) -> Recipe:
    """``base`` plus one S2 lever configuration (L7 sets the epoch count); the RBF encoding is kept."""
    config = LEVER_CONFIGS[config_id]
    return Recipe(base.family, base.learning_rate, config.epochs if config.epochs is not None else base.epochs,
                  (*base.lever_configs, config_id), base.rbf_distances)


def recipe_for_unit(unit: PlannedUnit, *, s1_winner: Optional[Recipe] = None,
                    selected: Optional[Recipe] = None) -> Recipe:
    """The concrete recipe of a planned unit; template units need the recipe their step was given."""
    if unit.recipe_source == RECIPE_FIXED:
        return Recipe(unit.family, tuple(unit.learning_rate), unit.epochs, tuple(unit.lever_configs),
                      unit.rbf_distances or RBF_DISTANCES_NORMALIZED)
    if unit.recipe_source == RECIPE_S1_WINNER:
        if s1_winner is None:
            raise CampaignPlanError(f"{unit.unit_id} needs the S1 winner recipe.")
        return recipe_with_lever(s1_winner, unit.lever_configs[0])
    if selected is None:
        raise CampaignPlanError(f"{unit.unit_id} needs the recipe chosen by {unit.recipe_source}.")
    return selected


def amendment2_recipe(family: str, learning_rate: Sequence[tuple[str, float]], *,
                      epochs: int = DEFAULT_EPOCHS) -> Recipe:
    """A recipe under the amendment-2 defaults: raw-distance RBFs for the GVP families, no S2 levers."""
    return Recipe(family, tuple(learning_rate), epochs, (), AMENDMENT2_RBF_DISTANCES[family_spec(family).family])


def default_smoke_recipe(family: str, *, epochs: int = 1, lr_index: int = 1,
                         rbf_distances: str = RBF_DISTANCES_NORMALIZED) -> Recipe:
    """Engineering-smoke recipe: the family's ``lr_index`` grid setting (default the second, which
    gives F3/F4 a distinct GVP-trunk rate and so exercises the split optimizer groups)."""
    if family not in WHOLE_PROTEIN_PRETRAIN_FAMILIES:
        check_family(family)
    grid = recipe_learning_rate_grid(family)
    if not 0 <= lr_index < len(grid):
        raise CampaignPlanError(f"{family} has {len(grid)} learning-rate settings; index {lr_index} is out of range.")
    return Recipe(family, tuple(grid[lr_index]), epochs, (), rbf_distances)


def smoke_unit_id(fold: int, recipe: Recipe, *, refit: bool, seed: int, inner_fold: int = 0) -> str:
    config = s1_config_name(recipe.family, recipe.learning_rate, recipe.epochs)
    for config_id in recipe.lever_configs:
        config += f"_{config_id}"
    if recipe.rbf_distances == RBF_DISTANCES_RAW:
        config += "_rbfraw"
    return unit_id(SMOKE_UNIT_PREFIX, _check_fold(fold), "smoke", config,
                   inner_fold=None if refit else inner_fold, seed=seed)


def fit_counts(units: Iterable[PlannedUnit]) -> dict[str, int]:
    counts = {step: 0 for step in STEPS}
    for unit in units:
        counts[unit.step] += 1
    counts["total"] = sum(counts.values())
    return counts
