"""Frozen scorer for the EC class versus CLEAN comparison on the official CLEAN split30 benchmark.

Two actions, each run once in the single scoring pass (campaign design §5 and §7, Phase 7):

``claim``
  Before any label is read, validates every frozen prediction input label-free and hashes it,
  with the strata and cluster tables, the input manifest, the Freeze-2 manifest and the scorer
  code, into a write-once claim file ``<report_dir>/prediction_claim.json`` (mode ``x``). It
  refuses a missing input, an existing claim or report, an existing scoring record of the same
  Freeze 2, a fold subset, and inputs or code that Freeze 2 does not pin. It never opens a
  label-side input (the label table and the optional EC-class transfer list): for the whole
  claim an audit hook refuses any open, listing or scan of those paths (also through another
  name of the same file), and the claim records only their declared SHA-256 values.

``score``
  Verifies the claim file against the SHA-256 recorded when it was written, re-hashes every
  claimed input and the live scorer code (against the claim and the Freeze-2 pins), rebuilds the
  predictions between two hash checks and compares them with the claim's coverage, records the
  claim in the write-once scoring record of its Freeze 2, then reads the label table once (the
  only label read; its bytes must hash to the declared value) and writes ``report.json`` and
  ``report.md`` once (mode ``x``).

One scoring pass per Freeze 2
  ``score`` creates ``<test_pass_output_root>/ec_clean30_scoring_pass.json`` (mode ``x``) before
  the label read, holding the claim and label-table SHA-256. A different claim is refused for
  that Freeze 2 from then on; the same claim may be scored again only while its reports do not
  exist (a crashed pass). This holds within one copy of the output root; a copy on another
  machine is outside its reach.

Inputs (one JSON manifest, ``kind`` ``ec_clean30_scoring_inputs``; relative paths resolve against
the manifest's folder)
  - ``test_membership``: CSV with exactly the columns ``fold, accession`` (curated test proteins
    of the prepared cohort; IDs only).
  - ``label_table``: ``{"path", "sha256"}`` of a tab-separated table with at least the columns
    ``accession`` and ``ec_numbers`` (native EC strings, ``;``-separated). Read only by ``score``.
  - ``ec_class_transfers`` (optional): ``{"path", "sha256"}`` of a CSV with exactly the columns
    ``ec_number, transferred_to`` (``;``-separated targets), built from the published nomenclature
    record of transferred entries and never from test labels. Rows whose targets all keep the
    source's EC class are ignored; a protein is flagged when one of its native EC numbers is a
    listed source that crosses an EC class. Read only by ``score``, like the label table.
  - ``freeze2_manifest``: ``{"path", "sha256"}`` of the Freeze-2 manifest (the format of
    ``training.ec_prediction_export``). Its ``code_sha256`` and ``additional_code_sha256`` must pin
    every scorer code file with the live digest, its ``checkpoint_sha256s`` must list every
    DeepMzyme and CLEAN checkpoint, and every test pass must name it.
  - ``strata``: per fold, ``strata_fold<k>.tsv`` of ``ec_clean30.identity`` (contract C5) next to
    its ``strata_receipt.json``; ``clusters``: ``components_cohort.tsv`` next to its receipt.
  - ``deepmzyme``: per fold, ``test_passes`` (label-free test pass folders of contract C3, one per
    refit seed), ``refit_receipts`` (the refits' ``ec_prediction_receipt.json``, probe mode, from
    the fold's S4 refit units) and ``logit_adjustment_tau`` (lever L9; 0 when not selected).
  - ``clean``: per arm (``r1``, ``r2``, optional ``c_esmc``) and fold, the test-mode distance
    exports of contract C4, one per seed, each with exactly the fold's test membership as queries.
  - ``baseline`` and optional ``b3``: class tables with exactly the columns ``fold, accession,
    predicted_ec_class, status`` and one row per test protein (status ``predicted``, ``no_hit``,
    ``tie`` or ``failure``; the class token is given only for ``predicted``).
  - ``settings``: ``acknowledgement_name`` (required; a bracketed placeholder is refused). The
    statistical and population settings are fixed in code (``FROZEN_SETTINGS``, pinned through
    Freeze 2); a manifest may repeat them but not change them.

Systems and arms
  An arm is one prediction vector pooled over folds (one seed, or one ensemble). DeepMzyme-final
  averages the logits of its refit seeds per protein, then applies the logit adjustment of
  ``ec_clean30.logit_adjustment`` with the refit training priors; each seed arm is adjusted with
  the same priors and tau. CLEAN distances become EC classes by the projection of
  ``run_clean_split30_comparator`` (nearest EC centre, then its first field; a centre of another
  EC class within 1e-6 of the minimum is a tie and counts as wrong); the CLEAN ensemble averages
  each EC label's distance over seeds first. A seed-mean system averages its seed arms' metrics.

Errors
  A protein a system did not predict (an explicit structure failure, a CLEAN tie, a baseline
  protein without a hit) is an error for that system: it adds to its true class's support and to
  no correct count. A test protein without any row is a contract violation, refused at claim time.
  Count tables have no abstention column, so an error goes off the diagonal: in the class-7
  column for true classes 1-6 (class 7 is not averaged, so no class 1-6 precision changes) and,
  for true class 7, in the class-1 column for accuracy and balanced accuracy, but in the class-7
  column for macro-F1 over classes 1-6, where only class precision could change (that metric
  never reads class 7). Per-class precision and F1 count real predictions only. Reported
  confusion matrices hold real predictions, with the errors listed by true class.

Populations (pooled over folds; one row per fold and accession)
  B1-all: membership proteins with one EC class (several EC classes: excluded and counted).
  B1-NH30: ``no_d3_hit_no_shared_donor``; B1-NH30-seq: ``no_d3_hit``; B1-D2: no D2 edge (the
  strata flag named by ``d2_flag``); B1-bins: ``identity_bin``. An accession listed in several
  folds is weighted to a total of one and resampled as one cluster.

Statistics (``ec_clean30.stats``, one bootstrap run per population and metric, shared draws)
  - P1 (B1-all) and P2 (B1-NH30): intersection-union tests of DeepMzyme-final minus R1 seed-mean
    and minus R1-ens, balanced accuracy over EC classes 1-6, one-sided, Holm by p-value (the
    smaller p at 0.025, the other at 0.05 only if the first passes). A co-primary bootstrap that
    exceeds the skip limit stops the run (``BootstrapAbort``), as the design states.
  - Rare-class dependence: each co-primary rerun without the proteins of EC class 5 or 6, at the
    level its Holm step used; a passing claim whose rerun bound is not above 0 is worded as
    dependent on that class (also when the rerun cannot be bootstrapped).
  - Secondary family (Holm at ``secondary_family_alpha`` over a fixed family of seven): P1/P2
    versus R2 (``r2_form``), accuracy and macro-F1 (EC classes 1-6) for P1/P2 versus both R1
    forms, and the similarity difference in differences
    (R1 seed-mean on the [30,50] and >50 bins minus on B1-NH30) - (DeepMzyme-final, same strata).
    A hypothesis whose bootstrap cannot run is untestable, ordered with p = 1, and never passes.
    The stratum support gate is metric-aware: a single-cluster class-7-only stratum leaves balanced
    accuracy over EC classes 1-6 testable, but not accuracy, nor macro-F1 when a compared system
    predicts one of its proteins as an EC class 1-6 present in the population; a gated hypothesis
    keeps its point estimates.
  - Descriptive results carry no claim: B1-NH30-seq (also the template check of P2, reported
    in both directions of a difference), B1-D2,
    B1-bins, seven-class balanced accuracy with per-protein class-7 outcomes, C-ESMC, the
    baseline, B3, DeepMzyme seed-mean versus R1 seed-mean, per-fold results and the rescore
    without EC-class transfers.

Decision wording follows the design exactly: "Outperforms CLEAN on ..." for a passing co-primary,
otherwise "Not shown to outperform CLEAN on ..."; never "equivalent". A fold subset (only reachable
from the synthetic tests) gets "No decision". The disclosures are fixed, apart from the count of
listed EC-class transfers.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Optional, Sequence

import numpy as np

from ec_clean30 import stats
from ec_clean30.campaign import CAMPAIGN_ID, KIND_REFIT, STEP_S4
from ec_clean30.identity import COMPONENT_COLUMNS, IDENTITY_BIN_LABELS, SCOPE_COHORT, STRATA_COLUMNS, read_tsv
from ec_clean30.logit_adjustment import adjust, predict_ec_class, protein_class_priors, validate_tau
from ec_clean30.selection import CONTROL_FAMILIES, REFIT_SEEDS
from project_paths import PROJECT_ROOT
from training.ec_class_metrics import N_EC_CLASSES, PRIMARY_CLASS_INDICES, ec_class_metrics
from training.ec_classes import (EC_CLASS_TOKEN_TO_INDEX, EC_CLASS_TOKENS, STATUS_AMBIGUOUS, STATUS_OK,
                                 classify_ec_class, normalize_ec_number_list)


COMPARATOR_PATH = PROJECT_ROOT / "scripts" / "run_clean_split30_comparator.py"
_COMPARATOR_MODULE = "run_clean_split30_comparator"


def _load_comparator():
    """The CLEAN comparator script as a module (contract C4 readers, projection and ensemble)."""
    module = sys.modules.get(_COMPARATOR_MODULE)
    if module is not None:
        if Path(module.__file__).resolve() != COMPARATOR_PATH.resolve():
            raise ImportError(f"{_COMPARATOR_MODULE} is already loaded from {module.__file__}")
        return module
    spec = importlib.util.spec_from_file_location(_COMPARATOR_MODULE, COMPARATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_COMPARATOR_MODULE] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_COMPARATOR_MODULE, None)
        raise
    return module


comparator = _load_comparator()


SCHEMA_VERSION = 1
MANIFEST_KIND = "ec_clean30_scoring_inputs"
CLAIM_KIND = "ec_clean30_prediction_claim"
REPORT_KIND = "ec_clean30_comparison_report"
CLAIM_NAME = "prediction_claim.json"
REPORT_JSON_NAME = "report.json"
REPORT_MD_NAME = "report.md"
ALL_FOLDS: tuple[int, ...] = tuple(range(comparator.N_FOLDS))
POINTS = 100.0

# Contract C3 (label-free test pass and run receipts), copied from training.ec_prediction_export,
# which imports torch; tests assert parity with the originals.
C3_LOGIT_COLUMNS: tuple[str, ...] = tuple(f"logit_{token}" for token in EC_CLASS_TOKENS)
C3_TEST_PROTEIN_PREDICTION_COLUMNS: tuple[str, ...] = ("pass_id", "accession", "n_pockets", "pred") + C3_LOGIT_COLUMNS
C3_TEST_PROTEIN_PREDICTIONS_NAME = "ec_test_protein_predictions.csv"
C3_TEST_RECEIPT_NAME = "ec_test_prediction_receipt.json"
C3_RUN_RECEIPT_NAME = "ec_prediction_receipt.json"
C3_MODE_PROBE = "probe"
C3_AGGREGATION = "mean_pocket_logits"
C3_FREEZE2_SCHEMA_VERSION = 1

SCORING_LEDGER_NAME = "ec_clean30_scoring_pass.json"
SCORING_LEDGER_KIND = "ec_clean30_scoring_pass"

MEMBERSHIP_COLUMNS = ("fold", "accession")
LABEL_TABLE_COLUMNS = ("accession", "ec_numbers")
TRANSFER_COLUMNS = ("ec_number", "transferred_to")
CLASS_TABLE_COLUMNS = ("fold", "accession", "predicted_ec_class", "status")

STATUS_PREDICTED = "predicted"
STATUS_TIE = "tie"
STATUS_NO_HIT = "no_hit"
STATUS_FAILURE = "failure"
PREDICTION_STATUSES = (STATUS_PREDICTED, STATUS_TIE, STATUS_NO_HIT, STATUS_FAILURE)

# Count-table columns of an error (see the module docstring).
ABSTENTION_COLUMN_FOR_CLASSES_1_6 = N_EC_CLASSES - 1
ABSTENTION_COLUMN_FOR_CLASS_7 = 0
ABSTENTION_COLUMN_FOR_MACRO_F1 = N_EC_CLASSES - 1
ABSTENTION_RULE = (
    "an unpredicted protein (explicit failure, tie, no hit) is an error: it counts in its true class's support "
    "and in no correct count; count tables place it in the class-7 column for true classes 1-6 and, for true "
    "class 7, in the class-1 column (accuracy, balanced accuracy) or the class-7 column (macro-F1 over classes "
    "1-6), so no error adds a false prediction to an averaged class; per-class precision and F1 count real "
    "predictions only"
)

POP_ALL = "B1-all"
POP_NH30 = "B1-NH30"
POP_NH30_SEQ = "B1-NH30-seq"
POP_D2 = "B1-D2"
CO_PRIMARY_POPULATIONS = {"P1": POP_ALL, "P2": POP_NH30}
HIGH_SIMILARITY_BINS: tuple[str, ...] = IDENTITY_BIN_LABELS[4:]
NO_D3_BINS: tuple[str, ...] = IDENTITY_BIN_LABELS[:4]
SUBSET_HIGH_SIMILARITY = "high_similarity"
RARE_CLASS_TOKENS = ("5", "6")
D2_FLAGS = ("has_d2", "has_d2_min_pairs")
R2_FORMS = ("seed_mean", "ensemble", "intersection_union")

CLEAN_ARMS = ("r1", "r2", "c_esmc")
REQUIRED_CLEAN_ARMS = ("r1", "r2")
CLEAN_ARM_REGIME = {"r1": "r1", "r2": "r2", "c_esmc": "r1"}

DM_FINAL = "deepmzyme_final"
DM_SEED_MEAN = "deepmzyme_seed_mean"
R1_SEED_MEAN = "r1_seed_mean"
R1_ENSEMBLE = "r1_ensemble"
R2_SEED_MEAN = "r2_seed_mean"
R2_ENSEMBLE = "r2_ensemble"
C_ESMC_SEED_MEAN = "c_esmc_seed_mean"
C_ESMC_ENSEMBLE = "c_esmc_ensemble"
BASELINE = "baseline"
B3 = "b3"
REQUIRED_SYSTEMS = (DM_FINAL, DM_SEED_MEAN, R1_SEED_MEAN, R1_ENSEMBLE, R2_SEED_MEAN, R2_ENSEMBLE, BASELINE)

PRIMARY_COMPONENTS = (f"{DM_FINAL}_minus_{R1_SEED_MEAN}", f"{DM_FINAL}_minus_{R1_ENSEMBLE}")
R2_COMPONENTS = {
    "seed_mean": (f"{DM_FINAL}_minus_{R2_SEED_MEAN}",),
    "ensemble": (f"{DM_FINAL}_minus_{R2_ENSEMBLE}",),
    "intersection_union": (f"{DM_FINAL}_minus_{R2_SEED_MEAN}", f"{DM_FINAL}_minus_{R2_ENSEMBLE}"),
}
DID_CONTRAST = "similarity_difference_in_differences"
# (name, minuend, subtrahend); a contrast is used only when both systems exist.
DESCRIPTIVE_CONTRASTS = (
    (f"{DM_FINAL}_minus_{DM_SEED_MEAN}", DM_FINAL, DM_SEED_MEAN),
    (f"{R1_ENSEMBLE}_minus_{R1_SEED_MEAN}", R1_ENSEMBLE, R1_SEED_MEAN),
    (f"{DM_SEED_MEAN}_minus_{R1_SEED_MEAN}", DM_SEED_MEAN, R1_SEED_MEAN),
    (f"{DM_FINAL}_minus_{BASELINE}", DM_FINAL, BASELINE),
    (f"{C_ESMC_SEED_MEAN}_minus_{R1_SEED_MEAN}", C_ESMC_SEED_MEAN, R1_SEED_MEAN),
    (f"{C_ESMC_ENSEMBLE}_minus_{R1_ENSEMBLE}", C_ESMC_ENSEMBLE, R1_ENSEMBLE),
    (f"{DM_FINAL}_minus_{C_ESMC_SEED_MEAN}", DM_FINAL, C_ESMC_SEED_MEAN),
    (f"{DM_FINAL}_minus_{B3}", DM_FINAL, B3),
)

DEFAULT_R1_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)
DEFAULT_THREE_SEEDS: tuple[int, ...] = (42, 43, 44)
# The scoring settings are code, so the Freeze-2 code pin fixes them; a manifest may only repeat them.
FROZEN_SETTINGS: dict[str, Any] = {
    "bootstrap_seed": stats.DEFAULT_BOOTSTRAP_SEED,
    "n_draws": stats.DEFAULT_N_DRAWS,
    "max_skip_fraction": stats.DEFAULT_MAX_SKIP_FRACTION,
    "secondary_family_alpha": 0.05,
    "descriptive_alpha": 0.05,
    "d2_flag": "has_d2",
    "r2_form": "seed_mean",
    "deepmzyme_seeds": list(REFIT_SEEDS),
    "r1_seeds": list(DEFAULT_R1_SEEDS),
    "r2_seeds": list(DEFAULT_THREE_SEEDS),
    "c_esmc_seeds": list(DEFAULT_THREE_SEEDS),
}
MANIFEST_SETTINGS = ("acknowledgement_name",)
_PLACEHOLDER_CHARACTERS = frozenset("[]{}<>")
# In-process switches for the synthetic tests only (fewer draws, two folds). No manifest key and
# no command-line flag reaches them; a report scored with them is marked and warned.
_TEST_SETTING_OVERRIDES: dict[str, Any] = {}
_TEST_ALLOW_FOLD_SUBSET = False

DECISION_PASS = {
    "P1": "Outperforms CLEAN on the official CLEAN split30 benchmark",
    "P2": "Outperforms CLEAN on test proteins without a ≥30% identity hit to training",
}
DECISION_NOT_SHOWN = {
    "P1": "Not shown to outperform CLEAN on the official CLEAN split30 benchmark",
    "P2": "Not shown to outperform CLEAN on test proteins without a ≥30% identity hit to training",
}
DECISION_NO_DECISION = "No decision: only folds {folds} of the five official folds were scored"
# Analysis amendment of 2026-10-30 (before Freeze 1): a population whose bootstrap has a stratum of
# EC classes 1-6 with a single homology cluster is reported descriptively, with no superiority
# inference; no class or cluster is removed to make it eligible.
DECISION_UNAVAILABLE = {
    "P1": "No superiority inference on the official CLEAN split30 benchmark: a bootstrap stratum of EC classes "
          "1-6 holds a single homology cluster, so between-family uncertainty cannot be estimated; results are "
          "descriptive",
    "P2": "No superiority inference on test proteins without a ≥30% identity hit to training: a bootstrap stratum "
          "of EC classes 1-6 holds a single homology cluster, so between-family uncertainty cannot be estimated; "
          "results are descriptive",
}
STRATUM_GATE_REASON = ("superiority inference unavailable: a bootstrap stratum of EC classes 1-6 holds a single "
                       "homology cluster ({strata})")
# The class-7-only exemption covers balanced accuracy over EC classes 1-6 only: class-7 proteins enter
# accuracy directly, and macro-F1 through false positives of EC classes 1-6.
METRIC_STRATUM_GATE_REASON = ("superiority inference unavailable for {metric}: a bootstrap stratum that this metric "
                              "reads holds a single homology cluster ({strata}); the class-7-only exemption covers "
                              "balanced accuracy over EC classes 1-6 only")
SIX_CLASSES: tuple[int, ...] = tuple(index + 1 for index in PRIMARY_CLASS_INDICES)
POPULATION_WORDING = {
    "P1": "official CLEAN split30 benchmark (nominal 30% clustering)",
    "P2": "test proteins without a ≥30% identity hit to training",
}

DISCLOSURES: tuple[tuple[str, str], ...] = (
    ("language_model_pretraining",
     "ESM-1b (CLEAN) and ESMC (DeepMzyme) were pretrained on UniRef sequences, which can include test proteins "
     "and their homologs; neither the official folds nor B1-NH30 control this exposure."),
    ("structure_prediction_inputs",
     "DeepMzyme reads AlphaFold models, whose multiple sequence alignments and templates can include training "
     "proteins and their homologs; B1-NH30 controls sequence identity and shared AlphaFill donors, not this "
     "exposure."),
    ("seen_ec_number_test_closure",
     "Upstream curation kept only test proteins whose full EC number occurs in their fold's training file; this "
     "closure favours CLEAN's nearest-centre design and is not changed."),
    ("metallo_cohort_screening_gap",
     "The metalloenzyme cohort was screened with AlphaFill and MAHOMES over fold-0 training and the fold-0 curated "
     "test only; about 439 proteins that train folds 1-4 were never screened, so those folds likely lack about 30 "
     "metalloenzymes each. DeepMzyme and CLEAN-R2 share this gap; CLEAN-R1 does not."),
    ("computational_metal_sites",
     "Metal sites are computational (AlphaFill transfers and MAHOMES predictions on predicted structures), not "
     "experimentally observed."),
    ("deepmzyme_trains_on_metalloenzymes_only",
     "DeepMzyme trains on each fold's metalloenzymes only (about 600 proteins); CLEAN-R1 trains on the fold's full "
     "official training file (about 8,100 enzymes)."),
    ("tuning_asymmetry",
     "CLEAN runs its published recipe without tuning, while DeepMzyme selects its recipe per fold from that fold's "
     "training proteins; the upstream epoch rule was chosen on these same folds."),
    ("primary_metric_deviation",
     "The primary metric, balanced accuracy over EC classes 1-6 from seven-way predictions, is a declared "
     "deviation from the default seven-class EC balanced accuracy, because class 7 has only a few test proteins; "
     "accuracy and seven-class balanced accuracy are reported beside it."),
    ("design_time_label_exposure",
     "Labels of fold-test proteins were read through other folds' training files (inherent to the five-fold "
     "design); the historical CARE development set behind the learning-rate evidence and family priors contains "
     "at least 48 of the 118 fold-0 metallo test proteins; the planning pools mixed fold-test proteins."),
    ("prior_test_metadata_exposure",
     "Earlier exposures of test metadata are recorded in the dataset ledger, docs/DATASETS.md; these records "
     "exposed no EC label of this benchmark's test proteins."),
    ("clean_keyed_masks",
     "CLEAN masked copies are drawn from a random stream keyed by (accession, seed) instead of upstream's unseeded "
     "global generator; the masking procedure itself is unchanged."),
    ("clean_fair_esm_version",
     "CLEAN embeddings use fair-esm 2.0.0, as the upstream requirements state; the CLEAN manuscript reports "
     "fair-esm 1.0.2."),
    ("clean_pythonhashseed",
     "Upstream CLEAN sets PYTHONHASHSEED inside the running process, where it has no effect; here each training "
     "process starts with PYTHONHASHSEED equal to its seed and the distance-map process with 0."),
)
TRANSFER_DISCLOSURE_ID = "ec_class_nomenclature_transfers"
ACKNOWLEDGEMENT_TEMPLATE = (
    "The CLEAN Software, used by {name}, was developed by the Department of Chemical and Biomolecular "
    "Engineering at the University of Illinois Urbana-Champaign."
)


def transfer_disclosure(row_summary: Mapping[str, Any], list_supplied: bool) -> str:
    """The disclosure of later nomenclature transfers that cross an EC class (plan §4.2)."""
    if not list_supplied:
        return ("No list of later nomenclature transfers that cross an EC class was supplied, so none is disclosed "
                "and the rescore without them was not run.")
    listed = int(row_summary["n_ec_class_transfers_listed"])
    flagged = int(row_summary["n_b1_all_rows_with_ec_class_transfer"])
    if not listed:
        return "The transfer list names no EC number that a later nomenclature transfer moved to another EC class."
    return (f"The transfer list names {listed} EC number(s) that a later nomenclature transfer moved to another EC "
            f"class; {flagged} B1-all row(s) carry one. Labels are not remapped; a descriptive rescore excludes "
            "them.")

# Every repository file the scorer imports or runs; the claim pins them and the score re-checks them.
SCORER_CODE_PATHS: tuple[str, ...] = (
    "src/ec_clean30/scoring.py",
    "scripts/score_ec_clean30_comparison.py",
    "src/ec_clean30/__init__.py",
    "src/ec_clean30/stats.py",
    "src/ec_clean30/logit_adjustment.py",
    "src/ec_clean30/selection.py",
    "src/ec_clean30/campaign.py",
    "src/ec_clean30/identity.py",
    "src/audit_sequence_remoteness.py",
    "src/project_paths.py",
    "src/training/__init__.py",
    "src/training/access_guard.py",
    "src/training/ec_class_metrics.py",
    "src/training/ec_classes.py",
    "src/training/ec_membership.py",
    "scripts/run_clean_split30_comparator.py",
)

_SHA256_HEX = frozenset("0123456789abcdef")
# A seed-mean system averages its arms in floating point, so identical predictions can differ from a
# single arm by a few units in the last place. Contrast values this close to zero are set to exactly
# zero, which keeps identical systems at p = 1 and a lower bound of 0 (conservative: a genuine
# difference of a metric fraction is many orders of magnitude larger).
CONTRAST_ZERO_TOLERANCE = 1e-12


class ScoringError(RuntimeError):
    """A scoring input, claim or output contract failed."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise ScoringError(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _check_sha256(value: Any, what: str) -> str:
    text = str(value).strip().lower()
    _require(len(text) == 64 and set(text) <= _SHA256_HEX, f"{what} must be a SHA-256 hex digest, got {value!r}")
    return text


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _jsonable(value: Any) -> Any:
    """JSON-safe copy: tuples become lists, numpy scalars plain numbers, non-finite floats None."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


def _json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"


def _write_text_once(path: Path, text: str) -> str:
    try:
        with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    except FileExistsError:
        raise ScoringError(f"Refusing to overwrite {path} (write-once)") from None
    return _sha256_bytes(text.encode("utf-8"))


# ---------------------------------------------------------------- label-side access guard

_GUARDED_EVENTS = frozenset({"open", "os.listdir", "os.scandir", "glob.glob", "glob.glob/2"})
_BLOCKED_PATHS: dict[str, int] = {}
_BLOCKED_INODES: dict[tuple[int, int], int] = {}
_BLOCKED_ROOTS: dict[str, int] = {}
_COUNTED_PATHS: dict[str, int] = {}
_HOOK_INSTALLED = False


def _normalize_path(path: Any) -> str:
    return os.path.normcase(os.path.realpath(os.fspath(path)))


def _inode(path: Any) -> Optional[tuple[int, int]]:
    """(device, inode) of an existing file, so another name of the same file (a hard link) is recognised."""
    try:
        status = os.stat(path)
    except (OSError, ValueError):
        return None
    return int(status.st_dev), int(status.st_ino)


def _audit_hook(event: str, args: tuple) -> None:
    if not (_BLOCKED_PATHS or _BLOCKED_ROOTS or _COUNTED_PATHS) or event not in _GUARDED_EVENTS or not args:
        return
    target = args[0]
    if isinstance(target, bytes):
        target = os.fsdecode(target)
    if not isinstance(target, (str, os.PathLike)):
        return
    normalized = _normalize_path(target)
    if normalized in _BLOCKED_PATHS or (_BLOCKED_INODES and _inode(target) in _BLOCKED_INODES):
        raise PermissionError(f"This step of the scorer may not read a label-side input: {target}")
    if any(normalized == root or normalized.startswith(root.rstrip(os.sep) + os.sep) for root in _BLOCKED_ROOTS):
        raise PermissionError(f"The scorer may not read under a held-out root: {target}")
    if normalized in _COUNTED_PATHS:
        _COUNTED_PATHS[normalized] += 1


def _install_audit_hook() -> None:
    global _HOOK_INSTALLED
    if not _HOOK_INSTALLED:
        sys.addaudithook(_audit_hook)
        _HOOK_INSTALLED = True


def _acquire(registry: dict, keys: Sequence) -> None:
    for key in keys:
        registry[key] = registry.get(key, 0) + 1


def _release(registry: dict, keys: Sequence) -> None:
    for key in keys:
        registry[key] -= 1
        if not registry[key]:
            del registry[key]


@contextmanager
def label_read_guard(paths: Sequence[Path | str], roots: Sequence[Path | str] = ()) -> Iterator[tuple[str, ...]]:
    """Refuse every open, listing or scan of ``paths``, and of anything under ``roots``, inside the block.

    An existing path is also refused under any other name of the same file (same device and inode).
    An audit hook (installed once per process) enforces it; outside every block it does nothing.
    """
    _install_audit_hook()
    keys = tuple(_normalize_path(path) for path in paths)
    inodes = tuple(inode for inode in (_inode(path) for path in paths) if inode is not None)
    root_keys = tuple(_normalize_path(root) for root in roots)
    _acquire(_BLOCKED_PATHS, keys)
    _acquire(_BLOCKED_INODES, inodes)
    _acquire(_BLOCKED_ROOTS, root_keys)
    try:
        yield keys
    finally:
        _release(_BLOCKED_PATHS, keys)
        _release(_BLOCKED_INODES, inodes)
        _release(_BLOCKED_ROOTS, root_keys)


@contextmanager
def label_read_counter(paths: Sequence[Path | str]) -> Iterator[dict[str, int]]:
    """Count the open, listing and scan events of ``paths`` inside the block."""
    _install_audit_hook()
    keys = tuple(_normalize_path(path) for path in paths)
    for key in keys:
        _require(key not in _COUNTED_PATHS, f"{key} is already counted")
        _COUNTED_PATHS[key] = 0
    counts: dict[str, int] = {}
    try:
        yield counts
    finally:
        for key in keys:
            counts[key] = _COUNTED_PATHS.pop(key)


# ---------------------------------------------------------------- settings and manifest

@dataclass(frozen=True)
class DeclaredFile:
    """A file named with the SHA-256 recorded when it was frozen."""

    path: Path
    sha256: str


LabelSideInput = DeclaredFile


@dataclass(frozen=True)
class DeepMzymeFoldInputs:
    test_passes: tuple[Path, ...]
    refit_receipts: tuple[Path, ...]
    tau: float


@dataclass(frozen=True)
class ScoringManifest:
    path: Path
    sha256: str
    report_dir: Path
    folds: tuple[int, ...]
    test_membership: Path
    label_table: LabelSideInput
    ec_class_transfers: Optional[LabelSideInput]
    strata: dict[int, Path]
    clusters: Path
    deepmzyme: dict[int, DeepMzymeFoldInputs]
    clean: dict[str, dict[int, tuple[Path, ...]]]
    baseline: Path
    b3: Optional[Path]
    freeze2: DeclaredFile
    settings: dict[str, Any]

    @property
    def label_side_paths(self) -> tuple[Path, ...]:
        inputs = [self.label_table] + ([self.ec_class_transfers] if self.ec_class_transfers else [])
        return tuple(item.path for item in inputs)

    @property
    def label_side_sha256s(self) -> frozenset[str]:
        inputs = [self.label_table] + ([self.ec_class_transfers] if self.ec_class_transfers else [])
        return frozenset(item.sha256 for item in inputs)


_MANIFEST_KEYS = {"schema_version", "kind", "report_dir", "folds", "test_membership", "label_table",
                  "ec_class_transfers", "freeze2_manifest", "strata", "clusters", "deepmzyme", "clean", "baseline",
                  "b3", "settings"}
_OPTIONAL_MANIFEST_KEYS = {"ec_class_transfers", "b3"}
_DM_FOLD_KEYS = {"test_passes", "refit_receipts", "logit_adjustment_tau"}


def _seed_list(value: Any, name: str) -> list[int]:
    _require(isinstance(value, list) and value and all(isinstance(seed, int) and not isinstance(seed, bool)
                                                       for seed in value),
             f"Setting {name} must be a non-empty list of integer seeds")
    _require(len(set(value)) == len(value), f"Setting {name} repeats a seed")
    return sorted(int(seed) for seed in value)


def resolve_settings(raw: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """The scoring settings: ``FROZEN_SETTINGS`` plus the manifest's ``acknowledgement_name``.

    A manifest may repeat a frozen setting but not change it; unknown keys are refused.
    """
    raw = dict(raw or {})
    unknown = sorted(set(raw) - set(FROZEN_SETTINGS) - set(MANIFEST_SETTINGS))
    _require(not unknown, f"Unknown scoring settings {unknown}")
    name = raw.get("acknowledgement_name")
    _require(isinstance(name, str) and name.strip() and not set(name) & _PLACEHOLDER_CHARACTERS,
             "Setting acknowledgement_name must name who uses the CLEAN Software (required; a placeholder such as "
             "'[name]' is refused)")
    frozen = validate_settings(FROZEN_SETTINGS)
    for key in sorted(set(raw) & set(FROZEN_SETTINGS)):
        given = validate_settings({**FROZEN_SETTINGS, key: raw[key]})[key]
        _require(given == frozen[key], f"Setting {key} = {raw[key]!r} differs from the frozen value {frozen[key]!r} "
                                       "(the scoring settings are fixed in code and pinned through Freeze 2)")
    settings = validate_settings({**FROZEN_SETTINGS, **_TEST_SETTING_OVERRIDES})
    settings["acknowledgement_name"] = name.strip()
    return settings


def settings_are_frozen(settings: Mapping[str, Any]) -> bool:
    """Whether ``settings`` hold exactly the frozen values (false only under the synthetic tests' overrides)."""
    frozen = validate_settings(FROZEN_SETTINGS)
    return all(settings.get(key) == value for key, value in frozen.items())


def validate_settings(values: Mapping[str, Any]) -> dict[str, Any]:
    """Type and range checks of the frozen setting keys; returns a normalised copy."""
    settings = dict(values)
    missing = sorted(set(FROZEN_SETTINGS) - set(settings))
    _require(not missing, f"Scoring settings lack {missing}")
    for name in ("bootstrap_seed", "n_draws"):
        _require(isinstance(settings[name], int) and not isinstance(settings[name], bool) and settings[name] >= 0,
                 f"Setting {name} must be a non-negative integer")
    _require(settings["n_draws"] >= 1, "Setting n_draws must be positive")
    for name in ("max_skip_fraction", "secondary_family_alpha", "descriptive_alpha"):
        _require(isinstance(settings[name], (int, float)) and not isinstance(settings[name], bool),
                 f"Setting {name} must be a number")
        settings[name] = float(settings[name])
    _require(0.0 <= settings["max_skip_fraction"] <= 1.0, "Setting max_skip_fraction must lie in [0, 1]")
    for name in ("secondary_family_alpha", "descriptive_alpha"):
        _require(0.0 < settings[name] < 1.0, f"Setting {name} must lie strictly between 0 and 1")
    _require(settings["d2_flag"] in D2_FLAGS, f"Setting d2_flag must be one of {D2_FLAGS}")
    _require(settings["r2_form"] in R2_FORMS, f"Setting r2_form must be one of {R2_FORMS}")
    for name in ("deepmzyme_seeds", "r1_seeds", "r2_seeds", "c_esmc_seeds"):
        settings[name] = _seed_list(settings[name], name)
    _require(len(settings["r1_seeds"]) >= 2, "R1-ens needs at least two R1 seeds")
    return settings


def _fold_key(key: Any, where: str) -> int:
    text = str(key)
    _require(text.isdigit() and int(text) in ALL_FOLDS, f"{where}: fold key {key!r} is not one of {ALL_FOLDS}")
    return int(text)


def _per_fold(value: Any, folds: Sequence[int], where: str) -> dict[int, Any]:
    _require(isinstance(value, Mapping), f"{where} must map fold numbers to inputs")
    result = {_fold_key(key, where): item for key, item in value.items()}
    _require(len(result) == len(value), f"{where} repeats a fold")
    _require(sorted(result) == list(folds),
             f"{where} covers folds {sorted(result)}, the manifest declares {list(folds)}")
    return result


def _path_list(value: Any, base: Path, where: str) -> tuple[Path, ...]:
    _require(isinstance(value, list) and value and all(isinstance(item, str) and item for item in value),
             f"{where} must be a non-empty list of paths")
    paths = tuple(_resolve(item, base) for item in value)
    _require(len(set(paths)) == len(paths), f"{where} repeats a path")
    return paths


def _resolve(value: Any, base: Path) -> Path:
    _require(isinstance(value, str) and value.strip(), f"Expected a path, got {value!r}")
    path = Path(value)
    return (path if path.is_absolute() else base / path).absolute()


def _declared_file(value: Any, base: Path, where: str) -> DeclaredFile:
    _require(isinstance(value, Mapping) and set(value) == {"path", "sha256"},
             f"{where} must be {{'path', 'sha256'}} (the SHA-256 recorded when the file was frozen)")
    return DeclaredFile(_resolve(value["path"], base), _check_sha256(value["sha256"], f"{where} sha256"))


def load_manifest(path: Path | str) -> ScoringManifest:
    """Parse the scoring input manifest (only the manifest file is read)."""
    path = Path(path).absolute()
    _require(path.is_file(), f"Scoring manifest not found: {path}")
    data = path.read_bytes()
    try:
        raw = json.loads(data.decode("utf-8"))
    except ValueError as exc:
        raise ScoringError(f"Scoring manifest {path} is not JSON: {exc}") from None
    _require(isinstance(raw, dict), "The scoring manifest must be a JSON object")
    unknown = sorted(set(raw) - _MANIFEST_KEYS)
    missing = sorted(_MANIFEST_KEYS - _OPTIONAL_MANIFEST_KEYS - set(raw))
    _require(not unknown and not missing, f"Scoring manifest keys: unknown {unknown}, missing {missing}")
    _require(raw["schema_version"] == SCHEMA_VERSION and raw["kind"] == MANIFEST_KIND,
             f"Scoring manifest must have schema_version {SCHEMA_VERSION} and kind {MANIFEST_KIND!r}")
    base = path.parent
    folds = raw["folds"]
    _require(isinstance(folds, list) and folds and all(isinstance(fold, int) and fold in ALL_FOLDS for fold in folds)
             and len(set(folds)) == len(folds), f"folds must be distinct values of {ALL_FOLDS}")
    folds = tuple(sorted(folds))

    deepmzyme = {}
    for fold, entry in _per_fold(raw["deepmzyme"], folds, "deepmzyme").items():
        _require(isinstance(entry, Mapping) and set(entry) == _DM_FOLD_KEYS,
                 f"deepmzyme fold {fold} must have exactly {sorted(_DM_FOLD_KEYS)}")
        try:
            tau = validate_tau(entry["logit_adjustment_tau"])
        except (TypeError, ValueError) as exc:
            raise ScoringError(f"deepmzyme fold {fold}: {exc}") from None
        deepmzyme[fold] = DeepMzymeFoldInputs(
            _path_list(entry["test_passes"], base, f"deepmzyme fold {fold} test_passes"),
            _path_list(entry["refit_receipts"], base, f"deepmzyme fold {fold} refit_receipts"), tau)

    clean_raw = raw["clean"]
    _require(isinstance(clean_raw, Mapping) and set(clean_raw) <= set(CLEAN_ARMS)
             and set(REQUIRED_CLEAN_ARMS) <= set(clean_raw),
             f"clean must hold {list(REQUIRED_CLEAN_ARMS)} and optionally c_esmc")
    clean = {arm: {fold: _path_list(paths, base, f"clean {arm} fold {fold}")
                   for fold, paths in _per_fold(clean_raw[arm], folds, f"clean {arm}").items()}
             for arm in CLEAN_ARMS if arm in clean_raw}

    transfers = raw.get("ec_class_transfers")
    b3 = raw.get("b3")
    return ScoringManifest(
        path=path,
        sha256=_sha256_bytes(data),
        report_dir=_resolve(raw["report_dir"], base),
        folds=folds,
        test_membership=_resolve(raw["test_membership"], base),
        label_table=_declared_file(raw["label_table"], base, "label_table"),
        ec_class_transfers=None if transfers is None else _declared_file(transfers, base, "ec_class_transfers"),
        strata={fold: _resolve(item, base) for fold, item in _per_fold(raw["strata"], folds, "strata").items()},
        clusters=_resolve(raw["clusters"], base),
        deepmzyme=deepmzyme,
        clean=clean,
        baseline=_resolve(raw["baseline"], base),
        b3=None if b3 is None else _resolve(b3, base),
        freeze2=_declared_file(raw["freeze2_manifest"], base, "freeze2_manifest"),
        settings=resolve_settings(raw["settings"]),
    )


def _require_all_folds(folds: Sequence[int]) -> None:
    _require(tuple(folds) == ALL_FOLDS or _TEST_ALLOW_FOLD_SUBSET,
             f"The scoring pass covers all folds {list(ALL_FOLDS)}; the manifest declares {list(folds)}")


# ---------------------------------------------------------------- Freeze 2 and the scoring record

@dataclass(frozen=True)
class Freeze2:
    """The parts of the Freeze-2 manifest the scorer checks."""

    path: Path
    sha256: str
    checkpoint_sha256s: frozenset[str]
    code_sha256: dict[str, str]  # ``code_sha256`` and ``additional_code_sha256`` together
    test_pass_output_root: Path

    @property
    def scoring_ledger(self) -> Path:
        return self.test_pass_output_root / SCORING_LEDGER_NAME


def read_freeze2_manifest(reference: DeclaredFile) -> Freeze2:
    """The Freeze-2 manifest, parsed from the bytes that hash to its declared SHA-256."""
    _require(reference.path.is_file(), f"Freeze-2 manifest not found: {reference.path}")
    data = reference.path.read_bytes()
    _require(_sha256_bytes(data) == reference.sha256,
             f"Freeze-2 manifest {reference.path} does not hash to its declared SHA-256 {reference.sha256}")
    try:
        payload = json.loads(data.decode("utf-8"))
    except ValueError as exc:
        raise ScoringError(f"Freeze-2 manifest {reference.path} is not JSON: {exc}") from None
    _require(isinstance(payload, dict) and payload.get("schema_version") == C3_FREEZE2_SCHEMA_VERSION,
             f"Freeze-2 manifest {reference.path} must be a JSON object with schema_version "
             f"{C3_FREEZE2_SCHEMA_VERSION}")
    checkpoints = payload.get("checkpoint_sha256s")
    _require(isinstance(checkpoints, list) and checkpoints, "Freeze-2 manifest lists no checkpoint_sha256s")
    code: dict[str, str] = {}
    for key in ("code_sha256", "additional_code_sha256"):
        table = payload.get(key, {} if key == "additional_code_sha256" else None)
        _require(isinstance(table, dict), f"Freeze-2 manifest {key} must map repository paths to SHA-256 values")
        for path, digest in table.items():
            digest = _check_sha256(digest, f"Freeze-2 manifest {key}[{path!r}]")
            _require(code.setdefault(str(path), digest) == digest, f"Freeze-2 manifest pins {path} twice, differently")
    root = payload.get("test_pass_output_root")
    _require(isinstance(root, str) and Path(root).is_absolute(),
             "Freeze-2 manifest test_pass_output_root must be an absolute path")
    return Freeze2(path=reference.path, sha256=reference.sha256,
                   checkpoint_sha256s=frozenset(_check_sha256(item, "Freeze-2 checkpoint") for item in checkpoints),
                   code_sha256=code, test_pass_output_root=Path(root))


def check_freeze2_code(freeze2: Freeze2, code: Mapping[str, str]) -> None:
    """Every scorer code file must be pinned by Freeze 2 with the same digest (plan Phase 6, step 3)."""
    missing = sorted(path for path in code if path not in freeze2.code_sha256)
    differing = sorted(path for path in code if path in freeze2.code_sha256 and freeze2.code_sha256[path] != code[path])
    _require(not missing, f"The Freeze-2 manifest does not pin {len(missing)} scorer code file(s): {missing[:5]}")
    _require(not differing, f"Scorer code differs from its Freeze-2 pin: {differing[:5]}")


def _same_root(left: Any, right: Path) -> bool:
    return isinstance(left, str) and bool(left) and Path(left).resolve() == Path(right).resolve()


def _read_scoring_ledger(path: Path) -> dict[str, Any]:
    record = _json_file(path)
    _require(record.get("kind") == SCORING_LEDGER_KIND, f"{path} is not a {SCORING_LEDGER_KIND} record")
    return record


def claim_scoring_pass(freeze2: Freeze2, claim_path: Path, claim_sha256: str,
                       label_sha256: str) -> dict[str, Any]:
    """Record the claim in the write-once scoring record of its Freeze 2, before the label read.

    A different claim is refused for this Freeze 2; the same claim may continue after a crash.
    """
    ledger = freeze2.scoring_ledger
    record = {"schema_version": SCHEMA_VERSION, "kind": SCORING_LEDGER_KIND, "created_utc": _utc_now(),
              "claim": {"path": str(claim_path), "sha256": claim_sha256}, "label_table_sha256": label_sha256,
              "freeze2_manifest_sha256": freeze2.sha256}
    ledger.parent.mkdir(parents=True, exist_ok=True)
    try:
        _write_text_once(ledger, _json_text(record))
        return {"path": str(ledger), "resumed": False}
    except ScoringError:
        existing = _read_scoring_ledger(ledger)
    same = ((existing.get("claim") or {}).get("sha256") == claim_sha256
            and existing.get("label_table_sha256") == label_sha256
            and existing.get("freeze2_manifest_sha256") == freeze2.sha256)
    _require(same, f"{ledger} already records the scoring pass of claim "
                   f"{(existing.get('claim') or {}).get('sha256')}; one scoring pass per Freeze 2")
    return {"path": str(ledger), "resumed": True, "first_created_utc": existing.get("created_utc")}


# ---------------------------------------------------------------- input files and code

@dataclass(frozen=True)
class InputFile:
    role: str
    path: Path
    fold: Optional[int] = None
    system: Optional[str] = None


def strata_receipt_path(strata_path: Path) -> Path:
    return strata_path.with_name("strata_receipt.json")


def clusters_receipt_path(clusters_path: Path) -> Path:
    return clusters_path.with_name(f"components_receipt_{SCOPE_COHORT}.json")


def manifest_input_files(manifest: ScoringManifest) -> list[InputFile]:
    """Every frozen input the claim hashes (label-side inputs are not among them)."""
    files = [InputFile("manifest", manifest.path), InputFile("freeze2_manifest", manifest.freeze2.path),
             InputFile("test_membership", manifest.test_membership), InputFile("clusters", manifest.clusters),
             InputFile("clusters_receipt", clusters_receipt_path(manifest.clusters))]
    for fold in manifest.folds:
        strata = manifest.strata[fold]
        files += [InputFile("strata", strata, fold), InputFile("strata_receipt", strata_receipt_path(strata), fold)]
        dm = manifest.deepmzyme[fold]
        for folder in dm.test_passes:
            files += [
                InputFile("deepmzyme_test_predictions", folder / C3_TEST_PROTEIN_PREDICTIONS_NAME, fold, DM_FINAL),
                InputFile("deepmzyme_test_receipt", folder / C3_TEST_RECEIPT_NAME, fold, DM_FINAL)]
        files += [InputFile("deepmzyme_refit_receipt", receipt, fold, DM_FINAL) for receipt in dm.refit_receipts]
        for arm, per_fold in manifest.clean.items():
            for npz in per_fold[fold]:
                files += [InputFile("clean_distance_export", npz, fold, arm),
                          InputFile("clean_distance_receipt", comparator.c4_receipt_path(npz), fold, arm)]
    files.append(InputFile("baseline_predictions", manifest.baseline, None, BASELINE))
    if manifest.b3 is not None:
        files.append(InputFile("b3_predictions", manifest.b3, None, B3))
    return files


def code_sha256() -> dict[str, str]:
    """SHA-256 of every repository file the scorer imports or runs."""
    digests = {}
    for relative in SCORER_CODE_PATHS:
        path = PROJECT_ROOT / relative
        _require(path.is_file(), f"Scorer code file missing: {relative}")
        digests[relative] = comparator.sha256_file(path)
    return digests


def held_out_roots() -> list[Path]:
    """The comparator's held-out roots (official split30 test files, mixed CLEAN30 folders, other tests)."""
    return comparator.resolve_forbidden_roots()


def _check_input_paths(manifest: ScoringManifest, files: Sequence[InputFile], roots: Sequence[Path]) -> None:
    label_keys = {_normalize_path(path) for path in manifest.label_side_paths}
    label_inodes = {inode for inode in (_inode(path) for path in manifest.label_side_paths) if inode is not None}
    clash = sorted(str(item.path) for item in files
                   if _normalize_path(item.path) in label_keys or _inode(item.path) in label_inodes)
    _require(not clash, f"A label-side input is listed as a prediction input: {clash}")
    root_keys = [_normalize_path(root) for root in roots]
    held_out = sorted(str(item.path) for item in files
                      if any(_normalize_path(item.path) == root or _normalize_path(item.path).startswith(root + os.sep)
                             for root in root_keys))
    _require(not held_out, f"Inputs under a held-out root: {held_out[:5]}")
    paths = [item.path for item in files]
    _require(len(set(paths)) == len(paths), "An input file is listed twice in the manifest")
    missing = sorted(str(item.path) for item in files if not item.path.is_file())
    _require(not missing, f"{len(missing)} frozen input file(s) are missing, e.g. {missing[:5]}")


def hash_input_files(files: Sequence[InputFile]) -> list[dict[str, Any]]:
    return [{"role": item.role, "fold": item.fold, "system": item.system, "path": str(item.path),
             "sha256": comparator.sha256_file(item.path), "size_bytes": item.path.stat().st_size} for item in files]


def _verify_hashed_inputs(entries: Sequence[Mapping[str, Any]], when: str) -> None:
    changed, missing = [], []
    for entry in entries:
        path = Path(entry["path"])
        if not path.is_file():
            missing.append(str(path))
        elif comparator.sha256_file(path) != entry["sha256"]:
            changed.append(str(path))
    _require(not missing, f"{len(missing)} claimed input(s) are missing, e.g. {missing[:5]}")
    _require(not changed, f"{len(changed)} claimed input(s) changed {when}, e.g. {changed[:5]}")


# ---------------------------------------------------------------- label-free readers

@dataclass(frozen=True)
class Prediction:
    token: Optional[str]  # EC class token "1".."7", or None for an error
    status: str


def _read_csv_text(text: str, columns: Sequence[str], what: str, *, delimiter: str = ",") -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    _require(tuple(reader.fieldnames or ()) == tuple(columns), f"{what} must have exactly the columns {list(columns)}")
    rows = []
    for row in reader:
        _require(None not in row and all(value is not None for value in row.values()), f"{what} has a malformed row")
        rows.append({key: value.strip() for key, value in row.items()})
    return rows


def _accession(value: str, what: str) -> str:
    _require(bool(comparator.ACCESSION_RE.match(value)), f"Invalid accession {value!r} in {what}")
    return value


def _fold_value(value: str, folds: Sequence[int], what: str) -> int:
    _require(value.isdigit() and int(value) in folds, f"{what}: fold {value!r} is not a scored fold {list(folds)}")
    return int(value)


def read_test_membership(path: Path, folds: Sequence[int]) -> dict[int, tuple[str, ...]]:
    """Curated test accessions per fold (IDs only)."""
    rows = _read_csv_text(path.read_text(encoding="utf-8"), MEMBERSHIP_COLUMNS, path.name)
    members: dict[int, list[str]] = {fold: [] for fold in folds}
    seen = set()
    for row in rows:
        fold = _fold_value(row["fold"], folds, path.name)
        accession = _accession(row["accession"], path.name)
        _require((fold, accession) not in seen, f"{path.name} repeats fold {fold} accession {accession}")
        seen.add((fold, accession))
        members[fold].append(accession)
    empty = [fold for fold, accessions in members.items() if not accessions]
    _require(not empty, f"{path.name} lists no test protein for folds {empty}")
    return {fold: tuple(sorted(accessions)) for fold, accessions in members.items()}


def _json_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ScoringError(f"{path} is not JSON: {exc}") from None
    _require(isinstance(payload, dict), f"{path} must hold a JSON object")
    return payload


def read_clusters(path: Path) -> tuple[dict[str, str], str]:
    """Cohort cluster IDs (components over all cohort proteins) and the receipt's Freeze 1 SHA-256."""
    expected = f"components_{SCOPE_COHORT}.tsv"
    _require(path.name == expected, f"clusters must be a {expected}, got {path.name}")
    receipt = _json_file(clusters_receipt_path(path))
    _require(receipt.get("scope") == SCOPE_COHORT, f"{clusters_receipt_path(path).name} is not a cohort receipt")
    _require(((receipt.get("outputs") or {}).get(expected) or {}).get("sha256") == comparator.sha256_file(path),
             f"{expected} does not match its receipt")
    clusters = {}
    for row in read_tsv(path, COMPONENT_COLUMNS):
        _require(row["accession"] not in clusters and row["component_id"], f"{expected} repeats or lacks a row")
        clusters[row["accession"]] = row["component_id"]
    return clusters, _check_sha256(receipt.get("freeze1_sha256"), "clusters receipt freeze1_sha256")


def _flag(row: Mapping[str, str], name: str, where: str) -> bool:
    value = row[name]
    _require(value in {"0", "1"}, f"{where}: {name} must be 0 or 1, got {value!r}")
    return value == "1"


def _check_strata_row(row: Mapping[str, str], clusters: Mapping[str, str], where: str) -> None:
    flags = {name: _flag(row, name, where) for name in (
        "has_any_hit", "has_bin_hit", "has_d3", "has_d2", "has_d2_min_pairs", "has_d1",
        "donor_shared_with_metallo_train", "no_d3_hit", "no_d3_hit_no_shared_donor")}
    _require(row["identity_bin"] in IDENTITY_BIN_LABELS, f"{where}: unknown identity bin {row['identity_bin']!r}")
    consistent = (
        flags["no_d3_hit"] == (not flags["has_d3"])
        and flags["no_d3_hit_no_shared_donor"] == (not flags["has_d3"] and not flags["donor_shared_with_metallo_train"])
        and (row["identity_bin"] in NO_D3_BINS) == (not flags["has_d3"])
        and (row["identity_bin"] == IDENTITY_BIN_LABELS[0]) == (not flags["has_bin_hit"])
        and (not flags["has_d3"] or flags["has_bin_hit"]) and (not flags["has_bin_hit"] or flags["has_any_hit"])
    )
    _require(consistent, f"{where}: identity flags and bin disagree")
    cluster = row["cluster_id"]
    _require(bool(cluster) and clusters.get(row["accession"]) == cluster,
             f"{where}: cluster_id is empty or differs from the cohort clusters table")


def read_strata(path: Path, fold: int, clusters: Mapping[str, str], clusters_sha256: str) -> tuple[dict, str]:
    """One fold's complement-stratum table (contract C5), checked against its receipt."""
    expected = f"strata_fold{fold}.tsv"
    _require(path.name == expected, f"Fold {fold} strata must be {expected}, got {path.name}")
    receipt = _json_file(strata_receipt_path(path))
    checks = {
        "step": receipt.get("step") == "strata", "fold": receipt.get("fold") == fold,
        "output hash": (((receipt.get("outputs") or {}).get(expected) or {}).get("sha256")
                        == comparator.sha256_file(path)),
        "labels_read": receipt.get("labels_read") is False,
        "curated_test_membership_read": receipt.get("curated_test_membership_read") is False,
        "identity bins": receipt.get("identity_bins") == list(IDENTITY_BIN_LABELS),
        "clusters": (receipt.get("inputs") or {}).get("clusters_sha256") == clusters_sha256,
    }
    failed = sorted(name for name, ok in checks.items() if not ok)
    _require(not failed, f"{strata_receipt_path(path)} fails the checks {failed}")
    rows = {}
    for row in read_tsv(path, STRATA_COLUMNS):
        where = f"{expected} {row['accession']}"
        _require(row["accession"] not in rows, f"{expected} repeats {row['accession']}")
        _check_strata_row(row, clusters, where)
        rows[row["accession"]] = row
    return rows, _check_sha256(receipt.get("freeze1_sha256"), f"{expected} receipt freeze1_sha256")


def _finite_logits(row: Mapping[str, str], where: str) -> list[float]:
    try:
        values = [float(row[column]) for column in C3_LOGIT_COLUMNS]
    except ValueError:
        raise ScoringError(f"{where}: a logit is not a number") from None
    _require(all(math.isfinite(value) for value in values), f"{where}: logits must be finite")
    return values


def _read_test_pass(folder: Path) -> dict[str, Any]:
    receipt = _json_file(folder / C3_TEST_RECEIPT_NAME)
    csv_path = folder / C3_TEST_PROTEIN_PREDICTIONS_NAME
    checks = {
        "label_free": receipt.get("label_free") is True, "labels_read": receipt.get("labels_read") is False,
        "metrics_computed": receipt.get("metrics_computed") is False,
        "ec_class_tokens": receipt.get("ec_class_tokens") == list(EC_CLASS_TOKENS),
        "aggregation": receipt.get("aggregation") == C3_AGGREGATION,
        "predictions file": (receipt.get("protein_predictions") or {}).get("path") == C3_TEST_PROTEIN_PREDICTIONS_NAME,
        "predictions hash": ((receipt.get("protein_predictions") or {}).get("sha256")
                             == comparator.sha256_file(csv_path)),
        "seed": isinstance(receipt.get("seed"), int), "failures": isinstance(receipt.get("failures"), list),
        "input_structures": isinstance(receipt.get("input_structures"), list),
        "model": isinstance(receipt.get("model"), dict),
    }
    failed = sorted(name for name, ok in checks.items() if not ok)
    _require(not failed, f"{folder / C3_TEST_RECEIPT_NAME} fails the checks {failed}")
    pass_id = str(receipt.get("pass_id", ""))
    rows = _read_csv_text(csv_path.read_text(encoding="utf-8"), C3_TEST_PROTEIN_PREDICTION_COLUMNS, str(csv_path))
    logits: dict[str, list[float]] = {}
    for row in rows:
        where = f"{csv_path} {row['accession']}"
        _require(row["pass_id"] == pass_id, f"{where}: pass_id differs from the receipt")
        accession = _accession(row["accession"], str(csv_path))
        _require(accession not in logits, f"{csv_path} repeats {accession}")
        _require(row["n_pockets"].isdigit() and int(row["n_pockets"]) >= 1, f"{where}: n_pockets must be positive")
        values = _finite_logits(row, where)
        top = sorted(values, reverse=True)
        _require(row["pred"] == str(int(np.argmax(values))) or top[0] == top[1],
                 f"{where}: pred is not the arg-max of the logits")
        logits[accession] = values
    recorded_rows = (receipt.get("protein_predictions") or {}).get("n_rows")
    _require(len(rows) == recorded_rows == receipt.get("n_proteins_predicted"),
             f"{csv_path}: row count differs from its receipt")
    failures = {_accession(str(item.get("accession", "")), f"{folder} failures") for item in receipt["failures"]}
    inputs = {_accession(str(item.get("accession", "")), f"{folder} inputs") for item in receipt["input_structures"]}
    _require(not failures & set(logits), f"{folder}: an accession is both predicted and failed")
    _require(inputs == failures | set(logits), f"{folder}: predictions and failures do not cover the input structures")
    return {"pass_id": pass_id, "seed": int(receipt["seed"]), "checkpoint_sha256": receipt.get("checkpoint_sha256"),
            "model": receipt["model"], "freeze2": receipt.get("freeze2_manifest") or {},
            "logits": logits, "failures": failures, "inputs": inputs}


def _read_refit_receipt(path: Path) -> dict[str, Any]:
    receipt = _json_file(path)
    _require(path.name == C3_RUN_RECEIPT_NAME, f"A refit receipt must be named {C3_RUN_RECEIPT_NAME}: {path}")
    counts = receipt.get("training_protein_class_counts")
    checks = {
        "mode": receipt.get("mode") == C3_MODE_PROBE, "seed": isinstance(receipt.get("seed"), int),
        "checkpoint": isinstance((receipt.get("checkpoint") or {}).get("sha256"), str),
        "ec_class_tokens": receipt.get("ec_class_tokens") == list(EC_CLASS_TOKENS),
        "model": isinstance(receipt.get("model"), dict),
        "class counts": isinstance(counts, dict) and sorted(counts) == sorted(EC_CLASS_TOKENS)
        and all(isinstance(counts[token], int) and counts[token] >= 0 for token in EC_CLASS_TOKENS),
        "run_id": isinstance(receipt.get("run_id"), str) and bool(receipt.get("run_id")),
        "campaign_run_identity": isinstance(receipt.get("campaign_run_identity"), dict),
    }
    failed = sorted(name for name, ok in checks.items() if not ok)
    _require(not failed, f"{path} fails the checks {failed}")
    return {"seed": int(receipt["seed"]), "checkpoint_sha256": receipt["checkpoint"]["sha256"],
            "model": receipt["model"], "counts": tuple(int(counts[token]) for token in EC_CLASS_TOKENS),
            "run_id": receipt["run_id"], "identity": receipt["campaign_run_identity"], "path": str(path)}


def _check_refit_identity(refit: Mapping[str, Any], fold: int, freeze1_sha256: str) -> None:
    """A refit receipt must come from this fold's S4 refit unit of the campaign, under the scored Freeze 1.

    Every fold-k test protein trains the other folds, so a refit bound to another fold has seen the test.
    """
    identity = refit["identity"]
    checks = {
        "campaign_id": identity.get("campaign_id") == CAMPAIGN_ID,
        "kind": identity.get("kind") == KIND_REFIT,
        "step": identity.get("step") == STEP_S4,
        "fold": identity.get("fold") == fold and not isinstance(identity.get("fold"), bool),
        "inner_fold": identity.get("inner_fold") is None,
        "smoke": identity.get("smoke") is False,
        "seed": identity.get("seed") == refit["seed"],
        "unit_id": identity.get("unit_id") == refit["run_id"],
        "freeze1_sha256": identity.get("freeze1_sha256") == freeze1_sha256,
        "recipe": isinstance(identity.get("recipe"), dict),
    }
    failed = sorted(name for name, ok in checks.items() if not ok)
    _require(not failed, f"{refit['path']}: the run identity is not fold {fold}'s S4 refit under the scored "
                         f"Freeze 1 (fails {failed})")


def refit_priors(counts: Sequence[int]) -> np.ndarray:
    """Add-one smoothed protein priors from the refit's training protein counts per class."""
    indices = np.repeat(np.arange(N_EC_CLASSES), np.asarray(counts, dtype=np.int64))
    _require(indices.size > 0, "Refit training protein counts are all zero")
    return protein_class_priors(indices)


def load_deepmzyme_fold(fold: int, inputs: DeepMzymeFoldInputs, membership: Sequence[str], settings: Mapping[str, Any],
                        freeze2: Freeze2, freeze1_sha256: str) -> tuple[dict[str, dict], dict[str, Any]]:
    """DeepMzyme-final and per-seed arms of one fold from its label-free test passes and refit receipts."""
    passes = [_read_test_pass(folder) for folder in inputs.test_passes]
    refits = [_read_refit_receipt(path) for path in inputs.refit_receipts]
    where = f"DeepMzyme fold {fold}"
    seeds = sorted(item["seed"] for item in passes)
    _require(seeds == settings["deepmzyme_seeds"] == sorted(item["seed"] for item in refits),
             f"{where}: test-pass seeds {seeds} and refit seeds must both equal {settings['deepmzyme_seeds']}")
    for refit in refits:
        _check_refit_identity(refit, fold, freeze1_sha256)
    recipes = {json.dumps(refit["identity"]["recipe"], sort_keys=True) for refit in refits}
    _require(len(recipes) == 1, f"{where}: the refit seeds differ in recipe")
    refit_by_checkpoint = {item["checkpoint_sha256"]: item for item in refits}
    _require(len(refit_by_checkpoint) == len(refits), f"{where}: refit receipts repeat a checkpoint")
    for item in passes:
        refit = refit_by_checkpoint.get(item["checkpoint_sha256"])
        _require(refit is not None and refit["seed"] == item["seed"],
                 f"{where}: test pass {item['pass_id']} has no refit receipt with its checkpoint and seed")
        _require(item["checkpoint_sha256"] in freeze2.checkpoint_sha256s,
                 f"{where}: test pass {item['pass_id']} used a checkpoint that Freeze 2 does not list")
        _require(item["freeze2"].get("sha256") == freeze2.sha256
                 and _same_root(item["freeze2"].get("test_pass_output_root"), freeze2.test_pass_output_root),
                 f"{where}: test pass {item['pass_id']} ran under another Freeze-2 manifest or output root")
    models = [item["model"] for item in passes + refits]
    _require(all(model == models[0] for model in models), f"{where}: the seeds differ in model identity")
    family = models[0].get("campaign_family")
    _require(family is not None and family not in CONTROL_FAMILIES,
             f"{where}: model family {family!r} is not a selectable candidate")
    counts = {item["counts"] for item in refits}
    _require(len(counts) == 1, f"{where}: refit training class counts differ between seeds")
    predicted = {frozenset(item["logits"]) for item in passes}
    failed = {frozenset(item["failures"]) for item in passes}
    _require(len(predicted) == 1 and len(failed) == 1, f"{where}: seeds predicted different proteins")
    accessions = sorted(next(iter(predicted)))
    member_set = set(membership)
    covered = set(accessions) | next(iter(failed))
    extra = sorted(covered - member_set)
    _require(not extra, f"{where}: predictions for accessions outside the fold's test membership, e.g. {extra[:5]}")
    uncovered = sorted(member_set - covered)
    _require(not uncovered, f"{where}: {len(uncovered)} test protein(s) have neither a prediction nor an explicit "
                            f"failure in the test passes, e.g. {uncovered[:5]}")

    priors = refit_priors(next(iter(counts)))
    tau = inputs.tau
    by_seed = sorted(passes, key=lambda item: item["seed"])
    stacked = np.array([[item["logits"][accession] for accession in accessions] for item in by_seed], dtype=np.float64)
    arms: dict[str, dict[str, Prediction]] = {}
    if accessions:
        final = predict_ec_class(adjust(stacked.mean(axis=0), priors, tau))
        arms[DM_FINAL] = {acc: Prediction(EC_CLASS_TOKENS[int(k)], STATUS_PREDICTED)
                          for acc, k in zip(accessions, final)}
        for item, seed_logits in zip(by_seed, stacked):
            seed_pred = predict_ec_class(adjust(seed_logits, priors, tau))
            arms[f"deepmzyme_seed{item['seed']}"] = {
                acc: Prediction(EC_CLASS_TOKENS[int(k)], STATUS_PREDICTED) for acc, k in zip(accessions, seed_pred)}
    else:
        arms[DM_FINAL] = {}
        arms.update({f"deepmzyme_seed{item['seed']}": {} for item in by_seed})
    for arm in arms.values():
        for accession in membership:
            arm.setdefault(accession, Prediction(None, STATUS_FAILURE))
    details = {
        "campaign_family": family, "model": models[0], "logit_adjustment_tau": tau,
        "refit_training_protein_class_counts": dict(zip(EC_CLASS_TOKENS, next(iter(counts)))),
        "refit_priors": priors.tolist(), "seeds": seeds, "pass_ids": sorted(item["pass_id"] for item in passes),
        "checkpoint_sha256s": sorted(item["checkpoint_sha256"] for item in passes),
        "refit_unit_ids": sorted(item["run_id"] for item in refits),
        "recipe": refits[0]["identity"]["recipe"],
        "freeze2_sha256": freeze2.sha256,
        "n_failures_reported": len(next(iter(failed))),
    }
    return arms, details


def load_clean_fold(arm: str, fold: int, exports_paths: Sequence[Path], membership: Sequence[str],
                    settings: Mapping[str, Any], freeze2: Freeze2) -> tuple[dict[str, dict], dict[str, Any]]:
    """Per-seed and ensemble arms of one CLEAN arm and fold from contract C4 test exports.

    Each export must query exactly the fold's test membership: a missing query would otherwise
    become a CLEAN error, which favours DeepMzyme. Each must also carry the test-export binding of
    the scored Freeze 2 (its manifest SHA-256 and output root, the pinned runner and the pinned
    export script; ``check_test_export_binding`` of the comparator).
    """
    where = f"CLEAN {arm} fold {fold}"
    try:
        exports = [comparator.read_distance_export(path) for path in exports_paths]
    except (comparator.ComparatorError, OSError, ValueError, KeyError) as exc:
        raise ScoringError(f"{where}: {exc}") from None
    seeds = sorted(export.receipt["seed"] for export in exports)
    _require(seeds == settings[f"{arm}_seeds"], f"{where}: seeds {seeds} must equal {settings[f'{arm}_seeds']}")
    embeddings = set()
    for export in exports:
        receipt = export.receipt
        _require(receipt["mode"] == "test" and receipt["fold"] == fold and receipt["regime"] == CLEAN_ARM_REGIME[arm],
                 f"{where}: an export has mode {receipt['mode']!r}, fold {receipt['fold']}, "
                 f"regime {receipt['regime']!r}")
        if arm == "c_esmc":
            _require(receipt["embedding"] != comparator.DEFAULT_EMBEDDING, f"{where}: C-ESMC uses the ESM-1b embedding")
        else:
            _require(receipt["embedding"] == comparator.DEFAULT_EMBEDDING
                     and receipt["input_dim"] == comparator.DEFAULT_INPUT_DIM,
                     f"{where}: R1 and R2 use {comparator.DEFAULT_EMBEDDING} at width {comparator.DEFAULT_INPUT_DIM}")
        extra = sorted(set(export.query_ids) - set(membership))
        _require(not extra, f"{where}: queries outside the fold's test membership, e.g. {extra[:5]}")
        absent = sorted(set(membership) - set(export.query_ids))
        _require(not absent, f"{where}: seed {receipt['seed']} lacks {len(absent)} test protein(s) of the fold, "
                             f"e.g. {absent[:5]}")
        _require(receipt["checkpoint_sha256"] in freeze2.checkpoint_sha256s,
                 f"{where}: seed {receipt['seed']} used a checkpoint that Freeze 2 does not list")
        try:
            comparator.check_test_export_binding(
                receipt, freeze2_sha256=freeze2.sha256, test_pass_output_root=freeze2.test_pass_output_root,
                runner_sha256=freeze2.code_sha256.get(comparator.RUNNER_KEY))
        except comparator.ComparatorError as exc:
            raise ScoringError(f"{where}: seed {receipt['seed']}: {exc}") from None
        embeddings.add((receipt["embedding"], receipt["input_dim"]))
    _require(len(embeddings) == 1, f"{where}: seeds use different embeddings")
    _require(len({export.receipt["checkpoint_sha256"] for export in exports}) == len(exports),
             f"{where}: two seeds share a checkpoint")
    arms: dict[str, dict[str, Prediction]] = {}
    ties: dict[str, int] = {}
    try:
        projections = [(f"{arm}_seed{export.receipt['seed']}",
                        comparator.project_ec_classes(export.distances, export.query_ids, export.ec_labels))
                       for export in exports]
        if len(exports) >= 2:
            projections.append((f"{arm}_ensemble", comparator.project_ensemble(exports)))
    except comparator.ComparatorError as exc:
        raise ScoringError(f"{where}: {exc}") from None
    for name, projection in projections:
        predictions = {}
        for query, token, tied in zip(projection.query_ids, projection.predicted_class, projection.tie):
            predictions[query] = Prediction(None, STATUS_TIE) if tied else Prediction(str(token), STATUS_PREDICTED)
        arms[name] = predictions
        ties[name] = projection.n_ties
    embedding, input_dim = next(iter(embeddings))
    details = {"seeds": seeds, "embedding": embedding, "input_dim": input_dim, "n_ties": ties,
               "checkpoint_sha256s": sorted(export.receipt["checkpoint_sha256"] for export in exports)}
    return arms, details


def read_class_table(path: Path, membership: Mapping[int, Sequence[str]],
                     system: str) -> dict[tuple[int, str], Prediction]:
    """Baseline or B3 class predictions: exactly one explicit row per test protein (``no_hit``,
    ``tie`` and ``failure`` are statuses, so a missing row is refused, not counted)."""
    folds = sorted(membership)
    what = f"{system} predictions {path.name}"
    predictions: dict[tuple[int, str], Prediction] = {}
    for row in _read_csv_text(path.read_text(encoding="utf-8"), CLASS_TABLE_COLUMNS, what):
        fold = _fold_value(row["fold"], folds, what)
        accession = _accession(row["accession"], what)
        key = (fold, accession)
        _require(accession in membership[fold], f"{what}: {accession} is not a fold-{fold} test protein")
        _require(key not in predictions, f"{what} repeats fold {fold} accession {accession}")
        status, token = row["status"], row["predicted_ec_class"]
        _require(status in PREDICTION_STATUSES, f"{what}: unknown status {status!r}")
        _require((status == STATUS_PREDICTED) == (token in EC_CLASS_TOKENS)
                 and (status == STATUS_PREDICTED or not token),
                 f"{what}: {accession} needs a class token exactly when its status is {STATUS_PREDICTED!r}")
        predictions[key] = Prediction(token or None, status)
    absent = sorted(f"fold {fold} {accession}" for fold, accessions in membership.items() for accession in accessions
                    if (fold, accession) not in predictions)
    _require(not absent, f"{what} has no row for {len(absent)} test protein(s), e.g. {absent[:5]}")
    return predictions


@dataclass
class LoadedInputs:
    membership: dict[int, tuple[str, ...]]
    strata: dict[int, dict[str, dict[str, str]]]
    clusters: dict[str, str]
    arms: dict[str, dict[tuple[int, str], Prediction]]
    systems: dict[str, tuple[str, ...]]
    details: dict[str, Any]

    def coverage(self) -> dict[str, Any]:
        """Prediction status counts per arm and fold over the test membership (label-free)."""
        record: dict[str, Any] = {}
        for arm in sorted(self.arms):
            per_fold = {}
            for fold, accessions in self.membership.items():
                statuses = [self.arms[arm][(fold, accession)].status for accession in accessions]
                per_fold[str(fold)] = {"n_membership": len(accessions),
                                       **{status: statuses.count(status) for status in PREDICTION_STATUSES}}
            record[arm] = per_fold
        return record


def _merge_fold_arms(target: dict[str, dict[tuple[int, str], Prediction]], fold: int,
                     arms: Mapping[str, Mapping[str, Prediction]]) -> None:
    for name, predictions in arms.items():
        target.setdefault(name, {}).update({(fold, accession): value for accession, value in predictions.items()})


def system_definitions(arm_names: Sequence[str], settings: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    """Systems as tuples of arms; a seed-mean system averages its seed arms' metrics."""
    names = set(arm_names)
    systems = {DM_FINAL: (DM_FINAL,),
               DM_SEED_MEAN: tuple(f"deepmzyme_seed{seed}" for seed in settings["deepmzyme_seeds"])}
    for arm in CLEAN_ARMS:
        seeds = tuple(f"{arm}_seed{seed}" for seed in settings[f"{arm}_seeds"])
        if all(seed in names for seed in seeds):
            systems[f"{arm}_seed_mean"] = seeds
        if f"{arm}_ensemble" in names:
            systems[f"{arm}_ensemble"] = (f"{arm}_ensemble",)
    for single in (BASELINE, B3):
        if single in names:
            systems[single] = (single,)
    unknown = sorted({arm for arms in systems.values() for arm in arms} - names)
    _require(not unknown, f"Systems refer to missing arms {unknown}")
    return systems


def load_label_free_inputs(manifest: ScoringManifest, freeze2: Freeze2) -> LoadedInputs:
    """Every prediction, stratum and cluster input, validated without reading any label."""
    settings = manifest.settings
    membership = read_test_membership(manifest.test_membership, manifest.folds)
    clusters, freeze1 = read_clusters(manifest.clusters)
    clusters_sha256 = comparator.sha256_file(manifest.clusters)
    strata = {}
    for fold in manifest.folds:
        rows, fold_freeze1 = read_strata(manifest.strata[fold], fold, clusters, clusters_sha256)
        _require(fold_freeze1 == freeze1, f"Fold {fold} strata record another Freeze 1 SHA-256 than the clusters")
        missing = sorted(set(membership[fold]) - set(rows))
        _require(not missing, f"Fold {fold}: test proteins without a strata row, e.g. {missing[:5]}")
        strata[fold] = rows
    repeated = sorted({acc for fold, accs in membership.items() for acc in accs
                       if sum(acc in others for others in membership.values()) > 1})
    for accession in repeated:
        ids = {strata[fold][accession]["cluster_id"] for fold in manifest.folds if accession in membership[fold]}
        _require(len(ids) == 1, f"{accession} is listed in several folds with different clusters")

    arms: dict[str, dict[tuple[int, str], Prediction]] = {}
    details: dict[str, Any] = {"freeze1_sha256": freeze1, "repeated_accessions": repeated,
                               "deepmzyme": {}, "clean": {arm: {} for arm in manifest.clean}}
    pass_ids: list[str] = []
    for fold in manifest.folds:
        dm_arms, dm_details = load_deepmzyme_fold(fold, manifest.deepmzyme[fold], membership[fold], settings,
                                                  freeze2, freeze1)
        _merge_fold_arms(arms, fold, dm_arms)
        details["deepmzyme"][str(fold)] = dm_details
        pass_ids += dm_details["pass_ids"]
        for arm, per_fold in manifest.clean.items():
            clean_arms, clean_details = load_clean_fold(arm, fold, per_fold[fold], membership[fold], settings,
                                                        freeze2)
            _merge_fold_arms(arms, fold, clean_arms)
            details["clean"][arm][str(fold)] = clean_details
    _require(len(set(pass_ids)) == len(pass_ids), "DeepMzyme test passes repeat a pass ID")
    details["freeze2_sha256"] = freeze2.sha256
    arms[BASELINE] = read_class_table(manifest.baseline, membership, BASELINE)
    if manifest.b3 is not None:
        arms[B3] = read_class_table(manifest.b3, membership, B3)
    expected_keys = {(fold, accession) for fold, accessions in membership.items() for accession in accessions}
    for name, predictions in arms.items():
        _require(set(predictions) == expected_keys, f"Arm {name} does not cover exactly the test membership")
    systems = system_definitions(sorted(arms), settings)
    missing = [name for name in REQUIRED_SYSTEMS if name not in systems]
    _require(not missing, f"Required systems are missing: {missing}")
    return LoadedInputs(membership, strata, clusters, arms, systems, details)


# ---------------------------------------------------------------- claim

def claim_predictions(manifest_path: Path | str) -> tuple[dict[str, Any], str]:
    """Write the write-once prediction claim; returns the claim and its file SHA-256.

    No label-side input is opened: an audit hook refuses any access to them for the whole claim.
    """
    manifest = load_manifest(manifest_path)
    _require_all_folds(manifest.folds)
    label_paths = manifest.label_side_paths
    _require(_normalize_path(manifest.path) not in {_normalize_path(path) for path in label_paths}
             and _inode(manifest.path) not in {_inode(path) for path in label_paths if _inode(path) is not None},
             "The manifest cannot be a label-side input")
    roots = held_out_roots()
    with label_read_guard(label_paths, roots):
        files = manifest_input_files(manifest)
        _check_input_paths(manifest, files, roots)
        report_dir = manifest.report_dir
        claim_path = report_dir / CLAIM_NAME
        for path in (claim_path, report_dir / REPORT_JSON_NAME, report_dir / REPORT_MD_NAME):
            _require(not os.path.lexists(path), f"{path} already exists; a claim is written once per report folder")
        freeze2 = read_freeze2_manifest(manifest.freeze2)
        _require(not os.path.lexists(freeze2.scoring_ledger),
                 f"{freeze2.scoring_ledger} exists: this Freeze 2 was already scored once")
        code = code_sha256()
        check_freeze2_code(freeze2, code)
        hashes = hash_input_files(files)
        copies = sorted(entry["path"] for entry in hashes if entry["sha256"] in manifest.label_side_sha256s)
        _require(not copies, f"A prediction input has the content of a label-side input: {copies}")
        loaded = load_label_free_inputs(manifest, freeze2)
        _verify_hashed_inputs(hashes, "while the claim read them")
        system_inputs = {DM_FINAL: [entry["path"] for entry in hashes if entry["system"] == DM_FINAL]}
        for arm in manifest.clean:
            exports = [entry["path"] for entry in hashes if entry["system"] == arm]
            system_inputs[f"{arm}_seed_mean"] = exports
            system_inputs[f"{arm}_ensemble"] = exports
        claim = {
            "schema_version": SCHEMA_VERSION,
            "kind": CLAIM_KIND,
            "created_utc": _utc_now(),
            "manifest": {"path": str(manifest.path), "sha256": manifest.sha256},
            "freeze2_manifest": {"path": str(freeze2.path), "sha256": freeze2.sha256,
                                 "test_pass_output_root": str(freeze2.test_pass_output_root),
                                 "scorer_code_pinned": True},
            "scoring_ledger": str(freeze2.scoring_ledger),
            "report_dir": str(report_dir),
            "folds": list(manifest.folds),
            "settings": manifest.settings,
            "settings_frozen": settings_are_frozen(manifest.settings),
            "label_side_inputs": {
                "label_table": {"path": str(manifest.label_table.path), "declared_sha256": manifest.label_table.sha256},
                "ec_class_transfers": None if manifest.ec_class_transfers is None else {
                    "path": str(manifest.ec_class_transfers.path),
                    "declared_sha256": manifest.ec_class_transfers.sha256},
                "opened_by_claim": False,
                "guard": "audit hook refusing open, listing and scan events on these paths, and on other names of "
                         "the same files, for the whole claim",
            },
            "held_out_roots_refused": [str(root) for root in roots],
            "inputs": hashes,
            "system_inputs": system_inputs,
            "systems": {name: list(arms) for name, arms in loaded.systems.items()},
            "coverage": loaded.coverage(),
            "details": loaded.details,
            "code_sha256": code,
            "labels_read": False,
        }
        report_dir.mkdir(parents=True, exist_ok=True)
        claim_sha256 = _write_text_once(claim_path, _json_text(claim))
    return claim, claim_sha256


# ---------------------------------------------------------------- labels and rows

@dataclass(frozen=True)
class ScoredRow:
    fold: int
    accession: str
    true_index: int
    weight: float
    cluster: str
    identity_bin: str
    b1_nh30: bool
    b1_nh30_seq: bool
    b1_d2: bool
    ec_class_transferred: bool = False


@dataclass(frozen=True)
class ScoringTable:
    """Rows of B1-all with every arm's prediction aligned to them."""

    rows: tuple[ScoredRow, ...]
    arms: Mapping[str, tuple[Prediction, ...]]
    systems: Mapping[str, tuple[str, ...]]
    folds: tuple[int, ...] = ALL_FOLDS  # the scored folds; a decision needs all five


def parse_label_table(data: bytes, what: str) -> dict[str, str]:
    """Accession -> native EC string list from the label table's bytes."""
    reader = csv.DictReader(io.StringIO(data.decode("utf-8")), delimiter="\t")
    header = tuple(reader.fieldnames or ())
    _require(set(LABEL_TABLE_COLUMNS) <= set(header),
             f"{what} needs the tab-separated columns {list(LABEL_TABLE_COLUMNS)}")
    labels: dict[str, str] = {}
    for row in reader:
        accession = (row.get("accession") or "").strip()
        value = (row.get("ec_numbers") or "").strip()
        _accession(accession, what)
        _require(labels.get(accession, value) == value, f"{what} gives {accession} two different EC annotations")
        labels[accession] = value
    _require(bool(labels), f"{what} holds no label")
    return labels


def parse_transfer_list(data: bytes, what: str) -> tuple[frozenset[str], int]:
    """Listed source EC numbers whose transfer crosses an EC class, and the count of ignored rows."""
    crossing, ignored = set(), 0
    for row in _read_csv_text(data.decode("utf-8"), TRANSFER_COLUMNS, what):
        source = row["ec_number"]
        targets = [entry for entry in normalize_ec_number_list(row["transferred_to"]).split(";") if entry]
        _require(classify_ec_class([source]).status == STATUS_OK and ";" not in normalize_ec_number_list(source),
                 f"{what}: {source!r} is not one EC number with an EC class")
        _require(bool(targets) and all(classify_ec_class([target]).status == STATUS_OK for target in targets),
                 f"{what}: the targets of {source} are not EC numbers with an EC class")
        if {target.split(".")[0] for target in targets} == {source.split(".")[0]}:
            ignored += 1
        else:
            crossing.add(source)
    return frozenset(crossing), ignored


def build_scoring_table(loaded: LoadedInputs, labels: Mapping[str, str], d2_flag: str,
                        transfers: frozenset[str] = frozenset(), n_transfer_rows_ignored: int = 0,
                        ) -> tuple[ScoringTable, dict[str, Any]]:
    """B1-all rows (one EC class each) with strata flags and aligned arm predictions."""
    _require(d2_flag in D2_FLAGS, f"d2_flag must be one of {D2_FLAGS}")
    keys, excluded = [], []
    for fold in sorted(loaded.membership):
        for accession in loaded.membership[fold]:
            _require(accession in labels, f"The label table has no row for fold-{fold} test protein {accession}")
            result = classify_ec_class(labels[accession].split(";"))
            if result.status == STATUS_AMBIGUOUS:
                excluded.append({"fold": fold, "accession": accession, "reason": STATUS_AMBIGUOUS})
                continue
            _require(result.status == STATUS_OK, f"Test protein {accession} has an unusable EC annotation "
                                                 f"({result.status}); the scoring pass stops")
            keys.append((fold, accession, result.token,
                         bool(transfers & set(normalize_ec_number_list(labels[accession]).split(";")))))
    _require(bool(keys), "No test protein has one EC class")
    occurrences: dict[str, int] = {}
    for _, accession, _, _ in keys:
        occurrences[accession] = occurrences.get(accession, 0) + 1
    rows = []
    for fold, accession, token, transferred in keys:
        stratum = loaded.strata[fold][accession]
        rows.append(ScoredRow(
            fold=fold, accession=accession, true_index=EC_CLASS_TOKEN_TO_INDEX[token],
            weight=1.0 / occurrences[accession], cluster=stratum["cluster_id"], identity_bin=stratum["identity_bin"],
            b1_nh30=stratum["no_d3_hit_no_shared_donor"] == "1", b1_nh30_seq=stratum["no_d3_hit"] == "1",
            b1_d2=stratum[d2_flag] == "0", ec_class_transferred=transferred))
    arms = {name: tuple(predictions[(row.fold, row.accession)] for row in rows)
            for name, predictions in loaded.arms.items()}
    summary = {"n_membership_rows": sum(len(v) for v in loaded.membership.values()), "n_rows": len(rows),
               "excluded": excluded, "n_excluded_ambiguous_ec_class": len(excluded),
               "n_repeated_accessions": sum(count > 1 for count in occurrences.values()),
               "n_ec_class_transfers_listed": len(transfers), "n_ec_class_transfer_rows_ignored": n_transfer_rows_ignored,
               "n_b1_all_rows_with_ec_class_transfer": sum(row.ec_class_transferred for row in rows)}
    return ScoringTable(tuple(rows), arms, dict(loaded.systems), tuple(sorted(loaded.membership))), summary


# ---------------------------------------------------------------- analysis

class _Arrays:
    """Row arrays and per-arm count-table indices of a scoring table."""

    def __init__(self, table: ScoringTable):
        rows = table.rows
        self.n = len(rows)
        self.accessions = [row.accession for row in rows]
        self.true = np.array([row.true_index for row in rows], dtype=np.int64)
        self.weights = np.array([row.weight for row in rows], dtype=np.float64)
        self.cluster_of: dict[str, str] = {}
        for row in rows:
            _require(self.cluster_of.setdefault(row.accession, row.cluster) == row.cluster,
                     f"{row.accession} has two clusters")
        self.fold = np.array([row.fold for row in rows], dtype=np.int64)
        self.masks = population_masks(rows)
        self.statuses: dict[str, tuple[str, ...]] = {}
        self.predicted: dict[str, np.ndarray] = {}
        self.predicted_for_macro_f1: dict[str, np.ndarray] = {}
        self.abstained: dict[str, np.ndarray] = {}
        placed = np.where(self.true == N_EC_CLASSES - 1, ABSTENTION_COLUMN_FOR_CLASS_7,
                          ABSTENTION_COLUMN_FOR_CLASSES_1_6)
        for arm, predictions in table.arms.items():
            _require(len(predictions) == self.n, f"Arm {arm} is not aligned to the rows")
            abstained = np.array([item.token is None for item in predictions], dtype=bool)
            tokens = np.array([EC_CLASS_TOKEN_TO_INDEX[item.token] if item.token is not None else -1
                               for item in predictions], dtype=np.int64)
            self.predicted[arm] = np.where(abstained, placed, tokens)
            self.predicted_for_macro_f1[arm] = np.where(abstained, ABSTENTION_COLUMN_FOR_MACRO_F1, tokens)
            self.abstained[arm] = abstained
            self.statuses[arm] = tuple(item.status for item in predictions)

    def predictions_for(self, metric: str) -> dict[str, np.ndarray]:
        """Count-table predictions for ``metric``: errors never add a false prediction to an averaged class."""
        return self.predicted_for_macro_f1 if metric == stats.METRIC_EC_CLASSES_1_6_MACRO_F1 else self.predicted


def population_masks(rows: Sequence[ScoredRow]) -> dict[str, np.ndarray]:
    """B1-all, B1-NH30, B1-NH30-seq, B1-D2 and one mask per identity bin."""
    masks = {
        POP_ALL: np.ones(len(rows), dtype=bool),
        POP_NH30: np.array([row.b1_nh30 for row in rows], dtype=bool),
        POP_NH30_SEQ: np.array([row.b1_nh30_seq for row in rows], dtype=bool),
        POP_D2: np.array([row.b1_d2 for row in rows], dtype=bool),
    }
    for label in IDENTITY_BIN_LABELS:
        masks[f"B1-bins {label}"] = np.array([row.identity_bin == label for row in rows], dtype=bool)
    masks["high_similarity"] = np.array([row.identity_bin in HIGH_SIMILARITY_BINS for row in rows], dtype=bool)
    masks["ec_class_transferred"] = np.array([row.ec_class_transferred for row in rows], dtype=bool)
    return masks


def _contrasts(systems: Mapping[str, Sequence[str]], extra: bool = True) -> list[stats.Contrast]:
    pairs = [(PRIMARY_COMPONENTS[0], DM_FINAL, R1_SEED_MEAN), (PRIMARY_COMPONENTS[1], DM_FINAL, R1_ENSEMBLE)]
    if extra:
        pairs += [(f"{DM_FINAL}_minus_{R2_SEED_MEAN}", DM_FINAL, R2_SEED_MEAN),
                  (f"{DM_FINAL}_minus_{R2_ENSEMBLE}", DM_FINAL, R2_ENSEMBLE)]
        pairs += [item for item in DESCRIPTIVE_CONTRASTS if item[1] in systems and item[2] in systems]
    return [stats.Contrast.difference(name, minuend, subtrahend) for name, minuend, subtrahend in pairs]


def _cluster_class_support(arrays: _Arrays, mask: np.ndarray) -> np.ndarray:
    """Weighted true-class support per cluster of the population, as the bootstrap table builds it."""
    index = np.flatnonzero(mask)
    clusters = sorted({arrays.cluster_of[arrays.accessions[i]] for i in index})
    position = {cluster: k for k, cluster in enumerate(clusters)}
    rows = np.array([position[arrays.cluster_of[arrays.accessions[i]]] for i in index], dtype=np.int64)
    support = np.zeros((len(clusters), N_EC_CLASSES))
    np.add.at(support, (rows, arrays.true[index]), arrays.weights[index])
    return support


def stratum_support(arrays: _Arrays, mask: np.ndarray) -> dict[str, Any]:
    """Class and cluster support of a population and its bootstrap strata (analysis amendment 2026-10-30)."""
    support = _cluster_class_support(arrays, mask)
    single = stats.single_cluster_primary_strata(support)
    sizes = stats.stratum_sizes(support)
    return {"clusters_per_ec_class": dict(zip(EC_CLASS_TOKENS, (support > 0).sum(axis=0).astype(int).tolist())),
            "stratum_sizes": sizes,
            "single_cluster_strata": list(single),
            "superiority_inference_available": not single,
            "single_cluster_class_7_only_stratum": sizes.get(stats.CLASS_7_ONLY_STRATUM) == 1}


def _contrast_systems(systems: Mapping[str, Sequence[str]],
                      contrasts: Sequence[stats.Contrast]) -> dict[str, tuple[str, ...]]:
    """The systems that ``contrasts`` compare."""
    names = {system for contrast in contrasts for _, system, _ in contrast.terms}
    return {name: tuple(systems[name]) for name in sorted(names) if name in systems}


def _stratum_gate(arrays: _Arrays, mask: np.ndarray, metric: str = stats.METRIC_EC_CLASSES_1_6_BALANCED_ACCURACY,
                  systems: Optional[Mapping[str, Sequence[str]]] = None) -> Optional[str]:
    """The reason superiority inference on ``metric`` is unavailable on a population, or None.

    Balanced accuracy over EC classes 1-6 reads the class support only (the class-7-only stratum is
    exempt). Accuracy and macro-F1 are judged on the count table of the compared ``systems``.
    """
    if metric == stats.METRIC_EC_CLASSES_1_6_BALANCED_ACCURACY:
        single = stats.single_cluster_primary_strata(_cluster_class_support(arrays, mask))
    else:
        _require(bool(systems), f"The stratum gate for {metric} needs the compared systems")
        single = stats.single_cluster_strata(_count_table(arrays, mask, systems, metric), metric)
    if not single:
        return None
    if stats.CLASS_7_ONLY_STRATUM in single:
        return METRIC_STRATUM_GATE_REASON.format(metric=metric, strata=", ".join(single))
    return STRATUM_GATE_REASON.format(strata=", ".join(single))


def _data_problem(arrays: _Arrays, mask: np.ndarray, subsets: Optional[Mapping[str, np.ndarray]]) -> Optional[str]:
    if not mask.any():
        return "the population is empty"
    for name, subset in (subsets or {stats.DEFAULT_SUBSET: mask}).items():
        if not subset[mask].any():
            return f"subset {name} is empty"
        if not np.isin(arrays.true[mask & subset], PRIMARY_CLASS_INDICES).any():
            return f"subset {name} holds no protein of EC classes 1-6"
    return None


def _bootstrap_problem(arrays: _Arrays, mask: np.ndarray, subsets: Optional[Mapping[str, np.ndarray]], metric: str,
                       systems: Mapping[str, Sequence[str]], contrasts: Sequence[stats.Contrast]) -> Optional[str]:
    problem = _data_problem(arrays, mask, subsets)
    if problem is not None:
        return problem
    return _stratum_gate(arrays, mask, metric, _contrast_systems(systems, contrasts))


def _count_table(arrays: _Arrays, mask: np.ndarray, systems: Mapping[str, Sequence[str]], metric: str,
                 subsets: Optional[Mapping[str, np.ndarray]] = None) -> stats.ClusterCountTable:
    index = np.flatnonzero(mask)
    arms = sorted({arm for name in systems for arm in systems[name]})
    predicted = arrays.predictions_for(metric)
    return stats.build_cluster_count_table(
        [arrays.accessions[i] for i in index], arrays.true[index], {arm: predicted[arm][index] for arm in arms},
        arrays.cluster_of, weights=arrays.weights[index],
        subsets=None if subsets is None else {name: subset[index] for name, subset in subsets.items()})


def _point_contrasts(arrays: _Arrays, mask: np.ndarray, systems: Mapping[str, Sequence[str]],
                     contrasts: Sequence[stats.Contrast], metric: str) -> dict[str, float]:
    """Contrast estimates with no draws, for a population reported descriptively."""
    points = stats.point_contrasts(_count_table(arrays, mask, systems, metric), contrasts,
                                   systems={name: tuple(arms_of) for name, arms_of in systems.items()}, metric=metric)
    return {name: float(_snap(np.array(value))) for name, value in points.items()}


def _run_bootstrap(arrays: _Arrays, mask: np.ndarray, systems: Mapping[str, Sequence[str]],
                   contrasts: Sequence[stats.Contrast], metric: str, settings: Mapping[str, Any],
                   subsets: Optional[Mapping[str, np.ndarray]] = None) -> stats.BootstrapResult:
    problem = _bootstrap_problem(arrays, mask, subsets, metric, systems, contrasts)
    _require(problem is None, f"Bootstrap not possible: {problem}")
    table = _count_table(arrays, mask, systems, metric, subsets)
    result = stats.paired_cluster_bootstrap(
        table, contrasts, systems={name: tuple(arms_of) for name, arms_of in systems.items()}, metric=metric,
        n_draws=settings["n_draws"], seed=settings["bootstrap_seed"], max_skip_fraction=settings["max_skip_fraction"])
    return snap_contrasts(result)


def _snap(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.where(np.abs(values) < CONTRAST_ZERO_TOLERANCE, 0.0, values)


def snap_contrasts(result: stats.BootstrapResult) -> stats.BootstrapResult:
    """The result with contrast points and draws within ``CONTRAST_ZERO_TOLERANCE`` of zero set to zero."""
    return replace(result, contrast_point=_snap(result.contrast_point), contrast_draws=_snap(result.contrast_draws))


def _try_bootstrap(arrays: _Arrays, mask: np.ndarray, systems: Mapping[str, Sequence[str]],
                   contrasts: Sequence[stats.Contrast], metric: str, settings: Mapping[str, Any],
                   subsets: Optional[Mapping[str, np.ndarray]] = None
                   ) -> tuple[Optional[stats.BootstrapResult], Optional[str]]:
    problem = _bootstrap_problem(arrays, mask, subsets, metric, systems, contrasts)
    if problem is not None:
        return None, problem
    try:
        return _run_bootstrap(arrays, mask, systems, contrasts, metric, settings, subsets), None
    except stats.BootstrapAbort as exc:
        return None, f"bootstrap aborted: {exc}"


def _systems_for(systems: Mapping[str, Sequence[str]], names: Sequence[str]) -> dict[str, tuple[str, ...]]:
    return {name: tuple(systems[name]) for name in names if name in systems}


def _iu_record(result: Optional[stats.BootstrapResult], components: Sequence[str], alpha: float, name: str,
               problem: Optional[str] = None) -> dict[str, Any]:
    if result is None:
        return {"name": name, "alpha": alpha, "available": False, "reason": problem, "passed": False}
    return {**stats.intersection_union_test(result, components, alpha, name=name).to_record(), "available": True}


def _point_metrics_for_arm(arrays: _Arrays, arm: str, mask: np.ndarray) -> dict[str, Any]:
    true, predicted, weights = arrays.true[mask], arrays.predicted[arm][mask], arrays.weights[mask]
    metrics = ec_class_metrics(true, predicted, weights=weights)
    # Macro-F1 over classes 1-6 from the placement the bootstrap uses for it (errors in the class-7 column).
    metrics["ec_classes_1_6_macro_f1"] = ec_class_metrics(
        true, arrays.predicted_for_macro_f1[arm][mask], weights=weights)["ec_classes_1_6_macro_f1"]
    abstained = arrays.abstained[arm][mask]
    confusion = np.zeros((N_EC_CLASSES, N_EC_CLASSES))
    np.add.at(confusion, (true[~abstained], predicted[~abstained]), weights[~abstained])
    abstentions = np.bincount(true[abstained], weights=weights[abstained], minlength=N_EC_CLASSES)
    # Per-class precision counts real predictions only; recall keeps every protein of the class.
    support = np.asarray(metrics["per_class_support"], dtype=np.float64)
    correct, real_predicted = np.diag(confusion), confusion.sum(axis=0)
    precision = np.divide(correct, real_predicted, out=np.zeros(N_EC_CLASSES), where=real_predicted > 0)
    recall = np.divide(correct, support, out=np.zeros(N_EC_CLASSES), where=support > 0)
    f1 = np.divide(2.0 * precision * recall, precision + recall, out=np.zeros(N_EC_CLASSES),
                   where=(precision + recall) > 0)
    statuses = [status for status, keep in zip(arrays.statuses[arm], mask) if keep]
    metrics.pop("confusion_matrix")
    metrics.pop("seven_class_macro_f1")
    metrics.update({
        "per_class_precision": precision.tolist(),
        "per_class_f1": f1.tolist(),
        "confusion_matrix_predictions_only": confusion.tolist(),
        "abstentions_by_true_class": abstentions.tolist(),
        "n_class_7_abstentions": float(abstentions[N_EC_CLASSES - 1]),
        "status_counts": {status: statuses.count(status) for status in PREDICTION_STATUSES},
    })
    return metrics


_SUMMARY_METRICS = ("accuracy", "ec_classes_1_6_balanced_accuracy", "ec_classes_1_6_macro_f1",
                    "seven_class_balanced_accuracy")


def system_point_metrics(arrays: _Arrays, systems: Mapping[str, Sequence[str]], mask: np.ndarray) -> dict[str, Any]:
    """Point metrics per system; a seed-mean system averages its arms (each arm also listed)."""
    if not mask.any():
        return {}
    records = {}
    for name, arms in systems.items():
        per_arm = {arm: _point_metrics_for_arm(arrays, arm, mask) for arm in arms}
        record: dict[str, Any] = {metric: float(np.mean([item[metric] for item in per_arm.values()]))
                                  for metric in _SUMMARY_METRICS}
        first = next(iter(per_arm.values()))
        record.update({key: first[key] for key in ("n_examples", "total_weight", "ec_classes_1_6_n_present",
                                                   "ec_classes_1_6_label", "seven_class_n_present",
                                                   "seven_class_full_support", "per_class_support")})
        record["n_arms"] = len(arms)
        record["per_class_recall_arm_mean"] = np.mean([item["per_class_recall"] for item in per_arm.values()],
                                                      axis=0).tolist()
        record["arms"] = None if len(arms) == 1 else {
            arm: {key: item[key] for key in (*_SUMMARY_METRICS, "per_class_recall", "confusion_matrix_predictions_only",
                                             "abstentions_by_true_class", "n_class_7_abstentions", "status_counts")}
            for arm, item in per_arm.items()}
        if len(arms) == 1:
            record.update({key: first[key] for key in ("per_class_recall", "per_class_precision", "per_class_f1",
                                                       "confusion_matrix_predictions_only",
                                                       "abstentions_by_true_class", "n_class_7_abstentions",
                                                       "status_counts")})
        records[name] = record
    return records


def population_summary(arrays: _Arrays, mask: np.ndarray) -> dict[str, Any]:
    support = np.bincount(arrays.true[mask], weights=arrays.weights[mask], minlength=N_EC_CLASSES)
    present = [index + 1 for index in PRIMARY_CLASS_INDICES if support[index] > 0]
    return {"n_rows": int(mask.sum()), "n_proteins": len({arrays.accessions[i] for i in np.flatnonzero(mask)}),
            "total_weight": float(arrays.weights[mask].sum()),
            "n_clusters": len({arrays.cluster_of[arrays.accessions[i]] for i in np.flatnonzero(mask)}),
            "class_support": dict(zip(EC_CLASS_TOKENS, support.tolist())),
            "ec_classes_1_6_present": present, "primary_metric_label": f"{len(present)}-class BA",
            "ec_classes_1_6_absent": [token for token in SIX_CLASSES if token not in present],
            "six_class_endpoint_evaluated": tuple(present) == SIX_CLASSES,
            **stratum_support(arrays, mask)}


def _class_list(classes: Sequence[int]) -> str:
    items = [str(item) for item in classes]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def class_scope(present: Sequence[int]) -> Optional[str]:
    """The scope clause of a decision whose population lacks some of EC classes 1-6, else None."""
    present = tuple(present)
    if present == SIX_CLASSES:
        return None
    absent = [token for token in SIX_CLASSES if token not in present]
    plural = len(absent) > 1
    return (f"balanced accuracy over EC class{'es' if len(present) > 1 else ''} {_class_list(present)} only; the "
            f"six-class endpoint over EC classes 1-6 was not evaluated because EC class{'es' if plural else ''} "
            f"{_class_list(absent)} {'are' if plural else 'is'} absent from this population")


def holm_fixed_family(hypotheses: Sequence[tuple[str, Optional[stats.BootstrapResult], Sequence[str], Optional[str]]],
                      family_alpha: float) -> list[dict[str, Any]]:
    """Holm step-down over a fixed family; each hypothesis is an intersection-union test of its components.

    A hypothesis without a bootstrap result is untestable: it is ordered with p = 1 and never passes.
    Equal p-values keep the given order. The j-th hypothesis (0-based) is tested at
    ``family_alpha / (m - j)`` only while every earlier one passed.
    """
    p_values = [stats.intersection_union_p_value(result, components) if result is not None else 1.0
                for _, result, components, _ in hypotheses]
    order = sorted(range(len(hypotheses)), key=lambda index: p_values[index])
    m, still_testing, steps = len(hypotheses), True, []
    for rank, index in enumerate(order):
        name, result, components, problem = hypotheses[index]
        alpha = family_alpha / (m - rank)
        tested = still_testing and result is not None
        test = stats.intersection_union_test(result, components, alpha, name=name) if tested else None
        passed = bool(test is not None and test.passed)
        steps.append({"name": name, "rank": rank + 1, "alpha": alpha, "p_value": p_values[index], "tested": tested,
                      "passed": passed, "available": result is not None, "reason": problem,
                      "test": None if test is None else test.to_record()})
        still_testing = passed
    return steps


def decision_wording(name: str, passed: bool, dependent_classes: Sequence[str] = (),
                     template_dependent: bool = False, folds: Sequence[int] = ALL_FOLDS, *,
                     available: bool = True, classes_present: Sequence[int] = SIX_CLASSES) -> str:
    """The design's decision sentence for co-primary ``name`` ("P1" or "P2"); none for a fold subset.

    A population lacking some of EC classes 1-6 gets a scope clause naming the classes evaluated and
    stating that the six-class endpoint was not evaluated, whatever the outcome.
    """
    if tuple(folds) != ALL_FOLDS:
        return DECISION_NO_DECISION.format(folds=list(folds))
    scope = class_scope(classes_present)
    if not available or not passed:
        text = (DECISION_UNAVAILABLE if not available else DECISION_NOT_SHOWN)[name]
        return text + (f" ({scope})" if scope else "")
    qualifiers = [scope] if scope else []
    if dependent_classes:
        qualifiers.append("dependent on EC class " + " and EC class ".join(sorted(dependent_classes)))
    if template_dependent:
        qualifiers.append("template-dependent: the sequence-only population B1-NH30-seq does not support it")
    return DECISION_PASS[name] + (f" ({'; '.join(qualifiers)})" if qualifiers else "")


def analyse_comparison(table: ScoringTable, settings: Mapping[str, Any]) -> dict[str, Any]:
    """Every pre-declared result for one scoring table (pure; no file access)."""
    arrays = _Arrays(table)
    systems = dict(table.systems)
    missing = [name for name in REQUIRED_SYSTEMS if name not in systems]
    _require(not missing, f"Required systems are missing: {missing}")
    masks = arrays.masks
    contrasts = _contrasts(systems)
    primary_contrasts = _contrasts(systems, extra=False)

    # Co-primary contrasts (a skip-limit abort stops the run). A population failing the stratum gate
    # gets no bootstrap; it stays in the two-hypothesis Holm family as unavailable.
    ba: dict[str, Optional[stats.BootstrapResult]] = {}
    gate: dict[str, Optional[str]] = {}
    for name, population in CO_PRIMARY_POPULATIONS.items():
        gate[population] = _stratum_gate(arrays, masks[population]) if masks[population].any() else None
        if gate[population] is not None:
            ba[population] = None
            continue
        try:
            ba[population] = _run_bootstrap(arrays, masks[population], systems, contrasts,
                                            stats.METRIC_EC_CLASSES_1_6_BALANCED_ACCURACY, settings)
        except stats.BootstrapAbort as exc:
            raise ScoringError(f"Co-primary {name} ({population}): {exc} "
                               "The pre-declared rule stops the run.") from None
    co_primary = stats.co_primary_holm(("P1", ba[POP_ALL], PRIMARY_COMPONENTS),
                                       ("P2", ba[POP_NH30], PRIMARY_COMPONENTS))
    summaries = {population: population_summary(arrays, masks[population])
                 for population in (POP_ALL, POP_NH30, POP_NH30_SEQ, POP_D2)}
    estimates = {name: ({component: ba[population].estimate(component) for component in PRIMARY_COMPONENTS}
                        if ba[population] is not None else
                        _point_contrasts(arrays, masks[population], systems, primary_contrasts,
                                         stats.METRIC_EC_CLASSES_1_6_BALANCED_ACCURACY))
                 for name, population in CO_PRIMARY_POPULATIONS.items()}

    # Rare-class dependence at each claim's own Holm level; not run for an unavailable claim, since
    # removing a class may not be used to recover eligibility.
    dependence: dict[str, Any] = {}
    dependent: dict[str, list[str]] = {}
    for name, population in CO_PRIMARY_POPULATIONS.items():
        step = co_primary.step(name)
        dependence[name] = {"alpha": step.alpha, "claim_passed": step.passed, "claim_available": step.available,
                            "removed": {}}
        dependent[name] = []
        if not step.available:
            dependence[name]["reason"] = "not run: " + str(gate[population])
            dependence[name]["dependent_on_ec_classes"] = []
            continue
        for token in RARE_CLASS_TOKENS:
            index = EC_CLASS_TOKEN_TO_INDEX[token]
            present = bool((arrays.true[masks[population]] == index).any())
            record: dict[str, Any] = {"class_present": present}
            if present:
                result, problem = _try_bootstrap(arrays, masks[population] & (arrays.true != index), systems,
                                                 primary_contrasts, stats.METRIC_EC_CLASSES_1_6_BALANCED_ACCURACY,
                                                 settings)
                record["test"] = _iu_record(result, PRIMARY_COMPONENTS, step.alpha, f"{name} without EC class {token}",
                                            problem)
                record["bootstrap"] = None if result is None else result.to_record()
                if step.passed and not record["test"]["passed"]:
                    dependent[name].append(token)
            dependence[name]["removed"][token] = record
        dependence[name]["dependent_on_ec_classes"] = dependent[name]

    # Template check of P2 on the sequence-only population (descriptive).
    seq_result, seq_problem = _try_bootstrap(arrays, masks[POP_NH30_SEQ], systems, contrasts,
                                             stats.METRIC_EC_CLASSES_1_6_BALANCED_ACCURACY, settings)
    p2_step = co_primary.step("P2")
    seq_test = _iu_record(seq_result, PRIMARY_COMPONENTS, p2_step.alpha, "P2 on B1-NH30-seq", seq_problem)
    template_dependent = bool(p2_step.passed and not seq_test["passed"])
    # The reverse difference: B1-NH30-seq (which keeps donor-sharing proteins) passes where P2 does not.
    sequence_only_passes_alone = bool(p2_step.available and not p2_step.passed and seq_test["passed"])

    decisions = {name: decision_wording(name, co_primary.passed(name), dependent[name],
                                        template_dependent if name == "P2" else False, table.folds,
                                        available=co_primary.step(name).available,
                                        classes_present=summaries[population]["ec_classes_1_6_present"])
                 for name, population in CO_PRIMARY_POPULATIONS.items()}
    endpoint_scope = {name: {"population": population,
                             "ec_classes_evaluated": summaries[population]["ec_classes_1_6_present"],
                             "ec_classes_absent": summaries[population]["ec_classes_1_6_absent"],
                             "six_class_endpoint_evaluated": summaries[population]["six_class_endpoint_evaluated"],
                             "superiority_inference_available": ba[population] is not None,
                             "reason": gate[population]}
                      for name, population in CO_PRIMARY_POPULATIONS.items()}

    # Secondary family (an unavailable population's hypotheses stay in the family, untestable). The
    # stratum gate is metric-aware: the class-7-only exemption covers balanced accuracy only. A gated
    # hypothesis keeps its contrast point estimates.
    hypotheses = []
    gated_points: dict[str, dict[str, float]] = {}
    r2_components = R2_COMPONENTS[settings["r2_form"]]
    for name, population in CO_PRIMARY_POPULATIONS.items():
        hypothesis = f"{name} versus R2 ({settings['r2_form']})"
        if gate[population] is not None:
            points = _point_contrasts(arrays, masks[population], systems, contrasts,
                                      stats.METRIC_EC_CLASSES_1_6_BALANCED_ACCURACY)
            gated_points[hypothesis] = {component: points[component] for component in r2_components}
        hypotheses.append((hypothesis, ba[population], r2_components, gate[population]))
    other_metric_runs = {}
    for metric in (stats.METRIC_ACCURACY, stats.METRIC_EC_CLASSES_1_6_MACRO_F1):
        for name, population in CO_PRIMARY_POPULATIONS.items():
            hypothesis = f"{name} {metric} versus R1"
            result, problem = _try_bootstrap(arrays, masks[population], systems, primary_contrasts, metric, settings)
            metric_gate = (None if _data_problem(arrays, masks[population], None) is not None else
                           _stratum_gate(arrays, masks[population], metric,
                                         _contrast_systems(systems, primary_contrasts)))
            if metric_gate is not None:
                gated_points[hypothesis] = _point_contrasts(arrays, masks[population], systems, primary_contrasts,
                                                            metric)
            other_metric_runs[f"{metric} {population}"] = result
            hypotheses.append((hypothesis, result, PRIMARY_COMPONENTS, problem))
    did_mask = masks["high_similarity"] | masks[POP_NH30]
    did_subsets = {SUBSET_HIGH_SIMILARITY: masks["high_similarity"], POP_NH30: masks[POP_NH30]}
    did_contrast = stats.Contrast(DID_CONTRAST, (
        (1.0, R1_SEED_MEAN, SUBSET_HIGH_SIMILARITY), (-1.0, R1_SEED_MEAN, POP_NH30),
        (-1.0, DM_FINAL, SUBSET_HIGH_SIMILARITY), (1.0, DM_FINAL, POP_NH30)))
    did_result, did_problem = _try_bootstrap(arrays, did_mask, _systems_for(systems, (R1_SEED_MEAN, DM_FINAL)),
                                             [did_contrast], stats.METRIC_EC_CLASSES_1_6_BALANCED_ACCURACY, settings,
                                             subsets=did_subsets)
    hypotheses.append((DID_CONTRAST, did_result, (DID_CONTRAST,), did_problem))
    secondary = holm_fixed_family(hypotheses, settings["secondary_family_alpha"])
    for step in secondary:
        step["point_estimates"] = gated_points.get(step["name"])

    # Descriptive results.
    descriptive_alpha = settings["descriptive_alpha"]
    sensitivity = {}
    for population in (POP_NH30_SEQ, POP_D2):
        result, problem = (seq_result, seq_problem) if population == POP_NH30_SEQ else _try_bootstrap(
            arrays, masks[population], systems, contrasts, stats.METRIC_EC_CLASSES_1_6_BALANCED_ACCURACY, settings)
        sensitivity[population] = {
            "population": population_summary(arrays, masks[population]),
            "intersection_union_versus_r1": _iu_record(result, PRIMARY_COMPONENTS, descriptive_alpha,
                                                       f"{population} versus R1", problem),
            "contrasts": _descriptive_contrasts(result, descriptive_alpha),
            "bootstrap": None if result is None else result.to_record(),
            "points": system_point_metrics(arrays, systems, masks[population]),
        }
    bins = {label: {"population": population_summary(arrays, masks[f"B1-bins {label}"]),
                    "points": system_point_metrics(arrays, systems, masks[f"B1-bins {label}"])}
            for label in IDENTITY_BIN_LABELS}
    per_fold = {}
    for fold in sorted(set(arrays.fold.tolist())):
        in_fold = arrays.fold == fold
        per_fold[str(fold)] = {
            population: {"population": population_summary(arrays, masks[population] & in_fold),
                         "points": system_point_metrics(arrays, systems, masks[population] & in_fold)}
            for population in CO_PRIMARY_POPULATIONS.values()}
    transfers = None
    if masks["ec_class_transferred"].any():
        transfers = {}
        for name, population in CO_PRIMARY_POPULATIONS.items():
            mask = masks[population] & ~masks["ec_class_transferred"]
            if gate[population] is not None:  # removing proteins may not recover eligibility
                result, problem = None, "not run: " + gate[population]
            else:
                result, problem = _try_bootstrap(arrays, mask, systems, primary_contrasts,
                                                 stats.METRIC_EC_CLASSES_1_6_BALANCED_ACCURACY, settings)
            transfers[name] = {"population": population_summary(arrays, mask),
                               "test": _iu_record(result, PRIMARY_COMPONENTS, co_primary.step(name).alpha,
                                                  f"{name} without EC-class transfers", problem),
                               "points": system_point_metrics(arrays, systems, mask)}
    class_7 = []
    for i in np.flatnonzero(arrays.true == N_EC_CLASSES - 1):
        row = table.rows[i]
        class_7.append({"fold": row.fold, "accession": row.accession, "b1_nh30": row.b1_nh30,
                        "identity_bin": row.identity_bin,
                        "outcomes": {arm: (table.arms[arm][i].token or table.arms[arm][i].status)
                                     for arm in sorted(table.arms)}})

    return {
        "populations": summaries,
        "co_primary": {
            **co_primary.to_record(),
            "components": list(PRIMARY_COMPONENTS),
            "estimates": estimates,
            "population_of": dict(CO_PRIMARY_POPULATIONS),
        },
        "folds_scored": list(table.folds),
        "decisions": decisions,
        "endpoint_scope": endpoint_scope,
        "rare_class_dependence": dependence,
        "template_check": {"test": seq_test, "p2_passed": bool(p2_step.passed), "p2_available": p2_step.available,
                           "claims_differ": bool(p2_step.passed) != bool(seq_test["passed"]),
                           "template_dependent": template_dependent,
                           "sequence_only_passes_without_p2": sequence_only_passes_alone},
        "secondary_family": {"family_alpha": settings["secondary_family_alpha"], "size": len(hypotheses),
                             "r2_form": settings["r2_form"], "steps": secondary,
                             "difference_in_differences": {
                                 "definition": "(R1 seed-mean on the [30,50] and >50 bins - on B1-NH30) - "
                                               "(DeepMzyme-final, same strata); BA over the EC classes 1-6 present",
                                 "estimate": None if did_result is None else did_result.estimate(DID_CONTRAST),
                                 "bootstrap": None if did_result is None else did_result.to_record(),
                                 "reason": did_problem}},
        "points": {population: system_point_metrics(arrays, systems, masks[population])
                   for population in CO_PRIMARY_POPULATIONS.values()},
        "descriptive": {
            "alpha": descriptive_alpha,
            "contrasts": {population: _descriptive_contrasts(ba[population], descriptive_alpha) if ba[population]
                          is not None else {component: {"estimate": estimates[name][component], "lower_bound": None,
                                                        "p_value": None} for component in PRIMARY_COMPONENTS}
                          for name, population in CO_PRIMARY_POPULATIONS.items()},
            "sensitivity": sensitivity,
            "identity_bins": bins,
            "per_fold": per_fold,
            "ec_class_transfer_rescore": transfers,
            "class_7_outcomes": class_7,
            "r1_epoch_variants": "not run: the published recipe only (decision U12)",
        },
        "bootstrap": {
            **{f"{stats.METRIC_EC_CLASSES_1_6_BALANCED_ACCURACY} {population}":
               None if ba[population] is None else ba[population].to_record()
               for population in CO_PRIMARY_POPULATIONS.values()},
            **{name: None if result is None else result.to_record() for name, result in other_metric_runs.items()},
        },
    }


def _descriptive_contrasts(result: Optional[stats.BootstrapResult], alpha: float) -> Optional[dict[str, Any]]:
    if result is None:
        return None
    return {name: {"estimate": result.estimate(name), "lower_bound": result.lower_bound(name, alpha),
                   "p_value": result.p_value(name)} for name in result.contrast_names}


# ---------------------------------------------------------------- score

def _verify_claim_inputs(claim: Mapping[str, Any], when: str = "after the claim") -> None:
    _verify_hashed_inputs(claim["inputs"], when)


def score_comparison(claim_path: Path | str, claim_sha256: str) -> dict[str, Any]:
    """Verify the claim, read the labels once, score and write the write-once reports."""
    claim_path = Path(claim_path).absolute()
    expected = _check_sha256(claim_sha256, "--claim-sha256")
    _require(claim_path.is_file(), f"Claim not found: {claim_path}")
    data = claim_path.read_bytes()
    _require(_sha256_bytes(data) == expected, f"Claim {claim_path} does not hash to {expected}")
    claim = json.loads(data.decode("utf-8"))
    _require(claim.get("schema_version") == SCHEMA_VERSION and claim.get("kind") == CLAIM_KIND,
             f"{claim_path} is not a {CLAIM_KIND} v{SCHEMA_VERSION}")
    report_dir = Path(claim["report_dir"])
    _require(claim_path.resolve() == (report_dir / CLAIM_NAME).resolve(),
             f"The claim must stay at {report_dir / CLAIM_NAME}")
    for name in (REPORT_JSON_NAME, REPORT_MD_NAME):
        _require(not os.path.lexists(report_dir / name), f"{report_dir / name} already exists (one scoring pass)")
    live_code = code_sha256()
    differing = sorted(path for path, digest in claim["code_sha256"].items() if live_code.get(path) != digest)
    _require(not differing and set(live_code) == set(claim["code_sha256"]),
             f"Scorer code differs from the claim: {differing or sorted(set(live_code) ^ set(claim['code_sha256']))}")

    manifest = load_manifest(claim["manifest"]["path"])
    _require(manifest.sha256 == claim["manifest"]["sha256"] and manifest.report_dir == report_dir
             and manifest.settings == claim["settings"], "The manifest no longer matches the claim")
    _require_all_folds(manifest.folds)
    _require(manifest.label_table.sha256 == claim["label_side_inputs"]["label_table"]["declared_sha256"],
             "The label table's declared SHA-256 differs from the claim")
    label_paths = list(manifest.label_side_paths)
    roots = held_out_roots()
    with label_read_guard(label_paths, roots):
        freeze2 = read_freeze2_manifest(manifest.freeze2)
        _require(freeze2.sha256 == claim["freeze2_manifest"]["sha256"]
                 and str(freeze2.scoring_ledger) == claim["scoring_ledger"],
                 "The Freeze-2 manifest no longer matches the claim")
        check_freeze2_code(freeze2, live_code)
        # Rebuild between two hash checks, so the predictions scored are the claimed bytes.
        _verify_claim_inputs(claim)
        loaded = load_label_free_inputs(manifest, freeze2)
        _verify_claim_inputs(claim, "while the score step read them")
        _require(_jsonable(loaded.coverage()) == claim["coverage"],
                 "Rebuilt predictions differ from the claim's coverage")
        scoring_pass = claim_scoring_pass(freeze2, claim_path, expected, manifest.label_table.sha256)

    # The only label read: each label-side file is read once, hashed and parsed from the same bytes.
    with label_read_guard([], roots), label_read_counter(label_paths) as open_counts:
        label_bytes = manifest.label_table.path.read_bytes()
        transfer_bytes = (manifest.ec_class_transfers.path.read_bytes()
                          if manifest.ec_class_transfers is not None else None)
    _require(_sha256_bytes(label_bytes) == manifest.label_table.sha256,
             f"The label table does not hash to its declared SHA-256 {manifest.label_table.sha256}")
    if transfer_bytes is not None:
        _require(_sha256_bytes(transfer_bytes) == manifest.ec_class_transfers.sha256,
                 "The EC-class transfer list does not hash to its declared SHA-256")
    labels = parse_label_table(label_bytes, manifest.label_table.path.name)
    transfers, n_ignored = (parse_transfer_list(transfer_bytes, manifest.ec_class_transfers.path.name)
                            if transfer_bytes is not None else (frozenset(), 0))
    table, row_summary = build_scoring_table(loaded, labels, manifest.settings["d2_flag"], transfers, n_ignored)
    with label_read_guard(label_paths, roots):
        results = analyse_comparison(table, manifest.settings)

    settings_frozen = settings_are_frozen(manifest.settings)
    warnings = []
    if tuple(manifest.folds) != ALL_FOLDS:
        warnings.append(f"Only folds {list(manifest.folds)} are scored; the benchmark has folds {list(ALL_FOLDS)}, "
                        "so no decision is issued.")
    if not settings_frozen:
        warnings.append("The settings differ from the frozen values (in-process test override); "
                        "this report is not a result.")
    if manifest.ec_class_transfers is None:
        warnings.append("No EC-class transfer list was supplied; the rescore without transfers was not run.")
    for name in (C_ESMC_SEED_MEAN, B3):
        if name not in table.systems:
            warnings.append(f"System {name} was not provided.")
    disclosures = [{"id": key, "text": text} for key, text in DISCLOSURES]
    disclosures.append({"id": TRANSFER_DISCLOSURE_ID,
                        "text": transfer_disclosure(row_summary, manifest.ec_class_transfers is not None)})
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "created_utc": _utc_now(),
        "claim": {"path": str(claim_path), "sha256": expected, "created_utc": claim["created_utc"]},
        "access_receipt": {
            "label_table": {"path": str(manifest.label_table.path), "sha256": manifest.label_table.sha256,
                            "open_events_during_score": open_counts.get(_normalize_path(manifest.label_table.path))},
            "ec_class_transfers": None if manifest.ec_class_transfers is None else {
                "path": str(manifest.ec_class_transfers.path), "sha256": manifest.ec_class_transfers.sha256},
            "labels_read_after_claim": True,
            "claimed_inputs_verified": len(claim["inputs"]),
            "code_sha256": live_code,
            "freeze2_manifest": {"path": str(freeze2.path), "sha256": freeze2.sha256, "scorer_code_pinned": True},
            "scoring_pass": scoring_pass,
        },
        "folds": list(manifest.folds),
        "all_five_folds": tuple(manifest.folds) == ALL_FOLDS,
        "settings": manifest.settings,
        "settings_frozen": settings_frozen,
        "warnings": warnings,
        "rules": {
            "primary_metric": "balanced accuracy over EC classes 1-6 from seven-way predictions (class 7 errors count, "
                              "class 7 is not averaged); a population missing a class averages the classes present",
            "projection": comparator.PROJECTION_RULE,
            "ensemble": comparator.ENSEMBLE_RULE,
            "deepmzyme_final": "mean logits over the refit seeds, then logits - tau * log(refit priors); arg-max with "
                               "exact ties to the lowest class",
            "errors": ABSTENTION_RULE,
            "bootstrap": "paired stratified cluster bootstrap (ec_clean30.stats); clusters are cohort components "
                         "restricted to the population; strata by the rarest EC class 1-6 of a cluster",
            "co_primary": "intersection-union over R1 seed-mean and R1-ens; Holm across P1 and P2 by p-value, "
                          "the smaller at 0.025 and the other at 0.05 only if the first passes",
            "rare_class_removal": "the proteins of the removed class leave the population; the claim's Holm level",
            "repeated_accessions": "weighted to a total of one and resampled as one cluster",
            "decision": "all five folds are required for a decision",
            "contrast_zero_tolerance": CONTRAST_ZERO_TOLERANCE,
        },
        "rows": row_summary,
        "coverage": loaded.coverage(),
        "inputs": loaded.details,
        **results,
        "disclosures": disclosures,
        "acknowledgement": ACKNOWLEDGEMENT_TEMPLATE.format(name=manifest.settings["acknowledgement_name"]),
    }
    json_text = _json_text(report)
    markdown = render_report_markdown(_jsonable(report))
    _write_text_once(report_dir / REPORT_JSON_NAME, json_text)
    _write_text_once(report_dir / REPORT_MD_NAME, markdown)
    return report


# ---------------------------------------------------------------- markdown

def _points(value: Any) -> str:
    return "n/a" if value is None else f"{POINTS * float(value):.2f}"


def _number(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def _points_table(points: Mapping[str, Any]) -> list[str]:
    lines = ["| System | BA, EC classes 1-6 present | Accuracy | Macro-F1 (1-6 present) | Seven-class BA | Classes | n |",
             "|---|---|---|---|---|---|---|"]
    for name, record in points.items():
        lines.append(f"| {name} | {_points(record['ec_classes_1_6_balanced_accuracy'])} | "
                     f"{_points(record['accuracy'])} | {_points(record['ec_classes_1_6_macro_f1'])} | "
                     f"{_points(record['seven_class_balanced_accuracy'])} | {record['ec_classes_1_6_label']} | "
                     f"{record['n_examples']} |")
    return lines


def render_report_markdown(report: Mapping[str, Any]) -> str:
    """Readable summary of ``report.json`` (values in percentage points)."""
    lines = ["# EC class versus CLEAN on the official CLEAN split30 benchmark", "",
             f"Claim SHA-256 `{report['claim']['sha256']}`; label table SHA-256 "
             f"`{report['access_receipt']['label_table']['sha256']}` (read once, after the claim). "
             f"Folds {report['folds']}.", ""]
    for warning in report["warnings"]:
        lines.append(f"- Warning: {warning}")
    lines += ["", "## Decisions", ""]
    for name, text in report["decisions"].items():
        lines.append(f"- {name} ({POPULATION_WORDING[name]}): {text}.")
    lines += ["", "## Endpoint scope and stratum support", ""]
    for name, scope in report["endpoint_scope"].items():
        summary = report["populations"][scope["population"]]
        evaluated = ", ".join(str(item) for item in scope["ec_classes_evaluated"])
        lines.append(
            f"- {name} ({scope['population']}): EC classes evaluated {evaluated}; six-class endpoint "
            + ("evaluated" if scope["six_class_endpoint_evaluated"] else
               "not evaluated (absent: " + ", ".join(str(item) for item in scope["ec_classes_absent"]) + ")")
            + f"; clusters per EC class {summary['clusters_per_ec_class']}; stratum sizes {summary['stratum_sizes']}; "
            + ("superiority inference available." if scope["superiority_inference_available"] else
               f"{scope['reason']}; reported descriptively."))
    lines += ["", "A stratum of EC classes 1-6 with a single homology cluster makes a population descriptive only. "
                  "The class-7-only stratum is exempt for balanced accuracy over EC classes 1-6 only: a single-cluster "
                  "class-7-only stratum also makes the accuracy comparison descriptive, and the macro-F1 comparison "
                  "when a compared system predicts one of its proteins as an EC class 1-6 present in the population. "
                  "Two or more clusters only avoid that degeneracy; they do not establish adequate power.", ""]
    lines += ["## Co-primary contrasts (one-sided; BA over the EC classes 1-6 present in each population)", "",
              "| Step | Contrast | Population | Classes | Available | Alpha | p | Tested | Passed | Component | "
              "Estimate | Lower bound |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    population_of = report["co_primary"]["population_of"]
    for step in report["co_primary"]["steps"]:
        label = report["populations"][population_of[step["name"]]]["primary_metric_label"]
        components = (step["test"] or {}).get("components") or [
            {"name": name, "estimate": report["co_primary"]["estimates"][step["name"]][name], "lower_bound": None}
            for name in report["co_primary"]["components"]]
        for component in components:
            lines.append(f"| {step['rank']} | {step['name']} | {population_of[step['name']]} | {label} | "
                         f"{step.get('available', True)} | {step['alpha']} | "
                         f"{_number(step['p_value'])} | {step['tested']} | {step['passed']} | {component['name']} | "
                         f"{_points(component['estimate'])} | {_points(component['lower_bound'])} |")
    lines += ["", "## Rare-class dependence", ""]
    for name, record in report["rare_class_dependence"].items():
        if not record.get("claim_available", True):
            lines.append(f"- {name}: {record['reason']}.")
            continue
        for token, removed in record["removed"].items():
            if not removed["class_present"]:
                lines.append(f"- {name} without EC class {token}: class absent from the population.")
                continue
            test = removed["test"]
            bounds = ", ".join(f"{item['name']} {_points(item['lower_bound'])}" for item in test.get("components", []))
            lines.append(f"- {name} without EC class {token} (alpha {record['alpha']}): passed {test['passed']}"
                         + (f"; lower bounds {bounds}" if bounds else f"; {test.get('reason')}") + ".")
    template = report["template_check"]
    if template["template_dependent"]:
        difference = "the claims differ: P2 passes but B1-NH30-seq does not (template-dependent)"
    elif template["sequence_only_passes_without_p2"]:
        difference = ("the claims differ: B1-NH30-seq passes but P2 does not (the sequence-only result depends on "
                      "test proteins that share a donor with training)")
    elif not template.get("p2_available", True):
        difference = "no comparison: P2 has no superiority inference"
    else:
        difference = "the claims agree"
    lines += ["", f"Template check (P2 on B1-NH30-seq at alpha {template['test']['alpha']}): P2 passed "
                  f"{template['p2_passed']}; B1-NH30-seq passed {template['test']['passed']}; {difference}.", "",
              f"## Secondary family (Holm, family alpha {report['secondary_family']['family_alpha']}, "
              f"{report['secondary_family']['size']} hypotheses)", "",
              "| Rank | Hypothesis | Alpha | p | Tested | Passed | Note |", "|---|---|---|---|---|---|---|"]
    for step in report["secondary_family"]["steps"]:
        note = step["reason"] or ""
        if step.get("point_estimates"):
            note += "; point estimates " + ", ".join(f"{name} {_points(value)}"
                                                     for name, value in step["point_estimates"].items())
        lines.append(f"| {step['rank']} | {step['name']} | {step['alpha']:.4f} | {_number(step['p_value'])} | "
                     f"{step['tested']} | {step['passed']} | {note} |")
    did = report["secondary_family"]["difference_in_differences"]
    lines += ["", f"Difference in differences: {did['definition']}; estimate {_points(did['estimate'])}.", ""]
    for population, points in report["points"].items():
        summary = report["populations"][population]
        lines += [f"## {population} (n = {summary['n_rows']}, {summary['n_clusters']} clusters, "
                  f"{summary['primary_metric_label']})", ""] + _points_table(points) + [""]
    lines += ["## Descriptive results (no claim)", ""]
    for population, record in report["descriptive"]["sensitivity"].items():
        lines += [f"### {population} (n = {record['population']['n_rows']})", ""]
        lines += _points_table(record["points"]) + [""]
    lines += ["### Identity bins", ""]
    for label, record in report["descriptive"]["identity_bins"].items():
        if record["points"]:
            lines += [f"#### Bin {label} (n = {record['population']['n_rows']})", ""]
            lines += _points_table(record["points"]) + [""]
    lines += ["### Per fold", ""]
    for fold, populations in report["descriptive"]["per_fold"].items():
        for population, record in populations.items():
            if record["points"]:
                lines += [f"#### Fold {fold}, {population} (n = {record['population']['n_rows']})", ""]
                lines += _points_table(record["points"]) + [""]
    lines += ["### Descriptive contrasts (BA over the EC classes 1-6 present)", ""]
    for population, contrasts in report["descriptive"]["contrasts"].items():
        for name, record in (contrasts or {}).items():
            lines.append(f"- {population}, {name}: {_points(record['estimate'])} (one-sided "
                         f"{1 - report['descriptive']['alpha']:.0%} lower bound {_points(record['lower_bound'])}).")
    lines += ["", f"Class-7 proteins: {len(report['descriptive']['class_7_outcomes'])} (per-protein outcomes in "
                  "report.json). R1 epoch variants: " + report["descriptive"]["r1_epoch_variants"] + ".", "",
              "## Coverage", "", f"Rows scored: {report['rows']['n_rows']} of {report['rows']['n_membership_rows']} "
              f"membership rows; excluded (several EC classes): {report['rows']['n_excluded_ambiguous_ec_class']}.", ""]
    lines += ["## Disclosures", ""] + [f"- {item['text']}" for item in report["disclosures"]]
    lines += ["", "## Acknowledgement", "", report["acknowledgement"], ""]
    return "\n".join(lines)


# ---------------------------------------------------------------- command line

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frozen scorer for the EC class versus CLEAN comparison "
                                                 "(official CLEAN split30 benchmark).")
    actions = parser.add_subparsers(dest="action", required=True)
    claim = actions.add_parser("claim", help="Hash every frozen prediction input into the write-once claim "
                                             "(never opens a label-side input).")
    claim.add_argument("--manifest", type=Path, required=True, help="Scoring input manifest (JSON).")
    score = actions.add_parser("score", help="Verify the claim, read the labels once and write the reports.")
    score.add_argument("--claim", type=Path, required=True, help=f"The claim file ({CLAIM_NAME}).")
    score.add_argument("--claim-sha256", required=True, help="The claim SHA-256 printed by the claim action.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.action == "claim":
            claim, digest = claim_predictions(args.manifest)
            summary = {"claim": str(Path(claim["report_dir"]) / CLAIM_NAME), "claim_sha256": digest,
                       "n_inputs": len(claim["inputs"]), "labels_read": False}
        else:
            report = score_comparison(args.claim, args.claim_sha256)
            summary = {"report_dir": str(Path(report["claim"]["path"]).parent), "decisions": report["decisions"]}
    except ScoringError as exc:
        raise SystemExit(f"error: {exc}") from None
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return 0
