"""Training-side sanity checks of the retrained CLEAN fits (amendment 2, rule A2-S7; numpy only).

The checks look for a broken comparator, never for a better one: a failure stops the fit from
entering any record until the maintainer has looked for defects, and the recipe, epochs, loss,
widths, learning rate and seeds never change. Records hold totals only (counts, fractions and
SHA-256 values; no accession list and no per-protein value).

In-sample check (every CLEAN fit: inner, R1, R2 and C-ESMC, all folds)
  Sample: the fit's own training proteins with a single EC class among EC classes 1-6 whose every
  full EC number has at least two training proteins, ordered by the SHA-256 of
  ``<fold>:<regime>:<inner fold or ->:<accession>``; the first ``IN_SAMPLE_N`` (all, if fewer).
  A validation-mode distance export of the fit with in-sample queries allowed (an engineering
  check, not an evaluation) is projected to EC classes with the comparator's rule. The raw
  embedding anchor projects the same queries onto EC centres that are the mean of the fit's own
  base input embeddings (ESM-1b layer 33, or the ESMC chain mean for C-ESMC) over each EC
  number's training proteins, with Euclidean distances and the same projection and tie rule.
  Pass needs all of: IS1 primary metric >= 0.80; IS2 primary metric >= the anchor's - 0.05;
  IS3 ties <= 5% of the queries; IS4 the export holds exactly the sample, all as in-sample queries.

Held-out check (fold 0: the four R1-inner and the four R2-inner fits)
  Queries: the inner-validation proteins whose full EC numbers all occur among the R1-inner
  training centres of their inner fold, with one EC class among EC classes 1-6. Pooled per arm,
  pass needs all of: HO1 primary metric >= the raw anchor's (same training tables, same queries)
  - 0.10; HO2 (R1-inner only) primary metric >= the accuracy of predicting the training table's
  most frequent EC class + 0.10; HO3 each fit's ties <= 5%; HO4 each fit's primary metric >= its
  own anchor's - 0.20. In-sample exports are refused.

The primary metric here is the accuracy over the proteins of EC classes 1-6 (a tie is wrong).
A record whose check fails has status ``comparator_diagnosis_required``. A disposition, written
by the maintainer after no defect was found, says ``proceed_disclosed`` (the fit is scored as
trained and the report says so) or ``stop``. The thresholds are fixed here and are not changed
after any record exists; loss trends are not used, and published CLEAN values never gate.

The held-out record carries a status per arm (``arm_status``: R1-inner and R2-inner); a failed arm
takes its own disposition, which names the arm. The consequence of a ``proceed_disclosed``
held-out arm is fixed by ``held_out_consequence`` (rule ``HELD_OUT_CONSEQUENCE_RULE``, choice C6b):
R1-inner makes both co-primaries and every secondary hypothesis against R1 unavailable (R1-inner
stands for the R1 recipe in fold 0); R2-inner makes only the hypotheses against R2 unavailable.
The scoring pass and the futility rule apply this one function.

Precision (rule A2-GOV-3): every fit entry of a record carries the fit's runtime precision policy,
read from the fit's log with ``ec_clean30.runtime_precision.read_policy_log`` at evaluation time
(``read_fit_policy``) and validated again with ``runtime_precision.validate_policy`` by
``verify_record``; a missing module or a policy that does not validate refuses the record.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from project_paths import PROJECT_ROOT
from training.ec_classes import EC_CLASS_TOKENS


SCHEMA_VERSION = 1
RECORD_KIND = "ec_clean30_clean_sanity"
RECORD_FORMAT = "clean_sanity_v1"
DISPOSITION_KIND = "ec_clean30_clean_sanity_disposition"
DISPOSITION_FORMAT = "disposition_v1"
CHECK_IN_SAMPLE = "in_sample"
CHECK_HELD_OUT = "held_out"
CHECKS = (CHECK_IN_SAMPLE, CHECK_HELD_OUT)
STATUS_PASSED = "passed"
STATUS_DIAGNOSIS_REQUIRED = "comparator_diagnosis_required"
STATUSES = (STATUS_PASSED, STATUS_DIAGNOSIS_REQUIRED)
DECISION_PROCEED_DISCLOSED = "proceed_disclosed"
DECISION_STOP = "stop"
DECISIONS = (DECISION_PROCEED_DISCLOSED, DECISION_STOP)
# The areas the maintainer searches for defects before any disposition (A2-S7 failure response).
DEFECT_AREAS = ("inputs", "embedding_farms", "ec_column_order", "checkpoint_identity", "distance_map",
                "projection", "receipts")

# CLEAN arms and the comparator regime each one is trained under.
ARM_REGIME = {"r1": "r1", "r2": "r2", "c_esmc": "r1", "r1_inner": "r1_inner", "r2_inner": "r2_inner"}
ARMS = tuple(ARM_REGIME)
HELD_OUT_ARMS = ("r1_inner", "r2_inner")
HELD_OUT_FOLD = 0
N_INNER_FOLDS = 4

# Thresholds (fractions of the primary metric), fixed before any record exists.
IS_MIN_PRIMARY = Fraction("0.80")
IS_MAX_ANCHOR_DEFICIT = Fraction("0.05")
MAX_TIE_SHARE = Fraction("0.05")
HO_MAX_ANCHOR_DEFICIT = Fraction("0.10")
HO_MIN_MAJORITY_MARGIN = Fraction("0.10")
HO_PER_FIT_MAX_ANCHOR_DEFICIT = Fraction("0.20")
IN_SAMPLE_N = 256
THRESHOLDS = {
    "IS_MIN_PRIMARY": str(IS_MIN_PRIMARY), "IS_MAX_ANCHOR_DEFICIT": str(IS_MAX_ANCHOR_DEFICIT),
    "MAX_TIE_SHARE": str(MAX_TIE_SHARE), "HO_MAX_ANCHOR_DEFICIT": str(HO_MAX_ANCHOR_DEFICIT),
    "HO_MIN_MAJORITY_MARGIN": str(HO_MIN_MAJORITY_MARGIN),
    "HO_PER_FIT_MAX_ANCHOR_DEFICIT": str(HO_PER_FIT_MAX_ANCHOR_DEFICIT), "IN_SAMPLE_N": IN_SAMPLE_N,
}
SAMPLE_RULE = ("training proteins with a single EC class among EC classes 1-6 whose every full EC number has at "
               "least two training proteins, ordered by SHA-256 of '<fold>:<regime>:<inner fold or ->:<accession>'; "
               f"the first {IN_SAMPLE_N}")
ANCHOR_RULE = ("EC centres are the mean base input embedding of each EC number's training proteins; Euclidean "
               "distance; the comparator's projection and tie rule")
PRIMARY_RULE = "accuracy over the proteins of EC classes 1-6; a tie is wrong"
_PLACEHOLDER_CHARACTERS = frozenset("[]{}<>")
_SHA256_HEX = frozenset("0123456789abcdef")
PRIMARY_CLASS_TOKENS = tuple(EC_CLASS_TOKENS[:6])

POLICY_DEVICES = ("cuda", "cpu")
PRECISION_RULE = ("each fit entry carries the fit's runtime precision policy (read from its log with "
                  "ec_clean30.runtime_precision.read_policy_log); verify_record validates it again with "
                  "runtime_precision.validate_policy for its device; a missing module or an invalid policy refuses")
# Choice C6b (pending the maintainer's confirmation with C1-C8): the consequence of a held-out arm recorded as
# "proceed, disclosed". The scoring pass and the futility rule both apply ``held_out_consequence``.
HELD_OUT_CONSEQUENCE_RULE = "held_out_arm_consequence_v1"
HELD_OUT_CONSEQUENCE_TEXT = (
    "an R1-inner held-out arm recorded as 'proceed, disclosed' makes both co-primaries and every secondary "
    "hypothesis against CLEAN R1 unavailable ('a CLEAN sanity check failed without an identified defect'), because "
    "R1-inner stands for the R1 recipe in fold 0; an R2-inner arm so recorded makes only the hypotheses against the "
    "R2 seed mean (secondary hypotheses 3 and 4) unavailable; the futility rule then returns stop_futility with 0 "
    "successes whenever no co-primary can be available")

COMPARATOR_PATH = PROJECT_ROOT / "scripts" / "run_clean_split30_comparator.py"
_COMPARATOR_MODULE = "run_clean_split30_comparator"


class SanityError(RuntimeError):
    """A sanity input, record or disposition failed its contract."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise SanityError(message)


def load_comparator():
    """The comparator script as a module, shared with the scorer through ``sys.modules``."""
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


def _runtime_precision():
    """``ec_clean30.runtime_precision`` (execution lane; scientific file), refused when it is missing."""
    try:
        return importlib.import_module("ec_clean30.runtime_precision")
    except ImportError as exc:
        raise SanityError("The CLEAN sanity verifier needs ec_clean30.runtime_precision to check the fit's "
                          "precision policy (rule A2-GOV-3)") from exc


def read_fit_policy(log_path: Path | str, *, device: str = "cuda") -> dict[str, Any]:
    """The fit's precision-policy entry from its log (``runtime_precision.read_policy_log``)."""
    _require(device in POLICY_DEVICES, f"The policy device must be one of {POLICY_DEVICES}")
    path = Path(log_path)
    _require(path.is_file() and not path.is_symlink(), f"The fit's precision-policy log {path.name} is missing")
    try:
        record = _runtime_precision().read_policy_log(path, device=device)
    except RuntimeError as exc:
        raise SanityError(f"The fit's precision-policy log {path.name} does not validate: {exc}") from None
    return {"device": device, "log_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "record": dict(record)}


def check_fit_policy(entry: Any, what: str) -> dict[str, Any]:
    """A fit's recorded precision-policy entry, validated again (``runtime_precision.validate_policy``)."""
    _require(isinstance(entry, Mapping) and entry.get("device") in POLICY_DEVICES
             and isinstance(entry.get("record"), Mapping) and _sha256(entry.get("log_sha256")),
             f"{what} records no precision policy (rule A2-GOV-3)")
    try:
        _runtime_precision().validate_policy(dict(entry["record"]), device=entry["device"])
    except RuntimeError as exc:
        raise SanityError(f"{what}: the recorded precision policy does not validate ({exc})") from None
    return dict(entry)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_lines(values: Sequence[str]) -> str:
    return hashlib.sha256("".join(f"{value}\n" for value in values).encode("utf-8")).hexdigest()


def json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"


def write_once(path: Path | str, payload: Mapping[str, Any]) -> str:
    """Write a record or disposition once (mode ``x``); returns the SHA-256 of the bytes written."""
    text = json_text(payload)
    try:
        with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    except FileExistsError:
        raise SanityError(f"Refusing to overwrite {path} (write-once)") from None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- training tables

def ec_class_tokens(ec_numbers: Sequence[str]) -> tuple[str, ...]:
    """Distinct EC classes (first fields) of a protein's EC numbers, in class order."""
    tokens = {str(number).split(".", 1)[0].strip() for number in ec_numbers}
    return tuple(token for token in EC_CLASS_TOKENS if token in tokens)


def single_ec_class(ec_numbers: Sequence[str]) -> Optional[str]:
    """The protein's EC class when it has exactly one, else None."""
    tokens = ec_class_tokens(ec_numbers)
    return tokens[0] if len(tokens) == 1 else None


def ec_members(training_ec_numbers: Mapping[str, Sequence[str]]) -> dict[str, frozenset[str]]:
    """EC number -> training proteins carrying it, in first-appearance order (the upstream centre order)."""
    members: dict[str, set[str]] = {}
    for accession, numbers in training_ec_numbers.items():
        for number in numbers:
            members.setdefault(str(number), set()).add(str(accession))
    return {number: frozenset(ids) for number, ids in members.items()}


def sample_key(fold: int, regime: str, inner_fold: Optional[int], accession: str) -> str:
    return f"{int(fold)}:{regime}:{'-' if inner_fold is None else int(inner_fold)}:{accession}"


def in_sample_eligible(training_ec_numbers: Mapping[str, Sequence[str]]) -> tuple[str, ...]:
    """Training proteins with a single EC class among EC classes 1-6 whose every EC number has >= 2 members."""
    members = ec_members(training_ec_numbers)
    eligible = []
    for accession, numbers in training_ec_numbers.items():
        token = single_ec_class(numbers)
        if token in PRIMARY_CLASS_TOKENS and numbers and all(len(members[str(number)]) >= 2 for number in numbers):
            eligible.append(str(accession))
    return tuple(eligible)


def in_sample_sample(training_ec_numbers: Mapping[str, Sequence[str]], *, fold: int, regime: str,
                     inner_fold: Optional[int] = None, n: int = IN_SAMPLE_N) -> tuple[str, ...]:
    """The hashed, deterministic in-sample query sample of one fit (``SAMPLE_RULE``)."""
    _require(regime in set(ARM_REGIME.values()), f"Unknown regime {regime!r}")
    _require((regime in HELD_OUT_ARMS) == (inner_fold is not None),
             f"Regime {regime} {'needs' if regime in HELD_OUT_ARMS else 'takes no'} inner fold")
    eligible = in_sample_eligible(training_ec_numbers)
    ordered = sorted(eligible, key=lambda accession: hashlib.sha256(
        sample_key(fold, regime, inner_fold, accession).encode("utf-8")).hexdigest())
    return tuple(ordered[:int(n)])


def majority_class(training_ec_numbers: Mapping[str, Sequence[str]]) -> str:
    """The most frequent EC class among the training proteins with a single EC class (ties: lower class)."""
    counts = Counter(token for token in (single_ec_class(numbers) for numbers in training_ec_numbers.values())
                     if token is not None)
    _require(bool(counts), "The training table has no protein with a single EC class")
    return min(counts, key=lambda token: (-counts[token], EC_CLASS_TOKENS.index(token)))


def structural_tie_groups(training_ec_numbers: Mapping[str, Sequence[str]]) -> dict[str, int]:
    """EC-number groups with identical training-member sets that span more than one EC class.

    Identical member sets give identical CLEAN centres, so a query nearest to one of them is a tie
    between EC classes by construction. Totals only.
    """
    members = ec_members(training_ec_numbers)
    groups: dict[frozenset[str], list[str]] = defaultdict(list)
    for number, ids in members.items():
        groups[ids].append(number)
    crossing = [(ids, numbers) for ids, numbers in groups.items()
                if len(numbers) > 1 and len({number.split(".", 1)[0] for number in numbers}) > 1]
    proteins = set().union(*(ids for ids, _ in crossing)) if crossing else set()
    return {"n_ec_numbers": len(members),
            "n_identical_member_groups": sum(len(numbers) > 1 for numbers in groups.values()),
            "n_cross_class_groups": len(crossing),
            "n_ec_numbers_in_cross_class_groups": sum(len(numbers) for _, numbers in crossing),
            "n_proteins_in_cross_class_groups": len(proteins)}


# ---------------------------------------------------------------- projections

def ec_centres(training_ec_numbers: Mapping[str, Sequence[str]],
               embeddings: Mapping[str, np.ndarray]) -> tuple[tuple[str, ...], np.ndarray]:
    """EC labels in first-appearance order and the mean base embedding of each label's training proteins."""
    members = ec_members(training_ec_numbers)
    missing = sorted({accession for ids in members.values() for accession in ids} - set(embeddings))
    _require(not missing, f"{len(missing)} training proteins have no base embedding")
    order = {accession: index for index, accession in enumerate(training_ec_numbers)}
    labels = tuple(members)
    centres = np.stack([np.mean([np.asarray(embeddings[accession], dtype=np.float64)
                                 for accession in sorted(members[label], key=order.__getitem__)], axis=0)
                        for label in labels])
    return labels, centres


def euclidean_distances(queries: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Euclidean distance of each query row to each centre row (float64, one query at a time)."""
    queries = np.asarray(queries, dtype=np.float64)
    centres = np.asarray(centres, dtype=np.float64)
    _require(queries.ndim == 2 and centres.ndim == 2 and queries.shape[1] == centres.shape[1],
             "Query and centre embeddings must share their width")
    return np.stack([np.sqrt(((centres - row) ** 2).sum(axis=1)) for row in queries]) if len(queries) else \
        np.zeros((0, centres.shape[0]))


def raw_anchor_projection(training_ec_numbers: Mapping[str, Sequence[str]], train_embeddings: Mapping[str, np.ndarray],
                          query_ids: Sequence[str], query_embeddings: Mapping[str, np.ndarray]):
    """The raw-embedding anchor: nearest training EC centre of the base embeddings, then the projection rule."""
    comparator = load_comparator()
    labels, centres = ec_centres(training_ec_numbers, train_embeddings)
    missing = [query for query in query_ids if query not in query_embeddings]
    _require(not missing, f"{len(missing)} queries have no base embedding")
    queries = np.stack([np.asarray(query_embeddings[query], dtype=np.float64) for query in query_ids]) \
        if query_ids else np.zeros((0, centres.shape[1]))
    return comparator.project_ec_classes(euclidean_distances(queries, centres), tuple(query_ids), labels)


def export_projection(export, query_ids: Sequence[str]):
    """The comparator's projection of an export, restricted to ``query_ids`` (all must be present)."""
    comparator = load_comparator()
    row_of = {query: index for index, query in enumerate(export.query_ids)}
    missing = [query for query in query_ids if query not in row_of]
    _require(not missing, f"{len(missing)} queries are not in the export")
    rows = [row_of[query] for query in query_ids]
    return comparator.project_ec_classes(export.distances[rows], tuple(query_ids), export.ec_labels)


def _score(projection, true_class: Mapping[str, str]) -> dict[str, int]:
    correct = sum(not tied and predicted == true_class[query]
                  for query, predicted, tied in zip(projection.query_ids, projection.predicted_class, projection.tie))
    return {"n_queries": len(projection.query_ids), "n_correct": int(correct), "n_ties": int(projection.n_ties)}


def _fraction(numerator: int, denominator: int) -> Fraction:
    return Fraction(int(numerator), int(denominator)) if denominator else Fraction(0)


def _value(numerator: int, denominator: int) -> Optional[float]:
    return float(Fraction(numerator, denominator)) if denominator else None


# ---------------------------------------------------------------- in-sample check

def _fit_identity(receipt: Mapping[str, Any], arm: str) -> dict[str, Any]:
    comparator = load_comparator()
    _require(arm in ARM_REGIME, f"Unknown CLEAN arm {arm!r}")
    _require(receipt.get("mode") == "validation", "A sanity check reads validation-mode exports only")
    _require(receipt.get("regime") == ARM_REGIME[arm], f"Arm {arm} is trained under regime {ARM_REGIME[arm]}, the "
                                                       f"export has {receipt.get('regime')!r}")
    if arm == "c_esmc":
        _require(receipt.get("embedding") != comparator.DEFAULT_EMBEDDING, "C-ESMC uses the ESM-1b embedding")
    else:
        _require(receipt.get("embedding") == comparator.DEFAULT_EMBEDDING
                 and receipt.get("input_dim") == comparator.DEFAULT_INPUT_DIM,
                 f"Arm {arm} uses {comparator.DEFAULT_EMBEDDING} at width {comparator.DEFAULT_INPUT_DIM}")
    return {"arm": arm, "fold": int(receipt["fold"]), "regime": receipt["regime"],
            "inner_fold": receipt.get("inner_fold"), "seed": int(receipt["seed"]), "embedding": receipt["embedding"],
            "input_dim": int(receipt["input_dim"]), "checkpoint_sha256": receipt["checkpoint_sha256"],
            "training_table_sha256": receipt["training_table_sha256"],
            "embedding_farm_sha256": receipt["embedding_farm_sha256"]}


def evaluate_in_sample(*, arm: str, export, training_ec_numbers: Mapping[str, Sequence[str]], sample: Sequence[str],
                       anchor, precision_policy: Mapping[str, Any],
                       created_utc: Optional[str] = None) -> dict[str, Any]:
    """The in-sample record of one fit from its in-sample validation export and the raw anchor;
    ``precision_policy`` is the fit's entry from ``read_fit_policy``."""
    receipt = export.receipt
    fit = _fit_identity(receipt, arm)
    fit["precision_policy"] = check_fit_policy(precision_policy, f"CLEAN {arm} fit")
    sample = tuple(sample)
    expected = in_sample_sample(training_ec_numbers, fold=fit["fold"], regime=fit["regime"],
                                inner_fold=fit["inner_fold"])
    _require(sample == expected, "The sample is not the hashed in-sample sample of this fit's training table")
    _require(bool(sample), "The fit's training table has no eligible in-sample protein")
    _require(tuple(anchor.query_ids) == sample, "The anchor projection does not cover exactly the sample")
    true_class = {accession: single_ec_class(training_ec_numbers[accession]) for accession in sample}
    present = [query for query in sample if query in set(export.query_ids)]
    fit_score = _score(export_projection(export, present), true_class)
    fit_score["n_queries"] = len(sample)  # a sample query missing from the export counts wrong (and fails IS4)
    anchor_score = _score(anchor, true_class)
    n_in_sample = receipt.get("n_in_sample_queries")
    is4 = (set(export.query_ids) == set(sample) and len(export.query_ids) == len(sample)
           and n_in_sample == len(sample))
    return build_in_sample_record(
        fit=fit,
        export={"npz_sha256": receipt.get("npz_sha256"), "mode": receipt["mode"], "n_query": len(export.query_ids),
                "n_in_sample_queries": n_in_sample, "query_ids_sha256": receipt.get("query_ids_sha256")},
        sample={"n": len(sample), "n_eligible": len(in_sample_eligible(training_ec_numbers)),
                "sha256": sha256_lines(sample), "export_matches_sample": bool(is4)},
        fit_totals=fit_score, anchor_totals=anchor_score, created_utc=created_utc)


def build_in_sample_record(*, fit: Mapping[str, Any], export: Mapping[str, Any], sample: Mapping[str, Any],
                           fit_totals: Mapping[str, int], anchor_totals: Mapping[str, int],
                           created_utc: Optional[str] = None) -> dict[str, Any]:
    """An in-sample record from its totals; the criteria and the status follow from them and the thresholds."""
    check_fit_policy(fit.get("precision_policy"), f"CLEAN {fit.get('arm')} fit")
    record = {
        "schema_version": SCHEMA_VERSION, "kind": RECORD_KIND, "format": RECORD_FORMAT, "check": CHECK_IN_SAMPLE,
        "created_utc": created_utc or utc_now(), "fit": dict(fit), "export": dict(export),
        "sample": {"rule": SAMPLE_RULE, "n_requested": IN_SAMPLE_N, **dict(sample)},
        "primary_rule": PRIMARY_RULE, "anchor_rule": ANCHOR_RULE, "thresholds": dict(THRESHOLDS),
        "values": {"fit": {key: int(fit_totals[key]) for key in ("n_queries", "n_correct", "n_ties")},
                   "anchor": {key: int(anchor_totals[key]) for key in ("n_queries", "n_correct", "n_ties")}},
    }
    return _finish_in_sample(record)


def _in_sample_criteria(record: Mapping[str, Any]) -> dict[str, bool]:
    fit, anchor = record["values"]["fit"], record["values"]["anchor"]
    primary = _fraction(fit["n_correct"], fit["n_queries"])
    anchor_primary = _fraction(anchor["n_correct"], anchor["n_queries"])
    return {"IS1": fit["n_queries"] > 0 and primary >= IS_MIN_PRIMARY,
            "IS2": primary >= anchor_primary - IS_MAX_ANCHOR_DEFICIT,
            "IS3": fit["n_queries"] > 0 and _fraction(fit["n_ties"], fit["n_queries"]) <= MAX_TIE_SHARE,
            "IS4": bool(record["sample"]["export_matches_sample"])}


def _finish_in_sample(record: dict[str, Any]) -> dict[str, Any]:
    fit, anchor = record["values"]["fit"], record["values"]["anchor"]
    record["values"]["primary"] = _value(fit["n_correct"], fit["n_queries"])
    record["values"]["anchor_primary"] = _value(anchor["n_correct"], anchor["n_queries"])
    record["values"]["tie_share"] = _value(fit["n_ties"], fit["n_queries"])
    record["criteria"] = _in_sample_criteria(record)
    record["status"] = STATUS_PASSED if all(record["criteria"].values()) else STATUS_DIAGNOSIS_REQUIRED
    record["totals_only"] = True
    return record


# ---------------------------------------------------------------- held-out check

@dataclass(frozen=True)
class HeldOutFit:
    """One fold-0 inner fit: its validation export, its own training table and the raw anchor on the queries."""

    arm: str
    export: Any
    training_ec_numbers: Mapping[str, Sequence[str]]
    anchor: Any
    precision_policy: Optional[Mapping[str, Any]] = None  # the fit's entry from ``read_fit_policy``


def held_out_queries(inner_validation: Sequence[str], true_ec_numbers: Mapping[str, Sequence[str]],
                     r1_inner_training_ec_numbers: Mapping[str, Sequence[str]]) -> tuple[str, ...]:
    """Inner-validation proteins with one EC class among EC classes 1-6 whose EC numbers all have an R1-inner centre."""
    centres = set(ec_members(r1_inner_training_ec_numbers))
    trained = set(r1_inner_training_ec_numbers)
    leaked = sorted(set(inner_validation) & trained)
    _require(not leaked, f"{len(leaked)} inner-validation proteins are in the R1-inner training table")
    chosen = []
    for accession in inner_validation:
        numbers = true_ec_numbers.get(accession)
        _require(numbers is not None, "An inner-validation protein has no training-side EC annotation")
        if single_ec_class(numbers) in PRIMARY_CLASS_TOKENS and all(str(number) in centres for number in numbers):
            chosen.append(str(accession))
    return tuple(sorted(chosen))


def evaluate_held_out(fits: Sequence[HeldOutFit], queries: Mapping[int, Sequence[str]],
                      true_ec_numbers: Mapping[str, Sequence[str]], *,
                      created_utc: Optional[str] = None) -> dict[str, Any]:
    """The fold-0 held-out record over the four R1-inner and four R2-inner fits.

    ``queries`` maps each inner fold to its closure-eligible inner-validation proteins
    (``held_out_queries``); ``true_ec_numbers`` holds their training-side EC numbers.
    """
    by_key: dict[tuple[str, int], HeldOutFit] = {}
    for item in fits:
        receipt = item.export.receipt
        fit = _fit_identity(receipt, item.arm)
        _require(item.arm in HELD_OUT_ARMS, f"The held-out check covers {HELD_OUT_ARMS}, not {item.arm}")
        _require(fit["fold"] == HELD_OUT_FOLD, f"The held-out check covers fold {HELD_OUT_FOLD} only")
        _require(receipt.get("n_in_sample_queries") == 0,
                 "The held-out check refuses an export with in-sample queries (an engineering export)")
        key = (item.arm, int(fit["inner_fold"]))
        _require(key not in by_key, f"Two fits for {item.arm} inner fold {key[1]}")
        by_key[key] = item
    expected = {(arm, inner) for arm in HELD_OUT_ARMS for inner in range(N_INNER_FOLDS)}
    _require(set(by_key) == expected, f"The held-out check needs one fit per arm and inner fold; missing "
                                      f"{sorted(expected - set(by_key))}")
    _require(sorted(queries) == list(range(N_INNER_FOLDS)), "Queries must be given for every inner fold")
    arms: dict[str, list[dict[str, Any]]] = {}
    for arm in HELD_OUT_ARMS:
        rows = []
        for inner in range(N_INNER_FOLDS):
            item = by_key[(arm, inner)]
            chosen = tuple(sorted(queries[inner]))
            leaked = sorted(set(chosen) & set(item.training_ec_numbers))
            _require(not leaked, f"{len(leaked)} held-out queries are training proteins of {arm} inner fold {inner}")
            _require(tuple(item.anchor.query_ids) == chosen, f"The anchor of {arm} inner fold {inner} does not cover "
                                                             "exactly its queries")
            true_class = {query: single_ec_class(true_ec_numbers[query]) for query in chosen}
            fit_score = _score(export_projection(item.export, chosen), true_class)
            anchor_score = _score(item.anchor, true_class)
            majority = majority_class(item.training_ec_numbers)
            majority_correct = sum(true_class[query] == majority for query in chosen)
            fit = _fit_identity(item.export.receipt, arm)
            rows.append({"inner_fold": inner, "seed": fit["seed"], "checkpoint_sha256": fit["checkpoint_sha256"],
                         "precision_policy": check_fit_policy(item.precision_policy, f"CLEAN {arm} inner fold {inner}"),
                         "training_table_sha256": fit["training_table_sha256"],
                         "npz_sha256": item.export.receipt.get("npz_sha256"), "queries_sha256": sha256_lines(chosen),
                         "fit": fit_score, "anchor": anchor_score, "majority_n_correct": int(majority_correct)})
        arms[arm] = rows
    return build_held_out_record(arms, created_utc=created_utc)


def build_held_out_record(arms: Mapping[str, Sequence[Mapping[str, Any]]],
                          created_utc: Optional[str] = None) -> dict[str, Any]:
    """A held-out record from per-fit totals (``fit`` and ``anchor`` totals, ``majority_n_correct``); the pooled
    totals, criteria and status follow from them and the thresholds."""
    entries = {}
    for arm, rows in arms.items():
        rows = [{**dict(row), "fit": dict(row["fit"]), "anchor": dict(row["anchor"])} for row in rows]
        for row in rows:
            check_fit_policy(row.get("precision_policy"), f"CLEAN {arm} inner fold {row.get('inner_fold')}")
        pooled = {"n_queries": sum(row["fit"]["n_queries"] for row in rows),
                  "n_correct": sum(row["fit"]["n_correct"] for row in rows),
                  "anchor_n_correct": sum(row["anchor"]["n_correct"] for row in rows),
                  "majority_n_correct": sum(int(row["majority_n_correct"]) for row in rows)}
        entries[arm] = {"fits": sorted(rows, key=lambda row: row["inner_fold"]), "pooled": pooled}
    record = {"schema_version": SCHEMA_VERSION, "kind": RECORD_KIND, "format": RECORD_FORMAT, "check": CHECK_HELD_OUT,
              "created_utc": created_utc or utc_now(), "fold": HELD_OUT_FOLD, "primary_rule": PRIMARY_RULE,
              "anchor_rule": ANCHOR_RULE, "thresholds": dict(THRESHOLDS), "arms": entries}
    return _finish_held_out(record)


def _held_out_criteria(record: Mapping[str, Any]) -> dict[str, Any]:
    per_arm: dict[str, dict[str, bool]] = {}
    for arm, entry in record["arms"].items():
        pooled = entry["pooled"]
        primary = _fraction(pooled["n_correct"], pooled["n_queries"])
        criteria = {"HO1": pooled["n_queries"] > 0
                    and primary >= _fraction(pooled["anchor_n_correct"], pooled["n_queries"]) - HO_MAX_ANCHOR_DEFICIT}
        if arm == "r1_inner":
            criteria["HO2"] = pooled["n_queries"] > 0 and primary >= (
                _fraction(pooled["majority_n_correct"], pooled["n_queries"]) + HO_MIN_MAJORITY_MARGIN)
        criteria["HO3"] = all(row["fit"]["n_queries"] > 0
                              and _fraction(row["fit"]["n_ties"], row["fit"]["n_queries"]) <= MAX_TIE_SHARE
                              for row in entry["fits"])
        criteria["HO4"] = all(_fraction(row["fit"]["n_correct"], row["fit"]["n_queries"]) >= (
            _fraction(row["anchor"]["n_correct"], row["anchor"]["n_queries"]) - HO_PER_FIT_MAX_ANCHOR_DEFICIT)
            for row in entry["fits"])
        per_arm[arm] = criteria
    return per_arm


def _finish_held_out(record: dict[str, Any]) -> dict[str, Any]:
    for entry in record["arms"].values():
        pooled = entry["pooled"]
        n = pooled["n_queries"]
        pooled.update(primary=_value(pooled["n_correct"], n), anchor_primary=_value(pooled["anchor_n_correct"], n),
                      majority_primary=_value(pooled["majority_n_correct"], n))
        for row in entry["fits"]:
            row["primary"] = _value(row["fit"]["n_correct"], row["fit"]["n_queries"])
            row["anchor_primary"] = _value(row["anchor"]["n_correct"], row["anchor"]["n_queries"])
            row["tie_share"] = _value(row["fit"]["n_ties"], row["fit"]["n_queries"])
    record["criteria"] = _held_out_criteria(record)
    record["arm_status"] = _arm_status(record["criteria"])
    passed = all(all(values.values()) for values in record["criteria"].values())
    record["status"] = STATUS_PASSED if passed else STATUS_DIAGNOSIS_REQUIRED
    record["totals_only"] = True
    return record


def _arm_status(criteria: Mapping[str, Mapping[str, bool]]) -> dict[str, str]:
    """The held-out status of each arm (R1-inner and R2-inner) from its criteria."""
    return {arm: STATUS_PASSED if all(values.values()) else STATUS_DIAGNOSIS_REQUIRED
            for arm, values in sorted(criteria.items())}


# ---------------------------------------------------------------- verification

def _sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _SHA256_HEX


def _load(source: Path | str | Mapping[str, Any]) -> tuple[dict[str, Any], Optional[str]]:
    if isinstance(source, Mapping):
        return dict(source), None
    path = Path(source)
    _require(path.is_file(), f"Sanity file not found: {path}")
    data = path.read_bytes()
    try:
        payload = json.loads(data.decode("utf-8"))
    except ValueError as exc:
        raise SanityError(f"{path} is not JSON: {exc}") from None
    _require(isinstance(payload, dict), f"{path} must hold a JSON object")
    return payload, hashlib.sha256(data).hexdigest()


def verify_record(source: Path | str | Mapping[str, Any]) -> dict[str, Any]:
    """Check a sanity record (format, fixed thresholds, criteria and status re-derived from its totals).

    Returns a summary: ``sha256`` (of the file; None for an in-memory record), ``check``,
    ``status``, the fit identity of an in-sample record, or the fits of a held-out record.
    """
    record, digest = _load(source)
    _require(record.get("schema_version") == SCHEMA_VERSION and record.get("kind") == RECORD_KIND
             and record.get("format") == RECORD_FORMAT, f"Not a {RECORD_FORMAT} sanity record")
    _require(record.get("check") in CHECKS, f"Unknown sanity check {record.get('check')!r}")
    _require(record.get("thresholds") == THRESHOLDS, "The record's thresholds differ from the fixed thresholds")
    _require(record.get("status") in STATUSES, f"Unknown status {record.get('status')!r}")
    _require(record.get("totals_only") is True, "A sanity record holds totals only")
    try:
        if record["check"] == CHECK_IN_SAMPLE:
            fit = record["fit"]
            _require(fit.get("arm") in ARM_REGIME and fit.get("regime") == ARM_REGIME[fit["arm"]],
                     "The record's arm and regime disagree")
            _require(_sha256(fit.get("checkpoint_sha256")), "The record names no checkpoint SHA-256")
            check_fit_policy(fit.get("precision_policy"), f"CLEAN {fit['arm']} fit")
            for key in ("fold", "seed"):
                _require(isinstance(fit.get(key), int) and not isinstance(fit.get(key), bool), f"Bad fit {key}")
            values = record["values"]
            _require(values["fit"]["n_queries"] == values["anchor"]["n_queries"] == record["sample"]["n"]
                     and 0 < record["sample"]["n"] <= IN_SAMPLE_N, "The record's query counts disagree with its sample")
            _require(values.get("primary") == _value(values["fit"]["n_correct"], values["fit"]["n_queries"])
                     and values.get("anchor_primary") == _value(values["anchor"]["n_correct"],
                                                                values["anchor"]["n_queries"])
                     and values.get("tie_share") == _value(values["fit"]["n_ties"], values["fit"]["n_queries"]),
                     "The record's values do not follow from its totals")
            criteria = _in_sample_criteria(record)
            summary = {"arm": fit["arm"], "fold": fit["fold"], "seed": fit["seed"], "inner_fold": fit.get("inner_fold"),
                       "checkpoint_sha256": fit["checkpoint_sha256"]}
            passed = all(criteria.values())
        else:
            _require(record.get("fold") == HELD_OUT_FOLD and sorted(record.get("arms", {})) == sorted(HELD_OUT_ARMS),
                     "A held-out record covers the fold-0 R1-inner and R2-inner fits")
            for arm, entry in record["arms"].items():
                rows = entry["fits"]
                _require(sorted(row["inner_fold"] for row in rows) == list(range(N_INNER_FOLDS)),
                         f"The held-out record of {arm} lacks an inner fold")
                sums = {"n_queries": sum(row["fit"]["n_queries"] for row in rows),
                        "n_correct": sum(row["fit"]["n_correct"] for row in rows),
                        "anchor_n_correct": sum(row["anchor"]["n_correct"] for row in rows),
                        "majority_n_correct": sum(row["majority_n_correct"] for row in rows)}
                _require(all(entry["pooled"][key] == value for key, value in sums.items()),
                         f"The pooled totals of {arm} are not the sums over its fits")
                _require(all(_sha256(row.get("checkpoint_sha256")) for row in rows),
                         f"A held-out fit of {arm} names no checkpoint SHA-256")
                for row in rows:
                    check_fit_policy(row.get("precision_policy"), f"CLEAN {arm} inner fold {row['inner_fold']}")
            criteria = _held_out_criteria(record)
            _require(record.get("arm_status") == _arm_status(criteria),
                     "The record's per-arm statuses do not follow from its criteria")
            summary = {"fold": HELD_OUT_FOLD, "checkpoint_sha256s": sorted(
                row["checkpoint_sha256"] for entry in record["arms"].values() for row in entry["fits"]),
                "checkpoint_sha256_by_fit": {arm: {str(row["inner_fold"]): row["checkpoint_sha256"]
                                                   for row in entry["fits"]}
                                             for arm, entry in sorted(record["arms"].items())},
                "arm_status": dict(record["arm_status"])}
            passed = all(all(values.values()) for values in criteria.values())
    except (KeyError, TypeError, ZeroDivisionError) as exc:
        raise SanityError(f"Malformed sanity record ({type(exc).__name__}: {exc})") from None
    _require(record.get("criteria") == criteria, "The record's criteria do not follow from its totals")
    _require((record["status"] == STATUS_PASSED) == passed, "The record's status does not follow from its criteria")
    return {"sha256": digest, "check": record["check"], "status": record["status"], **summary}


def build_disposition(*, record_sha256: str, decision: str, maintainer: str, note: str,
                      created_utc: Optional[str] = None, arm: Optional[str] = None) -> dict[str, Any]:
    """The maintainer's disposition of a failed record, written only after no defect was found. A disposition of
    the held-out record names its failed arm (``r1_inner`` or ``r2_inner``)."""
    payload = {"schema_version": SCHEMA_VERSION, "kind": DISPOSITION_KIND, "format": DISPOSITION_FORMAT,
               "created_utc": created_utc or utc_now(), "record_sha256": str(record_sha256).strip().lower(),
               "decision": decision, "maintainer": str(maintainer).strip(), "defects_checked": list(DEFECT_AREAS),
               "no_defect_found": True, "note": str(note).strip()}
    if arm is not None:
        payload["arm"] = arm
    verify_disposition(payload)
    return payload


def verify_disposition(source: Path | str | Mapping[str, Any],
                       records: Optional[Mapping[str, Mapping[str, Any]]] = None) -> dict[str, Any]:
    """Check a disposition; with ``records`` (SHA-256 -> ``verify_record`` summary) it must name a failed one."""
    payload, digest = _load(source)
    _require(payload.get("schema_version") == SCHEMA_VERSION and payload.get("kind") == DISPOSITION_KIND
             and payload.get("format") == DISPOSITION_FORMAT, f"Not a {DISPOSITION_FORMAT} disposition")
    _require(payload.get("decision") in DECISIONS, f"A disposition decides one of {DECISIONS}")
    _require(_sha256(payload.get("record_sha256")), "A disposition names its record by SHA-256")
    maintainer = payload.get("maintainer")
    _require(isinstance(maintainer, str) and maintainer.strip() and not set(maintainer) & _PLACEHOLDER_CHARACTERS,
             "A disposition names the maintainer who recorded it (a placeholder is refused)")
    _require(sorted(payload.get("defects_checked") or []) == sorted(DEFECT_AREAS),
             f"A disposition records the defect search over {list(DEFECT_AREAS)}")
    _require(payload.get("no_defect_found") is True,
             "A disposition is recorded only when no defect was found (a defect is fixed and the fit rechecked)")
    _require(isinstance(payload.get("note"), str) and payload["note"].strip(), "A disposition carries a note")
    arm = payload.get("arm")
    _require(arm is None or arm in HELD_OUT_ARMS, f"A disposition's arm is one of {HELD_OUT_ARMS}")
    if records is not None:
        target = records.get(payload["record_sha256"])
        _require(target is not None, "The disposition names no supplied sanity record")
        _require(target["status"] == STATUS_DIAGNOSIS_REQUIRED, "A disposition applies to a failed record only")
        if target.get("check") == CHECK_HELD_OUT:
            _require(arm is not None and (target.get("arm_status") or {}).get(arm) == STATUS_DIAGNOSIS_REQUIRED,
                     "A disposition of the held-out record names one of its failed arms")
        else:
            _require(arm is None, "A disposition of an in-sample record names no arm")
    return {"sha256": digest, "record_sha256": payload["record_sha256"], "decision": payload["decision"],
            "maintainer": maintainer.strip(), "arm": arm}


def require_passed_or_disposed(record: Path | str, dispositions: Sequence[Path | str] = ()) -> dict[str, Any]:
    """For consumers such as the G4 and futility records: a passed record, or a failed one with
    a ``proceed_disclosed`` disposition. Returns the record summary with its disposition."""
    summary = verify_record(record)
    _require(summary["sha256"] is not None, "Bind a record file, not an in-memory record")
    disposed = [verify_disposition(path, {summary["sha256"]: summary}) for path in dispositions]
    decision = None
    if summary["check"] == CHECK_HELD_OUT:
        arm_disposition: dict[str, Optional[str]] = {}
        for arm in HELD_OUT_ARMS:
            named = [item for item in disposed if item["arm"] == arm]
            if summary["arm_status"][arm] == STATUS_PASSED:
                _require(not named, f"The passed held-out arm {arm} takes no disposition")
                arm_disposition[arm] = None
            else:
                _require(len(named) == 1, f"The failed held-out arm {arm} needs exactly one disposition")
                _require(named[0]["decision"] == DECISION_PROCEED_DISCLOSED,
                         f"The disposition of the held-out arm {arm} is 'stop'")
                arm_disposition[arm] = DECISION_PROCEED_DISCLOSED
        if any(arm_disposition.values()):
            decision = DECISION_PROCEED_DISCLOSED
        return {**summary, "disposition": decision, "arm_disposition": arm_disposition}
    if summary["status"] != STATUS_PASSED:
        _require(len(disposed) == 1, "A failed sanity record needs exactly one disposition")
        decision = disposed[0]["decision"]
        _require(decision == DECISION_PROCEED_DISCLOSED, "The disposition of this sanity record is 'stop'")
    else:
        _require(not disposed, "A passed sanity record takes no disposition")
    return {**summary, "disposition": decision}


def held_out_binding(record: Path | str, dispositions: Sequence[Path | str] = ()) -> dict[str, Any]:
    """The fold-0 held-out record (rule A2-CLN-2) for the S1 and FUT records: a passed held-out record, or a failed
    one with exactly one ``proceed_disclosed`` disposition; returns its summary, SHA-256 and disposition."""
    summary = verify_record(record)
    _require(summary["check"] == CHECK_HELD_OUT and summary.get("fold") == HELD_OUT_FOLD,
             "The S1 and futility records bind the fold-0 held-out sanity record, not an in-sample record")
    if summary["status"] != STATUS_PASSED and not dispositions:
        raise SanityError(f"The held-out sanity record has status {summary['status']!r}: comparator diagnosis is "
                          "required; fix the defects first, without retuning")
    bound = require_passed_or_disposed(record, dispositions)
    return {"sha256": bound["sha256"], "check": bound["check"], "status": bound["status"],
            "disposition": bound["disposition"], "fold": HELD_OUT_FOLD,
            "arm_status": dict(bound["arm_status"]), "arm_disposition": dict(bound["arm_disposition"]),
            "checkpoint_sha256_by_fit": {arm: dict(values) for arm, values in bound["checkpoint_sha256_by_fit"].items()},
            "consequence": held_out_consequence(bound["arm_disposition"])}


def held_out_consequence(arm_disposition: Mapping[str, Optional[str]]) -> dict[str, Any]:
    """Choice C6b (``HELD_OUT_CONSEQUENCE_TEXT``): which comparisons a "proceed, disclosed" held-out arm removes.

    ``unavailable_versus`` holds ``r1`` (both co-primaries and the secondary hypotheses against R1) and/or ``r2``
    (the hypotheses against the R2 seed mean); ``co_primaries_available`` is False when R1-inner is disclosed.
    """
    unknown = sorted(set(arm_disposition) - set(HELD_OUT_ARMS))
    _require(not unknown, f"Unknown held-out arms {unknown}")
    disclosed = {arm for arm, value in arm_disposition.items() if value == DECISION_PROCEED_DISCLOSED}
    versus = sorted({{"r1_inner": "r1", "r2_inner": "r2"}[arm] for arm in disclosed})
    return {"rule_id": HELD_OUT_CONSEQUENCE_RULE, "disclosed_arms": sorted(disclosed), "unavailable_versus": versus,
            "co_primaries_available": "r1" not in versus}


def verify_all(records: Sequence[Path | str], dispositions: Sequence[Path | str] = ()) -> dict[str, Any]:
    """The ``verify-all`` receipt: every record and disposition verified together (each disposition names one
    supplied failed record, or one failed held-out arm), with per-record outcomes and SHA-256 values."""
    summaries: dict[str, dict[str, Any]] = {}
    for path in records:
        summary = verify_record(path)
        _require(summary["sha256"] is not None and summary["sha256"] not in summaries,
                 f"Sanity record {Path(path).name} is listed twice or is not a file")
        summaries[summary["sha256"]] = summary
    disposed: dict[tuple[str, Optional[str]], dict[str, Any]] = {}
    for path in dispositions:
        item = verify_disposition(path, summaries)
        key = (item["record_sha256"], item["arm"])
        _require(key not in disposed, "Two dispositions name the same record (and arm)")
        disposed[key] = item
    rows = []
    for digest, summary in sorted(summaries.items()):
        if summary["check"] == CHECK_HELD_OUT:
            arms = {}
            for arm in HELD_OUT_ARMS:
                item = disposed.get((digest, arm))
                if summary["arm_status"][arm] == STATUS_PASSED:
                    _require(item is None, f"The passed held-out arm {arm} takes no disposition")
                    arms[arm] = STATUS_PASSED
                else:
                    _require(item is not None, f"The failed held-out arm {arm} has no disposition")
                    arms[arm] = item["decision"]
            rows.append({"record_sha256": digest, "check": CHECK_HELD_OUT, "status": summary["status"],
                         "arm_outcome": arms, "consequence": held_out_consequence(
                             {arm: (value if value == DECISION_PROCEED_DISCLOSED else None)
                              for arm, value in arms.items()})})
        else:
            item = disposed.get((digest, None))
            if summary["status"] == STATUS_PASSED:
                _require(item is None, "A passed sanity record takes no disposition")
                outcome = STATUS_PASSED
            else:
                _require(item is not None, "A failed sanity record has no disposition")
                outcome = item["decision"]
            rows.append({"record_sha256": digest, "check": CHECK_IN_SAMPLE, "status": summary["status"],
                         "arm": summary["arm"], "fold": summary["fold"], "seed": summary["seed"],
                         "checkpoint_sha256": summary["checkpoint_sha256"], "outcome": outcome})
    return {"schema_version": SCHEMA_VERSION, "kind": "ec_clean30_clean_sanity_verify_all", "created_utc": utc_now(),
            "n_records": len(summaries), "n_dispositions": len(disposed), "records": rows,
            "dispositions": [{"sha256": item["sha256"], "record_sha256": item["record_sha256"], "arm": item["arm"],
                              "decision": item["decision"]} for _key, item in sorted(disposed.items(),
                                                                                       key=lambda kv: str(kv[0]))],
            "stopped": sorted(row["record_sha256"] for row in rows
                              if row.get("outcome") == DECISION_STOP
                              or DECISION_STOP in (row.get("arm_outcome") or {}).values()),
            "totals_only": True}


def clean_sanity_policy_record() -> dict[str, Any]:
    """Rules A2-CLN-1 to A2-CLN-5 as data (bound by amendment 2)."""
    return {
        "rule_id": "clean_sanity_v1",
        "record_format": RECORD_FORMAT,
        "disposition_format": DISPOSITION_FORMAT,
        "checks": list(CHECKS),
        "in_sample_n": IN_SAMPLE_N,
        "thresholds": {name: str(value) for name, value in THRESHOLDS.items()},
        "sample_rule": SAMPLE_RULE,
        "anchor_rule": ANCHOR_RULE,
        "primary_rule": PRIMARY_RULE,
        "held_out_fold": HELD_OUT_FOLD,
        "held_out_arms": list(HELD_OUT_ARMS),
        "statuses": list(STATUSES),
        "decisions": list(DECISIONS),
        "defect_areas": list(DEFECT_AREAS),
        "failure_response": ("comparator_diagnosis_required: the fit enters no record; defects in inputs or receipts "
                             "are fixed and the same pinned recipe and seeds refit and rechecked; a code defect "
                             "after fold-0 data exist is a class-(b) amendment; recipe, epochs, loss, widths, "
                             "learning rate and seeds never change"),
        "proceed_disclosed_consequence": ("every comparison whose CLEAN arm includes the fit reads 'No superiority "
                                          "inference' (rule A2-MET-2)"),
        "held_out_arm_status": "one status per held-out arm; a failed arm takes its own disposition naming the arm",
        "held_out_consequence": {"rule_id": HELD_OUT_CONSEQUENCE_RULE, "text": HELD_OUT_CONSEQUENCE_TEXT,
                                 "confirmation": "C6b"},
        "precision": PRECISION_RULE,
        "policy_devices": list(POLICY_DEVICES),
        "loss_trends_used": False,
        "records_hold_totals_only": True,
    }
