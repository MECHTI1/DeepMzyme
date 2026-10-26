"""R2 and R2-inner accession lists for the EC class vs CLEAN30 campaign (step P2-6; training side only; torch-free).

CLEAN R2 trains on fold k's prepared metallo training subset, and the fold-0 development gap gate
trains one R2-inner arm per inner validation fold J. This step writes those allowlists in the
format the comparator reads (``run_clean_split30_comparator.py prepare --accession-allowlist``:
one accession per line), and per fold a table of the R2 proteins that DeepMzyme does not train on.

Inputs (training side only; exact names; each file through its receipt):
- per fold k, the identity folder ``<identity root>/fold<k>`` (or ``--fold-work K=PATH``):
  ``nodes_metallo_train.tsv`` with the fasta, pinned search and edges receipts it chains to
  (``identity.load_inner_chain``); ``components_inner.tsv`` (its components receipt); the inner
  assignment receipt (its ``excluded_ambiguous_ec_class_accessions``); the certified inner
  membership (its certificate), of which only the accession, group and fold columns are used. A
  folder that holds any complement (test-side) output of the identity audit is refused;
- the cohort audit folder (``ec_clean30.cohort``): ``cohort_audit_receipt.json``, which must record
  a passed audit with no failed check, and a real (non-smoke) one unless this run is a smoke; and,
  only for a fold that the receipt flags as having load-time exclusions and whose table it records
  in ``outputs``, ``cohort_exclusions_fold<k>.tsv`` (the audit's columns ``accession, chain,
  load_status, load_reason``; ``load_status`` is the reason and ``load_reason`` is not read),
  checked against the SHA-256 and row count the receipt records. A ``quarantine.tsv`` stops the
  step. A passed audit has no load-time exclusions, so no table is read until decision A2 accepts
  exclusions; the path is kept so that R2 then keeps those proteins;
- fold 0's ``r1_inner_exclusions_fold0_inner<J>.tsv``, through their receipts.

Rules:
- R2, fold k: every fold-k metallo training protein (every node of the identity chain), so it
  holds the proteins with several EC classes and any load-time exclusion that DeepMzyme drops.
  DeepMzyme's fold-k training proteins are the certified inner membership (checked to be a subset
  of R2). Every R2 protein outside it must be a several-class protein of the assignment receipt or
  one of fold k's cohort exclusions (never both: the audit lists proteins with one EC class);
  ``r2_vs_deepmzyme_fold<k>.tsv`` (``accession, in_deepmzyme, reason``; no EC column) lists those
  proteins. Any other difference, and a cohort exclusion that is still in the membership, is a
  refusal.
- R2-inner, fold 0, inner fold J: proteins follow their inner component. The list holds the R2
  proteins whose component the membership assigns to an inner fold other than J, plus the R2
  proteins whose component holds no eligible protein (several-class proteins, and load-time
  exclusions if any), which are in every list. A list that shares any accession with the R1-inner
  exclusion list of the same J is refused; nothing is subtracted.

Outputs (a new folder, built in a staging folder next to it and then renamed; written once):
``r2_allowlist_fold<k>.txt``, ``r2_inner_allowlist_fold0_inner<J>.txt``,
``r2_vs_deepmzyme_fold<k>.tsv`` and ``r2_lists_receipt.json`` (private: it holds every fold's
counts and absolute paths). Refusal messages name folds, files and counts, never an accession.

The returned summary (what the action prints) gives list sizes and difference counts for fold 0
only; the other folds show their file names, and their sizes stay in the receipt. A fold's R2 size
minus its eligible proteins (which the inner assignment step reports) is its count of several-class
proteins, and per-fold class numbers are shown for one fold only: across five folds they give
test-side counts by subtraction.

Runner action (``scripts/run_ec_clean30_campaign.py``): build the sub-parser with
``add_r2_lists_arguments(sub.add_parser("r2-lists", help=R2_LISTS_HELP))`` and set
``handler=action_r2_lists``. The action refuses test-named paths, requires a clean tree for a real
run, builds the lists, prints the summary and returns 0. A refusal of its inputs (``R2ListError``)
is printed as ``error: ...`` and returns 2; ``HeldOutInputError`` and ``DirtyTreeError`` reach the
runner's own handler (exit 2). Arguments::

    r2-lists --identity-root DIR --cohort DIR --out DIR [--fold-work K=PATH ...] [--smoke]

    --identity-root  folder with one training-side identity folder fold<k> per outer fold
    --cohort         the cohort audit folder (holds cohort_audit_receipt.json)
    --out            new output folder (written once; no part named test, metadata, folds or
                     pinmymetal_files, which the comparator refuses for an allowlist)
    --fold-work      K=PATH: fold K's identity folder instead of <identity root>/fold<K> (repeatable)
    --smoke          engineering use: allow a dirty tree and a smoke cohort audit (recorded)
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Mapping, Optional, Sequence, Union

from ec_clean30 import identity
from ec_clean30.provenance import HeldOutInputError, provenance_record, real_action_git_state, refuse_test_paths
from training.ec_membership import EcFoldMembership

SCHEMA_VERSION = 1
STEP = "r2_lists"
R2_LISTS_HELP = "R2 and fold-0 R2-inner accession allowlists for CLEAN, from the training-side identity folders (P2-6)."
N_FOLDS = identity.N_FOLDS
N_INNER_FOLDS = identity.N_INNER_FOLDS
# The development gap gate runs on fold 0 only, so only fold 0 has R1-inner lists to pair with.
R2_INNER_OUTER_FOLDS: tuple[int, ...] = (0,)
# The printed summary gives sizes and difference counts for these folds only (see the module docstring).
SUMMARY_COUNT_FOLDS: tuple[int, ...] = (0,)

FOLD_WORK_NAME = "fold{fold}"
R2_ALLOWLIST_NAME = "r2_allowlist_fold{fold}.txt"
R2_INNER_ALLOWLIST_NAME = "r2_inner_allowlist_fold{fold}_inner{inner_fold}.txt"
DIFFERENCE_NAME = "r2_vs_deepmzyme_fold{fold}.tsv"
DIFFERENCE_COLUMNS = ("accession", "in_deepmzyme", "reason")
RECEIPT_NAME = "r2_lists_receipt.json"
REASON_SEVERAL_EC_CLASSES = "several_ec_classes"

# Names written by identity.build_r1_inner_exclusions (equality with its output is tested).
R1_INNER_EXCLUSIONS_NAME = "r1_inner_exclusions_fold{fold}_inner{inner_fold}.tsv"
R1_INNER_RECEIPT_NAME = "r1_inner_exclusions_receipt_fold{fold}_inner{inner_fold}.json"
R1_REASONS = (identity.R1_REASON_VALIDATION, identity.R1_REASON_D3)

# Names and values of the cohort audit (ec_clean30.cohort; equality is tested). They are repeated
# here so that this module does not import the audit and the view code.
COHORT_AUDIT_FORMAT = "deepmzyme_ec_clean30_cohort_audit_v1"
COHORT_AUDIT_KIND = "cohort_audit"
COHORT_PASSED = "passed"
COHORT_RECEIPT_NAME = "cohort_audit_receipt.json"
COHORT_EXCLUSIONS_NAME = "cohort_exclusions_fold{fold}.tsv"
COHORT_QUARANTINE_NAME = "quarantine.tsv"
COHORT_EXCLUSION_COLUMNS = ("accession", "chain", "load_status", "load_reason")
COHORT_EXCLUDING_LOAD_STATUSES = frozenset({"no_selected_sites", "selected_sites_without_pocket",
                                            "invalid_structure"})

# Outputs of the identity audit's complement (test-side) steps; a working folder holding any of
# them is not read here. ``strata_fold<k>.tsv`` is added per fold.
COMPLEMENT_OUTPUT_NAMES = (
    f"fasta_receipt_{identity.SET_COMPLEMENT}.json", f"query_{identity.SET_COMPLEMENT}.fasta",
    f"nodes_{identity.SET_COMPLEMENT}.tsv", f"search_{identity.SET_COMPLEMENT}",
    f"edges_receipt_{identity.SET_COMPLEMENT}.json", f"edges_{identity.SET_COMPLEMENT}.tsv", "strata_receipt.json",
)
COMPLEMENT_STRATA_NAME = "strata_fold{fold}.tsv"
# The comparator's held-out rules for an allowlist path (run_clean_split30_comparator.py
# HELD_OUT_NAME_RE and HELD_OUT_DIR_NAMES; equality is tested), applied to the output folder.
COMPARATOR_HELD_OUT_NAME_RE = re.compile(r"(^|[_.\-])test([_.\-]|$)", re.IGNORECASE)
COMPARATOR_HELD_OUT_DIR_NAMES = frozenset({"test", "metadata", "folds", "pinmymetal_files"})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CHECK_NAME_RE = re.compile(r"^[a-z0-9_]+$")

R2_RULE = ("every fold-k metallo training protein (the identity chain's node table), including several-class "
           "proteins and load-time exclusions that DeepMzyme drops")
DIFFERENCE_RULE = ("R2 proteins outside the certified inner membership (DeepMzyme's fold-k training proteins, a "
                   "subset of R2); each is a several-class protein of the assignment receipt (reason "
                   "several_ec_classes) or a cohort exclusion of fold k (reason: its load status), never both, "
                   "otherwise refused; no EC column")
R2_INNER_RULE = ("proteins follow their inner component: R2 proteins whose component is assigned to an inner fold "
                 "other than J, plus R2 proteins whose component holds no eligible protein (in every list)")
OVERLAP_RULE = ("an R2-inner list that shares any accession with the R1-inner exclusion list of the same inner "
                "fold is refused; nothing is subtracted")
COHORT_RULE = ("a passed cohort audit with no failed check (a smoke audit only for a smoke run); an exclusion "
               "table only where the receipt flags load-time exclusions and records the table's SHA-256 and rows")

PathLike = Union[str, os.PathLike]
FoldWork = Union[Mapping[int, PathLike], Sequence[str], None]


class R2ListError(ValueError):
    """The R2-list step refuses its inputs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise R2ListError(message)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise R2ListError(f"{path.name} is not valid JSON: {exc}") from exc


def _shown_name(value: Any) -> str:
    """A receipt value that is safe to show in a message (a plain identifier), otherwise a placeholder."""
    return value if isinstance(value, str) and CHECK_NAME_RE.match(value) else "<not shown>"


# ---------------------------------------------------------------- arguments and paths

def parse_fold_work(values: Optional[Sequence[str]]) -> dict[int, Path]:
    """``K=PATH`` values (repeatable ``--fold-work``) -> {K: PATH}; each fold at most once."""
    overrides: dict[int, Path] = {}
    for value in values or ():
        key, separator, path = str(value).partition("=")
        key, path = key.strip(), path.strip()
        _require(bool(separator) and key.isdigit() and bool(path),
                 f"--fold-work takes K=PATH with K in 0..{N_FOLDS - 1}, got {value!r}")
        fold = int(key)
        _require(0 <= fold < N_FOLDS, f"--fold-work fold {fold} is outside 0..{N_FOLDS - 1}")
        _require(fold not in overrides, f"--fold-work names fold {fold} twice")
        overrides[fold] = Path(path)
    return overrides


def fold_work_dirs(identity_root: PathLike, fold_work: FoldWork = None) -> dict[int, Path]:
    """Identity folder of every outer fold: ``<identity root>/fold<k>`` unless ``fold_work`` maps k elsewhere."""
    if isinstance(fold_work, Mapping):
        try:
            overrides = {int(fold): Path(path) for fold, path in fold_work.items()}
        except (TypeError, ValueError) as exc:
            raise R2ListError(f"Fold-work folds must be integers in 0..{N_FOLDS - 1}") from exc
        _require(len(overrides) == len(fold_work), "Fold-work names a fold twice")
        _require(all(0 <= fold < N_FOLDS for fold in overrides), f"Fold-work folds must be in 0..{N_FOLDS - 1}")
    else:
        overrides = parse_fold_work(fold_work)
    works = {fold: overrides.get(fold, Path(identity_root) / FOLD_WORK_NAME.format(fold=fold))
             for fold in range(N_FOLDS)}
    resolved = [work.resolve() for work in works.values()]
    _require(len(set(resolved)) == len(resolved), "Two folds name the same identity folder")
    return works


def add_r2_lists_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The ``r2-lists`` action's arguments (see the module docstring); the handler is ``action_r2_lists``."""
    parser.add_argument("--identity-root", required=True,
                        help="Identity root with one training-side folder fold<k> per outer fold.")
    parser.add_argument("--cohort", required=True, help=f"Cohort audit folder (holds {COHORT_RECEIPT_NAME}).")
    parser.add_argument("--out", required=True, help="New output folder (written once).")
    parser.add_argument("--fold-work", action="append", default=[], metavar="K=PATH",
                        help="Identity folder of fold K instead of <identity root>/fold<K>, e.g. a rerun "
                             "folder (repeatable).")
    parser.add_argument("--smoke", action="store_true",
                        help="Engineering use: allow a dirty tree and a smoke cohort audit (recorded).")
    return parser


def _check_input_dir(path: Path, what: str) -> Path:
    refuse_test_paths([path], what=what)
    _require(path.is_dir(), f"{what} not found: {path}")
    return path


def _check_output_dir(out: PathLike) -> Path:
    """A new folder whose path the comparator accepts for an allowlist (no held-out part anywhere)."""
    refuse_test_paths([out], what="--out")
    literal = Path(out).absolute()
    for candidate in (literal, literal.resolve()):
        blocked = [part for part in candidate.parts[1:]
                   if COMPARATOR_HELD_OUT_NAME_RE.search(part) or part.lower() in COMPARATOR_HELD_OUT_DIR_NAMES]
        if blocked:
            raise HeldOutInputError(f"--out {out} runs through {blocked}, which the comparator refuses for an "
                                    "accession allowlist")
    _require(not (literal.exists() or literal.is_symlink()),
             f"The output folder exists: {literal}; R2 lists are written once (a rerun takes a new folder)")
    return literal


# ---------------------------------------------------------------- cohort exclusions

def check_cohort_receipt(receipt: Any, *, smoke: bool = False) -> dict[str, Any]:
    """The audit's status record, after checking that ``receipt`` is a passed cohort audit.

    A smoke audit (``smoke: true``) is accepted only when this run is a smoke run too.
    """
    _require(isinstance(receipt, Mapping) and receipt.get("format") == COHORT_AUDIT_FORMAT
             and receipt.get("kind") == COHORT_AUDIT_KIND and isinstance(receipt.get("outputs"), Mapping),
             f"{COHORT_RECEIPT_NAME} is not a cohort audit receipt ({COHORT_AUDIT_FORMAT}, kind {COHORT_AUDIT_KIND})")
    status, failed, audit_smoke = receipt.get("status"), receipt.get("failed_checks"), receipt.get("smoke")
    _require(isinstance(status, str) and isinstance(failed, list) and all(isinstance(name, str) for name in failed)
             and isinstance(audit_smoke, bool), f"{COHORT_RECEIPT_NAME} has no valid status, failed checks or "
                                                "smoke flag")
    _require(status == COHORT_PASSED and not failed,
             f"The cohort audit did not pass (status {_shown_name(status)}; {len(failed)} failed check(s) "
             f"{sorted(_shown_name(name) for name in failed)}); R2 lists need a passed audit")
    _require(smoke or not audit_smoke, "The cohort audit is an engineering smoke; real R2 lists need a real "
                                       "(non-smoke) cohort audit")
    return {"status": status, "failed_checks": list(failed), "smoke": audit_smoke}


def _cohort_output(receipt: Mapping[str, Any], name: str) -> Optional[Mapping[str, Any]]:
    """The receipt's ``outputs`` entry for ``name`` (the audit keys its outputs by file name), or None."""
    entry = receipt["outputs"].get(name)
    if entry is None:
        return None
    _require(isinstance(entry, Mapping) and isinstance(entry.get("sha256"), str)
             and bool(SHA256_RE.match(entry["sha256"])), f"{COHORT_RECEIPT_NAME} records no valid SHA-256 for {name}")
    return entry


def _read_cohort_exclusions(path: Path) -> dict[str, str]:
    """accession -> load status of one cohort exclusions table (``chain`` and ``load_reason`` are not read)."""
    reasons: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        _require(tuple(reader.fieldnames or ()) == COHORT_EXCLUSION_COLUMNS,
                 f"{path.name} does not have the cohort audit's tab-separated columns {COHORT_EXCLUSION_COLUMNS}")
        for line, row in enumerate(reader, start=2):
            _require(None not in row and None not in row.values(), f"Malformed row {line} of {path.name}")
            accession, status = row["accession"], row["load_status"]
            _require(bool(identity.ACCESSION_RE.match(accession)), f"Invalid accession in row {line} of {path.name}")
            _require(accession not in reasons, f"Repeated accession in row {line} of {path.name}")
            _require(status in COHORT_EXCLUDING_LOAD_STATUSES,
                     f"Row {line} of {path.name} has a load status that is not a load-time exclusion "
                     f"(expected one of {sorted(COHORT_EXCLUDING_LOAD_STATUSES)})")
            reasons[accession] = status
    return reasons


def load_cohort_exclusions(cohort_dir: PathLike, *, smoke: bool = False,
                           ) -> tuple[dict[int, dict[str, str]], dict[str, Any]]:
    """Per fold accession -> load status of the cohort audit's exclusions, and the record of what was read.

    The receipt must be a passed audit (``check_cohort_receipt``). A fold's table is read only when
    the receipt's ``per_fold`` record flags load-time exclusions for it and ``outputs`` records the
    table by file name; its SHA-256 and row count must match. A table the receipt does not record,
    a recorded table for a fold flagged without exclusions, and a flag without a table are refused.
    """
    cohort_dir = Path(cohort_dir)
    receipt_path = cohort_dir / COHORT_RECEIPT_NAME
    _require(receipt_path.is_file() and not receipt_path.is_symlink(),
             f"Run the cohort audit first: {receipt_path} is missing or a link")
    receipt = _read_json(receipt_path)
    audit = check_cohort_receipt(receipt, smoke=smoke)
    _require(not os.path.lexists(cohort_dir / COHORT_QUARANTINE_NAME)
             and _cohort_output(receipt, COHORT_QUARANTINE_NAME) is None,
             f"The cohort audit quarantined EC annotation differences ({COHORT_QUARANTINE_NAME}); "
             "R2 lists wait until they are resolved")
    per_fold = receipt.get("per_fold")
    _require(isinstance(per_fold, Mapping), f"{COHORT_RECEIPT_NAME} has no per-fold record")
    exclusions: dict[int, dict[str, str]] = {}
    tables: dict[str, Optional[dict[str, Any]]] = {}
    for fold in range(N_FOLDS):
        name = COHORT_EXCLUSIONS_NAME.format(fold=fold)
        path = cohort_dir / name
        fold_record = per_fold.get(str(fold))
        none_flagged = fold_record.get("no_load_time_exclusions") if isinstance(fold_record, Mapping) else None
        _require(isinstance(none_flagged, bool),
                 f"{COHORT_RECEIPT_NAME} has no load-time exclusion flag for fold {fold}")
        entry = _cohort_output(receipt, name)
        if entry is None:
            _require(not os.path.lexists(path), f"{name} exists but {COHORT_RECEIPT_NAME} does not record it")
            _require(none_flagged, f"{COHORT_RECEIPT_NAME} flags load-time exclusions for fold {fold} but records "
                                   f"no {name}")
            exclusions[fold], tables[str(fold)] = {}, None
            continue
        _require(not none_flagged, f"{COHORT_RECEIPT_NAME} records {name} but flags fold {fold} as having no "
                                   "load-time exclusions")
        _require(path.is_file() and not path.is_symlink(), f"{name} is recorded by {COHORT_RECEIPT_NAME} but is "
                                                           "missing or a link")
        _require(identity.sha256_file(path) == entry["sha256"], f"{name} changed after the cohort audit")
        exclusions[fold] = _read_cohort_exclusions(path)
        _require(bool(exclusions[fold]) and entry.get("n_rows") == len(exclusions[fold]),
                 f"{name} is empty or holds another number of rows than {COHORT_RECEIPT_NAME} records")
        tables[str(fold)] = {"file": name, "sha256": entry["sha256"], "n_rows": len(exclusions[fold])}
    record = {"path": str(cohort_dir.absolute()), "receipt": COHORT_RECEIPT_NAME,
              "receipt_sha256": identity.sha256_file(receipt_path), **audit, "exclusion_tables": tables}
    return exclusions, record


# ---------------------------------------------------------------- identity inputs

@dataclass(frozen=True)
class FoldInputs:
    """Verified training-side identity inputs of one outer fold."""
    fold: int
    work: Path
    chain: identity.InnerChain
    nodes: frozenset[str]
    components: Mapping[str, str]
    membership: EcFoldMembership
    certificate_sha256: str
    inner_fold_of: Mapping[str, int]
    group_fold: Mapping[str, int]
    several: frozenset[str]
    record: Mapping[str, Any]


def complement_output_names(fold: int) -> tuple[str, ...]:
    """Names of the identity audit's complement (test-side) outputs in fold ``fold``'s working folder."""
    return (*COMPLEMENT_OUTPUT_NAMES, COMPLEMENT_STRATA_NAME.format(fold=fold))


def load_fold_inputs(work: PathLike, fold: int) -> FoldInputs:
    """One fold's node table, inner components, assignment receipt and certified membership, all re-verified."""
    work = Path(work)
    held = [name for name in complement_output_names(fold) if os.path.lexists(work / name)]
    _require(not held, f"The fold-{fold} identity folder {work} holds complement (test-side) outputs {held}; "
                       "R2 lists read training-side folders only")
    try:
        chain = identity.load_inner_chain(work, fold)
        components, components_receipt_sha = identity._load_inner_components(work, fold, chain)
        assignment = identity._load_assignment_receipt(work, fold)
        membership, certificate_sha = identity._load_certified_membership(work, fold, chain)
        node_rows = identity.read_tsv(work / chain.inputs["outputs"]["nodes"]["file"], identity.NODE_COLUMNS)
    except ValueError as exc:
        raise R2ListError(f"Fold {fold} ({work}): {exc}") from exc
    record = chain.record()
    _require({key: assignment["inputs"].get(key) for key in record} == record,
             f"Fold {fold}: {identity.INNER_ASSIGNMENT_RECEIPT} was built from other identity tables")
    _require(assignment["inputs"].get("components_receipt_sha256") == components_receipt_sha,
             f"Fold {fold}: {identity.INNER_ASSIGNMENT_RECEIPT} was built from another components step")
    several_list = assignment.get("excluded_ambiguous_ec_class_accessions")
    _require(isinstance(several_list, list) and len(set(several_list)) == len(several_list),
             f"Fold {fold}: {identity.INNER_ASSIGNMENT_RECEIPT} has no valid several-class list")
    nodes = frozenset(row["accession"] for row in node_rows)
    _require(len(nodes) == len(node_rows), f"Fold {fold}: the node table repeats an accession")
    _require(set(components) == nodes,
             f"Fold {fold}: components_{identity.SCOPE_INNER}.tsv and the node table name different proteins")
    inner_fold_of: dict[str, int] = {}
    group_fold: dict[str, int] = {}
    for row in membership.rows:
        accession, group, inner = row["accession"], row["group_id"], int(row["fold"])
        _require(0 <= inner < N_INNER_FOLDS, f"Fold {fold}: the membership uses an inner fold outside "
                                             f"0..{N_INNER_FOLDS - 1}")
        _require(components.get(accession) == group,
                 f"Fold {fold}: a membership group differs from the protein's inner component")
        _require(group_fold.setdefault(group, inner) == inner, f"Fold {fold}: a component spans two inner folds")
        inner_fold_of[accession] = inner
    outside = set(inner_fold_of) - nodes
    _require(not outside, f"Fold {fold}: {len(outside)} membership protein(s) are not metallo training proteins")
    _require(set(group_fold.values()) == set(range(N_INNER_FOLDS)), f"Fold {fold}: an inner fold has no protein")
    several = frozenset(several_list)
    _require(several <= nodes, f"Fold {fold}: {len(several - nodes)} several-class accession(s) of the assignment "
                               "receipt are not metallo training proteins")
    _require(not several & set(inner_fold_of),
             f"Fold {fold}: {len(several & set(inner_fold_of))} several-class protein(s) are in the membership")
    inputs = {
        **record,
        "fasta_receipt_sha256": identity.sha256_file(work / f"fasta_receipt_{identity.SET_METALLO_TRAIN}.json"),
        "components_receipt_sha256": components_receipt_sha,
        "components_sha256": identity.sha256_file(work / f"components_{identity.SCOPE_INNER}.tsv"),
        "assignment_receipt_sha256": identity.sha256_file(work / identity.INNER_ASSIGNMENT_RECEIPT),
        "certificate_sha256": certificate_sha,
        "membership_sha256": membership.file_sha256,
        "membership_identity_sha256": membership.identity_sha256,
        "pinned_binary": chain.search["pinned_binary"],
    }
    return FoldInputs(fold=fold, work=work, chain=chain, nodes=nodes, components=dict(components),
                      membership=membership, certificate_sha256=certificate_sha, inner_fold_of=inner_fold_of,
                      group_fold=group_fold, several=several, record=inputs)


def load_r1_inner_exclusions(inputs: FoldInputs, inner_fold: int) -> tuple[dict[str, str], dict[str, Any]]:
    """accession -> reason of the R1-inner exclusion list of one inner fold, built from this fold's chain."""
    fold, work = inputs.fold, inputs.work
    name = R1_INNER_EXCLUSIONS_NAME.format(fold=fold, inner_fold=inner_fold)
    receipt_name = R1_INNER_RECEIPT_NAME.format(fold=fold, inner_fold=inner_fold)
    receipt_path = work / receipt_name
    _require(receipt_path.is_file(), f"Run r1-inner-exclusions for fold {fold}, inner fold {inner_fold} first: "
                                     f"{receipt_path} is missing")
    receipt = _read_json(receipt_path)
    _require(receipt.get("step") == "r1_inner_exclusions" and receipt.get("fold") == fold
             and receipt.get("inner_fold") == inner_fold, f"{receipt_name} belongs to another fold or inner fold")
    _require(receipt.get("freeze1_sha256") == inputs.chain.freeze1_sha256,
             f"{receipt_name} records another Freeze 1 value than its identity chain")
    _require(receipt.get("pinned_binary") is True, f"{receipt_name} comes from an unpinned search")
    recorded = receipt.get("inputs") or {}
    chain_record = inputs.chain.record()
    _require({key: recorded.get(key) for key in chain_record} == chain_record,
             f"{receipt_name} was built from other identity tables")
    _require(recorded.get("membership_sha256") == inputs.membership.file_sha256
             and recorded.get("certificate_sha256") == inputs.certificate_sha256,
             f"{receipt_name} was built from another certified membership")
    output = (receipt.get("outputs") or {}).get(name)
    _require(isinstance(output, Mapping) and isinstance(output.get("sha256"), str), f"{receipt_name} does not "
                                                                                    f"name {name}")
    path = work / name
    _require(path.is_file() and identity.sha256_file(path) == output["sha256"],
             f"{name} is missing or changed after the r1-inner-exclusions step")
    try:
        rows = identity.read_tsv(path, identity.R1_EXCLUSION_COLUMNS)
    except ValueError as exc:
        raise R2ListError(f"{name}: {exc}") from exc
    reasons: dict[str, str] = {}
    for line, row in enumerate(rows, start=2):
        _require(row["reason"] in R1_REASONS and row["accession"] not in reasons,
                 f"Unknown reason or repeated accession at {name}:{line}")
        reasons[row["accession"]] = row["reason"]
    record = {"file": name, "sha256": output["sha256"], "receipt": receipt_name,
              "receipt_sha256": identity.sha256_file(receipt_path), "n_rows": len(rows)}
    return reasons, record


# ---------------------------------------------------------------- the lists

def r2_difference(inputs: FoldInputs, cohort_exclusions: Mapping[str, str]) -> tuple[list[str], list[dict[str, str]]]:
    """The fold's R2 accessions and its difference rows (R2 proteins DeepMzyme does not train on)."""
    fold = inputs.fold
    deepmzyme = set(inputs.inner_fold_of)
    excluded = set(cohort_exclusions)
    outside = excluded - inputs.nodes
    _require(not outside, f"Fold {fold}: {len(outside)} cohort exclusion(s) are not fold-{fold} metallo training "
                          "proteins")
    both = excluded & inputs.several
    _require(not both, f"Fold {fold}: {len(both)} cohort exclusion(s) are several-class proteins of the assignment "
                       "receipt; the cohort audit excludes proteins with one EC class only")
    still = excluded & deepmzyme
    _require(not still, f"Fold {fold}: {len(still)} load-time exclusion(s) of the cohort audit are still in the "
                        "certified inner membership; the inner steps must take the cohort exclusions first")
    unexplained = inputs.nodes - deepmzyme - inputs.several - excluded
    _require(not unexplained, f"Fold {fold}: {len(unexplained)} metallo training protein(s) are neither in the "
                              "certified inner membership nor excluded by the assignment receipt or the cohort audit")
    rows = [{"accession": accession, "in_deepmzyme": "0",
             "reason": REASON_SEVERAL_EC_CLASSES if accession in inputs.several else cohort_exclusions[accession]}
            for accession in sorted(inputs.nodes - deepmzyme)]
    return sorted(inputs.nodes), rows


def r2_inner_accessions(inputs: FoldInputs, inner_fold: int) -> list[str]:
    """R2 proteins whose inner component is not assigned to ``inner_fold`` (unassigned components included)."""
    return sorted(accession for accession in inputs.nodes
                  if inputs.group_fold.get(inputs.components[accession]) != inner_fold)


def _allowlist_text(accessions: Sequence[str], name: str) -> str:
    _require(bool(accessions), f"{name} would be empty")
    _require(all(identity.ACCESSION_RE.match(accession) for accession in accessions),
             f"{name} would hold an invalid accession")
    return "".join(f"{accession}\n" for accession in accessions)


def _write_new(path: Path, text: str) -> str:
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(text)
    return identity.sha256_text(text)


def read_allowlist(path: PathLike) -> list[str]:
    """Accessions of an allowlist written here: one per line, valid, unique, at least one."""
    accessions = Path(path).read_text(encoding="utf-8").splitlines()
    _require(bool(accessions) and all(identity.ACCESSION_RE.match(accession) for accession in accessions)
             and len(set(accessions)) == len(accessions), f"{Path(path).name} is not a valid accession allowlist")
    return accessions


def _check_provenance(provenance: Any, smoke: bool) -> None:
    _require(isinstance(provenance, Mapping), "The R2-list build needs the action's provenance record")
    _require(bool(provenance.get("smoke", smoke)) == bool(smoke),
             "The provenance record's smoke flag differs from this run's")
    git = provenance.get("git")
    _require(smoke or not (isinstance(git, Mapping) and git.get("clean") is False),
             "A real R2-list build needs a clean tree (the provenance record shows uncommitted or untracked "
             "changes); engineering runs use --smoke")


def build_r2_lists(identity_root: PathLike, cohort_dir: PathLike, out_dir: PathLike, *,
                   fold_work: FoldWork = None, provenance: Mapping[str, Any], smoke: bool = False) -> dict[str, Any]:
    """Write the R2 allowlists, fold 0's R2-inner allowlists, the difference tables and a receipt.

    ``fold_work`` maps folds to other identity folders (``K=PATH`` strings or {K: PATH}).
    ``provenance`` is the action's provenance record (``provenance_record``); ``smoke`` marks an
    engineering run, which may read a smoke cohort audit. The caller checks the tree
    (``action_r2_lists`` does); a real run refuses a provenance record that shows a dirty tree.
    Returns the summary that is safe to print (see the module docstring).
    """
    smoke = bool(smoke)
    _check_provenance(provenance, smoke)
    identity_root, cohort_dir = Path(identity_root).absolute(), Path(cohort_dir).absolute()
    refuse_test_paths([identity_root], what="--identity-root")
    works = {fold: Path(work).absolute() for fold, work in fold_work_dirs(identity_root, fold_work).items()}
    overrides = {fold for fold, work in works.items() if work != identity_root / FOLD_WORK_NAME.format(fold=fold)}
    for fold, work in works.items():
        _check_input_dir(work, f"The fold-{fold} identity folder")
    _check_input_dir(cohort_dir, "--cohort")
    out = _check_output_dir(out_dir)

    exclusions, cohort_record = load_cohort_exclusions(cohort_dir, smoke=smoke)
    folds = {fold: load_fold_inputs(work, fold) for fold, work in works.items()}
    texts: dict[str, str] = {}
    fold_records: dict[str, Any] = {}
    summary_lists: dict[str, Any] = {}
    for fold, inputs in folds.items():
        r2, difference = r2_difference(inputs, exclusions[fold])
        list_name, table_name = R2_ALLOWLIST_NAME.format(fold=fold), DIFFERENCE_NAME.format(fold=fold)
        texts[list_name] = _allowlist_text(r2, list_name)
        texts[table_name] = identity._tsv_text(DIFFERENCE_COLUMNS, difference)
        by_reason: dict[str, int] = {}
        for row in difference:
            by_reason[row["reason"]] = by_reason.get(row["reason"], 0) + 1
        unassigned = [accession for accession in inputs.nodes if inputs.components[accession] not in inputs.group_fold]
        counts = {"r2_proteins": len(r2), "deepmzyme_proteins": len(inputs.inner_fold_of),
                  "not_in_deepmzyme": len(difference), "not_in_deepmzyme_by_reason": dict(sorted(by_reason.items())),
                  "proteins_in_components_without_eligible_proteins": len(unassigned)}
        fold_records[str(fold)] = {
            "work": str(inputs.work), "work_override": fold in overrides, "inputs": dict(inputs.record),
            "cohort_exclusion_table": cohort_record["exclusion_tables"][str(fold)],
            "counts": counts,
            "outputs": {"r2_allowlist": list_name, "difference_table": table_name},
        }
        if fold in SUMMARY_COUNT_FOLDS:
            summary_lists[str(fold)] = {"file": list_name, "n_accessions": len(r2),
                                        **{key: counts[key] for key in ("deepmzyme_proteins", "not_in_deepmzyme",
                                                                        "not_in_deepmzyme_by_reason")}}
        else:
            summary_lists[str(fold)] = {"file": list_name, "size_in_receipt": True}

    inner_records: dict[str, Any] = {}
    summary_inner: dict[str, Any] = {}
    for fold in R2_INNER_OUTER_FOLDS:
        inputs = folds[fold]
        inner_records[str(fold)], summary_inner[str(fold)] = {}, {}
        for inner_fold in range(N_INNER_FOLDS):
            r1, r1_record = load_r1_inner_exclusions(inputs, inner_fold)
            validation = {accession for accession, inner in inputs.inner_fold_of.items() if inner == inner_fold}
            _require({accession for accession, reason in r1.items() if reason == identity.R1_REASON_VALIDATION}
                     == validation, f"{r1_record['file']} lists other inner-validation proteins than the certified "
                                    "membership")
            accessions = r2_inner_accessions(inputs, inner_fold)
            overlap = set(accessions) & set(r1)
            _require(not overlap, f"Fold {fold}, inner fold {inner_fold}: the R2-inner list would share "
                                  f"{len(overlap)} accession(s) with {r1_record['file']}; refused (nothing is "
                                  "subtracted)")
            name = R2_INNER_ALLOWLIST_NAME.format(fold=fold, inner_fold=inner_fold)
            texts[name] = _allowlist_text(accessions, name)
            inner_records[str(fold)][str(inner_fold)] = {
                "r1_inner_exclusions": r1_record, "output": name,
                "counts": {"r2_inner_proteins": len(accessions), "inner_validation_proteins": len(validation),
                           "overlap_with_r1_inner_exclusions": 0},
            }
            entry: dict[str, Any] = {"file": name, "overlap_with_r1_inner_exclusions": 0}
            if fold in SUMMARY_COUNT_FOLDS:
                entry["n_accessions"] = len(accessions)
            else:
                entry["size_in_receipt"] = True
            summary_inner[str(fold)][str(inner_fold)] = entry

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "smoke": smoke,
        "rules": {"r2": R2_RULE, "difference": DIFFERENCE_RULE, "r2_inner": R2_INNER_RULE, "overlap": OVERLAP_RULE,
                  "cohort": COHORT_RULE},
        "n_folds": N_FOLDS,
        "n_inner_folds": N_INNER_FOLDS,
        "r2_inner_outer_folds": list(R2_INNER_OUTER_FOLDS),
        "identity_root": str(identity_root),
        "cohort": cohort_record,
        "folds": fold_records,
        "r2_inner": inner_records,
        "labels_read": True,
        "label_use": ("training side only: the inner assignment receipt's several-class list (from fold k's "
                      "training structure stems) gives a difference reason; the membership's EC classes are read "
                      "by its loader and not used"),
        "ec_column_written": False,
        "code_sha256": {"src/ec_clean30/r2_lists.py": identity.sha256_file(Path(__file__).resolve()),
                        **identity.code_sha256()},
        "provenance": dict(provenance),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.staging-", dir=out.parent))
    try:
        receipt["outputs"] = {name: {"sha256": _write_new(staging / name, text),
                                     "n_records": text.count("\n") - (1 if name.endswith(".tsv") else 0)}
                              for name, text in sorted(texts.items())}
        receipt_sha = _write_new(staging / RECEIPT_NAME, identity._json_text(receipt))
        try:
            os.chmod(staging, 0o755)  # mkdtemp makes the folder 0700; a drive that ignores modes keeps its own
        except OSError:
            pass
        _require(not (out.exists() or out.is_symlink()), f"The output folder appeared while writing: {out}")
        os.rename(staging, out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"step": STEP, "smoke": smoke, "out": str(out), "receipt": RECEIPT_NAME, "receipt_sha256": receipt_sha,
            "cohort_audit_status": cohort_record["status"],
            "r2_allowlists": summary_lists, "r2_inner_allowlists": summary_inner,
            "checks": {"every_difference_explained": True, "r2_inner_overlap_with_r1_inner_exclusions": 0},
            "note": (f"sizes and difference counts for fold(s) {list(SUMMARY_COUNT_FOLDS)} only; the other folds' "
                     "sizes stay in the receipt")}


def action_r2_lists(args: argparse.Namespace) -> int:
    """The ``r2-lists`` action: test-path refusals, the clean-tree check (real runs), the build and its summary.

    Returns 0 after a build and 2 when the inputs are refused (``R2ListError``, printed as
    ``error: ...``). ``HeldOutInputError`` and ``DirtyTreeError`` propagate to the runner.
    """
    for name in ("identity_root", "cohort", "out"):
        refuse_test_paths([getattr(args, name, None)], what=f"--{name.replace('_', '-')}")
    for value in args.fold_work or ():
        refuse_test_paths([value, str(value).partition("=")[2] or None], what="--fold-work")
    try:
        parse_fold_work(args.fold_work)
        git = real_action_git_state(smoke=args.smoke)
        summary = build_r2_lists(args.identity_root, args.cohort, args.out, fold_work=args.fold_work,
                                 provenance=provenance_record(smoke=args.smoke, git=git), smoke=args.smoke)
    except R2ListError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


def verify_r2_lists(out_dir: PathLike) -> dict[str, Any]:
    """Rehash every output the receipt names and check the allowlists; returns the receipt's SHA-256."""
    out = Path(out_dir)
    receipt_path = out / RECEIPT_NAME
    _require(receipt_path.is_file(), f"{receipt_path} is missing")
    receipt = _read_json(receipt_path)
    _require(receipt.get("step") == STEP and receipt.get("schema_version") == SCHEMA_VERSION,
             f"{RECEIPT_NAME} is not an R2-list receipt of schema {SCHEMA_VERSION}")
    expected = {R2_ALLOWLIST_NAME.format(fold=fold) for fold in range(N_FOLDS)}
    expected |= {DIFFERENCE_NAME.format(fold=fold) for fold in range(N_FOLDS)}
    expected |= {R2_INNER_ALLOWLIST_NAME.format(fold=fold, inner_fold=inner)
                 for fold in receipt.get("r2_inner_outer_folds", ()) for inner in range(N_INNER_FOLDS)}
    outputs = receipt.get("outputs") or {}
    _require(set(outputs) == expected, f"{RECEIPT_NAME} names other outputs than the R2-list step writes")
    for name, output in sorted(outputs.items()):
        path = out / name
        _require(path.is_file() and not path.is_symlink(), f"{name} is missing or a link")
        _require(identity.sha256_file(path) == output.get("sha256"), f"{name} changed after the r2-lists step")
        if name.endswith(".txt"):
            _require(len(read_allowlist(path)) == output.get("n_records"), f"{name} has another size than recorded")
        else:
            _require(len(identity.read_tsv(path, DIFFERENCE_COLUMNS)) == output.get("n_records"),
                     f"{name} has another size than recorded")
    return {"status": "verified", "receipt_sha256": identity.sha256_file(receipt_path), "n_files": len(outputs)}
