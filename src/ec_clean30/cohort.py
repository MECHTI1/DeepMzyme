"""Training-side cohort audit of the EC class versus CLEAN30 campaign (step P2-3; torch-free at import).

The audit reads the five training sides of the metallo cohort (the union is 740 proteins on
the real data) once and checks, per protein and with the trainer's own functions, what a
fold's fits will see before any inner assignment is made:

- **load status, mapped as the trainer maps it** (``training.structure_loading``,
  ``training.data``): ``ok``, ``no_selected_sites``, ``selected_sites_without_pocket``,
  ``invalid_structure`` (with a reason) or ``ambiguous_ec_class``; ``fatal_load_error`` marks an
  outcome that stops a whole trainer run (an unsupported site metal), and
  ``structure_input_error`` a structure file that is missing, a symlink or differs from its
  manifest (a view build refuses it). Parsing and site selection use
  ``graph.structure_parsing`` and ``training.site_filter`` (key mode ``accession_ec``). The
  ESMC and external-feature checks predict the trainer's load-time alignment without calling
  the trainer's loader on the shared roots (it lists them);
- **selected sites (A3):** each selected site resolves to one metal atom of the element in
  ``metaltype`` within the coordinate tolerance of ``single_donor_coordinate_{x,y,z}``;
- **extraction counts** of the selected-site report (metal records, pockets, clusters, split
  components, residue change, sites without a pocket), with the fits' pocket grouping
  (``SELECTED_SITE_POCKET_GROUPING``: one pocket per protein from all its selected sites, so
  ``pocket_count`` is 1 for every loaded protein; the D2 pocket residues are the same union of
  residues near any selected metal under either grouping);
- **D2 geometry:** per unselected metal its distance to the nearest selected metal (2.0 Å or less
  flags a copy of the same site from another donor) and to the nearest pocket residue; per pocket
  residue the unselected metals' contribution to ``custom_charge_distance_proxy``, recomputed with
  the feature module's own residue geometry, pair term and metal formula
  (``feature_extraction.core``), and the parity of the recomputed value with the stored one.
  PROPKA (``dpka_titr``) also saw the unselected metals; that is flagged, not recomputed;
- **sequences:** the chain sequence (``embed_helpers.esmc.extract_chain_sequences``) against the
  ESMC sidecar's SHA-256 and the CLEAN sequence map (substitutions, or a length change), and the
  count of X residues;
- **inputs:** exact-name ``lstat`` of ``<esm>/<stem>_chain_<chain>_esmc.pt``, its ``.json`` sidecar
  and ``<external>/<stem>/residue_features.json`` (regular files only; a symlink is never read).
  The payload is read once for its SHA-256, the trainer's alignment rules and the chain-means
  builder's validation (``embed_helpers.esmc_chain_means.mean_from_residue_payload``); the sidecar
  model, width and sequence SHA-256; the external file's schema, structure ID, residue keys, values
  and PROPKA tooling. String values that name a test or mixed folder (A13) are counted, never shown;
- **labels (A4):** the EC numbers in the structure name against the native ones in the label table:
  equal, order only (allowed) or different (written to ``quarantine.tsv``; the audit then stops).

Inputs, by exact name only (no folder is listed and no test-side file is opened): the five
training manifests ``<data>/CLEAN_30_train_test_split_<k>/train/structure_manifest.csv`` and site
tables ``<data>/CLEAN_30_main/folds/CLEAN_30_train_test_split_<k>_train.csv`` through
``ec_clean30.views.read_training_sources`` (one structure name and SHA-256 per protein, and
identical site rows in every file that holds it, are required here); the structures through each
manifest's ``relative_path``; the per-structure files above; the native label table and the CLEAN
sequence map, each checked against its SHA-256 before it is parsed.

Outputs, written once into ``<campaign root>/cohort`` (the campaign root's folder name is the
campaign ID; ``cohort`` is one of the campaign folders every fit's read guard covers, so a copy
anywhere else, such as inside a view, is refused). The files are assembled in a staging folder
``cohort/.partial-<pid>`` and moved up one by one, the receipt last: a ``cohort`` folder without
the receipt is incomplete. After a hard kill (for example by the memory cap) the staging folder
stays inside ``cohort/``, still under the read guard; remove the whole ``cohort`` folder before a
rerun (the audit refuses a folder that is not empty).

- ``cohort_proteins.tsv``: one row per protein; no EC column, no fold column, no structure name
  (it carries the EC numbers);
- ``cohort_sites_label_free.csv`` (``accession, chain_resi, metaltype``): the selected sites, in
  the format of ``training.ec_prediction_export.read_test_site_table``;
- ``d2_unselected_metals.tsv`` and ``d2_pocket_residues.tsv``;
- ``cohort_counts_fold<k>.json``: counts over fold k's training proteins only, with the A1 inputs
  (share of selected-pocket residues with a non-zero unselected-metal contribution, the median
  share among them, the charge-proxy parity count and the same-site copy count) and no EC-class
  breakdown;
- ``cohort_exclusions_fold<k>.tsv`` (label-eligible proteins that would not load), ``quarantine.tsv``
  (A4) and ``site_check_failures.tsv`` (A3), only when not empty;
- ``cohort_audit_receipt.json``: inputs, rules, union totals, checks, output SHA-256 values and
  provenance; per-fold files are bound by SHA-256 and pass/fail flags only.

No table or receipt holds a list or count of union minus a fold's training proteins, or an
EC-class breakdown at union level. Messages and the printed summary carry no accession: per-protein
findings go to the private tables. A refusal raised by a reused reader (the training-source reader,
the site-summary check, the label-table and sequence-map parsers) is shown without its text, because
that text can name a structure of another fold's training file.

The A1 numbers of a fold count only when the stored charge proxy of every selected-pocket residue
equals the recomputed one (check ``charge_proxy_parity_complete``); otherwise the fold's A1 flag and
the recommended option are left undetermined (``None``) and the audit stops.

Runner action (``scripts/run_ec_clean30_campaign.py``): build the sub-parser with
``add_cohort_audit_arguments(sub.add_parser("cohort-audit", help=COHORT_AUDIT_HELP))`` and set
``handler=action_cohort_audit``; ``CohortAuditError`` is a ``ViewError``, so the runner's existing
error handling applies. Arguments::

    cohort-audit --out DIR --label-table FILE --label-table-sha256 SHA
                 --sequence-map FILE --sequence-map-sha256 SHA
                 [--coordinate-tolerance 0.01] [--data-root D] [--esm-dir E] [--external-root X]
                 [--expected-label-eligible-file F --expected-label-eligible-sha256 SHA]
                 [--expected-sequence-difference ACCESSION:DIFFERENCE ...] [--smoke]

    --out                           <campaign root>/cohort, absent or empty; the campaign root's folder
                                    name must be the campaign ID
    --label-table, --label-table-sha256      native_label_table.tsv of the comparator's label-table action
    --sequence-map, --sequence-map-sha256    sequence_map.tsv of the comparator's sequence-map action
    --coordinate-tolerance          A3 tolerance in Å (default and maximum 0.01; a looser value is refused)
    --data-root, --esm-dir, --external-root  as for build-view (defaults: the project's data root and
                                    its esm_embeddings and updated_feature_extraction folders)
    --expected-label-eligible-file, --expected-label-eligible-sha256
                                    a private JSON file holding a list of the five per-fold label-only
                                    eligible counts, and its SHA-256 (both or neither); a mismatch stops.
                                    The counts are per-fold class numbers, so they stay off the command
                                    line and out of the repository; the receipt records only the file's
                                    path and SHA-256, and the summary only a pass/fail flag
    --expected-sequence-difference  a structure-versus-map difference that is expected (repeatable), in the
                                    form the audit reports it, <map residue><position><structure residue>
                                    (e.g. P12345:V113I); any other difference stops
    --smoke                         engineering use: allow a dirty tree (the receipt is marked smoke)

A check that was not asked for (no expected-counts file) is ``None`` and is listed under
``checks_not_run`` in the receipt and the summary. The action exits 0 when every check passes, 1
when the audit stopped (outputs and receipt are written), and raises ``CohortAuditError`` or
``HeldOutInputError`` (the runner exits 2) when an input is refused.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import io
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence
import warnings

from ec_clean30.campaign import CAMPAIGN_ID, N_OUTER_FOLDS
from ec_clean30.identity import DONOR_COLUMN
from ec_clean30.provenance import (
    REPO_ROOT, HeldOutInputError, is_test_named_path, provenance_record, real_action_git_state, sha256_bytes,
    utc_now, write_new_json,
)
from ec_clean30.views import (
    EXTERNAL_FEATURE_FILE, FORBIDDEN_DATA_ROOT_NAMES, SourceLayout, ViewError, esm_file_name, esm_sidecar_name,
    read_training_sources,
)
from training.ec_classes import STATUS_AMBIGUOUS, STATUS_OK, STRUCTURE_ID_RE, normalize_ec_number_list


COHORT_AUDIT_FORMAT = "deepmzyme_ec_clean30_cohort_audit_v1"
# 2 (CP-B): pockets extracted with the structure pocket grouping (SELECTED_SITE_POCKET_GROUPING); a schema-1 audit
# counted the cluster grouping's pockets, so readers that require this version (the overlay builder) refuse it.
COHORT_SCHEMA_VERSION = 2
COHORT_AUDIT_HELP = "Training-side cohort audit over the five training sides (step P2-3)."

# The audit writes only <campaign root>/cohort, a campaign folder that every fit's read guard covers.
COHORT_FOLDER_NAME = "cohort"
STAGING_PREFIX = ".partial-"

PROTEINS_NAME = "cohort_proteins.tsv"
SITES_NAME = "cohort_sites_label_free.csv"
D2_METALS_NAME = "d2_unselected_metals.tsv"
D2_RESIDUES_NAME = "d2_pocket_residues.tsv"
FOLD_COUNTS_NAME = "cohort_counts_fold{fold}.json"
FOLD_EXCLUSIONS_NAME = "cohort_exclusions_fold{fold}.tsv"
QUARANTINE_NAME = "quarantine.tsv"
SITE_FAILURES_NAME = "site_check_failures.tsv"
RECEIPT_NAME = "cohort_audit_receipt.json"

COORDINATE_COLUMNS: tuple[str, ...] = ("single_donor_coordinate_x", "single_donor_coordinate_y",
                                       "single_donor_coordinate_z")
CATALYTIC_COLUMN = "whether_catalytic"
# Columns that must be identical in every training file that holds a protein's site rows.
SITE_IDENTITY_COLUMNS: tuple[str, ...] = ("structure", "chain_resi", "metaltype", "ecnumber", CATALYTIC_COLUMN,
                                          DONOR_COLUMN, *COORDINATE_COLUMNS)

# Decision A3: the selected-site tolerance is 0.01 Å and is never loosened.
A3_COORDINATE_TOLERANCE = 0.01
# The single-donor subset deduplicates sites closer than this (build_clean_single_donor_subset.py).
SAME_SITE_COPY_DISTANCE = 2.0
NEAR_METAL_DISTANCE = 8.0
# The feature module's pair term skips residue pairs whose CA atoms are farther apart (core.py).
RESIDUE_PAIR_CA_CUTOFF = 12.0
CHARGE_PROXY_REL_TOLERANCE = 1e-9
CHARGE_PROXY_ABS_TOLERANCE = 1e-6
# Decision A1 option (b) holds in a fold when both values stay under these limits.
A1_RESIDUE_FRACTION_LIMIT = 0.10
A1_MEDIAN_SHARE_LIMIT = 0.10
# Expected ESMC payloads (embed_helpers.esmc.DEFAULT_ESMC_MODEL_NAME; esm_feature_loading.DEFAULT_ESMC_EMBED_DIM).
ESMC_MODEL_NAME = "esmc_300m"
ESMC_EMBEDDING_DIM = 960
EXTERNAL_SCHEMA_VERSION = 1
LEGACY_EXTERNAL_FEATURES = frozenset({"fa_elec", "biotite_propka_electrostatic_proxy"})

LOAD_OK = "ok"
LOAD_NO_SELECTED_SITES = "no_selected_sites"
LOAD_SITES_WITHOUT_POCKET = "selected_sites_without_pocket"
LOAD_INVALID = "invalid_structure"
LOAD_FATAL = "fatal_load_error"
LOAD_INPUT_ERROR = "structure_input_error"
LOAD_STATUSES: tuple[str, ...] = (LOAD_OK, LOAD_NO_SELECTED_SITES, LOAD_SITES_WITHOUT_POCKET, LOAD_INVALID,
                                  LOAD_FATAL, LOAD_INPUT_ERROR)
TRAINER_STATUSES: tuple[str, ...] = (*LOAD_STATUSES, STATUS_AMBIGUOUS)
# Statuses whose protein a fold's fits would silently skip (a load-time exclusion).
EXCLUDING_LOAD_STATUSES = frozenset({LOAD_NO_SELECTED_SITES, LOAD_SITES_WITHOUT_POCKET, LOAD_INVALID})

A4_EQUAL = "equal"
A4_ORDER_ONLY = "order_only"
A4_DIFFERENT = "different"
A4_MISSING = "missing_from_label_table"

# ESMC and external-feature outcomes the trainer turns into an invalid structure.
ESM_TRAINER_FAILURES = frozenset({"missing", "symlink_refused", "not_regular_file", "payload_unreadable",
                                  "payload_misaligned"})
EXTERNAL_TRAINER_FAILURES = frozenset({"missing", "symlink_refused", "not_regular_file", "unreadable",
                                       "legacy_feature", "empty"})

PROTEIN_COLUMNS: tuple[str, ...] = (
    "accession", "chain", "n_training_files", "structure_sha256", "structure_size_bytes", "esm_sha256",
    "esm_sidecar_sha256", "external_sha256", "load_status", "load_reason", "n_selected_sites", "n_sites_verified",
    "max_site_distance", "metal_records_total", "metal_records_selected", "metal_records_dropped", "pocket_count",
    "selected_cluster_count", "components_with_unselected_metals_count", "split_component_count",
    "residue_count_delta_total", "selected_sites_without_pocket", "n_pocket_residues", "n_unselected_metals",
    "n_same_site_copies", "esm_status", "external_status", "pka_tooling", "propka_sees_unselected_metals",
    "sequence_length", "sequence_sha256", "n_x_residues", "sidecar_sequence_match", "sequence_map_match",
    "sequence_map_difference",
)
SITE_COLUMNS: tuple[str, ...] = ("accession", "chain_resi", "metaltype")
D2_METAL_COLUMNS: tuple[str, ...] = (
    "accession", "chain_resi", "icode", "atom_index", "element", "nearest_selected_distance", "same_site_copy",
    "nearest_pocket_residue_distance", "within_pocket_radius", "within_8a",
)
D2_RESIDUE_COLUMNS: tuple[str, ...] = (
    "accession", "chain", "resseq", "icode", "resname", "stored_proxy", "recomputed_proxy",
    "unselected_contribution", "unselected_share", "parity_ok",
)
EXCLUSION_COLUMNS: tuple[str, ...] = ("accession", "chain", "load_status", "load_reason")
QUARANTINE_COLUMNS: tuple[str, ...] = ("accession", "a4_outcome")
SITE_FAILURE_COLUMNS: tuple[str, ...] = ("accession", "chain_resi", "reason", "distance")
COMPACT_REPORT_KEYS: tuple[str, ...] = (
    "metal_records_total", "metal_records_selected", "metal_records_dropped", "selected_cluster_count",
    "components_with_unselected_metals_count", "split_component_count", "residue_count_delta_total",
)
# The campaign's fits load pockets with --selected-site-pocket-grouping structure (ec_clean30.commands), so the
# audit extracts them the same way (graph.structure_parsing.POCKET_GROUPING_STRUCTURE; kept here torch-free).
SELECTED_SITE_POCKET_GROUPING = "structure"
# Extra per-structure report counts of the structure grouping (training.structure_loading's compact report).
STRUCTURE_GROUPING_REPORT_KEYS: tuple[str, ...] = ("cluster_grouping_cluster_count", "selected_sites_without_residues")

RULES: Mapping[str, str] = {
    "load_status": "the trainer's selected-site path (parse, site keys in accession_ec mode, extraction, ESMC and "
                   "external alignment, pocket-to-site match, metal label); ambiguous_ec_class after a clean load",
    "load_time_exclusion": "a label-eligible protein with load status no_selected_sites, "
                           "selected_sites_without_pocket or invalid_structure",
    "pocket_grouping": "one pocket per protein around all its selected sites (residues within the pocket radius of "
                       "any selected metal; every selected metal), as the fits load it; other metals never enter it",
    "site_check": "one metal atom per selected (chain, resseq), element equal to metaltype, distance to "
                  "single_donor_coordinate_{x,y,z} at most the tolerance",
    "same_site_copy": f"an unselected metal within {SAME_SITE_COPY_DISTANCE} Å of a selected metal",
    "pocket_residue_distance": "minimum distance from the unselected metal to any atom of a selected-pocket residue",
    "unselected_contribution": "sum over metals outside the selected sites of q_residue * q_metal / max(d, 2.0), the "
                               "feature module's formula and constants (feature_extraction.core)",
    "unselected_share": "min(1, |unselected contribution| / |recomputed value with every metal|); 1 when that value "
                        "is 0; 0 when there is no contribution",
    "charge_proxy_parity": f"recomputed value with every metal equals the stored value within rel "
                           f"{CHARGE_PROXY_REL_TOLERANCE} or abs {CHARGE_PROXY_ABS_TOLERANCE}",
    "a1_inputs": "over trainer-ok training proteins of the fold, one row per pocket residue and protein",
    "a1_fraction": "pocket residues with a non-zero unselected contribution over the pocket residues whose "
                   "contribution was recomputed",
    "a1_option_b": f"fraction of pocket residues with a contribution < {A1_RESIDUE_FRACTION_LIMIT} and median "
                   f"share among them < {A1_MEDIAN_SHARE_LIMIT}; option (d) otherwise; undetermined (None) in a "
                   "fold whose charge-proxy parity is incomplete (any pocket residue not compared or not equal)",
    "esm_duplicate_residue_ids": "a payload that repeats a residue ID: the trainer keeps the last row of each ID "
                                 "(training.esm_feature_loading) and still loads it; the chain-means builder refuses "
                                 "it, so it is an input finding, not a load-time exclusion",
    "propka": "flagged when PROPKA ran (tooling.pka = propka) on a file that holds metals outside the selected sites",
    "a4": "EC numbers of the structure name versus the label table's ec_numbers: equal, order_only or different",
    "a13": "string values of the sidecar, the payload and the external file that name a test or mixed folder "
           "(ec_clean30.provenance.names_test_split on every path part); count only",
    "sequence_map_difference": "substitutions as <map residue><1-based position><structure residue> joined by ';', "
                               "or length:<structure length>vs<map length>",
}


class CohortAuditError(ViewError):
    """The cohort audit cannot run from its inputs, or its request is refused."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _real(path: str | os.PathLike) -> str:
    return os.path.normcase(os.path.realpath(os.fspath(path)))


def _under(path: str, roots: Iterable[str]) -> bool:
    return any(path == root or path.startswith(root.rstrip(os.sep) + os.sep) for root in roots)


def _file_problem(path: Path) -> Optional[str]:
    """Why ``path`` cannot be read as a regular file by exact name (``lstat``; a symlink is never followed)."""
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return "missing"
    except NotADirectoryError:
        return "missing"
    if stat.S_ISLNK(info.st_mode):
        return "symlink_refused"
    if not stat.S_ISREG(info.st_mode):
        return "not_regular_file"
    return None


def _dir_problem(path: Path) -> Optional[str]:
    try:
        info = os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        return "missing"
    if stat.S_ISLNK(info.st_mode):
        return "symlink_refused"
    if not stat.S_ISDIR(info.st_mode):
        return "not_regular_file"
    return None


def _withheld(exc: BaseException, what: str) -> CohortAuditError:
    return CohortAuditError(f"{what}: refused by the reader ({type(exc).__name__}); the reader's message is "
                            "withheld because it can name an accession. Run that reader on the file with its "
                            "output sent to a private file to see the detail.")


def _format_distance(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.6f}"


def _format_float(value: Optional[float]) -> str:
    return "" if value is None else repr(float(value))


def _flag(value: Optional[bool]) -> str:
    return "" if value is None else ("1" if value else "0")


def _quantile(values: Sequence[float], q: float) -> Optional[float]:
    """Nearest-rank quantile of sorted ``values`` (None when empty)."""
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return float(ordered[index])


def _median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    return float(ordered[middle]) if len(ordered) % 2 else float((ordered[middle - 1] + ordered[middle]) / 2.0)


def count_test_named_strings(value: Any) -> int:
    """String values (recursively) that name a test or mixed folder in any path part (A13; count only)."""
    if isinstance(value, str):
        return int(is_test_named_path(value, relative_to="/"))
    if isinstance(value, Mapping):
        return sum(count_test_named_strings(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(count_test_named_strings(item) for item in value)
    return 0


def _tsv_bytes(columns: Sequence[str], rows: Iterable[Mapping[str, Any]], *, delimiter: str = "\t") -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(columns), delimiter=delimiter, lineterminator="\n",
                            extrasaction="raise")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    return buffer.getvalue().encode("utf-8")


def _write_new(path: Path, data: bytes) -> str:
    with path.open("xb") as handle:
        handle.write(data)
    return sha256_bytes(data)


def _remove_tree(path: Path) -> None:
    import shutil

    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------------------
# Union of the five training sides
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UnionProtein:
    accession: str
    stem: str
    chain: str
    structure_name: str
    path: Path
    sha256: str
    size_bytes: int
    folds: tuple[int, ...]
    site_rows: tuple[Mapping[str, str], ...]
    label_status: str


@dataclass(frozen=True)
class UnionSources:
    layout: SourceLayout
    folds: Mapping[int, Mapping[str, Any]]
    proteins: Mapping[str, UnionProtein]
    site_labels: Mapping[tuple[str, str, str], str]


def _site_row_key(row: Mapping[str, str]) -> tuple[str, ...]:
    return tuple((row.get(column) or "").strip() for column in SITE_IDENTITY_COLUMNS)


def read_union_sources(layout: SourceLayout) -> UnionSources:
    """Read the five training manifests and site tables; one stem, structure and site-row set per protein.

    Each fold's files go through ``views.read_training_sources`` and the trainer's site-summary
    check (``training.site_filter.validate_selected_site_summary``). A protein must carry the same
    structure name, SHA-256, size and identical site rows (``SITE_IDENTITY_COLUMNS``) in every
    training file that holds it; refusals give counts only.

    The audit reads all five training files, so a structure named in a refusal of fold j's files can
    be a test protein of fold k: every refusal of the reused readers is re-raised without its text
    (a ``HeldOutInputError`` stays one, so the runner still reports it as refused).
    """
    from training.site_filter import load_allowed_site_metal_labels, validate_selected_site_summary

    folds: dict[int, dict[str, Any]] = {}
    proteins: dict[str, dict[str, Any]] = {}
    site_labels: dict[tuple[str, str, str], str] = {}
    conflicts: Counter[str] = Counter()
    for fold in range(N_OUTER_FOLDS):
        try:
            sources = read_training_sources(layout, fold)
        except HeldOutInputError:
            raise HeldOutInputError(f"Fold {fold}'s training sources were refused as test-named (a source file, or "
                                    "a manifest row whose path passes through a test or mixed folder); the "
                                    "reader's message is withheld because it can name a structure.") from None
        except Exception as exc:  # views.py and training.ec_classes messages can quote structure names
            raise _withheld(exc, f"Fold {fold}'s training sources") from None
        missing = [column for column in SITE_IDENTITY_COLUMNS if column not in sources["csv_header"]]
        if missing:
            raise CohortAuditError(f"Fold {fold}'s training site table lacks the columns {missing} that the "
                                   "selected-site check and the cross-file identity check need.")
        try:
            validate_selected_site_summary(sources["csv_path"], key_mode="accession_ec")
            labels = load_allowed_site_metal_labels(sources["csv_path"])
        except Exception as exc:
            raise _withheld(exc, f"Fold {fold}'s training site table") from None
        for key, metal in labels.items():
            if site_labels.setdefault(key, metal) != metal:
                conflicts["site_label"] += 1
        accessions = set()
        for stem, row in sources["sources"].items():
            accession = row.accession
            accessions.add(accession)
            site_rows = tuple(sorted(sources["rows_by_accession"][accession], key=_site_row_key))
            entry = proteins.setdefault(accession, {
                "stem": stem, "structure_name": row.structure_name, "path": row.path, "sha256": row.sha256,
                "size_bytes": row.size_bytes, "site_rows": site_rows, "label_status": sources["status"][stem],
                "folds": [],
            })
            if entry["stem"] != stem or entry["structure_name"] != row.structure_name:
                conflicts["structure_name"] += 1
            if entry["sha256"] != row.sha256 or entry["size_bytes"] != row.size_bytes:
                conflicts["structure_sha256"] += 1
            if [_site_row_key(item) for item in entry["site_rows"]] != [_site_row_key(item) for item in site_rows]:
                conflicts["site_rows"] += 1
            entry["folds"].append(fold)
        folds[fold] = {
            "manifest": {"path": sources["manifest_path"].relative_to(layout.data_root).as_posix(),
                         "sha256": sources["manifest_sha256"], "n_rows": sources["manifest_n_rows"]},
            "training_csv": {"path": sources["csv_path"].relative_to(layout.data_root).as_posix(),
                             "sha256": sources["csv_sha256"], "n_rows": sources["csv_n_rows"],
                             "accessions_sha256": sources["accessions_sha256"]},
            "accessions": frozenset(accessions),
        }
    if conflicts:
        raise CohortAuditError("The training files disagree about proteins they share: "
                               + ", ".join(f"{count} {kind} conflicts" for kind, count in sorted(conflicts.items()))
                               + ". Each protein needs one structure and identical site rows in every training file.")
    union = {}
    for accession, entry in sorted(proteins.items()):
        match = STRUCTURE_ID_RE.match(entry["stem"])
        chain = (match.group("chain").strip() or "_") if match else "_"
        union[accession] = UnionProtein(accession=accession, stem=entry["stem"], chain=chain,
                                        structure_name=entry["structure_name"], path=entry["path"],
                                        sha256=entry["sha256"], size_bytes=int(entry["size_bytes"]),
                                        folds=tuple(entry["folds"]), site_rows=entry["site_rows"],
                                        label_status=entry["label_status"])
    return UnionSources(layout=layout, folds=folds, proteins=union, site_labels=site_labels)


# ---------------------------------------------------------------------------
# Per-protein audit
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Context:
    layout: SourceLayout
    site_labels: Mapping[tuple[str, str, str], str]
    site_index: Any
    tolerance: float
    esm_model_name: str
    esm_dim: int
    map_sequences: Mapping[str, str]
    native_ec: Mapping[str, str]


def _selected_site_rows(protein: UnionProtein) -> list[dict[str, Any]]:
    """The protein's catalytic site rows as the trainer selects them, with their donor coordinates."""
    from training.site_filter import _row_is_catalytic, parse_summary_chain_resi

    rows = []
    for row in protein.site_rows:
        chain_resi = (row.get("chain_resi") or "").strip()
        if not _row_is_catalytic(row, CATALYTIC_COLUMN) or not chain_resi:
            continue
        if not (row.get("structure") or "").strip() or not normalize_ec_number_list(row.get("ecnumber") or ""):
            continue
        chain, resseq = parse_summary_chain_resi(chain_resi)
        coordinate = None
        try:
            coordinate = tuple(float((row.get(column) or "").strip()) for column in COORDINATE_COLUMNS)
        except ValueError:
            coordinate = None
        if coordinate is not None and not all(math.isfinite(value) for value in coordinate):
            coordinate = None
        rows.append({"chain_resi": chain_resi, "site": (chain, resseq),
                     "metaltype": (row.get("metaltype") or "").strip().upper(), "coordinate": coordinate})
    return rows


def _site_checks(protein: UnionProtein, site_rows: Sequence[Mapping[str, Any]], records_by_site: Mapping,
                 tolerance: float) -> tuple[list[dict[str, Any]], list[float]]:
    failures, distances = [], []
    for row in site_rows:
        records = records_by_site.get(row["site"], [])
        reason, distance = None, None
        if not records:
            reason = "unresolved"
        elif len(records) > 1:
            reason = "ambiguous"
        else:
            record = records[0]
            if row["coordinate"] is not None:
                coord = [float(value) for value in record.coord.tolist()]
                distance = math.dist(coord, row["coordinate"])
                distances.append(distance)
            if record.symbol != row["metaltype"]:
                reason = "element_mismatch"
            elif row["coordinate"] is None:
                reason = "no_coordinate"
            elif distance > tolerance:
                reason = "outside_tolerance"
        if reason is not None:
            failures.append({"accession": protein.accession, "chain_resi": row["chain_resi"], "reason": reason,
                             "distance": _format_distance(distance)})
    return failures, distances


def _esm_payload_checks(protein: UnionProtein, ctx: _Context, result: dict[str, Any], *, structure,
                        chain_input) -> None:
    """The payload: the trainer's alignment rules (``esm_feature_loading.load_esm_lookup_for_structure``)
    and the chain-means builder's validation. Sets ``status`` only when it is a payload finding."""
    import torch
    from embed_helpers.esmc_chain_means import mean_from_residue_payload
    from training.esm_feature_loading import embedding_tensor_and_keys_from_payload
    from training.global_esm_features import GlobalEsmCacheError, deserialize_payload

    name = esm_file_name(protein.stem)
    path = protein_layout_path(ctx.layout.esm_dir, name)
    problem = _file_problem(path)
    if problem is not None:
        result["status"] = problem
        return
    data = path.read_bytes()
    result["sha256"] = sha256_bytes(data)
    try:
        payload = deserialize_payload(data, origin="payload")
        embeddings, residue_ids = embedding_tensor_and_keys_from_payload(
            payload, structure=structure, candidate_path=Path(name), fallback_chain_id=protein.chain)
    except Exception:  # the trainer raises several types for unreadable or misshapen payloads
        result["status"] = "payload_unreadable"
        return
    if isinstance(payload, Mapping):
        result["a13"] += count_test_named_strings({key: value for key, value in payload.items()
                                                   if key not in ("embeddings", "residue_ids")})
    # The trainer raises on these two (a load-time failure).
    if embeddings.dim() != 2 or embeddings.size(0) != len(residue_ids):
        result["status"] = "payload_misaligned"
        return
    result["keys"] = frozenset(residue_ids)
    result["width"] = int(embeddings.size(1))
    if len(result["keys"]) != len(residue_ids):
        # Within one file the trainer's lookup keeps the last row of a repeated ID and loads the structure.
        result["status"] = "payload_duplicate_residue_ids"
    elif chain_input is None:
        result["status"] = "payload_invalid"
    else:
        try:
            mean_from_residue_payload(chain_input, data, origin="payload", model_name=ctx.esm_model_name,
                                      embedding_dim=ctx.esm_dim)
        except GlobalEsmCacheError:
            result["status"] = "payload_invalid"
    if result["status"] == "ok" and not bool(torch.isfinite(embeddings).all()):
        result["status"] = "payload_invalid"


def _esm_sidecar_checks(protein: UnionProtein, ctx: _Context, result: dict[str, Any], *, chain_input) -> None:
    """The ``.pt.json`` sidecar: read whenever it exists, so its sequence SHA-256 is always compared."""
    sidecar_path = protein_layout_path(ctx.layout.esm_dir, esm_sidecar_name(protein.stem))
    problem = _file_problem(sidecar_path)
    if problem is not None:
        if result["status"] == "ok":
            result["status"] = "sidecar_missing" if problem == "missing" else f"sidecar_{problem}"
        return
    sidecar_bytes = sidecar_path.read_bytes()
    result["sidecar_sha256"] = sha256_bytes(sidecar_bytes)
    try:
        sidecar = json.loads(sidecar_bytes.decode("utf-8"))
    except ValueError:
        sidecar = None
    if not isinstance(sidecar, Mapping):
        if result["status"] == "ok":
            result["status"] = "sidecar_invalid"
        return
    result["a13"] += count_test_named_strings(sidecar)
    result["sidecar_sequence_sha256"] = sidecar.get("source_sequence_sha256")
    sequence_sha = chain_input.sequence_sha256 if chain_input is not None else None
    if result["status"] == "ok" and (sidecar.get("esm_model_name") != ctx.esm_model_name
                                     or sidecar.get("embedding_dim") != ctx.esm_dim
                                     or sidecar.get("source_sequence_sha256") != sequence_sha
                                     or sidecar.get("n_residues", len(result["keys"])) != len(result["keys"])):
        result["status"] = "sidecar_invalid"


def _esm_checks(protein: UnionProtein, ctx: _Context, *, structure, chain_input) -> dict[str, Any]:
    """ESMC payload and sidecar: trainer alignment, the chain-means builder's validation and the sidecar fields.

    A payload finding takes precedence in ``status``; the sidecar is read either way.
    """
    result: dict[str, Any] = {"status": "ok", "sha256": "", "sidecar_sha256": "", "a13": 0,
                              "sidecar_sequence_sha256": None, "keys": None, "width": None}
    _esm_payload_checks(protein, ctx, result, structure=structure, chain_input=chain_input)
    _esm_sidecar_checks(protein, ctx, result, chain_input=chain_input)
    return result


def protein_layout_path(root: Path, name: str) -> Path:
    """``root/name`` for an exact constructed name; refuses a name that passes through a test or mixed folder."""
    path = Path(root) / name
    if is_test_named_path(path, relative_to=root):
        raise HeldOutInputError(f"A constructed input name under {root} names a test or mixed folder; refused.")
    return path


def _external_checks(protein: UnionProtein, ctx: _Context) -> dict[str, Any]:
    """External residue features: what the trainer's strict loader needs, then the campaign's validity rules."""
    from data_structures import EXTERNAL_FEATURE_NAME_ALIASES

    result: dict[str, Any] = {"status": "ok", "sha256": "", "a13": 0, "lookup": None, "pka": "", "payload": None}
    directory = protein_layout_path(ctx.layout.external_root, protein.stem)
    problem = _dir_problem(directory)
    path = directory / EXTERNAL_FEATURE_FILE
    if problem is None:
        problem = _file_problem(path)
    if problem is not None:
        result["status"] = problem
        return result
    data = path.read_bytes()
    result["sha256"] = sha256_bytes(data)
    try:
        payload = json.loads(data.decode("utf-8"))
    except ValueError:
        result["status"] = "unreadable"
        return result
    if not isinstance(payload, Mapping):
        result["status"] = "unreadable"
        return result
    result["a13"] = count_test_named_strings(payload)
    tooling = payload.get("tooling") if isinstance(payload.get("tooling"), Mapping) else {}
    result["pka"] = str(tooling.get("pka") or "")
    lookup: dict[tuple[str, int, str], dict[str, float]] = {}
    non_finite = False
    try:
        for entry in payload.get("residues", []) or []:
            key = (str(entry["chain_id"]), int(entry["resseq"]), str(entry.get("icode", "")).strip())
            features: dict[str, float] = {}
            for name, value in (entry.get("features") or {}).items():
                if name in LEGACY_EXTERNAL_FEATURES:
                    result["status"] = "legacy_feature"
                    return result
                number = float(value)
                non_finite = non_finite or not math.isfinite(number)
                features[EXTERNAL_FEATURE_NAME_ALIASES.get(name, name)] = number
            lookup[key] = features
    except (KeyError, TypeError, ValueError, AttributeError):
        result["status"] = "unreadable"
        return result
    if not lookup:
        result["status"] = "empty"
        return result
    result["lookup"] = lookup
    result["payload"] = payload
    if payload.get("schema_version") != EXTERNAL_SCHEMA_VERSION:
        result["status"] = "invalid_schema"
    elif payload.get("structure_id") != protein.stem:
        result["status"] = "invalid_structure_id"
    elif payload.get("n_residues") != len(lookup):
        result["status"] = "invalid_n_residues"
    elif non_finite:
        result["status"] = "non_finite"
    return result


def _feature_module_geometry(path: Path):
    """Atom array, residue geometry and metal atoms exactly as ``feature_extraction.core`` derives them."""
    import numpy as np
    from feature_extraction import core
    from feature_extraction.constants import METAL_CHARGE_PROXIES

    atom_array = core.load_structure(str(path))
    protein_atoms = atom_array[core.struc.filter_canonical_amino_acids(atom_array)]
    residues = core._residue_geometry_from_atom_array(protein_atoms)
    metal_mask = np.isin(atom_array.element, np.array(list(METAL_CHARGE_PROXIES)))
    metals = [(str(atom.element), atom.coord.astype(np.float64, copy=False),
               (str(atom.chain_id).strip(), int(atom.res_id))) for atom in atom_array[metal_mask]]
    return atom_array, metal_mask, residues, metals


def charge_proxy_contributions(path: Optional[Path] = None, *, pocket_keys: Sequence[tuple[str, int, str]],
                               selected_sites: Iterable[tuple[str, int]],
                               geometry: Optional[tuple] = None) -> dict[tuple[str, int, str], dict]:
    """Per pocket residue: the proxy recomputed with every metal, and the unselected metals' contribution.

    The pair term comes from ``feature_extraction.core._apply_pairwise_interaction_proxies`` on the
    residues within ``RESIDUE_PAIR_CA_CUTOFF`` of a pocket residue (structure order kept, so each
    pocket residue's sum is formed in the module's order) with the metals removed; the metal terms
    then follow the module's loop (``core.py``: every metal of the file in atom order,
    ``q_residue * q_metal / max(d, 2.0)``). Residues the module does not build are left out.
    ``geometry`` is the result of ``_feature_module_geometry`` when the file was already read.
    """
    import numpy as np
    from feature_extraction import core
    from feature_extraction.constants import METAL_CHARGE_PROXIES, RESIDUE_CHARGE_PROXIES
    from data_structures import EXTERNAL_FEATURE_CUSTOM_CHARGE_DISTANCE_PROXY as PROXY

    atom_array, metal_mask, residues, metals = geometry if geometry is not None else _feature_module_geometry(path)
    selected = {(str(chain).strip(), int(resseq)) for chain, resseq in selected_sites}
    by_key = {residue.key: residue for residue in residues}
    pocket = [by_key[key] for key in pocket_keys if key in by_key]
    pocket_set = {residue.key for residue in pocket}
    needed = [residue for residue in residues if residue.key in pocket_set or any(
        float(np.linalg.norm(residue.ca_coord - other.ca_coord)) <= RESIDUE_PAIR_CA_CUTOFF for other in pocket)]
    feature_map = {residue.key: core.default_feature_dict() for residue in needed}
    core._apply_pairwise_interaction_proxies(needed, atom_array[~metal_mask], feature_map)
    results = {}
    for residue in pocket:
        value = feature_map[residue.key][PROXY]
        unselected = 0.0
        charge = RESIDUE_CHARGE_PROXIES.get(residue.resname, 0.0)
        if charge:
            for element, coord, site in metals:
                metal_charge = METAL_CHARGE_PROXIES.get(element, 0.0)
                if not metal_charge:
                    continue
                min_distance = float(np.linalg.norm(residue.heavy_coords - coord, axis=1).min())
                term = (charge * metal_charge) / max(min_distance, 2.0)
                value = value + term
                if site not in selected:
                    unselected = unselected + term
        results[residue.key] = {"resname": residue.resname, "recomputed": float(value),
                                "unselected": float(unselected)}
    return results


def unselected_share(unselected: float, recomputed: float) -> float:
    if unselected == 0.0:
        return 0.0
    if recomputed == 0.0:
        return 1.0
    return min(1.0, abs(unselected) / abs(recomputed))


def _sequence_difference(structure_sequence: str, map_sequence: str) -> str:
    if len(structure_sequence) != len(map_sequence):
        return f"length:{len(structure_sequence)}vs{len(map_sequence)}"
    return ";".join(f"{expected}{position}{observed}"
                    for position, (observed, expected) in enumerate(zip(structure_sequence, map_sequence), start=1)
                    if observed != expected)


def compare_ec_numbers(stem_ec: str, native_ec: Optional[str]) -> str:
    """A4: the structure name's EC numbers against the label table's native ones."""
    if native_ec is None:
        return A4_MISSING
    stem = [entry for entry in normalize_ec_number_list(stem_ec).split(";") if entry]
    native = [entry for entry in normalize_ec_number_list(native_ec).split(";") if entry]
    if stem == native:
        return A4_EQUAL
    if sorted(stem) == sorted(native):
        return A4_ORDER_ONLY
    return A4_DIFFERENT


def trainer_status(load_status: str, label_status: str) -> str:
    """The trainer's per-structure outcome: load problems first, then the EC-class exclusion."""
    if load_status != LOAD_OK:
        return load_status
    return STATUS_AMBIGUOUS if label_status == STATUS_AMBIGUOUS else LOAD_OK


def audit_protein(protein: UnionProtein, ctx: _Context) -> dict[str, Any]:
    """Every per-protein check of the module docstring; returns the protein's row and its detail rows."""
    from data_structures import DEFAULT_POCKET_RADIUS, EXTERNAL_FEATURE_NAMES
    from embed_helpers.esmc import extract_chain_sequences
    from embed_helpers.esmc_chain_means import ChainInput
    from graph.structure_parsing import (
        AmbiguousSelectedSiteError, UnresolvedSelectedSiteError, collect_structure_residues_and_metals,
        extract_selected_site_pockets_from_structure, metal_record_selected_site_id, parse_structure_file,
    )
    from label_schemes import map_site_metal_symbols
    from training.esm_feature_loading import residue_keys_for_structure_chain
    from training.global_esm_features import sha256_text
    from training.site_filter import matched_site_metal_types, pocket_matches_allowed_sites, selected_site_ids_for_structure

    row: dict[str, Any] = {column: "" for column in PROTEIN_COLUMNS}
    row.update({"accession": protein.accession, "chain": protein.chain, "n_training_files": len(protein.folds),
                "structure_sha256": protein.sha256, "structure_size_bytes": protein.size_bytes})
    record: dict[str, Any] = {"row": row, "sites": [], "site_failures": [], "site_distances": [], "metals": [],
                              "residues": [], "a13": {"payload_and_sidecar": 0, "external": 0}, "warnings": 0,
                              "a4": compare_ec_numbers(STRUCTURE_ID_RE.match(protein.stem).group("ec"),
                                                       ctx.native_ec.get(protein.accession)),
                              "missing_feature_values": Counter(), "extraction": None, "propka_flag": False,
                              "sequence": {}}
    site_rows = _selected_site_rows(protein)
    record["sites"] = [{"accession": protein.accession, "chain_resi": item["chain_resi"],
                        "metaltype": item["metaltype"]} for item in site_rows]
    row["n_selected_sites"] = len(site_rows)

    def finish(load_status: str, reason: str = "") -> dict[str, Any]:
        row["load_status"], row["load_reason"] = load_status, reason
        record["trainer_status"] = trainer_status(load_status, protein.label_status)
        return record

    problem = _file_problem(protein.path)
    if problem is not None:
        return finish(LOAD_INPUT_ERROR, f"structure_{problem}")
    data = protein.path.read_bytes()
    if sha256_bytes(data) != protein.sha256 or len(data) != protein.size_bytes:
        return finish(LOAD_INPUT_ERROR, "structure_differs_from_manifest")
    if protein.path.suffix.lower() != Path(protein.structure_name).suffix.lower():
        return finish(LOAD_INPUT_ERROR, "structure_suffix_differs")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pseudo_path = Path(protein.structure_name)  # the name the trainer sees inside a view
        structure, selected, pockets, report, failure = None, None, [], None, None
        try:
            structure = parse_structure_file(str(protein.path), structure_id=protein.stem)
        except Exception:
            failure = "parse_error"
        if failure is None:
            try:
                selected = selected_site_ids_for_structure(pseudo_path, ctx.site_index, key_mode="accession_ec")
            except ValueError:
                failure = "site_key_error"
        if failure is None and selected:
            try:
                pockets, report = extract_selected_site_pockets_from_structure(
                    structure, structure_id=protein.stem, selected_site_ids=selected,
                    pocket_grouping=SELECTED_SITE_POCKET_GROUPING)
            except UnresolvedSelectedSiteError:
                failure = "unresolved_selected_site"
            except AmbiguousSelectedSiteError:
                failure = "ambiguous_selected_site"
            except Exception:
                failure = "extraction_error"

        records_by_site: dict[tuple[str, int], list] = {}
        if structure is not None:
            _residues, metal_records = collect_structure_residues_and_metals(structure)
            for metal in metal_records:
                records_by_site.setdefault(metal_record_selected_site_id(metal), []).append(metal)
        else:
            metal_records = []
        failures, distances = _site_checks(protein, site_rows, records_by_site, ctx.tolerance)
        record["site_failures"], record["site_distances"] = failures, distances
        row["n_sites_verified"] = len(site_rows) - len(failures)
        row["max_site_distance"] = _format_distance(max(distances) if distances else None)

        pocket_residues: dict[tuple[str, int, str], Any] = {}
        for pocket in pockets:
            for residue in pocket.residues:
                pocket_residues.setdefault(residue.residue_id(), residue)
        row["n_pocket_residues"] = len(pocket_residues)
        if report is not None:
            compact = {key: int(report[key]) for key in COMPACT_REPORT_KEYS}
            compact["selected_sites_without_pocket"] = len(report["selected_sites_without_pocket"])
            row.update(compact)
            row["pocket_count"] = int(report["pocket_count"])
            if report.get("pocket_grouping") == SELECTED_SITE_POCKET_GROUPING:
                # Kept in the extraction totals only (as in the trainer's summary), not as protein columns.
                cluster_count_key, bare_sites_key = STRUCTURE_GROUPING_REPORT_KEYS
                compact[cluster_count_key] = int(report[cluster_count_key])
                compact[bare_sites_key] = len(report[bare_sites_key])
            record["extraction"] = compact

        chain_input = None
        if structure is not None:
            try:
                sequence = extract_chain_sequences(structure).get(protein.chain)
            except ValueError:
                sequence = None
            if sequence is not None:
                residue_ids, _with_ca = residue_keys_for_structure_chain(structure, protein.chain)
                if len(residue_ids) == len(sequence):
                    chain_input = ChainInput(structure_id=protein.stem, chain_id=protein.chain,
                                             structure_file=protein.structure_name, structure_sha256=protein.sha256,
                                             sequence=sequence, sequence_sha256=sha256_text(sequence),
                                             residue_ids=tuple(residue_ids))
                sequence_sha = sha256_text(sequence)
                map_sequence = ctx.map_sequences.get(protein.accession)
                row.update({"sequence_length": len(sequence), "sequence_sha256": sequence_sha,
                            "n_x_residues": sequence.count("X")})
                record["sequence"] = {"sha256": sequence_sha, "in_map": map_sequence is not None}
                if map_sequence is not None:
                    row["sequence_map_match"] = _flag(sequence == map_sequence)
                    if sequence != map_sequence:
                        row["sequence_map_difference"] = _sequence_difference(sequence, map_sequence)

        esm = _esm_checks(protein, ctx, structure=structure, chain_input=chain_input) if structure is not None else {
            "status": "not_checked", "sha256": "", "sidecar_sha256": "", "a13": 0, "sidecar_sequence_sha256": None,
            "keys": None, "width": None}
        external = _external_checks(protein, ctx)
        row.update({"esm_status": esm["status"], "esm_sha256": esm["sha256"], "esm_sidecar_sha256": esm["sidecar_sha256"],
                    "external_status": external["status"], "external_sha256": external["sha256"],
                    "pka_tooling": external["pka"]})
        record["a13"] = {"payload_and_sidecar": esm["a13"], "external": external["a13"]}
        if record["sequence"]:
            row["sidecar_sequence_match"] = _flag(esm["sidecar_sequence_sha256"] == record["sequence"]["sha256"])

        # Feature-module checks: residue keys of the external file, and the D2 charge-proxy contributions.
        module_metals: list = []
        proxy: dict = {}
        if external["lookup"] is not None and structure is not None:
            try:
                geometry = _feature_module_geometry(protein.path)
                module_residues, module_metals = geometry[2], geometry[3]
                if external["status"] == "ok" and {residue.key for residue in module_residues} != set(external["lookup"]):
                    external["status"] = "residue_keys_differ"
                    row["external_status"] = external["status"]
                if pocket_residues:
                    proxy = charge_proxy_contributions(pocket_keys=list(pocket_residues),
                                                       selected_sites=selected or (), geometry=geometry)
            except Exception:  # biotite raises several types for files it cannot read
                if external["status"] == "ok":
                    external["status"] = "feature_module_failed"
                    row["external_status"] = external["status"]
        unselected_module_metals = [site for _element, _coord, site in module_metals if site not in (selected or ())]
        record["propka_flag"] = external["pka"] == "propka" and bool(unselected_module_metals)
        if external["lookup"] is not None:
            row["propka_sees_unselected_metals"] = _flag(record["propka_flag"])

        # D2: unselected metals (the trainer's metal records) and pocket residues.
        if report is not None and selected:
            selected_records = [metal for metal in metal_records if metal_record_selected_site_id(metal) in selected]
            pocket_atoms = [atom for residue in pocket_residues.values() for atom in residue.atoms.values()]
            atom_index: Counter = Counter()
            for metal in metal_records:
                site = metal_record_selected_site_id(metal)
                if site in selected:
                    continue
                coord = [float(value) for value in metal.coord.tolist()]
                nearest_selected = min((math.dist(coord, [float(v) for v in other.coord.tolist()])
                                        for other in selected_records), default=None)
                nearest_pocket = min((math.dist(coord, [float(v) for v in atom.tolist()]) for atom in pocket_atoms),
                                     default=None)
                chain_resi = f"{site[0]}_{site[1]}"
                record["metals"].append({
                    "accession": protein.accession, "chain_resi": chain_resi, "icode": metal.site_id[2],
                    "atom_index": atom_index[metal.site_id], "element": metal.symbol,
                    "nearest_selected_distance": nearest_selected,
                    "same_site_copy": nearest_selected is not None and nearest_selected <= SAME_SITE_COPY_DISTANCE,
                    "nearest_pocket_residue_distance": nearest_pocket,
                    "within_pocket_radius": nearest_pocket is not None and nearest_pocket <= DEFAULT_POCKET_RADIUS,
                    "within_8a": nearest_pocket is not None and nearest_pocket <= NEAR_METAL_DISTANCE,
                })
                atom_index[metal.site_id] += 1
            row["n_unselected_metals"] = len(record["metals"])
            row["n_same_site_copies"] = sum(1 for metal in record["metals"] if metal["same_site_copy"])
            for key, residue in sorted(pocket_residues.items()):
                stored = None
                if external["lookup"] is not None and key in external["lookup"]:
                    features = external["lookup"][key]
                    stored = features.get("custom_charge_distance_proxy")
                    for name in EXTERNAL_FEATURE_NAMES:
                        if name not in features or features.get(f"{name}_missing") == 1.0:
                            record["missing_feature_values"][name] += 1
                values = proxy.get(key)
                share = None if values is None else unselected_share(values["unselected"], values["recomputed"])
                parity = None
                if values is not None and stored is not None:
                    parity = math.isclose(values["recomputed"], stored, rel_tol=CHARGE_PROXY_REL_TOLERANCE,
                                          abs_tol=CHARGE_PROXY_ABS_TOLERANCE)
                record["residues"].append({
                    "accession": protein.accession, "chain": key[0], "resseq": key[1], "icode": key[2],
                    "resname": residue.resname, "stored_proxy": stored,
                    "recomputed_proxy": None if values is None else values["recomputed"],
                    "unselected_contribution": None if values is None else values["unselected"],
                    "unselected_share": share, "parity_ok": parity,
                })

        # Load status in the trainer's order (training.structure_loading.load_structure_pockets).
        status, reason = LOAD_OK, ""
        if failure is not None:
            status, reason = LOAD_INVALID, failure
        elif not selected:
            status = LOAD_NO_SELECTED_SITES
        elif not pockets:
            status = LOAD_SITES_WITHOUT_POCKET
        elif esm["status"] in ESM_TRAINER_FAILURES:
            status, reason = LOAD_INVALID, f"esm_{esm['status']}"
        elif external["status"] in EXTERNAL_TRAINER_FAILURES:
            status, reason = LOAD_INVALID, f"external_{external['status']}"
        else:
            for pocket in pockets:
                keys = [residue.residue_id() for residue in pocket.residues]
                if esm["keys"] is None or any(key not in esm["keys"] for key in keys):
                    status, reason = LOAD_INVALID, "esm_pocket_residue_missing"
                elif esm["width"] != ctx.esm_dim:
                    status, reason = LOAD_INVALID, "esm_width"
                elif any(key not in external["lookup"] for key in keys):
                    status, reason = LOAD_INVALID, "external_pocket_residue_missing"
                elif not pocket_matches_allowed_sites(pocket, pseudo_path, ctx.site_labels):
                    status, reason = LOAD_INVALID, "selected_pocket_without_summary_site"
                else:
                    try:
                        map_site_metal_symbols(sorted(matched_site_metal_types(pocket, pseudo_path, ctx.site_labels)),
                                               unsupported_metal_policy="error")
                    except ValueError:
                        status, reason = LOAD_FATAL, "unsupported_metal_label"
                if status != LOAD_OK:
                    break
    record["warnings"] = len(caught)
    return finish(status, reason)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def site_extraction_totals(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Totals in the form of ``training.data.site_extraction_summary`` over trainer-ok structures.

    The trainer sums the per-structure reports of the structures it keeps; the skip counts are
    over the given records. Under the structure grouping the totals add the grouping, the pocket
    count and the structures whose sites the default grouping would split, as the trainer does.
    """
    kept = [item for item in records if item["trainer_status"] == LOAD_OK and item["extraction"] is not None]
    grouped = SELECTED_SITE_POCKET_GROUPING != "cluster"
    totals: dict[str, Any] = {"site_extraction": "selected_sites", "n_structures": len(kept)}
    extra_keys = STRUCTURE_GROUPING_REPORT_KEYS if grouped else ()
    for key in (*COMPACT_REPORT_KEYS, "selected_sites_without_pocket", *extra_keys):
        totals[key] = sum(int(item["extraction"][key]) for item in kept)
    totals["structures_with_unselected_metals"] = sum(1 for item in kept if item["extraction"]["metal_records_dropped"] > 0)
    for reason in (LOAD_NO_SELECTED_SITES, LOAD_SITES_WITHOUT_POCKET):
        totals[f"structures_skipped_{reason}"] = sum(1 for item in records if item["row"]["load_status"] == reason)
    if grouped:
        totals["selected_site_pocket_grouping"] = SELECTED_SITE_POCKET_GROUPING
        totals["n_pockets"] = sum(int(item["row"]["pocket_count"]) for item in kept)
        cluster_count_key = STRUCTURE_GROUPING_REPORT_KEYS[0]
        totals["structures_with_merged_clusters"] = sum(1 for item in kept if item["extraction"][cluster_count_key] > 1)
    return totals


def d2_numbers(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """D2 geometry and the A1 inputs over the given records (callers pass trainer-ok proteins)."""
    residues = [residue for item in records for residue in item["residues"]]
    metals = [metal for item in records for metal in item["metals"]]
    computed = [residue for residue in residues if residue["unselected_contribution"] is not None]
    contributing = [residue for residue in computed if residue["unselected_contribution"] != 0.0]
    shares = [residue["unselected_share"] for residue in contributing]
    compared = [residue for residue in residues if residue["parity_ok"] is not None]
    n_equal = sum(1 for residue in compared if residue["parity_ok"])
    # The recomputed shares describe the stored features only when every pocket residue's stored value matches.
    parity_complete = n_equal == len(compared) == len(residues)
    missing: Counter = Counter()
    for item in records:
        missing.update(item["missing_feature_values"])
    fraction = (len(contributing) / len(computed)) if computed else 0.0
    median = _median(shares)
    option_b = fraction < A1_RESIDUE_FRACTION_LIMIT and (median or 0.0) < A1_MEDIAN_SHARE_LIMIT
    return {
        "n_proteins": len(records),
        "n_proteins_with_unselected_metals": sum(1 for item in records if item["metals"]),
        "n_unselected_metals": len(metals),
        "n_same_site_copies": sum(1 for metal in metals if metal["same_site_copy"]),
        "n_unselected_metals_within_pocket_radius": sum(1 for metal in metals if metal["within_pocket_radius"]),
        "n_unselected_metals_within_8a": sum(1 for metal in metals if metal["within_8a"]),
        "n_pocket_residues": len(residues),
        "n_pocket_residues_recomputed": len(computed),
        "n_pocket_residues_with_unselected_contribution": len(contributing),
        "fraction_with_unselected_contribution": fraction,
        "median_share_among_them": median,
        "charge_proxy_parity": {"n_compared": len(compared), "n_equal": n_equal,
                                "n_not_compared": len(residues) - len(compared), "complete": parity_complete},
        "n_proteins_propka_with_unselected_metals": sum(1 for item in records if item["propka_flag"]),
        "missing_feature_values_over_pocket_residues": dict(sorted(missing.items())),
        "a1_option_b_criteria_met": option_b if parity_complete else None,
    }


def _distance_summary(distances: Sequence[float]) -> dict[str, Any]:
    return {"n": len(distances), "max": max(distances) if distances else None,
            "p50": _quantile(distances, 0.5), "p90": _quantile(distances, 0.9), "p99": _quantile(distances, 0.99)}


def recommended_a1_option(flags: Iterable[Optional[bool]]) -> Optional[str]:
    """``b`` when option (b) holds in every fold, ``d`` when it fails in one, None when a fold is undetermined."""
    flags = list(flags)
    if any(flag is None for flag in flags):
        return None
    return "b" if all(flags) else "d"


def _fold_counts(fold: int, records: Sequence[Mapping[str, Any]], *, expected_eligible: Optional[int]) -> dict:
    statuses = Counter(item["trainer_status"] for item in records)
    eligible = [item for item in records if item["label_status"] == STATUS_OK]
    exclusions = [item for item in eligible if item["row"]["load_status"] in EXCLUDING_LOAD_STATUSES]
    ok = [item for item in records if item["trainer_status"] == LOAD_OK]
    n_failed_sites = sum(len(item["site_failures"]) for item in records)
    d2 = d2_numbers(ok)
    return {
        "format": COHORT_AUDIT_FORMAT, "schema_version": COHORT_SCHEMA_VERSION, "campaign_id": CAMPAIGN_ID,
        "kind": "cohort_counts_fold", "fold": int(fold),
        "scope": "proteins of this fold's training files only; no EC-class breakdown",
        "n_proteins": len(records), "n_label_eligible": len(eligible),
        "label_eligible_matches_expected": None if expected_eligible is None else len(eligible) == expected_eligible,
        "trainer_status_counts": {status: statuses.get(status, 0) for status in TRAINER_STATUSES},
        "n_load_time_exclusions": len(exclusions),
        "n_fatal_load_outcomes": statuses.get(LOAD_FATAL, 0),
        "site_extraction": site_extraction_totals(records),
        "selected_sites": {"n_sites": sum(int(item["row"]["n_selected_sites"]) for item in records),
                           "n_failed": n_failed_sites},
        "d2": d2,
        "a1": {"residue_fraction_limit": A1_RESIDUE_FRACTION_LIMIT, "median_share_limit": A1_MEDIAN_SHARE_LIMIT,
               "charge_proxy_parity_complete": d2["charge_proxy_parity"]["complete"],
               "option_b_criteria_met": d2["a1_option_b_criteria_met"]},
        "inputs": {"n_pka_not_propka": sum(1 for item in records if item["row"]["pka_tooling"] != "propka")},
    }, exclusions


def _parse_expected_differences(values: Iterable[str]) -> set[tuple[str, str]]:
    expected = set()
    for value in values:
        accession, separator, difference = str(value).partition(":")
        if not separator or not accession.strip() or not difference.strip():
            raise CohortAuditError("An expected sequence difference has the form ACCESSION:DIFFERENCE "
                                   "(for example P12345:V113I).")
        expected.add((accession.strip(), difference.strip()))
    return expected


# ---------------------------------------------------------------------------
# The audit
# ---------------------------------------------------------------------------

def _inside_repository_tree(path: Path) -> bool:
    real = _real(path)
    return _under(real, [_real(REPO_ROOT)]) and not _under(real, [_real(REPO_ROOT / "DeepMzyme_Data")])


def _refuse_non_empty(out_dir: Path) -> None:
    if out_dir.is_symlink() or (out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir()))):
        raise CohortAuditError(f"Cohort folder {out_dir} already exists and is not empty (a leftover "
                               f"{STAGING_PREFIX}<pid> folder marks an interrupted run); remove it first.")


def _check_output_location(out_dir: Path, layout: SourceLayout) -> None:
    if is_test_named_path(out_dir):
        raise HeldOutInputError(f"Cohort folder {out_dir} is test-named; refused.")
    real = _real(out_dir)
    shared = [layout.data_root / name for name in FORBIDDEN_DATA_ROOT_NAMES] + [layout.esm_dir, layout.external_root]
    if _under(real, [_real(root) for root in shared] + [os.path.abspath(root) for root in shared]):
        raise CohortAuditError(f"Cohort folder {out_dir} lies inside a shared or test-bearing data root; the audit "
                               "writes into a campaign folder.")
    if _inside_repository_tree(out_dir):
        raise CohortAuditError(f"Cohort folder {out_dir} lies inside the repository's tracked tree; it holds every "
                               "cohort protein and stays with the private campaign data.")
    # Only <campaign root>/cohort is under every fit's read guard; a nested copy (inside a view, a runs folder or
    # another campaign folder) would be readable by that folder's fits.
    if out_dir.name != COHORT_FOLDER_NAME or out_dir.parent.name != CAMPAIGN_ID:
        raise CohortAuditError(f"Cohort folder {out_dir} must be <campaign root>/{COHORT_FOLDER_NAME}, where the "
                               f"campaign root's folder name is {CAMPAIGN_ID}; the fits' read guard covers only that "
                               "folder.")
    _refuse_non_empty(out_dir)


def _read_checked_input(path: Path, expected_sha256: str, what: str) -> bytes:
    if is_test_named_path(path):
        raise HeldOutInputError(f"{what} {path} is test-named; refused.")
    problem = _file_problem(path)
    if problem is not None:
        raise CohortAuditError(f"{what} {path} cannot be read ({problem}).")
    data = path.read_bytes()
    expected = str(expected_sha256 or "").strip().lower()
    if sha256_bytes(data) != expected:
        raise CohortAuditError(f"{what} {path.name} does not hash to the given SHA-256.")
    return data


def read_expected_label_eligible(path: str | os.PathLike, expected_sha256: str) -> list[int]:
    """The per-fold label-only eligible counts from a private JSON file (a list of five integers).

    The counts are per-fold class numbers: across the five folds they give test-side counts by
    subtraction, so they are read from a private file, never taken from the command line, and no
    message shows them. A file inside the repository's tracked tree is refused.
    """
    path = Path(os.path.abspath(os.fspath(path)))
    if _inside_repository_tree(path):
        raise CohortAuditError(f"Expected-counts file {path} lies inside the repository's tracked tree; per-fold "
                               "counts stay with the private campaign records.")
    data = _read_checked_input(path, expected_sha256, "Expected-counts file")
    try:
        values = json.loads(data.decode("utf-8"))
    except ValueError:
        values = None
    if not (isinstance(values, list) and len(values) == N_OUTER_FOLDS
            and all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in values)):
        raise CohortAuditError(f"Expected-counts file {path.name} must hold a JSON list of {N_OUTER_FOLDS} "
                               "non-negative integers, one per fold (contents not shown).")
    return [int(value) for value in values]


def _move_new(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        raise CohortAuditError(f"{target.name} appeared in the cohort folder while the audit ran; refused.")
    os.rename(source, target)


def run_cohort_audit(
    *,
    layout: SourceLayout,
    out_dir: str | os.PathLike,
    label_table: str | os.PathLike,
    label_table_sha256: str,
    sequence_map: str | os.PathLike,
    sequence_map_sha256: str,
    coordinate_tolerance: float = A3_COORDINATE_TOLERANCE,
    expected_label_eligible_file: Optional[str | os.PathLike] = None,
    expected_label_eligible_sha256: Optional[str] = None,
    expected_sequence_differences: Iterable[str] = (),
    smoke: bool = False,
    provenance: Optional[Mapping[str, Any]] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    esm_model_name: str = ESMC_MODEL_NAME,
    esm_dim: int = ESMC_EMBEDDING_DIM,
) -> dict[str, Any]:
    """Run the audit and write ``<campaign root>/cohort``; returns the printable summary (union totals and flags).

    ``status`` is ``passed`` when every check holds and ``stopped`` otherwise (the outputs and the
    receipt are written either way). Refused inputs raise ``CohortAuditError`` or
    ``HeldOutInputError`` before anything is written. ``expected_label_eligible_file`` and its
    SHA-256 (both or neither) name the private expected-counts file (``read_expected_label_eligible``).
    """
    from embed_helpers.esmc_chain_means import read_sequence_map
    from ec_clean30.scoring import parse_label_table
    from training.site_filter import SiteKeyIndex

    tolerance = float(coordinate_tolerance)
    if not (0.0 < tolerance <= A3_COORDINATE_TOLERANCE):
        raise CohortAuditError(f"The coordinate tolerance must lie in (0, {A3_COORDINATE_TOLERANCE}] Å (decision "
                               "A3); a looser value needs a recorded decision.")
    if (expected_label_eligible_file is None) != (expected_label_eligible_sha256 is None):
        raise CohortAuditError("The expected-counts file and its SHA-256 go together; give both or neither.")
    expected_label_eligible: Optional[list[int]] = None
    expected_source: Optional[dict[str, str]] = None
    if expected_label_eligible_file is not None:
        expected_label_eligible = read_expected_label_eligible(expected_label_eligible_file,
                                                               str(expected_label_eligible_sha256))
        expected_source = {"path": os.path.abspath(os.fspath(expected_label_eligible_file)),
                           "sha256": str(expected_label_eligible_sha256).strip().lower()}
    expected_differences = _parse_expected_differences(expected_sequence_differences)
    out_dir = Path(os.path.abspath(os.fspath(out_dir)))
    _check_output_location(out_dir, layout)

    label_path, map_path = Path(os.fspath(label_table)), Path(os.fspath(sequence_map))
    label_bytes = _read_checked_input(label_path, label_table_sha256, "Label table")
    map_bytes = _read_checked_input(map_path, sequence_map_sha256, "Sequence map")
    try:
        native_ec = parse_label_table(label_bytes, "the label table")
    except (ValueError, RuntimeError) as exc:  # scoring.ScoringError is a RuntimeError
        raise _withheld(exc, "The label table") from None
    try:
        sequence_map_record = read_sequence_map(map_path, expected_sha256=sequence_map_sha256)
    except ValueError as exc:
        raise _withheld(exc, "The sequence map") from None

    sources = read_union_sources(layout)
    ctx = _Context(layout=layout, site_labels=sources.site_labels, site_index=SiteKeyIndex(sources.site_labels),
                   tolerance=tolerance, esm_model_name=esm_model_name, esm_dim=int(esm_dim),
                   map_sequences=sequence_map_record.sequence_by_accession, native_ec=native_ec)
    records: dict[str, dict[str, Any]] = {}
    proteins = list(sources.proteins.values())
    for done, protein in enumerate(proteins, start=1):
        try:
            record = audit_protein(protein, ctx)
        except HeldOutInputError:
            raise
        except Exception as exc:  # an unforeseen failure: its message can hold a path that names the protein
            raise _withheld(exc, f"The audit of protein {done} of {len(proteins)} (in accession order)") from None
        record["label_status"] = protein.label_status
        records[protein.accession] = record
        if progress is not None:
            progress(done, len(proteins))

    union = [records[accession] for accession in sorted(records)]
    rows = [item["row"] for item in union]
    observed_differences = {(row["accession"], row["sequence_map_difference"]) for row in rows
                            if row["sequence_map_difference"]}
    missing_from_map = sum(1 for item in union if item["sequence"] and not item["sequence"]["in_map"])
    a4 = Counter(item["a4"] for item in union)
    load_counts = Counter(row["load_status"] for row in rows)
    site_failures = [failure for item in union for failure in item["site_failures"]]
    site_distances = [distance for item in union for distance in item["site_distances"]]
    failed_distances = [float(failure["distance"]) for failure in site_failures if failure["distance"]]
    esm_counts = Counter(row["esm_status"] for row in rows)
    external_counts = Counter(row["external_status"] for row in rows)
    loaded = [item for item in union if item["row"]["load_status"] == LOAD_OK]
    d2_loaded = d2_numbers(loaded)

    # Staging sits inside the cohort folder, so a leftover after a hard kill stays under the read guard.
    _refuse_non_empty(out_dir)
    created = not out_dir.exists()
    out_dir.mkdir(parents=True, exist_ok=True)
    staging = out_dir / f"{STAGING_PREFIX}{os.getpid()}"
    staging.mkdir()
    moved: list[Path] = []
    try:
        outputs: dict[str, dict[str, Any]] = {}

        def table(name: str, columns: Sequence[str], table_rows: Sequence[Mapping[str, Any]], *,
                  delimiter: str = "\t", optional: bool = False) -> Optional[str]:
            if optional and not table_rows:
                return None
            digest = _write_new(staging / name, _tsv_bytes(columns, table_rows, delimiter=delimiter))
            outputs[name] = {"sha256": digest, "n_rows": len(table_rows)}
            return digest

        table(PROTEINS_NAME, PROTEIN_COLUMNS, rows)
        sites = sorted((site for item in union for site in item["sites"]),
                       key=lambda site: (site["accession"], site["chain_resi"]))
        table(SITES_NAME, SITE_COLUMNS, sites, delimiter=",")
        metal_rows = [{**metal, "nearest_selected_distance": _format_distance(metal["nearest_selected_distance"]),
                       "nearest_pocket_residue_distance": _format_distance(metal["nearest_pocket_residue_distance"]),
                       "same_site_copy": _flag(metal["same_site_copy"]),
                       "within_pocket_radius": _flag(metal["within_pocket_radius"]),
                       "within_8a": _flag(metal["within_8a"])}
                      for item in union for metal in item["metals"]]
        table(D2_METALS_NAME, D2_METAL_COLUMNS, metal_rows)
        residue_rows = [{**residue, "stored_proxy": _format_float(residue["stored_proxy"]),
                         "recomputed_proxy": _format_float(residue["recomputed_proxy"]),
                         "unselected_contribution": _format_float(residue["unselected_contribution"]),
                         "unselected_share": _format_float(residue["unselected_share"]),
                         "parity_ok": _flag(residue["parity_ok"])}
                        for item in union for residue in item["residues"]]
        table(D2_RESIDUES_NAME, D2_RESIDUE_COLUMNS, residue_rows)
        quarantine = [{"accession": item["row"]["accession"], "a4_outcome": item["a4"]} for item in union
                      if item["a4"] in (A4_DIFFERENT, A4_MISSING)]
        table(QUARANTINE_NAME, QUARANTINE_COLUMNS, quarantine, optional=True)
        table(SITE_FAILURES_NAME, SITE_FAILURE_COLUMNS, site_failures, optional=True)

        per_fold: dict[str, dict[str, Any]] = {}
        for fold in range(N_OUTER_FOLDS):
            fold_records = [records[accession] for accession in sorted(sources.folds[fold]["accessions"])]
            counts, exclusions = _fold_counts(fold, fold_records, expected_eligible=(
                None if expected_label_eligible is None else expected_label_eligible[fold]))
            exclusion_name = FOLD_EXCLUSIONS_NAME.format(fold=fold)
            exclusion_digest = table(exclusion_name, EXCLUSION_COLUMNS,
                                     [{column: item["row"][column] for column in EXCLUSION_COLUMNS}
                                      for item in exclusions], optional=True)
            counts["exclusions_file"] = exclusion_name if exclusion_digest else None
            counts["exclusions_sha256"] = exclusion_digest
            counts_name = FOLD_COUNTS_NAME.format(fold=fold)
            digest = write_new_json(staging / counts_name, counts)
            outputs[counts_name] = {"sha256": digest}
            per_fold[str(fold)] = {
                "file": counts_name, "sha256": digest,
                "a1_option_b_criteria_met": counts["a1"]["option_b_criteria_met"],
                "charge_proxy_parity_complete": counts["a1"]["charge_proxy_parity_complete"],
                "label_eligible_matches_expected": counts["label_eligible_matches_expected"],
                "no_load_time_exclusions": counts["n_load_time_exclusions"] == 0,
            }

        checks = {
            "structures_match_manifest": load_counts.get(LOAD_INPUT_ERROR, 0) == 0,
            "all_structures_load": load_counts.get(LOAD_OK, 0) == len(rows),
            "no_fatal_load_outcomes": load_counts.get(LOAD_FATAL, 0) == 0,
            "selected_sites_verified": not site_failures,
            "esmc_inputs_valid": esm_counts.get("ok", 0) == len(rows),
            "external_inputs_valid": external_counts.get("ok", 0) == len(rows),
            "structure_sequence_equals_sidecar": all(row["sidecar_sequence_match"] == "1" for row in rows),
            "sequence_map_differences_as_expected": missing_from_map == 0 and observed_differences == expected_differences,
            "no_ec_content_differences": a4.get(A4_DIFFERENT, 0) == 0 and a4.get(A4_MISSING, 0) == 0,
            "charge_proxy_parity_complete": d2_loaded["charge_proxy_parity"]["complete"],
            "label_eligible_as_expected": (None if expected_label_eligible is None else
                                           all(item["label_eligible_matches_expected"] for item in per_fold.values())),
        }
        failed = sorted(name for name, value in checks.items() if value is False)
        not_run = sorted(name for name, value in checks.items() if value is None)
        a1_flags = {fold: item["a1_option_b_criteria_met"] for fold, item in per_fold.items()}
        totals = {
            "n_proteins": len(rows),
            "n_selected_sites": len(sites),
            "load_status_counts": {status: load_counts.get(status, 0) for status in LOAD_STATUSES},
            "load_reasons": dict(sorted(Counter(row["load_reason"] for row in rows if row["load_reason"]).items())),
            "selected_site_check": {"tolerance": tolerance, "n_sites": len(sites), "n_failed": len(site_failures),
                                    "failures_by_reason": dict(sorted(Counter(f["reason"] for f in site_failures).items())),
                                    "distances": _distance_summary(site_distances),
                                    "failed_distances": _distance_summary(failed_distances)},
            "site_extraction": site_extraction_totals([{**item, "trainer_status": item["row"]["load_status"]}
                                                       for item in union]),
            "d2_over_loaded_proteins": d2_loaded,
            "esm_status_counts": dict(sorted(esm_counts.items())),
            "external_status_counts": dict(sorted(external_counts.items())),
            "n_pka_not_propka": sum(1 for row in rows if row["pka_tooling"] != "propka"),
            "sequences": {"n_sidecar_match": sum(1 for row in rows if row["sidecar_sequence_match"] == "1"),
                          "n_map_match": sum(1 for row in rows if row["sequence_map_match"] == "1"),
                          "n_map_differences": len(observed_differences),
                          "n_map_substitution_differences": sum(1 for _accession, diff in observed_differences
                                                                if not diff.startswith("length:")),
                          "n_map_length_differences": sum(1 for _accession, diff in observed_differences
                                                          if diff.startswith("length:")),
                          "n_missing_from_map": missing_from_map,
                          "n_expected_differences": len(expected_differences),
                          "n_with_x_residues": sum(1 for row in rows if row["n_x_residues"] not in ("", 0))},
            "a4": {outcome: a4.get(outcome, 0) for outcome in (A4_EQUAL, A4_ORDER_ONLY, A4_DIFFERENT, A4_MISSING)},
            "a13_test_named_string_count": {
                "payload_and_sidecar": sum(item["a13"]["payload_and_sidecar"] for item in union),
                "external": sum(item["a13"]["external"] for item in union)},
            "n_library_warnings": sum(item["warnings"] for item in union),
        }
        totals["a13_test_named_string_count"]["total"] = sum(totals["a13_test_named_string_count"].values())
        from data_structures import DEFAULT_MULTINUCLEAR_MERGE_DISTANCE, DEFAULT_POCKET_RADIUS

        receipt = {
            "format": COHORT_AUDIT_FORMAT, "schema_version": COHORT_SCHEMA_VERSION, "campaign_id": CAMPAIGN_ID,
            "kind": "cohort_audit", "smoke": bool(smoke), "created_utc": utc_now(),
            "status": "passed" if not failed else "stopped", "failed_checks": failed, "checks_not_run": not_run,
            "checks": checks,
            "inputs": {
                "layout": layout.to_dict(),
                "folds": {str(fold): {key: value for key, value in info.items() if key != "accessions"}
                          for fold, info in sources.folds.items()},
                "label_table": {"path": str(label_path), "sha256": sha256_bytes(label_bytes)},
                "sequence_map": {"path": str(map_path), "sha256": sha256_bytes(map_bytes),
                                 "ids_sha256": sequence_map_record.ids_sha256},
                # The expected counts themselves stay in their private file.
                "expected_label_eligible_file": expected_source,
                "expected_sequence_differences": sorted(f"{a}:{d}" for a, d in expected_differences),
            },
            "settings": {"coordinate_tolerance": tolerance, "a3_tolerance": A3_COORDINATE_TOLERANCE,
                         "site_key_mode": "accession_ec", "pocket_radius": float(DEFAULT_POCKET_RADIUS),
                         "multinuclear_merge_distance": float(DEFAULT_MULTINUCLEAR_MERGE_DISTANCE),
                         "selected_site_pocket_grouping": SELECTED_SITE_POCKET_GROUPING,
                         "same_site_copy_distance": SAME_SITE_COPY_DISTANCE, "near_distance": NEAR_METAL_DISTANCE,
                         "residue_pair_ca_cutoff": RESIDUE_PAIR_CA_CUTOFF, "esm_model_name": esm_model_name,
                         "esm_dim": int(esm_dim), "a1_residue_fraction_limit": A1_RESIDUE_FRACTION_LIMIT,
                         "a1_median_share_limit": A1_MEDIAN_SHARE_LIMIT},
            "rules": dict(RULES),
            "totals": totals,
            "per_fold": per_fold,
            "a1_recommended_option": recommended_a1_option(a1_flags.values()),
            "outputs": dict(sorted(outputs.items())),
            "provenance": dict(provenance) if provenance is not None else provenance_record(smoke=smoke),
        }
        receipt_digest = write_new_json(staging / RECEIPT_NAME, receipt)
        # Move every file up, the receipt last: a cohort folder without its receipt is incomplete.
        for name in [*sorted(path.name for path in staging.iterdir() if path.name != RECEIPT_NAME), RECEIPT_NAME]:
            _move_new(staging / name, out_dir / name)
            moved.append(out_dir / name)
        staging.rmdir()
    except BaseException:
        _remove_tree(staging)
        for path in moved:
            try:
                path.unlink()
            except OSError:
                pass
        if created:
            try:
                out_dir.rmdir()
            except OSError:
                pass
        raise
    return {
        "status": receipt["status"], "failed_checks": failed, "checks_not_run": not_run, "checks": checks,
        "out": str(out_dir),
        "receipt": str(out_dir / RECEIPT_NAME), "receipt_sha256": receipt_digest,
        "n_proteins": totals["n_proteins"], "n_selected_sites": totals["n_selected_sites"],
        "load_status_counts": totals["load_status_counts"],
        "selected_site_failures": totals["selected_site_check"]["n_failed"],
        "a4": totals["a4"], "a13_test_named_string_count": totals["a13_test_named_string_count"]["total"],
        "n_sequence_map_differences": totals["sequences"]["n_map_differences"],
        "d2_over_loaded_proteins": {key: totals["d2_over_loaded_proteins"][key] for key in (
            "n_unselected_metals", "n_same_site_copies", "n_pocket_residues",
            "fraction_with_unselected_contribution", "median_share_among_them", "charge_proxy_parity")},
        "a1_option_b_criteria_met_per_fold": a1_flags, "a1_recommended_option": receipt["a1_recommended_option"],
        "label_eligible_as_expected": checks["label_eligible_as_expected"],
    }


# ---------------------------------------------------------------------------
# Runner action
# ---------------------------------------------------------------------------

def add_cohort_audit_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Arguments of the runner's ``cohort-audit`` action (see the module docstring)."""
    parser.add_argument("--out", required=True,
                        help=f"<campaign root>/{COHORT_FOLDER_NAME}, absent or empty (the campaign root's folder name "
                             f"is {CAMPAIGN_ID}).")
    parser.add_argument("--label-table", required=True, help="native_label_table.tsv (comparator label-table).")
    parser.add_argument("--label-table-sha256", required=True)
    parser.add_argument("--sequence-map", required=True, help="sequence_map.tsv (comparator sequence-map).")
    parser.add_argument("--sequence-map-sha256", required=True)
    parser.add_argument("--coordinate-tolerance", type=float, default=A3_COORDINATE_TOLERANCE,
                        help=f"Selected-site tolerance in Å (A3; default and maximum {A3_COORDINATE_TOLERANCE}).")
    parser.add_argument("--data-root", default=None, help="Data root (default: the project's DeepMzyme_Data).")
    parser.add_argument("--esm-dir", default=None,
                        help="Per-residue ESMC-300m caches (default <data-root>/esm_embeddings; exact names only).")
    parser.add_argument("--external-root", default=None,
                        help="External residue features (default <data-root>/updated_feature_extraction).")
    parser.add_argument("--expected-label-eligible-file", default=None,
                        help="Private JSON file: a list of the five per-fold label-only eligible counts (kept off "
                             "the command line; outside the repository).")
    parser.add_argument("--expected-label-eligible-sha256", default=None,
                        help="SHA-256 of --expected-label-eligible-file (required with it).")
    parser.add_argument("--expected-sequence-difference", action="append", default=[],
                        help="An expected structure-versus-map difference, ACCESSION:DIFFERENCE (repeatable).")
    parser.add_argument("--smoke", action="store_true", help="Engineering use: allow a dirty tree.")
    return parser


def action_cohort_audit(args: argparse.Namespace) -> int:
    """The ``cohort-audit`` action: refusals, clean-tree check, the audit and a summary without accessions."""
    from ec_clean30.provenance import refuse_test_paths
    from project_paths import DATA_DIR

    for name in ("out", "label_table", "sequence_map", "data_root", "esm_dir", "external_root",
                 "expected_label_eligible_file"):
        refuse_test_paths([getattr(args, name, None)], what=f"--{name.replace('_', '-')}")
    git = real_action_git_state(smoke=args.smoke)
    layout = SourceLayout.from_data_root(args.data_root or DATA_DIR, esm_dir=args.esm_dir,
                                         external_root=args.external_root)

    def progress(done: int, total: int) -> None:
        if done == total or done % 50 == 0:
            print(f"[cohort-audit] {done}/{total} proteins", file=sys.stderr, flush=True)

    summary = run_cohort_audit(
        layout=layout, out_dir=args.out, label_table=args.label_table, label_table_sha256=args.label_table_sha256,
        sequence_map=args.sequence_map, sequence_map_sha256=args.sequence_map_sha256,
        coordinate_tolerance=args.coordinate_tolerance,
        expected_label_eligible_file=args.expected_label_eligible_file,
        expected_label_eligible_sha256=args.expected_label_eligible_sha256,
        expected_sequence_differences=args.expected_sequence_difference, smoke=args.smoke,
        provenance=provenance_record(smoke=args.smoke, git=git), progress=progress)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0 if summary["status"] == "passed" else 1
