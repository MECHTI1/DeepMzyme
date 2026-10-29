"""Training-only per-fold views of the EC class versus CLEAN30 campaign, and the dry-run probe.

A view is a self-contained folder that holds byte copies (never symlinks or hardlinks)
of exactly one outer fold's training inputs, so a fit can be pointed at it and at
nothing else. Training structures of fold k come only from the per-fold training
manifest ``<data>/CLEAN_30_train_test_split_<k>/train/structure_manifest.csv``
(``relative_path`` is relative to that ``train/`` folder), whose accessions must equal
the rows of the conservative training site table
``<data>/CLEAN_30_main/folds/CLEAN_30_train_test_split_<k>_train.csv``. No other file
under ``CLEAN_30_main`` is opened and no mixed folder is listed; per-structure inputs
are opened by exact constructed names:

- per-residue ESMC files ``<esm>/<stem>_chain_<chain>_esmc.pt`` and their sidecars
  ``<stem>_chain_<chain>_esmc.pt.json``;
- external residue features ``<external>/<stem>/residue_features.json`` (the layout the
  trainer resolves for manifest-backed structures, ``training.runtime_preparation``).

Source roots (docs/DATASETS.md, sections "Main Colab bundle v10/v11" and the data
layout in docs/README.md; defaults of ``src/project_paths.py``): per-residue ESMC-300m
caches in ``<data>/esm_embeddings`` and external features in
``<data>/updated_feature_extraction``. Both are shared mixed roots, so they are only
ever opened by exact name, and both can be overridden on the command line.

Layout of a view (data folders and files read-only after the build)::

    view_manifest.json        per-file SHA-256, source manifest and table SHA-256s
    allowlist.txt             the view's structure IDs, one per line
    structures/               <stem>.pdb copies + structure_manifest.csv (relative paths)
    site_summary_train.csv    the training site rows of the view's proteins
    esm/                      per-residue ESMC payloads and sidecars
    external/<stem>/residue_features.json
    ring/                     empty (no RING edges in this campaign)
    probe_structure_ids.txt   training structures whose refit predictions are replayed
    smoke_membership.csv      smoke views: 2-fold EC membership over the subset
    inner_membership.csv      fold views: the certified inner membership, when given
    inner_certification_receipt.json   fold views: its certificate (scripts/run_ec_clean30_identity.py)
    chain_means_final/        writable until built (src/embed_helpers/esmc_chain_means.py)
    chain_means_block26/      writable until built (L8 layer)
    caches/{parse,graph,tmp,dry_run}/   writable per-view caches
    receipts/                 writable verification and dry-run receipts

The view copies external features unchanged. In the shared roots they were computed on full
structures, including metals outside the selected sites (``EXTERNAL_FEATURES_POLICY``); a view
built from a data overlay states instead the overlay's decision A1 (D2) option and policy, read
from the overlay manifest, and refuses an option it does not know (``OVERLAY_D2_OPTIONS``) or a
policy text other than this code's text for that option (``OVERLAY_D2_POLICY``, which the
overlay builder writes).

Location: a fold view lives at ``<campaign root>/views/fold<k>`` and a smoke view at
``<campaign root>/smoke_views/<name>``, where the campaign root's name is the campaign ID;
build, loading, verification and commands refuse any other place. Their runs live (real
path) in ``<campaign root>/runs/fold<k>`` and ``<campaign root>/smoke_runs/<name>`` or in a
folder inside them (``runs_dir_problem``).

Read guard (manifest schema 2): the manifest records rules, not paths (``read_guard``):
data-root children (``FORBIDDEN_DATA_ROOT_NAMES`` and the source layout's ESMC and
external folders relative to the data root), campaign children
(``CAMPAIGN_FORBIDDEN_CHILDREN``, the other folds' views and runs, and the smoke or real
counterpart folders), for a smoke view every other entry of ``smoke_views/`` and
``smoke_runs/`` (``campaign_sibling_dirs``; names listed when the rules are expanded),
repository paths and the PinMyMetal root. ``expand_read_guard`` expands them on the
machine that runs: below the view's recorded source data root, every ``DeepMzyme_Data``
folder that contains the view and the project data root; below the campaign root two
levels above the view (given and real path); below the running checkout. An expanded
root that contains the view is refused, never dropped.

File modes: data files 0444, data folders 0555, writable folders 0755. A filesystem that
ignores or refuses chmod (Data1 under ntfs-3g) does not stop a build; the outcome goes to
``receipts/file_modes.json``. ``verify_view`` tests chmod in ``receipts/`` each time and
checks writable bits only where modes are enforced (content hashes always);
``apply_view_modes`` sets the modes again after a driver change (chmod only, no byte or
manifest change; ``receipts/file_modes_applied.json``).

Overlay binding: when ``<data root>/overlay_manifest.json`` exists, the build checks that
it lists every input the view copies with the same SHA-256 and records
``overlay: {id, manifest_sha256, d2_option, d2_policy}``.

Dry-run probe: ``python -m ec_clean30.views probe-child --request R`` runs the trainer's
preparation path (``training.run.prepare_run``: input checks, loading, split, graphs,
model and optimizer; no training step) under a recorder (``sys.addaudithook``) of every
file open, listing, write and subprocess, installed before the held-out guard of
``DEEPMZYME_FORBIDDEN_READ_ROOTS`` so a denied read is recorded too, with the three
feature generators replaced by sentinels. The parent classifies every recorded path: data
reads only inside the view, its caches and the probe folder; code reads only of Python
files under ``src/`` and of the Python installation (never the repository root or a
folder that contains it); system folders; listings additionally of the exact entries of
the probe's ``sys.path`` (the import system lists them); subprocesses only of
``ALLOWED_SUBPROCESS_PROGRAMS``.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from functools import cached_property
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import stat
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Optional, Sequence
import uuid

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ec_clean30.campaign import CAMPAIGN_ID, N_OUTER_FOLDS, canonical_sha256
from ec_clean30.provenance import (
    REPO_ROOT, SRC_ROOT, HeldOutInputError, is_test_named_path, names_test_split, provenance_record, sha256_bytes,
    sha256_file, utc_now,
)
from ec_clean30.selection import N_INNER_FOLDS
from training.ec_classes import (
    STATUS_AMBIGUOUS, STATUS_OK, STRUCTURE_ID_RE, accession_from_structure_id, classify_ec_class,
    ec_numbers_from_structure_id, normalize_ec_number_list,
)


VIEW_FORMAT = "deepmzyme_ec_clean30_view_v1"
# Schema 2: read-guard rules instead of absolute roots, overlay record, location rule. Schema-1 views
# (the G1 smoke view) no longer load; their files stay as records and are read by exact name only.
VIEW_SCHEMA_VERSION = 2
RETIRED_VIEW_SCHEMA_VERSIONS: frozenset[int] = frozenset({1})
KIND_FOLD_VIEW = "fold_training_view"
KIND_SMOKE_VIEW = "smoke_view"

VIEW_MANIFEST_NAME = "view_manifest.json"
ALLOWLIST_NAME = "allowlist.txt"
STRUCTURES_DIR = "structures"
STRUCTURE_MANIFEST_NAME = "structure_manifest.csv"
SUMMARY_NAME = "site_summary_train.csv"
ESM_DIR = "esm"
EXTERNAL_DIR = "external"
EXTERNAL_FEATURE_FILE = "residue_features.json"
RING_DIR = "ring"
PROBE_IDS_NAME = "probe_structure_ids.txt"
SMOKE_MEMBERSHIP_NAME = "smoke_membership.csv"
INNER_MEMBERSHIP_NAME = "inner_membership.csv"
# Written by ``certify-inner`` (ec_clean30.identity.INNER_CERTIFICATE) next to the inner membership.
INNER_CERTIFICATE_NAME = "inner_certification_receipt.json"
INNER_CERTIFICATE_STEP = "certify_inner"
CHAIN_MEAN_DIRS: Mapping[str, str] = {"final": "chain_means_final", "block26": "chain_means_block26"}
CACHE_DIR = "caches"
CACHE_SUBDIRS: tuple[str, ...] = ("parse", "graph", "tmp", "dry_run")
RECEIPTS_DIR = "receipts"
WRITABLE_DIRS: tuple[str, ...] = (CACHE_DIR, RECEIPTS_DIR, *CHAIN_MEAN_DIRS.values())
VERIFY_RECEIPT_NAME = "verify_view_receipt.json"
DRY_RUN_RECEIPT_NAME = "dry_run_receipt.json"
FILE_MODES_RECEIPT_NAME = "file_modes.json"
FILE_MODES_APPLIED_RECEIPT_NAME = "file_modes_applied.json"
# Written by the overlay builder at the overlay root (the data root of an overlay view).
OVERLAY_MANIFEST_NAME = "overlay_manifest.json"

STRUCTURE_MANIFEST_FIELDS = ("structure_name", "relative_path", "sha256", "size_bytes")
STRUCTURE_SUFFIXES = (".pdb", ".cif", ".mmcif")
SUMMARY_REQUIRED_COLUMNS = ("structure", "chain_resi", "metaltype", "ecnumber", "whether_catalytic")
SMOKE_N_FOLDS = 2
DEFAULT_PROBE_COUNT = 8
SMOKE_PROBE_COUNT = 4
PROBE_RULE = "first N eligible view structures ordered by SHA-256 of the structure ID (provisional until Freeze 1)"
SMOKE_SUBSET_RULE = ("eligible training structures ordered by SHA-256 of '<seed>:<structure ID>', first N; "
                     "folds alternate in that order")
EXTERNAL_FEATURES_POLICY = (
    "copied unchanged by exact name <external root>/<stem>/residue_features.json; the features were computed on the "
    "full structure, including metals outside the selected sites (decision D2 open until Freeze 1)"
)
# A view built from a data overlay states the overlay's decision A1 (D2) option and its policy text, both read from
# the overlay manifest. The texts live here because the view states them; the overlay builder writes them
# (ec_clean30.overlay.D2_POLICY). An overlay with another option, or another text for its option, is refused.
OVERLAY_D2_POLICY: Mapping[str, str] = {
    "b": "external features copied unchanged; they were computed on the full structure, including metals outside the "
         "selected sites (decision A1 option b; disclosed)",
    "d": "external features derived from the source files: custom_charge_distance_proxy counts only the protein's "
         "selected (catalytic) metal atoms, so a metal atom outside the selected sites, including a copy of a "
         "selected site from another donor, counts nowhere; the residue-pair term and every other value are kept "
         "as stored; PROPKA's dpka_titr still reflects every metal of the structure (decision A1 option d; "
         "disclosed)",
}
OVERLAY_D2_OPTIONS: tuple[str, ...] = tuple(OVERLAY_D2_POLICY)
OVERLAY_EXTERNAL_FEATURES_POLICY = (
    "copied unchanged by exact name <external root>/<stem>/residue_features.json from data overlay {overlay_id}, "
    "built under decision A1 (D2) option {option}: {policy}"
)
SOURCE_ROOT_BASIS = (
    "docs/DATASETS.md (v10 bundle roots esm_embeddings and updated_feature_extraction; v11 cache readiness for all "
    "740 shared CLEAN structures, esmc_300m, width 960) and src/project_paths.py defaults"
)
DATA_FILE_MODE = 0o444
DATA_DIR_MODE = 0o555
WRITABLE_DIR_MODE = 0o755
FILE_MODES_ENFORCED = "enforced"
FILE_MODES_IGNORED = "ignored"   # chmod returned, but the modes did not change (e.g. ntfs-3g without permissions)
FILE_MODES_REFUSED = "refused"   # chmod raised (PermissionError, EPERM, ...)

# Where views live, relative to the campaign root (whose name is the campaign ID).
VIEWS_DIR = "views"
RUNS_DIR = "runs"
SMOKE_VIEWS_DIR = "smoke_views"
SMOKE_RUNS_DIR = "smoke_runs"
# CLEAN embedding farms and active CLEAN fit folders on the internal disk, below the project data root.
CAMPAIGN_LOCAL_DIR_NAME = "campaign_local"

# Shared or test-bearing roots below a data root that a fit may never read (child guard). They are
# expanded below the view's source data root, the project's data root, the real paths of both, and every
# ``DeepMzyme_Data`` folder that contains the view (the campaign folder may be a symlink to another disk
# that holds its own copies of the shared caches).
DATA_DIR_NAME = "DeepMzyme_Data"
FORBIDDEN_DATA_ROOT_NAMES: tuple[str, ...] = (
    "CLEAN_30_main",
    "CLEAN_30_shared",
    "CLEAN_30_shared_single_donor_supported_metal_conservative",
    *(f"CLEAN_30_train_test_split_{fold}" for fold in range(N_OUTER_FOLDS)),
    "CLEAN_all_train_valid_splits",
    "structure_store",
    "esm_embeddings",
    "updated_feature_extraction",
    "RING_features",
    "DeepMzyme_Colab_Bundles",
    "bundle_metadata",
    "CARE_task1_30_clusterRes30_train_test_metallo",
    "CARE_dataset",
    "train_and_test_sets_structures_exact_pinmymetal",
    "train_and_test_sets_structures_common_pdbid_70_30_pinmymetal",
    "train_and_test_sets_structures_non_overlapped_pinmymetal",
    "train_and_test_sets_structures_harsh_pinmymetal",
    "train_and_test_sets_structures_zenodo_pmm_exact",
    "notebook_outputs",
    "backups",
    CAMPAIGN_LOCAL_DIR_NAME,
)
# Repository folders with test-side or mixed train/test files, past reports and run outputs.
FORBIDDEN_REPO_PATHS: tuple[str, ...] = (
    "prepare_training_and_test_set",
    "CLEAN_prepare_training_and_test_set",
    "CARE_prepare_training_and_test_set",
    "docs/notebook_outputs",
    "runs",
)
# Campaign folders that hold the 740-protein union, test-side or identity outputs, CLEAN work, records
# or other folds' material; a fit of any view may never read them. Real views also may not read the smoke
# folders, smoke views not the real ``views`` and ``runs``, and a fold-k view not ``views/fold<j>`` or
# ``runs/fold<j>`` (j != k): every fold-j view holds fold-k test proteins as training proteins.
CAMPAIGN_FORBIDDEN_CHILDREN: tuple[str, ...] = (
    "overlay", "identity", "identity_stdout", "labels", "cohort", "r2_lists", "clean", "clean_sequence_map",
    "clean_harness", "g1_smoke", "freeze", "simulation", "scoring_pass", "timing", "plans",
)
# Campaign folders whose entries other than the view's own name are forbidden for a smoke view: a smoke
# view of fold j holds fold-k test proteins as training proteins, and the rules cannot name the other
# smoke views in advance, so the entry names (only) are listed when the rules are expanded.
SMOKE_SIBLING_DIRS: tuple[str, ...] = (SMOKE_VIEWS_DIR, SMOKE_RUNS_DIR)
READ_GUARD_RULE_VERSION = 1
READ_GUARD_BASES = (
    "data_root_children and source_layout_dirs below each data root: the view's recorded source data root, every "
    "DeepMzyme_Data folder that contains the view (given and real path) and the project data root of the running "
    "checkout; campaign_children below the campaign root two levels above the view (given and real path); "
    "campaign_sibling_dirs: every entry of these campaign folders except the view's own name (entry names listed "
    "on the running machine, nothing inside opened); repository_paths below the running checkout; "
    "pinmymetal_sets_root is project_paths.PINMYMETAL_SETS_DIR of the running checkout"
)
GENERATOR_FUNCTIONS: tuple[str, ...] = (
    "_generate_missing_esm_embeddings",
    "_generate_missing_updated_external_features",
    "_generate_missing_ring_edges",
)
SYSTEM_READ_ROOTS: tuple[str, ...] = ("/proc", "/sys", "/dev", "/etc", "/usr", "/lib", "/lib64")
SYSTEM_WRITE_PATHS: tuple[str, ...] = ("/dev/null",)
# Files the probe may read under src/ (code only; any other file there is an outside read).
CODE_READ_SUFFIXES: tuple[str, ...] = (".py", ".pyc")
# Programs the trainer's preparation path may start (provenance and device queries).
ALLOWED_SUBPROCESS_PROGRAMS: frozenset[str] = frozenset({"git", "uname", "nvidia-smi"})
_LISTING_EVENTS = frozenset({"os.listdir", "os.scandir", "glob.glob", "glob.glob/2"})
_WRITE_EVENTS = frozenset({"os.mkdir", "os.rename", "os.replace", "os.remove", "os.rmdir", "os.unlink",
                           "os.chmod", "os.truncate", "shutil.copyfile", "shutil.move", "os.link", "os.symlink",
                           "shutil.rmtree"})


class ViewError(ValueError):
    """A view cannot be built from its sources, or its build request is refused."""


class ViewVerificationError(ValueError):
    """A built view does not verify; ``problems`` lists every failed check."""

    def __init__(self, problems: Sequence[str]):
        self.problems = list(problems)
        super().__init__("View verification failed:\n  " + "\n  ".join(self.problems[:40]))


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def _real(path: str | os.PathLike) -> str:
    return os.path.normcase(os.path.realpath(os.fspath(path)))


def _under(path: str, roots: Iterable[str]) -> bool:
    return any(path == root or path.startswith(root.rstrip(os.sep) + os.sep) for root in roots)


@dataclass(frozen=True)
class SourceLayout:
    """Where one fold's training inputs live (exact names only; see the module docstring)."""

    data_root: Path
    esm_dir: Path
    external_root: Path

    @classmethod
    def from_data_root(cls, data_root: str | os.PathLike, *, esm_dir: Optional[str | os.PathLike] = None,
                       external_root: Optional[str | os.PathLike] = None) -> "SourceLayout":
        root = Path(os.path.abspath(os.fspath(data_root)))
        return cls(root, Path(os.path.abspath(os.fspath(esm_dir))) if esm_dir else root / "esm_embeddings",
                   Path(os.path.abspath(os.fspath(external_root))) if external_root
                   else root / "updated_feature_extraction")

    def fold_train_dir(self, fold: int) -> Path:
        return self.data_root / f"CLEAN_30_train_test_split_{int(fold)}" / "train"

    def training_manifest(self, fold: int) -> Path:
        return self.fold_train_dir(fold) / STRUCTURE_MANIFEST_NAME

    def training_csv(self, fold: int) -> Path:
        return self.data_root / "CLEAN_30_main" / "folds" / f"CLEAN_30_train_test_split_{int(fold)}_train.csv"

    def to_dict(self) -> dict[str, str]:
        return {"data_root": str(self.data_root), "esm_dir": str(self.esm_dir), "external_root": str(self.external_root)}


def _path_forms(path: str | os.PathLike) -> list[Path]:
    """The given (absolute) and the real form of a path, without duplicates."""
    return list(dict.fromkeys([Path(os.path.abspath(os.fspath(path))), Path(os.path.realpath(os.fspath(path)))]))


def view_location_problem(view_root: str | os.PathLike, *, smoke: bool, fold: int) -> Optional[str]:
    """Why ``view_root`` is not a valid view location (``None`` when it is).

    A fold view sits at ``<campaign root>/views/fold<k>``, a smoke view at
    ``<campaign root>/smoke_views/<name>``; the campaign root's name is the campaign ID.
    """
    root = Path(os.path.abspath(os.fspath(view_root)))
    campaign_root = root.parent.parent
    if smoke:
        if root.parent.name != SMOKE_VIEWS_DIR:
            return f"a smoke view lives at <campaign root>/{SMOKE_VIEWS_DIR}/<name>, not at {root}"
    elif root.parent.name != VIEWS_DIR or root.name != f"fold{int(fold)}":
        return f"the fold-{int(fold)} view lives at <campaign root>/{VIEWS_DIR}/fold{int(fold)}, not at {root}"
    if campaign_root.name != CAMPAIGN_ID:
        return f"the campaign root {campaign_root} is not named {CAMPAIGN_ID!r}"
    return None


def campaign_root_of(view_root: str | os.PathLike, *, smoke: bool, fold: int) -> Path:
    """The campaign root two levels above a view at a valid location; raises ``ViewError`` otherwise."""
    problem = view_location_problem(view_root, smoke=smoke, fold=fold)
    if problem is not None:
        raise ViewError(f"View location refused: {problem}.")
    return Path(os.path.abspath(os.fspath(view_root))).parent.parent


def _campaign_children(*, fold: int, smoke: bool) -> list[str]:
    children = list(CAMPAIGN_FORBIDDEN_CHILDREN)
    if smoke:
        children += [VIEWS_DIR, RUNS_DIR]
    else:
        others = [other for other in range(N_OUTER_FOLDS) if other != int(fold)]
        children += [SMOKE_VIEWS_DIR, SMOKE_RUNS_DIR]
        children += [f"{VIEWS_DIR}/fold{other}" for other in others] + [f"{RUNS_DIR}/fold{other}" for other in others]
    return children


def source_layout_dirs(layout: SourceLayout) -> list[str]:
    """The layout's ESMC and external folders relative to its data root (POSIX; both must lie inside it)."""
    layout_dirs = []
    for flag, directory in (("--esm-dir", layout.esm_dir), ("--external-root", layout.external_root)):
        try:
            relative = Path(os.path.abspath(directory)).relative_to(Path(os.path.abspath(layout.data_root)))
        except ValueError:
            raise ViewError(f"{flag} {directory} lies outside the data root {layout.data_root}; the read guard records "
                            "the source folders relative to the data root, so they must lie inside it.") from None
        if not relative.parts:
            raise ViewError(f"{flag} {directory} is the data root itself; it must be a folder inside it.")
        layout_dirs.append(relative.as_posix())
    return list(dict.fromkeys(layout_dirs))


def read_guard_rules(layout: SourceLayout, *, fold: int, smoke: bool) -> dict[str, Any]:
    """The read-guard rules a view manifest records (relative names only; expanded where the view runs).

    The source layout's ESMC and external folders must lie inside the data root, so they can
    be recorded relative to it.
    """
    return {
        "rule_version": READ_GUARD_RULE_VERSION,
        "data_root_children": list(FORBIDDEN_DATA_ROOT_NAMES),
        "source_layout_dirs": source_layout_dirs(layout),
        "campaign_children": _campaign_children(fold=fold, smoke=smoke),
        "campaign_sibling_dirs": list(SMOKE_SIBLING_DIRS) if smoke else [],
        "repository_paths": list(FORBIDDEN_REPO_PATHS),
        "pinmymetal_sets_root": True,
        "bases": READ_GUARD_BASES,
    }


_RULE_NAME_KEYS = ("data_root_children", "source_layout_dirs", "campaign_children", "campaign_sibling_dirs",
                   "repository_paths")


def read_guard_rule_problems(rules: Mapping[str, Any], *, fold: int, smoke: bool,
                             layout: Optional[SourceLayout] = None) -> list[str]:
    """Why recorded rules are malformed or weaker than this code's rules for the same kind and fold.

    With ``layout`` (the view's recorded source layout), ``source_layout_dirs`` must also hold
    that layout's ESMC and external folders.
    """
    if not isinstance(rules, Mapping) or rules.get("rule_version") != READ_GUARD_RULE_VERSION:
        return [f"the read-guard rules are not rule version {READ_GUARD_RULE_VERSION}"]
    problems = []
    for key in _RULE_NAME_KEYS:
        values = rules.get(key)
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            problems.append(f"read-guard rule {key} is not a list of names")
            continue
        for value in values:
            parts = Path(value).parts
            if not value or Path(value).is_absolute() or ".." in parts or "." in parts or "\\" in value:
                problems.append(f"read-guard rule {key} holds {value!r}; rules are plain relative names")
    if problems:
        return problems
    expected: dict[str, Sequence[str]] = {"data_root_children": FORBIDDEN_DATA_ROOT_NAMES,
                                          "campaign_children": _campaign_children(fold=fold, smoke=smoke),
                                          "campaign_sibling_dirs": SMOKE_SIBLING_DIRS if smoke else (),
                                          "repository_paths": FORBIDDEN_REPO_PATHS}
    if layout is not None:
        try:
            expected["source_layout_dirs"] = source_layout_dirs(layout)
        except ViewError as exc:
            problems.append(f"the recorded source layout cannot be checked against the read guard: {exc}")
    for key, names in expected.items():
        missing = sorted(set(names) - set(rules[key]))
        if missing:
            problems.append(f"read-guard rule {key} lacks {missing[:6]} (weaker than this code's rules)")
    if rules.get("pinmymetal_sets_root") is not True:
        problems.append("read-guard rules do not forbid the PinMyMetal sets root")
    return problems


def _guard_data_roots(view_root: Path, source_data_root: Optional[str | os.PathLike]) -> list[Path]:
    """Data roots whose shared and test-bearing folders are forbidden (see ``READ_GUARD_BASES``)."""
    from project_paths import DATA_DIR

    candidates: list[Path] = [] if source_data_root is None else [Path(os.fspath(source_data_root))]
    for form in _path_forms(view_root):
        candidates.extend(ancestor for ancestor in (form, *form.parents) if ancestor.name == DATA_DIR_NAME)
    candidates.append(Path(DATA_DIR))
    return list(dict.fromkeys(form for candidate in candidates for form in _path_forms(candidate)))


def _sibling_entries(folder: Path, own_name: str) -> list[Path]:
    """Every entry of ``folder`` except ``own_name`` (entry names only; nothing inside is opened)."""
    try:
        names = sorted(os.listdir(folder))
    except (FileNotFoundError, NotADirectoryError):
        return []
    except OSError as exc:
        raise ViewError(f"Cannot list the entry names of {folder} to expand the read guard: "
                        f"{type(exc).__name__}") from None
    return [folder / name for name in names if name != own_name]


def _expand_rules(rules: Mapping[str, Any], *, view_root: Path,
                  source_data_root: Optional[str | os.PathLike]) -> list[str]:
    from project_paths import PINMYMETAL_SETS_DIR

    roots: list[Path] = []
    for data_root in _guard_data_roots(view_root, source_data_root):
        roots.extend(data_root / name for name in (*rules["data_root_children"], *rules["source_layout_dirs"]))
    for campaign_root in dict.fromkeys(form.parent.parent for form in _path_forms(view_root)):
        roots.extend(campaign_root / child for child in rules["campaign_children"])
        for sibling_dir in rules["campaign_sibling_dirs"]:
            roots.extend(_sibling_entries(campaign_root / sibling_dir, view_root.name))
    roots.extend(REPO_ROOT / relative for relative in rules["repository_paths"])
    if rules.get("pinmymetal_sets_root"):
        roots.append(Path(PINMYMETAL_SETS_DIR))
    return list(dict.fromkeys(str(Path(os.path.abspath(root))) for root in roots))


def expand_read_guard(rules: Mapping[str, Any], *, view_root: str | os.PathLike, smoke: bool, fold: int,
                      layout: Optional[SourceLayout] = None) -> list[str]:
    """Expand recorded read-guard rules for a view on this machine; the roots a fit of the view may never read.

    ``layout`` is the view's source layout (its data root is one expansion base, and its ESMC
    and external folders must be in the rules). Raises ``ViewError`` when the rules are
    malformed or weaker than this code's, when the view is not at a valid location, or when
    an expanded root contains the view (refused, never dropped).
    """
    campaign_root_of(view_root, smoke=smoke, fold=fold)
    problems = read_guard_rule_problems(rules, fold=fold, smoke=smoke, layout=layout)
    if problems:
        raise ViewError("The view's read guard is refused: " + "; ".join(problems))
    view_root = Path(os.path.abspath(os.fspath(view_root)))
    roots = _expand_rules(rules, view_root=view_root, source_data_root=None if layout is None else layout.data_root)
    forms = [str(form) for form in _path_forms(view_root)] + [_real(view_root)]
    for root in roots:
        if any(_under(form, [root, _real(root)]) for form in forms):
            raise ViewError(f"View folder {view_root} lies inside the forbidden read root {root} (a shared, "
                            "test-bearing or other-fold folder); views live in a campaign folder (never under "
                            "CLEAN_30_main or another shared root).")
    return roots


def forbidden_read_roots(layout: SourceLayout, *, view_root: str | os.PathLike, fold: int,
                         smoke: bool = False) -> list[str]:
    """Roots a fit of a view at ``view_root`` built from ``layout`` may never read (rules expanded here)."""
    return expand_read_guard(read_guard_rules(layout, fold=fold, smoke=smoke), view_root=view_root, smoke=smoke,
                             fold=fold, layout=layout)


def runs_dir_problem(view: "View", runs_dir: str | os.PathLike) -> Optional[str]:
    """Why ``runs_dir`` may not hold the runs of ``view``, or ``None``.

    Refused: a folder inside the view or inside an expanded read-guard root (given or real
    path), and any folder whose real path lies outside ``view.default_runs_dir``
    (``<campaign root>/runs/fold<k>``, or ``<campaign root>/smoke_runs/<name>`` for a smoke
    view; only the campaign root is resolved), so every fold's runs sit where the other
    folds' guards forbid them.
    """
    given, real = os.path.abspath(os.fspath(runs_dir)), _real(runs_dir)
    if _under(real, [_real(view.root)]) or _under(given, [str(view.root)]):
        return f"the runs folder {runs_dir} lies inside the view {view.root}"
    for root in view.forbidden_read_roots:
        if _under(real, [_real(root)]) or _under(given, [root]):
            return f"the runs folder {runs_dir} lies inside the forbidden read root {root}"
    expected = view.default_runs_dir
    if not _under(real, [os.path.join(_real(view.campaign_root), os.fspath(expected.relative_to(view.campaign_root)))]):
        place = f"{SMOKE_RUNS_DIR}/{view.root.name}" if view.smoke else f"{RUNS_DIR}/fold{view.fold}"
        return (f"the runs folder {runs_dir} of this {view.kind} must be <campaign root>/{place} or a folder inside "
                f"it (real path; expected {expected})")
    return None


def _check_relative_source(relative: str, *, what: str) -> None:
    path = Path(relative)
    if path.is_absolute():
        raise ViewError(f"{what} {relative!r} is absolute; source manifests must hold relative paths.")
    if any(names_test_split(part) for part in path.parts if part not in (".", "..")):
        raise HeldOutInputError(f"{what} {relative!r} passes through a test or mixed metadata folder; refused.")


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]], bytes]:
    data = path.read_bytes()
    reader = csv.DictReader(io.StringIO(data.decode("utf-8")))
    return list(reader.fieldnames or []), [dict(row) for row in reader], data


def _structure_parts(stem: str) -> tuple[str, str, str]:
    match = STRUCTURE_ID_RE.match(stem)
    if match is None:
        raise ViewError(f"Structure name {stem!r} does not follow '<accession>__chain_<chain>__EC_<ec>'.")
    chain = match.group("chain").strip() or "_"
    return match.group("pdbid"), chain, match.group("ec")


def esm_file_name(stem: str) -> str:
    return f"{stem}_chain_{_structure_parts(stem)[1]}_esmc.pt"


def esm_sidecar_name(stem: str) -> str:
    return esm_file_name(stem) + ".json"


def ids_sha256(ids: Iterable[str]) -> str:
    return canonical_sha256(sorted(ids))


def _lines_bytes(lines: Iterable[str]) -> bytes:
    return "".join(f"{line}\n" for line in lines).encode("utf-8")


def _ordered_by_hash(items: Iterable[str], salt: str = "") -> list[str]:
    return sorted(items, key=lambda item: (hashlib.sha256(f"{salt}{item}".encode("utf-8")).hexdigest(), item))


@dataclass(frozen=True)
class _SourceRow:
    stem: str
    structure_name: str
    accession: str
    path: Path
    sha256: str
    size_bytes: int


def read_training_sources(layout: SourceLayout, fold: int) -> dict[str, Any]:
    """Read fold k's training manifest and training site table; check that they describe the same proteins."""
    if not 0 <= int(fold) < N_OUTER_FOLDS:
        raise ViewError(f"Fold must be in 0..{N_OUTER_FOLDS - 1}, got {fold!r}.")
    manifest_path, csv_path = layout.training_manifest(fold), layout.training_csv(fold)
    for path in (manifest_path, csv_path):
        if is_test_named_path(path, relative_to=layout.data_root):
            raise HeldOutInputError(f"Training source {path} is test-named; refused.")
    if not csv_path.name.endswith("_train.csv"):
        raise HeldOutInputError(f"Training site table {csv_path.name} is not a *_train.csv file; refused.")
    for path in (manifest_path, csv_path):
        if not path.is_file():
            raise ViewError(f"Training source not found: {path}")
        if os.path.islink(path):
            raise ViewError(f"Training source {path} is a symlink; refused.")

    header, csv_rows, csv_bytes = _read_csv(csv_path)
    missing = [column for column in SUMMARY_REQUIRED_COLUMNS if column not in header]
    if missing:
        raise ViewError(f"{csv_path.name} lacks the columns {missing}.")
    rows_by_accession: dict[str, list[dict[str, str]]] = {}
    for row in csv_rows:
        accession = (row.get("structure") or "").strip()
        if not accession:
            raise ViewError(f"{csv_path.name} has a row with an empty 'structure' value.")
        rows_by_accession.setdefault(accession, []).append(row)

    manifest_header, manifest_rows, manifest_bytes = _read_csv(manifest_path)
    missing = [column for column in STRUCTURE_MANIFEST_FIELDS if column not in manifest_header]
    if missing:
        raise ViewError(f"{manifest_path.name} lacks the columns {missing}.")
    train_dir = manifest_path.parent
    forbidden_main = [_real(layout.data_root / "CLEAN_30_main")]
    sources: dict[str, _SourceRow] = {}
    accessions: dict[str, str] = {}
    for number, row in enumerate(manifest_rows, start=2):
        name = (row.get("structure_name") or "").strip()
        relative = (row.get("relative_path") or "").strip()
        digest = (row.get("sha256") or "").strip().lower()
        size_text = (row.get("size_bytes") or "").strip()
        if not name or not relative or not digest or not size_text:
            raise ViewError(f"Incomplete row {number} in {manifest_path.name}.")
        if Path(name).name != name or Path(name).suffix.lower() not in STRUCTURE_SUFFIXES:
            raise ViewError(f"Row {number} of {manifest_path.name}: invalid structure name {name!r}.")
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ViewError(f"Row {number} of {manifest_path.name}: invalid SHA-256.")
        _check_relative_source(relative, what=f"Row {number} relative path")
        stem = Path(name).stem
        accession = accession_from_structure_id(stem)
        if stem in sources or accession in accessions:
            raise ViewError(f"{manifest_path.name} lists {stem!r} or its accession more than once.")
        path = train_dir / relative
        if _under(_real(path), forbidden_main):
            raise HeldOutInputError(f"Row {number} of {manifest_path.name} resolves into CLEAN_30_main; refused.")
        sources[stem] = _SourceRow(stem, name, accession, path, digest, int(size_text))
        accessions[accession] = stem

    if set(accessions) != set(rows_by_accession):
        raise ViewError(
            f"Fold {fold}: the training manifest lists {len(accessions)} accessions and the training site table "
            f"{len(rows_by_accession)}; {len(set(accessions) - set(rows_by_accession))} and "
            f"{len(set(rows_by_accession) - set(accessions))} are not shared. Both must describe the same proteins."
        )
    ec_mismatch = []
    for accession, rows in rows_by_accession.items():
        stem_ec = normalize_ec_number_list(_structure_parts(accessions[accession])[2])
        if any(normalize_ec_number_list(row.get("ecnumber") or "") != stem_ec for row in rows):
            ec_mismatch.append(accessions[accession])
    if ec_mismatch:
        raise ViewError(f"{len(ec_mismatch)} training structures carry an EC number that differs from their site rows, "
                        f"e.g. {sorted(ec_mismatch)[:3]}.")

    status: dict[str, str] = {}
    fatal = []
    for stem in sources:
        result = classify_ec_class(ec_numbers_from_structure_id(stem))
        status[stem] = result.status
        if result.status not in (STATUS_OK, STATUS_AMBIGUOUS):
            fatal.append(f"{stem} ({result.status})")
    if fatal:
        raise ViewError(f"{len(fatal)} training structures have EC annotations without an EC class: {sorted(fatal)[:10]}")
    return {
        "manifest_path": manifest_path, "manifest_sha256": sha256_bytes(manifest_bytes),
        "manifest_n_rows": len(manifest_rows),
        "csv_path": csv_path, "csv_sha256": sha256_bytes(csv_bytes), "csv_bytes": csv_bytes, "csv_header": header,
        "csv_n_rows": len(csv_rows), "rows_by_accession": rows_by_accession,
        "sources": sources, "status": status,
        "accessions_sha256": ids_sha256(accessions),
    }


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _copy_exact(source: Path, target: Path, *, expected_sha256: Optional[str] = None,
                expected_size: Optional[int] = None) -> tuple[str, int]:
    """Byte copy of one regular file (never a symlink); returns (sha256, size)."""
    if os.path.islink(source):
        raise ViewError(f"Source {source} is a symlink; views hold byte copies of regular files only.")
    if not source.is_file():
        raise FileNotFoundError(str(source))
    data = source.read_bytes()
    digest = sha256_bytes(data)
    if expected_size is not None and len(data) != int(expected_size):
        raise ViewError(f"{source.name}: size {len(data)} differs from the manifest's {expected_size}.")
    if expected_sha256 is not None and digest != expected_sha256:
        raise ViewError(f"{source.name}: SHA-256 differs from the manifest.")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(data)
    return digest, len(data)


def _write_new(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
    return sha256_bytes(data)


def _csv_bytes(columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _check_output_location(out_dir: Path, rules: Mapping[str, Any], *, layout: SourceLayout, fold: int,
                           smoke: bool) -> list[str]:
    """Refuse a view folder at the wrong place or inside a forbidden root; returns the expanded roots."""
    if is_test_named_path(out_dir):
        raise HeldOutInputError(f"View folder {out_dir} is test-named; refused.")
    roots = expand_read_guard(rules, view_root=out_dir, smoke=smoke, fold=fold, layout=layout)
    real = _real(out_dir)
    if _under(real, [_real(SRC_ROOT)]):
        raise ViewError(f"View folder {out_dir} lies inside the source tree.")
    if real == _real(REPO_ROOT) or _under(_real(REPO_ROOT), [real]):
        raise ViewError(f"View folder {out_dir} is the repository root or contains it.")
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise ViewError(f"View folder {out_dir} already exists and is not empty; use a new folder.")
    return roots


def _chmod(path: Path, mode: int) -> None:
    """The one chmod call of view permissions (replaceable to model filesystems that ignore or refuse it)."""
    os.chmod(path, mode)


def filesystem_record(path: str | os.PathLike) -> Optional[dict[str, str]]:
    """Filesystem type, source, mount point and options of the mount holding ``path`` (``findmnt``), or ``None``."""
    try:
        completed = subprocess.run(["findmnt", "-n", "-P", "-o", "FSTYPE,SOURCE,TARGET,OPTIONS", "--target",
                                    os.fspath(path)], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    record = {}
    for token in shlex.split(completed.stdout.strip().splitlines()[0]):
        key, _sep, value = token.partition("=")
        record[key.lower()] = value
    return record


def _set_permissions(root: Path) -> dict[str, Any]:
    """Data files read-only, data folders read-only and listable; writable folders stay writable.

    Never raises for a chmod: each failure is caught and counted. The outcome is
    ``enforced`` (every mode set), ``ignored`` (chmod returned but a mode did not change) or
    ``refused`` (a chmod raised).
    """
    writable = {root / name for name in WRITABLE_DIRS}
    targets: list[tuple[Path, int]] = []
    for current, dirs, files in os.walk(root, topdown=True):
        current_path = Path(current)
        if any(current_path == path or path in current_path.parents for path in writable):
            targets.append((current_path, WRITABLE_DIR_MODE))
            continue
        targets.extend((current_path / name, DATA_FILE_MODE) for name in files)
    for current, dirs, _files in os.walk(root, topdown=False):
        current_path = Path(current)
        if any(current_path == path or path in current_path.parents for path in writable):
            continue
        targets.append((current_path, DATA_DIR_MODE))
    refused: list[str] = []
    ignored: list[str] = []
    for path, mode in targets:
        relative = path.relative_to(root).as_posix()
        try:
            _chmod(path, mode)
            actual = stat.S_IMODE(os.lstat(path).st_mode) & 0o777
        except OSError as exc:
            refused.append(f"{relative} ({type(exc).__name__}: errno {exc.errno})")
            continue
        if actual != mode:
            ignored.append(f"{relative} (mode {actual:o}, wanted {mode:o})")
    outcome = FILE_MODES_REFUSED if refused else FILE_MODES_IGNORED if ignored else FILE_MODES_ENFORCED
    return {"outcome": outcome, "n_paths": len(targets), "n_refused": len(refused), "n_ignored": len(ignored),
            "refused_sample": refused[:10], "ignored_sample": ignored[:10],
            "modes": {"data_file": f"{DATA_FILE_MODE:o}", "data_dir": f"{DATA_DIR_MODE:o}",
                      "writable_dir": f"{WRITABLE_DIR_MODE:o}"},
            "checked_utc": utc_now(), "filesystem": filesystem_record(root)}


def probe_file_modes(directory: Path) -> dict[str, Any]:
    """Whether chmod changes modes in ``directory`` now (a temporary file is created, changed and removed).

    ``enforced`` is ``None`` when no probe file can be created there.
    """
    probe = Path(directory) / f".file_mode_probe-{os.getpid()}-{uuid.uuid4().hex[:12]}"
    try:
        with probe.open("xb"):
            pass
    except OSError as exc:
        return {"enforced": None, "outcome": None, "detail": f"cannot create a probe file: {type(exc).__name__}",
                "filesystem": filesystem_record(directory)}
    try:
        try:
            _chmod(probe, 0o444)
            read_only = stat.S_IMODE(os.lstat(probe).st_mode) & 0o777
            _chmod(probe, 0o644)
            writable = stat.S_IMODE(os.lstat(probe).st_mode) & 0o777
        except OSError as exc:
            outcome, detail = FILE_MODES_REFUSED, f"{type(exc).__name__}: errno {exc.errno}"
        else:
            enforced = read_only == 0o444 and writable == 0o644
            outcome = FILE_MODES_ENFORCED if enforced else FILE_MODES_IGNORED
            detail = f"modes after chmod 444 and 644: {read_only:o}, {writable:o}"
    finally:
        try:
            probe.unlink()
        except OSError:
            pass
    return {"enforced": outcome == FILE_MODES_ENFORCED, "outcome": outcome, "detail": detail,
            "filesystem": filesystem_record(directory)}


def _data_relative(path: Path, data_root: Path) -> Optional[str]:
    """``path`` relative to the data root (POSIX, normalised), or ``None`` when it lies outside it."""
    relative = os.path.relpath(os.path.normpath(os.path.abspath(path)), os.path.abspath(data_root))
    if relative == os.curdir or relative.split(os.sep, 1)[0] == os.pardir:
        return None
    return Path(relative).as_posix()


def overlay_record(data_root: Path, used: Mapping[str, tuple[Optional[str], str]]) -> Optional[dict[str, Any]]:
    """The overlay a view is built from, after a light check; ``None`` when the data root is no overlay.

    ``used`` maps each source file the view copies (absolute path) to its path relative to
    the data root (``None`` outside it) and the SHA-256 of the copied bytes. The overlay manifest
    (``<data root>/overlay_manifest.json``, a regular file) must hold a non-empty
    ``overlay_id``, a ``files`` list of ``{path, sha256, ...}`` entries (paths relative to
    the overlay root) that lists every used file with the same SHA-256, and a ``d2`` block whose
    option is one of ``OVERLAY_D2_OPTIONS`` and whose policy text is ``OVERLAY_D2_POLICY`` of that
    option; the record carries both.
    """
    path = Path(data_root) / OVERLAY_MANIFEST_NAME
    if not os.path.lexists(path):
        return None
    if os.path.islink(path) or not path.is_file():
        raise ViewError(f"{path} is not a regular file; an overlay manifest is read only as one.")
    data = path.read_bytes()
    try:
        manifest = json.loads(data.decode("utf-8"))
    except ValueError as exc:
        raise ViewError(f"{path} is not a JSON overlay manifest: {exc}") from exc
    overlay_id = manifest.get("overlay_id") if isinstance(manifest, Mapping) else None
    files = manifest.get("files") if isinstance(manifest, Mapping) else None
    if not isinstance(overlay_id, str) or not overlay_id.strip() or not isinstance(files, list):
        raise ViewError(f"{path} needs a non-empty 'overlay_id' and a 'files' list.")
    d2 = manifest.get("d2")
    d2_option = d2.get("option") if isinstance(d2, Mapping) else None
    d2_policy = d2.get("policy") if isinstance(d2, Mapping) else None
    if d2_option not in OVERLAY_D2_OPTIONS:
        raise ViewError(f"{path} records decision A1 (D2) option {d2_option!r}; a view knows only "
                        f"{list(OVERLAY_D2_OPTIONS)} and cannot state the external-feature policy of another.")
    if d2_policy != OVERLAY_D2_POLICY[d2_option]:
        raise ViewError(f"{path} records another policy text for D2 option {d2_option!r} than this code states for "
                        "it; a view states only that text.")
    listed: dict[str, Optional[str]] = {}
    n_duplicates = 0
    for item in files:
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
            raise ViewError(f"{path} has a 'files' entry without a path.")
        n_duplicates += item["path"] in listed
        listed[item["path"]] = item.get("sha256")
    if n_duplicates:
        # Counted, not named: the overlay lists other folds' proteins, so an entry may name a test-side protein.
        raise ViewError(f"{path} lists {n_duplicates} paths more than once; an overlay manifest names each file once.")
    outside = [Path(source).name for source, (relative, _digest) in used.items() if relative is None]
    if outside:
        raise ViewError(f"{len(outside)} inputs of this view lie outside the overlay root {data_root}, e.g. "
                        f"{sorted(outside)[:3]}; an overlay view reads only the overlay.")
    differing = sorted(relative for relative, digest in used.values() if listed.get(relative) != digest)
    if differing:
        raise ViewError(f"{len(differing)} inputs of this view are missing from the overlay manifest or differ from "
                        f"its SHA-256, e.g. {differing[:3]}.")
    return {"id": overlay_id, "manifest_file": OVERLAY_MANIFEST_NAME, "manifest_sha256": sha256_bytes(data),
            "n_files_checked": len(used), "d2_option": d2_option, "d2_policy": d2_policy}


def external_features_policy(overlay: Optional[Mapping[str, Any]]) -> str:
    """The view manifest's external-feature policy: the shared-root text, or the overlay's D2 option and policy."""
    if overlay is None:
        return EXTERNAL_FEATURES_POLICY
    return OVERLAY_EXTERNAL_FEATURES_POLICY.format(overlay_id=overlay["id"], option=overlay["d2_option"],
                                                   policy=overlay["d2_policy"])


def _file_role(relative: str) -> str:
    first = relative.split("/", 1)[0]
    return {STRUCTURES_DIR: "structure", ESM_DIR: "esm_residue", EXTERNAL_DIR: "external_features"}.get(first, "table")


def build_view(
    *,
    fold: int,
    layout: SourceLayout,
    out_dir: str | os.PathLike,
    smoke: bool = False,
    max_proteins: Optional[int] = None,
    seed: Optional[int] = None,
    bundle_id: Optional[str] = None,
    inner_membership: Optional[tuple[Path, str]] = None,
    inner_certificate: Optional[str | os.PathLike] = None,
    probe_count: Optional[int] = None,
    provenance: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Write a view of fold ``fold``'s training inputs to a new folder; return its manifest and SHA-256.

    ``smoke`` builds an engineering subset: ``max_proteins`` eligible structures chosen by
    ``seed`` plus a 2-fold smoke membership. A fold view's ``inner_membership`` (path, SHA-256)
    needs ``inner_certificate``, the ``inner_certification_receipt.json`` of ``certify-inner``
    (default: next to the membership file) that certifies exactly this file for this fold; both
    are copied into the view. Nothing is left at ``out_dir`` unless the build succeeds (it is
    assembled in a sibling staging folder and renamed). ``out_dir`` must follow the location
    rule (``view_location_problem``). File modes are set after the rename and never stop the
    build; their outcome is written to ``receipts/file_modes.json``.
    """
    from training.ec_membership import read_ec_fold_membership, write_ec_fold_membership

    fold = int(fold)
    out_dir = Path(os.path.abspath(os.fspath(out_dir)))
    rules = read_guard_rules(layout, fold=fold, smoke=smoke)
    _check_output_location(out_dir, rules, layout=layout, fold=fold, smoke=smoke)
    if smoke:
        if max_proteins is None or int(max_proteins) < SMOKE_N_FOLDS or seed is None:
            raise ViewError(f"A smoke view needs --max-proteins >= {SMOKE_N_FOLDS} and --seed.")
        if inner_membership is not None:
            raise ViewError("A smoke view writes its own smoke membership; --inner-membership is for fold views.")
    elif max_proteins is not None or seed is not None:
        raise ViewError("--max-proteins and --seed select a smoke subset; a fold view holds every training protein.")
    certificate = None
    if inner_membership is not None:
        certificate = read_inner_certificate(inner_membership, fold=fold, certificate_path=inner_certificate)
    elif inner_certificate is not None:
        raise ViewError("An inner certificate is given without its inner membership.")
    sources = read_training_sources(layout, fold)
    stems_all = sorted(sources["sources"])
    eligible_all = [stem for stem in stems_all if sources["status"][stem] == STATUS_OK]
    if smoke:
        selected = sorted(_ordered_by_hash(eligible_all, f"{int(seed)}:")[: int(max_proteins)])
        if len(selected) < SMOKE_N_FOLDS:
            raise ViewError(f"Fold {fold} has only {len(selected)} eligible training structures for a smoke view.")
    else:
        selected = stems_all
    eligible = [stem for stem in selected if sources["status"][stem] == STATUS_OK]
    excluded = [stem for stem in selected if sources["status"][stem] != STATUS_OK]

    staging = out_dir.parent / f".{out_dir.name}.partial-{os.getpid()}"
    if staging.exists():
        raise ViewError(f"Staging folder {staging} exists; remove it first.")
    staging.mkdir(parents=True)
    try:
        structure_rows, missing = [], []
        # Every source file copied, for the overlay check: absolute path -> (data-root relative path, SHA-256).
        used: dict[str, tuple[Optional[str], str]] = {
            str(sources["manifest_path"]): (_data_relative(sources["manifest_path"], layout.data_root),
                                            sources["manifest_sha256"]),
            str(sources["csv_path"]): (_data_relative(sources["csv_path"], layout.data_root), sources["csv_sha256"]),
        }

        def copy_source(source: Path, target: Path, **expected: Any) -> tuple[str, int]:
            digest, size = _copy_exact(source, target, **expected)
            used[str(source)] = (_data_relative(source, layout.data_root), digest)
            return digest, size

        for stem in selected:
            row = sources["sources"][stem]
            try:
                digest, size = copy_source(row.path, staging / STRUCTURES_DIR / row.structure_name,
                                           expected_sha256=row.sha256, expected_size=row.size_bytes)
            except FileNotFoundError:
                missing.append(f"{stem}: structure file")
                continue
            structure_rows.append({"structure_name": row.structure_name, "relative_path": row.structure_name,
                                   "sha256": digest, "size_bytes": size})
            for name in (esm_file_name(stem), esm_sidecar_name(stem)):
                try:
                    copy_source(layout.esm_dir / name, staging / ESM_DIR / name)
                except FileNotFoundError:
                    missing.append(f"{stem}: {name}")
            try:
                copy_source(layout.external_root / stem / EXTERNAL_FEATURE_FILE,
                            staging / EXTERNAL_DIR / stem / EXTERNAL_FEATURE_FILE)
            except FileNotFoundError:
                missing.append(f"{stem}: {EXTERNAL_FEATURE_FILE}")
        if missing:
            raise ViewError(f"{len(missing)} required inputs are missing (exact names only), e.g. {missing[:5]}")
        _write_new(staging / STRUCTURES_DIR / STRUCTURE_MANIFEST_NAME,
                   _csv_bytes(STRUCTURE_MANIFEST_FIELDS, structure_rows))

        if smoke:
            selected_accessions = {sources["sources"][stem].accession for stem in selected}
            rows = [row for accession in sorted(selected_accessions) for row in sources["rows_by_accession"][accession]]
            summary_sha = _write_new(staging / SUMMARY_NAME, _csv_bytes(sources["csv_header"], rows))
            summary_rows = len(rows)
        else:
            summary_sha = _write_new(staging / SUMMARY_NAME, sources["csv_bytes"])
            summary_rows = sources["csv_n_rows"]
        allowlist_sha = _write_new(staging / ALLOWLIST_NAME, _lines_bytes(selected))
        n_probe = int(probe_count) if probe_count is not None else (SMOKE_PROBE_COUNT if smoke else DEFAULT_PROBE_COUNT)
        if n_probe < 1:
            raise ViewError("The probe list needs at least one structure.")
        probe_ids = sorted(_ordered_by_hash(eligible)[:n_probe])
        probe_sha = _write_new(staging / PROBE_IDS_NAME, _lines_bytes(probe_ids))
        (staging / RING_DIR).mkdir()
        for name in WRITABLE_DIRS:
            (staging / name).mkdir()
        for name in CACHE_SUBDIRS:
            (staging / CACHE_DIR / name).mkdir()

        membership_record = None
        if smoke:
            ordered = _ordered_by_hash(eligible, f"{int(seed)}:")
            membership_rows = [{"structure_id": stem, "accession": sources["sources"][stem].accession,
                                "group_id": sources["sources"][stem].accession, "fold": str(position % SMOKE_N_FOLDS),
                                "ec_class": classify_ec_class(ec_numbers_from_structure_id(stem)).token}
                               for position, stem in enumerate(ordered)]
            membership_sha = write_ec_fold_membership(membership_rows, staging / SMOKE_MEMBERSHIP_NAME)
            membership = read_ec_fold_membership(staging / SMOKE_MEMBERSHIP_NAME, membership_sha)
            membership_record = {"role": "smoke", "file": SMOKE_MEMBERSHIP_NAME, "sha256": membership_sha,
                                 "identity_sha256": membership.identity_sha256, "n_folds": SMOKE_N_FOLDS,
                                 "n_rows": len(membership.rows), "group_rule": "one group per accession"}
        elif inner_membership is not None:
            source_path, source_sha = Path(inner_membership[0]), str(inner_membership[1])
            membership = read_ec_fold_membership(source_path, source_sha)
            ids = {row["structure_id"] for row in membership.rows}
            if ids != set(eligible):
                # No counts here: per-fold eligible counts are class numbers (one EC class per protein).
                raise ViewError("The inner membership's structures differ from the fold's eligible training "
                                "structures. They must be identical.")
            folds = sorted({int(row["fold"]) for row in membership.rows})
            if folds != list(range(N_INNER_FOLDS)):
                raise ViewError(f"The inner membership uses folds {folds}, expected 0..{N_INNER_FOLDS - 1}.")
            digest, _size = copy_source(source_path, staging / INNER_MEMBERSHIP_NAME, expected_sha256=source_sha)
            certificate_sha, _size = copy_source(Path(certificate["path"]), staging / INNER_CERTIFICATE_NAME,
                                                 expected_sha256=certificate["sha256"])
            membership_record = {"role": "inner", "file": INNER_MEMBERSHIP_NAME, "sha256": digest,
                                 "identity_sha256": membership.identity_sha256, "n_folds": N_INNER_FOLDS,
                                 "n_rows": len(membership.rows), "source_name": source_path.name,
                                 "certificate_file": INNER_CERTIFICATE_NAME, "certificate_sha256": certificate_sha}
        overlay = overlay_record(layout.data_root, used)

        files = []
        for current, dirs, names in os.walk(staging):
            dirs.sort()
            relative_dir = Path(current).relative_to(staging)
            if relative_dir.parts and relative_dir.parts[0] in WRITABLE_DIRS:
                continue
            for name in sorted(names):
                path = Path(current) / name
                relative = (relative_dir / name).as_posix()
                files.append({"path": relative, "sha256": sha256_file(path), "size_bytes": path.stat().st_size,
                              "role": _file_role(relative)})
        files.sort(key=lambda item: item["path"])
        manifest = {
            "format": VIEW_FORMAT,
            "schema_version": VIEW_SCHEMA_VERSION,
            "campaign_id": CAMPAIGN_ID,
            "kind": KIND_SMOKE_VIEW if smoke else KIND_FOLD_VIEW,
            "fold": fold,
            "bundle_id": bundle_id or (f"ecc30-smoke-fold{fold}-view" if smoke else f"ecc30-fold{fold}-view"),
            "source": {
                **layout.to_dict(),
                "source_root_basis": SOURCE_ROOT_BASIS,
                "training_manifest": {"path": sources["manifest_path"].relative_to(layout.data_root).as_posix(),
                                      "sha256": sources["manifest_sha256"], "n_rows": sources["manifest_n_rows"]},
                "training_csv": {"path": sources["csv_path"].relative_to(layout.data_root).as_posix(),
                                 "sha256": sources["csv_sha256"], "n_rows": sources["csv_n_rows"],
                                 "n_accessions": len(sources["rows_by_accession"]),
                                 "accessions_sha256": sources["accessions_sha256"]},
            },
            "smoke_subset": ({"max_proteins": int(max_proteins), "seed": int(seed), "rule": SMOKE_SUBSET_RULE}
                             if smoke else None),
            "structures": {"n": len(selected), "ids_sha256": ids_sha256(selected), "allowlist": ALLOWLIST_NAME,
                           "allowlist_sha256": allowlist_sha},
            "eligible": {"n": len(eligible), "ids_sha256": ids_sha256(eligible),
                         "n_excluded_ambiguous_ec_class": len(excluded), "excluded_structure_ids": excluded},
            "site_summary": {"file": SUMMARY_NAME, "sha256": summary_sha, "n_rows": summary_rows},
            "probe": {"file": PROBE_IDS_NAME, "sha256": probe_sha, "n": len(probe_ids), "rule": PROBE_RULE},
            "membership": membership_record,
            "overlay": overlay,
            "external_features_policy": external_features_policy(overlay),
            "read_guard": rules,
            "layout": {"structures": STRUCTURES_DIR, "esm": ESM_DIR, "external": EXTERNAL_DIR, "ring": RING_DIR,
                       "chain_means": dict(CHAIN_MEAN_DIRS), "caches": CACHE_DIR, "cache_subdirs": list(CACHE_SUBDIRS),
                       "receipts": RECEIPTS_DIR},
            "writable_dirs": list(WRITABLE_DIRS),
            "files": files,
            "provenance": dict(provenance) if provenance is not None else provenance_record(smoke=smoke),
            "created_utc": utc_now(),
        }
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
        _write_new(staging / VIEW_MANIFEST_NAME, manifest_bytes)
        if out_dir.exists():
            out_dir.rmdir()
        os.replace(staging, out_dir)
    except BaseException:
        _remove_tree(staging)
        raise
    manifest_sha256 = sha256_bytes(manifest_bytes)
    # The manifest is final here; modes are applied afterwards and never stop the build.
    modes = _set_permissions(out_dir)
    write_receipt_file(out_dir / RECEIPTS_DIR / FILE_MODES_RECEIPT_NAME,
                       {"stage": "build", "view_manifest_sha256": manifest_sha256, **modes})
    return {"view": str(out_dir), "manifest": manifest, "manifest_sha256": manifest_sha256, "file_modes": modes}


def check_inner_certificate(certificate: Mapping[str, Any], *, fold: int, membership_name: str,
                            membership_sha256: str) -> list[str]:
    """Problems of a ``certify-inner`` receipt for one fold's inner membership (empty when it certifies it)."""
    problems = []
    if certificate.get("step") != INNER_CERTIFICATE_STEP or certificate.get("certified") is not True:
        problems.append("the inner certificate is not a passed certify-inner receipt")
    if certificate.get("fold") != int(fold):
        problems.append(f"the inner certificate is for fold {certificate.get('fold')!r}, not fold {fold}")
    if certificate.get("n_inner_folds") != N_INNER_FOLDS:
        problems.append(f"the inner certificate has {certificate.get('n_inner_folds')!r} inner folds, "
                        f"not {N_INNER_FOLDS}")
    record = certificate.get("membership") or {}
    if record.get("sha256") != membership_sha256:
        problems.append("the inner certificate certifies another membership file (SHA-256 differs)")
    if record.get("file") != membership_name:
        problems.append(f"the inner certificate names {record.get('file')!r}, not {membership_name!r}")
    return problems


def read_inner_certificate(inner_membership: tuple[Path, str], *, fold: int,
                           certificate_path: Optional[str | os.PathLike] = None) -> dict[str, Any]:
    """Read and check the certificate of an inner membership; returns its path and SHA-256."""
    membership_path, membership_sha = Path(inner_membership[0]), str(inner_membership[1]).strip().lower()
    path = Path(certificate_path) if certificate_path is not None else membership_path.parent / INNER_CERTIFICATE_NAME
    if not path.is_file() or os.path.islink(path):
        raise ViewError(f"The inner membership needs its certify-inner receipt (a regular file): {path}")
    if is_test_named_path(path):
        raise HeldOutInputError(f"Inner certificate {path} is test-named; refused.")
    data = path.read_bytes()
    try:
        certificate = json.loads(data.decode("utf-8"))
    except ValueError as exc:
        raise ViewError(f"{path} is not a JSON receipt: {exc}") from exc
    problems = check_inner_certificate(certificate, fold=fold, membership_name=membership_path.name,
                                       membership_sha256=membership_sha)
    if problems:
        raise ViewError(f"{path.name} does not certify {membership_path.name} for fold {fold}: " + "; ".join(problems))
    return {"path": path, "sha256": sha256_bytes(data), "certificate": certificate}


def _remove_tree(path: Path) -> None:
    if not path.exists():
        return
    for current, dirs, _files in os.walk(path):
        os.chmod(current, 0o755)
    shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------------------
# Loading and verification
# ---------------------------------------------------------------------------

def _read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@dataclass(frozen=True)
class View:
    root: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str

    @classmethod
    def load(cls, root: str | os.PathLike) -> "View":
        root = Path(os.path.abspath(os.fspath(root)))
        path = root / VIEW_MANIFEST_NAME
        if not path.is_file():
            raise ViewError(f"{root} holds no {VIEW_MANIFEST_NAME}; it is not a campaign view.")
        data = path.read_bytes()
        manifest = json.loads(data.decode("utf-8"))
        if manifest.get("format") != VIEW_FORMAT:
            raise ViewError(f"{path} is not a {VIEW_FORMAT} manifest.")
        if manifest.get("schema_version") in RETIRED_VIEW_SCHEMA_VERSIONS:
            raise ViewError(f"{path} is a schema-{manifest.get('schema_version')} view (absolute guard roots, no "
                            "location rule); it is retired as a view and no longer loads. Build a new view.")
        if manifest.get("schema_version") != VIEW_SCHEMA_VERSION:
            raise ViewError(f"{path} has view schema {manifest.get('schema_version')!r}; this code reads "
                            f"schema {VIEW_SCHEMA_VERSION}.")
        view = cls(root, manifest, sha256_bytes(data))
        # The location rule and the read guard are checked on every load, on the machine that loads it.
        view.forbidden_read_roots
        return view

    @property
    def kind(self) -> str:
        return self.manifest["kind"]

    @property
    def smoke(self) -> bool:
        return self.kind == KIND_SMOKE_VIEW

    @property
    def fold(self) -> int:
        return int(self.manifest["fold"])

    @property
    def bundle_id(self) -> str:
        return self.manifest["bundle_id"]

    def path(self, name: str) -> Path:
        return self.root / name

    @property
    def structures(self) -> Path:
        return self.root / STRUCTURES_DIR

    @property
    def summary(self) -> Path:
        return self.root / SUMMARY_NAME

    @property
    def esm(self) -> Path:
        return self.root / ESM_DIR

    @property
    def external(self) -> Path:
        return self.root / EXTERNAL_DIR

    @property
    def ring(self) -> Path:
        return self.root / RING_DIR

    @property
    def caches(self) -> Path:
        return self.root / CACHE_DIR

    @property
    def receipts(self) -> Path:
        return self.root / RECEIPTS_DIR

    def chain_means_dir(self, layer: str) -> Path:
        if layer not in CHAIN_MEAN_DIRS:
            raise ViewError(f"No chain-mean folder for layer {layer!r}; expected one of {list(CHAIN_MEAN_DIRS)}.")
        return self.root / CHAIN_MEAN_DIRS[layer]

    def chain_means_manifest_sha256(self, layer: str) -> Optional[str]:
        path = self.chain_means_dir(layer) / "manifest.json"
        return sha256_file(path) if path.is_file() else None

    def structure_ids(self) -> list[str]:
        return _read_lines(self.root / ALLOWLIST_NAME)

    def eligible_ids(self) -> list[str]:
        excluded = set(self.manifest["eligible"]["excluded_structure_ids"])
        return [stem for stem in self.structure_ids() if stem not in excluded]

    def probe_ids(self) -> list[str]:
        return _read_lines(self.root / PROBE_IDS_NAME)

    @property
    def probe_file(self) -> Path:
        return self.root / PROBE_IDS_NAME

    @property
    def probe_sha256(self) -> str:
        return self.manifest["probe"]["sha256"]

    @property
    def membership(self) -> Optional[Mapping[str, Any]]:
        return self.manifest.get("membership")

    def membership_file(self) -> Optional[Path]:
        record = self.membership
        return None if record is None else self.root / record["file"]

    @property
    def allowlist_sha256(self) -> str:
        return self.manifest["structures"]["allowlist_sha256"]

    @property
    def campaign_root(self) -> Path:
        return campaign_root_of(self.root, smoke=self.smoke, fold=self.fold)

    @property
    def read_guard(self) -> Mapping[str, Any]:
        return self.manifest["read_guard"]

    @property
    def source_layout(self) -> SourceLayout:
        """The source layout recorded at build time (paths of the build machine; no file is opened)."""
        source = self.manifest["source"]
        return SourceLayout.from_data_root(source["data_root"], esm_dir=source["esm_dir"],
                                           external_root=source["external_root"])

    @cached_property
    def _expanded_read_guard(self) -> tuple[str, ...]:
        return tuple(expand_read_guard(self.read_guard, view_root=self.root, smoke=self.smoke, fold=self.fold,
                                       layout=self.source_layout))

    @property
    def forbidden_read_roots(self) -> list[str]:
        """The view's read-guard rules expanded on this machine (see ``expand_read_guard``)."""
        return list(self._expanded_read_guard)

    @property
    def overlay(self) -> Optional[Mapping[str, Any]]:
        return self.manifest.get("overlay")

    @property
    def default_runs_dir(self) -> Path:
        """``<campaign root>/runs/fold<k>`` for a fold view, ``<campaign root>/smoke_runs/<name>`` for a smoke view."""
        if self.smoke:
            return self.campaign_root / SMOKE_RUNS_DIR / self.root.name
        return self.campaign_root / RUNS_DIR / f"fold{self.fold}"


def _walk_view(view: View, problems: list[str]) -> dict[str, Path]:
    """Every regular file of the view outside the writable folders; symlinks anywhere are problems."""
    found: dict[str, Path] = {}
    for current, dirs, names in os.walk(view.root, followlinks=False):
        current_path = Path(current)
        relative_dir = current_path.relative_to(view.root)
        scratch = bool(relative_dir.parts) and relative_dir.parts[0] in (CACHE_DIR, RECEIPTS_DIR)
        for name in list(dirs) + list(names):
            path = current_path / name
            relative = (relative_dir / name).as_posix()
            if os.path.islink(path):
                problems.append(f"symlink in view: {relative}")
            if names_test_split(name) and not scratch:
                problems.append(f"test-named entry in view: {relative}")
        if relative_dir.parts and relative_dir.parts[0] in WRITABLE_DIRS:
            continue
        for name in names:
            path = current_path / name
            relative = (relative_dir / name).as_posix()
            if relative == VIEW_MANIFEST_NAME or os.path.islink(path):
                continue
            found[relative] = path
    return found


def verify_view(
    view_root: str | os.PathLike,
    *,
    require_chain_means: Sequence[str] = (),
    check_sources: bool = False,
    write_receipt: bool = True,
    provenance: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Rehash and cross-check a view; raise ``ViewVerificationError`` listing every failed check.

    File modes are tested in ``receipts/`` first (``probe_file_modes``): only where chmod
    works are writable data files a problem; content hashes are always checked. The receipt
    records ``file_modes_enforced``.
    """
    from training.ec_membership import EcMembershipError, read_ec_fold_membership
    from training.global_esm_features import GlobalEsmCacheError, load_global_esm_vectors, read_manifest
    from training.runtime_preparation import discover_missing_esm_embeddings, discover_missing_updated_external_features
    from training.site_filter import validate_selected_site_summary
    from training.structure_loading import find_structure_files

    view = View.load(view_root)
    manifest = view.manifest
    problems: list[str] = []
    file_modes = probe_file_modes(view.receipts)
    modes_enforced = file_modes["enforced"]
    if modes_enforced is None:
        problems.append(f"file modes cannot be tested in {RECEIPTS_DIR}/ ({file_modes['detail']}); the folder must be "
                        "writable")
    found = _walk_view(view, problems)
    recorded = {item["path"]: item for item in manifest["files"]}
    for relative in sorted(set(found) - set(recorded)):
        problems.append(f"extra file not in the view manifest: {relative}")
    for relative in sorted(set(recorded) - set(found)):
        problems.append(f"file listed in the view manifest is missing: {relative}")
    writable_files = []
    for relative in sorted(set(found) & set(recorded)):
        path = found[relative]
        if path.stat().st_size != recorded[relative]["size_bytes"] or sha256_file(path) != recorded[relative]["sha256"]:
            problems.append(f"content differs from the view manifest: {relative}")
        if modes_enforced and path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            writable_files.append(relative)
    recorded_modes = recorded_file_modes_outcome(view)
    if writable_files and recorded_modes != FILE_MODES_ENFORCED:
        problems.append(f"{len(writable_files)} data files are writable, e.g. {writable_files[:3]}: file modes were not "
                        f"applied since the build (recorded outcome {recorded_modes!r}, chmod works now); run "
                        "apply-view-modes")
    else:
        problems.extend(f"data file is writable: {relative}" for relative in writable_files)
    problems.extend(read_guard_rule_problems(view.read_guard, fold=view.fold, smoke=view.smoke,
                                             layout=view.source_layout))

    stems = view.structure_ids()
    if ids_sha256(stems) != manifest["structures"]["ids_sha256"] or len(stems) != len(set(stems)):
        problems.append("allowlist IDs differ from the view manifest")
    if sha256_file(view.root / ALLOWLIST_NAME) != view.allowlist_sha256:
        problems.append("allowlist file hash differs from the view manifest")
    try:
        _header, structure_rows, _data = _read_csv(view.structures / STRUCTURE_MANIFEST_NAME)
        manifest_stems = sorted(Path(row["structure_name"]).stem for row in structure_rows)
        if any(row["relative_path"] != row["structure_name"] for row in structure_rows):
            problems.append("structure manifest rows must point at files directly inside structures/")
    except (OSError, KeyError) as exc:
        problems.append(f"structure manifest unreadable: {exc}")
        manifest_stems = []
    if manifest_stems != sorted(stems):
        problems.append("structure manifest IDs differ from the allowlist")
    esm_names = sorted(relative.split("/", 1)[1] for relative in recorded if relative.startswith(ESM_DIR + "/"))
    expected_esm = sorted([esm_file_name(stem) for stem in stems] + [esm_sidecar_name(stem) for stem in stems])
    if esm_names != expected_esm:
        problems.append("per-residue ESMC files differ from the allowlist (exact names)")
    external_stems = sorted(relative.split("/")[1] for relative in recorded if relative.startswith(EXTERNAL_DIR + "/"))
    if external_stems != sorted(stems):
        problems.append("external feature folders differ from the allowlist")
    try:
        _header, summary_rows, _data = _read_csv(view.summary)
        summary_accessions = sorted({(row.get("structure") or "").strip() for row in summary_rows})
    except OSError as exc:
        problems.append(f"site summary unreadable: {exc}")
        summary_accessions = []
    if summary_accessions != sorted(accession_from_structure_id(stem) for stem in stems):
        problems.append("site summary accessions differ from the allowlist")
    if view.kind == KIND_FOLD_VIEW:
        if ids_sha256(accession_from_structure_id(stem) for stem in stems) != \
                manifest["source"]["training_csv"]["accessions_sha256"]:
            problems.append("view accessions differ from the frozen source training accessions")
    eligible = view.eligible_ids()
    if ids_sha256(eligible) != manifest["eligible"]["ids_sha256"]:
        problems.append("eligible IDs differ from the view manifest")
    for stem in eligible:
        if classify_ec_class(ec_numbers_from_structure_id(stem)).status != STATUS_OK:
            problems.append(f"{stem} is listed as eligible but has no single EC class")

    if check_sources:
        layout = view.source_layout
        for key, path in (("training_manifest", layout.training_manifest(view.fold)),
                          ("training_csv", layout.training_csv(view.fold))):
            if not path.is_file() or sha256_file(path) != manifest["source"][key]["sha256"]:
                problems.append(f"source {key} differs from the hash frozen in the view manifest")
        if view.overlay is not None:
            overlay_path = layout.data_root / str(view.overlay.get("manifest_file") or OVERLAY_MANIFEST_NAME)
            if os.path.islink(overlay_path) or not overlay_path.is_file() \
                    or sha256_file(overlay_path) != view.overlay.get("manifest_sha256"):
                problems.append("the source overlay manifest differs from the hash recorded in the view manifest")

    structure_files: list[Path] = []
    try:
        structure_files = list(find_structure_files(view.structures))
    except (OSError, ValueError) as exc:
        problems.append(f"structures do not resolve through the view's manifest: {exc}")
    for path in structure_files:
        if not _under(_real(path), [_real(view.structures)]):
            problems.append(f"structure resolves outside the view: {path.name}")
    if structure_files:
        missing_esm = discover_missing_esm_embeddings(structure_files, view.esm)
        missing_external = discover_missing_updated_external_features(
            structure_files, structure_root=view.structures, external_features_root_dir=view.external)
        if missing_esm:
            problems.append(f"{len(missing_esm)} structures lack per-residue ESMC files in the view")
        if missing_external:
            problems.append(f"{len(missing_external)} structures lack external features in the view")
    try:
        summary_report = validate_selected_site_summary(view.summary)
    except (OSError, ValueError) as exc:
        problems.append(f"site summary fails selected-site validation: {exc}")
        summary_report = None
    if any(Path(view.ring).iterdir()):
        problems.append("the RING folder must stay empty")

    probe = view.probe_ids()
    if sha256_file(view.probe_file) != view.probe_sha256 or not set(probe) <= set(eligible) or not probe:
        problems.append("probe list differs from the manifest or names non-eligible structures")
    membership_report = None
    if view.membership is not None:
        try:
            membership = read_ec_fold_membership(view.membership_file(), view.membership["sha256"])
            ids = {row["structure_id"] for row in membership.rows}
            folds = sorted({int(row["fold"]) for row in membership.rows})
            if ids != set(eligible):
                problems.append("membership structures differ from the eligible view structures")
            if folds != list(range(int(view.membership["n_folds"]))):
                problems.append(f"membership folds {folds} differ from 0..{view.membership['n_folds'] - 1}")
            membership_report = {"file": view.membership["file"], "sha256": membership.file_sha256,
                                 "n_rows": len(membership.rows), "folds": folds}
            if view.membership.get("role") == "inner":
                certificate_path = view.root / str(view.membership.get("certificate_file") or INNER_CERTIFICATE_NAME)
                if not certificate_path.is_file() or sha256_file(certificate_path) != view.membership.get(
                        "certificate_sha256"):
                    problems.append("the inner certificate is missing or differs from the view manifest")
                else:
                    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
                    problems.extend(check_inner_certificate(
                        certificate, fold=view.fold, membership_name=str(view.membership.get("source_name")),
                        membership_sha256=membership.file_sha256))
                    membership_report["certificate_sha256"] = view.membership.get("certificate_sha256")
        except (OSError, ValueError, EcMembershipError) as exc:
            problems.append(f"membership does not verify: {exc}")
    elif view.smoke:
        problems.append("a smoke view must hold its smoke membership")

    chain_means: dict[str, Any] = {}
    for layer in CHAIN_MEAN_DIRS:
        directory = view.chain_means_dir(layer)
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            if layer in require_chain_means:
                problems.append(f"chain means for layer {layer} are required but not built in {directory.name}/")
            elif any(directory.iterdir()):
                problems.append(f"{directory.name}/ holds files but no finished manifest")
            continue
        digest = sha256_file(manifest_path)
        try:
            cache_manifest = read_manifest(directory, digest)
            if cache_manifest["allowlist"].get("file_sha256") != view.allowlist_sha256:
                problems.append(f"chain means of layer {layer} were built from another allowlist")
            load_global_esm_vectors(directory, digest, eligible, expected_layer=layer)
            chain_means[layer] = {"dir": directory.name, "manifest_sha256": digest,
                                  "n_keys": cache_manifest.get("n_keys")}
        except (GlobalEsmCacheError, OSError, ValueError) as exc:
            problems.append(f"chain means of layer {layer} do not verify: {exc}")

    if problems:
        raise ViewVerificationError(problems)
    receipt = {
        "status": "verified",
        "view": str(view.root),
        "view_manifest_sha256": view.manifest_sha256,
        "kind": view.kind,
        "fold": view.fold,
        "n_files": len(recorded),
        "n_structures": len(stems),
        "n_eligible": len(eligible),
        "membership": membership_report,
        "chain_means": chain_means,
        "site_summary": summary_report,
        "checked_sources": bool(check_sources),
        "file_modes_enforced": bool(modes_enforced),
        "file_modes": {**file_modes, "recorded_outcome": recorded_modes},
        "overlay": None if view.overlay is None else dict(view.overlay),
        "provenance": dict(provenance) if provenance is not None else provenance_record(smoke=view.smoke),
    }
    if write_receipt:
        write_receipt_file(view.receipts / VERIFY_RECEIPT_NAME, receipt)
    return receipt


def recorded_file_modes_outcome(view: View) -> Optional[str]:
    """The last recorded file-mode outcome: ``file_modes_applied.json`` if present, else ``file_modes.json``."""
    for name in (FILE_MODES_APPLIED_RECEIPT_NAME, FILE_MODES_RECEIPT_NAME):
        path = view.receipts / name
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8")).get("outcome")
            except (OSError, ValueError):
                return None
    return None


def apply_view_modes(view_root: str | os.PathLike, *, provenance: Optional[Mapping[str, Any]] = None
                     ) -> dict[str, Any]:
    """Set a built view's file modes again (chmod only) and write ``receipts/file_modes_applied.json``.

    No byte and no manifest entry changes (modes are not part of the manifest's ``files``
    entries), so the manifest SHA-256 and every receipt bound to it stay valid. Used after
    the filesystem driver changes, e.g. when a view built where chmod was ignored now sits
    on a mount that enforces modes.
    """
    view = View.load(view_root)
    modes = _set_permissions(view.root)
    after = sha256_file(view.root / VIEW_MANIFEST_NAME)
    if after != view.manifest_sha256:
        raise ViewError(f"{view.root}: the view manifest changed while modes were applied; refused.")
    receipt = {"stage": "apply_view_modes", "view": str(view.root), "view_manifest_sha256": view.manifest_sha256,
               **modes,
               "provenance": dict(provenance) if provenance is not None else provenance_record(smoke=view.smoke)}
    receipt["receipt_sha256"] = write_receipt_file(view.receipts / FILE_MODES_APPLIED_RECEIPT_NAME, receipt)
    return receipt


def read_dry_run_receipt(view: View) -> Optional[dict[str, Any]]:
    """The view's dry-run receipt with its file SHA-256, or ``None`` when there is none."""
    path = view.receipts / DRY_RUN_RECEIPT_NAME
    if not path.is_file():
        return None
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["_file_sha256"] = sha256_file(path)
    return receipt


def dry_run_problems(view: View, receipt: Optional[Mapping[str, Any]], *,
                     chain_means: Optional[Mapping[str, Any]] = None) -> list[str]:
    """Why a view's dry-run receipt does not clear it (empty when it passed on this exact view).

    ``chain_means`` (layer -> record with ``manifest_sha256``) must equal the chain means the
    dry run verified, so a cache rebuilt after the dry run needs a new dry run. The dry run of
    a fold view must have run its probes under a cap prefix.
    """
    if receipt is None:
        return [f"the view has no {DRY_RUN_RECEIPT_NAME}; run dry-run first"]
    problems = []
    if receipt.get("status") != "passed":
        problems.append(f"the dry run of this view did not pass (status {receipt.get('status')!r})")
    if receipt.get("view_manifest_sha256") != view.manifest_sha256:
        problems.append("the dry-run receipt names another view manifest")
    if receipt.get("verified_view") is not True:
        problems.append("the dry run did not verify the view")
    if not view.smoke and not receipt.get("cap_prefix"):
        problems.append("the dry run of this fold view ran its probes without a cap prefix")
    if chain_means is not None:
        verified = ((receipt.get("verification") or {}).get("chain_means")) or {}
        for layer, record in chain_means.items():
            if (verified.get(layer) or {}).get("manifest_sha256") != record.get("manifest_sha256"):
                problems.append(f"the {layer} chain means differ from the ones the dry run verified")
    return problems


def freeze1_view_record(view_root: str | os.PathLike, *, require_dry_run: bool,
                        provenance: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """What the Freeze-1 manifest pins of one view: verified now, with chain means, membership and dry run.

    The view is verified (final-layer chain means required) before it is recorded; the
    chain-mean folders stay writable, so their manifest SHA-256 values are frozen here.
    ``require_dry_run`` (real manifests) also needs a passed dry run of this exact view and
    these chain means.
    """
    view = View.load(view_root)
    try:
        verification = verify_view(view.root, require_chain_means=("final",), write_receipt=False,
                                   provenance=provenance)
    except ViewVerificationError as exc:
        raise ViewError(f"{view.root} does not verify; refused in a Freeze-1 manifest:\n  "
                        + "\n  ".join(exc.problems[:20])) from exc
    dry = read_dry_run_receipt(view)
    if require_dry_run:
        problems = dry_run_problems(view, dry, chain_means=verification["chain_means"])
        if problems:
            raise ViewError(f"{view.root}: " + "; ".join(problems))
    membership = view.membership
    return {
        "fold": view.fold,
        "kind": view.kind,
        "bundle_id": view.bundle_id,
        "view_manifest_sha256": view.manifest_sha256,
        "allowlist_sha256": view.allowlist_sha256,
        "eligible_ids_sha256": view.manifest["eligible"]["ids_sha256"],
        "probe_sha256": view.probe_sha256,
        "membership": None if membership is None else {
            key: membership.get(key) for key in ("role", "file", "sha256", "identity_sha256", "n_folds",
                                                 "certificate_sha256")},
        "chain_means": {layer: record["manifest_sha256"]
                        for layer, record in sorted(verification["chain_means"].items())},
        "file_modes_enforced": verification["file_modes_enforced"],
        "overlay": None if view.overlay is None else {key: view.overlay.get(key) for key in ("id", "manifest_sha256")},
        "dry_run_receipt": None if dry is None else {
            "status": dry.get("status"), "sha256": dry["_file_sha256"],
            "view_manifest_sha256": dry.get("view_manifest_sha256")},
    }


def write_receipt_file(path: Path, receipt: Mapping[str, Any]) -> str:
    """Write a receipt; an existing one is superseded and its hash recorded in the new one."""
    path = Path(path)
    payload = dict(receipt)
    if path.is_file():
        payload["supersedes_sha256"] = sha256_file(path)
    data = (json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_bytes(data)
    os.replace(temporary, path)
    return sha256_bytes(data)


# ---------------------------------------------------------------------------
# Dry-run probe (child side)
# ---------------------------------------------------------------------------

class _Recorder:
    def __init__(self) -> None:
        self.reads: dict[str, int] = {}
        self.writes: dict[str, int] = {}
        self.listings: dict[str, int] = {}
        self.existing_dirs: dict[str, int] = {}
        self.subprocesses: list[str] = []
        self.active = True

    def hook(self, event: str, args: tuple) -> None:
        if not self.active:
            return
        try:
            if event == "open" and args:
                target = args[0]
                if isinstance(target, int) or target is None:
                    return
                mode = args[1] if len(args) > 1 and isinstance(args[1], str) else ""
                flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
                write = any(token in mode for token in "wax+") or bool(
                    flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
                self._add(self.writes if write else self.reads, target)
            elif event in _LISTING_EVENTS and args:
                self._add(self.listings, args[0] if args[0] is not None else ".")
            elif event == "os.mkdir" and args and isinstance(args[0], (str, bytes, os.PathLike)) \
                    and os.path.isdir(args[0]):
                self._add(self.existing_dirs, args[0])  # mkdir(exist_ok=True) of a folder that exists
            elif event in _WRITE_EVENTS and args:
                self._add(self.writes, args[0])
                if event in {"os.rename", "os.replace", "shutil.copyfile", "shutil.move", "os.link"} and len(args) > 1:
                    self._add(self.writes, args[1])
            elif event == "subprocess.Popen" and len(args) > 1:
                self._add_process(event, args[0], args[1])
            elif event == "os.system" and args:
                self._add_process(event, "/bin/sh", ["/bin/sh", "-c", args[0]])
            elif event in ("os.posix_spawn", "os.exec") and len(args) > 1:
                self._add_process(event, args[0], args[1])
            elif event == "os.spawn" and len(args) > 2:
                self._add_process(event, args[1], args[2])
        except Exception:  # a recorder must never break the probed process
            pass

    def _add_process(self, event: str, executable: Any, args: Any) -> None:
        if isinstance(args, (str, bytes, os.PathLike)):
            text = os.fsdecode(args) if isinstance(args, bytes) else os.fspath(args)
            argv = shlex.split(text) if text.strip() else []
        else:
            argv = [os.fsdecode(item) if isinstance(item, bytes) else str(item) for item in (args or [])]
        program = executable if executable is not None else (argv[0] if argv else "")
        program = os.fsdecode(program) if isinstance(program, bytes) else str(program)
        self.subprocesses.append({"event": event, "program": os.path.basename(program),
                                  "command": " ".join(argv)[:500]})

    @staticmethod
    def _add(table: dict[str, int], target: Any) -> None:
        if isinstance(target, bytes):
            target = os.fsdecode(target)
        if not isinstance(target, (str, os.PathLike)):
            return
        text = os.fspath(target)
        if not os.path.isabs(text):
            text = os.path.join(os.getcwd(), text)
        table[text] = table.get(text, 0) + 1

    def snapshot(self) -> dict[str, Any]:
        self.active = False
        return {"reads": dict(self.reads), "writes": dict(self.writes), "listings": dict(self.listings),
                "existing_dirs": dict(self.existing_dirs), "subprocesses": list(self.subprocesses)}


def _python_roots() -> list[str]:
    import site

    roots = {sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix}
    roots.update(getattr(site, "getsitepackages", lambda: [])())
    if site.ENABLE_USER_SITE:
        user = site.getusersitepackages()
        if user and Path(user).is_dir():
            roots.add(user)
    return sorted(roots)


class GenerationAttempted(RuntimeError):
    """The probed preparation path tried to generate a feature file."""


def _probe_child(request_path: Path) -> int:
    # The recorder goes first: audit hooks run in the order they were added, and the guard raises,
    # so a recorder added after it would never see a denied read.
    recorder = _Recorder()
    sys.addaudithook(recorder.hook)
    from training.access_guard import install_guard_from_environment

    install_guard_from_environment()
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    report: dict[str, Any] = {"status": "started", "generation_attempts": [], "unit_id": request.get("unit_id")}
    exit_code = 2
    try:
        import training.runtime_preparation as runtime_preparation

        def sentinel(name: str):
            def raise_attempt(*_args, **_kwargs):
                report["generation_attempts"].append(name)
                raise GenerationAttempted(f"{name} was called; a campaign fit may not generate feature files")
            return raise_attempt

        for name in GENERATOR_FUNCTIONS:
            setattr(runtime_preparation, name, sentinel(name))
        from training.config import parse_args
        from training.run import prepare_run

        config = parse_args(list(request["argv"]))
        prepared = prepare_run(config)
        summary = prepared.dataset_summary
        runtime = summary.get("runtime_preparation") or {}
        report.update({
            "status": "prepared",
            "run_dir": str(prepared.run_dir),
            "train_structure_ids": sorted({pocket.structure_id for pocket in prepared.split.train_pockets}),
            "val_structure_ids": sorted({pocket.structure_id for pocket in prepared.split.val_pockets}),
            "generated": {key: runtime.get(key) for key in ("generated_esm_files",
                                                            "generated_updated_external_feature_files",
                                                            "generated_ring_edge_files")},
            "split_type": (summary.get("fold_membership") or {}).get("split_type"),
            "global_esm_attach": (summary.get("global_esm") or {}).get("attach"),
            "site_extraction_summary": (summary.get("feature_load_report") or {}).get("site_extraction_summary"),
        })
        exit_code = 0
    except BaseException as exc:  # recorded, then reported by the parent
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:4000]}
    finally:
        records = recorder.snapshot()
        report.update({"records": records, "python_roots": _python_roots(),
                       "sys_path": [entry for entry in sys.path if entry], "cwd": os.getcwd()})
        Path(request["report"]).write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
                                           encoding="utf-8")
    return exit_code


# ---------------------------------------------------------------------------
# Dry-run probe (parent side)
# ---------------------------------------------------------------------------

def classify_probe_records(report: Mapping[str, Any], *, view: View, probe_dir: Path) -> dict[str, Any]:
    """Split recorded paths into allowed and outside-of-view; returns the outside lists and counts.

    Allowed reads: the view, its caches and the probe folder; Python files under ``src/``;
    the Python installation (a root that is or contains the repository, the view or a data
    root does not count); system folders. Allowed listings: the same roots plus the exact
    entries of the probe's ``sys.path``. Nothing under a forbidden or data root is allowed
    unless it lies in the view. The repository root itself is never a code root, so its
    documents, reports and test-side folders are outside reads.
    """
    view_root = [_real(view.root)]
    write_roots = [_real(view.caches), _real(probe_dir)]
    allowed_writes = [*write_roots, *(_real(path) for path in SYSTEM_WRITE_PATHS)]
    src_root = [_real(SRC_ROOT)]
    system_roots = [_real(root) for root in SYSTEM_READ_ROOTS]
    data_like = [_real(root) for root in view.forbidden_read_roots] + [_real(view.manifest["source"]["data_root"])]
    protected = [_real(REPO_ROOT), *view_root, *data_like]
    python_roots = []
    for root in report.get("python_roots", []):
        real = _real(root)
        # A Python root that is or contains the repository, the view or a data root would allow all of them.
        if real != os.sep and not any(_under(path, [real]) for path in protected):
            python_roots.append(real)
    sys_path_dirs = sorted({_real(entry) for entry in report.get("sys_path", [])
                            if os.path.isdir(entry) and not _under(_real(entry), data_like)})
    own_roots = [*view_root, *write_roots]
    records = report.get("records", {})

    def read_allowed(real: str) -> bool:
        if _under(real, own_roots):
            return True
        if _under(real, data_like):
            return False
        if _under(real, src_root):
            return real.endswith(CODE_READ_SUFFIXES)
        return _under(real, [*python_roots, *system_roots])

    def listing_allowed(real: str) -> bool:
        if _under(real, own_roots):
            return True
        if _under(real, data_like):
            return False
        return real in sys_path_dirs or _under(real, [*src_root, *python_roots, *system_roots])

    def outside(table: Mapping[str, int], allowed) -> list[str]:
        # The resolved path decides: a symlink inside the view that leads outside is an outside read.
        return sorted({_real(path) for path in table if not allowed(_real(path))})

    outside_reads = outside(records.get("reads", {}), read_allowed)
    outside_listings = outside(records.get("listings", {}), listing_allowed)
    outside_writes = outside(records.get("writes", {}), lambda real: _under(real, allowed_writes))
    data_reads = sorted(path for path in records.get("reads", {}) if _under(_real(path), view_root))
    subprocesses = [entry if isinstance(entry, Mapping) else {"event": None, "program": None, "command": str(entry)}
                    for entry in records.get("subprocesses", [])]
    return {
        "n_reads": len(records.get("reads", {})),
        "n_listings": len(records.get("listings", {})),
        "n_writes": len(records.get("writes", {})),
        "existing_dirs_touched": sorted(records.get("existing_dirs", {})),
        "n_view_reads": len(data_reads),
        "outside_reads": outside_reads,
        "outside_listings": outside_listings,
        "outside_writes": outside_writes,
        "outside_subprocesses": [entry["command"] for entry in subprocesses
                                 if entry.get("program") not in ALLOWED_SUBPROCESS_PROGRAMS],
        "subprocesses": subprocesses,
    }


def run_probe(
    *,
    view: View,
    unit_id: str,
    argv: Sequence[str],
    env: Mapping[str, str],
    probe_root: Path,
    cap_prefix: Sequence[str],
    python: str = sys.executable,
    timeout: Optional[float] = None,
) -> dict[str, Any]:
    """Run the preparation probe of one unit in a fresh (capped) process; return its raw report.

    The probe gets its own empty parse and graph caches inside ``probe_root`` (still inside the
    view's caches), so it opens every input the fit would open on a cold cache.
    """
    probe_dir = Path(probe_root) / unit_id
    probe_dir.mkdir(parents=True, exist_ok=False)
    argv = list(argv)
    if "--runs-dir" in argv:
        argv[argv.index("--runs-dir") + 1] = str(probe_dir / "runs")
    else:
        argv += ["--runs-dir", str(probe_dir / "runs")]
    request = {"unit_id": unit_id, "argv": argv, "report": str(probe_dir / "probe_report.json")}
    request_path = probe_dir / "probe_request.json"
    request_path.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
    child_env = {key: value for key, value in os.environ.items() if key not in _CHILD_ENV_DROP}
    child_env.update(env)
    # Fresh parse and graph caches per probe, so every probe reads its inputs instead of a cache entry.
    child_env.update({"PYTHONPATH": str(SRC_ROOT), "PYTHONDONTWRITEBYTECODE": "1",
                      "DEEPMZYME_PARSE_CACHE_DIR": str(probe_dir / "parse_cache"),
                      "DEEPMZYME_GRAPH_CACHE_DIR": str(probe_dir / "graph_cache")})
    command = [*cap_prefix, python, "-m", "ec_clean30.views", "probe-child", "--request", str(request_path)]
    started = time.monotonic()
    with (probe_dir / "probe.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=str(probe_dir), env=child_env, stdout=log, stderr=subprocess.STDOUT,
                                   timeout=timeout)
    report_path = Path(request["report"])
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {
        "status": "no_report", "records": {}, "generation_attempts": []}
    report["returncode"] = completed.returncode
    report["wall_seconds"] = round(time.monotonic() - started, 3)
    report["probe_dir"] = str(probe_dir)
    report["command"] = command
    return report


_CHILD_ENV_DROP = frozenset({"PYTHONPATH", "DEEPMZYME_LOAD_WORKERS", "DEEPMZYME_RUNTIME_CONFIG", "RING_FEATURES_DIR",
                             "RING_EDGES_DIR", "PYTHONSTARTUP"})


def _probe_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Dry-run preparation probe (started by the campaign runner).")
    sub = parser.add_subparsers(dest="action", required=True)
    child = sub.add_parser("probe-child")
    child.add_argument("--request", type=Path, required=True)
    args = parser.parse_args(argv)
    return _probe_child(args.request)


if __name__ == "__main__":
    raise SystemExit(_probe_main())
