"""Data overlay of the EC class vs CLEAN30 campaign: builder and verifier (step P2-7; torch-free at import).

The overlay is one folder of byte copies (never links) of every training-side input the later
steps read, laid out like the source data root, so ``build-view --data-root <overlay>
--esm-dir <overlay>/esm --external-root <overlay>/external`` works unchanged and the inner tables
can be read after the identity audit makes the campaign's own identity folder test-bearing:

- ``CLEAN_30_train_test_split_<k>/train/structure_manifest.csv`` and
  ``CLEAN_30_main/folds/CLEAN_30_train_test_split_<k>_train.csv`` (the names ``SourceLayout`` reads);
- every structure the five manifests name, at the same path relative to the data root (so the
  copied manifests resolve inside the overlay), e.g. ``structure_store/objects/...``;
- ``esm/<stem>_chain_<chain>_esmc.pt`` with its ``.json`` sidecar and
  ``external/<stem>/residue_features.json`` of every protein (byte copies under decision A1
  option (b); the other options need the derived-feature builder, which is not built);
- ``identity/fold<k>/``: every file the identity loaders read for fold k's metallo training chain
  and inner steps (fasta receipt, node table and FASTAs; ``search_metallo_train/`` receipt and
  sorted output; edges receipt and tables; inner components, links and receipt; assignment
  receipt, inner membership and ``inner_certification_receipt.json``; for fold 0 the R1-inner
  exclusion lists and receipts), always at ``fold<k>`` even when ``--fold-work`` names another
  source folder;
- ``cohort/`` (the cohort audit's receipt and recorded outputs) and ``r2_lists/`` (the R2-list
  receipt and outputs);
- ``allowlist_fold<k>.txt`` (fold k's training structure IDs), ``eligible_fold<k>.txt`` (its
  certified inner membership) and ``label_free_index.tsv`` (``accession, chain,
  structure_sha256, esm_sha256, external_sha256``);
- ``overlay_manifest.json``: overlay ID, base dataset identity, sources, inputs, decisions (D2
  option, RING, excluded files), counts, the A13 string-scan count, rules, code SHA-256 values,
  the content-stable part of the provenance and one entry per file (path, SHA-256, size, role,
  origin). It holds no time stamp, so two builds from the same inputs and code are identical.

Next to the folder: ``<overlay id>.tar`` (Python ``tarfile``, POSIX format, sorted names under
``<overlay id>/``, mtime 0, uid and gid 0, no owner names, modes 0444 and 0555), its
``<overlay id>.tar.sha256`` in ``sha256sum`` format naming the tar by its bare file name, and
``<overlay id>.build_receipt.json`` (times, command line, full provenance, how the layout was
given, file-mode outcome; it binds the manifest and tar SHA-256 values). Mirror all three.

Rules: inputs by exact names only (no folder is listed and no test-side file is opened); the
cohort audit must have passed and the R2 lists must verify; every copied structure, ESMC payload,
sidecar and external file must hash to what the cohort audit recorded, and the training files to
what it read; the identity chain must load with the identity tool's own loaders, have read the
same five training site tables and manifests as the cohort audit (fasta receipt, assignment
receipt and inner certificate), name exactly the fold's training proteins, and match the chain
the R2 lists were built from; an identity folder that holds any complement (test-side) output is
refused; symlinks, test-named paths, an existing output and (for a real overlay) a smoke input or
a place other than ``<campaign root>/overlay/<overlay id>`` are refused. The folder is assembled
in a sibling staging folder, verified there, then renamed.
Messages name folds, roles, files without accessions and counts, never an accession or a
per-protein file name; a refusal of a reused reader that can name one is shown without its text,
and an operating-system error is shown as its errno name only (its text names the path).

``overlay verify`` rehashes every file, re-derives the expected file set from the overlay's own
copies (each protein once per role, the identity loaders on the copies, the R2 lists, the
derived lists), rescans the strings, and checks the tar: its SHA-256 against the ``.sha256``
file, its member list against the manifest, and a deterministic re-creation from the folder. The
build receipt must bind the manifest and the tar (required for a real overlay).

Runner wiring (``scripts/run_ec_clean30_campaign.py``)::

    from ec_clean30.overlay import OVERLAY_HELP, add_overlay_arguments
    add_overlay_arguments(sub.add_parser("overlay", help=OVERLAY_HELP))

``add_overlay_arguments`` adds the sub-actions ``build`` and ``verify`` with their handlers;
``OverlayError`` is a ``ViewError``, so the runner's existing error handling applies::

    overlay build --out DIR --cohort DIR --identity-root DIR --r2-lists DIR --d2-option {a,b,d}
                  [--fold-work K=PATH ...] [--data-root D [--esm-dir E] [--external-root X]] [--smoke]
    overlay verify --overlay DIR [--receipt FILE]

``build`` exits 0 when the overlay is written; ``verify`` exits 0 when every check passes and 1
(problems printed) otherwise. Both refuse a dirty tree unless the overlay is a smoke one.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import tarfile
from typing import Any, BinaryIO, Iterable, Mapping, Optional, Sequence

from ec_clean30 import cohort, identity, r2_lists
from ec_clean30.campaign import CAMPAIGN_ID, N_OUTER_FOLDS
from ec_clean30.provenance import (
    REPO_ROOT, HeldOutInputError, is_test_named_path, names_test_split, provenance_record, real_action_git_state,
    refuse_test_paths, sha256_bytes, utc_now, write_new_json,
)
from ec_clean30.views import (
    ESM_DIR, EXTERNAL_DIR, EXTERNAL_FEATURE_FILE, FORBIDDEN_DATA_ROOT_NAMES, OVERLAY_MANIFEST_NAME, SourceLayout,
    ViewError, _data_relative, _set_permissions, esm_file_name, esm_sidecar_name, read_training_sources,
)
from training.ec_classes import STATUS_AMBIGUOUS, STATUS_OK, accession_from_structure_id
from training.ec_membership import EcMembershipError


OVERLAY_FORMAT = "deepmzyme_ec_clean30_data_overlay_v1"
OVERLAY_SCHEMA_VERSION = 1
OVERLAY_KIND = "data_overlay"
OVERLAY_HELP = "Data overlay: byte copies of every training-side input of later steps, and its verifier (P2-7)."
DEFAULT_OVERLAY_ID = "ecc30-overlay-v1"
# A real overlay's folder name is its ID; it lives at <campaign root>/overlay/<overlay id>.
REAL_OVERLAY_ID_RE = re.compile(r"^ecc30-overlay-v[0-9]+$")
OVERLAY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
OVERLAYS_DIR = "overlay"

MANIFEST_NAME = OVERLAY_MANIFEST_NAME
TAR_SUFFIX = ".tar"
TAR_SHA256_SUFFIX = ".tar.sha256"
BUILD_RECEIPT_SUFFIX = ".build_receipt.json"
BUILD_RECEIPT_FORMAT = f"{OVERLAY_FORMAT}_build"
IDENTITY_DIR = "identity"
COHORT_DIR = "cohort"
R2_DIR = "r2_lists"
IDENTITY_FOLD_DIR = "fold{fold}"
ALLOWLIST_NAME = "allowlist_fold{fold}.txt"
ELIGIBLE_NAME = "eligible_fold{fold}.txt"
LABEL_FREE_INDEX_NAME = "label_free_index.tsv"
LABEL_FREE_INDEX_COLUMNS: tuple[str, ...] = ("accession", "chain", "structure_sha256", "esm_sha256", "external_sha256")
SEARCH_DIR = f"search_{identity.SET_METALLO_TRAIN}"

ROLE_TRAINING_MANIFEST = "training_manifest"
ROLE_TRAINING_SITE_TABLE = "training_site_table"
ROLE_STRUCTURE = "structure"
ROLE_ESM = "esm_residue"
ROLE_ESM_SIDECAR = "esm_sidecar"
ROLE_EXTERNAL = "external_features"
ROLE_IDENTITY = "identity"
ROLE_COHORT = "cohort"
ROLE_R2 = "r2_lists"
ROLE_ALLOWLIST = "allowlist"
ROLE_ELIGIBLE = "eligible_list"
ROLE_LABEL_FREE_INDEX = "label_free_index"
ROLES: tuple[str, ...] = (ROLE_TRAINING_MANIFEST, ROLE_TRAINING_SITE_TABLE, ROLE_STRUCTURE, ROLE_ESM, ROLE_ESM_SIDECAR,
                          ROLE_EXTERNAL, ROLE_IDENTITY, ROLE_COHORT, ROLE_R2, ROLE_ALLOWLIST, ROLE_ELIGIBLE,
                          ROLE_LABEL_FREE_INDEX)
# Roles whose file names carry a structure stem (accession and EC numbers): never printed.
PER_PROTEIN_ROLES = frozenset({ROLE_STRUCTURE, ROLE_ESM, ROLE_ESM_SIDECAR, ROLE_EXTERNAL})
SCAN_ROLES: tuple[str, ...] = (ROLE_ESM, ROLE_ESM_SIDECAR, ROLE_EXTERNAL)

# Decision A1 (D2): (b) copies external features unchanged; (d) and (a) need the derived-feature builder.
D2_OPTIONS: tuple[str, ...] = ("a", "b", "d")
D2_OPTIONS_BUILT: tuple[str, ...] = ("b",)
D2_POLICY = {
    "b": "external features copied unchanged; they were computed on the full structure, including metals outside the "
         "selected sites (decision A1 option b; disclosed)",
}
TAR_FILE_MODE = 0o444
TAR_DIR_MODE = 0o555
TAR_RULE = ("Python tarfile, POSIX (pax) format, members sorted by name under '<overlay id>/', every parent folder "
            "listed, mtime 0, uid and gid 0, empty owner names, files 0444 and folders 0555; the .sha256 file names "
            "the tar by its bare file name (sha256sum format)")

# The released dataset the shared roots were built from (docs/DATASETS.md, "Main Colab bundle v12").
BASE_IDENTITY: Mapping[str, str] = {
    "release": "Main Colab bundle v12",
    "archive": "DeepMzyme_Data_v12_manifest_exact_common70_nonoverlap_clean30_care30_complete_esm_ring_external.tar.gz",
    "archive_sha256": "90c0899829e0ac5ca94a5ef34484b74ca1014d3e9b6b1fba90bd338ee3440dee",
    "huggingface_commit": "0f9d1efb835bdd5597b2762b8608947d7b82cdf2",
    "record": "docs/DATASETS.md, section 'Main Colab bundle v12'",
    "note": "copied from the local tree v12 was built from; byte equality with v12 members not checked",
}
OVERLAY_LAYOUT: Mapping[str, str] = {
    "training_manifest": "CLEAN_30_train_test_split_{fold}/train/structure_manifest.csv",
    "training_site_table": "CLEAN_30_main/folds/CLEAN_30_train_test_split_{fold}_train.csv",
    "structures": "the path each training manifest's relative_path gives, relative to the data root",
    "esm": f"{ESM_DIR}/<stem>_chain_<chain>_esmc.pt and .pt.json",
    "external": f"{EXTERNAL_DIR}/<stem>/{EXTERNAL_FEATURE_FILE}",
    "identity": f"{IDENTITY_DIR}/fold{{fold}}",
    "cohort": COHORT_DIR,
    "r2_lists": R2_DIR,
    "build_view": "build-view --data-root <overlay> --esm-dir <overlay>/esm --external-root <overlay>/external",
}
RING_NOTE = "RING not used (decision A9; the runner refuses RING inputs); no RING file is copied"
EXCLUSIONS_STATEMENT: Mapping[str, Any] = {
    "statement": ("built from training files only, by exact names; no test folder, no *_test.csv, no "
                  "CLEAN_30_main/split_metadata.json (decision A10), no mixed manifest (CLEAN_30_main/structures), no "
                  "metadata folder and no complement (test-side) identity output was opened, listed or copied"),
    "complement_outputs": "refused: an identity folder holding any of them stops the build",
    "label_table": "kept outside the overlay; Freeze 1 binds it",
    "huggingface_copy": "none (decision A11)",
}
UNION_NOTE = ("holds every cohort protein, so every fold's test proteins are present as other folds' training "
              "proteins; never given whole to a fit (fold views copy one fold's training side) and not shipped to the "
              "GPU machine (M8)")
RULES: Mapping[str, str] = {
    "copies": "byte copies of regular files read by exact names; symlinks and test-named paths refused",
    "hashes": "structures, ESMC payloads, sidecars and external files hash to the cohort audit's values, the training "
              "manifests and site tables to the files the cohort audit read, cohort and R2 outputs to their receipts",
    "identity": "fold k's chain loads with ec_clean30.identity's loaders (fasta, pinned search, edges, inner "
                "components, assignment, certified membership; fold 0 also its R1-inner lists), names exactly fold k's "
                "training proteins, its membership covers exactly the training structures with one EC class, and the "
                "R2 lists were built from it",
    "allowlist": "fold k's training structure IDs (the stems of its structure manifest), sorted, one per line",
    "eligible_list": "fold k's certified inner membership structure IDs, sorted, one per line",
    "label_free_index": "one row per cohort protein from cohort_proteins.tsv, sorted by accession; no EC column",
    "string_scan": "string values of every ESMC payload (all fields except embeddings and residue_ids), sidecar and "
                   "external feature file that name a test or mixed folder (ec_clean30.cohort."
                   "count_test_named_strings, as the cohort audit's A13 count); count only",
}
COMPLEMENT_FILE_NAMES: tuple[str, ...] = (
    f"fasta_receipt_{identity.SET_COMPLEMENT}.json", f"query_{identity.SET_COMPLEMENT}.fasta",
    f"nodes_{identity.SET_COMPLEMENT}.tsv", f"search_{identity.SET_COMPLEMENT}",
    f"edges_receipt_{identity.SET_COMPLEMENT}.json", f"edges_{identity.SET_COMPLEMENT}.tsv", "strata_receipt.json",
)
REQUIRED_COHORT_OUTPUTS: tuple[str, ...] = (
    cohort.PROTEINS_NAME, cohort.SITES_NAME, cohort.D2_METALS_NAME, cohort.D2_RESIDUES_NAME,
    *(cohort.FOLD_COUNTS_NAME.format(fold=fold) for fold in range(N_OUTER_FOLDS)),
)
PROTEIN_SHA_COLUMNS: tuple[str, ...] = ("structure_sha256", "esm_sha256", "esm_sidecar_sha256", "external_sha256")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CHUNK = 1 << 20
# Messages name a file only when its role's names never carry a structure stem and it sits at that role's place;
# other paths are counted per top-level folder, and only these folder names are printed.
_NAMED_ROLE_FOLDERS: Mapping[str, str] = {ROLE_IDENTITY: IDENTITY_DIR, ROLE_COHORT: COHORT_DIR, ROLE_R2: R2_DIR}
_ROOT_DERIVED_NAMES = frozenset({LABEL_FREE_INDEX_NAME,
                                 *(ALLOWLIST_NAME.format(fold=fold) for fold in range(N_OUTER_FOLDS)),
                                 *(ELIGIBLE_NAME.format(fold=fold) for fold in range(N_OUTER_FOLDS))})
_PRINTABLE_TOP_FOLDERS = frozenset({ESM_DIR, EXTERNAL_DIR, IDENTITY_DIR, COHORT_DIR, R2_DIR, "CLEAN_30_main",
                                    "structure_store",
                                    *(f"CLEAN_30_train_test_split_{fold}" for fold in range(N_OUTER_FOLDS))})


class OverlayError(ViewError):
    """The overlay cannot be built from its inputs, or its request is refused."""


class OverlayVerificationError(ValueError):
    """A built overlay does not verify; ``problems`` lists every failed check (no accession is named)."""

    def __init__(self, problems: Sequence[str]):
        self.problems = list(problems)
        super().__init__("Overlay verification failed:\n  " + "\n  ".join(self.problems[:40]))


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _abs(path: str | os.PathLike) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _real(path: str | os.PathLike) -> str:
    return os.path.normcase(os.path.realpath(os.fspath(path)))


def _under(path: str, roots: Iterable[str]) -> bool:
    return any(path == root or path.startswith(root.rstrip(os.sep) + os.sep) for root in roots)


def _errno_text(exc: OSError) -> str:
    """An operating-system error as its type and errno name only: its text names the path, which can carry a stem."""
    code = errno.errorcode.get(exc.errno, str(exc.errno)) if isinstance(exc.errno, int) else "no errno"
    return f"{type(exc).__name__} {code}"


def _file_problem(path: Path) -> Optional[str]:
    """Why ``path`` is not a regular file read by exact name (``lstat``; a symlink is never followed)."""
    try:
        info = os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        return "missing"
    except OSError as exc:
        return f"not readable ({_errno_text(exc)})"
    if stat.S_ISLNK(info.st_mode):
        return "a symlink (refused)"
    if not stat.S_ISREG(info.st_mode):
        return "not a regular file"
    return None


def _read_regular(path: Path, what: str) -> bytes:
    problem = _file_problem(path)
    if problem is not None:
        raise OverlayError(f"{what} {path} cannot be read ({problem}).")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise OverlayError(f"{what} {path} cannot be read ({_errno_text(exc)}).") from None


def _parse_json(data: bytes, what: str) -> Any:
    try:
        return json.loads(data.decode("utf-8"))
    except ValueError as exc:
        raise OverlayError(f"{what} is not valid JSON: {exc}") from None


def _hash_file(path: Path) -> tuple[str, int]:
    digest, size = hashlib.sha256(), 0
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _lines_bytes(lines: Iterable[str]) -> bytes:
    return "".join(f"{line}\n" for line in lines).encode("utf-8")


def _tsv_bytes(columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(columns), delimiter="\t", lineterminator="\n",
                            extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _withheld(exc: BaseException, what: str) -> OverlayError:
    return OverlayError(f"{what} was refused by its reader ({type(exc).__name__}); the reader's message is withheld "
                        "because it can name an accession of another fold's training file. Run that reader on the "
                        "file with its output sent to a private file to see the detail.")


def _plain_name(name: Any, *, what: str, allowed_dirs: frozenset[str] = frozenset()) -> str:
    """A receipt-named file inside its folder: plain name, or ``<allowed dir>/<name>``; never test-named."""
    if not isinstance(name, str) or not name:
        raise OverlayError(f"{what} names no file.")
    parts = name.split("/")
    if (name.startswith("/") or "\\" in name or any(part in ("", ".", "..") for part in parts) or len(parts) > 2
            or (len(parts) == 2 and parts[0] not in allowed_dirs)):
        raise OverlayError(f"{what} names {name!r}, not a plain file name of its folder.")
    if any(names_test_split(part) for part in parts):
        raise HeldOutInputError(f"{what} names a test-named file {name!r}; refused.")
    return name


def _nameable(path: str, role: Optional[str]) -> bool:
    """Whether ``path`` may be printed: a file of a role whose names never carry a stem, at that role's own place."""
    if role in _NAMED_ROLE_FOLDERS:
        return path.startswith(_NAMED_ROLE_FOLDERS[role] + "/")
    if role in (ROLE_TRAINING_MANIFEST, ROLE_TRAINING_SITE_TABLE):
        return path in {OVERLAY_LAYOUT[role].format(fold=fold) for fold in range(N_OUTER_FOLDS)}
    if role in (ROLE_ALLOWLIST, ROLE_ELIGIBLE, ROLE_LABEL_FREE_INDEX):
        return path in _ROOT_DERIVED_NAMES
    return False


def _folder_label(path: str) -> str:
    """Where an unnamed path lies: a fixed top-level folder, the overlay root, or another top-level folder."""
    top, _, rest = path.partition("/")
    if not rest:
        return "the overlay root"
    if top in _PRINTABLE_TOP_FOLDERS:
        return f"{top}/"
    return "other top-level folders"


def _describe(paths: Sequence[str], message: str, *role_maps: Mapping[str, str]) -> str:
    """``message`` with the paths, named only where no name can carry a structure stem.

    A path is named when every role map that knows it gives a role whose file names carry no
    stem, at that role's place; every other path is counted per fixed top-level folder (or the
    overlay root, or other top-level folders).
    """
    def nameable(path: str) -> bool:
        known = [roles[path] for roles in role_maps if path in roles]
        return bool(known) and all(_nameable(path, role) for role in known)

    shown = sorted(path for path in paths if nameable(path))
    hidden = Counter(_folder_label(path) for path in paths if not nameable(path))
    parts = []
    if shown:
        parts.append(", ".join(shown[:5]) + (f" and {len(shown) - 5} more" if len(shown) > 5 else ""))
    if hidden:
        parts.append(", ".join(f"{count} under {folder}" for folder, count in sorted(hidden.items())))
    return f"{len(paths)} {message}: " + "; ".join(parts)


def _overlay_layout(root: Path) -> SourceLayout:
    return SourceLayout(root, root / ESM_DIR, root / EXTERNAL_DIR)


def overlay_side_paths(overlay_dir: str | os.PathLike) -> dict[str, Path]:
    """The tar, its ``.sha256`` file and the build receipt next to an overlay folder."""
    root = _abs(overlay_dir)
    return {"tar": root.parent / f"{root.name}{TAR_SUFFIX}",
            "tar_sha256": root.parent / f"{root.name}{TAR_SHA256_SUFFIX}",
            "build_receipt": root.parent / f"{root.name}{BUILD_RECEIPT_SUFFIX}"}


def _stable_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """The content-stable part of a provenance block (no times, command line or host), kept in the manifest."""
    git = provenance.get("git") if isinstance(provenance.get("git"), Mapping) else {}
    environment = provenance.get("environment") if isinstance(provenance.get("environment"), Mapping) else {}
    return {
        "smoke": provenance.get("smoke"),
        "git": {key: git.get(key) for key in ("head", "branch", "clean", "status_sha256")},
        "code_sha256": provenance.get("code_sha256"),
        "source_tree_sha256": provenance.get("source_tree_sha256"),
        "environment": {"python": environment.get("python"), "packages": environment.get("packages")},
    }


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Cohort:
    root: Path
    receipt: Mapping[str, Any]
    receipt_sha256: str
    outputs: Mapping[str, str]
    proteins: Mapping[str, Mapping[str, str]]


@dataclass(frozen=True)
class _Protein:
    accession: str
    stem: str
    structure_name: str
    source: Path
    relative: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class _Item:
    """One overlay file: where it goes, its role and origin, and its source (or bytes) with the expected SHA-256."""
    path: str
    role: str
    origin: Mapping[str, Any]
    source: Optional[Path] = None
    data: Optional[bytes] = None
    expected_sha256: Optional[str] = None


@dataclass(frozen=True)
class _Plan:
    items: tuple[_Item, ...]
    cohort: _Cohort
    folds: Mapping[int, Mapping[str, Any]]
    union: Mapping[str, _Protein]
    identity: Mapping[int, Mapping[str, Any]]
    r2: Mapping[str, Any]


def _parse_proteins(data: bytes) -> dict[str, dict[str, str]]:
    reader = csv.DictReader(io.StringIO(data.decode("utf-8")), delimiter="\t")
    if tuple(reader.fieldnames or ()) != tuple(cohort.PROTEIN_COLUMNS):
        raise OverlayError(f"{cohort.PROTEINS_NAME} does not have the cohort audit's columns.")
    rows: dict[str, dict[str, str]] = {}
    for number, row in enumerate(reader, start=2):
        accession = row.get("accession") or ""
        if not identity.ACCESSION_RE.match(accession) or accession in rows:
            raise OverlayError(f"{cohort.PROTEINS_NAME} row {number} has an invalid or repeated accession.")
        if any(not SHA256_RE.match(row.get(column) or "") for column in PROTEIN_SHA_COLUMNS):
            raise OverlayError(f"{cohort.PROTEINS_NAME} row {number} lacks a structure, ESMC, sidecar or external "
                               "SHA-256; a passed audit records all four.")
        rows[accession] = dict(row)
    if not rows:
        raise OverlayError(f"{cohort.PROTEINS_NAME} holds no protein.")
    return rows


def _load_cohort(cohort_dir: str | os.PathLike, *, smoke: bool) -> _Cohort:
    """The cohort audit's receipt and outputs (each against its recorded SHA-256); a passed audit only."""
    root = _abs(cohort_dir)
    refuse_test_paths([root], what="--cohort")
    receipt_data = _read_regular(root / cohort.RECEIPT_NAME, "The cohort audit receipt")
    receipt = _parse_json(receipt_data, cohort.RECEIPT_NAME)
    if (not isinstance(receipt, Mapping) or receipt.get("format") != cohort.COHORT_AUDIT_FORMAT
            or receipt.get("kind") != "cohort_audit" or receipt.get("schema_version") != cohort.COHORT_SCHEMA_VERSION
            or receipt.get("campaign_id") != CAMPAIGN_ID):
        raise OverlayError(f"{root / cohort.RECEIPT_NAME} is not a cohort audit receipt of schema "
                           f"{cohort.COHORT_SCHEMA_VERSION}.")
    if receipt.get("status") != "passed" or receipt.get("failed_checks"):
        raise OverlayError(f"The cohort audit did not pass (status {receipt.get('status')!r}, failed checks "
                           f"{list(receipt.get('failed_checks') or [])}); the overlay waits for a passed audit.")
    if not smoke and receipt.get("smoke") is not False:
        raise OverlayError("The cohort audit is an engineering smoke; a real overlay needs a real (non-smoke) audit.")
    outputs = receipt.get("outputs")
    if not isinstance(outputs, Mapping) or not outputs:
        raise OverlayError(f"{cohort.RECEIPT_NAME} has no outputs record.")
    if cohort.QUARANTINE_NAME in outputs or os.path.lexists(root / cohort.QUARANTINE_NAME):
        raise OverlayError(f"The cohort audit quarantined EC annotation differences ({cohort.QUARANTINE_NAME}); the "
                           "overlay waits until they are resolved.")
    exclusions = [name for name in (cohort.FOLD_EXCLUSIONS_NAME.format(fold=fold) for fold in range(N_OUTER_FOLDS))
                  if name in outputs or os.path.lexists(root / name)]
    if exclusions:
        raise OverlayError(f"The cohort audit found load-time exclusions ({exclusions}); an overlay of such a cohort "
                           "needs the exclusion input of the inner steps and views first.")
    recorded: dict[str, str] = {}
    for name, entry in sorted(outputs.items()):
        _plain_name(name, what=f"{cohort.RECEIPT_NAME} outputs")
        digest = entry.get("sha256") if isinstance(entry, Mapping) else None
        if not isinstance(digest, str) or not SHA256_RE.match(digest):
            raise OverlayError(f"{cohort.RECEIPT_NAME} records no valid SHA-256 for {name}.")
        recorded[name] = digest
    missing = [name for name in REQUIRED_COHORT_OUTPUTS if name not in recorded]
    if missing:
        raise OverlayError(f"{cohort.RECEIPT_NAME} does not record the outputs {missing}.")
    changed, proteins_data = [], b""
    for name, digest in recorded.items():
        path = root / name
        if name == cohort.PROTEINS_NAME:
            proteins_data = _read_regular(path, "The cohort output")
            actual = sha256_bytes(proteins_data)
        else:
            problem = _file_problem(path)
            if problem is not None:
                raise OverlayError(f"The cohort output {path} cannot be read ({problem}).")
            try:
                actual = _hash_file(path)[0]
            except OSError as exc:
                raise OverlayError(f"The cohort output {path} cannot be read ({_errno_text(exc)}).") from None
        if actual != digest:
            changed.append(name)
    if changed:
        raise OverlayError(f"{changed} changed after the cohort audit (SHA-256 differs from its receipt).")
    folds = (receipt.get("inputs") or {}).get("folds") if isinstance(receipt.get("inputs"), Mapping) else None
    if not isinstance(folds, Mapping) or set(folds) != {str(fold) for fold in range(N_OUTER_FOLDS)}:
        raise OverlayError(f"{cohort.RECEIPT_NAME} does not record the five training sides it read.")
    return _Cohort(root=root, receipt=receipt, receipt_sha256=sha256_bytes(receipt_data), outputs=recorded,
                   proteins=_parse_proteins(proteins_data))


def layout_from_cohort(cohort_dir: str | os.PathLike) -> SourceLayout:
    """The source layout the cohort audit recorded (its data root, ESMC folder and external-feature root)."""
    receipt = _parse_json(_read_regular(_abs(cohort_dir) / cohort.RECEIPT_NAME, "The cohort audit receipt"),
                          cohort.RECEIPT_NAME)
    record = ((receipt.get("inputs") or {}).get("layout") if isinstance(receipt, Mapping) else None) or {}
    try:
        return SourceLayout.from_data_root(record["data_root"], esm_dir=record["esm_dir"],
                                           external_root=record["external_root"])
    except (KeyError, TypeError):
        raise OverlayError(f"{cohort.RECEIPT_NAME} records no source layout; pass --data-root.") from None


def _read_fold_sources(layout: SourceLayout, fold: int) -> dict[str, Any]:
    try:
        return read_training_sources(layout, fold)
    except HeldOutInputError:
        raise HeldOutInputError(f"Fold {fold}'s training manifest or site table names a test or mixed folder; refused "
                                "(the reader's message is withheld because it can name an accession).") from None
    except (ViewError, ValueError, OSError, KeyError) as exc:
        raise _withheld(exc, f"Fold {fold}'s training manifest or site table") from None


def _reserved_top_level() -> frozenset[str]:
    names = {ESM_DIR, EXTERNAL_DIR, IDENTITY_DIR, COHORT_DIR, R2_DIR, MANIFEST_NAME, LABEL_FREE_INDEX_NAME,
             "CLEAN_30_main"}
    names |= {ALLOWLIST_NAME.format(fold=fold) for fold in range(N_OUTER_FOLDS)}
    names |= {ELIGIBLE_NAME.format(fold=fold) for fold in range(N_OUTER_FOLDS)}
    return frozenset(names)


def _union(layout: SourceLayout, cohort_info: _Cohort) -> tuple[dict[int, dict[str, Any]], dict[str, _Protein]]:
    """The five training sides (as the cohort audit read them) and one structure per protein across them."""
    recorded = cohort_info.receipt["inputs"]["folds"]
    folds: dict[int, dict[str, Any]] = {}
    entries: dict[str, dict[str, Any]] = {}
    conflicts = 0
    for fold in range(N_OUTER_FOLDS):
        sources = _read_fold_sources(layout, fold)
        record = recorded.get(str(fold)) or {}
        if ((record.get("manifest") or {}).get("sha256") != sources["manifest_sha256"]
                or (record.get("training_csv") or {}).get("sha256") != sources["csv_sha256"]):
            raise OverlayError(f"Fold {fold}'s training manifest or site table differs from the one the cohort audit "
                               "read (SHA-256); the overlay copies the files the audit checked.")
        stems = sorted(sources["sources"])
        folds[fold] = {
            "manifest_path": sources["manifest_path"], "manifest_sha256": sources["manifest_sha256"],
            "csv_path": sources["csv_path"], "csv_sha256": sources["csv_sha256"], "stems": stems,
            "eligible": [stem for stem in stems if sources["status"][stem] == STATUS_OK],
            "several": [stem for stem in stems if sources["status"][stem] == STATUS_AMBIGUOUS],
            "accessions": frozenset(row.accession for row in sources["sources"].values()),
        }
        for stem in stems:
            row = sources["sources"][stem]
            key = (stem, row.structure_name, row.sha256, int(row.size_bytes),
                   _data_relative(row.path, layout.data_root))
            entry = entries.setdefault(row.accession, {"key": key, "source": Path(row.path)})
            conflicts += entry["key"] != key
    if conflicts:
        raise OverlayError(f"The training manifests disagree about {conflicts} protein entries (structure name, "
                           "SHA-256, size or path); the overlay holds one structure per protein.")
    relatives = [entry["key"][4] for entry in entries.values()]
    outside = sum(relative is None for relative in relatives)
    reserved = sum(relative is not None and relative.split("/", 1)[0] in _reserved_top_level()
                   for relative in relatives)
    if outside or reserved:
        raise OverlayError(f"{outside} structure paths lie outside the data root and {reserved} inside a folder name "
                           "the overlay reserves; the overlay copies each manifest unchanged, so every structure "
                           "must lie inside the data root, outside the overlay's own folders.")
    if any(is_test_named_path(layout.data_root / relative, relative_to=layout.data_root) for relative in relatives):
        raise HeldOutInputError("A training structure path names a test or mixed folder; refused.")
    union = {accession: _Protein(accession=accession, stem=entry["key"][0], structure_name=entry["key"][1],
                                 source=entry["source"], relative=entry["key"][4], sha256=entry["key"][2],
                                 size_bytes=entry["key"][3])
             for accession, entry in sorted(entries.items())}
    return folds, union


def _complement_names(fold: int) -> tuple[str, ...]:
    return (*COMPLEMENT_FILE_NAMES, f"strata_fold{fold}.tsv")


def _load_identity_fold(work: Path, fold: int) -> r2_lists.FoldInputs:
    try:
        return r2_lists.load_fold_inputs(work, fold)
    except r2_lists.R2ListError as exc:
        if isinstance(exc.__cause__, EcMembershipError):
            raise _withheld(exc.__cause__, f"Fold {fold}'s inner membership") from None
        raise OverlayError(f"The fold-{fold} identity chain does not load: {exc}") from None
    except (KeyError, TypeError) as exc:
        raise OverlayError(f"The fold-{fold} identity receipts are malformed ({type(exc).__name__}).") from None


def _identity_file_names(work: Path, fold: int, inputs: r2_lists.FoldInputs,
                         r1_records: Mapping[int, Mapping[str, Any]]) -> list[str]:
    """Every file the identity loaders read for this fold's chain and inner steps, by the names the receipts give."""
    what = f"Fold {fold}'s identity receipts"
    try:
        chain = inputs.chain
        names = [f"fasta_receipt_{identity.SET_METALLO_TRAIN}.json"]
        names += [chain.inputs["outputs"][key]["file"] for key in ("target_fasta", "query_fasta", "nodes")]
        names += [f"{SEARCH_DIR}/search_receipt.json", chain.search["sorted_output"]["file"]]
        names += [f"edges_receipt_{identity.SET_METALLO_TRAIN}.json", *chain.edges["outputs"]]
        components_name = f"components_receipt_{identity.SCOPE_INNER}.json"
        components = _parse_json(_read_regular(work / components_name, "The components receipt"), components_name)
        names += [components_name, *components["outputs"]]
        assignment = identity._load_assignment_receipt(work, fold)
        names += [identity.INNER_ASSIGNMENT_RECEIPT, assignment["outputs"]["membership"]["file"],
                  identity.INNER_CERTIFICATE]
        for record in r1_records.values():
            names += [record["receipt"], record["file"]]
    except (KeyError, TypeError) as exc:
        raise OverlayError(f"{what} are malformed ({type(exc).__name__}).") from None
    except ValueError as exc:
        raise OverlayError(f"{what}: {exc}") from None
    checked = sorted({_plain_name(name, what=what, allowed_dirs=frozenset({SEARCH_DIR})) for name in names})
    for name in checked:
        problem = _file_problem(work / name)
        if problem is not None:
            raise OverlayError(f"Fold {fold}: identity file {name} cannot be copied ({problem}).")
    return checked


def _check_training_file_binding(work: Path, fold: int, inputs: r2_lists.FoldInputs,
                                 folds: Mapping[int, Mapping[str, Any]]) -> None:
    """Fold k's identity chain read the training site tables and manifests the cohort audit read (SHA-256).

    The fasta receipt records all five training site tables (and the structure manifests it
    used); the assignment receipt and the inner certificate record fold k's site table. Each
    must equal what the cohort audit read, or the copied site tables and inner components
    would disagree. Messages name folds only.
    """
    folds_set = {str(index) for index in range(N_OUTER_FOLDS)}
    try:
        sources = inputs.chain.inputs["sources"]
        tables = sources["metallo_training_files"]
        manifests = sources.get("metallo_structure_manifests") or {}
        if not isinstance(tables, Mapping) or set(tables) != folds_set or not isinstance(manifests, Mapping):
            raise OverlayError(f"Fold {fold}'s fasta receipt does not record the five training site tables it read.")
        other_tables = [index for index in range(N_OUTER_FOLDS)
                        if tables[str(index)].get("sha256") != folds[index]["csv_sha256"]]
        other_manifests = [index for index in range(N_OUTER_FOLDS) if manifests.get(str(index)) is not None
                           and manifests[str(index)].get("sha256") != folds[index]["manifest_sha256"]]
    except (KeyError, TypeError, AttributeError) as exc:
        raise OverlayError(f"Fold {fold}'s fasta receipt is malformed ({type(exc).__name__}).") from None
    if other_tables or other_manifests:
        raise OverlayError(f"Fold {fold}: the identity chain read other training files than the cohort audit (SHA-256 "
                           f"differs for the site tables of folds {other_tables} and the structure manifests of folds "
                           f"{other_manifests}); the identity folder belongs to other training files.")
    for name, label, pinned in ((identity.INNER_ASSIGNMENT_RECEIPT, "assignment receipt",
                                 inputs.record.get("assignment_receipt_sha256")),
                                (identity.INNER_CERTIFICATE, "inner certificate", inputs.certificate_sha256)):
        data = _read_regular(work / name, f"Fold {fold}'s {label}")
        if sha256_bytes(data) != pinned:
            raise OverlayError(f"Fold {fold}'s {label} changed while the identity chain was read.")
        receipt = _parse_json(data, f"Fold {fold}'s {label}")
        try:
            recorded = receipt["inputs"]["metallo_training_file"]["sha256"]
        except (KeyError, TypeError) as exc:
            raise OverlayError(f"Fold {fold}'s {label} is malformed ({type(exc).__name__}).") from None
        if recorded != folds[fold]["csv_sha256"]:
            raise OverlayError(f"Fold {fold}: the {label} was made from another fold-{fold} training site table than "
                               "the one the cohort audit read (SHA-256).")


def _identity_inputs(identity_root: str | os.PathLike, fold_work: Any,
                     folds: Mapping[int, Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    """Each fold's verified training-side identity chain, cross-checked with its training side."""
    identity_root = _abs(identity_root)
    refuse_test_paths([identity_root], what="--identity-root")
    try:
        works = r2_lists.fold_work_dirs(identity_root, fold_work)
    except r2_lists.R2ListError as exc:
        raise OverlayError(str(exc)) from None
    result: dict[int, dict[str, Any]] = {}
    for fold, work in sorted(works.items()):
        work = _abs(work)
        refuse_test_paths([work], what=f"The fold-{fold} identity folder")
        if not work.is_dir():
            raise OverlayError(f"The fold-{fold} identity folder is not a folder: {work}")
        held = [name for name in _complement_names(fold) if os.path.lexists(work / name)]
        if held:
            raise OverlayError(f"The fold-{fold} identity folder {work} holds complement (test-side) outputs {held}; "
                               "the overlay copies training-side identity folders only (before the identity audit).")
        if os.path.lexists(work / identity.INNER_ASSIGNMENT_REFUSAL):
            raise OverlayError(f"The fold-{fold} identity folder {work} holds a refused inner assignment.")
        inputs = _load_identity_fold(work, fold)
        _check_training_file_binding(work, fold, inputs, folds)
        r1: dict[int, Mapping[str, Any]] = {}
        if fold in r2_lists.R2_INNER_OUTER_FOLDS:
            for inner in range(identity.N_INNER_FOLDS):
                try:
                    r1[inner] = r2_lists.load_r1_inner_exclusions(inputs, inner)[1]
                except r2_lists.R2ListError as exc:
                    raise OverlayError(f"Fold {fold}: {exc}") from None
        names = _identity_file_names(work, fold, inputs, r1)
        side = folds[fold]
        if set(inputs.nodes) != set(side["accessions"]):
            raise OverlayError(f"Fold {fold}: the identity chain's node table and the fold's training manifest name "
                               "different proteins; the identity folder belongs to other training files.")
        membership_ids = sorted(row["structure_id"] for row in inputs.membership.rows)
        if membership_ids != list(side["eligible"]):
            raise OverlayError(f"Fold {fold}: the certified inner membership does not cover exactly the fold's "
                               "training structures with one EC class.")
        if set(inputs.several) != {accession_from_structure_id(stem) for stem in side["several"]}:
            raise OverlayError(f"Fold {fold}: the assignment receipt's several-class list differs from the fold's "
                               "training structures with several EC classes.")
        result[fold] = {"work": work, "override": work != identity_root / IDENTITY_FOLD_DIR.format(fold=fold),
                        "inputs": inputs, "names": names, "r1": r1, "membership_ids": membership_ids}
    return result


def _r2_inputs(r2_dir: str | os.PathLike, identity_info: Mapping[int, Mapping[str, Any]], cohort_info: _Cohort, *,
               smoke: bool) -> dict[str, Any]:
    """The verified R2 lists, built from this cohort audit and these identity chains."""
    root = _abs(r2_dir)
    refuse_test_paths([root], what="--r2-lists")
    try:
        verified = r2_lists.verify_r2_lists(root)
    except r2_lists.R2ListError as exc:
        raise OverlayError(f"The R2 lists do not verify: {exc}") from None
    data = _read_regular(root / r2_lists.RECEIPT_NAME, "The R2-list receipt")
    if sha256_bytes(data) != verified["receipt_sha256"]:
        raise OverlayError(f"{r2_lists.RECEIPT_NAME} changed while it was read.")
    receipt = _parse_json(data, r2_lists.RECEIPT_NAME)
    if not smoke and (receipt.get("provenance") or {}).get("smoke") is not False:
        raise OverlayError("The R2 lists are an engineering smoke; a real overlay needs real (non-smoke) R2 lists.")
    if (receipt.get("cohort") or {}).get("receipt_sha256") != cohort_info.receipt_sha256:
        raise OverlayError("The R2 lists were built from another cohort audit receipt.")
    try:
        for fold, info in identity_info.items():
            if receipt["folds"][str(fold)]["inputs"] != dict(info["inputs"].record):
                raise OverlayError(f"Fold {fold}: the R2 lists were built from other identity tables.")
            listed = r2_lists.read_allowlist(root / r2_lists.R2_ALLOWLIST_NAME.format(fold=fold))
            if listed != sorted(info["inputs"].nodes):
                raise OverlayError(f"Fold {fold}: the R2 allowlist differs from the identity chain's node table.")
            for inner, record in info["r1"].items():
                used = receipt["r2_inner"][str(fold)][str(inner)]["r1_inner_exclusions"]
                if (used.get("sha256"), used.get("receipt_sha256")) != (record["sha256"], record["receipt_sha256"]):
                    raise OverlayError(f"Fold {fold}, inner fold {inner}: the R2-inner list was built from other "
                                       "R1-inner exclusions.")
        outputs = {name: entry["sha256"] for name, entry in receipt["outputs"].items()}
    except r2_lists.R2ListError as exc:
        raise OverlayError(f"The R2 lists do not verify: {exc}") from None
    except (KeyError, TypeError, AttributeError) as exc:
        raise OverlayError(f"{r2_lists.RECEIPT_NAME} is malformed ({type(exc).__name__}).") from None
    for name in outputs:
        _plain_name(name, what=f"{r2_lists.RECEIPT_NAME} outputs")
    return {"root": root, "receipt": receipt, "receipt_sha256": verified["receipt_sha256"], "outputs": outputs}


def _collect(*, layout: SourceLayout, cohort_dir: str | os.PathLike, identity_root: str | os.PathLike,
             fold_work: Any, r2_dir: str | os.PathLike, smoke: bool) -> _Plan:
    """Read and cross-check every input; return the overlay's file list (nothing is written)."""
    cohort_info = _load_cohort(cohort_dir, smoke=smoke)
    folds, union = _union(layout, cohort_info)
    if set(union) != set(cohort_info.proteins):
        raise OverlayError(f"The training sides hold {len(union)} proteins and {cohort.PROTEINS_NAME} "
                           f"{len(cohort_info.proteins)}, with {len(set(union) ^ set(cohort_info.proteins))} not "
                           "shared; the cohort audit read other training files.")
    differing = sum(protein.sha256 != cohort_info.proteins[accession]["structure_sha256"]
                    for accession, protein in union.items())
    if differing:
        raise OverlayError(f"{differing} structures differ from the SHA-256 the cohort audit recorded.")
    identity_info = _identity_inputs(identity_root, fold_work, folds)
    r2_info = _r2_inputs(r2_dir, identity_info, cohort_info, smoke=smoke)

    items: list[_Item] = []
    for fold in range(N_OUTER_FOLDS):
        side = folds[fold]
        for role, path, digest in ((ROLE_TRAINING_MANIFEST, side["manifest_path"], side["manifest_sha256"]),
                                   (ROLE_TRAINING_SITE_TABLE, side["csv_path"], side["csv_sha256"])):
            relative = _data_relative(Path(path), layout.data_root)
            items.append(_Item(relative, role, {"kind": "copy", "root": "data_root", "path": relative},
                               source=Path(path), expected_sha256=digest))
    for accession, protein in union.items():
        recorded = cohort_info.proteins[accession]
        items.append(_Item(protein.relative, ROLE_STRUCTURE,
                           {"kind": "copy", "root": "data_root", "path": protein.relative},
                           source=protein.source, expected_sha256=protein.sha256))
        for role, name, column in ((ROLE_ESM, esm_file_name(protein.stem), "esm_sha256"),
                                   (ROLE_ESM_SIDECAR, esm_sidecar_name(protein.stem), "esm_sidecar_sha256")):
            items.append(_Item(f"{ESM_DIR}/{name}", role, {"kind": "copy", "root": "esm_dir", "path": name},
                               source=cohort.protein_layout_path(layout.esm_dir, name),
                               expected_sha256=recorded[column]))
        relative = f"{protein.stem}/{EXTERNAL_FEATURE_FILE}"
        external = cohort.protein_layout_path(layout.external_root, protein.stem) / EXTERNAL_FEATURE_FILE
        items.append(_Item(f"{EXTERNAL_DIR}/{relative}", ROLE_EXTERNAL,
                           {"kind": "copy", "root": "external_root", "path": relative},
                           source=external, expected_sha256=recorded["external_sha256"]))
    for fold, info in identity_info.items():
        folder = f"{IDENTITY_DIR}/{IDENTITY_FOLD_DIR.format(fold=fold)}"
        for name in info["names"]:
            items.append(_Item(f"{folder}/{name}", ROLE_IDENTITY,
                               {"kind": "copy", "root": f"identity_fold{fold}", "path": name},
                               source=info["work"] / name))
    items.append(_Item(f"{COHORT_DIR}/{cohort.RECEIPT_NAME}", ROLE_COHORT,
                       {"kind": "copy", "root": "cohort", "path": cohort.RECEIPT_NAME},
                       source=cohort_info.root / cohort.RECEIPT_NAME, expected_sha256=cohort_info.receipt_sha256))
    for name, digest in sorted(cohort_info.outputs.items()):
        items.append(_Item(f"{COHORT_DIR}/{name}", ROLE_COHORT, {"kind": "copy", "root": "cohort", "path": name},
                           source=cohort_info.root / name, expected_sha256=digest))
    items.append(_Item(f"{R2_DIR}/{r2_lists.RECEIPT_NAME}", ROLE_R2,
                       {"kind": "copy", "root": "r2_lists", "path": r2_lists.RECEIPT_NAME},
                       source=r2_info["root"] / r2_lists.RECEIPT_NAME, expected_sha256=r2_info["receipt_sha256"]))
    for name, digest in sorted(r2_info["outputs"].items()):
        items.append(_Item(f"{R2_DIR}/{name}", ROLE_R2, {"kind": "copy", "root": "r2_lists", "path": name},
                           source=r2_info["root"] / name, expected_sha256=digest))

    for fold in range(N_OUTER_FOLDS):
        inputs = identity_info[fold]["inputs"]
        items.append(_Item(ALLOWLIST_NAME.format(fold=fold), ROLE_ALLOWLIST,
                           {"kind": "derived", "rule": RULES["allowlist"],
                            "from": {"training_manifest_sha256": folds[fold]["manifest_sha256"]}},
                           data=_lines_bytes(folds[fold]["stems"])))
        items.append(_Item(ELIGIBLE_NAME.format(fold=fold), ROLE_ELIGIBLE,
                           {"kind": "derived", "rule": RULES["eligible_list"],
                            "from": {"membership_sha256": inputs.membership.file_sha256,
                                     "certificate_sha256": inputs.certificate_sha256}},
                           data=_lines_bytes(identity_info[fold]["membership_ids"])))
    index_rows = [{column: cohort_info.proteins[accession][column] for column in LABEL_FREE_INDEX_COLUMNS}
                  for accession in sorted(cohort_info.proteins)]
    items.append(_Item(LABEL_FREE_INDEX_NAME, ROLE_LABEL_FREE_INDEX,
                       {"kind": "derived", "rule": RULES["label_free_index"],
                        "from": {"cohort_proteins_sha256": cohort_info.outputs[cohort.PROTEINS_NAME]}},
                       data=_tsv_bytes(LABEL_FREE_INDEX_COLUMNS, index_rows)))

    paths = Counter(item.path for item in items)
    repeated = sorted(path for path, count in paths.items() if count > 1)
    if repeated:
        roles = {item.path: item.role for item in items}
        raise OverlayError(_describe(repeated, "overlay paths would receive two inputs", roles))
    if any(is_test_named_path(Path("/overlay") / item.path, relative_to="/overlay") for item in items):
        raise HeldOutInputError("An overlay path would name a test or mixed folder; refused.")
    return _Plan(items=tuple(sorted(items, key=lambda item: item.path)), cohort=cohort_info, folds=folds, union=union,
                 identity=identity_info, r2=r2_info)


# ---------------------------------------------------------------------------
# String scan (A13) and the tar
# ---------------------------------------------------------------------------

def string_scan(root: Path, items: Iterable[_Item]) -> dict[str, Any]:
    """A13 count over the per-protein inputs of an overlay folder (ESMC payload metadata, sidecars, external files)."""
    from training.global_esm_features import deserialize_payload

    counts: Counter[str] = Counter({role: 0 for role in SCAN_ROLES})
    unreadable: Counter[str] = Counter()
    for item in items:
        if item.role not in SCAN_ROLES:
            continue
        try:
            data = (root / item.path).read_bytes()
        except OSError:
            unreadable[item.role] += 1
            continue
        if item.role == ROLE_ESM:
            try:
                payload = deserialize_payload(data, origin="payload")
            except Exception:  # noqa: BLE001 - any payload reader failure is counted, never shown
                unreadable[item.role] += 1
                continue
            if isinstance(payload, Mapping):
                counts[item.role] += cohort.count_test_named_strings(
                    {key: value for key, value in payload.items() if key not in ("embeddings", "residue_ids")})
            continue
        try:
            payload = json.loads(data.decode("utf-8"))
        except ValueError:
            unreadable[item.role] += 1
            continue
        if isinstance(payload, Mapping):
            counts[item.role] += cohort.count_test_named_strings(payload)
    return {"rule": RULES["string_scan"], "count": sum(counts.values()), "by_role": dict(sorted(counts.items())),
            "n_unreadable": dict(sorted(unreadable.items()))}


class _HashSink:
    """A write-only file object for ``tarfile``: hashes what is written and optionally passes it on."""

    def __init__(self, target: Optional[BinaryIO] = None):
        self.digest = hashlib.sha256()
        self.offset = 0
        self.target = target

    def write(self, data: bytes) -> int:
        self.digest.update(data)
        self.offset += len(data)
        if self.target is not None:
            self.target.write(data)
        return len(data)

    def tell(self) -> int:
        return self.offset

    def hexdigest(self) -> str:
        return self.digest.hexdigest()


def tar_members(overlay_id: str, file_paths: Iterable[str]) -> list[tuple[str, bool]]:
    """(member name, is folder) of an overlay tar, sorted by name; every parent folder is listed."""
    files = sorted(set(file_paths))
    folders = {""}
    for path in files:
        parts = path.split("/")[:-1]
        folders.update("/".join(parts[:index]) for index in range(1, len(parts) + 1))
    members = [(overlay_id if not folder else f"{overlay_id}/{folder}", True) for folder in folders]
    members += [(f"{overlay_id}/{path}", False) for path in files]
    return sorted(members)


def write_tar(root: Path, overlay_id: str, file_paths: Iterable[str], fileobj: Any) -> None:
    """Write the reproducible tar of ``root``'s listed files into ``fileobj`` (needs ``write`` and ``tell``)."""
    with tarfile.open(fileobj=fileobj, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name, is_folder in tar_members(overlay_id, file_paths):
            info = tarfile.TarInfo(name)
            info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
            if is_folder:
                info.type, info.mode = tarfile.DIRTYPE, TAR_DIR_MODE
                archive.addfile(info)
                continue
            path = root / name[len(overlay_id) + 1:]
            problem = _file_problem(path)
            if problem is not None:
                raise OverlayError(f"An overlay file under {_folder_label(name[len(overlay_id) + 1:])} cannot be "
                                   f"archived ({problem}).")
            try:
                info.type, info.mode, info.size = tarfile.REGTYPE, TAR_FILE_MODE, os.lstat(path).st_size
                with path.open("rb") as handle:
                    archive.addfile(info, handle)
            except OSError as exc:
                raise OverlayError(f"An overlay file under {_folder_label(name[len(overlay_id) + 1:])} cannot be "
                                   f"archived ({_errno_text(exc)}).") from None


def _sha256_line(digest: str, tar_name: str) -> bytes:
    return f"{digest}  {tar_name}\n".encode("utf-8")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _manifest_entries(manifest: Mapping[str, Any], problems: list[str]) -> dict[str, Mapping[str, Any]]:
    entries: dict[str, Mapping[str, Any]] = {}
    files = manifest.get("files")
    if not isinstance(files, list):
        problems.append("the overlay manifest has no files list")
        return entries
    malformed = repeated = 0
    for entry in files:
        path = entry.get("path") if isinstance(entry, Mapping) else None
        parts = path.split("/") if isinstance(path, str) else []
        if (not parts or path.startswith("/") or "\\" in path or any(part in ("", ".", "..") for part in parts)
                or path == MANIFEST_NAME or entry.get("role") not in ROLES
                or not SHA256_RE.match(str(entry.get("sha256") or "")) or not isinstance(entry.get("size_bytes"), int)):
            malformed += 1
            continue
        if path in entries:
            repeated += 1
        entries[path] = entry
    if malformed or repeated:
        problems.append(f"the overlay manifest has {malformed} malformed and {repeated} repeated file entries")
    test_named = sum(is_test_named_path(Path("/overlay") / path, relative_to="/overlay")
                     or Path(path).name == "split_metadata.json" for path in entries)
    if test_named:
        problems.append(f"{test_named} manifest entries name a test or mixed folder or split_metadata.json")
    return entries


def _walk(root: Path) -> tuple[set[str], set[str], list[str]]:
    files: set[str] = set()
    folders: set[str] = set()
    links = test_named = 0
    for current, dirs, names in os.walk(root, followlinks=False):
        relative = Path(current).relative_to(root).as_posix()
        if relative != ".":
            folders.add(relative)
        for name in list(dirs) + list(names):
            path = Path(current) / name
            if os.path.islink(path):
                links += 1
            if names_test_split(name):
                test_named += 1
        for name in names:
            path = Path(current) / name
            if not os.path.islink(path):
                files.add((Path(current) / name).relative_to(root).as_posix())
    problems = []
    if links:
        problems.append(f"{links} symlinks in the overlay (byte copies only)")
    if test_named:
        problems.append(f"{test_named} test-named entries in the overlay")
    return files, folders, problems


def _verify_tree(root: Path, manifest: Mapping[str, Any], *, rescan: bool,
                 known: Optional[Mapping[str, tuple[str, int]]] = None) -> tuple[list[str], Optional[_Plan]]:
    """Every check of an overlay folder except the tar; returns the problems and the re-derived plan."""
    problems: list[str] = []
    entries = _manifest_entries(manifest, problems)
    roles = {path: entry["role"] for path, entry in entries.items()}
    found, folders, walk_problems = _walk(root)
    problems.extend(walk_problems)
    extra = sorted(found - set(entries) - {MANIFEST_NAME})
    missing = sorted(set(entries) - found)
    if extra:
        problems.append(_describe(extra, "files are not in the overlay manifest"))
    if missing:
        problems.append(_describe(missing, "files of the overlay manifest are missing", roles))
    expected_folders = {"/".join(path.split("/")[:index]) for path in entries
                        for index in range(1, len(path.split("/")))}
    extra_folders = sorted(folders - expected_folders)
    if extra_folders:
        problems.append(_describe(extra_folders, "folders hold no manifest file"))
    differing, unreadable, codes = [], [], Counter()
    for path, entry in sorted(entries.items()):
        if path not in found:
            continue
        if known is not None and path in known:
            actual = known[path]
        else:
            try:
                actual = _hash_file(root / path)
            except OSError as exc:
                unreadable.append(path)
                codes[_errno_text(exc)] += 1
                continue
        if actual != (entry["sha256"], entry["size_bytes"]):
            differing.append(path)
    if unreadable:
        reasons = ", ".join(f"{count} {code}" for code, count in sorted(codes.items()))
        problems.append(_describe(unreadable, f"files cannot be read ({reasons})", roles))
    if differing:
        problems.append(_describe(differing, "files differ from the overlay manifest", roles))

    plan = None
    try:
        plan = _collect(layout=_overlay_layout(root), cohort_dir=root / COHORT_DIR, identity_root=root / IDENTITY_DIR,
                        fold_work=None, r2_dir=root / R2_DIR, smoke=bool(manifest.get("smoke")))
    except (OverlayError, HeldOutInputError) as exc:
        problems.append(f"the overlay's own copies do not load as inputs: {exc}")
    except OSError as exc:
        problems.append(f"the overlay's own copies do not load as inputs ({_errno_text(exc)}; the path is withheld)")
    if plan is None:
        return problems, None

    items = {item.path: item for item in plan.items}
    needed = sorted(set(items) - set(entries))
    unneeded = sorted(set(entries) - set(items))
    item_roles = {path: item.role for path, item in items.items()}
    if needed:
        problems.append(_describe(needed, "files the overlay's inputs need are not in the manifest", item_roles))
    if unneeded:
        problems.append(_describe(unneeded, "manifest entries are not needed by the overlay's inputs", roles))
    wrong_role, wrong_hash = [], []
    for path, item in items.items():
        entry = entries.get(path)
        if entry is None:
            continue
        if entry["role"] != item.role:
            wrong_role.append(path)
        expected = sha256_bytes(item.data) if item.data is not None else item.expected_sha256
        if expected is not None and expected != entry["sha256"]:
            wrong_hash.append(path)
    if wrong_role:
        problems.append(_describe(wrong_role, "manifest entries carry another role than their input", roles,
                                  item_roles))
    if wrong_hash:
        problems.append(_describe(wrong_hash, "files differ from the SHA-256 their receipts or the cohort audit "
                                              "record, or from their derivation", roles, item_roles))
    by_protein = Counter(item.role for item in plan.items if item.role in PER_PROTEIN_ROLES)
    if any(by_protein[role] != len(plan.union) for role in PER_PROTEIN_ROLES):
        problems.append(f"the per-protein roles do not cover each of the {len(plan.union)} proteins once")

    inputs = manifest.get("inputs") if isinstance(manifest.get("inputs"), Mapping) else {}
    if (inputs.get("cohort") or {}).get("receipt_sha256") != plan.cohort.receipt_sha256:
        problems.append("the overlay's cohort receipt differs from the one the manifest records")
    if (inputs.get("r2_lists") or {}).get("receipt_sha256") != plan.r2["receipt_sha256"]:
        problems.append("the overlay's R2-list receipt differs from the one the manifest records")
    for fold, info in plan.identity.items():
        record = ((inputs.get("identity") or {}).get(str(fold)) or {})
        if record.get("record") != dict(info["inputs"].record):
            problems.append(f"the fold-{fold} identity copy differs from the chain the manifest records")
    counts = manifest.get("counts") if isinstance(manifest.get("counts"), Mapping) else {}
    if counts.get("n_proteins") != len(plan.union) or counts.get("n_files") != len(entries):
        problems.append("the manifest's protein or file count differs from the overlay")
    per_fold = counts.get("per_fold") if isinstance(counts.get("per_fold"), Mapping) else {}
    if any((per_fold.get(str(fold)) or {}).get("n_eligible") != len(plan.identity[fold]["membership_ids"])
           for fold in range(N_OUTER_FOLDS)):
        problems.append("the manifest's per-fold eligible counts differ from the overlay's memberships")
    if rescan and not missing:
        scan = string_scan(root, plan.items)
        if scan["count"] != (manifest.get("string_scan") or {}).get("count") or scan["n_unreadable"]:
            problems.append("the string scan of the per-protein inputs differs from the manifest's count")
    return problems, plan


def read_overlay_manifest(overlay_dir: str | os.PathLike) -> tuple[dict[str, Any], str]:
    """The overlay manifest of a folder (checked for format and ID) and its SHA-256."""
    root = _abs(overlay_dir)
    data = _read_regular(root / MANIFEST_NAME, "The overlay manifest")
    manifest = _parse_json(data, MANIFEST_NAME)
    if (not isinstance(manifest, dict) or manifest.get("format") != OVERLAY_FORMAT
            or manifest.get("schema_version") != OVERLAY_SCHEMA_VERSION or manifest.get("kind") != OVERLAY_KIND
            or manifest.get("campaign_id") != CAMPAIGN_ID):
        raise OverlayError(f"{root / MANIFEST_NAME} is not a schema-{OVERLAY_SCHEMA_VERSION} overlay manifest of "
                           f"{CAMPAIGN_ID}.")
    if manifest.get("overlay_id") != root.name:
        raise OverlayError(f"The overlay folder {root.name!r} does not carry its manifest's ID "
                           f"{manifest.get('overlay_id')!r}.")
    return manifest, sha256_bytes(data)


def _check_tar(root: Path, manifest: Mapping[str, Any], problems: list[str]) -> Optional[str]:
    """The tar next to the overlay: SHA-256 file, member list and a deterministic re-creation; returns its SHA-256."""
    overlay_id = str(manifest["overlay_id"])
    side = overlay_side_paths(root)
    tar_problem, sha_problem = _file_problem(side["tar"]), _file_problem(side["tar_sha256"])
    if tar_problem or sha_problem:
        problems.append("the tar or its .sha256 file next to the overlay cannot be read "
                        f"({tar_problem or sha_problem})")
        return None
    try:
        actual = _hash_file(side["tar"])[0]
        listed_line = side["tar_sha256"].read_bytes()
    except OSError as exc:
        problems.append(f"the tar or its .sha256 file next to the overlay cannot be read ({_errno_text(exc)})")
        return None
    if listed_line != _sha256_line(actual, side["tar"].name):
        problems.append(f"{side['tar_sha256'].name} does not hold the tar's SHA-256 next to its bare file name")
    files = [entry.get("path") for entry in manifest.get("files") or [] if isinstance(entry, Mapping)]
    expected = tar_members(overlay_id, [*files, MANIFEST_NAME])
    try:
        with tarfile.open(side["tar"], "r:") as archive:
            members = archive.getmembers()
    except (tarfile.TarError, OSError) as exc:
        problems.append(f"the tar's member list cannot be read ({type(exc).__name__})")
        members = None
    if members is not None:
        listed = [(member.name, member.isdir()) for member in members]
        if listed != expected:
            problems.append(f"the tar's member list differs from the overlay manifest ({len(listed)} members, "
                            f"{len(expected)} expected)")
        odd = sum(member.mtime != 0 or member.uid != 0 or member.gid != 0 or bool(member.uname) or bool(member.gname)
                  or member.mode != (TAR_DIR_MODE if member.isdir() else TAR_FILE_MODE)
                  or not (member.isdir() or member.isfile()) for member in members)
        if odd:
            problems.append(f"{odd} tar members differ from the reproducible header rule")
    sink = _HashSink()
    try:
        write_tar(root, overlay_id, [*files, MANIFEST_NAME], sink)
    except OverlayError as exc:
        problems.append(f"the tar cannot be re-created from the overlay folder ({exc})")
        return actual
    except OSError as exc:
        problems.append(f"the tar cannot be re-created from the overlay folder ({_errno_text(exc)})")
        return actual
    if sink.hexdigest() != actual:
        problems.append("the tar differs from a re-creation from the overlay folder")
    return actual


def _check_build_receipt(root: Path, manifest: Mapping[str, Any], manifest_sha: str, tar_sha: Optional[str],
                         problems: list[str]) -> Optional[str]:
    """The build receipt next to the overlay binds this manifest and tar; returns its SHA-256 (``None`` if absent).

    A real overlay must keep its build receipt (mirror it with the tar); a smoke overlay may lack one.
    """
    path = overlay_side_paths(root)["build_receipt"]
    problem = _file_problem(path)
    if problem == "missing" and manifest.get("smoke"):
        return None
    if problem is not None:
        hint = "; a real overlay keeps it next to the tar" if problem == "missing" else ""
        problems.append(f"the build receipt {path.name} next to the overlay cannot be read ({problem}){hint}")
        return None
    try:
        data = path.read_bytes()
    except OSError as exc:
        problems.append(f"the build receipt {path.name} cannot be read ({_errno_text(exc)})")
        return None
    try:
        receipt = json.loads(data.decode("utf-8"))
    except ValueError:
        receipt = None
    if (not isinstance(receipt, Mapping) or receipt.get("format") != BUILD_RECEIPT_FORMAT
            or receipt.get("campaign_id") != CAMPAIGN_ID):
        problems.append(f"{path.name} is not a build receipt of this campaign's overlays")
        return sha256_bytes(data)
    tar = receipt.get("tar") if isinstance(receipt.get("tar"), Mapping) else {}
    provenance = receipt.get("provenance") if isinstance(receipt.get("provenance"), Mapping) else {}
    mismatched = [name for name, ok in (
        ("overlay_id", receipt.get("overlay_id") == manifest.get("overlay_id")),
        ("smoke", receipt.get("smoke") is bool(manifest.get("smoke"))),
        ("manifest_sha256", receipt.get("manifest_sha256") == manifest_sha),
        ("tar.file", tar.get("file") == overlay_side_paths(root)["tar"].name),
        ("tar.sha256", tar_sha is not None and tar.get("sha256") == tar_sha),
        ("provenance", _stable_provenance(provenance) == manifest.get("provenance")),
    ) if not ok]
    if mismatched:
        problems.append(f"{path.name} does not bind this overlay ({', '.join(mismatched)} differ)")
    return sha256_bytes(data)


def verify_overlay(overlay_dir: str | os.PathLike, *, receipt_path: Optional[str | os.PathLike] = None,
                   provenance: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Rehash and re-derive an overlay and check its tar; raise ``OverlayVerificationError`` listing every problem.

    ``receipt_path`` (a new file outside the overlay) receives the result with ``provenance``.
    """
    root = _abs(overlay_dir)
    manifest, manifest_sha = read_overlay_manifest(root)
    if receipt_path is not None:
        receipt_path = _abs(receipt_path)
        if _under(_real(receipt_path), [_real(root)]) or os.path.lexists(receipt_path):
            raise OverlayError(f"The verification receipt {receipt_path} must be a new file outside the overlay.")
    problems, plan = _verify_tree(root, manifest, rescan=True)
    tar_sha = _check_tar(root, manifest, problems)
    build_receipt_sha = _check_build_receipt(root, manifest, manifest_sha, tar_sha, problems)
    if problems:
        raise OverlayVerificationError(problems)
    side = overlay_side_paths(root)
    result = {
        "status": "verified", "overlay_id": manifest["overlay_id"], "overlay": str(root), "smoke": manifest["smoke"],
        "manifest_sha256": manifest_sha, "tar": str(side["tar"]), "tar_sha256": tar_sha,
        "build_receipt": str(side["build_receipt"]) if build_receipt_sha else None,
        "build_receipt_sha256": build_receipt_sha,
        "n_files": len(manifest["files"]), "n_proteins": len(plan.union),
        "string_scan_count": manifest["string_scan"]["count"],
        "checks": ["every file rehashed", "no extra, missing or linked file", "each protein once per role",
                   "identity loaders on the copies", "R2 lists verified", "derived lists re-derived",
                   "string scan repeated", "tar SHA-256 file, member list and re-creation",
                   "build receipt binds the manifest and the tar"],
    }
    if receipt_path is not None:
        result["receipt"] = str(receipt_path)
        result["receipt_sha256"] = write_new_json(receipt_path, {**result, "verified_utc": utc_now(),
                                                                 "provenance": dict(provenance or {})})
    return result


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _check_output_location(out: Path, *, overlay_id: str, smoke: bool, layout: SourceLayout) -> None:
    from project_paths import DATA_DIR

    if is_test_named_path(out) or names_test_split(overlay_id):
        raise HeldOutInputError(f"Overlay folder {out} is test-named; refused.")
    if not OVERLAY_ID_RE.match(overlay_id):
        raise OverlayError(f"The overlay folder name {overlay_id!r} is not a valid overlay ID (letters, digits, '.', "
                           "'_' and '-').")
    for candidate in (out, Path(os.path.realpath(out))):
        blocked = [part for part in candidate.parts[1:] if r2_lists.COMPARATOR_HELD_OUT_NAME_RE.search(part)
                   or part.lower() in r2_lists.COMPARATOR_HELD_OUT_DIR_NAMES]
        if blocked:
            raise HeldOutInputError(f"Overlay folder {out} runs through {blocked}, which the comparator refuses for "
                                    "the R1-inner tables the overlay holds.")
    if not smoke:
        if not REAL_OVERLAY_ID_RE.match(overlay_id):
            raise OverlayError(f"A real overlay is named ecc30-overlay-v<N>, not {overlay_id!r}.")
        if out.parent.name != OVERLAYS_DIR or out.parent.parent.name != CAMPAIGN_ID:
            raise OverlayError(f"A real overlay lives at <campaign root>/{OVERLAYS_DIR}/<overlay id> (campaign root "
                               f"named {CAMPAIGN_ID!r}), not at {out}; fits are barred from that folder.")
    # Only the data roots themselves are resolved; the shared and test-bearing folders below them are named, never
    # looked up.
    data_roots = {form for root in (layout.data_root, Path(DATA_DIR)) for form in (os.path.abspath(root), _real(root))}
    shared = [os.path.join(root, name) for root in data_roots for name in FORBIDDEN_DATA_ROOT_NAMES]
    shared += [os.path.abspath(layout.esm_dir), os.path.abspath(layout.external_root)]
    real = _real(out)
    if _under(real, shared) or _under(str(out), shared):
        raise OverlayError(f"Overlay folder {out} lies inside a shared or test-bearing data root; it belongs in a "
                           "campaign folder.")
    if _under(real, [_real(REPO_ROOT)]) and not _under(real, [_real(REPO_ROOT / "DeepMzyme_Data")]):
        raise OverlayError(f"Overlay folder {out} lies inside the repository's tracked tree; it holds every cohort "
                           "protein and stays with the private campaign data.")
    existing = [path for path in (out, *overlay_side_paths(out).values()) if os.path.lexists(path)]
    if existing:
        raise OverlayError(f"{[str(path) for path in existing]} already exist; an overlay is written once (a rebuild "
                           "takes a new overlay ID).")


class _Uncopied(Exception):
    """One input cannot be copied; ``reason`` names no path (a missing file, a link, or an errno name)."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _copy_item(staging: Path, item: _Item) -> tuple[str, int]:
    """Copy one item into the staging folder (streaming, never through a link); returns (SHA-256, size)."""
    target = staging / item.path
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if item.data is not None:
            with target.open("xb") as handle:
                handle.write(item.data)
            return sha256_bytes(item.data), len(item.data)
        problem = _file_problem(item.source)
        if problem is not None:
            raise _Uncopied(problem)
        digest, size = hashlib.sha256(), 0
        with item.source.open("rb") as source, target.open("xb") as handle:
            for chunk in iter(lambda: source.read(_CHUNK), b""):
                digest.update(chunk)
                handle.write(chunk)
                size += len(chunk)
    except OSError as exc:
        raise _Uncopied(_errno_text(exc)) from None
    return digest.hexdigest(), size


def _remove_tree(path: Path) -> None:
    """Remove a staging folder or file; never raises (it runs while another error is on its way out)."""
    import shutil

    try:
        if not os.path.lexists(path):
            return
        if path.is_dir() and not path.is_symlink():
            for current, _dirs, _files in os.walk(path):
                try:
                    os.chmod(current, 0o755)
                except OSError:
                    pass
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    except OSError:
        pass


def _chmod_read_only(path: Path) -> None:
    try:
        os.chmod(path, TAR_FILE_MODE)
    except OSError:
        pass


def build_overlay(
    *,
    out_dir: str | os.PathLike,
    cohort_dir: str | os.PathLike,
    identity_root: str | os.PathLike,
    r2_lists_dir: str | os.PathLike,
    d2_option: str,
    fold_work: Any = None,
    layout: Optional[SourceLayout] = None,
    smoke: bool = False,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the overlay folder, its manifest, its tar and the tar's ``.sha256`` file; return a printable summary.

    ``layout`` defaults to the source layout the cohort audit recorded. ``fold_work`` maps folds to
    other identity folders (``K=PATH`` strings or {K: PATH}); the copies always go to
    ``identity/fold<k>``. ``provenance`` is the action's provenance block: its content-stable part
    goes into the manifest, the whole block into the build receipt. The summary holds union totals
    only (no per-fold numbers, no accession).
    """
    out = _abs(out_dir)
    overlay_id = out.name
    if d2_option not in D2_OPTIONS:
        raise OverlayError(f"--d2-option must be one of {list(D2_OPTIONS)} (option c is never used).")
    if d2_option not in D2_OPTIONS_BUILT:
        raise OverlayError(f"D2 option {d2_option} needs the derived-feature builder, which is not built; only "
                           f"{list(D2_OPTIONS_BUILT)} can be built now.")
    layout_from = "given" if layout is not None else "cohort_receipt"
    if layout is None:
        layout = layout_from_cohort(cohort_dir)
    _check_output_location(out, overlay_id=overlay_id, smoke=smoke, layout=layout)
    try:
        plan = _collect(layout=layout, cohort_dir=cohort_dir, identity_root=identity_root, fold_work=fold_work,
                        r2_dir=r2_lists_dir, smoke=smoke)
    except OSError as exc:
        raise OverlayError(f"An input cannot be read ({_errno_text(exc)}); the path is withheld because it can name a "
                           "structure.") from None

    staging = out.parent / f".{out.name}.partial-{os.getpid()}"
    tar_partial = out.parent / f".{out.name}{TAR_SUFFIX}.partial-{os.getpid()}"
    if os.path.lexists(staging) or os.path.lexists(tar_partial):
        raise OverlayError(f"Staging folder {staging} or {tar_partial} exists; remove it first.")
    side = overlay_side_paths(out)
    staging.mkdir(parents=True)
    try:
        copied: dict[str, tuple[str, int]] = {}
        unreadable: Counter[tuple[str, str]] = Counter()
        mismatched: list[str] = []
        for item in plan.items:
            try:
                copied[item.path] = _copy_item(staging, item)
            except _Uncopied as exc:
                unreadable[(item.role, exc.reason)] += 1
                continue
            if item.expected_sha256 is not None and copied[item.path][0] != item.expected_sha256:
                mismatched.append(item.path)
        roles = {item.path: item.role for item in plan.items}
        if unreadable:
            raise OverlayError("Inputs cannot be copied: " + ", ".join(
                f"{count} {role} ({reason})" for (role, reason), count in sorted(unreadable.items())))
        if mismatched:
            raise OverlayError(_describe(mismatched, "inputs differ from the SHA-256 the cohort audit or their "
                                                     "receipts record", roles))
        scan = string_scan(staging, plan.items)
        cohort_a13 = ((plan.cohort.receipt.get("totals") or {}).get("a13_test_named_string_count") or {}).get("total")
        if scan["n_unreadable"]:
            raise OverlayError(f"The string scan cannot read {scan['n_unreadable']} per-protein inputs.")
        if scan["count"] != cohort_a13:
            raise OverlayError(f"The string scan counts {scan['count']} test-named strings and the cohort audit "
                               f"{cohort_a13}; both read the same bytes with the same rule.")
        scan["cohort_a13_total"] = cohort_a13
        recommended = plan.cohort.receipt.get("a1_recommended_option")
        manifest = {
            "format": OVERLAY_FORMAT,
            "schema_version": OVERLAY_SCHEMA_VERSION,
            "campaign_id": CAMPAIGN_ID,
            "kind": OVERLAY_KIND,
            "overlay_id": overlay_id,
            "smoke": bool(smoke),
            "base": dict(BASE_IDENTITY),
            "layout": dict(OVERLAY_LAYOUT),
            "sources": {
                **layout.to_dict(), "cohort": str(plan.cohort.root),
                "identity_root": str(_abs(identity_root)), "r2_lists": str(plan.r2["root"]),
                "identity_folds": {str(fold): {"path": str(info["work"]), "override": bool(info["override"])}
                                   for fold, info in plan.identity.items()},
            },
            "inputs": {
                "cohort": {"receipt": f"{COHORT_DIR}/{cohort.RECEIPT_NAME}",
                           "receipt_sha256": plan.cohort.receipt_sha256,
                           "status": plan.cohort.receipt.get("status"), "smoke": plan.cohort.receipt.get("smoke")},
                "r2_lists": {"receipt": f"{R2_DIR}/{r2_lists.RECEIPT_NAME}",
                             "receipt_sha256": plan.r2["receipt_sha256"],
                             "smoke": (plan.r2["receipt"].get("provenance") or {}).get("smoke")},
                "identity": {str(fold): {"folder": f"{IDENTITY_DIR}/{IDENTITY_FOLD_DIR.format(fold=fold)}",
                                         "record": dict(info["inputs"].record),
                                         "r1_inner": {str(inner): dict(record) for inner, record in info["r1"].items()}}
                             for fold, info in plan.identity.items()},
            },
            "d2": {"option": d2_option, "policy": D2_POLICY[d2_option], "cohort_a1_recommended_option": recommended,
                   "follows_cohort_recommendation": d2_option == recommended},
            "ring": RING_NOTE,
            "exclusions": dict(EXCLUSIONS_STATEMENT),
            "union_note": UNION_NOTE,
            "counts": {
                "n_proteins": len(plan.union),
                "n_files": len(plan.items),
                "files_by_role": dict(sorted(Counter(item.role for item in plan.items).items())),
                "per_fold": {str(fold): {"n_eligible": len(info["membership_ids"])}
                             for fold, info in plan.identity.items()},
            },
            "string_scan": scan,
            "rules": dict(RULES),
            "code_sha256": {"src/ec_clean30/overlay.py": _hash_file(Path(__file__).resolve())[0],
                            **identity.code_sha256()},
            "provenance": _stable_provenance(provenance),
            "tar": {"file": side["tar"].name, "sha256_file": side["tar_sha256"].name, "member_root": overlay_id,
                    "rule": TAR_RULE},
            "files": [{"path": item.path, "sha256": copied[item.path][0], "size_bytes": copied[item.path][1],
                       "role": item.role, "origin": dict(item.origin)} for item in plan.items],
        }
        manifest_bytes = _json_bytes(manifest)
        with (staging / MANIFEST_NAME).open("xb") as handle:
            handle.write(manifest_bytes)
        problems, _plan = _verify_tree(staging, manifest, rescan=False, known=copied)
        if problems:
            raise OverlayError("The staged overlay does not verify: " + "; ".join(problems[:10]))
        with tar_partial.open("xb") as handle:
            sink = _HashSink(handle)
            write_tar(staging, overlay_id, [*copied, MANIFEST_NAME], sink)
        tar_sha = sink.hexdigest()
        if os.path.lexists(out) or os.path.lexists(side["tar"]):
            raise OverlayError(f"{out} or its tar appeared while the overlay was built.")
        os.replace(staging, out)
        os.replace(tar_partial, side["tar"])
    except BaseException:
        _remove_tree(staging)
        _remove_tree(tar_partial)
        raise
    with side["tar_sha256"].open("xb") as handle:
        handle.write(_sha256_line(tar_sha, side["tar"].name))
    modes = _set_permissions(out)
    manifest_sha = sha256_bytes(manifest_bytes)
    receipt = {"format": BUILD_RECEIPT_FORMAT, "schema_version": OVERLAY_SCHEMA_VERSION,
               "campaign_id": CAMPAIGN_ID, "overlay_id": overlay_id, "overlay": str(out), "smoke": bool(smoke),
               "created_utc": utc_now(), "layout_from": layout_from,
               "manifest_sha256": manifest_sha, "tar": {"file": side["tar"].name, "sha256": tar_sha},
               "file_modes": modes, "provenance": dict(provenance)}
    receipt_sha = write_new_json(side["build_receipt"], receipt)
    for path in (side["tar"], side["tar_sha256"], side["build_receipt"]):
        _chmod_read_only(path)
    return {
        "status": "built", "overlay_id": overlay_id, "overlay": str(out), "smoke": bool(smoke),
        "manifest_sha256": manifest_sha, "tar": str(side["tar"]), "tar_sha256": tar_sha,
        "tar_sha256_file": str(side["tar_sha256"]), "build_receipt": str(side["build_receipt"]),
        "build_receipt_sha256": receipt_sha, "n_proteins": len(plan.union), "n_files": len(plan.items),
        "files_by_role": manifest["counts"]["files_by_role"], "string_scan_count": scan["count"],
        "d2_option": d2_option, "cohort_a1_recommended_option": recommended, "file_modes": modes["outcome"],
        "note": "union totals only; per-fold numbers stay in the private manifest",
    }


# ---------------------------------------------------------------------------
# Runner actions
# ---------------------------------------------------------------------------

def add_overlay_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The runner's ``overlay`` action: sub-actions ``build`` and ``verify`` with their handlers."""
    actions = parser.add_subparsers(dest="overlay_action", required=True, metavar="{build,verify}")
    build = actions.add_parser("build", help="Build the overlay folder, its manifest and its reproducible tar.")
    build.add_argument("--out", required=True,
                       help="New overlay folder <campaign root>/overlay/<overlay id>; its name is the overlay ID "
                            f"(e.g. {DEFAULT_OVERLAY_ID}). The tar and its .sha256 file are written next to it.")
    build.add_argument("--cohort", required=True, help="Cohort audit folder (a passed cohort_audit_receipt.json).")
    build.add_argument("--identity-root", required=True,
                       help="Identity root with one training-side folder fold<k> per outer fold.")
    build.add_argument("--r2-lists", required=True, help=f"R2-list folder (holds {r2_lists.RECEIPT_NAME}).")
    build.add_argument("--d2-option", required=True, choices=D2_OPTIONS,
                       help="Decision A1: b copies external features unchanged (a and d are not built yet).")
    build.add_argument("--fold-work", action="append", default=[], metavar="K=PATH",
                       help="Identity folder of fold K instead of <identity root>/fold<K> (repeatable); the copy "
                            "still goes to identity/fold<K>.")
    build.add_argument("--data-root", default=None,
                       help="Data root (default: the layout the cohort audit recorded).")
    build.add_argument("--esm-dir", default=None, help="Per-residue ESMC folder (with --data-root).")
    build.add_argument("--external-root", default=None, help="External feature root (with --data-root).")
    build.add_argument("--smoke", action="store_true",
                       help="Engineering use: allow a dirty tree, smoke inputs and any location (recorded).")
    build.set_defaults(handler=action_overlay_build)
    verify = actions.add_parser("verify", help="Rehash and re-derive an overlay and check its tar.")
    verify.add_argument("--overlay", required=True, help="Overlay folder (the tar is read next to it).")
    verify.add_argument("--receipt", default=None, help="Write the verification receipt to this new file.")
    verify.set_defaults(handler=action_overlay_verify)
    return parser


def _print(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def action_overlay_build(args: argparse.Namespace) -> int:
    """``overlay build``: refusals, the clean-tree check (real overlays), the build and a printed summary."""
    for name in ("out", "cohort", "identity_root", "r2_lists", "data_root", "esm_dir", "external_root"):
        refuse_test_paths([getattr(args, name, None)], what=f"--{name.replace('_', '-')}")
    for value in args.fold_work or ():
        refuse_test_paths([str(value).partition("=")[2] or value], what="--fold-work")
    git = real_action_git_state(smoke=args.smoke)
    layout = None
    if args.data_root:
        layout = SourceLayout.from_data_root(args.data_root, esm_dir=args.esm_dir, external_root=args.external_root)
    elif args.esm_dir or args.external_root:
        raise OverlayError("--esm-dir and --external-root need --data-root.")
    _print(build_overlay(out_dir=args.out, cohort_dir=args.cohort, identity_root=args.identity_root,
                         r2_lists_dir=args.r2_lists, d2_option=args.d2_option, fold_work=args.fold_work,
                         layout=layout, smoke=args.smoke, provenance=provenance_record(smoke=args.smoke, git=git)))
    return 0


def action_overlay_verify(args: argparse.Namespace) -> int:
    """``overlay verify``: exit 0 when the overlay verifies, 1 with the problems printed otherwise."""
    refuse_test_paths([args.overlay], what="--overlay")
    refuse_test_paths([args.receipt], what="--receipt")
    manifest, _sha = read_overlay_manifest(args.overlay)
    smoke = bool(manifest.get("smoke"))
    git = real_action_git_state(smoke=smoke)
    try:
        result = verify_overlay(args.overlay, receipt_path=args.receipt,
                                provenance=provenance_record(smoke=smoke, git=git))
    except OverlayVerificationError as exc:
        _print({"status": "failed", "problems": exc.problems})
        return 1
    _print(result)
    return 0
