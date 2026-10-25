"""Identity search, edges, components and strata for the EC class vs CLEAN30 campaign.

One search operation everywhere (campaign design, identity search): the pinned
MMseqs2 release 18-8cc5c searches query proteins against fold k's full official
training file, so every hit shares one E-value scale.

Identity is recounted from the alignment strings: identical columns divided by
alignment columns including internal gaps. Residue pairs are columns without a
gap. Every row without an identical non-standard residue is also passed through
``audit_sequence_remoteness.alignment_metrics`` (import only) and must agree
with the recount. Identical non-standard residues (X, B, Z, J, U, O) count as
residue pairs but not as identities (``AMBIGUOUS_RESIDUE_POLICY``).

Hit flags (self-hits are dropped):
- ``any_hit``: E <= 1e-3;
- ``bin_hit``: any_hit and at least 50 residue pairs;
- ``d3`` (the campaign edge): bin_hit and identity >= 0.30, any coverage;
- ``d2``: any_hit, identity >= 0.30 and >= 50% coverage of the shorter sequence;
  ``d2_min_pairs`` also needs 50 residue pairs (Freeze 1 picks one);
- ``d1``: any_hit, identity >= 0.30 and >= 80% coverage of both sequences.

Inputs are training files only, opened by exact constructed names. Complement
queries (test-side), cohort clusters and the strata table refuse to run without
a Freeze 1 SHA-256; it is recorded here and verified against the campaign log
by a later step. Every step of one chain must record the same Freeze 1 value as
the steps it reads. Only searches by the pinned binary feed later steps; an
unpinned binary is an engineering check (``--allow-unpinned-binary``) and is
never accepted for complement queries, cohort clusters or strata. No output of
the steps up to strata carries an EC label or a structure stem.

Inner folds (development tools on fold k's training side; no Freeze 1 SHA-256
needed, but one given must match the chain):
- ``assign-inner``: whole inner components (D3, exact-sequence, alias and donor
  links) into 4 inner folds, stratified by the EC class of fold k's metallo
  training structure stems; writes the EC fold membership, which carries
  structure IDs and EC classes by that contract;
- ``certify-inner``: re-derives eligibility and links and certifies exact
  coverage, unique ownership and zero cross-fold links;
- ``r1-inner-exclusions``: the proteins of fold k's full official training file
  that CLEAN R1-inner drops for one inner validation fold, read from the same
  pinned metallo training search.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import subprocess
import tempfile
import time
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import audit_sequence_remoteness as remoteness
from project_paths import DATA_DIR, PROJECT_ROOT
from training.ec_classes import (
    EC_CLASS_TOKENS, STATUS_AMBIGUOUS, STRUCTURE_ID_RE, accession_from_structure_id, classify_ec_class,
    ec_numbers_from_structure_id,
)
from training.ec_membership import read_ec_fold_membership, write_ec_fold_membership


SCHEMA_VERSION = 1

MMSEQS_RELEASE = "18-8cc5c"
MMSEQS_VERSION = "8cc5ce367b5638c4306c2d7cfc652dd099a4643f"
MMSEQS_BINARY_SHA256 = "b7ef6e0e33df5dd4fa9cf988cbd8b4988c11a3a1255d2c377f13e1bb40c157fb"
DEFAULT_MMSEQS_BINARY = DATA_DIR / "tools" / "mmseqs2-18-8cc5c" / "mmseqs" / "bin" / "mmseqs"
DEFAULT_OFFICIAL_SPLIT_DIR = DATA_DIR / "CLEAN_all_train_valid_splits" / "split30"
DEFAULT_METALLO_FOLDS_DIR = DATA_DIR / "CLEAN_30_main" / "folds"
# Fold k's training-side structure manifest (the trainer's structure stems), by exact name under this root.
DEFAULT_METALLO_MANIFEST_ROOT = DATA_DIR
METALLO_MANIFEST_PATH = "CLEAN_30_train_test_split_{fold}/train/structure_manifest.csv"

N_FOLDS = 5
OFFICIAL_TRAIN_NAME = "split30_train_split_{fold}.csv"
METALLO_TRAIN_NAME = "CLEAN_30_train_test_split_{fold}_train.csv"
DONOR_COLUMN = "single_donor_selected_alphafill_pdb_id"
# Exact directory names that hold held-out files; any training path through them is refused.
FORBIDDEN_PATH_PARTS = frozenset({"test", "metadata", "pinmymetal_files"})

SET_METALLO_TRAIN = "metallo_train"
SET_COMPLEMENT = "complement"
QUERY_SETS = (SET_METALLO_TRAIN, SET_COMPLEMENT)
SCOPE_INNER = "inner"
SCOPE_COHORT = "cohort"

SEARCH_FIELDS: tuple[str, ...] = tuple(remoteness.FIELDS)
SEARCH_SETTINGS: tuple[tuple[str, str], ...] = (
    ("--search-type", "1"),
    ("-s", "7.5"),
    ("-e", "1e-3"),
    ("--alignment-mode", "3"),
    ("--seq-id-mode", "0"),
    ("-a", "1"),
    ("--min-seq-id", "0"),
    ("--min-aln-len", "0"),
    ("-c", "0"),
    ("--cov-mode", "0"),
    ("--mask", "1"),
    ("--comp-bias-corr", "1"),
    ("--max-seqs", "1000000"),
    ("--max-accept", "1000000"),
    ("--max-rejected", "1000000"),
)
SEARCH_RUN_FLAGS = ("--threads", "--split-memory-limit")
MAX_TARGET_SEQUENCES = 1_000_000

MAX_EVALUE = 1e-3
MIN_RESIDUE_PAIRS = 50
IDENTITY_THRESHOLD = Fraction(3, 10)
D2_SHORTER_COVERAGE = Fraction(1, 2)
D1_BOTH_COVERAGE = Fraction(4, 5)

SEQUENCE_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWYBXZJUO")
STANDARD_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWY")
AMBIGUOUS_RESIDUE_POLICY = "identical_nonstandard_residues_count_as_pairs_not_identities"
IDENTITY_RULE = "identical_non_gap_columns / alignment_columns_including_internal_gaps"
ACCESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")
# Four-character PDB entry ID after uppercasing; placeholders or chain-suffixed IDs are refused.
DONOR_ID_RE = re.compile(r"^[0-9][A-Z0-9]{3}$")
COMPLEMENT_SCOPE = ("metallo cohort complement: cohort proteins (any metallo training file) that are absent "
                    "from fold k's official training file; other proteins of the official complement are not queried")

# Strata bins over the maximum bin_hit identity. 0.30 sits in "[30,50]" so that the
# first four labels are exactly "no D3 hit"; the other boundaries await Freeze 1.
IDENTITY_BIN_LABELS = ("no_hit", "<=20", "(20,25]", "(25,30)", "[30,50]", ">50")
IDENTITY_BIN_STATUS = "provisional_until_freeze1"
NEAREST_HIT_ORDER = "highest bits, then lowest E-value, then target accession (provisional until Freeze 1)"

EDGE_COLUMNS = (
    "query", "target", "evalue", "bits", "alnlen", "nident", "residue_pairs", "identity",
    "query_span", "qlen", "target_span", "tlen", "shorter_coverage", "ambiguous_identical_columns",
    "any_hit", "bin_hit", "d3", "d2", "d2_min_pairs", "d1",
)
FLAG_NAMES = ("any_hit", "bin_hit", "d3", "d2", "d2_min_pairs", "d1")
PAIR_COLUMNS = ("a", "b", "n_directions", *FLAG_NAMES, "max_bin_identity", "min_evalue")
NODE_COLUMNS = ("accession", "sequence_sha256", "sequence_length", "donor_pdb_id", "n_structure_ids")
COMPONENT_COLUMNS = ("accession", "component_id", "component_size")
LINK_COLUMNS = ("link_type", "a", "b")
INNER_LINK_TYPES = ("d3", "exact_sequence", "alias", "donor")
COHORT_LINK_TYPES = ("any_hit", "donor")
STRATA_COLUMNS = (
    "accession", "identity_bin", "max_bin_identity", "max_bin_target",
    "has_any_hit", "has_bin_hit", "has_d3", "has_d2", "has_d2_min_pairs", "has_d1",
    "donor_shared_with_metallo_train", "no_d3_hit", "no_d3_hit_no_shared_donor",
    "nearest_target", "nearest_bits", "nearest_evalue", "cluster_id",
)

N_INNER_FOLDS = 4
INNER_SPLIT_SEED = 42
DEFAULT_MAX_COMPONENT_SHARE = Fraction(1, 4)
INNER_MEMBERSHIP_NAME = "inner_membership_fold{fold}.csv"
INNER_ASSIGNMENT_RECEIPT = "inner_assignment_receipt.json"
INNER_ASSIGNMENT_REFUSAL = "inner_assignment_refusal.json"
INNER_CERTIFICATE = "inner_certification_receipt.json"
INNER_ELIGIBILITY_RULE = (
    "fold-k metallo training proteins with one structure ID and one EC class (training.ec_classes."
    "classify_ec_class on the structure stem, as the trainer); several EC classes: excluded and counted, "
    "still component nodes; annotations that cannot give an EC class: refused"
)
INNER_BALANCE_RULE = (
    "one pass, no retries: components with eligible proteins in order of (training count of their rarest EC "
    "class, more eligible proteins first, SHA-256 of '<seed>:<component_id>'); each goes to the inner fold with "
    "the largest class-weighted deficit, the sum over its classes c of n_component(c) / N(c) * "
    "(N(c) / n_inner_folds - n_fold(c)), then to the fold with fewer proteins, then to the lower fold index"
)
MAX_SHARE_RULE = ("a component holding more than the maximum share of all eligible proteins is a refusal; "
                  "the giant-component fallback is a Freeze 1 decision")
MAX_SHARE_BASIS = ("eligible proteins of the component over all eligible proteins (several-class proteins not "
                   "counted); components_receipt_inner.json's largest_component_fraction and "
                   "largest_component_exceeds_quarter count all nodes")
RERUN_RULE = "a rerun with changed settings is a Freeze 1 decision of the maintainer and needs a new working directory"
CERTIFICATION_FAILURE_KEYS = (
    "repeated_accessions", "repeated_structure_ids", "eligible_missing", "unexpected_proteins",
    "structure_id_mismatches", "ec_class_mismatches", "folds_outside_range", "empty_inner_folds",
    "group_id_mismatches", "split_components", "cross_fold_links",
)
R1_EXCLUSION_COLUMNS = ("accession", "reason")
R1_REASON_VALIDATION = "inner_validation"
R1_REASON_D3 = "d3_neighbour"
R1_EXCLUSION_RULE = (
    "inner-validation proteins plus every protein of fold k's full official training file with a D3 edge to one "
    "of them: hits of a validation protein (query) to the full file, and hits of a metallo training protein "
    "(query) to a validation protein, all from the one pinned metallo training search"
)

Runner = Callable[..., subprocess.CompletedProcess]


# ---------------------------------------------------------------- small helpers

def sha256_file(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def write_once(path: Path, text: str) -> str:
    """Write ``text`` unless a different file already exists; returns its SHA-256."""
    path = Path(path)
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f"Refusing to replace an existing output with different content: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return sha256_text(text)


def _tsv_text(columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> str:
    lines = ["\t".join(columns)]
    for row in rows:
        values = [str(row[column]) for column in columns]
        if any("\t" in value or "\n" in value for value in values):
            raise ValueError(f"Tab or newline inside a table value: {values}")
        lines.append("\t".join(values))
    return "\n".join(lines) + "\n"


def read_tsv(path: Path, columns: Sequence[str]) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != tuple(columns):
            raise ValueError(f"Unexpected columns in {path}: {reader.fieldnames}")
        return list(reader)


def _flag(value: bool) -> str:
    return "1" if value else "0"


def _parse_flag(value: str) -> bool:
    if value not in {"0", "1"}:
        raise ValueError(f"Expected a 0/1 flag, got {value!r}")
    return value == "1"


def _fraction_text(value: Fraction) -> str:
    return f"{float(value):.6f}"


def require_freeze1(freeze1_sha256: Optional[str], what: str) -> str:
    """Complement queries, cohort clusters and strata are locked until Freeze 1 is hashed."""
    if not freeze1_sha256:
        raise ValueError(
            f"{what} is locked until Freeze 1: pass --freeze1-sha256 "
            "(recorded here; verified against the campaign log by a later step)."
        )
    if not re.fullmatch(r"[0-9a-f]{64}", freeze1_sha256):
        raise ValueError("--freeze1-sha256 must be 64 lowercase hexadecimal characters.")
    return freeze1_sha256


def require_same_freeze1(freeze1_sha256: Optional[str], receipt: Mapping[str, Any], source: str) -> None:
    """A step records the same Freeze 1 value (or none) as every receipt it builds on."""
    recorded = receipt.get("freeze1_sha256")
    if recorded != freeze1_sha256:
        raise ValueError(f"Freeze 1 SHA-256 {freeze1_sha256} differs from {recorded} recorded in {source}")


def code_sha256() -> dict[str, Optional[str]]:
    paths = {
        "src/ec_clean30/identity.py": Path(__file__).resolve(),
        "src/audit_sequence_remoteness.py": Path(remoteness.__file__).resolve(),
        "scripts/run_ec_clean30_identity.py": PROJECT_ROOT / "scripts" / "run_ec_clean30_identity.py",
    }
    return {name: sha256_file(path) if path.is_file() else None for name, path in paths.items()}


def normalize_sequence(sequence: str) -> str:
    """Same rule as ``audit_sequence_remoteness._sequence``: strip whitespace, uppercase, check alphabet."""
    sequence = "".join(str(sequence).split()).upper()
    if not sequence or any(residue not in SEQUENCE_ALPHABET for residue in sequence):
        raise ValueError("Missing or invalid protein sequence")
    return sequence


def _accession(value: str, source: str) -> str:
    accession = str(value).strip()
    if not ACCESSION_RE.match(accession):
        raise ValueError(f"Invalid accession {accession!r} in {source}")
    return accession


# ---------------------------------------------------------------- training-file readers

def training_file(directory: Path, kind: str, fold: int) -> Path:
    """Exact training-file path for ``kind`` ('official' or 'metallo'); no directory listing."""
    template = {"official": OFFICIAL_TRAIN_NAME, "metallo": METALLO_TRAIN_NAME}.get(kind)
    if template is None:
        raise ValueError(f"Unknown training-file kind {kind!r}")
    return assert_training_file(Path(directory) / template.format(fold=fold), kind, fold)


def assert_training_file(path: Path, kind: str, fold: int) -> Path:
    template = {"official": OFFICIAL_TRAIN_NAME, "metallo": METALLO_TRAIN_NAME}[kind]
    path = Path(path)
    if path.name != template.format(fold=fold):
        raise ValueError(f"Refusing a non-training file name for fold {fold}: {path.name}")
    if path.resolve().name != path.name:
        raise ValueError(f"Refusing a training file that resolves to another name: {path}")
    for parts in (path.parts[:-1], path.resolve().parts[:-1]):
        blocked = [part for part in parts if part.lower() in FORBIDDEN_PATH_PARTS]
        if blocked:
            raise ValueError(f"Refusing a training path through a held-out directory {blocked}: {path}")
    if not path.is_file():
        raise ValueError(f"Training file not found: {path}")
    return path


def _read_allowlisted(path: Path, columns: Mapping[str, tuple[str, ...]],
                      optional: frozenset[str] = frozenset()) -> list[dict[str, str]]:
    """Rows reduced to allowlisted columns; ``columns`` maps output key -> accepted headers."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        header_line = handle.readline()
        handle.seek(0)
        reader = csv.DictReader(handle, delimiter="\t" if "\t" in header_line else ",")
        by_name = {name.strip().lower(): name for name in reader.fieldnames or () if name}
        resolved = {}
        for key, accepted in columns.items():
            name = next((by_name[option] for option in accepted if option in by_name), None)
            if name is None and key not in optional:
                raise ValueError(f"Missing column {accepted[0]!r} in {path}")
            resolved[key] = name
        return [{key: (row.get(name) or "") if name else "" for key, name in resolved.items()}
                for row in reader]


def read_official_sequences(path: Path) -> tuple[dict[str, str], int]:
    """Accession -> sequence from an official training file (EC column never read)."""
    sequences: dict[str, str] = {}
    duplicates = 0
    for line, row in enumerate(_read_allowlisted(path, {"accession": ("entry", "id"),
                                                        "sequence": ("sequence", "sequences")}), start=2):
        accession = _accession(row["accession"], f"{path.name}:{line}")
        try:
            sequence = normalize_sequence(row["sequence"])
        except ValueError as exc:
            raise ValueError(f"{exc} for {accession} in {path.name}:{line}") from None
        if accession in sequences:
            if sequences[accession] != sequence:
                raise ValueError(f"Conflicting duplicate sequences for {accession} in {path.name}")
            duplicates += 1
            continue
        sequences[accession] = sequence
    return sequences, duplicates


@dataclass
class MetalloEntry:
    structure_ids: set[str]
    donors: set[str]


def metallo_structure_manifest(root: Path, fold: int) -> Path:
    """Exact path of fold k's training-side structure manifest; no directory listing."""
    path = Path(root) / METALLO_MANIFEST_PATH.format(fold=fold)
    for parts in (path.parts[:-1], path.resolve().parts[:-1]):
        blocked = [part for part in parts if part.lower() in FORBIDDEN_PATH_PARTS]
        if blocked:
            raise ValueError(f"Refusing a structure manifest through a held-out directory {blocked}: {path}")
    if path.resolve().name != path.name or path.resolve().parent.name != "train":
        raise ValueError(f"Refusing a structure manifest that resolves outside a train folder: {path}")
    return path


def read_structure_stems(path: Path) -> dict[str, set[str]]:
    """Accession -> structure stems (``structure_name`` without ``.pdb``) of a training structure manifest."""
    stems: dict[str, set[str]] = defaultdict(set)
    for row in _read_allowlisted(path, {"structure": ("structure_name",)}):
        name = row["structure"].strip()
        stem = name[:-len(".pdb")] if name.endswith(".pdb") else name
        stems[accession_from_structure_id(stem)].add(stem)
    return dict(stems)


def read_metallo_training(path: Path, manifest: Optional[Path] = None) -> tuple[dict[str, MetalloEntry], bool]:
    """Accession -> structure stems and selected donors from a site-level metallo training CSV.

    The ``structure`` column holds either the structure stem or, in the prepared CLEAN30 folds,
    the accession; then the stems come from the fold's training structure manifest. Also
    returns whether the manifest was read.
    """
    entries: dict[str, MetalloEntry] = {}
    stems_by_accession: Optional[dict[str, set[str]]] = None
    rows = _read_allowlisted(path, {"uniprot_id": ("uniprot_id",), "structure": ("structure",),
                                    "donor": (DONOR_COLUMN.lower(),)}, optional=frozenset({"uniprot_id"}))
    for line, row in enumerate(rows, start=2):
        value = row["structure"].strip()
        if STRUCTURE_ID_RE.match(value):
            stems = {value}
        else:
            if manifest is None or not Path(manifest).is_file():
                raise ValueError(f"{path.name}:{line}: 'structure' holds {value!r}, not a structure stem, and the "
                                 "fold's training structure manifest was not found")
            if stems_by_accession is None:
                stems_by_accession = read_structure_stems(Path(manifest))
            stems = stems_by_accession.get(_accession(value, f"{path.name}:{line}"), set())
            if not stems:
                raise ValueError(f"{value} ({path.name}:{line}) has no structure in {Path(manifest).name}")
        for stem in stems:
            stem_accession = accession_from_structure_id(stem)
            accession = _accession(row["uniprot_id"] or stem_accession, f"{path.name}:{line}")
            if stem_accession.upper() != accession.upper():
                raise ValueError(f"Structure stem accession differs from uniprot_id at {path.name}:{line}")
        donor = row["donor"].strip().upper()
        if donor and not DONOR_ID_RE.match(donor):
            raise ValueError(f"Invalid {DONOR_COLUMN} {row['donor']!r} at {path.name}:{line}; "
                             "expected a four-character PDB entry ID or an empty value")
        entry = entries.setdefault(accession, MetalloEntry(set(), set()))
        entry.structure_ids.update(stems)
        entry.donors.add(donor)
    if stems_by_accession is not None and set(stems_by_accession) != set(entries):
        raise ValueError(f"{path.name} and {Path(manifest).name} name different proteins")
    return entries, stems_by_accession is not None


def _metallo_manifest_record(path: Optional[Path]) -> Optional[dict[str, str]]:
    return {"path": str(path), "sha256": sha256_file(path)} if path is not None and path.is_file() else None


def _single_donor(accession: str, donors: set[str]) -> str:
    if len(donors) != 1:
        raise ValueError(f"Selected donor differs across metallo training rows or files for {accession}: "
                         f"{sorted(donors)}")
    return next(iter(donors))


# ---------------------------------------------------------------- FASTA inputs

def fasta_text(sequences: Mapping[str, str]) -> str:
    return "".join(f">{accession}\n{sequences[accession]}\n" for accession in sorted(sequences))


def read_fasta(path: Path) -> dict[str, str]:
    sequences: dict[str, str] = {}
    accession = None
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            accession = line[1:].strip()
            if accession in sequences:
                raise ValueError(f"Duplicate FASTA record {accession} in {path}")
            sequences[accession] = ""
        elif accession is None:
            raise ValueError(f"FASTA sequence before a header in {path}")
        else:
            sequences[accession] += line.strip()
    return sequences


def _target_name(fold: int) -> str:
    return f"target_train_fold{fold}.fasta"


def _check_query_set(query_set: str, freeze1_sha256: Optional[str]) -> Optional[str]:
    if query_set not in QUERY_SETS:
        raise ValueError(f"Unknown query set {query_set!r}; expected one of {QUERY_SETS}")
    if query_set == SET_COMPLEMENT or freeze1_sha256:
        return require_freeze1(freeze1_sha256, "The complement query set")
    return None


def build_inputs(
    work: Path,
    fold: int,
    *,
    query_set: str = SET_METALLO_TRAIN,
    official_split_dir: Path = DEFAULT_OFFICIAL_SPLIT_DIR,
    metallo_folds_dir: Path = DEFAULT_METALLO_FOLDS_DIR,
    metallo_manifest_root: Optional[Path] = None,
    n_folds: int = N_FOLDS,
    freeze1_sha256: Optional[str] = None,
) -> dict[str, Any]:
    """Write the fold's target FASTA, the query FASTA and node table, and their receipt.

    ``metallo_train``: fold-k metallo training proteins, sequences from fold k's file.
    ``complement`` (Freeze 1 only): metallo cohort proteins absent from fold k's official
    training file (the only test-side proteins the populations need), with sequences taken
    from the other official training files and asserted identical across them.
    """
    freeze1_sha256 = _check_query_set(query_set, freeze1_sha256)
    if not 0 <= fold < n_folds:
        raise ValueError(f"Fold {fold} outside 0..{n_folds - 1}")
    work = Path(work)
    official_paths = {k: training_file(official_split_dir, "official", k) for k in range(n_folds)
                      if k == fold or query_set == SET_COMPLEMENT}
    metallo_paths = {k: training_file(metallo_folds_dir, "metallo", k) for k in range(n_folds)}
    official = {k: read_official_sequences(path) for k, path in official_paths.items()}
    manifest_paths = {k: (metallo_structure_manifest(metallo_manifest_root, k) if metallo_manifest_root else None)
                      for k in range(n_folds)}
    metallo, manifests_read = {}, {}
    for k, path in metallo_paths.items():
        metallo[k], used = read_metallo_training(path, manifest_paths[k])
        manifests_read[k] = manifest_paths[k] if used else None
    target, duplicate_rows = official[fold]
    if len(target) > MAX_TARGET_SEQUENCES:
        raise ValueError("Training file exceeds the pinned MMseqs2 result caps")

    files_per_protein: dict[str, int] = Counter()
    if query_set == SET_METALLO_TRAIN:
        queries = {accession: target.get(accession) for accession in metallo[fold]}
        missing = sorted(accession for accession, sequence in queries.items() if sequence is None)
        if missing:
            raise ValueError(f"Metallo training proteins absent from the official training file: {missing[:20]}")
    else:
        queries = {}
        cohort = set().union(*(set(entries) for entries in metallo.values()))
        for k in sorted(official):
            if k == fold:
                continue
            for accession, sequence in official[k][0].items():
                if accession in target or accession not in cohort:
                    continue
                if queries.setdefault(accession, sequence) != sequence:
                    raise ValueError(f"Sequence of {accession} differs between official training files")
                files_per_protein[accession] += 1

    donors: dict[str, set[str]] = defaultdict(set)
    structures: dict[str, set[str]] = defaultdict(set)
    metallo_files_per_protein: dict[str, int] = Counter()
    for k in sorted(metallo):
        for accession, entry in metallo[k].items():
            if accession in queries:
                donors[accession] |= entry.donors
                structures[accession] |= entry.structure_ids
                metallo_files_per_protein[accession] += 1
    nodes = []
    for accession in sorted(queries):
        sequence = queries[accession]
        nodes.append({
            "accession": accession,
            "sequence_sha256": sha256_text(sequence),
            "sequence_length": len(sequence),
            "donor_pdb_id": _single_donor(accession, donors[accession]) if accession in donors else "",
            "n_structure_ids": len(structures.get(accession, ())),
        })

    outputs = {
        "target_fasta": (_target_name(fold), fasta_text(target), len(target)),
        "query_fasta": (f"query_{query_set}.fasta", fasta_text(queries), len(queries)),
        "nodes": (f"nodes_{query_set}.tsv", _tsv_text(NODE_COLUMNS, nodes), len(nodes)),
    }
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "step": "fasta",
        "fold": fold,
        "n_folds": n_folds,
        "query_set": query_set,
        "freeze1_sha256": freeze1_sha256,
        "freeze1_verification": "recorded_not_verified" if freeze1_sha256 else None,
        "labels_read": False,
        "sources": {
            "official_training_files": {str(k): {"path": str(path), "sha256": sha256_file(path)}
                                        for k, path in sorted(official_paths.items())},
            "metallo_training_files": {str(k): {"path": str(path), "sha256": sha256_file(path)}
                                       for k, path in sorted(metallo_paths.items())},
            "metallo_structure_manifests": {str(k): _metallo_manifest_record(path)
                                            for k, path in sorted(manifests_read.items())},
            "read_columns": {"official": ["Entry", "Sequence"],
                             "metallo": ["uniprot_id", "structure", DONOR_COLUMN],
                             "metallo_structure_manifest": ["structure_name"]},
        },
        "outputs": {},
        "counts": {
            "target_sequences": len(target),
            "target_identical_duplicate_rows": duplicate_rows,
            "query_sequences": len(queries),
            "queries_with_donor": sum(bool(node["donor_pdb_id"]) for node in nodes),
            "largest_donor_group_size": max(Counter(node["donor_pdb_id"] for node in nodes
                                                    if node["donor_pdb_id"]).values(), default=0),
            "queries_with_several_structure_ids": sum(node["n_structure_ids"] > 1 for node in nodes),
            "metallo_training_files_per_query": _histogram(
                metallo_files_per_protein.get(accession, 0) for accession in queries),
        },
        "sequence_policy": "official training-file sequence; whitespace removed, uppercase, "
                           "alphabet ACDEFGHIKLMNPQRSTVWYBXZJUO; identical duplicates collapsed, "
                           "conflicting duplicates refused",
        "donor_policy": f"{DONOR_COLUMN} stripped and uppercased; empty or a four-character PDB entry ID "
                        "(anything else refused); one value per accession across every metallo training "
                        "file and row, otherwise refused",
        "code_sha256": code_sha256(),
    }
    if query_set == SET_COMPLEMENT:
        receipt["complement_scope"] = COMPLEMENT_SCOPE
        receipt["counts"]["official_training_files_per_query"] = _histogram(files_per_protein.values())
    for key, (name, text, count) in outputs.items():
        receipt["outputs"][key] = {"file": name, "sha256": write_once(work / name, text), "n_records": count}
    receipt_name = f"fasta_receipt_{query_set}.json"
    write_once(work / receipt_name, _json_text(receipt))
    return {"step": "fasta", "fold": fold, "query_set": query_set, "receipt": receipt_name,
            "counts": receipt["counts"], "outputs": receipt["outputs"]}


def _histogram(values: Iterable[int]) -> dict[str, int]:
    counts = Counter(values)
    return {str(key): counts[key] for key in sorted(counts)}


def _load_fasta_receipt(work: Path, fold: int, query_set: str) -> dict[str, Any]:
    path = Path(work) / f"fasta_receipt_{query_set}.json"
    if not path.is_file():
        raise ValueError(f"Run the fasta step for {query_set} first: {path} is missing")
    receipt = json.loads(path.read_text())
    if receipt.get("fold") != fold or receipt.get("query_set") != query_set:
        raise ValueError(f"{path.name} belongs to fold {receipt.get('fold')}, set {receipt.get('query_set')}")
    for key, output in receipt["outputs"].items():
        if sha256_file(Path(work) / output["file"]) != output["sha256"]:
            raise ValueError(f"{output['file']} changed after the fasta step")
    return receipt


# ---------------------------------------------------------------- search

def build_search_command(binary: Path, query_fasta: Path, target_fasta: Path, output: Path, tmp_dir: Path,
                         *, threads: int, split_memory_limit: str) -> list[str]:
    command = [str(binary), "easy-search", str(query_fasta), str(target_fasta), str(output), str(tmp_dir)]
    for flag, value in SEARCH_SETTINGS:
        command += [flag, value]
    command += ["--threads", str(threads), "--split-memory-limit", split_memory_limit,
                "--format-mode", "4", "--format-output", ",".join(SEARCH_FIELDS)]
    return command


def check_search_command(command: Sequence[str], query_fasta: Path, target_fasta: Path, output: Path) -> None:
    """Pinned settings and run flags appear exactly once, nothing else; inputs and output are the bound files."""
    command = list(command)
    if len(command) < 6 or command[1] != "easy-search":
        raise ValueError("Search command must be a full easy-search invocation")
    # File names, not absolute paths, so a mirrored working directory still verifies;
    # the receipt binds the contents by SHA-256.
    for index, path, depth in ((2, query_fasta, 1), (3, target_fasta, 1), (4, output, 2)):
        if Path(command[index]).parts[-depth:] != Path(path).parts[-depth:]:
            raise ValueError(f"Search command argument {index} is not {Path(path).name}")
    if len(command) % 2:
        raise ValueError("Search command options must come as flag-value pairs")
    flags, values = command[6::2], command[7::2]
    expected = dict(SEARCH_SETTINGS) | {"--format-mode": "4", "--format-output": ",".join(SEARCH_FIELDS)}
    unpinned = sorted(set(flags) - set(expected) - set(SEARCH_RUN_FLAGS))
    if unpinned:
        raise ValueError(f"Search command has options outside the pinned settings: {unpinned}")
    for flag in (*expected, *SEARCH_RUN_FLAGS):
        if flags.count(flag) != 1:
            raise ValueError(f"Search command needs exactly one {flag}")
    settings = dict(zip(flags, values))
    for flag, value in expected.items():
        actual = settings[flag]
        same = actual == value if flag == "--format-output" else float(actual) == float(value)
        if not same:
            raise ValueError(f"Search setting {flag} is {actual}, pinned {value}")
    if not settings["--threads"].isdigit() or int(settings["--threads"]) < 1:
        raise ValueError(f"Search --threads must be a positive integer, got {settings['--threads']}")


def run_process_group(command: Sequence[str], *, timeout: Optional[float] = None,
                      capture_output: bool = False, **kwargs: Any) -> subprocess.CompletedProcess:
    """``subprocess.run`` in a new session whose whole process group is killed on timeout or interrupt.

    easy-search hands over to a shell workflow whose steps are child processes, so
    killing only the first process could leave a prefilter or align step running.
    """
    if capture_output:
        kwargs["stdout"] = kwargs["stderr"] = subprocess.PIPE
    with subprocess.Popen(list(command), start_new_session=True, **kwargs) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise
    return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)


def is_pinned_binary(binary_sha256: Optional[str], version: Optional[str]) -> bool:
    return binary_sha256 == MMSEQS_BINARY_SHA256 and version == MMSEQS_VERSION


def _check_unpinned_allowed(query_set: str, allow_unpinned_binary: bool) -> None:
    if allow_unpinned_binary and query_set == SET_COMPLEMENT:
        raise ValueError("The complement query set always needs the pinned MMseqs2 binary; "
                         "drop --allow-unpinned-binary")


def verify_binary(binary: Path, *, expected_sha256: str, expected_version: str,
                  runner: Runner = run_process_group) -> tuple[str, str]:
    binary = Path(binary)
    if not binary.is_file():
        raise ValueError(f"MMseqs2 binary not found: {binary}")
    actual_sha = sha256_file(binary)
    if actual_sha != expected_sha256:
        raise ValueError(f"MMseqs2 binary SHA-256 {actual_sha} differs from the pinned {expected_sha256}")
    completed = runner([str(binary), "version"], capture_output=True, text=True, timeout=60)
    version = (completed.stdout or "").strip()
    if completed.returncode or version != expected_version:
        raise ValueError(f"MMseqs2 version {version!r} differs from the pinned {expected_version}")
    return actual_sha, version


def sort_alignment_text(text: str) -> str:
    lines = text.splitlines()
    if not lines or tuple(lines[0].split("\t")) != SEARCH_FIELDS:
        raise ValueError(f"Expected MMseqs2 --format-mode 4 columns: {SEARCH_FIELDS}")
    rows = [line for line in lines[1:] if line]
    rows.sort(key=lambda line: (line.split("\t", 2)[:2], line))
    return "\n".join([lines[0], *rows]) + "\n"


def run_search(
    work: Path,
    fold: int,
    *,
    query_set: str = SET_METALLO_TRAIN,
    binary: Path = DEFAULT_MMSEQS_BINARY,
    expected_sha256: str = MMSEQS_BINARY_SHA256,
    expected_version: str = MMSEQS_VERSION,
    threads: int = 2,
    split_memory_limit: str = "2G",
    timeout_seconds: float = 7200,
    freeze1_sha256: Optional[str] = None,
    allow_unpinned_binary: bool = False,
    runner: Runner = run_process_group,
) -> dict[str, Any]:
    """Run the pinned search once and bind its sorted output in ``search_<set>/search_receipt.json``.

    ``allow_unpinned_binary`` is for engineering checks of the metallo training set only.
    """
    freeze1_sha256 = _check_query_set(query_set, freeze1_sha256)
    _check_unpinned_allowed(query_set, allow_unpinned_binary)
    if threads < 1 or timeout_seconds <= 0:
        raise ValueError("Threads and timeout must be positive")
    work = Path(work)
    inputs = _load_fasta_receipt(work, fold, query_set)
    require_same_freeze1(freeze1_sha256, inputs, f"fasta_receipt_{query_set}.json")
    query_fasta = work / inputs["outputs"]["query_fasta"]["file"]
    target_fasta = work / inputs["outputs"]["target_fasta"]["file"]
    directory = work / f"search_{query_set}"
    receipt_path = directory / "search_receipt.json"
    if receipt_path.exists():
        receipt = verify_search_receipt(work, fold, query_set, require_pinned=not allow_unpinned_binary)
        require_same_freeze1(freeze1_sha256, receipt, "the existing search receipt")
        return {"step": "search", "status": "reused_verified_search", "receipt": str(receipt_path)}
    raw, ordered, log, tmp_dir = (directory / name for name in
                                  ("alignments.raw.tsv", "alignments.sorted.tsv", "search.log", "tmp"))
    if any(path.exists() for path in (raw, ordered, log)):
        raise ValueError(f"Unreceipted earlier search output exists in {directory}; inspect it first")
    binary_sha, version = verify_binary(binary, expected_sha256=expected_sha256,
                                        expected_version=expected_version, runner=runner)
    if not is_pinned_binary(binary_sha, version) and not allow_unpinned_binary:
        raise ValueError(f"MMseqs2 binary {binary_sha} ({version}) is not the pinned {MMSEQS_RELEASE} build; "
                         "engineering checks of the metallo training set pass --allow-unpinned-binary")
    directory.mkdir(parents=True, exist_ok=True)
    command = build_search_command(Path(binary), query_fasta, target_fasta, raw, tmp_dir,
                                   threads=threads, split_memory_limit=split_memory_limit)
    check_search_command(command, query_fasta, target_fasta, raw)
    started = time.monotonic()
    with log.open("x") as handle:
        completed = runner(command, stdout=handle, stderr=subprocess.STDOUT, timeout=timeout_seconds)
    elapsed = time.monotonic() - started
    if completed.returncode:
        raise RuntimeError(f"MMseqs2 exited {completed.returncode}; inspect {log}")
    if not raw.is_file():
        raise RuntimeError(f"MMseqs2 finished without writing {raw}")
    sorted_text = sort_alignment_text(raw.read_text())
    write_once(ordered, sorted_text)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "step": "search",
        "fold": fold,
        "query_set": query_set,
        "freeze1_sha256": freeze1_sha256,
        "freeze1_verification": "recorded_not_verified" if freeze1_sha256 else None,
        "program": "MMseqs2",
        "release": MMSEQS_RELEASE,
        "binary_path": str(binary),
        "pinned_binary": is_pinned_binary(binary_sha, version),
        "allow_unpinned_binary": allow_unpinned_binary,
        "binary_sha256": binary_sha,
        "expected_binary_sha256": expected_sha256,
        "version": version,
        "expected_version": expected_version,
        "command": command,
        "settings": dict(SEARCH_SETTINGS),
        "threads": threads,
        "split_memory_limit": split_memory_limit,
        "database_convention": "target is the fold's full official training file",
        "query_fasta": {"file": query_fasta.name, "sha256": sha256_file(query_fasta)},
        "target_fasta": {"file": target_fasta.name, "sha256": sha256_file(target_fasta)},
        "raw_output": {"file": f"{directory.name}/{raw.name}", "sha256": sha256_file(raw)},
        "sorted_output": {"file": f"{directory.name}/{ordered.name}", "sha256": sha256_text(sorted_text),
                          "n_rows": sorted_text.count("\n") - 1},
        "row_order": "sorted by (query, target), then the full line",
        "search_log_sha256": sha256_file(log),
        "elapsed_seconds": round(elapsed, 3),
        "max_child_rss_kib_process_lifetime": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "code_sha256": code_sha256(),
        "labels_read": False,
    }
    write_once(receipt_path, _json_text(receipt))
    verify_search_receipt(work, fold, query_set, require_pinned=not allow_unpinned_binary)
    return {"step": "search", "status": "search_completed", "receipt": str(receipt_path),
            "n_rows": receipt["sorted_output"]["n_rows"], "elapsed_seconds": receipt["elapsed_seconds"]}


def verify_search_receipt(work: Path, fold: int, query_set: str, *, require_pinned: bool = True) -> dict[str, Any]:
    """Re-check a search receipt against its inputs and output; by default only the pinned binary passes."""
    work = Path(work)
    inputs = _load_fasta_receipt(work, fold, query_set)
    directory = work / f"search_{query_set}"
    receipt_path = directory / "search_receipt.json"
    if not receipt_path.is_file():
        raise ValueError(f"Run the search step for {query_set} first: {receipt_path} is missing")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("fold") != fold or receipt.get("query_set") != query_set:
        raise ValueError("Search receipt belongs to another fold or query set")
    if receipt.get("version") != receipt.get("expected_version") or \
            receipt.get("binary_sha256") != receipt.get("expected_binary_sha256"):
        raise ValueError("Search receipt does not identify the expected MMseqs2 binary")
    pinned = receipt.get("pinned_binary") is True and is_pinned_binary(receipt.get("binary_sha256"),
                                                                       receipt.get("version"))
    if (require_pinned or query_set == SET_COMPLEMENT) and not pinned:
        raise ValueError(f"Search receipt for {query_set} comes from an unpinned MMseqs2 binary; only the pinned "
                         f"{MMSEQS_RELEASE} build counts (engineering checks pass --allow-unpinned-binary)")
    require_same_freeze1(receipt.get("freeze1_sha256"), inputs, f"fasta_receipt_{query_set}.json")
    query_fasta = work / inputs["outputs"]["query_fasta"]["file"]
    target_fasta = work / inputs["outputs"]["target_fasta"]["file"]
    for key, path in (("query_fasta", query_fasta), ("target_fasta", target_fasta)):
        if receipt[key]["sha256"] != sha256_file(path):
            raise ValueError(f"Search {key} changed after the search")
    check_search_command(receipt["command"], query_fasta, target_fasta, work / receipt["raw_output"]["file"])
    if sha256_file(work / receipt["sorted_output"]["file"]) != receipt["sorted_output"]["sha256"]:
        raise ValueError("Sorted search output changed after the search")
    return receipt


# ---------------------------------------------------------------- recount and hit flags

@dataclass(frozen=True)
class Hit:
    query: str
    target: str
    evalue_text: str
    bits_text: str
    alnlen: int
    nident: int
    residue_pairs: int
    query_span: int
    qlen: int
    target_span: int
    tlen: int
    ambiguous_identical_columns: int = 0

    @property
    def evalue(self) -> float:
        return float(self.evalue_text)

    @property
    def bits(self) -> float:
        return float(self.bits_text)

    @property
    def identity(self) -> Fraction:
        return Fraction(self.nident, self.alnlen)

    @property
    def query_coverage(self) -> Fraction:
        return Fraction(self.query_span, self.qlen)

    @property
    def target_coverage(self) -> Fraction:
        return Fraction(self.target_span, self.tlen)

    @property
    def shorter_coverage(self) -> Fraction:
        # Same rule as alignment_metrics: the shorter sequence, or the smaller span at equal length.
        if self.qlen < self.tlen:
            return self.query_coverage
        if self.tlen < self.qlen:
            return self.target_coverage
        return min(self.query_coverage, self.target_coverage)

    @property
    def any_hit(self) -> bool:
        return self.evalue <= MAX_EVALUE

    @property
    def bin_hit(self) -> bool:
        return self.any_hit and self.residue_pairs >= MIN_RESIDUE_PAIRS

    @property
    def d3(self) -> bool:
        return self.bin_hit and self.identity >= IDENTITY_THRESHOLD

    @property
    def d2(self) -> bool:
        return self.any_hit and self.identity >= IDENTITY_THRESHOLD and self.shorter_coverage >= D2_SHORTER_COVERAGE

    @property
    def d2_min_pairs(self) -> bool:
        return self.d2 and self.residue_pairs >= MIN_RESIDUE_PAIRS

    @property
    def d1(self) -> bool:
        return (self.any_hit and self.identity >= IDENTITY_THRESHOLD
                and min(self.query_coverage, self.target_coverage) >= D1_BOTH_COVERAGE)

    def flags(self) -> dict[str, bool]:
        return {name: getattr(self, name) for name in FLAG_NAMES}

    def to_row(self) -> dict[str, str]:
        row = {
            "query": self.query, "target": self.target, "evalue": self.evalue_text, "bits": self.bits_text,
            "alnlen": str(self.alnlen), "nident": str(self.nident), "residue_pairs": str(self.residue_pairs),
            "identity": _fraction_text(self.identity), "query_span": str(self.query_span), "qlen": str(self.qlen),
            "target_span": str(self.target_span), "tlen": str(self.tlen),
            "shorter_coverage": _fraction_text(self.shorter_coverage),
            "ambiguous_identical_columns": str(self.ambiguous_identical_columns),
        }
        row.update({name: _flag(value) for name, value in self.flags().items()})
        return row

    @classmethod
    def from_row(cls, row: Mapping[str, str]) -> "Hit":
        hit = cls(query=row["query"], target=row["target"], evalue_text=row["evalue"], bits_text=row["bits"],
                  **{name: int(row[name]) for name in ("alnlen", "nident", "residue_pairs", "query_span", "qlen",
                                                        "target_span", "tlen", "ambiguous_identical_columns")})
        if any(_parse_flag(row[name]) != value for name, value in hit.flags().items()):
            raise ValueError(f"Stored flags disagree with the recount for {hit.query} -> {hit.target}")
        return hit


def recount_alignment(row: Mapping[str, str], sequences: Mapping[str, str]) -> Hit:
    """Exact recount of one MMseqs2 row against the bound input sequences."""
    query, target = row["query"], row["target"]
    if query not in sequences or target not in sequences:
        raise ValueError(f"Alignment names a sequence outside the search inputs: {query} -> {target}")
    qaln, taln = row["qaln"].upper(), row["taln"].upper()
    alnlen = int(row["alnlen"])
    if len(qaln) != alnlen or len(taln) != alnlen or alnlen == 0:
        raise ValueError(f"Alignment columns disagree with alnlen for {query} -> {target}")
    if any(a == b == "-" for a, b in zip(qaln, taln)):
        raise ValueError(f"Alignment contains double-gap columns for {query} -> {target}")
    pairs = sum(a != "-" and b != "-" for a, b in zip(qaln, taln))
    identical = sum(a == b and a != "-" for a, b in zip(qaln, taln))
    ambiguous = sum(a == b and a != "-" and a not in STANDARD_RESIDUES for a, b in zip(qaln, taln))
    nident = identical - ambiguous
    exported = int(row["nident"])
    # The pinned converter rebuilds nident from a rounded seqId and may lose one; it may or
    # may not count identical non-standard residues, so either convention is accepted.
    if identical - exported not in (0, 1) and nident - exported not in (0, 1):
        raise ValueError(f"nident disagrees with the aligned count for {query} -> {target}")
    spans = {}
    for prefix, alignment, sequence in (("q", qaln, sequences[query]), ("t", taln, sequences[target])):
        length, start, end = (int(row[prefix + suffix]) for suffix in ("len", "start", "end"))
        ungapped = alignment.replace("-", "")
        if length != len(sequence) or not 1 <= start <= end <= length:
            raise ValueError(f"Alignment coordinates or sequence length invalid for {query} -> {target}")
        if sequence[start - 1:end] != ungapped:
            raise ValueError(f"Alignment does not match the input sequence for {query} -> {target}")
        spans[prefix] = (len(ungapped), length)
    evalue, bits = float(row["evalue"]), float(row["bits"])
    if not math.isfinite(evalue) or evalue < 0 or not math.isfinite(bits):
        raise ValueError(f"Invalid E-value or bit score for {query} -> {target}")
    hit = Hit(query=query, target=target, evalue_text=row["evalue"], bits_text=row["bits"], alnlen=alnlen,
              nident=nident, residue_pairs=pairs, query_span=spans["q"][0], qlen=spans["q"][1],
              target_span=spans["t"][0], tlen=spans["t"][1], ambiguous_identical_columns=ambiguous)
    if not ambiguous:
        reference = remoteness.alignment_metrics(row, {query: {"sequence": sequences[query]},
                                            target: {"sequence": sequences[target]}})
        if (reference["identity"] != nident / alnlen or reference["residue_pairs"] != pairs
                or reference["shorter_coverage"] != float(hit.shorter_coverage)):
            raise ValueError(f"Recount disagrees with alignment_metrics for {query} -> {target}")
    return hit


def read_hits(path: Path, sequences: Mapping[str, str]) -> tuple[list[Hit], int]:
    """Recounted hits from a sorted search output, without self-hits; returns (hits, self-hit count)."""
    hits: list[Hit] = []
    seen: set[tuple[str, str]] = set()
    self_hits = 0
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != SEARCH_FIELDS:
            raise ValueError(f"Expected MMseqs2 --format-mode 4 columns: {SEARCH_FIELDS}")
        for row in reader:
            if row["query"] == row["target"]:
                self_hits += 1
                continue
            key = (row["query"], row["target"])
            if key in seen:
                raise ValueError(f"Several alignments for {key}; the pinned search reports one per pair")
            seen.add(key)
            hits.append(recount_alignment(row, sequences))
    return hits, self_hits


def _hit_sort_key(hit: Hit) -> tuple[str, str]:
    return hit.query, hit.target


def undirected_pairs(hits: Iterable[Hit], nodes: set[str]) -> list[dict[str, str]]:
    """Node-node pairs with each flag set when either direction sets it."""
    grouped: dict[tuple[str, str], list[Hit]] = defaultdict(list)
    for hit in hits:
        if hit.query in nodes and hit.target in nodes:
            grouped[tuple(sorted((hit.query, hit.target)))].append(hit)
    rows = []
    for (a, b), group in sorted(grouped.items()):
        row = {"a": a, "b": b, "n_directions": str(len(group))}
        row.update({name: _flag(any(getattr(hit, name) for hit in group)) for name in FLAG_NAMES})
        bin_identities = [hit.identity for hit in group if hit.bin_hit]
        row["max_bin_identity"] = _fraction_text(max(bin_identities)) if bin_identities else ""
        row["min_evalue"] = min(group, key=lambda hit: (hit.evalue, hit.evalue_text)).evalue_text
        rows.append(row)
    return rows


def build_edges(work: Path, fold: int, *, query_set: str = SET_METALLO_TRAIN,
                freeze1_sha256: Optional[str] = None, allow_unpinned_binary: bool = False) -> dict[str, Any]:
    """Recount the sorted search output into flagged hit tables (no labels)."""
    freeze1_sha256 = _check_query_set(query_set, freeze1_sha256)
    _check_unpinned_allowed(query_set, allow_unpinned_binary)
    work = Path(work)
    inputs = _load_fasta_receipt(work, fold, query_set)
    require_same_freeze1(freeze1_sha256, inputs, f"fasta_receipt_{query_set}.json")
    search = verify_search_receipt(work, fold, query_set, require_pinned=not allow_unpinned_binary)
    require_same_freeze1(freeze1_sha256, search, f"search_{query_set}/search_receipt.json")
    queries = read_fasta(work / inputs["outputs"]["query_fasta"]["file"])
    targets = read_fasta(work / inputs["outputs"]["target_fasta"]["file"])
    for accession in set(queries) & set(targets):
        if queries[accession] != targets[accession]:
            raise ValueError(f"Query and target sequences differ for {accession}")
    hits, self_hits = read_hits(work / search["sorted_output"]["file"], {**targets, **queries})
    for hit in hits:
        if hit.query not in queries or hit.target not in targets:
            raise ValueError(f"Hit outside the bound query or target set: {hit.query} -> {hit.target}")
    hits.sort(key=_hit_sort_key)
    tables = {f"edges_{query_set}.tsv": (EDGE_COLUMNS, [hit.to_row() for hit in hits])}
    if query_set == SET_METALLO_TRAIN:
        nodes = set(queries)
        tables["inner_pairs.tsv"] = (PAIR_COLUMNS, undirected_pairs(hits, nodes))
        # The R1-inner filter needs metallo -> non-metallo D3 hits (kept for every fold).
        tables["nonmetallo_d3_hits.tsv"] = (EDGE_COLUMNS, [hit.to_row() for hit in hits
                                                           if hit.d3 and hit.target not in nodes])
    outputs = {name: {"sha256": write_once(work / name, _tsv_text(columns, rows)), "n_rows": len(rows)}
               for name, (columns, rows) in tables.items()}
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "step": "edges",
        "fold": fold,
        "query_set": query_set,
        "freeze1_sha256": freeze1_sha256,
        "freeze1_verification": "recorded_not_verified" if freeze1_sha256 else None,
        "search": {"sorted_output_sha256": search["sorted_output"]["sha256"],
                   "binary_sha256": search["binary_sha256"], "version": search["version"],
                   "pinned_binary": search["pinned_binary"],
                   "settings": search["settings"]},
        "allow_unpinned_binary": allow_unpinned_binary,
        "rules": rules_record(),
        "counts": {"hits": len(hits), "self_hits_dropped": self_hits,
                   **{name: sum(getattr(hit, name) for hit in hits) for name in FLAG_NAMES},
                   "hits_with_ambiguous_identical_columns": sum(bool(hit.ambiguous_identical_columns)
                                                                for hit in hits)},
        "outputs": outputs,
        "code_sha256": code_sha256(),
        "labels_read": False,
    }
    write_once(work / f"edges_receipt_{query_set}.json", _json_text(receipt))
    return {"step": "edges", "fold": fold, "query_set": query_set, "counts": receipt["counts"], "outputs": outputs}


def load_edges_receipt(work: Path, fold: int, query_set: str, search: Mapping[str, Any]) -> dict[str, Any]:
    """Edges receipt of this fold and set, built from ``search``'s output, with unchanged tables."""
    work = Path(work)
    path = work / f"edges_receipt_{query_set}.json"
    if not path.is_file():
        raise ValueError(f"Run the edges step for {query_set} first: {path} is missing")
    receipt = json.loads(path.read_text())
    if receipt.get("fold") != fold or receipt.get("query_set") != query_set:
        raise ValueError(f"{path.name} belongs to fold {receipt.get('fold')}, set {receipt.get('query_set')}")
    if receipt["search"]["sorted_output_sha256"] != search["sorted_output"]["sha256"]:
        raise ValueError(f"{path.name} was built from another search output")
    for name, output in sorted(receipt["outputs"].items()):
        if sha256_file(work / name) != output["sha256"]:
            raise ValueError(f"{name} of fold {fold} changed after the edges step")
    return receipt


def rules_record() -> dict[str, Any]:
    return {
        "identity": IDENTITY_RULE,
        "identity_reference": "audit_sequence_remoteness.alignment_metrics on rows without identical "
                              "non-standard residues",
        "ambiguous_residue_policy": AMBIGUOUS_RESIDUE_POLICY,
        "max_evalue_inclusive": MAX_EVALUE,
        "min_residue_pairs": MIN_RESIDUE_PAIRS,
        "identity_threshold_inclusive": str(IDENTITY_THRESHOLD),
        "d2_shorter_coverage_inclusive": str(D2_SHORTER_COVERAGE),
        "d1_both_coverage_inclusive": str(D1_BOTH_COVERAGE),
        "self_hits": "dropped",
        "comparisons": "exact rational arithmetic on integer counts",
    }


# ---------------------------------------------------------------- components

@dataclass(frozen=True, order=True)
class Link:
    link_type: str
    a: str
    b: str


def group_links(link_type: str, groups: Mapping[str, Iterable[str]]) -> list[Link]:
    """Star links from each group's smallest member; connectivity equals all pairs."""
    links = []
    for members in groups.values():
        ordered = sorted(set(members))
        links.extend(Link(link_type, ordered[0], other) for other in ordered[1:])
    return links


def node_links(nodes: Sequence[Mapping[str, str]], link_types: Sequence[str]) -> list[Link]:
    """Exact-sequence, alias (accession case variants) and shared-donor links between nodes."""
    groups: dict[str, dict[str, list[str]]] = {name: defaultdict(list) for name in
                                               ("exact_sequence", "alias", "donor")}
    for node in nodes:
        groups["exact_sequence"][node["sequence_sha256"]].append(node["accession"])
        groups["alias"][node["accession"].upper()].append(node["accession"])
        if node["donor_pdb_id"]:
            groups["donor"][node["donor_pdb_id"]].append(node["accession"])
    return [link for name in groups if name in link_types for link in group_links(name, groups[name])]


def component_id(members: Iterable[str]) -> str:
    return "cmp_" + sha256_text(";".join(sorted(members)))[:16]


def build_components(nodes: Iterable[str], links: Iterable[Link]) -> dict[str, str]:
    """Accession -> component ID over the union of all links (union-find)."""
    nodes = sorted(set(nodes))
    forest = remoteness.Components(nodes)
    for link in links:
        if link.a not in forest.parents or link.b not in forest.parents:
            raise ValueError(f"{link.link_type} link names a protein outside the nodes: {link.a} - {link.b}")
        forest.join(link.a, link.b)
    members: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        members[forest.root(node)].append(node)
    identifiers = {root: component_id(group) for root, group in members.items()}
    return {node: identifiers[forest.root(node)] for node in nodes}


def component_summary(components: Mapping[str, str], links: Sequence[Link]) -> dict[str, Any]:
    sizes = Counter(components.values())
    largest = max(sizes.values(), default=0)
    n_nodes = len(components)
    return {
        "n_nodes": n_nodes,
        "n_components": len(sizes),
        "component_size_histogram": _histogram(sizes.values()),
        "largest_component_size": largest,
        "largest_component_fraction": round(largest / n_nodes, 6) if n_nodes else 0.0,
        "largest_component_exceeds_quarter": bool(n_nodes) and 4 * largest > n_nodes,
        "links_by_type": dict(sorted(Counter(link.link_type for link in links).items())),
    }


def _write_components(out: Path, scope: str, components: Mapping[str, str], links: Sequence[Link],
                      receipt: dict[str, Any]) -> dict[str, Any]:
    sizes = Counter(components.values())
    rows = [{"accession": node, "component_id": components[node], "component_size": sizes[components[node]]}
            for node in sorted(components)]
    link_rows = [{"link_type": link.link_type, "a": link.a, "b": link.b} for link in sorted(set(links))]
    tables = {f"components_{scope}.tsv": (COMPONENT_COLUMNS, rows),
              f"component_links_{scope}.tsv": (LINK_COLUMNS, link_rows)}
    receipt["outputs"] = {name: {"sha256": write_once(out / name, _tsv_text(columns, table_rows)),
                                 "n_rows": len(table_rows)}
                          for name, (columns, table_rows) in tables.items()}
    receipt["summary"] = component_summary(components, sorted(set(links)))
    receipt["code_sha256"] = code_sha256()
    receipt["labels_read"] = False
    write_once(out / f"components_receipt_{scope}.json", _json_text(receipt))
    return {"step": "components", "scope": scope, "summary": receipt["summary"], "outputs": receipt["outputs"]}


def build_inner_components(work: Path, fold: int, *, allow_unpinned_binary: bool = False) -> dict[str, Any]:
    """Fold-k metallo training components: D3 either direction, exact sequence, alias and donor links."""
    work = Path(work)
    inputs = _load_fasta_receipt(work, fold, SET_METALLO_TRAIN)
    search = verify_search_receipt(work, fold, SET_METALLO_TRAIN, require_pinned=not allow_unpinned_binary)
    edges_receipt = load_edges_receipt(work, fold, SET_METALLO_TRAIN, search)
    pairs_path = work / "inner_pairs.tsv"
    nodes = read_tsv(work / inputs["outputs"]["nodes"]["file"], NODE_COLUMNS)
    links = [Link("d3", row["a"], row["b"]) for row in read_tsv(pairs_path, PAIR_COLUMNS)
             if _parse_flag(row["d3"])]
    links += node_links(nodes, INNER_LINK_TYPES)
    components = build_components((node["accession"] for node in nodes), links)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "step": "components",
        "scope": SCOPE_INNER,
        "fold": fold,
        "link_types": list(INNER_LINK_TYPES),
        "node_policy": "every fold-k metallo training protein is a node, including proteins later "
                       "excluded from training for several EC classes (they can bridge components)",
        "inputs": {"nodes_sha256": inputs["outputs"]["nodes"]["sha256"],
                   "inner_pairs_sha256": edges_receipt["outputs"]["inner_pairs.tsv"]["sha256"],
                   "search_sorted_output_sha256": search["sorted_output"]["sha256"]},
        "pinned_binary": search["pinned_binary"],
        "allow_unpinned_binary": allow_unpinned_binary,
    }
    return _write_components(work, SCOPE_INNER, components, links, receipt)


def build_cohort_components(fold_works: Sequence[Path], out: Path, *, freeze1_sha256: Optional[str]) -> dict[str, Any]:
    """Bootstrap clusters over every cohort protein: any hit (E <= 1e-3, any length) or shared donor.

    Every pair of cohort proteins shares at least one fold's training file, so the union of
    the per-fold metallo searches, in both directions, covers every pair. Each fold's search
    must come from the pinned binary.
    """
    freeze1_sha256 = require_freeze1(freeze1_sha256, "Cohort clusters")
    nodes: dict[str, dict[str, str]] = {}
    links: list[Link] = []
    sources = {}
    folds_seen = set()
    hits_by_fold = []
    for work in map(Path, fold_works):
        receipt_path = work / f"fasta_receipt_{SET_METALLO_TRAIN}.json"
        if not receipt_path.is_file():
            raise ValueError(f"Run the fasta step for {SET_METALLO_TRAIN} first: {receipt_path} is missing")
        fold = json.loads(receipt_path.read_text())["fold"]
        if fold in folds_seen:
            raise ValueError(f"Fold {fold} given twice")
        folds_seen.add(fold)
        inputs = _load_fasta_receipt(work, fold, SET_METALLO_TRAIN)
        search = verify_search_receipt(work, fold, SET_METALLO_TRAIN, require_pinned=True)
        edges_receipt = load_edges_receipt(work, fold, SET_METALLO_TRAIN, search)
        edges_name = f"edges_{SET_METALLO_TRAIN}.tsv"
        for node in read_tsv(work / inputs["outputs"]["nodes"]["file"], NODE_COLUMNS):
            previous = nodes.setdefault(node["accession"], node)
            if any(previous[key] != node[key] for key in ("sequence_sha256", "donor_pdb_id")):
                raise ValueError(f"Sequence or donor of {node['accession']} differs between folds")
        hits_by_fold.append(read_tsv(work / edges_name, EDGE_COLUMNS))
        sources[str(fold)] = {"nodes_sha256": inputs["outputs"]["nodes"]["sha256"],
                              "edges_sha256": edges_receipt["outputs"][edges_name]["sha256"],
                              "search_sorted_output_sha256": search["sorted_output"]["sha256"],
                              "binary_sha256": search["binary_sha256"]}
    if folds_seen != set(range(len(folds_seen))) or len(folds_seen) != N_FOLDS:
        raise ValueError(f"Cohort clusters need folds 0..{N_FOLDS - 1}, got {sorted(folds_seen)}")
    for rows in hits_by_fold:
        for row in rows:
            hit = Hit.from_row(row)
            if hit.any_hit and hit.query in nodes and hit.target in nodes:
                a, b = sorted((hit.query, hit.target))
                links.append(Link("any_hit", a, b))
    links += node_links(list(nodes.values()), COHORT_LINK_TYPES)
    components = build_components(nodes, links)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "step": "components",
        "scope": SCOPE_COHORT,
        "freeze1_sha256": freeze1_sha256,
        "freeze1_verification": "recorded_not_verified",
        "link_types": list(COHORT_LINK_TYPES),
        "inputs": dict(sorted(sources.items())),
    }
    return _write_components(Path(out), SCOPE_COHORT, components, links, receipt)


# ---------------------------------------------------------------- strata

def identity_bin(identity: Optional[Fraction]) -> str:
    if identity is None:
        return "no_hit"
    if not 0 <= identity <= 1:
        raise ValueError(f"Identity {identity} outside [0, 1]")
    if identity <= Fraction(1, 5):
        return "<=20"
    if identity <= Fraction(1, 4):
        return "(20,25]"
    if identity < IDENTITY_THRESHOLD:
        return "(25,30)"
    if identity <= Fraction(1, 2):
        return "[30,50]"
    return ">50"


def strata_rows(queries: Sequence[Mapping[str, str]], hits: Iterable[Hit], training_donors: set[str],
                cluster_ids: Optional[Mapping[str, str]] = None) -> list[dict[str, str]]:
    """One row per query protein: identity stratum, edge flags, donor sharing and nearest hit."""
    by_query: dict[str, list[Hit]] = defaultdict(list)
    for hit in hits:
        by_query[hit.query].append(hit)
    rows = []
    for node in sorted(queries, key=lambda item: item["accession"]):
        accession = node["accession"]
        group = by_query.get(accession, [])
        bin_hits = [hit for hit in group if hit.bin_hit]
        best = min(bin_hits, key=lambda hit: (-hit.identity, hit.evalue, hit.target), default=None)
        any_hits = [hit for hit in group if hit.any_hit]
        nearest = min(any_hits, key=lambda hit: (-hit.bits, hit.evalue, hit.target), default=None)
        flags = {name: any(getattr(hit, name) for hit in group) for name in FLAG_NAMES}
        donor_shared = bool(node["donor_pdb_id"]) and node["donor_pdb_id"] in training_donors
        rows.append({
            "accession": accession,
            "identity_bin": identity_bin(best.identity if best else None),
            "max_bin_identity": _fraction_text(best.identity) if best else "",
            "max_bin_target": best.target if best else "",
            **{f"has_{name}": _flag(value) for name, value in flags.items()},
            "donor_shared_with_metallo_train": _flag(donor_shared),
            "no_d3_hit": _flag(not flags["d3"]),
            "no_d3_hit_no_shared_donor": _flag(not flags["d3"] and not donor_shared),
            "nearest_target": nearest.target if nearest else "",
            "nearest_bits": nearest.bits_text if nearest else "",
            "nearest_evalue": nearest.evalue_text if nearest else "",
            "cluster_id": (cluster_ids or {}).get(accession, ""),
        })
    return rows


def _load_cohort_clusters(clusters: Path, freeze1_sha256: str, fold: int,
                          metallo_nodes_sha256: str) -> dict[str, str]:
    """Cluster IDs from a receipted ``components_cohort.tsv`` of the same Freeze 1 and fold node table."""
    clusters = Path(clusters)
    expected_name = f"components_{SCOPE_COHORT}.tsv"
    receipt_path = clusters.parent / f"components_receipt_{SCOPE_COHORT}.json"
    if clusters.name != expected_name or not receipt_path.is_file():
        raise ValueError(f"--clusters must be a {expected_name} next to its {receipt_path.name}")
    receipt = json.loads(receipt_path.read_text())
    require_same_freeze1(freeze1_sha256, receipt, receipt_path.name)
    if receipt["outputs"][expected_name]["sha256"] != sha256_file(clusters):
        raise ValueError(f"{expected_name} changed after the cohort components step")
    if receipt["inputs"].get(str(fold), {}).get("nodes_sha256") != metallo_nodes_sha256:
        raise ValueError(f"Cohort clusters were built from another fold-{fold} metallo node table")
    return {row["accession"]: row["component_id"] for row in read_tsv(clusters, COMPONENT_COLUMNS)}


def build_strata(work: Path, fold: int, *, freeze1_sha256: Optional[str],
                 clusters: Optional[Path] = None) -> dict[str, Any]:
    """Complement-stratum table for fold k (Freeze 1 only; no curated test membership is used)."""
    freeze1_sha256 = require_freeze1(freeze1_sha256, "The strata table")
    work = Path(work)
    complement = _load_fasta_receipt(work, fold, SET_COMPLEMENT)
    require_same_freeze1(freeze1_sha256, complement, f"fasta_receipt_{SET_COMPLEMENT}.json")
    metallo = _load_fasta_receipt(work, fold, SET_METALLO_TRAIN)
    search = verify_search_receipt(work, fold, SET_COMPLEMENT, require_pinned=True)
    require_same_freeze1(freeze1_sha256, search, f"search_{SET_COMPLEMENT}/search_receipt.json")
    edges_receipt = load_edges_receipt(work, fold, SET_COMPLEMENT, search)
    require_same_freeze1(freeze1_sha256, edges_receipt, f"edges_receipt_{SET_COMPLEMENT}.json")
    edges_name = f"edges_{SET_COMPLEMENT}.tsv"
    queries = read_tsv(work / complement["outputs"]["nodes"]["file"], NODE_COLUMNS)
    training_donors = {node["donor_pdb_id"] for node in read_tsv(work / metallo["outputs"]["nodes"]["file"],
                                                                 NODE_COLUMNS) if node["donor_pdb_id"]}
    hits = [Hit.from_row(row) for row in read_tsv(work / edges_name, EDGE_COLUMNS)]
    cluster_ids = None
    if clusters is not None:
        cluster_ids = _load_cohort_clusters(clusters, freeze1_sha256, fold, metallo["outputs"]["nodes"]["sha256"])
    rows = strata_rows(queries, hits, training_donors, cluster_ids)
    name = f"strata_fold{fold}.tsv"
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "step": "strata",
        "fold": fold,
        "freeze1_sha256": freeze1_sha256,
        "freeze1_verification": "recorded_not_verified",
        "identity_bins": list(IDENTITY_BIN_LABELS),
        "identity_bin_status": IDENTITY_BIN_STATUS,
        "nearest_hit_order": NEAREST_HIT_ORDER,
        "rules": rules_record(),
        "complement_scope": complement.get("complement_scope"),
        "inputs": {"complement_nodes_sha256": complement["outputs"]["nodes"]["sha256"],
                   "metallo_train_nodes_sha256": metallo["outputs"]["nodes"]["sha256"],
                   "complement_search_sorted_output_sha256": search["sorted_output"]["sha256"],
                   "complement_edges_sha256": edges_receipt["outputs"][edges_name]["sha256"],
                   "clusters_sha256": sha256_file(clusters) if clusters is not None else None},
        "counts": {"rows": len(rows), **{f"identity_bin_{label}": sum(row["identity_bin"] == label for row in rows)
                                         for label in IDENTITY_BIN_LABELS}},
        "outputs": {name: {"sha256": write_once(work / name, _tsv_text(STRATA_COLUMNS, rows)), "n_rows": len(rows)}},
        "code_sha256": code_sha256(),
        "labels_read": False,
        "curated_test_membership_read": False,
    }
    write_once(work / "strata_receipt.json", _json_text(receipt))
    return {"step": "strata", "fold": fold, "counts": receipt["counts"], "outputs": receipt["outputs"]}


# ---------------------------------------------------------------- inner folds

@dataclass(frozen=True)
class InnerChain:
    """Verified training-side receipts of one fold: fasta, search and edges of the metallo training set."""
    inputs: dict[str, Any]
    search: dict[str, Any]
    edges: dict[str, Any]
    freeze1_sha256: Optional[str]

    def record(self) -> dict[str, str]:
        return {"nodes_sha256": self.inputs["outputs"]["nodes"]["sha256"],
                "search_sorted_output_sha256": self.search["sorted_output"]["sha256"],
                "edges_sha256": self.edges["outputs"][f"edges_{SET_METALLO_TRAIN}.tsv"]["sha256"],
                "inner_pairs_sha256": self.edges["outputs"]["inner_pairs.tsv"]["sha256"]}


def load_inner_chain(work: Path, fold: int, *, freeze1_sha256: Optional[str] = None,
                     allow_unpinned_binary: bool = False) -> InnerChain:
    """The metallo training chain the inner steps build on; a Freeze 1 value is optional but must match it."""
    freeze1_sha256 = require_freeze1(freeze1_sha256, "The inner steps") if freeze1_sha256 else None
    work = Path(work)
    inputs = _load_fasta_receipt(work, fold, SET_METALLO_TRAIN)
    require_same_freeze1(freeze1_sha256, inputs, f"fasta_receipt_{SET_METALLO_TRAIN}.json")
    search = verify_search_receipt(work, fold, SET_METALLO_TRAIN, require_pinned=not allow_unpinned_binary)
    edges = load_edges_receipt(work, fold, SET_METALLO_TRAIN, search)
    return InnerChain(inputs, search, edges, freeze1_sha256)


def _load_inner_components(work: Path, fold: int, chain: InnerChain) -> tuple[dict[str, str], str]:
    """Accession -> inner component ID from an unchanged components step over ``chain``, and its receipt SHA-256."""
    path = work / f"components_receipt_{SCOPE_INNER}.json"
    if not path.is_file():
        raise ValueError(f"Run the components step (inner scope) first: {path} is missing")
    receipt = json.loads(path.read_text())
    if receipt.get("fold") != fold or receipt.get("scope") != SCOPE_INNER:
        raise ValueError(f"{path.name} belongs to fold {receipt.get('fold')}, scope {receipt.get('scope')}")
    record = chain.record()
    built_from = {key: record[key] for key in ("nodes_sha256", "inner_pairs_sha256", "search_sorted_output_sha256")}
    if receipt.get("inputs") != built_from:
        raise ValueError(f"{path.name} was built from other node, pair or search tables")
    for name, output in sorted(receipt["outputs"].items()):
        if sha256_file(work / name) != output["sha256"]:
            raise ValueError(f"{name} of fold {fold} changed after the components step")
    rows = read_tsv(work / f"components_{SCOPE_INNER}.tsv", COMPONENT_COLUMNS)
    return {row["accession"]: row["component_id"] for row in rows}, sha256_file(path)


def _metallo_training_entries(chain: InnerChain, fold: int, metallo_folds_dir: Path,
                              metallo_manifest_root: Optional[Path] = None,
                              ) -> tuple[dict[str, str], dict[str, MetalloEntry]]:
    """Fold k's metallo training file and structure manifest by exact name, unchanged since the fasta step."""
    path = training_file(metallo_folds_dir, "metallo", fold)
    recorded = chain.inputs["sources"]["metallo_training_files"][str(fold)]["sha256"]
    actual = sha256_file(path)
    if actual != recorded:
        raise ValueError(f"{path.name} has SHA-256 {actual}, not the {recorded} the fasta step read")
    manifest = metallo_structure_manifest(metallo_manifest_root, fold) if metallo_manifest_root else None
    recorded_manifest = chain.inputs["sources"].get("metallo_structure_manifests", {}).get(str(fold))
    changed = ValueError(f"The fold-{fold} training structure manifest differs from the one the fasta step read")
    if recorded_manifest is not None and _metallo_manifest_record(manifest) != recorded_manifest:
        raise changed
    entries, used = read_metallo_training(path, manifest)
    if used != (recorded_manifest is not None):
        raise changed
    return {"path": str(path), "sha256": actual}, entries


def inner_eligible_proteins(entries: Mapping[str, MetalloEntry]) -> tuple[dict[str, tuple[str, str]], list[str]]:
    """Accession -> (structure ID, EC class) of the proteins that enter inner folds, and the excluded accessions.

    Same rule as the trainer's fixed seven-class vocabulary: the class comes from the structure
    stem, several EC classes are a counted exclusion, and unusable annotations stop the step.
    """
    several = sorted(accession for accession, entry in entries.items() if len(entry.structure_ids) != 1)
    if several:
        raise ValueError(f"{len(several)} metallo training protein(s) have several structure IDs, e.g. "
                         f"{several[:20]}; the EC membership holds one structure per protein")
    eligible: dict[str, tuple[str, str]] = {}
    excluded: list[str] = []
    fatal: list[str] = []
    for accession in sorted(entries):
        (structure_id,) = entries[accession].structure_ids
        result = classify_ec_class(ec_numbers_from_structure_id(structure_id))
        if result.is_fatal:
            fatal.append(f"{structure_id} ({result.status})")
        elif result.status == STATUS_AMBIGUOUS:
            excluded.append(accession)
        else:
            eligible[accession] = (structure_id, result.token)
    if fatal:
        raise ValueError(f"{len(fatal)} structure(s) have EC annotations that cannot give an EC class: "
                         + ", ".join(fatal[:20]))
    return eligible, excluded


def inner_d3_links(hits: Iterable[Hit], nodes: set[str]) -> list[Link]:
    """One D3 link per node pair with a D3 hit in either direction."""
    return sorted({Link("d3", *sorted((hit.query, hit.target))) for hit in hits
                   if hit.d3 and hit.query in nodes and hit.target in nodes})


def assign_components(class_counts: Mapping[str, Mapping[str, int]], *, seed: int,
                      n_folds: int = N_INNER_FOLDS) -> dict[str, int]:
    """Component ID -> inner fold by ``INNER_BALANCE_RULE``: one deterministic pass in exact arithmetic."""
    counts_by_component = {component: {token: n for token, n in counts.items() if n > 0}
                           for component, counts in class_counts.items()}
    empty = sorted(component for component, counts in counts_by_component.items() if not counts)
    if empty:
        raise ValueError(f"Components without eligible proteins cannot be assigned: {empty[:5]}")
    totals: Counter[str] = Counter()
    for counts in counts_by_component.values():
        totals.update(counts)

    def order(component: str) -> tuple[int, int, str]:
        counts = counts_by_component[component]
        return (min(totals[token] for token in counts), -sum(counts.values()),
                sha256_text(f"{seed}:{component}"))

    filled = [Counter() for _ in range(n_folds)]
    sizes = [0] * n_folds
    assignment: dict[str, int] = {}
    for component in sorted(counts_by_component, key=order):
        counts = counts_by_component[component]

        def deficit(fold: int) -> Fraction:
            return sum((Fraction(n, totals[token]) * (Fraction(totals[token], n_folds) - filled[fold][token])
                        for token, n in counts.items()), Fraction(0))

        best = min(range(n_folds), key=lambda fold: (-deficit(fold), sizes[fold], fold))
        assignment[component] = best
        filled[best].update(counts)
        sizes[best] += sum(counts.values())
    return assignment


def inner_fold_summary(rows: Sequence[Mapping[str, Any]], n_folds: int = N_INNER_FOLDS) -> dict[str, dict[str, Any]]:
    """Proteins, components and EC class counts per inner fold."""
    summary = {}
    for fold in range(n_folds):
        members = [row for row in rows if int(row["fold"]) == fold]
        summary[str(fold)] = {
            "n_proteins": len(members),
            "n_components": len({row["group_id"] for row in members}),
            "ec_class_counts": {token: sum(str(row.get("ec_class")) == token for row in members)
                                for token in EC_CLASS_TOKENS},
        }
    return summary


def _write_membership_once(rows: Sequence[Mapping[str, Any]], path: Path) -> str:
    """``write_ec_fold_membership`` under this module's write-once rule; returns the file SHA-256."""
    path = Path(path)
    if not path.exists():
        return write_ec_fold_membership(rows, path)
    with tempfile.TemporaryDirectory() as scratch:
        expected = write_ec_fold_membership(rows, Path(scratch) / path.name)
    if sha256_file(path) != expected:
        raise ValueError(f"Refusing to replace an existing output with different content: {path}")
    return expected


def max_share_value(value: Any) -> Fraction:
    """A maximum component share in (0, 1]; a float is read by its decimal text, so 0.3 is 3/10."""
    try:
        share = Fraction(str(value)) if isinstance(value, float) else Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"The maximum component share must be a fraction such as 1/4 or 0.25, got {value!r}") from exc
    if not 0 < share <= 1:
        raise ValueError(f"The maximum component share must be in (0, 1], got {share}")
    return share


def _share_arg(text: str) -> Fraction:
    try:
        return max_share_value(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _refuse_assignment(work: Path, record: Mapping[str, Any], reason: str) -> None:
    path = work / INNER_ASSIGNMENT_REFUSAL
    assigned = work / INNER_ASSIGNMENT_RECEIPT
    if assigned.exists():
        raise ValueError(f"Inner assignment of fold {record['fold']} refused ({reason}), but {assigned} already "
                         f"holds an assignment; no refusal written: {RERUN_RULE}")
    refusal = dict(record, status="refused", reason=reason,
                   decision="no membership written; any fallback is a Freeze 1 decision of the maintainer")
    write_once(path, _json_text(refusal))
    raise ValueError(f"Inner assignment of fold {record['fold']} refused: {reason}; see {path}")


def assign_inner(
    work: Path,
    fold: int,
    *,
    seed: int = INNER_SPLIT_SEED,
    max_component_share: Fraction = DEFAULT_MAX_COMPONENT_SHARE,
    metallo_folds_dir: Path = DEFAULT_METALLO_FOLDS_DIR,
    metallo_manifest_root: Optional[Path] = None,
    freeze1_sha256: Optional[str] = None,
    allow_unpinned_binary: bool = False,
) -> dict[str, Any]:
    """Whole inner components into 4 inner folds, stratified by EC class; writes the EC fold membership.

    Training labels only (fold k's metallo training structure stems). A component above
    ``max_component_share`` of the eligible proteins, or an empty inner fold, is a refusal:
    a receipt is written and no membership.
    """
    max_component_share = max_share_value(max_component_share)
    work = Path(work)
    _refuse_after_refusal(work, fold)
    chain = load_inner_chain(work, fold, freeze1_sha256=freeze1_sha256, allow_unpinned_binary=allow_unpinned_binary)
    components, components_receipt_sha = _load_inner_components(work, fold, chain)
    metallo_file, entries = _metallo_training_entries(chain, fold, metallo_folds_dir, metallo_manifest_root)
    if set(entries) != set(components):
        raise ValueError(f"{Path(metallo_file['path']).name} and components_{SCOPE_INNER}.tsv name different proteins")
    eligible, excluded = inner_eligible_proteins(entries)
    members: dict[str, list[str]] = defaultdict(list)
    for accession in sorted(eligible):
        members[components[accession]].append(accession)
    n_eligible = len(eligible)
    largest = max((len(accessions) for accessions in members.values()), default=0)
    record = {
        "schema_version": SCHEMA_VERSION,
        "step": "assign_inner",
        "fold": fold,
        "n_inner_folds": N_INNER_FOLDS,
        "seed": seed,
        "freeze1_sha256": chain.freeze1_sha256,
        "freeze1_verification": "recorded_not_verified" if chain.freeze1_sha256 else None,
        "eligibility_rule": INNER_ELIGIBILITY_RULE,
        "balance_rule": INNER_BALANCE_RULE,
        "max_component_share": str(max_component_share),
        "max_share_rule": MAX_SHARE_RULE,
        "max_share_basis": MAX_SHARE_BASIS,
        "inputs": {**chain.record(), "components_receipt_sha256": components_receipt_sha,
                   "metallo_training_file": metallo_file},
        "pinned_binary": chain.search["pinned_binary"],
        "allow_unpinned_binary": allow_unpinned_binary,
        "counts": {
            "nodes": len(entries),
            "eligible_proteins": n_eligible,
            "excluded_ambiguous_ec_class": len(excluded),
            "components_with_eligible_proteins": len(members),
            "components_without_eligible_proteins": len(set(components.values()) - set(members)),
            "eligible_by_ec_class": {token: sum(item[1] == token for item in eligible.values())
                                     for token in EC_CLASS_TOKENS},
            "largest_component_eligible_size": largest,
            "largest_component_share": str(Fraction(largest, n_eligible)) if n_eligible else "0",
        },
        "excluded_ambiguous_ec_class_accessions": excluded,
        "code_sha256": code_sha256(),
        "labels_read": True,
        "label_source": "structure stems of fold k's metallo training file (training labels only)",
    }
    if not n_eligible:
        _refuse_assignment(work, record, "no eligible proteins")
    oversized = [{"component_id": component, "eligible_size": len(accessions),
                  "share": str(Fraction(len(accessions), n_eligible))}
                 for component, accessions in sorted(members.items(), key=lambda item: (-len(item[1]), item[0]))
                 if Fraction(len(accessions), n_eligible) > max_component_share]
    if oversized:
        _refuse_assignment(work, record | {"components_over_max_share": oversized},
                           f"{len(oversized)} component(s) hold more than {max_component_share} of the "
                           f"{n_eligible} eligible proteins")
    class_counts = {component: Counter(eligible[accession][1] for accession in accessions)
                    for component, accessions in members.items()}
    assignment = assign_components(class_counts, seed=seed)
    rows = [{"structure_id": eligible[accession][0], "accession": accession, "group_id": components[accession],
             "fold": str(assignment[components[accession]]), "ec_class": eligible[accession][1]}
            for accession in sorted(eligible)]
    summary = inner_fold_summary(rows)
    empty = [int(inner) for inner, item in summary.items() if not item["n_proteins"]]
    if empty:
        _refuse_assignment(work, record | {"inner_folds": summary}, f"inner fold(s) {empty} would be empty")
    name = INNER_MEMBERSHIP_NAME.format(fold=fold)
    membership_sha = _write_membership_once(rows, work / name)
    present = [token for token in EC_CLASS_TOKENS if record["counts"]["eligible_by_ec_class"][token]]
    missing = {token: [int(inner) for inner, item in summary.items() if not item["ec_class_counts"][token]]
               for token in present}
    receipt = record | {
        "status": "assigned",
        "inner_folds": summary,
        "inner_folds_without_ec_class": {token: folds for token, folds in missing.items() if folds},
        "outputs": {"membership": {"file": name, "sha256": membership_sha, "n_rows": len(rows)}},
    }
    write_once(work / INNER_ASSIGNMENT_RECEIPT, _json_text(receipt))
    return {"step": "assign_inner", "fold": fold, "counts": receipt["counts"], "inner_folds": summary,
            "outputs": receipt["outputs"]}


def _refuse_after_refusal(work: Path, fold: int) -> None:
    path = Path(work) / INNER_ASSIGNMENT_REFUSAL
    if path.exists():
        raise ValueError(f"An earlier inner assignment of fold {fold} was refused ({path}); {RERUN_RULE}")


def _membership_file(output: Mapping[str, Any], fold: int, source: str) -> str:
    name = INNER_MEMBERSHIP_NAME.format(fold=fold)
    if output.get("file") != name:
        raise ValueError(f"{source} names the membership file {output.get('file')!r}, not {name}")
    return name


def _load_assignment_receipt(work: Path, fold: int) -> dict[str, Any]:
    path = Path(work) / INNER_ASSIGNMENT_RECEIPT
    if not path.is_file():
        raise ValueError(f"Run assign-inner first: {path} is missing")
    _refuse_after_refusal(work, fold)
    receipt = json.loads(path.read_text())
    if receipt.get("fold") != fold or receipt.get("status") != "assigned":
        raise ValueError(f"{path.name} is not an inner assignment of fold {fold}")
    _membership_file(receipt["outputs"]["membership"], fold, path.name)
    return receipt


def inner_certification_counts(rows: Sequence[Mapping[str, Any]], eligible: Mapping[str, tuple[str, str]],
                               components: Mapping[str, str], links: Sequence[Link],
                               n_folds: int = N_INNER_FOLDS) -> dict[str, Any]:
    """Coverage, ownership and cross-fold counts of a membership against re-derived components and links.

    ``components`` maps every node (excluded bridge proteins included) to its re-derived component.
    """
    accessions = [row["accession"] for row in rows]
    fold_of = {row["accession"]: int(row["fold"]) for row in rows}
    crossing = Counter(link.link_type for link in links
                       if link.a in fold_of and link.b in fold_of and fold_of[link.a] != fold_of[link.b])
    component_folds: dict[str, set[int]] = defaultdict(set)
    for accession, inner in fold_of.items():
        if accession in components:
            component_folds[components[accession]].add(inner)
    used = set(fold_of.values())
    return {
        "membership_rows": len(rows),
        "eligible_proteins": len(eligible),
        "repeated_accessions": len(accessions) - len(set(accessions)),
        "repeated_structure_ids": len(rows) - len({row["structure_id"] for row in rows}),
        "eligible_missing": len(set(eligible) - set(fold_of)),
        "unexpected_proteins": len(set(fold_of) - set(eligible)),
        "structure_id_mismatches": sum(row["accession"] in eligible
                                       and row["structure_id"] != eligible[row["accession"]][0] for row in rows),
        "ec_class_mismatches": sum(row["accession"] in eligible
                                   and str(row.get("ec_class")) != eligible[row["accession"]][1] for row in rows),
        "folds_outside_range": sum(not 0 <= inner < n_folds for inner in fold_of.values()),
        "empty_inner_folds": sum(inner not in used for inner in range(n_folds)),
        "group_id_mismatches": sum(row["group_id"] != components.get(row["accession"]) for row in rows),
        "split_components": sum(len(folds) > 1 for folds in component_folds.values()),
        "cross_fold_links": sum(crossing.values()),
        "cross_fold_links_by_type": {name: crossing.get(name, 0)
                                     for name in sorted(set(INNER_LINK_TYPES) | set(crossing))},
    }


def certify_inner(
    work: Path,
    fold: int,
    *,
    metallo_folds_dir: Path = DEFAULT_METALLO_FOLDS_DIR,
    metallo_manifest_root: Optional[Path] = None,
    freeze1_sha256: Optional[str] = None,
    allow_unpinned_binary: bool = False,
) -> dict[str, Any]:
    """Certify the inner membership against eligibility and links re-derived from the training-side tables.

    D3 links come from the directional edge table (flags re-checked against the recount), the
    exact-sequence, alias and donor links from the node table, and components are rebuilt
    with the excluded proteins as nodes. Writes ``inner_certification_receipt.json`` with
    ``certified`` true or false and every count, and raises when it is false.
    """
    work = Path(work)
    chain = load_inner_chain(work, fold, freeze1_sha256=freeze1_sha256, allow_unpinned_binary=allow_unpinned_binary)
    assignment = _load_assignment_receipt(work, fold)
    require_same_freeze1(chain.freeze1_sha256, assignment, INNER_ASSIGNMENT_RECEIPT)
    if {key: assignment["inputs"].get(key) for key in chain.record()} != chain.record():
        raise ValueError(f"{INNER_ASSIGNMENT_RECEIPT} was built from other identity tables")
    output = assignment["outputs"]["membership"]
    membership = read_ec_fold_membership(work / output["file"], output["sha256"])
    metallo_file, entries = _metallo_training_entries(chain, fold, metallo_folds_dir, metallo_manifest_root)
    eligible, excluded = inner_eligible_proteins(entries)
    nodes = read_tsv(work / chain.inputs["outputs"]["nodes"]["file"], NODE_COLUMNS)
    names = {node["accession"] for node in nodes}
    if names != set(entries):
        raise ValueError(f"{Path(metallo_file['path']).name} and the node table name different proteins")
    hits = [Hit.from_row(row) for row in read_tsv(work / f"edges_{SET_METALLO_TRAIN}.tsv", EDGE_COLUMNS)]
    links = inner_d3_links(hits, names) + node_links(nodes, INNER_LINK_TYPES)
    components = build_components(names, links)
    counts = inner_certification_counts(membership.rows, eligible, components, links)
    failures = [key for key in CERTIFICATION_FAILURE_KEYS if counts[key]]
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "step": "certify_inner",
        "fold": fold,
        "n_inner_folds": N_INNER_FOLDS,
        "certified": not failures,
        "failures": failures,
        "freeze1_sha256": chain.freeze1_sha256,
        "freeze1_verification": "recorded_not_verified" if chain.freeze1_sha256 else None,
        "membership": {"file": output["file"], "sha256": membership.file_sha256,
                       "identity_sha256": membership.identity_sha256},
        "assignment_receipt_sha256": sha256_file(work / INNER_ASSIGNMENT_RECEIPT),
        "inputs": {**chain.record(), "metallo_training_file": metallo_file},
        "pinned_binary": chain.search["pinned_binary"],
        "allow_unpinned_binary": allow_unpinned_binary,
        "link_types": list(INNER_LINK_TYPES),
        "checks": "exact coverage of the eligible proteins, one row per protein and structure, structure IDs and "
                  "EC classes as re-derived, group_id = re-derived component (excluded proteins are nodes), "
                  "no component in more than one inner fold, no qualifying link between inner folds",
        "counts": counts | {"excluded_ambiguous_ec_class": len(excluded)},
        "inner_folds": inner_fold_summary(membership.rows),
        "code_sha256": code_sha256(),
        "labels_read": True,
    }
    write_once(work / INNER_CERTIFICATE, _json_text(receipt))
    if failures:
        listed = ", ".join(f"{key}={counts[key]}" for key in failures)
        raise ValueError(f"Inner membership of fold {fold} is not certified: {listed}; see {work / INNER_CERTIFICATE}")
    return {"step": "certify_inner", "fold": fold, "certified": True, "counts": receipt["counts"],
            "membership": receipt["membership"]}


def _load_certified_membership(work: Path, fold: int, chain: InnerChain) -> tuple[Any, str]:
    """The certified inner membership of this chain, and the certificate's SHA-256."""
    path = work / INNER_CERTIFICATE
    if not path.is_file():
        raise ValueError(f"Run certify-inner first: {path} is missing")
    certificate = json.loads(path.read_text())
    if certificate.get("fold") != fold or certificate.get("certified") is not True:
        raise ValueError(f"The inner membership of fold {fold} is not certified ({path.name})")
    require_same_freeze1(chain.freeze1_sha256, certificate, path.name)
    if {key: certificate["inputs"].get(key) for key in chain.record()} != chain.record():
        raise ValueError(f"{path.name} was certified against other identity tables")
    assignment = _load_assignment_receipt(work, fold)
    if assignment["outputs"]["membership"]["sha256"] != certificate["membership"]["sha256"]:
        raise ValueError(f"{path.name} certifies another membership than {INNER_ASSIGNMENT_RECEIPT} names")
    if certificate.get("assignment_receipt_sha256") != sha256_file(work / INNER_ASSIGNMENT_RECEIPT):
        raise ValueError(f"{INNER_ASSIGNMENT_RECEIPT} changed after {path.name} was written")
    name = _membership_file(certificate["membership"], fold, path.name)
    membership = read_ec_fold_membership(work / name, certificate["membership"]["sha256"])
    return membership, sha256_file(path)


def d3_neighbours(proteins: set[str], hits: Iterable[Hit]) -> set[str]:
    """Proteins with a D3 hit to or from any of ``proteins`` (the proteins themselves not included)."""
    neighbours = set()
    for hit in hits:
        if hit.d3 and hit.query in proteins:
            neighbours.add(hit.target)
        if hit.d3 and hit.target in proteins:
            neighbours.add(hit.query)
    return neighbours - proteins


def build_r1_inner_exclusions(
    work: Path,
    fold: int,
    inner_fold: int,
    *,
    freeze1_sha256: Optional[str] = None,
    allow_unpinned_binary: bool = False,
) -> dict[str, Any]:
    """Proteins of fold k's full official training file that CLEAN R1-inner drops for one inner validation fold.

    The inner-validation proteins are fold-k metallo training proteins, so the pinned metallo
    training search already queried each of them against the full training file; no new
    search is run. The list holds accessions only.
    """
    if not 0 <= inner_fold < N_INNER_FOLDS:
        raise ValueError(f"Inner fold {inner_fold} outside 0..{N_INNER_FOLDS - 1}")
    work = Path(work)
    chain = load_inner_chain(work, fold, freeze1_sha256=freeze1_sha256, allow_unpinned_binary=allow_unpinned_binary)
    membership, certificate_sha = _load_certified_membership(work, fold, chain)
    validation = {row["accession"] for row in membership.rows if int(row["fold"]) == inner_fold}
    if not validation:
        raise ValueError(f"Inner fold {inner_fold} of fold {fold} has no proteins")
    queries = read_fasta(work / chain.inputs["outputs"]["query_fasta"]["file"])
    targets = read_fasta(work / chain.inputs["outputs"]["target_fasta"]["file"])
    outside = sorted(validation - set(queries))
    if outside:
        raise ValueError(f"Inner-validation proteins outside the fold-{fold} metallo training queries: {outside[:20]}")
    hits = [Hit.from_row(row) for row in read_tsv(work / f"edges_{SET_METALLO_TRAIN}.tsv", EDGE_COLUMNS)]
    neighbours = d3_neighbours(validation, hits)
    excluded = validation | neighbours
    unknown = sorted(excluded - set(targets))
    if unknown:
        raise ValueError(f"Excluded proteins outside fold {fold}'s official training file: {unknown[:20]}")
    rows = [{"accession": accession, "reason": R1_REASON_VALIDATION if accession in validation else R1_REASON_D3}
            for accession in sorted(excluded)]
    name = f"r1_inner_exclusions_fold{fold}_inner{inner_fold}.tsv"
    validation_sequences = {queries[accession] for accession in validation}
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "step": "r1_inner_exclusions",
        "fold": fold,
        "inner_fold": inner_fold,
        "freeze1_sha256": chain.freeze1_sha256,
        "freeze1_verification": "recorded_not_verified" if chain.freeze1_sha256 else None,
        "rule": R1_EXCLUSION_RULE,
        "inputs": {**chain.record(), "membership_sha256": membership.file_sha256,
                   "certificate_sha256": certificate_sha,
                   "target_fasta_sha256": chain.inputs["outputs"]["target_fasta"]["sha256"],
                   "official_training_file": chain.inputs["sources"]["official_training_files"][str(fold)]},
        "pinned_binary": chain.search["pinned_binary"],
        "allow_unpinned_binary": allow_unpinned_binary,
        "counts": {
            "full_training_proteins": len(targets),
            "inner_validation_proteins": len(validation),
            "d3_neighbours": len(neighbours),
            "d3_neighbours_in_metallo_training": len(neighbours & set(queries)),
            "d3_neighbours_outside_metallo_training": len(neighbours - set(queries)),
            "excluded": len(rows),
            "r1_inner_training_proteins": len(targets) - len(rows),
            "kept_proteins_identical_to_a_validation_sequence": sum(
                accession not in excluded and sequence in validation_sequences
                for accession, sequence in targets.items()),
        },
        "outputs": {name: {"sha256": write_once(work / name, _tsv_text(R1_EXCLUSION_COLUMNS, rows)),
                           "n_rows": len(rows)}},
        "code_sha256": code_sha256(),
        "labels_read": True,
        "labels_used": False,
        "label_note": "the membership carries EC classes; only its fold column is used",
    }
    write_once(work / f"r1_inner_exclusions_receipt_fold{fold}_inner{inner_fold}.json", _json_text(receipt))
    return {"step": "r1_inner_exclusions", "fold": fold, "inner_fold": inner_fold, "counts": receipt["counts"],
            "outputs": receipt["outputs"]}


# ---------------------------------------------------------------- command line

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(name: str, help_text: str, *, fold_required: bool = True) -> argparse.ArgumentParser:
        command = sub.add_parser(name, help=help_text, description=help_text)
        command.add_argument("--fold", type=int, required=fold_required)
        command.add_argument("--work", "--out", dest="work", type=Path, required=fold_required,
                             help="Fold working directory (all outputs and receipts).")
        command.add_argument("--freeze1-sha256", help="Required for complement queries, cohort clusters "
                                                      "and strata; recorded, verified later.")
        return command

    fasta = common("fasta", "Write the target FASTA, query FASTA and node table from training files.")
    fasta.add_argument("--query-set", choices=QUERY_SETS, default=SET_METALLO_TRAIN)
    fasta.add_argument("--official-split-dir", type=Path, default=DEFAULT_OFFICIAL_SPLIT_DIR)
    fasta.add_argument("--metallo-folds-dir", type=Path, default=DEFAULT_METALLO_FOLDS_DIR)
    fasta.add_argument("--metallo-manifest-root", type=Path, default=DEFAULT_METALLO_MANIFEST_ROOT,
                       help=f"Root of {METALLO_MANIFEST_PATH} (structure stems when the training CSV holds "
                            "accessions).")
    fasta.add_argument("--n-folds", type=int, default=N_FOLDS)

    search = common("search", "Run the pinned MMseqs2 search and write its receipt.")
    search.add_argument("--query-set", choices=QUERY_SETS, default=SET_METALLO_TRAIN)
    search.add_argument("--binary", type=Path, default=DEFAULT_MMSEQS_BINARY)
    search.add_argument("--binary-sha256", default=MMSEQS_BINARY_SHA256)
    search.add_argument("--expected-version", default=MMSEQS_VERSION)
    search.add_argument("--threads", type=int, default=2)
    search.add_argument("--split-memory-limit", default="2G")
    search.add_argument("--timeout-seconds", type=float, default=7200)

    edges = common("edges", "Recount the sorted search output into flagged hit tables.")
    edges.add_argument("--query-set", choices=QUERY_SETS, default=SET_METALLO_TRAIN)

    components = common("components", "Build components (inner: one fold; cohort: all folds, Freeze 1 only).",
                        fold_required=False)
    components.add_argument("--scope", choices=(SCOPE_INNER, SCOPE_COHORT), default=SCOPE_INNER)
    components.add_argument("--fold-work", type=Path, action="append", default=[],
                            help="Cohort scope: one working directory per fold.")

    strata = common("strata", "Complement-stratum table for one fold (Freeze 1 only).")
    strata.add_argument("--clusters", type=Path, help="components_cohort.tsv for cluster IDs.")

    assign = common("assign-inner", "Assign whole inner components to 4 inner folds, stratified by EC class, "
                                    "and write the EC fold membership.")
    assign.add_argument("--seed", type=int, default=INNER_SPLIT_SEED)
    assign.add_argument("--max-component-share", type=_share_arg, default=DEFAULT_MAX_COMPONENT_SHARE,
                        help="Refuse when one component holds more than this share of the eligible proteins "
                             "(a fraction such as 1/4 or 0.25; default 1/4).")
    certify = common("certify-inner", "Certify the inner membership: coverage, ownership, no cross-fold links.")
    for command in (assign, certify):
        command.add_argument("--metallo-folds-dir", type=Path, default=DEFAULT_METALLO_FOLDS_DIR)
        command.add_argument("--metallo-manifest-root", type=Path, default=DEFAULT_METALLO_MANIFEST_ROOT)
    r1 = common("r1-inner-exclusions", "List the full-training proteins CLEAN R1-inner drops for one inner fold.")
    r1.add_argument("--inner-fold", type=int, required=True)

    for command in (search, edges, components, assign, certify, r1):
        command.add_argument("--allow-unpinned-binary", action="store_true",
                             help="Engineering checks only: accept a search by another MMseqs2 binary for the "
                                  "metallo training set (recorded; refused for complement, cohort and strata).")
    return parser


def run_command(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "assign-inner":
        return assign_inner(args.work, args.fold, seed=args.seed, max_component_share=args.max_component_share,
                            metallo_folds_dir=args.metallo_folds_dir,
                            metallo_manifest_root=args.metallo_manifest_root, freeze1_sha256=args.freeze1_sha256,
                            allow_unpinned_binary=args.allow_unpinned_binary)
    if args.command == "certify-inner":
        return certify_inner(args.work, args.fold, metallo_folds_dir=args.metallo_folds_dir,
                             metallo_manifest_root=args.metallo_manifest_root,
                             freeze1_sha256=args.freeze1_sha256, allow_unpinned_binary=args.allow_unpinned_binary)
    if args.command == "r1-inner-exclusions":
        return build_r1_inner_exclusions(args.work, args.fold, args.inner_fold, freeze1_sha256=args.freeze1_sha256,
                                         allow_unpinned_binary=args.allow_unpinned_binary)
    if args.command == "fasta":
        return build_inputs(args.work, args.fold, query_set=args.query_set,
                            official_split_dir=args.official_split_dir, metallo_folds_dir=args.metallo_folds_dir,
                            metallo_manifest_root=args.metallo_manifest_root,
                            n_folds=args.n_folds, freeze1_sha256=args.freeze1_sha256)
    if args.command == "search":
        return run_search(args.work, args.fold, query_set=args.query_set, binary=args.binary,
                          expected_sha256=args.binary_sha256, expected_version=args.expected_version,
                          threads=args.threads, split_memory_limit=args.split_memory_limit,
                          timeout_seconds=args.timeout_seconds, freeze1_sha256=args.freeze1_sha256,
                          allow_unpinned_binary=args.allow_unpinned_binary)
    if args.command == "edges":
        return build_edges(args.work, args.fold, query_set=args.query_set, freeze1_sha256=args.freeze1_sha256,
                           allow_unpinned_binary=args.allow_unpinned_binary)
    if args.command == "components":
        if args.scope == SCOPE_COHORT:
            require_freeze1(args.freeze1_sha256, "Cohort clusters")
            if args.allow_unpinned_binary:
                raise ValueError("Cohort clusters always need the pinned MMseqs2 binary; "
                                 "drop --allow-unpinned-binary")
            if args.work is None or not args.fold_work:
                raise ValueError("Cohort scope needs --out and one --fold-work per fold")
            return build_cohort_components(args.fold_work, args.work, freeze1_sha256=args.freeze1_sha256)
        if args.fold is None or args.work is None:
            raise ValueError("Inner scope needs --fold and --work")
        return build_inner_components(args.work, args.fold, allow_unpinned_binary=args.allow_unpinned_binary)
    if args.command == "strata":
        return build_strata(args.work, args.fold, freeze1_sha256=args.freeze1_sha256, clusters=args.clusters)
    raise ValueError(f"Unknown command {args.command}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        result = run_command(args)
    except (ValueError, KeyError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"Identity tool refused: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
