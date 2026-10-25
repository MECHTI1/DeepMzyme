"""Retrained CLEAN comparator for the official CLEAN split30 benchmark (EC class campaign).

This file contains no CLEAN code. It prepares one CLEAN fit work directory from explicit
training-side inputs, launches a locally modified copy of the CLEAN training script in a
separate environment as a subprocess, and defines how CLEAN distances become EC-class
predictions. It never imports CLEAN.

Subcommands
- ``sequence-map``: the training-only accession -> sequence map of the five official
  ``split30_train_split_<fold>.csv`` files (opened by their exact names; never a test file):
  ``sequence_map.tsv`` (accession, sha256, length, n_files, sequence; it loads with
  ``embed_helpers.esmc_chain_means.read_sequence_map``), the sorted base FASTA of the same
  sequences and a receipt. It refuses a duplicate Entry within a file and any accession whose
  sequence or native EC string differs between files (the EC string is normalised for
  separators, whitespace and repeated numbers only, as ``prepare`` reads it).
- ``prepare``: the CLEAN training table (Entry, EC number, Sequence) from one official
  ``split30_train_split_<fold>.csv`` (never a test file), masked copies keyed by
  (accession, seed), optional links to precomputed embeddings (checked against each
  embedding directory's farm manifest) and distance maps, and a manifest with every input
  hash. Regimes: ``r1`` the full file; ``r2`` an accession allowlist (the metallo subset); the
  development gap arms (validation only) ``r1_inner``, the full file minus the identity tool's
  ``r1_inner_exclusions_fold<f>_inner<j>.tsv`` (the inner-validation proteins and their D3
  neighbours), and ``r2_inner``, an allowlist of the three metallo inner-training folds that must
  hold no protein of that exclusions list.
- ``fit``: verifies the CLEAN checkout (commit and file hashes, no ``app/data`` or
  ``app/results``), the private scripts (SHA-256) and every linked embedding, computes the
  distance map when it is not linked, trains in a fresh process and checks the training
  receipt.
- ``export``: the contract C4 distance export of one completed fit. It checks the fit, the
  label-free query list and the query embeddings, launches the private export script (SHA-256
  pinned) in a fresh guarded CLEAN process with ``PYTHONHASHSEED`` equal to the fit seed, then
  validates the written export against the fit (including the EC column order). Mode
  ``validation`` takes training-side queries only (every query must be in the fold's official
  training file) and refuses queries the fit trained on unless ``--allow-in-sample-queries``
  is given for an engineering check; the export records the in-sample count. Mode ``test``
  refuses unless a Freeze-2 manifest with the given SHA-256 has schema_version 1, lists the
  checkpoint in ``checkpoint_sha256s``, pins the live code (``code_sha256`` by the EC prediction
  export's rule, plus this runner in ``additional_code_sha256``) and has a
  ``test_pass_output_root`` that contains the output; it also refuses an export script other
  than the pinned one, any query the fit trained on, an inner regime, and a checkpoint that
  already has a claim file ``<test_pass_output_root>/clean_test_export_claims/<checkpoint
  SHA-256>.json`` (written once, before the launch, so a crashed export is not rerun silently).
- ``esmc-farm``: an embedding farm for the ESMC-input control from the sequence-keyed means of
  ``embed_helpers/esmc_chain_means.py sequences`` (one layer per farm), covering exactly the
  sequence-map accessions (``--kind base``) or one seed's masked copies (``--kind masked``).
- ``validate-export``: checks a distance export against contract C4.

Names: the CLEAN training-data name is ``split30_fold<f>_<regime>_<embedding>`` (the distance
map depends on fold, regime and embedding, not on the seed; an inner regime carries its inner
fold, e.g. ``r1_inner2``) and the model name adds ``_seed<s>_d<width>``, so every (fold, regime,
embedding, seed) fit is unique.

Masked copies follow the published CLEAN procedure: every protein carrying an EC number
that has a single protein in the training table gets 10 copies; each copy draws a mask
rate from Normal(0.10, 0.02), then ceil(length x rate) positions uniformly from 1..length-1
(0-based, with replacement), each replaced by ``<mask>``. The random stream is keyed by
(accession, seed) only, so a protein's copies are identical in every fold and regime.

Embedding farms: an embedding directory holds ``<label>.pt`` files and
``embedding_farm_manifest.tsv`` with the columns label, seed (empty for a base embedding, the
mask seed for a masked copy), sequence_sha256 (SHA-256 of the embedded string, ``<mask>``
tokens included), embedding, layer, dim and file_sha256. ``prepare`` links a label only when
its row matches this fit's sequence, seed, embedding and input width and the file hash
matches; it records a farm digest per kind (SHA-256 of the sorted ``label<TAB>file_sha256``
lines). ``fit`` re-hashes every link. A linked distance map must come from the same training
table, the same base farm digest and input width, the pinned helper and its fixed hash seed.
The upstream loader reads ``<label>.pt`` as either a dict holding ``mean_representations[33]``
(what the fair-esm extraction script writes, so ESM-1b farms keep that file unchanged) or a
plain tensor; ESMC farm files are plain 1-D float32 tensors. The ``layer`` column is fair-esm's
representation layer for ESM-1b (33, the final layer) and, for ESMC, the zero-based block
index of a block mean or the block count (30 for esmc_300m) for the final normalized mean.
The ESMC block numbers are therefore one below fair-esm's (``block26`` is the output of the
27th block, which fair-esm would call layer 27); fair-esm numbering would give the last block's
raw output the final mean's number, and the embedding name already tells the two apart.
A farm is labelled by its producer: a base farm holds every sequence-map accession, a masked
farm one seed's copies (the union over the masked-copy manifests ``prepare`` writes; a copy is
identical in every fold and regime), so one farm serves every fit that needs it.

Contract C4 (distance export, version 2): an ``.npz`` with ``distances`` (float32, [n_query,
n_ec]), ``query_ids`` and ``ec_labels`` (unicode arrays, unique), plus ``<stem>_receipt.json``
(contract, version, array hashes, fold, regime, seed, input width, embedding, mode, and the
SHA-256 of the checkpoint, of the training table, the base farm digest of the training table
(the centres' inputs; equal to the prepare manifest's and the distance-map receipt's) and the
same digest over the query embeddings). Conditional fields: ``inner_fold`` (inner regimes only,
which are validation-only); ``n_in_sample_queries`` (queries that are training proteins of the
fit; 0 in test mode); and in test mode the Freeze-2 binding ``freeze2_manifest_sha256``,
``test_pass_output_root``, ``runner_sha256`` and ``export_script_sha256``, all four or none.
The ``export`` action always writes the binding of a test export; ``check_test_export_binding``
checks it against the scored Freeze 2. The writer is write-once.

EC-class projection: the nearest EC centre, then the first field of its EC number. When
another centre lies within 1e-6 of the minimum distance and has a different first field,
the query is a tie and counts as wrong (the tie count is reported). The seed ensemble
averages each EC label's distance over seeds (each seed with its own centres), then takes
the arg-min under the same rule.

Setup (outside the repository; never commit CLEAN code, a modified copy or a diff)
1. Clone https://github.com/tttianhao/CLEAN with a sparse checkout that excludes
   ``app/data/`` and ``app/results/`` (``app/data`` holds published split files with test
   labels), then check out commit f2bf2a4f497fa2cc87dac2a1bb314fee587c0a15 detached. Do not
   run ``build.py install``; ``fit`` puts ``app/src`` on ``PYTHONPATH``.
2. Create a separate environment: Python 3.10.4, torch 1.11.0 (CPU build), numpy 1.22.3,
   pandas 1.4.2, scikit-learn 1.2.0, scipy 1.7.3, tqdm 4.64.0, matplotlib 3.7.0 and
   fair-esm 2.0.0 (the upstream requirements; the manuscript reports fair-esm 1.0.2).
3. Keep the modified training script and the distance-map helper next to the clone and pass
   their paths to ``fit``; their SHA-256 values are pinned below.

Local changes to the upstream training script ``app/train-triplet.py``, in words (the private
patch has SHA-256 d43a9fc40cd4bb548627aef808816041469b65aa45cf25978e5635999244cf64,
``TRAINING_PATCH_SHA256``):
- a ``--seed`` argument is passed to the upstream seeding function (default 1234, the
  upstream value);
- ``--epoch`` must be a positive multiple of ``--adaptive_rate``, because the upstream
  cleanup otherwise removes the best copy and fails before saving the final weights;
- the previous distance map and data loader are released before each distance-map refresh;
- an ``--input_dim`` argument (default 1280) replaces the first layer's input width;
- a per-epoch loss log (JSON lines) and a final receipt (seed, epochs, refresh rate, input
  width, final-weights SHA-256, peak memory) are written next to the weights.
With default arguments the weights and losses equal upstream bit for bit (private parity
check). ``fit`` also launches every process with a fixed ``PYTHONHASHSEED`` (upstream sets
it inside the running process, where it has no effect), no visible GPU, fixed thread counts,
no user site-packages and the forbidden-read guard.

The private export script (SHA-256 ``EXPORT_SCRIPT_SHA256``), in words: it rebuilds the fit's
model as the modified training script does (input width included), loads the final weights in
evaluation mode without gradients, and calls the upstream centre, query-embedding and
test-distance functions unchanged, with a query dictionary built from the ID list alone (the
upstream inference path would read the EC column of a query table). It repeats this file's
input checks with its own SHA-256 (the test-mode binding records the script that ran) and
writes the full float32 matrix, queries in list order and EC centres in the upstream centre
order (first appearance in the training table), through this file's C4 writer, plus a receipt
of its own.

Licence: CLEAN is distributed under the UIUC non-exclusive research-use licence. Retrained
CLEAN weights and the modified scripts stay private. Reports acknowledge: "The CLEAN Software,
used by [name], was developed by the Department of Chemical and Biomolecular Engineering at
the University of Illinois Urbana-Champaign."
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from project_paths import DATA_DIR, PROJECT_ROOT  # noqa: E402
from training.access_guard import (  # noqa: E402
    FORBIDDEN_READ_ROOTS_ENV, forbidden_roots_environment, install_forbidden_read_guard,
)
from training.ec_classes import EC_CLASS_TOKENS  # noqa: E402


SCHEMA_VERSION = 1

CLEAN_REPOSITORY = "https://github.com/tttianhao/CLEAN"
CLEAN_COMMIT = "f2bf2a4f497fa2cc87dac2a1bb314fee587c0a15"
# SHA-256 of the upstream files a fit executes, at CLEAN_COMMIT.
UPSTREAM_FILE_SHA256: dict[str, str] = {
    "app/train-triplet.py": "fe8b3e2e02220878a1034ff08a847953cc1e2007161be934767f306f6e61d3bc",
    "app/src/CLEAN/__init__.py": "4cbc9e1f119932dea48183adccf0cd4199e163a1ba5dd00c1daf82de136935ec",
    "app/src/CLEAN/dataloader.py": "4132e8c2b65453306704cc40128e97229366f285d479936ed7f5d5ea94142ae6",
    "app/src/CLEAN/distance_map.py": "db73ada3ab41e3b1e9a016ca10a716c8f4513be322ba7b0392ae35c9439601f9",
    "app/src/CLEAN/evaluate.py": "98c068a85c1cfb150da65c1ce7d5c1017de0388a982ca85ef40f21b32834d460",
    "app/src/CLEAN/infer.py": "8647254d58e49a0ce66556c2e4f10d6f32a8182617b922bc71af629a2b1b2189",
    "app/src/CLEAN/losses.py": "35c051e4ca95ace961da5a5acb083837661d5ecb042893b7d8db51186e1e89cd",
    "app/src/CLEAN/model.py": "bd524693ce8c8a5538c05bf8909f483bfded907e4c8751464d57e44a94915ab8",
    "app/src/CLEAN/uncertainty.py": "3e5de19012c1360e0c787f923915868c54e8c90cdf672e3119b53ac7e23646ad",
    "app/src/CLEAN/utils.py": "b8962ccb5b2d7d241eb0ff15e64686e26c7f77ad109cd38c607908db902fe23f",
}
HELD_OUT_CLONE_DIRS = ("app/data", "app/results")
CLEAN_SOURCE_DIR = "app/src"

# Pinned private files (SHA-256 only; the files stay outside the repository).
TRAINING_SCRIPT_SHA256 = "bcc8f407168dfff5ad6a870205566561b032d67f46cb61f24c39f6dfef48364e"
# The minimal patch that turns upstream app/train-triplet.py into the training script above.
TRAINING_PATCH_SHA256 = "d43a9fc40cd4bb548627aef808816041469b65aa45cf25978e5635999244cf64"
DISTANCE_MAP_HELPER_SHA256 = "9e7c8e510b4edb8322384300eb51f48e66efbb374f14a7df03b94c5380155d5c"
EXPORT_SCRIPT_SHA256 = "d1a40de6e2d2f78be7a3711b1e431697d8dd7b559c5d2ba21bb2b54d4b9ac19b"

N_FOLDS = 5
OFFICIAL_TRAIN_NAME = "split30_train_split_{fold}.csv"
SPLIT30_DIR = DATA_DIR / "CLEAN_all_train_valid_splits" / "split30"
# r1: the fold's full official training file; r2: an accession allowlist (metallo subset).
OUTER_REGIMES = ("r1", "r2")
# Development gap arms (validation only), one per inner fold: r1_inner drops the identity tool's
# exclusions list from the full file; r2_inner is an allowlist checked against that list.
INNER_REGIMES = ("r1_inner", "r2_inner")
REGIMES = OUTER_REGIMES + INNER_REGIMES
N_INNER_FOLDS = 4
# The identity tool's r1-inner-exclusions output (ec_clean30.identity; equality is tested).
EXCLUSIONS_NAME = "r1_inner_exclusions_fold{fold}_inner{inner_fold}.tsv"
EXCLUSIONS_COLUMNS = ("accession", "reason")
EXCLUSION_REASON_VALIDATION = "inner_validation"
EXCLUSION_REASONS = (EXCLUSION_REASON_VALIDATION, "d3_neighbour")
PUBLISHED_RECIPE: dict[str, Any] = {
    "learning_rate": 5e-4, "epochs": 2500, "adaptive_rate": 100, "hidden_dim": 512, "out_dim": 128,
}
DEFAULT_INPUT_DIM = 1280
DEFAULT_EMBEDDING = "esm1b_t33_650M_UR50S_mean_layer33"
# The embedding name doubles as a file-name tag.
EMBEDDING_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,79}$")
DEFAULT_THREADS = 2
# The distance map does not depend on the fit seed, so its process uses one fixed hash seed.
DISTANCE_MAP_HASH_SEED = 0

MASK_COPIES = 10
MASK_RATE_MEAN = 0.10
MASK_RATE_SD = 0.02
MASK_TOKEN = "<mask>"
MASK_STREAM = "deepmzyme-clean-masks-v1"

TABLE_HEADER = ("Entry", "EC number", "Sequence")
ACCESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")
SEQUENCE_RE = re.compile(r"^[A-Z]+$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
UNSIGNED_RE = re.compile(r"^[0-9]+$")

# Held-out inputs: test-named files or directories, held-out directory names and roots.
HELD_OUT_NAME_RE = re.compile(r"(^|[_.\-])test([_.\-]|$)", re.IGNORECASE)
HELD_OUT_DIR_NAMES = frozenset({"test", "metadata", "folds", "pinmymetal_files"})
# Exact constructed paths (no directory is listed): mixed CLEAN30 folders, the official split30
# test files, the CLEAN30 test folders, the CARE task-1 tests and the PinMyMetal folder.
DEFAULT_FORBIDDEN_ROOTS: tuple[Path, ...] = (
    DATA_DIR / "CLEAN_30_main" / "metadata",
    DATA_DIR / "CLEAN_30_main" / "folds",
    PROJECT_ROOT / "prepare_training_and_test_set" / "pinmymetal_files",
    DATA_DIR / "CLEAN_30_shared" / "metadata",
    DATA_DIR / "CLEAN_30_shared" / "folds",
    *(SPLIT30_DIR / name.format(fold=fold) for fold in range(N_FOLDS)
      for name in ("split30_test_split_{fold}.csv", "split30_test_split_{fold}_curate.csv")),
    *(DATA_DIR / f"CLEAN_30_train_test_split_{fold}" / "test" for fold in range(N_FOLDS)),
    DATA_DIR / "CARE_dataset" / "CARE_datasets" / "splits" / "task1" / "30_protein_test.csv",
    DATA_DIR / "CARE_task1_30_clusterRes30_train_test_metallo" / "test",
)

PREPARE_MANIFEST = "prepare_manifest.json"
FIT_MANIFEST = "fit_manifest.json"
FARM_MANIFEST = "embedding_farm_manifest.tsv"
FARM_MANIFEST_COLUMNS = ("label", "seed", "sequence_sha256", "embedding", "layer", "dim", "file_sha256")
FARM_LINKS = "embedding_farm_links.tsv"
FARM_KINDS = ("base", "masked")
FARM_RECEIPT = "farm_receipt.json"
ESMC_MODEL_NAME = "esmc_300m"
ESMC_DIM = 960

SEQUENCE_MAP_FILE = "sequence_map.tsv"
SEQUENCE_MAP_COLUMNS = ("accession", "sha256", "length", "n_files", "sequence")
SEQUENCE_MAP_FASTA = "sequence_map_base.fasta"
SEQUENCE_MAP_RECEIPT = "sequence_map_receipt.json"
# A label '<accession>_<n>' names masked copy n (no leading zeros) of a sequence-map accession.
COPY_LABEL_RE = re.compile(r"^(?P<accession>.+)_(?P<copy>0|[1-9][0-9]*)$")
ACCESS_GUARD_FILE = PROJECT_ROOT / "src" / "training" / "access_guard.py"
PASSTHROUGH_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "USER", "LOGNAME")
# Child bootstrap: install the forbidden-read guard, then run the script as __main__.
GUARDED_LAUNCH = "\n".join((
    "import importlib.util, os, runpy, sys",
    "spec = importlib.util.spec_from_file_location('deepmzyme_access_guard', sys.argv[1])",
    "guard = importlib.util.module_from_spec(spec)",
    "spec.loader.exec_module(guard)",
    "if not guard.install_guard_from_environment():",
    "    raise SystemExit('refusing to run without forbidden-read roots')",
    "script = os.path.abspath(sys.argv[2])",
    "sys.argv = [script] + sys.argv[3:]",
    "sys.path[0] = os.path.dirname(script)",
    "runpy.run_path(script, run_name='__main__')",
))

C4_CONTRACT = "clean_split30_distance_export"
C4_VERSION = 2
C4_ARRAYS = ("distances", "query_ids", "ec_labels")
C4_MODES = ("validation", "test")
C4_METADATA_FIELDS = ("fold", "regime", "seed", "input_dim", "embedding", "mode",
                      "checkpoint_sha256", "training_table_sha256", "embedding_farm_sha256",
                      "query_embeddings_sha256")
C4_SHA256_FIELDS = ("checkpoint_sha256", "training_table_sha256", "embedding_farm_sha256",
                    "query_embeddings_sha256")
# Test mode only, all four or none: the Freeze-2 pass the export belongs to and the code that ran.
C4_TEST_BINDING_FIELDS = ("freeze2_manifest_sha256", "test_pass_output_root", "runner_sha256",
                          "export_script_sha256")
# Conditional fields, written only when given (see the module docstring).
C4_OPTIONAL_FIELDS = ("inner_fold", "n_in_sample_queries") + C4_TEST_BINDING_FIELDS
# The seed exports of one ensemble agree on everything except the seed and the checkpoint.
ENSEMBLE_SHARED_FIELDS = ("fold", "regime", "inner_fold", "embedding", "input_dim", "mode", "training_table_sha256",
                          "embedding_farm_sha256", "query_embeddings_sha256")
C4_DISTANCE_RULE = ("float32 Euclidean distance from each query's model embedding to each training EC "
                    "centre (mean model embedding of that EC's training proteins)")
TIE_TOLERANCE = 1e-6
TIE_CLASS = ""  # predicted class of a tied query; it never equals a true class
PROJECTION_RULE = ("nearest EC centre, then the first field of its EC number; another centre within "
                   f"{TIE_TOLERANCE:g} of the minimum with a different first field makes a tie, counted wrong")
ENSEMBLE_RULE = "mean distance per EC label over seeds (each seed with its own centres), then the projection"
# Next to a C4 export: the export script's receipt (written by it), the launch log and run record.
EXPORT_SIDE_SUFFIXES = {"export_receipt": "_export_receipt.json", "log": "_export.log",
                        "run_record": "_export_run.json"}
EC_LABEL_ORDER = "upstream centre order: first appearance in the fit's training table"
FREEZE2_SCHEMA_VERSION = 1
# The Freeze-2 manifest pins this runner under additional_code_sha256 and, under code_sha256, the
# repository files it and its guarded launches execute (ec_clean30.provenance pins all of them).
RUNNER_KEY = "scripts/run_clean_split30_comparator.py"
RUNNER_SOURCE_PATHS = ("src/project_paths.py", "src/training/__init__.py", "src/training/access_guard.py",
                       "src/training/ec_classes.py")
# One claim file per checkpoint under the test pass root, written before the test export runs.
TEST_CLAIMS_DIR = "clean_test_export_claims"


class ComparatorError(RuntimeError):
    """A comparator input, launch or output contract failed."""


# ---------------------------------------------------------------- hashing and path rules

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_lines(values: Iterable[str]) -> str:
    return sha256_bytes("\n".join(values).encode("utf-8"))


def _is_under(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def resolve_forbidden_roots(extra: Iterable[Path | str] = (), clean_clone: Optional[Path] = None) -> list[Path]:
    roots = [Path(root) for root in DEFAULT_FORBIDDEN_ROOTS] + [Path(root) for root in extra]
    if clean_clone is not None:
        roots += [Path(clean_clone) / part for part in HELD_OUT_CLONE_DIRS]
    resolved: list[Path] = []
    for root in roots:
        root = root.resolve()
        if root not in resolved:
            resolved.append(root)
    return resolved


def refuse_held_out_path(path: Path | str, forbidden_roots: Sequence[Path], what: str) -> Path:
    """Resolved ``path``; refuses test-named parts, held-out directory names and forbidden roots."""
    literal = Path(path).absolute()
    resolved = literal.resolve()
    for candidate in (literal, resolved):
        blocked = [part for part in candidate.parts[1:]
                   if HELD_OUT_NAME_RE.search(part) or part.lower() in HELD_OUT_DIR_NAMES]
        if blocked:
            raise ComparatorError(f"Refusing {what} through a held-out name {blocked}: {path}")
        for root in forbidden_roots:
            if _is_under(candidate, Path(root).resolve()):
                raise ComparatorError(f"Refusing {what} under the forbidden root {root}: {path}")
    return resolved


def refuse_forbidden_root(path: Path | str, forbidden_roots: Sequence[Path], what: str) -> Path:
    """Resolved ``path``; refuses forbidden roots only (inputs of a frozen test pass may carry test names)."""
    literal = Path(path).absolute()
    for candidate in (literal, literal.resolve()):
        for root in forbidden_roots:
            if _is_under(candidate, Path(root).resolve()):
                raise ComparatorError(f"Refusing {what} under the forbidden root {root}: {path}")
    return literal.resolve()


def _require_new_directory(directory: Path, what: str) -> None:
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise ComparatorError(f"The {what} is not a new or empty directory: {directory}")


def refuse_inside_repository(path: Path | str, what: str, repository: Path = PROJECT_ROOT) -> Path:
    resolved = Path(path).resolve()
    if _is_under(resolved, Path(repository).resolve()):
        raise ComparatorError(f"Refusing {what} inside the repository work tree (CLEAN material stays "
                              f"outside it): {path}")
    return resolved


def assert_official_training_file(path: Path | str, fold: int, forbidden_roots: Sequence[Path]) -> Path:
    if not 0 <= fold < N_FOLDS:
        raise ComparatorError(f"Fold {fold} outside 0..{N_FOLDS - 1}")
    expected = OFFICIAL_TRAIN_NAME.format(fold=fold)
    literal = Path(path)
    resolved = refuse_held_out_path(literal, forbidden_roots, "an official training file")
    if literal.name != expected or resolved.name != expected:
        raise ComparatorError(f"Refusing {path}: fold {fold} needs the official training file {expected}")
    if not resolved.is_file():
        raise ComparatorError(f"Official training file not found: {path}")
    return resolved


def check_epochs(epochs: int, adaptive_rate: int) -> None:
    if adaptive_rate <= 0 or epochs <= 0 or epochs % adaptive_rate:
        raise ComparatorError(f"epochs {epochs} must be a positive multiple of adaptive_rate {adaptive_rate}; "
                              "the upstream cleanup fails otherwise")


# ---------------------------------------------------------------- training table

@dataclass(frozen=True)
class TrainingRow:
    accession: str
    ec_numbers: tuple[str, ...]
    sequence: str


def ec_class_of_label(label: str) -> str:
    """First field of an EC number ('6.2.1.n2' -> '6', '1.-.-.-' -> '1'); refuses other tokens."""
    token = str(label).split(".", 1)[0].strip()
    if token not in EC_CLASS_TOKENS:
        raise ComparatorError(f"EC number {label!r} has no EC class in {EC_CLASS_TOKENS}")
    return token


def _normalize_ec_field(value: str, where: str) -> tuple[tuple[str, ...], bool]:
    tokens = [token.strip() for token in value.split(";")]
    kept: list[str] = []
    for token in tokens:
        if not token:
            continue
        if re.search(r"[\s,]", token):
            raise ComparatorError(f"Malformed EC token {token!r} at {where}")
        ec_class_of_label(token)
        if token not in kept:
            kept.append(token)
    if not kept:
        raise ComparatorError(f"Missing EC number at {where}")
    return tuple(kept), ";".join(kept) != value


def read_official_training_rows(path: Path | str, fold: int,
                                forbidden_roots: Sequence[Path]) -> tuple[list[TrainingRow], dict[str, Any]]:
    """Rows of an official training file in file order; only Entry, EC number and Sequence are read."""
    path = assert_official_training_file(path, fold, forbidden_roots)
    rows: list[TrainingRow] = []
    seen: set[str] = set()
    normalized = 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = [name.strip().lower() for name in next(reader, [])]
        try:
            columns = [header.index(name.lower()) for name in TABLE_HEADER]
        except ValueError:
            raise ComparatorError(f"{path.name} needs the tab-separated columns {TABLE_HEADER}") from None
        for line, raw in enumerate(reader, start=2):
            if not any(field.strip() for field in raw):
                continue
            where = f"{path.name}:{line}"
            if len(raw) <= max(columns):
                raise ComparatorError(f"Short row at {where}")
            accession, ec_field, sequence = (raw[index] for index in columns)
            accession = accession.strip()
            if not ACCESSION_RE.match(accession):
                raise ComparatorError(f"Invalid accession {accession!r} at {where}")
            if accession in seen:
                raise ComparatorError(f"Duplicate Entry {accession} at {where}")
            seen.add(accession)
            ec_numbers, changed = _normalize_ec_field(ec_field, where)
            clean_sequence = re.sub(r"\s+", "", sequence)
            if not SEQUENCE_RE.match(clean_sequence) or len(clean_sequence) < 2:
                raise ComparatorError(f"Invalid sequence for {accession} at {where}")
            normalized += int(changed or clean_sequence != sequence)
            rows.append(TrainingRow(accession, ec_numbers, clean_sequence))
    if not rows:
        raise ComparatorError(f"No rows in {path}")
    return rows, {"path": str(path), "sha256": sha256_file(path), "n_rows": len(rows),
                  "n_rows_whitespace_or_separator_normalized": normalized}


def read_accession_allowlist(path: Path | str, forbidden_roots: Sequence[Path]) -> tuple[list[str], dict[str, Any]]:
    """One accession per line ('#' comments and blank lines skipped); duplicates are refused."""
    return _read_accession_lines(refuse_held_out_path(path, forbidden_roots, "an accession allowlist"))


def _read_accession_lines(path: Path) -> tuple[list[str], dict[str, Any]]:
    """A row with a tab, comma, semicolon or space (a table row, e.g. with an EC column) is refused
    without being echoed, so a refused labelled table prints no label."""
    accessions: list[str] = []
    seen: set[str] = set()
    for line, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        value = raw.split("#", 1)[0].strip()
        if not value:
            continue
        if re.search(r"[\s,;]", value):
            raise ComparatorError(f"{path.name}:{line} holds more than one field (a table row?); expected one "
                                  "accession per line")
        if not ACCESSION_RE.match(value):
            raise ComparatorError(f"Invalid accession {value!r} at {path.name}:{line}")
        if value in seen:
            raise ComparatorError(f"Duplicate accession {value} at {path.name}:{line}")
        seen.add(value)
        accessions.append(value)
    if not accessions:
        raise ComparatorError(f"Empty accession allowlist: {path}")
    return accessions, {"path": str(path), "sha256": sha256_file(path), "n_accessions": len(accessions)}


def select_rows(rows: Sequence[TrainingRow], accessions: Sequence[str]) -> list[TrainingRow]:
    """Allowlisted rows in official order; an allowlisted accession missing from the file is refused."""
    wanted = set(accessions)
    missing = sorted(wanted - {row.accession for row in rows})
    if missing:
        raise ComparatorError(f"{len(missing)} allowlisted accessions are not in the official training file, "
                              f"e.g. {missing[:10]}")
    return [row for row in rows if row.accession in wanted]


def read_inner_exclusions(path: Path | str, fold: int, inner_fold: int,
                          forbidden_roots: Sequence[Path]) -> tuple[dict[str, str], dict[str, Any]]:
    """accession -> reason from the identity tool's ``r1_inner_exclusions_fold<f>_inner<j>.tsv``.

    The file name must name this fold and inner fold; the header must be exactly
    ``accession, reason`` (a label-free list); reasons are ``inner_validation`` or ``d3_neighbour``,
    accessions are unique and at least one inner-validation protein is listed.
    """
    literal = Path(path)
    resolved = refuse_held_out_path(literal, forbidden_roots, "an inner exclusions list")
    expected = EXCLUSIONS_NAME.format(fold=fold, inner_fold=inner_fold)
    if literal.name != expected or resolved.name != expected:
        raise ComparatorError(f"Refusing {path}: fold {fold}, inner fold {inner_fold} needs {expected}")
    reasons: dict[str, str] = {}
    with resolved.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        if tuple(next(reader, ())) != EXCLUSIONS_COLUMNS:
            raise ComparatorError(f"{expected} needs exactly the tab-separated columns {EXCLUSIONS_COLUMNS}")
        for line, raw in enumerate(reader, start=2):
            where = f"{expected}:{line}"
            if len(raw) != len(EXCLUSIONS_COLUMNS):
                raise ComparatorError(f"Malformed exclusions row at {where}")
            accession, reason = raw
            if not ACCESSION_RE.match(accession) or accession in reasons:
                raise ComparatorError(f"Invalid or repeated accession {accession!r} at {where}")
            if reason not in EXCLUSION_REASONS:
                raise ComparatorError(f"Unknown exclusion reason {reason!r} at {where}")
            reasons[accession] = reason
    counts = Counter(reasons.values())
    if not counts[EXCLUSION_REASON_VALIDATION]:
        raise ComparatorError(f"{expected} lists no {EXCLUSION_REASON_VALIDATION} protein")
    return reasons, {"path": str(resolved), "sha256": sha256_file(resolved), "fold": fold, "inner_fold": inner_fold,
                     "n_rows": len(reasons), "n_by_reason": {reason: counts[reason] for reason in EXCLUSION_REASONS}}


def write_training_table(rows: Sequence[TrainingRow], path: Path) -> str:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(TABLE_HEADER)
        for row in rows:
            writer.writerow((row.accession, ";".join(row.ec_numbers), row.sequence))
    return sha256_file(path)


def single_protein_ec_accessions(rows: Sequence[TrainingRow]) -> list[str]:
    """Accessions carrying an EC number (complete or not) that only one protein of the table carries."""
    members: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        for ec in row.ec_numbers:
            members[ec].add(row.accession)
    single = {ec for ec, ids in members.items() if len(ids) == 1}
    return [row.accession for row in rows if any(ec in single for ec in row.ec_numbers)]


# ---------------------------------------------------------------- training-only sequence map

@dataclass(frozen=True)
class SequenceMapEntry:
    accession: str
    sequence: str
    ec_string: str  # normalised native EC string (identical in every file holding the accession)
    files: tuple[str, ...]  # official training files holding the accession, in fold order

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.sequence.encode("utf-8"))


def official_training_paths(split30_dir: Path | str) -> list[Path]:
    """The five official training files by exact constructed name (the folder is never listed)."""
    return [Path(split30_dir) / OFFICIAL_TRAIN_NAME.format(fold=fold) for fold in range(N_FOLDS)]


def canonical_ids_sha256(ids: Iterable[str]) -> str:
    """SHA-256 of the sorted IDs, one per line with a trailing newline (``read_sequence_map``'s ids_sha256)."""
    return sha256_bytes("".join(f"{item}\n" for item in sorted(ids)).encode("utf-8"))


def collect_sequence_map(official_files: Sequence[Path | str],
                         forbidden_roots: Sequence[Path]) -> tuple[list[SequenceMapEntry], list[dict[str, Any]]]:
    """Accession union of the five official training files (file ``k`` must be fold ``k``'s), sorted.

    The row reader refuses a duplicate Entry within a file. Across files, an accession must carry the
    same sequence and the same normalised native EC string; conflicts are refused, listing (up to 20)
    both values of each.
    """
    official_files = list(official_files)
    if len(official_files) != N_FOLDS:
        raise ComparatorError(f"The sequence map needs the {N_FOLDS} official training files, got "
                              f"{len(official_files)}")
    first: dict[str, tuple[str, str, str]] = {}  # accession -> (sequence, EC string, first file)
    files: dict[str, list[str]] = defaultdict(list)
    inputs: list[dict[str, Any]] = []
    problems: list[str] = []
    for fold, path in enumerate(official_files):
        rows, report = read_official_training_rows(path, fold, forbidden_roots)
        name = Path(report["path"]).name
        inputs.append({"fold": fold, "file": name, **report})
        for row in rows:
            ec_string = ";".join(row.ec_numbers)
            sequence, first_ec, first_file = first.setdefault(row.accession, (row.sequence, ec_string, name))
            if sequence != row.sequence:
                problems.append(f"{row.accession}: sequence in {first_file} is {sequence}; in {name} it is "
                                f"{row.sequence}")
            if first_ec != ec_string:
                problems.append(f"{row.accession}: EC string in {first_file} is {first_ec!r}; in {name} it is "
                                f"{ec_string!r}")
            files[row.accession].append(name)
    if problems:
        raise ComparatorError(f"{len(problems)} cross-file conflicts in the official training files:\n  "
                              + "\n  ".join(problems[:20]))
    ambiguous = sorted(accession for accession in first if (match := COPY_LABEL_RE.match(accession))
                       and match.group("accession") in first)
    if ambiguous:
        raise ComparatorError(f"Accessions that read as a masked copy of another accession: {ambiguous[:10]}")
    entries = [SequenceMapEntry(accession, first[accession][0], first[accession][1], tuple(files[accession]))
               for accession in sorted(first)]
    return entries, inputs


def write_sequence_map(*, split30_dir: Path | str = SPLIT30_DIR, out_dir: Path | str,
                       forbidden_roots: Sequence[Path] = ()) -> dict[str, Any]:
    """Write-once ``sequence_map.tsv``, the sorted base FASTA and a receipt in a new directory."""
    roots = resolve_forbidden_roots(forbidden_roots)
    out_dir = refuse_inside_repository(refuse_held_out_path(out_dir, roots, "the sequence-map directory"),
                                       "the sequence-map directory")
    _require_new_directory(out_dir, "sequence-map directory")
    entries, inputs = collect_sequence_map(official_training_paths(split30_dir), roots)
    out_dir.mkdir(parents=True, exist_ok=True)
    map_path, fasta_path = out_dir / SEQUENCE_MAP_FILE, out_dir / SEQUENCE_MAP_FASTA
    with map_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(SEQUENCE_MAP_COLUMNS)
        for entry in entries:
            writer.writerow((entry.accession, entry.sha256, len(entry.sequence), len(entry.files), entry.sequence))
    with fasta_path.open("x", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(f">{entry.accession}\n{entry.sequence}\n")
    histogram = Counter(len(entry.files) for entry in entries)
    receipt = {
        "schema_version": SCHEMA_VERSION, "action": "sequence-map", "inputs": inputs,
        "n_rows_total": sum(item["n_rows"] for item in inputs), "n_accessions": len(entries),
        "n_unique_sequences": len({entry.sha256 for entry in entries}),
        "n_files_histogram": {str(count): histogram[count] for count in sorted(histogram)},
        "accessions_sha256": canonical_ids_sha256(entry.accession for entry in entries),
        "outputs": {"sequence_map": {"file": map_path.name, "sha256": sha256_file(map_path)},
                    "base_fasta": {"file": fasta_path.name, "sha256": sha256_file(fasta_path)}},
        "rules": {"sequence_sha256": "SHA-256 of the UTF-8 residue string (whitespace removed)",
                  "ec_string": "native EC numbers joined by ';' after removing whitespace, empty fields and "
                               "repeated numbers; no number is remapped",
                  "conflicts": "a duplicate Entry within a file, or a different sequence or EC string across "
                               "files, is refused"},
        "forbidden_roots": [str(root) for root in roots],
        "runner": Path(__file__).name, "runner_sha256": sha256_file(Path(__file__)),
    }
    _write_json_once(out_dir / SEQUENCE_MAP_RECEIPT, receipt)
    return receipt


def read_sequence_map_file(path: Path | str, forbidden_roots: Sequence[Path],
                           expected_sha256: Optional[str] = None) -> tuple[dict[str, str], dict[str, Any]]:
    """accession -> sequence from a ``sequence-map`` output; each row's hash, length and residues are checked."""
    path = refuse_held_out_path(path, forbidden_roots, "a sequence map")
    data = path.read_bytes()
    digest = sha256_bytes(data)
    if expected_sha256 is not None and digest != str(expected_sha256).strip().lower():
        raise ComparatorError(f"Sequence map {path} has SHA-256 {digest}, expected {expected_sha256}")
    reader = csv.reader(io.StringIO(data.decode("utf-8"), newline=""), delimiter="\t")
    if tuple(next(reader, ())) != SEQUENCE_MAP_COLUMNS:
        raise ComparatorError(f"{path.name} needs the tab-separated columns {SEQUENCE_MAP_COLUMNS}")
    sequences: dict[str, str] = {}
    for line, raw in enumerate(reader, start=2):
        where = f"{path.name}:{line}"
        if len(raw) != len(SEQUENCE_MAP_COLUMNS):
            raise ComparatorError(f"Malformed sequence-map row at {where}")
        accession, sequence_sha256, length, n_files, sequence = raw
        if not ACCESSION_RE.match(accession) or accession in sequences:
            raise ComparatorError(f"Invalid or repeated accession {accession!r} at {where}")
        if not (SEQUENCE_RE.match(sequence) and UNSIGNED_RE.match(length) and UNSIGNED_RE.match(n_files)
                and int(length) == len(sequence) and sha256_bytes(sequence.encode("utf-8")) == sequence_sha256):
            raise ComparatorError(f"Sequence, length or SHA-256 of {accession} disagree at {where}")
        sequences[accession] = sequence
    if not sequences:
        raise ComparatorError(f"Empty sequence map: {path}")
    return sequences, {"path": str(path), "sha256": digest, "n_accessions": len(sequences),
                       "accessions_sha256": canonical_ids_sha256(sequences)}


# ---------------------------------------------------------------- masked copies

@dataclass(frozen=True)
class MaskedCopy:
    accession: str
    seed: int
    copy: int
    rate: float
    n_draws: int
    positions: tuple[int, ...]  # distinct masked 0-based positions, sorted
    sequence: str

    @property
    def label(self) -> str:
        return f"{self.accession}_{self.copy}"


def mask_stream(accession: str, seed: int) -> random.Random:
    """Random stream keyed by (accession, seed) only; only ``random()`` is used (stable across versions)."""
    key = hashlib.sha256(f"{MASK_STREAM}\x00{int(seed)}\x00{accession}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(key, "big"))


def _standard_normal(stream: random.Random) -> float:
    u1, u2 = stream.random(), stream.random()
    return math.sqrt(-2.0 * math.log(1.0 - u1)) * math.cos(2.0 * math.pi * u2)


def masked_copies(accession: str, sequence: str, seed: int, copies: int = MASK_COPIES) -> list[MaskedCopy]:
    """The published masking procedure, drawn from the (accession, seed) stream."""
    if len(sequence) < 2:
        raise ComparatorError(f"Sequence of {accession} is too short to mask")
    stream = mask_stream(accession, seed)
    result = []
    for copy in range(copies):
        rate = MASK_RATE_MEAN + MASK_RATE_SD * _standard_normal(stream)
        n_draws = max(0, math.ceil(len(sequence) * rate))
        draws = [1 + int(stream.random() * (len(sequence) - 1)) for _ in range(n_draws)]
        residues = list(sequence)
        for position in draws:
            residues[position] = MASK_TOKEN
        result.append(MaskedCopy(accession, int(seed), copy, rate, n_draws, tuple(sorted(set(draws))),
                                 "".join(residues)))
    return result


MASK_MANIFEST_COLUMNS = ("label", "accession", "seed", "copy", "rate", "n_draws", "n_masked",
                         "sequence_sha256", "masked_sha256")


def write_masked_copies(rows: Sequence[TrainingRow], seed: int, fasta_path: Path,
                        manifest_path: Path) -> tuple[list[MaskedCopy], dict[str, Any]]:
    """Sorted masked FASTA and manifest for the single-protein-EC accessions of ``rows``."""
    by_accession = {row.accession: row for row in rows}
    accessions = sorted(single_protein_ec_accessions(rows))
    copies = [copy for accession in accessions
              for copy in masked_copies(accession, by_accession[accession].sequence, seed)]
    collisions = sorted({copy.label for copy in copies} & set(by_accession))
    if collisions:
        raise ComparatorError(f"Masked-copy labels collide with accessions: {collisions[:10]}")
    with fasta_path.open("x", encoding="utf-8") as handle:
        for copy in copies:
            handle.write(f">{copy.label}\n{copy.sequence}\n")
    with manifest_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(MASK_MANIFEST_COLUMNS)
        for copy in copies:
            writer.writerow((copy.label, copy.accession, copy.seed, copy.copy, repr(copy.rate), copy.n_draws,
                             len(copy.positions), sha256_bytes(by_accession[copy.accession].sequence.encode()),
                             sha256_bytes(copy.sequence.encode())))
    return copies, {"seed": int(seed), "stream": MASK_STREAM, "n_accessions": len(accessions),
                    "n_copies": len(copies), "fasta": fasta_path.name, "fasta_sha256": sha256_file(fasta_path),
                    "manifest": manifest_path.name, "manifest_sha256": sha256_file(manifest_path)}


def read_masked_manifest(path: Path | str, forbidden_roots: Sequence[Path]) -> tuple[list[dict[str, str]],
                                                                                     dict[str, Any]]:
    path = refuse_held_out_path(path, forbidden_roots, "a masked-copy manifest")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        if tuple(next(reader, ())) != MASK_MANIFEST_COLUMNS:
            raise ComparatorError(f"{path.name} needs the tab-separated columns {MASK_MANIFEST_COLUMNS}")
        rows = []
        for line, raw in enumerate(reader, start=2):
            if len(raw) != len(MASK_MANIFEST_COLUMNS):
                raise ComparatorError(f"Malformed masked-copy manifest row at {path.name}:{line}")
            rows.append(dict(zip(MASK_MANIFEST_COLUMNS, raw)))
    return rows, {"path": str(path), "sha256": sha256_file(path), "n_rows": len(rows)}


def farm_labels(sequences: Mapping[str, str], *, kind: str, seed: Optional[int] = None,
                masked_manifests: Sequence[Path | str] = (),
                forbidden_roots: Sequence[Path] = ()) -> tuple[dict[str, tuple[Optional[int], str]], list[dict]]:
    """The labels one farm must hold, each with (mask seed or None, the string to embed).

    ``base``: every sequence-map accession. ``masked``: the union of the copies listed in the given
    masked-copy manifests (one per prepared fit) for ``seed``; each copy is regenerated from the
    (accession, seed) stream and must match its row, and its source must be the map's sequence.
    """
    if kind not in FARM_KINDS:
        raise ComparatorError(f"Unknown farm kind {kind!r}; expected one of {FARM_KINDS}")
    if kind == "base":
        if seed is not None or masked_manifests:
            raise ComparatorError("A base farm takes no mask seed and no masked-copy manifest")
        return {accession: (None, sequence) for accession, sequence in sequences.items()}, []
    if seed is None or int(seed) < 0 or not masked_manifests:
        raise ComparatorError("A masked farm needs a non-negative mask seed and at least one masked-copy manifest")
    labels: dict[str, tuple[Optional[int], str]] = {}
    records: list[dict[str, Any]] = []
    regenerated: dict[str, dict[str, MaskedCopy]] = {}
    for path in masked_manifests:
        rows, record = read_masked_manifest(path, forbidden_roots)
        records.append(record)
        for line, row in enumerate(rows, start=2):
            where = f"{Path(record['path']).name}:{line}"
            accession = row["accession"]
            if accession not in sequences:
                raise ComparatorError(f"{where}: {accession} is not in the sequence map")
            if row["seed"] != str(int(seed)):
                raise ComparatorError(f"{where}: copy of mask seed {row['seed']}, not {int(seed)}")
            if row["sequence_sha256"] != sha256_bytes(sequences[accession].encode("utf-8")):
                raise ComparatorError(f"{where}: the source sequence of {accession} differs from the sequence map")
            if accession not in regenerated:
                regenerated[accession] = {copy.label: copy
                                          for copy in masked_copies(accession, sequences[accession], int(seed))}
            copy = regenerated[accession].get(row["label"])
            if copy is None or [row["copy"], row["rate"], row["n_draws"], row["n_masked"], row["masked_sha256"]] != [
                    str(copy.copy), repr(copy.rate), str(copy.n_draws), str(len(copy.positions)),
                    sha256_bytes(copy.sequence.encode("utf-8"))]:
                raise ComparatorError(f"{where}: {row['label']} is not the keyed masked copy of {accession}")
            labels[copy.label] = (int(seed), copy.sequence)
    collisions = sorted(set(labels) & set(sequences))
    if collisions:
        raise ComparatorError(f"Masked-copy labels collide with accessions: {collisions[:10]}")
    return labels, records


# ---------------------------------------------------------------- work directory

def check_embedding_tag(embedding: str) -> str:
    """The embedding name, which is also a file-name tag; refuses unsafe or test-named values."""
    embedding = str(embedding)
    if not EMBEDDING_TAG_RE.match(embedding) or HELD_OUT_NAME_RE.search(embedding):
        raise ComparatorError(f"Embedding name {embedding!r} must match {EMBEDDING_TAG_RE.pattern} and carry "
                              "no held-out token")
    return embedding


def regime_tag(regime: str, inner_fold: Optional[int] = None) -> str:
    """The regime in names: an inner regime carries its inner fold ('r1_inner2'); an outer one takes none."""
    if regime not in REGIMES:
        raise ComparatorError(f"Unknown regime {regime!r}; expected one of {REGIMES}")
    if (regime in INNER_REGIMES) != (inner_fold is not None):
        raise ComparatorError(f"Regime {regime} {'needs' if regime in INNER_REGIMES else 'takes no'} inner fold")
    if inner_fold is None:
        return regime
    if isinstance(inner_fold, bool) or not isinstance(inner_fold, int) or not 0 <= inner_fold < N_INNER_FOLDS:
        raise ComparatorError(f"Inner fold {inner_fold!r} outside 0..{N_INNER_FOLDS - 1}")
    return f"{regime}{inner_fold}"


def training_name(fold: int, regime: str, embedding: str, inner_fold: Optional[int] = None) -> str:
    """CLEAN training-data name; the table and distance map depend on (fold, regime, embedding) only."""
    return f"split30_fold{fold}_{regime_tag(regime, inner_fold)}_{check_embedding_tag(embedding)}"


def model_name(fold: int, regime: str, embedding: str, seed: int, input_dim: int,
               inner_fold: Optional[int] = None) -> str:
    return (f"clean_split30_fold{fold}_{regime_tag(regime, inner_fold)}_{check_embedding_tag(embedding)}"
            f"_seed{seed}_d{input_dim}")


# ---------------------------------------------------------------- embedding farms

@dataclass(frozen=True)
class FarmEntry:
    label: str
    seed: Optional[int]  # None for a base embedding, the mask seed for a masked copy
    sequence_sha256: str  # SHA-256 of the embedded string ('<mask>' tokens included)
    embedding: str
    layer: int
    dim: int
    file_sha256: str


def embedding_farm_digest(file_sha256: Mapping[str, str]) -> str:
    """SHA-256 of the sorted ``label<TAB>file_sha256`` lines (the distance-map helper uses the same rule)."""
    return sha256_lines(sorted(f"{label}\t{digest}" for label, digest in file_sha256.items()))


def _check_farm_entry(entry: FarmEntry, where: str) -> None:
    if not ACCESSION_RE.match(entry.label):
        raise ComparatorError(f"Invalid farm label {entry.label!r} at {where}")
    if not (SHA256_RE.match(entry.sequence_sha256) and SHA256_RE.match(entry.file_sha256)):
        raise ComparatorError(f"Farm entry hashes are not SHA-256 at {where}")
    if entry.dim < 1 or entry.layer < 0 or (entry.seed is not None and entry.seed < 0):
        raise ComparatorError(f"Farm entry seed, layer or width out of range at {where}")
    check_embedding_tag(entry.embedding)


def write_farm_manifest(directory: Path | str, entries: Iterable[FarmEntry]) -> Path:
    """Write-once farm manifest, sorted by label (for embedding producers and synthetic fixtures)."""
    rows = sorted(entries, key=lambda entry: entry.label)
    labels = [entry.label for entry in rows]
    if len(set(labels)) != len(labels):
        raise ComparatorError("Farm manifest labels must be unique")
    for entry in rows:
        _check_farm_entry(entry, entry.label)
    path = Path(directory) / FARM_MANIFEST
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(FARM_MANIFEST_COLUMNS)
        for entry in rows:
            writer.writerow((entry.label, "" if entry.seed is None else entry.seed, entry.sequence_sha256,
                             entry.embedding, entry.layer, entry.dim, entry.file_sha256))
    return path


def read_farm_manifest(path: Path) -> dict[str, FarmEntry]:
    entries: dict[str, FarmEntry] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        if tuple(next(reader, ())) != FARM_MANIFEST_COLUMNS:
            raise ComparatorError(f"{path} needs the tab-separated columns {FARM_MANIFEST_COLUMNS}")
        for line, raw in enumerate(reader, start=2):
            where = f"{path.name}:{line}"
            if len(raw) != len(FARM_MANIFEST_COLUMNS):
                raise ComparatorError(f"Malformed farm-manifest row at {where}")
            label, seed, sequence_sha256, embedding, layer, dim, file_sha256 = raw
            if not ((seed == "" or UNSIGNED_RE.match(seed)) and UNSIGNED_RE.match(layer) and UNSIGNED_RE.match(dim)):
                raise ComparatorError(f"Farm seed, layer or width is not an unsigned integer at {where}")
            entry = FarmEntry(label, None if seed == "" else int(seed), sequence_sha256, embedding, int(layer),
                              int(dim), file_sha256)
            _check_farm_entry(entry, where)
            if label in entries:
                raise ComparatorError(f"Duplicate farm label {label} at {where}")
            entries[label] = entry
    return entries


def link_embedding_farm(esm_out: Path, expected: Mapping[str, tuple[Optional[int], str]], source_dir: Path,
                        forbidden_roots: Sequence[Path], what: str, *, embedding: str,
                        input_dim: int) -> tuple[dict[str, Any], dict[str, str]]:
    """Symlink ``<label>.pt`` by exact name (no listing) once every file matches its farm-manifest row.

    ``expected`` maps each label to (mask seed or None, SHA-256 of the sequence this fit embeds); the
    row must carry that seed, sequence hash, ``embedding`` and ``input_dim``, and the file its hash.
    """
    source_dir = refuse_held_out_path(source_dir, forbidden_roots, what)
    manifest_path = source_dir / FARM_MANIFEST
    if not manifest_path.is_file():
        raise ComparatorError(f"The {what} directory has no {FARM_MANIFEST}: {source_dir}")
    entries = read_farm_manifest(manifest_path)
    missing, wrong, file_sha256 = [], [], {}
    for label, (seed, sequence_sha256) in expected.items():
        entry, source = entries.get(label), source_dir / f"{label}.pt"
        if entry is None or not source.is_file():
            missing.append(label)
            continue
        differing = [field for field, same in (
            ("seed", entry.seed == seed), ("sequence", entry.sequence_sha256 == sequence_sha256),
            ("embedding", entry.embedding == embedding), ("width", entry.dim == input_dim)) if not same]
        if not differing and sha256_file(source) != entry.file_sha256:
            differing.append("file hash")
        if differing:
            wrong.append(f"{label} ({', '.join(differing)})")
            continue
        file_sha256[label] = entry.file_sha256
    if missing:
        raise ComparatorError(f"{len(missing)} {what} files are missing, e.g. {missing[:10]}")
    if wrong:
        raise ComparatorError(f"{len(wrong)} {what} files do not match this fit, e.g. {wrong[:10]}")
    layers = sorted({entries[label].layer for label in file_sha256})
    if len(layers) > 1:
        raise ComparatorError(f"The {what} files mix layers {layers}")
    for label in file_sha256:
        os.symlink((source_dir / f"{label}.pt").resolve(), esm_out / f"{label}.pt")
    return ({"dir": str(source_dir), "manifest_sha256": sha256_file(manifest_path), "n_files": len(file_sha256),
             "embedding": embedding, "layer": layers[0] if layers else None, "dim": int(input_dim),
             "farm_sha256": embedding_farm_digest(file_sha256)}, file_sha256)


def write_farm_links(path: Path, files: Mapping[str, Mapping[str, str]]) -> str:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("kind", "label", "file_sha256"))
        for kind in FARM_KINDS:
            for label in sorted(files[kind]):
                writer.writerow((kind, label, files[kind][label]))
    return sha256_file(path)


def verify_farm_links(work_dir: Path, farm: Mapping[str, Any]) -> dict[str, str]:
    """Re-hashes every linked embedding; the links file and both farm digests must equal the prepare record."""
    path = work_dir / farm["links_file"]
    if not path.is_file() or sha256_file(path) != farm["links_sha256"]:
        raise ComparatorError("The embedding links file changed after prepare")
    files: dict[str, dict[str, str]] = {kind: {} for kind in FARM_KINDS}
    changed = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader)
        for kind, label, digest in reader:
            linked = work_dir / "data" / "esm_data" / f"{label}.pt"
            if not linked.is_file() or sha256_file(linked) != digest:
                changed.append(label)
            files[kind][label] = digest
    if changed:
        raise ComparatorError(f"{len(changed)} linked embeddings changed after prepare, e.g. {changed[:10]}")
    digests = {kind: embedding_farm_digest(files[kind]) for kind in FARM_KINDS}
    if any(digests[kind] != farm[kind]["farm_sha256"] for kind in FARM_KINDS):
        raise ComparatorError("Embedding farm digests differ from the prepare manifest")
    return digests


def verify_distance_map(distance_dir: Path, name: str, table_sha256: str, *, base_farm_sha256: str,
                        input_dim: int, helper_sha256: str = DISTANCE_MAP_HELPER_SHA256) -> dict[str, Any]:
    """Receipt of the distance-map helper; it must match this fit's training table, base farm digest and
    input width, the pinned helper and its fixed hash seed, and every output hash."""
    receipt_path = distance_dir / f"{name}_receipt.json"
    if not receipt_path.is_file():
        raise ComparatorError(f"Distance-map receipt missing: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("training_table_sha256") != table_sha256:
        raise ComparatorError(f"Distance map {receipt_path} was built from another training table")
    if receipt.get("embedding_farm_sha256") != base_farm_sha256:
        raise ComparatorError(f"Distance map {receipt_path} was built from other base embeddings")
    shape = receipt.get("esm_embeddings_shape")
    if not (isinstance(shape, list) and len(shape) == 2 and shape[1] == input_dim):
        raise ComparatorError(f"Distance map {receipt_path} has embedding shape {shape}, not width {input_dim}")
    if receipt.get("helper_sha256") != helper_sha256:
        raise ComparatorError(f"Distance map {receipt_path} was built by another helper version")
    if receipt.get("pythonhashseed") != str(DISTANCE_MAP_HASH_SEED):
        raise ComparatorError(f"Distance map {receipt_path} was built with PYTHONHASHSEED "
                              f"{receipt.get('pythonhashseed')!r}, not {DISTANCE_MAP_HASH_SEED}")
    expected = {"distance_map": f"{name}.pkl", "esm_embeddings": f"{name}_esm.pkl"}
    for key, file_name in expected.items():
        output = receipt.get("outputs", {}).get(key, {})
        path = distance_dir / file_name
        if output.get("file") != file_name or not path.is_file() or sha256_file(path) != output.get("sha256"):
            raise ComparatorError(f"Distance-map output {file_name} does not match its receipt")
    return {"receipt": receipt_path.name, "receipt_sha256": sha256_file(receipt_path),
            "outputs": receipt["outputs"], "pythonhashseed": receipt["pythonhashseed"],
            "helper_sha256": receipt["helper_sha256"], "embedding_farm_sha256": base_farm_sha256,
            "esm_embeddings_shape": shape,
            # pickled tensors embed a process-dependent key; the content hash is reproducible
            "esm_embeddings_content_sha256": receipt.get("esm_embeddings_content_sha256")}


def esmc_embedding_tag(layer: str) -> str:
    """Embedding name of an ESMC farm, e.g. 'esmc_300m_mean_final' or 'esmc_300m_mean_block26'."""
    return check_embedding_tag(f"{ESMC_MODEL_NAME}_mean_{layer}")


def write_esmc_farm(*, cache_dir: Path | str, cache_manifest_sha256: str, layer: str | int,
                    sequence_map: Path | str, out_dir: Path | str, kind: str, seed: Optional[int] = None,
                    masked_manifests: Sequence[Path | str] = (), sequence_map_sha256: Optional[str] = None,
                    forbidden_roots: Sequence[Path] = ()) -> dict[str, Any]:
    """One CLEAN embedding farm (plain 1-D float32 ``<label>.pt`` files) from a sequence-mean cache.

    The cache is the output of ``esmc_chain_means.py sequences``; only its manifest and the files of
    the needed sequences are opened (by computed name, hash-checked). Every farm label must have been
    embedded under that label (the cache manifest's label rows) with the same string, and the cache
    must hold ``layer``; the farm then covers exactly the labels of ``farm_labels``.
    """
    import torch  # only this action needs torch; the module stays importable without it
    from training.global_esm_features import (
        FINAL_LAYER, KIND_SEQUENCE_MEAN, GlobalEsmCacheError, deserialize_payload, load_sequence_mean_vectors,
        normalize_layer, read_manifest, sequence_mean_filename,
    )

    roots = resolve_forbidden_roots(forbidden_roots)
    try:
        layer = normalize_layer(layer)
    except GlobalEsmCacheError as exc:
        raise ComparatorError(str(exc)) from None
    embedding = esmc_embedding_tag(layer)
    out_dir = refuse_inside_repository(refuse_held_out_path(out_dir, roots, "the farm directory"),
                                       "the farm directory")
    _require_new_directory(out_dir, "farm directory")
    cache_dir = refuse_held_out_path(cache_dir, roots, "the ESMC cache")
    sequences, map_record = read_sequence_map_file(sequence_map, roots, sequence_map_sha256)
    expected, manifest_records = farm_labels(sequences, kind=kind, seed=seed, masked_manifests=masked_manifests,
                                             forbidden_roots=roots)
    try:
        manifest = read_manifest(cache_dir, cache_manifest_sha256)
    except GlobalEsmCacheError as exc:
        raise ComparatorError(str(exc)) from None
    identity = {"kind": manifest.get("kind"), "esm_model_name": manifest.get("esm_model_name"),
                "embedding_dim": manifest.get("embedding_dim")}
    if identity != {"kind": KIND_SEQUENCE_MEAN, "esm_model_name": ESMC_MODEL_NAME, "embedding_dim": ESMC_DIM}:
        raise ComparatorError(f"The cache is not a {ESMC_MODEL_NAME} sequence-mean cache of width {ESMC_DIM}: "
                              f"{identity}")
    if layer not in manifest.get("layers", []):
        raise ComparatorError(f"Layer mismatch: the cache holds {manifest.get('layers')}, not {layer!r}")
    # A masked label '<acc>_<j>' names a different string for every mask seed, so one cache over the
    # masked FASTAs of several seeds holds that label several times: match (label, string) pairs.
    rows = manifest.get("labels", [])
    embedded = {(row.get("label"), row.get("sequence_sha256")) for row in rows}
    keys = {label: sha256_bytes(string.encode("utf-8")) for label, (_, string) in expected.items()}
    missing = sorted(label for label, key in keys.items() if (label, key) not in embedded)
    if missing:
        raise ComparatorError(f"{len(missing)} farm labels were not embedded under that label and string in the "
                              f"cache, e.g. {missing[:10]}")
    source_inputs = sorted({str(row.get("input")) for row in rows
                            if keys.get(row.get("label")) == row.get("sequence_sha256")})
    try:
        vectors = load_sequence_mean_vectors(cache_dir, cache_manifest_sha256, sorted(set(keys.values())),
                                             expected_layer=layer, expected_dim=ESMC_DIM,
                                             expected_model_name=ESMC_MODEL_NAME)
        # Every payload of a cache comes from one model; the first one gives the block count.
        sample = sorted(set(keys.values()))[0]
        payload = deserialize_payload((cache_dir / sequence_mean_filename(sample, layer, ESMC_MODEL_NAME))
                                      .read_bytes(), origin=sample)
    except (GlobalEsmCacheError, OSError) as exc:
        raise ComparatorError(f"ESMC cache refused: {exc}") from None
    n_blocks = (payload.get("source") or {}).get("n_layers")
    if not isinstance(n_blocks, int) or n_blocks < 1:
        raise ComparatorError("The cache payloads do not record the model's block count")
    layer_number = n_blocks if layer == FINAL_LAYER else int(payload["layer_index_zero_based"])

    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for label in sorted(expected):
        vector = vectors[keys[label]]
        if (vector.dim() != 1 or vector.dtype != torch.float32 or vector.numel() != ESMC_DIM
                or not bool(torch.isfinite(vector).all())):
            raise ComparatorError(f"The {label} vector is not a finite 1-D float32 tensor of width {ESMC_DIM}")
        buffer = io.BytesIO()  # in-memory save: the bytes do not depend on the file name
        torch.save(vector.detach().clone().contiguous(), buffer)
        path = out_dir / f"{label}.pt"
        with path.open("xb") as handle:
            handle.write(buffer.getvalue())
        entries.append(FarmEntry(label, expected[label][0], keys[label], embedding, layer_number, ESMC_DIM,
                                 sha256_bytes(buffer.getvalue())))
    manifest_path = write_farm_manifest(out_dir, entries)
    receipt = {
        "schema_version": SCHEMA_VERSION, "action": "esmc-farm", "kind": kind,
        "seed": None if seed is None else int(seed), "embedding": embedding, "layer": layer,
        "farm_layer": layer_number, "dim": ESMC_DIM, "n_files": len(entries),
        "farm_layer_rule": ("block mean: the zero-based block index (one below fair-esm's layer number); final "
                            "normalized mean: the block count"),
        "file_format": "plain 1-D float32 tensor (torch.save); the upstream loader passes it through",
        "source_cache": {"dir": str(cache_dir), "manifest_sha256": str(cache_manifest_sha256).lower(),
                         "model_blocks": n_blocks, "n_unique_sequences": len(set(keys.values())),
                         "inputs_used": source_inputs},
        "sequence_map": map_record, "masked_manifests": manifest_records,
        "farm_manifest_sha256": sha256_file(manifest_path),
        "farm_sha256": embedding_farm_digest({entry.label: entry.file_sha256 for entry in entries}),
        "torch_version": torch.__version__,
        "forbidden_roots": [str(root) for root in roots],
        "runner": Path(__file__).name, "runner_sha256": sha256_file(Path(__file__)),
    }
    _write_json_once(out_dir / FARM_RECEIPT, receipt)
    return receipt


def prepare_fit(*, official_train_csv: Path, fold: int, regime: str, seed: int, work_dir: Path,
                accession_allowlist: Optional[Path] = None, base_esm_dir: Optional[Path] = None,
                masked_esm_dir: Optional[Path] = None, distance_map_dir: Optional[Path] = None,
                input_dim: int = DEFAULT_INPUT_DIM, embedding: str = DEFAULT_EMBEDDING,
                distance_helper_sha256: str = DISTANCE_MAP_HELPER_SHA256, inner_fold: Optional[int] = None,
                accession_exclusions: Optional[Path] = None,
                forbidden_roots: Sequence[Path] = ()) -> dict[str, Any]:
    """Write one fit work directory in the CLEAN data layout and its manifest."""
    regime_tag(regime, inner_fold)  # known regime; an inner fold exactly for the inner regimes
    if (regime in ("r2", "r2_inner")) != (accession_allowlist is not None):
        raise ComparatorError("Regimes r2 and r2_inner need an accession allowlist; r1 and r1_inner use the full "
                              "training file")
    if (regime in INNER_REGIMES) != (accession_exclusions is not None):
        raise ComparatorError("Regimes r1_inner and r2_inner need the inner fold's exclusions list; r1 and r2 take "
                              "none")
    if (base_esm_dir is None) != (masked_esm_dir is None):
        raise ComparatorError("Pass both the base and the masked embedding directories, or neither")
    if distance_map_dir is not None and base_esm_dir is None:
        raise ComparatorError("A linked distance map needs the embedding farm it was built from; pass "
                              "--base-esm-dir and --masked-esm-dir")
    if int(seed) < 0 or int(input_dim) < 1:
        raise ComparatorError("seed must be non-negative and input_dim positive")
    name = training_name(fold, regime, embedding, inner_fold)
    roots = resolve_forbidden_roots(forbidden_roots)
    work_dir = refuse_inside_repository(refuse_held_out_path(work_dir, roots, "the work directory"),
                                        "the work directory")
    if work_dir.exists() and any(work_dir.iterdir()):
        raise ComparatorError(f"Work directory is not empty: {work_dir}")

    rows, official = read_official_training_rows(official_train_csv, fold, roots)
    allowlist = exclusions = None
    excluded: dict[str, str] = {}
    if accession_exclusions is not None:
        excluded, exclusions = read_inner_exclusions(accession_exclusions, fold, int(inner_fold), roots)
        unknown = sorted(set(excluded) - {row.accession for row in rows})
        if unknown:
            raise ComparatorError(f"{len(unknown)} excluded accessions are not in the official training file, "
                                  f"e.g. {unknown[:10]}")
    if accession_allowlist is not None:
        accessions, allowlist = read_accession_allowlist(accession_allowlist, roots)
        overlap = sorted(set(accessions) & set(excluded))
        if overlap:
            raise ComparatorError(f"{len(overlap)} allowlisted accessions are on inner fold {inner_fold}'s exclusions "
                                  f"list (inner-validation proteins or their D3 neighbours), e.g. {overlap[:10]}")
        rows = select_rows(rows, accessions)
    elif excluded:
        rows = [row for row in rows if row.accession not in excluded]

    data_dir = work_dir / "data"
    esm_out = data_dir / "esm_data"
    mask_dir = work_dir / "masks"
    for directory in (esm_out, mask_dir):
        directory.mkdir(parents=True, exist_ok=True)
    table_sha256 = write_training_table(rows, data_dir / f"{name}.csv")
    copies, masks = write_masked_copies(rows, seed, mask_dir / f"{name}_seed{seed}_masked.fasta",
                                        mask_dir / f"{name}_seed{seed}_masked_manifest.tsv")

    farm: dict[str, Any] = {"complete": False}
    if base_esm_dir is not None:
        farm_options = {"embedding": embedding, "input_dim": input_dim}
        base, base_files = link_embedding_farm(
            esm_out, {row.accession: (None, sha256_bytes(row.sequence.encode())) for row in rows}, base_esm_dir,
            roots, "base embedding", **farm_options)
        masked, masked_files = link_embedding_farm(
            esm_out, {copy.label: (int(seed), sha256_bytes(copy.sequence.encode())) for copy in copies},
            masked_esm_dir, roots, "masked-copy embedding", **farm_options)
        if masked["layer"] is not None and masked["layer"] != base["layer"]:
            raise ComparatorError(f"Base embeddings come from layer {base['layer']}, masked copies from "
                                  f"layer {masked['layer']}")
        farm = {"complete": True, "base": base, "masked": masked, "links_file": FARM_LINKS,
                "links_sha256": write_farm_links(work_dir / FARM_LINKS, {"base": base_files,
                                                                          "masked": masked_files})}
    distance_map = None
    if distance_map_dir is not None:
        source = refuse_held_out_path(distance_map_dir, roots, "a distance-map directory")
        distance_map = verify_distance_map(source, name, table_sha256, base_farm_sha256=farm["base"]["farm_sha256"],
                                           input_dim=input_dim, helper_sha256=distance_helper_sha256)
        target = data_dir / "distance_map"
        target.mkdir()
        for file_name in (f"{name}.pkl", f"{name}_esm.pkl"):
            os.symlink((source / file_name).resolve(), target / file_name)
        shutil.copyfile(source / f"{name}_receipt.json", target / f"{name}_receipt.json")
        distance_map["source_dir"] = str(source)

    ec_labels = {ec for row in rows for ec in row.ec_numbers}
    manifest = {
        "schema_version": SCHEMA_VERSION, "fold": fold, "regime": regime, "inner_fold": inner_fold,
        "seed": int(seed), "input_dim": int(input_dim), "embedding": embedding, "training_name": name,
        "official_training_file": official, "accession_allowlist": allowlist, "accession_exclusions": exclusions,
        "training_table": {"file": f"data/{name}.csv", "sha256": table_sha256, "n_proteins": len(rows),
                           "n_ec_labels": len(ec_labels),
                           "n_ec_labels_without_dash": sum("-" not in ec for ec in ec_labels)},
        "masked_copies": masks, "embedding_farm": farm, "distance_map": distance_map,
        "forbidden_roots": [str(root) for root in roots],
        "runner": Path(__file__).name, "runner_sha256": sha256_file(Path(__file__)),
    }
    _write_json_once(work_dir / PREPARE_MANIFEST, manifest)
    return manifest


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


# ---------------------------------------------------------------- launch

def _git_head(clone: Path) -> str:
    result = subprocess.run(["git", "-C", str(clone), "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=False)
    if result.returncode:
        raise ComparatorError(f"git rev-parse failed in {clone}: {result.stderr.strip()}")
    return result.stdout.strip()


def verify_clean_clone(clone: Path, expected_files: Optional[Mapping[str, str]] = None) -> dict[str, Any]:
    """Commit and file hashes of the CLEAN checkout; ``app/data`` and ``app/results`` must be absent."""
    clone = refuse_inside_repository(clone, "the CLEAN checkout")
    for part in HELD_OUT_CLONE_DIRS:
        if os.path.lexists(clone / part):
            raise ComparatorError(f"The CLEAN checkout contains {part}; use a sparse checkout without it")
    head = _git_head(clone)
    if head != CLEAN_COMMIT:
        raise ComparatorError(f"CLEAN checkout is at {head}, expected {CLEAN_COMMIT}")
    expected = UPSTREAM_FILE_SHA256 if expected_files is None else expected_files
    mismatched = [rel for rel, digest in expected.items()
                  if not (clone / rel).is_file() or sha256_file(clone / rel) != digest]
    if mismatched:
        raise ComparatorError(f"CLEAN files differ from commit {CLEAN_COMMIT}: {mismatched}")
    return {"clone": str(clone), "commit": head, "files_verified": len(expected)}


def verify_pinned_file(path: Path, expected_sha256: str, what: str) -> dict[str, Any]:
    path = refuse_inside_repository(path, what)
    if not SHA256_RE.match(expected_sha256):
        raise ComparatorError(f"Expected SHA-256 for {what} is not 64 hex characters")
    if not path.is_file():
        raise ComparatorError(f"{what} not found: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ComparatorError(f"{what} SHA-256 {actual} differs from the expected {expected_sha256}")
    return {"path": str(path), "sha256": actual}


def build_clean_env(*, clean_clone: Path, hash_seed: int, threads: int, forbidden_roots: Sequence[Path],
                    base_env: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    """Minimal child environment: fixed hash seed and threads, CPU only, no user site, guard roots set."""
    if threads < 1:
        raise ComparatorError("threads must be at least 1")
    if not forbidden_roots:
        raise ComparatorError("A CLEAN process needs forbidden-read roots")
    base = os.environ if base_env is None else base_env
    env = {key: base[key] for key in PASSTHROUGH_ENV if key in base}
    env.update({
        "PYTHONHASHSEED": str(int(hash_seed)), "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads),
        "MKL_THREADING_LAYER": "GNU", "PYTHONPATH": str(Path(clean_clone).resolve() / CLEAN_SOURCE_DIR),
        FORBIDDEN_READ_ROOTS_ENV: forbidden_roots_environment(forbidden_roots),
    })
    return env


def guarded_command(clean_python: Path, script: Path, script_args: Sequence[str]) -> list[str]:
    return [str(clean_python), "-c", GUARDED_LAUNCH, str(ACCESS_GUARD_FILE), str(script), *script_args]


def build_train_args(*, name: str, model: str, epochs: int, adaptive_rate: int, seed: int,
                     input_dim: int) -> list[str]:
    check_epochs(epochs, adaptive_rate)
    recipe = PUBLISHED_RECIPE
    return ["--training_data", name, "--model_name", model, "--epoch", str(epochs),
            "--learning_rate", repr(float(recipe["learning_rate"])), "--hidden_dim", str(recipe["hidden_dim"]),
            "--out_dim", str(recipe["out_dim"]), "--adaptive_rate", str(adaptive_rate),
            "--seed", str(int(seed)), "--input_dim", str(int(input_dim))]


def _run(command: Sequence[str], *, cwd: Path, env: Mapping[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as log:
        return subprocess.run(list(command), cwd=str(cwd), env=dict(env), stdout=log, stderr=subprocess.STDOUT,
                              check=False).returncode


def verify_fit_outputs(work_dir: Path, *, model: str, seed: int, epochs: int, adaptive_rate: int,
                       input_dim: int) -> dict[str, Any]:
    """Exactly the final weights, loss log and receipt remain, and the receipt matches the request."""
    model_dir = work_dir / "data" / "model"
    expected = {f"{model}.pth", f"{model}_loss_log.jsonl", f"{model}_receipt.json"}
    present = set(os.listdir(model_dir)) if model_dir.is_dir() else set()
    if present != expected:
        raise ComparatorError(f"Fit outputs differ from {sorted(expected)}: {sorted(present)}")
    receipt = json.loads((model_dir / f"{model}_receipt.json").read_text(encoding="utf-8"))
    wanted = {"seed": seed, "epochs": epochs, "adaptive_rate": adaptive_rate, "input_dim": input_dim,
              "model_name": model}
    wrong = {key: receipt.get(key) for key, value in wanted.items() if receipt.get(key) != value}
    if wrong:
        raise ComparatorError(f"Training receipt differs from the request: {wrong}")
    if (receipt.get("launch_env") or {}).get("PYTHONHASHSEED") != str(seed):
        raise ComparatorError("Training receipt does not show PYTHONHASHSEED equal to the seed at launch")
    if sha256_file(model_dir / f"{model}.pth") != receipt.get("final_weights_sha256"):
        raise ComparatorError("Final weights do not match the training receipt")
    losses = [json.loads(line) for line in (model_dir / f"{model}_loss_log.jsonl").read_text().splitlines()]
    if [entry.get("epoch") for entry in losses] != list(range(1, epochs + 1)):
        raise ComparatorError("Loss log does not cover every epoch once, in order")
    if not all(isinstance(entry.get("train_loss"), float) and math.isfinite(entry["train_loss"])
               for entry in losses):
        raise ComparatorError("Loss log has a missing or non-finite loss")
    return receipt


def run_fit(*, work_dir: Path, clean_python: Path, clean_clone: Path, training_script: Path,
            distance_helper: Path, training_script_sha256: str = TRAINING_SCRIPT_SHA256,
            distance_helper_sha256: str = DISTANCE_MAP_HELPER_SHA256,
            epochs: int = PUBLISHED_RECIPE["epochs"], adaptive_rate: int = PUBLISHED_RECIPE["adaptive_rate"],
            threads: int = DEFAULT_THREADS, extra_forbidden_roots: Sequence[Path] = (),
            dry_run: bool = False) -> dict[str, Any]:
    """Distance map (unless linked), then training, each in a fresh guarded CLEAN process."""
    check_epochs(epochs, adaptive_rate)
    work_dir = refuse_inside_repository(work_dir, "the work directory")
    manifest = json.loads((work_dir / PREPARE_MANIFEST).read_text(encoding="utf-8"))
    name, seed, input_dim = manifest["training_name"], int(manifest["seed"]), int(manifest["input_dim"])
    table = work_dir / manifest["training_table"]["file"]
    if sha256_file(table) != manifest["training_table"]["sha256"]:
        raise ComparatorError("Training table changed after prepare")
    farm = manifest["embedding_farm"]
    if not farm.get("complete"):
        raise ComparatorError("The work directory has no complete embedding farm; rerun prepare with "
                              "--base-esm-dir and --masked-esm-dir")
    inner_fold = manifest.get("inner_fold")
    if name != training_name(manifest["fold"], manifest["regime"], manifest["embedding"], inner_fold):
        raise ComparatorError(f"Training name {name} does not match the fold, regime and embedding")
    if any(os.path.lexists(work_dir / part) for part in (FIT_MANIFEST, "logs", "data/model")):
        raise ComparatorError(f"A fit already ran in {work_dir}; use a new work directory")
    farm_digests = verify_farm_links(work_dir, farm)

    clone = verify_clean_clone(clean_clone)
    scripts = {"training_script": verify_pinned_file(training_script, training_script_sha256, "training script"),
               "distance_helper": verify_pinned_file(distance_helper, distance_helper_sha256,
                                                     "distance-map helper"),
               "access_guard": {"path": str(ACCESS_GUARD_FILE), "sha256": sha256_file(ACCESS_GUARD_FILE)}}
    roots = resolve_forbidden_roots(list(extra_forbidden_roots) + manifest.get("forbidden_roots", []),
                                    clean_clone=Path(clean_clone))
    model = model_name(manifest["fold"], manifest["regime"], manifest["embedding"], seed, input_dim, inner_fold)
    steps = []
    if not (work_dir / "data" / "distance_map" / f"{name}_receipt.json").is_file():
        steps.append({"step": "distance_map", "hash_seed": DISTANCE_MAP_HASH_SEED,
                      "command": guarded_command(clean_python, distance_helper, ["--training_data", name])})
    steps.append({"step": "train", "hash_seed": seed,
                  "command": guarded_command(clean_python, training_script, build_train_args(
                      name=name, model=model, epochs=epochs, adaptive_rate=adaptive_rate, seed=seed,
                      input_dim=input_dim))})
    plan = {"work_dir": str(work_dir), "model_name": model, "threads": threads, "steps": steps}
    if dry_run:
        return plan

    for step in steps:
        env = build_clean_env(clean_clone=Path(clean_clone), hash_seed=step["hash_seed"], threads=threads,
                              forbidden_roots=roots)
        log_path = work_dir / "logs" / f"{step['step']}.log"
        code = _run(step["command"], cwd=work_dir, env=env, log_path=log_path)
        step["returncode"] = code
        step["log"] = str(log_path.relative_to(work_dir))
        step["env"] = {key: value for key, value in env.items() if key not in PASSTHROUGH_ENV}
        if code:
            raise ComparatorError(f"CLEAN {step['step']} step failed with exit code {code}; see {log_path}")
    distance_map = verify_distance_map(work_dir / "data" / "distance_map", name, manifest["training_table"]["sha256"],
                                       base_farm_sha256=farm["base"]["farm_sha256"], input_dim=input_dim,
                                       helper_sha256=distance_helper_sha256)
    receipt = verify_fit_outputs(work_dir, model=model, seed=seed, epochs=epochs, adaptive_rate=adaptive_rate,
                                 input_dim=input_dim)
    fit = {
        "schema_version": SCHEMA_VERSION, "prepare_manifest_sha256": sha256_file(work_dir / PREPARE_MANIFEST),
        "clean_clone": clone, "scripts": scripts, "clean_python": str(clean_python), "steps": steps,
        "forbidden_roots": [str(root) for root in roots], "embedding_farm_verified": farm_digests,
        "distance_map": distance_map,
        "recipe": {**PUBLISHED_RECIPE, "epochs": epochs, "adaptive_rate": adaptive_rate},
        "training_receipt": receipt,
        "training_receipt_sha256": sha256_file(work_dir / "data" / "model" / f"{model}_receipt.json"),
        "runner_sha256": sha256_file(Path(__file__)),
    }
    _write_json_once(work_dir / FIT_MANIFEST, fit)
    return fit


# ---------------------------------------------------------------- contract C4 and projection

@dataclass(frozen=True)
class DistanceExport:
    distances: np.ndarray
    query_ids: tuple[str, ...]
    ec_labels: tuple[str, ...]
    receipt: dict


def c4_receipt_path(npz_path: Path | str) -> Path:
    npz_path = Path(npz_path)
    return npz_path.with_name(f"{npz_path.stem}_receipt.json")


def _unique_strings(values: Sequence[str], what: str) -> tuple[str, ...]:
    values = tuple(str(value) for value in values)
    if not values or any(not value or value != value.strip() for value in values):
        raise ComparatorError(f"{what} must be non-empty strings without surrounding whitespace")
    if len(set(values)) != len(values):
        raise ComparatorError(f"{what} contain duplicates")
    return values


def _check_export_arrays(distances: Any, query_ids: Sequence[str],
                         ec_labels: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    query_ids = _unique_strings(query_ids, "query_ids")
    ec_labels = _unique_strings(ec_labels, "ec_labels")
    for label in ec_labels:
        ec_class_of_label(label)
    if not isinstance(distances, np.ndarray) or distances.dtype != np.float32:
        raise ComparatorError("distances must be a float32 numpy array")
    if distances.shape != (len(query_ids), len(ec_labels)):
        raise ComparatorError(f"distances shape {distances.shape} != ({len(query_ids)}, {len(ec_labels)})")
    if not np.all(np.isfinite(distances)) or np.any(distances < 0):
        raise ComparatorError("distances must be finite and non-negative")
    return query_ids, ec_labels


def _check_metadata(metadata: Mapping[str, Any]) -> None:
    missing = [field for field in C4_METADATA_FIELDS if field not in metadata]
    if missing:
        raise ComparatorError(f"Export metadata lacks {missing}")
    if metadata["mode"] not in C4_MODES:
        raise ComparatorError(f"Export mode {metadata['mode']!r} not in {C4_MODES}")
    if metadata["regime"] not in REGIMES:
        raise ComparatorError(f"Export regime {metadata['regime']!r} not in {REGIMES}")
    for field in ("fold", "seed", "input_dim"):
        if not isinstance(metadata[field], int) or isinstance(metadata[field], bool):
            raise ComparatorError(f"Export {field} must be an integer")
    if not 0 <= metadata["fold"] < N_FOLDS:
        raise ComparatorError(f"Export fold {metadata['fold']} outside 0..{N_FOLDS - 1}")
    for field in C4_SHA256_FIELDS:
        if not SHA256_RE.match(str(metadata[field])):
            raise ComparatorError(f"Export {field} is not a SHA-256")
    if not str(metadata["embedding"]).strip():
        raise ComparatorError("Export embedding must be named")
    inner_fold = metadata.get("inner_fold")
    regime_tag(metadata["regime"], inner_fold)  # an inner fold exactly for the inner regimes
    if inner_fold is not None and metadata["mode"] != "validation":
        raise ComparatorError(f"Regime {metadata['regime']} is a development arm; it has validation exports only")
    in_sample = metadata.get("n_in_sample_queries")
    if in_sample is not None and (isinstance(in_sample, bool) or not isinstance(in_sample, int) or in_sample < 0
                                  or (metadata["mode"] == "test" and in_sample)):
        raise ComparatorError("Export n_in_sample_queries must be a non-negative integer, 0 in test mode")
    binding = [field for field in C4_TEST_BINDING_FIELDS if metadata.get(field) is not None]
    if binding:
        if metadata["mode"] != "test" or len(binding) != len(C4_TEST_BINDING_FIELDS):
            raise ComparatorError(f"The Freeze-2 binding {C4_TEST_BINDING_FIELDS} belongs to test exports, all "
                                  "fields together")
        if not all(SHA256_RE.match(str(metadata[field])) for field in C4_TEST_BINDING_FIELDS
                   if field != "test_pass_output_root"):
            raise ComparatorError("Export Freeze-2 binding hashes are not SHA-256")
        if not Path(str(metadata["test_pass_output_root"])).is_absolute():
            raise ComparatorError("Export test_pass_output_root must be an absolute path")


def check_test_export_binding(receipt: Mapping[str, Any], *, freeze2_sha256: str,
                              test_pass_output_root: Path | str, runner_sha256: Optional[str] = None,
                              export_script_sha256: Optional[str] = None) -> None:
    """A test export must carry the binding of the scored Freeze 2: its manifest SHA-256 and output
    root, the runner it pins (when given) and the export script (default: the pinned one). For
    consumers of C4 such as the scorer."""
    export_script_sha256 = EXPORT_SCRIPT_SHA256 if export_script_sha256 is None else export_script_sha256
    if receipt.get("mode") != "test":
        raise ComparatorError(f"Not a test export (mode {receipt.get('mode')!r})")
    missing = [field for field in C4_TEST_BINDING_FIELDS if receipt.get(field) is None]
    if missing:
        raise ComparatorError(f"The test export carries no Freeze-2 binding {missing}")
    wrong = []
    if receipt["freeze2_manifest_sha256"] != str(freeze2_sha256).strip().lower():
        wrong.append("freeze2_manifest_sha256")
    if Path(str(receipt["test_pass_output_root"])).resolve() != Path(test_pass_output_root).resolve():
        wrong.append("test_pass_output_root")
    if runner_sha256 is not None and receipt["runner_sha256"] != str(runner_sha256).strip().lower():
        wrong.append("runner_sha256")
    if receipt["export_script_sha256"] != export_script_sha256:
        wrong.append("export_script_sha256")
    if wrong:
        raise ComparatorError(f"The test export belongs to another Freeze 2 or code: {wrong}")


def write_distance_export(npz_path: Path | str, distances: np.ndarray, query_ids: Sequence[str],
                          ec_labels: Sequence[str], metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Write-once C4 export (npz plus receipt); returns the receipt."""
    npz_path = Path(npz_path)
    if npz_path.suffix != ".npz":
        raise ComparatorError("A C4 export file must end in .npz")
    query_ids, ec_labels = _check_export_arrays(distances, query_ids, ec_labels)
    _check_metadata(metadata)
    receipt_path = c4_receipt_path(npz_path)
    for path in (npz_path, receipt_path):
        if os.path.lexists(path):
            raise ComparatorError(f"C4 export file already exists (write-once): {path}")
    with npz_path.open("xb") as handle:
        np.savez(handle, distances=distances, query_ids=np.array(query_ids, dtype=str),
                 ec_labels=np.array(ec_labels, dtype=str))
    receipt = {
        **{field: metadata[field] for field in C4_METADATA_FIELDS},
        **{field: metadata[field] for field in C4_OPTIONAL_FIELDS if metadata.get(field) is not None},
        "contract": C4_CONTRACT, "contract_version": C4_VERSION, "npz_file": npz_path.name,
        "npz_sha256": sha256_file(npz_path), "n_query": len(query_ids), "n_ec": len(ec_labels),
        "distance_dtype": "float32", "distance_rule": C4_DISTANCE_RULE,
        "query_ids_sha256": sha256_lines(query_ids), "ec_labels_sha256": sha256_lines(ec_labels),
        "projection_rule": PROJECTION_RULE, "tie_tolerance": TIE_TOLERANCE, "ensemble_rule": ENSEMBLE_RULE,
    }
    _write_json_once(receipt_path, receipt)
    return receipt


def read_distance_export(npz_path: Path | str, receipt_path: Optional[Path | str] = None) -> DistanceExport:
    """Validated C4 export; refuses extra or missing arrays, hash mismatches and malformed contents."""
    npz_path = Path(npz_path)
    receipt_path = c4_receipt_path(npz_path) if receipt_path is None else Path(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("contract") != C4_CONTRACT or receipt.get("contract_version") != C4_VERSION:
        raise ComparatorError(f"{receipt_path} is not a {C4_CONTRACT} v{C4_VERSION} receipt")
    _check_metadata(receipt)
    if receipt.get("npz_file") != npz_path.name or receipt.get("npz_sha256") != sha256_file(npz_path):
        raise ComparatorError(f"{npz_path.name} does not match its receipt")
    with np.load(npz_path, allow_pickle=False) as payload:
        if set(payload.files) != set(C4_ARRAYS):
            raise ComparatorError(f"C4 arrays must be exactly {C4_ARRAYS}, got {sorted(payload.files)}")
        distances = payload["distances"]
        query_array, label_array = payload["query_ids"], payload["ec_labels"]
    if query_array.dtype.kind != "U" or label_array.dtype.kind != "U":
        raise ComparatorError("query_ids and ec_labels must be unicode arrays")
    query_ids, ec_labels = _check_export_arrays(distances, query_array.tolist(), label_array.tolist())
    checks = {"n_query": len(query_ids), "n_ec": len(ec_labels), "query_ids_sha256": sha256_lines(query_ids),
              "ec_labels_sha256": sha256_lines(ec_labels)}
    wrong = sorted(key for key, value in checks.items() if receipt.get(key) != value)
    if wrong:
        raise ComparatorError(f"{npz_path.name} differs from its receipt in {wrong}")
    return DistanceExport(distances, query_ids, ec_labels, receipt)


@dataclass(frozen=True)
class EcClassProjection:
    query_ids: tuple[str, ...]
    nearest_ec: tuple[str, ...]
    predicted_class: tuple[str, ...]  # TIE_CLASS for a tie
    tie: np.ndarray
    min_distance: np.ndarray

    @property
    def n_ties(self) -> int:
        return int(self.tie.sum())


def project_ec_classes(distances: np.ndarray, query_ids: Sequence[str], ec_labels: Sequence[str],
                       tolerance: float = TIE_TOLERANCE) -> EcClassProjection:
    """Nearest EC centre, then its EC class; a near tie between different classes counts as wrong."""
    values = np.asarray(distances, dtype=np.float64)
    if values.shape != (len(query_ids), len(ec_labels)) or not np.all(np.isfinite(values)):
        raise ComparatorError("distances must be finite with shape [n_query, n_ec]")
    classes = np.array([ec_class_of_label(label) for label in ec_labels])
    nearest = values.argmin(axis=1)
    minimum = values[np.arange(len(query_ids)), nearest]
    near = values <= minimum[:, None] + tolerance
    tie = np.array([len(set(classes[row])) > 1 for row in near], dtype=bool)
    predicted = tuple(TIE_CLASS if tied else classes[index] for tied, index in zip(tie, nearest))
    return EcClassProjection(tuple(query_ids), tuple(ec_labels[index] for index in nearest), predicted, tie,
                             minimum)


def projection_correct(projection: EcClassProjection, true_class: Mapping[str, str]) -> np.ndarray:
    """Per-query correctness; a tie is never correct."""
    truth = [str(true_class[query]) for query in projection.query_ids]
    return np.array([not tied and predicted == expected for predicted, tied, expected
                     in zip(projection.predicted_class, projection.tie, truth)], dtype=bool)


def ensemble_distances(exports: Sequence[DistanceExport]) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    """Mean distance per (query, EC label) over the seed exports of one fold, regime and embedding."""
    if len(exports) < 2:
        raise ComparatorError("A seed ensemble needs at least two exports")
    first = exports[0]
    for export in exports[1:]:
        differing = [key for key in ENSEMBLE_SHARED_FIELDS if export.receipt.get(key) != first.receipt.get(key)]
        if differing:
            raise ComparatorError(f"Ensemble exports differ in {differing}")
    seeds = [export.receipt["seed"] for export in exports]
    if len(set(seeds)) != len(seeds):
        raise ComparatorError(f"Ensemble exports repeat a seed: {seeds}")
    total = np.zeros(first.distances.shape, dtype=np.float64)
    for export in exports:
        if set(export.query_ids) != set(first.query_ids) or set(export.ec_labels) != set(first.ec_labels):
            raise ComparatorError(f"Seed {export.receipt['seed']} covers other queries or EC labels")
        row_of = {query: index for index, query in enumerate(export.query_ids)}
        column_of = {label: index for index, label in enumerate(export.ec_labels)}
        rows = [row_of[query] for query in first.query_ids]
        columns = [column_of[label] for label in first.ec_labels]
        total += export.distances.astype(np.float64)[np.ix_(rows, columns)]
    return total / len(exports), first.query_ids, first.ec_labels


def project_ensemble(exports: Sequence[DistanceExport]) -> EcClassProjection:
    distances, query_ids, ec_labels = ensemble_distances(exports)
    return project_ec_classes(distances, query_ids, ec_labels)


# ---------------------------------------------------------------- distance export of a fit

def export_side_files(npz_path: Path | str) -> dict[str, Path]:
    npz_path = Path(npz_path)
    return {key: npz_path.with_name(f"{npz_path.stem}{suffix}") for key, suffix in EXPORT_SIDE_SUFFIXES.items()}


def read_freeze2_manifest(path: Path | str, expected_sha256: str) -> dict[str, Any]:
    """The Freeze-2 manifest, with the rules of the EC prediction export's reader."""
    path = Path(path)
    if not path.is_file():
        raise ComparatorError(f"Freeze-2 manifest not found: {path}")
    data = path.read_bytes()
    actual = sha256_bytes(data)
    if actual != str(expected_sha256).strip().lower():
        raise ComparatorError(f"Freeze-2 manifest {path} has SHA-256 {actual}, expected {expected_sha256}")
    manifest = json.loads(data.decode("utf-8"))
    if not (isinstance(manifest, dict) and manifest.get("schema_version") == FREEZE2_SCHEMA_VERSION
            and isinstance(manifest.get("checkpoint_sha256s"), list) and isinstance(manifest.get("code_sha256"), dict)
            and isinstance(manifest.get("test_pass_output_root"), str)):
        raise ComparatorError("Freeze-2 manifest lacks schema_version 1, checkpoint_sha256s, code_sha256 or "
                              "test_pass_output_root")
    if not Path(manifest["test_pass_output_root"]).is_absolute():
        raise ComparatorError("Freeze-2 manifest test_pass_output_root must be an absolute path")
    return manifest


def verify_freeze2_runner_pins(manifest: Mapping[str, Any]) -> dict[str, str]:
    """This runner (``additional_code_sha256``) and the repository files it executes (``code_sha256``)
    must equal their Freeze-2 pins. It needs no torch, so the export script repeats it in the CLEAN
    environment; ``run_export`` adds the full check of ``code_sha256``."""
    additional = manifest.get("additional_code_sha256")
    if not isinstance(additional, dict):
        raise ComparatorError("Freeze-2 manifest lacks additional_code_sha256 (where this runner is pinned)")
    live = {RUNNER_KEY: Path(__file__).resolve(), **{path: PROJECT_ROOT / path for path in RUNNER_SOURCE_PATHS}}
    pins = {RUNNER_KEY: additional.get(RUNNER_KEY), **{path: manifest["code_sha256"].get(path)
                                                       for path in RUNNER_SOURCE_PATHS}}
    missing = sorted(key for key, pin in pins.items() if pin is None)
    if missing:
        raise ComparatorError(f"The Freeze-2 manifest does not pin {missing}")
    actual = {key: sha256_file(path) for key, path in live.items()}
    differing = sorted(key for key, pin in pins.items() if actual[key] != str(pin).strip().lower())
    if differing:
        raise ComparatorError(f"Live code differs from the Freeze-2 manifest: {differing}")
    return actual


def verify_freeze2_code(manifest: Mapping[str, Any]) -> dict[str, str]:
    """``code_sha256`` by the EC prediction export's rule: every required file pinned, every pin live.

    That module imports torch and torch_geometric, so this runs in the DeepMzyme environment only.
    """
    from training.ec_prediction_export import EcPredictionExportError, _verify_freeze2_code

    try:
        return _verify_freeze2_code(manifest["code_sha256"])
    except EcPredictionExportError as exc:
        raise ComparatorError(str(exc)) from None


@dataclass(frozen=True)
class ExportPlan:
    """Checked inputs of one fit's distance export (shared by the public action and the export script)."""
    work_dir: Path
    out: Path
    mode: str
    training_name: str
    model_name: str
    table_path: Path
    weights_path: Path
    train_labels: tuple[str, ...]  # the training table's entries (the centres' inputs)
    query_ids: tuple[str, ...]
    query_files: dict[str, Path]  # query ID -> resolved embedding file
    hidden_dim: int
    out_dim: int
    metadata: dict[str, Any]  # the C4 metadata the export must carry
    inputs: dict[str, Any]  # hashes and records for receipts
    freeze2_manifest: Optional[dict] = None  # the parsed manifest (test mode)

    @property
    def seed(self) -> int:
        return int(self.metadata["seed"])

    @property
    def input_dim(self) -> int:
        return int(self.metadata["input_dim"])


def _read_json(path: Path, what: str) -> dict[str, Any]:
    if not path.is_file():
        raise ComparatorError(f"{what} not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _table_entries(path: Path) -> tuple[str, ...]:
    """Entry column of a CLEAN training table written by ``prepare`` (header first)."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        if tuple(next(reader, ())) != TABLE_HEADER:
            raise ComparatorError(f"{path.name} is not a CLEAN training table")
        return tuple(raw[0] for raw in reader if raw)


def table_ec_order(path: Path) -> tuple[str, ...]:
    """EC numbers of a CLEAN training table in upstream centre order (first appearance, row by row)."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        if tuple(next(reader, ())) != TABLE_HEADER:
            raise ComparatorError(f"{path.name} is not a CLEAN training table")
        return tuple(dict.fromkeys(ec for raw in reader if raw for ec in raw[1].split(";")))


def plan_distance_export(*, work_dir: Path | str, query_ids_file: Path | str, query_esm_dir: Path | str,
                         out: Path | str, mode: str, forbidden_roots: Sequence[Path],
                         freeze2_manifest: Optional[Path | str] = None, freeze2_sha256: Optional[str] = None,
                         export_script_sha256: Optional[str] = None,
                         allow_in_sample: bool = False) -> ExportPlan:
    """Every check that precedes a distance export; nothing is written.

    The fit must be complete (fit manifest bound to its prepare manifest, final weights equal to the
    training receipt, every linked embedding unchanged). Queries come from a label-free list (one ID
    per line) and a farm of base embeddings with this fit's embedding, width and layer. Validation
    queries must be proteins of the fold's official training file and, unless ``allow_in_sample``
    (engineering checks), not proteins the fit trained on. Test mode needs the Freeze-2 manifest
    (SHA-256, schema_version 1, this checkpoint listed, this runner and the files it executes
    pinned, output inside its test pass root) and the SHA-256 of the export script that runs; it
    refuses inner regimes and queries the fit trained on, and puts the Freeze-2 binding in the
    metadata.
    """
    if mode not in C4_MODES:
        raise ComparatorError(f"Export mode {mode!r} not in {C4_MODES}")
    given = (freeze2_manifest is not None, freeze2_sha256 is not None)
    if mode == "test" and given != (True, True):
        raise ComparatorError("Test mode needs the Freeze-2 manifest path and its SHA-256")
    if mode == "validation" and any(given):
        raise ComparatorError("Validation mode takes no Freeze-2 manifest")
    if mode == "test" and (allow_in_sample or not SHA256_RE.match(str(export_script_sha256 or ""))):
        raise ComparatorError("Test mode needs the export script's SHA-256 and allows no in-sample query")
    roots = list(forbidden_roots)
    work_dir = refuse_inside_repository(refuse_held_out_path(work_dir, roots, "the fit work directory"),
                                        "the fit work directory")
    prepare = _read_json(work_dir / PREPARE_MANIFEST, "Prepare manifest")
    fit = _read_json(work_dir / FIT_MANIFEST, "Fit manifest (the fit has not completed)")
    if fit.get("prepare_manifest_sha256") != sha256_file(work_dir / PREPARE_MANIFEST):
        raise ComparatorError("The fit manifest belongs to another prepare manifest")
    fold, regime, seed = int(prepare["fold"]), prepare["regime"], int(prepare["seed"])
    input_dim, embedding, name = int(prepare["input_dim"]), prepare["embedding"], prepare["training_name"]
    inner_fold = prepare.get("inner_fold")
    if mode == "test" and regime not in OUTER_REGIMES:
        raise ComparatorError(f"Regime {regime} is a development arm; it has validation exports only")
    model = model_name(fold, regime, embedding, seed, input_dim, inner_fold)
    receipt = fit.get("training_receipt") or {}
    if receipt.get("model_name") != model or receipt.get("input_dim") != input_dim:
        raise ComparatorError(f"The training receipt does not describe {model}")
    weights = work_dir / "data" / "model" / f"{model}.pth"
    if not weights.is_file() or sha256_file(weights) != receipt.get("final_weights_sha256"):
        raise ComparatorError(f"Final weights {weights.name} are missing or differ from the training receipt")
    table = work_dir / prepare["training_table"]["file"]
    if sha256_file(table) != prepare["training_table"]["sha256"]:
        raise ComparatorError("Training table changed after prepare")
    farm = prepare["embedding_farm"]
    base_farm_sha256 = verify_farm_links(work_dir, farm)["base"]  # equal to the prepare record, or refused

    out = Path(out)
    if out.suffix != ".npz":
        raise ComparatorError("A C4 export file must end in .npz")
    check = refuse_held_out_path if mode == "validation" else refuse_forbidden_root
    out = refuse_inside_repository(check(out, roots, "the export"), "the export")
    query_ids_file = check(query_ids_file, roots, "the query ID list")
    query_esm_dir = check(query_esm_dir, roots, "the query embedding directory")
    freeze2 = manifest = None
    binding: dict[str, Any] = {}
    if mode == "test":
        manifest = read_freeze2_manifest(freeze2_manifest, freeze2_sha256)
        if sha256_file(weights) not in manifest["checkpoint_sha256s"]:
            raise ComparatorError("The Freeze-2 manifest does not list this checkpoint")
        pins = verify_freeze2_runner_pins(manifest)
        pass_root = Path(manifest["test_pass_output_root"]).resolve()
        if not (pass_root in out.parents):
            raise ComparatorError(f"The test export {out} is not inside the Freeze-2 test_pass_output_root "
                                  f"{pass_root}")
        freeze2 = {"path": str(Path(freeze2_manifest).resolve()), "sha256": str(freeze2_sha256).strip().lower(),
                   "test_pass_output_root": str(pass_root), "runner_pins": pins}
        binding = {"freeze2_manifest_sha256": freeze2["sha256"], "test_pass_output_root": str(pass_root),
                   "runner_sha256": pins[RUNNER_KEY], "export_script_sha256": str(export_script_sha256)}
    for path in (out, c4_receipt_path(out), export_side_files(out)["export_receipt"]):
        if os.path.lexists(path):
            raise ComparatorError(f"Export output already exists (write-once): {path}")

    query_ids, ids_record = _read_accession_lines(query_ids_file)
    train_labels = _table_entries(table)
    trained_on = set(train_labels)
    in_sample = [query for query in query_ids if query in trained_on]
    if in_sample and (mode == "test" or not allow_in_sample):
        raise ComparatorError(f"{len(in_sample)} {mode} queries are training proteins of this fit (in-sample), e.g. "
                              f"{in_sample[:10]}" + ("" if mode == "test" else "; only an engineering check may "
                                                     "allow them (--allow-in-sample-queries)"))
    if mode == "validation":
        official = prepare["official_training_file"]
        rows, report = read_official_training_rows(official["path"], fold, roots)
        if report["sha256"] != official["sha256"]:
            raise ComparatorError("The fold's official training file changed after prepare")
        training_side = {row.accession for row in rows}
        outside = [query for query in query_ids if query not in training_side]
        if outside:
            raise ComparatorError(f"{len(outside)} validation queries are not training-side proteins of fold "
                                  f"{fold}, e.g. {outside[:10]}")
    manifest_path = query_esm_dir / FARM_MANIFEST
    if not manifest_path.is_file():
        raise ComparatorError(f"The query embedding directory has no {FARM_MANIFEST}: {query_esm_dir}")
    entries = read_farm_manifest(manifest_path)
    query_files: dict[str, Path] = {}
    query_sha256: dict[str, str] = {}
    problems = []
    for query in query_ids:
        entry, source = entries.get(query), query_esm_dir / f"{query}.pt"
        if entry is None or not source.is_file():
            problems.append(f"{query} (missing)")
            continue
        differing = [label for label, same in (
            ("not a base embedding", entry.seed is None), ("embedding", entry.embedding == embedding),
            ("width", entry.dim == input_dim), ("layer", entry.layer == farm["base"]["layer"]))
            if not same]
        if not differing and sha256_file(source) != entry.file_sha256:
            differing.append("file hash")
        if differing:
            problems.append(f"{query} ({', '.join(differing)})")
            continue
        query_files[query], query_sha256[query] = source.resolve(), entry.file_sha256
    if problems:
        raise ComparatorError(f"{len(problems)} query embeddings do not fit this export, e.g. {problems[:10]}")

    metadata = {"fold": fold, "regime": regime, "seed": seed, "input_dim": input_dim, "embedding": embedding,
                "mode": mode, "checkpoint_sha256": sha256_file(weights),
                "training_table_sha256": prepare["training_table"]["sha256"],
                "embedding_farm_sha256": base_farm_sha256,
                "query_embeddings_sha256": embedding_farm_digest(query_sha256),
                "n_in_sample_queries": len(in_sample), **binding}
    if inner_fold is not None:
        metadata["inner_fold"] = int(inner_fold)
    inputs = {"work_dir": str(work_dir), "prepare_manifest_sha256": fit["prepare_manifest_sha256"],
              "fit_manifest_sha256": sha256_file(work_dir / FIT_MANIFEST), "query_ids": ids_record,
              "query_ids_sha256": sha256_lines(query_ids),
              "query_farm": {"dir": str(query_esm_dir), "manifest_sha256": sha256_file(manifest_path)},
              "export_script_sha256": export_script_sha256, "allow_in_sample": bool(allow_in_sample),
              "freeze2_manifest": freeze2}
    return ExportPlan(work_dir=work_dir, out=out, mode=mode, training_name=name, model_name=model,
                      table_path=table, weights_path=weights, train_labels=train_labels,
                      query_ids=tuple(query_ids), query_files=query_files,
                      hidden_dim=int(receipt.get("hidden_dim", PUBLISHED_RECIPE["hidden_dim"])),
                      out_dim=int(receipt.get("out_dim", PUBLISHED_RECIPE["out_dim"])),
                      metadata=metadata, inputs=inputs, freeze2_manifest=manifest)


def run_export(*, work_dir: Path | str, query_ids_file: Path | str, query_esm_dir: Path | str, out: Path | str,
               mode: str, clean_python: Path, clean_clone: Path, export_script: Path,
               export_script_sha256: str = EXPORT_SCRIPT_SHA256, freeze2_manifest: Optional[Path | str] = None,
               freeze2_sha256: Optional[str] = None, threads: int = DEFAULT_THREADS,
               extra_forbidden_roots: Sequence[Path] = (), allow_in_sample: bool = False,
               dry_run: bool = False) -> dict[str, Any]:
    """Distance export of one completed fit in a fresh guarded CLEAN process, then its C4 validation.

    Test mode also runs only the pinned export script, checks the Freeze-2 ``code_sha256`` in full
    (DeepMzyme environment) and writes the checkpoint's claim file before the launch.
    """
    work_dir = refuse_inside_repository(work_dir, "the fit work directory")
    prepare = _read_json(work_dir / PREPARE_MANIFEST, "Prepare manifest")
    roots = resolve_forbidden_roots(list(extra_forbidden_roots) + prepare.get("forbidden_roots", []),
                                    clean_clone=Path(clean_clone))
    if mode == "test" and export_script_sha256 != EXPORT_SCRIPT_SHA256:
        raise ComparatorError(f"Test mode runs only the pinned export script (SHA-256 {EXPORT_SCRIPT_SHA256})")
    script = verify_pinned_file(export_script, export_script_sha256, "export script")
    plan = plan_distance_export(work_dir=work_dir, query_ids_file=query_ids_file, query_esm_dir=query_esm_dir,
                                out=out, mode=mode, forbidden_roots=roots, freeze2_manifest=freeze2_manifest,
                                freeze2_sha256=freeze2_sha256, export_script_sha256=script["sha256"],
                                allow_in_sample=allow_in_sample)
    claim_path = None
    if mode == "test":
        plan.inputs["freeze2_manifest"]["code_files_verified"] = len(verify_freeze2_code(plan.freeze2_manifest))
        claim_path = (Path(plan.inputs["freeze2_manifest"]["test_pass_output_root"]) / TEST_CLAIMS_DIR
                      / f"{plan.metadata['checkpoint_sha256']}.json")
        if os.path.lexists(claim_path):
            raise ComparatorError(f"This checkpoint already has a test export claim: {claim_path}")
    side = export_side_files(plan.out)
    for key in ("log", "run_record"):
        if os.path.lexists(side[key]):
            raise ComparatorError(f"Export output already exists (write-once): {side[key]}")
    clone = verify_clean_clone(clean_clone)
    script_args = ["--work-dir", str(plan.work_dir), "--query-ids", plan.inputs["query_ids"]["path"],
                   "--query-esm-dir", plan.inputs["query_farm"]["dir"], "--out", str(plan.out), "--mode", mode,
                   "--runner", str(Path(__file__).resolve()), "--threads", str(int(threads))]
    if mode == "test":
        script_args += ["--freeze2-manifest", plan.inputs["freeze2_manifest"]["path"],
                        "--freeze2-sha256", plan.inputs["freeze2_manifest"]["sha256"]]
    if allow_in_sample:
        script_args.append("--allow-in-sample-queries")
    command = guarded_command(clean_python, export_script, script_args)
    if dry_run:
        return {"out": str(plan.out), "mode": mode, "hash_seed": plan.seed, "command": command,
                "n_query": len(plan.query_ids), "metadata": plan.metadata}

    claim = None
    if claim_path is not None:
        # Written once before the launch: a crashed test export is not rerun under the same checkpoint
        # without removing this file by hand (a ledger entry).
        claim_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            _write_json_once(claim_path, {
                "schema_version": SCHEMA_VERSION, "status": "claimed", "out": str(plan.out),
                "claimed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                **{key: plan.metadata[key] for key in ("fold", "regime", "seed", "embedding", "input_dim",
                                                       "checkpoint_sha256", *C4_TEST_BINDING_FIELDS)}})
        except FileExistsError:
            raise ComparatorError(f"This checkpoint already has a test export claim: {claim_path}") from None
        claim = {"path": str(claim_path), "sha256": sha256_file(claim_path)}
    env = build_clean_env(clean_clone=Path(clean_clone), hash_seed=plan.seed, threads=threads, forbidden_roots=roots)
    plan.out.parent.mkdir(parents=True, exist_ok=True)
    code = _run(command, cwd=plan.out.parent, env=env, log_path=side["log"])
    if code:
        raise ComparatorError(f"CLEAN export failed with exit code {code}; see {side['log']}")
    export = read_distance_export(plan.out)
    wrong = {key: export.receipt.get(key) for key, value in plan.metadata.items() if export.receipt.get(key) != value}
    if export.query_ids != plan.query_ids:
        wrong["query_ids"] = "other IDs or order"
    if export.ec_labels != table_ec_order(plan.table_path):
        wrong["ec_labels"] = EC_LABEL_ORDER
    if wrong:
        raise ComparatorError(f"The export differs from this fit and query list: {sorted(wrong)}")
    if not side["export_receipt"].is_file():
        raise ComparatorError(f"The export script wrote no receipt: {side['export_receipt']}")
    record = {
        "schema_version": SCHEMA_VERSION, "action": "export", "mode": mode, "clean_clone": clone,
        "scripts": {"export_script": script,
                    "access_guard": {"path": str(ACCESS_GUARD_FILE), "sha256": sha256_file(ACCESS_GUARD_FILE)}},
        "clean_python": str(clean_python), "command": command, "returncode": code, "log": side["log"].name,
        "env": {key: value for key, value in env.items() if key not in PASSTHROUGH_ENV},
        "forbidden_roots": [str(root) for root in roots], "inputs": plan.inputs, "test_claim": claim,
        "export": {"npz": plan.out.name, "c4_receipt": c4_receipt_path(plan.out).name,
                   "c4_receipt_sha256": sha256_file(c4_receipt_path(plan.out)),
                   "export_receipt": side["export_receipt"].name,
                   "export_receipt_sha256": sha256_file(side["export_receipt"]),
                   "n_query": len(export.query_ids), "n_ec": len(export.ec_labels),
                   "n_in_sample_queries": export.receipt.get("n_in_sample_queries"),
                   "ec_label_order": EC_LABEL_ORDER},
        "runner_sha256": sha256_file(Path(__file__)),
    }
    _write_json_once(side["run_record"], record)
    return record


# ---------------------------------------------------------------- command line

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    sub = parser.add_subparsers(dest="action", required=True)

    sequence_map = sub.add_parser("sequence-map", help="Training-only accession -> sequence map of the five "
                                                       "official training files.")
    sequence_map.add_argument("--split30-dir", type=Path, default=SPLIT30_DIR,
                              help="Folder of split30_train_split_<fold>.csv; only those five exact names are "
                                   "opened and the folder is never listed.")
    sequence_map.add_argument("--out-dir", type=Path, required=True, help="New directory outside the repository.")
    sequence_map.add_argument("--forbidden-root", type=Path, action="append", default=[])

    prepare = sub.add_parser("prepare", help="Write one fit work directory from training-side inputs.")
    prepare.add_argument("--official-train-csv", type=Path, required=True,
                         help="The fold's official split30_train_split_<fold>.csv (never a test file).")
    prepare.add_argument("--fold", type=int, required=True)
    prepare.add_argument("--regime", choices=REGIMES, required=True)
    prepare.add_argument("--seed", type=int, required=True)
    prepare.add_argument("--work-dir", type=Path, required=True, help="New directory outside the repository.")
    prepare.add_argument("--accession-allowlist", type=Path, help="Regimes r2 and r2_inner: one accession per line.")
    prepare.add_argument("--inner-fold", type=int, help=f"Regimes r1_inner and r2_inner: 0..{N_INNER_FOLDS - 1}.")
    prepare.add_argument("--accession-exclusions", type=Path,
                         help="Regimes r1_inner and r2_inner: the identity tool's "
                              f"{EXCLUSIONS_NAME.format(fold='<f>', inner_fold='<j>')}.")
    prepare.add_argument("--base-esm-dir", type=Path,
                         help=f"Directory with <accession>.pt embeddings and {FARM_MANIFEST}.")
    prepare.add_argument("--masked-esm-dir", type=Path,
                         help=f"Directory with this seed's <accession>_<copy>.pt and {FARM_MANIFEST}.")
    prepare.add_argument("--distance-map-dir", type=Path,
                         help="Precomputed distance map of an identical training table and base farm (helper "
                              "receipt); needs the embedding directories.")
    prepare.add_argument("--input-dim", type=int, default=DEFAULT_INPUT_DIM)
    prepare.add_argument("--embedding", default=DEFAULT_EMBEDDING,
                         help="Embedding name; also a file-name tag and the farm manifests' embedding value.")
    prepare.add_argument("--distance-helper-sha256", default=DISTANCE_MAP_HELPER_SHA256)
    prepare.add_argument("--forbidden-root", type=Path, action="append", default=[])

    fit = sub.add_parser("fit", help="Distance map (unless linked) and training in the CLEAN environment.")
    fit.add_argument("--work-dir", type=Path, required=True)
    fit.add_argument("--clean-python", type=Path, required=True)
    fit.add_argument("--clean-clone", type=Path, required=True)
    fit.add_argument("--training-script", type=Path, required=True)
    fit.add_argument("--distance-helper", type=Path, required=True)
    fit.add_argument("--training-script-sha256", default=TRAINING_SCRIPT_SHA256)
    fit.add_argument("--distance-helper-sha256", default=DISTANCE_MAP_HELPER_SHA256)
    fit.add_argument("--epochs", type=int, default=PUBLISHED_RECIPE["epochs"])
    fit.add_argument("--adaptive-rate", type=int, default=PUBLISHED_RECIPE["adaptive_rate"])
    fit.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    fit.add_argument("--forbidden-root", type=Path, action="append", default=[])
    fit.add_argument("--dry-run", action="store_true", help="Verify inputs and print the commands only.")

    export = sub.add_parser("export", help="Contract C4 distance export of one completed fit (private script, "
                                           "CLEAN environment).")
    export.add_argument("--work-dir", type=Path, required=True, help="The fit's work directory.")
    export.add_argument("--query-ids", type=Path, required=True,
                        help="Label-free query list: one accession per line, no other column.")
    export.add_argument("--query-esm-dir", type=Path, required=True,
                        help=f"Base embeddings <accession>.pt of the queries and their {FARM_MANIFEST}.")
    export.add_argument("--out", type=Path, required=True, help="The C4 .npz to write (write-once).")
    export.add_argument("--mode", choices=C4_MODES, required=True)
    export.add_argument("--freeze2-manifest", type=Path, help="Test mode only.")
    export.add_argument("--freeze2-sha256", help="Test mode only: SHA-256 of the Freeze-2 manifest.")
    export.add_argument("--clean-python", type=Path, required=True)
    export.add_argument("--clean-clone", type=Path, required=True)
    export.add_argument("--export-script", type=Path, required=True)
    export.add_argument("--export-script-sha256", default=EXPORT_SCRIPT_SHA256)
    export.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    export.add_argument("--allow-in-sample-queries", action="store_true",
                        help="Validation mode, engineering checks only: accept queries the fit trained on (counted "
                             "in the export).")
    export.add_argument("--forbidden-root", type=Path, action="append", default=[])
    export.add_argument("--dry-run", action="store_true", help="Verify inputs and print the command only.")

    esmc = sub.add_parser("esmc-farm", help="Embedding farm for the ESMC-input control from sequence-keyed means.")
    esmc.add_argument("--cache-dir", type=Path, required=True,
                      help="Output folder of 'esmc_chain_means.py sequences'.")
    esmc.add_argument("--cache-manifest-sha256", required=True)
    esmc.add_argument("--layer", required=True, help="'final' or 'blockNN' (zero-based block index).")
    esmc.add_argument("--sequence-map", type=Path, required=True, help=f"{SEQUENCE_MAP_FILE} of 'sequence-map'.")
    esmc.add_argument("--sequence-map-sha256")
    esmc.add_argument("--kind", choices=FARM_KINDS, required=True)
    esmc.add_argument("--seed", type=int, help="Mask seed (masked farms).")
    esmc.add_argument("--masked-manifest", type=Path, action="append", default=[],
                      help="A prepared fit's masks/<name>_seed<s>_masked_manifest.tsv (masked farms; repeatable).")
    esmc.add_argument("--out-dir", type=Path, required=True, help="New directory outside the repository.")
    esmc.add_argument("--forbidden-root", type=Path, action="append", default=[])

    validate = sub.add_parser("validate-export", help="Check a distance export against contract C4.")
    validate.add_argument("--npz", type=Path, required=True)
    validate.add_argument("--receipt", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    clone = getattr(args, "clean_clone", None)
    roots = resolve_forbidden_roots(getattr(args, "forbidden_root", []), clean_clone=clone)
    install_forbidden_read_guard(roots)
    if args.action == "sequence-map":
        result = write_sequence_map(split30_dir=args.split30_dir, out_dir=args.out_dir, forbidden_roots=roots)
        summary = {key: result[key] for key in ("n_accessions", "n_rows_total", "n_unique_sequences",
                                                "n_files_histogram", "accessions_sha256", "outputs")}
        summary["out_dir"] = str(Path(args.out_dir).resolve())
    elif args.action == "export":
        result = run_export(work_dir=args.work_dir, query_ids_file=args.query_ids, query_esm_dir=args.query_esm_dir,
                            out=args.out, mode=args.mode, clean_python=args.clean_python,
                            clean_clone=args.clean_clone, export_script=args.export_script,
                            export_script_sha256=args.export_script_sha256, freeze2_manifest=args.freeze2_manifest,
                            freeze2_sha256=args.freeze2_sha256, threads=args.threads,
                            extra_forbidden_roots=args.forbidden_root,
                            allow_in_sample=args.allow_in_sample_queries, dry_run=args.dry_run)
        summary = result if args.dry_run else {"out": str(Path(args.out).resolve()), **result["export"]}
    elif args.action == "esmc-farm":
        result = write_esmc_farm(cache_dir=args.cache_dir, cache_manifest_sha256=args.cache_manifest_sha256,
                                 layer=args.layer, sequence_map=args.sequence_map, out_dir=args.out_dir,
                                 kind=args.kind, seed=args.seed, masked_manifests=args.masked_manifest,
                                 sequence_map_sha256=args.sequence_map_sha256, forbidden_roots=roots)
        summary = {key: result[key] for key in ("kind", "seed", "embedding", "farm_layer", "n_files",
                                                "farm_manifest_sha256", "farm_sha256")}
    elif args.action == "prepare":
        result = prepare_fit(official_train_csv=args.official_train_csv, fold=args.fold, regime=args.regime,
                             seed=args.seed, work_dir=args.work_dir, accession_allowlist=args.accession_allowlist,
                             base_esm_dir=args.base_esm_dir, masked_esm_dir=args.masked_esm_dir,
                             distance_map_dir=args.distance_map_dir, input_dim=args.input_dim,
                             embedding=args.embedding, distance_helper_sha256=args.distance_helper_sha256,
                             inner_fold=args.inner_fold, accession_exclusions=args.accession_exclusions,
                             forbidden_roots=roots)
        summary = {"work_dir": str(Path(args.work_dir).resolve()), "training_table": result["training_table"],
                   "masked_copies": result["masked_copies"], "farm_complete": result["embedding_farm"]["complete"]}
    elif args.action == "fit":
        summary = run_fit(work_dir=args.work_dir, clean_python=args.clean_python, clean_clone=args.clean_clone,
                          training_script=args.training_script, distance_helper=args.distance_helper,
                          training_script_sha256=args.training_script_sha256,
                          distance_helper_sha256=args.distance_helper_sha256, epochs=args.epochs,
                          adaptive_rate=args.adaptive_rate, threads=args.threads,
                          extra_forbidden_roots=args.forbidden_root, dry_run=args.dry_run)
    else:
        export = read_distance_export(args.npz, args.receipt)
        summary = {"npz": str(args.npz), "n_query": len(export.query_ids), "n_ec": len(export.ec_labels),
                   "fold": export.receipt["fold"], "regime": export.receipt["regime"],
                   "seed": export.receipt["seed"], "mode": export.receipt["mode"]}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ComparatorError as exc:
        raise SystemExit(f"error: {exc}") from None
