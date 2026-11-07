"""Per-fold whole-protein pretraining corpus of family F6 (EC class vs CLEAN30 campaign, amendment 2; training side).

F6 pretrains its whole-protein branch on the fold's full official training file before the
metallo fit (``training.global_esm_pretraining``). This step builds, inside fold k's view, the
folder ``whole_protein_pretrain/`` that such a fit reads:

- ``corpus.tsv``: ``accession, sequence_sha256, ec_class, exclude_inner0..3`` (one row per
  accession, sorted, 0/1 flags; ``training.global_esm_pretraining.CORPUS_COLUMNS``);
- ``seq_means_final/``: byte copies of the final-layer sequence-mean payloads of the corpus keys
  with a fresh manifest (``global_esm_features.build_manifest``: the corpus key list as its
  allowlist, ``build.action = pretrain_corpus_subset`` with the parent manifest SHA-256, no
  ``labels`` and no parent allowlist, so contract C2 holds and no other fold's accession is
  carried; ``copy_cache_subset`` is not used because it keeps both);
- ``corpus_receipt.json``: input SHA-256 values, rules, counts (private), the extra exclusions
  of rule (c), the conflicting duplicate groups and the parity record. It is written last; a
  folder without it is unfinished.

Rules (amendment 2, F6-2 and F6-3):

- Rows: every accession of fold k's official training file ``split30_train_split_<k>.csv`` (exact
  name; only its EC number column is read for labels), labelled with its EC class by
  ``training.ec_classes.classify_ec_class``. Proteins whose EC numbers span several EC classes are
  excluded and counted (the cohort rule); a fatal status (missing, malformed, out of range, mixed)
  refuses the build. Class 7 is kept. Identical sequences under different accessions stay separate
  rows. Sequences come from the identity folder's target FASTA (the bytes the pinned search used),
  which must equal the official file's sequences; the key is the SHA-256 of the UTF-8 residue
  string, and every key must exist in the parent sequence-mean cache (fail closed).
- ``exclude_inner<j>`` = (a) the inner-validation proteins of j (the certified membership, equal to
  the view's) ∪ (b) the CLEAN R1-inner exclusion list of (k, j) (the validation proteins and their
  D3 neighbours in the pinned metallo-training search; ``identity.build_r1_inner_exclusions``,
  read through its receipt) ∪ (c) every protein whose sequence equals a validation protein's or
  whose accession equals one up to letter case. No new search is run. An inner-training protein
  that would be flagged refuses the build.
- Parity (C9): for every view structure whose chain-mean ``source_sequence_sha256`` equals its
  official sequence's SHA-256, the sequence mean must equal the view's final chain mean within
  cosine >= 0.9999 and maximum absolute difference <= 1e-4; other structures are counted and keep
  their chain mean at fine-tuning. A failure refuses the build; tolerances are never changed.

A smoke view (engineering only) builds its corpus from the official file alone with the view's
smoke membership, which is not certified: flags are (a) ∪ (c) with (c) applied only to proteins
outside the membership, no D3 neighbours (``SMOKE_EXCLUSION_RULE``), recorded as smoke.

The returned summary (what the action prints) gives counts for fold 0 only; the other folds keep
them in the receipt: per-fold full-file class counts over five folds give test-side counts by
subtraction (as in ``r2_lists``). Refusal messages name files and counts, never an accession.

``parity_sample_keys`` and ``compare_sequence_mean_samples`` give the pre-declared label-free
GPU-versus-CPU check of a sequence-mean build (the 64 lexicographically first sequence SHA-256
keys plus the 8 longest sequences, same limits).

Runner action (``scripts/run_ec_clean30_campaign.py``): build the sub-parser with
``add_pretrain_corpus_arguments(sub.add_parser("pretrain-corpus", help=PRETRAIN_CORPUS_HELP))`` and set
``handler=action_pretrain_corpus``. Sub-actions::

    pretrain-corpus holdout-lists --overlay O --campaign-root R --fold K [--freeze1-sha256 F]
    pretrain-corpus build --campaign-root R --fold K --official-split-dir D --identity-work W
    pretrain-corpus fasta --campaign-root R --official-split-dir D
    pretrain-corpus attach --view V --official-split-dir D --seq-cache C --seq-manifest-sha256 S
                    [--identity-work W] [--corpus-table T --corpus-sha256 X] [--smoke]
    pretrain-corpus verify --view V

``build`` writes the cache-free corpus table of fold k (identity and the official training file only) into
``<campaign root>/amendment2/f6_corpus/fold<k>/`` (``corpus.tsv`` and ``corpus_table_receipt.json``), so the five
corpus SHA-256 values can be bound in the amendment-2 record before any sequence-mean cache exists. ``fasta`` writes
the training-side union FASTA of the five official training files (``f6_corpus/union_training.fasta`` with its
receipt), the input of the pod sequence-mean build. ``attach`` with ``--corpus-table`` and ``--corpus-sha256``
refuses unless its freshly built corpus has exactly the bound bytes. ``attach`` also requires the parent cache to be
an fp32 build whose runtime precision policy passes ``runtime_precision.validate_policy`` (rule A2-GOV-3; a smoke
attach records the check without refusing).

``holdout-lists`` (rule A2-F6-3, K17) byte-copies the overlay's ``identity/fold<k>/`` files, each checked against
the overlay manifest, into ``<campaign root>/amendment2/f6_holdout/fold<k>/work/`` and builds the four CLEAN
R1-inner lists there (``identity.build_r1_inner_exclusions``); the overlay is never written. For fold 0 the copy
already holds the overlay's lists, so ``write_once`` proves equality. The copy then serves as ``--identity-work``.
``attach`` builds the corpus and its fresh sequence-mean manifest into the view's attached
``whole_protein_pretrain/`` folder (``views.ATTACHED_DIRS``; outside the view manifest). ``verify`` is the
required check before a fold's first F6 unit (``verify_whole_protein_pretrain(required=True)``).

The module imports neither torch nor the trainer at import time (``view_pretrain_inputs`` serves
the torch-free command builder); the build and the checks import them when they run.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

from ec_clean30 import identity, r2_lists
from ec_clean30.provenance import provenance_record, real_action_git_state, refuse_test_paths
from training.ec_classes import accession_from_structure_id


SCHEMA_VERSION = 1
STEP = "pretrain_corpus"
PRETRAIN_CORPUS_HELP = ("Whole-protein pretraining corpus of family F6 inside a fold view, from the fold's official "
                        "training file, the identity folder and a sequence-mean cache (amendment 2).")
WHOLE_PROTEIN_PRETRAIN_DIR = "whole_protein_pretrain"
SEQ_MEANS_DIR = "seq_means_final"
CORPUS_NAME = "corpus.tsv"
CORPUS_RECEIPT_NAME = "corpus_receipt.json"
MANIFEST_NAME = "manifest.json"
N_INNER_FOLDS = identity.N_INNER_FOLDS
SUBSET_BUILD_ACTION = "pretrain_corpus_subset"
SUMMARY_COUNT_FOLDS: tuple[int, ...] = (0,)

PARITY_MIN_COSINE = 0.9999
PARITY_MAX_ABS_DIFFERENCE = 1e-4
PARITY_SAMPLE_FIRST_KEYS = 64
PARITY_SAMPLE_LONGEST = 8

CORPUS_RULE = ("every accession of fold k's official training file with one EC class (training.ec_classes."
               "classify_ec_class on its EC numbers; several EC classes excluded and counted; fatal statuses "
               "refused; class 7 kept); identical sequences under different accessions stay separate rows; key = "
               "SHA-256 of the UTF-8 residue string of the identity target FASTA, which equals the official file")
EXCLUSION_RULE = ("exclude_inner<j> = (a) inner-validation proteins of j (certified membership) | (b) the CLEAN "
                  "R1-inner exclusion list of (k, j) (validation proteins and their D3 neighbours, >= 30% identity, "
                  ">= 50 residue pairs, E <= 1e-3, any coverage, in the pinned metallo-training search against the "
                  "full file) | (c) proteins whose sequence equals a validation protein's or whose accession equals "
                  "one up to letter case | (d) proteins with a selected AlphaFill donor shared with a validation "
                  "protein (the identity nodes table); an inner-training protein that would be flagged refuses the "
                  "build")
SMOKE_EXCLUSION_RULE = ("smoke view: exclude_inner<j> = (a) validation proteins of the smoke membership's fold j | (c) "
                        "proteins outside the membership with a validation protein's sequence or accession up to "
                        "letter case; no D3 neighbours; the uncertified smoke membership is not checked for links "
                        "across its folds (engineering only)")
PARITY_RULE = ("for every view structure whose chain-mean source_sequence_sha256 equals its accession's official "
               f"sequence SHA-256: cosine >= {PARITY_MIN_COSINE} and max |difference| <= {PARITY_MAX_ABS_DIFFERENCE} "
               "between the sequence mean and the final chain mean; other structures are counted")
SAMPLE_RULE = (f"the {PARITY_SAMPLE_FIRST_KEYS} lexicographically first sequence SHA-256 keys of the sequence map plus "
               f"the keys of its {PARITY_SAMPLE_LONGEST} longest sequences (length descending, then key), as a set")
SUBSET_RULE = ("byte copies of the parent cache's final-layer payloads of the corpus keys; a fresh manifest with the "
               "corpus key list as its allowlist and no labels field")
PRECISION_RULE = ("the parent sequence-mean cache must be an fp32 build (build.runtime.forward_precision = 'fp32') "
                  "whose recorded runtime precision policy passes ec_clean30.runtime_precision.validate_policy for "
                  "its device type ('cuda', or 'cpu' for the CPU build); real attach refuses otherwise")
FORWARD_PRECISION_FP32 = "fp32"
F6_CORPUS_FOLDER = ("amendment2", "f6_corpus")
CORPUS_TABLE_RECEIPT_NAME = "corpus_table_receipt.json"
UNION_FASTA_NAME = "union_training.fasta"
UNION_FASTA_RECEIPT_NAME = "union_training_fasta_receipt.json"
N_OUTER_FOLDS = 5

PathLike = Union[str, os.PathLike]


class PretrainCorpusError(ValueError):
    """The pretraining-corpus step refuses its inputs or finds a folder that does not verify."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise PretrainCorpusError(message)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_key(sequence: str) -> str:
    """The sequence-map key: SHA-256 of the UTF-8 residue string."""
    return _sha256_bytes(sequence.encode("utf-8"))


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _write_new(path: Path, data: bytes) -> str:
    with Path(path).open("xb") as handle:
        handle.write(data)
    return _sha256_bytes(data)


# ---------------------------------------------------------------------------
# The view folder (torch-free; used by the command builder)


def pretrain_dir(view: Any) -> Path:
    return Path(view.root) / WHOLE_PROTEIN_PRETRAIN_DIR


@dataclass(frozen=True)
class PretrainInputs:
    """Where an F6 fit reads the view's pretraining corpus, with the SHA-256 values it pins."""

    root: Path
    cache_dir: Path
    manifest_sha256: str
    corpus: Path
    corpus_sha256: str
    receipt_sha256: str


def view_pretrain_inputs(view: Any) -> PretrainInputs:
    """The finished ``whole_protein_pretrain/`` folder of ``view`` (its receipt must bind this view)."""
    root = pretrain_dir(view)
    _require(root.is_dir() and not root.is_symlink(), f"{WHOLE_PROTEIN_PRETRAIN_DIR}/ is not built in {view.root}")
    receipt_path, corpus, manifest = root / CORPUS_RECEIPT_NAME, root / CORPUS_NAME, root / SEQ_MEANS_DIR / MANIFEST_NAME
    for path in (receipt_path, corpus, manifest):
        _require(path.is_file() and not path.is_symlink(),
                 f"{path.relative_to(view.root)} is missing or a link (an unfinished corpus build)")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise PretrainCorpusError(f"{CORPUS_RECEIPT_NAME} is not valid JSON: {exc}") from exc
    record = receipt.get("view") or {}
    _require(receipt.get("step") == STEP and receipt.get("schema_version") == SCHEMA_VERSION,
             f"{CORPUS_RECEIPT_NAME} is not a pretraining-corpus receipt of schema {SCHEMA_VERSION}")
    _require(record.get("manifest_sha256") == view.manifest_sha256 and receipt.get("fold") == view.fold
             and receipt.get("smoke") is view.smoke, f"{CORPUS_RECEIPT_NAME} was built for another view")
    corpus_sha, manifest_sha = _sha256_file(corpus), _sha256_file(manifest)
    outputs = receipt.get("outputs") or {}
    _require((outputs.get(CORPUS_NAME) or {}).get("sha256") == corpus_sha,
             f"{CORPUS_NAME} changed after the pretraining-corpus step")
    _require((outputs.get(f"{SEQ_MEANS_DIR}/{MANIFEST_NAME}") or {}).get("sha256") == manifest_sha,
             f"{SEQ_MEANS_DIR}/{MANIFEST_NAME} changed after the pretraining-corpus step")
    return PretrainInputs(root, root / SEQ_MEANS_DIR, manifest_sha, corpus, corpus_sha, _sha256_file(receipt_path))


# The declared pretraining-input stub of the amendment-2 dry runs (rule A2-GOV-4): F6 commands are built statically
# against placeholder SHA-256 values derived from this specification and the fold, inside the view's attached folder;
# a command built on it never runs and never verifies (commands.run_unit and verify_unit refuse it).
PRETRAIN_STUB_FORMAT = "deepmzyme_ec_clean30_f6_pretrain_input_stub_v1"
PRETRAIN_STUB_SPEC: Mapping[str, Any] = {
    "format": PRETRAIN_STUB_FORMAT,
    "use": "amendment-2 dry runs only (write=False, plan version 2); real commands refuse it",
    "corpus": f"<view>/{WHOLE_PROTEIN_PRETRAIN_DIR}/{CORPUS_NAME}",
    "cache_dir": f"<view>/{WHOLE_PROTEIN_PRETRAIN_DIR}/{SEQ_MEANS_DIR}",
    "corpus_sha256": f"sha256('{PRETRAIN_STUB_FORMAT}:corpus:fold<k>')",
    "manifest_sha256": f"sha256('{PRETRAIN_STUB_FORMAT}:manifest:fold<k>')",
    "deferred_input": "the fold's corpus table (pretrain-corpus build) and its attached subset manifest",
}
PRETRAIN_STUB_SHA256 = hashlib.sha256(json.dumps(PRETRAIN_STUB_SPEC, sort_keys=True,
                                                 separators=(",", ":")).encode("utf-8")).hexdigest()


def stub_pretrain_inputs(view: Any) -> PretrainInputs:
    """The declared pretraining-input stub of ``view`` (``PRETRAIN_STUB_SPEC``); its receipt SHA-256 is the
    stub's specification SHA-256, which marks every command built on it."""
    root = pretrain_dir(view)

    def placeholder(kind: str) -> str:
        return _sha256_bytes(f"{PRETRAIN_STUB_FORMAT}:{kind}:fold{int(view.fold)}".encode("utf-8"))

    return PretrainInputs(root, root / SEQ_MEANS_DIR, placeholder("manifest"), root / CORPUS_NAME,
                          placeholder("corpus"), PRETRAIN_STUB_SHA256)


# ---------------------------------------------------------------------------
# Pure rules


def read_official_ec_numbers(path: Path) -> dict[str, str]:
    """accession -> EC number string of an official training file (sequences are not read here)."""
    rows = identity._read_allowlisted(path, {"accession": ("entry", "id"), "ec": ("ec number", "ec_number", "ec")})
    numbers: dict[str, str] = {}
    for line, row in enumerate(rows, start=2):
        try:
            accession = identity._accession(row["accession"], f"{Path(path).name}:{line}")
        except ValueError as exc:
            raise PretrainCorpusError(f"Invalid accession at {Path(path).name}:{line}") from exc
        value = "".join(str(row["ec"]).split())
        _require(numbers.setdefault(accession, value) == value,
                 f"An accession of {Path(path).name} repeats with another EC number string (line {line})")
    _require(bool(numbers), f"{Path(path).name} holds no rows")
    return numbers


def classify_official_rows(ec_numbers: Mapping[str, str]) -> tuple[dict[str, str], set[str]]:
    """(accession -> EC class token, several-class accessions); any fatal status refuses."""
    from training.ec_classes import STATUS_AMBIGUOUS, STATUS_OK, classify_ec_class

    labels: dict[str, str] = {}
    several: set[str] = set()
    fatal: dict[str, int] = defaultdict(int)
    for accession, value in ec_numbers.items():
        result = classify_ec_class([value])
        if result.status == STATUS_OK:
            labels[accession] = str(result.token)
        elif result.status == STATUS_AMBIGUOUS:
            several.add(accession)
        else:
            fatal[result.status] += 1
    _require(not fatal, f"Untrusted EC annotations in the official training file {dict(sorted(fatal.items()))}; the "
                        "pretraining corpus is not built")
    return labels, several


@dataclass(frozen=True)
class CorpusBuild:
    rows: tuple[Any, ...]  # training.global_esm_pretraining.CorpusRow
    counts: Mapping[str, Any]
    extra_exclusions: Mapping[str, list[str]]  # inner fold -> accessions flagged only by rule (c)


def build_corpus_rows(
    *,
    labels: Mapping[str, str],
    sequences: Mapping[str, str],
    validation_by_inner: Mapping[int, Iterable[str]],
    training_by_inner: Mapping[int, Iterable[str]],
    r1_lists: Optional[Mapping[int, Mapping[str, str]]],
    n_inner_folds: int = N_INNER_FOLDS,
    donors: Optional[Mapping[str, str]] = None,
) -> CorpusBuild:
    """Corpus rows and their exclusion flags (rules in the module docstring); ``r1_lists=None`` is the smoke rule
    (rule (c) only for proteins outside the membership, which a smoke membership does not certify)."""
    from training.global_esm_pretraining import N_INNER_FOLDS as CORPUS_FLAG_COLUMNS, CorpusRow

    _require(1 <= n_inner_folds <= CORPUS_FLAG_COLUMNS, f"Inner folds must be 1..{CORPUS_FLAG_COLUMNS}")
    _require(set(validation_by_inner) == set(range(n_inner_folds)) == set(training_by_inner),
             "Validation and training proteins are needed for every inner fold")
    missing = set(labels) - set(sequences)
    _require(not missing, f"{len(missing)} labelled accession(s) have no sequence")
    keys = {accession: sequence_key(sequences[accession]) for accession in labels}
    flags: dict[str, list[bool]] = {accession: [False] * CORPUS_FLAG_COLUMNS for accession in labels}
    extra: dict[str, list[str]] = {}
    per_inner: dict[str, Any] = {}
    smoke = r1_lists is None
    membership = set().union(*(set(validation_by_inner[inner]) | set(training_by_inner[inner])
                               for inner in range(n_inner_folds)))
    for inner in range(n_inner_folds):
        validation = set(validation_by_inner[inner])
        training = set(training_by_inner[inner])
        _require(not validation & training, f"Inner fold {inner}: a protein is both validation and training")
        _require(bool(validation), f"Inner fold {inner} has no validation proteins")
        r1 = dict(r1_lists[inner]) if r1_lists is not None else {}
        if r1_lists is not None:
            listed_validation = {accession for accession, reason in r1.items()
                                 if reason == identity.R1_REASON_VALIDATION}
            _require(listed_validation == validation, f"The R1-inner list of inner fold {inner} names other "
                                                      "inner-validation proteins than the certified membership")
        validation_keys = {sequence_key(sequences[accession]) for accession in validation if accession in sequences}
        validation_upper = {accession.upper() for accession in validation}
        validation_donors = {donors[accession] for accession in validation if donors and donors.get(accession)}
        flagged = {accession for accession in labels
                   if accession in validation or accession in r1
                   or ((keys[accession] in validation_keys or accession.upper() in validation_upper)
                       and not (smoke and accession in membership))}
        by_donor = {accession for accession in labels
                    if donors and donors.get(accession) and donors[accession] in validation_donors}
        flagged |= by_donor
        bad = flagged & training
        _require(not bad, f"Inner fold {inner}: {len(bad)} inner-training protein(s) would be excluded from their own "
                          "fold's pretraining (an identity link across inner folds); refused")
        for accession in flagged:
            flags[accession][inner] = True
        extra[str(inner)] = sorted(flagged - set(r1) - validation)
        rule_d_only = by_donor - set(r1) - validation - {
            accession for accession in labels
            if keys[accession] in validation_keys or accession.upper() in validation_upper}
        per_inner[str(inner)] = {
            "flagged": len(flagged),
            "validation_proteins": len(validation),
            "validation_in_corpus": len(validation & set(labels)),
            "validation_absent_from_corpus": len(validation - set(labels)),
            "r1_inner_listed": len(r1),
            "r1_inner_listed_in_corpus": len(set(r1) & set(labels)),
            "extra_rule_c": len(extra[str(inner)]) - len(rule_d_only),
            "extra_rule_d": len(rule_d_only),
            "kept": len(labels) - len(flagged),
        }
    rows = tuple(CorpusRow(accession, keys[accession], labels[accession], tuple(flags[accession]))
                 for accession in sorted(labels))
    classes_by_key: dict[str, set[str]] = defaultdict(set)
    accessions_by_key: dict[str, int] = defaultdict(int)
    for row in rows:
        classes_by_key[row.sequence_sha256].add(row.ec_class)
        accessions_by_key[row.sequence_sha256] += 1
    class_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        class_counts[row.ec_class] += 1
    counts = {
        "corpus_rows": len(rows),
        "unique_sequences": len(classes_by_key),
        "identical_sequence_groups": sum(1 for count in accessions_by_key.values() if count > 1),
        "identical_sequence_groups_with_conflicting_ec_classes": sum(1 for classes in classes_by_key.values()
                                                                     if len(classes) > 1),
        "class_counts": dict(sorted(class_counts.items())),
        "n_inner_folds": n_inner_folds,
        "per_inner_fold": per_inner,
    }
    return CorpusBuild(rows, counts, extra)


# ---------------------------------------------------------------------------
# Parity (C9) and the GPU-versus-CPU sample


def vector_parity(first: Any, second: Any) -> tuple[float, float]:
    """(cosine, maximum absolute difference) of two vectors, computed in float64."""
    import torch

    a, b = first.detach().to("cpu", torch.float64).flatten(), second.detach().to("cpu", torch.float64).flatten()
    _require(a.numel() == b.numel() and a.numel() > 0, "Parity vectors must have one shared, non-zero width")
    denominator = float(a.norm() * b.norm())
    cosine = float(torch.dot(a, b)) / denominator if denominator > 0 else (1.0 if torch.equal(a, b) else 0.0)
    return cosine, float((a - b).abs().max())


def parity_passes(cosine: float, max_abs_difference: float) -> bool:
    return cosine >= PARITY_MIN_COSINE and max_abs_difference <= PARITY_MAX_ABS_DIFFERENCE


def parity_record(pairs: Mapping[str, tuple[Any, Any]], *, differs: Iterable[str] = (),
                  not_in_corpus: Iterable[str] = ()) -> dict[str, Any]:
    """The C9 record of ``{id: (sequence mean, chain mean)}``; ``passed`` is False if any compared pair fails."""
    cosines, differences, failed = [], [], []
    for name in sorted(pairs):
        cosine, difference = vector_parity(*pairs[name])
        cosines.append(cosine)
        differences.append(difference)
        if not parity_passes(cosine, difference):
            failed.append(name)
    differs, not_in_corpus = sorted(differs), sorted(not_in_corpus)
    return {
        "rule": PARITY_RULE, "min_cosine_limit": PARITY_MIN_COSINE, "max_abs_difference_limit": PARITY_MAX_ABS_DIFFERENCE,
        "n_compared": len(pairs), "n_failed": len(failed), "failed": failed,
        "min_cosine": min(cosines) if cosines else None, "max_abs_difference": max(differences) if differences else None,
        "n_structure_sequence_differs": len(differs), "structure_sequence_differs": differs,
        "n_not_in_corpus": len(not_in_corpus), "not_in_corpus": not_in_corpus,
        "passed": not failed,
    }


def parity_sample_keys(sequences: Mapping[str, str]) -> list[str]:
    """The pre-declared label-free sample of a sequence map (``SAMPLE_RULE``), sorted."""
    keys = {sequence_key(sequence): len(sequence) for sequence in sequences.values()}
    first = sorted(keys)[:PARITY_SAMPLE_FIRST_KEYS]
    longest = sorted(keys, key=lambda key: (-keys[key], key))[:PARITY_SAMPLE_LONGEST]
    return sorted(set(first) | set(longest))


def compare_sequence_mean_samples(cache_a: PathLike, manifest_sha256_a: str, cache_b: PathLike,
                                  manifest_sha256_b: str, keys: Sequence[str]) -> dict[str, Any]:
    """C9 comparison of two sequence-mean caches (e.g. a GPU build against a CPU build) on ``keys`` (final layer)."""
    from training.global_esm_features import FINAL_LAYER, load_sequence_mean_vectors

    first = load_sequence_mean_vectors(Path(cache_a), manifest_sha256_a, list(keys), FINAL_LAYER)
    second = load_sequence_mean_vectors(Path(cache_b), manifest_sha256_b, list(keys), FINAL_LAYER)
    record = parity_record({key: (first[key], second[key]) for key in keys})
    record["rule"] = f"sample check: {SAMPLE_RULE}; limits as C9"
    return record


# ---------------------------------------------------------------------------
# Build


def _load_identity_inputs(identity_work: Path, fold: int) -> tuple[Any, dict[str, str], dict[int, dict[str, str]],
                                                                    dict[str, Any]]:
    """Fold inputs, target FASTA sequences, the four R1-inner lists and the record of what was read."""
    try:
        inputs = r2_lists.load_fold_inputs(identity_work, fold)
        lists, records = {}, {}
        for inner in range(N_INNER_FOLDS):
            lists[inner], records[str(inner)] = r2_lists.load_r1_inner_exclusions(inputs, inner)
    except (r2_lists.R2ListError, ValueError) as exc:
        raise PretrainCorpusError(f"Fold {fold} identity folder: {exc}") from exc
    target = inputs.chain.inputs["outputs"]["target_fasta"]
    target_path = Path(identity_work) / target["file"]
    _require(target_path.is_file() and _sha256_file(target_path) == target["sha256"],
             f"{target['file']} is missing or changed after the identity step")
    sequences = identity.read_fasta(target_path)
    nodes = inputs.chain.inputs["outputs"]["nodes"]
    nodes_path = Path(identity_work) / nodes["file"]
    _require(nodes_path.is_file() and _sha256_file(nodes_path) == nodes["sha256"],
             f"{nodes['file']} is missing or changed after the identity step")
    donors = {row["accession"]: row["donor_pdb_id"].strip().upper()
              for row in identity.read_tsv(nodes_path, identity.NODE_COLUMNS) if row["donor_pdb_id"].strip()}
    record = {"identity_work": str(Path(identity_work).absolute()), "fold_inputs": dict(inputs.record),
              "target_fasta": {"file": target["file"], "sha256": target["sha256"], "n_records": len(sequences)},
              "nodes": {"file": nodes["file"], "sha256": nodes["sha256"], "n_with_donor": len(donors)},
              "r1_inner_exclusions": records}
    return inputs, sequences, lists, record, donors


def _membership_by_inner(rows: Iterable[Mapping[str, Any]], n_inner_folds: int
                         ) -> tuple[dict[int, set[str]], dict[int, set[str]]]:
    by_fold: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        by_fold[int(row["fold"])].add(str(row["accession"]))
    everyone = set().union(*by_fold.values()) if by_fold else set()
    validation = {inner: set(by_fold.get(inner, set())) for inner in range(n_inner_folds)}
    return validation, {inner: everyone - validation[inner] for inner in range(n_inner_folds)}


def parent_precision_record(parent_manifest: Mapping[str, Any], *, smoke: bool) -> dict[str, Any]:
    """Rule A2-GOV-3 for ``attach``: the parent cache's forward precision and its validated precision policy.

    A real attach refuses a parent that is not an fp32 build or whose policy does not pass
    ``ec_clean30.runtime_precision.validate_policy`` (or when that module is missing); a smoke attach records the
    result and continues.
    """
    runtime = (parent_manifest.get("build") or {}).get("runtime")
    runtime = runtime if isinstance(runtime, Mapping) else {}
    forward_precision = runtime.get("forward_precision")
    precision = runtime.get("precision") if isinstance(runtime.get("precision"), Mapping) else {}
    policy = precision.get("precision_policy")
    device_type = "cuda" if str(runtime.get("device", "")).startswith("cuda") else "cpu"
    record: dict[str, Any] = {"rule": PRECISION_RULE, "forward_precision": forward_precision,
                              "device_type": device_type, "precision_policy": policy, "passed": False}
    problem = None
    if forward_precision != FORWARD_PRECISION_FP32:
        problem = f"the parent cache was built with forward precision {forward_precision!r}, not 'fp32'"
    elif precision.get("parameter_dtypes") != ["float32"]:
        problem = "the parent cache records no float32 parameters"
    else:
        try:
            import importlib

            runtime_precision = importlib.import_module("ec_clean30.runtime_precision")
        except ImportError:
            problem = "ec_clean30.runtime_precision is not available to validate the recorded policy"
        else:
            try:
                record["precision_policy"] = dict(runtime_precision.validate_policy(policy, device=device_type))
            except (RuntimeError, TypeError, ValueError) as exc:
                problem = f"the recorded precision policy does not validate ({exc})"
    record["passed"] = problem is None
    if problem is not None:
        record["problem"] = problem
        _require(smoke, f"The parent sequence-mean cache fails the precision rule: {problem} (rule A2-GOV-3)")
    return record


def _corpus_rows_for_fold(*, fold: int, smoke: bool, official_split_dir: PathLike,
                          identity_work: Optional[PathLike], expected_membership_sha256: Optional[str],
                          smoke_membership: Any, smoke_n_folds: Optional[int]) -> dict[str, Any]:
    """The corpus rows of fold k with their counts and input record (identity and the official file only)."""
    try:
        official = identity.training_file(Path(official_split_dir), "official", fold)
    except ValueError as exc:
        raise PretrainCorpusError(str(exc)) from exc
    ec_numbers = read_official_ec_numbers(official)
    try:
        official_sequences, duplicate_rows = identity.read_official_sequences(official)
    except ValueError as exc:
        raise PretrainCorpusError(f"{official.name}: {exc}") from exc
    _require(set(ec_numbers) == set(official_sequences), f"{official.name}: EC and sequence columns name different "
                                                          "accessions")
    labels, several = classify_official_rows(ec_numbers)
    inputs_record: dict[str, Any] = {"official_training_file": {"file": official.name, "sha256": _sha256_file(official),
                                                                "n_accessions": len(ec_numbers),
                                                                "duplicate_rows": duplicate_rows}}
    if smoke:
        n_inner = int(smoke_n_folds)
        validation_by_inner, training_by_inner = _membership_by_inner(smoke_membership.rows, n_inner)
        sequences = official_sequences
        r1_lists = None
        donors = None
        inputs_record["membership"] = {"role": "smoke", "sha256": smoke_membership.file_sha256}
        exclusion_rule = SMOKE_EXCLUSION_RULE
    else:
        fold_inputs, sequences, r1_lists, identity_record, donors = _load_identity_inputs(Path(identity_work), fold)
        _require(expected_membership_sha256 is None or fold_inputs.membership.file_sha256 == expected_membership_sha256,
                 "The view's inner membership differs from the identity folder's certified membership")
        _require(sequences == official_sequences, f"The identity target FASTA differs from {official.name}")
        n_inner = N_INNER_FOLDS
        validation_by_inner, training_by_inner = _membership_by_inner(
            ({"accession": accession, "fold": inner} for accession, inner in fold_inputs.inner_fold_of.items()), n_inner)
        inputs_record.update(identity_record)
        inputs_record["membership"] = {"role": "inner", "sha256": fold_inputs.membership.file_sha256,
                                       "certificate_sha256": fold_inputs.certificate_sha256}
        exclusion_rule = EXCLUSION_RULE
    build = build_corpus_rows(labels=labels, sequences=sequences, validation_by_inner=validation_by_inner,
                              training_by_inner=training_by_inner, r1_lists=r1_lists, n_inner_folds=n_inner,
                              donors=None if smoke else donors)
    membership_accessions = set().union(*validation_by_inner.values())
    counts = {
        "official_accessions": len(ec_numbers),
        "several_ec_classes_excluded": len(several),
        "membership_proteins": len(membership_accessions),
        "membership_proteins_with_several_ec_classes_in_the_official_file": len(membership_accessions & several),
        **build.counts,
    }
    return {"build": build, "counts": counts, "inputs": inputs_record, "exclusion_rule": exclusion_rule}


def build_corpus_folder(
    out_dir: PathLike,
    *,
    fold: int,
    smoke: bool,
    official_split_dir: PathLike,
    seq_cache_dir: PathLike,
    seq_manifest_sha256: str,
    chain_means: tuple[PathLike, str],
    view_record: Mapping[str, Any],
    provenance: Mapping[str, Any],
    identity_work: Optional[PathLike] = None,
    expected_membership_sha256: Optional[str] = None,
    smoke_membership: Any = None,
    smoke_n_folds: Optional[int] = None,
    bound_corpus: Optional[tuple[PathLike, str]] = None,
) -> dict[str, Any]:
    """Write ``corpus.tsv``, ``seq_means_final/`` and ``corpus_receipt.json`` into ``out_dir`` (new or empty).

    ``bound_corpus`` = (the ``build`` corpus table, its bound SHA-256): the fresh corpus must have those bytes.

    Real build: ``identity_work`` (fold k's training-side identity folder) gives the certified
    membership, which must have ``expected_membership_sha256`` (the view's), the target FASTA and
    the four R1-inner lists. Smoke build: ``smoke_membership`` (an ``EcFoldMembership``) and no
    identity folder; ``smoke_n_folds`` is its fold count (the view's membership record).
    """
    from training.global_esm_features import (
        DEFAULT_EMBEDDING_DIM, DEFAULT_ESM_MODEL_NAME, FINAL_LAYER, KIND_CHAIN_MEAN, KIND_SEQUENCE_MEAN,
        GlobalEsmCacheError, _read_direct_child, build_manifest, canonical_ids_sha256, deserialize_payload,
        encode_cache_file, load_global_esm_vectors, load_sequence_mean_vectors, read_manifest, validate_mean_payload,
        write_manifest,
    )
    from training.global_esm_pretraining import render_corpus

    out = Path(out_dir)
    for value, what in ((out, "the output folder"), (official_split_dir, "--official-split-dir"),
                        (seq_cache_dir, "--seq-cache"), (identity_work, "--identity-work"),
                        (chain_means[0], "the chain-mean cache")):
        refuse_test_paths([value], what=what)
    _require(not out.is_symlink() and (not out.exists() or (out.is_dir() and not any(out.iterdir()))),
             f"{out} exists and is not an empty folder; a pretraining corpus is built once (a rebuild takes a "
             "new view or an emptied folder)")
    if smoke:
        _require(identity_work is None and smoke_membership is not None and smoke_n_folds is not None,
                 "A smoke corpus uses the smoke membership (with its fold count) and no identity folder")
    else:
        _require(identity_work is not None and smoke_membership is None and expected_membership_sha256,
                 "A real corpus needs the fold's identity folder and the view's inner-membership SHA-256")

    fold_rows = _corpus_rows_for_fold(fold=fold, smoke=smoke, official_split_dir=official_split_dir,
                                      identity_work=identity_work,
                                      expected_membership_sha256=expected_membership_sha256,
                                      smoke_membership=smoke_membership, smoke_n_folds=smoke_n_folds)
    build, counts = fold_rows["build"], fold_rows["counts"]
    inputs_record, exclusion_rule = fold_rows["inputs"], fold_rows["exclusion_rule"]
    corpus_bytes = render_corpus(build.rows)
    corpus_sha = _sha256_bytes(corpus_bytes)
    if bound_corpus is not None:
        _require(not smoke, "A bound corpus table belongs to a real build")
        _require(corpus_sha == str(bound_corpus[1]).strip().lower()
                 and _sha256_file(Path(bound_corpus[0])) == corpus_sha,
                 "The freshly built corpus differs from the bound corpus table (--corpus-table/--corpus-sha256)")
        inputs_record["bound_corpus_table"] = {"file": Path(bound_corpus[0]).name, "sha256": corpus_sha}

    # Parent cache: every corpus key must have a final-layer vector (fail closed).
    try:
        parent_dir = Path(seq_cache_dir)
        parent = read_manifest(parent_dir, seq_manifest_sha256)
    except (GlobalEsmCacheError, OSError) as exc:
        raise PretrainCorpusError(f"The parent sequence-mean cache does not load: {exc}") from exc
    _require(parent["kind"] == KIND_SEQUENCE_MEAN and FINAL_LAYER in parent.get("layers", []),
             "The parent cache is not a sequence-mean cache with a final layer")
    precision_record = parent_precision_record(parent, smoke=smoke)
    model_name, embedding_dim = parent.get("esm_model_name"), parent.get("embedding_dim")
    _require(model_name == DEFAULT_ESM_MODEL_NAME and embedding_dim == DEFAULT_EMBEDDING_DIM,
             f"The parent cache holds {model_name} vectors of width {embedding_dim}, expected "
             f"{DEFAULT_ESM_MODEL_NAME} and {DEFAULT_EMBEDDING_DIM}")
    entries = {entry["sequence_sha256"]: entry for entry in parent["files"] if entry.get("layer") == FINAL_LAYER}
    keys = sorted({row.sequence_sha256 for row in build.rows})
    absent = [key for key in keys if key not in entries]
    _require(not absent, f"{len(absent)} corpus sequence(s) have no final-layer vector in the parent cache (or their "
                         "residue strings follow another normalisation); refused")

    # Parity against the view's final chain means (before anything is written).
    chain_dir, chain_sha = Path(chain_means[0]), str(chain_means[1])
    try:
        chain_manifest = read_manifest(chain_dir, chain_sha)
        _require(chain_manifest["kind"] == KIND_CHAIN_MEAN, "The view's chain-mean cache is not a chain-mean cache")
        chain_entries = {entry["structure_id"]: entry for entry in chain_manifest["files"]
                         if entry.get("layer") == FINAL_LAYER}
        key_of = {row.accession: row.sequence_sha256 for row in build.rows}
        compared, differs, not_in_corpus = {}, [], []
        for structure_id, entry in sorted(chain_entries.items()):
            accession = accession_from_structure_id(structure_id)
            if accession not in key_of:
                not_in_corpus.append(structure_id)
            elif entry.get("source_sequence_sha256") == key_of[accession]:
                compared[structure_id] = key_of[accession]
            else:
                differs.append(structure_id)
        pairs: dict[str, tuple[Any, Any]] = {}
        if compared:
            chain_vectors = load_global_esm_vectors(chain_dir, chain_sha, sorted(compared), FINAL_LAYER,
                                                    int(embedding_dim))
            sequence_vectors = load_sequence_mean_vectors(parent_dir, seq_manifest_sha256,
                                                          sorted(set(compared.values())), FINAL_LAYER,
                                                          int(embedding_dim))
            pairs = {sid: (sequence_vectors[key], chain_vectors[sid]) for sid, key in compared.items()}
    except (GlobalEsmCacheError, OSError, ValueError) as exc:
        raise PretrainCorpusError(f"The parity check could not read its vectors: {exc}") from exc
    parity = parity_record(pairs, differs=differs, not_in_corpus=not_in_corpus)
    _require(parity["passed"], f"Parity (C9) fails for {parity['n_failed']} of {parity['n_compared']} structures "
                               "(sequence mean vs final chain mean); a defect to repair, the limits stay")

    # Copy the payloads, verified against the parent manifest, then write the fresh manifest.
    items = []
    payload_data: dict[str, bytes] = {}
    try:
        for key in keys:
            entry = entries[key]
            data = _read_direct_child(Path(os.path.realpath(parent_dir)), entry["file"])
            _require(_sha256_bytes(data) == entry["sha256"], "A parent payload's SHA-256 differs from its manifest")
            payload = deserialize_payload(data, origin=entry["file"])
            validate_mean_payload(payload, kind=KIND_SEQUENCE_MEAN, key=key, layer=FINAL_LAYER,
                                  embedding_dim=int(embedding_dim), model_name=str(model_name), origin=entry["file"])
            item = encode_cache_file(payload, data=data)
            _require(item.entry == {name: entry.get(name) for name in item.entry},
                     "A parent payload disagrees with its manifest entry")
            items.append(item)
            payload_data[item.file_name] = data
    except GlobalEsmCacheError as exc:
        raise PretrainCorpusError(f"The parent cache does not verify: {exc}") from exc
    allowlist = {"source": CORPUS_NAME, "rule": "unique sequence_sha256 values of the corpus", "corpus_sha256": corpus_sha,
                 "ids_sha256": canonical_ids_sha256(keys), "n_ids": len(keys)}
    manifest = build_manifest(items, kind=KIND_SEQUENCE_MEAN, allowlist=allowlist,
                              build={"action": SUBSET_BUILD_ACTION, "parent_manifest_sha256": seq_manifest_sha256,
                                     "parent_build": parent.get("build"), "rule": SUBSET_RULE},
                              model_name=str(model_name), embedding_dim=int(embedding_dim))

    out.mkdir(parents=True, exist_ok=True)
    seq_dir = out / SEQ_MEANS_DIR
    seq_dir.mkdir()
    for name, data in sorted(payload_data.items()):
        _write_new(seq_dir / name, data)
    _manifest_path, manifest_sha = write_manifest(seq_dir, manifest)
    _write_new(out / CORPUS_NAME, corpus_bytes)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "step": STEP,
        "fold": int(fold),
        "smoke": bool(smoke),
        "rules": {"corpus": CORPUS_RULE, "exclusion": exclusion_rule, "parity": PARITY_RULE, "subset": SUBSET_RULE,
                  "precision": PRECISION_RULE},
        "view": dict(view_record),
        "inputs": {**inputs_record, "parent_sequence_means": {"dir": str(parent_dir.absolute()),
                                                              "manifest_sha256": seq_manifest_sha256,
                                                              "build": parent.get("build"),
                                                              "forward_precision": precision_record["forward_precision"],
                                                              "precision_policy": precision_record},
                   "chain_means_final": {"dir": str(chain_dir.absolute()), "manifest_sha256": chain_sha}},
        "counts": counts,
        "extra_rule_c_exclusions": dict(build.extra_exclusions),
        "parity": parity,
        "outputs": {CORPUS_NAME: {"sha256": corpus_sha, "n_rows": len(build.rows)},
                    f"{SEQ_MEANS_DIR}/{MANIFEST_NAME}": {"sha256": manifest_sha, "n_keys": len(keys)}},
        "labels_read": True,
        "label_use": "EC classes of fold k's official training file (training side); the membership's classes unused",
        "code_sha256": {"src/ec_clean30/pretrain_corpus.py": _sha256_file(Path(__file__).resolve()),
                        **identity.code_sha256()},
        "provenance": dict(provenance),
    }
    receipt_sha = _write_new(out / CORPUS_RECEIPT_NAME, _json_bytes(receipt))
    summary: dict[str, Any] = {"step": STEP, "fold": int(fold), "smoke": bool(smoke), "out": str(out),
                               "receipt": CORPUS_RECEIPT_NAME, "receipt_sha256": receipt_sha,
                               "corpus_sha256": corpus_sha, "manifest_sha256": manifest_sha,
                               "parity": "passed"}
    if int(fold) in SUMMARY_COUNT_FOLDS:
        summary["counts"] = {key: counts[key] for key in ("official_accessions", "several_ec_classes_excluded",
                                                          "corpus_rows", "unique_sequences")}
        summary["counts"]["kept_per_inner_fold"] = {inner: record["kept"]
                                                    for inner, record in counts["per_inner_fold"].items()}
        summary["parity_counts"] = {key: parity[key] for key in ("n_compared", "n_structure_sequence_differs",
                                                                 "n_not_in_corpus")}
    else:
        summary["counts_in_receipt"] = True
    return summary


def f6_corpus_dir(campaign_root: PathLike, fold: Optional[int] = None) -> Path:
    """``<campaign root>/amendment2/f6_corpus/`` (the union FASTA) or its ``fold<k>/`` folder (the corpus table)."""
    root = Path(campaign_root).joinpath(*F6_CORPUS_FOLDER)
    return root if fold is None else root / f"fold{int(fold)}"


def build_corpus_table(campaign_root: PathLike, fold: int, *, official_split_dir: PathLike, identity_work: PathLike,
                       provenance: Mapping[str, Any]) -> dict[str, Any]:
    """``pretrain-corpus build``: the cache-free corpus table of fold k (rules A2-F6-2 and A2-F6-3).

    Reads only fold k's official training file and its training-side identity folder (or hold-out copy); writes
    ``corpus.tsv`` and ``corpus_table_receipt.json`` once into ``f6_corpus/fold<k>/``. The table's bytes equal the
    ``corpus.tsv`` that ``attach`` later builds into the view, so its SHA-256 can be bound before publication.
    """
    from training.global_esm_pretraining import render_corpus

    out = f6_corpus_dir(campaign_root, fold)
    for value, what in ((campaign_root, "--campaign-root"), (official_split_dir, "--official-split-dir"),
                        (identity_work, "--identity-work")):
        refuse_test_paths([value], what=what)
    _require(not out.is_symlink() and (not out.exists() or (out.is_dir() and not any(out.iterdir()))),
             f"{out} exists and is not an empty folder; a corpus table is built once")
    fold_rows = _corpus_rows_for_fold(fold=fold, smoke=False, official_split_dir=official_split_dir,
                                      identity_work=identity_work, expected_membership_sha256=None,
                                      smoke_membership=None, smoke_n_folds=None)
    corpus_bytes = render_corpus(fold_rows["build"].rows)
    out.mkdir(parents=True, exist_ok=True)
    corpus_sha = _write_new(out / CORPUS_NAME, corpus_bytes)
    counts = fold_rows["counts"]
    receipt = {
        "schema_version": SCHEMA_VERSION, "step": f"{STEP}_table", "fold": int(fold), "smoke": False,
        "rules": {"corpus": CORPUS_RULE, "exclusion": fold_rows["exclusion_rule"]},
        "inputs": fold_rows["inputs"], "counts": counts,
        "extra_rule_c_exclusions": dict(fold_rows["build"].extra_exclusions),
        "outputs": {CORPUS_NAME: {"sha256": corpus_sha, "n_rows": len(fold_rows["build"].rows)}},
        "cache_read": False, "labels_read": True,
        "label_use": "EC classes of fold k's official training file (training side)",
        "code_sha256": {"src/ec_clean30/pretrain_corpus.py": _sha256_file(Path(__file__).resolve()),
                        **identity.code_sha256()},
        "provenance": dict(provenance),
    }
    receipt_sha = _write_new(out / CORPUS_TABLE_RECEIPT_NAME, _json_bytes(receipt))
    summary: dict[str, Any] = {"step": f"{STEP}_table", "fold": int(fold), "corpus_sha256": corpus_sha,
                               "receipt": CORPUS_TABLE_RECEIPT_NAME, "receipt_sha256": receipt_sha}
    if int(fold) in SUMMARY_COUNT_FOLDS:
        summary["counts"] = {key: counts[key] for key in ("official_accessions", "several_ec_classes_excluded",
                                                          "corpus_rows", "unique_sequences")}
    else:
        summary["counts_in_receipt"] = True
    return summary


def build_union_fasta(campaign_root: PathLike, *, official_split_dir: PathLike, provenance: Mapping[str, Any],
                      folds: Sequence[int] = tuple(range(N_OUTER_FOLDS))) -> dict[str, Any]:
    """``pretrain-corpus fasta``: the training-side union FASTA of the five official training files.

    One record per accession (``>accession``, one sequence line), sorted by accession; an accession whose
    sequence differs between training files refuses the build. Only training files are opened (exact names).
    """
    out = f6_corpus_dir(campaign_root)
    for value, what in ((campaign_root, "--campaign-root"), (official_split_dir, "--official-split-dir")):
        refuse_test_paths([value], what=what)
    fasta_path, receipt_path = out / UNION_FASTA_NAME, out / UNION_FASTA_RECEIPT_NAME
    _require(not os.path.lexists(fasta_path) and not os.path.lexists(receipt_path),
             f"{UNION_FASTA_NAME} already exists (written once)")
    union: dict[str, str] = {}
    files = []
    for fold in folds:
        try:
            official = identity.training_file(Path(official_split_dir), "official", int(fold))
            sequences, duplicate_rows = identity.read_official_sequences(official)
        except ValueError as exc:
            raise PretrainCorpusError(f"Fold {fold} training file: {exc}") from exc
        for accession, sequence in sequences.items():
            _require(union.setdefault(accession, sequence) == sequence,
                     f"An accession of {official.name} has another sequence in an earlier training file; refused")
        files.append({"file": official.name, "sha256": _sha256_file(official), "n_accessions": len(sequences),
                      "duplicate_rows": duplicate_rows})
    text = "".join(f">{accession}\n{union[accession]}\n" for accession in sorted(union))
    out.mkdir(parents=True, exist_ok=True)
    fasta_sha = _write_new(fasta_path, text.encode("utf-8"))
    receipt = {"schema_version": SCHEMA_VERSION, "step": f"{STEP}_union_fasta", "folds": [int(f) for f in folds],
               "inputs": files, "outputs": {UNION_FASTA_NAME: {"sha256": fasta_sha, "n_records": len(union),
                                                               "n_unique_sequences": len(set(union.values()))}},
               "rule": "every accession of the five official training files, sorted, one record each; no test file",
               "labels_read": False,
               "code_sha256": {"src/ec_clean30/pretrain_corpus.py": _sha256_file(Path(__file__).resolve())},
               "provenance": dict(provenance)}
    receipt_sha = _write_new(receipt_path, _json_bytes(receipt))
    return {"step": f"{STEP}_union_fasta", "fasta_sha256": fasta_sha, "receipt_sha256": receipt_sha,
            "n_records": len(union)}


def build_view_pretrain_corpus(view_root: PathLike, *, official_split_dir: PathLike, seq_cache_dir: PathLike,
                               seq_manifest_sha256: str, provenance: Mapping[str, Any],
                               identity_work: Optional[PathLike] = None, smoke: bool = False,
                               bound_corpus: Optional[tuple[PathLike, str]] = None) -> dict[str, Any]:
    """Build ``<view>/whole_protein_pretrain/`` for a fold view (real) or a smoke view (``smoke=True``)."""
    from ec_clean30.views import View
    from training.ec_membership import read_ec_fold_membership

    view = View.load(view_root)
    _require(view.smoke is bool(smoke), "A smoke corpus belongs to a smoke view and a real corpus to a fold view")
    membership = view.membership
    _require(membership is not None, "The view holds no EC membership")
    _require(membership.get("role") == ("smoke" if smoke else "inner"),
             f"The view's membership role is {membership.get('role')!r}")
    chain_sha = view.chain_means_manifest_sha256("final")
    _require(chain_sha is not None, "The view's final-layer chain means are not built (the parity check needs them)")
    view_record = {"root": str(view.root), "manifest_sha256": view.manifest_sha256, "fold": view.fold,
                   "smoke": view.smoke, "bundle_id": view.bundle_id, "membership_sha256": membership["sha256"],
                   "chain_means_final_manifest_sha256": chain_sha}
    smoke_membership = read_ec_fold_membership(view.membership_file(), membership["sha256"]) if smoke else None
    return build_corpus_folder(
        pretrain_dir(view), fold=view.fold, smoke=smoke, official_split_dir=official_split_dir,
        seq_cache_dir=seq_cache_dir, seq_manifest_sha256=seq_manifest_sha256,
        chain_means=(view.chain_means_dir("final"), chain_sha), view_record=view_record, provenance=provenance,
        identity_work=identity_work, expected_membership_sha256=None if smoke else membership["sha256"],
        smoke_membership=smoke_membership, smoke_n_folds=int(membership["n_folds"]) if smoke else None,
        bound_corpus=bound_corpus,
    )


# ---------------------------------------------------------------------------
# Verification (for the view verifier)


def verify_whole_protein_pretrain(view: Any, *, required: bool = False) -> Optional[dict[str, Any]]:
    """Check a view's ``whole_protein_pretrain/`` folder; ``None`` when it is absent and not required.

    Checks: the receipt binds this view and the current corpus and manifest SHA-256 values; the
    corpus parses; the manifest is a fresh subset manifest (build action, no labels, no parent
    allowlist, allowlist = the corpus key list); every key loads; the exclusion flags agree with
    the view's membership (every validation protein of fold j in the corpus is flagged for j and no
    protein of another fold is).
    """
    from training.ec_membership import read_ec_fold_membership
    from training.global_esm_features import FINAL_LAYER, GlobalEsmCacheError, canonical_ids_sha256, read_manifest
    from training.global_esm_features import load_sequence_mean_vectors
    from training.global_esm_pretraining import GlobalEsmPretrainError, read_pretrain_corpus

    root = pretrain_dir(view)
    if not os.path.lexists(root):
        _require(not required, f"{WHOLE_PROTEIN_PRETRAIN_DIR}/ is required but not built")
        return None
    # Contract X9: unpack creates the folder empty; until attach fills it, an empty, non-link folder verifies as
    # "not attached" (a partly filled folder is refused below as an unfinished build).
    if root.is_dir() and not root.is_symlink() and not any(root.iterdir()):
        _require(not required, f"{WHOLE_PROTEIN_PRETRAIN_DIR}/ is required but empty (not attached)")
        return {"dir": WHOLE_PROTEIN_PRETRAIN_DIR, "attached": False}
    inputs = view_pretrain_inputs(view)
    try:
        corpus = read_pretrain_corpus(inputs.corpus, inputs.corpus_sha256)
        manifest = read_manifest(inputs.cache_dir, inputs.manifest_sha256)
    except (GlobalEsmPretrainError, GlobalEsmCacheError) as exc:
        raise PretrainCorpusError(f"{WHOLE_PROTEIN_PRETRAIN_DIR}/ does not verify: {exc}") from exc
    keys = sorted({row.sequence_sha256 for row in corpus.rows})
    build = manifest.get("build") or {}
    allowlist = manifest.get("allowlist") or {}
    _require(build.get("action") == SUBSET_BUILD_ACTION and "labels" not in manifest and "inputs" not in manifest
             and allowlist.get("source") == CORPUS_NAME and allowlist.get("ids_sha256") == canonical_ids_sha256(keys)
             and allowlist.get("corpus_sha256") == inputs.corpus_sha256 and manifest.get("n_keys") == len(keys)
             and manifest.get("layers") == [FINAL_LAYER],
             f"{SEQ_MEANS_DIR}/{MANIFEST_NAME} is not the fresh subset manifest of the corpus")
    try:
        load_sequence_mean_vectors(inputs.cache_dir, inputs.manifest_sha256, keys, FINAL_LAYER)
    except GlobalEsmCacheError as exc:
        raise PretrainCorpusError(f"{SEQ_MEANS_DIR}/ does not load: {exc}") from exc
    record = view.membership
    _require(record is not None, "The view holds no EC membership")
    membership = read_ec_fold_membership(view.membership_file(), record["sha256"])
    validation, training = _membership_by_inner(membership.rows, int(record["n_folds"]))
    by_accession = {row.accession: row for row in corpus.rows}
    for inner in validation:
        _require(all(by_accession[accession].exclude[inner] for accession in validation[inner]
                     if accession in by_accession),
                 f"A validation protein of inner fold {inner} is not flagged in {CORPUS_NAME}")
        _require(not any(by_accession[accession].exclude[inner] for accession in training[inner]
                         if accession in by_accession),
                 f"A training protein of inner fold {inner} is flagged in {CORPUS_NAME}")
    return {"dir": WHOLE_PROTEIN_PRETRAIN_DIR, "attached": True, "corpus_sha256": inputs.corpus_sha256,
            "manifest_sha256": inputs.manifest_sha256, "receipt_sha256": inputs.receipt_sha256,
            "n_rows": len(corpus.rows), "n_keys": len(keys)}


# ---------------------------------------------------------------------------
# Runner action


HOLDOUT_FOLDER = ("amendment2", "f6_holdout")
HOLDOUT_RECEIPT_NAME = "holdout_lists_receipt.json"
OVERLAY_MANIFEST_NAME = "overlay_manifest.json"


def holdout_work_dir(campaign_root: PathLike, fold: int) -> Path:
    return Path(campaign_root).joinpath(*HOLDOUT_FOLDER, f"fold{int(fold)}", "work")


def build_holdout_lists(overlay_dir: PathLike, campaign_root: PathLike, fold: int, *,
                        freeze1_sha256: Optional[str] = None, allow_unpinned_binary: bool = False,
                        provenance: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Rule A2-F6-3 (K17): the four CLEAN R1-inner lists of fold ``fold`` built in a scratch copy of the overlay's
    training-side identity files; the overlay is never written. Returns the receipt (counts for fold 0 only are
    printed by the action)."""
    overlay = Path(overlay_dir)
    refuse_test_paths([overlay, campaign_root], what="--overlay/--campaign-root")
    manifest_path = overlay / OVERLAY_MANIFEST_NAME
    _require(manifest_path.is_file() and not manifest_path.is_symlink(), f"{OVERLAY_MANIFEST_NAME} is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise PretrainCorpusError(f"{OVERLAY_MANIFEST_NAME} is not valid JSON: {exc}") from exc
    prefix = f"identity/fold{int(fold)}/"
    entries = [entry for entry in manifest.get("files") or [] if isinstance(entry, Mapping)
               and str(entry.get("path", "")).startswith(prefix)]
    _require(entries, f"The overlay manifest lists no file under {prefix}")
    work = holdout_work_dir(campaign_root, fold)
    _require(not os.path.lexists(work), f"{work} exists already; the hold-out lists are built once")
    staging = work.with_name(work.name + ".partial")
    _require(not os.path.lexists(staging), f"{staging} exists (an unfinished build); remove it first")
    copied = []
    try:
        for entry in sorted(entries, key=lambda item: item["path"]):
            relative = entry["path"][len(prefix):]
            parts = relative.split("/")
            _require(relative and not any(part in ("", ".", "..") for part in parts),
                     f"Malformed overlay path {entry['path']!r}")
            source = overlay / entry["path"]
            _require(source.is_file() and not source.is_symlink(), f"{entry['path']} is missing or a link")
            data = source.read_bytes()
            _require(_sha256_bytes(data) == entry.get("sha256") and len(data) == entry.get("size_bytes"),
                     f"{entry['path']} differs from the overlay manifest")
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            copied.append({"path": relative, "sha256": entry["sha256"]})
        os.replace(staging, work)
    except BaseException:
        if staging.exists():
            import shutil

            shutil.rmtree(staging, ignore_errors=True)
        raise
    results = {}
    for inner in range(N_INNER_FOLDS):
        try:
            results[str(inner)] = identity.build_r1_inner_exclusions(work, int(fold), inner,
                                                                     freeze1_sha256=freeze1_sha256,
                                                                     allow_unpinned_binary=allow_unpinned_binary)
        except ValueError as exc:
            raise PretrainCorpusError(f"Fold {fold} inner fold {inner}: the R1-inner list cannot be built ({exc})") \
                from exc
    receipt = {
        "schema_version": SCHEMA_VERSION, "step": "f6_holdout_lists", "fold": int(fold),
        "rule": "A2-F6-3 (b): the CLEAN R1-inner exclusion lists, built in a scratch copy of the overlay identity",
        "overlay_manifest_sha256": _sha256_file(manifest_path), "n_files_copied": len(copied),
        "copied_files_sha256": _sha256_bytes(_json_bytes(copied)),
        "outputs": {inner: result["outputs"] for inner, result in results.items()},
        "counts": {inner: result["counts"] for inner, result in results.items()},
        "provenance": dict(provenance) if provenance is not None else provenance_record(smoke=False),
    }
    receipt["receipt_sha256"] = _write_new(work.parent / HOLDOUT_RECEIPT_NAME, _json_bytes(receipt))
    return receipt


def add_pretrain_corpus_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The ``pretrain-corpus`` action's sub-actions; the handler is ``action_pretrain_corpus``."""
    sub = parser.add_subparsers(dest="corpus_action", required=True)
    holdout = sub.add_parser("holdout-lists", help="Build the R1-inner hold-out lists in a scratch identity copy.")
    holdout.add_argument("--overlay", required=True, help="The data overlay (its manifest checks every copied file).")
    holdout.add_argument("--campaign-root", required=True, help="Campaign root (the copy goes below amendment2/).")
    holdout.add_argument("--fold", type=int, required=True)
    holdout.add_argument("--freeze1-sha256", default=None)
    table = sub.add_parser("build", help="Build fold k's cache-free corpus table below amendment2/f6_corpus/.")
    table.add_argument("--campaign-root", required=True)
    table.add_argument("--fold", type=int, required=True)
    table.add_argument("--official-split-dir", required=True,
                       help="Folder of the official split30 training files (opened by exact name).")
    table.add_argument("--identity-work", required=True,
                       help="The fold's training-side identity folder or its hold-out copy.")
    fasta = sub.add_parser("fasta", help="Write the training-side union FASTA of the five official training files.")
    fasta.add_argument("--campaign-root", required=True)
    fasta.add_argument("--official-split-dir", required=True)
    attach = sub.add_parser("attach", help="Build the corpus into the view's attached whole_protein_pretrain/.")
    attach.add_argument("--view", required=True, help="The fold view (or a smoke view with --smoke).")
    attach.add_argument("--official-split-dir", required=True,
                        help="Folder of the official split30 training files (opened by exact name).")
    attach.add_argument("--seq-cache", required=True, help="Sequence-mean cache of the frozen sequence map.")
    attach.add_argument("--seq-manifest-sha256", required=True, help="SHA-256 of that cache's manifest.")
    attach.add_argument("--identity-work", default=None,
                        help="The fold's training-side identity folder or its hold-out copy (fold views).")
    attach.add_argument("--corpus-table", default=None,
                        help="The bound corpus table of 'pretrain-corpus build' (with --corpus-sha256).")
    attach.add_argument("--corpus-sha256", default=None, help="Its SHA-256 as bound in the amendment-2 record.")
    attach.add_argument("--smoke", action="store_true",
                        help="Engineering use on a smoke view: smoke membership, no identity folder (recorded).")
    verify = sub.add_parser("verify", help="The required check of the attached folder before a fold's first F6 unit.")
    verify.add_argument("--view", required=True)
    return parser


def action_pretrain_corpus(args: argparse.Namespace) -> int:
    """Test-path refusals, the clean-tree check (real runs), the sub-action and its summary (exit 0, or 2 on
    refusal). Printed counts are fold-0 counts only."""
    action = getattr(args, "corpus_action", "attach")
    for name in ("view", "official_split_dir", "seq_cache", "identity_work", "overlay", "campaign_root",
                 "corpus_table"):
        if getattr(args, name, None) is not None:
            refuse_test_paths([getattr(args, name)], what=f"--{name.replace('_', '-')}")
    if action == "verify":
        try:
            from ec_clean30.views import View

            record = verify_whole_protein_pretrain(View.load(args.view), required=True)
        except (PretrainCorpusError, OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps({"status": "verified", **(record or {})}, indent=2, sort_keys=True))
        return 0
    if action in ("build", "fasta"):
        try:
            git = real_action_git_state(smoke=False)
            provenance = provenance_record(smoke=False, git=git)
            if action == "build":
                summary = build_corpus_table(args.campaign_root, args.fold, official_split_dir=args.official_split_dir,
                                             identity_work=args.identity_work, provenance=provenance)
            else:
                summary = build_union_fasta(args.campaign_root, official_split_dir=args.official_split_dir,
                                            provenance=provenance)
        except PretrainCorpusError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(summary, indent=2, sort_keys=True, default=str))
        return 0
    if action == "holdout-lists":
        try:
            git = real_action_git_state(smoke=False)
            receipt = build_holdout_lists(args.overlay, args.campaign_root, args.fold,
                                          freeze1_sha256=args.freeze1_sha256,
                                          provenance=provenance_record(smoke=False, git=git))
        except PretrainCorpusError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        printed = {"fold": receipt["fold"], "receipt_sha256": receipt["receipt_sha256"],
                   "n_files_copied": receipt["n_files_copied"]}
        if receipt["fold"] == 0:
            printed["counts"] = receipt["counts"]
        print(json.dumps(printed, indent=2, sort_keys=True))
        return 0
    try:
        git = real_action_git_state(smoke=args.smoke)
        table = getattr(args, "corpus_table", None)
        table_sha = getattr(args, "corpus_sha256", None)
        _require((table is None) == (table_sha is None), "--corpus-table and --corpus-sha256 go together")
        summary = build_view_pretrain_corpus(
            args.view, official_split_dir=args.official_split_dir, seq_cache_dir=args.seq_cache,
            seq_manifest_sha256=args.seq_manifest_sha256, identity_work=args.identity_work, smoke=args.smoke,
            provenance=provenance_record(smoke=args.smoke, git=git),
            bound_corpus=None if table is None else (table, table_sha))
    except PretrainCorpusError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0
