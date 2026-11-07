"""Whole-protein branch pretraining on a full-training-file corpus (EC class campaign, amendment 2, family F6).

F6 is the F5 model (pocket ESMC encoder, site statistics, whole-protein branch, seven-way EC-class
head) whose whole-protein branch (``model.GlobalESMEncoder``) is first trained on the fold's full
official training file and then frozen. Everything here runs inside the fit, before the F6 model
is built; the trainer calls three functions (``--global-esm-branch-init pretrained_frozen``):

1. ``run_global_esm_pretraining(config, train_pockets=..., val_pockets=..., run_dir=...)`` after the
   EC group metadata is assigned. It reads the SHA-pinned corpus (``read_pretrain_corpus``),
   selects the split's rows with run-time checks (``select_pretrain_rows``), loads the
   final-layer sequence means (``load_corpus_vectors``), pretrains the branch with its own
   two-layer head (``pretrain_whole_protein_classifier``), writes the control predictions C-GF of
   an inner fit (``wp_pretrain_val_protein_predictions.csv``) and the receipt
   ``global_esm_pretrain_receipt.json``.
2. ``install_pretrained_global_branch(model, pretraining)`` after the model is built and before
   the optimizer: a strict load of ``global_esm_encoder`` and ``requires_grad_(False)`` on it.
   The single-group optimizer keeps every parameter; AdamW skips parameters without gradients
   (no state, no decay), so the realised optimizer steps of the other parameters are unchanged.
3. ``verify_wp_pretrain_unit(run_dir, ...)`` for the unit verifier: receipt, frozen branch digest
   in the terminal checkpoint, and the control predictions of an inner fit.

The corpus (``corpus.tsv``, written by ``ec_clean30.pretrain_corpus``) has one row per accession
of the fold's official training file with a single EC class: ``accession, sequence_sha256,
ec_class, exclude_inner0..3`` (sorted by accession; 0/1 flags). Split ``inner<j>`` drops the rows
flagged for inner fold j (inner-validation proteins, their D3 neighbours in the pinned search,
identical sequences and accession case variants); split ``refit`` uses every row.

Pretraining rule (fixed, not tuned; F6-5): cross-entropy with inverse-frequency class weights over
the split's rows (``training.loop.class_weights_from_labels``, mean 1 over the classes present),
AdamW (learning rate 1e-3, weight decay 1e-4), gradient clipping 1.0, batch 64, 30 epochs, cosine
schedule stepped per epoch, terminal weights; CPU, float32, one intra-op thread
(``torch.set_num_threads(1)`` inside the pretraining, restored afterwards). It runs inside
``torch.random.fork_rng(devices=[])`` on the CPU stream seeded with ``seed + WP_PRETRAIN_SEED_OFFSET``
and a dedicated batch-order generator, so the fit's global random state, and with it every other
F6 parameter, the batch order and the dropout draws, equal F5's for the same seed. The receipt
records the CPU identity (``cpu_identity``) and the thread count, which condition the same-host
branch-digest rule of the selector.

Control C-GF scores each inner-validation protein from its final-layer sequence-mean vector in the
attached corpus cache (the input distribution the pretraining head saw), never from the view's
structure chain mean; a validation protein absent from the corpus is refused.

Branch digest: SHA-256 over the branch tensors in sorted name order (names relative to the
branch), each as UTF-8 name, its shape as JSON and its little-endian float32 bytes.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import re
from typing import Any, Iterable, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from model import GlobalESMEncoder, build_classifier_head
from training.ec_classes import EC_CLASS_TOKEN_TO_INDEX, EC_CLASS_TOKENS, accession_from_structure_id
from training.global_esm_features import (
    FINAL_LAYER, GLOBAL_ESM_METADATA_KEY, GlobalEsmCacheError, load_sequence_mean_vectors,
)
from training.loop import class_weights_from_labels


SCHEMA_VERSION = 1
RECEIPT_FORMAT = "deepmzyme_global_esm_pretrain_receipt_v1"
RECEIPT_NAME = "global_esm_pretrain_receipt.json"
CONTROL_PREDICTIONS_NAME = "wp_pretrain_val_protein_predictions.csv"
CONTROL_ID = "C-GF"
BRANCH_INIT_PRETRAINED_FROZEN = "pretrained_frozen"
BRANCH_PREFIX = "global_esm_encoder."

N_INNER_FOLDS = 4
SPLIT_REFIT = "refit"
SPLIT_INNER_PREFIX = "inner"
CORPUS_COLUMNS: tuple[str, ...] = (
    "accession", "sequence_sha256", "ec_class", *(f"exclude_inner{index}" for index in range(N_INNER_FOLDS)),
)
N_CLASSES = len(EC_CLASS_TOKENS)

# Fixed pretraining rule (amendment 2, F6-5). The trainer defaults equal these values; the campaign
# renders them explicitly (ec_clean30.campaign.F6_PRETRAINING; equality is tested).
WP_PRETRAIN_EPOCHS = 30
WP_PRETRAIN_LEARNING_RATE = 1e-3
WP_PRETRAIN_BATCH_SIZE = 64
WP_PRETRAIN_WEIGHT_DECAY = 1e-4
WP_PRETRAIN_GRAD_CLIP_NORM = 1.0
WP_PRETRAIN_HEAD_LAYERS = 2
WP_PRETRAIN_HEAD_DROPOUT = 0.2
WP_PRETRAIN_CLASS_WEIGHT_MODE = "inverse_frequency"
# Seed offset of the pretraining stream; distinct from training.run.AUGMENTATION_SEED_OFFSET (7919).
WP_PRETRAIN_SEED_OFFSET = 7927

DIGEST_FORMAT = b"deepmzyme_global_esm_branch_digest_v1\n"
DIGEST_RULE = ("SHA-256 over the branch tensors in sorted name order (names relative to the branch): UTF-8 name, "
               "NUL, shape as JSON, NUL, little-endian float32 bytes")
SPLIT_RULE = ("inner<j>: every corpus row except those flagged exclude_inner<j> (inner-validation proteins of j, "
              "their D3 neighbours in the pinned search, identical sequences and accession case variants); "
              "refit: every corpus row")
RUN_TIME_CHECK_RULE = ("every inner-validation accession present in the corpus is flagged for the split; no "
                       "inner-training accession is flagged; the run's proteins and the corpus overlap")
# Rule A2-F6-5: pretraining (and the C-GF control logits) run on one CPU intra-op thread.
WP_PRETRAIN_NUM_THREADS = 1
CONTROL_INPUT_SOURCE = "corpus_sequence_means"
CONTROL_INPUT_RULE = ("each inner-validation protein is scored from its final-layer sequence-mean vector in the "
                      "attached corpus cache (looked up by the corpus row's sequence_sha256, flagged rows included); "
                      "a validation protein absent from the corpus is refused")

_ACCESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class GlobalEsmPretrainError(ValueError):
    """A pretraining corpus, split, receipt or branch transfer breaks its contract."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise GlobalEsmPretrainError(message)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Corpus table


@dataclass(frozen=True)
class CorpusRow:
    accession: str
    sequence_sha256: str
    ec_class: str
    exclude: tuple[bool, ...]  # one flag per inner fold

    def fields(self) -> tuple[str, ...]:
        return (self.accession, self.sequence_sha256, self.ec_class, *("1" if flag else "0" for flag in self.exclude))


@dataclass(frozen=True)
class PretrainCorpus:
    rows: tuple[CorpusRow, ...]
    file_sha256: str
    path: str

    @property
    def accessions(self) -> tuple[str, ...]:
        return tuple(row.accession for row in self.rows)


def render_corpus(rows: Iterable[CorpusRow]) -> bytes:
    """The corpus table's bytes: tab-separated, header ``CORPUS_COLUMNS``, rows sorted by accession, LF."""
    rows = sorted(rows, key=lambda row: row.accession)
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
    writer.writerow(CORPUS_COLUMNS)
    writer.writerows(row.fields() for row in rows)
    return buffer.getvalue().encode("utf-8")


def parse_corpus(data: bytes, *, origin: str) -> tuple[CorpusRow, ...]:
    """Strict reader: exact header, unique accessions in sorted order, SHA-256 keys, EC class 1-7, 0/1 flags."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GlobalEsmPretrainError(f"{origin} is not UTF-8 text") from exc
    _require(text.endswith("\n") and "\r" not in text, f"{origin} must end with a newline and use LF line ends")
    lines = text[:-1].split("\n")
    _require(tuple(lines[0].split("\t")) == CORPUS_COLUMNS, f"{origin} needs the tab-separated columns {CORPUS_COLUMNS}")
    rows: list[CorpusRow] = []
    previous = None
    for number, line in enumerate(lines[1:], start=2):
        values = line.split("\t")
        _require(len(values) == len(CORPUS_COLUMNS), f"Malformed row {number} of {origin}")
        accession, key, token, *flags = values
        _require(bool(_ACCESSION_RE.match(accession)), f"Invalid accession in row {number} of {origin}")
        _require(previous is None or accession > previous, f"Row {number} of {origin} repeats an accession or breaks "
                                                           "the sorted order")
        _require(bool(_SHA256_RE.match(key)), f"Invalid sequence_sha256 in row {number} of {origin}")
        _require(token in EC_CLASS_TOKEN_TO_INDEX, f"Invalid EC class in row {number} of {origin}")
        _require(all(flag in {"0", "1"} for flag in flags), f"Exclusion flags must be 0 or 1 (row {number} of {origin})")
        rows.append(CorpusRow(accession, key, token, tuple(flag == "1" for flag in flags)))
        previous = accession
    _require(bool(rows), f"{origin} holds no rows")
    return tuple(rows)


def read_pretrain_corpus(path: str | os.PathLike, expected_sha256: str) -> PretrainCorpus:
    """The corpus table, refused unless its bytes hash to ``expected_sha256``."""
    path = Path(path)
    data = path.read_bytes()
    digest = _sha256_bytes(data)
    _require(digest == str(expected_sha256 or "").strip().lower(),
             f"Pretraining corpus {path.name} has SHA-256 {digest}, expected {expected_sha256}")
    return PretrainCorpus(parse_corpus(data, origin=path.name), digest, str(path))


# ---------------------------------------------------------------------------
# Split selection with run-time checks


def parse_split(split: str) -> Optional[int]:
    """``None`` for ``refit``; the inner fold index for ``inner<j>``."""
    if split == SPLIT_REFIT:
        return None
    if isinstance(split, str) and split.startswith(SPLIT_INNER_PREFIX) and split[len(SPLIT_INNER_PREFIX):].isdigit():
        inner = int(split[len(SPLIT_INNER_PREFIX):])
        if 0 <= inner < N_INNER_FOLDS and split == f"{SPLIT_INNER_PREFIX}{inner}":
            return inner
    raise GlobalEsmPretrainError(f"Unknown pretraining split {split!r}; expected inner0..inner{N_INNER_FOLDS - 1} "
                                 f"or {SPLIT_REFIT}")


@dataclass(frozen=True)
class PretrainSelection:
    split: str
    inner_fold: Optional[int]
    rows: tuple[CorpusRow, ...]
    n_corpus_rows: int
    checks: Mapping[str, int]

    @property
    def n_excluded(self) -> int:
        return self.n_corpus_rows - len(self.rows)


def select_pretrain_rows(corpus: PretrainCorpus, split: str, *, training_accessions: Iterable[str],
                         validation_accessions: Iterable[str] = ()) -> PretrainSelection:
    """The split's rows, after the run-time checks against the fit's own proteins.

    Refused: a validation accession that is in the corpus but not flagged for ``inner<j>``; an
    inner-training accession that is flagged; a refit split with validation proteins; a corpus
    that holds none of the fit's training proteins (or, for an inner split, none of its
    validation proteins). Run proteins absent from the corpus (several EC classes in the
    official file) are counted, never trained on.
    """
    inner = parse_split(split)
    training, validation = set(training_accessions), set(validation_accessions)
    _require(not training & validation, "A protein is both an inner-training and an inner-validation protein")
    present = set(corpus.accessions)
    if inner is None:
        _require(not validation, "A refit pretraining split has no validation proteins")
        kept = corpus.rows
        flagged: set[str] = set()
    else:
        flagged = {row.accession for row in corpus.rows if row.exclude[inner]}
        unflagged_validation = (validation & present) - flagged
        _require(not unflagged_validation, f"{len(unflagged_validation)} inner-validation protein(s) of the fit are in "
                                           f"the pretraining corpus but not flagged for {split}; refused")
        flagged_training = training & flagged
        _require(not flagged_training, f"{len(flagged_training)} inner-training protein(s) of the fit are flagged for "
                                       f"{split}; refused")
        _require(bool(validation & present), f"No inner-validation protein of the fit is in the pretraining corpus "
                                             f"({split}); is it another fold's corpus?")
        kept = tuple(row for row in corpus.rows if not row.exclude[inner])
    _require(bool(training & present), "No training protein of the fit is in the pretraining corpus; is it another "
                                       "fold's corpus?")
    _require(bool(kept), f"The pretraining split {split} keeps no rows")
    checks = {
        "training_proteins": len(training),
        "training_in_corpus_kept": len(training & present),
        "training_absent_from_corpus": len(training - present),
        "training_flagged": 0,
        "validation_proteins": len(validation),
        "validation_in_corpus_flagged": len(validation & flagged),
        "validation_absent_from_corpus": len(validation - present),
    }
    return PretrainSelection(split, inner, tuple(kept), len(corpus.rows), checks)


def load_corpus_vectors(cache_dir: str | os.PathLike, manifest_sha256: str, rows: Sequence[CorpusRow], *,
                        embedding_dim: int) -> Tensor:
    """Final-layer sequence means of ``rows`` (row order), through the SHA-checked cache loader."""
    keys = sorted({row.sequence_sha256 for row in rows})
    vectors = load_sequence_mean_vectors(Path(cache_dir), manifest_sha256, keys, FINAL_LAYER, int(embedding_dim))
    return torch.stack([vectors[row.sequence_sha256] for row in rows]).to(torch.float32)


# ---------------------------------------------------------------------------
# Pretraining


def cpu_identity() -> dict[str, Optional[str]]:
    """The host CPU identity recorded with every pretraining (``platform`` and the first ``model name`` line of
    ``/proc/cpuinfo`` where it exists); it conditions the same-host branch-digest rule (A2-F6-5)."""
    model_name = None
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    model_name = line.split(":", 1)[1].strip() or None
                    break
    except OSError:
        model_name = None
    return {"machine": platform.machine() or None, "processor": platform.processor() or None,
            "model_name": model_name}


class _single_thread:
    """``torch.set_num_threads(WP_PRETRAIN_NUM_THREADS)`` for the block; the previous count is restored."""

    def __enter__(self) -> int:
        self.previous = int(torch.get_num_threads())
        torch.set_num_threads(WP_PRETRAIN_NUM_THREADS)
        return self.previous

    def __exit__(self, *exc: Any) -> None:
        torch.set_num_threads(self.previous)


class _Vectors:
    """The attribute ``GlobalESMEncoder.forward`` reads (``x_global_esm``)."""

    def __init__(self, vectors: Tensor):
        self.x_global_esm = vectors


def _state_copy(module: nn.Module) -> dict[str, Tensor]:
    return {name: tensor.detach().to("cpu").clone() for name, tensor in module.state_dict().items()}


def branch_digest(state: Mapping[str, Tensor], *, prefix: str = "") -> str:
    """Digest of the branch tensors (``prefix`` selects and strips them, e.g. ``BRANCH_PREFIX``)."""
    items = {name[len(prefix):]: tensor for name, tensor in state.items() if name.startswith(prefix)}
    _require(bool(items), f"No branch tensors with prefix {prefix!r}")
    digest = hashlib.sha256(DIGEST_FORMAT)
    for name in sorted(items):
        tensor = items[name].detach().to("cpu").contiguous()
        _require(tensor.dtype == torch.float32, f"Branch tensor {name} is {tensor.dtype}, expected float32")
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("utf-8") + b"\0")
        digest.update(tensor.numpy().astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


@dataclass
class PretrainOutcome:
    """The pretrained branch and head (CPU, eval mode) with the record of how they were trained."""

    branch: GlobalESMEncoder
    head: nn.Sequential
    branch_state: dict[str, Tensor]
    branch_digest: str
    history: list[dict[str, Any]]
    class_counts: dict[str, int]
    class_weights: list[float]
    settings: dict[str, Any]

    def logits(self, vectors: Tensor) -> Tensor:
        """The pretraining classifier's EC-class logits (eval mode, no gradients, no random draws, one thread)."""
        self.branch.eval()
        self.head.eval()
        with _single_thread(), torch.no_grad():
            return self.head(self.branch(_Vectors(vectors.detach().to("cpu", torch.float32))))


def pretrain_whole_protein_classifier(
    vectors: Tensor,
    labels: Tensor,
    *,
    seed: int,
    embedding_dim: int,
    hidden_dim: int,
    branch_dropout: float,
    epochs: int = WP_PRETRAIN_EPOCHS,
    learning_rate: float = WP_PRETRAIN_LEARNING_RATE,
    batch_size: int = WP_PRETRAIN_BATCH_SIZE,
) -> PretrainOutcome:
    """Train the whole-protein branch with a two-layer head (fixed rule) on one vector per accession.

    The fit's global random state is left exactly as it was (see the module docstring).
    """
    _require(isinstance(vectors, Tensor) and vectors.dim() == 2 and vectors.size(1) == int(embedding_dim),
             f"Pretraining vectors must have shape (n, {embedding_dim})")
    _require(vectors.size(0) > 0 and labels.dim() == 1 and labels.numel() == vectors.size(0),
             "Pretraining needs one label per vector")
    _require(bool(torch.isfinite(vectors).all()), "Pretraining vectors hold non-finite values")
    labels = labels.to(torch.long)
    _require(int(labels.min()) >= 0 and int(labels.max()) < N_CLASSES, "Pretraining labels are outside the seven classes")
    _require(int(epochs) >= 1 and int(batch_size) >= 1 and float(learning_rate) > 0, "Invalid pretraining settings")
    vectors = vectors.detach().to("cpu", torch.float32)
    labels = labels.detach().to("cpu")
    pretrain_seed = int(seed) + WP_PRETRAIN_SEED_OFFSET
    counts = torch.bincount(labels, minlength=N_CLASSES)
    weights = class_weights_from_labels(labels.tolist(), N_CLASSES, mode=WP_PRETRAIN_CLASS_WEIGHT_MODE)
    history: list[dict[str, Any]] = []
    with torch.random.fork_rng(devices=[]), _single_thread() as fit_num_threads:
        threads_used = int(torch.get_num_threads())
        _require(threads_used == WP_PRETRAIN_NUM_THREADS, "Pretraining must run on one CPU thread")
        torch.default_generator.manual_seed(pretrain_seed)  # the CPU stream only; restored on exit
        order_generator = torch.Generator().manual_seed(pretrain_seed)
        branch = GlobalESMEncoder(int(embedding_dim), int(hidden_dim), float(branch_dropout))
        head = build_classifier_head(in_dim=int(hidden_dim), hidden_dim=int(hidden_dim), out_dim=N_CLASSES,
                                     n_linear_layers=WP_PRETRAIN_HEAD_LAYERS, dropout=WP_PRETRAIN_HEAD_DROPOUT)
        parameters = [*branch.parameters(), *head.parameters()]
        optimizer = torch.optim.AdamW(parameters, lr=float(learning_rate), weight_decay=WP_PRETRAIN_WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(epochs)))
        n_rows = int(vectors.size(0))
        for epoch in range(1, int(epochs) + 1):
            branch.train()
            head.train()
            learning_rate_used = float(optimizer.param_groups[0]["lr"])
            loss_sum, correct, steps = 0.0, 0, 0
            order = torch.randperm(n_rows, generator=order_generator)
            for start in range(0, n_rows, int(batch_size)):
                index = order[start:start + int(batch_size)]
                logits = head(branch(_Vectors(vectors[index])))
                loss = F.cross_entropy(logits, labels[index], weight=weights)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, WP_PRETRAIN_GRAD_CLIP_NORM)
                optimizer.step()
                loss_sum += float(loss.detach()) * int(index.numel())
                correct += int((logits.detach().argmax(dim=-1) == labels[index]).sum())
                steps += 1
            scheduler.step()
            mean_loss = loss_sum / n_rows
            _require(math.isfinite(mean_loss), f"Pretraining loss is not finite at epoch {epoch}")
            history.append({"epoch": epoch, "learning_rate": learning_rate_used, "loss": mean_loss,
                            "train_accuracy_training_mode": correct / n_rows, "optimizer_steps": steps})
        branch.eval()
        head.eval()
    branch_state = _state_copy(branch)
    settings = {
        "epochs": int(epochs), "learning_rate": float(learning_rate), "batch_size": int(batch_size),
        "weight_decay": WP_PRETRAIN_WEIGHT_DECAY, "grad_clip_norm": WP_PRETRAIN_GRAD_CLIP_NORM,
        "schedule": "cosine_per_epoch", "t_max": max(1, int(epochs)), "checkpoint": "terminal",
        "loss": "cross_entropy", "class_weight_mode": WP_PRETRAIN_CLASS_WEIGHT_MODE,
        "head_layers": WP_PRETRAIN_HEAD_LAYERS, "head_dropout": WP_PRETRAIN_HEAD_DROPOUT,
        "hidden_dim": int(hidden_dim), "branch_dropout": float(branch_dropout), "embedding_dim": int(embedding_dim),
        "n_classes": N_CLASSES, "device": "cpu", "dtype": "float32",
        "seed": int(seed), "seed_offset": WP_PRETRAIN_SEED_OFFSET, "pretrain_seed": pretrain_seed,
        "rng": "torch.random.fork_rng(devices=[]) with the CPU stream and a batch-order generator seeded "
               "with seed + offset",
        "num_threads": threads_used, "fit_num_threads": int(fit_num_threads),
        "threads_rule": "torch.set_num_threads(1) inside the pretraining; the fit's count is restored afterwards",
    }
    return PretrainOutcome(
        branch=branch, head=head, branch_state=branch_state, branch_digest=branch_digest(branch_state),
        history=history,
        class_counts={token: int(counts[index]) for token, index in EC_CLASS_TOKEN_TO_INDEX.items()},
        class_weights=[float(value) for value in weights.tolist()], settings=settings,
    )


# ---------------------------------------------------------------------------
# Transfer into the F6 model


def install_pretrained_global_branch(model: nn.Module, pretraining: "GlobalBranchPretraining | PretrainOutcome"
                                     ) -> dict[str, Any]:
    """Load the pretrained branch into ``model.global_esm_encoder`` (strict) and freeze it."""
    outcome = pretraining.outcome if isinstance(pretraining, GlobalBranchPretraining) else pretraining
    encoder = getattr(model, "global_esm_encoder", None)
    _require(isinstance(encoder, GlobalESMEncoder), "The model has no whole-protein branch to receive the "
                                                    "pretrained weights")
    encoder.load_state_dict(outcome.branch_state, strict=True)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    digest = branch_digest(encoder.state_dict())
    _require(digest == outcome.branch_digest, "The installed branch differs from the pretrained branch")
    return {
        "branch_digest": digest,
        "frozen_tensors": sorted(BRANCH_PREFIX + name for name, _ in encoder.named_parameters()),
        "frozen_values": int(sum(parameter.numel() for parameter in encoder.parameters())),
        "trainable_values": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
    }


# ---------------------------------------------------------------------------
# Control C-GF


def _content_sha256(rows: Sequence[Mapping[str, str]]) -> str:
    """Hash of the prediction rows without ``run_id`` (equal for the two learning-rate settings of one fit)."""
    content = [{key: value for key, value in row.items() if key != "run_id"} for row in rows]
    return _sha256_bytes(json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def control_input_vectors(pockets: Sequence[Any], corpus: PretrainCorpus, cache_dir: str | os.PathLike,
                          manifest_sha256: str, *, embedding_dim: int) -> tuple[Tensor, dict[str, Any]]:
    """Per validation pocket, its protein's final-layer sequence-mean vector from the corpus cache (rule A2-F6-7).

    The cache holds every corpus key, flagged rows included. A validation protein absent from the corpus is
    refused; the pocket's own chain-mean metadata is never read.
    """
    row_of = {row.accession: row for row in corpus.rows}
    accessions = [accession_from_structure_id(str(pocket.structure_id)) for pocket in pockets]
    absent = sorted({accession for accession in accessions if accession not in row_of})
    _require(not absent, f"{len(absent)} inner-validation protein(s) of the fit are absent from the pretraining "
                         "corpus; C-GF needs their corpus sequence means (refused)")
    rows = [row_of[accession] for accession in sorted(set(accessions))]
    try:
        vectors = load_corpus_vectors(cache_dir, manifest_sha256, rows, embedding_dim=int(embedding_dim))
    except GlobalEsmCacheError as exc:
        raise GlobalEsmPretrainError(f"C-GF vectors do not load: {exc}") from exc
    position = {row.accession: index for index, row in enumerate(rows)}
    stacked = torch.stack([vectors[position[accession]] for accession in accessions]).to(torch.float32)
    record = {"source": CONTROL_INPUT_SOURCE, "rule": CONTROL_INPUT_RULE, "manifest_sha256": manifest_sha256,
              "layer": FINAL_LAYER, "n_proteins": len(rows), "n_pockets": len(accessions),
              "n_unique_sequences": len({row.sequence_sha256 for row in rows})}
    return stacked, record


def control_prediction_rows(outcome: PretrainOutcome, pockets: Sequence[Any], *, run_id: str,
                            fold_index: Optional[int], n_folds: Optional[int], seed: int,
                            vectors: Tensor) -> list[dict[str, str]]:
    """Protein rows (``EC_PROTEIN_PREDICTION_COLUMNS``) of the pretraining classifier on ``pockets``.

    ``vectors`` holds one row per pocket: its protein's corpus sequence mean (``control_input_vectors``), so
    every pocket of a protein has the same logits; the protein row is the mean of its pocket logits, as in the
    standard export, from the pockets' EC group metadata.
    """
    from training.ec_prediction_export import build_prediction_rows

    _require(bool(pockets), "The control needs validation pockets")
    _require(isinstance(vectors, Tensor) and vectors.dim() == 2 and vectors.size(0) == len(pockets),
             "The control needs one corpus sequence-mean vector per validation pocket")
    logits = outcome.logits(vectors)
    identity = {"run_id": str(run_id), "mode": "validation", "fold_index": "" if fold_index is None else str(fold_index),
                "n_folds": "" if n_folds is None else str(n_folds), "seed": str(int(seed))}
    _pocket_rows, protein_rows, conflicting = build_prediction_rows(list(pockets), logits, identity)
    _require(conflicting == 0, f"{conflicting} validation proteins have conflicting EC labels")
    return protein_rows


def write_control_predictions(path: Path, rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    """Write the C-GF protein rows once; returns path, SHA-256, rows and the run-independent content hash."""
    from training.ec_prediction_export import EC_PROTEIN_PREDICTION_COLUMNS

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(EC_PROTEIN_PREDICTION_COLUMNS), lineterminator="\n",
                            extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        handle.write(buffer.getvalue())
    return {"control": CONTROL_ID, "path": Path(path).name, "sha256": _sha256_file(path), "n_rows": len(rows),
            "content_sha256": _content_sha256(rows)}


# ---------------------------------------------------------------------------
# Trainer hook


@dataclass
class GlobalBranchPretraining:
    """What the trainer keeps between pretraining and the branch transfer."""

    outcome: PretrainOutcome
    receipt: dict[str, Any]
    receipt_sha256: str
    receipt_path: Path
    summary: dict[str, Any] = field(default_factory=dict)


def _validated_config(config: Any) -> None:
    _require(getattr(config, "global_esm_branch_init", None) == BRANCH_INIT_PRETRAINED_FROZEN,
             "Pretraining runs only with --global-esm-branch-init pretrained_frozen")
    _require(bool(config.use_global_esm_branch), "Pretraining needs the whole-protein branch")
    _require(not config.run_test_eval and config.test_structure_dir is None,
             "Pretraining never runs with a held-out test set")
    _require(config.global_esm_layer == FINAL_LAYER and config.global_esm_standardize == "none",
             "Pretraining uses final-layer vectors without standardisation")
    for name in ("global_esm_pretrain_cache_dir", "global_esm_pretrain_manifest_sha256", "global_esm_pretrain_corpus",
                 "global_esm_pretrain_corpus_sha256", "global_esm_pretrain_split"):
        _require(getattr(config, name, None) not in (None, ""), f"Pretraining needs {name}")


def run_global_esm_pretraining(config: Any, *, train_pockets: Sequence[Any], val_pockets: Sequence[Any],
                               run_dir: Path) -> GlobalBranchPretraining:
    """Pretrain the branch for one fit and write its receipt (and the C-GF predictions of an inner fit).

    Call after the whole-protein vectors are attached and the EC group metadata is assigned, and
    before the model is built.
    """
    _validated_config(config)
    run_dir = Path(run_dir)
    corpus = read_pretrain_corpus(config.global_esm_pretrain_corpus, config.global_esm_pretrain_corpus_sha256)
    training = {accession_from_structure_id(str(pocket.structure_id)) for pocket in train_pockets}
    validation = {accession_from_structure_id(str(pocket.structure_id)) for pocket in val_pockets}
    selection = select_pretrain_rows(corpus, str(config.global_esm_pretrain_split), training_accessions=training,
                                     validation_accessions=validation)
    if selection.inner_fold is not None:
        _require(config.fold_index is not None and int(config.fold_index) == selection.inner_fold,
                 f"Pretraining split {selection.split} differs from the fit's inner fold {config.fold_index}")
    try:
        vectors = load_corpus_vectors(config.global_esm_pretrain_cache_dir, config.global_esm_pretrain_manifest_sha256,
                                      selection.rows, embedding_dim=int(config.global_esm_dim))
    except GlobalEsmCacheError as exc:
        raise GlobalEsmPretrainError(f"Pretraining vectors do not load: {exc}") from exc
    labels = torch.tensor([EC_CLASS_TOKEN_TO_INDEX[row.ec_class] for row in selection.rows], dtype=torch.long)
    outcome = pretrain_whole_protein_classifier(
        vectors, labels, seed=int(config.seed), embedding_dim=int(config.global_esm_dim),
        hidden_dim=int(config.hidden_s), branch_dropout=float(config.global_esm_dropout),
        epochs=int(config.global_esm_pretrain_epochs), learning_rate=float(config.global_esm_pretrain_learning_rate),
        batch_size=int(config.global_esm_pretrain_batch_size),
    )
    control = None
    if selection.inner_fold is not None:
        control_vectors, control_input = control_input_vectors(
            val_pockets, corpus, config.global_esm_pretrain_cache_dir, config.global_esm_pretrain_manifest_sha256,
            embedding_dim=int(config.global_esm_dim))
        rows = control_prediction_rows(outcome, val_pockets, run_id=run_dir.name, fold_index=config.fold_index,
                                       n_folds=config.n_folds, seed=int(config.seed), vectors=control_vectors)
        control = write_control_predictions(run_dir / CONTROL_PREDICTIONS_NAME, rows)
        control["input"] = control_input
    receipt = {
        "format": RECEIPT_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "family_rule": "amendment 2, F6-5: whole-protein branch pretrained on the fold's full training file, frozen",
        "split": selection.split,
        "inner_fold": selection.inner_fold,
        "split_rule": SPLIT_RULE,
        "run_time_check_rule": RUN_TIME_CHECK_RULE,
        "split_checks": dict(selection.checks),
        "corpus": {"path": str(config.global_esm_pretrain_corpus), "sha256": corpus.file_sha256,
                   "n_rows": len(corpus.rows)},
        "cache": {"dir": str(config.global_esm_pretrain_cache_dir),
                  "manifest_sha256": config.global_esm_pretrain_manifest_sha256, "layer": FINAL_LAYER,
                  "n_unique_sequences": len({row.sequence_sha256 for row in selection.rows})},
        "rows_used": len(selection.rows),
        "rows_excluded_by_split": selection.n_excluded,
        "class_counts": outcome.class_counts,
        "class_weights": outcome.class_weights,
        "hyperparameters": outcome.settings,
        "history": outcome.history,
        "branch_digest": outcome.branch_digest,
        "branch_digest_rule": DIGEST_RULE,
        "branch_tensors": sorted(BRANCH_PREFIX + name for name in outcome.branch_state),
        "control_predictions": control,
        "runtime": {"torch_version": torch.__version__, "num_threads": outcome.settings["num_threads"],
                    "pretrain_num_threads": outcome.settings["num_threads"],
                    "fit_num_threads": outcome.settings["fit_num_threads"], "cpu_identity": cpu_identity()},
        # The scientific identity pins this module; verify_unit compares the recorded SHA-256 with it.
        "module_sha256": _sha256_file(Path(__file__)),
    }
    receipt_path = run_dir / RECEIPT_NAME
    with receipt_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    receipt_sha256 = _sha256_file(receipt_path)
    summary = {"receipt": RECEIPT_NAME, "receipt_sha256": receipt_sha256, "split": selection.split,
               "corpus_sha256": corpus.file_sha256, "manifest_sha256": config.global_esm_pretrain_manifest_sha256,
               "rows_used": len(selection.rows), "branch_digest": outcome.branch_digest,
               "control_predictions": None if control is None else control["path"]}
    return GlobalBranchPretraining(outcome, receipt, receipt_sha256, receipt_path, summary)


# ---------------------------------------------------------------------------
# Unit verification


def _read_control_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def verify_wp_pretrain_unit(run_dir: Path, *, identity: Mapping[str, Any], kind: str,
                            validation_structure_ids: Optional[Sequence[str]] = None,
                            checkpoint_name: str = "terminal_model_checkpoint.pt") -> dict[str, dict[str, Any]]:
    """Checks of an F6 unit: ``wp_pretrain_receipt``, ``wp_branch_frozen`` and, for inner units,
    ``wp_control_predictions`` (each ``{"ok": bool, "detail": ...}``)."""
    from training.ec_prediction_export import EC_PROTEIN_PREDICTION_COLUMNS

    run_dir = Path(run_dir)
    results: dict[str, dict[str, Any]] = {}

    def add(name: str, ok: bool, detail: Any = None) -> None:
        results[name] = {"ok": bool(ok), "detail": detail}

    inner = kind == "inner"
    receipt_path = run_dir / RECEIPT_NAME
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        add("wp_pretrain_receipt", False, f"unreadable: {exc}")
        add("wp_branch_frozen", False, "no receipt")
        if inner:
            add("wp_control_predictions", False, "no receipt")
        return results
    expected_split = f"{SPLIT_INNER_PREFIX}{identity.get('inner_fold')}" if inner else SPLIT_REFIT
    history = receipt.get("history") or []
    losses = [row.get("loss") for row in history]
    epochs = (receipt.get("hyperparameters") or {}).get("epochs")
    receipt_ok = (receipt.get("format") == RECEIPT_FORMAT
                  and (receipt.get("corpus") or {}).get("sha256") == identity.get("wp_pretrain_corpus_sha256")
                  and (receipt.get("cache") or {}).get("manifest_sha256") == identity.get("wp_pretrain_manifest_sha256")
                  and receipt.get("split") == expected_split
                  and epochs == WP_PRETRAIN_EPOCHS == len(history)
                  and (receipt.get("hyperparameters") or {}).get("num_threads") == WP_PRETRAIN_NUM_THREADS
                  and isinstance((receipt.get("runtime") or {}).get("cpu_identity"), Mapping)
                  and all(isinstance(value, (int, float)) and math.isfinite(value) for value in losses))
    add("wp_pretrain_receipt", receipt_ok, {"split": receipt.get("split"), "expected_split": expected_split,
                                            "epochs": epochs, "history": len(history),
                                            "num_threads": (receipt.get("hyperparameters") or {}).get("num_threads")})
    try:
        checkpoint = torch.load(run_dir / checkpoint_name, map_location="cpu", weights_only=False)
        digest = branch_digest(checkpoint["model_state_dict"], prefix=BRANCH_PREFIX)
    except (OSError, KeyError, RuntimeError, GlobalEsmPretrainError) as exc:
        add("wp_branch_frozen", False, f"no branch in the terminal checkpoint: {exc}")
    else:
        add("wp_branch_frozen", digest == receipt.get("branch_digest"),
            {"checkpoint": digest, "pretrained": receipt.get("branch_digest")})
    control = receipt.get("control_predictions")
    path = run_dir / CONTROL_PREDICTIONS_NAME
    if not inner:
        add("wp_no_control_predictions", control is None and not path.exists(), control)
        return results
    detail: dict[str, Any] = {"recorded": control}
    ok = isinstance(control, Mapping) and control.get("path") == CONTROL_PREDICTIONS_NAME and path.is_file() \
        and _sha256_file(path) == control.get("sha256") \
        and (control.get("input") or {}).get("source") == CONTROL_INPUT_SOURCE \
        and (control.get("input") or {}).get("manifest_sha256") == identity.get("wp_pretrain_manifest_sha256")
    if ok:
        columns, rows = _read_control_rows(path)
        structure_ids = sorted(sid for row in rows for sid in str(row.get("structure_ids") or "").split(";") if sid)
        ok = columns == list(EC_PROTEIN_PREDICTION_COLUMNS) and len(rows) == control.get("n_rows") \
            and validation_structure_ids is not None and structure_ids == sorted(validation_structure_ids)
        detail.update(n_rows=len(rows), n_structures=len(structure_ids))
    add("wp_control_predictions", ok, detail)
    return results
