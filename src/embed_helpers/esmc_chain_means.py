"""Build whole-protein ESMC mean-vector caches (contract C2 in ``training.global_esm_features``).

Actions
  from-residue-cache
      CPU only. For every structure ID in an explicit allowlist, opens exactly
      ``<embeddings-dir>/<stem>_chain_<chain>_esmc.pt`` (a format-v2 per-residue
      payload), checks it against the structure chain (residue IDs and order, sequence
      SHA-256, model name, width, finite values) and writes the mean of all residue
      rows as the ``final`` layer. Nothing is written unless every ID passes.
  from-model
      Runs ESMC once per allowlisted chain and writes two means per chain: the final
      normalized layer and zero-based hidden block ``--layer-index`` (the block output
      before the final norm). Device, parameter dtype, autocast, emitted dtypes and
      library versions are recorded in every payload. Payloads are written as they
      finish; ``--resume`` continues an interrupted run that had the same settings.
  sequences
      Sequence-keyed means ``<sha256>__esmc300m_<layer>_seqmean.pt`` for FASTA input,
      used by the CLEAN ESMC-input control. Every label must be an accession of the
      ``--sequence-map`` or ``<accession>_<n>``, a masked copy of one (see below).
      Identical sequences are embedded once.
  subset
      Writes a per-fold subset manifest next to a cache, or copies the subset's files
      into a new directory with its own ``manifest.json``.
  compare
      Per-ID agreement (max |difference| and cosine) between two chain-mean caches,
      e.g. model-computed final-layer means against means of the per-residue caches.
      Passes only when every cosine is at least ``--min-cosine`` and every difference
      at most ``--max-abs-diff``; the defaults (0.9999 and 1e-4) are the pre-declared
      parity tolerance.

Allowlists
  ``--allowlist`` holds one structure ID per line. ``--allowlist-sha256`` (and
  ``--sequence-map-sha256`` for ``sequences``) refuses an input whose bytes hash to
  anything else; the hash of the file used is always recorded in the manifest.

Structure files
  By default each ID resolves only to ``<structure-dir>/<id>.pdb|.cif|.mmcif``; no
  structure manifest or folder listing is read, so a shared folder exposes nothing
  beyond the requested names. ``--structure-manifest`` opts in to a structure manifest
  CSV, for example a training-only view's. Every row of that file is read, so it must
  list training structures only. Only allowlisted rows are resolved (against the
  manifest's folder), and each must record the SHA-256 that its file is checked against.

Sequence map and masked copies (``sequences``)
  The sequence map is a tab-separated file whose header holds at least ``accession``
  and ``sha256`` (of the unmasked sequence); an optional ``sequence`` column is checked
  against ``sha256``, and other columns are ignored. An unmasked record must hash to
  its accession's ``sha256``. A masked copy ``<accession>_<n>`` (``n`` written without
  leading zeros) must have as many residue tokens as its source sequence and equal it
  at every position that is not ``<mask>``. The source is the map's ``sequence`` entry
  or, failing that, the accession's own unmasked record in the input. An exact
  accession match takes precedence over reading a label as a copy.

Mask handling (``sequences``): the literal ``<mask>`` is one residue token (the ESMC
tokenizer reads it as a single token), so a sequence has
``len(sequence) - 5 * count('<mask>')`` residue tokens. The forward pass must return
exactly that many positions plus BOS and EOS. The mean covers every residue-token
position, masked positions included, and excludes BOS and EOS. ``--max-residue-tokens``
keeps the first N residue tokens, each ``<mask>`` counting as one; files stay keyed by
the SHA-256 of the full input sequence and record the token counts used.

Every input is opened by its exact name and must resolve inside a permitted root (the
given input directories plus any ``--permitted-root``). No directory is listed or
globbed, except that an output directory is checked for being empty. The manifest is
written last, so an interrupted run leaves no usable cache. Resumable actions
(``from-model``, ``sequences``) first write ``build_settings.json`` with the action,
model, width, layer index, requested device, allowlist or sequence-map hash, inputs and
truncation. ``--resume`` reopens only a folder whose marker records the same settings,
and reuses a payload only when its identity and source record match this run.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import importlib
from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import platform
import re
import sys
from typing import Any, Collection, Mapping, Sequence

import torch
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embed_helpers.esmc import (  # noqa: E402
    _esmc_sdk,
    expected_embedding_path,
    extract_chain_sequences,
    git_commit_hash,
    parse_structure,
    utc_timestamp,
)
from training.esm_feature_loading import (  # noqa: E402
    ResidueKey,
    deserialize_residue_ids,
    residue_keys_for_structure_chain,
)
from training.global_esm_features import (  # noqa: E402
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_ESM_MODEL_NAME,
    FINAL_LAYER,
    KIND_CHAIN_MEAN,
    KIND_SEQUENCE_MEAN,
    MANIFEST_FILENAME,
    CacheFile,
    GlobalEsmCacheError,
    build_manifest,
    canonical_ids_sha256,
    chain_id_from_structure_id,
    chain_mean_filename,
    chain_mean_payload,
    compute_mean,
    copy_cache_subset,
    deserialize_payload,
    encode_cache_file,
    is_sha256,
    layer_name,
    load_global_esm_vectors,
    normalize_layer,
    read_id_allowlist,
    require_sha256,
    sequence_mean_filename,
    sequence_mean_payload,
    sha256_bytes,
    sha256_text,
    validate_mean_payload,
    write_bytes_atomic,
    write_manifest,
    write_subset_manifest,
)
from training.reproducibility import installed_package_versions  # noqa: E402


BUILD_RECEIPT_FILENAME = "build_receipt.json"
BUILD_SETTINGS_FILENAME = "build_settings.json"
STRUCTURE_SUFFIXES = (".pdb", ".cif", ".mmcif")
STRUCTURE_MANIFEST_COLUMNS = ("structure_name", "relative_path", "sha256")
SEQUENCE_MAP_COLUMNS = ("accession", "sha256")
RESIDUE_CACHE_FORMAT_VERSION = 2
RECORDED_PACKAGES = ("torch", "esm", "numpy", "biopython")
# One forward returns the final normalized embeddings and every block's hidden state.
FORWARD_CONFIG = {
    "sequence": False,
    "return_embeddings": True,
    "return_hidden_states": True,
    "ith_hidden_layer": -1,
}
TOKEN_TRIM = "drop_bos_and_eos"
# Forward precision (amendment 2, rule A2-F6-4). "sdk" is the historical path (``ESMC.logits``: bfloat16 parameters
# and bfloat16 autocast on CUDA); "fp32" casts the parameters to float32 after loading, calls ``ESMC.forward``
# directly with autocast disabled, refuses flash attention, sets TF32 off and requires float32 outputs.
FORWARD_PRECISION_SDK = "sdk"
FORWARD_PRECISION_FP32 = "fp32"
FORWARD_PRECISIONS = (FORWARD_PRECISION_SDK, FORWARD_PRECISION_FP32)
MASK_TOKEN = "<mask>"
MASK_POLICY = "'<mask>' is one residue token; masked positions are included in the mean"
# Pre-declared tolerance for chain means of one protein computed by two routes.
PARITY_MIN_COSINE = 0.9999
PARITY_MAX_ABS_DIFF = 1e-4
_SEQUENCE_TOKEN_RE = re.compile(r"<mask>|[A-Z]")
_RESIDUE_SEQUENCE_RE = re.compile(r"^[A-Z]+$")
_COPY_LABEL_RE = re.compile(r"^(?P<accession>.+)_(?P<copy>0|[1-9][0-9]*)$")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _check_layer_index(layer_index: int) -> None:
    if isinstance(layer_index, bool) or not isinstance(layer_index, int) or layer_index < 0:
        raise GlobalEsmCacheError(f"--layer-index must be a non-negative integer, got {layer_index!r}.")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _permitted_roots(paths: Sequence[Path]) -> tuple[Path, ...]:
    return tuple(Path(os.path.realpath(path)) for path in paths)


def _inside_roots(path: Path, roots: Sequence[Path]) -> Path:
    resolved = Path(os.path.realpath(path))
    if not any(resolved == root or root in resolved.parents for root in roots):
        raise GlobalEsmCacheError(f"{path} resolves to {resolved}, outside the permitted roots.")
    return resolved


def _check_settings_marker(out_dir: Path, settings: Mapping[str, Any] | None) -> None:
    marker = out_dir / BUILD_SETTINGS_FILENAME
    if settings is None or not marker.is_file():
        raise GlobalEsmCacheError(
            f"{out_dir} holds files but no {BUILD_SETTINGS_FILENAME} from a resumable build; use a new directory."
        )
    try:
        recorded = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GlobalEsmCacheError(f"{marker} cannot be read ({exc}).") from exc
    expected = json.loads(_json_bytes(settings))
    if recorded != expected:
        recorded = recorded if isinstance(recorded, dict) else {}
        differs = sorted(name for name in set(recorded) | set(expected) if recorded.get(name) != expected.get(name))
        raise GlobalEsmCacheError(
            f"{out_dir} was started by a build that does not match this run ({', '.join(differs)}); "
            "use a new output directory."
        )


def _prepare_output_dir(
    out_dir: Path, *, resume: bool = False, settings: Mapping[str, Any] | None = None
) -> Path:
    """Create ``out_dir`` (recording ``settings`` when given), or reopen it for a resume.

    A non-empty folder is reopened only with ``resume`` and only when its start marker
    records exactly ``settings``.
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        if not out_dir.is_dir():
            raise GlobalEsmCacheError(f"Output path {out_dir} is not a directory.")
        if (out_dir / MANIFEST_FILENAME).exists():
            raise GlobalEsmCacheError(f"{out_dir} already holds a finished cache ({MANIFEST_FILENAME}).")
        if any(out_dir.iterdir()):
            if not resume:
                raise GlobalEsmCacheError(f"Output directory {out_dir} is not empty; use a new directory.")
            _check_settings_marker(out_dir, settings)
            return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if settings is not None:
        write_bytes_atomic(out_dir / BUILD_SETTINGS_FILENAME, _json_bytes(settings))
    return out_dir


def _structure_files_by_name(
    structure_dir: Path, wanted: Collection[str]
) -> tuple[dict[str, tuple[Path, str | None]], list[str]]:
    found: dict[str, tuple[Path, str | None]] = {}
    problems: list[str] = []
    for structure_id in sorted(wanted):
        candidates = [structure_dir / f"{structure_id}{suffix}" for suffix in STRUCTURE_SUFFIXES]
        existing = [path for path in candidates if path.is_file()]
        if len(existing) > 1:
            problems.append(f"{structure_id}: more than one structure file ({[p.name for p in existing]})")
        elif existing:
            found[structure_id] = (existing[0], None)
    return found, problems


def _structure_rows_from_manifest(
    manifest_path: Path, wanted: Collection[str]
) -> tuple[dict[str, tuple[Path, str | None]], list[str]]:
    found: dict[str, tuple[Path, str | None]] = {}
    problems: list[str] = []
    seen: set[str] = set()
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in STRUCTURE_MANIFEST_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise GlobalEsmCacheError(f"Structure manifest {manifest_path.name} lacks the columns {missing}.")
        for row in reader:
            name = (row.get("structure_name") or "").strip()
            stem = Path(name).stem
            if stem not in wanted:
                continue
            relative = (row.get("relative_path") or "").strip()
            digest = (row.get("sha256") or "").strip().lower()
            if stem in seen:
                problems.append(f"{stem}: listed more than once in {manifest_path.name}")
            elif Path(name).suffix.lower() not in STRUCTURE_SUFFIXES or not relative or Path(relative).is_absolute():
                problems.append(f"{stem}: invalid structure manifest row")
            elif not is_sha256(digest):
                problems.append(f"{stem}: the structure manifest row records no SHA-256")
            else:
                found[stem] = (manifest_path.parent / relative, digest)
            seen.add(stem)
    return found, problems


def resolve_structure_paths(
    structure_dir: Path,
    structure_ids: Sequence[str],
    roots: Sequence[Path],
    *,
    structure_manifest: Path | None = None,
) -> dict[str, tuple[Path, str | None]]:
    """Map each ID to its structure file and, in manifest mode, the SHA-256 recorded for it.

    Without ``structure_manifest`` only ``<structure_dir>/<id><suffix>`` is tried and no
    manifest is opened. With one, only rows whose stem is requested are resolved,
    relative to the manifest's folder, and each of them must record a SHA-256.
    """
    wanted = set(structure_ids)
    if structure_manifest is None:
        found, problems = _structure_files_by_name(Path(structure_dir), wanted)
    else:
        found, problems = _structure_rows_from_manifest(_inside_roots(Path(structure_manifest), roots), wanted)
    problems.extend(f"{structure_id}: no structure file" for structure_id in sorted(wanted - set(found)))
    resolved: dict[str, tuple[Path, str | None]] = {}
    for structure_id, (path, digest) in sorted(found.items()):
        try:
            resolved[structure_id] = (_inside_roots(path, roots), digest)
        except GlobalEsmCacheError as exc:
            problems.append(f"{structure_id}: {exc}")
    if problems:
        raise GlobalEsmCacheError("Structure inputs are incomplete:\n  " + "\n  ".join(problems[:20]))
    return resolved


# ---------------------------------------------------------------------------
# Chain inputs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChainInput:
    structure_id: str
    chain_id: str
    structure_file: str
    structure_sha256: str
    sequence: str
    sequence_sha256: str
    residue_ids: tuple[ResidueKey, ...]


def load_chain_input(structure_id: str, structure_path: Path, expected_sha256: str | None = None) -> ChainInput:
    digest = sha256_bytes(Path(structure_path).read_bytes())
    if expected_sha256 is not None and digest != expected_sha256:
        raise GlobalEsmCacheError(f"{structure_id}: structure SHA-256 differs from the structure manifest.")
    chain_id = chain_id_from_structure_id(structure_id)
    structure = parse_structure(structure_path)
    sequences = extract_chain_sequences(structure)
    if chain_id not in sequences:
        raise GlobalEsmCacheError(f"{structure_id}: chain {chain_id!r} has no polymer residues.")
    sequence = sequences[chain_id]
    residue_ids, _with_ca = residue_keys_for_structure_chain(structure, chain_id)
    if len(residue_ids) != len(sequence):
        raise GlobalEsmCacheError(f"{structure_id}: {len(residue_ids)} residue IDs for a {len(sequence)}-residue sequence.")
    return ChainInput(
        structure_id=structure_id,
        chain_id=chain_id,
        structure_file=Path(structure_path).name,
        structure_sha256=digest,
        sequence=sequence,
        sequence_sha256=sha256_text(sequence),
        residue_ids=tuple(residue_ids),
    )


def _load_chain_inputs(
    structure_dir: Path,
    structure_ids: Sequence[str],
    roots: Sequence[Path],
    *,
    structure_manifest: Path | None = None,
) -> list[ChainInput]:
    paths = resolve_structure_paths(structure_dir, structure_ids, roots, structure_manifest=structure_manifest)
    chains: list[ChainInput] = []
    problems: list[str] = []
    for structure_id in sorted(structure_ids):
        path, digest = paths[structure_id]
        try:
            chains.append(load_chain_input(structure_id, path, digest))
        except (GlobalEsmCacheError, OSError, ValueError) as exc:
            problems.append(f"{structure_id}: {exc}")
    if problems:
        raise GlobalEsmCacheError("Structure chains could not be read:\n  " + "\n  ".join(problems[:20]))
    return chains


def mean_from_residue_payload(
    chain: ChainInput,
    data: bytes,
    *,
    origin: str,
    model_name: str,
    embedding_dim: int,
) -> tuple[Tensor, Tensor]:
    """Validate a per-residue payload against its structure chain; returns (mean, rows)."""
    payload = deserialize_payload(data, origin=origin)
    if isinstance(payload, Tensor):
        raise GlobalEsmCacheError(f"{origin}: legacy raw-tensor payload; a format-v2 payload is required.")
    if not isinstance(payload, dict):
        raise GlobalEsmCacheError(f"{origin}: unsupported payload type {type(payload).__name__}.")
    metadata = payload.get("metadata")
    if payload.get("format_version") != RESIDUE_CACHE_FORMAT_VERSION or not isinstance(metadata, dict):
        raise GlobalEsmCacheError(f"{origin}: expected a format-v2 payload with metadata.")
    problems: list[str] = []
    if metadata.get("esm_model_name") != model_name:
        problems.append(f"model {metadata.get('esm_model_name')!r}, expected {model_name!r}")
    if metadata.get("embedding_dim") != embedding_dim:
        problems.append(f"metadata embedding_dim {metadata.get('embedding_dim')!r}, expected {embedding_dim}")
    if payload.get("structure_id") not in (None, chain.structure_id):
        problems.append(f"structure_id {payload.get('structure_id')!r}")
    if payload.get("chain_id") not in (None, chain.chain_id):
        problems.append(f"chain_id {payload.get('chain_id')!r}, expected {chain.chain_id!r}")
    if not metadata.get("source_sequence_sha256"):
        problems.append("payload records no source_sequence_sha256")
    elif metadata["source_sequence_sha256"] != chain.sequence_sha256:
        problems.append("sequence SHA-256 differs from the structure chain sequence")
    if "source_sequence" in metadata and metadata["source_sequence"] != chain.sequence:
        problems.append("recorded sequence differs from the structure chain sequence")
    rows = payload.get("embeddings")
    if not isinstance(rows, Tensor) or rows.dim() != 2 or not rows.is_floating_point():
        problems.append("embeddings must be a 2-D floating tensor")
    else:
        if rows.size(1) != embedding_dim:
            problems.append(f"embedding width {rows.size(1)}, expected {embedding_dim}")
        residue_ids = tuple(deserialize_residue_ids(list(payload.get("residue_ids") or [])))
        if rows.size(0) != len(residue_ids):
            problems.append(f"{rows.size(0)} rows for {len(residue_ids)} residue IDs")
        elif residue_ids != chain.residue_ids:
            problems.append("residue IDs differ from the structure chain residues")
        if not bool(torch.isfinite(rows).all()):
            problems.append("embeddings hold non-finite values")
    if problems:
        raise GlobalEsmCacheError(f"{origin}: " + "; ".join(problems))
    return compute_mean(rows), rows


# ---------------------------------------------------------------------------
# from-residue-cache
# ---------------------------------------------------------------------------

def _write_receipt(out_dir: Path, receipt: dict[str, Any]) -> Path:
    path = Path(out_dir) / BUILD_RECEIPT_FILENAME
    payload = {"generated_at": utc_timestamp(), "code_version": git_commit_hash(), **receipt}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def build_from_residue_cache(
    allowlist_path: Path,
    structure_dir: Path,
    embeddings_dir: Path,
    out_dir: Path,
    *,
    model_name: str = DEFAULT_ESM_MODEL_NAME,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    permitted_roots: Sequence[Path] = (),
    structure_manifest: Path | None = None,
    expected_allowlist_sha256: str | None = None,
) -> dict[str, Any]:
    allowlist = read_id_allowlist(allowlist_path, expected_sha256=expected_allowlist_sha256)
    roots = _permitted_roots([structure_dir, embeddings_dir, *permitted_roots])
    out_dir = _prepare_output_dir(out_dir)
    chains = _load_chain_inputs(structure_dir, allowlist.ids, roots, structure_manifest=structure_manifest)

    files: list[CacheFile] = []
    problems: list[str] = []
    for chain in chains:
        embedding_path = expected_embedding_path(Path(f"{chain.structure_id}.pdb"), chain.chain_id, Path(embeddings_dir))
        try:
            data = _inside_roots(embedding_path, roots).read_bytes()
            mean, rows = mean_from_residue_payload(
                chain, data, origin=embedding_path.name, model_name=model_name, embedding_dim=embedding_dim
            )
            source = {
                "kind": "per_residue_cache",
                "embedding_file": embedding_path.name,
                "embedding_sha256": sha256_bytes(data),
                "embedding_format_version": RESIDUE_CACHE_FORMAT_VERSION,
                "embedding_storage_dtype": str(rows.dtype).removeprefix("torch."),
                "structure_file": chain.structure_file,
                "structure_sha256": chain.structure_sha256,
            }
            payload = chain_mean_payload(
                mean,
                structure_id=chain.structure_id,
                layer=FINAL_LAYER,
                n_residues=int(rows.size(0)),
                source=source,
                source_sequence_sha256=chain.sequence_sha256,
                model_name=model_name,
                embedding_dim=embedding_dim,
            )
            files.append(encode_cache_file(payload))
        except FileNotFoundError:
            problems.append(f"{chain.structure_id}: missing {embedding_path.name}")
        except (GlobalEsmCacheError, OSError, ValueError) as exc:
            problems.append(f"{chain.structure_id}: {exc}")
    if problems:
        raise GlobalEsmCacheError(
            f"{len(problems)} of {len(chains)} chains failed; nothing was written:\n  " + "\n  ".join(problems[:20])
        )

    for item in files:
        write_bytes_atomic(out_dir / item.file_name, item.data)
    manifest = build_manifest(
        files,
        kind=KIND_CHAIN_MEAN,
        allowlist=allowlist.manifest_record(),
        build={"action": "from-residue-cache", "embedding_file_pattern": "<stem>_chain_<chain>_esmc.pt"},
        model_name=model_name,
        embedding_dim=embedding_dim,
    )
    manifest_path, manifest_sha256 = write_manifest(out_dir, manifest)
    summary = {
        "action": "from-residue-cache",
        "n_structures": len(chains),
        "n_files": len(files),
        "n_reused": 0,
        "allowlist_sha256_checked": expected_allowlist_sha256 is not None,
        "structure_manifest": Path(structure_manifest).name if structure_manifest is not None else None,
        "manifest": manifest_path.name,
        "manifest_sha256": manifest_sha256,
    }
    _write_receipt(out_dir, summary)
    return summary


# ---------------------------------------------------------------------------
# Model forward (from-model and sequences)
# ---------------------------------------------------------------------------

def _flash_attention_flags(model: Any) -> list[str]:
    """Module paths whose flash-attention flag is set (the SDK picks flash attention at construction)."""
    flagged = []
    modules = model.named_modules() if hasattr(model, "named_modules") else [("", model)]
    for name, module in modules:
        for attribute in ("_use_flash_attn", "use_flash_attn"):
            if getattr(module, attribute, False) is True:
                flagged.append(f"{name or '<model>'}.{attribute}")
    return flagged


def configure_fp32_precision(model: Any, device: str) -> dict[str, Any]:
    """The fp32 path: float32 parameters (the SDK casts to bfloat16 off CPU), no flash attention, TF32 off.

    TF32 is set off and recorded by ``ec_clean30.runtime_precision.configure_and_record``. On CUDA the module is
    required (rule A2-GOV-3): without it the build refuses, because no other record proves the TF32-off state that
    ``pretrain-corpus attach`` checks. On CPU without the module the two backend flags are set off here and the
    record says so; such a record does not pass ``runtime_precision.validate_policy``, so attach refuses that cache.
    """
    flagged = _flash_attention_flags(model)
    if flagged:
        raise GlobalEsmCacheError(f"The fp32 forward refuses flash attention (set at construction): {flagged[:4]}.")
    kind = torch.device(device).type
    try:
        # Scientific file of the execution lane (contract X10); import_module honours a sys.modules entry.
        runtime_precision = importlib.import_module("ec_clean30.runtime_precision")
    except ImportError:
        if kind == "cuda":
            raise GlobalEsmCacheError("The fp32 forward on CUDA needs ec_clean30.runtime_precision to set and record "
                                      "TF32 off (rule A2-GOV-3); refused instead of a local fallback.") from None
        runtime_precision = None
    model.to(torch.float32)
    dtypes = sorted({str(parameter.dtype).removeprefix("torch.") for parameter in model.parameters()})
    if dtypes != ["float32"]:
        raise GlobalEsmCacheError(f"The fp32 forward needs float32 parameters, found {dtypes}.")
    if runtime_precision is None:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        policy = {"device": kind, "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False,
                  "recorded_by": "esmc_chain_means (ec_clean30.runtime_precision not available)"}
    else:
        policy = dict(runtime_precision.configure_and_record(kind))
    return {"parameter_dtypes": dtypes, "precision_policy": policy}


def load_esmc_for_layers(model_name: str, device: str) -> tuple[Any, tuple[Any, Any, Any]]:
    sdk = _esmc_sdk()
    model = sdk[0].from_pretrained(model_name, device=torch.device(device))
    model.eval()
    return model, sdk


def describe_runtime(model: Any, *, model_name: str, requested_device: str,
                     forward_precision: str = FORWARD_PRECISION_SDK,
                     precision: Mapping[str, Any] | None = None) -> dict[str, Any]:
    parameter = next(iter(model.parameters()), None)
    device = parameter.device if parameter is not None else torch.device(requested_device)
    versions = installed_package_versions()
    use_flash_attn = getattr(model, "_use_flash_attn", None)
    return {
        "esm_model_name": model_name,
        "requested_device": str(requested_device),
        "device": str(device),
        "parameter_dtype": str(parameter.dtype).removeprefix("torch.") if parameter is not None else None,
        # ESMC.logits wraps the forward in bfloat16 autocast on CUDA and in nothing elsewhere.
        "autocast_dtype": "bfloat16" if device.type == "cuda" else None,
        "use_flash_attn": bool(use_flash_attn) if use_flash_attn is not None else None,
        "forward_config": dict(FORWARD_CONFIG),
        "python": platform.python_version(),
        "versions": {name: versions.get(name) for name in RECORDED_PACKAGES},
        # Only the fp32 path adds these keys, so every historical runtime record (and resume check) is unchanged.
        **({"forward_precision": FORWARD_PRECISION_FP32, "autocast_dtype": None, "forward_call": "ESMC.forward",
            "precision": dict(precision or {})}
           if forward_precision == FORWARD_PRECISION_FP32 else {}),
    }


@dataclass(frozen=True)
class LayerMeans:
    final: Tensor
    block: Tensor
    n_layers: int
    emitted_dtypes: dict[str, str]


def forward_layer_means(
    model: Any,
    sdk: tuple[Any, Any, Any],
    sequence: str,
    *,
    n_tokens: int,
    layer_index: int,
    embedding_dim: int,
    forward_precision: str = FORWARD_PRECISION_SDK,
) -> LayerMeans:
    """One ESMC forward; means of the final layer and of block ``layer_index`` without BOS/EOS."""
    _model_cls, protein_cls, config_cls = sdk
    if forward_precision == FORWARD_PRECISION_FP32:
        embeddings, hidden = _fp32_forward(model, protein_cls, sequence)
    else:
        with torch.no_grad():
            output = model.logits(model.encode(protein_cls(sequence=sequence)), config_cls(**FORWARD_CONFIG))
        embeddings, hidden = output.embeddings, output.hidden_states
    means = _layer_means(embeddings, hidden, n_tokens=n_tokens, layer_index=layer_index, embedding_dim=embedding_dim)
    if forward_precision == FORWARD_PRECISION_FP32 and set(means.emitted_dtypes.values()) != {"float32"}:
        raise GlobalEsmCacheError(f"The fp32 forward emitted {means.emitted_dtypes}, not float32.")
    return means


def _autocast_active() -> bool:
    return bool(torch.is_autocast_enabled())


def _fp32_forward(model: Any, protein_cls: Any, sequence: str) -> tuple[Any, Any]:
    """``ESMC.forward`` under ``no_grad`` with autocast disabled (bypassing ``ESMC.logits``, which opens its own
    bfloat16 autocast on CUDA); a forward hook refuses any active autocast. ``FORWARD_CONFIG`` keeps every hidden
    state and the final embeddings, as ``ESMC.logits`` would return them."""
    encoded = model.encode(protein_cls(sequence=sequence))
    tokens = encoded.sequence
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
    seen: list[bool] = []

    def hook(_module: Any, _inputs: Any, _output: Any) -> None:
        seen.append(_autocast_active())

    target = getattr(model, "transformer", model)
    handle = target.register_forward_hook(hook) if hasattr(target, "register_forward_hook") else None
    try:
        with torch.no_grad(), torch.autocast(device_type=tokens.device.type, enabled=False):
            output = model.forward(sequence_tokens=tokens)
    finally:
        if handle is not None:
            handle.remove()
    if any(seen):
        raise GlobalEsmCacheError("Autocast was active inside the fp32 forward.")
    hidden = output.hidden_states
    if FORWARD_CONFIG["ith_hidden_layer"] != -1 and hidden is not None:
        index = FORWARD_CONFIG["ith_hidden_layer"]
        hidden = hidden[index:index + 1]
    return output.embeddings, hidden


def _layer_means(embeddings: Any, hidden: Any, *, n_tokens: int, layer_index: int, embedding_dim: int) -> LayerMeans:
    if embeddings is None or hidden is None:
        raise GlobalEsmCacheError("The model returned no embeddings or hidden states.")
    if embeddings.dim() == 3 and embeddings.size(0) == 1:
        embeddings = embeddings[0]
    if hidden.dim() == 4 and hidden.size(1) == 1:
        hidden = hidden[:, 0]
    if embeddings.dim() != 2 or hidden.dim() != 3:
        raise GlobalEsmCacheError(
            f"Unexpected output shapes: embeddings {tuple(embeddings.shape)}, "
            f"hidden states {tuple(hidden.shape)} (one sequence per forward is required)."
        )
    expected_positions = n_tokens + 2
    if embeddings.size(0) != expected_positions or hidden.size(1) != expected_positions:
        raise GlobalEsmCacheError(
            f"The model returned {embeddings.size(0)} positions for {n_tokens} residue tokens; "
            f"expected {expected_positions} (BOS and EOS included)."
        )
    if embeddings.size(1) != embedding_dim or hidden.size(2) != embedding_dim:
        raise GlobalEsmCacheError(f"The model returned width {embeddings.size(1)}, expected {embedding_dim}.")
    n_layers = int(hidden.size(0))
    if layer_index >= n_layers:
        raise GlobalEsmCacheError(f"Layer index {layer_index} is out of range for a {n_layers}-block model.")
    return LayerMeans(
        final=compute_mean(embeddings[1:-1]),
        block=compute_mean(hidden[layer_index, 1:-1]),
        n_layers=n_layers,
        emitted_dtypes={
            "final": str(embeddings.dtype).removeprefix("torch."),
            "hidden": str(hidden.dtype).removeprefix("torch."),
        },
    )


def _model_and_runtime(
    model: Any, sdk: Any, *, model_name: str, device: str, layer_index: int,
    forward_precision: str = FORWARD_PRECISION_SDK,
) -> tuple[Any, Any, dict[str, Any]]:
    _check_layer_index(layer_index)
    if forward_precision not in FORWARD_PRECISIONS:
        raise GlobalEsmCacheError(f"--forward-precision must be one of {FORWARD_PRECISIONS}.")
    precision = None
    if model is None:
        model, sdk = load_esmc_for_layers(model_name, device)  # the one loading hook; fp32 is applied below
    elif sdk is None:
        raise GlobalEsmCacheError("An injected model needs its SDK triple (model class, protein class, config class).")
    if forward_precision == FORWARD_PRECISION_FP32:
        precision = configure_fp32_precision(model, device)
    blocks = getattr(getattr(model, "transformer", None), "blocks", None)
    if blocks is not None and layer_index >= len(blocks):
        raise GlobalEsmCacheError(f"Layer index {layer_index} is out of range for a {len(blocks)}-block model.")
    return model, sdk, describe_runtime(model, model_name=model_name, requested_device=device,
                                        forward_precision=forward_precision, precision=precision)


def _reuse_existing(
    out_dir: Path,
    expected: Sequence[tuple[str, str]],
    *,
    kind: str,
    key: str,
    model_name: str,
    embedding_dim: int,
    fields: Mapping[str, Any],
    source: Mapping[str, Any],
) -> list[CacheFile] | None:
    """Return the files of ``key`` written by an earlier run of this build, or None when any is absent.

    ``fields`` are top-level payload fields and ``source`` the source-record fields known
    before the forward pass; each must equal this run's value, or the run is refused.
    """
    paths = [(out_dir / file_name, layer) for file_name, layer in expected]
    if not all(path.is_file() for path, _ in paths):
        return None
    reused: list[CacheFile] = []
    for path, layer in paths:
        data = path.read_bytes()
        payload = deserialize_payload(data, origin=path.name)
        validate_mean_payload(
            payload, kind=kind, key=key, layer=layer, embedding_dim=embedding_dim, model_name=model_name, origin=path.name
        )
        recorded = payload.get("source") if isinstance(payload.get("source"), dict) else {}
        differs = [name for name, value in fields.items() if payload.get(name) != value]
        differs += [f"source.{name}" for name, value in source.items() if recorded.get(name) != value]
        if differs:
            raise GlobalEsmCacheError(
                f"Existing {path.name} does not match this run ({', '.join(differs)}); "
                "remove it or use a new output directory."
            )
        reused.append(encode_cache_file(payload, data=data))
    return reused


# ---------------------------------------------------------------------------
# from-model
# ---------------------------------------------------------------------------

def build_from_model(
    allowlist_path: Path,
    structure_dir: Path,
    out_dir: Path,
    *,
    layer_index: int,
    device: str,
    model_name: str = DEFAULT_ESM_MODEL_NAME,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    permitted_roots: Sequence[Path] = (),
    structure_manifest: Path | None = None,
    expected_allowlist_sha256: str | None = None,
    resume: bool = False,
    model: Any = None,
    sdk: Any = None,
    forward_precision: str = FORWARD_PRECISION_SDK,
) -> dict[str, Any]:
    _check_layer_index(layer_index)
    allowlist = read_id_allowlist(allowlist_path, expected_sha256=expected_allowlist_sha256)
    roots = _permitted_roots([structure_dir, *permitted_roots])
    settings = {
        "action": "from-model",
        "esm_model_name": model_name,
        "embedding_dim": embedding_dim,
        "layer_index_zero_based": layer_index,
        "requested_device": str(device),
        "allowlist_sha256": allowlist.file_sha256,
    }
    if forward_precision != FORWARD_PRECISION_SDK:
        settings["forward_precision"] = forward_precision
    out_dir = _prepare_output_dir(out_dir, resume=resume, settings=settings)
    chains = _load_chain_inputs(structure_dir, allowlist.ids, roots, structure_manifest=structure_manifest)
    model, sdk, runtime = _model_and_runtime(
        model, sdk, model_name=model_name, device=device, layer_index=layer_index, forward_precision=forward_precision
    )
    block_layer = layer_name(layer_index)

    files: list[CacheFile] = []
    n_reused = 0
    for chain in chains:
        layers = (FINAL_LAYER, block_layer)
        expected = [(chain_mean_filename(chain.structure_id, layer, model_name), layer) for layer in layers]
        fields = {"source_sequence_sha256": chain.sequence_sha256, "n_residues": len(chain.sequence)}
        source = {
            "kind": "esmc_forward",
            "structure_file": chain.structure_file,
            "structure_sha256": chain.structure_sha256,
            "runtime": runtime,
            "token_trim": TOKEN_TRIM,
        }
        existing = None
        if resume:
            existing = _reuse_existing(
                out_dir, expected, kind=KIND_CHAIN_MEAN, key=chain.structure_id,
                model_name=model_name, embedding_dim=embedding_dim, fields=fields, source=source,
            )
        if existing is not None:
            files.extend(existing)
            n_reused += 1
            continue
        means = forward_layer_means(
            model, sdk, chain.sequence, n_tokens=len(chain.sequence), layer_index=layer_index,
            embedding_dim=embedding_dim, forward_precision=forward_precision,
        )
        for layer, mean in ((FINAL_LAYER, means.final), (block_layer, means.block)):
            payload = chain_mean_payload(
                mean,
                structure_id=chain.structure_id,
                layer=layer,
                n_residues=len(chain.sequence),
                source={**source, "n_layers": means.n_layers, "emitted_dtypes": means.emitted_dtypes},
                source_sequence_sha256=chain.sequence_sha256,
                model_name=model_name,
                embedding_dim=embedding_dim,
            )
            item = encode_cache_file(payload)
            write_bytes_atomic(out_dir / item.file_name, item.data)
            files.append(item)

    manifest = build_manifest(
        files,
        kind=KIND_CHAIN_MEAN,
        allowlist=allowlist.manifest_record(),
        build={"action": "from-model", "layer_index_zero_based": layer_index, "runtime": runtime},
        model_name=model_name,
        embedding_dim=embedding_dim,
    )
    manifest_path, manifest_sha256 = write_manifest(out_dir, manifest)
    summary = {
        "action": "from-model",
        "n_structures": len(chains),
        "n_files": len(files),
        "n_reused": n_reused,
        "layers": [FINAL_LAYER, block_layer],
        "allowlist_sha256_checked": expected_allowlist_sha256 is not None,
        "structure_manifest": Path(structure_manifest).name if structure_manifest is not None else None,
        "manifest": manifest_path.name,
        "manifest_sha256": manifest_sha256,
        "runtime": runtime,
    }
    _write_receipt(out_dir, summary)
    return summary


# ---------------------------------------------------------------------------
# sequences
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SequenceMap:
    """Accession to unmasked-sequence SHA-256 (and, when the map holds them, the sequences)."""

    sha256_by_accession: Mapping[str, str]
    sequence_by_accession: Mapping[str, str]
    file_sha256: str
    ids_sha256: str

    def manifest_record(self) -> dict[str, Any]:
        return {"file_sha256": self.file_sha256, "ids_sha256": self.ids_sha256, "n_ids": len(self.sha256_by_accession)}


def read_sequence_map(path: Path, *, expected_sha256: str | None = None) -> SequenceMap:
    path = Path(path)
    data = path.read_bytes()
    file_sha256 = sha256_bytes(data)
    if expected_sha256 is not None:
        require_sha256(data, expected_sha256, what=f"Sequence map {path.name}")
    reader = csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""), delimiter="\t")
    missing = [column for column in SEQUENCE_MAP_COLUMNS if column not in (reader.fieldnames or [])]
    if missing:
        raise GlobalEsmCacheError(f"Sequence map {path.name} lacks the columns {missing}.")
    hashes: dict[str, str] = {}
    sequences: dict[str, str] = {}
    problems: list[str] = []
    for line_number, row in enumerate(reader, start=2):
        accession = (row.get("accession") or "").strip()
        digest = (row.get("sha256") or "").strip().lower()
        sequence = (row.get("sequence") or "").strip()
        where = f"{path.name}:{line_number}"
        if not accession or any(ch.isspace() for ch in accession) or "/" in accession or "\\" in accession:
            problems.append(f"{where}: invalid accession {accession!r}")
        elif accession in hashes:
            problems.append(f"{where}: accession {accession} repeats")
        elif not is_sha256(digest):
            problems.append(f"{where}: {accession} has no valid sha256")
        elif sequence and (not _RESIDUE_SEQUENCE_RE.match(sequence) or sha256_text(sequence) != digest):
            problems.append(f"{where}: {accession} sequence is not uppercase residues hashing to its sha256")
        else:
            hashes[accession] = digest
            if sequence:
                sequences[accession] = sequence
    if problems:
        raise GlobalEsmCacheError("Sequence map refused:\n  " + "\n  ".join(problems[:20]))
    if not hashes:
        raise GlobalEsmCacheError(f"Sequence map {path.name} holds no accessions.")
    return SequenceMap(
        sha256_by_accession=hashes,
        sequence_by_accession=sequences,
        file_sha256=file_sha256,
        ids_sha256=canonical_ids_sha256(hashes),
    )


def parse_fasta(text: str, *, name: str) -> list[tuple[str, str]]:
    """(label, sequence) pairs; the label is the first word of the header."""
    records: list[tuple[str, str]] = []
    label: str | None = None
    parts: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if label is not None:
                records.append((label, "".join(parts)))
            words = line[1:].split()
            if not words:
                raise GlobalEsmCacheError(f"{name}: empty FASTA header.")
            label, parts = words[0], []
        elif label is None:
            raise GlobalEsmCacheError(f"{name}: sequence text before the first header.")
        else:
            parts.append("".join(line.split()))
    if label is not None:
        records.append((label, "".join(parts)))
    duplicates = sorted(label for label, count in Counter(item for item, _ in records).items() if count > 1)
    if duplicates:
        raise GlobalEsmCacheError(f"{name}: repeated labels {duplicates[:10]}")
    return records


def read_fasta(path: Path) -> list[tuple[str, str]]:
    return parse_fasta(Path(path).read_text(encoding="utf-8"), name=Path(path).name)


def residue_tokens(sequence: str) -> list[str]:
    tokens = _SEQUENCE_TOKEN_RE.findall(sequence)
    if not sequence or "".join(tokens) != sequence:
        stray = sorted(set(_SEQUENCE_TOKEN_RE.sub("", sequence)))
        raise GlobalEsmCacheError(
            f"Sequence holds characters other than uppercase residue letters and '{MASK_TOKEN}': {stray[:10]}"
        )
    return tokens


def label_accession(label: str, accessions: Collection[str]) -> tuple[str, int | None]:
    """Split a FASTA label into (accession, copy index); an exact accession match wins."""
    if label in accessions:
        return label, None
    match = _COPY_LABEL_RE.match(label)
    if match is not None and match.group("accession") in accessions:
        return match.group("accession"), int(match.group("copy"))
    raise GlobalEsmCacheError(
        f"Label {label!r} is neither a sequence-map accession nor '<accession>_<n>' of one "
        "(n written without leading zeros)."
    )


def masked_copy_problem(tokens: Sequence[str], source: str) -> str | None:
    """Why ``tokens`` is not a masked copy of ``source``, or None when it is one."""
    if len(tokens) != len(source):
        return f"has {len(tokens)} residue tokens; its source has {len(source)}"
    differing = [
        index for index, (token, residue) in enumerate(zip(tokens, source))
        if token != MASK_TOKEN and token != residue
    ]
    if differing:
        return f"differs from its source at non-mask positions {differing[:10]}"
    return None


def read_sequence_inputs(
    fasta_paths: Sequence[Path], sequence_map: SequenceMap
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    """Validate FASTA inputs (read in file-name order) against the sequence map.

    Returns the input records, the label rows and the unique sequences by SHA-256.
    """
    paths = sorted((Path(path) for path in fasta_paths), key=lambda path: path.name)
    names = [path.name for path in paths]
    if not names or len(set(names)) != len(names):
        raise GlobalEsmCacheError("Give one or more FASTA files with distinct file names.")
    expected_hashes = sequence_map.sha256_by_accession
    sources = dict(sequence_map.sequence_by_accession)
    inputs: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    sequences: dict[str, str] = {}
    copies: list[tuple[str, str, list[str]]] = []
    problems: list[str] = []
    for path in paths:
        data = path.read_bytes()
        records = parse_fasta(data.decode("utf-8"), name=path.name)
        if not records:
            problems.append(f"{path.name}: holds no records")
        inputs.append({"file": path.name, "sha256": sha256_bytes(data), "n_records": len(records)})
        for label, sequence in records:
            where = f"{path.name}:{label}"
            try:
                accession, copy_index = label_accession(label, expected_hashes)
                tokens = residue_tokens(sequence)
            except GlobalEsmCacheError as exc:
                problems.append(f"{where}: {exc}")
                continue
            digest = sha256_text(sequence)
            if copy_index is None:
                if digest != expected_hashes[accession]:
                    problems.append(f"{where}: sequence differs from the sequence map")
                    continue
                sources.setdefault(accession, sequence)
            else:
                copies.append((where, accession, tokens))
            sequences[digest] = sequence
            labels.append({
                "input": path.name, "label": label, "accession": accession,
                "copy_index": copy_index, "sequence_sha256": digest,
            })
    # Copies are checked after every input is read, so a source record may sit in any file.
    for where, accession, tokens in copies:
        source = sources.get(accession)
        if source is None:
            problems.append(
                f"{where}: masked copy has no source sequence for {accession} "
                "(add its unmasked record or a sequence column to the map)"
            )
        else:
            problem = masked_copy_problem(tokens, source)
            if problem is not None:
                problems.append(f"{where}: masked copy {problem}")
    if problems:
        raise GlobalEsmCacheError("FASTA input refused:\n  " + "\n  ".join(problems[:20]))
    return inputs, sorted(labels, key=lambda row: (row["input"], row["label"])), sequences


def build_sequence_means(
    fasta_paths: Sequence[Path],
    sequence_map_path: Path,
    out_dir: Path,
    *,
    layer_index: int,
    device: str,
    model_name: str = DEFAULT_ESM_MODEL_NAME,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    max_residue_tokens: int | None = None,
    expected_sequence_map_sha256: str | None = None,
    resume: bool = False,
    model: Any = None,
    sdk: Any = None,
    forward_precision: str = FORWARD_PRECISION_SDK,
) -> dict[str, Any]:
    _check_layer_index(layer_index)
    if max_residue_tokens is not None and (isinstance(max_residue_tokens, bool) or max_residue_tokens <= 0):
        raise GlobalEsmCacheError("--max-residue-tokens must be positive.")
    sequence_map = read_sequence_map(sequence_map_path, expected_sha256=expected_sequence_map_sha256)
    inputs, labels, sequences = read_sequence_inputs(fasta_paths, sequence_map)
    truncation = {"max_residue_tokens": max_residue_tokens, "kept": "first_residue_tokens"}
    settings = {
        "action": "sequences",
        "esm_model_name": model_name,
        "embedding_dim": embedding_dim,
        "layer_index_zero_based": layer_index,
        "requested_device": str(device),
        "sequence_map_sha256": sequence_map.file_sha256,
        "inputs": [{"file": item["file"], "sha256": item["sha256"]} for item in inputs],
        "mask_policy": MASK_POLICY,
        "truncation": truncation,
    }
    if forward_precision != FORWARD_PRECISION_SDK:
        settings["forward_precision"] = forward_precision  # absent for the historical path (resume unchanged)

    out_dir = _prepare_output_dir(out_dir, resume=resume, settings=settings)
    model, sdk, runtime = _model_and_runtime(
        model, sdk, model_name=model_name, device=device, layer_index=layer_index, forward_precision=forward_precision
    )
    block_layer = layer_name(layer_index)

    files: list[CacheFile] = []
    n_reused = 0
    for digest in sorted(sequences):
        tokens = residue_tokens(sequences[digest])
        used = tokens[:max_residue_tokens] if max_residue_tokens is not None else tokens
        n_masks = sum(1 for token in used if token == MASK_TOKEN)
        expected = [(sequence_mean_filename(digest, layer, model_name), layer) for layer in (FINAL_LAYER, block_layer)]
        fields = {"n_tokens": len(used), "n_tokens_input": len(tokens), "n_mask_tokens": n_masks}
        source = {
            "kind": "esmc_forward",
            "runtime": runtime,
            "token_trim": TOKEN_TRIM,
            "mask_policy": MASK_POLICY,
            "truncation": truncation,
        }
        existing = None
        if resume:
            existing = _reuse_existing(
                out_dir, expected, kind=KIND_SEQUENCE_MEAN, key=digest,
                model_name=model_name, embedding_dim=embedding_dim, fields=fields, source=source,
            )
        if existing is not None:
            files.extend(existing)
            n_reused += 1
            continue
        means = forward_layer_means(
            model, sdk, "".join(used), n_tokens=len(used), layer_index=layer_index, embedding_dim=embedding_dim,
            forward_precision=forward_precision,
        )
        for layer, mean in ((FINAL_LAYER, means.final), (block_layer, means.block)):
            payload = sequence_mean_payload(
                mean,
                sequence_sha256=digest,
                layer=layer,
                n_tokens=len(used),
                n_mask_tokens=n_masks,
                n_tokens_input=len(tokens),
                source={**source, "n_layers": means.n_layers, "emitted_dtypes": means.emitted_dtypes},
                model_name=model_name,
                embedding_dim=embedding_dim,
            )
            item = encode_cache_file(payload)
            write_bytes_atomic(out_dir / item.file_name, item.data)
            files.append(item)

    manifest = build_manifest(
        files,
        kind=KIND_SEQUENCE_MEAN,
        allowlist=sequence_map.manifest_record(),
        build={
            "action": "sequences", "layer_index_zero_based": layer_index, "runtime": runtime,
            "mask_policy": MASK_POLICY, "truncation": truncation,
        },
        model_name=model_name,
        embedding_dim=embedding_dim,
        extra={"inputs": inputs, "labels": labels},
    )
    manifest_path, manifest_sha256 = write_manifest(out_dir, manifest)
    summary = {
        "action": "sequences",
        "n_labels": len(labels),
        "n_unique_sequences": len(sequences),
        "n_files": len(files),
        "n_reused": n_reused,
        "layers": [FINAL_LAYER, block_layer],
        "sequence_map_sha256_checked": expected_sequence_map_sha256 is not None,
        "manifest": manifest_path.name,
        "manifest_sha256": manifest_sha256,
        "runtime": runtime,
    }
    _write_receipt(out_dir, summary)
    return summary


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

def compare_chain_mean_caches(
    cache_a: Path,
    manifest_a_sha256: str,
    cache_b: Path,
    manifest_b_sha256: str,
    structure_ids: Sequence[str],
    *,
    layer: str | int = FINAL_LAYER,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    model_name: str = DEFAULT_ESM_MODEL_NAME,
    min_cosine: float = PARITY_MIN_COSINE,
    max_abs_diff: float = PARITY_MAX_ABS_DIFF,
) -> dict[str, Any]:
    """Per-ID agreement of two caches; ``passed`` requires both thresholds to hold for every ID."""
    if not -1.0 <= min_cosine <= 1.0 or not max_abs_diff >= 0.0:
        raise GlobalEsmCacheError("Thresholds need -1 <= min cosine <= 1 and max abs diff >= 0.")
    layer = normalize_layer(layer)
    first = load_global_esm_vectors(
        cache_a, manifest_a_sha256, structure_ids, layer, embedding_dim, expected_model_name=model_name
    )
    second = load_global_esm_vectors(
        cache_b, manifest_b_sha256, structure_ids, layer, embedding_dim, expected_model_name=model_name
    )
    rows = []
    for structure_id in structure_ids:
        a, b = first[structure_id].double(), second[structure_id].double()
        rows.append({
            "structure_id": structure_id,
            "max_abs_diff": float((a - b).abs().max()),
            "cosine": float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
        })
    worst_diff = max(row["max_abs_diff"] for row in rows)
    worst_cosine = min(row["cosine"] for row in rows)
    return {
        "layer": layer,
        "n_structures": len(rows),
        "manifest_a_sha256": manifest_a_sha256,
        "manifest_b_sha256": manifest_b_sha256,
        "max_abs_diff": worst_diff,
        "min_cosine": worst_cosine,
        "lowest_cosine": sorted(rows, key=lambda row: (row["cosine"], row["structure_id"]))[:5],
        "thresholds": {"min_cosine": min_cosine, "max_abs_diff": max_abs_diff},
        "passed": worst_cosine >= min_cosine and worst_diff <= max_abs_diff,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_structure_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--allowlist", type=Path, required=True, help="File with one structure ID per line.")
    parser.add_argument("--allowlist-sha256", default=None,
                        help="Refuse the allowlist unless its bytes have this SHA-256.")
    parser.add_argument("--structure-dir", type=Path, required=True,
                        help="Folder holding <id>.pdb|.cif|.mmcif; opened by exact name only.")
    parser.add_argument("--structure-manifest", type=Path, default=None,
                        help="Opt-in structure manifest CSV listing training structures only (every row is read); "
                             "allowlisted rows must record a SHA-256. Default: exact file names, no manifest.")
    parser.add_argument("--permitted-root", type=Path, action="append", default=[],
                        help="Extra folder that inputs may resolve into (repeatable).")


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--layer-index", type=int, required=True,
                        help="Zero-based ESMC block whose output (before the final norm) is averaged.")
    parser.add_argument("--device", required=True, help="Torch device for the forward pass, e.g. cpu or cuda.")
    parser.add_argument("--forward-precision", choices=FORWARD_PRECISIONS, default=FORWARD_PRECISION_SDK,
                        help="'fp32': float32 parameters, ESMC.forward without autocast or flash attention, TF32 "
                             "off (amendment 2); 'sdk' keeps the historical ESMC.logits path.")
    parser.add_argument("--resume", action="store_true",
                        help="Continue an interrupted run in --out that was started with the same settings.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build whole-protein ESMC mean-vector caches from explicit allowlists.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-name", default=DEFAULT_ESM_MODEL_NAME, help="ESMC model name.")
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM, help="Embedding width.")
    actions = parser.add_subparsers(dest="action", required=True)

    residue = actions.add_parser("from-residue-cache", help="Chain means of existing per-residue payloads (CPU).")
    _add_structure_arguments(residue)
    residue.add_argument("--embeddings-dir", type=Path, required=True,
                         help="Folder holding <stem>_chain_<chain>_esmc.pt payloads; opened by exact name only.")
    residue.add_argument("--out", type=Path, required=True, help="New or empty output folder.")

    model = actions.add_parser("from-model", help="Final-layer and hidden-block chain means from ESMC.")
    _add_structure_arguments(model)
    model.add_argument("--out", type=Path, required=True)
    _add_model_arguments(model)

    sequences = actions.add_parser("sequences", help="Sequence-keyed means for FASTA input.")
    sequences.add_argument("--fasta", type=Path, action="append", required=True, help="FASTA input (repeatable).")
    sequences.add_argument("--sequence-map", type=Path, required=True,
                           help="Tab-separated map with accession and sha256 columns (optional sequence column); "
                                "labels must be <accession> or <accession>_<n>.")
    sequences.add_argument("--sequence-map-sha256", default=None,
                           help="Refuse the sequence map unless its bytes have this SHA-256.")
    sequences.add_argument("--out", type=Path, required=True)
    sequences.add_argument("--max-residue-tokens", type=int, default=None,
                           help="Keep only the first N residue tokens ('<mask>' counts as one). Default: no limit.")
    _add_model_arguments(sequences)

    subset = actions.add_parser("subset", help="Per-fold subset manifest of an existing cache.")
    subset.add_argument("--cache-dir", type=Path, required=True)
    subset.add_argument("--manifest-sha256", required=True)
    subset.add_argument("--manifest-name", default=MANIFEST_FILENAME)
    subset.add_argument("--ids", type=Path, required=True, help="File with one key per line.")
    subset.add_argument("--name", required=True, help="Subset name, e.g. fold0_train.")
    subset.add_argument("--copy-to", type=Path, default=None,
                        help="Copy the subset into this new folder instead of writing manifest__<name>.json.")

    compare = actions.add_parser("compare", help="Agreement of two chain-mean caches on listed IDs.")
    compare.add_argument("--cache-a", type=Path, required=True)
    compare.add_argument("--manifest-a-sha256", required=True)
    compare.add_argument("--cache-b", type=Path, required=True)
    compare.add_argument("--manifest-b-sha256", required=True)
    compare.add_argument("--ids", type=Path, required=True, help="File with one structure ID per line.")
    compare.add_argument("--layer", default=FINAL_LAYER, help="'final', 'blockNN' or a zero-based block index.")
    compare.add_argument("--min-cosine", type=float, default=PARITY_MIN_COSINE,
                         help=f"Fail when any cosine is lower (default {PARITY_MIN_COSINE}).")
    compare.add_argument("--max-abs-diff", type=float, default=PARITY_MAX_ABS_DIFF,
                         help=f"Fail when any difference is larger (default {PARITY_MAX_ABS_DIFF}).")
    compare.add_argument("--out", type=Path, default=None, help="Optional JSON report path.")
    compare.add_argument("--kind", choices=("chain_mean", "sequence_mean"), default="chain_mean",
                         help="'sequence_mean' compares two sequence-mean caches on sequence SHA-256 keys (final "
                              "layer, the amendment-2 GPU-versus-CPU sample check).")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    common = {"model_name": args.model_name, "embedding_dim": args.embedding_dim}
    try:
        if args.action == "from-residue-cache":
            result = build_from_residue_cache(
                args.allowlist, args.structure_dir, args.embeddings_dir, args.out,
                permitted_roots=args.permitted_root, structure_manifest=args.structure_manifest,
                expected_allowlist_sha256=args.allowlist_sha256, **common,
            )
        elif args.action == "from-model":
            result = build_from_model(
                args.allowlist, args.structure_dir, args.out, layer_index=args.layer_index, device=args.device,
                permitted_roots=args.permitted_root, structure_manifest=args.structure_manifest,
                expected_allowlist_sha256=args.allowlist_sha256, resume=args.resume,
                forward_precision=args.forward_precision, **common,
            )
        elif args.action == "sequences":
            result = build_sequence_means(
                args.fasta, args.sequence_map, args.out, layer_index=args.layer_index, device=args.device,
                max_residue_tokens=args.max_residue_tokens, expected_sequence_map_sha256=args.sequence_map_sha256,
                resume=args.resume, forward_precision=args.forward_precision, **common,
            )
        elif args.action == "subset":
            keys = read_id_allowlist(args.ids).ids
            if args.copy_to is not None:
                path, digest = copy_cache_subset(
                    args.cache_dir, args.copy_to, manifest_sha256=args.manifest_sha256, keys=keys,
                    subset_name=args.name, manifest_name=args.manifest_name,
                )
            else:
                path, digest = write_subset_manifest(
                    args.cache_dir, manifest_sha256=args.manifest_sha256, keys=keys,
                    subset_name=args.name, manifest_name=args.manifest_name,
                )
            result = {"action": "subset", "manifest": str(path), "manifest_sha256": digest, "n_ids": len(keys)}
        elif getattr(args, "kind", "chain_mean") == "sequence_mean":
            from ec_clean30.pretrain_corpus import compare_sequence_mean_samples

            result = compare_sequence_mean_samples(args.cache_a, args.manifest_a_sha256, args.cache_b,
                                                   args.manifest_b_sha256, read_id_allowlist(args.ids).ids)
            result["passed"] = bool(result.get("passed", result.get("status") == "passed"))
            if args.out is not None:
                args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        else:
            result = compare_chain_mean_caches(
                args.cache_a, args.manifest_a_sha256, args.cache_b, args.manifest_b_sha256,
                read_id_allowlist(args.ids).ids, layer=args.layer, min_cosine=args.min_cosine,
                max_abs_diff=args.max_abs_diff, **common,
            )
            if args.out is not None:
                args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except GlobalEsmCacheError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
