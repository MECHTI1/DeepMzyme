"""Whole-protein ESMC mean-vector caches: format, manifest, loader and pocket attachment.

This module owns interface contract C2 of the EC-class vs CLEAN30 campaign.

Cache layout
    One payload file per (structure, layer) or (sequence, layer)::

        <structure_id>__esmc300m_<final|blockNN>_chainmean.pt
        <sequence_sha256>__esmc300m_<final|blockNN>_seqmean.pt

    plus ``manifest.json``, which records the SHA-256 of the allowlist the cache
    was built from and the SHA-256 and size of every payload file. Each manifest
    entry also records the file's layer, zero-based layer index and layer role, so
    the manifest stands in for per-file metadata sidecars. Per-fold subset
    manifests (``manifest__<name>.json``) list a subset of the same files.

    File names never end in ``_esmc.pt``: the per-residue loaders glob that suffix
    (``training.esm_feature_loading.embedding_path_candidates``), so a chain-mean
    file can never be mistaken for a per-residue cache.

Layers
    ``final`` is the final normalized ESMC embedding (what the per-residue caches
    hold). ``blockNN`` is the output of zero-based transformer block NN, taken
    before the final norm (``LogitsConfig(return_hidden_states=True)``). Layers may
    be given as ``"final"``, ``"blockNN"``, or a zero-based index (int or digits).

Reading
    ``load_global_esm_vectors`` verifies the manifest hash, then opens exactly the
    files of the requested IDs by their computed names and verifies each file
    hash. It never lists or globs a directory, rejects files that resolve outside
    the cache directory, and returns either every requested vector or raises.

Payload writes go through an in-memory ``torch.save`` so the bytes do not depend
on the output file name; payloads and manifests are byte-deterministic for equal
inputs. Run-specific details (time, code version) belong in a separate receipt.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor


CHAIN_MEAN_FORMAT = "deepmzyme_chain_mean_v1"
SEQUENCE_MEAN_FORMAT = "deepmzyme_sequence_mean_v1"
MANIFEST_FORMAT = "deepmzyme_esm_mean_manifest_v1"
MANIFEST_FILENAME = "manifest.json"

KIND_CHAIN_MEAN = "chain_mean"
KIND_SEQUENCE_MEAN = "sequence_mean"
KEY_FIELDS = {KIND_CHAIN_MEAN: "structure_id", KIND_SEQUENCE_MEAN: "sequence_sha256"}
PAYLOAD_FORMATS = {KIND_CHAIN_MEAN: CHAIN_MEAN_FORMAT, KIND_SEQUENCE_MEAN: SEQUENCE_MEAN_FORMAT}
FILE_SUFFIXES = {KIND_CHAIN_MEAN: "_chainmean.pt", KIND_SEQUENCE_MEAN: "_seqmean.pt"}

CHAIN_MEAN_POOLING = "mean_all_residue_rows"
SEQUENCE_MEAN_POOLING = "mean_all_residue_tokens"
POOLINGS = {KIND_CHAIN_MEAN: CHAIN_MEAN_POOLING, KIND_SEQUENCE_MEAN: SEQUENCE_MEAN_POOLING}

DEFAULT_ESM_MODEL_NAME = "esmc_300m"
DEFAULT_EMBEDDING_DIM = 960
FINAL_LAYER = "final"
LAYER_ROLE_FINAL = "final_normalized"
LAYER_ROLE_BLOCK = "block_output_before_final_norm"

# Pocket metadata key that carries the protein's vector after attachment.
GLOBAL_ESM_METADATA_KEY = "global_esm_vector"

STRUCTURE_ID_RE = re.compile(r"^(?P<pdbid>[^_]+)__chain_(?P<chain>[^_]+)__EC_(?P<ec>.+)$")
_BLOCK_LAYER_RE = re.compile(r"^block(?P<index>[0-9]+)$")
_DIGITS_RE = re.compile(r"^[0-9]+$")
_MODEL_NAME_RE = re.compile(r"^[a-z0-9_]+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SUBSET_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_MANIFEST_NAME_RE = re.compile(r"^manifest(__[A-Za-z0-9][A-Za-z0-9_.-]*)?\.json$")


class GlobalEsmCacheError(ValueError):
    """Raised when a mean-vector cache, manifest or attachment request is invalid."""


def _problem_list(problems: Sequence[str], limit: int = 20) -> str:
    shown = "\n  ".join(problems[:limit])
    more = f"\n  ... and {len(problems) - limit} more" if len(problems) > limit else ""
    return f"\n  {shown}{more}"


# ---------------------------------------------------------------------------
# Names and identities
# ---------------------------------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def is_sha256(text: str | None) -> bool:
    """True for a lowercase hexadecimal SHA-256 digest."""
    return isinstance(text, str) and _SHA256_RE.match(text) is not None


def require_sha256(data: bytes, expected_sha256: str, *, what: str) -> str:
    """Return the SHA-256 of ``data`` after checking that it equals ``expected_sha256``."""
    actual = sha256_bytes(data)
    if actual != (expected_sha256 or "").strip().lower():
        raise GlobalEsmCacheError(f"{what} has SHA-256 {actual}, expected {expected_sha256}.")
    return actual


def canonical_ids_sha256(ids: Iterable[str]) -> str:
    """SHA-256 of the sorted IDs, one per line with a trailing newline."""
    return sha256_text("".join(f"{item}\n" for item in sorted(ids)))


def model_file_tag(model_name: str) -> str:
    if not _MODEL_NAME_RE.match(model_name or ""):
        raise GlobalEsmCacheError(f"ESM model name {model_name!r} must use lowercase letters, digits and '_'.")
    return model_name.replace("_", "")


def layer_name(layer_index: int | None) -> str:
    """``None`` names the final normalized layer; an int names a zero-based block."""
    if layer_index is None:
        return FINAL_LAYER
    if isinstance(layer_index, bool) or not isinstance(layer_index, int) or layer_index < 0:
        raise GlobalEsmCacheError(f"Layer index must be a non-negative integer, got {layer_index!r}.")
    return f"block{layer_index:02d}"


def normalize_layer(layer: str | int | None) -> str:
    if layer is None or layer == FINAL_LAYER:
        return FINAL_LAYER
    if isinstance(layer, int):
        return layer_name(layer)
    if isinstance(layer, str) and _DIGITS_RE.match(layer):
        return layer_name(int(layer))
    match = _BLOCK_LAYER_RE.match(str(layer))
    if match is None:
        raise GlobalEsmCacheError(f"Unknown layer {layer!r}; use 'final', 'blockNN' or a zero-based block index.")
    return layer_name(int(match.group("index")))


def layer_index_from_name(layer: str | int | None) -> int | None:
    name = normalize_layer(layer)
    return None if name == FINAL_LAYER else int(name.removeprefix("block"))


def layer_role(layer: str | int | None) -> str:
    return LAYER_ROLE_FINAL if normalize_layer(layer) == FINAL_LAYER else LAYER_ROLE_BLOCK


def _validate_key(key: str, *, kind: str) -> str:
    if not isinstance(key, str) or not key or key != key.strip() or any(ch.isspace() for ch in key):
        raise GlobalEsmCacheError(f"Invalid {KEY_FIELDS[kind]} {key!r}.")
    if "/" in key or "\\" in key or key.startswith(".") or "\x00" in key:
        raise GlobalEsmCacheError(f"{KEY_FIELDS[kind]} {key!r} must be a plain name, not a path.")
    if kind == KIND_SEQUENCE_MEAN and not _SHA256_RE.match(key):
        raise GlobalEsmCacheError(f"Sequence key {key!r} is not a lowercase SHA-256 digest.")
    return key


def chain_id_from_structure_id(structure_id: str) -> str:
    match = STRUCTURE_ID_RE.match(structure_id)
    if match is None:
        raise GlobalEsmCacheError(
            f"Structure ID {structure_id!r} does not follow '<accession>__chain_<chain>__EC_<ec>'."
        )
    chain_id = match.group("chain").strip()
    return chain_id if chain_id else "_"


def mean_filename(key: str, layer: str | int | None, *, kind: str, model_name: str = DEFAULT_ESM_MODEL_NAME) -> str:
    _validate_key(key, kind=kind)
    name = f"{key}__{model_file_tag(model_name)}_{normalize_layer(layer)}{FILE_SUFFIXES[kind]}"
    assert not name.endswith("_esmc.pt")
    return name


def chain_mean_filename(structure_id: str, layer: str | int | None, model_name: str = DEFAULT_ESM_MODEL_NAME) -> str:
    return mean_filename(structure_id, layer, kind=KIND_CHAIN_MEAN, model_name=model_name)


def sequence_mean_filename(
    sequence_sha256: str, layer: str | int | None, model_name: str = DEFAULT_ESM_MODEL_NAME
) -> str:
    return mean_filename(sequence_sha256, layer, kind=KIND_SEQUENCE_MEAN, model_name=model_name)


@dataclass(frozen=True)
class IdAllowlist:
    """An explicit ID list read from a file: one ID per line, '#' comments and blank lines ignored."""

    ids: tuple[str, ...]  # sorted
    file_sha256: str
    ids_sha256: str

    def manifest_record(self) -> dict[str, Any]:
        return {"file_sha256": self.file_sha256, "ids_sha256": self.ids_sha256, "n_ids": len(self.ids)}


def read_id_allowlist(path: Path, *, expected_sha256: str | None = None) -> IdAllowlist:
    """Read an allowlist; with ``expected_sha256``, refuse a file whose bytes hash differently."""
    path = Path(path)
    data = path.read_bytes()
    if expected_sha256 is not None:
        require_sha256(data, expected_sha256, what=f"Allowlist {path.name}")
    ids: list[str] = []
    seen: set[str] = set()
    duplicates: list[str] = []
    for raw_line in data.decode("utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line in seen:
            duplicates.append(line)
            continue
        if any(ch.isspace() for ch in line) or "/" in line or "\\" in line:
            raise GlobalEsmCacheError(f"Allowlist {path.name} holds an invalid ID {line!r}.")
        seen.add(line)
        ids.append(line)
    if duplicates:
        raise GlobalEsmCacheError(f"Allowlist {path.name} repeats IDs: {sorted(set(duplicates))[:10]}")
    if not ids:
        raise GlobalEsmCacheError(f"Allowlist {path.name} holds no IDs.")
    return IdAllowlist(ids=tuple(sorted(ids)), file_sha256=sha256_bytes(data), ids_sha256=canonical_ids_sha256(ids))


# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------

def compute_mean(rows: Tensor) -> Tensor:
    """Mean over rows, accumulated in float64 and stored as float32."""
    if not isinstance(rows, Tensor) or rows.dim() != 2 or rows.size(0) == 0:
        shape = tuple(rows.shape) if isinstance(rows, Tensor) else type(rows).__name__
        raise GlobalEsmCacheError(f"Expected a non-empty 2-D tensor of rows, got {shape}.")
    if not rows.is_floating_point():
        raise GlobalEsmCacheError(f"Expected floating-point rows, got {rows.dtype}.")
    return rows.detach().cpu().double().mean(dim=0).float()


def _owned_copy(mean: Tensor) -> Tensor:
    """A CPU copy with its own storage: ``torch.save`` of a view would store its parent's whole storage."""
    return mean.detach().to("cpu").contiguous().clone()


def _checked_mean(mean: Tensor, embedding_dim: int, *, origin: str) -> Tensor:
    if not isinstance(mean, Tensor) or mean.dim() != 1 or mean.numel() != embedding_dim:
        shape = tuple(mean.shape) if isinstance(mean, Tensor) else type(mean).__name__
        raise GlobalEsmCacheError(f"{origin}: mean must have shape ({embedding_dim},), got {shape}.")
    if mean.dtype != torch.float32:
        raise GlobalEsmCacheError(f"{origin}: mean must be float32, got {mean.dtype}.")
    if not bool(torch.isfinite(mean).all()):
        raise GlobalEsmCacheError(f"{origin}: mean holds non-finite values.")
    return mean


def chain_mean_payload(
    mean: Tensor,
    *,
    structure_id: str,
    layer: str | int | None,
    n_residues: int,
    source: Mapping[str, Any],
    source_sequence_sha256: str,
    model_name: str = DEFAULT_ESM_MODEL_NAME,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
) -> dict[str, Any]:
    _validate_key(structure_id, kind=KIND_CHAIN_MEAN)
    layer = normalize_layer(layer)
    if not _SHA256_RE.match(source_sequence_sha256 or ""):
        raise GlobalEsmCacheError(f"{structure_id}: source_sequence_sha256 is not a SHA-256 digest.")
    if int(n_residues) <= 0:
        raise GlobalEsmCacheError(f"{structure_id}: n_residues must be positive.")
    return {
        "format": CHAIN_MEAN_FORMAT,
        "structure_id": structure_id,
        "chain_id": chain_id_from_structure_id(structure_id),
        "esm_model_name": model_name,
        "layer": layer,
        "layer_index_zero_based": layer_index_from_name(layer),
        "layer_role": layer_role(layer),
        "pooling": CHAIN_MEAN_POOLING,
        "embedding_dim": int(embedding_dim),
        "n_residues": int(n_residues),
        "mean": _owned_copy(_checked_mean(mean, embedding_dim, origin=structure_id)),
        "source": dict(source),
        "source_sequence_sha256": source_sequence_sha256,
    }


def sequence_mean_payload(
    mean: Tensor,
    *,
    sequence_sha256: str,
    layer: str | int | None,
    n_tokens: int,
    n_mask_tokens: int,
    n_tokens_input: int,
    source: Mapping[str, Any],
    model_name: str = DEFAULT_ESM_MODEL_NAME,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
) -> dict[str, Any]:
    _validate_key(sequence_sha256, kind=KIND_SEQUENCE_MEAN)
    layer = normalize_layer(layer)
    if int(n_tokens) <= 0 or int(n_tokens) > int(n_tokens_input) or int(n_mask_tokens) < 0:
        raise GlobalEsmCacheError(f"{sequence_sha256}: inconsistent token counts.")
    return {
        "format": SEQUENCE_MEAN_FORMAT,
        "sequence_sha256": sequence_sha256,
        "esm_model_name": model_name,
        "layer": layer,
        "layer_index_zero_based": layer_index_from_name(layer),
        "layer_role": layer_role(layer),
        "pooling": SEQUENCE_MEAN_POOLING,
        "embedding_dim": int(embedding_dim),
        "n_tokens": int(n_tokens),
        "n_tokens_input": int(n_tokens_input),
        "n_mask_tokens": int(n_mask_tokens),
        "truncated": int(n_tokens) < int(n_tokens_input),
        "mean": _owned_copy(_checked_mean(mean, embedding_dim, origin=sequence_sha256)),
        "source": dict(source),
    }


def serialize_payload(payload: Mapping[str, Any]) -> bytes:
    buffer = io.BytesIO()
    torch.save(dict(payload), buffer)
    return buffer.getvalue()


def deserialize_payload(data: bytes, *, origin: str) -> Any:
    try:
        return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    except Exception as exc:  # torch raises several unrelated types for corrupt archives
        raise GlobalEsmCacheError(f"{origin}: cannot read payload ({type(exc).__name__}: {exc}).") from exc


def write_bytes_atomic(path: Path, data: bytes) -> None:
    path = Path(path)
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(data)
    os.replace(partial, path)


def validate_mean_payload(
    payload: Any,
    *,
    kind: str,
    key: str,
    layer: str | int | None,
    embedding_dim: int,
    model_name: str,
    origin: str,
) -> Tensor:
    """Check a decoded payload against its expected identity; returns the mean vector."""
    if not isinstance(payload, dict):
        raise GlobalEsmCacheError(f"{origin}: payload must be a dict, got {type(payload).__name__}.")
    layer = normalize_layer(layer)
    expected = {
        "format": PAYLOAD_FORMATS[kind],
        KEY_FIELDS[kind]: key,
        "esm_model_name": model_name,
        "layer": layer,
        "layer_index_zero_based": layer_index_from_name(layer),
        "pooling": POOLINGS[kind],
        "embedding_dim": int(embedding_dim),
    }
    if kind == KIND_CHAIN_MEAN:
        expected["chain_id"] = chain_id_from_structure_id(key)
    mismatched = [
        f"{field}={payload.get(field)!r} (expected {value!r})"
        for field, value in expected.items()
        if payload.get(field) != value
    ]
    if mismatched:
        raise GlobalEsmCacheError(f"{origin}: payload identity mismatch: {', '.join(mismatched)}.")
    return _checked_mean(payload.get("mean"), embedding_dim, origin=origin)


# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CacheFile:
    """One encoded payload ready to be written, with its manifest entry."""

    file_name: str
    data: bytes
    entry: dict[str, Any]


def encode_cache_file(payload: Mapping[str, Any], *, data: bytes | None = None) -> CacheFile:
    """Encode ``payload``; pass ``data`` to describe already-written bytes of the same payload."""
    kind = next((name for name, fmt in PAYLOAD_FORMATS.items() if payload.get("format") == fmt), None)
    if kind is None:
        raise GlobalEsmCacheError(f"Unknown payload format {payload.get('format')!r}.")
    key = payload[KEY_FIELDS[kind]]
    file_name = mean_filename(key, payload["layer"], kind=kind, model_name=payload["esm_model_name"])
    if data is None:
        data = serialize_payload(payload)
    entry: dict[str, Any] = {
        KEY_FIELDS[kind]: key,
        "layer": payload["layer"],
        "layer_index_zero_based": layer_index_from_name(payload["layer"]),
        "layer_role": layer_role(payload["layer"]),
        "file": file_name,
        "sha256": sha256_bytes(data),
        "size_bytes": len(data),
    }
    if kind == KIND_CHAIN_MEAN:
        entry.update(
            chain_id=payload["chain_id"],
            n_residues=payload["n_residues"],
            source_sequence_sha256=payload["source_sequence_sha256"],
        )
    else:
        entry.update(n_tokens=payload["n_tokens"], n_mask_tokens=payload["n_mask_tokens"])
    return CacheFile(file_name=file_name, data=data, entry=entry)


def build_manifest(
    files: Sequence[CacheFile],
    *,
    kind: str,
    allowlist: Mapping[str, Any],
    build: Mapping[str, Any],
    model_name: str = DEFAULT_ESM_MODEL_NAME,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    key_field = KEY_FIELDS[kind]
    entries = sorted((item.entry for item in files), key=lambda entry: (entry[key_field], entry["layer"]))
    pairs = [(entry[key_field], entry["layer"]) for entry in entries]
    if len(pairs) != len(set(pairs)):
        raise GlobalEsmCacheError("A cache may hold only one file per key and layer.")
    layers = sorted({entry["layer"] for entry in entries})
    per_key = Counter(key for key, _ in pairs)
    keys = set(per_key)
    incomplete = sorted(key for key, count in per_key.items() if count != len(layers))
    if incomplete:
        raise GlobalEsmCacheError(f"Every key needs every layer; incomplete keys: {incomplete[:10]}")
    manifest: dict[str, Any] = {
        "format": MANIFEST_FORMAT,
        "kind": kind,
        "key_field": key_field,
        "esm_model_name": model_name,
        "embedding_dim": int(embedding_dim),
        "pooling": POOLINGS[kind],
        "layers": layers,
        "n_keys": len(keys),
        "allowlist": dict(allowlist),
        "build": dict(build),
        "subset": None,
        "files": entries,
    }
    for name, value in (extra or {}).items():
        if name in manifest:
            raise GlobalEsmCacheError(f"Manifest field {name!r} is reserved.")
        manifest[name] = value
    return manifest


def manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _check_manifest_name(name: str) -> str:
    if not _MANIFEST_NAME_RE.match(name or ""):
        raise GlobalEsmCacheError(f"Invalid manifest file name {name!r}.")
    return name


def write_manifest(directory: Path, manifest: Mapping[str, Any], *, name: str = MANIFEST_FILENAME) -> tuple[Path, str]:
    data = manifest_bytes(manifest)
    path = Path(directory) / _check_manifest_name(name)
    write_bytes_atomic(path, data)
    return path, sha256_bytes(data)


def _resolved_dir(directory: Path) -> Path:
    resolved = Path(os.path.realpath(directory))
    if not resolved.is_dir():
        raise GlobalEsmCacheError(f"Cache directory not found: {directory}")
    return resolved


def _read_direct_child(directory: Path, file_name: str) -> bytes:
    """Read ``directory/file_name`` only if it resolves to a file directly inside ``directory``."""
    if Path(file_name).name != file_name or file_name in {"", ".", ".."}:
        raise GlobalEsmCacheError(f"Manifest file name {file_name!r} is not a plain file name.")
    path = directory / file_name
    resolved = Path(os.path.realpath(path))
    if resolved.parent != directory:
        raise GlobalEsmCacheError(f"{file_name} resolves outside the cache directory ({resolved}).")
    if not resolved.is_file():
        raise GlobalEsmCacheError(f"{file_name} is missing from {directory}.")
    return resolved.read_bytes()


def read_manifest(cache_dir: Path, manifest_sha256: str, *, name: str = MANIFEST_FILENAME) -> dict[str, Any]:
    directory = _resolved_dir(Path(cache_dir))
    data = _read_direct_child(directory, _check_manifest_name(name))
    require_sha256(data, manifest_sha256, what=f"Manifest {name} in {cache_dir}")
    manifest = json.loads(data.decode("utf-8"))
    if not isinstance(manifest, dict) or manifest.get("format") != MANIFEST_FORMAT:
        raise GlobalEsmCacheError(f"{name} is not a {MANIFEST_FORMAT} manifest.")
    kind = manifest.get("kind")
    if kind not in KEY_FIELDS or manifest.get("key_field") != KEY_FIELDS[kind]:
        raise GlobalEsmCacheError(f"{name} declares an unknown cache kind {kind!r}.")
    if not isinstance(manifest.get("files"), list):
        raise GlobalEsmCacheError(f"{name} has no file list.")
    return manifest


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_mean_vectors(
    cache_dir: Path,
    manifest_sha256: str,
    keys: Sequence[str],
    expected_layer: str | int | None,
    expected_dim: int,
    *,
    kind: str,
    manifest_name: str,
    expected_model_name: str,
) -> dict[str, Tensor]:
    keys = list(keys)
    if not keys:
        raise GlobalEsmCacheError("No IDs were requested.")
    duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
    if duplicates:
        raise GlobalEsmCacheError(f"Requested IDs repeat: {duplicates[:10]}")
    for key in keys:
        _validate_key(key, kind=kind)
    layer = normalize_layer(expected_layer)
    directory = _resolved_dir(Path(cache_dir))
    manifest = read_manifest(directory, manifest_sha256, name=manifest_name)
    if manifest["kind"] != kind:
        raise GlobalEsmCacheError(f"Manifest describes a {manifest['kind']} cache, expected {kind}.")
    if manifest.get("esm_model_name") != expected_model_name:
        raise GlobalEsmCacheError(
            f"Manifest model {manifest.get('esm_model_name')!r} differs from expected {expected_model_name!r}."
        )
    if manifest.get("embedding_dim") != int(expected_dim):
        raise GlobalEsmCacheError(
            f"Manifest embedding_dim {manifest.get('embedding_dim')!r} differs from expected {expected_dim}."
        )
    if layer not in manifest.get("layers", []):
        raise GlobalEsmCacheError(f"Manifest holds layers {manifest.get('layers')}, not {layer!r}.")

    key_field = KEY_FIELDS[kind]
    entries: dict[str, dict[str, Any]] = {}
    for entry in manifest["files"]:
        if entry.get("layer") == layer:
            if entry.get(key_field) in entries:
                raise GlobalEsmCacheError(f"Manifest lists {entry.get(key_field)!r} twice for layer {layer}.")
            entries[entry.get(key_field)] = entry
    missing = [key for key in keys if key not in entries]
    if missing:
        raise GlobalEsmCacheError(
            f"{len(missing)} of {len(keys)} requested IDs have no {layer} vector in the manifest:"
            + _problem_list(missing)
        )

    vectors: dict[str, Tensor] = {}
    problems: list[str] = []
    for key in keys:
        entry = entries[key]
        expected_name = mean_filename(key, layer, kind=kind, model_name=expected_model_name)
        if entry.get("file") != expected_name:
            problems.append(f"{key}: manifest names {entry.get('file')!r}, expected {expected_name!r}")
            continue
        try:
            data = _read_direct_child(directory, expected_name)
            if sha256_bytes(data) != entry.get("sha256"):
                raise GlobalEsmCacheError(f"{expected_name}: SHA-256 differs from the manifest")
            payload = deserialize_payload(data, origin=expected_name)
            vectors[key] = validate_mean_payload(
                payload,
                kind=kind,
                key=key,
                layer=layer,
                embedding_dim=expected_dim,
                model_name=expected_model_name,
                origin=expected_name,
            )
        except (GlobalEsmCacheError, OSError) as exc:
            problems.append(f"{key}: {exc}")
    if problems:
        raise GlobalEsmCacheError(f"Could not load {len(problems)} of {len(keys)} vectors:" + _problem_list(problems))
    return vectors


def load_global_esm_vectors(
    cache_dir: Path,
    manifest_sha256: str,
    structure_ids: Sequence[str],
    expected_layer: str | int | None = FINAL_LAYER,
    expected_dim: int = DEFAULT_EMBEDDING_DIM,
    *,
    manifest_name: str = MANIFEST_FILENAME,
    expected_model_name: str = DEFAULT_ESM_MODEL_NAME,
) -> dict[str, Tensor]:
    """Return one float32 chain-mean vector per requested structure ID, in request order.

    All-or-nothing: raises ``GlobalEsmCacheError`` listing every missing or invalid ID.
    Only the manifest and the requested IDs' files are opened.
    """
    return _load_mean_vectors(
        cache_dir,
        manifest_sha256,
        structure_ids,
        expected_layer,
        expected_dim,
        kind=KIND_CHAIN_MEAN,
        manifest_name=manifest_name,
        expected_model_name=expected_model_name,
    )


def load_sequence_mean_vectors(
    cache_dir: Path,
    manifest_sha256: str,
    sequence_sha256s: Sequence[str],
    expected_layer: str | int | None = FINAL_LAYER,
    expected_dim: int = DEFAULT_EMBEDDING_DIM,
    *,
    manifest_name: str = MANIFEST_FILENAME,
    expected_model_name: str = DEFAULT_ESM_MODEL_NAME,
) -> dict[str, Tensor]:
    """Sequence-keyed counterpart of ``load_global_esm_vectors``."""
    return _load_mean_vectors(
        cache_dir,
        manifest_sha256,
        sequence_sha256s,
        expected_layer,
        expected_dim,
        kind=KIND_SEQUENCE_MEAN,
        manifest_name=manifest_name,
        expected_model_name=expected_model_name,
    )


# ---------------------------------------------------------------------------
# Subset manifests
# ---------------------------------------------------------------------------

def subset_manifest_name(subset_name: str) -> str:
    if not _SUBSET_NAME_RE.match(subset_name or ""):
        raise GlobalEsmCacheError(f"Invalid subset name {subset_name!r}.")
    return f"manifest__{subset_name}.json"


def select_manifest_subset(
    manifest: Mapping[str, Any],
    *,
    parent_manifest_sha256: str,
    keys: Sequence[str],
    subset_name: str,
) -> dict[str, Any]:
    """Manifest restricted to ``keys`` (every key must be present; all layers kept)."""
    subset_manifest_name(subset_name)
    keys = list(keys)
    if not keys:
        raise GlobalEsmCacheError("A subset needs at least one ID.")
    if len(set(keys)) != len(keys):
        raise GlobalEsmCacheError("Subset IDs repeat.")
    key_field = manifest["key_field"]
    available = {entry[key_field] for entry in manifest["files"]}
    missing = sorted(set(keys) - available)
    if missing:
        raise GlobalEsmCacheError(f"{len(missing)} subset IDs are not in the cache:" + _problem_list(missing))
    wanted = set(keys)
    subset = dict(manifest)
    subset["files"] = [entry for entry in manifest["files"] if entry[key_field] in wanted]
    subset["n_keys"] = len(wanted)
    if isinstance(manifest.get("labels"), list):
        subset["labels"] = [row for row in manifest["labels"] if row.get(key_field) in wanted]
    subset["subset"] = {
        "name": subset_name,
        "parent_manifest_sha256": parent_manifest_sha256,
        "parent_subset": manifest.get("subset"),
        "ids_sha256": canonical_ids_sha256(wanted),
        "n_ids": len(wanted),
    }
    return subset


def write_subset_manifest(
    cache_dir: Path,
    *,
    manifest_sha256: str,
    keys: Sequence[str],
    subset_name: str,
    manifest_name: str = MANIFEST_FILENAME,
) -> tuple[Path, str]:
    """Write ``manifest__<subset_name>.json`` next to the parent manifest; files are shared."""
    directory = _resolved_dir(Path(cache_dir))
    parent = read_manifest(directory, manifest_sha256, name=manifest_name)
    subset = select_manifest_subset(parent, parent_manifest_sha256=manifest_sha256, keys=keys, subset_name=subset_name)
    target_name = subset_manifest_name(subset_name)
    target = directory / target_name
    if target.exists():
        if target.read_bytes() != manifest_bytes(subset):
            raise GlobalEsmCacheError(f"{target_name} already exists with different content.")
        return target, sha256_bytes(target.read_bytes())
    return write_manifest(directory, subset, name=target_name)


def copy_cache_subset(
    source_dir: Path,
    dest_dir: Path,
    *,
    manifest_sha256: str,
    keys: Sequence[str],
    subset_name: str,
    manifest_name: str = MANIFEST_FILENAME,
) -> tuple[Path, str]:
    """Copy the subset's files (hash-verified) into an empty directory with its own manifest.json."""
    directory = _resolved_dir(Path(source_dir))
    parent = read_manifest(directory, manifest_sha256, name=manifest_name)
    subset = select_manifest_subset(parent, parent_manifest_sha256=manifest_sha256, keys=keys, subset_name=subset_name)
    dest_dir = Path(dest_dir)
    if dest_dir.exists() and any(dest_dir.iterdir()):
        raise GlobalEsmCacheError(f"Destination {dest_dir} is not empty.")
    dest_dir.mkdir(parents=True, exist_ok=True)
    for entry in subset["files"]:
        data = _read_direct_child(directory, entry["file"])
        if sha256_bytes(data) != entry["sha256"]:
            raise GlobalEsmCacheError(f"{entry['file']}: SHA-256 differs from the manifest.")
        write_bytes_atomic(dest_dir / entry["file"], data)
    return write_manifest(dest_dir, subset)


# ---------------------------------------------------------------------------
# Pocket attachment
# ---------------------------------------------------------------------------

def attach_global_esm_to_pockets(
    pockets: Sequence[Any],
    vectors: Mapping[str, Tensor],
    *,
    metadata_key: str = GLOBAL_ESM_METADATA_KEY,
) -> dict[str, Any]:
    """Set ``pocket.metadata[metadata_key]`` to its protein's vector for every pocket, or for none.

    Raises before changing any pocket when a structure has no vector, a pocket already
    carries the key, or the vectors are not finite 1-D tensors of one shared width.
    Every pocket gets its own float32 copy, so an in-place change to one pocket's
    vector (for example a standardisation) never reaches another pocket.
    """
    if not pockets:
        raise GlobalEsmCacheError("No pockets to attach global ESM vectors to.")
    structure_ids = [str(pocket.structure_id) for pocket in pockets]
    missing = sorted(set(structure_ids) - set(vectors))
    if missing:
        raise GlobalEsmCacheError(
            f"{len(missing)} structures have no global ESM vector:" + _problem_list(missing)
        )
    already = sorted({sid for sid, pocket in zip(structure_ids, pockets) if metadata_key in pocket.metadata})
    if already:
        raise GlobalEsmCacheError(f"Pockets already carry {metadata_key!r}:" + _problem_list(already))
    widths: set[int] = set()
    prepared: dict[str, Tensor] = {}
    for structure_id in sorted(set(structure_ids)):
        vector = vectors[structure_id]
        if not isinstance(vector, Tensor) or vector.dim() != 1 or not vector.is_floating_point():
            raise GlobalEsmCacheError(f"{structure_id}: global ESM vector must be a 1-D floating tensor.")
        if not bool(torch.isfinite(vector).all()):
            raise GlobalEsmCacheError(f"{structure_id}: global ESM vector holds non-finite values.")
        widths.add(int(vector.numel()))
        prepared[structure_id] = vector.detach().to(device="cpu", dtype=torch.float32)
    if len(widths) != 1:
        raise GlobalEsmCacheError(f"Global ESM vectors have mixed widths {sorted(widths)}.")
    for structure_id, pocket in zip(structure_ids, pockets):
        pocket.metadata[metadata_key] = prepared[structure_id].clone()
    return {
        "metadata_key": metadata_key,
        "n_pockets": len(pockets),
        "n_structures": len(prepared),
        "embedding_dim": widths.pop(),
        "n_unused_vectors": len(set(vectors) - set(prepared)),
    }
