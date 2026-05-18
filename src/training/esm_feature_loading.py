from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch import Tensor

from training.labels import parse_structure_identity


ESM_FILE_RE = re.compile(r"^(?P<structure_id>.+)_chain_(?P<chain>[^_]+)_esmc\.pt$")
DEFAULT_ESMC_EMBED_DIM = 960
ResidueKey = Tuple[str, int, str]
ESM_METADATA_SIDECAR_SUFFIX = ".json"


def normalize_chain_id(chain_id: str) -> str:
    normalized = str(chain_id).strip()
    return normalized if normalized else "_"


def normalize_icode(icode: Any) -> str:
    raw = str(icode).strip()
    return raw if raw else ""


def residue_keys_for_structure_chain(structure, chain_id: str) -> Tuple[List[ResidueKey], List[ResidueKey]]:
    wanted_chain = normalize_chain_id(chain_id)
    first_model = next(structure.get_models())
    residue_keys: List[ResidueKey] = []
    residue_keys_with_ca: List[ResidueKey] = []

    for chain in first_model:
        current_chain = normalize_chain_id(chain.id)
        if current_chain != wanted_chain:
            continue
        for residue in chain:
            if residue.id[0] != " ":
                continue
            _, resseq, icode = residue.id
            key = (current_chain, int(resseq), normalize_icode(icode))
            residue_keys.append(key)
            if residue.has_id("CA"):
                residue_keys_with_ca.append(key)

    if not residue_keys:
        raise ValueError(f"Could not find chain {wanted_chain!r} in the parsed structure.")
    return residue_keys, residue_keys_with_ca


def deserialize_residue_ids(raw_residue_ids: List[Any]) -> List[ResidueKey]:
    residue_ids: List[ResidueKey] = []
    for item in raw_residue_ids:
        if isinstance(item, dict):
            chain_id = item.get("chain_id")
            resseq = item.get("resseq")
            icode = item.get("icode", "")
        elif isinstance(item, (list, tuple)) and len(item) == 3:
            chain_id, resseq, icode = item
        else:
            raise ValueError(f"Unsupported residue id entry in embedding payload: {item!r}")
        residue_ids.append((normalize_chain_id(chain_id), int(resseq), normalize_icode(icode)))
    return residue_ids


def serialize_residue_ids(residue_ids: List[ResidueKey]) -> List[dict[str, Any]]:
    return [
        {
            "chain_id": normalize_chain_id(chain_id),
            "resseq": int(resseq),
            "icode": normalize_icode(icode),
        }
        for chain_id, resseq, icode in residue_ids
    ]


def build_embedding_payload(
    embeddings: Tensor,
    residue_ids: List[ResidueKey],
    *,
    structure_id: str | None = None,
    chain_id: str | None = None,
    source_path: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if embeddings.dim() != 2:
        raise ValueError(f"Expected a 2D embeddings tensor, got shape {tuple(embeddings.shape)}.")
    if embeddings.size(0) != len(residue_ids):
        raise ValueError(
            f"Embedding payload row count mismatch: got {embeddings.size(0)} rows for {len(residue_ids)} residue ids."
        )
    payload = {
        "format_version": 2,
        "structure_id": structure_id,
        "chain_id": normalize_chain_id(chain_id) if chain_id is not None else None,
        "source_path": source_path,
        "residue_ids": serialize_residue_ids(residue_ids),
        "embeddings": embeddings.float().cpu(),
    }
    if metadata is not None:
        payload["metadata"] = dict(metadata)
    return payload


def embedding_metadata_sidecar_path(embedding_path: Path) -> Path:
    return embedding_path.with_name(embedding_path.name + ESM_METADATA_SIDECAR_SUFFIX)


def embedding_metadata_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(payload.get("metadata") or {})
    metadata.setdefault("format_version", payload.get("format_version"))
    metadata.setdefault("structure_id", payload.get("structure_id"))
    metadata.setdefault("chain_id", payload.get("chain_id"))
    metadata.setdefault("source_path", payload.get("source_path"))
    embeddings = payload.get("embeddings")
    if isinstance(embeddings, torch.Tensor) and embeddings.dim() == 2:
        metadata.setdefault("embedding_dim", int(embeddings.size(1)))
        metadata.setdefault("n_residues", int(embeddings.size(0)))
    return metadata


def write_embedding_metadata_sidecar(embedding_path: Path, metadata: dict[str, Any]) -> Path:
    sidecar_path = embedding_metadata_sidecar_path(embedding_path)
    sidecar_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return sidecar_path


def load_embedding_metadata_sidecar(embedding_path: Path) -> dict[str, Any] | None:
    sidecar_path = embedding_metadata_sidecar_path(embedding_path)
    if not sidecar_path.is_file():
        return None
    return json.loads(sidecar_path.read_text(encoding="utf-8"))


def summarize_esm_embedding_metadata(
    structure_files: Sequence[Path],
    embeddings_dir: Path,
    *,
    sample_size: int = 5,
) -> dict[str, Any]:
    embedding_files: list[Path] = []
    seen: set[Path] = set()
    for structure_path in structure_files:
        for candidate in embedding_path_candidates(embeddings_dir, structure_path):
            if candidate.is_file() and candidate not in seen:
                seen.add(candidate)
                embedding_files.append(candidate)

    sidecar_payloads: list[dict[str, Any]] = []
    missing_sidecars: list[str] = []
    for embedding_file in embedding_files:
        sidecar = load_embedding_metadata_sidecar(embedding_file)
        if sidecar is None:
            missing_sidecars.append(str(embedding_file))
            continue
        payload = dict(sidecar)
        payload["embedding_path"] = str(embedding_file)
        sidecar_payloads.append(payload)

    model_names = sorted(
        {
            str(payload.get("esm_model_name"))
            for payload in sidecar_payloads
            if payload.get("esm_model_name")
        }
    )
    embedding_dims = sorted(
        {
            int(payload.get("embedding_dim"))
            for payload in sidecar_payloads
            if payload.get("embedding_dim") is not None
        }
    )
    return {
        "embedding_files_found": len(embedding_files),
        "metadata_sidecars_found": len(sidecar_payloads),
        "metadata_sidecars_missing": len(missing_sidecars),
        "esm_model_names": model_names or (["unknown_in_older_embeddings"] if embedding_files else []),
        "embedding_dims": embedding_dims,
        "metadata_examples": sidecar_payloads[:sample_size],
        "missing_sidecar_examples": missing_sidecars[:sample_size],
    }


def embedding_tensor_and_keys_from_payload(
    payload: Any,
    *,
    structure,
    candidate_path: Path,
    fallback_chain_id: str,
) -> Tuple[Tensor, List[ResidueKey]]:
    if isinstance(payload, dict):
        if "embeddings" not in payload:
            raise ValueError(f"Embedding payload {candidate_path} is missing an 'embeddings' tensor.")
        raw_residue_ids = payload.get("residue_ids")
        if raw_residue_ids is None:
            raise ValueError(f"Embedding payload {candidate_path} is missing 'residue_ids'.")
        return payload["embeddings"].float(), deserialize_residue_ids(list(raw_residue_ids))

    if not isinstance(payload, torch.Tensor):
        raise ValueError(f"Unsupported embedding payload type {type(payload)!r} in {candidate_path}.")

    payload_match = ESM_FILE_RE.match(candidate_path.name)
    chain_id = payload_match.group("chain") if payload_match is not None else fallback_chain_id
    residue_keys, residue_keys_with_ca = residue_keys_for_structure_chain(structure, chain_id)
    if payload.size(0) == len(residue_keys):
        return payload.float(), residue_keys
    if payload.size(0) == len(residue_keys_with_ca):
        return payload.float(), residue_keys_with_ca

    raise ValueError(
        f"Embedding length mismatch for {candidate_path}: got {payload.size(0)} rows, "
        f"expected {len(residue_keys)} chain residues or {len(residue_keys_with_ca)} CA residues."
    )


def embedding_path_candidates(embeddings_dir: Path, structure_path: Path) -> List[Path]:
    candidates: List[Path] = []
    seen = set()

    def add_candidate(path: Path) -> None:
        key = str(path)
        if key in seen:
            return
        seen.add(key)
        candidates.append(path)

    try:
        _pdbid, chain_id, _ec = parse_structure_identity(structure_path.stem)
        normalized_chain_ids = [normalize_chain_id(chain_id)]
    except ValueError:
        normalized_chain_ids = []

    for chain_id in normalized_chain_ids:
        add_candidate(embeddings_dir / f"{structure_path.stem}_chain_{chain_id}_esmc.pt")
        add_candidate(embeddings_dir / structure_path.stem / f"{structure_path.stem}_chain_{chain_id}_esmc.pt")

    add_candidate(embeddings_dir / f"{structure_path.stem}_esmc.pt")
    add_candidate(embeddings_dir / structure_path.stem / f"{structure_path.stem}_esmc.pt")

    for candidate in sorted(embeddings_dir.glob(f"{structure_path.stem}*_esmc.pt")):
        add_candidate(candidate)

    nested_dir = embeddings_dir / structure_path.stem
    if nested_dir.is_dir():
        for candidate in sorted(nested_dir.glob("*_esmc.pt")):
            add_candidate(candidate)

    return candidates


def load_esm_lookup_for_structure(
    structure,
    structure_path: Path,
    embeddings_dir: Path,
) -> Dict[ResidueKey, Tensor]:
    try:
        _pdbid, default_chain_id, _ec = parse_structure_identity(structure_path.stem)
        fallback_chain_id = normalize_chain_id(default_chain_id)
    except ValueError:
        fallback_chain_id = "_"

    esm_lookup: Dict[ResidueKey, Tensor] = {}
    found_files: List[Path] = []
    for candidate in embedding_path_candidates(embeddings_dir, structure_path):
        if not candidate.is_file():
            continue

        payload = torch.load(candidate, map_location="cpu", weights_only=True)
        embeddings, residue_ids = embedding_tensor_and_keys_from_payload(
            payload,
            structure=structure,
            candidate_path=candidate,
            fallback_chain_id=fallback_chain_id,
        )
        if embeddings.dim() != 2:
            raise ValueError(f"Expected a 2D embedding tensor in {candidate}, got shape {tuple(embeddings.shape)}.")
        if embeddings.size(0) != len(residue_ids):
            raise ValueError(
                f"Embedding payload {candidate} has {embeddings.size(0)} rows for {len(residue_ids)} residue ids."
            )

        overlap = set(esm_lookup).intersection(residue_ids)
        if overlap:
            raise ValueError(f"Duplicate ESM residue ids detected while loading {candidate}: {sorted(overlap)[:5]}")

        for residue_id, embedding in zip(residue_ids, embeddings):
            esm_lookup[residue_id] = embedding.float()
        found_files.append(candidate)

    if not found_files:
        raise FileNotFoundError(f"No ESM embedding file found for {structure_path.stem} under {embeddings_dir}.")
    return esm_lookup
