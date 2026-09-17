"""Cache integrity checks use synthetic coordinates and tensors, without inference."""
import hashlib
import builtins
import importlib
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from complete_care_caches import audit_structure, cache_paths
from feature_extraction.core import default_feature_dict
from training.esm_feature_loading import (
    build_embedding_payload,
    embedding_metadata_from_payload,
    write_embedding_metadata_sidecar,
    embedding_path_candidates,
)


@pytest.fixture(autouse=True)
def without_optional_esm_sdk(monkeypatch):
    """Exercise the locked CPU environment even when the local SDK is installed."""
    original = builtins.__import__

    def restricted_import(name, *args, **kwargs):
        if name == "esm" or name.startswith("esm."):
            raise ModuleNotFoundError("No module named 'esm'", name="esm")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", restricted_import)


def make_cache(root):
    structure = root / "synthetic__chain_A__EC_1.1.1.1.pdb"
    structure.write_text(
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 20.00           C  \n"
        "TER\nEND\n"
    )
    embedding = root / "esm_embeddings" / f"{structure.stem}_chain_A_esmc.pt"
    embedding.parent.mkdir()
    payload = build_embedding_payload(
        torch.ones(1, 960), [("A", 1, "")],
        metadata={"source_sequence_sha256": hashlib.sha256(b"A").hexdigest()},
    )
    torch.save(payload, embedding)
    write_embedding_metadata_sidecar(embedding, embedding_metadata_from_payload(payload))
    external = cache_paths(root, structure)["external"][0]
    external.parent.mkdir(parents=True)
    external.write_text(json.dumps({
        "tooling": {"pka": "propka"},
        "residues": [{"chain_id": "A", "resseq": 1, "icode": "", "features": default_feature_dict()}],
    }))
    ring = cache_paths(root, structure)["ring"][0]
    ring.parent.mkdir(parents=True)
    ring.write_text("NodeId1\tInteraction\tNodeId2\nA:1:_:ALA\tHBOND\tA:1:_:ALA\n")
    return structure, embedding, external


def test_complete_cache_and_residue_tamper(tmp_path):
    structure, embedding, _ = make_cache(tmp_path)
    assert len(audit_structure(tmp_path, structure)) == 4
    payload = torch.load(embedding, weights_only=True)
    payload["residue_ids"][0]["resseq"] = 2
    torch.save(payload, embedding)
    with pytest.raises(ValueError, match="alignment"):
        audit_structure(tmp_path, structure)


def test_nonfinite_embedding(tmp_path):
    structure, embedding, _ = make_cache(tmp_path)
    payload = torch.load(embedding, weights_only=True)
    payload["embeddings"][0, 0] = float("nan")
    torch.save(payload, embedding)
    with pytest.raises(ValueError, match="tensor"):
        audit_structure(tmp_path, structure)


def test_external_alignment(tmp_path):
    structure, _, external = make_cache(tmp_path)
    payload = json.loads(external.read_text())
    payload["residues"][0]["resseq"] = 2
    external.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="External residue alignment"):
        audit_structure(tmp_path, structure)


def test_external_unavailable_propka(tmp_path):
    structure, _, external = make_cache(tmp_path)
    payload = json.loads(external.read_text())
    payload["tooling"]["pka"] = "unavailable"
    external.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="External features incomplete"):
        audit_structure(tmp_path, structure)


def test_exact_cache_excludes_longer_ec_annotation_alias(tmp_path):
    structure, embedding, _ = make_cache(tmp_path)
    alias = embedding.with_name(embedding.name.replace("1.1.1.1_chain", "1.1.1.1,2.1.1.1_chain"))
    alias.write_bytes(embedding.read_bytes())
    found = [p for p in embedding_path_candidates(embedding.parent, structure) if p.is_file()]
    assert found == [embedding]
    embedding.unlink()
    found = [p for p in embedding_path_candidates(embedding.parent, structure) if p.is_file()]
    assert found == [alias]  # Preserve the unambiguous historical fallback.


def test_structure_utilities_import_without_optional_inference_sdk():
    from embed_helpers import esmc

    importlib.reload(esmc)
    assert callable(esmc.parse_structure)


def test_requested_embedding_generation_reports_missing_optional_sdk():
    from embed_helpers.esmc import load_esmc_model

    with pytest.raises(RuntimeError, match="generation requires the optional 'esm' package"):
        load_esmc_model(device="cpu")
