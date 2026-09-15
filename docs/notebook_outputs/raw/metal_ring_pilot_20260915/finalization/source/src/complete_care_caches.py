"""Resume CARE feature preparation without changing structures, labels, or splits.

Run inventory first, then independent external/RING and GPU ESM phases, then
audit. Reports record input and output hashes; an incomplete phase exits nonzero.
Test-side preparation is feature extraction only, never model evaluation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
import math
from pathlib import Path
import tempfile

from structure_store import read_structure_manifest, sha256_file

CARE_ROOT = "CARE_task1_30_clusterRes30_train_test_metallo"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def cache_paths(data, path):
    return {
        "esm": list((data / "esm_embeddings").glob(f"{path.stem}_chain_*_esmc.pt")),
        "external": [data / "updated_feature_extraction" / path.stem / "residue_features.json"],
        "ring": [data / "RING_features" / path.stem / f"{path.name}_ringEdges"],
    }


def cpu_feature(kind, structure, data):
    # Workers publish completed outputs only, making interrupted runs resumable.
    from feature_extraction.generate_features import generate_feature_file_for_structure
    from embed_helpers.Interaction_edge import ring_create_results

    destination = cache_paths(data, structure)[kind][0]
    destination.parent.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".care-", dir=destination.parent.parent) as tmp:
        if kind == "external":
            output = generate_feature_file_for_structure(
                structure, output_root=Path(tmp), propka_ph=7.0, include_propka=True,
            )
            payload = json.loads(output.read_text())
            if not payload["residues"] or payload["tooling"]["pka"] != "propka":
                raise ValueError(f"External generation incomplete: {payload.get('warnings')}")
        else:
            output = ring_create_results(Path(tmp), structure)
            if not output.is_file() or not output.stat().st_size:
                raise ValueError("RING did not produce an edge table")
        destination.parent.mkdir(parents=True, exist_ok=True)
        for artifact in output.parent.iterdir():
            if artifact.is_file():
                artifact.replace(destination.parent / artifact.name)
    return str(destination)


def generate_esm(paths, data, revision, source_commit):
    import torch
    from huggingface_hub import hf_hub_download
    from esm.models.esmc import ESMC
    from esm.tokenization import get_esmc_model_tokenizers
    from embed_helpers.esmc import create_resi_embed_pt, extract_chain_sequences, parse_structure
    from training.esm_feature_loading import embedding_metadata_from_payload, write_embedding_metadata_sidecar

    if not torch.cuda.is_available():
        raise RuntimeError("ESM phase requires a working CUDA device")
    checkpoint = Path(hf_hub_download(
        "biohub/esmc-300m-2024-12", "data/weights/esmc_300m_2024_12_v0.pth", revision=revision,
    ))
    checkpoint_sha = sha256_file(checkpoint)
    model = ESMC(d_model=960, n_heads=15, n_layers=30,
                 tokenizer=get_esmc_model_tokenizers(), use_flash_attn=False)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model = model.eval().to("cuda")
    output_root = data / "esm_embeddings"
    output_root.mkdir(exist_ok=True)
    failures, outputs = [], []
    for index, path in enumerate(paths, 1):
        try:
            sequences = extract_chain_sequences(parse_structure(path))
            if any(len(seq) > 2046 for seq in sequences.values()):
                raise ValueError("Sequence exceeds conservative ESMC context limit 2046")
            with tempfile.TemporaryDirectory(prefix=".care-", dir=output_root) as tmp:
                generated = create_resi_embed_pt(path, Path(tmp), model=model, device="cuda")
                for artifact in generated:
                    payload = torch.load(artifact, map_location="cpu", weights_only=True)
                    payload["metadata"].update(
                        model_repository="biohub/esmc-300m-2024-12", model_revision=revision,
                        checkpoint_sha256=checkpoint_sha, esm_sdk_version=version("esm"),
                        code_version=source_commit, source_structure_sha256=sha256_file(path),
                    )
                    torch.save(payload, artifact)
                    write_embedding_metadata_sidecar(artifact, embedding_metadata_from_payload(payload))
                for artifact in Path(tmp).iterdir():
                    destination = output_root / artifact.name
                    artifact.replace(destination)
                    outputs.append(str(destination))
            print(f"ESM {index}/{len(paths)} complete: {path.name}", flush=True)
        except Exception as exc:
            failures.append({"structure": path.name, "error": str(exc)})
            print(f"ESM failed: {path.name}: {exc}", flush=True)
    return outputs, failures


def audit_structure(data, path):
    import torch
    from Bio.PDB.Polypeptide import is_aa
    from embed_helpers.esmc import extract_chain_sequences, parse_structure, expected_embedding_path
    from training.esm_feature_loading import (
        deserialize_residue_ids, residue_keys_for_structure_chain, load_esm_lookup_for_structure,
    )
    from graph.ring_edges import parse_ring_node_id
    from feature_extraction.constants import FEATURE_NAMES

    structure = parse_structure(path)
    # Exercise the training resolver too: individual valid files can still
    # collide through a legacy filename alias.
    load_esm_lookup_for_structure(structure, path, data / "esm_embeddings")
    outputs = []
    for chain, sequence in extract_chain_sequences(structure).items():
        embedding = expected_embedding_path(path, chain, data / "esm_embeddings")
        payload = torch.load(embedding, map_location="cpu", weights_only=True)
        tensor = payload["embeddings"]
        expected_keys, _ = residue_keys_for_structure_chain(structure, chain)
        if deserialize_residue_ids(payload["residue_ids"]) != expected_keys:
            raise ValueError(f"ESM residue alignment mismatch: {embedding.name}")
        if tuple(tensor.shape) != (len(sequence), 960) or not torch.isfinite(tensor).all():
            raise ValueError(f"Invalid ESM tensor: {embedding.name}")
        meta = json.loads(Path(str(embedding) + ".json").read_text())
        if meta.get("source_sequence_sha256") != hashlib.sha256(sequence.encode()).hexdigest():
            raise ValueError(f"ESM sequence provenance mismatch: {embedding.name}")
        outputs.extend([embedding, Path(str(embedding) + ".json")])
    external = cache_paths(data, path)["external"][0]
    payload = json.loads(external.read_text())
    if not payload["residues"] or payload["tooling"]["pka"] != "propka":
        raise ValueError(f"External features incomplete: {path.name}")
    expected_external = {
        (chain.id.strip() or "_", residue.id[1], residue.id[2].strip())
        for chain in next(structure.get_models()) for residue in chain
        if is_aa(residue, standard=True)
    }
    external_keys = [(r["chain_id"], r["resseq"], r["icode"]) for r in payload["residues"]]
    if len(set(external_keys)) != len(external_keys) or set(external_keys) != expected_external:
        raise ValueError(f"External residue alignment mismatch: {path.name}")
    for residue in payload["residues"]:
        expected_features = set(FEATURE_NAMES) | {f"{name}_missing" for name in FEATURE_NAMES}
        if set(residue["features"]) != expected_features:
            raise ValueError(f"External feature schema mismatch: {path.name}")
        if not all(math.isfinite(float(value)) for value in residue["features"].values()):
            raise ValueError(f"Nonfinite external feature: {path.name}")
    ring = cache_paths(data, path)["ring"][0]
    lines = ring.read_text().splitlines()
    if not lines or "NodeId1" not in lines[0]:
        raise ValueError(f"Invalid RING header: {path.name}")
    for line in lines[1:]:
        if line.strip():
            fields = line.split()
            parse_ring_node_id(fields[0])
            parse_ring_node_id(fields[2])
    outputs.extend([external, ring])
    return [{"path": str(p.relative_to(data)), "bytes": p.stat().st_size,
             "sha256": sha256_file(p)} for p in outputs]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--phase", choices=["inventory", "external", "ring", "esm", "audit"], required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--model-revision", help="Required immutable 40-character HF revision for ESM")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--repair-audit", type=Path,
                        help="Also regenerate external files flagged incomplete by a previous audit")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    data = args.data_root.resolve()
    report = {"phase": args.phase, "source_commit": args.source_commit,
              "started_at": datetime.now(timezone.utc).isoformat(), "splits": {}, "failures": []}
    unique = {}
    for split in ("train", "test"):
        refs = read_structure_manifest(data / CARE_ROOT / split)
        report["splits"][split] = {"structures": len(refs), "missing": {kind: 0 for kind in ("esm", "external", "ring")}}
        for ref in refs:
            previous = unique.setdefault(ref.structure_name, ref)
            if previous.sha256 != ref.sha256:
                raise ValueError(f"Conflicting structure identity: {ref.structure_name}")
            for kind, files in cache_paths(data, ref.path).items():
                if not files or not all(p.is_file() and p.stat().st_size for p in files):
                    report["splits"][split]["missing"][kind] += 1
    report["inputs"] = [{"structure": r.structure_name, "sha256": r.sha256} for r in unique.values()]
    versions = {}
    for package in ("torch", "esm", "biopython", "biotite", "propka", "torch-geometric", "numpy"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    report["versions"] = versions
    atomic_json(args.report, report)
    if args.phase in ("external", "ring", "esm"):
        pending = [r.path for r in unique.values() if not cache_paths(data, r.path)[args.phase]
                   or not all(p.is_file() and p.stat().st_size for p in cache_paths(data, r.path)[args.phase])]
        if args.repair_audit:
            if args.phase != "external":
                parser.error("--repair-audit is supported only for external features")
            previous_audit = json.loads(args.repair_audit.read_text())
            if previous_audit.get("phase") != "audit":
                parser.error("--repair-audit must reference an audit report")
            incomplete = {row["structure"] for row in previous_audit["failures"]
                          if row["error"].startswith("External features incomplete:")}
            pending = sorted(set(pending) | {unique[name].path for name in incomplete})
            report["repair_audit"] = str(args.repair_audit)
        report["pending_count"] = len(pending)
        report["generated"] = []
        if args.phase == "esm" and pending:
            import re
            if not re.fullmatch(r"[0-9a-f]{40}", args.model_revision or ""):
                parser.error("ESM generation requires an immutable --model-revision")
            report["generated"], report["failures"] = generate_esm(
                pending, data, args.model_revision, args.source_commit,
            )
        elif pending:
            with ProcessPoolExecutor(max_workers=min(args.jobs, len(pending))) as pool:
                futures = {pool.submit(cpu_feature, args.phase, p, data): p for p in pending}
                for future in as_completed(futures):
                    try:
                        report["generated"].append(future.result())
                    except Exception as exc:
                        report["failures"].append({"structure": futures[future].name, "error": str(exc)})
                    atomic_json(args.report, report)
                    print(f"{args.phase}: {len(report['generated'])}/{len(pending)}, failures={len(report['failures'])}", flush=True)
    elif args.phase == "audit":
        report["validated_files"] = []
        for ref in unique.values():
            try:
                report["validated_files"].extend(audit_structure(data, ref.path))
            except Exception as exc:
                report["failures"].append({"structure": ref.structure_name, "error": str(exc)})
        report["complete"] = not report["failures"] and not any(
            count for split in report["splits"].values() for count in split["missing"].values()
        )
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(args.report, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("inputs", "generated", "validated_files")}, indent=2))
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
