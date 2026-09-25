"""Common-input certification for the PMM ion campaign (plan Step 2).

Three actions, all training-only:

``plan``      list every (structure, chain) whose residues enter a retained ion's
              10 Å context, with its first-model sequence and hash. Only these chains
              need ESMC embeddings. Identical sequences are embedded once.
``generate``  run ESMC-600M (``esmc_600m``, 1,152 dimensions) for the planned chains
              into a new versioned directory. Existing assets are reused only when
              model, dimension, sequence hash and residue alignment all match.
``certify``   load the complete frozen cohort through the strict training loader
              (fatal on any missing residue embedding), verify that the external
              feature channels are intentionally absent and masked, and write
              ``feature_inventory.json`` whose hash names the parse-cache namespace.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch  # noqa: E402

from benchmarking.pmm_ion_campaign import (  # noqa: E402
    ESM_DIM,
    ESM_MODEL_NAME,
    OMITTED_EXTERNAL_FEATURES,
    CampaignPaths,
    forbidden_read_roots,
    load_campaign,
    read_json,
    write_json,
)
from training.access_guard import install_forbidden_read_guard  # noqa: E402
from training.source_cohort import read_cohort_csv, sha256_file  # noqa: E402

PLAN_COLUMNS = ("structure_name", "chain", "sequence_sha256", "sequence_length", "n_ions_using_chain")
INVENTORY_SCHEMA_VERSION = 2


def _sequence_sha(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def _authorize_inputs(paths: CampaignPaths, source_dir: Path, reference_route_json: Path | None) -> None:
    if reference_route_json is None:
        install_forbidden_read_guard(forbidden_read_roots(source_dir))
    else:
        from benchmarking.pmm_ion_cohort import verify_reference_campaign_identity

        # The verifier checks completed-refit authorization before it opens any
        # reference metadata or structures, then binds their frozen identities.
        verify_reference_campaign_identity(source_dir, paths.root, reference_route_json)


def _planned_input_rows(paths: CampaignPaths, train_dir: Path) -> list[dict[str, Any]]:
    """Bind frozen plan rows to content-verified structures without parsing them."""
    from training.data import _cohort_structure_files

    manifest = load_campaign(paths)
    bindings = read_cohort_csv(paths.cohort, expected_sha256=manifest["cohort"]["sha256"])
    structures = {path.name: path for path in _cohort_structure_files(train_dir, bindings)}
    with paths.cohort.open(encoding="utf-8", newline="") as handle:
        needed = {(row["structure_name"], chain) for row in csv.DictReader(handle)
                  for chain in row["context_chains"].split(";")}
    plan_path = paths.root / "esm_generation_plan.csv"
    summary = read_json(paths.root / "esm_generation_plan.json")
    if sha256_file(plan_path) != summary["plan_csv_sha256"]:
        raise ValueError("ESM generation plan changed since it was frozen")
    with plan_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = [(row["structure_name"], row["chain"]) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != needed:
        raise ValueError("ESM generation plan does not cover exactly the frozen cohort chains")
    return [{**row, "structure_path": structures[row["structure_name"]]}
            for row in sorted(rows, key=lambda item: (item["structure_name"], item["chain"]))]


def _planned_inputs(paths: CampaignPaths, train_dir: Path) -> list[dict[str, Any]]:
    """Resolve frozen chains with sequences and exact residue order for certification."""
    from embed_helpers.esmc import extract_chain_sequences, parse_structure
    from training.esm_feature_loading import residue_keys_for_structure_chain

    output = []
    parsed_name, structure, sequences = None, None, None
    for row in _planned_input_rows(paths, train_dir):
        if row["structure_name"] != parsed_name:
            parsed_name = row["structure_name"]
            structure = parse_structure(row["structure_path"])
            sequences = extract_chain_sequences(structure)
        sequence = sequences[row["chain"]]
        if _sequence_sha(sequence) != row["sequence_sha256"] or len(sequence) != int(row["sequence_length"]):
            raise ValueError(f"{parsed_name}:{row['chain']} sequence differs from the frozen plan")
        residues, _ = residue_keys_for_structure_chain(structure, row["chain"])
        output.append({**row, "sequence": sequence, "residue_ids": residues})
    return output


def validate_embedding_file(path: Path, expected: dict[str, Any]) -> torch.Tensor:
    """Verify payload and sidecar together; a correctly shaped foreign sequence is invalid."""
    from training.esm_feature_loading import deserialize_residue_ids, embedding_metadata_from_payload

    payload = torch.load(path, map_location="cpu", weights_only=True)
    sidecar = read_json(path.with_name(path.name + ".json"))
    if not isinstance(payload, dict) or sidecar != embedding_metadata_from_payload(payload):
        raise ValueError(f"{path}: payload and metadata sidecar disagree")
    tensor = payload.get("embeddings")
    if (not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point()
            or tuple(tensor.shape) != (len(expected["residue_ids"]), ESM_DIM)
            or not bool(torch.isfinite(tensor).all())):
        raise ValueError(f"{path}: invalid embedding shape, dtype or nonfinite values")
    if (sidecar.get("esm_model_name") != ESM_MODEL_NAME
            or sidecar.get("embedding_dim") != ESM_DIM
            or sidecar.get("source_sequence_sha256") != expected["sequence_sha256"]
            or payload.get("structure_id") != Path(expected["structure_name"]).stem
            or payload.get("chain_id") != expected["chain"]):
        raise ValueError(f"{path}: embedding model, sequence or chain identity differs from the plan")
    if deserialize_residue_ids(payload.get("residue_ids", [])) != expected["residue_ids"]:
        raise ValueError(f"{path}: residue order differs from the frozen structure")
    return tensor


def _embedding_records(esm_dir: Path, planned: list[dict[str, Any]]) -> list[dict[str, str]]:
    records = []
    for row in planned:
        path = esm_dir / f"{Path(row['structure_name']).stem}_chain_{row['chain']}_esmc.pt"
        validate_embedding_file(path, row)
        records.append({"path": path.name, "sha256": sha256_file(path),
                        "sidecar_sha256": sha256_file(path.with_name(path.name + ".json")),
                        "sequence_sha256": row["sequence_sha256"]})
    return records


def verify_frozen_feature_inventory(paths: CampaignPaths, train_dir: Path, *,
                                    verify_payloads: bool = False,
                                    reference_route_json: Path | None = None) -> dict[str, Any]:
    """Rehash all certified inputs; optionally repeat full sequence/tensor validation."""
    _authorize_inputs(paths, train_dir, reference_route_json)
    inventory = read_json(paths.feature_inventory)
    manifest = load_campaign(paths)
    if (inventory.get("schema_version") != INVENTORY_SCHEMA_VERSION or not inventory.get("certified")
            or inventory.get("cohort_sha256") != manifest["cohort"]["sha256"]):
        raise ValueError("Feature inventory needs current, explicit certification")
    for directory in (paths.empty_external_features, paths.empty_esm):
        if not directory.is_dir() or any(directory.iterdir()):
            raise ValueError(f"{directory} must exist and remain empty")
    esm = inventory.get("esm")
    if esm is None:
        if reference_route_json is not None and inventory.get("certified_scope") == "structure_only":
            from training.data import _cohort_structure_files

            bindings = read_cohort_csv(paths.cohort, expected_sha256=manifest["cohort"]["sha256"])
            _cohort_structure_files(train_dir, bindings)
            return {"verified": True, "n_embedding_files": 0, "certified_scope": "structure_only",
                    "feature_inventory_sha256": sha256_file(paths.feature_inventory),
                    "payloads_verified": True}
        raise ValueError("Full campaign admission requires ESMC-600M certification")
    records = esm.get("files")
    if not records:
        raise ValueError("Feature inventory lacks frozen per-file content hashes; recertify explicitly")
    planned = _planned_input_rows(paths, train_dir)
    expected_files = [(f"{Path(row['structure_name']).stem}_chain_{row['chain']}_esmc.pt",
                       row["sequence_sha256"]) for row in planned]
    if ([(row["path"], row["sequence_sha256"]) for row in records] != expected_files
            or esm.get("n_files") != len(records)
            or esm.get("model_name") != ESM_MODEL_NAME or esm.get("embedding_dim") != ESM_DIM
            or esm.get("plan_csv_sha256") != sha256_file(paths.root / "esm_generation_plan.csv")):
        raise ValueError("Frozen embedding inventory differs from required cohort inputs")
    esm_dir = Path(esm["embeddings_dir"])
    for row in records:
        if Path(row["path"]).name != row["path"]:
            raise ValueError("Embedding inventory paths must be plain filenames")
        path = esm_dir / row["path"]
        if (sha256_file(path) != row["sha256"]
                or sha256_file(path.with_name(path.name + ".json")) != row["sidecar_sha256"]):
            raise ValueError(f"Frozen embedding content changed: {path}")
    if verify_payloads:
        for row in _planned_inputs(paths, train_dir):
            validate_embedding_file(esm_dir / f"{Path(row['structure_name']).stem}_chain_{row['chain']}_esmc.pt", row)
    return {"verified": True, "n_embedding_files": len(records),
            "feature_inventory_sha256": sha256_file(paths.feature_inventory),
            "payloads_verified": verify_payloads}


def plan_embeddings(paths: CampaignPaths, train_dir: Path, *,
                    reference_route_json: Path | None = None) -> dict[str, Any]:
    from embed_helpers.esmc import extract_chain_sequences, parse_structure
    from training.data import _cohort_structure_files

    _authorize_inputs(paths, train_dir, reference_route_json)
    manifest = load_campaign(paths)
    bindings = read_cohort_csv(paths.cohort, expected_sha256=manifest["cohort"]["sha256"])
    structures = {path.name: path for path in _cohort_structure_files(train_dir, bindings)}
    with paths.cohort.open(encoding="utf-8", newline="") as handle:
        context = {row["source_uid"]: row["context_chains"].split(";") for row in csv.DictReader(handle)}
    needed: dict[tuple[str, str], int] = defaultdict(int)
    for binding in bindings:
        for chain in context[binding.source_uid]:
            needed[(binding.structure_name, chain)] += 1
    rows = []
    by_structure: dict[str, list[str]] = defaultdict(list)
    for (name, chain) in needed:
        by_structure[name].append(chain)
    for index, name in enumerate(sorted(by_structure), start=1):
        sequences = extract_chain_sequences(parse_structure(structures[name]))
        for chain in sorted(by_structure[name]):
            if chain not in sequences:
                raise ValueError(f"{name}: context chain {chain!r} has no first-model sequence")
            rows.append({"structure_name": name, "chain": chain, "sequence_sha256": _sequence_sha(sequences[chain]),
                         "sequence_length": len(sequences[chain]), "n_ions_using_chain": needed[(name, chain)]})
        if index % 500 == 0:
            print(f"[ESM-PLAN] {index}/{len(by_structure)} structures", flush=True)
    plan_path = paths.root / "esm_generation_plan.csv"
    with plan_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PLAN_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    unique = {row["sequence_sha256"]: row["sequence_length"] for row in rows}
    residues = sum(row["sequence_length"] for row in rows)
    summary = {
        "n_chain_files": len(rows),
        "n_structures": len(by_structure),
        "n_unique_sequences": len(unique),
        "unique_sequence_residues": sum(unique.values()),
        "stored_residues": residues,
        "estimated_float32_storage_gb": residues * ESM_DIM * 4 / 1e9,
        "max_sequence_length": max(unique.values()),
        "plan_csv_sha256": sha256_file(plan_path),
    }
    write_json(paths.root / "esm_generation_plan.json", summary)
    return summary


def generate_embeddings(paths: CampaignPaths, train_dir: Path, out_dir: Path, *, device: str | None,
                        limit: int | None = None,
                        reference_route_json: Path | None = None) -> dict[str, Any]:
    """Embed each unique planned sequence once; write one payload per (structure, chain)."""
    from embed_helpers.esmc import (
        _esmc_sdk,
        clean_embedding_length,
        extract_chain_sequences,
        git_commit_hash,
        load_esmc_model,
        parse_structure,
        utc_timestamp,
    )
    from training.esm_feature_loading import (
        build_embedding_payload,
        embedding_metadata_from_payload,
        residue_keys_for_structure_chain,
        write_embedding_metadata_sidecar,
    )

    _authorize_inputs(paths, train_dir, reference_route_json)
    campaign_identity = load_campaign(paths)["campaign_id"]
    plan = _planned_inputs(paths, train_dir)
    if limit is not None:
        plan = plan[:limit]
    out_dir.mkdir(parents=True, exist_ok=True)
    by_structure: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in plan:
        by_structure[row["structure_name"]].append(row)

    def target_path(name: str, chain: str) -> Path:
        return out_dir / f"{Path(name).stem}_chain_{chain}_esmc.pt"

    # Index verified existing outputs before generation, including aliases later in
    # file order. This makes sequence deduplication survive process restarts.
    reusable_paths: set[Path] = set()
    existing_sequences: dict[str, Path] = {}
    for row in plan:
        path = target_path(row["structure_name"], row["chain"])
        if not path.is_file() or not path.with_name(path.name + ".json").is_file():
            continue
        try:
            validate_embedding_file(path, row)
        except (OSError, ValueError, RuntimeError, KeyError):
            continue
        reusable_paths.add(path)
        existing_sequences.setdefault(row["sequence_sha256"], path)
    # Loading the GPU model is unnecessary when every needed sequence exists.
    model, resolved_device, model_dtype = None, device, None
    storage_dtypes: set[str] = set()
    cache: dict[str, torch.Tensor] = {}
    written = reused = inferred = 0
    started = time.time()
    code_version = git_commit_hash()
    with torch.no_grad():
        for index, (name, rows) in enumerate(sorted(by_structure.items()), start=1):
            pending = [row for row in rows if target_path(name, row["chain"]) not in reusable_paths]
            reused += len(rows) - len(pending)
            if not pending:
                continue
            for row in pending:
                sequence = row["sequence"]
                embedding = cache.get(row["sequence_sha256"])
                if embedding is None:
                    source = existing_sequences.get(row["sequence_sha256"])
                    if source is not None:
                        embedding = torch.load(source, map_location="cpu", weights_only=True)["embeddings"]
                    else:
                        if model is None:
                            model, resolved_device = load_esmc_model(device, model_name=ESM_MODEL_NAME)
                            model_dtype = next(model.parameters()).dtype
                            _ESMC, ESMProtein, LogitsConfig = _esmc_sdk()
                        output = model.logits(model.encode(ESMProtein(sequence=sequence)),
                                              LogitsConfig(sequence=False, return_embeddings=True))
                        embedding = clean_embedding_length(output.embeddings, len(sequence)).detach().cpu()
                        inferred += 1
                    if embedding.size(1) != ESM_DIM:
                        raise ValueError(f"ESMC returned {embedding.size(1)} dims, expected {ESM_DIM}")
                    if not embedding.is_floating_point() or not bool(torch.isfinite(embedding).all()):
                        raise ValueError("ESMC returned nonfinite or non-floating embeddings")
                    # Preserve emitted precision. Parameter dtype alone does not prove
                    # that the SDK's final normalized embeddings have that dtype.
                    embedding = embedding.contiguous()
                    cache[row["sequence_sha256"]] = embedding
                storage_dtype = embedding.dtype
                storage_dtypes.add(str(storage_dtype))
                residue_ids = row["residue_ids"]
                payload = build_embedding_payload(
                    embedding.float(), residue_ids, structure_id=Path(name).stem, chain_id=row["chain"],
                    source_path=str(row["structure_path"]),
                    metadata={
                        "esm_model_name": ESM_MODEL_NAME, "esm_checkpoint_name": ESM_MODEL_NAME,
                        "embedding_dim": ESM_DIM, "n_residues": int(embedding.size(0)),
                        "generated_at": utc_timestamp(), "code_version": code_version,
                        "source_structure_id": Path(name).stem, "source_chain_id": row["chain"],
                        "source_sequence_length": len(sequence), "source_sequence_sha256": row["sequence_sha256"],
                        "campaign": campaign_identity,
                        "model_dtype": str(model_dtype).removeprefix("torch.") if model_dtype is not None else "reused",
                        "storage_dtype": str(storage_dtype).removeprefix("torch."),
                        "device": str(resolved_device),
                        "context_note": "full-length sequence, no truncation",
                    },
                )
                payload["embeddings"] = embedding  # stored in storage_dtype; loaders cast with .float()
                path = target_path(name, row["chain"])
                tmp = path.with_suffix(".pt.partial")
                torch.save(payload, tmp)
                tmp.replace(path)
                write_embedding_metadata_sidecar(path, embedding_metadata_from_payload(payload))
                written += 1
            if index % 100 == 0:
                rate = index / max(1e-9, time.time() - started)
                print(f"[ESM-GEN] {index}/{len(by_structure)} structures ({rate:.2f}/s), "
                      f"{written} written, {reused} reused, {len(cache)} unique sequences embedded", flush=True)
    summary = {"device": str(resolved_device), "model_dtype": str(model_dtype), "storage_dtypes": sorted(storage_dtypes),
               "written": written, "reused": reused, "unique_embedded": len(cache),
               "unique_inferred": inferred, "verified_existing_sequences": len(existing_sequences),
               "elapsed_seconds": time.time() - started, "out_dir": str(out_dir)}
    write_json(paths.root / "esm_generation_receipt.json", summary)
    return summary


def _external_channel_check(paths: CampaignPaths, graphs: list[Any]) -> dict[str, Any]:
    from data_structures import NODE_FEATURES_CONSERVATIVE

    empty_dir = paths.empty_external_features
    if any(empty_dir.iterdir()):
        raise ValueError(f"{empty_dir} must stay empty: external features are intentionally omitted")
    return {"external_features_root_dir": str(empty_dir), "directory_empty": True,
            "omitted_node_features": list(OMITTED_EXTERNAL_FEATURES),
            "zero_channels_verified_on_graphs": _zero_external_channels(graphs),
            "conservative_feature_names": list(NODE_FEATURES_CONSERVATIVE)}


def _zero_external_channels(graphs: list[Any]) -> bool:
    """External channels live in x_env_burial (SASA) and x_env_electrostatics (charge, dpKa)."""
    for graph in graphs:
        for field in ("x_env_burial", "x_env_electrostatics"):
            if hasattr(graph, field) and bool(getattr(graph, field).abs().sum() > 0):
                return False
    return True


def certify_inventory(paths: CampaignPaths, train_dir: Path, esm_dir: Path | None, *,
                      load_workers: int | None,
                      reference_route_json: Path | None = None) -> dict[str, Any]:
    """Certify common inputs. ``esm_dir=None`` certifies the structure-only scope (Only-GVP)."""
    from label_schemes import configure_active_metal_label_scheme
    from training.data import load_training_pockets_with_report_from_dir
    from training.graph_dataset import build_graph_data_list
    from training.esm_feature_loading import summarize_esm_embedding_metadata

    _authorize_inputs(paths, train_dir, reference_route_json)
    manifest = load_campaign(paths)
    paths.empty_external_features.mkdir(parents=True, exist_ok=True)
    paths.empty_esm.mkdir(parents=True, exist_ok=True)
    planned = _planned_inputs(paths, train_dir) if esm_dir is not None else []
    file_records = _embedding_records(esm_dir, planned) if esm_dir is not None else []
    configure_active_metal_label_scheme("split_all_metals")
    started = time.time()
    result = load_training_pockets_with_report_from_dir(
        structure_dir=train_dir,
        required_targets=("metal",),
        metal_example_unit="ion",
        metal_eligibility_scheme="six_class",
        esm_dim=ESM_DIM,
        esm_embeddings_dir=esm_dir if esm_dir is not None else paths.empty_esm,
        require_esm_embeddings=esm_dir is not None,
        external_features_root_dir=paths.empty_external_features,
        external_feature_source="updated",
        require_external_features=False,
        unsupported_metal_policy="error",
        invalid_structure_policy="error",
        source_cohort_csv=paths.cohort,
        source_cohort_sha256=manifest["cohort"]["sha256"],
        load_workers=load_workers,
    )
    pockets = result.pockets
    # Build every graph under the campaign contract, in chunks so memory stays bounded.
    n_graphs = empty_shell = 0
    zero_channels = True
    for start in range(0, len(pockets), 500):
        chunk = build_graph_data_list(pockets[start:start + 500], esm_dim=ESM_DIM, edge_radius=8.0,
                                      use_ring_edges=False, shell_role_source="geometry",
                                      node_feature_set="conservative",
                                      omit_node_features=OMITTED_EXTERNAL_FEATURES, metal_node_mode="none")
        n_graphs += len(chunk)
        empty_shell += sum(1 for graph in chunk if not bool((graph.x_role[:, 0] > 0.5).any()))
        zero_channels = zero_channels and _zero_external_channels(chunk)
        del chunk
    external = _external_channel_check(paths, [])
    external["zero_channels_verified_on_graphs"] = zero_channels
    if not zero_channels:
        raise ValueError("External feature channels are not all zero after omission")
    esm_record = None
    if esm_dir is not None:
        missing_esm = sum(1 for pocket in pockets for residue in pocket.residues if not residue.has_esm_embedding)
        if missing_esm:
            raise ValueError(f"{missing_esm} retained residues lack ESM embeddings")
        digest = hashlib.sha256()
        for record in file_records:
            digest.update(f"{record['path']}\t{record['sha256']}\n".encode())
        plan = read_json(paths.root / "esm_generation_plan.json")
        metadata_summary = summarize_esm_embedding_metadata(
            sorted({Path(p.metadata["source_path"]) for p in pockets}), esm_dir,
        )
        if metadata_summary["esm_model_names"] != [ESM_MODEL_NAME] or metadata_summary["embedding_dims"] != [ESM_DIM]:
            raise ValueError(f"Embedding metadata is not uniformly {ESM_MODEL_NAME}/{ESM_DIM}: {metadata_summary}")
        esm_record = {
            "model_name": ESM_MODEL_NAME, "embedding_dim": ESM_DIM, "embeddings_dir": str(esm_dir),
            "n_files": len(file_records), "files_sha256": digest.hexdigest(), "files": file_records,
            "plan_csv_sha256": plan["plan_csv_sha256"],
            "residue_coverage": {"retained_residues": sum(len(p.residues) for p in pockets), "missing": 0},
            "only_gvp": "does not load ESM; points at an intentionally empty directory",
        }
    elif any(residue.has_esm_embedding for pocket in pockets for residue in pocket.residues):
        raise ValueError("Structure-only certification unexpectedly loaded ESM embeddings")
    inventory = {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "campaign_id": manifest["campaign_id"],
        "certified": True,
        "certified_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "cohort_sha256": manifest["cohort"]["sha256"],
        "n_examples_loaded": len(pockets),
        "n_graphs_built": n_graphs,
        "n_empty_first_shell_graphs": empty_shell,
        "total_context_residues": sum(len(p.residues) for p in pockets),
        "certified_scope": "structure_and_esmc600m" if esm_record is not None else "structure_only",
        "esm": esm_record,
        "external_features": external,
        "graph_contract": {"edge_radius": 8.0, "pocket_radius": 10.0, "use_ring_edges": False,
                           "shell_role_source": "geometry", "first_shell_cutoff_angstrom": 2.7,
                           "metal_node_mode": "none", "node_feature_set": "conservative"},
        "parse_cache": {"root": str(paths.parse_cache),
                        "namespace_rule": "load settings + src hash + scheme + feature_inventory_sha256; "
                                          "raw examples only, never fold normalization"},
        "load_seconds": time.time() - started,
        "feature_fallbacks": len(result.feature_report.get("feature_fallbacks", [])),
    }
    write_json(paths.feature_inventory, inventory)
    return inventory


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("plan", "generate", "certify"))
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--esm-dir", type=Path, default=None,
                        help="New versioned ESMC-600M directory (default: <campaign>/inputs/esm_embeddings_esmc600m_v1).")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Generate only the first N planned chains (smoke).")
    parser.add_argument("--load-workers", type=int, default=None)
    parser.add_argument("--reference-route-json", type=Path, default=None,
                        help="Separately authorized frozen reference route; requires both completed final refits")
    parser.add_argument("--structure-only", action="store_true",
                        help="certify: structure/geometry inputs only (Only-GVP), before ESMC generation")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    paths = CampaignPaths(args.campaign_dir)
    esm_dir = args.esm_dir or (paths.root / "inputs" / "esm_embeddings_esmc600m_v1")
    if args.action == "certify" and args.structure_only:
        esm_dir = None
    if args.action == "plan":
        print(json.dumps(plan_embeddings(paths, args.train_dir, reference_route_json=args.reference_route_json), indent=2))
    elif args.action == "generate":
        print(json.dumps(generate_embeddings(paths, args.train_dir, esm_dir, device=args.device, limit=args.limit,
                                            reference_route_json=args.reference_route_json), indent=2))
    else:
        inventory = certify_inventory(paths, args.train_dir, esm_dir, load_workers=args.load_workers,
                                      reference_route_json=args.reference_route_json)
        print(json.dumps({key: inventory[key] for key in ("certified", "n_examples_loaded", "n_empty_first_shell_graphs")}, indent=2))


if __name__ == "__main__":
    main()
