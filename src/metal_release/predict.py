"""Label-free explicit-site inference with frozen checkpoint preprocessing.

The historical model loader changes a process-global label vocabulary. Calls are
serialized and restore that vocabulary; each member is collapsed using its own
recorded target before any other member is loaded.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import sys
import tempfile
import threading
import time
from typing import Any

import numpy as np

from .manifest import (
    ReleaseError, aggregate_probabilities, load_manifest, prediction_member_ids,
    native_to_common, verify_artifact,
)

COMMON_FOUR = ("Mn", "Cu", "Zn", "Class VIII")
_MODEL_LOCK = threading.RLock()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _center_from_atom(structure, locator: str) -> tuple[list[float], dict[str, Any]]:
    fields = locator.split(":")
    if len(fields) not in (4, 5):
        raise ReleaseError("Atom locator must be CHAIN:RESSEQ:ICODE:ATOM[:ALTLOC]; use _ for blanks")
    chain, resseq, icode, name = fields[:4]
    chain, icode = (" " if chain == "_" else chain), ("" if icode == "_" else icode)
    try:
        resseq = int(resseq)
    except ValueError as exc:
        raise ReleaseError("Atom locator RESSEQ must be an integer") from exc
    altloc = None if len(fields) == 4 else ("" if fields[4] == "_" else fields[4])
    matches = []
    for current_chain in next(structure.get_models()):
        if current_chain.id != chain:
            continue
        for residue in current_chain.get_unpacked_list():
            if int(residue.id[1]) != resseq or str(residue.id[2]).strip() != icode:
                continue
            for current in residue.get_unpacked_list():
                if current.name.strip() == name and (altloc is None or current.get_altloc().strip() == altloc):
                    matches.append(current)
    if len(matches) != 1:
        raise ReleaseError(f"Atom locator {locator!r} resolves to {len(matches)} atoms; specify an unambiguous altloc")
    selected = matches[0]
    return selected.coord.tolist(), {"atom": locator, "altloc": selected.get_altloc().strip()}


def _site_pocket(structure, structure_path: Path, site: dict[str, Any]):
    import torch
    from data_structures import DEFAULT_POCKET_RADIUS, GENERIC_METAL_ELEMENT, PocketRecord
    from graph.structure_parsing import (
        MetalAtomRecord, collect_structure_residues_and_metals, find_pocket_residues_near_metal_cluster,
    )

    if (site.get("center") is None) == (site.get("atom") is None):
        raise ReleaseError("Each site needs exactly one of center or atom")
    locator_report = {}
    center = site.get("center")
    if center is None:
        center, locator_report = _center_from_atom(structure, str(site["atom"]))
    try:
        center = np.asarray(center, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ReleaseError("Site center must contain three finite coordinates in Angstrom") from exc
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ReleaseError("Site center must contain three finite coordinates in Angstrom")
    residues, _ = collect_structure_residues_and_metals(structure, model_index=0)
    coord = torch.tensor(center, dtype=torch.float32)
    marker = MetalAtomRecord(coord=coord, symbol=GENERIC_METAL_ELEMENT, site_id=("", 0, ""))
    nearby = find_pocket_residues_near_metal_cluster(residues, [marker], DEFAULT_POCKET_RADIUS)
    if not nearby:
        raise ReleaseError(f"Site {site['id']!r} has no usable protein residues within {DEFAULT_POCKET_RADIUS} Angstrom")
    pocket = PocketRecord(
        structure_id=structure_path.stem, pocket_id=str(site["id"]), metal_element=GENERIC_METAL_ELEMENT,
        metal_coords=[coord], residues=nearby,
        metadata={"source_path": str(structure_path), "model_index": 0},
    )
    flags = ["known_site_classifier", "scores_are_uncalibrated", "base_site_summary_is_constant_in_ion_mode"]
    if site.get("center") is not None:
        flags.append("explicit_center_application_not_independently_validated")
    if len(list(structure.get_models())) > 1:
        flags.append("first_coordinate_model_only")
    chains = sorted({residue.chain_id for residue in nearby})
    if len(chains) > 1:
        flags.append("multichain_context")
    first_model = next(structure.get_models())
    dropped = sum(
        1 for residue in first_model.get_residues()
        if residue.id[0].strip() and len(residue.resname.strip()) == 3 and residue.resname.strip() != "HOH"
    )
    if dropped:
        flags.append("nonstandard_and_other_hetero_residues_excluded")
    return pocket, {
        "id": str(site["id"]), "center_angstrom": center.tolist(), "coordinate_model_index": 0,
        "site_source": "atom_locator" if site.get("atom") is not None else "explicit_center",
        **locator_report, "pocket_radius_angstrom": DEFAULT_POCKET_RADIUS,
        "protein_residue_count": len(nearby), "context_chains": chains, "quality_flags": flags,
    }


def build_site_pocket(structure_path: Path, *, center=None, atom: str | None = None):
    """Build one unlabeled ion-mode pocket; never infer targets from the file."""
    from graph.structure_parsing import parse_structure_file

    structure_path = Path(structure_path).resolve()
    if structure_path.suffix.lower() not in {".pdb", ".cif", ".mmcif"}:
        raise ReleaseError("Structure must be a PDB or mmCIF file")
    structure = parse_structure_file(str(structure_path))
    pocket, report = _site_pocket(structure, structure_path, {"id": "site_1", "center": center, "atom": atom})
    return structure, pocket, report


def _validated_member(manifest: dict, member_id: str) -> tuple[dict, dict, Path]:
    from data_structures import DEFAULT_POCKET_RADIUS, EXTERNAL_FEATURE_NAMES
    from label_schemes import normalize_metal_label_scheme_name

    member = manifest["members"][member_id]
    artifacts = member.get("artifacts", member)
    paths = {}
    for name in ("checkpoint", "run_config", "selected_checkpoint", "validation_predictions"):
        if artifacts.get(name) is not None:
            paths[name] = verify_artifact(artifacts[name])
    if not {"checkpoint", "run_config"} <= paths.keys():
        raise ReleaseError(f"{member_id}: checkpoint and run_config artifacts are required")
    payload = json.loads(paths["run_config"].read_text())
    config = payload.get("config", payload)
    if config.get("task") != "metal" or config.get("metal_example_unit") != "ion":
        raise ReleaseError(f"{member_id}: only saved metal-only ion-mode checkpoints are supported")
    if config.get("checkpoint_rule") != "terminal":
        raise ReleaseError(f"{member_id}: release requires the recorded terminal checkpoint rule")
    radius = config.get("pocket_radius", config.get("pocket_radius_angstrom", DEFAULT_POCKET_RADIUS))
    if radius != DEFAULT_POCKET_RADIUS or manifest.get("pocket_radius_angstrom", DEFAULT_POCKET_RADIUS) != DEFAULT_POCKET_RADIUS:
        raise ReleaseError(f"{member_id}: this adapter requires the historical {DEFAULT_POCKET_RADIUS} Angstrom pocket radius")
    if normalize_metal_label_scheme_name(config["metal_label_scheme"]) != normalize_metal_label_scheme_name(member["target"]):
        raise ReleaseError(f"{member_id}: target differs from saved run configuration")
    for field, config_field in (("fold", "fold_index"), ("seed", "seed")):
        if member.get(field) is not None and config.get(config_field) != member[field]:
            raise ReleaseError(f"{member_id}: {field} differs from saved run configuration")
    if config.get("use_ring_edges") or config.get("require_ring_edges"):
        raise ReleaseError(f"{member_id}: RING inputs require a separately implemented release adapter")
    if config.get("require_external_features") or not set(EXTERNAL_FEATURE_NAMES) <= set(config.get("omit_node_features", ())):
        raise ReleaseError(f"{member_id}: required external feature inputs are not supported by this release adapter")
    if "selected_checkpoint" in paths:
        selected = json.loads(paths["selected_checkpoint"].read_text())
        recorded = selected.get("selected_checkpoint_sha256", selected.get("checkpoint_sha256"))
        if recorded != artifacts["checkpoint"]["sha256"]:
            raise ReleaseError(f"{member_id}: selected receipt identifies a different checkpoint")
        if selected.get("checkpoint_rule", "terminal") != config["checkpoint_rule"]:
            raise ReleaseError(f"{member_id}: selected receipt checkpoint rule differs from saved configuration")
    return config, payload, paths["checkpoint"]


def _embedding_profile(device: str, precision: str = "bfloat16") -> dict:
    import torch

    try:
        esm_version = importlib.metadata.version("esm")
    except importlib.metadata.PackageNotFoundError:
        esm_version = None
    return {"device_type": torch.device(device).type, "model_dtype": precision,
            "torch_version": str(torch.__version__), "esm_version": esm_version,
            "generator_sha256": _sha256(Path(__file__)), "layer": "final_normalized"}


def _sequence_cache_path(cache: Path, model_name: str, sequence: str, profile: dict) -> Path:
    profile_sha = hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()
    return cache / model_name / profile_sha / f"{hashlib.sha256(sequence.encode()).hexdigest()}.pt"


def _validate_embedding(payload: Any, *, sequence: str, model_name: str, dim: int, path: Path,
                        residue_ids=None, profile=None, precision: str = "bfloat16"):
    import torch
    from training.esm_feature_loading import deserialize_residue_ids

    if not isinstance(payload, dict):
        raise ReleaseError(f"{path}: embedding cache needs model and sequence metadata")
    metadata = payload.get("metadata", {})
    expected_sha = hashlib.sha256(sequence.encode()).hexdigest()
    if metadata.get("esm_model_name") != model_name or metadata.get("source_sequence_sha256") != expected_sha:
        raise ReleaseError(f"{path}: embedding model or sequence identity mismatch")
    if metadata.get("model_dtype") != precision:
        raise ReleaseError(f"{path}: embedding backbone precision differs from the frozen {precision} contract")
    if profile is not None and metadata.get("generation_profile") != profile:
        raise ReleaseError(f"{path}: embedding generation source/runtime profile differs")
    embedding = payload.get("embeddings")
    if not isinstance(embedding, torch.Tensor) or tuple(embedding.shape) != (len(sequence), dim):
        raise ReleaseError(f"{path}: embedding dimensions do not match the full coordinate-chain sequence")
    if not embedding.is_floating_point() or not bool(torch.isfinite(embedding).all()):
        raise ReleaseError(f"{path}: embeddings must be finite floating point values")
    if residue_ids is not None and deserialize_residue_ids(payload.get("residue_ids", [])) != residue_ids:
        raise ReleaseError(f"{path}: embedding residue identities/order differ from the structure")
    return embedding.detach().cpu().float()


def _save_sequence_cache(path: Path, embedding, *, sequence: str, model_name: str, profile: dict | None = None) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    profile = profile or _embedding_profile("cpu")
    embedding = embedding.detach().cpu().contiguous()
    payload = {"embeddings": embedding, "metadata": {
        "esm_model_name": model_name, "source_sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
        "embedding_dim": int(embedding.size(1)), "layer": "final_normalized",
        "model_dtype": profile["model_dtype"], "tensor_storage_dtype": str(embedding.dtype).removeprefix("torch."),
        "emitted_dtype": str(embedding.dtype).removeprefix("torch."), "generation_profile": profile,
    }}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _embedding_lookup(structure, structure_path: Path, pockets: list, *, cache: Path,
                      model_name: str, dim: int, generate: bool, device: str,
                      precision: str = "bfloat16") -> tuple[dict, dict]:
    import torch
    from embed_helpers.esmc import (
        _esmc_sdk, clean_embedding_length, extract_chain_sequences,
    )
    from training.esm_feature_loading import normalize_chain_id, residue_keys_for_structure_chain

    sequences = extract_chain_sequences(structure)
    needed = sorted({normalize_chain_id(residue.chain_id) for pocket in pockets for residue in pocket.residues})
    lookup, files, reused, generated = {}, [], 0, 0
    sequence_values = {}
    model = None
    profile = _embedding_profile(device, precision)
    for chain in needed:
        if chain not in sequences:
            raise ReleaseError(f"Missing coordinate-chain sequence for {chain!r}")
        sequence = sequences[chain]
        sequence_sha = hashlib.sha256(sequence.encode()).hexdigest()
        residue_ids, _ = residue_keys_for_structure_chain(structure, chain)
        path = _sequence_cache_path(cache, model_name, sequence, profile)
        legacy = cache / f"{structure_path.stem}_chain_{chain}_esmc.pt"
        if sequence_sha in sequence_values:
            embedding = sequence_values[sequence_sha]
            reused += 1
        elif path.is_file() or legacy.is_file():
            source = path if path.is_file() else legacy
            payload = torch.load(source, map_location="cpu", weights_only=True)
            embedding = _validate_embedding(payload, sequence=sequence, model_name=model_name, dim=dim,
                                            path=source, residue_ids=residue_ids if source == legacy else None,
                                            profile=profile if source == path else None, precision=precision)
            metadata = payload["metadata"]
            files.append({"path": str(source), "sha256": _sha256(source), "model_dtype": metadata["model_dtype"],
                          "tensor_storage_dtype": str(payload["embeddings"].dtype).removeprefix("torch."),
                          "generation_profile": metadata.get("generation_profile"),
                          "provenance": "release_sequence_cache" if source == path else "historical_chain_cache"})
            reused += 1
        else:
            if not generate:
                raise ReleaseError(f"Missing verified ESM embeddings for chain {chain!r}; supply cache or explicitly enable generation")
            if model is None:
                ESMC, _, _ = _esmc_sdk()
                # Pass the device during construction: the upstream omitted-device
                # default can allocate CUDA even when a caller later requests CPU.
                model = ESMC.from_pretrained(model_name, device=torch.device(device))
                model = model.to(dtype=getattr(torch, precision)).eval()
                if str(next(model.parameters()).dtype).removeprefix("torch.") != precision:
                    raise ReleaseError("ESMC backbone does not match the frozen precision")
            _, ESMProtein, LogitsConfig = _esmc_sdk()
            with torch.no_grad():
                output = model.logits(model.encode(ESMProtein(sequence=sequence)),
                                      LogitsConfig(sequence=False, return_embeddings=True))
                embedding = clean_embedding_length(output.embeddings, len(sequence)).detach().cpu()
            _validate_embedding({"embeddings": embedding, "metadata": {
                "esm_model_name": model_name, "source_sequence_sha256": sequence_sha,
                "model_dtype": precision,
            }}, sequence=sequence, model_name=model_name, dim=dim, path=path)
            _save_sequence_cache(path, embedding, sequence=sequence, model_name=model_name, profile=profile)
            files.append({"path": str(path), "sha256": _sha256(path), "model_dtype": precision,
                          "tensor_storage_dtype": str(embedding.dtype).removeprefix("torch."),
                          "generation_profile": profile, "provenance": "release_sequence_cache"})
            generated += 1
        sequence_values[sequence_sha] = embedding
        for key, value in zip(residue_ids, embedding):
            lookup[key] = value.clone()
            if key[0] == "_":
                lookup[(" ", key[1], key[2])] = lookup[key]
    return lookup, {"model_name": model_name, "dimension": dim, "model_dtype": precision, "reused_chains": reused,
                    "generated_sequences": generated, "artifacts": files}


def _runtime_identity() -> dict:
    import torch
    from training.reproducibility import installed_package_versions

    source_root = Path(__file__).resolve().parents[1]
    source_files = [source_root / name for name in (
        "model.py", "featurization.py", "data_structures.py", "label_schemes.py", "export_validation_predictions.py",
        "training/run.py", "training/graph_dataset.py", "training/campaign_runtime.py", "embed_helpers/esmc.py",
        "training/esm_feature_loading.py", "training/raw_graph_cache.py",
        "metal_release/predict.py", "metal_release/manifest.py", "metal_release/catalog.py",
    )]
    source_files.extend((source_root / "graph").glob("*.py"))
    source_files.extend((source_root / "model_variants").glob("*.py"))
    return {
        "python": platform.python_version(), "executable": sys.executable, "platform": platform.platform(),
        "packages": installed_package_versions(),
        "torch_num_threads": torch.get_num_threads(), "torch_num_interop_threads": torch.get_num_interop_threads(),
        "source_sha256": {str(path.relative_to(source_root)): _sha256(path) for path in sorted(source_files)},
        "compatibility_status": "unverified_until_development_prediction_parity_passes_for_this_runtime",
    }


def predict_sites(manifest_path: Path, structure_path: Path, sites: list[dict], *,
                  candidate: str | None = None, embedding_cache: Path | None = None,
                  generate_embeddings: bool = False, device: str = "cpu", batch_size: int = 32,
                  include_member_predictions: bool = False) -> dict:
    """Predict explicit centers in one structure, sharing embeddings and model loads.

    CPU is the default. Generation is opt-in and uses the supplied device; this
    function never allocates cloud resources. Timings include artifact verification.
    """
    import torch
    import label_schemes
    from graph.feature_utils import attach_esm_embeddings
    from graph.structure_parsing import parse_structure_file
    from torch_geometric.loader import DataLoader
    from training.campaign_runtime import load_campaign_prediction_components
    from training.graph_dataset import PocketGraphDataset, build_graph_data_list
    from training.run import to_jsonable

    started, cpu_started = time.perf_counter(), time.process_time()
    if batch_size < 1 or not sites:
        raise ReleaseError("At least one site and a positive batch size are required")
    if any(not isinstance(site, dict) or not isinstance(site.get("id"), str) or not site["id"] for site in sites):
        raise ReleaseError("Each site requires a nonempty string id")
    if len({site["id"] for site in sites}) != len(sites):
        raise ReleaseError("Site ids must be unique")
    resolved_device = torch.device(device)
    if resolved_device.type not in {"cpu", "cuda"}:
        raise ReleaseError("Only explicit cpu or cuda devices are supported")
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise ReleaseError("Requested CUDA device is unavailable")
    if resolved_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved_device)
    manifest_path, structure_path = Path(manifest_path).resolve(), Path(structure_path).resolve()
    manifest = load_manifest(manifest_path)
    candidate = candidate or manifest["reference"]
    member_ids = prediction_member_ids(manifest, candidate)
    prepared = {member_id: _validated_member(manifest, member_id) for member_id in member_ids}
    if structure_path.suffix.lower() not in {".pdb", ".cif", ".mmcif"}:
        raise ReleaseError("Structure must be a PDB or mmCIF file")
    verification_seconds = time.perf_counter() - started
    parse_started = time.perf_counter()
    structure = parse_structure_file(str(structure_path))
    pocket_reports = [_site_pocket(structure, structure_path, site) for site in sites]
    pockets, reports = [item[0] for item in pocket_reports], [item[1] for item in pocket_reports]
    parse_seconds = time.perf_counter() - parse_started
    dimensions = {int(config["esm_dim"]) for config, _, _ in prepared.values()}
    if len(dimensions) != 1:
        raise ReleaseError("Ensemble members require incompatible embedding dimensions")
    dim = dimensions.pop()
    needs_esm = any(config.get("require_esm_embeddings") or config.get("model_architecture") != "only_gvp"
                    for config, _, _ in prepared.values())
    embedding_started = time.perf_counter()
    embedding_report = {"required": needs_esm, "generated_sequences": 0, "reused_chains": 0}
    if needs_esm:
        model_name = manifest.get("embedding_model") or manifest.get("embedding", {}).get("model_name")
        if not isinstance(model_name, str) or not model_name or Path(model_name).name != model_name:
            raise ReleaseError("Manifest must identify the frozen embedding_model")
        precision = manifest.get("embedding_precision", "bfloat16")
        if precision != "bfloat16":
            raise ReleaseError("This release requires the historical bfloat16 ESMC backbone precision")
        cache = Path(embedding_cache).expanduser() if embedding_cache is not None else (
            Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "deepmzyme" / "embeddings")
        lookup, embedding_report = _embedding_lookup(structure, structure_path, pockets, cache=cache,
            model_name=model_name, dim=dim, generate=generate_embeddings, device=str(resolved_device), precision=precision)
        for pocket in pockets:
            attach_esm_embeddings(pocket, lookup, dim, zero_if_missing=False)
    embedding_seconds = time.perf_counter() - embedding_started
    probabilities, member_timings, raw_graphs, member_predictions = {}, {}, {}, {}
    with _MODEL_LOCK:
        old_scheme = label_schemes.ACTIVE_METAL_LABEL_SCHEME
        try:
            for member_id in member_ids:
                config, payload, checkpoint_path = prepared[member_id]
                tick = time.perf_counter()
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                if to_jsonable(checkpoint.get("config")) != config:
                    raise ReleaseError(f"{member_id}: checkpoint and saved run configurations differ")
                model, normalization, options = load_campaign_prediction_components(checkpoint, device=str(resolved_device))
                model.eval()
                del checkpoint
                if resolved_device.type == "cuda":
                    torch.cuda.synchronize(resolved_device)
                load_seconds = time.perf_counter() - tick
                tick = time.perf_counter()
                graph_key = json.dumps(options, sort_keys=True)
                if graph_key not in raw_graphs:
                    raw_graphs[graph_key] = build_graph_data_list(pockets, **options)
                dataset = PocketGraphDataset(pockets, precomputed_data=raw_graphs[graph_key],
                                             normalization_stats=normalization, **options)
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                                    generator=torch.Generator().manual_seed(int(config["seed"]) + 2))
                graph_seconds = time.perf_counter() - tick
                tick = time.perf_counter()
                native = []
                with torch.no_grad():
                    for batch in loader:
                        logits = model(batch.to(resolved_device))["logits_metal"].float()
                        if not bool(torch.isfinite(logits).all()):
                            raise ReleaseError(f"{member_id}: nonfinite model logits")
                        native.append(logits.softmax(-1).cpu())
                native_values = torch.cat(native)
                native_to_common(native_values.numpy(), manifest["members"][member_id]["target"])
                # Match the historical evaluator: torch float32 collapse, then
                # float64 family/block arithmetic. Do not average native logits.
                common = torch.cat((native_values[:, :3], native_values[:, 3:].sum(-1, keepdim=True)), dim=-1)
                probabilities[member_id] = common.numpy().astype(np.float64)
                if include_member_predictions:
                    member = manifest["members"][member_id]
                    artifacts = member.get("artifacts", member)
                    member_predictions[member_id] = {
                        "native_probabilities": native_values.tolist(), "common4_probabilities": common.tolist(),
                        "native_predictions": native_values.argmax(-1).tolist(), "common4_predictions": common.argmax(-1).tolist(),
                        "checkpoint_sha256": artifacts["checkpoint"]["sha256"],
                    }
                member_timings[member_id] = {"checkpoint_load_seconds": load_seconds,
                    "graph_prepare_seconds": graph_seconds, "forward_and_normalization_seconds": time.perf_counter() - tick}
                del model, loader, dataset
        finally:
            label_schemes.configure_active_metal_label_scheme(old_scheme)
    combined = aggregate_probabilities(manifest, candidate, probabilities)
    for index, report in enumerate(reports):
        report.update({"predicted_class": COMMON_FOUR[int(combined[index].argmax())],
                       "class_scores": dict(zip(COMMON_FOUR, combined[index].tolist()))})
    peak_ram = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        peak_ram *= 1024
    member_provenance = {}
    for member_id, (config, payload, _) in prepared.items():
        member = manifest["members"][member_id]
        artifacts = member.get("artifacts", member)
        identity = config.get("campaign_run_identity")
        if isinstance(identity, str):
            try:
                identity = json.loads(identity)
            except json.JSONDecodeError:
                identity = None
        member_provenance[member_id] = {
            "checkpoint_sha256": artifacts["checkpoint"]["sha256"],
            "run_config_sha256": artifacts["run_config"]["sha256"],
            "target": member["target"], "fold": member.get("fold"), "seed": member["seed"],
            "saved_source_tree_sha256": identity.get("source_tree_sha256") if isinstance(identity, dict) else None,
            "saved_packages": payload.get("runtime_environment", {}).get("packages"),
        }
    result = {
        "schema": "deepmzyme-metal-predictions-v1", "model_version": manifest.get("version", manifest["schema"]),
        "candidate": candidate, "reference": manifest["reference"], "manifest_sha256": _sha256(manifest_path),
        "structure_sha256": _sha256(structure_path), "prediction_unit": "ion_center", "sites": reports,
        "class_order": list(COMMON_FOUR), "calibration": "none",
        "score_meaning": "normalized scores among supported metal classes, conditional on a supplied site",
        "member_count": len(member_ids), "member_provenance": member_provenance,
        "embeddings": embedding_report, "runtime": _runtime_identity(),
        "device": str(resolved_device), "timings": {
            "artifact_verification_seconds": verification_seconds, "structure_and_site_seconds": parse_seconds,
            "embeddings_seconds": embedding_seconds, "members": member_timings,
            "total_seconds": time.perf_counter() - started, "process_cpu_seconds": time.process_time() - cpu_started,
        },
        "memory": {"peak_process_rss_bytes": peak_ram, "rss_scope": "process_lifetime_high_water_mark",
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(resolved_device)) if resolved_device.type == "cuda" else None,
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(resolved_device)) if resolved_device.type == "cuda" else None},
    }
    if include_member_predictions:
        result["member_predictions"] = member_predictions
    return result


def predict_structure(manifest_path: Path, structure_path: Path, *, candidate: str | None = None,
                      center=None, atom: str | None = None, embedding_cache: Path | None = None,
                      generate_embeddings: bool = False, device: str = "cpu", batch_size: int = 32,
                      include_member_predictions: bool = False) -> dict:
    """Convenience wrapper for one site; see :func:`predict_sites` for batching."""
    return predict_sites(manifest_path, structure_path, [{"id": "site_1", "center": center, "atom": atom}],
        candidate=candidate, embedding_cache=embedding_cache, generate_embeddings=generate_embeddings,
        device=device, batch_size=batch_size, include_member_predictions=include_member_predictions)
