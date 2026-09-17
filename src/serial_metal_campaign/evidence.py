"""Training-only cohort preparation and verification of saved evidence."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path
import math
import statistics

import run_metal_architecture_pilot as base
from serial_metal_campaign import profile


def prepare(output):
    """CPU preparation before allocating Colab: hashes, eligibility, all folds."""
    from audit_metal_ring_inputs import collect_training_cache
    from label_schemes import configure_active_metal_label_scheme
    from training.data import load_training_pockets_with_report_from_dir
    from training.splits import split_pockets, split_pockets_k_fold, retained_split_identity

    output = Path(output)
    manifest = profile.verify_manifest(output)
    base.validate_feature_overlay(manifest["data_root"], manifest["external_features_root_dir"],
                                  manifest["feature_overlay_manifest"]["path"])
    row = next(r for r in manifest["runs"] if r["arm"] == "gvp_six")
    config = base.parse_config(row["command"])
    configure_active_metal_label_scheme(config.metal_label_scheme)
    cache_config = replace(config, ring_features_dir=Path(manifest["data_root"]) / "RING_features")
    cache = collect_training_cache(cache_config)
    # Loading pockets without ESM avoids retaining all tensors merely to freeze
    # grouping. Cache coverage above and actual ESM smokes gate later execution.
    loaded = load_training_pockets_with_report_from_dir(
        structure_dir=config.structure_dir, summary_csv=config.summary_csv,
        required_targets=("metal",), metal_eligibility_scheme="six_class",
        esm_dim=config.esm_dim, esm_embeddings_dir=None, require_esm_embeddings=False,
        ring_features_dir=config.ring_features_dir,
        external_features_root_dir=config.external_features_root_dir,
        external_feature_source=config.external_feature_source, require_external_features=True,
        unsupported_metal_policy="error", invalid_structure_policy="error", ec_label_depth=1)
    pockets = loaded.pockets
    base.require(pockets, "No eligible training pockets")
    references = {}
    discovery = split_pockets(pockets, .15, "pdbid", 42, task="metal", stratify_by="metal_site")
    splits = {"discovery": discovery}
    splits.update({f"fold_{fold}": split_pockets_k_fold(pockets, n_folds=5, fold_index=fold,
                  split_by="pdbid", seed=42, task="metal", stratify_by="metal_site") for fold in range(5)})
    for name, split in splits.items():
        summary = {"retained_split_identity": {
            "train": retained_split_identity(split.train_pockets, "pdbid"),
            "validation": retained_split_identity(split.val_pockets, "pdbid")}}
        for part in ("train", "validation"):
            support = Counter(r["y_metal"] for r in summary["retained_split_identity"][part]["examples"])
            base.require(set(support) == set(range(6)) and min(support.values()) > 0,
                         f"All six native classes must occur in {name}/{part}; do not repair folds after results")
        base.cohort_identity(summary)
        references[name] = summary
    all_examples = retained_split_identity(pockets, "pdbid")["examples"]
    plan = dict(split_seed=42, n_folds=5, seeds=list(profile.SEEDS),
                cohort_sha256=base.fingerprint([{k: r[k] for k in ("structure_id", "pocket_id", "group")}
                                                for r in all_examples]),
                references=references, native_class_support=dict(Counter(p.y_metal for p in pockets)),
                held_out_evaluation=False)
    profile.freeze(output / "fold_plan.json", plan)
    # Portable content identities omit host timestamps and absolute paths.
    content = cache_identity(cache, manifest)
    profile.freeze(output / "input_identity.json", dict(files=content, sha256=base.fingerprint(content)))
    base.save(output / "training_cache_audit.json", cache)
    profile.freeze(output / "preparation.json", dict(status="passed", manifest_sha256=base.digest(output / "campaign_manifest.json"),
                   folds_sha256=base.digest(output / "fold_plan.json"), inputs_sha256=base.digest(output / "input_identity.json"),
                   cohort_sha256=plan["cohort_sha256"], held_out_evaluation=False))
    return base.read(output / "preparation.json")


def cache_identity(cache, manifest):
    roots = {"external": Path(manifest["external_features_root_dir"]),
             "esm": Path(manifest["data_root"]) / "esm_embeddings",
             "esm_metadata": Path(manifest["data_root"]) / "esm_embeddings",
             "ring": Path(manifest["data_root"]) / "RING_features"}
    return sorted([dict(kind=row["kind"], relative_path=str(Path(row["path"]).relative_to(roots[row["kind"]])),
                        sha256=row["sha256"], bytes=row["bytes"]) for row in cache["files"]],
                  key=lambda row: (row["kind"], row["relative_path"]))


def verify_preparation(output, *, hash_contents=False):
    output = Path(output)
    manifest = profile.verify_manifest(output)
    prepared = base.read(output / "preparation.json", {})
    base.require(prepared.get("status") == "passed" and
                 prepared.get("manifest_sha256") == base.digest(output / "campaign_manifest.json"),
                 "Run CPU prepare before GPU readiness or execution")
    for file, key in (("fold_plan.json", "folds_sha256"), ("input_identity.json", "inputs_sha256")):
        base.require(base.digest(output / file) == prepared[key], f"Frozen preparation changed: {file}")
    cache = base.read(output / "training_cache_audit.json")
    base.require(cache_identity(cache, manifest) == base.read(output / "input_identity.json")["files"],
                 "Feature inventory differs from frozen content identity")
    for row in cache["files"]:
        path = Path(row["path"])
        info = path.stat()
        if hash_contents or info.st_mtime_ns != row["mtime_ns"] or info.st_size != row["bytes"]:
            base.require(info.st_size == row["bytes"] and base.digest(path) == row["sha256"],
                         f"Feature contents changed: {path}")
    base.validate_feature_overlay(manifest["data_root"], manifest["external_features_root_dir"],
                                  manifest["feature_overlay_manifest"]["path"])
    return prepared


def hardware_probe():
    import torch
    base.require(torch.cuda.is_available(), "CUDA is required; plan/prepare/report work on CPU")
    major, minor = torch.cuda.get_device_capability(0)
    architectures = torch.cuda.get_arch_list()
    base.require(f"sm_{major}{minor}" in architectures, "Stock PyTorch does not support the assigned GPU")
    tensor = torch.ones(8, device="cuda", requires_grad=True)
    tensor.square().sum().backward()
    torch.cuda.synchronize()
    return dict(gpu=torch.cuda.get_device_name(0), memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                torch=str(torch.__version__), cuda=torch.version.cuda, capability=[major, minor],
                compiled_architectures=architectures)


def _jsonable(value):
    """Compare checkpoint tensors with their persisted JSON representation."""
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return str(value) if isinstance(value, Path) else value


def checkpoint_parameter_count(checkpoint):
    """Rebuild the configured model and strictly validate every saved tensor."""
    import torch
    from model_variants.factory import build_pocket_classifier
    config, state = checkpoint["config"], checkpoint["model_state_dict"]
    base.require(all(not tensor.is_floating_point() or bool(torch.isfinite(tensor).all())
                     for tensor in state.values()), "Checkpoint contains nonfinite model tensors")
    names = """model_architecture esm_dim hidden_s hidden_v edge_hidden esm_fusion_dim
        head_mlp_layers head_mlp_dropout esm_graph_encoder_dropout node_rbf_sigma edge_rbf_sigma
        node_rbf_use_raw_distances classifier_pool_distance_cutoff structural_readout_scope
        site_geometry_features normalize_message_aggregation use_esm_branch fusion_mode
        cross_attention_layers cross_attention_heads cross_attention_dropout cross_attention_neighborhood
        cross_attention_bidirectional use_early_esm early_esm_dim early_esm_dropout early_esm_raw
        early_esm_scope joint_loss_weighting metal_loss_weight ec_loss_weight metal_loss_function
        metal_focal_gamma metal_label_smoothing metal_collapsed_loss_weight ec_contrastive_weight
        ec_contrastive_temperature""".split()
    kwargs = {name: config[name] for name in names}
    kwargs.update(n_layers=config["gvp_layers"], n_metal=len(checkpoint["metal_labels"]),
                  n_ec=max(1, len(checkpoint.get("ec_labels", {}))),
                  use_node_type_embedding=config["metal_node_mode"] != "none",
                  use_site_angle_features=config["metal_node_mode"] != "none",
                  predict_metal=config["task"] in ("metal", "joint"),
                  predict_ec=config["task"] in ("ec", "joint"))
    for name in ("metal_class_weights", "metal_collapsed4_class_weights", "ec_class_weights"):
        weights = state.get(name)
        expected = {"metal_class_weights": kwargs["n_metal"], "metal_collapsed4_class_weights": 4,
                    "ec_class_weights": kwargs["n_ec"]}[name]
        base.require(weights is None or weights.numel() == 0 or tuple(weights.shape) == (expected,),
                     f"Checkpoint class-weight shape differs: {name}")
        kwargs[name] = weights if weights is not None and weights.numel() else None
    # Verification must not advance the model RNG used by subsequent training.
    with torch.random.fork_rng(devices=[]):
        model = build_pocket_classifier(**kwargs)
    model.load_state_dict(state, strict=True)
    return sum(parameter.numel() for parameter in model.parameters())


def epoch_cap_diagnostics(history, selected_epoch, epochs):
    """Advisory late-trend evidence; this never changes checkpoint selection."""
    scores = [float(row[base.METRIC]) for row in history]
    late = scores[-5:]
    prior = scores[-10:-5]
    eligible = epochs == 50 and len(scores) == 50 and all(math.isfinite(v) for v in scores)
    gain = statistics.mean(late) - statistics.mean(prior) if eligible else None
    return dict(best_checkpoint_at_epoch_cap=epochs == 50 and selected_epoch == 50,
                selected_in_final_five=bool(eligible and selected_epoch >= 46),
                late_validation_mean_gain=gain,
                potentially_training_budget_limited=bool(eligible and (selected_epoch >= 46 or gain > .002)),
                epoch_cap_interpretation="advisory_only_no_extension_or_checkpoint_reselection")


def verify_result(output, run, directory, *, artifact_config=None):
    """Bind metric, checkpoint, config, cohort, and normalization before acceptance."""
    import torch
    output, directory = Path(output), Path(directory)
    reference_name = "discovery" if run.get("fold_index") is None else f"fold_{run['fold_index']}"
    reference = base.read(output / "fold_plan.json")["references"][reference_name]
    artifact_run = {**run, "config": artifact_config} if artifact_config is not None else run
    result = base.completed_result(artifact_run, directory, reference)
    metadata = base.read(directory / "run_metadata.json")
    config = base.read(directory / "run_config.json")
    checkpoint = torch.load(directory / "best_model_checkpoint.pt", map_location="cpu", weights_only=True)
    base.require(checkpoint.get("epoch") == result["selected_epoch"]
                 and checkpoint.get("selection_metric") == base.METRIC,
                 "Selected checkpoint identity differs from metadata")
    base.require(math.isclose(float(checkpoint.get("selection_metric_value", float("nan"))),
                              result["balanced_accuracy"], abs_tol=1e-10), "Checkpoint score differs")
    for key, expected in artifact_run["config"].items():
        if key not in ("runs_dir", "run_name"):
            base.require(_jsonable(checkpoint["config"].get(key)) == _jsonable(expected),
                         f"Checkpoint configuration mismatch: {key}")
    base.require(base.cohort_identity(checkpoint["dataset_summary"]) == base.cohort_identity(reference),
                 "Checkpoint has a different retained cohort")
    base.require(_jsonable(checkpoint["normalization_stats"]) == metadata["normalization_stats"],
                 "Checkpoint normalization differs from metadata")
    native_count = {"four_class": 4, "five_class": 5, "six_class": 6}[run["scheme"]]
    base.require(len(checkpoint["metal_labels"]) == native_count, "Wrong native output label count")
    native_names = {"four_class": ["Mn", "Cu", "Zn", "Class VIII"],
                    "five_class": ["Mn", "Cu", "Zn", "Fe", "Class VIII"],
                    "six_class": ["Mn", "Cu", "Zn", "Fe", "Co", "Ni"]}[run["scheme"]]
    base.require(_jsonable(checkpoint["metal_labels"]) == {str(i): name for i, name in enumerate(native_names)},
                 "Native checkpoint label ordering differs")
    selected = next(r for r in config["history"] if r["epoch"] == result["selected_epoch"])
    for name in ("val_metal_per_class_recall", "val_metal_collapsed4_per_class_recall"):
        recalls = selected.get(name, selected["val_metal_per_class_recall"])
        expected_count = 4 if "collapsed4" in name else native_count
        base.require(len(recalls) == expected_count and all(v is not None and math.isfinite(float(v))
                     and 0 <= float(v) <= 1 for v in recalls.values()), "Missing/nonfinite per-class recall")
    result.update({key: run.get(key) for key in ("stage", "arm", "parameters", "recipe_id", "fold_index", "epochs",
                                               "complexity_proxy", "config_sha256", "ring")})
    result.update(training_balanced_accuracy=selected.get("train_metal_balanced_acc"),
                  checkpoint_sha256=base.digest(directory / "best_model_checkpoint.pt"),
                  normalization_sha256=base.fingerprint(metadata["normalization_stats"]),
                  cohort_sha256=base.fingerprint(base.cohort_identity(reference)),
                  folds_sha256=base.digest(output / "fold_plan.json"),
                  inputs_sha256=base.digest(output / "input_identity.json"),
                  source_sha256=base.fingerprint(profile.verify_manifest(output)["source_files"]),
                  parameter_count=checkpoint_parameter_count(checkpoint))
    result.update(epoch_cap_diagnostics(config["history"], result["selected_epoch"], run["epochs"]))
    return result


def ring_input_audit(output, run, directory=None):
    """Fresh geometry-fixed controls are checked against all nodes in this fold."""
    from audit_metal_ring_inputs import audit
    output = Path(output)
    directory = Path(directory) if directory is not None else output / "ring_audits" / f"fold_{run['fold_index']}"
    receipt = directory / "ring_input_audit.json"
    reference = base.read(output / "fold_plan.json")["references"][f"fold_{run['fold_index']}"]
    if receipt.is_file():
        result = base.read(receipt)
        base.require(result.get("status") == "passed" and result.get("expected_cohort_sha256") ==
                     base.fingerprint(base.cohort_identity(reference)), "Invalid RING input audit")
        return result
    config = base.parse_config(run["command"])
    off = replace(config, device="cpu", esm_embeddings_dir=None, require_esm_embeddings=False,
                  use_ring_edges=False, require_ring_edges=False)
    on = replace(off, use_ring_edges=True, require_ring_edges=True)
    return audit(off, on, reference, directory)


OPERATIONAL_CONFIG_FIELDS = frozenset({
    "runs_dir", "run_name", "structure_dir", "summary_csv", "esm_embeddings_dir",
    "ring_features_dir", "external_features_root_dir", "test_structure_dir", "test_summary_csv",
    "device", "num_workers", "pin_memory",
})
ARTIFACT_NAMES = ("run_config.json", "run_metadata.json", "dataset_summary.json", "epoch_metrics.csv",
                  "best_model_checkpoint.pt", "last_model_checkpoint.pt")


def scientific_sources(manifest):
    """Ignore only documentation and named experiment orchestration surfaces."""
    def scientific(name):
        path = Path(name)
        orchestration = {"run_ec_baselines.py", "run_metal_architecture_pilot.py",
                         "run_metal_coordination_geometry_pilot.py", "run_metal_ring_pilot.py",
                         "run_serial_metal_campaign.py", "run_metal_single_gpu_campaign.py",
                         "report_runs.py", "report_paired_metal_routes.py",
                         "train_serial_metal_profile.py",
                         "audit_sequence_remoteness.py", "export_validation_predictions.py",
                         "report_remote_homology.py",
                         "stage6_standalone.py", "verify_colab_notebook_smoke.py"}
        return (name.startswith("src/") and not name.startswith("src/serial_metal_campaign/")
                and not (path.parent == Path("src") and path.name in orchestration))
    return {name: digest for name, digest in manifest["source_files"].items() if scientific(name)}


def _inactive_shell_branch_proof():
    """Check the live branch, failing closed if shell-source semantics change."""
    import ast
    import inspect
    from graph.shell_roles import compute_shell_roles
    tree = ast.parse(inspect.getsource(compute_shell_roles))
    condition = ast.parse('use_ring_edges and shell_role_source == "edge_mode"', mode="eval").body
    branches = [node for node in ast.walk(tree) if isinstance(node, ast.IfExp)]
    uses = [node for node in ast.walk(tree) if isinstance(node, ast.Name)
            and node.id == "shell_role_source" and isinstance(node.ctx, ast.Load)]
    base.require(len(branches) == 1 and ast.dump(branches[0].test) == ast.dump(condition)
                 and len(uses) == 2, "RING-off shell-source branch proof failed")


def compatible_config(source, target):
    """Return the two explicitly guarded inactive mappings; reject other drift."""
    from model_variants.factory import _apply_fusion_defaults, normalize_model_architecture
    mappings = []
    for key in sorted(set(source) | set(target)):
        if source.get(key) == target.get(key) and (key in source) == (key in target):
            continue
        if key in OPERATIONAL_CONFIG_FIELDS:
            continue
        if key == "early_esm_dropout" and source.get(key) == .2 and target.get(key) == 0:
            for config in (source, target):
                architecture = normalize_model_architecture(config["model_architecture"])
                resolved = _apply_fusion_defaults(config)
                base.require(config.get("use_early_esm") is False and
                             (architecture in ("only_gvp", "only_esm") or not resolved["use_early_esm"]),
                             "Early dropout compatibility requires an inactive early branch")
            mappings.append(dict(field=key, source=.2, target=0, proof="factory_resolved_early_branch_inactive"))
        elif key == "shell_role_source" and {source.get(key), target.get(key)} == {"geometry", "edge_mode"}:
            base.require(source.get("use_ring_edges") is False and target.get("use_ring_edges") is False,
                         "Shell-source compatibility requires RING off")
            _inactive_shell_branch_proof()
            mappings.append(dict(field=key, source=source[key], target=target[key], proof="live_ring_off_branch_identical"))
        else:
            raise ValueError(f"Scientific configuration differs: {key}")
    return mappings


def _historical_provenance(source):
    for name in ("final_capture_receipt.json", "final_run_provenance.json"):
        for directory in (source / "finalization", source):
            path = directory / name
            if path.is_file():
                return path, base.read(path)
    raise ValueError("Historical full-artifact provenance is required for reuse")


def _target_run(output, run_id):
    paths = [output / "campaign_manifest.json", output / "queue.json", output / "confirmation_manifest.json"]
    matches = [run for path in paths for run in base.read(path, {}).get("runs", []) if run["id"] == run_id]
    base.require(matches and all(row == matches[0] for row in matches), "Target run is absent or changed in frozen campaign")
    return matches[0]


def _verify_reuse(output, target_run, source_campaign, source_run_id):
    output, source = Path(output).resolve(), Path(source_campaign).resolve()
    prepared = verify_preparation(output, hash_contents=True)
    manifest = profile.verify_manifest(output)
    base.require(_target_run(output, target_run["id"]) == target_run, "Reuse target differs from frozen run")
    source_manifest_path = source / "campaign_manifest.json"
    source_manifest = base.read(source_manifest_path)
    base.require(scientific_sources(source_manifest) == scientific_sources(manifest),
                 "Scientific source hashes differ; historical run is not reusable")
    base.require(source_manifest["dataset_files"] == manifest["dataset_files"] and
                 source_manifest["bundle_sha256"] == manifest["bundle_sha256"],
                 "Historical dataset identity differs")
    base.require(source_manifest.get("feature_overlay_manifest", {}).get("sha256") ==
                 manifest.get("feature_overlay_manifest", {}).get("sha256"),
                 "Historical feature overlay identity differs")
    source_runs = [run for run in source_manifest["runs"] if run["id"] == source_run_id]
    base.require(len(source_runs) == 1, "Historical run is absent or duplicated")
    source_run = source_runs[0]
    for key in ("family", "scheme", "seed", "epochs"):
        base.require(source_run[key] == target_run[key], f"Historical run differs in {key}")
    mappings = compatible_config(source_run["config"], target_run["config"])
    provenance_path, provenance = _historical_provenance(source)
    base.require(provenance.get("manifest_sha256") == base.digest(source_manifest_path),
                 "Historical provenance does not bind its campaign manifest")
    records = [row for row in provenance["runs"] if row["run_id"] == source_run_id]
    base.require(len(records) == 1, "Historical artifact record is absent or duplicated")
    record = records[0]
    directory = source / "runs" / Path(record["run_dir"]).name
    base.require(directory.resolve().is_relative_to(source), "Historical run directory escapes campaign")
    artifacts = {}
    for name in ARTIFACT_NAMES:
        path = directory / name
        base.require(path.is_file() and base.digest(path) == record.get("files", {}).get(name),
                     f"Historical artifact is missing or changed: {name}")
        artifacts[str(path)] = base.digest(path)
    cache_path = source / "training_cache_audit.json"
    cache = base.read(cache_path)
    base.require(cache and cache.get("status") == "passed" and
                 cache_identity(cache, source_manifest) == base.read(output / "input_identity.json")["files"],
                 "Historical feature inventory differs from current inputs")
    # This audit binds the historical cache and fitted normalization. Campaigns
    # without such a certificate receive no credit until equivalent proof exists.
    audit_path = source / "ring_input_audit.json"
    audit = base.read(audit_path, {})
    base.require(audit.get("status") == "passed" and audit.get("cache_audit_sha256") == base.digest(cache_path),
                 "Historical input audit does not bind its cache inventory")
    result = verify_result(output, target_run, directory, artifact_config=source_run["config"])
    base.require(record.get("normalization_stats_sha256") == result["normalization_sha256"],
                 "Historical provenance normalization differs from checkpoint")
    normalization_side = "on" if source_run["config"].get("use_ring_edges") else "off"
    base.require(audit.get("normalization", {}).get(normalization_side + "_sha256") == result["normalization_sha256"],
                 "Historical training-fitted normalization differs from input audit")
    base.require(audit.get("expected_cohort_sha256") == result["cohort_sha256"],
                 "Historical input audit has a different cohort")
    for path in (source_manifest_path, provenance_path, cache_path, audit_path):
        artifacts[str(path)] = base.digest(path)
    result.update(reused=True, reuse_certified=True, source_run_id=source_run_id)
    return dict(run_id=target_run["id"], result=result, source_campaign=str(source), source_run_id=source_run_id,
                run_dir=str(directory), artifacts=artifacts, compatibility_mappings=mappings,
                certified=True, target_run_sha256=base.fingerprint(target_run),
                manifest_sha256=base.digest(output / "campaign_manifest.json"),
                inputs_sha256=prepared["inputs_sha256"], folds_sha256=prepared["folds_sha256"],
                scientific_source_sha256=base.fingerprint(scientific_sources(manifest)),
                held_out_evaluation=False)


def import_reuse(output, target_run, source_campaign, source_run_id):
    """Verify read-only historical evidence, then write only a reuse receipt."""
    output = Path(output).resolve()
    receipt = _verify_reuse(output, target_run, source_campaign, source_run_id)
    path = output / "reuse" / f"{target_run['id']}.json"
    base.require(path.resolve().is_relative_to(output / "reuse"), "Unsafe reuse run identifier")
    path.parent.mkdir(parents=True, exist_ok=True)
    profile.freeze(path, receipt)
    return receipt


def verify_reuse_receipt(output, receipt):
    """Rehash historical artifacts and current inputs before granting reuse credit."""
    output = Path(output).resolve()
    receipt = base.read(receipt) if isinstance(receipt, (str, Path)) else receipt
    expected = _verify_reuse(output, _target_run(output, receipt["run_id"]),
                             receipt["source_campaign"], receipt["source_run_id"])
    base.require(receipt == expected, "Reuse receipt differs from freshly verified evidence")
    return expected
