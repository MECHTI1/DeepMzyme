"""Immutable scientific recipes for the approved single-GPU campaign.

Command generation uses the existing notebook adapter. Discovery, confirmation,
and operational budgets are separate; this profile never opens held-out data.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import math
import shlex
import sys

import run_metal_architecture_pilot as base

PROFILE = "metal_single_gpu_20h_v2"
LRS = (1e-5, 3e-5, 1e-4)
SEEDS = (42, 43)
FAMILY_ORDER = ("gvp", "esm", "late", "early", "hybrid")
FAMILIES = dict(zip(FAMILY_ORDER, (*base.CORE, base.EARLY, base.HYBRID)))
ARMS = {f"{family}_{target}": (FAMILIES[family], f"{target}_class")
        for family in FAMILY_ORDER
        for target in (("four", "six") if family in FAMILY_ORDER[:3] else ("four",))}
CAPACITIES = {
    "compact": dict(hidden_s=128, hidden_v=8, edge_hidden=64, gvp_layers=2,
                    head_mlp_layers=1, esm_fusion_dim=64, early_esm_dim=32),
    "reference": dict(hidden_s=128, hidden_v=16, edge_hidden=64, gvp_layers=4,
                      head_mlp_layers=2, esm_fusion_dim=128, early_esm_dim=32),
}
LARGE_CAPACITIES = {
    "large_256": dict(hidden_s=256, hidden_v=32, edge_hidden=128),
    "large_384": dict(hidden_s=384, hidden_v=48, edge_hidden=192),
    "large_512": dict(hidden_s=512, hidden_v=64, edge_hidden=256),
}
EXTRA_ARMS = {"late_five": (FAMILIES["late"], "five_class"),
              "gvp_ring_on": (FAMILIES["gvp"], "four_class"),
              "gvp_ring_off": (FAMILIES["gvp"], "four_class")}
FIXED = dict(weight_decay=1e-4, head_mlp_dropout=.2, esm_graph_encoder_dropout=.1,
             early_esm_dropout=0., metal_class_weight_mode="inverse_frequency",
             edge_radius=6., classifier_pool_distance_cutoff=0., shell_role_source="edge_mode")
POLICY = dict(total_seconds=72000, operations_seconds=14400, discovery_seconds=21600,
              confirmation_seconds=36000, session_seconds=14400, closeout_seconds=900,
              admission_factor=1.25, epochs=50, split_seed=42, val_fraction=.15,
              n_folds=5, seeds=list(SEEDS), refinement_fits_per_arm=4,
              adaptive_fits_per_arm=6, repeat_candidates_per_arm=2,
              confirmation_order=["core", "five", "fusion", "ring"], bootstrap_resamples=10000,
              tie_epsilon=.002, max_per_class_recall_drop=.03,
              final_test_route="unresolved", held_out_evaluation=False)


def canonical(value):
    return json.loads(json.dumps(value, default=str))


def freeze(path, payload):
    """Idempotence is allowed; changing a frozen decision is not."""
    existing = base.read(path)
    base.require(existing is None or existing == canonical(payload),
                 f"Frozen artifact differs: {path}; use a new campaign identity.")
    base.save(path, payload)


def make_run(manifest, arm, parameters, *, seed=42, block="screen", stage="discovery",
             epochs=50, fold_index=None, ring=False):
    base.require(arm in ARMS or arm in EXTRA_ARMS, "Unknown campaign arm")
    family, scheme = (ARMS | EXTRA_ARMS)[arm]
    params = canonical(parameters)
    if arm.startswith("gvp_ring_"):
        params["shell_role_source"] = "geometry"
    recipe_id = base.fingerprint(dict(arm=arm, parameters=params))[:16]
    ident = f"{block}_{arm}_{recipe_id}_s{seed}"
    if fold_index is not None:
        ident += f"_f{fold_index}"
    command = list(manifest["templates"][family]["command"])
    output = Path(manifest["output_dir"])
    values = {**{k: v for k, v in params.items() if k != "capacity"},
              "metal_label_scheme": scheme, "metal_eligibility_scheme": "six_class",
              "selection_metric": base.METRIC, "epochs": epochs, "batch_size": 8,
              "seed": seed, "split_seed": 42, "val_fraction": .15,
              "split_stratify_by": "metal_site", "train_val_split_by": "pdbid",
              "metal_node_mode": "none", "structural_readout_scope": "residue_only",
              "site_geometry_features": "legacy", "run_name": ident, "runs_dir": output / "runs"}
    for key, value in values.items():
        command = base.replace_option(command, "--" + key.replace("_", "-"), value)
    if fold_index is not None:
        command = base.replace_option(command, "--n-folds", 5)
        command = base.replace_option(command, "--fold-index", fold_index)
    if ring:
        command = base.replace_option(
            command,
            "--ring-features-dir",
            Path(manifest["data_root"]) / "RING_features",
        )
        command += ["--use-ring-edges", "--require-ring-edges"]
    config = base.parse_config(command)
    validate(config, epochs=epochs, fold_index=fold_index, ring=ring)
    command[1] = str(Path(command[1]).with_name("train_serial_metal_profile.py"))
    payload = canonical(asdict(config))
    return dict(id=ident, arm=arm, family=family, scheme=scheme, parameters=params,
                recipe_id=recipe_id, seed=seed, lr=config.learning_rate, epochs=epochs,
                stage=stage, block=block, ring=ring, fold_index=fold_index,
                complexity_proxy={"compact": 0, "reference": 1, "large_256": 2,
                                  "large_384": 3, "large_512": 4}.get(params["capacity"], 5),
                command=list(map(str, command)),
                env={**manifest["templates"][family].get("env", {}),
                     "DEEPGM_METAL_LABEL_SCHEME": scheme},
                run_dir=str(output / "runs" / ident), config=payload,
                config_sha256=base.fingerprint(payload))


def validate(config, *, epochs=50, fold_index=None, ring=False):
    from training.run import validate_training_configuration
    validate_training_configuration(config)
    checks = {
        "metal-only native validation": config.task == "metal" and config.selection_metric == base.METRIC,
        "no held-out input": not config.run_test_eval and config.test_structure_dir is None
                             and config.test_summary_csv is None,
        "matched eligibility": config.metal_eligibility_scheme in ("six_class", "split_all_metals"),
        "fixed splitting": config.split_seed == 42 and config.train_val_split_by == "pdbid"
                           and config.split_stratify_by == "metal_site" and config.val_fraction == .15,
        "fixed folds": config.fold_index == fold_index and config.n_folds == (5 if fold_index is not None else None),
        "training budget": config.epochs == epochs and config.batch_size == 8 and config.seed in SEEDS,
        "fixed optimizer policy": config.lr_schedule == "fixed" and config.deterministic,
        "fixed features": config.node_feature_set == "conservative" and not config.omit_node_features
                          and config.require_all_task_classes and config.require_external_features,
        "no runtime feature generation": not config.prepare_missing_esm_embeddings and not config.prepare_missing_ring_edges,
        "geometry": config.edge_radius == 6 and config.classifier_pool_distance_cutoff == 0
                    and config.metal_node_mode == "none" and config.structural_readout_scope == "residue_only"
                    and config.site_geometry_features == "legacy",
        "fixed objective": config.metal_loss_function == "cross_entropy" and config.metal_collapsed_loss_weight == 0
                           and config.metal_label_smoothing == 0 and not config.balance_metal_site_symbols,
        "no augmentation": config.position_noise_std == config.second_shell_dropout == config.outer_residue_dropout == 0,
        "ring policy": config.use_ring_edges == ring and config.require_ring_edges == ring
                       and (not ring or config.shell_role_source == "geometry"),
    }
    base.require(all(checks.values()), "Serial profile mismatch: " + ", ".join(k for k, ok in checks.items() if not ok))


def reference_parameters(**overrides):
    return {**FIXED, **CAPACITIES["reference"], "capacity": "reference",
            "learning_rate": 3e-5, **overrides}


def late_five_candidate(manifest):
    """A frozen historical-recipe challenger, never a new five-class search."""
    return make_run(manifest, "late_five", reference_parameters(), block="five")


def initial_runs(manifest):
    """48 base cells, four mandatory large-late fits, and 13 readiness smokes."""
    runs = []
    for arm in ARMS:
        for capacity, sizes in CAPACITIES.items():
            for lr in LRS:
                runs.append(make_run(manifest, arm, {**FIXED, **sizes,
                                      "capacity": capacity, "learning_rate": lr}))
        runs.append(make_run(manifest, arm, reference_parameters(),
                             stage="operations", block="smoke", epochs=1))
    large = reference_parameters(**LARGE_CAPACITIES["large_256"], capacity="large_256")
    for arm in ("late_four", "late_six"):
        for seed in SEEDS:
            runs.append(make_run(manifest, arm, large, seed=seed, block="large"))
        runs.append(make_run(manifest, arm, large, stage="operations", block="smoke", epochs=1))
    runs.append(make_run(manifest, "late_five", reference_parameters(),
                         stage="operations", block="smoke", epochs=1))
    for arm in ("gvp_ring_off", "gvp_ring_on"):
        runs.append(make_run(manifest, arm, reference_parameters(), stage="operations",
                             block="smoke", epochs=1, ring=arm.endswith("on")))
    return runs


def top2_repeat_runs(manifest, selected):
    """Repeat exactly two frozen baseline-screen recipes per arm at seed 43."""
    base.require(set(selected) == set(ARMS), "Every base arm requires a two-recipe shortlist")
    runs = []
    for arm in ARMS:
        shortlist = selected[arm]
        base.require(len(shortlist) == 2 and len({row["recipe_id"] for row in shortlist}) == 2,
                     "Repeat shortlist must contain two distinct recipes per arm")
        for row in shortlist:
            base.require(row["arm"] == arm and row["block"] == "screen" and row["seed"] == 42,
                         "Top-two repeats must come from the baseline seed-42 screen")
            runs.append(make_run(manifest, arm, row["parameters"], seed=43, block="repeat"))
    return runs


def plan(root, data, output, external_root, overlay_manifest, source_commit="unknown"):
    root, data, output, external_root, overlay_manifest = map(
        lambda path: Path(path).resolve(), (root, data, output, external_root, overlay_manifest))
    base.validate_feature_overlay(data, external_root, overlay_manifest)
    templates = base._templates(root, data, output / "notebook_planning", external_root)
    train = data / base.DATASET / "train"
    source_files = sorted((root / "src").rglob("*.py")) + [
        root / "notebooks/DeepMzyme_training_colab.ipynb", root / "docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md"]
    for name in ("colab_serial_metal_host.py", "colab_serial_metal_supervisor.py"):
        host_controller = root / "scripts" / name
        if host_controller.is_file():
            source_files.append(host_controller)
    dataset_files = [train / "structure_manifest.csv",
                     train / "final_data_summarazing_table_transition_metals_only_catalytic.csv"]
    manifest = dict(profile=PROFILE, source_commit=source_commit, root=str(root), data_root=str(data),
                    output_dir=str(output), external_features_root_dir=str(external_root),
                    feature_overlay_manifest=dict(path=str(overlay_manifest), sha256=base.digest(overlay_manifest)),
                    dataset=base.DATASET, bundle=base.BUNDLE, bundle_sha256=base.BUNDLE_SHA,
                    source_files={str(p.relative_to(root)): base.digest(p) for p in source_files},
                    dataset_files={str(p.relative_to(data)): base.digest(p) for p in dataset_files},
                    templates=canonical(templates), policy=POLICY, arms=canonical(ARMS),
                    held_out_evaluation=False, evidence_status="planned", runs=[])
    # Keep template commands independent of the planning host interpreter.
    for template in manifest["templates"].values():
        template["command"] = list(map(str, template["command"]))
        template["command"][0] = sys.executable
    manifest["runs"] = initial_runs(manifest)
    output.mkdir(parents=True, exist_ok=True)
    freeze(output / "campaign_manifest.json", manifest)
    freeze(output / "queue.json", dict(runs=manifest["runs"], phase="screen"))
    (output / "commands.txt").write_text("\n\n".join(shlex.join(r["command"]) for r in manifest["runs"]) + "\n")
    base.csv_save(output / "run_matrix.csv", manifest["runs"],
                  ["id", "stage", "block", "arm", "family", "scheme", "seed", "epochs", "parameters", "run_dir"])
    return manifest


def verify_manifest(output):
    output = Path(output).resolve()
    manifest = base.read(output / "campaign_manifest.json")
    base.require(manifest and manifest.get("profile") == PROFILE and manifest["policy"] == POLICY,
                 "Missing or changed campaign profile")
    base.require(Path(manifest["output_dir"]) == output, "Use the frozen output path across sessions")
    for relative, expected in manifest["source_files"].items():
        base.require(base.digest(Path(manifest["root"]) / relative) == expected, f"Source changed: {relative}")
    for relative, expected in manifest["dataset_files"].items():
        base.require(base.digest(Path(manifest["data_root"]) / relative) == expected, f"Dataset changed: {relative}")
    overlay = manifest["feature_overlay_manifest"]
    base.require(base.digest(overlay["path"]) == overlay["sha256"], "Feature overlay manifest changed")
    return manifest


def refinement_parameters(family_key, candidates, repeated_results):
    """At most two settings, taking distinct triggered diagnoses first."""
    arms = [arm for arm in ARMS if arm.startswith(family_key + "_")]
    base.require(set(arms).issubset(candidates), "Missing paired family candidates")
    selected = {arm: candidates[arm] for arm in arms}
    rows = {arm: [r for r in repeated_results if r["arm"] == arm
                  and r["recipe_id"] == selected[arm]["recipe_id"] and r["seed"] in SEEDS] for arm in arms}
    base.require(all({r["seed"] for r in values} == set(SEEDS) and len(values) == 2 for values in rows.values()),
                 "Refinement requires both selected-recipe seeds")
    endpoints = {r["parameters"]["learning_rate"] for r in selected.values()} & {LRS[0], LRS[-1]}
    categories = []
    if endpoints:
        high = {"gvp": 3e-4, "esm": 2e-4, "late": 2e-4, "early": 2e-4, "hybrid": 1.5e-4}[family_key]
        changes = [{"learning_rate": value} for value in
                   ([5e-6] if LRS[0] in endpoints else []) + ([high] if LRS[-1] in endpoints else [])]
        categories.append(("boundary_learning_rate", changes))
    if any(any(float(value) == 0 for value in r["per_class_recall"].values()) for group in rows.values() for r in group):
        categories.append(("zero_native_recall", [{"metal_class_weight_mode": mode}
                                                 for mode in ("inverse_sqrt_frequency", "effective_number")]))
    if any(all(r.get("training_balanced_accuracy") is not None for r in group)
             and sum(float(r["training_balanced_accuracy"]) - r["balanced_accuracy"] for r in group) / 2 > .1
             for group in rows.values()):
        categories.append(("training_validation_gap", [dict(weight_decay=1e-3, head_mlp_dropout=.3),
                                                       dict(weight_decay=1e-4, head_mlp_dropout=.4)]))
    if family_key in ("early", "hybrid") and not categories:
        # "Weak" is only a scheduling diagnosis, never an architecture rejection.
        comparator = [r for r in repeated_results if r["arm"] == "late_four" and r["seed"] in SEEDS
                      and r["recipe_id"] == candidates["late_four"]["recipe_id"]]
        if len(comparator) == 2 and sum(r["balanced_accuracy"] for r in rows[arms[0]]) < sum(r["balanced_accuracy"] for r in comparator):
            categories.append(("early_bottleneck", [dict(early_esm_dim=value, early_esm_dropout=.1)
                                                    for value in (16, 64)]))
    # First take one variant from each category, then fill a remaining slot.
    chosen = [(name, variants[0]) for name, variants in categories][:2]
    if len(chosen) < 2:
        chosen.extend((name, variant) for name, variants in categories for variant in variants[1:])
    chosen = chosen[:2]
    parameters = {}
    for arm in arms:
        variants, boundary_slot = [], 0
        for name, change in chosen:
            if name == "boundary_learning_rate":
                own_lr = selected[arm]["parameters"]["learning_rate"]
                preferred = 5e-6 if own_lr == LRS[0] else high if own_lr == LRS[-1] else change["learning_rate"]
                menu = [preferred] + [value for value in
                    ([5e-6] if LRS[0] in endpoints else []) + ([high] if LRS[-1] in endpoints else [])
                    if value != preferred]
                change = {"learning_rate": menu[boundary_slot]}
                boundary_slot += 1
            variants.append({**selected[arm]["parameters"], **change})
        parameters[arm] = variants
    return dict(diagnosis="mixed" if len({name for name, _ in chosen}) > 1 else
                chosen[0][0] if chosen else "no_diagnostic",
                diagnoses=[name for name, _ in chosen], variants=[change for _, change in chosen],
                parameters=parameters, maximum_fits_per_arm=2 * len(chosen))


def confirmation_runs(manifest, candidates, include_late_five=True):
    base.require(set(candidates) == set(ARMS), "Freeze exactly one candidate per required base arm")
    runs = []
    blocks = [("core", [arm for arm in ARMS if arm.split("_")[0] in FAMILY_ORDER[:3]]),
              ("five", ["late_five"] if include_late_five else []),
              ("fusion", ["early_four", "hybrid_four"])]
    for block, arms in blocks:
        for arm in arms:
            chosen = late_five_candidate(manifest) if arm == "late_five" else candidates[arm]
            for fold in range(5):
                for seed in SEEDS:
                    runs.append(make_run(manifest, arm, chosen["parameters"], seed=seed,
                                         block=block, stage="confirmation", fold_index=fold))
    # The base screen preserves legacy shell roles. Fresh geometry-fixed controls
    # are intentionally safer than asserting equivalence from equal scores.
    for arm, ring in (("gvp_ring_off", False), ("gvp_ring_on", True)):
        for fold in range(5):
            for seed in SEEDS:
                runs.append(make_run(manifest, arm, candidates["gvp_four"]["parameters"], seed=seed,
                                     block="ring", stage="confirmation", fold_index=fold, ring=ring))
    return runs


def timing_identity(run):
    """Fields that prevent a cheap architecture from pricing a larger fit."""
    parameters = run.get("parameters", {})
    return dict(family=run["family"], scheme=run.get("scheme"), ring=bool(run.get("ring", False)),
                capacity={key: parameters.get(key) for key in
                          ("hidden_s", "hidden_v", "edge_hidden", "gvp_layers", "head_mlp_layers",
                           "esm_fusion_dim", "early_esm_dim")},
                fold_mode=run.get("fold_index") is not None,
                batch_size=run.get("config", {}).get("batch_size", 8))


def forecast_details(run, measurements):
    """Conservative measured proxy, exposing any capacity/fold extrapolation.

    The caller supplies only measurements from the active hardware. A larger
    capacity can conservatively price a smaller one, never the reverse.
    Discovery-to-fold extrapolation scales the epoch cost by the larger of the
    training and validation membership ratios; a new full fit replaces it.
    """
    target = timing_identity(run)
    costs = []
    for measured in measurements:
        if measured.get("family") != target["family"]:
            continue
        source = measured.get("timing_identity") or timing_identity(measured)
        if any(source.get(key) != target[key] for key in ("scheme", "ring", "batch_size")):
            continue
        if not all(isinstance(source["capacity"].get(key), (int, float)) and value is not None
                   and source["capacity"][key] >= value for key, value in target["capacity"].items()):
            continue
        same_fold_mode = source["fold_mode"] == target["fold_mode"]
        fold_factor = 1.0 if same_fold_mode else (4 / 3 if target["fold_mode"] else 17 / 16)
        if same_fold_mode and measured.get("epochs", 0) == run["epochs"]:
            cost = float(measured["elapsed_seconds"])
        elif measured.get("setup_seconds") is not None and measured.get("epoch_seconds") is not None:
            cost = float(measured["setup_seconds"]) + run["epochs"] * float(measured["epoch_seconds"]) * fold_factor
        else:
            continue
        base.require(math.isfinite(cost) and cost > 0, "Invalid measured full-fit cost")
        costs.append(dict(seconds=cost, full_fit=measured.get("epochs", 0) == run["epochs"],
                          capacity_upper_bound=source["capacity"] != target["capacity"],
                          fold_extrapolation=not same_fold_mode, source_run_id=measured.get("id", measured.get("run_id"))))
    base.require(costs, f"Missing compatible capacity/RING/fold timing for {run['family']}; profile it first")
    # Prefer direct full-fit measurements once the exact capacity and fold mode
    # are observed, rather than permanently retaining pessimistic smoke costs.
    exact = [row for row in costs if not row["capacity_upper_bound"] and not row["fold_extrapolation"]]
    exact_full = [row for row in exact if row["full_fit"]]
    return max(exact_full or exact or costs, key=lambda row: row["seconds"])


def forecast(run, measurements):
    return forecast_details(run, measurements)["seconds"]
