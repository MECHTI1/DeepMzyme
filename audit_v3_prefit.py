#!/usr/bin/env python3
"""Plan step A5: pre-fit gate on the real v3 fold set (CPU; training side only).

1. Scratch-prepare a v3 campaign root (never the real one) from the v2 cohort and
   the frozen v3 fold set; preparation refuses any fold that lacks a native class
   in training or validation.
2. Per fold: native and common-four support in training and validation; class
   weights finite and positive; equal weighted training mass for the four common
   classes; the weights each target scheme passes to the trainer.
3. Label collapse: every target scheme maps each native element to the same
   common-four class as the frozen element map.
4. Every planned step C-E unit builds a valid training command on the scratch root.
5. Timing on a seeded sample of training PDB entries: structure loading, one graph
   build, and one augmented rebuild (coordinate noise 0.1 A plus outer-residue
   dropout 0.1) per ion, used to forecast augmented runs (every training graph is
   rebuilt each epoch) and cache-bypassing RING runs.

Normalization on training graphs only and the trainer's use of the class weights
are covered by tests (test_v3_training_options, test_v3_campaign, test_v3_assessment).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import pmm_v3_campaign as v3  # noqa: E402
from benchmarking import pmm_ion_campaign as v2  # noqa: E402

V2_ROOT = Path("/media/mechti/Data1/DeepMzyme_Data/campaigns/pmm_ion_metal_v2_context")
FOLD_DIR = Path("/media/mechti/Data1/DeepMzyme_Data/campaigns/pmm_ion_metal_v3/folds/v3-seqid90-s42-b2")
TRAIN_DIR = Path("/media/mechti/Data1/DeepMzyme_PMM_Zenodo_Exact_Dataset/dataset/train")
ESM_DIR = V2_ROOT / "inputs" / "esm_embeddings_esmc600m_v1"
COMMON4 = ("Mn", "Cu", "Zn", "Class VIII")
ELEMENT_TO_COMMON4 = {"MN": 0, "CU": 1, "ZN": 2, "FE": 3, "CO": 3, "NI": 3}


def check(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def support_and_weights(paths: v3.V3Paths, failures: list[str]) -> dict[str, Any]:
    weights = json.loads(paths.fold_class_weights.read_text())
    out = {}
    for fold in range(v3.N_FOLDS):
        entry = weights["folds"][str(fold)]
        w, counts = entry["common_four_weights"], entry["train_common_four_counts"]
        check(entry["runnable"] and not entry["absent_native_classes"], f"fold {fold} not runnable", failures)
        check(all(math.isfinite(w[c]) and w[c] > 0 for c in COMMON4), f"fold {fold}: invalid weights", failures)
        mass = {c: w[c] * counts[c] for c in COMMON4}
        check(max(mass.values()) - min(mass.values()) <= 1e-6 * entry["n_train"],
              f"fold {fold}: weighted class mass not equal", failures)
        check(sum(counts.values()) == entry["n_train"], f"fold {fold}: common-four counts do not sum", failures)
        out[fold] = {"n_train": entry["n_train"], "n_val": entry["n_val"],
                     "train_native": entry["train_native_counts"], "val_native": entry["val_native_counts"],
                     "train_common_four": counts, "weights": w,
                     "trainer_weights": {"four_class": [w["Mn"], w["Cu"], w["Zn"], w["Class VIII"]],
                                         "five_class": [w["Mn"], w["Cu"], w["Zn"], w["Class VIII"], w["Class VIII"]],
                                         "six_class": [w["Mn"], w["Cu"], w["Zn"]] + [w["Class VIII"]] * 3}}
    return out


def collapse_check(failures: list[str]) -> dict[str, Any]:
    from label_schemes import METAL_LABEL_SCHEMES, collapsed_metal_target_for_label_name

    out = {}
    for target, scheme in v3.TARGET_SCHEMES.items():
        index_to_label, element_to_index = METAL_LABEL_SCHEMES[scheme]
        mapping = {element: collapsed_metal_target_for_label_name(index_to_label[index])
                   for element, index in element_to_index.items()}
        check(mapping == ELEMENT_TO_COMMON4, f"{target}: collapse differs from the element map", failures)
        out[target] = {"labels": [index_to_label[i] for i in sorted(index_to_label)], "element_to_common4": mapping}
    return out


def commands_check(paths: v3.V3Paths, failures: list[str]) -> dict[str, Any]:
    units = [u for step in ("C", "D-A", "D-B", "E-neutral") for u in v3.step_units(step)]
    built = 0
    for unit in units:
        try:
            command, _, _ = v3.build_command(paths, unit, python_bin=sys.executable, train_dir=TRAIN_DIR,
                                             esm_dir=ESM_DIR, device="cuda", lane=0)
            built += 1
        except (ValueError, SystemExit) as exc:
            failures.append(f"{unit.name}: command refused: {exc}")
            continue
        weights = json.loads(paths.fold_class_weights.read_text())["folds"][str(unit.fold)]["common_four_weights"]
        if v3.resolve_recipe(unit.recipe)["weight_mode"] == "manual":
            flag = command[command.index("--class-viii-loss-multiplier") + 1] if unit.target != "six_class" else \
                command[command.index("--ni-loss-multiplier") + 1]
            check(float(flag) == weights["Class VIII"], f"{unit.name}: Class VIII multiplier differs", failures)
    return {"units": len(units), "built": built}


def timing(paths: v3.V3Paths, n_entries: int, seed: int) -> dict[str, Any]:
    """Seconds per ion: loading, one graph build, and one augmented rebuild."""
    from training.config import parse_args
    from training.graph_dataset import augment_pocket_for_training, pocket_to_pyg_data
    from training.source_cohort import cohort_load_payload, read_cohort_csv
    from training.data import _cohort_structure_files
    from training.structure_loading import load_structure_pockets
    from label_schemes import configure_active_metal_label_scheme

    manifest = json.loads(paths.manifest.read_text())
    bindings = read_cohort_csv(paths.cohort, manifest["cohort"]["sha256"])
    groups = sorted({b.group_id for b in bindings})
    sample = set(random.Random(seed).sample(groups, n_entries))
    chosen = [b for b in bindings if b.group_id in sample]
    result = {}
    for family in ("only_gvp", "gvp_late_fusion"):
        unit = v3.Unit(family, "four_class", "baseline", 0, 42)
        command, _, _ = v3.build_command(paths, unit, python_bin=sys.executable, train_dir=TRAIN_DIR,
                                         esm_dir=ESM_DIR, device="cpu", lane=0)
        config = parse_args(command[3:])
        configure_active_metal_label_scheme(config.metal_label_scheme)
        options = dict(esm_dim=config.esm_dim, edge_radius=config.edge_radius, use_ring_edges=False,
                       require_ring_edges=False, node_feature_set=config.node_feature_set,
                       omit_node_features=config.omit_node_features, metal_node_mode=config.metal_node_mode,
                       shell_role_source=config.shell_role_source)
        started = time.perf_counter()
        pockets = []
        payload = cohort_load_payload(chosen)
        for path in _cohort_structure_files(TRAIN_DIR, chosen):
            loaded, _, _ = load_structure_pockets(
                structure_path=path, structure_root=TRAIN_DIR, allowed_site_metal_labels=None,
                esm_dim=config.esm_dim, embeddings_dir=Path(config.esm_embeddings_dir),
                require_esm_embeddings=config.require_esm_embeddings, ring_features_dir=None,
                feature_root_dir=Path(config.external_features_root_dir),
                external_feature_source=config.external_feature_source,
                require_external_features=config.require_external_features,
                unsupported_metal_policy=config.unsupported_metal_policy, ec_label_depth=config.ec_label_depth,
                metal_example_unit="ion", cohort_bindings=payload)
            pockets.extend(loaded)
        load_seconds = time.perf_counter() - started
        started = time.perf_counter()
        for pocket in pockets:
            pocket_to_pyg_data(pocket, **options)
        build_seconds = time.perf_counter() - started
        started = time.perf_counter()
        for pocket in pockets:
            pocket_to_pyg_data(augment_pocket_for_training(
                pocket, position_noise_std=0.1, outer_residue_dropout=0.1,
                shell_role_source=config.shell_role_source), **options)
        augment_seconds = time.perf_counter() - started
        n = len(pockets)
        result[family] = {"ions": n, "pdb_entries": len(sample),
                          "load_seconds_per_ion": load_seconds / n, "graph_seconds_per_ion": build_seconds / n,
                          "augmented_graph_seconds_per_ion": augment_seconds / n}
    return result


def forecast(paths: v3.V3Paths, timings: dict[str, Any]) -> dict[str, Any]:
    weights = json.loads(paths.fold_class_weights.read_text())["folds"]
    n_train = max(entry["n_train"] for entry in weights.values())
    n_all = sum(entry["n_val"] for entry in weights.values())
    epochs = v3.PROFILE["epochs"]
    out = {"n_train_max": n_train, "n_ions": n_all, "epochs": epochs,
           "basis": "this PC's CPU, one process; the VM CPU and loader workers change the absolute values"}
    for family, t in timings.items():
        out[family] = {
            "augmented_rebuild_hours_per_fit_serial": t["augmented_graph_seconds_per_ion"] * n_train * epochs / 3600,
            "uncached_graph_build_hours_per_fit": t["graph_seconds_per_ion"] * n_all / 3600,
            "structure_load_hours_per_fit_uncached": t["load_seconds_per_ion"] * n_all / 3600,
        }
    out["ring"] = ("RING runs bypass the graph cache (one uncached build per fit, above) and first need RING files "
                   "for every training structure; none exist, so RING generation time is not forecast here.")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--sample-entries", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    from training.access_guard import install_forbidden_read_guard

    install_forbidden_read_guard(v2.forbidden_read_roots(TRAIN_DIR))
    out = args.out_root / f"a5_prefit_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    out.mkdir(parents=True, exist_ok=False)
    paths = v3.V3Paths(out / "scratch_campaign")
    failures: list[str] = []
    manifest = v3.prepare_campaign(paths.root, v2_root=V2_ROOT, fold_dir=FOLD_DIR)
    report: dict[str, Any] = {"schema": "v3-a5-prefit-1", "scratch_root": str(paths.root),
                              "note": "scratch preparation only; the real campaign root is prepared after the last src change",
                              "fold_set": manifest["fold_set"], "cohort": manifest["cohort"],
                              "frozen_source_tree_sha256": manifest["frozen_source_tree_sha256"]}
    report["support_and_weights"] = support_and_weights(paths, failures)
    report["collapse"] = collapse_check(failures)
    report["commands"] = commands_check(paths, failures)
    report["timing"] = timing(paths, args.sample_entries, args.seed)
    report["forecast"] = forecast(paths, report["timing"])
    report["failures"] = failures
    report["passed"] = not failures
    (out / "a5_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "failures": failures[:10], "report": str(out / "a5_report.json")},
                     indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
