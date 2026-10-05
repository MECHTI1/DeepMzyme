#!/usr/bin/env python3
"""Plan step A3: CPU regression gate for the v3 code with every v3 option off.

Part 1, replay: the new code replays the nine pinned v2 fold-0 checkpoints on CPU.
  - Best checkpoints are compared with the saved predictions under the frozen
    pmm-core-replay-v1 policy (probability atol 1e-5; discrete fields exact).
  - Epoch-50 checkpoints are compared with the epoch-50 history record (the
    trainer's reconciliation: balanced accuracies within 1e-9, identical confusion
    matrices).
  A best checkpoint outside the policy is replayed again with the frozen v2
  checkout on the same CPU, so a device effect can be told from a code change.
Part 2, training: a short CPU fit on a synthetic campaign, run once with the new
  tree and once with the frozen v2 checkout from the same argv; every history value
  and every final weight must agree within 1e-6.

Kept outside src/ and scripts/ so it cannot change the frozen training identity.
Read-only on v2 data (outputs go to a new directory); never reads held-out data.
Each replay runs in its own process so label-scheme state never leaks between units.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
FROZEN_V2_TREE = Path("/media/mechti/Data1/DeepMzyme_Data/campaigns/_code/pmm_core_scope_v2")
V2_CAMPAIGN = Path("/media/mechti/Data1/DeepMzyme_Data/campaigns/pmm_ion_metal_v2_context")
TRAIN_DIR = Path("/media/mechti/Data1/DeepMzyme_PMM_Zenodo_Exact_Dataset/dataset/train")
# Paths recorded on the training VM -> their local copies.
PATH_MAP = {"/home/mechti/deepmzyme_data/pmm/train_and_test_sets_structures_zenodo_pmm_exact": str(TRAIN_DIR.parent),
            "/home/mechti/deepmzyme_runs/pmm_ion_metal_v2_context": str(V2_CAMPAIGN)}
WEIGHT_ATOL = 1e-6
HISTORY_ATOL = 1e-6
TRAINING_CONFIGS = (("only_esm", "four_class"), ("only_gvp", "six_class"),
                    ("gvp_late_fusion", "four_class"), ("gvp_late_fusion", "six_class"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Worker (one replay per process)
# ---------------------------------------------------------------------------

def worker(args) -> int:
    tree = ROOT if args.tree == "new" else FROZEN_V2_TREE
    sys.path.insert(0, str(tree / "src"))
    from training import campaign_runtime as runtime

    out = Path(args.out)
    result: dict[str, Any] = {"tree": args.tree, "tree_root": str(tree), "kind": args.worker,
                              "campaign_runtime": runtime.__file__}
    started = time.time()
    try:
        if args.worker == "best":
            receipt = runtime.replay_campaign_run(Path(args.run_dir), device="cpu", path_map=PATH_MAP,
                                                  output_dir=out / "replay")
        else:
            receipt = runtime.replay_epoch_checkpoint(Path(args.run_dir), "epoch_0050_checkpoint.pt",
                                                      device="cpu", path_map=PATH_MAP, output_dir=out / "replay")
        result.update(error=None, selected_epoch=receipt["selected_epoch"],
                      reconciliation_status=receipt["reconciliation_status"])
    except runtime.CampaignContractError as exc:
        result.update(error=str(exc))
    result["seconds"] = time.time() - started
    write_json(out / "worker_result.json", result)
    return 0


def spawn(kind: str, tree: str, run_dir: Path, out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "2"),
           "MKL_THREADING_LAYER": "GNU"}
    for key in ("DEEPMZYME_PARSE_CACHE_DIR", "DEEPMZYME_GRAPH_CACHE_DIR"):
        env.pop(key, None)
    with (out / "worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--worker", kind,
                                  "--tree", tree, "--run-dir", str(run_dir), "--out", str(out)],
                                 stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(out))
    if process.returncode != 0 or not (out / "worker_result.json").is_file():
        return {"error": f"worker exited {process.returncode}; see {out / 'worker.log'}"}
    return json.loads((out / "worker_result.json").read_text())


# ---------------------------------------------------------------------------
# Part 1: replay of the pinned v2 fold-0 checkpoints
# ---------------------------------------------------------------------------

def replay_unit(name: str, out: Path) -> dict[str, Any]:
    import pmm_core_replay as frozen_policy

    pin = frozen_policy.HISTORICAL_PINS[name]
    run_dir = V2_CAMPAIGN / "runs" / name
    target = name.split("__")[1]
    labels = frozen_policy.NATIVE_LABELS[target]
    record: dict[str, Any] = {"unit": name}
    record["pins_match"] = (sha256(run_dir / "best_model_checkpoint.pt") == pin["checkpoint_sha256"]
                            and sha256(run_dir / "val_predictions.csv") == pin["predictions_sha256"])
    if not record["pins_match"]:
        record["passed"] = False
        return record

    best = spawn("best", "new", run_dir, out / name / "best_new")
    record["best_new"] = best
    replay_csv = out / name / "best_new" / "replay" / "val_predictions.csv"
    if replay_csv.is_file():
        replayed = frozen_policy.read_rows(replay_csv)
        saved = frozen_policy.read_rows(run_dir / "val_predictions.csv")
        try:
            record["best_policy_comparison"] = frozen_policy.compare_rows(saved, replayed, labels,
                                                                          frozen_policy.POLICY)
        except ValueError as exc:
            record["best_policy_comparison"] = {"qualified": False, "error": str(exc)}
        record["best_replay_sha256"] = sha256(replay_csv)
        record["best_bitwise_equal_to_a_pinned_replay"] = record["best_replay_sha256"] in pin["replay_prediction_sha256s"]
    best_ok = bool(record.get("best_policy_comparison", {}).get("qualified"))
    if not best_ok:
        frozen = spawn("best", "frozen", run_dir, out / name / "best_frozen")
        record["best_frozen"] = frozen
        frozen_csv = out / name / "best_frozen" / "replay" / "val_predictions.csv"
        if replay_csv.is_file() and frozen_csv.is_file():
            record["new_equals_frozen_on_this_cpu"] = sha256(replay_csv) == sha256(frozen_csv)

    last = spawn("epoch50", "new", run_dir, out / name / "epoch50_new")
    record["epoch50_new"] = last
    record["epoch50_matches_history"] = last.get("reconciliation_status") == "match" and not last.get("error")
    if not record["epoch50_matches_history"]:
        receipt = out / name / "epoch50_new" / "replay" / "selected_checkpoint.json"
        if receipt.is_file():
            record["epoch50_reconciliation"] = json.loads(receipt.read_text()).get("reconciliation")
    record["passed"] = record["pins_match"] and best_ok and record["epoch50_matches_history"]
    record["passed_by_device_attribution"] = (record["pins_match"] and not best_ok
                                              and record.get("new_equals_frozen_on_this_cpu") is True
                                              and record["epoch50_matches_history"])
    return record


# ---------------------------------------------------------------------------
# Part 2: deterministic CPU training, new tree vs frozen v2 checkout
# ---------------------------------------------------------------------------

def build_synthetic_campaign(root: Path):
    """The test-suite synthetic campaign (30 PDB entries, 8-dimensional fake ESM embeddings)."""
    import pytest

    sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests")]
    from benchmarking import pmm_ion_campaign as v2
    from benchmarking import pmm_ion_features as features
    from pmm_campaign_fixtures import write_fake_embeddings
    from test_pmm_training_integrity import make_campaign
    from test_v3_campaign import certify_fake_inventory

    patch = pytest.MonkeyPatch()
    train, paths = make_campaign(root, patch, n_pdb=30)
    v2.freeze_folds(paths)
    patch.setattr(features, "ESM_DIM", 8)
    patch.setattr(v2, "ESM_DIM", 8)
    features.plan_embeddings(paths, train)
    esm_dir = root / "esm"
    write_fake_embeddings(train, esm_dir, dim=8)
    certify_fake_inventory(paths, esm_dir)
    return v2, train, paths, esm_dir


def compare_fits(new_dir: Path, frozen_dir: Path) -> dict[str, Any]:
    import math

    import torch

    new_hist = json.loads((new_dir / "run_config.json").read_text())["history"]
    old_hist = json.loads((frozen_dir / "run_config.json").read_text())["history"]
    worst_history, mismatched_keys = 0.0, []
    if len(new_hist) != len(old_hist):
        return {"passed": False, "error": "history lengths differ"}
    for a, b in zip(new_hist, old_hist):
        for key in sorted(set(a) | set(b)):
            x, y = a.get(key), b.get(key)
            if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
                if math.isfinite(x) and math.isfinite(y):
                    worst_history = max(worst_history, abs(float(x) - float(y)))
                elif not (math.isnan(x) and math.isnan(y)) and x != y:
                    mismatched_keys.append(key)
            elif x != y:
                mismatched_keys.append(key)
    weights = {}
    for name in ("last_model_checkpoint.pt", "best_model_checkpoint.pt"):
        a = torch.load(new_dir / name, map_location="cpu", weights_only=False)["model_state_dict"]
        b = torch.load(frozen_dir / name, map_location="cpu", weights_only=False)["model_state_dict"]
        if set(a) != set(b):
            weights[name] = {"error": "parameter names differ", "only_new": sorted(set(a) - set(b))[:10],
                             "only_frozen": sorted(set(b) - set(a))[:10]}
            continue
        worst = max(float((a[k].double() - b[k].double()).abs().max()) if a[k].numel() else 0.0 for k in a)
        weights[name] = {"max_abs_difference": worst, "n_tensors": len(a)}
    weight_ok = all("error" not in w and w["max_abs_difference"] <= WEIGHT_ATOL for w in weights.values())
    return {"passed": weight_ok and worst_history <= HISTORY_ATOL and not mismatched_keys,
            "history_max_abs_difference": worst_history, "history_mismatched_keys": sorted(set(mismatched_keys)),
            "weights": weights, "epochs": len(new_hist)}


def training_check(out: Path, epochs: int) -> dict[str, Any]:
    v2, train, paths, esm_dir = build_synthetic_campaign(out / "synthetic")
    results = {}
    for family, target in TRAINING_CONFIGS:
        config = v2.GridConfig(family, target, "none")
        runs = {}
        for tree, root in (("new", ROOT), ("frozen", FROZEN_V2_TREE)):
            command, env_extra, _ = v2.build_train_command(
                paths, python_bin=sys.executable, train_dir=train, config=config, fold=1, seed=42, device="cpu",
                runs_dir=out / f"runs_{tree}", epochs=epochs, load_workers=1, save_epoch_checkpoints=False)
            if v2.FAMILIES[family]["uses_esm"]:
                command[command.index("--esm-embeddings-dir") + 1] = str(esm_dir)
            command[2] = str(root / "src" / "train.py")
            env = {**os.environ, **env_extra, "MKL_THREADING_LAYER": "GNU"}
            for key in ("DEEPMZYME_PARSE_CACHE_DIR", "DEEPMZYME_GRAPH_CACHE_DIR"):
                env.pop(key, None)
            log = out / f"runs_{tree}" / f"{config.config_id}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("w", encoding="utf-8") as handle:
                process = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=env, cwd=str(root))
            runs[tree] = {"return_code": process.returncode, "run_dir": command[command.index("--run-name") + 1]}
        name = runs["new"]["run_dir"]
        if runs["new"]["return_code"] or runs["frozen"]["return_code"]:
            results[config.config_id] = {"passed": False, "runs": runs}
            continue
        results[config.config_id] = compare_fits(out / "runs_new" / name, out / "runs_frozen" / name)
    return {"passed": all(r["passed"] for r in results.values()), "configs": results,
            "weight_atol": WEIGHT_ATOL, "history_atol": HISTORY_ATOL, "epochs": epochs,
            "data": "synthetic test-suite campaign (30 PDB entries, fake 8-dim ESM); fold 1; seed 42"}


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--part", choices=("replay", "training", "both"), default="both")
    parser.add_argument("--units", nargs="*", help="replay: subset of the nine pinned units")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--worker", choices=("best", "epoch50"), help=argparse.SUPPRESS)
    parser.add_argument("--tree", choices=("new", "frozen"), help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", help=argparse.SUPPRESS)
    parser.add_argument("--out", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return worker(args)
    import pmm_core_replay as frozen_policy

    out = args.out_root / f"a3_regression_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    out.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {"schema": "v3-a3-regression-1", "audit_sha256": sha256(Path(__file__)),
                              "policy_id": frozen_policy.POLICY_ID, "policy_sha256": frozen_policy.POLICY_SHA256,
                              "frozen_tree": str(FROZEN_V2_TREE), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    write_json(out / "a3_report.json", report)
    if args.part in ("replay", "both"):
        units = args.units or sorted(frozen_policy.HISTORICAL_PINS)
        report["replay"] = {}
        for name in units:
            report["replay"][name] = replay_unit(name, out / "replay")
            write_json(out / "a3_report.json", report)
            print(name, "passed" if report["replay"][name]["passed"] else "NOT PASSED", flush=True)
    if args.part in ("training", "both"):
        report["training"] = training_check(out / "training", args.epochs)
        print("training", "passed" if report["training"]["passed"] else "NOT PASSED", flush=True)
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json(out / "a3_report.json", report)
    print(json.dumps({"report": str(out / "a3_report.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
