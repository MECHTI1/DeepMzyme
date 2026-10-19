#!/usr/bin/env python3
"""Evaluate Phase 1 & Phase 2 checkpoints on 5-Fold OOF CV and the Held-Out PinMyMetal Test Set."""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/mechti/PycharmProjects/DeepMzyme")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import pmm_v3_test_prep as prep  # noqa: E402
import pmm_v3_step_f_report as rep  # noqa: E402
import run_pmm_v3_step_f as step_f  # noqa: E402

DATA1_CAMPAIGN = Path("/media/mechti/Data1/DeepMzyme_Data/campaigns/pmm_ion_metal_v3")
NVME_V3_BACKUP = Path("/home/mechti/deepmzyme_backups/pmm_ion_metal_v3_closeout_20261013T023038Z/campaigns/pmm_ion_metal_v3")
NVME_TLR_BACKUP = Path("/home/mechti/deepmzyme_backups/deepmzyme_closeout_ec_readiness_20261013T190136Z/initial")
NVME_PHASE2_BACKUP = Path("/home/mechti/deepmzyme_backups/phase2_unified_and_180ckpt_eval_20261019T0317Z")

CAMPAIGN_DIR = DATA1_CAMPAIGN if DATA1_CAMPAIGN.exists() else NVME_V3_BACKUP
PREPARED_DIR = CAMPAIGN_DIR / "step_f_test/prepared_20261012T204640Z"
SOURCE_CSV = ROOT / "prepare_training_and_test_set/pinmymetal_files/classmodel_test_set"

BRAIN_OUT_DIR = Path("/home/mechti/.gemini/antigravity-acp/brain/b7335608-914a-409e-8e2d-7207fe2d6e18/scratch/phase1_results")
OUT_DIR = BRAIN_OUT_DIR if BRAIN_OUT_DIR.exists() else (NVME_PHASE2_BACKUP / "evaluation")

DATA1_UNIFIED = Path("/media/mechti/Data1/DeepMzyme_Data/campaigns/phase2_unified_2026-10/runs")
UNIFIED_DIR = DATA1_UNIFIED if DATA1_UNIFIED.exists() else (NVME_PHASE2_BACKUP / "runs")
REPO_SUMMARY_JSON = ROOT / "docs/campaigns/phase2_unified_2026-10/phase2_unified_summary.json"


def eval_oof_from_csvs(csv_paths: list[Path]) -> dict:
    cols = ["p_common4_Mn", "p_common4_Cu", "p_common4_Zn", "p_common4_Class_VIII"]
    oof_dfs = []
    fold_bas = []
    for p_csv in csv_paths:
        df = pd.read_csv(p_csv)
        oof_dfs.append(df)
        p = df[cols].to_numpy()
        y = df["y_common4"].to_numpy(dtype=int)
        pred = p.argmax(axis=1)
        recs = [float((pred[y == c] == c).mean()) for c in range(4)]
        fold_bas.append(100.0 * float(np.mean(recs)))

    oof_all = pd.concat(oof_dfs, ignore_index=True)
    p_oof = oof_all[cols].to_numpy()
    y_oof = oof_all["y_common4"].to_numpy(dtype=int)
    pred_oof = p_oof.argmax(axis=1)
    oof_recs = [100.0 * float((pred_oof[y_oof == c] == c).mean()) for c in range(4)]
    oof_ba = float(np.mean(oof_recs))
    return {
        "oof_ba": round(oof_ba, 2),
        "fold_mean_ba": round(float(np.mean(fold_bas)), 2),
        "fold_bas": [round(x, 2) for x in fold_bas],
        "oof_recalls": {rep.COMMON4[c]: round(oof_recs[c], 2) for c in range(4)},
    }


def main():
    t0 = time.time()
    cache_file = OUT_DIR / "test_pockets_and_graphs.pt"
    probs_npz_path = OUT_DIR / "test_checkpoint_probs.npz"
    loaded = np.load(probs_npz_path)
    saved_probs = {k: loaded[k] for k in loaded.files}
    print(f"Loaded {len(saved_probs)} existing checkpoints from {probs_npz_path}", flush=True)

    todo_folds = [
        f for f in range(5)
        if f"unified__P_five_class_trunklr1e4__fold{f}__seed42" not in saved_probs
    ]
    if todo_folds:
        import torch
        from torch_geometric.loader import DataLoader
        from training.campaign_runtime import load_campaign_prediction_components
        from training.graph_dataset import PocketGraphDataset

        print(f"Loading cached test pockets and graphs for folds {todo_folds}...", flush=True)
        cached_payload = torch.load(cache_file, map_location="cpu", weights_only=False)
        pockets = cached_payload["pockets"]
        graphs = cached_payload["graphs"]
        torch.set_num_threads(4)

        for f in todo_folds:
            key = f"unified__P_five_class_trunklr1e4__fold{f}__seed42"
            ckpt_path = UNIFIED_DIR / key / "terminal_model_checkpoint.pt"
            t_f = time.time()
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model, normalization, options = load_campaign_prediction_components(ckpt, device="cpu")
            del ckpt
            dataset = PocketGraphDataset(pockets, precomputed_data=graphs, normalization_stats=normalization, **options)
            loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
            model.eval()
            chunks = []
            with torch.no_grad():
                for batch in loader:
                    chunks.append(model(batch)["logits_metal"].float().cpu())
            logits = torch.cat(chunks, dim=0)
            probs = step_f.common_four(logits.softmax(-1)).numpy().astype(np.float64)
            saved_probs[key] = probs
            del dataset, loader, model
            gc.collect()
            print(f"  Evaluated {key} in {time.time() - t_f:.1f}s", flush=True)

        np.savez(probs_npz_path, **saved_probs)
        print(f"Saved {len(saved_probs)} total checkpoints to {probs_npz_path}", flush=True)
        del pockets, graphs, cached_payload
        gc.collect()

    # 1. OOF CV evaluation for Baseline 4-class, trunklr1e4 4-class, and Phase 2 Unified 5-Fold
    unified_csvs = [
        UNIFIED_DIR / f"unified__P_five_class_trunklr1e4__fold{f}__seed42" / "val_predictions.csv"
        for f in range(5)
    ]
    unified_cv = eval_oof_from_csvs(unified_csvs)
    print(
        f"\nUnified 5-Fold OOF CV Balanced Accuracy: {unified_cv['oof_ba']:.2f}% "
        f"(fold mean: {unified_cv['fold_mean_ba']:.2f}%)"
    )

    baseline_cv = None
    trunklr_cv = None
    if NVME_TLR_BACKUP.exists():
        base_csvs = [
            next(
                p for p in NVME_TLR_BACKUP.rglob("val_predictions.csv")
                if "baseline" in p.parent.name and f"fold{f}__seed42" in p.parent.name
            )
            for f in range(5)
        ]
        tlr_csvs = [
            next(
                p for p in NVME_TLR_BACKUP.rglob("val_predictions.csv")
                if "trunklr1e4" in p.parent.name and f"fold{f}__seed42" in p.parent.name
            )
            for f in range(5)
        ]
        baseline_cv = eval_oof_from_csvs(base_csvs)
        trunklr_cv = eval_oof_from_csvs(tlr_csvs)
        print(
            f"Baseline 4-class 5-Fold OOF CV BA: {baseline_cv['oof_ba']:.2f}% "
            f"(fold mean: {baseline_cv['fold_mean_ba']:.2f}%)"
        )
        print(
            f"trunklr1e4 4-class 5-Fold OOF CV BA: {trunklr_cv['oof_ba']:.2f}% "
            f"(fold mean: {trunklr_cv['fold_mean_ba']:.2f}%)"
        )

    # 2. Held-out test evaluation + 10,000-resample PDB-cluster bootstrap
    rows = rep.read_csv(PREPARED_DIR / "prepared_rows.csv")
    primaries = [r for r in rows if r["disposition"] != prep.UNSCORABLE and r["alias_primary_source_uid"] == r["source_uid"]]
    by_structure = {}
    for r in primaries:
        by_structure.setdefault(r["structure_name"], []).append(r)
    ordered_uids = [r["source_uid"] for name, group in sorted(by_structure.items()) for r in group]
    uid_to_idx = {u: i for i, u in enumerate(ordered_uids)}
    labels = rep.read_labels(SOURCE_CSV, expected_sha256=prep.SPEC["test_inputs"]["source_csv_sha256"], n_rows=int(prep.SPEC["test_rows"]["source_rows"]))

    def score_probs(probs_matrix, n_boot=10000):
        fake_preds = []
        for r in rows:
            rec = dict(r)
            if r["disposition"] == prep.UNSCORABLE:
                rec["pred_common4"] = ""
            else:
                p_uid = r["alias_primary_source_uid"]
                vec = probs_matrix[uid_to_idx[p_uid]]
                rec["pred_common4"] = str(int(vec.argmax()))
            fake_preds.append(rec)
        j_rows, _ = rep.joined_rows(fake_preds, labels)
        out = {}
        for line in ("full_test", "link_free", "training_eligible", "clean_subset"):
            members = rep.line_members(j_rows, line)
            supp, corr = rep.counts(members)
            recs = [100.0 * corr[c] / supp[c] for c in range(4)]
            ba = float(np.mean(recs))
            clusters = {}
            for m in members:
                clusters.setdefault(m["pdbid"], []).append((m["label"], int(m["scored"] and m["pred"] == m["label"])))
            c_keys = sorted(clusters.keys())
            n_c = len(c_keys)
            c_supp = np.zeros((n_c, 4), dtype=np.float64)
            c_corr = np.zeros((n_c, 4), dtype=np.float64)
            for idx_c, ck in enumerate(c_keys):
                for lbl, ok in clusters[ck]:
                    c_supp[idx_c, lbl] += 1.0
                    c_corr[idx_c, lbl] += float(ok)
            rng = np.random.default_rng(42)
            boot_idx = rng.integers(0, n_c, size=(n_boot, n_c))
            b_supp = c_supp[boot_idx].sum(axis=1)
            b_corr = c_corr[boot_idx].sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                b_recs = 100.0 * b_corr / b_supp
            b_ba = np.nanmean(b_recs, axis=1)
            ci_lo, ci_hi = float(np.percentile(b_ba, 2.5)), float(np.percentile(b_ba, 97.5))
            out[line] = {
                "rows": int(len(members)),
                "ba": round(ba, 2),
                "ci95": [round(ci_lo, 2), round(ci_hi, 2)],
                "recalls": {rep.COMMON4[c]: round(recs[c], 2) for c in range(4)},
            }
        elig = rep.line_members(j_rows, "training_eligible")
        by_pdb_metal = {}
        for r in elig:
            key = (r["pdbid"], r["label"])
            by_pdb_metal.setdefault(key, []).append(int(r["scored"] and r["pred"] == r["label"]))
        supp = [0] * 4
        corr = [0.0] * 4
        for (pdbid, lbl), vals in by_pdb_metal.items():
            supp[lbl] += 1
            corr[lbl] += float(np.mean(vals))
        out["pdb_dedup"] = {
            "pairs": int(sum(supp)),
            "ba": round(float(np.mean([100.0 * corr[c] / supp[c] for c in range(4)])), 2),
        }
        return out

    def fam(prefix: str, seeds: tuple[int, ...] = (42,)) -> np.ndarray:
        keys = [
            ("gvp_late_fusion__four_class__baseline" if (prefix == "mc_B_four_class" and s == 42) else prefix)
            + f"__fold{f}__seed{s}"
            for s in seeds
            for f in range(5)
        ]
        return np.mean([saved_probs[k] for k in keys], axis=0)

    P_stepf = saved_probs["stage6b_refit"]
    P_b4_s42 = fam("gvp_late_fusion__four_class__baseline")
    P_b4_3s = fam("mc_B_four_class", (42, 43, 44))
    P_tlr = fam("gvp_late_fusion__four_class__trunklr1e4")
    P_p5_s42 = fam("mc_P_five_class")
    P_p5_3s = fam("mc_P_five_class", (42, 43, 44))
    P_s5 = fam("mc_S_five_class")
    P_c6 = fam("gvp_late_fusion__six_class__baseline")
    P_unified = fam("unified__P_five_class_trunklr1e4")

    P_esmc4 = fam("only_esm__four_class__baseline")
    P_esmc5 = fam("only_esm__five_class__baseline")
    P_esmc6 = fam("only_esm__six_class__baseline")
    P_gvp4 = fam("only_gvp__four_class__baseline")
    P_gvpsca = fam("only_gvp__four_class__sitecountsangles")
    P_gvp5 = fam("only_gvp__five_class__baseline")
    P_gvp6 = fam("only_gvp__six_class__baseline")

    P_top3_lf = (P_tlr + P_p5_3s + P_c6) / 3.0
    P_top4_lf = (P_tlr + P_p5_3s + P_s5 + P_c6) / 4.0
    P_top5_lf = (P_unified + P_tlr + P_p5_3s + P_s5 + P_c6) / 5.0

    candidates = {
        "step_f_single_refit_s42": P_stepf,
        "baseline_5fold_s42": P_b4_s42,
        "baseline_15model_3s": P_b4_3s,
        "trunklr1e4_5fold_s42": P_tlr,
        "recipe_p_5class_5fold_s42": P_p5_s42,
        "recipe_p_5class_15model_3s": P_p5_3s,
        "unified_5fold_s42": P_unified,
        "top3_lf": P_top3_lf,
        "top4_lf": P_top4_lf,
        "unified_plus_top4_lf": P_top5_lf,
        "trimodal_cv_locked_top3_lf": 1.00 * P_top3_lf + 0.25 * ((P_esmc4 + P_esmc5 + P_esmc6) / 3.0) + 0.95 * ((P_gvpsca + P_gvp5 + P_gvp6) / 3.0),
        "unified_trimodal_cv_locked": 1.00 * P_top5_lf + 0.25 * ((P_esmc4 + P_esmc6) / 2.0) + 0.95 * ((P_gvp4 + P_gvpsca) / 2.0),
    }

    results = {
        "oof_cv": {
            "baseline_4class_s42": baseline_cv,
            "trunklr1e4_4class_s42": trunklr_cv,
            "unified_P_5class_trunklr1e4_s42": unified_cv,
            "recipe_p_5class_15model_3s_oof_ba": 75.82,
            "top3_lf_oof_ba": 76.62,
            "top4_lf_oof_ba": 76.81,
            "top5_lf_oof_ba": 76.37,
            "trimodal_cv_locked_top3_lf_oof_ba": 77.05,
            "unified_trimodal_cv_locked_oof_ba": 77.17,
        },
        "test_models": {},
    }
    print("\n" + "=" * 130)
    for name, P in candidates.items():
        res = score_probs(P)
        results["test_models"][name] = res
        rec = res["training_eligible"]["recalls"]
        print(
            f"{name:30s} | Full({res['full_test']['rows']})={res['full_test']['ba']:.2f}% {res['full_test']['ci95']} | "
            f"LF({res['link_free']['rows']})={res['link_free']['ba']:.2f}% | "
            f"Elig({res['training_eligible']['rows']})={res['training_eligible']['ba']:.2f}% {res['training_eligible']['ci95']} | "
            f"Dedup({res['pdb_dedup']['pairs']})={res['pdb_dedup']['ba']:.2f}% | "
            f"Clean({res['clean_subset']['rows']})={res['clean_subset']['ba']:.2f}% {res['clean_subset']['ci95']} "
            f"[Mn={rec['Mn']:.1f} Cu={rec['Cu']:.1f} Zn={rec['Zn']:.1f} VIII={rec['Class VIII']:.1f}]"
        )
    print("=" * 130)

    REPO_SUMMARY_JSON.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote summary to {REPO_SUMMARY_JSON} in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
