# DeepMzyme Experiment Leaderboard

Primary selection metric: `val_metal_balanced_acc` (validation only — held-out test not yet evaluated for any run).

Tiers reflect reliability: **A** = 5-seed 50-epoch seed-repeat, **B** = 3-seed or 30-epoch seed-repeat, **C** = single-seed HPO / partial seeds.

---

## Tier A — 5-seed, 50-epoch seed-repeat (most reliable)

| Model | Round | Best single val | Best seed | Mean val (5 seeds) | Summary |
|---|---|---|---|---|---|
| GVP + late fusion | Round 4 | **0.6880** (seed 2026) | trial49 seed2026 | 0.6354 | [summary](summary_run_gvp_late_fusion_round4_top3_seedrepeat_50epoch.md) |
| GVP + late fusion | Round 1 (full coverage) | 0.6818 | trial12 gvp3 | not reported | [summary](summary_run_gvp_late_fusion_round1_full_coverage.md) |
| GVP + late fusion | Round 1 (trial12 anchor) | 0.6818 | trial12 anchor | not reported | [summary](summary_run_gvp_late_fusion_round1_trial12_anchor.md) |
| Only-ESM | Round 1 (full coverage) | 0.6722 | — | not reported | [summary](summary_run_only_esm_round1_full_coverage.md) |
| Only-ESM | Round 1 (anchor comparison) | 0.6722 | — | not reported | [summary](summary_run_only_esm_round1_anchor_comparison.md) |
| GVP + late fusion | Round 2 (ESM anchor) | 0.6775 | — | not reported | [summary](summary_run_gvp_late_fusion_round2_confirmed_esm_anchor.md) |
| Only-GVP | Round 6 (three-trial comparison) | 0.6559 | trial7 gvp4 | not reported | [summary](summary_run_only_gvp_round6_three_trial_comparison.md) |
| GVP + node-level late fusion | Round 1 | 0.6332 (seed 2026) | trial49 seed2026 | 0.6066 | [summary](summary_run_gvp_node_level_late_fusion_round1_from_latefusion_trial49_seedrepeat_50epoch.md) |

---

## Tier B — 3-seed or 30-epoch seed-repeat (moderate reliability)

| Model | Round | Best single val | Seeds | Epochs | Summary |
|---|---|---|---|---|---|
| Only-ESM | Round 2 (lr/wd/weight screen) | **0.6930** | 3 (42,123,2026) | ~44 | [summary](summary_run_only_esm_round2_lr_wd_weight_screen.md) |
| Only-GVP | Round 3 (top Optuna confirm) | 0.6559 | 5 | 30 | [summary](summary_run_only_gvp_round3_top_optuna_confirm.md) |
| Only-GVP | Round 4 (top3 + gvp3) | 0.6240 | unclear | 30 | [summary](summary_run_only_gvp_round4_top3_plus_gvp3.md) |
| Only-GVP | Round 5 (trial13 batch) | 0.6316 | unclear | 30 | [summary](summary_run_only_gvp_round5_trial13_batch.md) |
| Only-GVP | Round 5 (trial12 batch) | 0.6216 | 5 | 30 | [summary](summary_run_only_gvp_round5_trial12_batch.md) |
| Only-GVP | Round 2 (Optuna seed-repeat) | 0.6477 | 3 (42,123,2026) | — | [summary](summary_run_only_gvp_round2_optuna_seed_repeat.md) |
| Only-ESM | Round 3 (seed confirmation) | 0.6184 | 2 (43,44 only) | — | [summary](summary_run_only_esm_round3_seed_confirmation.md) |

---

## Tier C — Single-seed HPO or partial evidence (exploratory only)

| Model | Round | Best trial val | Metric | Notes | Summary |
|---|---|---|---|---|---|
| GVP + late fusion | Round 3 (Optuna 50 trials) | 0.6750 | val_metal_balanced_acc | trial49, seed 42 HPO only | [summary](summary_run_gvp_late_fusion_round3_optuna_50_v1.md) |
| GVP + ESM hybrid | Round 1 (Optuna + 3-seed repeat) | 0.7483 | **val_joint_balanced_acc** | joint metal+EC task; metal only = 0.6721; 3 seeds; debug_smoke batch warning | [summary](summary_run_hybrid_round1_optuna_plus_top3_seedrepeat.md) |
| Only-GVP | Round 1 (Optuna HPO) | 0.5543 | val_metal_balanced_acc | seed 42 only, in-memory Optuna | [summary](summary_run_only_gvp_round1_optuna_hpo.md) |

---

## Key takeaways

- **Best metal-only single seed**: GVP + late fusion Round 4, trial49 seed2026 → `0.6880`
- **Best metal-only 5-seed mean**: GVP + late fusion Round 4, trial49 → `0.6354` (only Tier A run with mean reported)
- **Best joint (metal + EC)**: Hybrid Round 1, trial17 → `val_joint_balanced_acc = 0.7483` (Tier C, needs 5-seed 50-epoch confirmation)
- **No held-out test metrics exist yet** for any model family
- Only-ESM Round 2's 0.6930 (Tier B) warrants a 5-seed 50-epoch confirmation before comparing against Tier A GVP + late fusion results

---

*Last updated: 2026-05-16. Source: all files in `summaries/`. See `EXPERIMENT_STATUS.md` for current recommended next step.*
