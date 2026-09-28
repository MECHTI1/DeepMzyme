# Job A pre-checks (PLAN_v2 Appendix B, B0), 2026-09-28

All read-only. Worktree @ 3e912a6 unless stated.

| # | Check | Result | Consequence for Job A |
|---|---|---|---|
| B0.1 | Inbound links to `Plan.md#canonical-colab-metal-training-pipeline` or a G4-policy anchor (`rg --no-ignore`, whole tree) | None | The Plan heading may be renamed without an anchor shim |
| B0.2 | Parity/hype claims in active docs | `EXPERIMENT_STATUS.md:399` ("Full replication of … protocol"); `docs/EXACT_PINMYMETAL_5FOLD_CV_REPRODUCIBILITY.md:25` ("identical dataset, identical 5-fold stratification…", "beating"), `:30` ("Massive Breakthrough"). The playbook's "beats" lines (4335, 4435, 4538, 4649, 5138) are promotion rules, not claims | Edit only these three places |
| B0.3 | "Colab recommended" wording | Only `docs/GETTING_STARTED.md:26` ("Recommended cloud entry point"). `Plan.md:920` "Recommended default for cloud use" is about the data-input mode (`huggingface_link`), not compute | Edit GETTING_STARTED:26; leave Plan:920 |
| B0.4 | VM-route Optuna storage in `docs/GCP_GPU_RUNBOOK.md` or the skill | Not documented (only VM disk/storage-cost text) | The D1 edits keep "persistent Optuna storage is mandatory for Stage 4/5", scope Drive SQLite to the Colab route, and state that no VM-route recipe is documented. No new VM rule is invented |
| B0.5 | PMM ion campaign folds | `docs/plans/metal_level_metal_task_compared_PMM_final_plan.md:241` "Five PDB-grouped folds; `split_seed=42`, `split_stratify_by=metal_site`"; `:259` one frozen `fold_membership.csv`; `:239` `metal_example_unit=ion`; `docs/plans/pmm_core_scope_v2.json` `folds=[0,1,2,3,4]` | Fold-table row "Frozen PDB-grouped 5-fold" confirmed |
| B0.6 | Exact test vs non-overlap test (shared checkout data, read-only) | All 316 exact-test `.pdb` files match the non-overlap test manifest's sha256 (316/316); the catalytic summary CSV is byte-identical | The ledger may state that the exact test is the same structure set as the historically opened non-overlap test |
| B0.7 | gitleaks | Not installed (user confirmed) | Fallback scan used: `baseline/secret_scan.txt` |

Test-access artifacts for the ledger: [`test_access_artifacts.tsv`](test_access_artifacts.tsv)
(36 rows, 27 unique files; 9 byte-identical copies are marked in `duplicate_of`).
