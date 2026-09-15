# Completed EC1 standalone v12 baseline evidence

All twelve authorized runs completed 30 epochs. See the
[campaign summary](../../summaries/summary_ec1_standalone_v12_20260914.md).

`campaign_plan.json`, `commands.txt` and `run_matrix.csv` describe the exact
matrix. `readiness.json` and `expected_split.json` record CUDA and cohort
checks. `per_run_validation.csv`, `family_lr_validation.csv`,
`validation_results.json` and `report_tables.md` contain selected-checkpoint
validation results. `verification.json` checks all runs' selection, recalls,
shared split identities and persistence receipts.

`artifact_manifest.json` hashes original completed output bytes, including
checkpoint binaries kept outside git. Git may normalize copied CSV line
endings; archive checksums refer to the original archived bytes.
`transfer_receipts/` records local archive SHA256 checks and Drive metadata
and size checks. Full checkpoints and the exact source snapshot are in local
persistent storage and the [Drive campaign folder](https://drive.google.com/drive/folders/1hhfLcjlTSA4i8VBY3LxCP6yCpycvHy9y).

`resume_receipt.json` records recovery of completed run 4 without repetition.
`shutdown_receipt.json` confirms the campaign runtime was stopped after
verification. These completed records supersede the earlier interruption
state. No held-out inference or metrics were produced, and no model is
promoted. The metal campaign and all auxiliary training remain pending.
