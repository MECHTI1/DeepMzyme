# Partial EC1 standalone baseline evidence

See the [campaign summary](../../summaries/summary_ec1_standalone_v12_20260914.md).

This folder contains three completed, verified 30-epoch Only-GVP runs from
the authorized twelve-run v12 campaign. `validation_results.json` contains
selected-checkpoint scores and group-level per-class recalls. The remaining
matrix is planned, not measured: run 4 was launched but became unverified
after Colab connection loss, and runs 5–12 were not launched.

`campaign_plan.json`, `commands.txt`, and `run_matrix.csv` describe the full
authorized matrix. Their existence is not proof of completed execution.
`readiness.json` and `expected_split.json` record CUDA and shared-cohort
checks. `artifact_manifest.json` hashes completed outputs, including
checkpoint binaries kept outside git. `transfer_receipts/` records local
SHA256 verification and Drive metadata/size verification for runs 1–3.
`recovery_status.json` records the unresolved runtime state.

Full checkpoints and the exact source snapshot are in local persistent
storage and the [Drive campaign folder](https://drive.google.com/drive/folders/1hhfLcjlTSA4i8VBY3LxCP6yCpycvHy9y).
No held-out inference or metrics were produced. No model is promoted.
