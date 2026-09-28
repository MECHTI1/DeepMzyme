# Phase 0 baseline (worktree @ 3e912a6, 2026-09-28)

Worktree: `/media/mechti/Data1/DeepMzyme_worktrees/docs`, branch `docs-consolidation`
from `metal-pmm-ion-campaign` @ `3e912a6`. `git status --porcelain` was empty right
after checkout (NTFS, ntfs3). The shared checkout was not modified.

| File | Result |
|---|---|
| [`code_invariants.txt`](code_invariants.txt) | `source_tree_sha256` = pinned `adc95c42…`. Metal playbook: first python fence L1514, heading `### Exact standalone notebook block` L1512 (unique). EC playbook: fence L61, heading L59. Both LF only. Block hashes recorded |
| [`sha256sums_check.txt`](sha256sums_check.txt) | 11 of 12 tracked `SHA256SUMS` pass. `legacy_nonoverlap_test_access` fails for 3 CSVs: **pre-existing**, see below |
| [`pytest.txt`](pytest.txt) | 992 passed, 2 failed, 32 errors, 8 skipped, 15 subtests passed (235 s, capped 2 cores / 3 GB) |
| [`pytest_isolation_pmm_core.txt`](pytest_isolation_pmm_core.txt) | `tests/test_pmm_core_assessment.py` alone: 40 passed |
| [`smoke_checks.txt`](smoke_checks.txt) | 43 passed, 1 skipped (local multi-metal fixture absent); exit 0 |
| [`secret_scan.txt`](secret_scan.txt) | Credential-format scan (gitleaks not installed): 0 hits in 347 commits (`git log --all -p`) and 2,042 tracked files |
| [`fresh_agent_answers.md`](fresh_agent_answers.md) | Baseline fresh-agent answers from today's default read path |

## Pytest baseline: failures and errors

None is caused by this work; the docs branch had no changes when the suite ran.
Later runs are compared with this baseline in the worktree only.

| Test | Cause (evidence) |
|---|---|
| 32 setup errors in `tests/test_pmm_core_assessment.py` (`test_refit_drift_and_wrong_checkpoint_block`, `test_oof_and_pmm_coverage_required`, `test_collection_*`, `test_bootstrap_uses_paired_folds_and_fixed_seed`) | `Scientific module already loaded from another tree: data_structures`. The same file passes alone (40 passed), so the errors depend on test order in the full suite |
| `test_outer_loader_not_deserialized` | `training.data` has no attribute `load_structure_pockets`. Known stale test (fails on `main`) |
| `test_dry_run_flag` | `FileNotFoundError`: dataset `train_and_test_sets_structures_exact_pinmymetal` is absent in the worktree (local data is git-ignored) |

The CI status of the campaign branch was not checked (`gh` is not installed).

## Pre-existing evidence-manifest mismatch

`docs/notebook_outputs/raw/legacy_nonoverlap_test_access/{diagnostic_existing_test_reports,sweep_comparison,sweep_status}.csv`
were committed in `3846f47` (2026-08-22) while `~/.gitconfig` set `core.autocrlf=input`, so
git stored them with LF. `SHA256SUMS` records the CRLF bytes. The shared checkout's working
copies still have CRLF and pass; every fresh checkout fails. That directory is not
protected by `.gitattributes -text`. Evidence is immutable in this work: report only.
