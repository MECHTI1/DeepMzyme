# Job B workstation verification (2026-09-29)

Worktree `/media/mechti/Data1/DeepMzyme_worktrees/docs`, branch `docs-consolidation`
fast-forwarded from `3c0f80c` to the cloud tip `6808084` after the older 31-file staged
draft was saved outside the repository (its patch SHA-256 starts `f8ac2f27`, the same
patch the cloud session received) and dropped. Interpreter:
`/home/mechti/miniconda3/envs/DeepMzyme/bin/python` (checked with `sys.executable`).
Commands ran under the Job A cap (`systemd-run` 2 cores / 3 GB, `nice -n 19`).

| Check (PLAN_v2 §12.6) | Result |
|---|---|
| `source_tree_sha256`, worktree and shared checkout | Both `adc95c42…` (pinned) |
| Protected paths since `3c0f80c` (`src`, `scripts`, `tests`, `notebooks`, `docs/plans`, `docs/notebook_outputs`, `.github`) | No change |
| Shared checkout | HEAD `3c0f80c`, no tracked changes, no untracked files |
| Playbook invariants | Metal: 67 fenced blocks identical to `3c0f80c`, first python fence lines 1514–1624 with the baseline hash, unique `### Exact standalone notebook block` heading (line 1512). EC: 23 blocks identical, first fence 61–168 with the baseline hash. No CR bytes |
| `SHA256SUMS` under `docs/` | Same as the Job A baseline: 10 of 11 pass; `legacy_nonoverlap_test_access` fails for the three known CRLF CSVs. `pmm_replay_diagnostic_20260928/remote_SHA256SUMS` lists VM-side files by design (recorded for Job C/D) |
| `before_hashes.txt` (1,521 entries) | 14 changed, 0 missing; all 14 are documents (Job A and Job B edits), no evidence or plan file |
| `tools/check_docs_contract.py` (Conda) | 188 Markdown files, 794 links; 0 strict failures, 13 warnings (the accepted size caps, six resolved TECH issues kept in full, two duplicated campaign-README paragraphs) |
| `tests/smoke_checks.py` | 43 passed, the same absent-fixture skip; exit 0; same check names and results as the baseline ([output](job_b_workstation_smoke_checks.txt)) |
| Full CPU pytest | 991 passed, 3 failed, 32 errors, 8 skipped, 15 subtests passed ([output](job_b_workstation_pytest.txt)) |

## Pytest comparison with the baseline

The 32 setup errors, the two known failures (`test_outer_loader_not_deserialized`,
`test_dry_run_flag`) and all eight skips are the same as in
[`baseline/pytest.txt`](../baseline/pytest.txt). One extra failure,
`tests/test_train_serial_metal_profile.py::test_profiles_one_call_without_changing_arguments_or_return_value`,
is not caused by Job B: no code changed and the source hash is equal. This run set
`TMPDIR` on the NTFS data drive to spare the 2 GB left on `/`. Re-running that test
file alone gave 1 failure in 3 runs with the default `/tmp` (ext4) and 3 failures in
3 runs with `TMPDIR` on NTFS. The test checks that `prepare_status.json`'s mtime lies
between two `time.time()` readings (`src/train_serial_metal_profile.py:62-67`); file
timestamps come from a coarser clock and can fall slightly before the first reading,
so `setup_seconds` stays `None`. This is a pre-existing flaky check in frozen source,
reported for later code work, not fixed here.

## Not done here

No commit, merge or push. The fresh-agent run 2 acceptance is recorded in
[job_b_fresh_agent_test.md](job_b_fresh_agent_test.md).
