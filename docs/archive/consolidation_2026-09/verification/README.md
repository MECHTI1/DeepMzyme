# Job A final verification — 2026-09-28

Job A is ready for the user-controlled commit/integration checkpoint. The
post-edit checks reproduce the Phase 0 baseline; this is not an all-green test
suite. No source, training, notebook, protected plan, or experiment-evidence
file was changed. Job B has not started.

## Results

| Check | Result | Evidence |
|---|---|---|
| Frozen source hash, main and documentation worktrees | Both `adc95c42261448dc9d35572a138a3b8349de79124d9708618f511c27a848dd23` | [invariants.json](invariants.json) |
| Protected paths and main checkout | No protected changes/untracked files; main checkout clean; both branch tips at `3e912a6` | [invariants.json](invariants.json) |
| Pre-job hashes | 1,521 entries checked; only the nine intended existing documents differ | [invariants.json](invariants.json) |
| Playbook executable blocks | Both byte-identical to HEAD, LF-only; unique standalone heading precedes first Python fence | [invariants.json](invariants.json) |
| Evidence manifests | 10 of 11 tracked manifests pass; the same three legacy CSV checks fail as before Job A | [invariants.json](invariants.json), [baseline manifest output](../baseline/sha256sums_check.txt) |
| Local test-access ledger | All 36 paths match their SHA-256; 27 distinct hashes and nine copies | [invariants.json](invariants.json) |
| Full CPU suite | 992 passed, 2 failed, 32 errors, 8 skipped, 15 subtests passed; 243.82 seconds; exit 1 | [pytest.txt](pytest.txt), [comparison](pytest_comparison.json) |
| Baseline comparison | Counts, all 34 failure/error headings, and known exception messages match | [pytest_comparison.json](pytest_comparison.json) |
| Standalone smoke | 43 passed, one absent-fixture skip; exit 0; same named checks as baseline | [smoke_checks.txt](smoke_checks.txt) |
| Added links | 22 local links/anchors in changed core-document lines checked; no missing target | [links.json](links.json) |
| Corrected claims | Focused search finds none of the reviewed stale parity, grade or bias-correction phrases | [search result](stale_claim_search.txt) |
| Code/document consumers | Job A target-name scan repeated before final corrections; no file was moved or executable block changed | [consumer scan](../inventory/job_a_final_code_coupling.txt) |

The Phase 0 README's "11 of 12" manifest count is a summary error: its own
per-manifest output lists eleven manifests, of which ten pass. The same three
legacy CSVs fail because their recorded hashes describe CRLF bytes while the
checkout contains LF. Their evidence and manifests remain unchanged.

The two pytest failures concern a stale loader test and an absent ignored
dataset. The 32 setup errors concern a scientific module imported from another
tree during the full suite. See the [baseline explanations](../baseline/README.md).
Job A does not fix these pre-existing code/environment issues.

## Final corrections

- AGENTS retains mandatory persistent Optuna storage on both routes, with
  Drive SQLite specific to Colab and the missing VM-specific recipe explicit.
- The historical pocket-stratified benchmark is Grade 6 in both evidence owners.
- The selected-checkpoint results are not described as a statistical correction
  for selection bias. The collapsed-four epoch-max column and publication
  reconciliation caveat are distinguished from native metrics and recalls.
- The L4 continuation links to its own execution report. Ledger and audit links
  resolve. The fold table follows the metal-example definitions.
- The existing six-question answer key is marked approved according to the
  prior progress record; its required scientific answers are unchanged.

The baseline contains code-coupling and precheck inventories; separate
per-group subagent inventory files claimed in the old progress entry are not
present. This closeout does not claim to reconstruct that earlier activity.

## Reproduction

Commands ran in `/media/mechti/Data1/DeepMzyme_worktrees/docs`, using the
user-specified Conda interpreter after checking its executable. PyCharm reports
a different `.venv` SDK; no environment was changed. Both CPU commands used:

```bash
systemd-run --user --scope --quiet \
  -p CPUQuota=200% -p MemoryMax=3G -p MemorySwapMax=0 \
  env GIT_OPTIONAL_LOCKS=0 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  taskset -c 0,1 nice -n 19 /home/mechti/miniconda3/envs/DeepMzyme/bin/python \
  -m pytest -p no:cacheprovider -q -rs
```

For the second command, replace the final `-m pytest ...` arguments with
`tests/smoke_checks.py`. Standard output/error and exit codes are saved here.
Hash checks read files only, reproduce the frozen source path-plus-bytes
algorithm, compare both first Python fences against HEAD, and run
`sha256sum -c` from each manifest's documented root.

## Integration and handover

1. Commit the staged Job A documents and audit records in `docs-consolidation`.
2. If the campaign branch moved, reassess/rebase; stop on conflicting owned docs.
3. Obtain explicit approval before fast-forwarding the shared checkout. Push
   only with authorization. Neither operation has been performed here.
4. Start a fresh agent session after integration. Read PROGRESS and PLAN_v2.
5. Begin Job B with campaign history preservation and the normative-rule
   inventory; review modified/dropped rules before accepting the rewrites.

Job A moves no files, so MAP/MOVED entries are not applicable. The fresh-agent
acceptance test begins in Job B. The PMM campaign stays paused at nine fold-0
fits, with 36 further fits deferred; this documentation change authorizes no
compute. Frozen execution checkouts retain their existing documentation.
