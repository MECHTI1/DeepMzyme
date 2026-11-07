# Draft: campaign log entry for amendment 2

**Status: draft for merge (2026-11-04).** The [campaign log](../log.md) owns this entry. This worktree's log ends at
evc-033, and the execution v5 changes already hold unpublished entries evc-034 and evc-035, so the number below is
provisional and is assigned at merge, after the execution entries. The README and design anchors are updated in the
same change. This file is removed after the merge. The entry text follows; the outer fence is not part of it,
and its links are written relative to the log.

~~~~markdown
## evc-036

2026-11-04 — **Amendment 2 decided and drafted before any G2 or fold-0 data; nothing executed.** On 2026-11-04,
with G2 paused at 0/12 completed units, before any CLEAN embedding farm, F6 sequence-mean cache, G2 measurement,
model prediction, score, selection record, or CLEAN inner fit or export existed (the certified views' label-free
ESMC chain-mean caches predate the amendment), the maintainer accepted decisions A2-D1 to A2-D8 for a second amendment of Freeze 1. The
[amendment record](amendment2/amendment_draft.md) owns the decisions, rule text and declared deviations; the
[design](design.md#amendment-2-2026-11-04-before-any-g2-or-fold-0-data) summarises them.

| ID | Decision |
|---|---|
| A2-D1 | S1 selects the best pooled inner score among F2, F3, F4, F5 and F6, with a zero-recall harm guard against F2, a 0.5-point simplicity band and no significance test; controls are never selectable. |
| A2-D2 | Primary metric: accuracy over the proteins of EC classes 1–6; BA over EC classes 1–6 is the first secondary metric; populations, bootstrap, intersection-union and Holm unchanged. |
| A2-D3 | New candidate F6: F5 with its whole-protein branch pretrained on the fold's full official training file and frozen (U3 reversed for this branch only). |
| A2-D4 | S2 is L9 only (no fits), no S3, 30 epochs, raw-distance RBF default for the GVP families, F1 in fold 0 only. |
| A2-D5 | Binding fold-0 futility stop: continue to folds 1–4 only if the pre-declared projected probability of success is at least 0.20. |
| A2-D6 | Execution: Runpod only; G2 and fold 0 in one pod session. |
| A2-D7 | Execution: CLEAN fits and the ESM-1b and ESMC embeddings on the pod GPU with TF32 off. |
| A2-D8 | Execution: a pre-approved fold-0 ceiling and a time-box of at most two further failed capacity windows. |

The pre-data rationale for A2-D2 is a synthetic [power simulation](amendment2/power_simulation.md) that calls the
frozen statistics engine (bit-identical draws on 11 validation units). Under per-class-uniform gains, accuracy over
EC classes 1–6 needed about 0.6–0.65 times the BA margin (80%-power margins over the R1 seed mean, low-accuracy
configuration: 6.4 vs 10.3 points on B1-all, 8.8 vs about 13 on B1-NH30). The frozen BA test was anti-conservative
(any-claim rate 0.068–0.076 against 0.05; bootstrap SE over sampling SD 0.88–0.93 on B1-NH30), while accuracy was
about nominal (0.038–0.048). Gains confined to EC classes 5–6 are detectable only by BA. The numbers depend on the
stated assumptions and cannot change any rule.

Eight interpretations (C1–C8) await the maintainer's confirmation in the decisions file before the record is
built. The amount of the A2-D8 ceiling is recorded with execution v5.

**Fits.** Fold 0 has 117 DeepMzyme fits and folds 1–4 have 101 each (521 in all, against 745–905 under Freeze 1); a
futility stop spends 112. CLEAN keeps its 63 fits and adds 63 in-sample exports and one held-out record.

**Science source (staged on the science branch, not committed).** Twelve existing scientific files, one post-fit
analysis file (`scoring.py`) and six new scientific files are staged with their tests; the
[draft record](amendment2/amendment_draft.json) lists each file's SHA-256 at `c163995`, equal to the composite
Freeze-1 and S1-amendment pin, and of the staged bytes. Recorded for those staged bytes, under the two-core, 3 GB,
zero-swap cap: focused batches 474, 171, 203, 7 and 55 passed; CPU smoke 44 passed and 1 skipped; the full suite
4,127 passed, 63 skipped and 15 subtests passed, plus one source-tree hash failure because the new
`amendment_rules.py` was not yet indexed, which passed after staging. These checks are repeated at the final source
bytes before the record is built.

**Still required before the record is built** (pre-G2 checklist in the amendment record):
- The record builder `amendment2.py` and the `amend-2`, `verify-amendment-chain` and `dry-run-v2` actions, with an
  F6 pretraining-input stub so that F6 commands build statically.
- `pretrain-corpus fasta`, the futility operating-characteristic table, the CLEAN sanity `verify-all` receipt and the
  precision-policy checks in the sanity verifier; the FUT record must require a bound sanity record outside smoke use.
- Scoring reads the CLEAN sanity and chain receipts as JSON without importing those modules, checks the chain and
  FUT continuity, reports the BA correction effect (A2-STAT-7), and writes the bound refusal and decision fixtures.
- C-GF scored from the corpus sequence means (A2-F6-7); a decision on the staged change to
  `src/training/ec_class_metrics.py` (kept in scope or reverted); one rule-ID spelling for the scoring policy; rule
  labels in code docstrings aligned with the record.
- Done (staged): the record builder emits every record field that the general chain verifier reads, the verifier
  (`amendment_chain.py`, science side) is staged with its synthetic and end-to-end tests, and its module docstring
  points to the fixture's `record_shape`; the chain receipt that pods check (`check_chain_receipt`) is part of it.
- The required bindings of the role table (attestation, v1 resolved-configuration digests, futility operating
  characteristics, verification, F6 inputs, CLEAN tie exposure) are written and bound with `--bind ROLE=PATH`.
- Execution v5: `runtime_precision.py`, the comparator CUDA path, the delegated scorer pin line, manifest v5, the CPU
  replay function, the Phase-4 route and the fold-0 order.
- CPU builds before publication: hold-out lists and corpora for folds 0–4, plans v2, the v1 resolved-configuration
  digests at `c163995` and at the dry-run commit, and the five dry-run-v2 runs.

Added when the record is built: the line `Amendment chain link 2: ` with the amendment SHA-256 in backticks
(committed together with `amendment.json` and its rendering), the dry-run-v2 receipt table and the statement that
rule R-CMD restarts as `r_cmd_v2` from the dry-run commit. Freeze 1 (`28077f4c3701820c9aa20f53b570d12f8b70a46514211fc63e62976e393a3c0c`),
Freeze 1b and the S1 amendment are unchanged. The [test-use ledger](../../DATASETS.md#test-use-ledger) records that
fold-0 development will expose the curated test proteins of folds 1–4 and that F6 uses non-metalloenzyme training
proteins. No GPU work, training, embedding or test access occurred; root STATUS stays unchanged under C-10.
~~~~
