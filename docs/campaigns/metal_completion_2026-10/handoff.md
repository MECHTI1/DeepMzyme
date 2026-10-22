# Continuation checklist

This checklist covers the next working session on this campaign. It authorizes reading and CPU work only. It does
not authorize a GPU session, spending, training, candidate-test collection, a test-side read or held-out evaluation.
Optimization is complete ([log](log.md)) and the Stage 6B source is designated
([evidence](evidence.md#stage-6b-source-and-compatibility-2026-10-16)); do not rerun either.

Work in `<repo-root>`. The maintainer's priority is the independent evaluation of the selected metal model. The
completion campaign retained the unchanged B four-class configuration (the v3 late-fusion direct-four recipe), and the
v3 full non-test refit `a0007846…` is its verified, designated Stage 7 source. Keep its features and architecture
unchanged: no atom-level encoder and no extra surrounding-atom inputs. EC stays deferred.

First read the maintainer's local development guidelines (not published), `EXPERIMENT_STATUS.md`, the campaign
README, `evaluation.md`, the relevant `Plan.md` sections, `docs/DATASETS.md` and the documentation owner table. Verify
the prescribed project interpreter before Python commands. Read the maintainer's private GPU-use policy (not
published) before any compute planning; reading it grants no spending authority. Inspect the worktree and index.

Evidence lives on the secondary drive in `DeepMzyme_Data/campaigns/metal_completion_2026-10/preparation_20261014T164111Z/evidence/`:
- `execution_closeout.json`: the campaign fits;
- `stage6b_source_20261016/`: the compatibility record, 25 of 25 checks passed;
- `evaluation_prep_20261016/`: training-side prior-use release dates, obsolete entries and the blocked template.

Reuse this evidence. Repeat a check only when source, inputs or environment changed. The completion package has no
NVMe or off-site copy yet (backup state is recorded privately, outside this repository).

Work through the [remaining blockers](evaluation.md#remaining-blockers) in order. Act only on what the maintainer
explicitly approves; this checklist implies no approval:

1. **Route and one-shot rule (blocker 1):** the maintainer decides whether to confirm an external temporal PDB cohort
   as the primary route, disjoint from the training cohort, every opened test and their near-copies, and whether to
   approve an explicit exception to Plan's one-shot Stage 7 rule (or a Plan amendment) for this refit, which already
   had one secondary held-out pass. The source designation depends on both.
2. **Prior-use completion (blocker 2, with the maintainer's approval):** carry out exactly what blocker 2 lists,
   including its record allowlist and its separate network request, and record the read in DATASETS.
3. **Later blockers:** each needs its own approval, in the listed order.

Closure stays the maintainer's decision; the campaign remains open while Stage 7 preparation continues.

Host note: the GCP VM boots kernel 7.0.0-1014-gcp, for which the NVIDIA driver does not build; the playbook block
records the per-boot kernel selection.

Do not reopen the closed structural-context, symmetry or trunk-LR investigations. Keep EC paused until TECH-030 is
repaired and verified.

End the session with a record of the completed changes, verification evidence, unresolved blockers and the exact next
authorized step. Stage changes by explicit path, following the staging rules in the local development guidelines;
commit and push stay with the maintainer.
