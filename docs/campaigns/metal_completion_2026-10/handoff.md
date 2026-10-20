# Next Claude Code session prompt

Paste the following prompt into a new session. It authorizes reading and CPU work only. It does not authorize a
GPU session, spending, training, candidate-test collection, a test-side read or held-out evaluation. Optimization is
complete ([log](log.md)) and the Stage 6B source is designated
([evidence](evidence.md#stage-6b-source-and-compatibility-2026-10-16)); do not rerun either.

---

Work in `/home/mechti/PycharmProjects/DeepMzyme`. My priority is the independent evaluation of the selected metal
model. The completion campaign retained the unchanged B four-class configuration (the v3 late-fusion direct-four
recipe), and the v3 full non-test refit `a0007846…` is its verified, designated Stage 7 source. Keep its features
and architecture unchanged: no atom-level encoder and no extra surrounding-atom inputs. EC stays deferred.

First read `AGENTS.md`, `EXPERIMENT_STATUS.md`, the campaign README, `evaluation.md`, the relevant `Plan.md`
sections, `docs/DATASETS.md` and the documentation owner table. Verify the prescribed conda interpreter before
Python commands. Read the GPU skill before any compute planning; reading it grants no spending authority. Inspect
the worktree and index.

Evidence lives in Data1 `campaigns/metal_completion_2026-10/preparation_20261014T164111Z/evidence/`:
- `execution_closeout.json`: the campaign fits;
- `stage6b_source_20261016/`: the compatibility record, 25 of 25 checks passed;
- `evaluation_prep_20261016/`: training-side prior-use release dates, obsolete entries and the blocked template.

Reuse this evidence. Repeat a check only when source, inputs or environment changed. The completion package has no
NVMe or off-site copy yet (backup state is recorded privately, outside this repository).

Work through the [remaining blockers](evaluation.md#remaining-blockers) in order. Act only on what I explicitly
approve in this conversation:

1. **Route and one-shot rule (blocker 1):** ask me to confirm an external temporal PDB cohort as the primary
   route, disjoint from the training cohort, every opened test and their near-copies. Also ask whether I approve an
   explicit exception to Plan's one-shot Stage 7 rule (or a Plan amendment) for this refit, which already had one
   secondary held-out pass. The source designation depends on both.
2. **Prior-use completion (blocker 2, with my approval):** carry out exactly what blocker 2 lists, including its
   record allowlist and its separate network request, and record the read in DATASETS.
3. **Later blockers:** each needs its own approval, in the listed order.

Closure stays my decision; the campaign remains open while Stage 7 preparation continues.

Host note: the GCP VM boots kernel 7.0.0-1014-gcp, for which the NVIDIA driver does not build; the playbook block
records the per-boot kernel selection.

Do not reopen the closed structural-context, symmetry or trunk-LR investigations. Keep EC paused until TECH-030 is
repaired and verified.

Conclude with the completed changes, verification evidence, unresolved blockers and the exact next authorized step.
Follow the explicit-path staging rules and leave commit/push to me.
