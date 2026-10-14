# Next Claude Code session prompt

Paste the following prompt into a new session. It authorizes reading, CPU checks and the preparation of gate
artifacts; it does not authorize a GPU session, spending, candidate-test collection or held-out evaluation. The
2026-10-14 independent review is complete ([log](log.md), [evidence](evidence.md)); do not repeat it.

---

Work in `/home/mechti/PycharmProjects/DeepMzyme`. My priority is still to finish metal classification first:
compare `four_class`, `five_class` and `six_class`, retain a robust final model using development evidence,
and establish a defensible independent final evaluation. EC stays deferred.

First read `AGENTS.md`, `EXPERIMENT_STATUS.md`, the campaign README, the relevant `Plan.md` sections and the
documentation owner table. Verify the prescribed conda interpreter before Python commands. Read the GPU skill
before any compute planning or controller interaction; reading it grants no spending authority. Inspect the
worktree and index and preserve existing work.

The active successor is [metal_completion_2026-10](README.md). Its CPU preparation was implemented, independently
reviewed and re-sealed; `evidence.md` owns the package root, manifest hash and verified controls, and the
[canonical playbook block](../../METAL_TRAINING_PIPELINE_PLAYBOOK.md#metal-completion-successor-cpu-preparation-and-execution-gates)
owns the executable commands, the wrapper/exit-file contract, the persistence layout and the deployment
prerequisite. Reuse that evidence; repeat a check only when source, inputs or environment changed.

Decide with me, in this order, and act only on what I explicitly approve in this conversation:

1. The costed validation-only proposal in `evidence.md` (24 aggregate GPU hours, $30 gross, eight sessions of at
   most four hours, three FP32 lanes on the existing L4). If I approve it, record my exact decision text, bind it in
   the authorization policy with the manifest hash and the permitted units, and prepare the controller session and
   spending snapshots through `~/deepmzyme-vm/bin` only after I ask for a start. Fresh price verification and the
   provider automatic STOP are mandatory. The storage line is about $1.00/day while the VM is TERMINATED.
2. The deployment prerequisite: the GPU host must hold the package, parent, train dataset and ESM directory at the
   identical absolute Data1 paths; run the `plan` command there before any unit. If that is not feasible, propose a
   reviewed relocation mechanism instead of improvising paths.
3. Discovery (30 new fits), the frozen recipe decision, confirmation (29 or 39 new fits), the selected configuration,
   then verified reuse or a full non-test refit, frozen reporting/checkpoint/ensemble/calibration rules, and only
   then one-shot independent evaluation. Preserve the specification's refit policy. Historical secondary test
   results cannot select anything.

Keep the independent-evaluation work offline: `evaluation.md` and `docs/DATASETS.md` own its gates; candidate
queries, downloads, new test reads, embeddings, predictions and scoring each need their own reviewed authorization.
Do not reopen the closed structural-context, symmetry or trunk-LR investigations, and keep EC paused until TECH-030
is repaired and verified.

Preserve the accepted-work escalation rule: if an authorized ceiling becomes insufficient, offer the stop-and-close
or continue-with-larger-recorded-ceiling choices at a safe boundary. Do not silently expand scope or exceed recorded
authority. Conclude with completed changes, verification evidence, unresolved blockers and the exact next authorized
step, linking evidence and canonical commands. Follow the explicit-path staging rules; leave commit/push to me.
