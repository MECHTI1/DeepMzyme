# Next Claude Code session prompt

Paste the following prompt into a new session. This handoff authorizes independent review and necessary CPU fixes;
it does not authorize a GPU session, spending, candidate-test collection or held-out evaluation.

---

Work in `/home/mechti/PycharmProjects/DeepMzyme`. My priority is to finish metal classification first:
compare all three training objectives (`four_class`, `five_class`, `six_class`), retain a robust final
model using development evidence, and establish a defensible independent final evaluation. EC is deferred.
Review the implemented CPU preparation below and fix concrete defects without repeatedly asking permission for routine reversible fixes.
Use agents for distinct independent questions, with one coordinator and non-overlapping file ownership.

First read `AGENTS.md`, `EXPERIMENT_STATUS.md`, the current campaign README, the relevant sections of
`Plan.md`, and the documentation owner table. Read the Python skill and verify the prescribed conda
interpreter before Python commands. Read the GPU skill before any compute planning or interaction;
reading that skill grants no spending authority. Inspect the worktree and index and preserve existing work.

The active successor is [metal_completion_2026-10](README.md). Read its
[assessment specification](assessment_spec.json), [evidence](evidence.md), [log](log.md) and
[independent-evaluation proposal](evaluation.md), then inspect the actual implementation:
`metal_completion.py`, `metal_completion_campaign.py`, `run_metal_completion.py`,
`metal_completion_assessment.py`, `metal_completion_safety.py`, `metal_completion_evaluation.py`
and their focused tests. Use the
[canonical playbook block](../../METAL_TRAINING_PIPELINE_PLAYBOOK.md#metal-completion-successor-cpu-preparation-and-execution-gates)
for current executable instructions. Inspect real parser/help behavior before proposing a command.
The original Claude transcript and old next-session notes are historical context; newer source and
hash-bound evidence supersede stale recommendations. Do not restart a broad project audit.

Independently review the implemented successor before relying on it. Resolve concrete correctness,
scientific-validity, provenance, persistence and spending blockers, using the smallest safe changes.
Check the following against source and existing test evidence:

1. The exact approved B/P/S recipes use the unchanged frozen v3 trainer and correct manual
   `commonfour_equalized` weights for all three targets. P uses focal gamma zero and smoothing zero,
   giving population-normalized weighted CE; S uses CE with smoothing 0.1. Do not combine focal and
   smoothing or silently substitute generic inverse-frequency weights. Verify actual loss values and
   gradients, parser/config metadata and recipe identities. Native smoothing distributes probability
   mass differently after collapse for different class counts; preserve this disclosure.
2. Discovery is the fixed seed-42 five-fold grid for B/P/S and four/five/six. Reuse each compatible v3
   B control only after its exact source/config/cohort/fold/checkpoint/replay/persistence evidence is
   verified. Unverified records stay unresolved. Do not infer verified counts from expected counts or
   launch blanket v3 reruns. Select one shared recipe with the frozen assessor rule.
3. Confirmation uses that shared recipe for all three targets on seeds 43/44 and all five folds,
   with matched B-four controls, deduplicated if B wins. Bind the complete discovery selection receipt
   before preparing confirmation. Preserve the prespecified common-four contrasts, native recall
   gates, seed averaging within folds, multiplicity handling and direct-four tie/default rule. This
   is additional-seed confirmation on reused development folds, not independent validation. Do not
   choose different recipes per target or add new arms using observed scores.
4. Manifests bind actual executing source, frozen parent inputs, assessment policy, unit identities
   and selection receipts. Successor outputs/caches/claims cannot write into frozen parent paths.
   Missing approval or a stale/incompatible policy, session, spend or ownership receipt must refuse
   execution before training side effects. Verify controller/provider stop deadlines, full-session
   cost reserves, session counts across approved ceiling changes, external worker exit status,
   independent replay and durable collection. A successful plan command is not execution evidence.

Reuse the completed preparation recorded in evidence.md and log.md. Address only unfinished CPU work or concrete findings from the review: repair focused defects, run the smallest
relevant checks, prepare/reuse verified manifests and evidence artifacts, and reconcile coupled docs.
Keep relevant source unchanged during checks and reuse still-applicable checks. Run the required CPU
smoke and documentation contract when applicable. Do not read held-out caches to make tests pass.
Document precisely what is implemented, CPU-tested, reusable, experimentally evaluated or still pending.

Keep the independent-evaluation work offline at this stage. Review `evaluation.md` and
`docs/DATASETS.md` before any test-related decision. The proposed temporal experimental PDB route still
needs evidence-backed complete accounting of prior sources/opened tests, actual initial-release dates,
quality/disposition rules, and a prespecified group-based precision/support requirement. Do not invent
dates, sample sizes or verified declarations, and do not treat a generated query as a certified cohort.
Preserve the frozen v3 context-chain near-copy exclusions, label/feature separation and pretrained-
encoder exposure caveat. Candidate queries, downloads, new test reads, embeddings, predictions and
scoring require their separate reviewed authorization. Historical secondary test results cannot select
the successor model, training recipe, report, ensemble or calibration.

The intended later sequence is discovery, frozen recipe decision, confirmation, selected configuration,
verified reuse or full non-test final refit, frozen primary reporting/checkpoint/ensemble/calibration
rules, then one-shot independent evaluation. Preserve the specification's refit policy; CPU preparation
does not authorize a fit. Do not reopen the closed structural-context/symmetry or trunk-LR investigations
or implement lower-priority architecture sweeps. Keep EC paused; before its inventory tools are reused,
repair and verify [TECH-030](../../FOLLOW_UP_TECHNICAL_ISSUES.md#tech-030--discovered-ec-cache-files-can-bypass-the-training-path-guard).

Before any future paid execution, inspect the existing costed proposal and prepare current gate artifacts using the
installed GPU controller. Reuse any later explicit scoped approval in the conversation; this prompt itself supplies no execution authority.
Preserve the accepted-work escalation rule: if an authorized ceiling later becomes insufficient, offer
the required stop-and-close or continue-with-larger-recorded-ceiling choices at a safe boundary. Do not
silently expand scope, abandon accepted work at a forecast boundary, or exceed recorded authority.

Conclude with completed changes, verification evidence, unresolved blockers and the exact next authorized
step. Link the evidence and canonical commands rather than duplicating recipes. Follow the repository's
explicit-path staging rules while preserving pre-existing staged content; leave commit/push to me.
If launch or test collection remains blocked, make the remaining approval decision concrete and explain
the actual policy/authorization requirement. Do not describe an unexecuted campaign as a better final model.
