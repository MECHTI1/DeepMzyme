# CPU preparation evidence, execution record and spending

Prepared 2026-10-14; independently reviewed and re-sealed the same day; executed under the user's 2026-10-15
authorization on 2026-10-15/16. This page records what was verified, what ran, what it cost and where the durable
evidence lives. It grants no further execution authority.

## Verified reusable material

Local preparation root: Data1 `campaigns/metal_completion_2026-10/preparation_20261014T164111Z/` under
`/media/mechti/Data1/DeepMzyme_Data/`. `code/` holds the verified v3 source plus the reviewed completion modules;
`parent/` holds hash-identical frozen campaign inputs, folds, weights and execution settings. The first package
`preparation_20261014/` is superseded history ([log](log.md)); nothing in it was deleted or rewritten. `src/` and
`scripts/` retain source SHA-256 `212db0e3865968701475e7172b2e4cad772e492b35855da293238437fd97c15c`.

`evidence/control_records.json` SHA-256 `5a39f394599a6ec934774182a3b342f01e3311a0a762a7fdecc228f49bf26345`
contains 16 verified late-fusion B controls: all three targets × five folds at seed 42, plus four-class fold 0 at
seed 43. The four-class fold-0 seed-42 control is the step-B FP32 probe run that v3 recorded as that unit's fit in
`parent/execution_settings.json`. Verification used the frozen v3 collector, actual terminal checkpoint/prediction/
replay hashes, exact fold membership, target/probability-collapse validation and recomputed metric reconciliation;
`evidence/host_control_integration.json` records the reviewed host code re-verifying all 16 against bytes, and the
final collections re-verified them again. Reuse saved 15 discovery fits and one confirmation fit.

Discovery manifest `execution/manifest_discovery.json` SHA-256
`d405974eeb4fd0c029922ccfcd01adebef8fd85876c220d92953557753ce98f8` binds the seven executing modules (the six
completion modules and `pmm_v3_assessment.py`, which also equals the assessor hash frozen by the parent extension),
the parent inputs, the specification (`2bc2243d…`) and the 45 prescribed units; `evidence/readiness.json` records
the preparation verification (6,443 structures, four metadata files, 7,664 ESM payload/sidecar pairs). The
confirmation manifest `execution/manifest_confirmation.json` SHA-256
`8342e2ef496cb16e10024d55689639f1b8bfb415f128b239fdebf796fe4edd7b` binds the discovery manifest, the discovery
selection receipt and the 40 confirmation cells.

## Execution record (2026-10-15/16)

Authorization: the user's decision text is recorded verbatim in `execution/authorization/user_decision_2026-10-15.txt`
and bound by `policy_discovery.json` (SHA-256 `319890bc…`, 30 permitted units) and the superseding
`policy_confirmation.json` (SHA-256 `8aae188a…`, 39 permitted units); every worker start checked a fresh controller
session snapshot and gross-spending snapshot against these policies, and the host bound each session in
`execution/authorization_sessions/`. The package was deployed on the GCP L4 host at the identical absolute paths
(bind mounts recorded in `/etc/fstab`); the canonical `plan` passed on the host before every session
(`evidence/vm_plan_*.json`). A host incident (kernel without a buildable NVIDIA driver) and its repair are in
`evidence/host_incident_20261015.json`; the per-session kernel selection is in the playbook block.

Seven controller sessions ran 69 new fits (30 discovery, 39 confirmation); every unit was independently replayed
on the host, pulled, byte-verified and acknowledged by the workstation (`pmm_v3_host_pull.py`), placed under
`execution/lanes/laneK/`, and re-verified by the collector. Three discovery attempts were killed before any GPU
work during the driver incident and rerun once unchanged (`execution/failed_attempts/`); no other rerun was
needed. Per-session event logs are in `evidence/sessions/`; accounting in `evidence/session_N_accounting.json`;
totals in `evidence/campaign_spending_summary.json`; the closeout with every bound hash in
`evidence/execution_closeout.json`.

| Session | Start (UTC) | Running | Gross (USD) | New fits |
|---|---|---|---|---|
| 1 | 2026-10-15 12:35 | 3 h 34 min | 3.14 | 8 |
| 2 | 2026-10-15 16:24 | 3 h 06 min | 2.73 | 12 |
| 3 | 2026-10-15 19:43 | 2 h 33 min | 2.24 | 10 |
| 4 | 2026-10-15 23:52 | 2 h 11 min | 1.93 | 9 |
| 5 | 2026-10-16 04:04 | 2 h 53 min | 2.55 | 12 |
| 6 | 2026-10-16 07:31 | 3 h 27 min | 3.04 | 15 |
| 7 | 2026-10-16 11:11 | 0 h 49 min | 0.73 | 3 |

Totals: 18 h 37 min running, **$16.37 gross running cost** (controller ledger, list price, no credits), seven of
eight sessions, 86,040 of 86,400 authorized GPU-seconds allocated by planned session duration. Retained storage
(disk plus two snapshots, controller-conservative about $1.00/day) is additional and continues while the VM is
stopped. L4 capacity stockouts in `us-central1-a` delayed four of the seven starts (longest: two hours); the
existing VM was retried and no fallback or migration was used. The VM was stopped through the controller after
every session; final state TERMINATED, confirmed 2026-10-16T11:59Z.

## Decisions (frozen rules, validation only)

Discovery (`execution/selection_discovery.json`, SHA-256 `0e048226…`): shared recipe **P**
(population-normalized weighted cross-entropy), qualifying on five-class only (+0.44 common-four points over its
B control; S failed the mean-recall-drop guard on every target). Confirmation
(`execution/selection_confirmation.json`, SHA-256 `9e1fce84…`; seeds 43/44 averaged within fold, five folds,
Bonferroni over five contrasts): **no challenger passed every gate; the unchanged B four-class configuration is
retained.** P4−B4 +0.36 points with adjusted bootstrap [−0.39, +1.42]; P5−B4 +0.10; P6−B4 −0.56 with a Class VIII
recall drop above three points. Mean common-four BA (seed-averaged): B four 73.87, P four 74.24, P five 73.97,
P six 73.31. These are additional optimizer seeds on reused development folds, not independent validation; they
support no improvement claim and no five-versus-six claim, and by themselves no test readiness (Stage 6B source:
[below](#stage-6b-source-and-compatibility-2026-10-16)).

## Stage 6B source and compatibility (2026-10-16)

**Designated Stage 7 source:** the v3 full non-test refit `terminal_model_checkpoint.pt`, SHA-256
`a00078466cf60c93287315883dd0e5565ce3e9e5c6c5c4654d2d5da357e59593`, run
`stage6b_refit__gvp_late_fusion__four_class__baseline__e50__seed42` (Data1 `campaigns/pmm_ion_metal_v3/durable/lane0/runs/`;
the NVMe copy is identical). Authority: the user's 2026-10-16 instruction to reuse a compatible refit rather than
retrain for a campaign name, and the specification's pre-registered `future_refit.baseline_retained` rule. No GPU,
training or test-side read was used. Single model; no ensemble or calibration.

Record `evidence/stage6b_source_20261016/stage6b_source_compatibility.json` (SHA-256 `bd8154cf…`; generator
`stage6b_source_gen.py` `be3af887…`) recomputes 25 checks from bytes; all pass. Three independent CPU verifications
and three adversarial reviews (configuration/environment, data/features, policy), persisted in
`session_reviews_20261016.json` (`a359a969…`), found no mismatch:

- **Identity:** terminal epoch 50, seed 42, 50-epoch cosine, FP32, 807,626 finite parameters. The sibling
  `last_model_checkpoint.pt` (`f899a42d…`) has identical weights but is not the source. Freeze 1, the run status, the
  reload check (CUDA, 7,398 ions, |Δ| 1.1e-16) and the NVMe copy agree.
- **Configuration:** against the five seed-42 B-four fold controls, only the 13 refit keys allowed by the step F
  runner differ (fold/validation inputs, selection metric, the four loss multipliers, run name and identity), plus
  the non-identity output path `runs_dir` for folds 1–4. Loss, weighting mode, optimizer, rates, schedule, batch,
  precision and checkpoint rule are identical across the refit, the six reused controls and the nine completion
  B-four fits. Recipe B equals the v3 baseline.
- **Source and environment:** the trainer tree `212db0e3…` equals the package `code/`, and the runner hashes match.
  All 16 runs record one software/hardware environment. Within `src/` and `scripts/` the repository adds only two EC
  scripts, so v3-gated tooling takes the trainer tree from the package `code/`; the step F tools (for example
  `pmm_v3_test_prep.py`) exist only in the repository and are bound by the step F manifest.
- **Data:** the refit trained on exactly the 7,398-ion non-test cohort `745b19c1…` (the union of the five folds, with
  no validation partition), with weights N/(4 n_c) on that cohort. It used the same feature inventory `942d1d0e…` and
  ESMC-600M directory as the controls, and its normalization was fitted on the full cohort.

Conditions and disclosures (also in the record):

- **Prior held-out use (open conflict):** the weights predate Freeze 1. The checkpoint then received one pre-declared
  secondary pass on the opened exact Zenodo PMM test (2026-10-13). Plan calls Stage 7 "a one-shot held-out test event
  for the fixed final-training run" (section 2; section 6: "only then run the held-out test once"), and the playbook
  calls it "one-shot per final training/refit run". A primary Stage 7 report from this refit therefore conflicts with
  Plan until the user approves an explicit exception or amends Plan, before any scoring. A new refit would be a second,
  non-bit-identical realization of the same recipe trained after that read, not an unexposed model. PMM tests are
  never rerun for this source.
- **Scope of reuse:** reuse holds only for an external cohort disjoint from the training cohort, every opened test and
  their near-copies; a re-split of PMM data would invalidate it.
- **Later campaigns:** the trunk-LR and completion validation campaigns ran after that test read; both retained this
  model's configuration, and neither re-evaluated this checkpoint.
- **Reproducibility and checks not re-run:**
  - `deterministic=false`, so a rerun would not reproduce the checkpoint bit for bit;
  - CPU inference parity is recorded for a fold-1 B-four model (step F readiness D, 1.44e-6), not re-run for this
    checkpoint;
  - edge-feature normalization is consistent with the folds but could not be reconstructed from them;
  - B five/six versus B four rests on the v3 neutral test; the completion campaign re-tested five/six only under
    recipe P.
- **Before any candidate-test read:** freeze the report, reporting set, denominators, the no-ensemble/no-calibration
  rule and the evaluation environment.

## Verification and remaining boundaries

The combined focused suite passes **141 synthetic tests** for the bound modules plus 13 for the workstation
coordinator (`tests/test_metal_completion_session.py`); CPU smoke 42 passed with one absent-local-fixture skip;
docs contract zero strict failures. `evidence/cpu_verification.json` binds code/tests/logs and library versions;
`evidence/independent_review_findings.json` records the 2026-10-14 review and the coordinator-completed checks.
Synthetic success and the executed fits are validation evidence only.

Not done and not authorized: candidate-test retrieval, held-out evaluation and any further GPU session (the
24-hour aggregate ceiling is exhausted). No new refit is needed while the designation's conditions hold
([above](#stage-6b-source-and-compatibility-2026-10-16)). `evidence/evaluation_provenance_template.json` SHA-256
`88b87be6…` lists six source families and is superseded by the current tool's ten-family template
`evidence/evaluation_prep_20261016/provenance_template_blocked.json` (`c896d9d3…`); both stay blocked. The
[evaluation protocol](evaluation.md) is offline tooling, not a certified independent cohort.
