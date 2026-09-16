# DeepMzyme Current Experiment Status

This is the sole concise answer to: **Where am I now, and what should I do
next?** It is mutable. Scientific policy is in [`Plan.md`](Plan.md); exact
experiment history is in the [experiment index](docs/notebook_outputs/README.md).

Last historical experiment-evidence audit: 2026-08-20. Last execution audit: 2026-09-17.
Last scientific-policy documentation update: 2026-09-17.

## Current objective

**Paused by the user after a Colab efficiency audit (2026-09-17):** the owned
G4 session 9 stopped with provider verification at **00:51:07 Asia/Jerusalem**
(21:51:07 UTC on 2026-09-16); the final server listing showed no active sessions.
The other Codex execution is suspended to prevent further allocations. Its
completed first full discovery fit and earlier artifacts remain saved locally;
all thirteen persisted files of that full fit were rechecked against their
recorded SHA256 values. No fit ran in session 9.

The host ledger now totals **8,041.997553 seconds (2.233888 hours)**. Session 9's
**384.788991 seconds** are a closed host-only interval not yet imported into
worker `sessions.json`; reconcile it exactly once before any future launch.
The runtime directory named below contains `PAUSED_HANDOFF.md`,
`USER_REQUESTED_PAUSE_20260917.json`, and the provider-verified session 9 stop
receipt. Reconcile that handoff before resuming the suspended agent, whose
unfinished turn still contains an earlier continuation instruction.

The requested [GPU runtime efficiency implementation
plan](docs/GPU_RUNTIME_EFFICIENCY_PLAN.md) is now written. Its changes are **planned**, not
implemented or measured improvements. This planning request does not authorize
clearing the user pause or allocating another GPU.

**Continuation evidence before the pause (2026-09-17):** the user authorized additional
Colab time to finish the accepted metal campaign. A separately frozen worker
source/state snapshot carries the original scientific grid and every prior
allocation interval forward. Its budget authorization ledger initially records
33 cumulative hours and now records **36 hours** after measuring transfer
overhead; current executable ceilings and later increases are in
`DeepMzyme_Data/notebook_outputs/plans/metal_single_gpu_33h_v2_authorized_runtime_g4/budget_authorizations.json`.
The repaired RING-on probe completed in **106.190 seconds**, and the fresh GVP
anchor completed in **100.590 seconds**. Both are verified timing evidence,
not candidate-selection results. All four confirmation blocks now have timing
coverage. The first 50-epoch discovery fit also completed: compact Only-GVP,
direct-four, LR `1e-5`, seed 42, selected epoch 42, validation BA **65.2188%**.
It took **236.399 seconds** and is individually verified Grade-5 evidence,
not a family ranking. The owned session 8 is provider-verified stopped at
2.127002 cumulative allocation hours; session 9 is restoring the continuation.
The measured save/readback estimate is included in its operations forecast.
No held-out access or promotion is authorized through this campaign.

The continuation runtime, independent artifact copies and host allocation
receipts are in
`DeepMzyme_Data/notebook_outputs/plans/metal_single_gpu_33h_v2_authorized_runtime_g4/`.
Its frozen source includes the explicit budget-authorization controls; their
focused suite passes **168 tests**. The original closed 20-hour forecast below
is historical evidence, not the active authorization ceiling.
See the [continuation summary](docs/notebook_outputs/summaries/summary_metal_single_gpu_authorized_20260917.md)
and [portable evidence](docs/notebook_outputs/raw/metal_single_gpu_authorized_20260917/README.md).

**Original `metal_single_gpu_20h_v2` measured gate closed (2026-09-16):** the paused
campaign resumed on owned G4 allocations and reached its validation-only cost
gate. Twelve of thirteen initial one-epoch probes completed, covering every
planned family/capacity category except RING-on; a fresh same-hardware GVP
anchor also completed after recovery. The RING-on command first failed because
its frozen command omitted the cached RING root. Its one permitted retry was
interrupted by provider session loss and reconciled using provider-verified
death evidence. These are Grade-6 timing probes only; their one-epoch scores
are not model-selection evidence.

The measured decision is **full training not admitted**. Discovery forecasts
to **7.449 hours with the mandatory margin** against the six-hour cap, a
**1.449-hour deficit**. Future operations forecast to **3.194 hours** against
the host-corrected **2.421 hours remaining**, a **0.773-hour deficit**. Ten
grouped-fold/seed RING-on confirmation cells also lack compatible completed
timing. These independent failures mean no 50-epoch discovery or confirmation
fit may start under the frozen campaign. A future attempt requires a new
campaign identity with a revised, documented grid or budget; repairing the
RING path alone is insufficient.

All four profiling-lineage allocation intervals total **5,684.279 seconds
(1.578966 hours)**. Both execution-fix sessions are provider-verified stopped,
and the final server listing reported no active sessions. No held-out access,
Stage 6B/Stage 7 action, scientific ranking or promotion occurred. The local
runtime and independently persisted attempt/state archives are under
`DeepMzyme_Data/notebook_outputs/plans/metal_single_gpu_20h_v2_execfix_23d8ee4_runtime_g4/`.
See the [measured profile summary](docs/notebook_outputs/summaries/summary_metal_single_gpu_20h_v2_profile_20260916.md)
and [portable evidence](docs/notebook_outputs/raw/metal_single_gpu_20h_v2_profile_20260916/README.md).

**Implementation closeout (2026-09-16):** commit `23d8ee4` permits the
controller-owned `execution.log` prelaunch file. The subsequent local fixes
pass an explicit RING cache root to RING commands and prevent an exhausted
operations probe from masking a later allocation's fresh timing anchor; the
measured forecast still fails closed when required timing is absent. The
focused serial campaign suite now passes **165 tests**. Earlier unexecuted
preview directories remain planning artifacts, while the measured gate above
owns the current execution decision.

**Authorized RING continuation completed (2026-09-15):** all four smokes
and sixteen full 50-epoch fits in the
[bounded matched comparison](docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md#bounded-matched-ring-continuation--stage-2b)
are verified locally and in Drive. Both complete family blocks passed their
budget gates. Terminal checkpoint/configuration/normalization binding passed;
the owned G4 session `deepmzyme-metal-ring-20260915` stopped at
**16:20:52.944619 UTC**, and Colab reported no active sessions.
The original cumulative allocation is **7.312911 hours**, leaving
**2.687089 hours of the original ten-hour cap**. The two prior closed
allocations remain unchanged; the budget was not reset.

Only-GVP's mean RING-on-minus-off BA differences are **+1.426 pp** at
`3e-5` and **+0.387 pp** at `1e-4`; all four LR/seed differences are positive,
but mean Class VIII recall falls by **3.125 / 4.688 pp** respectively.
All four graph-level late-fusion pairs have identical selected BA and class
recalls, not proven identical models or predictions. This is **Grade 3** on
one shared validation split and two training seeds, with no promotion.
The audit found annotations on existing edges, no topology expansion and
matching normalization within all trained pairs. The existing local CPU
audit was not restarted; frozen source, previous pilots and Phase-3 analysis
were preserved. No held-out evaluation or auxiliary training was added.
See the [completed RING summary](docs/notebook_outputs/summaries/summary_metal_ring_pilot_20260915.md),
[portable evidence](docs/notebook_outputs/raw/metal_ring_pilot_20260915/README.md)
and [actual stop receipt](docs/notebook_outputs/raw/metal_ring_pilot_20260915/host_closeout_allocation3/session_stopped.json).

**Phase 3 completed locally (2026-09-15):** separate retained-training
PinMyMetal and CARE panels now have native-six/common-four counts, conditional
probabilities, protein-group-weighted measures and 9,999 whole-group
permutations. PinMyMetal contributes 1,163 usable pairs from 1,136 groups;
CARE contributes 983 common-four pairs from 743 groups, or 982 native-six
pairs from 742 groups. Common-four group-weighted Cramér's V is 0.32667 and
0.50600 respectively, with no finite-sample bias correction. These are
descriptive within-source associations, not evidence of auxiliary-learning
benefit. No validation statistics, external test inputs or model training
entered this analysis. Cross-source identity/homology and shared-training
holdout certification remain required. See the
[association summary](docs/notebook_outputs/summaries/summary_metal_ec1_association_20260915.md)
and [verified portable evidence](docs/notebook_outputs/raw/metal_ec1_association_20260915/README.md).

**Authorized metal pilots completed (2026-09-15):** all 30 original architecture
fits and 15 geometry fits, each 50 epochs, plus 12 model smokes are verified
locally and in Drive. Both queues passed genuine terminal-state verification;
all 15 geometry prediction exports are verified. The owned G4 session
`deepmzyme-metal-geometry-20260915` stopped at **11:37:34.517659 UTC**, and
the server reported no active sessions. Total allocation across both sessions
was **5.350957 hours of the original ten-hour cap**, including recovery,
setup, transfer and analysis.

In the original architecture pilot, selected-LR direct-four means ± sample SD
across training seeds 42/43 are
Only-ESM **74.342 ± 3.138%**, late fusion **72.437 ± 0.177%**, Only-GVP
**72.073 ± 1.663%**, and early fusion **65.926 ± 4.866%**. These are Grade-3
results on the same validation data, with no promotion. Late-five's
common-four mean is **74.718 ± 2.486%**, an exploratory target challenger
with class tradeoffs; native and common-four results come from each
native-selected checkpoint. See the [completed continuation report](docs/notebook_outputs/summaries/summary_metal_architecture_pilot_continuation_20260915.md)
and [verified stop receipt](docs/notebook_outputs/raw/metal_architecture_pilot_20260915/continuation/closeout_allocation2/post_stop/session_stopped.json).

All geometry arms selected LR `1e-4`; increasing LR improved seed-42 BA by
7.6–10.4 percentage points. Selected-LR means across training seeds 42/43 are
A 69.641%, B 70.186%, C 70.007%, D 68.865%, and E 68.030%. B−A and C−B
advantages reverse between seeds; D−B and E−D are negative in both high-LR
seeds. This is Grade-3 repetition on the **same validation data**, with no
architecture promotion or universal rejection. E's worst Zn recall is
15.625% (5/32), illustrating the class tradeoffs hidden by aggregate BA.
The fresh A control masks only the added coordination-count/angle slots: it
retains the original GVP geometry and four base metal-site statistics
(multinuclear flag, metal count, minimum/mean intermetal distances). Its
matched explicit machinery differs from the original legacy GVP baseline.
See the [completed geometry summary](docs/notebook_outputs/summaries/summary_metal_coordination_geometry_pilot_20260915.md)
and its verified prediction/paired-error evidence.

Fitted-normalization hashes match within A/B/C and within D/E, as required.
Adding metal nodes changes representation, connectivity and fitted edge
normalization together; those contrasts do not isolate topology alone.

Recovery verified the original frozen source and manifest, all 1,389 retained
pockets (1,181 train / 208 validation), unchanged feature contents and fresh
cache timestamps. Seven original smokes and all eight A1/A2 results are now
verified, including completed late-fusion linked retry `attempt_013`.
The original interrupted `attempt_012` is reconciled as incomplete; its exact
attempt duration is unknown and was not invented. The full first allocation
of 3,827.969 seconds remains charged to the shared 600-minute cap, together
with the second allocation and recovery. The prior two-allocation closed
total was 19,263.446 seconds, leaving 16,736.554 seconds before the RING
continuation; the current three-allocation total is reported above.
The geometry implementation and recovery controls passed 134 focused tests
in 30.37 seconds; 11 separate owned-teardown tests also passed. Legacy model
outputs remain bitwise unchanged in the checked compatibility comparison.
These are implementation/readiness checks, not geometry-model results.
The largest A1/A2 balanced accuracies are late fusion **0.723121** and
Only-ESM **0.721228**, a gap of only **0.001893**. These are experimentally
evaluated single-seed results, with no established architecture winner or
promotion. Hybrid is **deferred under the budget scheduling gate**, not
rejected: best early fusion did not exceed best Only-GVP by the required
margin. The bounded target/seed matrix is complete; full hybrid comparison
and grouped-fold/paired-CI confirmation, including RING, remain absent.
The separately completed bounded RING pilot does not fill those gates.
No held-out evaluation occurred;
historical multi-seed anchors remain separately labeled.
See the [first-allocation summary](docs/notebook_outputs/summaries/summary_metal_architecture_pilot_20260915.md)
and [geometry recipe](docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md#bounded-coordination-geometry-pilot--stage-2b).

**Completed EC reference (2026-09-14):** the v12 EC1 standalone campaign
finished all 12 runs (three families × two LRs × seeds 42/43), each with
30 epochs on the same fixed protein split. Every run was verified
and archived locally and in Drive; the Colab session is stopped. The largest
observed two-seed mean is Only-ESM at `0.0001`,
`0.969643` (sample SD `0.017678`), using
`val_ec_group_level_1_balanced_acc`. This is Grade-3 initial fixed-split
evidence, not promotion. The separate matched zero-auxiliary control and
joint-cohort/holdout certification remain later work. The development-only
association analysis is complete, as documented above; it does not certify
the auxiliary-learning protocol.
No held-out evaluation, auxiliary training or broader HPO was run.
See the [completed campaign summary](docs/notebook_outputs/summaries/summary_ec1_standalone_v12_20260914.md)
for all family/LR results, recalls and exact provenance.

Advance two independent primary missions: direct four-class transition-metal
classification and EC/function classification, beginning at EC depth 1. The
immediate cross-task roadmap starts with reconciling strong standalone models;
auxiliary shared learning remains a later challenger.

The completed Phase 1 pilot has reconciled recipes for the four-class endpoint:
(a) direct four-class training and (b) matched six-class training followed by
deterministic collapsed-four evaluation, plus a labeled five-class challenger.
All three targets completed both LR opportunities at seed 42 and one
native-selected-LR seed-43 repeat across Only-GVP, Only-ESM, and graph-level
late fusion. Direct-four early fusion completed its two-LR screen and repeat;
hybrid was deferred by the conditional scheduling gate. Every pilot arm selects
checkpoints by native validation balanced
accuracy and reports the common-four view from the same checkpoint. Existing
historical anchors use the six-class metal target and one fixed
`pdbid`-grouped validation split. They remain historical evidence and do not
constitute the required matched target-formulation comparison.

The separate metal-architecture mission still includes: (a) early versus late
versus hybrid ESMC fusion, (b) Only-ESM using ESMC versus Only-GVP versus
combined GVP+ESMC, and (c) matched GVP runs with versus without RING edges.
Those advanced comparisons are outside the first metal-EC auxiliary experiment.

No copied candidate has yet passed the current grouped-fold Stage 6 promotion
standard.

## Scientific mission and target status

| Item | Implemented state | Evidence state | Current interpretation |
|---|---|---|---|
| Primary metal reporting endpoint | `four_class` aliases `merge_fe_class_viii`; the bounded native-selected pilot recipe is implemented | A1/A2 screen and selected-LR repeats completed for all four families | Experimentally evaluated across two training seeds on one validation split; no promotion |
| Coordination-geometry comparison | Independent masked/count/angle controls, fixed input shapes, five-arm runner, and shared budget are implemented | All five smokes, 15 full fits, and selected-checkpoint validation export verified | Experimentally evaluated across two training seeds on one validation split; class tradeoffs and inconsistent count/angle gains, no promotion |
| Matched RING comparison | Geometry-fixed shell roles, strict cached-RING controls and whole-cohort input audit are implemented | All four smokes and sixteen full fits completed; selected checkpoints, terminal capture and actual teardown verified | Grade-3 GVP BA gains with class losses; late-fusion selected-metric ties; observed existing-edge annotations only, no promotion |
| Required target-formulation challenger | Native-six eligibility and same-checkpoint native/collapsed-four reports are implemented in the pilot | Three core families × four/five/six targets completed both seed-42 LRs and selected-LR seed-43 repeats | Initial matched fixed-split comparison complete; common-four comparisons remain exploratory, with grouped folds/paired CIs outstanding |
| Historical metal targets | Six-class and five-class schemes remain implemented | Indexed metal anchors are principally historical six-class results; one five-class local smoke was prepare-only | Preserve under their original scheme; never relabel as direct four-class training |
| Primary EC task | Depth-controlled single-label EC classification and structure-level group weighting exist | Three GPU smokes and all twelve CARE EC1 standalone baselines completed and archived | Experimentally evaluated on one fixed protein split across two seeds (Grade 3); no promoted model; deeper hierarchical and full multi-label claims remain unestablished |
| Current EC multi-annotation handling | Multiple annotations with one shared prefix can map to one class; conflicting prefixes at the selected depth receive no target | No full multi-label experiment is indexed | Current code does not solve full multi-label EC prediction |
| Shared-learning topology | A shared pocket representation feeds independent metal and EC heads, and both losses can update shared parameters | Exploratory joint runs exist under unmatched contexts | Implemented and explored; no controlled EC-only versus EC+auxiliary-metal result and no promotion |
| Predicted-metal conditioning | Not part of the inspected shared-head path or a certified primary recipe | No promoted conditioning result | Deferred optional ablation after the simpler auxiliary-loss question |

Status words are literal: a path can be implemented without being smoke-tested,
experimentally evaluated, or promoted. The existence of the joint code path
does not establish that shared learning improves either primary task.

## Where the project is now

- Original architecture-pilot direct-four reference: Only-ESM at `3e-5`
  has that pilot's largest two-seed
  mean BA, `0.743417096220`, sample SD `0.031380141465`; this is a fixed-split
  reference candidate, not a promoted model. Late-five remains a separately
  labeled common-four target challenger. Complete new evidence is in the
  continuation report, distinct from the historical six-class anchors below.
- Historical metal anchor: GVP + graph-level late fusion, namespaced candidate
  `metal/late-fusion/round4/trial49/fixed-split-5seed`.
- Evidence for that anchor: five seeds and 50 epochs on one validation split;
  mean `val_metal_balanced_acc = 0.635468206972`.
- Evidence grade: **3 — fixed validation split across seeds**, not grouped-fold
  confirmation.
- Reproduction status: **historical/non-rerunnable from the current repository**.
  The exact source bundle/checksum and checkpoint binaries are absent; do not
  present the anchor as an executable reproduction package.
- Stable comparison anchor: Only-ESM, LR `3e-5` with inverse-frequency
  weighting; five-seed mean `0.625325230595`.
- The tested node-level late-fusion trial-49-derived configuration did not
  replace graph-level trial 49.
- Hybrid and Hybrid+RING results are exploratory joint/single-split evidence,
  not promoted metal anchors.
- No current Stage 6 grouped-fold result, paired-CI promotion artifact, or
  completed Stage 6B final-refit artifact was found in the inspected evidence.

## Required controlled-comparison coverage

| Required question | Current evidence | What remains |
|---|---|---|
| Target formulation: direct four-class vs six-class training with collapsed-four evaluation | Completed matched initial four/five/six screen and selected-LR repeats for all three core families, using the common-four view | Shared grouped folds/paired CIs and class-recall promotion gates remain outstanding; selected recipes can use different LRs |
| ESMC fusion position: early vs late vs hybrid | Early/late completed direct-four two-LR screens and selected-LR repeats; hybrid smoke passed but full runs were deferred | Hybrid and grouped-fold comparisons remain absent; deferral does not establish that hybrid is ineffective |
| Modality: Only-ESM (ESMC) vs Only-GVP vs combined GVP+ESMC | All three completed direct-four screens and selected-LR repeats on the same validation split | No shared Stage 6 fold/seed comparison with paired confidence intervals and rare-class recall protection |
| GVP edge source: with RING vs without RING | Completed direct-four matched off/on × two LRs × two seeds for Only-GVP and graph-level late fusion; existing-edge annotations and matched normalization | Grade-3 bounded evidence includes GVP class tradeoffs and late-fusion selected-metric ties; shared Stage 6 folds/paired CIs and rare-class promotion gates remain outstanding |

Do not claim a significant advantage for target formulation, ESMC fusion
position, combined modalities, or RING until the applicable Stage 6 paired-CI
and rare-class gates pass; a complete bounded matrix alone is insufficient.
Compare the target-formulation arms on the common four-class view;
keep the other architecture comparisons direct-four. This matrix is separate
from the first EC-primary auxiliary experiment.

Detailed parameters and confidence limits:
[`docs/PARAMETER_FINDINGS.md`](docs/PARAMETER_FINDINGS.md).

## Whole-project workstream map

| Workstream | What is implemented or prepared | Result/evidence state | What is not complete |
|---|---|---|---|
| Unified training core | `metal`, `ec`, and `joint` dispatch; configuration, preflight, grouped splitting, training, reporting, and guarded final-test code | CLI help imports and parses successfully | A passing end-to-end suite on all materialized datasets is not established |
| Graph and feature pipeline | Conservative graph features, independent site-count/angle controls, generic metal nodes and geometry-fixed RING shell roles | Original/recovered readiness, geometry comparison and whole-cohort RING input/cache audits passed | Broader confirmation and additional node feature sets beyond `conservative` remain future work |
| Metal modeling | Original architecture, five-arm geometry and matched RING pilots with one cumulative serial budget | All 30 original, 15 geometry and 16 RING full fits, terminal captures and actual teardown verified | Grouped-fold confirmation and broader required comparisons remain; no architecture promotion |
| EC modeling | Single-label EC-depth handling, independent EC heads, group weighting, and optional contrastive loss exist | All twelve CARE EC1 standalone baselines completed on the fixed protein split | Later EC HPO/confirmation/final-test recipes remain uncertified; full multi-label prediction is not implemented |
| Auxiliary metal+EC modeling | Shared representation, independent heads, configurable task losses, and a joint task path exist | Hybrid and Hybrid+RING exploratory validation evidence exists | No matched EC-only versus EC+auxiliary-metal experiment, no promoted auxiliary configuration, and no certified cross-task protocol |
| Dataset preparation | Exact/non-overlap/Common70 PinMyMetal, CLEAN30 variants, and CARE clusterRes30 are prepared; provenance is tracked; local structures are content-addressed with manifest-backed split membership | Materialization, storage audit, and bundle status are in `docs/DATASETS.md` and `docs/STRUCTURE_STORE.md` | CLEAN10, the harsh root, CARE upstream citation, and final-test route remain unresolved |
| Metal confirmation | Stage 6 grouped folds × seeds, paired CI, and rare-class protection are implemented/documented | No completed current Stage 6 artifact found | Candidate set must be frozen and Stage 6 run |
| Final refit/reporting | Stage 6B and fail-closed Stage 7 workflow exist | Safety behavior is documented and smoke-covered before the current suite failure point | No completed Stage 6B refit; Stage 7 scientifically blocked |
| Colab execution | Unified notebook, main HF bundle, CLI/browser same-VM procedure, and G4/A100 compute probes exist | Stock Colab PyTorch worked on audited G4 and A100 runtimes; a separate PyTorch-free overlay is implemented | Unattended Drive mount remains open; legacy benchmark provenance requires authorized v2 regeneration/reruns |

For the shortest path through these owners, use
[`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md).

## Anchor and challengers

| Role | Configuration | Main validation evidence | Grade | Status |
|---|---|---|---:|---|
| Original architecture-pilot direct-four reference candidate | Only-ESM `3e-5` | mean `0.743417096220`, sample SD `0.031380141465`, two fixed-split training seeds | 3 | Largest direct-four mean in that original pilot; no promotion or cross-campaign maxima ranking |
| New target challenger | Five-class late fusion `3e-5`, common-four reporting | mean `0.747182399055`, sample SD `0.024859222776`, two fixed-split training seeds | 3 | Native-selected recipe; class tradeoffs, no promoted target formulation |
| Historical six-class metal anchor | Late-fusion trial 49 | mean `0.635468206972`, SD `0.043023727308`, five fixed-split seeds | 3 | Preserve as a six-class reference; it is not the direct four-class anchor |
| Historical six-class stable baseline | Only-ESM `3e-5` + inverse-frequency | mean `0.625325230595`, SD `0.031449451169`, five fixed-split seeds | 3 | Preserve within six-class evidence only |
| Rejected tested variant | Node-level late fusion derived from trial 49 | mean `0.606599196822`, five fixed-split seeds | 3 | Did not replace graph-level trial 49 |
| Exploratory challenger | Joint Hybrid trial 17 | three-seed joint mean `0.697376`; different selection metric | 6 | Incomplete provenance; not directly rankable |
| Exploratory challenger | Joint Hybrid+RING trial 114 | single-seed metal BA `0.7303469775006777` | 5 | Not confirmed; RING contribution inconclusive |

Trial 15—not trial 49—has the largest single late-fusion Round-4 result
(`0.6992755722978847`). Trial 49 remains the historical anchor because its
five-seed mean, variance, and worst-seed result led that batch. Neither fact is
grouped-fold promotion evidence.

## Model-family mission coverage

| Model/task path | Implemented | Audited outcome | Standing |
|---|---:|---|---|
| Metal Only-GVP | Yes | Direct-four two-seed mean BA `0.720729`, sample SD `0.016630`, LR `1e-4`; worst class recall `0.343750` | Bounded screen/repeats complete; grouped-fold confirmation pending |
| Metal Only-ESM | Yes | Direct-four two-seed mean BA `0.743417`, sample SD `0.031380`, LR `3e-5`; worst class recall `0.578125` | Reference candidate; historical six-class evidence remains separate |
| Metal GVP + graph-level late fusion | Yes | Direct-four two-seed mean BA `0.724369`, sample SD `0.001765`, LR `1e-4`; worst class recall `0.562500` | Completed including linked A1 retry; relative advantage over ESM changes sign between seeds |
| Metal GVP + early fusion | Yes | Direct-four two-seed mean BA `0.659261`, sample SD `0.048657`, LR `3e-5`; worst class recall `0.187500` | This tested configuration does not justify automatic hybrid escalation; no general architectural rejection |
| Metal GVP + node-level late fusion | Yes | Five-seed mean `0.606599196822` | Tested negative result; rejected as anchor replacement |
| Hybrid fusion | Yes | Joint trial-17 exploratory mean `0.697376`; pilot direct-four smoke passed | Historical joint evidence not directly rankable; pilot full screen deferred under the budget scheduling gate |
| Hybrid + RING | Yes | Trial-114 single-seed metal BA `0.7303469775006777` | Exploratory; RING effect not isolated |
| Cross-attention | Yes | No indexed completed comparison found | Implemented but experimentally unestablished |
| Matched RING/radius-only comparison | Yes | Sixteen full direct-four GVP/late fits; GVP mean BA deltas +1.426/+0.387 pp by LR, with VIII losses; all late selected-metric deltas zero | Bounded Grade-3 evaluation complete; existing-edge annotations only, not universal causal benefit or Stage 6 confirmation |
| EC depth-1 standalone campaign | Yes | All 12 CARE30 v12 runs completed, with matched cohort and two seeds per LR/family | Grade-3 initial fixed-split evidence; no promotion; later EC HPO/final-test recipes remain uncertified |
| EC-primary auxiliary metal comparison | Joint topology exists | No matched controlled result indexed | Planned; recipe and cross-task safeguards not certified |
| Stage 6 grouped-fold confirmation | Yes | No completed current artifact found | Not done |
| Stage 6B full-train refit | Yes | No completed/reused artifact found | Not done |
| Stage 7 final reporting | Guarded implementation exists | No approved current final report | Blocked by Stage 6/6B and dataset-route decisions |

“No indexed result found” is an evidence statement, not proof that no run ever
occurred outside the repository and audited local artifacts.

## Evidence-grade reminder

1. Grouped folds × seeds with paired CI.
2. Grouped folds.
3. Fixed validation split across seeds.
4. HPO discovery on one validation split.
5. Single-seed validation.
6. Exploratory/smoke/partial/incomplete.
7. Superseded historical evidence.

Do not promote a Grade-4/5 Optuna result or a Grade-3 fixed-split anchor as if it
were Grade 1 or 2.

## Dataset readiness

| Dataset | Current readiness |
|---|---|
| Exact PinMyMetal | Present locally and in v12; contains 177 overlapping PDB IDs |
| Non-overlapped PinMyMetal | Present locally/in v12, plus certified train-only PROPKA overlay for the pilot; the shared test was historically evaluated seven times and is not pristine |
| Harsh PinMyMetal | Absent locally and from v10/v11/v12 |
| Common-PDBID 70/30 | Present locally and in v12; custom comparison split |
| CLEAN30 original/conservative | Present; `CLEAN_30_main` points to conservative source |
| CLEAN10 | Not present or documented |
| CARE Task 1 clusterRes30 | Complete audited ESM/external/RING caches locally and in hosted v12 (see DATASETS) |
| CARE legacy base | Scripts/docs remain; distinct legacy output root not found |

Bundle names, hashes, commits, split counts, preparation rules, and provenance:
[`docs/DATASETS.md`](docs/DATASETS.md).

## Colab compute-readiness evidence

The completed session `deepmzyme-metal-geometry-20260915` used G4 RTX PRO
6000 Blackwell, stock PyTorch 2.11.0+cu128, and Python 3.13.15. Cross-session
recovery preserved original manifest
`14010a873801cdbc2d067b8e61137f5469afb4f5d9f833cdd2251fa84080cdd5`
and cohort hash
`60b8af63454883579a2e851fa9a1cc7ddecdc95c53d996240241a099c35bbbe1`.
Fresh cache-timestamp auditing confirmed unchanged feature contents and all
1,389 retained pockets. The old source remains frozen. Geometry source is a
separate implementation. Its source-specific GPU preflight passed in 98.128
seconds and all five geometry smokes passed. Matching normalization hashes
within A/B/C and D/E were verified before admitting the full 15-run comparison,
which is now complete along with selected-checkpoint prediction export. The
same runtime completed all remaining original fits. Genuine finalization and
[post-stop accounting](docs/notebook_outputs/raw/metal_architecture_pilot_20260915/continuation/closeout_allocation2/post_stop/post_stop_closeout.json)
verify both completed campaigns and shutdown within the shared budget.

On 2026-09-15, the metal pilot allocated session
`deepmzyme-metal-pilot-20260915` on a G4 RTX PRO 6000 Blackwell GPU, with
Python 3.13.15 and PyTorch 2.11.0+cu128 including `sm_120`. Its CUDA probe
passed. A first headless-planning failure was fixed with a versioned source-v2
snapshot before any model training. V2 GPU readiness then passed in 124.15
seconds: 1,181/208 train/validation pockets, 1,151/110 PDB groups, no overlap,
and zero-cutoff pooling agreement on all 1,389 pockets. All seven subsequent
one-epoch model smokes and three 50-epoch A1 runs passed and were archived.
The session interruption left A1 late-fusion completion unverified; teardown
was confirmed at 03:56:01.801127 UTC, with no server sessions remaining.
Smoke timings establish operational planning inputs, not an
architecture-quality ranking. The coarse 10-second GPU-memory sampler missed
short training phases and does not establish reliable peak-memory comparisons.
Source/overlay hashes, the
unchanged allocation timestamp, and portable preparation receipts are in the
[pilot summary](docs/notebook_outputs/summaries/summary_metal_architecture_pilot_20260915.md).

On 2026-09-14, CARE cache completion and actual notebook-command GPU training
were verified on Colab G4. All 817 train and 34 test structures have audited
ESM/external/RING files locally. Generation added 586 ESM and 586 external/RING
cache sets, and repaired 265 older external files whose original PROPKA step
had failed. The loader now prefers exact ESM cache names over older aliases.

Only-GVP, Only-ESM, and graph-level late fusion each completed the EC playbook's
one-epoch smoke on stock Python 3.13.15 / PyTorch 2.11.0+cu128. All three used
the same 993/175 training/validation pockets, full feature coverage, all seven
EC1 classes, and zero protein-group overlap. No held-out model evaluation
occurred. This is Grade 6 readiness evidence, not a family ranking or promotion;
all minimum group recalls remain zero after one epoch. The Colab runtime was
stopped. See the [cache/GPU evidence summary](docs/notebook_outputs/summaries/summary_colab_care_cache_smoke_20260914.md).
The completed v12 bundle is now published on Hugging Face and selected by the
local notebook; [DATASETS.md](docs/DATASETS.md) owns its identity and verification.
Publication adds no new training or held-out evaluation evidence.

The subsequent v12 standalone campaign completed all twelve 30-epoch runs
using these same caches and split identities. Its validation evidence is
separate from the smoke and throughput checks above; see the
[completed baseline summary](docs/notebook_outputs/summaries/summary_ec1_standalone_v12_20260914.md).

The following results are throughput checks for one realistic
GVP+ESM-hybrid training step, not validation accuracy, model selection, or an
end-to-end training result. Both use a 240-pocket sample from CARE
clusterRes30 training data, batch size 12, 3 warm-up steps, 20 measured steps,
FP32, and no held-out test data.

| Evidence | Assigned GPU | PyTorch/CUDA | Median step | Throughput | OOM |
|---|---|---|---:|---:|---:|
| [`bench/g4_realistic.json`](bench/g4_realistic.json) | NVIDIA RTX PRO 6000 Blackwell Server Edition, compute 12.0 | `2.11.0+cu128` / CUDA 12.8, includes `sm_120` | `0.0120814975 s` | `993.254 samples/s` | No |
| [`bench/a100_realistic.json`](bench/a100_realistic.json) | NVIDIA A100-SXM4-40GB, compute 8.0 | `2.11.0+cu128` / CUDA 12.8, includes `sm_80` | `0.0473745975 s` | `253.300 samples/s` | No |

The corresponding historical v1 subset is inventoried in
[`docs/DATASETS.md`](docs/DATASETS.md). These are immutable legacy result-schema
files: they lack the current runner's required input-artifact, command, commit,
runner-hash, and full environment fields, and the v1 subset contains a pickled
project class. Re-run the architecture preflight in
[`docs/COLAB_GPU_RUNBOOK.md`](docs/COLAB_GPU_RUNBOOK.md) because stock Colab
versions can change.

## Test-use status

> The non-overlap PinMyMetal test was historically evaluated in seven early
> runs and is therefore not pristine or unopened. Whether those values
> influenced subsequent selection is not established by repository evidence.
> These test metrics must not be used for current HPO recommendations or model
> selection.

The seven exact reports are now tracked under
[`legacy_nonoverlap_test_access/`](docs/notebook_outputs/raw/legacy_nonoverlap_test_access/).
Later copied model-family anchor batches inspected during cleanup are
validation-only and contain no held-out reports.

> **Primary final-test route: unresolved scientific decision required before final reporting.**

This status does not designate a replacement test, substitute exact PinMyMetal,
or change evaluation behavior.

## Safety corrections implemented 2026-08-20

- Primary Stage 7 now hard-requires semantically valid, completed/reused Stage
  6B final-refit provenance; Stage 6 artifacts are not an executable fallback.
- Raw structure-group overlap is blocked before held-out preparation/inference,
  with a second loaded-pocket/group check before graph construction.
- The canonical all-ranked-candidate primary-test workflow is disabled.
- Stage 7 is additionally fail-closed while the primary dataset-route status
  remains scientifically unresolved.
- Strict RING planning now permits configured/available preparation to run and
  enforces required completeness after preparation.

These corrections do not resolve or select the primary final-test dataset.

## Current blockers

1. The original 20-hour profile failed admission. The user has since
   authorized a separately frozen continuation with additional GPU time;
   RING-on timing is now complete. Carry every old allocation interval into
   the authorization ledger and bind revised ceilings to both controllers.
   Account for measured persistence overhead before sustained execution.
   The scientific matrix remains incomplete until its full fits and matched
   grouped-fold comparisons finish; do not reduce their folds, seeds, epochs
   or protection rules to force admission.
2. The original mixed 404/401 connection loss is still under investigation.
   Recovery and ownership-preserving teardown controls are implemented and
   tested. A later profiling allocation also lost its local provider session
   mapping; it was reconciled as interrupted from provider evidence and its
   owned endpoint was stopped. A later `whoami` refreshes OAuth before
   reporting validity, so it cannot certify the pre-failure token state.
   Proxy-token lifetime is a hypothesis, not an established cause. The primary
   final-test route separately remains unresolved.
3. The bounded metal pilot's exact four/five/six recipes and native-selection
   policy are implemented; all seven original smokes and eight A1/A2 full runs
   are verified, and all 15 geometry fits are complete. Original
   target-formulation blocks and all ten selected-LR repeats are also complete.
   The retained historical blocks are not the
   pilot profile; later paired HPO/Stage 6/6B/7 recipes remain to be reconciled.
4. Current promotion policy calls for grouped-fold Stage 6 evidence, but no such
   completed evidence was found for the historical six-class
   anchor/challengers.
5. The required controlled metal-model comparison matrix is incomplete despite
   the completed bounded target/family/seed, geometry and matched RING pilots.
   Full hybrid comparison and shared Stage 6 folds/paired CIs with rare-class
   protection remain absent; fixed-split RING evidence is not promotion.
6. A reportable Stage 6B final-refit artifact was not found.
7. Hybrid Round-1 full configuration/search-space provenance is missing.
8. The EC standalone Stage 0–2B recipe is executable and its twelve baselines
   are completed. Later EC HPO and Stage 6/6B/7 sections still have documented
   incompatibilities and are not certified executable.
9. No exact, certified recipe or matched evidence exists for the first
   EC-primary auxiliary comparison, and a future multi-source implementation
   must enforce cross-task group exclusion before training.

Execution-readiness issues that do not change the scientific next action:

- the smoke path repair and pytest/CPU-CI conversion are implemented locally;
  CI confirmation depends on running the new workflow (`TECH-007`);
- the notebook's live `MOUNT_DRIVE = True` can block unattended CLI notebook
  execution (`TECH-008`);
- a Python 3.12 Linux CPU lock and PyTorch-free Colab overlay now exist; Colab
  hardware validation remains runtime-specific (`TECH-009`).

Open implementation issues:
[`docs/FOLLOW_UP_TECHNICAL_ISSUES.md`](docs/FOLLOW_UP_TECHNICAL_ISSUES.md).

## Immediate next action

Continue the user-authorized runtime described under Current objective.
Persist and independently verify each terminal attempt, account for every
allocated second, and update remaining costs from actual measurements. Bind
each recorded budget increase before it is needed by a launch. Retain the
complete four/six-class discovery arms, shared folds/seeds, rare-class
protection and separate five-class diagnostic identity. The existing user
authorization permits this campaign's GPU continuation; it does not authorize
unrelated HPO, final refits or held-out evaluation.

Preserve the completed
[RING](docs/notebook_outputs/summaries/summary_metal_ring_pilot_20260915.md),
[architecture continuation](docs/notebook_outputs/summaries/summary_metal_architecture_pilot_continuation_20260915.md)
and [geometry](docs/notebook_outputs/summaries/summary_metal_coordination_geometry_pilot_20260915.md)
evidence and every closed allocation interval. Review the matched
LR/seed/class-recall findings and the measured v2 timing as context for a new
frozen comparison. Do not reset any allocation ledger. Reuse a completed fit
only through strict scientific-identity checks; otherwise retain it as
historical evidence.
Keep the original Only-ESM direct-four reference and separately labeled
late-five challenger without ranking unmatched cross-campaign maxima.

Any future target comparison must retain native checkpoint/LR selection and
same-checkpoint probability-collapsed-four reporting. Eventual promotion needs
grouped-fold **Stage 6**, followed by Stage 6B and a scientifically resolved
one-shot Stage 7 route. Full hybrid comparison, grouped-fold confirmation
including RING, have a new bounded recipe but no new execution evidence.
Reportable paired HPO/Stage 6B/Stage 7 recipes remain to be reconciled. The separate
EC-primary auxiliary experiment still requires certified cross-source identity,
homology and cross-task held-out exclusion before a runnable recipe or training.

## Next few actions

1. **Authorized single-GPU continuation — executing:** complete discovery,
   frozen candidate selection and the admitted shared-fold/seed matrix under
   the recorded budget authorizations. The repaired RING timing is complete.
   Preserve all old allocation charges and every failed/interrupted attempt.
   No primary final model is promoted through this exploratory route.
2. **Phase 1 — initial bounded metal pilots complete:** preserve the original
   30 architecture fits, 15 geometry fits and 16 matched RING fits (61 full
   fits total), plus 16 model smokes, geometry prediction export, terminal
   captures and actual closed allocation receipts. Review these fixed-split
   results before another controlled comparison; hybrid remains deferred and
   no model is promoted. RING controls are fresh fits, not reused matrix cells.
3. **Phase 2 — completed at the initial fixed-split level:** retain the twelve
   EC1 standalone baselines for the three initial families. Grouped-fold
   confirmation and final promotion remain future work. After the bounded metal
   campaign, cost EC1 confirmation separately and reconcile its later-stage
   workflow; do not transfer metal-family rankings to EC.
4. **Phase 3 — completed:** retain the predeclared training-only metal × EC1
   association evidence as separate source panels. Review the descriptive
   results; certify cross-source identity and held-out exclusion separately
   before the auxiliary-learning experiment.
5. **Phase 4:** certify, then add and run one controlled EC-primary comparison: EC-only versus
   EC plus auxiliary metal, using independent heads with shared learning and
   cross-task group exclusion.
6. **Phase 5:** test metal-only versus metal plus auxiliary EC only if useful or
   scientifically worthwhile.
7. **Phase 6:** only then consider soft metal conditioning or more complicated
   cross-task interactions. Continue the separate direct four-class metal
   architecture matrix when its standalone gates justify it.

Any candidate intended for final reporting still requires shared-fold/seed
validation, the applicable promotion gates, a frozen full non-test refit, and a
scientifically resolved final-test route before one-shot held-out evaluation.

Exact metal stage blocks remain in
[`docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md`](docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md).
Do not infer exact budgets from this status file.

## Evidence shortcuts

- [Getting started and navigation](docs/GETTING_STARTED.md)
- [Colab browser/CLI GPU runbook](docs/COLAB_GPU_RUNBOOK.md)
- [Experiment index](docs/notebook_outputs/README.md)
- [Parameter/HPO findings](docs/PARAMETER_FINDINGS.md)
- [Dataset and test-use authority](docs/DATASETS.md)
- [Completed matched RING pilot](docs/notebook_outputs/summaries/summary_metal_ring_pilot_20260915.md)
- [Original pilot first-allocation history and provenance](docs/notebook_outputs/summaries/summary_metal_architecture_pilot_20260915.md)
- [Late-fusion Round-4 summary](docs/notebook_outputs/summaries/summary_run_gvp_late_fusion_round4_top3_seedrepeat_50epoch.md)
- [Recovered late-fusion Round-4 JSON evidence](<docs/notebook_outputs/raw/GVP + late fusion/metal_late_fusion_optuna_top3_seedrepeat_50epoch_v1/>)
- [Node-level negative-result summary](docs/notebook_outputs/summaries/summary_run_gvp_node_level_late_fusion_round1_from_latefusion_trial49_seedrepeat_50epoch.md)
- [Hybrid+RING exploratory summary](docs/notebook_outputs/summaries/summary_run_hybrid_ring_round2_optuna_50epoch_wide_v1_trials105_176.md)

## Update rule

After a meaningful batch, update only:

- current objective/stage;
- anchor/challenger state and evidence grade;
- dataset/test readiness;
- blockers;
- immediate and next few actions;
- links to newly indexed evidence.

Put exact batch history in the experiment index, empirical parameter knowledge
in `PARAMETER_FINDINGS.md`, and stable policy in `Plan.md`.
