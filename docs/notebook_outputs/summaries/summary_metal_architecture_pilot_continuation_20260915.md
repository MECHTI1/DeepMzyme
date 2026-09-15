# Metal architecture pilot continuation — 2026-09-15

Experiment ID: `metal/nonoverlap/architecture-pilot-continuation/2026-09-15`.
Profile: `metal_architecture_pilot_10h_v1`.

## Outcome and scope

**All 30 scheduled original full 50-epoch fits and seven smokes completed,
were archived, and passed final verification. The GPU is stopped.** The
campaign contains eight A1/A2 architecture-screen fits, twelve T1/T2 target
fits, nine core-family seed repeats, and one early-fusion repeat. Hybrid was
deferred by its prespecified exploration-priority gate.

The selected-LR repeats are **Grade 3: two training seeds on one validation
split**; the initial LR screen is Grade 5. Direct-four mean BA is 74.342%
for Only-ESM, 72.437% for late fusion, 72.073% for Only-GVP, and 65.926%
for early fusion. These are bounded recipe results, with class tradeoffs and
no architecture or target-formulation promotion. No held-out inference or
metrics were used. The [first-allocation summary](summary_metal_architecture_pilot_20260915.md)
retains its historical three-full-run scope and verified teardown. The
separate [coordination-geometry summary](summary_metal_coordination_geometry_pilot_20260915.md)
owns that experiment's 15 full runs; they are not included in this campaign's
30-run count. Across both profiles, 45 full fits and 12 smokes completed.

The shared second session stopped at **11:37:34.517659 UTC on 2026-09-15**,
and the server reported no active sessions. Both allocations together used
**19,263.446 seconds = 5.350957 hours**, including setup, recovery, transfers
and analysis, within the original ten-hour cap. Final verification and
post-stop evidence are linked below.

Final evidence:
[selected results](../raw/metal_architecture_pilot_20260915/continuation/finalization/validation_results.json),
[coverage](../raw/metal_architecture_pilot_20260915/continuation/finalization/campaign_coverage.json),
[final run provenance](../raw/metal_architecture_pilot_20260915/continuation/finalization/final_run_provenance.json).
Earlier [attempt-021](../raw/metal_architecture_pilot_20260915/continuation/snapshots/attempt_021/)
and [attempt-029](../raw/metal_architecture_pilot_20260915/continuation/snapshots/attempt_029/)
snapshots preserve historical 12- and 20-run states; neither is the final state.

## Recovery and common comparison identity

The continuation revalidated the original frozen source/manifest, all 1,389
training/validation pockets, feature content hashes, and cache timestamps.
At recovery, seven smokes and three full fits were verified. Interrupted
`attempt_012` remains incomplete with unavailable exact attempt timing;
`attempt_013` is its completed linked late-fusion retry. The complete first
allocation's 3,827.969071 seconds remain charged to the shared original budget.
Recovery does not reset that budget or turn the interrupted fit into a result.
See [recovery readiness](../raw/metal_architecture_pilot_20260915/continuation/cross_session_recovery/readiness.json)
and [reconciliation](../raw/metal_architecture_pilot_20260915/continuation/cross_session_recovery/reconciliation.json).

Every row uses the same non-overlap PinMyMetal training membership, internal
`pdbid` split seed 42, validation fraction 0.15, and native-six-eligible
metal-site cohort: 1,181 training pockets in 1,151 PDB groups and 208 validation
pockets in 110 PDB groups, with no group overlap. Initial screens use model
seed 42; selected-LR repeats use seed 43 on the same validation data.
Native validation supports are Mn 97, Cu 15, Zn 32, Fe 44,
Co 13, and Ni 7. The four-class endpoint has Class VIII = Fe+Co+Ni, support 64.
PDB grouping alone does not certify separation by sequence homology; no such
confirmation was performed in this pilot.

The frozen recipe uses conservative features and the certified PROPKA overlay,
frozen ESMC embeddings where applicable, 10 Å pocket extraction, 6 Å radius
edges, pooling cutoff zero over all residue nodes, no explicit metal nodes,
no RING, and no augmentation. Exact configuration remains in the
[playbook](../../METAL_TRAINING_PIPELINE_PLAYBOOK.md#bounded-metal-architecture-pilot--stage-0-through-stage-2b)
and each copied run configuration. The
[original manifest](../raw/metal_architecture_pilot_20260915/readiness/campaign_manifest.json)
and [split](../raw/metal_architecture_pilot_20260915/readiness/expected_split.json)
own the complete frozen identities.

## Completed direct-four architecture screen

Each cell below reports the native-selected checkpoint's balanced accuracy
(BA), in percent. All eight runs trained for 50 epochs; selected epoch is
shown separately. This screen tests two learning rates, not a full HPO search.

| Family | LR 3e-5 BA % (epoch) | LR 1e-4 BA % (epoch) | Higher-LR minus lower-LR, pp |
|---|---:|---:|---:|
| Only-GVP | 64.711 (44) | 70.897 (35) | +6.186 |
| Only-ESM | 72.123 (22) | 71.748 (18) | −0.375 |
| GVP + early fusion | 69.367 (40) | 68.239 (16) | −1.128 |
| GVP + graph-level late fusion | 71.789 (17) | 72.312 (7) | +0.524 |

Within the direct-four screen, the largest observed single-seed score is
late fusion at LR 1e-4, only
**0.189 percentage points** above the best Only-ESM score. That small difference
does not establish a reliable family advantage. Learning-rate response varies
by family: GVP improves appreciably at the higher rate, whereas ESM and early
fusion do not in this screen. At each family's selected rate, minimum class
recall is 34.375% for GVP, 59.375% for ESM, 46.875% for early fusion, and
56.250% for late fusion. GVP's higher BA comes with lower Zn recall than its
lower-rate run (34.375% versus 40.625%).

Hybrid is **deferred by the prespecified scheduling rule**: the best early
BA is 1.530 percentage points below the best GVP BA, whereas the gate requires
at least a 1-point improvement. The minimum-recall condition passes. This is
an exploration-priority decision, not evidence rejecting hybrid fusion.
See the [architecture screen](../raw/metal_architecture_pilot_20260915/continuation/snapshots/attempt_029/architecture_screen.csv)
and [decision record](../raw/metal_architecture_pilot_20260915/continuation/snapshots/attempt_029/decision_record.md).

## Completed target-formulation screen, seed 42

**Use the common four-class score to compare training formulations.** Every
checkpoint is selected using native `val_metal_balanced_acc` for its own
four-, five-, or six-class training target. The common-four result is then
read from that same checkpoint; it is not the maximum collapsed-four score
at some other epoch. The seed-43 repeats likewise use each variant's learning
rate selected by its seed-42 native score.

Probability aggregation sums the native Fe/Co/Ni probabilities before taking
the four-class argmax (`logsumexp` over their logits). It does not merely
rename the native hard predictions or merge a native confusion matrix.
Even an unmerged class can therefore lose predictions: five-class ESM's Zn
recall is 78.125% natively and 68.750% after probability aggregation.

The complete seed-42 target screen has 18 fits: three core families × three
training targets × two learning rates. Direct-four rows are the matched
A1/A2 references; five/six rows come from completed T1/T2. Each score cell
shows **native BA % / same-checkpoint common-four BA %**, followed by the
native-selected epoch. The last column follows native BA for the completed
repeat, not whichever LR has the highest common-four score.

| Family | Training target | LR 3e-5: native / common-four (epoch) | LR 1e-4: native / common-four (epoch) | Native-selected LR |
|---|---|---:|---:|---:|
| Only-GVP | Four | 64.711 / 64.711 (44) | 70.897 / 70.897 (35) | 1e-4 |
| Only-GVP | Five | 60.947 / 65.062 (33) | 63.175 / 69.600 (26) | 1e-4 |
| Only-GVP | Six | 58.056 / 66.391 (42) | 57.648 / 67.031 (32) | 3e-5 |
| Only-ESM | Four | 72.123 / 72.123 (22) | 71.748 / 71.748 (18) | 3e-5 |
| Only-ESM | Five | 64.040 / 68.289 (20) | 63.468 / 70.077 (14) | 3e-5 |
| Only-ESM | Six | 60.856 / 69.396 (36) | 60.324 / 68.531 (10) | 3e-5 |
| GVP + late fusion | Four | 71.789 / 71.789 (17) | 72.312 / 72.312 (7) | 1e-4 |
| GVP + late fusion | Five | 67.999 / 72.960 (19) | 65.248 / 70.452 (14) | 3e-5 |
| GVP + late fusion | Six | 64.462 / 73.077 (48) | 59.749 / 71.390 (18) | 3e-5 |

The fixed-LR observations differ across families and rates. At 3e-5,
six-class GVP exceeds direct-four GVP by 1.679 common-four BA points; at
1e-4, direct-four GVP has the higher score. Late fusion's five/six targets
exceed direct-four at 3e-5 but fall below it at 1e-4. Direct-four ESM has
the higher common-four score at both tested rates. These are single-seed
observations, not a general target-formulation advantage.

Comparing native BA across target sizes would obscure these changes in
ordering. Selection also matters within a target: native BA chooses 3e-5
for six-class GVP and five-class ESM, although each has a higher common-four
score at 1e-4. Those native-selected choices were retained for the repeats.
Different native inverse-frequency class weights and checkpoint-selection
objectives are part of the recipes being compared. Neither switching to
another epoch's collapsed maximum nor silently changing the LR-selection
metric would preserve this comparison.

### Class-level qualifications

The following recalls use each target/family's native-selected learning rate
from the table above. Each remains a single seed-42 result.

| Target/family at selected LR | Common-four recalls %, Mn / Cu / Zn / VIII | Native Fe recall % | Native rare-metal recall % |
|---|---|---:|---|
| GVP, four-class | 85.567 / 93.333 / 34.375 / 70.313 | — | — |
| GVP, five-class | 83.505 / 93.333 / 28.125 / 73.438 | 90.909 | Co+Ni: 20.000 |
| GVP, six-class | 65.979 / 93.333 / 28.125 / 78.125 | 90.909 | Co: 7.692 (1/13); Ni: 57.143 (4/7) |
| ESM, four-class | 94.845 / 73.333 / 59.375 / 60.938 | — | — |
| ESM, five-class | 67.010 / 73.333 / 68.750 / 64.063 | 63.636 | Co+Ni: 35.000 |
| ESM, six-class | 91.753 / 73.333 / 50.000 / 62.500 | 68.182 | Co: 15.385 (2/13); Ni: 57.143 (4/7) |
| Late fusion, four-class | 72.165 / 73.333 / 87.500 / 56.250 | — | — |
| Late fusion, five-class | 73.196 / 73.333 / 75.000 / 70.313 | 84.091 | Co+Ni: 25.000 |
| Late fusion, six-class | 78.351 / 73.333 / 68.750 / 71.875 | 84.091 | Co: 7.692 (1/13); Ni: 71.429 (5/7) |

In native-five artifacts, the label `Class VIII` denotes **Co+Ni only**
(20 validation sites). In collapsed-four artifacts it denotes **Fe+Co+Ni**
(64 sites). Raw labels are preserved exactly; the table spells out that
distinction.

GVP's low-LR six-class common-four gain accompanies a Zn recall decrease
from 40.625% in the same-LR direct-four run to 28.125%. Its low-LR five-class
Zn recall is 18.750%. At native-selected rates, late-six has a slightly
higher common-four BA than late-four, but Zn recall is 68.750% versus
87.500%; Class VIII recall moves in the other direction. Native Co recall
is only 1/13 in five of the six six-class fits, and 2/13 in the remaining
ESM low-LR fit. A higher common-four score alone cannot justify promotion
or a claim that rare native metals are predicted well. No case-level error
claims are made from these aggregate metrics.

Even an improved minimum common-four recall can hide a class regression:
at LR 3e-5, late-six improves common-four BA by 1.289 points and minimum
recall by 12.500 points over late-four, while Zn recall falls by 15.625
points. Protect each common-four class when judging the primary endpoint.
Poor native Co recognition limits finer-metal claims; it does not invalidate
an otherwise correctly measured four-class endpoint.

Exact values and selected epochs:
[target screen](../raw/metal_architecture_pilot_20260915/continuation/snapshots/attempt_029/target_formulation_screen.csv),
[selected results](../raw/metal_architecture_pilot_20260915/continuation/snapshots/attempt_029/validation_results.json),
[independent 18-run audit](../raw/metal_architecture_pilot_20260915/continuation/analysis/target_formulation_snapshot_attempt_029.json),
[audited comparison table](../raw/metal_architecture_pilot_20260915/continuation/analysis/target_formulation_snapshot_attempt_029.csv),
[new run artifacts](../raw/metal_architecture_pilot_20260915/continuation/runs/),
[first-allocation A1 artifacts](../raw/metal_architecture_pilot_20260915/partial_a1/runs/).

## Completed selected-LR seed repeats

All ten selected recipes have completed model seeds 42/43. Scores are
percentages; SD is the sample standard deviation across those two training
seeds, not a confidence interval. The final column is the lowest common-four
class recall observed in either seed. Each native and common-four result
comes from the same native-selected checkpoint.

| Family | Target | Selected LR | Native BA mean ± SD | Common-four BA mean ± SD | Worst common-four recall |
|---|---|---:|---:|---:|---:|
| Only-GVP | Four | 1e-4 | 72.073 ± 1.663 | 72.073 ± 1.663 | 34.375 |
| Only-GVP | Five | 1e-4 | 61.831 ± 1.901 | 65.612 ± 5.640 | 18.750 |
| Only-GVP | Six | 3e-5 | 57.861 ± 0.275 | 66.832 ± 0.624 | 21.875 |
| Only-ESM | Four | 3e-5 | 74.342 ± 3.138 | 74.342 ± 3.138 | 57.813 |
| Only-ESM | Five | 3e-5 | 66.529 ± 3.521 | 72.878 ± 6.490 | 62.500 |
| Only-ESM | Six | 3e-5 | 62.417 ± 2.207 | 72.650 ± 4.602 | 50.000 |
| GVP + late fusion | Four | 1e-4 | 72.437 ± 0.177 | 72.437 ± 0.177 | 56.250 |
| GVP + late fusion | Five | 3e-5 | 69.499 ± 2.121 | 74.718 ± 2.486 | 70.313 |
| GVP + late fusion | Six | 3e-5 | 63.979 ± 0.683 | 72.300 ± 1.099 | 59.375 |
| GVP + early fusion | Four | 3e-5 | 65.926 ± 4.866 | 65.926 ± 4.866 | 18.750 |

### Architecture interpretation

Only-ESM has the largest direct-four mean. Its BA exceeds GVP in both seeds,
by 1.226 and 3.312 points, while its mean Cu and Class-VIII recalls are lower
by 20.000 and 10.938 points. Late fusion's small seed-42 advantage over ESM
reverses in seed 43; its mean is 1.905 points below ESM. ESM and late fusion
also trade Mn against Zn recall. These selected-recipe comparisons do not
establish a universal model-family advantage.

Early fusion's seed-43 BA is 62.486% (selected epoch 21), versus 69.367%
at epoch 40 in seed 42. Its worst Zn recall is 18.750%. These results do not
support automatically expanding this tested configuration into hybrid
fusion; they do not reject early fusion as an architecture class.

### Target-formulation interpretation

- GVP five/six recipes have lower common-four BA than direct-four in both
  seeds, with mean differences of −6.461 and −5.241 points.
- ESM five-class changes from below direct-four to slightly above it between
  seeds, with a −1.464-point mean difference. Six-class ESM is lower in both
  seeds, with a −1.691-point mean difference.
- Late-five exceeds late-four in both seeds, with a +2.281-point mean
  difference. Late-six changes direction between seeds and is nearly tied
  on the mean (−0.137 points). The late-five result is an exploratory
  challenger, not a promoted target formulation.

These compare separately selected training recipes: GVP-six and both late
target contrasts use different learning rates from their direct-four
reference. They do not isolate a target effect at one fixed learning rate.
The completed seed-42 table above preserves the fixed-LR comparisons.

Late-five's mean recalls are Mn 73.196%, Cu 73.333%, Zn 79.688%, and VIII
72.656%. Relative to late-four, its mean Zn and Mn recalls fall by 1.563
and 1.031 points while VIII improves by 11.719 points. Its improved aggregate
and minimum recall therefore do not mean every class improves. Native-six
Co recall remains weak: means are 7.692%, 11.538%, and 15.385% for GVP,
ESM, and late fusion, respectively; each has a worst seed of 1/13. This
limits fine-metal discrimination claims without invalidating the coarse
four-class measurements.

The [final analysis](../raw/metal_architecture_pilot_20260915/continuation/analysis/original_final_two_seed_analysis.json)
and [CSV](../raw/metal_architecture_pilot_20260915/continuation/analysis/original_final_two_seed_analysis.csv)
include all per-seed recalls, epochs, LR mappings and paired run differences.
The [checkpoint audit](../raw/metal_architecture_pilot_20260915/continuation/analysis/original_final_checkpoint_and_early_audit.json)
independently matches all 30 selected checkpoint hashes and supporting run
files to final provenance.

## Decision and limits

The bounded training matrix is complete; no architecture or target is
promoted. Training-seed repeats use the same validation data, so they are
not independent data confirmation or grouped-fold evidence. Learning-rate
selection used seed 42, and means/SDs remain conditional on that selection.
Original-pilot case-level paired predictions and grouped-fold confidence
intervals were not produced. Preserve common-four class recalls and native
Fe/Co/Ni diagnostics separately when choosing the next controlled comparison.

No HPO, Stage 6/6B, or held-out test ran. Hybrid remains deferred; the required
RING comparison is still separate. The exact PinMyMetal reference benchmark
and scientifically unresolved primary final-test route also remain separate.

## Portable evidence and provenance

The [portable continuation package](../raw/metal_architecture_pilot_20260915/continuation/)
contains small aggregate snapshots through attempts 021 and 029, recovery
evidence, 27 continuation run records, and their verified transfer receipts.
The earlier aggregate is preserved unchanged; the new snapshot adds the
completed T1/T2 screen. The ten later seed-repeat records are
supplemental to those snapshots. Run records are stored once for this report;
the first three A1 runs remain in the first-allocation package. The
[copy inventory](../raw/metal_architecture_pilot_20260915/continuation/portable_copy_manifest.tsv)
records exact SHA256 and byte counts for 194 files. Genuine terminal metadata
is retained in a separate [finalization directory](../raw/metal_architecture_pilot_20260915/continuation/finalization/).
The target audit checks
all 18 core runs against their configurations, metadata, selected histories,
archived aggregates, shared cohort, and common normalization. Checkpoints, feature caches, archive
binaries, and redundant standalone dataset summaries are not copied into Git.

The final capture verified zero pending runs, genuine completed queue states,
and unchanged frozen manifest source sets (73 original files; 74 geometry
files). Its [receipt](../raw/metal_architecture_pilot_20260915/continuation/finalization/transfer_receipt.json)
certifies local SHA and [Drive persistence](https://drive.google.com/file/d/1lgvyKm5fm5436TdiU_VhmVhNqkHWaum0/view)
for archive `be9aaf95392b96c88f649c06b732a6ec91980daf18c0a6829cf88e8b0f8a2e3e`.
Capture manifest SHA is
`e852a60ab70f1adb2be2396d6b0607321e44c2d61506c8b6d48b46edde31d743`.
Original scientific manifest remains
`14010a873801cdbc2d067b8e61137f5469afb4f5d9f833cdd2251fa84080cdd5`.

That capture occurred before teardown and truthfully retains
`gpu_stopped=false`. The later [session-stop receipt](../raw/metal_architecture_pilot_20260915/continuation/closeout_allocation2/post_stop/session_stopped.json)
and [closed allocation report](../raw/metal_architecture_pilot_20260915/continuation/closeout_allocation2/post_stop/post_stop_closeout.json)
prove shutdown and cumulative accounting. The second interval used
15,435.477 seconds; the first used 3,827.969 seconds. Total use was 5.350957
hours, leaving 4.649043 hours of the original cap. The
[post-stop transfer receipt](../raw/metal_architecture_pilot_20260915/continuation/closeout_allocation2/post_stop/transfer_receipt.json)
verifies [Drive closeout](https://drive.google.com/file/d/1O3JQJnZ_GxE784rPd4OwI_1qiyDwNNuU/view),
archive SHA `7aceea2f585081298e04d02cc692f86ef34f93ec82ae22b28f1113674ac79778`.
