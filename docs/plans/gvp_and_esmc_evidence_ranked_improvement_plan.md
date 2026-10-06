Evidence-ranked plan to improve GVP and GVP+ESMC

For v3, use the [campaign plan](../campaigns/pmm_ion_metal_v3/plan.md): it
supersedes this historical ranking's execution order and selection rules,
including the [regularization amendment](../METAL_TRAINING_PIPELINE_PLAYBOOK.md#v3-step-d-regularization-amendment).
The earlier measurements and planning estimates below remain historical.

1. Objective and evidence
Improve Only-GVP or GVP+ESMC metal prediction on the current corrected PMM ion cohort. Count an improvement as significant when it delivers:
•
At least +2 percentage points in validation balanced accuracy against its matched baseline.
•
A positive lower bound on the paired confidence interval.
•
No common-four class losing more than 3 percentage points of mean recall.
Use direct four_class training for architecture experiments. Keep five/six-class target comparisons separate.
The ranking consolidates Antigravity’s original 13 claims, the recovered Claude reviews, Codex follow-ups, and subsequent experiment evidence. The earlier 45–95% success estimates are unsupported. The ranges below are subjective planning estimates for a reproducible ≥2-point gain; overlapping ranges indicate uncertain ordering.
Current evidence changes the priorities:
Current direct-four model
Selected fold-0 BA
Mean BA over final 10 epochs
Only-GVP
86.02%
73.47%
Only-ESMC
88.39%
83.62%
GVP+ESMC late fusion
89.47%
85.78%
These are one-fold, seed-42 observations, not confirmed model rankings. The selected scores are preserved in the current comparison; the final-ten averages were recalculated from saved training histories.
Additional findings:
•
The historical branch-learning-rate change improved collapsed-four BA by 0.97 points, but native-five BA by only 0.12 points, in one pocket-level fold. It has not demonstrated the requested ≥2-point improvement. Matched diagnostic
•
In current validation predictions, GVP correctly classifies 105 ions that ESMC misses; fusion recovers 28 of them. This suggests complementary information exists, without establishing how much a learnable fusion can recover.
•
A fixed 50/50 blend of current GVP and ESMC predictions achieves 88.17%, below fusion’s 89.47%.
•
Earlier geometry gains changed direction between seeds. RING produced modest GVP gains with class tradeoffs and no selected-score improvement in the tested fusion pairs. Matched pilot findings
2. Ranked improvement candidates
Likelihood refers to achieving the success criterion in the model identified, rather than merely improving training loss.
Rank
Issue and proposed intervention
Estimated likelihood
Evidence and first decisive comparison
1
Unstable optimization: learning rate and schedule
20–35%, mainly GVP
Current GVP loses substantial validation performance after its early selected checkpoint. Compare fixed versus cosine scheduling at the same starting LR and 50 epochs. Evaluate selected BA and training stability separately.
2
Summed messages overwhelm local residue information
15–30%, mainly GVP
Current layers sum neighbor messages; degree normalization already exists but is disabled. Compare sum versus mean aggregation, keeping radius and capacity fixed. Measure node-state similarity and vector magnitudes to test the proposed mechanism.
3
Insufficient structural regularization
15–25%, mainly GVP
Current GVP finishes near 97.6% training BA while validation deteriorates. Test scalar/vector residual dropout at 0.1 against zero. Treat weight decay and coordinate augmentation as separate interventions.
4
Fusion trains structural readout components too slowly
10–20%, fusion
The trunk/readout LR disparity is real; the historical gain was below one point. Move only the input-vector projection and structural pooling/projection modules to the GVP rate. Keep ESM, fusion head and gate rates fixed initially.
5
Fusion receives weak structural learning signals
10–20%, fusion
Historical optimizer evidence supports weaker structural gradients, and current fusion misses many GVP-only correct cases. After resolving rank 4, compare a GVP-only auxiliary metal loss of weight 0.3 against baseline; test ESM modality dropout separately.
6
Coordination summaries are omitted or poorly scaled
10–20%, either
Physically plausible, but earlier end-to-end gains were small and seed-dependent. Compare masked, counts-only and counts-plus-angles inputs with identical model dimensions; separately test training-fitted normalization and explicit missing-angle handling.
7
Global pooling loses target-ion locality
8–15%, fusion
Add a separate first-shell summary alongside the existing global summary. This differs from the already screened first-shell bias, which did not improve BA. Keep this item deferred under the current binding-awareness pause.
8
Early/hybrid ESM representations have unsuitable scaling
5–12%, early/hybrid fusion
The early encoder lacks input normalization, but historical embedding-scale measurements concern a different ESMC model. Measure current activations first, then compare identical early/hybrid models with normalization off/on.
9
Structural augmentation is absent
5–12%, mainly GVP
Compare coordinate noise of 0.1 Å with baseline; test outer-residue dropout separately. Preserve the target ion and first-shell information. Account for graph-rebuilding cost.
10
Vector representation poorly exposes coordination direction
5–12%, mainly GVP
Screen vector normalization and identity-versus-random vector mixing separately. A donor-to-target-ion vector is a later, separately versioned feature experiment requiring cache reconstruction and label-independent donor selection.
11
Class weighting and small batches produce unstable minority-class updates
5–10%, either
Current GVP Class VIII recall is 64.61%. Compare existing endpoint-balanced weights with inverse-square-root weighting. Keep sampling unchanged; do not combine weighting and oversampling in the first comparison.
12
Asymmetric fusion gate
3–10%, fusion
A gate near 0.5 does not prove information loss: subsequent layers can compensate. Test direct concatenation with unchanged dimensions. Residual addition changes capacity and requires a separate comparison.
13
RING information is unused
3–10%, GVP; lower for fusion
Previous matched gains were modest and class-dependent. Audit ion-cohort edge coverage, then compare RING off/on without changing radius or shell definitions.
14
Prediction combination or averaging could exploit complementary errors
3–8%, combined predictor
Both historical and current validation reject an automatic 50/50 switch. Evaluate seed averaging and fitted blends only with separate fitting/evaluation partitions and matched model-count controls.
15
Classifier or backbone capacity is inappropriate
3–8%, either
Compression from 288 to 128 dimensions is not proof of a bottleneck. Compare the existing head with a linear head and width 256. Keep backbone size fixed before testing smaller/deeper trunks.
16
Final vector update is unused by readout
3–8%, either
Earlier vector norms already influence scalar states. Test an invariant final-vector readout; distinguish a possible predictive gain from removing dead computation.
17
Raw edge-vector magnitude harms training
1–5%, either
Test unit directions only after separating direction from physical length. Preserve the raw scalar edge length and distance RBF inputs. Unit vectors alone do not fix degree-dependent message accumulation.
18
Collapsed-four auxiliary loss
1–5% additional gain in a separate six-class arm
Direct-four training already aligns the output with the endpoint. The existing auxiliary implementation requires six classes; the proposed five-class command is invalid. Preserve native-six selection and common-four reporting.
19
Global gradient clipping couples the modalities
1–5%, fusion
Log pre-clipping branch norms first. Compare clipping at 1 versus 5 only if the current ion runs show persistent clipping and disproportionate branch suppression.
20
Pruning dead PROPKA channels and other cleanup
Approximately 0% on the current cohort
The current contract already masks PROPKA, SASA and charge-proxy channels. Removing zero inputs adds no information. Logging fixes, inactive focal-loss corrections and dead-parameter cleanup belong in maintenance.
Mean aggregation, vector normalization and vector-channel dropout have precedents in the original GVP implementation. These support the mechanisms, not the probability estimates. Likewise, multimodal optimization research supports investigating unequal learning dynamics without guaranteeing improvement here.
3. Prerequisites and implementation changes
Prepare a separately named gvp_ion_improvements_v1 study. Preserve the frozen PMM campaign, historical results and existing uncommitted work.
Before using the unfinished improvement suite:
•
Restore default checkpoint compatibility. A read-only synthetic check confirmed that its dropout edit moves scalar_mlp.2 parameters to .3, breaking strict loading even with dropout disabled.
•
Make unspecified classifier width inherit hidden_s; the new CLI default of 128 otherwise changes historical configurations using other widths.
•
Preserve scalar edge lengths when enabling unit-vector directions.
•
Replace the suite’s historical pocket defaults with the certified ion cohort, frozen PDB-grouped membership, ESMC-600M embeddings, 1,152-dimensional inputs and intentional feature omissions.
•
Remove hard-coded historical baseline scores, missing-metric-to-zero fallbacks and “winner” labels based on tiny single-fold deltas.
•
Require complete run status, matching configuration/input identities, checkpoint hashes and successful replay before reusing a run.
•
Make prediction joins require unique, identical validation IDs, matching targets and canonical probability-column order.
•
Replace same-data blend fitting/reporting with a separated procedure. Ordinary out-of-fold predictions alone do not eliminate stacking leakage when base-model training includes the eventual evaluation groups.
•
Keep all new behavior opt-in. Reuse existing scheduling and aggregation flags; add residual dropout separately from the draft’s scalar-MLP dropout.
•
Record exact optimizer parameter ownership, every group’s LR, selected checkpoint identity and diagnostic metrics.
Document executable recipes in the metal playbook before any future execution. Preserve the distinction between implemented, tested and experimentally evaluated.
4. Experiment sequence, tests and decision gates
First screening block: ranks 1–4.
Use seven configurations: baseline GVP, GVP with cosine, GVP with mean aggregation, GVP with residual dropout, baseline fusion, fusion with structural-branch LR grouping, and Only-ESMC as reference.
Compare them on frozen folds 0 and 1, using model seeds 42 and 43 and the existing split seed 42: 28 configuration/fold/seed cells, reduced only by verified compatible reuse. Preserve the current 50-epoch recipe, batch size, capacity, feature contract and endpoint-balanced class weights except for each explicitly tested factor.
Advance at most one candidate per family when its mean screening gain is at least 1 point, both seeds have positive mean differences, and class-recall protection passes. This is a screening threshold; it is not a significant-improvement claim. Preserve inconclusive results rather than declaring an architecture ineffective.
Confirmation.
Freeze the selected candidates before examining folds 2–4. Complete their comparisons against matched controls across all five folds × two seeds. Report the newly evaluated folds separately as a check on screening optimism.
Require:
•
Mean paired BA improvement ≥2 points.
•
Positive paired confidence-interval lower bound.
•
Positive fivefold mean improvement in both seeds.
•
No class’s mean recall dropping by more than 3 points, and no missing or zero-mean class recall.
•
A positive mean gain on the newly evaluated folds.
Use 10,000 paired bootstrap resamples, retaining PDB clusters and shared model/seed pairing. For two primary family-improvement claims, use 97.5% intervals to control multiplicity. Report support counts and uncertainty limitations. Report fusion versus Only-ESMC separately to show whether structure adds predictive value.
These remain development-validation comparisons. Selection and reporting on reused validation data can produce optimistic estimates; they are not an unbiased final generalization result. Validation-selection guidance
Subsequent work.
Proceed through the remaining ranking only after updating it with confirmed results. Combine changes only after separate comparisons establish their effects, and retain the single-change controls. Binding-aware work remains deferred until its distinct new hypothesis receives its own execution scope.
Required tests and artifacts.
Test strict legacy checkpoint loading, unchanged default outputs, actual optimizer grouping, finite/equivariant geometric behavior, preserved physical distances, correct dropout behavior, training-only normalization, fold identity, replay reconciliation and rejection of incomplete or mismatched results.
Produce a claim-to-evidence inventory, frozen experiment manifest, screening table, paired confirmation report and per-class recall report. Keep the study’s status and decisions in the existing documentation owners.
Execution boundary.
This is a plan only. The remaining 36 PMM fits stay deferred. Future compute requires an explicit scope and measured budget using the project GPU workflow, one GPU and one training worker. Historical pocket-run timings must not be reused as ion-campaign forecasts. Held-out evaluation remains outside this improvement study; any eventual final report must pass Stage 6B refit and Stage 7 safeguards.
