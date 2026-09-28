# PMM five-class fold-0 screen — 2026-09-28

Later evidence: the [core integration and concurrency summary](summary_pmm_core_continuation_20260928.md)
qualifies GVP5/GVP6 through explicit retrospective diagnostics. This historical
screen's original strict failures and values remain unchanged.

Five-class training scored higher than six-class training for Only-ESMC on
common-four balanced accuracy. Six-class training scored higher for graph-level
late fusion; the provisional Only-GVP comparison also favors six classes.
All three new five-class fits completed 50 epochs. ESMC and late fusion passed
the original strict independent replay; GVP did not.

## Matched common-four results

Values are percentages, from each target's native-BA-selected checkpoint.
The comparison uses the same 1,492 validation ions, frozen fold 0 and seed 42.

| Family | Five-class training | Six-class training | Five minus six, pp | Interpretation |
|---|---:|---:|---:|---|
| Only-ESMC | 89.4354 | 85.8393 | +3.5960 | Five higher; both strict replay passes |
| Only-GVP | 78.2397 | 82.8399 | −4.6002 | Six higher provisionally; qualifications below |
| GVP + ESMC, graph-level late fusion | 87.6608 | 88.8124 | −1.1515 | Six higher; both strict replay passes |

This is a single-fold screen, not fivefold confirmation or evidence of a
universal target-scheme advantage. No model is promoted. The strict
[screen report](../raw/pmm_five_class_screen_20260928/execution/five_class_screen.md)
excludes failed GVP5 from certified comparisons; its value above comes from
the separately labeled
[failure diagnostic](../raw/pmm_five_class_screen_20260928/execution/only_gvp__five_class__none__fold0__seed42/failed_replay_diagnostic.md).
Historical GVP6 retains supplemental v2.1 agreement only, rather than an original
strict-replay pass. Neither qualification is upgraded by this table.

Against matched direct-four training, the verified five-class ESMC result is
1.0409 pp higher in common-four BA, while five-class late fusion is 1.8045 pp
lower. ESMC's increase includes a 3.3149 pp decline in Zn recall. Per-class
recalls, support, confusion matrices and macro-F1 remain in the JSON reports;
an average score does not replace the rare-class safeguards.

## Training and evaluation definitions

Training used Mn, Cu, Zn, Fe and Co+Ni. The internal five-class name `Class VIII`
means Co+Ni only. Common-four reporting sums Fe and Co+Ni probabilities before
argmax, producing Mn, Cu, Zn and Fe+Co+Ni. Native balanced accuracies from
different class vocabularies are not compared as a shared endpoint.

| Five-class family | Selected epoch | Native-five BA | Native-five macro-F1 | Fe recall | Co+Ni recall |
|---|---:|---:|---:|---:|---:|
| Only-ESMC | 41 | 80.8663 | 78.8576 | 89.7906 | 45.3947 |
| Only-GVP, provisional | 47 | 68.1998 | 65.2286 | 72.7749 | 19.7368 |
| Graph-level late fusion | 50 | 84.5951 | 83.1475 | 77.4869 | 73.6842 |

The [frozen protocol](../../plans/pmm_five_class_screen_v1.json) fixes 50 epochs,
batch size 16, the same cohort/features/recipes as the controls and common-four
per-ion loss weights. Both Fe and Co+Ni receive the common-four Class VIII
weight. Checkpoints use highest native-five validation BA, earliest tie; no
common-four checkpoint reselection, HPO or held-out access occurred. Previously
completed fits, embeddings, PMM folds and preparation certificates were reused.

## Preserved GVP verification failure

GVP5's two exports have identical UIDs, row order, targets and native/common-four
class predictions. Their class-based metrics agree. However, 22 probability
fields across six ions differ by more than the fixed `1e-6` tolerance; the
maximum is `2.38e-6`. All 70 terminal files were verified and backed up.
No retry or tolerance change was performed. The historical nine-fit v2.1 policy
does not cover this new fit. A cause is not established by this CSV comparison;
see [TECH-023](../../FOLLOW_UP_TECHNICAL_ISSUES.md#tech-023--full-fit-gvp-independent-replay-exceeds-the-frozen-probability-tolerance).

After diagnosis and verified persistence, the separate authorized late-fusion
fit proceeded under the unchanged strict contract and passed. Its success does
not certify GVP5. Evidence grades are 5 for the verified single-fold fits and 6
for the uncertified GVP5 diagnostic.

## Execution and remaining scope

The same-region recovery selected one L4 in `us-central1-a`. All terminal
artifacts are independently verified on the host: 71 ESMC files, 70 GVP files
and 71 late-fusion files. See the
[closeout record](../raw/pmm_five_class_screen_20260928/execution/verified_closeout.json)
for provider shutdown, superseded-resource cleanup and accounting.

| Family | Preparation, minutes | Full fit plus replay attempt, minutes |
|---|---:|---:|
| Only-ESMC | 22.06 | 41.36 |
| Only-GVP | 21.22 | 48.42 |
| Graph-level late fusion | 1.61 | 26.58 |

Late fusion reused compatible cached graphs. These timers exclude admission,
input verification, host transfer and provider closeout; they are not GPU
utilization measurements. CPU preparation remains a substantial cost. No
scientific settings were changed to improve throughput.

The active v2 core scope has 9 of 45 fits trained and 36 untrained. Of those
nine, seven pass original strict replay, GVP6 has only historical supplemental
agreement, and GVP5 remains uncertified. The three historical binding-aware
fits are separate and further awareness work stays paused. TECH-023/025,
remaining-budget authorization, full-fold assessment and final-refit/held-out
gates remain open. The current session does not activate the older 30-hour/$34
proposal or claim the full comparison plan is complete.

Full checkpoints and original metadata remain in the canonical host/VM backup.
The [portable batch](../raw/pmm_five_class_screen_20260928/README.md) contains exact
small artifacts, explicit metadata excerpts with original hashes, and separate
successful and failed verification records. Preparation/stockout snapshots are
preserved independently of the later execution evidence.
