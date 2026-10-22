# PMM ion metal v3 — step F execution plan (draft v3, approved 2026-10-12)

Status: **approved by the maintainer on 2026-10-12 ([log v3-045](log.md#v3-045)) and executed: the refit, both freezes, the label-blind preparation, the single test pass and the report are complete ([log v3-046 to v3-048](log.md#v3-048)); step F is complete**
(2026-10-12: the maintainer chose to finalize with the 50-epoch selection, [log v3-042](log.md#v3-042); one CPU
correction round, [log v3-043](log.md#v3-043); the CPU ESMC compatibility check passed, [log v3-044](log.md#v3-044);
see [Implementation notes](#implementation-notes-2026-10-12-log-v3-042) and the [specification](step_f_spec.json)).
Draft v3 of
2026-10-12 ([log v3-038](log.md#v3-038)) supersedes draft v2 of 2026-10-08
([log v3-030](log.md#v3-030)), which corrected the working draft of 2026-10-08
08:45 UTC (`audits/session_notes_20261008/step_f_plan_DRAFT.md` on the campaign
data root, kept unchanged). The rules stay owned by [plan step F](plan.md#steps),
the [A4](assessment_spec.md) and [E2](e2_assessment_spec.md) specifications and
the [DATASETS ledger](../../../DATASETS.md#test-use-ledger); this page plans their
execution. No refit, test-data access or GPU start is authorized by it.

**Changes in v3:**
- **Schedule binding.** The refit's epoch count is no longer hard-coded at 50. It is resolved from a bound schedule
  decision: 50 by default, and 10 only if the maintainer approves the pre-step-F 10-epoch comparison
  ([handoff section 11](handoff.md#11-pre-step-f-review)) and it is completed and adopted under its frozen gates. The
  refit command, receipt, identity and every later binding carry the resolved count, so the configuration, refit
  duration and checkpoint provenance stay consistent with the cross-validation evidence (TECH-027: the same epoch
  count and terminal rule in CV and refit).
- **Test access in two dated freezes** (F3). Rules before any test-side read; derived membership before any
  prediction or label read. This resolves the earlier wording "rules with membership hashes before any test access",
  which could not be met literally: the memberships are derived from test structures.
- **The prior test-exposure disclosures are listed in full** (F4) as report requirements.
- **Costs refreshed.**

Corrections carried over from v2 against the 2026-10-08 working draft: the selection input is the E2 final selection; the refit
command is genuinely validation-free; four-, five- and six-class winners are handled explicitly; the refit checkpoint
is bound by hash through test reporting; label blindness, source-row accounting, unscorable-row denominators and
report recovery from immutable predictions are tested requirements.

## F0. Gates

1. Original step E complete: 44 units complete and replayed; the final A4
   assessment copy has a non-null `stage6_selection` and
   `held_out_test_accessed` false; evidence copied; log entry. **Met** (log v3-031).
2. Step E2 complete under the frozen E2 specification (36 fits verified, all 17
   declared comparisons reported, a machine-readable final selection), or a
   dated log entry by which the maintainer cancels E2. **Met** (log v3-037).
3. Label `both_results_secondary` recorded. **Met** (log v3-022).
4. Schedule decision recorded, by exactly one of the paths below. **Met by (b)** on 2026-10-12 (log v3-041):
   `S10_20261012T111859Z` (SHA-256 `8fbce69b…`), not adopted, so `E_sel` = 50.
   - (a) a dated log entry in which the maintainer declines the pre-step-F comparison or goes straight to step F
     (epochs = 50);
   - (b) the comparison's machine-readable schedule decision after its completed assessment (epochs = 10 if
     adopted, otherwise 50): `result.schedule_decision` of
     `step_s10_evidence/assessments/S10_<UTC>/assessment.json`, bound by SHA-256 (chosen and prepared, log v3-039);
   - (c) a dated entry recording that the comparison stopped without a decision (epochs = 50 unless the maintainer
     decides otherwise).
5. The operative ceiling, recorded in execution controls, covers the refreshed
   F forecast including storage; otherwise the budget question under the maintainer's local development guidelines (not published). **Pending:** `record-ceiling` before the
   refit launch ($55 recorded for E2 and S10; forecast in F6).
6. The CPU readiness checks of F8 pass on synthetic or training-side data. **Met** (log v3-043: the corrected tools,
   38 tests including the two synthetic fits, and a scripted rerun of readiness stages A–D).
7. The maintainer approves this plan. **Met** (log v3-045). Later, separately, the maintainer gives three go-aheads: Freeze 1
   before F3b, Freeze 2 after F3b, and the single test pass (F4) after Freeze 2 (see F7).

## F1. Selection input (CPU; no fit; no judgment)

- Inputs: the E2 final-selection artifact (selected cell, whether it replaced the original selection, the original A4
  selection, the E2 specification and manifest SHA-256 values, evidence references), and the schedule decision of F0
  gate 4 (log entry, or the comparison's decision artifact and its SHA-256). Only if the maintainer cancelled E2 by a dated
  log entry: the A4 `stage6_selection.selected` alone.
- The step F record binds:
  - the selection artifact path and SHA-256, and the original E assessment SHA-256;
  - the selected family, target and recipe, and the recipe definition from the hash-frozen runner;
  - the resolved epoch count `E_sel` with its source (log entry or decision-artifact SHA-256).
- Refused:
  - a missing artifact, a hash mismatch, `held_out_test_accessed` not false, or an incomplete assessment;
  - a selection that is not a cell of the campaign grid or of E2;
  - `E_sel` other than 50, or 10 without an adopting decision artifact for exactly the selected cell.

## F2. Stage 6B full non-test refit (GPU; one fit of `E_sel` epochs)

- Tooling (additive files; no `src/` change, so every campaign hash stays
  valid): a workstation launcher reusing the step C/D lane, admission, wait,
  pull, recovery and evidence logic, and a VM-side refit module that builds the
  command from the campaign's command builder for the selected cell and `E_sel`.
- Validation-free command, derived from the selected cell's fold command:
  remove `--n-folds`, `--fold-index`, `--fold-membership-csv`,
  `--fold-membership-sha256` and `--fold-split-source`; remove
  `--export-validation-predictions` (refused without a validation split) and
  the validation selection metric `--selection-metric val_metal_balanced_acc`
  (with no validation the trainer's descriptive metric defaults to
  `train_loss`); add `--val-fraction 0.0`; set `--epochs E_sel`. Unchanged:
  `--checkpoint-rule terminal`, the cosine schedule (over `E_sel` epochs), seed
  42, batch 16, FP32, both learning rates and weight decays, the recipe's flags,
  all model, feature and input options, no test flags (`--run-test-eval` and
  `--allow-final-refit-test-eval` absent; inference happens in F4 outside the
  trainer).
- Weighting policy preserved and recomputed on the full non-test cohort (all
  7,398 training ions): manual common-four equalized multipliers
  `w_c = N / (4 n_c)` by the campaign's weight function; `four_class`:
  Mn, Cu, Zn, Class VIII; `five_class`: Mn, Cu, Zn, Fe and Class VIII (Co+Ni),
  Fe and Co+Ni both at `w_VIII`; `six_class`: Mn, Cu, Zn and Fe, Co, Ni each at
  `w_VIII`. Normalization statistics are fitted by the trainer on the full
  training partition and saved with the run; nothing is refitted later.
- Identity: a refit identity replaces the fold identity. It records the selection SHA-256, the schedule-decision
  source, family, target, recipe, `E_sel`, the cohort and weights SHA-256, the source and runner hashes, seed and
  "full non-test cohort". Run directory:
  `step_f/refit/stage6b_refit__<family>__<target>__<recipe>__e<E_sel>__seed42` (implemented as
  `lanes/lane0/runs/…` through the campaign lanes; see the implementation notes).
- Receipt and binding: fit completed with `E_sel` history rows, terminal rule,
  selected epoch `E_sel` (planned = completed = selected), no validation
  partition, `test_report` none; SHA-256 of the terminal checkpoint and of
  `run_config.json` written into the step F record. Host pull with manifest and
  acknowledgment, evidence copy. Replay analogue without validation data: a CPU
  reload of the bound checkpoint with configuration equality (including
  `epochs = E_sel`), finite parameters and a re-evaluation of the epoch-`E_sel`
  training-set common-four BA against the logged value. F3 and F4 refuse any
  checkpoint other than the bound one.
- Failure: one unchanged rerun; a second failure stops F for diagnosis.

## F3. Test access in two dated freezes

Both freezes are dated entries in the [DATASETS test-use ledger](../../../DATASETS.md#test-use-ledger), each with a log
entry, and each needs the maintainer's go-ahead. The tools enforce the order from the records (F8 item 10): the test-side
actions read the freezes only at their canonical paths under the campaign root, need a `v3-NNN` go-ahead and no
template placeholder, run only with the exact code and specification Freeze 1 binds, and are refused after Freeze 2
(preparation) or after the prediction claim (prediction). The records cannot prove that a go-ahead was given; the
log entry each record names is the evidence.

**F3a. Freeze 1 — the rules, before any test-side read.** Recorded with SHA-256 values:
- The primary report specification: the F4 report function's code hash, the metrics, the allowed wording, the labels
  and the disclosure list of F4.
- The test-row rule of log v3-012. Score every reconstructable row under the training input contract; flag
  symmetry-LINK rows rather than dropping them; count unscorable rows as errors; report "N of 1,488 scored"; add the
  training-eligible sensitivity line.
- The clean-subset rule, label-free: exclusion by training PDB ID, by A1 near-copy chains against the training cohort,
  and by previously opened PDB IDs. The last are defined as the PDB IDs of the test structure sets recorded as opened
  in the ledger (the exact and non-overlapped PinMyMetal test set).
- The code hashes of the label-blind preparation builder (with its column allowlist), the test-side ESMC generation,
  the prediction tool and the report. They must equal the refit manifest's; only an amended Freeze 1 (F3c) may bind
  other code, and it lists the difference.
- The bound refit checkpoint SHA-256, seed 42, `E_sel`, the selection and schedule-decision hashes, no ensemble, no
  calibration, and the label `both_results_secondary`.
- What test-side material will be read (structures, sequences, identifiers), by which tool and when, on the
  workstation CPU. The statement on labels: label-bearing cells of the source CSV and the site manifest pass through
  the CSV reader and are discarded record by record, unused; no label value is used before F4's report.

**F3b. Label-blind preparation** (after Freeze 1).
- **Rows.** PMM's test side (1,488 source rows) is resolved with the training cohort's input contract under the
  v3-012 rule.
- **Label blindness.** The builder reads structures, sequences and identifiers through column allowlists; the
  label-bearing cells pass through the CSV reader and are discarded record by record, unused, and the source file is
  hashed whole. No inclusion, flag, de-duplication or alias decision uses a label value. The resolver uses the target
  residue's metal in memory only to ask whether it is one of the supported transition-metal ions, never which one,
  and no stored output carries it (`ion_key` hashes identifier fields only). The model inputs already exclude the
  metal identity (no metal node; residue-only readout).
- **Source-row accounting.** Each of the 1,488 rows receives exactly one disposition: scored, scored with the LINK
  flag, or unscorable with a reason. The counts sum to 1,488.
- **Clean subset (secondary).** The memberships and exclusion lists of the frozen rule are materialized with their
  reasons. This includes the previously opened PDB ID list, built from identifiers only.
- **Test-side ESMC embeddings** are generated label-free for every context chain with the existing plan, receipt and
  inventory mechanism, on the workstation CPU (the only tooled route; no test file goes to the VM), with the model
  in the training embeddings' precision (implementation notes). The preparation and the embeddings record the
  Freeze 1 they were made under.

**F3c. Freeze 2 — the derived memberships, after the preparation and before any prediction or label read.**
Recorded with SHA-256 values:
- the per-row disposition list with its counts;
- the scored and LINK-flagged lists;
- the clean-subset membership, the exclusion lists and the previously opened PDB ID list;
- the test embedding and prepared-graph inventories;
- the record of what was read, when and by which tool.

Nothing in Freeze 2 may change a Freeze 1 rule; Freeze 2 binds Freeze 1 by SHA-256, and the preparation and the
embeddings must record the same Freeze 1. After Freeze 2, the memberships are final.

**Freeze 1 amendment** (a defect found after Freeze 1 and before Freeze 2; spec `freeze1_amendment`).
- A defect that needs no code or specification change (an interrupted run, a wrong path) is re-prepared into a new
  directory under the same Freeze 1.
- A code fix needs an amended Freeze 1:
  1. a dated log entry describing the defect and the fix, with the maintainer's go-ahead for the amendment;
  2. the old `step_f_evidence/freeze1.json` archived by hand as `freeze1.superseded_<UTC>.json`, never deleted;
  3. the fix committed, with its tests and the affected readiness checks rerun on synthetic or training-side data;
  4. `record-freeze1 --amends <old SHA-256> --prior-test-side-read none|yes --log-entry v3-NNN`, which names the
     superseded record, whether a test-side read happened under it, and every tool whose hash differs from the
     refit manifest's;
  5. a dated DATASETS ledger entry, and the preparation and embeddings repeated into new directories (outputs made
     under the old record name it and are refused);
  6. the amendment disclosed in the report.
- A specification change after `bind-selection` also changes the refit's bound specification, which `refit-receipt`
  and Freeze 1 then refuse; it is not an amendment and needs the maintainer's decision (a refit rerun or a tooling change).
- After Freeze 2 or the prediction claim, there is no amendment: step F stops for the maintainer's decision.

## F4. The one test pass and its report

- **Inference** uses the bound checkpoint and the saved normalization: softmax, with five- or six-class probabilities
  summed into Class VIII before the argmax. No calibration, ensemble or threshold change.
- **The go-ahead and the claim.** The pass runs after the maintainer's own logged go-ahead for the single test pass. Before
  any test structure is opened or any prediction is made, the prediction tool verifies Freeze 1 (its own code) and
  Freeze 2 (every membership file, the recomputed embedding inventory and the bound checkpoint), then writes the
  campaign-wide write-once claim `step_f_evidence/f4_prediction_claim.json` with that go-ahead, the freezes, the
  checkpoint and the output directory. A second pass is refused into any directory. A pass that fails after its claim
  is not rerun without the maintainer's dated decision; the claim is archived by hand, never deleted, and the failure is
  disclosed.
- **Immutable predictions first.** `test_predictions.csv` has one row per source row, with the disposition, flags,
  common-four probabilities (native probabilities equal them for the four-class selection), and the checkpoint
  SHA-256 and `E_sel`. It is written once and hashed.
- **Report second**, as a pure function of the frozen predictions, the label file (its label values are used here
  for the first time) and the Freeze 2 row lists. It accepts only the claimed predictions directory, under the
  recorded freezes and its own frozen code hash. Rerunning the report reproduces it byte for byte without new
  predictions. It contains:
  - the full test "N of 1,488 scored", with unscorable rows counted as errors in every class-recall denominator;
  - common-four BA and the Mn, Cu, Zn and Class VIII recalls;
  - for a five- or six-class winner, the native metrics and the Fe/Co/Ni (six) or Fe and Co+Ni (five) recalls;
  - the LINK-free sensitivity line, and PDB-entry-group bootstrap intervals for uncertainty only;
  - the clean subset, from the same predictions;
  - PMM's published values beside the full-test numbers, in the allowed wording only.
- **Mandatory labels and disclosures** (the report carries all of them; the
  [ledger](../../../DATASETS.md#test-use-ledger) owns the details):
  1. Labels: `both_results_secondary`; the full test is an "exact, possibly overlapped PMM test; includes rows from
     previously opened tests"; the clean subset is "not pristine" and is never compared with PMM.
  2. The exact and non-overlapped PinMyMetal test structure set (the same 316 structures) was opened repeatedly
     before v3 (ledger: 2026-09-18, 2026-09-22, 2026-09-23 and earlier reports). The Harsh and Common-PDBID 70/30
     test IDs come from that set.
  3. The exact Zenodo PinMyMetal test is possibly evaluated: the 2026-09-24 `pmm-zenodo-v2` fold-0 relaunch is
     unconfirmed.
  4. 2026-10-03: the source test file's label and feature metadata were read in aggregate (leak match rates;
     per-label row counts printed); no evaluation.
  5. 2026-10-11: a recursive byte scan of the exact Zenodo test directory without display (log v3-032); no
     evaluation.
  6. Selection on development-validation folds:
     - Folds 1–4 were reused across A4, the exploratory E2 and the post-hoc 10-epoch comparison (run, not adopted),
       whose hypothesis came from fold-0 probes.
     - One training seed (42).
     - The terminal-checkpoint rule was chosen after the v2 fold-0 post-hoc epoch analysis.
     - E2 and the schedule comparison do not establish independently confirmed superiority.
  7. The PMM comparison uses published values only and is not matched; PMM features carry a true-metal label leak
     (TECH-028); no superiority claim, interval or p-value for the difference.
  8. Clean-subset limitation: a context chain shorter than 30 residues is never a near-copy under A1, so a pocket
     whose context chains are all shorter is not excluded by C2 even when the same peptide is in training; the
     preparation summary counts such rows.
  9. The test-side ESMC embeddings were generated on the workstation CPU with the model in bfloat16 under bfloat16
     autocast, as the SDK runs on the L4 GPU that generated the training embeddings; small numerical differences
     between devices are possible.
  10. Any Freeze 1 amendment (F3c), with its reason and whether a test-side read happened under the superseded record.
      The report binds the recorded Freeze 1 by SHA-256, and that record names the superseded one; the report tool
      does not render this item itself, so the written results add it from the amended record.

## F5. Records and closeout

- Before F3a: a dated log entry with the selection and `E_sel`.
- Freeze 1 and Freeze 2: ledger entries and log entries.
- After F4: ledger results, a log entry, STATUS, README and handoff.
- Campaign closure only by the maintainer's decision, through the closure procedure.

## F6. Cost and time (refreshed 2026-10-12, log v3-043)

- **Refit session** (the only VM use): the 50-epoch refit takes about 0.6–0.9 h; with the reload check and session
  overhead about 1.0–1.5 VM hours, $0.9–1.3 at $0.8586/h. Up to 2.25 h ($1.9) with the one unchanged rerun, which may
  need a second start (`--hours 2` allows about 117 minutes).
- **Test side on the workstation CPU** ($0 of VM time; estimates, not measured):
  - label-blind preparation: about 10–20 minutes (all 7,920 training rows took 21 minutes on two capped cores; the
    training reference chains load in about 6 minutes);
  - test ESMC (measured on three training chains in log v3-044):
    - the ESMC-600M weights are now cached (revision `e4d83bc7…`, `esmc_600m_2024_12_v0.pth` `8ef856e1…`, 2.3 GB);
    - a 6 GB memory cap instead of the usual 3 GB: the peak was 5.25–5.38 GB, set by model loading (the float32 load
      and the bfloat16 cast); keep about 6 GB free (close PyCharm if needed); chains longer than 497 residues were not
      measured;
    - about 1,000–1,300 unique context sequences (the 1,500 fold-1 rows have 1,049 of at least 30 residues), about
      1.5–3 hours on two cores. It took 3.4 s for a 274-residue chain and 14.9 s for a 497-residue chain, and very
      long chains add more; this i7-1165G7 has no native bfloat16 instructions;
    - outputs about 1–3 GB on Data1 (about 82 GB free);
  - prediction about 10 minutes (the 1,500 fold-1 rows took 9.6 minutes); the report takes seconds.
- **Storage:** $0.55 per calendar day until F completes and the disk is retired.
- **Campaign total to finish:** about $42.8–48.2: about $40.8 spent (estimate at 2026-10-12 17:30 UTC; no VM use since
  11:18:47 UTC), the refit $0.9–1.9 and storage for 2–10 days $1.1–5.5; within the recorded $55. The 10-epoch path
  no longer applies (not adopted).

## F7. Decisions for the maintainer (none inferred)

One list, in order; the handoff, the log and STATUS use the same one.

**Status 2026-10-12 ([log v3-045](log.md#v3-045)):** items 1–5 are decided (the plan approved, the check done, the
commit and the refit's GPU start authorized, plan.md aligned). Item 6, the three later go-aheads, is pending.
The refit ran and was receipted ([log v3-046](log.md#v3-046)): checkpoint `a0007846…`. Freeze 1, F3b, Freeze 2, the single
test pass and the report followed ([log v3-047](log.md#v3-047), [log v3-048](log.md#v3-048)): step F is complete.

1. Approve or amend this plan (draft v3 with the implementation notes of log v3-042 and v3-043), including:
   - the two-freeze order, the epoch binding (`E_sel` = 50), and the F4 go-ahead with its campaign-wide claim;
   - the clean-subset rule as implemented: C1; C2 under A1, where an identical eligible chain always matches;
     `C2_unverifiable`; C3 with its definition of the previously opened PDB IDs; the short-chain limitation kept and
     disclosed, with no short-chain exclusion;
   - the label-blind test-side reads as described in F3a and F3b (label-bearing cells parsed and discarded unused);
   - the workstation-CPU test ESMC route in bfloat16 with the requirements of F6, and the Freeze 1 amendment
     procedure of F3c;
   - the plan.md alignment of item 5.
2. Done ([log v3-044](log.md#v3-044)): the training-side CPU ESMC check passed on three chains fixed in advance (mean
   per-residue cosine with the stored GPU embeddings 0.99979–0.99982, minimum 0.99941), so the bfloat16 CPU route
   stays. It exposed one defect in the specification's `storage_dtype` field, which is fixed and covered by a test.
3. Commit (and decide whether to push) the v3-038 to v3-044 work. `bind-selection`, `build-manifest` and
   `record-freeze1` refuse until every bound step F file is committed unchanged.
4. The refit's GPU-start authorization: one session, `vm-start --confirm --hours 2`, after `bind-selection`,
   `build-manifest`, `record-ceiling` and, after the logged decision, `record-authorization` (costs in F6).
5. On approval, align [plan.md](plan.md#steps):
   - the step F sentence "Before any test access, freeze … the test-row and clean-subset rules with membership
     hashes …" becomes: "Before any test-side read, freeze the primary report, the test-row and clean-subset rules,
     the preparation code, the refit checkpoint, seed and epoch count, no ensemble and the calibration rule; after the
     label-blind preparation and before any prediction or label read, freeze the derived memberships by hash; record
     both in the DATASETS ledger.";
   - the clean-subset summary names the explicit rule (training PDB ID; an identical or near-copy training chain
     under A1, or unverifiable context chains; previously opened PDB IDs; chains under 30 residues never compared,
     a disclosed limitation).
6. Later, three logged go-aheads:
   - Freeze 1, then F3b on the workstation CPU (the ESMC weights are already cached);
   - Freeze 2, after the label-blind preparation;
   - the single test pass (`predict --log-entry v3-NNN`), then the report.

## F8. CPU readiness checks (synthetic or training-side data only)

1. Selection binding: an E2 final-selection artifact and the A4 fallback each
   resolve to the right cell; mismatched hashes, incomplete assessments and a
   true `held_out_test_accessed` refuse.
2. Four-, five- and six-class winners each produce the right label scheme,
   multiplier flags, native metric set and Class VIII collapse.
3. Refit command: no fold or validation flags, `--val-fraction 0.0`, terminal
   rule kept, `--epochs E_sel`; the trainer's own parser accepts it; full-cohort
   multipliers equal an independent computation; the refit identity is complete.
4. A tiny CPU refit smoke on a training-side fixture: no validation
   dependency, terminal checkpoint written at the requested epoch, receipt and
   binding produced, reload check passes.
5. Label blindness: the preparation output (row list, dispositions, flags,
   hashes) is identical when every label field of the fixture is blanked or
   permuted.
6. Source-row accounting: a fixture with duplicates, missing and unparsable
   structures and LINK rows gives exactly one disposition per source row,
   summing to the source count.
7. Unscorable-row denominators: synthetic predictions with unscorable rows give
   the hand-computed recalls and BA, with "N of M scored".
8. Report recovery and one-shot: deleting the report and rerunning reproduces
   it from the immutable predictions byte for byte; a second prediction pass
   refuses; a different checkpoint refuses.
9. Epoch binding: a 50-epoch and a 10-epoch schedule decision each give the right `--epochs`, receipt fields and run
   directory. Without a decision artifact, the epochs are 50. A 10 refuses without an adopting artifact for the
   selected cell, and a checkpoint whose recorded epochs differ from `E_sel` refuses in F3 and F4.
10. Freeze order: the preparation tools refuse without the recorded Freeze 1 (canonical path, `v3-NNN` go-ahead, no
    placeholder) binding their exact code, and after Freeze 2 or the claim; the prediction tool refuses without the
    recorded Freeze 2 matching every prepared file and the recomputed embedding inventory, and a second pass into any
    directory; the report accepts only the claimed predictions; no component other than the report uses a label value
    (a test fails if any other module names a label column).

## Implementation notes (2026-10-12, log v3-042)

Prepared on CPU under the maintainer's preparation-only authorization (log v3-042) and corrected in one CPU round
([log v3-043](log.md#v3-043)); the rules above include that round's corrections.
As of log v3-043 (all executed later, log v3-045 to v3-048): **Not run:** no GPU start, refit, test-side read, test
embedding or prediction. **Not recorded:** no
write-once record (F1 selection, manifest, ceiling, authorization, Freeze 1 or 2, prediction claim); the F1 record
was dry-run only.

**Specification.** [`step_f_spec.json`](step_f_spec.json) holds the rules as data:

- selection and refit;
- test-row rule and test inputs, with the source CSV SHA-256 `ec6427ad…` and 1,488 rows;
- clean subset, label file and report;
- prediction guard, disclosures and freeze order.

Its SHA-256 is pinned in `pmm_v3_test_prep.py` and `pmm_v3_step_f.py`; any other file is refused.
The [playbook section](../../../METAL_TRAINING_PIPELINE_PLAYBOOK.md#v3-step-f-stage-6b-refit-and-the-one-test-pass-tooling-prepared-2026-10-12-not-approved-not-run)
lists the tools and commands.

**Choices made in the implementation:**

- **`E_sel` = 50 only.** The selection is four-class, so the refit builder accepts only the
  four-class target and 50 epochs; any other count is refused. The five- and six-class branches
  and the 10-epoch path of F8 items 2 and 9 are not needed: the decision artifact is
  `S10_20261012T111859Z`, which did not adopt 10 epochs.
- **Refit location.** The refit runs through the campaign lanes: lane 0, run directory
  `lanes/lane0/runs/stage6b_refit__gvp_late_fusion__four_class__baseline__e50__seed42`. This uses
  the step C admission, claim, host-pull and evidence logic instead of a separate `step_f/refit`
  folder.
- **Weights.** The full-cohort multipliers are computed from the frozen cohort: Mn 0.7316,
  Cu 5.3921, Zn 0.9102 and Class VIII 0.7413, from counts 2,528 / 343 / 2,032 / 2,495. Against
  the fold-1 command, only the fold, validation-export, selection-metric, multiplier, run-name and
  identity values change.
- **Reload check.** It runs in the refit process right after training, on the same device. It
  checks the configuration and identity, finite parameters, and that the training-set common-four
  BA re-evaluated on all 7,398 ions matches the logged epoch-50 value within 0.001. It writes
  `refit_reload_check.json`. Failing it makes the refit `failed_verification`, which allows one
  unchanged rerun. `reload-check` reruns it on CPU.
- **Label blindness.** The site manifest is read only through its allowlisted columns, and the
  source CSV only through `pdbid` and `residueid_ion`. Label-bearing cells pass through the CSV
  reader and are discarded record by record; they are never kept, used or written. The resolver
  uses the residue's metal symbol in memory only to decide whether the residue is a supported
  ion. It never stores the symbol, and the stored outputs carry a hashed `ion_key` instead of an
  element, residue name, atom name or readable ion ID.
- **`ion_key`** (log v3-043). The SHA-256 of the structure content hash, model, chain, residue
  number, insertion code and alternate location: stored identifier columns only, so it adds no
  information. The first version also hashed the atom name, which spells the element of a metal
  ion; trying the six supported metals recovered the element of every training row. A scorable
  target is the only supported-metal atom at its residue, so the key stays unique.
- **Row accounting** (log v3-043). `check_row_accounting` runs inside every preparation: each
  source row once, in order, with one disposition; each alias points to the earliest row of its
  ion and has the same identifiers and coordinates; distinct keys plus aliases equal the scored
  rows; unscorable rows carry no key. The summary counts physical ions and aliases.
- **Prediction.** Pockets are built from in-memory bindings that are re-resolved from the
  structures and checked against the prepared `ion_key`. Their label field is cleared, and a
  custom inference loop keeps every row (the trainer's evaluation helper drops rows without a
  label). Aliases share their primary ion's prediction, and unscorable rows carry none.
  - A process-wide read guard blocks every file of the test directory except
    `structure_manifest.csv`, every file of the dataset root, and the source CSV.
  - The graph and parse caches are disabled.
  - External features come from an empty directory, as in training.
- **Report.**
  - Lines: the full test, LINK-free, training-eligible, and the clean subset.
  - Rows with a label code outside the four classes are listed and excluded.
  - PDB-entry bootstrap: each line uses `numpy.random.default_rng(42)`, 10,000 resamples drawn in
    chunks of 1,000. A resample missing a class gives no BA and is left out of that statistic's
    interval; such resamples are counted.
  - PMM wording appears on the full-test line only.
  - `report.json` and `report.md` are byte-reproducible; the receipt with times is written
    separately.
- **File names.** The reconstruction stores each PDB entry once per labelled chain, as hard links to identical
  content. The training cohort binds each entry to one canonical name (`pmm_ion_cohort.canonical_structure_names`).
  The test side keeps the site manifest's name throughout: in preparation, test ESMC file names and prediction.
  Physical ions and aliases are keyed by content SHA-256, so identical files give identical inputs and the same ion.
- **Where F3b and F4 run: the workstation CPU** (the maintainer's choice, log v3-043). The preparation,
  test ESMC generation, prediction and report run there; no test file goes to the VM, which only
  refits. `test-embeddings` refuses any device but the CPU. The requirements and times are in F6.
- **Test ESMC precision.** The training embeddings were generated on the L4 GPU, where the ESMC
  SDK casts the model to bfloat16 and runs it under bfloat16 autocast (training sidecars:
  `model_dtype` bfloat16, storage float32). On the CPU the SDK does neither, so the tool casts the
  model to bfloat16, sets the same autocast, stores float32 and records the emitted dtype, device
  and model dtype; Freeze 2 refuses other settings. Compared on three training chains (log
  v3-044): mean per-residue cosine with the stored GPU embeddings 0.99979–0.99982 (minimum
  0.99941), mean absolute difference 6.2–7.0e-4, about 2% relative L2. A float32 diagnostic was
  only marginally closer (5.6–6.2e-4), so bfloat16 stays.
- **Records that gate the test side** (log v3-043; `pmm_v3_test_prep.check_freeze1`,
  `check_freeze2`, `claim_prediction`, `check_prediction_claim`). The runner and the report read
  the freezes and the claim only under `<campaign>/step_f_evidence/`. A record needs its `kind`,
  a `v3-NNN` go-ahead, no template placeholder and this specification. Freeze 1 must hold the
  exact SHA-256 of the code that runs: the runner and preparation module for F3b and F4, the report
  and preparation module for the report. Freeze 2 must name this Freeze 1, bind exactly the nine
  membership files and the preparation summary, and match the recomputed embedding inventory and
  every payload and sidecar. The preparation and the embeddings record the Freeze 1 they were made
  under.
- **Code binding across the refit and the test side.** `record-freeze1` refuses step F code that
  differs from the refit manifest's (including the runner that the refit identity names), unless
  it records an amendment; F1, the manifest and Freeze 1 are written only when every bound step F
  file is committed unchanged at the commit they name.

**Clean-subset rule, explicit** (to confirm with F7 item 1). A source row receives every code
that applies, and it belongs to the clean subset only if it has no code:

- **C1 `C1_training_pdb_id`:** its lowercase PDB ID is the PDB ID of any of the 7,398 training
  ions.
- **C2 `C2_near_copy_chain`:** a context chain of the row (at least 30 residues) is identical to,
  or a near-copy of, a training reference chain of at least 30 residues under the A1 rule (an
  identical sequence always matches, as A1 grouping links identical sequences; otherwise identity
  ≥ 90% of the shorter chain plus its internal insertions, with the 5-mer prefilter).
  - The identical-sequence clause was added in log v3-043. A chain with more than 10% non-standard
    residues (mapped to X, never counted as identical) cannot reach 90% even against itself. On
    the training side this is 1 of the 1,049 fold-1 context sequences (4uer chain 9, 51 of 71
    residues standard).
  - Context chains are the chains with a residue within 10 Å of the target ion, as first-model
    sequences.
  - Reference chains are every context chain of the frozen training ESMC plan, checked against
    the cohort SHA-256.
  - Limitation, kept and disclosed (the maintainer's decision, log v3-043): a chain under 30 residues is
    never compared under A1, so it cannot exclude a row, even when the same peptide is in training.
    On the training side, 2 of the 1,500 fold-1 rows have only such chains. No short-chain
    exclusion is added; the preparation summary counts such rows, and the report discloses the
    limitation.
- **`C2_unverifiable`:** the row has no context chain (every unscorable row), or a context chain
  without a first-model sequence. Such a row cannot be certified and is excluded.
- **C3 `C3_previously_opened_pdb_id`:** the PDB ID is among the structure file names of
  `DeepMzyme_Data/train_and_test_sets_structures_exact_pinmymetal/test`. The file count must
  equal the ledger's 316, and only names are read, never contents.
- **Not an exclusion:** the exact Zenodo test itself (disclosed).

The clean subset is secondary and "not pristine", is never compared with PMM, and counts its
unscorable members as errors (none can be members under C2).

**F8 status** (synthetic or training-side data; [log v3-042](log.md#v3-042), [log v3-043](log.md#v3-043)):

- Items 1, 3–8 and 10 are covered by `tests/test_v3_step_f.py`: 38 tests, including the two opt-in synthetic fits,
  run as authorized after the corrections.
- Items 2 and 9 are reduced to refusals, because only four-class and 50 epochs are bound.
- Real training-side checks:
  - The first run, `$DATA/audits/f_readiness_20261012T122926Z/`, is preserved unchanged. Its stage C passed by a
    recorded reassessment, not as run, and its stage B ran with an earlier runner.
  - The scripted rerun on the final tools is `$DATA/audits/f_readiness_20261012T170111Z/`. The script records its own
    SHA-256 and computes every pass flag in one run. It covers:
    - the refit plan on the real configuration;
    - label-blind preparation of all 7,920 training rows against the cohort's own dispositions, with the row
      accounting and the ion-key partition;
    - the clean-subset machinery at scale with the identical-chain clause;
    - label-free prediction of the 1,500 fold-1 validation ions with the real fold-1 checkpoint against the saved
      predictions.
  - Results: in `f_readiness_20261012T170111Z` (`5b1ce5f0…`), A, B and D passed. C failed as run because of a scope
    error in the script's sequence-count check; every row-level check passed. The corrected script reran stage C in `$DATA/audits/f_readiness_20261012T174408Z/` (`step_f_readiness.py` `b73ece28…`, `readiness_checks.json` `7e61a339…`) and it passed: all 1,049 fold-1 context sequences of at least 30 residues are identical to a reference and flagged (1,048 in the first run; 4uer chain 9 is now caught), all 1,498 rows with such a chain are excluded by C2, and only the two short-chain rows (5wlm, 6wbe) remain. That script combines the earlier A, B and D (verified by SHA-256 and identical tool hashes) and computes ready: true.
