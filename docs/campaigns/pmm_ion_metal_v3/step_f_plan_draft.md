# PMM ion metal v3 — step F execution plan (draft v2 for approval)

Status: **draft, not approved; nothing in it is built or run.** Corrected on
2026-10-08 ([log v3-030](log.md#v3-030)) from the session draft of 2026-10-08
08:45 UTC (`audits/session_notes_20261008/step_f_plan_DRAFT.md` on the campaign
data root, kept unchanged). The rules stay owned by [plan step F](plan.md#steps),
the [A4](assessment_spec.md) and [E2](e2_assessment_spec.md) specifications and
the [DATASETS ledger](../../DATASETS.md#test-use-ledger); this page plans their
execution. No refit, test-data access or GPU start is authorized by it.

Corrections against the session draft: the selection input is the E2 final
selection (the session draft read only A4); the refit command is genuinely
validation-free (the session draft kept `--export-validation-predictions`,
which the trainer's parser refuses without a validation split, and a
validation selection metric); four-, five- and six-class winners are handled
explicitly; the refit checkpoint is bound by hash through test reporting;
label blindness, source-row accounting, unscorable-row denominators and report
recovery from immutable predictions become tested requirements; costs use the
current storage (the failed-fallback leftovers were deleted on 2026-10-08).

## F0. Gates

1. Original step E complete: 44 units complete and replayed; the final A4
   assessment copy has a non-null `stage6_selection` and
   `held_out_test_accessed` false; evidence copied; log entry.
2. Step E2 complete under the frozen E2 specification (36 fits verified, all 17
   declared comparisons reported, a machine-readable final selection), or a
   dated log entry by which the user cancels E2.
3. Label `both_results_secondary` recorded (done, log v3-022).
4. The operative ceiling, recorded in execution controls, covers the refreshed
   F forecast including storage; otherwise the AGENTS budget question.
5. The CPU readiness checks of F8 pass on synthetic or training-side data.
6. The user approves this plan; later, separately, the DATASETS ledger entries
   before F3 and the single test pass before F4.

## F1. Selection input (CPU; no fit; no judgment)

- Input: the E2 final-selection artifact (selected cell, whether it replaced
  the original selection, the original A4 selection, E2 specification and
  manifest SHA-256 values, evidence references). Only if the user cancelled E2
  by a dated log entry: the A4 `stage6_selection.selected` alone.
- The step F record binds the artifact path and SHA-256, the original E
  assessment SHA-256, the selected family, target and recipe, and the recipe
  definition from the hash-frozen runner. Refused: a missing artifact, a hash
  mismatch, `held_out_test_accessed` not false, an incomplete assessment, or a
  selection that is not a cell of the campaign grid or of E2.

## F2. Stage 6B full non-test refit (GPU; one 50-epoch fit)

- Tooling (additive files; no `src/` change, so every campaign hash stays
  valid): a workstation launcher reusing the step C/D lane, admission, wait,
  pull, recovery and evidence logic, and a VM-side refit module that builds the
  command from the campaign's command builder for the selected cell.
- Validation-free command, derived from the selected cell's fold command:
  remove `--n-folds`, `--fold-index`, `--fold-membership-csv`,
  `--fold-membership-sha256` and `--fold-split-source`; remove
  `--export-validation-predictions` (refused without a validation split) and
  the validation selection metric `--selection-metric val_metal_balanced_acc`
  (with no validation the trainer's descriptive metric defaults to
  `train_loss`); add `--val-fraction 0.0`. Unchanged: `--checkpoint-rule
  terminal`, 50 epochs, cosine schedule, seed 42, batch 16, FP32, the
  recipe's flags, all model, feature and input options, no test flags
  (`--run-test-eval` and `--allow-final-refit-test-eval` absent; inference
  happens in F4 outside the trainer).
- Weighting policy preserved and recomputed on the full non-test cohort (all
  7,398 training ions): manual common-four equalized multipliers
  `w_c = N / (4 n_c)` by the campaign's weight function; `four_class`:
  Mn, Cu, Zn, Class VIII; `five_class`: Mn, Cu, Zn, Fe and Class VIII (Co+Ni),
  Fe and Co+Ni both at `w_VIII`; `six_class`: Mn, Cu, Zn and Fe, Co, Ni each at
  `w_VIII`. Normalization statistics are fitted by the trainer on the full
  training partition and saved with the run; nothing is refitted later.
- Identity: a refit identity (selection SHA-256, family, target, recipe,
  cohort SHA-256, weights SHA-256, source and runner hashes, seed, epochs,
  "full non-test cohort") replaces the fold identity; run directory
  `step_f/refit/stage6b_refit__<family>__<target>__<recipe>__seed42`.
- Receipt and binding: fit completed with 50 history rows, terminal rule,
  selected epoch 50, no validation partition, `test_report` none; SHA-256 of
  the terminal checkpoint and of `run_config.json` written into the step F
  record. Host pull with manifest and acknowledgment, evidence copy. Replay
  analogue without validation data: a CPU reload of the bound checkpoint with
  configuration equality, finite parameters and a re-evaluation of the
  epoch-50 training-set common-four BA against the logged value. F3 and F4
  refuse any checkpoint other than the bound one.
- Failure: one unchanged rerun; a second failure stops F for diagnosis.

## F3. Label-blind test preparation (after the ledger entries)

- Rows: PMM's test side (1,488 source rows) resolved with the training
  cohort's input contract under the v3-012 rule: every reconstructable row is
  scored; symmetry-LINK rows are flagged, not dropped; rows that cannot be
  scored keep an explicit disposition.
- Genuine label blindness: the builder reads structures, sequences and
  identifiers through a column allowlist that excludes every label field; no
  inclusion, flag, de-duplication or alias decision reads a label or the metal
  element. The model inputs already exclude the metal identity (no metal
  node, residue-only readout).
- Source-row accounting: each of the 1,488 rows receives exactly one
  disposition (scored, scored with the LINK flag, or unscorable with a reason);
  the counts sum to 1,488 and the hashed row list is frozen before inference.
- Clean subset (secondary): exclusions by training PDB ID, by A1 near-copy
  chains against the training cohort, and by the list of previously opened PDB
  IDs, all label-free; hashed with reasons.
- Test-side ESMC embeddings for every context chain, generated label-free with
  the existing plan, receipt and inventory mechanism.
- Recorded in the DATASETS ledger before any prediction: the hashes of the
  scored, clean-subset and exclusion lists; the bound checkpoint SHA-256 and
  seed; no ensemble; no calibration; label `both_results_secondary`; what
  test-side material was read, when and by which tool.

## F4. The one test pass and its report

- Inference with the bound checkpoint and the saved normalization: softmax,
  five- or six-class probabilities summed into Class VIII before the argmax;
  no calibration, ensemble or threshold change.
- Immutable predictions first: `test_predictions.csv` (one row per source row,
  with disposition, flags, native and common-four probabilities and the
  checkpoint SHA-256), written once and hashed. The prediction tool refuses a
  second pass once predictions exist.
- Report second, as a pure function of the frozen predictions, the frozen
  label file and the frozen row lists: full test "N of 1,488 scored", with
  unscorable rows counted as errors in every class-recall denominator;
  common-four BA and Mn, Cu, Zn, Class VIII recalls; native metrics and Fe/Co/Ni
  (six) or Fe and Co+Ni (five) recalls for a five- or six-class winner; the
  LINK-free sensitivity line; PDB-entry-group bootstrap intervals for
  uncertainty only; the clean subset from the same predictions; PMM's
  published values beside the full-test numbers in the allowed wording only;
  labels "exact, possibly overlapped PMM test; includes rows from previously
  opened tests" and "both results secondary" with the recorded disclosures.
  Rerunning the report reproduces it byte for byte without new predictions.

## F5. Records and closeout

Before F3: ledger entries and a dated log entry with the selection. After F4:
ledger results, log entry, STATUS, README, handoff; campaign closure only by the
user's decision through the closure procedure.

## F6. Cost and time (refreshed 2026-10-08, log v3-030)

Refit alone on the VM about 0.6–0.9 h, test embeddings and inference about
0.2 h, session overhead about 0.3 h: about 1.0–1.5 h of VM time, up to 2.25 h
with the one-rerun allowance, $0.9–1.9 at $0.8586/h. Storage $0.55 per
calendar day until F completes (campaign disk and archive snapshot).

## F7. Decisions for the user (none inferred)

1. Approve or amend this plan.
2. Confirm the clean-subset exclusion rule and its previously-opened list.
3. Confirm that F3 may read test structures and sequences label-blind, with
   the test ESMC generation in the refit session.
4. Two later go-aheads: the ledger entries before F3; the single test pass
   before F4.

## F8. CPU readiness checks (synthetic or training-side data only)

1. Selection binding: an E2 final-selection artifact and the A4 fallback each
   resolve to the right cell; mismatched hashes, incomplete assessments and a
   true `held_out_test_accessed` refuse.
2. Four-, five- and six-class winners each produce the right label scheme,
   multiplier flags, native metric set and Class VIII collapse.
3. Refit command: no fold or validation flags, `--val-fraction 0.0`, terminal
   rule kept; the trainer's own parser accepts it; full-cohort multipliers
   equal an independent computation; the refit identity is complete.
4. A tiny CPU refit smoke on a training-side fixture: no validation
   dependency, terminal checkpoint written, receipt and binding produced,
   reload check passes.
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
