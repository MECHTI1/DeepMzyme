# PMM replay diagnostic v1

This bounded diagnostic investigates [TECH-023](../FOLLOW_UP_TECHNICAL_ISSUES.md#tech-023--full-fit-gvp-independent-replay-exceeds-the-frozen-probability-tolerance).
It reuses the completed six-class Only-GVP fold-0 fit, seed 42. It performs no
training, checkpoint selection, test access, or campaign promotion. Original
failed replay artifacts and the absolute `1e-6` certification rule remain intact.

## Frozen evidence and input comparison

Use the selected checkpoint, saved normalization, batch size 16, validation
order and FP32 evaluation from
`only_gvp__six_class__none__fold0__seed42`. Verify the training-source identity
`adc95c42261448dc9d35572a138a3b8349de79124d9708618f511c27a848dd23`
and retain hashes of the checkpoint, configuration, cohort, fold and predictions.
The diagnostic is outside `src/` and `scripts/`, whose bytes define the frozen
training identity; it records its own source hash separately.

Reconstruct all 1,492 validation graphs with raw-graph caching disabled. Compare
their tensor bytes, dtypes, shapes and order with original-fit cache entries in
namespace `117d3f2d78b3429930bcc78491821c7352c89d5e488b7df0ab3715b32c5c0f0f`,
created between `2026-09-27T18:14:00Z` and `2026-09-27T18:54:00Z`.
Verify cache payload checksums. Bind fresh graph content to frozen validation
identities and report ambiguous content-identical matches explicitly.

Compare saved-normalization outputs and collated batches as well. Only
`ec_group_id` and `ec_sample_weight` may be explicitly excluded from the metal
prediction comparison, with a recorded audit and guards confirming a standalone
metal task without an EC prediction head. All other fields must match. Report
that preserved cache comparison is retrospective evidence: original in-memory
training tensors were not separately fingerprinted when the fit ran.

Save one immutable input snapshot and its hashes. A mismatch stops numerical
interpretation; diagnose the differing inputs before drawing conclusions.

## Predeclared numerical experiment

Run two fresh processes per condition, with five complete forward passes in each
process: **20 evaluations total**, all using the same snapshot and checkpoint.

1. Original FP32 evaluation settings, including its recorded determinism setting.
2. Strict deterministic algorithms, with `warn_only=False` and the required CUDA
   workspace configuration set before CUDA initialization. Unsupported operations
   are diagnostic failures to report, never silently downgraded to warnings.

Preserve every pass, including failures. Record runtime/library/device versions,
effective numerical flags, logits, CPU-FP32 softmax probabilities, native and
collapsed-four predictions, and prediction margins. Compare passes within and
between processes and with the original export and both failed replay exports.
Verify model state and fixed inputs before and after evaluation. Do not repeat
until a favorable pass, average predictions to conceal differences, or rewrite
historical probabilities.

## Interpretation and resource boundary

Input equality plus variable outputs on those identical inputs would establish
runtime numerical variation. Improved repeatability under strict determinism
would further localize its source. Neither observation alone changes the
certification rule. Any replacement policy requires an independently justified
numerical bound, a separate versioned receipt, tests and an explicit consumer;
it must apply consistently to all compared configurations. Legacy receipts must
retain their original meaning.

Prepare and test locally before starting the existing L4 VM. The intended
diagnostic allocation requests 45 minutes under unchanged controller caps,
yielding approximately 42 usable minutes after controller overhead. Reserve
15 minutes for persistence and closeout and apply the existing 1.25 execution
forecast margin. Show verified price and actual hard stop before allocation;
stop promptly after copying and verifying results. This does not activate the
deferred full-grid budget or authorize another provider or replacement VM.

PyTorch documents both [CUDA reproducibility limits](https://docs.pytorch.org/docs/2.11/notes/randomness.html)
and [floating-point numerical accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html).
Those general explanations motivate the diagnostic; they do not establish the
cause of this particular mismatch.
