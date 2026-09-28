# PMM replay diagnostic v1.1: disabled EC label audit

This amendment follows the preserved failed input comparison from
[v1](pmm_replay_diagnostic_v1.md). It changes no checkpoint, metal label, graph
feature, prediction, probability threshold, or numerical experiment. No GPU
forward pass had run when the amendment was made.

All 1,492 fresh validation graphs have exactly one original-window cache
candidate with identical tensor fields except `y_ec`. The cached field is
an int64 scalar vector containing `0`; independently reconstructed replay
contains `-1`. Both `ec_group_id` and `ec_sample_weight` already match.
The selected model is standalone metal with `predict_ec=False` and no EC
prediction head. The frozen model's EC target/loss/contrastive accesses are
guarded by that disabled flag; metal targets remain exact.

Preserve the failed manifest and original diagnostic source. To avoid repeating
the expensive structure/graph reconstruction, recover each fresh graph from its
unique cache candidate, set only `y_ec` to the recorded fresh value, and require
every resulting field descriptor to equal the independently captured fresh
descriptor, including dtype, shape, ordering and SHA-256 of tensor bytes.
Verify the original cache time window and checksum again. No structure-derived
feature may be replaced, ignored or recomputed differently.

Use saved checkpoint normalization and the original validation order and batch
size. Require all normalized/collated fields except the explicitly audited,
disabled EC target to match. Verify exact CPU metal-logit equality for cached
versus recovered inputs, with all evaluated examples and results recorded.
Never allow this exception for EC/joint training, an enabled EC head, or
`y_metal`. Report the nonpredictive difference; do not describe the entire
unfiltered graph as byte-identical.

Write a new preparation directory with the failed-manifest hash, old and new
diagnostic source hashes, cache identities and recovery procedure. Do not
overwrite the failed preparation. The original two processes per condition,
five passes per process, strict deterministic controls and 20-pass experiment
remain fixed. The separate `pmm-screen-agreement-v2` probability rule remains
unchanged and still requires all diagnostic gates.

Reuse the current bounded VM session. Apply the measured admission margin and
900-second closeout reserve to the recovery command; no restart, extension,
new allocation or extra training is authorized by this amendment.
