# RING continuation pre-execution review

The original local audit (PID 189013) exited with `status: passed` after
1123.6842095851898 seconds. It was not restarted. Its cache-audit SHA256 is
`b2414386f46e5372e631dc040f5f717c2674f957c04fd9058cd772814abe5189` and was
verified against the saved `training_cache_audit.json`.

All 1181 training and 208 validation pockets have identical node/site inputs
between the two arms. RING annotates 34573 training and 6057 validation
undirected radius pairs, with no added pairs. Both independently fitted
normalization artifacts share SHA256
`bdf6af42b9ef1d6941ff5a9fff0429e4f27669ec0f9ffe714d7d702e63049177`.
All 1304 retained structures have the required pinned ESM, external and RING
files; CPU readiness hashes ESM tensors without deserializing them.

The frozen source archive remains
`7ad27026c2baaefe9b566d80025455d852c63d3acf8198113743f5685ae6523a`.
The checkout had advanced to user commit
`743d9a225b737156b521d3a9d312c6d0660ef2a2` when this continuation began.
All 141 corresponding source, notebook, requirement, script, test and document
files match the frozen snapshot. The historical snapshot retains its original
base-commit identity; it was not rebuilt or relabeled. No commit/push was
performed by this continuation.

The generated `local_plan/` contains 20 commands: four one-epoch timing
smokes, eight required Only-GVP fits and eight optional graph-level late-fusion
fits, each full fit using 50 epochs. An independent read-only review verified
LR/seed/off-on order, fixed direct-four targets and native-six eligibility,
PDB-grouped 15% validation with split seed 42, geometry shell roles, identical
paired controls, and absence of held-out inputs/evaluation. The expected split
is byte-identical to the original certified parent.

The immutable handoff retains the original two closed allocation intervals:
19263.44616508484 allocated seconds, 12561.789792060852 normal-work seconds,
and 308.3893711566925 retry seconds. Original caps and the 1.25 complete-family
forecast margin remain unchanged. New allocation and actual stop costs must
be read from the separate continuation ledger, not inferred from this note.

Host preparation and independent operator review passed. The actual named G4
allocation is `deepmzyme-metal-ring-20260915`; its receipt and start timestamp
are under `colab/`. Its watchdog is active. Bootstrap passed in
213.793625831604 seconds and preserved stock PyTorch 2.11.0+cu128 on an RTX PRO
6000 Blackwell GPU with sm_120 support. Remote whole-cohort plus actual-data
CUDA readiness was running when this review was recorded. This note is not
training evidence, a terminal-state receipt, or a teardown certificate.
