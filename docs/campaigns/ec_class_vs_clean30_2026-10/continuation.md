# Continuation after Freeze 1b

Status: updated, 2026-11-02; preparation and the S1 amendment are published through `3154a7f`.
WP-13 is published at `10e0e11`; the execution extension is published at `589af0a`.
The recovery extension is published at `4fb4bdd`. All three authorized zones returned stockout; the pass is
closed and temporary resources removed ([evc-018](log.md#evc-018)). [G2 readiness](g2_readiness.md) records the
remaining capacity blocker and preserved preparation.
This document records no new compute authorization. First scientific
milestone: the complete fold-0 S1 comparison with CLEAN and its frozen stop-or-continue decision.

The route is **execution readiness → measured G2 smoke → Freeze 1c and G3 budget → fold-0 S1/gap gate →
conditional S2/S3/refit → independent folds 1–4 → Freeze 2 → authorized one-shot scoring**.

The [design](design.md), [Freeze 1](freeze1.md), [S1 amendment](s1_amendment/amendment.md), [decisions](freeze1_decisions.json),
[EC recipe](../../EC_TRAINING_PIPELINE_PLAYBOOK.md#ec-class-versus-clean30-campaign-recipe) and
[Plan](../../../Plan.md) retain their authority. Exact executable blocks belong in the EC playbook when their
interfaces exist. The actions below that lack an implementation are specifications, not runnable commands.

**1. Preserve the completed preparation and identify the real gaps**

Reuse the five certified training views, final/block-26 chain means, dry runs, overlay, ENZYME transfer list,
environment records, Freeze 1 and Freeze 1b. [evc-011](log.md#evc-011) owns their completion evidence and hashes.
Do not regenerate embeddings, repeat the alpha ladder, repeat the identity audit or rerun the full suite merely
because execution moves to another phase. New code and a changed runtime require their relevant checks.

| Item | Current state | Continuation requirement |
|---|---|---|
| Fold plans, command construction, verification, selection and gap gate | Implemented; amended S1 validated | Reuse prepared inputs; authenticate the published amendment on fits, selection and gap records |
| WP-13 timing/transfer runner | Published; CPU-tested ([evidence](wp13/verification.json)) | Measure under G2 with the published automation extension |
| Fold-0 S1 and CLEAN queues | CPU-verified ([record](execution_v2/verification.json)) | Fill measured G3 authority; later S2/S3/refit orchestration remains separate |
| CLEAN wrapper and private harness | Reused by the new serial queue | Exact input closure, F9 guards, ext4 staging and verified persistence precede each fit |
| GPU cost and throughput | Historical forecasts only | Measure under G2 before setting G3 ceilings |
| Final simulation | Integrity verified; targets not uniformly met | Retain the limitation and fixed rule; no retuning |
| Root STATUS | Stale, explicitly preserved by C-10 | Use campaign README/log; no unrelated root edit here |

The simulation maxima were 12% ordinary, 18% L7 and 14% L9, against targets of 10%, 15% and 15%.
The original fold-2 calibration still met its targets. Uniform control across folds remains unestablished, and
the earlier simulation had low power for a true +2-point gain. Neither a promoted lever nor a passed gap gate
establishes eventual superiority. Preserve the full interpretation in [evc-010](log.md#evc-010).
The separate fixed-alpha S1 characterization is complete; [evc-012](log.md#evc-012) records its low power,
null selection rates and bootstrap aborts. It provides no uniform error-control guarantee and is not retuned.

**2. Verify and publish the CPU execution package**

WP-13 is implemented in new `src/ec_clean30/phase3.py` and `scripts/run_ec_clean30_phase3.py`, with focused tests.
It reuses [commands](../../../src/ec_clean30/commands.py), [views](../../../src/ec_clean30/views.py),
[freeze checks](../../../src/ec_clean30/freeze.py) and
[CampaignExecution](../../../src/benchmarking/pmm_execution.py). No new scheduler, tracking service,
database, cloud controller or training architecture is needed.

| Implemented action | Behavior |
|---|---|
| `pack-view` / `unpack-view` | Package only the recorded fold-0 files, manifests, receipts and both chain-mean caches. Exclude old probe/cache contents. Verify checksum, safe members and modes on extraction; preserve the original dry-run receipt. |
| `timing-inputs` | Select exactly 1,000 rows from the exact official fold-0 training file by sequence hash, with a deterministic accession tie-breaker. Write FASTA, accession/hash map and source receipt. |
| `vm-probe` | Verify published freezes, clean pushed source, original source pins, relocated view and read guards. Use `dry_run(write=False)` and a separate VM receipt. |
| `timing-unit` | Build the frozen family commands, use separate timing identities, verify terminal artifacts and replay independently; record preparation, fit, export, replay and persistence time. |
| `lane-gate` | Run the declared serial repeats and explicit three-process batch; check scientific agreement, throughput and host limits. |
| `embed-timing` | Call the existing ESMC sequence builder and private ESM-1b harness on the approved timing inputs only. |
| `freeze1c` | Bind measured timings, lane outcome, added execution-code hashes and approved L3 ceilings to Freeze 1. |

`commands.run_unit` is synchronous. Run it in a thin child process owned by the existing execution helper so
process ownership, interruption recovery and persistence gates remain effective. The runner owns work; the
installed VM controller owns allocation and shutdown. A verified completed unit is reusable. An interrupted fit
restarts from its original seed in a new attempt directory; there is no optimizer-state resume claim.

Test archive round trips, traversal/link/duplicate refusal, checksums, permission enforcement, relocated read
guards, immutable original receipts, dirty/unpushed-source refusal, changed-pin refusal and absent freeze bindings.
Exercise run → verify → replay → independent persistence acknowledgment on CPU. Test interrupted-worker recovery,
duplicate-launch refusal, deadline refusal and failed acknowledgment. Test the lane arithmetic and refusal paths,
and ensure the emitted Freeze-1c ceiling mapping is accepted by the existing selector.

Then run the approved small training-only smoke-view route, targeted campaign tests, CPU smoke and the required
full regression suite once at the final implementation. Keep source unchanged during these checks. Reuse the
current project interpreter and workstation caps; temporary fixtures stay outside the scientific data tree.
Update the EC playbook with actual interfaces, this campaign's log and README together. Stage the package for
review; publication must precede any real remote action.

Completion gate: the entire local round trip passes, all added code is reviewable, no frozen source/input changed,
and the G2 package can be assembled without debugging on a billed GPU.

**3. Resolve transfer, storage and timing details before billing**

Use a pushed descendant of `693697d` containing the S1 amendment and WP-13, with ancestry and original Freeze-1
per-file hashes checked except for the exact authenticated amendment changes.
The older wording requiring an exact CP-E checkout cannot run new code. Pin added execution code separately;
preserve both completed freezes byte for byte. WP-13 must verify the S1 amendment and pass its chain to the
existing command builder, and provide separately authenticated execution-code pins for its added files.
The implemented [execution manifest](wp13/execution_manifest.json) binds exactly those two additions. Its verifier
authenticates S1 in a temporary sparse checkout of `3154a7f`, then checks every live original and added source pin.
A necessary edit to an existing pinned file requires a recorded amendment and affected revalidation.

Transfer only the fold-0 view archive, the 1,000-row timing inputs, the pushed code and the explicitly approved
private CLEAN setup. Supply the logged Freeze-1b hash; do not ship the private freeze folder, full overlay, other
fold views, union sequence map or campaign identity tree. L8 is already prepared: transfer the existing fold-0
cache and omit the old all-740-chain GPU regeneration task.

[G2 storage readiness](g2_readiness.md#storage-placement) owns the measured free space and one-fit forecast.
Keep weights, environments, the prepared view archive and inactive artifacts on Data1. Copy only one active
CLEAN fit's hot vectors and working files to ext4; links back to Data1 retain the measured I/O slowdown.
Preserve the existing 5 GiB reserve, check actual staged bytes before each fit, and verify completed outputs
at their durable destination before reclaiming active scratch. No frozen evidence or backup cleanup is needed.

Declare these proposed operational choices in the CPU package before measuring results:

- Timing recipes: fold 0, inner fold 0, seed 42; five epochs for each of F1–F5 and C-G; LR-grid index 1, matching
  the existing smoke helper. One five-epoch F4+L3 probe uses the corresponding unaugmented F4 timing as reference.
- Lane recipe: F4, inner fold 0, seed 42, the same LR setting, 30 epochs; two serial repeats and one batch of
  three identical-recipe repeats. Compare every lane with both serial references.
- Equal-work speedup: `3 × mean(serial full-unit wall time) / three-lane batch wall time`. Include verification
  and equivalent persistence work consistently. Do not compare two serial fits directly with three parallel fits.
- Retain the frozen limits: at most 1 BA point, at most 3 recall points for supported classes, at least 95%
  top-1 agreement and at least 1.2× speedup. Resolve and record the supported-class BA scope before measuring:
  the original lane recipe uses classes with at least ten validation proteins. This lane diagnostic is separate
  from the EC1–6 selection endpoint. Thresholds are not tuned on timing outcomes.
- Set numerical CPU, RAM and VRAM limits from the actual approved VM and runtime profile before the lane test;
  OOM, swapping, a read-guard breach or an exceeded declared cap fails the lane gate. Admission must fit the whole
  batch without assuming the speedup under test. A failure records serial execution; no new lane-count search.
- CLEAN gap-arm seed: 42 for every R1-inner and R2-inner fit. The existing recipe declares one seed but the reader
  does not enforce its ID; this fixes execution before results and introduces no seed search.

**4. Run one bounded G2 measurement after its approval**

Follow the [GCP runbook](../../GCP_GPU_RUNBOOK.md) through the installed controller. Before requesting a start,
inspect ownership and current resource state with its read-only status, report, preflight and price tools;
preview the applicable start/restore route. No live cloud state was inspected while writing this plan.

The existing G2 proposal is one session, about two VM-hours, at most $3. These are not demonstrated sufficient
runtime or new authorization. Local controller configuration currently permits four hours/$6 per session and
twelve hours/$18 daily; older documentation has different daily defaults. Do not edit configuration from stale
examples or treat its wider ceilings as campaign authority. An explicit two-hour controller request currently
allows approximately 117 minutes before remaining daily limits; setup consumes part of that allowance.

The concrete G2 request must contain the exact unit list, current gross rate, storage/transfer charges, allowed
start route, duration, provider stop deadline, output destinations and private CLEAN environment/harness/weights
approval. Include the three-process lane test as a narrow exception on the same single L4. Ordinary work stays
serial until that gate passes. No automatic provider, region, machine or budget expansion is included.

Admit complete work only with the existing 1.25 forecast multiplier and 900-second persistence reserve, measured
from actual allocation start. The provider stop remains separate. Setup, input checks, downloads, replay and
transfers count toward the session. Use automatic STOP, preserve results and verify provider shutdown; retained
storage can still incur charges. Google documents both [runtime limits](https://docs.cloud.google.com/compute/docs/instances/limit-vm-runtime)
and [billing while stopped](https://docs.cloud.google.com/compute/docs/instances/suspend-stop-reset-instances-overview).

Within G2, verify the relocated view and VM probe first; then run the declared family/L3 timings, embedding batches
and lane test. Timing identities remain excluded from selection. Persist and independently hash-acknowledge each
terminal unit or declared batch before admitting another. Stop the owned VM after results are saved.

If accepted work no longer fits, present exactly the established choices at a safe boundary: stop and close out
at the evidence reached, or continue with a recorded larger ceiling. Provide measured use, forecast and requested
increase. Do not silently exceed the ceiling or terminate a safe active fit solely because the forecast changed.

Completion gate: timing, replay, memory, lane and persistence evidence; verified shutdown; and a measured budget
proposal. A partial session is labeled partial, not a passed G2 package.

**5. Freeze the measured budget and launch the prepared Phase-4 queues**

Create Freeze 1c with the timings, serial/passed-lane choice, added-code hashes and per-fold L3 ceilings. Use the
existing selector contract `campaign_id`, `freeze1_sha256` and `l3_vm_hours_per_fold`; do not create a parallel
budget schema. Obtain G3 approval for the numerical ceilings and exact workload before any Phase-4 fit.

Forecast from measured per-family cost and applicable configurations, including CPU preparation, embeddings,
graph rebuilds, verification, replay, transfers, boot/idle time, session reserves and retained storage. The
181-per-fold/905-total fit counts are workload bounds, not spending authority. The old dollar tables remain
historical estimates. The later CLEAN estimate is approximately 160 serial workstation hours, based on earlier
timing rather than completed benchmark fits; reforecast from the first real gap fits.

The new `phase4.py` dispatcher and workstation transfer loop implement the exact 96 S1 fits, the two required
CLEAN embedding farms and eight serial CLEAN inner fits. Their [execution pins](execution_v2/manifest.json)
extend the published WP-13 record without changing frozen scientific source. Each queue handles admission,
verification, persistence and recovery; the GPU queues stop their owned allocation. Numerical G3 values remain
to be filled from G2, but no further dispatcher implementation is planned between G2 and this Fold-0 comparison.
S2/S3 and final refit orchestration remain outside this first-milestone package.

For every CLEAN prepare/farm/fit/export call, compute and test the F9 forbidden roots for that fold and step.
The wrapper defaults alone do not cover all campaign folders. R1-inner uses its recorded homology exclusions;
R2-inner uses the component-aware allowlist, including the specified several-class proteins. Preserve the private
harness and its published recipe. Keep active CPU farms/fits on internal ext4, fits serial under the existing
caps, and each distance map alone under 4.5 GB. Do not move CPU fits onto a paid GPU VM as an unpriced convenience.

Prepare the base and keyed masked embeddings required by the eight fold-0 gap fits first, from their exact prepared
training/query inputs. Preserve all approved masks and sequence hashes. Defer material needed only by other folds
or final comparator seeds until the first continuation gate; reuse verified matching embeddings later. This
changes work order, not the final arm list or data. No blanket full-pool transfer is permitted.

Completion gate: tested real-unit orchestration, CLEAN input guards, resource/backup capacity, frozen execution
identities, numerical Freeze 1c and G3 authority. No GPU should sit billed while this code is being written.

**6. Obtain the first meaningful fold-0 result**

Run all S1 work: six families × two LR settings × four inner folds × seeds 42/43 = **96 fits**. Controls and
apparently unnecessary branch families cannot be pruned under the frozen contract. Use the terminal checkpoint
and fixed schedule; curves are descriptive and do not stop epochs or choose checkpoints.

Apply the authenticated S1 fixed-anchor rule in [the design](design.md): F3, F5 and F4 each compare directly with
F2, with one fixed Holm family at alpha 0.04 and all guards relative to F2. F4 remains reachable even when neither
single-addition family passes. Apply the unchanged LR tie rule, then the approved 0.5-point family tie band.
F1 and C-G remain reported controls. Report every family, both CLEAN arms, BA, accuracy, class recalls and
uncertainty. Selection does not establish superiority over every alternative or incremental component benefits.

Run four CLEAN R1-inner and four R2-inner fits, seed 42, using the published 2,500-epoch recipe. Start these serial
CPU jobs while remote S1 runs, subject to local resource limits, to avoid waiting on CLEAN with an idle GPU.
The earlier approximately 2.9-hour R1-fit estimate is only a scheduling guide, not a new budget.

Before the gap decision, verify all 96 DeepMzyme fits and eight CLEAN exports. Each query set must exactly equal
its certified inner-validation membership before applying the common full-EC closure. Reject duplicates, extra,
missing or in-sample queries. Report the closure denominator and use the existing gap implementation.

Gate G4: continue when the primary R1 EC1–6 gap is at least −3.0 BA points, with the existing inclusive tolerance.
Below that threshold, present stop or continue under the unchanged frozen algorithm. R2 and seven-class results
are descriptive. Do not start S2 or folds 1–4 speculatively. Passing G4 is a cost decision, not a superiority claim.

**7. Continue only through the frozen, conditional search**

| Work after S1 | Required fits | Efficiency and validity rule |
|---|---|---|
| S2, winner F2 / F5 / F3 / F4 | 48 / 56 / 64 / 72 | Applicable levers only, independently against S1; L3 last |
| L9 | 0 | Reuse logits and per-fit training priors; no extra training |
| S3 | 0 or 8 | No new fits for fewer than two passing levers, or exactly L9 plus one fitted lever; otherwise eight |
| S4 | 5 | Full fold-training refits, seeds 42–46, after the second gap gate |

For S2/S3 keep the ≥1-point gain, paired component-bootstrap lower bound, alpha 0.04 ordinary/L7 and 0.01 L9, recall
protection and zero-recall veto. L5 needs at least 90% positive minibatches in each of the exact eight S2 fits,
with authenticated complete counters; S3 coverage is descriptive. L7 retains its 12/20-versus-30 comparison and
tie rule. L3 skips require the separate measured accounting and explicit close-out disposition; missing fits
alone never authorize a skip, and complete eight-fit results must be screened.

Reuse the same CLEAN exports at the second gap check. Apply the fixed one-point discount only to the primary R1
gap after S3, then the same −3-point threshold. Retain the best single lever if the combination is not promoted.
Recompute training-only normalization, weights, priors and any L1 shrink for the final recipe's training set and
step count. Never retune tau when reusing the L9 combination.

**8. Scale, freeze final models and report once**

After fold 0 continues, run the same algorithm independently in folds 1–4. Do not copy fold-0 winning
hyperparameters into them or modify the algorithm from the fold-0 gap result. Finish all 25 selected DeepMzyme
refits, CLEAN R1 at five seeds per fold, R2 at three, and the approved C-ESMC control at three.

Replay every DeepMzyme refit and the fixed first S1 anchor in each fold. Use the updated frozen logit tolerances,
1e-6 on the same device and 1e-5 across devices, with the recorded near-tie handling; do not resurrect the older
probability-only wording. Bind every model, data identity and export/projection/scoring/bootstrap source in
Freeze 2 before any test prediction.

Retain the required MMseqs2 nearest-hit baseline and the declared released-CLEAN-weight B3 diagnostic. Prepare
and pin any remaining private B3 prediction adapter and weight identities before Freeze 2; B3 is a descriptive
exposure demonstration, never the retrained comparator or a selection input.

G5 separately authorizes the one-shot test pass. Build label-blind predictions for all declared systems and their
hash claim before the scorer reads labels. Preserve the singleton-stratum availability gate, class-qualified fallback, fixed Holm
families and both co-primary populations. A tie, loss or unavailable inference is a valid report. The named
CLEAN30 benchmark remains secondary and does not designate a pristine primary EC final-test route.

The campaign identity tree and its tar remain restricted under [DATASETS](../../DATASETS.md#test-use-ledger).
Development reads only the approved per-fold overlay copies. No full-data deployment fit, new benchmark,
architecture search, optimizer change, extra seed or framework is included here.

**Completion and authorization boundaries**

| Milestone | Concrete deliverable | Next decision |
|---|---|---|
| CPU readiness | Tested WP-13, smoke evidence, permitted transfer manifest, capacity forecast and G2 request | Approve the bounded G2 scope |
| Measured G2 | Timings, replay/lane/persistence results, stopped provider, proposed Freeze 1c | Approve G3 numerical ceilings and workload |
| Fold-0 S1 | 96 verified fits, eight CLEAN fits/exports, first gap record | Frozen G4 rule |
| Fold-0 S3/refit | Applicable lever evidence, second gap record, five verified refits | Continue the unchanged independent-fold procedure |
| Final models | All declared models and replay evidence, Freeze 2 | Separate G5 scoring authorization |
| Benchmark report | One scored claim, qualified conclusions and complete disclosures | Campaign close-out; deployment remains separate |

Maintain one coordinator. Do CPU code and artifact preparation in complete reviewable batches, reuse valid
evidence, and request only the actual outstanding compute or scientific decisions at their gates. If a ceiling
becomes insufficient, retain the accepted science and use the recorded stop-or-increase procedure.
