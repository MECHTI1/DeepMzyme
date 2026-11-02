# Freeze 1: ec_class_vs_clean30_2026-10

This page renders `freeze1.json`; that file and its SHA-256 (logged in `log.md`) are the record.

- Format: `deepmzyme_ec_clean30_freeze1_document_v1` (schema 1)
- Source commit: `d96527f965adcb5c0d2f27080258bff846f070b8`
- Inputs manifest: `freeze/freeze1_inputs.json`, SHA-256 `45cc324aa6702358bfd25fda313f5fab4c204326c72891c035082668da28a924`
- Decisions record: `docs/campaigns/ec_class_vs_clean30_2026-10/freeze1_decisions.json`, SHA-256 `9a2d93a6084bf4470e1a8def9cb492357a9f06d43ad74209b28e57eef8e1fe44`
- Overlay: `ecc30-overlay-v1`, manifest SHA-256 `6f259f952aeb48692f741debed3c880c24b0ad25622f3536829346331cf5562f`
- Transfer list: bound
- Amendments: 0

## Decisions

| ID | Answer |
|---|---|
| A1 | Use option d: recompute only the charge-distance proxy using selected catalytic metals, including no duplicate or unselected metals. Preserve structure bytes and PROPKA features, disclose their limitations, and apply the same feature construction at final testing. Each protein has one pocket around all selected catalytic metals. |
| A10 | Exclude CLEAN split_metadata.json entirely. |
| A11 | Do not create a private Hugging Face copy of the overlay in this campaign phase. |
| A12 | Check file modes only on filesystems that support them. Always check SHA-256; record file_modes_enforced as false on the current secondary-drive mount. |
| A13 | Count payload and sidecar strings naming test or mixed folders without printing them. Disclose the recorded 370 strings; preserve the source bytes. |
| A2 | All audited proteins loaded, so no DeepMzyme-only exclusion or WP-7 repair is required. A future failure stops for a decision. |
| A3 | Require selected-site coordinates within 0.01 angstrom and the same element. Report failures without silently loosening the tolerance. |
| A4 | Quarantine and stop on a content difference between the structure-name EC set and native EC set. Permit and disclose ordering-only differences. |
| A5 | Retain Q96KN2 V113I without re-embedding and disclose its sequence difference. |
| A6 | Report very short sequences and fragment limitations only; no short cohort sequence was found, and fragment status was not checked. |
| A7 | Build the native label table over all 10,202 training-source proteins, including non-metallo hits needed by the baseline. |
| A8 | Approve one download of the published ENZYME nomenclature record from https://ftp.expasy.org/databases/enzyme/enzyme.dat. Record its retrieval date, source and SHA-256 and bind the derived transfer list into Freeze 1. |
| A8-a | Resolve transfer chains to current entries. Drop branches ending in deleted entries; omit a source if all branches end in deletion. Missing targets and cycles stop the build. |
| A8-b | Use ec_transfer_list.csv with comma-separated ec_number and transferred_to columns, and require parity with the scorer reader. |
| A8-c | Match exact full EC numbers only. Partial native annotations never match the transfer list; disclose this limitation. |
| A9 | Do not use RING features; the campaign runner refuses them. |
| B1 | Use identity bins no_hit, <=20, (20,25], (25,30), [30,50], >50. The code marker is settled by this record without editing the identity code. |
| B10 | Assign R2-inner proteins by component, including several-class proteins in components with no eligible protein. Refuse overlap with R1-inner exclusions; never silently subtract it. |
| B11 | Inner-fold class-7 recall is undefined and descriptive only. |
| B12 | Require an uppercased four-character PDB donor identifier beginning with a digit. Anything else stops the fasta step. |
| B2 | Use has_d2 for the D2 subset without a 50-pair minimum. |
| B3 | Keep the frozen MMseqs2 search settings: search-type 1, sensitivity 7.5, E-value 1e-3, alignment-mode 3, seq-id-mode 0, alignment output on, minimum sequence identity and alignment length 0, coverage 0 with cov-mode 0, masking and composition-bias correction on, and max-seqs/max-accept/max-rejected 1000000. |
| B4 | Keep several-class proteins as links in identity components. |
| B5 | Nearest-hit order is highest bit score, then lowest E-value, then target accession. Identical nonstandard residues count as aligned pairs but not identities. |
| B6 | Keep the existing accession-case alias-link rule. |
| B7 | Use the metallo-cohort complement of each official training fold; do not query other proteins in the official complement. |
| B8 | If a component is too large, permit one retry at the one-third limit in a new identity folder; if that still fails, stop for a decision. No retry was needed for the existing folds. |
| B9 | Use inner split seed 42 without retries. |
| BP item 20 | The view builder checks inner-certificate fields; the real Freeze-1 completeness check additionally binds the certificate and membership by SHA-256 to the overlay identity copies. |
| BP-D11 | Skip the optional metal-release CPU parity run; no metal release is part of this continuation. |
| BP-D3 | Malformed EC annotation tokens stop the run. Proteins with several EC classes are counted exclusions from single-label training. |
| BP-D7 | Use the label-blind prediction-export pocket builder with placeholder __EC_0.0.0.0. |
| BP-D9 | Approve public release of the maintainer-owned CLEAN comparator wrapper with the reviewed commits. No CLEAN source, modified private scripts, patches or weights are included. |
| C-1 | An L3 cost skip requires a separate runtime disposition bound to the Freeze-1c ceiling, measured accounting and explicit stop authority. Missing fits alone never imply a cost skip. Exclude partial results from selection; a complete verified eight-fit set is always screened. |
| C-10 | Keep the literal R-CMD rule for root-level files and configuration directories. Updating root STATUS would require repeating the five dry runs; preserve its known stale state and keep current progress in campaign records. |
| C-11 | Require SA1, SA1-c, P2-L5, P2-9-A1, A8-a through A8-c and the separate C-1 through C-14 choices in the decision record, in addition to all original decisions. |
| C-12 | Require the clean_patch_manifest, clean_environment_lock, esmc_300m_weights and mmseqs2_binary pins. Pin ESM-1b weights when downloaded under the later approved phase. |
| C-13 | verify-freeze1 requires the document to be committed at HEAD with the same bytes and its SHA-256 in the committed campaign log. A failed check writes no receipt. |
| C-14 | Bind the transfer list and receipt into Freeze 1. The explicit no-transfer-list branch is available only if the download is declined; A8 is approved here. |
| C-2 | Apply the 90% L5 gate to the eight S2 screening fits only. Re-certify that S2 evidence when L5 enters S3 and record S3 coverage descriptively without adding an S3 coverage gate. |
| C-3 | When exactly L9 and one fitted lever pass, reuse that lever S2 fits with the fixed S2 tau and compare the combination against the best single lever under the ordinary promotion rule. Do not fit a redundant S3 recipe or retune tau. |
| C-4 | When L7 combines with other levers, the combination takes its selected epoch count; recompute L1 target shrink for the resulting step count. |
| C-5 | Discount only the gated EC1-6 R1 gap after S3 by 1.0 BA point. Report descriptive seven-class and R2-inner gaps without that discount. |
| C-6 | Require each CLEAN query set to equal the certified inner-validation proteins exactly before closing the gap-gate comparison. Reject extra as well as missing queries. |
| C-7 | Require the gap gate to receive the overlay identity components and verify the certified inner membership by the same rule as selection. |
| C-8 | Continue at a gap of at least -3.0 BA points, inclusive with tolerance 1e-9. |
| C-9 | Count each eligible outer-fold training protein once for the recall guard, using the union of its certified inner-validation folds. |
| C1 | Use eight refit-probe proteins per fold, ordered by SHA-256. |
| C10 | L9 uses tau values -1, -0.5, 0, 0.5 and 1, with add-one smoothing and per-fit training priors. Ties prefer smaller absolute tau, then the negative value. |
| C11 | Keep git-status checks inside fits. |
| C12 | Use the view-only refit route with val-fraction 0 and runner-certified coverage. |
| C13 | Use fused width 128. |
| C14 | Disclose that the implemented cosine schedule does not reach exactly zero in the last epoch. |
| C2 | Keep separate chain-mean caches per view; a cache retaining the full-cohort parent allowlist is not valid for a fold view. |
| C3 | L1 targets total shrink 0.059 on the GVP trunk only. Disclose that the prior whole-model Only-GVP gain and the trunk-only fusion result were unmatched exploratory evidence. |
| C4 | L3 uses coordinate noise 0.1 angstrom and outer-residue dropout 0.1, with a per-fit generator. Freeze 1 fixes its cost policy; Freeze 1c fixes the numerical ceiling. |
| C5 | L4 adds EC levels 2 and 3, loss weight 0.3, with auxiliary heads shaped like the EC head. |
| C6 | L5 uses loss weight 0.1, temperature 0.1, memory 256 and projection width 128. Positive-bearing training minibatches must reach 90% in each required S2 fit. |
| C7 | L6 uses head dropout 0.3 and ESM-encoder dropout 0.2. |
| C8 | L7 compares 12 and 20 epochs; ties choose 12. |
| C9 | L8 uses block 26 of 30, standardized using training proteins only. Cache parity requires cosine at least 0.9999 and maximum absolute error at most 1e-4. |
| Counts | Do not print, log or publish per-EC-class counts or several-class-protein counts for more than one fold. Preserve the previously disclosed fold-0 values; bind private receipts by hash. |
| D1 | Use alpha 0.04 for ordinary comparisons and L7, alpha 0.01 for L9, and minimum gain 1.0 BA point. The ladder selection and fresh-seed confirmation are complete. Disclose the low probability of promoting a true +2-point gain. |
| D2 | Use an unstratified inner component bootstrap because every fold has at least ten components per EC class 1-6. |
| D3 | Apply the zero-recall veto and recall guard to seed-mean predictions. |
| D4 | Keep the frozen tie rules: L7 prefers 12 epochs; L9 prefers smaller absolute tau then negative tau; a non-promoted S3 combination retains the best single lever; use the fixed Holm ordering. |
| D5 | Bootstrap seed 20261023, ordinary simulation seed 20261024, independent confirmation seed 20261030. Use 10000 bootstrap draws in batches of 1000. |
| D6 | Use the ecc30timing unit prefix for Phase 3 timing and lane-gate fits; never use them in selection. |
| D7 | Build and test selection and the gap gate before Freeze 1 so the applied scientific rule is pinned. |
| E1 | Keep the frozen scoring settings: has_d2, R2 seed mean, R2 and C-ESMC seeds 42-44, seven secondary hypotheses at alpha 0.05, the specified difference-in-differences strata and macro-F1 over present EC classes 1-6. |
| E10 | A crashed scoring-pass record can be removed only manually with a test-use ledger entry. Do not silently retry or replace it. |
| E2 | Set contrast values within 1e-12 of zero to zero. |
| E3 | Keep the scorer count-table treatment of errors, including tie and failure errors. |
| E4 | Count final-pass failures as errors. Refuse checkpoints not trained on the selected sites; retain Freeze-2 binding, pass identity and write-once outputs. |
| E5 | Use the floor(alpha times n) order statistic for the lower bound, define the rarest class by total protein weight, and retain EC5/EC6 removal analyses. Unavailable hypotheses remain in their multiplicity family and cannot pass. |
| E6 | mechti |
| E7 | No test-result contents were opened by the names-only search on 2026-11-01. No accessible matching titles were returned; this does not establish that no earlier evaluation occurred. Earlier CLEAN30 test outputs may have been produced. Preserve the existing test-use disclosures. |
| E8 | Disclose the campaign design limitations, Q96KN2, the selected-metal charge correction and retained PROPKA limitations, masked-copy differences, fair-esm 2.0.0, keyed masks, the A13 count, unsupported file modes, CPU-built L8 means, low promotion power, and population-specific stratum support and class scope. |
| E9 | A baseline hit with several EC classes is a tie and an error; a hit with no usable EC class is a failure and an error. |
| F1 | Use CLEAN f2bf2a4 with its published recipe, R1 seeds 42-46, distance-tie tolerance 1e-6 and the frozen mean-distance R1 ensemble rule. |
| F2 | Keep the recorded Python hash-seed policy, EC-column order, training-only validation queries, hash-bound base FASTA, and prepare/farms/prepare flow. Conflicting EC order is an error. The ESMC-farm wrapper contains no CLEAN code. |
| F3 | Disclose differences between ESM-1b and ESMC masked-copy means; do not force them to match. |
| F4 | Obtain final test-query sequences from the frozen sequence map without truncation. |
| F5 | Removing a claim file manually requires a ledger entry; refuse in-sample evaluation queries. |
| F6 | Pin the CLEAN patch manifest beginning f331019d, the CLEAN environment lock and private harness identities by SHA-256. Preserve private mirrors; distribute no private patch or harness source. |
| F7 | Use final ESMC layer 30 and retain zero-based block indices, including block26 as index 26. |
| F8 | Use serial CLEAN fits on internal ext4 storage as in M3/M4, with the measured approximately 2.9-hour per-fit calendar. Run each Phase 4 distance map alone under 4.5 GB; this continuation performs no such fits. |
| F9 | Each Phase 4 CLEAN prepare, farm, fit and export call receives the computed per-fold forbidden-root list covering campaign identity and stdout, labels, cohort, views, runs, freeze, simulation, timing, plans, smoke material, overlay features and structures, other-fold identity and R2 lists, and backups. Test every call against these restrictions without modifying the frozen comparator. |
| G1 | Use replay logit tolerance 1e-6 on the same device, 1e-5 across devices, with near ties at twice the applicable tolerance. |
| G2 | Replay each fold first S1 anchor fit: F2, learning rate 1e-4, inner fold 0, seed 42. |
| G3 | Lane-gate limits are at most 1.0 BA point, at most 3 recall points for classes with at least ten validation proteins, at least 95% top-1 agreement and at least 1.2-fold speed. |
| G4 | Freeze 1b records the identity audit before G2. Freeze 1c records the L3 cost ceiling, lane-gate result and timings before any Phase 4 fit; read the L3 ceiling there without modifying hashed plans. |
| G5 | Bind the campaign conda explicit-MD5 export and pip freeze export, CLEAN environment lock, pinned MMseqs2 binary, ESMC-300m weights and fair-esm 2.0.0. Record ESM-1b weight SHA-256 when downloaded under later authority. |
| G6 | Use the freeze1_amendment_chain plus a hashed amendment document for every later change. |
| G7 | Keep the implemented Freeze-2 manifest format, pass-ID ledger and selected-site checkpoint enforcement. |
| G8 | Freeze every tracked src and scripts file by its own SHA-256. Add later execution code only through the recorded new-file route with later pins; every unit checks protected files against Freeze 1. |
| G9 | No absolute paths in tracked documents. Keep the inputs manifest private and bind it by SHA-256. |
| Lock | The identity tool checks the Freeze-1 hash format; verify-freeze1 and Freeze 1b supply the mandatory committed-log, pushed-history and receipt checks. |
| M1 | Build the block-26 caches on CPU. The five-protein parity check passed, and the completed per-fold caches and their parity evidence are reused. |
| M2 | Build the Phase 3 runner later using new files and existing hooks, before G2. It is outside this continuation, which ends at Freeze 1b; changing protected command code requires repeating the five dry runs. |
| M3 | Use internal ext4 storage for CLEAN embedding farms and active fits; exclude it from DeepMzyme fits and copy completed outputs to the campaign store and independent mirror. |
| M4 | Run CLEAN fits serially on this workstation. Each Phase 4 distance map runs alone under a 4.5 GB limit. This record does not start Phase 4. |
| M5 | Keep DeepMzyme fold views on the secondary drive. Enforce modes where supported and always verify hashes; archives declare modes explicitly. |
| M6 | Defer the faster NTFS-driver restoration until convenient. Reuse the verified fuseblk mount; never force a mount or run ntfsfix. Any later driver change requires the recorded view-mode and hash checks. |
| M7 | A names-only search on 2026-11-01 returned no accessible matches for the earlier CLEAN output names. No result contents were opened. Search coverage is limited to accessible indexed titles and is not proof that no earlier evaluation occurred. |
| M8 | Phase 3 transfers only the fold-0 view archive, 1,000 training-only timing sequences and their map, the pushed code, and the separately approved private CLEAN setup. No full overlay, other fold view or private Freeze file; no Hugging Face copy. |
| Map run | The second CLEAN distance-map run for the internal-disk re-time was approved and recorded in evc-005. |
| P2-9 | Use validation-only seed-spread stand-ins from the archived metal campaign: 1.5 BA points centrally and 2.2 at the upper setting. Add the stand-in in quadrature to each simulated population noise floor. The CARE pair is descriptive only; no new EC fit is used. |
| P2-9-A1 | Retain the completed alpha ladder and fresh-draw confirmation: choose the largest passing alpha with at most 10% one-look corner adoption and at most 15% multi-look adoption using the declared divided-alpha fallback. Alpha 0.04 ordinary/L7 and 0.01 L9 were confirmed. Do not repeat the ladder. |
| P2-9b | Resolve the final-record noise scope before results: five folds, the eight previously declared pessimistic corners and looks 1, 2 and 4 give 120 null points and 48000 replicates. Use 400 replicates per point, simulation seed 20261024, alpha 0.04 for looks 1 and 2 and 0.01 for looks 4, minimum gain 1.0, 10000 bootstrap draws and seed 20261023. Corners cross fit_noise_sd 0.4/1.2, class_recall_offset -0.1/+0.1 and component_sd 0.5/1.5 with protein_sd 1.5. Recompute each fold/corner noise floor F using 1000 spread populations and set seed_spread_points to ceil(sqrt(F squared + 2.2 squared) times 10000) divided by 10000. Reuse existing v1c/v1d power evidence, run no new power or rule search, and report per-point intervals without pooling shared draws. Interpret the fixed policy as 10% single-look and 15% multi-look adoption; material contradictions require review without automatic retuning. |
| P2-L5 | Count actual training minibatches with positives under the batch-plus-memory rule, excluding same-protein positives. Include no-positive minibatches in the denominator. Bind complete per-epoch counters in checkpoint and run records, recompute totals, reject receipt mismatches, require the exact four-inner-fold by two-seed fit grid and apply the minimum fraction across the eight fits. Evaluation, replay and export do not increment counters; old checkpoints remain loadable but uncertified without valid counters. |
| Push | Approve the reviewed CP-C, CP-D and CP-E commits and their push to origin main. Freeze 1 must be publicly committed and verified as pushed before the identity audit. |
| SA1 | If a stratum contributing to the primary EC1-6 balanced accuracy contains only one independent cluster, retain point estimates but make superiority inference unavailable for that population. Do not drop classes or clusters to restore a claim. Missing classes require explicit k-class wording; the six-class endpoint remains unevaluated. Keep unavailable hypotheses in the fixed Holm families. |
| SA1-c | The EC7-only singleton exemption applies to primary EC1-6 balanced accuracy. It blocks accuracy inference and blocks macro-F1 inference when a compared system predicts one of those proteins as an averaged present EC1-6 class. Preserve point estimates and fixed multiplicity families. |
| U1 | Use the official CLEAN split30 folds unchanged. No new outer split. |
| U10 | Keep the fixed L1-L9 candidate levers and their inner-validation promotion rules; no new inputs or data. |
| U11 | Use CLEAN upstream f2bf2a4 with ESM-1b and a separate fair-esm 2.0.0 environment for non-profit research. Keep CLEAN source, modified scripts, patches and weights private. Publish only the maintainer-owned wrapper and permitted records. Reports carry the required acknowledgement and notification. |
| U12 | Use the published CLEAN recipe without local tuning or extra epoch variants. Disclose the unequal local tuning opportunity and upstream use of the same benchmark folds. |
| U13 | Keep the ESMC-input CLEAN control; its comparison changes the input pipeline and does not isolate only the language model. |
| U14 | Use five-seed DeepMzyme and CLEAN R1 ensembles, with seeds 42-46. |
| U17 | Keep campaign data on the secondary drive with an independent NVMe mirror, verified by SHA-256. |
| U2 | Include the whole-protein ESMC branch. |
| U3 | Exclude non-metalloenzyme training data, homology-retrieval additions and metal-element inputs from the core comparison. |
| U4 | Use at least 30% identity over at least 50 aligned residue pairs, E-value at most 1e-3, any coverage, plus shared selected AlphaFill donors. This defines the no-hit subset, identity bins and inner splits, without changing official outer folds. |
| U5 | No new training data or general reprocessing. Disclose the fold 1-4 screening gap and the separately approved selected-metal charge-proxy correction. |
| U6 | Leave the CARE clusterRes30 metallo test unused and unevaluated. |
| U7 | Keep all other campaigns at their recorded status. |
| U8 | Primary metric: balanced accuracy over EC classes 1-6 from seven-way predictions. Seven-class balanced accuracy, accuracy and macro-F1 are secondary; missing-class qualification and support gates apply. |
| U9 | Use the terminal checkpoint under the same cosine schedule in selection and refit; choose 12, 20 or 30 epochs through the fixed inner-selection rule. No weight averaging or warmup. Disclose that the implemented final epoch has a positive learning rate. |

## Plans

| Fold | Plan SHA-256 | Planned fits |
|---|---|---|
| 0 | `45c43b6529d538b3caaef3ce3b1f1382e1ab1e79195b27c77cbba9c34cba1718` | 181 |
| 1 | `5d974909feb83224aa8af35549760254faa40e247babe28e060750c650b19b7f` | 181 |
| 2 | `18cc61ab328436fe440174f9dc4c83a4f063e004650fc4e70d25cfd4241ed24f` | 181 |
| 3 | `3f935ffe1e7860ad4e76115c957aeb647d004a9b77f8e5176a14aed73c6a263b` | 181 |
| 4 | `388bc4d88696cf2156cd7d800f4f7548f48fd92545cbff8cc96a29285839e993` | 181 |

## Views

| Fold | View manifest SHA-256 | Inner membership SHA-256 | Certificate SHA-256 | Dry-run receipt SHA-256 | File modes enforced |
|---|---|---|---|---|---|
| 0 | `2d36e3bc6237dbd32c54e405ae1430d377423cfed11b48baca2a483aea23847e` | `862bbdcc4dcc49b79e5a4285cba21f981dd8dd589c4f174bf4fd009bed33b440` | `efdc17f90b78daf7b0860b408e4faef0026d8fa1c0f229940b67d2d34c369b78` | `f71214bd98cfd5b042549d4a0bd0e2922aa5a573c65288e8548219488dff495a` | false |
| 1 | `c704be501829555e0819c0c3e5c03ff141430c49cb9612a1258124b296f92598` | `d5f631327c59fa5e6625392230fb8ef63ad25047743813397ac8e5a59dfab0ea` | `01be3b211902766a32bff49237979f10d7618a01c42fc33796ac134f28cd9eae` | `e19350c09c7cc01f251ab46428c7ecaa08abeb4b1495c4dbde7e0c29c1ce0b5d` | false |
| 2 | `50fd05170f0c61790e4c70298b38d2abe4e8524ed963c0da9f0dbcff8f22baba` | `47a972f1872b4eef5520cfa5f79873b8e662f67d4ea296b1a9389bede593d72b` | `e40fc400f33c3e9eb6075a4895861ae1fd4ef645bac5dd0ca4d6336d49f4eb4a` | `422128fd0e3c3872b9c011fa02528ee740f1cf1e044df456851de255727ff5a0` | false |
| 3 | `bec69c7b7298f73f58c952ec59a527e4825f2d6b3149e24b399aaf60b6a22978` | `cb52619ba7a6878357367ddcc7954b545d72cc17ad60f305462c38d7e752ebbd` | `22a719381e0da6909e0c35226ed2dda34ff033f5bb1daa80c99c513289df2d5f` | `657ee6a60ef268df8d3ce9416085b1bc86b667fb964c3cf8931e8e5cbb658283` | false |
| 4 | `b1b0fe5903f5787384b58af0982cdf1913e5c603cd5ae7c5fef1e58a8b39c95d` | `a4c78e9b0f33746d437f296cd846e8d46f6188c2a2563c5e2a3a6a272d3d699a` | `2988b1f07dd11a113bd4f11d7061558d17cb95d528059c6f4ed9b16281e68ed8` | `e27ce825fd1c313d5f624282d01b5ddb1c7b02bed6c5c2dd678fce4c21c3d805` | false |

## Rule R-CMD

- Status: passed at `d96527f965adcb5c0d2f27080258bff846f070b8`
- Fold 0 dry-run HEAD: `451c190c4049f721b21e433c82679617174165aa`
- Fold 1 dry-run HEAD: `451c190c4049f721b21e433c82679617174165aa`
- Fold 2 dry-run HEAD: `451c190c4049f721b21e433c82679617174165aa`
- Fold 3 dry-run HEAD: `451c190c4049f721b21e433c82679617174165aa`
- Fold 4 dry-run HEAD: `451c190c4049f721b21e433c82679617174165aa`

## Bound files

| Role | Path | SHA-256 |
|---|---|---|
| cohort_audit_receipt | `cohort/cohort_audit_receipt.json` | `3c10d49afeac52531fa8bfd47f8cd575c20edb0dab4ab3bc613b94f14915e2c5` |
| environment_conda_explicit | `freeze/env/env_conda_explicit.txt` | `c809b698ea3434f66946db2ffaaedb8e6b42b0b7988a3c6050309cdfdac6ecee` |
| environment_pip_freeze | `freeze/env/env_pip_freeze.txt` | `8c64efba609359e896046aed4d1b6357b7b479515a56b815dad59ce4615b1c4f` |
| label_table | `labels/label_table/native_label_table.tsv` | `afc316a961bde53f618b440595bf6f4dbcba63f2755a10b926ad0c1efe01abb3` |
| label_table_receipt | `labels/label_table/label_table_receipt.json` | `a83f1d4197e5b4a857ea305b135c32e9bbff5ec2794008372c6339a71162c2e2` |
| r2_lists_receipt | `r2_lists/r2_lists_receipt.json` | `a86ef8e79ab8e9cc61b8c406af02e516f7da9b1a57c87f1d1b18fc6fe9f926b7` |
| sequence_map | `labels/sequence_map/sequence_map.tsv` | `e9d6ddbb9c772f893f7e9194c94ecac139c3b5893a7d40384368b41b3e487fe0` |
| sequence_map_receipt | `labels/sequence_map/sequence_map_receipt.json` | `552c297282b63a192b94927fb368a716401fd8afba500facd73016683646e63e` |
| simulation_summary | `simulation/final_v1/summary.tsv` | `cf3b80437fd3d3e0063390bcd7d9b3d328ddacc3f2e9b7009ac6804169acbc84` |
| simulation_sweep | `simulation/final_v1/sweep.json` | `51f1d48af4c0295f0e962ffe6b012b4d1dad9a91b21f6a39715a488f2d43c7d4` |
| simulation_sweep_receipt | `simulation/final_v1/sweep_receipt.json` | `d3685c31084cb603c3bdfb495c2e3f36ffd6294a9acbafd07b819a023e6c11aa` |
| transfer_list | `labels/ec_transfer_list/ec_transfer_list.csv` | `84ec98b484fd8b048ec73a5344b20efc29924ee6726d9ecef448ee33a31de2d7` |
| transfer_list_receipt | `labels/ec_transfer_list/transfer_list_receipt.json` | `adcf7e707754b54091c44e3df3a349eba2ff7d0f2a7ed159f274d322609e911f` |
| fold 0 assignment receipt | `overlay/ecc30-overlay-v1/identity/fold0/inner_assignment_receipt.json` | `912275973b8aaa4b8ab46b43a97090a543851721c6759c9e3c4c36bf81957788` |
| fold 0 certificate | `overlay/ecc30-overlay-v1/identity/fold0/inner_certification_receipt.json` | `efdc17f90b78daf7b0860b408e4faef0026d8fa1c0f229940b67d2d34c369b78` |
| fold 1 assignment receipt | `overlay/ecc30-overlay-v1/identity/fold1/inner_assignment_receipt.json` | `ad4951c11b7a8987b7d5df0e450e0f3f9076fca9b582295d493a126340674a1b` |
| fold 1 certificate | `overlay/ecc30-overlay-v1/identity/fold1/inner_certification_receipt.json` | `01be3b211902766a32bff49237979f10d7618a01c42fc33796ac134f28cd9eae` |
| fold 2 assignment receipt | `overlay/ecc30-overlay-v1/identity/fold2/inner_assignment_receipt.json` | `b0ab04f1638031c9316908af7bbb6b0c33710a7733edb9831104b5b2a91c0143` |
| fold 2 certificate | `overlay/ecc30-overlay-v1/identity/fold2/inner_certification_receipt.json` | `e40fc400f33c3e9eb6075a4895861ae1fd4ef645bac5dd0ca4d6336d49f4eb4a` |
| fold 3 assignment receipt | `overlay/ecc30-overlay-v1/identity/fold3/inner_assignment_receipt.json` | `2e201a3eb6136bb6d267a4638be71c91fb20b41698fadfdc5a16fd4783a4f74f` |
| fold 3 certificate | `overlay/ecc30-overlay-v1/identity/fold3/inner_certification_receipt.json` | `22a719381e0da6909e0c35226ed2dda34ff033f5bb1daa80c99c513289df2d5f` |
| fold 4 assignment receipt | `overlay/ecc30-overlay-v1/identity/fold4/inner_assignment_receipt.json` | `fbf76cf5299ef429bf7fde85558bae0e73f7271a3180810f4b8ad836f3d7e567` |
| fold 4 certificate | `overlay/ecc30-overlay-v1/identity/fold4/inner_certification_receipt.json` | `2988b1f07dd11a113bd4f11d7061558d17cb95d528059c6f4ed9b16281e68ed8` |

## Pins

| Name | SHA-256 |
|---|---|
| clean_environment_lock | `69748179b3f9dc00012a6be79aca562cb8e434317e5a3cdfe5ce16e605d1b41b` |
| clean_patch_manifest | `f331019de076ec42f05df057055052c76bab19774b03ed9fd23c6da460aaf53d` |
| clean_pip_freeze | `e137e500751c64dee15cd93ab1a038ec33aa8e544b37e42ea7e79e5724992bf2` |
| clean_python_version | `272989fe6f79973c628f08f9f84b35527eadb2064f02a410a4cb518a53a22127` |
| clean_torch_config | `9d177624a8887951d89f99425eaf73dd64039d3524451f2541199fc9dde8befe` |
| esmc_300m_weights | `323dff9fbf3fef297a74f4f18b6528e6f2e599b0bcf72b6927516804015becea` |
| mmseqs2_binary | `b7ef6e0e33df5dd4fa9cf988cbd8b4988c11a3a1255d2c377f13e1bb40c157fb` |

## Live constants

### baseline

```json
{
  "CLASS_TABLE_COLUMNS": [
    "fold",
    "accession",
    "predicted_ec_class",
    "status"
  ],
  "SEVERAL_CLASS_RULE": "a nearest training protein with several EC classes is a tie (an error), never a vote",
  "STATUSES": [
    "predicted",
    "no_hit",
    "tie",
    "failure"
  ],
  "STEP": "mmseqs2_nearest_hit_baseline"
}
```

### campaign

```json
{
  "campaign_constants": {
    "anchor_family": "F2",
    "campaign_id": "ec_class_vs_clean30_2026-10",
    "candidate_families": [
      "F2",
      "F3",
      "F4",
      "F5"
    ],
    "control_families": [
      "F1",
      "C-G"
    ],
    "default_epochs": 30,
    "epoch_looks": [
      12,
      20
    ],
    "families": {
      "C-G": {
        "family": "C-G",
        "fusion_mode": null,
        "global_branch": true,
        "has_gvp_trunk": false,
        "model_architecture": "only_global",
        "role": "control"
      },
      "F1": {
        "family": "F1",
        "fusion_mode": null,
        "global_branch": false,
        "has_gvp_trunk": true,
        "model_architecture": "only_gvp",
        "role": "control"
      },
      "F2": {
        "family": "F2",
        "fusion_mode": null,
        "global_branch": false,
        "has_gvp_trunk": false,
        "model_architecture": "only_esm",
        "role": "anchor"
      },
      "F3": {
        "family": "F3",
        "fusion_mode": "late_fusion",
        "global_branch": false,
        "has_gvp_trunk": true,
        "model_architecture": "gvp",
        "role": "candidate"
      },
      "F4": {
        "family": "F4",
        "fusion_mode": "late_fusion",
        "global_branch": true,
        "has_gvp_trunk": true,
        "model_architecture": "gvp",
        "role": "candidate"
      },
      "F5": {
        "family": "F5",
        "fusion_mode": null,
        "global_branch": true,
        "has_gvp_trunk": false,
        "model_architecture": "only_esm",
        "role": "candidate"
      }
    },
    "global_branch_families": [
      "C-G",
      "F4",
      "F5"
    ],
    "gvp_families": [
      "F1",
      "F3",
      "F4"
    ],
    "lever_configs": {
      "L1": {
        "chain_mean_layer": null,
        "epochs": null,
        "f1_options": null,
        "lever": "L1",
        "options": [
          [
            "--gvp-target-shrink",
            0.059
          ],
          [
            "--gvp-target-shrink-scope",
            "trunk"
          ]
        ],
        "provisional": [
          "--gvp-target-shrink-scope"
        ]
      },
      "L2": {
        "chain_mean_layer": null,
        "epochs": null,
        "f1_options": null,
        "lever": "L2",
        "options": [
          [
            "--rbf-use-raw-distances",
            true
          ],
          [
            "--edge-radius",
            8.0
          ]
        ],
        "provisional": []
      },
      "L3": {
        "chain_mean_layer": null,
        "epochs": null,
        "f1_options": null,
        "lever": "L3",
        "options": [
          [
            "--position-noise-std",
            0.1
          ],
          [
            "--outer-residue-dropout",
            0.1
          ],
          [
            "--augmentation-rng",
            "per_fit"
          ]
        ],
        "provisional": [
          "--position-noise-std",
          "--outer-residue-dropout"
        ]
      },
      "L4": {
        "chain_mean_layer": null,
        "epochs": null,
        "f1_options": null,
        "lever": "L4",
        "options": [
          [
            "--ec-aux-levels",
            "2,3"
          ],
          [
            "--ec-aux-weight",
            0.3
          ]
        ],
        "provisional": []
      },
      "L5": {
        "chain_mean_layer": null,
        "epochs": null,
        "f1_options": null,
        "lever": "L5",
        "options": [
          [
            "--ec-supcon-weight",
            0.1
          ],
          [
            "--ec-supcon-temperature",
            0.1
          ],
          [
            "--ec-supcon-memory",
            256
          ],
          [
            "--ec-supcon-projection-dim",
            128
          ]
        ],
        "provisional": [
          "--ec-supcon-weight",
          "--ec-supcon-temperature",
          "--ec-supcon-projection-dim"
        ]
      },
      "L6": {
        "chain_mean_layer": null,
        "epochs": null,
        "f1_options": [
          [
            "--head-mlp-dropout",
            0.3
          ]
        ],
        "lever": "L6",
        "options": [
          [
            "--head-mlp-dropout",
            0.3
          ],
          [
            "--esm-graph-encoder-dropout",
            0.2
          ]
        ],
        "provisional": []
      },
      "L7e12": {
        "chain_mean_layer": null,
        "epochs": 12,
        "f1_options": null,
        "lever": "L7",
        "options": [],
        "provisional": []
      },
      "L7e20": {
        "chain_mean_layer": null,
        "epochs": 20,
        "f1_options": null,
        "lever": "L7",
        "options": [],
        "provisional": []
      },
      "L8": {
        "chain_mean_layer": "block26",
        "epochs": null,
        "f1_options": null,
        "lever": "L8",
        "options": [
          [
            "--global-esm-layer",
            "block26"
          ],
          [
            "--global-esm-standardize",
            "train_proteins"
          ]
        ],
        "provisional": [
          "--global-esm-layer"
        ]
      }
    },
    "levers_without_fits": [
      "L9"
    ],
    "max_fits_per_fold": 181,
    "n_inner_folds": 4,
    "refit_seeds": [
      42,
      43,
      44,
      45,
      46
    ],
    "s1_learning_rate_grid": {
      "C-G": [
        {
          "learning_rate": 0.0001
        },
        {
          "learning_rate": 0.0003
        }
      ],
      "F1": [
        {
          "learning_rate": 0.0001
        },
        {
          "learning_rate": 0.0003
        }
      ],
      "F2": [
        {
          "learning_rate": 0.0001
        },
        {
          "learning_rate": 0.0003
        }
      ],
      "F3": [
        {
          "gvp_learning_rate": 0.0001,
          "learning_rate": 0.0001
        },
        {
          "gvp_learning_rate": 0.0003,
          "learning_rate": 0.0001
        }
      ],
      "F4": [
        {
          "gvp_learning_rate": 0.0001,
          "learning_rate": 0.0001
        },
        {
          "gvp_learning_rate": 0.0003,
          "learning_rate": 0.0001
        }
      ],
      "F5": [
        {
          "learning_rate": 0.0001
        },
        {
          "learning_rate": 0.0003
        }
      ]
    },
    "s2_execution_order": [
      "L1",
      "L2",
      "L4",
      "L5",
      "L6",
      "L7e12",
      "L7e20",
      "L8",
      "L3"
    ],
    "s2_sub_ceilings": {
      "L3": {
        "config_ids": [
          "L3"
        ],
        "max_fits_per_fold": 8,
        "status": "provisional until Freeze 1 (the VM-hour value comes from the timing smoke)",
        "vm_hours_per_fold": null,
        "when_exceeded": "record L3 as 'not tested (cost)'"
      }
    },
    "screening_seeds": [
      42,
      43
    ],
    "step_fit_caps": {
      "S1": 96,
      "S2": 72,
      "S3": 8,
      "S4": 5
    }
  },
  "l1_target_shrink": 0.059,
  "l8_layer": "block26",
  "provisional_lever_settings": {
    "L1": [
      "--gvp-target-shrink-scope"
    ],
    "L3": [
      "--position-noise-std",
      "--outer-residue-dropout"
    ],
    "L5": [
      "--ec-supcon-weight",
      "--ec-supcon-temperature",
      "--ec-supcon-projection-dim"
    ],
    "L8": [
      "--global-esm-layer"
    ]
  },
  "s2_sub_ceilings": {
    "L3": {
      "config_ids": [
        "L3"
      ],
      "max_fits_per_fold": 8,
      "status": "provisional until Freeze 1 (the VM-hour value comes from the timing smoke)",
      "vm_hours_per_fold": null,
      "when_exceeded": "record L3 as 'not tested (cost)'"
    }
  }
}
```

### comparator_published_recipe

```json
{
  "adaptive_rate": 100,
  "epochs": 2500,
  "hidden_dim": 512,
  "learning_rate": 0.0005,
  "out_dim": 128
}
```

### identity

```json
{
  "AMBIGUOUS_RESIDUE_POLICY": "identical_nonstandard_residues_count_as_pairs_not_identities",
  "COHORT_LINK_TYPES": [
    "any_hit",
    "donor"
  ],
  "COMPLEMENT_SCOPE": "metallo cohort complement: cohort proteins (any metallo training file) that are absent from fold k's official training file; other proteins of the official complement are not queried",
  "D1_BOTH_COVERAGE": "4/5",
  "D2_SHORTER_COVERAGE": "1/2",
  "DEFAULT_MAX_COMPONENT_SHARE": "1/4",
  "DONOR_ID_RE": "^[0-9][A-Z0-9]{3}$",
  "IDENTITY_BIN_LABELS": [
    "no_hit",
    "<=20",
    "(20,25]",
    "(25,30)",
    "[30,50]",
    ">50"
  ],
  "IDENTITY_BIN_STATUS": "provisional_until_freeze1",
  "IDENTITY_RULE": "identical_non_gap_columns / alignment_columns_including_internal_gaps",
  "IDENTITY_THRESHOLD": "3/10",
  "INNER_BALANCE_RULE": "one pass, no retries: components with eligible proteins in order of (training count of their rarest EC class, more eligible proteins first, SHA-256 of '<seed>:<component_id>'); each goes to the inner fold with the largest class-weighted deficit, the sum over its classes c of n_component(c) / N(c) * (N(c) / n_inner_folds - n_fold(c)), then to the fold with fewer proteins, then to the lower fold index",
  "INNER_ELIGIBILITY_RULE": "fold-k metallo training proteins with one structure ID and one EC class (training.ec_classes.classify_ec_class on the structure stem, as the trainer); several EC classes: excluded and counted, still component nodes; annotations that cannot give an EC class: refused",
  "INNER_LINK_TYPES": [
    "d3",
    "exact_sequence",
    "alias",
    "donor"
  ],
  "INNER_SPLIT_SEED": 42,
  "MAX_EVALUE": 0.001,
  "MAX_SHARE_BASIS": "eligible proteins of the component over all eligible proteins (several-class proteins not counted); components_receipt_inner.json's largest_component_fraction and largest_component_exceeds_quarter count all nodes",
  "MAX_SHARE_RULE": "a component holding more than the maximum share of all eligible proteins is a refusal; the giant-component fallback is a Freeze 1 decision",
  "MAX_TARGET_SEQUENCES": 1000000,
  "MIN_RESIDUE_PAIRS": 50,
  "MMSEQS_BINARY_SHA256": "b7ef6e0e33df5dd4fa9cf988cbd8b4988c11a3a1255d2c377f13e1bb40c157fb",
  "MMSEQS_RELEASE": "18-8cc5c",
  "MMSEQS_VERSION": "8cc5ce367b5638c4306c2d7cfc652dd099a4643f",
  "NEAREST_HIT_ORDER": "highest bits, then lowest E-value, then target accession (provisional until Freeze 1)",
  "N_INNER_FOLDS": 4,
  "R1_EXCLUSION_RULE": "inner-validation proteins plus every protein of fold k's full official training file with a D3 edge to one of them: hits of a validation protein (query) to the full file, and hits of a metallo training protein (query) to a validation protein, all from the one pinned metallo training search",
  "RERUN_RULE": "a rerun with changed settings is a Freeze 1 decision of the maintainer and needs a new working directory",
  "SEARCH_RUN_FLAGS": [
    "--threads",
    "--split-memory-limit"
  ],
  "SEARCH_SETTINGS": [
    [
      "--search-type",
      "1"
    ],
    [
      "-s",
      "7.5"
    ],
    [
      "-e",
      "1e-3"
    ],
    [
      "--alignment-mode",
      "3"
    ],
    [
      "--seq-id-mode",
      "0"
    ],
    [
      "-a",
      "1"
    ],
    [
      "--min-seq-id",
      "0"
    ],
    [
      "--min-aln-len",
      "0"
    ],
    [
      "-c",
      "0"
    ],
    [
      "--cov-mode",
      "0"
    ],
    [
      "--mask",
      "1"
    ],
    [
      "--comp-bias-corr",
      "1"
    ],
    [
      "--max-seqs",
      "1000000"
    ],
    [
      "--max-accept",
      "1000000"
    ],
    [
      "--max-rejected",
      "1000000"
    ]
  ],
  "rules": {
    "ambiguous_residue_policy": "identical_nonstandard_residues_count_as_pairs_not_identities",
    "comparisons": "exact rational arithmetic on integer counts",
    "d1_both_coverage_inclusive": "4/5",
    "d2_shorter_coverage_inclusive": "1/2",
    "identity": "identical_non_gap_columns / alignment_columns_including_internal_gaps",
    "identity_reference": "audit_sequence_remoteness.alignment_metrics on rows without identical non-standard residues",
    "identity_threshold_inclusive": "3/10",
    "max_evalue_inclusive": 0.001,
    "min_residue_pairs": 50,
    "self_hits": "dropped"
  }
}
```

### logit_adjustment

```json
{
  "PRIOR_SMOOTHING": 1.0,
  "TAUS": [
    -1.0,
    -0.5,
    0.0,
    0.5,
    1.0
  ]
}
```

### promotion

```json
{
  "L9_PROMOTION_ALPHA": 0.01,
  "LEVER_PROMOTION_ALPHA": {
    "L9": 0.01
  },
  "PROMOTION_ALPHA": 0.04,
  "promotion_config_by_lever": {
    "L1": {
      "alpha": 0.04,
      "bootstrap_seed": 20261023,
      "guard_min_training_proteins": 30,
      "guard_net_proteins": 3.0,
      "guard_recall_drop_points": 5.0,
      "max_skip_fraction": 0.01,
      "min_gain_points": 1.0,
      "n_draws": 10000,
      "stratified": false,
      "zero_recall_unit": "seed_mean"
    },
    "L2": {
      "alpha": 0.04,
      "bootstrap_seed": 20261023,
      "guard_min_training_proteins": 30,
      "guard_net_proteins": 3.0,
      "guard_recall_drop_points": 5.0,
      "max_skip_fraction": 0.01,
      "min_gain_points": 1.0,
      "n_draws": 10000,
      "stratified": false,
      "zero_recall_unit": "seed_mean"
    },
    "L3": {
      "alpha": 0.04,
      "bootstrap_seed": 20261023,
      "guard_min_training_proteins": 30,
      "guard_net_proteins": 3.0,
      "guard_recall_drop_points": 5.0,
      "max_skip_fraction": 0.01,
      "min_gain_points": 1.0,
      "n_draws": 10000,
      "stratified": false,
      "zero_recall_unit": "seed_mean"
    },
    "L4": {
      "alpha": 0.04,
      "bootstrap_seed": 20261023,
      "guard_min_training_proteins": 30,
      "guard_net_proteins": 3.0,
      "guard_recall_drop_points": 5.0,
      "max_skip_fraction": 0.01,
      "min_gain_points": 1.0,
      "n_draws": 10000,
      "stratified": false,
      "zero_recall_unit": "seed_mean"
    },
    "L5": {
      "alpha": 0.04,
      "bootstrap_seed": 20261023,
      "guard_min_training_proteins": 30,
      "guard_net_proteins": 3.0,
      "guard_recall_drop_points": 5.0,
      "max_skip_fraction": 0.01,
      "min_gain_points": 1.0,
      "n_draws": 10000,
      "stratified": false,
      "zero_recall_unit": "seed_mean"
    },
    "L6": {
      "alpha": 0.04,
      "bootstrap_seed": 20261023,
      "guard_min_training_proteins": 30,
      "guard_net_proteins": 3.0,
      "guard_recall_drop_points": 5.0,
      "max_skip_fraction": 0.01,
      "min_gain_points": 1.0,
      "n_draws": 10000,
      "stratified": false,
      "zero_recall_unit": "seed_mean"
    },
    "L7": {
      "alpha": 0.04,
      "bootstrap_seed": 20261023,
      "guard_min_training_proteins": 30,
      "guard_net_proteins": 3.0,
      "guard_recall_drop_points": 5.0,
      "max_skip_fraction": 0.01,
      "min_gain_points": 1.0,
      "n_draws": 10000,
      "stratified": false,
      "zero_recall_unit": "seed_mean"
    },
    "L8": {
      "alpha": 0.04,
      "bootstrap_seed": 20261023,
      "guard_min_training_proteins": 30,
      "guard_net_proteins": 3.0,
      "guard_recall_drop_points": 5.0,
      "max_skip_fraction": 0.01,
      "min_gain_points": 1.0,
      "n_draws": 10000,
      "stratified": false,
      "zero_recall_unit": "seed_mean"
    },
    "L9": {
      "alpha": 0.01,
      "bootstrap_seed": 20261023,
      "guard_min_training_proteins": 30,
      "guard_net_proteins": 3.0,
      "guard_recall_drop_points": 5.0,
      "max_skip_fraction": 0.01,
      "min_gain_points": 1.0,
      "n_draws": 10000,
      "stratified": false,
      "zero_recall_unit": "seed_mean"
    }
  },
  "promotion_config_default": {
    "alpha": 0.04,
    "bootstrap_seed": 20261023,
    "guard_min_training_proteins": 30,
    "guard_net_proteins": 3.0,
    "guard_recall_drop_points": 5.0,
    "max_skip_fraction": 0.01,
    "min_gain_points": 1.0,
    "n_draws": 10000,
    "stratified": false,
    "zero_recall_unit": "seed_mean"
  },
  "promotion_config_steps_without_lever": {
    "alpha": 0.04,
    "bootstrap_seed": 20261023,
    "guard_min_training_proteins": 30,
    "guard_net_proteins": 3.0,
    "guard_recall_drop_points": 5.0,
    "max_skip_fraction": 0.01,
    "min_gain_points": 1.0,
    "n_draws": 10000,
    "stratified": false,
    "zero_recall_unit": "seed_mean"
  }
}
```

### scoring

```json
{
  "ABSTENTION_RULE": "an unpredicted protein (explicit failure, tie, no hit) is an error: it counts in its true class's support and in no correct count; count tables place it in the class-7 column for true classes 1-6 and, for true class 7, in the class-1 column (accuracy, balanced accuracy) or the class-7 column (macro-F1 over classes 1-6), so no error adds a false prediction to an averaged class; per-class precision and F1 count real predictions only",
  "CO_PRIMARY_POPULATIONS": {
    "P1": "B1-all",
    "P2": "B1-NH30"
  },
  "FROZEN_SETTINGS": {
    "bootstrap_seed": 20261023,
    "c_esmc_seeds": [
      42,
      43,
      44
    ],
    "d2_flag": "has_d2",
    "deepmzyme_seeds": [
      42,
      43,
      44,
      45,
      46
    ],
    "descriptive_alpha": 0.05,
    "max_skip_fraction": 0.01,
    "n_draws": 10000,
    "r1_seeds": [
      42,
      43,
      44,
      45,
      46
    ],
    "r2_form": "seed_mean",
    "r2_seeds": [
      42,
      43,
      44
    ],
    "secondary_family_alpha": 0.05
  },
  "MANIFEST_SETTINGS": [
    "acknowledgement_name"
  ]
}
```

### selection

```json
{
  "ANCHOR_FAMILY": "F2",
  "CONTROL_FAMILIES": [
    "F1",
    "C-G"
  ],
  "DEFAULT_EPOCHS": 30,
  "DEFAULT_SIMULATION_SEED": 20261024,
  "EPOCH_LEVER": "L7",
  "EPOCH_LOOKS": [
    12,
    20
  ],
  "FAMILIES": [
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
    "C-G"
  ],
  "FLOAT_TOLERANCE": 1e-09,
  "GAIN_CONTRAST": "challenger_minus_incumbent",
  "GLOBAL_BRANCH_FAMILIES": [
    "C-G",
    "F4",
    "F5"
  ],
  "GUARD_MIN_TRAINING_PROTEINS": 30,
  "GUARD_NET_PROTEINS": 3.0,
  "GUARD_RECALL_DROP_POINTS": 5.0,
  "GVP_FAMILIES": [
    "F1",
    "F3",
    "F4"
  ],
  "L9_PROMOTION_ALPHA": 0.01,
  "LEVERS": [
    "L1",
    "L2",
    "L3",
    "L4",
    "L5",
    "L6",
    "L7",
    "L8",
    "L9"
  ],
  "LEVER_PROMOTION_ALPHA": {
    "L9": 0.01
  },
  "LOGIT_ADJUSTMENT_LEVER": "L9",
  "LR_TIE_BAND_POINTS": 0.5,
  "MIN_GAIN_POINTS": 1.0,
  "NULL_ADOPTION_TARGET": 0.1,
  "N_INNER_FOLDS": 4,
  "POINTS": 100.0,
  "PROMOTION_ALPHA": 0.04,
  "REFIT_SEEDS": [
    42,
    43,
    44,
    45,
    46
  ],
  "S1_LEARNING_RATE_GRID": {
    "C-G": [
      [
        [
          "learning_rate",
          0.0001
        ]
      ],
      [
        [
          "learning_rate",
          0.0003
        ]
      ]
    ],
    "F1": [
      [
        [
          "learning_rate",
          0.0001
        ]
      ],
      [
        [
          "learning_rate",
          0.0003
        ]
      ]
    ],
    "F2": [
      [
        [
          "learning_rate",
          0.0001
        ]
      ],
      [
        [
          "learning_rate",
          0.0003
        ]
      ]
    ],
    "F3": [
      [
        [
          "gvp_learning_rate",
          0.0001
        ],
        [
          "learning_rate",
          0.0001
        ]
      ],
      [
        [
          "gvp_learning_rate",
          0.0003
        ],
        [
          "learning_rate",
          0.0001
        ]
      ]
    ],
    "F4": [
      [
        [
          "gvp_learning_rate",
          0.0001
        ],
        [
          "learning_rate",
          0.0001
        ]
      ],
      [
        [
          "gvp_learning_rate",
          0.0003
        ],
        [
          "learning_rate",
          0.0001
        ]
      ]
    ],
    "F5": [
      [
        [
          "learning_rate",
          0.0001
        ]
      ],
      [
        [
          "learning_rate",
          0.0003
        ]
      ]
    ]
  },
  "SCREENING_SEEDS": [
    42,
    43
  ],
  "SUPCON_LEVER": "L5",
  "SUPCON_MIN_POSITIVE_STEP_FRACTION": 0.9,
  "ZERO_RECALL_UNITS": [
    "seed_mean",
    "per_seed"
  ]
}
```

### stats

```json
{
  "BOOTSTRAP_METRICS": [
    "ec_classes_1_6_balanced_accuracy",
    "ec_classes_1_6_macro_f1",
    "accuracy"
  ],
  "CLASS_7_ONLY_STRATUM": "class_7_only",
  "CO_PRIMARY_FAMILY_ALPHA": 0.05,
  "DEFAULT_BOOTSTRAP_SEED": 20261023,
  "DEFAULT_MAX_SKIP_FRACTION": 0.01,
  "DEFAULT_N_DRAWS": 10000,
  "DEFAULT_SUBSET": "all",
  "DRAW_BATCH_SIZE": 1000,
  "P_VALUE_TOLERANCE": 1e-12,
  "UNSTRATIFIED_STRATUM": "all_clusters"
}
```

## Code pins (221 tracked src/ and scripts/ files)

| Path | SHA-256 |
|---|---|
| `scripts/benchmarks/run_metal_model_comparison.py` | `0553e13d89e7c5a8d482d7111b32bd0880443305e240981257e7fb884a12f9f5` |
| `scripts/benchmarks/run_pinmymetal_comparison.py` | `727e0a616bf0ab30be3077597097971921e19036954c802cbfd4f36c674db734` |
| `scripts/benchmarks/run_pinmymetal_context_campaign.py` | `c4cc4036a7129e53afde441e34def21c1157193232f7dae1dae3abcebb25a836` |
| `scripts/benchmarks/run_pinmymetal_encoder_lr.py` | `3dfea060da71ff193afbbcfe0610732b0dbb7f04d4dd02aa7ebc1aeaaa92a7a9` |
| `scripts/benchmarks/run_pinmymetal_five_class_screen.py` | `d54f43b1285defc96279b0c54d9b6ca54673c488da162f3f8040396b7671b5cb` |
| `scripts/benchmarks/run_pinmymetal_reference_eval.py` | `0ee0555891f2a5b0bb79d869236606769ecd29db4b5a71134111d6eb741c18f0` |
| `scripts/benchmarks/run_pinmymetal_schedule_comparison.py` | `23edbdd26817dde144f23fdde4d3d9c4757b98bb66771d1aa6895026593b662c` |
| `scripts/colab_5fold_watchdog.py` | `3051d3e8462db617613fcd494bb58d5a42327679487c0e2d4bb676c68835254d` |
| `scripts/colab_artifact_streamer.py` | `3a348b171c5b2a1eb6bc619dac1255a4524778c622bf2a85ff713b6633e5527d` |
| `scripts/colab_metal_pilot_bootstrap.py` | `4ef0457f54f98edfcd3a41434dcba7849133bdae47bc49c63c0222776fe542b0` |
| `scripts/colab_metal_pilot_resume.py` | `a85a1a0f7b58d6ea0bb57915282921c830e083250a00ad666ea650765c771a88` |
| `scripts/colab_serial_metal_host.py` | `617a844918b16036585b84ed3b2976d340c21a9a3b327250674c55007d1f8b82` |
| `scripts/colab_serial_metal_supervisor.py` | `804a58e18545be5e5a60583579ab78aee5bd9dfac351c81fb642a8f6e42b8247` |
| `scripts/colab_smoke_readiness.py` | `661b1620765a2666a9581f2d0af76bbc77131a1143456f8b5dd82c5fb7e6da30` |
| `scripts/data/build_pinmymetal_folds.py` | `29fc35a40014cf793a0fcc5709dce356da8933002ff0ebed778105ad493fac18` |
| `scripts/inventory_ec_support_sources.py` | `f394ee4224fa7f088afa88a32ae8d3f3946316ad1d802551a0404a7863dd4245` |
| `scripts/prepare_colab_smoke_snapshot.py` | `c7ec1ffb7c9e7ceb96eca32ecca305209f6db82aad7075fc05abf333f0cb8c3c` |
| `scripts/repair_metal_pka_cache.py` | `b4b31cf286fa921c12ce932c2ecb6f500208c5d2f1bf3cfa99ba0f2958389f5d` |
| `scripts/report_ec1_component_support.py` | `ca708a040cf74c1d4516dff8f6347b4b8f032ca87f8d4991d3057adc33fbf1ac` |
| `scripts/reproduce_zenodo_pmm_benchmark.sh` | `c96b21b4e9f783da9b97750529686acce8adcd946dd6bf29dc5f9ecf38317cb3` |
| `scripts/resume_gvp_capacity_colab.py` | `420a0f72d903963a05e0db8e48c2c1c13a5b4d90c685ff7a316d3ef8aa4a16c6` |
| `scripts/run_clean_split30_comparator.py` | `a103160ecfca8c99d921d53a3d80f10737fa047a509f7d5daa7de50858eb0168` |
| `scripts/run_ec_clean30_baseline.py` | `07b2f3315682a4767327b3746615113ed6ab242755251eedcedb11295e550a0d` |
| `scripts/run_ec_clean30_campaign.py` | `52e40a26775f6a881b0519a14df81bc14d051c0e580da5a98c2816aaf977b893` |
| `scripts/run_ec_clean30_identity.py` | `390bc1b4cf00f6ef1356c4d11def738b65efeedbf9ccd023c30457b50d728df9` |
| `scripts/run_exact_pinmymetal_5fold_cv.py` | `7a5fac451cff0a8b709c6c47b1ff4d5a9d38948d65e0ab2115e812f3a16d9366` |
| `scripts/run_metal_5fold_cv.py` | `77d80fb3f010bb93d1a69223a854a82bc31e923fe498352e7823b92a39689cf5` |
| `scripts/run_metal_5fold_cv.sh` | `7c79883d6dd110ca1d07a7b78b93a2c63769644d0a469510c1e26460dca1fd51` |
| `scripts/run_remote_homology_search.py` | `754683708565d8eb2e81f1cd48f56ecb60977915a9406c30cb01d6decf48614b` |
| `scripts/run_zenodo_pmm_exact.sh` | `d84e53f97e6e01c0a73380dcf9c000821c1e05f933d3aac6d448b18021d9ac39` |
| `scripts/run_zenodo_pmm_exact_5fold_cv.py` | `515b19b6b636e63e8ec66e5d59794b36cfeb991d8d2b6a3c0bd2ffc5e3a9fb9d` |
| `scripts/score_ec_clean30_comparison.py` | `41a78925f95735ee346268e6be0c4dc2dd3a4544db263e91f904a010ff7cb154` |
| `scripts/verify_zenodo_pmm_ion_dataset.py` | `d73910051a91797389b06815bcc1c7013b7d0a8bbd7a4d9e6103ba608ed5e998` |
| `src/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `src/analyze_metal_ec_association.py` | `0566cfc58571fbfb5f4bf2427ef3dc3849a6dd37111bbd9d30a4a5fe98376ad0` |
| `src/audit_metal_ring_inputs.py` | `cbac19bff2f89447ba4250df9c03f5923cb4e384a297725153b9c8c52609ad27` |
| `src/audit_pmm_deposition_crosswalk.py` | `0d6e9b62d93e0890591c12552f2add56e07d27d3518d6c66186d75f8e8df685b` |
| `src/audit_pmm_feature_crosswalk.py` | `74a42936adb7057473a79138c6c849f669a3e3155e87769cbc05f8b0e36395c8` |
| `src/audit_sequence_remoteness.py` | `9137ba063a09c98c2f3171795120cd880f8832c0d573fdfe8de8c7f323c1f08a` |
| `src/benchmarking/__init__.py` | `dfd92a3e07af22e915b017559c4e40f0c1b17ed17e86ead33cb1ff1de07ab35c` |
| `src/benchmarking/artifacts.py` | `6dbc3b449cf5ba98eea5e017ccba360add62d50fa19cde2fc4d7628d9d5a06aa` |
| `src/benchmarking/pinmymetal/__init__.py` | `e46842dc3f134f5723465fb751468e0357058b3d8ef7b2e1d77f2e2e23aa9e4a` |
| `src/benchmarking/pinmymetal/encoder_learning_rate/__init__.py` | `e46842dc3f134f5723465fb751468e0357058b3d8ef7b2e1d77f2e2e23aa9e4a` |
| `src/benchmarking/pinmymetal/encoder_learning_rate/assessment.py` | `038c6ea2f6f5eb8c723e306cc48c0110d06419786bc97337ed576295c5e8a060` |
| `src/benchmarking/pinmymetal/encoder_learning_rate/campaign.py` | `62496a3c0c1ef985a3c93ae474df1f35730b0b93b4f3b9fffcafe026875d2159` |
| `src/benchmarking/pinmymetal/encoder_learning_rate/followup_assessment.py` | `9606cc74d9b031709272b1454e6ec8ea6fc5a62ba30b2d69583e3c3eb1c8696d` |
| `src/benchmarking/pinmymetal/encoder_learning_rate/followup_campaign.py` | `68a1d43122aa0bb8b352acd341f3ccf8c4afd1804e86b3dd49ef58c9e703d7c5` |
| `src/benchmarking/pinmymetal/encoder_learning_rate/safeguards.py` | `7a98fbc9f3a5d7237e803d07e109341f41cb7edc5ae4dcc31378122b7e23121d` |
| `src/benchmarking/pinmymetal/model_comparison/__init__.py` | `e46842dc3f134f5723465fb751468e0357058b3d8ef7b2e1d77f2e2e23aa9e4a` |
| `src/benchmarking/pinmymetal/model_comparison/assessment.py` | `43b54b9214ebad9259b32c8a20a048b1e76c363b1380ff7f2294492b3e5d1efe` |
| `src/benchmarking/pinmymetal/model_comparison/campaign.py` | `bc3ce839ff2ec2af0981240799be6d0d58c48908c54f2915c62a867d829f419c` |
| `src/benchmarking/pinmymetal/model_comparison/coordinator.py` | `595d78aabef59ddedd4471fb6b31218e2f51e8faa8ec877f6884ed8fb5d79451` |
| `src/benchmarking/pinmymetal/model_comparison/evaluation.py` | `84bc88ccba518855e511cb1784e9e15ab827a4906415bd3a9a77c2f079a649ba` |
| `src/benchmarking/pinmymetal/model_comparison/safety.py` | `df3d0c3e8e4882e3ddb68d978413629222fb8731d6a78bde7748e9e4d97dd863` |
| `src/benchmarking/pinmymetal/model_comparison/session.py` | `28565f29c67a3bbc0291a12ee70fb7facc15289fbe03a8f7b67be0f26cead575` |
| `src/benchmarking/pinmymetal/near_copy/__init__.py` | `e46842dc3f134f5723465fb751468e0357058b3d8ef7b2e1d77f2e2e23aa9e4a` |
| `src/benchmarking/pinmymetal/near_copy/assessment.py` | `b212633192e2e94e3b05420132469623b86dc84a37358366158e878f3bbc3b61` |
| `src/benchmarking/pinmymetal/near_copy/bundle.py` | `68df443734dfadee2aa4afcd2f646bd7ce541d2173ca46632c77f31375fdcea3` |
| `src/benchmarking/pinmymetal/near_copy/campaign.py` | `5635b63cd46c3c2f5451a037c1cd1003974238a414ad168a5b133a703ecc07de` |
| `src/benchmarking/pinmymetal/near_copy/confirmation_assessment.py` | `c2480d6225fe0d71861dab5bc768b0d0a445ce448787715ec7178c7169077f02` |
| `src/benchmarking/pinmymetal/near_copy/confirmation_stage.py` | `2d7405e5a13bb84c90436a4cf777f32489ae630d1c25a05a04d784c33ca62475` |
| `src/benchmarking/pinmymetal/near_copy/folds.py` | `9cebbd8ba603160349943197bd5680cc4555df6f368598c0cde7bef21ee549e4` |
| `src/benchmarking/pinmymetal/near_copy/host_pull.py` | `c67dccb4fc6b736e5b5aef8773c80db62841797a9ff15637cd45c2ade6003ed4` |
| `src/benchmarking/pinmymetal/near_copy/probes.py` | `6396bdad2b11fafc2d039bf8a3bf8fd40edc9a43095041dd1bdde013ba5dda73` |
| `src/benchmarking/pinmymetal/near_copy/reference_evaluation.py` | `d473ff18d1fa55d4430d2e3a93734f3baa61782a8002530002e280683dc7bda6` |
| `src/benchmarking/pinmymetal/near_copy/reference_report.py` | `9c4e78bdf7d761ca273363ecab0fce42c77ba72cca7add4ae11f571f7f276268` |
| `src/benchmarking/pinmymetal/near_copy/regularization_stage.py` | `31303df23174e6ed2fce2e3c71d8ddcfca23dd3c350abcb5682bddeade469c9c` |
| `src/benchmarking/pinmymetal/near_copy/schedule_assessment.py` | `4834ab6866c2247342b17b270e3d6cb1422b94c0ad514bdd0f320ff1ebe050d1` |
| `src/benchmarking/pinmymetal/near_copy/schedule_comparison.py` | `1f3f0cd26a55b12e7756ee84c716efa2f9bc0d32ced9cc3637eceb9af014f6d5` |
| `src/benchmarking/pinmymetal/near_copy/screening_stage.py` | `e059a14774c4ff5dacbbe502758f3a2cb754f1218346435ac4d77b41065b1c1d` |
| `src/benchmarking/pinmymetal/near_copy/speed_report.py` | `af885bcf4d2934cbea445ae8c79510906c0def5dbf07cf776f1dc6ffa816d3ab` |
| `src/benchmarking/pinmymetal/near_copy/test_preparation.py` | `18953cc573b1e2fba0fdbf9b9463d426f039d9fe74e0aec52202af60749d844a` |
| `src/benchmarking/pinmymetal/near_copy/throughput_stage.py` | `8adf12197b9cefc5a77c77d00a82692121c864caef55b2b72c682a8efaae8f7b` |
| `src/benchmarking/pinmymetal/source_context/__init__.py` | `e46842dc3f134f5723465fb751468e0357058b3d8ef7b2e1d77f2e2e23aa9e4a` |
| `src/benchmarking/pinmymetal/source_context/assessment.py` | `9af580653110f588988a43b2bb6e2504e39e128934a82b0a3458ac27f29493fa` |
| `src/benchmarking/pinmymetal/source_context/replay.py` | `545a8db8a1db1b3de24a36e081ce1e23c80e0f32a1a15a6f1b60947bc9d21345` |
| `src/benchmarking/pmm_campaign_bundle.py` | `43718604c24cda7a436a5ef8756a52889ba829a566eb50c36707049212429aaf` |
| `src/benchmarking/pmm_comparator.py` | `2889bfc4686bd47f263fdff4bf86805d3ecebbeedaa992a3a2b5965fde2d858e` |
| `src/benchmarking/pmm_execution.py` | `f83e25edffddfb95b3bc281d61decbc2398720456230713722219fd5cc0c7633` |
| `src/benchmarking/pmm_final_report.py` | `d3f705f64bcd7e00adbd0a885e467808c9f8eb86613f65e1b14df14bd544838a` |
| `src/benchmarking/pmm_ion_analysis.py` | `38d588cb463ac74412c15a46985f763e2dbe56318cf13b12d57e756c3160a5fc` |
| `src/benchmarking/pmm_ion_campaign.py` | `59dff8e6027a27f3c4efd44586710d32bf8580f8ee8673a8e2299e79712379e2` |
| `src/benchmarking/pmm_ion_cohort.py` | `c1e9de54d034ebba690cec7833ea56af2d6389747e83f71ed69981f260062bfd` |
| `src/benchmarking/pmm_ion_features.py` | `e4f710d713cffea3375fd709dc52c5a85a1f5492661d61b33f45ed086c2815f0` |
| `src/build_clean_single_donor_subset.py` | `3a8f2771002e35029ac3416c74fedc94c78cc48d5c2829795ebe1ec36952d34c` |
| `src/build_colab_bundle.py` | `b606f785286468de538ebd6e0cec3b2f52f4952f575fe8c81a21192a76cd1afc` |
| `src/build_dataset_csv.py` | `60f0865ed6e79c0d2783dd5a81d815f2acd28d5781b229f28f11c69c2d646472` |
| `src/build_pmm_site_dataset.py` | `d72889cb4ef6f5cf5b0d7ca1d52b66ccf65843dba4692e6e109aa759cd67d316` |
| `src/complete_care_caches.py` | `e4d4e6021df33c26d5511c15c27ea8269d84d0c890d4897f8b71b90777609ad2` |
| `src/data_structures.py` | `a931ee53961b10af424b95419b47069442c0d9b342ef762318d5716c50b88ce6` |
| `src/diagnostic_metal_pilot.py` | `29c71242faad2dbe08427f36bb1c7e79812da2de0cdfc7c135f773555c87d4fb` |
| `src/ec_auxiliary_objectives.py` | `fa9720128ff970f259b173526a815f15cf913cde175a336a461226472f77719a` |
| `src/ec_clean30/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `src/ec_clean30/baseline.py` | `9cec976db1c6f241f7b4aad247e6fce9e3e233c9c0ffa37905962d529023e795` |
| `src/ec_clean30/campaign.py` | `f4cbf5afd374beeaf69dc932292acc2dd2c5c5577cd78e1993541565d0325726` |
| `src/ec_clean30/cohort.py` | `514f5f573c65afa1535d8b044198899ec2d1c348f110850cfd4029eef1df8e23` |
| `src/ec_clean30/commands.py` | `6163c9836b2f6a4abdba1b7e33afbf464ae8bb71b8b4078df44ed185bfc9301f` |
| `src/ec_clean30/freeze.py` | `a37138fa530395d01bdfb9aec3ae5a9cd6aafbb425e33fa7849105b9da6d2195` |
| `src/ec_clean30/identity.py` | `8391ab8abe955820a3162dfd30c83872ffab9bf4e2823428f5293855e7258ba2` |
| `src/ec_clean30/logit_adjustment.py` | `7fdd280c98ef8eb52731e1ed496da89d2fcba1084baa4421c6c57e551b856387` |
| `src/ec_clean30/overlay.py` | `d0f50244af6007d1e7193de08b3069112e43409ea1bc6abcd5636c3f5defda1d` |
| `src/ec_clean30/provenance.py` | `9173c3a041f1190aa2a4028c187fa2572b90b39cf6c8be589c4abfa1a2235d07` |
| `src/ec_clean30/r2_lists.py` | `895775ce71469ac17bc268ebfc8f71d8513c2dd31d751fc1cfebc35eaaaee14e` |
| `src/ec_clean30/scoring.py` | `938d5b48e2056f5275fd93f30a38e98b9a216de495918d096278a5140a16a5e1` |
| `src/ec_clean30/selection.py` | `be4a5137787695d8ba32c90b3d2bd961cd6dcb0af8d597591c9d45e1edfa77f7` |
| `src/ec_clean30/selector.py` | `da88a4deecba771bf686a9d54ad50863286c42ae2a2d55c4228658e05794c85b` |
| `src/ec_clean30/simulation.py` | `7ea39a0eec562d1515a67d7a60a4a9b0e502739cd39310bc20c4bc105ae4c367` |
| `src/ec_clean30/stats.py` | `0c97cb4ba312b65010800d39ea22bcd10027d62870f24ca2731385fc0b018f09` |
| `src/ec_clean30/transfers.py` | `151008db8ff895f7cb05214624bd072f504bf2adddd62f84e4abcc7059aee201` |
| `src/ec_clean30/views.py` | `4acfe32fe834825bb1058ad3a5bef903d61561f1017b7cb349ef4975bcceb4b3` |
| `src/embed_helpers/Interaction_edge.py` | `626a2f7eb8eba02a05aef7e9217a61684c720083e667df407b1306033a9120a1` |
| `src/embed_helpers/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `src/embed_helpers/esmc.py` | `1651505cf27a36ae0d1e99af208a51863930c93e2a4161313a1a393bafeb7a07` |
| `src/embed_helpers/esmc_chain_means.py` | `3ede7f5d26e4a09a9ad1ec6f60e3933f865826031b08804a3372b68f2d6e4d9d` |
| `src/embed_helpers/physicochemical_features.py` | `07a2081751e5870402628c4c17b73d8abbe6d87f2cbb7f68a55d26251ff8f64c` |
| `src/export_validation_predictions.py` | `a7697d2cbe1beacb814dcf983ce40cc7241166c94706cfb59d0db5b68fe7b754` |
| `src/feature_extraction/README.md` | `e09f9252b6f5a1359ca44f832fedaac884361a69277d708c4bc8aff6918159a9` |
| `src/feature_extraction/__init__.py` | `7ee6c2919149c0fb74ca772a356fd426b801b4cca4c5b7f0ba6bfde13a1732b4` |
| `src/feature_extraction/constants.py` | `35dfcd2d4a63a5306a1fe843578315702d76d80f89d2c14ad361eb2f6c8b815a` |
| `src/feature_extraction/core.py` | `d174e8ba755c9aa4cb202b57708438949dbfd37aad47e2a1d289bf6143a95368` |
| `src/feature_extraction/generate_features.py` | `72a5c24246c0f916961de09920082e4c01f3de4e22f66de90dafae900edb8d4d` |
| `src/feature_extraction/propka_support.py` | `7dedaf4f2cfb467673155567005fe53eeed8f82284850323ba6f64a565eb5d2e` |
| `src/featurization.py` | `af02dab90eaf8ad3c4fdfc41a25e303ead8586c29431d08784c72c8ad529385d` |
| `src/graph/__init__.py` | `9727f29a515573f7e0cbb0f052bb212d2182818466a3b548ba2a04702dfc7397` |
| `src/graph/construction.py` | `1c531cdde3ae3a1ed6abf2677b055774591f3605b6ffbcf9f3364d0ca21f1786` |
| `src/graph/edge_geometry.py` | `66811c344703a65111a0436cb21b0fb43905e5c1c45b4710f38fb69bdda557a8` |
| `src/graph/edge_postprocess.py` | `88d3c31b4e37ce0b6d4310ffe5410d831153c8a3ad0a00ee62d0e68e9c04e111` |
| `src/graph/edge_records.py` | `78cead0066f029e5f29ec00ffab191929df813cfb2014fbe97cec93c99408d39` |
| `src/graph/edge_sources.py` | `91baeed8a75ba328104934f18d400cdabdd64a2caed9240f1c91f53a77e5dcf2` |
| `src/graph/feature_utils.py` | `fd6d833e804d076df5f20be0074b62054fce6d8340e6ec4bca5d47dd3dd24bef` |
| `src/graph/ring_edges.py` | `aa96cfe0a24a753bd6e34d7de857e90ddd2285cf8372d4cd1509a4ade217eb7c` |
| `src/graph/shell_roles.py` | `bb3f10b121e07b60fdd264c2f9f9942134f84b0f7b74dbeb8d8d671a2b27e665` |
| `src/graph/structure_parsing.py` | `1adf7b8ff95e2cb073d79bd0b5af6a2e79179c33a5f46b7ab0c6e9a07a364f44` |
| `src/label_schemes.py` | `d298d0b970d099091417216533c79abff5ae969ae84c74d2f044fabed510ab99` |
| `src/manage_structure_store.py` | `ed7949ec0c84826c53d2c971279e7ea4c5845e41c56971262e45dd1bb97a306e` |
| `src/metal_objectives.py` | `298681c7ef6d4b88bb96ccf040b3fbc04bc6bc32dda9109cbb1312343bb6d4db` |
| `src/metal_release/README.md` | `80a2b1e9e26d8c002e21a4f5686a3770ee8eabd82dc8af013014ae57ba0e555f` |
| `src/metal_release/__init__.py` | `b22ab8ad00b402ab4c8d6516cfc4fec392a8e454b59b825f15e36a3d1fccdc83` |
| `src/metal_release/__main__.py` | `6e47c60589d0d52ceb0d3af0f45bf3ee0df658cb799819111968ce5430bb55b7` |
| `src/metal_release/audit.py` | `56b9b737ebf28634f797d51e284e4fa896e230a512240775cc3d2a66334cc016` |
| `src/metal_release/benchmark.py` | `76bdbed5bfe8494dd32f024e0693353acb772e31e0db69185cf57b89ad396e6c` |
| `src/metal_release/catalog.py` | `bfc393c6375923dd2c10d55700a0004a8d846fc7736f18c63dac4cb841312c03` |
| `src/metal_release/inventory.py` | `831fce974c31e2694c1df3fecbf6d6b843e604c23e13a64ac99d9da4b2c6afa9` |
| `src/metal_release/manifest.py` | `1a393c9671f85b9f13c4dda8c310a226de94b4e2ec40d58bd2bac7e1596fba1d` |
| `src/metal_release/parity.py` | `637bedfb61a9334d8d9ca90625dba7f96f8de172af0fe11a1397329f36538fd4` |
| `src/metal_release/predict.py` | `7be4d35ec86dd781270df4c960a8e9b7223cf5ab43762f5a9eb6e529fd6f2efc` |
| `src/metal_release/selection.py` | `39d60f0d00bcb72f9502da4d2968a7c99e78f6b00f359870b09377dad0987546` |
| `src/metal_release_cli.py` | `203155448f69156c38524c21b5891790291e32d4f2b5448f5739343c5f21a38d` |
| `src/model.py` | `035ed0425af519f7cc2149e85acdda7b310d93be8571d78bc297b5aae70eee94` |
| `src/model_variants/__init__.py` | `a7625910fbe38737e588b81ebdb0d36a919c3edb7814882c648e9b736d240c8b` |
| `src/model_variants/factory.py` | `58df09974be6564b575f97bc1a98f621755c5fd24ab4235f73bfe84310f27508` |
| `src/model_variants/models.py` | `6792a0b2f21082f61659170403cb0938b87b789f0eab3df4b718d8f1fb2484a7` |
| `src/pmm_source_release.py` | `d92445773c7de6bfd4a6fefb0e437777b6ae074afb330cbe3e07c1362e0b8176` |
| `src/project_paths.py` | `f89a1d64f19632a412fb615f77200d61f42d3fa92720380fe2da69c3ec5193ae` |
| `src/rebuild_care_complete_bundle.py` | `8f1357dacc525c6e8419fa032758e68b79f1cf5ba44e9ece538900221293c600` |
| `src/report_paired_metal_routes.py` | `f5bb7bf4b4b8da8e9b28b1bfe4deef6d0b644b6d576cfe2dd26cef3cbd250549` |
| `src/report_remote_homology.py` | `de3cf20e6ec6338e82a469134ff2483dbe08f4260118a31cf963426e491560be` |
| `src/report_runs.py` | `eb2fb9d00a3be927869ee93878ba9036c87f30033b4f0b1bb879399457a6c7ce` |
| `src/requirements.txt` | `a6be9fb23ead6fa00fb56cfca4d4296bc440e611baa0c98a75808063307c975e` |
| `src/review_pmm_feature_matches.py` | `bf285f5812bf533694aa2629c631d9d6ab6e99216cdd217bc758a0cbbbf884a8` |
| `src/run_ec_baselines.py` | `f9d778de108eb51f6c931433618e6ba8db4aa2ec85d33cec8929f68dafb2cbc7` |
| `src/run_metal_architecture_pilot.py` | `7588fefade48b644c4cd07fd02aaa57355ffd51191fb370dbc9f70642f62bff1` |
| `src/run_metal_coordination_geometry_pilot.py` | `08145f5d07aa943b0053d7a6f895b6938f00239670d87b70454066de01ff588a` |
| `src/run_metal_ring_pilot.py` | `85aabd93c0281d7d1f007c7d8a7d31747bb10a77f50647996ebf3d6774ce1847` |
| `src/run_metal_single_gpu_campaign.py` | `ebd9606284a283dca431d65ab71e5c4fbd266fd2449582305eb65b0d60f09e96` |
| `src/serial_metal_campaign/__init__.py` | `aa77add656cb468b5b03d4921d4d7b34099abc5fac360c9f3920750c9b318dd1` |
| `src/serial_metal_campaign/adaptive.py` | `1fd89203828c9a507cb5f2aec06c28d21c777ce074f375ed4dba75c7d43d8bc5` |
| `src/serial_metal_campaign/budget.py` | `17358bbae22186fca643547f4b2489e1bd02409e73bf461e943bbdf3d69edd0b` |
| `src/serial_metal_campaign/control.py` | `bd6d002792ca054ea38acc20e9129a3011e938be2a8eca23e53a3b54b113dc80` |
| `src/serial_metal_campaign/evidence.py` | `b449e8ff641bf7b63461895c37d5b22c2aaa3c5a11feabe766d51a7f4e006475` |
| `src/serial_metal_campaign/portability.py` | `58d6ff957a388e0adf3709825dcc6f289b7e867798a60cd96856adf5837598c4` |
| `src/serial_metal_campaign/profile.py` | `b2205e55a494fe73107ba33978cd7baf936da6838012fb0f9ecb363abfd892e0` |
| `src/serial_metal_campaign/remote_queue.py` | `a5f24fec568522da556f9cdd52fc1d775e2a86b656298feb8ddfe307ee48f53a` |
| `src/serial_metal_campaign/reporting.py` | `86f9ac98e7436cead422d7239428b6aafd03e6811272d60a87d3e967b31ad270` |
| `src/serial_metal_campaign/runtime.py` | `38db67cc3ed00f61b772720a9b855bad701eb3cee0af07fba3c4fcace93789d0` |
| `src/serial_metal_campaign/supervisor.py` | `d3041ab54dc210df52e7abc7958fb9d7674c2ec74c158631d796d650f6ceca9f` |
| `src/serial_metal_campaign/workflow.py` | `531bcf942703eb333a639043010fef84dab1735cbb523531ec75ae7822ff4b31` |
| `src/stage6_standalone.py` | `bb2c968cd32b81b291eb44f47e20d53f75f1bed91c9daa92c9ed42bea6179750` |
| `src/statistical_validation.py` | `fc441a45117dcc3d24c3b730b355e3b61f45b2f38495a47a53e527eb05995057` |
| `src/structure_store.py` | `1d494a6b3f98e26032a8ead10b3d72efbd5ed4c27932a8b5ca58c384a48306ec` |
| `src/summarize_pmm_feature_crosswalk.py` | `4ac41158834f327b4d59593b26c892394605b35e4f5f3bcebe49ce8cbb07423c` |
| `src/train.py` | `f0b25cbb778f8863037999afa178b0f2b90517e209f9c3f73a450d7077b516f8` |
| `src/train_ec.py` | `db76df9c299667262bdf240e725df32dc27f8c7bc639b28ec073cb51c9d46a1d` |
| `src/train_metal.py` | `cc830557a5ef9d407ec398863a7633d2d7c93078ee45233520aa2358cb603439` |
| `src/train_serial_metal_profile.py` | `67ac95b46e74917a294fc326f44e23bf4e76705bcc1c1e536e5ea589f6be926b` |
| `src/training/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `src/training/access_guard.py` | `f10ce795f9c3b8adb274a57f10ea55cae93b04471235d09711ae823faf23ada7` |
| `src/training/campaign_runtime.py` | `b66a29df6234ca64f1702f3ac237a4972445e553c5ebbbe4e5c40b7bce77b183` |
| `src/training/config.py` | `17b8e4940875414489b7dd8aee4aee92011fda1c45a6cf4cf49c40cc4f5f6507` |
| `src/training/data.py` | `586d0c7943e05d658468690e9a23fd80da31a402f50ec28df5657722834c23ef` |
| `src/training/defaults.py` | `6cfd115fd29bb4a7e3bfc88ec4ac2574234491e8418962553baaf982a6f6cccb` |
| `src/training/ec_class_metrics.py` | `a107ba15bb898d428b18d6f368354336f4c783c683c665db3127a9b9cda9ca31` |
| `src/training/ec_classes.py` | `3333dc64cc2a046dfa73cbcc06d32999e6352f3b2280dd40b97a18009f55889b` |
| `src/training/ec_membership.py` | `9516da9aee394586eefebc2001d8b74f15dee1a8dc9d1c3bc1d13f4dd6f83fa1` |
| `src/training/ec_prediction_export.py` | `32cb8cef3c0c389c7a80b1245751948b68fa40fe4caa453e384c8cb2b8da4c3b` |
| `src/training/esm_feature_loading.py` | `57711f0fc7676c6b40cc6ee41264768cbf3241924bf33156b201bae9a496d1f8` |
| `src/training/evaluation_protocols.py` | `c06bbca83344e67fcaab3986a0697d2708e0d2ccda7fc7fd1c174a4332494c11` |
| `src/training/explicit_membership.py` | `27de817a241c5db930bc3732e2efd832fc15ed6e8623c13c4d090f8289decb33` |
| `src/training/external_feature_loading.py` | `7295d366f15a2b8459c0213c32373036f0dfe1b7c1a8a21d8c31ced50607e898` |
| `src/training/feature_paths.py` | `ea9587dc3b2b97e128777d7fe4794d348816ffb0f793b053e491744cbee27aef` |
| `src/training/feature_sources.py` | `2d2dab5680c863292d413c1df3f072172e22611f722ef5bd700942a2617862f6` |
| `src/training/final_test_reporting.py` | `bb4786238b344a159db93144db621ee0644ad87812ba76025cb0163bebdae7c2` |
| `src/training/global_esm_features.py` | `210b2856910abd6a4293b66aacb47e371ef60711f95f561a691e6e1666bd9e15` |
| `src/training/graph_dataset.py` | `d0e42cca39cfc5432856a3c100fbf26223c5fb87e9eea8a67cb625d5a483282d` |
| `src/training/labels.py` | `000cc3820231891627c900b9b9ed9cb0b94138d78f141976d6290a429c37dcdc` |
| `src/training/loop.py` | `92438ea074279563343c1979dd3bf0bce8555cfa0d19c23cde774c477b4c17b3` |
| `src/training/metal_examples.py` | `27673c7c8480191b32d9b2238cea5323de479d8fe3e1cd838a460e8d824f2cfa` |
| `src/training/parallel_loading.py` | `fd9c5cd84532e904ffbd35dc00ebb6fe5862eda8dacbab2b715548af076e3b08` |
| `src/training/pipeline_metric_policy.py` | `6904a7f5a3ac5a256865275c7a95d0bbff01566d49312bcd25479dce505fc6cb` |
| `src/training/preflight.py` | `3703299d7d033289f22488ab360aa4a85b661935e565ebcc840b5e31e49090bc` |
| `src/training/raw_graph_cache.py` | `c2ca28d720b13a8b3b2a02ba37b4311695d2253e1e421d46a6820618495c7800` |
| `src/training/reproducibility.py` | `20449d096fee8575bda5e896f42c3d8c15753f31db49f590ffac29090d98c2d9` |
| `src/training/run.py` | `a753cb5928b4a1489e4faf6228a0b88b848a64182420e32a37fdd3d0ddfe1d8d` |
| `src/training/runtime_preparation.py` | `b3bf92c6d777ec606e4b96315cc24e6d87733b50861f7d7997c37b8eaa7f23f4` |
| `src/training/site_filter.py` | `c159f151872726b2cf192d8308639e5c8905160dede497754daefa0954bbce5a` |
| `src/training/smoke_test.py` | `0e21eac71d0748c95cf9c569a8da7cab161df8a8fed81ce771cdd0834425e956` |
| `src/training/source_cohort.py` | `9b3da4ced44ff1da8cb5284f5780a7bc8cf15c03779c7527fe833f3783248018` |
| `src/training/splits.py` | `d308ee362014098c62534912d5bcaab9d0a584801543bc04b12a96c71acb9de8` |
| `src/training/structure_loading.py` | `74d5a33938762907cb26cf4d4a5aaa24f0e2550b46f3e27b779169e304ed02fe` |
| `src/training/task_entrypoint.py` | `28b1d5d60ca616994b31cd8808f9d21a69c49c9ec5a0f7f789b0c01db3bea853` |
| `src/verify_colab_notebook_smoke.py` | `cb5279ba0260b45c221777da7f3d070ef8ea353c9f1c64ddf78186df657baa7d` |
