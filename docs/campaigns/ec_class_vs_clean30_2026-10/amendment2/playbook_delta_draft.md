# Draft: EC playbook delta for amendment 2

**Status: draft for merge (2026-11-04).** The exact executable blocks belong to the
[EC playbook recipe](../../../EC_TRAINING_PIPELINE_PLAYBOOK.md#ec-class-versus-clean30-campaign-recipe), which is being
edited concurrently for execution v5. At merge, the subsection below is inserted after the S1-amendment block and
before "WP-13: measured G2 preparation"; existing blocks and anchors stay unchanged; this file is then removed. The
execution v5 blocks (pod session, queues, G2 v2) are owned by the execution side and are not part of this delta.

Variables are those of the recipe's first block (`PY`, `CAP`, `SMALL`, `C`, `O`, `R`, `X`, `E`). Upper-case words
(`SHA`, `NPZ`, `SPLITS`, `CACHE`, `HELD_OUT`) stand for values recorded at run time, as elsewhere in the recipe.

Text to insert (the outer fence is not part of it; its links are relative to `docs/`):

~~~~markdown
### Amendment 2 interfaces (planned continuation)

[Amendment 2](campaigns/ec_class_vs_clean30_2026-10/amendment2/amendment_draft.md) replaces the S1 rule, the primary
metric, the S2/S3 search and the gap gate for real execution under its chain; the Freeze-1 and S1-amendment blocks
above stay for replay. Every real action below refuses a chain that is not published and authenticated, and every
fit identity carries the fit chain. Plans v2 live beside the Freeze-1 plans, which are never rewritten.

Before publication (CPU, training side only; the hold-out lists are built in a scratch copy of the overlay's
identity files, and the overlay is never written; each fold's cache-free corpus table is built from that copy, and
the training-side union FASTA once):

```bash
D=docs/campaigns/ec_class_vs_clean30_2026-10; A2="$D/amendment2"; F=$(sha256sum "$D/freeze1.json" | cut -d' ' -f1)
for K in 0 1 2 3 4; do
  "${SMALL[@]}" "$PY" "$R" plan-v2 --fold "$K" --out "$C/plans_v2/fold$K.json" &&
  "${CAP[@]}" "$PY" "$R" pretrain-corpus holdout-lists --overlay "$O" --campaign-root "$C" --fold "$K" \
    --freeze1-sha256 "$F" &&
  "${CAP[@]}" "$PY" "$R" pretrain-corpus build --campaign-root "$C" --fold "$K" --official-split-dir SPLITS \
    --identity-work "$C/amendment2/f6_holdout/fold$K/work" || break; done
"${CAP[@]}" "$PY" "$R" pretrain-corpus fasta --campaign-root "$C" --official-split-dir SPLITS
"${SMALL[@]}" "$PY" scripts/run_ec_clean30_clean_sanity.py tie-exposure --official-training-file TRAIN_FILE \
  --fold K --r2-list R2_LIST --out "$C/amendment2/clean_tie_exposure.json"   # one file/fold pair per fold
```

Private inputs. The repository is public and `$C` is gitignored. The dry-run-v2 receipts, the tie exposure and the
F6 hold-out lists and corpus hold per-fold counts, accessions, absolute paths or host names, so each is written once
below `$C/amendment2/` and bound privately in the record, by path and SHA-256 only
(`{path, sha256, private: true}`); no byte copy of them goes into `docs/` or any other tracked folder. The record
build reads them in place; `verify-amendment-chain` checks them in the workstation's full checkout, which holds
them at their bound paths (a fresh clone has none and is refused). A pod checks the chain receipt and binds its fold
views without the dry-run-v2 receipts; a host that attaches F6 inputs or prepares the final CLEAN inputs holds the
fold's F6 files at their bound paths. Each fold binds an object of files: `tables.K.table` (`corpus.tsv`, matched by
the F6 attach) and `tables.K.receipt` (`corpus_table_receipt.json`, matched by the final CLEAN inputs), and
`f6_holdout.K.receipt` (`holdout_lists_receipt.json`, matched by the F6 attach). The repeated dry runs, at the frozen
source commit on a clean tree, and the record build:

```bash
for K in 0 1 2 3 4; do
  "${SMALL[@]}" "$PY" "$R" dry-run-v2 --view "$C/views/fold$K" --out "$C/amendment2/dry_runs/fold$K.json" ||
    break; done
P="$C/amendment2"; IN=(--bind "inputs.clean_tie_exposure=$P/clean_tie_exposure.json"
  --bind "inputs.f6_corpus.union_fasta=$P/f6_corpus/union_training.fasta")
for K in 0 1 2 3 4; do
  IN+=(--plan-v2 "$K=$C/plans_v2/fold$K.json" --dry-run-receipt "$K=$P/dry_runs/fold$K.json"
    --bind "inputs.f6_holdout.$K.receipt=$P/f6_holdout/fold$K/holdout_lists_receipt.json"
    --bind "inputs.f6_corpus.tables.$K.table=$P/f6_corpus/fold$K/corpus.tsv"
    --bind "inputs.f6_corpus.tables.$K.receipt=$P/f6_corpus/fold$K/corpus_table_receipt.json"); done
"${SMALL[@]}" "$PY" "$R" amend-2 --draft "$A2/amendment_draft.json" --decisions "$A2/maintainer_decisions.json" \
  --amendment-draft-md "$A2/amendment_draft.md" "${IN[@]}" --bind ROLE=PATH ... \
  --render "$A2/amendment.md" --out "$A2/amendment.json"     # ROLE=PATH: the public roles, committed files
```

F6 inputs for one fold view `V`, after the ESMC union cache exists. The union cache and its CPU parity sample use
`"$E" sequences ... --forward-precision fp32` (`--device cuda` on the pod GPU, `--device cpu` for the same-pod CPU
sample); the sample comparison is `"$E" compare --kind sequence_mean ...`. The pod-session block owns the full
commands. Then:

```bash
"${CAP[@]}" "$PY" "$R" pretrain-corpus attach --view "$V" --official-split-dir SPLITS --seq-cache CACHE \
  --seq-manifest-sha256 SHA --identity-work "$C/amendment2/f6_holdout/fold$K/work"
"${SMALL[@]}" "$PY" "$R" pretrain-corpus verify --view "$V"      # required before the fold's first F6 unit
```

Selection and futility under the amendment-2 chain (fold 0 shown; `select --step S2` takes the same flags as S1;
S4 runs only after a verified `continue`; folds 1-4 add `--fold0-step-chain "$C/selection/fold0_steps_v2.jsonl"`):

```bash
AM=(--freeze1-sha256 "$F" --freeze1-amendment "$A2/amendment.json" --campaign-root "$C")
"${CAP[@]}" "$PY" "$R" select --fold 0 --step S1 --plan "$C/plans_v2/fold0.json" --runs "$C/runs/fold0" \
  --components "$O/identity/fold0" "${AM[@]}" --clean-sanity-record HELD_OUT \
  --out "$C/selection/fold0_steps_v2.jsonl"
"${CAP[@]}" "$PY" "$R" futility --fold 0 --step-chain "$C/selection/fold0_steps_v2.jsonl" \
  --plan "$C/plans_v2/fold0.json" --runs "$C/runs/fold0" --components "$O/identity/fold0" "${AM[@]}" \
  --clean-r1-inner NPZ ... --clean-r2-inner NPZ ... --clean-sanity-record HELD_OUT \
  --f6-corpus CORPUS --r1-inner-list LIST ... --out "$C/selection/fold0_futility.json"
"${CAP[@]}" "$PY" "$R" select --fold 0 --step S4 --plan "$C/plans_v2/fold0.json" --runs "$C/runs/fold0" \
  --components "$O/identity/fold0" "${AM[@]}" --out "$C/selection/fold0_steps_v2.jsonl"
```

CLEAN sanity records for each CLEAN fit `FIT` (arms `r1`, `r2`, `c_esmc`, `r1_inner`, `r2_inner`); the in-sample
export is the comparator's validation export of `SAMPLE` with `--allow-in-sample-queries`, used for this check only:

```bash
Q=scripts/run_ec_clean30_clean_sanity.py
"${SMALL[@]}" "$PY" "$Q" sample --work-dir FIT --out SAMPLE
"${SMALL[@]}" "$PY" "$Q" in-sample --work-dir FIT --arm ARM --sample SAMPLE --export NPZ --out RECORD
"${SMALL[@]}" "$PY" "$Q" held-out --spec SPEC --out HELD_OUT                  # fold 0, eight inner fits
"${SMALL[@]}" "$PY" "$Q" disposition --record RECORD --decision proceed_disclosed --maintainer NAME \
  --note TEXT --out DISPOSITION                                                 # or --decision stop
```

Under the amendment-2 chain, `gap-gate` is refused: the S1, S2 and FUT records replace it, and the −3-point gap is a
descriptive field of the FUT record. `dry-run` keeps checking v1 plans for replay.
~~~~

## Changed commands and flags (summary for the merge)

| Command | Change | State |
|---|---|---|
| `"$R" plan-v2 --fold K --out "$C/plans_v2/foldK.json"` | new action; plan format v2 | implemented (staged) |
| `"$R" pretrain-corpus` with `holdout-lists`, `attach`, `verify` | new action group | implemented (staged) |
| `"$R" futility ...` | new action; a binding run writes the record, its pointer file and the FUT step record; a run that cannot bind is refused; `--smoke` writes the record only | implemented (staged) |
| `"$R" futility-oc --out F` | synthetic operating characteristics (`futility_oc.json`, written once) | implemented (staged) |
| `"$R" select` | new flags `--clean-sanity-record`, `--clean-sanity-disposition`, `--campaign-root`, `--fold0-step-chain`, `--amendment-chain-receipt`; plan v2 and the amendment-2 record under the new chain | implemented (staged) |
| `"$R" gap-gate` | refused under the amendment-2 chain | implemented (staged) |
| `"$E" from-model`, `"$E" sequences` | new flag `--forward-precision` (`sdk` default and historical, or `fp32`) | implemented (staged) |
| `"$E" compare` | new flag `--kind` (`chain_mean` default, or `sequence_mean`) | implemented (staged) |
| `scripts/run_ec_clean30_clean_sanity.py` with `sample`, `in-sample` and `held-out` (each with the fit's precision-policy log), `tie-exposure`, `disposition` (`--arm` for a held-out arm) | new script | implemented (staged) |
| `"$R" dry-run-v2 --view V --out F [--runs-dir RUNS]` | plan-v2 dry run against the declared pretraining-input stub; receipt written once outside the view, below `"$C/amendment2/"`, and bound privately; the in-view receipt unchanged | implemented (staged) |
| `"$R" amend-2 --draft D --decisions F --amendment-draft-md "$A2/amendment_draft.md" --bind ROLE=PATH ... --render "$A2/amendment.md" --out "$A2/amendment.json"`, `"$R" verify-amendment-chain ... [--expected-chain SHA ...]` | record build from the reviewed document and its rendering; the dry-run-v2 receipts and the tie-exposure and F6 inputs below `"$C/amendment2/"` bound privately (path and SHA-256 only), every other bound file committed; chain receipt through the general chain verifier on a full clone (`amendment_chain.chain_receipt_record`), checked on pods and per unit by `amendment_chain.check_chain_receipt` | implemented (staged) |
| `"$R" refit-recipe ... [--amendment-chain-receipt F]`, `"$R" futility ... [--amendment-chain-receipt F]` | verified refit recipe of one fold for the scoring manifest (rule A2-SEL-9); the receipt is checked instead of verifying the chain again | implemented (staged) |
| `"$R" pretrain-corpus build`, `"$R" pretrain-corpus fasta` | cache-free corpus table per fold; training-side union FASTA for the ESMC union cache | implemented (staged) |
| `scripts/run_ec_clean30_clean_sanity.py verify-all` | receipt over every record and disposition | implemented (staged) |

Planned commands get their exact block only when implemented and tested.
