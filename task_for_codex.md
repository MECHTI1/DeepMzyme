# DeepMzyme status update - 2026-04-30

## Colab bundle workflow status

Status: FIXED for the default local bundle-build path.

The non-overlapped PinMyMetal split contains real multi-metal structures. For example,
`1cob__chain_A__EC_1.15.1.1` has two catalytic metal sites in the site-level summary:

- `A_152`, `CU`
- `A_153`, `CO`

The training and Colab notebook paths use the included site-level MAHOMES summary CSVs,
so these sites remain unambiguous for training. The generated structure-level CSV artifacts
are metadata for inspection. In those artifacts, multi-metal structures are represented with
semicolon-joined labels such as `Co;Cu`.

`src/build_colab_bundle.py` now allows multi-metal structure-level CSV rows by default and
offers `--strict-single-metal-structures` for users who want the previous fail-fast behavior.
The Colab notebook default bundle filename now matches the script's default output archive.

# DeepMzyme status update - 2026-04-29

## Files changed

- `src/report_runs.py`
- `src/training/run.py`
- `src/training/loop.py`
- `task_for_codex.md`

## EC contrastive learning status

Status: VERIFIED for the exercised smoke-test scope.

Verified by inspection and command execution:

- Contrastive options are parsed in `src/training/config.py`:
  - `--ec-contrastive-weight`
  - `--ec-contrastive-temperature`
- Contrastive options are saved in `run_config.json` and `run_metadata.json`.
- `src/training/run.py` passes contrastive settings into model construction.
- `src/model.py` adds supervised EC cross-entropy first, then adds supervised contrastive loss only when `ec_contrastive_weight > 0`.
- With `ec_contrastive_weight == 0`, the contrastive branch is skipped by the existing conditional.
- The supervised EC cross-entropy remains active.
- `supervised_contrastive_loss` safely returns zero when a batch has fewer than two examples or has no positive pairs.
- `--ec-label-depth 1` was exercised by a real smoke command.
- `--ec-label-depth 2` was exercised by a real smoke command.
- Training and evaluation loss values in the exercised smoke runs were finite.

Hardening added:

- `src/training/run.py` now rejects negative `--ec-contrastive-weight`.
- `src/training/run.py` now rejects non-positive `--ec-contrastive-temperature`.
- `src/training/loop.py` now raises `FloatingPointError` if training or evaluation loss is NaN or inf.

## Reporting script status

Status: IMPLEMENTED AND TESTED.

Added `src/report_runs.py`.

Tested input modes:

- Explicit run directories with `--run-dirs`.
- Parent run directory with `--runs-dir`.

CSV output includes the requested run, model, contrastive, checkpoint-selection, split, overlap, validation, metal metric, and EC metric columns.

Optional figure output was exercised. Matplotlib is not installed in this environment, so the script printed a clear warning and skipped the figure without crashing.

## Reproducibility metadata verified or added

Existing or verified:

- Full config is saved.
- Random seed is saved.
- Dataset paths are saved.
- Task is saved.
- Model architecture is saved.
- Fusion mode is saved.
- Node feature set is saved.
- EC label depth is saved.
- EC contrastive settings are saved.
- Selection metric is saved.
- Label maps are saved.
- Normalization stats are saved.

Added or completed:

- `run_metadata.json` is now written for completed runs.
- `selected_checkpoint` is now saved in both `run_config.json` and `run_metadata.json`.
- `selected_checkpoint_epoch` and `selected_metric_value` are saved.
- Git commit hash is saved in config payload when available.
- Split name/type are inferred from known PinMyMetal paths when possible and saved.
- Test structure and summary paths are included in dataset summary when present.
- Held-out train/test overlap status is computed during test evaluation and saved in `test_report.json` and `run_metadata.json`.
- JSON output saving now converts tensors, paths, tuples, and non-string dict keys into JSON-safe values.

## Split and overlap reporting

Status: VERIFIED for smoke-run reporting.

The depth-2 smoke test wrote:

- `split_type: custom_or_unknown`
- `train_test_overlap_detected: false`
- overlap counts and examples in `test_report.json`

The generated report CSV contains:

- `split_name`
- `split_type`
- `train_test_overlap_detected`
- `overlap_warning`

Known PinMyMetal paths are labeled as:

- `non_overlapped_pinmymetal`
- `exact_pinmymetal_possibly_overlapped`
- `custom_or_unknown`

The exact PinMyMetal label includes a warning that it should be interpreted only as a secondary/reference result.

## Smoke-test output paths

Temporary smoke data:

- `/tmp/deepmzyme_ec_contrastive_smoke_data/train`
- `/tmp/deepmzyme_ec_contrastive_smoke_data/test`

Completed smoke runs:

- `/tmp/deepmzyme_smoke_runs/ec_contrastive_smoke_20260429_rerun`
- `/tmp/deepmzyme_smoke_runs/ec_contrastive_depth1_smoke_20260429`

Report outputs:

- `/tmp/deepmzyme_run_summary.csv`
- `/tmp/deepmzyme_run_summary_parent.csv`

Optional figure path requested during testing:

- `/tmp/deepmzyme_run_summary.png`

The figure was not created because matplotlib is unavailable.

## Exact commands run

Interpreter verification:

```bash
/home/mechti/miniconda3/envs/DeepMzyme/bin/python -c "import sys; print(sys.executable)"
```

Context inspection:

```bash
sed -n '1,220p' Plan.md
sed -n '1,260p' task_for_codex.md
sed -n '1,220p' AGENTS.md
rg -n "contrastive|ec_label_depth|selection_metric|selected_checkpoint|test_report|training_history|metadata|config" src/training src/model.py src/train.py
sed -n '1,260p' src/training/config.py
sed -n '1,360p' src/training/run.py
sed -n '1,320p' src/model.py
sed -n '260,470p' src/training/config.py
sed -n '320,930p' src/training/run.py
sed -n '360,750p' src/model.py
sed -n '1,260p' src/training/loop.py
sed -n '740,880p' src/model.py
sed -n '1,380p' src/training/splits.py
rg -n "split_name|split_type|overlap|pdb|source_path|structure_id|pocket_split_key|build_dataset_summary|summary" src/training src | head -200
find DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal -maxdepth 2 -type f | head -20
find DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal -maxdepth 2 -type f \( -name '*.csv' -o -name '*.pdb' \) | sed -n '1,80p'
find DeepMzyme_Data/DeepMzyme_Colab_Bundles -maxdepth 3 -type f -name '*.csv' | sed -n '1,80p'
head -5 DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal/train/final_data_summarazing_table_transition_metals_only_catalytic.csv 2>/dev/null || true
head -5 DeepMzyme_Data/DeepMzyme_Colab_Bundles/train_and_test_sets_structures_non_overlapped_pinmymetal/train_and_test_sets_structures_non_overlapped_pinmymetal_train.csv 2>/dev/null || true
sed -n '1,120p' src/training/site_filter.py
```

Smoke-data creation:

```bash
/home/mechti/miniconda3/envs/DeepMzyme/bin/python - <<'PY'
from pathlib import Path
import shutil
from collections import defaultdict

repo = Path('/home/mechti/PycharmProjects/DeepMzyme')
src_root = repo / 'DeepMzyme_Data/train_and_test_sets_structures_non_overlapped_pinmymetal'
out_root = Path('/tmp/deepmzyme_ec_contrastive_smoke_data')
if out_root.exists():
    shutil.rmtree(out_root)
(out_root / 'train').mkdir(parents=True)
(out_root / 'test').mkdir(parents=True)

def ec_depth2(path: Path) -> str:
    marker = '__EC_'
    ec = path.stem.split(marker, 1)[1].split(',', 1)[0]
    parts = ec.split('.')
    return '.'.join(parts[:2]) if len(parts) >= 2 else ec

def choose(split: str, target_count: int) -> list[Path]:
    groups = defaultdict(list)
    for path in sorted((src_root / split).glob('*.pdb')):
        groups[ec_depth2(path)].append(path)
    chosen = []
    for _label, paths in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        chosen.extend(paths[:2])
        if len(chosen) >= target_count:
            break
    return chosen[:target_count]

for split, count in [('train', 16), ('test', 8)]:
    for path in choose(split, count):
        shutil.copy2(path, out_root / split / path.name)
    shutil.copy2(
        src_root / split / 'final_data_summarazing_table_transition_metals_only_catalytic.csv',
        out_root / split / 'final_data_summarazing_table_transition_metals_only_catalytic.csv',
    )

print(out_root)
print('train_pdbs', len(list((out_root / 'train').glob('*.pdb'))))
print('test_pdbs', len(list((out_root / 'test').glob('*.pdb'))))
PY
```

Syntax and help checks:

```bash
/home/mechti/miniconda3/envs/DeepMzyme/bin/python -m py_compile src/report_runs.py src/training/config.py src/training/run.py src/model.py
/home/mechti/miniconda3/envs/DeepMzyme/bin/python src/report_runs.py --help
```

Initial depth-2 smoke command that exposed a JSON serialization bug after training completed:

```bash
PYTHONPATH=src /home/mechti/miniconda3/envs/DeepMzyme/bin/python src/train.py --task ec --structure-dir /tmp/deepmzyme_ec_contrastive_smoke_data/train --summary-csv /tmp/deepmzyme_ec_contrastive_smoke_data/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --test-structure-dir /tmp/deepmzyme_ec_contrastive_smoke_data/test --test-summary-csv /tmp/deepmzyme_ec_contrastive_smoke_data/test/final_data_summarazing_table_transition_metals_only_catalytic.csv --run-test-eval --model-architecture only_gvp --allow-missing-esm-embeddings --no-prepare-missing-esm-embeddings --allow-missing-external-features --epochs 1 --batch-size 4 --val-fraction 0.25 --device cpu --ec-label-depth 2 --ec-contrastive-weight 0.05 --runs-dir /tmp/deepmzyme_smoke_runs --run-name ec_contrastive_smoke_20260429
```

Successful depth-2 EC contrastive smoke:

```bash
PYTHONPATH=src /home/mechti/miniconda3/envs/DeepMzyme/bin/python src/train.py --task ec --structure-dir /tmp/deepmzyme_ec_contrastive_smoke_data/train --summary-csv /tmp/deepmzyme_ec_contrastive_smoke_data/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --test-structure-dir /tmp/deepmzyme_ec_contrastive_smoke_data/test --test-summary-csv /tmp/deepmzyme_ec_contrastive_smoke_data/test/final_data_summarazing_table_transition_metals_only_catalytic.csv --run-test-eval --model-architecture only_gvp --allow-missing-esm-embeddings --no-prepare-missing-esm-embeddings --allow-missing-external-features --epochs 1 --batch-size 4 --val-fraction 0.25 --device cpu --ec-label-depth 2 --ec-contrastive-weight 0.05 --runs-dir /tmp/deepmzyme_smoke_runs --run-name ec_contrastive_smoke_20260429_rerun
```

Successful depth-1 EC contrastive smoke:

```bash
PYTHONPATH=src /home/mechti/miniconda3/envs/DeepMzyme/bin/python src/train.py --task ec --structure-dir /tmp/deepmzyme_ec_contrastive_smoke_data/train --summary-csv /tmp/deepmzyme_ec_contrastive_smoke_data/train/final_data_summarazing_table_transition_metals_only_catalytic.csv --model-architecture only_gvp --allow-missing-esm-embeddings --no-prepare-missing-esm-embeddings --allow-missing-external-features --epochs 1 --batch-size 4 --val-fraction 0.25 --device cpu --ec-label-depth 1 --ec-contrastive-weight 0.05 --runs-dir /tmp/deepmzyme_smoke_runs --run-name ec_contrastive_depth1_smoke_20260429
```

Final syntax and help checks after patches:

```bash
/home/mechti/miniconda3/envs/DeepMzyme/bin/python -m py_compile src/report_runs.py src/training/config.py src/training/run.py src/model.py
/home/mechti/miniconda3/envs/DeepMzyme/bin/python src/report_runs.py --help
```

Report-generation smoke with explicit run dirs:

```bash
/home/mechti/miniconda3/envs/DeepMzyme/bin/python src/report_runs.py --run-dirs /tmp/deepmzyme_smoke_runs/ec_contrastive_smoke_20260429_rerun /tmp/deepmzyme_smoke_runs/ec_contrastive_depth1_smoke_20260429 --out-csv /tmp/deepmzyme_run_summary.csv --out-figure /tmp/deepmzyme_run_summary.png
```

Smoke-output verification:

```bash
/home/mechti/miniconda3/envs/DeepMzyme/bin/python - <<'PY'
import csv, json, math
from pathlib import Path
runs = [
    Path('/tmp/deepmzyme_smoke_runs/ec_contrastive_smoke_20260429_rerun'),
    Path('/tmp/deepmzyme_smoke_runs/ec_contrastive_depth1_smoke_20260429'),
]
for run in runs:
    cfg = json.loads((run / 'run_config.json').read_text())
    meta = json.loads((run / 'run_metadata.json').read_text())
    history = cfg['history']
    losses = [record[key] for record in history for key in record if key.endswith('_loss')]
    assert losses, f'no losses in {run}'
    assert all(math.isfinite(float(value)) for value in losses), f'non-finite loss in {run}'
    assert cfg['config']['ec_contrastive_weight'] == 0.05
    assert cfg['config']['ec_contrastive_temperature'] == 0.1
    assert cfg['selected_checkpoint'].endswith('best_model_checkpoint.pt')
    assert meta['selected_checkpoint'].endswith('best_model_checkpoint.pt')
    assert meta['normalization_stats']

depth2 = runs[0]
test_report = json.loads((depth2 / 'test_report.json').read_text())
assert 'test_ec_level_1_acc' in test_report['metrics']
assert 'test_ec_level_2_acc' in test_report['metrics']
assert test_report['split_type'] == 'custom_or_unknown'
assert test_report['train_test_overlap_detected'] is False

with open('/tmp/deepmzyme_run_summary.csv', newline='', encoding='utf-8') as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 2
assert 'split_type' in rows[0]
assert 'train_test_overlap_detected' in rows[0]
assert rows[0]['ec_contrastive_weight'] == '0.05'
print('verified_runs', ','.join(str(run) for run in runs))
print('summary_rows', len(rows))
PY
```

Report-generation smoke with parent runs dir:

```bash
/home/mechti/miniconda3/envs/DeepMzyme/bin/python src/report_runs.py --runs-dir /tmp/deepmzyme_smoke_runs --out-csv /tmp/deepmzyme_run_summary_parent.csv
```

Final inspection:

```bash
head -5 /tmp/deepmzyme_run_summary.csv
git status --short
git diff -- src/training/run.py src/training/loop.py src/report_runs.py | sed -n '1,260p'
```

## Plan.md remaining work not yet captured

- [~] PARTIAL: Created `notebooks/DeepMzyme_training_colab.ipynb` with Google Drive mounting, Colab bundle unpacking, configurable model/task/hyperparameter selection, baseline-first presets, CLI training execution, copying outputs back to Drive, running `src/report_runs.py`, and displaying the summary CSV/figure. Lightweight JSON/IPYNB structure validation passed locally, but real Google Colab execution/training has not been verified.
- [x] DONE: Updated `README.md` to mention `list_train_commands.md` and `notebooks/DeepMzyme_training_colab.ipynb`.
- [~] PARTIAL: Added `tests/smoke_checks.py` for fast CLI/config/docs checks without writing into `DeepMzyme_Data/`; still TODO: add a simple CI workflow and/or dataset-backed smoke fixture if needed.
- [ ] TODO: Run real baseline-first experiments on the non-overlapped PinMyMetal split, starting with Only-GVP, Only-ESM, GVP + late fusion, and GVP + early fusion.
- [ ] TODO: Rebuild or validate the Colab bundle for the trusted non-overlapped split so it includes train/test structures, site-level summary CSVs, and an explicit structure-vs-CSV metal/EC consistency check.
- [x] DONE: Added configurable `--metal-loss-weight` and `--ec-loss-weight` options with defaults preserving previous behavior.
- [ ] TODO: Define and run the EC label-depth experiment ladder beyond level 1, then report selected held-out EC metrics only after validation-based model selection.
- [ ] TODO: After selecting checkpoints by validation metrics, report final held-out test metrics on the non-overlapped split for metal 6-class, metal collapsed 4-class, and supported EC levels.
- [?] REVIEW_NEEDED: Confirm whether final real experiment outputs, not smoke tests, clearly preserve split identity, overlap status, selected checkpoint, validation metric, and held-out test metrics.

## Remaining risks

- The smoke runs are intentionally tiny CPU checks. They verify the code path and output structure, not model quality.
- The depth-2 smoke used a `/tmp` subset, so split inference correctly reports `custom_or_unknown`; full trusted non-overlapped split runs should be labeled from the canonical `DeepMzyme_Data/...non_overlapped_pinmymetal...` paths.
- The contrastive branch was exercised with a nonzero weight, but the report does not yet break out supervised EC loss and contrastive loss as separate history fields.
- Optional figure generation could not be fully tested because matplotlib is not installed.
- `git status --short` showed unrelated pre-existing/generated files outside this task, including staged `src/model_variants/__pycache__/*.pyc` entries and other `__pycache__` directories. They were not used as task outputs.
