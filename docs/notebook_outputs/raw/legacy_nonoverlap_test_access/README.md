# Legacy Non-Overlap PinMyMetal Test Access

This directory is a tracked, byte-for-byte copy of lightweight evidence that was
previously available only under the Git-ignored local path
`DeepMzyme_Data/notebook_outputs/runs/`.

## Scientific interpretation

The non-overlap PinMyMetal test was historically evaluated in seven early
Only-GVP runs and is therefore not pristine or unopened. Whether those values
influenced subsequent selection is not established by repository evidence.
These test metrics must not be used for current HPO recommendations or model
selection.

The runs were created on 2026-05-01 through 2026-05-03, before the later
validation/HPO model-family batches. All used:

- task `metal`;
- six metal classes;
- Only-GVP with radius-only edges and conservative nodes;
- seed and split seed `42`;
- batch size `8`;
- weight decay `1e-4`;
- hidden scalar/vector dimensions `128/16`;
- edge hidden dimension `64`;
- four GVP layers;
- edge radius `8`;
- `pdbid` validation grouping with fraction `0.15`;
- non-overlapped PinMyMetal train/test roots;
- `run_test_eval=true`;
- 352 test pockets;
- `train_test_overlap_detected=false`.

The overlap result addresses split membership only; it does not undo the fact
that the test outcomes were observed.

## Access ledger

The exact per-class recalls, supports, overlap diagnostics, configurations, and
additional metrics remain in each run's JSON files and in
`diagnostic_existing_test_reports.csv`.

| Run configuration | Epoch budget | Selected validation balanced accuracy | Test balanced accuracy | Test macro F1 | Collapsed-4 test balanced accuracy |
|---|---:|---:|---:|---:|---:|
| LR `1e-3` | 10 | `0.1802512886597938` | `0.19713738675632764` | `0.16475521482743438` | `0.27014412053717657` |
| LR `1e-4` | 10 | `0.4599042459545037` | `0.3790024075735936` | `0.33844466057057304` | `0.4571573337183515` |
| LR `1e-4` | 30 | `0.4606145429599038` | `0.39236344721622785` | `0.36507023833659785` | `0.5126751854204035` |
| LR `1e-5` | 30 | `0.43269063053212536` | `0.38745910257063604` | `0.2978318141354199` | `0.33620689655172414` |
| LR `3e-4` | 30 | `0.4198911269143228` | `0.2614966051605197` | `0.2561185511255443` | `0.3605035761374559` |
| LR `3e-4` | 10 | `0.3508102936582318` | `0.19079128596717798` | `0.14487373737373738` | `0.30147000688977493` |
| LR `3e-5` | 30 | `0.5104804702742847` | `0.3971046789166193` | `0.36123836086522654` | `0.5388919491966533` |

The LR `3e-5` run selected checkpoint epoch `25`. The validation value may be
treated only as historical single-seed validation evidence; none of the test
columns above are eligible parameter-selection evidence.

## Preserved files

- `run_config.json`, `run_metadata.json`, and `test_report.json` for all seven
  runs;
- `diagnostic_existing_test_reports.csv`;
- `sweep_status.csv`, including the recorded `--run-test-eval` commands;
- `sweep_comparison.csv`;
- one canonical `dataset_summary_canonical.json`.

The seven original `dataset_summary.json` files were byte-identical, with
SHA256
`537bd2f46d5faed579f3d4794298b126bf2cf67d0cf7d6964215e9891f621918`.
The referenced checkpoint binaries are no longer present. `SHA256SUMS` covers
the tracked portable copies.
