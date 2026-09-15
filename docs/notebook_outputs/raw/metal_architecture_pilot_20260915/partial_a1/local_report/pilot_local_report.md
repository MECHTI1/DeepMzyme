# Metal architecture pilot: local evidence report

Completed full 50-epoch runs: **3**. Verified smokes: 7 (excluded from rankings).

Native balanced accuracy selects checkpoints and learning rates within each training target. Four-, five-, and six-class models are compared using common-four metrics from those same checkpoints.

## Coverage

| Block | Completed / expected | Status |
|---|---:|---|
| S | 7 / 7 | complete |
| A1 | 3 / 4 | incomplete |
| A2 | 0 / 4 | pending |
| T1 | 0 / 6 | pending |
| T2 | 0 / 6 | pending |
| H | 0 / 2 | conditional_pending |
| R | 0 / 9 | pending |
| RE | 0 / 1 | pending |
| RH | 0 / 1 | conditional_pending |

## Completed full runs

Rows are listed by execution block. Incomplete blocks remain diagnostic evidence.

| Block | Family | Target | LR | Seed | Native BA | Common-4 BA | Native minimum recall | Minutes | Block complete |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| A1 | GVP + early fusion | four_class | 3e-05 | 42 | 0.6937 | 0.6937 | 0.4688 | 4.8 | False |
| A1 | Only-ESM | four_class | 3e-05 | 42 | 0.7212 | 0.7212 | 0.5938 | 3.4 | False |
| A1 | Only-GVP | four_class | 3e-05 | 42 | 0.6471 | 0.6471 | 0.4062 | 4.5 | False |

## Comparison eligibility

- Target comparison at LR 3e-05: incomplete (2/9 core runs).
- Target comparison at LR 0.0001: incomplete (0/9 core runs).
- Comparing targets after separate native-LR selection: deferred until both LR blocks are complete.
- Matched seed-42/43 repeats: 0 configurations.

Two-seed ranges are descriptive. No confidence intervals, promotion, or held-out-test claims are made.

Full native/common-four class recalls and support, selected epochs, timing, and local/remote paths are retained in JSON and CSV.
