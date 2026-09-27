# PMM campaign: exploratory fold-0 screen

Verified configurations: 2/9. Matched validation ions: 1492.

Exploratory single-fold, single-seed validation. Checkpoints selected on this fold. No promotion, confidence interval, paper-parity or superiority claim. Full grid requires all 45 fits.

| Configuration | Epoch | Common-four BA | Macro-F1 | Mn recall | Cu recall | Zn recall | VIII recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| only_esm__four_class__none | 36 | 88.394% | 84.295% | 85.366% | 97.436% | 89.503% | 81.273% |
| only_esm__four_class__first_shell_bias | 36 | 88.394% | 84.295% | 85.366% | 97.436% | 89.503% | 81.273% |
| PMM released recipe, same fold | — | 72.455% | 71.060% | 93.360% | 48.718% | 89.503% | 58.240% |

PMM uses its released features; DeepMzyme uses the declared ESM/geometry inputs. This is a matched known-site comparison, not paper protocol reproduction.

Six-class native metrics and Fe/Co/Ni recalls are retained in the CSV/JSON; common-four predictions sum Fe+Co+Ni probabilities before argmax.

Pending: only_esm__six_class__none, only_gvp__four_class__none, only_gvp__six_class__none, gvp_late_fusion__four_class__none, gvp_late_fusion__six_class__none, only_gvp__four_class__first_shell_bias, gvp_late_fusion__four_class__first_shell_bias.
