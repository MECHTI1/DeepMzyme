# PMM campaign: exploratory fold-0 screen

Verified configurations: 8/9. Matched validation ions: 1492.

Exploratory single-fold, single-seed validation. Checkpoints selected on this fold. No promotion, confidence interval, paper-parity or superiority claim. Full grid requires all 45 fits.

In configuration names, `four_class` or `six_class` is the training target. Every score in the first table evaluates four classes: Mn, Cu, Zn, and Class VIII = Fe+Co+Ni. No five-class model is part of this screen.

| Configuration | Epoch | Common-four BA | Macro-F1 | Mn recall | Cu recall | Zn recall | VIII recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| only_esm__four_class__none | 36 | 88.394% | 84.295% | 85.366% | 97.436% | 89.503% | 81.273% |
| only_esm__six_class__none | 33 | 85.839% | 79.883% | 71.409% | 97.436% | 90.055% | 84.457% |
| only_gvp__four_class__none | 12 | 86.023% | 77.236% | 89.431% | 100.000% | 90.055% | 64.607% |
| gvp_late_fusion__four_class__none | 11 | 89.465% | 80.728% | 82.656% | 100.000% | 93.370% | 81.835% |
| gvp_late_fusion__six_class__none | 32 | 88.812% | 85.322% | 82.927% | 97.436% | 90.055% | 84.831% |
| only_esm__four_class__first_shell_bias | 36 | 88.394% | 84.295% | 85.366% | 97.436% | 89.503% | 81.273% |
| only_gvp__four_class__first_shell_bias | 27 | 84.944% | 81.996% | 78.997% | 94.872% | 89.503% | 76.404% |
| gvp_late_fusion__four_class__first_shell_bias | 45 | 88.728% | 81.046% | 83.740% | 97.436% | 91.713% | 82.022% |
| PMM released recipe, same fold | — | 72.455% | 71.060% | 93.360% | 48.718% | 89.503% | 58.240% |

PMM uses its released features; DeepMzyme uses the declared ESM/geometry inputs. This is a matched known-site comparison, not paper protocol reproduction.

Six-class native metrics and Fe/Co/Ni recalls are retained in the CSV/JSON; common-four predictions sum Fe+Co+Ni probabilities before argmax.

## Native six-class evaluation

These models were trained and evaluated on Mn, Cu, Zn, Fe, Co and Ni separately. Each checkpoint was selected by native six-class validation BA; its collapsed-four result above uses that same checkpoint.

| Configuration | Epoch | Six-class BA | Six-class macro-F1 | Fe recall | Co recall | Ni recall |
|---|---:|---:|---:|---:|---:|---:|
| only_esm__six_class__none | 33 | 72.251% | 65.812% | 79.843% | 34.286% | 58.974% |
| gvp_late_fusion__six_class__none | 32 | 72.013% | 72.137% | 85.864% | 22.857% | 51.282% |

Pending certification: only_gvp__six_class__none.

Checkpoint artifacts exist for only_gvp__six_class__none, but independent replay is not certified. These configurations are excluded from every score table and contrast above. See the batch README and completion execution receipt for failure diagnostics.
