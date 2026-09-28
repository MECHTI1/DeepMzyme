# PMM fold-0 retrospective screen agreement v2.1

V2.1 agreement: 9/9. Legacy replay: 8/9.

Retrospective agreement of the nine frozen fold-0, seed-42, 50-epoch validation exports. This does not pass or replace the legacy replay contract, authorize the legacy runner, establish determinism, promote a model, or permit held-out access.

Four/six in configuration names describes training; the first metric columns evaluate four classes.

| Configuration | Epoch | Common-four BA | Common-four macro-F1 | Native BA | Fe/Co/Ni recall | V2.1 agreement |
|---|---:|---:|---:|---:|---|---|
| only_esm__four_class__none | 36 | 88.394% | 84.295% | 88.394% | — | True |
| only_esm__six_class__none | 33 | 85.839% | 79.883% | 72.251% | 79.843%/34.286%/58.974% | True |
| only_gvp__four_class__none | 12 | 86.023% | 77.236% | 86.023% | — | True |
| only_gvp__six_class__none | 31 | 82.840% | 75.498% | 63.627% | 75.654%/34.286%/18.803% | True |
| gvp_late_fusion__four_class__none | 11 | 89.465% | 80.728% | 89.465% | — | True |
| gvp_late_fusion__six_class__none | 32 | 88.812% | 85.322% | 72.013% | 85.864%/22.857%/51.282% | True |
| only_esm__four_class__first_shell_bias | 36 | 88.394% | 84.295% | 88.394% | — | True |
| only_gvp__four_class__first_shell_bias | 27 | 84.944% | 81.996% | 84.944% | — | True |
| gvp_late_fusion__four_class__first_shell_bias | 45 | 88.728% | 81.046% | 88.728% | — | True |

No legacy receipt changed. TECH-023's legacy gate remains open; do not resume that arm through the frozen runner.

Single-fold, single-seed evidence; no confidence intervals or promotion. See JSON for every attempt, worst differences and provenance.
