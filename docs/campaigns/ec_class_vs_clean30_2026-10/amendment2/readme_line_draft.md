# Draft: campaign README lines for amendment 2

**Status: draft for merge (2026-11-04).** The [campaign README](../README.md) is being edited concurrently for
execution v5, which owns its status lines. At merge, add the link line below after the S1-amendment line, apply the
two replacements, keep the README under its 60-line target, and remove this file.

**Link line** (after "S1 amendment"; written relative to the README):

```markdown
- [Amendment 2](amendment2/amendment_draft.md): pre-G2 decisions of 2026-11-04 (best-inner S1, accuracy endpoint, F6, futility stop), rules, deviations and scope; a draft until published.
```

**Replacement for the endpoint sentence** ("Official folds stay unchanged. The design defines … five-seed DeepMzyme
refits."):

```markdown
Official folds stay unchanged. The [design](design.md) defines both co-primary populations, the endpoint (accuracy over the proteins of EC classes 1–6 under amendment 2; balanced accuracy over EC classes 1–6 before it, now the first secondary metric), paired homology-cluster inference, published CLEAN comparator and five-seed DeepMzyme refits.
```

**Replacement for phase 4** ("4 fold-0 development with a stop-or-continue gap gate"):

```markdown
4 fold-0 development with a binding futility stop (amendment 2; the Freeze-1 gap gate is descriptive)
```
