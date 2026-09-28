# RULES — Normative Statements and Modifications (Docs Consolidation 2026-09)

This register tracks every normative statement (rule, policy, protocol constraint) modified, kept, merged, or dropped during docs consolidation.

## Job A Modifications (2026-09-28)

| ID | Location | Old Text / Policy | New Text / Policy | Status | Rationale |
|---|---|---|---|:---:|---|
| RULE-A01 | `AGENTS.md:231` | "For G4-class GPU planning, this is where serious/custom Optuna budgets..." | Labeled historical: "For historical G4-class GPU planning, this records the historical serious/custom Optuna budgets..." | Modified | Reflects D1 decision (L4 VM primary; G4 historical). |
| RULE-A02 | `AGENTS.md:465-470` at base `3e912a6` | "Hardware: G4-class GPU... Persistent Optuna storage in Drive is mandatory for Stage 4 and Stage 5." | Primary route is GCP L4 VM under `gpu-use-skill`; Colab authorized fallback. Persistent storage remains mandatory for Stage 4/5 on both routes. Drive SQLite is Colab-specific; the VM-specific Optuna recipe remains undocumented. | Modified | D1 changes the provider/storage wording without removing durability requirements or inventing a VM recipe; see B0.4. |
| RULE-A03 | `AGENTS.md:552` | "Confirm persistent Drive SQLite storage for serious Optuna stages." | "Confirm persistent Drive SQLite storage for serious Optuna stages on the Colab route." | Modified | Route-specific clarification. |
| RULE-A04 | `Plan.md:252` | "Canonical Colab metal-training pipeline" / "The canonical metal-training workflow is notebooks/..." | "Canonical staged metal-training pipeline" / staged blocks are canonical; CLI runners on GCP L4 VM primary; notebook secondary. | Modified | Removes Colab-first bias while preserving staged gates. |
| RULE-A05 | `Plan.md:268` | "Prefer a verified G4-class GPU when available; measure the actual accelerator... on each Colab allocation." | Measure actual accelerator on each allocation (GCP L4 VM primary, Colab fallback). Historical budgets reference G4. | Modified | Generalized to GCP L4 VM + Colab fallback. |
| RULE-A06 | `docs/GETTING_STARTED.md:26` | Colab notebook: "Recommended cloud entry point" | GCP L4 VM added as primary cloud execution route; Colab notebook marked "Authorized fallback; secondary interface". | Modified | Accurate entry point guidance per D1. |
| RULE-A07 | `playbook:2689, 3834` | Retained serious-HPO budgets target verified G4-class GPU. | Prefixed with "Historical (G4-class):". | Modified | Preserves numerical budget info while labeling hardware target as historical. |
| RULE-A08 | `docs/DATASETS.md:219` | Exact PinMyMetal: "Completed test evaluation found: no" | "Completed test evaluation found: yes — opened 2026-09-22 and 2026-09-23". | Modified | Empirical fact correction (D2; 3 single-split + 15 fold + 3 ensemble reports found). |
| RULE-A09 | `docs/DATASETS.md:220` | Exact PinMyMetal: "Selection use established: no" | "Selection use established: yes, exploratory (test deltas used for model ranking/claims; no model promoted)." | Modified | Accurate history of test score influence on ranking and exploratory blending. |
| RULE-A10 | `docs/DATASETS.md:294` | Non-overlap PinMyMetal: "Historical model evaluations found: exactly seven." | Seven early reports, plus six undocumented evaluations on 2026-09-18. | Modified | Captures 2026-09-18 non-overlap runs (3 in benchmark_50epochs, 3 in benchmark_replicated_72pct). |
| RULE-A11 | `docs/EXACT...:25, 30` | "identical dataset, identical 5-fold stratification... beating Fig 2a by +5.14 pp... Massive Breakthrough" | Removed parity and hype claims; inserted methodological qualification note; added max-over-epochs caveat and selected-checkpoint OOF scores (79.84% late fusion, 78.98% ESM-C, 73.87% GVP). Kept all numbers. | Modified | Scientific integrity and anti-overclaiming enforcement. |
| RULE-A12 | `EXPERIMENT_STATUS.md:399` | "Full replication of the Nature Communications (2025) PinMyMetal 5-fold cross-validation protocol" | "Exploratory benchmark under --train-val-split-by pocket_id --n-folds 5... (Caveat: Not a like-for-like comparison...)" | Modified | De-escalates replication claim to exploratory benchmark with caveats. |
| RULE-A13 | `docs/PARAMETER_FINDINGS.md:816` & `docs/notebook_outputs/README.md:52` | Exact 5-fold benchmark assigned Grade 2 | Corrected to Grade 6 (exploratory). Does not qualify for Grade 2 due to `pocket_id` stratification and cross-fold PDB leakage. | Modified | Adheres strictly to project evidence grading hierarchy. |
| RULE-A14 | `docs/EXACT...:73` & `EXPERIMENT_STATUS.md:400` | Table 1 CV metrics presented as ordinary CV results | Reported collapsed-four means labeled as epoch maxima, alongside selected-checkpoint out-of-fold means (79.84% late fusion, 78.98% ESM-C, 73.87% GVP). The audit's publication-reconciliation caveat stays explicit; checkpoint selection is not a statistical bias correction. Historical deltas remain labeled arithmetic. | Modified | Preserve values and distinguish checkpoint-bound results from maxima without reclassifying every native metric or recall in the table. |

The final Job A pass also repairs evidence links, places the fold-regime table
after the metal-example definitions as requested by Appendix B, and records
post-edit verification. These are navigation and audit-record changes, not new
scientific rules. Base-file line numbers above refer to `3e912a6`; moved lines
are located by heading. No training budget, source code, fold membership,
checkpoint, test-access permission, or campaign resume authority is changed.
