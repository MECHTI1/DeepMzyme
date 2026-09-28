# AGENTS.md — DeepMzyme

## Read path and authority

Read this file, [current status](EXPERIMENT_STATUS.md), then the [PMM campaign README](docs/campaigns/pmm_ion_metal/README.md). Restart your agent thread after AGENTS changes are integrated.
Consult [the owner/navigation table](docs/README.md) on demand; read the relevant Plan section and implementation before a major change or a concrete claim.
[Plan.md](Plan.md) owns design/scientific policy. Source and run outputs establish implemented behavior and measured results; preserve clearly newer working logic and report conflicts explicitly. Do not make large architectural changes that contradict Plan unless explicitly requested.
STATUS is mutable, lower authority than Plan. Read it before suggesting an experiment, baseline or sweep; use the specific evidence it names, then summaries before raw files.
For fresh validation-only planning, previous raw runs are context/guardrails: choose the largest sensible search space within stage, runtime, model-family, feature, split and test constraints. If asked to rely on previous results, inspect that evidence to narrow, continue or repeat it.
For notebook behavior inspect the notebook itself. [Config](src/training/config.py) and `src/train.py --help` own CLI flags; do not invent flags from examples.

## Scientific boundaries

DeepMzyme has two independent primary missions: metal classification and EC/function classification from protein pocket graphs and ESMC representations. Shared learning is a controlled challenger; keep both tasks independently trainable, selectable, reportable and publishable, and either standalone model may remain better. Keep experiments reproducible and publishable.
Use [Plan's target policy](Plan.md#2-train-the-metal-classification-model) and [auxiliary phase order](Plan.md#metal-ec-relationship-and-auxiliary-learning-policy):
- Primary metal endpoint: Mn, Cu, Zn, Class VIII = Fe+Co+Ni. Direct `four_class` has canonical name `merge_fe_class_viii`; never invent another scheme or relabel historical five/six-class evidence.
- Direct-four and six-class training followed by deterministic common-four evaluation are different formulations with separate run/study identities; the program's controlled comparison needs both. Compare Only-GVP, Only-ESM (ESMC), and graph-level late fusion on shared validation units; retain native-six metrics and Fe/Co/Ni recalls.
- Before planning a new metal campaign, ask the user which [target combination](Plan.md#per-campaign-target-selection) to train: six only, four + six, or four + five + six; record it in the campaign README/scope. Every trained model is evaluated on collapsed-four; five/six models also keep native metrics.
- [PMM core v2](docs/plans/pmm_core_scope_v2.json) keeps its recorded four + five + six; preserve the separate `five_class` identity and native metrics while comparing on common-four. Direct-four remains primary. Scope is not execution permission.
- EC starts at depth 1. Distinguish deeper hierarchical targets, multiple source annotations, and full multi-label prediction. The current selected-depth target path is single-label.
- Establish standalone Only-GVP, Only-ESM and graph-level late fusion for both tasks before cross-task claims. The first shared experiment asks whether auxiliary metal helps EC: independent heads, neither output feeding the other, with possible negative transfer. No hybrid, node-level late fusion or cross-attention in that first experiment; reverse-direction auxiliary learning is secondary/optional.
- Predicted metal is not a mandatory EC input. Soft conditioning and known-metal diagnostics are later optional ablations. Distinguish observed, curated, transferred AlphaFill/MAHOMES and predicted metal; transferred assignments are not perfect ground truth.
- Read actual `metal_example_unit` before counts, predictions or multinuclear claims. Follow [metal example terminology](Plan.md#metal-example-terminology): ion examples, parent pockets and PDB groups are separate; `PocketRecord`/`pocket_id` do not prove the unit, and pocket splitting uses the parent when present. Preserve historical units and protein/structure-level EC group weighting.
- A protein held out for either task is excluded from every shared-encoder training loss, including the other task's labels. Certify cross-source identity/homology and cross-task exclusion before the auxiliary recipe or training.
- Metal × EC1 association is development-only and descriptive: counts, conditional probabilities, Cramer's V, appropriate chi-square, mutual information and normalized mutual information do not establish learning benefit.
- Preserve the separate [controlled metal matrix](Plan.md#required-controlled-metal-model-comparison-matrix): target formulation on common-four; early/late/hybrid fusion, modality and matched RING/radius-only comparisons keep direct-four fixed. Shared folds/seeds, paired CIs and rare-class recall protection precede superiority claims; unmatched maxima are insufficient.
- Use planned, implemented, smoke-tested, experimentally evaluated and promoted literally. A code path is not validation evidence; fixed-split seeds are not grouped-fold confirmation. “No indexed result” does not prove no run occurred.

## Selection and stage requests

Selection is validation-only: never use test metrics for HPO, ranking, promotion, rejection or checkpoint choice. No held-out evaluation before Stage 7.
Stage 6 selects one configuration through grouped-fold confirmation or an explicitly labeled fallback; Stage 6B must complete/reuse its final full non-test refit, frozen as the one-shot Stage 7 source. Resolve the primary final-test route scientifically first; a historically opened test set is never pristine.
Freeze primary report, source checkpoint/refit, ensemble list/averaging and calibration rule before opening the test. Test scores cannot choose a different report or configuration.
For metal stage requests, follow the [required answer format and checks](docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md#required-answer-format-for-metal-notebook-stage-requests): state current/fresh stage assumption, copy the exact canonical block, give safety checks, outputs/config artifacts and decision gate. Use exact stage names; repair a missing block in the playbook before recommending it. Never invent budgets.
The [metal playbook](docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md) owns exact recipes/budgets/gates; the [EC playbook](docs/EC_TRAINING_PIPELINE_PLAYBOOK.md) owns EC recipes and its compatibility warning must be read for affected HPO/final-test blocks. Notebook cells own live defaults; the [guide](docs/METAL_NOTEBOOK_CONFIGURATION_GUIDE.md) owns option meanings.
Persistent Optuna storage is mandatory for Stage 4/5 on both routes; Colab uses Drive SQLite. VM persistence follows the GPU skill/runbook; its specific Optuna recipe remains undocumented. Keep one `MODEL_PRESET` per study and incompatible-study reuse blocked unless explicitly authorized as recovery/debug.

## Compute and Python

Before any GPU planning, capacity search, connection, monitoring, recovery, training, HPO, embeddings, inference or GPU benchmark, read [gpu-use-skill](.agents/skills/gpu-use-skill/SKILL.md). Pure CPU/code/docs work needs no GPU.
Primary route: CLI runners on the GCP L4 VM, operated only through `~/deepmzyme-vm/bin/*` under the skill. Colab is the authorized fallback and its notebook the secondary interface; historical G4 budgets stay labeled historical.
One coordinator owns allocation/submission/shutdown; an optional monitor is read-only and never replaces stop controls. At most one active GCP GPU. A skill invocation grants no spending, migration, provider or larger-budget authority; reuse existing authorization and verified artifacts only within scope.
Follow provider runbooks/controller safeguards. No retired provisioning/replacement shortcuts. Bounded `vm-fallback` may temporarily retain a positively stopped source VM; preserve its data until fit/replay/independent-backup verification permits approved cleanup. Manual zone changes do not migrate disks or bypass state checks.
Read the [Colab runbook](docs/COLAB_GPU_RUNBOOK.md) before that route; its Python 3.12 ESMC preparation for Python 3.13 runtimes is a tested convenience, not a requirement.
Use `/home/mechti/miniconda3/envs/DeepMzyme/bin/python` unless explicitly directed otherwise; before scripts verify with that executable's `-c "import sys; print(sys.executable)"`. Never assume shell `python`/`python3` selects this environment.
Do not launch expensive training unless explicitly requested. A pause remains effective until an explicit resume within verified authorization.

## Accepted-work budget escalation

Operational GPU-hour/trial/runtime budgets are auditable ceilings, not permission to abandon accepted work unfinished.
Before stopping solely because the plan no longer fits its ceiling, ask for exactly two choices: **stop and close out at the current evidence boundary**, or **continue beyond the planned ceiling and record a larger ceiling**. Give measured use, forecast, requested increase and consequences.
Ask at the first safe boundary before work requiring the increase. Do not terminate an active safe fit solely because a forecast changed; provider/session shutdown reserves and ownership safeguards still apply.
Reuse an already authorized overrun within scope. A higher ceiling does not change grid, folds, seeds, targets, test policy or other accepted scientific details.
Record approved increases in campaign evidence and executable controls. Ask again if insufficient; neither silently stop nor silently exceed the recorded authority.

## Development and verification

Inspect relevant files, compare with Plan, make the smallest safe change and preserve useful options. `src/model.py` may be experimental: prefer additive/configurable changes, backward compatibility and conservative resolution against Plan.
Use explicit configuration, clear names, small helpers, readable errors, comments only for non-obvious logic and minimal duplication. Avoid unnecessary dependencies, hidden constants, silent failures, unrelated changes, broad rewrites and hardcoded absolute paths outside existing convention.
Use repository-root-based paths with `pathlib.Path`; never assume data paths are relative to `src/`. Make model/fusion/loss options configurable where reasonable; use actual config/help flags.
Save model configuration, feature set, seed, all splits, loss, weights/sampling, LR/scheduler and checkpoint selection. Serious/final runs also record bundle ID/checksum and key library versions (PyTorch, torch-geometric, ESM/ESMC, Optuna, NumPy, scikit-learn when available). Use clear experiment names and structured output directories.
After code changes run the smallest relevant syntax/smoke checks first, then required checks; `tests/smoke_checks.py` is the CPU smoke entry. If relevant, smoke before long training. Temporary test files belong outside `DeepMzyme_Data/` unless explicitly needed; clean them immediately.
Summarize what changed and what did not, verification and limitations. When editing AGENTS, summarize the policy delta for review.

## Documentation contract

Follow the [coordination protocol](docs/README.md#documentation-coordination-protocol); identify all coupled owners before editing and repair cross-references together. Report unresolved conflicts.
Each fact has one owner in [docs/README.md](docs/README.md); other pages link instead of duplicating it. Exact executable metal blocks stay in the metal playbook, EC equivalents in the EC playbook; do not copy them into Plan, AGENTS or the option guide.
Overwrite STATUS rather than appending; preserve dated history in the campaign's `log.md` first. Keep mutable run IDs, trials, disk state and best results out of stable policy docs.
Campaign-specific text belongs in `docs/campaigns/<id>/`; protected `docs/plans/` stays in place, linked from the campaign README. Existing campaign sections elsewhere migrate only through the reviewed Job C steps; this rule does not authorize bulk movement now.
To close a campaign, set `Status: closed`, move its folder to `docs/archive/campaigns/<id>` with `git mv`, add one validation-only line to `docs/PARAMETER_FINDINGS.md` and update STATUS. Closure requires evidence or an explicit user decision.
A new top-level doc requires merging/archiving an existing one in the same change. Resolved issues move to `docs/archive/issues_resolved.md`; preserve linked `TECH-###` headings with one-line stubs and never reuse issue numbers.
The planned checker command is `/home/mechti/miniconda3/envs/DeepMzyme/bin/python tools/check_docs_contract.py`. After Job B enforcement lands, run it after every Markdown edit; fix text rather than raising caps/allowlists without approval. Until then, the [review record](docs/archive/consolidation_2026-09/PROGRESS.md) states the pending checkpoint.
Use `rg --no-ignore` for history/evidence and safety verification; Job B's planned `.ignore` hides `docs/archive/`, raw evidence and `uv.lock`, while `docs/MOVED.md` stays visible.
Raw evidence, existing summaries, hash-bound/provenance trees and frozen executable playbook blocks stay unchanged during consolidation. [PLAN_v2](docs/archive/consolidation_2026-09/PLAN_v2.md#0-safety-invariants-user-2026-09-28-they-override-everything-below) defines the protected paths and review stops; no consolidation permission implies compute authority.

## Git workflow

Inspect worktree and index before edits/staging. Preserve pre-existing changes; stage only task hunks if mixed; if safe separation is impossible, leave those task changes unstaged and explain why. Never reset the user's index.
On completing repository changes, stage explicit paths with `git add -- <paths>` including intended new files/deletions. No blanket `git add .`/`git add -A`, forced ignored data, generated junk, secrets or cleanup merely to make status empty.
Verify staged diff, report staged changes and any pre-existing staged content, and give suggested commit/push commands in the final response; at an explicit intermediate review stop, report the staged draft without them. Leave commit/push to the user unless explicitly requested; stage authorization is not commit/push authorization.
Read-only tasks need no staging or commit instructions; honor requests not to stage and report the remaining unstaged task changes. Consolidation agents follow PLAN_v2 review stops: ask before any commit, merge or push, and treat the shared campaign checkout as read-only.

## Review-only and prompt safety

Treat reviews/audits as read-only unless edits are requested: no file changes, training/HPO/test evaluation, installs, commits or evidence reorganization.
Inspect Plan, STATUS, evidence index and directly relevant code before claims; report conflicts instead of silently resolving them.
Verify each proposed improvement against implementation; label it implemented, planned, stale, risky or missing, with evidence and scope (docs, validation or implementation). WandB, MLflow and Neptune are optional adapters unless requested, not mandatory dependencies.
Do not treat an improvement list as established fact. Check existing config before recommending AMP, accumulation, schedulers, features or tracking. For schema concerns prefer targeted dataset/preflight checks and explicit model modes before broad PyG/architecture rewrites.
Rewrite unsafe prompts around evidence inspection and staged validation; avoid broad implementation instructions without authorization. CI complements Stage 1 smoke, Stage 6 confirmation and Stage 7 policy. Recommendations must preserve grouped folds, paired bootstrap, rare-class/promotion gates, final refit and one-shot testing unless a policy change is explicitly requested.
