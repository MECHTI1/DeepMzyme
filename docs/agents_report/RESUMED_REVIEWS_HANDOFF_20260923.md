# Handoff: resumed GVP / GVP+ESMC improvement reviews (2026-09-23)

**Status: work in progress.** This is a session-handoff note for the next agent, not
a scientific evidence document and not authority for any claim. Nothing in the repo
source, run outputs, Colab sessions or datasets was modified by this work. No
training run was launched. `Plan.md`, `EXPERIMENT_STATUS.md` and
`docs/notebook_outputs/` retain their existing authority and content.

## 1. What the user asked for (the "unfinished 2 works")

Two Claude Code sessions from 2026-09-23 were interrupted before delivering an answer.

### Work 1 — session `182d0760-c196-46f2-b952-6c6fa3f10d4d` (17:31 UTC)

The user pasted the completed exact-PinMyMetal 5-fold benchmark results
(`EXPERIMENT_STATUS.md` "Current objective" block) and asked:

> Please read the project's gvp and fusion model and try to figure out if there are
> clear/substantial reasons/causes that might have interfere/harm the performance of
> the gvp+esmc?

**This session never executed.** Every assistant turn failed with
`Credit balance is too low`. Zero tool calls, zero output. Nothing to recover from
it except the question itself.

### Work 2 — session `a01b9fd7-e079-4567-9ac2-56dd187bb871` (18:12–18:24 UTC)

Two requests:

1. (18:12) Read the Antigravity (Gemini ACP) chat stored in the PyCharm IDE, which
   proposed issues whose fixes "might have high potential improving the model".
   The user asked for an opinion on each, a ranking by estimated likelihood of
   substantial improvement, and any additional suggestions of Claude's own.
2. (18:22) *"Please, do independent new from scratch review for the gvp and gvp+esmc
   (with focus on the fusion) for find what are the issues that fix/change them in
   the code might bring substantial improvements to the performance."*

The session decoded the Antigravity chat, scouted the code and run artifacts, and
launched two background workflows. **Both were cut off mid-run when the session
ended.** The user received only two preliminary findings in chat and no final answer.

## 2. Source material

| Item | Path |
| --- | --- |
| Antigravity chat, decoded (13 claims C1–C13) | `<scratch-a01b9fd7>/antigravity_chat.txt` |
| Antigravity raw event log | PyCharm `aia-task-history/8778fdeb-4b17-48a5-8952-e01916092eb1.events` (base64 JSON lines) |
| Antigravity SQLite conversation | `~/.gemini/antigravity-acp/conversations/2e750abe-5179-4652-b4a3-af901b0c5b19.db` |
| Benchmark run outputs | `runs/benchmark_exact_pinmymetal_5fold/{benchmark_only_esm,benchmark_enhanced_only_gvp,benchmark_enhanced_gvp_esmc}_fold{0..4}/` |

`<scratch-a01b9fd7>` =
`/tmp/claude-1000/-home-mechti-PycharmProjects-DeepMzyme/a01b9fd7-e079-4567-9ac2-56dd187bb871/scratchpad`
(session-scoped tmp; a copy of `antigravity_chat.txt` was placed in this session's
scratchpad as well).

## 3. State of the two interrupted workflows

| Workflow | Run ID | Transcript dir (under `~/.claude/projects/-home-mechti-PycharmProjects-DeepMzyme/a01b9fd7-.../subagents/workflows/`) | Completed before the cut |
| --- | --- | --- | --- |
| `deepmzyme-improvement-audit` | `wf_2cb72b1d-af6` | `wf_2cb72b1d-af6/` | 13/13 `verify:C*`, 1/13 `challenge:C*` (C2). Discovery lenses started, none returned. |
| `gvp-fusion-fresh-review` | `wf_64387913-254` | `wf_64387913-254/` | 4/7 finders: `gvp-core-math`, `readout-pooling`, `fusion-design`, `optimizer-training-loop`. |

Original scripts are preserved at
`~/.claude/projects/-home-mechti-PycharmProjects-DeepMzyme/a01b9fd7-e079-4567-9ac2-56dd187bb871/workflows/scripts/`.

`resumeFromRunId` is **same-session only**, so those runs cannot be resumed directly
from a new session. The recovery method used instead is in section 4.

## 4. What this session (`756401be-6fa4-40e3-87b1-c5609796e660`) did

1. Read both session transcripts and both workflow journals; confirmed Work 1 produced
   nothing and Work 2 stopped mid-flight.
2. **Extracted every completed agent result** from each `journal.jsonl` by joining
   `started` records (which carry `label`) to `result` records on `key`, producing
   `cache_audit.json` (14 results) and `cache_fresh.json` (4 results).
3. **Built one combined continuation script** that reproduces both original scripts
   verbatim, except:
   - `agent()` is shadowed per branch by a `cachedAgent()` wrapper that returns the
     cached result when a label is present and otherwise spawns the agent for real,
     so only the missing agents run;
   - both branches run concurrently under one shared agent-concurrency cap;
   - a final **synthesis** stage merges both reviews into one ranked list and
     explicitly answers Work 1's question;
   - the per-agent resource rules were corrected (see caveat 5 below).
4. Launched it as run `wf_4c859baa-61b` and armed a memory monitor.
5. Independently spot-checked the most-cited premise (see section 6, C1).

Artifacts of this session:

| Item | Path (under `/tmp/claude-1000/-home-mechti-PycharmProjects-DeepMzyme/756401be-6fa4-40e3-87b1-c5609796e660/scratchpad/`) |
| --- | --- |
| Combined continuation script | `resume/resume_reviews.js` |
| Cached results replayed by it | `resume/cache_audit.json`, `resume/cache_fresh.json` |
| Script assembly parts | `resume/parts_*.js` |
| New agents' scratch space | `agents/`, `fresh_review/` |

Run `wf_4c859baa-61b` journal:
`~/.claude/projects/-home-mechti-PycharmProjects-DeepMzyme/756401be-6fa4-40e3-87b1-c5609796e660/subagents/workflows/wf_4c859baa-61b/journal.jsonl`.

### How to resume again if this run is also interrupted

```bash
# 1. Harvest completed results from the journal (label -> result)
python3 - <<'EOF'
import json
lab, res = {}, {}
p = '<transcript-dir>/journal.jsonl'
for line in open(p):
    d = json.loads(line)
    if d['type'] == 'started': lab[d['key']] = d['label']
    if d['type'] == 'result':  res[d['key']] = d['result']
json.dump({lab[k]: v for k, v in res.items()}, open('cache.json', 'w'))
EOF
# 2. Merge into resume/cache_audit.json / cache_fresh.json, stripping the
#    'audit:' / 'fresh:' label prefixes the wrapper adds, then relaunch
#    resume/resume_reviews.js. Cached labels are skipped; the rest run live.
```

Within the same session, `Workflow({scriptPath, resumeFromRunId: "wf_4c859baa-61b"})`
is the simpler path.

## 5. What still has to happen

1. **Finish run `wf_4c859baa-61b`** — remaining: 12 adversarial challenges (C1, C3–C13),
   5 discovery lenses + merge + per-proposal adversarial checks + critic and gap
   follow-ups (audit branch); 3 finders + loop-until-dry sweeps + merge + per-issue
   premise/impact checks + critic (fresh branch); then synthesis.
2. **Deliver to the user**, in one message:
   - a direct answer to Work 1 (are there clear/substantial code-level causes that
     harmed GVP+ESMC?);
   - the ranked list with reconciled probabilities, cost, and the cheapest decisive
     experiment per item;
   - the per-claim table for C1–C13 with Claude's opinion vs Antigravity's ranking,
     which is what request 2a asked for;
   - which items the independent review found that Antigravity missed, and where the
     two disagree.
3. **Decide nothing scientific from it without an experiment.** Every item is a
   hypothesis with a proposed validation-only paired test; none is evidence.

## 6. Substantive findings already established (cached, re-checkable)

These come from the completed agents and from direct checks. They are code/artifact
observations and validation-based reasoning — not experimental results.

**Headline: Antigravity's confidence is badly calibrated.** All 13 claims were graded
`partly_correct` or `mostly_correct`; none was fully correct. Every reconciled
probability of a ≥1.5 pp gain in mean out-of-fold validation balanced accuracy came
out at **0.04–0.15**, against Antigravity's stated 45–95%.

Verified facts worth keeping:

- **C1 param-group split is real** (`src/training/run.py:1486-1507`). The fast
  `gvp_learning_rate` (3e-4) is given only to parameters whose names start with
  `layers.`, `node_scalar_encoder.` or `edge_scalar_encoder.` (prefix tuple at
  `src/training/run.py:1492`). In the joint model the GVP input projection
  (`init_vec_proj`, `src/model.py:746`) and the GVP readout (`gvp_attn_pool`
  `src/model.py:745`, `gvp_fusion_proj` `src/model.py:763`) therefore train at 3e-5,
  10× slower, while in Only-GVP they train at 3e-4. This was found independently by
  three of the four fresh-review finders, making it the most corroborated single
  issue. Reconciled probability is still modest (≈0.08–0.20): the head and gate do
  train, and a fully trained GVP branch adds only ~1 pp on validation.
- **The "GVP overfitting" story does not hold** (C2). Validation loss bottoms at
  epoch ~21–30 and rises in *all three* models, including ESM-only, which has no GVP
  trunk. Validation balanced accuracy keeps rising to epochs 40–50. The rising loss is
  overconfidence, not accuracy collapse.
- **The "50/50 blend beats joint training" claim is a test-set artifact** (C3). It
  reproduces on test (81.17 vs 79.92 BA4) but the bootstrap 95% CI of the difference
  is [−1.9, +4.5] pp, and on out-of-fold **validation** the blend is *worse* than the
  joint model (BA4 77.70 vs 79.84; BA5 71.68 vs 74.51). It must not be adopted.
- **Reporting caveat:** the "Table 1" CV numbers in `EXPERIMENT_STATUS.md` and the
  2026-09-23 summary (80.29 / 74.42 / 80.22) appear to be the **max over epochs** of
  validation collapsed-4 BA, not the value at the selected checkpoint. At the selected
  checkpoints the out-of-fold means are only_esm BA4 78.98, only_gvp 73.87,
  joint 79.84. **This needs reconciling before publication.**
- **Split caveat:** `--train-val-split-by pocket_id` puts pockets of the same PDB on
  both sides. Fold 0 has 58 validation pockets whose PDB ID also appears in training,
  54 of them the same chain (`split_diagnostics.json`).
- **Minor latent defects noted in passing:** the PROPKA `dpka_titr` node feature is
  identically zero in every fold (dead input); the focal-loss path computes `p_t` from
  class-weighted CE then takes an unweighted mean (unused in this benchmark); the
  joint model's logged `lr` is the GVP-trunk LR, not the head/ESM LR
  (`src/training/run.py:1584`).

The full structured verdicts (factual findings, mechanism, computed numbers, proposed
experiment per claim) are in `resume/cache_audit.json` and `resume/cache_fresh.json`.

## 7. Constraints any continuing agent must respect

1. **Held-out test data must not drive design choices** (`Plan.md`, `AGENTS.md`).
   Test numbers may be reproduced only to fact-check a claim, labelled as such.
   All improvement judgements rest on out-of-fold validation, code reasoning and
   prior project evidence.
2. **Read-only on the repo** for the review agents; scratch output goes to the
   session scratchpad.
3. **No training runs, no dataset builds, no forward passes over data.** The local
   machine has no GPU. Every proposal is delivered as an experiment to run later.
4. Use `/home/mechti/miniconda3/envs/DeepMzyme/bin/python`, never bare `python`.
5. **Memory is the real constraint.** ~15.7 GB total, with PyCharm, Chrome and Claude
   Desktop resident; free memory was ~1.6 GB when this work started. Importing torch
   costs ~400 MB per process. The earlier run's agent prompts advertised "~1–2 GB free"
   and probably contributed to a desktop crash; the resumed script tells agents to
   keep **one** torch process alive at a time and exit promptly. Keep the concurrency
   cap and that rule.

## 8. Status label

Per `AGENTS.md` §1c: this work is **implemented and in progress** as an analysis; its
outputs will be **hypotheses with proposed experiments**, not smoke-tested, not
experimentally evaluated, and not promoted. No model, parameter or architecture
conclusion may cite this document as evidence.
