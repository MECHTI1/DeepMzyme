---
name: gpu-use-skill
description: Find, connect to, run on, monitor, and close GPU resources for DeepMzyme training, HPO, embedding generation, inference, and benchmarks. Use for project GPU execution, capacity failures, or recovery; reuse owned resources and apply the bounded GCP/Colab cascade within existing authorization.
---

# DeepMzyme GPU use

This is the project's required GPU entry point. Read it before GPU execution or
GPU operational planning, including when the agent cannot discover skills
automatically. It routes through existing tools; it does not create a second
controller. Creating or reading this skill does not authorize cloud spending.

## Establish the task and recover state

1. Read the latest user request, root `AGENTS.md`,
   [current status](../../../EXPERIMENT_STATUS.md), and the applicable campaign
   receipt/handoff. Use the task playbook for the exact scientific command.
   A narrower approved screen takes precedence over a deferred larger grid.
2. Identify completed units, frozen source/input identities, saved caches, the
   output destination, and any running worker. Verify receipts before reuse;
   do not repeat completed preparation, smokes, embeddings, or fits by default.
3. Identify the existing authorization: work, provider/hardware scope, allowed
   starts/candidates, hours or compute units, cost ceiling, and expiry/stop rule.
   Reuse applicable authorization without asking again. A named bounded cascade
   approved by the user covers its specified candidates within that pass; a
   same-VM retry request does not authorize migration or a new provider.
   Missing material authority needs one concrete question with the blocking
   instruction cited. Never invent a confirmation phrase or widen a budget.
4. Choose one coordinator to own allocation, job submission and shutdown.
   Prepare/checksum inputs and the smallest useful command before billing starts.
   Default to one owned allocation and one training process.
   Inspection alone does not transfer lifecycle ownership; report to the
   existing coordinator without submitting or stopping its work.

## Select a route from actual resource state

Read the provider instructions only for routes being inspected or used:

- GCP: [GCP runbook](../../../docs/GCP_GPU_RUNBOOK.md), then the installed
  `~/deepmzyme-vm/AGENTS.md` and current `config.env`. All lifecycle actions go
  through `~/deepmzyme-vm/bin/*`; if that controller is missing, report the
  missing prerequisite instead of recreating it with raw cloud commands.
- Colab: [Colab runbook](../../../docs/COLAB_GPU_RUNBOOK.md) and the available
  `colab-gpu` skill, including its CLI reference when executing commands.
  Resolve its installed path from the skill catalog; do not assume another
  workstation has the same installation. If unavailable, inspect CLI help and
  the project runbook; do not guess accelerator names or authentication flags.
- An explicitly requested local GPU: verify the project interpreter, device and
  available memory locally. Cloud provisioning is unnecessary when this route
  satisfies the authorized task.

For GCP run `vm-status` and `vm-report --no-ssh` before any start. For an
authorized Colab route inspect named sessions, ownership and entitlement before
allocating. A request to inspect or connect does not itself authorize a start.
For Colab, a local session name alone does not establish ownership: the provider
skill permits sessions created by the acting agent. Reusing another
coordinator's runtime requires an applicable explicit user-authorized handoff.

| Observed state | Action |
|---|---|
| Healthy, owned running resource | Connect to that exact endpoint; inspect process/log state before submitting. An active fit is monitored, not duplicated. |
| Existing stopped GCP VM | Start that VM only under applicable start authorization. Its disk and caches stay in its zone. |
| No managed GCP VM anywhere in the project | The new-allocation cascade below can apply when authorized. |
| Stockout while starting an existing VM | Preserve it and its disk. Follow an already approved retry/fallback scope; otherwise present a concrete continuation choice. |
| Unknown ownership, another job, in-flight start, disconnect or timeout | Reconcile the same resource first. Do not allocate a replacement to resolve uncertainty. |

**Existing-VM limitation:** the current controller's project-wide duplicate guard
rejects another managed GCP VM even if the old one is stopped. Editing `ZONE`
does not move the VM/disk and does not make that candidate eligible. Do not
rename, delete, unlabel, or bypass the guard to obtain a second VM. A separately
authorized migration needs a supported procedure preserving artifacts; this
skill supplies none. An authorized Colab fallback can be considered with the
GCP VM confirmed stopped, keeping its disk charges in the total. Explain this
constraint instead of repeatedly trying ineligible GCP candidates.

## Bounded capacity cascade

Apply this default only when provider, region and hardware changes are within
the user's scope; an explicitly specified working route takes precedence.

1. For an eligible **new** GCP allocation, try L4 `g2-standard-8` in
   `us-central1-a`, then `us-east4-a`, then `us-west1-a`: at most one creation
   attempt per candidate, three total, and ten minutes for initiating the pass.
   Check elapsed time between attempts and resolve any in-flight request before
   advancing. Do not reset the pass by spawning another agent or retry loop.
2. Before an authorized configuration change, save the exact controller config.
   Use the GCP runbook's region/subnet/SKU mapping, current quota/price checks,
   and controller preflight. Preserve session/daily caps and on-demand policy.
   Present the actual hard stop and maximum gross charge before allocation.
   `vm-zones` lists machine offerings, **not live capacity**; quota is not capacity.
3. Only confirmed resource exhaustion advances the capacity pass. Diagnose
   authentication, quota, pricing, duplicate-resource, lock and configuration
   refusals; do not bypass them by hopping providers. After an ambiguous error,
   check real cloud state before deciding whether the attempt failed.
4. On GCP success, stop searching and retain its config so status/stop target
   that resource. If every attempt fails or the search is abandoned, restore
   the saved config only after confirming no candidate is active/ambiguous.
5. If GCP capacity is exhausted, or its existing-VM restriction makes new GCP
   candidates ineligible, inspect an **authorized** Colab fallback. Reuse a
   named owned runtime when possible; otherwise request one explicitly named
   L4 runtime. A100 is eligible only under a reviewed, already authorized
   fallback with verified compatibility and metered cost. A 400 from `colab new`
   is an entitlement failure, not evidence for automatic hardware substitution;
   diagnose 400 responses from other commands according to their actual error.
   On 401/403, inspect identity per the Colab skill and report the blocker;
   do not repair authentication by starting browser/ADC flows yourself.
6. Stop after the first suitable allocation. If allowed routes are exhausted,
   save the attempts and exact blocker; ask only for the missing choice or
   authority. Never leave an unbounded background allocation loop running.

## Connect, admit, run, and close

- Use `vm-connect -- <command>` or the controller-generated IAP SSH alias for
  GCP. Use an explicit session name on every Colab command. Browser/CLI views
  of the same Colab runtime must not create a second allocation. Respect the
  Colab skill's ownership restrictions for untracked browser sessions.
- Verify actual hardware, interpreter, PyTorch/CUDA versions and a small
  relevant CUDA workload. An absent exact architecture string is not by itself
  incompatibility; use compatible-kernel evidence. Repeat environment checks
  after a runtime change, but reuse valid scientific input certificates.
  Run ESMC/long-sequence readiness only when generation or its changed runtime
  needs it; cached-embedding training does not require regenerating embeddings.
- Keep model, batch size, precision, targets, folds, seeds and test policy fixed.
  Measure full-unit cost, not just kernel time. CPU parsing, graph construction,
  checkpoint replay, admission hashing and transfer can dominate GPU use.
  Preserve raw caches across fits; normalization remains training-fold-only.
- Bind execution to the actual allocation start and hard-stop deadline. Use
  the campaign's measured admission rule; PMM requires a 1.25 forecast multiplier
  plus a 900-second persistence/closeout reserve. Count setup/idle/transfer and
  all provider charges against the applicable budget. Colab has separate CU
  accounting: do not treat it as free, apply GCP dollar rates to it, or assume
  an hour cap transfers automatically. Confirm an enforceable stop mechanism
  for the chosen route; a host watchdog is not a provider guarantee.
- Use detached execution with bounded log/status polls for long jobs. A CLI
  timeout is not proof the worker failed. For PMM, a partial fit restarts from
  its original seed; only verified completed fits or supported replay-only
  recovery can be reused. Do not claim optimizer-state resume.
- Persist each terminal unit and verify local/mounted hashes and the required
  acknowledgment before another launch. Honor existing pause/ownership state.
  If accepted work exceeds its ceiling, follow root `AGENTS.md`'s budget rule;
  do not shrink the science or abandon a safe active fit solely due to forecast.
- On success or failure, persist artifacts, stop the owned resource and confirm
  provider state unless explicit keep-running authorization applies. GCP uses
  `vm-stop` then `vm-status`; Colab uses named stop and session verification.
  Never delete the GCP disk for ordinary closeout. A stopped ledger or dead
  worker alone is insufficient proof of provider shutdown.
  For a local GPU, clean up only this task's processes; do not shut down the host.

## Optional reviewer or monitor

`gpu_audit` is a task role, not a persistent service or allocation tool: it
reviews environment, timing, compatibility or orchestration evidence.
`gpu_monitor` observes a specific live job and reports to the coordinator.
Neither role acquires lifecycle authority by its name.

Use no extra agent for a short single-fit task the coordinator can watch.
Consider **one read-only monitor** for a long/multi-fit run when the coordinator
has independent work, or a focused auditor when a concrete uncertainty warrants
it. Read [the monitor handoff](references/monitor.md) only when delegating.
An agent is not a watchdog or billing safeguard and cannot promise monitoring
after its execution stops. Provider controls remain mandatory.

## Handoff

Update the campaign's existing runtime receipt/handoff and current-status owner:
authorization scope, attempts/errors, resource identity, actual hardware,
allocation start/deadline, budget spent/remaining, exact active/completed unit,
source/input identity, verified artifact location, and provider stop evidence.
Keep live state and measured results out of this skill. Distinguish prepared,
running, verified complete and blocked; report the precise next action.
