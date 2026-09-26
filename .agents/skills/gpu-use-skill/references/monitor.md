# Optional GPU monitor handoff

This is a suggested delegation pattern, not a service to launch on every run.
One coordinator remains the sole lifecycle and training operator. Use an
available subagent only when independent monitoring materially helps; otherwise
the coordinator performs these checks. The monitor does not spawn other agents.

Give the monitor concrete values, not a request to find its own VM:

- Provider, project/zone/VM and immutable instance ID or explicit Colab session,
  and coordinator identity. Include the selected endpoint and fallback receipt
  when recovery has changed zones; a reused VM name is insufficient identity.
- Existing authorization and the active allocation receipt with start, hard
  stop, closeout reserve and budget; read-only command/connection route.
- Exact PID/job/run, log and runtime-profile paths, expected phase/output files,
  persistence acknowledgment path, and last verified progress.
- A measured phase-specific no-progress threshold. If none is measured, report
  silence as uncertainty and inspect activity; do not invent a failure verdict.

Respect provider ownership rules during delegation. For Colab, a monitor can
read coordinator-provided local log/status mirrors; operating a session created
by another agent needs the explicit user-authorized handoff described in the
project GPU skill. A monitor assignment alone does not supply that authority.

Suggested assignment:

> Read the project GPU skill and inspect only the named owned job. Use
> read-only process/log/device/filesystem checks and bounded waits. Do not
> allocate, start, stop, restart, kill, submit training, alter settings, install
> packages, or modify shared artifacts. Send observations and recommendations
> to the coordinator; the coordinator makes and performs operational decisions.
> During recovery, observe the named receipt and resource only; do not advance
> candidates, reset attempt limits, select endpoints or finalize cleanup.
> Continue until the supplied deadline or coordinator closeout, then return a
> final observation with its UTC timestamp. If access fails, report it promptly
> and stop claiming coverage; do not authenticate or replace the runtime.

Check progress about once a minute while active; combine a log tail, process
state and resource sample where useful instead of repeatedly loading all model
files or querying billing. Use waits of at most 60 seconds and no busy loops.
More expensive inventories and artifact hash verification remain boundary
checks owned by the coordinator. A known slow CPU preparation phase can justify
less frequent heavy checks without treating low GPU utilization as a failure.

Report only meaningful changes:

- Phase/epoch progress, elapsed time and estimated remaining time.
- Process exit, traceback/OOM/nonfinite output, growing RSS/VRAM, low disk space,
  progress beyond its measured expected duration, or stale persistence.
- Remaining time entering the closeout reserve or conflicting ownership/state.

Each alert includes UTC time, exact evidence, confidence/uncertainty, remaining
deadline and one recommended coordinator action. Do not interrupt a safe fit
merely because a single utilization sample is low. A terminal success message
is provisional until the coordinator verifies replay/outputs and persistence.

The coordinator reads alerts while it works, performs any required action,
and verifies provider shutdown itself. On monitor failure, it resumes direct
polling. Neither a live subagent nor a local watchdog survives every host/chat
failure; preserve the provider stop or explicitly documented fallback limits.
