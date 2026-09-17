"""Bounded host queue using a prepared endpoint and verified transfer adapter."""
from __future__ import annotations

from pathlib import Path
import time

from serial_metal_campaign import control, runtime


def run(output, adapter, *, max_completed=None, poll_seconds=2.0, max_poll_seconds=10.0,
        transport_retries=2, sleep=time.sleep):
    """Run without conversational transitions; never provision or replace a VM.

    The adapter owns provider I/O. Every launch follows terminal persistence,
    admission, durable state and a newly refreshed host receipt. Retryable
    transport operations must be idempotent; ambiguous launch results are
    reconciled by the same intent identity, never submitted under a new one.
    """
    if poll_seconds <= 0 or max_poll_seconds < poll_seconds or not 0 <= transport_retries <= 5:
        raise ValueError("Invalid bounded observation/recovery settings.")
    if max_completed is not None and max_completed < 1:
        raise ValueError("max_completed must be positive when supplied.")
    output = Path(output)
    events = output / "host_control" / "supervisor_events.jsonl"

    def log(event, **values):
        import json
        import os
        events.parent.mkdir(parents=True, exist_ok=True)
        with events.open("a") as stream:
            stream.write(json.dumps({"epoch": time.time(), "event": event, **values}, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def call(name, *args, **kwargs):
        for retry in range(transport_retries + 1):
            if name in {"verify_setup", "launch", "await_attempt"} or (
                    name == "persist_state" and kwargs.get("prepare_next")):
                control.require_running(output)
            started = time.monotonic()
            try:
                value = getattr(adapter, name)(*args, **kwargs)
                log(name, elapsed_seconds=time.monotonic() - started, retry=retry)
                return value
            except (TimeoutError, ConnectionError) as exc:
                log("transport_recovery", operation=name, retry=retry, error=type(exc).__name__)
                if retry == transport_retries:
                    raise
                # Adapter reconciliation checks this same endpoint and does not
                # allocate. An absent/unowned provider endpoint is a hard stop.
                adapter.reconnect()
                sleep(min(2 ** retry, 8))

    with control.controller_lease(output) as lease:
        control.require_running(output)
        completed, persisted = set(), set()
        delay = poll_seconds
        reason = "queue_complete"
        try:
            log("controller_started", token=lease["token"])
            call("verify_setup")
            while True:
                observation = call("observe")
                if observation["active"] or observation["running"]:
                    if control.status(output)["paused"]:
                        call("persist_state", prepare_next=False)
                        reason = "user_paused"
                        break
                    sleep(delay)
                    delay = min(max_poll_seconds, delay * 1.5)
                    continue
                delay = poll_seconds
                for attempt_id in observation["pending_persistence"]:
                    result = call("persist_attempt", attempt_id)
                    persisted.add(attempt_id)
                    if result["attempt_status"] == "completed":
                        completed.add(attempt_id)
                if control.status(output)["paused"]:
                    call("persist_state", prepare_next=False)
                    reason = "user_paused"
                    break
                if max_completed is not None and len(completed) >= max_completed:
                    call("persist_state", prepare_next=False)
                    reason = "requested_segment_complete"
                    break
                state = call("persist_state", prepare_next=True)
                if not state["next"]:
                    reason = "training_window_closed" if state.get("rollover_required") else "queue_complete"
                    break
                launch = call("launch", state["next"])
                log("launch_acknowledged", **launch)
                # A detached dispatcher may need a short time to acquire the
                # worker lock. Reobserve its exact identity before advancing.
                call("await_attempt", state["next"])
            return {"status": "stopped", "reason": reason, "completed_this_segment": len(completed),
                    "persisted_terminal_attempts": len(persisted)}
        except BaseException as exc:
            try:
                reason = "user_paused" if control.status(output)["paused"] else "controller_fault"
            except (ValueError, TypeError, KeyError):
                reason = "invalid_campaign_control"
            log("controller_failed", reason=reason, error_type=type(exc).__name__, error=str(exc))
            if reason == "user_paused":
                try:
                    observed = call("observe")
                    for attempt_id in observed["pending_persistence"]:
                        call("persist_attempt", attempt_id)
                    call("persist_state", prepare_next=False)
                except Exception as drain_error:
                    log("pause_drain_failed", error=str(drain_error))
            raise
        finally:
            # The independent watchdog remains armed if closeout fails or this
            # process is killed. Only provider absence closes the host interval.
            receipt = adapter.stop(reason)
            log("provider_stop_verified", receipt=receipt, reason=reason)
