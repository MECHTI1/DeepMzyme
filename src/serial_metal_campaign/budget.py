"""Cost-only coverage decisions, independent of validation scores."""
from __future__ import annotations

from pathlib import Path
import math

import run_metal_architecture_pilot as base
from serial_metal_campaign import profile

ORDER = ("core", "five", "fusion", "ring")
HISTORICAL_SECONDS = {"gvp": 270.345, "esm": 205.157, "late": 305.499,
                      "early": 290.365, "hybrid": 411.294, "ring": 274.601}


def nonnegative(value, name):
    number = float(value)
    base.require(math.isfinite(number) and number >= 0, f"Invalid {name}")
    return number


def coverage(block_seconds, available_seconds, *, factor=1.25):
    """Choose a whole, ordered prefix; never inspect scores or partial units."""
    remaining = nonnegative(available_seconds, "confirmation allowance")
    planned, deferred, costs = [], [], {}
    for block in ORDER:
        required = factor * nonnegative(block_seconds[block], block)
        costs[block] = required
        if not deferred and required <= remaining:
            planned.append(block)
            remaining -= required
        else:
            deferred.append(block)
    return dict(planned_blocks=planned, deferred_blocks=deferred,
                buffered_seconds=costs, unreserved_seconds=remaining,
                complete=not deferred, core_admitted="core" in planned)


def historical_preview():
    """An explicitly non-admissible proxy, never a hardware-readiness receipt."""
    t = HISTORICAL_SECONDS
    discovery = 1.25 * (16*t["gvp"] + 16*t["esm"] + 20*t["late"]
                        + 8*t["early"] + 8*t["hybrid"])
    blocks = dict(core=20*(t["gvp"] + t["esm"] + t["late"]), five=10*t["late"],
                  fusion=10*(t["early"] + t["hybrid"]), ring=20*t["ring"])
    selected = coverage(blocks, 36000)
    return dict(profile=profile.PROFILE, status="preview_only_not_admitted", certified_reuse=0,
                counts=dict(screen=48, top_two_repeats=16, larger_late=4, mandatory_discovery=68,
                            diagnostic_maximum=32, continuation_maximum=48, confirmation=110,
                            full_coverage_fits_before_optional=178, full_fit_inventory_maximum=258,
                            core_only_fits_before_optional=128, initial_smoke_maximum=13),
                discovery_buffered_seconds=discovery,
                confirmation_buffered_seconds={key: 1.25*value for key, value in blocks.items()},
                operations_reservation_seconds=14400,
                full_coverage_buffered_seconds=14400+discovery+1.25*sum(blocks.values()),
                discovery_fits_six_hour_cap=discovery <= 21600,
                fallback=selected,
                limitations=["No active-hardware timing or certified reuse credit.",
                             "Hybrid is a one-epoch extrapolation; larger late is optimistically priced as reference late.",
                             "Selected larger candidates also increase confirmation cost.",
                             "Four operations hours is a cap, not demonstrated sufficiency.",
                             "Discovery and confirmation must each pass measured admission before full training."])


def estimate_runs(runs, measurements):
    return {run["id"]: profile.forecast(run, measurements) for run in runs}


def forecast(output, manifest, queue, results, measurements, operations_forecast_seconds,
             *, candidates=None, extra_runs=(), status=None):
    """Reserve top-two repeats even before their identities are selected.

    Candidates affect cost only. Calling code freezes their scientific selection
    before confirmation and may not replace them using confirmation metrics.
    """
    from serial_metal_campaign import runtime
    status = status or runtime.budget_status(output)
    done = {row["id"] for row in results}
    scientific = [r for r in queue["runs"] if r["stage"] == "discovery"]
    pending = [r for r in scientific if r["id"] not in done]
    pending.extend(extra_runs)
    costs = estimate_runs(pending, measurements)
    raw_discovery = sum(costs.values())
    # Until top-two identities are frozen, reserve two fresh repeats per arm.
    # Historical repeats have no credit until their exact target cells exist.
    if not queue.get("top_two"):
        for arm in profile.ARMS:
            possibilities = [r for r in scientific if r["arm"] == arm and r["block"] == "screen"]
            raw_discovery += 2 * max(profile.forecast(r, measurements) for r in possibilities)
    if candidates is None:
        # Cost envelope includes every live discovery candidate, including a
        # potential larger winner. No validation score selects fallback coverage.
        candidates = {}
        for arm in profile.ARMS:
            rows = [r for r in [*scientific, *extra_runs] if r["arm"] == arm]
            candidates[arm] = max(rows, key=lambda r: profile.forecast(r, measurements))
    confirmation_runs = profile.confirmation_runs(manifest, candidates)
    block_costs = {block: 0.0 for block in ORDER}
    for run in confirmation_runs:
        if run["id"] not in done:
            block_costs[run["block"]] += profile.forecast(run, measurements)
    operations_raw = nonnegative(operations_forecast_seconds, "operations forecast")
    operations_buffer = 1.25 * operations_raw
    discovery_buffer = 1.25 * raw_discovery
    operations_ok = operations_buffer <= status["remaining_seconds"]["operations"]
    discovery_ok = discovery_buffer <= status["remaining_seconds"]["discovery"]
    # Before closure protect ten hours. Transfer only the unspent discovery
    # allocation that will still remain if all forecast work completes.
    transfer = max(0., status["remaining_seconds"]["discovery"] - discovery_buffer)
    confirmation_available = min(status["remaining_seconds"]["confirmation"] +
                                 (0 if status.get("discovery_closed") else transfer),
                                 status["total_remaining_seconds"] - operations_buffer - discovery_buffer)
    decision = coverage(block_costs, max(0., confirmation_available))
    return dict(status="measured_forecast", admission_factor=1.25,
                pending_discovery_runs=len(pending), discovery_raw_seconds=raw_discovery,
                discovery_buffered_seconds=discovery_buffer, discovery_admitted=discovery_ok,
                operations_raw_seconds=operations_raw, operations_buffered_seconds=operations_buffer,
                operations_admitted=operations_ok, confirmation=decision,
                confirmation_raw_seconds=block_costs, run_forecasts=costs,
                full_training_admitted=operations_ok and discovery_ok and decision["core_admitted"],
                budget=status, held_out_evaluation=False)


def write_preview(output, payload):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    base.save(output / "budget_forecast.json", payload)
    if payload["status"] == "preview_only_not_admitted":
        lines = ["# Single-GPU campaign budget preview", "", "Status: preview only; training is not admitted.", "",
                 "68 mandatory discovery fits + 110 confirmation fits = 178 before verified reuse.",
                 "Diagnostics add at most 32 and chains/controls at most 48; none are pre-funded.", "",
                 f"Discovery with margin: {payload['discovery_buffered_seconds']/3600:.3f} h (cap 6 h).",
                 "Confirmation blocks with margin:", ""]
        lines += [f"- {name}: {seconds/3600:.3f} h" for name, seconds in payload["confirmation_buffered_seconds"].items()]
        lines += ["", f"Full coverage plus 4 h operations: {payload['full_coverage_buffered_seconds']/3600:.3f} h (cap 20 h).",
                  "", "13 initial one-epoch smokes are charged to operations; later probes are additional operations.",
                  "Cost-only fallback order: core (60 fits), five-class late fusion (10), early/hybrid (20), RING (20).",
                  "The historical proxy fits the first 90 confirmation fits inside ten hours; RING is deferred.",
                  "Mandatory discovery still exceeds its six-hour cap, so this fallback is not launch admission.", ""]
        lines += ["- " + note for note in payload["limitations"]]
    else:
        limits = payload["budget"]["limit_seconds"]
        lines = ["# Measured campaign forecast", "", f"Full-training admission: {payload['full_training_admitted']}",
                 f"Discovery: {payload['discovery_buffered_seconds']/3600:.3f} h including margin "
                 f"(authorized cap {limits['discovery_seconds']/3600:.3f} h).",
                 f"Future operations: {payload['operations_buffered_seconds']/3600:.3f} h including margin "
                 f"(authorized cap {limits['operations_seconds']/3600:.3f} h).",
                 f"Confirmation blocks: {', '.join(payload['confirmation']['planned_blocks']) or 'none'}.",
                 f"Deferred: {', '.join(payload['confirmation']['deferred_blocks']) or 'none'}.",
                 f"Cumulative authorized cap: {limits['total_seconds']/3600:.3f} h; "
                 f"budget authorizations: {limits['authorization_count']}.",
                 "", "An admission forecast does not itself change the authorized cumulative allocation cap."]
    (output / "budget_forecast.md").write_text("\n".join(lines) + "\n")
    return payload
