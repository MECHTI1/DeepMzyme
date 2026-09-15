"""Small shared operational helpers; importing this module performs no I/O."""
import hashlib
import json
import math
from pathlib import Path
import time

PROFILE = "metal_ring_pilot_v1"
BUDGET_PROFILE = "metal_ring_continuation_budget_v1"
TOTAL_CAP = 36000.0
TRAINING_CAP = 34200.0
MAIN_CAP = 27000.0
PRIOR_SECONDS = 19263.44616508484
PRIOR_MAIN_SECONDS = 12561.789792060852
PRIOR_RETRY_SECONDS = 308.3893711566925


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else default


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric(value, name):
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
            "Invalid numeric " + name)
    return float(value)


def validate_handoff(handoff):
    require(handoff.get("profile") == BUDGET_PROFILE and handoff.get("total_cap_seconds") == TOTAL_CAP
            and handoff.get("training_cap_seconds") == TRAINING_CAP and handoff.get("main_cap_seconds") == MAIN_CAP,
            "Continuation cannot reset the original budget caps")
    intervals = handoff.get("prior_intervals", [])
    require(len(intervals) == 2, "Expected exactly two verified closed prior allocation intervals")
    total, previous_end = 0.0, None
    for interval in intervals:
        start = numeric(interval.get("started_epoch"), "prior interval start")
        end = numeric(interval.get("ended_epoch"), "prior interval end")
        require(0 < start <= end and (previous_end is None or start >= previous_end), "Invalid prior allocation interval")
        total += end - start
        previous_end = end
    require(math.isclose(total, PRIOR_SECONDS, abs_tol=1e-6, rel_tol=0)
            and math.isclose(total, numeric(handoff.get("prior_allocated_seconds"), "prior total"), abs_tol=1e-6, rel_tol=0),
            "Prior closed allocation time differs from the authorized handoff")
    require(0 <= numeric(handoff.get("prior_main_seconds"), "prior main time") < MAIN_CAP,
            "Invalid prior main budget usage")
    require(0 <= numeric(handoff.get("prior_retry_seconds"), "prior retry time") < 3600,
            "Invalid prior retry budget usage")
    require(math.isclose(handoff["prior_main_seconds"], PRIOR_MAIN_SECONDS, abs_tol=1e-6, rel_tol=0)
            and math.isclose(handoff["prior_retry_seconds"], PRIOR_RETRY_SECONDS, abs_tol=1e-6, rel_tol=0),
            "Recorded main/retry costs cannot be reset for this continuation")
    return handoff


def allocation_deadline(config, *, training=False, stop_margin=0):
    prior = numeric(config["prior_allocated_seconds"], "prior allocation")
    require(math.isclose(prior, PRIOR_SECONDS, abs_tol=1e-6, rel_tol=0), "Prior allocation total was reset")
    return numeric(config["allocation_started_epoch"], "allocation start") + (TRAINING_CAP if training else TOTAL_CAP) - prior - stop_margin


def alive(record):
    pid = record.get("pid") if isinstance(record, dict) else None
    if not isinstance(pid, int) or pid <= 0:
        return False
    path = Path("/proc") / str(pid) / "stat"
    return path.is_file() and path.read_text().rsplit(")", 1)[1].split()[0] != "Z"


def require_idle(output):
    for name in ("host_step_process.json", "active_process.json", "bootstrap_process.json"):
        require(not alive(read(Path(output) / name, {})), "An owned worker is already active: " + name)
