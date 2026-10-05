"""Plan step B: the fixed probe manifest (standard library only; shared by runner, report and launcher).

Every step-B probe runs ``gvp_late_fusion__four_class__baseline__fold0__seed42``. A probe tag fixes its
precision, epoch count, lane and the batch it belongs to; the runner refuses any other combination
before a probe is admitted, so a mistyped tag or flag never spends GPU time.

  steps (in order)         tags                                  lanes   epochs
  serial FP32              w1-fp32-r1, then w1-fp32-r2           0       SHORT_EPOCHS
  FP32 x2 (together)       w2-fp32-a, w2-fp32-b                  0, 1    SHORT_EPOCHS
  FP32 x3 (together)       w3-fp32-a, w3-fp32-b, w3-fp32-c       0, 1, 2 SHORT_EPOCHS
  serial AMP               w1-amp-r1 [then w1-amp-r2]            0       SHORT_EPOCHS
  full FP32                full-fp32                             0       FULL_EPOCHS
  conditional              full-amp; w2-amp-*/w3-amp-* batches   as above

Whole-batch retry (timing batches only, user decision 2026-10-05): a batch of k >= 2 lanes may be
rerun once as a whole under the same tags with RETRY_SUFFIX, and only when its first attempt is
invalid under the outcome-blind rules of pmm_v3_speed_report.batch_attempt / resolve_attempts. Batch
members are never rerun individually (an archived member counts as failed). Serial probes and full
runs follow the ordinary rule (archive-failed, then one unchanged rerun of the same tag).
"""

from __future__ import annotations

from typing import Any

PROBE_UNIT_NAME = "gvp_late_fusion__four_class__baseline__fold0__seed42"
SHORT_EPOCHS = 10  # long enough for a steady training phase in the CPU-pressure gate (log v3-009)
FULL_EPOCHS = 50
RETRY_SUFFIX = "-retry"
BATCHES: dict[tuple[str, int], tuple[str, ...]] = {
    ("fp32", 1): ("w1-fp32-r1", "w1-fp32-r2"), ("fp32", 2): ("w2-fp32-a", "w2-fp32-b"),
    ("fp32", 3): ("w3-fp32-a", "w3-fp32-b", "w3-fp32-c"),
    ("amp", 1): ("w1-amp-r1", "w1-amp-r2"), ("amp", 2): ("w2-amp-a", "w2-amp-b"),
    ("amp", 3): ("w3-amp-a", "w3-amp-b", "w3-amp-c")}
FULL = {"fp32": "full-fp32", "amp": "full-amp"}


def retry_tags(precision: str, lanes: int) -> tuple[str, ...]:
    if lanes < 2:
        raise ValueError("Only concurrent timing batches (2 or 3 lanes) have a whole-batch retry")
    return tuple(tag + RETRY_SUFFIX for tag in BATCHES[(precision, lanes)])


def probe_spec(tag: str) -> dict[str, Any]:
    """The fixed settings of one probe tag; unknown tags raise ValueError."""
    for precision, full_tag in FULL.items():
        if tag == full_tag:
            return {"tag": tag, "precision": precision, "amp": precision == "amp", "epochs": FULL_EPOCHS,
                    "lane": 0, "lanes": 1, "batch": None, "attempt": 1, "kind": "full"}
    for (precision, lanes), tags in BATCHES.items():
        for attempt, names in ((1, tags), (2, tuple(t + RETRY_SUFFIX for t in tags) if lanes > 1 else ())):
            if tag in names:
                return {"tag": tag, "precision": precision, "amp": precision == "amp", "epochs": SHORT_EPOCHS,
                        "lane": 0 if lanes == 1 else names.index(tag), "lanes": lanes,
                        "batch": (precision, lanes), "attempt": attempt,
                        "kind": "serial" if lanes == 1 else "batch", "members": names}
    raise ValueError(f"{tag!r} is not a step-B probe tag (see pmm_v3_probes.py)")


def all_tags() -> tuple[str, ...]:
    tags = [t for names in BATCHES.values() for t in names] + list(FULL.values())
    tags += [t for (precision, lanes) in BATCHES if lanes > 1 for t in retry_tags(precision, lanes)]
    return tuple(dict.fromkeys(tags))
