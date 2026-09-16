"""Pure, bounded discovery continuation decisions; never launch training.

Only predeclared parent/candidate comparisons enter this module. Numeric trends
must survive both model seeds and class-recall protection before one more
matched setting is proposed. A proposal consumes the same six-fit allowance as
any missing isolation controls needed to justify it.
"""
from __future__ import annotations

import math
import statistics

from serial_metal_campaign import profile


AXES = ("learning_rate", "weight_decay", "head_mlp_dropout", "early_esm_dim", "late_capacity")
WIDTHS = ((128, 16, 64), (256, 32, 128), (384, 48, 192), (512, 64, 256))
WIDTH_FIELDS = ("hidden_s", "hidden_v", "edge_hidden")
LADDERS = {
    "weight_decay": (1e-6, 1e-5, 1e-4, 1e-3, 1e-2),
    "head_mlp_dropout": (0., .1, .2, .3, .4, .5),
    "early_esm_dim": (8, 16, 32, 64, 128),
}


def _fields(axis):
    return WIDTH_FIELDS if axis == "late_capacity" else (axis,)


def _effective(parameters, family_key=None):
    ignored = {"capacity"}
    if family_key == "esm":
        ignored.update(("hidden_v", "edge_hidden", "gvp_layers", "early_esm_dim", "early_esm_dropout"))
    elif family_key == "gvp":
        ignored.update(("esm_fusion_dim", "esm_graph_encoder_dropout", "early_esm_dim", "early_esm_dropout"))
    elif family_key == "late":
        ignored.update(("early_esm_dim", "early_esm_dropout"))
    return {key: value for key, value in parameters.items() if key not in ignored}


def _direction(parent, candidate, axis):
    if axis == "late_capacity":
        before = tuple(parent.get(field) for field in WIDTH_FIELDS)
        after = tuple(candidate.get(field) for field in WIDTH_FIELDS)
        if before not in WIDTHS or after not in WIDTHS:
            return None
        delta = WIDTHS.index(after) - WIDTHS.index(before)
    else:
        before, after = parent.get(axis), candidate.get(axis)
        if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
            return None
        if not math.isfinite(before) or not math.isfinite(after):
            return None
        delta = after - before
    return 1 if delta > 0 else -1 if delta < 0 else None


def compare_axes(parent_parameters, candidate_parameters, family_key=None):
    """Return (axis, direction) for one effective numeric change, else None."""
    before, after = (_effective(p, family_key) for p in (parent_parameters, candidate_parameters))
    changed = {key for key in before.keys() | after.keys() if before.get(key) != after.get(key)}
    for axis in AXES:
        if changed and changed <= set(_fields(axis)):
            if axis == "late_capacity" and family_key not in (None, "late"):
                return None
            direction = _direction(before, after, axis)
            return (axis, direction) if direction else None
    return None


def next_parameters(parameters, axis, direction, family_key):
    """One frozen numeric step; None means that the declared boundary is hit."""
    if axis not in AXES or direction not in (-1, 1):
        raise ValueError("Unknown continuation axis or direction")
    updated = dict(parameters)
    if axis == "learning_rate":
        current = float(parameters[axis])
        high = {"gvp": 3e-4, "esm": 2e-4, "late": 2e-4,
                "early": 2e-4, "hybrid": 1.5e-4}[family_key]
        if direction > 0 and math.isclose(current, profile.LRS[-1]):
            value = high
        elif direction < 0 and math.isclose(current, profile.LRS[0]):
            value = 5e-6
        else:
            value = current * (2 if direction > 0 else .5)
        if not 1e-6 <= value <= 1e-3:
            return None
        updated[axis] = value
    elif axis == "late_capacity":
        if family_key != "late":
            return None
        current = tuple(parameters.get(field) for field in WIDTH_FIELDS)
        if current not in WIDTHS:
            return None
        index = WIDTHS.index(current) + direction
        if not 0 <= index < len(WIDTHS):
            return None
        updated.update(zip(WIDTH_FIELDS, WIDTHS[index]))
        updated["capacity"] = "reference" if index == 0 else f"large_{WIDTHS[index][0]}"
    else:
        if axis == "early_esm_dim" and family_key not in ("early", "hybrid"):
            return None
        ladder = LADDERS[axis]
        current = parameters.get(axis)
        indices = [i for i, value in enumerate(ladder) if math.isclose(value, current)] if isinstance(current, (int, float)) else []
        if not indices or not 0 <= indices[0] + direction < len(ladder):
            return None
        updated[axis] = ladder[indices[0] + direction]
    return updated


def _pair(rows):
    indexed = {row["seed"]: row for row in rows}
    if len(rows) != 2 or set(indexed) != set(profile.SEEDS):
        raise ValueError("A continuation comparison requires exactly seeds 42 and 43")
    cohorts = {row.get("cohort_sha256") for row in rows}
    if len(cohorts) != 1 or None in cohorts or "" in cohorts:
        raise ValueError("Paired discovery results require a shared certified cohort")
    return indexed


def paired_gain(control_rows, candidate_rows):
    """Two native improvements, mean > .002, common-four and native protection."""
    control, candidate = _pair(control_rows), _pair(candidate_rows)
    if control[42]["cohort_sha256"] != candidate[42]["cohort_sha256"]:
        raise ValueError("Continuation parent and child cohorts differ")
    deltas, class_differences, native_zero_regression = [], {}, False
    for seed in profile.SEEDS:
        before, after = control[seed], candidate[seed]
        if before.get("scheme") != after.get("scheme"):
            raise ValueError("Native continuation scores cannot compare different targets")
        delta = float(after["balanced_accuracy"]) - float(before["balanced_accuracy"])
        if not math.isfinite(delta):
            raise ValueError("Non-finite continuation metric")
        deltas.append(delta)
        for key in ("per_class_recall", "collapsed4_per_class_recall"):
            if set(before[key]) != set(after[key]) or not before[key]:
                raise ValueError("Continuation recall labels differ or are absent")
            for label, value in before[key].items():
                other = after[key][label]
                if not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in (value, other)):
                    raise ValueError("Invalid continuation recall")
                if key == "per_class_recall":
                    native_zero_regression |= value > 0 and other == 0
                else:
                    class_differences.setdefault(label, []).append(other - value)
    means = {label: statistics.mean(values) for label, values in class_differences.items()}
    if len(means) != 4:
        raise ValueError("Continuation requires all four common-endpoint recalls")
    mean_delta = statistics.mean(deltas)
    passed = (all(delta > 0 for delta in deltas) and mean_delta > .002 + 1e-12
              and all(delta >= -.03 - 1e-12 for delta in means.values()) and not native_zero_regression)
    return dict(passed=passed, native_deltas=deltas, mean_native_delta=mean_delta,
                common_four_recall_differences=means, native_zero_regression=native_zero_regression)


def _rows(results, arm, parameters, family_key):
    selected = [row for row in results if row.get("arm") == arm and row.get("seed") in profile.SEEDS
                and row.get("stage", "discovery") == "discovery"
                and _effective(row["parameters"], family_key) == _effective(parameters, family_key)]
    by_seed = {}
    for row in selected:
        if row["seed"] in by_seed:
            previous = by_seed[row["seed"]]
            fields = ("balanced_accuracy", "per_class_recall", "collapsed4_per_class_recall", "cohort_sha256")
            if any(previous.get(key) != row.get(key) for key in fields):
                raise ValueError("Ambiguous repeated evidence for one continuation setting/seed")
        else:
            by_seed[row["seed"]] = row
    return list(by_seed.values())


def propose_next(family_key, comparisons, results, *, fits_used_per_arm=None, active_axis=None,
                 active_direction=None):
    """Return a missing-control block or one next paired setting, never execute.

    comparisons contains frozen parent_runs/candidate_runs mappings. Returned
    controls are {arm, parameters, seeds}; parameters maps each family arm to
    its next setting. Completing controls counts against the same six-fit cap.
    """
    arms = [arm for arm in profile.ARMS if arm.startswith(family_key + "_")]
    if not arms:
        raise ValueError("Unknown continuation family")
    spent = {arm: int((fits_used_per_arm or {}).get(arm, 0)) for arm in arms}
    if any(value < 0 or value > 6 for value in spent.values()):
        raise ValueError("Invalid continuation fit accounting")
    possibilities = []
    for comparison in comparisons:
        parents, children = comparison["parent_runs"], comparison["candidate_runs"]
        if not set(arms) <= parents.keys() or not set(arms) <= children.keys():
            raise ValueError("A core-family continuation must preserve both targets")
        for axis in AXES:
            if active_axis is not None and axis != active_axis:
                continue
            directions = {_direction(parents[arm]["parameters"], children[arm]["parameters"], axis)
                          for arm in arms}
            for direction in sorted(directions - {None}):
                if active_direction is not None and direction != active_direction:
                    continue
                controls, gates, next_values, requirements, next_missing = [], {}, {}, {}, {}
                possible = True
                for arm in arms:
                    parent, child = parents[arm]["parameters"], children[arm]["parameters"]
                    child_rows = _rows(results, arm, child, family_key)
                    if len(child_rows) != 2:
                        possible = False
                        break
                    # Hold every other changed field at the candidate value.
                    # This creates an isolation control for coupled proposals.
                    isolated = dict(child)
                    isolated.update({field: parent[field] for field in _fields(axis) if field in parent})
                    if axis == "late_capacity":
                        isolated["capacity"] = parent["capacity"]
                    observed_change = compare_axes(isolated, child, family_key)
                    qualifies_direction = observed_change == (axis, direction)
                    parent_rows = _rows(results, arm, isolated, family_key) if qualifies_direction else []
                    missing = sorted(set(profile.SEEDS) - {row["seed"] for row in parent_rows}) if qualifies_direction else []
                    following = next_parameters(child, axis, direction, family_key)
                    if following is None:
                        possible = False
                        break
                    observed_next = _rows(results, arm, following, family_key)
                    next_missing[arm] = sorted(set(profile.SEEDS) - {row["seed"] for row in observed_next})
                    required = len(missing) + len(next_missing[arm])
                    if spent[arm] + required > profile.POLICY["adaptive_fits_per_arm"]:
                        possible = False
                        break
                    requirements[arm] = required
                    next_values[arm] = following
                    if missing:
                        controls.append(dict(arm=arm, parameters=isolated, seeds=missing))
                    elif qualifies_direction:
                        gates[arm] = paired_gain(parent_rows, child_rows)
                    else:
                        gates[arm] = dict(passed=False, reason="Observed change does not qualify the chosen family direction")
                if not possible:
                    continue
                if not any(next_missing.values()):
                    # Every target already received the next setting; a later
                    # frozen comparison can assess continuation from it.
                    continue
                if not controls and not any(gate["passed"] for gate in gates.values()):
                    continue
                possibilities.append(dict(status="controls_required" if controls else "continue",
                    axis=axis, direction=direction, controls=controls,
                    parameters={} if controls else next_values,
                    gates=gates, required_fits_per_arm=requirements, next_missing_seeds=next_missing,
                    parent_runs=children,
                    comparison=comparison))
    if not possibilities:
        return dict(status="stop", reason="No eligible protected paired trend within the frozen bounds and six-fit cap")
    # The order is fixed before seeing gains. Magnitude never chooses an axis.
    possibilities.sort(key=lambda item: (AXES.index(item["axis"]), item["direction"],
                       tuple(item["parent_runs"][arm].get("id", "") for arm in arms)))
    return possibilities[0]
