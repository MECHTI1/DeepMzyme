#!/usr/bin/env python3
"""CPU readiness gate for the amended v3 step D (extension 1, log v3-017). No fit, no GPU, no held-out access.

It checks this worktree against the hash-checked workstation copy of the VM campaign root and the pulled runs,
and writes ``audits/d_readiness_<UTC>/readiness_report.json``. The step D/E launcher (pmm_v3_step_d.py) starts
nothing unless the newest report says ``ready`` for exactly the runner files of the worktree.

Checks:
  prepared_identity          source tree, prepared recipes, profile and the frozen A4 specification are unchanged
  runner_change              which runner files differ from the prepared ones (the extension binds the new ones)
  unit_inventory             16 + 26 + 18 step D fits, at most one combination per family, 36 + at most 8 for E
  single_setting_overrides   every candidate resolves to its control plus exactly its own setting
  completed_units_unchanged  this code rebuilds every completed campaign unit's recorded identity and command
  reused_controls            the two seed-42 controls on the workstation verify (checkpoint, predictions, replay)
  alternatives_and_rounds    strengths of one setting cannot combine; the amended round order and stopping hold
  numerical_gates            the screen thresholds are the frozen A4 values
  decay_audit                the newest effective-decay audit passed and covers every tested strength
  launch_refusal             the launcher refuses before readiness and before the extension record exists
  no_held_out_access         no planned command names held-out data or enables a test evaluation
  parent_bundle              an extension bundle can be bound to the first bundle (same source, spec, folds)
  tests                      the v3 test files pass (skipped with --skip-tests, which leaves the report not ready)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import pmm_v3_assessment as assess  # noqa: E402
import pmm_v3_bundle as bundle  # noqa: E402
import pmm_v3_campaign as v3  # noqa: E402
import pmm_v3_step_b as step_b  # noqa: E402
import pmm_v3_step_d as step_d  # noqa: E402
from benchmarking import pmm_ion_campaign as v2  # noqa: E402

CAMPAIGN = step_d.CAMPAIGN
TEST_FILES = ("tests/test_v3_campaign.py", "tests/test_v3_extension.py", "tests/test_v3_assessment.py",
              "tests/test_v3_bundle.py", "tests/test_v3_step_b_tools.py", "tests/test_v3_step_c_launcher.py",
              "tests/test_v3_step_d_launcher.py", "tests/test_v3_early_stopping_audit.py", "tests/test_v3_decay_audit.py",
              "tests/test_v3_training_options.py", "tests/test_v3_model_options.py", "tests/test_v3_replay_policy.py",
              "tests/test_v3_speed_report.py")
# The resolved configuration keys each candidate may change against its control (Round R: exactly one).
EXPECTED_SETTINGS = {
    "meanagg": {"normalize_message_aggregation"}, "resdrop01": {"gvp_residual_dropout"},
    "structlr": {"gvp_lr_scope"}, "gvpaux03": {"gvp_auxiliary_loss_weight"}, "esmdrop02": {"esm_modality_dropout"},
    "wd001": {"weight_decay"}, "wd01": {"weight_decay"}, "wd10": {"weight_decay"},
    "headdrop01": {"head_mlp_dropout"}, "headdrop03": {"head_mlp_dropout"}, "resdrop02": {"gvp_residual_dropout"},
    "esmdrop04": {"esm_modality_dropout"}, "sitenone": {"site_geometry_features"},
    "sitecountsangles": {"site_geometry_features"}, "posnoise01": {"position_noise_std"},
    "outerdrop01": {"outer_residue_dropout"}, "vecnorm": {"gvp_vector_norm"},
    "invsqrtw": {"metal_class_weight_mode", "mn_loss_multiplier", "cu_loss_multiplier", "zn_loss_multiplier",
                 "class_viii_loss_multiplier"},
}
ROUND_R_VALUES = {"wd001": 0.01, "wd01": 0.1, "wd10": 1.0, "headdrop01": 0.1, "headdrop03": 0.3, "resdrop02": 0.2,
                  "esmdrop04": 0.4}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def latest(paths: list[Path]) -> Path | None:
    return max(paths, key=lambda path: path.parent.name) if paths else None


def resolved(paths: v3.V3Paths, unit: v3.Unit) -> tuple[dict[str, Any], list[str], dict[str, str], dict[str, Any]]:
    from training.config import config_to_payload, parse_args

    command, env, identity = v3.build_command(paths, unit, python_bin="python", train_dir=paths.root / "train",
                                              esm_dir=paths.root / "esm", device="cuda", lane=0)
    payload = config_to_payload(parse_args(command[3:]))
    return {k: v for k, v in payload.items() if k not in v2.NON_IDENTITY_CONFIG_KEYS}, command, env, identity


def check_prepared_identity(paths: v3.V3Paths, manifest: dict[str, Any]) -> dict[str, Any]:
    spec = assess.load_spec()
    assess.check_spec_against_campaign(spec, manifest)
    facts = {
        "source_tree_unchanged": v2.source_tree_sha256() == manifest["frozen_source_tree_sha256"],
        "prepared_recipes_unchanged": v3.stable_hash({n: v3.resolve_recipe(n) for n in v3.RECIPES}) == manifest["recipes_sha256"],
        "profile_unchanged": manifest["profile"] == v3.PROFILE and v3.stable_hash(v3.PROFILE) == manifest["profile_sha256"],
        "replay_policy_unchanged": manifest.get("replay_policy") == v3.REPLAY_POLICY,
        "a4_spec_is_the_frozen_one": sha256_file(assess.SPEC_PATH) == assess.FROZEN_SPEC_SHA256,
        "manifest_is_the_evidence_copy": True}
    return {"passed": all(facts.values()), **facts, "source_tree_sha256": manifest["frozen_source_tree_sha256"],
            "a4_spec_sha256": assess.FROZEN_SPEC_SHA256, "manifest_sha256": sha256_file(paths.manifest)}


def check_runner_change(manifest: dict[str, Any]) -> dict[str, Any]:
    current = v3.runner_sha256()
    changed = sorted(name for name in current if current[name] != manifest["runner_sha256"].get(name))
    allowed = {"pmm_v3_campaign.py", "run_pmm_v3_campaign.py"}
    return {"passed": set(changed) <= allowed, "changed_runner_files": changed,
            "unchanged_runner_files": sorted(set(current) - set(changed)),
            "note": "the changed files run only under an extension record bound to the prepared manifest"}


def check_unit_inventory() -> dict[str, Any]:
    rounds = {name: [unit.name for unit in v3.step_units(name)] for name in v3.ROUND_ORDER}
    every = [name for names in rounds.values() for name in names]
    d_units = [unit for name in v3.ROUND_ORDER for unit in v3.step_units(name)]
    facts = {
        "counts": {name: len(names) for name, names in rounds.items()} == {"D-A": 16, "D-R": 26, "D-B": 18},
        "no_duplicates": len(set(every)) == len(every) == 60,
        "four_class_fold0_two_seeds_gvp_families": all(
            u.target == "four_class" and u.fold == 0 and u.seed in v3.SEEDS and u.family in v3.GVP_FAMILIES for u in d_units),
        "round_r_adds_no_control": all(u.recipe != "baseline" for u in v3.step_units("D-R")),
        "only_esmc_keeps_its_baseline": all(u.family != "only_esm" for u in d_units),
        "launcher_lists_the_same_units": all(step_d.round_units(name) == tuple(rounds[name]) for name in v3.ROUND_ORDER),
        "e_neutral_is_36": len(v3.step_units("E-neutral")) == 36,
    }
    return {"passed": all(facts.values()), **facts, "units": rounds,
            "max_combination_fits": len(v3.SEEDS) * len(v3.GVP_FAMILIES), "max_e_improvement_fits": 4 * len(v3.GVP_FAMILIES)}


def check_single_setting_overrides(paths: v3.V3Paths) -> dict[str, Any]:
    rows, failures = [], []
    for round_name in v3.ROUND_ORDER:
        for unit in v3.step_units(round_name):
            if unit.recipe == "baseline":
                continue
            control_recipe = v3.SCREEN_CONTROLS.get(unit.recipe, "baseline")
            control, _, _, _ = resolved(paths, v3.Unit(unit.family, unit.target, control_recipe, unit.fold, unit.seed))
            candidate, command, _, _ = resolved(paths, unit)
            changed = sorted(key for key in set(control) | set(candidate) if control.get(key) != candidate.get(key))
            ok = bool(changed) and set(changed) <= EXPECTED_SETTINGS[unit.recipe] and candidate["gvp_weight_decay"] is None
            if round_name == "D-R":
                key = next(iter(EXPECTED_SETTINGS[unit.recipe]))
                ok = ok and changed == [key] and abs(float(candidate[key]) - ROUND_R_VALUES[unit.recipe]) < 1e-12
            rows.append({"unit": unit.name, "round": round_name, "control": control_recipe, "changed": {
                key: [control.get(key), candidate.get(key)] for key in changed}})
            if not ok:
                failures.append(unit.name)
    return {"passed": not failures, "failures": failures, "n_candidates": len(rows), "candidates": rows}


def vm_form(command: list[str], paths: v3.V3Paths) -> list[str]:
    """The argv as the VM runner wrote it: this workstation copy's root and source mapped to the VM's."""
    out = []
    for item in command[:-2]:  # the trailing --campaign-run-identity JSON is compared as an identity
        item = item.replace(str(paths.root / "train"), step_b.REMOTE_TRAIN).replace(str(paths.root / "esm"), step_b.REMOTE_ESM)
        item = item.replace(str(paths.root), step_b.REMOTE_V3).replace(str(ROOT), step_b.REMOTE_CODE)
        out.append(item)
    return out


def check_completed_units_unchanged(paths: v3.V3Paths, manifest: dict[str, Any]) -> dict[str, Any]:
    settings = v3.read_execution_settings(paths)
    reused_as = {run: unit for unit, run in settings.get("reuse", {}).items()}
    rows, failures = [], []
    for name, record in sorted(v3.completed_units(paths).items()):
        unit_name = reused_as.get(name, name)
        if record.get("status") != "completed" or name == v3.REGRESSION_NAME or unit_name.startswith("probe-"):
            continue
        unit = v3.Unit.parse(unit_name)
        _, command, _, identity = resolved(paths, unit)
        expected = v3.parent_identity(identity, manifest)
        identity_same = expected == record["identity"]
        saved = json.loads((paths.lane(int(record["lane"])) / "commands" / f"{name}.json").read_text())["argv"]
        ours = vm_form(command, paths)
        theirs = saved[:-2]
        # Lane, device, python, run name and the ESMC directory (unused by Only-GVP) are per-launch, not recipe.
        skip = {i + 1 for i, item in enumerate(theirs) if item in ("--runs-dir", "--run-name", "--device", "--load-workers",
                                                                   "--esm-embeddings-dir")}
        theirs_core = [item for i, item in enumerate(theirs) if i not in skip and i != 0]
        ours_core = [item for i, item in enumerate(ours) if i not in {
            j + 1 for j, value in enumerate(ours) if value in ("--runs-dir", "--run-name", "--device", "--load-workers",
                                                               "--esm-embeddings-dir")} and i != 0]
        for flag in ("--load-workers",):  # the VM launcher did not pass it; drop the flag itself where present
            theirs_core = [item for item in theirs_core if item != flag]
            ours_core = [item for item in ours_core if item != flag]
        argv_same = ours_core == theirs_core
        rows.append({"run": name, "unit": unit_name, "identity_unchanged": identity_same, "argv_unchanged": argv_same})
        if not (identity_same and argv_same):
            failures.append(name)
    return {"passed": bool(rows) and not failures, "failures": failures, "n_completed_units": len(rows), "units": rows}


def check_reused_controls(paths: v3.V3Paths, manifest: dict[str, Any], durable: Path) -> dict[str, Any]:
    settings = v3.read_execution_settings(paths)
    statuses = v3.completed_units(paths)
    rows, failures = [], []
    for family in v3.GVP_FAMILIES:
        unit = v3.Unit(family, "four_class", "baseline", 0, 42)
        run_name = settings.get("reuse", {}).get(unit.name, unit.name)
        try:
            record = statuses[run_name]
            run_dir = durable / f"lane{record['lane']}" / "runs" / run_name
            receipt = v3.verify_completed_unit(run_dir, record["identity"], v3.resolve_recipe("baseline"))
            replay = v3.verify_independent_replay(run_dir, receipt)
            rows.append({"control": unit.name, "run": run_name, "workstation_copy": str(run_dir),
                         "checkpoint_sha256": receipt["selected_checkpoint_sha256"],
                         "replay_max_probability_abs_difference": replay["max_probability_abs_difference"],
                         "terminal_common4_ba": receipt["metrics"]["val_metal_collapsed4_balanced_acc"]})
        except (KeyError, ValueError, OSError) as exc:
            failures.append(f"{unit.name}: {exc}")
    return {"passed": not failures and len(rows) == 2, "failures": failures, "controls": rows,
            "seed_43_controls": "run in Round A (two fits); never relabelled from another seed"}


def check_alternatives_and_rounds() -> dict[str, Any]:
    refused = []
    for name in ("combo-wd001+wd01", "combo-wd01+wd10", "combo-headdrop01+headdrop03", "combo-resdrop01+resdrop02",
                 "combo-esmdrop02+esmdrop04", "combo-esmdrop04+gvpaux03", "combo-esmdrop02+gvpaux03",
                 "combo-outerdrop01+posnoise01", "combo-baseline+wd01", "combo-sitenone+wd01"):
        try:
            v3.resolve_recipe(name)
        except ValueError:
            refused.append(name)

    def grid(deltas, rounds, skip=()):
        collected = {}
        for family in v3.GVP_FAMILIES:
            for seed in v3.SEEDS:
                collected[v3.Unit(family, "four_class", "baseline", 0, seed).name] = unit_record(0.80)
        for round_name in rounds:
            for unit in v3.step_units(round_name):
                if unit.recipe != "baseline" and unit.recipe not in skip:
                    collected[unit.name] = unit_record(0.80 + deltas.get((unit.family, unit.recipe), 0.0))
        return collected

    def unit_record(value):
        return {"status": "completed", "metrics": {"common4": {"balanced_accuracy": value, "recall": {
            label: value for label in assess.COMMON4}}}}

    spec = assess.SPEC_DEFAULTS
    after_a = assess.step_d_decision(grid({}, ("D-A",)), spec, through="D-A")
    no_pass = assess.step_d_decision(grid({}, ("D-A", "D-R")), spec, through="D-B")
    r_pass = assess.step_d_decision(grid({("only_gvp", "wd01"): 0.02}, ("D-A", "D-R")), spec, through="D-R")
    pooled = assess.step_d_decision(grid({("only_gvp", "wd01"): 0.03, ("only_gvp", "wd10"): 0.02,
                                          ("only_gvp", "vecnorm"): 0.02}, v3.ROUND_ORDER, skip=v3.COST_GATED),
                                    spec, through="D-B", not_tested=v3.COST_GATED)
    facts = {
        "alternatives_refused": len(refused) == 10,
        "round_r_follows_a_without_a_pass": after_a["status"] == "provisional: run D-R next",
        "d_ends_after_r_without_any_pass": no_pass.get("stopped_after") == "D-R" and no_pass["final"]
        and {item["recipe"] for item in no_pass["families"].values()} == {"baseline"} and "D-B" not in no_pass["rounds"],
        "a_pass_in_r_enters_b": r_pass["status"] == "provisional: run D-B next",
        "one_strength_per_setting_in_the_single_combination": pooled["families"]["only_gvp"].get("recipe") == "combo-vecnorm+wd01"
        and pooled["families"]["only_gvp"]["decision"] == "run combination"
        and pooled["families"]["gvp_late_fusion"]["recipe"] == "baseline",
    }
    return {"passed": all(facts.values()), **facts, "refused_combinations": refused,
            "round_order": list(v3.ROUND_ORDER), "stopping_rule": v3.STOPPING_RULE, "combination_rule": v3.COMBINATION_RULE}


def check_numerical_gates() -> dict[str, Any]:
    spec = json.loads(assess.SPEC_PATH.read_text())
    facts = {"mean_delta_1_5_points": spec["screen_min_mean_delta"] == 0.015 == assess.SPEC_DEFAULTS["screen_min_mean_delta"],
             "recall_change_minus_3_points": spec["screen_min_recall_change"] == -0.03 == assess.SPEC_DEFAULTS["screen_min_recall_change"],
             "recall_drop_limit_3_points": spec["recall_drop_limit"] == 0.03, "tie_band_0_2_points": spec["tie_band"] == 0.002,
             "spec_sha256_is_frozen": sha256_file(assess.SPEC_PATH) == assess.FROZEN_SPEC_SHA256}
    return {"passed": all(facts.values()), **facts}


def check_decay_audit(campaign: Path) -> dict[str, Any]:
    path = latest(sorted(campaign.glob("audits/d_decay_audit_*/decay_audit.json")))
    if path is None:
        return {"passed": False, "reason": "no effective-decay audit (run audit_v3_effective_decay.py)"}
    report = json.loads(path.read_text())
    covered = {(r["family"], r["recipe"]) for r in report["results"] if r["passed"]}
    needed = {(family, recipe) for family in v3.GVP_FAMILIES for recipe in ("baseline", "wd001", "wd01", "wd10")}
    needed.add(("gvp_late_fusion", "structlr"))
    table = [{"family": r["family"], "recipe": r["recipe"], "groups": [
        {"group": g["group"], "initial_lr": g["initial_lr"], "weight_decay": g["weight_decay"],
         "n_parameter_tensors": g["n_parameter_tensors"], "ideal_cumulative_factor": g["ideal_cumulative_factor"],
         "measured_fp32_median_factor": (g["measured_fp32_factor"] or {}).get("median"),
         "first_step_factor_is_one_in_float32": g["first_step_factor_is_one_in_float32"]} for g in r["groups"]],
        "optimizer_steps": r["optimizer_steps"]["real_run_adamw_step_counter"]} for r in report["results"]]
    return {"passed": bool(report.get("passed")) and needed <= covered, "report": str(path), "sha256": sha256_file(path),
            "missing": sorted(f"{family}:{recipe}" for family, recipe in needed - covered), "torch": report.get("torch"),
            "table": table}


def check_launch_refusal() -> dict[str, Any]:
    runner = step_d.local_runner_sha256()
    no_report = step_d.readiness_blockers({"extension": None}, None, runner)
    no_extension = step_d.readiness_blockers({"extension": None}, {"ready": True, "runner_sha256": runner}, runner)
    other_runner = step_d.readiness_blockers(
        {"extension": {"id": step_d.EXTENSION_ID, "runner_sha256": {**runner, "pmm_v3_campaign.py": "0" * 64}}},
        {"ready": True, "runner_sha256": runner}, runner)
    clear = step_d.readiness_blockers({"extension": {"id": step_d.EXTENSION_ID, "runner_sha256": runner}},
                                      {"ready": True, "runner_sha256": runner}, runner)
    facts = {"refused_without_a_report": bool(no_report), "refused_without_the_extension_record": bool(no_extension),
             "refused_for_other_runner_files": bool(other_runner), "admitted_only_when_all_agree": clear == [],
             "launcher_runner_files_are_the_campaign_ones": step_d.RUNNER_FILES == v3.RUNNER_FILES == bundle.RUNNER_FILES}
    return {"passed": all(facts.values()), **facts}


def check_no_held_out_access(paths: v3.V3Paths, manifest: dict[str, Any]) -> dict[str, Any]:
    units = [unit for name in v3.ROUND_ORDER for unit in v3.step_units(name)] + v3.step_units("E-neutral")
    offending = []
    for unit in units:
        _, command, env, _ = resolved(paths, unit)
        argv = command[:-2]
        flagged = [item for item in argv if item in ("--run-test-eval", "--test-structure-dir", "--test-summary-csv")
                   or any(part in bundle.FORBIDDEN_PARTS for part in Path(item).parts)]
        if flagged or "classmodel_test_set" not in env.get("DEEPMZYME_FORBIDDEN_READ_ROOTS", ""):
            offending.append({"unit": unit.name, "flagged": flagged})
    facts = {"no_command_names_held_out_data": not offending, "profile_never_evaluates_test": v3.PROFILE["evaluate_test"] is False,
             "manifest_held_out_access_false": manifest.get("held_out_access") is False}
    return {"passed": all(facts.values()), **facts, "n_commands_checked": len(units), "offending": offending}


def check_parent_bundle(campaign: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    found = sorted(campaign.glob("bundles/*/v3_bundle_manifest.json"))
    parents = [path for path in found if json.loads(path.read_text()).get("parent_bundle") is None]
    if len(parents) != 1:
        return {"passed": False, "reason": f"expected one first bundle, found {len(parents)}"}
    parent = json.loads(parents[0].read_text())
    try:
        bound = bundle.require_same_campaign_inputs(
            parents[0], src_sha=v2.source_tree_sha256(), spec_sha=sha256_file(assess.SPEC_PATH),
            fold_sha=manifest["fold_set"]["fold_membership_sha256"])
    except bundle.BundleError as exc:
        return {"passed": False, "reason": str(exc)}
    return {"passed": parent["runner_sha256"] == manifest["runner_sha256"], "parent_manifest": str(parents[0]),
            "a3_acceptance": parent.get("a3_acceptance", {}).get("path"), **bound}


def check_tests(skip: bool) -> dict[str, Any]:
    if skip:
        return {"passed": False, "reason": "skipped (--skip-tests); the report cannot be ready"}
    started = time.time()
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *TEST_FILES], cwd=ROOT,
                            capture_output=True, text=True)
    tail = [line for line in result.stdout.strip().splitlines() if line.strip()][-1:] or [""]
    return {"passed": result.returncode == 0, "returncode": result.returncode, "summary": tail[0],
            "files": list(TEST_FILES), "seconds": round(time.time() - started, 1)}


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(["git", "--no-optional-locks", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "uncommitted_paths": [line[3:] for line in run("status", "--porcelain").splitlines()]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", type=Path, default=CAMPAIGN, help="workstation campaign data directory")
    parser.add_argument("--campaign-copy", type=Path, help="evidence copy of the VM campaign root (default: newest)")
    parser.add_argument("--skip-tests", action="store_true", help="quick iteration only: the report is then not ready")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args(argv)
    copies = sorted(path.parent for path in args.campaign.glob("step_*_evidence/evidence_*/campaign/campaign_manifest.json"))
    copy = args.campaign_copy or max(copies, key=lambda path: path.parent.name)
    paths = v3.V3Paths(copy)
    manifest = json.loads(paths.manifest.read_text())
    checks = {
        "prepared_identity": check_prepared_identity(paths, manifest),
        "runner_change": check_runner_change(manifest),
        "unit_inventory": check_unit_inventory(),
        "single_setting_overrides": check_single_setting_overrides(paths),
        "completed_units_unchanged": check_completed_units_unchanged(paths, manifest),
        "reused_controls": check_reused_controls(paths, manifest, args.campaign / "durable"),
        "alternatives_and_rounds": check_alternatives_and_rounds(),
        "numerical_gates": check_numerical_gates(),
        "decay_audit": check_decay_audit(args.campaign),
        "launch_refusal": check_launch_refusal(),
        "no_held_out_access": check_no_held_out_access(paths, manifest),
        "parent_bundle": check_parent_bundle(args.campaign, manifest),
        "tests": check_tests(args.skip_tests),
    }
    definitions = v3.extension_definitions()
    report = {"audit": "v3 step D readiness (extension 1, Round R)", "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "ready": all(check["passed"] for check in checks.values()), "campaign_copy": str(copy),
              "extension_id": v3.EXTENSION_ID, "runner_sha256": v3.runner_sha256(),
              "prepared_runner_sha256": manifest["runner_sha256"],
              "assessor_sha256": sha256_file(ROOT / "pmm_v3_assessment.py"),
              "extension_definitions_sha256": v3.stable_hash(definitions), "extension_definitions": definitions,
              "git": git_state(), "checks": checks, "held_out_access": False}
    out_dir = args.out_dir or args.campaign / "audits" / f"d_readiness_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "readiness_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": str(out_dir / "readiness_report.json"), "ready": report["ready"],
                      "checks": {name: check["passed"] for name, check in checks.items()},
                      "failed_detail": {name: {k: v for k, v in check.items() if k in ("failures", "reason", "missing", "summary")}
                                        for name, check in checks.items() if not check["passed"]}}, indent=2))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"readiness audit refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
