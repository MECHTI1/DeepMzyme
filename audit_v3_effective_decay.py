#!/usr/bin/env python3
"""Step D readiness: effective weight decay per optimizer group (CPU only; no data, no fit, no held-out access).

The amended step D (log v3-017) screens AdamW decay strengths. AdamW shrinks a weight by ``lr x wd`` per step,
so one coefficient acts differently in each learning-rate group and may do nothing in float32 (TECH-029). For
each family and recipe this audit records, with the trainer's own code:

- the parameter names of each optimizer group (the real model, rebuilt from the family's completed baseline
  checkpoint exactly as the independent replay rebuilds it; group membership from
  ``training.run.gvp_learning_rate_prefixes``), checked against the groups the real run saved;
- the resolved learning rates and decay coefficients of the campaign command (``pmm_v3_campaign.build_command``);
- the optimizer-step count (the real run's AdamW step counter, checked against training ions / batch size);
- the ideal cumulative factor ``product(1 - lr_step * wd)`` over the actual cosine schedule
  (``training.run.build_scheduler``);
- a zero-gradient FP32 run of ``torch.optim.AdamW`` over every step of that schedule on the checkpoint's
  weights: with zero gradients the Adam update is zero, so the measured shrink is the decay operation alone.

This measures the decay operation. It is not a forecast of trained weight norms or of accuracy. A recipe whose
model differs from the baseline's (new modules or input widths) needs ``--checkpoint FAMILY=RUN_DIR`` of a
completed run of that architecture; a combination that changes learning-rate group membership is audited by
naming it with ``--recipe``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import pmm_v3_campaign as v3  # noqa: E402

CAMPAIGN = Path("/media/mechti/Data1/DeepMzyme_Data/campaigns/pmm_ion_metal_v3")
DEFAULT_RECIPES = ("baseline", "wd001", "wd01", "wd10", "structlr")
# Recipes whose model has other parameters or input widths than the family baseline.
ARCHITECTURE_CHANGING = {"gvpaux03", "vecnorm", "sitenone", "sitecountsangles"}
FACTOR_TOLERANCE = 1e-3  # |measured - ideal| cumulative factor, for strengths the amendment tests


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def latest_campaign_copy(campaign: Path) -> Path:
    """The newest hash-checked evidence copy of the VM campaign root (manifest, class weights, statuses)."""
    copies = sorted(path.parent for path in campaign.glob("step_*_evidence/evidence_*/campaign/campaign_manifest.json"))
    require(bool(copies), f"No evidence copy of the campaign root under {campaign}")
    return max(copies, key=lambda path: path.parent.name)


def baseline_run_dir(copy: Path, durable: Path, family: str) -> Path:
    """The family's completed fold-0 seed-42 four-class baseline (late fusion: the reused step-B run)."""
    settings = json.loads((copy / "execution_settings.json").read_text())
    unit = v3.Unit(family, "four_class", "baseline", 0, 42).name
    run_name = settings.get("reuse", {}).get(unit, unit)
    found = sorted(durable.glob(f"lane*/runs/{run_name}"))
    require(len(found) == 1, f"Expected one durable copy of {run_name}, found {len(found)}")
    return found[0]


def resolved_config(paths: v3.V3Paths, unit: v3.Unit):
    from training.config import parse_args

    command, _, _ = v3.build_command(paths, unit, python_bin="python", train_dir=paths.root / "train",
                                     esm_dir=paths.root / "esm", device="cpu", lane=0)
    return parse_args(command[3:])


def trainer_groups(model, config) -> list[dict[str, Any]]:
    """The optimizer groups of training.run.prepare_run, in its order (GVP-rate group first when it exists).
    Mirrored here because the trainer builds them inline; verify_against_saved_groups checks the mirror."""
    from training.run import gvp_learning_rate_prefixes

    named = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    gvp_lr = config.gvp_learning_rate
    if gvp_lr is not None and gvp_lr != config.learning_rate and hasattr(model, "layers"):
        prefixes = gvp_learning_rate_prefixes(config.gvp_lr_scope)
        trunk = [(n, p) for n, p in named if any(n.startswith(prefix) for prefix in prefixes)]
        head = [(n, p) for n, p in named if not any(n.startswith(prefix) for prefix in prefixes)]
        decay = config.gvp_weight_decay if config.gvp_weight_decay is not None else config.weight_decay
        return [{"group": f"gvp_rate ({config.gvp_lr_scope})", "lr": float(gvp_lr), "weight_decay": float(decay),
                 "named": trunk},
                {"group": "base_rate", "lr": float(config.learning_rate), "weight_decay": float(config.weight_decay),
                 "named": head}]
    return [{"group": "all", "lr": float(config.learning_rate), "weight_decay": float(config.weight_decay),
             "named": named}]


def verify_against_saved_groups(groups: list[dict[str, Any]], checkpoint: dict[str, Any]) -> dict[str, Any]:
    """The mirrored groups must be the ones the real run's optimizer saved (sizes, initial rates, decay)."""
    saved = checkpoint["optimizer_state_dict"]["param_groups"]
    ours = [{"n": len(g["named"]), "initial_lr": g["lr"], "weight_decay": g["weight_decay"]} for g in groups]
    theirs = [{"n": len(g["params"]), "initial_lr": float(g["initial_lr"]), "weight_decay": float(g["weight_decay"])}
              for g in saved]
    return {"passed": ours == theirs, "mirrored": ours, "saved_by_the_real_run": theirs}


def zero_gradient_decay(config, groups: list[dict[str, Any]], *, steps_per_epoch: int,
                        updated: set[int]) -> list[dict[str, Any]]:
    """Run the real AdamW and scheduler with zero gradients over the whole schedule (FP32)."""
    import torch
    from training.run import build_scheduler

    originals = {name: param.detach().clone() for group in groups for name, param in group["named"]}
    optimizer = torch.optim.AdamW([{"params": [p for _, p in g["named"]], "lr": g["lr"],
                                    "weight_decay": g["weight_decay"]} for g in groups])
    scheduler = build_scheduler(optimizer, config)
    index = 0
    for group in groups:  # the real optimizer skips a parameter that never receives a gradient
        for _, param in group["named"]:
            param.grad = torch.zeros_like(param) if index in updated else None
            index += 1
    ideal = [1.0] * len(groups)
    rates: list[list[float]] = [[] for _ in groups]
    for _ in range(int(config.epochs)):
        for k, state in enumerate(optimizer.param_groups):
            rates[k].append(float(state["lr"]))
            ideal[k] *= (1.0 - float(state["lr"]) * float(state["weight_decay"])) ** steps_per_epoch
        for _ in range(steps_per_epoch):
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
    out, index = [], 0
    for k, group in enumerate(groups):
        ratios, skipped = [], []
        for name, param in group["named"]:
            before = float(originals[name].double().norm())
            if index not in updated:
                skipped.append(name)
            elif before > 0.0:
                ratios.append(float(param.detach().double().norm()) / before)
            index += 1
        ratios.sort()
        per_step = 1.0 - group["lr"] * group["weight_decay"]
        out.append({
            "group": group["group"], "initial_lr": group["lr"], "weight_decay": group["weight_decay"],
            "n_parameter_tensors": len(group["named"]), "n_values": sum(p.numel() for _, p in group["named"]),
            "parameter_names": [name for name, _ in group["named"]],
            "not_updated_no_gradient_in_the_real_run": skipped,
            "first_epoch_lr": rates[k][0], "last_epoch_lr": rates[k][-1],
            "first_step_factor_float64": per_step,
            "first_step_factor_is_one_in_float32": float(torch.tensor(per_step, dtype=torch.float32)) == 1.0,
            "ideal_cumulative_factor": ideal[k], "ideal_fraction_removed": 1.0 - ideal[k],
            "measured_fp32_factor": (None if not ratios else
                                     {"min": ratios[0], "median": ratios[len(ratios) // 2], "max": ratios[-1]}),
            "measured_minus_ideal": None if not ratios else ratios[len(ratios) // 2] - ideal[k]})
    return out


def audit_recipe(paths: v3.V3Paths, family: str, recipe: str, run_dir: Path) -> dict[str, Any]:
    import torch
    from training.campaign_runtime import load_campaign_prediction_components

    unit = v3.Unit(family, "four_class", recipe, 0, 42)
    config = resolved_config(paths, unit)
    checkpoint = torch.load(run_dir / "terminal_model_checkpoint.pt", map_location="cpu", weights_only=False)
    model, _, _ = load_campaign_prediction_components(checkpoint, device="cpu")
    groups = trainer_groups(model, config)
    saved_state = checkpoint["optimizer_state_dict"]["state"]
    steps = sorted({int(state["step"]) for state in saved_state.values()})
    require(len(steps) == 1, f"{run_dir.name}: parameters disagree on the optimizer step count: {steps}")
    saved_epochs = int(checkpoint["config"]["epochs"])
    steps_per_epoch = steps[0] // saved_epochs
    summary = json.loads((paths.fold_class_weights).read_text())["folds"][str(unit.fold)]
    n_train = sum(summary["train_native_counts"].values())
    batches = math.ceil(n_train / int(config.batch_size))
    baseline_config = resolved_config(paths, v3.Unit(family, "four_class", "baseline", 0, 42))
    membership = verify_against_saved_groups(trainer_groups(model, baseline_config), checkpoint)
    result = {
        "family": family, "recipe": recipe, "unit": unit.name, "checkpoint_run": run_dir.name,
        "checkpoint_sha256": hashlib.sha256((run_dir / "terminal_model_checkpoint.pt").read_bytes()).hexdigest(),
        "resolved": {"learning_rate": config.learning_rate, "gvp_learning_rate": config.gvp_learning_rate,
                     "gvp_lr_scope": config.gvp_lr_scope, "weight_decay": config.weight_decay,
                     "gvp_weight_decay": config.gvp_weight_decay, "lr_schedule": config.lr_schedule,
                     "epochs": config.epochs, "batch_size": config.batch_size},
        "optimizer_steps": {"real_run_adamw_step_counter": steps[0], "epochs": saved_epochs,
                            "steps_per_epoch": steps_per_epoch, "training_ions": n_train,
                            "ceil_training_ions_over_batch": batches},
        "baseline_groups_match_the_real_run": membership,
        "groups": zero_gradient_decay(config, groups, steps_per_epoch=steps_per_epoch,
                                      updated={int(index) for index in saved_state}),
    }
    tested = recipe != "baseline" and any(part in v3.EXTENSION_RECIPES and part.startswith("wd")
                                          for part in v3.recipe_components(recipe))
    deviations = [abs(g["measured_minus_ideal"]) for g in result["groups"] if g["measured_minus_ideal"] is not None]
    result["checks"] = {
        "gvp_group_inherits_the_tested_coefficient": config.gvp_weight_decay is None,
        "steps_match_training_ions_over_batch": steps_per_epoch == batches and steps[0] == batches * saved_epochs
        and int(config.epochs) == saved_epochs,
        "baseline_groups_match_the_real_run": membership["passed"],
        "fp32_decay_within_tolerance_of_ideal": (not tested) or (bool(deviations) and max(deviations) <= FACTOR_TOLERANCE),
    }
    result["passed"] = all(result["checks"].values())
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", type=Path, default=CAMPAIGN, help="workstation campaign data directory")
    parser.add_argument("--campaign-copy", type=Path, help="evidence copy of the VM campaign root (default: newest)")
    parser.add_argument("--family", action="append", choices=v3.GVP_FAMILIES, help="default: both step D families")
    parser.add_argument("--recipe", action="append", help=f"repeatable; default {DEFAULT_RECIPES}")
    parser.add_argument("--checkpoint", action="append", default=[],
                        help="FAMILY=RUN_DIR: a completed run whose model has the audited recipe's architecture")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args(argv)
    import torch

    copy = args.campaign_copy or latest_campaign_copy(args.campaign)
    paths = v3.V3Paths(copy)
    overrides = dict(item.split("=", 1) for item in args.checkpoint)
    started = time.time()
    results = []
    for family in args.family or v3.GVP_FAMILIES:
        for recipe in args.recipe or DEFAULT_RECIPES:
            if family not in v3.resolve_recipe(recipe)["families"]:
                continue
            changing = sorted(set(v3.recipe_components(recipe)) & ARCHITECTURE_CHANGING)
            require(not changing or family in overrides,
                    f"{recipe} changes the model ({changing}); pass --checkpoint {family}=RUN_DIR of such a run")
            run_dir = Path(overrides[family]) if family in overrides else baseline_run_dir(
                copy, args.campaign / "durable", family)
            results.append(audit_recipe(paths, family, recipe, run_dir))
            last = results[-1]
            print(f"{family:16s} {recipe:10s} " + "; ".join(
                f"{g['group']}: lr {g['initial_lr']:g} wd {g['weight_decay']:g} ideal {g['ideal_cumulative_factor']:.6f} "
                f"fp32 {g['measured_fp32_factor']['median']:.6f}" for g in last["groups"]) +
                f" | {'ok' if last['passed'] else 'CHECK FAILED'}", flush=True)
    report = {"audit": "v3 step D effective weight decay", "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "campaign_copy": str(copy), "torch": torch.__version__, "dtype": "float32",
              "factor_tolerance": FACTOR_TOLERANCE, "seconds": round(time.time() - started, 1),
              "note": "Decay operation only (zero gradients, checkpoint weights); not a forecast of trained "
                      "weights or accuracy. No parameter is exempt from decay except those that receive no "
                      "gradient in the family (the optimizer never updates them).",
              "results": results, "passed": bool(results) and all(r["passed"] for r in results)}
    out_dir = args.out_dir or args.campaign / "audits" / f"d_decay_audit_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "decay_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": str(out_dir / "decay_audit.json"), "passed": report["passed"],
                      "seconds": report["seconds"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"decay audit refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
