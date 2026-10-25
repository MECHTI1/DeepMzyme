"""Runner of the EC class vs CLEAN30 campaign: unit matrix, training-only views, dry run, smoke units, manifests.

Actions
  plan          Per-fold unit matrix (S1 96, S2 <= 72, S3 8, S4 5; at most 181 fits) as deterministic JSON.
  build-view    Training-only view of one outer fold (byte copies, per-file SHA-256, read-only); an inner
                membership needs its certify-inner receipt.
  smoke-view    Engineering subset view: --max-proteins eligible structures by --seed, plus a 2-fold
                smoke EC membership.
  verify-view   Rehash a view and cross-check its IDs, inputs, membership, probe list and chain means.
  dry-run       Check every planned command of a view, then run the trainer's preparation path for one
                inner and one refit unit per probed family under the held-out guard and a read recorder.
  run-unit      One capped engineering smoke fit (--smoke only; real fits come with GPU execution) on a
                view that verifies again and holds a passed dry-run receipt.
  verify-unit   Check the receipts of finished units (terminal checkpoint, steps, no generation, ...);
                --delete-smoke-checkpoints then removes the verified smoke checkpoints (gate G1.10).
  manifest      Freeze manifests: --kind freeze1 (pre-registration inputs; each view verified, with its
                chain means, membership and dry run) or freeze2 (models and code pinned for the label-free
                test pass; every refit re-verified; a smoke manifest is marked so the test pass refuses it).
  select, gap-gate, refit, execute
                Not implemented yet (they follow gate G1); each prints its planned interface and exits 3.

Real (non-smoke) actions refuse a tree with uncommitted or untracked changes, every action refuses
test flags and test-named paths, and every receipt records git HEAD, tree status, code hashes and
the environment. Fits and probes run under the workstation cap (2 cores, 3 GB, no swap). See
src/ec_clean30/{campaign,views,commands,provenance}.py for the rules.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

EXIT_NOT_IMPLEMENTED = 3

PLANNED_INTERFACES = {
    "select": (
        "select --fold K --step {S1,S2,S3} --runs R [--runs R2 ...] --components C5_TABLE --out STEP_RECORD\n"
        "  Pools the verified inner units' EC prediction exports per (configuration, seed) with\n"
        "  ec_clean30.selection.pool_inner_oof, applies the S1 learning-rate choice and family escalation,\n"
        "  the S2 lever screen (L7 two looks, L9 on pooled logits) or the S3 combination, and appends a\n"
        "  hash-chained step record (ec_clean30.selection.append_step_record)."
    ),
    "gap-gate": (
        "gap-gate --fold 0 --after {S1,S3} --step-chain RECORDS --clean-r1-inner P --clean-r2-inner P --out G\n"
        "  Fold 0 only. Scores DeepMzyme's seed-averaged inner OOF and the CLEAN inner arms on inner-validation\n"
        "  proteins whose full EC numbers occur in R1-inner training (denominators reported); continues if the\n"
        "  gap is >= -3.0 points (after S3, a 1.0-point optimism discount first), else records\n"
        "  maintainer_decision_required."
    ),
    "refit": (
        "refit --fold K --step-chain RECORDS --view V --runs R\n"
        "  Builds the five S4 refit units (seeds 42-46) of the selected recipe on the fold view, with\n"
        "  --val-fraction 0 and the view's probe list, and hands them to execute."
    ),
    "execute": (
        "execute --plan PLAN --view V --runs R --lane N --deadline-unix T --max-total-seconds S --durable-root D\n"
        "  GPU execution: admits one unit at a time through benchmarking.pmm_execution.CampaignExecution\n"
        "  (forecast x 1.25 + reserve, one unchanged rerun, verified persistence), refuses folds 1-4 until fold 0\n"
        "  completed under the same Freeze-1 hash, and never provisions or stops compute."
    ),
}


def _csv(value: Optional[str]) -> Optional[list[str]]:
    if value is None:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def _print(payload) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _git_for(smoke: bool):
    from ec_clean30.provenance import real_action_git_state

    return real_action_git_state(smoke=smoke)


def _refuse_paths(args: argparse.Namespace, names: Sequence[str]) -> None:
    from ec_clean30.provenance import refuse_test_paths

    for name in names:
        value = getattr(args, name, None)
        values = value if isinstance(value, list) else [value]
        refuse_test_paths(values, what=f"--{name.replace('_', '-')}")


def _layout(args: argparse.Namespace):
    from ec_clean30.views import SourceLayout
    from project_paths import DATA_DIR

    return SourceLayout.from_data_root(args.data_root or DATA_DIR, esm_dir=args.esm_dir,
                                       external_root=args.external_root)


def action_plan(args: argparse.Namespace) -> int:
    from ec_clean30.campaign import build_fold_plan
    from ec_clean30.provenance import provenance_record, write_new_json

    plan = build_fold_plan(args.fold, families=_csv(args.families), levers=_csv(args.levers),
                           s1_winner=args.s1_winner)
    if args.out is None:
        _print(plan if args.full else {"fold": plan["fold"], "counts": plan["counts"], "plan_sha256": plan["plan_sha256"],
                                       "filters": plan["filters"], "levers_not_applicable": plan["levers_not_applicable"]})
        return 0
    _refuse_paths(args, ["out"])
    git = _git_for(args.smoke)
    out = Path(args.out)
    digest = write_new_json(out, plan)
    receipt = {"plan": out.name, "plan_file_sha256": digest, "plan_sha256": plan["plan_sha256"],
               "counts": plan["counts"], "provenance": provenance_record(smoke=args.smoke, git=git)}
    write_new_json(out.with_name(out.name + ".receipt.json"), receipt)
    _print({"plan": str(out), "plan_sha256": plan["plan_sha256"], "counts": plan["counts"]})
    return 0


def _build(args: argparse.Namespace, *, smoke: bool) -> int:
    from ec_clean30.provenance import provenance_record
    from ec_clean30.views import build_view

    _refuse_paths(args, ["out", "data_root", "esm_dir", "external_root", "inner_membership", "inner_certificate"])
    git = _git_for(smoke)
    inner = None
    if getattr(args, "inner_membership", None):
        if not args.inner_membership_sha256:
            raise SystemExit("--inner-membership needs --inner-membership-sha256")
        inner = (Path(args.inner_membership), args.inner_membership_sha256)
    result = build_view(fold=args.fold, layout=_layout(args), out_dir=args.out, smoke=smoke,
                        max_proteins=getattr(args, "max_proteins", None), seed=getattr(args, "seed", None),
                        bundle_id=args.bundle_id, inner_membership=inner,
                        inner_certificate=getattr(args, "inner_certificate", None), probe_count=args.probe_count,
                        provenance=provenance_record(smoke=smoke, git=git))
    manifest = result["manifest"]
    _print({"view": result["view"], "view_manifest_sha256": result["manifest_sha256"], "kind": manifest["kind"],
            "n_structures": manifest["structures"]["n"], "n_eligible": manifest["eligible"]["n"],
            "membership": manifest["membership"], "allowlist_sha256": manifest["structures"]["allowlist_sha256"]})
    return 0


def action_build_view(args: argparse.Namespace) -> int:
    return _build(args, smoke=False)


def action_smoke_view(args: argparse.Namespace) -> int:
    return _build(args, smoke=True)


def _load_view(args: argparse.Namespace):
    from ec_clean30.views import View

    _refuse_paths(args, ["view"])
    view = View.load(args.view)
    if bool(getattr(args, "smoke", False)) != view.smoke:
        raise SystemExit(f"{view.root} is a {view.kind}; pass --smoke exactly for smoke views.")
    return view


def action_verify_view(args: argparse.Namespace) -> int:
    from ec_clean30.provenance import provenance_record
    from ec_clean30.views import ViewVerificationError, verify_view

    view = _load_view(args)
    git = _git_for(view.smoke)
    try:
        receipt = verify_view(view.root, require_chain_means=_csv(args.require_chain_means) or (),
                              check_sources=args.check_sources, provenance=provenance_record(smoke=view.smoke, git=git))
    except ViewVerificationError as exc:
        _print({"status": "failed", "problems": exc.problems})
        return 1
    _print({key: receipt[key] for key in ("status", "view_manifest_sha256", "n_structures", "n_eligible",
                                          "membership", "chain_means")})
    return 0


def action_dry_run(args: argparse.Namespace) -> int:
    from ec_clean30.commands import dry_run
    from ec_clean30.provenance import provenance_record
    from ec_clean30.views import ViewVerificationError

    view = _load_view(args)
    _refuse_paths(args, ["runs"])
    git = _git_for(view.smoke)
    try:
        receipt = dry_run(view, runs_dir=args.runs, probe_families=_csv(args.probe_families) or ("F4",),
                          exclude_levers=_csv(args.exclude_levers) or (), timeout=args.timeout,
                          provenance=provenance_record(smoke=view.smoke, git=git))
    except ViewVerificationError as exc:
        _print({"status": "failed", "stage": "verify-view", "problems": exc.problems})
        return 1
    _print({"status": receipt["status"], "receipt": str(view.receipts / "dry_run_receipt.json"),
            "n_commands_checked": receipt["n_commands_checked"], "static_failures": receipt["static_failures"][:10],
            "probes": [{key: probe.get(key) for key in ("unit_id", "kind", "status", "problems")}
                       for probe in receipt["probes"]]})
    return 0 if receipt["status"] == "passed" else 1


def action_run_unit(args: argparse.Namespace) -> int:
    from ec_clean30.campaign import KIND_INNER, KIND_REFIT, default_smoke_recipe, smoke_unit_id
    from ec_clean30.commands import build_unit_command, run_unit
    from ec_clean30.provenance import provenance_record, source_tree_sha256

    if not args.smoke:
        print("run-unit for real fits is not implemented yet: real units run through 'execute' "
              "(GPU execution).\n" + PLANNED_INTERFACES["execute"])
        return EXIT_NOT_IMPLEMENTED
    view = _load_view(args)
    _refuse_paths(args, ["runs"])
    git = _git_for(True)
    recipe = default_smoke_recipe(args.family, epochs=args.epochs, lr_index=args.lr_index)
    kind = KIND_REFIT if args.refit else KIND_INNER
    unit_id = smoke_unit_id(view.fold, recipe, refit=args.refit, seed=args.seed, inner_fold=args.inner_fold)
    command = build_unit_command(view, recipe, unit_id=unit_id, kind=kind, seed=args.seed, runs_dir=args.runs,
                                 step="smoke", inner_fold=None if args.refit else args.inner_fold,
                                 git_head=git.get("head"), tree_sha256=source_tree_sha256())
    receipt = run_unit(command, view=view, timeout=args.timeout, provenance=provenance_record(smoke=True, git=git))
    _print({key: receipt.get(key) for key in ("unit_id", "status", "returncode", "wall_seconds",
                                              "children_peak_rss_bytes", "log")})
    return 0 if receipt["status"] == "completed" else 1


def action_verify_unit(args: argparse.Namespace) -> int:
    from ec_clean30.commands import (
        SMOKE_CHECKPOINT_DELETION_RECEIPT_NAME, VERIFY_UNIT_RECEIPT_NAME, delete_smoke_checkpoints, verify_units,
    )
    from ec_clean30.provenance import provenance_record, utc_now
    from ec_clean30.views import write_receipt_file

    _refuse_paths(args, ["runs"])
    if args.delete_smoke_checkpoints and not args.smoke:
        raise SystemExit("--delete-smoke-checkpoints removes engineering smoke checkpoints only; it needs --smoke.")
    git = _git_for(args.smoke)
    provenance = provenance_record(smoke=args.smoke, git=git)
    runs = Path(args.runs)
    receipt = verify_units(runs, unit_ids=args.unit or None, provenance=provenance, write=False)
    smoke_units = [unit["unit_id"] for unit in receipt["units"] if unit.get("smoke")]
    if bool(smoke_units) != args.smoke or (args.smoke and len(smoke_units) != len(receipt["units"])):
        print("Smoke and real units are verified separately; pass --smoke exactly for engineering smoke units.")
        return 1
    receipt_sha = write_receipt_file(runs / VERIFY_UNIT_RECEIPT_NAME, receipt)
    printed = {"status": receipt["status"], "receipt": str(runs / VERIFY_UNIT_RECEIPT_NAME),
               "receipt_sha256": receipt_sha,
               "units": [{"unit_id": unit["unit_id"], "passed": unit["passed"],
                          "failed_checks": sorted(name for name, item in unit["checks"].items() if not item["ok"])}
                         for unit in receipt["units"]]}
    if args.delete_smoke_checkpoints and receipt["status"] == "passed":
        deleted = delete_smoke_checkpoints(receipt, runs)
        deletion = {"schema_version": 1, "created_utc": utc_now(), "verify_unit_receipt_sha256": receipt_sha,
                    "deleted_checkpoints": deleted, "provenance": provenance}
        printed["deletion_receipt_sha256"] = write_receipt_file(runs / SMOKE_CHECKPOINT_DELETION_RECEIPT_NAME,
                                                                deletion)
        printed["n_deleted_checkpoints"] = len(deleted)
    _print(printed)
    return 0 if receipt["status"] == "passed" else 1


def action_manifest(args: argparse.Namespace) -> int:
    from ec_clean30.campaign import N_OUTER_FOLDS, build_fold_plan
    from ec_clean30.commands import collect_clean_checkpoints, collect_refit_checkpoints
    from ec_clean30.provenance import (
        build_freeze1_manifest, build_freeze2_manifest, provenance_record, write_new_json,
    )
    from ec_clean30.views import View, freeze1_view_record

    _refuse_paths(args, ["out", "runs", "view", "clean_fit"])
    git = _git_for(args.smoke)
    provenance = provenance_record(smoke=args.smoke, git=git)
    if args.kind == "freeze1":
        views = []
        for path in args.view or []:
            view = View.load(path)
            if view.smoke and not args.smoke:
                raise SystemExit(f"{view.root} is a smoke view; refused in a real Freeze-1 manifest.")
            # Verified now (chain means required); a real manifest also needs the view's passed dry run.
            views.append(freeze1_view_record(view.root, require_dry_run=not args.smoke, provenance=provenance))
        folds = [record["fold"] for record in views]
        if len(folds) != len(set(folds)):
            raise SystemExit(f"Each outer fold may have one view in a Freeze-1 manifest, got folds {folds}.")
        plans = [build_fold_plan(fold) for fold in range(N_OUTER_FOLDS)]
        manifest = build_freeze1_manifest(plans=plans, views=views, smoke=args.smoke, provenance=provenance)
    else:
        if not args.runs or (not args.smoke and not (args.test_pass_output_root and args.clean_fit)):
            raise SystemExit("--kind freeze2 needs --runs and (for a real manifest) --clean-fit and "
                             "--test-pass-output-root")
        checkpoints = collect_refit_checkpoints(args.runs, smoke=args.smoke)
        if args.clean_fit:
            checkpoints += collect_clean_checkpoints(args.clean_fit, smoke=args.smoke)
        manifest = build_freeze2_manifest(checkpoints=checkpoints,
                                          test_pass_output_root=args.test_pass_output_root, smoke=args.smoke,
                                          provenance=provenance)
    digest = write_new_json(Path(args.out), manifest)
    _print({"manifest": str(args.out), "kind": args.kind, "sha256": digest})
    return 0


def _not_implemented(name: str):
    def action(_args: argparse.Namespace) -> int:
        print(f"{name} is not implemented yet; it follows gate G1. Planned interface:\n{PLANNED_INTERFACES[name]}")
        return EXIT_NOT_IMPLEMENTED
    return action


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)

    plan = sub.add_parser("plan", help="Per-fold unit matrix as deterministic JSON.")
    plan.add_argument("--fold", type=int, required=True)
    plan.add_argument("--families", default=None, help="Comma list (default all six).")
    plan.add_argument("--levers", default=None, help="Comma list of L1..L9 (default all).")
    plan.add_argument("--s1-winner", default=None, help="Restrict S2 to the levers that apply to this family.")
    plan.add_argument("--out", default=None, help="Write the plan here (a new file) plus a provenance receipt.")
    plan.add_argument("--full", action="store_true", help="Print every unit when no --out is given.")
    plan.add_argument("--smoke", action="store_true", help="Engineering use: allow a dirty tree for --out.")
    plan.set_defaults(handler=action_plan)

    def source_arguments(target: argparse.ArgumentParser) -> None:
        target.add_argument("--fold", type=int, required=True)
        target.add_argument("--out", required=True, help="New view folder (outside CLEAN_30_main and shared roots).")
        target.add_argument("--data-root", default=None, help="Data root (default: the project's DeepMzyme_Data).")
        target.add_argument("--esm-dir", default=None,
                            help="Per-residue ESMC-300m caches (default <data-root>/esm_embeddings; exact names only).")
        target.add_argument("--external-root", default=None,
                            help="External residue features (default <data-root>/updated_feature_extraction).")
        target.add_argument("--bundle-id", default=None, help="Dataset bundle ID recorded by every fit.")
        target.add_argument("--probe-count", type=int, default=None, help="Size of the refit probe list.")

    build = sub.add_parser("build-view", help="Training-only view of one outer fold.")
    source_arguments(build)
    build.add_argument("--inner-membership", default=None, help="Certified inner EC membership to copy into the view.")
    build.add_argument("--inner-membership-sha256", default=None)
    build.add_argument("--inner-certificate", default=None,
                       help="certify-inner receipt of the membership (default: inner_certification_receipt.json "
                            "next to it).")
    build.set_defaults(handler=action_build_view)

    smoke = sub.add_parser("smoke-view", help="Engineering subset view with a 2-fold smoke membership.")
    source_arguments(smoke)
    smoke.add_argument("--max-proteins", type=int, required=True)
    smoke.add_argument("--seed", type=int, required=True)
    smoke.set_defaults(handler=action_smoke_view, bundle_id="g1-smoke-view")

    verify = sub.add_parser("verify-view", help="Rehash and cross-check a view.")
    verify.add_argument("--view", required=True)
    verify.add_argument("--smoke", action="store_true")
    verify.add_argument("--require-chain-means", default=None, help="Comma list of layers (final, block26).")
    verify.add_argument("--check-sources", action="store_true",
                        help="Also rehash the source training manifest and site table (training files only).")
    verify.set_defaults(handler=action_verify_view)

    dry = sub.add_parser("dry-run", help="Check every planned command of a view and probe preparation.")
    dry.add_argument("--view", required=True)
    dry.add_argument("--smoke", action="store_true")
    dry.add_argument("--runs", default=None, help="Runs folder the commands name (default <view>/../runs).")
    dry.add_argument("--probe-families", default="F4", help="Comma list of families to probe (inner and refit).")
    dry.add_argument("--exclude-levers", default=None, help="Levers whose inputs are not built yet (e.g. L8).")
    dry.add_argument("--timeout", type=float, default=None, help="Seconds per probe.")
    dry.set_defaults(handler=action_dry_run)

    run = sub.add_parser("run-unit", help="One capped engineering smoke fit.")
    run.add_argument("--smoke", action="store_true")
    run.add_argument("--view", required=True)
    run.add_argument("--family", required=True, choices=("F1", "F2", "F3", "F4", "F5", "C-G", "CG"))
    run.add_argument("--epochs", type=int, default=1)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--refit", action="store_true", help="Train on the whole view with --val-fraction 0 and the probe list.")
    run.add_argument("--inner-fold", type=int, default=0)
    run.add_argument("--lr-index", type=int, default=1, help="Grid setting of the family (default the second).")
    run.add_argument("--runs", required=True)
    run.add_argument("--timeout", type=float, default=None)
    run.set_defaults(handler=action_run_unit)

    check = sub.add_parser("verify-unit", help="Check the receipts of finished units.")
    check.add_argument("--runs", required=True)
    check.add_argument("--unit", action="append", default=[])
    check.add_argument("--smoke", action="store_true")
    check.add_argument("--delete-smoke-checkpoints", action="store_true",
                       help="After every smoke unit passed, delete their terminal checkpoints and write a deletion "
                            "receipt that names the verification receipt's SHA-256.")
    check.set_defaults(handler=action_verify_unit)

    manifest = sub.add_parser("manifest", help="Freeze-1 inputs or Freeze-2 manifest.")
    manifest.add_argument("--kind", required=True, choices=("freeze1", "freeze2"))
    manifest.add_argument("--out", required=True, help="New manifest file.")
    manifest.add_argument("--runs", action="append", default=[], help="Verified runs folder (freeze2; repeatable).")
    manifest.add_argument("--view", action="append", default=[], help="View folder (freeze1; repeatable).")
    manifest.add_argument("--clean-fit", action="append", default=[],
                          help="Finished CLEAN fit work directory of run_clean_split30_comparator.py (freeze2; "
                               "repeatable; a real manifest needs R1 fits in every outer fold).")
    manifest.add_argument("--test-pass-output-root", default=None,
                          help="Absolute folder of the single test pass (real freeze2; a smoke manifest omits it).")
    manifest.add_argument("--smoke", action="store_true")
    manifest.set_defaults(handler=action_manifest)

    for name in ("select", "gap-gate", "refit", "execute"):
        stub = sub.add_parser(name, help=f"Not implemented yet (after G1). {PLANNED_INTERFACES[name].splitlines()[0]}")
        stub.set_defaults(handler=_not_implemented(name), stub=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    if unknown and not getattr(args, "stub", False):
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    if getattr(args, "family", None) == "CG":
        args.family = "C-G"
    from ec_clean30.campaign import CampaignPlanError
    from ec_clean30.commands import CommandError
    from ec_clean30.provenance import DirtyTreeError, FreezeManifestError, HeldOutInputError
    from ec_clean30.views import ViewError

    try:
        return int(args.handler(args))
    except (DirtyTreeError, HeldOutInputError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    except (CampaignPlanError, CommandError, FreezeManifestError, ViewError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
