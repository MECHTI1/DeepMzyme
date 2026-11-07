"""Runner of the EC class vs CLEAN30 campaign: unit matrix, training-side inputs, views, dry run, smoke units,
simulation, manifests.

Actions
  plan          Per-fold unit matrix (S1 96, S2 <= 72, S3 8, S4 5; at most 181 fits) as deterministic JSON.
  cohort-audit  Training-side cohort audit over the five training sides (P2-3): load status as the trainer
                maps it, selected sites, D2 geometry and charge-proxy parity, sequences, per-protein inputs
                and labels; writes <campaign root>/cohort once; exit 1 when a check stops it.
  r2-lists      CLEAN R2 and fold-0 R2-inner accession allowlists, from the training-side identity folders
                and a passed cohort audit (P2-6).
  overlay build Data overlay (P2-7): copies of every training-side input of later steps, laid out like the
                data root, with its manifest, reproducible tar, .sha256 file and build receipt. Copies are
                byte copies, except that under D2 option d the external feature files carry a charge proxy
                derived from the selected metals only.
  overlay verify
                Rehash and re-derive an overlay and check its tar and build receipt; exit 1 with the problems.
  transfer-list EC-class transfer list (P2-9d) from the downloaded nomenclature file, in the scorer's
                ec_class_transfers.csv format, with a receipt; keyed by EC number, no label is read.
  build-view    Training-only view of one outer fold (byte copies, per-file SHA-256, read-only); an inner
                membership needs its certify-inner receipt.
  smoke-view    Engineering subset view: --max-proteins eligible structures by --seed, plus a 2-fold
                smoke EC membership.
  verify-view   Rehash a view and cross-check its IDs, inputs, membership, probe list and chain means;
                writable data files count only where chmod works (file_modes_enforced).
  apply-view-modes
                Set a built view's file modes again (chmod only; no byte or manifest change), e.g. after
                Data1 changes driver; writes receipts/file_modes_applied.json.
  dry-run       Check every planned command of a view, then run the trainer's preparation path for one
                inner and one refit unit per probed family under the held-out guard and a read recorder
                (--cap-prefix: the probes' cap, default the workstation cap).
  simulate      Null-adoption sweep of the promotion rule (P2-9) on fold populations calibrated from the
                data overlay's identity copies (--calibrate-from K=PATH, one per fold the sweep uses);
                write-once records; exit 4 when the CPU-hour forecast stops it.
  simulate-s1   Source-sealed characterization of the fixed amended S1 rule; historical noise inputs,
                no model training; --first-only times the first scenario before continuing.
  run-unit      One capped engineering smoke fit (--smoke only; real fits come with GPU execution) on a
                view that verifies again and holds a passed dry-run receipt.
  verify-unit   Check the receipts of finished units (terminal checkpoint, steps, no generation, ...);
                --delete-smoke-checkpoints then removes the verified smoke checkpoints (gate G1.10).
  manifest      Freeze manifests: --kind freeze1 (pre-registration inputs; each view verified, with its
                chain means, membership and dry run; a real one needs all five folds, --plans-dir and rule
                R-CMD), freeze1-document (the hashed Freeze-1 document and its rendering), freeze1b (the
                identity audit addendum) or freeze2 (models and code pinned for the label-free test pass;
                every refit re-verified; a smoke manifest is marked so the test pass refuses it).
  verify-freeze1
                Check that the Freeze-1 document hashes to F, that F is in log.md at HEAD and that the
                commit introducing it is an ancestor of HEAD (--require-pushed: also on origin/main).
  amend-s1      Build an immutable, source-pinned draft S1 amendment from completed verification evidence.
  verify-s1-amendment
                Authenticate the amendment; --draft permits review before publication, never real execution.
  select        Inner-fold selection step S1, S2 or S3 of one outer fold from verified fits (L5 certified
                on the plan's eight inner fits), appended to the hash-chained step records.
  gap-gate      Fold-0 development gap gate after S1 or S3 against the CLEAN inner arms (refused under
                amendment 2, where its computation is a descriptive block of the futility record).
  plan-v2       Amendment-2 per-fold unit matrix (fold 0: S1 112, S4 5; folds 1-4: S1 96, S4 5) as
                deterministic JSON, written once to <campaign root>/plans_v2/fold<K>.json.
  futility      Amendment-2 fold-0 futility rule: projected probability that the final claim succeeds;
                continue to folds 1-4 only at >= 0.20; a binding run writes the record, its pointer file and the
                FUT step record once (a run that cannot bind is refused; --smoke writes the record only);
                prints totals only.
  futility-oc   Synthetic operating characteristics of the futility rule (futility_oc.json, written once).
  amend-2       Build the canonical amendment-2 record once from the reviewed draft, the rule text and the
                maintainer's decisions file (C1-C8, C6b); --render writes its rendering.
  verify-amendment-chain
                Authenticate the amendment chain with the general chain verifier; write its receipt (the
                scoring pass binds it).
  dry-run-v2    Plan-v2 dry run of one view against the declared pretraining-input stub (write=False); the
                receipt is written once outside the view.
  refit-recipe  The verified refit recipe of one fold (fold 0: with the binding futility continue), for the
                scoring pass (rule A2-SEL-9).
  pretrain-corpus
                Amendment-2 F6 inputs: holdout-lists (R1-inner lists in a scratch identity copy), build (the
                cache-free corpus table bound before publication), fasta (the training-side union FASTA), attach
                (the corpus and its sequence means in the view's attached folder) and verify.
  refit, execute
                Not implemented yet (Phase 3 and 4 execution code); each prints its planned interface and
                exits 3.

Real (non-smoke) campaign actions refuse a tree with uncommitted or untracked changes. simulate-s1
is a source-sealed development characterization and does not confer training authority. Every action refuses
test flags and test-named paths, and every receipt records git HEAD, tree status, code hashes and
the environment. Fits and probes run under the workstation cap (2 cores, 3 GB, no swap). The input
actions (cohort-audit, r2-lists, overlay, simulate) print no accession; counts that are per-fold
class numbers are printed for fold 0 at most, and their receipts and records keep the rest. Exit
codes: 0 done, 1 a check or unit did not pass, 2 refused or an input error, 3 not implemented, 4 a
simulation stopped by its forecast. See src/ec_clean30/{campaign,views,commands,provenance,cohort,r2_lists,
overlay,simulation,transfers,freeze,selector}.py for the rules.
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
    membership = manifest["membership"]
    # No eligible or membership row counts on screen: with n_structures they give the fold's count of proteins
    # with several EC classes, and per-fold class numbers are never printed for more than one fold. They stay
    # in the view manifest and the receipts.
    _print({"view": result["view"], "view_manifest_sha256": result["manifest_sha256"], "kind": manifest["kind"],
            "n_structures": manifest["structures"]["n"], "eligible_ids_sha256": manifest["eligible"]["ids_sha256"],
            "eligible_matches_membership": None if membership is None else True,
            "membership": None if membership is None else {key: value for key, value in membership.items()
                                                           if key != "n_rows"},
            "allowlist_sha256": manifest["structures"]["allowlist_sha256"]})
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
    # No eligible or membership row counts on screen (per-fold class numbers; see _build); the receipt keeps them.
    membership = receipt["membership"]
    _print({**{key: receipt[key] for key in ("status", "view_manifest_sha256", "n_structures", "chain_means",
                                             "file_modes_enforced")},
            "eligible_matches_membership": None if membership is None else True,
            "membership": None if membership is None else {key: value for key, value in membership.items()
                                                           if key != "n_rows"}})
    return 0


def action_apply_view_modes(args: argparse.Namespace) -> int:
    from ec_clean30.provenance import provenance_record
    from ec_clean30.views import FILE_MODES_APPLIED_RECEIPT_NAME, apply_view_modes

    view = _load_view(args)
    git = _git_for(view.smoke)
    receipt = apply_view_modes(view.root, provenance=provenance_record(smoke=view.smoke, git=git))
    _print({"outcome": receipt["outcome"], "view_manifest_sha256": receipt["view_manifest_sha256"],
            "n_paths": receipt["n_paths"], "n_ignored": receipt["n_ignored"], "n_refused": receipt["n_refused"],
            "filesystem": receipt["filesystem"], "receipt": str(view.receipts / FILE_MODES_APPLIED_RECEIPT_NAME),
            "receipt_sha256": receipt["receipt_sha256"]})
    return 0


def action_dry_run(args: argparse.Namespace) -> int:
    import shlex

    from ec_clean30.commands import CAP_PREFIX, dry_run
    from ec_clean30.provenance import provenance_record
    from ec_clean30.views import ViewVerificationError

    view = _load_view(args)
    _refuse_paths(args, ["runs"])
    # The probes' cap prefix: one shell-quoted string, or the workstation cap when not given (kept inside this
    # function, so the dry-run source hash of rule R-CMD covers it).
    cap_prefix = CAP_PREFIX if args.cap_prefix is None else tuple(shlex.split(args.cap_prefix))
    if not cap_prefix:
        raise SystemExit("--cap-prefix needs a command prefix; dry-run probes always run under a cap.")
    git = _git_for(view.smoke)
    try:
        receipt = dry_run(view, runs_dir=args.runs, probe_families=_csv(args.probe_families) or ("F4",),
                          exclude_levers=_csv(args.exclude_levers) or (), timeout=args.timeout,
                          cap_prefix=cap_prefix, provenance=provenance_record(smoke=view.smoke, git=git))
    except ViewVerificationError as exc:
        _print({"status": "failed", "stage": "verify-view", "problems": exc.problems})
        return 1
    _print({"status": receipt["status"], "receipt": str(view.receipts / "dry_run_receipt.json"),
            "n_commands_checked": receipt["n_commands_checked"], "static_failures": receipt["static_failures"][:10],
            "probes": [{key: probe.get(key) for key in ("unit_id", "kind", "status", "problems")}
                       for probe in receipt["probes"]]})
    return 0 if receipt["status"] == "passed" else 1


# Folds whose per-fold counts the input actions print (as the r2-lists summary does); the others stay in the
# private records: eligible protein counts with the training structure counts give each fold's count of proteins
# with several EC classes, and per-fold class numbers are shown for one fold at most.
PRINTED_COUNT_FOLDS: tuple[int, ...] = (0,)


def action_simulate(args: argparse.Namespace) -> int:
    from ec_clean30.provenance import refuse_test_paths
    from ec_clean30.simulation import SimulationError, run_simulate_action

    # Every path is refused by name before the clean-tree check (the sweep checks them again); a K=PATH value
    # also by its path part alone, since "K=" is part of the value's first component.
    _refuse_paths(args, ["sweep", "out", "calibrate_from"])
    for value in args.calibrate_from or ():
        refuse_test_paths([str(value).partition("=")[2] or None], what="--calibrate-from")
    try:
        code, payload = run_simulate_action(args)
    except SimulationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    calibrations = payload.get("calibrations") or {}
    payload = {**payload, "calibrations": {
        fold: record if int(fold) in PRINTED_COUNT_FOLDS else {key: value for key, value in record.items()
                                                                if key != "n_proteins"}
        for fold, record in calibrations.items()}}
    _print(payload)
    return code


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
    from ec_clean30 import freeze

    # The real Freeze-1 inputs, the Freeze-1 document and Freeze 1b (ec_clean30.freeze); the smoke Freeze-1
    # inputs and Freeze 2 keep the path below.
    handler = freeze.route_manifest(args)
    if handler is not None:
        return handler(args)

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


def action_amend_s1(args: argparse.Namespace) -> int:
    from ec_clean30.freeze import build_s1_amendment
    from ec_clean30.provenance import sha256_file

    document = build_s1_amendment(
        freeze1_path=args.freeze1,
        evidence={"simulation_spec": Path(args.simulation_spec),
                  "simulation_summary": Path(args.simulation_summary), "verification": Path(args.verification)},
        out=args.out,
    )
    _print({"status": "draft", "amendment_sha256": sha256_file(args.out),
            "s1_rule": document["s1_rule"], "publication_required_before_execution": True})
    return 0


def action_simulate_s1(args: argparse.Namespace) -> int:
    from ec_clean30.s1_simulation import action

    return action(args)


def action_verify_s1_amendment(args: argparse.Namespace) -> int:
    from ec_clean30.freeze import verify_s1_amendment

    record = verify_s1_amendment(args.amendment, expected_freeze1_sha256=args.freeze1_sha256,
                                 require_published=not args.draft)
    _print({key: record[key] for key in ("document_sha256", "amendment_chain", "s1_rule", "published")})
    return 0


def action_plan_v2(args: argparse.Namespace) -> int:
    """Amendment 2: the fold's v2 plan (``campaign.build_fold_plan_v2``), written once with its receipt to
    ``<campaign root>/plans_v2/fold<K>.json``, or its counts printed."""
    from ec_clean30.campaign import build_fold_plan_v2
    from ec_clean30.provenance import provenance_record, write_new_json

    plan = build_fold_plan_v2(args.fold)
    if args.out is None:
        _print(plan if args.full else {"fold": plan["fold"], "counts": plan["counts"], "plan_sha256": plan["plan_sha256"],
                                       "families": plan["families"]})
        return 0
    _refuse_paths(args, ["out"])
    out = Path(args.out)
    if out.parent.name != "plans_v2" or out.name != f"fold{args.fold}.json":
        print(f"error: --out must be <campaign root>/plans_v2/fold{args.fold}.json", file=sys.stderr)
        return 2
    git = _git_for(args.smoke)
    digest = write_new_json(out, plan)
    receipt = {"plan": out.name, "plan_file_sha256": digest, "plan_sha256": plan["plan_sha256"],
               "counts": plan["counts"], "provenance": provenance_record(smoke=args.smoke, git=git)}
    write_new_json(out.with_name(out.name + ".receipt.json"), receipt)
    _print({"plan": str(out), "plan_sha256": plan["plan_sha256"], "counts": plan["counts"]})
    return 0


def build_parser() -> argparse.ArgumentParser:
    # Torch-free at import; each module defines its own action's arguments and handler.
    from ec_clean30 import cohort, freeze, futility, overlay, pretrain_corpus, r2_lists, selector, transfers

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

    cohort.add_cohort_audit_arguments(sub.add_parser("cohort-audit", help=cohort.COHORT_AUDIT_HELP)).set_defaults(
        handler=cohort.action_cohort_audit)
    r2_lists.add_r2_lists_arguments(sub.add_parser("r2-lists", help=r2_lists.R2_LISTS_HELP)).set_defaults(
        handler=r2_lists.action_r2_lists)
    overlay.add_overlay_arguments(sub.add_parser("overlay", help=overlay.OVERLAY_HELP))  # build, verify
    transfers.add_transfer_list_arguments(sub.add_parser("transfer-list", help=transfers.TRANSFER_LIST_HELP)) \
        .set_defaults(handler=transfers.action_transfer_list)

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

    modes = sub.add_parser("apply-view-modes", help="Set a built view's file modes again (chmod only).")
    modes.add_argument("--view", required=True)
    modes.add_argument("--smoke", action="store_true")
    modes.set_defaults(handler=action_apply_view_modes)

    dry = sub.add_parser("dry-run", help="Check every planned command of a view and probe preparation.")
    dry.add_argument("--view", required=True)
    dry.add_argument("--smoke", action="store_true")
    dry.add_argument("--runs", default=None,
                     help="Runs folder the commands name: <campaign root>/runs/fold<k> (smoke view: "
                          "<campaign root>/smoke_runs/<name>) or a folder inside it, by real path; that folder "
                          "is the default.")
    dry.add_argument("--probe-families", default="F4", help="Comma list of families to probe (inner and refit).")
    dry.add_argument("--exclude-levers", default=None, help="Levers whose inputs are not built yet (e.g. L8).")
    dry.add_argument("--timeout", type=float, default=None, help="Seconds per probe.")
    dry.add_argument("--cap-prefix", default=None,
                     help="Cap each probe runs under, as one shell-quoted string (default the workstation cap: "
                          "systemd-run --user --scope -q -p CPUQuota=200%% -p MemoryMax=3G -p MemorySwapMax=0).")
    dry.set_defaults(handler=action_dry_run)

    simulate = sub.add_parser("simulate",
                              help="Null-adoption sweep of the promotion rule on calibrated fold populations.")
    simulate.add_argument("--sweep", required=True,
                          help="Sweep file (format deepmzyme_ec_clean30_simulation_sweep_v1).")
    simulate.add_argument("--out", required=True,
                          help="Simulation folder (new, or one this sweep started); not under the campaign's identity "
                               "folder, a calibration folder or a data overlay.")
    simulate.add_argument("--calibrate-from", action="append", default=[], metavar="K=PATH",
                          help="Fold K's identity folder in the data overlay (<overlay>/identity/fold<K>); one for "
                               "every fold the sweep uses, on every call.")
    simulate.add_argument("--points", default=None,
                          help="Comma list of point IDs (default: every point without a record).")
    simulate.add_argument("--max-forecast-cpu-hours", type=float, default=8.0,
                          help="Stop before the next point when the whole-sweep forecast exceeds this.")
    simulate.add_argument("--smoke", action="store_true",
                          help="Engineering use: dirty tree, folders outside the overlay.")
    simulate.set_defaults(handler=action_simulate)

    run = sub.add_parser("run-unit", help="One capped engineering smoke fit.")
    run.add_argument("--smoke", action="store_true")
    run.add_argument("--view", required=True)
    run.add_argument("--family", required=True, choices=("F1", "F2", "F3", "F4", "F5", "C-G", "CG"))
    run.add_argument("--epochs", type=int, default=1)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--refit", action="store_true", help="Train on the whole view with --val-fraction 0 and the probe list.")
    run.add_argument("--inner-fold", type=int, default=0)
    run.add_argument("--lr-index", type=int, default=1, help="Grid setting of the family (default the second).")
    run.add_argument("--runs", required=True,
                     help="<campaign root>/smoke_runs/<name> of the smoke view, or a folder inside it.")
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

    manifest = sub.add_parser("manifest", help="Freeze-1 inputs, Freeze-1 document, Freeze 1b or Freeze-2 manifest.")
    manifest.add_argument("--kind", required=True, choices=freeze.MANIFEST_KINDS)
    manifest.add_argument("--out", required=True, help="New manifest file.")
    manifest.add_argument("--runs", action="append", default=[], help="Verified runs folder (freeze2; repeatable).")
    manifest.add_argument("--view", action="append", default=[], help="View folder (freeze1; repeatable).")
    manifest.add_argument("--clean-fit", action="append", default=[],
                          help="Finished CLEAN fit work directory of run_clean_split30_comparator.py (freeze2; "
                               "repeatable; a real manifest needs R1 fits in every outer fold).")
    manifest.add_argument("--test-pass-output-root", default=None,
                          help="Absolute folder of the single test pass (real freeze2; a smoke manifest omits it).")
    manifest.add_argument("--smoke", action="store_true")
    freeze.add_manifest_freeze_arguments(manifest)
    manifest.set_defaults(handler=action_manifest)
    freeze.add_verify_freeze1_arguments(sub.add_parser("verify-freeze1", help=freeze.VERIFY_FREEZE1_HELP)) \
        .set_defaults(handler=freeze.action_verify_freeze1)

    amendment = sub.add_parser("amend-s1", help="Build the immutable draft S1 amendment.")
    for flag in ("freeze1", "simulation-spec", "simulation-summary", "verification", "out"):
        amendment.add_argument("--" + flag, required=True)
    amendment.set_defaults(handler=action_amend_s1)
    amendment_check = sub.add_parser("verify-s1-amendment", help="Authenticate the amended S1 source and evidence.")
    amendment_check.add_argument("--amendment", required=True)
    amendment_check.add_argument("--freeze1-sha256", required=True)
    amendment_check.add_argument("--draft", action="store_true", help="Review only; do not require publication.")
    amendment_check.set_defaults(handler=action_verify_s1_amendment)

    selector.add_select_arguments(sub.add_parser("select", help=selector.SELECT_HELP)) \
        .set_defaults(handler=selector.action_select)
    selector.add_gap_gate_arguments(sub.add_parser("gap-gate", help=selector.GAP_GATE_HELP)) \
        .set_defaults(handler=selector.action_gap_gate)

    # Amendment 2 (new actions only; the pinned runner functions are unchanged).
    plan_v2 = sub.add_parser("plan-v2", help="Amendment-2 per-fold unit matrix (plan format v2).")
    plan_v2.add_argument("--fold", type=int, required=True)
    plan_v2.add_argument("--out", default=None, help="<campaign root>/plans_v2/fold<K>.json (written once).")
    plan_v2.add_argument("--full", action="store_true", help="Print the whole plan instead of its counts.")
    plan_v2.add_argument("--smoke", action="store_true", help="Engineering use: allow a dirty tree.")
    plan_v2.set_defaults(handler=action_plan_v2)
    futility.add_futility_arguments(sub.add_parser("futility", help=futility.FUTILITY_HELP)) \
        .set_defaults(handler=futility.action_futility)
    futility_oc = sub.add_parser("futility-oc", help="Synthetic operating characteristics of the futility rule.")
    futility_oc.add_argument("--out", required=True, help="futility_oc.json (written once).")
    futility_oc.set_defaults(handler=futility.action_futility_oc)
    from ec_clean30 import amendment2

    amendment2.add_amendment2_arguments(sub)
    pretrain_corpus.add_pretrain_corpus_arguments(
        sub.add_parser("pretrain-corpus", help=pretrain_corpus.PRETRAIN_CORPUS_HELP)) \
        .set_defaults(handler=pretrain_corpus.action_pretrain_corpus)

    s1_simulation = sub.add_parser("simulate-s1", help="Bounded fixed-rule S1 characterization.")
    for flag in ("spec", "freeze1", "campaign-root", "out"):
        s1_simulation.add_argument("--" + flag, required=True)
    s1_simulation.add_argument("--first-only", action="store_true", help="Time and retain the first scenario.")
    s1_simulation.set_defaults(handler=action_simulate_s1)

    for name in ("refit", "execute"):
        first_line = PLANNED_INTERFACES[name].splitlines()[0]
        stub = sub.add_parser(name, help=f"Not implemented yet (Phase 3 and 4). {first_line}")
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
    from ec_clean30.r2_lists import R2ListError
    from ec_clean30.transfers import TransferListError
    from ec_clean30.views import ViewError

    try:
        return int(args.handler(args))
    except DirtyTreeError as exc:
        # The tree check names up to five status entries; an untracked file's name could carry a held-out
        # name, so only the reason and the entry count are shown (git status lists the entries).
        reason, sample, _rest = str(exc).partition(" (e.g. ")
        print(f"refused: {reason}" + (". Commit first: real campaign actions run only from a clean tree "
                                      "(engineering smokes use --smoke); git status lists the entries."
                                      if sample else ""), file=sys.stderr)
        return 2
    except HeldOutInputError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    except (CampaignPlanError, CommandError, FreezeManifestError, ViewError, R2ListError, TransferListError,
            FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
