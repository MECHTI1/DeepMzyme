"""Command-line entry point; kept separate from historical test evaluators."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .manifest import ReleaseError, load_manifest


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Audit and run frozen DeepMzyme metal ensembles; no training or test evaluation.")
    commands = root.add_subparsers(dest="command", required=True)
    catalog = commands.add_parser("catalog", help="Write the fixed, initially unbound Phase 2 release catalog")
    catalog.add_argument("--output", required=True, type=Path)
    bind = commands.add_parser("bind", help="Locate and hash existing development artifacts; never retrain missing members")
    bind.add_argument("--run-root", required=True, action="append", type=Path)
    bind.add_argument("--fold-membership", required=True, type=Path)
    bind.add_argument("--candidate", help="Bind only this candidate's required artifacts; "
                                          "omitting this requires the complete catalog")
    bind.add_argument("--output", required=True, type=Path)
    audit = commands.add_parser("audit", help="Reconstruct OOF predictions and descriptive comparisons from development artifacts")
    audit.add_argument("--manifest", required=True, type=Path)
    audit.add_argument("--output-dir", required=True, type=Path)
    audit.add_argument("--bootstrap-samples", type=int, default=10000)
    audit.add_argument("--seed", type=int, default=42)
    audit.add_argument("--similarity-margin-pp", type=float, default=1.5,
                       help="Descriptive bar: allowed one-sided 95%% lower bound of the BA loss versus the reference")
    audit.add_argument("--recall-floor-pp", type=float, default=3.0,
                       help="Descriptive bar: allowed pooled recall loss per class versus the reference")
    nested = commands.add_parser("nested-selection",
                                 help="Repeat ensemble selection inside cross-validation using saved OOF predictions")
    nested.add_argument("--manifest", required=True, type=Path)
    nested.add_argument("--run-root", action="append", default=[], type=Path,
                        help="Extra development run root to search for complete five-fold families (repeatable)")
    nested.add_argument("--expect-family", action="append", default=[],
                        help="Family name whose OOF availability should be reported (repeatable)")
    nested.add_argument("--late-fusion-candidate", default="top4_lf",
                        help="Catalog candidate compared with the nested late-fusion procedure")
    nested.add_argument("--output", required=True, type=Path)
    nested.add_argument("--bootstrap-samples", type=int, default=10000)
    nested.add_argument("--seed", type=int, default=42)
    parity = commands.add_parser("parity", help="Replay hash-bound development fixtures and compare eligible member predictions")
    parity.add_argument("--manifest", required=True, type=Path)
    parity.add_argument("--fixture", required=True, type=Path)
    parity.add_argument("--output-dir", required=True, type=Path)
    predict = commands.add_parser("predict", help="Predict supported metal identity conditional on a supplied structure and site")
    predict.add_argument("--manifest", required=True, type=Path)
    predict.add_argument("--candidate", help="Catalog candidate to run; defaults to the manifest reference. "
                                             "Missing checkpoints fail explicitly")
    predict.add_argument("--structure", required=True, type=Path)
    site = predict.add_mutually_exclusive_group(required=True)
    site.add_argument("--center", nargs=3, type=float, metavar=("X", "Y", "Z"))
    site.add_argument("--atom", help="CHAIN:RESSEQ:ICODE:ATOM[:ALTLOC]; use _ for blank identifiers")
    site.add_argument("--sites", type=Path, help="JSON list of {id, center:[x,y,z]} or {id, atom:locator}")
    predict.add_argument("--embedding-cache", type=Path)
    predict.add_argument("--generate-embeddings", action="store_true", help="Compute missing ESMC embeddings; default requires cache")
    predict.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    predict.add_argument("--batch-size", type=int, default=32)
    predict.add_argument("--member-predictions", action="store_true", help="Include member scores for development parity diagnostics")
    predict.add_argument("--output", required=True, type=Path)
    benchmark = commands.add_parser("benchmark", help="Measure inference on a hash-bound development fixture, in separate CPU processes")
    benchmark.add_argument("--manifest", required=True, type=Path)
    benchmark.add_argument("--fixture", required=True, type=Path)
    benchmark.add_argument("--output-dir", required=True, type=Path)
    benchmark.add_argument("--repeats", type=int, default=3)
    benchmark.add_argument("--generate-embeddings", action="store_true", help="Include an empty-cache ESMC generation measurement")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "catalog":
            from .catalog import release_catalog
            write_json(args.output, release_catalog())
            result = {"status": "complete", "manifest": str(args.output), "note": "Bind existing artifacts before use."}
        elif args.command == "bind":
            from .inventory import bind_catalog
            result = bind_catalog(args.run_root, args.fold_membership, args.output, candidate=args.candidate)
        elif args.command == "audit":
            from .audit import audit
            result = audit(load_manifest(args.manifest), args.output_dir,
                           bootstrap_samples=args.bootstrap_samples, seed=args.seed,
                           similarity_margin_pp=args.similarity_margin_pp, recall_floor_pp=args.recall_floor_pp)
        elif args.command == "nested-selection":
            from .selection import run_nested_selection
            result = run_nested_selection(args.manifest, args.run_root, args.output,
                                          expected_families=args.expect_family,
                                          late_fusion_candidate=args.late_fusion_candidate,
                                          bootstrap_samples=args.bootstrap_samples, seed=args.seed)
        elif args.command == "parity":
            from .parity import run_development_parity
            result = run_development_parity(args.manifest, args.fixture, args.output_dir)
        elif args.command == "predict":
            from .predict import predict_sites, predict_structure
            if args.output.exists():
                raise ReleaseError(f"Output already exists: {args.output}")
            kwargs = {"candidate": args.candidate, "embedding_cache": args.embedding_cache,
                      "generate_embeddings": args.generate_embeddings, "device": args.device,
                      "batch_size": args.batch_size, "include_member_predictions": args.member_predictions}
            if args.sites:
                sites = json.loads(args.sites.read_text(encoding="utf-8"))
                result = predict_sites(args.manifest, args.structure, sites=sites, **kwargs)
            else:
                result = predict_structure(args.manifest, args.structure, center=args.center, atom=args.atom, **kwargs)
            write_json(args.output, result)
        else:
            from .benchmark import benchmark
            result = benchmark(args.manifest, args.fixture, args.output_dir, repeats=args.repeats,
                               generate_embeddings=args.generate_embeddings)
        status = result.get("status", result.get("release_status", "complete"))
        print(json.dumps({"status": status,
                          "reference": result.get("reference"), "command": args.command}, sort_keys=True))
        return 2 if status == "blocked" else 0
    except (ReleaseError, OSError, ValueError) as exc:
        print(f"metal-release: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
