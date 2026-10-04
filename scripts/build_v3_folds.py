#!/usr/bin/env python3
"""Build the near-copy-disjoint v3 folds (plan step A1) from the frozen v2 training cohort.

Reads only training-side inputs: the v2 campaign's ``train_cohort.csv``,
``campaign_manifest.json`` and ``esm_generation_plan.csv``, and the cohort
structure files under ``--train-dir``. Held-out paths are blocked by the
project's read guard. Outputs go to a new directory and are never overwritten.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmarking import v3_folds  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-root", type=Path, required=True,
                        help="Frozen v2 campaign root (train_cohort.csv, campaign_manifest.json, esm_generation_plan.csv)")
    parser.add_argument("--train-dir", type=Path, required=True,
                        help="Training structure directory of the Zenodo PMM dataset (contains structure_manifest.csv)")
    parser.add_argument("--out-dir", type=Path, required=True, help="New output directory for the v3 fold set")
    parser.add_argument("--seed", type=int, default=v3_folds.DEFAULT_SEED)
    parser.add_argument("--starts", type=int, default=v3_folds.DEFAULT_STARTS)
    parser.add_argument("--workers", type=int, default=2, help="Structure-parsing processes")
    args = parser.parse_args(argv)

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error(f"{args.out_dir} is not empty; a fold set is never overwritten")
    started = time.time()
    loaded = v3_folds.load_inputs(args.v2_root.resolve(), args.train_dir.resolve(), workers=args.workers)
    print(f"[V3-FOLDS] inputs verified: {len(loaded['bindings'])} ions, "
          f"{len(loaded['pdb_sequences'])} PDB entries ({time.time() - started:.0f} s)", flush=True)
    result = v3_folds.build_folds(loaded["bindings"], loaded["pdb_sequences"], seed=args.seed,
                                  starts=args.starts, progress=True)
    receipt = v3_folds.write_outputs(args.out_dir, result, inputs=loaded["inputs"],
                                     seed=args.seed, starts=args.starts)
    report = json.loads((args.out_dir / "fold_balance_report.json").read_text(encoding="utf-8"))
    print(json.dumps({"accepted": receipt["accepted"], "summary": receipt["summary"],
                      "passing_starts": report["passing_starts"], "failures": report["failures"][:10],
                      "violations": report["violations"][:10]}, indent=2), flush=True)
    print(f"[V3-FOLDS] done in {time.time() - started:.0f} s; outputs in {args.out_dir}", flush=True)
    return 0 if receipt["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
