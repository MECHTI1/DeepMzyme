"""Frozen scorer for the EC class versus CLEAN comparison on the official CLEAN split30 benchmark.

Subcommands:
- claim --manifest M: hashes every frozen prediction input into <report_dir>/prediction_claim.json
  (write-once) without opening the label table, and prints the claim SHA-256;
- score --claim C --claim-sha256 S: verifies the claim, reads the label table once and writes
  report.json and report.md (write-once).

See src/ec_clean30/scoring.py for the input contracts and the pre-declared analysis.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ec_clean30.scoring import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
