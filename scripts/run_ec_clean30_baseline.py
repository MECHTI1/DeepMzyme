"""MMseqs2 nearest-hit baseline for the EC class vs CLEAN30 campaign (Phase 7 only).

Writes the baseline class table (fold, accession, predicted_ec_class, status) from each fold's
strata table and official training file. Needs --freeze1-sha256 and the Freeze-2 manifest with its
SHA-256. See src/ec_clean30/baseline.py for the rules.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ec_clean30.baseline import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
