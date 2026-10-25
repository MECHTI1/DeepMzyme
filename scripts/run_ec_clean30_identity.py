"""Identity tools for the EC class vs CLEAN30 campaign (search, edges, components, strata, inner folds).

Subcommands: fasta, search, edges, components, strata, assign-inner,
certify-inner and r1-inner-exclusions. Run each step with the same --fold and
--work. Complement queries, cohort clusters and strata need --freeze1-sha256;
the inner-fold steps do not. See src/ec_clean30/identity.py for the rules.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ec_clean30.identity import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
