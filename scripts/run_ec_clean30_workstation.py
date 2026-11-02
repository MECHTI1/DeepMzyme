#!/usr/bin/env python3
"""Entry point for the bounded workstation G2 driver."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from ec_clean30.workstation import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
