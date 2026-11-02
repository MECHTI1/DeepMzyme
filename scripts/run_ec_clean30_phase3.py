#!/usr/bin/env python3
"""EC CLEAN30 Phase-3 preparation and bounded timing entry point."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from ec_clean30.phase3 import main


if __name__ == "__main__":
    raise SystemExit(main())
