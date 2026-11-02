#!/usr/bin/env python
"""Run the frozen Fold-0 S1 and CLEAN-inner execution package."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from ec_clean30.phase4 import main

if __name__ == "__main__":
    raise SystemExit(main())
