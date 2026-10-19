from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for entry in (str(ROOT), str(ROOT / "src"), str(ROOT / "tests"), str(ROOT / "tests" / "benchmarking")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from project_paths import install_relocated_module_aliases  # noqa: E402

install_relocated_module_aliases()
