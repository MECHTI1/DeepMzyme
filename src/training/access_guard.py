"""Process-wide read guard for development-only (validation) execution.

``install_forbidden_read_guard`` registers a ``sys.audit`` hook that raises
``PermissionError`` when this process opens, lists, scans or globs anything under a
forbidden root (for example a dataset's held-out ``test/`` directory or a mixed
train/test crosswalk). Child training processes inherit the roots through
``DEEPMZYME_FORBIDDEN_READ_ROOTS`` (``os.pathsep``-separated) and install the same
guard at start-up. Audit hooks cannot be removed, so the guard lasts for the life
of the process.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

FORBIDDEN_READ_ROOTS_ENV = "DEEPMZYME_FORBIDDEN_READ_ROOTS"
_GUARDED_EVENTS = {"open", "os.listdir", "os.scandir", "glob.glob", "os.chdir"}
_INSTALLED_ROOTS: list[str] = []


def _normalize(path: str) -> str:
    return os.path.normcase(os.path.realpath(path))


def _is_under(path: str, roots: list[str]) -> bool:
    return any(path == root or path.startswith(root.rstrip(os.sep) + os.sep) for root in roots)


def install_forbidden_read_guard(roots: Iterable[str | os.PathLike]) -> list[str]:
    """Deny reads under ``roots`` for the rest of this process; returns all guarded roots."""
    new_roots = [_normalize(os.fspath(root)) for root in roots if os.fspath(root)]
    if not new_roots:
        return list(_INSTALLED_ROOTS)
    first_install = not _INSTALLED_ROOTS
    _INSTALLED_ROOTS.extend(root for root in new_roots if root not in _INSTALLED_ROOTS)
    if first_install:
        def hook(event: str, args: tuple) -> None:
            if event not in _GUARDED_EVENTS or not args:
                return
            target = args[0]
            if isinstance(target, int) or target is None:
                return
            if isinstance(target, bytes):
                target = os.fsdecode(target)
            if not isinstance(target, (str, os.PathLike)):
                return
            if _is_under(_normalize(os.fspath(target)), _INSTALLED_ROOTS):
                raise PermissionError(f"Development-only process may not read held-out path: {target}")

        sys.addaudithook(hook)
    return list(_INSTALLED_ROOTS)


def install_guard_from_environment() -> list[str]:
    raw = os.environ.get(FORBIDDEN_READ_ROOTS_ENV, "")
    roots = [part for part in raw.split(os.pathsep) if part.strip()]
    return install_forbidden_read_guard(roots) if roots else []


def forbidden_roots_environment(roots: Iterable[str | os.PathLike]) -> str:
    return os.pathsep.join(str(Path(root).resolve()) for root in roots)
