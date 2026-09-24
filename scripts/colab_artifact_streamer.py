#!/usr/bin/env python3
"""Incrementally mirror a Colab run directory to local disk while a job runs.

Colab VMs are reclaimed without warning, and the Colab CLI keep-alive daemon runs on
the *local* workstation -- so a local reboot reaps the VM too. Anything that exists
only under ``/content`` is lost at that moment. On 2026-09-24 that cost a full
Zenodo PinMyMetal fold-0 run (see
``docs/agents_report/HANDOFF_ZENODO_PMM_EXACT_BENCHMARK.md`` section 7).

This streamer polls the remote run directory and copies new or changed files down as
they appear, so an interrupted campaign loses at most one poll interval of work.

Example:
    python3 scripts/colab_artifact_streamer.py \\
        --session pmm-zenodo \\
        --remote-dir /content/runs/runs_zenodo_pmm_exact \\
        --local-dir ~/zenodo_pmm_artifacts \\
        --interval 300

It is read-only with respect to the VM: it lists and downloads, never deletes.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

DEFAULT_COLAB_BIN = "/home/mechti/.local/bin/colab"

# Small, always-worth-having outputs. Checkpoints are handled separately because they
# are large and the local root filesystem is routinely short on space.
DEFAULT_PATTERNS = (
    ".json",
    ".csv",
    ".md",
    ".log",
    ".txt",
)

CHECKPOINT_SUFFIX = ".pt"


def log(message: str) -> None:
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def run_colab(colab_bin: str, args: list[str], *, stdin_text: str | None = None, timeout: int = 300):
    """Run a Colab CLI command, returning (returncode, stdout, stderr)."""
    try:
        proc = subprocess.run(
            [colab_bin, *args],
            input=stdin_text,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout}s"
    except OSError as exc:  # colab binary missing, etc.
        return 127, "", str(exc)


LISTING_SNIPPET = """
import json, os
root = {remote_dir!r}
entries = {{}}
if os.path.isdir(root):
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            full = os.path.join(dirpath, name)
            try:
                stat = os.stat(full)
            except OSError:
                continue
            entries[os.path.relpath(full, root)] = [stat.st_size, int(stat.st_mtime)]
print("STREAMER_LISTING_BEGIN")
print(json.dumps(entries))
print("STREAMER_LISTING_END")
"""


def list_remote_files(colab_bin: str, session: str, remote_dir: str) -> dict[str, list[int]] | None:
    """Return {relative_path: [size, mtime]} for the remote directory, or None if unreachable."""
    code = LISTING_SNIPPET.format(remote_dir=remote_dir)
    rc, out, err = run_colab(colab_bin, ["exec", "-s", session], stdin_text=code)
    combined = f"{out}\n{err}"
    if "not found" in combined.lower() and "session" in combined.lower():
        log(f"Session {session!r} no longer exists on the backend.")
        return None
    if rc != 0:
        log(f"Listing failed (rc={rc}): {err.strip()[:300]}")
        return {}
    try:
        payload = combined.split("STREAMER_LISTING_BEGIN", 1)[1].split("STREAMER_LISTING_END", 1)[0]
        return json.loads(payload.strip())
    except (IndexError, ValueError) as exc:
        log(f"Could not parse remote listing: {exc}")
        return {}


def wanted(rel_path: str, patterns: tuple[str, ...], include_checkpoints: bool) -> bool:
    lowered = rel_path.lower()
    if lowered.endswith(CHECKPOINT_SUFFIX):
        return include_checkpoints
    return any(lowered.endswith(suffix) for suffix in patterns)


def select_checkpoints(entries: dict[str, list[int]], keep_per_run: int) -> set[str]:
    """Keep only the newest ``keep_per_run`` checkpoints per run directory.

    A 50-epoch run with --save-epoch-checkpoints writes 50 checkpoints; mirroring all
    of them would exhaust local disk for no benefit, since only the latest is needed
    to resume and the selected one is written at the end under its own name.
    """
    by_run: dict[str, list[tuple[int, str]]] = {}
    for rel, (_size, mtime) in entries.items():
        if not rel.lower().endswith(CHECKPOINT_SUFFIX):
            continue
        run_key = str(Path(rel).parent)
        by_run.setdefault(run_key, []).append((mtime, rel))

    selected: set[str] = set()
    for run_key, items in by_run.items():
        named = [rel for _mtime, rel in items if "epoch_" not in Path(rel).name]
        selected.update(named)
        epochwise = sorted(
            ((mtime, rel) for mtime, rel in items if "epoch_" in Path(rel).name),
            reverse=True,
        )
        selected.update(rel for _mtime, rel in epochwise[:keep_per_run])
    return selected


def download_file(colab_bin: str, session: str, remote_path: str, local_path: Path) -> bool:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    rc, _out, err = run_colab(
        colab_bin,
        ["download", "-s", session, remote_path, str(local_path)],
        timeout=900,
    )
    if rc != 0:
        log(f"  download failed for {remote_path}: {err.strip()[:200]}")
        return False
    return True


def free_bytes(path: Path) -> int:
    target = path
    while not target.exists():
        target = target.parent
    stats = os.statvfs(target)
    return stats.f_bavail * stats.f_frsize


def sync_once(
    *,
    colab_bin: str,
    session: str,
    remote_dir: str,
    local_dir: Path,
    state: dict[str, list[int]],
    patterns: tuple[str, ...],
    include_checkpoints: bool,
    keep_checkpoints: int,
    min_free_gb: float,
) -> bool:
    """Mirror one pass. Returns False when the session is gone and polling should stop."""
    entries = list_remote_files(colab_bin, session, remote_dir)
    if entries is None:
        return False
    if not entries:
        log("Remote run directory is empty or unreadable; nothing to mirror yet.")
        return True

    checkpoint_allowlist = (
        select_checkpoints(entries, keep_checkpoints) if include_checkpoints else set()
    )

    pending: list[str] = []
    for rel, meta in sorted(entries.items()):
        if not wanted(rel, patterns, include_checkpoints):
            continue
        if rel.lower().endswith(CHECKPOINT_SUFFIX) and rel not in checkpoint_allowlist:
            continue
        if state.get(rel) == meta and (local_dir / rel).exists():
            continue
        pending.append(rel)

    if not pending:
        log(f"Up to date ({len(entries)} remote files, nothing changed).")
        return True

    log(f"{len(pending)} file(s) new or changed; downloading.")
    for rel in pending:
        available_gb = free_bytes(local_dir) / 1e9
        if available_gb < min_free_gb:
            log(
                f"  STOPPING downloads: only {available_gb:.1f} GB free at {local_dir}, "
                f"below the --min-free-gb floor of {min_free_gb:.1f} GB."
            )
            break
        remote_path = f"{remote_dir.rstrip('/')}/{rel}"
        if download_file(colab_bin, session, remote_path, local_dir / rel):
            state[rel] = entries[rel]
            log(f"  saved {rel}")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", required=True, help="Colab session name")
    parser.add_argument("--remote-dir", required=True, help="Run directory on the VM, e.g. /content/runs/...")
    parser.add_argument("--local-dir", required=True, type=Path, help="Local mirror destination")
    parser.add_argument("--interval", type=int, default=300, help="Seconds between polls (default: 300)")
    parser.add_argument("--max-polls", type=int, default=0, help="Stop after N polls (0 = until the session ends)")
    parser.add_argument("--colab-bin", default=DEFAULT_COLAB_BIN, help="Path to the colab CLI")
    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="Mirror only metrics/report files, never .pt weights",
    )
    parser.add_argument(
        "--keep-checkpoints",
        type=int,
        default=1,
        help="Newest per-epoch checkpoints to mirror per run directory (default: 1)",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=3.0,
        help="Pause downloading when local free space falls below this (default: 3.0)",
    )
    args = parser.parse_args(argv)

    local_dir: Path = args.local_dir.expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    state_path = local_dir / ".streamer_state.json"
    state: dict[str, list[int]] = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
        except ValueError:
            state = {}

    log(f"Mirroring {args.session}:{args.remote_dir} -> {local_dir} every {args.interval}s")
    log(f"Free space at destination: {free_bytes(local_dir) / 1e9:.1f} GB")

    polls = 0
    try:
        while True:
            polls += 1
            alive = sync_once(
                colab_bin=args.colab_bin,
                session=args.session,
                remote_dir=args.remote_dir,
                local_dir=local_dir,
                state=state,
                patterns=DEFAULT_PATTERNS,
                include_checkpoints=not args.no_checkpoints,
                keep_checkpoints=args.keep_checkpoints,
                min_free_gb=args.min_free_gb,
            )
            state_path.write_text(json.dumps(state, indent=2, sort_keys=True))
            if not alive:
                log("Session ended. Mirror holds everything downloaded so far.")
                return 0
            if args.max_polls and polls >= args.max_polls:
                log(f"Reached --max-polls={args.max_polls}; exiting.")
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        log("Interrupted; state saved.")
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True))
        return 130


if __name__ == "__main__":
    sys.exit(main())
