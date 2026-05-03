import os
import sys
from pathlib import Path
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import PROJECT_ROOT as REPO_ROOT, resolve_ring_features_dir


DEFAULT_RING_EXE = Path("DeepMzyme_Data") / "ring-4.0" / "out" / "bin" / "ring"
DIR_RESULTS = resolve_ring_features_dir(
    os.getenv("RING_FEATURES_DIR") or os.getenv("RING_EDGES_DIR") or os.getenv("EMBEDDINGS_DIR")
)


def resolve_ring_path(path: str | Path) -> Path:
    ring_path = Path(path).expanduser()
    return ring_path if ring_path.is_absolute() else REPO_ROOT / ring_path


def resolve_ring_executable() -> Path:
    configured_path = (os.getenv("RING_EXE_PATH") or "").strip()
    ring_exe = resolve_ring_path(configured_path if configured_path else DEFAULT_RING_EXE)
    if not ring_exe.is_file():
        raise FileNotFoundError(
            f"RING executable not found: {ring_exe}. "
            "Set RING_EXE_PATH to the RING binary path before requesting RING edge generation."
        )
    if not os.access(ring_exe, os.X_OK):
        raise PermissionError(
            f"RING executable is not executable: {ring_exe}. "
            f"Run: chmod +x {ring_exe}"
        )
    return ring_exe


def expected_ring_edges_path(dir_results, path_structure) -> Path:
    dir_results = Path(dir_results)
    path_structure = Path(path_structure)
    return dir_results / path_structure.stem / f"{path_structure.name}_ringEdges"


def ring_create_results(dir_results, path_structure):
    dir_results = Path(dir_results)
    path_structure = Path(path_structure)
    ring_exe = resolve_ring_executable()

    dir_results.mkdir(parents=True, exist_ok=True)

    dir_structure_results = dir_results / path_structure.stem
    dir_structure_results.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            str(ring_exe),
            "-i",
            str(path_structure),
            "--out_dir",
            str(dir_structure_results),
        ],
        check=True
    )
    return expected_ring_edges_path(dir_results, path_structure)


def create_ring_edges_batch(
    structure_files: Sequence[str | Path],
    dir_results: str | Path | None = None,
    *,
    overwrite: bool = False,
    jobs: int = 1,
) -> dict[str, object]:
    if dir_results is None:
        dir_results = DIR_RESULTS
    dir_results = Path(dir_results)
    dir_results.mkdir(parents=True, exist_ok=True)

    resolve_ring_executable()
    if jobs < 1:
        raise ValueError(f"jobs must be at least 1, got {jobs}")

    processed = 0
    failed: list[dict[str, str]] = []
    saved_files: list[str] = []
    pending_structure_files: list[Path] = []

    for structure_file in structure_files:
        structure_path = Path(structure_file)
        out_file = expected_ring_edges_path(dir_results, structure_path)
        if out_file.is_file() and not overwrite:
            saved_files.append(str(out_file))
            continue
        pending_structure_files.append(structure_path)

    if jobs == 1:
        for structure_path in pending_structure_files:
            try:
                saved_path = ring_create_results(dir_results, structure_path)
                processed += 1
                saved_files.append(str(saved_path))
            except Exception as exc:
                failed.append(
                    {
                        "structure_file": str(structure_path),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
    else:
        max_workers = min(jobs, len(pending_structure_files))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_structure_path = {
                executor.submit(ring_create_results, dir_results, structure_path): structure_path
                for structure_path in pending_structure_files
            }
            for future in as_completed(future_to_structure_path):
                structure_path = future_to_structure_path[future]
                try:
                    saved_path = future.result()
                    processed += 1
                    saved_files.append(str(saved_path))
                except Exception as exc:
                    failed.append(
                        {
                            "structure_file": str(structure_path),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )

    return {
        "processed_structures": processed,
        "failed_structures": failed,
        "saved_files": saved_files,
        "out_dir": str(dir_results),
    }


if __name__ == "__main__":
    # path_test_structure = "/home/mechti/Downloads/AF-P06965-F1-model_v6.cif"
    path_test_structure = "/media/Data/pinmymetal_sets/mahomes/train_set/job_0/1a0e__chain_A__EC_5.3.1.5.pdb"
    ring_create_results(DIR_RESULTS, path_test_structure)
