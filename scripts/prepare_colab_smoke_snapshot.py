"""Package the current working files for Colab without changing Git or training."""
from __future__ import annotations

import hashlib
import argparse
import io
import json
from pathlib import Path
import subprocess
import tarfile


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="train_and_test_sets_structures_common_pdbid_70_30_pinmymetal")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "DeepMzyme_Data/notebook_outputs/plans/standalone_v1/colab")
    parser.add_argument("--archive-name", default="deepmzyme-chat4-code.tar.gz")
    args = parser.parse_args()
    if Path(args.dataset).name != args.dataset or Path(args.archive_name).name != args.archive_name:
        parser.error("dataset and archive-name must be simple names")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Deliberate allowlist: include new source files, exclude data, secrets and Git internals.
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z",
         "src", "requirements", "tests", "scripts", "notebooks", "docs", "Plan.md",
         "README.md", "EXPERIMENT_STATUS.md"], cwd=root,
    ).decode().split("\0")
    names = sorted({name for name in names if name and (root / name).is_file()
                    and not name.startswith(("docs/notebook_outputs/raw/", "docs/archive/"))})
    payloads = {name: (root / name).read_bytes() for name in names}
    # Portable membership/content check: the hosted bundle has legacy structure paths.
    import sys
    sys.path.insert(0, str(root / "src"))
    from structure_store import read_structure_manifest
    dataset = args.dataset
    train_dir = root / "DeepMzyme_Data" / dataset / "train"
    references = read_structure_manifest(train_dir)
    expected = {
        "dataset": dataset,
        "summary_sha256": hashlib.sha256((train_dir / "final_data_summarazing_table_transition_metals_only_catalytic.csv").read_bytes()).hexdigest(),
        "train_structures": {ref.structure_name: ref.sha256 for ref in references},
    }
    payloads["colab_expected_metal_inputs.json"] = json.dumps(expected, indent=2).encode()
    manifest = {
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "working_tree_status": subprocess.check_output(["git", "status", "--short"], cwd=root, text=True),
        "kind": "working_tree_snapshot_including_uncommitted_files_not_a_clean_commit",
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()},
    }
    payloads["code_snapshot_manifest.json"] = json.dumps(manifest, indent=2).encode()
    archive_path = output / args.archive_name
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, data in payloads.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    (output / (args.archive_name + ".sha256")).write_text(f"{digest}  {archive_path.name}\n")
    (output / "code_snapshot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(archive_path)
    print(f"SHA256: {digest}")
    print(f"Files: {len(payloads)}; bytes: {archive_path.stat().st_size}; Git worktree/index untouched.")


if __name__ == "__main__":
    main()
