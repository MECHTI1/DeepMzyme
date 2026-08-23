from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPO_ROOT
    / "prepare_training_and_test_set"
    / "step6_create_additional_split_non_overalpped_structures.py"
)
SPEC = importlib.util.spec_from_file_location("deepmzyme_nonoverlap_builder", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BUILDER
SPEC.loader.exec_module(BUILDER)


PRIMARY_CSV = "final_data_summarazing_table_transition_metals_only_catalytic.csv"


def _write_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    payload = "structure,chain_resi,metaltype,ecnumber,whether_catalytic\n"
    payload += "".join(f"{structure},A_1,{metal},1.1.1.1,True\n" for structure, metal in rows)
    path.write_text(payload, encoding="utf-8")


def _synthetic_exact_root(tmp_path: Path) -> tuple[Path, Path, Path]:
    exact_root = tmp_path / "train_and_test_sets_structures_exact_pinmymetal"
    train_dir = exact_root / "train"
    test_dir = exact_root / "test"
    train_dir.mkdir(parents=True)
    test_dir.mkdir()
    (train_dir / "1abc__chain_A__EC_1.1.1.1.pdb").write_bytes(b"shared-train")
    (train_dir / "2def__chain_A__EC_1.1.1.1.pdb").write_bytes(b"clean-train")
    (test_dir / "1abc__chain_B__EC_1.1.1.1.pdb").write_bytes(b"shared-test")
    (test_dir / "3ghi__chain_A__EC_1.1.1.1.pdb").write_bytes(b"test-only")
    _write_csv(train_dir / PRIMARY_CSV, [("1abc__chain_A", "ZN"), ("2def__chain_A", "FE")])
    _write_csv(test_dir / PRIMARY_CSV, [("1abc__chain_B", "ZN"), ("3ghi__chain_A", "CU")])
    (exact_root / "split_metadata.json").write_text(
        json.dumps({"split_type": "metal_split_pinmymetal_possibly_overlapped"}) + "\n",
        encoding="utf-8",
    )
    return exact_root, train_dir, test_dir


def test_nonoverlap_builder_is_transactional_and_preserves_test_bytes(tmp_path: Path) -> None:
    _exact_root, train_dir, test_dir = _synthetic_exact_root(tmp_path)
    output_dir = tmp_path / "train_and_test_sets_structures_non_overlapped_pinmymetal"
    source_test_bytes = {path.name: path.read_bytes() for path in test_dir.iterdir() if path.is_file()}
    metadata = BUILDER.build_split(
        train_dir=train_dir,
        test_dir=test_dir,
        output_dir=output_dir,
        enforce_current_exact_profile=False,
    )
    assert sorted(path.name for path in (output_dir / "train").glob("*.pdb")) == [
        "2def__chain_A__EC_1.1.1.1.pdb"
    ]
    assert {path.name: path.read_bytes() for path in (output_dir / "test").iterdir() if path.is_file()} == source_test_bytes
    assert metadata["counts"]["output_overlap_pdbids"] == 0
    assert metadata["validation"]["test_tree_byte_identical_to_exact"] is True
    assert (output_dir / "removed_exact_test_pdbids_from_train.txt").read_text(encoding="utf-8") == "1abc\n"
    train_csv = (output_dir / "train" / PRIMARY_CSV).read_text(encoding="utf-8")
    assert "1abc" not in train_csv
    assert "2def" in train_csv

    with pytest.raises(FileExistsError, match="Refusing to modify"):
        BUILDER.build_split(
            train_dir=train_dir,
            test_dir=test_dir,
            output_dir=output_dir,
            enforce_current_exact_profile=False,
        )


def test_nonoverlap_builder_fails_closed_on_ambiguous_structure_filename(tmp_path: Path) -> None:
    _exact_root, train_dir, test_dir = _synthetic_exact_root(tmp_path)
    (train_dir / "unknown_structure.pdb").write_bytes(b"ambiguous")
    output_dir = tmp_path / "train_and_test_sets_structures_non_overlapped_pinmymetal"
    with pytest.raises(ValueError, match="Refusing to skip ambiguous structures"):
        BUILDER.build_split(
            train_dir=train_dir,
            test_dir=test_dir,
            output_dir=output_dir,
            enforce_current_exact_profile=False,
        )
    assert not output_dir.exists()


def test_nonoverlap_dry_run_writes_nothing(tmp_path: Path) -> None:
    _exact_root, train_dir, test_dir = _synthetic_exact_root(tmp_path)
    output_dir = tmp_path / "train_and_test_sets_structures_non_overlapped_pinmymetal"
    metadata = BUILDER.build_split(
        train_dir=train_dir,
        test_dir=test_dir,
        output_dir=output_dir,
        dry_run=True,
        enforce_current_exact_profile=False,
    )
    assert metadata["counts"]["source_overlap_pdbids"] == 1
    assert not output_dir.exists()
