"""Frozen EC fold membership: schema, writer/reader and split (synthetic; no dataset access)."""
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training.ec_membership import (
    EcMembershipError, ec_membership_split, membership_identity_sha256, read_ec_fold_membership,
    write_ec_fold_membership,
)


def _rows(n=6, with_class=False):
    rows = []
    for index in range(n):
        row = {"structure_id": f"P{index}__chain_A__EC_3.1.1.{index}", "accession": f"P{index}",
               "group_id": f"g{index // 2}", "fold": str((index // 2) % 3)}
        if with_class:
            row["ec_class"] = "3"
        rows.append(row)
    return rows


def _pockets(structure_ids, per_structure=1):
    return [SimpleNamespace(structure_id=sid, pocket_id=f"{sid}_METAL_{k}", metadata={})
            for sid in structure_ids for k in range(per_structure)]


def test_write_read_round_trip_is_deterministic(tmp_path):
    rows = _rows(with_class=True)
    first = write_ec_fold_membership(rows, tmp_path / "a.csv")
    second = write_ec_fold_membership(list(reversed(rows)), tmp_path / "b.csv")
    assert first == second
    assert b"\r\n" not in (tmp_path / "a.csv").read_bytes()
    membership = read_ec_fold_membership(tmp_path / "a.csv", first)
    assert membership.file_sha256 == first
    assert membership.identity_sha256 == membership_identity_sha256(membership.rows)
    assert membership.fold_by_structure["P5__chain_A__EC_3.1.1.5"] == 2


def test_read_rejects_wrong_hash_and_columns(tmp_path):
    sha = write_ec_fold_membership(_rows(), tmp_path / "m.csv")
    with pytest.raises(EcMembershipError, match="SHA-256"):
        read_ec_fold_membership(tmp_path / "m.csv", "0" * 64)
    (tmp_path / "x.csv").write_text("structure_id,accession,fold\nP1__chain_A__EC_3.1.1.1,P1,0\n")
    from training.ec_membership import _sha256_bytes
    with pytest.raises(EcMembershipError, match="missing columns"):
        read_ec_fold_membership(tmp_path / "x.csv", _sha256_bytes((tmp_path / "x.csv").read_bytes()))
    assert sha


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows.append(dict(rows[0])), "repeated structure_id"),
        (lambda rows: rows.append(dict(rows[0], structure_id="P0__chain_B__EC_3.1.1.0")), "repeated accession"),
        (lambda rows: rows[0].update(accession="Q9"), "does not match"),
        (lambda rows: rows[0].update(group_id=""), "empty group_id"),
        (lambda rows: rows[0].update(fold="x"), "not an integer"),
        (lambda rows: rows[0].update(fold="-1"), "non-negative"),
        (lambda rows: rows[1].update(fold="2"), "more than one fold"),
        (lambda rows: rows.clear(), "at least one row"),
    ],
)
def test_validation_refusals(tmp_path, mutate, message):
    rows = _rows()
    mutate(rows)
    with pytest.raises(EcMembershipError, match=message):
        write_ec_fold_membership(rows, tmp_path / "m.csv")


def test_ec_class_column_is_checked(tmp_path):
    rows = _rows(with_class=True)
    rows[0]["ec_class"] = "8"
    with pytest.raises(EcMembershipError, match="outside 1-7"):
        write_ec_fold_membership(rows, tmp_path / "m.csv")
    rows = _rows(with_class=True)
    del rows[0]["ec_class"]
    with pytest.raises(EcMembershipError, match="every row or no row"):
        write_ec_fold_membership(rows, tmp_path / "m.csv")


def test_split_follows_the_membership_and_keeps_order(tmp_path):
    rows = _rows()
    membership = read_ec_fold_membership(tmp_path / "m.csv", write_ec_fold_membership(rows, tmp_path / "m.csv"))
    ids = [row["structure_id"] for row in rows]
    pockets = _pockets(list(reversed(ids)), per_structure=2)
    split, receipt = ec_membership_split(pockets, membership, fold_index=1, n_folds=3)
    assert [p.structure_id for p in split.val_pockets] == [ids[3], ids[3], ids[2], ids[2]]
    assert {p.structure_id for p in split.train_pockets} == set(ids) - {ids[2], ids[3]}
    assert receipt["split_type"] == "frozen_ec_membership"
    assert (receipt["n_val_structures"], receipt["n_val_pockets"]) == (2, 4)
    assert receipt["membership_file_sha256"] == membership.file_sha256


def test_split_never_falls_back(tmp_path, monkeypatch):
    import training.splits as splits

    def refuse(*args, **kwargs):
        raise AssertionError("a computed split was attempted")

    monkeypatch.setattr(splits, "split_pockets_k_fold", refuse, raising=False)
    monkeypatch.setattr(splits, "split_pockets", refuse, raising=False)
    rows = _rows()
    membership = read_ec_fold_membership(tmp_path / "m.csv", write_ec_fold_membership(rows, tmp_path / "m.csv"))
    ids = [row["structure_id"] for row in rows]
    with pytest.raises(EcMembershipError, match="not loaded"):
        ec_membership_split(_pockets(ids[:-1]), membership, fold_index=0, n_folds=3)
    with pytest.raises(EcMembershipError, match="not in the EC membership"):
        ec_membership_split(_pockets(ids + ["Z1__chain_A__EC_1.1.1.1"]), membership, fold_index=0, n_folds=3)
    with pytest.raises(EcMembershipError, match="outside range"):
        ec_membership_split(_pockets(ids), membership, fold_index=0, n_folds=2)
    with pytest.raises(EcMembershipError, match="no members"):
        ec_membership_split(_pockets(ids), membership, fold_index=4, n_folds=5)


def test_schema_part_does_not_import_torch():
    src = Path(__file__).resolve().parents[1] / "src"
    code = f"import sys; sys.path.insert(0, {str(src)!r}); import training.ec_membership; print('torch' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip() == "False"


def _membership_for(paths, tmp_path, n_folds=2):
    rows = []
    for index, (accession, stem) in enumerate(sorted(paths["stems"].items())):
        rows.append({"structure_id": stem, "accession": accession, "group_id": accession,
                     "fold": str(index % n_folds), "ec_class": stem.split("__EC_")[1].split(".")[0]})
    path = tmp_path / "membership.csv"
    return path, write_ec_fold_membership(rows, path)


def test_config_ec_membership_mode_and_refusals(tmp_path):
    from training.config import parse_args

    base = ["--task", "ec", "--fold-split-source", "membership", "--fold-membership-csv", "m.csv",
            "--fold-membership-sha256", "0" * 64, "--n-folds", "2", "--fold-index", "0",
            "--no-prepare-missing-esm-embeddings"]
    config = parse_args(base + ["--no-prepare-missing-external-features"])
    assert config.fold_split_source == "membership" and config.prepare_missing_external_features is False
    assert parse_args(["--task", "ec"]).prepare_missing_external_features is True
    for argv in (
        [a for a in base if a != "--no-prepare-missing-esm-embeddings"],
        base[:-5] + ["--fold-index", "0", "--no-prepare-missing-esm-embeddings"],
        base + ["--run-test-eval"],
        base + ["--test-structure-dir", "x"],
    ):
        with pytest.raises(SystemExit):
            parse_args(argv)


def test_runtime_preparation_refuses_to_generate_external_features(tmp_path, monkeypatch):
    import training.runtime_preparation as prep

    (tmp_path / "s").mkdir()
    (tmp_path / "s" / "P1__chain_A__EC_3.1.1.1.pdb").write_text("END\n")
    monkeypatch.setattr(prep, "discover_missing_updated_external_features", lambda files, **kw: list(files))

    def must_not_generate(*args, **kwargs):
        raise AssertionError("generation was attempted")

    monkeypatch.setattr(prep, "_generate_missing_updated_external_features", must_not_generate)
    with pytest.raises(ValueError, match="generation is disabled"):
        prep.prepare_runtime_inputs(
            structure_dir=tmp_path / "s", esm_embeddings_dir=tmp_path / "esm", require_esm_embeddings=False,
            prepare_missing_esm_embeddings=False, require_ring_edges=False, prepare_missing_ring_edges=False,
            external_features_root_dir=tmp_path / "ext", prepare_missing_external_features=False,
        )


def test_tiny_ec_fit_with_fixed_classes_and_frozen_membership(tmp_path):
    import json

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ec_fixtures import default_ec_proteins, tiny_ec_argv, write_ec_dataset
    from training.config import parse_args
    from training.run import run_training

    paths = write_ec_dataset(tmp_path / "data", default_ec_proteins())
    membership, sha = _membership_for(paths, tmp_path)
    argv = tiny_ec_argv(paths, tmp_path / "runs",
                        "--ec-class-vocabulary", "fixed_seven", "--ec-conflicting-labels", "fail",
                        "--fold-split-source", "membership", "--fold-membership-csv", str(membership),
                        "--fold-membership-sha256", sha, "--n-folds", "2", "--fold-index", "1",
                        "--checkpoint-rule", "terminal", "--lr-schedule", "cosine",
                        "--no-prepare-missing-external-features", "--run-name", "ec_tiny")
    run_dir = run_training(parse_args(argv))
    run_config = json.loads((run_dir / "run_config.json").read_text())
    summary = run_config["dataset_summary"]
    assert summary["fold_membership"]["split_type"] == "frozen_ec_membership"
    assert summary["fold_membership"]["membership_file_sha256"] == sha
    support = summary["split_diagnostics"]["ec_class_support"]
    assert support["ec_classes_absent_from_training"] == ["4", "5", "6", "7"]
    assert run_config["ec_labels"] == {str(i): str(i + 1) for i in range(7)}
    last = run_config["history"][-1]
    assert last["val_ec_group_ec_classes_1_6_n_present"] == 3
    assert 0.0 <= last["val_ec_group_ec_classes_1_6_balanced_acc"] <= 1.0
    groups = json.loads((run_dir / "optimizer_groups.json").read_text())
    assert groups["realised_steps_match"] and groups["split_reason"] == []
