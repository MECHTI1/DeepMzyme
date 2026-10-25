"""MMseqs2 nearest-hit baseline of the EC class vs CLEAN30 campaign (synthetic files only; no dataset access)."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ec_clean30 import baseline  # noqa: E402
from ec_clean30.baseline import BaselineError, build_baseline  # noqa: E402
from ec_clean30.identity import STRATA_COLUMNS, sha256_file  # noqa: E402

FREEZE1 = "a" * 64
SEQUENCE = "ACDEFGHIKLMNPQRSTVWY" * 3
# Fold 0: a one-class hit, no hit, a hit with two EC classes, a hit without a usable EC class.
NEAREST = {0: {"Q00001": "T00001", "Q00002": "", "Q00003": "T00002", "Q00004": "T00003", "Q00009": "T00001"},
           1: {"Q00005": "T00004"}}
TRAINING = {0: [("T00001", "3.4.24.17"), ("T00002", "1.1.1.1;2.7.1.1"), ("T00003", "-"), ("T00009", "4.2.1.1")],
            1: [("T00004", "7.1.1.1"), ("T00004", "7.1.1.1")]}
MEMBERSHIP = [(0, "Q00001"), (0, "Q00002"), (0, "Q00003"), (0, "Q00004"), (1, "Q00005")]


def write_strata(work: Path, fold: int, nearest: dict, freeze1: str = FREEZE1) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    path = work / f"strata_fold{fold}.tsv"
    rows = []
    for accession, target in sorted(nearest.items()):
        row = {column: "0" for column in STRATA_COLUMNS}
        row.update(accession=accession, identity_bin="no_hit" if not target else "[30,50]", max_bin_identity="",
                   max_bin_target=target, nearest_target=target, nearest_bits="50" if target else "",
                   nearest_evalue="1.000E-10" if target else "", cluster_id=f"K_{accession}")
        rows.append(row)
    path.write_text("\n".join(["\t".join(STRATA_COLUMNS)] + ["\t".join(row[c] for c in STRATA_COLUMNS) for row in rows])
                    + "\n")
    receipt = {"step": "strata", "fold": fold, "freeze1_sha256": freeze1,
               "nearest_hit_order": "highest bits, then lowest E-value, then target accession",
               "outputs": {path.name: {"sha256": sha256_file(path), "n_rows": len(rows)}}}
    (work / "strata_receipt.json").write_text(json.dumps(receipt))
    return path


@pytest.fixture()
def campaign(tmp_path):
    split = tmp_path / "split30"
    split.mkdir()
    for fold, rows in TRAINING.items():
        lines = ["Entry\tEC number\tSequence"] + [f"{accession}\t{ec}\t{SEQUENCE}" for accession, ec in rows]
        (split / f"split30_train_split_{fold}.csv").write_text("\n".join(lines) + "\n")
    strata = {fold: write_strata(tmp_path / f"identity_fold{fold}", fold, nearest) for fold, nearest in NEAREST.items()}
    membership = tmp_path / "phase7" / "test_membership.csv"
    membership.parent.mkdir()
    membership.write_text("fold,accession\n" + "".join(f"{fold},{accession}\n" for fold, accession in MEMBERSHIP))
    freeze2 = tmp_path / "phase7" / "freeze2_manifest.json"
    freeze2.write_text(json.dumps({"schema_version": 1}))
    return {"membership": membership, "strata": strata, "official_split_dir": split,
            "out": tmp_path / "phase7" / "baseline" / "baseline_classes.csv", "freeze1_sha256": FREEZE1,
            "freeze2_manifest": freeze2, "freeze2_sha256": sha256_file(freeze2)}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_statuses_follow_the_nearest_training_annotation(campaign):
    result = build_baseline(**campaign)
    rows = read_rows(campaign["out"])
    assert [tuple(row.values()) for row in rows] == [
        ("0", "Q00001", "3", "predicted"), ("0", "Q00002", "", "no_hit"), ("0", "Q00003", "", "tie"),
        ("0", "Q00004", "", "failure"), ("1", "Q00005", "7", "predicted")]
    assert result["counts"] == {"0": {"predicted": 1, "no_hit": 1, "tie": 1, "failure": 1},
                                "1": {"predicted": 1, "no_hit": 0, "tie": 0, "failure": 0}}
    receipt = json.loads((campaign["out"].parent / baseline.RECEIPT_NAME).read_text())
    assert receipt["test_labels_read"] is False and receipt["labels_read"] == "official_training_files_only"
    assert receipt["freeze2_manifest"]["sha256"] == campaign["freeze2_sha256"]
    assert receipt["outputs"]["baseline_classes.csv"]["sha256"] == sha256_file(campaign["out"])
    assert receipt["inputs"]["folds"]["0"]["official_training_file"] == "split30_train_split_0.csv"
    assert set(receipt["code_sha256"]) >= {"src/ec_clean30/baseline.py", "scripts/run_ec_clean30_baseline.py"}


def test_the_scorer_reads_the_class_table(campaign):
    from ec_clean30 import scoring

    build_baseline(**campaign)
    membership = {0: ["Q00001", "Q00002", "Q00003", "Q00004"], 1: ["Q00005"]}
    predictions = scoring.read_class_table(campaign["out"], membership, "baseline")
    assert predictions[(0, "Q00001")] == scoring.Prediction("3", "predicted")
    assert predictions[(0, "Q00003")] == scoring.Prediction(None, "tie")


def test_outputs_are_written_once(campaign):
    build_baseline(**campaign)
    with pytest.raises(BaselineError, match="existing baseline"):
        build_baseline(**campaign)


@pytest.mark.parametrize("change, message", [
    ({"freeze2_sha256": "b" * 64}, "does not hash"),
    ({"freeze2_sha256": "B" * 64}, "64 lowercase"),
    ({"freeze1_sha256": "c" * 64}, "another Freeze 1"),
    ({"freeze1_sha256": None}, "locked until Freeze 1"),
])
def test_freeze_bindings_are_checked(campaign, change, message):
    with pytest.raises(ValueError, match=message):
        build_baseline(**{**campaign, **change})
    assert not campaign["out"].exists()


def test_a_changed_strata_table_is_refused(campaign):
    path = campaign["strata"][0]
    path.write_text(path.read_text().replace("T00002", "T00001"))
    with pytest.raises(BaselineError, match="changed after the strata step"):
        build_baseline(**campaign)


def test_membership_and_target_contract(campaign, tmp_path):
    campaign["membership"].write_text("fold,accession\n0,Q00001\n0,Q00077\n1,Q00005\n")
    with pytest.raises(BaselineError, match="Q00077 has no strata row"):
        build_baseline(**campaign)
    campaign["membership"].write_text("fold,accession\n0,Q00001\n")
    with pytest.raises(BaselineError, match="Strata are given for folds"):
        build_baseline(**campaign)
    write_strata(tmp_path / "other_fold1", 1, {"Q00005": "T09999"})
    campaign["membership"].write_text("fold,accession\n0,Q00001\n1,Q00005\n")
    with pytest.raises(BaselineError, match="T09999 of Q00005 is not in the fold-1 official training file"):
        build_baseline(**{**campaign, "strata": {**campaign["strata"], 1: tmp_path / "other_fold1" /
                                                 "strata_fold1.tsv"}})


def test_conflicting_training_annotations_are_refused(campaign):
    path = campaign["official_split_dir"] / "split30_train_split_1.csv"
    path.write_text(path.read_text().replace("T00004\t7.1.1.1", "T00004\t7.1.1.2", 1))
    with pytest.raises(BaselineError, match="two EC annotations"):
        build_baseline(**campaign)


def test_held_out_directories_are_refused(campaign, tmp_path):
    held_out = tmp_path / "test"
    held_out.mkdir()
    moved = held_out / "test_membership.csv"
    moved.write_text(campaign["membership"].read_text())
    with pytest.raises(BaselineError, match="held-out directory"):
        build_baseline(**{**campaign, "membership": moved})


def test_cli_runs_and_refuses_a_repeated_fold(campaign):
    script = ROOT / "scripts" / "run_ec_clean30_baseline.py"
    argv = [sys.executable, str(script), "--membership", str(campaign["membership"]),
            "--official-split-dir", str(campaign["official_split_dir"]), "--out", str(campaign["out"]),
            "--freeze1-sha256", FREEZE1, "--freeze2-manifest", str(campaign["freeze2_manifest"]),
            "--freeze2-sha256", campaign["freeze2_sha256"]]
    strata = [arg for fold, path in campaign["strata"].items() for arg in ("--strata", f"{fold}={path}")]
    repeated = subprocess.run(argv + strata + ["--strata", f"0={campaign['strata'][0]}"], capture_output=True,
                              text=True)
    assert repeated.returncode == 2 and "given twice" in repeated.stderr
    completed = subprocess.run(argv + strata, capture_output=True, text=True, check=True)
    assert json.loads(completed.stdout)["outputs"]["baseline_classes.csv"]["n_rows"] == 5


def test_module_imports_without_torch():
    code = (f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r}); import ec_clean30.baseline; "
            "print('torch' in sys.modules)")
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert completed.stdout.strip() == "False"
