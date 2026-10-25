"""Selected-site extraction through the structure loader, config and trainer (synthetic; no dataset access)."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ec_fixtures import default_ec_proteins, tiny_ec_argv, write_ec_dataset


def _mixed_dataset(root, *, with_unsummarised=True):
    proteins = default_ec_proteins(n_per_class=2, classes=("1", "3"))
    proteins["Q0001"] = {
        "ec": "3.4.1.1",
        "ions": [("A", 900, "", "ZN", [50.0, 0.0, 0.0]), ("A", 901, "", "FE", [53.0, 0.0, 0.0])],
        "summary_sites": [("A", 900, "ZN")],
    }
    paths = write_ec_dataset(root, proteins)
    if with_unsummarised:
        # A structure with no summary row at all (and no embeddings; only the loader sees it).
        from pmm_campaign_fixtures import write_pdb
        write_pdb(paths["structures"] / "Z9999__chain_A__EC_1.1.1.1.pdb", ions=[("A", 900, "", "ZN", [90.0, 0.0, 0.0])])
    return paths


def _load(paths, mode):
    from training.data import load_training_pockets_with_report_from_dir

    return load_training_pockets_with_report_from_dir(
        paths["structures"], required_targets=("ec",), summary_csv=paths["summary"],
        esm_embeddings_dir=paths["esm"], esm_dim=8, require_esm_embeddings=False,
        external_features_root_dir=paths["external"], require_external_features=False,
        load_workers=1, site_extraction=mode,
    )


def test_selected_mode_drops_unselected_metals_and_reports(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPMZYME_PARSE_CACHE_DIR", str(tmp_path / "cache"))
    paths = _mixed_dataset(tmp_path / "data")
    component = {p.structure_id: p for p in _load(paths, "component").pockets}
    selected_result = _load(paths, "selected_sites")
    selected = {p.structure_id: p for p in selected_result.pockets}
    mixed = "Q0001__chain_A__EC_3.4.1.1"
    assert component[mixed].metal_count() == 2
    assert selected[mixed].metal_count() == 1 and selected[mixed].metal_element == "ZN"
    assert selected[mixed].metadata["site_extraction"] == "selected_sites"
    assert set(selected) == set(component)
    summary = selected_result.feature_report["site_extraction_summary"]
    assert summary["metal_records_dropped"] == 1 and summary["structures_with_unselected_metals"] == 1
    assert summary["structures_skipped_no_selected_sites"] == 1
    skipped = selected_result.feature_report["skipped_pockets"]
    assert {"structure_id": "Z9999__chain_A__EC_1.1.1.1", "pocket_id": "", "reason": "no_selected_sites"} in skipped
    assert "site_extraction_summary" not in _load(paths, "component").feature_report


def test_selected_mode_refusals():
    from training.structure_loading import load_structure_pockets

    common = dict(structure_path=Path("x.pdb"), structure_root=Path("."), esm_dim=8, embeddings_dir=Path("."),
                  require_esm_embeddings=False, feature_root_dir=Path("."), external_feature_source="auto",
                  require_external_features=False, site_extraction="selected_sites")
    with pytest.raises(ValueError, match="needs a site summary"):
        load_structure_pockets(allowed_site_metal_labels=None, **common)
    with pytest.raises(ValueError, match="needs a site summary"):
        load_structure_pockets(allowed_site_metal_labels={}, metal_example_unit="ion", **common)
    with pytest.raises(ValueError, match="Unsupported site extraction"):
        load_structure_pockets(allowed_site_metal_labels={}, **{**common, "site_extraction": "other"})


def test_config_refusals_and_default():
    from training.config import TrainConfig, parse_args
    from training.run import validate_training_configuration

    assert parse_args(["--task", "ec"]).site_extraction == "component"
    assert parse_args(["--task", "ec", "--site-extraction", "selected_sites"]).site_extraction == "selected_sites"
    with pytest.raises(SystemExit):
        parse_args(["--task", "metal", "--metal-example-unit", "ion", "--site-extraction", "selected_sites"])
    with pytest.raises(ValueError, match="selected_sites requires"):
        validate_training_configuration(TrainConfig(task="ec", site_extraction="selected_sites", summary_csv=None))


def test_legacy_exporter_refuses_selected_and_fixed_runs():
    from export_validation_predictions import validate_config

    base = {"task": "ec", "train_val_split_by": "pdbid", "selection_metric": "val_ec_group_balanced_acc",
            "ec_label_depth": 1, "structure_dir": "/x/train/structures", "summary_csv": "/x/train/s.csv"}
    for extra, message in (({"site_extraction": "selected_sites"}, "component pockets only"),
                           ({"ec_class_vocabulary": "fixed_seven"}, "observed EC vocabulary only")):
        with pytest.raises(Exception, match=message):
            validate_config({**base, **extra})


def test_tiny_ec_fit_with_selected_sites(tmp_path, monkeypatch):
    import json

    from training.config import parse_args
    from training.run import run_training

    monkeypatch.setenv("DEEPMZYME_PARSE_CACHE_DIR", str(tmp_path / "cache"))
    paths = _mixed_dataset(tmp_path / "data", with_unsummarised=False)
    argv = tiny_ec_argv(paths, tmp_path / "runs", "--site-extraction", "selected_sites", "--val-fraction", "0.3",
                        "--ec-class-vocabulary", "fixed_seven", "--checkpoint-rule", "terminal",
                        "--run-name", "selected_tiny")
    run_dir = run_training(parse_args(argv))
    summary = json.loads((run_dir / "run_config.json").read_text())["dataset_summary"]
    extraction = summary["feature_load_report"]["site_extraction_summary"]
    assert extraction["metal_records_dropped"] == 1 and extraction["structures_with_unselected_metals"] == 1
    assert json.loads((run_dir / "run_config.json").read_text())["config"]["site_extraction"] == "selected_sites"
