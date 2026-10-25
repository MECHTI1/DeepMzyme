"""Fixed seven-class EC-class vocabulary and strict label checks (synthetic; no dataset access)."""
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training.ec_classes import (
    EC_CLASS_TOKEN_TO_INDEX, EC_CLASS_TOKENS, EC_PRIMARY_CLASS_TOKENS, FLAG_INCOMPLETE_DEEPER,
    FLAG_NONCANONICAL_DEEPER, STATUS_AMBIGUOUS, STATUS_MALFORMED, STATUS_MISSING, STATUS_MIXED_UNKNOWN,
    STATUS_OK, STATUS_OUT_OF_RANGE, accession_from_structure_id, apply_fixed_ec_class_vocabulary,
    assert_consistent_ec_protein_labels, classify_ec_class, complete_ec_prefix, ec_numbers_from_structure_id,
    normalize_ec_number_list,
)


def _pocket(structure_id, ec_numbers=None, token=None):
    metadata = {}
    if ec_numbers is not None:
        metadata["ec_numbers"] = list(ec_numbers)
    if token is not None:
        metadata["ec_label_token"] = token
    return SimpleNamespace(structure_id=structure_id, metadata=metadata)


def test_fixed_vocabulary_constants():
    assert EC_CLASS_TOKENS == ("1", "2", "3", "4", "5", "6", "7")
    assert EC_CLASS_TOKEN_TO_INDEX == {str(n): n - 1 for n in range(1, 8)}
    assert EC_PRIMARY_CLASS_TOKENS == EC_CLASS_TOKENS[:6]


@pytest.mark.parametrize(
    ("values", "token", "status", "flags"),
    [
        (["6.2.1.n2"], "6", STATUS_OK, (FLAG_NONCANONICAL_DEEPER,)),
        (["3.-.-.-"], "3", STATUS_OK, (FLAG_INCOMPLETE_DEEPER,)),
        (["3.1.1.1", "3.2.1.1"], "3", STATUS_OK, ()),
        (["3.1.1.1;3.2.1.1"], "3", STATUS_OK, ()),
        ([" 3.1.1.1"], "3", STATUS_OK, ()),
        (["7.1.1.9"], "7", STATUS_OK, ()),
        (["2.5.1.3", "4.1.99.17"], None, STATUS_AMBIGUOUS, ()),
        ([], None, STATUS_MISSING, ()),
        (["-.-.-.-"], None, STATUS_MISSING, ()),
        (["0.0.0.0"], None, STATUS_OUT_OF_RANGE, ()),
        (["8.1.1.1"], None, STATUS_OUT_OF_RANGE, ()),
        (["10.1.1.1"], None, STATUS_OUT_OF_RANGE, ()),
        (["x.1.1.1"], None, STATUS_MALFORMED, ()),
        (["3.1.1.1", "0.0.0.0"], None, STATUS_MIXED_UNKNOWN, ()),
        (["3.1.1.1", "-.-.-.-"], None, STATUS_MIXED_UNKNOWN, ()),
    ],
)
def test_classify_ec_class_table(values, token, status, flags):
    result = classify_ec_class(values)
    assert (result.token, result.status, result.flags) == (token, status, flags)
    assert result.is_fatal == (status in {STATUS_MISSING, STATUS_OUT_OF_RANGE, STATUS_MALFORMED, STATUS_MIXED_UNKNOWN})


@pytest.mark.parametrize(
    ("values", "depth", "expected"),
    [
        (["6.2.1.n2"], 3, "6.2.1"),
        (["6.2.1.n2"], 4, None),
        (["3.1.1.1", "3.1.1.2"], 3, "3.1.1"),
        (["3.1.1.1", "3.1.2.1"], 3, None),
        (["3.1.-.-", "3.1.4.1"], 3, "3.1.4"),
        (["3.-.-.-"], 2, None),
        (["2.7.11.1"], 2, "2.7"),
    ],
)
def test_complete_ec_prefix(values, depth, expected):
    assert complete_ec_prefix(values, depth) == expected


def test_normalisation_and_structure_id_match_legacy_labels():
    from training import labels

    for value in ["3.1.1.1;3.2.1.1", " 3.1.1.1 , 3.1.1.1;", "6.2.1.n2", "-.-.-.-,2.7.1.1", ""]:
        assert normalize_ec_number_list(value) == labels.normalize_ec_number_list(value)
    for stem in ["P12345__chain_A__EC_3.1.1.1", "Q9XYZ1__chain_B__EC_2.5.1.3,4.1.99.17"]:
        assert ec_numbers_from_structure_id(stem) == labels.parse_structure_ec_numbers(stem)
        assert accession_from_structure_id(stem).lower() == labels.parse_structure_identity(stem)[0]
    for values in (["3.1.1.1"], ["6.2.1.n2"], ["3.-.-.-"], ["7.1.1.9"], ["2.5.1.3", "4.1.99.17"]):
        strict = classify_ec_class(values)
        legacy = labels.ec_label_token_from_numbers(tuple(values), depth=1)
        assert strict.token == legacy


def test_apply_fixed_vocabulary_report_and_ambiguous_exclusion():
    pockets = [
        _pocket("P1__chain_A__EC_3.1.1.1", ["3.1.1.1"], token="3"),
        _pocket("P1__chain_A__EC_3.1.1.1", ["3.1.1.1"], token="3"),
        _pocket("P2__chain_A__EC_6.2.1.n2", ["6.2.1.n2"], token="6"),
        _pocket("P3__chain_A__EC_2.5.1.3,4.1.99.17", ["2.5.1.3", "4.1.99.17"]),
        _pocket("P4__chain_A__EC_ 3.1.1.1", [" 3.1.1.1"]),
    ]
    report = apply_fixed_ec_class_vocabulary(pockets)
    assert [p.metadata.get("ec_label_token") for p in pockets] == ["3", "3", "6", None, "3"]
    assert pockets[3].metadata["ec_label_status"] == STATUS_AMBIGUOUS
    assert report["n_structures"] == 4
    assert report["structures_by_status"] == {STATUS_AMBIGUOUS: 1, STATUS_OK: 3}
    assert report["structures_by_flag"] == {FLAG_NONCANONICAL_DEEPER: 1}
    assert report["structures_by_ec_class"]["3"] == 2
    assert report["structures_by_ec_class"]["5"] == 0
    assert report["excluded_ambiguous_structure_ids"] == ["P3__chain_A__EC_2.5.1.3,4.1.99.17"]


@pytest.mark.parametrize("values", [["0.0.0.0"], ["8.1.1.1"], ["-.-.-.-"], ["3.1.1.1", "0.0.0.0"]])
def test_apply_fixed_vocabulary_raises_on_fatal_annotations(values):
    with pytest.raises(ValueError, match="cannot give an EC class"):
        apply_fixed_ec_class_vocabulary([_pocket("P9__chain_A__EC_x", values)])


def test_apply_fixed_vocabulary_rejects_legacy_disagreement():
    with pytest.raises(ValueError, match="disagrees with legacy token"):
        apply_fixed_ec_class_vocabulary([_pocket("P1__chain_A__EC_3.1.1.1", ["3.1.1.1"], token="4")])


def test_apply_fixed_vocabulary_falls_back_to_structure_id():
    pocket = _pocket("P5__chain_A__EC_5.3.1.1")
    apply_fixed_ec_class_vocabulary([pocket])
    assert pocket.metadata["ec_label_token"] == "5"


def test_consistent_protein_labels():
    good = [
        _pocket("P1__chain_A__EC_3.1.1.1", ["3.1.1.1"], token="3"),
        _pocket("P1__chain_A__EC_3.1.1.1", ["3.1.1.1"], token="3"),
        _pocket("P2__chain_A__EC_2.1.1.1", ["2.1.1.1"], token="2"),
    ]
    assert assert_consistent_ec_protein_labels(good, membership_ec_class={"P1": "3"}) == {
        "n_accessions": 2, "conflicts": 0,
    }
    with pytest.raises(ValueError, match="membership EC class"):
        assert_consistent_ec_protein_labels(good, membership_ec_class={"P1": "4"})
    two_structures = good + [_pocket("P1__chain_B__EC_3.1.1.1", ["3.1.1.1"], token="3")]
    with pytest.raises(ValueError, match="structures"):
        assert_consistent_ec_protein_labels(two_structures)


def test_module_does_not_import_torch():
    src = Path(__file__).resolve().parents[1] / "src"
    code = (
        f"import sys; sys.path.insert(0, {str(src)!r}); "
        "import training.ec_classes, training.ec_class_metrics; print('torch' in sys.modules)"
    )
    output = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert output.strip() == "False"


def _record(structure_id, ec_numbers):
    import torch
    from data_structures import PocketRecord

    return PocketRecord(
        structure_id=structure_id, pocket_id=f"{structure_id}_P0", metal_element="ZN",
        metal_coords=[torch.zeros(3)], residues=[],
        metadata={"ec_numbers": list(ec_numbers), "ec_label_token": ec_numbers[0].split(".")[0] if ec_numbers else None},
    )


def _stub_loader(monkeypatch, tmp_path, specs):
    import training.data as data
    import training.parallel_loading as parallel_loading

    paths = [tmp_path / f"{sid}.pdb" for sid in specs]
    monkeypatch.setattr(data, "find_structure_files", lambda p: paths)
    monkeypatch.setattr(data, "resolve_allowed_site_metal_labels", lambda p: {})
    monkeypatch.setattr(data, "resolve_runtime_feature_paths", lambda **kw: (tmp_path, tmp_path))

    def load(**kwargs):
        sid = kwargs["structure_path"].stem
        pocket = _record(sid, specs[sid])
        if pocket.metadata["ec_label_token"] is None:
            pocket.metadata.pop("ec_label_token")
        return [pocket], [], []

    monkeypatch.setattr(parallel_loading, "load_structure_pockets", load)
    return data


def test_loader_fixed_seven_keeps_indices_and_counts_ambiguous(monkeypatch, tmp_path):
    specs = {
        "P1__chain_A__EC_3.1.1.1": ["3.1.1.1"],
        "P2__chain_A__EC_6.2.1.n2": ["6.2.1.n2"],
        "P3__chain_A__EC_2.5.1.3,4.1.99.17": ["2.5.1.3", "4.1.99.17"],
    }
    data = _stub_loader(monkeypatch, tmp_path, specs)
    result = data.load_training_pockets_with_report_from_dir(
        tmp_path, required_targets=("ec",), load_workers=1, ec_class_vocabulary="fixed_seven",
    )
    assert result.ec_label_to_index == EC_CLASS_TOKEN_TO_INDEX
    assert {p.structure_id: p.y_ec for p in result.pockets} == {
        "P1__chain_A__EC_3.1.1.1": 2, "P2__chain_A__EC_6.2.1.n2": 5,
    }
    skipped = result.feature_report["skipped_pockets"]
    assert result.ec_label_report["excluded_ambiguous_structure_ids"] == ["P3__chain_A__EC_2.5.1.3,4.1.99.17"]
    assert result.ec_label_report["structures_by_ec_class"]["5"] == 0
    assert [(row["structure_id"], row["reason"]) for row in skipped] == [
        ("P3__chain_A__EC_2.5.1.3,4.1.99.17", "ambiguous_ec_class"),
    ]


def test_loader_observed_default_is_unchanged(monkeypatch, tmp_path):
    specs = {"P1__chain_A__EC_3.1.1.1": ["3.1.1.1"], "P2__chain_A__EC_6.2.1.1": ["6.2.1.1"]}
    data = _stub_loader(monkeypatch, tmp_path, specs)
    result = data.load_training_pockets_with_report_from_dir(tmp_path, required_targets=("ec",), load_workers=1)
    assert result.ec_label_to_index == {"3": 0, "6": 1}
    assert result.ec_label_report is None


def test_loader_held_out_class_absent_from_training_keeps_its_index(monkeypatch, tmp_path):
    specs = {"P5__chain_A__EC_5.3.1.1": ["5.3.1.1"]}
    data = _stub_loader(monkeypatch, tmp_path, specs)
    result = data.load_training_pockets_with_report_from_dir(
        tmp_path, required_targets=("ec",), load_workers=1, ec_class_vocabulary="fixed_seven",
        ec_label_to_index=dict(EC_CLASS_TOKEN_TO_INDEX),
    )
    assert [p.y_ec for p in result.pockets] == [4]


def test_loader_fixed_seven_refusals(monkeypatch, tmp_path):
    data = _stub_loader(monkeypatch, tmp_path, {"P1__chain_A__EC_0.0.0.0": ["0.0.0.0"]})
    with pytest.raises(ValueError, match="cannot give an EC class"):
        data.load_training_pockets_with_report_from_dir(
            tmp_path, required_targets=("ec",), load_workers=1, ec_class_vocabulary="fixed_seven",
        )
    with pytest.raises(ValueError, match="different label map"):
        data.load_training_pockets_with_report_from_dir(
            tmp_path, required_targets=("ec",), load_workers=1, ec_class_vocabulary="fixed_seven",
            ec_label_to_index={"3": 0},
        )
    with pytest.raises(ValueError, match="depth 1"):
        data.load_training_pockets_with_report_from_dir(
            tmp_path, required_targets=("metal",), load_workers=1, ec_class_vocabulary="fixed_seven",
        )


def test_config_flags_and_refusals():
    from training.config import parse_args

    default = parse_args(["--task", "ec"])
    assert default.ec_class_vocabulary == "observed" and default.ec_conflicting_labels == "skip"
    fixed = parse_args(["--task", "ec", "--ec-class-vocabulary", "fixed_seven", "--ec-conflicting-labels", "fail",
                        "--selection-metric", "val_ec_group_ec_classes_1_6_balanced_acc"])
    assert fixed.ec_class_vocabulary == "fixed_seven" and fixed.ec_conflicting_labels == "fail"
    for argv in (
        ["--task", "metal", "--ec-class-vocabulary", "fixed_seven"],
        ["--task", "joint", "--ec-class-vocabulary", "fixed_seven"],
        ["--task", "ec", "--ec-label-depth", "2", "--ec-class-vocabulary", "fixed_seven"],
        ["--task", "ec", "--ec-class-vocabulary", "fixed_seven", "--require-all-task-classes"],
        ["--task", "metal", "--ec-conflicting-labels", "fail"],
        ["--task", "ec", "--selection-metric", "val_ec_group_ec_classes_1_6_balanced_acc"],
    ):
        with pytest.raises(SystemExit):
            parse_args(argv)


def test_validate_training_configuration_mirrors_parse_checks():
    from training.config import TrainConfig
    from training.run import validate_training_configuration

    with pytest.raises(ValueError, match="fixed_seven requires"):
        validate_training_configuration(TrainConfig(task="joint", ec_class_vocabulary="fixed_seven"))
    with pytest.raises(ValueError, match="requires an EC task"):
        validate_training_configuration(TrainConfig(task="metal", ec_conflicting_labels="fail"))


def test_grouped_metrics_fixed_classes_and_conflicts():
    import torch
    from training.run import ec_group_metrics_from_logits

    label_map = {index: token for token, index in EC_CLASS_TOKEN_TO_INDEX.items()}
    logits = torch.zeros(4, 7)
    logits[0, 2] = logits[1, 2] = 1.0
    logits[2, 6] = 1.0
    logits[3, 1] = 1.0
    targets = torch.tensor([2, 2, 0, 1])
    groups = torch.tensor([0, 0, 1, 2])
    metrics = ec_group_metrics_from_logits(logits, targets, groups, ec_label_map=label_map, ec_label_depth=1,
                                           fixed_ec_classes=True)
    assert metrics["n_groups"] == 3
    assert metrics["ec_classes_1_6_balanced_accuracy"] == pytest.approx((1.0 + 0.0 + 1.0) / 3)
    assert metrics["ec_classes_1_6_n_present"] == 3
    conflicting = torch.tensor([2, 1, 0, 1])
    skipped = ec_group_metrics_from_logits(logits, conflicting, groups, ec_label_map=label_map, ec_label_depth=1)
    assert skipped["n_conflicting_groups"] == 1
    with pytest.raises(ValueError, match="different EC labels"):
        ec_group_metrics_from_logits(logits, conflicting, groups, ec_label_map=label_map, ec_label_depth=1,
                                     fail_on_conflict=True)
    with pytest.raises(ValueError, match="fixed seven-class label map"):
        ec_group_metrics_from_logits(logits, targets, groups, ec_label_map={0: "3"}, ec_label_depth=1,
                                     fixed_ec_classes=True)


def test_split_support_counts_absent_classes():
    from training.ec_classes import ec_class_split_support

    train = [_pocket("P1__chain_A__EC_3.1.1.1", token="3"), _pocket("P2__chain_A__EC_1.1.1.1", token="1")]
    val = [_pocket("P3__chain_A__EC_5.1.1.1", token="5"), _pocket("P4__chain_A__EC_3.2.1.1", token="3")]
    support = ec_class_split_support(train, val)
    assert support["train_proteins_by_ec_class"]["3"] == 1
    assert "5" in support["ec_classes_absent_from_training"]
    assert support["n_val_proteins_in_classes_absent_from_training"] == 1
