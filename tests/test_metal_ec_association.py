"""Statistical and split safeguards for descriptive development association."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analyze_metal_ec_association import (
    association_statistics, build_tables, certified_training_examples,
    checked_site_labels, digest, execute, holm_adjust, permutation_mi, save,
)


def row(group, metal, ec=1, native=None):
    return {"group": group, "common4": metal, "native6": metal if native is None else native, "ec1": str(ec)}


def test_independence_and_perfect_association_have_known_statistics():
    independent = association_statistics([[10, 10], [10, 10]])
    assert independent["mi_nats"] == pytest.approx(0)
    assert independent["cramers_v"] == pytest.approx(0)
    perfect = association_statistics([[10, 0], [0, 10]])
    assert perfect["mi_nats"] == pytest.approx(np.log(2))
    assert perfect["nmi"] == pytest.approx(1)
    assert perfect["cramers_v"] == pytest.approx(1)
    assert perfect["asymptotic_chi_square_p_value"] is None


def test_empty_marginals_and_sparse_assumptions_are_explicit():
    assert association_statistics(np.zeros((6, 7)))["nmi"] is None
    assert association_statistics([[3, 0], [0, 0]])["cramers_v"] is None
    sparse = association_statistics([[1, 0], [0, 1]])
    assert sparse["expected_any_below_1"]
    assert not sparse["conventional_expected_cell_heuristic_passes"]
    with pytest.raises(ValueError):
        association_statistics([[1, -1], [0, 2]])


def test_each_group_has_one_weight_regardless_of_pocket_multiplicity():
    rows = [row("a", 0)]*10 + [row("b", 1, 2)]
    site, weighted, profiles, labels, audit = build_tables(rows, "common4")
    assert site.sum() == 11
    assert weighted.sum() == pytest.approx(2)
    assert np.allclose(profiles.sum(1), 1)
    assert weighted[0, 0] == weighted[1, 1] == 1
    assert audit["eligible_groups"] == 2
    assert set(labels) == {0, 1}


def test_mixed_metal_and_missing_ec_exclusions_are_view_specific():
    rows = [row("a", 3, native=""), row("b", 0), row("c", ""),
            {**row("d", 1), "ec1": ""}]
    assert build_tables(rows, "common4")[-1]["eligible_pockets"] == 2
    assert build_tables(rows, "native6")[-1]["eligible_pockets"] == 1
    assert build_tables(rows, "common4_native6_eligible")[-1]["eligible_pockets"] == 1
    assert build_tables(rows, "common4_native6_eligible")[-1]["excluded_additional_native6_eligibility"] == 1


def test_conflicting_ec_within_protein_is_rejected():
    with pytest.raises(ValueError, match="Conflicting"):
        build_tables([row("a", 0), row("a", 1, 2)], "common4")
    with pytest.raises(ValueError, match="Conflicting"):
        build_tables([row("a", 0), row("a", "", 2)], "common4")


def test_group_permutation_is_seeded_and_has_plus_one_correction():
    _, _, profiles, labels, _ = build_tables(
        [row(f"a{i}", 0) for i in range(5)] + [row(f"b{i}", 1, 2) for i in range(5)], "common4")
    first = permutation_mi(profiles, labels, 99, 42)
    assert first == permutation_mi(profiles, labels, 99, 42)
    assert .01 <= first["p_value"] <= 1
    # Duplicating sites within every group does not alter the permutation data.
    _, _, repeated, repeated_labels, _ = build_tables(
        [row(f"a{i}", 0) for i in range(5)]*3 + [row(f"b{i}", 1, 2) for i in range(5)], "common4")
    assert np.array_equal(profiles, repeated)
    assert np.array_equal(labels, repeated_labels)


def test_holm_correction_is_monotone_and_restores_order():
    assert holm_adjust([.04, .001, .02]) == pytest.approx([.04, .003, .04])


def split():
    return {"structure_dir": "/data/demo/train", "ec_label_depth": 1,
            "test_structure_dir": None, "test_summary_csv": None,
            "retained_split_identity": {
                "train": {"examples": [{"group": "a", "structure_id": "s", "pocket_id": "p"}]},
                "validation": {"examples": [{"group": "b"}]}}}


def test_train_source_and_internal_group_exclusion_fail_closed():
    valid = split()
    assert len(certified_training_examples(valid, "demo")) == 1
    invalid = split()
    invalid["test_summary_csv"] = "/heldout.csv"
    with pytest.raises(ValueError, match="test input"):
        certified_training_examples(invalid, "demo")
    invalid = split()
    invalid["retained_split_identity"]["validation"]["examples"][0]["group"] = "a"
    with pytest.raises(ValueError, match="overlap"):
        certified_training_examples(invalid, "demo")


def test_conflicting_duplicate_site_annotation_fails(tmp_path):
    path = tmp_path / "train.csv"
    path.write_text("structure,chain_resi,metaltype,ecnumber,whether_catalytic\na,A_1,FE,1.1.1.1,True\na,A_1,CO,1.1.1.1,True\n")
    with pytest.raises(ValueError, match="Conflicting"):
        checked_site_labels(path)


def test_changed_manifest_is_rejected_before_statistics(tmp_path):
    manifest = tmp_path / "analysis_manifest.json"
    save(manifest, {"phase": "prepared"})
    (tmp_path / "analysis_manifest.sha256").write_text(digest(manifest))
    save(manifest, {"phase": "modified"})
    with pytest.raises(ValueError, match="manifest hash"):
        execute(tmp_path)
