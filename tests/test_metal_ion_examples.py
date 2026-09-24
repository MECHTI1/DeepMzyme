"""Ion labels and pocket-grouped validation must agree for mixed-metal sites."""

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_structures import PocketRecord, ResidueRecord
from label_schemes import configure_active_metal_label_scheme
from training.config import parse_args
from training.metal_examples import metal_ion_examples
from training.splits import pocket_split_key, retained_split_identity, split_pockets_k_fold
from training import structure_loading


def _residue(number: int, x: float) -> ResidueRecord:
    return ResidueRecord(
        chain_id="A", resseq=number, icode="", resname="ALA",
        atoms={"CA": torch.tensor([x, 0.0, 0.0])},
    )


def _mixed_parent(index: int = 0) -> PocketRecord:
    fe = torch.tensor([0.0, 0.0, 0.0])
    mn = torch.tensor([3.0, 0.0, 0.0])
    site_ids = [("A", 101, ""), ("A", 102, "")]
    return PocketRecord(
        structure_id=f"1abc__chain_A__EC_1.1.1.{index + 1}",
        pocket_id=f"1abc__chain_A__EC_1.1.1.{index + 1}_METAL_0",
        metal_element="M",
        metal_coords=[fe, mn],
        residues=[_residue(1, -9.0), _residue(2, 0.0),
                  _residue(3, 3.0), _residue(4, 12.0)],
        metadata={"metal_site_ids": site_ids, "metal_site_symbols": ["FE", "MN"],
                  "metal_symbols_observed": ["FE", "MN"],
                  "metal_site_coord_map": dict(zip(site_ids, [fe, mn]))},
    )


def _summary(parent: PocketRecord) -> dict[tuple[str, str, str], str]:
    ec = parent.structure_id.rsplit("__EC_", 1)[1]
    return {("1abc", ec, "A_101"): "FE", ("1abc", ec, "A_102"): "MN"}


def test_mixed_fe_mn_produces_two_distinct_ion_targets_in_one_pocket_group():
    configure_active_metal_label_scheme("five_class")
    parent = _mixed_parent()
    ions, skipped = metal_ion_examples(
        parent, Path(parent.structure_id + ".pdb"), _summary(parent),
        unsupported_metal_policy="error",
    )
    assert skipped == []
    assert len(ions) == 2
    assert [ion.y_metal for ion in ions] == [3, 0]
    assert len({ion.pocket_id for ion in ions}) == 2
    assert {pocket_split_key(ion, "pocket_id") for ion in ions} == {parent.pocket_id}
    assert [ion.metal_count() for ion in ions] == [1, 1]
    assert [ion.metadata["ion_site_id"] for ion in ions] == parent.metadata["metal_site_ids"]
    assert ions[0].residues[0].resseq == 1 and ions[0].residues[-1].resseq == 3
    assert ions[1].residues[0].resseq == 2 and ions[1].residues[-1].resseq == 4
    identity = retained_split_identity(ions, "pocket_id")
    assert identity["n_examples"] == 2 and identity["n_groups"] == 1
    assert all(row["parent_pocket_id"] == parent.pocket_id for row in identity["examples"])


def test_summary_filter_is_per_ion_and_mismatched_sites_are_skipped():
    configure_active_metal_label_scheme("five_class")
    parent = _mixed_parent()
    path = Path(parent.structure_id + ".pdb")
    summary = _summary(parent)
    summary.pop(("1abc", "1.1.1.1", "A_102"))
    ions, skipped = metal_ion_examples(parent, path, summary, unsupported_metal_policy="error")
    assert [ion.metal_element for ion in ions] == ["FE"]
    assert [row["reason"] for row in skipped] == ["ion_not_in_catalytic_summary"]
    summary[("1abc", "1.1.1.1", "A_102")] = "ZN"
    ions, skipped = metal_ion_examples(parent, path, summary, unsupported_metal_policy="error")
    assert [ion.metal_element for ion in ions] == ["FE"]
    assert [row["reason"] for row in skipped] == ["observed_summary_metal_mismatch"]


def test_five_fold_pocket_grouping_never_separates_sibling_ions():
    configure_active_metal_label_scheme("five_class")
    ions = []
    for index in range(10):
        parent = _mixed_parent(index)
        children, skipped = metal_ion_examples(
            parent, Path(parent.structure_id + ".pdb"), _summary(parent),
            unsupported_metal_policy="error",
        )
        assert not skipped
        ions.extend(children)
    for fold in range(5):
        split = split_pockets_k_fold(
            ions, n_folds=5, fold_index=fold, split_by="pocket_id", seed=42, task="metal",
        )
        train_groups = {pocket_split_key(ion, "pocket_id") for ion in split.train_pockets}
        val_groups = {pocket_split_key(ion, "pocket_id") for ion in split.val_pockets}
        assert train_groups.isdisjoint(val_groups)
        assert len(split.train_pockets) + len(split.val_pockets) == 20


def test_cli_option_is_metal_only_and_defaults_to_pocket():
    assert parse_args(["--task", "metal"]).metal_example_unit == "pocket"
    config = parse_args(["--task", "metal", "--metal-example-unit", "ion",
                         "--train-val-split-by", "pocket_id"])
    assert config.metal_example_unit == "ion"
    assert config.train_val_split_by == "pocket_id"
    with pytest.raises(SystemExit):
        parse_args(["--task", "ec", "--metal-example-unit", "ion"])


def test_structure_loader_keeps_mixed_ions_when_selected(monkeypatch, tmp_path):
    configure_active_metal_label_scheme("five_class")
    parent = _mixed_parent()
    path = tmp_path / (parent.structure_id + ".pdb")
    monkeypatch.setattr(structure_loading, "parse_structure_file", lambda *args, **kwargs: object())
    monkeypatch.setattr(structure_loading, "extract_metal_pockets_from_structure", lambda *args, **kwargs: [parent])
    monkeypatch.setattr(structure_loading, "load_structure_feature_sources", lambda **kwargs: object())
    monkeypatch.setattr(structure_loading, "attach_structure_features_to_pocket", lambda *args, **kwargs: None)
    options = dict(
        structure_path=path, structure_root=tmp_path,
        allowed_site_metal_labels=_summary(parent), esm_dim=4,
        embeddings_dir=tmp_path, require_esm_embeddings=False,
        feature_root_dir=tmp_path, external_feature_source="none",
        require_external_features=False, unsupported_metal_policy="skip",
    )
    old, _, old_skips = structure_loading.load_structure_pockets(**options)
    ions, _, ion_skips = structure_loading.load_structure_pockets(**options, metal_example_unit="ion")
    assert old == [] and old_skips[0]["reason"] == "unsupported_metal_label"
    assert [ion.y_metal for ion in ions] == [3, 0]
    assert ion_skips == []
