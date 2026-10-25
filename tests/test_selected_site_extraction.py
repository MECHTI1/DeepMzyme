"""Selected-site pocket extraction: only summary-selected metal sites define pockets.

All structures and summaries are small synthetic files written to ``tmp_path``.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from featurization import MultinuclearSiteHandler  # noqa: E402
from graph.shell_roles import compute_shell_roles  # noqa: E402
from graph.structure_parsing import (  # noqa: E402
    NO_SELECTED_SITES_REASON,
    SELECTED_SITE_EXTRACTION,
    AmbiguousSelectedSiteError,
    UnresolvedSelectedSiteError,
    extract_metal_pockets_from_structure,
    extract_selected_site_pockets_from_structure,
    parse_structure_file,
)
from training.site_filter import (  # noqa: E402
    SiteKeyIndex,
    load_allowed_site_metal_labels,
    matched_site_metal_types,
    parse_summary_chain_resi,
    pocket_matches_allowed_sites,
    selected_site_ids_for_structure,
    validate_selected_site_summary,
)

STEM = "P12345__chain_A__EC_3.4.24.1"
SUMMARY_COLUMNS = ("structure", "chain_resi", "ecnumber", "metaltype", "whether_catalytic")


def _pdb_line(record, serial, name, resname, chain, resseq, icode, xyz, element):
    atom_name = f" {name:<3}" if len(name) < 4 and len(element) == 1 else f"{name:<4}"
    x, y, z = xyz
    return (f"{record:<6}{serial:>5} {atom_name} {resname:>3} {chain:1}{resseq:>4}{icode:1}   "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{1.0:>6.2f}{20.0:>6.2f}          {element:>2}")


def _residue(resseq, resname, ca, donor=None):
    """Protein residue with a CA atom and optionally one ``(name, element, xyz)`` donor atom."""
    atoms = [("CA", "C", ca)]
    if donor is not None:
        atoms.append(donor)
    return "A", resseq, resname, atoms


def _metal(resseq, element, xyz, *, icode="", atom_name=None):
    return "A", resseq, icode, element, atom_name or element, xyz


def _write_structure(tmp_path, residues, metals, *, n_models=1, stem=STEM):
    lines = []
    for model in range(n_models):
        if n_models > 1:
            lines.append(f"MODEL     {model + 1:>4}")
        serial = 1
        for chain, resseq, resname, atoms in residues:
            for name, element, xyz in atoms:
                lines.append(_pdb_line("ATOM", serial, name, resname, chain, resseq, "", xyz, element))
                serial += 1
        for chain, resseq, icode, element, atom_name, xyz in metals:
            lines.append(_pdb_line("HETATM", serial, atom_name, element, chain, resseq, icode, xyz, element))
            serial += 1
        if n_models > 1:
            lines.append("ENDMDL")
    lines.append("END")
    path = tmp_path / f"{stem}.pdb"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path, parse_structure_file(str(path), structure_id=path.stem)


def _write_summary(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(SUMMARY_COLUMNS)
        writer.writerows(rows)
    return path


def _residue_numbers(pocket):
    return [residue.resseq for residue in pocket.residues]


def _mixed_cluster_structure(tmp_path, unselected_element):
    """Selected ZN A_900 at the origin and an unselected ion A_901 3 Å away.

    Residue 2 (ASP) binds only A_901 and residue 3 lies within the pocket
    radius of A_901 but not of A_900; residue 5 is outside both pockets.
    """
    residues = [
        _residue(1, "HIS", (0.0, 4.0, 0.0), ("NE2", "N", (0.0, 2.1, 0.0))),
        _residue(2, "ASP", (6.6, 1.0, 0.0), ("OD1", "O", (5.1, 0.0, 0.0))),
        _residue(3, "ALA", (12.5, 0.0, 0.0)),
        _residue(4, "ALA", (-8.0, 0.0, 0.0)),
        _residue(5, "ALA", (30.0, 0.0, 0.0)),
    ]
    metals = [_metal(900, "ZN", (0.0, 0.0, 0.0)), _metal(901, unselected_element, (3.0, 0.0, 0.0))]
    return _write_structure(tmp_path, residues, metals)


@pytest.mark.parametrize("unselected_element", ["ZN", "FE"])
def test_mixed_cluster_keeps_only_the_selected_ion(tmp_path, unselected_element):
    path, structure = _mixed_cluster_structure(tmp_path, unselected_element)

    pockets, report = extract_selected_site_pockets_from_structure(
        structure, structure_id=path.stem, selected_site_ids=frozenset({("A", 900)}),
    )

    assert len(pockets) == 1
    pocket = pockets[0]
    assert pocket.pocket_id == f"{STEM}_METAL_0"
    assert pocket.metal_count() == 1
    assert pocket.metal_element == "ZN"
    assert _residue_numbers(pocket) == [1, 2, 4]
    assert pocket.metadata["metal_site_ids"] == [("A", 900, "")]
    assert pocket.metadata["site_extraction"] == SELECTED_SITE_EXTRACTION
    assert pocket.metadata["selected_site_keys"] == [("A", 900)]
    assert pocket.metadata["legacy_component_site_ids"] == [("A", 900, ""), ("A", 901, "")]
    assert MultinuclearSiteHandler.site_metal_stats(pocket).tolist() == [0.0, 1.0, 0.0, 0.0]
    first_shell = [flags[0] for flags in compute_shell_roles(pocket, use_ring_edges=False,
                                                             shell_role_source="geometry")]
    assert first_shell == [True, False, False]

    assert report["metal_records_total"] == 2
    assert report["metal_records_selected"] == 1
    assert report["metal_records_dropped"] == 1
    assert report["components_with_unselected_metals_count"] == 1
    assert report["components_with_unselected_metals"][0]["unselected_site_ids"] == [("A", 901, "")]
    assert report["split_component_count"] == 0
    assert report["residue_count_delta_total"] == -1
    row = report["pockets"][0]
    assert row["legacy_pocket_id"] == f"{STEM}_METAL_0"
    assert (row["residue_count"], row["legacy_residue_count"], row["residue_count_delta"]) == (3, 4, -1)
    assert row["legacy_component_metal_count"] == 2

    legacy = extract_metal_pockets_from_structure(structure, structure_id=path.stem)
    assert len(legacy) == 1
    assert legacy[0].metal_count() == 2
    assert legacy[0].metal_element == ("ZN" if unselected_element == "ZN" else "METAL")
    assert _residue_numbers(legacy[0]) == [1, 2, 3, 4]
    assert "site_extraction" not in legacy[0].metadata
    legacy_first_shell = [flags[0] for flags in compute_shell_roles(legacy[0], use_ring_edges=False,
                                                                    shell_role_source="geometry")]
    assert legacy_first_shell == [True, True, False, False]


def test_summary_keys_select_sites_that_the_legacy_matcher_accepts(tmp_path):
    path, structure = _mixed_cluster_structure(tmp_path, "FE")
    summary = _write_summary(tmp_path / "summary.csv", [
        ("P12345", "A_900", "3.4.24.1", "Zn", "1"),
        ("P12345", "A_901", "3.4.24.1", "Fe", "0"),
    ])
    labels = load_allowed_site_metal_labels(summary)

    selected = selected_site_ids_for_structure(path, labels)
    pockets, _report = extract_selected_site_pockets_from_structure(
        structure, structure_id=path.stem, selected_site_ids=selected,
    )

    assert selected == frozenset({("A", 900)})
    assert [pocket_matches_allowed_sites(pocket, path, labels) for pocket in pockets] == [True]
    assert matched_site_metal_types(pockets[0], path, labels) == {"ZN"}


def test_selected_ions_within_merge_distance_stay_one_multinuclear_pocket(tmp_path):
    residues = [
        _residue(1, "HIS", (0.0, 4.0, 0.0), ("NE2", "N", (0.0, 2.1, 0.0))),
        _residue(2, "ALA", (-8.0, 0.0, 0.0)),
    ]
    metals = [_metal(900, "ZN", (0.0, 0.0, 0.0)), _metal(901, "ZN", (3.0, 0.0, 0.0))]
    path, structure = _write_structure(tmp_path, residues, metals)

    pockets, report = extract_selected_site_pockets_from_structure(
        structure, structure_id=path.stem, selected_site_ids={("A", 900), ("A", 901)},
    )

    assert len(pockets) == 1
    assert pockets[0].metal_count() == 2
    assert pockets[0].metadata["selected_site_keys"] == [("A", 900), ("A", 901)]
    legacy = extract_metal_pockets_from_structure(structure, structure_id=path.stem)
    assert _residue_numbers(pockets[0]) == _residue_numbers(legacy[0])
    assert report["metal_records_dropped"] == 0
    assert report["components_with_unselected_metals_count"] == 0
    assert report["split_component_count"] == 0
    assert report["residue_count_delta_total"] == 0


def test_unselected_bridging_ion_splits_one_component_into_two_pockets(tmp_path):
    residues = [
        _residue(1, "ALA", (-9.5, 0.0, 0.0)),
        _residue(2, "ALA", (0.0, 5.0, 0.0)),
        _residue(3, "ALA", (8.0, 5.0, 0.0)),
        _residue(4, "ALA", (17.5, 0.0, 0.0)),
    ]
    metals = [
        _metal(900, "ZN", (0.0, 0.0, 0.0)),
        _metal(901, "ZN", (4.0, 0.0, 0.0)),
        _metal(902, "ZN", (8.0, 0.0, 0.0)),
    ]
    path, structure = _write_structure(tmp_path, residues, metals)

    pockets, report = extract_selected_site_pockets_from_structure(
        structure, structure_id=path.stem, selected_site_ids={("A", 900), ("A", 902)},
    )

    assert [pocket.pocket_id for pocket in pockets] == [f"{STEM}_METAL_0", f"{STEM}_METAL_1"]
    assert [pocket.metadata["selected_site_keys"] for pocket in pockets] == [[("A", 900)], [("A", 902)]]
    assert [pocket.metal_count() for pocket in pockets] == [1, 1]
    assert [_residue_numbers(pocket) for pocket in pockets] == [[1, 2, 3], [2, 3, 4]]
    all_site_ids = [("A", 900, ""), ("A", 901, ""), ("A", 902, "")]
    assert all(pocket.metadata["legacy_component_site_ids"] == all_site_ids for pocket in pockets)
    assert report["split_component_count"] == 1
    assert report["split_components"][0]["pocket_ids"] == [pocket.pocket_id for pocket in pockets]
    assert report["split_components"][0]["selected_cluster_count"] == 2
    assert report["components_with_unselected_metals_count"] == 1
    assert [row["legacy_pocket_id"] for row in report["pockets"]] == [f"{STEM}_METAL_0"] * 2
    assert [row["residue_count_delta"] for row in report["pockets"]] == [-1, -1]
    # The two pockets together cover all four legacy residues.
    assert report["residue_count_delta_total"] == 0

    legacy = extract_metal_pockets_from_structure(structure, structure_id=path.stem)
    assert [pocket.metal_count() for pocket in legacy] == [3]


def test_split_component_counts_pockets_not_selected_clusters(tmp_path):
    # A_902 is selected but has no residue within the pocket radius, so the
    # bridged component yields one pocket and is not reported as split.
    residues = [_residue(1, "ALA", (-5.0, 0.0, 0.0))]
    metals = [
        _metal(900, "ZN", (0.0, 0.0, 0.0)),
        _metal(901, "ZN", (4.0, 0.0, 0.0)),
        _metal(902, "ZN", (8.0, 0.0, 0.0)),
    ]
    path, structure = _write_structure(tmp_path, residues, metals)

    pockets, report = extract_selected_site_pockets_from_structure(
        structure, structure_id=path.stem, selected_site_ids={("A", 900), ("A", 902)},
    )

    assert [pocket.metadata["selected_site_keys"] for pocket in pockets] == [[("A", 900)]]
    assert report["selected_cluster_count"] == 2
    assert report["split_component_count"] == 0
    assert report["selected_sites_without_pocket"] == [("A", 902)]
    assert report["residue_count_delta_total"] == 0


def test_pocket_ids_number_filtered_clusters_not_legacy_components(tmp_path):
    residues = [_residue(1, "ALA", (40.0, 5.0, 0.0)), _residue(2, "ALA", (0.0, 5.0, 0.0))]
    metals = [_metal(800, "ZN", (40.0, 0.0, 0.0)), _metal(900, "ZN", (0.0, 0.0, 0.0))]
    path, structure = _write_structure(tmp_path, residues, metals)

    pockets, report = extract_selected_site_pockets_from_structure(
        structure, structure_id=path.stem, selected_site_ids={("A", 900)},
    )

    assert [pocket.pocket_id for pocket in pockets] == [f"{STEM}_METAL_0"]
    assert pockets[0].metadata["legacy_component_site_ids"] == [("A", 900, "")]
    assert report["pockets"][0]["legacy_pocket_id"] == f"{STEM}_METAL_1"
    assert report["legacy_components_without_selected_sites"] == 1
    assert report["components_with_unselected_metals_count"] == 0
    legacy = extract_metal_pockets_from_structure(structure, structure_id=path.stem)
    assert [(pocket.pocket_id, pocket.metadata["metal_site_ids"]) for pocket in legacy] == [
        (f"{STEM}_METAL_0", [("A", 800, "")]),
        (f"{STEM}_METAL_1", [("A", 900, "")]),
    ]


def test_selected_site_without_metal_atom_raises(tmp_path):
    residues = [_residue(1, "ALA", (0.0, 5.0, 0.0))]
    metals = [_metal(900, "ZN", (0.0, 0.0, 0.0)), _metal(950, "MG", (20.0, 0.0, 0.0))]
    path, structure = _write_structure(tmp_path, residues, metals)

    assert issubclass(UnresolvedSelectedSiteError, ValueError)
    for missing_site in (("A", 999), ("A", 1), ("A", 950)):
        with pytest.raises(UnresolvedSelectedSiteError, match="without a metal atom record"):
            extract_selected_site_pockets_from_structure(
                structure, structure_id=path.stem, selected_site_ids={("A", 900), missing_site},
            )


def test_selected_site_with_several_metal_atoms_raises(tmp_path):
    residues = [_residue(1, "ALA", (0.0, 5.0, 0.0))]
    two_atoms = [
        _metal(900, "ZN", (0.0, 0.0, 0.0), atom_name="ZN1"),
        _metal(900, "ZN", (1.0, 0.0, 0.0), atom_name="ZN2"),
    ]
    path, structure = _write_structure(tmp_path, residues, two_atoms)
    with pytest.raises(AmbiguousSelectedSiteError, match="more than one metal atom record"):
        extract_selected_site_pockets_from_structure(
            structure, structure_id=path.stem, selected_site_ids={("A", 900)},
        )

    icode_variants = [
        _metal(900, "ZN", (0.0, 0.0, 0.0)),
        _metal(900, "ZN", (1.0, 0.0, 0.0), icode="B"),
        _metal(901, "ZN", (20.0, 0.0, 0.0)),
    ]
    path, structure = _write_structure(tmp_path, residues, icode_variants)
    with pytest.raises(AmbiguousSelectedSiteError):
        extract_selected_site_pockets_from_structure(
            structure, structure_id=path.stem, selected_site_ids={("A", 900)},
        )
    pockets, report = extract_selected_site_pockets_from_structure(
        structure, structure_id=path.stem, selected_site_ids={("A", 901)},
    )
    assert report["metal_records_selected"] == 1 and pockets == []
    assert report["selected_sites_without_pocket"] == [("A", 901)]

    path, structure = _write_structure(tmp_path, residues, [_metal(900, "ZN", (0.0, 0.0, 0.0))], n_models=2)
    with pytest.raises(AmbiguousSelectedSiteError):
        extract_selected_site_pockets_from_structure(
            structure, structure_id=path.stem, selected_site_ids={("A", 900)},
        )
    pockets, report = extract_selected_site_pockets_from_structure(
        structure, structure_id=path.stem, selected_site_ids={("A", 900)}, model_index=0,
    )
    assert [pocket.metal_count() for pocket in pockets] == [1]
    assert report["model_index"] == 0


def test_no_selected_sites_gives_no_pockets(tmp_path):
    """An empty selection is not an error: the extractor returns no pockets and
    flags ``report["no_selected_sites"]``; the structure loader records one
    skipped row with reason ``NO_SELECTED_SITES_REASON`` for the structure."""
    path, structure = _mixed_cluster_structure(tmp_path, "ZN")
    other_keys = {("q99999", "3.4.24.1", "A_900"): "ZN", ("p12345", "1.1.1.1", "A_900"): "ZN"}

    selected = selected_site_ids_for_structure(path, other_keys)
    pockets, report = extract_selected_site_pockets_from_structure(
        structure, structure_id=path.stem, selected_site_ids=selected,
    )

    assert selected == frozenset()
    assert pockets == []
    assert NO_SELECTED_SITES_REASON == "no_selected_sites"
    assert report["no_selected_sites"] is True
    assert report["pocket_count"] == 0
    assert report["metal_records_total"] == report["metal_records_dropped"] == 2
    assert report["legacy_components_without_selected_sites"] == 1


def test_chain_resi_parsing_and_key_normalization(tmp_path):
    assert parse_summary_chain_resi(" A_900 ") == ("A", 900)
    assert parse_summary_chain_resi("AB_C_12") == ("AB_C", 12)
    assert parse_summary_chain_resi("A_-5") == ("A", -5)
    malformed_values = (
        "A900", "_900", "A_", "A_900B", "A_9.0", "A_0900", "A_+900", "", "A_1_000", "A _ 900", "A_ 900", "A _900",
    )
    for malformed in malformed_values:
        with pytest.raises(ValueError, match="Malformed chain_resi"):
            parse_summary_chain_resi(malformed)

    path = tmp_path / f"{STEM}.pdb"
    keys = {
        ("p12345", "3.4.24.1", "A_900"),
        ("p12345", "3.4.24.1", "B_7"),
        ("p12345", "1.1.1.1", "A_901"),
        ("q99999", "3.4.24.1", "A_902"),
        ("q99999", "3.4.24.1", "not-a-site"),
        # Not in the form the summary reader writes: the matcher never matches these.
        ("P12345 ", "3.4.24.1", "A_903"),
        ("p12345", "3.4.24.1,", "A_904"),
    }
    assert selected_site_ids_for_structure(path, keys) == frozenset({("A", 900), ("B", 7)})
    assert selected_site_ids_for_structure(path, keys, key_mode="accession") == frozenset(
        {("A", 900), ("B", 7), ("A", 901), ("A", 904)}
    )
    index = SiteKeyIndex(keys)
    assert selected_site_ids_for_structure(path, index) == selected_site_ids_for_structure(path, keys)
    assert selected_site_ids_for_structure(path, index, key_mode="accession") == selected_site_ids_for_structure(
        path, keys, key_mode="accession"
    )
    for bad_key in (("p12345", "3.4.24.1", "A_900B"), ("p12345", "3.4.24.1", " A_900 ")):
        with pytest.raises(ValueError, match="Malformed chain_resi"):
            selected_site_ids_for_structure(path, keys | {bad_key})
    with pytest.raises(ValueError, match="Site keys must be"):
        selected_site_ids_for_structure(path, [("p12345", "A_900")])
    with pytest.raises(ValueError, match="Site keys must be"):
        SiteKeyIndex([("p12345", "A_900")])
    with pytest.raises(ValueError, match="Unsupported site key mode"):
        selected_site_ids_for_structure(path, keys, key_mode="ec")

    far_apart = [_metal(900 + offset, "ZN", (20.0 * offset, 0.0, 0.0)) for offset in range(5)]
    near_residues = [_residue(offset + 1, "ALA", (20.0 * offset, 5.0, 0.0)) for offset in range(5)]
    path, structure = _write_structure(tmp_path, near_residues, far_apart)
    chain_a_keys = {key for key in keys if not key[2].startswith("B_")}
    legacy = extract_metal_pockets_from_structure(structure, structure_id=path.stem)
    legacy_matched = {
        pocket.metadata["metal_site_ids"][0][:2]
        for pocket in legacy
        if pocket_matches_allowed_sites(pocket, path, chain_a_keys)
    }
    assert selected_site_ids_for_structure(path, chain_a_keys) == legacy_matched == {("A", 900)}

    residues = [_residue(1, "ALA", (0.0, 5.0, 0.0))]
    path, structure = _write_structure(tmp_path, residues, [_metal(900, "ZN", (0.0, 0.0, 0.0), icode="B")])
    pockets, _report = extract_selected_site_pockets_from_structure(
        structure, structure_id=path.stem, selected_site_ids={(" A ", 900)},
    )
    assert pockets[0].metadata["metal_site_ids"] == [("A", 900, "B")]
    assert pocket_matches_allowed_sites(pockets[0], path, {("p12345", "3.4.24.1", "A_900")})
    for bad_ids in ({"A_900"}, {("A", "900")}, {("", 900)}, {("A", True)}):
        with pytest.raises(ValueError, match="Selected site ids must be"):
            extract_selected_site_pockets_from_structure(
                structure, structure_id=path.stem, selected_site_ids=bad_ids,
            )


def test_summary_validation_reports_counts_and_rejects_silent_collapse(tmp_path):
    clean_rows = [
        ("P12345", "A_900", "3.4.24.1", "Zn", "1"),
        ("P12345", "A_901", "3.4.24.1", "Fe", "1"),
        ("Q99999", "B_10", "1.1.1.1", "Mn", "true"),
        ("Q99999", "B_11", "1.1.1.1", "", "0"),
        ("R00001", "", "2.7.1.1", "Cu", "1"),
    ]
    report = validate_selected_site_summary(_write_summary(tmp_path / "clean.csv", clean_rows))
    assert report["catalytic_column"] == "whether_catalytic"
    assert (report["n_rows"], report["n_catalytic_rows"], report["n_non_catalytic_rows"]) == (5, 4, 1)
    assert report["n_incomplete_catalytic_rows"] == 1
    assert (report["n_site_rows"], report["n_sites"], report["n_structures"], report["n_accessions"]) == (3, 3, 2, 2)
    assert report["metal_type_counts"] == {"FE": 1, "MN": 1, "ZN": 1}

    duplicate = _write_summary(tmp_path / "duplicate.csv", clean_rows + [("p12345", " A_900 ", "3.4.24.1", "ZN", "1")])
    assert len(load_allowed_site_metal_labels(duplicate)) == 3
    with pytest.raises(ValueError, match="1 duplicate site keys"):
        validate_selected_site_summary(duplicate)

    empty_type = _write_summary(tmp_path / "empty_type.csv", clean_rows + [("Q99999", "B_12", "1.1.1.1", " ", "1")])
    assert len(load_allowed_site_metal_labels(empty_type)) == 3
    with pytest.raises(ValueError, match="1 catalytic rows with an empty metaltype"):
        validate_selected_site_summary(empty_type)

    malformed = _write_summary(tmp_path / "malformed.csv", clean_rows + [("Q99999", "B12", "1.1.1.1", "MN", "1")])
    with pytest.raises(ValueError, match="1 malformed chain_resi values"):
        validate_selected_site_summary(malformed)

    # The reader keeps "A _ 902" as written, so the pocket matcher could never label it.
    inner_space_row = ("P12345", "A _ 902", "3.4.24.1", "FE", "1")
    inner_space = _write_summary(tmp_path / "inner_space.csv", clean_rows + [inner_space_row])
    assert ("p12345", "3.4.24.1", "A _ 902") in load_allowed_site_metal_labels(inner_space)
    with pytest.raises(ValueError, match="1 malformed chain_resi values"):
        validate_selected_site_summary(inner_space)

    two_ec = _write_summary(tmp_path / "two_ec.csv", [
        ("P12345", "A_900", "3.4.24.1", "ZN", "1"),
        ("P12345", "A_900", "1.1.1.1", "ZN", "1"),
    ])
    report = validate_selected_site_summary(two_ec)
    assert (report["n_sites"], report["n_structures"], report["n_accessions"]) == (2, 2, 1)
    with pytest.raises(ValueError, match="duplicate site keys"):
        validate_selected_site_summary(two_ec, key_mode="accession")
