"""PROPKA overlay tests use synthetic coordinates and a mocked PROPKA result."""

import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import repair_metal_pka_cache as overlay
from feature_extraction.propka_support import PropkaResidueFeatures, PropkaRunResult


@pytest.fixture(autouse=True)
def installed_propka_version(monkeypatch):
    monkeypatch.setattr(overlay, "version", lambda package: "synthetic-propka-version")


@pytest.fixture
def cache(tmp_path):
    structure = tmp_path / "synthetic__chain_A__EC_1.1.1.1.pdb"
    structure.write_text(
        "ATOM      1  CA  ASP A   1       1.000   2.000   3.000  1.00 20.00           C  \n"
        "ATOM      2  CA  ALA A   2       2.000   2.000   3.000  1.00 20.00           C  \n"
        "ATOM      3  CA  CYS A   4       3.000   2.000   3.000  1.00 20.00           C  \n"
        "HETATM    5 ZN    ZN A   1       1.000   0.000   0.000  1.00 20.00          ZN  \n"
        "TER\n"
        "ATOM      4  CA  HIS B   3       4.000   2.000   3.000  1.00 20.00           C  \n"
        "TER\nEND\n"
    )
    source = tmp_path / "base_cache" / structure.stem / "residue_features.json"
    source.parent.mkdir(parents=True)
    rows = []
    for index, (chain, resseq, icode) in enumerate([("A", 1, ""), ("A", 2, ""),
                                                   ("A", 4, ""), ("B", 3, "")]):
        rows.append(dict(chain_id=chain, resseq=resseq, icode=icode, features={
            "biotite_residue_sasa": 10.5 + index,
            "biotite_residue_sasa_missing": 0.0,
            "custom_charge_distance_proxy": -0.75 + index,
            "custom_charge_distance_proxy_missing": 0.0,
            "dpka_titr": 0.0, "dpka_titr_missing": 1.0,
        }))
    payload = dict(
        schema_version=1, structure_id=structure.stem, source_path=str(structure),
        tooling={"geometry": "biotite", "pka": "unavailable"},
        warnings=["Original PROPKA executable was unavailable"],
        feature_names=["biotite_residue_sasa", "custom_charge_distance_proxy", "dpka_titr"],
        n_residues=len(rows), residues=rows,
    )
    source.write_text(json.dumps(payload, indent=2) + "\n")
    return structure, source, tmp_path / "overlay", payload


@pytest.fixture
def propka(monkeypatch):
    calls = []

    def run(structure, *, ph):
        calls.append((structure, ph))
        return PropkaRunResult(
            residues={
                ("A", 1, "ASP"): PropkaResidueFeatures(-1.25),
                ("A", 4, "CYS"): PropkaResidueFeatures(0.375),
                ("B", 3, "HIS"): PropkaResidueFeatures(0.0),
            }, warnings=["Synthetic successful run"],
        )

    monkeypatch.setattr(overlay, "run_propka_for_structure", run)
    return calls


def test_overlay_changes_only_propka_fields_and_records_original_provenance(cache, propka):
    structure, source, output, before = cache
    original_bytes = source.read_bytes()
    original_structure = structure.read_bytes()
    result = overlay.repair_one(structure, source, output)
    assert propka == [(structure, 7.0)]
    assert not result["reused"] and result["updated_residues"] == 3
    target = Path(result["path"])
    assert target == output / structure.stem / "residue_features.json"
    assert overlay.sha(target) == result["sha256"]
    repaired = json.loads(target.read_text())

    values = [(r["features"]["dpka_titr"], r["features"]["dpka_titr_missing"])
              for r in repaired["residues"]]
    # Unmapped ALA remains missing; mapped zero-valued HIS is a measured zero.
    assert values == [(-1.25, 0.0), (0.0, 1.0), (0.375, 0.0), (0.0, 0.0)]
    for original, changed in zip(before["residues"], repaired["residues"]):
        expected = copy.deepcopy(original)
        for name in ("dpka_titr", "dpka_titr_missing"):
            expected["features"][name] = changed["features"][name]
        assert changed == expected
    for field in ("schema_version", "structure_id", "source_path", "feature_names", "n_residues"):
        assert repaired[field] == before[field]
    assert repaired["tooling"] == {"geometry": "biotite", "pka": "propka"}
    assert repaired["warnings"] == ["Synthetic successful run"]
    provenance = repaired["pka_repair"]
    assert provenance["source_features_sha256"] == overlay.sha(source)
    assert provenance["structure_sha256"] == overlay.sha(structure)
    assert provenance["original_warnings"] == before["warnings"]
    assert provenance["method"] == "existing_geometry_features_plus_propka"
    assert provenance["propka_version"] == "synthetic-propka-version"
    assert provenance["ph"] == 7.0 and provenance["updated_residues"] == 3
    assert provenance["residue_numbering_mode"] == "original_residue_numbers"
    assert source.read_bytes() == original_bytes
    assert structure.read_bytes() == original_structure


def test_reuse_preserves_identical_output_without_another_propka_run(cache, propka):
    structure, source, output, _ = cache
    first = overlay.repair_one(structure, source, output)
    target = Path(first["path"])
    original_bytes = target.read_bytes()
    second = overlay.repair_one(structure, source, output)
    assert second["reused"]
    assert second["sha256"] == first["sha256"]
    assert target.read_bytes() == original_bytes
    assert len(propka) == 1


@pytest.mark.parametrize("changed_input", ["features", "structure"])
def test_changed_input_invalidates_existing_overlay_without_overwriting_it(cache, propka, changed_input):
    structure, source, output, _ = cache
    first = overlay.repair_one(structure, source, output)
    target = Path(first["path"])
    original_overlay = target.read_bytes()
    if changed_input == "features":
        payload = json.loads(source.read_text())
        payload["residues"][0]["features"]["biotite_residue_sasa"] += 1
        source.write_text(json.dumps(payload))
    else:
        structure.write_text(structure.read_text().replace("1.000", "1.001", 1))
    with pytest.raises(ValueError, match="Incompatible existing overlay"):
        overlay.repair_one(structure, source, output)
    assert target.read_bytes() == original_overlay
    assert len(propka) == 1


@pytest.mark.parametrize("tampered_field,value", [
    ("ph", 6.0),
    ("method", "different_method"),
    ("updated_residues", 0),
    ("nonfinite_feature", float("nan")),
])
def test_incompatible_or_nonfinite_overlay_is_not_reused(cache, propka, tampered_field, value):
    structure, source, output, _ = cache
    first = overlay.repair_one(structure, source, output)
    target = Path(first["path"])
    payload = json.loads(target.read_text())
    if tampered_field == "nonfinite_feature":
        payload["residues"][0]["features"]["dpka_titr"] = value
    else:
        payload["pka_repair"][tampered_field] = value
    target.write_text(json.dumps(payload))
    tampered_bytes = target.read_bytes()
    with pytest.raises(ValueError, match="Incompatible existing overlay"):
        overlay.repair_one(structure, source, output)
    assert target.read_bytes() == tampered_bytes
    assert len(propka) == 1


@pytest.mark.parametrize("bad_result,match", [
    ({}, "No PROPKA residue results"),
    ({("Z", 999, "ASP"): PropkaResidueFeatures(1.0)}, "did not match"),
    ({("A", 1, "ASP"): PropkaResidueFeatures(float("nan"))}, "Nonfinite"),
    ({("A", 1, "ASP"): PropkaResidueFeatures(float("inf"))}, "Nonfinite"),
])
def test_invalid_propka_output_never_writes_overlay_or_changes_original(cache, monkeypatch, bad_result, match):
    structure, source, output, _ = cache
    original_bytes = source.read_bytes()
    monkeypatch.setattr(overlay, "run_propka_for_structure", lambda *args, **kwargs:
                        PropkaRunResult(residues=bad_result, warnings=[]))
    with pytest.raises(ValueError, match=match):
        overlay.repair_one(structure, source, output)
    assert not (output / structure.stem / "residue_features.json").exists()
    assert source.read_bytes() == original_bytes


@pytest.fixture
def insertion_cache(cache):
    structure, source, output, payload = cache
    lines = structure.read_text().splitlines(keepends=True)
    for index, icode in ((0, " "), (2, "C")):
        lines[index] = lines[index][:17] + "GLU" + lines[index][20:22] + " 254" + icode + lines[index][27:]
        payload["residues"][index]["resseq"] = 254
        payload["residues"][index]["icode"] = icode.strip()
    structure.write_text("".join(lines))
    source.write_text(json.dumps(payload))
    return structure, source, output, payload


def test_insertions_get_independent_propka_features_with_unchanged_atoms(insertion_cache, monkeypatch):
    structure, source, output, before = insertion_cache
    original_structure = structure.read_bytes()
    original_cache = source.read_bytes()
    calls = []

    def run(temporary_structure, *, ph):
        assert temporary_structure != structure
        assert ph == 7.0
        calls.append(temporary_structure)
        original_lines = structure.read_text().splitlines()
        temporary_lines = temporary_structure.read_text().splitlines()
        assert len(original_lines) == len(temporary_lines)
        for original, temporary in zip(original_lines, temporary_lines):
            if original.startswith("ATOM"):
                assert temporary[:22] == original[:22]
                assert temporary[27:] == original[27:]
                assert not temporary[26].strip()
            else:
                assert temporary == original
        parsed = overlay.parse_structure_file(str(temporary_structure))
        identifiers = [(chain.id, residue.id[1], residue.id[2], residue.resname)
                       for chain in next(parsed.get_models()) for residue in chain if residue.id[0] == " "]
        assert identifiers == [("A", 1, " ", "GLU"), ("A", 2, " ", "ALA"),
                               ("A", 3, " ", "GLU"), ("B", 1, " ", "HIS")]
        return PropkaRunResult(residues={
            ("A", 1, "GLU"): PropkaResidueFeatures(-1.25),
            ("A", 3, "GLU"): PropkaResidueFeatures(0.375),
            ("B", 1, "HIS"): PropkaResidueFeatures(0.0),
        }, warnings=[])

    monkeypatch.setattr(overlay, "run_propka_for_structure", run)
    result = overlay.repair_one(structure, source, output)
    repaired = json.loads(Path(result["path"]).read_text())
    rows = {(row["chain_id"], row["resseq"], row["icode"]): row for row in repaired["residues"]}
    assert rows[("A", 254, "")]["features"]["dpka_titr"] == -1.25
    assert rows[("A", 254, "C")]["features"]["dpka_titr"] == 0.375
    assert rows[("A", 254, "")]["features"]["dpka_titr_missing"] == 0
    assert rows[("A", 254, "C")]["features"]["dpka_titr_missing"] == 0
    provenance = repaired["pka_repair"]
    assert provenance["residue_numbering_mode"] == "temporary_unique_residue_numbers_for_insertions"
    mapping = provenance["residue_numbering_map"]
    assert provenance["residue_numbering_map_sha256"] == overlay.numbering_metadata(mapping)["residue_numbering_map_sha256"]
    assert [row["temporary_resseq"] for row in mapping if row["resname"] == "GLU"] == [1, 3]
    assert all(not p.exists() for p in calls)
    assert structure.read_bytes() == original_structure and source.read_bytes() == original_cache
    for original, changed in zip(before["residues"], repaired["residues"]):
        for feature in ("biotite_residue_sasa", "custom_charge_distance_proxy"):
            assert original["features"][feature] == changed["features"][feature]
    assert overlay.repair_one(structure, source, output)["reused"]
    assert len(calls) == 1
    provenance.pop("residue_numbering_map_sha256")
    Path(result["path"]).write_text(json.dumps(repaired))
    with pytest.raises(ValueError, match="Incompatible existing overlay"):
        overlay.repair_one(structure, source, output)


def test_refresh_selection_includes_insertion_codes(insertion_cache):
    structure, _, _, _ = insertion_cache
    assert overlay.needs_numbering_refresh(structure)


def test_unmeasurable_insertion_does_not_borrow_same_number_residues_value(insertion_cache, monkeypatch):
    structure, source, output, _ = insertion_cache
    monkeypatch.setattr(overlay, "run_propka_for_structure", lambda *args, **kwargs:
                        PropkaRunResult(residues={("A", 1, "GLU"): PropkaResidueFeatures(-0.22)},
                                        warnings=["GLU 3 lacks a titratable sidechain"]))
    result = overlay.repair_one(structure, source, output)
    payload = json.loads(Path(result["path"]).read_text())
    ordinary, inserted = payload["residues"][0], payload["residues"][2]
    assert ordinary["features"]["dpka_titr"] == -0.22
    assert ordinary["features"]["dpka_titr_missing"] == 0
    assert inserted["features"]["dpka_titr"] == 0
    assert inserted["features"]["dpka_titr_missing"] == 1
    assert payload["pka_repair"]["updated_residues"] == 1


def test_ordinary_residue_numbers_do_not_need_refresh(cache):
    structure, _, _, _ = cache
    assert not overlay.needs_numbering_refresh(structure)
