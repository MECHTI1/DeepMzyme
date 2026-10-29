"""Pocket grouping of selected (catalytic) metal sites, ``--selected-site-pocket-grouping`` (synthetic only).

``cluster`` (the default) is the selected-site extraction as it was: selected metals within the
multinuclear merge distance form one pocket, so a structure can give several, overlapping pockets.
``structure`` builds exactly one pocket per structure around all its selected metals; unselected
metals (other sites, another donor's copy of a selected site) never enter it. The golden digests
below were taken from the extraction and loader outputs before the option existed, so the default
stays byte-identical. Every structure, summary and fit is synthetic; no real data is read.
"""
from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from ec_fixtures import default_ec_proteins, tiny_ec_argv, write_ec_dataset  # noqa: E402
from graph.structure_parsing import (  # noqa: E402
    extract_metal_pockets_from_structure, extract_selected_site_pockets_from_structure,
)
from test_selected_site_extraction import STEM, _metal, _residue, _write_structure  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic structures and a canonical form of the outputs
# ---------------------------------------------------------------------------

def _two_sites(folder: Path, separation: float):
    """Selected ZN A_900 at the origin and selected ZN A_901 ``separation`` Å along x.

    Residues 1 and 2 bind A_900 and A_901; residue 3 lies near A_900 only, residue 4 near A_901
    only (each 9 Å away), and residue 5 near neither.
    """
    residues = [
        _residue(1, "HIS", (0.0, 4.0, 0.0), ("NE2", "N", (0.0, 2.1, 0.0))),
        _residue(2, "HIS", (separation, 4.0, 0.0), ("NE2", "N", (separation, 2.1, 0.0))),
        _residue(3, "ALA", (-9.0, 0.0, 0.0)),
        _residue(4, "ALA", (separation + 9.0, 0.0, 0.0)),
        _residue(5, "ALA", (separation / 2.0, 0.0, 30.0)),
    ]
    metals = [_metal(900, "ZN", (0.0, 0.0, 0.0)), _metal(901, "ZN", (separation, 0.0, 0.0))]
    return _write_structure(folder, residues, metals)


def _copy_and_unselected(folder: Path):
    """Selected ZN A_900 and A_901 (6 Å apart), another donor's copy of A_900 (ZN A_950, 1 Å away) and an
    unselected FE A_960 30 Å away.

    Residue 3 lies 9.6 Å from the copy but 10.6 Å from A_900; residue 4 binds only the FE.
    """
    residues = [
        _residue(1, "HIS", (0.0, 4.0, 0.0), ("NE2", "N", (0.0, 2.1, 0.0))),
        _residue(2, "HIS", (6.0, 4.0, 0.0), ("NE2", "N", (6.0, 2.1, 0.0))),
        _residue(3, "ALA", (-10.6, 0.0, 0.0)),
        _residue(4, "ASP", (30.0, 4.0, 0.0), ("OD1", "O", (30.0, 2.1, 0.0))),
    ]
    metals = [_metal(900, "ZN", (0.0, 0.0, 0.0)), _metal(901, "ZN", (6.0, 0.0, 0.0)),
              _metal(950, "ZN", (-1.0, 0.0, 0.0)), _metal(960, "FE", (30.0, 0.0, 0.0))]
    return _write_structure(folder, residues, metals)


def _bridge_and_bare_site(folder: Path):
    """Selected A_900 and A_902 joined only through the unselected A_901, and a selected A_903 with no residue
    within the pocket radius."""
    residues = [
        _residue(1, "ALA", (-9.5, 0.0, 0.0)),
        _residue(2, "ALA", (0.0, 5.0, 0.0)),
        _residue(3, "ALA", (8.0, 5.0, 0.0)),
        _residue(4, "ALA", (17.5, 0.0, 0.0)),
    ]
    metals = [_metal(900, "ZN", (0.0, 0.0, 0.0)), _metal(901, "ZN", (4.0, 0.0, 0.0)),
              _metal(902, "ZN", (8.0, 0.0, 0.0)), _metal(903, "ZN", (60.0, 0.0, 0.0))]
    return _write_structure(folder, residues, metals)


SELECTED = {
    "six": {("A", 900), ("A", 901)},
    "twenty_five": {("A", 900), ("A", 901)},
    "copy_and_unselected": {("A", 900), ("A", 901)},
    "single_with_copy": {("A", 900)},
    "bridge_and_bare_site": {("A", 900), ("A", 902), ("A", 903)},
}


def extraction_cases(base: Path) -> dict[str, tuple]:
    """Case name -> (structure, selected site IDs); each structure in its own folder under ``base``."""
    builders = {
        "six": lambda folder: _two_sites(folder, 6.0),
        "twenty_five": lambda folder: _two_sites(folder, 25.0),
        "copy_and_unselected": _copy_and_unselected,
        "single_with_copy": _copy_and_unselected,
        "bridge_and_bare_site": _bridge_and_bare_site,
    }
    cases = {}
    for name, build in builders.items():
        folder = base / name
        folder.mkdir(parents=True)
        _path, structure = build(folder)
        cases[name] = (structure, SELECTED[name])
    return cases


def canonical(value, roots: tuple[tuple[str, str], ...] = ()):
    """A JSON-ready form that keeps every field, type, value and order; ``roots`` replace path prefixes."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return [type(value).__name__, [[field.name, canonical(getattr(value, field.name), roots)]
                                       for field in dataclasses.fields(value)]]
    if isinstance(value, torch.Tensor):
        return ["tensor", str(value.dtype), list(value.shape), value.detach().flatten().tolist()]
    if isinstance(value, np.ndarray):
        return ["ndarray", str(value.dtype), list(value.shape), value.flatten().tolist()]
    if isinstance(value, np.generic):
        return canonical(value.item(), roots)
    if isinstance(value, dict):
        return ["dict", [[canonical(key, roots), canonical(item, roots)] for key, item in value.items()]]
    if isinstance(value, (set, frozenset)):
        return [type(value).__name__, sorted((canonical(item, roots) for item in value), key=repr)]
    if isinstance(value, (list, tuple)):
        return [type(value).__name__, [canonical(item, roots) for item in value]]
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        for name, root in roots:
            value = value.replace(root, name)
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise TypeError(f"No canonical form for {type(value).__name__}")


def digest(value, roots: tuple[tuple[str, str], ...] = ()) -> str:
    return hashlib.sha256(json.dumps(canonical(value, roots)).encode("utf-8")).hexdigest()


def cluster_extraction_digests(base: Path) -> dict[str, str]:
    """Digest of (pockets, report) of the default selected-site extraction, and of the component extraction."""
    result = {}
    for name, (structure, selected) in extraction_cases(base).items():
        result[name] = digest(extract_selected_site_pockets_from_structure(
            structure, structure_id=STEM, selected_site_ids=selected))
        result[f"{name}__component"] = digest(extract_metal_pockets_from_structure(structure, structure_id=STEM))
    return result


LOADER_PROTEINS = {
    "Q1000": {"ec": "3.1.1.1", "ions": [("A", 900, "", "ZN", [0.0, 0.0, 0.0]), ("A", 901, "", "ZN", [6.0, 0.0, 0.0])]},
    "Q2000": {"ec": "1.1.1.1", "ions": [("A", 900, "", "ZN", [0.0, 0.0, 0.0]), ("A", 901, "", "ZN", [25.0, 0.0, 0.0])]},
    "Q3000": {"ec": "2.1.1.1",
              "ions": [("A", 900, "", "ZN", [0.0, 0.0, 0.0]), ("A", 950, "", "ZN", [1.0, 0.0, 0.0]),
                       ("A", 960, "", "FE", [30.0, 0.0, 0.0])],
              "summary_sites": [("A", 900, "ZN")]},
}


def load_pockets(paths: dict, **options):
    from training.data import load_training_pockets_with_report_from_dir

    return load_training_pockets_with_report_from_dir(
        paths["structures"], required_targets=("ec",), summary_csv=paths["summary"],
        esm_embeddings_dir=paths["esm"], esm_dim=8, require_esm_embeddings=False,
        external_features_root_dir=paths["external"], require_external_features=False, load_workers=1, **options)


def loader_digests(base: Path) -> dict[str, str]:
    """Digest of the pockets and extraction summary of the trainer's loader in both site-extraction modes."""
    paths = write_ec_dataset(base / "data", LOADER_PROTEINS)
    roots = (("<DATA>", str(paths["root"])), ("<REPO>", str(ROOT)))
    result = {}
    for mode in ("component", "selected_sites"):
        loaded = load_pockets(paths, site_extraction=mode)
        result[mode] = digest([loaded.pockets, loaded.feature_report.get("site_extraction_summary"),
                               loaded.feature_report["skipped_pockets"]], roots)
    return result


CONFIG_ARGVS = {"metal_default": [], "ec_default": ["--task", "ec"],
                "ec_selected": ["--task", "ec", "--site-extraction", "selected_sites"]}

# Taken with the code before the grouping option (main at 36498cd) by calling the functions above
# (the configuration digests over ``config_to_payload(parse_args(argv))`` with ``_config_roots()``).
GOLDEN_EXTRACTION_DIGESTS = {
    "bridge_and_bare_site": "74c3c8b639bd7a7e36b511d019890498d4acf9c785624ef424544ee06511d899",
    "bridge_and_bare_site__component": "55421c8a8dd5c832c05202c41aa81e0e741382a5eada80002d42343ea25830c1",
    "copy_and_unselected": "4567b7727309ce75849aca8a74f2d3122eaac6f69fe27347fdba3749d633fd10",
    "copy_and_unselected__component": "cb178d7a38dbe3def1e66caf17b5523f02c3676dec585a5eef8a05d313e25211",
    "single_with_copy": "445dd47640fc2434eeed9c8f14727f5ebbcda20862b5d3351875bef48d8160c0",
    "single_with_copy__component": "cb178d7a38dbe3def1e66caf17b5523f02c3676dec585a5eef8a05d313e25211",
    "six": "94ea304e1eb26ff4fbeedbe95cecca2714fb06ae0e1a12d6d6979aa399c3870e",
    "six__component": "c571d69ba669957f82fc5b0be4dcda2728205f9050f598b430c48a57cb4e2ca5",
    "twenty_five": "c50f296e8b77341d25c26d4db731282361407b7cfa75a19c703c361d3b31f29e",
    "twenty_five__component": "99b517490e786b79cb41d64117d3be41e3c3a7ff131adbc1d75a173d026e70dd",
}
GOLDEN_LOADER_DIGESTS = {
    "component": "a4fbbbbe85e41f6481c756a90ba4a325a9a94e81f007afbc61d309d44387371d",
    "selected_sites": "2faa4d64a7d8096eaedaa926e7ad8a77ec63a2071e4cbccd3b8e8c5633408e98",
}
GOLDEN_CONFIG_DIGESTS = {
    "metal_default": "3f54e517692c087439a75b751554e908eb485498e1775861fc1fa6c49e64273f",
    "ec_default": "1a2c6e4050524aeb460a1032e86816604eddc5f116ab97832b60a55e92c4f3e1",
    "ec_selected": "28089a4e0b6d34590b59e802437bdd5144a898375afce6775756ce277fc618d6",
}


def _config_roots() -> tuple[tuple[str, str], ...]:
    """Default data paths depend on the checkout and on the media data root setting."""
    import project_paths

    return (("<MEDIA>", str(project_paths.MEDIA_DATA_ROOT)), ("<REPO>", str(project_paths.PROJECT_ROOT)))


def _residue_numbers(pocket) -> list[int]:
    return [residue.resseq for residue in pocket.residues]


def _extract(case, grouping=None):
    structure, selected = case
    options = {} if grouping is None else {"pocket_grouping": grouping}
    return extract_selected_site_pockets_from_structure(structure, structure_id=STEM, selected_site_ids=selected,
                                                        **options)


# ---------------------------------------------------------------------------
# The default stays as it was
# ---------------------------------------------------------------------------

def test_default_extraction_is_byte_identical(tmp_path):
    assert cluster_extraction_digests(tmp_path / "default") == GOLDEN_EXTRACTION_DIGESTS
    explicit = {name: digest(_extract(case, "cluster"))
                for name, case in extraction_cases(tmp_path / "explicit").items()}
    assert explicit == {name: value for name, value in GOLDEN_EXTRACTION_DIGESTS.items() if "__" not in name}


def test_default_loader_is_byte_identical(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPMZYME_PARSE_CACHE_DIR", raising=False)
    # The pockets record the default RING edge path (a string only; nothing there is read or created).
    monkeypatch.delenv("RING_FEATURES_DIR", raising=False)
    monkeypatch.delenv("RING_EDGES_DIR", raising=False)
    assert loader_digests(tmp_path) == GOLDEN_LOADER_DIGESTS
    paths = write_ec_dataset(tmp_path / "explicit", LOADER_PROTEINS)
    loaded = load_pockets(paths, site_extraction="selected_sites", selected_site_pocket_grouping="cluster")
    roots = (("<DATA>", str(paths["root"])), ("<REPO>", str(ROOT)))
    assert digest([loaded.pockets, loaded.feature_report.get("site_extraction_summary"),
                   loaded.feature_report["skipped_pockets"]], roots) == GOLDEN_LOADER_DIGESTS["selected_sites"]


def test_default_configuration_gains_only_the_grouping_key():
    from training.config import config_to_payload, parse_args

    for name, argv in CONFIG_ARGVS.items():
        payload = config_to_payload(parse_args(argv))
        assert payload.pop("selected_site_pocket_grouping") == "cluster"
        assert digest(payload, _config_roots()) == GOLDEN_CONFIG_DIGESTS[name], name


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name, cluster_residues", [
    ("six", [[1, 2, 3], [1, 2, 4]]),
    ("twenty_five", [[1, 3], [2, 4]]),
])
def test_two_sites_give_one_pocket_under_structure_grouping(tmp_path, name, cluster_residues):
    case = extraction_cases(tmp_path)[name]

    cluster_pockets, cluster_report = _extract(case)
    assert [_residue_numbers(pocket) for pocket in cluster_pockets] == cluster_residues
    assert [pocket.pocket_id for pocket in cluster_pockets] == [f"{STEM}_METAL_0", f"{STEM}_METAL_1"]
    assert cluster_report["selected_cluster_count"] == cluster_report["pocket_count"] == 2
    assert all("selected_site_pocket_grouping" not in pocket.metadata for pocket in cluster_pockets)

    pockets, report = _extract(case, "structure")
    assert len(pockets) == 1
    pocket = pockets[0]
    assert pocket.pocket_id == f"{STEM}_METAL_0"
    assert _residue_numbers(pocket) == [1, 2, 3, 4]
    assert pocket.metal_count() == 2 and pocket.metal_element == "ZN"
    assert pocket.metadata["metal_site_ids"] == [("A", 900, ""), ("A", 901, "")]
    assert pocket.metadata["selected_site_keys"] == [("A", 900), ("A", 901)]
    assert pocket.metadata["selected_site_pocket_grouping"] == "structure"
    assert pocket.metadata["site_extraction"] == "selected_sites"
    assert (report["selected_cluster_count"], report["pocket_count"], report["cluster_grouping_cluster_count"]) == (
        1, 1, 2)
    assert report["pocket_grouping"] == "structure"
    assert report["split_components"] == [] and report["split_component_count"] == 0
    assert report["selected_sites_without_pocket"] == report["selected_sites_without_residues"] == []
    row = report["pockets"][0]
    assert row["legacy_pocket_ids"] == [f"{STEM}_METAL_0", f"{STEM}_METAL_1"]
    assert (row["residue_count"], row["legacy_residue_count"], row["residue_count_delta"]) == (4, 4, 0)
    # The legacy-component diagnostics do not depend on the grouping.
    for key in ("metal_records_total", "metal_records_selected", "metal_records_dropped", "legacy_component_count",
                "components_with_unselected_metals_count", "residue_count_delta_total"):
        assert report[key] == cluster_report[key], key


def test_unselected_metal_and_same_site_copy_never_enter_the_pocket(tmp_path):
    case = extraction_cases(tmp_path)["copy_and_unselected"]

    cluster_pockets, cluster_report = _extract(case)
    # The default gives two pockets with the same residues: the duplicate the structure grouping removes.
    assert [_residue_numbers(pocket) for pocket in cluster_pockets] == [[1, 2], [1, 2]]

    pockets, report = _extract(case, "structure")
    assert len(pockets) == 1
    pocket = pockets[0]
    assert _residue_numbers(pocket) == [1, 2]  # residue 3 (near the copy only) and 4 (near the FE only) stay out
    assert pocket.metadata["metal_site_ids"] == [("A", 900, ""), ("A", 901, "")]
    assert set(pocket.metadata["metal_site_coord_map"]) == {("A", 900, ""), ("A", 901, "")}
    assert torch.equal(torch.stack(pocket.metal_coords), torch.tensor([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]]))
    assert pocket.metadata["legacy_component_site_ids"] == [("A", 900, ""), ("A", 950, ""), ("A", 901, "")]
    assert (report["metal_records_total"], report["metal_records_selected"], report["metal_records_dropped"]) == (
        4, 2, 2)
    assert report["components_with_unselected_metals_count"] == 1
    assert report["components_with_unselected_metals"][0]["unselected_site_ids"] == [("A", 950, "")]
    assert report["legacy_components_without_selected_sites"] == 1  # the FE
    row = report["pockets"][0]
    assert (row["residue_count"], row["legacy_residue_count"], row["residue_count_delta"]) == (2, 3, -1)
    assert report["residue_count_delta_total"] == cluster_report["residue_count_delta_total"] == -1

    legacy = extract_metal_pockets_from_structure(case[0], structure_id=STEM)
    assert 3 in _residue_numbers(legacy[0]) and legacy[0].metal_count() == 2


def test_bridged_sites_and_a_site_without_residues(tmp_path):
    case = extraction_cases(tmp_path)["bridge_and_bare_site"]

    cluster_pockets, cluster_report = _extract(case)
    assert [_residue_numbers(pocket) for pocket in cluster_pockets] == [[1, 2, 3], [2, 3, 4]]
    assert cluster_report["split_component_count"] == 1
    assert cluster_report["selected_sites_without_pocket"] == [("A", 903)]

    pockets, report = _extract(case, "structure")
    assert [_residue_numbers(pocket) for pocket in pockets] == [[1, 2, 3, 4]]
    assert pockets[0].metadata["metal_site_ids"] == [("A", 900, ""), ("A", 902, ""), ("A", 903, "")]
    assert ("A", 901, "") not in pockets[0].metadata["metal_site_coord_map"]
    assert report["selected_sites_without_residues"] == [("A", 903)]
    assert report["selected_sites_without_pocket"] == [] and report["split_component_count"] == 0
    assert (report["selected_cluster_count"], report["cluster_grouping_cluster_count"]) == (1, 3)
    assert report["pockets"][0]["legacy_pocket_ids"] == [f"{STEM}_METAL_0", f"{STEM}_METAL_1"]
    assert report["residue_count_delta_total"] == cluster_report["residue_count_delta_total"]


def test_structure_grouping_without_a_pocket_or_without_sites(tmp_path):
    structure, _selected = extraction_cases(tmp_path)["bridge_and_bare_site"]

    pockets, report = _extract((structure, {("A", 903)}), "structure")
    assert pockets == [] and report["pocket_count"] == 0 and report["selected_cluster_count"] == 1
    assert report["selected_sites_without_pocket"] == report["selected_sites_without_residues"] == [("A", 903)]
    assert report["pockets"] == []

    pockets, report = _extract((structure, set()), "structure")
    assert pockets == [] and report["no_selected_sites"] is True
    assert (report["selected_cluster_count"], report["cluster_grouping_cluster_count"]) == (0, 0)

    with pytest.raises(ValueError, match="Unsupported selected-site pocket grouping"):
        _extract((structure, {("A", 900)}), "protein")


# ---------------------------------------------------------------------------
# Loader, parse cache and raw graph cache
# ---------------------------------------------------------------------------

def test_loader_gives_one_pocket_per_structure_and_keys_the_parse_cache(tmp_path, monkeypatch):
    import label_schemes
    import training.data as data
    from training.parallel_loading import resolve_parse_cache_dir

    monkeypatch.setenv("DEEPMZYME_PARSE_CACHE_DIR", str(tmp_path / "parse_cache"))
    seen: list[dict] = []
    original = data.iter_structure_load_results

    def spy(structure_files, load_kwargs, *, workers):
        seen.append(dict(load_kwargs))
        return original(structure_files, load_kwargs, workers=workers)

    monkeypatch.setattr(data, "iter_structure_load_results", spy)
    paths = write_ec_dataset(tmp_path / "data", LOADER_PROTEINS)
    cluster = load_pockets(paths, site_extraction="selected_sites")
    grouped = load_pockets(paths, site_extraction="selected_sites", selected_site_pocket_grouping="structure")

    assert "selected_site_pocket_grouping" not in seen[0]
    assert seen[1]["selected_site_pocket_grouping"] == "structure"
    scheme = label_schemes.ACTIVE_METAL_LABEL_SCHEME
    assert resolve_parse_cache_dir(seen[0], scheme) != resolve_parse_cache_dir(seen[1], scheme)

    assert len(cluster.pockets) == 5  # 2 + 2 + 1 cluster pockets, cached first
    assert [pocket.structure_id for pocket in grouped.pockets] == sorted(paths["stems"].values())
    assert all(pocket.metadata["selected_site_pocket_grouping"] == "structure" for pocket in grouped.pockets)
    by_accession = {pocket.structure_id.split("__")[0]: pocket for pocket in grouped.pockets}
    assert by_accession["Q3000"].metadata["metal_site_ids"] == [("A", 900, "")]
    assert [pocket.metal_count() for pocket in grouped.pockets] == [2, 2, 1]
    summary = grouped.feature_report["site_extraction_summary"]
    assert summary["selected_site_pocket_grouping"] == "structure"
    assert summary["n_structures"] == summary["n_pockets"] == summary["selected_cluster_count"] == 3
    assert (summary["cluster_grouping_cluster_count"], summary["structures_with_merged_clusters"]) == (5, 2)
    assert (summary["metal_records_dropped"], summary["structures_with_unselected_metals"]) == (2, 1)
    assert summary["selected_sites_without_residues"] == 0
    assert "selected_site_pocket_grouping" not in cluster.feature_report["site_extraction_summary"]


def test_summary_counts_every_loaded_pocket(tmp_path):
    """``n_pockets`` counts pockets, not structures, so a second pocket of one structure shows (and fails the check)."""
    from ec_clean30.commands import one_pocket_per_protein
    from training.data import site_extraction_summary

    paths = write_ec_dataset(tmp_path / "data", LOADER_PROTEINS)
    pockets = list(load_pockets(paths, site_extraction="selected_sites",
                                selected_site_pocket_grouping="structure").pockets)
    assert one_pocket_per_protein(site_extraction_summary(pockets, [], "structure"), 3)[0]
    second = dataclasses.replace(pockets[0], pocket_id=f"{pockets[0].structure_id}_METAL_1")
    summary = site_extraction_summary([*pockets, second], [], "structure")
    assert (summary["n_pockets"], summary["n_structures"], summary["selected_cluster_count"]) == (4, 3, 3)
    ok, detail = one_pocket_per_protein(summary, 3)
    assert not ok and (detail["pockets"], detail["structures"], detail["loaded"]) == (4, 3, 3)
    assert not one_pocket_per_protein(site_extraction_summary(pockets, [], "structure"), 4)[0]
    assert "n_pockets" not in site_extraction_summary(pockets, [])


def test_loader_refuses_two_pockets_of_one_structure(tmp_path, monkeypatch):
    import training.data as data

    monkeypatch.delenv("DEEPMZYME_PARSE_CACHE_DIR", raising=False)
    original = data.iter_structure_load_results

    def with_a_second_pocket(structure_files, load_kwargs, *, workers):
        for structure_path, (status, outcome) in original(structure_files, load_kwargs, workers=workers):
            if status == "ok" and outcome[0]:
                pocket = outcome[0][0]
                outcome = ([*outcome[0], dataclasses.replace(pocket, pocket_id=f"{pocket.structure_id}_METAL_1")],
                           *outcome[1:])
            yield structure_path, (status, outcome)

    monkeypatch.setattr(data, "iter_structure_load_results", with_a_second_pocket)
    paths = write_ec_dataset(tmp_path / "data", LOADER_PROTEINS)
    with pytest.raises(ValueError, match="several pockets for 3 structure"):
        load_pockets(paths, site_extraction="selected_sites", selected_site_pocket_grouping="structure")


def test_loader_refusals(tmp_path):
    from training.structure_loading import load_structure_pockets

    paths = write_ec_dataset(tmp_path / "data", LOADER_PROTEINS)
    with pytest.raises(ValueError, match="needs site_extraction='selected_sites'"):
        load_pockets(paths, selected_site_pocket_grouping="structure")
    with pytest.raises(ValueError, match="Unsupported selected-site pocket grouping"):
        load_pockets(paths, site_extraction="selected_sites", selected_site_pocket_grouping="protein")
    common = dict(structure_path=Path("x.pdb"), structure_root=Path("."), allowed_site_metal_labels={}, esm_dim=8,
                  embeddings_dir=Path("."), require_esm_embeddings=False, feature_root_dir=Path("."),
                  external_feature_source="auto", require_external_features=False)
    with pytest.raises(ValueError, match="needs selected-site extraction"):
        load_structure_pockets(selected_site_pocket_grouping="structure", **common)
    with pytest.raises(ValueError, match="Unsupported selected-site pocket grouping"):
        load_structure_pockets(site_extraction="selected_sites", selected_site_pocket_grouping="x", **common)


def test_raw_graph_cache_never_reuses_a_cluster_graph(tmp_path, monkeypatch):
    from torch_geometric.data import Data

    from training.raw_graph_cache import GRAPH_CACHE_ENV, build_cached_graphs

    case = extraction_cases(tmp_path / "structures")["single_with_copy"]
    cluster, _ = _extract(case)
    grouped, _ = _extract(case, "structure")
    # One selected site: the same residues, ion and pocket ID; only the grouping marker differs.
    assert [residue.residue_id() for residue in cluster[0].residues] == \
        [residue.residue_id() for residue in grouped[0].residues]
    assert torch.equal(torch.stack(cluster[0].metal_coords), torch.stack(grouped[0].metal_coords))
    assert cluster[0].pocket_id == grouped[0].pocket_id

    monkeypatch.setenv(GRAPH_CACHE_ENV, str(tmp_path / "graph_cache"))
    built: list[str] = []

    def builder(pocket, **_options):
        built.append(pocket.metadata.get("selected_site_pocket_grouping", "cluster"))
        return Data(x=torch.zeros(len(pocket.residues), 1))

    for pockets in (cluster, cluster, grouped):
        build_cached_graphs(pockets, {"esm_dim": 8}, builder)
    assert built == ["cluster", "structure"]  # the second cluster call is a cache hit, the structure call is not


# ---------------------------------------------------------------------------
# Trainer configuration
# ---------------------------------------------------------------------------

def test_configuration_option_refusals_and_identity():
    from ec_clean30.commands import resolved_config_sha256
    from training.config import TrainConfig, config_to_payload, parse_args
    from training.run import validate_training_configuration

    selected = ["--task", "ec", "--site-extraction", "selected_sites"]
    structure = [*selected, "--selected-site-pocket-grouping", "structure"]
    assert parse_args(selected).selected_site_pocket_grouping == "cluster"
    assert parse_args(structure).selected_site_pocket_grouping == "structure"
    assert config_to_payload(parse_args(structure))["selected_site_pocket_grouping"] == "structure"
    for argv in (["--task", "ec", "--selected-site-pocket-grouping", "structure"],
                 [*selected, "--selected-site-pocket-grouping", "protein"]):
        with pytest.raises(SystemExit):
            parse_args(argv)
    with pytest.raises(ValueError, match="structure requires --site-extraction selected_sites"):
        validate_training_configuration(TrainConfig(task="ec", selected_site_pocket_grouping="structure"))
    with pytest.raises(ValueError, match="Unsupported selected-site pocket grouping"):
        validate_training_configuration(TrainConfig(task="ec", site_extraction="selected_sites",
                                                    selected_site_pocket_grouping="protein"))
    assert resolved_config_sha256(selected) != resolved_config_sha256(structure)


def test_configuration_choices_match_the_parser_constants():
    import graph.structure_parsing as parsing
    from ec_clean30 import cohort, commands
    from training import config

    assert config.VALID_SELECTED_SITE_POCKET_GROUPING_CHOICES == parsing.SELECTED_SITE_POCKET_GROUPINGS
    assert config.DEFAULT_SELECTED_SITE_POCKET_GROUPING == parsing.POCKET_GROUPING_CLUSTER
    assert commands.POCKET_GROUPING == cohort.SELECTED_SITE_POCKET_GROUPING == parsing.POCKET_GROUPING_STRUCTURE


def test_held_out_evaluation_loads_with_the_configured_grouping(tmp_path, monkeypatch):
    """The trainer's own held-out pass (never used by campaign units) loads with the run's grouping; nothing is read."""
    import types

    import training.run as run
    from training.config import TrainConfig

    class Stop(Exception):
        pass

    seen: dict = {}

    def spy(**kwargs):
        seen.update(kwargs)
        raise Stop

    monkeypatch.setattr(run, "validate_held_out_structure_disjointness", lambda config: None)
    monkeypatch.setattr(run, "load_training_pockets_with_report_from_dir", spy)
    config = TrainConfig(task="ec", site_extraction="selected_sites", selected_site_pocket_grouping="structure",
                         run_test_eval=True, test_structure_dir=tmp_path / "blind_structures",
                         test_summary_csv=tmp_path / "blind_sites.csv")
    prepared = types.SimpleNamespace(model=torch.nn.Linear(1, 1), ec_label_to_index={})
    with pytest.raises(Stop):
        run.evaluate_held_out_test_split(prepared, config, checkpoint=None)
    assert (seen.get("site_extraction"), seen.get("selected_site_pocket_grouping")) == ("selected_sites", "structure")


# ---------------------------------------------------------------------------
# A fit with structure grouping: export, replay and the label-free test pass
# ---------------------------------------------------------------------------

def _second_site_proteins(proteins: dict[str, dict]) -> dict[str, dict]:
    """Each protein gets a second zinc, 6 Å (even positions) or 25 Å (odd) from the first: two clusters each."""
    result = {}
    for index, (accession, spec) in enumerate(sorted(proteins.items())):
        chain, resseq, icode, element, xyz = spec["ions"][0]
        offset = [6.0, 0.0, 0.0] if index % 2 == 0 else [0.0, 0.0, 25.0]
        second = (chain, resseq + 1, icode, element, [value + delta for value, delta in zip(xyz, offset)])
        result[accession] = {**spec, "ions": [spec["ions"][0], second]}
    return result


@pytest.fixture(scope="module")
def structure_fit(tmp_path_factory):
    """A tiny EC-class fit with --selected-site-pocket-grouping structure and its export (synthetic)."""
    from training.config import parse_args
    from training.ec_membership import write_ec_fold_membership
    from training.run import run_training

    base = tmp_path_factory.mktemp("pocket_grouping_fit")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("DEEPMZYME_PARSE_CACHE_DIR", str(base / "parse_cache"))
        patch.delenv("DEEPMZYME_GRAPH_CACHE_DIR", raising=False)
        patch.delenv("DEEPMZYME_FORBIDDEN_READ_ROOTS", raising=False)
        paths = write_ec_dataset(base / "data", _second_site_proteins(default_ec_proteins()))
        rows = [{"structure_id": stem, "accession": accession, "group_id": accession, "fold": str(index % 2)}
                for index, (accession, stem) in enumerate(sorted(paths["stems"].items()))]
        membership = base / "membership.csv"
        membership_sha = write_ec_fold_membership(rows, membership)
        argv = tiny_ec_argv(
            paths, base / "runs", "--ring-features-dir", str(base / "ring"),  # never the default under the repository
            "--ec-class-vocabulary", "fixed_seven", "--ec-conflicting-labels", "fail",
            "--checkpoint-rule", "terminal", "--lr-schedule", "cosine", "--no-prepare-missing-external-features",
            "--export-ec-predictions", "--run-name", "structure_grouping", "--fold-split-source", "membership",
            "--fold-membership-csv", str(membership), "--fold-membership-sha256", membership_sha, "--n-folds", "2",
            "--fold-index", "1", "--site-extraction", "selected_sites", "--selected-site-pocket-grouping", "structure")
        run_dir = run_training(parse_args(argv))
    return {"base": base, "paths": paths, "run_dir": run_dir, "rows": rows}


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_fit_records_the_grouping_and_one_pocket_per_protein(structure_fit):
    run_config = json.loads((structure_fit["run_dir"] / "run_config.json").read_text())
    assert run_config["config"]["selected_site_pocket_grouping"] == "structure"
    summary = run_config["dataset_summary"]["feature_load_report"]["site_extraction_summary"]
    assert summary["selected_site_pocket_grouping"] == "structure"
    assert summary["n_structures"] == summary["n_pockets"] == summary["selected_cluster_count"] == 12
    assert summary["cluster_grouping_cluster_count"] == 24 and summary["structures_with_merged_clusters"] == 12


def test_export_protein_logits_are_the_single_pocket_logits(structure_fit):
    import training.ec_prediction_export as export

    run_dir = structure_fit["run_dir"]
    receipt = json.loads((run_dir / export.EC_PREDICTION_RECEIPT_NAME).read_text())
    assert receipt["reconciliation_status"] == "match"
    assert receipt["selected_site_pocket_grouping"] == "structure"
    pockets = _csv_rows(run_dir / "ec_val_pocket_predictions.csv")
    proteins = _csv_rows(run_dir / "ec_val_protein_predictions.csv")
    validation = sorted(row["structure_id"] for row in structure_fit["rows"] if row["fold"] == "1")
    assert sorted(row["structure_id"] for row in pockets) == validation  # one pocket per protein
    assert all(row["pocket_id"].endswith("_METAL_0") and row["site_ids"] == "A_900;A_901" for row in pockets)
    assert len(proteins) == len(pockets)
    by_group = {row["ec_group_key"]: row for row in pockets}
    for protein in proteins:
        pocket = by_group[protein["ec_group_key"]]
        assert protein["n_pockets"] == "1" and protein["structure_ids"] == pocket["structure_id"]
        assert [protein[column] for column in (*export.LOGIT_COLUMNS, "pred", "y")] == \
            [pocket[column] for column in (*export.LOGIT_COLUMNS, "pred", "y")]


def test_replay_reloads_with_the_recorded_grouping(structure_fit, tmp_path):
    import training.ec_prediction_export as export

    env = {key: value for key, value in os.environ.items()
           if key not in {"DEEPMZYME_FORBIDDEN_READ_ROOTS", "DEEPMZYME_PARSE_CACHE_DIR", "DEEPMZYME_GRAPH_CACHE_DIR"}}
    env.update(PYTHONPATH=str(ROOT / "src"), MKL_THREADING_LAYER="GNU", PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run(
        [sys.executable, "-m", "training.ec_prediction_export", "replay", "--run-dir", str(structure_fit["run_dir"]),
         "--output-dir", str(tmp_path / "replay")], cwd=str(tmp_path), env=env, capture_output=True, text=True,
        timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    receipt = json.loads((tmp_path / "replay" / export.EC_REPLAY_RECEIPT_NAME).read_text())
    assert receipt["replay_status"] == "match" and receipt["max_pocket_logit_abs_difference"] <= 1e-6


def test_test_pass_uses_the_checkpoint_grouping(structure_fit, tmp_path):
    import training.ec_prediction_export as export

    proteins = _second_site_proteins({f"T{index:04d}": {"ec": export.TEST_PLACEHOLDER_EC,
                                                        "ions": [("A", 900, "", "ZN", [10.0 * index, 7.0, 0.0])]}
                                      for index in range(4)})
    paths = write_ec_dataset(tmp_path / "blind", proteins)
    site_table = tmp_path / "test_sites.csv"
    site_table.write_text("accession,chain_resi,metaltype\n"
                          + "".join(f"{accession},{site},ZN\n" for accession in proteins for site in ("A_900", "A_901")))
    checkpoint = structure_fit["run_dir"] / "terminal_model_checkpoint.pt"
    output_root = tmp_path / "test_passes"
    pinned = {path: hashlib.sha256((export.REPO_ROOT / path).read_bytes()).hexdigest()
              for path in export.freeze2_required_code_files()}
    manifest = tmp_path / "freeze2_manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "code_sha256": pinned,
                                    "checkpoint_sha256s": [hashlib.sha256(checkpoint.read_bytes()).hexdigest()],
                                    "test_pass_output_root": str(output_root)}, sort_keys=True))
    receipt = export.export_ec_test_predictions(
        manifest, hashlib.sha256(manifest.read_bytes()).hexdigest(), "grouping-pass", checkpoint_path=checkpoint,
        structure_dir=paths["structures"], site_table_csv=site_table, output_root=output_root,
        esm_embeddings_dir=paths["esm"], external_features_root_dir=paths["external"])
    assert receipt["selected_site_pocket_grouping"] == "structure" and not receipt["failures"]
    rows = _csv_rows(output_root / "grouping-pass" / export.TEST_PROTEIN_PREDICTIONS_NAME)
    assert sorted(row["accession"] for row in rows) == sorted(proteins)
    assert {row["n_pockets"] for row in rows} == {"1"}
