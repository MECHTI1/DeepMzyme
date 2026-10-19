from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from metal_release.benchmark import _fixture, benchmark
from metal_release.catalog import REFERENCE, release_catalog
from metal_release.manifest import ReleaseError, load_manifest, sha256_file


@pytest.fixture
def tmp_path(tmp_path_factory):
    # Keep synthetic development artifacts outside directories named test_*;
    # production guards intentionally refuse those before reading any bytes.
    return tmp_path_factory.mktemp("release_benchmark")


def fixture_files(tmp_path):
    structure = tmp_path / "1abc.pdb"
    structure.write_text("END\n")
    folds = tmp_path / "folds.csv"
    folds.write_text("source_uid,pdbid,physical_ion_id,fold\nu1,1abc,ion1,0\nu2,1abc,ion2,0\n")
    structure_hash = sha256_file(structure)
    cohort = tmp_path / "cohort.csv"
    cohort.write_text("source_uid,pdbid,physical_ion_id,structure_name,structure_sha256,model_index,chain,resseq,icode,atom_name,altloc,coord_x,coord_y,coord_z\n"
                      f"u1,1abc,ion1,1abc.pdb,{structure_hash},0,A,101,,ZN,,1,2,3\n"
                      f"u2,1abc,ion2,1abc.pdb,{structure_hash},0,A,102,,ZN,,4,5,6\n")
    structures = tmp_path / "structures.csv"
    structures.write_text(f"structure_name,relative_path,sha256\n1abc.pdb,1abc.pdb,{structure_hash}\n")
    config = tmp_path / "run_config.json"
    config.write_text(json.dumps({"config": {"source_cohort_sha256": sha256_file(cohort)}}))
    manifest = release_catalog()
    for member in manifest["members"].values():
        member["run_config"] = {"path": str(config), "sha256": sha256_file(config)}
    manifest.update(release_status="provisional", fold_membership={"path": str(folds), "sha256": sha256_file(folds)})
    manifest_path = tmp_path / "release.json"
    manifest_path.write_text(json.dumps(manifest))
    fixture = {"schema": "deepmzyme-metal-benchmark-v1", "split": "development",
               "structure": {"path": "1abc.pdb", "sha256": sha256_file(structure)},
               "source_cohort": {"path": "cohort.csv", "sha256": sha256_file(cohort)},
               "fold_membership": {"path": "folds.csv", "sha256": sha256_file(folds)},
               "structure_manifest": {"path": "structures.csv", "sha256": sha256_file(structures)},
               "embedding_cache": "embeddings", "sites": [
                   {"id": "first", "source_uid": "u1", "center": [1, 2, 3]},
                   {"id": "second", "source_uid": "u2", "center": [4, 5, 6]}]}
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(fixture))
    return manifest_path, path


def fake_prediction(command, **_):
    output = command[command.index("--output") + 1]
    sites_path = command[command.index("--sites") + 1]
    with open(sites_path) as handle:
        sites = json.load(handle)
    for site in sites:
        site.update(predicted_class="Mn", class_scores={"Mn": .7, "Cu": .1, "Zn": .1, "Class VIII": .1})
    with open(output, "w") as handle:
        json.dump({"sites": sites, "timings": {"embeddings_seconds": 1},
                   "memory": {"peak_process_rss_bytes": 1234}, "runtime": {"python": "fixture"}}, handle)
    return SimpleNamespace(returncode=0, stderr="")


def test_benchmark_reports_isolated_ram_and_reference_without_selecting(tmp_path, monkeypatch):
    manifest, fixture = fixture_files(tmp_path)
    monkeypatch.setattr("metal_release.benchmark.subprocess.run", fake_prediction)
    result = benchmark(manifest, fixture, tmp_path / "out", repeats=2)
    assert result["status"] == "complete" and result["reference"] == REFERENCE
    assert result["promotion_authorized"] is False
    assert result["runner_sha256"] == sha256_file(Path(result["runner_path"]))
    assert result["fixture_verification"]["status"] == "passed"
    assert result["fixture_verification"]["sites"][0]["physical_ion_id"] == "ion1"
    timed = [name for name, item in release_catalog()["candidates"].items() if not item.get("proxy_only")]
    assert sorted(result["candidates"]) == sorted(timed)
    for candidate in result["candidates"].values():
        for case in candidate.values():
            assert len(case["runs"]) == 2
            assert case["runs"][0]["peak_cpu_ram_bytes"] == 1234
    with pytest.raises(FileExistsError):
        benchmark(manifest, fixture, tmp_path / "out")


def test_benchmark_refuses_missing_development_identity(tmp_path):
    manifest, fixture = fixture_files(tmp_path)
    payload = json.loads(fixture.read_text())
    payload["sites"][0]["source_uid"] = "not-development"
    fixture.write_text(json.dumps(payload))
    with pytest.raises(ReleaseError, match="development structure"):
        benchmark(manifest, fixture, tmp_path / "out")


def test_benchmark_failure_is_recorded_without_fallback(tmp_path, monkeypatch):
    manifest, fixture = fixture_files(tmp_path)
    monkeypatch.setattr("metal_release.benchmark.subprocess.run", lambda *a, **k:
                        SimpleNamespace(returncode=2, stderr="missing checkpoint"))
    result = benchmark(manifest, fixture, tmp_path / "out", repeats=1)
    assert result["status"] == "blocked" and result["reference"] == REFERENCE
    assert result["candidates"][REFERENCE]["cached_single_site"]["runs"][0]["reason"] == "missing checkpoint"


def test_benchmark_detects_changed_batch_predictions(tmp_path, monkeypatch):
    manifest, fixture = fixture_files(tmp_path)

    def changed(command, **kwargs):
        result = fake_prediction(command, **kwargs)
        if "same_protein_sites.json" in command[command.index("--sites") + 1]:
            path = command[command.index("--output") + 1]
            with open(path) as handle:
                payload = json.load(handle)
            payload["sites"][0]["predicted_class"] = "Zn"
            with open(path, "w") as handle:
                json.dump(payload, handle)
        return result

    monkeypatch.setattr("metal_release.benchmark.subprocess.run", changed)
    result = benchmark(manifest, fixture, tmp_path / "out", repeats=1)
    assert result["status"] == "blocked"


@pytest.mark.parametrize("change,expected", [
    ("center", "center differs"), ("atom", "atom locator differs"),
    ("structure", "frozen development structure manifest"),
    ("cache", "held-out path"), ("cohort", "training cohort"),
    ("model", "coordinate model differs"),
])
def test_benchmark_binds_development_structure_and_site(tmp_path, change, expected):
    manifest, fixture = fixture_files(tmp_path)
    payload = json.loads(fixture.read_text())
    if change == "center":
        payload["sites"][0]["center"][0] += 1
    elif change == "atom":
        payload["sites"][0].pop("center")
        payload["sites"][0]["atom"] = "A:999:_:ZN:_"
    elif change == "structure":
        alternate = tmp_path / "1abc_alternate.pdb"
        alternate.write_text("END\n")
        payload["structure"] = {"path": alternate.name, "sha256": sha256_file(alternate)}
    elif change == "cache":
        protected = tmp_path / "heldout_embeddings"
        protected.mkdir()
        alias = tmp_path / "cache_alias"
        alias.symlink_to(protected, target_is_directory=True)
        payload["embedding_cache"] = alias.name
    else:
        cohort = tmp_path / "cohort.csv"
        cohort.write_text(cohort.read_text().replace(
            ",0,A,101,", ",1,A,101,") if change == "model" else cohort.read_text() + "\n")
        payload["source_cohort"]["sha256"] = sha256_file(cohort)
        if change == "model":
            config = tmp_path / "run_config.json"
            config.write_text(json.dumps({"source_cohort_sha256": sha256_file(cohort)}))
            release = json.loads(manifest.read_text())
            for member in release["members"].values():
                member["run_config"]["sha256"] = sha256_file(config)
            manifest.write_text(json.dumps(release))
    fixture.write_text(json.dumps(payload))
    with pytest.raises(ReleaseError, match=expected):
        benchmark(manifest, fixture, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_benchmark_accepts_exact_frozen_atom_locator(tmp_path):
    manifest, fixture = fixture_files(tmp_path)
    payload = json.loads(fixture.read_text())
    payload["sites"][0].pop("center")
    payload["sites"][0]["atom"] = "A:101:_:ZN:_"
    fixture.write_text(json.dumps(payload))
    verified = _fixture(fixture, load_manifest(manifest))
    assert verified["verification"]["sites"][0]["center_angstrom"] == [1, 2, 3]


def test_cold_embedding_drift_is_separate_from_same_cache_parity(tmp_path, monkeypatch):
    manifest, fixture = fixture_files(tmp_path)

    def regenerated(command, **kwargs):
        result = fake_prediction(command, **kwargs)
        if "--generate-embeddings" in command:
            output = Path(command[command.index("--output") + 1])
            payload = json.loads(output.read_text())
            payload["sites"][0].update(predicted_class="Zn", class_scores={
                "Mn": .1, "Cu": .1, "Zn": .7, "Class VIII": .1})
            output.write_text(json.dumps(payload))
        return result

    monkeypatch.setattr("metal_release.benchmark.subprocess.run", regenerated)
    result = benchmark(manifest, fixture, tmp_path / "out", repeats=2, generate_embeddings=True)
    assert result["status"] == "complete"
    for drift in result["cold_vs_historical_drift"].values():
        assert drift["status"] == "measured"
        assert drift["sites"][0]["predicted_class_changed"] is True
        assert drift["sites"][0]["max_abs_probability_difference"] == pytest.approx(.6)
    for candidate in result["candidates"].values():
        assert all(run["status"] == "complete" for case in candidate.values() for run in case["runs"])
