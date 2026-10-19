"""Compare explicit-site replay with eligible saved development predictions only."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .manifest import (COMMON4, NATIVE, ReleaseError, candidate_member_ids, development_artifact, load_manifest,
                       require)


def _rows_by_uid(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(rows and all(row.get("source_uid") for row in rows), f"Missing source identities: {path}")
    output = {row["source_uid"]: row for row in rows}
    require(len(output) == len(rows), f"Duplicate source identities: {path}")
    return output


def _compare(manifest, reports, fixtures, tolerance):
    require(np.isfinite(tolerance) and 0 < tolerance <= 1e-5, "Parity tolerance must be positive and at most 1e-5")
    require(bool(fixtures) and bool(reports), "Parity needs development fixtures and prediction reports")
    required = {"source_uid", "fold", "site_id", "structure_sha256", "center_angstrom", "physical_ion_id"}
    require(all(isinstance(fixture, dict) and required <= fixture.keys() for fixture in fixtures),
            f"Every parity fixture requires {sorted(required)}")
    require(len({f["source_uid"] for f in fixtures}) == len(fixtures), "Duplicate development fixture UID")
    membership_path = development_artifact(manifest.get("fold_membership"), "parity fold membership")
    membership = _rows_by_uid(membership_path)
    member_ids = candidate_member_ids(manifest, manifest["reference"])
    require(all(manifest["members"][member_id]["fold"] is not None for member_id in member_ids),
            "Development OOF parity cannot use a full-data refit")
    validation = {}
    for member_id in member_ids:
        member = manifest["members"][member_id]
        artifacts = member.get("artifacts", member)
        validation[member_id] = _rows_by_uid(development_artifact(artifacts.get("validation_predictions"), "parity validation predictions"))
    checked, errors, covered = [], [], set()
    runtime_signature = None
    for fixture in fixtures:
        uid = fixture["source_uid"]
        require(uid in membership, f"Fixture is absent from frozen development folds: {uid}")
        expected = membership[uid]
        require(int(expected["fold"]) == fixture["fold"], f"Fixture fold differs from frozen membership: {uid}")
        require(expected.get("physical_ion_id") == fixture["physical_ion_id"],
                f"Fixture physical ion differs from frozen membership: {uid}")
        matches = []
        for report in reports:
            if report.get("structure_sha256") != fixture["structure_sha256"]:
                continue
            for index, site in enumerate(report.get("sites", [])):
                if site.get("id") == fixture["site_id"]:
                    matches.append((report, index, site))
        require(len(matches) == 1, f"Fixture needs exactly one structure-hash/site-id matched report: {uid}")
        report, index, site = matches[0]
        require(report.get("manifest_sha256") == manifest["manifest_sha256"], f"Replay used a different manifest: {uid}")
        require(report.get("candidate") == manifest["reference"], f"Replay did not use the complete frozen reference: {uid}")
        center = np.asarray(fixture["center_angstrom"], dtype=float)
        actual_center = np.asarray(site.get("center_angstrom"), dtype=float)
        require(center.shape == (3,) and actual_center.shape == (3,) and np.isfinite(center).all()
                and np.allclose(center, actual_center, atol=1e-6, rtol=0), f"Replay center differs from fixture: {uid}")
        runtime = report.get("runtime", {})
        require(bool(runtime.get("source_sha256")) and bool(runtime.get("packages")), "Replay lacks runtime source/environment identities")
        signature = {key: runtime.get(key) for key in (
            "source_sha256", "packages", "python", "executable", "torch_num_threads", "torch_num_interop_threads")}
        require(runtime_signature is None or runtime_signature == signature,
                "Development parity reports came from different source/environment runtimes")
        runtime_signature = signature
        predictions = report.get("member_predictions")
        require(isinstance(predictions, dict) and set(predictions) == set(member_ids),
                "Replay must contain every frozen reference member prediction")
        for member_id in member_ids:
            member = manifest["members"][member_id]
            if member["fold"] != fixture["fold"]:
                continue
            covered.add(member_id)
            artifacts = member.get("artifacts", member)
            recorded = validation[member_id].get(uid)
            require(recorded is not None, f"{member_id}: fixture is missing from its eligible validation CSV")
            require(recorded.get("physical_ion_id") == fixture["physical_ion_id"]
                    and int(recorded["fold"]) == member["fold"] and int(recorded["model_seed"]) == member["seed"],
                    f"{member_id}: saved prediction identity differs from fixture/member")
            replayed = predictions[member_id]
            require(recorded.get("checkpoint_sha256") == artifacts["checkpoint"]["sha256"]
                    and replayed.get("checkpoint_sha256") == artifacts["checkpoint"]["sha256"],
                    f"{member_id}: prediction checkpoint hash differs from frozen manifest")
            expected_native = np.asarray([float(recorded[f"p_native_{name.replace(' ', '_')}"])
                                          for name in NATIVE[member["target"]]])
            expected_common = np.asarray([float(recorded[f"p_common4_{name.replace(' ', '_')}"])
                                          for name in COMMON4])
            native = np.asarray(replayed["native_probabilities"], dtype=float)
            common = np.asarray(replayed["common4_probabilities"], dtype=float)
            require(native.shape == (len(report["sites"]), len(expected_native))
                    and common.shape == (len(report["sites"]), 4) and np.isfinite(native).all()
                    and np.isfinite(common).all(), f"{member_id}: malformed replay probability arrays")
            native_delta = float(np.max(np.abs(native[index] - expected_native)))
            common_delta = float(np.max(np.abs(common[index] - expected_common)))
            classes_match = (int(native[index].argmax()) == int(recorded["pred_native"])
                             and int(common[index].argmax()) == int(recorded["pred_common4"])
                             and int(replayed["native_predictions"][index]) == int(recorded["pred_native"])
                             and int(replayed["common4_predictions"][index]) == int(recorded["pred_common4"]))
            passed = classes_match and max(native_delta, common_delta) <= tolerance
            checked.append({"source_uid": uid, "member_id": member_id, "fold": member["fold"],
                "native_max_abs_difference": native_delta, "common4_max_abs_difference": common_delta,
                "predicted_classes_match": classes_match, "passed": passed})
            if not passed:
                errors.append(f"{member_id}/{uid}: development prediction parity failed")
    missing = sorted(set(member_ids) - covered)
    if missing:
        errors.append(f"Missing eligible development parity coverage for {len(missing)} reference members")
    return {
        "status": "passed" if not errors else "blocked", "errors": errors, "checks": checked,
        "covered_members": sorted(covered), "uncovered_members": missing,
        "max_abs_difference": max((max(item["native_max_abs_difference"], item["common4_max_abs_difference"])
                                   for item in checked), default=None),
        "runtime_identities": [report["runtime"] for report in reports],
    }


def compare_development_parity(manifest_path: Path, prediction_reports: list[dict], fixtures: list[dict],
                               *, tolerance: float = 1e-5) -> dict:
    """Return a complete reference-member compatibility gate, never quality evidence.

    Fixtures must bind source UID, fold, physical ion, structure hash and center.
    Only the fold checkpoint eligible for each development ion is compared with
    its saved validation CSV. Missing coverage blocks certification.
    """
    manifest = load_manifest(manifest_path)
    common = {"schema": "deepmzyme-metal-development-parity-v1", "manifest_sha256": manifest["manifest_sha256"],
              "reference": manifest["reference"], "probability_tolerance": tolerance,
              "fixture_sha256": hashlib.sha256(json.dumps(fixtures, sort_keys=True).encode()).hexdigest(),
              "prediction_report_sha256": [hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
                                            for report in prediction_reports],
              "scope": "development fixture preprocessing/numerical compatibility; not an unbiased performance estimate"}
    try:
        result = _compare(manifest, prediction_reports, fixtures, tolerance)
    except (ReleaseError, KeyError, TypeError, ValueError, OSError) as exc:
        result = {"status": "blocked", "errors": [str(exc)], "checks": [], "covered_members": [],
                  "uncovered_members": candidate_member_ids(manifest, manifest["reference"]), "max_abs_difference": None}
    return {**common, **result}


def _load_development_fixture(manifest: dict, fixture_path: Path) -> tuple[list[dict], dict]:
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    require(payload.get("split") == "development" and bool(payload.get("fixtures")),
            "Parity runner requires explicitly development-only fixtures")

    def resolve_ref(ref, role):
        require(isinstance(ref, dict) and "path" in ref, f"Missing fixture artifact: {role}")
        path = Path(ref["path"]).expanduser()
        path = (fixture_path.parent / path).resolve() if not path.is_absolute() else path.resolve()
        resolved = {**ref, "path": str(path)}
        development_artifact(resolved, role)
        return resolved

    refs = {name: resolve_ref(payload.get(name), name) for name in (
        "source_cohort", "fold_membership", "structure_manifest")}
    require(refs["fold_membership"]["sha256"] == manifest["fold_membership"]["sha256"],
            "Fixture and release use different development folds")
    cohort = _rows_by_uid(Path(refs["source_cohort"]["path"]))
    membership = _rows_by_uid(Path(refs["fold_membership"]["path"]))
    for member_id in candidate_member_ids(manifest, manifest["reference"]):
        member = manifest["members"][member_id]
        artifacts = member.get("artifacts", member)
        config_path = development_artifact(artifacts.get("run_config"), "parity member config")
        run = json.loads(config_path.read_text())
        config = run.get("config", run)
        require(config.get("source_cohort_sha256") == refs["source_cohort"]["sha256"],
                f"{member_id}: fixture source cohort differs from the checkpoint's training cohort")
    structure_manifest_path = Path(refs["structure_manifest"]["path"])
    with structure_manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    structures = {row["structure_name"]: row for row in rows}
    require(len(structures) == len(rows), "Development structure manifest has duplicate names")
    fixtures = []
    for item in payload["fixtures"]:
        require(item.get("source_uid") in cohort and item["source_uid"] in membership,
                "Fixture is not in the frozen training cohort and folds")
        source, fold = cohort[item["source_uid"]], membership[item["source_uid"]]
        require(item.get("fold") == int(fold["fold"]) and item.get("physical_ion_id") == source["physical_ion_id"]
                and source["physical_ion_id"] == fold["physical_ion_id"], "Fixture ion/fold identity differs")
        center = np.asarray(item.get("center_angstrom"), dtype=float)
        source_center = np.asarray([float(source[f"coord_{axis}"]) for axis in "xyz"])
        require(center.shape == (3,) and np.allclose(center, source_center, atol=1e-8, rtol=0),
                "Fixture center differs from the frozen training ion")
        structure = resolve_ref(item.get("structure"), "parity development structure")
        structure_path = Path(structure["path"])
        entry = structures.get(source["structure_name"])
        require(entry is not None and entry["sha256"] == source["structure_sha256"] == structure["sha256"]
                and structure_path == (structure_manifest_path.parent / entry["relative_path"]).resolve(),
                "Fixture structure does not match the frozen development structure manifest")
        cache = Path(item["embedding_cache"]).expanduser()
        cache = (fixture_path.parent / cache).resolve() if not cache.is_absolute() else cache.resolve()
        chains = sorted(source["context_chains"].split(";"))
        require(sorted(item.get("embeddings", {})) == chains == sorted(item.get("context_chains", [])),
                "Fixture embedding inventory differs from frozen context chains")
        embeddings = {}
        for chain, embedding in item["embeddings"].items():
            embedding_refs = {key: resolve_ref(embedding.get(key), f"parity embedding {key}")
                              for key in ("payload", "sidecar")}
            expected = cache / f"{structure_path.stem}_chain_{chain}_esmc.pt"
            require(Path(embedding_refs["payload"]["path"]) == expected
                    and Path(embedding_refs["sidecar"]["path"]) == expected.with_name(expected.name + ".json"),
                    "Fixture embedding paths do not match the supplied cache")
            embeddings[chain] = embedding_refs
        fixtures.append({**item, "structure": structure, "structure_sha256": structure["sha256"],
                         "embedding_cache": str(cache), "embeddings": embeddings})
    return fixtures, refs


def run_development_parity(manifest_path: Path, fixture_path: Path, output_dir: Path) -> dict:
    """Reproduce a hash-bound development compatibility check on CPU without refits.

    Refuses existing output directories and verifies the cohort, fold, structure
    membership and embedding payload/sidecar bindings before starting inference.
    Each structure's sites share one backbone cache and one load of each member.
    """
    from .manifest import sha256_file
    from .predict import predict_sites

    manifest_path, fixture_path, output_dir = Path(manifest_path).resolve(), Path(fixture_path).resolve(), Path(output_dir).resolve()
    require(not output_dir.exists(), f"Parity output already exists: {output_dir}")
    manifest = load_manifest(manifest_path)
    fixtures, refs = _load_development_fixture(manifest, fixture_path)
    groups = {}
    for fixture in fixtures:
        key = (fixture["structure"]["path"], fixture["embedding_cache"])
        groups.setdefault(key, []).append(fixture)
    output_dir.mkdir(parents=True, exist_ok=False)
    reports, report_paths = [], []
    try:
        for index, ((structure_path, cache), group) in enumerate(groups.items()):
            sites = [{"id": item["site_id"], "center": item["center_angstrom"]} for item in group]
            report = predict_sites(manifest_path, Path(structure_path), sites, candidate=manifest["reference"],
                embedding_cache=Path(cache), generate_embeddings=False, device="cpu", include_member_predictions=True)
            allowed = {(ref["payload"]["path"], ref["payload"]["sha256"])
                       for fixture in group for ref in fixture["embeddings"].values()}
            require(all((artifact["path"], artifact["sha256"]) in allowed
                        for artifact in report["embeddings"]["artifacts"]),
                    "Replay consumed embeddings outside the frozen development fixture")
            path = output_dir / f"structure_{index:02d}.json"
            with path.open("x", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2, sort_keys=True)
                handle.write("\n")
            reports.append(report)
            report_paths.append({"path": path.name, "sha256": sha256_file(path)})
        result = compare_development_parity(manifest_path, reports, fixtures)
    except (ReleaseError, ValueError, RuntimeError, OSError) as exc:
        result = {"schema": "deepmzyme-metal-development-parity-v1", "status": "blocked", "errors": [str(exc)],
                  "manifest_sha256": manifest["manifest_sha256"], "reference": manifest["reference"]}
    result.update({"fixture_file_sha256": sha256_file(fixture_path), "input_artifacts": refs,
                   "prediction_reports": report_paths, "output_dir": str(output_dir)})
    with (output_dir / "parity.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return result
