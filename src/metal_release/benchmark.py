"""CPU inference timing on an explicit hash-bound development fixture."""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from .manifest import ReleaseError, development_artifact, is_held_out_path, load_manifest, require, sha256_file
from .parity import _rows_by_uid


def _development_cache_path(path: Path) -> Path:
    # Cache directories are not hashable artifacts, so only the path rule guards them.
    require(not is_held_out_path(path), f"Refusing held-out path for benchmark embedding cache: {path}")
    return path.resolve()


def _fixture(path: Path, manifest: dict) -> dict:
    path = Path(path).resolve()
    fixture = json.loads(path.read_text(encoding="utf-8"))
    if fixture.get("schema") != "deepmzyme-metal-benchmark-v1" or fixture.get("split") != "development":
        raise ReleaseError("Benchmark requires an explicitly development-only fixture")

    def resolve_ref(ref, role):
        require(isinstance(ref, dict) and ref.get("path"), f"Missing benchmark artifact: {role}")
        resolved = {**ref, "path": str(path.parent / Path(ref["path"]).expanduser())}
        resolved["path"] = str(development_artifact(resolved, role).resolve())
        return resolved

    refs = {name: resolve_ref(fixture.get(name), name) for name in (
        "structure", "source_cohort", "fold_membership", "structure_manifest")}
    development_artifact(manifest.get("fold_membership"), "release fold_membership")
    require(refs["fold_membership"]["sha256"] == manifest["fold_membership"]["sha256"],
            "Benchmark fixture and release use different development folds")
    structure = Path(refs["structure"]["path"])
    rows = _rows_by_uid(Path(refs["fold_membership"]["path"]))
    cohort = _rows_by_uid(Path(refs["source_cohort"]["path"]))
    for member_id, member in manifest["members"].items():
        artifacts = member.get("artifacts", member)
        config_path = development_artifact(artifacts.get("run_config"), "benchmark member config")
        run = json.loads(config_path.read_text(encoding="utf-8"))
        require(run.get("config", run).get("source_cohort_sha256") == refs["source_cohort"]["sha256"],
                f"{member_id}: benchmark source cohort differs from the checkpoint's training cohort")
    structure_manifest = Path(refs["structure_manifest"]["path"])
    with structure_manifest.open(newline="", encoding="utf-8") as handle:
        structure_rows = list(csv.DictReader(handle))
    structures = {row["structure_name"]: row for row in structure_rows}
    require(len(structures) == len(structure_rows), "Development structure manifest has duplicate names")
    sites = fixture.get("sites")
    if not isinstance(sites, list) or len(sites) < 2:
        raise ReleaseError("Benchmark fixture requires at least two development sites in one protein")
    seen, seen_ids, verified_sites = set(), set(), []
    for site in sites:
        uid = site.get("source_uid")
        if uid not in rows or uid not in cohort or uid in seen:
            raise ReleaseError("Benchmark site is missing, duplicated, or belongs to another development structure")
        require(bool(site.get("id")) and site["id"] not in seen_ids, "Benchmark site IDs must be nonempty and unique")
        source, fold = cohort[uid], rows[uid]
        entry = structures.get(source["structure_name"])
        require(entry is not None and entry["sha256"] == source["structure_sha256"] == refs["structure"]["sha256"]
                and structure == (structure_manifest.parent / entry["relative_path"]).resolve(),
                "Benchmark structure differs from the frozen development structure manifest")
        require(source.get("physical_ion_id") and source["physical_ion_id"] == fold.get("physical_ion_id")
                and source.get("pdbid", "").lower() == fold.get("pdbid", "").lower()
                and int(source["model_index"]) == 0,
                "Benchmark physical ion or coordinate model differs from the frozen development identity")
        if (site.get("center") is None) == (site.get("atom") is None):
            raise ReleaseError("Benchmark site needs exactly one center or atom locator")
        center = [float(source[f"coord_{axis}"]) for axis in "xyz"]
        if site.get("center") is not None:
            supplied = site["center"]
            require(isinstance(supplied, list) and len(supplied) == 3 and all(
                isinstance(value, (float, int)) and math.isfinite(value) and abs(value - expected) <= 1e-8
                for value, expected in zip(supplied, center)),
                "Benchmark center differs from the frozen development ion")
        else:
            expected_atom = ":".join(str(source[key]).strip() or "_" for key in (
                "chain", "resseq", "icode", "atom_name", "altloc"))
            require(site["atom"] == expected_atom,
                    "Benchmark atom locator differs from the frozen development ion; explicit altloc is required")
        verified_sites.append({"source_uid": uid, "site_id": site["id"], "fold": int(fold["fold"]),
                               "physical_ion_id": source["physical_ion_id"], "center_angstrom": center})
        seen.add(uid)
        seen_ids.add(site["id"])
    fixture["structure_path"] = structure
    fixture["embedding_cache"] = _development_cache_path(path.parent / fixture["embedding_cache"])
    fixture["verification"] = {"status": "passed", "artifacts": refs, "sites": verified_sites,
                               "scope": "frozen development identity and locator binding; no prediction accuracy claim"}
    return fixture


def _prediction_difference(previous: dict, current: dict) -> dict:
    before, after = previous["class_scores"], current["class_scores"]
    require(set(before) == set(after) and bool(before), "Benchmark prediction score classes differ")
    require(all(math.isfinite(value) for value in [*before.values(), *after.values()]),
            "Benchmark prediction scores must be finite")
    return {"max_abs_probability_difference": max(abs(after[name] - before[name]) for name in before),
            "predicted_class_changed": previous["predicted_class"] != current["predicted_class"]}


def benchmark(manifest_path: Path, fixture_path: Path, output_dir: Path, *, repeats: int = 3,
              generate_embeddings: bool = False) -> dict:
    """Time independent processes so checkpoint loads and peak RAM are comparable.

    same_protein_sites is a measured multi-site call with shared embeddings and
    models; it is not described as isolated marginal latency. No labels score
    these timings and no candidate is selected from them.
    """
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ReleaseError("repeats must be a positive integer")
    manifest_path = Path(manifest_path).resolve()
    manifest = load_manifest(manifest_path)
    if manifest.get("release_status") == "blocked":
        raise ReleaseError("Release artifacts are blocked; benchmark cannot use a smaller fallback")
    fixture = _fixture(fixture_path, manifest)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    sites = [{key: value for key, value in site.items() if key in {"id", "center", "atom"}}
             for site in fixture["sites"]]
    one_path, batch_path = output_dir / "single_site.json", output_dir / "same_protein_sites.json"
    one_path.write_text(json.dumps(sites[:1]) + "\n")
    batch_path.write_text(json.dumps(sites) + "\n")
    cli = Path(__file__).resolve().parents[1] / "metal_release_cli.py"
    report = {"schema": "deepmzyme-metal-inference-benchmark-v1", "status": "complete",
              "runner_path": str(Path(__file__).resolve()), "runner_sha256": sha256_file(Path(__file__)),
              "reference": manifest["reference"], "manifest_sha256": manifest["manifest_sha256"],
              "device": "cpu", "repeats": repeats, "fixture": str(Path(fixture_path).resolve()),
              "fixture_sha256": sha256_file(Path(fixture_path)), "structure_sha256": fixture["structure"]["sha256"],
              "fixture_verification": fixture["verification"], "cold_vs_historical_drift": {},
              "promotion_authorized": False, "candidates": {}, "cold_embedding_generation":
              "measured" if generate_embeddings else "not measured; existing embeddings reused"}
    for candidate, definition in manifest["candidates"].items():
        if definition.get("proxy_only"):
            continue
        observations = {}
        expected_predictions = {}
        cases = [("cached_single_site", one_path, fixture["embedding_cache"], False, repeats),
                 ("same_protein_sites", batch_path, fixture["embedding_cache"], False, repeats)]
        if generate_embeddings:
            cases.insert(0, ("first_prediction_with_embeddings", one_path,
                             output_dir / f"cold_embeddings_{candidate}", True, 1))
        for case, sites_path, cache, generate, count in cases:
            runs = []
            for repeat in range(count):
                result_path = output_dir / f"{candidate}__{case}__{repeat}.json"
                command = [sys.executable, str(cli), "predict", "--manifest", str(manifest_path),
                           "--candidate", candidate, "--structure", str(fixture["structure_path"]),
                           "--sites", str(sites_path), "--embedding-cache", str(cache), "--device", "cpu",
                           "--output", str(result_path)]
                if generate:
                    command.append("--generate-embeddings")
                started = time.perf_counter()
                environment = dict(os.environ)
                environment.setdefault("OMP_NUM_THREADS", "4")
                environment.setdefault("MKL_NUM_THREADS", "4")
                process = subprocess.run(command, capture_output=True, text=True, env=environment)
                elapsed = time.perf_counter() - started
                if process.returncode != 0:
                    runs.append({"status": "blocked", "reason": process.stderr.strip(),
                                 "wall_seconds": elapsed, "returncode": process.returncode})
                    report["status"] = "blocked"
                    break
                prediction = json.loads(result_path.read_text())
                profile_predictions = expected_predictions.setdefault(str(cache), {})
                for site in prediction["sites"]:
                    previous = profile_predictions.setdefault(site["id"], site)
                    difference = _prediction_difference(previous, site)
                    if difference["predicted_class_changed"] or difference["max_abs_probability_difference"] > 1e-5:
                        report["status"] = "blocked"
                        prediction["benchmark_parity_error"] = "Repeated/single/batch predictions changed with the same embedding cache"
                n_sites = 1 if sites_path == one_path else len(sites)
                runs.append({"status": "complete", "wall_seconds": elapsed,
                             "sites_per_second": n_sites / elapsed, "prediction_artifact": result_path.name,
                             "timings": prediction.get("timings", {}),
                             "embeddings": prediction.get("embeddings", {}),
                             "peak_cpu_ram_bytes": prediction.get("memory", {}).get("peak_process_rss_bytes"),
                             "environment": prediction.get("runtime", {})})
                if prediction.get("benchmark_parity_error"):
                    runs[-1].update(status="blocked", reason=prediction["benchmark_parity_error"])
            successful = [run["wall_seconds"] for run in runs if run["status"] == "complete"]
            observations[case] = {"runs": runs, "median_wall_seconds": statistics.median(successful) if successful else None}
        report["candidates"][candidate] = observations
        if generate_embeddings:
            cold = expected_predictions.get(str(output_dir / f"cold_embeddings_{candidate}"), {})
            historical = expected_predictions.get(str(fixture["embedding_cache"]), {})
            shared = sorted(set(cold) & set(historical))
            report["cold_vs_historical_drift"][candidate] = {
                "status": "measured" if shared else "unavailable", "sites": [
                    {"site_id": site_id, **_prediction_difference(historical[site_id], cold[site_id]),
                     "historical_class": historical[site_id]["predicted_class"],
                     "regenerated_class": cold[site_id]["predicted_class"]} for site_id in shared],
                "interpretation": "Different embedding generation profiles can change scores; this descriptive drift is separate from same-cache batching parity and does not establish cold-input quality."}
    report["interpretation"] = (
        "CPU only; wall times include interpreter/import/checkpoint loading. Same-protein batch shares embeddings "
        "and member loads within the call. Peak RAM is isolated per process. Same-cache parity is checked at 1e-5; "
        "cold-versus-historical embedding drift is reported separately. No test inputs, quality claim or promotion.")
    (output_dir / "benchmark.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
