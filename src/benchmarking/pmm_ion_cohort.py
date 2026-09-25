"""Training-only provenance audit for the Zenodo PinMyMetal ion cohort (plan Step 1).

The default CLI reads only the training side of
``train_and_test_sets_structures_zenodo_pmm_exact`` and the pinned training source.
The separately gated ``run_reference_audit`` API reuses the same resolver only
after completed refits and frozen reference-report authorization. Neither path
uses the mixed train/test crosswalk.

Every source row receives exactly one disposition:

- ``retained``: one physical ion, supported element compatible with the source code;
- ``duplicate_alias``: another source UID already resolved to the same physical ion
  and label (the retained row lists its aliases);
- ``missing``: no reconstruction row, structure file, or target atom;
- ``unsupported``: the target residue is not a supported transition metal;
- ``conflicting_label``: resolved element incompatible with the source class code, or
  aliases with different native labels (all quarantined);
- ``unresolved``: ambiguous insertion code/atom/alternate location, no residue
  context, or incomplete PinMyMetal features.

Outputs: ``train_cohort.csv``, ``train_row_dispositions.csv``, ``train_audit.json``
and ``campaign_manifest.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch  # noqa: E402

from data_structures import DEFAULT_POCKET_RADIUS, PocketRecord  # noqa: E402
from graph.shell_roles import compute_shell_roles  # noqa: E402
from graph.structure_parsing import (  # noqa: E402
    MetalAtomRecord,
    collect_structure_residues_and_metals,
    find_pocket_residues_near_metal_cluster,
    parse_structure_file,
)
from pmm_source_release import (  # noqa: E402
    PMM_NON_FEATURE_COLUMNS,
    PMM_SOURCE_DEFAULT,
    PMM_SOURCE_RELEASE,
)
import pmm_source_release  # noqa: E402
from training.source_cohort import (  # noqa: E402
    COHORT_COLUMNS,
    COHORT_SCHEMA_VERSION,
    format_coord,
    physical_ion_id,
    sha256_file,
    target_atom_candidates,
)

CAMPAIGN_ID = "pmm_ion_metal_v2_context"
CONTEXT_POLICY = "exclude_explicit_protein_symmetry_links_v1"
PMM_SOURCE_SHA256 = pmm_source_release.PMM_SOURCE_SHA256
DATASET_NAME = "train_and_test_sets_structures_zenodo_pmm_exact"
LEGACY_DATASET_NAME = "train_and_test_sets_structures_exact_pinmymetal"
# PinMyMetal class codes -> admissible resolved elements.
SOURCE_LABEL_ELEMENTS = {"1": ("MN",), "6": ("CU",), "7": ("ZN",), "2": ("FE", "CO", "NI")}
TRAIN_FILES = (
    "site_manifest.csv",
    "structure_manifest.csv",
    "structure_chain_manifest.csv",
    "final_data_summarazing_table.csv",
)
NEIGHBOR_DIAGNOSTIC_RADIUS = 7.0
DISPOSITIONS = ("retained", "duplicate_alias", "missing", "unsupported", "conflicting_label", "unresolved")
DISPOSITION_COLUMNS = (
    "source_row",
    "source_uid",
    "pdbid",
    "source_label_code",
    "source_residueid_ion",
    "source_metalid",
    "manifest_structure_name",
    "manifest_chain",
    "manifest_resseq",
    "manifest_element",
    "manifest_resolution_method",
    "resolved_element",
    "physical_ion_id",
    "disposition",
    "reason",
    "detail",
    "retained_source_uid",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()
        except Exception:
            return None

    import hashlib

    status = run("status", "--porcelain")
    digest = hashlib.sha256()
    for path in sorted([*SRC_ROOT.rglob("*.py"), *(REPO_ROOT / "scripts").glob("*.py")]):
        digest.update(str(path.relative_to(REPO_ROOT)).encode() + b"\0" + path.read_bytes())
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status) if status is not None else None,
            "source_tree_sha256": digest.hexdigest(),
            "source_tree_rule": "sha256 over sorted src/**/*.py and scripts/*.py (path + content)"}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def assert_training_only_root(train_dir: Path) -> Path:
    train_dir = Path(train_dir).resolve()
    if train_dir.name != "train":
        raise ValueError(f"The training-only audit expects the dataset's train/ directory, got {train_dir}")
    if LEGACY_DATASET_NAME in str(train_dir):
        raise ValueError("Refusing the legacy exact_pinmymetal cohort; this campaign binds the Zenodo source rows")
    for name in TRAIN_FILES:
        if not (train_dir / name).is_file():
            raise FileNotFoundError(f"Missing training metadata file {train_dir / name}")
    return train_dir


def pmm_feature_columns(fieldnames: list[str]) -> list[str]:
    return [name for name in fieldnames if name not in PMM_NON_FEATURE_COLUMNS]


def _is_missing_value(value: str | None) -> bool:
    if value is None:
        return True
    text = value.strip()
    if not text or text.lower() in {"nan", "na", "null", "none"}:
        return True
    try:
        return math.isnan(float(text))
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Per-structure resolution (runs in worker processes)
# ---------------------------------------------------------------------------


def resolve_structure_rows(structure_path: str, requests: list[dict[str, Any]]) -> dict[str, Any]:
    """Resolve every requested (chain, resseq) target in model 0 of one structure file."""
    path = Path(structure_path)
    structure = parse_structure_file(str(path), structure_id=path.stem)
    _model_count, symmetry_facts = pdb_symmetry_facts(path) if path.suffix.lower() == ".pdb" else (0, [])
    n_models = len(list(structure))
    residues, metal_records = collect_structure_residues_and_metals(structure, model_index=0)
    residue_coords = [torch.stack([coord.float() for coord in residue.atoms.values()], dim=0) for residue in residues]
    results: list[dict[str, Any]] = []
    for request in requests:
        chain, resseq = request["chain"], int(request["resseq"])
        candidates = target_atom_candidates(structure, model_index=0, chain=chain, resseq=resseq)
        other_models = sum(
            len(target_atom_candidates(structure, model_index=index, chain=chain, resseq=resseq))
            for index in range(1, n_models)
        )
        result: dict[str, Any] = {"source_row": request["source_row"], "n_models": n_models,
                                  "n_candidates_other_models": other_models,
                                  "candidates": [
                                      {"icode": c.icode, "resname": c.resname, "symbol": c.symbol,
                                       "atom_name": c.atom_name, "selected_altloc": c.selected_altloc,
                                       "altlocs": [[a, list(xyz), occ] for a, xyz, occ in c.altloc_coords],
                                       "coord": list(c.coord), "n_atoms_in_residue": c.n_atoms_in_residue}
                                      for c in candidates]}
        metal_candidates = [c for c in candidates if c.symbol is not None]
        if len(metal_candidates) == 1:
            candidate = metal_candidates[0]
            result["protein_symmetry_links"] = protein_symmetry_links(
                symmetry_facts, (chain, resseq, candidate.icode, candidate.resname, candidate.atom_name))
            coord = torch.tensor(candidate.coord, dtype=torch.float32)
            target = MetalAtomRecord(coord=coord, symbol=candidate.symbol, site_id=(chain, resseq, candidate.icode))
            nearby = find_pocket_residues_near_metal_cluster(
                residues, [target], pocket_radius=DEFAULT_POCKET_RADIUS, residue_coord_tensors=residue_coords,
            )
            n_first_shell = 0
            if nearby:
                pocket = PocketRecord(structure_id=path.stem, pocket_id="audit", metal_element=candidate.symbol,
                                      metal_coords=[coord], residues=nearby, metadata={})
                roles = compute_shell_roles(pocket, use_ring_edges=False, shell_role_source="geometry")
                n_first_shell = sum(1 for first, _second in roles if first)
            other_metals = [
                record for record in metal_records
                if record.site_id != (chain, resseq, candidate.icode)
                and float(torch.linalg.vector_norm(record.coord - coord)) <= NEIGHBOR_DIAGNOSTIC_RADIUS
            ]
            result.update({
                "context_chains": sorted({residue.chain_id for residue in nearby}),
                "n_context_residues": len(nearby),
                "n_first_shell_residues": n_first_shell,
                "n_other_metal_atoms_within_7A": len(other_metals),
                "other_metal_symbols_within_7A": sorted(record.symbol for record in other_metals),
            })
        results.append(result)
    return {"structure_path": structure_path, "results": results}


def _resolve_task(task: tuple[str, list[dict[str, Any]]]) -> dict[str, Any]:
    torch.set_num_threads(1)
    try:
        return resolve_structure_rows(*task)
    except Exception as exc:  # recorded as a per-row disposition, never silently dropped
        return {"structure_path": task[0], "error": f"{type(exc).__name__}: {exc}",
                "results": [{"source_row": request["source_row"]} for request in task[1]]}


# ---------------------------------------------------------------------------
# Audit orchestration
# ---------------------------------------------------------------------------


def canonical_structure_names(structure_manifest: list[dict[str, str]], chain_manifest: list[dict[str, str]]) -> tuple[dict[str, str], dict[str, str]]:
    """Map every structure file to one canonical file per identical PDB content.

    The reconstruction stores each PDB entry once per labelled chain as hard links
    to the same full-entry file. Binding every ion of a PDB entry to one canonical
    file prevents the same ion from being parsed (and potentially ingested) once per
    chain alias. Files of one PDB entry with different content keep their own name.
    """
    sha_by_name = {row["structure_name"]: row["sha256"] for row in structure_manifest}
    pdbid_by_name = {row["structure_name"]: row["pdbid"].lower() for row in chain_manifest}
    by_content: dict[tuple[str, str], list[str]] = defaultdict(list)
    for name, sha in sha_by_name.items():
        by_content[(pdbid_by_name.get(name, name), sha)].append(name)
    canonical: dict[str, str] = {}
    for names in by_content.values():
        first = sorted(names)[0]
        for name in names:
            canonical[name] = first
    return canonical, sha_by_name


def verify_structure_hashes(train_dir: Path, sha_by_name: dict[str, str], names: set[str], workers: int) -> dict[str, str]:
    """Recompute content hashes (once per inode) and compare with the manifest."""
    problems: dict[str, str] = {}
    seen_inode: dict[tuple[int, int], str] = {}
    for name in sorted(names):
        path = train_dir / "structures" / name
        if not path.is_file():
            problems[name] = "missing_structure_file"
            continue
        stat = path.stat()
        key = (stat.st_dev, stat.st_ino)
        actual = seen_inode.get(key)
        if actual is None:
            actual = sha256_file(path)
            seen_inode[key] = actual
        if actual != sha_by_name.get(name):
            problems[name] = f"sha256_mismatch:{actual}"
    return problems


def _run_source_audit(train_dir: Path, pmm_source_csv: Path, out_dir: Path, *,
                      policy: _AuditSourcePolicy, workers: int = 2) -> dict[str, Any]:
    if policy.side == "train":
        train_dir = assert_training_only_root(train_dir)
    elif policy.side == "test" and policy.authorization_route_sha256:
        train_dir = Path(train_dir).resolve()
        if train_dir.name != "test":
            raise ValueError("Reference reconstruction requires the declared test/ source side")
    else:
        raise ValueError("Source policy is not authorized")
    out_dir.mkdir(parents=True, exist_ok=True)
    started = utc_now()

    source_sha = sha256_file(pmm_source_csv)
    if source_sha != policy.expected_source_sha256:
        raise ValueError(f"PinMyMetal source hash {source_sha} differs from pinned {policy.expected_source_sha256}")
    source_rows = read_csv_rows(pmm_source_csv)
    feature_columns = pmm_feature_columns(list(source_rows[0].keys()))
    manifest_rows = read_csv_rows(train_dir / "site_manifest.csv")
    structure_manifest = read_csv_rows(train_dir / "structure_manifest.csv")
    chain_manifest = read_csv_rows(train_dir / "structure_chain_manifest.csv")
    summary_rows = read_csv_rows(train_dir / "final_data_summarazing_table.csv")

    file_hashes = {name: sha256_file(train_dir / name) for name in TRAIN_FILES}
    manifest_by_row: dict[int, dict[str, str]] = {}
    manifest_errors: list[str] = []
    for row in manifest_rows:
        if row["source_side"] != policy.side:
            manifest_errors.append(f"wrong-side manifest row {row['source_uid']}")
        prefix, _, row_index = row["source_uid"].rpartition(":row:")
        if prefix != f"sha256:{source_sha}" or int(row_index) != int(row["source_row"]):
            manifest_errors.append(f"source_uid does not bind the pinned source: {row['source_uid']}")
        source_row = int(row["source_row"])
        if source_row in manifest_by_row:
            manifest_errors.append(f"repeated manifest source_row {source_row}")
        manifest_by_row[source_row] = row
    if manifest_errors:
        raise ValueError("Site manifest is not bound to the pinned training source: " + "; ".join(manifest_errors[:5]))

    canonical, sha_by_name = canonical_structure_names(structure_manifest, chain_manifest)
    hash_problems = verify_structure_hashes(train_dir, sha_by_name, set(sha_by_name), workers)

    dispositions: dict[int, dict[str, Any]] = {}
    requests_by_structure: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source_row, source in enumerate(source_rows, start=1):
        base = {
            "source_row": source_row,
            "source_uid": f"sha256:{source_sha}:row:{source_row}",
            "pdbid": source["pdbid"].lower(),
            "source_label_code": source["label_metal"],
            "source_residueid_ion": source["residueid_ion"],
            "source_metalid": source["metalid"],
            "manifest_structure_name": "", "manifest_chain": "", "manifest_resseq": "",
            "manifest_element": "", "manifest_resolution_method": "", "resolved_element": "",
            "physical_ion_id": "", "disposition": "", "reason": "", "detail": "", "retained_source_uid": "",
        }
        dispositions[source_row] = base
        manifest = manifest_by_row.get(source_row)
        if manifest is None:
            base.update(disposition="missing", reason="no_reconstruction_row")
            continue
        base.update(manifest_structure_name=manifest["structure_name"], manifest_chain=manifest["chain"],
                    manifest_resseq=manifest["resseq"], manifest_element=manifest["element"].upper(),
                    manifest_resolution_method=manifest["resolution_method"])
        mismatches = [field for field, source_field in (("pdbid", "pdbid"), ("residueid_ion", "residueid_ion"),
                                                         ("label_metal", "label_metal"))
                      if manifest[field].strip().lower() != source[source_field].strip().lower()]
        if mismatches:
            base.update(disposition="unresolved", reason="manifest_source_field_mismatch", detail=";".join(mismatches))
            continue
        if source["label_metal"] not in SOURCE_LABEL_ELEMENTS:
            base.update(disposition="unsupported", reason="unsupported_source_label_code")
            continue
        missing_features = [name for name in feature_columns if _is_missing_value(source.get(name))]
        if missing_features:
            base.update(disposition="unresolved", reason="pmm_features_incomplete",
                        detail=";".join(missing_features[:10]))
            continue
        name = manifest["structure_name"]
        if name in hash_problems or name not in canonical:
            base.update(disposition="missing", reason="structure_file_unverified", detail=hash_problems.get(name, "not_in_manifest"))
            continue
        requests_by_structure[canonical[name]].append(
            {"source_row": source_row, "chain": manifest["chain"], "resseq": int(manifest["resseq"])}
        )

    tasks = [(str(train_dir / "structures" / name), requests) for name, requests in sorted(requests_by_structure.items())]
    print(f"[AUDIT] resolving {sum(len(r) for _, r in tasks)} rows in {len(tasks)} structures with {workers} workers", flush=True)
    resolutions: dict[int, dict[str, Any]] = {}
    structure_of_row: dict[int, str] = {}
    errors: list[dict[str, str]] = []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        for done, outcome in enumerate(executor.map(_resolve_task, tasks, chunksize=4), start=1):
            if "error" in outcome:
                errors.append({"structure_path": outcome["structure_path"], "error": outcome["error"]})
            for result in outcome["results"]:
                resolutions[int(result["source_row"])] = result
                structure_of_row[int(result["source_row"])] = Path(outcome["structure_path"]).name
            if done % 250 == 0 or done == len(tasks):
                print(f"[AUDIT] {done}/{len(tasks)} structures", flush=True)

    cohort_candidates: list[dict[str, Any]] = []
    resolved_codes_by_ion: dict[str, set[str]] = defaultdict(set)
    for source_row, result in sorted(resolutions.items()):
        base = dispositions[source_row]
        if "candidates" not in result:
            base.update(disposition="unresolved", reason="structure_parse_error")
            continue
        candidates = result["candidates"]
        metal = [c for c in candidates if c["symbol"] is not None]
        if not candidates:
            base.update(disposition="missing", reason="target_residue_absent_in_model_0",
                        detail=f"other_model_candidates={result['n_candidates_other_models']}")
            continue
        if not metal:
            base.update(disposition="unsupported", reason="target_residue_not_supported_metal",
                        detail=";".join(sorted({c["resname"] for c in candidates})))
            continue
        # Only competing supported-metal residues make the target ambiguous; a water or
        # ligand sharing the residue number cannot be the labelled metal ion.
        if len({(c["icode"], c["resname"]) for c in metal}) > 1:
            base.update(disposition="unresolved", reason="ambiguous_insertion_code_or_residue",
                        detail=";".join(sorted({f"{c['resname']}:{c['icode'] or '-'}" for c in metal})))
            continue
        non_metal_same_number = sorted({f"{c['resname']}:{c['icode'] or '-'}" for c in candidates if c["symbol"] is None})
        if len(metal) != 1:
            base.update(disposition="unresolved", reason="multiple_atoms_in_target_residue", detail=str(len(metal)))
            continue
        candidate = metal[0]
        if len(candidate["altlocs"]) > 1:
            base.update(disposition="unresolved", reason="target_atom_alternate_locations",
                        detail=";".join(f"{a or '-'}@{occ:.2f}" for a, _xyz, occ in candidate["altlocs"]))
            continue
        element = candidate["symbol"]
        base["resolved_element"] = element
        manifest_chain, manifest_resseq = base["manifest_chain"], int(base["manifest_resseq"])
        ion_id = physical_ion_id(base["pdbid"], 0, manifest_chain, manifest_resseq, candidate["icode"],
                                 candidate["atom_name"], candidate["altlocs"][0][0])
        base["physical_ion_id"] = ion_id
        resolved_codes_by_ion[ion_id].add(base["source_label_code"])
        if element not in SOURCE_LABEL_ELEMENTS[base["source_label_code"]]:
            base.update(disposition="conflicting_label", reason="resolved_element_incompatible_with_source_code",
                        detail=f"{element} vs code {base['source_label_code']}")
            continue
        if not result.get("n_context_residues"):
            base.update(disposition="unresolved", reason="no_protein_residues_within_pocket_radius")
            continue
        if policy.context_policy == CONTEXT_POLICY and result.get("protein_symmetry_links"):
            base.update(disposition="unresolved", reason="missing_protein_symmetry_context",
                        detail=json.dumps(result["protein_symmetry_links"], sort_keys=True))
            continue
        detail = []
        if base["manifest_element"] and base["manifest_element"] != element:
            detail.append(f"manifest_element={base['manifest_element']}")
        if result["n_models"] > 1:
            detail.append(f"multi_model_file_models={result['n_models']};model_0_used")
        if non_metal_same_number:
            detail.append("non_metal_residue_same_number=" + "|".join(non_metal_same_number))
        base["detail"] = ";".join(detail)
        cohort_candidates.append({
            "source_uid": base["source_uid"], "source_row": source_row, "pdbid": base["pdbid"],
            "group_id": base["pdbid"], "structure_name": structure_of_row[source_row],
            "structure_sha256": sha_by_name[structure_of_row[source_row]], "model_index": 0,
            "chain": manifest_chain, "resseq": manifest_resseq, "icode": candidate["icode"],
            "resname": candidate["resname"], "atom_name": candidate["atom_name"],
            "altloc": candidate["altlocs"][0][0],
            "coord_x": format_coord(candidate["coord"][0]), "coord_y": format_coord(candidate["coord"][1]),
            "coord_z": format_coord(candidate["coord"][2]), "native_element": element,
            "source_label_code": base["source_label_code"], "source_residueid_ion": base["source_residueid_ion"],
            "source_metalid": base["source_metalid"], "physical_ion_id": ion_id, "alias_source_uids": "",
            "context_chains": ";".join(result["context_chains"]), "n_context_residues": result["n_context_residues"],
            "n_first_shell_residues": result["n_first_shell_residues"],
            "_n_other_metal_atoms_within_7A": result["n_other_metal_atoms_within_7A"],
        })

    # Physical-ion aliasing: one loss contribution and one metric vote per ion.
    by_ion: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cohort_candidates:
        by_ion[row["physical_ion_id"]].append(row)
    cohort: list[dict[str, Any]] = []
    for ion_id, rows in by_ion.items():
        rows.sort(key=lambda item: item["source_row"])
        # Every source row resolved to this ion counts, including rows already excluded
        # for an incompatible code: a conflicting source label quarantines the ion.
        codes = resolved_codes_by_ion[ion_id]
        labels = {row["native_element"] for row in rows} | codes
        if len({row["native_element"] for row in rows}) > 1 or len(codes) > 1:
            for row in rows:
                dispositions[row["source_row"]].update(disposition="conflicting_label", reason="alias_label_conflict",
                                                       detail=";".join(sorted(labels)))
            continue
        keeper, aliases = rows[0], rows[1:]
        keeper["alias_source_uids"] = ";".join(row["source_uid"] for row in aliases)
        dispositions[keeper["source_row"]]["disposition"] = "retained"
        dispositions[keeper["source_row"]]["retained_source_uid"] = keeper["source_uid"]
        for row in aliases:
            dispositions[row["source_row"]].update(disposition="duplicate_alias", reason="same_physical_ion_and_label",
                                                   retained_source_uid=keeper["source_uid"])
        cohort.append(keeper)
    cohort.sort(key=lambda item: item["source_row"])

    unassigned = [row for row, item in dispositions.items() if not item["disposition"]]
    if unassigned:
        raise AssertionError(f"{len(unassigned)} source rows have no disposition: {unassigned[:5]}")

    neighbor_stats = _train_only_neighbor_diagnostics(cohort)
    cohort_rows = [{key: row[key] for key in COHORT_COLUMNS} for row in cohort]
    cohort_path = out_dir / "train_cohort.csv"
    dispositions_path = out_dir / "train_row_dispositions.csv"
    write_csv(cohort_path, cohort_rows, COHORT_COLUMNS)
    write_csv(dispositions_path, [dispositions[row] for row in sorted(dispositions)], DISPOSITION_COLUMNS)

    disposition_counts = Counter(item["disposition"] for item in dispositions.values())
    reason_counts = Counter(f"{item['disposition']}:{item['reason']}" for item in dispositions.values())
    audit = {
        "campaign_id": policy.campaign_id,
        "schema_version": COHORT_SCHEMA_VERSION,
        "generated_at": started,
        "completed_at": utc_now(),
        "scope": "training_only" if policy.side == "train" else "secondary_reference",
        "test_side_accessed": policy.side == "test",
        "context_policy": policy.context_policy,
        "authorization_route_sha256": policy.authorization_route_sha256,
        "crosswalk_accessed": False,
        "n_source_rows": len(source_rows),
        "n_manifest_rows": len(manifest_rows),
        "n_summary_rows": len(summary_rows),
        "disposition_counts": {name: disposition_counts.get(name, 0) for name in DISPOSITIONS},
        "reason_counts": dict(sorted(reason_counts.items())),
        "n_retained_ions": len(cohort),
        "n_retained_groups": len({row["group_id"] for row in cohort}),
        "retained_native_counts": dict(sorted(Counter(row["native_element"] for row in cohort).items())),
        "retained_source_code_counts": dict(sorted(Counter(row["source_label_code"] for row in cohort).items())),
        "source_label_mapping": {code: list(elements) for code, elements in SOURCE_LABEL_ELEMENTS.items()},
        "n_structure_files": len(sha_by_name),
        "n_canonical_structure_files": len(set(canonical.values())),
        "n_structure_files_used": len({row["structure_name"] for row in cohort}),
        "structure_hash_problems": hash_problems,
        "structure_parse_errors": errors,
        "empty_first_shell_ions": sum(1 for row in cohort if int(row["n_first_shell_residues"]) == 0),
        "context": {
            "model_policy": "first model only (model_index=0); residues and ions from other models ignored",
            "chain_policy": "all protein chains of the asymmetric-unit file within the pocket radius",
            "symmetry_policy": "no crystallographic symmetry expansion; applied identically to every ion",
            "pocket_radius_angstrom": DEFAULT_POCKET_RADIUS,
            "n_multi_chain_context_ions": sum(1 for row in cohort if ";" in row["context_chains"]),
            "n_context_chain_files_needed": len({(row["structure_name"], chain) for row in cohort
                                                 for chain in row["context_chains"].split(";")}),
        },
        "placeholders": {
            "whether_catalytic": "compatibility placeholder (=1), not a biological annotation; unused in cohort mode",
            "ecnumber": "compatibility placeholder (0.0.0.0); no EC head or loss",
        },
        "pmm_features": {"n_feature_columns": len(feature_columns), "non_feature_columns": list(PMM_NON_FEATURE_COLUMNS),
                         "dropna_rows_removed": sum(1 for item in dispositions.values() if item["reason"] == "pmm_features_incomplete")},
        "train_only_neighbor_diagnostics": neighbor_stats,
        "files": {"train_cohort.csv": sha256_file(cohort_path), "train_row_dispositions.csv": sha256_file(dispositions_path)},
    }
    write_json(out_dir / "train_audit.json", audit)
    manifest = {
        "campaign_id": policy.campaign_id,
        "schema_version": COHORT_SCHEMA_VERSION,
        "created_at": started,
        "dataset_name": DATASET_NAME,
        "dataset_train_dir_at_creation": str(train_dir),
        "identity_rule": "content hashes below; folder names and row counts are not identity",
        "source_release": {**PMM_SOURCE_RELEASE, "source_file": f"classmodel_{policy.side}_set", "source_file_sha256": source_sha},
        "source_side": policy.side,
        "context_policy": policy.context_policy,
        "authorization_route_sha256": policy.authorization_route_sha256,
        "train_metadata_sha256": file_hashes,
        "structure_content_sha256": _aggregate_structure_hash(sha_by_name),
        "n_structure_files": len(sha_by_name),
        "code": git_state(),
        "cohort": {"path": "train_cohort.csv", "sha256": audit["files"]["train_cohort.csv"], "n_rows": len(cohort)},
        "dispositions": {"path": "train_row_dispositions.csv", "sha256": audit["files"]["train_row_dispositions.csv"]},
        "audit": {"path": "train_audit.json", "sha256": sha256_file(out_dir / "train_audit.json")},
    }
    write_json(out_dir / "campaign_manifest.json", manifest)
    return audit


def run_audit(train_dir: Path, pmm_source_csv: Path, out_dir: Path, *, workers: int = 2) -> dict[str, Any]:
    return _run_source_audit(train_dir, pmm_source_csv, out_dir, workers=workers,
                             policy=_AuditSourcePolicy("train", PMM_SOURCE_SHA256, CAMPAIGN_ID))


def run_reference_audit(reference_dir: Path, reference_source_csv: Path, out_dir: Path, *,
                        route_path: Path, workers: int = 2) -> dict[str, Any]:
    """Reuse the source resolver only after completed refits and frozen reporting authorization."""
    from benchmarking.pmm_ion_campaign import CampaignPaths
    from benchmarking.pmm_final_report import require_reference_authorization
    route = require_reference_authorization(CampaignPaths(out_dir), reference_dir, route_path)
    parent_manifest = json.loads((Path(route["campaign_dir"]) / "campaign_manifest.json").read_text(encoding="utf-8"))
    if parent_manifest.get("context_policy") != CONTEXT_POLICY:
        raise ValueError("Reference reconstruction requires the same certified context policy as training")
    if (out_dir / "campaign_manifest.json").exists():
        raise ValueError("Reference reconstruction already exists; reconcile the recorded attempt before retrying")
    audit = _run_source_audit(reference_dir, reference_source_csv, out_dir, workers=workers,
                              policy=_AuditSourcePolicy("test", route["reference_source_sha256"], CAMPAIGN_ID,
                                                        authorization_route_sha256=sha256_file(route_path)))
    context = audit_training_context(reference_dir, out_dir, reference_route_path=route_path)
    if not context["input_contract_certified"]:
        raise ValueError("Reference context has unresolved model, alias or protein-symmetry defects")
    return audit


def _aggregate_structure_hash(sha_by_name: dict[str, str]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for name in sorted(sha_by_name):
        digest.update(f"{name}\t{sha_by_name[name]}\n".encode())
    return digest.hexdigest()


def _train_only_neighbor_diagnostics(cohort: list[dict[str, Any]]) -> dict[str, Any]:
    """Descriptive, predeclared 7 Å sibling counts among retained training ions."""
    by_pdb: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cohort:
        by_pdb[row["pdbid"]].append(row)
    with_labelled_sibling = 0
    with_hetero_native_sibling = 0
    with_hetero_collapsed_sibling = 0
    for rows in by_pdb.values():
        coords = torch.tensor([[float(r["coord_x"]), float(r["coord_y"]), float(r["coord_z"])] for r in rows])
        distances = torch.cdist(coords, coords)
        for index, row in enumerate(rows):
            siblings = [rows[j] for j in range(len(rows)) if j != index and distances[index, j] <= NEIGHBOR_DIAGNOSTIC_RADIUS]
            if siblings:
                with_labelled_sibling += 1
            if any(s["native_element"] != row["native_element"] for s in siblings):
                with_hetero_native_sibling += 1
            if any(s["source_label_code"] != row["source_label_code"] for s in siblings):
                with_hetero_collapsed_sibling += 1
    return {
        "radius_angstrom": NEIGHBOR_DIAGNOSTIC_RADIUS,
        "definition": "other retained training ions of the same PDB entry within the radius (predeclared, descriptive)",
        "ions_with_labelled_sibling": with_labelled_sibling,
        "ions_with_different_native_sibling": with_hetero_native_sibling,
        "ions_with_different_common_four_sibling": with_hetero_collapsed_sibling,
        "ions_with_any_other_metal_atom_within_radius": sum(1 for row in cohort if row["_n_other_metal_atoms_within_7A"]),
        "note": "structure-wide ion multiplicity is not local multinuclearity",
    }


def _verify_campaign_identity_files(train_dir: Path, campaign_dir: Path) -> dict[str, Any]:
    """Recompute training metadata and structure hashes against a frozen manifest."""
    manifest = json.loads((campaign_dir / "campaign_manifest.json").read_text(encoding="utf-8"))
    problems = []
    for name, expected in manifest["train_metadata_sha256"].items():
        if sha256_file(train_dir / name) != expected:
            problems.append(f"{name} hash changed")
    sha_by_name = {row["structure_name"]: row["sha256"] for row in read_csv_rows(train_dir / "structure_manifest.csv")}
    if _aggregate_structure_hash(sha_by_name) != manifest["structure_content_sha256"]:
        problems.append("structure manifest content hash changed")
    actual_problems = verify_structure_hashes(train_dir, sha_by_name, set(sha_by_name), workers=1)
    if actual_problems:
        problems.append(f"{len(actual_problems)} structure files differ from their frozen content hashes")
    for key in ("cohort", "dispositions", "audit"):
        if sha256_file(campaign_dir / manifest[key]["path"]) != manifest[key]["sha256"]:
            problems.append(f"{manifest[key]['path']} changed after freezing")
    if problems:
        raise ValueError("Campaign identity verification failed: " + "; ".join(problems))
    return manifest


def verify_campaign_identity(train_dir: Path, campaign_dir: Path) -> dict[str, Any]:
    return _verify_campaign_identity_files(assert_training_only_root(train_dir), campaign_dir)


def verify_reference_campaign_identity(source_dir: Path, campaign_dir: Path, route_path: Path) -> dict[str, Any]:
    from benchmarking.pmm_ion_campaign import CampaignPaths
    from benchmarking.pmm_final_report import require_reference_authorization

    route = require_reference_authorization(CampaignPaths(campaign_dir), source_dir, route_path)
    manifest = _verify_campaign_identity_files(source_dir, campaign_dir)
    if (manifest.get("source_side") != "test" or manifest.get("context_policy") != CONTEXT_POLICY
            or manifest["source_release"]["source_file_sha256"] != route["reference_source_sha256"]):
        raise ValueError("Reference profile differs from its authorized source or training context policy")
    return manifest


def audit_training_context(train_dir: Path, campaign_dir: Path, *, reference_route_path: Path | None = None) -> dict[str, Any]:
    """Bounded, training-only model/alias/annotated-symmetry audit of the frozen cohort.

    LINK records can positively identify omitted symmetry-related ligands. Their
    absence cannot establish parity with the source's historical crystal context.
    The report keeps these two claims separate and never rewrites the cohort.
    """
    manifest = (verify_campaign_identity(train_dir, campaign_dir) if reference_route_path is None
                else verify_reference_campaign_identity(train_dir, campaign_dir, reference_route_path))
    cohort = read_csv_rows(campaign_dir / "train_cohort.csv")
    by_structure: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_hash: dict[str, set[str]] = defaultdict(set)
    for row in cohort:
        by_structure[row["structure_name"]].append(row)
        by_hash[row["structure_sha256"]].add(row["pdbid"])
    aliases = [sorted(pdbids) for pdbids in by_hash.values() if len(pdbids) > 1]
    alias_groups = {row["pdbid"]: row["group_id"] for row in cohort}
    ungrouped_aliases = [pdbids for pdbids in aliases if len({alias_groups[p] for p in pdbids}) > 1]
    multi_model, symmetry_links, unsupported_formats = [], [], []
    for name, rows in sorted(by_structure.items()):
        path = Path(train_dir) / "structures" / name
        if path.suffix.lower() != ".pdb":
            unsupported_formats.append(name)
            continue
        targets = {(row["chain"], int(row["resseq"]), row["icode"], row["resname"], row["atom_name"]): row
                   for row in rows}
        model_count = 0
        with path.open(encoding="utf-8", errors="strict") as handle:
            for line in handle:
                if line.startswith("MODEL "):
                    model_count += 1
                if not line.startswith("LINK  "):
                    continue
                # PDB v3 LINK columns: target atom/residue identities and two symmetry operators.
                try:
                    left = (line[21:22].strip(), int(line[22:26]), line[26:27].strip(),
                            line[17:20].strip(), line[12:16].strip())
                    right = (line[51:52].strip(), int(line[52:56]), line[56:57].strip(),
                             line[47:50].strip(), line[42:46].strip())
                except ValueError as exc:
                    raise ValueError(f"Malformed LINK identity in {name}") from exc
                operators = (line[59:65].strip(), line[66:72].strip())
                # Equal operators describe the same copy; differing operators explicitly require another copy.
                normalized = tuple(value or "1555" for value in operators)
                if normalized[0] == normalized[1]:
                    continue
                for identity, other in ((left, right), (right, left)):
                    if identity in targets:
                        symmetry_links.append({"source_uid": targets[identity]["source_uid"],
                                               "structure_name": name, "partner": list(other),
                                               "symmetry_operators": list(operators)})
        if model_count > 1:
            multi_model.append({"structure_name": name, "n_models": model_count,
                                "source_uids": [row["source_uid"] for row in rows]})
    from Bio.PDB.Polypeptide import is_aa
    protein_links = [link for link in symmetry_links if is_aa(link["partner"][3], standard=False)]
    blockers = []
    if protein_links:
        blockers.append("Explicit target LINK records require protein symmetry context absent from the declared input")
    if multi_model:
        blockers.append("Multi-model source identity requires an explicit provenance decision")
    if ungrouped_aliases:
        blockers.append("Exact structure-content aliases cross declared identity groups")
    if unsupported_formats:
        blockers.append("Structural formats need an equivalent explicit symmetry/model audit")
    report = {
        "schema_version": 1, "campaign_id": manifest["campaign_id"],
        "scope": "training_only" if reference_route_path is None else "secondary_reference",
        "cohort_sha256": manifest["cohort"]["sha256"], "audited_at": utc_now(),
        "structure_content_sha256": manifest["structure_content_sha256"],
        "n_ions": len(cohort), "n_structures": len(by_structure),
        "model_ambiguities": multi_model, "exact_content_aliases": aliases,
        "ungrouped_exact_content_aliases": ungrouped_aliases, "target_symmetry_links": symmetry_links,
        "n_ions_with_target_symmetry_links": len({row["source_uid"] for row in symmetry_links}),
        "target_protein_symmetry_links": protein_links,
        "n_ions_with_target_protein_symmetry_links": len({row["source_uid"] for row in protein_links}),
        "unsupported_formats": unsupported_formats, "blockers": blockers,
        "input_contract_certified": not blockers,
        "input_contract": "frozen first-model asymmetric-unit protein coordinates; all local protein chains",
        "source_context_parity_certified": False,
        "source_context_parity_note": "No claim of historical source/paper assembly parity; absent LINK annotations do not prove absent symmetry contacts",
        "test_side_accessed": reference_route_path is not None,
    }
    write_json(campaign_dir / "train_context_audit.json", report)
    return report


def derive_context_complete_campaign(train_dir: Path, parent_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Version a frozen training cohort by the pre-fit protein-symmetry exclusion rule."""
    from Bio.PDB.Polypeptide import is_aa

    if out_dir.exists():
        raise ValueError(f"Refusing to overwrite an existing campaign: {out_dir}")
    parent = verify_campaign_identity(train_dir, parent_dir)
    context_path = parent_dir / "train_context_audit.json"
    context = json.loads(context_path.read_text(encoding="utf-8"))
    if (context["cohort_sha256"] != parent["cohort"]["sha256"]
            or context["structure_content_sha256"] != parent["structure_content_sha256"]
            or context["model_ambiguities"] or context["ungrouped_exact_content_aliases"] or context["unsupported_formats"]):
        raise ValueError("Parent context audit is stale or has unresolved defects beyond protein symmetry context")
    excluded = {row["source_uid"] for row in context["target_symmetry_links"] if is_aa(row["partner"][3], standard=False)}
    parent_cohort = read_csv_rows(parent_dir / "train_cohort.csv")
    if not excluded or not excluded.issubset({row["source_uid"] for row in parent_cohort}):
        raise ValueError("Context exclusions are empty or do not belong to the parent cohort")
    cohort = [row for row in parent_cohort if row["source_uid"] not in excluded]
    dispositions = read_csv_rows(parent_dir / "train_row_dispositions.csv")
    for row in dispositions:
        if row["source_uid"] in excluded or row["retained_source_uid"] in excluded:
            row.update(disposition="unresolved", reason="missing_protein_symmetry_context", retained_source_uid="",
                       detail="Target LINK to a protein residue with differing symmetry operators; excluded consistently")
    out_dir.mkdir(parents=True)
    write_csv(out_dir / "train_cohort.csv", cohort, COHORT_COLUMNS)
    write_csv(out_dir / "train_row_dispositions.csv", dispositions, DISPOSITION_COLUMNS)
    audit = json.loads((parent_dir / "train_audit.json").read_text(encoding="utf-8"))
    audit.update(campaign_id=CAMPAIGN_ID, generated_at=utc_now(), completed_at=utc_now(), context_policy=CONTEXT_POLICY,
                 n_retained_ions=len(cohort), n_retained_groups=len({row["group_id"] for row in cohort}),
                 disposition_counts=dict(Counter(row["disposition"] for row in dispositions)),
                 reason_counts=dict(Counter(f"{row['disposition']}:{row['reason']}" for row in dispositions)),
                 retained_native_counts=dict(Counter(row["native_element"] for row in cohort)),
                 retained_source_code_counts=dict(Counter(row["source_label_code"] for row in cohort)),
                 empty_first_shell_ions=sum(int(row["n_first_shell_residues"]) == 0 for row in cohort),
                 n_structure_files_used=len({row["structure_name"] for row in cohort}),
                 parent_campaign_manifest_sha256=sha256_file(parent_dir / "campaign_manifest.json"),
                 context_exclusion_count=len(excluded), parent_context_audit_sha256=sha256_file(context_path))
    # Mutable subgroup numbers from the parent must not be restated as subset observations.
    audit.pop("train_only_neighbor_diagnostics", None)
    audit["context"] = {**audit["context"], "n_multi_chain_context_ions": sum(";" in row["context_chains"] for row in cohort),
                         "n_context_chain_files_needed": len({(row["structure_name"], chain) for row in cohort
                                                              for chain in row["context_chains"].split(";")})}
    audit["files"] = {name: sha256_file(out_dir / name) for name in ("train_cohort.csv", "train_row_dispositions.csv")}
    write_json(out_dir / "train_audit.json", audit)
    manifest = {**parent, "campaign_id": CAMPAIGN_ID, "created_at": utc_now(), "context_policy": CONTEXT_POLICY,
                "source_side": "train", "code": git_state(),
                "parent_campaign": {"campaign_id": parent["campaign_id"], "path_at_creation": str(parent_dir),
                                    "manifest_sha256": sha256_file(parent_dir / "campaign_manifest.json"),
                                    "context_audit_sha256": sha256_file(context_path)},
                "cohort": {"path": "train_cohort.csv", "sha256": audit["files"]["train_cohort.csv"], "n_rows": len(cohort)},
                "dispositions": {"path": "train_row_dispositions.csv", "sha256": audit["files"]["train_row_dispositions.csv"]},
                "audit": {"path": "train_audit.json", "sha256": sha256_file(out_dir / "train_audit.json")}}
    write_json(out_dir / "campaign_manifest.json", manifest)
    # Independently rescan the retained structures; do not infer certification solely from the exclusion count.
    certified = audit_training_context(train_dir, out_dir)
    if not certified["input_contract_certified"]:
        raise ValueError("Derived cohort still has unresolved declared-input context")
    return audit


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-dir", type=Path, required=True, help="The dataset's train/ directory only.")
    parser.add_argument("--pmm-source-csv", type=Path, default=PMM_SOURCE_DEFAULT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("DEEPMZYME_LOAD_WORKERS", "2")))
    parser.add_argument("--verify-only", action="store_true", help="Only re-verify a frozen campaign identity.")
    parser.add_argument("--context-audit", action="store_true", help="Audit frozen training model/alias/annotated symmetry context.")
    parser.add_argument("--derive-context-from", type=Path, help="Derive a new version from a frozen parent plus its context audit.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.derive_context_from is not None:
        audit = derive_context_complete_campaign(args.train_dir, args.derive_context_from, args.out_dir)
        print(json.dumps({key: audit[key] for key in ("campaign_id", "n_retained_ions", "n_retained_groups",
                                                     "context_exclusion_count")}, indent=2))
        return
    if args.context_audit:
        print(json.dumps(audit_training_context(args.train_dir, args.out_dir), indent=2))
        return
    if args.verify_only:
        manifest = verify_campaign_identity(args.train_dir, args.out_dir)
        print(json.dumps({"verified": True, "cohort_sha256": manifest["cohort"]["sha256"]}, indent=2))
        return
    if (args.out_dir / "campaign_manifest.json").exists():
        raise SystemExit(f"Refusing to overwrite a frozen campaign at {args.out_dir}")
    audit = run_audit(args.train_dir, args.pmm_source_csv, args.out_dir, workers=args.workers)
    print(json.dumps({key: audit[key] for key in ("disposition_counts", "n_retained_ions", "retained_native_counts",
                                                  "empty_first_shell_ions")}, indent=2))


def pdb_symmetry_facts(path: Path) -> tuple[int, list[dict[str, Any]]]:
    """Read PDB model count and nonidentity LINK operations, without inferring missing annotations."""
    models, links = 0, []
    with path.open(encoding="utf-8", errors="strict") as handle:
        for line in handle:
            if line.startswith("MODEL "):
                models += 1
            if not line.startswith("LINK  "):
                continue
            try:
                left = (line[21:22].strip(), int(line[22:26]), line[26:27].strip(),
                        line[17:20].strip(), line[12:16].strip())
                right = (line[51:52].strip(), int(line[52:56]), line[56:57].strip(),
                         line[47:50].strip(), line[42:46].strip())
            except ValueError as exc:
                raise ValueError(f"Malformed LINK identity in {path.name}") from exc
            operators = (line[59:65].strip(), line[66:72].strip())
            if (operators[0] or "1555") != (operators[1] or "1555"):
                links.append({"left": left, "right": right, "symmetry_operators": operators})
    return max(models, 1), links


def protein_symmetry_links(facts: list[dict[str, Any]], identity: tuple) -> list[dict[str, Any]]:
    from Bio.PDB.Polypeptide import is_aa
    return [{"partner": list(other), "symmetry_operators": list(link["symmetry_operators"])}
            for link in facts for target, other in ((link["left"], link["right"]), (link["right"], link["left"]))
            if target == identity and is_aa(other[3], standard=False)]


@dataclass(frozen=True)
class _AuditSourcePolicy:
    side: str
    expected_source_sha256: str
    campaign_id: str
    context_policy: str = CONTEXT_POLICY
    authorization_route_sha256: str | None = None


if __name__ == "__main__":
    main()
