"""Boundary tests for the read-only EC1 component-support and EC7 attrition reporter (synthetic fixtures only)."""
import copy
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import report_ec1_component_support as rep  # noqa: E402
from audit_sequence_remoteness import DEFAULT_PROTOCOL, FIELDS, annotate_task, digest, saved_membership  # noqa: E402

AA = "ACDEFGHIKLMNPQRSTVWY"


def rnd(seed, n):
    generator = random.Random(seed)
    return "".join(generator.choice(AA) for _ in range(n))


def mutate(sequence, keep):
    """Keep positions where keep(i) is true; change every other residue to a different one."""
    return "".join(c if keep(i) else AA[(AA.index(c) + 1) % len(AA)] for i, c in enumerate(sequence))


B = rnd(1, 100)
SEQUENCES = {
    "a1": rnd(5, 100),
    "b7": B,
    "c7": mutate(B, lambda i: i % 5 < 3) + rnd(3, 20),  # identity 0.6 with b7, length 120
    "e7": mutate(B, lambda i: i % 5 == 0),  # identity exactly 0.20 with b7
    "f2": rnd(6, 100),
    "h1": mutate(B, lambda i: i % 2 == 0),  # identity 0.5 with b7 (other class in the EC7 component)
}
SEQUENCES["d7"] = SEQUENCES["c7"][:50] + rnd(4, 50)  # coverage exactly 0.50 of d7 against c7
SEQUENCES["g3"] = SEQUENCES["f2"]  # exact-sequence alias
TARGETS = {"a1": 0, "b7": 6, "c7": 6, "d7": 6, "e7": 6, "f2": 1, "g3": 2, "h1": 0}
SIDES = {"a1": "train", "b7": "train", "c7": "validation", "d7": "train", "e7": "validation", "f2": "validation",
         "g3": "train", "h1": "train"}
LINKS = [("b7", "c7", 0, 0, 100), ("d7", "c7", 0, 0, 50), ("b7", "e7", 0, 0, 100), ("h1", "b7", 0, 0, 100)]


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sid(group):
    return "s_" + group


def alignment_row(q, t, qs, ts, length):
    qseq, tseq = SEQUENCES[q], SEQUENCES[t]
    qaln, taln = qseq[qs:qs + length], tseq[ts:ts + length]
    return {"query": sid(q), "target": sid(t), "nident": str(sum(a == b for a, b in zip(qaln, taln))),
            "alnlen": str(length), "qstart": str(qs + 1), "qend": str(qs + length), "qlen": str(len(qseq)),
            "tstart": str(ts + 1), "tend": str(ts + length), "tlen": str(len(tseq)), "evalue": "1e-10",
            "bits": "100", "qaln": qaln, "taln": taln}


def baseline_summary(contradict=False):
    parts = {"train": [], "validation": []}
    for group in sorted(TARGETS):
        pockets = 2 if group == "a1" else 1
        for k in range(pockets):
            target = TARGETS[group] if not (contradict and k == 1) else 3
            parts[SIDES[group]].append({"structure_id": f"{group.upper()}__chain_A__EC_x", "group": group,
                                        "pocket_id": f"{group.upper()}_METAL_{k}", "y_metal": 0, "y_ec": target})
    return {"task": "ec", "ec_label_depth": 1, "structure_dir": "/data/CARE/train",
            "summary_csv": "/data/CARE/train/final.csv", "controlled_ec_auxiliary": False,
            "retained_split_identity": {side: {"n_examples": len(rows), "ordered_examples_sha256": digest(rows),
                                               "examples": rows} for side, rows in parts.items()}}


def make_world(tmp, *, repeats=2, reorder=False, contradict=False, bad_alignment=False, candidate_test_row=False,
               final_override=None, prediction_disagrees=False):
    rng = random.Random(7)
    order = (lambda rows: rng.sample(rows, len(rows))) if reorder else (lambda rows: list(rows))
    world = {"baseline": tmp / "raw" / "expected_split.json", "remote": tmp / "remote", "meta": tmp / "meta",
             "work": tmp / "work"}
    world["baseline"].parent.mkdir(parents=True)
    summary = baseline_summary(contradict)
    world["baseline"].write_text(json.dumps(summary))
    remote = world["remote"]
    (remote / "ec").mkdir(parents=True)
    (remote / "full_protein_readiness").mkdir()
    protocol = {**DEFAULT_PROTOCOL, "component_coverage": 0.5, "component_rule": "fixture",
                "vocabulary": {"ec": list("1234567"), "metal": ["Mn", "Cu", "Zn", "Class VIII"]}}
    (remote / "protocol.json").write_text(json.dumps(protocol))
    membership = saved_membership(summary, "ec", "run0")
    (remote / "memberships.json").write_text(json.dumps({"schema_version": 1, "memberships": [
        {**copy.deepcopy(membership), "run_id": f"run{k}"} for k in range(repeats)]}))
    records = [{"task": "ec", "structure_id": f"{g.upper()}__chain_A__EC_x", "group_id": g, "chain_id": "A",
                "sequence": SEQUENCES[g], "sequence_sha256": hashlib.sha256(SEQUENCES[g].encode()).hexdigest(),
                "sequence_id": sid(g), "status": "resolved", "scope": "represented_coordinate_chain",
                "care_relationship": "exact"} for g in order(sorted(TARGETS))]
    fasta = remote / "ec" / "development.fasta"
    fasta.write_text("".join(f">{r['sequence_id']}\n{r['sequence']}\n" for r in records))
    rows = [alignment_row(g, g, 0, 0, len(SEQUENCES[g])) for g in sorted(TARGETS)]
    for q, t, qs, ts, length in LINKS:
        rows += [alignment_row(q, t, qs, ts, length), alignment_row(t, q, ts, qs, length)]
    if bad_alignment:  # a consistent alignment row whose target residues are not the frozen target sequence
        taln = mutate(rows[-1]["taln"], lambda i: i != 3)
        rows[-1] = {**rows[-1], "taln": taln,
                    "nident": str(sum(a == b for a, b in zip(rows[-1]["qaln"], taln)))}
    alignments = remote / "ec" / "alignments.tsv"
    with alignments.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(order(rows))
    sha = rep.asr.sha256
    manifest = {"schema_version": 1, "endpoint": "represented_coordinate_chain", "sequences": records,
                "protocol_sha256": sha(remote / "protocol.json"), "memberships_sha256": sha(remote / "memberships.json"),
                "reuse_ledger_sha256": "0" * 64,
                "fasta_files": {"ec": {"path": "ec/development.fasta", "sha256": sha(fasta), "n_sequences": len(records)}}}
    (remote / "sequence_manifest.json").write_text(json.dumps(manifest))
    (remote / "preparation_freeze.json").write_text(json.dumps({
        "sequence_manifest_sha256": sha(remote / "sequence_manifest.json"),
        "memberships_sha256": sha(remote / "memberships.json"), "protocol_sha256": sha(remote / "protocol.json")}))
    flags = ["--search-type", "1", "-s", "7.5", "-e", "0.001", "--alignment-mode", "3", "--seq-id-mode", "0", "-a",
             "1", "--min-seq-id", "0", "--min-aln-len", "50", "-c", "0", "--cov-mode", "0", "--mask", "1",
             "--comp-bias-corr", "1", "--max-seqs", "100", "--max-accept", "100", "--max-rejected", "100"]
    (remote / "ec" / "search_receipt.json").write_text(json.dumps({
        "program": "MMseqs2", "version": DEFAULT_PROTOCOL["version"], "release": DEFAULT_PROTOCOL["release"],
        "binary_sha256": "1" * 64, "fasta_sha256": sha(fasta), "alignments_sha256": sha(alignments),
        "database_convention": DEFAULT_PROTOCOL["database_convention"],
        "command": ["mmseqs", "easy-search", str(fasta), str(fasta), str(alignments), str(tmp / "t"), *flags]}))
    (remote / "full_protein_readiness" / "preparation_report.json").write_text(
        json.dumps({"care_discrepancies": [], "blockers": []}))

    meta, work = world["meta"], world["work"]
    focal_in = ["B7", "C7", "D7", "E7"]
    outside = {"X1": "NO_UNIPROT_SUPPORTED_TRANSITION_METAL", "X2": "OK", "X3": rep.MISSING_ASSETS,
               "X4": rep.MISSING_ASSETS, "X5": rep.MISSING_ASSETS, "X6": rep.MISSING_ASSETS}
    proteins = [{"split": "train", "source_split": "train", "uniprot_id": a, "protein_id": a,
                 "ecnumber": "7.1.1.9", "sequence": rnd(len(a) + i, 80)}
                for i, a in enumerate(focal_in + sorted(outside))]
    proteins.append({"split": "train", "source_split": "train", "uniprot_id": "A1", "protein_id": "A1",
                     "ecnumber": "1.1.1.1;7.1.1.9", "sequence": rnd(99, 80)})
    write_csv(meta / rep.CARE_METADATA_FILES["train_proteins"], order(proteins))
    build = []
    for accession in focal_in + sorted(outside):
        status = "OK" if accession in focal_in else outside[accession]
        sites = 2 if accession == "X2" else (1 if status == "OK" else 0)
        build.append({"split": "train", "uniprot_id": accession, "status": status, "selected_site_count": str(sites),
                      "uniprot_supported_transition_metals": {"X1": "", "X5": "CU"}.get(accession, "FE"),
                      "detail": {"X5": "missing json/cif"}.get(accession, "missing cif" if status == rep.MISSING_ASSETS
                                                               else "")})
    write_csv(meta / rep.CARE_METADATA_FILES["build_summary"], order(build))
    candidates, predictions, finals = [], [], []
    for accession in focal_in + ["X2"]:
        for k in range(2 if accession == "X2" else 1):
            site = {"structure": accession, "chain_resi": f"A_{9001 + k}", "metaltype": "CU", "ecnumber": "7.1.1.9",
                    "uniprot_id": accession, "source_split": "train"}
            catalytic = 0 if accession == "X2" else 1
            if final_override and accession in final_override:
                catalytic = final_override[accession]
            candidates.append({**site, "whether_catalytic": "0"})  # the build's placeholder
            label = "Catalytic" if catalytic else "Not Catalytic"
            if prediction_disagrees and accession == "B7":
                label = "Not Catalytic"
            predictions.append({**site, "whether_catalytic": str(catalytic), "prediction": label})
            finals.append({**site, "whether_catalytic": str(catalytic)})
    if candidate_test_row:
        candidates.append({**candidates[0], "source_split": "test"})
    write_csv(meta / rep.CARE_METADATA_FILES["candidate_sites"], order(candidates))
    write_csv(meta / rep.CARE_METADATA_FILES["prediction_results"], order(predictions))
    write_csv(meta / rep.CARE_METADATA_FILES["final_sites"], order(finals))
    fetch = [{"split": "train", "uniprot_id": "X3", "json_status": "OK", "cif_status": rep.DELIBERATE_CIF_SKIP,
              "json_candidate_count": "1", "json_supported_candidate_count": "0",
              "uniprot_supported_transition_metals": "FE"},
             {"split": "train", "uniprot_id": "X4", "json_status": "OK", "cif_status": rep.DELIBERATE_CIF_SKIP,
              "json_candidate_count": "0", "json_supported_candidate_count": "0",
              "uniprot_supported_transition_metals": "FE"},
             {"split": "train", "uniprot_id": "X5", "json_status": "HTTP_404", "cif_status": "NOT_REQUESTED",
              "json_candidate_count": "", "json_supported_candidate_count": "",
              "uniprot_supported_transition_metals": "CU"},
             {"split": "train", "uniprot_id": "X6", "json_status": "OK", "cif_status": rep.DELIBERATE_CIF_SKIP,
              "json_candidate_count": "0", "json_supported_candidate_count": "0",
              "uniprot_supported_transition_metals": "FE"},
             # a held-out row for an allowlisted accession: discarded on read, never used
             {"split": "test", "uniprot_id": "X3", "json_status": "HTTP_500", "cif_status": "FAILED",
              "json_candidate_count": "9", "json_supported_candidate_count": "9",
              "uniprot_supported_transition_metals": "ZN"}]
    write_csv(meta / rep.CARE_METADATA_FILES["fetch_summary"], order(fetch))
    (meta / "build_parameters.json").write_text(json.dumps({
        "min_alphafill_identity": 0.3, "min_alignment_length": 85, "site_dedup_distance": 1.0,
        "uniprot_metal_policy": "require_supported", "supported_transition_metals": ["CO", "CU", "FE", "MN", "NI", "ZN"]}))
    jdir = work / "alphafill" / "train" / "json"
    jdir.mkdir(parents=True)

    def hit(identity, length, compounds):
        return {"pdb_id": "1ABC", "alignment": {"identity": identity, "length": length},
                "transplants": [{"compound_id": c, "analogue_id": c, "asym_id": "B"} for c in compounds]}

    (jdir / "X3.json").write_text(json.dumps({"hits": [hit(0.5, 100, ["ZN", "FES"])]}))
    (jdir / "X4.json").write_text(json.dumps({"hits": [hit(0.5, 100, ["SF4", "MG"])]}))
    (jdir / "X6.json").write_text(json.dumps({"hits": [hit(0.2, 100, ["ZN"]), hit(0.9, 50, ["FE"])]}))
    write_csv(work / rep.ANNOTATION_CACHE, [
        {"split": "train", "uniprot_id": a, "status": "OK", "uniprot_supported_transition_metals": m,
         "cofactor": f"COFACTOR: Name={n};", "binding_site": ""}
        for a, m, n in (("X3", "FE", "Fe cation"), ("X4", "FE", "[4Fe-4S] cluster"), ("X5", "CU", "Cu cation"),
                        ("X6", "FE", "Fe cation"))] + [
        {"split": "test", "uniprot_id": "X4", "status": "OK", "uniprot_supported_transition_metals": "ZN",
         "cofactor": "", "binding_site": ""}])
    return world


def args_for(world, out):
    return rep.parse_args(["--baseline-split", str(world["baseline"]), "--remoteness-root", str(world["remote"]),
                           "--care-metadata-root", str(world["meta"]), "--care-work-root", str(world["work"]),
                           "--output-dir", str(out)])


def report(tmp_path, **kwargs):
    world = make_world(tmp_path / "w", **kwargs)
    return rep.build_report(args_for(world, tmp_path / "out"))


def test_multiple_pockets_count_once_and_classes(tmp_path):
    result = report(tmp_path)["results"]["baseline_support"]
    assert result["proteins"] == 8 and result["pockets"] == 9
    assert result["per_class"]["1"] == {"proteins": 2, "pockets": 3, "components": 2}
    assert result["per_class"]["7"]["proteins"] == 4


def test_repeated_run_memberships_do_not_inflate_support(tmp_path):
    once = report(tmp_path / "a", repeats=1)
    thrice = report(tmp_path / "b", repeats=3)
    assert thrice["results"]["membership_check"]["ec_run_memberships"] == 3
    assert thrice["results"]["membership_check"]["unique_ec_memberships"] == 1
    assert once["results"]["baseline_support"] == thrice["results"]["baseline_support"]


def test_exact_aliases_and_transitive_links_join(tmp_path):
    proteins = {row["accession"]: row for row in report(tmp_path)["tables"]["proteins"]}
    assert proteins["F2"]["component_id"] == proteins["G3"]["component_id"]
    # b7-c7 and c7-d7 qualify; b7 and d7 join through c7 without a direct link
    assert proteins["B7"]["component_id"] == proteins["D7"]["component_id"] == proteins["H1"]["component_id"]


def test_identity_020_does_not_qualify_and_coverage_050_does(tmp_path):
    out = report(tmp_path)
    proteins = {row["accession"]: row for row in out["tables"]["proteins"]}
    links = {(r["protein_a"], r["protein_b"]): r for r in out["tables"]["links"]}
    assert links[("B7", "E7")]["best_identity"] == 0.2 and not links[("B7", "E7")]["qualifies_component_rule"]
    assert proteins["E7"]["component_id"] != proteins["B7"]["component_id"]
    assert links[("C7", "D7")]["best_shorter_coverage"] == 0.5 and links[("C7", "D7")]["qualifies_component_rule"]
    focal = out["results"]["focal_dependence"]
    assert focal["components"] == 2 and focal["qualifying_unordered_focal_pairs"] == 2
    assert focal["other_class_members"] == [{"protein": "H1", "ec_class": "1"}]


def test_components_equal_the_frozen_annotate_task_rule(tmp_path):
    world = make_world(tmp_path / "w")
    out = rep.build_report(args_for(world, tmp_path / "out"))
    remote = rep.load_remoteness(world["remote"], rep.load_baseline(world["baseline"]))
    membership = rep.asr.read_json(world["remote"] / "memberships.json")["memberships"][0]
    frozen = annotate_task(remote["records"], [membership], remote["alignments"], remote["protocol"])
    proteins = {row["accession"].lower(): row["component_id"] for row in out["tables"]["proteins"]}
    assert frozen and all(row["component_id"] == proteins[row["group_id"]] for row in frozen)


def test_alignment_sequence_disagreement_fails(tmp_path):
    with pytest.raises(ValueError, match="Alignment does not match frozen input sequence"):
        report(tmp_path, bad_alignment=True)


def test_contradictory_labels_fail(tmp_path):
    with pytest.raises(rep.ReportError, match="Contradictory EC labels"):
        report(tmp_path, contradict=True)


def test_non_training_candidates_are_rejected(tmp_path):
    with pytest.raises(rep.ReportError, match="non-training record"):
        report(tmp_path, candidate_test_row=True)


def test_held_out_fetch_and_cache_rows_are_never_used(tmp_path):
    rows = {r["accession"]: r for r in report(tmp_path)["tables"]["rejections"]}
    assert rows["X3"]["recorded_json_status"] == "OK" and rows["X3"]["recorded_json_candidate_count"] == "1"
    assert rows["X4"]["annotated_metals_cache"] == "FE"


def test_placeholder_catalytic_values_are_not_final_predictions(tmp_path):
    summary = report(tmp_path)["results"]["focal_attrition"]
    assert summary["final_catalytic_sites"] == 4 and summary["final_noncatalytic_sites"] == 2
    assert summary["attrition_reasons"]["prepared_noncatalytic_prediction_only"] == 1
    flipped = report(tmp_path / "f", final_override={"X2": 1})["results"]["focal_attrition"]
    assert flipped["final_catalytic_sites"] == 6 and "prepared_noncatalytic_prediction_only" not in flipped[
        "attrition_reasons"]
    with pytest.raises(rep.ReportError, match="disagree"):
        report(tmp_path / "d", prediction_disagrees=True)


def test_deliberate_skip_is_distinguished_from_download_failure(tmp_path):
    out = report(tmp_path)
    rows = {r["accession"]: r for r in out["tables"]["rejections"]}
    assert rows["X3"]["cif_skip_kind"] == "deliberate_skip_no_json_supported_candidate"
    assert rows["X5"]["json_fetch_kind"] == "download_failure_http_404"
    assert rows["X5"]["rejection_category"] == "recorded_missing_json"
    assert rows["X3"]["rejection_category"] == "elemental_candidates_rejected_by_annotation"
    assert rows["X4"]["rejection_category"] == "compound_ids_outside_exact_element_mapping"
    assert rows["X6"]["rejection_category"] == "no_qualifying_donor_or_transplant"
    summary = out["results"]["asset_rejections"]
    assert summary["deliberate_cif_skips"] == 3 and summary["json_http_404"] == 1
    assert summary["annotation_compatible_candidates"] == 0
    assert out["results"]["focal_attrition"]["focal_ambiguous_multi_class_proteins"] == 1


def test_input_reordering_preserves_scientific_results(tmp_path):
    plain, shuffled = report(tmp_path / "a"), report(tmp_path / "b", reorder=True)
    assert plain["results"] == shuffled["results"] and plain["tables"] == shuffled["tables"]


def test_run_keeps_evidence_unchanged_and_refuses_unsafe_outputs(tmp_path):
    world = make_world(tmp_path / "w")
    files = sorted(p for p in (tmp_path / "w").rglob("*") if p.is_file())
    before = {p: rep.asr.sha256(p) for p in files}
    result = rep.run(args_for(world, tmp_path / "out"))
    assert {p: rep.asr.sha256(p) for p in files} == before
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == sorted([
        "support_report.json", "proteins.csv", "ec7_links.csv", "ec7_source_inventory.csv",
        "ec7_asset_rejections.csv", "artifact_manifest.json", "report.md", "next_scientific_decision.md"])
    manifest = json.loads((tmp_path / "out" / "artifact_manifest.json").read_text())
    assert manifest["inputs_unchanged_after_run"] and "alphafill_json/X3" in manifest["inputs_before"]
    assert all(manifest["outputs"][n]["sha256"] == rep.asr.sha256(tmp_path / "out" / n) for n in manifest["outputs"])
    assert result["artifact_manifest_sha256"] == rep.asr.sha256(tmp_path / "out" / "artifact_manifest.json")
    with pytest.raises(rep.ReportError, match="not empty"):
        rep.run(args_for(world, tmp_path / "out"))
    with pytest.raises(rep.ReportError, match="inside the input"):
        rep.run(args_for(world, world["remote"] / "report"))
    (tmp_path / "once").mkdir()
    rep.write_once(tmp_path / "once" / "a.txt", "x")
    rep.write_once(tmp_path / "once" / "a.txt", "x")
    with pytest.raises(rep.ReportError, match="overwrite"):
        rep.write_once(tmp_path / "once" / "a.txt", "y")
