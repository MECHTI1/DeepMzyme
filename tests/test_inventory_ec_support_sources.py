"""Boundary tests for the read-only EC7/EC6/EC5 training-source inventory (synthetic fixtures only)."""
import builtins
import csv
import hashlib
import io
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT / "tests")]
import inventory_ec_support_sources as inv  # noqa: E402
import test_ec1_component_support as world_fixture  # noqa: E402  (the reporter's synthetic CARE/remoteness world)

POLICY = SimpleNamespace(min_alphafill_identity=0.30, min_alignment_length=85, max_local_rmsd=None,
                         max_binding_site_rmsd=None, max_local_environment_rmsd=None, max_pae_mean=None,
                         min_binding_site_atom_count=None, uniprot_metal_policy="require_supported")
PARAMETERS = {"min_alphafill_identity": 0.3, "min_alignment_length": 85, "site_dedup_distance": 1.0,
              "uniprot_metal_policy": "require_supported", "supported_transition_metals": ["CO", "CU", "FE", "MN", "NI", "ZN"]}
CARE_FIELDS = ["", "Entry", "Entry Name", "Sequence", "EC number", "Length", "EC All", "clusterRes30", "clusterRes50",
               "clusterRes70", "clusterRes90", "EC3", "EC2", "EC1"]
CACHE_FIELDS = ["split", "uniprot_id", "status", "uniprot_supported_transition_metals", "cofactor", "binding_site"]


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def care_row(entry, ec, sequence, cluster):
    return {"": "0", "Entry": entry, "Entry Name": entry, "Sequence": sequence, "EC number": ec, "Length": len(sequence),
            "EC All": ec, "clusterRes30": cluster, "clusterRes50": cluster, "clusterRes70": cluster,
            "clusterRes90": cluster, "EC3": "", "EC2": "", "EC1": ec.split(".")[0]}


def alphafill(compound, identity=0.5, length=120, pdb="1abc"):
    return {"hits": [{"alignment": {"identity": identity, "length": length}, "pdb_id": pdb,
                      "transplants": [{"compound_id": compound, "asym_id": "B"}]}]}


# Development (unit tests): DEV7, representative of cluster DEV7.
CARE_ROWS = [
    care_row("DEV7", "7.1.1.9", "MDEVSEVENAAA", "DEV7"),
    care_row("NR_DEVCLUST", "7.1.1.9", "MNRDEVCLUSTA", "DEV7"),         # non-rep beside a development protein
    care_row("NR_ALIAS", "7.1.1.9", "MDEVSEVENAAA", "REPX"),            # exact alias of DEV7's sequence
    care_row("REPX", "7.2.1.1", "MREPXSEQAAAA", "REPX"),                # representative, not in the membership
    care_row("NR_MULTI", "7.1.1.1", "MNRMULTIAAAA", "REPX"),            # two rows, one class
    care_row("NR_MULTI", "7.2.2.2", "MNRMULTIAAAA", "REPX"),
    care_row("NR_AMBIG", "5.1.1.1", "MNRAMBIGAAAA", "REPX"),            # rows span EC5 and EC6
    care_row("NR_AMBIG", "6.1.1.1", "MNRAMBIGAAAA", "REPX"),
    care_row("NR_NOREP", "6.3.3.3", "MNRNOREPAAAA", "ABSENT"),          # representative absent from entries
    care_row("NR_TESTONLY", "6.3.3.4", "MNRTESTONLYA", "REPX"),         # only a test-side cache row
    care_row("NR_NOJSON", "5.3.3.3", "MNRNOJSONAAA", "REPX"),           # supported metal, no AlphaFill JSON
    care_row("NR_JSONONLY", "5.4.4.4", "MNRJSONONLYA", "REPX"),         # compatible candidate, CIF absent
    care_row("NR_READY", "5.4.4.5", "MNRREADYAAAA", "REPX"),            # compatible candidate, CIF present
    care_row("NR_FES", "5.4.4.6", "MNRFESAAAAAA", "REPX"),              # only an Fe-S compound transplant
    care_row("NR_DUP1", "6.1.1.2", "MSAMESEQAAAA", "REPX"),             # exact duplicates of each other
    care_row("NR_DUP2", "6.1.1.2", "MSAMESEQAAAA", "REPX"),
    care_row("NR_DUP3", "5.2.2.2", "MOTHERDUPAAA", "REPX"),             # a second duplicate group
    care_row("NR_DUP4", "5.2.2.2", "MOTHERDUPAAA", "REPX"),
    care_row("NR_BUILT", "7.4.4.4", "MNRBUILTAAAA", "REPX"),            # an unclustered smoke-build record
    care_row("PMMACC", "7.1.1.9", "MPMMACCAAAAA", "DEV7"),              # PinMyMetal chain accession
]
CACHE_ROWS = [
    {"split": "train", "uniprot_id": a, "status": "OK", "uniprot_supported_transition_metals": m, "cofactor": "",
     "binding_site": ""}
    for a, m in [("NR_DEVCLUST", "CU"), ("NR_ALIAS", "CU"), ("NR_MULTI", ""), ("NR_AMBIG", "ZN"), ("NR_NOREP", "MN"),
                 ("NR_NOJSON", "ZN"), ("NR_JSONONLY", "ZN"), ("NR_READY", "MN;ZN"), ("NR_FES", "FE"),
                 ("NR_DUP1", "ZN"), ("NR_DUP2", "ZN"), ("NR_DUP3", ""), ("NR_DUP4", ""), ("NR_BUILT", "ZN"),
                 ("PMMACC", "CU")]
] + [{"split": "test", "uniprot_id": "NR_TESTONLY", "status": "OK", "uniprot_supported_transition_metals": "ZN",
      "cofactor": "", "binding_site": ""},
     {"split": "test", "uniprot_id": "NR_MULTI", "status": "OK", "uniprot_supported_transition_metals": "ZN",
      "cofactor": "", "binding_site": ""}]
JSONS = {"NR_JSONONLY": alphafill("ZN"), "NR_READY": alphafill("MN", pdb="2xyz"), "NR_FES": alphafill("SF4"),
         "NR_DEVCLUST": alphafill("CU"), "NR_DUP1": alphafill("ZN", identity=0.2), "NR_BUILT": alphafill("ZN")}
CIFS = {"NR_READY", "NR_DEVCLUST", "NR_BUILT"}


def development():
    return {"pockets": {"DEV7": 2}, "accessions": {"DEV7"}, "ec_sequence_hashes": {sha("MDEVSEVENAAA"): "DEV7"},
            "metal_sequences": {"9xyz__chain_A__EC_7.1.1.9": {"sha256": sha("MPMMCHAINAAA"), "group": "9xyz"}},
            "metal_by_hash": defaultdict(set, {sha("MPMMCHAINAAA"): {"9xyz__chain_A__EC_7.1.1.9"}}),
            "components": {"DEV7": {"component_id": "C001", "ec_class": "7"}}, "previous_report": {}}


def work_root(tmp_path, cache_rows=CACHE_ROWS):
    root = tmp_path / "care_sets" / "task1_30"
    write_csv(root / "uniprot" / "uniprot_annotation_cache.csv", cache_rows, CACHE_FIELDS)
    for accession, data in JSONS.items():
        path = root / "alphafill" / "train" / "json" / f"{accession}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    (root / "alphafill" / "train" / "cif").mkdir(parents=True, exist_ok=True)
    for accession in CIFS:
        (root / "alphafill" / "train" / "cif" / f"{accession}.cif").write_text("data_x\n")
    (root / "alphafill" / "test" / "json").mkdir(parents=True, exist_ok=True)  # decoy: must never be listed or read
    (root / "alphafill" / "test" / "json" / "NR_TESTONLY.json").write_text(json.dumps(alphafill("ZN")))
    (root / "mahomes_inputs").mkdir(exist_ok=True)
    (root / "mahomes_inputs" / "build_parameters.json").write_text(json.dumps(PARAMETERS))
    write_csv(root / "mahomes_inputs" / "train" / "build_summary.csv",
              [{"split": "train", "uniprot_id": "NR_BUILT", "status": "FAILED"},
               {"split": "train", "uniprot_id": "OTHER", "status": "OK"}], ["split", "uniprot_id", "status"])
    (root / "rcsb" / "entry").mkdir(parents=True, exist_ok=True)
    (root / "rcsb" / "entry" / "2XYZ.json").write_text("{}")
    return root


def care_file(tmp_path, rows=CARE_ROWS):
    path = tmp_path / "CARE_datasets" / "splits" / "task1" / "protein_train.csv"
    write_csv(path, rows, CARE_FIELDS)
    return path


def nonrep(tmp_path):
    care = inv.load_care_full_train(care_file(tmp_path))["proteins"]
    dev = development()
    clusters = inv.cluster_index(care, dev["accessions"])
    rows, assets, json_inputs = inv.nonrepresentative_rows(care, clusters, dev, work_root(tmp_path), POLICY,
                                                           inv.FOCAL_CLASSES)
    return care, {(r["unit_id"], r["focal_class"]): r for r in rows}, assets, json_inputs


def test_multi_row_accession_is_one_protein_with_the_union_of_its_ecs(tmp_path):
    care = inv.load_care_full_train(care_file(tmp_path))["proteins"]
    record = care["NR_MULTI"]
    assert record["rows"] == 2 and record["ecs"] == {"7.1.1.1", "7.2.2.2"} and record["token"] == "7"
    assert not record["representative"] and care["REPX"]["representative"]


def test_inconsistent_rows_of_one_accession_are_refused(tmp_path):
    bad = CARE_ROWS + [care_row("NR_MULTI", "7.3.3.3", "MDIFFERENTAA", "REPX")]
    with pytest.raises(inv.ReportError):
        inv.load_care_full_train(care_file(tmp_path, bad))


def test_ambiguous_label_gives_no_target_in_each_class_it_touches(tmp_path):
    _, rows, _, _ = nonrep(tmp_path)
    for focal in ("5", "6"):
        row = rows[("NR_AMBIG", focal)]
        assert row["ec_ambiguous"] and row["evidence_level"] == "ambiguous_ec_no_target"
        assert row["eligibility"] == "ineligible_documented"
    table = inv.attrition(list(rows.values()), inv.FOCAL_CLASSES, set())
    step1 = {(a["source"], a["focal_class"]): a["units"] for a in table if a["step"] == 1}
    assert step1[(inv.SOURCES[0], "5")] == sum(1 for (u, c), r in rows.items() if c == "5" and not r["ec_ambiguous"])


def test_relations_duplicates_and_cluster_membership(tmp_path):
    _, rows, _, _ = nonrep(tmp_path)
    assert rows[("NR_DEVCLUST", "7")]["relation"] == "same_clusterres30_as_development_protein"
    assert rows[("NR_ALIAS", "7")]["relation"] == "exact_sequence_alias_of_development_protein"
    absent = rows[("NR_NOREP", "6")]
    assert absent["relation"] == "cluster_representative_absent_from_training_entries"
    assert absent["homology_status"] == inv.HOMOLOGY_UNKNOWN  # no positive relation: homology unknown
    assert not inv.USER_CATEGORIES["known_duplicate_or_related"](absent)
    assert inv.USER_CATEGORIES["cluster_representative_absent_heldout_status_unknown"](absent)
    assert rows[("NR_DUP1", "6")]["exact_sequence_duplicates"] == "NR_DUP2"
    assert inv.USER_CATEGORIES["known_duplicate_or_related"](rows[("NR_DUP1", "6")])
    assert ("REPX", "7") not in rows and ("DEV7", "7") not in rows  # representatives belong to the cohort source


def test_only_training_annotation_rows_are_used(tmp_path):
    _, rows, _, _ = nonrep(tmp_path)
    multi = rows[("NR_MULTI", "7")]  # train row: no supported metal; the test row's ZN must be ignored
    assert multi["supported_metals"] == "" and multi["eligibility"] == "ineligible_documented"
    testonly = rows[("NR_TESTONLY", "6")]  # only a test-side row exists
    assert testonly["annotation_status"] == "NO_TRAINING_CACHE_ROW" and testonly["eligibility"] == "unknown"
    assert "annotation" in testonly["missing_evidence"]


def test_missing_evidence_is_unknown_never_eligible(tmp_path):
    _, rows, assets, json_inputs = nonrep(tmp_path)
    assert rows[("NR_NOJSON", "5")]["evidence_level"] == "metadata_only_supported_metal"
    assert rows[("NR_JSONONLY", "5")]["evidence_level"] == "compatible_candidate_cif_missing"
    ready = rows[("NR_READY", "5")]
    assert ready["evidence_level"] == "compatible_candidate_cif_present_not_built" and ready["donor_pdb_ids"] == "2XYZ"
    fes = rows[("NR_FES", "5")]
    assert fes["evidence_level"] == "alphafill_no_annotation_compatible_candidate"
    assert fes["metal_containing_compound_ids"] == "SF4" and fes["eligibility"] == "ineligible_documented"
    low = rows[("NR_DUP1", "6")]  # identity 0.20 < 0.30: no candidate
    assert low["evidence_level"] == "alphafill_no_annotation_compatible_candidate"
    built = rows[("NR_BUILT", "7")]  # a FAILED smoke-build record stays unknown, never documented
    assert built["evidence_level"] == "unclustered_build_failed" and built["eligibility"] == "unknown"
    for key in (("NR_NOJSON", "5"), ("NR_JSONONLY", "5"), ("NR_READY", "5"), ("NR_BUILT", "7")):
        assert rows[key]["eligibility"] == "unknown" and "catalytic_prediction" in rows[key]["missing_evidence"]
    assert not any(r["eligibility"].startswith("eligible") for r in rows.values())
    assert {r["accession"] for r in json_inputs} == {"NR_JSONONLY", "NR_READY", "NR_FES", "NR_DUP1", "NR_DEVCLUST",
                                                     "NR_BUILT"}
    assert assets["unclustered_build_records_for_focal_accessions"] == 1


def test_no_held_out_path_is_listed_or_opened(tmp_path, monkeypatch):
    touched = []
    real_open, real_listdir = builtins.open, os.listdir

    def record_open(file, *args, **kwargs):
        touched.append(str(file))
        return real_open(file, *args, **kwargs)

    def record_listdir(path="."):
        touched.append(str(path))
        return real_listdir(path)

    care = inv.load_care_full_train(care_file(tmp_path))["proteins"]
    dev = development()
    clusters = inv.cluster_index(care, dev["accessions"])
    root = work_root(tmp_path)
    pmm = pmm_dir(tmp_path)
    monkeypatch.setattr(builtins, "open", record_open)
    monkeypatch.setattr(io, "open", record_open)
    monkeypatch.setattr(os, "listdir", record_listdir)
    inv.nonrepresentative_rows(care, clusters, dev, root, POLICY, inv.FOCAL_CLASSES)
    inv.pmm_rows(pmm, care, clusters, dev, root, inv.FOCAL_CLASSES)
    assert touched and not [p for p in touched if "test" in Path(p).parts]


@pytest.mark.parametrize("name", ["test/protein_train.csv", "splits/task1/30_protein_test.csv",
                                  "metadata/care_task1_30_test_proteins.csv", "x/test_pdbids.txt",
                                  "testset/protein_train.csv", "held_out/x.csv", "x/protein_test2.csv"])
def test_held_out_paths_are_refused(tmp_path, name):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x\n")
    with pytest.raises(inv.ReportError):
        inv.training_path(path)


def test_symlink_into_a_held_out_file_is_refused(tmp_path):
    target = tmp_path / "splits" / "30_protein_test.csv"
    target.parent.mkdir(parents=True)
    target.write_text("x\n")
    link = tmp_path / "elsewhere" / "protein_train.csv"
    link.parent.mkdir()
    link.symlink_to(target)
    with pytest.raises(inv.ReportError):
        inv.training_path(link)


def test_training_path_allows_training_folders_of_mixed_roots(tmp_path):
    path = tmp_path / "train_and_test_sets_structures_non_overlapped_pinmymetal" / "train" / "x.csv"
    path.parent.mkdir(parents=True)
    path.write_text("x\n")
    assert inv.training_path(path) == path


def pmm_dir(tmp_path, shuffle_seed=None):
    root = tmp_path / "train_and_test_sets_structures_non_overlapped_pinmymetal"
    train = root / "train"
    sites = [{"structure": "1aaa", "chain_resi": "A_10", "metaltype": "MN", "ecnumber": "6.3.5.5", "whether_catalytic": "1"},
             {"structure": "2bbb", "chain_resi": "A_11", "metaltype": "CU", "ecnumber": "1.9.3.1", "whether_catalytic": "1"},
             {"structure": "3ccc", "chain_resi": "B_12", "metaltype": "ZN", "ecnumber": "5.1.1.1;6.1.1.1",
              "whether_catalytic": "1"},
             {"structure": "4ddd", "chain_resi": "A_13", "metaltype": "ZN", "ecnumber": "5.4.2.2", "whether_catalytic": "1"},
             {"structure": "5eee", "chain_resi": "A_14", "metaltype": "ZN", "ecnumber": "3.4.1.1", "whether_catalytic": "1"}]
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(sites)
    write_csv(train / inv.PMM_FILES["sites"], sites, list(sites[0]))
    write_csv(train / inv.PMM_FILES["native"], [{**s, "native": "1"} for s in sites], list(sites[0]) + ["native"])
    sifts = [{"structure": "1aaa", "chain": "A", "UniProt ID": "NOTINCARE", "Protein Name": "", "Annotated Cofactors": "",
              "Annotated Metal Symbols": "", "Status": "ok"},
             {"structure": "2bbb", "chain": "A", "UniProt ID": "PMMACC", "Protein Name": "", "Annotated Cofactors": "",
              "Annotated Metal Symbols": "", "Status": "ok"}]
    write_csv(train / inv.PMM_FILES["sifts"], sifts, list(sifts[0]))
    manifest = []
    for structure, chain, ec in [("1aaa", "A", "6.3.5.5"), ("2bbb", "A", "1.9.3.1"), ("3ccc", "B", "5.1.1.1,6.1.1.1"),
                                 ("4ddd", "A", "5.4.2.2"), ("5eee", "A", "3.4.1.1")]:
        name = f"{structure}__chain_{chain}__EC_{ec}.pdb"
        store = root / "store" / name
        store.parent.mkdir(parents=True, exist_ok=True)
        store.write_text("ATOM\n")
        manifest.append({"structure_name": name, "relative_path": f"../store/{name}", "sha256": "", "size_bytes": 5})
    write_csv(train / inv.PMM_FILES["manifest"], manifest, list(manifest[0]))
    (root / "test").mkdir(exist_ok=True)  # decoy: never read
    (root / "test" / "test_pdbids.txt").write_text("9zzz\n")
    return train


def test_pinmymetal_labels_mapping_and_definition(tmp_path):
    care = inv.load_care_full_train(care_file(tmp_path))["proteins"]
    dev = development()
    rows, summary = inv.pmm_rows(pmm_dir(tmp_path), care, inv.cluster_index(care, dev["accessions"]), dev,
                                 work_root(tmp_path), inv.FOCAL_CLASSES)
    by = {(r["unit_id"], r["focal_class"]): r for r in rows}
    assert set(by) == {("1aaa:A", "6"), ("2bbb:A", "7"), ("3ccc:B", "5"), ("3ccc:B", "6"), ("4ddd:A", "5")}
    obsolete = by[("2bbb:A", "7")]  # PDB EC1 (1.9.3.1) but CARE EC7
    assert obsolete["care_label_agrees"] == "no" and obsolete["relation"] == "same_clusterres30_as_development_protein"
    assert obsolete["annotation_compatibility"] == "compatible" and not obsolete["ec_ambiguous"]
    assert by[("1aaa:A", "6")]["relation"] == "none_known"
    assert by[("1aaa:A", "6")]["care_label_agrees"].startswith("unknown")
    assert by[("3ccc:B", "5")]["ec_ambiguous"] and by[("3ccc:B", "5")]["relation"] == "unmapped_chain"
    assert by[("4ddd:A", "5")]["relation"] == "unmapped_chain"
    assert all(r["eligibility"] == "outside_current_definition" for r in rows)
    assert all(r["structure_file_present"] for r in rows)
    assert summary["label_ambiguous_chains"] == 1 and summary["chains"] == 5


def test_pinmymetal_folder_outside_the_non_overlap_root_is_refused(tmp_path):
    folder = tmp_path / "exact_pinmymetal" / "train"
    folder.mkdir(parents=True)
    with pytest.raises(inv.ReportError):
        inv.pmm_rows(folder, {}, {"development_clusters": set()}, development(), work_root(tmp_path), ("7",))


def test_prepared_cohort_reconciliation(tmp_path):
    meta = tmp_path / "metadata"
    fields = ["source_dataset", "source_task", "source_split_name", "source_split", "split", "source_record_id",
              "protein_id", "uniprot_id", "ecnumber", "sequence", "source_file", "source_row", "source_record_ids",
              "n_source_rows"]
    proteins = [{f: "" for f in fields} | {"split": "train", "source_split": "train", "uniprot_id": a, "protein_id": a,
                                           "ecnumber": ec} for a, ec in [("DEV7", "7.1.1.9"), ("AMB", "2.7.1.5;5.3.1.14")]]
    write_csv(meta / "care_task1_30_train_proteins.csv", proteins, fields)
    prepared = tmp_path / "prepared" / "train"
    sites = [{"uniprot_id": a, "source_split": "train"} for a in ("DEV7", "AMB", "AMB")]
    write_csv(prepared / inv.PREPARED_FILES["sites"], sites, ["uniprot_id", "source_split"])
    write_csv(prepared / inv.PREPARED_FILES["manifest"], [{"structure_name": "x"}], ["structure_name"])
    out = inv.prepared_reconciliation(prepared, development(), meta, inv.FOCAL_CLASSES)
    assert out["absent_from_development"] == [{"accession": "AMB", "ec_numbers": "2.7.1.5;5.3.1.14",
                                               "depth1_classes": "2;5", "catalytic_site_rows": 2,
                                               "reason": "ambiguous_ec_no_target"}]
    assert out["absent_with_focal_class"] == {"7": [], "6": [], "5": ["AMB"]}
    write_csv(prepared / inv.PREPARED_FILES["sites"], sites + [{"uniprot_id": "X", "source_split": "test"}],
              ["uniprot_id", "source_split"])
    with pytest.raises(inv.ReportError):
        inv.prepared_reconciliation(prepared, development(), meta, inv.FOCAL_CLASSES)


# ---------------------------------------------------------------------------
# The reporter's synthetic world: real cohort mapping, development loading and end-to-end runs
# ---------------------------------------------------------------------------

WORLD_EXTRA = [  # full-train rows around the world's clusterRes30 cohort (development: A1 B7 C7 D7 E7 F2 G3 H1)
    care_row("NRB7", "7.1.1.9", "MNRBSEVENAAA", "B7"),     # beside a development protein
    care_row("NR_READY", "7.4.4.5", "MNRREADYAAAA", "X5"),  # partially prepared, cluster of a 404'd representative
    care_row("NR_NOJSON", "6.3.3.3", "MNRNOJSONAAA", "X1"),  # metadata only
    care_row("NR_DUP1", "5.1.1.2", "MSAMESEQAAAA", "X1"), care_row("NR_DUP2", "5.1.1.2", "MSAMESEQAAAA", "X1"),
    care_row("NR_DUP3", "5.2.2.2", "MOTHERDUPAAA", "X2"), care_row("NR_DUP4", "5.2.2.2", "MOTHERDUPAAA", "X2"),
]
WORLD_CACHE = [{"split": "train", "uniprot_id": a, "status": "OK", "uniprot_supported_transition_metals": m,
                "cofactor": "", "binding_site": ""}
               for a, m in [("NRB7", "CU"), ("NR_READY", "MN"), ("NR_NOJSON", "ZN"), ("NR_DUP1", ""),
                            ("NR_DUP2", ""), ("NR_DUP3", ""), ("NR_DUP4", ""), ("PMMACC", "CU")]] + [
    {"split": "test", "uniprot_id": "NR_NOJSON", "status": "OK", "uniprot_supported_transition_metals": "",
     "cofactor": "", "binding_site": ""}]


def world_args(tmp_path, *, shuffle=None):
    world = world_fixture.make_world(tmp_path / "w")
    previous = tmp_path / "previous_report"
    world_fixture.rep.run(world_fixture.args_for(world, previous))
    rng = random.Random(shuffle)
    proteins = world_fixture.rep.read_csv_rows(world["meta"] / world_fixture.rep.CARE_METADATA_FILES["train_proteins"])
    care = [care_row(p["uniprot_id"], p["ecnumber"].split(";")[0], p["sequence"], p["uniprot_id"]) for p in proteins]
    care += [care_row(p["uniprot_id"], ec, p["sequence"], p["uniprot_id"]) for p in proteins
             for ec in p["ecnumber"].split(";")[1:]]
    for accession in ("F2", "G3", "H1"):  # development proteins without CARE metadata rows in the world
        care.append(care_row(accession, {"F2": "2.1.1.1", "G3": "3.1.1.1", "H1": "1.1.1.1"}[accession],
                             f"M{accession}SEQAAAA", accession))
    care += WORLD_EXTRA + [care_row("PMMACC", "7.1.1.9", "MPMMACCAAAAA", "B7")]
    cache = list(WORLD_CACHE)
    if shuffle is not None:
        rng.shuffle(care)
        rng.shuffle(cache)
    full_root = tmp_path / "care_sets" / "task1_30"
    write_csv(full_root / "uniprot" / "uniprot_annotation_cache.csv", cache, CACHE_FIELDS)
    for kind, suffix in (("json", ".json"), ("cif", ".cif")):
        folder = full_root / "alphafill" / "train" / kind
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"NR_READY{suffix}").write_text(json.dumps(alphafill("MN", pdb="2xyz")) if kind == "json" else "x\n")
    (full_root / "mahomes_inputs").mkdir()
    (full_root / "mahomes_inputs" / "build_parameters.json").write_text(json.dumps(PARAMETERS))
    prepared = tmp_path / "prepared" / "train"
    write_csv(prepared / inv.PREPARED_FILES["sites"], [{"uniprot_id": a, "source_split": "train"}
                                                       for a in ("B7", "C7", "A1")], ["uniprot_id", "source_split"])
    write_csv(prepared / inv.PREPARED_FILES["manifest"], [{"structure_name": "x"}], ["structure_name"])
    return SimpleNamespace(
        baseline_split=world["baseline"], remoteness_root=world["remote"], previous_report=previous,
        care_full_train=care_file(tmp_path, care), care_full_work_root=full_root, care_metadata_root=world["meta"],
        care_clusterres30_work_root=world["work"], care_prepared_train_dir=prepared,
        pmm_train_dir=pmm_dir(tmp_path, shuffle), focal_classes=list(inv.FOCAL_CLASSES),
        output_dir=tmp_path / "out"), world


def test_cohort_mapping_uses_documented_evidence_only(tmp_path):
    args, world = world_args(tmp_path)
    proteins_path = world["meta"] / world_fixture.rep.CARE_METADATA_FILES["train_proteins"]
    build_path = world["meta"] / world_fixture.rep.CARE_METADATA_FILES["build_summary"]
    proteins = world_fixture.rep.read_csv_rows(proteins_path)
    proteins.append({**proteins[-1], "uniprot_id": "X7", "protein_id": "X7", "ecnumber": "7.9.9.9"})
    write_csv(proteins_path, proteins, list(proteins[0]))
    build = world_fixture.rep.read_csv_rows(build_path)
    build.append({**build[-1], "uniprot_id": "X7", "status": "FAILED", "detail": "exception"})
    write_csv(build_path, build, list(build[0]))
    care = inv.load_care_full_train(args.care_full_train)["proteins"]
    dev = inv.load_development(args.baseline_split, args.remoteness_root, args.previous_report)
    rows, summaries, prepared, _ = inv.cohort_rows(args.care_metadata_root, args.care_clusterres30_work_root,
                                                   args.care_prepared_train_dir, dev, care,
                                                   world_fixture.rep.alphafill_policy(args.care_metadata_root),
                                                   inv.FOCAL_CLASSES)
    by = {(r["unit_id"], r["focal_class"]): r for r in rows}
    assert by[("B7", "7")]["eligibility"] == "eligible_in_development_cohort"
    assert by[("X1", "7")]["eligibility"] == "ineligible_documented"           # no supported metal
    assert by[("X2", "7")]["evidence_level"] == "prepared_noncatalytic_prediction_only"
    assert by[("X3", "7")]["eligibility"] == "ineligible_documented"           # JSON with no compatible candidate
    assert by[("X5", "7")]["evidence_level"] == "alphafill_json_download_failed"
    assert by[("X5", "7")]["eligibility"] == "unknown" and "alphafill_json" in by[("X5", "7")]["missing_evidence"]
    assert by[("X7", "7")]["eligibility"] == "unknown"                         # FAILED build: missing, not documented
    assert by[("A1", "7")]["ec_ambiguous"] and by[("A1", "7")]["evidence_level"] == "ambiguous_ec_no_target"
    funnel = inv.attrition(rows, ("7",), set())
    last = [a for a in funnel if a["source"] == inv.SOURCES[1] and a["condition"].startswith("at least one")][0]
    assert last["units"] == 0  # X2 is built but has no predicted catalytic site
    assert prepared["absent_from_development"] == []  # A1, B7, C7 are development members


def test_development_loading_refuses_a_changed_previous_report(tmp_path):
    args, _ = world_args(tmp_path)
    proteins = args.previous_report / "proteins.csv"
    proteins.write_text(proteins.read_text() + "\n")
    with pytest.raises(inv.ReportError):
        inv.load_development(args.baseline_split, args.remoteness_root, args.previous_report)


def strip_volatile(inventory):
    return {k: v for k, v in inventory.items() if k not in {"created_at", "inputs", "sources"}}


def test_end_to_end_is_deterministic_under_input_reordering(tmp_path):
    first, _ = world_args(tmp_path / "a")
    second, _ = world_args(tmp_path / "b", shuffle=11)
    inv.run(first)
    inv.run(second)
    for name in ("candidate_inventory.csv", "attrition.csv", "download_candidates.csv", "build_donor_entries.csv",
                 "recommendation.md"):
        a = (first.output_dir / name).read_text().replace(str(tmp_path / "a"), "<root>")
        b = (second.output_dir / name).read_text().replace(str(tmp_path / "b"), "<root>")
        assert a == b, name
    a = json.loads((first.output_dir / "inventory.json").read_text().replace(str(tmp_path / "a"), "<root>"))
    b = json.loads((second.output_dir / "inventory.json").read_text().replace(str(tmp_path / "b"), "<root>"))
    assert strip_volatile(a) == strip_volatile(b)
    assert a["results"]["care_full_train"]["exact_sequence_duplicate_groups"] == [["NR_DUP1", "NR_DUP2"],
                                                                                  ["NR_DUP3", "NR_DUP4"]]
    assert a["results"]["eligible_outside_development"] == 0 and a["expectation_check"]["checked"] is True
    rows = list(csv.DictReader((first.output_dir / "candidate_inventory.csv").open()))
    ready = [r for r in rows if r["accession"] == "NR_READY"][0]
    assert ready["evidence_level"] == "compatible_candidate_cif_present_not_built"
    assert ready["relation"] == "none_known" and ready["eligibility"] == "unknown"
    donors = list(csv.DictReader((first.output_dir / "build_donor_entries.csv").open()))
    assert donors == [{"donor_pdb_id": "2XYZ", "candidates": "1", "focal_classes": "7", "cached_unclustered_root": "False",
                       "cached_clusterres30_root": "False"}]
    manifest = json.loads((first.output_dir / "artifact_manifest.json").read_text())
    assert manifest["boundary"]["held_out_files_opened"] is False and manifest["inputs_unchanged_after_run"] is True
    assert set(manifest["outputs"]) == {"candidate_inventory.csv", "attrition.csv", "download_candidates.csv",
                                        "build_donor_entries.csv", "alphafill_json_inputs.csv",
                                        "reused_asset_inputs.csv", "inventory.json", "report.md", "recommendation.md"}


def test_inputs_are_unchanged_and_outputs_are_never_overwritten(tmp_path):
    args, _ = world_args(tmp_path)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*") if p.is_file()}
    inv.run(args)
    after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before}
    assert before == after
    with pytest.raises(inv.ReportError):  # non-empty output directory
        inv.run(args)
    args.output_dir = args.pmm_train_dir / "out"
    with pytest.raises(inv.ReportError):  # inside an input root
        inv.run(args)


def test_unclustered_root_built_under_another_policy_is_refused(tmp_path):
    args, _ = world_args(tmp_path)
    (args.care_full_work_root / "mahomes_inputs" / "build_parameters.json").write_text(
        json.dumps({**PARAMETERS, "min_alphafill_identity": 0.2}))
    with pytest.raises(inv.ReportError):
        inv.run(args)
