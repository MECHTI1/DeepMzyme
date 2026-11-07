"""Training-side CLEAN sanity checks of amendment 2 (synthetic embeddings and exports only; no dataset access)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT / "tests")]

from ec_clean30 import clean_sanity  # noqa: E402
from ec_clean30.clean_sanity import SanityError  # noqa: E402
import ec_clean30_precision_stub as precision  # noqa: E402


@pytest.fixture(autouse=True)
def runtime_precision(monkeypatch):
    return precision.install(monkeypatch)

comparator = clean_sanity.load_comparator()
DIM = 8
EC_NUMBERS = ("1.1.1.1", "1.2.1.1", "2.1.1.1", "2.3.1.1", "3.1.1.1", "3.4.1.1", "4.1.1.1", "4.2.1.1",
              "5.1.1.1", "5.2.1.1", "6.1.1.1", "6.3.1.1")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def training_world(members_per_ec: int = 6, seed: int = 0):
    """EC numbers with well separated centres; each training protein carries one EC number."""
    rng = np.random.default_rng(seed)
    centres = {number: rng.normal(scale=10.0, size=DIM) for number in EC_NUMBERS}
    table, embeddings = {}, {}
    for number in EC_NUMBERS:
        for k in range(members_per_ec):
            accession = f"P{number.replace('.', '')}{k:02d}"
            table[accession] = (number,)
            embeddings[accession] = centres[number] + rng.normal(scale=0.3, size=DIM)
    return table, embeddings


def receipt(*, arm="r1", fold=0, regime=None, inner_fold=None, n_in_sample=None, n_query=0, seed=42, label="fit"):
    regime = regime or clean_sanity.ARM_REGIME[arm]
    embedding = "esmc300m_final_mean" if arm == "c_esmc" else comparator.DEFAULT_EMBEDDING
    payload = {"mode": "validation", "fold": fold, "regime": regime, "seed": seed, "embedding": embedding,
               "input_dim": 960 if arm == "c_esmc" else comparator.DEFAULT_INPUT_DIM,
               "checkpoint_sha256": sha(f"checkpoint-{label}"), "training_table_sha256": sha(f"table-{label}"),
               "embedding_farm_sha256": sha(f"farm-{label}"), "query_embeddings_sha256": sha(f"queries-{label}"),
               "n_in_sample_queries": n_query if n_in_sample is None else n_in_sample,
               "npz_sha256": sha(f"npz-{label}"), "query_ids_sha256": sha(f"ids-{label}")}
    if inner_fold is not None:
        payload["inner_fold"] = inner_fold
    return payload


def export_from(queries, query_embeddings, table, train_embeddings, *, transform=None, **receipt_args):
    """A distance export whose 'model' is the raw embedding itself (a healthy synthetic fit)."""
    labels, centres = clean_sanity.ec_centres(table, train_embeddings)
    rows = np.stack([query_embeddings[query] for query in queries])
    distances = clean_sanity.euclidean_distances(rows, centres)
    labels = list(labels)
    if transform is not None:
        distances, labels = transform(distances, labels)
    return comparator.DistanceExport(np.asarray(distances, dtype=np.float32), tuple(queries), tuple(labels),
                                     receipt(n_query=len(queries), **receipt_args))


def in_sample_record(table, embeddings, *, transform=None, query_embeddings=None, arm="r1", **receipt_args):
    regime = clean_sanity.ARM_REGIME[arm]
    sample = clean_sanity.in_sample_sample(table, fold=0, regime=regime, inner_fold=receipt_args.get("inner_fold"))
    anchor = clean_sanity.raw_anchor_projection(table, embeddings, sample, embeddings)
    export = export_from(sample, query_embeddings or embeddings, table, embeddings, transform=transform, arm=arm,
                         **receipt_args)
    return clean_sanity.evaluate_in_sample(arm=arm, export=export, training_ec_numbers=table, sample=sample,
                                           anchor=anchor, precision_policy=precision.fit_policy_entry(),
                                           created_utc="2026-11-04T00:00:00+00:00")


def test_in_sample_sample_is_hashed_deterministic_and_excludes_single_member_and_several_class_proteins():
    table, _ = training_world(members_per_ec=3)
    table["LONE1"] = ("1.9.9.9",)  # its EC number has a single training protein
    table["TWO1"] = ("1.1.1.1", "2.1.1.1")  # two EC classes
    table["SEVEN1"], table["SEVEN2"] = ("7.1.1.1",), ("7.1.1.1",)  # class 7
    table["PAIR1"] = ("3.1.1.1", "3.4.1.1")  # one EC class, every EC number shared
    eligible = clean_sanity.in_sample_eligible(table)
    assert {"LONE1", "TWO1", "SEVEN1", "SEVEN2"}.isdisjoint(eligible) and "PAIR1" in eligible
    assert len(eligible) == 3 * len(EC_NUMBERS) + 1
    sample = clean_sanity.in_sample_sample(table, fold=2, regime="r1")
    key = lambda accession: hashlib.sha256(f"2:r1:-:{accession}".encode()).hexdigest()  # noqa: E731
    assert sample == tuple(sorted(eligible, key=key))
    assert sample == clean_sanity.in_sample_sample(dict(reversed(list(table.items()))), fold=2, regime="r1")
    assert clean_sanity.in_sample_sample(table, fold=3, regime="r1") != sample
    assert clean_sanity.in_sample_sample(table, fold=2, regime="r1", n=5) == sample[:5]
    big, _ = training_world(members_per_ec=30)
    assert len(clean_sanity.in_sample_sample(big, fold=0, regime="r2")) == clean_sanity.IN_SAMPLE_N == 256
    inner = clean_sanity.in_sample_sample(table, fold=0, regime="r1_inner", inner_fold=1)
    assert inner == tuple(sorted(eligible, key=lambda a: hashlib.sha256(f"0:r1_inner:1:{a}".encode()).hexdigest()))
    with pytest.raises(SanityError, match="needs inner fold"):
        clean_sanity.in_sample_sample(table, fold=0, regime="r1_inner")
    with pytest.raises(SanityError, match="takes no inner fold"):
        clean_sanity.in_sample_sample(table, fold=0, regime="r1", inner_fold=0)


def _collapse(distances, labels):
    return np.ones_like(distances), labels


def _permute(distances, labels):
    order = list(range(2, len(labels))) + [0, 1]  # columns shifted by one EC class: centres renamed
    return distances, [labels[index] for index in order]


def _flip(distances, labels):
    return distances.max() - distances, labels


def test_healthy_fit_passes_and_broken_fits_fail():
    table, embeddings = training_world(members_per_ec=25)
    healthy = in_sample_record(table, embeddings)
    assert healthy["status"] == clean_sanity.STATUS_PASSED and all(healthy["criteria"].values())
    assert healthy["values"]["primary"] == 1.0 and healthy["values"]["anchor_primary"] == 1.0
    assert healthy["sample"]["n"] == 256 and healthy["fit"]["arm"] == "r1"
    for transform, failing in ((_collapse, {"IS1", "IS2", "IS3"}), (_permute, {"IS1", "IS2"}),
                               (_flip, {"IS1", "IS2"})):
        record = in_sample_record(table, embeddings, transform=transform)
        assert record["status"] == clean_sanity.STATUS_DIAGNOSIS_REQUIRED
        assert {name for name, ok in record["criteria"].items() if not ok} == failing, transform.__name__
    rng = np.random.default_rng(3)
    accessions = list(embeddings)
    shuffled = dict(zip(accessions, [embeddings[a] for a in rng.permutation(accessions)]))
    record = in_sample_record(table, embeddings, query_embeddings=shuffled)
    assert not record["criteria"]["IS1"] and record["status"] == clean_sanity.STATUS_DIAGNOSIS_REQUIRED
    # IS4: the export must hold exactly the sample, all counted as in-sample queries.
    record = in_sample_record(table, embeddings, n_in_sample=0)
    assert record["criteria"] == {"IS1": True, "IS2": True, "IS3": True, "IS4": False}
    with pytest.raises(SanityError, match="validation-mode"):
        sample = clean_sanity.in_sample_sample(table, fold=0, regime="r1")
        export = export_from(sample, embeddings, table, embeddings)
        export.receipt["mode"] = "test"
        clean_sanity.evaluate_in_sample(arm="r1", export=export, training_ec_numbers=table, sample=sample,
                                        anchor=clean_sanity.raw_anchor_projection(table, embeddings, sample,
                                                                                  embeddings),
                                        precision_policy=precision.fit_policy_entry())
    with pytest.raises(SanityError, match="regime"):
        in_sample_record(table, embeddings, arm="r2", regime="r1")
    with pytest.raises(SanityError, match="not the hashed in-sample sample"):
        sample = clean_sanity.in_sample_sample(table, fold=0, regime="r1")[:-1]
        clean_sanity.evaluate_in_sample(arm="r1", export=export_from(sample, embeddings, table, embeddings),
                                        training_ec_numbers=table, sample=sample,
                                        anchor=clean_sanity.raw_anchor_projection(table, embeddings, sample,
                                                                                  embeddings),
                                        precision_policy=precision.fit_policy_entry())


def test_raw_anchor_uses_training_means_and_the_projection_rule():
    embeddings = {"A1": np.array([0.0, 0.0]), "A2": np.array([2.0, 0.0]), "B1": np.array([10.0, 10.0]),
                  "B2": np.array([12.0, 10.0]), "T1": np.array([-10.0, 5.0])}
    table = {"A1": ("1.1.1.1",), "A2": ("1.1.1.1",), "B1": ("2.1.1.1",), "B2": ("2.1.1.1",),
             "T1": ("3.1.1.1", "4.1.1.1")}  # one protein carries two EC numbers of different classes
    labels, centres = clean_sanity.ec_centres(table, embeddings)
    assert labels == ("1.1.1.1", "2.1.1.1", "3.1.1.1", "4.1.1.1")  # first appearance in the table
    assert np.allclose(centres[0], [1.0, 0.0]) and np.allclose(centres[1], [11.0, 10.0])
    queries = {"Q1": np.array([1.2, 0.1]), "Q2": np.array([-9.0, 5.0]), "Q3": np.array([10.5, 9.0])}
    projection = clean_sanity.raw_anchor_projection(table, embeddings, ["Q1", "Q2", "Q3"], queries)
    assert projection.predicted_class[0] == "1" and projection.predicted_class[2] == "2"
    assert projection.tie.tolist() == [False, True, False]  # identical centres of EC classes 3 and 4
    expected = comparator.project_ec_classes(
        clean_sanity.euclidean_distances(np.stack(list(queries.values())), centres), ("Q1", "Q2", "Q3"), labels)
    assert projection.predicted_class == expected.predicted_class
    assert clean_sanity.majority_class({"a": ("2.1.1.1",), "b": ("2.2.2.2",), "c": ("1.1.1.1",),
                                        "d": ("1.1.1.1", "3.1.1.1")}) == "2"
    assert clean_sanity.majority_class({"a": ("2.1.1.1",), "b": ("1.1.1.1",)}) == "1"  # ties: lower class
    with pytest.raises(SanityError, match="no base embedding"):
        clean_sanity.ec_centres(table, {key: value for key, value in embeddings.items() if key != "T1"})


def _held_out_world():
    """Fold 0: four inner folds; R1-inner trains on everything outside the inner fold, R2-inner on a subset."""
    table, embeddings = training_world(members_per_ec=12, seed=5)
    accessions = sorted(table)
    inner_of = {accession: index % 4 for index, accession in enumerate(accessions)}
    validation = {inner: sorted(a for a in accessions if inner_of[a] == inner) for inner in range(4)}
    r1_tables = {inner: {a: table[a] for a in accessions if inner_of[a] != inner} for inner in range(4)}
    r2_tables = {inner: {a: numbers for a, numbers in r1_tables[inner].items() if int(a[-2:]) % 2 == 0}
                 for inner in range(4)}
    return table, embeddings, validation, r1_tables, r2_tables


def _held_out_fits(table, embeddings, validation, r1_tables, r2_tables, *, transform=None, in_sample=0):
    queries, fits = {}, []
    for inner in range(4):
        queries[inner] = clean_sanity.held_out_queries(validation[inner], table, r1_tables[inner])
        for arm, tables in (("r1_inner", r1_tables), ("r2_inner", r2_tables)):
            own = tables[inner]
            anchor = clean_sanity.raw_anchor_projection(own, embeddings, queries[inner], embeddings)
            export = export_from(validation[inner], embeddings, own, embeddings, arm=arm, inner_fold=inner,
                                 n_in_sample=in_sample, transform=transform if arm == "r1_inner" else None,
                                 label=f"{arm}-{inner}")
            fits.append(clean_sanity.HeldOutFit(arm, export, own, anchor, precision.fit_policy_entry()))
    return fits, queries


def test_held_out_requires_inner_membership_closure_and_refuses_in_sample_exports():
    table, embeddings, validation, r1_tables, r2_tables = _held_out_world()
    # Closure: a query whose EC number has no R1-inner centre, or with two EC classes, is left out.
    extra = dict(table, NEW1=("5.9.9.9",), MIX1=("1.1.1.1", "2.1.1.1"))
    chosen = clean_sanity.held_out_queries(validation[0] + ["NEW1", "MIX1"], extra, r1_tables[0])
    assert "NEW1" not in chosen and "MIX1" not in chosen and set(chosen) == set(validation[0])
    with pytest.raises(SanityError, match="in the R1-inner training table"):
        clean_sanity.held_out_queries(validation[0] + [next(iter(r1_tables[0]))], table, r1_tables[0])
    fits, queries = _held_out_fits(table, embeddings, validation, r1_tables, r2_tables)
    record = clean_sanity.evaluate_held_out(fits, queries, table, created_utc="2026-11-04T00:00:00+00:00")
    assert record["status"] == clean_sanity.STATUS_PASSED
    assert record["criteria"]["r1_inner"] == {"HO1": True, "HO2": True, "HO3": True, "HO4": True}
    assert set(record["criteria"]["r2_inner"]) == {"HO1", "HO3", "HO4"}  # HO2 is for R1-inner only
    assert record["arms"]["r1_inner"]["pooled"]["n_queries"] == len(table)
    summary = clean_sanity.verify_record(record)
    assert summary["check"] == "held_out" and len(summary["checkpoint_sha256s"]) == 8
    broken, _ = _held_out_fits(table, embeddings, validation, r1_tables, r2_tables, transform=_flip)
    failed = clean_sanity.evaluate_held_out(broken, queries, table)
    assert failed["status"] == clean_sanity.STATUS_DIAGNOSIS_REQUIRED and not failed["criteria"]["r1_inner"]["HO1"]
    assert failed["criteria"]["r2_inner"] == {"HO1": True, "HO3": True, "HO4": True}
    in_sample, _ = _held_out_fits(table, embeddings, validation, r1_tables, r2_tables, in_sample=3)
    with pytest.raises(SanityError, match="refuses an export with in-sample queries"):
        clean_sanity.evaluate_held_out(in_sample, queries, table)
    with pytest.raises(SanityError, match="one fit per arm and inner fold"):
        clean_sanity.evaluate_held_out(fits[:-1], queries, table)


def test_failure_status_and_dispositions(tmp_path_factory):
    folder = tmp_path_factory.mktemp("sanity")
    table, embeddings = training_world(members_per_ec=25)
    passed = folder / "passed.json"
    failed = folder / "failed.json"
    passed_sha = clean_sanity.write_once(passed, in_sample_record(table, embeddings))
    failed_sha = clean_sanity.write_once(failed, in_sample_record(table, embeddings, transform=_collapse))
    assert clean_sanity.verify_record(failed)["status"] == "comparator_diagnosis_required"
    with pytest.raises(SanityError, match="write-once"):
        clean_sanity.write_once(failed, {"kind": "anything"})
    with pytest.raises(SanityError, match="placeholder"):
        clean_sanity.build_disposition(record_sha256=failed_sha, decision="proceed_disclosed", maintainer="[name]",
                                       note="no defect found")
    with pytest.raises(SanityError, match="decides one of"):
        clean_sanity.build_disposition(record_sha256=failed_sha, decision="retune", maintainer="the maintainer",
                                       note="no defect found")
    proceed = folder / "proceed.json"
    clean_sanity.write_once(proceed, clean_sanity.build_disposition(
        record_sha256=failed_sha, decision="proceed_disclosed", maintainer="the maintainer", note="no defect found"))
    stop = folder / "stop.json"
    clean_sanity.write_once(stop, clean_sanity.build_disposition(
        record_sha256=failed_sha, decision="stop", maintainer="the maintainer", note="no defect found"))
    summary = clean_sanity.require_passed_or_disposed(failed, [proceed])
    assert summary["disposition"] == "proceed_disclosed" and summary["sha256"] == failed_sha
    with pytest.raises(SanityError, match="'stop'"):
        clean_sanity.require_passed_or_disposed(failed, [stop])
    with pytest.raises(SanityError, match="exactly one disposition"):
        clean_sanity.require_passed_or_disposed(failed)
    assert clean_sanity.require_passed_or_disposed(passed)["disposition"] is None
    with pytest.raises(SanityError, match="names no supplied sanity record"):
        clean_sanity.require_passed_or_disposed(passed, [proceed])
    other = folder / "other.json"
    clean_sanity.write_once(other, clean_sanity.build_disposition(
        record_sha256=passed_sha, decision="proceed_disclosed", maintainer="the maintainer", note="no defect found"))
    with pytest.raises(SanityError, match="failed record only"):
        clean_sanity.require_passed_or_disposed(passed, [other])
    payload = json.loads(proceed.read_text())
    payload["no_defect_found"] = False
    with pytest.raises(SanityError, match="only when no defect was found"):
        clean_sanity.verify_disposition(payload)


def test_structural_tie_groups_identical_member_sets_across_ec_classes():
    table = {"A": ("1.1.1.1", "4.2.1.1"), "B": ("2.1.1.1",), "C": ("2.1.1.1", "2.7.1.1"), "D": ("3.1.1.1",),
             "E": ("3.1.1.1",), "F": ("5.1.1.1", "6.1.1.1"), "G": ("5.1.1.1", "6.1.1.1")}
    totals = clean_sanity.structural_tie_groups(table)
    # {A}: 1.1.1.1 and 4.2.1.1 (classes 1 and 4); {F, G}: 5.1.1.1 and 6.1.1.1; {C}: 2.7.1.1 alone.
    assert totals == {"n_ec_numbers": 7, "n_identical_member_groups": 2, "n_cross_class_groups": 2,
                      "n_ec_numbers_in_cross_class_groups": 4, "n_proteins_in_cross_class_groups": 3}
    same_class = {"A": ("2.1.1.1", "2.7.1.1"), "B": ("3.1.1.1",)}
    assert clean_sanity.structural_tie_groups(same_class)["n_cross_class_groups"] == 0


def test_records_hold_totals_only():
    table, embeddings = training_world(members_per_ec=25)
    record = in_sample_record(table, embeddings)
    text = json.dumps(record)
    assert not any(accession in text for accession in table)  # no accession, only totals and hashes
    assert record["totals_only"] is True and record["thresholds"] == clean_sanity.THRESHOLDS
    assert clean_sanity.THRESHOLDS == {"IS_MIN_PRIMARY": "4/5", "IS_MAX_ANCHOR_DEFICIT": "1/20",
                                       "MAX_TIE_SHARE": "1/20", "HO_MAX_ANCHOR_DEFICIT": "1/10",
                                       "HO_MIN_MAJORITY_MARGIN": "1/10", "HO_PER_FIT_MAX_ANCHOR_DEFICIT": "1/5",
                                       "IN_SAMPLE_N": 256}
    assert clean_sanity.verify_record(record)["status"] == "passed"
    for edit, message in ((lambda r: r["thresholds"].update(IS_MIN_PRIMARY="1/2"), "thresholds differ"),
                          (lambda r: r.update(status="comparator_diagnosis_required"), "status does not follow"),
                          (lambda r: r["values"]["fit"].update(n_correct=10), "values do not follow"),
                          (lambda r: r.update(totals_only=False), "totals only")):
        edited = json.loads(text)
        edit(edited)
        with pytest.raises(SanityError, match=message):
            clean_sanity.verify_record(edited)
    table_h, embeddings_h, validation, r1_tables, r2_tables = _held_out_world()
    fits, queries = _held_out_fits(table_h, embeddings_h, validation, r1_tables, r2_tables)
    held = json.dumps(clean_sanity.evaluate_held_out(fits, queries, table_h))
    assert not any(accession in held for accession in table_h)


def test_module_is_torch_free_and_the_command_line_parses():
    code = (f"import sys; sys.path[:0] = [{str(ROOT / 'src')!r}, {str(ROOT / 'scripts')!r}]; "
            "import ec_clean30.clean_sanity; import runpy; "
            f"runpy.run_path({str(ROOT / 'scripts' / 'run_ec_clean30_clean_sanity.py')!r}, run_name='not_main'); "
            "print('torch' in sys.modules)")
    output = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert output.strip() == "False"
    help_text = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_ec_clean30_clean_sanity.py"), "--help"],
                               capture_output=True, text=True, check=True).stdout
    for action in ("sample", "in-sample", "held-out", "tie-exposure", "disposition", "verify-all"):
        assert action in help_text


def _cli(*argv):
    """The sanity runner in a child interpreter with the precision stub installed."""
    code = (precision.SUBPROCESS_PRELUDE.format(tests=str(ROOT / "tests"))
            + f"sys.argv = {[str(ROOT / 'scripts' / 'run_ec_clean30_clean_sanity.py'), *map(str, argv)]!r}; "
            + f"runpy.run_path({str(ROOT / 'scripts' / 'run_ec_clean30_clean_sanity.py')!r}, run_name='__main__')")
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_command_line_tie_exposure_and_disposition(tmp_path_factory):
    folder = tmp_path_factory.mktemp("sanity_cli")
    official = folder / "split30_train_split_0.csv"
    official.write_text("Entry\tEC number\tSequence\nA\t1.1.1.1;4.2.1.1\tMKV\nB\t2.1.1.1\tMKL\n"
                        "C\t5.1.1.1;6.1.1.1\tMAA\nD\t5.1.1.1;6.1.1.1\tMAG\n", encoding="utf-8")
    r2 = folder / "r2_fold0.txt"
    r2.write_text("C\nD\n", encoding="utf-8")
    script = str(ROOT / "scripts" / "run_ec_clean30_clean_sanity.py")
    out = folder / "tie_exposure.json"
    result = subprocess.run([sys.executable, script, "tie-exposure", "--official-training-file", str(official),
                             "--fold", "0", "--r2-list", str(r2), "--out", str(out)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    receipt = json.loads(out.read_text())
    assert receipt["folds"]["0"]["r1"]["n_cross_class_groups"] == 2 and receipt["folds"]["0"]["r2"][
        "n_cross_class_groups"] == 1
    assert json.loads(result.stdout)["fold_0"]["r1"]["n_proteins_in_cross_class_groups"] == 3
    again = subprocess.run([sys.executable, script, "tie-exposure", "--official-training-file", str(official),
                            "--fold", "0", "--out", str(out)], capture_output=True, text=True)
    assert again.returncode != 0 and "write-once" in again.stderr
    table, embeddings = training_world(members_per_ec=25)
    record = folder / "failed.json"
    clean_sanity.write_once(record, in_sample_record(table, embeddings, transform=_collapse))
    disposition = folder / "disposition.json"
    done = _cli("disposition", "--record", record, "--decision", "proceed_disclosed", "--maintainer",
                "the maintainer", "--note", "no defect found", "--out", disposition)
    assert done.returncode == 0, done.stderr
    assert clean_sanity.require_passed_or_disposed(record, [disposition])["disposition"] == "proceed_disclosed"


# ---------------------------------------------------------------------------
# Review fixes: per-arm held-out status and consequence (C6b), precision policy, verify-all


def _held_out_failing(arm_to_break):
    table, embeddings, validation, r1_tables, r2_tables = _held_out_world()
    fits, queries = _held_out_fits(table, embeddings, validation, r1_tables, r2_tables)
    broken, _ = _held_out_fits(table, embeddings, validation, r1_tables, r2_tables, transform=_flip)
    if arm_to_break == "r1_inner":
        fits = broken
    return clean_sanity.evaluate_held_out(fits, queries, table, created_utc="2026-11-04T00:00:00+00:00")


def test_the_held_out_record_has_a_status_per_arm_and_each_failed_arm_takes_its_own_disposition(tmp_path):
    record = _held_out_failing("r1_inner")
    assert record["arm_status"] == {"r1_inner": "comparator_diagnosis_required", "r2_inner": "passed"}
    path = tmp_path / "held_out.json"
    digest = clean_sanity.write_once(path, record)
    summary = clean_sanity.verify_record(path)
    assert summary["arm_status"] == record["arm_status"]
    assert set(summary["checkpoint_sha256_by_fit"]) == {"r1_inner", "r2_inner"}
    assert summary["checkpoint_sha256_by_fit"]["r1_inner"]["0"] == sha("checkpoint-r1_inner-0")
    with pytest.raises(SanityError, match="names one of its failed arms"):
        clean_sanity.verify_disposition(clean_sanity.build_disposition(
            record_sha256=digest, decision="proceed_disclosed", maintainer="the maintainer", note="n"),
            {digest: summary})
    with pytest.raises(SanityError, match="names one of its failed arms"):
        clean_sanity.verify_disposition(clean_sanity.build_disposition(
            record_sha256=digest, decision="proceed_disclosed", maintainer="the maintainer", note="n",
            arm="r2_inner"), {digest: summary})
    disposition = tmp_path / "r1.json"
    clean_sanity.write_once(disposition, clean_sanity.build_disposition(
        record_sha256=digest, decision="proceed_disclosed", maintainer="the maintainer", note="n", arm="r1_inner"))
    binding = clean_sanity.held_out_binding(path, [disposition])
    assert binding["arm_disposition"] == {"r1_inner": "proceed_disclosed", "r2_inner": None}
    assert binding["consequence"] == {"rule_id": "held_out_arm_consequence_v1", "disclosed_arms": ["r1_inner"],
                                      "unavailable_versus": ["r1"], "co_primaries_available": False}
    edited = dict(record, arm_status={"r1_inner": "passed", "r2_inner": "passed"})
    with pytest.raises(SanityError, match="per-arm statuses"):
        clean_sanity.verify_record(edited)


def test_an_r2_inner_disposition_removes_only_the_r2_hypotheses():
    assert clean_sanity.held_out_consequence({"r1_inner": None, "r2_inner": "proceed_disclosed"}) == {
        "rule_id": "held_out_arm_consequence_v1", "disclosed_arms": ["r2_inner"], "unavailable_versus": ["r2"],
        "co_primaries_available": True}
    assert clean_sanity.held_out_consequence({"r1_inner": None, "r2_inner": None})["unavailable_versus"] == []
    assert "C6b" == clean_sanity.clean_sanity_policy_record()["held_out_consequence"]["confirmation"]


def test_records_need_a_validated_precision_policy(monkeypatch, tmp_path):
    table, embeddings = training_world(members_per_ec=25)
    record = in_sample_record(table, embeddings)
    assert record["fit"]["precision_policy"]["record"]["cudnn_allow_tf32"] is False
    without = json.loads(json.dumps(record))
    del without["fit"]["precision_policy"]
    with pytest.raises(SanityError, match="records no precision policy"):
        clean_sanity.verify_record(without)
    tf32 = json.loads(json.dumps(record))
    tf32["fit"]["precision_policy"]["record"]["cuda_matmul_allow_tf32"] = True
    with pytest.raises(SanityError, match="does not validate"):
        clean_sanity.verify_record(tf32)
    log = tmp_path / "fit.log"
    log.write_text(precision.policy_log_text("cuda"))
    entry = clean_sanity.read_fit_policy(log, device="cuda")
    assert entry["record"]["device"] == "cuda" and entry["log_sha256"] == hashlib.sha256(log.read_bytes()).hexdigest()
    with pytest.raises(SanityError, match="does not validate"):
        clean_sanity.read_fit_policy(log, device="cpu")
    monkeypatch.setitem(sys.modules, "ec_clean30.runtime_precision", None)
    with pytest.raises(SanityError, match="needs ec_clean30.runtime_precision"):
        clean_sanity.verify_record(record)


def test_verify_all_writes_one_receipt_over_records_and_dispositions(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("sanity_all")
    table, embeddings = training_world(members_per_ec=25)
    passed = tmp_path / "passed.json"
    clean_sanity.write_once(passed, in_sample_record(table, embeddings))
    held = tmp_path / "held_out.json"
    held_sha = clean_sanity.write_once(held, _held_out_failing("r1_inner"))
    with pytest.raises(SanityError, match="has no disposition"):
        clean_sanity.verify_all([passed, held])
    disposition = tmp_path / "r1.json"
    clean_sanity.write_once(disposition, clean_sanity.build_disposition(
        record_sha256=held_sha, decision="proceed_disclosed", maintainer="the maintainer", note="n", arm="r1_inner"))
    receipt = clean_sanity.verify_all([passed, held], [disposition])
    rows = {row["check"]: row for row in receipt["records"]}
    assert rows["held_out"]["arm_outcome"] == {"r1_inner": "proceed_disclosed", "r2_inner": "passed"}
    assert rows["held_out"]["consequence"]["co_primaries_available"] is False
    assert rows["in_sample"]["outcome"] == "passed" and receipt["stopped"] == []
    out = tmp_path / "verify_all.json"
    result = _cli("verify-all", "--record", passed, "--record", held, "--disposition", disposition, "--out", out)
    assert result.returncode == 0, result.stderr
    assert json.loads(out.read_text())["n_records"] == 2
