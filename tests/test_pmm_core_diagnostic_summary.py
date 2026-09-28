"""Synthetic CPU diagnostic evidence: four/five/six native labels, no GPU."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import Batch, Data

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("summary_tool", ROOT / "summarize_pmm_core_diagnostic.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def fixture(tmp_path, target="five_class", recovered=False):
    import audit_pmm_replay as audit
    from audit_pmm_replay import graph_hash

    campaign = tmp_path / "campaign"
    diagnostic = campaign / "runtime" / "diagnostic"
    prepared = diagnostic / ("prepared_recovered" if recovered else "prepared")
    prepared.mkdir(parents=True)
    labels = summary.LABELS[target]
    identity = dict(family="only_gvp", target_scheme=target, readout="none", fold=0, model_seed=42,
                    source_tree_sha256="a" * 64, cohort_sha256="b" * 64, fold_membership_sha256="c" * 64)
    name = f"only_gvp__{target}__none__fold0__seed42"
    run = campaign / "runs" / name
    run.mkdir(parents=True)
    checkpoint = run / "best_model_checkpoint.pt"
    checkpoint.write_bytes(b"fixture checkpoint")
    write(run / "run_config.json", {"identity": identity})
    receipt = dict(selected_checkpoint_sha256=summary.sha(checkpoint), campaign_run_identity=identity)
    write(run / "selected_checkpoint.json", receipt)
    (campaign / "train_cohort.csv").write_text("fixture cohort")
    (campaign / "fold_membership.csv").write_text("fixture folds")
    logits = torch.full((2, len(labels)), -2., dtype=torch.float32)
    logits[0, 0], logits[1, 1] = 2., 2.
    native = torch.softmax(logits, -1)
    common = torch.cat([native[:, :3], native[:, 3:].sum(1, keepdim=True)], 1)
    item = dict(logits=logits, native=native, common4=common, y_native=torch.tensor([0, 1]),
                pred_native=native.argmax(1), pred_common4=common.argmax(1), seconds=0.1)
    uids = ["ion_a", "ion_b"]
    rows = []
    for i, uid in enumerate(uids):
        row = {"source_uid": uid, "y_native": str(i), "pred_native": str(i), "pred_common4": str(i)}
        for view, vocabulary in (("native", labels), ("common4", summary.FOUR)):
            row.update({f"p_{view}_{label.replace(' ', '_')}": f"{float(item[view][i,j]):.8f}"
                        for j, label in enumerate(vocabulary)})
        rows.append(row)
    for path in (run / "val_predictions.csv", run / "independent_validation_replay/val_predictions.csv"):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    batch = Batch.from_data_list([Data(x=torch.ones(2, 3), y_metal=torch.tensor([i])) for i in range(2)])
    schema = "pmm-replay-diagnostic-v1.1-disabled-ec-target" if recovered else "pmm-replay-diagnostic-v1"
    torch.save({"schema": schema, "batches": [batch], "ordered_uids": uids, "saved_rows": rows}, prepared / "input_snapshot.pt")
    audit_tool = diagnostic / "audit_pmm_replay.py"
    audit_tool.write_bytes(Path(audit.__file__).read_bytes())
    protocol, script = diagnostic / "protocol.md", diagnostic / "run_diagnostic.sh"
    protocol.write_text("Four fresh processes, five fixed passes each.")
    script.write_text("# fixture execution commands\n")
    examples = []
    for uid in uids:
        record = dict(source_uid=uid, fresh_fields={"y_ec": "fresh", "y_metal": "same"})
        if recovered:
            record.update(cached_fields={"y_ec": "cached", "y_metal": "same"}, cache_sha256="d" * 64)
        else:
            record.update(raw_field_differences={}, matching_cache_files=["fixture.pkl"], matching_cache_sha256=["d" * 64])
        examples.append(record)
    manifest = dict(schema=schema, status="predictive_inputs_matched", certifies_fit=False, training_performed=False,
                    normalization_refitted=False, campaign_run_identity=identity, run_dir=str(run),
                    tool_sha256=summary.sha(audit_tool), n_examples=2, n_batches=1, examples=examples,
                    batches=[{"fresh_sha256": graph_hash(batch)}],
                    snapshot_sha256=summary.sha(prepared / "input_snapshot.pt"),
                    input_files={str(run / file): summary.sha(run / file) for file in (
                        "best_model_checkpoint.pt", "selected_checkpoint.json", "val_predictions.csv", "run_config.json")},
                    cohort_file={"sha256": summary.sha(campaign / "train_cohort.csv")},
                    fold_file={"sha256": summary.sha(campaign / "fold_membership.csv")},
                    excluded_from_predictive_equality=["y_ec"] if recovered else [],
                    ambiguous_identical_content_examples=0, retrospective_limit="Fixture cache match is retrospective.")
    if recovered:
        previous = {**manifest, "schema": "pmm-replay-diagnostic-v1", "status": "input_mismatch"}
        write(diagnostic / "failed_input_manifest.json", previous)
        manifest.update(previous_failed_manifest_sha256=summary.sha(diagnostic / "failed_input_manifest.json"),
                        previous_tool_sha256=summary.sha(audit_tool),
                        previous_failed_manifest_path="original/failed/input_manifest.json",
                        disabled_ec_cpu_check=dict(predict_ec=False, predict_metal=True, logits_metal_bitwise_equal=True,
                                                   loss_bitwise_equal=True, model_state_sha256_before_and_after="e" * 64))
    write(prepared / "input_manifest.json", manifest)
    for condition in ("original", "strict"):
        for process in (1, 2):
            directory = diagnostic / f"{condition}_{process}"
            directory.mkdir()
            saved = []
            for i in range(5):
                path = directory / f"pass_{i:02d}.pt"
                torch.save(item, path)
                saved.append(dict(path=path.name, sha256=summary.sha(path), **{"pass": i}))
            references = {view: torch.tensor([[float(row[f"p_{view}_{label.replace(' ', '_')}"])
                                               for label in vocabulary] for row in rows], dtype=torch.float64)
                          for view, vocabulary in (("native", labels), ("common4", summary.FOUR))}
            torch.save(dict(ordered_uids=uids, passes=[item] * 5, saved_probabilities=references), directory / "forward_passes.pt")
            report = dict(schema=schema, status="diagnosis_complete", condition=condition, planned_repeats=5,
                          completed_repeats=5, deterministic_warn_only=False, deterministic_algorithms=condition == "strict",
                          input_manifest_sha256=summary.sha(prepared / "input_manifest.json"), tool_sha256=summary.sha(audit_tool),
                          input_hashes_before_and_after=[graph_hash(batch)], model_state_sha256_before_and_after="e" * 64,
                          preserved_passes=saved, forward_passes_sha256=summary.sha(directory / "forward_passes.pt"))
            write(directory / "evaluation_report.json", report)
    return SimpleNamespace(diagnostic_dir=diagnostic, campaign_dir=campaign, prepared_dir=prepared,
                           audit_tool=audit_tool, support_file=[protocol, script], reference=[])


@pytest.mark.parametrize("target", list(summary.LABELS))
@pytest.mark.parametrize("recovered", [False, True])
def test_all_twenty_passes_for_each_native_vocabulary(tmp_path, target, recovered):
    args = fixture(tmp_path, target, recovered)
    result = summary.summarize(args)
    assert result["completed_original_passes"] == result["completed_strict_passes"] == 10
    assert result["native_labels"] == list(summary.LABELS[target])
    assert result["all_classes_stable"] and not result["original_variable"] and not result["certifies_fit"]
    assert result["pairwise"]["all_conditions"]["native"]["pairs"] == 190
    assert len([p for p in result["evidence_files"] if Path(p["path"]).name == "input_manifest.json"]) == 1
    assert len([p for p in result["evidence_files"] if Path(p["path"]).name.startswith("pass_")]) == 20
    before = (args.diagnostic_dir / "diagnostic_summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        summary.summarize(args)
    assert (args.diagnostic_dir / "diagnostic_summary.json").read_bytes() == before


@pytest.mark.parametrize("corruption", ["snapshot", "pass", "checkpoint", "protocol_missing", "missing_process",
                                       "wrong_condition", "wrong_count", "model_mutation", "wrong_uid"])
def test_corrupt_incomplete_or_unbound_evidence_refused(tmp_path, corruption):
    args = fixture(tmp_path)
    report_path = args.diagnostic_dir / "strict_2/evaluation_report.json"
    if corruption in {"snapshot", "pass", "checkpoint"}:
        paths = {"snapshot": args.prepared_dir / "input_snapshot.pt",
                 "pass": args.diagnostic_dir / "original_1/pass_00.pt",
                 "checkpoint": args.campaign_dir / "runs/only_gvp__five_class__none__fold0__seed42/best_model_checkpoint.pt"}
        paths[corruption].write_bytes(paths[corruption].read_bytes() + b"changed")
    elif corruption == "protocol_missing":
        args.support_file = args.support_file[1:]
    elif corruption == "missing_process":
        report_path.unlink()
    elif corruption == "wrong_uid":
        arrays_path = args.diagnostic_dir / "strict_2/forward_passes.pt"
        arrays = torch.load(arrays_path, weights_only=False)
        arrays["ordered_uids"].reverse()
        torch.save(arrays, arrays_path)
        report = summary.read(report_path)
        report["forward_passes_sha256"] = summary.sha(arrays_path)
        write(report_path, report)
    else:
        report = summary.read(report_path)
        key, value = {"wrong_condition": ("condition", "original"), "wrong_count": ("completed_repeats", 4),
                      "model_mutation": ("model_state_sha256_before_and_after", "f" * 64)}[corruption]
        report[key] = value
        write(report_path, report)
    with pytest.raises((ValueError, FileNotFoundError)):
        summary.summarize(args)
    assert not (args.diagnostic_dir / "diagnostic_summary.json").exists()


def test_all_archived_exports_are_bound_even_when_byte_identical(tmp_path):
    args = fixture(tmp_path)
    run = args.campaign_dir / "runs/only_gvp__five_class__none__fold0__seed42"
    archive = run / "_incomplete_replays/attempt_123/val_predictions.csv"
    archive.parent.mkdir(parents=True)
    archive.write_bytes((run / "val_predictions.csv").read_bytes())
    result = summary.summarize(args)
    assert result["reference_files"] == 3 and result["distinct_reference_exports"] == 1
    assert any("attempt_123/val_predictions.csv" in row["path"] for row in result["evidence_files"])


def test_one_ulp_logit_probability_rounding_allowed_but_large_or_nonfinite_rejected(tmp_path):
    args = fixture(tmp_path)
    item = torch.load(args.diagnostic_dir / "original_1/pass_00.pt", weights_only=False)
    labels = summary.LABELS["five_class"]
    rows = [{"y_native": "0"}, {"y_native": "1"}]
    item["native"][0, 0] = torch.nextafter(item["native"][0, 0], torch.tensor(1.))
    item["common4"] = torch.cat((item["native"][:, :3], item["native"][:, 3:].sum(1, keepdim=True)), 1)
    summary.check_pass(item, uids=["a", "b"], rows=rows, labels=labels)
    item["native"][0, 0] -= 0.001
    item["native"][0, 1] += 0.001
    with pytest.raises(ValueError, match="saved logits"):
        summary.check_pass(item, uids=["a", "b"], rows=rows, labels=labels)
    item["native"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        summary.check_pass(item, uids=["a", "b"], rows=rows, labels=labels)
