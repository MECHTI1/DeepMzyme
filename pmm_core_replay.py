"""CPU-only, append-only core replay qualification; never alters legacy receipts.

Import is standard-library only. Call bind_frozen_source before scientific imports.
The prospective engineering contract is bound before fitting. Nine pinned historical
fold-0 fits use a separately labeled retrospective integration, retaining all failures.
This module neither launches fits nor grants promotion or held-out access.
"""
from __future__ import annotations
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import sys

FOUR = ("Mn", "Cu", "Zn", "Class VIII")
FIVE = ("Mn", "Cu", "Zn", "Fe", "Class VIII")  # native Class VIII is Co+Ni
SIX = ("Mn", "Cu", "Zn", "Fe", "Co", "Ni")
NATIVE_LABELS = {"four_class": FOUR, "five_class": FIVE, "six_class": SIX}
FAMILIES = ("only_esm", "only_gvp", "gvp_late_fusion")
SOURCE_SHA256 = "adc95c42261448dc9d35572a138a3b8349de79124d9708618f511c27a848dd23"
DIAGNOSTIC_SCHEMA = "pmm-replay-diagnostic-v1.1-disabled-ec-target"
DIAGNOSTIC_TOOL_SHA256 = "e02809f4df6bf5964ef939c86f34de7a6c994716f9160726f0e1879a1c2e0cdd"
DIAGNOSTIC_INDEX = Path("runtime/core_replay_v1/historical_diagnostics.json")
HISTORICAL_FAILED = {"only_gvp__five_class__none__fold0__seed42", "only_gvp__six_class__none__fold0__seed42"}
HISTORICAL_PINS = {'gvp_late_fusion__five_class__none__fold0__seed42': {'checkpoint_sha256': 'ac81027cec32cacbe73ddd34e599b666536037a59f7f924bb937fc4d09370681',
                                                      'predictions_sha256': 'ef844c02183f9418356bfa9c918d4058816a73cde542b75f84717205f0d7fe8f',
                                                      'replay_prediction_sha256s': ['85192a84187b4ca6a97fa9189b2a07c295b883b618f866b43cc418046b29edea']},
 'gvp_late_fusion__four_class__none__fold0__seed42': {'checkpoint_sha256': '6142d5b4fa8e4447b77fc4fbd4b036d82cd9cab2e44773befe4bb1d99a8e9514',
                                                      'predictions_sha256': '4331306b2dcb68d6ff6bae220654cb52d75cf62fed343fd3ae29b678b2e90f92',
                                                      'replay_prediction_sha256s': ['6491baf375322221e7521950b1d59397054c46bb30045c6baf21b434ff3c78e9']},
 'gvp_late_fusion__six_class__none__fold0__seed42': {'checkpoint_sha256': 'b596db995bc3d929cffeb9abaa94f1ba6c275872b7214cbfda637921cc3ec719',
                                                     'predictions_sha256': 'f16a84f50837e468f20211d272f6433e1e16e13b0fd5e9100cb16fd089ea84c9',
                                                     'replay_prediction_sha256s': ['159cedcf739a733a43b7dde47f724cd92c4d0d877040cedb2c80875e02ba649e']},
 'only_esm__five_class__none__fold0__seed42': {'checkpoint_sha256': '4db74f19cc401ef30ef31662c005dc99059132171c2d4345297a860880aeb3bc',
                                               'predictions_sha256': 'ea82a4790a6a18187104756cc2a0d96b7b50b54e00dc18e6843aef3ab16ad6f9',
                                               'replay_prediction_sha256s': ['3dfabee7a1289f1be9f11c6219887c179f123c099be4dd19d55b3ff17974ad0e']},
 'only_esm__four_class__none__fold0__seed42': {'checkpoint_sha256': 'a0af72fd1ed1d2830c429c9ec74386275a283f4833d238d90dce7962096a9839',
                                               'predictions_sha256': '94cd57b402be37edd88c7cfa6625b6ac6101be60e14b701411eb04bea0a67bf1',
                                               'replay_prediction_sha256s': ['2fb9b03051ce9f9c83dbb7e5678084b45103a4091227c26d2b282e56da2afd24']},
 'only_esm__six_class__none__fold0__seed42': {'checkpoint_sha256': 'edf67d8861d9ff70137360834a0ddce552f546feb5e8c0fecf6fc51763305c6d',
                                              'predictions_sha256': 'ff8a4a5b43c1e80cc009c7f43cc6705e9faa9a1d3c47dc62f33395f8451a804a',
                                              'replay_prediction_sha256s': ['97858fe4414ad928e28f412ea8edc9bc80b73ae19ce3afa11db6ba83e795df63']},
 'only_gvp__five_class__none__fold0__seed42': {'checkpoint_sha256': '0e53c06eb58210f92e9cec4a40a4840135a3cc356eead88d59b12ae981ebddf6',
                                               'predictions_sha256': 'f03c50dfed6bc8d297a138ab300c3656fa8017f59997b50aaff5ad967adea515',
                                               'replay_prediction_sha256s': ['886eb1512f0437352a9a94b548ac0aa9eb5119ec26119a4b75dbd25234fca9c4']},
 'only_gvp__four_class__none__fold0__seed42': {'checkpoint_sha256': 'c258412690b0451d27796f71a9b0ada0a46f6a8a93bf2ae4bc977c8997700de0',
                                               'predictions_sha256': '75347d15be9fd28ef5f39ce33ccd4960e25000aeac4898dea7f19fb5c132f8a4',
                                               'replay_prediction_sha256s': ['c7a1ea39db08259d769406a4f7b9b852abd1d0f992fdcbc83fc01355787b7b16']},
 'only_gvp__six_class__none__fold0__seed42': {'checkpoint_sha256': 'a07a608924a19af7561ffe7e40b067db37bd4b99ae1c6d3fd42c790cbf7fabc0',
                                              'predictions_sha256': 'ac4e4ef6e8aa35fb54e47ec16c9b729a105b64d7468e28799c31dd43b68477bf',
                                              'replay_prediction_sha256s': ['10cd28af1d491b93167ca4411d044c71cb867d7723b3012d1fdb8360ddb8fa86',
                                                                            'ad792528c497ed29334e2358f02509296a6eb0fb3794c4ee91153e27b7c21f90']}}
POLICY_ID = "pmm-core-replay-v1"
POLICY = {
    "policy_id": POLICY_ID, "source_tree_sha256": SOURCE_SHA256,
    "campaign_id": "pmm_ion_metal_v2_context", "families": list(FAMILIES),
    "targets": list(NATIVE_LABELS), "readout": "none", "seed": 42, "epochs": 50,
    "prospective_folds": [1, 2, 3, 4], "historical_pins": HISTORICAL_PINS,
    "probability_atol": "0.00001", "probability_rtol": "0",
    "legacy_probability_atol": "0.000001", "metric_atol": 1e-9,
    "serialized_probability_consistency_atol": "0.0000002", "probability_simplex_atol": "0.000002",
    "all_preserved_attempts_required": True, "exact_classes_and_confusions": True,
    "historical_failure_diagnostic_processes": {"original": 2, "strict": 2, "passes_each": 5},
    "historical_acceptance_is_post_observation": True,
    "interpretation": "Fixed engineering agreement convention, not a proved numerical error bound or determinism claim.",
    "promotion_or_heldout_authorized": False,
}
POLICY_SHA256 = hashlib.sha256(json.dumps(POLICY, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class ReplayValidationError(ValueError):
    """A core unit lacks the evidence required by the frozen replay contract."""


def require(condition, message):
    if not condition:
        raise ReplayValidationError(message)


def bind_frozen_source(source_root):
    root = Path(source_root).resolve()
    digest = hashlib.sha256()
    files = sorted([*(root / "src").rglob("*.py"), *(root / "scripts").glob("*.py")])
    for path in files:
        if "__pycache__" not in path.parts:
            digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    require(digest.hexdigest() == SOURCE_SHA256, "Frozen scientific source hash mismatch")
    names = {p.stem for p in (root / "src").glob("*.py")} | {p.name for p in (root / "src").iterdir() if p.is_dir()}
    for name, module in tuple(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if name.split(".")[0] in names and file:
            require(Path(file).resolve().is_relative_to(root / "src"),
                    f"Scientific module already loaded from another tree: {name}")
    sys.dont_write_bytecode = True
    if str(root / "src") in sys.path:
        sys.path.remove(str(root / "src"))
    sys.path.insert(0, str(root / "src"))
    return root


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def checked_file(path, expected):
    path = Path(path).resolve()
    require(path.is_file() and sha(path) == expected, f"Artifact hash mismatch: {path}")
    return {"path": str(path), "sha256": expected}


def read_rows(path):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        require(len(columns) == len(set(columns)) and columns, f"Repeated/empty CSV columns: {path}")
        rows = list(reader)
    require(rows and all(set(row) == set(columns) and None not in row.values() for row in rows),
            f"Malformed/empty predictions: {path}")
    require(len({row["source_uid"] for row in rows}) == len(rows), f"Duplicate prediction UIDs: {path}")
    return rows


def validate_probabilities(rows, labels, policy):
    """Decimal checks keep the frozen absolute boundary independent of float rounding."""
    allowance = Decimal(policy["serialized_probability_consistency_atol"])
    simplex_allowance = Decimal(policy["probability_simplex_atol"])
    columns = {"p_native_" + label.replace(" ", "_") for label in labels}
    columns |= {"p_common4_" + label.replace(" ", "_") for label in FOUR}
    minimum_margins = {"native": Decimal(1), "common4": Decimal(1)}
    for row in rows:
        require({key for key in row if key.startswith("p_")} == columns, "Probability vocabulary mismatch")
        views = {}
        for view, vocabulary in (("native", labels), ("common4", FOUR)):
            try:
                values = [Decimal(row[f"p_{view}_{label.replace(' ', '_')}"]) for label in vocabulary]
            except InvalidOperation as exc:
                raise ReplayValidationError("Malformed probability") from exc
            require(all(value.is_finite() and 0 <= value <= 1 for value in values), "Invalid finite probability range")
            require(abs(sum(values) - 1) <= simplex_allowance, "Probability vector does not sum to one")
            predicted = int(row[f"pred_{view}"])
            # Decimal serialization can tie an original FP32 argmax; no larger gap is accepted.
            require(predicted in range(len(values)) and max(values) == values[predicted],
                    "Prediction disagrees with serialized probability argmax")
            ordered = sorted(values, reverse=True)
            minimum_margins[view] = min(minimum_margins[view], ordered[0] - ordered[1])
            views[view] = values
        collapsed = views["native"][:3] + [sum(views["native"][3:])] if len(labels) > 4 else views["native"]
        require(all(abs(a - b) <= allowance for a, b in zip(collapsed, views["common4"])),
                "Native-to-common-four collapse mismatch")
    return {key: float(value) for key, value in minimum_margins.items()}


def compare_rows(original, replay, labels, policy):
    validate_probabilities(original, labels, policy)
    margins = validate_probabilities(replay, labels, policy)
    left = {row["source_uid"]: row for row in original}
    right = {row["source_uid"]: row for row in replay}
    require(len(left) == len(original) and len(right) == len(replay) and left.keys() == right.keys(),
            "Missing, extra or duplicate replay UIDs")
    atol, legacy = Decimal(policy["probability_atol"]), Decimal(policy["legacy_probability_atol"])
    worst = {"abs_difference": 0.0, "source_uid": None, "column": None}
    maximum = Decimal(0)
    over_legacy = over_policy = 0
    for uid, row in left.items():
        other = right[uid]
        require(row.keys() == other.keys(), f"Replay columns differ at {uid}")
        for key, value in row.items():
            if key.startswith("p_"):
                difference = abs(Decimal(value) - Decimal(other[key]))
                over_legacy += difference > legacy
                over_policy += difference > atol
                if difference > maximum:
                    maximum = difference
                    worst = {"abs_difference": float(difference), "source_uid": uid, "column": key}
            else:
                require(value == other[key], f"Replay discrete/metadata mismatch: {uid}/{key}")
    return {"qualified": over_policy == 0, "worst": worst, "above_legacy_tolerance_fields": over_legacy,
            "above_policy_tolerance_fields": over_policy, "minimum_prediction_margins": margins,
            "n_rows": len(original)}


def same_metrics(left, right, tolerance=1e-9):
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            same_metrics(left[key], right[key], tolerance) for key in left)
    if isinstance(left, list):
        return isinstance(right, list) and len(left) == len(right) and all(
            same_metrics(a, b, tolerance) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and not isinstance(left, bool):
        return isinstance(right, (int, float)) and not isinstance(right, bool) and math.isfinite(left) and math.isfinite(right) and abs(left - right) <= tolerance
    return type(left) is type(right) and left == right


def diagnostic_gate(path, *, identity, run_name, checkpoint_sha256, n_examples, original_rows=None):
    policy = {**POLICY, "validation_ions": n_examples}
    path = Path(path).resolve()
    summary = read_json(path)
    for key in ("predictive_inputs_matched", "all_classes_stable", "no_mutation"):
        require(summary.get(key) is True, f"Diagnostic gate not met: {key}")
    require(summary.get("source_tree_sha256") == policy["source_tree_sha256"], "Diagnostic scientific source differs")
    require(summary.get("run_name") == run_name and summary.get("checkpoint_sha256") == checkpoint_sha256, "Wrong diagnostic arm")
    difference = summary.get("original_max_abs_probability_difference")
    require(isinstance(difference, (float, int)) and math.isfinite(difference)
            and 0 <= difference <= float(policy["probability_atol"]), "Diagnostic probability bound not met")
    bound_files = [checked_file(path, sha(path))]
    evidence = summary.get("evidence_files", [])
    require(evidence, "Diagnostic evidence files are absent")
    for item in evidence:
        bound_files.append(checked_file(path.parent / item["path"], item["sha256"]))
    manifests = [Path(item["path"]) for item in bound_files if Path(item["path"]).name == "input_manifest.json"]
    require(len(manifests) == 1, "Exactly one bound diagnostic input manifest is required")
    manifest = read_json(manifests[0])
    require(manifest.get("schema") in ("pmm-replay-diagnostic-v1", DIAGNOSTIC_SCHEMA), "Unknown diagnostic input schema")
    require(manifest.get("campaign_run_identity") == identity, "Diagnostic full scientific identity differs")
    require(manifest.get("cohort_file", {}).get("sha256") == identity["cohort_sha256"]
            and manifest.get("fold_file", {}).get("sha256") == identity["fold_membership_sha256"],
            "Diagnostic cohort/fold hash differs")
    effect = {}
    if manifest["schema"] == DIAGNOSTIC_SCHEMA:
        require(manifest.get("excluded_from_predictive_equality") == ["y_ec"],
                "Diagnostic input exclusions exceed the disabled-EC-target amendment")
        effect = manifest.get("disabled_ec_cpu_check", {})
        require(effect.get("predict_ec") is False and effect.get("predict_metal") is True
                and effect.get("logits_metal_bitwise_equal") is True and effect.get("loss_bitwise_equal") is True,
                "Disabled-EC standalone-model and CPU equivalence gates did not pass")
        require(manifest.get("certifies_fit") is False and manifest.get("training_performed") is False
                and manifest.get("normalization_refitted") is False, "Diagnostic purpose or fit invariants differ")
        previous_hash = manifest.get("previous_failed_manifest_sha256")
        previous_files = [Path(item["path"]) for item in bound_files if item["sha256"] == previous_hash]
        require(isinstance(previous_hash, str) and len(previous_hash) == 64 and len(previous_files) == 1,
                "Previous failed diagnostic manifest is not uniquely hash-bound")
        previous = read_json(previous_files[0])
        require(previous.get("schema") == "pmm-replay-diagnostic-v1" and previous.get("status") == "input_mismatch"
                and previous.get("campaign_run_identity") == manifest.get("campaign_run_identity")
                and previous.get("input_files") == manifest.get("input_files"),
                "Previous failed diagnostic source or input identity differs")
        old_tool = manifest.get("previous_tool_sha256")
        require(isinstance(old_tool, str) and len(old_tool) == 64 and old_tool == previous.get("tool_sha256")
                and any(item["sha256"] == old_tool for item in bound_files),
                "Previous failed diagnostic tool source is not hash-bound")
    else:
        require(not manifest.get("excluded_from_predictive_equality"), "Exact-input diagnostic cannot exclude fields")
    require(manifest.get("certifies_fit") is False and manifest.get("training_performed") is False
            and manifest.get("normalization_refitted") is False, "Diagnostic purpose or fit invariants differ")
    require(manifest.get('tool_sha256') == DIAGNOSTIC_TOOL_SHA256
            and any(item["sha256"] == manifest.get("tool_sha256") for item in bound_files),
            "Current diagnostic tool source is not hash-bound")
    require(manifest.get("status") == "predictive_inputs_matched"
            and manifest.get("n_examples") == policy["validation_ions"]
            and manifest["campaign_run_identity"]["source_tree_sha256"] == policy["source_tree_sha256"],
            "Diagnostic input manifest did not pass or describes other inputs")
    require(summary["checkpoint_sha256"] in [digest for name, digest in manifest["input_files"].items()
                                             if Path(name).name == "best_model_checkpoint.pt"],
            "Diagnostic checkpoint hash is not bound to its inputs")
    bound_files.append(checked_file(manifests[0].parent / "input_snapshot.pt", manifest["snapshot_sha256"]))
    require(not manifest.get('unmatched_uids'), 'Diagnostic retains unmatched validation inputs')
    examples = manifest.get('examples', [])
    require(len(examples) == n_examples, 'Diagnostic raw graph evidence count differs')
    allowed = set(manifest.get('excluded_from_predictive_equality', []))
    cache_records = {item['path']: item for item in manifest.get('matched_cache_files', [])}
    for example in examples:
        fresh = example['fresh_fields']
        if manifest['schema'] == DIAGNOSTIC_SCHEMA:
            cached = example['cached_fields']
            require(len(example.get('cache_sha256', '')) == 64, 'Diagnostic cache file hash missing')
        else:
            candidates = example.get('matching_cache_files', [])
            require(candidates and not example.get('raw_field_differences'), 'Exact raw cache match absent')
            require(example.get('matching_cache_sha256') == [cache_records[name]['sha256'] for name in candidates],
                    'Diagnostic matched cache hashes differ')
            require(all(cache_records[name]['fields'] == fresh for name in candidates), 'Diagnostic raw cache fields differ')
            cached = cache_records[candidates[0]]['fields']
        require(fresh.keys() == cached.keys() and 'y_metal' in fresh
                and all(fresh[key] == cached[key] for key in fresh if key not in allowed),
                'Diagnostic raw graph field equality failed')
    if original_rows is not None:
        require([row['source_uid'] for row in examples] == [row['source_uid'] for row in original_rows],
                'Diagnostic graph UID order differs')
    input_hashes = [item["fresh_sha256"] for item in manifest["batches"]]
    require(input_hashes and len(input_hashes) == manifest["n_batches"], "Missing diagnostic batch identities")
    for batch in manifest['batches']:
        require(set(batch.get('field_differences', {})).issubset(allowed), 'Normalized predictive batch fields differ')
        if not allowed:
            require(batch['fresh_sha256'] == batch['cached_sha256'], 'Normalized batch hashes differ')
    model_hashes, all_passes = [], []
    processes = summary.get("processes", [])
    require(len(processes) == 4 and all(sum(item.get("condition") == condition for item in processes) == 2
                                      for condition in ("original", "strict")), "Diagnostic needs two processes per condition")
    report_paths = []
    for item in processes:
        report_path = (path.parent / item["report_path"]).resolve()
        report_paths.append(report_path)
        bound_files.append(checked_file(report_path, item["report_sha256"]))
        report = read_json(report_path)
        require(report.get("schema") == manifest["schema"], "Unknown diagnostic evaluation schema")
        require(report.get("input_manifest_sha256") == sha(manifests[0])
                and report.get("tool_sha256") == manifest["tool_sha256"], "Diagnostic report input/tool hash differs")
        require(report.get("condition") == item["condition"] and report.get("planned_repeats") == 5,
                "Diagnostic process condition/repeats mismatch")
        require(report.get("deterministic_warn_only") is False
                and report.get("deterministic_algorithms") is (item["condition"] == "strict"),
                "Diagnostic effective determinism settings differ")
        preserved = report.get("preserved_passes", [])
        require(len(preserved) == item["completed_passes"] == report.get("completed_repeats"),
                "Diagnostic preserved-pass count differs")
        require([saved.get('pass') for saved in preserved] == list(range(len(preserved)))
                and len({(report_path.parent / saved['path']).resolve() for saved in preserved}) == len(preserved),
                'Diagnostic pass indices or files are reused')
        if item["status"] == "passed":
            require(item["completed_passes"] == 5 and report.get("status") == "diagnosis_complete",
                    "Diagnostic process did not complete all five passes")
            require(report.get("input_hashes_before_and_after") == input_hashes,
                    "Diagnostic input invariant hashes differ")
            model_hashes.append(report.get("model_state_sha256_before_and_after"))
            require(len(report.get("comparisons_to_saved", [])) == 5
                    and len(report.get("pairwise_comparisons", [])) == 10, "Missing diagnostic comparisons")
            for comparisons in (report.get("comparisons_to_saved", []), report.get("pairwise_comparisons", [])):
                require(comparisons, "Diagnostic probability comparisons missing")
                for comparison in comparisons:
                    for view in ("native", "common4"):
                        require(comparison[view]["changed_predictions"] == 0, "Diagnostic class predictions changed")
                        require(math.isfinite(comparison[view]["max_abs_difference"])
                                and 0 <= comparison[view]["max_abs_difference"] <= float(policy["probability_atol"]),
                                "Diagnostic pass exceeds fixed bound")
        else:
            require(item["condition"] == "strict" and item["status"] == "unsupported"
                    and report.get("status") == "evaluation_failed" and "determin" in report.get("error", "").lower(),
                    "Only explicitly recorded unsupported strict evaluation can replace successful passes")
        for saved in preserved:
            bound_files.append(checked_file(report_path.parent / saved["path"], saved["sha256"]))
        if report.get("forward_passes_sha256"):
            bound_files.append(checked_file(report_path.parent / "forward_passes.pt", report["forward_passes_sha256"]))
        if item["status"] == "passed":
            require(report.get("forward_passes_sha256"), "Completed diagnostic has no preserved arrays")
        if original_rows is not None:
            all_passes.extend(verify_diagnostic_arrays(report_path, report, original_rows, NATIVE_LABELS[identity["target_scheme"]]))
    require(len(set(report_paths)) == 4, "Diagnostic processes reuse a report")
    require(model_hashes and isinstance(model_hashes[0], str) and len(model_hashes[0]) == 64
            and len(set(model_hashes)) == 1, "Diagnostic model state invariant hashes differ")
    if effect:
        require(effect.get("model_state_sha256_before_and_after") == model_hashes[0],
                "CPU equivalence check used a different model state")
    for index, left in enumerate(all_passes):
        for right in all_passes[index + 1:]:
            require(all((left[view].double() - right[view].double()).abs().max().item()
                        <= float(POLICY['probability_atol']) for view in ('native', 'common4')),
                    'Diagnostic cross-process arrays exceed fixed bound')
    return summary, bound_files


def verify_diagnostic_arrays(report_path, report, rows, labels):
    """Recompute class/target/order and numerical agreement from preserved CPU arrays."""
    import torch
    expected_uids = [row['source_uid'] for row in rows]
    references = {view: torch.tensor([[float(row[f'p_{view}_{label.replace(" ", "_")}'])
                                      for label in vocabulary] for row in rows], dtype=torch.float64)
                  for view, vocabulary in (('native', labels), ('common4', FOUR))}
    passes = []
    for item in report['preserved_passes']:
        result = torch.load(report_path.parent / item['path'], map_location='cpu', weights_only=False)
        require(result['y_native'].tolist() == [int(row['y_native']) for row in rows], 'Diagnostic target order differs')
        for view in references:
            values = result[view]
            require(values.dtype == torch.float32 and values.shape == references[view].shape
                    and torch.isfinite(values).all().item() and ((values >= 0) & (values <= 1)).all().item(),
                    'Diagnostic array dtype/shape/probabilities differ')
            require(values.argmax(1).tolist() == [int(row[f'pred_{view}']) for row in rows],
                    'Diagnostic array classes differ')
            require((values.double() - references[view]).abs().max().item() <= float(POLICY['probability_atol']),
                    'Diagnostic array exceeds fixed bound')
            for previous in passes:
                require((values.double() - previous[view].double()).abs().max().item() <= float(POLICY['probability_atol']),
                        'Diagnostic pairwise arrays exceed fixed bound')
        require(torch.allclose(result['common4'], torch.cat((result['native'][:, :3],
                result['native'][:, 3:].sum(1, keepdim=True)), 1), atol=2e-7, rtol=0),
                'Diagnostic native/common-four arrays disagree')
        passes.append(result)
    if report.get('forward_passes_sha256'):
        bundled = torch.load(report_path.parent / 'forward_passes.pt', map_location='cpu', weights_only=False)
        require(bundled['ordered_uids'] == expected_uids and len(bundled['passes']) == len(passes),
                'Diagnostic bundle UID order/count differs')
        for view in references:
            require(torch.equal(bundled['saved_probabilities'][view], references[view]), 'Diagnostic saved reference differs')
        for left, right in zip(passes, bundled['passes']):
            require(all(torch.equal(left[key], right[key]) for key in ('native', 'common4', 'y_native')),
                    'Diagnostic bundled passes differ from individual files')
    return passes


def replay_directories(run, campaign_dir=None):
    run = Path(run)
    current = run / 'independent_validation_replay'
    found = {current}
    for path in run.rglob('*'):
        if not path.is_dir():
            continue
        if path.name.startswith('independent_validation_replay') or path.name.startswith('_incomplete_replay'):
            # Containers are traversed; every attempt, including an empty failed one, must be explained.
            if path.name == '_incomplete_replays':
                found.update(child for child in path.iterdir() if child.is_dir())
            else:
                found.add(path)
    if campaign_dir is not None:
        # Include preserved backup copies, not just replay attempts beside the live run.
        runtime = Path(campaign_dir) / 'runtime'
        preserved = set(runtime.rglob('selected_checkpoint.json')) | set(runtime.rglob('val_predictions.csv'))
        for path in preserved:
            if run.name in path.parts:
                trailing = path.parts[path.parts.index(run.name) + 1:-1]
                if any('replay' in part for part in trailing):
                    found.add(path.parent)
        for path in runtime.rglob('independent_validation_replay*'):
            if path.is_dir() and run.name in path.parts:
                found.add(path)
        for archived in (Path(campaign_dir) / 'runs/_incomplete_attempts').glob(run.name + '*'):
            require(not archived.exists(), 'An archived training attempt requires explicit reconciliation')
    directories = [current] + sorted(found - {current})
    require(all((path / "val_predictions.csv").is_file() and (path / "selected_checkpoint.json").is_file()
                for path in directories), "Preserved replay attempt lacks predictions/receipt")
    return directories


def verify_legacy_consistency(legacy_receipt_pass, current_comparison):
    require(not legacy_receipt_pass or current_comparison["above_legacy_tolerance_fields"] == 0,
            "Legacy receipt claims a pass but current predictions fail its unchanged tolerance")



def expected_core_identity(paths, train_dir, family, target, fold):
    """Rebuild the frozen recipe, including native-five common-four ion weights."""
    from benchmarking import pmm_ion_campaign as campaign
    from training.config import config_to_payload, parse_args

    require(family in FAMILIES and target in NATIVE_LABELS and type(fold) is int and fold in range(5),
            "Unsupported core selector")
    base_target = 'four_class' if target == 'five_class' else target
    command, _env, identity = campaign.build_train_command(
        paths, python_bin='python', train_dir=Path(train_dir), config=campaign.GridConfig(family, base_target, 'none'),
        fold=fold, seed=42, device='cuda', runs_dir=paths.runs)
    if target == 'five_class':
        index = command.index('--campaign-run-identity')
        del command[index:index + 2]
        command[command.index('--metal-label-scheme') + 1] = 'five_class'
        command += ['--fe-loss-multiplier', command[command.index('--class-viii-loss-multiplier') + 1]]
        resolved = config_to_payload(parse_args(command[3:]))
        identity.update(target_scheme=target, resolved_config_sha256=campaign.stable_hash(
            {key: value for key, value in resolved.items() if key not in campaign.NON_IDENTITY_CONFIG_KEYS}))
        if fold == 0:
            identity['five_class_screen'] = {
                'protocol_id': 'pmm-five-class-screen-v1',
                'protocol_sha256': 'c6d1da7fa0b7aa32b147f0bfbc4914a95ed5d127cdbf22d11716e26c42321607',
                'adapter_sha256': '5c038c30d3540ed88c893a973dd66ec11425d7f28e60c2aa6605b4b1fa508c8d'}
    if fold:
        identity['core_replay_contract'] = {'policy_id': POLICY_ID, 'policy_sha256': POLICY_SHA256}
    return identity


def validate_native_targets(rows, target):
    mapping = {'four_class': {'MN': 0, 'CU': 1, 'ZN': 2, 'FE': 3, 'CO': 3, 'NI': 3},
               'five_class': {'MN': 0, 'CU': 1, 'ZN': 2, 'FE': 3, 'CO': 4, 'NI': 4},
               'six_class': {'MN': 0, 'CU': 1, 'ZN': 2, 'FE': 3, 'CO': 4, 'NI': 5}}[target]
    for row in rows:
        element = row['native_element'].upper()
        require(element in mapping and int(row['y_native']) == mapping[element],
                'Native target differs from frozen element')


def check_history(history, receipt):
    require(len(history) == 50 and [row['epoch'] for row in history] == list(range(1, 51)),
            'Incomplete epoch history')
    require(all(math.isfinite(row['val_metal_balanced_acc']) for row in history), 'Non-finite native BA history')
    chosen = max(history, key=lambda row: row['val_metal_balanced_acc'])
    require(chosen['epoch'] == receipt['selected_epoch'] and receipt['selection_metric'] == 'val_metal_balanced_acc',
            'Native-BA checkpoint selection/earliest-tie rule differs')
    require(same_metrics(receipt['selection_metric_value'], chosen['val_metal_balanced_acc']),
            'Selected metric value differs from epoch history')
    return chosen


def historical_qualification(campaign_dir, run_name, identity, receipt, legacy, n_examples, original_rows=None):
    """Never upgrades the original strict status; reads a separate integration index."""
    pin = HISTORICAL_PINS[run_name]
    require(receipt['selected_checkpoint_sha256'] == pin['checkpoint_sha256']
            and receipt['validation_predictions']['sha256'] == pin['predictions_sha256'],
            'Historical unit differs from the nine frozen artifacts')
    failed = run_name in HISTORICAL_FAILED
    require(legacy is (not failed), 'Historical original strict status changed')
    evidence = {}
    if failed:
        index_path = Path(campaign_dir) / DIAGNOSTIC_INDEX
        require(index_path.is_file(), 'Historical failed unit needs input/repeatability diagnostic index')
        index = read_json(index_path)
        require(index.get('schema') == 'pmm-core-historical-diagnostics-v1'
                and index.get('policy_id') == POLICY_ID and index.get('policy_sha256') == POLICY_SHA256,
                'Historical diagnostic index contract differs')
        entry = index.get('units', {}).get(run_name)
        require(isinstance(entry, dict), 'Historical failed unit has no diagnostic evidence')
        summary_path = (Path(campaign_dir) / entry['summary_path']).resolve()
        require(not Path(entry['summary_path']).is_absolute()
                and summary_path.is_relative_to(Path(campaign_dir).resolve()), 'Diagnostic must be campaign-relative')
        checked_file(summary_path, entry['summary_sha256'])
        _summary, files = diagnostic_gate(summary_path, identity=identity, run_name=run_name,
                                         checkpoint_sha256=pin['checkpoint_sha256'], n_examples=n_examples,
                                         original_rows=original_rows)
        diagnostic_manifest = read_json(next(item['path'] for item in files if Path(item['path']).name == 'input_manifest.json'))
        inputs = {Path(name).name: digest for name, digest in diagnostic_manifest['input_files'].items()}
        run = Path(campaign_dir) / 'runs' / run_name
        require(all(inputs.get(name) == sha(run / name) for name in (
            'run_config.json', 'best_model_checkpoint.pt', 'selected_checkpoint.json', 'val_predictions.csv')),
            'Diagnostic saved input file hashes differ from this frozen fit')
        evidence.update({item['path']: item['sha256'] for item in files})
        evidence[str(index_path.resolve())] = sha(index_path)
    qualification = {
        'acceptance_basis': 'retrospective_core_agreement', 'post_observation': True,
        'original_strict_status': 'original_strict_failed' if failed else 'passed',
        'qualified_for_comparison': True, 'diagnostic_required': failed,
        'limitations': ['Historical agreement is post-observation; original strict receipts and failures are unchanged.',
                       'Input cache matching is retrospective; original in-memory training tensors were not fingerprinted.',
                       'This qualification alone does not authorize promotion, refit or held-out evaluation.']}
    return qualification, evidence


def _validate_core_unit(campaign_dir, train_dir, family, target, fold, *, source_root):
    require(hashlib.sha256(json.dumps(POLICY, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            == POLICY_SHA256, 'Replay policy mutated in memory')
    source_root = bind_frozen_source(source_root)
    import torch
    from dataclasses import fields
    from benchmarking import pmm_ion_campaign as campaign
    from benchmarking.pmm_comparator import read_campaign_contract, validate_prediction_rows
    from benchmarking.pmm_ion_analysis import prediction_metrics
    from training.config import TrainConfig
    from training.run import to_jsonable

    paths = campaign.CampaignPaths(Path(campaign_dir).resolve())
    manifest, membership, _cohort = read_campaign_contract(paths.root)
    require(manifest['campaign_id'] == POLICY['campaign_id'] and manifest.get('source_side') == 'train',
            'Wrong campaign or held-out source')
    identity = expected_core_identity(paths, train_dir, family, target, fold)
    require(identity['source_tree_sha256'] == SOURCE_SHA256, 'Frozen scientific source changed')
    run = paths.runs / f'{family}__{target}__none__fold{fold}__seed42'
    receipt = campaign.completed_run_receipt(run, identity, require_independent=False)
    require(receipt is not None, f'Completed fit identity/hash checks failed: {run.name}')
    bound = {str(run / name): sha(run / name) for name in (
        'run_config.json', 'run_metadata.json', 'dataset_summary.json', 'best_model_checkpoint.pt',
        'selected_checkpoint.json', 'val_predictions.csv')}
    for path in (paths.manifest, paths.cohort, paths.fold_membership, paths.fold_class_weights, paths.feature_inventory):
        bound[str(path)] = sha(path)
    bound[str(Path(__file__).resolve())] = sha(__file__)
    payload = read_json(run / 'run_config.json')
    checkpoint = torch.load(run / 'best_model_checkpoint.pt', map_location='cpu', weights_only=False)
    require(to_jsonable(checkpoint['config']) == payload['config'], 'Checkpoint/config mismatch')
    require(to_jsonable(checkpoint['normalization_stats']) == payload['normalization_stats'], 'Normalization mismatch')
    require(to_jsonable(checkpoint['dataset_summary']) == payload['dataset_summary']
            == read_json(run / 'dataset_summary.json'), 'Checkpoint/dataset-summary mismatch')
    cfg = payload['config']
    require(json.loads(cfg['campaign_run_identity']) == identity, 'Saved configuration identity differs')
    require(checkpoint['epoch'] == receipt['selected_epoch'], 'Checkpoint/receipt epoch mismatch')
    require(receipt['completed_epochs'] == receipt['planned_epochs'] == 50, 'Receipt epoch count differs')
    scientific = {field.name: cfg[field.name] for field in fields(TrainConfig)
                  if field.name not in campaign.NON_IDENTITY_CONFIG_KEYS}
    require(campaign.stable_hash(scientific) == identity['resolved_config_sha256'],
            'Resolved scientific configuration hash differs')
    require(cfg.get('task') == 'metal' and cfg.get('metal_example_unit') == 'ion', 'Wrong scientific task/unit')
    require(not any(cfg.get(key) for key in ('run_test_eval', 'test_structure_dir', 'test_summary_csv',
                                            'allow_final_refit_test_eval', 'allow_train_loss_test_eval_debug')),
            'Held-out configuration forbidden')
    chosen = check_history(payload['history'], receipt)
    labels = NATIVE_LABELS[target]
    require([checkpoint['metal_labels'][key] for key in sorted(checkpoint['metal_labels'], key=int)] == list(labels),
            'Checkpoint target vocabulary differs')
    del checkpoint
    original = read_rows(run / 'val_predictions.csv')
    require(receipt['selected_checkpoint'] == 'best_model_checkpoint.pt'
            and receipt['validation_predictions']['path'] == 'val_predictions.csv'
            and receipt['validation_predictions']['n_rows'] == len(original), 'Receipt artifact/count differs')
    require(all(int(row['selected_epoch']) == receipt['selected_epoch']
                and row['metal_label_scheme'] == cfg['metal_label_scheme']
                and row['binding_residue_pooling'] == 'none' for row in original),
            'Prediction epoch/target/readout differs from the fit')
    # The frozen comparator validates common-four identity; native five is handled explicitly here.
    validate_prediction_rows(original, membership, fold, seed=42,
                             checkpoint_sha256=receipt['selected_checkpoint_sha256'])
    validate_native_targets(original, target)
    margins = validate_probabilities(original, labels, POLICY)
    metrics = prediction_metrics(original, native_labels=labels)
    for view, prefix in (('native', 'val_metal_'), ('common4', 'val_metal_collapsed4_')):
        for key, field in (('balanced_accuracy', 'balanced_acc'), ('macro_f1', 'macro_f1')):
            require(same_metrics(metrics[view][key], receipt['metrics'][prefix + field]), 'Selected receipt metric mismatch')
        require(metrics[view]['confusion_matrix'] == chosen[prefix + 'confusion_matrix'], 'Selected confusion matrix mismatch')
    attempts = []
    directories = replay_directories(run, paths.root)
    if fold == 0:
        require({sha(directory / 'val_predictions.csv') for directory in directories}
                == set(HISTORICAL_PINS[run.name]['replay_prediction_sha256s']),
                'Historical preserved replay export set changed')
    if run.name == 'only_gvp__six_class__none__fold0__seed42':
        require(len({sha(directory / 'val_predictions.csv') for directory in directories}) >= 2,
                'Missing distinct preserved failed GVP6 replays')
    for directory in directories:
        replay_receipt = read_json(directory / 'selected_checkpoint.json')
        replay_csv = directory / 'val_predictions.csv'
        require(replay_receipt['validation_predictions']['path'] == replay_csv.name
                and replay_receipt['validation_predictions']['n_rows'] == len(original), 'Replay artifact/count differs')
        checked_file(replay_csv, replay_receipt['validation_predictions']['sha256'])
        for key in ('selected_checkpoint', 'selected_checkpoint_sha256', 'selected_epoch', 'campaign_run_identity',
                    'selection_metric', 'selection_metric_value', 'completed_epochs', 'planned_epochs',
                    'fit_status', 'reconciliation_status'):
            require(replay_receipt[key] == receipt[key], f'Replay receipt differs: {directory}/{key}')
        require(same_metrics(replay_receipt['metrics'], receipt['metrics']), 'Replay metrics differ')
        rows = read_rows(replay_csv)
        comparison = compare_rows(original, rows, labels, POLICY)
        require(comparison['qualified'], f'Replay probability difference exceeds fixed policy: {directory}')
        require(prediction_metrics(rows, native_labels=labels) == metrics, 'Replayed discrete metrics differ')
        attempts.append({'directory': str(directory), **comparison})
        for name in ('val_predictions.csv', 'selected_checkpoint.json', 'replay_receipt.json'):
            path = directory / name
            if path.is_file():
                bound[str(path)] = sha(path)
    legacy = campaign.completed_run_receipt(run, identity) is not None
    verify_legacy_consistency(legacy, attempts[0])
    if (run / 'independent_validation_replay/replay_receipt.json').exists():
        require(legacy, 'Present strict receipt is invalid; do not downgrade invalid evidence to absent')
    if fold == 0:
        qualification, extra = historical_qualification(paths.root, run.name, identity, receipt, legacy, len(original), original)
        bound.update(extra)
    else:
        qualification = {
            'acceptance_basis': 'prospective_core_agreement', 'post_observation': False,
            'original_strict_status': 'passed' if legacy else 'original_strict_failed',
            'qualified_for_comparison': True, 'diagnostic_required': False,
            'limitations': ['Fixed probability agreement convention does not prove deterministic execution.',
                           'This qualification alone does not authorize promotion, refit or held-out evaluation.']}
    return {'run_name': run.name, 'receipt': receipt, 'rows': original, 'metrics': metrics,
            'identity': identity, 'replay_policy_id': POLICY_ID, 'replay_policy_sha256': POLICY_SHA256,
            'replay_qualification': qualification, 'legacy_replay_pass': legacy,
            'attempts': attempts, 'minimum_prediction_margins': margins, 'evidence_files': bound}


def validate_core_unit(campaign_dir, train_dir, family, target, fold, *, source_root):
    """Return a qualified immutable unit record or fail closed; performs CPU reads only."""
    try:
        return _validate_core_unit(campaign_dir, train_dir, family, target, fold, source_root=source_root)
    except ReplayValidationError:
        raise
    except (ValueError, KeyError, TypeError, OSError, IndexError) as exc:
        raise ReplayValidationError(f'Missing, malformed or inconsistent core evidence: {exc}') from exc
