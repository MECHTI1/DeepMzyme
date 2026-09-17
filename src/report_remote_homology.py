"""Validation-only, paired reporting over predeclared sequence-remoteness bins.

This module never loads a model or selects a checkpoint. Sequence eligibility
must be frozen before prediction exports are joined. Missing target classes
make a full-task metric undefined, rather than changing its vocabulary.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


BINS = (">30", "(20,30]", "(15,20]", "<=15", "<=20")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def bound_file(base, entry):
    if not isinstance(entry, dict) or not entry.get("sha256"):
        raise ValueError("Every input artifact must have a path and SHA-256 binding")
    path = Path(entry["path"])
    if not path.is_absolute():
        path = base / path
    if sha256(path) != entry["sha256"]:
        raise ValueError(f"Artifact hash mismatch: {path}")
    return path


def verify_replay_receipt(base, run, counts_sha256):
    """Bind reproduced predictions to the counts frozen before inference."""
    receipt = load_json(bound_file(base, run["receipt"]))
    for record in (run, receipt):
        if record.get("validation_only") is not True or record.get("reproduction_passed") is not True:
            raise ValueError("Run lacks a successful validation-only reproduction receipt")
        if record.get("counts_freeze_sha256") != counts_sha256:
            raise ValueError("Prediction replay is not bound to the pre-prediction counts freeze")
    for field in ("run_id", "task", "family", "seed", "fold_id"):
        if receipt.get(field) != run.get(field):
            raise ValueError(f"Reproduction receipt belongs to a different {field}")
    if receipt.get("target_vocabulary") != run["target_vocabulary"]:
        raise ValueError("Reproduction receipt target vocabulary differs")
    for flag in ("training_performed", "optimizer_created", "checkpoint_reselection", "feature_generation", "normalization_refitted"):
        if receipt.get(flag) is not False:
            raise ValueError(f"Reproduction receipt does not certify inference-only behavior: {flag}")
    if run["task"] == "ec" and receipt.get("ec_aggregation") != "mean_logits":
        raise ValueError("EC replay must certify group-level logit averaging")
    artifact_name = "group_predictions" if run["task"] == "ec" else "pocket_predictions"
    artifact = run[artifact_name]
    recorded = receipt.get("prediction_artifacts", {}).get(artifact_name)
    if recorded is None or recorded.get("sha256") != artifact["sha256"]:
        raise ValueError("Prediction artifact is not bound to its reproduction receipt")
    prediction_file = bound_file(base, artifact)
    if prediction_file.resolve() != bound_file(base, recorded).resolve():
        raise ValueError("Prediction path differs from its reproduction receipt")
    return prediction_file


def normalized_bin(value):
    return str(value).replace("%", "").replace(" ", "").replace("≤", "<=")


def in_bin(row, bin_name):
    value = normalized_bin(row.get("bin", ""))
    return value in ("(15,20]", "<=15") if bin_name == "<=20" else value == bin_name


def classification_metrics(confusion):
    matrix = np.asarray(confusion, dtype=float)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    recall = np.divide(np.diag(matrix), support, out=np.full(len(support), np.nan), where=support > 0)
    denominator = support + predicted
    f1 = np.divide(2 * np.diag(matrix), denominator, out=np.zeros(len(support)), where=denominator > 0)
    complete = bool(np.all(support > 0))
    return {
        "balanced_accuracy": float(recall.mean()) if complete else None,
        "macro_f1": float(f1.mean()) if complete else None,
        "recall": [float(x) if np.isfinite(x) else None for x in recall],
        "support": support.tolist(),
        "confusion_matrix": matrix.tolist(),
        "full_vocabulary_supported": complete,
    }


def confusion_matrix(rows, n_classes):
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for row in rows:
        matrix[row["target"], row["prediction"]] += 1
    return matrix


def support_counts(rows, n_classes):
    return [{
        "target": target,
        "examples": len({row["example_id"] for row in rows if row["target"] == target}),
        "groups": len({row["group_id"] for row in rows if row["target"] == target}),
        "components": len({row["component_id"] for row in rows if row["target"] == target}),
    } for target in range(n_classes)]


def aggregate_ec_remoteness(rows):
    """Use the least-remote pocket and reject partially unmapped proteins."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["group_id"]].append(row)
    result = []
    for group, pockets in grouped.items():
        targets = {int(row["target"]) for row in pockets}
        components = {row["component_id"] for row in pockets}
        if len(targets) != 1 or len(components) != 1:
            raise ValueError(f"Inconsistent protein targets/components: {group}")
        valid = all(normalized_bin(row.get("bin", "")) in BINS for row in pockets)
        row = dict(pockets[0], example_id=group)
        if valid:
            nearest = max(pockets, key=lambda item: float(item["max_identity"]))
            row.update(bin=nearest["bin"], max_identity=nearest["max_identity"])
        else:
            row.update(bin=None, status="incomplete_group_remoteness", max_identity=None)
        result.append(row)
    return result


def join_predictions(predictions, remoteness, n_classes):
    def keyed(rows, kind):
        result = {}
        for row in rows:
            key = str(row["example_id"])
            if key in result:
                raise ValueError(f"Duplicate {kind} example: {key}")
            result[key] = row
        return result

    prediction_map = keyed(predictions, "prediction")
    remote_map = keyed(remoteness, "remoteness")
    if prediction_map.keys() != remote_map.keys():
        raise ValueError("Prediction and frozen validation memberships differ")
    joined = []
    for key, remote in remote_map.items():
        pred = prediction_map[key]
        target, prediction = int(pred["target"]), int(pred["prediction"])
        if not (0 <= target < n_classes and 0 <= prediction < n_classes):
            raise ValueError(f"Out-of-vocabulary label: {key}")
        if target != int(remote["target"]) or str(pred["group_id"]) != str(remote["group_id"]):
            raise ValueError(f"Target/group mismatch: {key}")
        if not remote.get("component_id"):
            raise ValueError(f"Missing sequence component: {key}")
        joined.append(dict(remote, target=target, prediction=prediction))
    return joined


def validate_pairs(left, right):
    def indexed(rows):
        result = {}
        for row in rows:
            key = (int(row["seed"]), str(row["example_id"]))
            if key in result:
                raise ValueError(f"Repeated example within a model seed (OOF overlap): {key}")
            result[key] = row
        return result

    lhs, rhs = indexed(left), indexed(right)
    if not lhs or lhs.keys() != rhs.keys():
        raise ValueError("Compared families must have identical validation examples and active seeds")
    for key, a in lhs.items():
        b = rhs[key]
        for field in ("target", "group_id", "component_id", "bin", "status", "fold_id"):
            if a.get(field) != b.get(field):
                raise ValueError(f"Paired {field} mismatch for {key}")
    per_seed = defaultdict(set)
    for seed, example in lhs:
        per_seed[seed].add(example)
    if any(examples != next(iter(per_seed.values())) for examples in per_seed.values()):
        raise ValueError("Every active model seed must cover the same evaluation examples")
    return [lhs[key] for key in sorted(lhs)], [rhs[key] for key in sorted(lhs)]


def paired_component_bootstrap(left, right, n_classes, *, replicates=10000,
                               seed=20260917, min_components=10, confidence=0.95):
    """Resample a component once across both families, bins and model seeds.

    Point estimates average seed-specific metrics, without forming an ensemble.
    Folds contribute out-of-fold examples within each seed. Draws missing any
    target class have no full-vocabulary metric and are counted explicitly.
    """
    left, right = validate_pairs(left, right)
    seeds = sorted({row["seed"] for row in left})
    components = sorted({row["component_id"] for row in left})
    component_index = {value: index for index, value in enumerate(components)}
    seed_index = {value: index for index, value in enumerate(seeds)}
    # component, family, seed, bin, true, predicted
    counts = np.zeros((len(components), 2, len(seeds), len(BINS), n_classes, n_classes), dtype=np.int64)
    for family, rows in enumerate((left, right)):
        for row in rows:
            for bi, bin_name in enumerate(BINS):
                if in_bin(row, bin_name):
                    counts[component_index[row["component_id"]], family, seed_index[row["seed"]], bi,
                           row["target"], row["prediction"]] += 1

    def differences(matrix):
        support = matrix.sum(axis=-1)
        diagonal = np.diagonal(matrix, axis1=-2, axis2=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ba = (diagonal / support).mean(axis=-1).mean(axis=1)
        return ba[0] - ba[1]

    point = differences(counts.sum(axis=0))
    draws = np.full((replicates, len(BINS) + 1), np.nan)
    generator = np.random.default_rng(seed)
    for index in range(replicates):
        multiplicity = generator.multinomial(len(components), np.full(len(components), 1 / len(components)))
        delta = differences(np.tensordot(multiplicity, counts, axes=(0, 0)))
        draws[index, :-1] = delta
        draws[index, -1] = delta[-1] - delta[0]
    estimates = np.r_[point, point[-1] - point[0]]
    result = {}
    for index, name in enumerate((*BINS, "interaction_<=20_minus_>30")):
        selected_bins = ("<=20", ">30") if index == len(BINS) else (name,)
        support = {bin_name: support_counts([r for r in left if in_bin(r, bin_name)], n_classes)
                   for bin_name in selected_bins}
        adequate = all(item["components"] >= min_components for entries in support.values() for item in entries)
        valid = draws[:, index][np.isfinite(draws[:, index])]
        stable = len(valid) >= replicates * 0.95
        alpha = (1 - confidence) / 2
        ci = np.quantile(valid, [alpha, 1 - alpha]).tolist() if len(valid) and stable else None
        result[name] = {
            "difference": float(estimates[index]) if np.isfinite(estimates[index]) else None,
            "paired_confidence_interval": ci,
            "confidence_level": confidence,
            "bootstrap_replicates": replicates,
            "valid_replicates": int(len(valid)),
            "support": support,
            "support_gate_passed": adequate,
            "interpretation": "eligible_component_interval" if adequate and stable and np.isfinite(estimates[index]) else "descriptive_or_inconclusive",
        }
    return {"model_seeds": seeds, "independent_components": len(components), "contrasts": result}


def remoteness_view(rows, *, audit=False):
    """Keep sensitivity strata separate while retaining the same dependence units."""
    prefix = "audit50_" if audit else ""
    result = []
    for row in rows:
        fields = ("bin", "status", "max_identity")
        if any(prefix + field not in row for field in fields):
            raise ValueError(f"Missing frozen {'audit50' if audit else 'primary'} remoteness fields")
        result.append(dict(row, **{field: row[prefix + field] for field in fields}))
    return result


def run_bin_report(run, joined, n_classes):
    bin_reports = {}
    for name in BINS:
        subset = [row for row in joined if in_bin(row, name)]
        bin_reports[name] = dict(classification_metrics(confusion_matrix(subset, n_classes)),
                                 unit_counts=support_counts(subset, n_classes))
    exclusions = defaultdict(int)
    for row in joined:
        if normalized_bin(row.get("bin", "")) not in BINS:
            exclusions[str(row.get("status", "unknown"))] += 1
    return {"run_id": run["run_id"], "family": run["family"], "seed": run["seed"],
            "fold_id": run.get("fold_id", "fixed"), "bins": bin_reports, "excluded": dict(exclusions)}


def family_comparisons(families, protocol, task, n_classes):
    statistics = protocol["statistics"]
    comparisons = {}
    for comparison in protocol.get("comparisons", []):
        if comparison.get("task", task) != task:
            continue
        left, right = comparison["left_family"], comparison["right_family"]
        if left not in families or right not in families:
            comparisons[comparison["name"]] = {"status": "missing_family_exports"}
            continue
        comparisons[comparison["name"]] = paired_component_bootstrap(
            families[left], families[right], n_classes, replicates=statistics["bootstrap_replicates"],
            seed=statistics["bootstrap_seed"], min_components=statistics["min_components_per_class"],
            confidence=statistics["confidence_level"])
    return comparisons


def build_report(task, protocol_path, prediction_path, remoteness_path, counts_path):
    protocol, exports = load_json(protocol_path), load_json(prediction_path)
    remote, frozen = load_json(remoteness_path), load_json(counts_path)
    if frozen.get("protocol_sha256") != sha256(protocol_path):
        raise ValueError("Protocol differs from the eligibility freeze")
    if frozen.get("remoteness_manifest_sha256") != sha256(remoteness_path):
        raise ValueError("Remoteness differs from the eligibility freeze")
    statistics = protocol.get("statistics", {})
    for key in ("bootstrap_replicates", "bootstrap_seed", "min_components_per_class", "confidence_level"):
        if key not in statistics:
            raise ValueError(f"Statistics must be predeclared in the frozen protocol: {key}")
    if statistics["bootstrap_replicates"] < 1 or statistics["min_components_per_class"] < 1 or not 0 < statistics["confidence_level"] < 1:
        raise ValueError("Invalid frozen statistical settings")
    if protocol.get("primary_shorter_coverage") != 0.8 or protocol.get("audit_shorter_coverage") != 0.5:
        raise ValueError("This report requires frozen 80% primary and 50% sensitivity coverage")
    if frozen.get("primary_shorter_sequence_coverage") != 0.8 or frozen.get("sensitivity_audit", {}).get("shorter_sequence_coverage") != 0.5:
        raise ValueError("Counts freeze must cover both primary and sensitivity eligibility")
    analyses = {"primary": {"families": defaultdict(list), "runs": []},
                "audit50": {"families": defaultdict(list), "runs": []}}
    vocabulary = None
    counts_sha256 = sha256(counts_path)
    for run in exports["runs"]:
        if run["task"] != task:
            continue
        labels = run["target_vocabulary"]
        if labels != protocol.get("vocabulary", {}).get(task):
            raise ValueError("Prediction vocabulary differs from the predeclared task endpoint")
        if vocabulary is not None and labels != vocabulary:
            raise ValueError("Cannot mix target vocabularies in one task report")
        vocabulary = labels
        prediction_file = verify_replay_receipt(Path(prediction_path).parent, run, counts_sha256)
        with prediction_file.open(newline="") as handle:
            predictions = list(csv.DictReader(handle))
        rows = [row for row in remote["rows"] if row["run_id"] == run["run_id"] and row["task"] == task]
        for name, analysis in analyses.items():
            selected = remoteness_view(rows, audit=name == "audit50")
            if task == "ec":
                selected = aggregate_ec_remoteness(selected)
            joined = join_predictions(predictions, selected, len(labels))
            for row in joined:
                row.update(seed=int(run["seed"]), fold_id=str(run.get("fold_id", "fixed")))
            analysis["families"][run["family"]].extend(joined)
            analysis["runs"].append(run_bin_report(run, joined, len(labels)))
    if not analyses["primary"]["runs"]:
        raise ValueError(f"No eligible validation exports for {task}")
    for analysis in analyses.values():
        analysis["comparisons"] = family_comparisons(analysis["families"], protocol, task, len(vocabulary))
    return {"schema_version": 1, "task": task, "validation_only": True,
            "endpoint": remote.get("endpoint", "uncertified"), "target_vocabulary": vocabulary,
            "scope": "conditional_development_reporting_no_promotion",
            "inputs": {str(Path(path).name): sha256(path) for path in
                       (protocol_path, prediction_path, remoteness_path, counts_path)},
            "primary_shorter_sequence_coverage": protocol["primary_shorter_coverage"],
            "runs": analyses["primary"]["runs"], "comparisons": analyses["primary"]["comparisons"],
            "sensitivity_audit": {"shorter_sequence_coverage": protocol["audit_shorter_coverage"],
                                  "role": "predeclared_domain_sensitive_secondary_analysis",
                                  "runs": analyses["audit50"]["runs"],
                                  "comparisons": analyses["audit50"]["comparisons"]},
            "limitations": ["Sequence identity is not a certificate of absence of homology.",
                            "Intervals are conditional on validation-based checkpoint and recipe selection.",
                            "Sequence components are bootstrap units; repeated model seeds are not new proteins.",
                            "A missing true class makes full-task balanced accuracy and macro-F1 undefined.",
                            "The 50% coverage sensitivity audit does not replace the 80% primary endpoint.",
                            "Component intervals do not replace Stage 6 paired-fold promotion gates."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("metal", "ec"), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--remoteness-manifest", type=Path, required=True)
    parser.add_argument("--counts-freeze", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    counts = args.counts_freeze or args.remoteness_manifest.with_name("counts_freeze.json")
    report = build_report(args.task, args.protocol, args.prediction_manifest, args.remoteness_manifest, counts)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "remote_homology_report.json"
    content = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if destination.exists() and destination.read_text() != content:
        raise ValueError("Refusing to overwrite an existing report with changed results")
    destination.write_text(content)
    summary = [f"# {args.task} validation remoteness report", "", f"Endpoint: {report['endpoint']}.",
               "Development-only reporting; no selection or promotion.", "",
               "| Coverage rule | Comparison | Bin/interaction | Difference | Paired CI | Interpretation |",
               "|---|---|---|---:|---|---|"]
    for coverage, comparisons in (("80% primary", report["comparisons"]),
                                  ("50% sensitivity", report["sensitivity_audit"]["comparisons"])):
        for name, comparison in comparisons.items():
            for bin_name, result in comparison.get("contrasts", {}).items():
                difference = ("unestimable" if result["difference"] is None
                              else f"{100 * result['difference']:+.3f} pp")
                bounds = result["paired_confidence_interval"]
                interval = ("unavailable" if bounds is None
                            else f"[{100 * bounds[0]:.3f}, {100 * bounds[1]:.3f}] pp")
                summary.append(f"| {coverage} | {name} | {bin_name} | {difference} | {interval} | {result['interpretation']} |")
    summary.extend(["", *report["limitations"]])
    (args.output_dir / "remote_homology_report.md").write_text("\n".join(summary) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
