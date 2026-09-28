"""CPU-only, append-only report of the five-class fold-0 screen and matched controls."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
CAMPAIGN = Path("/media/mechti/Data1/DeepMzyme_Data/campaigns/pmm_ion_metal_v2_context")
CODE = CAMPAIGN.parent / "_code/pmm_core_scope_v2"
TRAIN = Path("/media/mechti/Data1/DeepMzyme_PMM_Zenodo_Exact_Dataset/dataset/train")
FOUR = ("Mn", "Cu", "Zn", "Class VIII")
FIVE = ("Mn", "Cu", "Zn", "Fe", "Co+Ni")
SIX = ("Mn", "Cu", "Zn", "Fe", "Co", "Ni")
SUPPLEMENT = CAMPAIGN / "runtime/screen_agreement_v2_1_checked_20260928/screen_agreement_v2_1.json"
SUPPLEMENT_SHA = "8887fc8a144ca203354254cac96ca268c77699a592bdcd9fc441ad35fe294b47"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def classification(rows, view, labels):
    matrix = [[0] * len(labels) for _ in labels]
    for row in rows:
        truth, pred = int(row[f"y_{view}"]), int(row[f"pred_{view}"])
        require(0 <= truth < len(labels) and 0 <= pred < len(labels), "Invalid class index")
        matrix[truth][pred] += 1
    support = [sum(row) for row in matrix]
    require(all(support), "A required validation class is absent; do not average available classes")
    recall = [matrix[i][i] / support[i] for i in range(len(labels))]
    f1 = [2 * matrix[i][i] / (support[i] + sum(row[i] for row in matrix)) for i in range(len(labels))]
    return {"balanced_accuracy": sum(recall) / len(labels), "macro_f1": sum(f1) / len(labels),
            "accuracy": sum(matrix[i][i] for i in range(len(labels))) / sum(support),
            "recall": dict(zip(labels, recall)), "support": dict(zip(labels, support)),
            "confusion_matrix": matrix}


def unit_fingerprint(rows):
    fields = ("source_uid", "physical_ion_id", "group_id", "native_element", "y_common4", "fold", "model_seed")
    require(len({row["source_uid"] for row in rows}) == len(rows), "Duplicate validation UID")
    data = [tuple(row[key] for key in fields) for row in sorted(rows, key=lambda row: row["source_uid"])]
    return hashlib.sha256(json.dumps(data, separators=(",", ":")).encode()).hexdigest()


class Evidence:
    def __init__(self):
        self.files = {}

    def bind(self, path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        require(expected is None or digest == expected, f"Evidence changed: {path}")
        require(str(path) not in self.files or self.files[str(path)] == digest, f"Concurrent change: {path}")
        self.files[str(path)] = digest
        return path

    def json(self, path):
        return json.loads(self.bind(path).read_text())

    def recheck(self):
        for path, digest in self.files.items():
            require(sha(path) == digest, f"Evidence changed during reporting: {path}")


def collect(screen, campaign, paths, membership, family, target, protocol_sha, evidence):
    from benchmarking.pmm_comparator import validate_prediction_rows

    name = f"{family}__{target}__none__fold0__seed42"
    run, command_path = paths.runs / name, paths.commands / f"{name}.json"
    item = {"run_name": name, "family": family, "target": target, "status": "missing",
            "strict_replay_certified": False}
    try:
        status_path = paths.root / f"run_status_five_screen_{name}.json"
        if target == "five_class" and status_path.exists():
            terminal = evidence.json(status_path)["units"]
            require(len(terminal) == 1 and terminal[0]["run_name"] == name, "Wrong terminal unit")
            item["terminal_status"] = terminal[0]["status"]
            if terminal[0]["status"] != "completed":
                item["status"] = terminal[0]["status"]
                item["terminal_detail"] = terminal[0]
                return item
        if not run.exists():
            if command_path.exists() or (paths.runs / f"{name}.log").exists():
                item["status"] = "incomplete_or_running"
            return item
        item["status"] = "incomplete_or_running"
        if not all((run / filename).is_file() for filename in
                   ("selected_checkpoint.json", "run_metadata.json", "run_config.json", "best_model_checkpoint.pt")):
            return item
        if target == "five_class":
            args = SimpleNamespace(python_bin=sys.executable, train_dir=TRAIN, action="plan", load_workers=None)
            identity = screen.build_command(paths, args, family, protocol_sha)[2]
        else:
            identity = campaign.build_train_command(paths, python_bin=sys.executable, train_dir=TRAIN,
                config=campaign.GridConfig(family, target, "none"), fold=0, seed=42,
                device="cpu", runs_dir=paths.runs)[2]
        require(evidence.json(command_path)["identity"] == identity, "Command identity differs from frozen recipe")
        for filename in ("selected_checkpoint.json", "run_metadata.json", "run_config.json", "best_model_checkpoint.pt"):
            evidence.bind(run / filename)
        receipt = campaign.completed_run_receipt(run, identity)
        if receipt is not None:
            if target == "five_class":
                screen.validate_completed_unit(paths, family, identity)
            item.update(status="strict_replay_certified", strict_replay_certified=True)
            replay = evidence.json(run / "independent_validation_replay/replay_receipt.json")
            evidence.bind(run / "independent_validation_replay" / replay["validation_predictions"]["path"],
                          replay["validation_predictions"]["sha256"])
        elif family == "only_gvp" and target == "six_class":
            receipt = campaign.completed_run_receipt(run, identity, require_independent=False)
            require(receipt is not None, "Historical GVP6 fit no longer verifies")
            historical = json.loads(evidence.bind(SUPPLEMENT, SUPPLEMENT_SHA).read_text())
            arm = historical["arms"]["only_gvp__six_class__none"]
            require(historical["status"] == "agreement_qualified" and arm["qualified"] is True
                    and arm["legacy_replay_pass"] is False and arm["identity"] == identity
                    and arm["checkpoint_sha256"] == receipt["selected_checkpoint_sha256"],
                    "Historical supplemental qualification does not bind this GVP6 fit")
            for path, digest in arm["input_files"].items():
                evidence.bind(path, digest)
            item.update(status="historical_supplemental_v2_1_only",
                        qualification="Retrospective v2.1 agreement; NOT a legacy strict replay pass")
        else:
            item["status"] = "fit_incomplete_or_not_strictly_certified"
            return item
        prediction = evidence.bind(run / receipt["validation_predictions"]["path"],
                                   receipt["validation_predictions"]["sha256"])
        with prediction.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        labels = {"four_class": FOUR, "five_class": FIVE, "six_class": SIX}[target]
        if target != "five_class":
            validate_prediction_rows(rows, membership, 0, native_labels=labels, seed=42,
                                     checkpoint_sha256=receipt["selected_checkpoint_sha256"])
        native, common = classification(rows, "native", labels), classification(rows, "common4", FOUR)
        for view, values in (("val_metal", native), ("val_metal_collapsed4", common)):
            for metric, key in (("balanced_accuracy", "balanced_acc"), ("macro_f1", "macro_f1")):
                require(abs(values[metric] - receipt["metrics"][f"{view}_{key}"]) <= 1e-9,
                        "CSV metrics differ from selected-checkpoint receipt")
        item.update(selected_epoch=receipt["selected_epoch"], n_ions=len(rows),
                    n_pdb_groups=len({row["group_id"] for row in rows}),
                    native=native, common4=common, uid_identity_sha256=unit_fingerprint(rows),
                    checkpoint_sha256=receipt["selected_checkpoint_sha256"], identity=identity)
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        item.update(status="verification_failed", strict_replay_certified=False,
                    error=f"{type(exc).__name__}: {exc}")
        for key in ("native", "common4", "uid_identity_sha256"):
            item.pop(key, None)
    return item


def comparisons(units):
    result = []
    for five in units:
        if five["target"] != "five_class" or not five["strict_replay_certified"]:
            continue
        for other in units:
            if other["family"] != five["family"] or other["target"] == "five_class" or "common4" not in other:
                continue
            require(five["uid_identity_sha256"] == other["uid_identity_sha256"], "Control validation units differ")
            result.append({"family": five["family"], "comparison": "five_class minus " + other["target"],
                "control_status": other["status"], "n_paired_ions": five["n_ions"],
                "common4_ba_delta_pp": 100 * (five["common4"]["balanced_accuracy"] - other["common4"]["balanced_accuracy"]),
                "common4_macro_f1_delta_pp": 100 * (five["common4"]["macro_f1"] - other["common4"]["macro_f1"]),
                "common4_recall_delta_pp": {label: 100 * (five["common4"]["recall"][label] - other["common4"]["recall"][label]) for label in FOUR}})
    return result


def self_test():
    rows = [{"y_native": str(i), "pred_native": str(i)} for i in range(5)]
    assert classification(rows, "native", FIVE)["macro_f1"] == 1
    rows[4]["pred_native"] = "3"
    result = classification(rows, "native", FIVE)
    assert result["balanced_accuracy"] == 0.8 and abs(result["macro_f1"] - 11 / 15) < 1e-12
    assert result["recall"]["Co+Ni"] == 0 and result["recall"]["Fe"] == 1
    try:
        classification(rows[:4], "native", FIVE)
    except ValueError:
        pass
    else:
        raise AssertionError("Missing class accepted")
    assert comparisons([{"target": "five_class", "strict_replay_certified": False}]) == []
    metrics = {"balanced_accuracy": 0.5, "macro_f1": 0.4, "recall": dict.fromkeys(FOUR, 0.5)}
    five = {"target": "five_class", "strict_replay_certified": True, "family": "only_esm",
            "uid_identity_sha256": "same", "n_ions": 4, "common4": metrics}
    control = {"target": "four_class", "family": "only_esm", "uid_identity_sha256": "same",
               "status": "strict_replay_certified", "common4": metrics}
    assert comparisons([five, control])[0]["common4_ba_delta_pp"] == 0
    control["uid_identity_sha256"] = "different"
    try:
        comparisons([five, control])
    except ValueError:
        pass
    else:
        raise AssertionError("Mismatched comparison UIDs accepted")
    print("Reporting self-test passed: exact metrics, Co+Ni mapping, missing-class refusal, uncertified exclusion, paired-UID gate")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    require(args.out_dir is not None, "--out-dir must name a new report directory")
    output = args.out_dir.resolve()
    require(output.parent == Path(__file__).resolve().parent and not output.exists(),
            "Use a new direct child directory beside this helper; existing results cannot be overwritten")
    sys.path[:0] = [str(CODE), str(CODE / "src")]
    spec = importlib.util.spec_from_file_location("five_screen", CODE / "run_pmm_five_class_screen.py")
    screen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(screen)
    from benchmarking import pmm_ion_campaign as campaign
    from training.access_guard import install_forbidden_read_guard
    from training.campaign_runtime import read_fold_membership

    install_forbidden_read_guard(campaign.forbidden_read_roots(TRAIN))
    protocol, protocol_sha = screen.load_protocol(screen.PROTOCOL_PATH)
    require(screen.source_tree_sha256() == protocol["source_tree_sha256"], "Frozen training source changed")
    paths, evidence = campaign.CampaignPaths(CAMPAIGN), Evidence()
    require(campaign.load_campaign(paths)["campaign_id"] == protocol["campaign_id"], "Wrong campaign")
    for path in (__file__, CODE / "run_pmm_five_class_screen.py", screen.PROTOCOL_PATH, paths.manifest,
                 paths.cohort, paths.fold_membership, paths.fold_class_weights, paths.feature_inventory):
        evidence.bind(path)
    membership = read_fold_membership(paths.fold_membership, sha(paths.fold_membership))
    units = [collect(screen, campaign, paths, membership, family, target, protocol_sha, evidence)
             for family in protocol["families"] for target in ("four_class", "five_class", "six_class")]
    paired = comparisons(units)
    certified = sum(u["target"] == "five_class" and u["strict_replay_certified"] for u in units)
    report = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "five_class_strictly_certified": certified, "five_class_required": 3,
              "status": "screen_complete" if certified == 3 else "screen_incomplete",
              "source_tree_sha256": protocol["source_tree_sha256"], "held_out_access": False,
              "promotion": False, "confidence_intervals": None,
              "limitations": "Exploratory single fold, seed42. Native metrics use different vocabularies and cannot rank formulations. Only common-four deltas are paired. GVP6 control retains supplemental v2.1 status; no new strict receipt. TECH-023/025 full-grid gates remain open. Backup/provider shutdown verification is separate.",
              "units": units, "comparisons": paired, "input_sha256": evidence.files}
    lines = ["# Five-class fold-0 screen", "", f"Five-class strict replay passes: {certified}/3.", "", report["limitations"], "",
             "Native five-class Co+Ni is exported internally as Class VIII; common-four Class VIII means Fe+Co+Ni. Values below are percentages.", "",
             "| Family | Target | Status | Epoch | Common4 BA | Common4 F1 | Native BA | Native F1 | Fe recall | Co+Ni recall |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    flat = []
    pct = lambda value: "—" if value is None else f"{100 * value:.4f}"
    for unit in units:
        row = {key: unit.get(key) for key in ("family", "target", "status", "selected_epoch", "n_ions", "n_pdb_groups")}
        for view in ("native", "common4"):
            metrics = unit.get(view, {})
            row.update({f"{view}_{metric}": metrics.get(metric) for metric in ("balanced_accuracy", "macro_f1")})
            row.update({f"{view}_recall_{label}": metrics.get("recall", {}).get(label) for label in (*SIX, "Class VIII", "Co+Ni")})
        flat.append(row)
        lines.append(f"| {unit['family']} | {unit['target']} | {unit['status']} | {unit.get('selected_epoch', '—')} | " + " | ".join(pct(row[key]) for key in
            ("common4_balanced_accuracy", "common4_macro_f1", "native_balanced_accuracy", "native_macro_f1", "native_recall_Fe", "native_recall_Co+Ni")) + " |")
        if unit.get("error"):
            lines.append(f"\nVerification refusal for {unit['run_name']}: {unit['error']}\n")
    lines += ["", "Common-four paired deltas (percentage points; no formal significance claim):", ""]
    lines += [f"- {p['family']}, {p['comparison']}: BA {p['common4_ba_delta_pp']:+.4f}; macro-F1 {p['common4_macro_f1_delta_pp']:+.4f}; control {p['control_status']}." for p in paired]
    lines += ["", "CSV values are fractions; JSON includes every class recall/support, confusion matrices, identity and input hashes."]
    evidence.recheck()
    output.mkdir(exist_ok=False)
    (output / "five_class_screen.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (output / "five_class_screen.md").write_text("\n".join(lines) + "\n")
    with (output / "five_class_screen.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps({"status": report["status"], "certified": certified, "out_dir": str(output)}))
    return 0 if certified == 3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
