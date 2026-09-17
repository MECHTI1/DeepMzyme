from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import export_validation_predictions as export


class ReplayContractTests(unittest.TestCase):
    def valid_config(self):
        return {"task": "metal", "run_test_eval": False, "test_structure_dir": None,
                "test_summary_csv": None, "train_val_split_by": "pdbid",
                "selection_metric": "val_metal_balanced_acc", "ec_label_depth": 1,
                "structure_dir": "/data/dataset/train", "summary_csv": "/data/dataset/train/sites.csv"}

    def test_rejects_test_flags_paths_and_loss_selection(self):
        export.validate_config(self.valid_config())
        for replacement in ({"run_test_eval": True}, {"test_summary_csv": "/data/test.csv"},
                            {"structure_dir": "/data/test"}, {"selection_metric": "train_loss"},
                            {"task": "joint"}, {"controlled_ec_auxiliary": True}):
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                export.validate_config({**self.valid_config(), **replacement})

    def test_remapping_obeys_path_boundaries(self):
        mapping = {"/content/data": "/local/data", "/content/data/special": "/local/overlay"}
        self.assertEqual(export.remap_path("/content/data/special/chain.json", mapping), Path("/local/overlay/chain.json"))
        self.assertEqual(export.remap_path("/content/database/file", mapping), Path("/content/database/file"))

    def summary(self):
        def split(group):
            examples = [{"structure_id": group, "pocket_id": group + "_METAL_0", "group": group,
                         "y_metal": 0, "y_ec": 1}]
            return {"examples": examples, "n_examples": 1, "ordered_examples_sha256": export.fingerprint(examples)}
        return {"retained_split_identity": {"train": split("train-protein"), "validation": split("val-protein")}}

    def test_membership_integrity_and_group_leakage(self):
        export.verify_membership(self.summary())
        corrupt = self.summary()
        corrupt["retained_split_identity"]["validation"]["examples"][0]["y_metal"] = 3
        with self.assertRaisesRegex(ValueError, "digest"):
            export.verify_membership(corrupt)
        overlap = self.summary()
        overlap["retained_split_identity"]["validation"] = copy.deepcopy(overlap["retained_split_identity"]["train"])
        with self.assertRaisesRegex(ValueError, "overlap"):
            export.verify_membership(overlap)

    def test_aggregate_metric_agreement_does_not_hide_changed_errors(self):
        selected = {"val_metal_balanced_acc": 0.5, "val_metal_confusion_matrix": [[1, 1], [1, 1]],
                    "val_metal_per_class_recall": {"A": 0.5, "B": 0.5}}
        export.compare_metrics(selected, selected, "val_metal_balanced_acc")
        changed = {**selected, "val_metal_confusion_matrix": [[2, 0], [2, 0]]}
        with self.assertRaisesRegex(ValueError, "confusion_matrix"):
            export.compare_metrics(changed, selected, "val_metal_balanced_acc")

    def test_frozen_factory_has_no_current_default_fallback(self):
        source = """def prepare_run(config):
    model = build_pocket_classifier(model_architecture=config.model_architecture,
        hidden_s=config.hidden_s, n_ec=max(1, len(load_result.ec_index_to_label)),
        n_metal=len(METAL_TARGET_LABELS), metal_class_weights=metal_class_weights,
        use_node_type_embedding=config.metal_node_mode != 'none',
        predict_ec=task_predicts_ec(config.task))
    optimizer = forbidden_training_operation()
"""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.py"
            path.write_text(source)
            config = {"model_architecture": "only_gvp", "hidden_s": 64, "metal_node_mode": "none", "task": "ec"}
            kwargs = export.factory_kwargs(path, config, {"model_state_dict": {"metal_class_weights": "saved"}}, {0: "Mn"}, {0: "1", 1: "2"})
            self.assertEqual(kwargs["n_ec"], 2)
            self.assertTrue(kwargs["predict_ec"])
            self.assertEqual(kwargs["metal_class_weights"], "saved")
            del config["hidden_s"]
            with self.assertRaisesRegex(ValueError, "absent"):
                export.factory_kwargs(path, config, {"model_state_dict": {}}, {0: "Mn"}, {})
            path.write_text(source.replace("config.hidden_s", "dangerous()"))
            with self.assertRaisesRegex(ValueError, "Unsupported frozen model expression"):
                export.factory_kwargs(path, config, {"model_state_dict": {}}, {0: "Mn"}, {})

    def test_frozen_source_drift_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "src/training/run.py"
            path.parent.mkdir(parents=True)
            path.write_text("original")
            files = {"src/training/run.py": export.digest(path)}
            export.verify_source(root, files)
            path.write_text("changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                export.verify_source(root, files)

    def test_sidecars_are_immutable(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "receipt.json"
            export.write_json(path, {"verified": True})
            with self.assertRaises(FileExistsError):
                export.write_json(path, {"verified": False})
            self.assertEqual(json.loads(path.read_text()), {"verified": True})


class LogitExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch

    def test_ec_uses_mean_logits_and_preserves_all_pockets(self):
        torch = self.torch
        logits = torch.tensor([[10., -2.], [-1., 2.], [-1., 2.], [0., 1.]])
        y = torch.tensor([0, 0, 0, 1])
        groups = torch.tensor([0, 0, 0, 1])
        values, truths, examples = export.aggregate_ec_logits(logits, y, groups, {0: "protein-a", 1: "protein-b"})
        self.assertEqual(values.argmax(-1).tolist(), [0, 1])
        self.assertEqual(int(logits[:3].softmax(-1).mean(0).argmax()), 1)
        self.assertEqual(truths.tolist(), [0, 1])
        self.assertEqual(examples[0]["n_pockets"], 3)
        rows = export.prediction_rows(values, truths, examples, {0: "1", 1: "2"})
        self.assertEqual(rows[0]["group_id"], "protein-a")
        self.assertAlmostEqual(rows[0]["logit_0"], 8 / 3, places=6)

    def test_ec_conflicts_are_not_silently_discarded(self):
        torch = self.torch
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            export.aggregate_ec_logits(torch.zeros(2, 2), torch.tensor([0, 1]), torch.tensor([0, 0]), {0: "protein-a"})

    def test_nonfinite_logits_rejected(self):
        torch = self.torch
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            export.prediction_rows(torch.tensor([[float("nan"), 0.]]), torch.tensor([0]), [{"example_id": "a"}], {0: "1", 1: "2"})


class CompletedReplayTests(unittest.TestCase):
    def fixture(self, root):
        source = root / "source"
        source_file = source / "src/training/run.py"
        source_file.parent.mkdir(parents=True)
        source_file.write_text("# frozen source")
        source_files = {"src/training/run.py": export.digest(source_file)}
        run_dir = root / "run"
        run_dir.mkdir()
        (run_dir / "run_config.json").write_text("{}")
        (run_dir / "best_model_checkpoint.pt").write_bytes(b"selected checkpoint fixture")
        output = root / "predictions"
        child = output / "selected"
        child.mkdir(parents=True)
        entry = {"run_id": "selected", "task": "metal", "family": "only_gvp", "seed": 42,
                 "fold_id": "fixed", "source_root": str(source), "source_files": source_files,
                 "run_dir": str(run_dir), "expected_sha256": {name: export.digest(run_dir / name)
                 for name in ("run_config.json", "best_model_checkpoint.pt")}}
        ledger_path = root / "ledger.json"
        export.write_json(ledger_path, {"schema_version": 1, "runs": [entry]})
        artifact = export.write_csv(child / "pocket_predictions.csv", [{"example_id": "x", "target": 0, "prediction": 0}])
        receipt = {**{key: entry[key] for key in ("run_id", "task", "family", "seed", "fold_id")},
                   "validation_only": True, "reproduction_passed": True,
                   "reuse_ledger_sha256": export.digest(ledger_path),
                   "run_config_sha256": entry["expected_sha256"]["run_config.json"],
                   "checkpoint_sha256": entry["expected_sha256"]["best_model_checkpoint.pt"],
                   "source": export.verify_source(source, source_files),
                   "prediction_artifacts": {"pocket_predictions": artifact}}
        export.write_json(child / "receipt.json", receipt)
        result = {**{key: entry[key] for key in ("run_id", "task", "family", "seed", "fold_id")},
                  "validation_only": True, "reproduction_passed": True, "pocket_predictions": artifact,
                  "receipt": {"path": str(child / "receipt.json"), "sha256": export.digest(child / "receipt.json")}}
        export.write_json(child / "manifest_entry.json", result)
        return entry, ledger_path, output, child

    def test_resume_never_repeats_completed_inference_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, ledger, output, child = self.fixture(Path(temporary))
            original = {path.name: export.digest(path) for path in child.iterdir()}
            arguments = ["--reuse-ledger", str(ledger), "--validation-only", "--output-dir", str(output), "--resume"]
            with patch.object(export.subprocess, "run", side_effect=AssertionError("Inference repeated")):
                export.main(arguments)
                export.main(arguments)
            self.assertEqual(len(list((output / "manifest_generations").glob("*.json"))), 1)
            self.assertEqual(original, {path.name: export.digest(path) for path in child.iterdir()})
            self.assertEqual(len(export.read_json(output / "manifest.json")["runs"]), 1)

    def test_resume_rejects_changed_checkpoint_and_incomplete_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            entry, ledger, _, child = self.fixture(Path(temporary))
            checkpoint = Path(entry["run_dir"]) / "best_model_checkpoint.pt"
            checkpoint.write_bytes(b"replacement")
            with self.assertRaisesRegex(ValueError, "Artifact hash mismatch"):
                export.verified_completed_export(entry, ledger, child)
            (child / "receipt.json").unlink()
            with self.assertRaisesRegex(ValueError, "operator archival"):
                export.verified_completed_export(entry, ledger, child)

    def test_explicit_metadata_hash_is_not_ignored(self):
        with tempfile.TemporaryDirectory() as temporary:
            entry, ledger, _, _ = self.fixture(Path(temporary))
            metadata = Path(entry["run_dir"]) / "run_metadata.json"
            metadata.write_text('{"selected_epoch": 7}')
            entry["expected_sha256"]["run_metadata.json"] = export.digest(metadata)
            export.verify_bound_run_files(entry, ledger.parent)
            metadata.write_text('{"selected_epoch": 8}')
            with self.assertRaisesRegex(ValueError, "run_metadata.json"):
                export.verify_bound_run_files(entry, ledger.parent)

    def test_worker_source_snapshot_survives_repository_edits_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "exporter.py"
            source.write_text('VERSION = "first"\n')
            original_hash = export.digest(source)
            snapshot = export.snapshot_worker_source(root, source)
            source.write_text('VERSION = "second"\n')
            self.assertEqual(export.digest(snapshot), original_hash)
            self.assertNotEqual(export.snapshot_worker_source(root, source), snapshot)
            source.write_text('VERSION = "first"\n')
            snapshot.write_text('VERSION = "tampered"\n')
            with self.assertRaisesRegex(ValueError, "snapshot changed"):
                export.snapshot_worker_source(root, source)


if __name__ == "__main__":
    unittest.main()
