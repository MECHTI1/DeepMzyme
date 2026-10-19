"""The release inventory cannot bless corrupt receipts or substitute a smaller model."""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from metal_release.catalog import REFERENCE, release_catalog
from metal_release.inventory import BASELINE_PROBE, REFIT_DIRECTORY, bind_catalog
from metal_release.manifest import ReleaseError, load_manifest, sha256_file, verify_artifact


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="metal-release-inventory-")
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.runs = self.root / "runs"
        self.runs.mkdir()
        self.folds = self.root / "fold_membership.csv"
        self.folds.write_text("source_uid,physical_ion_id,group_id,native_element,fold,pdbid\nu,p,g,MN,0,1abc\n")
        self.output = self.root / "release.json"
        self.member = "gvp_late_fusion__four_class__trunklr1e4__fold0__seed42"

    def run_fixture(self, member_id=None, directory=None):
        member_id = member_id or self.member
        member = release_catalog()["members"][member_id]
        run = self.runs / (directory or member_id)
        run.mkdir(parents=True)
        checkpoint = run / "terminal_model_checkpoint.pt"
        checkpoint.write_bytes(b"opaque checkpoint bytes; inventory never unpickles these")
        identity = {"fold": member["fold"], "model_seed": member["seed"],
                    "target_scheme": member["target"], "fold_membership_sha256": sha256_file(self.folds)}
        config = {"config": {"campaign_run_identity": json.dumps(identity)}, "checkpoint_rule": "terminal"}
        if member["fold"] is None:
            identity.update({"validation": "none", "phase": "stage6b_full_non_test_refit"})
            config["config"]["campaign_run_identity"] = json.dumps(identity)
            receipt = {"passed": True, "checkpoint_sha256": sha256_file(checkpoint)}
            receipt_name = "refit_reload_check.json"
        else:
            prediction = run / "val_predictions.csv"
            prediction.write_text("source_uid,checkpoint_sha256\nu," + sha256_file(checkpoint) + "\n")
            receipt = {"campaign_run_identity": identity, "checkpoint_rule": "terminal", "fit_status": "completed",
                       "selected_checkpoint": checkpoint.name, "selected_checkpoint_sha256": sha256_file(checkpoint),
                       "validation_predictions": {"path": prediction.name, "sha256": sha256_file(prediction)}}
            receipt_name = "selected_checkpoint.json"
        (run / receipt_name).write_text(json.dumps(receipt))
        (run / "run_config.json").write_text(json.dumps(config))
        return run

    def bind(self, roots=None):
        return bind_catalog(roots or [self.runs], self.folds, self.output)

    def test_missing_members_are_retained_and_reference_never_changes(self):
        self.run_fixture()
        result = self.bind()
        self.assertEqual(result["reference"], REFERENCE)
        self.assertEqual(result["release_status"], "blocked")
        self.assertEqual(set(result["members"]), set(release_catalog()["members"]))
        loaded = load_manifest(self.output)
        self.assertTrue(verify_artifact(loaded["members"][self.member]["checkpoint"]).is_file())
        self.assertTrue(result["members"][self.member]["receipt_binding_verified"])

    def test_altered_checkpoint_is_not_certified_with_its_new_hash(self):
        run = self.run_fixture()
        (run / "terminal_model_checkpoint.pt").write_bytes(b"altered")
        result = self.bind()
        self.assertIsNone(result["members"][self.member]["checkpoint"])
        self.assertTrue(any("Recorded checkpoint SHA-256 mismatch" in b["reason"] for b in result["inventory_blockers"]))

    def test_changed_predictions_are_rejected(self):
        run = self.run_fixture()
        (run / "val_predictions.csv").write_text("changed\n")
        result = self.bind()
        self.assertTrue(any("prediction SHA-256 mismatch" in b["reason"] for b in result["inventory_blockers"]))

    def test_conflicting_backup_is_not_silently_ignored(self):
        run = self.run_fixture()
        second = self.root / "backup" / run.name
        shutil.copytree(run, second)
        (second / "run_config.json").write_text((second / "run_config.json").read_text() + "\n")
        result = self.bind([self.runs, second.parent])
        self.assertIsNone(result["members"][self.member]["checkpoint"])
        self.assertTrue(any("Conflicting copies of run_config" in b["reason"] for b in result["inventory_blockers"]))

    def test_identical_backups_are_allowed(self):
        run = self.run_fixture()
        second = self.root / "backup" / run.name
        shutil.copytree(run, second)
        result = self.bind([self.runs, second.parent])
        self.assertIsNotNone(result["members"][self.member]["checkpoint"])

    def test_wrong_fold_hash_and_wrong_identity_fail(self):
        run = self.run_fixture()
        receipt_path = run / "selected_checkpoint.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["campaign_run_identity"]["fold"] = 1
        receipt_path.write_text(json.dumps(receipt))
        result = self.bind()
        self.assertTrue(any("Campaign fold differs" in b["reason"] for b in result["inventory_blockers"]))

    def test_wrong_frozen_membership_is_rejected(self):
        self.run_fixture()
        self.folds.write_text("new membership\n")
        result = self.bind()
        self.assertTrue(any("Frozen fold membership hash differs" in b["reason"] for b in result["inventory_blockers"]))

    def test_only_exact_names_and_allowed_directories_are_inspected(self):
        run = self.run_fixture(directory="near_" + self.member)
        (run / "run_config.json").write_text("this should never be opened")
        hidden = self.runs / "heldout" / self.member
        shutil.copytree(run, hidden)
        result = self.bind()
        self.assertEqual([b["reason"] for b in result["inventory_blockers"] if b["member"] == self.member],
                         ["Missing exact run directory"])
        with self.assertRaises(ReleaseError):
            bind_catalog([hidden.parent], self.folds, self.root / "other.json")

    def test_canonical_probe_and_completion_aliases(self):
        probe = "gvp_late_fusion__four_class__baseline__fold0__seed42"
        self.run_fixture(probe, BASELINE_PROBE)
        mc = "mc_P_five_class__fold0__seed43"
        self.run_fixture(mc, "metal_completion__confirmation__P__five_class__fold0__seed43")
        result = self.bind()
        self.assertIsNotNone(result["members"][probe]["checkpoint"])
        self.assertIsNotNone(result["members"][mc]["checkpoint"])

    def test_refit_uses_its_existing_reload_receipt_without_oof(self):
        self.run_fixture("stage6b_refit", REFIT_DIRECTORY)
        result = self.bind()
        refit = result["members"]["stage6b_refit"]
        self.assertEqual(refit["receipt_kind"], "refit_reload_check")
        self.assertIsNone(refit["validation_predictions"])
        self.assertIsNotNone(refit["checkpoint"])

    def test_outputs_are_exclusive(self):
        self.bind()
        original = self.output.read_bytes()
        with self.assertRaisesRegex(ReleaseError, "already exist"):
            self.bind()
        self.assertEqual(self.output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
