from __future__ import annotations

import csv
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from manage_structure_store import inventory_structures, migrate, verify_store
from structure_store import (
    STRUCTURE_MANIFEST_FILENAME,
    StructureReference,
    read_structure_manifest,
    resolve_structure_files,
    sha256_file,
    store_structure,
    write_structure_manifest,
)
from training.structure_loading import find_structure_files


def _test_manifest_resolves_canonical_files_and_preserves_names(tmp_path: Path) -> None:
    data_root = tmp_path / "DeepMzyme_Data"
    store_root = data_root / "structure_store"
    split_dir = data_root / "example" / "train"
    source_dir = tmp_path / "sources"
    split_dir.mkdir(parents=True)
    source_dir.mkdir()
    source = source_dir / "1abc__chain_A__EC_1.1.1.1.pdb"
    source.write_bytes(b"ATOM\n")

    stored = store_structure(source, store_root)
    write_structure_manifest(split_dir, [stored], verify_hashes=True)

    references = read_structure_manifest(split_dir, verify_hashes=True)
    assert references == [stored]
    assert find_structure_files(split_dir) == [stored.path]
    manifest_text = (split_dir / STRUCTURE_MANIFEST_FILENAME).read_text(encoding="utf-8")
    assert str(tmp_path) not in manifest_text
    assert "../../structure_store/objects/" in manifest_text


def _test_manifest_rejects_tampered_content_when_hash_check_is_requested(tmp_path: Path) -> None:
    structure_dir = tmp_path / "split"
    target = tmp_path / "1abc.pdb"
    structure_dir.mkdir()
    target.write_bytes(b"original")
    reference = StructureReference(target.name, target, sha256_file(target), target.stat().st_size)
    write_structure_manifest(structure_dir, [reference])
    target.write_bytes(b"changed!")

    try:
        read_structure_manifest(structure_dir, verify_hashes=True)
    except ValueError as exc:
        assert "SHA-256 mismatch" in str(exc)
    else:
        raise AssertionError("Tampered content unexpectedly passed SHA-256 verification")


def _test_migration_keeps_same_name_different_content_as_distinct_objects(tmp_path: Path) -> None:
    data_root = tmp_path / "DeepMzyme_Data"
    left = data_root / "CARE" / "train"
    right = data_root / "CLEAN" / "train"
    duplicate = data_root / "exact" / "test"
    for directory in (left, right, duplicate):
        directory.mkdir(parents=True)
    name = "same_name.pdb"
    (left / name).write_bytes(b"CARE bytes")
    (right / name).write_bytes(b"CLEAN bytes")
    (duplicate / name).write_bytes(b"CARE bytes")

    store_root = data_root / "structure_store"
    before = inventory_structures(data_root, store_root)
    assert len(before) == 3
    report = migrate(data_root, store_root, apply=True)

    assert report["logical_structure_references"] == 3
    assert report["unique_sha256_contents"] == 2
    assert report["same_name_different_content_count"] == 1
    assert list(data_root.glob("**/*.pdb"))  # canonical objects remain
    assert not list(left.glob("*.pdb"))
    assert not list(right.glob("*.pdb"))
    assert not list(duplicate.glob("*.pdb"))
    assert len(list((store_root / "objects").glob("**/*.pdb"))) == 2
    assert len(find_structure_files(left)) == 1
    assert len(find_structure_files(right)) == 1
    verify_store(data_root, store_root)


def _test_migration_cleans_symlink_after_its_legacy_target(tmp_path: Path) -> None:
    data_root = tmp_path / "DeepMzyme_Data"
    shared = data_root / "A_shared" / "structures"
    fold = data_root / "B_fold" / "train"
    shared.mkdir(parents=True)
    fold.mkdir(parents=True)
    source = shared / "shared.pdb"
    source.write_bytes(b"shared bytes")
    (fold / source.name).symlink_to(source.resolve())

    store_root = data_root / "structure_store"
    migrate(data_root, store_root, apply=True)

    assert not source.exists()
    assert not (fold / source.name).is_symlink()
    assert find_structure_files(shared)[0] == find_structure_files(fold)[0]
    verify_store(data_root, store_root)


class StructureStoreTests(unittest.TestCase):
    def test_unlisted_or_unreadable_leftovers_block_migration(self) -> None:
        for kind in ("unlisted", "broken_symlink"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / "sample.pdb"
                source.write_bytes(b"original")
                data = root / "data"
                store = data / "structure_store"
                member = data / "dataset" / "train"
                ref = store_structure(source, store)
                write_structure_manifest(member, [ref])
                if kind == "unlisted":
                    leftover = member / "extra.pdb"
                    leftover.write_bytes(b"extra")
                    message = "absent from manifest"
                else:
                    leftover = member / ref.structure_name
                    leftover.symlink_to(root / "missing.pdb")
                    message = "unreadable legacy structure"
                manifest_before = (member / STRUCTURE_MANIFEST_FILENAME).read_bytes()
                with patch("manage_structure_store.write_json") as write_report:
                    with self.assertRaisesRegex(ValueError, message):
                        migrate(data, store, apply=True)
                    write_report.assert_not_called()
                self.assertTrue(leftover.exists() or leftover.is_symlink())
                self.assertEqual((member / STRUCTURE_MANIFEST_FILENAME).read_bytes(), manifest_before)
                self.assertEqual(ref.path.read_bytes(), b"original")

    def test_conflicting_legacy_file_aborts_before_any_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source" / "same.pdb"
            source.parent.mkdir()
            source.write_bytes(b"original")
            data = root / "data"
            store = data / "structure_store"
            member = data / "dataset" / "train"
            ref = store_structure(source, store, prefer_hardlink=False)
            write_structure_manifest(member, [ref])
            leftover = member / source.name
            leftover.write_bytes(b"conflict")
            self.assertNotEqual(sha256_file(leftover), ref.sha256)
            before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            with patch("manage_structure_store.store_structure") as write_object, \
                 patch("manage_structure_store.write_structure_manifest") as write_manifest, \
                 patch("manage_structure_store.write_json") as write_report:
                with self.assertRaisesRegex(ValueError, "Conflicting legacy structure.*same.pdb"):
                    migrate(data, store, apply=True)
                write_object.assert_not_called()
                write_manifest.assert_not_called()
                write_report.assert_not_called()
            self.assertTrue(leftover.is_file())
            self.assertEqual(leftover.read_bytes(), b"conflict")
            self.assertEqual(before, {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()})

    def test_matching_legacy_cleanup_and_repeat_migration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            store = data / "structure_store"
            member = data / "dataset" / "train"
            member.mkdir(parents=True)
            source = member / "same.pdb"
            source.write_bytes(b"original")
            ref = store_structure(source, store)
            write_structure_manifest(member, [ref])
            original_manifest = (member / STRUCTURE_MANIFEST_FILENAME).read_bytes()
            migrate(data, store, apply=True)
            self.assertFalse(source.exists())
            migrate(data, store, apply=True)
            self.assertEqual(read_structure_manifest(member), [ref])
            self.assertEqual((member / STRUCTURE_MANIFEST_FILENAME).read_bytes(), original_manifest)
            self.assertEqual(ref.path.read_bytes(), b"original")

    def test_same_size_hardlink_corruption_rejected_by_normal_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "sample.pdb"
            source.write_bytes(b"original")
            ref = store_structure(source, root / "store")
            member = root / "split"
            write_structure_manifest(member, [ref])
            self.assertEqual(find_structure_files(member), [ref.path])
            self.assertTrue(source.samefile(ref.path))
            source.write_bytes(b"changed!")
            self.assertEqual(ref.path.stat().st_size, ref.size_bytes)
            # Even an old caller explicitly passing False cannot bypass verification.
            for resolve in (read_structure_manifest, resolve_structure_files, find_structure_files):
                with self.subTest(resolver=resolve.__name__):
                    with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                        resolve(member)
            for resolve in (read_structure_manifest, resolve_structure_files):
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    resolve(member, verify_hashes=False)

    def test_legacy_resolution_and_portable_manifest_bundle_dependencies(self) -> None:
        from build_colab_bundle import append_manifest_structure_dependencies

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "original"
            legacy = original / "legacy"
            legacy.mkdir(parents=True)
            source = legacy / "sample.pdb"
            source.write_bytes(b"ATOM synthetic\n")
            self.assertEqual(resolve_structure_files(legacy), [source])
            self.assertEqual(find_structure_files(legacy), [source])
            ref = store_structure(source, original / "store")
            member = original / "split"
            write_structure_manifest(member, [ref])
            (member / "not_a_member.pdb").write_bytes(b"ignored")
            self.assertEqual(find_structure_files(member), [ref.path])
            paths = []
            append_manifest_structure_dependencies(paths, member)
            append_manifest_structure_dependencies(paths, member)
            self.assertEqual(paths, [ref.path])
            moved = root / "moved"
            shutil.copytree(original, moved)
            resolved = find_structure_files(moved / "split")
            self.assertEqual(resolved, [moved / ref.path.relative_to(original)])

    def test_invalid_manifest_paths_do_not_fall_back_to_legacy_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "sample.pdb"
            source.write_bytes(b"original")
            ref = store_structure(source, root / "store")
            member = root / "split"
            write_structure_manifest(member, [ref])
            manifest = member / STRUCTURE_MANIFEST_FILENAME
            with manifest.open(newline="") as handle:
                row = next(csv.DictReader(handle))
            for update in ({"relative_path": str(ref.path)}, {"structure_name": "../sample.pdb"},
                           {"relative_path": "missing/sample.pdb"}, {"relative_path": "other.pdb"}):
                with self.subTest(update=update):
                    with manifest.open("w", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(row))
                        writer.writeheader()
                        writer.writerow({**row, **update})
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        find_structure_files(member)

    def run_in_temporary_directory(self, test_function) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            test_function(Path(temporary_directory))

    def test_manifest_resolves_canonical_files_and_preserves_names(self) -> None:
        self.run_in_temporary_directory(_test_manifest_resolves_canonical_files_and_preserves_names)

    def test_manifest_rejects_tampered_content_when_hash_check_is_requested(self) -> None:
        self.run_in_temporary_directory(_test_manifest_rejects_tampered_content_when_hash_check_is_requested)

    def test_migration_keeps_same_name_different_content_as_distinct_objects(self) -> None:
        self.run_in_temporary_directory(_test_migration_keeps_same_name_different_content_as_distinct_objects)

    def test_migration_cleans_symlink_after_its_legacy_target(self) -> None:
        self.run_in_temporary_directory(_test_migration_cleans_symlink_after_its_legacy_target)


if __name__ == "__main__":
    unittest.main()
