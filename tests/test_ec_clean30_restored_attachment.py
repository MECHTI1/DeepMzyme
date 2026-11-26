"""A resumed later-fold attachment verifies the existing corpus without rebuilding it."""
import json
import shutil
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from ec_clean30 import futility, identity, phase3, phase4, pretrain_corpus
from ec_clean30_fold0_fixtures import IDENTITY, SOURCE, sha, write
from test_ec_clean30_pretrain_corpus import _real_inputs, pinned_binary, runtime_precision, world  # noqa: F401


@pytest.fixture(scope="module")
def prepared_world(world):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(identity, "MMSEQS_BINARY_SHA256", world.binary_sha)
        for inner in range(identity.N_INNER_FOLDS):
            identity.build_r1_inner_exclusions(world.work(1), 1, inner)
    return world


@pytest.mark.parametrize("damage", [None, "missing", "empty", "payload", "attestation", "legacy_missing"])
def test_restored_fold_attachment_verifies_existing_bytes(prepared_world, tmp_path, monkeypatch, damage):
    world = prepared_world
    options = _real_inputs(world, tmp_path, fold=1)
    folder = tmp_path / "view" / pretrain_corpus.WHOLE_PROTEIN_PRETRAIN_DIR
    pretrain_corpus.build_corpus_folder(folder, **options)
    membership = world.work(1) / "inner_membership_fold1.csv"
    view = SimpleNamespace(root=folder.parent, smoke=False, fold=1, manifest_sha256=options["view_record"]["manifest_sha256"],
                           membership={"sha256": sha(membership), "n_folds": 4}, membership_file=lambda: membership)
    root = tmp_path / "execution"
    unit = "ecc30_pretrain_fold1_attempt1"
    source = {**SOURCE, "amendment_path": "tip.json", "amendment_chain_receipt": "chain.json"}
    bound_corpus = tmp_path / "bound.tsv"
    shutil.copyfile(folder / pretrain_corpus.CORPUS_NAME, bound_corpus)
    holdout = write(tmp_path / pretrain_corpus.HOLDOUT_RECEIPT_NAME, {"fold": 1})
    request = {"execution_root": str(root), "unit_id": unit, "fold": 1, "execution_manifest": "manifest.json",
               "view": str(view.root), "fold0_step_chain": "steps.jsonl", "campaign_root": str(tmp_path),
               "corpus_table": str(bound_corpus), "holdout_receipt": str(holdout), "identity_work": str(tmp_path / "work"),
               "seq_cache": str(options["seq_cache_dir"]), "seq_manifest_sha256": options["seq_manifest_sha256"],
               "official_split_dir": str(options["official_split_dir"])}
    monkeypatch.setattr(phase4, "_runtime", lambda *a, **kw: (source, view, None, None, None))
    monkeypatch.setattr(phase4, "_execution", lambda *a: nullcontext())
    monkeypatch.setattr(phase3, "verify_execution_source", lambda *a, **kw: source)
    monkeypatch.setattr(phase3, "authenticated_chain", lambda *a: {})
    monkeypatch.setattr(phase3, "require_record_precedence", phase3.require_same_scientific_identity)
    monkeypatch.setattr(futility, "verify_futility_decision", lambda *a, **kw: {**IDENTITY, "decision": "continue"})
    monkeypatch.setattr(phase3.views.View, "load", lambda *a: view)
    monkeypatch.setattr(phase3, "_bind_view", lambda *a: None)
    monkeypatch.setattr(phase3, "_bound_pretrain_file", lambda source, role, path: {"path": path.name, "sha256": sha(path)})
    record = {**phase3.attach_pretrain(request), "unit_id": unit}
    marker = write(root / "runs" / unit / "complete.json", record)
    marker_digest = sha(marker)
    if damage != "legacy_missing":
        request["restored_complete_sha256"] = marker_digest
    if damage in {"missing", "empty", "legacy_missing"}:
        shutil.rmtree(folder)
        if damage == "empty":
            folder.mkdir()
    elif damage == "payload":
        next((folder / pretrain_corpus.SEQ_MEANS_DIR).rglob("*.pt")).write_bytes(b"changed")
    elif damage == "attestation":
        # Internally valid evidence still has to match the predecessor's exact receipt.
        original = folder / pretrain_corpus.CORPUS_RECEIPT_NAME
        write(original, {**json.loads(original.read_text()), "annotation": "later edit"})

    if damage == "legacy_missing":
        def rebuild(root, **kwargs):
            assert root == view.root and kwargs["seq_manifest_sha256"] == options["seq_manifest_sha256"]
            pretrain_corpus.build_corpus_folder(folder, **options)

        monkeypatch.setattr(pretrain_corpus, "build_view_pretrain_corpus", rebuild)
        assert phase4.attach_fold(request) == record
        assert pretrain_corpus.verify_whole_protein_pretrain(view, required=True) == record["verification"]
        return

    def files():
        return {p.relative_to(folder): (sha(p), p.stat().st_mtime_ns) for p in folder.rglob("*") if p.is_file()}

    before = files()
    monkeypatch.setattr(pretrain_corpus, "build_view_pretrain_corpus",
                        lambda *a, **kw: pytest.fail("A restored attachment tried to rebuild its corpus"))
    if damage is None:
        assert phase4.attach_fold(request) == record
    else:
        reason = {"missing": "required but not built", "empty": "required but empty", "payload": "does not load",
                  "attestation": "differs from its completed evidence"}[damage]
        with pytest.raises(ValueError, match=reason):
            phase4.attach_fold(request)
    assert folder.exists() == (damage != "missing")
    assert files() == before
    assert sha(marker) == marker_digest
