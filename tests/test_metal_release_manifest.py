from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from metal_release.catalog import (ADDED_IN_V2, ADDED_IN_V3, CATALOG_V1, CATALOG_V2, CATALOG_V3,
                                   CATALOG_VERSIONS, PUBLIC_UNIFIED, REFERENCE, release_catalog)
from metal_release.manifest import (ReleaseError, aggregate_probabilities, candidate_member_ids,
                                    load_manifest, native_to_common, sha256_file, validate_manifest,
                                    verify_artifact)


def test_catalog_preserves_shortlist_and_55_member_identity():
    manifest = release_catalog()
    validate_manifest(manifest)
    assert [len(candidate_member_ids(manifest, key)) for key in
            ("step_f_single_refit_s42", "trunklr1e4_5fold_s42", "top4_lf_seed42", "top4_lf", REFERENCE)] == [1, 5, 20, 30, 55]
    assert manifest["candidates"]["step_f_single_refit_s42"]["oof_proxy"] == "baseline_5fold_s42"
    assert len(manifest["members"]) == 61


@pytest.mark.parametrize("catalog_id", sorted(CATALOG_VERSIONS))
def test_each_catalog_version_validates_with_its_own_shortlist(catalog_id):
    manifest = release_catalog(catalog_id)
    validate_manifest(manifest)
    expected = set(release_catalog(CATALOG_V3)["candidates"])
    if catalog_id in {CATALOG_V1, CATALOG_V2}:
        expected -= set(ADDED_IN_V3)
    if catalog_id == CATALOG_V1:
        expected -= set(ADDED_IN_V2)
    assert set(manifest["candidates"]) == expected


def test_public_candidate_is_exact_archived_five_class_family():
    manifest = release_catalog()
    ids = candidate_member_ids(manifest, PUBLIC_UNIFIED)
    assert ids == [f"unified__P_five_class_trunklr1e4__fold{fold}__seed42" for fold in range(5)]
    assert all(manifest["members"][mid]["target"] == "five_class" for mid in ids)
    assert manifest["candidates"][PUBLIC_UNIFIED]["groups"][0]["weight"] == 1.0


def test_v2_manifest_cannot_carry_v3_candidates():
    manifest = release_catalog(CATALOG_V2)
    manifest["candidates"][PUBLIC_UNIFIED] = release_catalog()["candidates"][PUBLIC_UNIFIED]
    with pytest.raises(ReleaseError, match="shortlist changed"):
        validate_manifest(manifest)


@pytest.mark.parametrize("scope", [None, {}, {"candidates": []}, {"candidates": ["unknown"]},
                                    {"candidates": [PUBLIC_UNIFIED, PUBLIC_UNIFIED]}])
def test_invalid_binding_scopes_fail(scope):
    manifest = release_catalog()
    manifest["binding_scope"] = scope
    with pytest.raises(ReleaseError, match="Binding scope"):
        validate_manifest(manifest)


def test_v1_manifest_cannot_carry_v2_candidates():
    manifest = release_catalog(CATALOG_V1)
    manifest["candidates"]["top3_lf"] = release_catalog(CATALOG_V2)["candidates"]["top3_lf"]
    with pytest.raises(ReleaseError, match="shortlist changed"):
        validate_manifest(manifest)


def test_original_hierarchical_arithmetic_and_normalization():
    manifest = release_catalog()
    rng = np.random.default_rng(8)
    probabilities = {key: rng.dirichlet(np.ones(4), size=10).astype(np.float32)
                     for key in manifest["members"]}
    groups = manifest["candidates"][REFERENCE]["groups"]
    means = [np.mean([probabilities[i] for i in group["members"]], axis=0) for group in groups]
    original = (means[0] + means[1] + means[2] + means[3] + means[4]) / 5
    original = original + .25 * ((means[5] + means[6]) / 2) + .95 * ((means[7] + means[8]) / 2)
    actual = aggregate_probabilities(manifest, REFERENCE, probabilities)
    np.testing.assert_array_equal(actual, original / 2.2)
    np.testing.assert_array_equal(actual.argmax(1), original.argmax(1))
    np.testing.assert_allclose(actual.sum(1), 1, atol=2e-7)


def test_equal_group_is_not_equal_checkpoint_weighting():
    manifest = release_catalog()
    probabilities = {}
    for group in manifest["candidates"]["top4_lf"]["groups"]:
        for member in group["members"]:
            probabilities[member] = np.array([[1., 0, 0, 0]] if group["id"] == "p5" else [[0., 1, 0, 0]])
    actual = aggregate_probabilities(manifest, "top4_lf", probabilities)
    np.testing.assert_array_equal(actual, [[.25, .75, 0, 0]])
    naive = np.mean(list(probabilities.values()), axis=0)
    assert not np.array_equal(naive, actual)


def test_oof_uses_only_eligible_fold_and_never_silently_drops_members():
    manifest = release_catalog()
    probabilities = {key: np.array([[0., 1, 0, 0]]) for key, member in manifest["members"].items()
                     if member["fold"] == 2}
    np.testing.assert_array_equal(aggregate_probabilities(manifest, REFERENCE, probabilities, fold=2), [[0, 1, 0, 0]])
    with pytest.raises(ReleaseError, match="Missing probabilities"):
        aggregate_probabilities(manifest, REFERENCE, probabilities)
    probabilities.pop("mc_P_five_class__fold2__seed44")
    with pytest.raises(ReleaseError, match="Missing probabilities"):
        aggregate_probabilities(manifest, REFERENCE, probabilities, fold=2)


def test_collapse_precedes_argmax():
    native = np.array([[.4, 0, 0, .25, .2, .15]])
    common = native_to_common(native, "six_class")
    assert native.argmax(1)[0] == 0 and common.argmax(1)[0] == 3
    np.testing.assert_allclose(common, [[.4, 0, 0, .6]])
    np.testing.assert_allclose(native_to_common(np.array([[.4, 0, 0, .3, .3]]), "five_class"), common)


@pytest.mark.parametrize("mutation", [
    lambda m: m["candidates"][REFERENCE]["groups"][0].update(weight=.21),
    lambda m: m["candidates"][REFERENCE]["groups"][0]["members"].pop(),
    lambda m: m.update(reference="top4_lf"),
    lambda m: m["members"]["only_esm__four_class__baseline__fold0__seed42"].update(fold=1),
    lambda m: m["candidates"][REFERENCE].pop("blocks"),
])
def test_frozen_catalog_cannot_be_changed(mutation):
    manifest = release_catalog()
    mutation(manifest)
    with pytest.raises(ReleaseError):
        validate_manifest(manifest)


@pytest.mark.parametrize("bad", [np.array([[np.nan, 0, 0, 0]]), np.array([[.9, .2, 0, 0]]),
                                  np.array([[-.1, 1.1, 0, 0]]), np.ones((1, 3))])
def test_invalid_probabilities_refused(bad):
    with pytest.raises(ReleaseError):
        native_to_common(bad, "four_class")


def test_artifact_hash_and_manifest_relative_paths(tmp_path):
    artifact = tmp_path / "folds.csv"
    artifact.write_text("source_uid,fold\nu,0\n")
    ref = {"path": "folds.csv", "sha256": sha256_file(artifact)}
    manifest = release_catalog()
    manifest["fold_membership"] = ref
    path = tmp_path / "release.json"
    path.write_text(json.dumps(manifest))
    loaded = load_manifest(path)
    assert verify_artifact(loaded["fold_membership"]) == artifact
    assert loaded["manifest_sha256"] == sha256_file(path)
    artifact.write_text("changed")
    with pytest.raises(ReleaseError, match="mismatch"):
        verify_artifact(loaded["fold_membership"])
    with pytest.raises(ReleaseError, match="Missing"):
        verify_artifact(None)
