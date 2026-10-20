"""Fixed historical Phase 2 shortlist with its ensemble arithmetic reproduced from source.

Membership is historical; the release audit records its provenance and limits.
"""
from __future__ import annotations

from .manifest import ARTIFACT_KEYS, SCHEMA

REFERENCE = "unified_trimodal_cv_locked"
CATALOG_V1 = "phase2-unified-20261019"
CATALOG_V2 = "phase2-unified-20261019-v2"
CATALOG_V3 = "phase2-unified-20261019-v3"
CATALOG_VERSIONS = {CATALOG_V1: "phase2-preserved-v1", CATALOG_V2: "phase2-preserved-v2",
                    CATALOG_V3: "phase2-preserved-v3"}
# v2 adds two compact comparisons built only from v1 members; the v1 binding stays valid.
ADDED_IN_V2 = ("top3_lf", "recipe_p_5class_3seed")
# v3 exposes the archived family without changing existing ensembles or bindings.
PUBLIC_UNIFIED = "unified_p5_trunklr1e4_5fold_s42"
ADDED_IN_V3 = (PUBLIC_UNIFIED,)


def release_catalog(catalog_id: str = CATALOG_V3) -> dict:
    if catalog_id not in CATALOG_VERSIONS:
        raise ValueError(f"Unknown release catalog: {catalog_id}")
    members = {}

    def family(prefix, target, seeds=(42,)):
        ids = []
        for seed in seeds:
            for fold in range(5):
                name = f"{prefix}__fold{fold}__seed{seed}"
                members[name] = {"target": target, "fold": fold, "seed": seed,
                                 **dict.fromkeys(ARTIFACT_KEYS)}
                ids.append(name)
        return ids

    baseline = family("gvp_late_fusion__four_class__baseline", "four_class")
    trunk = family("gvp_late_fusion__four_class__trunklr1e4", "four_class")
    p5 = family("mc_P_five_class", "five_class", (42, 43, 44))
    s5 = family("mc_S_five_class", "five_class")
    six = family("gvp_late_fusion__six_class__baseline", "six_class")
    unified = family("unified__P_five_class_trunklr1e4", "five_class")
    esm4 = family("only_esm__four_class__baseline", "four_class")
    esm6 = family("only_esm__six_class__baseline", "six_class")
    gvp4 = family("only_gvp__four_class__baseline", "four_class")
    geometry = family("only_gvp__four_class__sitecountsangles", "four_class")
    members["stage6b_refit"] = {"target": "four_class", "fold": None, "seed": 42,
                               **dict.fromkeys(ARTIFACT_KEYS)}

    def group(name, ids, weight=1.0):
        return {"id": name, "weight": weight, "members": ids}

    candidates = {
        "step_f_single_refit_s42": {"groups": [group("refit", ["stage6b_refit"])],
                                   "oof_proxy": "baseline_5fold_s42"},
        "baseline_5fold_s42": {"groups": [group("baseline", baseline)], "proxy_only": True,
                               "reported_oof_ba_percent": 74.34},
        "trunklr1e4_5fold_s42": {"groups": [group("trunk", trunk)], "reported_oof_ba_percent": 75.51},
        PUBLIC_UNIFIED: {"groups": [group("unified", unified)], "reported_oof_ba_percent": 74.91},
        "recipe_p_5class_3seed": {"groups": [group("p5", p5)], "reported_oof_ba_percent": 75.82},
        "top4_lf_seed42": {"groups": [group("trunk", trunk), group("p5", p5[:5]),
                                       group("s5", s5), group("six", six)], "reported_oof_ba_percent": 76.34},
        "top3_lf": {"groups": [group("trunk", trunk), group("p5", p5), group("six", six)],
                    "reported_oof_ba_percent": 76.62},
        "top4_lf": {"groups": [group("trunk", trunk), group("p5", p5), group("s5", s5), group("six", six)],
                    "reported_oof_ba_percent": 76.81},
        REFERENCE: {"groups": [group("unified", unified, .2), group("trunk", trunk, .2),
                               group("p5", p5, .2), group("s5", s5, .2), group("six", six, .2),
                               group("esm4", esm4, .125), group("esm6", esm6, .125),
                               group("gvp4", gvp4, .475), group("geometry", geometry, .475)],
                    "blocks": [{"groups": ["unified", "trunk", "p5", "s5", "six"], "weight": 1.0},
                               {"groups": ["esm4", "esm6"], "weight": .25},
                               {"groups": ["gvp4", "geometry"], "weight": .95}],
                    "reported_oof_ba_percent": 77.17},
    }
    if catalog_id == CATALOG_V1:
        for name in ADDED_IN_V2:
            del candidates[name]
    if catalog_id in {CATALOG_V1, CATALOG_V2}:
        for name in ADDED_IN_V3:
            del candidates[name]
    return {
        "schema": SCHEMA, "catalog_id": catalog_id, "version": CATALOG_VERSIONS[catalog_id],
        "reference": REFERENCE, "n_folds": 5, "fold_membership": None,
        "release_status": "blocked", "embedding_model": "esmc_600m",
        "class_order": ["Mn", "Cu", "Zn", "Class VIII"],
        "selection_provenance": {"status": "unverified", "note": "Historical CV locking needs dated evidence."},
        "members": members,
        "candidates": candidates,
    }
