"""Shared synthetic helpers of the amendment-2 tests (synthetic run folders in tmp_path only; no dataset access)."""
from __future__ import annotations

from collections.abc import Mapping
import json

import numpy as np

from ec_clean30 import campaign, selection, selector
from ec_clean30.campaign import CAMPAIGN_ID, build_fold_plan, build_fold_plan_v2, canonical_json, plan_units
from ec_clean30.commands import VERIFY_UNIT_RECEIPT_NAME
from ec_clean30.provenance import sha256_file
from ec_clean30.selection import PromotionConfig

# Stand-in rows for the amendment-2 family table. The whole-protein pretraining rule defines the real
# ``campaign.FAMILY_SPECS_V2``; these tests install the stand-in only while that table is absent.
F6_SPEC = campaign.FamilySpec("F6", "only_esm", None, True, False, "candidate")
CGF_SPEC = campaign.FamilySpec("C-GF", "only_global", None, True, False, "control")


def install_family_table(monkeypatch) -> None:
    if not isinstance(getattr(campaign, "FAMILY_SPECS_V2", None), Mapping):
        monkeypatch.setattr(campaign, "FAMILY_SPECS_V2", {**campaign.FAMILY_SPECS, "F6": F6_SPEC, "C-GF": CGF_SPEC},
                            raising=False)


FREEZE1 = "f1" * 32
A1, A2 = "a1" * 32, "a2" * 32
CHAIN = [A1, A2]
MEMBERSHIP = "ee" * 32
FAST = PromotionConfig(n_draws=1000)
N_PROTEINS = 48
IDS = tuple(f"Q{index:03d}" for index in range(N_PROTEINS))
TRUE = {pid: index % 6 for index, pid in enumerate(IDS)}
INNER = {pid: (index // 6) % 4 for index, pid in enumerate(IDS)}
EC_NUMBER = {pid: f"{TRUE[pid] + 1}.1.1.{(index // 6) % 3 + 1}" for index, pid in enumerate(IDS)}
COMPONENTS = {pid: f"comp{index // 2}" for index, pid in enumerate(IDS)}
# One uniform draw per (seed, protein): a configuration with accuracy a is right where u < a.
_DRAWS = {seed: dict(zip(IDS, np.random.default_rng(seed).random(N_PROTEINS))) for seed in range(40, 47)}
TRAIN_COUNTS = {str(token): (6 if token <= 6 else 0) for token in range(1, 8)}
TABLES = selector.InnerTables(COMPONENTS, INNER, {"source": "synthetic"}, MEMBERSHIP)


def logits_of(pid, seed, accuracy):
    values = [0.0] * 7
    true = TRUE[pid]
    if _DRAWS[seed][pid] < accuracy:
        values[true], values[(true + 1) % 6] = 3.0, 1.0
    else:
        values[(true + 1) % 6], values[true] = 3.0, 1.0
    return values


CONTROL_ACCURACY = 0.6
CONTROL_COLUMNS = None  # the standard protein-prediction columns (selector.PROTEIN_PREDICTION_COLUMNS)


def _protein_rows(unit, accuracy, *, seed_offset=0):
    rows = []
    members = [pid for pid in IDS if INNER[pid] == unit.inner_fold]
    for pid in members:
        logits = logits_of(pid, unit.seed + seed_offset if unit.seed + seed_offset in _DRAWS else unit.seed, accuracy)
        sid = f"{pid}__chain_A__EC_{EC_NUMBER[pid]}"
        rows.append([unit.unit_id, "validation", str(unit.inner_fold), "4", str(unit.seed), pid, pid, sid, "1",
                     str(TRUE[pid]), str(int(np.argmax(logits))), *[repr(value) for value in logits]])
    return rows


def write_control_files(run_dir, unit, *, accuracy=CONTROL_ACCURACY, branch_digest="b1" * 32, changes=None):
    """The F6 by-products of an inner fit: the pretraining receipt and the C-GF control file (rule A2-F6-7)."""
    import hashlib

    header = list(selector.PROTEIN_PREDICTION_COLUMNS)
    rows = _protein_rows(unit, accuracy, seed_offset=1)
    path = run_dir / selector.WP_CONTROL_PREDICTIONS_NAME
    path.write_text("\n".join([",".join(header), *(",".join(row) for row in rows)]) + "\n")
    content = [{key: value for key, value in zip(header, row) if key != "run_id"} for row in rows]
    content_sha = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    receipt = {"format": "deepmzyme_global_esm_pretrain_receipt_v1", "split": f"inner{unit.inner_fold}",
               "branch_digest": branch_digest,
               "control_predictions": {"control": "C-GF", "path": path.name, "sha256": sha256_file(path),
                                       "n_rows": len(rows), "content_sha256": content_sha}}
    receipt.update(changes or {})
    (run_dir / selector.WP_PRETRAIN_RECEIPT_NAME).write_text(json.dumps(receipt, indent=2, sort_keys=True))


def write_unit(runs, unit, *, accuracy=0.7, chain=CHAIN, identity_changes=None, control=True):
    """One finished synthetic plan-v2 unit with the standard prediction receipt (F6 adds its control files)."""
    runs.mkdir(parents=True, exist_ok=True)
    run_dir = runs / unit.unit_id
    run_dir.mkdir()
    identity = {"campaign_id": CAMPAIGN_ID, "unit_id": unit.unit_id, "step": unit.step, "kind": unit.kind,
                "fold": unit.fold, "inner_fold": unit.inner_fold, "seed": unit.seed, "smoke": False,
                "recipe": selector.unit_recipe_record_v2(unit), "membership_sha256": MEMBERSHIP,
                "freeze1_sha256": FREEZE1, "freeze1_amendment_chain": list(chain)}
    identity.update(identity_changes or {})
    runner = {"schema_version": 1, "unit_id": unit.unit_id, "status": "completed", "returncode": 0,
              "unit": {"unit_id": unit.unit_id, "kind": identity["kind"], "step": identity["step"],
                       "seed": identity["seed"], "inner_fold": identity["inner_fold"], "recipe": identity["recipe"],
                       "identity": identity}}
    (runs / f"{unit.unit_id}.runner.json").write_text(json.dumps(runner, indent=2, sort_keys=True))
    config = {"seed": unit.seed, "campaign_run_identity": canonical_json(identity)}
    epochs = unit.epochs
    history = [{"epoch": epoch, "train_loss": 0.5} for epoch in range(1, epochs + 1)]
    (run_dir / "run_config.json").write_text(json.dumps({"config": config, "history": history}, sort_keys=True))
    checkpoint = run_dir / "terminal_model_checkpoint.pt"
    checkpoint.write_bytes(f"checkpoint {unit.unit_id}".encode())
    rows = ["run_id,mode,fold_index,n_folds,seed,accession,ec_group_key,structure_ids,n_pockets,y,pred,"
            + ",".join(selector.LOGIT_COLUMNS)]
    members = [pid for pid in IDS if INNER[pid] == unit.inner_fold]
    for pid in members:
        logits = logits_of(pid, unit.seed, accuracy)
        sid = f"{pid}__chain_A__EC_{EC_NUMBER[pid]}"
        rows.append(",".join([unit.unit_id, "validation", str(unit.inner_fold), "4", str(unit.seed), pid, pid, sid, "1",
                              str(TRUE[pid]), str(int(np.argmax(logits))), *[repr(value) for value in logits]]))
    csv_path = run_dir / "ec_val_protein_predictions.csv"
    csv_path.write_text("\n".join(rows) + "\n")
    receipt = {
        "mode": "validation", "run_id": unit.unit_id, "campaign_run_identity": identity, "fold_index": unit.inner_fold,
        "n_folds": 4, "seed": unit.seed,
        "checkpoint": {"path": checkpoint.name, "sha256": sha256_file(checkpoint), "epoch": epochs,
                       "checkpoint_rule": "terminal"},
        "run_config_sha256": sha256_file(run_dir / "run_config.json"),
        "training_protein_class_counts": dict(TRAIN_COUNTS),
        "epochs": {"planned": epochs, "completed": epochs}, "ec_supcon_coverage": None,
        "protein_predictions": {"path": csv_path.name, "sha256": sha256_file(csv_path), "n_rows": len(members)},
        "reconciliation_status": "match",
    }
    (run_dir / "ec_prediction_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True))
    if control and unit.family == "F6":
        write_control_files(run_dir, unit)


def write_verify_receipt(runs):
    units = []
    for path in sorted(runs.glob("*.runner.json")):
        unit_id = path.name[: -len(".runner.json")]
        units.append({"unit_id": unit_id, "passed": True,
                      "terminal_checkpoint_sha256": sha256_file(runs / unit_id / "terminal_model_checkpoint.pt")})
    (runs / VERIFY_UNIT_RECEIPT_NAME).write_text(json.dumps({"status": "passed", "units": units}, indent=2))


def write_plan_v2(root, fold=0):
    path = root / "plans_v2" / f"fold{fold}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_fold_plan_v2(fold), indent=2, sort_keys=True) + "\n")
    return path


def write_plan_v1(root, fold=0):
    path = root / "plans" / f"fold{fold}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_fold_plan(fold), indent=2, sort_keys=True) + "\n")
    return path


def amendment2(monkeypatch, tmp_path, *, chain=CHAIN, published=True, policies=None, tip_sequence=None,
               fit_chain=None, identity="5c" * 32):
    """An amendment-2 document whose authentication returns the given chain and fit chain (the chain verifier's own
    tests, ``test_ec_clean30_amendment_chain`` and ``test_ec_clean30_amendment2_build_chain``, authenticate real
    documents)."""
    path = tmp_path / "amendment2" / "amendment.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"format": selector.AMENDMENT_V2_FORMAT}))
    calls = []
    if policies is None:
        policies = {"AM2-S1": {"rule_id": selection.BEST_INNER_S1_RULE,
                               "policy": selection.selection_policy_record_v2()}}

    def verify(given, freeze1_sha256, *, smoke, chain_receipt=None, expected_chain=None):
        assert given == path and freeze1_sha256 == FREEZE1
        # A later check of one call expects the chain the first check authenticated.
        assert expected_chain is None or list(expected_chain) == list(chain)
        calls.append(smoke)
        return {"chain": list(chain), "fit_chain": list(chain if fit_chain is None else fit_chain),
                "tip_sequence": len(chain) if tip_sequence is None else tip_sequence, "policies": policies,
                "published": published, "scientific_identity_sha256": identity}

    monkeypatch.setattr(selector, "_verify_amendment_chain", verify)
    return path, calls


def world(tmp_path, *, accuracies=None, skip=None, identity_changes=None):
    """The plan-v2 fold-0 S1 units as finished synthetic fits (F6 with its C-GF control files); ``accuracies``
    per family."""
    root = tmp_path / CAMPAIGN_ID
    plan_path = write_plan_v2(root)
    runs = root / "runs" / "fold0"
    for unit in plan_units(build_fold_plan_v2(0)):
        if unit.step != "S1" or (skip and skip(unit)):
            continue
        accuracy = (accuracies or {}).get(unit.family, 0.7)
        write_unit(runs, unit, accuracy=accuracy, identity_changes=(identity_changes or {}).get(unit.unit_id))
    write_verify_receipt(runs)
    return {"root": root, "plan_path": plan_path, "runs": runs}


def select(world_, step, out, amendment, **options):
    if step == "S1" and "clean_sanity_record_path" not in options:
        options["clean_sanity_record_path"] = world_.get("sanity")
    if step == "S4" and "campaign_root" not in options:
        options["campaign_root"] = world_["root"]
    return selector.run_select(fold=0, step=step, plan_path=world_["plan_path"], runs_dirs=[world_["runs"]],
                               tables=TABLES, freeze1_sha256=FREEZE1, out=out, provenance={"note": "synthetic"},
                               base_config=FAST, freeze1_amendment_path=amendment, **options)


def sanity_stub(monkeypatch, tmp_path, *, status="passed", disposition=None, name="held_out_sanity.json"):
    """A held-out CLEAN sanity record file whose verification returns ``status`` (the record format belongs to
    ``ec_clean30.clean_sanity``, whose own tests build real records)."""
    from ec_clean30 import clean_sanity

    path = tmp_path / "sanity" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"synthetic": name}) + "\n")

    def binding(given, dispositions=()):
        if status != "passed" and disposition is None:
            raise clean_sanity.SanityError(f"The held-out sanity record has status {status!r}: comparator diagnosis is "
                                           "required; fix the defects first, without retuning")
        return {"sha256": sha256_file(given), "check": "held_out", "status": status, "disposition": disposition,
                "fold": 0}

    monkeypatch.setattr(clean_sanity, "held_out_binding", binding)
    return path


def write_clean_exports(folder, arm, *, wrong=(), tie=None, metadata_changes=None, drop_query=None):
    """Four synthetic CLEAN inner exports (contract C4) of one arm on the fixture's inner folds."""
    from ec_clean30 import scoring

    comparator = scoring.comparator
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for inner in range(4):
        queries = [pid for pid in IDS if INNER[pid] == inner and pid != drop_query]
        labels = sorted({EC_NUMBER[pid] for pid in IDS})
        distances = np.full((len(queries), len(labels)), 5.0, dtype=np.float32)
        for row, pid in enumerate(queries):
            target = EC_NUMBER[pid]
            if pid in wrong:
                target = next(label for label in labels if not label.startswith(f"{TRUE[pid] + 1}."))
            distances[row, labels.index(target)] = 1.0
            if pid == tie:
                other = next(label for label in labels if not label.startswith(f"{TRUE[pid] + 1}."))
                distances[row, labels.index(other)] = 1.0
        metadata = {"fold": 0, "regime": arm, "seed": 7, "input_dim": 4, "embedding": "synthetic", "mode": "validation",
                    "checkpoint_sha256": "a" * 64, "training_table_sha256": "b" * 64,
                    "embedding_farm_sha256": "c" * 64, "query_embeddings_sha256": "d" * 64, "inner_fold": inner,
                    "n_in_sample_queries": 0, **(metadata_changes or {})}
        path = folder / f"{arm}_fold0_inner{inner}.npz"
        comparator.write_distance_export(path, distances, queries, labels, metadata)
        paths.append(path)
    return paths


def append_fut_record(world_, out, *, decision="continue", binding=True, sequence=2, record_sha256="fe" * 32,
                      pointer=True, campaign_root=None):
    """A FUT step record and its pointer file as ``futility.run_futility`` writes them (the projection itself is
    tested in ``test_ec_clean30_futility``)."""
    from ec_clean30 import futility

    records = selection.read_step_chain(out, anchor_sha256=FREEZE1)
    chain = selector._fold_records(records, 0, selector.STEP_RECORD_FORMAT_V2)
    s1_payload, s1_sha = chain["S1"]
    s2_payload, s2_sha = chain["S2"]
    keys = ("freeze1_amendment_chain", "s1_rule", "selection_pipeline", "selection_policy_sha256",
            "amendment_tip_chain", "scientific_identity_sha256")
    policy = {key: s2_payload[key] for key in keys if key in s2_payload}
    root = world_["root"] if campaign_root is None else campaign_root
    pointer_sha = None
    if pointer:
        record = {"format": futility.FUTILITY_POINTER_FORMAT, "schema_version": 1, "campaign_id": CAMPAIGN_ID,
                  "sequence": sequence, "decision": decision, "binding": binding,
                  "futility_record_name": "futility.json", "futility_record_sha256": record_sha256,
                  "fit_chain": list(policy["freeze1_amendment_chain"]),
                  "amendment_tip_chain": list(policy.get("amendment_tip_chain") or policy["freeze1_amendment_chain"]),
                  "step_chain_head_before_sha256": selection.record_sha256(records[-1])}
        path = futility.pointer_path(root, sequence)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2, sort_keys=True))
        pointer_sha = sha256_file(path)
    payload = {"format": selector.STEP_RECORD_FORMAT_V2, "schema_version": 2, "campaign_id": CAMPAIGN_ID, "fold": 0,
               "step": "FUT", "plan_sha256": s2_payload["plan_sha256"], "freeze1_sha256": FREEZE1, **policy,
               "smoke": False, "components": s2_payload["components"], "s1_record_sha256": s1_sha,
               "s2_record_sha256": s2_sha, "futility_record_sha256": record_sha256, "decision": decision,
               "binding": binding, "pointer_sequence": sequence, "pointer_sha256": pointer_sha}
    return selection.append_step_record(out, "FUT", payload, anchor_sha256=FREEZE1)
