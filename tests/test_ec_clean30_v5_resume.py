"""Synthetic A2-D6 resume checks: carried G3 allocations, retried farms and batched restore.

A fake transport runs the pod side of the restore as the same program on local folders; no
provider, network, GPU, training or private campaign data is used.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ec_clean30 import campaign, execution_v5 as v5, phase3, phase4, workstation

IDENTITY = {"scientific_identity_sha256": "a" * 64, "fit_chain": ["b" * 64, "c" * 64]}
SOURCE = {"format": phase3.EXECUTION_V5_FORMAT, "published": True, **IDENTITY, "git_head": "e" * 40,
          "freeze1_sha256": "d" * 64, "execution_manifest_sha256": "9" * 64, "amendment_chain": IDENTITY["fit_chain"]}
HOUR = 3600
PLAN = "2" * 64
BASE, MASKED = "ecc30_clean_embed_base", "ecc30_clean_embed_masked"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value, sort_keys=True) + "\n")
    return path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def session(name, started, hours=72, reserve=120):
    # The controller's shape: the stop-request deadline lies one stop reserve before the hard stop.
    hard = started + hours * HOUR
    return {"pod_id": f"{name}-pod", "session_id": name, "allocation_started_unix": started,
            "hard_stop_unix": hard, "deadline_unix": hard - reserve, "max_total_seconds": hours * HOUR}


class Transport:
    """Uploads copy local files; the restore command runs locally as the pod would run it."""

    def __init__(self):
        self.uploads, self.commands = [], []
        self.stop_controller = SimpleNamespace(register_output_roots=lambda outputs: None,
                                               admit_output_root=lambda root: None, config_sha="8" * 64)

    def upload(self, local, target, *, timeout):
        assert timeout > 0
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(local).read_bytes())
        self.uploads.append(target)

    def remote(self, argv, *, timeout):
        assert argv[:3] == [sys.executable, "-c", v5.RESTORE_PROGRAM] and timeout > 0
        self.commands.append(argv)
        result = subprocess.run(argv, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr[-2000:])
        return result.stdout


@pytest.fixture
def build(tmp_path, monkeypatch):
    mirrors = set()
    original_stat = Path.stat

    def stat(path, **kwargs):  # Device separation is synthetic; every byte and hash check stays real.
        value = original_stat(path, **kwargs)
        if path in mirrors:
            fields = list(value)
            fields[2] += 1
            return os.stat_result(fields)
        return value

    monkeypatch.setattr(Path, "stat", stat)
    recipes = tuple((f"science_recipe_{index}", object()) for index in range(7))
    monkeypatch.setattr(campaign, "timing_recipes_v2", lambda: recipes, raising=False)
    monkeypatch.setattr(campaign, "build_fold_plan_v2", lambda fold: {"plan_sha256": PLAN}, raising=False)
    monkeypatch.setattr(phase3, "require_record_precedence",
                        lambda record, source: phase3.require_same_scientific_identity(record, source))
    monkeypatch.setattr(phase3, "verify_pretrain_evidence", lambda *args, **kwargs: {}, raising=False)

    def preserve(name, execution_root, files):
        primary, mirror = tmp_path / f"old_{name}", tmp_path / f"old_{name}_mirror"
        mirrors.add(mirror)
        rows = []
        for relative, data in files.items():
            for root in (primary, mirror):
                write(root / relative, data)
            rows.append({"path": relative, "size_bytes": len(data.encode()), "sha256": sha(primary / relative)})
        relative = "closeout/remote_inventory.json"
        for root in (primary, mirror):
            write(root / relative, {"session_id": "s1", "files": rows, "pending_transfer": None, "pending_unit": None})
        rows.append({"path": relative, "size_bytes": (primary / relative).stat().st_size, "sha256": sha(primary / relative)})
        receipt = write(tmp_path / f"receipt_{name}.json", {"format": "deepmzyme_ec_clean30_runpod_preservation_v1",
            "session_id": "s1", "pod_id": "s1-pod", "execution_root": execution_root, "primary_root": str(primary),
            "mirror_root": str(mirror), "files": rows, "remote_inventory_path": relative})
        return {"path": str(receipt), "sha256": sha(receipt)}, primary

    def make(*, g2_files=None, phase4_started=True, s1_cost=85, s2_cost=170, g3=None, idle_allocation=False):
        # g3: session 1 materialized its G3 (default: exactly when its Phase-4 work started).
        # idle_allocation: that G3 also holds an earlier session never admitted to Phase 4.
        g3 = phase4_started if g3 is None else g3
        remote = tmp_path / "pod" / "campaign"
        manifest = write(tmp_path / "manifest.json", SOURCE)
        evidence = []
        for index in range(10):
            path = write(tmp_path / "g2_evidence" / f"measurement{index}.json", IDENTITY)
            evidence.append({"path": str(path), "sha256": sha(path)})
        pretrain = write(tmp_path / "g2_evidence" / "pretrain.json", {"status": "passed", **IDENTITY})
        freeze = write(tmp_path / "freeze1c_v2.json", {**IDENTITY, "format": "deepmzyme_ec_clean30_freeze1c_v2",
            "campaign_id": campaign.CAMPAIGN_ID, "freeze1_sha256": SOURCE["freeze1_sha256"], "lanes": 1,
            "evidence": evidence, "timing_recipes": [name for name, _ in recipes],
            "pretrain_evidence": {"path": str(pretrain), "sha256": sha(pretrain)}})
        family = {"campaign_id": campaign.CAMPAIGN_ID, "scope": phase4.V5_SCOPE, "allowed_folds": [0],
                  "plan_sha256_by_fold": {"0": PLAN}, "allowed_steps": {"0": ["S1", "S4"]},
                  "clean_unit_ids": sorted(unit["unit_id"] for unit in phase4.clean_units()),
                  "clean_inputs_sha256": "3" * 64, "hot_work_root": str(tmp_path / "hot")}

        def preapproval(path, cost, hours):
            return write(path, {"format": v5.PREAPPROVAL_FORMAT, "status": "approved", **IDENTITY, **family,
                                "ceilings": {"gpu_seconds": hours * HOUR, "cpu_seconds": HOUR, "cost_usd": cost}})

        s1, s2 = session("s1", 1_000_000.0), session("s2", 2_000_000.0)
        prior_state_root = tmp_path / "s1_state"
        pre1 = preapproval(tmp_path / "s1_preapproval.json", s1_cost, 72)
        prior_g3 = prior_state_root / "g3_v5.json"
        if g3:
            v5.materialize_g3(json.loads(pre1.read_text()), preapproval_sha256=sha(pre1), freeze1c=freeze,
                              out=prior_g3, source=SOURCE, session=s1, gross_ceiling_usd=s1_cost)
        if idle_allocation:
            document = json.loads(prior_g3.read_text())
            idle = session("s0", 500_000.0, hours=1)
            document["allocations"]["s0"] = {"resource": "gpu", "cost_ceiling_usd": 5,
                **{key: idle[key] for key in ("allocation_started_unix", "deadline_unix", "max_total_seconds")}}
            prior_g3.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n")
        root = remote / "runs/fold0/phase4"
        receipts = []
        g2 = g2_files if g2_files is not None else {"runs/ecc30timing_kept/g2_complete.json": json.dumps({"kept": True})}
        g2_receipt, g2_primary = preserve("g2", str(remote / "runs/fold0/g2"), g2)
        receipts.append(g2_receipt)
        p4_primary = tmp_path / "old_p4"
        if phase4_started:
            state = {"schema_version": 1, "allocations": {"s1": {"started_unix": s1["allocation_started_unix"],
                     "start_basis": "provider_allocation", "deadline_unix": s1["deadline_unix"],
                     "max_total_seconds": s1["max_total_seconds"], "provider_stop_verified": False}},
                     "pending_unit": {"run_name": BASE, "status": "completed", "persisted": True},
                     "active_child": None, "owner": None}
            files = {"execution_state.json": json.dumps(state),
                     f"runs/{BASE}/attempt1/unit_result.json": json.dumps({"status": "passed", "unit_id": BASE}),
                     f"runs/{BASE}/attempt1/farm/farm_manifest.tsv": "label\tfile_sha256\n",
                     f"runs/{BASE}/complete.json": json.dumps({"status": "passed", "unit_id": BASE, "attempt": 1}),
                     f"runs/{MASKED}/attempt1/farm/partial.pt": "lost before its result",
                     f"commands/{BASE}_attempt1.json": "{}", f"commands/{MASKED}_attempt1.json": "{}",
                     f"queue/s1/{BASE}_attempt1.json.job/command.json": json.dumps({"argv": []})}
            for unit in (BASE, MASKED):
                files[f"queue/s1/{unit}_attempt1.json"] = json.dumps({"session_id": "s1", "unit_id": unit,
                                                                      "g3_sha256": sha(prior_g3)})
            p4_receipt, p4_primary = preserve("p4", str(root), files)
            receipts.append(p4_receipt)
        proof = write(tmp_path / "preservation_set.json", {"format": "deepmzyme_ec_clean30_runpod_preservation_set_v1",
                                                            "session_id": "s1", "pod_id": "s1-pod", "receipts": receipts})
        prior_request = write(tmp_path / "s1_request.json", {"durable_root": str(tmp_path / "old_p4"),
                                                             "mirror_root": str(tmp_path / "old_p4_mirror")})
        prior = {"hardware_profile": "runpod_rtx4090", "runtime_environments": {"pinned": True},
                 "remote_repo": str(tmp_path / "pod/repo"), "remote_campaign": str(remote),
                 "remote_python": "/runtime/python", "bootstrap_python": sys.executable, "embedding": {"pinned": True},
                 "state_root": str(prior_state_root), "execution_manifest": str(manifest),
                 "runpod": {"gpu_id": "4090", "image": "sha256:fixed", "runtime_environment": {"TF32": "0"},
                            "state_root": str(tmp_path / "s1_controller"), "gross_ceiling_usd": s1_cost},
                 "durable_root": str(tmp_path / "old_g2"), "mirror_root": str(tmp_path / "old_g2_mirror"),
                 "phase4": {"request_path": str(prior_request), "request_sha256": sha(prior_request),
                            "preapproval_path": str(pre1), "preapproval_sha256": sha(pre1)}}
        prior_path = write(tmp_path / "s1_config.json", prior)
        driver = {"closed": True, "config_sha256": sha(prior_path), "session_id": "s1", "pod_id": "s1-pod",
                  "status": "interrupted", "completed": [],
                  "closeout": {"status": "DELETED", "pod_id": "s1-pod", "provider_absence_verified": True,
                               "preservation_sha256": sha(proof)}}
        if g3:
            driver["g3_authority"] = {"path": str(prior_g3), "sha256": sha(prior_g3)}
        template = {"execution_root": str(root), "hot_work_root": family["hot_work_root"],
                    **{f"{kind}_farm": str(root / "runs" / f"ecc30_clean_embed_{kind}" / "attempt1" / "farm")
                       for kind in ("base", "masked")}}
        request = write(tmp_path / "s2_request.json", {**IDENTITY, "fold": 0, "execution_manifest": str(manifest),
            "durable_root": str(tmp_path / "new_p4"), "mirror_root": str(tmp_path / "new_p4_mirror"),
            "host_state_root": str(tmp_path / "new_host"), "max_download_bytes": 10 ** 9,
            "remote_request_template": template, "stage_files": []})
        pre2 = preapproval(tmp_path / "s2_preapproval.json", s2_cost, 144)
        world = SimpleNamespace(tmp=tmp_path, remote=remote, root=root, freeze=freeze, pretrain=pretrain,
                                s1=s1, s2=s2, prior_g3=prior_g3, prior=prior, driver=driver, proof=proof,
                                g2_primary=g2_primary, p4_primary=p4_primary, transport=Transport(), checks=[])
        world.config = {**prior, "state_root": str(tmp_path / "s2_state"), "durable_root": str(tmp_path / "new_g2"),
                        "mirror_root": str(tmp_path / "new_g2_mirror"),
                        "runpod": {**prior["runpod"], "state_root": str(tmp_path / "s2_controller"),
                                   "gross_ceiling_usd": 85, "max_runtime_seconds": 72 * HOUR},
                        "phase4": {"request_path": str(request), "request_sha256": sha(request),
                                   "preapproval_path": str(pre2), "preapproval_sha256": sha(pre2)}}
        world.state_path = Path(world.config["state_root"]) / "driver_state.json"
        world.rebind = lambda: rebind(world)
        rebind(world)
        return world

    def rebind(world):
        driver = write(world.tmp / "s1_driver.json", world.driver)
        world.config["resume"] = {"config_path": str(world.tmp / "s1_config.json"),
                                  "config_sha256": sha(world.tmp / "s1_config.json"),
                                  "state_path": str(driver), "state_sha256": sha(driver),
                                  "preservation_path": str(world.proof), "preservation_sha256": sha(world.proof)}
        world.config_path = write(world.tmp / "s2_config.json", world.config)

    return make


def restore(world):
    state = {}
    v5.restore_session(world.config, world.transport, state, world.state_path, SOURCE,
                       lambda: world.checks.append("check"))
    return state


def prepare(world, state):
    state.update(freeze1c={"local": str(world.freeze), "remote": f"{world.remote}/g2_requests/freeze1c_v2.json"},
                 g2_evidence={}, pretrain_preparation={"sha256": sha(world.pretrain), "local_path": str(world.pretrain),
                                                       "remote_path": "/remote/pretrain.json", "record": {}})
    return v5.prepare_phase4(world.config, None, state, world.s2, {"source": SOURCE}, "/remote/manifest.json",
                             "/remote/probe.json", lambda: None)


def runtime_request(world, prepared, monkeypatch):
    """One Phase-4 unit request as the pod's _runtime gate sees it."""
    view = SimpleNamespace(fold=0, campaign_root=world.remote, default_runs_dir=world.remote / "runs/fold0",
                           manifest_sha256="5" * 64)
    monkeypatch.setattr(phase4, "_source_view", lambda request: (SOURCE, view))
    monkeypatch.setattr(phase4, "require_futility", lambda *args, **kwargs: None)
    probe = write(world.tmp / "probe.json", {"source": IDENTITY, "status": "passed", "view_manifest_sha256": "5" * 64})
    return {**{key: world.s2[key] for key in ("session_id", "allocation_started_unix", "deadline_unix",
                                               "max_total_seconds")},
            "freeze1c": str(world.freeze), "g3_authority": prepared["g3_authority"], "g3_sha256": prepared["g3_sha256"],
            "provider": "runpod", "campaign_root": str(world.remote), "execution_root": str(world.root),
            "cap_prefix": ["cap", "--"], "unit_timeout_seconds": 90, "forecast_seconds": 100,
            "memory_limit_bytes": 1000, "device": "cuda", "persistence_mode": "host_pull", "vm_probe": str(probe)}


def test_resume_after_phase4_started_carries_allocations_and_admits_units(build, monkeypatch):
    world = build()
    state = restore(world)
    assert state["resume_g3"] == {**world.driver["g3_authority"], "sessions": ["s1"]}
    # The finished embedding unit is kept at its result; the lost one gets its single identical retry.
    assert state["resume_attempts"] == {BASE: 1, MASKED: 2}
    for name in (f"runs/{BASE}/complete.json", f"runs/{BASE}/attempt1/unit_result.json", "execution_state.json"):
        assert (world.root / name).read_bytes() == (world.p4_primary / name).read_bytes()
    prepared = prepare(world, state)
    g3 = json.loads(Path(prepared["g3_authority"]).read_text())
    prior = json.loads(world.prior_g3.read_text())
    assert list(g3["allocations"]) == ["s1", "s2"] and g3["allocations"]["s1"] == prior["allocations"]["s1"]
    assert g3["allocations"]["s2"]["allocation_started_unix"] == world.s2["allocation_started_unix"]
    assert state["g3_authority"] == {"path": prepared["g3_authority"], "sha256": prepared["g3_sha256"]}
    # Recovery of the same session reuses the write-once G3 unchanged.
    assert prepare(world, state)["g3_sha256"] == prepared["g3_sha256"]
    # The pod's real Phase-4 gate now admits the resumed session's units over the restored allocation history.
    unit = runtime_request(world, prepared, monkeypatch)
    source, view, policy, authority, _ = phase4._runtime(unit, resource="gpu")
    assert set(authority["allocations"]) == {"s1", "s2"} and policy.max_total_seconds == world.s2["max_total_seconds"]
    # A G3 holding only the new session, as written before this fix, is refused by the same gate.
    single = write(world.tmp / "single_session_g3.json", {**g3, "allocations": {"s2": g3["allocations"]["s2"]}})
    with pytest.raises(phase4.Phase4Error, match="cannot reset or omit"):
        phase4._runtime({**unit, "g3_authority": str(single), "g3_sha256": sha(single)}, resource="gpu")


def launch_gate(world, tmp_path, monkeypatch, costs):
    """The pre-allocation launch gate for each G3 preapproval cost: True if accepted."""
    monkeypatch.setattr(phase3, "authenticated_chain", lambda source: source)
    monkeypatch.setattr(workstation, "validate_phase4_plan", lambda *args: None)
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o700, exist_ok=True)
    authorization = {"format": v5.AUTHORIZATION_FORMAT, "launch_authorized": True, "provider": "runpod",
                     "scope": "g2_then_fold0_v5", "approved_by": "maintainer-fixture", **IDENTITY,
                     "execution_manifest_sha256": SOURCE["execution_manifest_sha256"], "fold0_cost_ceiling_usd": 85,
                     "failed_window_limit": 2, "failure_ledger": str(shared / "ledger.json"),
                     "failure_ledger_authorization_sha256": "f" * 64}
    world.config.update(provider="runpod", execution_scope="g2_then_fold0_v5")
    results = {}
    for cost in costs:
        pre = json.loads(Path(world.config["phase4"]["preapproval_path"]).read_text())
        path = write(tmp_path / f"preapproval_{cost}.json", {**pre, "ceilings": {**pre["ceilings"], "cost_usd": cost}})
        world.config["phase4"].update(preapproval_path=str(path), preapproval_sha256=sha(path))
        world.rebind()
        auth = write(tmp_path / f"authorization_{cost}.json", {**authorization, "config_sha256": sha(world.config_path)})
        try:
            v5.validate_launch_authorization(world.config, world.config_path, auth, SOURCE)
            results[cost] = True
        except v5.LaunchWindowError as exc:
            assert "G3 preapproval" in str(exc)
            results[cost] = False
    return results


def test_resume_g3_cost_must_equal_the_cumulative_total(build, tmp_path, monkeypatch):
    world = build(s2_cost=120)
    state = restore(world)
    with pytest.raises(v5.LaunchWindowError, match="must cover every carried-forward allocation"):
        prepare(world, state)
    assert not (Path(world.config["state_root"]) / "g3_v5.json").exists()
    # The launch gate binds the exact cumulative cost before any provider access: never less, never a looser bound.
    assert launch_gate(world, tmp_path, monkeypatch, (85, 120, 170, 171, 1700)) == {
        85: False, 120: False, 170: True, 171: False, 1700: False}


def test_resume_before_any_phase4_admission_keeps_the_per_session_cost(build, tmp_path, monkeypatch):
    # Session 1 materialized its G3 but admitted no Phase-4 unit: nothing is carried forward.
    world = build(phase4_started=False, g3=True, s2_cost=85)
    assert world.driver["g3_authority"]
    state = restore(world)
    assert state["resume_g3"] is None
    g3 = json.loads(Path(prepare(world, state)["g3_authority"]).read_text())
    assert list(g3["allocations"]) == ["s2"]
    assert launch_gate(world, tmp_path, monkeypatch, (85, 170)) == {85: True, 170: False}


def test_resume_carries_only_sessions_admitted_to_phase4(build, tmp_path, monkeypatch):
    world = build(idle_allocation=True)
    assert set(json.loads(world.prior_g3.read_text())["allocations"]) == {"s0", "s1"}
    state = restore(world)
    assert state["resume_g3"]["sessions"] == ["s1"]
    g3 = json.loads(Path(prepare(world, state)["g3_authority"]).read_text())
    assert set(g3["allocations"]) == {"s1", "s2"}
    assert launch_gate(world, tmp_path, monkeypatch, (170, 175)) == {170: True, 175: False}


def test_resume_may_add_but_never_drop_or_change_a_clean_scope(build):
    world = build()
    reference = world.driver["g3_authority"]
    pre = json.loads(Path(world.config["phase4"]["preapproval_path"]).read_text())
    final = sorted({*pre["clean_unit_ids"], *(unit["unit_id"] for unit in phase4.clean_final_units(0))})
    grown = {**pre, "clean_unit_ids": final, "clean_inputs_sha256_by_scope": {
        "fold0_inner": pre["clean_inputs_sha256"], "fold0_final": "4" * 64}}
    assert v5._authenticated_g3(reference, SOURCE, grown)["allocations"]
    for changed in ({**pre, "clean_unit_ids": pre["clean_unit_ids"][1:]},
                    {**pre, "clean_inputs_sha256": "5" * 64},
                    {**grown, "clean_inputs_sha256_by_scope": {"fold0_final": "4" * 64}},
                    {**pre, "clean_cap_profile": {"cpu_cores": 4}}):
        with pytest.raises(v5.LaunchWindowError, match="keep the prior G3 scope"):
            v5._authenticated_g3(reference, SOURCE, changed)


@pytest.mark.parametrize("change", [{"max_total_seconds": -HOUR}, {"cost_ceiling_usd": -85},
                                    {"deadline_unix": 0.0}, {"max_total_seconds": 1}])
def test_carried_allocation_rows_need_the_pods_own_row_checks(build, change):
    world = build()
    row = json.loads(world.prior_g3.read_text())["allocations"]["s1"]
    with pytest.raises(v5.LaunchWindowError, match="carried-forward G3 allocation is malformed"):
        v5._allocation_totals({"s1": {**row, **change}})
    assert v5._allocation_totals({"s1": row})["cost"] == 85


@pytest.mark.parametrize("problem", ["tampered", "omits_admitted", "foreign_request", "scope", "missing"])
def test_tampered_or_incomplete_prior_g3_is_refused(build, problem):
    world = build()
    prior = json.loads(world.prior_g3.read_text())
    if problem == "tampered":
        world.prior_g3.write_text(json.dumps({**prior, "ceilings": {**prior["ceilings"], "cost_usd": 10 ** 6}}))
        match = "prior G3 authority changed"
    elif problem == "omits_admitted":
        # A self-consistent G3 and driver record that drop the admitted session are still refused.
        world.prior_g3.write_text(json.dumps({**prior, "allocations": {"other": prior["allocations"]["s1"]}}))
        world.driver["g3_authority"]["sha256"] = sha(world.prior_g3)
        match = "omits a session already admitted|another G3"
    elif problem == "foreign_request":
        other = world.tmp / "s1_state" / "other.json"
        other.write_bytes(world.prior_g3.read_bytes())
        world.driver["g3_authority"]["path"] = str(other)
        match = "its own state root"
    elif problem == "scope":
        world.prior_g3.write_text(json.dumps({**prior, "hot_work_root": "/elsewhere"}))
        world.driver["g3_authority"]["sha256"] = sha(world.prior_g3)
        match = "keep the prior G3 scope"
    else:
        del world.driver["g3_authority"]
        match = "no authenticated G3"
    world.rebind()
    with pytest.raises(v5.LaunchWindowError, match=match):
        restore(world)
    assert not world.transport.uploads and not (world.root / "execution_state.json").exists()


def test_preapproval_still_cannot_bring_its_own_allocations(build):
    world = build()
    pre = json.loads(Path(world.config["phase4"]["preapproval_path"]).read_text())
    with pytest.raises(v5.LaunchWindowError, match="fabricate"):
        v5.materialize_g3({**pre, "allocations": {}}, preapproval_sha256="1" * 64, freeze1c=world.freeze,
                          out=world.tmp / "g3.json", source=SOURCE, session=world.s2, gross_ceiling_usd=85,
                          prior=world.driver["g3_authority"])


def test_retried_embedding_unit_moves_inner_clean_fits_to_the_retry_farm(build, monkeypatch):
    world = build()
    state = restore(world)
    request = prepare(world, state)
    template = request["remote_request_template"]
    farm = lambda unit, attempt: str(PurePosixPath(world.root) / "runs" / unit / f"attempt{attempt}" / "farm")
    assert (template["base_farm"], template["masked_farm"]) == (farm(BASE, 1), farm(MASKED, 2))
    # Dispatch the eight inner CLEAN fits through the real remote queue on an owned session.
    owned = {**world.s2, "provider": "runpod", "runpod_config_sha256": world.transport.stop_controller.config_sha}
    request.update(owned, remote_runner="/pod/repo/scripts/run_ec_clean30_phase4.py",
                   clean_forecasts={unit["unit_id"]: 100 for unit in phase4.clean_units()},
                   clean_timeouts={unit["unit_id"]: 90 for unit in phase4.clean_units()})
    template.update(owned, device="cuda", expected_chain=SOURCE["amendment_chain"], amendment_chain_receipt="/receipt")
    monkeypatch.setattr(phase3, "verify_execution_source", lambda *args, **kwargs: SOURCE)
    monkeypatch.setattr(phase4, "verify_authority", lambda *args, **kwargs: None)
    monkeypatch.setattr(phase4, "_attach_session", lambda *args, **kwargs: None)
    monkeypatch.setattr(workstation, "HostPull", lambda *args, **kwargs: SimpleNamespace(sync_pending=lambda root: None))
    dispatched = []

    def complete(transport, host, current, **kwargs):
        dispatched.append(current)
        return {"status": "passed", "unit_id": current["unit_id"], **IDENTITY}

    monkeypatch.setattr(workstation, "complete_remote_unit", complete)
    world.transport.deadline_unix, world.transport.reserve_seconds = None, 30
    phase4.run_clean_remote_queue(request, transport=world.transport, check_session=lambda: None, close_session=False)
    assert [row["unit_id"] for row in dispatched] == [unit["unit_id"] for unit in phase4.clean_units()]
    # The real CLEAN worker reads its farms from the dispatched request: the retry's masked farm.
    copied = []

    class Stop(Exception):
        pass

    def copy_farm(source, target, labels, roots):
        copied.append(str(source))
        if len(copied) == 2:
            raise Stop

    current = dispatched[0]
    hot = Path(current["hot_work_root"]) / current["session_id"] / current["unit_id"] / "attempt1"
    monkeypatch.setattr(phase4, "_runtime", lambda request, **kwargs: (SOURCE, SimpleNamespace(fold=0), None,
                                                                        {"hot_work_root": request["hot_work_root"]}, None))
    monkeypatch.setattr(phase4, "_clean_inputs", lambda *args: (world.tmp / "inputs", {"units": [
        {"unit_id": current["unit_id"], "allowlist": None, "exclusions": "x", "queries": "q", "masked_manifest": "m"}]}))
    monkeypatch.setattr(phase4.hot_storage, "worker_hot_attempt", lambda marker, **identity: hot)
    monkeypatch.setattr(phase4, "check_hot_storage", lambda *args, **kwargs: None)
    monkeypatch.setattr(phase4, "_clean_roots", lambda view, request: [])
    monkeypatch.setattr(phase4.clean, "read_official_training_rows", lambda *args: ([SimpleNamespace(accession="P1")], None))
    monkeypatch.setattr(phase4.clean, "read_inner_exclusions", lambda *args: ({}, None))
    monkeypatch.setattr(phase4.clean, "read_accession_allowlist", lambda *args: (["Q1"], None))
    monkeypatch.setattr(phase4.clean, "read_masked_manifest", lambda *args: ([{"label": "P1_masked"}], None))
    monkeypatch.setattr(phase4, "_copy_farm", copy_farm)
    with pytest.raises(Stop):
        phase4.clean_worker({**current, "distance_map_bytes": 1, "working_bytes": 1})
    assert copied == [farm(BASE, 1), farm(MASKED, 2)]


@pytest.mark.parametrize("batch_files,batches", [(v5.RESTORE_BATCH_FILES, 1), (2000, 3)])
def test_thousands_of_preserved_files_restore_in_bounded_batches(build, monkeypatch, batch_files, batches):
    monkeypatch.setattr(v5, "RESTORE_BATCH_FILES", batch_files)
    files = {f"runs/ecc30timing_bulk/cache/{index:05d}.pt": f"payload {index}\n" for index in range(5000)}
    world = build(g2_files=files, phase4_started=False)
    state = restore(world)
    assert len(world.transport.uploads) == len(world.transport.commands) == batches
    assert world.checks == ["check"] * (2 * batches)
    restored = sorted((world.remote / "runs/fold0/g2/runs/ecc30timing_bulk/cache").iterdir())
    assert len(restored) == 5000 and restored[1234].read_text() == "payload 1234\n"
    assert state["resume_g3"] is None and state["resume_restored"] == world.config["resume"]["preservation_sha256"]
    # Staging and the local archives are removed; a repeated call restores nothing again.
    assert not list((world.remote / "g2_requests/resume_restore").iterdir())
    assert not list((Path(world.config["state_root"]) / "resume_restore").iterdir())
    restore_again = {**state}
    v5.restore_session(world.config, world.transport, restore_again, world.state_path, SOURCE, lambda: None)
    assert len(world.transport.uploads) == batches


def test_restore_never_replaces_a_mismatching_existing_remote_file(build):
    files = {f"runs/ecc30timing_bulk/cache/{index:05d}.pt": f"payload {index}\n" for index in range(5000)}
    world = build(g2_files=files, phase4_started=False)
    existing = world.remote / "runs/fold0/g2/runs/ecc30timing_bulk/cache/00042.pt"
    write(existing, "different bytes\n")
    with pytest.raises(RuntimeError, match="never replaced"):
        restore(world)
    assert existing.read_text() == "different bytes\n"
    assert [path.name for path in existing.parent.iterdir()] == ["00042.pt"]
    assert not list((world.remote / "g2_requests/resume_restore").iterdir())
    state = json.loads(world.state_path.read_text())
    assert "resume_restored" not in state and len(world.transport.uploads) == 1


def test_matching_existing_remote_file_is_kept_and_archives_are_deterministic(build, tmp_path):
    world = build()
    kept = world.remote / "runs/fold0/g2/runs/ecc30timing_kept/g2_complete.json"
    write(kept, json.dumps({"kept": True}))
    before = kept.stat().st_mtime_ns
    restore(world)
    assert kept.stat().st_mtime_ns == before
    rows = sorted(world.p4_primary.rglob("*.json"))
    batch = [(str(path.relative_to(world.p4_primary)), {"local": str(path), "sha256": sha(path)}) for path in rows]
    first, second = tmp_path / "first.tar", tmp_path / "second.tar"
    assert v5._restore_archive(first, batch) == v5._restore_archive(second, batch)
    assert first.read_bytes() == second.read_bytes()
    with pytest.raises(v5.LaunchWindowError, match="changed before it was archived"):
        v5._restore_archive(tmp_path / "third.tar", [(batch[0][0], {**batch[0][1], "sha256": "0" * 64})])


class MountedTransport(Transport):
    """The planned pod layout: a rename between the container disk and the campaign volume fails with EXDEV."""

    def __init__(self, disk):
        super().__init__()
        self.prelude = ("import errno,os\n"
                        f"DISK={str(disk)!r}\n"
                        "def _on_disk(path):\n"
                        "    path=os.path.realpath(os.path.dirname(os.path.abspath(path)))\n"
                        "    return path==DISK or path.startswith(DISK+os.sep)\n"
                        "def _mounted(move):\n"
                        "    def checked(source,target,*args,**kwargs):\n"
                        "        if _on_disk(source)!=_on_disk(target):\n"
                        "            raise OSError(errno.EXDEV,os.strerror(errno.EXDEV))\n"
                        "        return move(source,target,*args,**kwargs)\n"
                        "    return checked\n"
                        "os.rename,os.replace=_mounted(os.rename),_mounted(os.replace)\n")

    def remote(self, argv, *, timeout):
        assert argv[:3] == [sys.executable, "-c", v5.RESTORE_PROGRAM] and timeout > 0
        self.commands.append(argv)
        result = subprocess.run([argv[0], "-c", self.prelude + argv[2], *argv[3:]], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr[-2000:])
        return result.stdout


def posix_pod(tmp_path):
    """<campaign>/views and <campaign>/g2_requests as runpod_bootstrap.prepare_posix_storage links them."""
    campaign, disk = tmp_path / "workspace" / "campaign", tmp_path / "container_disk" / "deepmzyme"
    campaign.mkdir(parents=True)
    for name, target in (("views", "views"), ("g2_requests", "requests")):
        (disk / target).mkdir(parents=True)
        (campaign / name).symlink_to(disk / target, target_is_directory=True)
    config = {"remote_campaign": str(campaign), "state_root": str(tmp_path / "state"),
              "bootstrap_python": sys.executable, "runpod": {"posix_root": str(disk)}}
    return campaign, disk, config


def preserved_rows(tmp_path, campaign, files):
    rows = []
    for relative, data in files.items():
        local = write(tmp_path / "preserved" / relative, data)
        rows.append({"local": str(local), "remote": str(campaign / relative), "sha256": sha(local)})
    return rows


def test_restore_through_posix_storage_aliases_writes_on_each_targets_filesystem(tmp_path):
    campaign, disk, config = posix_pod(tmp_path)
    files = {"runs/fold0/g2/runs/ecc30timing_kept/g2_complete.json": "{}\n",
             "views/fold0/whole_protein_pretrain/seq_means_final/x.pt": "pt\n"}
    transport = MountedTransport(tmp_path / "container_disk")
    # The emulated mount boundary is real for a move between the archive's disk and the volume.
    source, target = write(disk / "requests" / "probe", "x"), campaign / "probe"
    probe = subprocess.run([sys.executable, "-c", transport.prelude + "import sys\nos.replace(sys.argv[1],sys.argv[2])",
                            str(source), str(target)], capture_output=True, text=True)
    assert probe.returncode and "Invalid cross-device link" in probe.stderr and not target.exists()
    source.unlink()
    v5._restore_files(config, preserved_rows(tmp_path, campaign, files), transport, lambda: None)
    for relative, data in files.items():
        assert (campaign / relative).read_text() == data
    assert (disk / "views/fold0/whole_protein_pretrain/seq_means_final/x.pt").read_text() == "pt\n"
    # The archive crossed the request alias to the container disk and was removed there; no partial is left.
    assert transport.uploads[0].parent == campaign / "g2_requests/resume_restore"
    assert not list((disk / "requests/resume_restore").iterdir())
    assert not [path for root in (campaign, disk) for path in root.rglob("*") if "restore-partial" in path.name]


@pytest.mark.parametrize("problem", ["foreign_alias_target", "escaping_link"])
def test_restore_refuses_any_other_link_before_any_write(tmp_path, problem):
    campaign, disk, config = posix_pod(tmp_path)
    files = {"runs/fold0/g2/runs/ecc30timing_kept/g2_complete.json": "{}\n",
             "views/fold0/whole_protein_pretrain/seq_means_final/x.pt": "pt\n"}
    outside = tmp_path / "outside"
    outside.mkdir()
    if problem == "foreign_alias_target":
        config["runpod"]["posix_root"] = str(tmp_path / "elsewhere")
        match = "alias differs from its declared target"
    else:
        (campaign / "runs").symlink_to(outside, target_is_directory=True)
        match = "escapes the campaign root"
    with pytest.raises(RuntimeError, match=match):
        v5._restore_files(config, preserved_rows(tmp_path, campaign, files), MountedTransport(disk.parent),
                          lambda: None)
    written = [path for root in (outside, disk) for path in root.rglob("*") if path.is_file()]
    assert written == [] and not (campaign / "runs").resolve().joinpath("fold0").exists()


def test_restore_targets_keep_the_restricted_path_guard(tmp_path):
    campaign, disk, config = posix_pod(tmp_path)
    rows = preserved_rows(tmp_path, campaign, {"runs/fold0/g2/identity/fold0/components.tsv": "x\n"})
    transport = MountedTransport(disk.parent)
    with pytest.raises(phase3.Phase3Error, match="Restricted identity"):
        v5._restore_files(config, rows, transport, lambda: None)
    assert not transport.uploads and not transport.commands
