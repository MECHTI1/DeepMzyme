"""Offline v5 orchestration contracts with fake workers and provider transport."""
import hashlib
import json
import copy
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from ec_clean30 import campaign, workstation as ws  # noqa: E402


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


@pytest.fixture
def case(tmp_path, monkeypatch):
    chain = [sha("first"), sha("second")]
    source = {"format": ws.phase3.EXECUTION_V5_FORMAT, "amendment_chain": chain, "fit_chain": chain,
              "scientific_identity_sha256": sha("science"), "execution_manifest_sha256": sha("execution")}
    state_root = tmp_path / "state"
    state_root.mkdir()
    for name in ("main_model_probe.json", "esm1b_model_probe.json"):
        (state_root / name).write_text("{}")
    config = {"provider": "runpod", "execution_scope": "g2_then_fold0_v5", "hardware_profile": "runpod_rtx4090",
              "state_root": str(state_root), "remote_campaign": "/workspace/campaign", "remote_repo": "/workspace/repo",
              "remote_python": "/runtime/python", "durable_root": str(tmp_path / "durable"),
              "mirror_root": str(tmp_path / "mirror"), "execution_manifest": str(tmp_path / "manifest.json"),
              "g2_max_download_bytes": 2000000, "timing_inputs": "/inputs/timing",
              "profiles": {key: {"cap_prefix": ["cap", "--"], "memory_limit_bytes": 100000}
                           for key in ("fit", "esm1b")}, "embedding": {"esmc": {}, "esm1b": {}},
              "lane_limits": {"cpu_percent": 600, "ram_bytes": 24000, "vram_bytes": 21000}}
    session = {"pod_id": "owned-pod", "session_id": "owned-session", "allocation_started_unix": time.time(),
               "deadline_unix": time.time() + 50000, "max_total_seconds": 50000}
    recipes = tuple((f"science_recipe_{index}", object()) for index in range(7))
    monkeypatch.setattr(campaign, "timing_recipes_v2", lambda: recipes, raising=False)
    config["bounds"] = {name: {"forecast_seconds": 100, "unit_timeout_seconds": 90}
                        for _, name in ws.measurement_queue(source)}
    options = {key: f"/inputs/{key}" for key in ("union_fasta", "corpus_table", "holdout_receipt",
               "official_split_dir", "identity_work", "hf_home", "esmc_weights")}
    options.update(cpu_cap_prefix=["cpu-cap", "--"], forecast_seconds=100, unit_timeout_seconds=90)
    transport = SimpleNamespace(stop_controller=SimpleNamespace(config_sha=sha("config")),
                                read_json=lambda path: {"wall_seconds": 10})
    state = {"completed": []}
    events = []
    monkeypatch.setattr(ws.phase3, "require_record_precedence",
                        lambda record, current: ws.phase3.require_same_scientific_identity(record, current))

    def complete(transport, host, request, **kwargs):
        events.append((kwargs["action"], dict(request)))
        record = {"status": "passed", **ws.phase3.source_identity(source), "request": request}
        local = Path(config["durable_root"]) / kwargs["completed_relative"]
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(json.dumps(record))
        return record

    monkeypatch.setattr(ws, "complete_remote_unit", complete)
    return SimpleNamespace(config=config, source=source, session=session, options=options, transport=transport,
                           state=state, events=events, state_path=state_root / "driver_state.json")


def prepare(case):
    def callback(config, transport, *_args):
        assert transport is None
        return {"pretrain_request": case.options, "max_download_bytes": config["g2_max_download_bytes"]}
    ws._prepare_g2_v5(case.config, case.transport, case.state, case.state_path, case.session,
                      {"source": case.source}, "/repo/manifest.json", "/probe.json", lambda: case.session, callback)


def test_science_recipes_supply_v5_queue_without_changing_historical_queue(case):
    queue = ws.measurement_queue(case.source)
    assert [name for action, name in queue[:7]] == [name for name, _ in campaign.timing_recipes_v2()]
    assert queue[7:] == (("timing-unit", "lane_serial_1"), ("timing-unit", "lane_serial_2"),
                         ("lane-gate", "lane_batch"), ("embed-timing", "esmc"), ("embed-timing", "esm1b"))
    assert ws.measurement_queue({"format": "historical"}) == ws.QUEUE


def test_preparation_is_preserved_before_timing_and_passed_to_every_unit(case):
    prepare(case)
    evidence = case.state["pretrain_preparation"]
    assert case.events[0][0] == "pretrain-prepare"
    assert Path(evidence["local_path"]).is_file() and evidence["record"]["status"] == "passed"
    assert evidence["sha256"] == ws._sha(Path(evidence["local_path"]))
    ws._execute_queue(case.config, case.transport, case.state, case.state_path, case.session,
                       {"source": case.source}, "/repo/manifest.json", "/probe.json", lambda: case.session)
    assert case.state["completed"] == [name for _, name in ws.measurement_queue(case.source)]
    assert len(case.state["g2_evidence"]) == len(ws.measurement_queue(case.source))
    assert all(request["pretrain_evidence"] == evidence["remote_path"] for _, request in case.events[1:])
    assert all(request["expected_chain"] == case.source["amendment_chain"] for _, request in case.events)
    assert all(request["plan_version"] == 2 for _, request in case.events)


def test_no_timing_before_preparation(case):
    with pytest.raises(ws.WorkstationError, match="completed cache"):
        ws._execute_queue(case.config, case.transport, case.state, case.state_path, case.session,
                           {"source": case.source}, "/repo/manifest.json", "/probe.json", lambda: case.session)
    assert not case.events


@pytest.mark.parametrize("key,value", [("device", "cpu"), ("execution_manifest", "/other.json"),
                                       ("preparation", True)])
def test_preparation_hook_cannot_override_runtime_authority(case, key, value):
    case.options[key] = value
    with pytest.raises(ws.WorkstationError, match="only its bound F6 inputs"):
        prepare(case)
    assert not case.events


def test_preparation_identity_is_immutable_on_recovery(case):
    prepare(case)
    case.options["corpus_table"] = "/inputs/changed"
    with pytest.raises(ws.WorkstationError, match="changed during recovery"):
        prepare(case)
    assert len(case.events) == 1


def test_embedding_options_cannot_replace_preparation_evidence(case):
    prepare(case)
    case.config["embedding"]["esmc"]["pretrain_evidence"] = "/foreign"
    with pytest.raises(ws.WorkstationError, match="cannot override session"):
        ws._execute_queue(case.config, case.transport, case.state, case.state_path, case.session,
                           {"source": case.source}, "/repo/manifest.json", "/probe.json", lambda: case.session)
    assert not any(action == "embed-timing" for action, _ in case.events)


def test_freeze1c_is_created_from_completed_queue_and_staged_once(case, tmp_path, monkeypatch):
    prepare(case)
    ws._execute_queue(case.config, case.transport, case.state, case.state_path, case.session,
                       {"source": case.source}, "/repo/manifest.json", "/probe.json", lambda: case.session)
    local = tmp_path / "freeze1c.json"
    local.write_text(json.dumps({"format": "deepmzyme_ec_clean30_freeze1c_v2", **ws.phase3.source_identity(case.source)}))
    stages = []
    monkeypatch.setattr(ws, "_stage_file", lambda config, transport, item: stages.append(item))

    def finalize(config, transport, state, *_args):
        assert transport is None and state["g2_evidence"] and state["pretrain_preparation"]
        return {"local": str(local), "remote": "/workspace/campaign/g2_requests/freeze1c_v2.json", "sha256": ws._sha(local)}

    ws._complete_g2_v5(case.config, case.transport, case.state, case.state_path, case.session,
                        {"source": case.source}, "/repo/manifest.json", "/probe.json", lambda: case.session, finalize)
    assert stages == [{**case.state["freeze1c"], "kind": "file"}]
    case.state["completed"].pop()
    with pytest.raises(ws.WorkstationError, match="complete authenticated"):
        ws._complete_g2_v5(case.config, case.transport, case.state, case.state_path, case.session,
                            {"source": case.source}, "/repo/manifest.json", "/probe.json", lambda: case.session, finalize)
    assert len(stages) == 1


def test_runtime_locks_are_not_skipped_for_v5(case):
    with pytest.raises(ws.WorkstationError, match="bind its complete published runtime lock"):
        ws._validate_published_runtime_locks(case.config, case.source)


def test_runtime_lock_binding_checks_actual_bytes(case, tmp_path, monkeypatch):
    monkeypatch.setattr(ws.phase3, "REPO_ROOT", tmp_path)
    lock = tmp_path / "locks.json"
    lock.write_text("{}")
    with pytest.raises(ws.WorkstationError, match="missing or changed"):
        ws._validate_published_runtime_locks(case.config, {**case.source,
            "runtime_locks": {"path": "locks.json", "sha256": sha("foreign")}})


@pytest.mark.parametrize("options", [[], ["--authorization", "/approval"], ["--window-id", "window"]])
def test_explicit_v5_cli_requires_both_authority_arguments(tmp_path, monkeypatch, options):
    monkeypatch.setattr(ws, "_json", lambda *_: pytest.fail("config read before complete authority arguments"))
    assert ws.main(["--config", str(tmp_path / "config.json"), "--run-v5", *options]) == 2


def test_authority_arguments_cannot_enable_legacy_dispatch(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "run_g2", lambda *_: pytest.fail("legacy launcher reached"))
    assert ws.main(["--config", str(tmp_path / "config.json"), "--run", "--authorization", "/approval",
                    "--window-id", "window"]) == 2
    with pytest.raises(SystemExit):
        ws.main(["--config", str(tmp_path / "config.json"), "--run", "--run-v5"])


def test_v5_cli_calls_only_explicit_authorized_dispatcher(tmp_path, monkeypatch):
    from ec_clean30 import execution_v5

    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    calls = []
    monkeypatch.setattr(execution_v5, "run_v5", lambda *args, **kwargs: calls.append((args, kwargs)) or {"status": "fixture"})
    monkeypatch.setattr(ws.signal, "signal", lambda *_: None)
    monkeypatch.setattr(ws, "run_g2", lambda *_: pytest.fail("legacy launcher reached"))
    assert ws.main(["--config", str(config_path), "--run-v5", "--authorization", "/approval",
                    "--window-id", "window"]) == 0
    assert calls == [(({}, config_path), {"authorization_path": Path("/approval"), "window_id": "window"})]


def test_closeout_reuses_each_roots_preservation_budget(case, monkeypatch):
    calls = []

    def preserve(config, transport, controller, session, **kwargs):
        calls.append(kwargs)
        out = Path(config["state_root"]) / kwargs["receipt_name"]
        out.write_text("{}")
        return out

    monkeypatch.setattr(ws, "_preserve_runpod_outputs", preserve)
    phase4_state = str(Path(case.config["state_root"]) / "phase4")
    state = {"g2_max_download_bytes": 2000000,
             "phase4": {"preservation_roots": [{"execution_root": "/g2"}, {"execution_root": "/phase4"}],
                        "max_download_bytes": 3000000, "host_state_root": phase4_state}}
    ws._preserve_runpod_session_outputs(case.config, case.transport, case.transport.stop_controller,
                                        case.session, state)
    assert [(row["max_download_bytes"], row["host_state_root"]) for row in calls] == [
        (2000000, None), (3000000, phase4_state)]


@pytest.fixture
def reviewed_phase4(case, tmp_path, monkeypatch):
    from ec_clean30 import container_caps, phase4

    plan = {"format": "deepmzyme_ec_clean30_fold_plan_v2", "fold": 0, "plan_sha256": sha("canonical plan")}
    monkeypatch.setattr(campaign, "build_fold_plan_v2", lambda fold: plan, raising=False)
    monkeypatch.setattr(campaign, "fold_units_v2", lambda plan, step: (
        SimpleNamespace(family="F2"), SimpleNamespace(family="F6")), raising=False)
    local = tmp_path / "plan.json"
    local.write_text(json.dumps(plan))
    config = case.config
    config.update(cgroup_root="/sys/fs/cgroup/owned", runpod={"max_runtime_seconds": 50000})
    config["embedding"]["esm1b"]["clean_python"] = "/runtime-clean/bin/python"
    config["remote_python"] = "/runtime-fit/bin/python"
    root = "/workspace/campaign/runs/fold0/phase4"
    profile = {"cgroup_root": config["cgroup_root"], "cpu_cores": 2,
               "memory_bytes": 3 * ws.GIB, "distance_map_memory_bytes": 4500000000}
    science = {key: value for key, value in profile.items() if key != "distance_map_memory_bytes"}
    template = {"plan": "/workspace/campaign/inputs/plan.json", "view": "/workspace/campaign/views/fold0",
                "campaign_root": "/workspace/campaign", "execution_root": root,
                "durable_root": str(tmp_path / "phase4-durable"), "python": config["remote_python"],
                "clean_python": config["embedding"]["esm1b"]["clean_python"],
                "embedding_python": config["embedding"]["esm1b"]["clean_python"],
                "device": "cuda", "persistence_mode": "host_pull", "step_chain": root + "/step_chain.json",
                "fold0_step_chain": root + "/step_chain.json", "science_cap_profile": science,
                "clean_cap_profile": profile, "memory_limit_bytes": profile["memory_bytes"], "threads": 2,
                "hot_work_root": "/workspace/clean-hot", "working_bytes": 2 * ws.GIB,
                "distance_map_bytes": 1000000000, "weights_sha256": sha("weights"),
                "contact_weights_sha256": sha("contact")}
    for name in ("clean_inputs", "base_farm", "masked_farm", "clean_clone", "training_script", "distance_helper",
                 "export_script", "harness", "extract_script", "weights"):
        template[name] = "/workspace/campaign/inputs/" + name
    for kind in ("base", "masked"):
        template[kind + "_farm"] = root + f"/runs/ecc30_clean_embed_{kind}/attempt1/farm"
    for key, memory in (("cap_prefix", "memory_bytes"), ("distance_cap_prefix", "distance_map_memory_bytes")):
        template[key] = container_caps.cap_prefix(config["remote_python"], config["remote_repo"], config["cgroup_root"],
                                                 cpu_cores=2, memory_bytes=profile[memory])
    template["scientific_inputs"] = {"components": "/workspace/campaign/inputs/components.tsv",
        "clean_sanity_record": root + "/sanity.json", "clean_sanity_dispositions": [],
        "clean_sanity_verify_all": root + "/sanity_all.json", "futility_out": root + "/futility.json",
        "f6_corpus": "/workspace/campaign/inputs/corpus.tsv",
        "r1_inner_lists": [f"/workspace/campaign/inputs/r1_inner_{index}.txt" for index in range(4)]}
    ids = {unit["unit_id"] for unit in phase4.clean_units()}
    request = {"fold": 0, "local_plan": str(local), "execution_manifest": config["execution_manifest"],
               "remote_runner": "/workspace/repo/scripts/run_ec_clean30_phase4.py", **ws.phase3.source_identity(case.source),
               "durable_root": str(tmp_path / "phase4-durable"), "mirror_root": str(tmp_path / "phase4-mirror"),
               "host_state_root": str(tmp_path / "phase4-state"), "max_download_bytes": 3000000,
               "remote_request_template": template, "queue_actions": ["clean_embeddings", "clean_fits", "s1"],
               "refit_forecast_seconds": 100, "refit_timeout_seconds": 90,
               "stage_files": [{"local": str(local), "remote": template["plan"], "sha256": ws._sha(local)}]}
    for group, names in {"family": {"F2", "F6"}, "clean": ids, "embedding": {"base", "masked"},
                         "scientific": set(phase4.SCIENTIFIC_STEPS)}.items():
        request[group + "_forecasts"] = dict.fromkeys(names, 100)
        request[group + "_timeouts"] = dict.fromkeys(names, 90)
    preapproval = {"campaign_id": campaign.CAMPAIGN_ID, "scope": phase4.V5_SCOPE, "allowed_folds": [0],
                  "plan_sha256_by_fold": {"0": plan["plan_sha256"]}, "allowed_steps": {"0": ["S1", "S4"]},
                  "clean_unit_ids": sorted(ids), "clean_cap_profile": profile, "science_cap_profile": science,
                  "clean_inputs_sha256": sha("inputs"), "hot_work_root": template["hot_work_root"]}
    monkeypatch.setattr(ws, "_storage_facts", lambda path: (2 if "mirror" in str(path) else 1, 100 * ws.GIB))
    return SimpleNamespace(config=config, source=case.source, request=request, preapproval=preapproval)


def test_reviewed_phase4_plan_is_validated_without_creating_outputs(reviewed_phase4, tmp_path):
    case = reviewed_phase4
    before = sorted(str(path) for path in tmp_path.rglob("*"))
    result = ws.validate_phase4_plan(case.config, case.source, case.request, case.preapproval)
    assert result["status"] == "passed" and result["max_download_bytes"] == 3000000
    assert sorted(str(path) for path in tmp_path.rglob("*")) == before


@pytest.mark.parametrize("change,match", [
    (lambda r, p: r.pop("remote_runner"), "transport request is incomplete"),
    (lambda r, p: r["family_forecasts"].pop("F6"), "exact canonical queue"),
    (lambda r, p: r["scientific_timeouts"].update(futility=101), "timeout exceeds"),
    (lambda r, p: r["embedding_forecasts"].update(base=float("nan")), "finite and positive"),
    (lambda r, p: p.update(allowed_steps={"0": ["S1"]}), "exactly the reviewed"),
    (lambda r, p: r.update(queue_requests={"s1": {"attempts": {}}}), "without queue overrides"),
    (lambda r, p: r["remote_request_template"].update(view="/foreign/view"), "owned runtime"),
    (lambda r, p: r["remote_request_template"].update(base_farm="/foreign/farm"), "generated outputs"),
    (lambda r, p: r.update(attempts={"ecc30_clean_embed_base": 2}), "generated outputs"),
    (lambda r, p: r["remote_request_template"].update(clean_scope="final"), "inner closure"),
    (lambda r, p: r["remote_request_template"]["scientific_inputs"].pop("f6_corpus"), "scientific inputs"),
    (lambda r, p: r["remote_request_template"]["scientific_inputs"].update(futility_out="/foreign/out.json"), "Scientific outputs"),
    (lambda r, p: r["remote_request_template"].update(cap_prefix=["uncapped"]), "kernel resource profile"),
    (lambda r, p: r["remote_request_template"].update(hot_work_root="/workspace/repo"), "working-space forecasts"),
    (lambda r, p: r["stage_files"][0].update(sha256=sha("changed")), "staged file is missing, changed"),
    (lambda r, p: r.update(max_download_bytes=0), "preservation-byte ceiling"),
    (lambda r, p: r.update(mirror_root=r["durable_root"]), "roots cannot overlap"),
])
def test_incomplete_reviewed_phase4_refuses_before_g2(reviewed_phase4, change, match):
    case = reviewed_phase4
    request, preapproval = copy.deepcopy(case.request), copy.deepcopy(case.preapproval)
    change(request, preapproval)
    with pytest.raises(ws.WorkstationError, match=match):
        ws.validate_phase4_plan(case.config, case.source, request, preapproval)


def test_phase4_capacity_includes_g2_bytes_on_the_same_device(reviewed_phase4, monkeypatch):
    case = reviewed_phase4
    monkeypatch.setattr(ws, "_storage_facts", lambda path: (2 if "mirror" in str(path) else 1,
                        5 * ws.GIB + case.request["max_download_bytes"]))
    with pytest.raises(ws.WorkstationError, match="combined G2 and Phase-4 allowances"):
        ws.validate_phase4_plan(case.config, case.source, case.request, case.preapproval)


def test_phase4_mirror_must_be_a_different_device(reviewed_phase4, monkeypatch):
    case = reviewed_phase4
    monkeypatch.setattr(ws, "_storage_facts", lambda path: (1, 100 * ws.GIB))
    with pytest.raises(ws.WorkstationError, match="independent primary and mirror"):
        ws.validate_phase4_plan(case.config, case.source, case.request, case.preapproval)


def test_final_clean_needs_separate_bound_inputs_and_farms(reviewed_phase4, monkeypatch):
    from ec_clean30 import phase4

    case = reviewed_phase4
    final_ids = {unit["unit_id"] for unit in phase4.clean_final_units(0)}
    case.request["run_final_clean"] = True
    case.preapproval["clean_unit_ids"] = sorted(set(case.preapproval["clean_unit_ids"]) | final_ids)
    case.preapproval["clean_inputs_sha256_by_scope"] = {"fold0_inner": sha("inputs"), "fold0_final": sha("final")}
    case.preapproval.pop("clean_inputs_sha256")
    with pytest.raises(ws.WorkstationError, match="separately reviewed input closure"):
        ws.validate_phase4_plan(case.config, case.source, case.request, case.preapproval)
    case.request["final_clean_request"] = {"clean_scope": "final", "clean_inputs": "/workspace/campaign/inputs/final",
        "clean_forecasts": dict.fromkeys(final_ids, 100), "clean_timeouts": dict.fromkeys(final_ids, 90),
        "farms_by_unit": phase4.final_farms_by_unit({**case.request, "clean_scope": "final"}),
        "embedding_forecasts": {unit["unit_id"]: 100 for unit in phase4.embedding_units({**case.request, "clean_scope": "final"})},
        "embedding_timeouts": {unit["unit_id"]: 90 for unit in phase4.embedding_units({**case.request, "clean_scope": "final"})}}
    case.request["remote_request_template"].update(hf_home="/cache", esmc_weights="/cache/weights")
    assert ws.validate_phase4_plan(case.config, case.source, case.request, case.preapproval)["status"] == "passed"
    case.request["final_clean_request"]["farms_by_unit"].pop(next(iter(final_ids)))
    with pytest.raises(ws.WorkstationError, match="per-seed producer"):
        ws.validate_phase4_plan(case.config, case.source, case.request, case.preapproval)


@pytest.mark.parametrize("marker", ["pretrain_attachment.json", "scientific_outputs_pending.json"])
def test_pending_external_outputs_block_cleanup_inventory(case, tmp_path, monkeypatch, marker):
    root = tmp_path / "remote-output"
    root.mkdir()
    (root / marker).write_text(json.dumps({"status": "pending"}))
    case.config["bootstrap_python"] = sys.executable

    def remote(argv, **kwargs):
        monkeypatch.setattr(sys, "argv", ["-c", str(root)])
        exec(compile(argv[2], "<remote-inventory>", "exec"), {})
        pytest.fail("Pending external scientific output did not block cleanup")

    case.transport.remote = remote
    with pytest.raises(AssertionError, match="Pending .* data blocks cleanup"):
        ws._preserve_runpod_outputs(case.config, case.transport, case.transport.stop_controller, case.session)


@pytest.mark.parametrize("fold", [1, 2, 3, 4])
def test_later_fold_request_uses_its_own_plan_and_retains_fold0_chain(reviewed_phase4, monkeypatch, fold):
    case = reviewed_phase4
    request = json.loads(json.dumps(case.request).replace("/fold0/", f"/fold{fold}/").replace("/views/fold0", f"/views/fold{fold}"))
    request["fold"] = fold
    template = request["remote_request_template"]
    template["fold0_step_chain"] = "/workspace/campaign/runs/fold0/phase4/step_chain.json"
    template.update({key: "/workspace/campaign/inputs/" + key for key in
                     ("corpus_table", "holdout_receipt", "identity_work", "official_split_dir")})
    plan = {"format": "deepmzyme_ec_clean30_fold_plan_v2", "fold": fold, "plan_sha256": sha(f"plan{fold}")}
    Path(request["local_plan"]).write_text(json.dumps(plan))
    request["stage_files"][0]["sha256"] = ws._sha(Path(request["local_plan"]))
    monkeypatch.setattr(campaign, "build_fold_plan_v2", lambda fold: plan)
    for group in ("clean", "embedding"):
        request[group + "_forecasts"] = {}
        request[group + "_timeouts"] = {}
    for suffix in ("forecasts", "timeouts"):
        request["scientific_" + suffix] = {key: value for key, value in request["scientific_" + suffix].items()
                                          if key in {"select_s1", "select_s2", "select_s4"}}
    request.update(attachment_forecast_seconds=100, attachment_timeout_seconds=90)
    case.config.update(execution_scope="fold_execution_v5", execution_fold=fold, resume={"bound": True})
    preapproval = {**case.preapproval, "allowed_folds": [fold], "plan_sha256_by_fold": {str(fold): plan["plan_sha256"]},
                   "allowed_steps": {str(fold): ["S1", "S4"]}, "clean_unit_ids": [], "clean_inputs_sha256_by_scope": {}}
    assert ws.validate_phase4_plan(case.config, case.source, request, preapproval)["status"] == "passed"
