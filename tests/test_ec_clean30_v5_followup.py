"""CPU regression checks for record routing, seed farms and preserved-unit recovery."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ec_clean30 import execution_v5, phase3, phase4


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def science_case(tmp_path, monkeypatch):
    held = write(tmp_path / "held.json", {"held": True})
    fits = [write(tmp_path / f"fit{i}.json", {"fit": i}) for i in range(2)]
    dispositions = [write(tmp_path / f"d{i}.json", {"record_sha256": phase3._sha(path)})
                    for i, path in enumerate([held, *fits])]
    inputs = {"clean_sanity_record": str(held), "clean_sanity_dispositions": [str(p) for p in dispositions],
              "clean_sanity_verify_all": str(tmp_path / "verified.json"), "components": "components",
              "f6_corpus": "corpus", "futility_out": str(tmp_path / "futility.json"),
              "r1_inner_lists": [f"list{i}" for i in range(4)]}
    monkeypatch.setattr(phase4, "_science_paths", lambda *a: inputs)
    monkeypatch.setattr(phase4, "_completed_fit_inputs", lambda *a: ["runs"])
    monkeypatch.setattr(phase4, "_completed_clean_inputs", lambda *a: ({}, [str(p) for p in fits], {}))
    request = {"python": "python", "plan": "plan", "step_chain": str(tmp_path / "chain"), "campaign_root": str(tmp_path)}
    source = {"freeze1_sha256": "a" * 64, "amendment_path": "link", "amendment_chain_receipt": "receipt",
              "amendment_chain": ["a", "b"]}
    return request, source, inputs, dispositions


@pytest.mark.parametrize("step", ["select_s1", "select_s2", "select_s4", "futility"])
def test_selector_sanity_flags_only_reach_consumers(science_case, tmp_path, step):
    request, source, inputs, dispositions = science_case
    argv, _ = phase4._science_command({**request, "scientific_step": step}, source, SimpleNamespace(fold=0), tmp_path)
    consumes = step in {"select_s1", "futility"}
    assert ("--clean-sanity-record" in argv) == consumes
    assert (str(dispositions[0]) in argv) == consumes
    assert all(str(path) not in argv for path in dispositions[1:])


def test_verify_all_keeps_dispositions_for_its_whole_record_set(science_case, tmp_path):
    request, source, inputs, dispositions = science_case
    foreign = write(tmp_path / "foreign.json", {"record_sha256": "f" * 64})
    inputs["clean_sanity_dispositions"].append(str(foreign))
    argv, _ = phase4._science_command({**request, "scientific_step": "verify_sanity"}, source,
                                     SimpleNamespace(fold=0), tmp_path)
    assert all(str(path) in argv for path in dispositions)
    assert str(foreign) not in argv


def test_unit_dispositions_do_not_leak_between_fits(tmp_path):
    first, second = write(tmp_path / "first.json", {"fit": 1}), write(tmp_path / "second.json", {"fit": 2})
    one = write(tmp_path / "one.json", {"record_sha256": phase3._sha(first)})
    two = write(tmp_path / "two.json", {"record_sha256": phase3._sha(second)})
    request = {"unit_id": "first", "clean_sanity_dispositions": [str(two)],
               "clean_sanity_dispositions_by_unit": {"first": [str(one), str(two)], "second": [str(two)]}}
    assert phase4._unit_dispositions(request, first) == [one]
    assert phase4._unit_dispositions({**request, "unit_id": "second"}, first) == []


@pytest.mark.parametrize("fold", range(5))
def test_final_farms_cover_exact_frozen_seeds_and_share_only_base(fold):
    request = {"fold": fold, "clean_scope": "final", "remote_request_template": {"execution_root": f"/campaign/runs/fold{fold}/phase4"}}
    units = phase4.embedding_units(request)
    assert [u["seed"] for u in units if u["embedding"] == "esm1b" and u["kind"] == "masked"] == list(range(42, 47))
    assert [u["seed"] for u in units if u["embedding"] == "esmc" and u["kind"] == "masked"] == [42, 43, 44]
    farms = phase4.final_farms_by_unit(request)
    for unit in phase4.clean_final_units(fold):
        assigned = farms[unit["unit_id"]]
        assert f"seed{unit['seed']}" in assigned["masked_farm"]
        assert ("_esmc_" in assigned["base_farm"]) == (unit["arm"] == "c_esmc")
    assert len({row["base_farm"] for row in farms.values()}) == 2
    assert len({row["masked_farm"] for row in farms.values()}) == 8


@pytest.mark.parametrize("seed", [43, 46])
def test_masked_producer_receives_only_its_seed_and_embedding(seed):
    record = {"units": [{"seed": s, "arm": arm, "masked_manifest": f"{arm}-{s}.tsv"}
                        for s in range(42, 47) for arm in ("r1", "r2", "c_esmc")]}
    assert phase4._embedding_mask_names(record, {"seed": seed, "embedding": "esm1b"}) == [f"r1-{seed}.tsv", f"r2-{seed}.tsv"]
    assert phase4._embedding_mask_names(record, {"seed": seed, "embedding": "esmc"}) == [f"c_esmc-{seed}.tsv"]


def test_preserved_result_is_never_replaced_by_retry(tmp_path):
    root = tmp_path / "runs/unit"
    write(root / "attempt1/unit_result.json", {"status": "passed"})
    assert execution_v5.resume_attempts(tmp_path) == {"unit": 1}
    write(root / "complete.json", {"attempt": 1})
    assert execution_v5.resume_attempts(tmp_path) == {"unit": 1}
    write(root / "attempt2/unit_result.json", {"status": "passed"})
    with pytest.raises(execution_v5.LaunchWindowError, match="Multiple preserved"):
        execution_v5.resume_attempts(tmp_path)


def test_lost_unit_gets_one_identical_retry_then_stops(tmp_path):
    (tmp_path / "runs/unit/attempt1").mkdir(parents=True)
    assert execution_v5.resume_attempts(tmp_path) == {"unit": 2}
    (tmp_path / "runs/unit/attempt2").mkdir()
    with pytest.raises(execution_v5.LaunchWindowError, match="already been used"):
        execution_v5.resume_attempts(tmp_path)


def test_completed_result_missing_bytes_never_becomes_a_retry(tmp_path):
    write(tmp_path / "runs/unit/complete.json", {"attempt": 1})
    (tmp_path / "runs/unit/attempt1").mkdir()
    with pytest.raises(execution_v5.LaunchWindowError, match="lost its preserved result"):
        execution_v5.resume_attempts(tmp_path)


def test_partial_scientific_output_requires_diagnosis(tmp_path):
    (tmp_path / "runs/ecc30_science_fold0_futility/attempt1").mkdir(parents=True)
    with pytest.raises(execution_v5.LaunchWindowError, match="Partial scientific"):
        execution_v5.resume_attempts(tmp_path)


@pytest.fixture
def preserved_session(tmp_path, monkeypatch):
    identity = {"scientific_identity_sha256": "a" * 64, "fit_chain": ["b" * 64, "c" * 64]}
    source = {**identity, "format": phase3.EXECUTION_V5_FORMAT}
    manifest = write(tmp_path / "manifest.json", source)
    primary, mirror = tmp_path / "old_primary", tmp_path / "old_mirror"
    remote = tmp_path / "pod/campaign"
    for root in (primary, mirror):
        root.mkdir()
    # Device separation is synthetic; every file/hash check remains real.
    original_stat = Path.stat
    def stat(path, **kwargs):
        value = original_stat(path, **kwargs)
        if path == mirror:
            import os
            fields = list(value)
            fields[2] += 1
            return os.stat_result(fields)
        return value
    monkeypatch.setattr(Path, "stat", stat)
    artifact = write(primary / "kept.json", {"completed": True})
    files = [{"path": artifact.name, "sha256": phase3._sha(artifact), "size_bytes": artifact.stat().st_size}]
    inventory = write(primary / "closeout/remote_inventory.json", {"session_id": "old", "files": files,
                         "pending_transfer": None, "pending_unit": None})
    files = [*files, {"path": "closeout/remote_inventory.json", "sha256": phase3._sha(inventory),
                      "size_bytes": inventory.stat().st_size}]
    import shutil
    for row in files:
        dest = mirror / row["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(primary / row["path"], dest)
    proof = write(tmp_path / "preservation.json", {"format": "deepmzyme_ec_clean30_runpod_preservation_v1",
        "session_id": "old", "pod_id": "old-pod", "execution_root": str(remote / "runs/fold0/g2"),
        "primary_root": str(primary), "mirror_root": str(mirror), "files": files,
        "remote_inventory_path": "closeout/remote_inventory.json"})
    prior_request = write(tmp_path / "prior_request.json", {"durable_root": str(tmp_path / "old_phase4"),
                            "mirror_root": str(tmp_path / "old_phase4_mirror")})
    prior = {"hardware_profile": "runpod_rtx4090", "runtime_environments": {"pinned": True},
             "remote_repo": str(tmp_path / "pod/repo"), "remote_campaign": str(remote), "remote_python": "/runtime/python",
             "bootstrap_python": sys.executable, "embedding": {"pinned": True}, "state_root": str(tmp_path / "old_state"),
             "runpod": {"gpu_id": "4090", "image": "sha256:fixed", "runtime_environment": {"TF32": "0"},
                        "state_root": str(tmp_path / "old_controller")},
             "execution_manifest": str(manifest), "durable_root": str(primary), "mirror_root": str(mirror),
             "phase4": {"request_path": str(prior_request), "request_sha256": phase3._sha(prior_request)}}
    config_path = write(tmp_path / "prior_config.json", prior)
    driver = write(tmp_path / "prior_driver.json", {"closed": True, "config_sha256": phase3._sha(config_path),
        "session_id": "old", "pod_id": "old-pod", "status": "interrupted", "completed": [],
        "closeout": {"status": "DELETED", "pod_id": "old-pod", "provider_absence_verified": True,
                     "preservation_sha256": phase3._sha(proof)}})
    reviewed = write(tmp_path / "new_request.json", {"durable_root": str(tmp_path / "new_phase4"),
        "mirror_root": str(tmp_path / "new_phase4_mirror"), "host_state_root": str(tmp_path / "new_host"),
        "max_download_bytes": 1000000, "remote_request_template": {"execution_root": str(remote / "runs/fold0/phase4")}})
    config = {**prior, "state_root": str(tmp_path / "new_state"), "durable_root": str(tmp_path / "new_primary"),
              "mirror_root": str(tmp_path / "new_mirror"),
              "runpod": {**prior["runpod"], "state_root": str(tmp_path / "new_controller")},
              "phase4": {"request_path": str(reviewed), "request_sha256": phase3._sha(reviewed)},
              "resume": {"config_path": str(config_path), "config_sha256": phase3._sha(config_path),
                         "state_path": str(driver), "state_sha256": phase3._sha(driver),
                         "preservation_path": str(proof), "preservation_sha256": phase3._sha(proof)}}
    return config, source, artifact, remote


def test_resume_authenticates_copies_and_restores_without_provider_work(preserved_session):
    config, source, artifact, remote = preserved_session
    assert execution_v5.validate_resume(config, source)["state"]["session_id"] == "old"
    uploads = []
    class Transport:
        stop_controller = SimpleNamespace(register_output_roots=lambda outputs: None)
        def upload(self, local, target, *, timeout):
            target = Path(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(local).read_bytes())
            uploads.append(target)
        def remote(self, argv, *, timeout):
            # The pod side of a batched restore runs here as the same program.
            assert argv[:3] == [sys.executable, "-c", execution_v5.RESTORE_PROGRAM]
            return subprocess.run(argv, check=True, capture_output=True, text=True).stdout
    state = {}
    state_path = Path(config["state_root"]) / "driver_state.json"
    execution_v5.restore_session(config, Transport(), state, state_path, source, lambda: None)
    assert (remote / "runs/fold0/g2/kept.json").read_bytes() == artifact.read_bytes()
    assert len(uploads) == 1 and state["resume_restored"] == config["resume"]["preservation_sha256"]
    execution_v5.restore_session(config, Transport(), state, state_path, source, lambda: None)
    assert len(uploads) == 1


@pytest.mark.parametrize("mutation", ["gpu", "environment", "result", "source", "destination"])
def test_resume_refuses_changed_hardware_environment_results_or_identity(preserved_session, mutation):
    config, source, artifact, remote = preserved_session
    if mutation == "gpu":
        config["runpod"]["gpu_id"] = "other"
    elif mutation == "environment":
        config["runtime_environments"] = {"pinned": False}
    elif mutation == "result":
        artifact.write_text("changed")
    elif mutation == "source":
        source["scientific_identity_sha256"] = "f" * 64
    else:
        config["durable_root"] = str(artifact.parent)
    with pytest.raises((RuntimeError, ValueError)):
        execution_v5.validate_resume(config, source)


def test_g2_resume_keeps_completed_result_and_bounds_missing_result(tmp_path, monkeypatch):
    from ec_clean30 import workstation
    monkeypatch.setattr(workstation, "measurement_queue", lambda source: (("timing-unit", "F2"),))
    write(tmp_path / "runs/ecc30timing_F2_attempt1/g2_complete.json", {"status": "passed"})
    write(tmp_path / "commands/ecc30_pretrain_fold0_attempt1.json", {})
    assert execution_v5.g2_resume_attempts(tmp_path, {}) == {"pretrain": 2, "F2": 1}
    write(tmp_path / "commands/ecc30_pretrain_fold0_attempt2.json", {})
    with pytest.raises(execution_v5.LaunchWindowError, match="already been used"):
        execution_v5.g2_resume_attempts(tmp_path, {})


@pytest.mark.parametrize("seed", [43, 46])
def test_esm1b_worker_invokes_one_seed_specific_wrapper(tmp_path, monkeypatch, seed):
    source, view = {}, SimpleNamespace()
    root = tmp_path / "execution"
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"weights")
    contact = tmp_path / "weights-contact-regression.pt"
    contact.write_bytes(b"contact")
    harness, extract = tmp_path / "harness.py", tmp_path / "extract.py"
    pins = {str(harness): "73c25c5396f89a603a2ca97cf2adf82bb6b220bccf4d9f45801d1bdcf60bede4",
            str(extract): "902a69cffa0b35ab16798ba655d7b0e357d0e8efd5a268558cc4b064d2e7719d"}
    original_sha = phase4._sha
    monkeypatch.setattr(phase4, "_sha", lambda path: pins[str(path)] if str(path) in pins else original_sha(path))
    monkeypatch.setattr(phase4, "_runtime", lambda *a, **k: (source, view, None, {}, None))
    record = {"files": {"sequence_map.tsv": "a" * 64}, "units": [
        {"seed": s, "arm": "r1", "masked_manifest": f"mask{s}.tsv"} for s in range(42, 47)]}
    monkeypatch.setattr(phase4, "_clean_inputs", lambda *a: (inputs, record))
    monkeypatch.setattr(phase4, "_clean_roots", lambda *a: [])
    monkeypatch.setattr(phase3, "_esm1b_cuda_preflight", lambda *a: {})
    monkeypatch.setattr(phase3, "_esm1b_gpu_evidence", lambda *a: {})
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        farm = Path(argv[argv.index("--out-dir") + 1])
        write(farm / "farm_receipt.json", {})
        (farm / phase4.clean.FARM_MANIFEST).write_text("manifest")
    monkeypatch.setattr(phase4.subprocess, "run", run)
    request = {"execution_root": str(root), "fold": 0, "clean_scope": "final",
               "unit_id": f"ecc30_clean_embed_fold0_esm1b_masked_seed{seed}", "harness": str(harness),
               "extract_script": str(extract), "weights": str(weights), "weights_sha256": original_sha(weights),
               "contact_weights_sha256": original_sha(contact), "embedding_python": "pinned-python",
               "threads": 2, "cap_prefix": ["cap"], "g3_sha256": "g" * 64}
    assert phase4.embedding_worker(request)["status"] == "passed"
    assert len(calls) == 1
    argv = calls[0]
    assert argv[argv.index("--seed") + 1] == str(seed)
    assert argv.count("--masked-manifest") == 1 and str(inputs / f"mask{seed}.tsv") in argv


def test_esmc_worker_runs_fp32_sequences_then_farm(tmp_path, monkeypatch):
    directory = tmp_path / "attempt"
    directory.mkdir()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    source = {"scientific_identity_sha256": "a" * 64, "fit_chain": ["b" * 64, "c" * 64], "git_head": "d" * 40}
    record = {"files": {"sequence_map.tsv": "e" * 64}, "units": []}
    monkeypatch.setattr(phase4.clean, "read_sequence_map_file", lambda *a: ({"A": "MALA"}, {}))
    monkeypatch.setattr(phase4.clean, "farm_labels", lambda *a, **k: ({"A": (None, "MALA")}, []))
    monkeypatch.setattr(phase3, "pretrain_embedding_command", lambda *a, **k:
        (["python", "esmc_chain_means", "sequences", "--forward-precision", "fp32"], {"HF_HUB_OFFLINE": "1"}))
    monkeypatch.setattr(phase4, "_esmc_farm_evidence", lambda farm: {"verified": True})
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        if "sequences" in argv:
            write(directory / "sequence_cache/manifest.json", {})
        else:
            assert "--layer" in argv and argv[argv.index("--layer") + 1] == "final"
            write(directory / "farm" / phase4.clean.FARM_RECEIPT, {})
            (directory / "farm" / phase4.clean.FARM_MANIFEST).write_text("farm")
    monkeypatch.setattr(phase4.subprocess, "run", run)
    result = phase4._esmc_farm_worker({"python": "python", "cap_prefix": ["cap"], "g3_sha256": "g" * 64},
        source, None, inputs, record, {"unit_id": "base", "kind": "base", "embedding": "esmc", "seed": None}, directory, [])
    assert result["status"] == "passed" and len(calls) == 2
    assert "sequences" in calls[0] and "fp32" in calls[0] and "esmc-farm" in calls[1]
