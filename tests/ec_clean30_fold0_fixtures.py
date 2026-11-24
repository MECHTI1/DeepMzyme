"""Synthetic fold-0 fixtures for dispatch, resume, preservation and closeout tests."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from benchmarking.pmm_execution import CampaignExecution, ExecutionPolicy
from ec_clean30 import campaign, execution_v5 as v5, phase3, phase4, runpod_bootstrap as bootstrap, workstation

IDENTITY = {"scientific_identity_sha256": "a" * 64, "fit_chain": ["b" * 64, "c" * 64]}
SOURCE = {"format": phase3.EXECUTION_V5_FORMAT, "published": True, **IDENTITY, "git_head": "e" * 40,
          "hardware_profiles": ["runpod_l4", "runpod_rtx4090"],
          "freeze1_sha256": "d" * 64, "execution_manifest_sha256": "9" * 64, "amendment_chain": IDENTITY["fit_chain"]}
HOUR = 3600
PLAN = "2" * 64
BASE, MASKED = "ecc30_clean_embed_base", "ecc30_clean_embed_masked"


def write(path, value):
    """Preserve literal text and serialize records with the bytes used by the resume checksums."""
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
    """Build a preserved first session and a second session configured to resume it."""
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


@pytest.fixture
def approved(tmp_path, monkeypatch):
    """Create matching synthetic launch authorization, configuration and failure ledger."""
    identity = {"scientific_identity_sha256": "a" * 64, "fit_chain": ["b" * 64, "c" * 64]}
    source = {"format": phase3.EXECUTION_V5_FORMAT, "published": True, **identity,
              "amendment_chain": identity["fit_chain"], "execution_manifest_sha256": "d" * 64,
              "git_head": "e" * 40}
    request = write(tmp_path / "reviewed-request.json", identity)
    preapproval = write(tmp_path / "preapproval.json", {"format": v5.PREAPPROVAL_FORMAT,
        "status": "approved", "campaign_id": phase3.CAMPAIGN_ID, **identity,
        "scope": "amendment2_fold_execution_v5", "ceilings": {"gpu_seconds": 3600, "cpu_seconds": 3600, "cost_usd": 8}})
    manifest = write(tmp_path / "manifest.json", source)
    write(request, {**identity, "execution_manifest": str(manifest)})
    config = {"provider": "runpod", "execution_scope": "g2_then_fold0_v5", "state_root": str(tmp_path / "pass"),
              "execution_manifest": str(manifest), "runpod": {"gross_ceiling_usd": 8, "max_runtime_seconds": 3600},
              "phase4": {"request_path": str(request), "request_sha256": sha(request),
                         "preapproval_path": str(preapproval), "preapproval_sha256": sha(preapproval)}}
    config_path = write(tmp_path / "config.json", config)
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o700)
    authorization = {"format": v5.AUTHORIZATION_FORMAT, "launch_authorized": True, "provider": "runpod",
        "scope": "g2_then_fold0_v5", "approved_by": "maintainer-fixture", **identity,
        "config_sha256": sha(config_path), "execution_manifest_sha256": source["execution_manifest_sha256"],
        "fold0_cost_ceiling_usd": 8, "failed_window_limit": 2, "failure_ledger": str(shared / "ledger.json"),
        "failure_ledger_authorization_sha256": "f" * 64}
    auth_path = write(tmp_path / "authorization.json", authorization)
    ledger = write(Path(authorization["failure_ledger"]), v5.empty_failure_ledger(
        authorization["failure_ledger"], authorization["failure_ledger_authorization_sha256"]))
    ledger.chmod(0o600)
    monkeypatch.setattr(phase3, "authenticated_chain", lambda source: source)
    monkeypatch.setattr(phase3, "verify_execution_source", lambda *args, **kwargs: source)
    monkeypatch.setattr(phase3, "require_record_precedence", lambda record, source: phase3.require_same_scientific_identity(record, source))
    monkeypatch.setattr(workstation, "prepare_session_chain_receipt", lambda config: {})
    monkeypatch.setattr(workstation, "validate_phase4_plan", lambda *args: None, raising=False)
    return config, config_path, auth_path, authorization, source


@pytest.fixture
def cached_driver(tmp_path, monkeypatch):
    """An owned allocation with inert controller and transport boundaries."""
    events = []
    main, clean = tmp_path / "main.tar.gz", tmp_path / "clean.tar.gz"
    main.write_bytes(b"main archive")
    clean.write_bytes(b"clean archive")
    config = {"provider": "runpod", "artifact_cache": {"fixture": True},
              "state_root": str(tmp_path / "state"), "gcp_controller_root": str(tmp_path / "gcp"),
              "gcp_stopped_identity": {"name": "retained"}, "bootstrap_python": "/usr/bin/python3",
              "stage": [{"local": str(main), "remote": "/main", "kind": "file"},
                        {"local": str(clean), "remote": "/clean", "kind": "file"}],
              "runtime_environments": {"fit": {"archive": "/main"}, "esm1b": {"archive": "/clean"}},
              "remote_python": "/main/bin/python", "profiles": {"fit": {"cap_prefix": ["limits"]}},
              "remote_repo": "/repo", "hardware_profile": "approved"}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    source = {"format": phase3.BOOTSTRAP_FORMAT, "published": True}
    ready = {"source": source, "staging_peak_bytes": 1}
    begun = time.time()
    session_record = {"session_id": "owned-session", "pod_id": "owned-pod", "allocation_started_unix": begun,
                      "deadline_unix": begun + 6900, "hard_stop_unix": begun + 7020}

    class Controller:
        def start(self):
            events.append("start")
            return self.session()

        def session(self, previous=None):
            assert previous in (None, "owned-session")
            return dict(session_record)

        @contextmanager
        def maintain_heartbeat(self, *, interval):
            assert interval == 30
            events.append("heartbeat_enter")
            try:
                yield SimpleNamespace(check=lambda: events.append("heartbeat_check"))
            finally:
                events.append("heartbeat_exit")

        def write_ssh_config(self):
            return tmp_path / "ssh_config"

        def admit_workload(self):
            events.append("admit")

        def stop(self):
            events.append("stop")
            return {"desiredStatus": "EXITED", "runtimeStatus": "stopped"}

    class Transport:
        def remote(self, argv, **kwargs):
            if "import shutil; print(shutil.disk_usage('/workspace').free)" in argv:
                return str(1000 * workstation.GIB)
            if "cuda proof" in argv:
                return "{}"
            return "ready"

    def preserve(*args, max_download_bytes):
        assert type(max_download_bytes) is int and max_download_bytes > 0
        events.append("preserve")
        return None

    monkeypatch.setattr(workstation, "_prepared_preflight", lambda *a, **kw: ready)
    monkeypatch.setattr(bootstrap, "verify_seeded_cache", lambda *a, **kw: events.append("cache_verified"))
    monkeypatch.setattr(bootstrap, "staging_scratch_bytes", lambda *a: 1)
    monkeypatch.setattr(workstation.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="GCP TERMINATED"))
    monkeypatch.setattr(workstation, "_status", lambda *a: "TERMINATED")
    monkeypatch.setattr(workstation, "_bootstrap_runpod", lambda *a: events.append("bootstrap"))
    monkeypatch.setattr(workstation, "_runtime_preflight", lambda *a: events.append("runtime_preflight") or ("manifest", "probe"))
    monkeypatch.setattr(workstation, "_execute_queue", lambda *a: events.append("queue"))
    monkeypatch.setattr(workstation, "_preserve_runpod_outputs", preserve)
    return SimpleNamespace(config=config, path=path, ready=ready, events=events,
                           controller=Controller(), transport=Transport(), session=session_record)


def inventory_world(base, n_sequences):
    """A G2 root laid out like F6 preparation: two sequence-mean cache files per union sequence."""
    campaign = base / "remote" / "campaign"
    root = campaign / "runs/fold0/g2"
    attempt = root / "runs/ecc30_pretrain_fold0_attempt1"
    cache = attempt / "gpu_cache"
    cache.mkdir(parents=True)
    for index in range(n_sequences):
        digest = hashlib.sha256(f"union-{index}".encode()).hexdigest()
        for layer in ("final", "block26"):
            (cache / f"{digest}__esmc300m_{layer}_seqmean.pt").write_bytes(b"%d\n" % index)
    (attempt / "pretrain_attachment.json").write_text(json.dumps({"status": "restored"}))
    (root / "execution_state.json").write_text("{}")
    deadline = time.time() + 3600
    config = {"state_root": str(base / "state"), "runpod": {"ssh_alias": "synthetic-pod"},
              "bootstrap_python": sys.executable, "remote_campaign": str(campaign),
              "durable_root": str(base / "primary"), "mirror_root": str(base / "mirror")}
    controller = SimpleNamespace(heartbeat=lambda: None, config_sha="a" * 64)
    transport = workstation.RunpodTransport(config, controller)
    transport.deadline_unix, transport.reserve_seconds = deadline, 15
    session = {"pod_id": "pod", "session_id": "session", "deadline_unix": deadline}
    return SimpleNamespace(config=config, controller=controller, transport=transport, session=session, root=root,
                           campaign=campaign)


@pytest.fixture
def local_provider(monkeypatch):
    """Run every ssh command in a local shell; scp and rsync copy local files."""
    real_run = subprocess.run
    log = SimpleNamespace(ssh=[], tamper=False)

    def run(argv, *args, **kwargs):
        tool = argv[0]
        if tool == "ssh":
            result = real_run(["bash", "-c", argv[-1]], *args, **kwargs)
            log.ssh.append({"timeout": kwargs["timeout"], "stdout_chars": len(result.stdout or "")})
            return result
        if tool == "scp" and ":" not in argv[-2]:  # An upload: the host's acknowledgment goes to the pod.
            target = Path(argv[-1].split(":", 1)[1])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(argv[-2], target)
            return subprocess.CompletedProcess(argv, 0)
        if tool == "scp":
            remote, local = argv[-2].split(":", 1)[1], Path(argv[-1])
            data = Path(remote).read_bytes()
            if log.tamper:
                # Same length, still JSON: one recorded digest changes.
                match = re.search(rb'"sha256": "([0-9a-f])', data)
                flipped = b"1" if match.group(1) != b"1" else b"2"
                data = data[:match.start(1)] + flipped + data[match.end(1):]
            local.write_bytes(data)
            return subprocess.CompletedProcess(argv, 0)
        if tool == "rsync":
            listing = next(item for item in argv if item.startswith("--files-from=")).split("=", 1)[1]
            source, target = Path(argv[-2].split(":", 1)[1]), Path(argv[-1])
            for name in Path(listing).read_text().splitlines():
                (target / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / name, target / name)
            return subprocess.CompletedProcess(argv, 0)
        raise AssertionError(f"Unexpected provider process: {tool}")

    monkeypatch.setattr(workstation.subprocess, "run", run)
    # Thousands of synthetic files: durability syncs add nothing to this transfer check.
    monkeypatch.setattr(workstation.os, "fsync", lambda descriptor: None)
    return log


def fold_world(tmp_path):
    """A fold output root on the synthetic pod, with the session fields the pod's own execution record needs."""
    world = inventory_world(tmp_path, 1)
    world.config.update(remote_repo=str(Path(__file__).resolve().parents[1]), remote_python=sys.executable)
    world.session.update(allocation_started_unix=time.time() - 60, max_total_seconds=7200.0)
    world.fold = world.campaign / "runs/fold0/phase4"
    world.output = {"execution_root": str(world.fold), "primary_root": str(tmp_path / "fold-primary"),
                    "mirror_root": str(tmp_path / "fold-mirror")}
    policy = ExecutionPolicy(deadline_unix=world.session["deadline_unix"], max_total_seconds=7200.0,
                             allocation_started_unix=world.session["allocation_started_unix"])
    world.execution = CampaignExecution(world.fold, policy=policy, session_id=world.session["session_id"],
                                        durable_root=Path(world.output["primary_root"]), persistence_mode="host_pull")
    return world


def preserve_fold(world, tmp_path):
    path = workstation._preserve_runpod_outputs(world.config, world.transport, world.controller, world.session,
                                       output=world.output, receipt_name="preservation_receipt_1.json",
                                       host_state_root=str(tmp_path / "fold-state"))
    receipt = json.loads(path.read_text())
    return receipt, json.loads((Path(world.output["primary_root"]) / receipt["remote_inventory_path"]).read_text())
