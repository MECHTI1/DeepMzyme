"""Tests for the additive trunk-LR safeguards v2 (pmm_trunklr_safeguards_v2.py, pmm_trunklr_v2.py,
pmm_trunklr_assessment_v2.py): session authorization at submission and the final-worker completion gate.

CPU only, synthetic fixtures in tmp_path; nothing contacts a VM. The comparison's intended limits are used: one
session of at most 9,000 s. The only real files read (when Data1 is mounted) are the historical authorization and
assessment, which are never written.
"""

from __future__ import annotations


import sys as _sys
from pathlib import Path as _Path
for _p in (
    str(_Path(__file__).resolve().parents[2]),
    str(_Path(__file__).resolve().parents[2] / "src"),
    str(_Path(__file__).resolve().parents[1]),
    str(_Path(__file__).resolve().parent),
):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
from project_paths import RELOCATED_FILES, install_relocated_module_aliases, resolve_existing_path, resolve_repo_file
install_relocated_module_aliases()
import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests")]

import pmm_trunklr as tl  # noqa: E402
import pmm_trunklr_assessment_v2 as av  # noqa: E402
import pmm_trunklr_safeguards_v2 as sg  # noqa: E402
import pmm_trunklr_v2 as tl2  # noqa: E402
from test_trunklr import UNITS, gated, ready_state, step_for  # noqa: E402
from test_v3_step_c_launcher import FakeVM  # noqa: E402

MAX_SECONDS = 9000.0
NOW = 1_791_910_000.0  # 2026-10-13 16:46:40 UTC
RUNNER_SHA = "ab" * 32
LANES = {0: 1, 1: 0, 2: 1, 3: 2, 4: 0}  # the completed campaign's fold -> lane mapping
HISTORICAL = tl.RECORDS / "assessments" / "TLR_20261013T174637Z" / "assessment.json"
HISTORICAL_SHA256 = "400509066999f66769d8792111734d824d36da34dd02a01961c78dc044d92814"


def sha(path: Path) -> str:
    return hashlib.sha256(resolve_existing_path(path).read_bytes()).hexdigest()


def a_session(session_id="session-A", *, instance="111", start=NOW - 600.0, seconds=8820.0, remaining=None):
    deadline = start + seconds
    return {"session_id": session_id, "instance_id": instance, "allocation_started": start, "deadline": deadline,
            "max_seconds": seconds, "remaining_seconds": deadline - NOW if remaining is None else remaining}


# ---------------------------------------------------------------------------
# Session authorization
# ---------------------------------------------------------------------------

def plain_records(tmp_path: Path) -> Path:
    """Minimal recorded manifest, ceiling and v1 authorization (the frozen gates are tested in test_trunklr)."""
    records = tmp_path / "records"
    paths = tl.record_paths(records)
    tl.write_json_once(paths["manifest"], {"units": UNITS})
    tl.write_json_once(paths["ceiling"], {"ceiling_usd": 2.5})
    tl.write_json_once(paths["authorization"], {"manifest_sha256": sha(paths["manifest"]),
                                                "ceiling_sha256": sha(paths["ceiling"]), "log_entry": "tlr-002"})
    return records


def record_policy(records: Path, **overrides) -> dict:
    kwargs = {"log_entry": "tlr-900", "decision": "test decision", "units": list(UNITS), "max_sessions": 1,
              "max_session_seconds": MAX_SECONDS, "expires_at": "2026-10-20T00:00:00Z"}
    record = sg.policy_record(records=records, **{**kwargs, **overrides})
    return tl.write_json_once(sg.policy_paths(records)["policy"], record)


def gate(records, session, units=UNITS[:1], now=NOW):
    return sg.launch_gate(records, units=list(units), session=session, now=now)


def test_a_valid_first_session_is_bound_once_and_later_launches_in_it_reuse_the_binding(tmp_path):
    records = plain_records(tmp_path)
    record_policy(records)
    problems, binding = gate(records, a_session())
    assert problems == [] and binding["ordinal"] == 1 and binding["session_id"] == "session-A"
    bound = sg.bind_session_once(records, binding)
    assert bound["instance_id"] == "111" and bound["recorded_duration_seconds"] == 8820.0
    assert bound["policy_sha256"] == sha(sg.policy_paths(records)["policy"])
    for units in (UNITS[1:3], UNITS[3:]):
        assert gate(records, a_session(), units=units) == ([], None)
    with pytest.raises(tl.LauncherError, match="already bound"):
        sg.bind_session_once(records, binding)
    assert sorted(p.name for p in sg.policy_paths(records)["sessions"].iterdir()) == ["session_1.json"]


def test_a_second_session_is_refused_even_with_unused_dollars(tmp_path):
    records = plain_records(tmp_path)
    record_policy(records)
    sg.bind_session_once(records, gate(records, a_session())[1])
    problems, binding = gate(records, a_session("session-B", start=NOW - 60.0, seconds=3600.0))
    assert binding is None and any("would be session 2 of at most 1" in p and "ceiling has room" in p for p in problems)


def test_a_session_longer_than_the_limit_is_refused_although_it_fits_the_dollar_ceiling(tmp_path):
    records = plain_records(tmp_path)
    record_policy(records)
    problems, binding = gate(records, a_session(seconds=9600.0))  # $8.2 of compute, still under a large ceiling
    assert binding is None and any("above the authorized 9000 s" in p for p in problems)
    assert gate(records, a_session(seconds=9000.0))[0] == []


@pytest.mark.parametrize("key,value", [("instance", "222"), ("start", NOW - 500.0), ("seconds", 8000.0)])
def test_a_bound_session_whose_instance_start_or_deadline_changed_is_refused(tmp_path, key, value):
    records = plain_records(tmp_path)
    record_policy(records)
    sg.bind_session_once(records, gate(records, a_session())[1])
    problems, binding = gate(records, a_session(**{key: value}))
    assert binding is None and any("changed since it was bound" in p for p in problems)


def test_an_expired_authorization_is_refused(tmp_path):
    records = plain_records(tmp_path)
    record_policy(records, expires_at="2026-10-13T18:47:39Z")
    problems, _ = gate(records, a_session(), now=sg.e2.stamp_seconds("2026-10-13T19:00:00Z"))
    assert any("expired" in p for p in problems)


def test_the_historical_v1_authorization_is_never_a_v2_launch_policy(tmp_path):
    records = plain_records(tmp_path)
    shutil.copyfile(tl.record_paths(records)["authorization"], sg.policy_paths(records)["policy"])
    problems, binding = gate(records, a_session())
    assert binding is None and any("lacks the required v2 bindings" in p for p in problems)


@pytest.mark.skipif(not (tl.RECORDS / "tlr_authorization.json").is_file(), reason="Data1 records not mounted")
def test_the_real_historical_authorization_lacks_the_v2_bindings():
    authorization = sg.e2.read_json(tl.RECORDS / "tlr_authorization.json")
    problems = sg.policy_problems(authorization, manifest_sha256=None, ceiling_sha256=None, authorization_sha256=None,
                                  code_sha256=sg.safety_code_sha256(), now=NOW)
    assert len(problems) == 1 and "lacks the required v2 bindings" in problems[0]
    for key in ("safety_code_sha256", "authorized_units", "max_sessions", "max_session_seconds", "expires_at"):
        assert key in problems[0]


@pytest.mark.parametrize("damage,message", [
    (lambda r: sg.policy_paths(r)["policy"].write_text("{not json"), "unreadable"),
    (lambda r: rewrite(sg.policy_paths(r)["policy"], max_sessions="one"), "maximum session count is malformed"),
    (lambda r: rewrite(sg.policy_paths(r)["policy"], max_session_seconds=-1), "duration is malformed"),
    (lambda r: rewrite(sg.policy_paths(r)["policy"], authorization_ref={"log_entry": "v3-001"}), "dated authorization"),
    (lambda r: rewrite(sg.policy_paths(r)["policy"], expires_at="soon"), "expiry is malformed"),
    (lambda r: rewrite(sg.policy_paths(r)["policy"], safety_code_sha256={}), "safety code changed"),
    (lambda r: rewrite(tl.record_paths(r)["ceiling"], ceiling_usd=5.0), "ceiling_sha256"),
    (lambda r: rewrite(tl.record_paths(r)["manifest"], units=[]), "manifest_sha256"),
    (lambda r: sg.policy_paths(r)["policy"].unlink(), "no v2 launch policy"),
])
def test_a_missing_malformed_or_inconsistent_policy_is_refused(tmp_path, damage, message):
    records = plain_records(tmp_path)
    record_policy(records)
    damage(records)
    problems, binding = gate(records, a_session())
    assert binding is None and any(message in p for p in problems), problems


def rewrite(path: Path, **changes) -> None:
    payload = json.loads(path.read_text())
    payload.update(changes)
    path.write_text(json.dumps(payload))


@pytest.mark.parametrize("damage,message", [
    (lambda s: (s / "session_1.json").write_text("garbage"), "unreadable"),
    (lambda s: (s / "session_1.json").rename(s / "session_2.json"), "history is malformed"),
    (lambda s: (s / "stray.json").write_text("{}"), "history is malformed"),
    (lambda s: rewrite(s / "session_1.json", policy_sha256="0" * 64), "another launch policy"),
    (lambda s: rewrite(s / "session_1.json", ordinal=3), "inconsistent with its position"),
    (lambda s: (lambda p: (p.write_text(json.dumps({k: v for k, v in json.loads(p.read_text()).items()
                                                    if k != "deadline"}))))(s / "session_1.json"), "malformed"),
])
def test_a_missing_or_malformed_session_history_is_refused(tmp_path, damage, message):
    records = plain_records(tmp_path)
    record_policy(records)
    sg.bind_session_once(records, gate(records, a_session())[1])
    damage(sg.policy_paths(records)["sessions"])
    problems, binding = gate(records, a_session())
    assert binding is None and any(message in p for p in problems), problems


@pytest.mark.parametrize("session", [None, {"session_id": "s"}, {**a_session(), "instance_id": ""},
                                     {**a_session(), "deadline": "late"}, {**a_session(), "deadline": NOW - 9000.0}])
def test_a_malformed_controller_session_is_refused(tmp_path, session):
    records = plain_records(tmp_path)
    record_policy(records)
    problems, binding = gate(records, session)
    assert binding is None and problems


def test_units_outside_the_policy_are_refused(tmp_path):
    records = plain_records(tmp_path)
    record_policy(records, units=UNITS[:2])
    problems, _ = gate(records, a_session(), units=UNITS[1:3])
    assert any("not authorized by the launch policy" in p for p in problems)


# ---------------------------------------------------------------------------
# Submission through the v2 launcher (no launch call on any refusal)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def no_lock():
    yield


def launch_setup(tmp_path, monkeypatch):
    campaign, records, manifest = gated(tmp_path, monkeypatch)
    record_policy(records, units=list(UNITS))
    step = step_for(campaign, records, manifest)
    return records, manifest, step


def launch(records, manifest, step, *, unit, lane, session, fresh=None, lock=no_lock):
    fake = FakeVM(ready_state(manifest))
    try:
        out = tl2.cmd_launch_v2(fake, session, [(unit, lane)], step=step, records=records, wait=False,
                                read_session=lambda: fresh or session, lock=lock, now=lambda: NOW)
    except tl.LauncherError as exc:
        return fake, exc
    return fake, out


def test_the_v2_launcher_binds_the_first_session_and_admits_repeated_launches_in_it(tmp_path, monkeypatch):
    records, manifest, step = launch_setup(tmp_path, monkeypatch)
    fake, out = launch(records, manifest, step, unit=UNITS[0], lane=0, session=a_session())
    assert len(fake.launched()) == 1 and out["session_binding"]["session_id"] == "session-A"
    assert "--session-id session-A" in fake.launched()[0]
    fake, out = launch(records, manifest, step, unit=UNITS[1], lane=1, session=a_session())
    assert len(fake.launched()) == 1 and out["session_binding"] == "already bound"


@pytest.mark.parametrize("case", ["second_session", "too_long", "changed_instance", "changed_since_start",
                                  "no_policy", "expired", "lock_busy", "admission"])
def test_every_refusal_sends_no_launch_to_the_vm(tmp_path, monkeypatch, case):
    records, manifest, step = launch_setup(tmp_path, monkeypatch)
    first = launch(records, manifest, step, unit=UNITS[0], lane=0, session=a_session())[0]
    assert len(first.launched()) == 1
    kwargs = {"unit": UNITS[1], "lane": 1, "session": a_session()}
    if case == "second_session":
        kwargs["session"] = a_session("session-B", seconds=3600.0)
    elif case == "too_long":
        kwargs["session"] = a_session("session-B", seconds=9600.0)
    elif case == "changed_instance":
        kwargs["session"] = a_session(instance="999")
    elif case == "changed_since_start":
        kwargs["fresh"] = a_session("session-C")
    elif case == "no_policy":
        sg.policy_paths(records)["policy"].unlink()
    elif case == "expired":
        rewrite(sg.policy_paths(records)["policy"], expires_at="2020-01-01T00:00:00Z")
    elif case == "lock_busy":
        controller = tmp_path / "controller"
        (controller / "state").mkdir(parents=True)
        (controller / "state" / "controller.lock").write_text("{}")
        holder = open(controller / "state" / "controller.lock")
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)  # a controller lifecycle action in progress
        kwargs["lock"] = lambda: sg.controller_lock(controller, wait_seconds=0)
    elif case == "admission":
        kwargs["session"] = a_session(remaining=1000.0)
    fake, out = launch(records, manifest, step, **kwargs)
    assert isinstance(out, tl.LauncherError), out
    assert fake.launched() == []
    assert sorted(p.name for p in sg.policy_paths(records)["sessions"].iterdir()) == ["session_1.json"]


def test_evidence_and_recovery_commands_pass_through_without_the_policy(monkeypatch):
    seen = []
    monkeypatch.setattr(tl, "main", lambda argv=None: seen.append(argv) or 0)
    for command in (["status"], ["wait"], ["pull", "--lanes", "0"], ["recover-lane", "1"], ["archive-failed", UNITS[0]],
                    ["evidence"], ["gates"], ["spend"], ["session"]):
        assert tl2.main(command) == 0
    assert [argv[0] for argv in seen] == ["status", "wait", "pull", "recover-lane", "archive-failed", "evidence",
                                          "gates", "spend", "session"]
    assert set(tl2.V2_COMMANDS) == {"launch", "policy", "record-policy"}


def test_the_controller_lock_is_shared_and_never_rewrites_the_owner_record(tmp_path):
    controller = tmp_path / "controller"
    (controller / "state").mkdir(parents=True)
    lock = controller / "state" / "controller.lock"
    lock.write_text('{"identity": "owner"}')
    with sg.controller_lock(controller, wait_seconds=0):
        with open(lock) as other:  # a second reader may hold it too; an exclusive controller cannot
            fcntl.flock(other, fcntl.LOCK_SH | fcntl.LOCK_NB)
            with pytest.raises(BlockingIOError), open(lock) as exclusive:
                fcntl.flock(exclusive, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert lock.read_text() == '{"identity": "owner"}'


# ---------------------------------------------------------------------------
# Final worker completion
# ---------------------------------------------------------------------------

def spec():
    return tl.load_spec()


def identity(fold):
    s = spec()
    return {"family": s["family"], "target_scheme": s["target"], "recipe": s["recipe"], "model_seed": s["seed"],
            "epochs": s["epochs"], "fold": fold,
            "lr_extension": {"campaign_id": tl.CAMPAIGN_ID, "gvp_learning_rate": s["candidate_gvp_learning_rate"],
                             "control_gvp_learning_rate": s["control_gvp_learning_rate"],
                             "runner_file": tl.VM_RUNNER, "runner_sha256": RUNNER_SHA}}


def write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload, sort_keys=True))


def rehash(evidence: Path) -> None:
    files = {p.relative_to(evidence).as_posix(): sha(p) for p in sorted(evidence.rglob("*"))
             if p.is_file() and p.name != "evidence_manifest.json"}
    write(evidence / "evidence_manifest.json", {"copied_at": "2026-10-13T17:44:57Z", "files": files, "mismatched": []})


def synthetic_evidence(tmp_path: Path, session=None) -> tuple[Path, Path, dict]:
    """A completed comparison's evidence copy and durable lanes, shaped like the real ones."""
    session = session or a_session("session-A", start=1791908449.0, seconds=8820.0)
    evidence, durable = tmp_path / "evidence_20261013T174424Z", tmp_path / "data" / "durable"
    launch_dir, campaign = evidence / "campaign" / "step_tlr" / "launch", evidence / "campaign"
    for fold, unit in enumerate(UNITS):
        lane = LANES[fold]
        argv = tl.unit_command(unit, lane, session, fit_seconds=2400.0)
        write(launch_dir / f"{unit}.launch.json", {"argv": argv, "batch": "tlr-1", "fit_seconds": 2400.0, "kind": "unit",
                                                   "lane": lane, "launched_at": "2026-10-13T17:10:54Z",
                                                   "session_id": session["session_id"], "unit": unit})
        write(launch_dir / f"{unit}.rc", "0\n")
        write(launch_dir / f"{unit}.boot", "boot-1\n")
        write(launch_dir / f"{unit}.out", "fit log\n")
        write(launch_dir / f"{unit}.pid", "4242\n")
        claim = {"claimed_at": "2026-10-13T17:11:57+0000", "identity": identity(fold), "lane": lane, "run_name": unit,
                 "session_id": session["session_id"], "worker": {"boot_id": "boot-1", "pid": 4243}}
        write(campaign / "claims" / f"{unit}.json", claim)
        status = {"status": "completed", "unit": unit, "run_name": unit, "lane": lane, "identity": identity(fold),
                  "selected_epoch": 50, "selected_checkpoint": "terminal_model_checkpoint.pt",
                  "selected_checkpoint_sha256": hashlib.sha256(f"ckpt{fold}".encode()).hexdigest()}
        write(campaign / "lanes" / f"lane{lane}" / f"run_status_{unit}.json", status)
        (durable / f"lane{lane}").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(campaign / "lanes" / f"lane{lane}" / f"run_status_{unit}.json",
                        durable / f"lane{lane}" / f"run_status_{unit}.json")
        write(durable / f"lane{lane}" / "runs" / unit / "terminal_model_checkpoint.pt", f"ckpt{fold}")
    rehash(evidence)
    return evidence, durable, session


def report(evidence, durable=None, sessions=None):
    return sg.completion_report(evidence, spec=spec(), runner_sha256=RUNNER_SHA, sessions=sessions, durable=durable)


def test_valid_evidence_verifies_every_final_attempt(tmp_path):
    evidence, durable, session = synthetic_evidence(tmp_path)
    out = report(evidence, durable, sessions={session["session_id"]: session})
    assert out["verified"] and out["problems"] == [] and out["sessions_used"] == ["session-A"]
    assert {u["lane"] for u in out["units"].values()} == {0, 1, 2}


def unit_paths(evidence, fold=2):
    unit = UNITS[fold]
    launch = evidence / "campaign/step_tlr/launch"
    return unit, {"rc": launch / f"{unit}.rc", "launch": launch / f"{unit}.launch.json", "boot": launch / f"{unit}.boot",
                  "out": launch / f"{unit}.out", "claim": evidence / "campaign/claims" / f"{unit}.json",
                  "status": evidence / "campaign/lanes" / f"lane{LANES[fold]}" / f"run_status_{unit}.json"}


@pytest.mark.parametrize("change,message", [
    (lambda p: p["rc"].unlink(), "rc"),
    (lambda p: p["rc"].write_text("1\n"), "exited with code 1"),
    (lambda p: p["rc"].write_text("137\n"), "exited with code 137"),
    (lambda p: p["rc"].write_text("zero\n"), "not an integer exit code"),
    (lambda p: p["rc"].write_text(""), "not an integer exit code"),
    (lambda p: p["rc"].write_text("0 0\n"), "not an integer exit code"),
])
def test_a_missing_nonzero_or_malformed_final_exit_blocks_completion(tmp_path, change, message):
    evidence, durable, _ = synthetic_evidence(tmp_path)
    unit, paths = unit_paths(evidence)
    change(paths)
    rehash(evidence)
    out = report(evidence, durable)
    assert not out["verified"] and any(message in p for p in out["units"][unit]["problems"]), out["units"][unit]
    assert all(out["units"][u]["verified"] for u in UNITS if u != unit)


@pytest.mark.parametrize("key", ["rc", "boot", "out", "launch", "claim", "status"])
def test_a_file_whose_hash_differs_from_the_evidence_manifest_blocks_completion(tmp_path, key):
    evidence, durable, _ = synthetic_evidence(tmp_path)
    unit, paths = unit_paths(evidence)
    with paths[key].open("a") as handle:
        handle.write(" ")
    out = report(evidence, durable)
    assert not out["verified"] and any("SHA-256 differs" in p for p in out["units"][unit]["problems"])


def test_an_exit_file_missing_from_the_evidence_manifest_blocks_completion(tmp_path):
    evidence, durable, _ = synthetic_evidence(tmp_path)
    unit, _ = unit_paths(evidence)
    manifest = json.loads((evidence / "evidence_manifest.json").read_text())
    del manifest["files"][f"campaign/step_tlr/launch/{unit}.rc"]
    write(evidence / "evidence_manifest.json", manifest)
    assert any("not listed" in p for p in report(evidence, durable)["units"][unit]["problems"])


def edit(path: Path, update) -> None:
    payload = json.loads(path.read_text())
    update(payload)
    write(path, payload)


@pytest.mark.parametrize("change,message", [
    (lambda p: edit(p["launch"], lambda x: x.update(unit=UNITS[3])), "another unit"),
    (lambda p: edit(p["claim"], lambda x: x.update(lane=2)), "unit, lane or session differs"),
    (lambda p: edit(p["claim"], lambda x: x.update(session_id="session-B")), "unit, lane or session differs"),
    (lambda p: edit(p["status"], lambda x: x.update(run_name=UNITS[3])), "unit, run or lane differs"),
    (lambda p: edit(p["status"], lambda x: x.update(status="failed")), "not completed"),
    (lambda p: edit(p["status"], lambda x: x["identity"].update(fold=3)), "scientific identity differs"),
    (lambda p: [edit(p[k], lambda x: x["identity"].update(fold=3)) for k in ("status", "claim")], "in ['fold']"),
    (lambda p: [edit(p[k], lambda x: x["identity"]["lr_extension"].update(gvp_learning_rate="3e-4"))
                for k in ("status", "claim")], "learning-rate extension"),
    (lambda p: edit(p["status"], lambda x: x.update(selected_epoch=49)), "terminal epoch"),
    (lambda p: edit(p["launch"], lambda x: x["argv"].__setitem__(x["argv"].index("--lane") + 1, "2")), "argv"),
    (lambda p: edit(p["launch"], lambda x: x.update(session_id="session-B")), "argv names another session"),
    (lambda p: p["boot"].write_text("boot-2\n"), "final launch's boot"),
])
def test_a_wrong_unit_lane_session_or_identity_blocks_completion(tmp_path, change, message):
    evidence, durable, _ = synthetic_evidence(tmp_path)
    unit, paths = unit_paths(evidence)
    change(paths)
    rehash(evidence)
    out = report(evidence, durable)
    assert not out["verified"] and any(message in p for p in out["units"][unit]["problems"]), out["units"][unit]


def test_a_launch_outside_the_bound_session_allocation_blocks_completion(tmp_path):
    evidence, durable, session = synthetic_evidence(tmp_path)
    for other in ({**session, "deadline": session["deadline"] + 60}, {**session, "max_seconds": 9000.0}):
        out = report(evidence, durable, sessions={session["session_id"]: other})
        assert not out["verified"] and all(not u["verified"] for u in out["units"].values())
    assert not report(evidence, durable, sessions={})["verified"]


@pytest.mark.parametrize("variant", ["renamed_success_then_failure", "only_archived_success", "claim_of_earlier_attempt"])
def test_an_earlier_successful_attempt_cannot_mask_a_final_failure(tmp_path, variant):
    evidence, durable, _ = synthetic_evidence(tmp_path)
    unit, paths = unit_paths(evidence)
    archived = paths["rc"].with_name(paths["rc"].name + ".1")
    if variant == "renamed_success_then_failure":
        archived.write_text("0\n")
        paths["rc"].write_text("1\n")
    elif variant == "only_archived_success":
        paths["rc"].rename(archived)
    else:  # the claim kept from an attempt launched before the final launch
        edit(paths["claim"], lambda x: x.update(claimed_at="2026-10-13T16:30:00+0000"))
        archived.write_text("0\n")
    rehash(evidence)
    out = report(evidence, durable)
    assert not out["verified"] and not out["units"][unit]["verified"]
    if variant != "claim_of_earlier_attempt":
        assert out["units"][unit]["earlier_attempts"]["renamed_launch_files"] == [archived.name]


def test_paths_must_stay_inside_the_evidence_root(tmp_path):
    evidence, durable, _ = synthetic_evidence(tmp_path)
    unit, paths = unit_paths(evidence)
    outside = tmp_path / "outside_claim.json"
    shutil.copyfile(paths["claim"], outside)
    paths["claim"].unlink()
    os.symlink(outside, paths["claim"])
    out = report(evidence, durable)
    assert not out["verified"] and any("symbolic link" in p for p in out["units"][unit]["problems"])
    for rel in ("../escape.json", "/etc/passwd", "campaign/../../x"):
        assert sg._confined(evidence, rel)[1] is not None


def test_a_durable_copy_that_differs_blocks_completion(tmp_path):
    evidence, durable, _ = synthetic_evidence(tmp_path)
    unit, _ = unit_paths(evidence)
    (durable / f"lane{LANES[2]}" / "runs" / unit / "terminal_model_checkpoint.pt").write_text("other")
    assert any("durable selected checkpoint" in p for p in report(evidence, durable)["units"][unit]["problems"])


# ---------------------------------------------------------------------------
# The decision gate and preservation of the scientific result
# ---------------------------------------------------------------------------

def assess_setup(tmp_path):
    evidence, durable, _ = synthetic_evidence(tmp_path)
    records = tmp_path / "records"
    manifest_path = records / "tlr_manifest.json"
    write(manifest_path, {"vm_runner": {"sha256": RUNNER_SHA}})
    return evidence, records, manifest_path


def test_a_failed_completion_check_prevents_any_scientific_decision(tmp_path, monkeypatch):
    evidence, records, manifest_path = assess_setup(tmp_path)
    unit, paths = unit_paths(evidence)
    paths["rc"].write_text("1\n")
    rehash(evidence)
    monkeypatch.setattr(av, "run_frozen", lambda **_: pytest.fail("the frozen assessor must not run"))
    out = av.assess(data=tmp_path / "data", records=records, manifest_path=manifest_path, evidence_dir=evidence,
                    out_dir=tmp_path / "out", cohort=tmp_path / "cohort.csv", sessions=None)
    assert out["decision"] is None and out["status"].startswith("blocked") and not out["completion"]["verified"]
    assert json.loads((tmp_path / "out" / "assessment_v2.json").read_text())["decision"] is None


def test_verified_completion_runs_the_frozen_assessor_into_a_fresh_directory(tmp_path, monkeypatch):
    evidence, records, manifest_path = assess_setup(tmp_path)

    def fake_frozen(*, out_dir, **_):
        out_dir.mkdir(parents=True)
        (out_dir / "assessment.json").write_text(json.dumps(
            {"result": {"status": "complete", "decision": {"outcome": "reference_retained"}}}))
        return out_dir / "assessment.json"

    monkeypatch.setattr(av, "run_frozen", fake_frozen)
    out = av.assess(data=tmp_path / "data", records=records, manifest_path=manifest_path, evidence_dir=evidence,
                    out_dir=tmp_path / "out", cohort=tmp_path / "cohort.csv", sessions=None)
    assert out["decision"] == {"outcome": "reference_retained"} and out["completion"]["verified"]
    with pytest.raises(tl.LauncherError, match="fresh directory"):
        av.assess(data=tmp_path / "data", records=records, manifest_path=manifest_path, evidence_dir=evidence,
                  out_dir=tmp_path / "out", cohort=tmp_path / "cohort.csv", sessions=None)
    with pytest.raises(tl.LauncherError, match="assessments folder"):
        av.fresh_dir(records / "assessments" / "TLR_new", records)


def historical_like(mean=0.0116):
    return {"result": {"status": "complete", "final_failures": [], "reasons": ["(b) not interval-supported"],
                       "decision": {"outcome": "reference_retained", "reasons": ["(b) not interval-supported"],
                                    "gvp_learning_rate": "3e-4"},
                       "comparison": {"fold_common4_ba": {"candidate": {"0": 0.76}, "control": {"0": 0.72}},
                                      "fold_differences": [0.0485], "mean_difference": mean,
                                      "unadjusted": {"bootstrap": [-0.0065, 0.0311], "t": [-0.0189, 0.0422]}}},
            "unit_evidence": {UNITS[0]: {"checkpoint_sha256": "a", "predictions_sha256": "b"}},
            "control_parity": {"equal": True}, "configuration_difference": {"all_expected": True},
            "spec_sha256": "61728a8f", "held_out_test_accessed": False, "created_at": "x", "purpose": "assessment"}


def test_valid_evidence_preserves_the_original_scientific_result():
    history = historical_like()
    rerun = {**historical_like(), "created_at": "later", "purpose": "retrospective verification re-run"}
    assert av.differences(av.scientific_view(history), av.scientific_view(rerun)) == []
    assert av.scientific_view(rerun)["decision_outcome"] == "reference_retained"
    changed = av.differences(av.scientific_view(history), av.scientific_view(historical_like(mean=0.02)))
    assert changed == ["/mean_difference: 0.0116 != 0.02"]


@pytest.mark.skipif(not HISTORICAL.is_file(), reason="Data1 records not mounted")
def test_the_historical_assessment_is_unchanged_and_retains_the_reference():
    assert sha(HISTORICAL) == HISTORICAL_SHA256
    history = json.loads(HISTORICAL.read_text())
    view = av.scientific_view(history)
    assert view["decision_outcome"] == "reference_retained" and view["result"]["status"] == "complete"
    assert round(100 * view["mean_difference"], 2) == 1.16 and view["held_out_test_accessed"] is False


def test_the_v2_launcher_never_imports_torch_and_no_frozen_file_changed():
    import subprocess

    probe = f"import sys; sys.path[:0] = [{str(ROOT)!r}, {str(ROOT / 'src')!r}]; import project_paths, pmm_trunklr_v2, pmm_trunklr_safeguards_v2; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", probe], cwd=ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
    manifest_path = tl.RECORDS / "tlr_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        tracked = subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "--", *(str(resolve_existing_path(ROOT / name)) for name in {**manifest["tooling_sha256"], **manifest["runner_sha256"]})])
        assert tracked.returncode == 0
    assert sha(tl.SPEC_JSON) == tl.FROZEN_SPEC_SHA256
