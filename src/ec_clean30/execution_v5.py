"""Persistent launch-window accounting; this module grants no execution authority.

The published-source launch gate must supply the approved ledger path and
authorization digest. It must reserve a window before any provider action and
verify terminal provider evidence before closing that window here.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import time
from typing import Any, Iterator, Mapping
import uuid

FORMAT = "deepmzyme_ec_clean30_failed_launch_windows_v1"
FAILED_WINDOW_LIMIT = 2
REPO_ROOT = Path(__file__).resolve().parents[2]
TERMINAL_STATES = {"verified_stopped", "verified_absent", "not_allocated"}
AUTHORIZATION_FORMAT = "deepmzyme_ec_clean30_runpod_launch_authorization_v5"
PREAPPROVAL_FORMAT = "deepmzyme_ec_clean30_g3_preapproval_v5"


class LaunchWindowError(RuntimeError):
    """The shared ledger cannot admit or resolve this launch window."""


def _require(value: Any, message: str) -> None:
    if not value:
        raise LaunchWindowError(message)


def _sha256(value: Any, name: str) -> str:
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), f"{name} must be a SHA-256 digest.")
    return value


def _path(value: str | Path, name: str) -> Path:
    _require(isinstance(value, (str, Path)), f"{name} must be an explicit path.")
    path = Path(value)
    _require(path.is_absolute() and path == path.resolve() and ".." not in path.parts,
             f"{name} must be an absolute canonical path without symlinks.")
    _require(path != REPO_ROOT and REPO_ROOT not in path.parents, f"{name} must be outside the source repository.")
    return path


def empty_failure_ledger(path: str | Path, authorization_sha256: str) -> dict[str, Any]:
    """Return a document for review; do not create a file or an approval record."""
    return {"format": FORMAT, "ledger_path": str(_path(path, "ledger_path")),
            "authorization_sha256": _sha256(authorization_sha256, "authorization_sha256"),
            "failed_window_limit": FAILED_WINDOW_LIMIT, "windows": []}


class FailedWindowLedger:
    """One externally bound ledger survives pass directories and config changes."""

    def __init__(self, path: str | Path, authorization_sha256: str):
        self.path = _path(path, "ledger_path")
        self.authorization_sha256 = _sha256(authorization_sha256, "authorization_sha256")
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        _require(self.path.is_file(), "The bound launch-window ledger is missing; fresh launch is blocked.")
        parent = self.path.parent.stat()
        _require(parent.st_uid == os.getuid() and stat.S_IMODE(parent.st_mode) == 0o700,
                 "The launch-window ledger requires a private owned POSIX directory (0700).")

    @staticmethod
    def _private_file(descriptor: int) -> None:
        info = os.fstat(descriptor)
        _require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid()
                 and stat.S_IMODE(info.st_mode) == 0o600 and info.st_nlink == 1,
                 "Ledger and lock must be owned regular files, mode 0600, without hard-link aliases.")

    @contextmanager
    def _locked(self) -> Iterator[None]:
        _require(self.path.parent.resolve() == self.path.parent and self.path.resolve() == self.path,
                 "The bound ledger path changed or became a symlink.")
        parent = self.path.parent.stat()
        _require(parent.st_uid == os.getuid() and stat.S_IMODE(parent.st_mode) == 0o700,
                 "The launch-window ledger directory is no longer private and owned.")
        try:
            descriptor = os.open(self.lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        except OSError as exc:
            raise LaunchWindowError(f"Cannot acquire the shared launch-window lock: {exc}") from exc
        try:
            self._private_file(descriptor)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            os.close(descriptor)

    def _read(self) -> dict[str, Any]:
        try:
            descriptor = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(descriptor, "r") as handle:
                self._private_file(handle.fileno())
                document = json.load(handle)
        except (OSError, ValueError) as exc:
            raise LaunchWindowError(f"Cannot read the bound launch-window ledger: {exc}") from exc
        _require(isinstance(document, dict) and set(document) ==
                 {"format", "ledger_path", "authorization_sha256", "failed_window_limit", "windows"},
                 "Invalid launch-window ledger schema.")
        _require(document["format"] == FORMAT and document["ledger_path"] == str(self.path)
                 and document["authorization_sha256"] == self.authorization_sha256
                 and document["failed_window_limit"] == FAILED_WINDOW_LIMIT,
                 "Ledger path, authorization identity or failed-window ceiling differs.")
        rows = document["windows"]
        _require(isinstance(rows, list), "Launch windows must be an ordered list.")
        seen, unresolved, failures = set(), 0, 0
        for row in rows:
            _require(isinstance(row, dict), "Invalid launch-window record.")
            self._window_id(row.get("window_id"))
            _require(row["window_id"] not in seen, "A launch-window identifier is duplicated.")
            seen.add(row["window_id"])
            for key in ("config_sha256", "execution_manifest_sha256"):
                _sha256(row.get(key), key)
            _path(row.get("pass_root", ""), "pass_root")
            opened = row.get("opened_unix")
            _require(type(opened) in (int, float) and math.isfinite(opened) and opened > 0,
                     "Invalid launch-window opening time.")
            status = row.get("status")
            _require(status in {"open", "failed", "completed"}, "Unknown launch-window outcome.")
            base = {"window_id", "config_sha256", "execution_manifest_sha256", "pass_root", "opened_unix", "status"}
            if status == "open":
                unresolved += 1
                _require(set(row) == base and row is rows[-1], "An unresolved window must be the final ledger entry.")
            else:
                _require(set(row) == base | {"closed_unix", "closeout_sha256", "provider_terminal"},
                         "Closed launch-window evidence is incomplete.")
                _sha256(row["closeout_sha256"], "closeout_sha256")
                _require(type(row["closed_unix"]) in (int, float) and math.isfinite(row["closed_unix"])
                         and row["closed_unix"] >= opened and row["provider_terminal"] in TERMINAL_STATES,
                         "The launch window lacks a valid terminal provider observation.")
                failures += status == "failed"
        _require(unresolved <= 1 and failures <= FAILED_WINDOW_LIMIT
                 and not (unresolved and failures == FAILED_WINDOW_LIMIT),
                 "Launch-window history exceeds its failed-window bound.")
        return document

    @staticmethod
    def _window_id(value: Any) -> None:
        _require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value),
                 "A safe explicit launch-window identifier is required.")

    def _write(self, document: dict[str, Any]) -> None:
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.partial")
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(descriptor, "w") as handle:
                json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            directory = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)

    def status(self) -> dict[str, Any]:
        with self._locked():
            document = self._read()
        failures = sum(row["status"] == "failed" for row in document["windows"])
        return {"ledger_path": str(self.path), "authorization_sha256": self.authorization_sha256,
                "failed_windows": failures, "remaining_failed_windows": FAILED_WINDOW_LIMIT - failures,
                "unresolved_window": next((row for row in document["windows"] if row["status"] == "open"), None)}

    def begin_window(self, window_id: str, *, config_sha256: str, execution_manifest_sha256: str,
                     pass_root: str | Path) -> dict[str, Any]:
        """Reserve before provider access, or recover exactly the same unresolved window."""
        self._window_id(window_id)
        binding = {"window_id": window_id, "config_sha256": _sha256(config_sha256, "config_sha256"),
                   "execution_manifest_sha256": _sha256(execution_manifest_sha256, "execution_manifest_sha256"),
                   "pass_root": str(_path(pass_root, "pass_root"))}
        _require(self.path != Path(binding["pass_root"]) and Path(binding["pass_root"]) not in self.path.parents,
                 "The shared ledger must not live inside a pass directory.")
        with self._locked():
            document = self._read()
            for row in document["windows"]:
                if row["window_id"] == window_id:
                    _require(row["status"] == "open" and all(row[key] == value for key, value in binding.items()),
                             "A closed or differently bound launch window cannot be restarted.")
                    return dict(row)
            _require(not any(row["status"] == "open" for row in document["windows"]),
                     "An unresolved launch window blocks every fresh launch.")
            _require(sum(row["status"] == "failed" for row in document["windows"]) < FAILED_WINDOW_LIMIT,
                     "Two additional launch windows have failed; fresh launch is blocked.")
            row = {**binding, "opened_unix": time.time(), "status": "open"}
            document["windows"].append(row)
            self._write(document)
            return dict(row)


    def finish_window(self, window_id: str, *, outcome: str, closeout_sha256: str,
                      provider_terminal: str) -> dict[str, Any]:
        """Record already verified closeout evidence; no provider operation is performed."""
        self._window_id(window_id)
        _require(outcome in {"failed", "completed"}, "A launch window needs an explicit failed or completed outcome.")
        _require(provider_terminal in TERMINAL_STATES, "Verified terminal provider evidence is required.")
        result = {"status": outcome, "closeout_sha256": _sha256(closeout_sha256, "closeout_sha256"),
                  "provider_terminal": provider_terminal}
        with self._locked():
            document = self._read()
            row = next((row for row in document["windows"] if row["window_id"] == window_id), None)
            _require(row is not None, "Cannot close an unreserved launch window.")
            if row["status"] != "open":
                _require(all(row[key] == value for key, value in result.items()),
                         "A recorded launch-window outcome cannot be changed.")
                return dict(row)
            row.update(result, closed_unix=max(time.time(), row["opened_unix"]))
            self._write(document)
            return dict(row)

def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    _require(path.is_file() and not path.is_symlink() and path.absolute() == path.resolve(),
             "Execution records must be regular canonical files.")
    value = json.loads(path.read_bytes())
    _require(isinstance(value, dict), "Execution record must be a JSON object.")
    return value


def _bound_record(config: Mapping[str, Any], name: str) -> tuple[Path, dict[str, Any]]:
    path = Path(config[f"{name}_path"])
    value = _record(path)
    _require(_file_sha(path) == _sha256(config[f"{name}_sha256"], f"{name}_sha256"),
             f"The reviewed {name} bytes changed.")
    return path, value


def validate_launch_authorization(config: Mapping[str, Any], config_path: Path,
                                  authorization_path: Path, source: Mapping[str, Any]) -> dict[str, Any]:
    """Read an explicit maintainer authorization; never create or infer one."""
    from ec_clean30 import phase3, workstation

    authorization = _record(authorization_path)
    _require(_record(config_path) == dict(config), "The in-memory configuration differs from its reviewed file.")
    _require(config.get("provider") == "runpod" and config.get("execution_scope") == "g2_then_fold0_v5",
             "Execution v5 authorizes only its Runpod G2-through-fold-0 session.")
    _require(source.get("format") == phase3.EXECUTION_V5_FORMAT and source.get("published") is True,
             "Launch requires published execution v5 and amendment 2.")
    phase3.authenticated_chain(source)
    _require(authorization.get("format") == AUTHORIZATION_FORMAT
             and authorization.get("launch_authorized") is True
             and authorization.get("provider") == "runpod"
             and authorization.get("scope") == "g2_then_fold0_v5"
             and isinstance(authorization.get("approved_by"), str) and authorization["approved_by"].strip(),
             "A separate explicit maintainer launch authorization is required.")
    _require(authorization.get("config_sha256") == _file_sha(config_path)
             and authorization.get("execution_manifest_sha256") == source["execution_manifest_sha256"],
             "Launch authorization does not bind this exact reviewed execution configuration.")
    phase3.require_same_scientific_identity(authorization, source)
    ceiling = authorization.get("fold0_cost_ceiling_usd")
    _require(type(ceiling) in (int, float) and math.isfinite(ceiling) and ceiling > 0
             and ceiling == config.get("runpod", {}).get("gross_ceiling_usd"),
             "The fold-0 cost ceiling must be explicitly approved and equal the controller ceiling.")
    _require(authorization.get("failed_window_limit") == FAILED_WINDOW_LIMIT,
             "Authorization must retain the two-failed-window stop.")
    _path(authorization.get("failure_ledger", ""), "failure_ledger")
    _sha256(authorization.get("failure_ledger_authorization_sha256"), "failure_ledger_authorization_sha256")
    _, request = _bound_record(config["phase4"], "request")
    phase3.require_same_scientific_identity(request, source)
    _require(request.get("execution_manifest") == config["execution_manifest"],
             "The reviewed Phase-4 request names another execution manifest.")
    _, preapproval = _bound_record(config["phase4"], "preapproval")
    _require(preapproval.get("format") == PREAPPROVAL_FORMAT and preapproval.get("status") == "approved",
             "The session needs its reviewed numerical G3 preapproval before allocation.")
    phase3.require_same_scientific_identity(preapproval, source)
    _require(preapproval.get("ceilings", {}).get("cost_usd") == ceiling,
             "G3 preapproval and launch cost ceiling differ.")
    ceilings = preapproval["ceilings"]
    _require(all(type(ceilings.get(key)) in (int, float) and math.isfinite(ceilings[key])
                 and ceilings[key] > 0 for key in ("gpu_seconds", "cpu_seconds", "cost_usd"))
             and type(config["runpod"].get("max_runtime_seconds")) in (int, float)
             and 0 < config["runpod"]["max_runtime_seconds"] <= ceilings["gpu_seconds"],
             "The controller allocation must fit the positive G3 preapproved ceilings before launch.")
    workstation.validate_phase4_plan(config, source, request, preapproval)
    return authorization


def materialize_g3(preapproval: Mapping[str, Any], *, preapproval_sha256: str, freeze1c: Path,
                   out: Path, source: Mapping[str, Any], session: Mapping[str, Any],
                   gross_ceiling_usd: float) -> dict[str, Any]:
    """Bind an already approved numerical envelope to measured G2 and the existing allocation."""
    from ec_clean30 import phase3, provenance

    _require(preapproval.get("format") == PREAPPROVAL_FORMAT and preapproval.get("status") == "approved",
             "No approved numerical G3 envelope was supplied.")
    phase3.require_same_scientific_identity(preapproval, source)
    measured = _record(freeze1c)
    phase3.require_record_precedence(measured, source)
    _require(measured.get("format") == "deepmzyme_ec_clean30_freeze1c_v2",
             "G3 v5 needs measured Freeze 1c v2.")
    ceilings = preapproval.get("ceilings", {})
    _require(all(type(ceilings.get(key)) in (int, float) and math.isfinite(ceilings[key])
                 and ceilings[key] > 0 for key in ("gpu_seconds", "cpu_seconds", "cost_usd")),
             "G3 ceilings must be explicit, positive and finite.")
    _require(session["max_total_seconds"] <= ceilings["gpu_seconds"]
             and gross_ceiling_usd <= ceilings["cost_usd"], "The owned session exceeds G3 preapproval.")
    _require(not {"allocations", "freeze1c_sha256", "git_head", "created_utc"}.intersection(preapproval),
             "G3 preapproval must not fabricate measured allocation or provenance fields.")
    document = {**preapproval, "format": "deepmzyme_ec_clean30_g3_authority_v5",
                "preapproval_sha256": _sha256(preapproval_sha256, "preapproval_sha256"),
                "freeze1c_sha256": _file_sha(freeze1c),
                "allocations": {session["session_id"]: {"resource": "gpu",
                    **{key: session[key] for key in ("allocation_started_unix", "deadline_unix", "max_total_seconds")},
                    "cost_ceiling_usd": gross_ceiling_usd}},
                "git_head": source["git_head"]}
    if out.exists():
        prior = _record(out)
        _require(all(prior.get(key) == value for key, value in document.items()),
                 "The materialized G3 authority changed during recovery.")
        phase3.require_record_precedence(prior, source)
        return prior
    document["created_utc"] = provenance.utc_now()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as handle:
        json.dump(document, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    return document


def prepare_g2(config, transport, state, session, ready, manifest, probe, check_session):
    """Return reviewed preparation metadata; this callback never accesses a pod."""
    _require(transport is None, "G2 preparation metadata must be constructed locally.")
    return {"pretrain_request": dict(config["pretrain_request"]),
            "max_download_bytes": config["g2_max_download_bytes"]}


def complete_g2(config, transport, state, session, ready, manifest, probe, check_session):
    from ec_clean30 import phase3, workstation

    _require(transport is None, "Freeze 1c is built from independently preserved local receipts.")
    evidence = state["g2_evidence"]
    paths = [Path(row["local_path"]) for name, row in evidence.items()
             if name not in {"lane_serial_1", "lane_serial_2"}]
    out = Path(config["state_root"]) / "freeze1c_v2.json"
    source = ready["source"]
    preparation = state["pretrain_preparation"]
    prep_path = Path(preparation["local_path"])
    _require(_file_sha(prep_path) == preparation["sha256"], "Preserved pretraining receipt changed.")
    phase3.verify_pretrain_evidence(prep_path, source)
    if not out.exists():
        phase3.build_freeze1c(paths, out, execution_manifest_path=Path(config["execution_manifest"]),
            chain_receipt_path=workstation.session_chain_receipt_path(config),
            expected_chain=source["amendment_chain"],
            pretrain_evidence=prep_path)
    else:
        prior = _record(out)
        phase3.require_record_precedence(prior, source)
        _require({row["sha256"] for row in prior["evidence"]} == {_file_sha(path) for path in paths},
                 "G2 measurements differ from the write-once Freeze 1c v2.")
        _require(prior.get("pretrain_evidence", {}).get("sha256") == preparation["sha256"],
                 "Freeze 1c v2 retains a different pretraining receipt.")
    return {"local": str(out), "remote": f"{config['remote_campaign']}/g2_requests/freeze1c_v2.json",
            "sha256": _file_sha(out)}


def prepare_phase4(config, transport, state, session, ready, manifest, probe, check_session):
    from ec_clean30 import phase3

    _require(transport is None, "Phase-4 metadata must be constructed before remote staging.")
    _, request = _bound_record(config["phase4"], "request")
    approval_path, preapproval = _bound_record(config["phase4"], "preapproval")
    source = ready["source"]
    phase3.require_same_scientific_identity(request, source)
    freeze = state["freeze1c"]
    g3 = Path(config["state_root"]) / "g3_v5.json"
    materialize_g3(preapproval, preapproval_sha256=_file_sha(approval_path), freeze1c=Path(freeze["local"]),
                   out=g3, source=source, session=session,
                   gross_ceiling_usd=config["runpod"]["gross_ceiling_usd"])
    remote_g3 = f"{config['remote_campaign']}/g2_requests/g3_v5.json"
    _require(request.get("execution_manifest") == config["execution_manifest"],
             "The reviewed Phase-4 request names another execution manifest.")
    request.update(freeze1c=freeze["local"], g3_authority=str(g3), g3_sha256=_file_sha(g3),
                   measurement_files={row["sha256"]: row["local_path"] for row in state["g2_evidence"].values()},
                   remote_campaign=config["remote_campaign"])
    preparation = state["pretrain_preparation"]
    request["measurement_files"][preparation["sha256"]] = preparation["local_path"]
    template = dict(request["remote_request_template"])
    template.update(execution_manifest=manifest, freeze1c=freeze["remote"], g3_authority=remote_g3,
                    g3_sha256=_file_sha(g3), vm_probe=probe,
                    measurement_files={row["sha256"]: row["remote_path"] for row in state["g2_evidence"].values()})
    template["measurement_files"][preparation["sha256"]] = preparation["remote_path"]
    request["remote_request_template"] = template
    request["stage_files"] = [*request.get("stage_files", []),
                              {"local": str(g3), "remote": remote_g3, "sha256": _file_sha(g3)}]
    return request


def _stage_pipeline_files(request, transport, check_session):
    from ec_clean30 import provenance

    root = PurePosixPath(request["remote_campaign"])
    for row in request.get("stage_files", []):
        check_session()
        local, remote = Path(row["local"]), PurePosixPath(row["remote"])
        provenance.refuse_test_paths([local, str(remote)], what="Phase-4 staged inputs")
        _require(remote.is_absolute() and ".." not in remote.parts and root in remote.parents
                 and not local.is_symlink() and local.is_file() and _file_sha(local) == row["sha256"],
                 "A declared Phase-4 staging file has changed or lies outside its campaign.")
        expected = {"sha256": row["sha256"], "size_bytes": local.stat().st_size}
        existing = transport.stat_file(str(remote))
        if existing is None:
            partial = str(remote) + ".partial"
            transport.upload(local, partial)
            _require(transport.stat_file(partial) == expected, "Phase-4 staging readback differs.")
            transport.remote(["mv", "--", partial, str(remote)])
            existing = transport.stat_file(str(remote))
        _require(existing == expected, "Existing Phase-4 staging bytes differ.")


def run_phase4_pipeline(request, *, transport, check_session, record_progress):
    """The fixed amendment-2 order on one already owned pod; this function never allocates."""
    from ec_clean30 import phase3, phase4

    source = phase3.verify_execution_source(Path(request["execution_manifest"]),
        chain_receipt_path=Path(request["amendment_chain_receipt"]), expected_chain=request["expected_chain"])
    phase3.require_same_scientific_identity(request, source)
    _require(request.get("provider") == "runpod", "The v5 pipeline requires the owned Runpod session.")
    fold = request.get("fold", 0)
    _require(type(fold) is int and fold in range(5), "Invalid requested fold.")
    _require(type(request.get("run_final_clean", False)) is bool, "Final CLEAN dispatch must be an explicit boolean.")
    _stage_pipeline_files(request, transport, check_session)
    options = {"transport": transport, "check_session": check_session, "close_session": False}

    def record(name, callback, *, unit_request=None, **kwargs):
        check_session()
        result = callback(request if unit_request is None else unit_request, **options, **kwargs)
        _require(result.get("status") == "passed", f"The pipeline stopped at unverified {name}.")
        record_progress(name, result)
        return result

    if fold == 0:
        record("clean_embeddings", phase4.run_embedding_queue)
        record("clean_inner", phase4.run_clean_remote_queue)
        record("held_out_sanity", phase4.run_scientific_step, step="held_out_sanity")
        record("verify_sanity", phase4.run_scientific_step, step="verify_sanity")
    record("s1", phase4.run_s1_queue)
    record("select_s1", phase4.run_scientific_step, step="select_s1")
    record("select_s2", phase4.run_scientific_step, step="select_s2")
    if fold == 0:
        decision = record("futility", phase4.run_scientific_step, step="futility")
        phase3.require_same_scientific_identity(decision, source)
        _require(decision.get("decision") in {"continue", "stop_futility"}, "Futility returned no binding decision.")
        if decision["decision"] == "stop_futility":
            return {"status": "stop_futility", **phase3.source_identity(source)}
    record("select_s4", phase4.run_scientific_step, step="select_s4")
    record("s4", phase4.run_refit_queue)
    if request.get("run_final_clean", False):
        final = dict(request["final_clean_request"])
        allowed = {"clean_scope", "clean_inputs", "clean_forecasts", "clean_timeouts", "farms_by_unit", "farms_by_arm"}
        _require(set(final) <= allowed and final.get("clean_scope") == "final",
                 "Final CLEAN overrides may not change source, session, caps or persistence.")
        template = {**request["remote_request_template"],
                    **{key: value for key, value in final.items() if key not in {"clean_forecasts", "clean_timeouts"}}}
        record("clean_final", phase4.run_clean_remote_queue,
               unit_request={**request, **final, "remote_request_template": template})
    return {"status": "passed", "fold": fold, **phase3.source_identity(source)}


def run_v5(config: Mapping[str, Any], config_path: Path, *, authorization_path: Path, window_id: str,
           transport=None, controller=None) -> dict[str, Any]:
    """Explicit launch entry point; all publication, approval and retry gates precede provider access."""
    from ec_clean30 import phase3, workstation

    _require(config.get("provider") == "runpod" and config.get("execution_scope") == "g2_then_fold0_v5",
             "Execution v5 requires its explicit Runpod configuration.")
    workstation.prepare_session_chain_receipt(config)
    source = phase3.verify_execution_source(Path(config["execution_manifest"]),
        chain_receipt_path=workstation.session_chain_receipt_path(config),
        expected_chain=_record(Path(config["execution_manifest"]))["amendment_chain"])
    authorization = validate_launch_authorization(config, config_path, authorization_path, source)
    authorization_sha = _file_sha(authorization_path)
    ledger = FailedWindowLedger(authorization["failure_ledger"], authorization["failure_ledger_authorization_sha256"])
    reserved = False

    def before_allocation(current, path, ready):
        nonlocal reserved
        _require(_file_sha(authorization_path) == authorization_sha, "Launch authorization changed.")
        validate_launch_authorization(current, path, authorization_path, ready["source"])
        ledger.begin_window(window_id, config_sha256=_file_sha(path),
                            execution_manifest_sha256=source["execution_manifest_sha256"],
                            pass_root=Path(config["state_root"]))
        reserved = True
        return {"config_sha256": _file_sha(path), "execution_manifest_sha256": source["execution_manifest_sha256"],
                "authorization_sha256": authorization_sha, "window_id": window_id,
                "failure_ledger": str(ledger.path)}

    try:
        return workstation._run_runpod_g2(config, config_path, transport=transport, controller=controller,
            pre_allocation=before_allocation, prepare_g2=prepare_g2, complete_g2=complete_g2,
            prepare_phase4=prepare_phase4)
    finally:
        if reserved:
            state_path = Path(config["state_root"]) / "driver_state.json"
            state = _record(state_path) if state_path.exists() else {}
            terminal = verified_terminal_state(state)
            if terminal is not None:
                _require(state_path.is_file(), "A reserved launch window lacks its durable driver record.")
                closeout = state.get("closeout", {})
                preserved = bool(closeout.get("preservation_sha256")) and (
                    closeout.get("provider_absence_verified") is True or closeout.get("provider_stop_verified") is True)
                outcome = ("completed" if state.get("status") in {"phase4_complete", "stopped_futility"}
                           and preserved and not state.get("preservation_error") else "failed")
                ledger.finish_window(window_id, outcome=outcome, closeout_sha256=_file_sha(state_path),
                                      provider_terminal=terminal)


def verified_terminal_state(state: Mapping[str, Any]) -> str | None:
    """Classify the controller's actual terminal receipt; uncertain create outcomes stay open."""
    closeout, stop = state.get("closeout", {}), state.get("stop_status", {})
    owned = state.get("pod_id")
    if closeout.get("status") == "DELETED" and closeout.get("provider_absence_verified") is True:
        return "verified_absent" if owned and closeout.get("pod_id") == owned else None
    if stop.get("status") == "EXITED" and stop.get("runtime_status") == "stopped":
        return "verified_stopped" if owned and stop.get("pod_id") == owned else None
    if stop.get("status") == "CREATE_REJECTED_NO_POD" and stop.get("provider_absence_verified") is True:
        return "verified_absent" if not owned else None
    if stop.get("status") in {"NO_CREATE_ATTEMPT", "CREATE_CANCELLED_BEFORE_POST"}:
        return "not_allocated" if not owned else None
    if not state.get("start_attempted") and not owned:
        return "not_allocated"
    return None
