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
import tarfile
import time
from typing import Any, Iterator, Mapping
import uuid

FORMAT = "deepmzyme_ec_clean30_failed_launch_windows_v1"
FAILED_WINDOW_LIMIT = 2
REPO_ROOT = Path(__file__).resolve().parents[2]
TERMINAL_STATES = {"verified_stopped", "verified_absent", "not_allocated"}
AUTHORIZATION_FORMAT = "deepmzyme_ec_clean30_runpod_launch_authorization_v5"
PREAPPROVAL_FORMAT = "deepmzyme_ec_clean30_g3_preapproval_v5"
G3_AUTHORITY_FORMAT = "deepmzyme_ec_clean30_g3_authority_v5"
# A resumed fold keeps the scope, inputs and caps of its earlier G3; only the ceilings may grow, and
# its CLEAN unit ids and input pins may only gain the final scope (_authenticated_g3).
G3_FAMILY_KEYS = ("campaign_id", "scope", "allowed_folds", "plan_sha256_by_fold", "allowed_steps", "clean_unit_ids",
                  "clean_cap_profile", "science_cap_profile", "hot_work_root", "clean_inputs_sha256",
                  "clean_inputs_sha256_by_scope")
# A resume restores preserved bytes in a few verified archives, never one provider check per file.
RESTORE_BATCH_BYTES = 1024 ** 3
RESTORE_BATCH_FILES = 50_000
# The pod writes, syncs and renames every restored file; this allowance bounds one archive's remote command.
RESTORE_FILE_SECONDS = 0.1
# An unbilled copy of the inherited files into the new pass's roots (--prepare-resume), journaled per batch.
PRESTAGE_FORMAT = "deepmzyme_ec_clean30_resume_prestage_v1"
PRESTAGE_PROOF = "resume_prestage.json"
PRESTAGE_JOURNAL = "resume_prestage_journal.jsonl"
PRESTAGE_BATCH_FILES = 2000
PRESTAGE_BATCH_BYTES = 1024 ** 3
RESUME_FILE_CHECKS = ("full", "records")


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

    def _binding(self, window_id: str, config_sha256: str, execution_manifest_sha256: str,
                 pass_root: str | Path) -> dict[str, str]:
        self._window_id(window_id)
        binding = {"window_id": window_id, "config_sha256": _sha256(config_sha256, "config_sha256"),
                   "execution_manifest_sha256": _sha256(execution_manifest_sha256, "execution_manifest_sha256"),
                   "pass_root": str(_path(pass_root, "pass_root"))}
        _require(self.path != Path(binding["pass_root"]) and Path(binding["pass_root"]) not in self.path.parents,
                 "The shared ledger must not live inside a pass directory.")
        return binding

    def holds_open(self, window_id: str, *, config_sha256: str, execution_manifest_sha256: str,
                   pass_root: str | Path) -> bool:
        """Read only: whether exactly this binding is the ledger's unresolved window."""
        binding = self._binding(window_id, config_sha256, execution_manifest_sha256, pass_root)
        with self._locked():
            document = self._read()
        return any(row["status"] == "open" and all(row[key] == value for key, value in binding.items())
                   for row in document["windows"])

    def begin_window(self, window_id: str, *, config_sha256: str, execution_manifest_sha256: str,
                     pass_root: str | Path) -> dict[str, Any]:
        """Reserve before provider access, or recover exactly the same unresolved window."""
        binding = self._binding(window_id, config_sha256, execution_manifest_sha256, pass_root)
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
                                  authorization_path: Path, source: Mapping[str, Any], *,
                                  resume_files: str = "full") -> dict[str, Any]:
    """Read an explicit maintainer authorization; never create or infer one.

    resume_files selects the depth of the preserved-copy check (validate_resume). The launch gate
    runs the full two-copy check right before it reserves a window; earlier calls check the records.
    """
    from ec_clean30 import phase3, workstation

    authorization = _record(authorization_path)
    _require(_record(config_path) == dict(config), "The in-memory configuration differs from its reviewed file.")
    _require(config.get("provider") == "runpod" and config.get("execution_scope") in {"g2_then_fold0_v5", "fold_execution_v5"},
             "Execution v5 requires an explicit Runpod session scope.")
    _require(source.get("format") == phase3.EXECUTION_V5_FORMAT and source.get("published") is True,
             "Launch requires published execution v5 and amendment 2.")
    phase3.authenticated_chain(source)
    _require(authorization.get("format") == AUTHORIZATION_FORMAT
             and authorization.get("launch_authorized") is True
             and authorization.get("provider") == "runpod"
             and authorization.get("scope") == config["execution_scope"]
             and isinstance(authorization.get("approved_by"), str) and authorization["approved_by"].strip(),
             "A separate explicit maintainer launch authorization is required.")
    _require(authorization.get("config_sha256") == _file_sha(config_path)
             and authorization.get("execution_manifest_sha256") == source["execution_manifest_sha256"],
             "Launch authorization does not bind this exact reviewed execution configuration.")
    phase3.require_same_scientific_identity(authorization, source)
    fold = config.get("execution_fold", 0)
    _require(type(fold) is int and fold in range(5)
             and (fold == 0 or config["execution_scope"] == "fold_execution_v5"), "Later folds need explicit fold execution scope.")
    ceiling = authorization.get("fold0_cost_ceiling_usd" if fold == 0 else "fold_cost_ceiling_usd")
    _require(type(ceiling) in (int, float) and math.isfinite(ceiling) and ceiling > 0
             and ceiling == config.get("runpod", {}).get("gross_ceiling_usd"),
             "The fold cost ceiling must be explicitly approved and equal the controller ceiling.")
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
    carry, rows = preserved_g3(config, validate_resume(config, source, files=resume_files), source)
    carried = _allocation_totals(rows or {})
    cost = preapproval.get("ceilings", {}).get("cost_usd")
    if carry is None:
        _require(cost == ceiling, "G3 preapproval and launch cost ceiling differ.")
    else:
        # Exact, as for a fresh launch: the reviewed G3 cost is the cumulative fold total, never a looser bound.
        _require(type(cost) in (int, float) and math.isfinite(cost) and cost == carried["cost"] + ceiling,
                 "A resumed fold's G3 preapproval cost must equal the cumulative cost: every carried-forward "
                 "allocation's cost ceiling plus this launch ceiling.")
    ceilings = preapproval["ceilings"]
    _require(all(type(ceilings.get(key)) in (int, float) and math.isfinite(ceilings[key])
                 and ceilings[key] > 0 for key in ("gpu_seconds", "cpu_seconds", "cost_usd"))
             and type(config["runpod"].get("max_runtime_seconds")) in (int, float)
             and 0 < config["runpod"]["max_runtime_seconds"] <= ceilings["gpu_seconds"] - carried["gpu"]
             and carried["cpu"] <= ceilings["cpu_seconds"],
             "The controller allocation must fit the positive G3 preapproved ceilings before launch; "
             "a resumed fold's ceilings must also cover every carried-forward allocation.")
    _require(request.get("fold", 0) == config.get("execution_fold", 0), "Reviewed configuration and requested fold differ.")
    workstation.validate_phase4_plan(config, source, request, preapproval)
    return authorization


def _allocation_totals(allocations: Mapping[str, Any]) -> dict[str, float]:
    """Sum allocations in their stored order, with the row checks of phase4.verify_authority."""
    totals = {"gpu": 0.0, "cpu": 0.0, "cost": 0.0}
    for row in allocations.values():
        _require(isinstance(row, dict) and row.get("resource") in ("gpu", "cpu")
                 and all(type(row.get(key)) in (int, float) and math.isfinite(row[key]) for key in
                         ("allocation_started_unix", "deadline_unix", "max_total_seconds", "cost_ceiling_usd"))
                 and row["max_total_seconds"] > 0 and row["cost_ceiling_usd"] >= 0
                 and 0 < row["deadline_unix"] - row["allocation_started_unix"] <= row["max_total_seconds"],
                 "A carried-forward G3 allocation is malformed.")
        totals[row["resource"]] += row["max_total_seconds"]
        totals["cost"] += row["cost_ceiling_usd"]
    return totals


def _clean_scope(document: Mapping[str, Any]) -> tuple[set, dict]:
    pins = document.get("clean_inputs_sha256_by_scope")
    pins = dict(pins) if isinstance(pins, dict) else {"fold0_inner": document.get("clean_inputs_sha256")}
    return set(document.get("clean_unit_ids") or ()), pins


def _authenticated_g3(reference: Any, source: Mapping[str, Any], preapproval: Mapping[str, Any]) -> dict[str, Any]:
    """Read a preserved earlier G3 authority only through its recorded digest, format, identity and family."""
    from ec_clean30 import phase3

    _require(isinstance(reference, dict) and {"path", "sha256"} <= set(reference) <= {"path", "sha256", "sessions"},
             "A preserved G3 reference names exactly its path and digest.")
    path = Path(reference["path"])
    _require(path.is_absolute(), "A preserved G3 authority needs an absolute path.")
    document = _record(path)
    _require(_file_sha(path) == _sha256(reference["sha256"], "preserved G3 sha256"),
             "The preserved prior G3 authority changed.")
    _require(document.get("format") == G3_AUTHORITY_FORMAT and document.get("status") == "approved"
             and isinstance(document.get("allocations"), dict) and bool(document["allocations"]),
             "A preserved prior G3 must be an approved materialized v5 authority with its allocations.")
    _sha256(document.get("preapproval_sha256"), "preserved G3 preapproval_sha256")
    phase3.require_record_precedence(document, source)
    (old_units, old_pins), (new_units, new_pins) = _clean_scope(document), _clean_scope(preapproval)
    # A later session may add final CLEAN to the fold; it never drops or changes a CLEAN scope already bound.
    _require(all(document.get(key) == preapproval.get(key) for key in G3_FAMILY_KEYS
                 if key not in ("clean_unit_ids", "clean_inputs_sha256", "clean_inputs_sha256_by_scope"))
             and old_units <= new_units and old_pins.items() <= new_pins.items(),
             "A resumed fold's G3 preapproval must keep the prior G3 scope, inputs and caps; only its ceilings "
             "and an added CLEAN scope may grow.")
    _allocation_totals(document["allocations"])
    return document


def preserved_g3(config: Mapping[str, Any], resume: Mapping[str, Any] | None,
                 source: Mapping[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """The G3 allocations a resumed fold must carry forward, or (None, None) when there are none.

    Only sessions already admitted to the preserved Phase-4 root are carried, because phase4._runtime
    requires exactly those; before any admission the per-session cost equality stays unchanged.
    The bound prior driver state names the digest of the G3 its session materialized (or inherited
    at its own resume); its own G3 must lie in its state root and match every preserved Phase-4 unit
    request of that session. Admitted work without any G3 stops for a decision. Returns the reference
    {path, sha256, sessions} and the carried allocation rows in the G3's stored order.
    """
    fold = config.get("execution_fold", 0)
    if resume is None or resume["config"].get("execution_fold", 0) != fold:
        return None, None
    state = resume["state"]
    own, inherited = state.get("g3_authority"), state.get("resume_g3")
    record = next((row for row in resume["inventories"]
                   if row["execution_root"].endswith(f"/runs/fold{fold}/phase4")), None)
    files = {entry["path"] for entry in record["files"]} if record else set()
    admitted = (_record(Path(record["primary_root"]) / "execution_state.json").get("allocations", {})
                if "execution_state.json" in files else {})
    _require(isinstance(admitted, dict), "Preserved Phase-4 allocations must be an object.")
    reference = own or inherited
    if reference is None:
        _require(not admitted, "Preserved Phase-4 work has no authenticated G3 authority; stop for a decision.")
        return None, None
    if not admitted:
        return None, None
    _, preapproval = _bound_record(config["phase4"], "preapproval")
    document = _authenticated_g3(reference, source, preapproval)
    if own:
        _require(Path(own["path"]) == Path(resume["config"]["state_root"]) / "g3_v5.json",
                 "The prior session's G3 must be the one materialized in its own state root.")
        prefix = f"queue/{state['session_id']}/"
        for name in sorted(files):
            if name.startswith(prefix) and name.endswith(".json") and "/" not in name[len(prefix):]:
                _require(_record(Path(record["primary_root"]) / name).get("g3_sha256") == own["sha256"],
                         "A preserved Phase-4 request of the prior session names another G3 authority.")
    for session_id, row in admitted.items():
        recorded = document["allocations"].get(session_id, {})
        _require(isinstance(row, dict) and recorded.get("allocation_started_unix") == row.get("started_unix")
                 and recorded.get("max_total_seconds", 0) >= row.get("max_total_seconds", 0),
                 "The preserved prior G3 omits a session already admitted to Phase 4.")
    rows = {session_id: row for session_id, row in document["allocations"].items() if session_id in admitted}
    return {"path": reference["path"], "sha256": reference["sha256"], "sessions": list(rows)}, rows


def materialize_g3(preapproval: Mapping[str, Any], *, preapproval_sha256: str, freeze1c: Path,
                   out: Path, source: Mapping[str, Any], session: Mapping[str, Any],
                   gross_ceiling_usd: float, prior: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Bind an already approved numerical envelope to measured G2 and the existing allocation.

    prior is the reference (preserved_g3) to the G3 of earlier sessions of a resumed fold and the
    sessions already admitted to its Phase-4 root. Exactly those allocations are carried forward
    unchanged, so phase4._runtime still finds them, and the ceilings cover the cumulative total.
    """
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
    _require(not {"allocations", "freeze1c_sha256", "git_head", "created_utc"}.intersection(preapproval),
             "G3 preapproval must not fabricate measured allocation or provenance fields.")
    carried = {}
    if prior is not None:
        earlier = _authenticated_g3(prior, source, preapproval)["allocations"]
        sessions = prior.get("sessions")
        _require(isinstance(sessions, list) and bool(sessions) and len(set(sessions)) == len(sessions)
                 and all(isinstance(name, str) and name in earlier for name in sessions),
                 "A resumed fold carries only the admitted sessions of its preserved G3.")
        carried = {name: row for name, row in earlier.items() if name in sessions}
    _require(session["session_id"] not in carried, "A resumed session needs its own new allocation identifier.")
    allocations = {**carried, session["session_id"]: {"resource": "gpu",
                   **{key: session[key] for key in ("allocation_started_unix", "deadline_unix", "max_total_seconds")},
                   "cost_ceiling_usd": gross_ceiling_usd}}
    # phase4.verify_authority sums every allocation, in the sorted order it reads them back, against these ceilings.
    totals = _allocation_totals(dict(sorted(allocations.items())))
    _require(totals["gpu"] <= ceilings["gpu_seconds"] and totals["cpu"] <= ceilings["cpu_seconds"]
             and totals["cost"] <= ceilings["cost_usd"],
             "The owned session exceeds G3 preapproval; a resumed fold's preapproval ceilings must cover every "
             "carried-forward allocation plus this session.")
    document = {**preapproval, "format": G3_AUTHORITY_FORMAT,
                "preapproval_sha256": _sha256(preapproval_sha256, "preapproval_sha256"),
                "freeze1c_sha256": _file_sha(freeze1c), "allocations": allocations,
                "git_head": source["git_head"]}
    if out.exists():
        existing = _record(out)
        _require(all(existing.get(key) == value for key, value in document.items()),
                 "The materialized G3 authority changed during recovery.")
        phase3.require_record_precedence(existing, source)
        return existing
    document["created_utc"] = provenance.utc_now()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as handle:
        json.dump(document, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    return document


def validate_resume(config, source, *, files="full"):
    """Authenticate a terminal session and both preserved copies before any allocation.

    files="full" also reads every preserved member in both copies (size and SHA-256); the launch
    gate runs it right before it reserves a window. files="records" keeps every other check:
    receipts, identity, roots, devices, member names, the remote inventory, the small G2 and
    preparation records and the attempt histories. The result carries the attempt maps of the
    preserved G2 and Phase-4 roots, so an unsupported history is refused before any provider call.
    """
    from ec_clean30 import phase3

    _require(files in RESUME_FILE_CHECKS, "Resume validation checks either the full copies or the records.")
    spec = config.get("resume")
    if spec is None:
        _require(config.get("execution_fold", 0) == 0, "Later folds require preserved fold-0 records.")
        return None
    _require(isinstance(spec, dict) and set(spec) == {"config_path", "config_sha256", "state_path", "state_sha256",
                                                    "preservation_path", "preservation_sha256"},
             "Resume needs exact prior configuration, driver state and preservation bindings.")
    _, prior = _bound_record(spec, "config")
    _, state = _bound_record(spec, "state")
    _, proof = _bound_record(spec, "preservation")
    _require(state.get("closed") is True and verified_terminal_state(state) in {"verified_absent", "verified_stopped"}
             and state.get("config_sha256") == spec["config_sha256"]
             and state.get("closeout", {}).get("preservation_sha256") == spec["preservation_sha256"]
             and not state.get("preservation_error"), "Resume requires verified terminal closeout and complete preservation.")
    _require(state.get("status") != "stopped_futility", "A binding futility stop cannot resume.")
    for key in ("hardware_profile", "runtime_environments", "remote_repo", "remote_campaign", "remote_python", "embedding"):
        _require(config.get(key) == prior.get(key) and key in config, f"Resume changes pinned {key}.")
    for key in ("gpu_id", "image", "runtime_environment"):
        _require(config["runpod"].get(key) == prior["runpod"].get(key) and key in config["runpod"],
                 f"Resume changes the GPU type or pinned environment ({key}).")
    _require(config["state_root"] != prior["state_root"]
             and config["runpod"]["state_root"] != prior["runpod"]["state_root"],
             "Resume needs new driver and controller state roots; prior records stay immutable.")
    for current, previous in ((config, prior), (config["phase4"], prior["phase4"])):
        if "request_path" in current:
            _, current = _bound_record(current, "request")
            _, previous = _bound_record(previous, "request")
        for key in ("durable_root", "mirror_root"):
            left, right = Path(current[key]), Path(previous[key])
            _require(left != right and left not in right.parents and right not in left.parents,
                     "Resume destinations must preserve the prior independent copies unchanged.")
    previous_manifest = _record(prior["execution_manifest"])
    phase3.require_same_scientific_identity(previous_manifest, source)
    preparation = state.get("pretrain_preparation")
    if preparation:
        phase3.require_same_scientific_identity(preparation["record"], source)
        _require(_file_sha(Path(preparation["local_path"])) == preparation["sha256"], "Preserved F6 preparation changed.")
    if config.get("execution_fold", 0) > 0:
        from ec_clean30 import workstation
        _require(preparation and state.get("completed") == [name for _, name in workstation.measurement_queue(source)],
                 "Later folds require all preserved G2 measurements.")
    for item in state.get("g2_evidence", {}).values():
        _require(_file_sha(Path(item["local_path"])) == item["sha256"], "Preserved G2 evidence changed.")
        phase3.require_record_precedence(_record(item["local_path"]), source)
    rows = proof.get("receipts") if proof.get("format") == "deepmzyme_ec_clean30_runpod_preservation_set_v1" else [
        {"path": spec["preservation_path"], "sha256": spec["preservation_sha256"]}]
    _require(isinstance(rows, list) and rows, "Resume preservation has no inventories.")
    inventories = []
    for row in rows:
        path = Path(row["path"])
        record = _record(path)
        _require(_file_sha(path) == row["sha256"] and record.get("session_id") == state["session_id"]
                 and record.get("pod_id") == state["pod_id"], "Resume inventory is changed or foreign.")
        roots = [Path(record[key]) for key in ("primary_root", "mirror_root")]
        _require(all(root.is_dir() and root.resolve() == root for root in roots)
                 and roots[0].stat().st_dev != roots[1].stat().st_dev, "Resume requires independent preserved devices.")
        seen = set()
        for entry in record["files"]:
            relative = PurePosixPath(entry["path"])
            _require(not relative.is_absolute() and ".." not in relative.parts and str(relative) not in seen,
                     "Unsafe or duplicate preserved member.")
            seen.add(str(relative))
            for root in roots if files == "full" else ():
                file = root / str(relative)
                _require(file.resolve() == file and file.is_file() and file.stat().st_size == entry["size_bytes"]
                         and _file_sha(file) == entry["sha256"], "A preserved result is missing or changed; never replace it.")
        inventory = _record(roots[0] / record["remote_inventory_path"])
        members = {entry["path"]: entry for entry in record["files"]}
        _require(all(entry == members.get(entry.get("path")) for entry in inventory.get("files", []))
                 and inventory.get("session_id") == state["session_id"], "Preservation omits the remote inventory.")
        _require(inventory.get("pending_transfer") is None
                 and (not inventory.get("pending_unit") or inventory["pending_unit"].get("persisted") is True),
                 "Resume refuses unacknowledged work.")
        inventories.append(record)
    fold = config.get("execution_fold", 0)
    g2 = next((record for record in inventories if record["execution_root"].endswith("/fold0/g2")), None)
    phase = next((record for record in inventories
                  if record["execution_root"].endswith(f"/runs/fold{fold}/phase4")), None)
    # Directory listings of the preserved primary copies, as the restore has always read them.
    return {"config": prior, "state": state, "inventories": inventories,
            "preservation_sha256": spec["preservation_sha256"],
            "g2_attempts": None if g2 is None else g2_resume_attempts(Path(g2["primary_root"]), source),
            "attempts": {} if phase is None else resume_attempts(Path(phase["primary_root"]))}


def g2_unit_paths(action, name):
    if action == "timing-unit":
        return f"ecc30timing_{name}", "g2_complete.json"
    if action == "lane-gate":
        return "ecc30timing_lane_batch", "lane_gate.json"
    return f"ecc30timing_embed_{name}", "embedding_complete.json"


def g2_resume_attempts(root, source):
    from ec_clean30 import workstation
    result = {}
    queue = [("pretrain-prepare", "pretrain"), *workstation.measurement_queue(source)]
    for action, name in queue:
        prefix, complete = (("ecc30_pretrain_fold0", "pretrain_complete.json") if name == "pretrain"
                            else g2_unit_paths(action, name))
        partial = {"timing-unit": "g2_timing.json", "lane-gate": "batch_timing.json",
                   "embed-timing": "embedding_timing.json", "pretrain-prepare": "pretrain_preparation.json"}[action]
        numbers = set()
        for base in (Path(root) / "runs", Path(root) / "commands"):
            for path in base.glob(prefix + "_attempt*"):
                # The runner record a G2 unit leaves beside its run directory counts as that attempt.
                match = re.fullmatch(re.escape(prefix) + r"_attempt([12])(?:\.json|\.log|\.runner\.json)?",
                                     path.name)
                _require(match is not None, "G2 resume has an unsupported retry history.")
                numbers.add(int(match[1]))
        terminal = [n for n in sorted(numbers) if any((Path(root) / "runs" / f"{prefix}_attempt{n}" / file).is_file()
                                                      for file in (complete, partial))]
        _require(len(terminal) <= 1, "G2 has multiple results; no result selection is permitted.")
        if terminal:
            result[name] = terminal[0]
        elif numbers:
            _require(numbers == {1}, "The identical G2 retry has already been used.")
            result[name] = 2
    return result


def inherited_artifacts(config):
    """Read the authenticated prior inventory once per preservation verification."""
    spec = config.get("resume")
    if not spec:
        return {}
    _, proof = _bound_record(spec, "preservation")
    rows = proof.get("receipts", [{"path": spec["preservation_path"], "sha256": spec["preservation_sha256"]}])
    artifacts = {}
    for row in rows:
        path = Path(row["path"])
        _require(_file_sha(path) == row["sha256"], "Prior preservation receipt changed.")
        record = _record(path)
        for entry in record["files"]:
            name = str(PurePosixPath(record["execution_root"]) / entry["path"])
            _require(name not in artifacts, "Duplicate inherited artifact.")
            artifacts[name] = entry["sha256"]
    return artifacts


def resume_attempts(root):
    """Keep every terminal result; allow exactly one identical retry of a lost unit."""
    attempts = {}
    for unit in (Path(root) / "runs").glob("*"):
        directories = sorted(unit.glob("attempt[0-9]*"))
        if not directories:
            attachment = unit / "attachment_result.json"
            _require(not (unit / "complete.json").exists() or attachment.is_file(),
                     "A finished unit lost its preserved result; refuse replacement.")
            continue
        numbers = [int(path.name.removeprefix("attempt")) for path in directories]
        _require(numbers == list(range(1, max(numbers) + 1)), "Resume attempt history has a gap.")
        results = [path for path in directories if (path / "unit_result.json").exists()]
        _require(len(results) <= 1, "Multiple preserved results cannot be selected or replaced.")
        complete = unit / "complete.json"
        if complete.exists():
            _require(len(results) == 1, "A finished unit lost its preserved result; refuse replacement.")
            attempt = _record(complete).get("attempt", 1)
            _require(attempt == int(results[0].name.removeprefix("attempt")), "Completed attempt differs from its result.")
        elif results:
            attempt = int(results[0].name.removeprefix("attempt"))
        else:
            _require(max(numbers) == 1, "The identical planned retry has already been used; stop for a decision.")
            _require(not unit.name.startswith("ecc30_science_"), "Partial scientific outputs require diagnosis before resume.")
            attempt = 2
        attempts[unit.name] = attempt
    commands = {}
    for command in (Path(root) / "commands").glob("*_attempt*.json"):
        match = re.fullmatch(r"(.+)_attempt([12])\.json", command.name)
        _require(match is not None, "An unsupported attempt command cannot be resumed.")
        commands.setdefault(match[1], set()).add(int(match[2]))
    for name, numbers in commands.items():
        if name in attempts:
            _require(max(numbers) <= attempts[name], "A later attempt cannot replace a preserved result.")
        elif not (Path(root) / "runs" / name).exists():
            _require(numbers == {1}, "The identical planned retry has already been used.")
            attempts[name] = 2
    return attempts


def _state_document_sha(document: Mapping[str, Any]) -> str:
    """The digest of the bytes workstation._write gives this document."""
    text = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    return hashlib.sha256(text.encode()).hexdigest()


def _resume_outputs(config, resume):
    """The new pass's preservation roots: G2, this fold's Phase 4 and one pair per other preserved root."""
    _, reviewed = _bound_record(config["phase4"], "request")
    outputs = [{"execution_root": f"{config['remote_campaign']}/runs/fold0/g2",
                "primary_root": config["durable_root"], "mirror_root": config["mirror_root"]},
               {"execution_root": reviewed["remote_request_template"]["execution_root"],
                "primary_root": reviewed["durable_root"], "mirror_root": reviewed["mirror_root"]}]
    for record in resume["inventories"]:
        remote = record["execution_root"]
        if remote in {item["execution_root"] for item in outputs}:
            continue
        suffix = "_preserved_" + PurePosixPath(remote).parent.name
        outputs.append({"execution_root": remote, "primary_root": reviewed["durable_root"] + suffix,
                        "mirror_root": reviewed["mirror_root"] + suffix})
    return outputs, reviewed


def plan_restore(config, resume, source):
    """The complete local restore plan of an authenticated resume; no transport call and no write.

    Archived bytes, pointer snapshots, execution states and markers are read from each receipt's
    mirror copy; every staged byte is hashed against its preserved digest again when archived and
    once more on the pod. The launch gate runs this plan before it reserves a window, so every
    local restore refusal comes before any provider access.
    """
    from ec_clean30 import phase3, phase4, workstation

    # Authenticate the fold's earlier G3 before any upload; Phase-4 preparation carries it forward.
    g3, _ = preserved_g3(config, resume, source)
    campaign_root = PurePosixPath(config["remote_campaign"])
    staged, active = {}, []
    outputs, reviewed = _resume_outputs(config, resume)

    def add(local, remote, digest):
        path = PurePosixPath(remote)
        _require(campaign_root in path.parents and ".." not in path.parts, "Resume output escapes the campaign.")
        previous = staged.get(str(path))
        _require(previous is None or previous["sha256"] == digest, "Preserved output snapshots disagree.")
        staged[str(path)] = {"local": str(local), "remote": str(path), "sha256": digest}

    for record in resume["inventories"]:
        local = Path(record["mirror_root"])
        remote = PurePosixPath(record["execution_root"])
        members = {entry["path"]: entry for entry in record["files"]}
        for entry in record["files"]:
            if entry["path"] == "execution_state.json" or entry["path"].startswith("closeout/"):
                continue  # New allocation bookkeeping; the original stays in the immutable inventory.
            add(local / entry["path"], remote / entry["path"], entry["sha256"])
            if Path(entry["path"]).name == phase3.PRETRAIN_ATTACHMENT_MARKER:
                marker = phase3.verify_preserved_pretrain_attachment(local / entry["path"],
                    original_attempt=remote / Path(entry["path"]).parent)
                finished = any(((local / entry["path"]).parent / name).is_file()
                               for name in ("pretrain_preparation.json", "attachment_result.json"))
                for file in marker["files"] if finished else []:
                    add((local / entry["path"]).parent / "attachment_snapshot" / file["path"],
                        PurePosixPath(marker["identity"]["view_root"]) / "whole_protein_pretrain" / file["path"], file["sha256"])
            if Path(entry["path"]).name == phase4.SCIENTIFIC_PENDING:
                marker = phase4.verify_preserved_scientific_outputs(local / entry["path"],
                    original_attempt=remote / Path(entry["path"]).parent)
                # Out-of-root pointers are immutable. The live step chain is already in the inventory.
                for file in marker["files"]:
                    if file["exists"] and remote not in PurePosixPath(file["path"]).parents:
                        add((local / entry["path"]).parent / file["snapshot"], file["path"], file["sha256"])
        execution = local / "execution_state.json"
        if execution.exists():
            preserved = members.get("execution_state.json")
            _require(preserved is not None and _file_sha(execution) == preserved["sha256"],
                     "A preserved execution state differs from its receipt.")
            prior = _record(execution)
            _require(not prior.get("pending_transfer"), "Resume cannot discard a pending transfer.")
            target, digest = execution, preserved["sha256"]
            if prior.get("active_child"):
                # The bound terminal provider receipt proves that this old child cannot still run.
                prior.update(owner=None, active_child=None)
                target = Path(config["state_root"]) / ("resume_" + hashlib.sha256(str(remote).encode()).hexdigest() + ".json")
                digest = _file_sha(target) if target.exists() else _state_document_sha(prior)
                active.append((target, prior, digest))
            add(target, remote / "execution_state.json", digest)
    g2 = next((record for record in resume["inventories"] if record["execution_root"].endswith("/fold0/g2")), None)
    _require(g2 is not None and resume.get("g2_attempts") is not None, "Resume preservation lacks its G2 root.")
    evidence = {}
    members = {entry["path"]: entry for entry in g2["files"]}
    for action, name in workstation.measurement_queue(source):
        prefix, terminal = g2_unit_paths(action, name)
        relative = f"runs/{prefix}_attempt{resume['g2_attempts'].get(name, 1)}/{terminal}"
        path = Path(g2["primary_root"]) / relative
        if path.is_file():
            copy = Path(g2["mirror_root"]) / relative
            digest = _file_sha(copy)
            _require(members.get(relative, {}).get("sha256") == digest,
                     "Preserved G2 evidence differs from its receipt.")
            phase3.require_record_precedence(_record(copy), source)
            # The recorded path stays the primary copy, as earlier sessions recorded it.
            evidence[name] = {"local_path": str(path), "sha256": digest,
                              "remote_path": str(PurePosixPath(g2["execution_root"]) / relative)}
    # The per-row checks of the restore itself; a released execution state is written by the restore.
    released = {str(target) for target, _, _ in active}
    _restore_batches(config, [row for row in staged.values() if row["local"] not in released])
    return {"outputs": outputs, "reviewed": reviewed, "rows": list(staged.values()), "g3": g3,
            "attempts": resume["attempts"], "g2_attempts": resume["g2_attempts"],
            "active_child_states": active, "g2_evidence": evidence}


def restore_session(config, transport, state, state_path, source, check_session):
    """Restore authenticated bytes at their original paths, without replacing a result."""
    from ec_clean30 import workstation

    resume = validate_resume(config, source, files="records")
    if resume is None:
        return
    if state.get("resume_restored"):
        _require(state["resume_restored"] == resume["preservation_sha256"], "Restored session binding changed.")
        return
    plan = plan_restore(config, resume, source)
    outputs, reviewed = plan["outputs"], plan["reviewed"]
    state["resume_output_roots"] = outputs
    state["phase4"] = {"preservation_roots": outputs, "max_download_bytes": reviewed["max_download_bytes"],
                       "host_state_root": reviewed["host_state_root"]}
    transport.stop_controller.register_output_roots(outputs)
    workstation._write(state_path, state)
    for target, document, digest in plan["active_child_states"]:
        if not target.exists():
            workstation._write(target, document)
        _require(_file_sha(target) == digest, "A released execution state changed during restore.")
    previous_fold = resume["config"].get("execution_fold", 0)
    if config.get("execution_fold", 0) != previous_fold:
        old = resume["config"]
        view = f"{config['remote_campaign']}/views/fold{previous_fold}"
        if transport.read_json(f"{view}/view_manifest.json", optional=True) is None:
            item = next(item for item in old["stage"] if item["remote"] == old["view_archive"])
            workstation._stage_file(config, transport, item)
            transport.remote([config["remote_python"], f"{config['remote_repo']}/scripts/run_ec_clean30_phase3.py",
                "unpack-view", "--archive", old["view_archive"], "--sha256", old["view_archive_sha256"],
                "--destination", view, "--fold", str(previous_fold)], timeout=300)
    _restore_files(config, plan["rows"], transport, check_session)
    prior = resume["state"]
    for key in ("completed", "g2_evidence", "pretrain_preparation"):
        if key in prior:
            state[key] = prior[key]
    state["resume_g2_attempts"] = plan["g2_attempts"]
    for name, row in plan["g2_evidence"].items():
        state.setdefault("g2_evidence", {})[name] = row
    state["completed"] = [name for _, name in workstation.measurement_queue(source) if name in state.get("g2_evidence", {})]
    state.update(resume_restored=resume["preservation_sha256"], resume_attempts=plan["attempts"], resume_g3=plan["g3"])
    workstation._write(state_path, state)

def _resume_receipts(spec):
    """The bound preservation receipts in their stored order, as validate_resume reads them."""
    _, proof = _bound_record(spec, "preservation")
    rows = proof.get("receipts") if proof.get("format") == "deepmzyme_ec_clean30_runpod_preservation_set_v1" else [
        {"path": spec["preservation_path"], "sha256": spec["preservation_sha256"]}]
    _require(isinstance(rows, list) and rows and all(isinstance(row, dict) and {"path", "sha256"} <= set(row)
                                                     for row in rows), "Resume preservation has no inventories.")
    return rows


def _prestaged_member(name):
    """Members a pre-stage copies: exactly those the restore puts back from each root's own inventory."""
    return name != "execution_state.json" and not name.startswith("closeout/")


def _prestage_journal(path, preservation_sha256):
    """(execution root, primary root, mirror root, member) -> (size, digest, primary print, mirror print).

    Only rows of this boot count: a witness from before a restart is void, and its files are checked
    again. A torn last line from an interrupted pre-stage is ignored likewise; a later row wins.
    """
    from ec_clean30 import workstation

    done = {}
    current = workstation.boot_id()
    if current is None or not path.is_file():
        return done
    for line in path.read_bytes().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not (isinstance(row, dict) and row.get("format") == PRESTAGE_FORMAT
                and row.get("preservation_sha256") == preservation_sha256 and row.get("boot_id") == current
                and isinstance(row.get("files"), list)):
            continue
        for item in row["files"]:
            parsed = workstation.witness_item(item)
            if parsed is not None:
                done[(row.get("execution_root"), row.get("primary_root"), row.get("mirror_root"), parsed[0])] = parsed[1:]
    return done


def _append_line(path, row):
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _prestage_target(root, name):
    target = root / name
    _require(root in target.resolve().parents and not target.is_symlink(), "A pre-stage target escapes its root.")
    return target


def _unchanged_since(witness, paths, entry):
    """Both copies still carry the fingerprints recorded when this boot verified them as this digest."""
    from ec_clean30 import workstation

    if witness is None or witness[:2] != (entry["size_bytes"], entry["sha256"]):
        return False
    current = [workstation.copy_fingerprint(path) for path in paths]
    return all(value is not None for value in current) and tuple(tuple(value) for value in current) == witness[2:]


def _prestage_copy(source, targets, entry):
    """Copy one preserved member into both new copies; an existing file is kept only if it is identical."""
    existing = [target for target in targets if os.path.lexists(target)]
    for target in existing:
        _require(target.is_file() and not target.is_symlink() and target.stat().st_size == entry["size_bytes"]
                 and _file_sha(target) == entry["sha256"],
                 "An existing pre-staged file differs from its preserved digest; it is never replaced.")
    missing = [target for target in targets if target not in existing]
    if not missing:
        return False
    partials = []
    try:
        for target in missing:
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name("." + target.name + ".prestage-partial")
            _require(not partial.is_symlink(), "Unexpected pre-stage partial path.")
            partial.unlink(missing_ok=True)
            partials.append(partial)
        digest, size = hashlib.sha256(), 0
        handles = [partial.open("xb") for partial in partials]
        try:
            with source.open("rb") as reader:
                _require(stat.S_ISREG(os.fstat(reader.fileno()).st_mode), "A preserved source is not a regular file.")
                for chunk in iter(lambda: reader.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
                    for handle in handles:
                        handle.write(chunk)
            for handle in handles:
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            for handle in handles:
                handle.close()
        _require(size == entry["size_bytes"] and digest.hexdigest() == entry["sha256"],
                 "A preserved source differs from its receipt; nothing is pre-staged from it.")
        for partial, target in zip(partials, missing):
            _require(not os.path.lexists(target), "A pre-stage target appeared during the copy.")
            os.replace(partial, target)
        partials = []
    finally:
        for partial in partials:
            partial.unlink(missing_ok=True)
    return True


def prestage_resume(config, config_path, *, source):
    """Before any launch: copy the inherited preserved files into the new pass's two copies.

    Bytes are read from each receipt's mirror copy, hashed against the receipt, written to both new
    copies and, per batch, synced and read back before the batch is journaled with this boot and the
    stat fingerprint of both copies. A file is kept without reading it again only while a witness of
    this boot still matches both copies; any other existing copy is hashed again. The earlier roots
    are only read. An earlier proof is removed first, so only a finished run leaves one; it binds the
    preservation set, every receipt and the journal. Closeout skips its billed hash and copy only for
    a file whose single witnessed digest is the pod's own and whose copies are unchanged since, and
    the full two-copy check before deletion is unchanged.
    """
    from ec_clean30 import workstation

    spec = config.get("resume")
    _require(spec is not None, "Pre-staging needs a resume configuration.")
    _require(source.get("published") is True, "Pre-staging needs the published execution source.")
    _require(_record(config_path) == dict(config), "The in-memory configuration differs from its reviewed file.")
    state_root = Path(config["state_root"])
    state_root.mkdir(parents=True, exist_ok=True)
    with (state_root / "driver.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise LaunchWindowError("Another process holds this pass's driver lock.") from exc
        state_path = state_root / "driver_state.json"
        _require(not (state_path.exists() and _record(state_path).get("start_attempted")),
                 "Pre-staging runs only before this pass attempts a start.")
        # A refused or interrupted run must not leave an earlier proof binding its files.
        (state_root / PRESTAGE_PROOF).unlink(missing_ok=True)
        _sync_directory(state_root)
        boot = workstation.boot_id()
        _require(boot is not None, "Pre-staging needs this machine's boot identity.")
        resume = validate_resume(config, source, files="records")
        receipts = _resume_receipts(spec)
        _require(len(receipts) == len(resume["inventories"]), "Resume receipts and inventories differ.")
        outputs, request = _resume_outputs(config, resume)
        by_root = {row["execution_root"]: row for row in outputs}
        journal = state_root / PRESTAGE_JOURNAL
        done = _prestage_journal(journal, spec["preservation_sha256"])
        plan, pending = [], {}
        for receipt, record in zip(receipts, resume["inventories"]):
            output = by_root[record["execution_root"]]
            targets = [Path(output["primary_root"]), Path(output["mirror_root"])]
            for root in targets:
                _require(root.is_absolute() and root.resolve() == root, "Pre-stage roots must be canonical.")
                root.mkdir(parents=True, exist_ok=True)
            _require(targets[0].stat().st_dev != targets[1].stat().st_dev,
                     "Pre-staging needs the new pass's two independent devices.")
            members = [entry for entry in record["files"] if _prestaged_member(entry["path"])]
            plan.append((receipt, record, output, targets, members))
            for entry in members:
                for root in targets:
                    if not os.path.lexists(root / entry["path"]):
                        device = root.stat().st_dev
                        pending[device] = pending.get(device, 0) + entry["size_bytes"]
        # The launch gate's own storage rule must still hold once every remaining byte is staged.
        for device, free, required in workstation.preservation_storage(config, source, request):
            _require(free - pending.get(device, 0) >= required,
                     "Pre-staging would leave too little space for the launch's preservation allowances.")
        timings = {"started_unix": time.time(), "roots": []}
        rows = []
        for receipt, record, output, targets, members in plan:
            source_root = Path(record["mirror_root"])
            copied = kept = 0
            started = time.monotonic()
            batch, size = [], 0

            def flush(batch):
                os.sync()
                items = []
                for entry in batch:
                    paths = [root / entry["path"] for root in targets]
                    before = [workstation.copy_fingerprint(path) for path in paths]
                    for path in paths:
                        _require(path.stat().st_size == entry["size_bytes"] and _file_sha(path) == entry["sha256"],
                                 "A pre-staged copy does not read back as its preserved digest.")
                    # A copy that changed while it was read back gets no witness and is never kept unread.
                    _require(all(value is not None for value in before)
                             and before == [workstation.copy_fingerprint(path) for path in paths],
                             "A pre-staged copy changed while it was read back.")
                    items.append([entry["path"], entry["size_bytes"], entry["sha256"], *before])
                _append_line(journal, {"format": PRESTAGE_FORMAT, "preservation_sha256": spec["preservation_sha256"],
                                       "execution_root": record["execution_root"], "primary_root": str(targets[0]),
                                       "mirror_root": str(targets[1]), "boot_id": boot, "files": items})

            for entry in members:
                paths = [_prestage_target(root, entry["path"]) for root in targets]
                key = (record["execution_root"], str(targets[0]), str(targets[1]), entry["path"])
                if _unchanged_since(done.get(key), paths, entry):
                    kept += 1
                    continue
                if _prestage_copy(_prestage_target(source_root, entry["path"]), paths, entry):
                    copied += 1
                else:
                    kept += 1
                batch.append(entry)
                size += entry["size_bytes"]
                if len(batch) >= PRESTAGE_BATCH_FILES or size >= PRESTAGE_BATCH_BYTES:
                    flush(batch)
                    batch, size = [], 0
            if batch:
                flush(batch)
            timings["roots"].append({"execution_root": record["execution_root"], "files": len(members),
                                     "copied": copied, "kept": kept, "seconds": time.monotonic() - started})
            rows.append({"execution_root": record["execution_root"], "receipt_path": receipt["path"],
                         "receipt_sha256": receipt["sha256"], "primary_root": str(targets[0]),
                         "mirror_root": str(targets[1]), "files": len(members),
                         "bytes": sum(entry["size_bytes"] for entry in members)})
        proof = {"format": PRESTAGE_FORMAT, "preservation_path": spec["preservation_path"],
                 "preservation_sha256": spec["preservation_sha256"], "roots": rows, "boot_id": boot,
                 "journal_sha256": _file_sha(journal) if journal.is_file() else None}
        workstation._write(state_root / PRESTAGE_PROOF, proof)
        _sync_directory(state_root)
        _timing_record(state_root / "prestage_timings.json", "prestage", timings)
        return {"status": "prestaged", "proof": str(state_root / PRESTAGE_PROOF),
                "proof_sha256": _file_sha(state_root / PRESTAGE_PROOF), "roots": rows}


def _bound_prestage(config):
    """(proof, journal witnesses) when the pass's pre-stage proof binds exactly, else None.

    Binding: the format, this boot, the preservation set, every receipt by path and digest, and the
    journal by its digest at the end of the finished run.
    """
    from ec_clean30 import workstation

    spec = config.get("resume")
    if not spec:
        return None
    path = Path(config["state_root"]) / PRESTAGE_PROOF
    if not path.is_file():
        return None
    try:
        proof = _record(path)
        journal = Path(config["state_root"]) / PRESTAGE_JOURNAL
        if (proof.get("format") != PRESTAGE_FORMAT or proof.get("preservation_sha256") != spec["preservation_sha256"]
                or proof.get("boot_id") is None or proof.get("boot_id") != workstation.boot_id()
                or not journal.is_file() or _file_sha(journal) != proof.get("journal_sha256")):
            return None
        bound = {(row["path"], row["sha256"]) for row in _resume_receipts(spec)}
        roots = proof.get("roots")
        if not (isinstance(roots, list) and roots and all(
                isinstance(row, dict) and (row.get("receipt_path"), row.get("receipt_sha256")) in bound
                and _file_sha(Path(row["receipt_path"])) == row["receipt_sha256"] for row in roots)):
            return None
        return proof, _prestage_journal(journal, spec["preservation_sha256"])
    except (OSError, ValueError, KeyError, TypeError, LaunchWindowError):
        return None


def prestaged_witnesses(config, execution_root, primary_root, mirror_root):
    """Witness items [member, size, digest, primary print, mirror print] the bound pre-stage covers.

    Only receipt members with a journal witness of this boot and of this exact root pair are
    returned; anything that does not bind (another preservation set, receipt, journal, boot or root
    pair) yields none, and closeout then takes its full path. Closeout still compares both prints
    with the copies before it skips a file.
    """
    bound = _bound_prestage(config)
    if bound is None:
        return []
    proof, done = bound
    try:
        for row in proof["roots"]:
            if (row.get("execution_root") == execution_root and row.get("primary_root") == str(primary_root)
                    and row.get("mirror_root") == str(mirror_root)):
                record = _record(row["receipt_path"])
                if record.get("execution_root") != execution_root:
                    return []
                items = []
                for entry in record["files"]:
                    witness = done.get((execution_root, str(primary_root), str(mirror_root), entry["path"]))
                    if (_prestaged_member(entry["path"]) and witness is not None
                            and witness[:2] == (entry["size_bytes"], entry["sha256"])):
                        items.append([entry["path"], *witness[:2], list(witness[2]), list(witness[3])])
                return items
    except (OSError, ValueError, KeyError, TypeError, LaunchWindowError):
        return []
    return []


def verify_prestage(config):
    """Unbilled launch check: a pre-stage proof, if present, still covers unchanged copies of this boot.

    Without a proof nothing is checked (closeout then takes its full path). With one, every pre-staged
    member's two new copies must still match their fingerprints; otherwise the launch stops before any
    window is reserved, so the pre-stage is run again rather than repaired in billed time.
    """
    spec = config.get("resume")
    if not spec or not (Path(config["state_root"]) / PRESTAGE_PROOF).is_file():
        return None
    bound = _bound_prestage(config)
    _require(bound is not None, "The pre-stage proof no longer binds this pass, boot or journal; "
                                "run --prepare-resume again before the launch.")
    proof, done = bound
    checked = 0
    for row in proof["roots"]:
        roots = [Path(row["primary_root"]), Path(row["mirror_root"])]
        record = _record(row["receipt_path"])
        for entry in record["files"]:
            if not _prestaged_member(entry["path"]):
                continue
            key = (row["execution_root"], str(roots[0]), str(roots[1]), entry["path"])
            _require(_unchanged_since(done.get(key), [root / entry["path"] for root in roots], entry),
                     "A pre-staged copy changed since the pre-stage; run --prepare-resume again before the launch.")
            checked += 1
    return {"status": "prestaged_unchanged", "files": checked}


def inherited_receipts(config):
    """The bound earlier preservation receipts of a resume, each checked against its digest; none without one."""
    spec = config.get("resume")
    if not spec:
        return []
    records = []
    for row in _resume_receipts(spec):
        path = Path(row["path"])
        _require(_file_sha(path) == row["sha256"], "Prior preservation receipt changed.")
        records.append(_record(path))
    return records


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
    _require(not config.get("resume") or "resume_g3" in state,
             "A resumed session restores its preserved records before Phase 4.")
    materialize_g3(preapproval, preapproval_sha256=_file_sha(approval_path), freeze1c=Path(freeze["local"]),
                   out=g3, source=source, session=session,
                   gross_ceiling_usd=config["runpod"]["gross_ceiling_usd"],
                   prior=state.get("resume_g3") if config.get("resume") else None)
    # The bound driver state names this G3 for a later resume of the same fold.
    reference = {"path": str(g3), "sha256": _file_sha(g3)}
    _require(state.get("g3_authority", reference) == reference, "The session's G3 authority changed during recovery.")
    state["g3_authority"] = reference
    remote_g3 = f"{config['remote_campaign']}/g2_requests/g3_v5.json"
    _require(request.get("execution_manifest") == config["execution_manifest"],
             "The reviewed Phase-4 request names another execution manifest.")
    request.update(freeze1c=freeze["local"], g3_authority=str(g3), g3_sha256=_file_sha(g3),
                   measurement_files={row["sha256"]: row["local_path"] for row in state["g2_evidence"].values()},
                   remote_campaign=config["remote_campaign"])
    preparation = state["pretrain_preparation"]
    request["measurement_files"][preparation["sha256"]] = preparation["local_path"]
    if state.get("resume_attempts"):
        _require(not request.get("attempts") or request["attempts"] == state["resume_attempts"],
                 "Reviewed attempts conflict with preserved results.")
        request["attempts"] = state["resume_attempts"]
        from ec_clean30 import phase4
        # Inner CLEAN fits and the held-out sanity read the template farms: each must name the
        # attempt holding its embedding producer's preserved result, as final farms do below.
        template = dict(request["remote_request_template"])
        root = PurePosixPath(template["execution_root"])
        for unit in phase4.embedding_units({**request, "clean_scope": "inner"}):
            template[f"{unit['kind']}_farm"] = str(root / "runs" / unit["unit_id"]
                                                   / f"attempt{request['attempts'].get(unit['unit_id'], 1)}" / "farm")
        request["remote_request_template"] = template
        if request.get("run_final_clean"):
            request["final_clean_request"] = {**request["final_clean_request"],
                "farms_by_unit": phase4.final_farms_by_unit({**request, "clean_scope": "final"})}
    template = dict(request["remote_request_template"])
    template.update(fold=request["fold"], execution_manifest=manifest, freeze1c=freeze["remote"], g3_authority=remote_g3,
                    g3_sha256=_file_sha(g3), vm_probe=probe,
                    measurement_files={row["sha256"]: row["remote_path"] for row in state["g2_evidence"].values()})
    template["measurement_files"][preparation["sha256"]] = preparation["remote_path"]
    if request["fold"] > 0:
        cache = preparation["record"]["caches"]["cuda"]
        template.update(seq_cache=cache["path"], seq_manifest_sha256=cache["sha256"])
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


# One remote command per archive: check every member's size and digest against the reviewed listing
# and every target's containment, refuse before any write if an existing file differs, then write
# each missing file to a partial beside its own target (one filesystem, so the campaign's POSIX
# aliases views/ and g2_requests/ on another disk are safe) and rename it into place.
RESTORE_PROGRAM = """import hashlib,json,os,pathlib,sys,tarfile
archive,root,expected,count,aliases=sys.argv[1:]
archive,root,aliases=pathlib.Path(archive),pathlib.Path(root),json.loads(aliases)
def allowed(name):
    first=name.split('/',1)[0]; link=root/first
    if first in aliases and link.is_symlink():
        assert os.readlink(link)==aliases[first], 'A campaign storage alias differs from its declared target: '+first
        return pathlib.Path(os.path.realpath(aliases[first]))
    return pathlib.Path(os.path.realpath(root))
def inside(name):
    parent,base=pathlib.Path(os.path.realpath((root/name).parent)),allowed(name)
    return parent==base or base in parent.parents
partial=None
try:
    with tarfile.open(archive) as tar:
        members=tar.getmembers()
        names=[member.name for member in members]
        assert all(member.isfile() for member in members) and names==sorted(set(names)) and len(names)==int(count), 'Unexpected restore archive members'
        assert all(name and not name.startswith('/') and '..' not in name.split('/') and str(pathlib.PurePosixPath(name))==name and not any(ord(c)<32 for c in name) for name in names), 'Unsafe restore member name'
        rows=[[member.name,member.size,hashlib.file_digest(tar.extractfile(member),'sha256').hexdigest()] for member in members]
        assert hashlib.sha256(json.dumps(rows,separators=(',',':')).encode()).hexdigest()==expected, 'Restored bytes differ from the preserved inventory'
        moves=[]
        for member,(name,size,sha) in zip(members,rows):
            assert inside(name), 'Restore target escapes the campaign root or its declared aliases: '+name
            target=root/name
            if os.path.lexists(target):
                assert target.is_file() and not target.is_symlink() and target.stat().st_size==size, 'An existing remote file differs; a preserved result is never replaced: '+name
                with target.open('rb') as handle:
                    assert hashlib.file_digest(handle,'sha256').hexdigest()==sha, 'An existing remote file differs; a preserved result is never replaced: '+name
            else:
                moves.append((member,name,target,sha))
        for member,name,target,sha in moves:
            target.parent.mkdir(parents=True,exist_ok=True)
            partial=target.with_name('.'+target.name+'.restore-partial')
            if os.path.lexists(partial):
                assert partial.is_file() and not partial.is_symlink(), 'Unexpected restore partial path: '+name
                partial.unlink()
            hashed=hashlib.sha256()
            with tar.extractfile(member) as source, open(partial,'xb') as output:
                while chunk:=source.read(8*1024*1024):
                    hashed.update(chunk); output.write(chunk)
                output.flush(); os.fsync(output.fileno())
            os.chmod(partial,member.mode&0o755)
            assert hashed.hexdigest()==sha and inside(name) and not os.path.lexists(target), 'A restore target changed during restore: '+name
            os.replace(partial,target); partial=None
    print(json.dumps({'restored':len(moves),'kept':len(rows)-len(moves)}))
finally:
    if partial is not None and partial.is_file() and not partial.is_symlink():
        partial.unlink()
    archive.unlink(missing_ok=True)
"""


class _HashingReader:
    """Hash exactly the bytes tarfile copies into an archive member."""

    def __init__(self, handle):
        self.handle, self.hash, self.size = handle, hashlib.sha256(), 0

    def read(self, size=-1):
        data = self.handle.read(size)
        self.hash.update(data)
        self.size += len(data)
        return data


def _restore_archive(path: Path, batch: list[tuple[str, Mapping[str, Any]]]) -> tuple[str, int]:
    """A deterministic archive of regular files whose archived bytes match the preserved digests."""
    listing = []
    with tarfile.open(path, "x", format=tarfile.PAX_FORMAT) as archive:
        for name, row in batch:
            with Path(row["local"]).open("rb") as handle:
                info = os.fstat(handle.fileno())
                _require(stat.S_ISREG(info.st_mode), "A preserved restore source is not a regular file.")
                member = tarfile.TarInfo(name)
                member.size, member.mode = info.st_size, 0o755 if info.st_mode & 0o111 else 0o644
                reader = _HashingReader(handle)
                archive.addfile(member, reader)
            _require(reader.size == info.st_size and reader.hash.hexdigest() == row["sha256"],
                     "A preserved file changed before it was archived for restore.")
            listing.append([name, info.st_size, row["sha256"]])
    return (hashlib.sha256(json.dumps(listing, separators=(",", ":")).encode()).hexdigest(),
            sum(size for _, size, _ in listing))


def _restore_batches(config, rows):
    """Check every restore row and split the rows into bounded archives; no transport call and no write."""
    from ec_clean30 import provenance, workstation

    root = PurePosixPath(config["remote_campaign"])
    # The only links a restore may follow: the POSIX storage aliases runpod_bootstrap.prepare_posix_storage makes.
    posix = config.get("runpod", {}).get("posix_root")
    aliases = {} if posix is None else {"views": str(PurePosixPath(posix) / "views"),
                                        "g2_requests": str(PurePosixPath(posix) / "requests")}
    members = []
    for row in rows:
        local, remote = Path(row["local"]), PurePosixPath(row["remote"])
        provenance.refuse_test_paths([local, str(remote)], what="Resume restored files")
        _require(remote.is_absolute() and ".." not in remote.parts and root in remote.parents
                 and not any(c in str(remote) for c in "\n\r\0") and not local.is_symlink() and local.is_file(),
                 "A preserved restore file has changed or lies outside its campaign.")
        workstation._absolute(str(remote))  # The restricted identity/test guard of the per-file route.
        _sha256(row["sha256"], "restored file sha256")
        members.append((str(remote.relative_to(root)), row, local.stat().st_size))
    members.sort(key=lambda item: item[0])
    _require(len({name for name, _, _ in members}) == len(members), "A restored path is declared twice.")
    batches, current, size = [], [], 0
    for name, row, length in members:
        if current and (size + length > RESTORE_BATCH_BYTES or len(current) >= RESTORE_BATCH_FILES):
            batches.append(current)
            current, size = [], 0
        current.append((name, row))
        size += length
    if current:
        batches.append(current)
    return root, aliases, batches, len(members)


def _restore_files(config, rows, transport, check_session):
    """Restore preserved files in a few bounded archives, never replacing a remote result.

    Each archive costs one upload, one remote command and two session checks, whatever its file count.
    """
    from ec_clean30 import workstation

    root, aliases, batches, count = _restore_batches(config, rows)
    scratch = Path(config["state_root"]) / "resume_restore"
    scratch.mkdir(parents=True, exist_ok=True)
    timings = {"files": count, "batches": []}
    for index, batch in enumerate(batches):
        check_session()
        local = scratch / f"batch{index}.tar"
        local.unlink(missing_ok=True)  # A partial archive from an interrupted restore is rebuilt.
        row = {"index": index, "files": len(batch)}
        timings["batches"].append(row)
        try:
            started = time.monotonic()
            digest, length = _restore_archive(local, batch)
            row.update(bytes=length, archive_seconds=time.monotonic() - started)
            archive = f"{root}/g2_requests/resume_restore/{digest}.tar"
            started = time.monotonic()
            transport.upload(local, archive, timeout=workstation.INVENTORY_MIN_SECONDS
                             + local.stat().st_size / workstation.INVENTORY_TRANSFER_BYTES_PER_SECOND)
            row["upload_seconds"] = time.monotonic() - started
            check_session()
            started = time.monotonic()
            result = json.loads(transport.remote([config["bootstrap_python"], "-c", RESTORE_PROGRAM, archive,
                str(root), digest, str(len(batch)), json.dumps(aliases, sort_keys=True)],
                timeout=workstation.INVENTORY_MIN_SECONDS + len(batch) * RESTORE_FILE_SECONDS
                + 3 * length / workstation.INVENTORY_HASH_BYTES_PER_SECOND))
            row["remote_seconds"] = time.monotonic() - started
            _require(isinstance(result, dict) and set(result) == {"restored", "kept"}
                     and all(type(value) is int and value >= 0 for value in result.values())
                     and result["restored"] + result["kept"] == len(batch), "The remote restore receipt is malformed.")
            row.update(result)
        finally:
            local.unlink(missing_ok=True)
            _timing_record(Path(config["state_root"]) / "restore_timings.json", "restore", timings)


def _timing_record(path: Path, key: str, value: Any) -> None:
    """Host-side timings for sizing later reserves; a failed write never stops the session."""
    from ec_clean30 import workstation

    try:
        document = _record(path) if path.is_file() else {}
        document[key] = value
        workstation._write(path, document)
    except (OSError, ValueError, TypeError, LaunchWindowError):
        pass


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
    if fold > 0:
        record("attach_fold", phase4.run_fold_attachment)
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
        allowed = {"clean_scope", "clean_inputs", "clean_forecasts", "clean_timeouts", "farms_by_unit",
                   "embedding_forecasts", "embedding_timeouts"}
        _require(set(final) <= allowed and final.get("clean_scope") == "final",
                 "Final CLEAN overrides may not change source, session, caps or persistence.")
        template = {**request["remote_request_template"],
                    **{key: value for key, value in final.items() if key not in {"clean_forecasts", "clean_timeouts", "embedding_forecasts", "embedding_timeouts"}}}
        final_request = {**request, **final, "remote_request_template": template}
        farms = phase4.final_farms_by_unit(final_request)
        _require(final.get("farms_by_unit") == farms, "Final CLEAN farms must match their seed-specific producers.")
        record("final_embeddings", phase4.run_embedding_queue, unit_request=final_request)
        record("clean_final", phase4.run_clean_remote_queue, unit_request=final_request)
    return {"status": "passed", "fold": fold, **phase3.source_identity(source)}


def run_v5(config: Mapping[str, Any], config_path: Path, *, authorization_path: Path, window_id: str,
           transport=None, controller=None) -> dict[str, Any]:
    """Explicit launch entry point; all publication, approval and retry gates precede provider access."""
    from ec_clean30 import phase3, workstation

    _require(config.get("provider") == "runpod" and config.get("execution_scope") in {"g2_then_fold0_v5", "fold_execution_v5"},
             "Execution v5 requires its explicit Runpod configuration.")
    workstation.prepare_session_chain_receipt(config)
    source = phase3.verify_execution_source(Path(config["execution_manifest"]),
        chain_receipt_path=workstation.session_chain_receipt_path(config),
        expected_chain=_record(Path(config["execution_manifest"]))["amendment_chain"])
    # The records only: the full two-copy check runs in the launch gate below, after the quote check
    # and before any window is reserved, so a long check cannot age the quote it relies on.
    authorization = validate_launch_authorization(config, config_path, authorization_path, source,
                                                  resume_files="records")
    authorization_sha = _file_sha(authorization_path)
    ledger = FailedWindowLedger(authorization["failure_ledger"], authorization["failure_ledger_authorization_sha256"])
    reserved = False
    state_path = Path(config["state_root"]) / "driver_state.json"
    # Only the exact unresolved window may skip quote freshness, and only to reach the
    # allocation it already owns; a window that never attempted a start, or a new
    # window, needs a fresh quote before any create.
    recovering = ledger.holds_open(window_id, config_sha256=_file_sha(config_path),
                                   execution_manifest_sha256=source["execution_manifest_sha256"],
                                   pass_root=Path(config["state_root"]))

    def launch_binding(path):
        return {"config_sha256": _file_sha(path), "execution_manifest_sha256": source["execution_manifest_sha256"],
                "authorization_sha256": authorization_sha, "window_id": window_id,
                "failure_ledger": str(ledger.path)}

    def before_allocation(current, path, ready):
        nonlocal reserved
        _require(_file_sha(authorization_path) == authorization_sha, "Launch authorization changed.")
        validate_launch_authorization(current, path, authorization_path, ready["source"], resume_files="full")
        # Every local restore refusal comes here, before any window is reserved or provider access begins.
        resume = validate_resume(current, ready["source"], files="records")
        if resume is not None:
            plan_restore(current, resume, ready["source"])
            # The new copies a pre-stage made are unchanged since it verified them (unbilled stat check).
            verify_prestage(current)
        ledger.begin_window(window_id, config_sha256=_file_sha(path),
                            execution_manifest_sha256=source["execution_manifest_sha256"],
                            pass_root=Path(config["state_root"]))
        reserved = True
        return launch_binding(path)

    def finish():
        state = _record(state_path) if state_path.exists() else {}
        terminal = verified_terminal_state(state)
        if workstation.closeout_interrupted(state):
            # The pod is stopped and its preservation set bound; the identical re-run finishes the closeout.
            return state, None
        if terminal is not None:
            _require(state_path.is_file(), "A reserved launch window lacks its durable driver record.")
            closeout = state.get("closeout", {})
            preserved = bool(closeout.get("preservation_sha256")) and (
                closeout.get("provider_absence_verified") is True or closeout.get("provider_stop_verified") is True)
            outcome = ("completed" if (state.get("workload_admitted") or
                        state.get("status") in {"phase4_complete", "stopped_futility"})
                       and preserved and not state.get("preservation_error") else "failed")
            ledger.finish_window(window_id, outcome=outcome, closeout_sha256=_file_sha(state_path),
                                  provider_terminal=terminal)
        return state, terminal

    # A coordinator that died after writing its closed driver record but before recording it here
    # left this window open; the re-run records that verified closeout instead of being refused.
    if recovering and state_path.exists():
        prior = _record(state_path)
        if prior.get("closed") and prior.get("launch_binding") == launch_binding(config_path):
            if workstation.closeout_interrupted(prior):
                try:
                    workstation.finish_pending_closeout(config, config_path, state_path, controller=controller)
                except Exception:
                    finish()  # A real verification failure is recorded; the window closes as failed.
                    raise
            state, terminal = finish()
            if terminal is not None:
                return state
    try:
        return workstation._run_runpod_g2(config, config_path, transport=transport, controller=controller,
            pre_allocation=before_allocation, prepare_g2=prepare_g2, complete_g2=complete_g2,
            prepare_phase4=prepare_phase4, recover_window=recovering)
    finally:
        if reserved:
            finish()


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
