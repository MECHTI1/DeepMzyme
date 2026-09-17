"""Maintain a serial queue on one already prepared, owned Colab endpoint.

This command never allocates a GPU, clears a pause, or changes the training
grid. Run only from a separately frozen operational continuation whose staged
source and state include the maintained remote_queue module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serial_metal_campaign import control, remote_queue, runtime, supervisor
import colab_serial_metal_host as host


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _copy_exact(source, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name("." + target.name + "." + uuid.uuid4().hex)
    with temporary.open("xb") as stream:
        stream.write(Path(source).read_bytes())
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)


class ColabQueueAdapter:
    def __init__(self, output, session_id, *, colab_executable="colab", persistence_seconds):
        self.output = Path(output).resolve()
        self.session_id = session_id
        self.manifest = runtime.read_json(self.output / "campaign_manifest.json")
        self.remote_output = self.manifest["output_dir"]
        self.remote_root = self.manifest["root"]
        self.worker_python = remote_queue.worker_python(self.manifest)
        self.cli = shutil.which(colab_executable)
        if self.cli is None:
            raise ValueError("The configured Colab CLI executable is unavailable.")
        self.work = self.output / "host_control" / "queue_transport"
        self.work.mkdir(parents=True, exist_ok=True)
        self.persistence_seconds = runtime._number(persistence_seconds, "persistence reserve")
        if self.persistence_seconds <= 0:
            raise ValueError("A positive measured per-fit persistence allowance is required.")
        self.guard_path = "/content/deepmzyme_worker_receipt_" + self.session_id + ".json"

    def _call(self, arguments, timeout=60):
        path = self.work / (uuid.uuid4().hex + ".log")
        try:
            with path.open("x") as stream:
                result = subprocess.run([self.cli, *arguments], stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError("Colab transport timed out; reconcile the same endpoint.") from exc
        if result.returncode:
            # Authentication/configuration errors are not blindly retried.
            raise RuntimeError(f"Colab transport exited {result.returncode}; diagnostic log: {path}")

    def _upload(self, source, target):
        self._call(["upload", "-s", self.session_id, str(source), str(target)], timeout=60)

    def _download(self, source, target):
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name("." + target.name + "." + uuid.uuid4().hex)
        self._call(["download", "-s", self.session_id, str(source), str(temporary)], timeout=60)
        os.replace(temporary, target)

    def _rpc(self, operation, **kwargs):
        ident = uuid.uuid4().hex
        script, answer = self.work / (ident + ".py"), self.work / (ident + ".json")
        remote_answer = "/content/deepmzyme_queue_reply_" + ident + ".json"
        worker_source = (
            "import json, pathlib, sys, traceback\n"
            + "sys.path.insert(0, " + repr(self.remote_root + "/src") + ")\n"
            + "from serial_metal_campaign import remote_queue\n"
            + "try:\n    value = getattr(remote_queue, " + repr(operation) + ")(" + repr(self.remote_output)
            + ", **" + repr(kwargs) + ")\n    reply = {'ok': True, 'value': value}\n"
            + "except BaseException as exc:\n    reply = {'ok': False, 'error': str(exc), 'traceback': traceback.format_exc()}\n"
            + "pathlib.Path(" + repr(remote_answer) + ").write_text(json.dumps(reply, default=str, allow_nan=False))\n"
        )
        worker_script = "/content/deepmzyme_queue_worker_" + ident + ".py"
        script.write_text("import pathlib, subprocess\n"
                          + "pathlib.Path(" + repr(worker_script) + ").write_text(" + repr(worker_source) + ")\n"
                          + "subprocess.run(" + repr([self.worker_python, worker_script]) + ", check=True, timeout=45)\n")
        self._call(["exec", "-s", self.session_id, "-f", str(script), "--timeout", "50"], timeout=60)
        self._download(remote_answer, answer)
        reply = runtime.read_json(answer)
        if not reply["ok"]:
            raise RuntimeError(reply["traceback"])
        return reply["value"]

    def _refresh_guard(self):
        receipt = host.worker_receipt(self.output, self.session_id)
        path = self.work / "current_worker_receipt.json"
        runtime.atomic_json(path, receipt)
        self._upload(path, self.guard_path)
        return receipt

    def verify_setup(self):
        control.require_running(self.output)
        self._refresh_guard()
        return self._rpc("verify_setup", manifest_sha256=_sha(self.output / "campaign_manifest.json"),
                         host_receipt=self.guard_path)

    def reconnect(self):
        control.require_running(self.output)
        checked = host.status(self.output, self.session_id)
        if not checked["work_allowed"]:
            raise RuntimeError("Same endpoint no longer permits work; automatic replacement is disabled.")
        return checked

    def observe(self):
        return self._rpc("observe")

    def persist_attempt(self, attempt_id):
        payload = self._rpc("package_attempt", attempt_id=attempt_id)
        archive = self.output / "persistence" / (attempt_id + ".tar.gz")
        self._download(payload["archive"], archive)
        if _sha(archive) != payload["archive_sha256"]:
            raise ValueError("Downloaded attempt archive differs.")
        attempt = payload["attempt"]
        remote_queue.verify_archive(archive, attempt["artifacts"], self.output / "persisted_attempts" / attempt_id)
        readback_archive = self.remote_output + "/queue_transport/" + attempt_id + "_readback.tar.gz"
        self._upload(archive, readback_archive)
        receipt = {"method": "verified_transfer", "artifacts": attempt["artifacts"],
                   "destination_uri": "host-verified-reupload://" + str(self.output) + "/" + attempt_id + "/" + payload["archive_sha256"],
                   "readback_root": self.remote_output + "/queue_transport/readback/" + attempt_id}
        result = self._rpc("acknowledge_attempt", attempt_id=attempt_id, archive=readback_archive,
                           archive_sha256=payload["archive_sha256"], receipt=receipt)
        runtime.atomic_json(self.output / "persistence" / (attempt_id + ".json"), result)
        return {"attempt_status": attempt["status"], "receipt": result}

    def persist_state(self, *, prepare_next):
        if prepare_next:
            self._refresh_guard()
        payload = self._rpc("package_state", prepare_next=prepare_next, host_receipt=self.guard_path,
                            persistence_seconds=self.persistence_seconds)
        archive = self.output / "persistence" / ("state_" + payload["state_sha256"] + ".tar.gz")
        self._download(payload["archive"], archive)
        if _sha(archive) != payload["archive_sha256"]:
            raise ValueError("Downloaded state archive differs.")
        mirror = self.output / "state_snapshots" / payload["state_sha256"]
        remote_queue.verify_archive(archive, payload["files"], mirror)
        # Host user decisions must never be overwritten by an older worker
        # snapshot, including a STOP arriving during a transfer.
        with runtime._lock(self.output / "control_lock"):
            local_controls = {path.name: _sha(path) for path in self.output.glob("USER_REQUESTED_PAUSE*.json")}
            if (self.output / "control_events.json").is_file():
                local_controls["control_events.json"] = _sha(self.output / "control_events.json")
            remote_controls = {name: digest for name, digest in payload["files"].items()
                               if name == "control_events.json" or name.startswith("USER_REQUESTED_PAUSE")}
            if local_controls != remote_controls:
                raise RuntimeError("Worker snapshot has stale user controls; preserve host pause/resume state.")
            for name in payload["files"]:
                _copy_exact(mirror / name, self.output / name)
        readback_archive = self.remote_output + "/queue_transport/state_" + payload["state_sha256"] + "_readback.tar.gz"
        self._upload(archive, readback_archive)
        receipt = {"state_sha256": payload["state_sha256"],
                   "destination_uri": "host-verified-reupload://" + str(self.output) + "/" + payload["state_sha256"],
                   "readback_root": self.remote_output + "/queue_transport/readback/state_" + payload["state_sha256"]}
        self._rpc("acknowledge_state", archive=readback_archive, archive_sha256=payload["archive_sha256"], receipt=receipt)
        runtime.atomic_json(self.work / "last_verified_state.json", {**payload, "receipt": receipt})
        return payload

    def launch(self, next_info):
        self._refresh_guard()
        control.require_running(self.output)
        return self._rpc("launch", next_info=next_info, host_receipt=self.guard_path)

    def await_attempt(self, next_info):
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            control.require_running(self.output)
            state = self.observe()
            if (state.get("last") or {}).get("attempt_id") == next_info["attempt_id"]:
                return state
            time.sleep(1)
        # Reusing this exact launch request reconciles an existing dispatcher;
        # an ambiguous dispatch is a hard failure, never a second submission.
        self.launch(next_info)
        raise RuntimeError("Dispatcher did not register the intended attempt within 30 seconds.")

    def stop(self, reason):
        request = host._request(self.output, self.session_id)
        if request.get("stopped_epoch") is None:
            runtime.atomic_json(host._session(self.output, self.session_id) / "stop_required.json",
                                {"reason": reason, "epoch": time.time()})
        receipt = host.stop(self.output, self.session_id)
        worker = next((row for row in runtime._sessions(self.output) if row["session_id"] == self.session_id), None)
        if worker is not None:
            runtime.close_session(self.output, self.session_id, receipt["stopped_epoch"], stop_evidence=receipt)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--colab-executable", default="colab")
    parser.add_argument("--persistence-seconds", type=float, required=True,
                        help="Measured per-fit transfer allowance from the operational forecast; no invented default.")
    parser.add_argument("--max-completed", type=int)
    args = parser.parse_args()
    control.require_running(args.output)
    adapter = ColabQueueAdapter(args.output, args.session_id, colab_executable=args.colab_executable,
                                persistence_seconds=args.persistence_seconds)
    print(json.dumps(supervisor.run(args.output, adapter, max_completed=args.max_completed), indent=2))


if __name__ == "__main__":
    main()
