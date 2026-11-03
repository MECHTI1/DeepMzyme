"""Exact public-model transport without a cloud or model download."""
import base64
import hashlib
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace
import urllib.request

import pytest

from ec_clean30 import runpod_bootstrap as rb


def digest(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def archive_case(tmp_path, monkeypatch):
    body = b"frozen public model contents\x00" * 23
    local = tmp_path / "original.tar"
    with tarfile.open(local, "w", format=tarfile.USTAR_FORMAT) as archive:
        member = tarfile.TarInfo("models/esmc/data/weights/esmc_300m_2024_12_v0.pth")
        member.size = len(body)
        archive.addfile(member, io.BytesIO(body))
    archive_bytes = local.read_bytes()
    monkeypatch.setattr(rb, "ESMC_BODY_BYTES", len(body))
    monkeypatch.setattr(rb, "ESMC_BODY_SHA256", digest(body))
    monkeypatch.setattr(rb, "ESMC_ARCHIVE_SHA256", digest(archive_bytes))
    root = tmp_path / "remote"
    item = {"local": str(local), "remote": str(root / "inputs" / "esmc.tar"),
            "sha256": digest(archive_bytes)}
    source = {"frozen_esmc_weight_sha256": digest(body)}
    descriptor = rb.public_archive_descriptor(item, source)
    request = {"descriptor": descriptor, "destination": item["remote"],
               "allowed_root": str(root), "deadline_unix": 2000.0}
    request_path = tmp_path / "request.json"
    monkeypatch.setattr(rb.time, "time", lambda: 1000.0)

    def save():
        request_path.write_text(json.dumps(request))
        return request_path

    return SimpleNamespace(body=body, archive_bytes=archive_bytes, item=item, source=source,
                           descriptor=descriptor, request=request, save=save,
                           destination=Path(item["remote"]), root=root, tmp_path=tmp_path)


def public_response(monkeypatch, body, *, url=None, status=200):
    calls = []

    class Response(io.BytesIO):
        def __init__(self):
            super().__init__(body)
            self.url = url or rb.ESMC_URL
            self.status = status

    class Opener:
        def open(self, requested_url, *, timeout):
            calls.append((requested_url, timeout))
            return Response()

    monkeypatch.setattr(rb.urllib.request, "build_opener", lambda *handlers: Opener())
    return calls


def no_download(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("The request must be rejected before any network download.")
    monkeypatch.setattr(rb.urllib.request, "build_opener", fail)


def test_descriptor_and_download_restore_original_tar_exactly(archive_case, monkeypatch):
    case = archive_case
    prefix = base64.b64decode(case.descriptor["prefix_base64"])
    suffix = base64.b64decode(case.descriptor["suffix_base64"])
    assert prefix + case.body + suffix == case.archive_bytes
    calls = public_response(monkeypatch, case.body)
    receipt = rb.reconstruct_public_archive(case.save())
    assert case.destination.read_bytes() == case.archive_bytes
    assert receipt["status"] == "verified"
    assert receipt["sha256"] == digest(case.archive_bytes)
    assert receipt["cloud_download_bytes"] == len(case.body)
    assert receipt["laptop_payload_bytes"] == 0
    assert calls == [(rb.ESMC_URL, 30)]
    assert not list(case.destination.parent.glob("*.partial"))


def test_verified_existing_archive_needs_no_download(archive_case, monkeypatch):
    case = archive_case
    case.destination.parent.mkdir(parents=True)
    case.destination.write_bytes(case.archive_bytes)
    no_download(monkeypatch)
    receipt = rb.reconstruct_public_archive(case.save())
    assert receipt == {"status": "verified_existing", "sha256": digest(case.archive_bytes),
                       "laptop_payload_bytes": 0}


def test_existing_wrong_archive_is_preserved_and_refused(archive_case, monkeypatch):
    case = archive_case
    case.destination.parent.mkdir(parents=True)
    corrupt = b"X" + case.archive_bytes[1:]
    case.destination.write_bytes(corrupt)
    no_download(monkeypatch)
    with pytest.raises(RuntimeError, match="Existing public archive differs"):
        rb.reconstruct_public_archive(case.save())
    assert case.destination.read_bytes() == corrupt


@pytest.mark.parametrize("kind", ["corrupt", "truncated", "oversized"])
def test_bad_model_bytes_never_publish_archive(archive_case, monkeypatch, kind):
    case = archive_case
    body = {"corrupt": b"X" + case.body[1:], "truncated": case.body[:-1],
            "oversized": case.body + b"X"}[kind]
    public_response(monkeypatch, body)
    with pytest.raises(RuntimeError, match="frozen (pin|size)"):
        rb.reconstruct_public_archive(case.save())
    assert not case.destination.exists()


def test_failed_download_can_retry_without_accepting_its_partial(archive_case, monkeypatch):
    case = archive_case
    public_response(monkeypatch, case.body[:-1])
    with pytest.raises(RuntimeError, match="frozen pin"):
        rb.reconstruct_public_archive(case.save())
    partial = case.destination.with_name(".g2-public-" + case.item["sha256"] + ".partial")
    assert partial.exists()
    public_response(monkeypatch, case.body)
    assert rb.reconstruct_public_archive(case.save())["status"] == "verified"
    assert case.destination.read_bytes() == case.archive_bytes
    assert not partial.exists()


def test_deadline_expiring_during_download_prevents_publication(archive_case, monkeypatch):
    case = archive_case
    clock = [1000.0]
    monkeypatch.setattr(rb.time, "time", lambda: clock[0])

    class Response(io.BytesIO):
        status = 200
        url = rb.ESMC_URL

        def read(self, size):
            block = super().read(size)
            clock[0] = 2001.0
            return block

    monkeypatch.setattr(rb.urllib.request, "build_opener", lambda *handlers:
                        SimpleNamespace(open=lambda *args, **kwargs: Response(case.body)))
    with pytest.raises(RuntimeError, match="stop reserve"):
        rb.reconstruct_public_archive(case.save())
    assert not case.destination.exists()


def test_wrong_tar_envelope_fails_even_with_correct_model(archive_case, monkeypatch):
    case = archive_case
    prefix = base64.b64decode(case.descriptor["prefix_base64"])
    case.descriptor["prefix_base64"] = base64.b64encode(b"X" + prefix[1:]).decode()
    public_response(monkeypatch, case.body)
    with pytest.raises(RuntimeError, match="closed archive differs"):
        rb.reconstruct_public_archive(case.save())
    assert not case.destination.exists()


@pytest.mark.parametrize("field,value", [("body_sha256", "0" * 64), ("archive_sha256", "0" * 64),
                                        ("body_size_bytes", 1), ("url", "https://other.example/model")])
def test_recipe_changes_refused_before_network(archive_case, monkeypatch, field, value):
    case = archive_case
    case.descriptor[field] = value
    no_download(monkeypatch)
    with pytest.raises(RuntimeError, match="frozen ESMC recipe"):
        rb.reconstruct_public_archive(case.save())


@pytest.mark.parametrize("kind", ["too_large", "wrong_size", "invalid_base64"])
def test_unbounded_or_invalid_closure_refused(archive_case, monkeypatch, kind):
    case = archive_case
    if kind == "too_large":
        case.descriptor["prefix_base64"] = base64.b64encode(b"X" * (65536 + 1)).decode()
    elif kind == "wrong_size":
        case.descriptor["archive_size_bytes"] += 1
    else:
        case.descriptor["prefix_base64"] = "not-valid-base64!"
    no_download(monkeypatch)
    with pytest.raises((RuntimeError, ValueError)):
        rb.reconstruct_public_archive(case.save())


@pytest.mark.parametrize("kind", ["outside", "parent_traversal", "symlink"])
def test_destination_must_stay_in_declared_nonsymlink_root(archive_case, monkeypatch, kind):
    case = archive_case
    if kind == "outside":
        case.request["destination"] = str(case.tmp_path / "outside.tar")
    elif kind == "parent_traversal":
        case.request["destination"] = str(case.root / ".." / "outside.tar")
    else:
        case.root.mkdir()
        (case.root / "inputs").symlink_to(case.tmp_path, target_is_directory=True)
    no_download(monkeypatch)
    with pytest.raises(RuntimeError, match="destination"):
        rb.reconstruct_public_archive(case.save())
    assert not (case.tmp_path / "outside.tar").exists()


@pytest.mark.parametrize("deadline", [999.0, 1000.0, float("inf"), float("nan"), True])
def test_invalid_or_expired_deadline_refused(archive_case, monkeypatch, deadline):
    case = archive_case
    case.request["deadline_unix"] = deadline
    no_download(monkeypatch)
    with pytest.raises(RuntimeError, match="deadline is exhausted"):
        rb.reconstruct_public_archive(case.save())


@pytest.mark.parametrize("url,status", [("http://cdn.example/model", 200),
                                      ("https://cdn.example/model", 206)])
def test_download_requires_complete_https_response(archive_case, monkeypatch, url, status):
    case = archive_case
    public_response(monkeypatch, case.body, url=url, status=status)
    with pytest.raises(RuntimeError, match="complete HTTPS content"):
        rb.reconstruct_public_archive(case.save())
    assert not case.destination.exists()


def test_https_redirect_handler_refuses_protocol_downgrade():
    request = urllib.request.Request(rb.ESMC_URL)
    with pytest.raises(RuntimeError, match="redirect is not HTTPS"):
        rb._HTTPSRedirect().redirect_request(request, None, 302, "redirect", {}, "http://cdn.example/model")


@pytest.mark.parametrize("which", ["archive", "body"])
def test_descriptor_requires_both_original_pins(archive_case, which):
    case = archive_case
    if which == "archive":
        case.item["sha256"] = "0" * 64
    else:
        case.source["frozen_esmc_weight_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="frozen ESMC pins"):
        rb.public_archive_descriptor(case.item, case.source)


def test_descriptor_refuses_wrong_model_layout(archive_case):
    case = archive_case
    with tarfile.open(case.item["local"], "w", format=tarfile.USTAR_FORMAT) as archive:
        member = tarfile.TarInfo("unrelated-model.pth")
        member.size = len(case.body)
        archive.addfile(member, io.BytesIO(case.body))
    with pytest.raises(RuntimeError, match="unexpected model layout"):
        rb.public_archive_descriptor(case.item, case.source)


def test_hydration_skips_unrelated_artifact_without_transport():
    assert rb.hydrate_public_artifact({}, None, {"sha256": "0" * 64}, {}, 2000) is False


def test_hydration_transfers_only_recipe_and_helper_then_verifies_readback(archive_case):
    case = archive_case
    uploads, commands = [], []

    class Transport:
        reserve_seconds = 120

        def stat_file(self, path):
            if path == case.item["remote"]:
                return {"sha256": case.item["sha256"], "size_bytes": len(case.archive_bytes)}
            for local, remote in uploads:
                if path == remote:
                    return {"sha256": rb._sha(local), "size_bytes": local.stat().st_size}
            return None

        def upload(self, local, remote):
            uploads.append((Path(local), remote))

        def remote(self, command, *, timeout):
            commands.append((command, timeout))
            return json.dumps({"status": "verified", "laptop_payload_bytes": 0})

    config = {"remote_campaign": str(case.root), "state_root": str(case.tmp_path / "state"),
              "bootstrap_python": "/usr/bin/python3"}
    assert rb.hydrate_public_artifact(config, Transport(), case.item, case.source, 1500)
    assert len(uploads) == 2
    assert all(local != Path(case.item["local"]) for local, _ in uploads)
    request = json.loads(uploads[1][0].read_text())
    assert request["deadline_unix"] == 1380
    assert request["descriptor"] == case.descriptor
    assert commands[0][0][-3:] == ["public-archive", "--request", uploads[1][1]]
    assert commands[0][1] == 380
    receipt = json.loads((Path(config["state_root"]) / "staging_receipts" /
                          (case.item["sha256"] + ".json")).read_text())
    assert receipt["laptop_payload_bytes"] == 0
