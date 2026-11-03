"""Transport integrity, resumption and authorization boundaries without cloud writes."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import time
import uuid

import pytest

from ec_clean30 import runpod_bootstrap as rb


def digest(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def small_chunks(monkeypatch):
    monkeypatch.setattr(rb, "CHUNK_BYTES", 16)


def configuration(tmp_path):
    root = tmp_path / "local"
    root.mkdir()
    stage = []
    payloads = (b"main runtime payload repeated twice", b"clean locked environment payload", b"original fold zero view archive")
    for name, data in zip(("main.tar.gz", "clean.tar.gz", "fold0.tar"), payloads):
        local = root / name
        local.write_bytes(data)
        stage.append({"local": str(local), "remote": str(tmp_path / "remote" / name), "sha256": digest(data), "kind": "file"})
    compressed = root / "view.transport.gz"
    compressed.write_bytes(gzip.compress(payloads[2], mtime=0))
    stage[2]["transport"] = {"local": str(compressed), "encoding": "gzip"}
    result = {"provider": "runpod", "artifact_cache": {"volume_id": "cache_123", "data_center_id": rb.REGION,
              "endpoint": rb.ENDPOINT, "profile": "runpod-cache", "chunk_bytes": rb.CHUNK_BYTES, "workers": 2},
              "runtime_environments": {"fit": {"archive": stage[0]["remote"]}, "esm1b": {"archive": stage[1]["remote"]}},
              "view_archive": stage[2]["remote"], "stage": stage, "state_root": str(tmp_path / "state"),
              "remote_campaign": str(tmp_path / "remote"), "bootstrap_python": "/usr/bin/python3"}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(result))
    return result, path


class MissingObject(Exception):
    response = {"Error": {"Code": "NoSuchKey"}}


class S3:
    def __init__(self):
        self.objects = {}
        self.events = []
        self.fail_chunk = None

    def get_object(self, *, Bucket, Key):
        self.events.append(("get", Key))
        if Key not in self.objects:
            raise MissingObject()
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, *, Bucket, Key, Body):
        self.events.append(("put", Key))
        if self.fail_chunk and Key.endswith(self.fail_chunk):
            raise OSError("write interrupted")
        self.objects[Key] = bytes(Body)

    def list_objects_v2(self, **kwargs):
        return {"Contents": [{"Key": key, "Size": len(data)} for key, data in self.objects.items()], "IsTruncated": False}


@pytest.mark.parametrize("compressed_main", [False, True])
def test_staging_scratch_covers_main_first_and_two_remaining_assemblies(tmp_path, small_chunks, compressed_main):
    config, _ = configuration(tmp_path)
    if compressed_main:
        item = config["stage"][0]
        transport = tmp_path / "main.transport.gz"
        transport.write_bytes(gzip.compress(Path(item["local"]).read_bytes(), mtime=0))
        item["transport"] = {"local": str(transport), "encoding": "gzip"}
    rb.prepare_cache_descriptors(config)
    rows = [json.loads(rb._descriptor_path(config, item).read_text()) for item in config["stage"]]
    scratch = [row["transport_size_bytes"] * (2 if row["encoding"] == "gzip" else 1) for row in rows]
    assert rb.staging_scratch_bytes(config) == max(scratch[0], scratch[1] + scratch[2])
    config["stage"][1]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="absent"):
        rb.staging_scratch_bytes(config)


def fake_cache(monkeypatch, client):
    monkeypatch.setattr(rb, "_client", lambda cache: (client, object()))
    monkeypatch.setattr(rb, "_signed_get", lambda cache, key, credentials: {
        "url": f"{cache['endpoint']}/{cache['volume_id']}/{key}",
        "headers": {"Authorization": "private exact GET signature"}})

    def fetch(request, destination, *, expected_sha256, expected_size, endpoint, deadline):
        key = request["url"].split("/cache_123/", 1)[1]
        data = client.objects[key]
        assert len(data) == expected_size and digest(data) == expected_sha256
        destination.write_bytes(data)

    monkeypatch.setattr(rb, "_fetch", fetch)


def signed_request(tmp_path, descriptor, *, endpoint=None):
    directory = tmp_path / "requests"
    directory.mkdir(exist_ok=True)
    path = directory / (uuid.uuid4().hex + ".json")
    root = tmp_path / "remote"
    request = {"descriptor": descriptor, "destination": str(root / "archive.tar"), "allowed_root": str(root),
               "endpoint": endpoint or rb.ENDPOINT, "deadline_unix": time.time() + 600, "workers": 1,
               "requests": {r["sha256"]: {"url": f"{rb.ENDPOINT}/cache_123/{rb.PREFIX}/chunks/{r['sha256']}",
                   "headers": {"Authorization": "sensitive-signature"}} for r in descriptor["chunks"]}}
    path.write_text(json.dumps(request))
    path.chmod(0o600)
    return path, request


@pytest.mark.parametrize("key,value", [("endpoint", "http://s3api-eu-ro-1.runpod.io"), ("endpoint", "https://foreign.invalid"),
    ("data_center_id", "ANY_SECURE"), ("workers", 3), ("workers", True), ("chunk_bytes", 7),
    ("size_gb", 21), ("size_gb", 20.0), ("profile", ""), ("api_key", "forbidden")])
def test_cache_configuration_refuses_unapproved_endpoint_capacity_or_inline_credentials(tmp_path, key, value):
    config, _ = configuration(tmp_path)
    config["artifact_cache"][key] = value
    with pytest.raises(RuntimeError):
        rb.validate_cache_config(config)


def test_identity_transport_preserves_exact_existing_pin(tmp_path, small_chunks):
    data = b"0123456789abcdef" * 3 + b"last"
    path = tmp_path / "artifact"
    path.write_bytes(data)
    value = rb.descriptor_for_file(path, target_sha256=digest(data), target_size_bytes=len(data))
    assert [r["size_bytes"] for r in value["chunks"]] == [16, 16, 16, 4]
    assert value["target_sha256"] == value["transport_sha256"] == digest(data)
    with pytest.raises(RuntimeError, match="Identity"):
        rb.descriptor_for_file(path, target_sha256="0" * 64, target_size_bytes=len(data))


@pytest.mark.parametrize("case", ["count", "size", "digest", "encoding", "bool_size"])
def test_descriptor_tampering_is_refused(tmp_path, small_chunks, case):
    path = tmp_path / "artifact"
    path.write_bytes(b"x" * 33)
    value = rb.descriptor_for_file(path, target_sha256=digest(path.read_bytes()), target_size_bytes=33)
    if case == "count":
        value["chunks"].pop()
    elif case == "size":
        value["chunks"][0]["size_bytes"] = 17
    elif case == "digest":
        value["chunks"][0]["sha256"] = "../escape"
    elif case == "encoding":
        value["encoding"] = "pickle"
    else:
        value["target_size_bytes"] = True
    with pytest.raises(RuntimeError):
        rb.validate_descriptor(value)


def test_gzip_descriptor_proves_original_view_bytes(tmp_path, small_chunks):
    data = b"certified original view" * 10
    path = tmp_path / "transport.gz"
    path.write_bytes(gzip.compress(data, mtime=0))
    value = rb.descriptor_for_file(path, target_sha256=digest(data), target_size_bytes=len(data), encoding="gzip")
    assert value["transport_sha256"] == digest(path.read_bytes())
    with pytest.raises(RuntimeError, match="original pinned"):
        rb.descriptor_for_file(path, target_sha256="1" * 64, target_size_bytes=len(data), encoding="gzip")


def test_preparation_reuses_valid_maps_after_full_caller_preflight(tmp_path, small_chunks, monkeypatch):
    config, _ = configuration(tmp_path)
    first = rb.prepare_cache_descriptors(config, source={"git_head": "published"})
    monkeypatch.setattr(rb, "descriptor_for_file", lambda *a, **k: pytest.fail("Unchanged maps were recomputed"))
    second = rb.prepare_cache_descriptors(config, source={"git_head": "next published source"})
    assert first["artifacts"] == second["artifacts"]
    assert second["source"] != first["source"]


def test_seed_is_compute_free_and_warm_seed_uploads_no_unchanged_chunks(tmp_path, small_chunks, monkeypatch):
    config, config_path = configuration(tmp_path)
    client = S3()
    fake_cache(monkeypatch, client)
    source = {"published": True, "git_head": "clean-commit"}
    first = rb.seed_cache(config, config_path, source=source)
    first_puts = [event for event in client.events if event[0] == "put"]
    assert first["status"] == "verified" and first["compute_allocated"] is False
    assert first["uploaded_chunks"] > 0 and len(first["artifacts"]) == 3
    assert any("/probes/" in key for _, key in first_puts)
    assert "/descriptors/" in first_puts[-1][1]
    client.events.clear()
    second = rb.seed_cache(config, config_path, source=source)
    assert second["uploaded_bytes"] == 0 and second["reused_chunks"] == first["uploaded_chunks"]
    assert not any(event[0] == "put" for event in client.events)
    serialized = json.dumps(second)
    assert "Authorization" not in serialized and "private exact" not in serialized


def test_cache_compatibility_failure_blocks_bulk_upload_and_readiness(tmp_path, small_chunks, monkeypatch):
    config, config_path = configuration(tmp_path)
    client = S3()
    fake_cache(monkeypatch, client)
    monkeypatch.setattr(rb, "_fetch", lambda *a, **k: (_ for _ in ()).throw(OSError("incompatible")))
    with pytest.raises(RuntimeError, match="bulk seeding"):
        rb.seed_cache(config, config_path, source={"published": True})
    assert all("/probes/" in key for key in client.objects)
    assert not (Path(config["state_root"]) / "cache_seed_receipt.json").exists()


def test_interrupted_seed_never_publishes_descriptors_and_resumes_verified_chunks(tmp_path, small_chunks, monkeypatch):
    config, config_path = configuration(tmp_path)
    client = S3()
    fake_cache(monkeypatch, client)
    rb.prepare_cache_descriptors(config)
    descriptor = rb._read(rb._descriptor_path(config, config["stage"][1]))
    client.fail_chunk = descriptor["chunks"][0]["sha256"]
    with pytest.raises(RuntimeError, match="upload failed"):
        rb.seed_cache(config, config_path, source={"published": True})
    assert not any("/descriptors/" in key for key in client.objects)
    assert not (Path(config["state_root"]) / "cache_seed_receipt.json").exists()
    client.fail_chunk = None
    result = rb.seed_cache(config, config_path, source={"published": True})
    assert result["reused_chunks"] > 0


def test_local_transport_corruption_after_preparation_refuses_seed(tmp_path, small_chunks, monkeypatch):
    config, config_path = configuration(tmp_path)
    rb.prepare_cache_descriptors(config)
    path = Path(config["stage"][0]["local"])
    path.write_bytes(b"!" * path.stat().st_size)
    fake_cache(monkeypatch, S3())
    with pytest.raises(RuntimeError, match="changed after preparation"):
        rb.seed_cache(config, config_path, source={"published": True})


def test_unpublished_seed_never_opens_s3(tmp_path, small_chunks, monkeypatch):
    config, path = configuration(tmp_path)
    monkeypatch.setattr(rb, "_client", lambda *a: pytest.fail("S3 touched"))
    with pytest.raises(RuntimeError, match="published"):
        rb.seed_cache(config, path, source={"published": False})


def test_hydration_restores_gzip_and_removes_all_signed_request_material(tmp_path, small_chunks, monkeypatch):
    data = b"original safe tar archive" * 10
    compressed = tmp_path / "transport.gz"
    compressed.write_bytes(gzip.compress(data, mtime=0))
    descriptor = rb.descriptor_for_file(compressed, target_sha256=digest(data), target_size_bytes=len(data), encoding="gzip")
    client = S3()
    for i, row in enumerate(descriptor["chunks"]):
        client.objects[f"{rb.PREFIX}/chunks/{row['sha256']}"] = compressed.read_bytes()[i*16:i*16+row["size_bytes"]]
    fake_cache(monkeypatch, client)
    path, request = signed_request(tmp_path, descriptor)
    receipt = rb.hydrate_request(path)
    assert Path(request["destination"]).read_bytes() == data
    assert not path.exists() and receipt["status"] == "verified"
    assert "Authorization" not in json.dumps(receipt)
    assert not list(Path(request["destination"]).parent.glob(".g2-chunks-*"))
    # A complete existing destination is re-read, not downloaded again.
    path, _ = signed_request(tmp_path, descriptor)
    monkeypatch.setattr(rb, "_fetch", lambda *a, **k: pytest.fail("Warm artifact was fetched"))
    assert rb.hydrate_request(path)["status"] == "verified_existing"


def test_interrupted_hydration_reuses_only_complete_verified_chunks(tmp_path, small_chunks, monkeypatch):
    data = b"first chunk.....second chunk....third chunk....."
    source = tmp_path / "source"
    source.write_bytes(data)
    descriptor = rb.descriptor_for_file(source, target_sha256=digest(data), target_size_bytes=len(data))
    counts = []
    fail = [True]

    def fetch(request, destination, *, expected_sha256, expected_size, **kwargs):
        counts.append(expected_sha256)
        index = next(i for i, row in enumerate(descriptor["chunks"]) if row["sha256"] == expected_sha256)
        if index == 1 and fail[0]:
            raise OSError("interrupted")
        destination.write_bytes(data[index*16:index*16+expected_size])

    monkeypatch.setattr(rb, "_fetch", fetch)
    path, request = signed_request(tmp_path, descriptor)
    with pytest.raises(OSError, match="interrupted"):
        rb.hydrate_request(path)
    assert not path.exists() and not Path(request["destination"]).exists()
    first_sha = descriptor["chunks"][0]["sha256"]
    assert counts.count(first_sha) == 1
    fail[0] = False
    path, _ = signed_request(tmp_path, descriptor)
    receipt = rb.hydrate_request(path)
    assert receipt["reused_chunks"] >= 1 and counts.count(first_sha) == 1
    assert Path(request["destination"]).read_bytes() == data


def test_failed_full_archive_hash_never_promotes_target(tmp_path, small_chunks, monkeypatch):
    data = b"view archive payload"
    source = tmp_path / "source.gz"
    source.write_bytes(gzip.compress(data, mtime=0))
    descriptor = rb.descriptor_for_file(source, target_sha256=digest(data), target_size_bytes=len(data), encoding="gzip")
    client = S3()
    for i, row in enumerate(descriptor["chunks"]):
        client.objects[f"{rb.PREFIX}/chunks/{row['sha256']}"] = source.read_bytes()[i*16:i*16+row["size_bytes"]]
    fake_cache(monkeypatch, client)
    descriptor["target_sha256"] = "0" * 64
    path, request = signed_request(tmp_path, descriptor)
    with pytest.raises(RuntimeError, match="Restored original"):
        rb.hydrate_request(path)
    assert not path.exists() and not Path(request["destination"]).exists()


@pytest.mark.parametrize("case", ["endpoint", "key", "destination", "permissions", "expired"])
def test_hydration_refuses_foreign_request_or_destination(tmp_path, small_chunks, monkeypatch, case):
    source = tmp_path / "source"
    source.write_bytes(b"archive")
    descriptor = rb.descriptor_for_file(source, target_sha256=digest(b"archive"), target_size_bytes=7)
    path, request = signed_request(tmp_path, descriptor)
    if case == "endpoint":
        request["endpoint"] = "https://foreign.invalid"
    elif case == "key":
        request["requests"][descriptor["chunks"][0]["sha256"]]["url"] += "/foreign"
    elif case == "destination":
        request["destination"] = str(tmp_path / "outside")
    elif case == "expired":
        request["deadline_unix"] = time.time() - 1
    path.write_text(json.dumps(request))
    if case == "permissions":
        path.chmod(0o644)
    monkeypatch.setattr(rb, "_fetch", lambda *a, **k: pytest.fail("Foreign request was fetched"))
    with pytest.raises(RuntimeError):
        rb.hydrate_request(path)
    assert not path.exists()


def test_symlink_lock_does_not_modify_another_file(tmp_path):
    destination = tmp_path / "target"
    foreign = tmp_path / "foreign"
    foreign.write_text("preserved")
    destination.with_name(destination.name + ".g2.lock").symlink_to(foreign)
    with pytest.raises(OSError):
        with rb._writer_lock(destination):
            pytest.fail("Symlink writer lock accepted")
    assert foreign.read_text() == "preserved"


def test_fetch_refuses_redirect_response_and_removes_partial(tmp_path, monkeypatch):
    data = b"payload"
    url = f"{rb.ENDPOINT}/cache_123/chunk"

    class Response(io.BytesIO):
        status = 200
        headers = {"Content-Length": str(len(data))}

        def geturl(self):
            return "https://foreign.invalid/redirect"

    monkeypatch.setattr(rb.urllib.request, "build_opener", lambda *a: SimpleNamespace(open=lambda *a, **k: Response(data)))
    destination = tmp_path / "partial"
    with pytest.raises(RuntimeError, match="redirected"):
        rb._fetch({"url": url, "headers": {"Authorization": "secret"}}, destination,
                  expected_sha256=digest(data), expected_size=len(data), endpoint=rb.ENDPOINT, deadline=time.time()+60)
    assert not destination.exists()


def test_real_sigv4_headers_do_not_contain_permanent_secret():
    from botocore.credentials import Credentials
    cache = {"endpoint": rb.ENDPOINT, "volume_id": "cache_123", "data_center_id": rb.REGION}
    secret = "rps_private_credential_value"
    request = rb._signed_get(cache, f"{rb.PREFIX}/chunks/{'1'*64}", Credentials("user_example", secret))
    assert "Authorization" in request["headers"]
    assert request["headers"]["User-Agent"] == "DeepMzyme-ArtifactCache/1.0"
    assert secret not in json.dumps(request) and "?" not in request["url"]


def test_signed_get_fetch_preserves_explicit_client_user_agent(tmp_path, monkeypatch):
    from botocore.credentials import Credentials
    data = b"EC-cache-v1\n"
    cache = {"endpoint": rb.ENDPOINT, "volume_id": "cache_123", "data_center_id": rb.REGION}
    secret = "rps_private_credential_value"
    signed = rb._signed_get(cache, f"{rb.PREFIX}/probes/{digest(data)}", Credentials("user_example", secret))

    class Response(io.BytesIO):
        status = 200
        headers = {"Content-Length": str(len(data))}

        def geturl(self):
            return signed["url"]

    def open_request(request, **kwargs):
        assert request.get_method() == "GET" and request.full_url == signed["url"]
        assert request.get_header("User-agent") == "DeepMzyme-ArtifactCache/1.0"
        assert request.get_header("Authorization") == signed["headers"]["Authorization"]
        assert "?" not in request.full_url and secret not in str(request.header_items())
        return Response(data)

    monkeypatch.setattr(rb.urllib.request, "build_opener", lambda *a: SimpleNamespace(open=open_request))
    destination = tmp_path / "fixture"
    rb._fetch(signed, destination, expected_sha256=digest(data), expected_size=len(data),
              endpoint=rb.ENDPOINT, deadline=time.time() + 60)
    assert destination.read_bytes() == data


def test_warm_completion_check_reads_only_small_descriptors(tmp_path, small_chunks, monkeypatch):
    config, path = configuration(tmp_path)
    client = S3()
    fake_cache(monkeypatch, client)
    source = {"published": True, "git_head": "same-published-source"}
    rb.seed_cache(config, path, source=source)
    client.events.clear()
    result = rb.verify_seeded_cache(config, source=source)
    assert result["status"] == "verified" and result["full_payload_downloaded"] is False
    assert len(client.events) == 3 and all("/descriptors/" in key for action, key in client.events)
    next_source = {**source, "git_head": "changed"}
    next_ready = rb.verify_seeded_cache(config, source=next_source)
    assert next_ready["source"] == next_source and next_ready["original_seed_source"] == source
    with pytest.raises(RuntimeError, match="published execution"):
        rb.verify_seeded_cache(config, source={**source, "published": False})
    key = result["artifacts"][0]["key"]
    client.objects[key] += b"corrupt"
    with pytest.raises(RuntimeError, match="absent or changed"):
        rb.verify_seeded_cache(config, source=source)


def test_completion_check_refuses_cold_cache_without_opening_s3(tmp_path, small_chunks, monkeypatch):
    config, _ = configuration(tmp_path)
    monkeypatch.setattr(rb, "_client", lambda *a: pytest.fail("Cold cache opened S3"))
    with pytest.raises(RuntimeError, match="absent"):
        rb.verify_seeded_cache(config, source={"published": True})


def test_capacity_refuses_full_or_foreign_cache_before_bulk_writes(tmp_path, small_chunks, monkeypatch):
    config, path = configuration(tmp_path)
    client = S3()
    fake_cache(monkeypatch, client)
    client.list_objects_v2 = lambda **kwargs: {"Contents": [{"Key": f"{rb.PREFIX}/chunks/{'0'*64}",
                                                           "Size": 20 * 10 ** 9}], "IsTruncated": False}
    with pytest.raises(RuntimeError, match="lacks capacity"):
        rb.seed_cache(config, path, source={"published": True})
    assert not any("/chunks/" in key for key in client.objects)
    client.list_objects_v2 = lambda **kwargs: {"Contents": [{"Key": "unrelated/preserved", "Size": 1}]}
    with pytest.raises(RuntimeError, match="foreign files"):
        rb.seed_cache(config, path, source={"published": True})


def test_host_hydration_uploads_only_helper_and_private_requests(tmp_path, small_chunks, monkeypatch):
    config, path = configuration(tmp_path)
    client = S3()
    fake_cache(monkeypatch, client)
    rb.seed_cache(config, path, source={"published": True})
    monkeypatch.setattr(rb.Path, "home", classmethod(lambda cls: tmp_path / "home"))

    class Transport:
        reserve_seconds = 0

        def __init__(self):
            self.uploads, self.commands = [], []

        def stat_file(self, name):
            f = Path(name)
            return {"sha256": digest(f.read_bytes()), "size_bytes": f.stat().st_size} if f.is_file() else None

        def upload(self, local, remote):
            self.uploads.append(str(local))
            target = Path(remote)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local, target)

        def remote(self, argv, **kwargs):
            self.commands.append(argv)
            if argv[0] == "mkdir":
                Path(argv[-1]).mkdir(parents=True, exist_ok=True, mode=0o700)
            elif argv[0] == "chmod":
                Path(argv[-1]).chmod(int(argv[1], 8))
            elif argv[0] == "rm":
                Path(argv[-1]).unlink(missing_ok=True)
            elif "hydrate" in argv:
                return json.dumps(rb.hydrate_request(Path(argv[-1])))
            else:
                pytest.fail("Unexpected transport command")
            return ""

    transport = Transport()
    item = config["stage"][0]
    result = rb.hydrate_cached_artifact(config, transport, item, time.time() + 600)
    assert result["laptop_payload_bytes"] == 0 and result["status"] == "verified"
    assert len(transport.uploads) == 2
    assert item["local"] not in transport.uploads
    assert not list((tmp_path / "home/.cache/deepmzyme/runpod_requests").iterdir())
    assert not list((Path(config["remote_campaign"]) / "g2_requests/bootstrap/requests").iterdir())
    assert "private exact GET signature" not in json.dumps(transport.commands)
    assert rb.hydrate_cached_artifact(config, transport, item, time.time() + 600)["status"] == "verified_existing"
