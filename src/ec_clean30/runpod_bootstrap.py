"""Verified, resumable artifact transport for the bounded Runpod preparation path.

The remote subcommand uses only the standard library. S3 credentials and signing
remain on the workstation; temporary exact-GET authorizations are not receipts.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import gzip
import hashlib
import http.client
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tarfile
import tempfile
import threading
import time
from typing import Any, Mapping
import urllib.error
import urllib.parse
import urllib.request
import uuid

CHUNK_BYTES = 64 * 1024 ** 2
MAX_ARTIFACT_BYTES = 20 * 10 ** 9
DESCRIPTOR_FORMAT = "deepmzyme_runpod_artifact_chunks_v1"
CACHE_FORMAT = "deepmzyme_runpod_artifact_cache_v1"
ENDPOINT = "https://s3api-eu-ro-1.runpod.io"
REGION = "EU-RO-1"
PREFIX = "deepmzyme-artifacts-v1"
ESMC_URL = ("https://huggingface.co/EvolutionaryScale/esmc-300m-2024-12/resolve/"
            "7f10b20ae75017b2dbc884070e03434515709a8d/data/weights/esmc_300m_2024_12_v0.pth")
ESMC_BODY_SHA256 = "323dff9fbf3fef297a74f4f18b6528e6f2e599b0bcf72b6927516804015becea"
ESMC_BODY_BYTES = 1332095738
ESMC_ARCHIVE_SHA256 = "47182be2bbc30d1d2b3bd772b8660f7dd36beba71d2803339fe2cb06cfcf9cf9"
POSIX_STORAGE_FORMAT = "deepmzyme_ec_clean30_posix_storage_v1"
POSIX_RESERVE_BYTES = 5 * 1024 ** 3
CAMPAIGN_ID = "ec_class_vs_clean30_2026-10"
_DIGEST = re.compile(r"[0-9a-f]{64}")


def require(condition: Any, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    try:
        with temporary.open("xb") as handle:
            handle.write(_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read(path: Path) -> Any:
    require(path.is_file() and not path.is_symlink(), "Required preparation record is absent or is a symlink.")
    return json.loads(path.read_text())


def _regular(path: Path) -> None:
    require(path.is_file() and not path.is_symlink(), "Artifact must be a regular, nonsymlink file.")


def validate_posix_storage_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Check the explicit storage layout without reading remote filesystem state."""
    require(isinstance(request, Mapping), "POSIX storage request must be an object.")
    value = dict(request)
    require(set(value) == {"format", "campaign_root", "local_root", "required_free_bytes", "runtime_prefixes",
                           "component_bytes"}
            and value["format"] == POSIX_STORAGE_FORMAT, "Unknown POSIX storage request.")

    def absolute(raw: Any) -> Path:
        require(isinstance(raw, str) and raw and not any(c in raw for c in "\n\r\0"),
                "POSIX storage paths must be explicit absolute paths.")
        path = Path(raw)
        require(path.is_absolute() and ".." not in path.parts and path != Path(path.anchor),
                "POSIX storage paths must be explicit absolute paths without traversal.")
        return path

    campaign, local = absolute(value["campaign_root"]), absolute(value["local_root"])
    require(campaign.name == CAMPAIGN_ID, "POSIX campaign root must retain the canonical campaign name.")
    require(campaign != local and campaign not in local.parents and local not in campaign.parents,
            "Bulk archive and local POSIX roots must be separate.")
    required = value["required_free_bytes"]
    require(type(required) is int and required >= POSIX_RESERVE_BYTES,
            "POSIX capacity must include the five-GiB reserve.")
    prefixes = value["runtime_prefixes"]
    require(isinstance(prefixes, list) and len(prefixes) == 2,
            "POSIX storage must declare exactly two runtime prefixes.")
    paths = [absolute(prefix) for prefix in prefixes]
    require(len(set(paths)) == 2 and all(path.parent == local and path.name not in {"views", "requests"}
                                       for path in paths),
            "Runtime prefixes must be distinct direct children of the local root.")
    components = value["component_bytes"]
    require(isinstance(components, dict) and set(components) == {"views", *(path.name for path in paths)}
            and all(type(size) is int and size > 0 for size in components.values()),
            "POSIX component sizes must match both runtime prefixes and views.")
    require(required >= sum(components.values()) + POSIX_RESERVE_BYTES,
            "POSIX capacity must cover all components and the five-GiB reserve.")
    value.update(campaign_root=str(campaign), local_root=str(local), runtime_prefixes=[str(path) for path in paths])
    return value


def _no_symlink_ancestors(path: Path) -> None:
    require(not any(current.is_symlink() for current in (path, *path.parents)),
            "POSIX storage path contains a symlink.")


def _private_directory(path: Path) -> None:
    _no_symlink_ancestors(path)
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid()
            and stat.S_IMODE(info.st_mode) == 0o700,
            "POSIX storage directory must be owned, nonsymlink and mode 0700.")


def _probe_posix_modes(root: Path) -> dict[str, int]:
    directory = Path(tempfile.mkdtemp(prefix=".g2-posix-probe-", dir=root))
    path = directory / "mode-probe"
    measured = {}
    try:
        directory.chmod(0o700)
        measured["private_directory"] = stat.S_IMODE(directory.stat().st_mode)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(b"POSIX mode probe\n")
        measured["private_file"] = stat.S_IMODE(path.stat().st_mode)
        path.chmod(0o444)
        measured["readonly_file"] = stat.S_IMODE(path.stat().st_mode)
        directory.chmod(0o555)
        measured["readonly_directory"] = stat.S_IMODE(directory.stat().st_mode)
        require(measured == {"private_directory": 0o700, "private_file": 0o600,
                             "readonly_file": 0o444, "readonly_directory": 0o555},
                "Local storage does not enforce the required POSIX modes.")
        return measured
    finally:
        directory.chmod(0o700)
        path.unlink(missing_ok=True)
        directory.rmdir()


def _component_storage_bytes(path: Path, device: int, seen: set[tuple[int, int]]) -> int:
    """Credit owned allocated bytes only; runtime symlinks never add capacity."""
    if not path.exists() and not path.is_symlink():
        return 0
    pending, total = [path], 0
    while pending:
        current = pending.pop()
        info = current.lstat()
        require(info.st_uid == os.getuid() and info.st_dev == device,
                "POSIX component contains a foreign owner or filesystem.")
        require(current != path or stat.S_ISDIR(info.st_mode), "POSIX component root must be a directory.")
        if stat.S_ISLNK(info.st_mode):
            require(current != path, "POSIX component root cannot be a symlink.")
            continue
        if stat.S_ISDIR(info.st_mode):
            with os.scandir(current) as entries:
                pending.extend(Path(entry.path) for entry in entries)
        elif stat.S_ISREG(info.st_mode):
            identity = (info.st_dev, info.st_ino)
            if identity not in seen:
                total += min(info.st_blocks * 512, info.st_size)
                seen.add(identity)
        else:
            raise RuntimeError("POSIX component contains an unsupported file type.")
    return total


def prepare_posix_storage(request: Mapping[str, Any]) -> dict[str, Any]:
    """Prepare private requests and read-only views without relocating archives."""
    value = validate_posix_storage_request(request)
    campaign, local = Path(value["campaign_root"]), Path(value["local_root"])
    _no_symlink_ancestors(campaign)
    _no_symlink_ancestors(local)
    for prefix in value["runtime_prefixes"]:
        _no_symlink_ancestors(Path(prefix))
    links = {campaign / "views": local / "views", campaign / "g2_requests": local / "requests"}
    for path, target in links.items():
        if path.exists() or path.is_symlink():
            require(path.is_symlink() and path.lstat().st_uid == os.getuid() and path.readlink() == target,
                    "POSIX storage alias already has an unexpected target or directory.")
    _private_directory(local)
    modes = _probe_posix_modes(local)
    free = shutil.disk_usage(local).free
    seen: set[tuple[int, int]] = set()
    credited = {name: min(size, _component_storage_bytes(local / name, local.stat().st_dev, seen))
                for name, size in value["component_bytes"].items()}
    remaining = max(POSIX_RESERVE_BYTES, value["required_free_bytes"] - sum(credited.values()))
    require(free >= remaining, "Insufficient local POSIX storage and reserve.")
    campaign.mkdir(parents=True, exist_ok=True)
    require(campaign.is_dir() and campaign.stat().st_uid == os.getuid(),
            "Campaign storage root must be an owned directory.")
    for path, target in links.items():
        _private_directory(target)
        if not path.is_symlink():
            path.symlink_to(target, target_is_directory=True)
        require(path.readlink() == target and path.resolve() == target,
                "POSIX storage alias differs from its declared target.")
    return {"format": POSIX_STORAGE_FORMAT, "status": "passed", "campaign_root": str(campaign),
            "local_root": str(local), "runtime_prefixes": value["runtime_prefixes"],
            "required_free_bytes": value["required_free_bytes"], "free_bytes": free, "modes": modes,
            "component_bytes": value["component_bytes"], "credited_bytes": credited,
            "remaining_required_bytes": remaining,
            "links": {str(path): str(target) for path, target in links.items()},
            "helper_sha256": _sha(Path(__file__).resolve())}


def validate_cache_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the approved private cache without accessing credentials or S3."""
    require(config.get("provider") == "runpod", "Artifact cache requires the Runpod route.")
    cache = dict(config.get("artifact_cache", {}))
    require(set(cache) <= {"volume_id", "data_center_id", "endpoint", "profile", "chunk_bytes", "workers", "size_gb"},
            "Unknown cache configuration field; credentials must remain in the protected profile.")
    require(re.fullmatch(r"[A-Za-z0-9_-]{1,100}", str(cache.get("volume_id", ""))), "Explicit cache volume ID required.")
    require(cache.get("data_center_id") == REGION and cache.get("endpoint", "").rstrip("/") == ENDPOINT,
            "Cache must use the reviewed EU-RO-1 HTTPS endpoint.")
    require(re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", str(cache.get("profile", ""))), "Protected named S3 profile required.")
    require(type(cache.get("chunk_bytes", CHUNK_BYTES)) is int and cache.get("chunk_bytes", CHUNK_BYTES) == CHUNK_BYTES,
            "Artifact chunks must retain their fixed 64-MiB size.")
    workers = cache.get("workers", 2)
    require(type(workers) is int and 1 <= workers <= 2, "Cache transfer workers must be one or two.")
    require(cache.get("size_gb", 20) == 20 and type(cache.get("size_gb", 20)) is int,
            "The approved cache capacity is 20 GB.")
    require(not any(key.lower() in {"secret", "secret_key", "access_key", "api_key", "password", "authorization"}
                    for key in cache), "S3 credentials do not belong in configuration.")
    cache.update(endpoint=ENDPOINT, workers=workers, chunk_bytes=CHUNK_BYTES, size_gb=20)
    return cache


def _cached_items(config: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    remotes = {config["runtime_environments"][name]["archive"] for name in ("fit", "esm1b")}
    remotes.add(config["view_archive"])
    items = [item for item in config["stage"] if item["remote"] in remotes]
    require(len(items) == len(remotes) == 3, "Cache must name exactly both runtimes and the Fold-0 archive.")
    return items


def validate_descriptor(value: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(value)
    require(value.get("format") == DESCRIPTOR_FORMAT and value.get("encoding") in {"identity", "gzip"},
            "Unknown artifact transport descriptor.")
    for key in ("target_sha256", "transport_sha256"):
        require(isinstance(value.get(key), str) and _DIGEST.fullmatch(value[key]), "Artifact checksum is invalid.")
    for key in ("target_size_bytes", "transport_size_bytes"):
        require(type(value.get(key)) is int and 0 < value[key] <= MAX_ARTIFACT_BYTES, "Artifact size is invalid.")
    require(value.get("chunk_bytes") == CHUNK_BYTES, "Descriptor chunk size differs.")
    chunks = value.get("chunks")
    expected_count = (value["transport_size_bytes"] + CHUNK_BYTES - 1) // CHUNK_BYTES
    require(isinstance(chunks, list) and len(chunks) == expected_count, "Descriptor chunk count differs.")
    for index, item in enumerate(chunks):
        expected = min(CHUNK_BYTES, value["transport_size_bytes"] - index * CHUNK_BYTES)
        require(isinstance(item, dict) and set(item) == {"sha256", "size_bytes"}
                and isinstance(item["sha256"], str) and _DIGEST.fullmatch(item["sha256"])
                and type(item["size_bytes"]) is int and item["size_bytes"] == expected,
                "Descriptor chunk checksum or length differs.")
    if value["encoding"] == "identity":
        require(value["target_sha256"] == value["transport_sha256"]
                and value["target_size_bytes"] == value["transport_size_bytes"], "Identity transport differs from its target.")
    return value


def _transport(item: Mapping[str, Any]) -> tuple[Path, str]:
    specification = item.get("transport", {})
    require(isinstance(specification, dict) and set(specification) <= {"local", "encoding"}, "Unknown transport override.")
    encoding = specification.get("encoding", "identity")
    require(encoding in {"identity", "gzip"}, "Unknown transport encoding.")
    path = Path(specification.get("local", item["local"]))
    _regular(path)
    return path, encoding


def _descriptor_path(config: Mapping[str, Any], item: Mapping[str, Any]) -> Path:
    require(isinstance(item.get("sha256"), str) and _DIGEST.fullmatch(item["sha256"]), "Stage checksum is invalid.")
    return Path(config["state_root"]) / "cache_descriptors" / (item["sha256"] + ".json")


def descriptor_for_file(path: Path, *, target_sha256: str, target_size_bytes: int,
                        encoding: str = "identity") -> dict[str, Any]:
    """Describe a transport once, retaining the existing scientific archive pin."""
    _regular(path)
    digest, chunks, size = hashlib.sha256(), [], 0
    with path.open("rb") as handle:
        while block := handle.read(CHUNK_BYTES):
            digest.update(block)
            size += len(block)
            chunks.append({"sha256": hashlib.sha256(block).hexdigest(), "size_bytes": len(block)})
    value = {"format": DESCRIPTOR_FORMAT, "encoding": encoding, "target_sha256": target_sha256,
             "target_size_bytes": target_size_bytes, "transport_sha256": digest.hexdigest(),
             "transport_size_bytes": size, "chunk_bytes": CHUNK_BYTES, "chunks": chunks}
    validate_descriptor(value)
    if encoding == "gzip":
        restored, restored_size = hashlib.sha256(), 0
        with gzip.open(path, "rb") as handle:
            while block := handle.read(1024 ** 2):
                restored.update(block)
                restored_size += len(block)
                require(restored_size <= target_size_bytes, "Compressed view exceeds its original pinned size.")
        require(restored_size == target_size_bytes and restored.hexdigest() == target_sha256,
                "Compressed transport does not restore the original pinned archive.")
    return value


def prepare_cache_descriptors(config: Mapping[str, Any], *, source: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Prepare transport maps after the caller's full source/input preflight."""
    validate_cache_config(config)
    records = []
    for item in _cached_items(config):
        target = Path(item["local"])
        _regular(target)
        path, encoding = _transport(item)
        out = _descriptor_path(config, item)
        if out.exists():
            descriptor = validate_descriptor(_read(out))
            require(descriptor["target_sha256"] == item["sha256"]
                    and descriptor["target_size_bytes"] == target.stat().st_size
                    and descriptor["transport_size_bytes"] == path.stat().st_size
                    and descriptor["encoding"] == encoding, "Existing transport map differs; prepare a new artifact version.")
        else:
            descriptor = descriptor_for_file(path, target_sha256=item["sha256"],
                                             target_size_bytes=target.stat().st_size, encoding=encoding)
            _write(out, descriptor)
        records.append({"remote": item["remote"], "descriptor": str(out), "descriptor_sha256": _sha(out),
                        "target_sha256": item["sha256"], "transport_size_bytes": descriptor["transport_size_bytes"]})
    require(sum(row["transport_size_bytes"] for row in records) <= MAX_ARTIFACT_BYTES,
            "Prepared artifacts exceed the cache capacity.")
    receipt = {"format": CACHE_FORMAT, "status": "prepared", "source": dict(source or {}), "artifacts": records}
    _write(Path(config["state_root"]) / "cache_preparation.json", receipt)
    return receipt


def _client(cache: Mapping[str, Any]):
    import boto3
    from botocore.config import Config
    session = boto3.Session(profile_name=cache["profile"])
    credentials = session.get_credentials()
    require(credentials is not None and credentials.method not in {"iam-role", "container-role"},
            "A protected explicit Runpod S3 profile is required.")
    frozen = credentials.get_frozen_credentials()
    require(frozen.access_key.startswith("user_") and frozen.secret_key.startswith("rps_"),
            "The selected profile is not a Runpod S3 credential.")
    client = session.client("s3", endpoint_url=cache["endpoint"], region_name=cache["data_center_id"],
                            config=Config(signature_version="s3v4", s3={"addressing_style": "path"},
                                          connect_timeout=15, read_timeout=90, retries={"max_attempts": 2}))
    return client, frozen


def _signed_get(cache: Mapping[str, Any], key: str, credentials) -> dict[str, Any]:
    from botocore.auth import S3SigV4Auth
    from botocore.awsrequest import AWSRequest
    url = f"{cache['endpoint']}/{cache['volume_id']}/{key}"
    request = AWSRequest(method="GET", url=url, headers={"User-Agent": "DeepMzyme-ArtifactCache/1.0"})
    S3SigV4Auth(credentials, "s3", cache["data_center_id"]).add_auth(request)
    return {"url": url, "headers": dict(request.headers.items())}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, newurl):
        raise RuntimeError("Artifact endpoint redirects are refused.")


def _fetch_once(request: Mapping[str, Any], destination: Path, *, expected_sha256: str,
           expected_size: int, endpoint: str, deadline: float) -> None:
    parsed, origin = urllib.parse.urlsplit(request["url"]), urllib.parse.urlsplit(endpoint)
    require(parsed.scheme == origin.scheme == "https" and parsed.netloc == origin.netloc
            and not parsed.username and not parsed.password and not parsed.query and not parsed.fragment,
            "Signed artifact request has an unapproved origin.")
    require(type(expected_size) is int and 0 < expected_size <= CHUNK_BYTES, "Invalid chunk download size.")
    require(time.time() < deadline, "Artifact download deadline is exhausted.")
    headers = request.get("headers")
    require(isinstance(headers, dict) and headers and all(isinstance(k, str) and isinstance(v, str)
            and not any(c in k + v for c in "\n\r\0") for k, v in headers.items()), "Invalid signed request headers.")
    require(not any(k.lower() == "range" for k in headers), "Range requests are not supported.")
    query = urllib.request.Request(request["url"], headers=headers, method="GET")
    digest, size = hashlib.sha256(), 0
    try:
        with urllib.request.build_opener(_NoRedirect).open(query, timeout=min(45, deadline - time.time())) as response:
            require(response.status == 200 and response.geturl() == request["url"], "Artifact response is incomplete or redirected.")
            length = response.headers.get("Content-Length")
            require(length is None or int(length) == expected_size, "Artifact Content-Length differs.")
            with destination.open("xb") as handle:
                while block := response.read(1024 ** 2):
                    require(time.time() < deadline, "Artifact download reached its deadline.")
                    size += len(block)
                    require(size <= expected_size, "Artifact response exceeds its pinned size.")
                    digest.update(block)
                    handle.write(block)
                handle.flush()
                os.fsync(handle.fileno())
        require(size == expected_size and digest.hexdigest() == expected_sha256, "Artifact response checksum or size differs.")
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def _fetch(request: Mapping[str, Any], destination: Path, *, expected_sha256: str,
           expected_size: int, endpoint: str, deadline: float) -> None:
    for attempt in range(3):
        try:
            return _fetch_once(request, destination, expected_sha256=expected_sha256,
                               expected_size=expected_size, endpoint=endpoint, deadline=deadline)
        except (http.client.IncompleteRead, TimeoutError, ConnectionError, urllib.error.URLError) as exc:
            if isinstance(exc, urllib.error.HTTPError) and exc.code not in {408, 429, 500, 502, 503, 504}:
                raise
            delay = attempt + 1
            if attempt == 2 or time.time() + delay >= deadline:
                raise RuntimeError("Private artifact transport retries exhausted; no verified chunk was produced.") from None
            time.sleep(delay)


def _object_digest_once(client, volume: str, key: str) -> tuple[str, int] | None:
    try:
        response = client.get_object(Bucket=volume, Key=key)
    except Exception as exc:
        code = getattr(exc, "response", {}).get("Error", {}).get("Code")
        if str(code) in {"NoSuchKey", "404", "NotFound"}:
            return None
        raise RuntimeError("Private cache read failed; inspect protected S3 configuration.") from None
    digest, size = hashlib.sha256(), 0
    body = response["Body"]
    try:
        while block := body.read(1024 ** 2):
            digest.update(block)
            size += len(block)
            require(size <= max(CHUNK_BYTES, 64 * 1024), "Cache response exceeds its finite allowance.")
    finally:
        body.close()
    return digest.hexdigest(), size


def _object_digest(client, volume: str, key: str) -> tuple[str, int] | None:
    from botocore.exceptions import ResponseStreamingError, ReadTimeoutError, ConnectionClosedError
    for attempt in range(3):
        try:
            return _object_digest_once(client, volume, key)
        except (ResponseStreamingError, ReadTimeoutError, ConnectionClosedError):
            if attempt == 2:
                raise RuntimeError("Private cache stream retries exhausted; complete readback is required.") from None
            time.sleep(attempt + 1)


def _put_verified(client, cache: Mapping[str, Any], key: str, data: bytes) -> bool:
    expected = hashlib.sha256(data).hexdigest(), len(data)
    if _object_digest(client, cache["volume_id"], key) == expected:
        return False
    try:
        client.put_object(Bucket=cache["volume_id"], Key=key, Body=data)
    except Exception:
        raise RuntimeError("Private cache upload failed; no complete artifact was published.") from None
    require(_object_digest(client, cache["volume_id"], key) == expected, "Closed cache object readback differs.")
    return True


def _compatibility_probe(client, credentials, cache: Mapping[str, Any], state_root: Path) -> dict[str, Any]:
    payload = b"EC-cache-v1\n"
    digest = hashlib.sha256(payload).hexdigest()
    key = f"{PREFIX}/probes/{digest}"
    _put_verified(client, cache, key, payload)
    # No SDK or persistent credentials are involved in this receiving path.
    with tempfile.TemporaryDirectory(prefix="deepmzyme-cache-probe-") as temporary:
        destination = Path(temporary) / "fixture"
        try:
            _fetch(_signed_get(cache, key, credentials), destination, expected_sha256=digest,
                   expected_size=len(payload), endpoint=cache["endpoint"], deadline=time.time() + 60)
        except BaseException:
            raise RuntimeError("Independent authenticated cache GET failed; bulk seeding is refused.") from None
        require(_sha(destination) == digest, "Independent cache GET readback differs.")
    receipt = {"status": "passed", "method": "independent_sigv4_header_get", "sha256": digest,
               "size_bytes": len(payload), "endpoint": cache["endpoint"], "volume_id": cache["volume_id"]}
    _write(state_root / "cache_compatibility.json", receipt)
    return receipt


def _capacity_check(client, cache: Mapping[str, Any], chunks: Mapping[str, int]) -> dict[str, Any]:
    """Account for existing revisions and concurrent PUT scratch before seeding."""
    objects, cursor = {}, None
    try:
        while True:
            arguments = {"Bucket": cache["volume_id"], "MaxKeys": 1000}
            if cursor is not None:
                arguments["ContinuationToken"] = cursor
            response = client.list_objects_v2(**arguments)
            for row in response.get("Contents", []):
                require(isinstance(row.get("Key"), str) and type(row.get("Size")) is int and row["Size"] >= 0,
                        "Cache capacity response contains invalid entries.")
                require(row["Key"].startswith(PREFIX + "/") and row["Key"] not in objects,
                        "Cache contains foreign files or duplicate listing entries.")
                objects[row["Key"]] = row["Size"]
            require(len(objects) <= 10000, "Cache listing exceeds the bounded object allowance.")
            if not response.get("IsTruncated"):
                break
            next_cursor = response.get("NextContinuationToken")
            require(isinstance(next_cursor, str) and next_cursor and next_cursor != cursor,
                    "Cache capacity pagination did not advance.")
            cursor = next_cursor
    except RuntimeError:
        raise
    except Exception:
        raise RuntimeError("Private cache capacity inspection failed; seeding is refused.") from None
    occupied = sum(objects.values())
    additional = sum(max(0, size - objects.get(f"{PREFIX}/chunks/{digest}", 0)) for digest, size in chunks.items())
    reserve = 2 * CHUNK_BYTES + 64 * 1024
    require(occupied + additional + reserve <= cache["size_gb"] * 10 ** 9,
            "The cache lacks capacity for its retained revisions and bounded upload scratch.")
    return {"occupied_bytes": occupied, "additional_chunk_bytes": additional, "scratch_reserve_bytes": reserve,
            "capacity_bytes": cache["size_gb"] * 10 ** 9}


def seed_cache(config: Mapping[str, Any], config_path: Path, *, source: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Populate the configured artifact cache; this function never allocates compute."""
    cache = validate_cache_config(config)
    require(source and source.get("published") is True, "Cache seeding requires the caller's published source preflight.")
    root = Path(config["state_root"])
    root.mkdir(parents=True, exist_ok=True)
    config_sha = _sha(Path(config_path))
    with (root / "cache_seed.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepared = prepare_cache_descriptors(config, source=source)
        client, credentials = _client(cache)
        compatibility = _compatibility_probe(client, credentials, cache, root)
        uploaded, reused, uploaded_bytes = 0, 0, 0
        progress_lock = threading.Lock()
        begun = time.time()
        seed_deadline = begun + 7200
        # One work item keeps one chunk buffer alive; at most two buffers exist.
        def put_chunk(task):
            require(time.time() < seed_deadline, "Cache seeding reached its two-hour preparation bound.")
            item, index, row = task
            path, _ = _transport(item)
            with path.open("rb") as handle:
                handle.seek(index * CHUNK_BYTES)
                data = handle.read(row["size_bytes"])
            require(len(data) == row["size_bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"],
                    "Local transport changed after preparation.")
            changed = _put_verified(client, cache, f"{PREFIX}/chunks/{row['sha256']}", data)
            with progress_lock:
                nonlocal uploaded, reused, uploaded_bytes
                uploaded += int(changed)
                reused += int(not changed)
                uploaded_bytes += len(data) if changed else 0
                _write(root / "cache_seed_progress.json", {"uploaded_chunks": uploaded, "reused_chunks": reused,
                       "uploaded_bytes": uploaded_bytes, "elapsed_seconds": time.time() - begun})

        tasks, descriptors = [], []
        seen = set()
        for item in _cached_items(config):
            descriptor = validate_descriptor(_read(_descriptor_path(config, item)))
            descriptors.append(descriptor)
            for index, row in enumerate(descriptor["chunks"]):
                if row["sha256"] not in seen:
                    tasks.append((item, index, row))
                    seen.add(row["sha256"])
        capacity = _capacity_check(client, cache, {row["sha256"]: row["size_bytes"] for _, _, row in tasks})
        with ThreadPoolExecutor(max_workers=cache["workers"]) as pool:
            list(pool.map(put_chunk, tasks))
        # Descriptor publication is last: a failed chunk never advertises readiness.
        descriptor_rows = []
        for descriptor in descriptors:
            require(time.time() < seed_deadline and _sha(Path(config_path)) == config_sha,
                    "Cache preparation deadline or unchanged-configuration check failed.")
            payload = _bytes(descriptor)
            digest = hashlib.sha256(payload).hexdigest()
            key = f"{PREFIX}/descriptors/{digest}.json"
            _put_verified(client, cache, key, payload)
            descriptor_rows.append({"target_sha256": descriptor["target_sha256"], "descriptor_sha256": digest, "key": key})
        receipt = {"format": CACHE_FORMAT, "status": "verified", "volume_id": cache["volume_id"],
                   "endpoint": cache["endpoint"], "source": dict(source), "config_sha256": config_sha,
                   "preparation_sha256": _sha(root / "cache_preparation.json"), "compatibility": compatibility, "capacity": capacity,
                   "artifacts": descriptor_rows, "uploaded_chunks": uploaded, "reused_chunks": reused,
                   "uploaded_bytes": uploaded_bytes, "elapsed_seconds": time.time() - begun,
                   "compute_allocated": False}
        _write(root / "cache_seed_receipt.json", receipt)
        return receipt


def verify_seeded_cache(config: Mapping[str, Any], *, source: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Read back the small completion descriptors before any compute allocation.

    This does not replace each destination's chunk and complete-archive hashing.
    It avoids downloading all archive bytes to the workstation on a warm start.
    """
    cache = validate_cache_config(config)
    root = Path(config["state_root"])
    seed = _read(root / "cache_seed_receipt.json")
    require(seed.get("status") == "verified" and seed.get("volume_id") == cache["volume_id"]
            and seed.get("endpoint") == cache["endpoint"] and seed.get("compatibility", {}).get("status") == "passed",
            "No verified cache completion record exists for the selected volume.")
    require(source and source.get("published") is True and seed.get("source", {}).get("published") is True,
            "Current and original cache preparation must identify published execution source.")
    client, _ = _client(cache)
    records = []
    for item in _cached_items(config):
        descriptor = validate_descriptor(_read(_descriptor_path(config, item)))
        require(descriptor["target_sha256"] == item["sha256"]
                and descriptor["target_size_bytes"] == Path(item["local"]).stat().st_size,
                "Prepared cache descriptor differs from the current staging closure.")
        payload = _bytes(descriptor)
        digest = hashlib.sha256(payload).hexdigest()
        key = f"{PREFIX}/descriptors/{digest}.json"
        row = {"target_sha256": item["sha256"], "descriptor_sha256": digest, "key": key}
        require(row in seed.get("artifacts", []) and _object_digest(client, cache["volume_id"], key) == (digest, len(payload)),
                "Cache completion descriptor is absent or changed; seed the cache before allocation.")
        records.append(row)
    require(len(seed["artifacts"]) == len(records), "Cache completion includes foreign or duplicate descriptors.")
    return {"format": CACHE_FORMAT, "status": "verified", "volume_id": cache["volume_id"],
            "artifacts": records, "source": dict(source), "original_seed_source": seed["source"],
            "seed_receipt_sha256": _sha(root / "cache_seed_receipt.json"),
            "full_payload_downloaded": False, "compute_allocated": False}


def staging_scratch_bytes(config: Mapping[str, Any]) -> int:
    """Bound chunk/assembly overlap for the main-first, two-writer scheduler.

    The workstation already reserves every final archive and extracted file.
    Identity transport adds retained chunks; gzip adds both chunks and its
    compressed assembly while restoring the final archive. Public downloads
    write directly to the final archive's reserved temporary pathname.
    """
    validate_cache_config(config)
    main = config["runtime_environments"]["fit"]["archive"]
    first, remaining = 0, []
    for item in _cached_items(config):
        descriptor = validate_descriptor(_read(_descriptor_path(config, item)))
        require(descriptor["target_sha256"] == item["sha256"]
                and descriptor["target_size_bytes"] == Path(item["local"]).stat().st_size,
                "Scratch reservation descriptor differs from its staged target.")
        scratch = descriptor["transport_size_bytes"] * (2 if descriptor["encoding"] == "gzip" else 1)
        if item["remote"] == main:
            first = scratch
        else:
            remaining.append(scratch)
    return max(first, sum(sorted(remaining, reverse=True)[:2]))


def _safe_destination(value: str, allowed_root: str) -> Path:
    path, root = Path(value), Path(allowed_root)
    require(path.is_absolute() and root.is_absolute() and ".." not in path.parts and root in path.parents
            and not any(c in value + allowed_root for c in "\n\r\0"), "Artifact destination leaves its declared root.")
    for current in [path, *path.parents]:
        require(not current.is_symlink(), "Artifact destination contains a symlink.")
    return path


@contextmanager
def _writer_lock(destination: Path):
    path = destination.with_name(destination.name + ".g2.lock")
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a+") as lock:
        require(os.fstat(lock.fileno()).st_uid == os.getuid(), "Artifact writer lock has a foreign owner.")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def hydrate_request(request_path: Path) -> dict[str, Any]:
    """Remote standard-library entry point; retry only complete chunk boundaries."""
    require(request_path.parent.name == "requests" and re.fullmatch(r"[0-9a-f]{32}\.json", request_path.name),
            "Signed request must use its dedicated temporary request path.")
    require(request_path.is_file() and not request_path.is_symlink() and request_path.stat().st_uid == os.getuid(),
            "Signed request file must be owned and nonsymlink.")
    try:
        require(stat.S_IMODE(request_path.stat().st_mode) == 0o600, "Signed request file must be private.")
        request = _read(request_path)
        descriptor = validate_descriptor(request["descriptor"])
        destination = _safe_destination(request["destination"], request["allowed_root"])
        endpoint = request["endpoint"]
        require(endpoint == ENDPOINT, "Artifact endpoint differs from the approved cache.")
        deadline = request["deadline_unix"]
        require(type(deadline) in (int, float) and math.isfinite(deadline) and deadline > time.time(),
                "Artifact deadline is invalid or exhausted.")
        require(type(request.get("workers")) is int and 1 <= request["workers"] <= 2, "Invalid download worker bound.")
        signed = request["requests"]
        require(isinstance(signed, dict) and set(signed) == {row["sha256"] for row in descriptor["chunks"]},
                "Signed requests differ from the prepared chunk map.")
        for digest, row in signed.items():
            require(re.fullmatch(r"/[A-Za-z0-9_-]{1,100}/" + PREFIX + r"/chunks/" + digest,
                                 urllib.parse.urlsplit(row["url"]).path),
                    "Signed request key differs from its chunk checksum.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with _writer_lock(destination):
            if destination.exists():
                _regular(destination)
                require(destination.stat().st_size == descriptor["target_size_bytes"]
                        and _sha(destination) == descriptor["target_sha256"], "Existing destination differs from its pin.")
                return {"status": "verified_existing", "sha256": descriptor["target_sha256"], "downloaded_bytes": 0}
            scratch = destination.parent / (".g2-chunks-" + descriptor["transport_sha256"] + "-" + destination.name)
            require(not scratch.is_symlink(), "Chunk scratch is a symlink.")
            scratch.mkdir(exist_ok=True)
            reused, downloaded = 0, 0
            progress_lock = threading.Lock()

            def download(row):
                nonlocal reused, downloaded
                complete = scratch / row["sha256"]
                if complete.exists():
                    _regular(complete)
                    if complete.stat().st_size == row["size_bytes"] and _sha(complete) == row["sha256"]:
                        with progress_lock:
                            reused += 1
                        return
                    complete.unlink()
                partial = scratch / (row["sha256"] + ".partial")
                require(not partial.is_symlink(), "Partial chunk is a symlink.")
                partial.unlink(missing_ok=True)
                _fetch(signed[row["sha256"]], partial, expected_sha256=row["sha256"],
                       expected_size=row["size_bytes"], endpoint=endpoint, deadline=deadline)
                require(_sha(partial) == row["sha256"], "Closed downloaded chunk differs.")
                partial.replace(complete)
                with progress_lock:
                    downloaded += row["size_bytes"]

            unique = {row["sha256"]: row for row in descriptor["chunks"]}
            with ThreadPoolExecutor(max_workers=request["workers"]) as pool:
                list(pool.map(download, unique.values()))
            transport = destination.with_name(destination.name + ".transport.partial")
            final = destination.with_name(destination.name + ".assemble.partial")
            for path in (transport, final):
                require(not path.is_symlink(), "Assembly partial is a symlink.")
                path.unlink(missing_ok=True)
            digest, size = hashlib.sha256(), 0
            with transport.open("xb") as output:
                for row in descriptor["chunks"]:
                    with (scratch / row["sha256"]).open("rb") as chunk:
                        while block := chunk.read(1024 ** 2):
                            require(time.time() < deadline, "Artifact assembly reached its deadline.")
                            digest.update(block)
                            size += len(block)
                            output.write(block)
                output.flush()
                os.fsync(output.fileno())
            require(size == descriptor["transport_size_bytes"] and digest.hexdigest() == descriptor["transport_sha256"]
                    and _sha(transport) == descriptor["transport_sha256"], "Assembled transport differs from its pin.")
            if descriptor["encoding"] == "gzip":
                digest, size = hashlib.sha256(), 0
                with gzip.open(transport, "rb") as compressed, final.open("xb") as output:
                    while block := compressed.read(1024 ** 2):
                        require(time.time() < deadline, "View restoration reached its deadline.")
                        size += len(block)
                        require(size <= descriptor["target_size_bytes"], "Restored view exceeds its pin.")
                        digest.update(block)
                        output.write(block)
                    output.flush()
                    os.fsync(output.fileno())
                require(size == descriptor["target_size_bytes"] and digest.hexdigest() == descriptor["target_sha256"],
                        "Restored original archive differs from its pin.")
            else:
                transport.replace(final)
            require(final.stat().st_size == descriptor["target_size_bytes"] and _sha(final) == descriptor["target_sha256"],
                    "Final closed archive readback differs.")
            final.replace(destination)
            transport.unlink(missing_ok=True)
            for row in unique.values():
                (scratch / row["sha256"]).unlink(missing_ok=True)
            scratch.rmdir()
            receipt = {"status": "verified", "sha256": descriptor["target_sha256"], "size_bytes": descriptor["target_size_bytes"],
                       "downloaded_bytes": downloaded, "reused_chunks": reused, "method": "authenticated_cache_chunks"}
            _write(destination.with_name(destination.name + ".cache_receipt.json"), receipt)
            return receipt
    finally:
        request_path.unlink(missing_ok=True)


def hydrate_cached_artifact(config: Mapping[str, Any], transport, item: Mapping[str, Any], deadline: float) -> dict[str, Any] | None:
    """Hydrate a configured archive through the ordinary authenticated transport."""
    if item["remote"] not in {row["remote"] for row in _cached_items(config)}:
        return None
    cache = validate_cache_config(config)
    descriptor = validate_descriptor(_read(_descriptor_path(config, item)))
    require(descriptor["target_sha256"] == item["sha256"]
            and descriptor["target_size_bytes"] == Path(item["local"]).stat().st_size, "Cache descriptor differs from stage.")
    seed = _read(Path(config["state_root"]) / "cache_seed_receipt.json")
    descriptor_sha = hashlib.sha256(_bytes(descriptor)).hexdigest()
    require(seed.get("status") == "verified" and seed.get("volume_id") == cache["volume_id"]
            and any(row.get("target_sha256") == item["sha256"] and row.get("descriptor_sha256") == descriptor_sha
                    for row in seed.get("artifacts", [])), "Artifact lacks a matching verified cache seed.")
    existing = transport.stat_file(item["remote"])
    expected = {"sha256": item["sha256"], "size_bytes": descriptor["target_size_bytes"]}
    if existing is not None:
        require(existing == expected, "Existing remote artifact differs from its pin.")
        return {"status": "verified_existing", **expected, "downloaded_bytes": 0}
    _, credentials = _client(cache)
    requests = {row["sha256"]: _signed_get(cache, f"{PREFIX}/chunks/{row['sha256']}", credentials)
                for row in descriptor["chunks"]}
    helper = Path(__file__).resolve()
    helper_sha = _sha(helper)
    remote_root = str(PurePosixPath(config["remote_campaign"]) / "g2_requests" / "bootstrap")
    remote_helper = f"{remote_root}/{helper_sha}.py"
    if transport.stat_file(remote_helper) != {"sha256": helper_sha, "size_bytes": helper.stat().st_size}:
        transport.upload(helper, remote_helper)
        require(transport.stat_file(remote_helper) == {"sha256": helper_sha, "size_bytes": helper.stat().st_size},
                "Published cache helper transfer differs.")
    private = Path.home() / ".cache" / "deepmzyme" / "runpod_requests"
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    require(not private.is_symlink() and private.stat().st_uid == os.getuid()
            and stat.S_IMODE(private.stat().st_mode) == 0o700, "Signed-request directory is not private.")
    remote_request = f"{remote_root}/requests/{uuid.uuid4().hex}.json"
    request = {"descriptor": descriptor, "requests": requests, "destination": item["remote"],
               "allowed_root": config["remote_campaign"], "endpoint": cache["endpoint"], "workers": cache["workers"],
               "deadline_unix": deadline - float(getattr(transport, "reserve_seconds", 900))}
    local_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=private, prefix="get-", suffix=".json", delete=False) as handle:
            local_path = Path(handle.name)
            handle.write(_bytes(request))
            handle.flush()
            os.fsync(handle.fileno())
        require(stat.S_IMODE(local_path.stat().st_mode) == 0o600, "Signed request file is not private.")
        transport.remote(["mkdir", "-p", "-m", "700", remote_root + "/requests"])
        transport.remote(["chmod", "700", remote_root + "/requests"])
        transport.upload(local_path, remote_request)
        transport.remote(["chmod", "600", remote_request])
        begun = time.time()
        result = json.loads(transport.remote([config["bootstrap_python"], remote_helper, "hydrate", "--request", remote_request],
                                            timeout=max(1, request["deadline_unix"] - time.time())))
        require(transport.stat_file(item["remote"]) == expected, "Hydrated remote archive readback differs.")
        return {**result, "elapsed_seconds": time.time() - begun, "descriptor_sha256": descriptor_sha,
                "helper_sha256": helper_sha, "laptop_payload_bytes": 0}
    finally:
        if local_path is not None:
            local_path.unlink(missing_ok=True)
        try:
            transport.remote(["rm", "-f", "--", remote_request], timeout=10)
        except Exception:
            # The remote subcommand also removes its request in its own finally.
            pass


def public_archive_descriptor(item: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    """Describe the exact frozen tar without reading its large model body again."""
    require(item.get("sha256") == ESMC_ARCHIVE_SHA256
            and source.get("frozen_esmc_weight_sha256") == ESMC_BODY_SHA256,
            "Public archive reconstruction requires the frozen ESMC pins.")
    local = Path(item["local"])
    _regular(local)
    with tarfile.open(local, "r:") as archive:
        matches = [member for member in archive if member.isfile() and member.size == ESMC_BODY_BYTES
                   and member.name.endswith("/data/weights/esmc_300m_2024_12_v0.pth")]
    require(len(matches) == 1, "Frozen ESMC archive has an unexpected model layout.")
    member = matches[0]
    suffix_bytes = local.stat().st_size - member.offset_data - member.size
    require(0 <= member.offset_data <= 64 * 1024 and 0 <= suffix_bytes <= 64 * 1024,
            "Frozen ESMC archive contains an unexpected non-model payload.")
    with local.open("rb") as stream:
        prefix = stream.read(member.offset_data)
        stream.seek(member.offset_data + member.size)
        suffix = stream.read()
    return {"url": ESMC_URL, "body_sha256": ESMC_BODY_SHA256, "body_size_bytes": ESMC_BODY_BYTES,
            "archive_sha256": ESMC_ARCHIVE_SHA256, "archive_size_bytes": local.stat().st_size,
            "prefix_base64": base64.b64encode(prefix).decode(), "suffix_base64": base64.b64encode(suffix).decode()}


class _HTTPSRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        require(urllib.parse.urlsplit(newurl).scheme == "https", "Public model redirect is not HTTPS.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def reconstruct_public_archive(request_path: Path) -> dict[str, Any]:
    """Download public model bytes and restore their original authenticated tar."""
    request = _read(request_path)
    descriptor = request["descriptor"]
    require(descriptor["url"] == ESMC_URL and descriptor["body_sha256"] == ESMC_BODY_SHA256
            and descriptor["body_size_bytes"] == ESMC_BODY_BYTES
            and descriptor["archive_sha256"] == ESMC_ARCHIVE_SHA256,
            "Public model request differs from the frozen ESMC recipe.")
    prefix = base64.b64decode(descriptor["prefix_base64"], validate=True)
    suffix = base64.b64decode(descriptor["suffix_base64"], validate=True)
    require(len(prefix) <= 64 * 1024 and len(suffix) <= 64 * 1024
            and len(prefix) + ESMC_BODY_BYTES + len(suffix) == descriptor["archive_size_bytes"],
            "Public archive descriptor has an invalid bounded closure.")
    destination = _safe_destination(request["destination"], request["allowed_root"])
    deadline = request["deadline_unix"]
    require(type(deadline) in (int, float) and math.isfinite(deadline) and time.time() < deadline,
            "Public archive setup deadline is exhausted.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _writer_lock(destination):
        if destination.exists():
            _regular(destination)
            require(destination.stat().st_size == descriptor["archive_size_bytes"]
                    and _sha(destination) == ESMC_ARCHIVE_SHA256, "Existing public archive differs from its pin.")
            return {"status": "verified_existing", "sha256": ESMC_ARCHIVE_SHA256, "laptop_payload_bytes": 0}
        partial = destination.with_name(".g2-public-" + ESMC_ARCHIVE_SHA256 + ".partial")
        require(not partial.is_symlink(), "Public model partial is a symlink.")
        partial.unlink(missing_ok=True)
        body_hash, archive_hash, count = hashlib.sha256(), hashlib.sha256(), 0
        opener = urllib.request.build_opener(_HTTPSRedirect())
        with opener.open(ESMC_URL, timeout=min(30, deadline - time.time())) as response, partial.open("xb") as output:
            require(response.status == 200 and urllib.parse.urlsplit(response.url).scheme == "https",
                    "Public model download did not return complete HTTPS content.")
            output.write(prefix)
            archive_hash.update(prefix)
            while block := response.read(1024 ** 2):
                require(time.time() < deadline, "Public model setup reached its stop reserve.")
                count += len(block)
                require(count <= ESMC_BODY_BYTES, "Public model exceeds its frozen size.")
                body_hash.update(block)
                archive_hash.update(block)
                output.write(block)
            require(count == ESMC_BODY_BYTES and body_hash.hexdigest() == ESMC_BODY_SHA256,
                    "Public ESMC model bytes differ from their frozen pin.")
            output.write(suffix)
            archive_hash.update(suffix)
            output.flush()
            os.fsync(output.fileno())
        require(partial.stat().st_size == descriptor["archive_size_bytes"]
                and archive_hash.hexdigest() == ESMC_ARCHIVE_SHA256 and _sha(partial) == ESMC_ARCHIVE_SHA256,
                "Reconstructed closed archive differs from its original pin.")
        partial.replace(destination)
    return {"status": "verified", "sha256": ESMC_ARCHIVE_SHA256,
            "size_bytes": descriptor["archive_size_bytes"], "cloud_download_bytes": count,
            "laptop_payload_bytes": 0, "method": "exact_public_model_archive_reconstruction"}


def hydrate_public_artifact(config: Mapping[str, Any], transport, item: Mapping[str, Any],
                            source: Mapping[str, Any], deadline: float) -> bool:
    """Avoid the workstation uplink for the already-frozen public ESMC archive."""
    if item.get("sha256") != ESMC_ARCHIVE_SHA256:
        return False
    descriptor = public_archive_descriptor(item, source)
    request = {"descriptor": descriptor, "destination": item["remote"],
               "allowed_root": config["remote_campaign"],
               "deadline_unix": min(deadline - float(getattr(transport, "reserve_seconds", 900)), time.time() + 900)}
    helper = Path(__file__).resolve()
    root = str(PurePosixPath(config["remote_campaign"]) / "g2_requests/bootstrap")
    remote_helper = f"{root}/{_sha(helper)}.py"
    if transport.stat_file(remote_helper) != {"sha256": _sha(helper), "size_bytes": helper.stat().st_size}:
        transport.upload(helper, remote_helper)
        require(transport.stat_file(remote_helper) == {"sha256": _sha(helper), "size_bytes": helper.stat().st_size},
                "Public reconstruction helper transfer differs.")
    local = Path(config["state_root"]) / "public_esmc_request.json"
    _write(local, request)
    remote_request = root + "/public_esmc_request.json"
    transport.upload(local, remote_request)
    begun = time.time()
    result = json.loads(transport.remote([config["bootstrap_python"], remote_helper, "public-archive",
                        "--request", remote_request], timeout=max(1, request["deadline_unix"] - time.time())))
    require(transport.stat_file(item["remote"]) == {"sha256": item["sha256"], "size_bytes": Path(item["local"]).stat().st_size},
            "Public reconstruction readback differs from its original archive.")
    _write(Path(config["state_root"]) / "staging_receipts" / (item["sha256"] + ".json"),
           {**result, "elapsed_seconds": time.time() - begun})
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    hydrate = actions.add_parser("hydrate")
    hydrate.add_argument("--request", type=Path, required=True)
    public = actions.add_parser("public-archive")
    public.add_argument("--request", type=Path, required=True)
    posix = actions.add_parser("prepare-posix-storage")
    posix.add_argument("--request", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "prepare-posix-storage":
            result = prepare_posix_storage(_read(args.request))
        else:
            function = hydrate_request if args.action == "hydrate" else reconstruct_public_archive
            result = function(args.request)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    except (RuntimeError, OSError, ValueError, KeyError) as exc:
        # Request authorizations and SDK errors are never rendered here.
        print(f"Artifact preparation refused ({type(exc).__name__}).", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
