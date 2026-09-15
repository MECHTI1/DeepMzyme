"""RING copy of the verified chunk contract, adding a literal finalization ID."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tarfile
import tempfile

CHUNK_BYTES = 1024 * 1024


def digest_file(path):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def verify_and_extract(descriptor, local_root):
    attempt = descriptor["attempt_id"]
    if not isinstance(attempt, str) or not re.fullmatch(r"(?:attempt_\d{3,}|finalization)", attempt):
        raise ValueError(f"Invalid attempt identifier: {attempt!r}")
    basename = Path(descriptor["archive"]).name
    if basename != f"{attempt}.tar.gz":
        raise ValueError("Archive filename does not match the attempt identifier")
    expected_sha, expected_size = descriptor["archive_sha256"], int(descriptor["bytes"])
    local_root = Path(local_root).resolve()
    archives = local_root / "archives"
    archives.mkdir(parents=True, exist_ok=True)
    archive = archives / basename
    if archive.exists() and digest_file(archive) != (expected_sha, expected_size):
        raise ValueError(f"Conflicting existing archive, preserved unchanged: {archive}")
    parts = descriptor["parts"]
    if not parts:
        raise ValueError("Archive descriptor contains no parts")
    part_names = [Path(part["path"]).name for part in parts]
    if part_names != [f"{basename}.part{index:03d}" for index in range(len(parts))]:
        raise ValueError("Archive parts must have consecutive, unique part filenames")

    temporary = None
    destination = None
    joined_sha, joined_size = hashlib.sha256(), 0
    try:
        if not archive.exists():
            destination = tempfile.NamedTemporaryFile(prefix=f".{attempt}.", suffix=".tmp",
                                                       dir=archives, delete=False)
            temporary = Path(destination.name)
        for part, name in zip(parts, part_names):
            part_sha, part_size = hashlib.sha256(), 0
            with (archives / name).open("rb") as stream:
                for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
                    part_sha.update(chunk)
                    part_size += len(chunk)
                    joined_sha.update(chunk)
                    joined_size += len(chunk)
                    if destination is not None:
                        destination.write(chunk)
            if (part_sha.hexdigest(), part_size) != (part["sha256"], int(part["bytes"])):
                raise ValueError(f"Part SHA-256 or size mismatch: {name}")
        if (joined_sha.hexdigest(), joined_size) != (expected_sha, expected_size):
            raise ValueError("Joined archive SHA-256 or size does not match its descriptor")
        if destination is not None:
            destination.flush()
            os.fsync(destination.fileno())
            destination.close()
            destination = None
            try:
                # Creating a link fails if another process creates this archive;
                # an existing conflicting file can never be overwritten.
                os.link(temporary, archive)
            except FileExistsError:
                if digest_file(archive) != (expected_sha, expected_size):
                    raise ValueError(f"Conflicting existing archive, preserved unchanged: {archive}")
    finally:
        if destination is not None:
            destination.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    if digest_file(archive) != (expected_sha, expected_size):
        raise ValueError("Saved local archive failed SHA-256 or size verification")
    with tarfile.open(archive, "r:gz") as stream:
        try:
            member = stream.getmember("campaign_manifest.json")
        except KeyError as exc:
            raise ValueError("Archive does not contain campaign_manifest.json") from exc
        if not member.isfile():
            raise ValueError("Archived campaign manifest must be a regular file")
        with stream.extractfile(member) as manifest:
            manifest_sha = hashlib.sha256(manifest.read()).hexdigest()
        if manifest_sha != descriptor["manifest_sha256"]:
            raise ValueError("Archived campaign manifest SHA-256 does not match its descriptor")
        stream.extractall(local_root, filter="data")
    local_manifest_sha, _ = digest_file(local_root / "campaign_manifest.json")
    if local_manifest_sha != descriptor["manifest_sha256"]:
        raise ValueError("Extracted local campaign manifest SHA-256 does not match its descriptor")
    return dict(attempt_id=attempt, local_sha256_verified=True, archive=str(archive),
                archive_sha256=expected_sha, bytes=expected_size,
                manifest_sha256=local_manifest_sha, verified_parts=len(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        descriptor = json.loads(args.descriptor.read_text())
        receipt = verify_and_extract(descriptor, args.local_root)
    except (OSError, ValueError, KeyError, tarfile.TarError) as exc:
        parser.exit(1, f"Pilot archive verification failed: {exc}\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
