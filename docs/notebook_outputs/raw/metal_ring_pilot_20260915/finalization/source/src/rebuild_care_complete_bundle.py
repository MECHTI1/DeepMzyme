"""Build a local data-only successor to v11 after a successful CARE cache audit.

Preserves the base bundle's selected roots and scientific memberships. Writes
checksums and verifies every archived regular file and symlink. Never uploads.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

from build_colab_bundle import build_bundle, PROJECT_ROOT
from complete_care_caches import atomic_json, CARE_ROOT
from structure_store import sha256_file


def file_identity(path):
    return {"size": path.stat().st_size, "sha256": sha256_file(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--care-audit", type=Path, required=True)
    parser.add_argument("--output-bundle", type=Path, required=True)
    parser.add_argument("--release", default="v12")
    args = parser.parse_args()
    output = args.output_bundle.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite release archive: {output}")
    audit = json.loads(args.care_audit.read_text())
    if not audit.get("complete") or audit.get("failures"):
        raise ValueError("CARE audit must be complete and failure-free")
    data = PROJECT_ROOT / "DeepMzyme_Data"
    for row in audit["validated_files"]:
        actual = file_identity(data / row["path"])
        if actual != {"size": row["bytes"], "sha256": row["sha256"]}:
            raise ValueError(f"Local cache differs from audited cache: {row['path']}")
    base = json.loads(args.base_manifest.read_text())
    roots = [PROJECT_ROOT / p for p in base["included_roots"]]
    files, symlinks = set(), {}
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(root)
        for path in [root, *root.rglob("*")] if not root.is_symlink() else [root]:
            if path.is_symlink():
                symlinks[str(path.relative_to(PROJECT_ROOT))] = os.readlink(path)
            elif path.is_file():
                files.add(path)
    ordered = sorted(files)
    with ThreadPoolExecutor(max_workers=4) as pool:
        identities = dict(zip(
            (str(p.relative_to(PROJECT_ROOT)) for p in ordered), pool.map(file_identity, ordered),
        ))
    # Membership and coordinate bytes must remain identical to the base release.
    changes = {name for name, identity in base["files"].items() if identities.get(name) != identity}
    allowed_prefix = "DeepMzyme_Data/updated_feature_extraction/"
    if any(not name.startswith(allowed_prefix) for name in changes):
        raise ValueError(f"Unexpected changes from base release: {sorted(changes)}")
    coverage = base["feature_coverage"]
    for split, values in audit["splits"].items():
        coverage[f"{CARE_ROOT}/{split}"] = {
            "structures": values["structures"],
            **{f"missing_{key}": count for key, count in values["missing"].items()},
            "audit": "all structures: finite 960D ESM, sequence/residue alignment, external schema and PROPKA, RING syntax",
        }
    manifest = {
        **base, "bundle_filename": output.name,
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "source_git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
        "source_git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True)),
        "files": identities, "symlinks": symlinks, "feature_coverage": coverage,
        "parent_bundle": {"filename": base["bundle_filename"], "manifest_sha256": sha256_file(args.base_manifest)},
        "care_audit_sha256": sha256_file(args.care_audit),
        "changed_existing_files": sorted(changes),
        "added_files": sorted(set(identities) - set(base["files"])),
        "caveats": [c for c in base["caveats"] if not c.startswith("CARE caches are incomplete")]
                    + ["Local release only; no remote publication performed.",
                       f"{len(changes)} prior external files were regenerated because PROPKA was unavailable; shared users of those files receive the repaired features."],
        "validation_scope": "All archived file hashes; complete CARE cache audit; no held-out model evaluation.",
    }
    metadata_dir = data / "bundle_metadata" / args.release
    atomic_json(metadata_dir / "bundle_manifest.json", manifest)
    atomic_json(metadata_dir / "care_cache_audit.json", audit)
    selected = roots + [metadata_dir]
    build_bundle(selected, output_bundle=output)
    expected = dict(identities)
    expected.update({str(p.relative_to(PROJECT_ROOT)): file_identity(p) for p in metadata_dir.iterdir() if p.is_file()})
    seen, seen_links = set(), {}
    with tarfile.open(output, "r|gz") as archive:
        for member in archive:
            name = member.name.removeprefix("./")
            if member.isfile():
                with archive.extractfile(member) as stream:
                    actual = {"size": member.size, "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}
                if expected.get(name) != actual or name in seen:
                    raise ValueError(f"Archive integrity mismatch: {name}")
                seen.add(name)
            elif member.issym():
                seen_links[name] = member.linkname
            elif not member.isdir():
                raise ValueError(f"Unexpected archive member type: {name}")
    if seen != set(expected) or seen_links != symlinks:
        raise ValueError("Archive membership differs from manifest")
    checksum = sha256_file(output)
    output.with_name(output.name + ".sha256").write_text(f"{checksum}  {output.name}\n")
    atomic_json(output.with_name(output.name + ".manifest.json"), manifest)
    atomic_json(output.with_name(output.name + ".validation.json"), {
        "sha256": checksum, "bytes": output.stat().st_size, "verified_regular_files": len(seen),
        "verified_symlinks": seen_links, "care_audit_complete": True, "published": False,
    })
    print(f"Verified local bundle: {output}\nSHA256: {checksum}")


if __name__ == "__main__":
    main()
