"""Copy exact RING text evidence after verified stop and post-stop analysis."""
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[4]
PLAN = Path(__file__).resolve().parent
CAMPAIGN = ROOT / "DeepMzyme_Data/notebook_outputs/campaigns/metal_ring_pilot_v1_20260915"
DESTINATION = ROOT / "docs/notebook_outputs/raw/metal_ring_pilot_20260915"
TEXT_SUFFIXES = {".json", ".csv", ".md", ".txt", ".log", ".py", ".ipynb", ".toml", ".yaml", ".yml", ".sha256"}


def proof(path):
    content = path.read_bytes()
    return {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}


def collect():
    closeout = CAMPAIGN / "host_closeout_allocation3"
    closed = json.loads((closeout / "post_stop_receipt.json").read_text())
    if closed["status"] != "stopped_verified_and_final_archive_preserved":
        raise ValueError("Verified actual stop and closeout are required")
    analysis = CAMPAIGN / "analysis/ring_post_stop_analysis.json"
    report = json.loads(analysis.read_text())
    if report["terminal_state"]["status"] not in {"completed", "budget_stopped"}:
        raise ValueError("Terminal post-stop analysis is required")
    if not (CAMPAIGN / "host_closeout_allocation3_drive_receipt.json").is_file():
        raise ValueError("Persist and verify the post-stop closeout archive first")
    sources = {}
    for name in ("finalization", "analysis", "drive_readbacks", "host_closeout_allocation3"):
        directory = CAMPAIGN / name
        for path in sorted(directory.rglob("*")):
            relative = path.relative_to(CAMPAIGN)
            if path.is_file() and path.suffix in TEXT_SUFFIXES and "archives" not in relative.parts:
                sources[str(relative)] = path
    for path in sorted((CAMPAIGN / "archives").glob("*_archive.json")):
        sources["attempt_archive_descriptors/" + path.name] = path
    sources["host_closeout_allocation3_drive_receipt.json"] = CAMPAIGN / "host_closeout_allocation3_drive_receipt.json"
    for name in ("PRE_EXECUTION_REVIEW_20260915.md", "prepared_execution_review_20260915_drive_receipt.json"):
        sources["preparation/" + name] = PLAN / name
    for name in ("ring_input_audit.json", "training_cache_audit.json", "audit_provenance.json"):
        sources["preparation/local_audit/" + name] = PLAN / "local_audit" / name
    sources["analysis/analyze_completed_ring.py"] = PLAN / "analyze_completed_ring.py"
    sources["analysis/collect_portable_evidence.py"] = Path(__file__)
    inventory = {}
    for name, source in sources.items():
        expected = proof(source)
        target = DESTINATION / name
        if target.exists() and proof(target) != expected:
            raise ValueError("Existing portable evidence differs; preserved: " + str(target))
        inventory[name] = {**expected, "source": str(source)}
    index = DESTINATION / "portable_evidence_inventory.json"
    encoded = json.dumps({"profile": "metal_ring_pilot_v1", "exact_copied_files": inventory}, indent=2) + "\n"
    if index.exists() and index.read_text() != encoded:
        raise ValueError("Existing portable inventory differs; preserved")
    for name, source in sources.items():
        target = DESTINATION / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
        if proof(source) != proof(target) or proof(target) != {key: inventory[name][key] for key in ("sha256", "bytes")}:
            raise ValueError("Evidence changed during copying: " + name)
    if not index.exists():
        with index.open("x") as stream:
            stream.write(encoded)
    return {"files_verified": len(inventory), "bytes": sum(row["bytes"] for row in inventory.values()),
            "destination": str(DESTINATION), "inventory_sha256": proof(index)["sha256"]}


if __name__ == "__main__":
    print(json.dumps(collect(), indent=2))
