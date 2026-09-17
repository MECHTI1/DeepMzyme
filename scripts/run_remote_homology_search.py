"""Run the frozen, CPU-only development search and bind its exact outputs.

Run after audit_sequence_remoteness.py prepare and before annotate. Completed
verified searches are reused. Failed searches retain logs and need a fresh
output directory or explicit operator recovery; they are never hidden retries.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from audit_sequence_remoteness import FIELDS, protocol_at, read_json, sha256, verify_search_receipt, write_frozen


def search(root, protocol_path, binary, binary_sha256, tasks, threads, memory, timeout):
    protocol = protocol_at(protocol_path)
    manifest = read_json(root / "sequence_manifest.json")
    if sha256(protocol_path) != manifest["protocol_sha256"]:
        raise ValueError("Protocol changed after preparation")
    if sha256(binary) != binary_sha256:
        raise ValueError("Search binary does not match its expected checksum")
    version = subprocess.check_output([str(binary), "version"], text=True, timeout=15).strip()
    if version != protocol["version"]:
        raise ValueError(f"Search binary version differs from the protocol: {version}")
    for task in tasks or sorted(manifest["fasta_files"]):
        reference = manifest["fasta_files"][task]
        directory = root / task
        fasta = root / reference["path"]
        if sha256(fasta) != reference["sha256"]:
            raise ValueError("Development FASTA differs from the prepared manifest")
        if (directory / "search_receipt.json").exists():
            verify_search_receipt(root, task, manifest, protocol)
            print(json.dumps({"task": task, "status": "reused_verified_search"}), flush=True)
            continue
        output = directory / "alignments.tsv"
        log = directory / "search.log"
        if output.exists() or log.exists():
            raise ValueError(f"Unreceipted prior search exists for {task}; inspect it before recovery")
        command = [str(binary), "easy-search", str(fasta), str(fasta), str(output), str(directory / "tmp"),
                   "--search-type", "1", "-s", "7.5", "-e", "0.001", "--alignment-mode", "3",
                   "--seq-id-mode", "0", "-a", "1", "--min-seq-id", "0", "--min-aln-len", "50",
                   "-c", "0", "--cov-mode", "0", "--mask", "1", "--comp-bias-corr", "1",
                   "--max-seqs", "1000000", "--max-accept", "1000000", "--max-rejected", "1000000",
                   "--threads", str(threads), "--split-memory-limit", memory,
                   "--format-mode", "4", "--format-output", ",".join(FIELDS)]
        if reference["n_sequences"] > 1000000:
            raise ValueError("Development database exceeds the frozen result caps")
        print(json.dumps({"task": task, "status": "search_started", "sequences": reference["n_sequences"]}), flush=True)
        started = time.monotonic()
        with log.open("x") as handle:
            completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, timeout=timeout)
        elapsed = time.monotonic() - started
        if completed.returncode:
            raise RuntimeError(f"MMseqs2 exited {completed.returncode}; inspect {log}")
        receipt = {"schema_version": 1, "program": "MMseqs2", "release": protocol["release"],
                   "version": version, "binary_sha256": binary_sha256, "command": command,
                   "fasta_sha256": sha256(fasta), "alignments_sha256": sha256(output),
                   "database_convention": protocol["database_convention"],
                   "protocol_sha256": sha256(protocol_path), "held_out_evaluation": False,
                   "elapsed_seconds": elapsed, "search_log_sha256": sha256(log),
                   "max_child_rss_kib_process_lifetime": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                   "runner_sha256": sha256(Path(__file__).resolve())}
        write_frozen(directory / "search_receipt.json", receipt)
        verify_search_receipt(root, task, manifest, protocol)
        print(json.dumps({"task": task, "status": "search_completed", "elapsed_seconds": elapsed}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--task", choices=("metal", "ec"), action="append")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--split-memory-limit", default="2G")
    parser.add_argument("--timeout-seconds", type=float, default=900)
    args = parser.parse_args()
    if args.threads < 1 or args.timeout_seconds <= 0:
        parser.error("Threads and timeout must be positive")
    search(args.output_dir.resolve(), args.protocol.resolve(), args.binary.resolve(),
           args.binary_sha256, args.task, args.threads, args.split_memory_limit, args.timeout_seconds)


if __name__ == "__main__":
    main()
