"""Development-only sequence remoteness; never creates folds or reads predictions.

``prepare`` freezes exact saved memberships and sequence provenance. ``annotate``
uses an externally recorded MMseqs2 search to freeze counts before prediction
joins. Coordinate-chain diagnostics and full-protein endpoints are separate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


BINS = (">30", "(20,30]", "(15,20]", "<=15")
FIELDS = ("query target nident alnlen qstart qend qlen tstart tend tlen "
          "evalue bits qaln taln").split()
DEFAULT_PROTOCOL = {
    "schema_version": 1,
    "program": "MMseqs2",
    "version": "8cc5ce367b5638c4306c2d7cfc652dd099a4643f",
    "release": "18-8cc5c",
    "sensitivity": 7.5,
    "max_evalue": 0.001,
    "min_residue_pairs": 50,
    "primary_shorter_coverage": 0.8,
    "audit_shorter_coverage": 0.5,
    "identity_denominator": "aligned_columns_including_internal_gaps",
    "component_identity_threshold": 0.2,
    "component_coverage": 0.5,
    "primary_remote_threshold": 0.2,
    "extreme_remote_threshold": 0.15,
    "minimum_components_per_class_per_primary_stratum": 10,
    "database_convention": "fixed_task_development_database",
    "no_qualifying_hit": "unclassified_not_zero_identity",
    "held_out_evaluation": False,
}


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_frozen(path, value):
    """Idempotent writes, refusing to replace an earlier frozen artifact."""
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Refusing to replace frozen artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def resolve(base, value):
    path = Path(value)
    return path if path.is_absolute() else Path(base) / path


def protocol_at(path):
    protocol = read_json(path)
    for key, value in DEFAULT_PROTOCOL.items():
        if protocol.get(key) != value:
            raise ValueError(f"Protocol field {key} must be {value!r}")
    return protocol


def identity_bin(identity):
    if identity is None:
        return None
    if not math.isfinite(identity) or not 0 <= identity <= 1:
        raise ValueError("Identity must be a finite fraction in [0, 1]")
    if identity > 0.30:
        return ">30"
    if identity > 0.20:
        return "(20,30]"
    if identity > 0.15:
        return "(15,20]"
    return "<=15"


def saved_membership(value, task, run_id):
    if task not in {"metal", "ec"}:
        raise ValueError(f"Unsupported task: {task}")
    # This is a saved development split; test records are never iterated.
    membership = value["retained_split_identity"]
    result = {"task": task, "run_id": run_id}
    for split in ("train", "val"):
        saved = membership["validation" if split == "val" else split]
        examples = saved["examples"]
        if saved.get("ordered_examples_sha256") != digest(examples):
            raise ValueError(f"Invalid saved membership digest in {run_id}/{split}")
        if saved.get("n_examples") != len(examples):
            raise ValueError(f"Invalid saved example count in {run_id}/{split}")
        rows = []
        for row in examples:
            target = row[f"y_{task}"]
            if not isinstance(target, int) or target < 0:
                raise ValueError(f"Missing active target in {run_id}/{split}")
            rows.append({"example_id": row["pocket_id"],
                         "structure_id": row["structure_id"],
                         "group_id": str(row["group"]), "target": target})
        if len({row["example_id"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate pocket identifiers in {run_id}/{split}")
        result[split] = rows
    for key in ("example_id", "group_id"):
        if {r[key] for r in result["train"]} & {r[key] for r in result["val"]}:
            raise ValueError(f"Train/validation overlap in {run_id}: {key}")
    result["membership_sha256"] = digest({s: result[s] for s in ("train", "val")})
    return result


def _development_source(summary):
    # Recorded data location must explicitly identify external training membership.
    for key in ("structure_dir", "summary_csv"):
        value = summary.get(key)
        if not value or "train" not in Path(value).parts or "test" in Path(value).parts:
            raise ValueError(f"Uncertified external-training source: {key}={value!r}")
    if summary.get("controlled_ec_auxiliary"):
        raise ValueError("This protocol is for standalone task memberships")


def load_memberships(ledger_path, fold_plan=None):
    ledger = read_json(ledger_path)
    base = Path(ledger_path).parent
    memberships = []
    used = set()
    for run in ledger["runs"]:
        key = (run["task"], run["run_id"])
        if key in used:
            raise ValueError(f"Duplicate run: {key}")
        used.add(key)
        summary_path = resolve(base, run.get("dataset_summary_path") or
                               str(Path(run["run_dir"]) / "dataset_summary.json"))
        summary = read_json(summary_path)
        _development_source(summary)
        if summary.get("task") != run["task"]:
            raise ValueError("Ledger and dataset task disagree")
        expected = run.get("expected_sha256", {}).get("dataset_summary.json")
        if not expected or sha256(summary_path) != expected:
            raise ValueError("Dataset summary checksum missing or changed")
        if run["task"] == "metal" and summary.get("metal_label_scheme") not in ("four_class", "merge_fe_class_viii"):
            raise ValueError("Remote comparison accepts only direct-four metal runs")
        if run["task"] == "ec" and summary.get("ec_label_depth") != 1:
            raise ValueError("Remote comparison accepts only EC depth 1")
        item = saved_membership(summary, run["task"], run["run_id"])
        n_classes = 4 if run["task"] == "metal" else 7
        if any(row["target"] >= n_classes for split in ("train", "val") for row in item[split]):
            raise ValueError("Active target outside frozen task vocabulary")
        item.update(dataset_summary_path=str(summary_path.resolve()),
                    dataset_summary_sha256=sha256(summary_path))
        memberships.append(item)
    if fold_plan:
        plan = read_json(fold_plan)
        if plan.get("held_out_evaluation") is not False:
            raise ValueError("Fold plan must explicitly disable held-out evaluation")
        development = {}
        for membership in memberships:
            if membership["task"] != "metal":
                continue
            for split in ("train", "val"):
                for row in membership[split]:
                    key = row["example_id"]
                    if key in development and development[key] != row:
                        raise ValueError("Allowlisted runs disagree on development target or identity")
                    development[key] = row
        seen = []
        for name, reference in sorted(plan["references"].items()):
            if not name.startswith("fold_"):
                continue
            item = saved_membership(reference, "metal", name)
            rows = {r["example_id"]: r for split in ("train", "val") for r in item[split]}
            # Prepared folds use native-six labels; membership identity is checked
            # separately, then targets are mapped from the direct-four ledger.
            if set(rows) != set(development):
                raise ValueError("Fold cohort differs from allowlisted development membership")
            for row in rows.values():
                original = development[row["example_id"]]
                if any(row[k] != original[k] for k in ("structure_id", "group_id")):
                    raise ValueError("Fold group or structure changed")
                row["target"] = original["target"]
            item["membership_sha256"] = digest({s: item[s] for s in ("train", "val")})
            item["fold_plan_sha256"] = sha256(Path(fold_plan))
            item["planned_fold"] = True
            seen.extend(r["example_id"] for r in item["val"])
            memberships.append(item)
        if len(seen) != len(development) or set(seen) != set(development):
            raise ValueError("Prepared folds must cover each development example exactly once")
    return ledger, memberships


def _sequence(sequence):
    sequence = "".join(sequence.split()).upper()
    if not sequence or any(c not in "ACDEFGHIKLMNPQRSTVWYBXZJUO" for c in sequence):
        raise ValueError("Missing or invalid protein sequence")
    return sequence


def collect_sequences(ledger, ledger_base, memberships, endpoint, overrides=None):
    sources = ledger.get("sequence_sources", ledger)
    esm_root = resolve(ledger_base, sources["esm_embeddings_dir"])
    care_path = sources.get("care_source_csv")
    care = {}
    if care_path:
        care_path = resolve(ledger_base, care_path)
        with care_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("split") != "train" or row.get("source_split") != "train":
                    raise ValueError("CARE source table includes non-training records")
                identifier = row["protein_id"]
                sequence = _sequence(row["sequence"])
                if identifier in care and care[identifier] != sequence:
                    raise ValueError(f"Conflicting CARE sequence: {identifier}")
                care[identifier] = sequence
        care_sha = sha256(care_path)
    custom = {}
    if overrides:
        for row in read_json(overrides)["sequences"]:
            key = (row["task"], row["structure_id"], row["chain_id"])
            if key in custom:
                raise ValueError("Duplicate reviewed chain mapping")
            custom[key] = row
    structures = {}
    for membership in memberships:
        for split in ("train", "val"):
            for row in membership[split]:
                key = (membership["task"], row["structure_id"])
                if key in structures and structures[key] != row["group_id"]:
                    raise ValueError("One structure maps to multiple groups")
                structures[key] = row["group_id"]
    records = []
    for (task, structure), group in sorted(structures.items()):
        matches = list(esm_root.glob(f"{structure}_chain_*_esmc.pt.json"))
        if not matches:
            records.append(dict(task=task, structure_id=structure, group_id=group,
                                status="missing_sequence", scope=endpoint))
            continue
        for path in sorted(matches):
            sidecar = read_json(path)
            sequence = _sequence(sidecar["source_sequence"])
            seq_sha = hashlib.sha256(sequence.encode()).hexdigest()
            if seq_sha != sidecar.get("source_sequence_sha256"):
                raise ValueError(f"Invalid sidecar sequence hash: {path}")
            if sidecar.get("structure_id") != structure:
                raise ValueError(f"Sidecar structure mismatch: {path}")
            record = dict(task=task, structure_id=structure, group_id=group,
                          chain_id=str(sidecar["chain_id"]), sequence=sequence,
                          scope="represented_coordinate_chain", status="resolved",
                          site_chain_coverage_certified=False,
                          source_path=str(path.resolve()), source_sha256=sha256(path),
                          coordinate_sequence_sha256=seq_sha)
            # The trainer lowercases full accession grouping keys; CARE keeps
            # canonical uppercase accessions. Do not truncate either identifier.
            care_identifier = group.upper()
            if task == "ec" and care_identifier in care:
                full = care[care_identifier]
                relationship = ("exact" if full == sequence else "coordinate_subsequence"
                                if sequence in full else "source_subsequence"
                                if full in sequence else "sequence_disagreement")
                record["care_relationship"] = relationship
                record["care_protein_id"] = care_identifier
                record["care_sequence_sha256"] = hashlib.sha256(full.encode()).hexdigest()
                if endpoint == "full_protein":
                    record.update(sequence=full, scope="full_protein", source_path=str(care_path.resolve()),
                                  source_sha256=care_sha, site_chain_coverage_certified=True,
                                  status="resolved" if relationship == "exact" else "unresolved_mapping")
            elif endpoint == "full_protein":
                record.update(scope="full_protein", status="uncertified_full_protein")
            if (task, structure, record["chain_id"]) in custom:
                override = custom[(task, structure, record["chain_id"])]
                # Explicit reviewed mapping evidence, not silent replacement.
                for field in ("mapping_evidence", "source_path", "source_sha256", "sequence",
                              "scope", "site_chain_coverage_certified"):
                    if not override.get(field):
                        raise ValueError(f"Reviewed mapping lacks {field}")
                evidence = resolve(ledger_base, override["source_path"])
                if sha256(evidence) != override["source_sha256"]:
                    raise ValueError("Reviewed sequence source checksum mismatch")
                if override["scope"] != endpoint:
                    raise ValueError("Reviewed sequence scope differs from selected endpoint")
                if any(override.get(k, record[k]) != record[k] for k in ("task", "structure_id", "group_id", "chain_id")):
                    raise ValueError("Reviewed sequence mapping cannot change membership identifiers")
                for field in ("mapping_evidence", "source_path", "source_sha256", "sequence",
                              "scope", "site_chain_coverage_certified"):
                    record[field] = override[field]
                record["status"] = "resolved"
                record["source_path"] = str(evidence.resolve())
                record["sequence"] = _sequence(override["sequence"])
            record["sequence_sha256"] = hashlib.sha256(record["sequence"].encode()).hexdigest()
            record["sequence_id"] = "s_" + digest([task, structure, record["chain_id"], endpoint])[:24]
            records.append(record)
    return records


def prepare(args):
    root = Path(args.output_dir)
    protocol = protocol_at(args.protocol)
    ledger, memberships = load_memberships(args.reuse_ledger, args.fold_plan)
    records = collect_sequences(ledger, Path(args.reuse_ledger).parent, memberships,
                                args.endpoint, args.sequence_records)
    blockers = [{k: r.get(k) for k in ("task", "structure_id", "group_id", "status")}
                for r in records if r["status"] != "resolved"]
    report = {"schema_version": 1, "endpoint": args.endpoint,
              "primary_full_protein_certified": args.endpoint == "full_protein" and not blockers,
              "n_sequences": len(records), "blockers": blockers,
              "care_discrepancies": [{k: r.get(k) for k in ("group_id", "care_relationship",
                                      "coordinate_sequence_sha256", "care_sequence_sha256")}
                                     for r in records if r.get("care_relationship") not in (None, "exact")],
              "held_out_evaluation": False, "new_split_created": False}
    write_frozen(root / "preparation_report.json", report)
    if args.endpoint == "full_protein" and blockers:
        raise ValueError("Full-protein endpoint blocked; inspect preparation_report.json")
    manifest = {"schema_version": 1, "endpoint": args.endpoint, "sequences": records,
                "reuse_ledger_sha256": sha256(Path(args.reuse_ledger)),
                "protocol_sha256": sha256(Path(args.protocol)), "fasta_files": {}}
    for task in sorted({m["task"] for m in memberships}):
        task_records = [r for r in records if r["task"] == task and r["status"] == "resolved"]
        path = root / task / "development.fasta"
        content = "".join(f">{r['sequence_id']}\n{r['sequence']}\n" for r in task_records)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_text() != content:
            raise ValueError(f"Refusing to replace frozen FASTA: {path}")
        path.write_text(content)
        manifest["fasta_files"][task] = {"path": f"{task}/development.fasta", "sha256": sha256(path),
                                         "n_sequences": len(task_records)}
    write_frozen(root / "memberships.json", {"schema_version": 1, "memberships": memberships})
    manifest["memberships_sha256"] = sha256(root / "memberships.json")
    write_frozen(root / "sequence_manifest.json", manifest)
    write_frozen(root / "preparation_freeze.json", {
        "sequence_manifest_sha256": sha256(root / "sequence_manifest.json"),
        "memberships_sha256": sha256(root / "memberships.json"),
        "protocol_sha256": sha256(Path(args.protocol)),
    })
    write_frozen(root / "protocol.json", protocol)
    return report


def alignment_metrics(row, sequences):
    q, t = sequences[row["query"]], sequences[row["target"]]
    qaln, taln = row["qaln"].upper(), row["taln"].upper()
    alnlen = int(row["alnlen"])
    if len(qaln) != alnlen or len(taln) != alnlen or alnlen == 0:
        raise ValueError("Alignment columns disagree with alnlen")
    if any(a == b == "-" for a, b in zip(qaln, taln)):
        raise ValueError("Alignment contains double-gap columns")
    pairs = sum(a != "-" and b != "-" for a, b in zip(qaln, taln))
    exported_nident = int(row["nident"])
    nident = sum(a == b and a != "-" for a, b in zip(qaln, taln))
    # Pinned convertalignments.cpp reconstructs nident from serialized seqId,
    # rounding seqId * alnLen. It can lose one identical residue. Bound aligned
    # sequences permit an exact recount without moving threshold boundaries.
    if nident - exported_nident not in (0, 1):
        raise ValueError("nident disagrees with exact aligned count beyond verified rounding")
    if any(a == b and a not in "ACDEFGHIKLMNPQRSTVWY-" for a, b in zip(qaln, taln)):
        raise ValueError("Ambiguous identical residue requires an explicit identity policy")
    spans = []
    for prefix, alignment, record in (("q", qaln, q), ("t", taln, t)):
        length, start, end = (int(row[prefix + suffix]) for suffix in ("len", "start", "end"))
        ungapped = alignment.replace("-", "")
        if length != len(record["sequence"]) or not 1 <= start <= end <= length:
            raise ValueError("Alignment coordinates or sequence length invalid")
        if record["sequence"][start - 1:end] != ungapped:
            raise ValueError("Alignment does not match frozen input sequence")
        spans.append(len(ungapped) / length)
    qlen, tlen = int(row["qlen"]), int(row["tlen"])
    coverage = spans[0] if qlen < tlen else spans[1] if tlen < qlen else min(spans)
    evalue = float(row["evalue"])
    if not math.isfinite(evalue) or evalue < 0:
        raise ValueError("Invalid alignment E-value")
    return {"query": row["query"], "target": row["target"],
            "identity": nident / alnlen, "shorter_coverage": coverage,
            "residue_pairs": pairs, "evalue": evalue,
            "nident_recount_minus_export": nident - exported_nident}


def read_alignments(path, sequences, diagnostics=None):
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != FIELDS:
            raise ValueError(f"Expected MMseqs2 --format-mode 4 columns: {FIELDS}")
        for row in reader:
            metrics = alignment_metrics(row, sequences)
            if diagnostics is not None:
                diagnostics["alignment_rows"] += 1
                diagnostics["nident_recount_differs_from_export"] += metrics["nident_recount_minus_export"] != 0
            yield metrics


class Components:
    def __init__(self, values):
        self.parents = {v: v for v in values}

    def root(self, value):
        while self.parents[value] != value:
            self.parents[value] = self.parents[self.parents[value]]
            value = self.parents[value]
        return value

    def join(self, a, b):
        a, b = self.root(a), self.root(b)
        self.parents[max(a, b)] = min(a, b)


def annotate_task(records, memberships, alignments, protocol):
    groups = {r["group_id"] for r in records}
    components = Components(groups)
    by_id = {r["sequence_id"]: r for r in records if r.get("sequence_id")}
    by_group = defaultdict(list)
    for r in records:
        by_group[r["group_id"]].append(r)
    identical = {}
    for r in records:
        if r["status"] == "resolved":
            key = r["sequence_sha256"]
            if key in identical:
                components.join(r["group_id"], identical[key])
            identical[key] = r["group_id"]
    hits = defaultdict(list)
    for hit in alignments:
        if hit["evalue"] > protocol["max_evalue"] or hit["residue_pairs"] < protocol["min_residue_pairs"]:
            continue
        if hit["shorter_coverage"] < protocol["audit_shorter_coverage"]:
            continue
        a, b = by_id[hit["query"]], by_id[hit["target"]]
        if hit["identity"] > protocol["component_identity_threshold"]:
            components.join(a["group_id"], b["group_id"])
        hits[hit["query"]].append((hit["target"], hit))
        if hit["query"] != hit["target"]:
            hits[hit["target"]].append((hit["query"], hit))
    output = []
    for membership in memberships:
        train_structures = {r["structure_id"] for r in membership["train"]}
        train_ids = {r["sequence_id"] for r in records if r["structure_id"] in train_structures
                     and r["status"] == "resolved"}
        training_incomplete = any(r["status"] != "resolved" for r in records
                                  if r["structure_id"] in train_structures)
        for row in membership["val"]:
            group_records = by_group[row["group_id"]]
            item = {**row, "run_id": membership["run_id"], "task": membership["task"],
                    "component_id": "c_" + digest([membership["task"], components.root(row["group_id"])])[:20],
                    "endpoint": group_records[0]["scope"]}
            for prefix, coverage in (("", protocol["primary_shorter_coverage"]),
                                     ("audit50_", protocol["audit_shorter_coverage"])):
                candidates = [(target, hit) for r in group_records if r["status"] == "resolved"
                              for target, hit in hits[r["sequence_id"]]
                              if target in train_ids and hit["shorter_coverage"] >= coverage]
                best = max(candidates, key=lambda x: x[1]["identity"], default=None)
                status = ("unresolved_sequence" if any(r["status"] != "resolved" for r in group_records)
                          else "incomplete_training_sequences" if training_incomplete
                          else "qualifying_hit" if best else "no_qualifying_hit")
                identity = best[1]["identity"] if best else None
                # Partial mapping permits a lower-bound diagnostic, never a certified bin.
                item[prefix + "status"] = status
                item[prefix + "max_identity"] = identity
                item[prefix + "bin"] = identity_bin(identity) if status == "qualifying_hit" else None
                item[prefix + "nearest_training_sequence_id"] = best[0] if best else None
            output.append(item)
    return output


def support_counts(rows, protocol):
    output = []
    keys = sorted({(r["task"], r["run_id"]) for r in rows})
    for task, run_id in keys:
        subset = [r for r in rows if (r["task"], r["run_id"]) == (task, run_id)]
        vocabulary = list(range(4 if task == "metal" else 7))
        for name in (*BINS, "<=20", "unclassified"):
            cases = [r for r in subset if (r["bin"] in ("(15,20]", "<=15") if name == "<=20"
                                          else r["bin"] is None if name == "unclassified"
                                          else r["bin"] == name)]
            counts = []
            for target in vocabulary:
                selected = [r for r in cases if r["target"] == target]
                counts.append({"target": target, "pockets": len(selected),
                               "groups": len({r["group_id"] for r in selected}),
                               "components": len({r["component_id"] for r in selected})})
            output.append({"task": task, "run_id": run_id, "bin": name, "classes": counts,
                           "support_gate_passed": all(c["components"] >= protocol[
                               "minimum_components_per_class_per_primary_stratum"] for c in counts)})
    return output


def verify_search_receipt(root, task, manifest, protocol):
    path = root / task / "search_receipt.json"
    receipt = read_json(path)
    if receipt.get("program") != "MMseqs2" or receipt.get("version") != protocol["version"]:
        raise ValueError("Search receipt must identify pinned MMseqs2 release 18")
    if receipt.get("release") != protocol["release"] or len(receipt.get("binary_sha256", "")) != 64:
        raise ValueError("Search receipt lacks release or binary checksum")
    fasta = root / manifest["fasta_files"][task]["path"]
    alignment = root / task / "alignments.tsv"
    if sha256(fasta) != manifest["fasta_files"][task]["sha256"] or receipt.get("fasta_sha256") != sha256(fasta):
        raise ValueError("Search FASTA identity changed")
    if receipt.get("alignments_sha256") != sha256(alignment):
        raise ValueError("Search alignment checksum mismatch")
    command = receipt.get("command", [])
    if len(command) < 6 or command[1] != "easy-search":
        raise ValueError("Search receipt must record the full easy-search invocation")
    if any(Path(command[index]).resolve() != fasta.resolve() for index in (2, 3)):
        raise ValueError("Search query and database must both be the frozen development FASTA")
    if Path(command[4]).resolve() != alignment.resolve():
        raise ValueError("Search receipt output path differs from frozen alignments")
    required = {"-s": "7.5", "-e": "0.001", "--alignment-mode": "3", "--seq-id-mode": "0",
                "-a": "1", "--min-seq-id": "0", "--min-aln-len": "50", "-c": "0",
                "--cov-mode": "0", "--mask": "1", "--comp-bias-corr": "1", "--search-type": "1"}
    for flag, value in required.items():
        if command.count(flag) != 1:
            raise ValueError(f"Search receipt needs one {flag}")
        actual = command[command.index(flag) + 1]
        if float(actual) != float(value):
            raise ValueError(f"Search flag differs from frozen protocol: {flag}")
    n_sequences = manifest["fasta_files"][task]["n_sequences"]
    for flag in ("--max-seqs", "--max-accept", "--max-rejected"):
        if command.count(flag) != 1 or int(command[command.index(flag) + 1]) < n_sequences:
            raise ValueError("Search hit cap below development database size")
    if receipt.get("database_convention") != protocol["database_convention"]:
        raise ValueError("E-value/database convention is not frozen")
    return alignment, sha256(path)


def annotate(args):
    root = Path(args.output_dir)
    protocol = protocol_at(args.protocol)
    manifest = read_json(root / "sequence_manifest.json")
    freeze = read_json(root / "preparation_freeze.json")
    for filename in ("sequence_manifest", "memberships"):
        if freeze[filename + "_sha256"] != sha256(root / (filename + ".json")):
            raise ValueError(f"Frozen {filename} changed after preparation")
    if manifest["memberships_sha256"] != sha256(root / "memberships.json"):
        raise ValueError("Membership manifest checksum mismatch")
    if sha256(Path(args.protocol)) != manifest["protocol_sha256"]:
        raise ValueError("Protocol changed after sequence preparation")
    if args.reuse_ledger and sha256(Path(args.reuse_ledger)) != manifest["reuse_ledger_sha256"]:
        raise ValueError("Reuse ledger changed after preparation")
    memberships = read_json(root / "memberships.json")["memberships"]
    rows, receipts, alignment_diagnostics = [], {}, {}
    for task in sorted(manifest["fasta_files"]):
        records = [r for r in manifest["sequences"] if r["task"] == task]
        alignment, receipt = verify_search_receipt(root, task, manifest, protocol)
        sequences = {r["sequence_id"]: r for r in records if r["status"] == "resolved"}
        diagnostics = {"alignment_rows": 0, "nident_recount_differs_from_export": 0,
                       "identity_numerator": "exact_identical_non_gap_residues_in_bound_alignments",
                       "accepted_recount_minus_export": [0, 1],
                       "ambiguous_identical_residues": "refused",
                       "pinned_converter_source": "https://raw.githubusercontent.com/soedinglab/MMseqs2/18-8cc5c/src/util/convertalignments.cpp"}
        rows.extend(annotate_task(records, [m for m in memberships if m["task"] == task],
                                  read_alignments(alignment, sequences, diagnostics), protocol))
        receipts[task] = receipt
        alignment_diagnostics[task] = diagnostics
    result = {"schema_version": 1, "endpoint": manifest["endpoint"], "rows": rows,
              "protocol_sha256": sha256(Path(args.protocol)),
              "sequence_manifest_sha256": sha256(root / "sequence_manifest.json"),
              "memberships_sha256": sha256(root / "memberships.json"),
              "alignment_diagnostics": alignment_diagnostics,
              "search_receipts_sha256": receipts, "held_out_evaluation": False}
    write_frozen(root / "remoteness_manifest.json", result)
    counts = {"schema_version": 1, "counts_only": True, "prediction_metrics_accessed": False,
              "protocol_sha256": sha256(Path(args.protocol)),
              "remoteness_manifest_sha256": sha256(root / "remoteness_manifest.json"),
              "primary_shorter_sequence_coverage": protocol["primary_shorter_coverage"],
              "support": support_counts(rows, protocol),
              "sensitivity_audit": {
                  "shorter_sequence_coverage": protocol["audit_shorter_coverage"],
                  "support": support_counts([
                      dict(row, bin=row["audit50_bin"], status=row["audit50_status"],
                           max_identity=row["audit50_max_identity"]) for row in rows
                  ], protocol),
              }}
    write_frozen(root / "counts_freeze.json", counts)
    return {"n_rows": len(rows), "endpoint": manifest["endpoint"],
            "counts_freeze_sha256": sha256(root / "counts_freeze.json")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--reuse-ledger", required=True)
    prep.add_argument("--protocol", required=True)
    prep.add_argument("--output-dir", required=True)
    prep.add_argument("--fold-plan")
    prep.add_argument("--sequence-records")
    prep.add_argument("--endpoint", choices=("represented_coordinate_chain", "full_protein"),
                      default="represented_coordinate_chain")
    ann = sub.add_parser("annotate")
    ann.add_argument("--protocol", required=True)
    ann.add_argument("--reuse-ledger")
    ann.add_argument("--output-dir", required=True)
    ann.add_argument("--counts-only", action="store_true", help="Always counts only; no prediction input exists")
    args = parser.parse_args(argv)
    try:
        result = prepare(args) if args.command == "prepare" else annotate(args)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"Sequence audit refused: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
