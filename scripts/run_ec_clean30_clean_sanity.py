"""Training-side sanity checks of the retrained CLEAN fits (amendment 2, rule A2-S7).

Subcommands (see ``src/ec_clean30/clean_sanity.py`` for the rules and thresholds):
- ``sample --work-dir W --out F``: the hashed in-sample query sample of a completed fit, one
  accession per line (write-once); pass it to the comparator's validation-mode ``export`` with
  ``--allow-in-sample-queries`` (an engineering check, not an evaluation).
- ``in-sample --work-dir W --arm A --sample F --export E --policy-log L [--policy-device D] --out R``: the
  in-sample record of the fit from that export and the raw-embedding anchor over the fit's own linked base
  embeddings; ``L`` is the fit's log holding its runtime precision-policy line (rule A2-GOV-3).
- ``held-out --spec S --out R``: the fold-0 held-out record of the four R1-inner and four R2-inner
  fits (``S`` is a JSON file naming each fit's work directory, validation export and precision-policy log
  (``policy_log``, optional ``policy_device``), the inner exclusions lists, the fold-0 official training file
  and the base farm of the queries).
- ``verify-all --record R ... [--disposition D ...] --out F``: every record and disposition verified together;
  a write-once receipt with each record's outcome (and each held-out arm's), totals only.
- ``tie-exposure --official-training-file P --fold K [--r2-list L] --out R``: structural tie
  exposure of the training tables (EC numbers with identical member sets across EC classes);
  totals only, printed for fold 0 only (private receipt).
- ``disposition --record R --decision D --maintainer M --note N [--arm A] --out F``: the maintainer's
  disposition of a failed record (of one failed arm of the held-out record), after no defect was found.

Every input and output path passes the comparator's held-out path refusal; records are
write-once and hold totals only. Embedding files are read with torch, imported only by the
actions that load them.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ec_clean30 import clean_sanity  # noqa: E402

comparator = clean_sanity.load_comparator()


def _roots() -> list[Path]:
    return comparator.resolve_forbidden_roots()


def _checked(path: Path | str, what: str) -> Path:
    try:
        return comparator.refuse_held_out_path(path, _roots(), what)
    except comparator.ComparatorError as exc:
        raise clean_sanity.SanityError(str(exc)) from None


def read_training_table(path: Path) -> dict[str, tuple[str, ...]]:
    """Accession -> EC numbers of a CLEAN training table written by the comparator's ``prepare``."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        clean_sanity._require(tuple(next(reader, ())) == comparator.TABLE_HEADER,
                              f"{Path(path).name} is not a CLEAN training table")
        table: dict[str, tuple[str, ...]] = {}
        for raw in reader:
            if raw:
                clean_sanity._require(raw[0] not in table, f"{Path(path).name} repeats an entry")
                table[raw[0]] = tuple(item for item in raw[1].split(";") if item)
    return table


def fit_inputs(work_dir: Path | str) -> dict[str, Any]:
    """A completed fit's prepare record, training table (hash-checked) and verified base embedding links."""
    work_dir = _checked(work_dir, "the fit work directory")
    prepare = json.loads((work_dir / comparator.PREPARE_MANIFEST).read_text(encoding="utf-8"))
    table_path = work_dir / prepare["training_table"]["file"]
    clean_sanity._require(comparator.sha256_file(table_path) == prepare["training_table"]["sha256"],
                          "The training table changed after prepare")
    comparator.verify_farm_links(work_dir, prepare["embedding_farm"])
    base = []
    with (work_dir / prepare["embedding_farm"]["links_file"]).open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader)
        base = [label for kind, label, _ in reader if kind == "base"]
    return {"work_dir": work_dir, "prepare": prepare, "table": read_training_table(table_path),
            "table_sha256": prepare["training_table"]["sha256"], "base_labels": tuple(base)}


def load_embedding(path: Path) -> np.ndarray:
    """A ``<label>.pt`` farm file: a fair-esm dict (``mean_representations[33]``) or a plain 1-D tensor."""
    import torch  # only the actions that read embeddings need it

    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(payload, dict):
        payload = payload["mean_representations"][33]
    return np.asarray(payload.detach().cpu().numpy(), dtype=np.float64).ravel()


def linked_embeddings(inputs: Mapping[str, Any], accessions: Sequence[str]) -> dict[str, np.ndarray]:
    folder = inputs["work_dir"] / "data" / "esm_data"
    missing = sorted(set(accessions) - set(inputs["base_labels"]))
    clean_sanity._require(not missing, f"{len(missing)} proteins have no linked base embedding")
    return {accession: load_embedding(folder / f"{accession}.pt") for accession in accessions}


def farm_embeddings(farm_dir: Path | str, accessions: Sequence[str], *, embedding: str,
                    dim: int) -> dict[str, np.ndarray]:
    """Base embeddings from a farm directory, each checked against its farm-manifest row and file hash."""
    farm_dir = _checked(farm_dir, "the query embedding farm")
    entries = comparator.read_farm_manifest(farm_dir / comparator.FARM_MANIFEST)
    loaded = {}
    for accession in accessions:
        entry, path = entries.get(accession), farm_dir / f"{accession}.pt"
        clean_sanity._require(entry is not None and path.is_file() and entry.seed is None
                              and entry.embedding == embedding and entry.dim == dim
                              and comparator.sha256_file(path) == entry.file_sha256,
                              "A query embedding does not match its farm-manifest row")
        loaded[accession] = load_embedding(path)
    return loaded


def _sample_for(inputs: Mapping[str, Any]) -> tuple[str, ...]:
    prepare = inputs["prepare"]
    return clean_sanity.in_sample_sample(inputs["table"], fold=int(prepare["fold"]), regime=prepare["regime"],
                                         inner_fold=prepare.get("inner_fold"))


def action_sample(args: argparse.Namespace) -> dict[str, Any]:
    inputs = fit_inputs(args.work_dir)
    sample = _sample_for(inputs)
    out = _checked(args.out, "the sample list")
    with out.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("".join(f"{accession}\n" for accession in sample))
    return {"sample": str(out), "n": len(sample), "sha256": clean_sanity.sha256_lines(sample)}


def _read_sample(path: Path | str) -> tuple[str, ...]:
    return tuple(line.strip() for line in _checked(path, "the sample list").read_text(encoding="utf-8").splitlines()
                 if line.strip())


def action_in_sample(args: argparse.Namespace) -> dict[str, Any]:
    inputs = fit_inputs(args.work_dir)
    sample = _read_sample(args.sample)
    clean_sanity._require(sample == _sample_for(inputs), "The sample list is not this fit's hashed sample")
    export = comparator.read_distance_export(_checked(args.export, "the in-sample export"))
    fit_manifest = json.loads((inputs["work_dir"] / comparator.FIT_MANIFEST).read_text(encoding="utf-8"))
    clean_sanity._require(export.receipt["checkpoint_sha256"]
                          == (fit_manifest.get("training_receipt") or {}).get("final_weights_sha256")
                          and export.receipt["training_table_sha256"] == inputs["table_sha256"],
                          "The export belongs to another fit")
    members = sorted({accession for ids in clean_sanity.ec_members(inputs["table"]).values() for accession in ids})
    embeddings = linked_embeddings(inputs, members)
    anchor = clean_sanity.raw_anchor_projection(inputs["table"], embeddings, sample, embeddings)
    policy = clean_sanity.read_fit_policy(_checked(args.policy_log, "the fit's precision-policy log"),
                                          device=args.policy_device)
    record = clean_sanity.evaluate_in_sample(arm=args.arm, export=export, training_ec_numbers=inputs["table"],
                                             sample=sample, anchor=anchor, precision_policy=policy)
    digest = clean_sanity.write_once(_checked(args.out, "the sanity record"), record)
    return {"record": str(args.out), "sha256": digest, "status": record["status"], "criteria": record["criteria"]}


def action_held_out(args: argparse.Namespace) -> dict[str, Any]:
    spec = json.loads(_checked(args.spec, "the held-out specification").read_text(encoding="utf-8"))
    rows, _ = comparator.read_official_training_rows(spec["official_training_file"], clean_sanity.HELD_OUT_FOLD,
                                                     _roots())
    true_ec = {row.accession: tuple(row.ec_numbers) for row in rows}
    fits_spec = {(item["arm"], int(item["inner_fold"])): item for item in spec["fits"]}
    loaded = {key: fit_inputs(item["work_dir"]) for key, item in fits_spec.items()}
    queries: dict[int, tuple[str, ...]] = {}
    for inner in range(clean_sanity.N_INNER_FOLDS):
        reasons, _ = comparator.read_inner_exclusions(spec["inner_exclusions"][str(inner)], clean_sanity.HELD_OUT_FOLD,
                                                      inner, _roots())
        validation = sorted(accession for accession, reason in reasons.items()
                            if reason == comparator.EXCLUSION_REASON_VALIDATION)
        queries[inner] = clean_sanity.held_out_queries(validation, true_ec, loaded[("r1_inner", inner)]["table"])
    all_queries = sorted({query for chosen in queries.values() for query in chosen})
    query_embeddings = farm_embeddings(spec["query_farm"], all_queries, embedding=comparator.DEFAULT_EMBEDDING,
                                       dim=comparator.DEFAULT_INPUT_DIM)
    fits = []
    for (arm, inner), item in sorted(fits_spec.items()):
        inputs = loaded[(arm, inner)]
        members = sorted({accession for ids in clean_sanity.ec_members(inputs["table"]).values() for accession in ids})
        anchor = clean_sanity.raw_anchor_projection(inputs["table"], linked_embeddings(inputs, members),
                                                    queries[inner], query_embeddings)
        export = comparator.read_distance_export(_checked(item["export"], "a held-out export"))
        policy = clean_sanity.read_fit_policy(_checked(item["policy_log"], "a held-out fit's precision-policy log"),
                                              device=item.get("policy_device", "cuda"))
        fits.append(clean_sanity.HeldOutFit(arm, export, inputs["table"], anchor, policy))
    record = clean_sanity.evaluate_held_out(fits, queries, true_ec)
    digest = clean_sanity.write_once(_checked(args.out, "the sanity record"), record)
    return {"record": str(args.out), "sha256": digest, "status": record["status"], "criteria": record["criteria"]}


def action_tie_exposure(args: argparse.Namespace) -> dict[str, Any]:
    clean_sanity._require(len(args.official_training_file) == len(args.fold), "Give one --fold per training file")
    r2_lists = list(args.r2_list or [])
    clean_sanity._require(not r2_lists or len(r2_lists) == len(args.fold), "Give one --r2-list per fold or none")
    receipt: dict[str, Any] = {"schema_version": clean_sanity.SCHEMA_VERSION, "kind": "ec_clean30_tie_exposure",
                               "created_utc": clean_sanity.utc_now(), "totals_only": True, "folds": {}}
    for index, (path, fold) in enumerate(zip(args.official_training_file, args.fold)):
        rows, report = comparator.read_official_training_rows(path, int(fold), _roots())
        table = {row.accession: tuple(row.ec_numbers) for row in rows}
        entry = {"official_training_file_sha256": report["sha256"],
                 "r1": clean_sanity.structural_tie_groups(table)}
        if r2_lists:
            allowlist, allow_report = comparator.read_accession_allowlist(r2_lists[index], _roots())
            r2_rows = comparator.select_rows(rows, allowlist)
            entry["r2_list_sha256"] = allow_report["sha256"]
            entry["r2"] = clean_sanity.structural_tie_groups({row.accession: tuple(row.ec_numbers) for row in r2_rows})
        receipt["folds"][str(int(fold))] = entry
    digest = clean_sanity.write_once(_checked(args.out, "the tie-exposure receipt"), receipt)
    # Only fold 0's count may be stated; the other folds stay in the private receipt.
    return {"receipt": str(args.out), "sha256": digest, "fold_0": receipt["folds"].get("0")}


def action_disposition(args: argparse.Namespace) -> dict[str, Any]:
    summary = clean_sanity.verify_record(_checked(args.record, "the sanity record"))
    clean_sanity._require(summary["status"] == clean_sanity.STATUS_DIAGNOSIS_REQUIRED,
                          "A disposition applies to a failed record only")
    payload = clean_sanity.build_disposition(record_sha256=summary["sha256"], decision=args.decision,
                                             maintainer=args.maintainer, note=args.note,
                                             arm=getattr(args, "arm", None))
    clean_sanity.verify_disposition(payload, {summary["sha256"]: summary})
    digest = clean_sanity.write_once(_checked(args.out, "the disposition"), payload)
    return {"disposition": str(args.out), "sha256": digest, "decision": args.decision}


def action_verify_all(args: argparse.Namespace) -> dict[str, Any]:
    receipt = clean_sanity.verify_all([_checked(path, "a sanity record") for path in args.record],
                                      [_checked(path, "a disposition") for path in (args.disposition or [])])
    digest = clean_sanity.write_once(_checked(args.out, "the verify-all receipt"), receipt)
    return {"receipt": str(args.out), "sha256": digest, "n_records": receipt["n_records"],
            "n_dispositions": receipt["n_dispositions"], "stopped": len(receipt["stopped"])}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Training-side sanity checks of the retrained CLEAN fits "
                                                 "(amendment 2, A2-S7).")
    actions = parser.add_subparsers(dest="action", required=True)
    sample = actions.add_parser("sample", help="Write the hashed in-sample query sample of a fit.")
    sample.add_argument("--work-dir", type=Path, required=True)
    sample.add_argument("--out", type=Path, required=True)
    in_sample = actions.add_parser("in-sample", help="Write the in-sample record of a fit.")
    in_sample.add_argument("--work-dir", type=Path, required=True)
    in_sample.add_argument("--arm", choices=clean_sanity.ARMS, required=True)
    in_sample.add_argument("--sample", type=Path, required=True)
    in_sample.add_argument("--export", type=Path, required=True)
    in_sample.add_argument("--policy-log", type=Path, required=True,
                           help="The fit's log holding its runtime precision-policy line.")
    in_sample.add_argument("--policy-device", choices=clean_sanity.POLICY_DEVICES, default="cuda")
    in_sample.add_argument("--out", type=Path, required=True)
    held_out = actions.add_parser("held-out", help="Write the fold-0 held-out record of the inner fits.")
    held_out.add_argument("--spec", type=Path, required=True)
    held_out.add_argument("--out", type=Path, required=True)
    ties = actions.add_parser("tie-exposure", help="Count the structural tie exposure of the training tables.")
    ties.add_argument("--official-training-file", type=Path, action="append", required=True)
    ties.add_argument("--fold", type=int, action="append", required=True)
    ties.add_argument("--r2-list", type=Path, action="append")
    ties.add_argument("--out", type=Path, required=True)
    disposition = actions.add_parser("disposition", help="Record the maintainer's disposition of a failed record.")
    disposition.add_argument("--record", type=Path, required=True)
    disposition.add_argument("--decision", choices=clean_sanity.DECISIONS, required=True)
    disposition.add_argument("--maintainer", required=True)
    disposition.add_argument("--note", required=True)
    disposition.add_argument("--arm", choices=clean_sanity.HELD_OUT_ARMS, default=None,
                             help="The failed arm of the held-out record that this disposition decides.")
    disposition.add_argument("--out", type=Path, required=True)
    verify_all = actions.add_parser("verify-all", help="Verify every record and disposition; write a receipt.")
    verify_all.add_argument("--record", type=Path, action="append", required=True)
    verify_all.add_argument("--disposition", type=Path, action="append")
    verify_all.add_argument("--out", type=Path, required=True)
    return parser


ACTIONS = {"sample": action_sample, "in-sample": action_in_sample, "held-out": action_held_out,
           "tie-exposure": action_tie_exposure, "disposition": action_disposition, "verify-all": action_verify_all}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = ACTIONS[args.action](args)
    except (clean_sanity.SanityError, comparator.ComparatorError, FileExistsError) as exc:
        raise SystemExit(f"error: {exc}") from None
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
