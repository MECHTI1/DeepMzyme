"""Amendment 2, family F6: the per-fold whole-protein pretraining corpus inside a view (synthetic data only).

The real-build tests reuse the synthetic five-fold world of ``test_ec_clean30_overlay`` (data root, cohort
audit, training-side identity chain with a fake pinned search, fold 0's R1-inner lists); the smoke-build
tests use a synthetic smoke view. No campaign file and no data folder is read.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import types

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT / "tests")]

from ec_clean30 import identity, pretrain_corpus as pc, r2_lists  # noqa: E402
from ec_clean30.campaign import CAMPAIGN_ID  # noqa: E402
from ec_clean30.provenance import HeldOutInputError  # noqa: E402
from ec_clean30.views import SMOKE_VIEWS_DIR, View  # noqa: E402
from ec_clean30_fixtures import (  # noqa: E402
    AMBIGUOUS_ACCESSION, build_smoke_view, campaign_proteins, make_writable, unlock, write_clean30_data_root,
)
from training.global_esm_features import (  # noqa: E402
    KIND_CHAIN_MEAN, KIND_SEQUENCE_MEAN, build_manifest, chain_mean_payload, encode_cache_file, load_global_esm_vectors,
    read_manifest, sequence_mean_payload, write_bytes_atomic, write_manifest,
)
from training.global_esm_pretraining import CORPUS_COLUMNS, parse_corpus  # noqa: E402

DIM = 960
PROVENANCE = {"note": "synthetic test provenance", "smoke": False}
SMOKE_PROVENANCE = {"note": "synthetic test provenance", "smoke": True}


def _key(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


STUB_HELPER_SHA256 = "5" * 64


def stub_policy(device: str = "cuda", **changes) -> dict:
    """A precision-policy record of the shape ``runtime_precision.configure_and_record`` writes (synthetic)."""
    record = {"device": device, "nvidia_tf32_override": "0", "cuda_matmul_allow_tf32": False,
              "cudnn_allow_tf32": False, "torch_version": "2.x", "cuda_device_count": 1 if device == "cuda" else 0,
              "cuda_device_name": "synthetic GPU" if device == "cuda" else None, "helper_sha256": STUB_HELPER_SHA256}
    record.update(changes)
    return record


def runtime_precision_stub() -> types.ModuleType:
    """Stand-in for ``ec_clean30.runtime_precision`` (execution lane): the same acceptance rule on a stub hash."""
    module = types.ModuleType("ec_clean30.runtime_precision")

    def validate_policy(record, *, device):
        valid = (isinstance(record, dict) and record.get("device") == device
                 and record.get("nvidia_tf32_override") == "0" and record.get("cuda_matmul_allow_tf32") is False
                 and record.get("cudnn_allow_tf32") is False and record.get("helper_sha256") == STUB_HELPER_SHA256
                 and record.get("cuda_device_count") == (1 if device == "cuda" else 0))
        if not valid:
            raise RuntimeError("Child precision policy does not prove the requested device and TF32-off state.")
        return dict(record)

    module.validate_policy = validate_policy
    return module


@pytest.fixture(autouse=True)
def runtime_precision(monkeypatch):
    stub = runtime_precision_stub()
    monkeypatch.setitem(sys.modules, "ec_clean30.runtime_precision", stub)
    return stub


def fp32_runtime(device: str = "cuda:0", **policy_changes) -> dict:
    return {"device": device, "forward_precision": "fp32", "forward_call": "ESMC.forward",
            "precision": {"parameter_dtypes": ["float32"],
                          "precision_policy": stub_policy("cuda" if device.startswith("cuda") else "cpu",
                                                          **policy_changes)}}


def write_parent_cache(directory: Path, sequences: dict[str, str], *, seed: int = 0,
                       overrides: dict[str, torch.Tensor] | None = None, skip: set[str] = frozenset(),
                       runtime: dict | None = None) -> str:
    """A union sequence-mean cache like the ``sequences`` build: final + block26 layers, labels and inputs
    (``runtime``: the build's runtime record, by default an fp32 GPU build with a valid precision policy)."""
    generator = torch.Generator().manual_seed(seed)
    directory.mkdir(parents=True, exist_ok=True)
    files, labels = [], []
    for accession, sequence in sorted(sequences.items()):
        labels.append({"input": "base.fasta", "label": accession, "sequence_sha256": _key(sequence)})
    for key in sorted({_key(sequence) for sequence in sequences.values()} - set(skip)):
        for layer in ("final", "block26"):
            vector = (overrides or {}).get(key) if layer == "final" else None
            vector = torch.randn(DIM, generator=generator) if vector is None else vector
            payload = sequence_mean_payload(vector.float(), sequence_sha256=key, layer=layer, n_tokens=30,
                                            n_mask_tokens=0, n_tokens_input=30, source={"kind": "synthetic"})
            item = encode_cache_file(payload)
            write_bytes_atomic(directory / item.file_name, item.data)
            files.append(item)
    manifest = build_manifest(files, kind=KIND_SEQUENCE_MEAN, allowlist={"file_sha256": "e" * 64, "n_ids": 99},
                              build={"action": "sequences", "runtime": fp32_runtime() if runtime is None else runtime},
                              extra={"inputs": [{"file": "base.fasta"}], "labels": labels})
    return write_manifest(directory, manifest)[1]


def write_chain_means(directory: Path, entries: dict[str, tuple[torch.Tensor, str]]) -> str:
    """A final-layer chain-mean cache: structure ID -> (mean, source_sequence_sha256)."""
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    for structure_id, (mean, source_sha) in sorted(entries.items()):
        payload = chain_mean_payload(mean.float(), structure_id=structure_id, layer="final", n_residues=30,
                                     source={"kind": "synthetic"}, source_sequence_sha256=source_sha)
        item = encode_cache_file(payload)
        write_bytes_atomic(directory / item.file_name, item.data)
        files.append(item)
    manifest = build_manifest(files, kind=KIND_CHAIN_MEAN, allowlist={"file_sha256": "d" * 64}, build={"action": "t"})
    return write_manifest(directory, manifest)[1]


# ---------------------------------------------------------------------------
# Pure rules


def _sequences(*accessions: str, seed: int = 1) -> dict[str, str]:
    rng = random.Random(seed)
    return {accession: "".join(rng.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(40)) for accession in accessions}


def _toy():
    """Eight metallo proteins in four inner folds, a D3 neighbour, an identical sequence and a case alias."""
    metallo = {f"M000{index}": index % 4 for index in range(8)}
    sequences = _sequences(*metallo, "N0001", "N0002", "N0003", "m0001", "N0004")
    sequences["N0002"] = sequences["M0000"]  # identical to a validation protein of inner fold 0
    labels = {accession: "7" if accession == "N0004" else "2" for accession in sequences}
    validation = {inner: {acc for acc, fold in metallo.items() if fold == inner} for inner in range(4)}
    training = {inner: set(metallo) - validation[inner] for inner in range(4)}
    r1 = {inner: {acc: identity.R1_REASON_VALIDATION for acc in validation[inner]} for inner in range(4)}
    r1[0]["N0001"] = identity.R1_REASON_D3  # one-hop D3 neighbour of an inner-fold-0 validation protein
    return metallo, sequences, labels, validation, training, r1


def test_flags_are_the_r1_list_plus_identical_sequences_plus_case_aliases():
    metallo, sequences, labels, validation, training, r1 = _toy()
    build = pc.build_corpus_rows(labels=labels, sequences=sequences, validation_by_inner=validation,
                                 training_by_inner=training, r1_lists=r1)
    rows = {row.accession: row for row in build.rows}
    assert [row.accession for row in build.rows] == sorted(labels)
    flagged = {inner: {acc for acc, row in rows.items() if row.exclude[inner]} for inner in range(4)}
    assert flagged[0] == set(r1[0]) | {"N0002"}  # validation, the D3 neighbour N0001 and M0000's identical sequence
    assert flagged[1] == validation[1] | {"m0001"}  # the case alias of M0001
    assert flagged[2] == validation[2] and flagged[3] == validation[3]
    assert build.extra_exclusions == {"0": ["N0002"], "1": ["m0001"], "2": [], "3": []}
    assert rows["N0004"].ec_class == "7" and not any(rows["N0004"].exclude)
    assert all(row.sequence_sha256 == _key(sequences[row.accession]) for row in build.rows)
    assert build.counts["identical_sequence_groups"] == 1
    assert build.counts["per_inner_fold"]["0"]["extra_rule_c"] == 1
    assert build.counts["per_inner_fold"]["0"]["r1_inner_listed"] == 3


def test_an_identity_link_into_inner_training_and_a_foreign_r1_list_refuse():
    metallo, sequences, labels, validation, training, r1 = _toy()
    linked = dict(sequences, M0001=sequences["M0000"])  # inner-training protein of fold 0 equal to a validation one
    with pytest.raises(pc.PretrainCorpusError, match="inner-training protein"):
        pc.build_corpus_rows(labels=labels, sequences=linked, validation_by_inner=validation,
                             training_by_inner=training, r1_lists=r1)
    foreign = {**r1, 2: {**r1[2], "M0001": identity.R1_REASON_VALIDATION}}
    with pytest.raises(pc.PretrainCorpusError, match="other inner-validation proteins"):
        pc.build_corpus_rows(labels=labels, sequences=sequences, validation_by_inner=validation,
                             training_by_inner=training, r1_lists=foreign)
    smoke = {row.accession: row for row in pc.build_corpus_rows(
        labels=labels, sequences=linked, validation_by_inner=validation, training_by_inner=training,
        r1_lists=None).rows}
    assert not smoke["N0001"].exclude[0]  # the smoke rule has no D3 neighbours
    assert smoke["N0002"].exclude[0] and not smoke["M0001"].exclude[0]  # (c) only outside the smoke membership


def test_official_ec_classes_exclude_several_classes_keep_class_7_and_refuse_untrusted(tmp_path):
    path = tmp_path / "split30_train_split_0.csv"
    rows = [("A0001", "1.1.1.1"), ("A0002", "2.7.1.1;3.1.1.1"), ("A0003", "7.1.2.2"), ("A0004", "3.4.21.4;3.4.21.5")]
    path.write_text("Entry\tEC number\tSequence\n" + "".join(f"{a}\t{e}\tACDE\n" for a, e in rows), encoding="utf-8")
    labels, several = pc.classify_official_rows(pc.read_official_ec_numbers(path))
    assert labels == {"A0001": "1", "A0003": "7", "A0004": "3"} and several == {"A0002"}
    for value in ("", "9.1.1.1", "x.1.1.1", "1.1.1.1;-"):
        with pytest.raises(pc.PretrainCorpusError, match="Untrusted EC annotations"):
            pc.classify_official_rows({"A0001": value})
    path.write_text("Entry\tEC number\tSequence\nA0001\t1.1.1.1\tACDE\nA0001\t2.1.1.1\tACDE\n", encoding="utf-8")
    with pytest.raises(pc.PretrainCorpusError, match="another EC number string"):
        pc.read_official_ec_numbers(path)


def test_parity_record_applies_the_c9_limits():
    base = torch.randn(DIM, generator=torch.Generator().manual_seed(3))
    passed = pc.parity_record({"a": (base, base.clone()), "b": (base, base + 5e-5)}, differs=["c"], not_in_corpus=["d"])
    assert passed["passed"] and passed["n_compared"] == 2 and passed["n_structure_sequence_differs"] == 1
    assert passed["structure_sequence_differs"] == ["c"] and passed["n_not_in_corpus"] == 1
    shifted = base.clone()
    shifted[0] += 2e-4  # cosine stays above 0.9999 but the maximum difference breaks 1e-4
    assert pc.vector_parity(base, shifted)[0] > pc.PARITY_MIN_COSINE
    failed = pc.parity_record({"a": (base, shifted)})
    assert not failed["passed"] and failed["failed"] == ["a"]
    assert not pc.parity_passes(0.99989, 0.0) and pc.parity_passes(0.9999, 1e-4)


def test_the_gpu_cpu_sample_is_the_first_64_keys_plus_the_8_longest():
    rng = random.Random(5)
    sequences = {f"S{index:04d}": "A" * rng.randrange(10, 400) + str(index) for index in range(300)}
    sequences = {name: sequence.rstrip("0123456789") + "C" * (index % 7) for index, (name, sequence)
                 in enumerate(sequences.items())}
    keys = pc.parity_sample_keys(sequences)
    lengths = {_key(sequence): len(sequence) for sequence in sequences.values()}
    first = sorted(lengths)[:64]
    longest = sorted(lengths, key=lambda key: (-lengths[key], key))[:8]
    assert keys == sorted(set(first) | set(longest)) and 64 <= len(keys) <= 72
    assert pc.parity_sample_keys(dict(reversed(list(sequences.items())))) == keys


def test_the_gpu_cpu_sample_comparison_applies_the_same_limits(tmp_path):
    sequences = _sequences("A0001", "A0002", "A0003")
    keys = sorted(_key(sequence) for sequence in sequences.values())
    base = {key: torch.randn(DIM, generator=torch.Generator().manual_seed(index)) for index, key in enumerate(keys)}
    cpu_sha = write_parent_cache(tmp_path / "cpu", sequences, overrides=base)
    close_sha = write_parent_cache(tmp_path / "gpu", sequences, overrides={key: value + 2e-5 for key, value in base.items()})
    far = {**base, keys[1]: base[keys[1]] + 3e-4}
    far_sha = write_parent_cache(tmp_path / "gpu_bf16", sequences, overrides=far)
    passed = pc.compare_sequence_mean_samples(tmp_path / "cpu", cpu_sha, tmp_path / "gpu", close_sha, keys)
    assert passed["passed"] and passed["n_compared"] == 3
    failed = pc.compare_sequence_mean_samples(tmp_path / "cpu", cpu_sha, tmp_path / "gpu_bf16", far_sha, keys)
    assert not failed["passed"] and failed["failed"] == [keys[1]]


# ---------------------------------------------------------------------------
# Real builds on the synthetic five-fold world


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    from test_ec_clean30_overlay import build_world

    return build_world(tmp_path_factory.mktemp("pcworld"))


@pytest.fixture(autouse=True)
def pinned_binary(request, monkeypatch):
    if "world" in request.fixturenames:
        monkeypatch.setattr(identity, "MMSEQS_BINARY_SHA256", request.getfixturevalue("world").binary_sha)


def _split30(world) -> Path:
    return world.base / "source" / "split30"


def _official_sequences(world, fold: int) -> dict[str, str]:
    return identity.read_official_sequences(_split30(world) / f"split30_train_split_{fold}.csv")[0]


def _real_inputs(world, tmp_path: Path, *, fold: int = 0, perturb: bool = False, skip: set[str] = frozenset(),
                 runtime: dict | None = None):
    sequences = _official_sequences(world, fold)
    metallo = sorted(set(world.training(fold)))[:3]
    seq_vectors = {_key(sequences[acc]): torch.randn(DIM, generator=torch.Generator().manual_seed(index))
                   for index, acc in enumerate(metallo)}
    parent = tmp_path / "parent_seq_means"
    parent_sha = write_parent_cache(parent, sequences, overrides=seq_vectors, skip=skip, runtime=runtime)
    entries = {}
    for index, acc in enumerate(metallo):
        key = _key(sequences[acc])
        mean = seq_vectors[key] + (1e-3 if perturb and index == 0 else 1e-6)
        entries[f"{acc}__chain_A__EC_1.1.1.1"] = (mean, key)
    entries[f"{metallo[0]}__chain_B__EC_1.1.1.1"] = (torch.randn(DIM), "f" * 64)  # a fragment: other sequence
    chain_sha = write_chain_means(tmp_path / "chain_means_final", entries)
    membership_sha = r2_lists.load_fold_inputs(world.work(fold), fold).membership.file_sha256
    return dict(fold=fold, smoke=False, official_split_dir=_split30(world), seq_cache_dir=parent,
                seq_manifest_sha256=parent_sha, chain_means=(tmp_path / "chain_means_final", chain_sha),
                view_record={"root": "synthetic", "manifest_sha256": "c" * 64, "fold": fold},
                provenance=PROVENANCE, identity_work=world.work(fold), expected_membership_sha256=membership_sha)


def test_a_real_build_writes_the_corpus_a_fresh_manifest_and_a_receipt(world, tmp_path):
    options = _real_inputs(world, tmp_path)
    out = tmp_path / "view" / pc.WHOLE_PROTEIN_PRETRAIN_DIR
    summary = pc.build_corpus_folder(out, **options)
    corpus_bytes = (out / pc.CORPUS_NAME).read_bytes()
    rows = parse_corpus(corpus_bytes, origin="corpus.tsv")
    sequences = _official_sequences(world, 0)
    assert [row.accession for row in rows] == sorted(sequences)
    assert all(row.sequence_sha256 == _key(sequences[row.accession]) for row in rows)
    assert {row.ec_class for row in rows} == {"1", "4"}  # metallo rows "1.1.1.1", the other enzymes "4.2.1.1"
    inputs = r2_lists.load_fold_inputs(world.work(0), 0)
    for inner in range(4):
        r1, _record = r2_lists.load_r1_inner_exclusions(inputs, inner)
        assert {row.accession for row in rows if row.exclude[inner]} == set(r1)
        assert not any(row.exclude[inner] for row in rows
                       if inputs.inner_fold_of.get(row.accession, inner) != inner)
    manifest_path = out / pc.SEQ_MEANS_DIR / pc.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    keys = sorted({row.sequence_sha256 for row in rows})
    assert "labels" not in manifest and "inputs" not in manifest and manifest["layers"] == ["final"]
    assert manifest["allowlist"]["source"] == pc.CORPUS_NAME and manifest["allowlist"]["n_ids"] == len(keys)
    assert manifest["build"]["action"] == pc.SUBSET_BUILD_ACTION
    assert manifest["build"]["parent_manifest_sha256"] == options["seq_manifest_sha256"]
    parent = read_manifest(options["seq_cache_dir"], options["seq_manifest_sha256"])
    by_key = {entry["sequence_sha256"]: entry for entry in parent["files"] if entry["layer"] == "final"}
    for entry in manifest["files"]:
        copied = (out / pc.SEQ_MEANS_DIR / entry["file"]).read_bytes()
        assert copied == (options["seq_cache_dir"] / by_key[entry["sequence_sha256"]]["file"]).read_bytes()
    receipt = json.loads((out / pc.CORPUS_RECEIPT_NAME).read_text())
    assert receipt["step"] == pc.STEP and receipt["fold"] == 0 and receipt["smoke"] is False
    assert receipt["outputs"][pc.CORPUS_NAME]["sha256"] == summary["corpus_sha256"] == hashlib.sha256(
        corpus_bytes).hexdigest()
    assert receipt["parity"]["passed"] and receipt["parity"]["n_compared"] == 3
    assert receipt["parity"]["n_structure_sequence_differs"] == 1
    assert receipt["inputs"]["membership"]["sha256"] == options["expected_membership_sha256"]
    assert summary["counts"]["corpus_rows"] == len(rows) and summary["parity_counts"]["n_compared"] == 3
    # Same inputs, same bytes.
    again = tmp_path / "again" / pc.WHOLE_PROTEIN_PRETRAIN_DIR
    pc.build_corpus_folder(again, **options)
    for name in (pc.CORPUS_NAME, pc.CORPUS_RECEIPT_NAME, f"{pc.SEQ_MEANS_DIR}/{pc.MANIFEST_NAME}"):
        assert (again / name).read_bytes() == (out / name).read_bytes()
    with pytest.raises(pc.PretrainCorpusError, match="not an empty folder"):
        pc.build_corpus_folder(out, **options)


def test_real_build_refusals_leave_nothing_behind(world, tmp_path):
    out = tmp_path / "view" / pc.WHOLE_PROTEIN_PRETRAIN_DIR
    with pytest.raises(pc.PretrainCorpusError, match="Parity"):
        pc.build_corpus_folder(out, **_real_inputs(world, tmp_path / "perturbed", perturb=True))
    assert not out.exists()
    sequences = _official_sequences(world, 0)
    other = sorted(set(sequences) - set(world.training(0)))[0]  # one of the other enzymes
    with pytest.raises(pc.PretrainCorpusError, match="no final-layer vector"):
        pc.build_corpus_folder(out, **_real_inputs(world, tmp_path / "missing", skip={_key(sequences[other])}))
    options = _real_inputs(world, tmp_path / "plain")
    with pytest.raises(pc.PretrainCorpusError, match="certified membership"):
        pc.build_corpus_folder(out, **{**options, "expected_membership_sha256": "0" * 64})
    with pytest.raises(pc.PretrainCorpusError, match="identity folder"):
        pc.build_corpus_folder(out, **{**options, "identity_work": None})
    named = tmp_path / "test"
    shutil.copytree(options["seq_cache_dir"], named)
    with pytest.raises(HeldOutInputError):
        pc.build_corpus_folder(out, **{**options, "seq_cache_dir": named})
    assert not out.exists()


def test_real_attach_requires_an_fp32_parent_with_a_validated_precision_policy(world, tmp_path, monkeypatch):
    out = tmp_path / "view" / pc.WHOLE_PROTEIN_PRETRAIN_DIR
    sdk = {"device": "cuda:0"}  # the SDK path records no forward precision
    with pytest.raises(pc.PretrainCorpusError, match="forward precision None, not 'fp32'"):
        pc.build_corpus_folder(out, **_real_inputs(world, tmp_path / "sdk", runtime=sdk))
    with pytest.raises(pc.PretrainCorpusError, match="does not validate"):
        pc.build_corpus_folder(out, **_real_inputs(world, tmp_path / "tf32", runtime=fp32_runtime(
            cudnn_allow_tf32=True)))
    with pytest.raises(pc.PretrainCorpusError, match="does not validate"):  # a CPU-device policy on a GPU build
        pc.build_corpus_folder(out, **_real_inputs(world, tmp_path / "device", runtime={
            **fp32_runtime(), "precision": {"parameter_dtypes": ["float32"], "precision_policy": stub_policy("cpu")}}))
    local = {"device": "cpu", "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False,
             "recorded_by": "esmc_chain_means (ec_clean30.runtime_precision not available)"}
    with pytest.raises(pc.PretrainCorpusError, match="does not validate"):  # the local CPU fallback record
        pc.build_corpus_folder(out, **_real_inputs(world, tmp_path / "local", runtime={
            **fp32_runtime("cpu"), "precision": {"parameter_dtypes": ["float32"], "precision_policy": local}}))
    options = _real_inputs(world, tmp_path / "ok")
    monkeypatch.setitem(sys.modules, "ec_clean30.runtime_precision", None)
    with pytest.raises(pc.PretrainCorpusError, match="runtime_precision is not available"):
        pc.build_corpus_folder(out, **options)
    assert not out.exists()
    monkeypatch.setitem(sys.modules, "ec_clean30.runtime_precision", runtime_precision_stub())
    pc.build_corpus_folder(out, **options)
    receipt = json.loads((out / pc.CORPUS_RECEIPT_NAME).read_text())
    parent = receipt["inputs"]["parent_sequence_means"]
    assert parent["forward_precision"] == "fp32" and parent["precision_policy"]["passed"] is True
    assert parent["precision_policy"]["precision_policy"]["cudnn_allow_tf32"] is False
    # The CPU build of check (i) is accepted with a CPU policy.
    cpu_out = tmp_path / "cpu_view" / pc.WHOLE_PROTEIN_PRETRAIN_DIR
    pc.build_corpus_folder(cpu_out, **_real_inputs(world, tmp_path / "cpu", runtime=fp32_runtime("cpu")))
    assert json.loads((cpu_out / pc.CORPUS_RECEIPT_NAME).read_text())["inputs"]["parent_sequence_means"][
        "precision_policy"]["device_type"] == "cpu"


def test_a_smoke_attach_records_a_failed_precision_check_without_refusing():
    record = pc.parent_precision_record({"build": {"runtime": {"device": "cpu"}}}, smoke=True)
    assert record["passed"] is False and "not 'fp32'" in record["problem"]
    with pytest.raises(pc.PretrainCorpusError, match="precision rule"):
        pc.parent_precision_record({"build": {"runtime": {"device": "cpu"}}}, smoke=False)


def test_the_cache_free_corpus_table_equals_the_attached_corpus_and_binds_it(world, tmp_path):
    campaign = tmp_path / "campaign"
    summary = pc.build_corpus_table(campaign, 0, official_split_dir=_split30(world), identity_work=world.work(0),
                                    provenance=PROVENANCE)
    table = pc.f6_corpus_dir(campaign, 0) / pc.CORPUS_NAME
    receipt = json.loads((pc.f6_corpus_dir(campaign, 0) / pc.CORPUS_TABLE_RECEIPT_NAME).read_text())
    assert receipt["cache_read"] is False and receipt["outputs"][pc.CORPUS_NAME]["sha256"] == summary["corpus_sha256"]
    assert summary["counts"]["corpus_rows"] == len(parse_corpus(table.read_bytes(), origin="t"))
    with pytest.raises(pc.PretrainCorpusError, match="built once"):
        pc.build_corpus_table(campaign, 0, official_split_dir=_split30(world), identity_work=world.work(0),
                              provenance=PROVENANCE)
    options = _real_inputs(world, tmp_path / "attach")
    out = tmp_path / "view" / pc.WHOLE_PROTEIN_PRETRAIN_DIR
    attached = pc.build_corpus_folder(out, **options, bound_corpus=(table, summary["corpus_sha256"]))
    assert attached["corpus_sha256"] == summary["corpus_sha256"]
    assert (out / pc.CORPUS_NAME).read_bytes() == table.read_bytes()
    assert json.loads((out / pc.CORPUS_RECEIPT_NAME).read_text())["inputs"]["bound_corpus_table"]["sha256"] == \
        summary["corpus_sha256"]
    other = tmp_path / "other_view" / pc.WHOLE_PROTEIN_PRETRAIN_DIR
    with pytest.raises(pc.PretrainCorpusError, match="differs from the bound corpus table"):
        pc.build_corpus_folder(other, **_real_inputs(world, tmp_path / "attach2"), bound_corpus=(table, "0" * 64))
    assert not other.exists()


def test_the_union_fasta_holds_every_training_accession_once(world, tmp_path):
    campaign = tmp_path / "campaign"
    summary = pc.build_union_fasta(campaign, official_split_dir=_split30(world), provenance=PROVENANCE)
    text = (pc.f6_corpus_dir(campaign) / pc.UNION_FASTA_NAME).read_text()
    union = {}
    for fold in range(5):
        union.update(_official_sequences(world, fold))
    labels = [line[1:] for line in text.splitlines() if line.startswith(">")]
    assert labels == sorted(union) and summary["n_records"] == len(union)
    assert all(f">{accession}\n{union[accession]}\n" in text for accession in union)
    with pytest.raises(pc.PretrainCorpusError, match="written once"):
        pc.build_union_fasta(campaign, official_split_dir=_split30(world), provenance=PROVENANCE)


def test_a_fold_without_its_r1_inner_lists_refuses_and_other_folds_print_no_counts(world, tmp_path):
    with pytest.raises(pc.PretrainCorpusError, match="r1-inner-exclusions"):
        pc.build_corpus_folder(tmp_path / "f1" / pc.WHOLE_PROTEIN_PRETRAIN_DIR, **_real_inputs(world, tmp_path / "a",
                                                                                              fold=1))
    for inner in range(identity.N_INNER_FOLDS):
        identity.build_r1_inner_exclusions(world.work(1), 1, inner)
    summary = pc.build_corpus_folder(tmp_path / "f1" / pc.WHOLE_PROTEIN_PRETRAIN_DIR,
                                     **_real_inputs(world, tmp_path / "b", fold=1))
    assert summary["counts_in_receipt"] is True and "counts" not in summary and "parity_counts" not in summary
    assert summary["fold"] == 1 and summary["parity"] == "passed"


# ---------------------------------------------------------------------------
# Smoke view build, the view accessor and the folder verifier


def build_smoke_view_with_corpus(base: Path) -> tuple[View, dict]:
    """A synthetic smoke view of fold 0 with ``whole_protein_pretrain/`` built by the smoke rule.

    The official file holds the view's proteins (the first with its structure sequence, so parity
    compares it; the synthetic structures share one sequence, so the others get their own), the
    several-class protein, other enzymes (class 7 among them), an identical sequence and a case
    alias of the first view protein.
    """
    from embed_helpers.esmc_chain_means import load_chain_input

    source = write_clean30_data_root(base / "source")
    view_root = base / CAMPAIGN_ID / SMOKE_VIEWS_DIR / "f6view"
    view = build_smoke_view(source, view_root, provenance=SMOKE_PROVENANCE)
    chain_dir, chain_sha = view.chain_means_dir("final"), view.chain_means_manifest_sha256("final")
    chain_ids = [entry["structure_id"] for entry in read_manifest(chain_dir, chain_sha)["files"]]
    chain_vectors = load_global_esm_vectors(chain_dir, chain_sha, chain_ids)
    sequences, classes = {}, {}
    proteins = campaign_proteins()
    rng = random.Random(9)
    for structure_id in chain_ids:
        accession = structure_id.split("__")[0]
        sequences[accession] = "".join(rng.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(55))
        classes[accession] = proteins[accession]["ec"]
    first = sorted(sequences)[0]
    first_id = next(structure_id for structure_id in chain_ids if structure_id.startswith(f"{first}__"))
    sequences[first] = load_chain_input(first_id, view.structures / f"{first_id}.pdb").sequence
    extra = {"Q0001": "7.1.1.1", "Q0002": "4.2.1.1", "Q0003": "3.1.1.1"}
    for accession in extra:
        sequences[accession] = "".join(rng.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(50))
    sequences["Q0004"], extra["Q0004"] = sequences[first], "2.1.1.1"
    sequences[first.lower()] = "".join(rng.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(45))
    extra[first.lower()] = "5.1.1.1"
    sequences[AMBIGUOUS_ACCESSION] = "".join(rng.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(60))
    extra[AMBIGUOUS_ACCESSION] = "2.5.1.3;4.1.99.17"
    classes.update(extra)
    split30 = base / "official" / "split30"
    split30.mkdir(parents=True)
    with (split30 / "split30_train_split_0.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["Entry", "EC number", "Sequence"])
        writer.writerows((accession, classes[accession], sequences[accession]) for accession in sorted(sequences))
    overrides = {_key(sequences[first]): chain_vectors[first_id]}
    parent = base / "parent_seq_means"
    parent_sha = write_parent_cache(parent, sequences, overrides=overrides)
    os.chmod(view.root, 0o755)
    summary = pc.build_view_pretrain_corpus(view.root, official_split_dir=split30, seq_cache_dir=parent,
                                            seq_manifest_sha256=parent_sha, provenance=SMOKE_PROVENANCE, smoke=True)
    return View.load(view.root), {"summary": summary, "source": source, "sequences": sequences, "first": first,
                                  "split30": split30, "parent": parent, "parent_sha": parent_sha}


@pytest.fixture(scope="module")
def smoke_corpus(tmp_path_factory):
    view, info = build_smoke_view_with_corpus(tmp_path_factory.mktemp("pcsmoke"))
    yield view, info
    unlock(info["source"])


def test_a_smoke_build_follows_the_smoke_membership(smoke_corpus):
    from training.ec_membership import read_ec_fold_membership

    view, info = smoke_corpus
    inputs = pc.view_pretrain_inputs(view)
    rows = {row.accession: row for row in parse_corpus(inputs.corpus.read_bytes(), origin="corpus.tsv")}
    assert AMBIGUOUS_ACCESSION not in rows and rows["Q0001"].ec_class == "7"
    membership = read_ec_fold_membership(view.membership_file(), view.membership["sha256"])
    fold_of = {row["accession"]: int(row["fold"]) for row in membership.rows}
    first = info["first"]
    for accession, row in rows.items():
        expected = [fold_of.get(accession) == inner for inner in range(4)]
        if accession in ("Q0004", first.lower()):  # identical sequence and case alias of a view protein
            expected = [fold_of[first] == inner for inner in range(4)]
        assert list(row.exclude) == expected, accession
    receipt = json.loads((inputs.root / pc.CORPUS_RECEIPT_NAME).read_text())
    assert receipt["smoke"] is True and receipt["rules"]["exclusion"] == pc.SMOKE_EXCLUSION_RULE
    assert receipt["counts"]["several_ec_classes_excluded"] == 1
    assert receipt["parity"]["passed"] and receipt["parity"]["n_compared"] == 1
    assert receipt["parity"]["n_structure_sequence_differs"] >= len(fold_of) - 1
    assert sorted(receipt["extra_rule_c_exclusions"][str(fold_of[first])]) == sorted(["Q0004", first.lower()])
    assert info["summary"]["counts"]["several_ec_classes_excluded"] == 1  # fold 0 shows its counts


def test_the_view_accessor_and_the_folder_verifier(smoke_corpus, tmp_path):
    view, _info = smoke_corpus
    record = pc.verify_whole_protein_pretrain(view, required=True)
    inputs = pc.view_pretrain_inputs(view)
    assert record["corpus_sha256"] == inputs.corpus_sha256 and record["manifest_sha256"] == inputs.manifest_sha256
    copy_root = tmp_path / CAMPAIGN_ID / SMOKE_VIEWS_DIR / "copy"
    shutil.copytree(view.root, copy_root)
    make_writable(copy_root)
    copy = View.load(copy_root)
    corpus = copy_root / pc.WHOLE_PROTEIN_PRETRAIN_DIR / pc.CORPUS_NAME
    corpus.write_text(corpus.read_text() + "Z9999\t" + "0" * 64 + "\t1\t0\t0\t0\t0\n")
    with pytest.raises(pc.PretrainCorpusError, match="changed after"):
        pc.view_pretrain_inputs(copy)
    shutil.rmtree(copy_root / pc.WHOLE_PROTEIN_PRETRAIN_DIR)
    assert pc.verify_whole_protein_pretrain(copy) is None
    with pytest.raises(pc.PretrainCorpusError, match="required"):
        pc.verify_whole_protein_pretrain(copy, required=True)
    with pytest.raises(pc.PretrainCorpusError, match="not built"):
        pc.view_pretrain_inputs(copy)


def test_the_runner_action_refuses_test_named_inputs_and_reports_refusals(smoke_corpus, tmp_path, capsys):
    import argparse

    view, info = smoke_corpus
    parser = pc.add_pretrain_corpus_arguments(argparse.ArgumentParser())
    args = parser.parse_args(["attach", "--view", str(view.root), "--official-split-dir", str(tmp_path / "test"),
                              "--seq-cache", str(info["parent"]), "--seq-manifest-sha256", info["parent_sha"],
                              "--smoke"])
    with pytest.raises(HeldOutInputError):
        pc.action_pretrain_corpus(args)
    args = parser.parse_args(["attach", "--view", str(view.root), "--official-split-dir", str(info["split30"]),
                              "--seq-cache", str(info["parent"]), "--seq-manifest-sha256", info["parent_sha"],
                              "--smoke"])
    assert pc.action_pretrain_corpus(args) == 2  # the folder exists already: built once
    assert "not an empty folder" in capsys.readouterr().err
    # ``verify`` is the required check before a fold's first F6 unit (review F-M3).
    assert pc.action_pretrain_corpus(parser.parse_args(["verify", "--view", str(view.root)])) == 0
    assert '"status": "verified"' in capsys.readouterr().out
    with pytest.raises(SystemExit):
        parser.parse_args(["--view", str(view.root)])  # a sub-action is required
    assert CORPUS_COLUMNS[-1] == "exclude_inner3"


# ---------------------------------------------------------------------------
# Amendment-2 review fixes: donor rule (d), hold-out lists in a scratch copy, the attached folder in verify_view


def test_rule_d_flags_a_shared_selected_donor():
    metallo, sequences, labels, validation, training, r1 = _toy()
    donors = {"M0000": "1ABC", "N0003": "1ABC", "M0002": "2XYZ"}
    build = pc.build_corpus_rows(labels=labels, sequences=sequences, validation_by_inner=validation,
                                 training_by_inner=training, r1_lists=r1, donors=donors)
    rows = {row.accession: row for row in build.rows}
    inner = metallo["M0000"]
    assert rows["N0003"].exclude[inner] and build.counts["per_inner_fold"][str(inner)]["extra_rule_d"] == 1
    assert "N0003" in build.extra_exclusions[str(inner)]
    without = {row.accession: row for row in pc.build_corpus_rows(
        labels=labels, sequences=sequences, validation_by_inner=validation, training_by_inner=training,
        r1_lists=r1).rows}
    assert not without["N0003"].exclude[inner]
    # A donor link into an inner-training protein refuses the build (it would cut across inner folds).
    trainer = next(acc for acc in training[inner] if acc in labels)
    with pytest.raises(pc.PretrainCorpusError, match="inner-training protein"):
        pc.build_corpus_rows(labels=labels, sequences=sequences, validation_by_inner=validation,
                             training_by_inner=training, r1_lists=r1, donors={**donors, trainer: "1ABC"})


def _fake_overlay(world, base, fold):
    """An overlay folder holding a byte copy of the fold's identity files and a manifest that lists them."""
    source = world.identity_root / f"fold{fold}"
    overlay = base / "overlay"
    files = []
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        relative = f"identity/fold{fold}/{path.relative_to(source).as_posix()}"
        target = overlay / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        files.append({"path": relative, "role": "identity", "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                      "size_bytes": target.stat().st_size})
    (overlay / pc.OVERLAY_MANIFEST_NAME).write_text(json.dumps({"files": files}, indent=2))
    return overlay


def _tree_digest(root):
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()}


@pytest.mark.parametrize("fold", [0, 1])
def test_holdout_lists_are_built_in_a_scratch_copy_and_the_overlay_is_never_written(world, tmp_path, fold):
    overlay = _fake_overlay(world, tmp_path, fold)
    before = _tree_digest(overlay)
    receipt = pc.build_holdout_lists(overlay, tmp_path / "campaign", fold, provenance={"note": "synthetic"})
    assert _tree_digest(overlay) == before
    work = pc.holdout_work_dir(tmp_path / "campaign", fold)
    assert work == tmp_path / "campaign" / "amendment2" / "f6_holdout" / f"fold{fold}" / "work"
    for inner in range(identity.N_INNER_FOLDS):
        assert (work / f"r1_inner_exclusions_fold{fold}_inner{inner}.tsv").is_file()
    assert sorted(receipt["outputs"]) == ["0", "1", "2", "3"] and receipt["n_files_copied"] == len(before) - 1
    if fold == 0:  # the copy already held the overlay's lists: write_once proved them equal
        source = world.identity_root / "fold0"
        for inner in range(identity.N_INNER_FOLDS):
            name = f"r1_inner_exclusions_fold0_inner{inner}.tsv"
            assert (work / name).read_bytes() == (source / name).read_bytes()
    with pytest.raises(pc.PretrainCorpusError, match="built once"):
        pc.build_holdout_lists(overlay, tmp_path / "campaign", fold)


def test_holdout_lists_refuse_a_file_that_differs_from_the_overlay_manifest(world, tmp_path):
    overlay = _fake_overlay(world, tmp_path, 1)
    manifest = json.loads((overlay / pc.OVERLAY_MANIFEST_NAME).read_text())
    target = overlay / manifest["files"][0]["path"]
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(pc.PretrainCorpusError, match="differs from the overlay manifest"):
        pc.build_holdout_lists(overlay, tmp_path / "campaign", 1)
    assert not pc.holdout_work_dir(tmp_path / "campaign", 1).exists()
    assert not pc.holdout_work_dir(tmp_path / "campaign", 1).with_name("work.partial").exists()


def test_an_empty_attached_folder_verifies_as_not_attached_and_a_partial_one_is_refused(smoke_corpus, tmp_path):
    from ec_clean30 import views

    view, _info = smoke_corpus
    copy_root = tmp_path / CAMPAIGN_ID / SMOKE_VIEWS_DIR / "empty"
    shutil.copytree(view.root, copy_root)
    make_writable(copy_root)
    folder = copy_root / pc.WHOLE_PROTEIN_PRETRAIN_DIR
    manifest_before = (copy_root / views.VIEW_MANIFEST_NAME).read_bytes()
    shutil.rmtree(folder)
    folder.mkdir()  # contract X9: unpack creates it empty
    views._set_permissions(copy_root)  # data read-only again; the attached folder stays as it is
    try:
        copy = View.load(copy_root)
        assert pc.verify_whole_protein_pretrain(copy) == {"dir": pc.WHOLE_PROTEIN_PRETRAIN_DIR, "attached": False}
        with pytest.raises(pc.PretrainCorpusError, match="required but empty"):
            pc.verify_whole_protein_pretrain(copy, required=True)
        receipt = views.verify_view(copy_root, write_receipt=False, provenance={"note": "synthetic"})
        assert receipt["whole_protein_pretrain"] == {"dir": pc.WHOLE_PROTEIN_PRETRAIN_DIR, "attached": False}
        (folder / pc.CORPUS_NAME).write_text("accession\n")
        with pytest.raises(pc.PretrainCorpusError, match="unfinished corpus build"):
            pc.verify_whole_protein_pretrain(View.load(copy_root))
        with pytest.raises(views.ViewVerificationError, match="does not verify"):
            views.verify_view(copy_root, write_receipt=False, provenance={"note": "synthetic"})
        assert (copy_root / views.VIEW_MANIFEST_NAME).read_bytes() == manifest_before
        assert View.load(copy_root).manifest_sha256 == view.manifest_sha256
    finally:
        make_writable(copy_root)


def test_the_attached_folder_is_outside_the_view_manifest_and_verified_when_present(smoke_corpus):
    from ec_clean30 import views

    view, _info = smoke_corpus
    assert pc.WHOLE_PROTEIN_PRETRAIN_DIR == views.WHOLE_PROTEIN_PRETRAIN_DIR and views.ATTACHED_DIRS == (
        views.WHOLE_PROTEIN_PRETRAIN_DIR,)
    assert not any(item["path"].startswith("whole_protein_pretrain/") for item in view.manifest["files"])
    assert "whole_protein_pretrain" not in view.manifest["writable_dirs"]
    receipt = views.verify_view(view.root, require_whole_protein_pretrain=True, write_receipt=False,
                                provenance={"note": "synthetic"})
    assert receipt["whole_protein_pretrain"]["corpus_sha256"] == pc.view_pretrain_inputs(view).corpus_sha256
    plain = views.verify_view(view.root, write_receipt=False, provenance={"note": "synthetic"})
    assert plain["whole_protein_pretrain"] == receipt["whole_protein_pretrain"]  # checked when present
