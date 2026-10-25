"""Model-derived ESMC layer means (from-model, sequences) with a fake model; no weights, no network."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import embed_helpers.esmc_chain_means as chain_means  # noqa: E402
from embed_helpers.esmc import extract_chain_sequences, parse_structure  # noqa: E402
from embed_helpers.esmc_chain_means import (  # noqa: E402
    build_from_model,
    build_from_residue_cache,
    build_sequence_means,
    compare_chain_mean_caches,
    main,
)
from pmm_campaign_fixtures import write_pdb  # noqa: E402
from training.esm_feature_loading import (  # noqa: E402
    ESM_FILE_RE,
    build_embedding_payload,
    embedding_path_candidates,
    residue_keys_for_structure_chain,
)
from training.global_esm_features import (  # noqa: E402
    GlobalEsmCacheError,
    chain_mean_filename,
    load_global_esm_vectors,
    load_sequence_mean_vectors,
    sequence_mean_filename,
)

DIM = 960
N_LAYERS = 4
STEMS = ("P00001__chain_A__EC_1.1.1.1", "P00002__chain_A__EC_3.4.24.1", "P00003__chain_B__EC_2.7.1.1")
_TOKEN_RE = re.compile(r"<mask>|.")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def token_rows(tokens: list[str], offset: float) -> torch.Tensor:
    """Deterministic rows for BOS + tokens + EOS; BOS and EOS rows are far from the rest."""
    codes = torch.tensor([0.0] + [7.0 if token == "<mask>" else float(ord(token)) for token in tokens] + [1.0],
                         dtype=torch.float64)
    position = torch.arange(len(codes), dtype=torch.float64)[:, None]
    columns = torch.arange(DIM, dtype=torch.float64)[None, :]
    rows = torch.sin(0.01 * (columns + 1) * (position + 1) + codes[:, None]) + offset
    rows[0] += 100.0
    rows[-1] -= 100.0
    return rows.float()


def fake_final(sequence: str) -> torch.Tensor:
    return token_rows(_TOKEN_RE.findall(sequence), 0.0) * 0.5 + 0.25


def fake_hidden(sequence: str) -> torch.Tensor:
    tokens = _TOKEN_RE.findall(sequence)
    return torch.stack([token_rows(tokens, 0.1 * layer) for layer in range(N_LAYERS)])


class FakeESMC:
    """Stands in for ESMC: encode() passes the sequence through, logits() returns fixed rows."""

    def __init__(self, *, dtype=torch.float32, drop_eos=False, fail_after=None, with_blocks=True, device="cpu"):
        self.calls: list[str] = []
        self.dtype, self.drop_eos, self.fail_after = dtype, drop_eos, fail_after
        self.device = torch.device(device)
        if with_blocks:
            self.transformer = SimpleNamespace(blocks=[None] * N_LAYERS)

    def parameters(self):
        # Only device and dtype are read, so a stand-in records a CUDA placement without a GPU.
        return iter([SimpleNamespace(device=self.device, dtype=self.dtype)])

    def encode(self, protein):
        return protein.sequence

    def logits(self, sequence, config):
        assert config == {"sequence": False, "return_embeddings": True, "return_hidden_states": True,
                          "ith_hidden_layer": -1}
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RuntimeError("simulated interruption")
        self.calls.append(sequence)
        final, hidden = fake_final(sequence), fake_hidden(sequence)
        if self.drop_eos:
            final, hidden = final[:-1], hidden[:, :-1]
        return SimpleNamespace(embeddings=final[None], hidden_states=hidden[:, None])


FAKE_SDK = (FakeESMC, lambda sequence: SimpleNamespace(sequence=sequence), lambda **kwargs: dict(kwargs))


def expected_mean(rows: torch.Tensor) -> torch.Tensor:
    return rows[1:-1].double().mean(dim=0).float()


def chain_of(stem: str) -> str:
    return stem.split("__chain_")[1].split("__")[0]


def write_sequence_map(path: Path, sequences: dict[str, str], *, with_sequences: bool = False) -> Path:
    header = "accession\tsha256\tlength" + ("\tsequence" if with_sequences else "")
    rows = [f"{accession}\t{_sha(sequence.encode())}\t{len(sequence)}" + (f"\t{sequence}" if with_sequences else "")
            for accession, sequence in sequences.items()]
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


def make_structures(root: Path) -> SimpleNamespace:
    structures = root / "structures"
    structures.mkdir(parents=True)
    sequences = {}
    for index, stem in enumerate(STEMS):
        path = structures / f"{stem}.pdb"
        ions = [("A", 900 + k, "", "ZN", (12.0 * k, 0.0, 0.0)) for k in range(1 + index % 2)]
        write_pdb(path, ions=ions, chains=("A", "B"))
        sequences[stem] = extract_chain_sequences(parse_structure(path))[chain_of(stem)]
    allowlist = root / "allowlist.txt"
    allowlist.write_text("".join(f"{stem}\n" for stem in STEMS), encoding="utf-8")
    return SimpleNamespace(structures=structures, allowlist=allowlist, sequences=sequences)


def run_from_model(data, out, model=None, **kwargs):
    return build_from_model(data.allowlist, data.structures, out, layer_index=2, device="cpu",
                            model=model or FakeESMC(), sdk=FAKE_SDK, **kwargs)


def test_from_model_writes_final_and_block_means(tmp_path):
    data = make_structures(tmp_path)
    model = FakeESMC()
    summary = run_from_model(data, tmp_path / "means", model=model)
    out = tmp_path / "means"
    sha = summary["manifest_sha256"]
    assert summary["layers"] == ["final", "block02"] and summary["n_files"] == 6
    assert model.calls == [data.sequences[stem] for stem in sorted(STEMS)]

    final = load_global_esm_vectors(out, sha, list(STEMS), "final", DIM)
    block = load_global_esm_vectors(out, sha, list(STEMS), 2, DIM)
    for stem in STEMS:
        sequence = data.sequences[stem]
        assert torch.equal(final[stem], expected_mean(fake_final(sequence)))
        assert torch.equal(block[stem], expected_mean(fake_hidden(sequence)[2]))

    payload = torch.load(out / chain_mean_filename(STEMS[0], "block02"), weights_only=True)
    assert (payload["layer"], payload["layer_index_zero_based"], payload["layer_role"]) == (
        "block02", 2, "block_output_before_final_norm")
    runtime = payload["source"]["runtime"]
    assert runtime["device"] == "cpu" and runtime["parameter_dtype"] == "float32"
    assert runtime["autocast_dtype"] is None and runtime["forward_config"]["ith_hidden_layer"] == -1
    assert set(runtime["versions"]) == {"torch", "esm", "numpy", "biopython"} and runtime["versions"]["torch"]
    assert payload["source"]["n_layers"] == N_LAYERS and payload["source"]["token_trim"] == "drop_bos_and_eos"
    assert payload["source_sequence_sha256"] == _sha(data.sequences[STEMS[0]].encode())
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["layers"] == ["block02", "final"] and manifest["build"]["layer_index_zero_based"] == 2
    assert {(entry["layer"], entry["layer_index_zero_based"], entry["layer_role"]) for entry in manifest["files"]} == {
        ("final", None, "final_normalized"), ("block02", 2, "block_output_before_final_norm")}
    settings = json.loads((out / "build_settings.json").read_text())
    assert settings["layer_index_zero_based"] == 2 and settings["allowlist_sha256"] == _sha(data.allowlist.read_bytes())


def test_cuda_runtime_records_bfloat16_autocast(tmp_path):
    data = make_structures(tmp_path)
    model = FakeESMC(dtype=torch.bfloat16, device="cuda:0")
    model._use_flash_attn = True
    build_from_model(data.allowlist, data.structures, tmp_path / "means", layer_index=1, device="cuda",
                     model=model, sdk=FAKE_SDK)
    runtime = torch.load(tmp_path / "means" / chain_mean_filename(STEMS[0], "final"), weights_only=True)["source"]["runtime"]
    assert (runtime["requested_device"], runtime["device"], runtime["parameter_dtype"]) == ("cuda", "cuda:0", "bfloat16")
    assert runtime["autocast_dtype"] == "bfloat16" and runtime["use_flash_attn"] is True


def test_layer_index_is_validated(tmp_path):
    data = make_structures(tmp_path)
    model = FakeESMC()
    with pytest.raises(GlobalEsmCacheError, match="out of range for a 4-block model"):
        build_from_model(data.allowlist, data.structures, tmp_path / "a", layer_index=N_LAYERS, device="cpu",
                         model=model, sdk=FAKE_SDK)
    assert model.calls == []
    with pytest.raises(GlobalEsmCacheError, match="out of range for a 4-block model"):
        build_from_model(data.allowlist, data.structures, tmp_path / "b", layer_index=N_LAYERS, device="cpu",
                         model=FakeESMC(with_blocks=False), sdk=FAKE_SDK)
    with pytest.raises(GlobalEsmCacheError, match="non-negative"):
        build_from_model(data.allowlist, data.structures, tmp_path / "c", layer_index=-1, device="cpu",
                         model=FakeESMC(), sdk=FAKE_SDK)
    with pytest.raises(SystemExit):
        main(["from-model", "--allowlist", str(data.allowlist), "--structure-dir", str(data.structures),
              "--out", str(tmp_path / "d"), "--device", "cpu"])


def test_unexpected_token_positions_are_refused(tmp_path):
    data = make_structures(tmp_path)
    with pytest.raises(GlobalEsmCacheError, match="BOS and EOS included"):
        run_from_model(data, tmp_path / "means", model=FakeESMC(drop_eos=True))


def write_residue_cache(data: SimpleNamespace, embeddings: Path, *, offset: float = 0.0) -> Path:
    embeddings.mkdir()
    for stem in STEMS:
        chain = chain_of(stem)
        structure = parse_structure(data.structures / f"{stem}.pdb")
        residue_ids, _ = residue_keys_for_structure_chain(structure, chain)
        sequence = data.sequences[stem]
        payload = build_embedding_payload(
            fake_final(sequence)[1:-1] + offset, residue_ids, structure_id=stem, chain_id=chain,
            metadata={"esm_model_name": "esmc_300m", "embedding_dim": DIM,
                      "source_sequence_sha256": _sha(sequence.encode())},
        )
        torch.save(payload, embeddings / f"{stem}_chain_{chain}_esmc.pt")
    return embeddings


def test_final_layer_matches_the_residue_cache_route(tmp_path):
    data = make_structures(tmp_path)
    embeddings = write_residue_cache(data, tmp_path / "esm")
    from_cache = build_from_residue_cache(data.allowlist, data.structures, embeddings, tmp_path / "from_cache")
    from_model = run_from_model(data, tmp_path / "from_model")
    report = compare_chain_mean_caches(tmp_path / "from_cache", from_cache["manifest_sha256"],
                                       tmp_path / "from_model", from_model["manifest_sha256"], list(STEMS))
    assert report["max_abs_diff"] == 0.0 and report["min_cosine"] == pytest.approx(1.0, abs=1e-12)
    assert main(["compare", "--cache-a", str(tmp_path / "from_cache"),
                 "--manifest-a-sha256", from_cache["manifest_sha256"], "--cache-b", str(tmp_path / "from_model"),
                 "--manifest-b-sha256", from_model["manifest_sha256"], "--ids", str(data.allowlist),
                 "--min-cosine", "0.9999", "--out", str(tmp_path / "parity.json")]) == 0
    assert json.loads((tmp_path / "parity.json").read_text())["passed"] is True

    # Without threshold flags the pre-declared tolerance applies, so a shifted cache fails.
    shifted = build_from_residue_cache(data.allowlist, data.structures,
                                       write_residue_cache(data, tmp_path / "esm_shifted", offset=1e-3),
                                       tmp_path / "shifted")
    assert main(["compare", "--cache-a", str(tmp_path / "shifted"), "--manifest-a-sha256", shifted["manifest_sha256"],
                 "--cache-b", str(tmp_path / "from_model"), "--manifest-b-sha256", from_model["manifest_sha256"],
                 "--ids", str(data.allowlist), "--out", str(tmp_path / "shifted.json")]) == 1
    report = json.loads((tmp_path / "shifted.json").read_text())
    assert report["passed"] is False and report["thresholds"] == {"min_cosine": 0.9999, "max_abs_diff": 1e-4}
    assert report["max_abs_diff"] == pytest.approx(1e-3, rel=1e-3)

    # A layer given as digits, as a command-line flag passes it, names the zero-based block.
    assert main(["compare", "--cache-a", str(tmp_path / "from_model"), "--manifest-a-sha256", from_model["manifest_sha256"],
                 "--cache-b", str(tmp_path / "from_model"), "--manifest-b-sha256", from_model["manifest_sha256"],
                 "--ids", str(data.allowlist), "--layer", "2", "--out", str(tmp_path / "block.json")]) == 0
    assert json.loads((tmp_path / "block.json").read_text())["layer"] == "block02"


def test_resume_reuses_finished_payloads_and_matches_an_uninterrupted_run(tmp_path):
    data = make_structures(tmp_path)
    reference = run_from_model(data, tmp_path / "reference")
    out = tmp_path / "resumed"
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_from_model(data, out, model=FakeESMC(fail_after=2))
    assert not (out / "manifest.json").exists()
    (out / (chain_mean_filename(STEMS[2], "final") + ".partial")).write_bytes(b"torn write")

    with pytest.raises(GlobalEsmCacheError, match="not empty"):
        run_from_model(data, out)
    with pytest.raises(GlobalEsmCacheError, match=r"does not match this run \(source\.runtime\)"):
        run_from_model(data, out, model=FakeESMC(dtype=torch.bfloat16), resume=True)
    with pytest.raises(GlobalEsmCacheError, match=r"does not match this run \(layer_index_zero_based\)"):
        build_from_model(data.allowlist, data.structures, out, layer_index=1, device="cpu",
                         model=FakeESMC(), sdk=FAKE_SDK, resume=True)
    stray = tmp_path / "stray"
    stray.mkdir()
    (stray / "unrelated.txt").write_text("x")
    with pytest.raises(GlobalEsmCacheError, match="no build_settings.json"):
        run_from_model(data, stray, resume=True)

    model = FakeESMC()
    summary = run_from_model(data, out, model=model, resume=True)
    assert summary["n_reused"] == 2 and model.calls == [data.sequences[sorted(STEMS)[2]]]
    assert summary["manifest_sha256"] == reference["manifest_sha256"]
    assert (out / "manifest.json").read_bytes() == (tmp_path / "reference" / "manifest.json").read_bytes()
    with pytest.raises(GlobalEsmCacheError, match="finished cache"):
        run_from_model(data, out, resume=True)


def test_cli_from_model_loads_the_model_through_one_hook(tmp_path, monkeypatch, capsys):
    data = make_structures(tmp_path)
    requested = []

    def fake_loader(model_name, device):
        requested.append((model_name, device))
        return FakeESMC(), FAKE_SDK

    monkeypatch.setattr(chain_means, "load_esmc_for_layers", fake_loader)
    argv = ["from-model", "--allowlist", str(data.allowlist), "--structure-dir", str(data.structures),
            "--layer-index", "1", "--device", "cpu"]
    assert main([*argv, "--out", str(tmp_path / "means"), "--allowlist-sha256", _sha(data.allowlist.read_bytes())]) == 0
    assert requested == [("esmc_300m", "cpu")]
    summary = json.loads(capsys.readouterr().out)
    assert summary["layers"] == ["final", "block01"] and summary["allowlist_sha256_checked"] is True
    assert main([*argv, "--out", str(tmp_path / "refused"), "--allowlist-sha256", "0" * 64]) == 2
    assert "SHA-256" in capsys.readouterr().err and requested == [("esmc_300m", "cpu")]


def write_fasta(path: Path, records) -> Path:
    path.write_text("".join(f">{label} description\n{sequence}\n" for label, sequence in records), encoding="utf-8")
    return path


def test_sequences_handle_masks_dedup_and_labels(tmp_path):
    accessions = write_sequence_map(tmp_path / "sequence_map.tsv", {"P1": "MKTAYIAK", "P2": "GGSA"})
    masked = "MK<mask>AYI<mask>K"
    fasta = write_fasta(tmp_path / "seed42.fasta",
                        [("P1", "MKTAYIAK"), ("P1_0", masked), ("P2", "GGSA"), ("P2_0", "GGSA")])
    other = write_fasta(tmp_path / "seed43.fasta", [("P1_0", "MKTAY<mask>AK")])
    model = FakeESMC()
    summary = build_sequence_means([fasta, other], accessions, tmp_path / "seq", layer_index=1, device="cpu",
                                   model=model, sdk=FAKE_SDK)
    unique = sorted({_sha(s.encode()) for s in ("MKTAYIAK", masked, "GGSA", "MKTAY<mask>AK")})
    assert summary["n_labels"] == 5 and summary["n_unique_sequences"] == 4 and summary["n_files"] == 8
    assert [_sha(call.encode()) for call in model.calls] == unique

    out = tmp_path / "seq"
    manifest = json.loads((out / "manifest.json").read_text())
    assert [(row["input"], row["label"], row["copy_index"]) for row in manifest["labels"]] == [
        ("seed42.fasta", "P1", None), ("seed42.fasta", "P1_0", 0), ("seed42.fasta", "P2", None),
        ("seed42.fasta", "P2_0", 0), ("seed43.fasta", "P1_0", 0)]
    assert manifest["inputs"][0]["sha256"] == _sha(fasta.read_bytes())
    assert manifest["allowlist"]["file_sha256"] == _sha(accessions.read_bytes()) and manifest["allowlist"]["n_ids"] == 2

    digest = _sha(masked.encode())
    payload = torch.load(out / sequence_mean_filename(digest, "block01"), weights_only=True)
    assert payload["n_tokens"] == len(masked) - 5 * 2 == 8 and payload["n_mask_tokens"] == 2
    assert payload["truncated"] is False and "masked positions are included" in payload["source"]["mask_policy"]
    vectors = load_sequence_mean_vectors(out, summary["manifest_sha256"], [digest], "block01", DIM)
    assert torch.equal(vectors[digest], expected_mean(fake_hidden(masked)[1]))
    final = load_sequence_mean_vectors(out, summary["manifest_sha256"], [digest], "final", DIM)
    assert torch.equal(final[digest], expected_mean(fake_final(masked)))

    # Inputs are read in file-name order, so the order of the arguments does not change the cache.
    again = build_sequence_means([other, fasta], accessions, tmp_path / "seq_again", layer_index=1, device="cpu",
                                 model=FakeESMC(), sdk=FAKE_SDK)
    assert again["manifest_sha256"] == summary["manifest_sha256"]
    assert all((tmp_path / "seq_again" / name).read_bytes() == (out / name).read_bytes()
               for name in (entry["file"] for entry in manifest["files"]))


@pytest.mark.parametrize(
    "text, message",
    [
        (">Q9\nMK\n", "neither a sequence-map accession"),
        (">P1_x\nMK\n", "neither a sequence-map accession"),
        (">P1\nMK\n>P1_01\nM<mask>\n>P1_1\nM<mask>\n", "P1_01: Label 'P1_01' is neither"),
        (">P1\nMK<cls>A\n", "characters other than"),
        (">P1\nmk\n", "characters other than"),
        (">P1\nMK\n>P1\nMK\n", "repeated labels"),
        ("MK\n>P1\nMK\n", "before the first header"),
        ("", "holds no records"),
        (">P1\nMR\n", "sequence differs from the sequence map"),
        (">P1\nMK\n>P1_1\nM<mask>K\n", "has 3 residue tokens; its source has 2"),
        (">P1\nMK\n>P1_1\nR<mask>\n", r"non-mask positions \[0\]"),
        (">P1_1\nM<mask>\n", "no source sequence for P1"),
    ],
)
def test_sequences_refuse_bad_input(tmp_path, text, message):
    accessions = write_sequence_map(tmp_path / "sequence_map.tsv", {"P1": "MK"})
    fasta = tmp_path / "input.fasta"
    fasta.write_text(text)
    model = FakeESMC()
    with pytest.raises(GlobalEsmCacheError, match=message):
        build_sequence_means([fasta], accessions, tmp_path / "seq", layer_index=1, device="cpu",
                             model=model, sdk=FAKE_SDK)
    assert model.calls == [] and not (tmp_path / "seq").exists()


def test_sequence_map_supplies_sources_and_is_hash_checked(tmp_path):
    accessions = write_sequence_map(tmp_path / "sequence_map.tsv", {"P1": "MKTAYIAK"}, with_sequences=True)
    fasta = write_fasta(tmp_path / "seed42.fasta", [("P1_7", "MKT<mask>YIAK")])
    summary = build_sequence_means([fasta], accessions, tmp_path / "seq", layer_index=0, device="cpu",
                                   expected_sequence_map_sha256=_sha(accessions.read_bytes()),
                                   model=FakeESMC(), sdk=FAKE_SDK)
    assert summary["n_labels"] == 1 and summary["sequence_map_sha256_checked"] is True
    with pytest.raises(GlobalEsmCacheError, match="non-mask positions"):
        build_sequence_means([write_fasta(tmp_path / "bad.fasta", [("P1_7", "MKT<mask>YIAR")])], accessions,
                             tmp_path / "bad", layer_index=0, device="cpu", model=FakeESMC(), sdk=FAKE_SDK)
    with pytest.raises(GlobalEsmCacheError, match="SHA-256"):
        build_sequence_means([fasta], accessions, tmp_path / "hash", layer_index=0, device="cpu",
                             expected_sequence_map_sha256="0" * 64, model=FakeESMC(), sdk=FAKE_SDK)

    inconsistent = tmp_path / "inconsistent.tsv"
    inconsistent.write_text(f"accession\tsha256\tsequence\nP1\t{_sha(b'MKTAYIAK')}\tMKTAYIAR\n")
    repeated = tmp_path / "repeated.tsv"
    repeated.write_text(f"accession\tsha256\nP1\t{_sha(b'MK')}\nP1\t{_sha(b'MK')}\n")
    no_hash = tmp_path / "no_hash.tsv"
    no_hash.write_text("accession\tlength\nP1\t2\n")
    for bad_map, message in ((inconsistent, "not uppercase residues hashing"), (repeated, "repeats"),
                             (no_hash, "lacks the columns")):
        with pytest.raises(GlobalEsmCacheError, match=message):
            build_sequence_means([fasta], bad_map, tmp_path / bad_map.stem, layer_index=0, device="cpu",
                                 model=FakeESMC(), sdk=FAKE_SDK)
    assert not any((tmp_path / name).exists() for name in ("bad", "hash", "inconsistent", "repeated", "no_hash"))


def test_sequence_resume_checks_the_settings_and_truncation_records(tmp_path):
    accessions = write_sequence_map(tmp_path / "sequence_map.tsv", {"P1": "MKTAYIAK", "P2": "GGSA"})
    fasta = write_fasta(tmp_path / "input.fasta", [("P1", "MKTAYIAK"), ("P2", "GGSA")])

    def run(out, model=None, **kwargs):
        return build_sequence_means([fasta], accessions, tmp_path / out, layer_index=1, device="cpu",
                                    model=model or FakeESMC(), sdk=FAKE_SDK, **kwargs)

    reference = run("reference")
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run("resumed", model=FakeESMC(fail_after=1))
    with pytest.raises(GlobalEsmCacheError, match=r"does not match this run \(truncation\)"):
        run("resumed", max_residue_tokens=500, resume=True)

    # A limit longer than every sequence leaves the token counts unchanged; only the
    # truncation record differs, and a payload carrying it is still refused.
    run("limit500", max_residue_tokens=500)
    first = sorted(_sha(sequence.encode()) for sequence in ("MKTAYIAK", "GGSA"))[0]
    names = [sequence_mean_filename(first, layer) for layer in ("final", "block01")]
    originals = {name: (tmp_path / "resumed" / name).read_bytes() for name in names}
    for name in names:
        (tmp_path / "resumed" / name).write_bytes((tmp_path / "limit500" / name).read_bytes())
    with pytest.raises(GlobalEsmCacheError, match=r"does not match this run \(source\.truncation\)"):
        run("resumed", resume=True)

    for name, data in originals.items():
        (tmp_path / "resumed" / name).write_bytes(data)
    model = FakeESMC()
    summary = run("resumed", model=model, resume=True)
    assert summary["n_reused"] == 1 and len(model.calls) == 1
    assert summary["manifest_sha256"] == reference["manifest_sha256"]


def test_sequences_truncate_by_residue_tokens(tmp_path):
    accessions = write_sequence_map(tmp_path / "sequence_map.tsv", {"P1": "MKTAYIAK"}, with_sequences=True)
    sequence = "MK<mask>AYIAK"
    fasta = write_fasta(tmp_path / "input.fasta", [("P1_3", sequence)])
    model = FakeESMC()
    build_sequence_means([fasta], accessions, tmp_path / "seq", layer_index=0, device="cpu",
                         max_residue_tokens=3, model=model, sdk=FAKE_SDK)
    assert model.calls == ["MK<mask>"]
    payload = torch.load(tmp_path / "seq" / sequence_mean_filename(_sha(sequence.encode()), "final"), weights_only=True)
    assert (payload["n_tokens"], payload["n_tokens_input"], payload["n_mask_tokens"], payload["truncated"]) == (
        3, 8, 1, True)
    assert payload["source"]["truncation"]["max_residue_tokens"] == 3
    assert torch.equal(payload["mean"], expected_mean(fake_final("MK<mask>")))


def test_outputs_are_never_picked_up_by_the_residue_loader(tmp_path):
    data = make_structures(tmp_path)
    out = tmp_path / "means"
    run_from_model(data, out)
    accessions = write_sequence_map(tmp_path / "sequence_map.tsv", {"P1": "MKT"})
    build_sequence_means([write_fasta(tmp_path / "in.fasta", [("P1", "MKT")])], accessions, out / "seq",
                         layer_index=1, device="cpu", model=FakeESMC(), sdk=FAKE_SDK)
    written = [path for folder in (out, out / "seq") for path in folder.iterdir() if path.suffix == ".pt"]
    assert len(written) == 8
    assert all(ESM_FILE_RE.match(path.name) is None for path in written)
    for stem in STEMS:
        found = [path for path in embedding_path_candidates(out, Path(f"{stem}.pdb")) if path.is_file()]
        assert found == []
