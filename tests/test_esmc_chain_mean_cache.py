"""Chain-mean ESMC caches built from per-residue payloads: format, manifest, loader, attachment.

Synthetic structures and payloads only; no dataset access.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from embed_helpers.esmc import extract_chain_sequences, parse_structure  # noqa: E402
from embed_helpers.esmc_chain_means import build_from_residue_cache, main  # noqa: E402
from pmm_campaign_fixtures import write_pdb  # noqa: E402
from structure_store import write_structure_manifest_for_paths  # noqa: E402
from training.esm_feature_loading import (  # noqa: E402
    ESM_FILE_RE,
    build_embedding_payload,
    residue_keys_for_structure_chain,
)
from training.global_esm_features import (  # noqa: E402
    CHAIN_MEAN_FORMAT,
    GLOBAL_ESM_METADATA_KEY,
    GlobalEsmCacheError,
    attach_global_esm_to_pockets,
    chain_mean_filename,
    chain_mean_payload,
    copy_cache_subset,
    layer_name,
    load_global_esm_vectors,
    normalize_layer,
    sequence_mean_filename,
    serialize_payload,
    write_subset_manifest,
)

DIM = 960
STEMS = ("P00001__chain_A__EC_1.1.1.1", "P00002__chain_A__EC_3.4.24.1", "P00003__chain_B__EC_2.7.1.1")
DECOY_STEM = "P00001__chain_A__EC_1.1.1.10"
UNLISTED_STEM = "P00004__chain_A__EC_4.2.1.1"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_structure(structure_dir: Path, stem: str, n_ions: int = 1) -> Path:
    ions = [("A", 900 + k, "", "ZN", (12.0 * k, 0.0, 0.0)) for k in range(n_ions)]
    path = structure_dir / f"{stem}.pdb"
    write_pdb(path, ions=ions, chains=("A", "B"))
    return path


def write_residue_payload(embeddings_dir: Path, structure_path: Path, *, seed: int, mutate=None, name=None) -> torch.Tensor:
    stem = structure_path.stem
    chain = stem.split("__chain_")[1].split("__")[0]
    structure = parse_structure(structure_path)
    sequence = extract_chain_sequences(structure)[chain]
    residue_ids, _ = residue_keys_for_structure_chain(structure, chain)
    rows = torch.randn(len(residue_ids), DIM, generator=torch.Generator().manual_seed(seed))
    payload = build_embedding_payload(
        rows, residue_ids, structure_id=stem, chain_id=chain, source_path=structure_path.name,
        metadata={"esm_model_name": "esmc_300m", "embedding_dim": DIM, "source_sequence": sequence,
                  "source_sequence_sha256": _sha(sequence.encode())},
    )
    if mutate is not None:
        payload = mutate(payload)
    torch.save(payload, embeddings_dir / (name or f"{stem}_chain_{chain}_esmc.pt"))
    return rows


def make_dataset(root: Path, stems=STEMS) -> SimpleNamespace:
    structures, embeddings = root / "structures", root / "esm"
    structures.mkdir(parents=True)
    embeddings.mkdir(parents=True)
    rows = {}
    for index, stem in enumerate(stems):
        path = write_structure(structures, stem, n_ions=1 + index % 2)
        rows[stem] = write_residue_payload(embeddings, path, seed=index)
    allowlist = root / "allowlist.txt"
    allowlist.write_text("# training IDs\n" + "".join(f"{stem}\n" for stem in reversed(stems)), encoding="utf-8")
    return SimpleNamespace(root=root, structures=structures, embeddings=embeddings, rows=rows, allowlist=allowlist)


def build_cache(data: SimpleNamespace, out: Path) -> str:
    summary = build_from_residue_cache(data.allowlist, data.structures, data.embeddings, out)
    assert summary["manifest_sha256"] == _sha((out / "manifest.json").read_bytes())
    return summary["manifest_sha256"]


def test_file_names_and_layers():
    stem = STEMS[0]
    assert chain_mean_filename(stem, "final") == f"{stem}__esmc300m_final_chainmean.pt"
    assert chain_mean_filename(stem, 26) == f"{stem}__esmc300m_block26_chainmean.pt"
    assert chain_mean_filename(stem, "block5") == f"{stem}__esmc300m_block05_chainmean.pt"
    assert sequence_mean_filename("a" * 64, "final") == f"{'a' * 64}__esmc300m_final_seqmean.pt"
    assert normalize_layer(None) == "final" and layer_name(3) == "block03"
    # Command-line flags pass layers as strings; digits name a zero-based block.
    assert normalize_layer("26") == "block26" and normalize_layer("05") == "block05" and normalize_layer(7) == "block07"
    assert chain_mean_filename(stem, "26") == chain_mean_filename(stem, 26)
    for name in (chain_mean_filename(stem, "final"), chain_mean_filename(stem, 26)):
        assert not name.endswith("_esmc.pt") and ESM_FILE_RE.match(name) is None
    for bad in (lambda: chain_mean_filename("../x", "final"), lambda: chain_mean_filename(stem, "layer3"),
                lambda: layer_name(-1), lambda: normalize_layer("-1"), lambda: normalize_layer("2.5"),
                lambda: sequence_mean_filename("abc", "final"),
                lambda: chain_mean_filename(stem, "final", model_name="ESMC-300M")):
        with pytest.raises(GlobalEsmCacheError):
            bad()


def test_from_residue_cache_reproduces_the_direct_mean(tmp_path):
    data = make_dataset(tmp_path)
    out = tmp_path / "chain_means"
    assert main(["from-residue-cache", "--allowlist", str(data.allowlist),
                 "--allowlist-sha256", _sha(data.allowlist.read_bytes()), "--structure-dir", str(data.structures),
                 "--embeddings-dir", str(data.embeddings), "--out", str(out)]) == 0
    manifest_bytes = (out / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    vectors = load_global_esm_vectors(out, _sha(manifest_bytes), list(STEMS), "final", DIM)
    assert list(vectors) == list(STEMS)
    for stem in STEMS:
        expected = data.rows[stem].double().mean(dim=0).float()
        assert vectors[stem].dtype == torch.float32 and vectors[stem].shape == (DIM,)
        assert torch.isfinite(vectors[stem]).all() and torch.equal(vectors[stem], expected)

    stem = STEMS[2]
    payload = torch.load(out / chain_mean_filename(stem, "final"), weights_only=True)
    sequence = extract_chain_sequences(parse_structure(data.structures / f"{stem}.pdb"))["B"]
    residue_file = data.embeddings / f"{stem}_chain_B_esmc.pt"
    assert payload["format"] == CHAIN_MEAN_FORMAT and payload["pooling"] == "mean_all_residue_rows"
    assert (payload["chain_id"], payload["layer"], payload["layer_index_zero_based"]) == ("B", "final", None)
    assert payload["n_residues"] == len(sequence) and payload["source_sequence_sha256"] == _sha(sequence.encode())
    assert payload["source"]["embedding_sha256"] == _sha(residue_file.read_bytes())
    assert manifest["allowlist"]["file_sha256"] == _sha(data.allowlist.read_bytes())
    assert manifest["n_keys"] == 3 and [entry["structure_id"] for entry in manifest["files"]] == sorted(STEMS)
    assert all(entry["sha256"] == _sha((out / entry["file"]).read_bytes()) for entry in manifest["files"])
    # Manifest entries carry the layer record that per-file sidecars would otherwise hold.
    assert {(entry["layer"], entry["layer_index_zero_based"], entry["layer_role"]) for entry in manifest["files"]} == {
        ("final", None, "final_normalized")}
    receipt = json.loads((out / "build_receipt.json").read_text())
    assert receipt["manifest_sha256"] == _sha(manifest_bytes) and receipt["allowlist_sha256_checked"] is True
    assert receipt["structure_manifest"] is None


def _set_metadata(field, value):
    def mutate(payload):
        payload["metadata"][field] = value
        return payload
    return mutate


def _drop_metadata(field):
    def mutate(payload):
        payload["metadata"].pop(field)
        return payload
    return mutate


def _set_rows(change):
    def mutate(payload):
        payload["embeddings"] = change(payload["embeddings"].clone())
        return payload
    return mutate


def _nan_row(rows):
    rows[0, 0] = float("nan")
    return rows


@pytest.mark.parametrize(
    "mutate, message",
    [
        (_set_metadata("source_sequence_sha256", "0" * 64), "sequence SHA-256 differs"),
        (_set_metadata("source_sequence_sha256", None), "records no source_sequence_sha256"),
        (_set_metadata("source_sequence", "HCEADW"), "recorded sequence differs"),
        (_set_metadata("esm_model_name", "esmc_600m"), "model"),
        (_drop_metadata("embedding_dim"), "metadata embedding_dim None"),
        (_set_rows(lambda rows: rows[:, :-1]), "width"),
        (_set_rows(_nan_row), "non-finite"),
        (lambda payload: {**payload, "residue_ids": payload["residue_ids"][:-1]}, "rows for"),
        (lambda payload: {**payload, "residue_ids": payload["residue_ids"][::-1]}, "residue IDs differ"),
        (lambda payload: {**payload, "chain_id": "Z"}, "chain_id"),
        (lambda payload: {**payload, "structure_id": DECOY_STEM}, "structure_id '"),
        (lambda payload: payload["embeddings"], "legacy raw-tensor"),
    ],
)
def test_bad_residue_payloads_fail_and_write_nothing(tmp_path, mutate, message):
    data = make_dataset(tmp_path)
    write_residue_payload(data.embeddings, data.structures / f"{STEMS[0]}.pdb", seed=9, mutate=mutate)
    out = tmp_path / "chain_means"
    with pytest.raises(GlobalEsmCacheError, match=message):
        build_from_residue_cache(data.allowlist, data.structures, data.embeddings, out)
    assert list(out.iterdir()) == []


def test_missing_inputs_and_bad_allowlists(tmp_path):
    data = make_dataset(tmp_path)
    (data.embeddings / f"{STEMS[1]}_chain_A_esmc.pt").unlink()
    with pytest.raises(GlobalEsmCacheError, match=f"{STEMS[1]}: missing"):
        build_from_residue_cache(data.allowlist, data.structures, data.embeddings, tmp_path / "a")

    with pytest.raises(GlobalEsmCacheError, match="SHA-256"):
        build_from_residue_cache(data.allowlist, data.structures, data.embeddings, tmp_path / "hash",
                                 expected_allowlist_sha256="0" * 64)
    assert not (tmp_path / "hash").exists()

    repeated = tmp_path / "repeated.txt"
    repeated.write_text(f"{STEMS[0]}\n{STEMS[0]}\n")
    with pytest.raises(GlobalEsmCacheError, match="repeats"):
        build_from_residue_cache(repeated, data.structures, data.embeddings, tmp_path / "b")

    unknown = tmp_path / "unknown.txt"
    unknown.write_text(f"{UNLISTED_STEM}\n")
    with pytest.raises(GlobalEsmCacheError, match="no structure file"):
        build_from_residue_cache(unknown, data.structures, data.embeddings, tmp_path / "c")

    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "other.txt").write_text("x")
    with pytest.raises(GlobalEsmCacheError, match="not empty"):
        build_from_residue_cache(data.allowlist, data.structures, data.embeddings, busy)


def test_inputs_resolving_outside_permitted_roots_are_refused(tmp_path):
    data = make_dataset(tmp_path)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    target = data.embeddings / f"{STEMS[0]}_chain_A_esmc.pt"
    moved = outside / target.name
    target.rename(moved)
    target.symlink_to(moved)
    with pytest.raises(GlobalEsmCacheError, match="outside the permitted roots"):
        build_from_residue_cache(data.allowlist, data.structures, data.embeddings, tmp_path / "a")
    summary = build_from_residue_cache(data.allowlist, data.structures, data.embeddings, tmp_path / "b",
                                       permitted_roots=[outside])
    assert summary["n_files"] == 3


def _set_structure_manifest_sha(manifest: Path, structure_name: str, digest: str) -> None:
    lines = manifest.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        fields = line.split(",")
        if fields[0] == structure_name:
            fields[2] = digest
            lines[index] = ",".join(fields)
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_structure_manifest_is_read_only_when_named(tmp_path):
    data = make_dataset(tmp_path)
    extra = write_structure(data.structures, UNLISTED_STEM)
    manifest = write_structure_manifest_for_paths(
        data.structures, [data.structures / f"{stem}.pdb" for stem in STEMS] + [extra])
    extra.unlink()  # a row outside the allowlist must never be opened or hashed

    def build(out, **kwargs):
        return build_from_residue_cache(data.allowlist, data.structures, data.embeddings, tmp_path / out, **kwargs)

    summary = build("named", structure_manifest=manifest)
    assert summary["structure_manifest"] == manifest.name
    assert len(load_global_esm_vectors(tmp_path / "named", summary["manifest_sha256"], list(STEMS))) == 3

    with (data.structures / f"{STEMS[0]}.pdb").open("a") as handle:
        handle.write("REMARK changed after the manifest was written\n")
    with pytest.raises(GlobalEsmCacheError, match="structure SHA-256 differs"):
        build("changed", structure_manifest=manifest)
    # Without the opt-in the manifest is not used: exact names resolve and nothing is compared to it.
    assert build("by_name")["structure_manifest"] is None

    _set_structure_manifest_sha(manifest, f"{STEMS[1]}.pdb", "")
    with pytest.raises(GlobalEsmCacheError, match=f"{STEMS[1]}: the structure manifest row records no SHA-256"):
        build("no_sha", structure_manifest=manifest)

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    moved = elsewhere / manifest.name
    moved.write_bytes(manifest.read_bytes())
    with pytest.raises(GlobalEsmCacheError, match="outside the permitted roots"):
        build("outside", structure_manifest=moved)


def test_reads_open_exact_names_only(tmp_path):
    data = make_dataset(tmp_path)
    cache = tmp_path / "chain_means"
    sha = build_cache(data, cache)
    # Shared-prefix decoys and an unlisted protein sit next to the requested files.
    decoy_structure = write_structure(data.structures, DECOY_STEM)
    write_residue_payload(data.embeddings, decoy_structure, seed=21)
    unlisted_structure = write_structure(data.structures, UNLISTED_STEM)
    write_residue_payload(data.embeddings, unlisted_structure, seed=22)
    # A shared-folder structure manifest is never opened unless it is named explicitly.
    shared_manifest = write_structure_manifest_for_paths(
        data.structures, [data.structures / f"{stem}.pdb" for stem in (*STEMS, DECOY_STEM, UNLISTED_STEM)])
    decoy_mean = cache / chain_mean_filename(DECOY_STEM, "final")
    decoy_mean.write_bytes((cache / chain_mean_filename(STEMS[0], "final")).read_bytes())
    (cache / (chain_mean_filename(STEMS[0], "final") + ".partial")).write_bytes(b"partial")
    forbidden = [decoy_structure, data.embeddings / f"{DECOY_STEM}_chain_A_esmc.pt", unlisted_structure,
                 data.embeddings / f"{UNLISTED_STEM}_chain_A_esmc.pt", decoy_mean,
                 cache / chain_mean_filename(STEMS[2], "final"), cache / "build_receipt.json", shared_manifest]
    watched = [data.structures, data.embeddings, cache]
    script = f"""
import json, os, sys
sys.path.insert(0, {str(SRC)!r})
from embed_helpers.esmc_chain_means import build_from_residue_cache
from training.access_guard import install_forbidden_read_guard
from training.global_esm_features import load_global_esm_vectors
watched = {[os.path.realpath(path) for path in watched]!r}
events = []
def record(event, args):
    if event in {{"open", "os.listdir", "os.scandir", "glob.glob"}} and args and isinstance(args[0], (str, os.PathLike)):
        events.append((event, os.path.realpath(os.fspath(args[0]))))
sys.addaudithook(record)
install_forbidden_read_guard({[str(path) for path in forbidden]!r})
build_from_residue_cache({str(data.allowlist)!r}, {str(data.structures)!r}, {str(data.embeddings)!r},
                         {str(tmp_path / "rebuilt")!r})
vectors = load_global_esm_vectors({str(cache)!r}, {sha!r}, {[STEMS[0], STEMS[1]]!r})
print(json.dumps({{
    "listed": [path for event, path in events if event != "open" and path in watched],
    "opened_cache": sorted(os.path.basename(path) for event, path in events
                           if event == "open" and os.path.dirname(path) == watched[2]),
    "n": len(vectors),
}}))
"""
    # Torch subprocesses use the GNU threading layer, as the campaign runners do, whatever MKL setting
    # an earlier import in this test process left in the environment.
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                            env={**os.environ, "MKL_THREADING_LAYER": "GNU"})
    assert result.returncode == 0, result.stderr[-2000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["listed"] == [] and report["n"] == 2
    assert report["opened_cache"] == sorted(
        ["manifest.json", chain_mean_filename(STEMS[0], "final"), chain_mean_filename(STEMS[1], "final")]
    )


def test_manifest_and_file_hash_failures(tmp_path):
    data = make_dataset(tmp_path)
    cache = tmp_path / "chain_means"
    sha = build_cache(data, cache)
    with pytest.raises(GlobalEsmCacheError, match="SHA-256"):
        load_global_esm_vectors(cache, "0" * 64, list(STEMS))
    with pytest.raises(GlobalEsmCacheError, match="not 'block26'"):
        load_global_esm_vectors(cache, sha, list(STEMS), "block26")
    with pytest.raises(GlobalEsmCacheError, match="embedding_dim"):
        load_global_esm_vectors(cache, sha, list(STEMS), "final", 959)
    with pytest.raises(GlobalEsmCacheError, match="model"):
        load_global_esm_vectors(cache, sha, list(STEMS), expected_model_name="esmc_600m")
    with pytest.raises(GlobalEsmCacheError, match="repeat"):
        load_global_esm_vectors(cache, sha, [STEMS[0], STEMS[0]])

    # All or nothing: one unknown ID fails the whole request and is named.
    with pytest.raises(GlobalEsmCacheError, match=UNLISTED_STEM):
        load_global_esm_vectors(cache, sha, [STEMS[0], UNLISTED_STEM])

    first, second = (cache / chain_mean_filename(stem, "final") for stem in STEMS[:2])
    original = second.read_bytes()
    second.write_bytes(first.read_bytes())
    with pytest.raises(GlobalEsmCacheError, match="differs from the manifest"):
        load_global_esm_vectors(cache, sha, list(STEMS))
    second.write_bytes(original)

    outside = tmp_path / "outside.pt"
    outside.write_bytes(first.read_bytes())
    first.unlink()
    first.symlink_to(outside)
    with pytest.raises(GlobalEsmCacheError, match="resolves outside the cache directory"):
        load_global_esm_vectors(cache, sha, [STEMS[0]])

    manifest = json.loads((cache / "manifest.json").read_text())
    manifest["files"][1]["file"] = "../" + manifest["files"][1]["file"]
    edited = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    (cache / "manifest.json").write_bytes(edited)
    with pytest.raises(GlobalEsmCacheError, match="manifest names"):
        load_global_esm_vectors(cache, _sha(edited), [STEMS[1]])


def _pocket(structure_id):
    return SimpleNamespace(structure_id=structure_id, metadata={})


def test_attach_is_all_or_nothing():
    vectors = {stem: torch.full((DIM,), float(index)) for index, stem in enumerate(STEMS)}
    pockets = [_pocket(STEMS[0]), _pocket(STEMS[0]), _pocket(STEMS[1]), _pocket(STEMS[2])]
    partial = {stem: vectors[stem] for stem in STEMS[:2]}
    with pytest.raises(GlobalEsmCacheError, match=STEMS[2]):
        attach_global_esm_to_pockets(pockets, partial)
    assert all(GLOBAL_ESM_METADATA_KEY not in pocket.metadata for pocket in pockets)

    report = attach_global_esm_to_pockets(pockets, {**vectors, "P00009__chain_A__EC_1.1.1.1": torch.zeros(DIM)})
    assert report == {"metadata_key": GLOBAL_ESM_METADATA_KEY, "n_pockets": 4, "n_structures": 3,
                      "embedding_dim": DIM, "n_unused_vectors": 1}
    first, second = (pocket.metadata[GLOBAL_ESM_METADATA_KEY] for pocket in pockets[:2])
    assert first is not second and torch.equal(first, second)
    first.add_(10.0)  # an in-place change to one pocket's vector reaches no other pocket
    assert torch.equal(second, torch.zeros(DIM))
    vectors[STEMS[1]].add_(5.0)  # the attached copy is independent of the caller's tensor
    assert torch.equal(pockets[2].metadata[GLOBAL_ESM_METADATA_KEY], torch.full((DIM,), 1.0))
    with pytest.raises(GlobalEsmCacheError, match="already carry"):
        attach_global_esm_to_pockets(pockets, vectors)

    fresh = [_pocket(STEMS[0]), _pocket(STEMS[1])]
    with pytest.raises(GlobalEsmCacheError, match="non-finite"):
        attach_global_esm_to_pockets(fresh, {STEMS[0]: torch.full((DIM,), float("inf")), STEMS[1]: torch.zeros(DIM)})
    with pytest.raises(GlobalEsmCacheError, match="mixed widths"):
        attach_global_esm_to_pockets(fresh, {STEMS[0]: torch.zeros(DIM), STEMS[1]: torch.zeros(DIM - 1)})
    assert all(not pocket.metadata for pocket in fresh)


def test_outputs_are_byte_deterministic(tmp_path):
    data = make_dataset(tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"
    assert build_cache(data, first) == build_cache(data, second)

    def contents(folder):
        return {path.name: path.read_bytes() for path in sorted(folder.iterdir()) if path.name != "build_receipt.json"}

    assert contents(first) == contents(second) and len(contents(first)) == 4


def test_payload_of_a_view_stores_only_its_own_values():
    rows = torch.randn(10, DIM, generator=torch.Generator().manual_seed(3))
    record = dict(structure_id=STEMS[0], layer="final", n_residues=10, source={}, source_sequence_sha256="a" * 64)
    from_view = serialize_payload(chain_mean_payload(rows[3], **record))
    assert from_view == serialize_payload(chain_mean_payload(rows[3].clone(), **record))
    assert len(from_view) < rows.numel() * rows.element_size()


def test_subset_manifests(tmp_path, capsys):
    data = make_dataset(tmp_path)
    cache = tmp_path / "chain_means"
    sha = build_cache(data, cache)
    subset_ids = [STEMS[0], STEMS[2]]
    path, subset_sha = write_subset_manifest(cache, manifest_sha256=sha, keys=subset_ids, subset_name="fold0_train")
    assert path.name == "manifest__fold0_train.json"
    subset = json.loads(path.read_text())
    assert subset["subset"]["parent_manifest_sha256"] == sha and subset["n_keys"] == 2
    vectors = load_global_esm_vectors(cache, subset_sha, subset_ids, manifest_name=path.name)
    assert torch.equal(vectors[STEMS[0]], load_global_esm_vectors(cache, sha, [STEMS[0]])[STEMS[0]])
    with pytest.raises(GlobalEsmCacheError, match=STEMS[1]):
        load_global_esm_vectors(cache, subset_sha, [STEMS[1]], manifest_name=path.name)
    assert write_subset_manifest(cache, manifest_sha256=sha, keys=subset_ids, subset_name="fold0_train")[1] == subset_sha
    with pytest.raises(GlobalEsmCacheError, match="not in the cache"):
        write_subset_manifest(cache, manifest_sha256=sha, keys=[UNLISTED_STEM], subset_name="other")

    copied = tmp_path / "fold0_view"
    copy_path, copy_sha = copy_cache_subset(cache, copied, manifest_sha256=sha, keys=subset_ids, subset_name="fold0_train")
    assert copy_sha == subset_sha and copy_path.name == "manifest.json"
    assert sorted(p.name for p in copied.iterdir()) == sorted(
        ["manifest.json"] + [chain_mean_filename(stem, "final") for stem in subset_ids]
    )
    assert set(load_global_esm_vectors(copied, copy_sha, subset_ids)) == set(subset_ids)

    ids_file = tmp_path / "fold1_ids.txt"
    ids_file.write_text(f"{STEMS[1]}\n")
    assert main(["subset", "--cache-dir", str(cache), "--manifest-sha256", sha, "--ids", str(ids_file),
                 "--name", "fold1_train"]) == 0
    assert json.loads(capsys.readouterr().out)["n_ids"] == 1


def test_cli_runs_as_a_script():
    result = subprocess.run([sys.executable, str(SRC / "embed_helpers" / "esmc_chain_means.py"), "--help"],
                            capture_output=True, text=True,
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "MKL_THREADING_LAYER": "GNU"})
    assert result.returncode == 0, result.stderr[-2000:]
    assert "from-residue-cache" in result.stdout and "sequences" in result.stdout
