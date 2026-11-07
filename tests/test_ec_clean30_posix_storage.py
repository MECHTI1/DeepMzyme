"""POSIX staging layout, capacity accounting and refusal boundaries without cloud use."""
import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from ec_clean30 import runpod_bootstrap as rb


@pytest.fixture
def storage_request(tmp_path, monkeypatch):
    local = tmp_path / "posix"
    monkeypatch.setattr(rb.shutil, "disk_usage", lambda path: SimpleNamespace(free=25 * 1024 ** 3))
    components = {"main_env": 8192, "clean_env": 4096, "views": 2048}
    return {"format": rb.POSIX_STORAGE_FORMAT, "campaign_root": str(tmp_path / "bulk" / rb.CAMPAIGN_ID),
            "local_root": str(local), "required_free_bytes": rb.POSIX_RESERVE_BYTES + 32768,
            "runtime_prefixes": [str(local / "main_env"), str(local / "clean_env")],
            "component_bytes": components}


def test_prepares_private_aliases_and_leaves_archives_and_runtime_prefixes_untouched(storage_request):
    campaign, local = Path(storage_request["campaign_root"]), Path(storage_request["local_root"])
    archive = campaign / "g2_inputs" / "locked.tar"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"frozen archive bytes")
    result = rb.prepare_posix_storage(storage_request)
    assert result["status"] == "passed"
    assert result["modes"] == {"private_directory": 0o700, "private_file": 0o600,
                               "readonly_file": 0o444, "readonly_directory": 0o555}
    assert stat.S_IMODE(local.stat().st_mode) == 0o700
    assert (campaign / "views").readlink() == local / "views"
    assert (campaign / "g2_requests").readlink() == local / "requests"
    assert not any(Path(prefix).exists() for prefix in storage_request["runtime_prefixes"])
    assert archive.read_bytes() == b"frozen archive bytes" and not archive.is_symlink()
    assert not any(local.glob(".g2-posix-probe-*"))
    assert rb._safe_destination(str(archive), str(campaign)) == archive


def test_preparation_is_idempotent_and_preserves_existing_views_and_requests(storage_request):
    first = rb.prepare_posix_storage(storage_request)
    local = Path(storage_request["local_root"])
    view = local / "views" / "fold0"
    view.mkdir()
    data = view / "immutable.json"
    data.write_bytes(b"frozen view")
    data.chmod(0o444)
    view.chmod(0o555)
    private = local / "requests" / "temporary.json"
    private.write_bytes(b"private storage_request")
    private.chmod(0o600)
    second = rb.prepare_posix_storage(storage_request)
    assert second["links"] == first["links"]
    assert data.read_bytes() == b"frozen view" and stat.S_IMODE(data.stat().st_mode) == 0o444
    assert private.read_bytes() == b"private storage_request" and stat.S_IMODE(private.stat().st_mode) == 0o600


@pytest.mark.parametrize("replacement", ["directory", "wrong_symlink"])
def test_existing_unexpected_alias_is_preserved_and_refused(storage_request, replacement):
    campaign = Path(storage_request["campaign_root"])
    campaign.mkdir(parents=True)
    link = campaign / "views"
    if replacement == "directory":
        link.mkdir()
        (link / "keep").write_text("untouched")
    else:
        link.symlink_to(campaign.parent / "foreign")
    with pytest.raises(RuntimeError, match="unexpected target or directory"):
        rb.prepare_posix_storage(storage_request)
    assert not Path(storage_request["local_root"]).exists()
    assert (link / "keep").read_text() == "untouched" if replacement == "directory" else link.is_symlink()


@pytest.mark.parametrize("field", ["campaign_root", "local_root"])
def test_symlink_ancestor_is_refused(storage_request, tmp_path, field):
    target = tmp_path / "actual"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    if field == "campaign_root":
        storage_request[field] = str(alias / rb.CAMPAIGN_ID)
    else:
        storage_request[field] = str(alias / "posix")
        storage_request["runtime_prefixes"] = [str(Path(storage_request[field]) / name) for name in ("main_env", "clean_env")]
    with pytest.raises(RuntimeError, match="contains a symlink"):
        rb.prepare_posix_storage(storage_request)


@pytest.mark.parametrize("case", ["name", "relative", "traversal", "nested", "prefix", "duplicate",
                                  "small_reserve", "bool_size", "missing_component", "extra", "undersized"])
def test_invalid_layout_is_rejected_without_filesystem_changes(storage_request, case):
    local = Path(storage_request["local_root"])
    if case == "name":
        storage_request["campaign_root"] = str(Path(storage_request["campaign_root"]).with_name("campaign"))
    elif case == "relative":
        storage_request["local_root"] = "relative/local"
    elif case == "traversal":
        storage_request["local_root"] += "/../other"
    elif case == "nested":
        storage_request["local_root"] = storage_request["campaign_root"] + "/local"
    elif case == "prefix":
        storage_request["runtime_prefixes"][0] = str(local / "nested" / "main_env")
    elif case == "duplicate":
        storage_request["runtime_prefixes"][0] = storage_request["runtime_prefixes"][1]
    elif case == "small_reserve":
        storage_request["required_free_bytes"] = 1
    elif case == "bool_size":
        storage_request["component_bytes"]["views"] = True
    elif case == "missing_component":
        del storage_request["component_bytes"]["views"]
    elif case == "extra":
        storage_request["unrecognized"] = "not accepted"
    else:
        storage_request["required_free_bytes"] = rb.POSIX_RESERVE_BYTES + 1
    with pytest.raises(RuntimeError):
        rb.prepare_posix_storage(storage_request)
    assert not local.exists()


def test_insecure_existing_local_root_is_refused_without_chmod(storage_request):
    local = Path(storage_request["local_root"])
    local.mkdir(mode=0o755)
    with pytest.raises(RuntimeError, match="mode 0700"):
        rb.prepare_posix_storage(storage_request)
    assert stat.S_IMODE(local.stat().st_mode) == 0o755
    assert not (Path(storage_request["campaign_root"]) / "views").is_symlink()


def test_foreign_local_root_is_refused(storage_request, monkeypatch):
    local = Path(storage_request["local_root"])
    local.mkdir(mode=0o700)
    current_uid = os.getuid()
    monkeypatch.setattr(rb.os, "getuid", lambda: current_uid + 1)
    with pytest.raises(RuntimeError, match="must be owned"):
        rb.prepare_posix_storage(storage_request)


def test_non_posix_modes_are_refused_and_probe_cleaned(storage_request, monkeypatch):
    monkeypatch.setattr(Path, "chmod", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="does not enforce"):
        rb.prepare_posix_storage(storage_request)
    assert not any(Path(storage_request["local_root"]).glob(".g2-posix-probe-*"))
    assert not (Path(storage_request["campaign_root"]) / "views").is_symlink()


def test_insufficient_space_refuses_before_creating_aliases(storage_request, monkeypatch):
    monkeypatch.setattr(rb.shutil, "disk_usage", lambda path: SimpleNamespace(free=storage_request["required_free_bytes"] - 1))
    with pytest.raises(RuntimeError, match="Insufficient"):
        rb.prepare_posix_storage(storage_request)
    assert not Path(storage_request["campaign_root"]).exists()
    assert not (Path(storage_request["local_root"]) / "views").exists()


def test_retry_credits_bounded_allocated_bytes_without_following_symlinks_or_double_counting(storage_request, monkeypatch):
    rb.prepare_posix_storage(storage_request)
    local = Path(storage_request["local_root"])
    main = local / "main_env"
    main.mkdir()
    data = main / "data"
    data.write_bytes(b"x" * 16384)
    os.link(data, main / "hardlink")
    (main / "external-link").symlink_to("/unreadable-and-irrelevant")
    credited = min(storage_request["component_bytes"]["main_env"], data.stat().st_blocks * 512, data.stat().st_size)
    monkeypatch.setattr(rb.shutil, "disk_usage", lambda path: SimpleNamespace(
        free=storage_request["required_free_bytes"] - credited))
    result = rb.prepare_posix_storage(storage_request)
    assert result["credited_bytes"] == {"main_env": credited, "clean_env": 0, "views": 0}
    assert result["remaining_required_bytes"] == storage_request["required_free_bytes"] - credited


def test_sparse_files_do_not_credit_unallocated_bytes(storage_request, monkeypatch):
    rb.prepare_posix_storage(storage_request)
    main = Path(storage_request["runtime_prefixes"][0])
    main.mkdir()
    with (main / "sparse").open("wb") as handle:
        handle.truncate(1024 ** 2)
    result = rb.prepare_posix_storage(storage_request)
    allocated = min((main / "sparse").stat().st_blocks * 512, (main / "sparse").stat().st_size)
    assert result["credited_bytes"]["main_env"] == min(storage_request["component_bytes"]["main_env"], allocated)


def test_hardlinks_share_one_credit_across_components(storage_request):
    rb.prepare_posix_storage(storage_request)
    main, clean = [Path(prefix) for prefix in storage_request["runtime_prefixes"]]
    main.mkdir()
    clean.mkdir()
    data = main / "shared"
    data.write_bytes(b"x" * 4096)
    os.link(data, main / "same-component")
    os.link(data, clean / "other-component")
    result = rb.prepare_posix_storage(storage_request)
    allocated = min(data.stat().st_blocks * 512, data.stat().st_size)
    assert result["credited_bytes"]["main_env"] == allocated
    assert result["credited_bytes"]["clean_env"] == 0


def test_component_file_cannot_replace_runtime_directory(storage_request):
    local = Path(storage_request["local_root"])
    local.mkdir(mode=0o700)
    Path(storage_request["runtime_prefixes"][0]).write_bytes(b"not a directory")
    with pytest.raises(RuntimeError, match="root must be a directory"):
        rb.prepare_posix_storage(storage_request)


def test_component_foreign_filesystem_is_refused(storage_request, monkeypatch):
    rb.prepare_posix_storage(storage_request)
    main = Path(storage_request["runtime_prefixes"][0])
    main.mkdir()
    data = main / "data"
    data.write_bytes(b"data")
    original = Path.lstat

    def foreign(path):
        info = original(path)
        if path == data:
            fields = list(info)
            fields[2] += 1
            return os.stat_result(fields)
        return info

    monkeypatch.setattr(Path, "lstat", foreign)
    with pytest.raises(RuntimeError, match="foreign owner or filesystem"):
        rb.prepare_posix_storage(storage_request)


def test_cli_prepares_from_a_nonsensitive_json_request(storage_request, tmp_path, capsys):
    path = tmp_path / "storage_request.json"
    path.write_text(json.dumps(storage_request))
    assert rb.main(["prepare-posix-storage", "--request", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "passed"
