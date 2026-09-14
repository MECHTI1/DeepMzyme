import sys
import tarfile
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from build_colab_bundle import build_bundle


@pytest.mark.parametrize("suffix", [".tar.gz", ".tgz", ".tar"])
def test_bundle_round_trip_preserves_alias(tmp_path, suffix):
    data = tmp_path / "DeepMzyme_Data"
    shared = data / "shared"
    shared.mkdir(parents=True)
    (shared / "example.csv").write_text("structure,label\nexample,ZN\n")
    alias = data / "CLEAN_30_main"
    alias.symlink_to("shared", target_is_directory=True)
    output = tmp_path / ("bundle" + suffix)
    with patch("build_colab_bundle.PROJECT_ROOT", tmp_path):
        build_bundle([shared, alias], output_bundle=output)
    if suffix != ".tar":
        assert output.read_bytes()[:2] == b"\x1f\x8b"
    with tarfile.open(output) as archive:
        assert archive.getmember("DeepMzyme_Data/CLEAN_30_main").issym()
        archive.extractall(tmp_path / "extracted", filter="data")
    assert (tmp_path / "extracted/DeepMzyme_Data/CLEAN_30_main/example.csv").read_text().endswith("example,ZN\n")


def test_unknown_archive_suffix_fails_before_writing(tmp_path):
    output = tmp_path / "bundle.zip"
    with pytest.raises(ValueError, match="Bundle output"):
        build_bundle([], output_bundle=output)
    assert not output.exists()
