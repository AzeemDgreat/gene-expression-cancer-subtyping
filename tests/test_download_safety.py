import importlib.util
import io
import tarfile
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_data.py"


@pytest.fixture(scope="module")
def dl():
    spec = importlib.util.spec_from_file_location("download_data", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_zip_path_traversal_is_rejected(dl, tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.txt", "x")
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(RuntimeError, match="unsafe"):
        dl.safe_extract_zip(archive, dest)
    assert not (tmp_path / "escape.txt").exists()


def test_tar_path_traversal_is_rejected(dl, tmp_path):
    archive = tmp_path / "evil.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        info = tarfile.TarInfo("../escape.txt")
        info.size = 1
        tf.addfile(info, io.BytesIO(b"x"))
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(RuntimeError, match="unsafe"):
        dl.safe_extract_tar(archive, dest)
    assert not (tmp_path / "escape.txt").exists()


def test_tar_symlinks_are_rejected(dl, tmp_path):
    archive = tmp_path / "link.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tf.addfile(info)
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(RuntimeError, match="unsupported"):
        dl.safe_extract_tar(archive, dest)


def test_nested_archive_files_are_found(dl, tmp_path):
    inner = tmp_path / "inner.tar.gz"
    payload = tmp_path / "payload"
    (payload / "TCGA-PANCAN").mkdir(parents=True)
    (payload / "TCGA-PANCAN" / "data.csv").write_text("a\n")
    (payload / "TCGA-PANCAN" / "labels.csv").write_text("b\n")
    with tarfile.open(inner, "w:gz") as tf:
        tf.add(payload / "TCGA-PANCAN", arcname="TCGA-PANCAN")
    outer = tmp_path / "outer.zip"
    with zipfile.ZipFile(outer, "w") as zf:
        zf.write(inner, "inner.tar.gz")

    extracted = tmp_path / "extracted"
    extracted.mkdir()
    dl.safe_extract_zip(outer, extracted)
    dl.safe_extract_tar(next(extracted.rglob("*.tar.gz")), extracted)
    assert dl.find_one(extracted, "data.csv").read_text() == "a\n"
    assert dl.find_one(extracted, "labels.csv").read_text() == "b\n"


def test_find_one_reports_missing_file(dl, tmp_path):
    with pytest.raises(FileNotFoundError):
        dl.find_one(tmp_path, "data.csv")
