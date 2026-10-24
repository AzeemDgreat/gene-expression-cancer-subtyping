"""Download the UCI PANCAN RNA-Seq dataset and place data.csv and labels.csv in data/.

Source: https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq
License: CC BY 4.0. Weinstein et al., The Cancer Genome Atlas Pan-Cancer analysis project,
Nature Genetics 45, 1113-1120 (2013). DOI of the UCI record: 10.24432/C5R88H.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gexpr.data import load_dataset  # noqa: E402

URL = "https://archive.ics.uci.edu/static/public/401/gene+expression+cancer+rna+seq.zip"


def _inside(base: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def safe_extract_zip(archive: Path, dest: Path) -> None:
    """Extract a zip, refusing any member that would land outside ``dest``."""
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            if not _inside(dest, dest / member.filename):
                raise RuntimeError(f"unsafe path in archive: {member.filename}")
        zf.extractall(dest)


def safe_extract_tar(archive: Path, dest: Path) -> None:
    """Extract a tar, allowing only regular files and directories inside ``dest``."""
    with tarfile.open(archive) as tf:
        members = []
        for member in tf.getmembers():
            if not _inside(dest, dest / member.name):
                raise RuntimeError(f"unsafe path in archive: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise RuntimeError(f"unsupported entry type in archive: {member.name}")
            members.append(member)
        # Our own checks above already reject traversal and links; the "data" filter (where
        # available) adds a second layer and silences the Python 3.14 default-change warning.
        if hasattr(tarfile, "data_filter"):
            tf.extractall(dest, members=members, filter="data")
        else:
            tf.extractall(dest, members=members)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_one(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name))
    if not matches:
        raise FileNotFoundError(f"{name} not found after extraction under {root}")
    return matches[0]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="data", help="where to place data.csv and labels.csv")
    parser.add_argument("--url", default=URL)
    parser.add_argument("--force", action="store_true", help="download again even if present")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if (data_dir / "data.csv").exists() and (data_dir / "labels.csv").exists() and not args.force:
        print(f"Data already present in {data_dir}/. Use --force to download again.")
    else:
        raw = data_dir / "raw"
        if raw.exists():
            shutil.rmtree(raw)
        raw.mkdir()
        archive = raw / "download.zip"
        print(f"Downloading {args.url}")
        urllib.request.urlretrieve(args.url, archive)  # noqa: S310 (fixed https URL)
        print(f"  saved {archive.stat().st_size / 1e6:.1f} MB, sha256 {sha256(archive)}")

        extracted = raw / "extracted"
        extracted.mkdir()
        safe_extract_zip(archive, extracted)
        # The UCI zip may wrap the files in a tar.gz.
        for inner in sorted(extracted.rglob("*.tar.gz")) + sorted(extracted.rglob("*.tgz")):
            safe_extract_tar(inner, extracted)

        for name in ("data.csv", "labels.csv"):
            shutil.copy(find_one(extracted, name), data_dir / name)
        shutil.rmtree(raw)

    X, y = load_dataset(data_dir)
    print(f"OK: {X.shape[0]} samples x {X.shape[1]} genes")
    print("Class counts:", y.value_counts().sort_index().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
