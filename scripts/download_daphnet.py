"""Download and verify the public UCI Daphnet dataset."""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
import zipfile
from pathlib import Path

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/245/"
    "daphnet+freezing+of+gait.zip"
)
EXPECTED_SHA256 = "27c6173b3acee75a69284903756b9011a00766c1bbb743c37fc884c5dbc39399"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    destination_resolved = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if target != destination_resolved and destination_resolved not in target.parents:
            raise ValueError(f"Unsafe path in Daphnet archive: {member.filename}")
    archive.extractall(destination)


def download_daphnet(destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / "daphnet+freezing+of+gait.zip"
    if not archive_path.exists():
        urllib.request.urlretrieve(DATASET_URL, archive_path)
    actual_hash = sha256(archive_path)
    if actual_hash != EXPECTED_SHA256:
        raise ValueError(
            "Daphnet archive checksum mismatch: "
            f"expected {EXPECTED_SHA256}, got {actual_hash}"
        )
    extraction_path = destination / "daphnet"
    if not extraction_path.exists():
        with zipfile.ZipFile(archive_path) as archive:
            _safe_extract(archive, extraction_path)
    return extraction_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    path = download_daphnet(args.destination)
    print(path)


if __name__ == "__main__":
    main()
