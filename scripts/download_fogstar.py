"""Download and verify the public FoG-STAR v3.0 files from Zenodo."""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

RECORD_URL = "https://zenodo.org/records/17838806/files"
FILES = {
    "sensor_data.csv": "952a37ab147da35e6d4e7a1e9bac44cb",
    "clinical_data.csv": "8b9a2339c8fcb8eaa470bcb1784d31c5",
    "README.txt": "5b424641b0fcf2780037c0e9e838ae10",
}


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_fogstar(destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    for filename, expected_hash in FILES.items():
        path = destination / filename
        if not path.exists():
            urllib.request.urlretrieve(f"{RECORD_URL}/{filename}?download=1", path)
        actual_hash = md5(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"FoG-STAR checksum mismatch for {filename}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/raw/fogstar"),
    )
    args = parser.parse_args()
    print(download_fogstar(args.destination))


if __name__ == "__main__":
    main()
