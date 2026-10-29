"""Download every manifest-locked public XLSX without decoding any target values."""

import argparse
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("config/stanford_files_v1.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text("utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def fetch(entry):
        target = args.output_dir / Path(entry["path"]).name
        if target.exists():
            data = target.read_bytes()
        else:
            url = f"https://raw.githubusercontent.com/stanfordnmbl/imu-fog-detection/{manifest['commit']}/{entry['path']}"
            request = urllib.request.Request(url, headers={"User-Agent": "FoGGuard-fixed-external-audit/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read(entry["size"] + 1)
        digest = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if len(data) != entry["size"] or digest != entry["git_blob_sha1"]:
            raise ValueError(f"Workbook changed: {entry['path']}")
        if not target.exists():
            target.write_bytes(data)
        print(f"Verified {target.name}", flush=True)
        return {**entry, "sha256": hashlib.sha256(data).hexdigest()}

    with ThreadPoolExecutor(max_workers=4) as pool:
        checked = list(pool.map(fetch, manifest["files"]))
    (args.output_dir / "DOWNLOAD_COMPLETE.json").write_text(json.dumps({"commit": manifest["commit"], "files": checked,
        "target_labels_decoded": False}, indent=2), encoding="utf-8")
    print(f"All {len(checked)} fixed workbooks verified; no label values decoded", flush=True)


if __name__ == "__main__":
    main()
