"""Inspect fixed public workbook headers/timestamps only; never read target labels."""

import argparse
import hashlib
import json
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from itertools import pairwise
from pathlib import Path

FILES = {
    "data/imus6_subjects7_configs.xlsx": "ef95834cd4ab7a54401936b76604b63659ead65e",
    "data/raw/imus6_subjects7/pt1_visit_24_tbc_walklr_0_trial_2.xlsx": "a3f5dfa9862c4ab35815305c4346c84accac9291",
}
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def inspect(path):
    with zipfile.ZipFile(path) as book:
        strings = []
        if "xl/sharedStrings.xml" in book.namelist():
            strings = ["".join(x.itertext()) for x in ET.fromstring(book.read("xl/sharedStrings.xml")).findall("s:si", NS)]
        with book.open("xl/worksheets/sheet1.xml") as sheet:
            headers, times = [], []
            for _, row in ET.iterparse(sheet, events=("end",)):
                if row.tag != "{" + NS["s"] + "}row":
                    continue
                if not headers:
                    for cell in row.findall("s:c", NS):
                        v = cell.findtext("s:v", default="", namespaces=NS)
                        value = strings[int(v)] if cell.get("t") == "s" else "".join(cell.find("s:is", NS).itertext()) if cell.get("t") == "inlineStr" else v
                        headers.append({"column": cell.get("r"), "name": value})
                else:
                    # Column B is time in the source README; do not decode any other data cell.
                    cell = next((c for c in row.findall("s:c", NS) if c.get("r", "").rstrip("0123456789") == "B"), None)
                    if cell is not None:
                        value = cell.findtext("s:v", default="", namespaces=NS)
                        try:
                            times.append(float(value))
                        except ValueError:
                            pass
                    if len(times) >= 129:
                        break
                row.clear()
        differences = [b - a for a, b in pairwise(times)]
        return {"headers": headers, "timestamp_intervals_inspected": len(differences),
            "median_dt_s": sorted(differences)[len(differences) // 2] if differences else None,
            "target_label_values_read": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Use a fresh schema-only directory")
    args.output_dir.mkdir(parents=True)
    results = {}
    for relative, expected in FILES.items():
        url = "https://raw.githubusercontent.com/stanfordnmbl/imu-fog-detection/main/" + relative
        request = urllib.request.Request(url, headers={"User-Agent": "FoGGuard-schema-audit/1.0"})
        with urllib.request.urlopen(request, timeout=45) as response:
            content = response.read(20_000_001)
        if len(content) > 20_000_000 or hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest() != expected:
            raise ValueError("Source workbook does not match fixed Git blob or size limit")
        path = args.output_dir / Path(relative).name
        path.write_bytes(content)
        results[relative] = {"url": url, "git_blob_sha1": expected, "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content), **inspect(path)}
    summary = {"scope": "Schema/timestamps only, not external model evaluation", "source_repository": "https://github.com/stanfordnmbl/imu-fog-detection",
        "frozen_model_scored": False, "target_label_values_read": False, "files": results}
    (args.output_dir / "schema.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
