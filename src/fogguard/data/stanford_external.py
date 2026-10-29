"""Fixed XLSX channel reader and causal 128-to-60Hz adapter, without target fitting."""

import re
import xml.etree.ElementTree as ET
import zipfile
from itertools import pairwise

import numpy as np
import pandas as pd
from scipy.signal import firwin, upfirdn

from fogguard.modeling.a1_bilateral import DUAL

NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
SOURCE_GYRO = tuple(f"imu_ankle_{side}_g{axis}" for side in ("l", "r") for axis in "xyz")
FILTER = firwin(641, 1 / 32, window=("kaiser", 5.0)) * 15
DELAY_S = 1 / 6


def source_identity(name):
    match = re.fullmatch(r"pt(\d+)_visit_([^_]+)_.+\.xlsx", name)
    if not match:
        raise ValueError("Unexpected source filename")
    return int(match[1]), match[2]


def read_workbook(path, *, labels=False):
    """Only decode requested data cells; freeze_label never decoded in sensor mode."""
    wanted = (["subject_ID", "time", *SOURCE_GYRO, "freeze_label"] if labels == "both" else
              ["subject_ID", "time", "freeze_label"] if labels else ["subject_ID", "time", *SOURCE_GYRO])
    values, header = {name: [] for name in wanted}, None
    with zipfile.ZipFile(path) as book:
        strings = []
        if "xl/sharedStrings.xml" in book.namelist():
            strings = ["".join(n.itertext()) for n in ET.fromstring(book.read("xl/sharedStrings.xml")).findall("s:si", NS)]

        def value(cell):
            if cell.get("t") == "inlineStr":
                return "".join(cell.find("s:is", NS).itertext())
            raw = cell.findtext("s:v", default="", namespaces=NS)
            return strings[int(raw)] if cell.get("t") == "s" else raw

        with book.open("xl/worksheets/sheet1.xml") as sheet:
            for _, row in ET.iterparse(sheet, events=("end",)):
                if row.tag != "{" + NS["s"] + "}row":
                    continue
                if header is None:
                    header = {c.get("r").rstrip("0123456789"): value(c) for c in row.findall("s:c", NS)}
                    if not set(wanted) <= set(header.values()):
                        raise ValueError(f"Missing required columns in {path.name}")
                else:
                    result = {name: np.nan for name in wanted}
                    for cell in row.findall("s:c", NS):
                        name = header.get(cell.get("r").rstrip("0123456789"))
                        if name in result:
                            try:
                                result[name] = float(value(cell))
                            except ValueError:
                                pass
                    for name in wanted:
                        values[name].append(result[name])
                row.clear()
    result = {k: np.asarray(v, float) for k, v in values.items()}
    if not len(result["time"]):
        raise ValueError("Empty source workbook")
    return result


def canonical_clock(time):
    time = np.asarray(time, float)
    if not len(time) or not np.isfinite(time).all() or (np.diff(time) <= 0).any():
        raise ValueError("Source clock must be finite and strictly increasing")
    cuts = np.r_[0, np.flatnonzero(np.diff(time) > 1.5 / 128) + 1, len(time)]
    canonical, blocks = time.copy(), []
    for a, b in pairwise(cuts):
        expected = time[a] + np.arange(b - a) / 128
        if len(expected) > 1 and np.max(abs(np.diff(time[a:b]) - 1 / 128)) > 2e-6:
            raise ValueError("Source clock is not 128Hz within fixed per-step quantization tolerance")
        canonical[a:b] = expected
        blocks.append((int(a), int(b)))
    return canonical, blocks


def resample_gyro(time, gyro_rad):
    """Causal FIR, fully warmed, no synthetic tail. NaNs propagate through FIR."""
    time, blocks = canonical_clock(time)
    gyro_rad = np.asarray(gyro_rad, float)
    if gyro_rad.shape != (len(time), 6):
        raise ValueError("Expected six ordered ankle gyro channels")
    outputs, timestamps = [], []
    for a, b in blocks:
        last = ((b - a - 1) * 15) // 32
        if last < 20:
            continue
        k = np.arange(20, last + 1)
        filtered = upfirdn(FILTER, np.rad2deg(gyro_rad[a:b]), up=15, down=32, axis=0)
        outputs.append(filtered[k])
        timestamps.append(time[a] + k / 60 - DELAY_S)
    if not outputs:
        return pd.DataFrame(columns=["timestamp", *DUAL]), time, blocks
    frame = pd.DataFrame(np.vstack(outputs), columns=DUAL)
    frame.insert(0, "timestamp", np.concatenate(timestamps))
    return frame, time, blocks


def project_labels(source_time, labels, logical_times):
    source_time, _ = canonical_clock(source_time)
    labels = np.asarray(labels, float)
    if labels.shape != source_time.shape or not np.isin(labels, [0, 1]).all():
        raise ValueError("Expected finite binary source labels")
    owner = np.searchsorted(source_time, np.asarray(logical_times) + 1e-9, side="right") - 1
    if (owner < 0).any() or (np.asarray(logical_times) - source_time[owner] >= 1 / 128 + 1e-6).any():
        raise ValueError("Target label projection crosses a missing source cell")
    return labels[owner].astype(int)


def native_intervals(time, active):
    """Native128Hz truth/observation runs, preserving clock gaps."""
    time = np.asarray(time, float)
    indices = np.flatnonzero(active)
    if not len(indices):
        return []
    cuts = np.flatnonzero((np.diff(indices) != 1) | (np.diff(time[indices]) > 1.5 / 128)) + 1
    return [(float(time[p[0]]), float(time[p[-1]] + 1 / 128)) for p in np.split(indices, cuts)]
