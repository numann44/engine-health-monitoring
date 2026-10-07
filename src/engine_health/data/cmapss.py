import hashlib
import io
import math
from pathlib import Path
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from engine_health.common import atomic_json, digest, identity, now, read_json

SUBSETS = ("FD001", "FD002", "FD003", "FD004")
COLUMNS = ["unit", "cycle"] + [f"setting_{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)]
PRIMARY = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
MIRROR = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
ARCHIVE_SHA = "74bef434a34db25c7bf72e668ea4cd52afe5f2cf8e44367c55a82bfd91a5a34f"
COUNTS = {"FD001": (100, 100), "FD002": (260, 259), "FD003": (100, 100), "FD004": (249, 248)}
HASHES = {
    "train_FD001.txt": "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8",
    "test_FD001.txt": "3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851",
    "RUL_FD001.txt": "a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca",
    "train_FD002.txt": "dac6c4dbc4e7c1bdeb5747da3d313d05c395bb99801b44a002b26a2ba13d788f",
    "test_FD002.txt": "de7b5bf7e998a985c378488480528b7c02cff1406a46740def362dda8d9b4e02",
    "RUL_FD002.txt": "c851dd96a6ea6998d3c4a8f834d3c8013aa90e93a6ed950dc826ad0655b2906b",
    "train_FD003.txt": "2abbe9968cc5e8eb091980f51b20f62bb4127336d3482cb52071d53bf23329e2",
    "test_FD003.txt": "299babd63c8d987cef079c4a425429f33b3a34797d803bbe2ad48c29dbd0d790",
    "RUL_FD003.txt": "df1e0566306b174a2de41c67a3e7a51877889598b78643fc3e5685259091b7cb",
    "train_FD004.txt": "27ef6160b6a1dcb2613a88de9c239f763b223f02cdc41dc5cdedc5dc189b6218",
    "test_FD004.txt": "1dc675fff0624bac10786927c6715b37d1297657137400d2b1a3138d777a3ba5",
    "RUL_FD004.txt": "196b836b85a95ac7fdbbf29c5fdf1657382eafa445644d114ffaaf50dc2975e1",
}


def download(root):
    root = Path(root)
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    if all((raw / n).exists() for n in HASHES):
        for n, h in HASHES.items():
            if digest(raw / n) != h:
                raise ValueError(f"Existing data checksum mismatch: {n}")
        return read_json(root / "download.json")
    errors = []
    for url in (PRIMARY, MIRROR):
        try:
            with urllib.request.urlopen(url, timeout=90) as response:
                content = response.read(30_000_001)
            if len(content) > 30_000_000:
                raise ValueError("Archive exceeds declared size limit")
            archive = zipfile.ZipFile(io.BytesIO(content))
            if url == MIRROR:
                name = next(n for n in archive.namelist() if n.endswith("CMAPSSData.zip"))
                content = archive.read(name)
                archive = zipfile.ZipFile(io.BytesIO(content))
            if hashlib.sha256(content).hexdigest() != ARCHIVE_SHA:
                raise ValueError("Archive identity changed; stop and audit source")
            if archive.testzip() is not None:
                raise ValueError("ZIP CRC failure")
            for n, h in HASHES.items():
                value = archive.read(n)
                if hashlib.sha256(value).hexdigest() != h:
                    raise ValueError(f"File identity mismatch: {n}")
            for n in HASHES:
                (raw / n).write_bytes(archive.read(n))
            (raw / "readme.txt").write_bytes(archive.read("readme.txt"))
            record = {"source": url, "retrieved": now(), "sha256": ARCHIVE_SHA,
                      "files": HASHES, "license": "License not specified by source catalog"}
            atomic_json(root / "download.json", record)
            return record
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
    raise RuntimeError("No verified data source available: " + "; ".join(errors))


def load_table(root, subset, part="train"):
    if subset not in SUBSETS or part not in ("train", "test"):
        raise ValueError("Unknown dataset or partition")
    path = Path(root) / "raw" / f"{part}_{subset}.txt"
    if digest(path) != HASHES[path.name]:
        raise ValueError(f"Changed data: {path.name}")
    frame = pd.read_csv(path, sep=r"\s+", header=None)
    if frame.shape[1] != 26:
        raise ValueError("Expected 26 numeric columns")
    frame.columns = COLUMNS
    return frame


def audit(root):
    root = Path(root)
    counts = {}
    seen = {}
    duplicates = []
    for subset in SUBSETS:
        entry = {}
        for part, count in zip(("train", "test"), COUNTS[subset]):
            df = load_table(root, subset, part)
            if not np.isfinite(df.to_numpy()).all() or df.duplicated().any():
                raise ValueError("Nonfinite or duplicate row")
            if not np.equal(df[["unit", "cycle"]], np.floor(df[["unit", "cycle"]])).all().all():
                raise ValueError("Noninteger unit/cycle")
            units = sorted(df.unit.unique())
            if units != list(range(1, count + 1)):
                raise ValueError("Unexpected motor IDs")
            for unit, group in df.groupby("unit", sort=True):
                if list(group.cycle) != list(range(1, len(group) + 1)):
                    raise ValueError("Cycles must be ordered, unique and consecutive")
                key = f"{subset}:{part}:{int(unit)}"
                h = hashlib.sha256(group.drop(columns="unit").to_numpy(dtype="<f8").tobytes()).hexdigest()
                if h in seen:
                    duplicates.append([seen[h], key])
                seen[h] = key
            entry[part] = {"engines": count, "rows": len(df)}
        path = root / "raw" / f"RUL_{subset}.txt"
        if digest(path) != HASHES[path.name]:
            raise ValueError("RUL source identity mismatch")
        labels = np.loadtxt(path, ndmin=1)
        if labels.shape != (COUNTS[subset][1],) or not np.isfinite(labels).all() or (labels < 0).any():
            raise ValueError("Invalid RUL alignment")
        entry["rul_rows"] = len(labels)
        counts[subset] = entry
    if duplicates:
        raise ValueError(f"Duplicate complete trajectories require group review: {duplicates}")
    result = {"created": now(), "counts": counts, "hashes": HASHES,
              "duplicates": duplicates, "test_access": "structural only; no model outcomes",
              "fd004_note": "Archive contains 249 train / 248 test; source prose reverses them"}
    atomic_json(root / "audit.json", result)
    return result


def prepare(root, output):
    """Split complete engines before any windowing or normalization."""
    root, output = Path(root), Path(output)
    if output.exists():
        record = read_json(output)
        if record["digest"] != identity({k: v for k, v in record.items() if k != "digest"}):
            raise ValueError("Partition manifest changed")
        if record["files"] != HASHES:
            raise ValueError("Partition data mismatch")
        return record
    audit(root)
    subsets = {}
    for subset in SUBSETS:
        df = load_table(root, subset)
        ids = [int(x) for x in sorted(df.unit.unique())]
        rng = np.random.default_rng(20261007)
        rng.shuffle(ids)
        n = math.floor(.8 * len(ids))
        train, val = sorted(ids[:n]), sorted(ids[n:])
        bank = []
        for unit in val:
            length = int(df.loc[df.unit == unit, "cycle"].max())
            for fraction in (.25, .5, .75, .9):
                cycle = max(1, math.floor(length * fraction))
                key = f"{subset}:train:{unit}:{cycle}"
                seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
                bank.append({"unit": unit, "cycle": cycle, "seed": seed})
        subsets[subset] = {"train": train, "validation": val, "bank": bank}
    record = {"schema": 1, "split_seed": 20261007, "files": HASHES, "subsets": subsets}
    record["digest"] = identity(record)
    atomic_json(output, record)
    return record
