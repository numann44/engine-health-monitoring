"""Checksum-pinned common inference; labels are not accepted by the engine."""
import io
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import torch

from engine_health.common import digest, read_json
from engine_health.data import COLUMNS
from engine_health.models import RULNet, predict_net
from engine_health.preprocessing import Preprocessor, features
from engine_health.preprocessing.stress import perturb


class Predictor:
    def __init__(self, root, spec, subset):
        if spec["subset"] != subset:
            raise ValueError("Model/dataset mismatch")
        self.spec = spec
        self.models = []
        root = Path(root).resolve()
        for item in spec["assets"]:
            path = (root / item["path"]).resolve()
            if not path.is_relative_to(root) or digest(path) != item["sha256"]:
                raise ValueError("Model checksum or path mismatch")
            if spec["family"] in ("gru", "robust_gru"):
                cp = torch.load(path, map_location="cpu", weights_only=True)
                if cp["family"] != spec["family"] or cp["provenance"]["subset"] != subset:
                    raise ValueError("Checkpoint family/subset mismatch")
                model = RULNet()
                model.load_state_dict(cp["model"])
            else:
                # Only local or release-manifest checksum-pinned project artifacts.
                model = joblib.load(path)
            self.models.append(model)

    def predict(self, x):
        values = [predict_net(m, x) if self.spec["family"] in ("gru", "robust_gru")
                  else np.maximum(0, m.predict(features(x))) for m in self.models]
        result = np.mean(values, axis=0)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite prediction")
        return result


class Engine:
    def __init__(self, root, registry, subset, family=None):
        self.subset = subset
        self.root = Path(root)
        record = registry["subsets"][subset]
        family = family or record["default"]
        self.predictor = Predictor(root, record["models"][family], subset)
        prep = record["preprocessor"]
        path = self.root / prep["path"]
        if digest(path) != prep["sha256"]:
            raise ValueError("Preprocessing identity mismatch")
        self.prep = Preprocessor(read_json(path))
        self.model_id = f"{subset}:{family}:" + record["models"][family]["id"][:12]

    def inspect(self, frame, unit, cycle=None, condition="clean", seed=20261007, channel=None, magnitude=None):
        group = frame.loc[frame.unit == unit]
        if cycle is not None:
            group = group.loc[group.cycle <= cycle]
        if not len(group):
            raise ValueError("No history before requested cycle")
        started = time.perf_counter()
        window = self.prep.window_from_rows(group.iloc[:, 2:26].to_numpy())
        x = perturb(window[None], condition, [seed], self.prep.active, channel, magnitude)
        prediction = float(self.predictor.predict(x)[0])
        warnings = []
        if len(group) < 30:
            warnings.append("Short history: left padded to 30 cycles")
        missing = int(window[:, 24:45].sum())
        if missing:
            warnings.append("Missing sensor readings were causally imputed")
        return {"subset": self.subset, "unit": int(unit), "cycle": int(group.cycle.iloc[-1]),
                "rul_cycles": prediction, "model_id": self.model_id,
                "assets": [a["sha256"] for a in self.predictor.spec["assets"]],
                "history_length": min(len(group), 30), "missing_readings": missing,
                "latency_ms": (time.perf_counter()-started)*1000,
                "stress": {"condition": condition, "seed": seed, "channel": channel, "magnitude": magnitude},
                "warnings": warnings}


def parse_upload(content):
    if len(content) > 10*1024*1024:
        raise ValueError("CSV exceeds 10 MB")
    frame = pd.read_csv(io.BytesIO(content), nrows=50001)
    if len(frame) > 50000 or not len(frame):
        raise ValueError("CSV must contain 1 to 50,000 rows")
    if list(frame.columns) != COLUMNS:
        raise ValueError("Expected unit, cycle, setting_1..3, s1..s21 in that order")
    try:
        frame = frame.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("All columns must be numeric; empty sensor cells are allowed") from exc
    if not np.isfinite(frame.iloc[:, :5]).all().all() or np.isinf(frame.iloc[:, 5:]).any().any():
        raise ValueError("Unit, cycle and settings must be finite; infinity is not allowed")
    if ((frame[["unit", "cycle"]] < 1) | (frame[["unit", "cycle"]] % 1 != 0)).any().any():
        raise ValueError("Unit and cycle must be positive integers")
    for _, group in frame.groupby("unit", sort=False):
        if len(group) > 1 and not np.all(np.diff(group.cycle) == 1):
            raise ValueError("Each engine must have consecutive, increasing cycles without duplicates")
    return frame
