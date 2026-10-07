"""Tiny real-data training -> safe load -> inference on Linux. No test outcomes."""
from pathlib import Path
import tempfile

import numpy as np

from engine_health.common import digest, read_json
from engine_health.inference import Predictor
from engine_health.training.study import subset_data
from engine_health.training.train import run_gru

root = Path(__file__).resolve().parents[1]
prep, train, validation = subset_data(root, "FD001", read_json(root / "protocols/study_v1/splits.json"))
train = tuple(x[:128] for x in train)
validation = tuple(x[:8] for x in validation)
config = {"epochs": 1, "patience": 1, "lr": .001, "weight_decay": .0001,
          "batch": 64, "gradient_clip": 1., "huber_delta": .1}
with tempfile.TemporaryDirectory() as temp:
    run = Path(temp) / "run"
    run_gru(run, train, validation, prep.active, config, {"subset": "FD001"}, "gru", 42)
    spec = {"subset": "FD001", "family": "gru", "assets": [{"path": "run/selected.pt", "sha256": digest(run / "selected.pt")}]}
    prediction = Predictor(Path(temp), spec, "FD001").predict(validation[0])
    assert prediction.shape == (8,) and np.isfinite(prediction).all() and (prediction >= 0).all()
print("Real-data smoke passed; no official test outcomes read")
