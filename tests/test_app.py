"""UI fixtures use synthetic engines only; official tests remain unopened."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from streamlit.testing.v1 import AppTest
import torch

from engine_health.common import atomic_json, digest
from engine_health.data import COLUMNS
from engine_health.models import RULNet
from engine_health.preprocessing import Preprocessor


def test_live_views_with_synthetic_bundle(tmp_path, monkeypatch):
    import engine_health.data
    import engine_health.inference.release
    rows = [[1, c, 1, 2, 3, *[c*.1+i*.01 for i in range(21)]] for c in range(1, 41)]
    frame = pd.DataFrame(rows, columns=COLUMNS)
    prep = Preprocessor.fit(frame, "FD001")
    atomic_json(tmp_path / "prep.json", prep.state)
    dummy = DummyRegressor(strategy="constant", constant=55).fit([[0]], [55])
    joblib.dump(dummy, tmp_path / "model.joblib")
    models = {}
    for family in ("mean", "ridge", "boost", "gru", "robust_gru"):
        if "gru" in family:
            p = tmp_path / f"{family}.pt"
            torch.save({"model": RULNet().state_dict(), "family": family, "provenance": {"subset": "FD001"}}, p)
        else:
            p = tmp_path / "model.joblib"
        models[family] = {"family": family, "subset": "FD001", "id": "synthetic-fixture",
                          "assets": [{"path": p.name, "sha256": digest(p)}]}
    registry = {"declaration_digest": "synthetic-fixture", "subsets": {"FD001": {"default": "mean", "models": models,
                 "preprocessor": {"path": "prep.json", "sha256": digest(tmp_path / "prep.json")}}}}
    atomic_json(tmp_path / "registry.json", registry)
    conditions = ("clean", "missing1", "noise025", "stuck10", "missing3", "noise050", "bias_plus", "bias_minus")
    result = {"subsets": {"FD001": {"default": "mean", "engines": 1,
             "models": {f: {"conditions": {c: {"rmse": 10., "mae": 9.} for c in conditions}} for f in models},
             "hypothesis": {"point_target_met": False}}}}
    atomic_json(tmp_path / "results.json", result)
    monkeypatch.setattr(engine_health.inference.release, "release_root", lambda root: tmp_path)
    monkeypatch.setattr(engine_health.data, "download", lambda root: None)
    monkeypatch.setattr(engine_health.data, "load_table", lambda *args: frame)
    monkeypatch.setattr(np, "loadtxt", lambda *args, **kwargs: np.array([15.]))
    app = Path(__file__).resolve().parents[1] / "app/streamlit_app.py"
    at = AppTest.from_file(str(app), default_timeout=20).run()
    assert not at.exception
    assert at.metric[0].value == "55.0"
    assert len(at.tabs) == 3
    assert len(at.get("download_button")) == 4
    at.slider[0].set_value(10).run()
    assert not at.exception
    assert at.metric[1].value == "45"
    condition = next(x for x in at.selectbox if x.label == "Measurement condition")
    condition.select("stuck10").run()
    assert not at.exception
    at.sidebar.radio[0].set_value("Upload CSV").run()
    assert not at.exception
    assert len(at.get("file_uploader")) == 1
