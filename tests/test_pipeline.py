from copy import deepcopy
import json

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.dummy import DummyRegressor

from engine_health.common import atomic_json, digest, heavy_lock, read_json
from engine_health.data import COLUMNS
from engine_health.evaluation.metrics import bootstrap, metrics, selection_score
from engine_health.inference import Engine, Predictor, parse_upload
from engine_health.models import RULNet
from engine_health.preprocessing import Preprocessor, features, fill_missing, make_windows
from engine_health.preprocessing.stress import CONDITIONS, augment, perturb
from engine_health.training.train import equal_engine_weights, run_gru


@pytest.fixture
def frame():
    rng = np.random.default_rng(6)
    rows = []
    for unit in range(1, 7):
        for cycle in range(1, 41+unit):
            rows.append([unit, cycle, 1, 2, 3, *rng.normal(cycle*.02, .1, 21)])
    return pd.DataFrame(rows, columns=COLUMNS)


@pytest.fixture
def setup(frame):
    tr = frame.loc[frame.unit <= 4]
    prep = Preprocessor.fit(tr, "FD001")
    train = make_windows(tr, prep)
    bank = [{"unit": u, "cycle": c} for u in (5, 6) for c in (10, 20, 30)]
    validation = list(make_windows(frame.loc[frame.unit > 4], prep, bank))
    validation[3] = np.arange(6)+100
    return prep, train, tuple(validation)


def test_targets_and_no_future(frame, setup):
    prep, train, _ = setup
    x, y, unit, cycles = train
    ix = np.flatnonzero((unit == 1) & (cycles == 10))[0]
    assert y[ix] == 31
    original = prep.window_from_rows(frame.loc[(frame.unit == 1) & (frame.cycle <= 10)].iloc[:, 2:].to_numpy())
    np.testing.assert_array_equal(x[ix], original)
    changed = frame.copy()
    changed.loc[changed.cycle > 10, "s1"] = 1e6
    future = prep.window_from_rows(changed.loc[(changed.unit == 1) & (changed.cycle <= 10)].iloc[:, 2:].to_numpy())
    np.testing.assert_array_equal(original, future)
    assert original[:, 45].sum() == 10
    assert np.all(original[:20] == 0)


def test_normalizer_train_only(frame):
    fitted = Preprocessor.fit(frame.loc[frame.unit <= 4], "FD001")
    frame.loc[frame.unit > 4, "s1"] = -1e9
    second = Preprocessor.fit(frame.loc[frame.unit <= 4], "FD001")
    assert fitted.state == second.state
    assert fitted.state["fit_units"] == [1, 2, 3, 4]


def test_conditions_and_constant_channels(frame):
    frame["s1"] = 3.
    frame.loc[:, "setting_1"] = (frame.cycle % 6).astype(float)
    prep = Preprocessor.fit(frame, "FD002")
    assert len(prep.state["centers"]) == 6
    assert 0 not in prep.active
    assert np.isfinite(prep.window_from_rows(frame.iloc[:15, 2:].to_numpy())).all()


def test_causal_fill():
    x = np.zeros((1, 4, 46), dtype=np.float32)
    x[:, :, 45] = 1
    x[0, :, 0] = [np.nan, 2, np.nan, 9]
    got = fill_missing(x)
    np.testing.assert_array_equal(got[0, :, 0], [0, 2, 2, 9])
    np.testing.assert_array_equal(got[0, :, 24], [1, 0, 1, 0])


@pytest.mark.parametrize("condition", CONDITIONS)
def test_corruption_is_fixed_and_keeps_padding(setup, condition):
    prep, _, val = setup
    before = val[0].copy()
    a = perturb(before, condition, val[3], prep.active)
    b = perturb(before, condition, val[3], prep.active)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(val[0], before)
    np.testing.assert_array_equal(a[before[:, :, 45] == 0], before[before[:, :, 45] == 0])
    assert np.isfinite(a).all()
    if condition in ("stuck10", "noise025", "noise050", "bias_plus", "bias_minus"):
        np.testing.assert_array_equal(a[:, :, 24:45], before[:, :, 24:45])


def test_missing_channel_not_recovered_from_hidden_values(setup):
    prep, _, val = setup
    a = perturb(val[0], "missing1", val[3], prep.active, channel=0)
    changed = val[0].copy()
    changed[:, :, 0] += 900
    b = perturb(changed, "missing1", val[3], prep.active, channel=0)
    np.testing.assert_array_equal(a, b)


def test_features_exclude_padding(setup):
    prep, _, val = setup
    x = val[0][:1].copy()
    f = features(x)
    x[:, :20, :45] = 100
    np.testing.assert_allclose(features(x), f)
    assert f.shape == (1, 180)


def test_equal_engine_contribution():
    units = np.array([1, 1, 2, 2, 2, 2])
    w = equal_engine_weights(units)
    np.testing.assert_allclose(w[units == 1].sum(), w[units == 2].sum())


def test_metrics_and_selection():
    m = metrics([0, 10, 20], [0, 13, 44])
    assert m["mae"] == 9
    assert m["over20_count"] == 1
    predictions = {c: np.array([0, 13, 44]) for c in CONDITIONS[:4]}
    assert selection_score(np.array([0, 10, 20]), predictions) == m["rmse"]
    result = bootstrap([1, 2, 3], [2, 3, 4], [1, 1, 2], [2, 3, 4], repetitions=20)
    assert result["ci95"] == [0, 0]
    assert result["engines"] == 2


@pytest.mark.parametrize("family", ["gru", "robust_gru"])
def test_resume_exact_and_provenance(tmp_path, setup, family):
    prep, train, val = setup
    config = {"epochs": 3, "patience": 15, "lr": .001, "weight_decay": .0001,
              "batch": 64, "gradient_clip": 1., "huber_delta": .1}
    provenance = {"subset": "FD001", "split": "fixed"}
    a, b = tmp_path / "continuous", tmp_path / "resumed"
    run_gru(a, train, val, prep.active, config, provenance, family, 42)
    run_gru(b, train, val, prep.active, config, provenance, family, 42, stop_after=1)
    run_gru(b, train, val, prep.active, config, provenance, family, 42, resume=True)
    ac = torch.load(a / "last.pt", weights_only=True)
    bc = torch.load(b / "last.pt", weights_only=True)
    for key in ac["model"]:
        assert torch.equal(ac["model"][key], bc["model"][key])
    assert ac["best"] == bc["best"]
    assert ac["rng_augmentation"] == bc["rng_augmentation"]
    with pytest.raises(ValueError, match="provenance"):
        run_gru(b, train, val, prep.active, config, {"subset": "FD002"}, family, 42, resume=True)
    with pytest.raises(FileExistsError):
        run_gru(a, train, val, prep.active, config, provenance, family, 42)


def test_augmentation_rng_isolation(setup):
    prep, train, _ = setup
    torch.manual_seed(42)
    before = torch.get_rng_state().clone()
    augment(train[0][:20], np.random.default_rng(42), prep.active)
    assert torch.equal(before, torch.get_rng_state())


def test_checkpoint_and_engine_parity(tmp_path, frame, setup):
    prep, train, _ = setup
    atomic_json(tmp_path / "prep.json", prep.state)
    model = DummyRegressor(strategy="constant", constant=55).fit(features(train[0]), train[1])
    joblib.dump(model, tmp_path / "model.joblib")
    spec = {"subset": "FD001", "family": "mean", "id": "abc123",
            "assets": [{"path": "model.joblib", "sha256": digest(tmp_path / "model.joblib")}]}
    registry = {"subsets": {"FD001": {"default": "mean", "models": {"mean": spec},
                 "preprocessor": {"path": "prep.json", "sha256": digest(tmp_path / "prep.json")}}}}
    engine = Engine(tmp_path, registry, "FD001")
    result = engine.inspect(frame, 1, 10)
    assert result["rul_cycles"] == 55
    assert result["cycle"] == 10 and result["history_length"] == 10
    changed = frame.copy()
    changed.loc[changed.cycle > 10, "s1"] = 1e6
    assert engine.inspect(changed, 1, 10)["rul_cycles"] == 55
    with pytest.raises(ValueError, match="mismatch"):
        Predictor(tmp_path, spec, "FD002")
    bad = deepcopy(spec)
    bad["assets"][0]["sha256"] = "incorrect"
    with pytest.raises(ValueError, match="checksum"):
        Predictor(tmp_path, bad, "FD001")
    serialized = json.loads(json.dumps(result))
    assert serialized["rul_cycles"] == result["rul_cycles"]


def test_upload(frame):
    content = frame.to_csv(index=False).encode()
    result = parse_upload(content)
    assert len(result) == len(frame)
    frame.loc[0, "s1"] = np.nan
    assert np.isnan(parse_upload(frame.to_csv(index=False).encode()).s1.iloc[0])


@pytest.mark.parametrize("issue", ["columns", "infinite", "setting", "duplicate", "order", "gap", "size", "rows"])
def test_upload_errors(frame, issue):
    if issue == "columns":
        frame = frame.drop(columns="s1")
    elif issue == "infinite":
        frame.loc[0, "s1"] = np.inf
    elif issue == "setting":
        frame.loc[0, "setting_1"] = np.nan
    elif issue == "duplicate":
        frame.loc[1, "cycle"] = 1
    elif issue == "order":
        frame = frame.iloc[::-1]
    elif issue == "gap":
        frame = frame.drop(index=1)
    elif issue == "rows":
        frame = pd.concat([frame]*250)
    content = b"x"*(10*1024*1024+1) if issue == "size" else frame.to_csv(index=False).encode()
    with pytest.raises(ValueError):
        parse_upload(content)


def test_atomic_json_and_exclusive_lock(tmp_path):
    path = tmp_path / "record.json"
    atomic_json(path, {"a": 1})
    assert read_json(path) == {"a": 1}
    with heavy_lock(tmp_path / "lease"):
        with pytest.raises(RuntimeError):
            with heavy_lock(tmp_path / "lease"):
                pass


def test_network_output_is_nonnegative():
    torch.manual_seed(1)
    m = RULNet().eval()
    with torch.inference_mode():
        assert torch.all(m(torch.zeros(3, 30, 46)) >= 0)
