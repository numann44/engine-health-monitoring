import numpy as np
from engine_health.preprocessing.stress import MILD


def metrics(y, prediction):
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    if y.shape != prediction.shape or not len(y) or not np.isfinite(y+prediction).all():
        raise ValueError("Invalid prediction/label alignment")
    error = prediction-y
    over = error[error > 0]
    return {"n": len(y), "rmse": float(np.sqrt(np.mean(error**2))),
            "mae": float(np.mean(np.abs(error))), "bias": float(error.mean()),
            "overestimate_count": int((error > 0).sum()),
            "overestimate_rate": float((error > 0).mean()),
            "mean_overestimate": float(over.mean()) if len(over) else 0.,
            "over20_count": int((error > 20).sum()),
            "over20_rate": float((error > 20).mean())}


def selection_score(y, predictions):
    clean = metrics(y, predictions["clean"])["rmse"]
    stressed = np.mean([metrics(y, predictions[c])["rmse"] for c in MILD])
    return float(.5*clean+.5*stressed)


def bootstrap(y, prediction, groups, comparison=None, repetitions=2000):
    """Resample motors, retaining all correlated endpoints of each motor."""
    y, prediction, groups = np.asarray(y), np.asarray(prediction), np.asarray(groups)
    unique = np.unique(groups)
    indices = [np.flatnonzero(groups == g) for g in unique]
    rng = np.random.default_rng(20261007)
    values = []
    for _ in range(repetitions):
        ix = np.concatenate([indices[i] for i in rng.integers(len(unique), size=len(unique))])
        value = np.sqrt(np.mean((prediction[ix]-y[ix])**2))
        if comparison is not None:
            value -= np.sqrt(np.mean((np.asarray(comparison)[ix]-y[ix])**2))
        values.append(value)
    return {"unit": "engine", "engines": len(unique), "repetitions": repetitions,
            "metric": "paired_rmse_difference" if comparison is not None else "rmse",
            "ci95": [float(x) for x in np.quantile(values, [.025, .975])]}
