"""Train-only operating-condition normalization and causal windows."""
import numpy as np
from sklearn.cluster import KMeans


class Preprocessor:
    window = 30

    def __init__(self, state):
        self.state = state
        self.subset = state["subset"]
        self.active = np.asarray(state["active"], dtype=int)

    @classmethod
    def fit(cls, frame, subset):
        settings = frame.iloc[:, 2:5].to_numpy(dtype=float)
        sensors = frame.iloc[:, 5:26].to_numpy(dtype=float)
        sm, ss = settings.mean(0), settings.std(0)
        ss[ss < 1e-8] = 1
        z = (settings - sm) / ss
        k = 6 if subset in ("FD002", "FD004") else 1
        centers = KMeans(k, random_state=20261007, n_init=10).fit(z).cluster_centers_ if k > 1 else z.mean(0, keepdims=True)
        labels = ((z[:, None] - centers[None]) ** 2).sum(-1).argmin(1)
        means = np.stack([sensors[labels == i].mean(0) for i in range(k)])
        scales = np.stack([sensors[labels == i].std(0) for i in range(k)])
        active = np.flatnonzero((scales >= 1e-5).any(0))
        scales[scales < 1e-5] = 1
        return cls({"schema": 1, "subset": subset, "window": 30,
                    "settings_mean": sm.tolist(), "settings_scale": ss.tolist(),
                    "centers": centers.tolist(), "means": means.tolist(),
                    "scales": scales.tolist(), "active": active.tolist(),
                    "fit_units": sorted(int(u) for u in frame.unit.unique())})

    def normalize(self, rows):
        """Rows contain settings (3) and sensors (21), with no target/age input."""
        a = np.asarray(rows, dtype=np.float64)
        if a.ndim != 2 or a.shape[1] != 24 or not len(a):
            raise ValueError("Expected nonempty rows with 3 settings and 21 sensors")
        if not np.isfinite(a[:, :3]).all() or np.isinf(a[:, 3:]).any():
            raise ValueError("Settings must be finite; sensors may be missing, not infinite")
        z = (a[:, :3] - self.state["settings_mean"]) / self.state["settings_scale"]
        labels = ((z[:, None] - np.asarray(self.state["centers"])[None]) ** 2).sum(-1).argmin(1)
        s = (a[:, 3:] - np.asarray(self.state["means"])[labels]) / np.asarray(self.state["scales"])[labels]
        out = np.zeros((len(a), 46), dtype=np.float32)
        out[:, :21], out[:, 21:24], out[:, 45] = s, z, 1
        out[:, 24:45] = np.isnan(s)
        return out

    def window_from_rows(self, rows):
        normalized = self.normalize(np.asarray(rows)[-self.window:])
        out = np.zeros((self.window, 46), dtype=np.float32)
        out[-len(normalized):] = normalized
        return fill_missing(out[None])[0]


def fill_missing(windows):
    """Causal fill in normalized space; zero is training condition mean."""
    x = windows.copy()
    previous = np.zeros((len(x), 21), dtype=np.float32)
    for t in range(x.shape[1]):
        real = x[:, t, 45:46] > 0
        missing = np.isnan(x[:, t, :21]) | (x[:, t, 24:45] > 0)
        x[:, t, 24:45] = missing & real
        value = np.where(missing, previous, x[:, t, :21])
        x[:, t, :21] = np.where(real, value, 0)
        previous = np.where(real, value, previous)
    return x


def make_windows(frame, prep, bank=None):
    xs, ys, units, cycles = [], [], [], []
    endpoints = {}
    if bank is not None:
        for row in bank:
            endpoints.setdefault(row["unit"], []).append(row["cycle"])
    for unit, group in frame.groupby("unit", sort=True):
        if bank is not None and int(unit) not in endpoints:
            continue
        a = prep.normalize(group.iloc[:, 2:26].to_numpy())
        padded = np.pad(a, ((29, 0), (0, 0)))
        ends = endpoints[int(unit)] if bank is not None else group.cycle.tolist()
        for cycle in ends:
            xs.append(padded[int(cycle)-1:int(cycle)+29])
            ys.append(len(group)-int(cycle))
            units.append(int(unit))
            cycles.append(int(cycle))
    return fill_missing(np.stack(xs)), np.array(ys, dtype=np.float32), np.array(units), np.array(cycles)


def features(x):
    """Window summaries exclude padding; missingness remains observable."""
    mask = x[:, :, 45:46]
    n = np.maximum(mask.sum(1), 1)
    values = x[:, :, :45]
    mean = (values * mask).sum(1) / n
    variance = (((values - mean[:, None]) ** 2) * mask).sum(1) / n
    t = np.arange(x.shape[1], dtype=np.float32)[None, :, None]
    mt = (t * mask).sum(1) / n
    denom = (((t - mt[:, None]) ** 2) * mask).sum(1)
    slope = ((t-mt[:, None]) * (values-mean[:, None]) * mask).sum(1) / np.maximum(denom, 1)
    return np.concatenate([values[:, -1], mean, np.sqrt(variance), slope], axis=1)
