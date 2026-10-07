"""Measurement corruptions in training-condition standard-deviation units."""
import numpy as np
from . import fill_missing

CONDITIONS = ("clean", "missing1", "noise025", "stuck10", "missing3", "noise050", "bias_plus", "bias_minus")
MILD = ("missing1", "noise025", "stuck10")


def perturb(x, condition, seeds, active, channel=None, magnitude=None):
    if condition not in CONDITIONS:
        raise ValueError("Unknown stress condition")
    if condition == "clean":
        return x.copy()
    result = x.copy()
    active = np.asarray(active)
    if not len(active):
        raise ValueError("No variable sensors")
    for i, seed in enumerate(seeds):
        rng = np.random.default_rng(int(seed))
        channels = rng.permutation(active) if channel is None else np.array([channel])
        if not set(channels).issubset(set(active)):
            raise ValueError("Sensor is outside declared variable channels")
        valid = np.flatnonzero(result[i, :, 45] > 0)
        c = channels[0]
        if condition.startswith("missing"):
            count = 3 if condition == "missing3" else 1
            for c in channels[:count]:
                result[i, valid, c] = np.nan
                result[i, valid, 24+c] = 1
        elif condition.startswith("noise"):
            sigma = magnitude if magnitude is not None else (.25 if condition == "noise025" else .5)
            # One channel, not all channels: same selected sensor across conditions.
            result[i, valid, c] += rng.normal(0, sigma, len(valid))
        elif condition == "stuck10":
            length = int(magnitude) if magnitude is not None else 10
            affected = valid[-length:]
            if len(affected):
                result[i, affected, c] = result[i, affected[0], c]
        else:
            shift = magnitude if magnitude is not None else .5
            result[i, valid, c] += shift if condition == "bias_plus" else -shift
    return fill_missing(result)


def augment(x, rng, active):
    result = x.copy()
    for i in range(len(x)):
        if rng.random() < .5:
            continue
        condition = ("missing1", "noise050", "stuck10")[int(rng.integers(3))]
        mag = float(rng.uniform(0, .5)) if condition == "noise050" else int(rng.integers(5, 11)) if condition == "stuck10" else None
        result[i:i+1] = perturb(result[i:i+1], condition, [rng.integers(2**32)], active, magnitude=mag)
    return result
