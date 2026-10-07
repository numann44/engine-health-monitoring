import os
from pathlib import Path
import random
import tempfile
import time

import joblib
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch

from engine_health.common import atomic_json, digest, identity, read_json
from engine_health.models import RULNet, predict_net
from engine_health.preprocessing import features
from engine_health.preprocessing.stress import CONDITIONS, augment, perturb
from engine_health.evaluation.metrics import selection_score


def save_checkpoint(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".checkpoint-")
    os.close(fd)
    try:
        with open(tmp, "wb") as stream:
            torch.save(record, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def banks(x, seeds, active, full=False):
    conditions = CONDITIONS if full else CONDITIONS[:4]
    return {c: perturb(x, c, seeds, active) for c in conditions}


def equal_engine_weights(units):
    _, inv, counts = np.unique(units, return_inverse=True, return_counts=True)
    weights = 1. / counts[inv]
    return (weights / weights.mean()).astype(np.float32)


def run_gru(run, train, validation, active, config, provenance, family, seed, device="cpu", resume=False, stop_after=None):
    run = Path(run)
    if run.exists() and not resume:
        raise FileExistsError(f"Run already exists: {run}; explicit resume required")
    run.mkdir(parents=True, exist_ok=True)
    x, y, units, _ = train
    vx, vy, _, vc = validation
    valbank = banks(vx, vc, active)
    weights = equal_engine_weights(units)
    torch.set_num_threads(1)
    random.seed(seed)
    torch.manual_seed(seed)
    shuffle = np.random.default_rng(seed)
    augmentation = np.random.default_rng(seed + 9_000_000)
    model = RULNet().to(device)
    initial_digest = identity({k: v.cpu().tolist() for k, v in model.state_dict().items()})
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    start, best, stale, history = 0, float("inf"), 0, []
    best_epoch = 0
    run_identity = identity({"provenance": provenance, "config": config,
                             "family": family, "seed": seed, "device": device})
    last = run / "last.pt"
    if resume:
        if not last.exists():
            raise FileNotFoundError("No compatible last checkpoint")
        cp = torch.load(last, map_location="cpu", weights_only=True)
        if cp["identity"] != run_identity:
            raise ValueError("Checkpoint provenance/config/device mismatch")
        model.load_state_dict(cp["model"])
        optimizer.load_state_dict(cp["optimizer"])
        random.setstate(cp["rng_python"])
        torch.set_rng_state(cp["rng_torch"])
        if device == "mps":
            torch.mps.set_rng_state(cp["rng_mps"])
        shuffle.bit_generator.state = cp["rng_shuffle"]
        augmentation.bit_generator.state = cp["rng_augmentation"]
        start, best, best_epoch, stale, history = cp["epoch"], cp["best"], cp["best_epoch"], cp["stale"], cp["history"]
    for epoch in range(start+1, config["epochs"]+1):
        if stale >= config["patience"]:
            break
        started = time.perf_counter()
        model.train()
        loss_sum = 0.
        order = shuffle.permutation(len(x))
        for pos in range(0, len(order), config["batch"]):
            ix = order[pos:pos+config["batch"]]
            inputs = x[ix]
            if family == "robust_gru":
                inputs = augment(inputs, augmentation, active)
            tx = torch.from_numpy(inputs).to(device)
            ty = torch.from_numpy(y[ix] / 100).to(device)
            tw = torch.from_numpy(weights[ix]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = (torch.nn.functional.huber_loss(model(tx), ty, reduction="none", delta=config["huber_delta"]) * tw).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss; run stopped")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(ix)
        predictions = {c: predict_net(model, a, device) for c, a in valbank.items()}
        score = selection_score(vy, predictions)
        improved = score < best-1e-8
        if improved:
            best, best_epoch, stale = score, epoch, 0
        else:
            stale += 1
        history.append({"epoch": epoch, "loss": loss_sum/len(x), "validation_score": score,
                        "seconds": time.perf_counter()-started, "stale": stale})
        cp = {"schema": 1, "identity": run_identity, "provenance": provenance,
              "config": config, "family": family, "seed": seed, "device": device,
              "initial_state_sha256": initial_digest, "model": model.state_dict(),
              "optimizer": optimizer.state_dict(), "epoch": epoch, "best": best,
              "best_epoch": best_epoch, "stale": stale, "history": history,
              "rng_python": random.getstate(), "rng_torch": torch.get_rng_state(),
              "rng_shuffle": shuffle.bit_generator.state,
              "rng_augmentation": augmentation.bit_generator.state,
              "rng_mps": torch.mps.get_rng_state() if device == "mps" else None}
        save_checkpoint(last, cp)
        if improved:
            selected = {k: cp[k] for k in ("schema", "identity", "provenance", "config", "family", "seed", "epoch", "best", "initial_state_sha256", "model")}
            save_checkpoint(run / "selected.pt", selected)
        atomic_json(run / "history.json", history)
        atomic_json(run / "progress.json", {"epoch": epoch, "best_epoch": best_epoch, "best_score": best, "stale": stale, "seconds": history[-1]["seconds"]})
        print(f"{run.name} epoch={epoch} score={score:.4f} best={best:.4f} seconds={history[-1]['seconds']:.1f}", flush=True)
        if stop_after is not None and epoch >= stop_after:
            return None
    summary = {"family": family, "seed": seed, "epochs": len(history), "best_epoch": best_epoch,
               "validation_score": best, "selected_sha256": digest(run / "selected.pt"),
               "provenance": provenance, "identity": run_identity,
               "initial_state_sha256": initial_digest,
               "early_stopped": stale >= config["patience"]}
    atomic_json(run / "summary.json", summary)
    return summary


def run_baseline(run, train, validation, active, family, parameter, provenance):
    run = Path(run)
    if run.exists():
        summary = read_json(run / "summary.json")
        if summary["provenance"] != provenance or digest(run / "model.joblib") != summary["selected_sha256"]:
            raise ValueError("Changed baseline checkpoint")
        return summary
    x, y, units, _ = train
    vx, vy, _, seeds = validation
    weights = equal_engine_weights(units)
    f = features(x)
    if family == "mean":
        model = DummyRegressor(strategy="mean")
        model.fit(f, y, sample_weight=weights)
    elif family == "ridge":
        model = make_pipeline(StandardScaler(), Ridge(alpha=parameter))
        model.fit(f, y, standardscaler__sample_weight=weights, ridge__sample_weight=weights)
    elif family == "boost":
        model = HistGradientBoostingRegressor(max_leaf_nodes=int(parameter), max_iter=200,
                    learning_rate=.05, l2_regularization=1, early_stopping=False, random_state=42)
        model.fit(f, y, sample_weight=weights)
    else:
        raise ValueError("Unknown baseline")
    predictions = {c: np.maximum(0, model.predict(features(a))) for c, a in banks(vx, seeds, active).items()}
    run.mkdir(parents=True)
    joblib.dump(model, run / "model.joblib")
    summary = {"family": family, "parameter": parameter,
               "validation_score": selection_score(vy, predictions),
               "selected_sha256": digest(run / "model.joblib"), "provenance": provenance}
    atomic_json(run / "summary.json", summary)
    return summary
