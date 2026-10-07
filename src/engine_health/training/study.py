"""Frozen 52-fit study; all four subsets freeze before any test outcomes."""
import gc
import os
from pathlib import Path
import time

import numpy as np
import torch

from engine_health.common import atomic_json, commit, digest, environment, heavy_lock, identity, now, read_json, source_identity
from engine_health.data import SUBSETS, audit, load_table, prepare
from engine_health.inference import Predictor
from engine_health.models import RULNet
from engine_health.preprocessing import Preprocessor, make_windows
from engine_health.evaluation.metrics import selection_score
from .train import banks, run_baseline, run_gru


def subset_data(root, subset, split):
    df = load_table(root / "data", subset)
    item = split["subsets"][subset]
    train_df = df.loc[df.unit.isin(item["train"])]
    validation_df = df.loc[df.unit.isin(item["validation"])]
    prep = Preprocessor.fit(train_df, subset)
    train = make_windows(train_df, prep)
    validation = list(make_windows(validation_df, prep, item["bank"]))
    # Seeds are bound to each (unit, endpoint), independent of model or run seed.
    seeds = {(r["unit"], r["cycle"]): r["seed"] for r in item["bank"]}
    validation[3] = np.array([seeds[(int(u), int(c))] for u, c in zip(validation[2], validation[3])])
    return prep, train, tuple(validation)


def pilot(root):
    root = Path(root)
    destination = root / "outputs/pilot.json"
    split = prepare(root / "data", root / "protocols/study_v1/splits.json")
    prep, train, _ = subset_data(root, "FD001", split)
    x, y = train[0][:128], train[1][:128] / 100
    values = {}
    for device in ("cpu", "mps"):
        if device == "mps" and not torch.backends.mps.is_available():
            values[device] = {"available": False}
            continue
        try:
            torch.set_num_threads(1)
            torch.manual_seed(20261007)
            model = RULNet().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
            tx, ty = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
            def sync():
                if device == "mps":
                    torch.mps.synchronize()
            times = []
            for step in range(23):
                sync()
                start = time.perf_counter()
                optimizer.zero_grad()
                loss = torch.nn.functional.huber_loss(model(tx), ty, delta=.1)
                loss.backward()
                optimizer.step()
                sync()
                if step >= 3:
                    times.append(time.perf_counter()-start)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite pilot")
            values[device] = {"available": True, "median_batch_seconds": float(np.median(times)),
                              "measured_steps": 20, "batch": 128}
        except (RuntimeError, ValueError) as exc:
            values[device] = {"available": False, "error": str(exc)}
    device = min((d for d in values if values[d]["available"]), key=lambda d: values[d]["median_batch_seconds"])
    record = {"created": now(), "devices": values, "selected": device,
              "note": "Real FD001 training windows; disposable optimizer steps; no model selection or test access",
              "environment": environment()}
    atomic_json(destination, record)
    return record


def freeze(root):
    root = Path(root)
    path = root / "outputs/study-v1/declaration.json"
    if path.exists():
        record = read_json(path)
        verify(root, record)
        return record
    audit(root / "data")
    split = prepare(root / "data", root / "protocols/study_v1/splits.json")
    config = read_json(root / "configs/study_v1.json")
    selection = read_json(root / "outputs/pilot.json")
    record = {"created": now(), "config": config, "device": selection["selected"],
              "split_digest": split["digest"], "split_sha256": digest(root / "protocols/study_v1/splits.json"),
              "source": source_identity(root), "commit": commit(root), "environment": environment(),
              "fit_budget": {"neural": 24, "classical": 24, "constant": 4},
              "test_state": "structure audited; no model outcomes opened"}
    record["digest"] = identity(record)
    atomic_json(path, record)
    return record


def verify(root, declaration):
    if declaration["digest"] != identity({k: v for k, v in declaration.items() if k != "digest"}):
        raise ValueError("Declaration was edited")
    if declaration["source"] != source_identity(root) or declaration["environment"] != environment():
        raise ValueError("Frozen source/environment changed; do not continue this study")
    if declaration["split_sha256"] != digest(root / "protocols/study_v1/splits.json"):
        raise ValueError("Frozen split changed")


def make_spec(root, subset, family, run_paths):
    assets = []
    for path in run_paths:
        file = path / ("selected.pt" if family in ("gru", "robust_gru") else "model.joblib")
        summary = read_json(path / "summary.json")
        if digest(file) != summary["selected_sha256"]:
            raise ValueError("Completed model was modified")
        assets.append({"path": str(file.relative_to(root)), "sha256": digest(file)})
    return {"subset": subset, "family": family, "assets": assets, "id": identity(assets)}


def execute(root):
    root = Path(root).resolve()
    out = root / "outputs/study-v1"
    with heavy_lock(root.parent / ".engine-health-heavy.lock"):
        declaration = freeze(root)
        config = declaration["config"]
        split = read_json(root / "protocols/study_v1/splits.json")
        frozen = out / "frozen-models.json"
        if frozen.exists():
            verify(root, declaration)
            return read_json(frozen)
        registry = {"schema": 1, "declaration_digest": declaration["digest"], "subsets": {}}
        total_done = 0
        try:
            for subset in SUBSETS:
                verify(root, declaration)
                prep, train, validation = subset_data(root, subset, split)
                prep_path = out / subset / "preprocessor.json"
                if prep_path.exists() and read_json(prep_path) != prep.state:
                    raise ValueError("Refitted preprocessing differs")
                atomic_json(prep_path, prep.state)
                provenance = {"declaration_digest": declaration["digest"], "subset": subset,
                              "preprocessor_sha256": digest(prep_path), "split_digest": split["digest"]}
                candidates = []
                for family, params in (("mean", [0]), ("ridge", config["ridge_alphas"]), ("boost", config["boost_leaves"])):
                    for param in params:
                        path = out / subset / f"{family}-{param}"
                        atomic_json(out / "status.json", {"state": "training", "job": f"{subset}/{path.name}", "completed_fits": total_done, "total_fits": 52, "pid": os.getpid(), "updated": now()})
                        summary = run_baseline(path, train, validation, prep.active, family, param, provenance)
                        candidates.append((family, path, summary))
                        total_done += 1
                models = {}
                for family in ("mean", "ridge", "boost"):
                    choices = sorted([row for row in candidates if row[0] == family], key=lambda row: (row[2]["validation_score"], str(row[1])))
                    models[family] = make_spec(root, subset, family, [choices[0][1]])
                for family in config["families"]:
                    paths = []
                    for seed in config["seeds"]:
                        verify(root, declaration)
                        path = out / subset / f"{family}-seed{seed}"
                        atomic_json(out / "status.json", {"state": "training", "job": f"{subset}/{path.name}", "completed_fits": total_done, "total_fits": 52, "pid": os.getpid(), "updated": now()})
                        if (path / "summary.json").exists():
                            summary = read_json(path / "summary.json")
                            if summary["provenance"] != provenance or summary["selected_sha256"] != digest(path / "selected.pt"):
                                raise ValueError("Completed run provenance mismatch")
                        else:
                            run_gru(path, train, validation, prep.active, config, provenance, family, seed,
                                    declaration["device"], resume=(path / "last.pt").exists())
                        paths.append(path)
                        total_done += 1
                    models[family] = make_spec(root, subset, family, paths)
                for seed in config["seeds"]:
                    a = read_json(out / subset / f"gru-seed{seed}/summary.json")
                    b = read_json(out / subset / f"robust_gru-seed{seed}/summary.json")
                    if a["initial_state_sha256"] != b["initial_state_sha256"]:
                        raise ValueError("Paired architectures did not start from identical weights")
                # No official test input is loaded in this module.
                bank = banks(validation[0], validation[3], prep.active)
                for family, spec in models.items():
                    predictor = Predictor(root, spec, subset)
                    predictions = {c: predictor.predict(x) for c, x in bank.items()}
                    spec["validation_score"] = selection_score(validation[1], predictions)
                    predictor.predict(validation[0][:1])
                    samples = []
                    for _ in range(10):
                        start = time.perf_counter()
                        predictor.predict(validation[0][:1])
                        samples.append((time.perf_counter()-start)*1000)
                    spec["cpu_latency_ms"] = float(np.median(samples))
                default = min(models, key=lambda f: (round(models[f]["validation_score"], 8), models[f]["cpu_latency_ms"]))
                registry["subsets"][subset] = {"default": default, "models": models,
                    "preprocessor": {"path": str(prep_path.relative_to(root)), "sha256": digest(prep_path)}}
                atomic_json(out / f"{subset}/selection.json", registry["subsets"][subset])
                del train, validation, bank
                gc.collect()
            verify(root, declaration)
            registry["frozen_at"] = now()
            registry["digest"] = identity(registry)
            atomic_json(frozen, registry)
            atomic_json(out / "status.json", {"state": "frozen_awaiting_evaluation", "completed_fits": total_done, "total_fits": 52, "updated": now()})
            return registry
        except BaseException as exc:
            atomic_json(out / "status.json", {"state": "failed", "error": str(exc), "type": type(exc).__name__, "completed_fits": total_done, "updated": now()})
            raise
