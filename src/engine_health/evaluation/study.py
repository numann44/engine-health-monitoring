import hashlib
from pathlib import Path
import time

import numpy as np

from engine_health.common import atomic_json, digest, heavy_lock, identity, now, read_json
from engine_health.data import SUBSETS, load_table
from engine_health.data.cmapss import HASHES
from engine_health.inference import Predictor
from engine_health.preprocessing import Preprocessor, make_windows
from engine_health.preprocessing.stress import CONDITIONS, MILD, perturb
from engine_health.training.study import verify
from .metrics import bootstrap, metrics


def evaluate(root):
    root = Path(root).resolve()
    out = root / "outputs/study-v1"
    with heavy_lock(root.parent / ".engine-health-heavy.lock"):
        declaration = read_json(out / "declaration.json")
        verify(root, declaration)
        registry = read_json(out / "frozen-models.json")
        if registry["digest"] != identity({k: v for k, v in registry.items() if k != "digest"}):
            raise ValueError("Frozen registry changed")
        if set(registry["subsets"]) != set(SUBSETS):
            raise ValueError("All four subset selections must freeze before test access")
        path = out / "evaluation-declaration.json"
        evaluation_identity = {"source": declaration["source"], "registry_sha256": digest(out / "frozen-models.json"),
                               "conditions": list(CONDITIONS), "bootstrap": 2000,
                               "target": "uncapped final-observation RUL cycles",
                               "selection": "none; all models fixed before test outcomes",
                               "test_role": "official held-out test for this preregistered study"}
        if path.exists() and read_json(path) != evaluation_identity:
            raise ValueError("Evaluation declaration changed")
        atomic_json(path, evaluation_identity)
        results = {"schema": 1, "evaluation": evaluation_identity, "subsets": {}}
        for subset in SUBSETS:
            destination = out / subset / "evaluation.json"
            if destination.exists():
                saved = read_json(destination)
                if saved["evaluation_digest"] != identity(evaluation_identity):
                    raise ValueError("Existing evaluation has different provenance")
                results["subsets"][subset] = saved
                continue
            atomic_json(out / "status.json", {"state": "evaluating", "subset": subset, "updated": now()})
            record = registry["subsets"][subset]
            if digest(root / record["preprocessor"]["path"]) != record["preprocessor"]["sha256"]:
                raise ValueError("Preprocessor was changed")
            prep = Preprocessor(read_json(root / record["preprocessor"]["path"]))
            df = load_table(root / "data", subset, "test")
            endpoints = [{"unit": int(unit), "cycle": int(group.cycle.max())} for unit, group in df.groupby("unit", sort=True)]
            x, _, units, cycles = make_windows(df, prep, endpoints)
            label_path = root / "data/raw" / f"RUL_{subset}.txt"
            if digest(label_path) != HASHES[label_path.name]:
                raise ValueError("Test labels changed since audit")
            y = np.loadtxt(label_path, ndmin=1)
            seeds = [int(hashlib.sha256(f"{subset}:test:{int(u)}:{int(c)}".encode()).hexdigest()[:8], 16) for u, c in zip(units, cycles)]
            bank = {c: perturb(x, c, seeds, prep.active) for c in CONDITIONS}
            stress_record = {"conditions": list(CONDITIONS), "active_channels_zero_based": prep.active.tolist(),
                             "endpoints": [{**e, "seed": int(s)} for e, s in zip(endpoints, seeds)]}
            atomic_json(out / subset / "test-stress-manifest.json", stress_record)
            raw = {"y": y, "unit": units, "cycle": cycles}
            scores = {}
            for family, spec in record["models"].items():
                predictor = Predictor(root, spec, subset)
                predictions = {c: predictor.predict(a) for c, a in bank.items()}
                predictor.predict(x[:1])
                latencies = []
                for _ in range(30):
                    start = time.perf_counter()
                    predictor.predict(x[:1])
                    latencies.append((time.perf_counter()-start)*1000)
                scores[family] = {"conditions": {}, "cpu_median_ms": float(np.median(latencies)),
                                  "cpu_p95_ms": float(np.quantile(latencies, .95)),
                                  "asset_bytes": sum((root/a["path"]).stat().st_size for a in spec["assets"])}
                for c, prediction in predictions.items():
                    m = metrics(y, prediction)
                    m["uncertainty"] = bootstrap(y, prediction, units)
                    scores[family]["conditions"][c] = m
                    raw[f"{family}__{c}"] = prediction
                if family in ("gru", "robust_gru"):
                    per_seed = []
                    for asset in spec["assets"]:
                        member = Predictor(root, {**spec, "assets": [asset]}, subset)
                        member_predictions = {c: member.predict(a) for c, a in bank.items()}
                        per_seed.append({"asset": asset["sha256"], "conditions": {c: metrics(y, p) for c, p in member_predictions.items()}})
                    scores[family]["members"] = per_seed
            control = scores["gru"]["conditions"]
            robust = scores["robust_gru"]["conditions"]
            control_mild = float(np.mean([control[c]["rmse"] for c in MILD]))
            robust_mild = float(np.mean([robust[c]["rmse"] for c in MILD]))
            mild_reduction = 1-robust_mild/control_mild if control_mild else None
            clean_increase = robust["clean"]["rmse"]/control["clean"]["rmse"]-1 if control["clean"]["rmse"] else None
            differences = {c: bootstrap(y, raw[f"robust_gru__{c}"], units, raw[f"gru__{c}"]) for c in CONDITIONS}
            hypothesis = {"mild_rmse_reduction": mild_reduction, "clean_rmse_increase": clean_increase,
                          "point_target_met": bool(mild_reduction is not None and clean_increase is not None and mild_reduction >= .1 and clean_increase <= .05),
                          "paired_rmse_intervals": differences,
                          "note": "Point-estimate hypothesis; CIs reported separately. Sensor copies are correlated."}
            np.savez_compressed(out / subset / "predictions.npz", **raw)
            result = {"evaluation_digest": identity(evaluation_identity), "subset": subset,
                      "engines": len(y), "default": record["default"], "models": scores,
                      "hypothesis": hypothesis, "predictions_sha256": digest(out / subset / "predictions.npz")}
            atomic_json(destination, result)
            results["subsets"][subset] = result
        results["completed_at"] = now()
        atomic_json(out / "results.json", results)
        atomic_json(out / "status.json", {"state": "evaluated", "updated": now(), "training_complete": True})
        return results
