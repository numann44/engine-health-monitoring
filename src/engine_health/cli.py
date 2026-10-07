import argparse
import json
from pathlib import Path

from engine_health.common import read_json


def main():
    parser = argparse.ArgumentParser(description="Auditable C-MAPSS remaining-life experiments")
    parser.add_argument("command", choices=["download", "audit", "prepare", "pilot", "freeze", "train", "resume", "study", "evaluate", "robustness", "report", "package", "predict", "benchmark"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--full", action="store_true", help="After all selections freeze, evaluate, report and package")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--subset", choices=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--unit", type=int, default=1)
    parser.add_argument("--family", choices=["mean", "ridge", "boost", "gru", "robust_gru"])
    parser.add_argument("--cycle", type=int)
    parser.add_argument("--condition", default="clean", choices=["clean", "missing1", "noise025", "stuck10", "missing3", "noise050", "bias_plus", "bias_minus"])
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command in ("download", "audit", "prepare"):
        from engine_health.data import download, audit, prepare
        if args.command == "prepare":
            result = prepare(root / "data", root / "protocols/study_v1/splits.json")
        else:
            result = (download if args.command == "download" else audit)(root / "data")
    elif args.command in ("pilot", "freeze", "train", "resume", "study"):
        from engine_health.training.study import pilot, freeze, execute
        if args.command == "pilot":
            result = pilot(root)
        elif args.command == "freeze":
            result = freeze(root)
        else:
            result = execute(root)
            if args.full:
                from engine_health.evaluation.study import evaluate
                from engine_health.evaluation.report import report, package
                evaluate(root)
                report(root)
                result = package(root)
    elif args.command in ("evaluate", "robustness"):
        from engine_health.evaluation.study import evaluate
        result = evaluate(root)
    elif args.command in ("report", "package"):
        from engine_health.evaluation.report import report, package
        result = (report if args.command == "report" else package)(root)
    else:
        from engine_health.inference import Engine, parse_upload
        if args.csv is None or args.subset is None:
            parser.error("predict requires --csv and --subset")
        local_registry = root / "outputs/study-v1/frozen-models.json"
        if local_registry.exists():
            model_root, registry = root, read_json(local_registry)
        else:
            from engine_health.inference.release import release_root
            model_root = release_root(root)
            registry = read_json(model_root / "registry.json")
        engine = Engine(model_root, registry, args.subset, args.family)
        frame = parse_upload(args.csv.read_bytes())
        def inspect():
            return engine.inspect(frame, args.unit, cycle=args.cycle, condition=args.condition)
        if args.command == "benchmark":
            import numpy as np
            for _ in range(3):
                inspect()
            samples = [inspect()["latency_ms"] for _ in range(30)]
            result = {"model_id": engine.model_id, "trials": 30, "warmups": 3, "device": "cpu",
                      "median_ms": float(np.median(samples)), "p95_ms": float(np.quantile(samples, .95)),
                      "scope": "single-engine preprocessing and inference; file read/model load excluded"}
        else:
            result = inspect()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
