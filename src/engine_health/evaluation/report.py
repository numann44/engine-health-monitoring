"""Generate presentation from measured records; never synthesize results."""
import copy
import io
from pathlib import Path
import shutil
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from engine_health.common import atomic_json, digest, identity, read_json
from engine_health.data import SUBSETS
from engine_health.preprocessing.stress import CONDITIONS

COLORS = {"mean": "#94a3b8", "ridge": "#6366f1", "boost": "#3b82f6", "gru": "#d97706", "robust_gru": "#0d9488"}


def png_chart(actual, predicted, cycles, family="robust_gru"):
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    ax.plot(cycles, predicted, color=COLORS[family], label="Predicted RUL")
    if actual is not None:
        ax.plot(cycles, actual, "--", color="#334155", label="Actual RUL")
    ax.set(xlabel="Observed cycle", ylabel="Remaining cycles")
    ax.legend(frameon=False)
    ax.grid(alpha=.15)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150)
    plt.close(fig)
    return buffer.getvalue()


def report(root):
    root = Path(root).resolve()
    out = root / "outputs/study-v1"
    results = read_json(out / "results.json")
    docs = root / "docs/results"
    images = root / "docs/images"
    docs.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    atomic_json(docs / "study-v1.json", results)
    lines = ["# Controlled sensor robustness study", "",
             "All four subset selections were frozen before official test evaluation. Each test engine contributes one final-observation prediction. Targets are uncapped RUL cycles.", "",
             "A different model is fitted per subset. This is not evidence of one model generalizing across all engine regimes or of reliability on real engines.", "",
             "| Subset | Default (validation-selected) | Clean RMSE | Clean MAE | >20-cycle overestimates | Engines |",
             "|---|---|---:|---:|---:|---:|"]
    table = []
    for subset in SUBSETS:
        row = results["subsets"][subset]
        m = row["models"][row["default"]]["conditions"]["clean"]
        table.append(f"| {subset} | {row['default']} | {m['rmse']:.2f} | {m['mae']:.2f} | {m['over20_count']} | {row['engines']} |")
    lines += table
    lines += ["", "## Predeclared hypothesis", "", "At least 10% lower mean mild-stress RMSE than the standard GRU, with at most 5% higher clean RMSE. This is a point-estimate criterion; intervals are in the machine-readable report.", "",
              "| Subset | Mild RMSE reduction | Clean RMSE increase | Point target |", "|---|---:|---:|---|"]
    for subset in SUBSETS:
        h = results["subsets"][subset]["hypothesis"]
        lines.append(f"| {subset} | {h['mild_rmse_reduction']:.1%} | {h['clean_rmse_increase']:.1%} | {'Met' if h['point_target_met'] else 'Not met'} |")
    for subset in SUBSETS:
        row = results["subsets"][subset]
        path = out / subset / "predictions.npz"
        if digest(path) != row["predictions_sha256"]:
            raise ValueError("Prediction evidence changed")
        p = np.load(path)
        y, pred = p["y"], p[f"{row['default']}__clean"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        axes[0].scatter(y, pred, s=18, alpha=.7, color="#0d9488")
        top = float(max(y.max(), pred.max()))
        axes[0].plot([0, top], [0, top], "--", color="#64748b")
        axes[0].set(xlabel="Actual remaining cycles", ylabel="Predicted remaining cycles", title=f"{subset} · {row['default']}")
        axes[1].hist(pred-y, bins=20, color="#0d9488", alpha=.8)
        axes[1].axvline(20, color="#d97706", linestyle="--", label="20-cycle overestimate")
        axes[1].set(xlabel="Prediction − actual (cycles)", ylabel="Engines", title="Signed errors")
        axes[1].legend(frameon=False)
        fig.savefig(images / f"{subset.lower()}-errors.png", dpi=160)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(11, 4), layout="constrained")
        for family, item in row["models"].items():
            ax.plot(CONDITIONS, [item["conditions"][c]["rmse"] for c in CONDITIONS], marker="o", label=family, color=COLORS[family])
        ax.set(title=f"{subset} · fixed sensor stress conditions", ylabel="RMSE (cycles)")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(frameon=False, ncol=3)
        ax.grid(alpha=.15)
        fig.savefig(images / f"{subset.lower()}-robustness.png", dpi=160)
        plt.close(fig)
        order = np.argsort(np.abs(pred-y), kind="stable")
        chosen = [order[0], order[len(order)//2], order[-1]]
        lines += ["", f"## {subset}", "", f"![Measured prediction errors](../images/{subset.lower()}-errors.png)", "", f"![Fixed sensor stress](../images/{subset.lower()}-robustness.png)", "",
                  "Best, median and worst absolute-error engines, selected by a declared rule after evaluation. These examples are descriptive and never used for model selection.", "",
                  "| Example | Engine | Actual RUL | Predicted RUL | Signed error |", "|---|---:|---:|---:|---:|"]
        for label, ix in zip(("Best", "Median", "Worst"), chosen):
            lines.append(f"| {label} | {int(p['unit'][ix])} | {y[ix]:.0f} | {pred[ix]:.2f} | {pred[ix]-y[ix]:+.2f} |")
        lines += ["", "| Model | Clean RMSE (95% engine-bootstrap interval) | CPU median / p95 ms |", "|---|---|---|"]
        for family, item in row["models"].items():
            m = item["conditions"]["clean"]
            low, high = m["uncertainty"]["ci95"]
            lines.append(f"| {family} | {m['rmse']:.2f} [{low:.2f}, {high:.2f}] | {item['cpu_median_ms']:.2f} / {item['cpu_p95_ms']:.2f} |")
    lines += ["", "## Limits", "", "Uncapped targets and motor-grouped validation differ from some published benchmark protocols; numbers are not leaderboard claims. Four training-life fractions form validation endpoints and need not match the distribution of official test stopping points. Sensor perturbations simulate measurement problems, not physical engine damage. No test-driven retuning is part of this release.", "", "Intervals resample engines (2,000 draws), not correlated stress copies. Latency is warm, single-engine CPU inference, 30 trials, without concurrent model training. See the JSON for every condition, individual GRU members and paired RMSE intervals."]
    (docs / "STUDY_V1.md").write_text("\n".join(lines)+"\n")
    return {"report": str(docs / "STUDY_V1.md"), "table": "\n".join(table)}


def package(root):
    root = Path(root).resolve()
    registry = read_json(root / "outputs/study-v1/frozen-models.json")
    results = read_json(root / "outputs/study-v1/results.json")
    folder = root / "outputs/release-v0.1.0"
    folder.mkdir(parents=True, exist_ok=True)
    portable = copy.deepcopy(registry)
    portable.pop("digest")
    portable["study_registry_digest"] = registry["digest"]
    paths = {}
    for subset, record in portable["subsets"].items():
        items = [record["preprocessor"]] + [a for m in record["models"].values() for a in m["assets"]]
        for item in items:
            source = root / item["path"]
            if digest(source) != item["sha256"]:
                raise ValueError("Release source checksum mismatch")
            relative = f"models/{subset}/{item['sha256'][:16]}-{source.name}"
            dest = folder / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            item["path"] = relative
            paths[relative] = item["sha256"]
    portable["digest"] = identity(portable)
    atomic_json(folder / "registry.json", portable)
    atomic_json(folder / "results.json", results)
    paths["registry.json"] = digest(folder / "registry.json")
    paths["results.json"] = digest(folder / "results.json")
    atomic_json(folder / "checksums.json", paths)
    archive = folder / "engine-health-v0.1.0.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(paths):
            z.write(folder / path, path)
        z.write(folder / "checksums.json", "checksums.json")
    release = {"schema": 1, "version": "v0.1.0", "archive_sha256": digest(archive),
               "registry_sha256": digest(folder / "registry.json"),
               "url": "https://github.com/numann44/engine-health-monitoring/releases/download/v0.1.0/engine-health-v0.1.0.zip"}
    atomic_json(root / "assets/model-registry.json", release)
    return release
