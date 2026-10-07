"""Interactive retrospective experiments on simulated engines."""
# ruff: noqa: E402 -- Streamlit also supports a source checkout without editable install.
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from engine_health.common import read_json
from engine_health.data import COLUMNS, SUBSETS, download, load_table
from engine_health.evaluation.report import COLORS, png_chart
from engine_health.inference import Engine, parse_upload
from engine_health.inference.release import release_root
from engine_health.preprocessing.stress import CONDITIONS, perturb

st.set_page_config(page_title="Engine Health Monitoring", page_icon="◉", layout="wide")
torch.set_num_threads(1)
st.markdown("""
<style>
.block-container {max-width: 1240px; padding-top: 2.1rem;}
h1 {letter-spacing: -.055em; font-size: 3.1rem !important; font-weight: 750 !important;}
h2,h3 {letter-spacing: -.025em;}
.eyebrow {color: #0D9488; font-size: .76rem; letter-spacing: .19em; font-weight: 750;}
.intro {color: #526575; max-width: 800px; font-size: 1.05rem; line-height: 1.7; margin-bottom: 1.5rem;}
div[data-testid="stMetric"] {background: white; border: 1px solid #E2E8F0; padding: 1.2rem; border-radius: 12px;}
div[data-testid="stMetricLabel"] {color: #526575;}
.note {padding: .8rem 1rem; background: #EDF8F6; border-left: 3px solid #0D9488; color: #345564; font-size: .88rem;}
</style>
<div class="eyebrow">TIME SERIES / PREDICTIVE MAINTENANCE</div>
<h1>Engine Health Monitoring</h1>
<div class="intro">Explore remaining-life predictions. Change a sensor. See what holds up.<br>
A controlled study of four NASA C-MAPSS simulation scenarios.</div>
""", unsafe_allow_html=True)


@st.cache_resource
def resources():
    folder = release_root(ROOT)
    registry = read_json(folder / "registry.json")
    results = read_json(folder / "results.json")
    return folder, registry, results


@st.cache_resource
def load_engine(folder, registry_json, subset, family):
    return Engine(Path(folder), json.loads(registry_json), subset, family)


@st.cache_data(show_spinner=False)
def example_data(subset):
    download(ROOT / "data")
    df = load_table(ROOT / "data", subset, "test")
    labels = np.loadtxt(ROOT / "data/raw" / f"RUL_{subset}.txt", ndmin=1)
    return df, labels


def line_chart(cycles, predicted, actual=None, family="robust_gru"):
    fig = go.Figure()
    if actual is not None:
        fig.add_trace(go.Scatter(x=cycles, y=actual, name="Actual RUL", mode="lines",
                                line=dict(color="#475569", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=cycles, y=predicted, name="Predicted RUL", mode="lines",
                            line=dict(color=COLORS[family], width=3)))
    fig.update_layout(height=390, margin=dict(l=20, r=20, t=20, b=20),
                      template="plotly_white", hovermode="x unified",
                      xaxis_title="Observed cycle", yaxis_title="Remaining cycles",
                      legend=dict(orientation="h", y=1.12), paper_bgcolor="rgba(0,0,0,0)")
    return fig


try:
    folder, registry, results = resources()
except (OSError, ValueError, KeyError) as exc:
    st.info("The declared study is in progress. Verified model assets will appear here after all four selections and their evaluation are complete.")
    st.caption(str(exc))
    st.link_button("Read the experiment protocol", "https://github.com/numann44/engine-health-monitoring/blob/main/docs/experiment-protocol.md")
    st.stop()

with st.sidebar:
    st.markdown("### Experiment controls")
    subset = st.selectbox("Simulation scenario", SUBSETS)
    record = registry["subsets"][subset]
    mode = st.radio("Input", ["Example engine", "Upload CSV"])
    families = list(record["models"])
    family = st.selectbox("Prediction model", families, index=families.index(record["default"]))
    st.caption(f"Validation-selected default: {record['default']}")
    st.divider()
    st.caption("Units: operational cycles. No conversion to hours or days.")
    st.caption("Separate models per scenario. Simulated engines; no real-aircraft validation.")
    st.link_button("Source & measured evidence", "https://github.com/numann44/engine-health-monitoring")

engine = load_engine(str(folder), json.dumps(registry, sort_keys=True), subset, family)
labels = None
if mode == "Example engine":
    try:
        with st.spinner("Loading verified NASA examples…"):
            frame, labels = example_data(subset)
    except (OSError, ValueError, RuntimeError) as exc:
        st.error(f"Verified example data are unavailable: {exc}")
        st.stop()
else:
    st.markdown("#### Inspect your measurement history")
    st.caption("CSV: unit, cycle, setting_1–3, s1–s21. Empty sensor cells are allowed. Files remain in session memory.")
    st.download_button("Download CSV header template", (",".join(COLUMNS)+"\n").encode(), "measurement-template.csv", "text/csv")
    upload = st.file_uploader("Measurement CSV · up to 10 MB / 50,000 rows", type=["csv"])
    if upload is None:
        st.stop()
    try:
        frame = parse_upload(upload.getvalue())
    except (ValueError, pd.errors.ParserError, UnicodeError) as exc:
        st.error(f"Cannot inspect this file: {exc}")
        st.stop()

c1, c2 = st.columns([1, 3])
with c1:
    unit = st.selectbox("Engine", [int(x) for x in sorted(frame.unit.unique())])
group = frame.loc[frame.unit == unit]
first, last = int(group.cycle.min()), int(group.cycle.max())
with c2:
    cycle = st.slider("History available through cycle", first, last, last) if first < last else last
history = group.loc[group.cycle <= cycle]
result = engine.inspect(history, unit)
actual = float(labels[unit-1]+last-cycle) if labels is not None else None

m1, m2, m3, m4 = st.columns(4)
m1.metric("Predicted remaining life", f"{result['rul_cycles']:.1f}", "cycles", delta_color="off")
m2.metric("Actual remaining life", f"{actual:.0f}" if actual is not None else "Unknown", "cycles" if actual is not None else "No label supplied", delta_color="off")
m3.metric("Prediction − actual", f"{result['rul_cycles']-actual:+.1f}" if actual is not None else "—", "cycles" if actual is not None else None, delta_color="off")
m4.metric("Inference", f"{result['latency_ms']:.1f} ms", f"{result['history_length']}-cycle window", delta_color="off")
st.caption("Actual values are retrospective reference labels, never model inputs. Positive error means an overly optimistic life estimate.")
for warning in result["warnings"]:
    st.warning(warning)

explore, stress, evidence = st.tabs(["Explore an Engine", "Sensor Stress Lab", "Benchmark & Evidence"])
with explore:
    st.subheader("The life estimate, over time")
    rows = history.iloc[:, 2:26].to_numpy()
    stops = np.unique(np.linspace(1, len(rows), min(len(rows), 300), dtype=int))
    plot_history = history.iloc[stops-1]
    windows = np.stack([engine.prep.window_from_rows(rows[:stop]) for stop in stops])
    predictions = engine.predictor.predict(windows)
    truth = labels[unit-1]+last-plot_history.cycle.to_numpy() if labels is not None else None
    st.plotly_chart(line_chart(plot_history.cycle, predictions, truth, family), width="stretch")
    st.caption("Every point uses only its own past. At most 300 evenly spaced endpoints are displayed and exported; the final observation is always included. The line is not smoothed or forced to decrease.")
    choices = [f"s{i+1}" for i in engine.prep.active]
    sensor = st.selectbox("Sensor history", choices)
    sf = go.Figure(go.Scatter(x=history.cycle, y=history[sensor], mode="lines", line=dict(color="#142D42", width=2)))
    sf.update_layout(height=220, margin=dict(l=20, r=20, t=10, b=10), xaxis_title="Observed cycle", yaxis_title=f"{sensor} · source units", template="plotly_white")
    st.plotly_chart(sf, width="stretch")
    export_frame = pd.DataFrame({"unit": unit, "cycle": plot_history.cycle.to_numpy(), "predicted_rul_cycles": predictions})
    if truth is not None:
        export_frame["actual_rul_cycles"] = truth
    dl1, dl2, dl3 = st.columns(3)
    dl1.download_button("Download predictions · CSV", export_frame.to_csv(index=False).encode(), f"{subset}-engine-{unit}.csv", "text/csv")
    dl2.download_button("Download inspection · JSON", json.dumps(result, indent=2).encode(), "inspection.json", "application/json")
    dl3.download_button("Download trajectory · PNG", png_chart(truth, predictions, plot_history.cycle, family), "rul-trajectory.png", "image/png")

with stress:
    st.subheader("One engine. Two measurement conditions.")
    st.write("Change the measurement, not the engine. Both GRU families see the same corrupted window.")
    a, b = st.columns(2)
    with a:
        condition = st.selectbox("Measurement condition", CONDITIONS, index=1)
    with b:
        channel = st.selectbox("Affected sensor", [int(c) for c in engine.prep.active], format_func=lambda c: f"s{c+1}", disabled=condition == "missing3")
    if condition == "missing3":
        affected_channels = np.random.default_rng(20261007).permutation(engine.prep.active)[:3]
        channel = int(affected_channels[0])
        st.caption("Removed channels: " + ", ".join(f"s{c+1}" for c in affected_channels) + f". Chart shows s{channel+1}.")
    st.caption("Interactive scenario: user-selected sensor. Published benchmark channels use the frozen endpoint-seed manifest. For missing3, three deterministic channels are used.")
    raw_window = engine.prep.window_from_rows(rows)
    changed = perturb(raw_window[None], condition, [20261007], engine.prep.active, channel=None if condition == "missing3" else channel)[0]
    valid = raw_window[:, 45] > 0
    observed_cycles = history.cycle.to_numpy()[-int(valid.sum()):]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=observed_cycles, y=raw_window[valid, channel], name="Original", line=dict(color="#142D42")))
    fig.add_trace(go.Scatter(x=observed_cycles, y=changed[valid, channel], name="After corruption / imputation", line=dict(color="#D97706", dash="dash")))
    fig.update_layout(height=300, template="plotly_white", margin=dict(l=20, r=20, t=20, b=20), xaxis_title="Observed cycle", yaxis_title="Training-condition standard deviations", legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig, width="stretch")
    comparisons = []
    comparison_models = {}
    for name in ("gru", "robust_gru"):
        member = load_engine(str(folder), json.dumps(registry, sort_keys=True), subset, name)
        comparison_models[name] = {"model_id": member.model_id, "assets": [a["sha256"] for a in member.predictor.spec["assets"]]}
        original = member.predictor.predict(raw_window[None])[0]
        disturbed = member.predictor.predict(changed[None])[0]
        comparisons.append({"Model": name, "Clean RUL": float(original), "Stressed RUL": float(disturbed), "Change (cycles)": float(disturbed-original)})
    st.dataframe(pd.DataFrame(comparisons).round(2), hide_index=True, width="stretch")
    stress_export = {"subset": subset, "unit": unit, "cycle": cycle, "condition": condition, "selected_sensor": f"s{channel+1}", "scenario": "interactive, not aggregate benchmark", "seed": 20261007, "affected_sensors": [f"s{int(c)+1}" for c in affected_channels] if condition == "missing3" else [f"s{channel+1}"], "models": comparison_models, "comparisons": comparisons}
    st.download_button("Download stress comparison · JSON", json.dumps(stress_export, indent=2).encode(), "stress-comparison.json", "application/json")
    st.markdown('<div class="note">A stable prediction is not automatically a correct prediction. Compare the error against the reference value and the full benchmark.</div>', unsafe_allow_html=True)

with evidence:
    st.subheader("Evidence across all four scenarios")
    summary = []
    for s, row in results["subsets"].items():
        m = row["models"][row["default"]]["conditions"]["clean"]
        h = row["hypothesis"]
        summary.append({"Scenario": s, "Default model": row["default"], "Engines": row["engines"], "Clean RMSE": round(m["rmse"], 2), "Clean MAE": round(m["mae"], 2), "Robustness point target": "Met" if h["point_target_met"] else "Not met"})
    st.dataframe(pd.DataFrame(summary), hide_index=True, width="stretch")
    st.caption("Defaults were chosen on validation only. The robustness target compares the two GRU families, regardless of which model is the default.")
    chart = go.Figure()
    colors = {"mean": "#94A3B8", "ridge": "#6366F1", "boost": "#3B82F6", "gru": "#D97706", "robust_gru": "#0D9488"}
    for name, row in results["subsets"][subset]["models"].items():
        chart.add_trace(go.Scatter(x=list(CONDITIONS), y=[row["conditions"][c]["rmse"] for c in CONDITIONS], name=name, mode="lines+markers", line=dict(color=colors[name], dash={"mean": "dot", "ridge": "dash", "boost": "dashdot", "gru": "longdash", "robust_gru": "solid"}[name])))
    chart.update_layout(height=380, template="plotly_white", yaxis_title="RMSE (cycles)", legend=dict(orientation="h", y=1.15))
    st.plotly_chart(chart, width="stretch")
    st.caption("Point targets are not confidence guarantees. Intervals resample complete engines; stressed copies do not increase sample size.")
    st.link_button("Open complete results and failure analysis", "https://github.com/numann44/engine-health-monitoring/blob/main/docs/results/STUDY_V1.md")

with st.expander("Model identity and reproducibility"):
    st.json({"model_id": result["model_id"], "assets": result["assets"], "study": registry["declaration_digest"], "target": "uncapped operational cycles"})
st.divider()
st.caption("NASA C-MAPSS · simulation-data research · code by Ahmet Numan Şahin · no NASA endorsement")
