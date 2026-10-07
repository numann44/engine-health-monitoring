# Architecture

The Python package is the source of truth; the Streamlit UI and CLI call the same `Engine.inspect` method. No notebook execution is needed.

1. Verified archive → structural audit → versioned engine split and endpoint bank.
2. Train-only condition normalizer → fixed causal windows and classical summaries.
3. Serialized training → atomic resumable checkpoints → validation-only selection.
4. Freeze all four selections → official test and fixed measurement stresses.
5. Checksum-pinned release bundle → common CPU inference → UI/CLI exports.

The preprocessor is serialized as JSON arrays, including fit engine IDs. Model artifacts contain no raw sensor trajectories. Neural checkpoints are loaded with `weights_only=True`. Classical joblib files are loaded only after verification against the trusted project release manifest; users cannot upload model files.

Checkpoint provenance includes source, configuration, split, environment and operating device. Atomic replacement protects the previous checkpoint from interrupted writes. The OS process lease prevents a second heavy runner. The diagnostic PID is never treated as proof of lock ownership.

## Input/output contract

CSV columns, in order: `unit,cycle,setting_1,setting_2,setting_3,s1,...,s21`. Numeric unit/cycle values must be positive integers; each engine's cycles must increase consecutively. Settings must be finite. Empty sensor cells are supported; infinite values are rejected. Up to 10 MiB / 50,000 rows.

Inference returns subset, engine, final observed cycle, RUL in cycles, model ID, asset checksums, history length, missing-reading count, latency, corruption settings and warnings. It does not receive actual RUL labels. Missingness and short history are displayed without inventing a confidence score.

## Boundaries

Different subsets have separate models. Test labels are accessed only by the evaluator and, after release, retrospective demo charts. Future rows are excluded before constructing each prediction. Uploaded files remain in session memory and are not logged or saved.
