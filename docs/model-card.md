# Model card

**Status:** frozen training and official-test evaluation complete. External publication status is recorded in the delivery ledger.

**Task:** remaining useful life regression, in operational cycles, from a 30-cycle multivariate sensor window. Targets are uncapped and outputs nonnegative.

**Data:** NASA C-MAPSS FD001–FD004, separate simulation scenarios. Whole-engine training/validation split, official test held out from model selection. Source catalog license is unspecified; code license does not cover data.

**Candidates:** mean, Ridge, histogram gradient boosting, GRU and corruption-trained GRU. Neural models use random initialization and three fixed seeds; family predictions average all three. Default per-subset model is chosen on validation only.

**Intended use:** reproducible ML research, learning and engineering portfolio demonstration.

**Limitations:** no real-aircraft validation; separate models per subset; no universal health percentage; RUL does not equal hours. Synthetic sensor faults do not model every acquisition problem. Known settings, sampling and feature conventions are required. Validation life fractions can differ from official test stopping-point distributions. Individual prediction intervals are not calibrated; bootstrap intervals describe aggregate evaluation uncertainty only.

**Publication gate:** all declared runs, source/asset verification, frozen evaluation, Linux checks and actual hosted demo behavior. Failing the robustness hypothesis is reported rather than hidden or followed by undeclared searches.

## Measured defaults

| Scenario | Validation-selected default | Clean RMSE [95% CI] | MAE | >20-cycle overestimates |
|---|---|---:|---:|---:|
| FD001 | boost | 23.33 [19.09, 27.52] | 16.39 | 27/100 |
| FD002 | gru | 26.53 [23.29, 29.82] | 17.74 | 36/259 |
| FD003 | robust_gru | 29.03 [22.81, 34.94] | 19.00 | 27/100 |
| FD004 | robust_gru | 28.32 [25.07, 31.53] | 19.84 | 57/248 |

## Robustness hypothesis

| Scenario | Mild RMSE reduction | Clean RMSE increase | Point criterion |
|---|---:|---:|---|
| FD001 | 4.7% | -3.2% | Not met |
| FD002 | 5.1% | -4.3% | Not met |
| FD003 | -1.5% | 0.5% | Not met |
| FD004 | -4.8% | 8.4% | Not met |

See [the full report](results/STUDY_V1.md) for all methods, stress conditions, paired intervals, model sizes and timing.
