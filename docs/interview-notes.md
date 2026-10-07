# Technical interview notes

## A 60-second explanation

I built a remaining-life forecasting experiment on NASA's simulated turbofan trajectories. The engineering challenge was preventing leakage across engines and across time, then measuring whether a model learned a robust relationship rather than exploiting one sensor. I compared classical methods against two identical GRU architectures, one trained on clean windows and the other on controlled measurement corruptions. Each neural family averages three fixed seeds. I froze all four subset selections before opening official test outcomes and reported errors, paired bootstrap intervals and negative results. The dashboard lets a reviewer reproduce a prediction and corrupt a sensor while inspecting exact model identities.

## Decisions worth explaining

- **Whole engines, not random windows.** Adjacent windows share nearly all their measurements. Randomly splitting them would give an unrealistically easy validation task.
- **Causal preprocessing.** An estimate at cycle t uses only rows at or before t. Missing channels are never filled with future measurements. Test labels are kept outside inference.
- **Operating regimes.** FD002/FD004 vary by operating condition. Six clusters and their normalization are fitted only on training engines, using three operating settings. Separate subset models avoid claiming cross-scenario generalization.
- **Uncapped targets.** Remaining life is not truncated at a convenient ceiling. That makes early-life errors visible and rules out casual comparison with capped-label leaderboards.
- **Equal engine contribution.** Long-lived trajectories must not dominate merely by having more windows. Training weights sum equally per engine.
- **One selection rule.** Half clean RMSE and half mean mild-stress RMSE selects classical parameters and every neural checkpoint. Seed averaging avoids cherry-picking a lucky initialization.
- **A controlled intervention.** The robust and ordinary GRUs begin with identical paired initial weights. The intended difference is exposure to synthetic measurement corruption, not an extra architecture search.
- **An honest negative result.** Training augmentation need not help all scenarios. A stable prediction can also be consistently wrong. The declared criterion is at least 10% lower mild-stress RMSE with no more than 5% clean degradation, evaluated separately in every scenario.
- **Uncertainty.** Bootstrap draws resample engines. Corrupted copies and multiple life fractions are correlated, not additional independent engines. Aggregate error intervals are not confidence bounds on one engine's RUL.
- **Deployment parity.** CLI, evaluation and dashboard share preprocessing and model prediction. Serialized weights, preprocessors and the bundle have checksum identities. No training runs on the demo server.

## What I would improve in a separate future study

An independently declared study could investigate mismatch between the four validation life fractions and official test stopping points, real acquisition faults or different target formulations. None of those ideas change the selection after seeing this release's test results. Independent physical engine data and a domain-specific validation program would be needed for practical maintenance claims.

## Evidence to open during discussion

The [full measured report](results/STUDY_V1.md), [failure analysis](results/FAILURE_ANALYSIS.md), [fixed protocol](experiment-protocol.md), [data audit](results/DATA_AUDIT.json) and exact-resume/causality tests in `tests/`.
