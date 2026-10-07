# Uncertainty around the robustness criterion

All methods and checkpoint selections were frozen before official test evaluation. These intervals add descriptive uncertainty to the original point criterion; they do not redefine it or trigger new training. Both models and every condition use the same resampled engine indices. There are 2,000 draws, seed 20261007. Intervals are percentile intervals, without a four-subset multiplicity adjustment.

| Scenario | Mild RMSE reduction [95% interval] | Clean RMSE increase [95% interval] |
|---|---:|---:|
| FD001 | 4.7% [1.7%, 7.9%] | -3.2% [-6.3%, -0.3%] |
| FD002 | 5.1% [1.9%, 7.8%] | -4.3% [-7.1%, -0.8%] |
| FD003 | -1.5% [-4.9%, 1.1%] | 0.5% [-1.4%, 3.0%] |
| FD004 | -4.8% [-10.7%, 0.9%] | 8.4% [2.2%, 15.1%] |

Positive reduction is better; positive clean increase is worse. The target is reduction ≥10% and clean increase ≤5%. A point criterion passing does not establish that both population limits hold. Aggregate bootstrap uncertainty is not a calibrated prediction interval for an individual motor.
