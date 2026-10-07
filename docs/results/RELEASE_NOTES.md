# v0.1.0 — Controlled engine-health robustness study

A bounded research release on simulated NASA C-MAPSS engines. All 24 from-scratch GRU runs, 24 classical candidate fits and four constant references are complete. All four model selections were frozen before official test evaluation; the release opens no further architecture or seed search.

| Scenario | Validation-selected default | Clean RMSE (cycles) | Robustness point criterion |
|---|---|---:|---|
| FD001 | boost | 23.33 | Not met |
| FD002 | gru | 26.53 | Not met |
| FD003 | robust_gru | 29.03 | Not met |
| FD004 | robust_gru | 28.32 | Not met |

The robustness criterion compares the two three-seed GRU ensembles: at least 10% lower mean mild-stress RMSE and no more than 5% higher clean RMSE. Engine-bootstrap intervals and negative outcomes accompany the point estimates. This is not evidence of real-aircraft reliability.

## Assets

`engine-health-v0.1.0.zip` contains all validation-selected model families, preprocessing and measured evidence for four scenarios. It contains no raw NASA trajectories. The registry identifies every model file by SHA-256.

Archive SHA-256: `e47bf5465c282221e699b7593bfe5e72d78b3334dcb18f0ffd5c2c53af5506ef`

Registry SHA-256: `31cea7eb65594fa2f22a029a9ec10ae74fa881840a053c20929e1708dc89d829`

`study-v1-frozen-source.tar.gz` preserves the exact original training/evaluation source before deployment conveniences were added.

Frozen source SHA-256: `89b12d39885a0cb89dfaa11213e3825d9e1db717658d9408c1ec78f2e4c8c8fe`

Code is MIT licensed; NASA data have separate source terms and are downloaded from the official source. See the README, measured report, failure analysis, model card and reproduction guide. Hosted deployment and Linux checks must be recorded separately; this file alone is not proof they passed.
