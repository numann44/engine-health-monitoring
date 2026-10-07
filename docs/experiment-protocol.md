# Predeclared study v1

This protocol was committed and content-hashed locally before training and official test outcomes. It was not submitted to an external preregistration service; public upload was delayed by repository access. Configuration, source hashes, dependency versions, split hashes and the code commit are recorded in the declaration. The queue refuses incompatible source, environment or checkpoints. Tests of software behavior use synthetic data or official training engines only.

## Task and separation

Fit separate models to FD001, FD002, FD003 and FD004. Use the final observed window of each official test engine and the corresponding supplied RUL value. Main targets and metrics are **uncapped cycles**. No test-based retuning, RUL cap search or seed selection is allowed. Engine lifetime and cycle count are never input features.

Normalizers see only the training partition. FD002/FD004 use six K-means centers fitted on standardized operating settings, random_state 20261007, n_init 10; per-condition sensor means and scales are fitted on those training assignments. Other subsets use one condition. Standard deviations below 1e-5 are replaced by one. Perturbation channels must vary in at least one training condition. No test normalization is fitted.

Windows contain 30 rows of 21 standardized sensors, three standardized operating settings, 21 missingness flags and one real-row flag (46 values per row). Left padding is zero with real-row flag zero. Missing sensors are causally filled inside the 30-cycle window, beginning at the training-condition mean. Future values and data preceding this finite window are not used. The same window contract is used in training and inference.

## Fixed comparisons

- Mean reference: motor-weighted mean training RUL.
- Ridge: alpha 0.1, 1, 10; training-only feature scaling.
- Histogram Gradient Boosting: max_leaf_nodes 15, 31, 63; max_iter 200, learning_rate .05, l2_regularization 1, early_stopping False, random_state 42.
- GRU and robust GRU: two layers, 64 hidden units, dropout .1, 64→32→1 ReLU/Softplus head. AdamW .001, weight decay .0001, Huber delta .1 on RUL/100, gradient clipping 1, batch 128. Maximum 100 epochs, 15 stale epochs. A new best requires a score improvement >1e-8.

Classical features are last, mean, standard deviation and least-squares slope of the 45 non-padding input channels (180 values); padding is excluded. The regression outputs are clipped only below zero. Every engine has equal total training loss/sample weight.

Exactly 24 neural runs (two families × seeds 42,43,44 × four subsets), 24 classical fits and four constants. Identical seeds produce identical initial weights in paired GRUs. Shuffling, augmentation and PyTorch dropout use separate reproducible random streams. No architecture or parameter search outside this list.

## Validation and stress

Validation uses floor(25%, 50%, 75%, 90%) of each held-out training motor's lifetime, minimum one cycle. Hash-derived seeds bind perturbations to each endpoint. Source motors, not endpoints or corruptions, are the statistical units.

Fixed conditions: clean; missing one sensor; Gaussian noise sigma .25; last 10 cycles stuck; missing three sensors; Gaussian noise sigma .5; bias +.5; bias −.5. Noise/bias are in training-condition standard deviation units, applied to one deterministically selected variable channel. Missing channels cover the entire real window. Stuck values copy the first value of the affected interval. No fault flag is supplied for noise, bias or stuck readings. Operating settings are never corrupted.

Robust training keeps 50% of windows clean. The rest uniformly choose one missing channel, one noisy channel with sigma uniform[0,.5), or one stuck channel over 5–10 final rows. Missing-three and constant bias are not trained. The common missing-data policy is used by every model.

Selection score: .5 clean RMSE + .5 mean RMSE over missing1, noise025, stuck10. This selects checkpoints and classical candidates. Three selected GRU seeds are averaged, not cherry-picked. The lowest validation-score family per subset is the demo default, with warm CPU inference time breaking equal scores. All four subset selections freeze before opening any official test outcome.

## Evaluation and stopping

Report each model, each condition, individual neural members, RMSE, MAE, signed bias, overestimation frequency/magnitude, >20-cycle overestimates, warm CPU timing (30 trials) and asset size. Compute 95% percentile intervals from 2,000 engine bootstrap samples, seed 20261007; paired differences retain the same engine draws.

The hypothesis passes its point target on a subset if robust GRU mean mild-stress RMSE is at least 10% lower and clean RMSE at most 5% higher than standard GRU. Intervals accompany point results; they are not replaced by a success badge. No universal pass is inferred from averages.

Best, median and worst absolute-error examples are selected mechanically after evaluation, for explanation only. Training and evaluation complete the bounded study regardless of hypothesis outcome. A different method requires a separate future protocol, with reused tests labeled development-inspected.
