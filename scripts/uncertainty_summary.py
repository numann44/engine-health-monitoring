"""Descriptive paired engine bootstrap of the fixed research criterion."""
from pathlib import Path

import numpy as np

from engine_health.common import atomic_json, digest, read_json
from engine_health.preprocessing.stress import MILD

ROOT = Path(__file__).resolve().parents[1]


def main():
    study = ROOT / 'outputs/study-v1'
    results = read_json(study / 'results.json')
    output = {'role': 'Descriptive uncertainty; does not select, tune or redefine the frozen point criterion',
              'repetitions': 2000, 'seed': 20261007, 'unit': 'engine', 'interval': 'percentile 95%; no multiplicity adjustment', 'subsets': {}}
    lines = ['# Uncertainty around the robustness criterion', '',
             'All methods and checkpoint selections were frozen before official test evaluation. These intervals add descriptive uncertainty to the original point criterion; they do not redefine it or trigger new training. Both models and every condition use the same resampled engine indices. There are 2,000 draws, seed 20261007. Intervals are percentile intervals, without a four-subset multiplicity adjustment.', '',
             '| Scenario | Mild RMSE reduction [95% interval] | Clean RMSE increase [95% interval] |', '|---|---:|---:|']
    for subset, row in results['subsets'].items():
        path = study / subset / 'predictions.npz'
        assert digest(path) == row['predictions_sha256']
        raw = np.load(path)
        assert len(np.unique(raw['unit'])) == len(raw['y'])
        rng = np.random.default_rng(20261007)
        draws = []
        for _ in range(2000):
            ix = rng.integers(len(raw['y']), size=len(raw['y']))
            errors = {family: {c: np.sqrt(np.mean((raw[f'{family}__{c}'][ix]-raw['y'][ix])**2))
                              for c in ('clean', *MILD)} for family in ('gru', 'robust_gru')}
            reduction = 1-np.mean([errors['robust_gru'][c] for c in MILD])/np.mean([errors['gru'][c] for c in MILD])
            increase = errors['robust_gru']['clean']/errors['gru']['clean']-1
            draws.append((reduction, increase))
        intervals = np.quantile(draws, [.025, .975], axis=0)
        h = row['hypothesis']
        output['subsets'][subset] = {'engines': len(raw['y']), 'mild_rmse_reduction_ci95': intervals[:, 0].tolist(),
                                    'clean_rmse_increase_ci95': intervals[:, 1].tolist(),
                                    'point_target_met': h['point_target_met']}
        lines.append(f"| {subset} | {h['mild_rmse_reduction']:.1%} [{intervals[0,0]:.1%}, {intervals[1,0]:.1%}] | {h['clean_rmse_increase']:.1%} [{intervals[0,1]:.1%}, {intervals[1,1]:.1%}] |")
    lines += ['', 'Positive reduction is better; positive clean increase is worse. The target is reduction ≥10% and clean increase ≤5%. A point criterion passing does not establish that both population limits hold. Aggregate bootstrap uncertainty is not a calibrated prediction interval for an individual motor.']
    atomic_json(ROOT / 'docs/results/hypothesis-uncertainty.json', output)
    (ROOT / 'docs/results/HYPOTHESIS_UNCERTAINTY.md').write_text('\n'.join(lines)+'\n')
    print('Paired hypothesis intervals generated without changing any model or selection')


if __name__ == '__main__':
    main()
