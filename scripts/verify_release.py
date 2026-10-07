"""Read-only parity and integrity audit of a completed local release bundle."""
from pathlib import Path
import json

import numpy as np
import torch

from engine_health.common import atomic_json, digest, identity, now, read_json
from engine_health.data import SUBSETS, load_table
from engine_health.inference import Engine
from engine_health.inference.release import release_root
from engine_health.preprocessing.stress import CONDITIONS

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    study = ROOT / 'outputs/study-v1'
    frozen = read_json(study / 'frozen-models.json')
    results = read_json(study / 'results.json')
    assert set(frozen['subsets']) == set(SUBSETS)
    assert frozen['digest'] == identity({k: v for k, v in frozen.items() if k != 'digest'})
    folder = release_root(ROOT)
    registry = read_json(folder / 'registry.json')
    checksums = read_json(folder / 'checksums.json')
    for relative, checksum in checksums.items():
        assert digest(folder / relative) == checksum, relative
    assert registry['study_registry_digest'] == frozen['digest']
    records = []
    fits, neural = 0, 0
    qa = ROOT / 'outputs/ui-qa'
    qa.mkdir(parents=True, exist_ok=True)
    for subset in SUBSETS:
        summaries = list((study / subset).glob('*/summary.json'))
        assert len(summaries) == 13
        fits += len(summaries)
        for path in summaries:
            summary = read_json(path)
            model = path.parent / ('selected.pt' if 'gru' in summary['family'] else 'model.joblib')
            assert digest(model) == summary['selected_sha256']
            assert summary['provenance']['declaration_digest'] == frozen['declaration_digest']
            if 'gru' in summary['family']:
                neural += 1
                checkpoint = torch.load(model, map_location='cpu', weights_only=True)
                assert checkpoint['epoch'] == summary['best_epoch']
                assert checkpoint['best'] == summary['validation_score']
                assert 1 <= summary['epochs'] <= 100
                assert len(read_json(path.parent / 'history.json')) == summary['epochs']
        predictions = study / subset / 'predictions.npz'
        assert digest(predictions) == results['subsets'][subset]['predictions_sha256']
        raw = np.load(predictions)
        df = load_table(ROOT / 'data', subset, 'test')
        manifest = read_json(study / subset / 'test-stress-manifest.json')
        for family in registry['subsets'][subset]['models']:
            engine = Engine(folder, registry, subset, family)
            for ix in (0, len(raw['y'])//2, len(raw['y'])-1):
                unit = int(raw['unit'][ix])
                endpoint = manifest['endpoints'][ix]
                assert endpoint['unit'] == unit
                for condition in CONDITIONS:
                    result = engine.inspect(df, unit, condition=condition, seed=endpoint['seed'])
                    expected = float(raw[f'{family}__{condition}'][ix])
                    error = abs(result['rul_cycles']-expected)
                    assert np.isclose(result['rul_cycles'], expected, atol=2e-4, rtol=1e-5), (subset, family, condition, error)
                    records.append({'subset': subset, 'family': family, 'unit': unit, 'condition': condition, 'absolute_difference_cycles': error})
        if subset == 'FD001':
            group = df.loc[df.unit == 1]
            group.to_csv(qa / 'official-FD001-engine-1.csv', index=False)
            missing = group.copy()
            missing.loc[missing.index[-5:], 's2'] = np.nan
            missing.to_csv(qa / 'missing-sensor.csv', index=False)
            invalid = group.copy()
            invalid.iloc[1, 1] = invalid.iloc[0, 1]
            invalid.to_csv(qa / 'duplicate-cycle.csv', index=False)
    assert fits == 52 and neural == 24
    result = {'verified_at': now(), 'fits': fits, 'neural_runs': neural,
              'verified_bundle_files': len(checksums), 'scalar_to_batch_parity_cases': len(records),
              'max_absolute_difference_cycles': max(r['absolute_difference_cycles'] for r in records),
              'registry_sha256': digest(folder / 'registry.json'), 'checks': records,
              'note': 'Local artifact integrity and common inference parity. This is not a hosted-browser or Linux execution check.'}
    atomic_json(ROOT / 'docs/results/ARTIFACT_VERIFICATION.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'checks'}, indent=2))


if __name__ == '__main__':
    main()
