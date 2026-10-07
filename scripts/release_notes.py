"""Generate factual GitHub release notes from the measured result registry."""
from pathlib import Path

from engine_health.common import digest, read_json

root = Path(__file__).resolve().parents[1]
results = read_json(root / 'outputs/study-v1/results.json')
manifest = read_json(root / 'assets/model-registry.json')
lines = ['# v0.1.0 — Controlled engine-health robustness study', '',
         'A bounded research release on simulated NASA C-MAPSS engines. All 24 from-scratch GRU runs, 24 classical candidate fits and four constant references are complete. All four model selections were frozen before official test evaluation; the release opens no further architecture or seed search.', '',
         '| Scenario | Validation-selected default | Clean RMSE (cycles) | Robustness point criterion |', '|---|---|---:|---|']
for subset, row in results['subsets'].items():
    lines.append(f"| {subset} | {row['default']} | {row['models'][row['default']]['conditions']['clean']['rmse']:.2f} | {'Met' if row['hypothesis']['point_target_met'] else 'Not met'} |")
lines += ['', 'The robustness criterion compares the two three-seed GRU ensembles: at least 10% lower mean mild-stress RMSE and no more than 5% higher clean RMSE. Engine-bootstrap intervals and negative outcomes accompany the point estimates. This is not evidence of real-aircraft reliability.', '',
          '## Assets', '', '`engine-health-v0.1.0.zip` contains all validation-selected model families, preprocessing and measured evidence for four scenarios. It contains no raw NASA trajectories. The registry identifies every model file by SHA-256.', '',
          f"Archive SHA-256: `{manifest['archive_sha256']}`", '',
          f"Registry SHA-256: `{manifest['registry_sha256']}`", '',
          '`study-v1-frozen-source.tar.gz` preserves the exact original training/evaluation source before deployment conveniences were added.', '',
          f"Frozen source SHA-256: `{digest(root / 'outputs/study-v1-frozen-source.tar.gz')}`", '',
          'Code is MIT licensed; NASA data have separate source terms and are downloaded from the official source. See the README, measured report, failure analysis, model card and reproduction guide. Hosted deployment and Linux checks must be recorded separately; this file alone is not proof they passed.']
(root / 'docs/results/RELEASE_NOTES.md').write_text('\n'.join(lines)+'\n')
