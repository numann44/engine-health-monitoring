"""Publish measured presentation artifacts after the fixed study is complete."""
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from engine_health.common import atomic_json, digest, read_json
from engine_health.data import SUBSETS, load_table
from engine_health.inference import Engine
from engine_health.evaluation.report import COLORS
from engine_health.preprocessing.stress import CONDITIONS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/study-v1'
DOCS = ROOT / 'docs'
IMAGES = DOCS / 'images'


def main():
    results = read_json(OUT / 'results.json')
    registry = read_json(OUT / 'frozen-models.json')
    if set(results['subsets']) != set(SUBSETS):
        raise ValueError('All four evaluations must finish first')
    IMAGES.mkdir(parents=True, exist_ok=True)
    rows, hypotheses, experiments, failures = [], [], [], []
    for subset in SUBSETS:
        record = results['subsets'][subset]
        family = record['default']
        m = record['models'][family]['conditions']['clean']
        lo, hi = m['uncertainty']['ci95']
        h = record['hypothesis']
        rows.append(f"| {subset} | {family} | {m['rmse']:.2f} [{lo:.2f}, {hi:.2f}] | {m['mae']:.2f} | {m['over20_count']}/{record['engines']} |")
        hypotheses.append(f"| {subset} | {h['mild_rmse_reduction']:.1%} | {h['clean_rmse_increase']:.1%} | {'Met' if h['point_target_met'] else 'Not met'} |")
        fig, ax = plt.subplots(figsize=(11, 4), layout='constrained')
        styles = {'mean': ':', 'ridge': '--', 'boost': '-.', 'gru': (0, (5, 2)), 'robust_gru': '-'}
        for name, item in record['models'].items():
            ax.plot(CONDITIONS, [item['conditions'][c]['rmse'] for c in CONDITIONS], marker='o', label=name, color=COLORS[name], linestyle=styles[name])
        ax.set(title=f'{subset} · fixed sensor stress conditions', ylabel='RMSE (cycles)')
        ax.tick_params(axis='x', rotation=20)
        ax.legend(frameon=False, ncol=3)
        ax.grid(alpha=.15)
        fig.savefig(IMAGES / f'{subset.lower()}-robustness.png', dpi=160)
        plt.close(fig)
        raw_path = OUT / subset / 'predictions.npz' 
        if digest(raw_path) != record['predictions_sha256']:
            raise ValueError('Changed prediction evidence')
        p = np.load(raw_path)
        ix = int(np.argmax(np.abs(p[family+'__clean']-p['y'])))
        unit = int(p['unit'][ix])
        data = load_table(ROOT / 'data', subset, 'test')
        group = data.loc[data.unit == unit]
        engine = Engine(ROOT, registry, subset)
        original = group.iloc[:, 2:26].to_numpy()
        windows = np.stack([engine.prep.window_from_rows(original[:stop]) for stop in range(1, len(original)+1)])
        predictions = engine.predictor.predict(windows)
        truth = p['y'][ix] + group.cycle.max()-group.cycle.to_numpy()
        fig, ax = plt.subplots(figsize=(10, 4.5), layout='constrained')
        ax.plot(group.cycle, truth, '--', color='#475569', label='Retrospective actual RUL')
        ax.plot(group.cycle, predictions, color=COLORS[family], label=f'{family} prediction')
        ax.scatter([group.cycle.iloc[-1]], [predictions[-1]], color='#d97706', zorder=3)
        ax.set(xlabel='Observed cycle', ylabel='Remaining cycles', title=f'{subset} · largest absolute final-observation error · engine {unit}')
        ax.grid(alpha=.15)
        ax.legend(frameon=False)
        fig.savefig(IMAGES / f'{subset.lower()}-failure.png', dpi=160)
        plt.close(fig)
        delta = float(predictions[-1]-p['y'][ix])
        failures += [f'## {subset} · engine {unit}', '', f"The validation-selected {family} predicts **{predictions[-1]:.1f} cycles**, while the reference is **{p['y'][ix]:.0f}**: a **{delta:+.1f}-cycle** error. This is the largest absolute error among {record['engines']} final-observation predictions. {'The optimistic estimate could delay a maintenance decision.' if delta > 0 else 'The pessimistic estimate could cause premature maintenance.'} These are implications of the signed error, not observed real-world outcomes.", '', f'![Worst final-observation error](../images/{subset.lower()}-failure.png)', '', 'Every trajectory point is recomputed from its available past only. Earlier points are descriptive retrospective examples; the primary test metric uses the last point. The failure was selected after evaluation for explanation, never to change weights or hyperparameters. A sensor plot alone cannot establish the physical cause of an error.', '']
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
        for family_name, color in [('gru', '#d97706'), ('robust_gru', '#0d9488')]:
            for seed, style in zip((42, 43, 44), ('-', '--', ':')):
                directory = OUT / subset / f'{family_name}-seed{seed}'
                summary = read_json(directory / 'summary.json')
                history = read_json(directory / 'history.json')
                experiments.append({'subset': subset, 'family': family_name, 'seed': seed, 'summary': summary, 'history': history})
                # History keys are checked rather than inferred from log text.
                epochs = [r['epoch'] for r in history]
                axes[0].plot(epochs, [r['loss'] for r in history], color=color, linestyle=style, label=f'{family_name} · {seed}')
                axes[1].plot(epochs, [r['validation_score'] for r in history], color=color, linestyle=style)
        axes[0].set(xlabel='Epoch', ylabel='Scaled Huber training loss', title=f'{subset} · fixed runs')
        axes[1].set(xlabel='Epoch', ylabel='Validation selection score (cycles)', title='Checkpoint selection evidence')
        axes[0].legend(frameon=False, fontsize=8)
        fig.savefig(IMAGES / f'{subset.lower()}-learning.png', dpi=160)
        plt.close(fig)
    atomic_json(DOCS / 'results/training-history.json', experiments)
    shutil.copy2(OUT / 'frozen-models.json', DOCS / 'results/frozen-models.json')
    shutil.copy2(OUT / 'evaluation-declaration.json', DOCS / 'results/evaluation-declaration.json')
    (DOCS / 'results/FAILURE_ANALYSIS.md').write_text('# Failure analysis\n\nExamples are selected by a deterministic maximum-absolute-error rule. No further experiments are opened by these outcomes.\n\n'+'\n'.join(failures))
    table = '| Scenario | Validation-selected default | Clean RMSE [95% CI] | MAE | >20-cycle overestimates |\n|---|---|---:|---:|---:|\n'+'\n'.join(rows)
    hypothesis_table = '| Scenario | Mild RMSE reduction | Clean RMSE increase | Point criterion |\n|---|---:|---:|---|\n'+'\n'.join(hypotheses)
    summary = table+'\n\n'+hypothesis_table+'\n\nIntervals resample engines 2,000 times. The robustness criterion compares the two three-seed GRU ensembles, even where the default model is classical. Negative clean increases indicate improved clean RMSE. These are uncapped-RUL results, not directly comparable with capped-label leaderboards.\n\n[Full measured report](docs/results/STUDY_V1.md) · [Failure analysis](docs/results/FAILURE_ANALYSIS.md) · [Machine-readable results](docs/results/study-v1.json)'
    readme = ROOT / 'README.md'
    text = readme.read_text()
    start = text.index('## Measured results')
    end = text.index('## Interactive demo', start) if '## Interactive demo' in text[start:] else text.index('## Architecture', start)
    text = text[:start]+'## Measured results\n\n'+summary+'\n\n'+text[end:]
    marker = '> **Study in progress.' if '> **Study in progress.' in text else '> **Measured research release.'
    start = text.index(marker)
    end = text.index('\n\n', start)
    text = text[:start]+'> **Measured research release.** All 52 declared fits and the fixed official-test evaluation are complete. Publication and hosted checks are tracked in [the delivery ledger](docs/DELIVERY.md). The hypothesis is reported separately for each subset; no test-driven retuning follows this release.'+text[end:]
    readme.write_text(text)
    card = DOCS / 'model-card.md'
    text = card.read_text().split('\n## Measured defaults')[0].replace('**Status:** bounded study in progress; no official test results claimed yet.', '**Status:** frozen training and official-test evaluation complete. External publication status is recorded in the delivery ledger.')
    card.write_text(text+'\n## Measured defaults\n\n'+table+'\n\n## Robustness hypothesis\n\n'+hypothesis_table+'\n\nSee [the full report](results/STUDY_V1.md) for all methods, stress conditions, paired intervals, model sizes and timing.\n')
    counts = sum(int(x['hypothesis']['point_target_met']) for x in results['subsets'].values())
    cv = f'''# CV project descriptions

[Repository](https://github.com/numann44/engine-health-monitoring) · [Live demo](https://numan-engine-health-monitoring.streamlit.app) · [Release](https://github.com/numann44/engine-health-monitoring/releases/tag/v0.1.0). This is simulation-data research, not a deployed aircraft maintenance system.

## English

**Engine Health Monitoring — Predictive Maintenance & Robustness**  
Python · PyTorch · scikit-learn · Streamlit · NASA C-MAPSS

- Built a reproducible remaining-useful-life pipeline across four C-MAPSS scenarios (709 training and 707 test engines), using engine-disjoint validation and train-only operating-condition normalization.
- Trained 24 GRUs from scratch and 24 classical candidates; compared three-seed ensembles under missing, noisy, stuck and biased sensors with paired engine-bootstrap uncertainty.
- Implemented resumable training, checksum-verified inference, and an interactive sensor-stress dashboard. The predeclared robustness point criterion was met on {counts}/4 scenarios; retained negative results and failure analysis.

## Türkçe

**Engine Health Monitoring — Kestirimci Bakım ve Sensör Dayanıklılığı**

- NASA C-MAPSS'in dört senaryosunda 709 eğitim ve 707 test motoruyla, motor bazında ayrılmış doğrulama ve yalnız eğitimden öğrenilen normalizasyon kullanan kalan ömür tahmini hattı geliştirdim.
- Sıfırdan 24 GRU ve 24 klasik model adayı eğiterek eksik, gürültülü, takılı ve sapmış sensör koşullarını üç başlangıcın ortalaması ve motor bazında bootstrap belirsizliğiyle karşılaştırdım.
- Eğitime kaldığı yerden devam etme, checksum doğrulamalı model yükleme ve etkileşimli sensör deneyi paneli geliştirdim. Önceden belirlenen dayanıklılık nokta ölçütü {counts}/4 senaryoda sağlandı; olumsuz sonuçları ve hata analizlerini raporladım.
'''
    (DOCS / 'cv-project.md').write_text(cv)
    print('Presentation generated from complete measured results')


if __name__ == '__main__':
    main()
