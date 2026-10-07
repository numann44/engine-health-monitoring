"""Make a cover from actual measured errors, never illustrative accuracy."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from engine_health.common import read_json
from engine_health.preprocessing.stress import MILD

root = Path(__file__).resolve().parents[1]
results = read_json(root / 'outputs/study-v1/results.json')
fig = plt.figure(figsize=(16, 8), dpi=100, facecolor='#f8fafc')
fig.text(.065, .91, 'TIME SERIES  /  PREDICTIVE MAINTENANCE', color='#0d9488', fontsize=12, weight='bold')
fig.text(.065, .82, 'Engine Health Monitoring', color='#142d42', fontsize=34, weight='bold')
fig.text(.065, .755, 'A controlled study of remaining-life prediction under sensor degradation.', color='#526575', fontsize=16)
ax = fig.add_axes([.075, .22, .865, .43], facecolor='#f8fafc')
names = list(results['subsets'])
x = np.arange(len(names))
for offset, (family, color, label) in zip((-.18, .18), [('gru', '#d97706', 'Standard GRU'), ('robust_gru', '#0d9488', 'Corruption-trained GRU')]):
    values = [np.mean([results['subsets'][s]['models'][family]['conditions'][c]['rmse'] for c in MILD]) for s in names]
    bars = ax.bar(x+offset, values, width=.32, color=color, label=label)
    ax.bar_label(bars, fmt='%.1f', padding=6, color='#142d42', fontsize=12)
ax.set_xticks(x, names, fontsize=13)
ax.set_ylabel('Mean mild-stress RMSE (cycles)', color='#526575', fontsize=12)
ax.spines[['top', 'right', 'left']].set_visible(False)
ax.spines['bottom'].set_color('#d1dae3')
ax.grid(axis='y', alpha=.12)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(0, 1.2), ncol=2, fontsize=12)
ax.margins(y=.2)
fig.text(.065, .11, '4 scenarios  ·  24 GRU runs  ·  3-seed ensembles  ·  707 test engines', color='#142d42', fontsize=14)
fig.text(.065, .06, 'NASA C-MAPSS simulation data · uncapped targets · negative results retained', color='#526575', fontsize=11)
fig.savefig(root / 'docs/images/social-preview.png', facecolor=fig.get_facecolor())
plt.close(fig)
