"""
plot_fig_metric_suite.py
------------------------
One figure per saliency metric, in the layout of fig_head_selection_corrected.png but
with the between-task panel dropped: Describe and Social only, six agents, bars as a
percentage of that metric's own split-half human ceiling.

Seven metrics from Bylinskii et al. (2019): CC, NSS, AUC-Judd, shuffled AUC, KL, SIM,
information gain. Source is task4_metric_suite/run_metric_suite_cv.py, which recomputes
the suite on Task 1's cross-validated heads and stores per-image values so these bars
carry SEMs and the LSTM-vs-Transformer contrast can be tested per metric.

Model order pairs each LSTM with the Transformer that shares its encoder, so the
decoder effect is the within-pair comparison read left to right.

Normalization is per metric family, since "% of ceiling" is only meaningful once each
metric is put on a scale with a defined zero:
  CC, NSS, SIM, IG   value / ceiling                   (chance = 0)
  AUC, sAUC          (value - .5) / (ceiling - .5)     (chance = .5, not 0)
  KL                 ceiling / value                   (lower is better; >100% means
                                                        the model's KL is below the
                                                        human split-half's)

Outputs: plots/head_selection/fig_metric_{CC,NSS,AUC,sAUC,KL,SIM,IG}.png
         plots/head_selection/fig_metric_overview.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared'))
import figstyle as FST


HERE = os.path.dirname(os.path.abspath(__file__))
REBUTTAL = os.path.dirname(HERE)
RES = os.path.join(REBUTTAL, 'task4_metric_suite', 'results')
OUT = os.path.join(HERE, 'head_selection')
os.makedirs(OUT, exist_ok=True)

summ = pd.read_csv(os.path.join(RES, 'metric_summary_cv.csv'))
con = pd.read_csv(os.path.join(RES, 'metric_contrasts_cv.csv'))

COLORS = FST.ALL_COLORS
TICK = {'CNN-Transformer': 'CNN-Trans.', 'ViT-Transformer': 'ViT-Trans.',
        'Molmo': 'Molmo 7B-D', 'Qwen3.5': 'Qwen 9B-3.5'}
ORDER = ['CNN-LSTM', 'CNN-Transformer', 'ViT-LSTM', 'ViT-Transformer', 'Molmo', 'Qwen3.5']
METRICS = ['CC', 'NSS', 'AUC', 'sAUC', 'KL', 'SIM', 'IG']
FULL = {'CC': 'Correlation coefficient (CC)', 'NSS': 'Normalized scanpath saliency (NSS)',
        'AUC': 'AUC-Judd', 'sAUC': 'Shuffled AUC (sAUC)',
        'KL': 'KL divergence (lower is better)', 'SIM': 'Similarity (SIM)',
        'IG': 'Information gain over center prior (IG)'}
TASK_NAME = {'describe': 'Describe Task', 'social': 'Social Task'}
FS_TITLE, FS_TICK, FS_LABEL, FS_LETTER, FS_LEG = 34, 24, 26, 40, 20

# KL and the AUCs need their own y-label: for KL the split-half value is not an upper
# bound at all, and for the AUCs the normalizer is the above-chance range, not the raw
# score. Saying so on the axis is cheaper than a caption nobody reads.
YLAB = {'KL': '% of human split-half KL\n(inverted: higher = lower KL)',
        'AUC': '% of human upperbound\n(above-chance, chance = 0.5)',
        'sAUC': '% of human upperbound\n(above-chance, chance = 0.5)'}
NOTE = {'KL': 'every model scores past 100% because KL(human||model) rewards diffuse maps, '
              'so the split-half value is not a ceiling here — read the ordering, not the level',
        'IG': 'negative = worse at predicting fixation than the center prior alone'}


def normalize(metric, v, sem, ceil):
    """-> (% of ceiling, propagated SEM in the same units)."""
    if metric in ('AUC', 'sAUC'):
        den = ceil - 0.5
        return 100 * (v - 0.5) / den, 100 * sem / den
    if metric == 'KL':
        # pct = 100 * ceil / v, so d(pct)/dv = -100 * ceil / v^2
        return 100 * ceil / v, 100 * ceil * sem / (v ** 2)
    return 100 * v / ceil, 100 * sem / ceil


def panel(ax, metric, task, letter, show_ylabel):
    sub = summ[(summ.metric == metric) & (summ.task == task)].set_index('model')
    ceil = float(sub['ceiling'].iloc[0])
    vals, errs = [], []
    for m in ORDER:
        r = sub.loc[m]
        p, e = normalize(metric, float(r['value']), float(r['sem']), ceil)
        vals.append(p); errs.append(e)
    x = np.arange(len(ORDER))
    ax.bar(x, vals, 0.62, color=[COLORS[m] for m in ORDER],
           edgecolor='black', linewidth=0.6, zorder=3)
    ax.errorbar(x, vals, yerr=errs, fmt='none', color='black',
                capsize=5, linewidth=1.8, zorder=4)
    ax.axhline(100.0, color='black', linestyle='--', linewidth=2.4, zorder=2)
    ax.axhline(0, color='black', linewidth=0.9, zorder=2)

    # bracket the two within-encoder decoder contrasts
    lo, hi = min(min(vals), 0), max(max(vals), 100)
    span = hi - lo
    for (i, j), pair in [((0, 1), 'CNN-LSTM'), ((2, 3), 'ViT-LSTM')]:
        c = con[(con.task == task) & (con.metric == metric) & (con.lstm == pair)]
        if c.empty:
            continue
        cc = c.iloc[0]
        y = max(vals[i], vals[j]) + span * 0.10
        ax.plot([i, i, j, j], [y, y + span * 0.02, y + span * 0.02, y],
                color='black', lw=1.6, zorder=5)
        pt = '***' if cc['p_perm'] < .001 else '**' if cc['p_perm'] < .01 else \
             '*' if cc['p_perm'] < .05 else 'n.s.'
        ax.text((i + j) / 2, y + span * 0.035, f"{pt}  $d_z$={cc['dz']:.2f}",
                ha='center', va='bottom', fontsize=FS_LEG * 0.8, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([TICK.get(m, m) for m in ORDER], fontsize=FS_TICK,
                       rotation=45, ha='right')
    ax.tick_params(axis='y', labelsize=FS_TICK)
    ax.set_ylim(min(lo * 1.25, -5), hi + span * 0.32)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle=':', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(TASK_NAME[task], fontsize=FS_TITLE, fontweight='bold', pad=12)
    ax.text(-0.10, 1.10, letter, transform=ax.transAxes,
            fontsize=FS_LETTER, fontweight='bold', va='top', ha='left')
    if show_ylabel:
        ax.set_ylabel(YLAB.get(metric, '% of human upperbound'),
                      fontsize=FS_LABEL * (0.8 if metric in YLAB else 1.0), labelpad=10)
    ax.margins(x=0.05)


for metric in METRICS:
    fig, axes = plt.subplots(1, 2, figsize=(19, 9))
    for ax, task, letter in zip(axes, ['describe', 'social'], 'AB'):
        panel(ax, metric, task, letter, show_ylabel=(task == 'describe'))
    ymax = max(a.get_ylim()[1] for a in axes)
    ymin = min(a.get_ylim()[0] for a in axes)
    for a in axes:
        a.set_ylim(ymin, ymax)
    sub = f'normalized by this metric\'s own split-half human ceiling; ' \
          f'Transformers and VLMs on cross-validated heads'
    if metric in NOTE:
        sub += f'\n{NOTE[metric]}'
    fig.suptitle(f'{FULL[metric]}\n{sub}', fontsize=FS_TITLE * 0.62, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    f = os.path.join(OUT, f'fig_metric_{metric}.png')
    fig.savefig(f, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('Saved:', f)


# ── overview: all seven metrics on one axis pair ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(24, 9), sharey=True)
W = 0.8 / len(ORDER)
for ax, task, letter in zip(axes, ['describe', 'social'], 'AB'):
    xm = np.arange(len(METRICS))
    for k, m in enumerate(ORDER):
        vals, errs = [], []
        for metric in METRICS:
            sub = summ[(summ.metric == metric) & (summ.task == task)].set_index('model')
            ceil = float(sub['ceiling'].iloc[0])
            r = sub.loc[m]
            p, e = normalize(metric, float(r['value']), float(r['sem']), ceil)
            vals.append(p); errs.append(e)
        off = (k - (len(ORDER) - 1) / 2) * W
        ax.bar(xm + off, vals, W, color=COLORS[m], label=TICK.get(m, m),
               edgecolor='black', linewidth=0.4, zorder=3)
        ax.errorbar(xm + off, vals, yerr=errs, fmt='none', color='black',
                    capsize=2, linewidth=1.0, zorder=4)
    ax.axhline(100.0, color='black', ls='--', lw=2.4, zorder=2)
    ax.axhline(0, color='black', lw=0.9, zorder=2)
    ax.set_xticks(xm)
    ax.set_xticklabels(METRICS, fontsize=FS_TICK)
    ax.tick_params(axis='y', labelsize=FS_TICK)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle=':', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(TASK_NAME[task], fontsize=FS_TITLE, fontweight='bold', pad=12)
    ax.text(-0.06, 1.10, letter, transform=ax.transAxes,
            fontsize=FS_LETTER, fontweight='bold', va='top', ha='left')
axes[0].set_ylabel('% of human upperbound', fontsize=FS_LABEL, labelpad=10)
axes[0].legend(fontsize=FS_LEG, ncol=3, loc='lower left', frameon=True,
               framealpha=0.92, edgecolor='none')
fig.suptitle('All seven saliency metrics, each normalized by its own split-half human ceiling\n'
'within every encoder pair the LSTM outscores the Transformer, on every metric, in both tasks\n'
'KL sits above 100% for every model because the split-half value is not a ceiling for it '
'(see fig_metric_KL.png); compare within KL, not across metrics',
fontsize=FS_TITLE * 0.55, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.90])
f = os.path.join(OUT, 'fig_metric_overview.png')
fig.savefig(f, dpi=300, bbox_inches='tight')
print('Saved:', f)

# ── console ───────────────────────────────────────────────────────────────────
print('\n=== LSTM vs matched Transformer (positive dz = LSTM better) ===')
print(con[['task', 'lstm', 'transformer', 'metric', 'dz', 'p_perm', 'lstm_wins']]
      .to_string(index=False))
print(f"\nLSTM wins {int(con.lstm_wins.sum())} of {len(con)} comparisons")
losses = con[~con.lstm_wins]
if len(losses):
    print('EXCEPTIONS:')
    print(losses.to_string(index=False))
