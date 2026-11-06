"""
Task 1 (step 2) — de-biased head-selection analysis.
Consumes head_table_perimage.parquet. For each Transformer/VLM model and task:
  - selection rules: best_head (biased), mean_all_heads, median_head,
    best_layer_mean, last_layer_mean, cv_honest (5-fold over images)
  - winner's-curse: bootstrap over images, E[max-over-heads] vs cv-honest
  - full per-head distribution figure with LSTM single-map reference line
Outputs: results/selection_comparison.csv, results/winners_curse.json,
         figures/fig_head_selection.png
"""
import os, sys, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, 'results')
FIG = os.path.join(HERE, '..', 'figures'); os.makedirs(FIG, exist_ok=True)
DATA = os.path.join(HERE, '..', 'data')
RNG = np.random.default_rng(0)

ceil = json.load(open(os.path.join(DATA, 'ceilings.json')))
summ = pd.read_csv(os.path.join(DATA, 'alignment_summary_best_head.csv'))
LSTM_R = {(r['model'], r['task']): r['mean_r'] for _, r in summ.iterrows()
          if r['model'] in ('CNN-LSTM', 'ViT-LSTM')}


def pivot(df_m):
    """(layer,head) x image matrix of subject-mean z.

    The row label is the string 'L{layer}H{head}' rather than a (layer, head)
    tuple: with a tuple label, M.loc[label, cols] is parsed by pandas as a
    (row, column) pair instead of a single index entry.
    """
    df_m = df_m.copy()
    df_m['lh'] = ('L' + df_m['layer'].astype(int).astype(str)
                  + 'H' + df_m['head'].astype(int).astype(str))
    M = df_m.pivot_table(index='lh', columns='image_id', values='mean_z')
    return M  # rows=heads, cols=images


def _layer_of(lh):
    """'L3H5' -> 3"""
    return int(lh.split('H')[0][1:])


def rules(M):
    head_mean = M.mean(axis=1)                        # overall mean z per head
    out = {}
    out['best_head'] = float(head_mean.max())
    out['mean_all_heads'] = float(head_mean.mean())
    out['median_head'] = float(head_mean.median())
    # best layer (by mean over its heads)
    layers = {}
    for lh in M.index:
        layers.setdefault(_layer_of(lh), []).append(head_mean[lh])
    layer_mean = {L: np.mean(v) for L, v in layers.items()}
    best_L = max(layer_mean, key=layer_mean.get)
    out['best_layer_mean'] = float(layer_mean[best_L])
    last_L = max(layers.keys())
    out['last_layer_mean'] = float(layer_mean[last_L])
    return out, head_mean


def cv_honest(M, k=5):
    cols = list(M.columns); n = len(cols)
    idx = RNG.permutation(n); folds = np.array_split(idx, k)
    ev = []
    for f in range(k):
        te = folds[f]; tr = np.concatenate([folds[j] for j in range(k) if j != f])
        tr_c = [cols[i] for i in tr]; te_c = [cols[i] for i in te]
        best = M[tr_c].mean(axis=1).idxmax()
        ev.append(M.loc[best, te_c].mean())
    return float(np.mean(ev)), float(np.std(ev, ddof=1) / np.sqrt(k))


def winners_curse(M, cv_est, nboot=2000):
    cols = np.array(M.columns); n = len(cols); Mv = M.values
    idxrows = M.index
    maxes = []
    for _ in range(nboot):
        bi = RNG.integers(0, n, n)
        hm = Mv[:, bi].mean(axis=1)
        maxes.append(hm.max())
    exp_max = float(np.mean(maxes))
    return dict(expected_max_r=float(np.tanh(exp_max)),
                cv_honest_r=float(np.tanh(cv_est)),
                inflation_z=exp_max - cv_est,
                inflation_r=float(np.tanh(exp_max) - np.tanh(cv_est)))


def main():
    pim = pd.read_parquet(os.path.join(RES, 'head_table_perimage.parquet'))
    models = ['CNN-Transformer', 'ViT-Transformer', 'Molmo', 'Qwen3.5']
    rows = []; wc = {}
    dist = {}
    for task in ['social', 'describe']:
        for m in models:
            sub = pim[(pim['model'] == m) & (pim['task'] == task)]
            if sub.empty:
                print(f'  (no data {m}/{task})'); continue
            M = pivot(sub)
            rule_vals, head_mean = rules(M)
            cv_r, cv_sem = cv_honest(M)
            rule_vals['cv_honest'] = cv_r
            for rule, z in rule_vals.items():
                rows.append(dict(model=m, task=task, rule=rule,
                                 mean_r=float(np.tanh(z)),
                                 pct_ceiling=100 * z / ceil[task]))
            wc[f'{m}_{task}'] = winners_curse(M, cv_r)
            dist[(m, task)] = np.tanh(head_mean.values)
    comp = pd.DataFrame(rows)
    comp.to_csv(os.path.join(RES, 'selection_comparison.csv'), index=False)
    json.dump(wc, open(os.path.join(RES, 'winners_curse.json'), 'w'), indent=2)
    print(comp.to_string(index=False))

    # ── figure: distribution of per-head r with LSTM lines + rule markers ──
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, task in zip(axes, ['social', 'describe']):
        xs = []
        for i, m in enumerate(models):
            vals = dist.get((m, task))
            if vals is None:
                continue
            jit = i + RNG.uniform(-0.15, 0.15, len(vals))
            ax.scatter(jit, vals, s=10, alpha=0.4, color='#4c72b0')
            cr = comp[(comp.model == m) & (comp.task == task)]
            for rule, mark, col in [('best_head', '*', 'red'),
                                    ('cv_honest', 'D', 'green'),
                                    ('mean_all_heads', '_', 'black')]:
                v = cr[cr.rule == rule]['mean_r']
                if len(v):
                    ax.scatter([i], [v.iloc[0]], marker=mark, s=140, color=col, zorder=5,
                               label=rule if i == 0 else None)
        # LSTM reference lines
        for m, ls, c in [('CNN-LSTM', '--', 'orange'), ('ViT-LSTM', ':', 'green')]:
            if (m, task) in LSTM_R:
                ax.axhline(LSTM_R[(m, task)], ls=ls, color=c, lw=2, label=m)
        # ceilings.json stores mean Fisher-z; this axis is in r
        ax.axhline(np.tanh(ceil[task]), ls='-', color='gray', lw=1.5, alpha=0.6,
                   label='human ceiling')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.replace('-', '-\n') for m in models])
        ax.set_title(f'{task.capitalize()} task — per-head alignment')
        ax.set_ylabel('Spearman r')
    axes[0].legend(fontsize=8, loc='upper right', ncol=2)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_head_selection.png'), dpi=150)
    print('Saved figure.')


if __name__ == '__main__':
    main()
