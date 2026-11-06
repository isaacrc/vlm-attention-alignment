"""
Blur sweep re-run on the CV-HONEST per-task heads (Task 1), not the paper's
designated heads.

The first sweep blurred CNN-Transformer L3H5 / ViT-Transformer L3H8. In the Social
task those are not the best heads (L4H4 and L4H8 are), so the Transformer curve
started ~.04 too low and the "blurring never reaches LSTM" claim was tested against
a weakened opponent. This re-runs the sweep from each model's honestly-selected
head, which is the strongest form of the smoothness objection.

Outputs: results/blur_sweep_cvhonest.csv
Figure:  figures/fig_blur_sweep_cvhonest.png
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'shared'))
import alignment_core as A

RES = os.path.join(HERE, 'results'); FIG = os.path.join(HERE, '..', 'figures')
DATA = os.path.join(HERE, '..', 'data')
SEL = json.load(open(os.path.join(HERE, '..', 'task1_head_selection',
                                  'results', 'selected_heads.json')))
ceil = json.load(open(os.path.join(DATA, 'ceilings.json')))
summ = pd.read_csv(os.path.join(DATA, 'alignment_summary_cv_honest.csv'))

SIGMAS = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]


def main():
    imgs = A.target_images()
    rows = []
    for task in ['social', 'describe']:
        refs = {m: float(summ[(summ.model == m) & (summ.task == task)]['mean_r'].iloc[0])
                for m in ['CNN-LSTM', 'ViT-LSTM']}
        for m in ['CNN-Transformer', 'ViT-Transformer']:
            s = SEL[f'{m}_{task}']
            L, H = s['layer'], s['head']
            base = A.trained_maps(m, task, imgs, layer=L, head=H)
            for sg in SIGMAS:
                bl = {im: (gaussian_filter(mp, sg) if sg > 0 else mp)
                      for im, mp in base.items()}
                z = A.subject_means(A.per_image_z(bl, task, imgs, m))['mean_z'].values
                rows.append(dict(task=task, model=m, layer=L, head=H, sigma=sg,
                                 mean_r=float(np.tanh(z.mean())),
                                 cnn_lstm_ref=refs['CNN-LSTM'],
                                 vit_lstm_ref=refs['ViT-LSTM']))
            print(f'  {task} {m} L{L}H{H} done', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, 'blur_sweep_cvhonest.csv'), index=False)
    print(df.pivot_table(index='sigma', columns=['task', 'model'],
                         values='mean_r').round(4).to_string())

    for task in ['social', 'describe']:
        d = df[df.task == task]
        for m in ['CNN-Transformer', 'ViT-Transformer']:
            sub = d[d.model == m]
            pk = sub.loc[sub['mean_r'].idxmax()]
            ref = pk['cnn_lstm_ref'] if m.startswith('CNN') else pk['vit_lstm_ref']
            print(f'{task:9s} {m:16s} peak r={pk["mean_r"]:.4f} at sigma={int(pk["sigma"])} '
                  f'| matched LSTM {ref:.4f} | gap {ref - pk["mean_r"]:+.4f}')

    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, task in zip(axes, ['social', 'describe']):
        d = df[df.task == task]
        for m, c in [('CNN-Transformer', '#d62728'), ('ViT-Transformer', '#9467bd')]:
            sub = d[d.model == m]
            lab = f'{m} (L{sub["layer"].iloc[0]}H{sub["head"].iloc[0]})'
            ax.plot(sub['sigma'], sub['mean_r'], marker='o', color=c, label=lab)
        ax.axhline(d['cnn_lstm_ref'].iloc[0], ls='--', color='orange', label='CNN-LSTM')
        ax.axhline(d['vit_lstm_ref'].iloc[0], ls=':', color='green', label='ViT-LSTM')
        ax.axhline(np.tanh(ceil[task]), ls='-', color='gray', alpha=0.6, label='human ceiling')
        ax.set_xlabel('Gaussian blur sigma (px)'); ax.set_ylabel('Spearman r')
        ax.set_title(f'{task.capitalize()} — blur sweep from CV-honest head')
        ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_blur_sweep_cvhonest.png'), dpi=150)
    print('Saved figure.')


if __name__ == '__main__':
    main()
