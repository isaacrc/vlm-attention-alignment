"""
run_headrule_center.py — center-bias-controlled alignment under an explicit head rule.
======================================================================================

Why this exists
---------------
`run_center_controlled.py` produced the center-corrected table that the rebuttal
quotes ("under center-bias control Molmo becomes the best-aligned agent"). But it
read each Transformer's map through `alignment_core.trained_maps` with no explicit
(layer, head), which falls back to the *paper's* fixed heads: CNN-Transformer L3H5
and ViT-Transformer L3H8 in BOTH tasks. Task 1's cross-validated sweep found the
Social heads are actually L4H4 and L4H8, and Molmo's Describe head is L8H7, not the
L10H27 argmax. The published center-corrected table is therefore on a *mixed* head
rule: paper-fixed for the trained Transformers, sweep-argmax for the VLMs. That is
exactly the cherry-picking the reviewers objected to, so the figure that sits next
to `fig_head_selection_corrected.png` cannot be built from it.

This script recomputes the center-controlled metric under two explicit rules:

  cv_best     the cross-validated head from Task 1 (results/selected_heads.json).
              No head is ever chosen on the metric it is scored against.
  layer_mean  mean attention over ALL layers x ALL heads, then the standard
              post-processing. A predefined rule with no selection at all -- the
              remedy the reviewers list ("averaged across heads/layers by a
              predefined rule"). Computed for the two trained Transformers, whose
              (4, 8, 196) head_layer_alphas make it exact and cheap.
  layerN      mean over the 8 heads WITHIN decoder layer N (N = 1..4), no averaging
              across layers. Decomposes layer_mean: it shows whether the all-layer
              mean's behaviour comes from every layer alike or from one layer, and
              it is still a predefined rule (a layer is named in advance, not
              picked by score).

Both are scored under both metrics, in one pass, so raw and partial are strictly
like-for-like:

  raw       Spearman r(model, human), the paper's metric
  partial   Spearman r(model, human | center prior), prior built leave-one-image-out

Models with a single attention map (CNN-LSTM, ViT-LSTM) and the center prior have
no rule to vary and are carried through unchanged as reference bars.

Outputs
-------
  results/headrule_center_long.parquet     per subject x image, both metrics
  results/headrule_center_summary.csv      per model x rule x task
  results/headrule_center_contrasts.csv    paired permutation tests

Run: sbatch run_headrule_center.slurm
"""
import os, sys, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'shared'))
sys.path.insert(0, HERE)

import alignment_core as A
import vlm_maps as V
from run_center_controlled import loio_prior_ranked, partial_rows, partial_ceiling

RES = os.path.join(HERE, 'results')
T1 = os.path.join(HERE, '..', 'task1_head_selection', 'results')
os.makedirs(RES, exist_ok=True)

SELECTED = json.load(open(os.path.join(T1, 'selected_heads.json')))

# (model, rule) pairs to evaluate. 'na' = nothing to select.
JOBS = [
    ('CENTER_PRIOR',    'na'),
    ('CNN-LSTM',        'na'),
    ('ViT-LSTM',        'na'),
    ('CNN-Transformer', 'cv_best'),
    ('CNN-Transformer', 'layer_mean'),
    ('ViT-Transformer', 'cv_best'),
    ('ViT-Transformer', 'layer_mean'),
    ('Molmo',           'cv_best'),
    ('Qwen3.5',         'cv_best'),
] + [(m, f'layer{n}') for m in ('CNN-Transformer', 'ViT-Transformer') for n in (1, 2, 3, 4)]


def layer_mean_maps(model_name, task, imgs, layer=None):
    """Mean over a trained Transformer's cross-attention heads.

    head_layer_alphas is (n_layers, n_heads, 196) per image. With layer=None the
    mean is over all layers x heads; with layer=N (1-indexed) it is over the 8
    heads of that layer only. Averaging happens on the raw attention weights,
    before the 14x14 -> 256x256 upsample and min-max normalization, so the rule is
    applied to attention rather than to already rescaled maps.
    """
    info = A.TRAINED[model_name]
    npz = os.path.join(A.ARCH_DIR, info['cond'], 'ckpt_cache', A.NPZ_FILE[info['cond']][task])
    hla = A._load_npz(npz)['head_layer_alphas']
    out = {}
    for im in imgs:
        k = im if im in hla else im.replace('.jpg', '')
        if k not in hla:
            continue
        arr = np.asarray(hla[k], np.float32)
        vec = arr.mean(axis=(0, 1)) if layer is None else arr[layer - 1].mean(axis=0)
        out[im] = A.alpha_to_map(vec)
    return out


def get_maps(model, rule, task, imgs, prior_rk):
    if model == 'CENTER_PRIOR':
        return {im: prior_rk[i].reshape(A.OUT_SIZE, A.OUT_SIZE) for i, im in enumerate(imgs)}
    if rule == 'layer_mean':
        return layer_mean_maps(model, task, imgs)
    if rule.startswith('layer'):
        return layer_mean_maps(model, task, imgs, layer=int(rule[5:]))
    if rule == 'cv_best':
        d = SELECTED[f'{model}_{task}']
        L, Hd = int(d['layer']), int(d['head'])
        if model in ('Molmo', 'Qwen3.5'):
            return V.vlm_maps(model, task, L, Hd)
        return A.trained_maps(model, task, imgs, layer=L, head=Hd)
    return A.trained_maps(model, task, imgs)          # LSTMs: single map


def head_label(model, rule, task):
    if rule == 'layer_mean':
        return 'all layers x heads'
    if rule.startswith('layer'):
        return f'L{rule[5:]} mean over heads'
    if rule == 'cv_best':
        d = SELECTED[f'{model}_{task}']
        return f"L{d['layer']}H{d['head']} ({d['fold_agreement']})"
    return '-'


def summarize(df, rule, raw_ceil_z, part_ceil_z):
    """Subject-level Fisher-z means -> r, SEM, and % of the matching ceiling.

    r_raw here is the ordinary Spearman the paper reports; r_partial is the same
    subject x image pairs with the center prior removed. Both are normalized by
    the ceiling computed under their own metric.
    """
    out = []
    for (model, task), g in df.groupby(['model', 'task']):
        row = dict(model=model, rule=rule, task=task,
                   head=head_label(model, rule, task), n_sub=g.subject_id.nunique())
        for metric, ceil_z in [('raw', raw_ceil_z), ('partial', part_ceil_z)]:
            col = 'r_raw' if metric == 'raw' else 'r_partial'
            zs = g.groupby('subject_id')[col].apply(
                lambda s: np.arctanh(np.clip(s, -.999999, .999999)).mean())
            zbar, n = float(zs.mean()), len(zs)
            sem_z = float(zs.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            row[f'r_{metric}'] = float(np.tanh(zbar))
            row[f'sem_{metric}'] = float(np.tanh(zbar + sem_z) - np.tanh(zbar))
            row[f'pct_{metric}'] = 100.0 * zbar / ceil_z
        out.append(row)
    return out


def main():
    imgs = A.target_images()
    print('building leave-one-image-out center prior...', flush=True)
    prior_rk = loio_prior_ranked(imgs)

    all_rows, summary, ceilings = [], [], {}
    for task in ['social', 'describe']:
        raw_c, part_c = A.ceiling(task), partial_ceiling(task, imgs, prior_rk)
        ceilings[task] = dict(raw_z=raw_c, raw_r=float(np.tanh(raw_c)),
                              partial_z=part_c, partial_r=float(np.tanh(part_c)))
        print(f'\n[{task}] ceiling raw r={np.tanh(raw_c):.4f} | '
              f'partial r={np.tanh(part_c):.4f}', flush=True)

        for model, rule in JOBS:
            maps = get_maps(model, rule, task, imgs, prior_rk)
            df = partial_rows(maps, task, imgs, prior_rk, model)
            if df.empty:
                print(f'  {model:16s} {rule:11s} SKIPPED (no maps)', flush=True)
                continue
            df['rule'] = rule
            all_rows.append(df)
            s = summarize(df, rule, raw_c, part_c)
            summary += s
            r = s[0]
            print(f'  {model:16s} {rule:11s} {r["head"]:20s} '
                  f'raw r={r["r_raw"]:.4f} ({r["pct_raw"]:5.1f}%)  '
                  f'partial r={r["r_partial"]:.4f} ({r["pct_partial"]:5.1f}%)', flush=True)

    long = pd.concat(all_rows, ignore_index=True)
    long.to_parquet(os.path.join(RES, 'headrule_center_long.parquet'), index=False)
    sdf = pd.DataFrame(summary)
    sdf.to_csv(os.path.join(RES, 'headrule_center_summary.csv'), index=False)
    json.dump(ceilings, open(os.path.join(RES, 'headrule_center_ceilings.json'), 'w'), indent=2)

    # ── paired permutation tests, per-image, on both metrics ──
    # Image-level pairing (not subject-level) matches run_center_controlled.py so
    # the dz values stay comparable to the numbers already in task3_baselines.md.
    rng = np.random.default_rng(0)
    long['key'] = long['model'] + '|' + long['rule']
    tests = []
    for metric in ['r_raw', 'r_partial']:
        zcol = np.arctanh(np.clip(long[metric], -.999999, .999999))
        pim = long.assign(_z=zcol).groupby(['key', 'task', 'image_id'])['_z'].mean().reset_index()
        pairs = [
            # does the decoder effect survive each rule?
            ('CNN-LSTM|na', 'CNN-Transformer|cv_best'),
            ('CNN-LSTM|na', 'CNN-Transformer|layer_mean'),
            ('ViT-LSTM|na', 'ViT-Transformer|cv_best'),
            ('ViT-LSTM|na', 'ViT-Transformer|layer_mean'),
            # what does dropping head selection cost?
            ('CNN-Transformer|cv_best', 'CNN-Transformer|layer_mean'),
            ('ViT-Transformer|cv_best', 'ViT-Transformer|layer_mean'),
            # is Molmo's lead over our best trained model real?
            ('Molmo|cv_best', 'CNN-LSTM|na'),
            ('Molmo|cv_best', 'CNN-Transformer|cv_best'),
            # everything above the center prior?
            ('CNN-Transformer|layer_mean', 'CENTER_PRIOR|na'),
            ('ViT-Transformer|layer_mean', 'CENTER_PRIOR|na'),
            ('Molmo|cv_best', 'CENTER_PRIOR|na'),
            ('Qwen3.5|cv_best', 'CENTER_PRIOR|na'),
        ] + [(f'{l}|na', f'{t}|layer{n}')          # decoder effect, layer by layer
             for l, t in [('CNN-LSTM', 'CNN-Transformer'), ('ViT-LSTM', 'ViT-Transformer')]
             for n in (1, 2, 3, 4)]
        for task in ['social', 'describe']:
            sub = pim[pim.task == task]
            for a, b in pairs:
                x = sub[sub.key == a].set_index('image_id')['_z']
                y = sub[sub.key == b].set_index('image_id')['_z']
                c = x.index.intersection(y.index)
                if len(c) == 0:
                    continue
                d = (x.loc[c] - y.loc[c]).values
                obs = d.mean()
                null = (rng.choice([-1, 1], size=(10000, len(d))) * d).mean(axis=1)
                tests.append(dict(metric=metric, task=task, a=a, b=b,
                                  mean_diff=float(obs),
                                  dz=float(obs / (d.std(ddof=1) + 1e-12)), n_img=len(d),
                                  p_perm=float((np.sum(np.abs(null) >= abs(obs)) + 1) / 10001)))
    tdf = pd.DataFrame(tests)
    tdf.to_csv(os.path.join(RES, 'headrule_center_contrasts.csv'), index=False)

    print('\n' + sdf.to_string(index=False))
    print('\n' + tdf[tdf.metric == 'r_partial'].to_string(index=False))
    print('\nWrote results/headrule_center_{long.parquet,summary.csv,contrasts.csv}')


if __name__ == '__main__':
    main()
