"""
make_table_baselines.py -- Appendix E table of main_2026-10-23.tex (overleaf/table_baselines_v4.tex).
Same rows, sources and percentages as scripts/make_baseline_table.py in its
MOLMO_BILINEAR=1 MOLMO_BILINEAR_DIR=molmo_bilinear_pairwise mode (table_baselines_v3.tex):
percent = r / human upperbound r under the same metric. Changes: Molmo and Qwen3.5 move to their
own block as baselines, and edge energy and luminance (results/edge_lum_baselines.csv,
score_edge_lum.py) are added to the baseline block.
"""
import json, os
import pandas as pd

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
HERE = os.path.dirname(os.path.abspath(__file__))
Q = os.path.join(ROOT, 'neurips_final', 'final_figures', 'molmo_bilinear_pairwise', 'results')
ceil = json.load(open(os.path.join(Q, 'headrule_center_ceilings.json')))
hr = pd.read_csv(os.path.join(Q, 'headrule_center_summary.csv'))
tb = pd.read_csv(os.path.join(ROOT, 'rebuttal', 'task3_baselines', 'results', 'tierB_baselines.csv'))
el = pd.read_csv(os.path.join(HERE, 'results', 'edge_lum_baselines.csv'))


def from_hr(name, rule):
    return {t: tuple(hr[(hr.model == name) & (hr.rule == rule) & (hr.task == t)].iloc[0][['r_raw', 'r_partial']]) for t in ('social', 'describe')}


def from_csv(df, name):
    return {t: tuple(df[(df.baseline == name) & (df.task == t)].iloc[0][['raw_r', 'partial_r']]) for t in ('social', 'describe')}


blocks = [
    [('CNN-LSTM', from_hr('CNN-LSTM', 'na')), ('ViT-LSTM', from_hr('ViT-LSTM', 'na')),
     ('CNN-Transformer', from_hr('CNN-Transformer', 'cv_best')), ('ViT-Transformer', from_hr('ViT-Transformer', 'cv_best'))],
    [('Molmo 7B-D', from_hr('Molmo', 'cv_best')),
     ('Qwen3.5 9B', from_hr('Qwen3.5', 'cv_best'))],
    [('Center prior (mean fixation map)', from_hr('CENTER_PRIOR', 'na')),
     ('Itti--Koch \\citep{itti1998model}', from_csv(tb, 'itti_koch')),
     ('GBVS \\citep{harel2006graph}', from_csv(tb, 'gbvs')),
     ('Spectral residual \\citep{hou2007saliency}', from_csv(tb, 'spectral_residual')),
     ('Edge energy (Sobel)', from_csv(el, 'edge_energy')),
     ('Luminance', from_csv(el, 'luminance')),
     ('Person mask (COCO annotation)', from_csv(tb, 'coco_person')),
     ('Face blobs (COCO keypoints)', from_csv(tb, 'coco_face'))]]


def fmt(label, d):
    cells = []
    for t in ('social', 'describe'):
        rr, pr = d[t]
        cells += [f'{rr:.3f}', f"{100 * rr / ceil[t]['raw_r']:.0f}", f'{pr:.3f}', f"{100 * pr / ceil[t]['partial_r']:.0f}"]
    return label + ' & ' + ' & '.join(cells) + r' \\'


c = ceil
lines = [r'\begin{table}[t]', r'\centering',
         r'\caption{Alignment of model attention and baselines with human fixations, before and after removing center bias. '
         r"Raw: Spearman $r$ against each participant's fixation map (mean over participants and images). "
         r'Partial: the same, controlling for a leave-one-image-out center prior. '
         r'Percentages are of the human upperbound computed under the same metric (raw: $r = '
         + f"{c['social']['raw_r']:.3f}" + r'$ Social, $' + f"{c['describe']['raw_r']:.3f}" + r'$ Describe; partial: $'
         + f"{c['social']['partial_r']:.3f}" + r'$, $' + f"{c['describe']['partial_r']:.3f}" + r'$). '
         r'Top: our four trained models. Middle: the two pretrained VLMs. Bottom: the other baselines. '
         r'Transformer and VLM rows use the cross-validated per-task head. '
         r'Saliency, low-level, and semantic-prior baselines are reported at their own best Gaussian blur. '
         r'Edge energy and luminance are computed in each cell of the $14{\times}14$ grid, as in the neural analysis. '
         r'The two semantic priors use ground-truth annotation that no model receives.}',
         r'\label{tab:baselines}', r'\small', r'\setlength{\tabcolsep}{4.5pt}', r'\begin{tabular}{lrrrrrrrr}', r'\toprule',
         r' & \multicolumn{4}{c}{Social} & \multicolumn{4}{c}{Describe} \\',
         r'\cmidrule(lr){2-5} \cmidrule(lr){6-9}',
         r' & raw $r$ & \% & partial $r$ & \% & raw $r$ & \% & partial $r$ & \% \\']
for b in blocks:
    lines.append(r'\midrule')
    lines += [fmt(l, d) for l, d in b]
lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
out = os.path.join(ROOT, 'neurips_final', 'overleaf', 'table_baselines_v4.tex')
assert not os.path.exists(out) or open(out).read().startswith(r'\begin{table}[t]')
open(out, 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
