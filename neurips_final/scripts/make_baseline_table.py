"""Build the LaTeX baselines table (Results §4.2) from the rebuttal result files.
Percentages are r / (human ceiling r under the same metric), using the ceilings in
headrule_center_ceilings.json, so every row is normalized identically."""
import json, pandas as pd, numpy as np
R = "/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task3_baselines/results/"
ceil = json.load(open(R + 'headrule_center_ceilings.json'))
# Model rows (incl. Qwen3.5 thinking-off) from neurips_final/qwen35_nothink/rerun_appendix.py; baselines unchanged.
Q = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/final_figures/qwen35_nothink/results/'
# MOLMO_BILINEAR=1: model rows, ceilings (leave-one-out) from the molmo_bilinear run, and
# percentages as ratios of Fisher z, the convention of the main text; writes table_baselines_v2.tex
BILIN = __import__('os').environ.get('MOLMO_BILINEAR') == '1'
# MOLMO_BILINEAR_DIR=molmo_bilinear_pairwise: bilinear Molmo with the original pairwise ceilings and
# r-ratio percentages; writes table_baselines_v3.tex
PAIR = BILIN and __import__('os').environ.get('MOLMO_BILINEAR_DIR', 'molmo_bilinear') == 'molmo_bilinear_pairwise'
if BILIN:
    Q = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/final_figures/' + ('molmo_bilinear_pairwise' if PAIR else 'molmo_bilinear') + '/results/'
    ceil = json.load(open(Q + 'headrule_center_ceilings.json'))
hr = pd.read_csv(Q + 'headrule_center_summary.csv')
tb = pd.read_csv(R + 'tierB_baselines.csv')

def pct(r, task, metric):  # metric in {'raw','partial'}
    if BILIN and not PAIR:
        return 100 * np.arctanh(r) / ceil[task][f'{metric}_z']
    return 100 * r / ceil[task][f'{metric}_r']

rows = []  # (label, group, {task: (raw_r, partial_r)})
def model_row(label, name, rule):
    d = {}
    for t in ('social', 'describe'):
        s = hr[(hr.model == name) & (hr.rule == rule) & (hr.task == t)].iloc[0]
        d[t] = (s.r_raw, s.r_partial)
    rows.append((label, 'model', d))
model_row('CNN-LSTM', 'CNN-LSTM', 'na'); model_row('ViT-LSTM', 'ViT-LSTM', 'na')
model_row('CNN-Transformer', 'CNN-Transformer', 'cv_best'); model_row('ViT-Transformer', 'ViT-Transformer', 'cv_best')
model_row('Molmo 7B-D', 'Molmo', 'cv_best'); model_row('Qwen3.5 9B', 'Qwen3.5', 'cv_best')
d = {}
for t in ('social', 'describe'):
    s = hr[(hr.model == 'CENTER_PRIOR') & (hr.task == t)].iloc[0]; d[t] = (s.r_raw, s.r_partial)
rows.append(('Center prior (mean fixation map)', 'baseline', d))
def base_row(label, name):
    d = {}
    for t in ('social', 'describe'):
        s = tb[(tb.baseline == name) & (tb.task == t)].iloc[0]; d[t] = (s.raw_r, s.partial_r)
    rows.append((label, 'baseline', d))
base_row('Itti--Koch \\citep{itti1998model}', 'itti_koch'); base_row('GBVS \\citep{harel2006graph}', 'gbvs')
base_row('Spectral residual \\citep{hou2007saliency}', 'spectral_residual')
base_row('Person mask (COCO annotation)', 'coco_person'); base_row('Face blobs (COCO keypoints)', 'coco_face')

lines = []
lines.append(r'\begin{table}[t]')
lines.append(r'\centering')
CEIL_TXT = (r'Percentages are of the leave-one-out human noise ceiling computed under the same metric (raw: $r = ' + f"{ceil['social']['raw_r']:.3f}" + r'$ Social, $' + f"{ceil['describe']['raw_r']:.3f}" + r'$ Describe; partial: $' + f"{ceil['social']['partial_r']:.3f}" + r'$, $' + f"{ceil['describe']['partial_r']:.3f}" + r'$).') if BILIN and not PAIR else None
lines.append(r'\caption{Alignment of model attention and baselines with human fixations, before and after removing center bias. Raw: Spearman $r$ against each participant' + "'" + r's fixation map (mean over participants and images). Partial: the same, controlling for a leave-one-image-out center prior. Percentages are of the human upperbound computed under the same metric (raw: $r = 0.272$ Social, $0.248$ Describe; partial: $0.237$, $0.203$). Transformer and VLM rows use the cross-validated per-task head. Saliency and semantic-prior baselines are reported at their own best Gaussian blur. The two semantic priors use ground-truth annotation that no model receives.}')
lines.append(r'\label{tab:baselines}')
lines.append(r'\small')
lines.append(r'\begin{tabular}{lrrrrrrrr}')
lines.append(r'\toprule')
lines.append(r' & \multicolumn{4}{c}{Social} & \multicolumn{4}{c}{Describe} \\')
lines.append(r'\cmidrule(lr){2-5} \cmidrule(lr){6-9}')
lines.append(r' & raw $r$ & \% & partial $r$ & \% & raw $r$ & \% & partial $r$ & \% \\')
lines.append(r'\midrule')
def fmt(label, d):
    cells = []
    for t in ('social', 'describe'):
        rr, pr = d[t]
        cells += [f'{rr:.3f}', f'{pct(rr, t, "raw"):.0f}', f'{pr:.3f}', f'{pct(pr, t, "partial"):.0f}']
    return label + ' & ' + ' & '.join(cells) + r' \\'
for label, g, d in rows:
    if label.startswith('Center'): lines.append(r'\midrule')
    lines.append(fmt(label, d))
lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')
lines.append(r'\end{table}')
if PAIR:
    lines[2] = lines[2].replace('(raw: $r = 0.272$ Social, $0.248$ Describe; partial: $0.237$, $0.203$)', '(raw: $r = ' + f"{ceil['social']['raw_r']:.3f}" + r'$ Social, $' + f"{ceil['describe']['raw_r']:.3f}" + r'$ Describe; partial: $' + f"{ceil['social']['partial_r']:.3f}" + r'$, $' + f"{ceil['describe']['partial_r']:.3f}" + r'$)')
elif BILIN:
    lines[2] = lines[2].replace('Percentages are of the human upperbound computed under the same metric (raw: $r = 0.272$ Social, $0.248$ Describe; partial: $0.237$, $0.203$).', CEIL_TXT)
out = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/overleaf/' + ('table_baselines_v3.tex' if PAIR else 'table_baselines_v2.tex' if BILIN else 'table_baselines.tex')
open(out, 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
