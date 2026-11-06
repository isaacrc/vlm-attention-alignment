"""
lmm_effect_coded.py
-------------------
The section 4.1 mixed model (run_stats.py: z ~ encoder*decoder*task, participant groups,
image variance component) refit with effect coding (+-0.5), so each coefficient is a main
effect or interaction averaged over the other factors rather than a simple effect at the
CNN / LSTM / Describe reference cell. Same data and random-effects structure.

Output: final_figures/qwen35_nothink/results/lmm_summary_effect_coded.txt
"""
import numpy as np, pandas as pd, statsmodels.formula.api as smf
RES = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/final_figures/qwen35_nothink/results'
long = pd.read_parquet(f'{RES}/alignment_long_cv_honest.parquet')
d = long[long.encoder.isin(['CNN', 'ViT'])].copy()
d['image_id'] = d['image_id'].astype(str)
d['ViT'] = np.where(d.encoder == 'ViT', 0.5, -0.5)
d['Transformer'] = np.where(d.decoder == 'Transformer', 0.5, -0.5)
d['Social'] = np.where(d.task == 'social', 0.5, -0.5)
m = smf.mixedlm('z ~ ViT*Transformer*Social', d, groups=d['subject_id'],
                vc_formula={'image': '0+C(image_id)'}).fit(method='lbfgs', maxiter=500)
with open(f'{RES}/lmm_summary_effect_coded.txt', 'w') as f:
    f.write(str(m.summary()))
    f.write('\n\nEffect coding: ViT, Transformer, Social = +0.5; CNN, LSTM, Describe = -0.5.\n'
            'Each coefficient is the difference between levels averaged over the other factors\n'
            '(interactions: difference of differences). Four trained models only.\n')
print(m.summary())
