"""
caption_embed.py -- re-run of the Figure 6 caption similarity (scripts/fig_caption_cosine_slides.py)
that keeps what the original discarded. Same captions, loaders, exclusions, embedder
(all-mpnet-base-v2, normalized) and averaging; the loaders are executed from the original file.
Saves
  results/caption_embeddings_{task}.npz       every caption embedding (human per participant, models)
  results/caption_cosine_perimage.csv         task, model, subject_id, image_id, cos
                                              (model rows: model vs that participant's caption;
                                               Human rows: that participant vs every other participant)
  results/caption_cosine_persubject.csv       mean over images per participant
and checks the participant-level means against final_figures/slides/caption_cosine_summary.csv.
"""
import os
import numpy as np, pandas as pd

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, 'neurips_final', 'scripts', 'fig_caption_cosine_slides.py')
RES = os.path.join(HERE, 'results'); os.makedirs(RES, exist_ok=True)

src = open(SRC).read()
g = {'__file__': SRC, '__name__': 'caption_src'}
exec(compile(src[:src.index('MODELS_6 = [')], SRC, 'exec'), g)       # imports, paths, loaders
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer('all-mpnet-base-v2')
enc = lambda texts: embedder.encode(texts, normalize_embeddings=True, batch_size=128)

MODELS_6 = ['CNN-LSTM', 'CNN-Transformer', 'ViT-LSTM', 'ViT-Transformer', 'Molmo', 'Qwen3.5']
imgs = sorted(g['img_key'](x) for x in pd.read_csv(os.path.join(g['BASE'], 'rnn_model', 'results', 'target_imgs.csv')).iloc[:, 0])
rows = []
for task in ['describe', 'social']:
    hum = g['human_captions'](task)
    subs = sorted(hum)
    H = {s: dict(zip(hum[s], enc(list(hum[s].values())))) for s in subs}
    store = {}
    for s in subs:
        ids = sorted(H[s]); store[f'human__{s}__ids'] = np.array(ids); store[f'human__{s}__emb'] = np.stack([H[s][i] for i in ids])
        store[f'human__{s}__text'] = np.array([hum[s][i] for i in ids], dtype=object)
        for i in imgs:
            if i in H[s]:
                others = [H[s][i] @ H[o][i] for o in subs if o != s and i in H[o]]
                rows.append(dict(task=task, model='Human', subject_id=s, image_id=i, cos=float(np.mean(others))))
    for m in MODELS_6:
        caps = g['model_captions'](m, task)
        M = dict(zip(caps, enc(list(caps.values()))))
        ids = sorted(M); store[f'model__{m}__ids'] = np.array(ids); store[f'model__{m}__emb'] = np.stack([M[i] for i in ids])
        store[f'model__{m}__text'] = np.array([caps[i] for i in ids], dtype=object)
        for s in subs:
            for i in imgs:
                if i in M and i in H[s]:
                    rows.append(dict(task=task, model=m, subject_id=s, image_id=i, cos=float(M[i] @ H[s][i])))
        print(task, m, f'{len(caps)} captions', flush=True)
    np.savez_compressed(os.path.join(RES, f'caption_embeddings_{task}.npz'), **store)

pim = pd.DataFrame(rows)
pim.to_csv(os.path.join(RES, 'caption_cosine_perimage.csv'), index=False)
psub = pim.groupby(['task', 'model', 'subject_id'])['cos'].mean().rename('cos').reset_index()
psub.to_csv(os.path.join(RES, 'caption_cosine_persubject.csv'), index=False)

summ = psub.groupby(['task', 'model'])['cos'].agg(mean='mean', sem=lambda x: x.std(ddof=1) / np.sqrt(len(x)), n='size').reset_index()
ref = pd.read_csv(os.path.join(ROOT, 'neurips_final', 'final_figures', 'slides', 'caption_cosine_summary.csv'))
chk = summ.merge(ref, on=['task', 'model'], suffixes=('', '_fig6'))
chk['diff'] = (chk['mean'] - chk['mean_fig6']).abs()
print(chk.to_string())
assert (chk.n == chk.n_fig6).all() and chk['diff'].max() < 1e-4, 'does not reproduce Figure 6'
print('REPRODUCES FIGURE 6 (max |diff| = %.2e)' % chk['diff'].max())
