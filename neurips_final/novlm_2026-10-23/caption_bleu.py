"""
caption_bleu.py -- BLEU-1..4 for model and human captions, for reference only (the caption
analysis is not in main_2026-10-23.tex). Captions are the texts saved by caption_embed.py (same
loaders and exclusions as Figure 6 of the 10-22 version).

Leave-one-participant-out, so models and humans face identical reference sets: for each
participant s, the references for every image are the captions of all OTHER participants.
  Human  hypothesis = participant s's caption
  Model  hypothesis = the model's caption
Corpus BLEU (nltk, uniform n-gram weights) over the images participant s captioned; the table
reports the mean and SEM over participants. Text is lowercased and stripped of punctuation.
Social sub-004 is excluded everywhere: its cleaned transcript is mostly a copy of sub-003's
(183 of 199 captions identical), which would inflate the human references.
Writes final_figures/manuscript_2026-10-23/caption_bleu.txt
"""
import os, re
import numpy as np
from nltk.translate.bleu_score import corpus_bleu

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ['CNN-LSTM', 'CNN-Transformer', 'ViT-LSTM', 'ViT-Transformer', 'Molmo', 'Qwen3.5']
EXCLUDE = {'social': {'sub-004'}, 'describe': set()}
W = {1: (1, 0, 0, 0), 2: (.5, .5, 0, 0), 3: (1 / 3, 1 / 3, 1 / 3, 0), 4: (.25, .25, .25, .25)}
tok = lambda s: re.sub(r"[^a-z0-9' ]", ' ', str(s).lower().replace('<unk>', ' ')).split()

lines = [__doc__.strip(), '',
         'Reading the numbers: BLEU counts n-gram overlap with the references, so it rewards short,',
         'conventional phrasing that many participants share ("the man is paying attention to the").',
         'The trained models write in exactly that register, so they can score above an individual',
         'participant without describing the image better. The VLMs write longer, differently phrased',
         'captions, which lowers their n-gram precision, most visibly in Describe. Model SEMs are tiny',
         'because the model caption is fixed and only the held-out reference set changes.', '']
for task in ('describe', 'social'):
    z = np.load(os.path.join(HERE, 'results', f'caption_embeddings_{task}.npz'), allow_pickle=True)
    subs = sorted({k.split('__')[1] for k in z.files if k.startswith('human__')} - EXCLUDE[task])
    H = {s: {int(i): tok(t) for i, t in zip(z[f'human__{s}__ids'], z[f'human__{s}__text'])} for s in subs}
    M = {m: {int(i): tok(t) for i, t in zip(z[f'model__{m}__ids'], z[f'model__{m}__text'])} for m in MODELS}
    scores = {a: {n: [] for n in W} for a in ['Human'] + MODELS}
    for s in subs:
        ids = [i for i in sorted(H[s]) if H[s][i] and any(H[o].get(i) for o in subs if o != s)]
        refs = [[H[o][i] for o in subs if o != s and H[o].get(i)] for i in ids]
        for a in ['Human'] + MODELS:
            hyp = [H[s][i] for i in ids] if a == 'Human' else [M[a].get(i, []) for i in ids]
            for n, w in W.items():
                scores[a][n].append(corpus_bleu(refs, hyp, weights=w))
    lines += [f'{task.upper()} (N = {len(subs)} participants; references = the other {len(subs) - 1})',
              f'{"":24s}' + ''.join(f'{"BLEU-" + str(n):>16s}' for n in W)]
    for a in ['Human'] + MODELS:
        tag = a + (' (VLM baseline)' if a in ('Molmo', 'Qwen3.5') else '')
        lines.append(f'{tag:24s}' + ''.join(f'{np.mean(scores[a][n]):>9.3f} ±{np.std(scores[a][n], ddof=1) / np.sqrt(len(subs)):.3f}' for n in W))
    lines.append('')
out = os.path.join(ROOT, 'neurips_final', 'final_figures', 'manuscript_2026-10-23', 'caption_bleu.txt')
open(out, 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines[-20:])); print('wrote', out)
