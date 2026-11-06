"""
fit_subset.py --subject N --models M [M ...]
-------------------------------------------
nsd_norm/fit_norm.py (cvbehav heads, same vertices, folds and solver, every map scaled to
sum to 1) with one more change: the fit uses only the shared images listed in NSD_SUBSET
(an .npy of nsdIds), so every predictor fit this way is scored on the same image subset.
Output folder: task8_nsd/$NSD_ENC_DIR.
"""
import os, sys
from pathlib import Path

NSD = Path('/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd')
subset = os.environ['NSD_SUBSET']
src = (NSD / 'fit_encoding.py').read_text()

old = 'out_dir = HERE / ("encoding_r2score_supplement" if args.metric == "r2" else "encoding")'
assert old in src
src = src.replace(old, f'out_dir = HERE / {os.environ["NSD_ENC_DIR"]!r}')

old = '    order = np.load(resp / f"shared_nsdIds_subj{subj:02d}.npy")\n'
assert src.count(old) == 1
src = src.replace(old, old + (
    f'    _sel = np.isin(order, np.load({subset!r}))\n'
    '    order, Y_full = order[_sel], Y_full[_sel]\n'
    '    print(f"subset: {_sel.sum()} of {len(_sel)} shared images", flush=True)\n'))

# sum-to-1 normalization, as in nsd_norm/fit_norm.py
patch = '''
_load_attention_raw = load_attention
def load_attention(model, task, order, head_rule="mean"):
    loaded = _load_attention_raw(model, task, order, head_rule)
    if loaded is None:
        return None
    X, ok = loaded
    X = X.copy()
    s = X[ok].sum(1, keepdims=True)
    assert X[ok].min() >= 0 and s.min() > 0, (model, task, float(X[ok].min()), float(s.min()))
    X[ok] = X[ok] / s
    return X, ok


def main():'''
assert src.count('\ndef main():') == 1
src = src.replace('\ndef main():', patch, 1)
sys.argv = ['fit_encoding.py'] + sys.argv[1:] + ['--head-rule', 'cvbehav']
exec(compile(src, str(NSD / 'fit_encoding.py'), 'exec'),
     {'__file__': str(NSD / 'fit_encoding.py'), '__name__': '__main__'})
