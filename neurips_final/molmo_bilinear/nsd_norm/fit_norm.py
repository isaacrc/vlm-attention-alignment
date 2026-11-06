"""
fit_norm.py --subject N --models M [M ...]
------------------------------------------
rebuttal/task8_nsd/fit_encoding.py (cvbehav heads, same vertices, folds and solver),
with one change: every attention map is scaled to sum to 1 over its 196 locations before
fitting. The raw maps differ in total attention per image -- 1 for the trained
Transformers (averaged over words), roughly the caption length for the LSTMs and Molmo
(summed over words), and the share of attention landing on image tokens for the VLMs --
and the per-feature standardization lets that total act as a non-spatial predictor.
Writes task8_nsd/encoding_norm/; the published encoding/ fits are untouched.
"""
import os, sys
from pathlib import Path

NSD = Path('/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd')
src = (NSD / 'fit_encoding.py').read_text()
old = 'out_dir = HERE / ("encoding_r2score_supplement" if args.metric == "r2" else "encoding")'
assert old in src
# NSD_ENC_DIR: output folder under task8_nsd (exploratory fits use their own)
src = src.replace(old, f'out_dir = HERE / {os.environ.get("NSD_ENC_DIR", "encoding_norm")!r}')
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
