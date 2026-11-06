"""Does restricting the fit to one ROI change its per-vertex R^2?

If the ridge solves independently per vertex, a fit run on the 164 lateral vertices
must reproduce exactly what the full 5032-vertex fit produced at those same vertices.
Run once to justify re-aggregating stored fits onto new masks without refitting.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fit_encoding import load_attention, fit_single

m = np.load('rois/kastner_stream_masks_fs5.npz')
full = np.load('encoding/r2_CNN_LSTM_social_subj01_cvbehav.npz')
V = full['vertices']
sel = m['lateral'][V].astype(bool)
print(f"lateral vertices inside the fit: {sel.sum()}")

Y = np.load('responses/Y_shared_subj01.npy')
order = np.load('responses/shared_nsdIds_subj01.npy')
X, _ = load_attention('CNN_LSTM', 'social', order, 'cvbehav')

r2_sub = fit_single(X, Y[:, V[sel]])      # refit on the ROI alone
r2_full = full['r2_attn'][sel]            # same vertices, from the 5032-vertex fit
d = np.abs(r2_sub - r2_full)
print(f"  max |difference| : {d.max():.3e}")
print(f"  mean |difference|: {d.mean():.3e}")
print(f"  ROI mean, refit  : {r2_sub.mean():.6f}")
print(f"  ROI mean, stored : {r2_full.mean():.6f}")
print("\n=> identical" if d.max() < 1e-10 else "\n=> NOT identical: the ROI changes the fit")
