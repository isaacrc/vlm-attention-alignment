"""
Tier-B, part 3 -- DeepGaze IIE (Linardos et al., ICCV 2021).

The strong modern reference. A reviewer asking "does your model beat a real
saliency model?" means this one. Two variants:

  deepgaze2e          centerbias = our empirical human center prior (standard
                      usage; gives DeepGaze its best shot on the raw metric)
  deepgaze2e_uniform  centerbias = uniform, isolating the image-driven component.
                      This is the honest variant for the center-controlled metric,
                      since the empirical prior would otherwise inject exactly the
                      bias we partial out.

Each image is run at its NATIVE resolution (DeepGaze is trained on natural-aspect
images) and only the resulting log-density is square-resized to 256x256, so the
model is never fed a stretched image.

Outputs: data/tierB_maps/deepgaze2e{,_uniform}.npz
"""
import os, sys, warnings, time
warnings.filterwarnings('ignore')
import numpy as np
import torch
from scipy.ndimage import zoom
from scipy.special import logsumexp
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared'))
import tierB_common as T
import alignment_core as A

from deepgaze_pytorch import DeepGazeIIE


def empirical_center_prior(imgs):
    """Mean raw human map over all subjects, images and both tasks (256x256).
    Same construction as the Tier-A center-bias prior."""
    acc = np.zeros((256, 256), np.float64)
    cnt = 0
    for task in ['social', 'describe']:
        hm_dir = A.HM_DIR[task]
        for fn in sorted(os.listdir(hm_dir)):
            if not fn.endswith('_heatmaps.npy'):
                continue
            if fn.replace('_heatmaps.npy', '') in A.EXCLUDE[task]:
                continue
            sd = np.load(os.path.join(hm_dir, fn), allow_pickle=True).item()
            for im in imgs:
                hm = sd.get(im, sd.get(im.replace('.jpg', ''), sd.get(im + '.jpg')))
                if hm is None:
                    continue
                h = np.nan_to_num(hm, nan=0.0).astype(np.float32)
                acc += np.asarray(Image.fromarray(h).resize((256, 256), Image.NEAREST),
                                  np.float64)
                cnt += 1
    return acc / max(cnt, 1)


def centerbias_for(prior256, h, w, uniform=False):
    """Log-density centerbias at (h, w), normalized so logsumexp == 0."""
    if uniform:
        cb = np.zeros((h, w), np.float64)
    else:
        p = np.clip(prior256, 1e-8, None)
        cb = np.log(p)
        cb = zoom(cb, (h / cb.shape[0], w / cb.shape[1]), order=1, mode='nearest')
    return cb - logsumexp(cb)


def main():
    imgs = A.target_images()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {dev}', flush=True)

    prior = empirical_center_prior(imgs)
    model = DeepGazeIIE(pretrained=True).to(dev).eval()

    out = {'deepgaze2e': {}, 'deepgaze2e_uniform': {}}
    t0 = time.time()
    for i, name in enumerate(imgs):
        rgb = T.load_rgb(name)
        h, w = rgb.shape[:2]
        x = torch.tensor(rgb.transpose(2, 0, 1)[None].astype(np.float32)).to(dev)
        for key, uni in [('deepgaze2e', False), ('deepgaze2e_uniform', True)]:
            cb = centerbias_for(prior, h, w, uniform=uni)
            c = torch.tensor(cb[None].astype(np.float32)).to(dev)
            with torch.no_grad():
                ld = model(x, c)
            ld = ld.detach().cpu().numpy()[0, 0]          # log density, native res
            # Spearman is rank-based, so keeping the log density is equivalent to
            # the density; square-resize into the pipeline's 256x256 frame.
            out[key][name] = T.normalize(T.to_square(ld))
        if (i + 1) % 20 == 0:
            el = time.time() - t0
            print(f'  {i+1}/{len(imgs)}  {el:.0f}s elapsed, '
                  f'{el/(i+1)*(len(imgs)-i-1):.0f}s left', flush=True)

    for k, v in out.items():
        arr = np.stack(list(v.values()))
        print(f'{k}: finite={np.isfinite(arr).all()} std={arr.std():.4f}', flush=True)
        T.save_maps(k, v)


if __name__ == '__main__':
    main()
