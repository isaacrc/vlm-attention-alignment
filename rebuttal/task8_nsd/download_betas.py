"""Download NSD fsaverage5 betas for the trials in nsd_trial_index.csv.

Uses HTTP range requests: fsaverage5 is the first 10242 vertices of fsaverage (verified to
1e-7), and .mgh stores frames contiguously, so one trial = one contiguous 40968-byte read
at offset 284 + frame*163842*4. That is 1.75 GB total instead of 315 GB of session files.

Per subject, writes betas_subj{NN}.npz:
    betas   (n_trials, 20484) float32   lh then rh, fsaverage5
    meta    the index rows, aligned to betas rows

Run: python download_betas.py [--subjects 1 2 ...] [--workers 16]
"""

import argparse
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

BUCKET = "https://natural-scenes-dataset.s3.amazonaws.com"
HDR, VERTS_FS7, VERTS_FS5 = 284, 163842, 10242
OUT = Path(__file__).parent / "betas"
OUT.mkdir(exist_ok=True)


def fetch(url, start, end, retries=5):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) == end - start + 1:
                return data
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(2 ** attempt)
    raise RuntimeError(f"failed after {retries} tries: {url} [{start}-{end}]")


def beta_url(subj, hemi, ses):
    return (f"{BUCKET}/nsddata_betas/ppdata/subj{subj:02d}/fsaverage/"
            f"betas_fithrf_GLMdenoise_RR/{hemi}.betas_session{ses:02d}.mgh")


def fetch_trial(args):
    subj, ses, frame = args
    start = HDR + frame * VERTS_FS7 * 4
    end = start + VERTS_FS5 * 4 - 1
    out = np.empty(2 * VERTS_FS5, dtype=np.float32)
    for i, hemi in enumerate(("lh", "rh")):
        raw = fetch(beta_url(subj, hemi, ses), start, end)
        out[i * VERTS_FS5:(i + 1) * VERTS_FS5] = np.frombuffer(raw, dtype=">f4")
    return out


def fetch_ncsnr(subj):
    path = OUT / f"ncsnr_subj{subj:02d}.npy"
    if path.exists():
        return
    arrs = []
    for hemi in ("lh", "rh"):
        url = (f"{BUCKET}/nsddata_betas/ppdata/subj{subj:02d}/fsaverage/"
               f"betas_fithrf_GLMdenoise_RR/{hemi}.ncsnr.mgh")
        raw = fetch(url, HDR, HDR + VERTS_FS5 * 4 - 1)
        arrs.append(np.frombuffer(raw, dtype=">f4").astype(np.float32))
    np.save(path, np.concatenate(arrs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 9)))
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    idx = pd.read_csv(Path(__file__).parent / "nsd_trial_index.csv")

    for subj in args.subjects:
        out_path = OUT / f"betas_subj{subj:02d}.npz"
        if out_path.exists():
            print(f"subj{subj:02d}: exists, skipping", flush=True)
            continue
        sub = idx[idx["subject"] == subj].reset_index(drop=True)
        print(f"subj{subj:02d}: {len(sub)} trials, {args.workers} workers", flush=True)
        fetch_ncsnr(subj)

        t0 = time.time()
        jobs = list(zip(sub["subject"], sub["session"], sub["frame"]))
        betas = np.empty((len(sub), 2 * VERTS_FS5), dtype=np.float32)
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for i, arr in enumerate(ex.map(fetch_trial, jobs)):
                betas[i] = arr
                if (i + 1) % 500 == 0:
                    el = time.time() - t0
                    print(f"  {i+1}/{len(sub)}  {el:.0f}s  "
                          f"eta {el/(i+1)*(len(sub)-i-1):.0f}s", flush=True)

        np.savez_compressed(
            out_path, betas=betas,
            **{c: sub[c].to_numpy() for c in
               ["nsdId", "cocoId", "rep", "trial", "session", "frame"]},
            image_set=sub["image_set"].to_numpy().astype("U12"),
        )
        print(f"subj{subj:02d}: done in {time.time()-t0:.0f}s -> "
              f"{out_path.stat().st_size/1e6:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
