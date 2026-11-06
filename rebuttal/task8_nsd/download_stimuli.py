"""Download the 1106 NSD stimuli (shared-1000 + our 106 overlap images) as presented.

nsd_stimuli.hdf5 stores imgBrick as (73000, 425, 425, 3) uint8 CONTIGUOUS (chunks=None),
so image i is one 542 KB block at a computable offset. We read the dataset's base offset
from the file's own metadata via h5py, then fetch each image with a single range request
rather than downloading the 39.6 GB file.

Output: stimuli/{nsdId:05d}.png  plus stimuli_index.csv
"""

import argparse
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent))
from http_h5 import HTTPRangeFile, NSD_STIMULI_URL

HERE = Path(__file__).parent
OUT = HERE / "stimuli"
OUT.mkdir(exist_ok=True)
H, W, C = 425, 425, 3
IMG_BYTES = H * W * C


def fetch(url, start, end, retries=5):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=180) as r:
                data = r.read()
            if len(data) == end - start + 1:
                return data
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(2 ** attempt)
    raise RuntimeError(f"failed: {start}-{end}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    idx = pd.read_csv(HERE / "nsd_trial_index.csv")
    imgs = (idx[["nsdId", "cocoId", "image_set"]]
            .drop_duplicates("nsdId").sort_values("nsdId").reset_index(drop=True))
    imgs.to_csv(HERE / "stimuli_index.csv", index=False)
    print(f"{len(imgs)} unique images "
          f"({(imgs.image_set=='shared1000').sum()} shared, "
          f"{(imgs.image_set=='overlap106').sum()} overlap)")

    # Ask the file where imgBrick's data actually starts.
    f = HTTPRangeFile(NSD_STIMULI_URL)
    with h5py.File(f, "r", driver="fileobj") as h5:
        d = h5["imgBrick"]
        base = d.id.get_offset()
        assert d.chunks is None and d.shape == (73000, H, W, C), (d.chunks, d.shape)
    print(f"imgBrick contiguous at byte offset {base}")

    def grab(row):
        nsd_id = int(row.nsdId)
        path = OUT / f"{nsd_id:05d}.png"
        if path.exists():
            return
        start = base + nsd_id * IMG_BYTES
        raw = fetch(NSD_STIMULI_URL, start, start + IMG_BYTES - 1)
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(H, W, C)
        Image.fromarray(arr).save(path)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, _ in enumerate(ex.map(grab, imgs.itertuples())):
            if (i + 1) % 100 == 0:
                el = time.time() - t0
                print(f"  {i+1}/{len(imgs)}  {el:.0f}s  "
                      f"eta {el/(i+1)*(len(imgs)-i-1):.0f}s", flush=True)
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
