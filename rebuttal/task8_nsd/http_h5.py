"""Minimal read-only file-like object backed by HTTP range requests, so h5py's
`fileobj` driver can read a few images out of the 39.6 GB nsd_stimuli.hdf5 without
downloading it. No extra dependencies (no ros3, no aiohttp/fsspec).
"""

import io
import urllib.request


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url, block_size=1 << 20):
        self.url = url
        self._pos = 0
        self.block_size = block_size
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as r:
            self.size = int(r.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        else:
            self._pos = self.size + offset
        return self._pos

    def tell(self):
        return self._pos

    def read(self, size=-1):
        if size < 0:
            size = self.size - self._pos
        if size == 0 or self._pos >= self.size:
            return b""
        end = min(self._pos + size, self.size) - 1
        req = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self._pos}-{end}"}
        )
        with urllib.request.urlopen(req) as r:
            data = r.read()
        self._pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


NSD_STIMULI_URL = (
    "https://natural-scenes-dataset.s3.amazonaws.com/"
    "nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5"
)
