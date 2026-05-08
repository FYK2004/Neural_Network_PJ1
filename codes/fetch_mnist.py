"""缺省时下载 MNIST 四个 .gz（stdlib）。"""
from __future__ import annotations

import os
import urllib.error
import urllib.request

MNIST_FILES = [
    'train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz',
    't10k-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz',
]

_BASE_URLS = (
    'https://ossci-datasets.s3.amazonaws.com/mnist/',
    'http://yann.lecun.com/exdb/mnist/',
)


def _download_one(url: str, dest_path: str, chunk: int = 1 << 15) -> None:
    tmp = dest_path + '.tmp'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            with open(tmp, 'wb') as out:
                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    out.write(buf)
        os.replace(tmp, dest_path)
    finally:
        if os.path.isfile(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def ensure_mnist(root: str) -> None:
    """
    Ensure all four MNIST gzip files exist under ``root`` (download if absent).
    """
    root = os.path.normpath(root)
    os.makedirs(root, exist_ok=True)

    for name in MNIST_FILES:
        path = os.path.join(root, name)
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            continue

        last_err: Exception | None = None
        for base in _BASE_URLS:
            url = base + name
            try:
                print(f'[mnist] downloading {name} ...')
                _download_one(url, path)
                break
            except (urllib.error.URLError, OSError, TimeoutError) as e:
                last_err = e
        else:
            raise RuntimeError(
                f'Could not download {name}. Last error: {last_err!r}. '
                'Place files manually under ' + root + ' or check network / firewall.'
            ) from last_err
