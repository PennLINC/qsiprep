#!/usr/bin/env python3
"""Download test data without requiring a full QSIPrep install."""

from __future__ import annotations

import argparse
import gzip
import lzma
import tarfile
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

URLS = {
    'HBCD': 'https://upenn.box.com/shared/static/gn1ec8x7mtk1f07l97d0th9idn4qv3yx.xz',
    'DSCSDSI': 'https://upenn.box.com/shared/static/eq6nvnyazi2zlt63uowqd0zhnlh6z4yv.xz',
    'DSCSDSI_BUDS': 'https://upenn.box.com/shared/static/bvhs3sw2swdkdyekpjhnrhvz89x3k87t.xz',
    'DSDTI': 'https://upenn.box.com/shared/static/iefjtvfez0c2oug0g1a9ulozqe5il5xy.xz',
    'twoses': 'https://upenn.box.com/shared/static/c949fjjhhen3ihgnzhkdw5jympm327pp.xz',
    'multishell_output': (
        'https://upenn.box.com/shared/static/hr7xnxicbx9iqndv1yl35bhtd61fpalp.xz'
    ),
    'singleshell_output': (
        'https://upenn.box.com/shared/static/9jhf0eo3ml6ojrlxlz6lej09ny12efgg.gz'
    ),
    'drbuddi_rpe_series': (
        'https://upenn.box.com/shared/static/j5mxts5wu0em1toafmrlzdndves1jnfv.xz'
    ),
    'drbuddi_epi': 'https://upenn.box.com/shared/static/plyuee1nbj9v8eck03s38ojji8tkspwr.xz',
    'DSDTI_fmap': 'https://upenn.box.com/shared/static/rxr6qbi6ezku9gw3esfpnvqlcxaw7n5n.gz',
    'DSCSDSI_fmap': 'https://upenn.box.com/shared/static/l561psez1ojzi4p3a12eidaw9vbizwdc.gz',
    'maternal_brain_project': (
        'https://upenn.box.com/shared/static/tkahg1ctipmfihvpa1gmibvcv0gb721h.xz'
    ),
    'forrest_gump': 'https://upenn.box.com/shared/static/qat58an322bzzyixrrsk7cmf52q3bepq.xz',
}


def download_dataset(name: str, data_dir: Path) -> Path:
    if name not in URLS:
        raise ValueError(f'Unknown dataset {name!r}. Valid options: {", ".join(URLS)}')

    out_dir = data_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    url = URLS[name]

    # Skip download if already present
    if any(out_dir.iterdir()):
        return out_dir

    if not url.startswith(('http://', 'https://')):
        raise ValueError(f'Unexpected URL scheme for {name}: {url}')

    with urlopen(url, timeout=120) as resp:  # noqa: S310
        payload = resp.read()

    if url.endswith('.xz'):
        with lzma.open(BytesIO(payload)) as fobj:
            with tarfile.open(fileobj=fobj) as tar:
                tar.extractall(out_dir)  # noqa: S202
    elif url.endswith('.gz'):
        with tarfile.open(fileobj=gzip.GzipFile(fileobj=BytesIO(payload))) as tar:
            tar.extractall(out_dir)  # noqa: S202
    else:
        raise ValueError(f'Unknown file type for {name} ({url})')

    return out_dir


def main():
    parser = argparse.ArgumentParser(description='Download QSIPrep test datasets')
    parser.add_argument('data_dir', type=Path, help='Root directory to store datasets')
    parser.add_argument(
        'datasets',
        nargs='+',
        help='Datasets to fetch (use "*" to download all known datasets)',
    )
    args = parser.parse_args()

    datasets = list(URLS) if args.datasets == ['*'] else args.datasets
    data_dir = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        download_dataset(dataset, data_dir=data_dir)


if __name__ == '__main__':
    main()
