#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Data grabbers
"""
from __future__ import print_function, division, absolute_import, unicode_literals

from ..data.utils import fetch_file


OSF_PROJECT_URL = ('https://files.osf.io/v1/resources/fvuh8/providers/osfstorage/')
OSF_RESOURCES = {
    'brainweb': ('57f32b96b83f6901f194c3ca', '384263fbeadc8e2cca92ced98f224c4b'),
    # Conte69-atlas meshes in 32k resolution
    'conte69': ('5b198ec5ec24e20011b48548', 'bd944e3f9f343e0e51e562b440960529'),
    'ds003_downsampled': ('57f328f6b83f6901ef94cf70', '5a558961c1eb5e5f162696d8afa956e8'),
    'fMRIPrep': ('5bc12155ac011000176bff82', '1aec4d286bd89f4f90316ce2cde63218'),
    # conversion between fsaverage5/6 and fs_LR(32k)
    'hcpLR32k': ('5b198ec6b796ba000f3e4858', '0ba9adcaa42fa88616a4cea5a1ce0c5a'),
    'MNI152Lin': ('5bc984f8ccdb6b0018abb993', 'a1eb0a121d2aa68fc8a6e75ab148f1db'),
    'MNI152NLin2009cAsym': ('5b0dbce20f461a000db8fa3d', '5d386d7db9c1dec30230623db25e05e1'),
    'NKI': ('5bc3fad82aa873001bc5a553', '092e56fb3700f9f57b8917a0db887db6'),
    'OASIS30ANTs': ('5b0dbce34c28ef0012c7f788', 'f625a0390eb32a7852c7b0d71ac428cd'),
    # Mindboggle DKT31 label atlas in MNI152NLin2009cAsym space
    'OASISTRT20': ('5b16f17aeca4a80012bd7542', '1b5389bc3a895b2bd5c0d47401107176'),
}

BIDS_EXAMPLES = {
    'BIDS-examples-1-1.0.0-rc3u5': (
        'https://github.com/chrisfilo/BIDS-examples-1/archive/1.0.0-rc3u5.tar.gz',
        '035fe54445c56eff5bd845ef3795fd56'),
    'BIDS-examples-1-enh-ds054': (
        'http://github.com/chrisfilo/BIDS-examples-1/archive/enh/ds054.zip',
        '56cee272860624924bc23efbe868acb7'),
}

# Map names of templates to OSF_RESOURCES keys
TEMPLATE_ALIASES = {
    'MNI152Lin': 'MNI152Lin',
    'MNI152NLin2009cAsym': 'MNI152NLin2009cAsym',
    'OASIS': 'OASIS30ANTs',
    'NKI': 'NKI',
}


def get_dataset(dataset_name, dataset_prefix=None, data_dir=None,
                url=None, resume=True, verbose=1):
    """Download and load the BIDS-fied brainweb 1mm normal

    :param str data_dir: path of the data directory. Used to force data storage
        in a non-standard location.
    :param str url: download URL of the dataset. Overwrite the default URL.

    """
    file_id, md5 = OSF_RESOURCES[dataset_name]
    if url is None:
        url = '{}/{}'.format(OSF_PROJECT_URL, file_id)
    return fetch_file(dataset_name, url, data_dir, dataset_prefix=dataset_prefix,
                      filetype='tar', resume=resume, verbose=verbose, md5sum=md5)


def get_template(template_name, data_dir=None, url=None, resume=True, verbose=1):
    """Download and load a template"""
    if template_name.startswith('tpl-'):
        template_name = template_name[4:]

    # An aliasing mechanism. Please avoid
    template_name = TEMPLATE_ALIASES.get(template_name, template_name)
    return get_dataset(template_name, dataset_prefix='tpl-', data_dir=data_dir,
                       url=url, resume=resume, verbose=verbose)


def get_brainweb_1mm_normal(data_dir=None, url=None, resume=True, verbose=1):
    """Download and load the BIDS-fied brainweb 1mm normal

    :param str data_dir: path of the data directory. Used to force data storage
        in a non-standard location.
    :param str url: download URL of the dataset. Overwrite the default URL.

    """
    return get_dataset('brainweb', data_dir=data_dir, url=url,
                       resume=resume, verbose=verbose)


def get_ds003_downsampled(data_dir=None, url=None, resume=True, verbose=1):
    """Download and load the BIDS-fied ds003_downsampled

    :param str data_dir: path of the data directory. Used to force data storage
        in a non-standard location.
    :param str url: download URL of the dataset. Overwrite the default URL.

    """
    return get_dataset('ds003_downsampled', data_dir=data_dir, url=url,
                       resume=resume, verbose=verbose)


def get_bids_examples(data_dir=None, url=None, resume=True, verbose=1,
                      variant='BIDS-examples-1-1.0.0-rc3u5'):
    """Download BIDS-examples-1"""
    variant = 'BIDS-examples-1-1.0.0-rc3u5' if variant not in BIDS_EXAMPLES else variant
    if url is None:
        url = BIDS_EXAMPLES[variant][0]
    md5 = BIDS_EXAMPLES[variant][1]
    return fetch_file(variant, url, data_dir, resume=resume, verbose=verbose,
                      md5sum=md5)
