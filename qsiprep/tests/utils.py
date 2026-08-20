"""Utility functions for tests."""

import json
import lzma
import os
import tarfile
from glob import glob
from gzip import GzipFile
from io import BytesIO
from pathlib import Path

import nibabel as nb
import numpy as np
import requests
from nipype import logging
from niworkflows.utils.testing import generate_bids_skeleton

from qsiprep import config

LOGGER = logging.getLogger('nipype.utils')

# A complex-valued DWI acquisition: the magnitude and phase parts of one run.
# Neither part carries its own gradients or metadata, so both must reach the
# shared, non-part-specific files through the BIDS inheritance principle.
COMPLEX_DWI_SKELETON = {
    '01': [
        {
            'dwi': [
                {'part': 'mag', 'suffix': 'dwi'},
                {'part': 'phase', 'suffix': 'dwi'},
            ],
        },
    ],
}

# Gradients shared by every part of the acquisition above.
SHARED_DWI_GRADIENTS = {
    'sub-01/dwi/sub-01_dwi.bval': '0 1000\n',
    'sub-01/dwi/sub-01_dwi.bvec': '1 0\n0 1\n0 0\n',
}

# The complex-valued equivalent for an EPI fieldmap, with a shared "secret" bval.
COMPLEX_EPI_SKELETON = {
    '01': [
        {
            'fmap': [
                {'dir': 'PA', 'part': 'mag', 'suffix': 'epi'},
                {'dir': 'PA', 'part': 'phase', 'suffix': 'epi'},
            ],
        },
    ],
}

SHARED_EPI_GRADIENTS = {'sub-01/fmap/sub-01_dir-PA_epi.bval': '0 2000 0\n'}


def build_test_dataset(root, skeleton, extra_files=None, n_volumes=1, affine=None):
    """Build a small BIDS dataset from a ``generate_bids_skeleton`` description.

    ``generate_bids_skeleton`` only creates empty ``.nii.gz`` files, and it can
    only write a sidecar next to the image it describes. This wrapper fills the
    images with real data so they can be loaded, and writes any additional files
    the skeleton cannot express -- sidecars placed higher up the hierarchy for
    the inheritance principle, and ``.bval``/``.bvec`` files.

    Parameters
    ----------
    root : :obj:`str` or :obj:`pathlib.Path`
        Where to build the dataset. Must not already exist.
    skeleton : :obj:`dict`
        A ``generate_bids_skeleton`` dataset description.
    extra_files : :obj:`dict`, optional
        Maps a dataset-relative path to its contents. A :obj:`dict` value is
        written as JSON, a :obj:`str` value verbatim.
    n_volumes : :obj:`int`, optional
        Number of volumes to give each generated image. The default of 1 writes
        3D images.
    affine : :obj:`numpy.ndarray`, optional
        Affine to give each generated image. Defaults to an identity affine,
        which is RAS+.

    Returns
    -------
    :obj:`pathlib.Path`
        The dataset root.
    """
    root = Path(root)
    generate_bids_skeleton(str(root), skeleton)

    affine = np.eye(4) if affine is None else affine
    shape = (2, 2, 2) if n_volumes == 1 else (2, 2, 2, n_volumes)
    for nifti_file in sorted(root.glob('**/*.nii.gz')):
        nb.Nifti1Image(np.zeros(shape, dtype=np.float32), affine).to_filename(nifti_file)

    for relative_path, contents in (extra_files or {}).items():
        target = root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(contents) if isinstance(contents, dict) else contents)

    return root


def download_test_data(dset, data_dir=None):
    """Download test data."""
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
        # CS-DSI (HASC55) reverse-PE *series*, downsampled + defaced -- exercises
        # the non-shelled DIFFPREP rpe_series path through stock DRBUDDI.
        # Extracts to a ``csdsi_hasc55/`` BIDS root (sub-2345/ses-1, AP+PA HASC55
        # + defaced T1w/T2w).
        'csdsi_rpe_series': (
            'https://upenn.box.com/shared/static/3mmagbtddgb4lpmlc5vs4jnsyf1etp3d.xz'
        ),
        'drbuddi_epi': 'https://upenn.box.com/shared/static/plyuee1nbj9v8eck03s38ojji8tkspwr.xz',
        'DSDTI_fmap': 'https://upenn.box.com/shared/static/rxr6qbi6ezku9gw3esfpnvqlcxaw7n5n.gz',
        'DSCSDSI_fmap': 'https://upenn.box.com/shared/static/l561psez1ojzi4p3a12eidaw9vbizwdc.gz',
        'maternal_brain_project': (
            'https://upenn.box.com/shared/static/tkahg1ctipmfihvpa1gmibvcv0gb721h.xz'
        ),
        'forrest_gump': 'https://upenn.box.com/shared/static/qat58an322bzzyixrrsk7cmf52q3bepq.xz',
        'nibs': 'https://upenn.box.com/shared/static/bkllff4ik51jy9ju6nben2r5zrq4a5me.xz',
    }
    if dset == '*':
        for k in URLS:
            download_test_data(k, data_dir=data_dir)

        return

    if dset not in URLS:
        raise ValueError(f'dset ({dset}) must be one of: {", ".join(URLS.keys())}')

    if not data_dir:
        data_dir = os.path.join(os.path.dirname(get_test_data_path()), 'test_data')

    out_dir = os.path.join(data_dir, dset)

    if os.path.isdir(out_dir):
        config.loggers.utils.info(
            f'Dataset {dset} already exists. '
            'If you need to re-download the data, please delete the folder.'
        )
        return out_dir
    else:
        config.loggers.utils.info(f'Downloading {dset} to {out_dir}')

    os.makedirs(out_dir, exist_ok=True)
    url = URLS[dset]
    with requests.get(url, stream=True, timeout=60) as req:
        if url.endswith('.xz'):
            with lzma.open(BytesIO(req.content)) as f:
                with tarfile.open(fileobj=f) as t:
                    t.extractall(out_dir)  # noqa: S202
        elif url.endswith('.gz'):
            with tarfile.open(fileobj=GzipFile(fileobj=BytesIO(req.content))) as t:
                t.extractall(out_dir)  # noqa: S202
        else:
            raise ValueError(f'Unknown file type for {dset} ({url})')

    return out_dir


def field_of_view(img):
    """Return the spatial extent of an image in mm."""
    import numpy as np

    return np.array(img.shape[:3]) * np.array(img.header.get_zooms()[:3])


def get_test_data_path():
    """Return the path to test datasets, terminated with separator.

    Test-related data are kept in tests folder in "data".
    Based on function by Yaroslav Halchenko used in Neurosynth Python package.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), 'data') + os.path.sep)


def check_generated_files(output_dir, output_list_file, optional_output_list_file):
    """Compare files generated by qsiprep with a list of expected files."""
    found_files = sorted(glob(os.path.join(output_dir, '**/*'), recursive=True))
    found_files = [os.path.relpath(f, output_dir) for f in found_files]

    # Ignore figures
    found_files = sorted({f for f in found_files if 'figures' not in f})

    # Ignore logs
    found_files = sorted({f for f in found_files if 'log' not in f.split(os.path.sep)})

    with open(output_list_file) as fo:
        expected_files = fo.readlines()
        expected_files = [f.rstrip() for f in expected_files]

    optional_files = []
    if optional_output_list_file:
        with open(optional_output_list_file) as fo:
            optional_files = fo.readlines()
            optional_files = [f.rstrip() for f in optional_files]

    if sorted(found_files) != sorted(expected_files):
        expected_not_found = sorted(set(expected_files) - set(found_files))
        found_not_expected = sorted(set(found_files) - set(expected_files))

        msg = ''
        if expected_not_found:
            msg += '\nExpected but not found:\n\t'
            msg += '\n\t'.join(expected_not_found)

        if found_not_expected:
            # Check that the found files are in the optional file list
            found_not_expected = [f for f in found_not_expected if f not in optional_files]

        if found_not_expected:
            msg += '\nFound but not expected:\n\t'
            msg += '\n\t'.join(found_not_expected)

        if msg:
            raise ValueError(msg)


def reorder_expected_outputs():
    """Load each of the expected output files and sort the lines alphabetically.

    This function is called manually by devs when they modify the test outputs.
    """
    test_data_path = get_test_data_path()
    expected_output_files = sorted(glob(os.path.join(test_data_path, '*_outputs.txt')))
    for expected_output_file in expected_output_files:
        LOGGER.info(f'Sorting {expected_output_file}')

        with open(expected_output_file) as fo:
            file_contents = fo.readlines()

        file_contents = sorted(set(file_contents))

        with open(expected_output_file, 'w') as fo:
            fo.writelines(file_contents)
