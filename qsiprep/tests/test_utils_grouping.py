"""Tests for the grouping utils."""

import os

from bids.layout import BIDSLayout
from niworkflows.utils.testing import generate_bids_skeleton

from qsiprep.utils import grouping

dset_multipartid = {
    '01': [
        {
            'dwi': [
                {
                    'acq': '99dir',
                    'dir': 'AP',
                    'run': '1',
                    'suffix': 'dwi',
                    'metadata': {
                        'MultipartID': 'apgroup',
                    },
                },
                {
                    'acq': '98dir',
                    'dir': 'AP',
                    'run': '2',
                    'suffix': 'dwi',
                    'metadata': {
                        'MultipartID': 'apgroup',
                    },
                },
                {
                    'acq': '99dir',
                    'dir': 'AP',
                    'run': '3',
                    'suffix': 'dwi',
                },
            ],
        },
    ],
}
dset_entities = {
    '01': [
        {
            'dwi': [
                {
                    'acq': '99dir',
                    'dir': 'AP',
                    'run': '1',
                    'suffix': 'dwi',
                },
                {
                    'acq': '98dir',
                    'dir': 'AP',
                    'run': '2',
                    'suffix': 'dwi',
                },
                {
                    'acq': '99dir',
                    'dir': 'AP',
                    'run': '3',
                    'suffix': 'dwi',
                },
            ],
        },
    ],
}


def test_get_entity_groups_with_multipartid(tmpdir):
    """Test the get_entity_groups function."""
    bids_dir = tmpdir / 'test_get_entity_groups'
    generate_bids_skeleton(str(bids_dir), dset_multipartid)
    layout = BIDSLayout(str(bids_dir))
    subject_data = {'dwi': layout.get(suffix='dwi', extension='nii.gz')}
    entity_groups = grouping.get_entity_groups(layout, subject_data, combine_all_dwis=True)
    expected = [
        [
            'sub-01_acq-98dir_dir-AP_run-2_dwi.nii.gz',
            'sub-01_acq-99dir_dir-AP_run-1_dwi.nii.gz',
        ],
        ['sub-01_acq-99dir_dir-AP_run-3_dwi.nii.gz'],
    ]
    check_expected(entity_groups, expected)

    entity_groups = grouping.get_entity_groups(layout, subject_data, combine_all_dwis=False)
    expected = [
        ['sub-01_acq-98dir_dir-AP_run-2_dwi.nii.gz'],
        ['sub-01_acq-99dir_dir-AP_run-1_dwi.nii.gz'],
        ['sub-01_acq-99dir_dir-AP_run-3_dwi.nii.gz'],
    ]
    check_expected(entity_groups, expected)


def test_get_entity_groups_without_multipartid(tmpdir):
    """Test the get_entity_groups function."""
    bids_dir = tmpdir / 'test_get_entity_groups'
    generate_bids_skeleton(str(bids_dir), dset_entities)
    layout = BIDSLayout(str(bids_dir))
    subject_data = {'dwi': layout.get(suffix='dwi', extension='nii.gz')}
    entity_groups = grouping.get_entity_groups(layout, subject_data, combine_all_dwis=True)
    expected = [
        [
            'sub-01_acq-98dir_dir-AP_run-2_dwi.nii.gz',
            'sub-01_acq-99dir_dir-AP_run-1_dwi.nii.gz',
            'sub-01_acq-99dir_dir-AP_run-3_dwi.nii.gz',
        ],
    ]
    check_expected(entity_groups, expected)

    entity_groups = grouping.get_entity_groups(layout, subject_data, combine_all_dwis=False)
    expected = [
        ['sub-01_acq-98dir_dir-AP_run-2_dwi.nii.gz'],
        ['sub-01_acq-99dir_dir-AP_run-1_dwi.nii.gz'],
        ['sub-01_acq-99dir_dir-AP_run-3_dwi.nii.gz'],
    ]
    check_expected(entity_groups, expected)


def check_expected(subject_data, expected):
    """Check expected values."""
    if isinstance(expected, str):
        assert subject_data is not None, 'subject_data is None.'
        assert os.path.basename(subject_data) == expected
    elif isinstance(expected, list):
        assert subject_data is not None, 'subject_data is None.'
        assert len(subject_data) == len(expected)
        for item, expected_item in zip(subject_data, expected, strict=False):
            if isinstance(expected_item, list):
                # Handle nested lists
                assert isinstance(item, list), f'Expected list but got {type(item)}'
                assert len(item) == len(expected_item)
                for subitem, expected_subitem in zip(item, expected_item, strict=False):
                    assert os.path.basename(subitem) == expected_subitem
            else:
                assert os.path.basename(item) == expected_item
    else:
        assert subject_data is expected
