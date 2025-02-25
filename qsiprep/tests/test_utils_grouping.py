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

dset_fmap_intendedfor_relpath = {
    '01': [
        {
            'fmap': [
                {
                    'dir': 'PA',
                    'suffix': 'epi',
                    'metadata': {
                        'IntendedFor': ['dwi/sub-01_dir-AP_dwi.nii.gz'],
                    },
                },
            ],
            'dwi': [
                {
                    'dir': 'AP',
                    'suffix': 'dwi',
                },
            ],
        },
    ],
}
dset_fmap_intendedfor_bidsuri = {
    '01': [
        {
            'fmap': [
                {
                    'dir': 'PA',
                    'suffix': 'epi',
                    'metadata': {
                        'IntendedFor': ['bids::sub-01/dwi/sub-01_dir-AP_dwi.nii.gz'],
                    },
                },
            ],
            'dwi': [
                {
                    'dir': 'AP',
                    'suffix': 'dwi',
                },
            ],
        },
    ],
}
dset_fmap_b0fields = {
    '01': [
        {
            'fmap': [
                {
                    'dir': 'PA',
                    'suffix': 'epi',
                    'metadata': {
                        'B0FieldIdentifier': 'pepolar',
                    },
                },
            ],
            'dwi': [
                {
                    'dir': 'AP',
                    'suffix': 'dwi',
                    'metadata': {
                        'B0FieldIdentifier': 'pepolar',
                        'B0FieldSource': 'pepolar',
                    },
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


def test_get_fieldmaps(tmp_path_factory):
    """Test the get_fieldmaps function."""
    base_dir = tmp_path_factory.mktemp('test_get_fieldmaps')

    # Check that relative paths are correctly handled
    bids_dir = base_dir / 'dset_fmap_intendedfor_relpath'
    generate_bids_skeleton(str(bids_dir), dset_fmap_intendedfor_relpath)
    layout = BIDSLayout(str(bids_dir))
    dwi_file = layout.get(suffix='dwi', extension='nii.gz', return_type='filename')[0]
    fieldmaps = layout.get_fieldmap(dwi_file, return_list=True)
    assert len(fieldmaps) == 1
    assert fieldmaps[0]['suffix'] == 'epi'
    assert fieldmaps[0]['metadata']['IntendedFor'] == ['dwi/sub-01_dir-AP_dwi.nii.gz']

    # Check that BIDS URI paths are correctly handled
    bids_dir = base_dir / 'dset_fmap_intendedfor_bidsuri'
    generate_bids_skeleton(str(bids_dir), dset_fmap_intendedfor_bidsuri)
    layout = BIDSLayout(str(bids_dir))
    dwi_file = layout.get(suffix='dwi', extension='nii.gz', return_type='filename')[0]
    fieldmaps = layout.get_fieldmap(dwi_file, return_list=True)
    assert len(fieldmaps) == 1
    assert fieldmaps[0]['suffix'] == 'epi'
    assert fieldmaps[0]['metadata']['IntendedFor'] == ['bids::sub-01/dwi/sub-01_dir-AP_dwi.nii.gz']

    # Check that B0FieldIdentifier and B0FieldSource are correctly handled
    bids_dir = base_dir / 'dset_fmap_b0fields'
    generate_bids_skeleton(str(bids_dir), dset_fmap_b0fields)
    layout = BIDSLayout(str(bids_dir))
    dwi_file = layout.get(suffix='dwi', extension='nii.gz', return_type='filename')[0]
    fieldmaps = layout.get_fieldmap(dwi_file, return_list=True)
    assert len(fieldmaps) == 1
    assert fieldmaps[0]['suffix'] == 'epi'


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
