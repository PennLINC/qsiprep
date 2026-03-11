"""Tests for the grouping utils."""

import os
from pprint import pformat

import pytest
from bids.layout import BIDSLayout
from niworkflows.utils.testing import generate_bids_skeleton

from qsiprep.tests.utils import get_test_data_path
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
    subject_data = {'dwi': layout.get(suffix='dwi', extension='nii.gz', return_type='file')}
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
    subject_data = {'dwi': layout.get(suffix='dwi', extension='nii.gz', return_type='file')}
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


def test_get_fieldmaps_relpaths(tmp_path_factory):
    """Test the get_fieldmaps function."""
    base_dir = tmp_path_factory.mktemp('test_get_fieldmaps_relpaths')

    # Check that relative paths are correctly handled
    bids_dir = base_dir / 'dset_fmap_intendedfor_relpath'
    generate_bids_skeleton(str(bids_dir), dset_fmap_intendedfor_relpath)
    layout = BIDSLayout(str(bids_dir))
    dwi_file = layout.get(suffix='dwi', extension='nii.gz', return_type='file')[0]
    fieldmaps = layout.get_fieldmap(dwi_file, return_list=True)
    assert len(fieldmaps) == 1
    assert fieldmaps[0]['suffix'] == 'epi'
    assert layout.get_file(fieldmaps[0]['epi']).get_metadata()['IntendedFor'] == [
        'dwi/sub-01_dir-AP_dwi.nii.gz'
    ]


@pytest.mark.xfail(reason='BIDS URI paths not supported by pybids')
def test_get_fieldmaps_bidsuri(tmp_path_factory):
    # Check that BIDS URI paths are correctly handled
    base_dir = tmp_path_factory.mktemp('test_get_fieldmaps_bidsuri')

    # XXX: This currently fails because pybids does not support BIDS URI paths.
    bids_dir = base_dir / 'dset_fmap_intendedfor_bidsuri'
    generate_bids_skeleton(str(bids_dir), dset_fmap_intendedfor_bidsuri)
    layout = BIDSLayout(str(bids_dir))
    dwi_file = layout.get(suffix='dwi', extension='nii.gz', return_type='file')[0]
    fieldmaps = layout.get_fieldmap(dwi_file, return_list=True)
    assert len(fieldmaps) == 1
    assert fieldmaps[0]['suffix'] == 'epi'
    assert layout.get_file(fieldmaps[0]['epi']).get_metadata()['IntendedFor'] == [
        'bids::sub-01/dwi/sub-01_dir-AP_dwi.nii.gz'
    ]


@pytest.mark.xfail(reason='B0Field* not supported by pybids')
def test_get_fieldmaps_b0fields(tmp_path_factory):
    """Test the get_fieldmaps function."""
    base_dir = tmp_path_factory.mktemp('test_get_fieldmaps_b0fields')

    # Check that B0FieldIdentifier and B0FieldSource are correctly handled
    bids_dir = base_dir / 'dset_fmap_b0fields'
    generate_bids_skeleton(str(bids_dir), dset_fmap_b0fields)
    layout = BIDSLayout(str(bids_dir))
    dwi_file = layout.get(suffix='dwi', extension='nii.gz', return_type='file')[0]
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
        assert len(subject_data) == len(expected), pformat(subject_data)
        for item, expected_item in zip(subject_data, expected, strict=False):
            check_expected(item, expected_item)
    elif isinstance(expected, dict):
        assert subject_data is not None, 'subject_data is None.'
        assert len(subject_data) == len(expected), pformat(subject_data)
        for key, value in expected.items():
            assert key in subject_data, f'{key} not in subject_data.'
            check_expected(subject_data[key], value)
    else:
        assert subject_data is expected


@pytest.mark.xfail(reason='B0Field* not supported by pybids')
def test_group_dwi_scans_with_complex_b0fields(tmpdir):
    """Test the group_dwi_scans function.

    In the test dataset, we have the following::

    fmap/
        sub-01_dir-AP_epi.nii.gz
        sub-01_dir-PA_epi.nii.gz
    dwi/
        sub-01_dir-AP_run-1_dwi.nii.gz
        sub-01_dir-AP_run-2_dwi.nii.gz
        sub-01_dir-PA_dwi.nii.gz

    The first two DWI runs have different B0 field identifiers, but link to the same fieldmap.
    The third DWI run has a different B0 field identifier, and links to a different fieldmap.

    We expect the first two DWI runs to be grouped together, and the third DWI run to be grouped
    separately, based on having the same phase encoding direction.
    """
    bids_dir = tmpdir / 'test_group_dwi_scans_with_complex_b0fields'
    dset_yaml = os.path.join(get_test_data_path(), 'skeleton_complex_b0fields.yml')
    generate_bids_skeleton(str(bids_dir), dset_yaml)
    layout = BIDSLayout(str(bids_dir))
    subject_data = {'dwi': layout.get(suffix='dwi', extension='nii.gz', return_type='file')}
    scan_groups, _ = grouping.group_dwi_scans(
        layout=layout,
        subject_data=subject_data,
        combine_scans=True,
        ignore_fieldmaps=False,
    )
    expected = [
        {
            'concatenated_bids_name': 'sub-01_dir-AP',
            'dwi_series': [
                'sub-01_dir-AP_run-1_dwi.nii.gz',
                'sub-01_dir-AP_run-2_dwi.nii.gz',
            ],
            'dwi_series_pedir': 'j',
            'fieldmap_info': {'suffix': None},
        },
        {
            'concatenated_bids_name': 'sub-01_dir-PA',
            'dwi_series': ['sub-01_dir-PA_dwi.nii.gz'],
            'dwi_series_pedir': 'j-',
            'fieldmap_info': {'suffix': None},
        },
    ]
    check_expected(scan_groups, expected)

    scan_groups, _ = grouping.group_dwi_scans(
        layout=layout,
        subject_data=subject_data,
        combine_scans=True,
        ignore_fieldmaps=False,
    )
    expected = [
        [
            'sub-01_dir-AP_run-1_dwi.nii.gz',
            'sub-01_dir-AP_run-2_dwi.nii.gz',
        ],
        ['sub-01_dir-PA_dwi.nii.gz'],
    ]
    check_expected(scan_groups, expected)

    scan_groups, _ = grouping.group_dwi_scans(
        layout=layout,
        subject_data=subject_data,
        combine_scans=False,
        ignore_fieldmaps=False,
    )
    expected = [
        [
            'sub-01_dir-AP_run-1_dwi.nii.gz',
            'sub-01_dir-AP_run-2_dwi.nii.gz',
        ],
        ['sub-01_dir-PA_dwi.nii.gz'],
    ]
    check_expected(scan_groups, expected)

    scan_groups, _ = grouping.group_dwi_scans(
        layout=layout,
        subject_data=subject_data,
        combine_scans=False,
        ignore_fieldmaps=False,
    )
    expected = [
        [
            'sub-01_dir-AP_run-1_dwi.nii.gz',
            'sub-01_dir-AP_run-2_dwi.nii.gz',
        ],
        ['sub-01_dir-PA_dwi.nii.gz'],
    ]
    check_expected(scan_groups, expected)

    scan_groups, _ = grouping.group_dwi_scans(
        layout=layout,
        subject_data=subject_data,
        combine_scans=True,
        ignore_fieldmaps=True,
    )
    expected = [
        [
            'sub-01_dir-AP_run-1_dwi.nii.gz',
            'sub-01_dir-AP_run-2_dwi.nii.gz',
        ],
        ['sub-01_dir-PA_dwi.nii.gz'],
    ]
    check_expected(scan_groups, expected)

    scan_groups, _ = grouping.group_dwi_scans(
        layout=layout,
        subject_data=subject_data,
        combine_scans=True,
        ignore_fieldmaps=False,
    )
    expected = [
        [
            'sub-01_dir-AP_run-1_dwi.nii.gz',
            'sub-01_dir-AP_run-2_dwi.nii.gz',
            'sub-01_dir-PA_dwi.nii.gz',
        ],
    ]
    check_expected(scan_groups, expected)


@pytest.fixture
def simple_multiped_dataset(tmpdir):
    """Create a BIDS dataset with multiple DWI series."""
    bids_dir = tmpdir / 'simple_multiped_dataset'
    dset_yaml = os.path.join(get_test_data_path(), 'skeleton_simple_multiped.yml')
    generate_bids_skeleton(str(bids_dir), dset_yaml)
    layout = BIDSLayout(str(bids_dir))
    subject_data = {'dwi': layout.get(suffix='dwi', extension='nii.gz', return_type='file')}
    return layout, subject_data


@pytest.fixture
def multirun_multiped_dataset(tmpdir):
    """Create a BIDS dataset with multiple DWI series."""
    bids_dir = tmpdir / 'multirun_multiped_dataset'
    dset_yaml = os.path.join(get_test_data_path(), 'skeleton_multirun_multiped.yml')
    generate_bids_skeleton(str(bids_dir), dset_yaml)
    layout = BIDSLayout(str(bids_dir))
    subject_data = {'dwi': layout.get(suffix='dwi', extension='nii.gz', return_type='file')}
    return layout, subject_data


@pytest.fixture
def complex_relpaths_dataset(tmpdir):
    """Create a BIDS dataset with complex relative paths for testing."""
    bids_dir = tmpdir / 'test_group_dwi_scans_with_complex_relpaths'
    dset_yaml = os.path.join(get_test_data_path(), 'skeleton_complex_relpaths.yml')
    generate_bids_skeleton(str(bids_dir), dset_yaml)
    layout = BIDSLayout(str(bids_dir))
    subject_data = {'dwi': layout.get(suffix='dwi', extension='nii.gz', return_type='file')}
    return layout, subject_data


@pytest.mark.parametrize(
    ('combine_scans', 'ignore_fieldmaps', 'expected'),
    [
        # Test case 1: combine_scans=True, ignore_fieldmaps=False
        (
            True,
            False,
            [
                {
                    'concatenated_bids_name': 'sub-01',
                    'dwi_series': ['sub-01_dir-PA_dwi.nii.gz'],
                    'dwi_series_pedir': 'j',
                    'fieldmap_info': {
                        'epi': [
                            'sub-01_dir-AP_epi.nii.gz',
                            'sub-01_dir-PA_epi.nii.gz',
                        ],
                        'rpe_series': [
                            'sub-01_dir-AP_run-1_dwi.nii.gz',
                            'sub-01_dir-AP_run-2_dwi.nii.gz',
                        ],
                        'suffix': 'rpe_series',
                    },
                },
            ],
        ),
        # Test case 2: combine_scans=True, ignore_fieldmaps=True
        (
            True,
            True,
            [
                {
                    'concatenated_bids_name': 'sub-01_dir-AP',
                    'dwi_series': [
                        'sub-01_dir-AP_run-1_dwi.nii.gz',
                        'sub-01_dir-AP_run-2_dwi.nii.gz',
                    ],
                    'dwi_series_pedir': 'j-',
                    'fieldmap_info': {'suffix': None},
                },
                {
                    'concatenated_bids_name': 'sub-01_dir-PA',
                    'dwi_series': ['sub-01_dir-PA_dwi.nii.gz'],
                    'dwi_series_pedir': 'j',
                    'fieldmap_info': {'suffix': None},
                },
            ],
        ),
        # Test case 3: combine_scans=False, ignore_fieldmaps=False
        (
            False,
            False,
            [
                {
                    'concatenated_bids_name': 'sub-01',
                    'dwi_series': ['sub-01_dir-PA_dwi.nii.gz'],
                    'dwi_series_pedir': 'j',
                    'fieldmap_info': {
                        'epi': [
                            'sub-01_dir-AP_epi.nii.gz',
                            'sub-01_dir-PA_epi.nii.gz',
                        ],
                        'rpe_series': [
                            'sub-01_dir-AP_run-1_dwi.nii.gz',
                            'sub-01_dir-AP_run-2_dwi.nii.gz',
                        ],
                        'suffix': 'rpe_series',
                    },
                },
            ],
        ),
        # Test case 4: combine_scans=False, ignore_fieldmaps=True
        (
            False,
            True,
            [
                {
                    'concatenated_bids_name': 'sub-01_dir-AP_run-1',
                    'dwi_series': ['sub-01_dir-AP_run-1_dwi.nii.gz'],
                    'dwi_series_pedir': 'j-',
                    'fieldmap_info': {'suffix': None},
                },
                {
                    'concatenated_bids_name': 'sub-01_dir-AP_run-2',
                    'dwi_series': ['sub-01_dir-AP_run-2_dwi.nii.gz'],
                    'dwi_series_pedir': 'j-',
                    'fieldmap_info': {'suffix': None},
                },
                {
                    'concatenated_bids_name': 'sub-01_dir-PA',
                    'dwi_series': ['sub-01_dir-PA_dwi.nii.gz'],
                    'dwi_series_pedir': 'j',
                    'fieldmap_info': {'suffix': None},
                },
            ],
        ),
    ],
)
@pytest.mark.xfail(reason='Old 2-tuple return signature replaced by 4-dict return')
def test_group_dwi_scans_with_complex_relpaths(
    complex_relpaths_dataset, combine_scans, ignore_fieldmaps, expected
):
    """Test the group_dwi_scans function with complex relative paths.

    This is the same as test_group_dwi_scans_with_complex_b0fields,
    but with IntendedFors using relative paths instead of B0Field* fields.
    """
    layout, subject_data = complex_relpaths_dataset

    scan_groups, _ = grouping.group_dwi_scans(
        layout=layout,
        subject_data=subject_data,
        combine_scans=combine_scans,
        ignore_fieldmaps=ignore_fieldmaps,
    )
    check_expected(scan_groups, expected)


@pytest.mark.parametrize(
    ('combine_scans', 'estimate_per_axis', 'ignore_fieldmaps', 'expected'),
    [
        # Test case 1: combine_scans=True, estimate_per_axis=True, ignore_fieldmaps=False
        # Grouping: LR/RL, PA/AP
        (
            True,
            True,
            False,
            [
                {
                    'concatenated_bids_name': 'sub-01',
                    'dwi_series': ['sub-01_dir-LR_dwi.nii.gz'],
                    'dwi_series_pedir': 'i',
                    'fieldmap_info': {
                        'rpe_series': ['sub-01_dir-RL_dwi.nii.gz'],
                        'suffix': 'rpe_series',
                    },
                },
                {
                    'concatenated_bids_name': 'sub-01',
                    'dwi_series': ['sub-01_dir-PA_dwi.nii.gz'],
                    'dwi_series_pedir': 'j',
                    'fieldmap_info': {
                        'rpe_series': ['sub-01_dir-AP_dwi.nii.gz'],
                        'suffix': 'rpe_series',
                    },
                },
            ],
        ),
        # Test case 2: combine_scans=True, estimate_per_axis=True, ignore_fieldmaps=True
        # Grouping: LR/RL, PA/AP
        (
            True,
            True,
            True,
            [
                {
                    'concatenated_bids_name': 'sub-01',
                    'dwi_series': ['sub-01_dir-LR_dwi.nii.gz'],
                    'dwi_series_pedir': 'i',
                    'fieldmap_info': {
                        'rpe_series': ['sub-01_dir-RL_dwi.nii.gz'],
                        'suffix': 'rpe_series',
                    },
                },
                {
                    'concatenated_bids_name': 'sub-01',
                    'dwi_series': ['sub-01_dir-PA_dwi.nii.gz'],
                    'dwi_series_pedir': 'j',
                    'fieldmap_info': {
                        'rpe_series': ['sub-01_dir-AP_dwi.nii.gz'],
                        'suffix': 'rpe_series',
                    },
                },
            ],
        ),
        # Test case 3: combine_scans=False, estimate_per_axis=True, ignore_fieldmaps=False
        # Grouping: AP, LR, PA, RL
        (
            False,
            True,
            False,
            [
                {
                    'concatenated_bids_name': 'sub-01_dir-AP',
                    'dwi_series': ['sub-01_dir-AP_dwi.nii.gz'],
                    'dwi_series_pedir': 'j-',
                    'fieldmap_info': {'suffix': None},
                },
                {
                    'concatenated_bids_name': 'sub-01_dir-LR',
                    'dwi_series': ['sub-01_dir-LR_dwi.nii.gz'],
                    'dwi_series_pedir': 'i',
                    'fieldmap_info': {'suffix': None},
                },
                {
                    'concatenated_bids_name': 'sub-01_dir-PA',
                    'dwi_series': ['sub-01_dir-PA_dwi.nii.gz'],
                    'dwi_series_pedir': 'j',
                    'fieldmap_info': {'suffix': None},
                },
                {
                    'concatenated_bids_name': 'sub-01_dir-RL',
                    'dwi_series': ['sub-01_dir-RL_dwi.nii.gz'],
                    'dwi_series_pedir': 'i-',
                    'fieldmap_info': {'suffix': None},
                },
            ],
        ),
        # Test case 4: combine_scans=False, estimate_per_axis=True, ignore_fieldmaps=True
        # Grouping: AP, LR, PA, RL
        (
            False,
            True,
            True,
            [
                {
                    'concatenated_bids_name': 'sub-01_dir-AP',
                    'dwi_series': ['sub-01_dir-AP_dwi.nii.gz'],
                    'dwi_series_pedir': 'j-',
                    'fieldmap_info': {'suffix': None},
                },
                {
                    'concatenated_bids_name': 'sub-01_dir-LR',
                    'dwi_series': ['sub-01_dir-LR_dwi.nii.gz'],
                    'dwi_series_pedir': 'i',
                    'fieldmap_info': {'suffix': None},
                },
                {
                    'concatenated_bids_name': 'sub-01_dir-PA',
                    'dwi_series': ['sub-01_dir-PA_dwi.nii.gz'],
                    'dwi_series_pedir': 'j',
                    'fieldmap_info': {'suffix': None},
                },
                {
                    'concatenated_bids_name': 'sub-01_dir-RL',
                    'dwi_series': ['sub-01_dir-RL_dwi.nii.gz'],
                    'dwi_series_pedir': 'i-',
                    'fieldmap_info': {'suffix': None},
                },
            ],
        ),
        # Test case 5: combine_scans=True, estimate_per_axis=False, ignore_fieldmaps=False
        # Grouping: LR/AP/RL/PA
        # XXX: Getting the following instead of the expected:
        # [
        #     {
        #         'concatenated_bids_name': 'sub-01',
        #         'dwi_series': ['sub-01_dir-LR_dwi.nii.gz'],
        #         'dwi_series_pedir': 'i',
        #         'fieldmap_info': {
        #             'rpe_series': ['sub-01_dir-RL_dwi.nii.gz'],
        #             'suffix': 'rpe_series',
        #         },
        #     },
        #     {
        #         'concatenated_bids_name': 'sub-01',
        #         'dwi_series': ['sub-01_dir-PA_dwi.nii.gz'],
        #         'dwi_series_pedir': 'j',
        #         'fieldmap_info': {
        #             'rpe_series': ['sub-01_dir-AP_dwi.nii.gz'],
        #             'suffix': 'rpe_series',
        #         },
        #     },
        # ]
        (
            True,
            False,
            False,
            [
                {
                    'concatenated_bids_name': 'sub-01',
                    'dwi_series': ['sub-01_dir-LR_dwi.nii.gz'],
                    'dwi_series_pedir': 'i',
                    'fieldmap_info': {
                        'rpe_series': [
                            'sub-01_dir-AP_dwi.nii.gz',
                            'sub-01_dir-RL_dwi.nii.gz',
                            'sub-01_dir-PA_dwi.nii.gz',
                        ],
                        'suffix': 'rpe_series',
                    },
                },
            ],
        ),
        # Test case 6: combine_scans=True, estimate_per_axis=False, ignore_fieldmaps=True
        # Grouping: LR/AP/RL/PA
        # XXX: Getting the following instead of the expected:
        # [
        #     {
        #        'concatenated_bids_name': 'sub-01',
        #         'dwi_series': [
        #             'sub-01_dir-AP_dwi.nii.gz',
        #             'sub-01_dir-LR_dwi.nii.gz',
        #             'sub-01_dir-PA_dwi.nii.gz',
        #         ],
        #         'dwi_series_pedir': 'i',
        #         'fieldmap_info': {
        #             'rpe_series': [
        #                 'sub-01_dir-AP_dwi.nii.gz',
        #                 'sub-01_dir-PA_dwi.nii.gz',
        #                 'sub-01_dir-RL_dwi.nii.gz',
        #             ],
        #             'suffix': 'rpe_series',
        #         },
        #     },
        #     {
        #         'concatenated_bids_name': 'sub-01',
        #         'dwi_series': [
        #             'sub-01_dir-LR_dwi.nii.gz',
        #             'sub-01_dir-PA_dwi.nii.gz',
        #             'sub-01_dir-RL_dwi.nii.gz',
        #         ],
        #         'dwi_series_pedir': 'j',
        #         'fieldmap_info': {
        #             'rpe_series': [
        #                 'sub-01_dir-AP_dwi.nii.gz',
        #                 'sub-01_dir-LR_dwi.nii.gz',
        #                 'sub-01_dir-RL_dwi.nii.gz',
        #             ],
        #             'suffix': 'rpe_series',
        #         },
        #     },
        # ]
        (
            True,
            False,
            True,
            [
                {
                    'concatenated_bids_name': 'sub-01',
                    'dwi_series': ['sub-01_dir-LR_dwi.nii.gz'],
                    'dwi_series_pedir': 'i',
                    'fieldmap_info': {
                        'rpe_series': [
                            'sub-01_dir-AP_dwi.nii.gz',
                            'sub-01_dir-RL_dwi.nii.gz',
                            'sub-01_dir-PA_dwi.nii.gz',
                        ],
                        'suffix': 'rpe_series',
                    },
                },
            ],
        ),
        # Test case 7: combine_scans=False, estimate_per_axis=False, ignore_fieldmaps=False
        # Grouping: AP, LR, PA, RL
        (
            False,
            False,
            False,
            [
                {
                    'concatenated_bids_name': 'sub-01_dir-AP',
                    'dwi_series': ['sub-01_dir-AP_dwi.nii.gz'],
                    'dwi_series_pedir': 'j-',
                    'fieldmap_info': {'suffix': None},
                },
                {
                    'concatenated_bids_name': 'sub-01_dir-LR',
                    'dwi_series': ['sub-01_dir-LR_dwi.nii.gz'],
                    'dwi_series_pedir': 'i',
                    'fieldmap_info': {'suffix': None},
                },
                {
                    'concatenated_bids_name': 'sub-01_dir-PA',
                    'dwi_series': ['sub-01_dir-PA_dwi.nii.gz'],
                    'dwi_series_pedir': 'j',
                    'fieldmap_info': {'suffix': None},
                },
                {
                    'concatenated_bids_name': 'sub-01_dir-RL',
                    'dwi_series': ['sub-01_dir-RL_dwi.nii.gz'],
                    'dwi_series_pedir': 'i-',
                    'fieldmap_info': {'suffix': None},
                },
            ],
        ),
        # Test case 8: combine_scans=False, estimate_per_axis=False, ignore_fieldmaps=True
        # Grouping: AP, LR, PA, RL
        (
            False,
            False,
            True,
            [
                {
                    'concatenated_bids_name': 'sub-01_dir-AP',
                    'dwi_series': ['sub-01_dir-AP_dwi.nii.gz'],
                    'dwi_series_pedir': 'j-',
                    'fieldmap_info': {'suffix': None},
                },
                {
                    'concatenated_bids_name': 'sub-01_dir-LR',
                    'dwi_series': ['sub-01_dir-LR_dwi.nii.gz'],
                    'dwi_series_pedir': 'i',
                    'fieldmap_info': {'suffix': None},
                },
                {
                    'concatenated_bids_name': 'sub-01_dir-PA',
                    'dwi_series': ['sub-01_dir-PA_dwi.nii.gz'],
                    'dwi_series_pedir': 'j',
                    'fieldmap_info': {'suffix': None},
                },
                {
                    'concatenated_bids_name': 'sub-01_dir-RL',
                    'dwi_series': ['sub-01_dir-RL_dwi.nii.gz'],
                    'dwi_series_pedir': 'i-',
                    'fieldmap_info': {'suffix': None},
                },
            ],
        ),
    ],
)
@pytest.mark.xfail(reason='Old 2-tuple return signature replaced by 4-dict return')
def test_group_dwi_scans_with_simple_multiped(
    simple_multiped_dataset,
    combine_scans,
    estimate_per_axis,
    ignore_fieldmaps,
    expected,
):
    """Test the group_dwi_scans function with complex relative paths.

    This is the same as test_group_dwi_scans_with_complex_b0fields,
    but with IntendedFors using relative paths instead of B0Field* fields.

    Parameters
    ----------
    combine_scans : :obj:`bool`
        If True, group scans together based on their BIDS entities.
    estimate_per_axis : :obj:`bool`
        If True, limit phase encoding direction-based grouping to reverse PED scans.
    ignore_fieldmaps : :obj:`bool`
        If True, ignore fieldmaps. Placed last in the parameter list because it has no effect.
    expected : :obj:`list` of :obj:`dict`
        The expected output of the group_dwi_scans function.

    Notes
    -----

    If combine_scans is False, the scans should be kept separate regardless of the other
    parameters.

    If combine_scans is True and estimate_per_axis is False, all of the scans should be grouped
    together, with the primary scan (dwi_series) being the one with the alphabetically first
    phase encoding direction.
    XXX: This combination is not working as expected.

    If combine_scans is True and estimate_per_axis is True, the scans should be grouped together
    with the reverse PED scans. So AP/PA and RL/LR, but not AP/LR/PA/RL.
    The primary scan should be the one with the alphabetically first phase encoding direction.

    ignore_fieldmaps should have no effect, since there are no fieldmaps in the dataset.
    """
    layout, subject_data = simple_multiped_dataset

    scan_groups, _ = grouping.group_dwi_scans(
        layout=layout,
        subject_data=subject_data,
        combine_scans=combine_scans,
        estimate_per_axis=estimate_per_axis,
        ignore_fieldmaps=ignore_fieldmaps,
    )
    check_expected(scan_groups, expected)


@pytest.mark.parametrize(
    ('combine_scans', 'estimate_per_axis', 'ignore_fieldmaps', 'expected'),
    [
        # Test case 1: combine_scans=True, estimate_per_axis=True, ignore_fieldmaps=False
        # Grouping: LR/RL, PA/AP
        (
            True,
            True,
            False,
            [
                {
                    'concatenated_bids_name': 'sub-01',
                    'dwi_series': [
                        'sub-01_dir-LR_run-1_dwi.nii.gz',
                        'sub-01_dir-LR_run-2_dwi.nii.gz',
                    ],
                    'dwi_series_pedir': 'i',
                    'fieldmap_info': {
                        'rpe_series': [
                            'sub-01_dir-RL_run-1_dwi.nii.gz',
                            'sub-01_dir-RL_run-2_dwi.nii.gz',
                        ],
                        'suffix': 'rpe_series',
                    },
                },
                {
                    'concatenated_bids_name': 'sub-01',
                    'dwi_series': [
                        'sub-01_dir-PA_run-1_dwi.nii.gz',
                        'sub-01_dir-PA_run-2_dwi.nii.gz',
                    ],
                    'dwi_series_pedir': 'j',
                    'fieldmap_info': {
                        'rpe_series': [
                            'sub-01_dir-AP_run-1_dwi.nii.gz',
                            'sub-01_dir-AP_run-2_dwi.nii.gz',
                        ],
                        'suffix': 'rpe_series',
                    },
                },
            ],
        ),
        # Test case 2: combine_scans=True, estimate_per_axis=True, ignore_fieldmaps=True
        # Grouping: LR/RL, PA/AP
        (
            True,
            True,
            True,
            [
                {
                    'concatenated_bids_name': 'sub-01',
                    'dwi_series': [
                        'sub-01_dir-LR_run-1_dwi.nii.gz',
                        'sub-01_dir-LR_run-2_dwi.nii.gz',
                    ],
                    'dwi_series_pedir': 'i',
                    'fieldmap_info': {
                        'rpe_series': [
                            'sub-01_dir-RL_run-1_dwi.nii.gz',
                            'sub-01_dir-RL_run-2_dwi.nii.gz',
                        ],
                        'suffix': 'rpe_series',
                    },
                },
                {
                    'concatenated_bids_name': 'sub-01',
                    'dwi_series': [
                        'sub-01_dir-PA_run-1_dwi.nii.gz',
                        'sub-01_dir-PA_run-2_dwi.nii.gz',
                    ],
                    'dwi_series_pedir': 'j',
                    'fieldmap_info': {
                        'rpe_series': [
                            'sub-01_dir-AP_run-1_dwi.nii.gz',
                            'sub-01_dir-AP_run-2_dwi.nii.gz',
                        ],
                        'suffix': 'rpe_series',
                    },
                },
            ],
        ),
        # Test case 3: combine_scans=False, estimate_per_axis=True, ignore_fieldmaps=False
        # Grouping: AP, LR, PA, RL
        (
            False,
            True,
            False,
            [
                {
                    'concatenated_bids_name': 'sub-01_dir-AP_run-1',
                    'dwi_series': ['sub-01_dir-AP_run-1_dwi.nii.gz'],
                    'dwi_series_pedir': 'j-',
                    'fieldmap_info': {'suffix': None},
                },
                {
                    'concatenated_bids_name': 'sub-01_dir-AP_run-2',
                    'dwi_series': ['sub-01_dir-AP_run-2_dwi.nii.gz'],
                    'dwi_series_pedir': 'j-',
                    'fieldmap_info': {'suffix': None},
                },
                {
                    'concatenated_bids_name': 'sub-01_dir-LR_run-1',
                    'dwi_series': ['sub-01_dir-LR_run-1_dwi.nii.gz'],
                    'dwi_series_pedir': 'i',
                    'fieldmap_info': {'suffix': None},
                },
                {
                    'concatenated_bids_name': 'sub-01_dir-LR_run-2',
                    'dwi_series': ['sub-01_dir-LR_run-2_dwi.nii.gz'],
                    'dwi_series_pedir': 'i',
                    'fieldmap_info': {'suffix': None},
                },
                {
                    'concatenated_bids_name': 'sub-01_dir-PA_run-1',
                    'dwi_series': ['sub-01_dir-PA_run-1_dwi.nii.gz'],
                    'dwi_series_pedir': 'j',
                    'fieldmap_info': {'suffix': None},
                },
                {
                    'concatenated_bids_name': 'sub-01_dir-PA_run-2',
                    'dwi_series': ['sub-01_dir-PA_run-2_dwi.nii.gz'],
                    'dwi_series_pedir': 'j',
                    'fieldmap_info': {'suffix': None},
                },
                {
                    'concatenated_bids_name': 'sub-01_dir-RL_run-1',
                    'dwi_series': ['sub-01_dir-RL_run-1_dwi.nii.gz'],
                    'dwi_series_pedir': 'i-',
                    'fieldmap_info': {'suffix': None},
                },
                {
                    'concatenated_bids_name': 'sub-01_dir-RL_run-2',
                    'dwi_series': ['sub-01_dir-RL_run-2_dwi.nii.gz'],
                    'dwi_series_pedir': 'i-',
                    'fieldmap_info': {'suffix': None},
                },
            ],
        ),
        # Test case 4: combine_scans=False, estimate_per_axis=True, ignore_fieldmaps=True
        # Grouping: AP, LR, PA, RL
        (
            False,
            True,
            True,
            [
                {
                    'concatenated_bids_name': 'sub-01_dir-AP_run-1',
                    'dwi_series': ['sub-01_dir-AP_run-1_dwi.nii.gz'],
                    'dwi_series_pedir': 'j-',
                    'fieldmap_info': {'suffix': None},
                },
                {
                    'concatenated_bids_name': 'sub-01_dir-AP_run-2',
                    'dwi_series': ['sub-01_dir-AP_run-2_dwi.nii.gz'],
                    'dwi_series_pedir': 'j-',
                    'fieldmap_info': {'suffix': None},
                },
                {
                    'concatenated_bids_name': 'sub-01_dir-LR_run-1',
                    'dwi_series': ['sub-01_dir-LR_run-1_dwi.nii.gz'],
                    'dwi_series_pedir': 'i',
                    'fieldmap_info': {'suffix': None},
                },
                {
                    'concatenated_bids_name': 'sub-01_dir-LR_run-2',
                    'dwi_series': ['sub-01_dir-LR_run-2_dwi.nii.gz'],
                    'dwi_series_pedir': 'i',
                    'fieldmap_info': {'suffix': None},
                },
                {
                    'concatenated_bids_name': 'sub-01_dir-PA_run-1',
                    'dwi_series': ['sub-01_dir-PA_run-1_dwi.nii.gz'],
                    'dwi_series_pedir': 'j',
                    'fieldmap_info': {'suffix': None},
                },
                {
                    'concatenated_bids_name': 'sub-01_dir-PA_run-2',
                    'dwi_series': ['sub-01_dir-PA_run-2_dwi.nii.gz'],
                    'dwi_series_pedir': 'j',
                    'fieldmap_info': {'suffix': None},
                },
                {
                    'concatenated_bids_name': 'sub-01_dir-RL_run-1',
                    'dwi_series': ['sub-01_dir-RL_run-1_dwi.nii.gz'],
                    'dwi_series_pedir': 'i-',
                    'fieldmap_info': {'suffix': None},
                },
                {
                    'concatenated_bids_name': 'sub-01_dir-RL_run-2',
                    'dwi_series': ['sub-01_dir-RL_run-2_dwi.nii.gz'],
                    'dwi_series_pedir': 'i-',
                    'fieldmap_info': {'suffix': None},
                },
            ],
        ),
        # Test case 5: combine_scans=True, estimate_per_axis=False, ignore_fieldmaps=False
        # Grouping: LR/AP/RL/PA
        # XXX: Getting the following instead of the expected:
        # [
        #     {
        #         'concatenated_bids_name': 'sub-01',
        #         'dwi_series': ['sub-01_dir-LR_dwi.nii.gz'],
        #         'dwi_series_pedir': 'i',
        #         'fieldmap_info': {
        #             'rpe_series': ['sub-01_dir-RL_dwi.nii.gz'],
        #             'suffix': 'rpe_series',
        #         },
        #     },
        #     {
        #         'concatenated_bids_name': 'sub-01',
        #         'dwi_series': ['sub-01_dir-PA_dwi.nii.gz'],
        #         'dwi_series_pedir': 'j',
        #         'fieldmap_info': {
        #             'rpe_series': ['sub-01_dir-AP_dwi.nii.gz'],
        #             'suffix': 'rpe_series',
        #         },
        #     },
        # ]
        (
            True,
            False,
            False,
            [
                {
                    'concatenated_bids_name': 'sub-01',
                    'dwi_series': [
                        'sub-01_dir-LR_run-1_dwi.nii.gz',
                        'sub-01_dir-LR_run-2_dwi.nii.gz',
                    ],
                    'dwi_series_pedir': 'i',
                    'fieldmap_info': {
                        'rpe_series': [
                            'sub-01_dir-AP_run-1_dwi.nii.gz',
                            'sub-01_dir-AP_run-2_dwi.nii.gz',
                            'sub-01_dir-RL_run-1_dwi.nii.gz',
                            'sub-01_dir-RL_run-2_dwi.nii.gz',
                            'sub-01_dir-PA_run-1_dwi.nii.gz',
                            'sub-01_dir-PA_run-2_dwi.nii.gz',
                        ],
                        'suffix': 'rpe_series',
                    },
                },
            ],
        ),
        # Test case 6: combine_scans=True, estimate_per_axis=False, ignore_fieldmaps=True
        # Grouping: LR/AP/RL/PA
        # XXX: Getting the following instead of the expected:
        # [
        #     {
        #        'concatenated_bids_name': 'sub-01',
        #         'dwi_series': [
        #             'sub-01_dir-AP_dwi.nii.gz',
        #             'sub-01_dir-LR_dwi.nii.gz',
        #             'sub-01_dir-PA_dwi.nii.gz',
        #         ],
        #         'dwi_series_pedir': 'i',
        #         'fieldmap_info': {
        #             'rpe_series': [
        #                 'sub-01_dir-AP_dwi.nii.gz',
        #                 'sub-01_dir-PA_dwi.nii.gz',
        #                 'sub-01_dir-RL_dwi.nii.gz',
        #             ],
        #             'suffix': 'rpe_series',
        #         },
        #     },
        #     {
        #         'concatenated_bids_name': 'sub-01',
        #         'dwi_series': [
        #             'sub-01_dir-LR_dwi.nii.gz',
        #             'sub-01_dir-PA_dwi.nii.gz',
        #             'sub-01_dir-RL_dwi.nii.gz',
        #         ],
        #         'dwi_series_pedir': 'j',
        #         'fieldmap_info': {
        #             'rpe_series': [
        #                 'sub-01_dir-AP_dwi.nii.gz',
        #                 'sub-01_dir-LR_dwi.nii.gz',
        #                 'sub-01_dir-RL_dwi.nii.gz',
        #             ],
        #             'suffix': 'rpe_series',
        #         },
        #     },
        # ]
        (
            True,
            False,
            True,
            [
                {
                    'concatenated_bids_name': 'sub-01',
                    'dwi_series': [
                        'sub-01_dir-LR_run-1_dwi.nii.gz',
                        'sub-01_dir-LR_run-2_dwi.nii.gz',
                    ],
                    'dwi_series_pedir': 'i',
                    'fieldmap_info': {
                        'rpe_series': [
                            'sub-01_dir-AP_run-1_dwi.nii.gz',
                            'sub-01_dir-AP_run-2_dwi.nii.gz',
                            'sub-01_dir-RL_run-1_dwi.nii.gz',
                            'sub-01_dir-RL_run-2_dwi.nii.gz',
                            'sub-01_dir-PA_run-1_dwi.nii.gz',
                            'sub-01_dir-PA_run-2_dwi.nii.gz',
                        ],
                        'suffix': 'rpe_series',
                    },
                },
            ],
        ),
        # Test case 7: combine_scans=False, estimate_per_axis=False, ignore_fieldmaps=False
        # Grouping: AP, LR, PA, RL
        (
            False,
            False,
            False,
            [
                {
                    'concatenated_bids_name': 'sub-01_dir-AP_run-1',
                    'dwi_series': ['sub-01_dir-AP_run-1_dwi.nii.gz'],
                    'dwi_series_pedir': 'j-',
                    'fieldmap_info': {'suffix': None},
                },
                {
                    'concatenated_bids_name': 'sub-01_dir-AP_run-2',
                    'dwi_series': ['sub-01_dir-AP_run-2_dwi.nii.gz'],
                    'dwi_series_pedir': 'j-',
                    'fieldmap_info': {'suffix': None},
                },
                {
                    'concatenated_bids_name': 'sub-01_dir-LR_run-1',
                    'dwi_series': ['sub-01_dir-LR_run-1_dwi.nii.gz'],
                    'dwi_series_pedir': 'i',
                    'fieldmap_info': {'suffix': None},
                },
                {
                    'concatenated_bids_name': 'sub-01_dir-LR_run-2',
                    'dwi_series': ['sub-01_dir-LR_run-2_dwi.nii.gz'],
                    'dwi_series_pedir': 'i',
                    'fieldmap_info': {'suffix': None},
                },
                {
                    'concatenated_bids_name': 'sub-01_dir-PA_run-1',
                    'dwi_series': ['sub-01_dir-PA_run-1_dwi.nii.gz'],
                    'dwi_series_pedir': 'j',
                    'fieldmap_info': {'suffix': None},
                },
                {
                    'concatenated_bids_name': 'sub-01_dir-PA_run-2',
                    'dwi_series': ['sub-01_dir-PA_run-2_dwi.nii.gz'],
                    'dwi_series_pedir': 'j',
                    'fieldmap_info': {'suffix': None},
                },
                {
                    'concatenated_bids_name': 'sub-01_dir-RL_run-1',
                    'dwi_series': ['sub-01_dir-RL_run-1_dwi.nii.gz'],
                    'dwi_series_pedir': 'i-',
                    'fieldmap_info': {'suffix': None},
                },
                {
                    'concatenated_bids_name': 'sub-01_dir-RL_run-2',
                    'dwi_series': ['sub-01_dir-RL_run-2_dwi.nii.gz'],
                    'dwi_series_pedir': 'i-',
                    'fieldmap_info': {'suffix': None},
                },
            ],
        ),
        # Test case 8: combine_scans=False, estimate_per_axis=False, ignore_fieldmaps=True
        # Grouping: AP, LR, PA, RL
        (
            False,
            False,
            True,
            [
                {
                    'concatenated_bids_name': 'sub-01_dir-AP_run-1',
                    'dwi_series': ['sub-01_dir-AP_run-1_dwi.nii.gz'],
                    'dwi_series_pedir': 'j-',
                    'fieldmap_info': {'suffix': None},
                },
                {
                    'concatenated_bids_name': 'sub-01_dir-AP_run-2',
                    'dwi_series': ['sub-01_dir-AP_run-2_dwi.nii.gz'],
                    'dwi_series_pedir': 'j-',
                    'fieldmap_info': {'suffix': None},
                },
                {
                    'concatenated_bids_name': 'sub-01_dir-LR_run-1',
                    'dwi_series': ['sub-01_dir-LR_run-1_dwi.nii.gz'],
                    'dwi_series_pedir': 'i',
                    'fieldmap_info': {'suffix': None},
                },
                {
                    'concatenated_bids_name': 'sub-01_dir-LR_run-2',
                    'dwi_series': ['sub-01_dir-LR_run-2_dwi.nii.gz'],
                    'dwi_series_pedir': 'i',
                    'fieldmap_info': {'suffix': None},
                },
                {
                    'concatenated_bids_name': 'sub-01_dir-PA_run-1',
                    'dwi_series': ['sub-01_dir-PA_run-1_dwi.nii.gz'],
                    'dwi_series_pedir': 'j',
                    'fieldmap_info': {'suffix': None},
                },
                {
                    'concatenated_bids_name': 'sub-01_dir-PA_run-2',
                    'dwi_series': ['sub-01_dir-PA_run-2_dwi.nii.gz'],
                    'dwi_series_pedir': 'j',
                    'fieldmap_info': {'suffix': None},
                },
                {
                    'concatenated_bids_name': 'sub-01_dir-RL_run-1',
                    'dwi_series': ['sub-01_dir-RL_run-1_dwi.nii.gz'],
                    'dwi_series_pedir': 'i-',
                    'fieldmap_info': {'suffix': None},
                },
                {
                    'concatenated_bids_name': 'sub-01_dir-RL_run-2',
                    'dwi_series': ['sub-01_dir-RL_run-2_dwi.nii.gz'],
                    'dwi_series_pedir': 'i-',
                    'fieldmap_info': {'suffix': None},
                },
            ],
        ),
    ],
)
@pytest.mark.xfail(reason='Old 2-tuple return signature replaced by 4-dict return')
def test_group_dwi_scans_with_multirun_multiped(
    multirun_multiped_dataset,
    combine_scans,
    estimate_per_axis,
    ignore_fieldmaps,
    expected,
):
    """Test the group_dwi_scans function with complex relative paths.

    This is the same as test_group_dwi_scans_with_complex_b0fields,
    but with IntendedFors using relative paths instead of B0Field* fields.

    Parameters
    ----------
    combine_scans : :obj:`bool`
        If True, group scans together based on their BIDS entities.
    estimate_per_axis : :obj:`bool`
        If True, limit phase encoding direction-based grouping to reverse PED scans.
    ignore_fieldmaps : :obj:`bool`
        If True, ignore fieldmaps. Placed last in the parameter list because it has no effect.
    expected : :obj:`list` of :obj:`dict`
        The expected output of the group_dwi_scans function.

    Notes
    -----

    If combine_scans is False, the scans should be kept separate regardless of the other
    parameters.

    If combine_scans is True and estimate_per_axis is False, all of the scans should be grouped
    together, with the primary scan (dwi_series) being the one with the alphabetically first
    phase encoding direction.
    XXX: This combination is not working as expected.

    If combine_scans is True and estimate_per_axis is True, the scans should be grouped together
    with the reverse PED scans. So AP/PA and RL/LR, but not AP/LR/PA/RL.
    The primary scan should be the one with the alphabetically first phase encoding direction.

    ignore_fieldmaps should have no effect, since there are no fieldmaps in the dataset.
    """
    layout, subject_data = multirun_multiped_dataset

    scan_groups, _ = grouping.group_dwi_scans(
        layout=layout,
        subject_data=subject_data,
        combine_scans=combine_scans,
        estimate_per_axis=estimate_per_axis,
        ignore_fieldmaps=ignore_fieldmaps,
    )
    check_expected(scan_groups, expected)


def test_add_acq_entity():
    """Test adding an indexed acq entity to a concatenated BIDS name."""
    assert grouping._add_acq_entity('sub-1_dir-AP', 1) == 'sub-1_acq-1_dir-AP'
    assert grouping._add_acq_entity('sub-1_acq-HCP_dir-AP', 2) == 'sub-1_acq-HCP+2_dir-AP'


def test_get_unique_concatenated_bids_names_without_acq():
    """Duplicate names without acq entities should get indexed acq labels."""
    dwi_groups = [
        [
            '/data/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
            '/data/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz',
        ],
        [
            '/data/sub-1/dwi/sub-1_dir-AP_run-3_dwi.nii.gz',
            '/data/sub-1/dwi/sub-1_dir-AP_run-4_dwi.nii.gz',
        ],
    ]
    assert grouping._get_unique_concatenated_bids_names(dwi_groups) == [
        'sub-1_acq-1_dir-AP',
        'sub-1_acq-2_dir-AP',
    ]


def test_get_unique_concatenated_bids_names_with_acq():
    """Duplicate names with acq entities should append indexed suffixes."""
    dwi_groups = [
        [
            '/data/sub-1/dwi/sub-1_acq-HCP_dir-AP_run-1_dwi.nii.gz',
            '/data/sub-1/dwi/sub-1_acq-HCP_dir-AP_run-2_dwi.nii.gz',
        ],
        [
            '/data/sub-1/dwi/sub-1_acq-HCP_dir-AP_run-3_dwi.nii.gz',
            '/data/sub-1/dwi/sub-1_acq-HCP_dir-AP_run-4_dwi.nii.gz',
        ],
    ]
    assert grouping._get_unique_concatenated_bids_names(dwi_groups) == [
        'sub-1_acq-HCP+1_dir-AP',
        'sub-1_acq-HCP+2_dir-AP',
    ]


# ---------------------------------------------------------------------------
# Test skeleton datasets for the new 4-dict group_dwi_scans interface.
# Each dict is passed to generate_bids_skeleton to create a temporary BIDS
# dataset.  Tests below validate all four outputs of the refactored
# group_dwi_scans: distortion_groups, fmap_estimation_groups,
# fmap_application_groups, and concatenation_groups.
# ---------------------------------------------------------------------------

_SHARED_SHIM = [3767, 2516, 398, 115, 44, -134, -53, 43]

dset_multirun_multiped_no_metadata = {
    '01': [
        {
            'anat': [{'suffix': 'T1w'}],
            'dwi': [
                {
                    'dir': 'AP',
                    'run': '1',
                    'suffix': 'dwi',
                    'metadata': {
                        'PhaseEncodingDirection': 'j-',
                        'ShimSetting': _SHARED_SHIM,
                    },
                },
                {
                    'dir': 'AP',
                    'run': '2',
                    'suffix': 'dwi',
                    'metadata': {
                        'PhaseEncodingDirection': 'j-',
                        'ShimSetting': _SHARED_SHIM,
                    },
                },
                {
                    'dir': 'PA',
                    'run': '1',
                    'suffix': 'dwi',
                    'metadata': {
                        'PhaseEncodingDirection': 'j',
                        'ShimSetting': _SHARED_SHIM,
                    },
                },
                {
                    'dir': 'PA',
                    'run': '2',
                    'suffix': 'dwi',
                    'metadata': {
                        'PhaseEncodingDirection': 'j',
                        'ShimSetting': _SHARED_SHIM,
                    },
                },
                {
                    'dir': 'LR',
                    'run': '1',
                    'suffix': 'dwi',
                    'metadata': {
                        'PhaseEncodingDirection': 'i',
                        'ShimSetting': _SHARED_SHIM,
                    },
                },
                {
                    'dir': 'LR',
                    'run': '2',
                    'suffix': 'dwi',
                    'metadata': {
                        'PhaseEncodingDirection': 'i',
                        'ShimSetting': _SHARED_SHIM,
                    },
                },
                {
                    'dir': 'RL',
                    'run': '1',
                    'suffix': 'dwi',
                    'metadata': {
                        'PhaseEncodingDirection': 'i-',
                        'ShimSetting': _SHARED_SHIM,
                    },
                },
                {
                    'dir': 'RL',
                    'run': '2',
                    'suffix': 'dwi',
                    'metadata': {
                        'PhaseEncodingDirection': 'i-',
                        'ShimSetting': _SHARED_SHIM,
                    },
                },
            ],
        },
    ],
}

dset_multirun_multiped_split_b0field = {
    '01': [
        {
            'anat': [{'suffix': 'T1w'}],
            'dwi': [
                {
                    'dir': 'AP',
                    'run': '1',
                    'suffix': 'dwi',
                    'metadata': {
                        'PhaseEncodingDirection': 'j-',
                        'ShimSetting': _SHARED_SHIM,
                        'B0FieldIdentifier': 'fmap_run1',
                        'B0FieldSource': 'fmap_run1',
                    },
                },
                {
                    'dir': 'AP',
                    'run': '2',
                    'suffix': 'dwi',
                    'metadata': {
                        'PhaseEncodingDirection': 'j-',
                        'ShimSetting': _SHARED_SHIM,
                        'B0FieldIdentifier': 'fmap_run2',
                        'B0FieldSource': 'fmap_run2',
                    },
                },
                {
                    'dir': 'PA',
                    'run': '1',
                    'suffix': 'dwi',
                    'metadata': {
                        'PhaseEncodingDirection': 'j',
                        'ShimSetting': _SHARED_SHIM,
                        'B0FieldIdentifier': 'fmap_run1',
                        'B0FieldSource': 'fmap_run1',
                    },
                },
                {
                    'dir': 'PA',
                    'run': '2',
                    'suffix': 'dwi',
                    'metadata': {
                        'PhaseEncodingDirection': 'j',
                        'ShimSetting': _SHARED_SHIM,
                        'B0FieldIdentifier': 'fmap_run2',
                        'B0FieldSource': 'fmap_run2',
                    },
                },
                {
                    'dir': 'LR',
                    'run': '1',
                    'suffix': 'dwi',
                    'metadata': {
                        'PhaseEncodingDirection': 'i',
                        'ShimSetting': _SHARED_SHIM,
                        'B0FieldIdentifier': 'fmap_run1',
                        'B0FieldSource': 'fmap_run1',
                    },
                },
                {
                    'dir': 'LR',
                    'run': '2',
                    'suffix': 'dwi',
                    'metadata': {
                        'PhaseEncodingDirection': 'i',
                        'ShimSetting': _SHARED_SHIM,
                        'B0FieldIdentifier': 'fmap_run2',
                        'B0FieldSource': 'fmap_run2',
                    },
                },
                {
                    'dir': 'RL',
                    'run': '1',
                    'suffix': 'dwi',
                    'metadata': {
                        'PhaseEncodingDirection': 'i-',
                        'ShimSetting': _SHARED_SHIM,
                        'B0FieldIdentifier': 'fmap_run1',
                        'B0FieldSource': 'fmap_run1',
                    },
                },
                {
                    'dir': 'RL',
                    'run': '2',
                    'suffix': 'dwi',
                    'metadata': {
                        'PhaseEncodingDirection': 'i-',
                        'ShimSetting': _SHARED_SHIM,
                        'B0FieldIdentifier': 'fmap_run2',
                        'B0FieldSource': 'fmap_run2',
                    },
                },
            ],
        },
    ],
}

dset_multirun_multiped_conflicting_concat = {
    '01': [
        {
            'anat': [{'suffix': 'T1w'}],
            'dwi': [
                {
                    'dir': 'AP',
                    'run': '1',
                    'suffix': 'dwi',
                    'metadata': {
                        'PhaseEncodingDirection': 'j-',
                        'ShimSetting': _SHARED_SHIM,
                        'B0FieldIdentifier': 'fmap_all',
                        'B0FieldSource': 'fmap_all',
                        'MultipartID': 'group_run1',
                    },
                },
                {
                    'dir': 'AP',
                    'run': '2',
                    'suffix': 'dwi',
                    'metadata': {
                        'PhaseEncodingDirection': 'j-',
                        'ShimSetting': _SHARED_SHIM,
                        'B0FieldIdentifier': 'fmap_all',
                        'B0FieldSource': 'fmap_all',
                        'MultipartID': 'group_run2',
                    },
                },
                {
                    'dir': 'PA',
                    'run': '1',
                    'suffix': 'dwi',
                    'metadata': {
                        'PhaseEncodingDirection': 'j',
                        'ShimSetting': _SHARED_SHIM,
                        'B0FieldIdentifier': 'fmap_all',
                        'B0FieldSource': 'fmap_all',
                        'MultipartID': 'group_run1',
                    },
                },
                {
                    'dir': 'PA',
                    'run': '2',
                    'suffix': 'dwi',
                    'metadata': {
                        'PhaseEncodingDirection': 'j',
                        'ShimSetting': _SHARED_SHIM,
                        'B0FieldIdentifier': 'fmap_all',
                        'B0FieldSource': 'fmap_all',
                        'MultipartID': 'group_run2',
                    },
                },
                {
                    'dir': 'LR',
                    'run': '1',
                    'suffix': 'dwi',
                    'metadata': {
                        'PhaseEncodingDirection': 'i',
                        'ShimSetting': _SHARED_SHIM,
                        'B0FieldIdentifier': 'fmap_all',
                        'B0FieldSource': 'fmap_all',
                        'MultipartID': 'group_run1',
                    },
                },
                {
                    'dir': 'LR',
                    'run': '2',
                    'suffix': 'dwi',
                    'metadata': {
                        'PhaseEncodingDirection': 'i',
                        'ShimSetting': _SHARED_SHIM,
                        'B0FieldIdentifier': 'fmap_all',
                        'B0FieldSource': 'fmap_all',
                        'MultipartID': 'group_run2',
                    },
                },
                {
                    'dir': 'RL',
                    'run': '1',
                    'suffix': 'dwi',
                    'metadata': {
                        'PhaseEncodingDirection': 'i-',
                        'ShimSetting': _SHARED_SHIM,
                        'B0FieldIdentifier': 'fmap_all',
                        'B0FieldSource': 'fmap_all',
                        'MultipartID': 'group_run1',
                    },
                },
                {
                    'dir': 'RL',
                    'run': '2',
                    'suffix': 'dwi',
                    'metadata': {
                        'PhaseEncodingDirection': 'i-',
                        'ShimSetting': _SHARED_SHIM,
                        'B0FieldIdentifier': 'fmap_all',
                        'B0FieldSource': 'fmap_all',
                        'MultipartID': 'group_run2',
                    },
                },
            ],
        },
    ],
}

dset_with_intendedfor_fmaps = {
    '01': [
        {
            'anat': [{'suffix': 'T1w'}],
            'fmap': [
                {
                    'dir': 'PA',
                    'suffix': 'epi',
                    'metadata': {
                        'IntendedFor': [
                            'dwi/sub-01_dir-AP_run-1_dwi.nii.gz',
                            'dwi/sub-01_dir-AP_run-2_dwi.nii.gz',
                        ],
                        'PhaseEncodingDirection': 'j',
                    },
                },
            ],
            'dwi': [
                {
                    'dir': 'AP',
                    'run': '1',
                    'suffix': 'dwi',
                    'metadata': {
                        'PhaseEncodingDirection': 'j-',
                        'ShimSetting': _SHARED_SHIM,
                    },
                },
                {
                    'dir': 'AP',
                    'run': '2',
                    'suffix': 'dwi',
                    'metadata': {
                        'PhaseEncodingDirection': 'j-',
                        'ShimSetting': _SHARED_SHIM,
                    },
                },
            ],
        },
    ],
}

dset_with_b0field_fmaps = {
    '01': [
        {
            'anat': [{'suffix': 'T1w'}],
            'fmap': [
                {
                    'dir': 'PA',
                    'suffix': 'epi',
                    'metadata': {
                        'B0FieldIdentifier': 'pepolar01',
                        'PhaseEncodingDirection': 'j',
                    },
                },
            ],
            'dwi': [
                {
                    'dir': 'AP',
                    'run': '1',
                    'suffix': 'dwi',
                    'metadata': {
                        'PhaseEncodingDirection': 'j-',
                        'ShimSetting': _SHARED_SHIM,
                        'B0FieldIdentifier': 'pepolar01',
                        'B0FieldSource': 'pepolar01',
                    },
                },
                {
                    'dir': 'AP',
                    'run': '2',
                    'suffix': 'dwi',
                    'metadata': {
                        'PhaseEncodingDirection': 'j-',
                        'ShimSetting': _SHARED_SHIM,
                        'B0FieldIdentifier': 'pepolar01',
                        'B0FieldSource': 'pepolar01',
                    },
                },
            ],
        },
    ],
}

dset_multi_session = {
    '01': [
        {
            'session': '01',
            'anat': [{'suffix': 'T1w'}],
            'dwi': [
                {
                    'dir': 'AP',
                    'suffix': 'dwi',
                    'metadata': {
                        'PhaseEncodingDirection': 'j-',
                        'ShimSetting': _SHARED_SHIM,
                    },
                },
                {
                    'dir': 'PA',
                    'suffix': 'dwi',
                    'metadata': {
                        'PhaseEncodingDirection': 'j',
                        'ShimSetting': _SHARED_SHIM,
                    },
                },
            ],
        },
        {
            'session': '02',
            'dwi': [
                {
                    'dir': 'AP',
                    'suffix': 'dwi',
                    'metadata': {
                        'PhaseEncodingDirection': 'j-',
                        'ShimSetting': _SHARED_SHIM,
                    },
                },
                {
                    'dir': 'PA',
                    'suffix': 'dwi',
                    'metadata': {
                        'PhaseEncodingDirection': 'j',
                        'ShimSetting': _SHARED_SHIM,
                    },
                },
            ],
        },
    ],
}

dset_missing_ped = {
    '01': [
        {
            'anat': [{'suffix': 'T1w'}],
            'dwi': [
                {
                    'dir': 'AP',
                    'suffix': 'dwi',
                    'metadata': {
                        'PhaseEncodingDirection': 'j-',
                        'ShimSetting': _SHARED_SHIM,
                    },
                },
                {
                    'run': '1',
                    'suffix': 'dwi',
                    'metadata': {
                        'ShimSetting': _SHARED_SHIM,
                    },
                },
            ],
        },
    ],
}

dset_single_dwi = {
    '01': [
        {
            'anat': [{'suffix': 'T1w'}],
            'dwi': [
                {
                    'dir': 'AP',
                    'suffix': 'dwi',
                    'metadata': {
                        'PhaseEncodingDirection': 'j-',
                        'ShimSetting': _SHARED_SHIM,
                    },
                },
            ],
        },
    ],
}

dset_b0field_cross_axis = {
    '01': [
        {
            'anat': [{'suffix': 'T1w'}],
            'dwi': [
                {
                    'dir': 'AP',
                    'suffix': 'dwi',
                    'metadata': {
                        'PhaseEncodingDirection': 'j-',
                        'ShimSetting': _SHARED_SHIM,
                        'B0FieldIdentifier': 'cross_axis',
                        'B0FieldSource': 'cross_axis',
                    },
                },
                {
                    'dir': 'LR',
                    'suffix': 'dwi',
                    'metadata': {
                        'PhaseEncodingDirection': 'i',
                        'ShimSetting': _SHARED_SHIM,
                        'B0FieldIdentifier': 'cross_axis',
                        'B0FieldSource': 'cross_axis',
                    },
                },
            ],
        },
    ],
}

dset_separate_all_with_multipartid = {
    '01': [
        {
            'anat': [{'suffix': 'T1w'}],
            'dwi': [
                {
                    'dir': 'AP',
                    'run': '1',
                    'suffix': 'dwi',
                    'metadata': {
                        'PhaseEncodingDirection': 'j-',
                        'ShimSetting': _SHARED_SHIM,
                        'MultipartID': 'group_a',
                    },
                },
                {
                    'dir': 'AP',
                    'run': '2',
                    'suffix': 'dwi',
                    'metadata': {
                        'PhaseEncodingDirection': 'j-',
                        'ShimSetting': _SHARED_SHIM,
                        'MultipartID': 'group_a',
                    },
                },
                {
                    'dir': 'PA',
                    'suffix': 'dwi',
                    'metadata': {
                        'PhaseEncodingDirection': 'j',
                        'ShimSetting': _SHARED_SHIM,
                    },
                },
            ],
        },
    ],
}


def _make_layout(tmpdir, dset_dict, name):
    """Helper: generate a BIDS skeleton and return (layout, subject_data)."""
    bids_dir = tmpdir / name
    generate_bids_skeleton(str(bids_dir), dset_dict)
    layout = BIDSLayout(str(bids_dir))
    subject_data = {'dwi': layout.get(suffix='dwi', extension='nii.gz', return_type='file')}
    return layout, subject_data


def _basenames(paths):
    """Return sorted basenames from a list of paths."""
    return sorted(os.path.basename(p) for p in paths)


# ---------------------------------------------------------------------------
# Tests targeting the new 4-dict return signature of group_dwi_scans.
# ---------------------------------------------------------------------------


class TestScenario1ADefault:
    """Scenario 1A: multi-run multi-PED, no curator metadata, default flags.

    combine_scans=True, ignore_fieldmaps=False, estimate_per_axis=False.
    Expected: 4 distortion groups, 1 fmap estimation group, 1 concatenation group.
    """

    def test_distortion_groups(self, tmpdir):
        layout, subject_data = _make_layout(
            tmpdir, dset_multirun_multiped_no_metadata, 'scenario_1a',
        )
        dg, _, _, _ = grouping.group_dwi_scans(
            layout=layout,
            subject_data=subject_data,
            combine_scans=True,
            ignore_fieldmaps=False,
            estimate_per_axis=False,
        )
        assert len(dg) == 4
        expected_keys = {'sub-01_dir-AP', 'sub-01_dir-LR', 'sub-01_dir-PA', 'sub-01_dir-RL'}
        assert set(dg.keys()) == expected_keys
        for key, files in dg.items():
            assert len(files) == 2, f'{key} should have 2 files'

    def test_fmap_estimation_groups(self, tmpdir):
        layout, subject_data = _make_layout(
            tmpdir, dset_multirun_multiped_no_metadata, 'scenario_1a_fme',
        )
        _, fme, _, _ = grouping.group_dwi_scans(
            layout=layout,
            subject_data=subject_data,
            combine_scans=True,
            ignore_fieldmaps=False,
            estimate_per_axis=False,
        )
        assert len(fme) == 1
        group_members = list(fme.values())[0]
        assert set(group_members) == {
            'sub-01_dir-AP',
            'sub-01_dir-PA',
            'sub-01_dir-LR',
            'sub-01_dir-RL',
        }

    def test_concatenation_groups(self, tmpdir):
        layout, subject_data = _make_layout(
            tmpdir, dset_multirun_multiped_no_metadata, 'scenario_1a_cg',
        )
        _, _, _, cg = grouping.group_dwi_scans(
            layout=layout,
            subject_data=subject_data,
            combine_scans=True,
            ignore_fieldmaps=False,
            estimate_per_axis=False,
        )
        assert len(cg) == 1
        group_members = list(cg.values())[0]
        assert set(group_members) == {
            'sub-01_dir-AP',
            'sub-01_dir-PA',
            'sub-01_dir-LR',
            'sub-01_dir-RL',
        }


class TestScenario1AEstimatePerAxis:
    """Scenario 1A with estimate_per_axis=True.

    Expected: 4 distortion groups, 2 fmap estimation groups (AP/PA and LR/RL).
    """

    def test_fmap_estimation_groups(self, tmpdir):
        layout, subject_data = _make_layout(
            tmpdir, dset_multirun_multiped_no_metadata, 'scenario_1a_epa',
        )
        _, fme, _, _ = grouping.group_dwi_scans(
            layout=layout,
            subject_data=subject_data,
            combine_scans=True,
            ignore_fieldmaps=False,
            estimate_per_axis=True,
        )
        assert len(fme) == 2
        member_sets = [set(v) for v in fme.values()]
        assert {'sub-01_dir-AP', 'sub-01_dir-PA'} in member_sets
        assert {'sub-01_dir-LR', 'sub-01_dir-RL'} in member_sets


class TestScenario1ASeparateAll:
    """Scenario 1A with combine_scans=False (--separate-all-dwis).

    Expected: 8 distortion groups (one per file), each its own concatenation group.
    """

    def test_distortion_groups(self, tmpdir):
        layout, subject_data = _make_layout(
            tmpdir, dset_multirun_multiped_no_metadata, 'scenario_1a_sep',
        )
        dg, _, _, _ = grouping.group_dwi_scans(
            layout=layout,
            subject_data=subject_data,
            combine_scans=False,
            ignore_fieldmaps=False,
            estimate_per_axis=False,
        )
        assert len(dg) == 8
        for files in dg.values():
            assert len(files) == 1

    def test_concatenation_groups(self, tmpdir):
        layout, subject_data = _make_layout(
            tmpdir, dset_multirun_multiped_no_metadata, 'scenario_1a_sep_cg',
        )
        dg, _, _, cg = grouping.group_dwi_scans(
            layout=layout,
            subject_data=subject_data,
            combine_scans=False,
            ignore_fieldmaps=False,
            estimate_per_axis=False,
        )
        assert len(cg) == 8
        for members in cg.values():
            assert len(members) == 1


class TestScenario1AIgnoreFieldmaps:
    """Scenario 1A with ignore_fieldmaps=True.

    Fmap estimation groups should only use DWI-based heuristics (no fmap/ files),
    but since there are no fmap files in this dataset the result is the same as
    the default case.
    """

    def test_no_fmap_files_in_estimation(self, tmpdir):
        layout, subject_data = _make_layout(
            tmpdir, dset_multirun_multiped_no_metadata, 'scenario_1a_ign',
        )
        _, fme, _, _ = grouping.group_dwi_scans(
            layout=layout,
            subject_data=subject_data,
            combine_scans=True,
            ignore_fieldmaps=True,
            estimate_per_axis=False,
        )
        all_members = []
        for members in fme.values():
            all_members.extend(members)
        for member in all_members:
            assert 'fmap' not in member


class TestScenario1BB0FieldSplitsRuns:
    """Scenario 1B: B0FieldIdentifier splits run-1 and run-2 into separate fmap groups.

    Expected: 8 distortion groups (because the fmap split forces distortion group
    refinement), 2 fmap estimation groups keyed by B0FieldIdentifier, 1 concatenation
    group.
    """

    def test_distortion_groups(self, tmpdir):
        layout, subject_data = _make_layout(
            tmpdir, dset_multirun_multiped_split_b0field, 'scenario_1b',
        )
        dg, _, _, _ = grouping.group_dwi_scans(
            layout=layout,
            subject_data=subject_data,
            combine_scans=True,
            ignore_fieldmaps=False,
            estimate_per_axis=False,
        )
        assert len(dg) == 8
        for files in dg.values():
            assert len(files) == 1

    def test_fmap_estimation_groups(self, tmpdir):
        layout, subject_data = _make_layout(
            tmpdir, dset_multirun_multiped_split_b0field, 'scenario_1b_fme',
        )
        _, fme, _, _ = grouping.group_dwi_scans(
            layout=layout,
            subject_data=subject_data,
            combine_scans=True,
            ignore_fieldmaps=False,
            estimate_per_axis=False,
        )
        assert 'fmap_run1' in fme
        assert 'fmap_run2' in fme
        assert len(fme['fmap_run1']) == 4
        assert len(fme['fmap_run2']) == 4

    def test_one_concatenation_group(self, tmpdir):
        layout, subject_data = _make_layout(
            tmpdir, dset_multirun_multiped_split_b0field, 'scenario_1b_cg',
        )
        _, _, _, cg = grouping.group_dwi_scans(
            layout=layout,
            subject_data=subject_data,
            combine_scans=True,
            ignore_fieldmaps=False,
            estimate_per_axis=False,
        )
        assert len(cg) == 1
        members = list(cg.values())[0]
        assert len(members) == 8


class TestScenario1CConflict:
    """Scenario 1C: One fmap estimation group spans both MultipartID concatenation groups.

    This should raise an error because fmap estimation groups are not subsets of
    concatenation groups.
    """

    def test_raises(self, tmpdir):
        layout, subject_data = _make_layout(
            tmpdir, dset_multirun_multiped_conflicting_concat, 'scenario_1c',
        )
        with pytest.raises(ValueError, match='subset'):
            grouping.group_dwi_scans(
                layout=layout,
                subject_data=subject_data,
                combine_scans=True,
                ignore_fieldmaps=False,
                estimate_per_axis=False,
            )


class TestIntendedForFmaps:
    """fmap/ EPI linked to DWI via IntendedFor.

    The fmap estimation group should contain both the fmap file path and the
    DWI distortion group IDs.  The fmap application group should contain only
    the DWI distortion group IDs (the fmap is a source, not a target).
    """

    def test_fmap_in_estimation(self, tmpdir):
        layout, subject_data = _make_layout(
            tmpdir, dset_with_intendedfor_fmaps, 'intendedfor',
        )
        _, fme, _, _ = grouping.group_dwi_scans(
            layout=layout,
            subject_data=subject_data,
            combine_scans=True,
            ignore_fieldmaps=False,
            estimate_per_axis=False,
        )
        assert len(fme) == 1
        members = list(fme.values())[0]
        fmap_members = [m for m in members if 'epi' in m]
        dwi_members = [m for m in members if 'epi' not in m]
        assert len(fmap_members) >= 1
        assert len(dwi_members) >= 1

    def test_fmap_not_in_application(self, tmpdir):
        layout, subject_data = _make_layout(
            tmpdir, dset_with_intendedfor_fmaps, 'intendedfor_app',
        )
        _, _, fma, _ = grouping.group_dwi_scans(
            layout=layout,
            subject_data=subject_data,
            combine_scans=True,
            ignore_fieldmaps=False,
            estimate_per_axis=False,
        )
        for targets in fma.values():
            for target in targets:
                assert 'epi' not in target


class TestB0FieldFmaps:
    """fmap/ EPI linked via B0FieldIdentifier + B0FieldSource.

    Keys of fmap estimation groups should be the B0FieldIdentifier string.
    """

    def test_b0field_keys(self, tmpdir):
        layout, subject_data = _make_layout(
            tmpdir, dset_with_b0field_fmaps, 'b0field_fmaps',
        )
        _, fme, _, _ = grouping.group_dwi_scans(
            layout=layout,
            subject_data=subject_data,
            combine_scans=True,
            ignore_fieldmaps=False,
            estimate_per_axis=False,
        )
        assert 'pepolar01' in fme
        members = fme['pepolar01']
        fmap_members = [m for m in members if 'epi' in m]
        dwi_members = [m for m in members if 'epi' not in m]
        assert len(fmap_members) >= 1
        assert len(dwi_members) >= 1


class TestIgnoreFmapFiles:
    """ignore_fieldmaps=True should exclude fmap/ files from estimation groups.

    Even though IntendedFor exists, fmap files should not appear.
    """

    def test_fmap_excluded(self, tmpdir):
        layout, subject_data = _make_layout(
            tmpdir, dset_with_intendedfor_fmaps, 'ignore_fmaps',
        )
        _, fme, _, _ = grouping.group_dwi_scans(
            layout=layout,
            subject_data=subject_data,
            combine_scans=True,
            ignore_fieldmaps=True,
            estimate_per_axis=False,
        )
        all_members = []
        for members in fme.values():
            all_members.extend(members)
        for member in all_members:
            assert 'fmap' not in member


class TestMultiSession:
    """Two sessions should produce independent groups that do not cross boundaries."""

    def test_distortion_groups_per_session(self, tmpdir):
        layout, subject_data = _make_layout(
            tmpdir, dset_multi_session, 'multi_session',
        )
        dg, _, _, _ = grouping.group_dwi_scans(
            layout=layout,
            subject_data=subject_data,
            combine_scans=True,
            ignore_fieldmaps=False,
            estimate_per_axis=False,
        )
        assert len(dg) == 4
        for key in dg:
            assert 'ses-01' in key or 'ses-02' in key

    def test_concatenation_groups_per_session(self, tmpdir):
        layout, subject_data = _make_layout(
            tmpdir, dset_multi_session, 'multi_session_cg',
        )
        _, _, _, cg = grouping.group_dwi_scans(
            layout=layout,
            subject_data=subject_data,
            combine_scans=True,
            ignore_fieldmaps=False,
            estimate_per_axis=False,
        )
        assert len(cg) == 2


class TestMissingPEDMetadata:
    """Files without PhaseEncodingDirection should become singleton distortion groups."""

    def test_singleton_for_missing_ped(self, tmpdir):
        layout, subject_data = _make_layout(
            tmpdir, dset_missing_ped, 'missing_ped',
        )
        dg, _, _, _ = grouping.group_dwi_scans(
            layout=layout,
            subject_data=subject_data,
            combine_scans=True,
            ignore_fieldmaps=False,
            estimate_per_axis=False,
        )
        assert len(dg) == 2
        singleton_groups = [files for files in dg.values() if len(files) == 1]
        assert len(singleton_groups) >= 1


class TestSingleDWI:
    """Single DWI file: 1 distortion group, no fmap estimation, 1 concatenation group."""

    def test_single_file(self, tmpdir):
        layout, subject_data = _make_layout(
            tmpdir, dset_single_dwi, 'single_dwi',
        )
        dg, fme, fma, cg = grouping.group_dwi_scans(
            layout=layout,
            subject_data=subject_data,
            combine_scans=True,
            ignore_fieldmaps=False,
            estimate_per_axis=False,
        )
        assert len(dg) == 1
        assert len(fme) == 0
        assert len(fma) == 0
        assert len(cg) == 1


class TestSeparateAllOverridesMultipartID:
    """combine_scans=False should override MultipartID and produce separate groups.

    A warning should be raised.
    """

    def test_separate_overrides(self, tmpdir):
        layout, subject_data = _make_layout(
            tmpdir, dset_separate_all_with_multipartid, 'sep_multipart',
        )
        with pytest.warns(UserWarning):
            dg, _, _, cg = grouping.group_dwi_scans(
                layout=layout,
                subject_data=subject_data,
                combine_scans=False,
                ignore_fieldmaps=False,
                estimate_per_axis=False,
            )
        assert len(dg) == 3
        for files in dg.values():
            assert len(files) == 1
        assert len(cg) == 3


class TestEstimatePerAxisConflictsB0Field:
    """B0FieldIdentifier groups AP and LR together but estimate_per_axis=True.

    This should raise an error because the B0 field spans multiple PE axes.
    """

    def test_raises(self, tmpdir):
        layout, subject_data = _make_layout(
            tmpdir, dset_b0field_cross_axis, 'epa_conflict',
        )
        with pytest.raises(ValueError, match='axis'):
            grouping.group_dwi_scans(
                layout=layout,
                subject_data=subject_data,
                combine_scans=True,
                ignore_fieldmaps=False,
                estimate_per_axis=True,
            )
