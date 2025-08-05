"""Tests for the grouping utils."""

import os
from pprint import pformat

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
    assert layout.get_file(fieldmaps[0]['epi']).get_metadata()['IntendedFor'] == [
        'dwi/sub-01_dir-AP_dwi.nii.gz'
    ]

    # Check that BIDS URI paths are correctly handled
    # XXX: This currently fails because pybids does not support BIDS URI paths.
    # bids_dir = base_dir / 'dset_fmap_intendedfor_bidsuri'
    # generate_bids_skeleton(str(bids_dir), dset_fmap_intendedfor_bidsuri)
    # layout = BIDSLayout(str(bids_dir))
    # dwi_file = layout.get(suffix='dwi', extension='nii.gz', return_type='filename')[0]
    # fieldmaps = layout.get_fieldmap(dwi_file, return_list=True)
    # assert len(fieldmaps) == 1
    # assert fieldmaps[0]['suffix'] == 'epi'
    # assert layout.get_file(fieldmaps[0]['epi']).get_metadata()['IntendedFor'] == [
    #     'bids::sub-01/dwi/sub-01_dir-AP_dwi.nii.gz'
    # ]

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
        assert len(subject_data) == len(expected), pformat(subject_data)
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
    subject_data = {'dwi': layout.get(suffix='dwi', extension='nii.gz')}
    scan_groups, _ = grouping.group_dwi_scans(
        layout=layout,
        subject_data=subject_data,
        using_fsl=True,
        combine_scans=True,
        ignore_fieldmaps=False,
        concatenate_distortion_groups=True,
    )
    expected = [
        {
            'concatenated_bids_name': 'sub-01_dir-AP',
            'dwi_series': [
                'sub-01/dwi/sub-01_dir-AP_run-1_dwi.nii.gz',
                'sub-01/dwi/sub-01_dir-AP_run-2_dwi.nii.gz',
            ],
            'dwi_series_pedir': 'j',
            'fieldmap_info': {'suffix': None},
        },
        {
            'concatenated_bids_name': 'sub-01_dir-PA',
            'dwi_series': ['sub-01/dwi/sub-01_dir-PA_dwi.nii.gz'],
            'dwi_series_pedir': 'j-',
            'fieldmap_info': {'suffix': None},
        },
    ]
    check_expected(scan_groups, expected)

    scan_groups, _ = grouping.group_dwi_scans(
        layout=layout,
        subject_data=subject_data,
        using_fsl=False,
        combine_scans=True,
        ignore_fieldmaps=False,
        concatenate_distortion_groups=False,
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
        using_fsl=True,
        combine_scans=False,
        ignore_fieldmaps=False,
        concatenate_distortion_groups=False,
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
        using_fsl=False,
        combine_scans=False,
        ignore_fieldmaps=False,
        concatenate_distortion_groups=False,
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
        using_fsl=True,
        combine_scans=True,
        ignore_fieldmaps=True,
        concatenate_distortion_groups=False,
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
        using_fsl=True,
        combine_scans=True,
        ignore_fieldmaps=False,
        concatenate_distortion_groups=True,
    )
    expected = [
        [
            'sub-01_dir-AP_run-1_dwi.nii.gz',
            'sub-01_dir-AP_run-2_dwi.nii.gz',
            'sub-01_dir-PA_dwi.nii.gz',
        ],
    ]
    check_expected(scan_groups, expected)


def test_group_dwi_scans_with_complex_relpaths(tmpdir):
    """Test the group_dwi_scans function.

    This is the same as test_group_dwi_scans_with_complex_b0fields,
    but with IntendedFors using relative paths instead of B0Field* fields.
    """
    bids_dir = tmpdir / 'test_group_dwi_scans_with_complex_relpaths'
    dset_yaml = os.path.join(get_test_data_path(), 'skeleton_complex_relpaths.yml')
    generate_bids_skeleton(str(bids_dir), dset_yaml)
    layout = BIDSLayout(str(bids_dir))
    subject_data = {'dwi': layout.get(suffix='dwi', extension='nii.gz')}
    scan_groups, _ = grouping.group_dwi_scans(
        layout=layout,
        subject_data=subject_data,
        using_fsl=True,
        combine_scans=True,
        ignore_fieldmaps=False,
        concatenate_distortion_groups=True,
    )
    expected = [
        {
            'concatenated_bids_name': 'sub-01_dir-AP',
            'dwi_series': [
                'sub-01/dwi/sub-01_dir-AP_run-1_dwi.nii.gz',
                'sub-01/dwi/sub-01_dir-AP_run-2_dwi.nii.gz',
            ],
            'dwi_series_pedir': 'j',
            'fieldmap_info': {'suffix': None},
        },
        {
            'concatenated_bids_name': 'sub-01_dir-PA',
            'dwi_series': ['sub-01/dwi/sub-01_dir-PA_dwi.nii.gz'],
            'dwi_series_pedir': 'j-',
            'fieldmap_info': {'suffix': None},
        },
    ]
    check_expected(scan_groups, expected)

    scan_groups, _ = grouping.group_dwi_scans(
        layout=layout,
        subject_data=subject_data,
        using_fsl=False,
        combine_scans=True,
        ignore_fieldmaps=False,
        concatenate_distortion_groups=False,
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
        using_fsl=True,
        combine_scans=False,
        ignore_fieldmaps=False,
        concatenate_distortion_groups=False,
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
        using_fsl=False,
        combine_scans=False,
        ignore_fieldmaps=False,
        concatenate_distortion_groups=False,
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
        using_fsl=True,
        combine_scans=True,
        ignore_fieldmaps=True,
        concatenate_distortion_groups=False,
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
        using_fsl=True,
        combine_scans=True,
        ignore_fieldmaps=False,
        concatenate_distortion_groups=True,
    )
    expected = [
        [
            'sub-01_dir-AP_run-1_dwi.nii.gz',
            'sub-01_dir-AP_run-2_dwi.nii.gz',
            'sub-01_dir-PA_dwi.nii.gz',
        ],
    ]
    check_expected(scan_groups, expected)
