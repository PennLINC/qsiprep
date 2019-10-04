"""
Test that scans are grouped by fieldmap correctly.
"""
import pytest
from get_data import bids_data, BIDS_DIR


test_grouping_conditions = [
    # combine_all_dwis, ignore, use_syn, prefer_dedicated_fmaps, using_eddy,
    (True, [], False, False, True, ('sub-tester_acq-HASC55',), ('rpe_series',)),
    (True, [], False, False, False, ('sub-tester_acq-HASC55',), ('rpe_series',)),

    # Don't combine images, but do use fieldmaps
    (False, [], False, False, True, ('sub-tester_acq-HASC55AP', 'sub-tester_acq-HASC55PA'),
        ('epi', 'epi')),
    (False, [], False, False, False, ('sub-tester_acq-HASC55AP', 'sub-tester_acq-HASC55PA'),
        ('epi', 'epi')),
    # requests that fieldmaps are ignored but scans are combined. Should print error.
    # For now runs them separately.
    (True, ['fieldmaps'], False, True, True,
     ('sub-tester_acq-HASC55AP', 'sub-tester_acq-HASC55PA'), (None, None)),
    (True, [], True, True, True,
     ('sub-tester_acq-HASC55AP', 'sub-tester_acq-HASC55PA'), ('epi', 'epi')),
    (True, ['fieldmaps'], True, True, True,
     ('sub-tester_acq-HASC55AP', 'sub-tester_acq-HASC55PA'), ('syn', 'syn')),
]


@pytest.mark.parametrize(
    "combine_all_dwis,ignore,use_syn,prefer_dedicated_fmaps,output_groups",
    test_grouping_conditions)
def test_grouping_options(
    combine_all_dwis, ignore, use_syn, prefer_dedicated_fmaps,using_eddy, output_groups, fieldmap_types, bids_data):
    from qsiprep.utils.bids import collect_data
    from qsiprep.workflows.base import group_by_warpspace, get_session_groups, _get_output_fname
    subject_data, layout = collect_data(BIDS_DIR, "tester")

    # Get session groups
    dwi_session_groups = get_session_groups(layout, subject_data, combine_all_dwis)
    dwi_fmap_groups = []
    for dwi_session_group in dwi_session_groups:
        dwi_fmap_groups.extend(
            group_by_warpspace(dwi_session_group,
                               layout,
                               prefer_dedicated_fmaps,
                               using_eddy,
                               "fieldmaps" in ignore,
                               combine_all_dwis,
                               use_syn))
    outputs_to_files = dict([
        (_get_output_fname(dwi_group), dwi_group) for dwi_group in dwi_fmap_groups
    ])
    output_keys = tuple(sorted(outputs_to_files.keys()))
    assert output_keys == output_groups
    for key, fmap_type in zip(output_groups, fieldmap_types):
        assert outputs_to_files[key]['fieldmap_info']['type'] == fmap_type
