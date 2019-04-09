"""
Test that scans are grouped by fieldmap correctly.
"""
import pytest
from get_data import bids_data, BIDS_DIR


test_grouping_conditions = [
    # Combine all dwis, use fielmaps, no SyN, no dedicated fmape (results in BUDS)
    (True, [], False, False, ('sub-tester_buds-j',)),
    # Don't combine images, but do use fieldmaps
    (False, [], False, False, ('sub-tester_acq-HASC55AP', 'sub-tester_acq-HASC55PA')),
    # requests that fieldmaps are ignored but scans are combined. Should print error.
    # For now runs them separately.
    (True, ['fieldmaps'], False, True,
     ('sub-tester_acq-HASC55AP', 'sub-tester_acq-HASC55PA')),
    (True, [], True, True,
     ('sub-tester_acq-HASC55AP', 'sub-tester_acq-HASC55PA'))
]


@pytest.mark.parametrize(
    "combine_all_dwis,ignore,force_syn,prefer_dedicated_fmaps,output_groups",
    test_grouping_conditions)
def test_grouping_options(combine_all_dwis, ignore, force_syn, prefer_dedicated_fmaps,
                          output_groups, bids_data):
    from qsiprep.utils.bids import collect_data
    from qsiprep.workflows.base import group_by_fieldmaps, get_session_groups, _get_output_fname
    subject_data, layout = collect_data(BIDS_DIR, "tester")

    # Get session groups
    dwi_session_groups = get_session_groups(layout, subject_data, combine_all_dwis)
#    assert len(dwi_session_groups) == 1

    # Now group by fieldmaps for session x fieldmap
    dwi_fmap_groups = []
    for dwi_session_group in dwi_session_groups:
        dwi_fmap_groups.extend(
            group_by_fieldmaps(dwi_session_group, layout,
                               "fieldmaps" in ignore or force_syn,
                               prefer_dedicated_fmaps,
                               combine_all_dwis))
    outputs_to_files = dict([
        (_get_output_fname(dwi_group), dwi_group) for dwi_group in dwi_fmap_groups
    ])
    output_keys = tuple(sorted(outputs_to_files.keys()))
    assert output_keys == output_groups
