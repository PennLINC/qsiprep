# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Utilities to group scans based on their acquisition parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Download many variations of fieldmaps and dwi data

Examples:
---------

    Setup tests
    >>> import os
    >>> from qsiprep.utils.testing import get_grouping_test_data
    >>> data_root = get_grouping_test_data()
    >>> os.chdir(data_root)
"""
from collections import defaultdict
import logging
from nipype.utils.filemanip import split_filename
from ..interfaces.bids import get_bids_params
LOGGER = logging.getLogger('nipype.workflow')


def group_dwi_scans(bids_layout, subject_data, using_fsl=False, combine_scans=True,
                    ignore_fieldmaps=False, concatenate_distortion_groups=False):
    """Determine which scans can be concatenated based on their acquisition parameters.

    **Parameters**
        bids_layout : layout
            A PyBIDS layout
        group_for_eddy : bool
            Should a plus and minus series be grouped together for TOPUP/eddy?
        combine_scans : bool
            Should scan concatention happen?
        concatenate_distortion_groups : bool
            Will distortion groups get merged at the end of the pipeline?

    **Returns**
        scan_groups : list of dictionaries
            A dict where the keys are the BIDS derivatives name of the output file after
            concatenation. The values are lists of dwi files in that group
    """

    # Handle the grouping of multiple dwi files within a session
    dwi_session_groups = get_session_groups(bids_layout, subject_data, combine_scans)

    # Group them by their warp group
    dwi_fmap_groups = []
    for dwi_session_group in dwi_session_groups:
        dwi_fmap_groups.extend(
            group_by_warpspace(dwi_session_group, bids_layout, ignore_fieldmaps))

    if using_fsl:
        return group_for_eddy(dwi_fmap_groups)

    if concatenate_distortion_groups:
        return dwi_fmap_groups, group_for_concatenation(dwi_fmap_groups)

    return dwi_fmap_groups, {}


def get_session_groups(layout, subject_data, combine_all_dwis):
    # Handle the grouping of multiple dwi files within a session
    sessions = layout.get_sessions() if layout is not None else []
    all_dwis = subject_data['dwi']
    dwi_session_groups = []
    if not combine_all_dwis:
        dwi_session_groups = [[dwi] for dwi in all_dwis]
    else:
        if sessions:
            LOGGER.info('Combining all dwi files within each available session:')
            for session in sessions:
                session_files = [img for img in all_dwis if 'ses-'+session in img]
                LOGGER.info('\t- %d scans in session %s', len(session_files), session)
                dwi_session_groups.append(session_files)
        else:
            LOGGER.info('Combining all %d dwis within the single available session',
                        len(all_dwis))
            dwi_session_groups = [all_dwis]
    return dwi_session_groups


FMAP_PRIORITY = {
    'epi': 0,
    'dwi': 1,
    'fieldmap': 2,
    'phasediff': 3,
    'phase1': 4,
    'phase': 4,
    'syn': 5
}


def get_highest_priority_fieldmap(fmap_infos):
    """Return a dictionary describing the highest priority fieldmap.

    **Examples**

    Invent some potential fieldmaps
    >>> epi_fmap1 = {"epi": "/data/sub-1/fmap/sub-1_dir-AP_run-1_epi.nii.gz", "suffix": "epi"}
    >>> epi_fmap2 = {"epi": "/data/sub-1/fmap/sub-1_dir-AP_run-2_epi.nii.gz", "suffix": "epi"}
    >>> epi_fmap3 = {"epi": "/data/sub-1/fmap/sub-1_dir-PA_epi.nii.gz", "suffix": "epi"}
    >>>
    >>> phasediff_fmap = {"phasediff": "/data/sub-1/fmap/sub-1_phasediff.nii.gz",
    ...                   "suffix": "phasediff"}
    >>> phases_fmap = {"phase1": "/data/sub-1/fmap/sub-1_phase1.nii.gz",
    ...                "suffix": "phase1"}
    >>>
    >>> dwi_fmap1 = {"dwi": "/data/sub-1/dwi/sub-1_dir-AP_dwi.nii.gz", "suffix": "dwi"}
    >>> dwi_fmap2 = {'suffix': 'dwi',
    ...   'dwi': ['/data/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
    ...    '/data/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz',
    ...    '/data/sub-1/dwi/sub-1_dir-AP_run-3_dwi.nii.gz']}

    When there are no fieldmaps in ``fmaps/``, but a reverse PE DWI series
    >>> get_highest_priority_fieldmap([dwi_fmap1])
    {'dwi': '/data/sub-1/dwi/sub-1_dir-AP_dwi.nii.gz', 'suffix': 'dwi'}

    There is both an epi fieldmap and a phase1/phase2 GRE fieldmap
    >>> get_highest_priority_fieldmap([epi_fmap1, phases_fmap])
    {'suffix': 'epi', 'epi': ['/data/sub-1/fmap/sub-1_dir-AP_run-1_epi.nii.gz']}

    Multiple EPI fieldmaps
    >>> get_highest_priority_fieldmap(
    ...     [epi_fmap1, epi_fmap2, epi_fmap3]) # doctest: +NORMALIZE_WHITESPACE
    {'suffix': 'epi',
     'epi': ['/data/sub-1/fmap/sub-1_dir-AP_run-1_epi.nii.gz',
             '/data/sub-1/fmap/sub-1_dir-AP_run-2_epi.nii.gz',
             '/data/sub-1/fmap/sub-1_dir-PA_epi.nii.gz']}

    An EPI fieldmap from ``fmap/`` should be chosen over a reverse PE DWI series
    >>> get_highest_priority_fieldmap([epi_fmap1, dwi_fmap2])
    {'suffix': 'epi', 'epi': ['/data/sub-1/fmap/sub-1_dir-AP_run-1_epi.nii.gz']}

    """
    # Find fieldmaps
    default_priority = max(FMAP_PRIORITY.values()) + 1
    priority = default_priority
    selected_fmap_info = {'suffix': None}

    # collapse multiple EPI fieldmaps into one entry
    epi_fmaps = sorted([fmap_info["epi"] for fmap_info in fmap_infos
                        if fmap_info.get('suffix') == 'epi'])
    if epi_fmaps:
        epi_info = {"suffix": "epi", "epi": epi_fmaps}
        fmap_infos = [fmap_info for fmap_info in fmap_infos
                      if fmap_info.get('suffix') != 'epi'] + [epi_info]

    # Select the highest priority fieldmap
    for fmap_info in fmap_infos:
        if fmap_info.get('suffix') == 'phase':
            fmap_info['suffix'] = 'phase1'
        fmap_type = fmap_info.get('suffix')
        if fmap_type not in FMAP_PRIORITY:
            continue
        this_priority = FMAP_PRIORITY[fmap_type]
        if this_priority < priority:
            priority = this_priority
            selected_fmap_info = fmap_info

    return selected_fmap_info


def find_fieldmaps_from_other_dwis(dwi_files, dwi_file_metadatas):
    """Find a list of files in the dwi/ directory that can be used for distortion correction.

    It is common to acquire DWI scans with opposite phase encoding directions so they can be
    used to correct each other's EPI distortion. There is currently no mechanism in BIDS to
    specify whether b=0 scans in dwi/ can be used as fieldmaps for one another.

    **Examples**

    A single scan with no opportunities to SDC with a DWI scan
    >>> from qsiprep.utils.grouping import find_fieldmaps_from_other_dwis
    >>> single_dwi_file = ["/data/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz"]
    >>> single_dwi_file_metadatas = [{"PhaseEncodingDirection": "j"}]
    >>> find_fieldmaps_from_other_dwis(single_dwi_file, single_dwi_file_metadatas)
    {'/data/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz': {}}

    Two scans with the same PE direction: again no opportunities to SDC
    >>> repeat_dwi_files = ["/data/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz",
    ...                     "/data/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz"]
    >>> repeat_dwi_file_metadatas = [{"PhaseEncodingDirection": "j"},
    ...                              {"PhaseEncodingDirection": "j"}]
    >>> find_fieldmaps_from_other_dwis(repeat_dwi_files,
    ...     repeat_dwi_file_metadatas) # doctest: +NORMALIZE_WHITESPACE
    {'/data/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz': {},
     '/data/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz': {}}

    Paired scans, each in opposite PE directions
    >>> paired_dwi_files = [
    ...     "/data/sub-1/dwi/sub-1_dir-AP_dwi.nii.gz",
    ...     "/data/sub-1/dwi/sub-1_dir-PA_dwi.nii.gz"]
    >>> paired_dwi_file_metadatas = [
    ...     {"PhaseEncodingDirection": "j"},
    ...     {"PhaseEncodingDirection": "j-"}]
    >>> find_fieldmaps_from_other_dwis(paired_dwi_files,
    ...     paired_dwi_file_metadatas) # doctest: +NORMALIZE_WHITESPACE
    {'/data/sub-1/dwi/sub-1_dir-AP_dwi.nii.gz': {'suffix': 'dwi',
      'dwi': ['/data/sub-1/dwi/sub-1_dir-PA_dwi.nii.gz']},
     '/data/sub-1/dwi/sub-1_dir-PA_dwi.nii.gz': {'suffix': 'dwi',
      'dwi': ['/data/sub-1/dwi/sub-1_dir-AP_dwi.nii.gz']}}

    Multiple scans in multiple PE directions
    >>> multi_dwi_files = [
    ...     "/data/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz",
    ...     "/data/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz",
    ...     "/data/sub-1/dwi/sub-1_dir-AP_run-3_dwi.nii.gz",
    ...     "/data/sub-1/dwi/sub-1_dir-PA_run-1_dwi.nii.gz",
    ...     "/data/sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz"]
    >>> multi_dwi_file_metadatas = [
    ...     {"PhaseEncodingDirection": "j"},
    ...     {"PhaseEncodingDirection": "j"},
    ...     {"PhaseEncodingDirection": "j"},
    ...     {"PhaseEncodingDirection": "j-"},
    ...     {"PhaseEncodingDirection": "j-"}]
    >>> find_fieldmaps_from_other_dwis(multi_dwi_files,
    ...     multi_dwi_file_metadatas) # doctest: +NORMALIZE_WHITESPACE
    {'/data/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz': {'suffix': 'dwi',
      'dwi': ['/data/sub-1/dwi/sub-1_dir-PA_run-1_dwi.nii.gz',
       '/data/sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz']},
     '/data/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz': {'suffix': 'dwi',
      'dwi': ['/data/sub-1/dwi/sub-1_dir-PA_run-1_dwi.nii.gz',
       '/data/sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz']},
     '/data/sub-1/dwi/sub-1_dir-AP_run-3_dwi.nii.gz': {'suffix': 'dwi',
      'dwi': ['/data/sub-1/dwi/sub-1_dir-PA_run-1_dwi.nii.gz',
       '/data/sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz']},
     '/data/sub-1/dwi/sub-1_dir-PA_run-1_dwi.nii.gz': {'suffix': 'dwi',
      'dwi': ['/data/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
       '/data/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz',
       '/data/sub-1/dwi/sub-1_dir-AP_run-3_dwi.nii.gz']},
     '/data/sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz': {'suffix': 'dwi',
      'dwi': ['/data/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
       '/data/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz',
       '/data/sub-1/dwi/sub-1_dir-AP_run-3_dwi.nii.gz']}}

    No information available
    >>> empty_dwi_files = [
    ...     "/data/sub-1/dwi/sub-1_run-1_dwi.nii.gz",
    ...     "/data/sub-1/dwi/sub-1_run-2_dwi.nii.gz"]
    >>> empty_dwi_file_metadatas = [
    ...     {},
    ...     {}]
    >>> find_fieldmaps_from_other_dwis(empty_dwi_files,
    ...     empty_dwi_file_metadatas) # doctest: +NORMALIZE_WHITESPACE
    {'/data/sub-1/dwi/sub-1_run-1_dwi.nii.gz': {},
     '/data/sub-1/dwi/sub-1_run-2_dwi.nii.gz': {}}
    """

    scans_to_pe_dirs = {fname: meta.get("PhaseEncodingDirection", 'None') for fname, meta in
                        zip(dwi_files, dwi_file_metadatas)}
    pe_dirs_to_scans = defaultdict(list)
    for scan_name, scan_dir in scans_to_pe_dirs.items():
        pe_dirs_to_scans[scan_dir].append(scan_name)

    dwi_series_fieldmaps = {}
    for dwi_file in dwi_files:
        dwi_series_fieldmaps[dwi_file] = {}
        pe_dir = scans_to_pe_dirs[dwi_file]
        # if there is no information, don't assume it's ok to combine
        if pe_dir is None:
            continue
        opposite_pe = pe_dir[0] if pe_dir.endswith('-') else pe_dir + '-'
        rpe_dwis = pe_dirs_to_scans[opposite_pe]

        if rpe_dwis:
            dwi_series_fieldmaps[dwi_file] = {"suffix": "dwi", "dwi": sorted(rpe_dwis)}

    return dwi_series_fieldmaps


def split_by_phase_encoding_direction(dwi_files, metadatas):
    """If no fieldmaps have been found for a group of dwi files, split them by PE direction.

    **Examples**

    One of each direction (Not likely to see in the wild)
    >>> dwi_files = [
    ...     '/data/sub-1/dwi/sub-1_dir-AP_dwi.nii.gz',
    ...     '/data/sub-1/dwi/sub-1_dir-PA_dwi.nii.gz',
    ...     '/data/sub-1/dwi/sub-1_dir-RL_dwi.nii.gz',
    ...     '/data/sub-1/dwi/sub-1_dir-LR_dwi.nii.gz',
    ...     '/data/sub-1/dwi/sub-1_dir-IS_dwi.nii.gz',
    ...     '/data/sub-1/dwi/sub-1_dir-SI_dwi.nii.gz'
    ... ]
    >>> metadatas = [
    ...     {'PhaseEncodingDirection': 'j'},
    ...     {'PhaseEncodingDirection': 'j-'},
    ...     {'PhaseEncodingDirection': 'i'},
    ...     {'PhaseEncodingDirection': 'i-'},
    ...     {'PhaseEncodingDirection': 'k'},
    ...     {'PhaseEncodingDirection': 'k-'}
    ... ]
    >>> split_by_phase_encoding_direction(dwi_files, metadatas) # doctest: +NORMALIZE_WHITESPACE
    [{'dwi_series': ['/data/sub-1/dwi/sub-1_dir-RL_dwi.nii.gz'],
      'fieldmap_info': {'suffix': None},
      'dwi_series_pedir': 'i',
      'concatenated_bids_name': 'sub-1_dir-RL'},
     {'dwi_series': ['/data/sub-1/dwi/sub-1_dir-LR_dwi.nii.gz'],
      'fieldmap_info': {'suffix': None},
      'dwi_series_pedir': 'i-',
      'concatenated_bids_name': 'sub-1_dir-LR'},
     {'dwi_series': ['/data/sub-1/dwi/sub-1_dir-AP_dwi.nii.gz'],
      'fieldmap_info': {'suffix': None},
      'dwi_series_pedir': 'j',
      'concatenated_bids_name': 'sub-1_dir-AP'},
     {'dwi_series': ['/data/sub-1/dwi/sub-1_dir-PA_dwi.nii.gz'],
      'fieldmap_info': {'suffix': None},
      'dwi_series_pedir': 'j-',
      'concatenated_bids_name': 'sub-1_dir-PA'},
     {'dwi_series': ['/data/sub-1/dwi/sub-1_dir-IS_dwi.nii.gz'],
      'fieldmap_info': {'suffix': None},
      'dwi_series_pedir': 'k',
      'concatenated_bids_name': 'sub-1_dir-IS'},
     {'dwi_series': ['/data/sub-1/dwi/sub-1_dir-SI_dwi.nii.gz'],
      'fieldmap_info': {'suffix': None},
      'dwi_series_pedir': 'k-',
      'concatenated_bids_name': 'sub-1_dir-SI'}]

    Repeats of some:
    >>> dwi_files = [
    ...     '/data/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
    ...     '/data/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz',
    ...     '/data/sub-1/dwi/sub-1_dir-AP_run-3_dwi.nii.gz',
    ...     '/data/sub-1/dwi/sub-1_dir-PA_dwi.nii.gz',
    ...     '/data/sub-1/dwi/sub-1_dir-RL_dwi.nii.gz'
    ... ]
    >>> metadatas = [
    ...     {'PhaseEncodingDirection': 'j'},
    ...     {'PhaseEncodingDirection': 'j'},
    ...     {'PhaseEncodingDirection': 'j'},
    ...     {'PhaseEncodingDirection': 'j-'},
    ...     {'PhaseEncodingDirection': 'i'}
    ... ]
    >>> split_by_phase_encoding_direction(dwi_files, metadatas) # doctest: +NORMALIZE_WHITESPACE
    [{'dwi_series': ['/data/sub-1/dwi/sub-1_dir-RL_dwi.nii.gz'],
      'fieldmap_info': {'suffix': None},
      'dwi_series_pedir': 'i',
      'concatenated_bids_name': 'sub-1_dir-RL'},
     {'dwi_series': ['/data/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
       '/data/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz',
       '/data/sub-1/dwi/sub-1_dir-AP_run-3_dwi.nii.gz'],
      'fieldmap_info': {'suffix': None},
      'dwi_series_pedir': 'j',
      'concatenated_bids_name': 'sub-1_dir-AP'},
     {'dwi_series': ['/data/sub-1/dwi/sub-1_dir-PA_dwi.nii.gz'],
      'fieldmap_info': {'suffix': None},
      'dwi_series_pedir': 'j-',
      'concatenated_bids_name': 'sub-1_dir-PA'}]

    Some missing metadata
    >>> dwi_files = [
    ...     '/data/sub-1/dwi/sub-1_run-1_dwi.nii.gz',
    ...     '/data/sub-1/dwi/sub-1_run-2_dwi.nii.gz',
    ...     '/data/sub-1/dwi/sub-1_run-3_dwi.nii.gz',
    ...     '/data/sub-1/dwi/sub-1_run-4_dwi.nii.gz',
    ...     '/data/sub-1/dwi/sub-1_run-5_dwi.nii.gz'
    ... ]
    >>> metadatas = [
    ...     {'PhaseEncodingDirection': 'j'},
    ...     {'PhaseEncodingDirection': 'j'},
    ...     {'PhaseEncodingDirection': 'j-'},
    ...     {},
    ...     {}
    ... ]
    >>> split_by_phase_encoding_direction(dwi_files, metadatas) # doctest: +NORMALIZE_WHITESPACE
    [{'dwi_series': ['/data/sub-1/dwi/sub-1_run-1_dwi.nii.gz',
       '/data/sub-1/dwi/sub-1_run-2_dwi.nii.gz'],
      'fieldmap_info': {'suffix': None},
      'dwi_series_pedir': 'j',
      'concatenated_bids_name': 'sub-1'},
     {'dwi_series': ['/data/sub-1/dwi/sub-1_run-3_dwi.nii.gz'],
      'fieldmap_info': {'suffix': None},
      'dwi_series_pedir': 'j-',
      'concatenated_bids_name': 'sub-1_run-3'},
     {'dwi_series': ['/data/sub-1/dwi/sub-1_run-4_dwi.nii.gz'],
      'fieldmap_info': {'suffix': None},
      'dwi_series_pedir': '',
      'concatenated_bids_name': 'sub-1_run-4'},
     {'dwi_series': ['/data/sub-1/dwi/sub-1_run-5_dwi.nii.gz'],
      'fieldmap_info': {'suffix': None},
      'dwi_series_pedir': '',
      'concatenated_bids_name': 'sub-1_run-5'}]



    """
    pe_dir_groups = defaultdict(list)
    unknowns = []
    for dwi_file, meta in zip(dwi_files, metadatas):
        pe_dir = meta.get("PhaseEncodingDirection")
        if pe_dir:
            pe_dir_groups[pe_dir].append(dwi_file)
        else:
            unknowns.append(dwi_file)

    dwi_groups = []
    for pe_dir, dwi_group in sorted(pe_dir_groups.items()):
        dwi_groups.append(
            {'dwi_series': dwi_group,
             'fieldmap_info': {'suffix': None},
             'dwi_series_pedir': pe_dir,
             'concatenated_bids_name': get_concatenated_bids_name(dwi_group)})
    for unknown in unknowns:
        dwi_groups.append(
            {'dwi_series': [unknown],
             'fieldmap_info': {'suffix': None},
             'dwi_series_pedir': '',
             'concatenated_bids_name': get_concatenated_bids_name([unknown])})

    return dwi_groups


def group_by_warpspace(dwi_files, layout, ignore_fieldmaps):
    """Groups a session's DWI files by their acquisition parameters.

    DWIs are grouped by their **warped space**. Two DWI series that are
    listed in the IntendedFor field of a fieldmap are assumed to have the same
    susceptibility distortions and therefore be in the same warped space. The goal
    of this function is to combine DWI series into groups of acquisitions that
    are in the same warped space into a list of scans that can be combined after
    unwarping.

    Parameters
    -----------

        dwi_files: list
            A list of full paths to dwi nifti files in a BIDS tree

        layout: BIDSLayout
            A representation of the BIDS tree

        ignore_fieldmaps: bool
            If True, ignore any fieldmaps in the ``fmap/`` directory. Images in
            ``dwi/`` will still be considered for SDC.

    Examples:
    ---------

    Set up tests
    >>> from qsiprep.utils.bids import collect_data
    >>> SUBJECT_ID = "1"

    No fieldmap data, a single DWI series
    >>> subject_data, layout = collect_data("easy", SUBJECT_ID)
    >>> group_by_warpspace(
    ...     subject_data['dwi'], layout, False) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [{'dwi_series': ['...sub-1_dwi.nii.gz'],
     'fieldmap_info': {'suffix': None},
     'dwi_series_pedir': 'j',
     'concatenated_bids_name': 'sub-1'}]

    Two DWIs with the same PE direction, to be concatenated
    >>> subject_data, layout = collect_data("concat1", SUBJECT_ID)
    >>> group_by_warpspace(
    ...     subject_data['dwi'], layout, False) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [{'dwi_series': ['.../concat1/sub-1/dwi/sub-1_run-01_dwi.nii.gz',
                     '.../concat1/sub-1/dwi/sub-1_run-02_dwi.nii.gz'],
     'fieldmap_info': {'suffix': None},
     'dwi_series_pedir': 'j',
     'concatenated_bids_name': 'sub-1'}]

    Two DWI series intended to SDC each other
    >>> subject_data, layout = collect_data("opposite", SUBJECT_ID)
    >>> group_by_warpspace(
    ...     subject_data['dwi'], layout, False) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [{'dwi_series': ['.../opposite/sub-1/dwi/sub-1_dir-AP_dwi.nii.gz'],
      'fieldmap_info': {'suffix': 'dwi',
       'dwi': ['.../opposite/sub-1/dwi/sub-1_dir-PA_dwi.nii.gz']},
      'dwi_series_pedir': 'j',
      'concatenated_bids_name': 'sub-1_dir-AP'},
     {'dwi_series': ['.../opposite/sub-1/dwi/sub-1_dir-PA_dwi.nii.gz'],
      'fieldmap_info': {'suffix': 'dwi',
       'dwi': ['.../opposite/sub-1/dwi/sub-1_dir-AP_dwi.nii.gz']},
      'dwi_series_pedir': 'j-',
      'concatenated_bids_name': 'sub-1_dir-PA'}]

    Multiple DWI series in two different PE directions
    >>> subject_data, layout = collect_data("opposite_concat", SUBJECT_ID)
    >>> group_by_warpspace(
    ...     subject_data['dwi'], layout, False) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [{'dwi_series': ['.../opposite_concat/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
                     '.../opposite_concat/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz'],
      'fieldmap_info': {'suffix': 'dwi',
       'dwi': ['.../opposite_concat/sub-1/dwi/sub-1_dir-PA_run-1_dwi.nii.gz',
               '.../opposite_concat/sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz']},
      'dwi_series_pedir': 'j',
      'concatenated_bids_name': 'sub-1_dir-AP'},
     {'dwi_series': ['.../opposite_concat/sub-1/dwi/sub-1_dir-PA_run-1_dwi.nii.gz',
                     '.../opposite_concat/sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz'],
      'fieldmap_info': {'suffix': 'dwi',
       'dwi': ['.../opposite_concat/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
               '.../opposite_concat/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz']},
      'dwi_series_pedir': 'j-',
      'concatenated_bids_name': 'sub-1_dir-PA'}]

    A phasediff fieldmap defines the warped group
    >>> subject_data, layout = collect_data("phasediff", SUBJECT_ID)
    >>> group_by_warpspace(
    ...     subject_data['dwi'], layout, False) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [{'dwi_series': ['.../phasediff/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
                     '.../phasediff/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz'],
      'fieldmap_info': {'phasediff': '.../phasediff/sub-1/fmap/sub-1_phasediff.nii.gz',
                        'magnitude1': '.../magnitude1/sub-1/fmap/sub-1_magnitude1.nii.gz',
                        'suffix': 'phasediff'},
      'dwi_series_pedir': 'j',
      'concatenated_bids_name': 'sub-1_dir-AP'}]

    Two DWI series, each with its own fieldmap/warped space
    >>> subject_data, layout = collect_data("separate_fmaps", SUBJECT_ID)
    >>> group_by_warpspace(
    ...     subject_data['dwi'], layout, False) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [{'dwi_series': ['.../separate_fmaps/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz'],
      'fieldmap_info': {'suffix': 'epi',
       'epi': ['.../separate_fmaps/sub-1/fmap/sub-1_dir-PA_run-1_epi.nii.gz']},
      'dwi_series_pedir': 'j',
      'concatenated_bids_name': 'sub-1_dir-AP_run-1'},
     {'dwi_series': ['.../separate_fmaps/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz'],
      'fieldmap_info': {'suffix': 'epi',
       'epi': ['.../separate_fmaps/sub-1/fmap/sub-1_dir-PA_run-2_epi.nii.gz']},
      'dwi_series_pedir': 'j',
      'concatenated_bids_name': 'sub-1_dir-AP_run-2'}]

    Same as above but ignoring fieldmaps. Data gets concatenated
    >>> subject_data, layout = collect_data("separate_fmaps", SUBJECT_ID)
    >>> group_by_warpspace(
    ...     subject_data['dwi'], layout, True) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [{'dwi_series': ['.../separate_fmaps/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
       '.../separate_fmaps/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz'],
      'fieldmap_info': {'suffix': None},
      'dwi_series_pedir': 'j',
      'concatenated_bids_name': 'sub-1_dir-AP'}]

    Two DWI series, opposite PE directions, dedicated EPI fieldmap for each
    >>> subject_data, layout = collect_data("mixed_fmaps", SUBJECT_ID)
    >>> group_by_warpspace(
    ...     subject_data['dwi'], layout, False) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [{'dwi_series': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz'],
      'fieldmap_info': {'suffix': 'epi',
       'epi': ['.../mixed_fmaps/sub-1/fmap/sub-1_dir-PA_run-1_epi.nii.gz']},
      'dwi_series_pedir': 'j',
      'concatenated_bids_name': 'sub-1_dir-AP_run-1'},
     {'dwi_series': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz'],
      'fieldmap_info': {'suffix': 'epi',
       'epi': ['.../mixed_fmaps/sub-1/fmap/sub-1_dir-AP_run-2_epi.nii.gz']},
       'dwi_series_pedir': 'j-',
       'concatenated_bids_name': 'sub-1_dir-PA_run-2'}]

    Same as last one, but ignore fieldmaps. The DWI series will be used for SDC instead
    >>> subject_data, layout = collect_data("mixed_fmaps", SUBJECT_ID)
    >>> group_by_warpspace(
    ...     subject_data['dwi'], layout, True) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [{'dwi_series': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz'],
      'fieldmap_info': {'suffix': 'dwi',
       'dwi': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz']},
      'dwi_series_pedir': 'j',
      'concatenated_bids_name': 'sub-1_dir-AP_run-1'},
     {'dwi_series': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz'],
      'fieldmap_info': {'suffix': 'dwi',
       'dwi': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz']},
      'dwi_series_pedir': 'j-',
      'concatenated_bids_name': 'sub-1_dir-PA_run-2'}]


    There is no metadata related to epi distortion: don't concatenate anything
    >>> subject_data, layout = collect_data("missing_info", SUBJECT_ID)
    >>> group_by_warpspace(
    ...     subject_data['dwi'], layout, False) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [{'dwi_series': ['.../missing_info/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz'],
      'fieldmap_info': {'suffix': None},
      'dwi_series_pedir': '',
      'concatenated_bids_name': 'sub-1_dir-AP_run-1'},
     {'dwi_series': ['.../missing_info/sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz'],
      'fieldmap_info': {'suffix': None},
      'dwi_series_pedir': '',
      'concatenated_bids_name': 'sub-1_dir-PA_run-2'}]

    A bizarre mix of PE directions and some missing data
    >>> subject_data, layout = collect_data("wtf", SUBJECT_ID)
    >>> group_by_warpspace(
    ...     subject_data['dwi'], layout, False) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [{'dwi_series': ['.../wtf/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
       '.../wtf/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz'],
      'fieldmap_info': {'suffix': 'dwi',
       'dwi': ['.../wtf/sub-1/dwi/sub-1_dir-PA_run-1_dwi.nii.gz',
        '.../wtf/sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz']},
      'dwi_series_pedir': 'j',
      'concatenated_bids_name': 'sub-1_dir-AP'},
     {'dwi_series': ['.../wtf/sub-1/dwi/sub-1_dir-IS_dwi.nii.gz'],
      'fieldmap_info': {'suffix': None},
      'dwi_series_pedir': 'k-',
      'concatenated_bids_name': 'sub-1_dir-IS'},
     {'dwi_series': ['.../wtf/sub-1/dwi/sub-1_run-1_dwi.nii.gz'],
      'fieldmap_info': {'suffix': None},
      'dwi_series_pedir': '',
      'concatenated_bids_name': 'sub-1_run-1'},
     {'dwi_series': ['.../wtf/sub-1/dwi/sub-1_run-2_dwi.nii.gz'],
      'fieldmap_info': {'suffix': None},
      'dwi_series_pedir': '',
      'concatenated_bids_name': 'sub-1_run-2'},
     {'dwi_series': ['.../wtf/sub-1/dwi/sub-1_dir-PA_run-1_dwi.nii.gz',
       '.../wtf/sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz'],
      'fieldmap_info': {'suffix': 'dwi',
       'dwi': ['.../wtf/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
        '.../wtf/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz']},
      'dwi_series_pedir': 'j-',
      'concatenated_bids_name': 'sub-1_dir-PA'}]



    """
    # For doc-building
    if layout is None:
        LOGGER.warning("Assuming we're building docs")
        return [{'dwi_series': dwi_files,
                 'fieldmap_info': {'suffix': None},
                 'dwi_series_pedir': 'j',
                 'concatenated_bids_name': 'sub-1'}]

    # Get the metadata from every dwi file
    dwi_metadatas = [layout.get_metadata(dwi_file) for dwi_file in dwi_files]
    # Check for any data in dwi/ that could be used for distortion correction
    dwi_series_fieldmaps = find_fieldmaps_from_other_dwis(dwi_files, dwi_metadatas)

    # Find the best fieldmap for each file.
    best_fieldmap = {}
    grouped_by_fmap = defaultdict(list)
    for dwi_file in dwi_files:
        all_fmaps = [dwi_series_fieldmaps[dwi_file]]
        if not ignore_fieldmaps:
            fmap_fmaps = layout.get_fieldmap(dwi_file, return_list=True)
            all_fmaps += fmap_fmaps

        # Find the highest priority fieldmap for this dwi file
        best_fmap = get_highest_priority_fieldmap(all_fmaps)
        best_fieldmap[dwi_file] = best_fmap

        # Add the dwi file to a list of those corrected by this fieldmap
        fmap_key = tuple(best_fmap[best_fmap['suffix']]) if best_fmap['suffix'] else 'None'
        grouped_by_fmap[fmap_key].append(dwi_file)

    # Create the final groups
    dwi_groups = []
    for fmap_key, dwi_group in grouped_by_fmap.items():
        if fmap_key == 'None':
            dwi_groups.extend(
                split_by_phase_encoding_direction(dwi_group,
                                                  [layout.get_metadata(dwi_file) for dwi_file
                                                   in dwi_group]))
        else:
            example_dwi_file = dwi_group[0]
            pe_direction = layout.get_metadata(example_dwi_file).get('PhaseEncodingDirection')
            dwi_groups.append(
                {'dwi_series': dwi_group,
                 'fieldmap_info': best_fieldmap[example_dwi_file],
                 'dwi_series_pedir': pe_direction,
                 'concatenated_bids_name': get_concatenated_bids_name(dwi_group)})

    return dwi_groups


def merge_dwi_groups(dwi_groups_plus, dwi_groups_minus):
    """Convert two dwi groups into a single group that will be concatenated for FSL.

    Examples:
    ---------

    Set up tests
    >>> SUBJECT_ID = "1"

    AP/PA fieldmaps and paired DWI series
    >>> plus_groups = [
    ...     {'dwi_series': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz'],
    ...      'fieldmap_info': {'suffix': 'epi',
    ...       'epi': ['.../mixed_fmaps/sub-1/fmap/sub-1_dir-PA_run-1_epi.nii.gz']},
    ...      'dwi_series_pedir': 'j',
    ...      'concatenated_bids_name': 'sub-1_dir-AP_run-1'}]
    >>> minus_groups = [
    ...     {'dwi_series': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz'],
    ...      'fieldmap_info': {'suffix': 'epi',
    ...       'epi': ['.../mixed_fmaps/sub-1/fmap/sub-1_dir-AP_run-2_epi.nii.gz']},
    ...       'dwi_series_pedir': 'j-',
    ...       'concatenated_bids_name': 'sub-1_dir-PA_run-2'}]
    >>> merge_dwi_groups(plus_groups, minus_groups) # doctest: +NORMALIZE_WHITESPACE
    {'dwi_series': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz'],
     'dwi_series_pedir': 'j',
     'fieldmap_info': {'suffix': 'rpe_series',
      'rpe_series': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz'],
      'epi': ['.../mixed_fmaps/sub-1/fmap/sub-1_dir-AP_run-2_epi.nii.gz',
       '.../mixed_fmaps/sub-1/fmap/sub-1_dir-PA_run-1_epi.nii.gz']},
     'concatenated_bids_name': 'sub-1'}

    Two series SDC each other
    >>> plus_groups = [
    ...     {'dwi_series': ['.../opposite/sub-1/dwi/sub-1_dir-AP_dwi.nii.gz'],
    ...      'fieldmap_info': {'suffix': 'dwi',
    ...       'dwi': ['.../opposite/sub-1/dwi/sub-1_dir-PA_dwi.nii.gz']},
    ...      'dwi_series_pedir': 'j',
    ...      'concatenated_bids_name': 'sub-1_dir-AP'}]
    >>> minus_groups = [
    ...     {'dwi_series': ['.../opposite/sub-1/dwi/sub-1_dir-PA_dwi.nii.gz'],
    ...      'fieldmap_info': {'suffix': 'dwi',
    ...       'dwi': ['.../opposite/sub-1/dwi/sub-1_dir-AP_dwi.nii.gz']},
    ...      'dwi_series_pedir': 'j-',
    ...      'concatenated_bids_name': 'sub-1_dir-PA'}]
    >>> merge_dwi_groups(plus_groups, minus_groups) # doctest: +NORMALIZE_WHITESPACE
    {'dwi_series': ['.../opposite/sub-1/dwi/sub-1_dir-AP_dwi.nii.gz'],
     'dwi_series_pedir': 'j',
     'fieldmap_info': {'suffix': 'rpe_series',
      'rpe_series': ['.../opposite/sub-1/dwi/sub-1_dir-PA_dwi.nii.gz']},
     'concatenated_bids_name': 'sub-1'}

    An odd case: one has an EPI
    >>> plus_groups = [
    ...     {'dwi_series': ['.../opposite/sub-1/dwi/sub-1_dir-AP_dwi.nii.gz'],
    ...      'fieldmap_info': {'suffix': 'dwi',
    ...       'dwi': ['.../opposite/sub-1/dwi/sub-1_dir-PA_dwi.nii.gz']},
    ...      'dwi_series_pedir': 'j',
    ...      'concatenated_bids_name': 'sub-1_dir-AP'}]
    >>> minus_groups = [
    ...     {'dwi_series': ['.../opposite/sub-1/dwi/sub-1_dir-PA_dwi.nii.gz'],
    ...      'fieldmap_info': {'suffix': 'epi',
    ...       'epi': ['.../mixed_fmaps/sub-1/fmap/sub-1_dir-AP_run-2_epi.nii.gz']},
    ...      'dwi_series_pedir': 'j-',
    ...      'concatenated_bids_name': 'sub-1_dir-PA'}]
    >>> merge_dwi_groups(plus_groups, minus_groups) # doctest: +NORMALIZE_WHITESPACE
    {'dwi_series': ['.../opposite/sub-1/dwi/sub-1_dir-AP_dwi.nii.gz'],
     'dwi_series_pedir': 'j',
     'fieldmap_info': {'suffix': 'rpe_series',
      'rpe_series': ['.../opposite/sub-1/dwi/sub-1_dir-PA_dwi.nii.gz'],
      'epi': ['.../mixed_fmaps/sub-1/fmap/sub-1_dir-AP_run-2_epi.nii.gz']},
     'concatenated_bids_name': 'sub-1'}

    """
    dwi_files = []
    rpe_files = []
    fmap_files = []

    for dwi_group in dwi_groups_plus:
        dwi_files += dwi_group['dwi_series']
        fmap_type = dwi_group['fieldmap_info'].get('suffix')
        if fmap_type == 'dwi':
            rpe_files += dwi_group['fieldmap_info']['dwi']
        elif fmap_type == 'epi':
            fmap_files += dwi_group['fieldmap_info']['epi']
        pe_dir = dwi_group['dwi_series_pedir']

    for dwi_group in dwi_groups_minus:
        rpe_files += dwi_group['dwi_series']
        fmap_type = dwi_group['fieldmap_info'].get('suffix')
        if fmap_type == 'dwi':
            dwi_files += dwi_group['fieldmap_info']['dwi']
        elif fmap_type == 'epi':
            fmap_files += dwi_group['fieldmap_info']['epi']

    dwi_files = sorted(set(dwi_files))
    rpe_files = sorted(set(rpe_files))
    fmap_files = sorted(set(fmap_files))
    fieldmap_info = {
        "suffix": "rpe_series",
        "rpe_series": rpe_files
    }
    if fmap_files:
        fieldmap_info["epi"] = fmap_files

    merged_group = {
        "dwi_series": dwi_files,
        "dwi_series_pedir": pe_dir,
        "fieldmap_info": fieldmap_info,
        "concatenated_bids_name": get_concatenated_bids_name(dwi_files + rpe_files)
    }
    return merged_group


def group_for_eddy(all_dwi_fmap_groups):
    """Find matched pairs of phase encoding directions that can be combined for TOPUP/eddy.

    Any groups that don't have a phase encoding direction won't be correctable by eddy/TOPUP.

    Examples:
    ----------

    Paired DWI series to correct each other:
    >>> dwi_groups = [
    ...  {'dwi_series': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz'],
    ...      'fieldmap_info': {'suffix': 'dwi',
    ...      'dwi': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz']},
    ...   'dwi_series_pedir': 'j',
    ...      'concatenated_bids_name': 'sub-1_dir-AP_run-1'},
    ...  {'dwi_series': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz'],
    ...      'fieldmap_info': {'suffix': 'dwi',
    ...      'dwi': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz']},
    ...      'dwi_series_pedir': 'j-',
    ...   'concatenated_bids_name': 'sub-1_dir-PA_run-2'}]
    >>> group_for_eddy(dwi_groups) # doctest: +NORMALIZE_WHITESPACE
    [{'dwi_series': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz'],
      'dwi_series_pedir': 'j',
      'fieldmap_info': {'suffix': 'rpe_series',
       'rpe_series': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz']},
      'concatenated_bids_name': 'sub-1'}]

    AP/PA EPI fieldmaps
    >>> dwi_groups = [
    ...     {'dwi_series': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz'],
    ...      'fieldmap_info': {'suffix': 'epi',
    ...       'epi': ['.../mixed_fmaps/sub-1/fmap/sub-1_dir-PA_run-1_epi.nii.gz']},
    ...      'dwi_series_pedir': 'j',
    ...      'concatenated_bids_name': 'sub-1_dir-AP_run-1'},
    ...     {'dwi_series': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz'],
    ...      'fieldmap_info': {'suffix': 'epi',
    ...       'epi': ['.../mixed_fmaps/sub-1/fmap/sub-1_dir-AP_run-2_epi.nii.gz']},
    ...       'dwi_series_pedir': 'j-',
    ...       'concatenated_bids_name': 'sub-1_dir-PA_run-2'}]
    >>> group_for_eddy(dwi_groups) # doctest: +NORMALIZE_WHITESPACE
    [{'dwi_series': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz'],
      'dwi_series_pedir': 'j',
      'fieldmap_info': {'suffix': 'rpe_series',
       'rpe_series': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz'],
       'epi': ['.../mixed_fmaps/sub-1/fmap/sub-1_dir-AP_run-2_epi.nii.gz',
               '.../mixed_fmaps/sub-1/fmap/sub-1_dir-PA_run-1_epi.nii.gz']},
      'concatenated_bids_name': 'sub-1'}]

    Repeated scans per PE direction
    >>> dwi_groups = [
    ...    {'dwi_series': ['.../opposite_concat/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
    ...                     '.../opposite_concat/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz'],
    ...      'fieldmap_info': {'suffix': 'dwi',
    ...       'dwi': ['.../opposite_concat/sub-1/dwi/sub-1_dir-PA_run-1_dwi.nii.gz',
    ...               '.../opposite_concat/sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz']},
    ...      'dwi_series_pedir': 'j',
    ...      'concatenated_bids_name': 'sub-1_dir-AP'},
    ...     {'dwi_series': ['.../opposite_concat/sub-1/dwi/sub-1_dir-PA_run-1_dwi.nii.gz',
    ...                     '.../opposite_concat/sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz'],
    ...      'fieldmap_info': {'suffix': 'dwi',
    ...       'dwi': ['.../opposite_concat/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
    ...               '.../opposite_concat/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz']},
    ...      'dwi_series_pedir': 'j-',
    ...      'concatenated_bids_name': 'sub-1_dir-PA'}]
    >>> group_for_eddy(dwi_groups) # doctest: +NORMALIZE_WHITESPACE
    [{'dwi_series': ['.../opposite_concat/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
                     '.../opposite_concat/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz'],
      'dwi_series_pedir': 'j',
      'fieldmap_info': {'suffix': 'rpe_series',
       'rpe_series': ['.../opposite_concat/sub-1/dwi/sub-1_dir-PA_run-1_dwi.nii.gz',
                      '.../opposite_concat/sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz']},
      'concatenated_bids_name': 'sub-1'}]

    A phasediff fieldmap (Not used by eddy)
    >>> dwi_groups = [
    ...    {'dwi_series': ['.../phasediff/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
    ...                     '.../phasediff/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz'],
    ...      'fieldmap_info': {'phasediff': '.../phasediff/sub-1/fmap/sub-1_phasediff.nii.gz',
    ...                        'magnitude1': '.../magnitude1/sub-1/fmap/sub-1_magnitude1.nii.gz',
    ...                        'suffix': 'phasediff'},
    ...      'dwi_series_pedir': 'j',
    ...      'concatenated_bids_name': 'sub-1_dir-AP'}]
    >>> group_for_eddy(dwi_groups) # doctest: +NORMALIZE_WHITESPACE
    [{'dwi_series': ['.../phasediff/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
       '.../phasediff/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz'],
      'fieldmap_info': {'phasediff': '.../phasediff/sub-1/fmap/sub-1_phasediff.nii.gz',
       'magnitude1': '.../magnitude1/sub-1/fmap/sub-1_magnitude1.nii.gz',
       'suffix': 'phasediff'},
      'dwi_series_pedir': 'j',
      'concatenated_bids_name': 'sub-1_dir-AP'}]

    """
    eddy_dwi_groups = []
    eddy_compatible_suffixes = ('dwi', 'epi')
    session_groups = _group_by_sessions(all_dwi_fmap_groups)
    for _, dwi_fmap_groups in session_groups.items():
        for pe_dir in 'ijk':
            plus_series = [dwi_group for dwi_group in dwi_fmap_groups if
                           dwi_group.get('dwi_series_pedir') == pe_dir
                           and dwi_group['fieldmap_info'].get('suffix') in
                           eddy_compatible_suffixes]
            minus_series = [dwi_group for dwi_group in dwi_fmap_groups if
                            dwi_group.get('dwi_series_pedir') == pe_dir + '-'
                            and dwi_group['fieldmap_info'].get('suffix') in
                            eddy_compatible_suffixes]

            # Can these be grouped?
            if plus_series and minus_series:
                eddy_dwi_groups.append(merge_dwi_groups(plus_series, minus_series))
            else:
                eddy_dwi_groups.extend(plus_series + minus_series)

        # Add separate groups for non-compatible fieldmaps
        for dwi_group in dwi_fmap_groups:
            if dwi_group['fieldmap_info'].get('suffix') not in eddy_compatible_suffixes:
                eddy_dwi_groups.append(dwi_group)

    return eddy_dwi_groups, {group['concatenated_bids_name']: group['concatenated_bids_name']
                             for group in eddy_dwi_groups}


def group_for_concatenation(all_dwi_fmap_groups):
    """Find matched pairs of phase encoding directions that can be combined after SHORELine.
    """
    concatenation_grouping = {}
    session_groups = _group_by_sessions(all_dwi_fmap_groups)
    for _, dwi_fmap_groups in session_groups.items():
        all_images = []
        for group in dwi_fmap_groups:
            all_images.extend(group['dwi_series'])
        group_name = get_concatenated_bids_name(all_images)
        # Add separate groups for non-compatible fieldmaps
        for group in dwi_fmap_groups:
            concatenation_grouping[group['concatenated_bids_name']] = group_name

    return concatenation_grouping


def get_concatenated_bids_name(dwi_group):
    """Derive the output file name for a group of dwi files.

    Strip away non-shared key/values from the input list of files. This function
    assumes you have already split the dwi group into something meaningful and
    really want to combine all the inputs.

    **Examples**
    >>> get_concatenated_bids_name([
    ...    '/data/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
    ...    '/data/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz',
    ...    '/data/sub-1/dwi/sub-1_dir-AP_run-3_dwi.nii.gz',
    ...    '/data/sub-1/dwi/sub-1_dir-AP_run-4_dwi.nii.gz'
    ... ])
    'sub-1_dir-AP'

    >>> get_concatenated_bids_name([
    ...    '/data/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
    ...    '/data/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz',
    ...    '/data/sub-1/dwi/sub-1_dir-PA_run-1_dwi.nii.gz',
    ...    '/data/sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz'
    ... ])
    'sub-1'


    >>> get_concatenated_bids_name([
    ...    '/data/sub-1/dwi/sub-1_acq-HCP-dir-AP_run-1_dwi.nii.gz',
    ...    '/data/sub-1/dwi/sub-1_acq-HCP_dir-AP_run-2_dwi.nii.gz',
    ...    '/data/sub-1/dwi/sub-1_acq-HCP_dir-PA_run-1_dwi.nii.gz',
    ...    '/data/sub-1/dwi/sub-1_acq-HCP_dir-PA_run-2_dwi.nii.gz'
    ... ])
    'sub-1_acq-HCP'

    """
    # If a single file, use its name, otherwise use the common prefix
    if len(dwi_group) > 1:
        fname = _get_common_bids_fields(dwi_group)
        parts = fname.split('_')
        full_parts = [part for part in parts if not part.endswith('-')]
        fname = '_'.join(full_parts)
    else:
        input_fname = dwi_group[0]
        fname = split_filename(input_fname)[1]

    if fname.endswith("_dwi"):
        fname = fname[:-4]

    return fname.replace(".", "").replace(" ", "")


def _get_common_bids_fields(fnames):
    bids_keys = defaultdict(set)
    for fname in fnames:
        basename = split_filename(fname)[1]
        for token in basename.split("_"):
            parts = token.split("-")
            if len(parts) == 2:
                key, value = parts
                bids_keys[key].update((value,))

    # Find all the keys with a single unique value
    common_bids = []
    for key in ['sub', 'ses', 'acq', 'dir', 'run']:
        if len(bids_keys[key]) == 1:
            common_bids.append(key + "-" + bids_keys[key].pop())
    return "_".join(common_bids)


def _group_by_sessions(dwi_fmap_groups):
    """Create a lookup of distortion groups by session

    Paired DWI series to correct each other:
    >>> dwi_groups = [
    ...  {'dwi_series': ['.../mixed_fmaps/sub-1/ses-1/dwi/sub-1_ses-1_dir-AP_run-1_dwi.nii.gz'],
    ...      'fieldmap_info': {'suffix': 'dwi',
    ...      'dwi': ['.../mixed_fmaps/sub-1/ses-1/dwi/sub-1_ses-1_dir-PA_run-2_dwi.nii.gz']},
    ...   'dwi_series_pedir': 'j',
    ...      'concatenated_bids_name': 'sub-1_ses-1_dir-AP_run-1'},
    ...  {'dwi_series': ['.../mixed_fmaps/sub-1/ses-1/dwi/sub-1_ses-1_dir-PA_run-2_dwi.nii.gz'],
    ...      'fieldmap_info': {'suffix': 'dwi',
    ...      'dwi': ['.../mixed_fmaps/sub-1/ses-1/dwi/sub-1_ses-1_dir-AP_run-1_dwi.nii.gz']},
    ...      'dwi_series_pedir': 'j-',
    ...   'concatenated_bids_name': 'sub-1_ses-1_dir-PA_run-2'},
    ...  {'dwi_series': [
    ...          '.../opposite_concat/sub-1/ses-2/dwi/sub-1_ses-2/dir-AP_run-1_dwi.nii.gz',
    ...          '.../opposite_concat/sub-1/ses-2/dwi/sub-1_ses-2/dir-AP_run-2_dwi.nii.gz'],
    ...      'fieldmap_info': {'suffix': 'dwi',
    ...       'dwi': ['.../opposite_concat/sub-1/ses-2/dwi/sub-1_ses-2_dir-PA_run-1_dwi.nii.gz',
    ...               '.../opposite_concat/sub-1/ses-2/dwi/sub-1_ses-2_dir-PA_run-2_dwi.nii.gz']},
    ...      'dwi_series_pedir': 'j',
    ...      'concatenated_bids_name': 'sub-1_ses-2_dir-AP'},
    ...     {'dwi_series': [
    ...           '.../opposite_concat/sub-1/ses-2/dwi/sub-1_ses-2_dir-PA_run-1_dwi.nii.gz',
    ...           '.../opposite_concat/sub-1/ses-2/dwi/sub-1_ses-2_dir-PA_run-2_dwi.nii.gz'],
    ...      'fieldmap_info': {'suffix': 'dwi',
    ...       'dwi': ['.../opposite_concat/sub-1/ses-2/dwi/sub-1_ses-2_dir-AP_run-1_dwi.nii.gz',
    ...               '.../opposite_concat/sub-1/ses-2/dwi/sub-1_ses-2_dir-AP_run-2_dwi.nii.gz']},
    ...      'dwi_series_pedir': 'j-',
    ...      'concatenated_bids_name': 'sub-1_ses-2_dir-PA'}]


    """
    ses_lookup = defaultdict(list)
    for group in dwi_fmap_groups:
        bids_info = get_bids_params(group['concatenated_bids_name'])
        ses_lookup[bids_info['session_id']].append(group)
    return ses_lookup
