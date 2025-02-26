# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Utilities to group scans based on their acquisition parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Download many variations of fieldmaps and dwi data

Examples
--------
Set up tests
>>> import os
>>> from qsiprep.utils.testing import get_grouping_test_data
>>> data_root = get_grouping_test_data()
>>> os.chdir(data_root)
"""

import logging
import pprint
from collections import defaultdict

from nipype.utils.filemanip import split_filename

from .. import config
from ..interfaces.bids import get_bids_params

LOGGER = logging.getLogger('nipype.workflow')


def group_dwi_scans(
    subject_data,
    using_fsl=False,
    combine_scans=True,
    ignore_fieldmaps=False,
    concatenate_distortion_groups=False,
):
    """Determine which scans can be concatenated based on their acquisition parameters.

    Parameters
    ----------
    bids_layout : :obj:`pybids.BIDSLayout`
        A PyBIDS layout
    using_fsl : :obj:`bool`
        Should a plus and minus series be grouped together for TOPUP/eddy?
    combine_scans : :obj:`bool`
        Should scan concatention happen?
    ignore_fieldmaps : :obj:`bool`
        Should fieldmaps be ignored?
    concatenate_distortion_groups : :obj:`bool`
        Will distortion groups get merged at the end of the pipeline?

    Returns
    -------
    scan_groups : :obj:`list` of :obj:`dict`
        A dict where the keys are the BIDS derivatives name of the output file after
        concatenation. The values are lists of dwi files in that group.
    concatenation_grouping : :obj:`dict`
        A dictionary mapping the concatenated BIDS name of each group to the name of the
        group that it should be concatenated with.
    """
    config.loggers.workflow.info('Grouping DWI scans')

    # Handle the grouping of multiple dwi files within a session
    dwi_entity_groups = get_entity_groups(config.execution.layout, subject_data, combine_scans)

    # Split the entity groups into groups of files with compatible warp groups
    dwi_fmap_groups = []
    for dwi_entity_group in dwi_entity_groups:
        dwi_fmap_groups.extend(
            group_by_warpspace(dwi_entity_group, config.execution.layout, ignore_fieldmaps)
        )

    if using_fsl:
        eddy_groups, concatenation_grouping = group_for_eddy(dwi_fmap_groups)
        config.loggers.workflow.info('Finished grouping DWI scans')
        return eddy_groups, concatenation_grouping

    if concatenate_distortion_groups:
        concatenation_grouping = group_for_concatenation(dwi_fmap_groups)
        config.loggers.workflow.info('Finished grouping DWI scans')
        return dwi_fmap_groups, concatenation_grouping

    config.loggers.workflow.info('Finished grouping DWI scans')
    return dwi_fmap_groups, {}


def get_entity_groups(layout, subject_data, combine_all_dwis):
    """Handle the grouping of multiple DWI files.

    This function will group DWI files based on the MultipartID metadata field,
    when available, and will default to considering all DWIs in a session as a
    "group" when it is not.

    Parameters
    ----------
    layout : :obj:`pybids.BIDSLayout`
        A PyBIDS layout
    subject_data : :obj:`dict`
        A dictionary of BIDS data for a single subject
    combine_all_dwis : :obj:`bool`
        If True, combine all DWI files within a session into a single group

    Returns
    -------
    dwi_entity_groups : :obj:`list` of :obj:`list`
        A list of lists of DWI files.
        Each list of DWI files is a group of scans that can be concatenated together.
    """
    all_dwis = subject_data['dwi']
    if not combine_all_dwis:
        dwi_groups = [[dwi] for dwi in all_dwis]
        return dwi_groups

    all_metadata = []
    for f in all_dwis:
        metadata = layout.get_file(f).get_metadata()
        all_metadata.append(metadata)

    grouping_method = 'entities'  # default is to group by entities
    if any('MultipartID' in metadata for metadata in all_metadata):
        grouping_method = 'metadata'

    dwi_entities = {}
    grouping_metadata = {}
    for i_file, f in enumerate(all_dwis):
        if grouping_method == 'metadata':
            # One MultipartID in any DWI metadata file means we use MultipartID to group
            # If a DWI has no MultipartID, it will be placed in a group by itself
            grouping_metadata[f] = all_metadata[i_file].get('MultipartID', None)
        else:
            # Group by entity instead
            f_entities = layout.get_file(f).get_entities()
            for k, v in f_entities.items():
                if k in dwi_entities:
                    if v not in dwi_entities[k]:
                        dwi_entities[k].append(v)
                else:
                    dwi_entities[k] = [v]

    if grouping_method == 'metadata':
        # Overwrite the existing dwi_groups (list) with a dict of lists
        LOGGER.info('Using MultipartID to group DWI files')
        dwi_groups = {}
        none_counter = 0
        for f in all_dwis:
            group = grouping_metadata[f]
            if group is None:
                # ! is not common, so it should be safe to use here without
                # conflicting with a valid MultipartID
                group = f'!none{none_counter}'
                none_counter += 1

            if group not in dwi_groups:
                dwi_groups[group] = []

            dwi_groups[group].append(f)

        for multipart_id, group_files in dwi_groups.items():
            if multipart_id.startswith('!'):
                LOGGER.info(
                    '\t- %d scan without MultipartID (set to %s)',
                    len(group_files),
                    multipart_id[1:],
                )
            else:
                LOGGER.info(
                    '\t- %d scans with MultipartID %s',
                    len(group_files),
                    multipart_id,
                )

        # Convert to list of lists
        dwi_groups = list(dwi_groups.values())

    elif grouping_method == 'entities':
        LOGGER.info('Combining all DWI files within each available session')
        # Group by session
        dwi_groups = []
        sessions = dwi_entities.get('session', [None])
        for session in sessions:
            session_files = [
                img for img in all_dwis if layout.get_file(img).entities.get('session') == session
            ]

            LOGGER.info(
                '\t- %d scans in session %s',
                len(session_files),
                session,
            )
            dwi_groups.append(session_files)

    return dwi_groups


FMAP_PRIORITY = {
    'dwi': 0,
    'epi': 1,
    'fieldmap': 2,
    'phasediff': 3,
    'phase1': 4,
    'phase': 4,
    'syn': 5,
}


def get_highest_priority_fieldmap(fmap_infos):
    """Return a dictionary describing the highest priority fieldmap.

    Parameters
    ----------
    fmap_infos : :obj:`list` of :obj:`dict`
        A list of dictionaries describing fieldmaps. Each dictionary must have a
        ``suffix`` key and may have an ``epi`` key.

    Returns
    -------
    selected_fmap_info : :obj:`dict`
        The dictionary describing the highest priority fieldmap.
        This will be the entry from ``fmap_infos`` with the highest priority value.
        If no fieldmaps are found, the dictionary will have a ``suffix`` key with a
        value of ``None``.

    Examples
    --------
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
    epi_fmaps = sorted(
        [fmap_info['epi'] for fmap_info in fmap_infos if fmap_info.get('suffix') == 'epi']
    )
    if epi_fmaps:
        epi_info = {'suffix': 'epi', 'epi': epi_fmaps}
        fmap_infos = [
            fmap_info for fmap_info in fmap_infos if fmap_info.get('suffix') != 'epi'
        ] + [epi_info]

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

    Parameters
    ----------
    dwi_files : :obj:`list` of :obj:`str`
        A list of full paths to dwi nifti files in a BIDS tree.
    dwi_file_metadatas : :obj:`list` of :obj:`dict`
        A list of dictionaries containing metadata for each dwi file.
        Each dictionary should have a ``PhaseEncodingDirection`` key.

    Returns
    -------
    dwi_series_fieldmaps : :obj:`dict`
        A dictionary where the keys are the full paths to dwi files and the values are
        dictionaries describing the fieldmap. If no fieldmap is found, the dictionary
        will be empty.

    Examples
    --------

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

    scans_to_pe_dirs = {
        fname: meta.get('PhaseEncodingDirection', 'None')
        for fname, meta in zip(dwi_files, dwi_file_metadatas, strict=False)
    }
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
            dwi_series_fieldmaps[dwi_file] = {'suffix': 'dwi', 'dwi': sorted(rpe_dwis)}

    return dwi_series_fieldmaps


def split_by_phase_encoding_direction(dwi_files, metadatas):
    """If no fieldmaps have been found for a group of dwi files, split them by PE direction.

    Parameters
    ----------
    dwi_files : :obj:`list` of :obj:`str`
        A list of full paths to dwi nifti files in a BIDS tree.
    metadatas : :obj:`list` of :obj:`dict`
        A list of dictionaries containing metadata for each dwi file.
        The only field that is used i "PhaseEncodingDirection".

    Returns
    -------
    dwi_groups : :obj:`list` of :obj:`dict`
        A list of dictionaries describing each group of dwi files. Each dictionary
        has the following keys:

        - ``dwi_series``: A list of full paths to dwi nifti files in a BIDS tree.
        - ``fieldmap_info``: A dictionary describing the fieldmap.
            If no fieldmap is found, the dictionary will be empty.
        - ``dwi_series_pedir``: The phase encoding direction of the dwi series.
            If no information is available, the value will be an empty string.
        - ``concatenated_bids_name``: The BIDS name of the concatenated dwi series.
            If no information is available, the value will be an empty string.

    Examples
    --------

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
    for dwi_file, meta in zip(dwi_files, metadatas, strict=False):
        pe_dir = meta.get('PhaseEncodingDirection')
        if pe_dir:
            pe_dir_groups[pe_dir].append(dwi_file)
        else:
            unknowns.append(dwi_file)

    dwi_groups = []
    for pe_dir, dwi_group in sorted(pe_dir_groups.items()):
        dwi_groups.append(
            {
                'dwi_series': dwi_group,
                'fieldmap_info': {'suffix': None},
                'dwi_series_pedir': pe_dir,
                'concatenated_bids_name': get_concatenated_bids_name(dwi_group),
            }
        )
    for unknown in unknowns:
        dwi_groups.append(
            {
                'dwi_series': [unknown],
                'fieldmap_info': {'suffix': None},
                'dwi_series_pedir': '',
                'concatenated_bids_name': get_concatenated_bids_name([unknown]),
            }
        )

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
    ----------
    dwi_files : :obj:`list` of :obj:`str`
        A list of full paths to dwi nifti files in a BIDS tree
    layout : :obj:`pybids.BIDSLayout`
        A representation of the BIDS tree
    ignore_fieldmaps : :obj:`bool`
        If True, ignore any fieldmaps in the ``fmap/`` directory. Images in
        ``dwi/`` will still be considered for SDC.

    Returns
    -------
    dwi_groups : :obj:`list` of :obj:`dict`
        A list of dictionaries describing each group of dwi files. Each dictionary
        has the following keys:

        - ``dwi_series``: A list of full paths to dwi nifti files in a BIDS tree.
        - ``fieldmap_info``: A dictionary describing the fieldmap.
            If no fieldmap is found, the dictionary will be empty.
        - ``dwi_series_pedir``: The phase encoding direction of the dwi series.
            If no information is available, the value will be an empty string.
        - ``concatenated_bids_name``: The BIDS name of the concatenated dwi series.
            If no information is available, the value will be an empty string.

    Examples
    --------

    Set up tests
    >>> from pprint import pprint
    >>> from qsiprep.utils.bids import collect_data
    >>> SUBJECT_ID = "1"

    No fieldmap data, a single DWI series
    >>> subject_data, layout = collect_data("easy", SUBJECT_ID)
    >>> pprint(group_by_warpspace(
    ...     subject_data['dwi'], layout, False)) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [{'dwi_series': ['...sub-1_dwi.nii.gz'],
     'fieldmap_info': {'suffix': None},
     'dwi_series_pedir': 'j',
     'concatenated_bids_name': 'sub-1'}]

    Two DWIs with the same PE direction, to be concatenated
    >>> subject_data, layout = collect_data("concat1", SUBJECT_ID)
    >>> pprint(group_by_warpspace(
    ...     subject_data['dwi'], layout, False)) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [{'dwi_series': ['.../concat1/sub-1/dwi/sub-1_run-01_dwi.nii.gz',
                     '.../concat1/sub-1/dwi/sub-1_run-02_dwi.nii.gz'],
     'fieldmap_info': {'suffix': None},
     'dwi_series_pedir': 'j',
     'concatenated_bids_name': 'sub-1'}]

    Two DWI series intended to SDC each other
    >>> subject_data, layout = collect_data("opposite", SUBJECT_ID)
    >>> pprint(group_by_warpspace(
    ...     subject_data['dwi'], layout, False)) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
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
    >>> pprint(group_by_warpspace(
    ...     subject_data['dwi'], layout, False)) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
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
    >>> pprint(group_by_warpspace(
    ...     subject_data['dwi'], layout, False)) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [{'dwi_series': ['.../phasediff/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
                     '.../phasediff/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz'],
      'fieldmap_info': {'phasediff': '.../phasediff/sub-1/fmap/sub-1_phasediff.nii.gz',
                        'magnitude1': '.../magnitude1/sub-1/fmap/sub-1_magnitude1.nii.gz',
                        'suffix': 'phasediff'},
      'dwi_series_pedir': 'j',
      'concatenated_bids_name': 'sub-1_dir-AP'}]

    Two DWI series, each with its own fieldmap/warped space
    >>> subject_data, layout = collect_data("separate_fmaps", SUBJECT_ID)
    >>> pprint(group_by_warpspace(
    ...     subject_data['dwi'], layout, False)) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
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
    >>> pprint(group_by_warpspace(
    ...     subject_data['dwi'], layout, True)) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [{'dwi_series': ['.../separate_fmaps/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
       '.../separate_fmaps/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz'],
      'fieldmap_info': {'suffix': None},
      'dwi_series_pedir': 'j',
      'concatenated_bids_name': 'sub-1_dir-AP'}]

    Two DWI series, opposite PE directions, dedicated EPI fieldmap for each
    >>> subject_data, layout = collect_data("mixed_fmaps", SUBJECT_ID)
    >>> pprint(group_by_warpspace(
    ...     subject_data['dwi'], layout, False)) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
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
    >>> pprint(group_by_warpspace(
    ...     subject_data['dwi'], layout, True)) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
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
    >>> pprint(group_by_warpspace(
    ...     subject_data['dwi'], layout, False)) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
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
    >>> pprint(group_by_warpspace(
    ...     subject_data['dwi'], layout, False)) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
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
        return [
            {
                'dwi_series': dwi_files,
                'fieldmap_info': {'suffix': None},
                'dwi_series_pedir': 'j',
                'concatenated_bids_name': 'sub-1',
            }
        ]

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
                split_by_phase_encoding_direction(
                    dwi_group, [layout.get_metadata(dwi_file) for dwi_file in dwi_group]
                )
            )
        else:
            example_dwi_file = dwi_group[0]
            pe_direction = layout.get_metadata(example_dwi_file).get('PhaseEncodingDirection')
            dwi_groups.append(
                {
                    'dwi_series': dwi_group,
                    'fieldmap_info': best_fieldmap[example_dwi_file],
                    'dwi_series_pedir': pe_direction,
                    'concatenated_bids_name': get_concatenated_bids_name(dwi_group),
                }
            )

    config.loggers.workflow.info(
        f'Found {len(dwi_groups)} groups of DWI series based on their warp spaces:\n'
        f'{pprint.pformat(dwi_groups, indent=2, width=120)}'
    )

    return dwi_groups


def merge_dwi_groups(dwi_groups_plus, dwi_groups_minus):
    """Convert two dwi groups into a single group that will be concatenated for FSL.

    Parameters
    ----------
    dwi_groups_plus : :obj:`list` of :obj:`dict`
        A list of dictionaries describing each group of dwi files.
        Each dictionary has the following keys:

        - ``dwi_series``: A list of full paths to dwi nifti files in a BIDS tree.
        - ``fieldmap_info``: A dictionary describing the fieldmap.
            If no fieldmap is found, the dictionary will be empty.
        - ``dwi_series_pedir``: The phase encoding direction of the dwi series.
            If no information is available, the value will be an empty string.
        - ``concatenated_bids_name``: The BIDS name of the concatenated dwi series.
            If no information is available, the value will be an empty string.

    dwi_groups_minus : :obj:`list` of :obj:`dict`
        A list of dictionaries describing each group of dwi files.
        Each dictionary has the same keys as ``dwi_groups_plus``.

    Returns
    -------
    merged_group : :obj:`dict`
        A dictionary describing the merged group of dwi files.
        The dictionary has the same keys as ``dwi_groups_plus``.

    Examples
    --------

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
    fieldmap_info = {'suffix': 'rpe_series', 'rpe_series': rpe_files}
    if fmap_files:
        fieldmap_info['epi'] = fmap_files

    merged_group = {
        'dwi_series': dwi_files,
        'dwi_series_pedir': pe_dir,
        'fieldmap_info': fieldmap_info,
        'concatenated_bids_name': get_concatenated_bids_name(dwi_files + rpe_files),
    }
    return merged_group


def group_for_eddy(all_dwi_fmap_groups):
    """Find matched pairs of phase encoding directions that can be combined for TOPUP/eddy.

    Any groups that don't have a phase encoding direction won't be correctable by eddy/TOPUP.

    Parameters
    ----------
    all_dwi_fmap_groups : :obj:`list` of :obj:`dict`
        A list of dictionaries describing each group of dwi files.

    Returns
    -------
    eddy_groups : :obj:`list` of :obj:`dict`
        A list of dictionaries describing each group of dwi files.
        Each dictionary has the following keys:

        - ``dwi_series``: A list of full paths to dwi nifti files in a BIDS tree.
        - ``fieldmap_info``: A dictionary describing the fieldmap.
            If no fieldmap is found, the dictionary will be empty.
        - ``dwi_series_pedir``: The phase encoding direction of the dwi series.
            If no information is available, the value will be an empty string.
        - ``concatenated_bids_name``: The BIDS name of the concatenated dwi series.
            If no information is available, the value will be an empty string.
    concatenation_grouping : :obj:`dict`
        A dictionary mapping the concatenated BIDS name of each group to the name of the
        group that it should be concatenated with.

    Examples
    --------

    Paired DWI series to correct each other:
    >>> from pprint import pprint
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
    >>> pprint(group_for_eddy(dwi_groups)) # doctest: +NORMALIZE_WHITESPACE
    ([{'concatenated_bids_name': 'sub-1',
       'dwi_series': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz'],
       'dwi_series_pedir': 'j',
       'fieldmap_info': {'rpe_series': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz'],
                         'suffix': 'rpe_series'}}],
     {'sub-1': 'sub-1'})

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
    >>> pprint(group_for_eddy(dwi_groups)) # doctest: +NORMALIZE_WHITESPACE
    ([{'concatenated_bids_name': 'sub-1',
       'dwi_series': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz'],
       'dwi_series_pedir': 'j',
       'fieldmap_info': {'epi': ['.../mixed_fmaps/sub-1/fmap/sub-1_dir-AP_run-2_epi.nii.gz',
                                 '.../mixed_fmaps/sub-1/fmap/sub-1_dir-PA_run-1_epi.nii.gz'],
                         'rpe_series': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz'],
                         'suffix': 'rpe_series'}}],
     {'sub-1': 'sub-1'})

    Val's scenario
    >>> dwi_groups = [
    ...     {'dwi_series': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz'],
    ...      'fieldmap_info': {'suffix': 'epi',
    ...       'epi': ['.../mixed_fmaps/sub-1/fmap/sub-1_dir-PA_run-1_epi.nii.gz']},
    ...      'dwi_series_pedir': 'j',
    ...      'concatenated_bids_name': 'sub-1_dir-AP_run-1'},
    ...     {'dwi_series': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz'],
    ...      'fieldmap_info': {'suffix': 'epi',
    ...       'epi': ['.../mixed_fmaps/sub-1/fmap/sub-1_dir-PA_run-2_epi.nii.gz']},
    ...       'dwi_series_pedir': 'j',
    ...       'concatenated_bids_name': 'sub-1_dir-AP_run-2'}]
    >>> pprint(group_for_eddy(dwi_groups)) # doctest: +NORMALIZE_WHITESPACE
    ([{'concatenated_bids_name': 'sub-1_dir-AP_run-1',
       'dwi_series': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz'],
       'dwi_series_pedir': 'j',
       'fieldmap_info': {'epi': ['.../mixed_fmaps/sub-1/fmap/sub-1_dir-PA_run-1_epi.nii.gz'],
                         'suffix': 'epi'}},
      {'concatenated_bids_name': 'sub-1_dir-AP_run-2',
       'dwi_series': ['.../mixed_fmaps/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz'],
       'dwi_series_pedir': 'j',
       'fieldmap_info': {'epi': ['.../mixed_fmaps/sub-1/fmap/sub-1_dir-PA_run-2_epi.nii.gz'],
                         'suffix': 'epi'}}],
        {'sub-1_dir-AP_run-1': 'sub-1_dir-AP_run-1',
         'sub-1_dir-AP_run-2': 'sub-1_dir-AP_run-2'})

    Repeated scans per PE direction
    >>> dwi_groups = [
    ...    {'dwi_series': ['.../sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
    ...                     '.../sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz'],
    ...      'fieldmap_info': {'suffix': 'dwi',
    ...       'dwi': ['.../sub-1/dwi/sub-1_dir-PA_run-1_dwi.nii.gz',
    ...               '.../sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz']},
    ...      'dwi_series_pedir': 'j',
    ...      'concatenated_bids_name': 'sub-1_dir-AP'},
    ...     {'dwi_series': ['.../sub-1/dwi/sub-1_dir-PA_run-1_dwi.nii.gz',
    ...                     '.../sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz'],
    ...      'fieldmap_info': {'suffix': 'dwi',
    ...       'dwi': ['.../sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
    ...               '.../sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz']},
    ...      'dwi_series_pedir': 'j-',
    ...      'concatenated_bids_name': 'sub-1_dir-PA'}]
    >>> pprint(group_for_eddy(dwi_groups)) # doctest: +NORMALIZE_WHITESPACE
    ([{'concatenated_bids_name': 'sub-1',
       'dwi_series': ['.../sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
                      '.../sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz'],
       'dwi_series_pedir': 'j',
       'fieldmap_info': {'rpe_series': ['.../sub-1/dwi/sub-1_dir-PA_run-1_dwi.nii.gz',
                                        '.../sub-1/dwi/sub-1_dir-PA_run-2_dwi.nii.gz'],
                         'suffix': 'rpe_series'}}],
     {'sub-1': 'sub-1'})

    A phasediff fieldmap (Not used by eddy)
    >>> dwi_groups = [
    ...    {'dwi_series': ['.../phasediff/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
    ...                     '.../phasediff/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz'],
    ...      'fieldmap_info': {'phasediff': '.../phasediff/sub-1/fmap/sub-1_phasediff.nii.gz',
    ...                        'magnitude1': '.../magnitude1/sub-1/fmap/sub-1_magnitude1.nii.gz',
    ...                        'suffix': 'phasediff'},
    ...      'dwi_series_pedir': 'j',
    ...      'concatenated_bids_name': 'sub-1_dir-AP'}]
    >>> pprint(group_for_eddy(dwi_groups)) # doctest: +NORMALIZE_WHITESPACE
    ([{'concatenated_bids_name': 'sub-1_dir-AP',
       'dwi_series': ['.../phasediff/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
                      '.../phasediff/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz'],
       'dwi_series_pedir': 'j',
       'fieldmap_info': {'magnitude1': '.../magnitude1/sub-1/fmap/sub-1_magnitude1.nii.gz',
                         'phasediff': '.../phasediff/sub-1/fmap/sub-1_phasediff.nii.gz',
                         'suffix': 'phasediff'}}],
     {'sub-1_dir-AP': 'sub-1_dir-AP'})

    """
    eddy_dwi_groups = []
    eddy_compatible_suffixes = ('dwi', 'epi')
    session_groups = _group_by_sessions(all_dwi_fmap_groups)
    for _, dwi_fmap_groups in session_groups.items():
        for pe_dir in 'ijk':
            plus_series = [
                dwi_group
                for dwi_group in dwi_fmap_groups
                if dwi_group.get('dwi_series_pedir') == pe_dir
                and dwi_group['fieldmap_info'].get('suffix') in eddy_compatible_suffixes
            ]
            minus_series = [
                dwi_group
                for dwi_group in dwi_fmap_groups
                if dwi_group.get('dwi_series_pedir') == pe_dir + '-'
                and dwi_group['fieldmap_info'].get('suffix') in eddy_compatible_suffixes
            ]

            # Can these be grouped?
            if plus_series and minus_series:
                eddy_dwi_groups.append(merge_dwi_groups(plus_series, minus_series))
            else:
                eddy_dwi_groups.extend(plus_series + minus_series)

        # Add separate groups for non-compatible fieldmaps
        for dwi_group in dwi_fmap_groups:
            if dwi_group['fieldmap_info'].get('suffix') not in eddy_compatible_suffixes:
                eddy_dwi_groups.append(dwi_group)

    config.loggers.workflow.info(
        f'Found {len(eddy_dwi_groups)} groups of DWI series that can be corrected by eddy:\n'
        f'{pprint.pformat(eddy_dwi_groups, indent=2, width=120)}'
    )

    return eddy_dwi_groups, {
        group['concatenated_bids_name']: group['concatenated_bids_name']
        for group in eddy_dwi_groups
    }


def group_for_concatenation(all_dwi_fmap_groups):
    """Find matched pairs of phase encoding directions that can be combined after SHORELine.

    Any groups that don't have a phase encoding direction won't be correctable by SHORELine.

    Parameters
    ----------
    all_dwi_fmap_groups : :obj:`list` of :obj:`dict`
        A list of dictionaries describing each group of dwi files.

    Returns
    -------
    concatenation_grouping : :obj:`dict`
        A dictionary mapping the concatenated BIDS name of each group to the name of the
        group that it should be concatenated with.
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

    config.loggers.workflow.info(
        f'Found {len(concatenation_grouping)} groups of DWI series that can be concatenated:\n'
        f'{pprint.pformat(concatenation_grouping, indent=2, width=120)}'
    )

    return concatenation_grouping


def get_concatenated_bids_name(dwi_group):
    """Derive the output file name for a group of dwi files.

    Strip away non-shared key/values from the input list of files. This function
    assumes you have already split the dwi group into something meaningful and
    really want to combine all the inputs.

    Parameters
    ----------
    dwi_group : :obj:`list` of :obj:`str`
        A list of full paths to dwi nifti files in a BIDS tree.

    Returns
    -------
    fname : :obj:`str`
        The BIDS name of the concatenated dwi series.

    Examples
    --------
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

    if fname.endswith('_dwi'):
        fname = fname[:-4]

    return fname.replace('.', '').replace(' ', '')


def _get_common_bids_fields(fnames):
    """Find the common fields in a list of BIDS filenames.

    Parameters
    ----------
    fnames : :obj:`list` of :obj:`str`
        A list of full paths to dwi nifti files in a BIDS tree.

    Returns
    -------
    fname : :obj:`str`
        The common fields in the filenames.
    """
    bids_keys = defaultdict(set)
    for fname in fnames:
        basename = split_filename(fname)[1]
        for token in basename.split('_'):
            parts = token.split('-')
            if len(parts) == 2:
                key, value = parts
                bids_keys[key].update((value,))

    # Find all the keys with a single unique value
    common_bids = []
    for key in ['sub', 'ses', 'acq', 'dir', 'run']:
        if len(bids_keys[key]) == 1:
            common_bids.append(key + '-' + bids_keys[key].pop())

    return '_'.join(common_bids)


def _group_by_sessions(dwi_fmap_groups):
    """Create a lookup of distortion groups by session

    Parameters
    ----------
    dwi_fmap_groups : :obj:`list` of :obj:`dict`
        A list of dictionaries describing each group of dwi files.

    Returns
    -------
    ses_lookup : :obj:`dict`
        A dictionary mapping session ids to lists of dwi_fmap_groups.

    Examples
    --------
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
