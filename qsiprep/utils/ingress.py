#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Utilities to handle data from other preprocessing pipelines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

"""
import re
from pathlib import Path

UKB_DIR_PATTERN = re.compile("(\d+)_(\d+)_(\d+)")


def missing_from_ukb_directory(ukb_subject_dir):
    """Check for missing files in a ukb subject directory.

    Parameters
    ----------
    ukb_subject_dir : :obj:`pathlib.Path`
        The path to the ukb subject directory.

    Returns
    -------
    missing_files : :obj:`list` of :obj:`str`
        A list of missing files.
    """
    dmri_dir = ukb_subject_dir / "DTI" / "dMRI" / "dMRI"

    # Files needed to run recon workflows
    required_files = [
        dmri_dir / "bvals",
        dmri_dir / "bvecs",
        dmri_dir / "data_ud.nii.gz",
        dmri_dir / "dti_FA.nii.gz",
        # The anatomical data is not strictly necessary for recon
        # anat_dir / "T1_brain.nii.gz",
        # anat_dir / "T1_brain_mask.nii.gz"
    ]

    return [str(fpath) for fpath in required_files if not fpath.exists()]


def find_ukb_directory(ukb_directory_list, subject_id):
    """Find a UKB directory for a given subject ID.

    Parameters
    ----------
    ukb_directory_list : :obj:`list` of :obj:`pathlib.Path`
        A list of ukb directories to search.
    subject_id : :obj:`str`
        The subject ID to search for.

    Returns
    -------
    ukb_directory : :obj:`pathlib.Path`
        The path to the ukb directory.
    """
    potential_directories = [
        subdir for subdir in ukb_directory_list if subdir.name.startswith(subject_id)
    ]

    # If nothing starts with the subject id, then we're out of luck
    if not potential_directories:
        raise Exception(f"No UKB directory available for {subject_id}")

    complete_dirs = []
    for potential_directory in potential_directories:
        missing_files = missing_from_ukb_directory(potential_directory)
        if not missing_files:
            complete_dirs.append(potential_directory)

    # Too many complete matches: ambiguous subject ID
    if len(complete_dirs) > 1:
        raise Exception(
            "Provide a more specific subject filter: More than 1 directories match "
            + subject_id
            + "\n"
            + "\n".join(map(str, complete_dirs))
        )

    # There were potential directories, but none were complete
    if not complete_dirs:
        error_report = "\n".join(
            [
                str(pdir.absolute())
                + " missing:\n    "
                + "\n    ".join(missing_from_ukb_directory(pdir))
                for pdir in potential_directories
            ]
        )
        raise Exception(f"No complete directories found for {subject_id}:\n{error_report}")

    return


def create_ukb_layout(ukb_dir, participant_label=None):
    """Find all valid ukb directories under ukb_dir.

    Parameters
    ----------
    ukb_dir : :obj:`pathlib.Path`
        The path to the ukb directory.
    participant_label : :obj:`list` of :obj:`str`
        A list of participant labels to search for.

    Returns
    -------
    ukb_layout : :obj:`list` of :obj:`dict`
        A list of dictionaries containing the subject ID, session ID, path to the ukb directory,
        and the path to the fake dwi file.
    """
    # find directories starting with a number. These are the candidate directories
    ukb_layout = []
    for potential_dir in Path(ukb_dir).iterdir():
        if participant_label and not potential_dir.name.startswith(participant_label):
            continue

        match = re.match(UKB_DIR_PATTERN, potential_dir.name)
        if not match:
            continue

        if missing_from_ukb_directory(potential_dir):
            continue

        subject, ses_major, ses_minor = match.groups()
        renamed_ses = "%02d%02d" % (int(ses_major), int(ses_minor))
        fake_dwi_file = (
            f"/bids/sub-{subject}/ses-{renamed_ses}/dwi/sub-{subject}_ses-{renamed_ses}_dwi.nii.gz"
        )
        ukb_layout.append(
            {
                "subject": subject,
                "session": renamed_ses,
                "path": potential_dir,
                "bids_dwi_file": fake_dwi_file,
            }
        )

    return ukb_layout


def ukb_dirname_to_bids(ukb_dir):
    """Convert a UKB directory name to a BIDS subject ID.

    Parameters
    ----------
    ukb_dir : :obj:`pathlib.Path`
        The path to the ukb directory.

    Returns
    -------
    bids_subject_id : :obj:`str`
        The BIDS subject ID and session ID, combined in a string.
    """
    ukb_path = Path(ukb_dir)
    match = re.match(UKB_DIR_PATTERN, ukb_path.name)
    subject, ses_major, ses_minor = match.groups()
    renamed_ses = "%02d%02d" % (int(ses_major), int(ses_minor))
    return f"sub-{subject}_ses-{renamed_ses}"


def collect_ukb_participants(ukb_layout, participant_label):
    """Collect all valid UKB participants.

    Parameters
    ----------
    ukb_layout : :obj:`list` of :obj:`dict`
        A list of dictionaries containing the subject ID, session ID, path to the ukb directory,
        and the path to the fake dwi file.
    participant_label : :obj:`list` of :obj:`str`
        A list of participant labels to search for.

    Returns
    -------
    participants : :obj:`list` of :obj:`str`
        A list of participant labels.
    """
    all_participants = set([spec["subject"] for spec in ukb_layout])

    # No --participant-label was set, return all
    if not participant_label:
        return sorted(all_participants)

    if isinstance(participant_label, str):
        participant_label = [participant_label]

    participant_label = set(participant_label)

    # Remove labels not found
    found_labels = sorted(participant_label & all_participants)
    requested_but_missing = sorted(participant_label - all_participants)

    if requested_but_missing:
        raise Exception(
            "Requested subjects [{}] do not have complete UKB directories under".format(
                ", ".join(requested_but_missing)
            )
        )
    if not found_labels:
        raise Exception("No complete UKB directories were found")

    return sorted(found_labels)
