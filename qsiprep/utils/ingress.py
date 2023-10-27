#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Utilities to handle data from other preprocessing pipelines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

"""
import re
import os
import sys
import json
import warnings
import numpy as np
import nibabel as nb

UKB_DIR_PATTERN = re.compile("(\d+)_(\d+)_(\d+)")

def missing_from_ukb_directory(ukb_sujbect_dir):
    """Check for missing files in a ukb subject directory."""
    dmri_dir = ukb_sujbect_dir / "DTI" / "dMRI" / "dMRI"
    anat_dir = ukb_sujbect_dir / "T1"

    # Files needed to run recon workflows
    required_files = [
        dmri_dir / "bvals",
        dmri_dir / "bvecs",
        dmri_dir / "data_ud.nii.gz",
        anat_dir / "T1_brain.nii.gz",
        anat_dir / "T1_brain_mask.nii.gz"
    ]

    return [str(fpath) for fpath in required_files if not fpath.exists()]


def find_ukb_directory(ukb_directory_list, subject_id):

    potential_directories = [subdir for subdir in ukb_directory_list if subdir.name.startswith(subject_id)]

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
            "Provide a more specific subject filter: More than 1 directories match " \
            + subject_id + "\n" + "\n".join(map(str, complete_dirs)))

    # There were potential directories, but none were complete
    if not complete_dirs:
        error_report = "\n".join([
            str(pdir.absolute()) + " missing:\n    " + "\n    ".join(missing_from_ukb_directory(pdir))
            for pdir in potential_directories])
        raise Exception(f"No complete directories found for {subject_id}:\n{error_report}")

    return


def create_ukb_layout(ukb_dir, participant_label=None):
    """
    Find all valid ukb directories under ukb_dir.

    returns a list of directories and their subject/session info
    """

    # find directories starting with a number. These are the candidate directories
    ukb_layout = []
    for potential_dir in ukb_dir.iterdir():
        if participant_label and not potential_dir.name.startswith(participant_label):
            continue
        match = re.match(UKB_DIR_PATTERN, potential_dir.name)
        if not match:
            continue
        if missing_from_ukb_directory(potential_dir):
            continue

        subject, ses_major, ses_minor = match.groups()
        renamed_ses = "%02d%02d" % (int(ses_major), int(ses_minor))
        fake_dwi_file = f"/bids/sub-{subject}/ses-{renamed_ses}/dwi/sub-{subject}_ses-{renamed_ses}_dwi.nii.gz"
        ukb_layout.append(
            {"subject": subject,
             "session": renamed_ses,
             "path": potential_dir,
             "dwi_file": fake_dwi_file}
        )

    return ukb_layout


def collect_ukb_participants(ukb_layout, participant_label):

    all_participants = set([spec['subject'] for spec in ukb_layout])

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
        raise Exception('Requested subjects [{}] do not have complete UKB directories under'.format(
            ', '.join(requested_but_missing)))
    if not found_labels:
        raise Exception('No complete UKB directories were found')


    return sorted(found_labels)
