#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Miscellaneous utility functions."""


def check_deps(workflow):
    from nipype.utils.filemanip import which

    return sorted(
        (node.interface.__class__.__name__, node.interface._cmd)
        for node in workflow._get_all_nodes()
        if (hasattr(node.interface, "_cmd") and which(node.interface._cmd.split()[0]) is None)
    )


def fix_multi_T1w_source_name(in_files):
    """Make up a generic source name when there are multiple T1s.

    >>> fix_multi_T1w_source_name([
    ...     '/path/to/sub-045_ses-test_T1w.nii.gz',
    ...     '/path/to/sub-045_ses-retest_T1w.nii.gz'])
    '/path/to/sub-045_T1w.nii.gz'
    """
    import os

    from nipype.utils.filemanip import filename_to_list

    base, in_file = os.path.split(filename_to_list(in_files)[0])
    subject_label = in_file.split("_", 1)[0].split("-")[1]
    return os.path.join(base, f"sub-{subject_label}_T1w.nii.gz")


def fix_multi_source_name(in_files, dwi_only, anatomical_contrast="T1w"):
    """Make up a generic source name when there are multiple source files.

    >>> fix_multi_source_name([
    ...     '/path/to/sub-045_ses-test_T1w.nii.gz',
    ...     '/path/to/sub-045_ses-retest_T1w.nii.gz'])
    '/path/to/sub-045_T1w.nii.gz'
    """
    import os

    from nipype.utils.filemanip import filename_to_list

    base, in_file = os.path.split(filename_to_list(in_files)[0])

    # Remove the session label
    base = os.path.abspath(base)
    folders = base.split(os.sep)
    folders = [f for f in folders if not f.startswith("ses-")]
    base = os.sep.join(folders)

    subject_label = in_file.split("_", 1)[0].split("-")[1]
    if dwi_only:
        anatomical_contrast = "dwi"
        base = base.replace("/dwi", "/anat")

    return os.path.join(base, f"sub-{subject_label}_{anatomical_contrast}.nii.gz")


def add_suffix(in_files, suffix):
    """Wrap nipype's fname_presuffix to conveniently just add a suffixfix.

    >>> add_suffix([
    ...     '/path/to/sub-045_ses-test_T1w.nii.gz',
    ...     '/path/to/sub-045_ses-retest_T1w.nii.gz'], '_test')
    'sub-045_ses-test_T1w_test.nii.gz'
    """
    import os.path as op

    from nipype.utils.filemanip import filename_to_list, fname_presuffix

    return op.basename(fname_presuffix(filename_to_list(in_files)[0], suffix=suffix))


def generate_interactive_report_summary(output_dir):
    """
    Gather the dwiqc values from the outputs in a
    """
    import json
    from pathlib import Path

    report_errors = []
    qc_report = {
        "report_type": "dwi_qc_report",
        "pipeline": "qsiprep",
        "pipeline_version": 0,
        "boilerplate": "",
        "metric_explanation": {
            "raw_dimension_x": "Number of x voxels in raw images",
            "raw_dimension_y": "Number of y voxels in raw images",
            "raw_dimension_z": "Number of z voxels in raw images",
            "raw_voxel_size_x": "Voxel size in x direction in raw images",
            "raw_voxel_size_y": "Voxel size in y direction in raw images",
            "raw_voxel_size_z": "Voxel size in z direction in raw images",
            "raw_max_b": "Maximum b-value in s/mm^2 in raw images",
            "raw_neighbor_corr": "Neighboring DWI Correlation (NDC) of raw images",
            "raw_num_bad_slices": "Number of bad slices in raw images (from DSI Studio)",
            "raw_num_directions": "Number of directions sampled in raw images",
            "t1_dimension_x": "Number of x voxels in preprocessed images",
            "t1_dimension_y": "Number of y voxels in preprocessed images",
            "t1_dimension_z": "Number of z voxels in preprocessed images",
            "t1_voxel_size_x": "Voxel size in x direction in preprocessed images",
            "t1_voxel_size_y": "Voxel size in y direction in preprocessed images",
            "t1_voxel_size_z": "Voxel size in z direction in preprocessed images",
            "t1_max_b": "Maximum b-value s/mm^2 in preprocessed images",
            "t1_neighbor_corr": "Neighboring DWI Correlation (NDC) of preprocessed images",
            "t1_num_bad_slices": "Number of bad slices in preprocessed images (from DSI Studio)",
            "t1_num_directions": "Number of directions sampled in preprocessed images",
            "mean_fd": "Mean framewise displacement from head motion",
            "max_fd": "Maximum framewise displacement from head motion",
            "max_rotation": "Maximum rotation from head motion",
            "max_translation": "Maximum translation from head motion",
            "max_rel_rotation": "Maximum rotation relative to the previous head position",
            "max_rel_translation": "Maximum translation relative to the previous head position",
            "t1_dice_distance": "Dice score for the overlap of the T1w-based brain mask "
            "and the b=0 ref mask",
        },
    }
    qc_values = []
    output_path = Path(output_dir)
    dwiqc_jsons = output_path.rglob("**/sub-*dwiqc.json")

    for qc_file in dwiqc_jsons:
        try:
            with open(qc_file, "r") as qc_json:
                dwi_qc = json.load(qc_json)["qc_scores"]
                dwi_qc["participant_id"] = dwi_qc.get("subject_id", "subject")
            qc_values.append(dwi_qc)
        except Exception:
            report_errors.append(1)

    errno = sum(report_errors)
    if errno:
        import logging

        logger = logging.getLogger("cli")
        logger.warning("Errors occurred while generating interactive report summary.")
    qc_report["subjects"] = qc_values
    with open(output_path / "dwiqc.json", "w") as project_qc:
        json.dump(qc_report, project_qc, indent=2)

    return errno


if __name__ == "__main__":
    pass
