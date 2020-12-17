#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import os.path as op
import json
from collections import defaultdict
import numpy as np
from nipype import logging
import pandas as pd
from nipype.utils.filemanip import fname_presuffix, split_filename
from .images import to_lps
from .reports import topup_selection_to_report
from nilearn.image import load_img, index_img, concat_imgs

LOGGER = logging.getLogger('nipype.interface')
CRITICAL_KEYS = ["PhaseEncodingDirection", "TotalReadoutTime", "EffectiveEchoSpacing"]


def _merge_metadata(metadatas):
    # Combine metadata from merged b=0 images
    if not metadatas:
        return {}

    merged_metadata = metadatas[0]
    for next_metadata in metadatas[1:]:
        for critical_key in CRITICAL_KEYS:
            current_value = merged_metadata.get(critical_key)
            next_value = next_metadata.get(critical_key)
            if not current_value == next_value:
                LOGGER.warning("%s inconsistent in fieldmaps: %s, %s", critical_key,
                               str(current_value), str(next_value))
    return merged_metadata


def read_nifti_sidecar(json_file):
    if not json_file.endswith(".json"):
        json_file = fname_presuffix(json_file, suffix='.json', use_ext=False)
        if not op.exists(json_file):
            raise Exception("No corresponding json file found")

    with open(json_file, "r") as f:
        metadata = json.load(f)
    pe_dir = metadata['PhaseEncodingDirection']
    slice_times = metadata.get("SliceTiming")
    trt = metadata.get("TotalReadoutTime")
    if trt is None:
        pass
    return {"PhaseEncodingDirection": pe_dir,
            "SliceTiming": slice_times,
            "TotalReadoutTime": trt}


acqp_lines = {
    "i": '1 0 0 %.6f',
    "j": '0 1 0 %.6f',
    "k": '0 0 1 %.6f',
    "i-": '-1 0 0 %.6f',
    "j-": '0 -1 0 %.6f',
    "k-": '0 0 -1 %.6f'}


def load_epi_dwi_fieldmaps(fmap_list, b0_threshold):
    """Creates a 4D image of b=0s from a list of input images.

    Parameters:
    -----------

    fmap_list: list
        List of paths to epi fieldmap images
    b0_threshold: int
        Maximum b value for an image to be considered a b=0

    Returns:
    --------

    concatenated_images: spatial image
        The b=0 volumes concatenated into a 4D image
    b0_indices: list
        List of the indices in the concatenated images that contain usable images
    original_files: list
        List of the original files where each b=0 image came from.

    """
    # Add in the rpe data, if it exists
    b0_indices = []
    original_files = []
    image_series = []

    for fmap_file in fmap_list:
        pth, fname, _ = split_filename(fmap_file)
        potential_bval_file = op.join(pth, fname) + ".bval"
        starting_index = len(original_files)
        fmap_img = load_img(fmap_file)
        image_series.append(fmap_img)
        num_images = 1 if fmap_img.ndim == 3 else fmap_img.shape[3]
        original_files += [fmap_file] * num_images

        # Which images are b=0 images?
        if op.exists(potential_bval_file):
            bvals = np.loadtxt(potential_bval_file)
            too_large = np.flatnonzero(bvals > b0_threshold)
            too_large_values = bvals[too_large]
            if too_large.size:
                LOGGER.warning("Excluding volumes %s from the %s because b=%s is greater than %d",
                               str(too_large), fmap_file, str(too_large_values), b0_threshold)
            _b0_indices = np.flatnonzero(bvals < b0_threshold) + starting_index
        else:
            _b0_indices = np.arange(num_images) + starting_index
        b0_indices += _b0_indices.tolist()

    concatenated_images = concat_imgs(image_series, auto_resample=True)
    return concatenated_images, b0_indices, original_files


def get_distortion_grouping(origin_file_list):
    """Discover which distortion groups are present, then assign each volume to a group.
    """
    unique_files = sorted(set(origin_file_list))
    unique_acqps = []
    line_lookup = {}
    for unique_dwi in unique_files:
        spec = read_nifti_sidecar(unique_dwi)
        spec_line = acqp_lines[spec['PhaseEncodingDirection']]
        acqp_line = spec_line % spec['TotalReadoutTime']
        if acqp_line not in unique_acqps:
            unique_acqps.append(acqp_line)
        line_lookup[unique_dwi] = unique_acqps.index(acqp_line) + 1

    group_numbers = [line_lookup[dwi_file] for dwi_file in origin_file_list]
    return unique_acqps, group_numbers


def eddy_inputs_from_dwi_files(origin_file_list, eddy_prefix):
    unique_acqps, group_numbers = get_distortion_grouping(origin_file_list)

    # Create the acqp file
    acqp_file = eddy_prefix + "acqp.txt"
    with open(acqp_file, "w") as f:
        f.write("\n".join(unique_acqps))

    # Create the index file
    index_file = eddy_prefix + "index.txt"
    with open(index_file, "w") as f:
        f.write(" ".join(map(str, group_numbers)))

    return acqp_file, index_file


def get_best_b0_topup_inputs_from(
        dwi_file, bval_file, b0_threshold, cwd, bids_origin_files, epi_fmaps=None,
        max_per_spec=3, topup_requested=False):

    """Create a datain spec and a slspec from a concatenated dwi series.

    Create inputs for TOPUP that come from data in ``dwi/`` and epi fieldmaps in ``fmap/``.
    The ``nii_file`` input may be the result of concatenating a number of scans with different
    distortions present. The original source of each volume in ``nii_file`` is listed in
    ``bids_origin_files``.

    The strategy is to select ``max_per_spec`` b=0 images from each distortion group.
    Here, distortion group uses the FSL definition of a phase encoding direction and
    total readout time, as specified in the datain file used by TOPUP (i.e. "0 -1 0 0.087").

    Parameters:
    ===========

        nii_file : str
            A 4D DWI Series
        b0_indices: array-like
            indices into nii_file that can be used by topup
        topup_prefix: str
            file prefix for topup inputs
        bids_origin_files: list
            A list with the original bids file of each image in ``nii_file``. This is
            necessary because merging may have happened earlier in the pipeline
        epi_fmaps:
            A list of images from the fmaps/ directory.
        max_per_spec: int
            The maximum number of b=0 images to extract from a PE direction / image set

    """

    # Start with the DWI file. Determine which images are b=0 and where they came from
    dwi_b0_df = split_into_b0s_and_origins(b0_threshold, bids_origin_files, dwi_file,
                                           cwd, bval_file=bval_file, b0_indices=None)

    # If there are epi fieldmaps, add them to the table
    if epi_fmaps:
        epi_4d, epi_b0_indices, epi_original_files = load_epi_dwi_fieldmaps(
                epi_fmaps, b0_threshold)
        epi_b0_df = split_into_b0s_and_origins(b0_threshold, epi_original_files, epi_4d, cwd,
                                               bval_file=None, b0_indices=epi_b0_indices)
        dwi_b0_df = pd.concat([dwi_b0_df, epi_b0_df], axis=0, ignore_index=True)

    unique_bids_files = dwi_b0_df.bids_origin_file.unique().tolist()
    spec_lookup = {}
    slicetime_lookup = {}
    for unique_bids_file in unique_bids_files:
        spec = read_nifti_sidecar(unique_bids_file)
        spec_line = acqp_lines[spec['PhaseEncodingDirection']]
        spec_lookup[unique_bids_file] = spec_line % spec['TotalReadoutTime']
        slicetime_lookup[unique_bids_file] = spec['SliceTiming']

    # Group the b=0 images by their spec
    dwi_b0_df["fsl_spec"] = dwi_b0_df["bids_origin_file"].map(spec_lookup)
    # Write the datain text file and make sure it's usable if it's needed
    if len(dwi_b0_df["fsl_spec"].unique()) < 2 and topup_requested:
        print(dwi_b0_df["fsl_spec"])
        raise Exception("Unable to run TOPUP: not enough distortion groups. "
                        "Check \"IntendedFor\" fields or consider using --ignore fieldmaps.")
    spec_groups = dwi_b0_df.groupby("fsl_spec")
    max_per_spec = min(max_per_spec, min(spec_groups.apply(len)))

    # Calculate the "quality" of each image:
    dwi_b0_df["qc_score"] = spec_groups["nii_3d_files"].transform(calculate_best_b0s)
    dwi_b0_df["qc_rank"] = spec_groups["qc_score"].transform(np.argsort)

    # Select only the top
    dwi_b0_df["selected_for_sdc"] = dwi_b0_df["qc_rank"] < max_per_spec
    sdc_selections = dwi_b0_df[dwi_b0_df["selected_for_sdc"]].reset_index()
    # Make sure the first image in topup imain has the same distortion as the
    # first b=0 volume in the eddy inputs
    sdc_selections['same_as_first'] = \
        sdc_selections['fsl_spec'] == dwi_b0_df.loc[0, 'fsl_spec']
    sdc_selections.sort_values(by=["same_as_first", 'index'],
                               ascending=[False, True], inplace=True)

    imain_output = cwd + "/topup_imain.nii.gz"
    imain_img = concat_imgs(
        [to_lps(img, new_axcodes=('L', 'A', 'S')) for img in
         sdc_selections["nii_3d_files"]], auto_resample=True)
    imain_img.to_filename(imain_output)

    datain_file = cwd + "/topup_datain.txt"
    with open(datain_file, "w") as f:
        f.write("\n".join(sdc_selections['fsl_spec']))

    b0_csv = cwd + "/b0_selection_info.csv"
    dwi_b0_df.drop("nii_3d_files", 1).to_csv(b0_csv, index=False)

    # get out reference images from the topup and eddy data
    topup_reg_file = cwd + "/topup_reg_image.nii.gz"
    index_img(imain_output, 0).to_filename(topup_reg_file)

    topup_report = topup_selection_to_report(
        np.flatnonzero(dwi_b0_df["selected_for_sdc"]),
        dwi_b0_df["bids_origin_file"],
        spec_lookup,
        image_source="data")
    return datain_file, imain_output, topup_report, b0_csv, \
        topup_reg_file, dwi_b0_df.loc[0, 'nii_3d_files']


def relative_b0_index(b0_indices, original_files):
    """Find the index of each b=0 image in its original imaging series

    >>> b0_indices = [0, 7, 11, 15, 17, 30, 37, 41, 45]
    >>> original_files = ["sub-1_dir-AP_dwi.nii.gz"] * 30 + ["sub-1_dir-PA_dwi.nii.gz"] * 30
    >>> print(
    ... relative_b0_index(b0_indices,
    ...                   original_files))  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    [0, 7, 11, 15, 17, 0, 7, 11, 15]


    Or

    >>> original_files = ["sub-1_dir-AP_run-1_dwi.nii.gz"] * 15 + [
    ...                   "sub-1_dir-AP_run-2_dwi.nii.gz"] * 15 + [
    ...                   "sub-1_dir-PA_dwi.nii.gz"] * 30
    >>> print(relative_b0_index(b0_indices, original_files))
    [0, 7, 11, 0, 2, 0, 7, 11, 15]

    """
    image_counts = defaultdict(int)
    ordered_files = []
    for original_file in original_files:
        if original_file not in image_counts:
            ordered_files.append(original_file)
        image_counts[original_file] += 1
    offsets = [0]
    for original_file in ordered_files:
        offsets.append(offsets[-1] + image_counts[original_file])
    image_offsets = dict(zip(ordered_files, offsets))

    original_indices = []
    for b0_index in b0_indices:
        original_file = original_files[b0_index]
        original_index = b0_index - image_offsets[original_file]
        original_indices.append(original_index)

    return original_indices


def calculate_best_b0s(b0_list, radius=4):
    import SimpleITK as sitk
    imgs = [sitk.ReadImage(fname, sitk.sitkFloat64) for fname in b0_list]
    no_reg = sitk.ImageRegistrationMethod()
    no_reg.SetMetricSamplingStrategy(no_reg.NONE)
    no_reg.SetMetricAsCorrelation()
    pairwise = np.eye(len(b0_list))
    for id0, id1 in zip(*np.triu_indices(len(b0_list), 1)):
        pairwise[id0, id1] = no_reg.MetricEvaluate(imgs[id0], imgs[id1])
    pairwise = pairwise + pairwise.T
    return pairwise.mean(0)


def split_into_b0s_and_origins(b0_threshold, original_files, img_file, cwd,
                               b0_indices=None, bval_file=None):
    """ """
    b0_bids_files = []
    b0_nii_files = []
    full_img = load_img(img_file)

    if b0_indices is None:
        if bval_file is not None:
            # Start with the DWI file. Determine which images are b=0
            bvals = np.loadtxt(bval_file)
            b0_indices = np.flatnonzero(bvals < b0_threshold)
            if not b0_indices.size:
                raise RuntimeError("No b=0 images available.")
        else:
            b0_indices = np.array([0]) if full_img.ndim < 4 else \
                np.arange(full_img.shape[3], dtype=np.int)

    relative_indices = relative_b0_index(b0_indices, original_files)

    # find the original files accompanying each b=0
    for b0_index, original_index in zip(b0_indices, relative_indices):
        original_file = original_files[b0_index]
        b0_bids_files.append(original_file)
        new_b0_path = fname_presuffix(original_file, suffix="_b0-%02d" % original_index,
                                      newpath=cwd)
        index_img(full_img, b0_index).to_filename(new_b0_path)
        b0_nii_files.append(new_b0_path)

    return pd.DataFrame({"nii_3d_files": b0_nii_files,
                         "bids_origin_file": b0_bids_files,
                         "original_volume": relative_indices})
