"""Handle merging and spliting of DSI files."""

import json
import os.path as op

import nibabel as nb
import numpy as np
import pandas as pd
from nilearn.image import concat_imgs, index_img, iter_img, load_img, math_img
from nipype import logging
from nipype.interfaces import ants
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    InputMultiObject,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.utils.filemanip import fname_presuffix

from ..workflows.dwi.util import _get_concatenated_bids_name
from .fmap import get_distortion_grouping

LOGGER = logging.getLogger("nipype.workflow")
MAX_COMBINED_SCANS = 100


class MergeDWIsInputSpec(BaseInterfaceInputSpec):
    dwi_files = InputMultiObject(File(), mandatory=True, desc="list of dwi files")
    bids_dwi_files = InputMultiObject(
        File(), mandatory=True, desc="list of original (BIDS) dwi files"
    )
    bval_files = InputMultiObject(File(exists=True), mandatory=True, desc="list of bval files")
    bvec_files = InputMultiObject(File(exists=True), mandatory=True, desc="list of bvec files")
    b0_threshold = traits.Int(100, usedefault=True, desc="Maximum b=0 value")
    denoising_confounds = InputMultiObject(
        File(exists=True, desc="list of confound files associated with each input dwi")
    )
    harmonize_b0_intensities = traits.Bool(
        True, usedefault=True, desc="Force scans to have the same mean b=0 intensity"
    )
    raw_concatenated_files = InputMultiObject(
        File(), mandatory=False, desc="list of raw concatenated images"
    )
    b0_refs = InputMultiObject(File(), mandatory=False, desc="list of b=0 reference images")
    carpetplot_data = InputMultiObject(
        File(exists=True), mandatory=False, desc="list of carpetplot_data files"
    )
    scan_metadata = traits.Dict(desc="Dict of metadata for the to-be-combined scans")


class MergeDWIsOutputSpec(TraitedSpec):
    out_dwi = File(desc="the merged dwi image")
    out_bval = File(desc="the merged bval file")
    out_bvec = File(desc="the merged bvec file")
    original_images = traits.List()
    merged_metadata = File(exists=True)
    merged_denoising_confounds = File(exists=True)
    merged_b0_ref = File(exists=True)
    merged_raw_dwi = File(exists=True, mandatory=False)
    merged_raw_bvec = File(exists=True, mandatory=False)
    merged_carpetplot_data = File(exists=True)


class MergeDWIs(SimpleInterface):
    input_spec = MergeDWIsInputSpec
    output_spec = MergeDWIsOutputSpec

    def _run_interface(self, runtime):
        bvals = self.inputs.bval_files
        bvecs = self.inputs.bvec_files
        num_dwis = len(self.inputs.dwi_files)

        to_concat, b0_means, corrections = harmonize_b0s(
            self.inputs.dwi_files,
            bvals,
            self.inputs.b0_threshold,
            self.inputs.harmonize_b0_intensities,
        )

        # Create a merged metadata json file for
        if isdefined(self.inputs.scan_metadata):
            combined_metadata = combine_metadata(
                self.inputs.bids_dwi_files,
                self.inputs.scan_metadata,
            )
            merged_metadata_file = op.join(runtime.cwd, "merged_metadata.json")
            with open(merged_metadata_file, "w") as merged_meta_f:
                json.dump(combined_metadata, merged_meta_f, sort_keys=True, indent=4)
            self._results["merged_metadata"] = merged_metadata_file

        # Get basic qc / provenance per volume
        provenance_df = create_provenance_dataframe(
            self.inputs.bids_dwi_files, to_concat, b0_means, corrections
        )

        # Collect the confounds
        if isdefined(self.inputs.denoising_confounds):
            confounds = [pd.read_csv(fname) for fname in self.inputs.denoising_confounds]
            _confounds_df = pd.concat(confounds, axis=0, ignore_index=True)
            confounds_df = pd.concat([provenance_df, _confounds_df], axis=1, ignore_index=False)
        else:
            confounds_df = provenance_df

        # Load the gradient information
        all_bvals = combined_bval_array(self.inputs.bval_files)
        all_bvecs = combined_bvec_array(self.inputs.bvec_files)
        confounds_df["original_bval"] = all_bvals
        confounds_df["original_bx"] = all_bvecs[0]
        confounds_df["original_by"] = all_bvecs[1]
        confounds_df["original_bz"] = all_bvecs[2]
        confounds_df = confounds_df.loc[:, ~confounds_df.columns.duplicated()]

        # Concatenate the gradient information
        if num_dwis > 1:
            merged_output = _get_concatenated_bids_name(
                {"dwi_series": self.inputs.dwi_files, "fieldmap_info": {"suffix": None}}
            )
            merged_fname = op.join(runtime.cwd, merged_output + "_merged.nii.gz")
            out_bval = fname_presuffix(
                merged_fname, suffix=".bval", use_ext=False, newpath=runtime.cwd
            )
            out_bvec = fname_presuffix(
                merged_fname, suffix=".bvec", use_ext=False, newpath=runtime.cwd
            )
        else:
            merged_fname = self.inputs.dwi_files[0]
            out_bval = bvals[0]
            out_bvec = bvecs[0]

        merged_confounds = fname_presuffix(
            merged_fname, suffix="_confounds.csv", use_ext=False, newpath=runtime.cwd
        )
        confounds_df = confounds_df.drop("Unnamed: 0", axis=1, errors="ignore")
        confounds_df.to_csv(merged_confounds, index=False)

        self._results["merged_denoising_confounds"] = merged_confounds
        self._results["original_images"] = confounds_df["original_file"].tolist()
        self._results["out_dwi"] = merged_fname
        self._results["out_bval"] = out_bval
        self._results["out_bvec"] = out_bvec

        # If one and only one carpetplot data was specified, add it to outputs
        if len(self.inputs.carpetplot_data) > 1:
            raise NotImplementedError("Can't handle multiple carpetplots in merging")
        if len(self.inputs.carpetplot_data) == 1:
            self._results["merged_carpetplot_data"] = self.inputs.carpetplot_data[0]

        if num_dwis == 1:
            return runtime

        # Write the merged gradients
        combine_bvals(bvals, output_file=out_bval)
        combine_bvecs(bvecs, output_file=out_bvec)
        # Concatenate into a single file
        merged_nii = concat_imgs(to_concat, auto_resample=True)
        # Remove any negative values introduced during interpolation (if it occurrs)
        pos_merged_nii = math_img("np.clip(img, 0, None)", img=merged_nii)
        pos_merged_nii.to_filename(merged_fname)

        return runtime


def combine_metadata(scan_list, metadata_dict, merge_method="first"):
    """Create a merged metadata dictionary.

    Most importantly, combine the slice timings in some way.

    Parameters
    ----------
    scan_list: list
        List of BIDS inputs in the order in which they'll be concatenated
    medadata_dict: dict
        Mapping keys (values in ``scan_list``) to BIDS metadata dictionaries
    merge_method: str
        How to combine the metadata when multiple scans are being concatenated.
        If "first" the metadata from the first scan is selected. Someday other
        methods like "average" may be added.

    Returns
    -------
    metadata: dict
        A BIDS metadata dictionary

    """
    if merge_method == "first":
        return metadata_dict[scan_list[0]]
    raise NotImplementedError(f"merge_method '{merge_method}' is not implemented")


class AveragePEPairsInputSpec(MergeDWIsInputSpec):
    original_bvec_files = InputMultiObject(
        File(exists=True), mandatory=True, desc="list of original bvec files"
    )
    carpetplot_data = InputMultiObject(
        File(exists=True), mandatory=True, desc="list of carpetplot_data files"
    )
    verbose = traits.Bool(False, usedefault=True)


class AveragePEPairsOutputSpec(MergeDWIsOutputSpec):
    merged_raw_concatenated = File(exists=True)


class AveragePEPairs(SimpleInterface):
    input_spec = AveragePEPairsInputSpec
    output_spec = AveragePEPairsOutputSpec

    def _run_interface(self, runtime):
        distortion_groups, assignments = get_distortion_grouping(self.inputs.bids_dwi_files)
        num_distortion_groups = len(distortion_groups)
        if not num_distortion_groups == 2:
            raise Exception(
                "Unable to merge using strategy 'average': exactly"
                " two distortion groups must be present in data."
                " Found %d" % num_distortion_groups
            )

        # Get the gradient info for each PE group
        original_bvecs = combined_bvec_array(self.inputs.original_bvec_files)
        rotated_bvecs = combined_bvec_array(self.inputs.bvec_files)
        bvals = combined_bval_array(self.inputs.bval_files)

        # Find which images should be averaged together in the o
        # Also, average the carpetplot matrices and motion params
        image_pairs, averaged_raw_bvec = find_image_pairs(original_bvecs, bvals, assignments)
        (
            combined_images,
            combined_raw_images,
            combined_bvals,
            combined_bvecs,
            error_report,
            avg_carpetplot,
        ) = average_image_pairs(
            image_pairs,
            self.inputs.dwi_files,
            rotated_bvecs,
            bvals,
            self.inputs.denoising_confounds,
            self.inputs.raw_concatenated_files,
            self.inputs.carpetplot_data,
            verbose=self.inputs.verbose,
        )

        # Save the averaged outputs
        out_dwi_path = op.join(runtime.cwd, "averaged_pairs.nii.gz")
        combined_images.to_filename(out_dwi_path)
        self._results["out_dwi"] = out_dwi_path
        out_bval_path = op.join(runtime.cwd, "averaged_pairs.bval")
        self._results["out_bval"] = combine_bvals(combined_bvals, out_bval_path)
        out_bvec_path = op.join(runtime.cwd, "averaged_pairs.bvec")
        self._results["out_bvec"] = combine_bvecs(combined_bvecs, out_bvec_path)
        out_confounds_path = op.join(runtime.cwd, "averaged_pairs_confounds.tsv")
        error_report.to_csv(out_confounds_path, index=False, sep="\t")
        self._results["merged_denoising_confounds"] = out_confounds_path
        self._results["original_images"] = self.inputs.bids_dwi_files

        # Write the merged carpetplot data
        out_carpetplot_path = op.join(runtime.cwd, "merged_carpetplot.json")
        with open(out_carpetplot_path, "w") as carpet_f:
            json.dump(avg_carpetplot, carpet_f)
        self._results["merged_carpetplot_data"] = out_carpetplot_path

        # write the averaged raw data
        out_raw_concatenated = op.join(runtime.cwd, "merged_raw.nii.gz")
        self._results["merged_raw_dwi"] = out_raw_concatenated
        combined_raw_images.to_filename(out_raw_concatenated)
        out_raw_bvec = op.join(runtime.cwd, "merged_raw.bvec")
        self._results["merged_raw_bvec"] = combine_bvecs(averaged_raw_bvec, out_raw_bvec)

        # Make a new b=0 template
        b0_indices = np.flatnonzero(bvals < self.inputs.b0_threshold)
        b0_ref = ants.AverageImages(
            dimension=3, normalize=True, images=[self.inputs.dwi_files[idx] for idx in b0_indices]
        )
        result = b0_ref.run()
        self._results["merged_b0_ref"] = result.outputs.output_average_image

        return runtime


class _SplitResampledDWIsInputSpec(BaseInterfaceInputSpec):
    dwi_file = File(exists=True, mandatory=True)
    bval_file = File(exists=True, mandatory=True)
    bvec_file = File(exists=True, mandatory=True)
    confounds = File(exists=True, mandatory=True)
    n_images = traits.Int(1)


class _SplitResampledDWIsOutputSpec(TraitedSpec):
    pass


# Add slots for the possibly
for subscan in np.arange(MAX_COMBINED_SCANS) + 1:
    _SplitResampledDWIsOutputSpec.add_class_trait("dwi_file_%d" % subscan, File(exists=True))
    _SplitResampledDWIsOutputSpec.add_class_trait("bval_file_%d" % subscan, File(exists=True))
    _SplitResampledDWIsOutputSpec.add_class_trait("bvec_file_%d" % subscan, File(exists=True))
    _SplitResampledDWIsOutputSpec.add_class_trait("source_file_%d" % subscan, traits.Str())


class SplitResampledDWIs(SimpleInterface):
    input_spec = _SplitResampledDWIsInputSpec
    output_spec = _SplitResampledDWIsOutputSpec

    def _run_interface(self, runtime):
        # Load the confounds
        confounds_df = pd.read_csv(self.inputs.confounds, sep="\t")
        original_files = confounds_df["original_file"].unique().tolist()
        if not len(original_files) == self.inputs.n_images:
            raise Exception(
                "Found %d files in confounds file, but expected %d"
                % (len(original_files), self.inputs.n_images)
            )
        resampled_img = load_img(self.inputs.dwi_file)
        for file_num, original_file in enumerate(original_files, start=1):
            image_indices = np.flatnonzero(
                (confounds_df["original_file"] == original_file).to_numpy()
            )
            dwi_subfile = fname_presuffix(
                original_file,
                prefix="resampled_",
                suffix=".nii.gz",
                use_ext=False,
                newpath=runtime.cwd,
            )
            bval_subfile = dwi_subfile.replace(".nii.gz", ".bval")
            bvec_subfile = dwi_subfile.replace(".nii.gz", ".bvec")
            index_img(resampled_img, image_indices).to_filename(dwi_subfile)
            subset_bvals(self.inputs.bval_file, image_indices, bval_subfile)
            subset_bvecs(self.inputs.bvec_file, image_indices, bvec_subfile)

            self._results["dwi_file_%d" % file_num] = dwi_subfile
            self._results["bval_file_%d" % file_num] = bval_subfile
            self._results["bvec_file_%d" % file_num] = bvec_subfile
            self._results["source_file_%d" % file_num] = original_file
        return runtime


class _MergeFinalConfoundsInputSpec(BaseInterfaceInputSpec):
    confounds = File(exists=True, mandatory=True)
    bias_correction_confounds = InputMultiObject(File(exists=True), mandatory=False)
    patch2self_correction_confounds = File(exists=True, mandatory=False)


class _MergeFinalConfoundsOutputSpec(TraitedSpec):
    confounds = File(exists=True)


class MergeFinalConfounds(SimpleInterface):
    input_spec = _MergeFinalConfoundsInputSpec
    output_spec = _MergeFinalConfoundsOutputSpec

    def _run_interface(self, runtime):

        to_concat_horizontally = []
        # New confounds from bias correction
        if isdefined(self.inputs.bias_correction_confounds):
            # There may be multuple files that need to be vertically stacked
            biascorrection_df = pd.concat(
                [pd.read_csv(bc_csv) for bc_csv in self.inputs.bias_correction_confounds],
                axis=0,
                ignore_index=True,
            )
            to_concat_horizontally.append(biascorrection_df)
        # New confounds from patch2self
        if isdefined(self.inputs.patch2self_correction_confounds):
            to_concat_horizontally.append(pd.read_csv(self.inputs.patch2self_correction_confounds))

        # If we have new ones, append the columns, prefixed by "final_"
        if to_concat_horizontally:
            new_confounds_file = fname_presuffix(
                self.inputs.confounds, newpath=runtime.cwd, prefix="final_"
            )
            original_confounds = pd.read_csv(self.inputs.confounds, sep="\t")
            extra_confounds = pd.concat(to_concat_horizontally, axis=1)
            extra_confounds.columns = ["final_" + col for col in extra_confounds.columns.tolist()]
            final_confounds = pd.concat([original_confounds, extra_confounds], axis=1)
            final_confounds.to_csv(new_confounds_file, sep="\t", index=False)
            self._results["confounds"] = new_confounds_file
        else:
            self._results["confounds"] = self.inputs.confounds

        return runtime


def find_image_pairs(original_bvecs, bvals, assignments):
    assignments = np.array(assignments)
    group1_mask = assignments == 1
    group2_mask = assignments == 2
    image_nums = np.arange(len(assignments))
    group1 = {
        "bvals": bvals[group1_mask],
        "original_bvecs": original_bvecs[:, group1_mask],
        "indices": image_nums[group1_mask],
    }
    group2 = {
        "bvals": bvals[group2_mask],
        "original_bvecs": original_bvecs[:, group2_mask],
        "indices": image_nums[group2_mask],
    }

    # If this is HCP-style, the bvals and bvecs will match directly
    if not group2["bvals"].shape == group1["bvals"].shape:
        raise Exception("Unable to perform HCP-style merge, different numbers of images")
    if np.allclose(group2["bvals"], group1["bvals"], atol=50) and np.allclose(
        group2["original_bvecs"], group1["original_bvecs"], atol=0.0001
    ):
        pairs = list(zip(group1["indices"], group2["indices"]))
        bvecs = group1["original_bvecs"]
    else:
        raise Exception("Bvecs do not match - ensure matching bvecs")

    return pairs, bvecs


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """Returns the angle in degrees between vectors 'v1' and 'v2'::
    >>> angle_between((1, 0, 0), (0, 1, 0))
    90.0
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    180.0
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) * 180 / np.pi


def average_image_pairs(
    image_pairs,
    image_paths,
    rotated_bvecs,
    bvals,
    confounds_tsvs,
    raw_concatenated_files,
    carpetplots,
    verbose=False,
):
    """Create 4D series of averaged images, gradients, and confounds"""
    averaged_images = []
    new_bvecs = []
    confounds = pd.concat([pd.read_csv(fname, delimiter="\t") for fname in confounds_tsvs])
    merged_confounds = []
    merged_bvals = []

    # Load the raw concatenated images for qc
    raw_concatenated_img = concat_imgs(raw_concatenated_files)
    raw_averaged_images = []

    confounds1_rename = {col: col + "_1" for col in confounds.columns}
    confounds2_rename = {col: col + "_2" for col in confounds.columns}
    for index1, index2 in image_pairs:
        confounds1 = confounds.iloc[index1].copy().rename(confounds1_rename)
        confounds2 = confounds.iloc[index2].copy().rename(confounds2_rename)
        # Make a single row containing both 1 and 2
        confounds_both = confounds1.append(confounds2)
        averaged_images.append(math_img("(a+b)/2", a=image_paths[index1], b=image_paths[index2]))
        raw_averaged_images.append(
            math_img("(a[..., %d] + a[..., %d]) / 2" % (index1, index2), a=raw_concatenated_img)
        )

        new_bval = (bvals[index1] + bvals[index2]) / 2.0
        merged_bvals.append(new_bval)
        rotated1 = rotated_bvecs[:, index1]
        rotated2 = rotated_bvecs[:, index2]
        new_bvec, bvec_error = average_bvec(rotated1, rotated2)
        new_bvecs.append(new_bvec)

        confounds_both["vec_averaging_error"] = bvec_error
        confounds_both["rotated_grad_x_1"] = rotated1[0]
        confounds_both["rotated_grad_y_1"] = rotated1[1]
        confounds_both["rotated_grad_z_1"] = rotated1[2]
        confounds_both["rotated_grad_x_2"] = rotated2[0]
        confounds_both["rotated_grad_y_2"] = rotated2[1]
        confounds_both["rotated_grad_z_2"] = rotated2[2]
        confounds_both["grad_x"] = new_bvec[0]
        confounds_both["grad_y"] = new_bvec[1]
        confounds_both["grad_z"] = new_bvec[2]
        confounds_both["bval"] = new_bval
        merged_confounds.append(confounds_both)
        if verbose:
            print(
                "%d: %d [%.4fdeg error]\n\t%d (%.4f %.4f %.4f)"
                % (index1, index2, bvec_error, new_bval, new_bvec[0], new_bvec[1], new_bvec[2])
            )

    # Make columns that can be used in the interactive report
    averaged_confounds = pd.DataFrame(merged_confounds)
    needed_for_interactive_report = [
        "trans_x",
        "trans_y",
        "trans_z",
        "rot_x",
        "rot_y",
        "rot_z",
        "framewise_displacement",
    ]
    for key in needed_for_interactive_report:
        confs1, confs2 = averaged_confounds[[key + "_1", key + "_2"]].to_numpy().T
        averaged_confounds[key] = get_worst(confs1, confs2)

    # Original file is actually two files!
    averaged_confounds["original_file"] = averaged_confounds[
        ["original_file_1", "original_file_2"]
    ].agg("+".join, axis=1)

    # Get the averaged carpetplot data for the interactive report
    averaged_carpetplot = average_carpetplots(carpetplots, np.array(image_pairs))
    return (
        concat_imgs(averaged_images),
        concat_imgs(raw_averaged_images),
        np.array(merged_bvals),
        np.array(new_bvecs),
        averaged_confounds,
        averaged_carpetplot,
    )


def get_worst(values1, values2):
    """finds the highest magnitude value per index in values1, values2"""
    values = np.column_stack([values1, values2])
    highest_index = np.argmax(np.abs(values), axis=1)
    return values[np.arange(values.shape[0]), highest_index]


def average_carpetplots(carpet_list, image_pairs):
    """Averages carpetplot data for display when pe pairs are averaged.

    Reminder: incoming data is a dict of
    {"carpetplot": [[one image's slice scores],
                    [next image's slice scores],
                    ...
                    [last image's slice scores]]}
    and the image_pairs should be a n x 2 matrix where the columns
    are the first image index and second image index.

    """
    if not isinstance(carpet_list, list) and len(carpet_list) == 1:
        raise Exception("Not implemented for SHORELine")
    carpet_path = carpet_list[0]
    with open(carpet_path, "r") as carpet_f:
        carpet_dict = json.load(carpet_f)
    carpet_data = np.array(carpet_dict["carpetplot"])
    worst_rows = []
    for index1, index2 in image_pairs:
        worst_rows.append(get_worst(carpet_data[index1], carpet_data[index2]).tolist())
    return {"carpetplot": worst_rows}


def average_bvec(bvec1, bvec2):
    # return straight away if the bvecs are identical
    # This prevents comparison of zero vectors
    if (bvec1 == bvec2).all():
        return np.copy(bvec1), 0.0

    bvec_diff = angle_between(bvec1, bvec2)

    mean_bvec_plus = (bvec1 + bvec2) / 2.0
    mean_bvec_plus = mean_bvec_plus / np.linalg.norm(mean_bvec_plus)
    mean_bvec_minus = (bvec1 - bvec2) / 2.0
    mean_bvec_minus = mean_bvec_minus / np.linalg.norm(mean_bvec_minus)

    if angle_between(bvec1, mean_bvec_plus) < angle_between(
        bvec1, mean_bvec_minus
    ) and angle_between(bvec2, mean_bvec_plus) < angle_between(bvec2, mean_bvec_minus):
        return mean_bvec_plus, bvec_diff
    if angle_between(bvec1, mean_bvec_plus) > angle_between(
        bvec1, mean_bvec_minus
    ) and angle_between(bvec2, mean_bvec_plus) < angle_between(bvec2, mean_bvec_minus):
        return mean_bvec_minus, bvec_diff
    LOGGER.warning("Ambiguous direcions of vectors: assuming plus")
    return mean_bvec_plus, bvec_diff


class StackConfoundsInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiObject(File(exists=True), mandatory=True)
    axis = traits.Enum(0, 1, default=0, usedefault=True)
    out_file = File()


class StackConfoundsOutputSpec(TraitedSpec):
    confounds_file = File(desc="the stacked confound data")


class StackConfounds(SimpleInterface):
    input_spec = StackConfoundsInputSpec
    output_spec = StackConfoundsOutputSpec

    def _run_interface(self, runtime):
        if not self.inputs.in_files:
            return runtime
        dfs = [pd.read_csv(fname) for fname in self.inputs.in_files]
        stacked = pd.concat(dfs, axis=self.inputs.axis, ignore_index=self.inputs.axis == 0)
        out_file = op.join(runtime.cwd, "confounds.csv")
        stacked = stacked.drop("Unnamed: 0", axis=1, errors="ignore")
        stacked.to_csv(out_file)
        self._results["confounds_file"] = out_file
        return runtime


def subset_bvals(bval_file, indices, out_bval_file):
    original_bvals = np.loadtxt(bval_file)
    bval_subset = original_bvals[indices]
    np.savetxt(out_bval_file, bval_subset, fmt=str("%i"))


def subset_bvecs(bvec_file, indices, out_bvec_file):
    original_bvecs = np.loadtxt(bvec_file)
    bvec_subset = original_bvecs[:, indices]
    np.savetxt(out_bvec_file, bvec_subset, fmt=str("%.8f"))


def combined_bval_array(bval_files):
    collected_vals = []
    for bval_file in bval_files:
        if isinstance(bval_file, str):
            collected_vals.append(np.atleast_1d(np.loadtxt(bval_file)))
        else:
            collected_vals.append(np.atleast_1d(bval_file))
    return np.concatenate(collected_vals)


def combine_bvals(bvals, output_file="restacked.bval"):
    """Load, merge and save fsl-style bvals files."""
    final_bvals = combined_bval_array(bvals)
    np.savetxt(output_file, final_bvals, fmt=str("%i"))
    return op.abspath(output_file)


def combined_bvec_array(bvec_files):
    collected_vecs = []
    for bvec_file in bvec_files:
        if isinstance(bvec_file, str):
            collected_vecs.append(np.loadtxt(bvec_file))
        else:
            collected_vecs.append(bvec_file)
    return np.column_stack(collected_vecs)


def combine_bvecs(bvecs, output_file="restacked.bvec"):
    """Load, merge and save fsl-style bvecs files."""
    final_bvecs = combined_bvec_array(bvecs)
    np.savetxt(output_file, final_bvecs, fmt=str("%.8f"))
    return op.abspath(output_file)


def get_nvols(img):
    """Returns the number of volumes in a 3/4D nifti file."""
    shape = img.shape
    if len(shape) < 4:
        return 1
    return shape[3]


def harmonize_b0s(dwi_files, bvals, b0_threshold, harmonize_b0s):
    """Find the mean intensity of b=0 images in a dwi file and calculate corrections.

    Parameters
    ----------

        dwi_files: list
            List of paths to dwi Nifti files that will be concatenated
        bvals: list
            List of paths to bval files corresponding to the files in ``dwi_files``
        b0_threshold: int
            maximum b values for an image to be considered a b=0
        harmonize_b0s: bool
            Apply a correction to each image so that their mean b=0 images are equal

    Returns
    -------
        to_concat: list
            List of NiftiImage objects to be concatenated. May have been harmonized.
            Same length as the input ``dwi_files``.
        corrections: list
            The correction that would be applied to each image to harmonize their b=0's.
            Same length as the input ``dwi_files``.

    """
    # Load the dwi data and get the mean values from the b=0 images
    num_dwis = len(dwi_files)
    dwi_niis = []
    b0_means = []
    for dwi_file, bval_file in zip(dwi_files, bvals):
        dwi_nii = load_img(dwi_file)
        _bvals = np.loadtxt(bval_file)
        b0_indices = np.flatnonzero(_bvals < b0_threshold)
        if b0_indices.size == 0:
            b0_mean = np.nan
        else:
            if len(b0_indices) > 1:
                b0_mean = index_img(dwi_nii, b0_indices).get_fdata().mean()
            else:
                b0_mean = dwi_nii.get_fdata().mean()
        b0_means.append(b0_mean)
        dwi_niis.append(dwi_nii)

    # Apply the b0 harmonization if requested
    if harmonize_b0s:
        b0_all_mean = np.nanmean(b0_means)
        corrections = b0_all_mean / np.array(b0_means)
        harmonized_niis = []
        for nii_img, correction in zip(dwi_niis, corrections):
            if np.isnan(b0_mean):
                harmonized_niis.append(nii_img)
                LOGGER.warning("An image has no b=0 images and cannot be harmonized")
            else:
                harmonized_niis.append(math_img("img*%.32f" % correction, img=nii_img))
        to_concat = harmonized_niis
    else:
        to_concat = dwi_niis
        corrections = np.ones(num_dwis)

    return to_concat, b0_means, corrections


def create_provenance_dataframe(
    bids_sources, harmonized_niis, b0_means, harmonization_corrections
):
    series_confounds = []
    nvols_per_image = [get_nvols(img) for img in harmonized_niis]
    total_vols = np.sum(nvols_per_image)
    # Check whether the bids sources are per file or per volume
    if not len(bids_sources) == total_vols:
        images_per_volume = []
        for source_image, img_nvols in zip(bids_sources, nvols_per_image):
            images_per_volume.extend([source_image] * img_nvols)
        if not len(images_per_volume) == total_vols:
            raise Exception("Mismatch in number of images and BIDS sources")
        bids_sources = images_per_volume

    for correction, harmonized_nii, b0_mean, nvols in zip(
        harmonization_corrections, harmonized_niis, b0_means, nvols_per_image
    ):
        series_confounds.append(
            pd.DataFrame(
                {
                    "image_mean": [img.get_fdata().mean() for img in iter_img(harmonized_nii)],
                    "series_b0_mean": [b0_mean] * nvols,
                    "series_b0_correction": [correction] * nvols,
                }
            )
        )

    image_df = pd.concat(series_confounds, axis=0, ignore_index=True)
    image_df["original_file"] = bids_sources
    return image_df


class _PhaseToRadInputSpec(BaseInterfaceInputSpec):
    """Output spec for PhaseToRad interface.

    STATEMENT OF CHANGES: This class is derived from sources licensed under the Apache-2.0 terms,
    and the code has been changed.

    Notes
    -----
    The code is derived from
    https://github.com/nipreps/sdcflows/blob/c6cd42944f4b6d638716ce020ffe51010e9eb58a/\
    sdcflows/utils/phasemanip.py#L26.

    License
    -------
    ORIGINAL WORK'S ATTRIBUTION NOTICE:

    Copyright 2021 The NiPreps Developers <nipreps@gmail.com>

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    We support and encourage derived works from this project, please read
    about our expectations at

        https://www.nipreps.org/community/licensing/

    """

    phase_file = File(exists=True, mandatory=True)


class _PhaseToRadOutputSpec(TraitedSpec):
    """Output spec for PhaseToRad interface.

    STATEMENT OF CHANGES: This class is derived from sources licensed under the Apache-2.0 terms,
    and the code has been changed.

    Notes
    -----
    The code is derived from
    https://github.com/nipreps/sdcflows/blob/c6cd42944f4b6d638716ce020ffe51010e9eb58a/\
    sdcflows/utils/phasemanip.py#L26.

    License
    -------
    ORIGINAL WORK'S ATTRIBUTION NOTICE:

    Copyright 2021 The NiPreps Developers <nipreps@gmail.com>

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    We support and encourage derived works from this project, please read
    about our expectations at

        https://www.nipreps.org/community/licensing/

    """

    phase_file = File(exists=True)


class PhaseToRad(SimpleInterface):
    """Convert phase image from arbitrary units (au) to radians.

    This method assumes that the phase image's minimum and maximum values correspond to
    -pi and pi, respectively, and scales the image to be between 0 and 2*pi.

    STATEMENT OF CHANGES: This class is derived from sources licensed under the Apache-2.0 terms,
    and the code has not been changed.

    Notes
    -----
    The code is derived from
    https://github.com/nipreps/sdcflows/blob/c6cd42944f4b6d638716ce020ffe51010e9eb58a/\
    sdcflows/utils/phasemanip.py#L26.

    License
    -------
    ORIGINAL WORK'S ATTRIBUTION NOTICE:

    Copyright 2021 The NiPreps Developers <nipreps@gmail.com>

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    We support and encourage derived works from this project, please read
    about our expectations at

        https://www.nipreps.org/community/licensing/

    """

    input_spec = _PhaseToRadInputSpec
    output_spec = _PhaseToRadOutputSpec

    def _run_interface(self, runtime):
        im = nb.load(self.inputs.phase_file)
        data = im.get_fdata(caching="unchanged")  # Read as float64 for safety
        hdr = im.header.copy()

        # Rescale to [0, 2*pi]
        data = (data - data.min()) * (2 * np.pi / (data.max() - data.min()))

        # Round to float32 and clip
        data = np.clip(np.float32(data), 0.0, 2 * np.pi)

        hdr.set_data_dtype(np.float32)
        hdr.set_xyzt_units("mm")

        # Set the output file name
        self._results["phase_file"] = fname_presuffix(
            self.inputs.phase_file,
            suffix="_rad.nii.gz",
            newpath=runtime.cwd,
            use_ext=False,
        )

        # Save the output image
        nb.Nifti1Image(data, None, hdr).to_filename(self._results["phase_file"])

        return runtime
