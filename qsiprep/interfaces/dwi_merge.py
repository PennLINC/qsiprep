"""Handle merging and spliting of DSI files."""
import os.path as op
import numpy as np
import pandas as pd
from nilearn.image import concat_imgs, load_img, index_img, math_img, iter_img
from nipype.interfaces.base import (BaseInterfaceInputSpec, TraitedSpec, File, SimpleInterface,
                                    InputMultiObject, traits, isdefined)
from nipype.utils.filemanip import fname_presuffix
from nipype import logging
import nibabel as nb
from ..workflows.dwi.util import _get_concatenated_bids_name
LOGGER = logging.getLogger('nipype.workflow')


class MergeDWIsInputSpec(BaseInterfaceInputSpec):
    dwi_files = InputMultiObject(
        File(), mandatory=True, desc='list of dwi files')
    bids_dwi_files = InputMultiObject(
        File(), mandatory=True, desc='list of original (BIDS) dwi files')
    bval_files = InputMultiObject(
        File(exists=True), mandatory=True, desc='list of bval files')
    bvec_files = InputMultiObject(
        File(exists=True), mandatory=True, desc='list of bvec files')
    b0_threshold = traits.Int(100, usedefault=True, desc='Maximum b=0 value')
    denoising_confounds = InputMultiObject(
        File(exists=True, desc='list of confound files associated with each input dwi'))
    harmonize_b0_intensities = traits.Bool(True, usedefault=True,
                                           desc='Force scans to have the same mean b=0 intensity')


class MergeDWIsOutputSpec(TraitedSpec):
    out_dwi = File(desc='the merged dwi image')
    out_bval = File(desc='the merged bvec file')
    out_bvec = File(desc='the merged bval file')
    original_images = traits.List()
    merged_metadata = traits.Dict()
    merged_denoising_confounds = File(exists=True)


class MergeDWIs(SimpleInterface):
    input_spec = MergeDWIsInputSpec
    output_spec = MergeDWIsOutputSpec

    def _run_interface(self, runtime):
        bvals = self.inputs.bval_files
        bvecs = self.inputs.bvec_files
        num_dwis = len(self.inputs.dwi_files)

        # Load the dwi data and get the mean values from the b=0 images
        dwi_niis = []
        b0_means = []
        for dwi_file, bval_file in zip(self.inputs.dwi_files, bvals):
            dwi_nii = load_img(dwi_file)
            _bvals = np.loadtxt(bval_file)
            b0_indices = np.flatnonzero(_bvals < self.inputs.b0_threshold)
            if b0_indices.size == 0:
                b0_mean = np.nan
            else:
                b0_mean = index_img(dwi_nii, b0_indices).get_fdata().mean()
            b0_means.append(b0_mean)
            dwi_niis.append(dwi_nii)

        # Apply the b0 harmonization if requested
        if self.inputs.harmonize_b0_intensities:
            b0_all_mean = np.nanmean(b0_means)
            corrections = b0_all_mean / np.array(b0_means)
            harmonized_niis = []
            for nii_img, correction in zip(dwi_niis, corrections):
                if np.isnan(b0_mean):
                    harmonized_niis.append(nii_img)
                    LOGGER.warning('An image has no b=0 images and cannot be harmonized')
                else:
                    harmonized_niis.append(math_img('img*%.32f' % correction, img=nii_img))
            to_concat = harmonized_niis
        else:
            to_concat = dwi_niis
            corrections = np.ones(num_dwis)

        # Get basic qc / provenance per volume
        sources = self.inputs.bids_dwi_files
        series_confounds = []
        for bids_source, b0_mean, correction, nii, bval, bvec in zip(
                sources, b0_means, corrections, dwi_niis, bvals, bvecs):
            nvols = get_nvols(bids_source)
            bx, by, bz = np.loadtxt(bvec)
            series_confounds.append(
                pd.DataFrame({
                    "original_file": [bids_source] * nvols,
                    "original_bval": np.loadtxt(bval),
                    "image_mean": [img.get_fdata().mean() for img in iter_img(nii)],
                    "original_bx": bx,
                    "original_by": by,
                    "original_bz": bz,
                    "series_b0_mean": [b0_mean] * nvols,
                    "series_b0_correction": [correction] * nvols}))
        image_df = pd.concat(series_confounds, axis=0, ignore_index=True)

        # Collect the confounds
        if isdefined(self.inputs.denoising_confounds):
            confounds = [pd.read_csv(fname) for fname in self.inputs.denoising_confounds]
            _confounds_df = pd.concat(confounds, axis=0, ignore_index=True)
            confounds_df = pd.concat([image_df, _confounds_df], axis=1, ignore_index=False)
        else:
            confounds_df = image_df

        # Concatenate the gradient information
        if num_dwis > 1:
            merged_output = _get_concatenated_bids_name(
                {'dwi_series': self.inputs.dwi_files,
                 'fieldmap_info': {'suffix': None}})
            merged_fname = op.join(runtime.cwd, merged_output + "_merged.nii.gz")
            out_bval = fname_presuffix(merged_fname, suffix=".bval", use_ext=False,
                                       newpath=runtime.cwd)
            out_bvec = fname_presuffix(merged_fname, suffix=".bvec", use_ext=False,
                                       newpath=runtime.cwd)
        else:
            merged_fname = self.inputs.dwi_files[0]
            out_bval = bvals[0]
            out_bvec = bvecs[0]
        merged_confounds = fname_presuffix(merged_fname, suffix="_confounds.csv", use_ext=False,
                                           newpath=runtime.cwd)
        confounds_df = confounds_df.drop('Unnamed: 0', axis=1, errors='ignore')
        confounds_df.to_csv(merged_confounds, index=False)

        self._results['merged_denoising_confounds'] = merged_confounds
        self._results['original_images'] = confounds_df['original_file'].tolist()
        self._results['out_dwi'] = merged_fname
        self._results['out_bval'] = out_bval
        self._results['out_bvec'] = out_bvec

        if num_dwis == 1:
            return runtime

        # Write the merged gradients
        combine_bvals(bvals, output_file=out_bval)
        combine_bvecs(bvecs, output_file=out_bvec)
        # Concatenate into a single file
        merged_nii = concat_imgs(to_concat, auto_resample=True)
        # Remove any negative values introduced during interpolation (if it occurrs)
        pos_merged_nii = math_img('np.clip(img, 0, None)', img=merged_nii)
        pos_merged_nii.to_filename(merged_fname)

        return runtime


class StackConfoundsInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiObject(File(exists=True), mandatory=True)
    axis = traits.Enum(0, 1, default=0, usedefault=True)
    out_file = File()


class StackConfoundsOutputSpec(TraitedSpec):
    confounds_file = File(desc='the stacked confound data')


class StackConfounds(SimpleInterface):
    input_spec = StackConfoundsInputSpec
    output_spec = StackConfoundsOutputSpec

    def _run_interface(self, runtime):
        if not self.inputs.in_files:
            return runtime
        dfs = [pd.read_csv(fname) for fname in self.inputs.in_files]
        stacked = pd.concat(dfs, axis=self.inputs.axis, ignore_index=self.inputs.axis == 0)
        out_file = op.join(runtime.cwd, 'confounds.csv')
        stacked = stacked.drop('Unnamed: 0', axis=1, errors='ignore')
        stacked.to_csv(out_file)
        self._results['confounds_file'] = out_file
        return runtime


def combine_bvals(bvals, output_file="restacked.bval"):
    """Load, merge and save fsl-style bvals files."""
    collected_vals = []
    for bval_file in bvals:
        collected_vals.append(np.atleast_1d(np.loadtxt(bval_file)))
    final_bvals = np.concatenate(collected_vals)
    np.savetxt(output_file, final_bvals, fmt=str("%i"))
    return op.abspath(output_file)


def combine_bvecs(bvecs, output_file="restacked.bvec"):
    """Load, merge and save fsl-style bvecs files."""
    collected_vecs = []
    for bvec_file in bvecs:
        collected_vecs.append(np.loadtxt(bvec_file))
    final_bvecs = np.column_stack(collected_vecs)
    np.savetxt(output_file, final_bvecs, fmt=str("%.8f"))
    return op.abspath(output_file)


def get_nvols(img):
    """Returns the number of volumes in a 3/4D nifti file."""
    shape = nb.load(img).shape
    if len(shape) < 4:
        return 1
    return shape[3]
