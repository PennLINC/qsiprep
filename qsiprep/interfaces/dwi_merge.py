"""Handle merging and spliting of DSI files."""

import numpy as np
from nilearn.image import concat_imgs, load_img, index_img, math_img
import os.path as op
from nipype.interfaces.base import (BaseInterfaceInputSpec, TraitedSpec, File, SimpleInterface,
                                    InputMultiObject, traits, isdefined)
from nipype.utils.filemanip import fname_presuffix
from nipype import logging
import nibabel as nb
LOGGER = logging.getLogger('nipype.workflow')

class MergeDWIsInputSpec(BaseInterfaceInputSpec):
    dwi_files = InputMultiObject(
        File(exists=True), mandatory=True, desc='list of dwi files')
    bids_dwi_files = InputMultiObject(
        File(exists=True), mandatory=True, desc='list of original (BIDS) dwi files')
    bval_files = InputMultiObject(
        File(exists=True), mandatory=True, desc='list of bval files')
    bvec_files = InputMultiObject(
        File(exists=True), mandatory=True, desc='list of bvec files')
    b0_threshold = traits.Int(100, usedefault=True, desc='Maximum b=0 value')
    denoising_confounds = InputMultiObject(
        File(exists=True), desc='list of confound files associated with each input dwi')
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

        if len(self.inputs.dwi_files) == 1:
            dwi_file = self.inputs.dwi_files[0]
            bids_dwi_file = self.inputs.bids_dwi_files[0]
            self._results['out_dwi'] = dwi_file
            self._results['out_bval'] = bvals[0]
            self._results['out_bvec'] = bvecs[0]
            self._results['original_images'] = [bids_dwi_file] * get_nvols(bids_dwi_file)
            confounds = self.inputs.denoising_confounds
            if isdefined(confounds):
                self._results['merged_denoising_confounds'] = confounds
            return runtime

        # Concatenate the gradient information
        merged_fname = fname_presuffix(
            self.inputs.dwi_files[0], suffix=".nii.gz", use_ext=False, newpath=runtime.cwd)
        out_bvec = fname_presuffix(merged_fname, suffix=".bvec", use_ext=False,
                                   newpath=runtime.cwd)
        out_bval = fname_presuffix(merged_fname, suffix=".bval", use_ext=False,
                                   newpath=runtime.cwd)
        self._results['out_bval'] = combine_bvals(bvals, output_file=out_bval)
        self._results['out_bvec'] = combine_bvecs(bvecs, output_file=out_bvec)

        # Load the dwi data and get the mean values from the b=0 images
        dwi_niis = []
        b0_means = []
        for dwi_file, bval_file in zip(self.inputs.dwi_files, bvals):
            dwi_nii = load_img(dwi_file)
            bvals = np.loadtxt(bval_file)
            b0_indices = np.flatnonzero(bvals < self.inputs.b0_threshold)
            if b0_indices.size == 0:
                b0_means.append(np.nan)
            else:
                b0_means.append(index_img(dwi_nii, b0_indices).get_fdata().mean())
            dwi_niis.append(dwi_nii)

        # Apply the b0 harmonization if requested
        if self.inputs.harmonize_b0_intensities:
            b0_all_mean = np.nanmean(b0_means)
            harmonized_niis = []
            for nii_img, b0_mean in zip(dwi_niis, b0_means):
                if np.isnan(b0_mean):
                    harmonized_niis.append(nii_img)
                    LOGGER.warning('An image has no b=0 images and cannot be harmonized')
                else:
                    correction = b0_all_mean / b0_mean
                    harmonized_niis.append(math_img('img*%.32f' % correction, img=nii_img))
            to_concat = harmonized_niis
        else:
            to_concat = dwi_niis

        # Concatenate into a single file
        merged_nii = concat_imgs(to_concat, auto_resample=True)
        # Remove any negative values introduced during interpolation
        pos_merged_nii = math_img('np.clip(img, 0, None)', img=merged_nii)
        pos_merged_nii.to_filename(merged_fname)
        self._results['out_dwi'] = merged_fname

        # Gather the BIDS-source images.
        sources = []
        for img, b0_mean in zip(self.inputs.bids_dwi_files, b0_means):
            sources += [img] * get_nvols(img)
        self._results['original_images'] = sources

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
