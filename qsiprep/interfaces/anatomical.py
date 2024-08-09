#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Image tools interfaces
~~~~~~~~~~~~~~~~~~~~~~


"""

import nibabel as nb
import nilearn.image as nim
import numpy as np
from dipy.segment.threshold import otsu
from nipype import logging
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.utils.filemanip import fname_presuffix
from pkg_resources import resource_filename as pkgr
from scipy import ndimage
from scipy.spatial import distance

LOGGER = logging.getLogger("nipype.interface")
KNOWN_TEMPLATES = ["MNI152NLin2009cAsym", "infant"]


class _DiceOverlapInputSpec(BaseInterfaceInputSpec):
    anatomical_mask = File(exists=True, mandatory=True, desc="Mask from a T1w image")
    dwi_mask = File(exists=True, mandatory=True, desc="Mask from a DWI image")


class _DiceOverlapOutputSpec(TraitedSpec):
    dice_score = traits.Float()


class DiceOverlap(SimpleInterface):
    input_spec = _DiceOverlapInputSpec
    output_spec = _DiceOverlapOutputSpec

    def _run_interface(self, runtime):
        t1_img = nb.load(self.inputs.anatomical_mask)
        dwi_img = nb.load(self.inputs.dwi_mask)

        if not t1_img.shape == dwi_img.shape:
            raise Exception("Cannot compare masks with different shapes")

        self._results["dice_score"] = distance.dice(
            t1_img.get_fdata().flatten(), dwi_img.get_fdata().flatten()
        )
        return runtime


class _VoxelSizeChooserInputSpec(BaseInterfaceInputSpec):
    voxel_size = traits.Float()
    input_image = File(exists=True)
    anisotropic_strategy = traits.Enum("min", "max", "mean", usedefault=True)


class _VoxelSizeChooserOutputSpec(TraitedSpec):
    voxel_size = traits.Float()


class VoxelSizeChooser(SimpleInterface):
    input_spec = _VoxelSizeChooserInputSpec
    output_spec = _VoxelSizeChooserOutputSpec

    def _run_interface(self, runtime):
        if not isdefined(self.inputs.input_image) and not isdefined(self.inputs.voxel_size):
            raise Exception("Either voxel_size or input_image need to be defined")

        # A voxel size was specified without an image
        if isdefined(self.inputs.voxel_size):
            voxel_size = self.inputs.voxel_size
        else:
            # An image was provided
            img = nb.load(self.inputs.input_image)
            zooms = img.header.get_zooms()[:3]
            if self.inputs.anisotropic_strategy == "min":
                voxel_size = min(zooms)
            elif self.inputs.anisotropic_strategy == "max":
                voxel_size = max(zooms)
            else:
                voxel_size = np.round(np.mean(zooms), 2)

        self._results["voxel_size"] = voxel_size
        return runtime


class _FakeSegmentationInputSpec(BaseInterfaceInputSpec):
    mask_file = File(exists=True, mandatory=True)


class _FakeSegmentationOutputSpec(TraitedSpec):
    dseg_file = File(exists=True)


class FakeSegmentation(SimpleInterface):
    input_spec = _FakeSegmentationInputSpec
    output_spec = _FakeSegmentationOutputSpec

    def _run_interface(self, runtime):
        img = nb.load(self.inputs.mask_file)
        orig_mask = img.get_fdata() > 0
        eroded1 = ndimage.binary_erosion(orig_mask, iterations=3)
        eroded2 = ndimage.binary_erosion(eroded1, iterations=3)
        final = orig_mask.astype(int) + eroded1 + eroded2
        out_img = nb.Nifti1Image(final, img.affine, header=img.header)
        out_fname = fname_presuffix(self.inputs.mask_file, suffix="_dseg", newpath=runtime.cwd)
        out_img.to_filename(out_fname)
        self._results["dseg_file"] = out_fname

        return runtime


class _DesaturateSkullInputSpec(BaseInterfaceInputSpec):
    skulled_t2w_image = File(exists=True, mandatory=True, desc="Skull-on T2w image")
    brain_mask_image = File(
        exists=True, mandatory=True, desc="Binary brain mask in the same grid as skulled_t2w_image"
    )
    brain_to_skull_ratio = traits.CFloat(
        8.0, usedefault=True, desc="Ratio of signal in the brain to signal in the skull"
    )


class _DesaturateSkullOutputSpec(TraitedSpec):
    desaturated_t2w = File(exists=True)
    head_scaling_factor = traits.Float(0.0)


class DesaturateSkull(SimpleInterface):
    input_spec = _DesaturateSkullInputSpec
    output_spec = _DesaturateSkullOutputSpec

    def _run_interface(self, runtime):

        out_file = fname_presuffix(
            self.inputs.skulled_t2w_image,
            newpath=runtime.cwd,
            suffix="_desaturated.nii",
            use_ext=False,
        )
        skulled_img = nim.load_img(self.inputs.skulled_t2w_image)
        brainmask_img = nim.load_img(self.inputs.brain_mask_image)
        brain_median, nonbrain_head_median = calculate_nonbrain_saturation(
            skulled_img, brainmask_img
        )

        actual_brain_to_skull_ratio = brain_median / nonbrain_head_median
        LOGGER.info("found brain to skull ratio: %.3f", actual_brain_to_skull_ratio)
        desat_data = skulled_img.get_fdata(dtype="float32").copy()
        adjustment = 1.0
        if actual_brain_to_skull_ratio < self.inputs.brain_to_skull_ratio:
            # We need to downweight the non-brain voxels
            adjustment = actual_brain_to_skull_ratio / self.inputs.brain_to_skull_ratio
            LOGGER.info("Desaturating outside-brain signal by %.5f" % adjustment)
            nonbrain_mask = brainmask_img.get_fdata() < 1
            # Apply the adjustment
            desat_data[nonbrain_mask] = desat_data[nonbrain_mask] * adjustment

        desat_img = nim.new_img_like(skulled_img, desat_data, copy_header=True)
        desat_img.header.set_data_dtype("float32")
        desat_img.to_filename(out_file)
        self._results["desaturated_t2w"] = out_file
        self._results["head_scaling_factor"] = adjustment
        return runtime


def calculate_nonbrain_saturation(head_img, brain_mask_img):
    # Calculate the
    head_data = head_img.get_fdata()
    brain_mask = brain_mask_img.get_fdata() > 0

    def clip_values(values):
        _, top_percent = np.percentile(values, np.array([0, 99.75]), axis=None)
        return np.clip(values, 0, top_percent)

    nonbrain_voxels = head_data[np.logical_not(brain_mask)]
    winsorized_nonbrain_voxels = clip_values(nonbrain_voxels)
    threshold = otsu(winsorized_nonbrain_voxels) * 0.5

    nbmask = np.zeros_like(head_img.get_fdata())
    nbmask[head_data > threshold] = 2
    nbmask[brain_mask] = 0

    in_brain_median = np.median(head_data[brain_mask])
    non_brain_head_median = np.median(head_data[nbmask > 0])

    return in_brain_median, non_brain_head_median


class _GetTemplateInputSpec(BaseInterfaceInputSpec):
    template_name = traits.Str("MNI152NLin2009cAsym", usedefault=True, mandatory=True)
    t1_file = File(exists=True)
    t2_file = File(exists=True)
    mask_file = File(exists=True)
    infant_mode = traits.Bool(False, usedefault=True)
    anatomical_contrast = traits.Enum("T1w", "T2w", "none")


class _GetTemplateOutputSpec(BaseInterfaceInputSpec):
    template_name = traits.Str()
    template_file = File(exists=True)
    template_brain_file = File(exists=True)
    template_mask_file = File(exists=True)


class GetTemplate(SimpleInterface):
    input_spec = _GetTemplateInputSpec
    output_spec = _GetTemplateOutputSpec

    def _run_interface(self, runtime):
        self._results["template_name"] = self.inputs.template_name
        contrast_name = self.inputs.anatomical_contrast.lower()
        if contrast_name == "none":
            LOGGER.info("Using T1w modality template for ACPC alignment")
            contrast_name = "t1w"

        # Cover the cases where the template images are actually in the
        # qsiprep package. This is for common use cases (MNI2009cAsym and Infant)
        # and legacy
        if self.inputs.template_name in KNOWN_TEMPLATES or self.inputs.infant_mode:
            if not self.inputs.infant_mode:
                ref_img = pkgr("qsiprep", "data/mni_1mm_%s_lps.nii.gz" % contrast_name)
                ref_img_brain = pkgr("qsiprep", "data/mni_1mm_%s_lps_brain.nii.gz" % contrast_name)
                ref_img_mask = pkgr("qsiprep", "data/mni_1mm_t1w_lps_brainmask.nii.gz")
            else:
                ref_img = pkgr("qsiprep", "data/mni_1mm_%s_lps_infant.nii.gz" % contrast_name)
                ref_img_brain = pkgr(
                    "qsiprep", "data/mni_1mm_%s_lps_brain_infant.nii.gz" % contrast_name
                )
                ref_img_mask = pkgr("qsiprep", "data/mni_1mm_t1w_lps_brainmask_infant.nii.gz")
            self._results["template_file"] = ref_img
            self._results["template_brain_file"] = ref_img_brain
            self._results["template_mask_file"] = ref_img_mask
        else:
            raise NotImplementedError("Arbitrary templates not available yet")

        return runtime
