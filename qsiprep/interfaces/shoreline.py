#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Image tools interfaces
~~~~~~~~~~~~~~~~~~~~~~


"""
import nibabel as nb
import numpy as np
import os
import os.path as op
from dipy.segment.mask import median_otsu

from tempfile import TemporaryDirectory

from nipype import logging
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec, File, SimpleInterface, InputMultiObject,
    OutputMultiObject, isdefined
)
from .gradients import concatenate_bvecs, concatenate_bvals, _unique_bvecs
from dipy.core.gradients import gradient_table
from ..utils.brainsuite_shore import BrainSuiteShoreModel, brainsuite_shore_basis

LOGGER = logging.getLogger('nipype.interface')

class B0MaskInputSpec(BaseInterfaceInputSpec):
    b0_images = InputMultiObject(File(exists=True))


class B0MaskOutputSpec(TraitedSpec):
    mask_image = File(exists=True)
    average_image = File(exists=True)


class B0Mask(SimpleInterface):
    input_spec = B0MaskInputSpec
    output_spec = B0MaskOutputSpec

    def _run_interface(self, runtime):
        b0_images = [nb.load(fname) for fname in self.inputs.b0_images]
        b0_mean = np.stack([img.get_fdata() for img in b0_images], -1).mean(3)
        mean_file = op.join(runtime.cwd, "b0_mean.nii.gz")
        nb.Nifti1Image(b0_mean, b0_images[0].affine, b0_images[0].header
                       ).to_filename(mean_file)
        self._results['average_image'] = mean_file
        masked_data, data_mask = median_otsu(b0_mean, median_radius=4, numpass=2, autocrop=False,
                                             dilate=1)
        mask_file = op.join(runtime.cwd, "b0_mask.nii.gz")
        nb.Nifti1Image(data_mask, b0_images[0].affine, b0_images[0].header
                       ).to_filename(mask_file)
        self._results['mask_image'] = mask_file

        return runtime


class ExtractDWISForModelInputSpec(BaseInterfaceInputSpec):
    dwi_files = InputMultiObject(File(exists=True))
    bval_files = InputMultiObject(File(exists=True))
    bvec_files = InputMultiObject(File(exists=True))
    transforms = InputMultiObject()
    b0_indices = traits.List()


class ExtractDWISForModelOutputSpec(TraitedSpec):
    model_dwi_files = OutputMultiObject(File(exists=True))
    model_bvals = OutputMultiObject(File(exists=True))
    model_bvecs = OutputMultiObject(File(exists=True))
    transforms = InputMultiObject()


class ExtractDWIsForModel(SimpleInterface):
    """Take a DWI series with interspersed b0 images and create a model-ready version"""
    input_spec = ExtractDWISForModelInputSpec
    output_spec = ExtractDWISForModelOutputSpec

    def _run_interface(self, runtime):
        all_images = self.inputs.dwi_files
        all_bvecs = self.inputs.bvec_files
        all_bvals = self.inputs.bval_files
        b0_indices = self.inputs.b0_indices
        transforms = self.inputs.transforms
        if not len(all_images) == len(all_bvecs) == len(all_bvals) == len(transforms):
            raise Exception("Image, bval, bvec inputs must be of the same length")
        ok_indices = [idx for idx in range(len(all_images)) if idx not in b0_indices]
        self._results['model_dwi_files'] = [all_images[idx] for idx in ok_indices]
        self._results['model_bvals'] = [all_bvals[idx] for idx in ok_indices]
        self._results['model_bvecs'] = [all_bvecs[idx] for idx in ok_indices]
        self._results['transforms'] = [transforms[idx] for idx in ok_indices]

        return runtime


def quick_load_images(image_list, dtype=np.float32):
    example_img = nb.load(image_list[0])
    num_images = len(image_list)
    output_matrix = np.zeros(tuple(example_img.shape) + (num_images,), dtype=dtype)
    for image_num, image_path in enumerate(image_list):
        output_matrix[..., image_num] = nb.load(image_path).get_fdata(dtype=dtype)
    return output_matrix


class SignalPredictionInputSpec(BaseInterfaceInputSpec):
    aligned_dwis = InputMultiObject(File(exists=True))
    aligned_bvecs = traits.Either(InputMultiObject(File(exists=True)), traits.Array)
    bvals = traits.Either(InputMultiObject(File(exists=True)), traits.Array)
    aligned_mask = File(exists=True, mandatory=True)
    aligned_b0_mean = File(exists=True, mandatory=True)
    bvec_to_predict = traits.Array()
    bval_to_predict = traits.Float()
    minimal_q_distance = traits.Float(2.0, usedefault=True)


class SignalPredictionOutputSpec(TraitedSpec):
    predicted_image = File(exists=True)


class SignalPrediction(SimpleInterface):
    """
    """
    input_spec = SignalPredictionInputSpec
    output_spec = SignalPredictionOutputSpec

    def _run_interface(self, runtime):

        # Load the mask image:
        mask_img = nb.load(self.inputs.aligned_mask)
        mask_array = mask_img.get_data() > 1e-6
        all_images = self.inputs.aligned_dwis
        if isinstance(self.inputs.aligned_bvecs, np.ndarray):
            bvecs = self.inputs.aligned_bvecs
        else:
            bvecs = concatenate_bvecs(self.inputs.aligned_bvecs)
        if isinstance(self.inputs.bvals, np.ndarray):
            bvals = self.inputs.bvals
        else:
            bvals = concatenate_bvals(self.inputs.bvals, None)

        # Which sample points are too close to the one we want to predict?
        training_indices = _unique_bvecs(bvals, bvecs, self.inputs.minimal_q_distance)
        training_image_paths = [self.inputs.aligned_b0_mean] + [
            all_images[idx] for idx in training_indices]
        training_bvecs = np.row_stack([np.zeros(3)] + [bvecs[n] for n in training_indices])
        training_bvals = np.array([0.] + [bvals[n] for n in training_indices])
        LOGGER.info("Training with %d of %d", len(training_indices), len(all_images))

        # Load training data and fit the model
        training_data = quick_load_images(training_image_paths)
        training_gtab = gradient_table(bvals=training_bvals, bvecs=training_bvecs)
        shore_model = BrainSuiteShoreModel(training_gtab, regularization="L2")
        shore_fit = shore_model.fit(training_data, mask=mask_array)

        # Get the shore vector for the desired coordinate
        pred_vec = self.inputs.bvec_to_predict
        pred_val = self.inputs.bval_to_predict
        prediction_bvecs = np.tile(pred_vec, (10, 1))
        prediction_bvals = np.ones(10) * pred_val
        prediction_gtab = gradient_table(bvals=prediction_bvals, bvecs=prediction_bvecs)
        prediction_shore = brainsuite_shore_basis(shore_model.radial_order, shore_model.zeta,
                                                  prediction_gtab, shore_model.tau)
        prediction_dir = prediction_shore[0]

        # Calculate the signal prediction, reshape to 3D and save
        shore_array = shore_fit._shore_coef[mask_array]
        output_data = np.zeros(mask_array.shape)
        output_data[mask_array] = np.dot(shore_array, prediction_dir)
        prediction_file = op.join(
            runtime.cwd,
            "predicted_b%d_%.2f_%.2f_%.2f.nii.gz" % ((pred_val,) + tuple(pred_vec)))
        nb.Nifti1Image(output_data, mask_img.affine, mask_img.header
                       ).to_filename(prediction_file)
        self._results['predicted_image'] = prediction_file

        return runtime


class ReorderTransformsInputSpec(BaseInterfaceInputSpec):
    b0_indices = traits.List()
    initial_transforms = InputMultiObject(traits.List())
    model_based_transforms = InputMultiObject(traits.List())


class ReorderTransformsOutputSpec(TraitedSpec):
    transforms = OutputMultiObject(traits.List())


class ReorderTransforms(SimpleInterface):
    input_spec = ReorderTransformsInputSpec
    output_spec = ReorderTransformsOutputSpec

    def _run_interface(self, runtime):

        return runtime
