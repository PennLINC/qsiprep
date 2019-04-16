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


def _nonoverlapping_qspace_samples(prediction_bval, prediction_bvec,
                                   all_bvals, all_bvecs, cutoff):
    """Ensure that none of the training samples are too close to the sample to predict.

    Parameters
    """
    min_bval = min(all_bvals.min(), prediction_bval)
    all_qvals = np.sqrt(all_bvals - min_bval)
    prediction_qval = np.sqrt(prediction_bval - min_bval)

    # Convert q values to percent of maximum qval
    max_qval = max(all_qvals.max(), prediction_qval)
    all_qvals_scaled = all_qvals / max_qval * 100
    prediction_qval_scaled = prediction_qval / max_qval * 100
    scaled_qvecs = all_bvecs * all_qvals_scaled[:, np.newaxis]
    scaled_prediction_qvec = prediction_bvec * prediction_qval_scaled

    # Calculate the distance between the sampled qvecs and the prediction qvec
    distances = np.linalg.norm(scaled_qvecs - scaled_prediction_qvec, axis=1)
    distances_flip = np.linalg.norm(scaled_qvecs + scaled_prediction_qvec, axis=1)
    ok_samples = (distances > cutoff) * (distances_flip > cutoff)

    return ok_samples


class B0MeanInputSpec(BaseInterfaceInputSpec):
    b0_images = InputMultiObject(File(exists=True), mandatory=True)


class B0MeanOutputSpec(TraitedSpec):
    average_image = File(exists=True)


class B0Mean(SimpleInterface):
    input_spec = B0MeanInputSpec
    output_spec = B0MeanOutputSpec

    def _run_interface(self, runtime):
        b0_images = [nb.load(fname) for fname in self.inputs.b0_images]
        b0_mean = np.stack([img.get_fdata() for img in b0_images], -1).mean(3)
        mean_file = op.join(runtime.cwd, "b0_mean.nii.gz")
        nb.Nifti1Image(b0_mean, b0_images[0].affine, b0_images[0].header
                       ).to_filename(mean_file)
        self._results['average_image'] = mean_file
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
        pred_vec = self.inputs.bvec_to_predict
        pred_val = self.inputs.bval_to_predict
        # Load the mask image:
        mask_img = nb.load(self.inputs.aligned_mask)
        mask_array = mask_img.get_data() > 1e-6
        all_images = self.inputs.aligned_dwis
        if isinstance(self.inputs.aligned_bvecs, np.ndarray):
            bvecs = self.inputs.aligned_bvecs
        else:
            bvecs = concatenate_bvecs(self.inputs.aligned_bvecs)
        all_bvecs = np.row_stack([np.zeros(3)] + bvecs.tolist())
        if isinstance(self.inputs.bvals, np.ndarray):
            bvals = self.inputs.bvals
        else:
            bvals = concatenate_bvals(self.inputs.bvals, None)
        all_bvals = np.array([0.] + bvals.tolist())

        # Which sample points are too close to the one we want to predict?
        training_mask = _nonoverlapping_qspace_samples(
            pred_val, pred_vec, all_bvals, all_bvecs, self.inputs.minimal_q_distance)
        training_indices = np.flatnonzero(training_mask[1:])
        training_image_paths = [self.inputs.aligned_b0_mean] + [
            all_images[idx] for idx in training_indices]
        training_bvecs = all_bvecs[training_mask]
        training_bvals = all_bvals[training_mask]
        LOGGER.info("Training with %d of %d", training_mask.sum(), len(training_mask))

        # Load training data and fit the model
        training_data = quick_load_images(training_image_paths)
        training_gtab = gradient_table(bvals=training_bvals, bvecs=training_bvecs)
        shore_model = BrainSuiteShoreModel(training_gtab, regularization="L2")
        shore_fit = shore_model.fit(training_data, mask=mask_array)

        # Get the shore vector for the desired coordinate

        prediction_bvecs = np.tile(pred_vec, (10, 1))
        prediction_bvals = np.ones(10) * pred_val
        prediction_bvals[9] = 0  # prevent warning
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


class ReorderOutputsInputSpec(BaseInterfaceInputSpec):
    b0_indices = traits.List(mandatory=True)
    b0_mean = File(exists=True, mandatory=True)
    initial_transforms = InputMultiObject(File(exists=True), mandatory=True)
    model_based_transforms = InputMultiObject(traits.List(), mandatory=True)
    model_predicted_images = InputMultiObject(File(exists=True), mandatory=True)


class ReorderOutputsOutputSpec(TraitedSpec):
    full_transforms = OutputMultiObject(traits.List())
    full_predicted_dwi_series = OutputMultiObject(traits.File(exists=True))


class ReorderOutputs(SimpleInterface):
    input_spec = ReorderOutputsInputSpec
    output_spec = ReorderOutputsOutputSpec

    def _run_interface(self, runtime):
        full_transforms = []
        full_predicted_dwi_series = []
        model_transforms = self.inputs.model_based_transforms[::-1]
        model_images = self.inputs.model_predicted_images[::-1]
        b0_transforms = [self.inputs.initial_transforms[idx] for idx in
                         self.inputs.b0_indices][::-1]
        num_dwis = len(self.inputs.initial_transforms)
        for imagenum in range(num_dwis):
            if imagenum in self.inputs.b0_indices:
                full_predicted_dwi_series.append(self.inputs.b0_mean)
                full_transforms.append(b0_transforms.pop())
            else:
                full_transforms.append(model_transforms.pop())
                full_predicted_dwi_series.append(model_images.pop())
        if not len(model_transforms) == len(b0_transforms) == len(model_images) == 0:
            raise Exception("Unable to recombine images and transforms")
        self._results['full_transforms'] = full_transforms
        self._results['full_predicted_dwi_series'] = full_predicted_dwi_series

        return runtime
