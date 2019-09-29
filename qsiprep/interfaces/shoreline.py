#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
SHORELine interfaces
~~~~~~~~~~~~~~~~~~~~~~


"""
import os.path as op
import nibabel as nb
import numpy as np
import os
import pandas as pd
from skimage import measure
import imageio
from nipype import logging
from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec, File, SimpleInterface, InputMultiObject,
    OutputMultiObject, isdefined
)
from .gradients import concatenate_bvecs, concatenate_bvals
from dipy.core.gradients import gradient_table
from ..utils.brainsuite_shore import BrainSuiteShoreModel, brainsuite_shore_basis
from .reports import SummaryInterface, SummaryOutputSpec
import seaborn as sns
import matplotlib.pyplot as plt

LOGGER = logging.getLogger('nipype.interface')


class GroupImagesInputSpec(BaseInterfaceInputSpec):
    image_groups = traits.Dict(mandatory=True)
    dwi_files = InputMultiObject(File(exists=True))
    bval_files = InputMultiObject(File(exists=True))
    bvec_files = InputMultiObject(File(exists=True))
    original_files = InputMultiObject(File(exists=True))


class GroupImagesOutputSpec(TraitedSpec):
    plus_dwi_files = OutputMultiObject(File(exists=True))
    plus_bval_files = OutputMultiObject(File(exists=True))
    plus_bvec_files = OutputMultiObject(File(exists=True))
    plus_original_files = OutputMultiObject(File(exists=True))
    minus_dwi_files = OutputMultiObject(File(exists=True))
    minus_bval_files = OutputMultiObject(File(exists=True))
    minus_bvec_files = OutputMultiObject(File(exists=True))
    minus_original_files = OutputMultiObject(File(exists=True))


class GroupImages(SimpleInterface):
    input_spec = GroupImagesInputSpec
    output_spec = GroupImagesOutputSpec


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
    model = traits.Str('3dSHORE', usedefault=True)


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
        if self.inputs.model == '3dSHORE':
            shore_model = BrainSuiteShoreModel(training_gtab, regularization="L2")
        else:
            raise NotImplementedError('Unsupported model: ' + self.inputs.model)
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
            "predicted_b%d_%.2f_%.2f_%.2f.nii.gz" % (
                (pred_val,) + tuple(np.round(pred_vec, decimals=2))))
        nb.Nifti1Image(output_data, mask_img.affine, mask_img.header
                       ).to_filename(prediction_file)
        self._results['predicted_image'] = prediction_file

        return runtime


class CalculateCNRInputSpec(BaseInterfaceInputSpec):
    hmc_warped_images = InputMultiObject(File(exists=True))
    predicted_images = InputMultiObject(File(exists=True))
    mask_image = File(exists=True)


class CalculateCNROutputSpec(TraitedSpec):
    cnr_image = File(exists=True)


class CalculateCNR(SimpleInterface):
    input_spec = CalculateCNRInputSpec
    output_spec = CalculateCNROutputSpec

    def _run_interface(self, runtime):
        cnr_file = op.join(runtime.cwd, "SHORELine_CNR.nii.gz")
        model_images = quick_load_images(self.inputs.predicted_images)
        observed_images = quick_load_images(self.inputs.hmc_warped_images)
        mask_image = nb.load(self.inputs.mask_image)
        mask = mask_image.get_data() > 1e-6
        signal_vals = model_images[mask]
        b0 = signal_vals[:, 0][:, np.newaxis]
        signal_vals = signal_vals / b0
        signal_var = np.var(signal_vals, 1)
        observed_vals = observed_images[mask] / b0
        noise_var = np.var(signal_vals - observed_vals, 1)
        snr = np.nan_to_num(signal_var / noise_var)
        out_mat = np.zeros(mask_image.shape)
        out_mat[mask] = snr
        nb.Nifti1Image(out_mat, mask_image.affine,
                       header=mask_image.header).to_filename(cnr_file)
        self._results['cnr_image'] = cnr_file
        return runtime


class ReorderOutputsInputSpec(BaseInterfaceInputSpec):
    b0_indices = traits.List(mandatory=True)
    b0_mean = File(exists=True, mandatory=True)
    warped_b0_images = InputMultiObject(File(exists=True), mandatory=True)
    warped_dwi_images = InputMultiObject(File(exists=True), mandatory=True)
    initial_transforms = InputMultiObject(File(exists=True), mandatory=True)
    model_based_transforms = InputMultiObject(traits.List(), mandatory=True)
    model_predicted_images = InputMultiObject(File(exists=True), mandatory=True)


class ReorderOutputsOutputSpec(TraitedSpec):
    full_transforms = OutputMultiObject(traits.List())
    full_predicted_dwi_series = OutputMultiObject(File(exists=True))
    hmc_warped_images = OutputMultiObject(File(exists=True))


class ReorderOutputs(SimpleInterface):
    input_spec = ReorderOutputsInputSpec
    output_spec = ReorderOutputsOutputSpec

    def _run_interface(self, runtime):
        full_transforms = []
        full_predicted_dwi_series = []
        full_warped_images = []
        warped_b0_images = self.inputs.warped_b0_images[::-1]
        warped_dwi_images = self.inputs.warped_dwi_images[::-1]
        model_transforms = self.inputs.model_based_transforms[::-1]
        model_images = self.inputs.model_predicted_images[::-1]
        b0_transforms = [self.inputs.initial_transforms[idx] for idx in
                         self.inputs.b0_indices][::-1]
        num_dwis = len(self.inputs.initial_transforms)

        for imagenum in range(num_dwis):
            if imagenum in self.inputs.b0_indices:
                full_predicted_dwi_series.append(self.inputs.b0_mean)
                full_transforms.append(b0_transforms.pop())
                full_warped_images.append(warped_b0_images.pop())
            else:
                full_transforms.append(model_transforms.pop())
                full_predicted_dwi_series.append(model_images.pop())
                full_warped_images.append(warped_dwi_images.pop())

        if not len(model_transforms) == len(b0_transforms) == len(model_images) == 0:
            raise Exception("Unable to recombine images and transforms")

        self._results['hmc_warped_images'] = full_warped_images
        self._results['full_transforms'] = full_transforms
        self._results['full_predicted_dwi_series'] = full_predicted_dwi_series

        return runtime


class IterationSummaryInputSpec(BaseInterfaceInputSpec):
    collected_motion_files = InputMultiObject(File(exists=True))


class IterationSummaryOutputSpec(TraitedSpec):
    iteration_summary_file = File(exists=True)
    plot_file = File(exists=True)


class IterationSummary(SummaryInterface):
    input_spec = IterationSummaryInputSpec
    output_spec = IterationSummaryOutputSpec

    def _run_interface(self, runtime):
        motion_files = self.inputs.collected_motion_files
        output_fname = op.join(runtime.cwd, "iteration_summary.csv")
        fig_output_fname = op.join(runtime.cwd, "iterdiffs.svg")
        if not isdefined(motion_files):
            return runtime

        all_iters = []
        for fnum, fname in enumerate(motion_files):
            df = pd.read_csv(fname)
            df['iter_num'] = fnum
            path_parts = fname.split(os.sep)
            itername = '' if 'iter' not in path_parts[-3] else path_parts[-3]
            df['iter_name'] = itername
            all_iters.append(df)
        combined = pd.concat(all_iters, axis=0, ignore_index=True)

        combined.to_csv(output_fname, index=False)
        self._results['iteration_summary_file'] = output_fname

        # Create a figure for the report
        _iteration_summary_plot(combined, fig_output_fname)
        self._results['plot_file'] = fig_output_fname

        return runtime


class SHORELineReportInputSpec(BaseInterfaceInputSpec):
    iteration_summary = File(exists=True)
    registered_images = InputMultiObject(File(exists=True))
    original_images = InputMultiObject(File(exists=True))
    model_predicted_images = InputMultiObject(File(exists=True))


class SHORELineReportOutputSpec(SummaryOutputSpec):
    plot_file = File(exists=True)


class SHORELineReport(SummaryInterface):
    input_spec = SHORELineReportInputSpec
    output_spec = SHORELineReportOutputSpec

    def _run_interface(self, runtime):
        images = []
        for imagenum, (orig_file, aligned_file, model_file) in enumerate(zip(
                self.inputs.original_images, self.inputs.registered_images,
                self.inputs.model_predicted_images)):

            images.extend(before_after_images(orig_file, aligned_file, model_file, imagenum))

        out_file = op.join(runtime.cwd, "shoreline_reg.gif")
        imageio.mimsave(out_file, images, fps=1)
        self._results['plot_file'] = out_file
        return runtime


def scaled_mip(img1, img2, img3, axis):
    mip1 = img1.max(axis=axis).T
    mip2 = img2.max(axis=axis).T
    mip3 = img3.max(axis=axis).T
    max_obs = max(mip1.max(), mip2.max(), mip3.max())
    vmax = 0.98 * max_obs
    return (np.clip(mip1, 0, vmax) / vmax,
            np.clip(mip2, 0, vmax) / vmax,
            np.clip(mip3, 0, vmax) / vmax)


def to_image(fig):
    fig.subplots_adjust(hspace=0, left=0, right=1, wspace=0)
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image


def before_after_images(orig_file, aligned_file, model_file, imagenum):
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    fig.subplots_adjust(hspace=0, left=0, right=1, wspace=0)
    for _ax in ax:
        _ax.clear()
    orig_img = nb.load(orig_file).get_fdata()
    aligned_img = nb.load(aligned_file).get_fdata()
    model_img = nb.load(model_file).get_fdata()
    orig_mip, aligned_mip, target_mip = scaled_mip(orig_img, aligned_img, model_img, 0)

    # Get contours for the orig, aligned images
    orig_contours = measure.find_contours(orig_mip, 0.7)
    aligned_contours = measure.find_contours(aligned_mip, 0.7)
    target_contours = measure.find_contours(target_mip, 0.7)

    orig_contours_low = measure.find_contours(orig_mip, 0.05)
    aligned_contours_low = measure.find_contours(aligned_mip, 0.05)
    target_contours_low = measure.find_contours(target_mip, 0.05)

    # Plot before
    ax[0].imshow(orig_mip, vmax=1., vmin=0, origin="lower", cmap="gray",
                 interpolation="nearest")
    ax[1].imshow(target_mip, vmax=1., vmin=0, origin="lower", cmap="gray",
                 interpolation="nearest")
    ax[0].text(1, 1, "%03d: Before" % imagenum, fontsize=16, color='white')
    for contour in target_contours + target_contours_low:
        ax[0].plot(contour[:, 1], contour[:, 0], linewidth=2, alpha=0.9, color="#e7298a")
        ax[1].plot(contour[:, 1], contour[:, 0], linewidth=2, alpha=0.9, color="#e7298a")
    for contour in orig_contours + orig_contours_low:
        ax[1].plot(contour[:, 1], contour[:, 0], linewidth=2, alpha=0.9, color="#d95f02")
        ax[0].plot(contour[:, 1], contour[:, 0], linewidth=2, alpha=0.9, color="#d95f02")
    for axis in ax:
        axis.set_xticks([])
        axis.set_yticks([])

    before_image = to_image(fig)

    # Plot after
    for _ax in ax:
        _ax.clear()
    ax[0].imshow(aligned_mip, vmax=1., vmin=0, origin="lower", cmap="gray",
                 interpolation="nearest")
    ax[1].imshow(target_mip, vmax=1., vmin=0, origin="lower", cmap="gray",
                 interpolation="nearest")
    ax[0].text(1, 1, "%03d: After" % imagenum, fontsize=16, color='white')
    for contour in target_contours + target_contours_low:
        ax[0].plot(contour[:, 1], contour[:, 0], linewidth=2, alpha=0.9, color="#e7298a")
        ax[1].plot(contour[:, 1], contour[:, 0], linewidth=2, alpha=0.9, color="#e7298a")
    for contour in aligned_contours + aligned_contours_low:
        ax[1].plot(contour[:, 1], contour[:, 0], linewidth=2, alpha=0.9, color="#d95f02")
        ax[0].plot(contour[:, 1], contour[:, 0], linewidth=2, alpha=0.9, color="#d95f02")
    for axis in ax:
        axis.set_xticks([])
        axis.set_yticks([])
    after_image = to_image(fig)

    return before_image, after_image


def _iteration_summary_plot(iters_df, out_file):
    iters = list([item[1] for item in iters_df.groupby('iter_num')])
    shift_cols = ["shiftX", "shiftY", "shiftZ"]
    rotate_cols = ["rotateX", "rotateY", "rotateZ"]
    shifts = np.stack([df[shift_cols] for df in iters], -1)
    rotations = np.stack([df[rotate_cols] for df in iters], -1)

    rot_diffs = np.diff(rotations, axis=2).squeeze()
    shift_diffs = np.diff(shifts, axis=2).squeeze()
    if len(iters) == 2:
        rot_diffs = rot_diffs[..., np.newaxis]
        shift_diffs = shift_diffs[..., np.newaxis]

    shiftdiff_dfs = []
    rotdiff_dfs = []
    for diffnum, (rot_diff, shift_diff) in enumerate(zip(rot_diffs.T, shift_diffs.T)):
        shiftdiff_df = pd.DataFrame(shift_diff.T, columns=shift_cols)
        shiftdiff_df['difference_num'] = "%02d" % diffnum
        shiftdiff_dfs.append(shiftdiff_df)

        rotdiff_df = pd.DataFrame(rot_diff.T, columns=rotate_cols)
        rotdiff_df['difference_num'] = "%02d" % diffnum
        rotdiff_dfs.append(rotdiff_df)

    shift_diffs = pd.concat(shiftdiff_dfs, axis=0)
    rotate_diffs = pd.concat(rotdiff_dfs, axis=0)

    # Plot shifts
    sns.set()
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    sns.violinplot(x="variable", y="value",
                   hue="difference_num",
                   ax=ax[0],
                   data=shift_diffs.melt(id_vars=['difference_num']))
    ax[0].set_ylabel("mm")
    ax[0].set_title("Shift")

    # Plot rotations
    sns.violinplot(x="variable", y="value",
                   hue="difference_num",
                   data=rotate_diffs.melt(id_vars=['difference_num']))
    ax[1].set_ylabel("Degrees")
    ax[1].set_title("Rotation")
    sns.despine(offset=10, trim=True, fig=fig)
    fig.savefig(out_file)
