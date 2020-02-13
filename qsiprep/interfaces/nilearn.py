#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Image tools interfaces
~~~~~~~~~~~~~~~~~~~~~~


"""
import os
from tempfile import NamedTemporaryFile
import nibabel as nb
import numpy as np
from scipy import ndimage
from scipy.ndimage.morphology import binary_fill_holes
from skimage import morphology as sim
from dipy.segment.threshold import otsu
from sklearn.preprocessing import robust_scale, power_transform

from nilearn.masking import compute_epi_mask, _post_process_mask
from nilearn.image import (concat_imgs, load_img, new_img_like, math_img, iter_img)
from nilearn.plotting import plot_epi

from nipype import logging
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    traits, isdefined, TraitedSpec, BaseInterfaceInputSpec,
    File, InputMultiPath, SimpleInterface
)

LOGGER = logging.getLogger('nipype.interface')


class MaskEPIInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiPath(File(exists=True), mandatory=True,
                              desc='input EPI or list of files')
    lower_cutoff = traits.Float(0.2, usedefault=True)
    upper_cutoff = traits.Float(0.85, usedefault=True)
    connected = traits.Bool(True, usedefault=True)
    enhance_t2 = traits.Bool(False, usedefault=True,
                             desc='enhance T2 contrast on image')
    opening = traits.Int(2, usedefault=True)
    closing = traits.Bool(True, usedefault=True)
    fill_holes = traits.Bool(True, usedefault=True)
    exclude_zeros = traits.Bool(False, usedefault=True)
    ensure_finite = traits.Bool(True, usedefault=True)
    target_affine = traits.Either(None, traits.File(exists=True),
                                  default=None, usedefault=True)
    target_shape = traits.Either(None, traits.File(exists=True),
                                 default=None, usedefault=True)
    no_sanitize = traits.Bool(False, usedefault=True)


class MaskEPIOutputSpec(TraitedSpec):
    out_mask = File(exists=True, desc='output mask')


class MaskEPI(SimpleInterface):
    input_spec = MaskEPIInputSpec
    output_spec = MaskEPIOutputSpec

    def _run_interface(self, runtime):

        in_files = self.inputs.in_files

        if self.inputs.enhance_t2:
            in_files = [_enhance_t2_contrast(f, newpath=runtime.cwd)
                        for f in in_files]

        masknii = compute_epi_mask(
            in_files,
            lower_cutoff=self.inputs.lower_cutoff,
            upper_cutoff=self.inputs.upper_cutoff,
            connected=self.inputs.connected,
            opening=self.inputs.opening,
            exclude_zeros=self.inputs.exclude_zeros,
            ensure_finite=self.inputs.ensure_finite,
            target_affine=self.inputs.target_affine,
            target_shape=self.inputs.target_shape
        )

        if self.inputs.closing:
            closed = sim.binary_closing(masknii.get_data().astype(
                np.uint8), sim.ball(1)).astype(np.uint8)
            masknii = masknii.__class__(closed, masknii.affine,
                                        masknii.header)

        if self.inputs.fill_holes:
            filled = binary_fill_holes(masknii.get_data().astype(
                np.uint8), sim.ball(6)).astype(np.uint8)
            masknii = masknii.__class__(filled, masknii.affine,
                                        masknii.header)

        if self.inputs.no_sanitize:
            in_file = self.inputs.in_files
            if isinstance(in_file, list):
                in_file = in_file[0]
            nii = nb.load(in_file)
            qform, code = nii.get_qform(coded=True)
            masknii.set_qform(qform, int(code))
            sform, code = nii.get_sform(coded=True)
            masknii.set_sform(sform, int(code))

        self._results['out_mask'] = fname_presuffix(
            self.inputs.in_files[0], suffix='_mask', newpath=runtime.cwd)
        masknii.to_filename(self._results['out_mask'])
        return runtime


class MergeInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiPath(File(exists=True), mandatory=True,
                              desc='input list of files to merge')
    dtype = traits.Enum('f4', 'f8', 'u1', 'u2', 'u4', 'i2', 'i4',
                        usedefault=True, desc='numpy dtype of output image')
    header_source = File(exists=True, desc='a Nifti file from which the header should be copied')
    compress = traits.Bool(True, usedefault=True, desc='Use gzip compression on .nii output')
    is_dwi = traits.Bool(True, usedefault=True, desc='if True, negative values are set to zero')


class MergeOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output merged file')


class Merge(SimpleInterface):
    input_spec = MergeInputSpec
    output_spec = MergeOutputSpec

    def _run_interface(self, runtime):
        ext = '.nii.gz' if self.inputs.compress else '.nii'
        self._results['out_file'] = fname_presuffix(
            self.inputs.in_files[0], suffix='_merged' + ext, newpath=runtime.cwd, use_ext=False)
        new_nii = concat_imgs(self.inputs.in_files, dtype=self.inputs.dtype)

        if isdefined(self.inputs.header_source):
            src_hdr = nb.load(self.inputs.header_source).header
            new_nii.header.set_xyzt_units(t=src_hdr.get_xyzt_units()[-1])
            new_nii.header.set_zooms(list(new_nii.header.get_zooms()[:3]) +
                                     [src_hdr.get_zooms()[3]])
        if self.inputs.is_dwi:
            new_nii = nb.Nifti1Image(np.abs(new_nii.get_data()), new_nii.affine, new_nii.header)

        new_nii.to_filename(self._results['out_file'])

        return runtime


class _EnhanceAndSkullstripB0InputSpec(BaseInterfaceInputSpec):
    b0_file = File(exists=True, mandatory=True)
    t1_mask = File(exists=True, mandatory=True)


class _EnhanceAndSkullstripB0OutputSpec(TraitedSpec):
    mask_file = File(exists=True)
    bias_corrected_file = File(exists=True)
    enhanced_file = File(exists=True)
    skull_stripped_file = File(exists=True)
    plotting_mask_file = File(exists=True)


class EnhanceAndSkullstripB0(SimpleInterface):
    input_spec = _EnhanceAndSkullstripB0InputSpec
    output_spec = _EnhanceAndSkullstripB0OutputSpec

    def _run_interface(self, runtime):
        input_img = nb.squeeze_image(load_img(self.inputs.b0_file))
        t1_mask = nb.squeeze_image(load_img(self.inputs.t1_mask))
        t1_mask_data = t1_mask.get_fdata()
        t1_brain_voxels = t1_mask_data.sum()

        # Get a mask. Choose a good method depending on the resolution
        voxel_size = np.array(input_img.header.get_zooms()[:3])
        if np.any(voxel_size > 3.5):
            LOGGER.warning("Using simple EPI masking due to large voxel size")
            mask_img = compute_epi_mask(input_img)
        else:
            mask_img = watershed_refined_b0_mask(input_img, show_plot=False,
                                                 cwd=runtime.cwd)

        # The brain mask should occupy a similar size as the t1 mask
        input_img = nb.squeeze_image(load_img(self.inputs.b0_file))
        min_size = t1_brain_voxels * .6
        max_size = t1_brain_voxels * 1.4
        mask_voxels = mask_img.get_fdata().sum()
        if mask_voxels < min_size or mask_voxels > max_size:
            LOGGER.warning("Masking appears to have failed. Using a backup method")
            mask_img = compute_epi_mask(input_img)
            mask_voxels = mask_img.get_fdata().sum()
            if mask_voxels < min_size or mask_voxels > max_size:
                LOGGER.warning('Unable to compute a reasonable mask on this b=0 image,'
                               ' using the t1 mask')
        plotting_data = mask_img.get_fdata() + t1_mask_data
        binary_data = (plotting_data > 0).astype(np.uint8)
        plotting_img = new_img_like(input_img, plotting_data)
        binary_img = new_img_like(input_img, binary_data)
        out_mask = fname_presuffix(self.inputs.b0_file, suffix='_mask', newpath=runtime.cwd)
        plotting_mask = fname_presuffix(self.inputs.b0_file, suffix='_plottingmask',
                                        newpath=runtime.cwd)
        binary_img.to_filename(out_mask)
        plotting_img.to_filename(plotting_mask)
        self._results['mask_file'] = out_mask
        self._results['plotting_mask_file'] = plotting_mask

        # Make a smoothed mask for N4
        dilated_mask = ndimage.binary_dilation(mask_img.get_fdata().astype(np.int),
                                               structure=sim.cube(3))
        smoothed_dilated_mask = ndimage.gaussian_filter(dilated_mask.astype(np.float), sigma=3)
        weight_img = new_img_like(input_img, smoothed_dilated_mask)

        # Find a bias field and correct the original input
        masked_input = math_img('np.clip(b0_img * mask_weight, 0, None)',
                                b0_img=input_img, mask_weight=weight_img)
        _, bias_img = biascorrect(masked_input, cwd=runtime.cwd)
        bias_corrected = math_img('np.nan_to_num(orig / biasfield)',
                                  orig=input_img, biasfield=bias_img)

        # Apply the bias field correction to the whole image
        out_bias_corrected = fname_presuffix(self.inputs.b0_file, suffix='_unbiased',
                                             newpath=runtime.cwd)
        bias_corrected.to_filename(out_bias_corrected)
        self._results['bias_corrected_file'] = out_bias_corrected

        # Sharpen the bias-corrected image
        out_enhanced = fname_presuffix(self.inputs.b0_file, suffix='_unbiasedsharpened',
                                       newpath=runtime.cwd)
        enhanced = run_imagemath(bias_corrected, 'Sharpen', [], cwd=runtime.cwd)
        enhanced.to_filename(out_enhanced)
        self._results['enhanced_file'] = out_enhanced

        # Apply the soft mask to the bias-corrected, sharpened image
        skullstripped_img = math_img('weights * enhanced',
                                     weights=weight_img, enhanced=enhanced)
        out_skullstripped = fname_presuffix(self.inputs.b0_file, suffix='_brain',
                                            newpath=runtime.cwd)
        skullstripped_img.to_filename(out_skullstripped)
        self._results['skull_stripped_file'] = out_skullstripped

        return runtime


class _EnhanceB0InputSpec(BaseInterfaceInputSpec):
    b0_file = File(exists=True, mandatory=True)


class _EnhanceB0OutputSpec(TraitedSpec):
    mask_file = File(exists=True)
    bias_corrected_file = File(exists=True)
    enhanced_file = File(exists=True)


class EnhanceB0(SimpleInterface):
    input_spec = _EnhanceB0InputSpec
    output_spec = _EnhanceB0OutputSpec

    def _run_interface(self, runtime):
        input_img = nb.squeeze_image(load_img(self.inputs.b0_file))
        bias_corrected, bias_img = biascorrect(input_img, cwd=runtime.cwd)
        out_bias_corrected = fname_presuffix(self.inputs.b0_file, suffix='_unbiased',
                                             newpath=runtime.cwd)
        bias_corrected.to_filename(out_bias_corrected)
        self._results['bias_corrected_file'] = out_bias_corrected

        # Sharpen the bias-corrected image
        out_enhanced = fname_presuffix(self.inputs.b0_file, suffix='_unbiasedsharpened',
                                       newpath=runtime.cwd)
        enhanced = run_imagemath(bias_corrected, 'Sharpen', [], cwd=runtime.cwd)
        enhanced.to_filename(out_enhanced)
        self._results['enhanced_file'] = out_enhanced

        return runtime


class _MaskB0SeriesInputSpec(BaseInterfaceInputSpec):
    b0_series = File(exists=True, mandatory=True)


class _MaskB0SeriesOutputSpec(TraitedSpec):
    mask_file = File(exists=True)


class MaskB0Series(SimpleInterface):
    input_spec = _MaskB0SeriesInputSpec
    output_spec = _MaskB0SeriesOutputSpec

    def _run_interface(self, runtime):
        b0_img = load_img(self.inputs.b0_series)
        big_voxels = np.any(np.array(b0_img.header.get_zooms()) > 4.0)
        masks = []
        for img in iter_img(b0_img):
            if big_voxels:
                mask, _, _ = calculate_gradmax_b0_mask(img, show_plot=False, cwd=runtime.cwd)
            else:
                mask = watershed_refined_b0_mask(img, show_plot=False, cwd=runtime.cwd)
            masks.append(mask)

        if len(masks) > 1:
            all_masks = concat_imgs(masks)
            final_mask = new_img_like(b0_img, all_masks.get_fdata().sum(3))
        else:
            final_mask = masks[0]

        output_fname = fname_presuffix(self.inputs.b0_series, suffix="_masked")
        output_nii = new_img_like(b0_img, final_mask.get_fdata() >= 0.5)
        output_nii.to_filename(output_fname)
        self._results['mask_file'] = output_fname

        return runtime


def _enhance_t2_contrast(in_file, newpath=None, offset=0.5):
    """
    Performs a logarithmic transformation of intensity that
    effectively splits brain and background and makes the
    overall distribution more Gaussian.
    """
    out_file = fname_presuffix(in_file, suffix='_t1enh',
                               newpath=newpath)
    nii = nb.load(in_file)
    data = nii.get_data()
    maxd = data.max()
    newdata = np.log(offset + data / maxd)
    newdata -= newdata.min()
    newdata *= maxd / newdata.max()
    nii = nii.__class__(newdata, nii.affine, nii.header)
    nii.to_filename(out_file)
    return out_file


def run_imagemath(nii, op, args, copy_input_header=True, cwd=None):
    tmpf_in = NamedTemporaryFile(dir=cwd)
    tmpf_out = NamedTemporaryFile(dir=cwd)
    in_fname = tmpf_in.name + ".nii.gz"
    out_fname = tmpf_out.name + ".nii.gz"
    nii.to_filename(in_fname)
    imath_cmd = ["ImageMath", "3", out_fname, op, in_fname] + args
    os.system(' '.join(imath_cmd))
    new_img = load_img(out_fname)
    tmpf_in.close()
    tmpf_out.close()
    if copy_input_header:
        out_nii = nb.Nifti1Image(new_img.get_fdata(), nii.affine, nii.header)
    else:
        out_nii = new_img
    return out_nii


def biascorrect(nii, copy_input_header=True, cwd=None):
    tmpf_in = NamedTemporaryFile(dir=cwd)
    tmpf_out = NamedTemporaryFile(dir=cwd)
    in_fname = tmpf_in.name + ".nii.gz"
    out_fname = tmpf_out.name + ".nii.gz"
    out_bias_fname = tmpf_out.name + "_bias.nii.gz"
    nii.to_filename(in_fname)
    cmd = ["N3BiasFieldCorrection", "3", in_fname, out_fname, "4", "none",
           "50", "4", out_bias_fname]
    os.system(' '.join(cmd))
    new_img = load_img(out_fname)
    bias_img = load_img(out_bias_fname)
    tmpf_in.close()
    tmpf_out.close()
    if copy_input_header:
        out_nii = nb.Nifti1Image(new_img.get_fdata(), nii.affine, nii.header)
        out_bias = nb.Nifti1Image(bias_img.get_fdata(), nii.affine, nii.header)
    else:
        out_nii = new_img
        out_bias = bias_img
    return out_nii, out_bias


def calculate_gradmax_b0_mask(b0_nii, show_plot=False, quantile_max=0.8, pad_size=10,
                              cwd=None):
    """Robustly finds a brain mask from a low-res b=0 image.

    The steps for finding a mask for a b=0 image

      1. Remove spiky outliers with a median filter
      2. Non-aggressively bias correct the image using N3
      3. Calculate the magnitude of the spatial gradient
      4. Clip the intensity values and rescale them using a Box-Cox transform
      5. Calculate a foreground threshold using Otsu's Method
      6. Try a series of orders for opening. Select the order that maximizes the gradient
         from (3) at the edge of the opened mask.

    **Returns**

        mask_nii: spatial image
            binary gradient-optimizing mask
        scaled_nii: spatial image
            robust scaled image for brain extraction
        gradient_nii: spatial image
            gradient image
    """
    total_voxels = np.prod(b0_nii.shape)
    if pad_size:
        padded_nii = run_imagemath(b0_nii, 'PadImage', [str(pad_size)], copy_input_header=False,
                                   cwd=cwd)
    else:
        padded_nii = b0_nii

    # First apply a median filter to the data
    footprint = sim.cube(3)
    data = padded_nii.get_fdata()
    mask = data > 0
    median_filt = ndimage.median_filter(data, footprint=footprint) * mask
    median_nii = new_img_like(padded_nii, median_filt)
    bc_nii, _ = biascorrect(median_nii, cwd=cwd)

    # Calculate the gradient on the bias-corrected, median filtered image
    grad_nii = run_imagemath(bc_nii, "Grad", ['0'], cwd=cwd)
    grad_data = grad_nii.get_fdata()

    # Make an edge map
    values = np.abs(data[mask].reshape(-1, 1))
    clipped = robust_scale(values, quantile_range=(0, quantile_max),
                           with_centering=False)
    scaled = np.clip(
        power_transform(clipped, method='box-cox', standardize=False).squeeze(),
        0, None)
    cutoff = otsu(scaled)
    binary = scaled > cutoff
    data[mask] = binary
    scaled_image = data.copy()
    scaled_image[mask] = scaled

    # Make a distance-weighted gradient
    maurer_abs = new_img_like(
        padded_nii,
        np.abs(run_imagemath(new_img_like(padded_nii, data),
                             'MaurerDistance', [], cwd=cwd).get_fdata()))
    weighted_edges = math_img('1/(img+1)**2 * grad', img=maurer_abs, grad=grad_nii)
    grad_data = weighted_edges.get_fdata()

    # Send it out for post processing
    edge_scores = []
    opening_values = np.array([2, 4, 6, 8, 10, 12], dtype=np.int)
    opened_masks = []
    selected_voxels = []
    for opening_test in opening_values:
        processed_mask, _ = _post_process_mask(data, b0_nii.affine, opening=opening_test)
        # Make a mask around the edge of the mask
        dilated_mask = ndimage.binary_dilation(processed_mask)
        eroded_mask = ndimage.binary_erosion(processed_mask)
        mask_edge = dilated_mask ^ eroded_mask
        opened_masks.append(processed_mask)
        selected_voxels.append(processed_mask.sum() / total_voxels * 100)
        # How many edges are captured by the mask edge?
        edge_scores.append(grad_data[mask_edge].mean())

    best_mask = np.argmax(edge_scores)
    processed_mask = opened_masks[best_mask]

    if best_mask.sum() < 0.1 * total_voxels:
        LOGGER.warning("Degenerate Mask case. Using compute_epi_mask")
        epi_mask = compute_epi_mask(new_img_like(padded_nii, scaled_image))
        processed_mask = epi_mask.get_fdata().astype(np.uint8)

    if pad_size:
        processed_mask = processed_mask[pad_size:-pad_size,
                                        pad_size:-pad_size,
                                        pad_size:-pad_size]
        scaled_image = scaled_image[pad_size:-pad_size,
                                    pad_size:-pad_size,
                                    pad_size:-pad_size]
        grad_data = grad_data[pad_size:-pad_size,
                              pad_size:-pad_size,
                              pad_size:-pad_size]

    mask_img = new_img_like(b0_nii, processed_mask)
    scaled_img = new_img_like(b0_nii, scaled_image)
    grad_img = new_img_like(b0_nii, grad_data)
    if show_plot:
        import matplotlib.pyplot as plt
        print("picked opening=", opening_values[best_mask])
        plot_epi(padded_nii, display_mode='z', cut_coords=10, title='Input Image')
        plot_epi(median_nii, display_mode='z', cut_coords=10, title='Median Filtered')
        plot_epi(bc_nii, display_mode='z', cut_coords=10, title='Bias Corrected')
        fig, ax = plt.subplots(ncols=3)
        ax[0].hist(scaled, bins=256)
        ax[0].axvline(cutoff, color='k')
        ax[0].set_title("Step 2: BoxCox")
        ax[1].plot(opening_values, edge_scores, 'o-')
        ax[1].set_title("Mean Boundary Gradient")
        ax[2].plot(selected_voxels, 'o-')
        ax[2].set_title("Mask Size (% FOV)")
        vmax = np.percentile(values, quantile_max * 100)
        display = plot_epi(b0_nii, cmap='gray', vmax=vmax, display_mode='z',
                           cut_coords=10)
        display.add_contours(mask_img, linewidths=2)
        disp2 = plot_epi(grad_img, cmap='gray', resampling_interpolation='nearest',
                         display_mode='z', cut_coords=10)
        disp2.add_contours(mask_img, linewidths=0.5)
    return mask_img, scaled_img, grad_img


def watershed_refined_b0_mask(b0_nii, show_plot=False, pad_size=10, quantile_max=0.8,
                              ribbon_size=5, cwd=None):
    """Refine the boundary of a mask using the watershed algorithm.

    **Returns**

        mask_nii: spatial image
            binary gradient-optimizing mask
        weighting_mask: spatial image
            smoothed mask for use with N4
    """

    initial_mask_nii, initial_scaled_nii, _ = \
        calculate_gradmax_b0_mask(b0_nii, show_plot=show_plot, quantile_max=quantile_max, cwd=cwd)

    if pad_size:
        initial_mask_nii = run_imagemath(initial_mask_nii, 'PadImage', [str(pad_size)],
                                         copy_input_header=False, cwd=cwd)
        initial_scaled_nii = run_imagemath(initial_scaled_nii, 'PadImage', [str(pad_size)],
                                           copy_input_header=False, cwd=cwd)

    mask_image = initial_mask_nii.get_fdata().astype(np.uint8)
    scaled_image = initial_scaled_nii.get_fdata()

    # Find a ribbon to detect the boundary in
    morph_size = ribbon_size // 2 if (ribbon_size // 2) % 2 == 1 else ribbon_size // 2 + 1
    eroded_mask = ndimage.binary_erosion(mask_image, structure=sim.cube(morph_size))
    dilated_mask = ndimage.binary_dilation(mask_image, structure=sim.cube(morph_size))
    definitely_outer = ndimage.binary_dilation(
        dilated_mask, structure=sim.cube(morph_size)) ^ dilated_mask
    ribbon_mask = (dilated_mask ^ eroded_mask)

    # Down-weight data deep in the mask
    inner_weights = ndimage.gaussian_filter(eroded_mask.astype(np.float), sigma=morph_size)
    inner_weights = 1. - inner_weights / inner_weights.max()

    # Down-weight data as it gets far from the mask
    maurer = run_imagemath(new_img_like(initial_mask_nii, eroded_mask),
                           'MaurerDistance', [], cwd=cwd).get_fdata()
    outside_mask_distance = np.clip(maurer, 2, None)
    outer_weights = 1 / outside_mask_distance

    morph_grad_weights = inner_weights * outer_weights
    smoothed_weights = ndimage.gaussian_filter(morph_grad_weights, sigma=morph_size/2)
    smoothed_weights = smoothed_weights / smoothed_weights.max()

    # Calculate the morphological gradient
    morph_grad = ndimage.morphological_gradient(scaled_image,
                                                footprint=sim.cube(3)) \
        * smoothed_weights

    markers = select_markers_for_rw(morph_grad, eroded_mask, ribbon_mask, definitely_outer)
    watershed_seg = sim.watershed(morph_grad, markers)
    ws_mask = watershed_seg == 2

    if pad_size:
        ws_mask = ws_mask[pad_size:-pad_size,
                          pad_size:-pad_size,
                          pad_size:-pad_size]
        morph_grad = morph_grad[pad_size:-pad_size,
                                pad_size:-pad_size,
                                pad_size:-pad_size]

    # Ensure headers are the same as the input image
    mask_img = new_img_like(b0_nii, ws_mask)
    grad_img = new_img_like(b0_nii, morph_grad)

    if show_plot:
        display = plot_epi(b0_nii, cmap='gray', display_mode='z',
                           cut_coords=10)
        display.add_contours(mask_img, linewidths=2)
        disp2 = plot_epi(grad_img, cmap='gray', resampling_interpolation='nearest',
                         display_mode='z', cut_coords=10)
        disp2.add_contours(mask_img, linewidths=0.5)
    return mask_img


def select_markers_for_rw(image, inner_mask, empty_mask, outer_mask,
                          sample_proportion=.5):
    markers = np.zeros_like(image) - 1.
    use_as_inner_marker = np.random.rand(inner_mask.sum()) < sample_proportion
    use_as_outer_marker = np.random.rand(outer_mask.sum()) < sample_proportion
    markers[inner_mask > 0] = use_as_inner_marker.astype(np.int) * 2
    markers[outer_mask > 0] = use_as_outer_marker.astype(np.int) * 1
    markers[empty_mask > 0] = 0

    return markers
