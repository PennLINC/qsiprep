#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Interfaces to deal with the various types of fieldmap sources

    .. testsetup::

        >>> tmpdir = getfixture('tmpdir')
        >>> tmp = tmpdir.chdir() # changing to a temporary directory
        >>> nb.Nifti1Image(np.zeros((90, 90, 60)), None, None).to_filename(
        ...     tmpdir.join('epi.nii.gz').strpath)


"""
import os.path as op
import json
from collections import defaultdict
import numpy as np
import nibabel as nb
from nipype import logging
from nipype.utils.filemanip import fname_presuffix, split_filename
from nipype.interfaces.base import (
    BaseInterfaceInputSpec, TraitedSpec, File, isdefined, traits,
    SimpleInterface, InputMultiObject, OutputMultiObject)
from .images import to_lps
from .reports import topup_selection_to_report
from nilearn.image import load_img, index_img, concat_imgs, iter_img

LOGGER = logging.getLogger('nipype.interface')
CRITICAL_KEYS = ["PhaseEncodingDirection", "TotalReadoutTime", "EffectiveEchoSpacing"]


class B0RPEFieldmapInputSpec(BaseInterfaceInputSpec):
    b0_file = InputMultiObject(File(exists=True))
    output_3d_images = traits.Bool(False, usedefault=True)
    max_num_b0s = traits.Int(3, usedefault=True)
    orientation = traits.Enum('LPS', 'LAS', default='LPS', usedefault=True)
    b0_threshold = traits.Int(100, usedefault=True)


class B0RPEFieldmapOutputSpec(TraitedSpec):
    fmap_file = OutputMultiObject(File(exists=True))
    fmap_info = OutputMultiObject(File(exists=True))
    fmap_report = traits.Str()


class B0RPEFieldmap(SimpleInterface):
    """Prepares b=0 EPI fieldmaps to be used for distortion correction.
    Some siemens scanners are unable to make a b=0 image by itself, and will produce
    a dwi series (with bvals and bvecs). This interface removes the b>0 volumes and
    writes the b=0 images in the resuested orientation (LAS+ for FSL, or LPS+ for
    everything else).

    **Inputs**
        b0_file: str
            List of paths to b=0 epi fieldmaps in fmaps/ or an RPE series in dwi/
        output_3d_images: bool
            Write outputs as multiple 3d images
        max_num_b0s: int
            Include a maximum number of b=0 images in the outputs
        orientation: str
            Write the outputs in either 'LAS' or 'LPS' orientation

    """
    input_spec = B0RPEFieldmapInputSpec
    output_spec = B0RPEFieldmapOutputSpec

    def _run_interface(self, runtime):

        # Get b=0 images from all the inputs
        b0_series, b0_indices, original_files = load_epi_dwi_fieldmaps(
            self.inputs.b0_file, self.inputs.b0_threshold)

        # Only get the requested number of images
        _, fmap_imain, fmap_report, _ = topup_inputs_from_4d_file(
            b0_series, b0_indices, original_files, image_source="EPI fieldmap",
            max_per_spec=self.inputs.max_num_b0s)
        LOGGER.info(fmap_report)

        # Get b=0 images and metadata from all the input images
        b0_fieldmap_metadata = []
        for image_path in set(original_files):
            pth, fname, _ = split_filename(image_path)
            original_json = op.join(pth, fname) + ".json"
            b0_fieldmap_metadata.append(original_json)

        # Warn the user if the metadata does not match
        merged_metadata = _merge_metadata(b0_fieldmap_metadata)
        merged_b0s = to_lps(fmap_imain, tuple(self.inputs.orientation))
        # Output just one 3/4d image and a sidecar
        if not self.inputs.output_3d_images:
            # Save the conformed fmap
            output_fmap = fname_presuffix(self.inputs.b0_file[0], suffix="conform",
                                          newpath=runtime.cwd)
            output_json = fname_presuffix(output_fmap, use_ext=False, suffix=".json")
            fmap_imain.to_filename(output_fmap)
            with open(output_json, "w") as sidecar:
                json.dump(merged_metadata, sidecar)
            self._results['fmap_file'] = output_fmap
            self._results['fmap_info'] = output_json
            return runtime

        image_list = []
        json_list = []
        for imgnum, img in enumerate(iter_img(merged_b0s)):

            # Save the conformed fmap and metadata
            output_fmap = fname_presuffix(self.inputs.b0_file[0],
                                          suffix="%s_%03d" % (self.inputs.orientation, imgnum),
                                          newpath=runtime.cwd)
            output_json = fname_presuffix(output_fmap, use_ext=False, suffix=".json")
            with open(output_json, "w") as sidecar:
                json.dump(merged_metadata, sidecar)
            img.to_filename(output_fmap)

            # Append to lists
            image_list.append(output_fmap)
            json_list.append(output_json)

        self._results['fmap_file'] = image_list
        self._results['fmap_info'] = json_list
        return runtime


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


class FieldEnhanceInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input fieldmap')
    in_mask = File(exists=True, desc='brain mask')
    in_magnitude = File(exists=True, desc='input magnitude')
    unwrap = traits.Bool(False, usedefault=True, desc='run phase unwrap')
    despike = traits.Bool(True, usedefault=True, desc='run despike filter')
    bspline_smooth = traits.Bool(True, usedefault=True, desc='run 3D bspline smoother')
    mask_erode = traits.Int(1, usedefault=True, desc='mask erosion iterations')
    despike_threshold = traits.Float(0.2, usedefault=True, desc='mask erosion iterations')
    num_threads = traits.Int(1, usedefault=True, nohash=True, desc='number of jobs')


class FieldEnhanceOutputSpec(TraitedSpec):
    out_file = File(desc='the output fieldmap')
    out_unwrapped = File(desc='unwrapped fieldmap')


class FieldEnhance(SimpleInterface):
    """
    The FieldEnhance interface wraps a workflow to massage the input fieldmap
    and return it masked, despiked, etc.
    """
    input_spec = FieldEnhanceInputSpec
    output_spec = FieldEnhanceOutputSpec

    def _run_interface(self, runtime):
        from scipy import ndimage as sim

        fmap_nii = nb.load(self.inputs.in_file)
        data = np.squeeze(fmap_nii.get_data().astype(np.float32))

        # Despike / denoise (no-mask)
        if self.inputs.despike:
            data = _despike2d(data, self.inputs.despike_threshold)

        mask = None
        if isdefined(self.inputs.in_mask):
            masknii = nb.load(self.inputs.in_mask)
            mask = masknii.get_data().astype(np.uint8)

            # Dilate mask
            if self.inputs.mask_erode > 0:
                struc = sim.iterate_structure(sim.generate_binary_structure(3, 2), 1)
                mask = sim.binary_erosion(
                    mask, struc,
                    iterations=self.inputs.mask_erode
                ).astype(np.uint8)  # pylint: disable=no-member

        self._results['out_file'] = fname_presuffix(
            self.inputs.in_file, suffix='_enh', newpath=runtime.cwd)
        datanii = nb.Nifti1Image(data, fmap_nii.affine, fmap_nii.header)

        if self.inputs.unwrap:
            data = _unwrap(data, self.inputs.in_magnitude, mask)
            self._results['out_unwrapped'] = fname_presuffix(
                self.inputs.in_file, suffix='_unwrap', newpath=runtime.cwd)
            nb.Nifti1Image(data, fmap_nii.affine, fmap_nii.header).to_filename(
                self._results['out_unwrapped'])

        if not self.inputs.bspline_smooth:
            datanii.to_filename(self._results['out_file'])
            return runtime
        else:
            from ..utils import bspline as fbsp
            from statsmodels.robust.scale import mad

            # Fit BSplines (coarse)
            bspobj = fbsp.BSplineFieldmap(datanii, weights=mask,
                                          njobs=self.inputs.num_threads)
            bspobj.fit()
            smoothed1 = bspobj.get_smoothed()

            # Manipulate the difference map
            diffmap = data - smoothed1.get_data()
            sderror = mad(diffmap[mask > 0])
            LOGGER.info('SD of error after B-Spline fitting is %f', sderror)
            errormask = np.zeros_like(diffmap)
            errormask[np.abs(diffmap) > (10 * sderror)] = 1
            errormask *= mask

            nslices = 0
            try:
                errorslice = np.squeeze(np.argwhere(errormask.sum(0).sum(0) > 0))
                nslices = errorslice[-1] - errorslice[0]
            except IndexError:  # mask is empty, do not refine
                pass

            if nslices > 1:
                diffmapmsk = mask[..., errorslice[0]:errorslice[-1]]
                diffmapnii = nb.Nifti1Image(
                    diffmap[..., errorslice[0]:errorslice[-1]] * diffmapmsk,
                    datanii.affine, datanii.header)

                bspobj2 = fbsp.BSplineFieldmap(diffmapnii, knots_zooms=[24., 24., 4.],
                                               njobs=self.inputs.num_threads)
                bspobj2.fit()
                smoothed2 = bspobj2.get_smoothed().get_data()

                final = smoothed1.get_data().copy()
                final[..., errorslice[0]:errorslice[-1]] += smoothed2
            else:
                final = smoothed1.get_data()

            nb.Nifti1Image(final, datanii.affine, datanii.header).to_filename(
                self._results['out_file'])

        return runtime


class FieldToRadSInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input fieldmap')
    fmap_range = traits.Float(desc='range of input field map')


class FieldToRadSOutputSpec(TraitedSpec):
    out_file = File(desc='the output fieldmap')
    fmap_range = traits.Float(desc='range of input field map')


class FieldToRadS(SimpleInterface):
    """
    The FieldToRadS converts from arbitrary units to rad/s
    """
    input_spec = FieldToRadSInputSpec
    output_spec = FieldToRadSOutputSpec

    def _run_interface(self, runtime):
        fmap_range = None
        if isdefined(self.inputs.fmap_range):
            fmap_range = self.inputs.fmap_range
        self._results['out_file'], self._results['fmap_range'] = _torads(
            self.inputs.in_file, fmap_range, newpath=runtime.cwd)
        return runtime


class FieldToHzInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input fieldmap')
    range_hz = traits.Float(mandatory=True, desc='range of input field map')


class FieldToHzOutputSpec(TraitedSpec):
    out_file = File(desc='the output fieldmap')


class FieldToHz(SimpleInterface):
    """
    The FieldToHz converts from arbitrary units to Hz
    """
    input_spec = FieldToHzInputSpec
    output_spec = FieldToHzOutputSpec

    def _run_interface(self, runtime):
        self._results['out_file'] = _tohz(
            self.inputs.in_file, self.inputs.range_hz, newpath=runtime.cwd)
        return runtime


class Phasediff2FieldmapInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input fieldmap')
    metadata = traits.Dict(mandatory=True, desc='BIDS metadata dictionary')


class Phasediff2FieldmapOutputSpec(TraitedSpec):
    out_file = File(desc='the output fieldmap')


class Phasediff2Fieldmap(SimpleInterface):
    """
    Convert a phase difference map into a fieldmap in Hz
    """
    input_spec = Phasediff2FieldmapInputSpec
    output_spec = Phasediff2FieldmapOutputSpec

    def _run_interface(self, runtime):
        self._results['out_file'] = phdiff2fmap(
            self.inputs.in_file,
            _delta_te(self.inputs.metadata),
            newpath=runtime.cwd)
        return runtime


class Phases2FieldmapInputSpec(BaseInterfaceInputSpec):
    phase_files = InputMultiObject(
        File(exists=True), mandatory=True, desc='list of phase1, phase2 files')
    metadatas = traits.List(
        traits.Dict, mandatory=True, desc='list of phase1, phase2 metadata dicts')


class Phases2FieldmapOutputSpec(TraitedSpec):
    out_file = File(desc='the output fieldmap')
    phasediff_metadata = traits.Dict(desc='the phasediff metadata')


class Phases2Fieldmap(SimpleInterface):
    """
    Convert a phase1, phase2 into a difference map
    """
    input_spec = Phases2FieldmapInputSpec
    output_spec = Phases2FieldmapOutputSpec

    def _run_interface(self, runtime):
        # Get the echo times
        fmap_file, merged_metadata = phases2fmap(self.inputs.phase_files, self.inputs.metadatas,
                                                 newpath=runtime.cwd)
        self._results['phasediff_metadata'] = merged_metadata
        self._results['out_file'] = fmap_file
        return runtime


def phases2fmap(phase_files, metadatas, newpath=None):
    """Calculates a phasediff from two phase images. Assumes monopolar
    readout. """
    import numpy as np
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix
    from copy import deepcopy

    phasediff_file = fname_presuffix(phase_files[0], suffix='_phasediff', newpath=newpath)
    echo_times = [meta.get("EchoTime") for meta in metadatas]
    if None in echo_times or echo_times[0] == echo_times[1]:
        raise RuntimeError()
    # Determine the order of subtraction
    short_echo_index = echo_times.index(min(echo_times))
    long_echo_index = echo_times.index(max(echo_times))

    short_phase_image = phase_files[short_echo_index]
    long_phase_image = phase_files[long_echo_index]

    image0 = nb.load(short_phase_image)
    phase0 = image0.get_fdata()
    image1 = nb.load(long_phase_image)
    phase1 = image1.get_fdata()

    def rescale_image(img):
        if np.any(img < -128):
            # This happens sometimes on 7T fieldmaps
            LOGGER.info("Found negative values in phase image: rescaling")
            imax = img.max()
            imin = img.min()
            scaled = 2 * ((img - imin) / (imax - imin) - 0.5)
            return np.pi * scaled
        mask = img > 0
        imax = img.max()
        imin = img.min()
        max_check = imax - 4096
        if np.abs(max_check) > 10 or np.abs(imin) > 10:
            LOGGER.warning("Phase image may be scaled incorrectly: check results")
        return mask * (img / 2048 * np.pi - np.pi)

    # Calculate fieldmaps
    rad0 = rescale_image(phase0)
    rad1 = rescale_image(phase1)
    a = np.cos(rad0)
    b = np.sin(rad0)
    c = np.cos(rad1)
    d = np.sin(rad1)
    fmap = -np.arctan2(b * c - a * d, a * c + b * d)

    phasediff_nii = nb.Nifti1Image(fmap, image0.affine)
    phasediff_nii.set_data_dtype(np.float32)
    phasediff_nii.to_filename(phasediff_file)

    merged_metadata = deepcopy(metadatas[0])
    del merged_metadata['EchoTime']
    merged_metadata['EchoTime1'] = float(echo_times[short_echo_index])
    merged_metadata['EchoTime2'] = float(echo_times[long_echo_index])

    return phasediff_file, merged_metadata


def _despike2d(data, thres, neigh=None):
    """
    despiking as done in FSL fugue
    """

    if neigh is None:
        neigh = [-1, 0, 1]
    nslices = data.shape[-1]

    for k in range(nslices):
        data2d = data[..., k]

        for i in range(data2d.shape[0]):
            for j in range(data2d.shape[1]):
                vals = []
                thisval = data2d[i, j]
                for ii in neigh:
                    for jj in neigh:
                        try:
                            vals.append(data2d[i + ii, j + jj])
                        except IndexError:
                            pass
                vals = np.array(vals)
                patch_range = vals.max() - vals.min()
                patch_med = np.median(vals)

                if (patch_range > 1e-6 and
                        (abs(thisval - patch_med) / patch_range) > thres):
                    data[i, j, k] = patch_med
    return data


def _unwrap(fmap_data, mag_file, mask=None):
    from math import pi
    from nipype.interfaces.fsl import PRELUDE
    magnii = nb.load(mag_file)

    if mask is None:
        mask = np.ones_like(fmap_data, dtype=np.uint8)

    fmapmax = max(abs(fmap_data[mask > 0].min()), fmap_data[mask > 0].max())
    fmap_data *= pi / fmapmax

    nb.Nifti1Image(fmap_data, magnii.affine).to_filename('fmap_rad.nii.gz')
    nb.Nifti1Image(mask, magnii.affine).to_filename('fmap_mask.nii.gz')
    nb.Nifti1Image(magnii.get_data(), magnii.affine).to_filename('fmap_mag.nii.gz')

    # Run prelude
    res = PRELUDE(phase_file='fmap_rad.nii.gz',
                  magnitude_file='fmap_mag.nii.gz',
                  mask_file='fmap_mask.nii.gz').run()

    unwrapped = nb.load(res.outputs.unwrapped_phase_file).get_data() * (fmapmax / pi)
    return unwrapped


def get_ees(in_meta, in_file=None):
    """
    Calculate the *effective echo spacing* :math:`t_\\text{ees}`
    for an input :abbr:`EPI (echo-planar imaging)` scan.


    There are several procedures to calculate the effective
    echo spacing. The basic one is that an ``EffectiveEchoSpacing``
    field is set in the JSON sidecar. The following examples
    use an ``'epi.nii.gz'`` file-stub which has 90 pixels in the
    j-axis encoding direction.

    >>> meta = {'EffectiveEchoSpacing': 0.00059,
    ...         'PhaseEncodingDirection': 'j-'}
    >>> get_ees(meta)
    0.00059

    If the *total readout time* :math:`T_\\text{ro}` (``TotalReadoutTime``
    BIDS field) is provided, then the effective echo spacing can be
    calculated reading the number of voxels :math:`N_\\text{PE}` along the
    readout direction and the parallel acceleration
    factor of the EPI

      .. math ::

           =  T_\\text{ro} \\,  (N_\\text{PE} / f_\\text{acc} - 1)^{-1}

    where :math:`N_y` is the number of pixels along the phase-encoding direction
    :math:`y`, and :math:`f_\\text{acc}` is the parallel imaging acceleration factor
    (:abbr:`GRAPPA (GeneRalized Autocalibrating Partial Parallel Acquisition)`,
    :abbr:`ARC (Autocalibrating Reconstruction for Cartesian imaging)`, etc.).

    >>> meta = {'TotalReadoutTime': 0.02596,
    ...         'PhaseEncodingDirection': 'j-',
    ...         'ParallelReductionFactorInPlane': 2}
    >>> get_ees(meta, in_file='epi.nii.gz')
    0.00059

    Some vendors, like Philips, store different parameter names
    (see http://dbic.dartmouth.edu/pipermail/mrusers/attachments/\
20141112/eb1d20e6/attachment.pdf):

    >>> meta = {'WaterFatShift': 8.129,
    ...         'MagneticFieldStrength': 3,
    ...         'PhaseEncodingDirection': 'j-',
    ...         'ParallelReductionFactorInPlane': 2}
    >>> get_ees(meta, in_file='epi.nii.gz')
    0.00041602630141921826

    """

    import nibabel as nb
    from qsiprep.interfaces.fmap import _get_pe_index

    # Use case 1: EES is defined
    ees = in_meta.get('EffectiveEchoSpacing', None)
    if ees is not None:
        return ees

    # All other cases require the parallel acc and npe (N vox in PE dir)
    acc = float(in_meta.get('ParallelReductionFactorInPlane', 1.0))
    npe = nb.load(in_file).shape[_get_pe_index(in_meta)]
    etl = npe // acc

    # Use case 2: TRT is defined
    trt = in_meta.get('TotalReadoutTime', None)
    if trt is not None:
        return trt / (etl - 1)

    # Use case 3 (philips scans)
    wfs = in_meta.get('WaterFatShift', None)
    if wfs is not None:
        fstrength = in_meta['MagneticFieldStrength']
        wfd_ppm = 3.4  # water-fat diff in ppm
        g_ratio_mhz_t = 42.57  # gyromagnetic ratio for proton (1H) in MHz/T
        wfs_hz = fstrength * wfd_ppm * g_ratio_mhz_t
        return wfs / (wfs_hz * etl)

    raise ValueError('Unknown effective echo-spacing specification')


def get_trt(in_meta, in_file=None):
    """
    Calculate the *total readout time* for an input
    :abbr:`EPI (echo-planar imaging)` scan.


    There are several procedures to calculate the total
    readout time. The basic one is that a ``TotalReadoutTime``
    field is set in the JSON sidecar. The following examples
    use an ``'epi.nii.gz'`` file-stub which has 90 pixels in the
    j-axis encoding direction.

    >>> meta = {'TotalReadoutTime': 0.02596}
    >>> get_trt(meta)
    0.02596

    If the *effective echo spacing* :math:`t_\\text{ees}`
    (``EffectiveEchoSpacing`` BIDS field) is provided, then the
    total readout time can be calculated reading the number
    of voxels along the readout direction :math:`T_\\text{ro}`
    and the parallel acceleration factor of the EPI :math:`f_\\text{acc}`.

      .. math ::

          T_\\text{ro} = t_\\text{ees} \\, (N_\\text{PE} / f_\\text{acc} - 1)

    >>> meta = {'EffectiveEchoSpacing': 0.00059,
    ...         'PhaseEncodingDirection': 'j-',
    ...         'ParallelReductionFactorInPlane': 2}
    >>> get_trt(meta, in_file='epi.nii.gz')
    0.02596

    Some vendors, like Philips, store different parameter names:

    >>> meta = {'WaterFatShift': 8.129,
    ...         'MagneticFieldStrength': 3,
    ...         'PhaseEncodingDirection': 'j-',
    ...         'ParallelReductionFactorInPlane': 2}
    >>> get_trt(meta, in_file='epi.nii.gz')
    0.018721183563864822

    """

    # Use case 1: TRT is defined
    trt = in_meta.get('TotalReadoutTime', None)
    if trt is not None:
        return trt

    # All other cases require the parallel acc and npe (N vox in PE dir)
    acc = float(in_meta.get('ParallelReductionFactorInPlane', 1.0))
    npe = nb.load(in_file).shape[_get_pe_index(in_meta)]
    etl = npe // acc

    # Use case 2: TRT is defined
    ees = in_meta.get('EffectiveEchoSpacing', None)
    if ees is not None:
        return ees * (etl - 1)

    # Use case 3 (philips scans)
    wfs = in_meta.get('WaterFatShift', None)
    if wfs is not None:
        fstrength = in_meta['MagneticFieldStrength']
        wfd_ppm = 3.4  # water-fat diff in ppm
        g_ratio_mhz_t = 42.57  # gyromagnetic ratio for proton (1H) in MHz/T
        wfs_hz = fstrength * wfd_ppm * g_ratio_mhz_t
        return wfs / wfs_hz

    raise ValueError('Unknown total-readout time specification')


def _get_pe_index(meta):
    pe = meta['PhaseEncodingDirection']
    try:
        return {'i': 0, 'j': 1, 'k': 2}[pe[0]]
    except KeyError:
        raise RuntimeError('"%s" is an invalid PE string' % pe)


def _torads(in_file, fmap_range=None, newpath=None):
    """
    Convert a field map to rad/s units

    If fmap_range is None, the range of the fieldmap
    will be automatically calculated.

    Use fmap_range=0.5 to convert from Hz to rad/s
    """
    from math import pi
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix

    out_file = fname_presuffix(in_file, suffix='_rad', newpath=newpath)
    fmapnii = nb.load(in_file)
    fmapdata = fmapnii.get_data()

    if fmap_range is None:
        fmap_range = max(abs(fmapdata.min()), fmapdata.max())
    fmapdata = fmapdata * (pi / fmap_range)
    out_img = nb.Nifti1Image(fmapdata, fmapnii.affine, fmapnii.header)
    out_img.set_data_dtype('float32')
    out_img.to_filename(out_file)
    return out_file, fmap_range


def _tohz(in_file, range_hz, newpath=None):
    """Convert a field map to Hz units"""
    from math import pi
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix

    out_file = fname_presuffix(in_file, suffix='_hz', newpath=newpath)
    fmapnii = nb.load(in_file)
    fmapdata = fmapnii.get_data()
    fmapdata = fmapdata * (range_hz / pi)
    out_img = nb.Nifti1Image(fmapdata, fmapnii.affine, fmapnii.header)
    out_img.set_data_dtype('float32')
    out_img.to_filename(out_file)
    return out_file


def phdiff2fmap(in_file, delta_te, newpath=None):
    r"""
    Converts the input phase-difference map into a fieldmap in Hz,
    using the eq. (1) of [Hutton2002]_:

    .. math::

        \Delta B_0 (\text{T}^{-1}) = \frac{\Delta \Theta}{2\pi\gamma \Delta\text{TE}}


    In this case, we do not take into account the gyromagnetic ratio of the
    proton (:math:`\gamma`), since it will be applied inside TOPUP:

    .. math::

        \Delta B_0 (\text{Hz}) = \frac{\Delta \Theta}{2\pi \Delta\text{TE}}

    """
    import math
    import numpy as np
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix
    #  GYROMAG_RATIO_H_PROTON_MHZ = 42.576

    out_file = fname_presuffix(in_file, suffix='_fmap', newpath=newpath)
    image = nb.load(in_file)
    data = (image.get_data().astype(np.float32) / (2. * math.pi * delta_te))
    nii = nb.Nifti1Image(data, image.affine, image.header)
    nii.set_data_dtype(np.float32)
    nii.to_filename(out_file)
    return out_file


def _delta_te(in_values, te1=None, te2=None):
    """Read :math:`\Delta_\text{TE}` from BIDS metadata dict"""
    if isinstance(in_values, float):
        te2 = in_values
        te1 = 0.

    if isinstance(in_values, dict):
        te1 = in_values.get('EchoTime1')
        te2 = in_values.get('EchoTime2')

        if not all((te1, te2)):
            te2 = in_values.get('EchoTimeDifference')
            te1 = 0

    if isinstance(in_values, list):
        te2, te1 = in_values
        if isinstance(te1, list):
            te1 = te1[1]
        if isinstance(te2, list):
            te2 = te2[1]

    # For convienience if both are missing we should give one error about them
    if te1 is None and te2 is None:
        raise RuntimeError('EchoTime1 and EchoTime2 metadata fields not found. '
                           'Please consult the BIDS specification.')
    if te1 is None:
        raise RuntimeError(
            'EchoTime1 metadata field not found. Please consult the BIDS specification.')
    if te2 is None:
        raise RuntimeError(
            'EchoTime2 metadata field not found. Please consult the BIDS specification.')

    return abs(float(te2) - float(te1))


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


def get_topup_inputs_from(dwi_file, bval_file, b0_threshold, topup_prefix,
                          bids_origin_files, epi_fmaps=None, max_per_spec=3,
                          topup_requested=False):
    """Create a datain spec and a slspec from a concatenated dwi series.

    Create inputs for TOPUP that come from data in ``dwi/`` and epi fieldmaps in ``fmap/``.
    The ``nii_file`` input may be the result of concatenating a number of scans with different
    distortions present. The original source of each volume in ``nii_file`` is listed in
    ``bids_origin_files``.

    The strategy is to select ``max_per_spec`` b=0 images from each distortion group.
    Here, distortion group uses the FSL definition of a phase encoding direction and
    total readout time, as specified in the datain file used by TOPUP (i.e. "0 -1 0 0.087").


    Case: Two opposing PE direction dwi series
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    For example if the following b=0 images are found at the following indices into
    ``nii_file``:

    ============ ============================= ==================
    Image Index  BIDS source file for a b=0    Distortion Group
    ------------ ----------------------------- ------------------
    0            sub-1_dir-AP_run-1_dwi.nii.gz ``0 -1 0 0.087``
    15           sub-1_dir-AP_run-1_dwi.nii.gz ``0 -1 0 0.087``
    30           sub-1_dir-AP_run-1_dwi.nii.gz ``0 -1 0 0.087``
    45           sub-1_dir-AP_run-1_dwi.nii.gz ``0 -1 0 0.087``
    60           sub-1_dir-AP_run-2_dwi.nii.gz ``0 -1 0 0.087``
    75           sub-1_dir-AP_run-2_dwi.nii.gz ``0 -1 0 0.087``
    90           sub-1_dir-AP_run-2_dwi.nii.gz ``0 -1 0 0.087``
    105          sub-1_dir-AP_run-2_dwi.nii.gz ``0 -1 0 0.087``
    120          sub-1_dir-PA_run-1_dwi.nii.gz ``0 1 0 0.087``
    135          sub-1_dir-PA_run-1_dwi.nii.gz ``0 1 0 0.087``
    150          sub-1_dir-PA_run-1_dwi.nii.gz ``0 1 0 0.087``
    165          sub-1_dir-PA_run-1_dwi.nii.gz ``0 1 0 0.087``
    ============ ============================= ==================

    This will select images 0, 45 and 105 to represent the distortion group ``0 -1 0 0.087`` and
    images 120, 135 and 165 to represent ``0 1 0 0.087``. The ``--datain`` file would then
    contain::

        0 -1 0 0.087
        0 -1 0 0.087
        0 -1 0 0.087
        0 1 0 0.087
        0 1 0 0.087
        0 1 0 0.087

    Case: one DWI series and an EPI fieldmap
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    If a reverse-phase encoding fieldmap image (or images) are passed in through ``epi_fmaps``,
    these will undergo the same selection process using ``max_per_spec``. The images will be
    added to the *end* of the image series, though, to ensure that the fieldmap correction will
    be aligned to the first b=0 image in ``nii_file``. For example if ``nii_file`` contains

    ============ ============================= ==================
    Image Index  BIDS source file for a b=0    Distortion Group
    ------------ ----------------------------- ------------------
    0            sub-1_dir-AP_run-1_dwi.nii.gz ``0 -1 0 0.087``
    15           sub-1_dir-AP_run-1_dwi.nii.gz ``0 -1 0 0.087``
    30           sub-1_dir-AP_run-1_dwi.nii.gz ``0 -1 0 0.087``
    45           sub-1_dir-AP_run-1_dwi.nii.gz ``0 -1 0 0.087``
    ============ ============================= ==================

    and the file from fmaps contains

    ============ ============================= ==================
    Image Index  BIDS source file for a b=0    Distortion Group
    ------------ ----------------------------- ------------------
    0            sub-1_dir-PA_epi.nii.gz       ``0 1 0 0.087``
    1            sub-1_dir-PA_epi.nii.gz       ``0 1 0 0.087``
    ============ ============================= ==================

    images 0, 15 and 45 would be selected to represent ``0 -1 0 0.087`` and images 0 and 1
    would be selected to represent ``0 1 0 0.087``, resulting in a ``--datain`` file that
    contains::

        0 -1 0 0.087
        0 -1 0 0.087
        0 -1 0 0.087
        0 1 0 0.087
        0 1 0 0.087


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
            A list of b=0 images from the fmaps/ directory.
        max_per_spec: int
            The maximum number of b=0 images to extract from a PE direction / image set

    """

    # Start with the DWI file. Determine which images are b=0
    bvals = np.loadtxt(bval_file)
    b0_indices = np.flatnonzero(bvals < b0_threshold)
    if not b0_indices.size:
        raise RuntimeError("No b=0 images available for TOPUP from the dwi.")
    dwi_nii = load_img(dwi_file)
    # Gather images from just the dwi series
    dwi_spec_lines, dwi_imain, dwi_report, _ = topup_inputs_from_4d_file(
        dwi_nii, b0_indices, bids_origin_files, image_source="combined DWI series",
        max_per_spec=max_per_spec)

    # If there are EPI fieldmaps, add them to the END of the topup spec
    if epi_fmaps and isdefined(epi_fmaps):
        topup_imain, topup_spec_lines, fmap_report = add_epi_fmaps_to_dwi_b0s(
            epi_fmaps, b0_threshold, max_per_spec, dwi_spec_lines, dwi_imain)
        topup_text = dwi_report + fmap_report
    else:
        topup_imain = dwi_imain
        topup_spec_lines = dwi_spec_lines
        topup_text = dwi_report

    imain_output = topup_prefix + "imain.nii.gz"
    imain_img = to_lps(topup_imain, new_axcodes=('L', 'A', 'S'))
    assert imain_img.shape[3] == len(topup_spec_lines)
    imain_img.to_filename(imain_output)

    # Write the datain text file and make sure it's usable if it's needed
    if len(set(topup_spec_lines)) < 2 and topup_requested:
        print(topup_spec_lines)
        raise Exception("Unable to run TOPUP: not enough distortion groups. "
                        "Check \"IntendedFor\" fields or consider using --ignore fieldmaps.")

    datain_file = topup_prefix + "datain.txt"
    with open(datain_file, "w") as f:
        f.write("\n".join(topup_spec_lines))

    return datain_file, imain_output, topup_text


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
        List of the original indices of the images in ``concatenated_images``
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


def topup_inputs_from_4d_file(nii_file, b0_indices, bids_origin_files=None,
                              image_source="combined DWI series", max_per_spec=3):
    """Represent distortion groups from a concatenated image and its origins.

    Create inputs for TOPUP that come from data in ``dwi/`` and epi fieldmaps in ``fmap/``.
    The ``nii_file`` input may be the result of concatenating a number of scans with different
    distortions present. The original source of each volume in ``nii_file`` is listed in
    ``bids_origin_files``.

    The strategy is to select ``max_per_spec`` b=0 images from each distortion group.
    Here, distortion group uses the FSL definition of a phase encoding direction and
    total readout time, as specified in the datain file used by TOPUP (i.e. "0 -1 0 0.087").

    **Parameters**

        nii_file : Nibabel image
            A 4D Image
        b0_indices: array-like
            indices into nii_file that can be used by topup
        bids_origin_files: list
            A list with the original bids file of each image in ``nii_file``. This is
            necessary because merging may have happened earlier in the pipeline
        max_per_spec: int
            The maximum number of b=0 images to extract from a PE direction / image set


    """

    # Start with the DWI file. Determine which images are b=0
    if not len(b0_indices):
        raise RuntimeError("No b=0 images available for TOPUP.")

    # find the original files accompanying each b=0
    b0_bids_origins = [bids_origin_files[idx] for idx in b0_indices]

    # Create a lookup-table for each file that was merged into nii_file
    # spec_lookup maps original_bids_file -> acqp line
    unique_files = list(set(b0_bids_origins))
    spec_lookup = {}
    slicetime_lookup = {}
    for unique_dwi in unique_files:
        spec = read_nifti_sidecar(unique_dwi)
        spec_line = acqp_lines[spec['PhaseEncodingDirection']]
        spec_lookup[unique_dwi] = spec_line % spec['TotalReadoutTime']
        slicetime_lookup[unique_dwi] = spec['SliceTiming']

    # Which spec does each b=0 belong to?
    spec_indices = defaultdict(list)
    for b0_index, bids_file in zip(b0_indices, b0_bids_origins):
        spec_line = spec_lookup[bids_file]
        spec_indices[spec_line].append(b0_index)

    # The first image needs to be the first b=0 from the dwi series
    first_b0_spec_line = spec_lookup[b0_bids_origins[0]]
    first_b0_spec_indices = spec_indices.pop(first_b0_spec_line)
    selected_b0_indices = get_evenly_spaced_b0s(first_b0_spec_indices, max_per_spec)
    spec_lines = [first_b0_spec_line] * len(selected_b0_indices)

    # Iterate over the remaining unique spec lines
    for spec_line in spec_indices:
        spec_b0_indices = get_evenly_spaced_b0s(spec_indices[spec_line], max_per_spec)
        selected_b0_indices += spec_b0_indices
        spec_lines += [spec_line] * len(spec_b0_indices)

    # Load and subset the image
    imain_nii = index_img(nii_file, selected_b0_indices)
    report = topup_selection_to_report(selected_b0_indices, bids_origin_files, spec_lookup,
                                       image_source=image_source)

    return spec_lines, imain_nii, report, spec_lookup


def get_evenly_spaced_b0s(b0_indices, max_per_spec):
    """Choose up to ``max_per_spec`` b=0 images from a list of b0 indices."""
    if len(b0_indices) <= max_per_spec:
        return b0_indices
    selected_indices = np.linspace(0, len(b0_indices)-1, num=max_per_spec,
                                   endpoint=True, dtype=np.int)
    return [b0_indices[idx] for idx in selected_indices]


def add_epi_fmaps_to_dwi_b0s(epi_fmaps, b0_threshold, max_per_spec, dwi_spec_lines, dwi_imain):
    """Add additional images from EPI fieldmaps for distortion correction.

    In order to fill out the maximum number of images per distortion group, images
    from files in the fmap/ directory can be added to those already extracted from the
    DWI series.

    Examples:
    ---------

    >>> epi_fmaps = ["/data/sub-1/fmap/sub-1_dir-AP_epi.nii.gz",
    ...              "/data/sub-1/fmap/sub-1_dir-PA_epi.nii.gz"]

    """
    # Extract b=0 images as if we were only pulling images from epi fmaps.
    fmaps_4d, fmap_b0_indices, fmap_original_files = load_epi_dwi_fieldmaps(epi_fmaps,
                                                                            b0_threshold)
    fmap_spec_lines, fmap_imain, fmap_report, fmap_spec_map = topup_inputs_from_4d_file(
        fmaps_4d, fmap_b0_indices, fmap_original_files, image_source="EPI fieldmap",
        max_per_spec=max_per_spec)

    # Check how many are present in each group from just the dwi files
    spec_counts = defaultdict(int)
    for dwi_spec in dwi_spec_lines:
        spec_counts[dwi_spec] += 1

    # Only add as many as you need to fill out max_per_spec
    fmap_indices_to_add = []
    for image_num, epi_spec in enumerate(fmap_spec_lines):
        if spec_counts[epi_spec] + 1 > max_per_spec:
            continue
        fmap_indices_to_add.append(image_num)
        spec_counts[epi_spec] += 1

    # No additional epi fmaps to add
    if not fmap_indices_to_add:
        return dwi_imain, dwi_spec_lines, \
            ' No Additional images from EPI fieldmaps were added because the maximum ' \
            'number of images per distortion group was reached.'

    # Add the epi b=0's to the dwi b=0's
    topup_imain = concat_imgs([dwi_imain, index_img(fmap_imain, fmap_indices_to_add)],
                              auto_resample=True)
    topup_spec_lines = dwi_spec_lines + [fmap_spec_lines[idx] for idx in fmap_indices_to_add]

    new_report = topup_selection_to_report(
        fmap_indices_to_add, fmap_original_files, fmap_spec_map, image_source='EPI fieldmap')

    return topup_imain, topup_spec_lines, new_report


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
