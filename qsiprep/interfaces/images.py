#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Image tools interfaces
~~~~~~~~~~~~~~~~~~~~~~


"""

import os
from subprocess import Popen, PIPE
from textwrap import indent
import numpy as np
import nibabel as nb
import nilearn.image as nli
from dipy.io import read_bvals_bvecs
from nipype import logging
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (isdefined, traits, TraitedSpec, BaseInterfaceInputSpec,
                                    SimpleInterface, File, InputMultiObject, OutputMultiObject)
from nipype.interfaces import fsl
# from qsiprep.interfaces.images import (
#    nii_ones_like, extract_wm, SignalExtraction, MatchHeader,
#    FilledImageLike, DemeanImage, TemplateDimensions)
from .mrtrix import SS3T_ROOT
from ..niworkflows.interfaces.images import ValidateImageInputSpec

LOGGER = logging.getLogger('nipype.interface')


class SplitDWIsInputSpec(BaseInterfaceInputSpec):
    dwi_file = File(desc='the dwi image')
    bvec_file = File(desc='the bvec file')
    bval_file = File(desc='the bval file')
    deoblique_bvecs = traits.Bool(False, usedefault=True,
                                  desc='write LPS+ world coordinate bvecs')
    b0_threshold = traits.Int(50, usedefault=True,
                              desc='Maximum b-value that can be considered a b0')


class SplitDWIsOutputSpec(TraitedSpec):
    dwi_files = OutputMultiObject(File(exists=True), desc='single volume dwis')
    bval_files = OutputMultiObject(File(exists=True), desc='single volume bvals')
    bvec_files = OutputMultiObject(File(exists=True), desc='single volume bvecs')
    b0_images = OutputMultiObject(File(exists=True), desc='just the b0s')
    b0_indices = traits.List(desc='list of original indices for each b0 image')
    original_files = OutputMultiObject(File(exists=True))


class SplitDWIs(SimpleInterface):
    input_spec = SplitDWIsInputSpec
    output_spec = SplitDWIsOutputSpec

    def _run_interface(self, runtime):
        split = fsl.Split(dimension='t', in_file=self.inputs.dwi_file)
        split_dwi_files = split.run().outputs.out_files

        split_bval_files, split_bvec_files = split_bvals_bvecs(
            self.inputs.bval_file, self.inputs.bvec_file, split_dwi_files,
            self.inputs.deoblique_bvecs, runtime.cwd)

        bvalues = np.loadtxt(self.inputs.bval_file)
        b0_indices = np.flatnonzero(bvalues < self.inputs.b0_threshold)
        b0_paths = [split_dwi_files[idx] for idx in b0_indices]
        self._results['dwi_files'] = split_dwi_files
        self._results['bval_files'] = split_bval_files
        self._results['bvec_files'] = split_bvec_files
        self._results['b0_images'] = b0_paths
        self._results['b0_indices'] = b0_indices.tolist()

        return runtime


def _flatten(in_list):
    out_list = []
    for item in in_list:
        if isinstance(item, (list, tuple)):
            out_list.extend(item)
        else:
            out_list.append(item)
    return out_list


class NiftiInfoInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='Input NIfTI file')


class NiftiInfoOutputSpec(TraitedSpec):
    voxel_size = traits.Tuple()


class NiftiInfo(SimpleInterface):
    input_spec = NiftiInfoInputSpec
    output_spec = NiftiInfoOutputSpec

    def _run_interface(self, runtime):
        in_file = self.inputs.in_file
        img = nb.load(in_file)
        self._results['voxel_size'] = tuple(img.header.get_zooms()[:3])
        return runtime


class IntraModalMergeInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiObject(File(exists=True), mandatory=True,
                                desc='input files')
    hmc = traits.Bool(True, usedefault=True)
    zero_based_avg = traits.Bool(True, usedefault=True)
    to_lps = traits.Bool(True, usedefault=True)


class IntraModalMergeOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='merged image')
    out_avg = File(exists=True, desc='average image')
    out_mats = OutputMultiObject(File(exists=True), desc='output matrices')
    out_movpar = OutputMultiObject(File(exists=True), desc='output movement parameters')


class IntraModalMerge(SimpleInterface):
    input_spec = IntraModalMergeInputSpec
    output_spec = IntraModalMergeOutputSpec

    def _run_interface(self, runtime):
        in_files = self.inputs.in_files
        if not isinstance(in_files, list):
            in_files = [self.inputs.in_files]

        # Generate output average name early
        self._results['out_avg'] = fname_presuffix(self.inputs.in_files[0],
                                                   suffix='_avg', newpath=runtime.cwd)

        if self.inputs.to_lps:
            in_files = [reorient(inf, newpath=runtime.cwd)
                        for inf in in_files]

        if len(in_files) == 1:
            filenii = nb.load(in_files[0])
            filedata = filenii.get_data()

            # magnitude files can have an extra dimension empty
            if filedata.ndim == 5:
                sqdata = np.squeeze(filedata)
                if sqdata.ndim == 5:
                    raise RuntimeError('Input image (%s) is 5D' % in_files[0])
                else:
                    in_files = [fname_presuffix(in_files[0], suffix='_squeezed',
                                                newpath=runtime.cwd)]
                    nb.Nifti1Image(sqdata, filenii.get_affine(),
                                   filenii.get_header()).to_filename(in_files[0])

            if np.squeeze(nb.load(in_files[0]).get_data()).ndim < 4:
                self._results['out_file'] = in_files[0]
                self._results['out_avg'] = in_files[0]
                # TODO: generate identity out_mats and zero-filled out_movpar
                return runtime
            in_files = in_files[0]
        else:
            magmrg = fsl.Merge(dimension='t', in_files=self.inputs.in_files)
            in_files = magmrg.run().outputs.merged_file
        mcflirt = fsl.MCFLIRT(cost='normcorr', save_mats=True, save_plots=True,
                              ref_vol=0, in_file=in_files)
        mcres = mcflirt.run()
        self._results['out_mats'] = mcres.outputs.mat_file
        self._results['out_movpar'] = mcres.outputs.par_file
        self._results['out_file'] = mcres.outputs.out_file

        hmcnii = nb.load(mcres.outputs.out_file)
        hmcdat = hmcnii.get_data().mean(axis=3)
        if self.inputs.zero_based_avg:
            hmcdat -= hmcdat.min()

        nb.Nifti1Image(
            hmcdat, hmcnii.get_affine(), hmcnii.get_header()).to_filename(
            self._results['out_avg'])

        return runtime


CONFORMATION_TEMPLATE = """\t\t<h3 class="elem-title">Anatomical Conformation</h3>
\t\t<ul class="elem-desc">
\t\t\t<li>Input T1w images: {n_t1w}</li>
\t\t\t<li>Output orientation: LPS</li>
\t\t\t<li>Output dimensions: {dims}</li>
\t\t\t<li>Output voxel size: {zooms}</li>
\t\t\t<li>Discarded images: {n_discards}</li>
{discard_list}
\t\t</ul>
"""

DISCARD_TEMPLATE = """\t\t\t\t<li><abbr title="{path}">{basename}</abbr></li>"""


class ConformInputSpec(BaseInterfaceInputSpec):
    in_file = File(mandatory=True, desc='Input image')
    target_zooms = traits.Tuple(traits.Float, traits.Float, traits.Float,
                                desc='Target zoom information')
    target_shape = traits.Tuple(traits.Int, traits.Int, traits.Int,
                                desc='Target shape information')
    deoblique_header = traits.Bool(False, usedfault=True)


class ConformOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='Conformed image')
    transform = File(exists=True, desc='Conformation transform')
    # report = File(exists=True, desc='reportlet about orientation')


class Conform(SimpleInterface):
    """Conform a series of T1w images to enable merging.

    Performs two basic functions:

    1. Orient to LPS (right-left, anterior-posterior, inferior-superior)
    2. Resample to target zooms (voxel sizes) and shape (number of voxels)

    """
    input_spec = ConformInputSpec
    output_spec = ConformOutputSpec

    def _run_interface(self, runtime):
        # Load image, orient as LPS
        fname = self.inputs.in_file
        orig_img = nb.load(fname)
        reoriented = to_lps(orig_img)

        # Set target shape information
        target_zooms = np.array(self.inputs.target_zooms)
        target_shape = np.array(self.inputs.target_shape)
        target_span = target_shape * target_zooms

        zooms = np.array(reoriented.header.get_zooms()[:3])
        shape = np.array(reoriented.shape[:3])

        # Reconstruct transform from orig to reoriented image
        ornt_xfm = nb.orientations.inv_ornt_aff(
            nb.io_orientation(reoriented.affine), orig_img.shape)
        # Identity unless proven otherwise
        target_affine = reoriented.affine.copy()
        conform_xfm = np.eye(4)
        # conform_xfm = np.diag([-1, -1, 1, 1])

        xyz_unit = reoriented.header.get_xyzt_units()[0]
        if xyz_unit == 'unknown':
            # Common assumption; if we're wrong, unlikely to be the only thing that breaks
            xyz_unit = 'mm'

        # Set a 0.05mm threshold to performing rescaling
        atol = {'meter': 1e-5, 'mm': 0.01, 'micron': 10}[xyz_unit]

        # Rescale => change zooms
        # Resize => update image dimensions
        rescale = not np.allclose(zooms, target_zooms, atol=atol)
        resize = not np.all(shape == target_shape)
        if rescale or resize:
            if rescale:
                scale_factor = target_zooms / zooms
                target_affine[:3, :3] = reoriented.affine[:3, :3].dot(np.diag(scale_factor))

            if resize:
                # The shift is applied after scaling.
                # Use a proportional shift to maintain relative position in dataset
                size_factor = target_span / (zooms * shape)
                # Use integer shifts to avoid unnecessary interpolation
                offset = (reoriented.affine[:3, 3] * size_factor - reoriented.affine[:3, 3])
                target_affine[:3, 3] = reoriented.affine[:3, 3] + offset.astype(int)

            data = nli.resample_img(reoriented, target_affine, target_shape).get_data()
            conform_xfm = np.linalg.inv(reoriented.affine).dot(target_affine)
            reoriented = reoriented.__class__(data, target_affine, reoriented.header)

        if self.inputs.deoblique_header:
            is_oblique = np.any(np.abs(nb.affines.obliquity(reoriented.affine)) > 0)
            if is_oblique:
                LOGGER.warning("Removing obliquity from image affine")
                new_affine = reoriented.affine.copy()
                new_affine[:, :-1] = 0
                new_affine[(0, 1, 2), (0, 1, 2)] = reoriented.header.get_zooms()[:3] \
                    * np.sign(reoriented.affine[(0, 1, 2), (0, 1, 2)])
                reoriented = nb.Nifti1Image(reoriented.get_fdata(), new_affine, reoriented.header)

        # Image may be reoriented, rescaled, and/or resized
        if reoriented is not orig_img:
            out_name = fname_presuffix(fname, suffix='_lps', newpath=runtime.cwd)
            reoriented.to_filename(out_name)
            transform = ornt_xfm.dot(conform_xfm)
            if not np.allclose(orig_img.affine.dot(transform), target_affine):
                LOGGER.warning("Check alignment of anatomical image.")

        else:
            out_name = fname
            transform = np.eye(4)

        mat_name = fname_presuffix(fname, suffix='.mat', newpath=runtime.cwd, use_ext=False)
        np.savetxt(mat_name, transform, fmt='%.08f')
        self._results['transform'] = mat_name
        self._results['out_file'] = out_name

        return runtime


class ConformDwiInputSpec(BaseInterfaceInputSpec):
    dwi_file = File(mandatory=True, desc='dwi image')
    bval_file = File(exists=True)
    bvec_file = File(exists=True)
    orientation = traits.Enum('LPS', 'LAS', default='LPS', usedefault=True)


class ConformDwiOutputSpec(TraitedSpec):
    dwi_file = File(exists=True, desc='conformed dwi image')
    bvec_file = File(exists=True, desc='conformed bvec file')
    bval_file = File(exists=True, desc='conformed bval file')
    out_report = File(exists=True, desc='HTML segment containing warning')


class ConformDwi(SimpleInterface):
    """Conform a series of dwi images to enable merging.
    Performs three basic functions:
    #. Orient image to requested orientation
    #. Validate the qform and sform, set qform code to 1
    #. Flip bvecs accordingly
    #. Do nothing to the bvals
    Note: This is not as nuanced as fmriprep's version
    """
    input_spec = ConformDwiInputSpec
    output_spec = ConformDwiOutputSpec

    def _run_interface(self, runtime):
        fname = self.inputs.dwi_file
        orientation = self.inputs.orientation
        suffix = "_" + orientation
        out_fname = fname_presuffix(fname, suffix=suffix, newpath=runtime.cwd)

        # If not defined, find it
        if isdefined(self.inputs.bval_file):
            bval_fname = self.inputs.bval_file
        else:
            bval_fname = fname_presuffix(fname, suffix=".bval", use_ext=False)

        if isdefined(self.inputs.bvec_file):
            bvec_fname = self.inputs.bvec_file
        else:
            bvec_fname = fname_presuffix(fname, suffix=".bvec", use_ext=False)

        out_bvec_fname = fname_presuffix(bvec_fname, suffix=suffix, newpath=runtime.cwd)
        validator = ValidateImage(in_file=fname)
        validated = validator.run()
        self._results['out_report'] = validated.outputs.out_report
        input_img = nb.load(validated.outputs.out_file)

        input_axcodes = nb.aff2axcodes(input_img.affine)
        # Is the input image oriented how we want?
        new_axcodes = tuple(orientation)

        if not input_axcodes == new_axcodes:
            # Re-orient
            LOGGER.info("Re-orienting %s to %s", fname, orientation)
            input_orientation = nb.orientations.axcodes2ornt(input_axcodes)
            desired_orientation = nb.orientations.axcodes2ornt(new_axcodes)
            transform_orientation = nb.orientations.ornt_transform(
                input_orientation, desired_orientation)
            reoriented_img = input_img.as_reoriented(transform_orientation)
            reoriented_img.to_filename(out_fname)
            self._results['dwi_file'] = out_fname

            # Flip the bvecs
            if os.path.exists(bvec_fname):
                LOGGER.info('Reorienting %s to %s', bvec_fname, orientation)
                bvec_array = np.loadtxt(bvec_fname)
                if not bvec_array.shape[0] == transform_orientation.shape[0]:
                    raise ValueError("Unrecognized bvec format")
                output_array = np.zeros_like(bvec_array)
                for this_axnum, (axnum, flip) in enumerate(transform_orientation):
                    output_array[this_axnum] = bvec_array[int(axnum)] * flip
                np.savetxt(out_bvec_fname, output_array, fmt="%.8f ")
                self._results['bvec_file'] = out_bvec_fname
                self._results['bval_file'] = bval_fname

        else:
            LOGGER.info("Not applying reorientation to %s: already in %s", fname, orientation)
            self._results['dwi_file'] = fname
            if os.path.exists(bvec_fname):
                self._results['bvec_file'] = bvec_fname
                self._results['bval_file'] = bval_fname

        return runtime


class _ChooseInterpolatorInputSpec(BaseInterfaceInputSpec):
    dwi_files = InputMultiObject(File(exists=True), mandatory=True)
    output_resolution = traits.Float(mandatory=True)


class _ChooseInterpolatorOutputSpec(TraitedSpec):
    interpolation_method = traits.Enum("LanczosWindowedSinc", "BSpline")


class ChooseInterpolator(SimpleInterface):
    """If the requested output resolution is more than 10% smaller than the input, use BSpline.
    """
    input_spec = _ChooseInterpolatorInputSpec
    output_spec = _ChooseInterpolatorOutputSpec

    def _run_interface(self, runtime):
        output_resolution = np.array([self.inputs.output_resolution] * 3)
        interpolator = "LanczosWindowedSinc"
        for input_file in self.inputs.dwi_files:
            resolution_cutoff = 0.9 * np.array(nb.load(input_file).header.get_zooms()[:3])
            print(output_resolution, resolution_cutoff)
            if np.any(output_resolution < resolution_cutoff):
                interpolator = "BSpline"
                LOGGER.warning("Using BSpline interpolation for upsampling")
                break
        self._results['interpolation_method'] = interpolator
        return runtime


class ValidateImageOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='validated image')
    out_report = File(exists=True, desc='HTML segment containing warning')


class ValidateImage(SimpleInterface):
    """
    Check the correctness of x-form headers (matrix and code)
    This interface implements the `following logic
    <https://github.com/poldracklab/fmriprep/issues/873#issuecomment-349394544>`_:
    +-------------------+------------------+------------------+------------------\
+------------------------------------------------+
    | valid quaternions | `qform_code > 0` | `sform_code > 0` | `qform == sform` \
| actions                                        |
    +===================+==================+==================+==================\
+================================================+
    | True              | True             | True             | True             \
| None                                           |
    +-------------------+------------------+------------------+------------------\
+------------------------------------------------+
    | True              | True             | False            | *                \
| sform, scode <- qform, qcode                   |
    +-------------------+------------------+------------------+------------------\
+------------------------------------------------+
    | *                 | *                | True             | False            \
| qform, qcode <- sform, scode                   |
    +-------------------+------------------+------------------+------------------\
+------------------------------------------------+
    | *                 | False            | True             | *                \
| qform, qcode <- sform, scode                   |
    +-------------------+------------------+------------------+------------------\
+------------------------------------------------+
    | *                 | False            | False            | *                \
| sform, qform <- best affine; scode, qcode <- 1 |
    +-------------------+------------------+------------------+------------------\
+------------------------------------------------+
    | False             | *                | False            | *                \
| sform, qform <- best affine; scode, qcode <- 1 |
    +-------------------+------------------+------------------+------------------\
+------------------------------------------------+
    """
    input_spec = ValidateImageInputSpec
    output_spec = ValidateImageOutputSpec

    def _run_interface(self, runtime):
        img = nb.load(self.inputs.in_file)
        out_report = os.path.join(runtime.cwd, 'report.html')

        # Retrieve xform codes
        sform_code = int(img.header._structarr['sform_code'])
        qform_code = int(img.header._structarr['qform_code'])

        # Check qform is valid
        valid_qform = False
        try:
            qform = img.get_qform()
            valid_qform = True
        except ValueError:
            pass

        sform = img.get_sform()
        if np.linalg.det(sform) == 0:
            valid_sform = False
        else:
            RZS = sform[:3, :3]
            zooms = np.sqrt(np.sum(RZS * RZS, axis=0))
            valid_sform = np.allclose(zooms, img.header.get_zooms()[:3])

        # Matching affines
        matching_affines = valid_qform and np.allclose(qform, sform)

        # Both match, qform valid (implicit with match), codes okay -> do nothing, empty report
        if matching_affines and qform_code > 0 and sform_code > 0:
            self._results['out_file'] = self.inputs.in_file
            open(out_report, 'w').close()
            self._results['out_report'] = out_report
            return runtime

        # A new file will be written
        out_fname = fname_presuffix(self.inputs.in_file, suffix='_valid', newpath=runtime.cwd)
        self._results['out_file'] = out_fname

        # Row 2:
        if valid_qform and qform_code > 0 and (sform_code == 0 or not valid_sform):
            img.set_sform(qform, qform_code)
            warning_txt = 'Note on orientation: sform matrix set'
            description = """\
<p class="elem-desc">The sform has been copied from qform.</p>
"""
        # Rows 3-4:
        # Note: if qform is not valid, matching_affines is False
        elif (valid_sform and sform_code > 0) and (not matching_affines or qform_code == 0):
            img.set_qform(img.get_sform(), sform_code)
            warning_txt = 'Note on orientation: qform matrix overwritten'
            description = """\
<p class="elem-desc">The qform has been copied from sform.</p>
"""
            if not valid_qform and qform_code > 0:
                warning_txt = 'WARNING - Invalid qform information'
                description = """\
<p class="elem-desc">
    The qform matrix found in the file header is invalid.
    The qform has been copied from sform.
    Checking the original qform information from the data produced
    by the scanner is advised.
</p>
"""
        # Rows 5-6:
        else:
            affine = img.header.get_base_affine()
            img.set_sform(affine, nb.nifti1.xform_codes['scanner'])
            img.set_qform(affine, nb.nifti1.xform_codes['scanner'])
            warning_txt = 'WARNING - Missing orientation information'
            description = """\
<p class="elem-desc">
    QSIPrep could not retrieve orientation information from the image header.
    The qform and sform matrices have been set to a default, LAS-oriented affine.
    Analyses of this dataset MAY BE INVALID.
</p>
"""
        snippet = '<h3 class="elem-title">%s</h3>\n%s:\n\t %s\n' % (
            warning_txt, self.inputs.in_file, description)
        # Store new file and report
        img.to_filename(out_fname)
        with open(out_report, 'w') as fobj:
            fobj.write(indent(snippet, '\t' * 3))

        self._results['out_report'] = out_report
        return runtime


def bvec_to_rasb(bval_file, bvec_file, img_file, workdir):
    """Use mrinfo to convert a bvec to RAS+ world coordinate reference frame"""

    # Make a temporary bvec file that mrtrix likes
    temp_bvec = fname_presuffix(bvec_file, suffix="_fix", newpath=workdir)
    lps_bvec = np.loadtxt(bvec_file).reshape(3, -1)
    np.savetxt(temp_bvec, lps_bvec * np.array([[-1], [1], [1]]))

    # Run mrinfo to to get the RAS+ vector
    cmd = [SS3T_ROOT + '/mrinfo', '-dwgrad',
           '-fslgrad', temp_bvec, bval_file, img_file]
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE)
    out, err = proc.communicate()
    LOGGER.info(' '.join(cmd))
    if err:
        raise Exception(str(err))

    return np.fromstring(out, dtype=float, sep=' ')[:3]


def split_bvals_bvecs(bval_file, bvec_file, img_files, deoblique, working_dir):
    """Split bvals and bvecs into one text file per image."""
    if deoblique:
        LOGGER.info('Converting oblique-image bvecs to workd coordinate reference frame')
    bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)
    split_bval_files = []
    split_bvec_files = []
    for nsample, (bval, bvec, img_file) in enumerate(zip(bvals[:, None], bvecs, img_files)):
        bval_fname = fname_presuffix(bval_file, suffix='_%04d' % nsample, newpath=working_dir)
        bvec_suffix = '_ortho_%04d' % nsample if not deoblique else '_%04d' % nsample
        bvec_fname = fname_presuffix(bvec_file, bvec_suffix, newpath=working_dir)
        np.savetxt(bval_fname, bval)
        np.savetxt(bvec_fname, bvec)

        # re-write the bvec deobliqued, if requested
        if deoblique:
            rasb = bvec_to_rasb(bval_fname, bvec_fname, img_file, working_dir)
            # Convert to image axis orientation
            ornt = nb.aff2axcodes(nb.load(img_file).affine)
            flippage = np.array(
                [1 if ornt[n] == "RAS"[n] else -1 for n in [0, 1, 2]])
            deobliqued_bvec = rasb * flippage
            np.savetxt(bvec_fname, deobliqued_bvec)

        split_bval_files.append(bval_fname)
        split_bvec_files.append(bvec_fname)

    return split_bval_files, split_bvec_files


def reorient(in_file, newpath=None):
    """Reorient Nifti files to LPS."""
    out_file = fname_presuffix(in_file, suffix='_lps', newpath=newpath)
    to_lps(nb.load(in_file)).to_filename(out_file)
    return out_file


def reorient_to(in_file, orientation="LPS", newpath=None):
    out_file = fname_presuffix(in_file, suffix='_'+orientation, newpath=newpath)
    to_lps(in_file, tuple(orientation)).to_filename(out_file)
    return out_file


def to_lps(input_img, new_axcodes=("L", "P", "S")):
    if isinstance(input_img, str):
        input_img = nb.load(input_img)
    input_axcodes = nb.aff2axcodes(input_img.affine)
    # Is the input image oriented how we want?
    if not input_axcodes == new_axcodes:
        # Re-orient
        input_orientation = nb.orientations.axcodes2ornt(input_axcodes)
        desired_orientation = nb.orientations.axcodes2ornt(new_axcodes)
        transform_orientation = nb.orientations.ornt_transform(input_orientation,
                                                               desired_orientation)
        reoriented_img = input_img.as_reoriented(transform_orientation)
        return reoriented_img
    else:
        return input_img
