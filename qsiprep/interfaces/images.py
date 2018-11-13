#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Image tools interfaces
~~~~~~~~~~~~~~~~~~~~~~


"""

import os
import numpy as np
import nibabel as nb
import nilearn.image as nli
from textwrap import indent

from nipype import logging
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec, SimpleInterface,
    File, InputMultiPath, OutputMultiPath)
from nipype.interfaces import fsl
from fmriprep.interfaces.images import (
    normalize_xform, demean, nii_ones_like, extract_wm, SignalExtraction, MatchHeader,
    FilledImageLike, DemeanImage, ValidateImage, TemplateDimensions)

LOGGER = logging.getLogger('nipype.interface')


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
    in_files = InputMultiPath(File(exists=True), mandatory=True,
                              desc='input files')
    hmc = traits.Bool(True, usedefault=True)
    zero_based_avg = traits.Bool(True, usedefault=True)
    to_lps = traits.Bool(True, usedefault=True)


class IntraModalMergeOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='merged image')
    out_avg = File(exists=True, desc='average image')
    out_mats = OutputMultiPath(File(exists=True), desc='output matrices')
    out_movpar = OutputMultiPath(File(exists=True), desc='output movement parameters')


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
    in_file = File(exists=True, mandatory=True, desc='Input image')
    target_zooms = traits.Tuple(traits.Float, traits.Float, traits.Float,
                                desc='Target zoom information')
    target_shape = traits.Tuple(traits.Int, traits.Int, traits.Int,
                                desc='Target shape information')


class ConformOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='Conformed image')
    transform = File(exists=True, desc='Conformation transform')


class Conform(SimpleInterface):
    """Conform a series of T1w images to enable merging.

    Performs two basic functions:

    #. Orient to LPS (right-left, anterior-posterior, inferior-superior)
    #. Resample to target zooms (voxel sizes) and shape (number of voxels)
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

        # Image may be reoriented, rescaled, and/or resized
        if reoriented is not orig_img:
            out_name = fname_presuffix(fname, suffix='_lps', newpath=runtime.cwd)
            reoriented.to_filename(out_name)
        else:
            out_name = fname

        transform = ornt_xfm.dot(conform_xfm)
        assert np.allclose(orig_img.affine.dot(transform), target_affine)

        mat_name = fname_presuffix(fname, suffix='.mat', newpath=runtime.cwd, use_ext=False)
        np.savetxt(mat_name, transform, fmt='%.08f')

        self._results['out_file'] = out_name
        self._results['transform'] = mat_name

        return runtime


class ConformDwiInputSpec(BaseInterfaceInputSpec):
    dwi_file = File(exists=True, mandatory=True, desc='dwi image')
    bvec_file = File(exists=True, mandatory=True, desc='bvec file')
    bval_file = File(exists=True, mandatory=True, desc='bval file')


class ConformDwiOutputSpec(TraitedSpec):
    dwi_file = File(exists=True, mandatory=True, desc='conformed dwi image')
    bvec_file = File(exists=True, mandatory=True, desc='conformed bvec file')
    bval_file = File(exists=True, mandatory=True, desc='conformed bval file')


class ConformDwi(SimpleInterface):
    """Conform a series of dwi images to enable merging.

    Performs three basic functions:

    #. Orient to LPS (right-left, anterior-posterior, inferior-superior)
    #. Flip bvecs accordingly
    #. Do nothing to the bvals

    Note: This is not as nuanced as fmriprep's version
    """
    input_spec = ConformDwiInputSpec
    output_spec = ConformDwiOutputSpec

    def _run_interface(self, runtime):
        fname = self.inputs.dwi_file
        out_fname = fname_presuffix(fname, suffix='_lps', newpath=runtime.cwd)
        bvec_fname = self.inputs.bvec_file
        out_bvec_fname = fname_presuffix(fname, suffix='_lps', newpath=runtime.cwd)
        input_img = nb.load(fname)
        input_axcodes = nb.aff2axcodes(input_img.affine)
        # Is the input image oriented how we want?
        new_axcodes = ('L', 'P', 'S')
        if not input_axcodes == new_axcodes:
            # Re-orient
            input_orientation = nb.orientations.axcodes2ornt(input_axcodes)
            desired_orientation = nb.orientations.axcodes2ornt(new_axcodes)
            transform_orientation = nb.orientations.ornt_transform(
                        input_orientation, desired_orientation)
            reoriented_img = input_img.as_reoriented(transform_orientation)
            reoriented_img.to_filename(out_fname)

            # Flip the bvecs
            bvec_array = np.loadtxt(bvec_fname)
            if not bvec_array.shape[0] == transform_orientation.shape[0]:
                raise ValueError("Unrecognized bvec format")
            output_array = np.zeros_like(bvec_array)
            for this_axnum, (axnum, flip) in enumerate(transform_orientation):
                output_array[this_axnum] = bvec_array[int(axnum)] * flip
            np.savetxt(out_bvec_fname, output_array, fmt="%.8f ")

        else:
            self._results['dwi_file'] = fname

        self._results['bval_file'] = self.inputs.bval_file

        return runtime


def reorient(in_file, newpath=None):
    """Reorient Nifti files to LPS."""
    out_file = fname_presuffix(in_file, suffix='_lps', newpath=newpath)
    to_lps(nb.load(in_file)).to_filename(out_file)
    return out_file


def to_lps(input_img):
    new_axcodes = ("L", "P", "S")
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
