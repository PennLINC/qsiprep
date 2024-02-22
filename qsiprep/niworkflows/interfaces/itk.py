#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
ITK files handling
~~~~~~~~~~~~~~~~~~


"""
from mimetypes import guess_type

import nibabel as nb
import numpy as np
from nipype import logging
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)
from nipype.utils.filemanip import fname_presuffix

LOGGER = logging.getLogger('nipype.interface')


class FUGUEvsm2ANTSwarpInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True,
                   desc='input displacements field map')
    pe_dir = traits.Enum('i', 'i-', 'j', 'j-', 'k', 'k-',
                         desc='phase-encoding axis')


class FUGUEvsm2ANTSwarpOutputSpec(TraitedSpec):
    out_file = File(desc='the output warp field')


class FUGUEvsm2ANTSwarp(SimpleInterface):

    """
    Convert a voxel-shift-map to ants warp

    """
    input_spec = FUGUEvsm2ANTSwarpInputSpec
    output_spec = FUGUEvsm2ANTSwarpOutputSpec

    def _run_interface(self, runtime):

        nii = nb.load(self.inputs.in_file)

        phaseEncDim = {'i': 0, 'j': 1, 'k': 2}[self.inputs.pe_dir[0]]

        if len(self.inputs.pe_dir) == 2:
            phaseEncSign = 1.0
        else:
            phaseEncSign = -1.0

        # Fix header
        hdr = nii.header.copy()
        hdr.set_data_dtype(np.dtype('<f4'))
        hdr.set_intent('vector', (), '')

        # Get data, convert to mm
        data = nii.get_fdata()

        aff = np.diag([1.0, 1.0, -1.0])
        if np.linalg.det(aff) < 0 and phaseEncDim != 0:
            # Reverse direction since ITK is LPS
            aff *= -1.0

        aff = aff.dot(nii.affine[:3, :3])

        data *= phaseEncSign * nii.header.get_zooms()[phaseEncDim]

        # Add missing dimensions
        zeros = np.zeros_like(data)
        field = [zeros, zeros]
        field.insert(phaseEncDim, data)
        field = np.stack(field, -1)
        # Add empty axis
        field = field[:, :, :, np.newaxis, :]

        # Write out
        self._results['out_file'] = fname_presuffix(
            self.inputs.in_file, suffix='_antswarp', newpath=runtime.cwd)
        nb.Nifti1Image(
            field.astype(np.dtype('<f4')), nii.affine, hdr).to_filename(
                self._results['out_file'])

        return runtime


def _mat2itk(args):
    from nipype.interfaces.c3 import C3dAffineTool
    from nipype.utils.filemanip import fname_presuffix

    in_file, in_ref, in_src, index, newpath = args
    # Generate a temporal file name
    out_file = fname_presuffix(in_file, suffix='_itk-%05d.txt' % index,
                               newpath=newpath)

    # Run c3d_affine_tool
    C3dAffineTool(transform_file=in_file, reference_file=in_ref, source_file=in_src,
                  fsl2ras=True, itk_transform=out_file, resource_monitor=False).run()
    transform = '#Transform %d\n' % index
    with open(out_file) as itkfh:
        transform += ''.join(itkfh.readlines()[2:])

    return (index, transform)


def _applytfms(args):
    """
    Applies ANTs' antsApplyTransforms to the input image.
    All inputs are zipped in one tuple to make it digestible by
    multiprocessing's map
    """
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix

    from ...niworkflows.interfaces.fixes import (
        FixHeaderApplyTransforms as ApplyTransforms,
    )

    in_file, in_xform, ifargs, index, newpath = args
    out_file = fname_presuffix(in_file, suffix='_xform-%05d' % index,
                               newpath=newpath, use_ext=True)

    copy_dtype = ifargs.pop('copy_dtype', False)
    xfm = ApplyTransforms(
        input_image=in_file, transforms=in_xform, output_image=out_file, **ifargs)
    xfm.terminal_output = 'allatonce'
    xfm.resource_monitor = False
    runtime = xfm.run().runtime

    if copy_dtype:
        nii = nb.load(out_file)
        in_dtype = nb.load(in_file).get_data_dtype()

        # Overwrite only iff dtypes don't match
        if in_dtype != nii.get_data_dtype():
            nii.set_data_dtype(in_dtype)
            nii.to_filename(out_file)

    return (out_file, runtime.cmdline)


def _arrange_xfms(transforms, num_files, tmp_folder):
    """
    Convenience method to arrange the list of transforms that should be applied
    to each input file
    """
    base_xform = ['#Insight Transform File V1.0', '#Transform 0']
    # Initialize the transforms matrix
    xfms_T = []
    for i, tf_file in enumerate(transforms):
        # If it is a deformation field, copy to the tfs_matrix directly
        if guess_type(tf_file)[0] != 'text/plain':
            xfms_T.append([tf_file] * num_files)
            continue

        with open(tf_file) as tf_fh:
            tfdata = tf_fh.read().strip()

        # If it is not an ITK transform file, copy to the tfs_matrix directly
        if not tfdata.startswith('#Insight Transform File'):
            xfms_T.append([tf_file] * num_files)
            continue

        # Count number of transforms in ITK transform file
        nxforms = tfdata.count('#Transform')

        # Remove first line
        tfdata = tfdata.split('\n')[1:]

        # If it is a ITK transform file with only 1 xform, copy to the tfs_matrix directly
        if nxforms == 1:
            xfms_T.append([tf_file] * num_files)
            continue

        if nxforms != num_files:
            raise RuntimeError('Number of transforms (%d) found in the ITK file does not match'
                               ' the number of input image files (%d).' % (nxforms, num_files))

        # At this point splitting transforms will be necessary, generate a base name
        out_base = fname_presuffix(tf_file, suffix='_pos-%03d_xfm-{:05d}' % i,
                                   newpath=tmp_folder.name).format
        # Split combined ITK transforms file
        split_xfms = []
        for xform_i in range(nxforms):
            # Find start token to extract
            startidx = tfdata.index('#Transform %d' % xform_i)
            next_xform = base_xform + tfdata[startidx + 1:startidx + 4] + ['']
            xfm_file = out_base(xform_i)
            with open(xfm_file, 'w') as out_xfm:
                out_xfm.write('\n'.join(next_xform))
            split_xfms.append(xfm_file)
        xfms_T.append(split_xfms)

    # Transpose back (only Python 3)
    return list(map(list, zip(*xfms_T)))
