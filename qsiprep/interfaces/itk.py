# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
ITK files handling
~~~~~~~~~~~~~~~~~~


"""

import os
import os.path as op
import subprocess
from mimetypes import guess_type

import nibabel as nb
import numpy as np
import SimpleITK as sitk
from dipy.core import geometry as geom
from nipype import logging
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    InputMultiObject,
    OutputMultiObject,
    SimpleInterface,
    TraitedSpec,
    traits,
)
from nipype.utils.filemanip import fname_presuffix
from niworkflows.viz.utils import compose_view

from ..viz.utils import plot_acpc

LOGGER = logging.getLogger('nipype.interface')


class _AffineToRigidInputSpec(BaseInterfaceInputSpec):
    affine_transform = InputMultiObject(File(exists=True, mandatory=True))


class _AffineToRigidOutputSpec(TraitedSpec):
    rigid_transform = traits.List(File(exists=True))
    rigid_transform_inverse = traits.List(File(exists=True))
    translation_transform = traits.List(File(exists=True))


class AffineToRigid(SimpleInterface):
    input_spec = _AffineToRigidInputSpec
    output_spec = _AffineToRigidOutputSpec

    def _run_interface(self, runtime):
        if len(self.inputs.affine_transform) > 1:
            raise Exception('Only one transform allowed')
        affine_transform = self.inputs.affine_transform[0]
        rigid_itk, rigid_itk_inverse, translation_itk = itk_affine_to_rigid(
            affine_transform, runtime.cwd
        )
        self._results['rigid_transform'] = [rigid_itk]
        self._results['rigid_transform_inverse'] = [rigid_itk_inverse]
        self._results['translation_transform'] = [translation_itk]
        return runtime


class _ACPCReportInputSpec(BaseInterfaceInputSpec):
    translation_image = File(exists=True, desc='only translated to ACPC', mandatory=True)
    rigid_image = File(exists=True, desc='rigid transformed to ACPC')


class _ACPCReportOutputSpec(TraitedSpec):
    out_report = File(exists=True)


class ACPCReport(SimpleInterface):
    input_spec = _ACPCReportInputSpec
    output_spec = _ACPCReportOutputSpec

    def _run_interface(self, runtime):
        out_report = runtime.cwd + '/ACPCReport.svg'
        # Call composer
        compose_view(
            plot_acpc(
                nb.load(self.inputs.translation_image),
                'moving-image',
                estimate_brightness=True,
                label='Original',
                compress=False,
            ),
            plot_acpc(
                nb.load(self.inputs.rigid_image),
                'fixed-image',
                estimate_brightness=True,
                label='AC-PC',
                compress=False,
            ),
            out_file=out_report,
        )
        self._results['out_report'] = out_report

        return runtime


class DisassembleTransformInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='ANTs composite transform (h5)')


class DisassembleTransformOutputSpec(TraitedSpec):
    out_transforms = OutputMultiObject(File(exists=True))


class DisassembleTransform(SimpleInterface):
    """Sloppy interface to split h5 transforms to a warp and an affine."""

    input_spec = DisassembleTransformInputSpec
    output_spec = DisassembleTransformOutputSpec

    def _run_interface(self, runtime):
        transforms = disassemble_transform(self.inputs.in_file, runtime.cwd)
        self._results['out_transforms'] = transforms
        return runtime


def _applytfms(args):
    """
    Applies ANTs' antsApplyTransforms to the input image.
    All inputs are zipped in one tuple to make it digestible by
    multiprocessing's map
    """
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix
    from niworkflows.interfaces.fixes import FixHeaderApplyTransforms as ApplyTransforms

    in_file, in_xform, ifargs, index, newpath = args
    out_file = fname_presuffix(
        in_file, suffix='_xform-%05d' % index, newpath=newpath, use_ext=True
    )

    copy_dtype = ifargs.pop('copy_dtype', False)
    xfm = ApplyTransforms(
        input_image=in_file, transforms=in_xform, output_image=out_file, **ifargs
    )
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
    to each input file. Not needed in qsiprep
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
            raise RuntimeError(
                'Number of transforms (%d) found in the ITK file does not match'
                ' the number of input image files (%d).' % (nxforms, num_files)
            )

        # At this point splitting transforms will be necessary, generate a base name
        out_base = fname_presuffix(
            tf_file, suffix='_pos-%03d_xfm-{:05d}' % i, newpath=tmp_folder.name
        ).format
        # Split combined ITK transforms file
        split_xfms = []
        for xform_i in range(nxforms):
            # Find start token to extract
            startidx = tfdata.index('#Transform %d' % xform_i)
            next_xform = base_xform + tfdata[startidx + 1 : startidx + 4] + ['']
            xfm_file = out_base(xform_i)
            with open(xfm_file, 'w') as out_xfm:
                out_xfm.write('\n'.join(next_xform))
            split_xfms.append(xfm_file)
        xfms_T.append(split_xfms)

    # Transpose back (only Python 3)
    return list(map(list, zip(*xfms_T, strict=False)))


def disassemble_transform(transform_file, cwd):
    cmd = ['CompositeTransformUtil', '--disassemble', transform_file, 'disassemble']
    affine_out = cwd + '/00_disassemble_AffineTransform.mat'
    warp_out = cwd + '/01_disassemble_DisplacementFieldTransform.nii.gz'
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    LOGGER.info(' '.join(cmd))
    out, err = proc.communicate()

    if not op.exists(affine_out):
        raise Exception('unable to unpack composite transform')
    transforms = [affine_out]
    if op.exists(warp_out):
        transforms.append(warp_out)
    return transforms


def compose_affines(reference_image, affine_list, output_file):
    """Use antsApplyTransforms to get a single affine from multiple affines."""
    cmd = f'antsApplyTransforms -d 3 -r {reference_image} -o Linear[{output_file}, 1] '
    cmd += ' '.join([f'--transform {trf}' for trf in affine_list])
    os.system(cmd)
    assert os.path.exists(output_file)
    return output_file


def itk_affine_to_rigid(transform_file, cwd):
    """uses c3d_affine_tool and FSL's aff2rigid to convert an itk linear
    transform from affine to rigid"""

    rigid_mat_file = cwd + '/6DOFrigid.mat'
    translation_mat_file = cwd + '/translation.mat'
    inverse_mat_file = cwd + '/6DOFinverse.mat'
    raw_transform = sitk.ReadTransform(transform_file)
    aff_transform = sitk.AffineTransform(3)
    aff_transform.SetFixedParameters(raw_transform.GetFixedParameters())
    aff_transform.SetParameters(raw_transform.GetParameters())

    full_matrix = np.eye(4)
    full_matrix[:3, :3] = np.array(aff_transform.GetMatrix()).reshape((3, 3), order='C')
    _, _, angles, _, _ = geom.decompose_matrix(full_matrix)
    rot_mat = geom.euler_matrix(angles[0], angles[1], angles[2])

    rigid = sitk.Euler3DTransform()
    rigid.SetCenter(aff_transform.GetCenter())
    rigid.SetTranslation(aff_transform.GetTranslation())
    # Write a translation-only transform
    sitk.WriteTransform(rigid, translation_mat_file)
    # Write the full rigid (translation + rotation) transform
    rigid.SetMatrix(tuple(rot_mat[:3, :3].flatten(order='C')))
    sitk.WriteTransform(rigid, rigid_mat_file)
    # Write the inverse rigid transform
    sitk.WriteTransform(rigid.GetInverse(), inverse_mat_file)

    if False in (
        op.exists(rigid_mat_file),
        op.exists(translation_mat_file),
        op.exists(inverse_mat_file),
    ):
        raise Exception('unable to create rigid AC-PC transform')
    return rigid_mat_file, inverse_mat_file, translation_mat_file
