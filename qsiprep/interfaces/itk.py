# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""ITK files handling."""

import glob
import os
import os.path as op
import subprocess
from mimetypes import guess_type

import nibabel as nb
import nilearn.image as nim
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


class _InvertITKAffineInputSpec(BaseInterfaceInputSpec):
    in_file = InputMultiObject(
        File(exists=True),
        mandatory=True,
        desc='ITK affine/rigid transform (.mat); a one-element list is accepted because '
        "ANTs' forward_transforms arrives as a list",
    )


class _InvertITKAffineOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='the inverse transform, as AffineTransform_double_3_3')


class InvertITKAffine(SimpleInterface):
    """Write the inverse of an ITK ``.mat`` affine or rigid transform.

    ANTs' ``reverse_transforms`` is the forward ``.mat`` with an inverse flag that the file
    cannot carry, so a derivative written from it under a ``from-A_to-B`` name would hold the
    ``from-B_to-A`` matrix. This computes the inverse (with ITK's fixed-centre convention
    folded in) and writes a plain ``AffineTransform_double_3_3`` with a zero centre.
    """

    input_spec = _InvertITKAffineInputSpec
    output_spec = _InvertITKAffineOutputSpec

    def _run_interface(self, runtime):
        in_files = self.inputs.in_file
        if len(in_files) != 1:
            raise ValueError(f'InvertITKAffine takes exactly one transform, got {in_files}')
        self._results['out_file'] = invert_itk_affine(
            in_files[0],
            fname_presuffix(in_files[0], suffix='_inverse', newpath=runtime.cwd),
        )
        return runtime


def linear_transform_matrix(transform):
    """Return the 4x4 LPS point map (fixed -> moving) of a linear ITK transform.

    The map is read off where the transform sends the origin and the unit vectors, so
    Euler, affine, translation and composite transforms, their centres of rotation and
    the composite application order all come from ITK rather than from a re-implementation.

    Parameters
    ----------
    transform : str or :obj:`os.PathLike` or :class:`SimpleITK.Transform`
        A transform file (``.txt``, ``.mat`` or ``.h5``) or an in-memory transform.

    Returns
    -------
    :obj:`numpy.ndarray`
        4x4 homogeneous matrix mapping fixed-image points to moving-image points, in LPS mm.
    """
    if not isinstance(transform, sitk.Transform):
        transform = sitk.ReadTransform(str(transform))
    origin = np.array(transform.TransformPoint((0.0, 0.0, 0.0)))
    out = np.eye(4)
    out[:3, 3] = origin
    for axis in range(3):
        unit = [0.0, 0.0, 0.0]
        unit[axis] = 1.0
        out[:3, axis] = np.array(transform.TransformPoint(unit)) - origin
    return out


def linear_transform(transform):
    """Return a linear ITK transform as its typed ``SimpleITK`` object.

    The result answers ``GetMatrix()``, ``GetTranslation()`` and ``GetCenter()`` (and
    ``GetAngleX()`` etc. for a rigid), so callers never index into ``GetParameters()``.

    Parameters
    ----------
    transform : str or :obj:`os.PathLike` or :class:`SimpleITK.Transform`
        A transform file or an in-memory transform. A one-member composite (ANTs' ``.h5``
        output) is unwrapped.

    Returns
    -------
    :class:`SimpleITK.Transform`
        The downcast transform (``Euler3DTransform``, ``AffineTransform``, ...).

    Raises
    ------
    ValueError
        If the transform is a composite with several members, which has no single
        translation; use :func:`linear_transform_matrix` for the combined map.
    """
    if not isinstance(transform, sitk.Transform):
        transform = sitk.ReadTransform(str(transform))
    transform = transform.Downcast()
    if isinstance(transform, sitk.CompositeTransform):
        if transform.GetNumberOfTransforms() != 1:
            raise ValueError(
                f'{transform.GetNumberOfTransforms()} transforms in the composite; '
                'use linear_transform_matrix for the combined map'
            )
        transform = transform.GetNthTransform(0).Downcast()
    return transform


def affine_from_matrix(matrix):
    """Build a zero-centre ``SimpleITK.AffineTransform`` from a 4x4 LPS point map.

    Parameters
    ----------
    matrix : array-like
        4x4 homogeneous matrix mapping fixed-image points to moving-image points, in LPS mm.

    Returns
    -------
    :class:`SimpleITK.AffineTransform`
        The same map, with the centre of rotation at the origin.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    out = sitk.AffineTransform(3)
    out.SetMatrix(matrix[:3, :3].ravel().tolist())
    out.SetTranslation(matrix[:3, 3].tolist())
    return out


def invert_itk_affine(in_file, out_file):
    """Invert an ITK ``.mat`` affine or rigid transform, writing ``AffineTransform_double_3_3``.

    Parameters
    ----------
    in_file : str
        The transform to invert.
    out_file : str
        Where to write the inverse, as a zero-centre affine.

    Returns
    -------
    str
        ``out_file``.
    """
    inverse = np.linalg.inv(linear_transform_matrix(in_file))
    sitk.WriteTransform(affine_from_matrix(inverse), out_file)
    return out_file


class _GuardRefinedTransformInputSpec(BaseInterfaceInputSpec):
    initial_transform = File(
        exists=True, mandatory=True, desc='the linear transform the refinement started from'
    )
    refined_transform = File(exists=True, mandatory=True, desc='the refined linear transform')
    reference_image = File(
        exists=True,
        mandatory=True,
        desc='image whose field-of-view centre the shift is measured at',
    )
    max_shift_mm = traits.Float(
        10.0, usedefault=True, desc='largest move of the FOV centre a refinement may make'
    )
    max_rotation_deg = traits.Float(
        10.0, usedefault=True, desc='largest rotation a refinement may add'
    )


class _GuardRefinedTransformOutputSpec(TraitedSpec):
    out_transform = File(exists=True, desc='the refined transform, or the initial one if rejected')
    accepted = traits.Bool(desc='whether the refinement was kept')
    shift_mm = traits.Float(desc='how far the refinement moved the FOV centre')
    rotation_deg = traits.Float(desc='the rotation the refinement added')


class GuardRefinedTransform(SimpleInterface):
    """Keep a refined linear transform only if it stayed near the transform it started from.

    A registration that refines an earlier result (a second pass against a better target,
    initialised from the first pass) should move by at most the effect it corrects. With
    random metric sampling the optimiser occasionally runs away instead, and a transform tens
    of millimetres off would silently ruin everything downstream, so a refinement that moves
    the reference's FOV centre by more than ``max_shift_mm`` or rotates by more than
    ``max_rotation_deg`` is dropped in favour of the initial transform, with a warning.
    """

    input_spec = _GuardRefinedTransformInputSpec
    output_spec = _GuardRefinedTransformOutputSpec

    def _run_interface(self, runtime):
        initial = sitk.ReadTransform(self.inputs.initial_transform)
        refined = sitk.ReadTransform(self.inputs.refined_transform)
        img = nb.load(self.inputs.reference_image)
        centre_ras = img.affine[:3, :3] @ (np.array(img.shape[:3]) / 2.0) + img.affine[:3, 3]
        centre = tuple(float(v) for v in np.diag([-1.0, -1.0, 1.0]) @ centre_ras)  # LPS
        shift = np.array(refined.TransformPoint(centre)) - np.array(initial.TransformPoint(centre))
        rot = linear_transform_matrix(refined)[:3, :3] @ np.linalg.inv(
            linear_transform_matrix(initial)[:3, :3]
        )
        angle = float(np.degrees(np.arccos(np.clip((np.trace(rot) - 1) / 2, -1, 1))))
        self._results['shift_mm'] = float(np.linalg.norm(shift))
        self._results['rotation_deg'] = angle
        accepted = (
            self._results['shift_mm'] <= self.inputs.max_shift_mm
            and angle <= self.inputs.max_rotation_deg
        )
        self._results['accepted'] = accepted
        if not accepted:
            LOGGER.warning(
                'The refined transform moved %.1f mm / %.1f deg away from the transform it '
                'started from (limits %.1f mm / %.1f deg); keeping the initial transform.',
                self._results['shift_mm'],
                angle,
                self.inputs.max_shift_mm,
                self.inputs.max_rotation_deg,
            )
        source = self.inputs.refined_transform if accepted else self.inputs.initial_transform
        out_file = fname_presuffix(source, newpath=runtime.cwd, prefix='guarded_')
        sitk.WriteTransform(sitk.ReadTransform(source), out_file)
        self._results['out_transform'] = out_file
        return runtime


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
        translation_img = nb.load(self.inputs.translation_image)
        rigid_img = nb.load(self.inputs.rigid_image)
        # combine images so crop offset captures both
        sum_img = nim.math_img('a + b', a=translation_img, b=rigid_img)
        _, crop_offset = nim.crop_img(sum_img, return_offset=True)

        # Call composer
        compose_view(
            plot_acpc(
                translation_img,
                'moving-image',
                estimate_brightness=True,
                label='Original',
                crop_offset=crop_offset,
                compress=False,
            ),
            plot_acpc(
                rigid_img,
                'fixed-image',
                estimate_brightness=True,
                label='AC-PC',
                crop_offset=crop_offset,
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
    """Apply ANTs' antsApplyTransforms to the input image.

    All inputs are zipped in one tuple to make it digestible by
    multiprocessing's map.
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
    """Arrange the list of transforms that should be applied to each input file.

    Convenience method. Not needed in qsiprep.
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
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)
    LOGGER.info(' '.join(cmd))
    out, err = proc.communicate()

    if proc.returncode != 0:
        LOGGER.error(
            'CompositeTransformUtil failed (exit %d): %s',
            proc.returncode,
            err.decode(),
        )

    # ANTs v2.4.x names files as {index}_{prefix}_{ClassName}.{ext},
    # while v2.5+ uses {prefix}_{index}_{ClassName}.{ext}.
    # ITK 5.4+ may also report MatrixOffsetTransformBase instead of AffineTransform.
    # Use glob patterns to find the actual output files regardless of naming convention.
    affine_candidates = sorted(
        glob.glob(op.join(cwd, '*disassemble*Affine*.mat'))
        + glob.glob(op.join(cwd, '*disassemble*MatrixOffset*.mat'))
    )
    warp_candidates = sorted(glob.glob(op.join(cwd, '*disassemble*DisplacementField*.nii.gz')))

    if not affine_candidates:
        raise Exception(
            'Unable to unpack composite transform. '
            f'stdout: {out.decode()}, stderr: {err.decode()}, '
            f'files in cwd: {os.listdir(cwd)}'
        )

    transforms = [affine_candidates[0]]
    if warp_candidates:
        transforms.append(warp_candidates[0])
    return transforms


def compose_affines(reference_image, affine_list, output_file):
    """Use antsApplyTransforms to get a single affine from multiple affines."""
    cmd = f'antsApplyTransforms -d 3 -r {reference_image} -o Linear[{output_file}, 1] '
    cmd += ' '.join([f'--transform {trf}' for trf in affine_list])
    os.system(cmd)
    assert os.path.exists(output_file)
    return output_file


def itk_affine_to_rigid(transform_file, cwd):
    """Convert an ITK linear transform from affine to rigid.

    Uses c3d_affine_tool and FSL's aff2rigid.
    """
    rigid_mat_file = cwd + '/6DOFrigid.mat'
    translation_mat_file = cwd + '/translation.mat'
    inverse_mat_file = cwd + '/6DOFinverse.mat'
    aff_transform = linear_transform(transform_file)

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
