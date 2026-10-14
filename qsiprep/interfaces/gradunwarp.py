"""Interfaces to TORTOISE V4's gradient nonlinearity tools.

Three hazards in the upstream binaries drive the shape of these wrappers, all
verified against TORTOISEV4 at ``main``:

1. ``CreateNonlinearityDisplacementMap`` takes the coefficient file as its
   *first* positional argument (``mk_displacement(argv[1], img, is_GE)`` in
   ``src/tools/gradnonlin/mk_displacementMaps.cxx``, the file
   ``CMakeLists.txt:251`` actually builds). A stale, unbuilt duplicate at
   ``src/tools/CreateNonlinearityDisplacementMap/`` has the arguments
   reversed.
2. That built tool reads a supplied fourth argument via
   ``(bool)atoi(argv[4])``, so passing ``"0"`` correctly means false; the
   argument is appended only when GE. (The stale, unbuilt duplicate in point
   1 instead reads ``is_GE=(bool)(argv[4])`` -- a cast of the *pointer*, not
   its contents, under which passing ``"0"`` would incorrectly evaluate
   true -- but that file governs nothing we invoke.)
   ``CreateGradientNonlinearityBMatrix`` uses a required ``--isGE`` flag
   instead: its ``getIsGE()`` also uses ``atoi()``, so ``--isGE 0`` is
   correctly false.
3. ``mk_displacement`` returns the field TORTOISE names ``gradwarp_field_inv``,
   and that is the file ``FINALDATA.cxx:548`` composes and ``DRBUDDI.cxx:141``
   resamples with. It must **not** be inverted here.
"""

import os.path as op

import nibabel as nb
import numpy as np
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    CommandLine,
    CommandLineInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)

#: Displacement components zeroed for each warp dimensionality, mirroring
#: ``TORTOISE.cxx:1994-2012``.
_ZEROED_COMPONENTS = {'3D': (), '2D': (2,), '1D': (0, 1)}


class _MaskWarpDimensionsInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='ITK displacement field')
    warp_dim = traits.Enum(
        '3D',
        '2D',
        '1D',
        usedefault=True,
        desc='Which displacement components to keep. "3D" keeps all; "2D" '
        'zeroes the through-plane component; "1D" keeps only it.',
    )


class _MaskWarpDimensionsOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='Displacement field with components zeroed')


class MaskWarpDimensions(SimpleInterface):
    """Zero displacement components a scanner has already corrected."""

    input_spec = _MaskWarpDimensionsInputSpec
    output_spec = _MaskWarpDimensionsOutputSpec

    def _run_interface(self, runtime):
        img = nb.load(self.inputs.in_file)
        data = np.array(img.dataobj, dtype='float32')
        # ITK vector fields are (X, Y, Z, 1, 3); indexing the last axis works
        # whether or not the singleton dimension is present.
        for component in _ZEROED_COMPONENTS[self.inputs.warp_dim]:
            data[..., component] = 0
        out_file = op.join(runtime.cwd, 'gradwarp_field_masked.nii')
        nb.Nifti1Image(data, img.affine, img.header).to_filename(out_file)
        self._results['out_file'] = out_file
        return runtime


class _InvertDisplacementFieldInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='ITK displacement field')
    iterations = traits.Int(
        10, usedefault=True, desc='Fixed-point iterations; 10 is far past convergence for gradwarp'
    )


class _InvertDisplacementFieldOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='ITK displacement field mapping the other way')


class InvertDisplacementField(SimpleInterface):
    """Invert a small, smooth displacement field by fixed-point iteration.

    An ITK field ``d`` sends a point ``x`` to ``x + d(x)``. Its inverse ``u``
    satisfies ``u(y) = -d(y + u(y))``, a contraction whenever ``d`` varies
    slowly compared to its own size -- true of gradwarp fields, which are a
    few millimetres at most and vary over centimetres. The field is sampled
    trilinearly and taken to be zero outside its own grid.
    """

    input_spec = _InvertDisplacementFieldInputSpec
    output_spec = _InvertDisplacementFieldOutputSpec

    def _run_interface(self, runtime):
        img = nb.load(self.inputs.in_file)
        data = np.asarray(img.dataobj, dtype='float64')
        out_file = op.join(runtime.cwd, 'displacement_field_inverse.nii')
        nb.Nifti1Image(
            invert_displacement_field(data, img.affine, self.inputs.iterations).astype('float32'),
            img.affine,
            img.header,
        ).to_filename(out_file)
        self._results['out_file'] = out_file
        return runtime


def invert_displacement_field(data, affine, iterations=10):
    """Compute the fixed-point inverse of an ITK vector field on its own grid.

    ``data`` is (X, Y, Z, [1,] 3) in ITK's LPS millimetres; the result has the
    same shape and convention.
    """
    from scipy.ndimage import map_coordinates

    shape = data.shape
    lps_to_ras = np.array([-1.0, -1.0, 1.0])
    field = data.reshape(shape[:3] + (3,)) * lps_to_ras
    inv_affine = np.linalg.inv(affine)
    ijk = np.indices(shape[:3]).reshape(3, -1).astype('float64')
    xyz = affine[:3, :3] @ ijk + affine[:3, 3:4]
    u = np.zeros_like(xyz)
    for _ in range(iterations):
        probe = inv_affine[:3, :3] @ (xyz + u) + inv_affine[:3, 3:4]
        u = -np.stack(
            [
                map_coordinates(field[..., c], probe, order=1, mode='constant', cval=0.0)
                for c in range(3)
            ]
        )
    return (u.T * lps_to_ras).reshape(shape)


class _CreateNonlinearityDisplacementMapInputSpec(CommandLineInputSpec):
    coeff_file = File(
        exists=True,
        mandatory=True,
        argstr='%s',
        position=0,
        desc='Scanner gradient coefficient file (.grad, .dat, .gc) or gcal file',
    )
    ref_image = File(
        exists=True,
        mandatory=True,
        argstr='%s',
        position=1,
        desc='NIfTI defining the grid the field is generated on',
    )
    out_field = traits.Str(
        'gradwarp_field.nii',
        usedefault=True,
        argstr='%s',
        position=2,
        desc='Output displacement field name. Must end in .nii for the ITK writer.',
    )
    # No argstr: appended in _parse_inputs only when True. See module docstring.
    is_ge = traits.Bool(False, usedefault=True, desc='Scanner is GE')


class _CreateNonlinearityDisplacementMapOutputSpec(TraitedSpec):
    out_field = File(exists=True, desc='Gradwarp displacement field, native space')


class CreateNonlinearityDisplacementMap(CommandLine):
    """Expand gradient coefficients into a displacement field.

    The output is TORTOISE's ``gradwarp_field_inv``, which is what gets
    composed and what resamples b=0 images. Do not invert it.
    """

    input_spec = _CreateNonlinearityDisplacementMapInputSpec
    output_spec = _CreateNonlinearityDisplacementMapOutputSpec
    _cmd = 'CreateNonlinearityDisplacementMap'

    def _parse_inputs(self, skip=None):
        parsed = super()._parse_inputs(skip=skip)
        # The built tool reads a supplied fourth argument via
        # ``(bool)atoi(argv[4])`` (see module docstring): both a "0" and
        # omitting the argument entirely mean false. We omit it, matching the
        # existing "appended only when GE" convention.
        if self.inputs.is_ge:
            parsed.append('1')
        return parsed

    def _list_outputs(self):
        return {'out_field': op.abspath(self.inputs.out_field)}


class _CreateGradientNonlinearityBMatrixInputSpec(CommandLineInputSpec):
    final_image = File(
        exists=True,
        mandatory=True,
        copyfile=True,
        argstr='-f %s',
        desc='Final preprocessed b=0, in the output space. The tool writes its '
        'outputs beside this file, so it is staged into the node directory.',
    )
    nonlinearity = File(
        exists=True,
        mandatory=True,
        argstr='-g %s',
        desc='Coefficient file or ITK gradwarp displacement field',
    )
    initial_image = File(
        exists=True,
        argstr='-i %s',
        desc='Raw native-space b=0. Omitted means the final image is native.',
    )
    is_ge = traits.Bool(False, usedefault=True, argstr='--isGE %d', desc='Scanner is GE')


class _CreateGradientNonlinearityBMatrixOutputSpec(TraitedSpec):
    grad_dev = File(exists=True, desc='9-component gradient deviation (L) map')
    gradwarp_field = File(exists=True, desc='Gradwarp displacement field')


class CreateGradientNonlinearityBMatrix(CommandLine):
    """Compute the voxelwise gradient deviation tensor.

    Emits the HCP-style 9-component L matrix per voxel. Applied downstream as
    ``L @ g``: because L carries scaling and shear, not just rotation, both the
    b-vector and the b-value deviate per voxel.
    """

    input_spec = _CreateGradientNonlinearityBMatrixInputSpec
    output_spec = _CreateGradientNonlinearityBMatrixOutputSpec
    _cmd = 'CreateGradientNonlinearityBMatrix'

    def _graddev_suffix(self):
        """Return the output suffix the tool uses for how the nonlinearity was supplied."""
        if '.nii' in op.basename(self.inputs.nonlinearity):
            return '_graddev_f.nii'
        return '_graddev_c.nii'

    def _list_outputs(self):
        # Outputs are named from the -f input's stem, in its directory. Because
        # final_image is copyfile=True, that directory is this node's cwd.
        staged = op.abspath(op.basename(self.inputs.final_image))
        stem = staged[: staged.rfind('.nii')]
        return {
            'grad_dev': stem + self._graddev_suffix(),
            'gradwarp_field': stem + '_gradwarp_field.nii',
        }


def _mask_on_grid(mask_file, shape, affine):
    """Return a boolean brain mask resampled (nearest neighbor) onto a 3D grid."""
    from nibabel.processing import resample_from_to

    mask = nb.load(mask_file)
    if mask.shape[:3] != tuple(shape) or not np.allclose(mask.affine, affine, atol=1e-3):
        mask = resample_from_to(mask, (tuple(shape), affine), order=0)
    return np.asanyarray(mask.dataobj) > 0


def graddev_rotation_degrees(tensors):
    """Return the rotation angle, in degrees, of each 3x3 gradient deviation tensor.

    The rotation is the orthogonal factor of the polar decomposition, the part of
    the tensor left after its scaling and shear are removed. Its angle is the same
    for a tensor and its transpose, so it does not depend on which side the tensor
    multiplies the gradient from.
    """
    u, _, vt = np.linalg.svd(tensors)
    rotations = u @ vt
    # An improper orthogonal factor is a reflection; flip its last axis.
    improper = np.linalg.det(rotations) < 0
    u[improper, :, -1] *= -1
    rotations[improper] = u[improper] @ vt[improper]
    cos_angle = (np.trace(rotations, axis1=-2, axis2=-1) - 1) / 2
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def _summary(prefix, values):
    if values.size == 0:
        return {}
    return {
        f'{prefix}_median': [float(np.median(values))],
        f'{prefix}_p95': [float(np.percentile(values, 95))],
        f'{prefix}_maximum': [float(np.max(values))],
    }


def gradient_nonlinearity_qc(grad_dev, grad_dev_mask, gradwarp_field=None, gradwarp_mask=None):
    """Summarize gradient nonlinearity inside the brain, as ``image_qc.tsv`` columns.

    Parameters
    ----------
    grad_dev : str
        The 9-component gradient deviation tensor map.
    grad_dev_mask : str
        A brain mask in the space of ``grad_dev``.
    gradwarp_field : str or None, optional
        The applied gradwarp displacement field (ITK, mm).
    gradwarp_mask : str or None, optional
        A brain mask in the space of ``gradwarp_field``.

    Returns
    -------
    dict
        The median, 95th percentile and maximum of the applied displacement length
        (``gradwarp_displacement_mm_*``, only with a field) and of the rotation angle
        of the deviation tensor (``graddev_rotation_deg_*``).
    """
    values = {}
    if gradwarp_field and gradwarp_mask:
        field_img = nb.load(gradwarp_field)
        vectors = np.asanyarray(field_img.dataobj, dtype=np.float64)
        vectors = vectors.reshape(field_img.shape[:3] + (-1,))
        mask = _mask_on_grid(gradwarp_mask, field_img.shape[:3], field_img.affine)
        values.update(_summary('gradwarp_displacement_mm', np.linalg.norm(vectors[mask], axis=-1)))

    graddev_img = nb.load(grad_dev)
    tensors = np.asanyarray(graddev_img.dataobj, dtype=np.float64)
    mask = _mask_on_grid(grad_dev_mask, tensors.shape[:3], graddev_img.affine)
    values.update(
        _summary('graddev_rotation_deg', graddev_rotation_degrees(tensors[mask].reshape(-1, 3, 3)))
    )
    return values
