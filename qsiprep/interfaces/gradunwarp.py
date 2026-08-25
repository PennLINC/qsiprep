"""Interfaces to TORTOISE V4's gradient nonlinearity tools.

Three hazards in the upstream binaries drive the shape of these wrappers, all
verified against TORTOISEV4 at ``main``:

1. ``CreateNonlinearityDisplacementMap`` takes the coefficient file as its
   *first* positional argument (``mk_displacement(argv[1], img, is_GE)`` in
   ``src/tools/gradnonlin/mk_displacementMaps.cxx``). A stale, unbuilt copy of
   that file elsewhere in the tree has the arguments reversed.
2. That tool reads ``is_GE`` as ``(bool)(argv[4])`` -- a cast of the *pointer*,
   not its contents -- so passing ``0`` yields **true**. The argument is
   appended only when GE. ``CreateGradientNonlinearityBMatrix`` is different:
   its ``getIsGE()`` uses ``atoi()``, so ``--isGE 0`` is correctly false.
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
        # ``is_GE=(bool)(argv[4])`` casts the pointer: ANY fourth argument is
        # true. Omitting it is the only way to say false.
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
        """The tool names its output for how the nonlinearity was supplied."""
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
