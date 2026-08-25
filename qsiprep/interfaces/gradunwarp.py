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
