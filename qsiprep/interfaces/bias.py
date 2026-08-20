"""Keep N4's bias-field fit from being dragged around by dark voxels.

``dwibiascorrect ants`` hands the brain mask to N4 as a *weight* image
(``-w``, never ``-x`` -- see ``mrtrix3/dwibiascorrect/ants.py``). That weight is
binary, so every voxel inside the mask counts equally in a least-squares fit
performed on log intensities -- and the near-zero voxels a b=0 EPI always
contains (susceptibility dropout, the ragged mask edge) become enormous negative
outliers there, which can make the fit diverge.

This dampens the weights rather than shrinking the mask, so the correction is
still applied everywhere the mask covers; only the influence of unreliable
voxels on the fit is reduced.
"""

import os

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

LOGGER = logging.getLogger('nipype.interface')


class _N4WeightMaskInputSpec(BaseInterfaceInputSpec):
    dwi_file = File(exists=True, mandatory=True, desc='4D DWI series')
    bval_file = File(exists=True, mandatory=True, desc='bvals for dwi_file')
    mask_file = File(exists=True, mandatory=True, desc='brain mask to condition')
    b0_threshold = traits.Int(100, usedefault=True)
    background_factor = traits.Float(
        2.0,
        usedefault=True,
        desc='drop in-mask voxels dimmer than this multiple of the background mean',
    )
    min_retained = traits.Float(
        0.5,
        usedefault=True,
        desc='never discard more than this fraction of the mask; bail out instead',
    )
    out_file = File('n4_weights.nii.gz', usedefault=True)


class _N4WeightMaskOutputSpec(TraitedSpec):
    out_file = File(exists=True)
    n_dropped = traits.Int()
    fraction_dropped = traits.Float()


class N4WeightMask(SimpleInterface):
    """Zero out near-background voxels in a mask destined for ``N4 -w``."""

    input_spec = _N4WeightMaskInputSpec
    output_spec = _N4WeightMaskOutputSpec

    def _run_interface(self, runtime):
        img = nb.load(self.inputs.dwi_file)
        bvals = np.loadtxt(self.inputs.bval_file).ravel()
        b0_idx = np.flatnonzero(bvals < self.inputs.b0_threshold)
        if not b0_idx.size:
            raise ValueError(
                f'No b=0 volumes below {self.inputs.b0_threshold} in {self.inputs.bval_file}; '
                'cannot estimate a background level.'
            )

        # mean b=0, built the same way dwibiascorrect builds the image it fits.
        # One volume at a time: the array proxy rejects fancy indexing and these
        # series run to hundreds of volumes.
        acc = None
        for i in b0_idx:
            vol = np.asanyarray(img.dataobj[..., int(i)], dtype='float32')
            acc = vol if acc is None else acc + vol
        mean_b0 = acc / float(b0_idx.size)

        mask_img = nb.load(self.inputs.mask_file)
        mask = np.asanyarray(mask_img.dataobj) > 0
        if mask.shape != mean_b0.shape:
            raise ValueError(f'Mask shape {mask.shape} does not match DWI shape {mean_b0.shape}.')
        if not mask.any():
            raise ValueError(f'{self.inputs.mask_file} is empty.')

        outside = mean_b0[~mask]
        background = float(outside.mean()) if outside.size else 0.0
        floor = self.inputs.background_factor * background

        weights = mask & (mean_b0 > floor)
        n_dropped = int(mask.sum() - weights.sum())
        fraction = n_dropped / float(mask.sum())

        if weights.sum() < self.inputs.min_retained * mask.sum():
            # A mask that is mostly dim means something upstream is already
            # wrong. Reducing it further would hide that, so keep the mask as
            # given and let the bias-field check downstream flag the result.
            LOGGER.warning(
                'N4WeightMask: %.1f%% of the mask falls below %.1f x background '
                '(%.1f); refusing to discard that much and passing the mask '
                'through unchanged.',
                100 * fraction,
                self.inputs.background_factor,
                background,
            )
            weights = mask
            n_dropped = 0
            fraction = 0.0
        else:
            LOGGER.info(
                'N4WeightMask: dropped %d of %d mask voxels (%.2f%%) below %.1f',
                n_dropped,
                int(mask.sum()),
                100 * fraction,
                floor,
            )

        out_file = os.path.join(runtime.cwd, self.inputs.out_file)
        nb.Nifti1Image(weights.astype('uint8'), mask_img.affine, mask_img.header).to_filename(
            out_file
        )

        self._results['out_file'] = out_file
        self._results['n_dropped'] = n_dropped
        self._results['fraction_dropped'] = fraction
        return runtime
