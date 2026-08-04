"""Temporal SNR for a diffusion series, computed over the b=0 volumes.

A DWI series has no resting baseline, so "temporal" SNR only means anything over
volumes that should look alike. The b=0 volumes are the only such set: same
contrast, acquired throughout the run, so their variance is dominated by noise
and residual motion rather than by diffusion weighting.

Computing it over all volumes instead would measure diffusion contrast, not
noise, and would look worst exactly where the data is most informative. The
number of b=0 volumes is therefore reported alongside the map -- with only a
handful the estimate is noisy and should not be over-read.
"""

import nibabel as nb
import numpy as np
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)


class _DWITSNRInputSpec(BaseInterfaceInputSpec):
    dwi_file = File(exists=True, mandatory=True)
    bval_file = File(exists=True, mandatory=True)
    mask_file = File(exists=True)
    b0_threshold = traits.Int(100, usedefault=True)
    out_file = File('tsnr.nii.gz', usedefault=True)


class _DWITSNROutputSpec(TraitedSpec):
    out_file = File(exists=True)
    n_b0 = traits.Int()
    median_tsnr = traits.Float()


class DWITSNR(SimpleInterface):
    """Voxelwise mean/SD across the b=0 volumes."""

    input_spec = _DWITSNRInputSpec
    output_spec = _DWITSNROutputSpec

    def _run_interface(self, runtime):
        import os

        from nipype.interfaces.base import isdefined

        img = nb.load(self.inputs.dwi_file)
        bvals = np.loadtxt(self.inputs.bval_file).ravel()
        b0_idx = np.flatnonzero(bvals < self.inputs.b0_threshold)

        if b0_idx.size < 2:
            # One b=0 gives no variance estimate at all. Emit an explicit
            # zero map rather than a silently meaningless one.
            from nipype import logging

            logging.getLogger('nipype.interface').warning(
                'DWITSNR: %d b=0 volume(s) found; TSNR needs at least 2. '
                'Writing an empty map.',
                b0_idx.size,
            )
            tsnr = np.zeros(img.shape[:3], dtype='float32')
        else:
            # index one volume at a time: nibabel's array proxy rejects fancy
            # indexing, and a 279-volume series should not be loaded whole
            acc = None
            sq = None
            for i in b0_idx:
                vol = np.asanyarray(img.dataobj[..., int(i)], dtype='float32')
                acc = vol if acc is None else acc + vol
                sq = vol**2 if sq is None else sq + vol**2
            n = float(b0_idx.size)
            mean = acc / n
            var = np.maximum(sq / n - mean**2, 0.0)
            sd = np.sqrt(var * n / max(n - 1.0, 1.0))  # unbiased
            with np.errstate(divide='ignore', invalid='ignore'):
                tsnr = np.where(sd > 0, mean / sd, 0.0)
            tsnr[~np.isfinite(tsnr)] = 0.0
            tsnr = tsnr.astype('float32')

        if isdefined(self.inputs.mask_file):
            mask = np.asanyarray(nb.load(self.inputs.mask_file).dataobj) > 0
            if mask.shape == tsnr.shape:
                tsnr = tsnr * mask
                inside = tsnr[mask & (tsnr > 0)]
            else:
                inside = tsnr[tsnr > 0]
        else:
            inside = tsnr[tsnr > 0]

        out_file = os.path.join(runtime.cwd, self.inputs.out_file)
        nb.Nifti1Image(tsnr, img.affine, img.header).to_filename(out_file)

        self._results['out_file'] = out_file
        self._results['n_b0'] = int(b0_idx.size)
        self._results['median_tsnr'] = float(np.median(inside)) if inside.size else 0.0
        return runtime
