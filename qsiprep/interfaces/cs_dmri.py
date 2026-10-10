"""Image quality measures from cs_dmri.

cs_dmri computes neighboring DWI correlation, the DWI contrast ratio,
within-volume outlier slices and a fixel-coherence index. Its columns sit
beside DSI Studio's in ``desc-image_qc.tsv``; see ``docs/outputs.rst``.
"""

import logging
import os.path as op

import numpy as np
import pandas as pd
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)

LOGGER = logging.getLogger('nipype.interface')


class _CsDmriQCInputSpec(BaseInterfaceInputSpec):
    dwi_file = File(exists=True, mandatory=True, desc='4D DWI series')
    bval_file = File(exists=True, mandatory=True)
    bvec_file = File(exists=True, mandatory=True)
    bvec_convention = traits.Enum(
        'DIPY',
        'FSL',
        usedefault=True,
        desc='DIPY: bvecs in the voxel frame of the image. FSL: as DIPY, but with x '
        'negated when the image is stored in neurological order.',
    )
    mask_file = File(
        exists=True,
        desc='Brain mask on the grid of dwi_file. Without it, the masked measures use '
        "cs_dmri's b=0 mask.",
    )
    b0_threshold = traits.Float(50.0, usedefault=True)
    n_threads = traits.Int(1, usedefault=True, nohash=True)


class _CsDmriQCOutputSpec(TraitedSpec):
    qc_file = File(exists=True, desc='One-row CSV of cs_dmri QC measures')
    warning = traits.Str(desc='Why cs_dmri produced no QC measures. Undefined when it succeeded.')


def _voxel_frame_bvecs(bvecs, affine, convention):
    """Return bvecs in the voxel frame cs_dmri expects."""
    bvecs = np.array(bvecs, dtype=float)
    if convention == 'FSL' and np.linalg.det(affine[:3, :3]) > 0:
        bvecs[:, 0] *= -1
    return bvecs


class CsDmriQC(SimpleInterface):
    """Compute cs_dmri's image quality measures on one DWI series.

    As with DSI Studio's QC, a failure must not fail the run: the measures
    become n/a and the reason is returned in ``warning``.
    """

    input_spec = _CsDmriQCInputSpec
    output_spec = _CsDmriQCOutputSpec

    def _run_interface(self, runtime):
        import cs_dmri as cs

        qc_file = op.join(runtime.cwd, 'cs_dmri_qc.csv')
        try:
            row = self._measure(cs)
        except Exception as exc:  # noqa: BLE001 - QC must not fail the run
            problem = f'cs_dmri QC failed: {type(exc).__name__}: {exc}'
            LOGGER.warning(
                'cs_dmri QC values will be n/a for %s. %s', self.inputs.dwi_file, problem
            )
            self._results['warning'] = problem
            row = dict.fromkeys(cs.qc.columns(), np.nan)
        pd.DataFrame({key: [value] for key, value in row.items()}).to_csv(qc_file, index=False)
        self._results['qc_file'] = qc_file
        return runtime

    def _measure(self, cs):
        import nibabel as nb

        img = nb.load(self.inputs.dwi_file)
        bvals, bvecs = cs.read_bvals_bvecs(self.inputs.bval_file, self.inputs.bvec_file)
        bvecs = _voxel_frame_bvecs(bvecs, img.affine, self.inputs.bvec_convention)
        gtab = cs.GradientTable(bvals, bvecs, b0_threshold=self.inputs.b0_threshold)
        mask = self.inputs.mask_file if isdefined(self.inputs.mask_file) else None
        dwi = cs.DWI.from_nibabel(img, gtab, mask=mask)
        report = dwi.qc(n_threads=self.inputs.n_threads)
        for warning in report.warnings:
            LOGGER.info('cs_dmri QC note for %s: %s', self.inputs.dwi_file, warning)
        return {
            key: (np.nan if value is None else value) for key, value in report.to_dict().items()
        }
