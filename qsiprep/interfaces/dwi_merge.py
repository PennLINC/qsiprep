"""Handle merging and spliting of DSI files."""
import numpy as np
from nipype.interfaces import afni
import os.path as op
from fmriprep.interfaces.bids import _splitext
from nipype.interfaces.base import (BaseInterfaceInputSpec, TraitedSpec, File, SimpleInterface,
                                    InputMultiObject)


class MergeDWIsInputSpec(BaseInterfaceInputSpec):
    dwi_files = InputMultiObject(
        File(exists=True), mandatory=True, desc='list of dwi files')
    original_files = InputMultiObject(
        File(exists=True), mandatory=True, desc='list of dwi files')


class MergeDWIsOutputSpec(TraitedSpec):
    out_dwi = File(desc='the merged dwi image')
    out_bval = File(desc='the merged bvec file')
    out_bvec = File(desc='the merged bval file')


class MergeDWIs(SimpleInterface):
    input_spec = MergeDWIsInputSpec
    output_spec = MergeDWIsOutputSpec

    def _run_interface(self, runtime):
        dwi_file = self.inputs.original_files[0]
        base_dir = op.split(dwi_file)[0]
        bvals = [op.join(base_dir, _splitext(x)[0] + '.bval')
                 for x in self.inputs.original_files]
        bvecs = [op.join(base_dir, _splitext(x)[0] + '.bvec')
                 for x in self.inputs.original_files]

        if len(self.inputs.dwi_files) > 1:
            dwimrg = afni.TCat(in_files=self.inputs.dwi_files, outputtype='NIFTI_GZ')
            self._results['out_dwi'] = dwimrg.run().outputs.out_file
            self._results['out_bval'] = combine_bvals(bvals)
            self._results['out_bvec'] = combine_bvecs(bvecs)
        else:
            self._results['out_dwi'] = dwi_file
            self._results['out_bval'] = bvals[0]
            self._results['out_bvec'] = bvecs[0]
        return runtime


def combine_bvals(bvals):
    """Load, merge and save fsl-style bvals files."""
    collected_vals = []
    for bval_file in bvals:
        collected_vals.append(np.loadtxt(bval_file))
    final_bvals = np.concatenate(collected_vals)
    np.savetxt("restacked.bval", final_bvals, fmt=str("%i"))
    return op.abspath("restacked.bval")


def combine_bvecs(bvecs):
    """Load, merge and save fsl-style bvecs files."""
    collected_vecs = []
    for bvec_file in bvecs:
        collected_vecs.append(np.loadtxt(bvec_file))
    final_bvecs = np.column_stack(collected_vecs)
    np.savetxt("restacked.bvec", final_bvecs, fmt=str("%.8f"))
    return op.abspath("restacked.bvec")
