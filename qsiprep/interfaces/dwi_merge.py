"""Handle merging and spliting of DSI files."""
import numpy as np
from nipype.interfaces import afni
from os.path import abspath
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
    """Convert a phase1, phase2 into a difference map."""

    input_spec = MergeDWIsInputSpec
    output_spec = MergeDWIsOutputSpec

    def _run_interface(self, runtime):
        bvals = [_splitext(x)[0] + '.bval' for x in self.inputs.original_files]
        bvecs = [_splitext(x)[0] + '.bvec' for x in self.inputs.original_files]

        if len(self.inputs.dwi_files) > 1:
            dwimrg = afni.Tcat(in_files=self.inputs.dwi_files, output_type='NIFTI_GZ')
            self._results['out_dwi'] = dwimrg.run().outputs.out_file
            self._results['out_bval'] = combine_bvals(bvals)
            self._results['out_bvec'] = combine_bvecs(bvecs)
        else:
            self._results['out_dwi'] = self.inputs.dwi_files
            self._results['out_bval'] = bvals
            self._results['out_bvec'] = bvecs
        return runtime


def combine_bvals(bvals):
    """Load, merge and save fsl-style bvals files."""
    collected_vals = []
    for bval_file in bvals:
        collected_vals.append(np.loadtxt(bval_file))
    final_bvals = np.concatenate(collected_vals)
    np.savetxt(
        "restacked.bval",
        np.concatenate([np.array([0]), final_bvals]),
        fmt=str("%i"))
    return abspath("restacked.bval")


def combine_bvecs(bvecs):
    """Load, merge and save fsl-style bvecs files."""
    collected_vecs = []
    for bvec_file in bvecs:
        collected_vecs.append(np.loadtxt(bvec_file))
    final_bvecs = np.column_stack(collected_vecs)
    np.savetxt("restacked.bvec", final_bvecs, fmt=str("%.8f"))
    return abspath("restacked.bvec")
