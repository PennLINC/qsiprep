"""Handle merging and spliting of DSI files."""
import numpy as np
from nipype.interfaces import afni
import os.path as op
from nipype.interfaces.base import (BaseInterfaceInputSpec, TraitedSpec, File, SimpleInterface,
                                    InputMultiObject, traits)
from nipype.utils.filemanip import fname_presuffix
import nibabel as nb


class MergeDWIsInputSpec(BaseInterfaceInputSpec):
    dwi_files = InputMultiObject(
        File(exists=True), mandatory=True, desc='list of dwi files')
    bids_dwi_files = InputMultiObject(
        File(exists=True), mandatory=True, desc='list of original (BIDS) dwi files')
    bval_files = InputMultiObject(
        File(exists=True), mandatory=True, desc='list of bval files')
    bvec_files = InputMultiObject(
        File(exists=True), mandatory=True, desc='list of bvec files')


class MergeDWIsOutputSpec(TraitedSpec):
    out_dwi = File(desc='the merged dwi image')
    out_bval = File(desc='the merged bvec file')
    out_bvec = File(desc='the merged bval file')
    original_images = traits.List()


class MergeDWIs(SimpleInterface):
    input_spec = MergeDWIsInputSpec
    output_spec = MergeDWIsOutputSpec

    def _run_interface(self, runtime):
        bvals = self.inputs.bval_files
        bvecs = self.inputs.bvec_files

        def get_nvols(img):
            shape = nb.load(img).shape
            if len(shape) < 4:
                return 1
            return shape[3]

        if len(self.inputs.dwi_files) > 1:
            dwimrg = afni.TCat(in_files=self.inputs.dwi_files, outputtype='NIFTI_GZ')
            merged_fname = dwimrg.run().outputs.out_file
            self._results['out_dwi'] = merged_fname
            out_bvec = fname_presuffix(merged_fname, suffix=".bvec", use_ext=False,
                                       newpath=runtime.cwd)
            out_bval = fname_presuffix(merged_fname, suffix=".bval", use_ext=False,
                                       newpath=runtime.cwd)
            self._results['out_bval'] = combine_bvals(bvals, output_file=out_bval)
            self._results['out_bvec'] = combine_bvecs(bvecs, output_file=out_bvec)
            sources = []
            for img in self.inputs.bids_dwi_files:
                sources += [img] * get_nvols(img)
            self._results['original_images'] = sources
        else:
            dwi_file = self.inputs.dwi_files[0]
            bids_dwi_file = self.inputs.bids_dwi_files[0]
            self._results['out_dwi'] = dwi_file
            self._results['out_bval'] = bvals[0]
            self._results['out_bvec'] = bvecs[0]
            self._results['original_images'] = [bids_dwi_file] * get_nvols(bids_dwi_file)
        return runtime


def combine_bvals(bvals, output_file="restacked.bval"):
    """Load, merge and save fsl-style bvals files."""
    collected_vals = []
    for bval_file in bvals:
        collected_vals.append(np.atleast_1d(np.loadtxt(bval_file)))
    final_bvals = np.concatenate(collected_vals)
    np.savetxt(output_file, final_bvals, fmt=str("%i"))
    return op.abspath(output_file)


def combine_bvecs(bvecs, output_file="restacked.bvec"):
    """Load, merge and save fsl-style bvecs files."""
    collected_vecs = []
    for bvec_file in bvecs:
        collected_vecs.append(np.loadtxt(bvec_file))
    final_bvecs = np.column_stack(collected_vecs)
    np.savetxt(output_file, final_bvecs, fmt=str("%.8f"))
    return op.abspath(output_file)
