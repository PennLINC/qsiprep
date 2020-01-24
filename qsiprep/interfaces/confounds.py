# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Handling confounds
^^^^^^^^^^^^^^^^^^

    >>> import os
    >>> import pandas as pd

"""
import os
import re
import json
import numpy as np
import pandas as pd
from nipype import logging
from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec, File, isdefined,
    SimpleInterface, InputMultiObject
)
from .gradients import concatenate_bvecs, concatenate_bvals

from .niworkflows import dMRIPlot

LOGGER = logging.getLogger('nipype.interface')


class GatherConfoundsInputSpec(BaseInterfaceInputSpec):
    fd = File(exists=True, desc='input framewise displacement')
    motion = File(exists=True, desc='input motion parameters')
    sliceqc_file = File(exists=True, desc='output from sliceqc')
    original_files = traits.List(desc='original grouping of each volume')
    original_bvecs = InputMultiObject(File(exists=True), desc='original bvec files')
    original_bvals = InputMultiObject(File(exists=True), desc='originals bval files')
    denoising_confounds = File(exists=True, desc='descriptive statistics from denoising')


class GatherConfoundsOutputSpec(TraitedSpec):
    confounds_file = File(exists=True, desc='output confounds file')
    confounds_list = traits.List(traits.Str, desc='list of headers')


class GatherConfounds(SimpleInterface):
    """
    Combine various sources of confounds in one TSV file

    """
    input_spec = GatherConfoundsInputSpec
    output_spec = GatherConfoundsOutputSpec

    def _run_interface(self, runtime):
        combined_out, confounds_list = _gather_confounds(
            fdisp=self.inputs.fd,
            sliceqc_file=self.inputs.sliceqc_file,
            motion=self.inputs.motion,
            original_files=self.inputs.original_files,
            original_bvals=concatenate_bvals(self.inputs.original_bvals, None),
            original_bvecs=concatenate_bvecs(self.inputs.original_bvecs),
            denoising_confounds=self.inputs.denoising_confounds,
            newpath=runtime.cwd,
        )
        self._results['confounds_file'] = combined_out
        self._results['confounds_list'] = confounds_list
        return runtime


def _gather_confounds(fdisp=None, motion=None, sliceqc_file=None, newpath=None,
                      original_files=None, original_bvals=None, original_bvecs=None,
                      denoising_confounds=None):
    """
    Load confounds from the filenames, concatenate together horizontally
    and save new file.

    >>> from tempfile import TemporaryDirectory
    >>> tmpdir = TemporaryDirectory()
    >>> os.chdir(tmpdir.name)
    >>> pd.DataFrame({'FramewiseDisplacement': [0.1]}).to_csv('FD.txt', index=False, na_rep='n/a')
    >>> pd.DataFrame({'trans_x': [0.2], 'trans_y': [0.3], 'trans_z': [0.4],
    ...               'rot_x': [0.5], 'rot_y': [0.6], 'rot_z': [0.7]}).to_csv(
    ...                   'spm_motion.tsv', index=False, na_rep='n/a')
    >>> out_file, confound_list = _gather_confounds('FD.txt', 'spm_motion.tsv')
    >>> confound_list
    ['Framewise displacement', 'Motion parameters']

    >>> pd.read_csv(out_file, sep='\s+', index_col=None,
    ...             engine='python')  # doctest: +NORMALIZE_WHITESPACE
       framewise_displacement trans_x,trans_y,trans_z,rot_x,rot_y,rot_z
    0                     0.1                   0.2,0.3,0.4,0.5,0.6,0.7


    """

    def less_breakable(a_string):
        ''' hardens the string to different envs (i.e. case insensitive, no whitespace, '#' '''
        return ''.join(a_string.split()).strip('#')

    # Taken from https://stackoverflow.com/questions/1175208/
    # If we end up using it more than just here, probably worth pulling in a well-tested package
    def camel_to_snake(name):
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    def _adjust_indices(left_df, right_df):
        # This forces missing values to appear at the beggining of the DataFrame
        # instead of the end
        index_diff = len(left_df.index) - len(right_df.index)
        if index_diff > 0:
            right_df.index = range(index_diff,
                                   len(right_df.index) + index_diff)
        elif index_diff < 0:
            left_df.index = range(-index_diff,
                                  len(left_df.index) - index_diff)

    all_files = []
    confounds_list = []
    for confound, name in ((fdisp, 'Framewise displacement'),
                           (motion, 'Motion parameters')):
        if confound is not None and isdefined(confound):
            confounds_list.append(name)
            if os.path.exists(confound) and os.stat(confound).st_size > 0:
                all_files.append(confound)

    confounds_data = pd.DataFrame()
    for file_name in all_files:  # assumes they all have headings already
        new = pd.read_csv(file_name, sep="\t")
        for column_name in new.columns:
            new.rename(columns={column_name: camel_to_snake(less_breakable(column_name))},
                       inplace=True)

        _adjust_indices(confounds_data, new)
        confounds_data = pd.concat((confounds_data, new), axis=1)

    # Add in the sliceqc measures
    if isdefined(sliceqc_file) and sliceqc_file is not None:
        if sliceqc_file.endswith(".npz"):
            sqc = np.load(sliceqc_file)
            confounds_data['hmc_r2'] = sqc['wb_r2s']
            confounds_data['hmc_xcorr'] = sqc['wb_xcorrs']
            confounds_list += ['hmc_r2', 'hmc_xcorr']
        else:
            sqc = np.loadtxt(sliceqc_file, skiprows=1)
            confounds_data['eddy_stdevs'] = sqc.sum(axis=1)

    if newpath is None:
        newpath = os.getcwd()

    if original_files is not None and isdefined(original_files):
        file_array = np.array([os.path.split(fname)[1] for fname in original_files])
        confounds_data['original_file'] = file_array
        confounds_list += ['original_file']

    if original_bvecs is not None and isdefined(original_bvecs):
        confounds_data['grad_x'] = original_bvecs[:, 0]
        confounds_data['grad_y'] = original_bvecs[:, 1]
        confounds_data['grad_z'] = original_bvecs[:, 2]

    if original_bvals is not None and isdefined(original_bvals):
        confounds_data['bval'] = original_bvals

    if denoising_confounds is not None and isdefined(denoising_confounds):
        denoising = pd.read_csv(denoising_confounds)
        denoising.original_file = denoising.original_file.str.split("/").str[-1]
        denoising_check = denoising[['original_bx', 'original_by', 'original_bz',
                                     'original_bval']]
        confound_check = confounds_data[['grad_x', 'grad_y', 'grad_z', 'bval']]

        # Check that the gradients and original files match after recombining
        if not np.allclose(denoising_check.to_numpy(), confound_check.to_numpy()) or \
                not (denoising['original_file'] == confounds_data['original_file']).all():
            raise Exception("Gradients or original files don't match. File a bug report!")
        denoising.drop(columns=['original_file', 'original_bval', 'original_bx',
                                'original_by', 'original_bz'],
                       inplace=True)
        confounds_data = pd.concat([confounds_data, denoising], axis=1)
        confounds_list += denoising.columns.to_list()

    combined_out = os.path.join(newpath, 'confounds.tsv')
    confounds_data.to_csv(combined_out, sep='\t', index=False, na_rep='n/a')

    return combined_out, confounds_list


class DMRISummaryInputSpec(BaseInterfaceInputSpec):
    confounds_file = File(exists=True,
                          desc="BIDS' _confounds.tsv file")
    sliceqc_file = File(exists=True,
                        desc="output from SliceQC")
    sliceqc_mask = File(exists=True, desc='Mask')

    str_or_tuple = traits.Either(
        traits.Str,
        traits.Tuple(traits.Str, traits.Either(None, traits.Str)),
        traits.Tuple(traits.Str, traits.Either(None, traits.Str), traits.Either(None, traits.Str)))
    confounds_list = traits.List(
        str_or_tuple, minlen=1,
        desc='list of headers to extract from the confounds_file')
    bval_files = InputMultiObject(File(exists=True), desc='bvals files')
    orig_bvecs = InputMultiObject(File(exists=True), desc='original bvecs file')


class DMRISummaryOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='written file path')
    carpetplot_json = File(exists=True)


class DMRISummary(SimpleInterface):
    input_spec = DMRISummaryInputSpec
    output_spec = DMRISummaryOutputSpec

    def _run_interface(self, runtime):
        self._results['out_file'] = os.path.join(runtime.cwd, 'dmriplot.svg')

        dataframe = pd.read_csv(
            self.inputs.confounds_file,
            sep="\t", index_col=None, na_filter=True, na_values='n/a')

        plotter = dMRIPlot(
            sliceqc_file=self.inputs.sliceqc_file,
            mask_file=self.inputs.sliceqc_mask,
            confounds=dataframe)
        fig = plotter.plot()
        fig.savefig(self._results['out_file'], bbox_inches='tight')

        # Write a json file of the carpetplot data
        carpetplot_json = os.path.join(runtime.cwd, 'carpetplot.json')
        with open(carpetplot_json, "w") as carpet_file:
            json.dump(
                {'carpetplot': np.nan_to_num(plotter.qc_data['slice_scores']).tolist()},
                carpet_file)
        self._results['carpetplot_json'] = carpetplot_json
        return runtime
