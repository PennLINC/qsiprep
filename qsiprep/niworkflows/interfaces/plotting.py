# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Visualization tools

"""
import numpy as np
import pandas as pd

from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    File, BaseInterfaceInputSpec, TraitedSpec, SimpleInterface, traits
)
from ..viz.plots import fMRIPlot


class FMRISummaryInputSpec(BaseInterfaceInputSpec):
    in_func = File(exists=True, mandatory=True, desc='')
    in_mask = File(exists=True, mandatory=True, desc='')
    in_segm = File(exists=True, mandatory=True, desc='')
    in_spikes_bg = File(exists=True, mandatory=True, desc='')
    fd = File(exists=True, mandatory=True, desc='')
    fd_thres = traits.Float(0.2, usedefault=True, desc='')
    dvars = File(exists=True, mandatory=True, desc='')
    outliers = File(exists=True, mandatory=True, desc='')
    tr = traits.Either(None, traits.Float, usedefault=True,
                       desc='the TR')


class FMRISummaryOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='written file path')


class FMRISummary(SimpleInterface):
    """
    Copy the x-form matrices from `hdr_file` to `out_file`.
    """
    input_spec = FMRISummaryInputSpec
    output_spec = FMRISummaryOutputSpec

    def _run_interface(self, runtime):
        self._results['out_file'] = fname_presuffix(
            self.inputs.in_func,
            suffix='_fmriplot.svg',
            use_ext=False,
            newpath=runtime.cwd)

        dataframe = pd.DataFrame({
            'outliers': np.loadtxt(
                self.inputs.outliers, usecols=[0]).tolist(),
            # Pick non-standardize dvars (col 1)
            # First timepoint is NaN (difference)
            'DVARS': [np.nan] + np.loadtxt(
                self.inputs.dvars, skiprows=1, usecols=[1]).tolist(),
            # First timepoint is zero (reference volume)
            'FD': [0.0] + np.loadtxt(
                self.inputs.fd, skiprows=1, usecols=[0]).tolist(),
        })

        fig = fMRIPlot(
            self.inputs.in_func,
            mask_file=self.inputs.in_mask,
            seg_file=self.inputs.in_segm,
            spikes_files=[self.inputs.in_spikes_bg],
            tr=self.inputs.tr,
            data=dataframe[['outliers', 'DVARS', 'FD']],
            units={'outliers': '%', 'FD': 'mm'},
            vlines={'FD': [self.inputs.fd_thres]},
        ).plot()
        fig.savefig(self._results['out_file'], bbox_inches='tight')
        return runtime
