# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Interfaces for SynB0-DISCO synthetic b=0 generation."""

import os
import sys

import nibabel as nb
import numpy as np
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    CommandLine,
    CommandLineInputSpec,
    Directory,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)
from nipype.utils.filemanip import fname_presuffix

from ..data import load as load_data


def get_synb0_dir():
    """The SynB0-DISCO distribution directory (weights, atlases, model code).

    The qsiprep containers export ``SYNB0_ATLASES``; the distribution root is
    its parent directory. Returns ``None`` when unset (outside the containers).
    """
    atlases = os.environ.get('SYNB0_ATLASES')
    if not atlases:
        return None
    return os.path.dirname(atlases.rstrip('/'))


def get_synb0_atlas():
    """Path of the 2.5mm atlas defining the grid the U-Net was trained on.

    Returns ``None`` when the SynB0 distribution is not available.
    """
    atlases = os.environ.get('SYNB0_ATLASES')
    if not atlases:
        return None
    atlas = os.path.join(atlases, 'mni_icbm152_t1_tal_nlin_asym_09c_2_5.nii.gz')
    if not os.path.exists(atlas):
        return None
    return atlas


class _NormalizeForSynb0InputSpec(BaseInterfaceInputSpec):
    t1w_file = File(
        exists=True,
        mandatory=True,
        desc='bias-corrected T1w head image',
    )
    dseg_file = File(
        exists=True,
        mandatory=True,
        desc='tissue segmentation on the same grid as t1w_file',
    )
    wm_label = traits.Int(
        3,
        usedefault=True,
        desc='value marking white matter in dseg_file (3 in the ACT convention)',
    )


class _NormalizeForSynb0OutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='intensity-normalized T1w')


class NormalizeForSynb0(SimpleInterface):
    """Scale a T1w image into the intensity convention the SynB0 U-Net expects.

    The U-Net was trained on FreeSurfer-normalized T1s (``mri_nu_correct.mni``
    + ``mri_normalize``: white matter at 110, uint8 ceiling of 255) and its
    inference scales inputs by a fixed 0-150 range. This reproduces that
    convention without FreeSurfer: scale so the white-matter median lands on
    110 and clip to [0, 255] like the uint8 conversion would.
    """

    input_spec = _NormalizeForSynb0InputSpec
    output_spec = _NormalizeForSynb0OutputSpec

    def _run_interface(self, runtime):
        img = nb.load(self.inputs.t1w_file)
        data = img.get_fdata(dtype=np.float32)
        dseg = np.asanyarray(nb.load(self.inputs.dseg_file).dataobj)

        wm_values = data[dseg == self.inputs.wm_label]
        if wm_values.size == 0:
            raise ValueError(
                f'No voxels have label {self.inputs.wm_label} in {self.inputs.dseg_file}'
            )
        wm_median = np.median(wm_values)
        if wm_median <= 0:
            raise ValueError(f'Nonpositive white matter median ({wm_median}) in T1w')

        normalized = np.clip(data * (110.0 / wm_median), 0, 255)

        out_file = fname_presuffix(self.inputs.t1w_file, suffix='_synb0norm', newpath=runtime.cwd)
        nb.Nifti1Image(normalized, img.affine, img.header).to_filename(out_file)
        self._results['out_file'] = out_file
        return runtime


class _Synb0InferenceInputSpec(CommandLineInputSpec):
    t1_file = File(
        exists=True,
        mandatory=True,
        argstr='--t1 %s',
        desc='normalized T1w resampled to the 2.5mm atlas grid',
    )
    b0_file = File(
        exists=True,
        mandatory=True,
        argstr='--b0 %s',
        desc='distorted b=0 resampled to the 2.5mm atlas grid',
    )
    synb0_dir = Directory(
        exists=True,
        mandatory=True,
        argstr='--synb0-dir %s',
        desc='SynB0 distribution (dual_channel_unet/, model.py, util.py)',
    )
    out_file = File(
        'b0_u_atlas.nii.gz',
        usedefault=True,
        argstr='--out %s',
        desc='synthetic distortion-free b=0 (mean over the model folds)',
    )


class _Synb0InferenceOutputSpec(TraitedSpec):
    out_file = File(exists=True)


class Synb0Inference(CommandLine):
    """Run every fold of the SynB0-DISCO U-Net and average the results.

    The runner executes with the isolated torch environment's interpreter
    (``QSIPREP_TORCH_PYTHON``, falling back to the current one): qsiprep's own
    environment has no torch (see the note in ``pyproject.toml``).
    """

    input_spec = _Synb0InferenceInputSpec
    output_spec = _Synb0InferenceOutputSpec
    _cmd = 'synb0_runner'  # display only; see cmd

    @property
    def cmd(self):
        torch_python = os.environ.get('QSIPREP_TORCH_PYTHON', sys.executable)
        return f'{torch_python} {load_data("synb0_runner.py")}'

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs
