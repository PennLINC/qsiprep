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
    isdefined,
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


def get_synb0_atlas_mask():
    """Path of the 2.5mm atlas brain mask (``None`` outside the containers)."""
    atlases = os.environ.get('SYNB0_ATLASES')
    if not atlases:
        return None
    mask = os.path.join(atlases, 'mni_icbm152_t1_tal_nlin_asym_09c_mask_2_5.nii.gz')
    if not os.path.exists(mask):
        return None
    return mask


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
    scale_factor = traits.Float(desc='the 110/WM-median factor that was applied')
    clipped_fraction = traits.Float(desc='fraction of nonzero voxels clipped at the 255 ceiling')


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

        scale_factor = 110.0 / wm_median
        scaled = data * scale_factor
        normalized = np.clip(scaled, 0, 255)
        nonzero = scaled > 0
        clipped_fraction = (
            float((scaled[nonzero] > 255).sum() / nonzero.sum()) if nonzero.any() else 0.0
        )

        out_file = fname_presuffix(self.inputs.t1w_file, suffix='_synb0norm', newpath=runtime.cwd)
        nb.Nifti1Image(normalized, img.affine, img.header).to_filename(out_file)
        self._results['out_file'] = out_file
        self._results['scale_factor'] = float(scale_factor)
        self._results['clipped_fraction'] = clipped_fraction
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
    dispersion_file = File(
        'b0_u_atlas_dispersion.nii.gz',
        usedefault=True,
        argstr='--dispersion-out %s',
        desc="across-fold standard deviation (the ensemble's disagreement)",
    )


class _Synb0InferenceOutputSpec(TraitedSpec):
    out_file = File(exists=True)
    dispersion_file = File(exists=True)


class Synb0Inference(CommandLine):
    """Run every fold of the SynB0-DISCO U-Net and average the results.

    The runner executes with the isolated torch environment's interpreter
    (``QSIPREP_TORCH_PYTHON``, falling back to the current one): qsiprep's own
    environment has no torch (see the note in ``pyproject.toml``).
    """

    input_spec = _Synb0InferenceInputSpec
    output_spec = _Synb0InferenceOutputSpec
    _cmd = 'python'  # replaced in __init__; must stay which-able for check_deps

    def __init__(self, **inputs):
        super().__init__(**inputs)
        torch_python = os.environ.get('QSIPREP_TORCH_PYTHON', sys.executable)
        self._cmd = f'{torch_python} {load_data("synb0_runner.py")}'

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        outputs['dispersion_file'] = os.path.abspath(self.inputs.dispersion_file)
        return outputs


def _mutual_information(x, y, bins=32):
    """Histogram-based mutual information of two intensity samples."""
    x = np.clip(x, *np.percentile(x, [1, 99]))
    y = np.clip(y, *np.percentile(y, [1, 99]))
    joint, _, _ = np.histogram2d(x, y, bins=bins)
    pxy = joint / joint.sum()
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    nonzero = pxy > 0
    return float((pxy[nonzero] * np.log(pxy[nonzero] / (px @ py)[nonzero])).sum())


def _qq_correlation(x, y):
    """Correlation of the two samples' quantile functions (1 = same shape)."""
    quantiles = np.linspace(1, 99, 99)
    return float(np.corrcoef(np.percentile(x, quantiles), np.percentile(y, quantiles))[0, 1])


class _Synb0QCInputSpec(BaseInterfaceInputSpec):
    t1_atlas = File(exists=True, mandatory=True, desc='normalized T1w on the U-Net grid')
    b0_atlas = File(exists=True, mandatory=True, desc='acquired b=0 on the U-Net grid')
    synthetic_atlas = File(exists=True, mandatory=True, desc='U-Net output (fold mean)')
    dispersion_atlas = File(exists=True, mandatory=True, desc='across-fold std image')
    atlas_mask = File(exists=True, mandatory=True, desc='atlas brain mask on the U-Net grid')
    atlas_image = File(exists=True, mandatory=True, desc='the 2.5mm atlas T1w itself')
    coreg_metric = traits.Float(desc='Mattes metric from the b=0-to-anat coregistration')
    normalization_scale = traits.Float(desc='the 110/WM-median normalization factor')
    clipped_fraction = traits.Float(desc='fraction of T1w voxels clipped at 255')


class _Synb0QCOutputSpec(TraitedSpec):
    qc_file = File(exists=True)


class Synb0QC(SimpleInterface):
    """Scalar QC for the SynB0 generation, one row per run.

    Every column maps to an observed failure mode: a failed affine or
    normalization lowers ``normalized_t1w_vs_atlas_correlation``; mutual
    misalignment of the two U-Net input channels lowers
    ``unet_inputs_mutual_information``; off-distribution b=0 contrast (which
    TOPUP would convert into a bogus field) lowers
    ``synthetic_vs_acquired_qq_correlation`` and raises
    ``unet_fold_dispersion_cv`` (the ensemble's disagreement).
    """

    input_spec = _Synb0QCInputSpec
    output_spec = _Synb0QCOutputSpec

    def _run_interface(self, runtime):
        import pandas as pd

        mask = np.asanyarray(nb.load(self.inputs.atlas_mask).dataobj) > 0
        t1 = nb.load(self.inputs.t1_atlas).get_fdata(dtype=np.float32)[mask]
        b0 = nb.load(self.inputs.b0_atlas).get_fdata(dtype=np.float32)[mask]
        synthetic = nb.load(self.inputs.synthetic_atlas).get_fdata(dtype=np.float32)[mask]
        dispersion = nb.load(self.inputs.dispersion_atlas).get_fdata(dtype=np.float32)[mask]
        atlas = nb.load(self.inputs.atlas_image).get_fdata(dtype=np.float32)[mask]

        row = {
            'normalized_t1w_vs_atlas_correlation': float(np.corrcoef(t1, atlas)[0, 1]),
            'unet_inputs_mutual_information': _mutual_information(t1, b0),
            'synthetic_vs_acquired_qq_correlation': _qq_correlation(synthetic, b0),
            'unet_fold_dispersion_cv': float(np.mean(dispersion) / np.median(synthetic)),
            'wm_normalization_scale_factor': self.inputs.normalization_scale
            if isdefined(self.inputs.normalization_scale)
            else np.nan,
            't1w_clipped_voxel_fraction': self.inputs.clipped_fraction
            if isdefined(self.inputs.clipped_fraction)
            else np.nan,
            'b0_to_anat_coreg_mattes': self.inputs.coreg_metric
            if isdefined(self.inputs.coreg_metric)
            else np.nan,
        }
        qc_file = os.path.join(runtime.cwd, 'synb0_qc.tsv')
        pd.DataFrame([row]).to_csv(qc_file, sep='\t', index=False)
        self._results['qc_file'] = qc_file
        return runtime


class _Synb0FieldQCInputSpec(BaseInterfaceInputSpec):
    fieldmap = File(exists=True, mandatory=True, desc='TOPUP fieldmap in Hz')
    mask = File(exists=True, mandatory=True, desc='brain mask on the fieldmap grid')
    datain = File(exists=True, mandatory=True, desc='TOPUP datain (PE direction and readout)')


class _Synb0FieldQCOutputSpec(TraitedSpec):
    qc_file = File(exists=True)


class Synb0FieldQC(SimpleInterface):
    """Scalar QC for the SynB0-driven TOPUP field.

    ``fieldmap_halo_ratio_outside_over_inside`` detects the pathology where
    acquired-vs-synthetic intensity mismatch at the head boundary becomes a
    fictitious boundary field (a smooth halo wrapped around the head); the
    displacement column states how far the correction moves tissue.
    """

    input_spec = _Synb0FieldQCInputSpec
    output_spec = _Synb0FieldQCOutputSpec

    def _run_interface(self, runtime):
        import pandas as pd
        from scipy.ndimage import binary_dilation

        mask_img = nb.load(self.inputs.mask)
        mask = np.asanyarray(mask_img.dataobj) > 0
        field = nb.load(self.inputs.fieldmap).get_fdata(dtype=np.float32)
        field = field - np.median(field[mask])

        first_row = np.loadtxt(self.inputs.datain, ndmin=2)[0]
        pe_axis = int(np.argmax(np.abs(first_row[:3])))
        readout = float(first_row[3])
        pe_voxel_mm = float(mask_img.header.get_zooms()[pe_axis])

        ring = binary_dilation(mask, iterations=8) & ~mask
        p95_inside = float(np.percentile(np.abs(field[mask]), 95))
        p95_ring = float(np.percentile(np.abs(field[ring]), 95)) if ring.any() else np.nan

        row = {
            'fieldmap_p95_abs_hz_in_brain': p95_inside,
            # FSL convention: displacement in voxels = Hz x TotalReadoutTime
            'fieldmap_p95_displacement_mm_in_brain': p95_inside * readout * pe_voxel_mm,
            'fieldmap_halo_ratio_outside_over_inside': p95_ring / p95_inside
            if p95_inside > 0
            else np.nan,
        }
        qc_file = os.path.join(runtime.cwd, 'synb0_field_qc.tsv')
        pd.DataFrame([row]).to_csv(qc_file, sep='\t', index=False)
        self._results['qc_file'] = qc_file
        return runtime
