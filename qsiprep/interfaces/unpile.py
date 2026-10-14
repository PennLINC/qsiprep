# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Interfaces around `unpile <https://unpile.readthedocs.io>`__, the fieldmap-less SDC fit.

``unpile fit`` estimates the susceptibility field by matching an EPI to an undistorted
reference (a SynB0 synthetic b=0 or the subject's T2w) with a model of the readout, and
optionally refines a GRE fieldmap handed to it as a prior. Every image it reads must share
the EPI's grid, and ``--pe-dir`` refers to that file's own voxel axes. The interfaces here
put the inputs on the fit grid, run the binary, and bring its field back onto the native
EPI grid as an ITK displacement field.
"""

import json
import os.path as op

import nibabel as nb
import numpy as np
from nilearn import image
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    CommandLine,
    CommandLineInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.utils.filemanip import fname_presuffix

from .fmap import pe_step_world
from .images import reference_grid_at_spacing

PE_DIRS = ('i', 'i-', 'j', 'j-', 'k', 'k-')
#: The finest grid a T2w reference is fitted on, in mm.
FINEST_GRID_MM = 1.5


def unpile_fit_grid_mm(voxel_mm):
    """Return the isotropic grid a T2w reference is fitted on, or None for the native grid.

    The T2w has smaller voxels than the EPI, and resampling it down to the EPI grid
    discards the detail the cross-contrast metric needs. unpile's resolution sweeps put the
    gain at about two thirds of the EPI voxel; finer grids cost more and add nothing, so
    the grid is floored at :data:`FINEST_GRID_MM`, and an EPI that is already that fine
    keeps its own grid.
    """
    grid = max(2.0 * float(voxel_mm) / 3.0, FINEST_GRID_MM)
    return round(grid, 2) if grid < 0.9 * float(voxel_mm) else None


def pe_dir_on_grid(raw_dwi_file, pe_dir, image_file):
    """Express a sidecar ``PhaseEncodingDirection`` in the voxel axes of another image.

    BIDS gives the direction in the axes of the file as acquired. qsiprep conforms the
    data to LAS+ or LPS+ and unpile reads the direction in the axes of the file it is
    handed, so the world-space step along the acquired axis is taken from the raw header
    and re-expressed on ``image_file``'s grid, whose axes may be permuted or flipped.
    """
    affine = nb.load(image_file).affine
    step = np.linalg.solve(affine[:3, :3], pe_step_world(raw_dwi_file, pe_dir))
    axis = int(np.argmax(np.abs(step)))
    if abs(step[axis]) < 0.99 * np.linalg.norm(step):
        raise ValueError(
            f'The phase-encoding direction of {raw_dwi_file} is not along a voxel axis '
            f'of {image_file}.'
        )
    return 'ijk'[axis] + ('-' if step[axis] < 0 else '')


class _PrepareUnpileInputsInputSpec(BaseInterfaceInputSpec):
    epi = File(exists=True, mandatory=True, desc='the distorted b=0 reference, native grid')
    mask = File(exists=True, mandatory=True, desc='brain mask of ``epi``')
    reference = File(
        exists=True,
        mandatory=True,
        desc='undistorted reference registered into the frame of ``epi`` (any grid)',
    )
    fieldmap = File(exists=True, desc='GRE fieldmap in Hz on the ``epi`` grid, as a prior')
    raw_dwi_file = File(mandatory=True, desc='the DWI as acquired, whose axes pe_dir refers to')
    pe_dir = traits.Enum(*PE_DIRS, mandatory=True, desc='BIDS PhaseEncodingDirection')
    grid_mm = traits.Float(
        desc='isotropic voxel size of the fit grid; unset or coarser than the EPI keeps '
        'the native grid'
    )
    fine_grid = traits.Bool(
        False,
        usedefault=True,
        desc='without grid_mm, fit on the grid unpile_fit_grid_mm derives from the EPI '
        'voxel (about two thirds of it, no finer than 1.5 mm)',
    )


class _PrepareUnpileInputsOutputSpec(TraitedSpec):
    epi = File(exists=True, desc='the EPI on the fit grid')
    mask = File(exists=True, desc='the mask on the fit grid')
    reference = File(exists=True, desc='the reference on the fit grid')
    fieldmap = File(desc='the fieldmap on the fit grid, when one was given')
    pe_dir = traits.Str(desc='PhaseEncodingDirection in the fit grid voxel axes')
    acq_voxel_mm = traits.Float(desc='the acquired voxel size along the phase-encoding axis')
    resampled = traits.Bool(desc='whether the fit grid differs from the native grid')


class PrepareUnpileInputs(SimpleInterface):
    """Put the EPI, its mask, the reference and a fieldmap on the grid unpile fits on.

    Without ``grid_mm`` or ``fine_grid`` (or with a grid no finer than the EPI's
    phase-encoding voxel) only the reference is resampled, onto the EPI grid. With a finer
    grid every input is resampled onto an isotropic grid covering the EPI's field of view,
    which keeps the detail of an anatomical reference that the cross-contrast metric needs;
    ``acq_voxel_mm`` then tells unpile the acquired voxel size, since Hz times readout time
    is a shift in acquired voxels. The phase-encoding direction is translated to the fit
    grid's axes. The grid is decided here, at run time, from the EPI's header.
    """

    input_spec = _PrepareUnpileInputsInputSpec
    output_spec = _PrepareUnpileInputsOutputSpec

    def _run_interface(self, runtime):
        epi_img = nb.load(self.inputs.epi)
        if epi_img.ndim > 3:
            epi_img = image.mean_img(epi_img)
        native = np.asarray(epi_img.header.get_zooms()[:3], dtype=float)
        # The EPI keeps the acquired voxels, reoriented; its voxel along the acquired
        # phase-encoding axis is what Hz x readout time counts in.
        native_pe_dir = pe_dir_on_grid(
            self.inputs.raw_dwi_file, self.inputs.pe_dir, self.inputs.epi
        )
        acq_voxel = float(native['ijk'.index(native_pe_dir[0])])
        grid_mm = self.inputs.grid_mm if isdefined(self.inputs.grid_mm) else None
        if grid_mm is None and self.inputs.fine_grid:
            grid_mm = unpile_fit_grid_mm(acq_voxel)
        resample = grid_mm is not None and grid_mm < 0.999 * native.min()
        if resample:
            target = reference_grid_at_spacing(epi_img, [grid_mm] * 3)
            epi_out = image.resample_to_img(epi_img, target, interpolation='continuous')
        else:
            target = epi_img
            epi_out = epi_img

        def _write(img, name):
            out_file = op.join(runtime.cwd, f'unpile_{name}.nii.gz')
            img.to_filename(out_file)
            return out_file

        self._results['epi'] = _write(epi_out, 'epi')
        self._results['mask'] = _write(
            image.resample_to_img(nb.load(self.inputs.mask), target, interpolation='nearest'),
            'mask',
        )
        self._results['reference'] = _write(
            image.resample_to_img(
                nb.load(self.inputs.reference), target, interpolation='continuous'
            ),
            'reference',
        )
        if isdefined(self.inputs.fieldmap):
            self._results['fieldmap'] = _write(
                image.resample_to_img(
                    nb.load(self.inputs.fieldmap), target, interpolation='continuous'
                ),
                'fieldmap',
            )
        self._results['pe_dir'] = pe_dir_on_grid(
            self.inputs.raw_dwi_file, self.inputs.pe_dir, self._results['epi']
        )
        self._results['acq_voxel_mm'] = acq_voxel
        self._results['resampled'] = bool(resample)
        return runtime


class _UnpileFitInputSpec(CommandLineInputSpec):
    epi = File(exists=True, mandatory=True, argstr='--epi %s', desc='the distorted EPI')
    mask = File(exists=True, mandatory=True, argstr='--mask %s', desc='brain mask, EPI grid')
    reference = File(
        exists=True, argstr='--reference %s', desc='undistorted reference on the EPI grid'
    )
    fieldmap = File(
        exists=True, argstr='--fieldmap %s', desc='GRE fieldmap in Hz on the EPI grid (prior)'
    )
    fieldmap_refine = traits.Str(
        argstr='--fieldmap-refine %s',
        desc='fieldmap terms to refine: pose, shim, delta (comma-separated) or none',
    )
    metric = traits.Enum(
        'ssd',
        'lncc',
        argstr='--metric %s',
        desc='ssd for a same-contrast reference, lncc for a T2w',
    )
    pose = traits.Bool(
        False, usedefault=True, argstr='--pose', desc='fit a rigid pose of the reference too'
    )
    lam_mm = traits.Float(argstr='--lam-mm %g', desc='bending penalty in displacement units')
    levels = traits.Str(argstr='--levels %s', desc='control-grid spacings in mm, coarse to fine')
    iters = traits.Int(argstr='--iters %d', desc='L-BFGS iterations per level')
    pe_dir = traits.Enum(
        *PE_DIRS,
        mandatory=True,
        argstr='--pe-dir %s',
        desc='PhaseEncodingDirection in the voxel axes of the EPI file',
    )
    total_readout_time = traits.Float(
        mandatory=True, argstr='--total-readout-time %g', desc='TotalReadoutTime (s)'
    )
    acq_voxel_mm = traits.Float(
        argstr='--acq-voxel-mm %g',
        desc='acquired phase-encoding voxel size when the inputs were resampled',
    )
    threads = traits.Int(argstr='--threads %d', desc='CPU threads')
    device = traits.Enum(
        'flex', 'wgpu', usedefault=True, argstr='--device %s', desc='compute device'
    )
    truth_field = File(exists=True, argstr='--truth-field %s', desc='true field (Hz) to score')
    truth_xfm = File(
        exists=True, argstr='--truth-xfm %s', desc='ITK transform the reference pose is scored on'
    )
    truth_fieldmap_xfm = File(
        exists=True,
        argstr='--truth-fieldmap-xfm %s',
        desc='ITK transform the refined fieldmap pose is scored on',
    )
    reference_init_xfm = File(
        exists=True,
        argstr='--reference-init-xfm %s',
        desc='ITK transform already applied to the reference to put it on the EPI grid; '
        'the fitted pose is composed with it in pose_composed_xfm',
    )
    out_dir = traits.Str('unpile', usedefault=True, argstr='--out %s', desc='output directory')


class _UnpileFitOutputSpec(TraitedSpec):
    field_hz = File(exists=True, desc='the field in Hz on the EPI grid')
    displacement_mm = File(exists=True, desc='displacement along the phase-encoding axis')
    corrected = File(exists=True, desc='the EPI unwarped and intensity-corrected')
    sdc_warp = File(exists=True, desc='ITK displacement field (LPS mm) on the EPI grid')
    pose_xfm = File(desc='ITK affine of the reference pose, when --pose was given')
    pose_composed_xfm = File(
        desc='the reference pose composed with reference_init_xfm: EPI grid to the original '
        'reference, when reference_init_xfm was given'
    )
    reference_posed = File(
        desc='the reference as the fit placed it on the EPI grid, when --pose was given'
    )
    fieldmap_pose_xfm = File(desc='ITK affine of the refined fieldmap pose, when refined')
    score_json = File(exists=True, desc='the run settings and scores')


class UnpileFit(CommandLine):
    """Run ``unpile fit``: one EPI against a reference, optionally with a GRE prior."""

    input_spec = _UnpileFitInputSpec
    output_spec = _UnpileFitOutputSpec
    _cmd = 'unpile fit'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        out = op.abspath(self.inputs.out_dir)
        outputs['field_hz'] = op.join(out, 'field_hz.nii.gz')
        outputs['displacement_mm'] = op.join(out, 'disp_pe_mm.nii.gz')
        outputs['corrected'] = op.join(out, 'corrected.nii.gz')
        outputs['sdc_warp'] = op.join(out, 'sdc_warp.nii.gz')
        outputs['score_json'] = op.join(out, 'score.json')
        for key, fname in (
            ('pose_xfm', 'pose_xfm.txt'),
            ('pose_composed_xfm', 'pose_composed_xfm.txt'),
            ('reference_posed', 'reference_posed.nii.gz'),
            ('fieldmap_pose_xfm', 'gre_pose_xfm.txt'),
        ):
            candidate = op.join(out, fname)
            if op.exists(candidate):
                outputs[key] = candidate
        return outputs


class _UnpileFieldToWarpInputSpec(BaseInterfaceInputSpec):
    field_hz = File(exists=True, mandatory=True, desc='unpile field (Hz) on the fit grid')
    reference = File(exists=True, mandatory=True, desc='the native EPI grid the warp is built on')
    raw_dwi_file = File(mandatory=True, desc='the DWI as acquired, whose axes pe_dir refers to')
    pe_dir = traits.Enum(*PE_DIRS, mandatory=True, desc='BIDS PhaseEncodingDirection')
    readout_time = traits.Float(mandatory=True, desc='BIDS TotalReadoutTime (s)')


class _UnpileFieldToWarpOutputSpec(TraitedSpec):
    out_warp = File(exists=True, desc='ITK displacement field (LPS mm) on the reference grid')
    out_hz = File(exists=True, desc='the field (Hz) on the reference grid')
    pe_axis = traits.Int(desc="the reference grid's voxel axis the field displaces along")


class UnpileFieldToWarp(SimpleInterface):
    """Bring unpile's field onto the native EPI grid and write it as an ITK displacement field.

    The field is in Hz, so it resamples cleanly from a finer fit grid; the displacement is
    ``field_Hz * TotalReadoutTime`` native voxels along the acquired phase-encoding axis,
    which is the convention of unpile's own ``sdc_warp.nii.gz`` (identical to it on an
    unresampled fit) and of the TOPUP and GRE warps.
    """

    input_spec = _UnpileFieldToWarpInputSpec
    output_spec = _UnpileFieldToWarpOutputSpec

    def _run_interface(self, runtime):
        reference = nb.load(self.inputs.reference)
        if reference.ndim > 3:
            reference = image.mean_img(reference)
        field = nb.load(self.inputs.field_hz)
        same_grid = field.shape[:3] == reference.shape[:3] and np.allclose(
            field.affine, reference.affine, atol=1e-3
        )
        if not same_grid:
            field = image.resample_to_img(field, reference, interpolation='continuous')
        out_hz = fname_presuffix(
            self.inputs.field_hz, suffix='_native', newpath=runtime.cwd, use_ext=True
        )
        nb.Nifti1Image(np.asarray(field.dataobj, dtype=np.float32), reference.affine).to_filename(
            out_hz
        )
        # Imported here: the converter lives with the resampling workflow, which imports
        # the interfaces package.
        from ..workflows.dwi.resampling import _hz_to_warp

        pe_dir = pe_dir_on_grid(self.inputs.raw_dwi_file, self.inputs.pe_dir, out_hz)
        self._results['out_hz'] = out_hz
        self._results['out_warp'] = _hz_to_warp(
            out_hz, self.inputs.readout_time, pe_dir, newpath=runtime.cwd
        )
        self._results['pe_axis'] = 'ijk'.index(pe_dir[0])
        return runtime


def read_unpile_score(score_json):
    """Return the ``score.json`` of a fit without its argument echo."""
    with open(score_json) as fobj:
        score = json.load(fobj)
    score.pop('args', None)
    return score


__all__ = [
    'FINEST_GRID_MM',
    'PrepareUnpileInputs',
    'UnpileFieldToWarp',
    'UnpileFit',
    'pe_dir_on_grid',
    'read_unpile_score',
    'unpile_fit_grid_mm',
]
