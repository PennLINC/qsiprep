# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Image tools interfaces."""

import os
import shutil
from pathlib import Path

import nibabel as nb
import nilearn.image as nim
import numpy as np
from dipy.segment.threshold import otsu
from nipype import logging
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    InputMultiObject,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.utils.filemanip import fname_presuffix
from scipy import ndimage
from scipy.spatial import distance

LOGGER = logging.getLogger('nipype.interface')


class _DiceOverlapInputSpec(BaseInterfaceInputSpec):
    anatomical_mask = File(exists=True, mandatory=True, desc='Mask from a T1w image')
    dwi_mask = File(exists=True, mandatory=True, desc='Mask from a DWI image')


class _DiceOverlapOutputSpec(TraitedSpec):
    dice_score = traits.Float()


class DiceOverlap(SimpleInterface):
    input_spec = _DiceOverlapInputSpec
    output_spec = _DiceOverlapOutputSpec

    def _run_interface(self, runtime):
        t1_img = nb.load(self.inputs.anatomical_mask)
        dwi_img = nb.load(self.inputs.dwi_mask)

        if not t1_img.shape == dwi_img.shape:
            raise Exception('Cannot compare masks with different shapes')

        self._results['dice_score'] = distance.dice(
            t1_img.get_fdata().flatten(), dwi_img.get_fdata().flatten()
        )
        return runtime


class _VoxelSizeChooserInputSpec(BaseInterfaceInputSpec):
    voxel_size = traits.Float()
    input_images = InputMultiObject(File(exists=True))
    anisotropic_strategy = traits.Enum('min', 'max', 'mean', usedefault=True)


class _VoxelSizeChooserOutputSpec(TraitedSpec):
    voxel_size = traits.Float()


class VoxelSizeChooser(SimpleInterface):
    input_spec = _VoxelSizeChooserInputSpec
    output_spec = _VoxelSizeChooserOutputSpec

    def _run_interface(self, runtime):
        if not isdefined(self.inputs.input_images) and not isdefined(self.inputs.voxel_size):
            raise Exception(
                'VoxelSizeChooser: either voxel_size or input_images must be set '
                '(a native resolution, e.g. res-nativemin/res-nativemax, requires '
                'input_images).'
            )

        # An explicit size always wins; the strategies only apply to measured images.
        if isdefined(self.inputs.voxel_size):
            self._results['voxel_size'] = self.inputs.voxel_size
            return runtime

        zooms = []
        for image in self.inputs.input_images:
            zooms.extend(nb.load(image).header.get_zooms()[:3])

        if self.inputs.anisotropic_strategy == 'min':
            voxel_size = min(zooms)
        elif self.inputs.anisotropic_strategy == 'max':
            voxel_size = max(zooms)
        else:
            voxel_size = np.round(np.mean(zooms), 2)

        self._results['voxel_size'] = float(voxel_size)
        return runtime


class _FakeSegmentationInputSpec(BaseInterfaceInputSpec):
    mask_file = File(exists=True, mandatory=True)


class _FakeSegmentationOutputSpec(TraitedSpec):
    dseg_file = File(exists=True)


class FakeSegmentation(SimpleInterface):
    input_spec = _FakeSegmentationInputSpec
    output_spec = _FakeSegmentationOutputSpec

    def _run_interface(self, runtime):
        img = nb.load(self.inputs.mask_file)
        orig_mask = img.get_fdata() > 0
        eroded1 = ndimage.binary_erosion(orig_mask, iterations=3)
        eroded2 = ndimage.binary_erosion(eroded1, iterations=3)
        final = orig_mask.astype(int) + eroded1 + eroded2
        out_img = nb.Nifti1Image(final, img.affine, header=img.header)
        out_fname = fname_presuffix(self.inputs.mask_file, suffix='_dseg', newpath=runtime.cwd)
        out_img.to_filename(out_fname)
        self._results['dseg_file'] = out_fname

        return runtime


class _DesaturateSkullInputSpec(BaseInterfaceInputSpec):
    skulled_t2w_image = File(exists=True, mandatory=True, desc='Skull-on T2w image')
    brain_mask_image = File(
        exists=True, mandatory=True, desc='Binary brain mask in the same grid as skulled_t2w_image'
    )
    brain_to_skull_ratio = traits.CFloat(
        8.0, usedefault=True, desc='Ratio of signal in the brain to signal in the skull'
    )


class _DesaturateSkullOutputSpec(TraitedSpec):
    desaturated_t2w = File(exists=True)
    head_scaling_factor = traits.Float(0.0)


class DesaturateSkull(SimpleInterface):
    input_spec = _DesaturateSkullInputSpec
    output_spec = _DesaturateSkullOutputSpec

    def _run_interface(self, runtime):
        out_file = fname_presuffix(
            self.inputs.skulled_t2w_image,
            newpath=runtime.cwd,
            suffix='_desaturated.nii',
            use_ext=False,
        )
        skulled_img = nim.load_img(self.inputs.skulled_t2w_image)
        brainmask_img = nim.load_img(self.inputs.brain_mask_image)
        brain_median, nonbrain_head_median = calculate_nonbrain_saturation(
            skulled_img, brainmask_img
        )

        actual_brain_to_skull_ratio = brain_median / nonbrain_head_median
        LOGGER.info('found brain to skull ratio: %.3f', actual_brain_to_skull_ratio)
        desat_data = skulled_img.get_fdata(dtype='float32').copy()
        adjustment = 1.0
        if actual_brain_to_skull_ratio < self.inputs.brain_to_skull_ratio:
            # We need to downweight the non-brain voxels
            adjustment = actual_brain_to_skull_ratio / self.inputs.brain_to_skull_ratio
            LOGGER.info(f'Desaturating outside-brain signal by {adjustment:.5f}')
            nonbrain_mask = brainmask_img.get_fdata() < 1
            # Apply the adjustment
            desat_data[nonbrain_mask] = desat_data[nonbrain_mask] * adjustment

        desat_img = nim.new_img_like(skulled_img, desat_data, copy_header=True)
        desat_img.header.set_data_dtype('float32')
        desat_img.to_filename(out_file)
        self._results['desaturated_t2w'] = out_file
        self._results['head_scaling_factor'] = adjustment
        return runtime


def calculate_nonbrain_saturation(head_img, brain_mask_img):
    # Calculate the
    head_data = head_img.get_fdata()
    brain_mask = brain_mask_img.get_fdata() > 0

    def clip_values(values):
        _, top_percent = np.percentile(values, np.array([0, 99.75]), axis=None)
        return np.clip(values, 0, top_percent)

    nonbrain_voxels = head_data[np.logical_not(brain_mask)]
    winsorized_nonbrain_voxels = clip_values(nonbrain_voxels)
    threshold = otsu(winsorized_nonbrain_voxels) * 0.5

    nbmask = np.zeros_like(head_img.get_fdata())
    nbmask[head_data > threshold] = 2
    nbmask[brain_mask] = 0

    in_brain_median = np.median(head_data[brain_mask])
    non_brain_head_median = np.median(head_data[nbmask > 0])

    return in_brain_median, non_brain_head_median


class _GetTemplateInputSpec(BaseInterfaceInputSpec):
    template_name = traits.Str(desc='TemplateFlow template name', mandatory=True)
    cohort = traits.Str(desc='Cohort label, if the template has one')
    resolution = traits.Str('1', usedefault=True, desc='TemplateFlow resolution entity')
    anatomical_contrast = traits.Enum('T1w', 'T2w', 'none')


class _GetTemplateOutputSpec(BaseInterfaceInputSpec):
    template_file = File(exists=True)
    mask_file = File(exists=True)


def _templateflow_lock():
    """Return a cross-process lock over the shared TemplateFlow cache.

    TemplateFlow streams a download straight into its final cache path
    (``client._s3_get`` opens it ``'wb'`` and writes as the response arrives, with
    no temp file and no rename), so a file is observable half-written for the whole
    download. ``--output-spaces`` builds one ``GetTemplate`` per standard space in
    ``anat_preproc_wf`` and another in ``anat_derivatives_wf`` with no dependency
    between them, so MultiProc runs them together and one node copies what the
    other is still fetching -- which AFNI then refuses to load ("data bytes input =
    -1 ... is it complete?"). Serializing fetch-and-copy is what makes the copy
    safe; a warm cache skips the download, so the lock costs nothing after the
    first fetch.

    Returns ``None`` when the cache root cannot be created: a cache that cannot
    be written to cannot be mid-download either, so the caller fetches unlocked
    rather than failing the node.
    """
    from filelock import FileLock
    from templateflow.conf import TF_LAYOUT

    root = Path(os.getenv('TEMPLATEFLOW_HOME', TF_LAYOUT.root))
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return FileLock(root / '.qsiprep-templateflow.lock', timeout=600)


def _acquire_templateflow_lock(template_name):
    """Take the TemplateFlow cache lock, or return ``None`` to fetch unlocked.

    Acquiring is where a read-only cache fails (``mkdir(exist_ok=True)`` succeeds
    on an existing directory, creating the lock file does not), so the fallback
    has to happen here rather than when the lock object is built. A cache that
    is mounted read-only, as on a cluster with a shared pre-fetched copy, holds
    nothing half-written.
    """
    from filelock import Timeout

    lock = _templateflow_lock()
    if lock is None:
        return None
    try:
        lock.acquire()
    except Timeout:
        # flock is released when its holder dies, so a wait this long means a
        # live process is still downloading, not a stale lock. Racing it is
        # better than failing the run outright.
        LOGGER.warning(
            'Timed out waiting for the TemplateFlow cache lock; fetching %s without it.',
            template_name,
        )
        return None
    except OSError as exc:
        LOGGER.warning(
            'Cannot create the TemplateFlow cache lock (%s); fetching %s without it.',
            exc,
            template_name,
        )
        return None
    return lock


def template_queries(template_name, cohort, resolution, anatomical_contrast):
    """Build the TemplateFlow queries for a template's head image and brain mask.

    Templates are not filed uniformly: UNCInfant carries no ``res-`` entity at
    all and ships its brain mask as ``label-brain`` rather than ``desc-brain``.
    Each query follows what the template actually provides, read from the local
    TemplateFlow skeleton (no download), so a template the parser accepted can
    be fetched.
    """
    from templateflow.conf import TF_LAYOUT

    base = {'extension': '.nii.gz'}
    if cohort is not None:
        base['cohort'] = cohort
    if resolution is not None and TF_LAYOUT.get_resolutions(template=template_name):
        base['resolution'] = resolution

    image = {**base, 'desc': None, 'suffix': anatomical_contrast}
    mask = {**base, 'desc': 'brain', 'suffix': 'mask'}
    if not TF_LAYOUT.get(template=template_name, **mask):
        labelled = {**base, 'label': 'brain', 'suffix': 'mask'}
        if TF_LAYOUT.get(template=template_name, **labelled):
            mask = labelled
    return image, mask


def _one_file(found, template_name, query):
    """Return the single path ``templateflow.api.get`` found, or say why not."""
    if isinstance(found, list):
        if not found:
            raise FileNotFoundError(
                f'TemplateFlow has no file for template {template_name} matching {query}.'
            )
        raise ValueError(
            f'TemplateFlow has {len(found)} files for template {template_name} matching '
            f'{query}; the query must select exactly one.'
        )
    return Path(found)


class GetTemplate(SimpleInterface):
    input_spec = _GetTemplateInputSpec
    output_spec = _GetTemplateOutputSpec

    def _run_interface(self, runtime):
        from templateflow.api import get as get_template

        anatomical_contrast = self.inputs.anatomical_contrast
        if anatomical_contrast == 'none':
            LOGGER.info('Using T1w modality template for ACPC alignment')
            anatomical_contrast = 'T1w'

        template_name = self.inputs.template_name
        cohort = self.inputs.cohort if isdefined(self.inputs.cohort) else None
        resolution = self.inputs.resolution if isdefined(self.inputs.resolution) else None
        image_query, mask_query = template_queries(
            template_name, cohort, resolution, anatomical_contrast
        )

        # Copy inside the lock too: the copy is the read that a concurrent
        # download corrupts.
        lock = _acquire_templateflow_lock(template_name)
        try:
            template_path = _one_file(
                get_template(template_name, **image_query), template_name, image_query
            )
            mask_path = _one_file(
                get_template(template_name, **mask_query), template_name, mask_query
            )
            local_template = Path(runtime.cwd) / template_path.name
            local_mask = Path(runtime.cwd) / mask_path.name
            shutil.copy(template_path, local_template)
            shutil.copy(mask_path, local_mask)
        finally:
            if lock is not None:
                lock.release()

        self._results['template_file'] = str(local_template)
        self._results['mask_file'] = str(local_mask)

        return runtime
