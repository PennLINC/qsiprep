#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Image tools interfaces
~~~~~~~~~~~~~~~~~~~~~~


"""

import os.path as op
import numpy as np
from nipype import logging
from glob import glob
import nibabel as nb
from scipy.spatial import distance
from scipy import ndimage
from nipype.interfaces.base import (traits, TraitedSpec, BaseInterfaceInputSpec,
                                    SimpleInterface, File)
from nipype.utils.filemanip import fname_presuffix

LOGGER = logging.getLogger('nipype.interface')


class QsiprepAnatomicalIngressInputSpec(BaseInterfaceInputSpec):
    recon_input_dir = traits.Directory(exists=True, mandatory=True)
    subject_id = traits.Str()
    subjects_dir = File(exists=True)


class QsiprepAnatomicalIngressOutputSpec(TraitedSpec):
    # sub-1_desc-aparcaseg_dseg.nii.gz
    t1_aparc = File()
    # sub-1_dseg.nii.gz
    t1_seg = File()
    # sub-1_desc-aseg_dseg.nii.gz
    t1_aseg = File()
    # sub-1_desc-brain_mask.nii.gz
    t1_brain_mask = File()
    # sub-1_desc-preproc_T1w.nii.gz
    t1_preproc = File()
    # sub-1_label-CSF_probseg.nii.gz
    t1_csf_probseg = File()
    # sub-1_label-GM_probseg.nii.gz
    t1_gm_probseg = File()
    # sub-1_label-WM_probseg.nii.gz
    t1_wm_probseg = File()

    # sub-1_hemi-L_inflated.surf.gii
    left_inflated_surf = File()
    # sub-1_hemi-L_midthickness.surf.gii
    left_midthickness_surf = File()
    # sub-1_hemi-L_pial.surf.gii
    left_pial_surf = File()
    # sub-1_hemi-L_smoothwm.surf.gii
    left_smoothwm_surf = File()
    # sub-1_hemi-R_inflated.surf.gii
    right_inflated_surf = File()
    # sub-1_hemi-R_midthickness.surf.gii
    right_midthickness_surf = File()
    # sub-1_hemi-R_pial.surf.gii
    right_pial_surf = File()
    # sub-1_hemi-R_smoothwm.surf.gii
    right_smoothwm_surf = File()

    # sub-1_from-orig_to-T1w_mode-image_xfm.txt
    orig_to_t1_mode_forward_transform = File()
    # sub-1_from-T1w_to-fsnative_mode-image_xfm.txt
    t1_2_fsnative_forward_transform = File()
    # sub-1_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5
    t1_2_mni_reverse_transform = File()
    # sub-1_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5
    t1_2_mni_forward_transform = File()
    # sub-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz
    template_brain_mask = File()
    # sub-1_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz
    template_preproc = File()
    # sub-1_space-MNI152NLin2009cAsym_dseg.nii.gz
    template_seg = File()
    # sub-1_space-MNI152NLin2009cAsym_label-CSF_probseg.nii.gz
    template_csf_probseg = File()
    # sub-1_space-MNI152NLin2009cAsym_label-GM_probseg.nii.gz
    template_gm_probseg = File()
    # sub-1_space-MNI152NLin2009cAsym_label-WM_probseg.nii.gz
    template_wm_probseg = File()


class QsiprepAnatomicalIngress(SimpleInterface):
    """
    """
    input_spec = QsiprepAnatomicalIngressInputSpec
    output_spec = QsiprepAnatomicalIngressOutputSpec

    def _run_interface(self, runtime):
        # The path to the output from the qsiprep run
        sub = self.inputs.subject_id
        qp_root = op.join(self.inputs.recon_input_dir, 'sub-' + sub)
        anat_root = op.join(qp_root, 'anat')
        # space-T1w files
        self._get_if_exists(
            't1_aparc',
            '%s/sub-%s*desc-aparcaseg_dseg.nii.*' % (anat_root, sub),
            excludes=['space-MNI'])
        self._get_if_exists(
            't1_seg',
            '%s/sub-%s_*dseg.nii*' % (anat_root, sub),
            excludes=['space-MNI', 'aseg'])
        self._get_if_exists(
            't1_aseg',
            '%s/sub-%s_*aseg_dseg.nii*' % (anat_root, sub),
            excludes=['space-MNI', 'aparc'])
        self._get_if_exists(
            't1_brain_mask',
            '%s/sub-%s*_desc-brain_mask.nii*' % (anat_root, sub),
            excludes=['space-MNI'])
        self._get_if_exists(
            't1_preproc',
            "%s/sub-%s_desc-preproc_T1w.nii*" % (anat_root, sub),
            excludes=['space-MNI'])
        if 't1_preproc' not in self._results:
            LOGGER.warning("Unable to find a preprocessed T1w in %s", qp_root)
        self._get_if_exists(
            't1_csf_probseg',
            "%s/sub-%s*_label-CSF_probseg.nii*" % (anat_root, sub),
            excludes=["space-MNI"])
        self._get_if_exists(
            't1_gm_probseg',
            "%s/sub-%s*_label-GM_probseg.nii*" % (anat_root, sub),
            excludes=["space-MNI"])
        self._get_if_exists(
            't1_wm_probseg',
            "%s/sub-%s*_label-WM_probseg.nii*" % (anat_root, sub),
            excludes=["space-MNI"])

        # space-template files
        self._get_if_exists(
            'template_brain_mask',
            '%s/sub-%s_*space-MNI152NLin2009cAsym_desc-brain_mask.nii*' % (anat_root, sub))
        self._get_if_exists(
            'template_preproc',
            "%s/sub-%s_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii*" % (anat_root, sub))
        self._get_if_exists(
            't1_seg',
            '%s/sub-%s*_space-MNI152NLin2009cAsym_dseg*' % (anat_root, sub))
        self._get_if_exists(
            'template_csf_probseg',
            "%s/sub-%s*_space-MNI152NLin2009cAsym_label-CSF_probseg.nii*" % (anat_root, sub))
        self._get_if_exists(
            'template_gm_probseg',
            "%s/sub-%s*_space-MNI152NLin2009cAsym_label-GM_probseg.nii*" % (anat_root, sub))
        self._get_if_exists(
            'template_wm_probseg',
            "%s/sub-%s*_space-MNI152NLin2009cAsym_label-WM_probseg.nii*" % (anat_root, sub))

        self._get_if_exists(
            'left_inflated_surf',
            "%s/sub-%s*_hemi-L_inflated.surf.gii" % (anat_root, sub))
        self._get_if_exists(
            'left_midthickness_surf',
            "%s/sub-%s*_hemi-L_midthickness.surf.gii" % (anat_root, sub))
        self._get_if_exists(
            'left_pial_surf',
            "%s/sub-%s*_hemi-L_pial.surf.gii" % (anat_root, sub))
        self._get_if_exists(
            'left_smoothwm_surf',
            "%s/sub-%s*_hemi-L_smoothwm.surf.gii" % (anat_root, sub))
        self._get_if_exists(
            'right_inflated_surf',
            "%s/sub-%s*_hemi-R_inflated.surf.gii" % (anat_root, sub))
        self._get_if_exists(
            'right_midthickness_surf',
            "%s/sub-%s*_hemi-R_midthickness.surf.gii" % (anat_root, sub))
        self._get_if_exists(
            'right_pial_surf',
            "%s/sub-%s*_hemi-R_pial.surf.gii" % (anat_root, sub))
        self._get_if_exists(
            'right_smoothwm_surf',
            "%s/sub-%s*_hemi-R_smoothwm.surf.gii" % (anat_root, sub))

        self._get_if_exists(
            'orig_to_t1_mode_forward_transform',
            "%s/sub-%s*_from-orig_to-T1w_mode-image_xfm.txt" % (anat_root, sub))
        self._get_if_exists(
            't1_2_fsnative_forward_transform',
            "%s/sub-%s*_from-T1w_to-fsnative_mode-image_xfm.txt" % (anat_root, sub))
        self._get_if_exists(
            't1_2_mni_reverse_transform',
            "%s/sub-%s*_from-MNI152NLin2009cAsym_to-T1w*_xfm.h5" % (anat_root, sub))
        self._get_if_exists(
            't1_2_mni_forward_transform',
            "%s/sub-%s*_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5" % (anat_root, sub))
        return runtime

    def _get_if_exists(self, name, pattern, excludes=None):
        files = glob(pattern)

        if excludes is not None:
            files = [fname for fname in files if not
                     any([exclude in op.split(fname)[1] for exclude in excludes])]

        if len(files) == 1:
            self._results[name] = files[0]


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
            raise Exception("Cannot compare masks with different shapes")

        self._results['dice_score'] = distance.dice(t1_img.get_fdata().flatten(),
                                                    dwi_img.get_fdata().flatten())
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
        final = orig_mask.astype(np.int) + eroded1 + eroded2
        out_img = nb.Nifti1Image(final, img.affine, header=img.header)
        out_fname = fname_presuffix(self.inputs.mask_file, suffix="_dseg",
                                    newpath=runtime.cwd)
        out_img.to_filename(out_fname)
        self._results['dseg_file'] = out_fname

        return runtime
