#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Image tools interfaces
~~~~~~~~~~~~~~~~~~~~~~


"""

from pkg_resources import resource_filename as pkgr
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
from dipy.segment.threshold import otsu
import nilearn.image as nim

LOGGER = logging.getLogger('nipype.interface')
KNOWN_TEMPLATES = ['MNI152NLin2009cAsym', 'infant']


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
    # sub-1_from-orig_to-T1w_mode-image_xfm.txt
    orig_to_t1_mode_forward_transform = File()
    # sub-1_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5
    t1_2_mni_reverse_transform = File()
    # sub-1_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5
    t1_2_mni_forward_transform = File()


class QsiprepAnatomicalIngress(SimpleInterface):
    """ Get only the useful files from a QSIPrep anatomical output.

    Many of the preprocessed outputs aren't useful for reconstruction
    (mainly anything that has been mapped forward into template space).
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
        self._get_if_exists(
            'orig_to_t1_mode_forward_transform',
            "%s/sub-%s*_from-orig_to-T1w_mode-image_xfm.txt" % (anat_root, sub))
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
        final = orig_mask.astype(int) + eroded1 + eroded2
        out_img = nb.Nifti1Image(final, img.affine, header=img.header)
        out_fname = fname_presuffix(self.inputs.mask_file, suffix="_dseg",
                                    newpath=runtime.cwd)
        out_img.to_filename(out_fname)
        self._results['dseg_file'] = out_fname

        return runtime


"""

The spherical harmonic coefficients are stored as follows. First, since the
signal attenuation profile is real, it has conjugate symmetry, i.e. Y(l,-m) =
Y(l,m)* (where * denotes the complex conjugate). Second, the diffusion profile
should be antipodally symmetric (i.e. S(x) = S(-x)), implying that all odd l
components should be zero. Therefore, only the even elements are computed. Note
that the spherical harmonics equations used here differ slightly from those
conventionally used, in that the (-1)^m factor has been omitted. This should be
taken into account in all subsequent calculations. Each volume in the output
image corresponds to a different spherical harmonic component.

Each volume will
correspond to the following:

volume 0: l = 0, m = 0 ;
volume 1: l = 2, m = -2 (imaginary part of m=2 SH) ;
volume 2: l = 2, m = -1 (imaginary part of m=1 SH)
volume 3: l = 2, m = 0 ;
volume 4: l = 2, m = 1 (real part of m=1 SH) ;
volume 5: l = 2, m = 2 (real part of m=2 SH) ; etc…


lmax = 2

vol	l	m
0	0	0
1	2	-2
2	2	-1
3	2	0
4	2	1
5	2	2

"""


lmax_lut = {
    6: 2,
    15: 4,
    28: 6,
    45: 8
}


def get_l_m(lmax):
    ell = []
    m = []
    for _ell in range(0, lmax + 1, 2):
        for _m in range(-_ell, _ell+1):
            ell.append(_ell)
            m.append(_m)

    return np.array(ell), np.array(m)


def calculate_steinhardt(sh_l, sh_m, data, q_num):
    l_mask = sh_l == q_num
    images = data[..., l_mask]
    scalar = 4 * np.pi / (2 * q_num + 1)
    s_param = scalar * np.sum(images ** 2, 3)
    return np.sqrt(s_param)


class CalculateSOPInputSpec(BaseInterfaceInputSpec):
    sh_nifti = traits.File(mandatory=True, exists=True)
    order = traits.Enum(2,4,6,8, default=6, usedefault=True)


class CalculateSOPOutputSpec(TraitedSpec):
    q2_file = traits.File()
    q4_file = traits.File()
    q6_file = traits.File()
    q8_file = traits.File()


class CalculateSOP(SimpleInterface):
    input_spec = CalculateSOPInputSpec
    output_spec = CalculateSOPOutputSpec

    def _run_interface(self, runtime):

        # load the input nifti image
        img = nb.load(self.inputs.sh_nifti)

        # determine what the lmax was based on the number of volumes
        num_vols = img.shape[3]
        if not num_vols in lmax_lut:
            raise ValueError("Not an SH image")
        lmax = lmax_lut[num_vols]

        # Do we have enough SH coeffs to calculate all the SOPs?
        if self.inputs.order > lmax:
            raise Exception("Not enough SH coefficients (found {}) "
                            "to calculate SOP order {}".format(
                                num_vols, self.inputs.order))
        sh_l, sh_m = get_l_m(lmax)
        sh_data = img.get_fdata()

        # to get a specific order
        def calculate_order(order):
            out_fname = fname_presuffix(
                self.inputs.sh_nifti, suffix="q-%d_SOP.nii.gz" % order,
                use_ext=False,
                newpath=runtime.cwd)
            order_data = calculate_steinhardt(sh_l, sh_m, sh_data, order)
            # Save with the new name in the sandbox
            nb.Nifti1Image(order_data, img.affine).to_filename(out_fname)
            self._results["q%d_file" % order] = out_fname

        # calculate!
        for order in range(2, self.inputs.order + 2, 2):
            calculate_order(order)

        return runtime


class _DesaturateSkullInputSpec(BaseInterfaceInputSpec):
    skulled_t2w_image = File(
        exists=True,
        mandatory=True,
        desc="Skull-on T2w image")
    brain_mask_image = File(
        exists=True,
        mandatory=True,
        desc='Binary brain mask in the same grid as skulled_t2w_image')
    brain_to_skull_ratio = traits.CFloat(
        8.0,
        usedefault=True,
        desc="Ratio of signal in the brain to signal in the skull")


class _DesaturateSkullOutputSpec(TraitedSpec):
    desaturated_t2w = File(exists=True)
    head_scaling_factor = traits.Float(0.)

class DesaturateSkull(SimpleInterface):
    input_spec = _DesaturateSkullInputSpec
    output_spec = _DesaturateSkullOutputSpec

    def _run_interface(self, runtime):

        out_file = fname_presuffix(
            self.inputs.skulled_t2w_image,
            newpath=runtime.cwd,
            suffix='_desaturated.nii',
            use_ext=False)
        skulled_img = nim.load_img(self.inputs.skulled_t2w_image)
        brainmask_img = nim.load_img(self.inputs.brain_mask_image)
        brain_median, nonbrain_head_median = calculate_nonbrain_saturation(
            skulled_img, brainmask_img)

        actual_brain_to_skull_ratio = brain_median / nonbrain_head_median
        LOGGER.info("found brain to skull ratio: %.3f", actual_brain_to_skull_ratio)
        desat_data = skulled_img.get_fdata(dtype='float32').copy()
        adjustment = 1.
        if actual_brain_to_skull_ratio < self.inputs.brain_to_skull_ratio:
            # We need to downweight the non-brain voxels
            adjustment = actual_brain_to_skull_ratio / self.inputs.brain_to_skull_ratio
            LOGGER.info("Desaturating outside-brain signal by %.5f" % adjustment)
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
        _, top_percent = np.percentile(
            values, np.array([0, 99.75]), axis=None)
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

class _CustomApplyMaskInputSpec(BaseInterfaceInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        desc="Image to be masked")
    mask_file = File(
        exists=True,
        mandatory=True,
        desc='Mask to be applied')


class _CustomApplyMaskOutputSpec(TraitedSpec):
    out_file = File(exist=True, desc="Image with mask applied")


class CustomApplyMask(SimpleInterface):
    input_spec = _CustomApplyMaskInputSpec
    output_spec = _CustomApplyMaskOutputSpec

    def _run_interface(self, runtime):
        #define masked output name
        out_file = fname_presuffix(
            self.inputs.in_file,
            newpath=runtime.cwd,
            suffix='_masked.nii.gz',
            use_ext=False)

        #load in input and mask
        input_img = nb.load(self.inputs.in_file)
        input_data = input_img.get_fdata()
        mask_data = nb.load(self.inputs.mask_file).get_fdata()
        #elementwise multiplication to apply mask
        out_data = input_data*mask_data
        #save out masked image and pass on file name
        nb.Nifti1Image(out_data, input_img.affine, header=input_img.header).to_filename(out_file)
        self._results['out_file'] = out_file
        return runtime

class _GetTemplateInputSpec(BaseInterfaceInputSpec):
    template_name = traits.Str(
        'MNI152NLin2009cAsym',
        usedefault=True,
        mandatory=True)
    t1_file = File(exists=True)
    t2_file = File(exists=True)
    mask_file = File(exists=True)
    infant_mode = traits.Bool(False, usedefault=True)
    anatomical_contrast = traits.Enum("T1w", "T2w", "none")


class _GetTemplateOutputSpec(BaseInterfaceInputSpec):
    template_name = traits.Str()
    template_file = File(exists=True)
    template_brain_file = File(exists=True)
    template_mask_file = File(exists=True)


class GetTemplate(SimpleInterface):
    input_spec = _GetTemplateInputSpec
    output_spec = _GetTemplateOutputSpec

    def _run_interface(self, runtime):
        self._results['template_name'] = self.inputs.template_name
        contrast_name = self.inputs.anatomical_contrast.lower()
        if contrast_name == "none":
            LOGGER.info("Using T1w modality template for ACPC alignment")
            contrast_name = "t1w"

        # Cover the cases where the template images are actually in the
        # qsiprep package. This is for common use cases (MNI2009cAsym and Infant)
        # and legacy
        if self.inputs.template_name in KNOWN_TEMPLATES or self.inputs.infant_mode:
            if not self.inputs.infant_mode:
                ref_img = pkgr('qsiprep',
                                'data/mni_1mm_%s_lps.nii.gz' % contrast_name)
                ref_img_brain = pkgr('qsiprep',
                                        'data/mni_1mm_%s_lps_brain.nii.gz' % contrast_name)
                ref_img_mask = pkgr('qsiprep',
                                        'data/mni_1mm_t1w_lps_brainmask.nii.gz')
            else:
                ref_img = pkgr('qsiprep',
                                'data/mni_1mm_%s_lps_infant.nii.gz' % contrast_name)
                ref_img_brain = pkgr('qsiprep',
                                        'data/mni_1mm_%s_lps_brain_infant.nii.gz' % contrast_name)
                ref_img_mask = pkgr('qsiprep',
                                        'data/mni_1mm_t1w_lps_brainmask_infant.nii.gz')
            self._results['template_file'] = ref_img
            self._results['template_brain_file'] = ref_img_brain
            self._results['template_mask_file'] = ref_img_mask
        else:
            raise NotImplementedError("Arbitrary templates not available yet")

        return runtime

