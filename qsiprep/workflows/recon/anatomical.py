#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
qsiprep base reconstruction workflows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_qsirecon_wf
.. autofunction:: init_single_subject_wf

"""
from pathlib import Path
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
import nipype.interfaces.io as nio
from ...engine import Workflow
import logging
from ...interfaces.anatomical import QsiprepAnatomicalIngress
from ...interfaces.mrtrix import GenerateMasked5tt
from .interchange import anatomical_input_fields, freesurfer_output_names

LOGGER = logging.getLogger('nipype.workflow')

# Required freesurfer files for mrtrix's HSV 5tt generation
HSV_REQUIREMENTS = [
    "mri/aparc+aseg.mgz",
    "mri/brainmask.mgz",
    "mri/transforms/talairach.xfm",
    "surf/lh.white",
    "surf/lh.pial",
    "surf/rh.white",
    "surf/rh.pial"
]

QSIPREP_ANAT_REQUIREMENTS = [
    "anat/sub-ABCD_desc-brain_mask.nii.gz",
    "anat/sub-ABCD_desc-preproc_T1w.nii.gz"
]

QSIPREP_NORMALIZED_ANAT_REQUIREMENTS = [
    "anat/{subject_id}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5",
    "anat/{subject_id}_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5"
]

def init_recon_anatomical_wf(subject_id, recon_input_dir, extras_to_make,
                             freesurfer_dir="",
                             name='recon_anatomical_wf'):
    """
    This grabs anatomical outputs from qsiprep and calculates optional
    additional outputs like a dwi-resolution

    Parameters

        subject_id : str
            List of subject labels
        name : str
            Name of workflow
        recon_input_dir : str
            Root directory of the output from qsiprep
        extras_to_make : list
            list of optional derivatives that will be shared across images.
            For example ['mrtrix_5tt_fast', 'mrtrix_5tt_hsv'].
        freesurfer_dir : pathlike
            Path where the freesurfer outputs for `subject_id` go
    """

    workflow = Workflow(name=name)
    outputnode = pe.Node(
        niu.IdentityInterface(fields=anatomical_input_fields),
        name="outputnode")

    anat_ingress = pe.Node(
        QsiprepAnatomicalIngress(subject_id=subject_id,
                                 recon_input_dir=recon_input_dir),
        name='anat_ingress')
    
    # Are FreeSurfer Outputs available?
    freesurfer_path = Path(freesurfer_dir)
    subject_freesurfer_path = freesurfer_path / subject_id
    if freesurfer_dir and freesurfer_path.exists() and \
            subject_freesurfer_path.exists():
        missing_files = check_hsv_inputs(subject_freesurfer_path)

        # Check for specific files needed for HSVs. 
        if missing_files:
            if "mrtrix_5tt_hsv" in extras_to_make:
                raise Exception(" ".join(missing_files) + 
                                "are missing: unable to make a HSV.")
            LOGGER.warn(" ".join(missing_files) + "are missing from freesurfer.")
        
        # Connect the freesurfer outputs to the outputnode
        fs_source = pe.Node(nio.FreeSurferSource())
        workflow.connect([
            (fs_source, outputnode, 
                [(name,name) for name in freesurfer_output_names])
            ])


    workflow.connect([
        (
            anat_ingress,
            outputnode,
            [
                ('t1_aparc', 't1_aparc'),
                ('t1_seg', 't1_seg'),
                ('t1_aseg', 't1_aseg'),
                ('t1_brain_mask', 't1_brain_mask'),
                ('t1_preproc', 't1_preproc'),
                ('t1_csf_probseg', 't1_csf_probseg'),
                ('t1_gm_probseg', 't1_gm_probseg'),
                ('t1_wm_probseg', 't1_wm_probseg'),
                ('left_inflated_surf', 'left_inflated_surf'),
                ('left_midthickness_surf', 'left_midthickness_surf'),
                ('left_pial_surf', 'left_pial_surf'),
                ('left_smoothwm_surf', 'left_smoothwm_surf'),
                ('right_inflated_surf', 'right_inflated_surf'),
                ('right_midthickness_surf', 'right_midthickness_surf'),
                ('right_pial_surf', 'right_pial_surf'),
                ('right_smoothwm_surf', 'right_smoothwm_surf'),
                ('orig_to_t1_mode_forward_transform',
                 'orig_to_t1_mode_forward_transform'),
                ('t1_2_fsnative_forward_transform',
                 't1_2_fsnative_forward_transform'),
                ('t1_2_mni_reverse_transform', 't1_2_mni_reverse_transform'),
                ('t1_2_mni_forward_transform', 't1_2_mni_forward_transform'),
                ('template_brain_mask', 'template_brain_mask'),
                ('template_preproc', 'template_preproc'),
                ('template_seg', 'template_seg'),
                ('template_csf_probseg', 'template_csf_probseg'),
                ('template_gm_probseg', 'template_gm_probseg'),
                ('template_wm_probseg', 'template_wm_probseg')
            ])
    ])

    # Prepare extra outputs
    if 'mrtrix_5tt' in extras_to_make:
        create_5tt = pe.Node(GenerateMasked5tt(algorithm='fsl'), name='create_5tt')
        workflow.connect([
            (anat_ingress, create_5tt, [('t1_brain_mask', 'mask'),
                                        ('t1_preproc', 'in_file')]),
            (create_5tt, outputnode, [('out_file', 'mrtrix_5tt')])
        ])

    return workflow


def check_hsv_inputs(subj_fs_path):
    """Determine if a FreeSurfer directory has the required files for HSV."""
    missing = []
    for requirement in HSV_REQUIREMENTS:
        if not (subj_fs_path / requirement).exists():
            missing.append(requirement)
    return missing


def check_qsiprep_anatomical_outputs(subj_qsiprep_path):
    """Determines whether an aligned T1w exists in a qsiprep derivatives directory.
    
    It is possible that:
      - ``--dwi-only`` was used, in which case there is NO T1w available
      - ``--skip-t1-based-spatial-normalization``, there is a T1w but no transform to a template
      - Normal mode, there is a T1w and a transform to template space.
    """
    subject_id = subj_qsiprep_path.stem

    pass