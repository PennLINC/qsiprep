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

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from ...engine import Workflow
import logging
from ...interfaces.anatomical import QsiprepAnatomicalIngress
from ...interfaces.mrtrix import GenerateMasked5tt
from .interchange import anatomical_input_fields

LOGGER = logging.getLogger('nipype.workflow')


def init_recon_anatomical_wf(subject_id, recon_input_dir, extras_to_make,
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
            For example ['mrtrix_5tt'].
    """

    workflow = Workflow(name=name)
    outputnode = pe.Node(
        niu.IdentityInterface(fields=anatomical_input_fields),
        name="outputnode")

    anat_ingress = pe.Node(
        QsiprepAnatomicalIngress(subject_id=subject_id,
                                 recon_input_dir=recon_input_dir),
        name='anat_ingress')

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
