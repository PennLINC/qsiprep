#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
.. _fieldmapless :

Fieldmapless
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
from fmriprep.engine import Workflow
from nipype.pipeline import engine as pe
from qsiprep.workflows.dwi.hmc import init_hmc_wf
from qsiprep.workflows.dwi.registration import init_b0_to_anat_registration_wf
from qsiprep.workflows.dwi.merge import init_merge_and_denoise_wf
from nipype.interfaces import afni, utility as niu

def init_no_fieldmap_wf(use_syn,
                        pe_dir,
                        dwi_denoise_window,
                        output_spaces,
                        denoise_before_combining,
                        omp_nthreads=1,
                        name="no_fieldmap_wf"):
    """
    This workflow is for dwis that do not have a reverse PE reference scan. Either
    no SDC is performed or SyN SDC is performed

    If ``use_syn`` It also calculates a new mask for the input dataset that takes into
    account the distortions.

    .. workflow ::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.fieldmap.pepolar import init_pepolar_unwarp_wf
        wf = init_no_fieldmap_wf(pe_dir='j',
                                 dwi_denoise_window=7,
                                 output_spaces=['T1w', 'template']
                                 denoise_before_combining=True,
                                 omp_nthreads=8)


    Inputs

        input_dwis
            DWI series all of one PE direction
        t1w_brain
            Reference skull stripped T1w brain
        t1w_to_mni_affine
            itk affine from ``t1w_brain`` to LPS+ MNI space
        t1w_to_mni_warp
            ANTS-compatible displacement field that registers T1w to LPS+ MNI


    Outputs

        out_reference
            the ``in_reference`` after unwarping
        out_reference_brain
            the ``in_reference`` after unwarping and skullstripping
        out_warp
            the corresponding :abbr:`DFM (displacements field map)` to correct
            ``template_plus``
        dwi_2_t1w_affine
            the affine from the dwi reference to the T1w
        out_mask
            mask of the unwarped input file

    """

    workflow = Workflow(name=name)
    workflow.__desc__ = """\
A deformation field to correct for susceptibility distortions was estimated
based on two b0 templates created from dwi series with opposing phase-encoding
directions, using `3dQwarp` @afni (AFNI {afni_ver}).
""".format(afni_ver=''.join(['%02d' % v for v in afni.Info().version() or []]))

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['input_dwis', 't1w_brain', 't1w_to_mni_affine', 't1w_to_mni_warp']),
        name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['out_reference', 'out_reference_brain', 'out_affine_plus', 'out_warp_plus',
                'out_affine_minus', 'out_warp_minus', 'out_mask']), name='outputnode')

    merge_dwis = init_merge_and_denoise_wf(dwi_denoise_window=dwi_denoise_window,
                                           denoise_before_combining=denoise_before_combining,
                                           name="merge_dwis")

    b0_hmc = init_hmc_wf(name='b0_hmc')

    t1w_coreg = init_b0_to_anat_registration_wf(name="t1w_coreg")

    workflow = Workflow(name=name)

    return workflow
