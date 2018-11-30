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
from qsiprep.workflows.dwi.hmc import init_b0_hmc_wf
from qsiprep.workflows.dwi.registration import init_b0_to_anat_registration_wf
from qsiprep.workflows.dwi.merge import init_merge_and_denoise_wf
from nipype.interfaces import afni, utility as niu
from qsiprep.interfaces.images import SplitDWIs
from qsiprep.interfaces.gradients import WarpAndRecombineDWIs
from nipype.interfaces.base import Undefined


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

    If ``use_syn`` it also calculates a new mask for the input dataset that takes into
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
        t1_brain
            Reference skull stripped T1w brain
        t1_2_mni_forward_transform
            itk transforms from ``t1_brain`` to LPS+ MNI space


    Outputs

        out_reference
            the ``in_reference`` after unwarping
        out_reference_brain
            the ``in_reference`` after unwarping and skullstripping
        out_warps
            the corresponding :abbr:`DFM (displacements field map)` to correct
            ``template_plus``
        dwi_2_ref_affines
            the affine from the dwi reference to the dwi reference
        out_mask
            mask of the unwarped input file

    """

    workflow = Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['input_dwis', 't1_brain', 't1_2_mni_forward_transform', 't1w_to_mni_warp']),
        name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['bval', 'dwi_t1w', 'bvec_t1w', 'b0_ref', 'b0_ref_t1', 'sdc_unwarps']),
        name='outputnode')

    workflow.connect([
    ])

    # If use_syn, get the warp. Otherwise make an IdentityNode
    if use_syn:
        pass
    else:
        sdc_warp = pe.Node(niu.IdentityInterface(fields=['out_warp']))
        sdc_warp.inputs.out_warp = Undefined


    return workflow
