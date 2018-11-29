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
        fields=['input_dwis', 't1_brain', 't1_2_mni_forward_transform', 't1w_to_mni_warp']),
        name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['bval', 'dwi_t1w', 'bvec_t1w', 'dwi_mni', 'bval_mni']), name='outputnode')

    merge_dwis = init_merge_and_denoise_wf(dwi_denoise_window=dwi_denoise_window,
                                           denoise_before_combining=denoise_before_combining,
                                           name="merge_dwis")

    split_dwis = pe.Node(SplitDWIs(), name="split_dwis")

    b0_hmc = init_b0_hmc_wf()

    t1w_coreg = init_b0_to_anat_registration_wf()

    workflow.connect([
        (inputnode, merge_dwis, [('input_dwis', 'inputnode.dwi_files')]),
        (merge_dwis, split_dwis, [('outputnode.merged_image', 'dwi_file'),
                                  ('outputnode.merged_bval', 'bval_file'),
                                  ('outputnode.merged_bvec', 'bvec_file')]),
        (merge_dwis, outputnode, [('outputnode.merged_bval', 'bval')]),
        (split_dwis, b0_hmc, [('b0_images', 'inputnode.b0_images')]),
        (b0_hmc, t1w_coreg, [('outputnode.final_template', 'inputnode.b0_image')]),
        (inputnode, t1w_coreg, [('t1_brain', 'inputnode.anat_image')])
    ])

    # If use_syn, get the warp. Otherwise make an IdentityNode
    if use_syn:
        pass
    else:
        sdc_warp = pe.Node(niu.IdentityInterface(fields=['out_warp']))
        sdc_warp.inputs.out_warp = Undefined

    if "T1w" in output_spaces:
        warp_and_recombine_t1w = pe.Node(WarpAndRecombineDWIs(), name="warp_and_recombine_t1w")
        workflow.connect([
            (b0_hmc, warp_and_recombine_t1w, [('outputnode.forward_transforms',
                                               'b0_hmc_affines')]),
            (split_dwis, warp_and_recombine_t1w, [('dwi_files', 'dwi_files'),
                                                  ('bval_files', 'bval_files'),
                                                  ('bvec_files', 'bvec_files'),
                                                  ('b0_indices', 'original_b0_indices')]),
            (t1w_coreg, warp_and_recombine_t1w, [('outputnode.b0_to_anat_transform',
                                                  'dwi_ref_to_t1w_affine')]),
            (warp_and_recombine_t1w, outputnode, [('out_dwi', 'dwi_t1w'),
                                                  ('out_bval', 'bval_t1w'),
                                                  ('out_bvec', 'bvec_t1w')])
        ])

    if "template" in output_spaces:
        warp_and_recombine_mni = pe.Node(WarpAndRecombineDWIs(), name="warp_and_recombine_mni")
        workflow.connect([
            (inputnode, warp_and_recombine_mni, [('t1_2_mni_forward_transform',
                                                  't1_2_mni_forward_transform')]),
        ])
    return workflow
