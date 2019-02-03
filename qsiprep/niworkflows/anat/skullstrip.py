# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from __future__ import absolute_import, division, print_function, unicode_literals
from nipype.interfaces import ants, afni, fsl, utility as niu
from nipype.pipeline import engine as pe


def afni_wf(name='AFNISkullStripWorkflow', unifize=False, n4_nthreads=1):
    """
    Skull-stripping workflow

    Originally derived from the `codebase of the
    QAP <https://github.com/preprocessed-connectomes-project/\
quality-assessment-protocol/blob/master/qap/anatomical_preproc.py#L105>`_.
    Now, this workflow includes :abbr:`INU (intensity non-uniformity)` correction
    using the N4 algorithm and (optionally) intensity harmonization using
    ANFI's ``3dUnifize``.


    """

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['in_file']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['bias_corrected', 'out_file', 'out_mask', 'bias_image']), name='outputnode')

    inu_n4 = pe.Node(
        ants.N4BiasFieldCorrection(dimension=3, save_bias=True, num_threads=n4_nthreads,
                                   copy_header=True),
        n_procs=n4_nthreads,
        name='inu_n4')

    sstrip = pe.Node(afni.SkullStrip(outputtype='NIFTI_GZ'), name='skullstrip')
    sstrip_orig_vol = pe.Node(afni.Calc(
        expr='a*step(b)', outputtype='NIFTI_GZ'), name='sstrip_orig_vol')
    binarize = pe.Node(fsl.Threshold(args='-bin', thresh=1.e-3), name='binarize')

    if unifize:
        # Add two unifize steps, pre- and post- skullstripping.
        inu_uni_0 = pe.Node(afni.Unifize(outputtype='NIFTI_GZ'),
                            name='unifize_pre_skullstrip')
        inu_uni_1 = pe.Node(afni.Unifize(gm=True, outputtype='NIFTI_GZ'),
                            name='unifize_post_skullstrip')
        workflow.connect([
            (inu_n4, inu_uni_0, [('output_image', 'in_file')]),
            (inu_uni_0, sstrip, [('out_file', 'in_file')]),
            (inu_uni_0, sstrip_orig_vol, [('out_file', 'in_file_a')]),
            (sstrip_orig_vol, inu_uni_1, [('out_file', 'in_file')]),
            (inu_uni_1, outputnode, [('out_file', 'out_file')]),
            (inu_uni_0, outputnode, [('out_file', 'bias_corrected')]),
        ])
    else:
        workflow.connect([
            (inputnode, sstrip_orig_vol, [('in_file', 'in_file_a')]),
            (inu_n4, sstrip, [('output_image', 'in_file')]),
            (sstrip_orig_vol, outputnode, [('out_file', 'out_file')]),
            (inu_n4, outputnode, [('output_image', 'bias_corrected')]),
        ])

    # Remaining connections
    workflow.connect([
        (sstrip, sstrip_orig_vol, [('out_file', 'in_file_b')]),
        (inputnode, inu_n4, [('in_file', 'input_image')]),
        (sstrip_orig_vol, binarize, [('out_file', 'in_file')]),
        (binarize, outputnode, [('out_file', 'out_mask')]),
        (inu_n4, outputnode, [('bias_image', 'bias_image')]),
    ])
    return workflow
