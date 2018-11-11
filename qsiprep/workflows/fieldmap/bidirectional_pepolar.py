#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
.. _sdc_pepolar :

Phase Encoding POLARity (*PEPOLAR*) techniques
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""

import pkg_resources as pkgr

from nipype.pipeline import engine as pe
from nipype.interfaces import afni, ants, utility as niu
from niworkflows.interfaces import CopyHeader

from fmriprep.engine import Workflow
from fmriprep.workflows.bold.util import init_enhance_and_skullstrip_bold_wf
from ..dwi.unbiased_rigid_alignment import get_alignment_workflow


def init_bidirectional_b0_unwarping_wf(template_plus_pe, omp_nthreads=1,
                                       name="bidirectional_pepolar_unwarping_wf"):
    """
    This workflow takes in a set of b0 files with opposite phase encoding
    direction and calculates displacement fields
    (in other words, an ANTs-compatible warp file). This is intended to be run
    in the case where there are two dwi series in the same session with reverse
    phase encoding directions.

    The warp field correcting for the distortions is estimated using AFNI's
    3dQwarp, with displacement estimation limited to the target file phase
    encoding direction.

    It also calculates a new mask for the input dataset that takes into
    account the distortions.

    .. workflow ::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.fieldmap.pepolar import init_pepolar_unwarp_wf
        wf = init_pepolar_unwarp_wf(
            bold_meta={'PhaseEncodingDirection': 'j'},
            epi_fmaps=[('/dataset/sub-01/fmap/sub-01_epi.nii.gz', 'j-')],
            omp_nthreads=8)


    Inputs

        template_plus
            b0 template in one PE
        template_minus
            b0_template in the other PE

    Outputs

        out_reference
            the ``in_reference`` after unwarping
        out_reference_brain
            the ``in_reference`` after unwarping and skullstripping
        out_warp_plus
            the corresponding :abbr:`DFM (displacements field map)` to correct
            ``template_plus``
        out_warp_minus
            the corresponding :abbr:`DFM (displacements field map)` to correct
            ``template_minus``
        out_mask
            mask of the unwarped input file

    """
    args = '-noXdis -noYdis -noZdis'
    rm_arg = {'i': '-noXdis',
              'j': '-noYdis',
              'k': '-noZdis'}[template_plus_pe[0]]
    args = args.replace(rm_arg, '')

    workflow = Workflow(name=name)
    workflow.__desc__ = """\
A deformation field to correct for susceptibility distortions was estimated
based on two b0 templates created from dwi series with opposing phase-encoding
directions, using `3dQwarp` @afni (AFNI {afni_ver}).
""".format(afni_ver=''.join(['%02d' % v for v in afni.Info().version() or []]))

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['template_plus', 'template_minus']),
        name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['out_reference', 'out_reference_brain', 'out_affine_plus', 'out_warp_plus',
                'out_affine_minus', 'out_warp_minus', 'out_mask']), name='outputnode')

    inputs_to_list = pe.Node(niu.Merge(2), name='inputs_to_list')
    align_reverse_pe_wf = get_alignment_workflow(align_to='iterative', transform='Rigid')
    get_midpoint_transforms = pe.Node(niu.Split(splits=[1, 1], squeeze=True),
                                      name="get_midpoint_transforms")
    plus_to_midpoint = pe.Node(ants.ApplyTransforms(float=True,
                                                    interpolation='LanczosWindowedSinc',
                                                    dimension=3),
                               name='plus_to_midpoint')
    minus_to_midpoint = pe.Node(ants.ApplyTransforms(float=True,
                                                     interpolation='LanczosWindowedSinc',
                                                     dimension=3),
                                name='minus_to_midpoint')

    qwarp = pe.Node(afni.QwarpPlusMinus(pblur=[0.05, 0.05],
                                        blur=[-1, -1],
                                        noweight=True,
                                        minpatch=9,
                                        nopadWARP=True,
                                        environ={'OMP_NUM_THREADS': '%d' % omp_nthreads},
                                        args=args),
                    name='qwarp', n_procs=omp_nthreads)

    to_ants_plus = pe.Node(niu.Function(function=_fix_hdr), name='to_ants_plus',
                           mem_gb=0.01)
    to_ants_minus = pe.Node(niu.Function(function=_fix_hdr), name='to_ants_minus',
                            mem_gb=0.01)

    cphdr_plus_warp = pe.Node(CopyHeader(), name='cphdr_plus_warp', mem_gb=0.01)
    cphdr_minus_warp = pe.Node(CopyHeader(), name='cphdr_minus_warp', mem_gb=0.01)

    unwarp_plus_reference = pe.Node(ants.ApplyTransforms(dimension=3,
                                                         float=True,
                                                         interpolation='LanczosWindowedSinc'),
                                    name='unwarp_plus_reference')
    unwarp_minus_reference = pe.Node(ants.ApplyTransforms(dimension=3,
                                                          float=True,
                                                          interpolation='LanczosWindowedSinc'),
                                     name='unwarp_minus_reference')
    unwarped_to_list = pe.Node(niu.Merge(2), name="unwarped_to_list")
    merge_unwarped = pe.Node(ants.AverageImages(dimension=3, normalize=True),
                             name="merge_unwarped")
    enhance_and_skullstrip_bold_wf = init_enhance_and_skullstrip_bold_wf(omp_nthreads=omp_nthreads)

    workflow.connect([
        (inputnode, inputs_to_list, [('template_plus', 'in1'),
                                     ('template_minus', 'in2')]),
        (inputs_to_list, align_reverse_pe_wf, [('out', 'input_node.input_images')]),
        (align_reverse_pe_wf, get_midpoint_transforms, [('output_node.forward_transforms',
                                                         'inlist')]),
        (get_midpoint_transforms, outputnode, [('out1', 'out_affine_plus'),
                                               ('out2', 'out_affine_minus')]),
        (inputnode, plus_to_midpoint, [('template_plus', 'input_image')]),
        (inputnode, minus_to_midpoint, [('template_minus', 'input_image')]),
        (get_midpoint_transforms, plus_to_midpoint, [('out1', 'transforms')]),
        (align_reverse_pe_wf, plus_to_midpoint, [('output_node.final_template',
                                                  'reference_image')]),
        (get_midpoint_transforms, minus_to_midpoint, [('out2', 'transforms')]),
        (align_reverse_pe_wf, minus_to_midpoint, [('output_node.final_template',
                                                  'reference_image')]),
        (plus_to_midpoint, qwarp, [('output_image', 'in_file')]),
        (minus_to_midpoint, qwarp, [('output_image', 'base_file')]),
        (align_reverse_pe_wf, cphdr_plus_warp, [('output_node.final_template', 'hdr_file')]),
        (align_reverse_pe_wf, cphdr_minus_warp, [('output_node.final_template', 'hdr_file')]),
        (qwarp, cphdr_plus_warp, [('source_warp', 'in_file')]),
        (qwarp, cphdr_minus_warp, [('base_warp', 'in_file')]),
        (cphdr_plus_warp, to_ants_plus, [('out_file', 'in_file')]),
        (cphdr_minus_warp, to_ants_minus, [('out_file', 'in_file')]),

        (to_ants_minus, unwarp_minus_reference, [('out', 'transforms')]),
        (minus_to_midpoint, unwarp_minus_reference, [('output_image', 'reference_image'),
                                                     ('output_image', 'input_image')]),
        (to_ants_minus, outputnode, [('out', 'out_warp_minus')]),

        (to_ants_plus, unwarp_plus_reference, [('out', 'transforms')]),
        (plus_to_midpoint, unwarp_plus_reference, [('output_image', 'reference_image'),
                                                   ('output_image', 'input_image')]),
        (to_ants_minus, outputnode, [('out', 'out_warp_plus')]),

        (unwarp_plus_reference, unwarped_to_list, [('output_image', 'in1')]),
        (unwarp_minus_reference, unwarped_to_list, [('output_image', 'in2')]),
        (unwarped_to_list, merge_unwarped, [('out', 'images')]),

        (merge_unwarped, enhance_and_skullstrip_bold_wf, [('output_average_image',
                                                           'inputnode.in_file')]),
        (enhance_and_skullstrip_bold_wf, outputnode, [
            ('outputnode.mask_file', 'out_mask'),
            ('outputnode.skull_stripped_file', 'out_reference_brain')]),
    ])

    return workflow


def _fix_hdr(in_file, newpath=None):
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix

    nii = nb.load(in_file)
    hdr = nii.header.copy()
    hdr.set_data_dtype('<f4')
    hdr.set_intent('vector', (), '')
    out_file = fname_presuffix(in_file, "_warpfield", newpath=newpath)
    nb.Nifti1Image(nii.get_data().astype('<f4'), nii.affine, hdr).to_filename(
        out_file)
    return out_file
