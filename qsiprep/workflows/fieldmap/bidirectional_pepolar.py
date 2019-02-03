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
from ...niworkflows.interfaces import CopyHeader

from ...engine import Workflow
from ..dwi.hmc import init_b0_hmc_wf
from ..dwi.util import init_dwi_reference_wf


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
        fields=['template_plus', 'template_minus', 't1w_brain']),
        name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['out_reference', 'out_reference_brain', 'out_affine_plus', 'out_warp_plus',
                'out_affine_minus', 'out_warp_minus', 'out_mask']), name='outputnode')
    # Create high-contrast ref images
    plus_ref_wf = init_dwi_reference_wf(name='plus_ref_wf')
    minus_ref_wf = init_dwi_reference_wf(name='minus_ref_wf')

    # Align the two reference images to the midpoint
    inputs_to_list = pe.Node(niu.Merge(2), name='inputs_to_list')
    align_reverse_pe_wf = init_b0_hmc_wf(align_to='iterative', transform='Rigid')
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

    final_ref = init_dwi_reference_wf(name="final_ref")

    workflow.connect([
        (inputnode, plus_ref_wf, [('template_plus', 'inputnode.b0_template')]),
        (plus_ref_wf, inputs_to_list, [('outputnode.ref_image', 'in1')]),
        (inputnode, minus_ref_wf, [('template_minus', 'inputnode.b0_template')]),
        (minus_ref_wf, inputs_to_list, [('outputnode.ref_image', 'in2')]),
        (inputs_to_list, align_reverse_pe_wf, [('out', 'inputnode.b0_images')]),
        (align_reverse_pe_wf, get_midpoint_transforms, [('outputnode.forward_transforms',
                                                         'inlist')]),
        (get_midpoint_transforms, outputnode, [('out1', 'out_affine_plus'),
                                               ('out2', 'out_affine_minus')]),
        (plus_ref_wf, plus_to_midpoint, [('outputnode.ref_image', 'input_image')]),
        (minus_ref_wf, minus_to_midpoint, [('outputnode.ref_image', 'input_image')]),
        (get_midpoint_transforms, plus_to_midpoint, [('out1', 'transforms')]),
        (align_reverse_pe_wf, plus_to_midpoint, [('outputnode.final_template',
                                                  'reference_image')]),
        (get_midpoint_transforms, minus_to_midpoint, [('out2', 'transforms')]),
        (align_reverse_pe_wf, minus_to_midpoint, [('outputnode.final_template',
                                                  'reference_image')]),
        (plus_to_midpoint, qwarp, [('output_image', 'in_file')]),
        (minus_to_midpoint, qwarp, [('output_image', 'base_file')]),
        (align_reverse_pe_wf, cphdr_plus_warp, [('outputnode.final_template', 'hdr_file')]),
        (align_reverse_pe_wf, cphdr_minus_warp, [('outputnode.final_template', 'hdr_file')]),
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
        (to_ants_plus, outputnode, [('out', 'out_warp_plus')]),

        (unwarp_plus_reference, unwarped_to_list, [('output_image', 'in1')]),
        (unwarp_minus_reference, unwarped_to_list, [('output_image', 'in2')]),
        (unwarped_to_list, merge_unwarped, [('out', 'images')]),

        (merge_unwarped, final_ref, [('output_average_image', 'inputnode.b0_template')]),
        (final_ref, outputnode, [('outputnode.ref_image', 'out_reference'),
                                 ('outputnode.ref_image_brain', 'out_reference_brain'),
                                 ('outputnode.dwi_mask', 'out_mask')])
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
