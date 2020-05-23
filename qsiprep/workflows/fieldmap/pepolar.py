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
from ...niworkflows.interfaces.registration import ANTSApplyTransformsRPT

from ...engine import Workflow
from ...interfaces import StructuralReference
from ...interfaces.fmap import B0RPEFieldmap
from ...interfaces.nilearn import EnhanceB0


def init_pepolar_unwarp_wf(dwi_meta, epi_fmaps, omp_nthreads=1,
                           name="pepolar_unwarp_wf"):
    """
    This workflow takes in a set of EPI files with opposite phase encoding
    direction than the target file and calculates a displacements field
    (in other words, an ANTs-compatible warp file).

    This procedure works if there is only one '_epi' file is present
    (as long as it has the opposite phase encoding direction to the target
    file). The target file will be used to estimate the field distortion.
    However, if there is another '_epi' file present with a matching
    phase encoding direction to the target it will be used instead.

    Currently, different phase encoding dimension in the target file and the
    '_epi' file(s) (for example 'i' and 'j') is not supported.

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
            dwi_meta={'PhaseEncodingDirection': 'j'},
            epi_fmaps=[('/dataset/sub-01/fmap/sub-01_epi.nii.gz', 'j-')],
            omp_nthreads=8)


    Inputs

        in_reference
            the reference image
        in_reference_brain
            the reference image skullstripped
        in_mask
            a brain mask corresponding to ``in_reference``

    Outputs

        out_reference
            the ``in_reference`` after unwarping
        out_warp
            the corresponding :abbr:`DFM (displacements field map)` compatible with
            ANTs

    """
    dwi_file_pe = dwi_meta["PhaseEncodingDirection"]

    args = '-noXdis -noYdis -noZdis'
    rm_arg = {'i': '-noXdis',
              'j': '-noYdis',
              'k': '-noZdis'}[dwi_file_pe[0]]
    args = args.replace(rm_arg, '')

    workflow = Workflow(name=name)
    workflow.__desc__ = """\
A deformation field to correct for susceptibility distortions was estimated
based on two echo-planar imaging (EPI) references with opposing phase-encoding
directions, using `3dQwarp` @afni (AFNI {afni_ver}).
""".format(afni_ver=''.join(['%02d' % v for v in afni.Info().version() or []]))

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['in_reference', 'in_reference_brain', 'in_mask']), name='inputnode')

    outputnode = pe.Node(
        niu.IdentityInterface(fields=['out_reference', 'out_warp']),
        name='outputnode')

    prepare_epi_opposite_wf = init_prepare_dwi_epi_wf(omp_nthreads=omp_nthreads,
                                                      name="prepare_epi_opposite_wf")
    prepare_epi_opposite_wf.inputs.inputnode.fmaps = epi_fmaps

    qwarp = pe.Node(afni.QwarpPlusMinus(pblur=[0.05, 0.05],
                                        blur=[-1, -1],
                                        noweight=True,
                                        minpatch=9,
                                        nopadWARP=True,
                                        environ={'OMP_NUM_THREADS': '%d' % omp_nthreads},
                                        args=args),
                    name='qwarp', n_procs=omp_nthreads)

    workflow.connect([
        (inputnode, prepare_epi_opposite_wf, [('in_reference_brain', 'inputnode.ref_brain')]),
        (prepare_epi_opposite_wf, qwarp, [('outputnode.out_file', 'base_file')]),
        (inputnode, qwarp, [('in_reference_brain', 'in_file')])
    ])

    to_ants = pe.Node(niu.Function(function=_fix_hdr), name='to_ants',
                      mem_gb=0.01)

    cphdr_warp = pe.Node(CopyHeader(), name='cphdr_warp', mem_gb=0.01)

    unwarp_reference = pe.Node(ANTSApplyTransformsRPT(dimension=3,
                                                      generate_report=False,
                                                      float=True,
                                                      interpolation='LanczosWindowedSinc'),
                               name='unwarp_reference')

    workflow.connect([
        (inputnode, cphdr_warp, [('in_reference', 'hdr_file')]),
        (qwarp, cphdr_warp, [('source_warp', 'in_file')]),
        (cphdr_warp, to_ants, [('out_file', 'in_file')]),
        (to_ants, unwarp_reference, [('out', 'transforms')]),
        (inputnode, unwarp_reference, [('in_reference', 'reference_image'),
                                       ('in_reference', 'input_image')]),
        (unwarp_reference, outputnode, [('output_image', 'out_reference')]),
        (to_ants, outputnode, [('out', 'out_warp')]),
    ])

    return workflow


def init_prepare_dwi_epi_wf(omp_nthreads, orientation="LPS", name="prepare_epi_wf"):
    """
    This workflow takes in a set of dwi files with with the same phase
    encoding direction and returns a single 3D volume ready to be used in
    field distortion estimation. It removes b>0 volumes.

    The procedure involves: estimating a robust template using FreeSurfer's
    'mri_robust_template', bias field correction using ANTs N4BiasFieldCorrection
    and AFNI 3dUnifize, skullstripping using FSL BET and AFNI 3dAutomask,
    and rigid coregistration to the reference using ANTs.
    """
    inputnode = pe.Node(niu.IdentityInterface(fields=['fmaps', 'ref_brain']),
                        name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(fields=['out_file']),
                         name='outputnode')

    prepare_b0s = pe.MapNode(
        B0RPEFieldmap(output_3d_images=True, orientation=orientation),
        iterfield='b0_file', name='prepare_b0s')

    merge = pe.Node(
        StructuralReference(auto_detect_sensitivity=True,
                            initial_timepoint=1,
                            fixed_timepoint=True,  # Align to first image
                            intensity_scaling=True,
                            # 7-DOF (rigid + intensity)
                            no_iteration=True,
                            subsample_threshold=200,
                            out_file='template.nii.gz'),
        name='merge')

    enhance_b0 = pe.Node(EnhanceB0(), name='enhance_b0')
    ants_settings = pkgr.resource_filename('qsiprep',
                                           'data/translation_rigid.json')
    fmap2ref_reg = pe.Node(ants.Registration(from_file=ants_settings,
                                             output_warped_image=True),
                           name='fmap2ref_reg', n_procs=omp_nthreads)
    resample_epi_fmap = pe.Node(ANTSApplyTransformsRPT(dimension=3,
                                                       generate_report=False,
                                                       float=True,
                                                       interpolation='LanczosWindowedSinc'),
                                name='resample_epi_fmap')
    workflow = Workflow(name=name)

    def _flatten(l):
        from nipype.utils.filemanip import filename_to_list
        return [item for sublist in l for item in filename_to_list(sublist)]

    workflow.connect([
        (inputnode, prepare_b0s, [('fmaps', 'b0_file')]),
        (prepare_b0s, merge, [(('fmap_file', _flatten), 'in_files')]),
        (merge, enhance_b0, [('out_file', 'b0_file')]),
        (enhance_b0, fmap2ref_reg, [('enhanced_file', 'moving_image')]),
        (inputnode, fmap2ref_reg, [('ref_brain', 'fixed_image')]),
        (fmap2ref_reg, resample_epi_fmap, [('composite_transform', 'transforms')]),
        (enhance_b0, resample_epi_fmap, [('enhanced_file', 'input_image')]),
        (inputnode, resample_epi_fmap, [('ref_brain', 'reference_image')]),
        (resample_epi_fmap, outputnode, [('output_image', 'out_file')])
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
