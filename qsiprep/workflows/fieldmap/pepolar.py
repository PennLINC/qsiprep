# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
.. _sdc_pepolar :

Phase Encoding POLARity (*PEPOLAR*) techniques
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""

from nipype.interfaces import ants
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from niworkflows.interfaces.reportlets.registration import ANTSApplyTransformsRPT

from ...data import load as load_data
from ...interfaces import StructuralReference
from ...interfaces.fmap import B0RPEFieldmap, PEPOLARReport
from ...interfaces.images import ExtractWM
from ...interfaces.nilearn import EnhanceB0
from ..anatomical import init_synthstrip_wf


def init_prepare_dwi_epi_wf(omp_nthreads, orientation='LPS', name='prepare_epi_wf'):
    """
    This workflow takes in a set of dwi files with with the same phase
    encoding direction and returns a single 3D volume ready to be used in
    field distortion estimation. It removes b>0 volumes.

    The procedure involves: estimating a robust template using FreeSurfer's
    'mri_robust_template', bias field correction using ANTs N4BiasFieldCorrection
    and AFNI 3dUnifize, skullstripping using FSL BET and AFNI 3dAutomask,
    and rigid coregistration to the reference using ANTs.
    """
    inputnode = pe.Node(niu.IdentityInterface(fields=['fmaps', 'ref_brain']), name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(fields=['out_file']), name='outputnode')

    prepare_b0s = pe.MapNode(
        B0RPEFieldmap(output_3d_images=True, orientation=orientation),
        iterfield='b0_file',
        name='prepare_b0s',
    )

    merge = pe.Node(
        StructuralReference(
            auto_detect_sensitivity=True,
            initial_timepoint=1,
            fixed_timepoint=True,  # Align to first image
            intensity_scaling=True,
            # 7-DOF (rigid + intensity)
            no_iteration=True,
            subsample_threshold=200,
            out_file='template.nii.gz',
        ),
        name='merge',
    )

    enhance_b0 = pe.Node(EnhanceB0(), name='enhance_b0')
    ants_settings = str(load_data('translation_rigid.json'))
    fmap2ref_reg = pe.Node(
        ants.Registration(from_file=ants_settings, output_warped_image=True),
        name='fmap2ref_reg',
        n_procs=omp_nthreads,
    )
    resample_epi_fmap = pe.Node(
        ANTSApplyTransformsRPT(
            dimension=3, generate_report=False, float=True, interpolation='LanczosWindowedSinc'
        ),
        name='resample_epi_fmap',
    )
    workflow = Workflow(name=name)

    def _flatten(ell):
        from nipype.utils.filemanip import filename_to_list

        return [item for sublist in ell for item in filename_to_list(sublist)]

    workflow.connect([
        (inputnode, prepare_b0s, [('fmaps', 'b0_file')]),
        (prepare_b0s, merge, [(('fmap_file', _flatten), 'in_files')]),
        (merge, enhance_b0, [('out_file', 'b0_file')]),
        (enhance_b0, fmap2ref_reg, [('enhanced_file', 'moving_image')]),
        (inputnode, fmap2ref_reg, [('ref_brain', 'fixed_image')]),
        (fmap2ref_reg, resample_epi_fmap, [('composite_transform', 'transforms')]),
        (enhance_b0, resample_epi_fmap, [('enhanced_file', 'input_image')]),
        (inputnode, resample_epi_fmap, [('ref_brain', 'reference_image')]),
        (resample_epi_fmap, outputnode, [('output_image', 'out_file')]),
    ])  # fmt:skip

    return workflow


def init_extended_pepolar_report_wf(
    segment_t2w, omp_nthreads=1, name='extended_pepolar_report_wf'
):
    workflow = Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                't1w_seg_transform',
                't1w_seg',
                'b0_ref',
                'fieldmap_type',
                'b0_up_image',
                'b0_up_corrected_image',
                'b0_down_image',
                'b0_down_corrected_image',
                'up_fa_image',
                'up_fa_corrected_image',
                'down_fa_image',
                'down_fa_corrected_image',
                't2w_image',
            ]
        ),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['fa_sdc_report', 'b0_sdc_report']), name='outputnode'
    )

    pepolar_report = pe.Node(PEPOLARReport(), name='peoplar_report')

    workflow.connect([
        (inputnode, pepolar_report, [
            ('fieldmap_type', 'fieldmap_type'),
            ('b0_up_image', 'b0_up_image'),
            ('b0_up_corrected_image', 'b0_up_corrected_image'),
            ('b0_down_image', 'b0_down_image'),
            ('b0_down_corrected_image', 'b0_down_corrected_image'),
            ('up_fa_image', 'up_fa_image'),
            ('up_fa_corrected_image', 'up_fa_corrected_image'),
            ('down_fa_image', 'down_fa_image'),
            ('down_fa_corrected_image', 'down_fa_corrected_image'),
        ]),
        (pepolar_report, outputnode, [
            ('b0_sdc_report', 'b0_sdc_report'),
            ('fa_sdc_report', 'fa_sdc_report'),
        ]),
    ])  # fmt:skip

    # If we don't have a T1w segmentation, make one from the t2w
    if segment_t2w:
        t2w_n4 = pe.Node(
            ants.N4BiasFieldCorrection(dimension=3), name='t2w_n4', n_procs=omp_nthreads
        )

        strip_t2w_wf = init_synthstrip_wf(do_padding=True)

        t2w_atropos = pe.Node(
            ants.Atropos(
                dimension=3,
                initialization='Otsu',
                mrf_radius=[1, 1, 1],
                posterior_formulation='Socrates',
                use_mixture_model_proportions=False,
                mrf_smoothing_factor=0.1,
                number_of_tissue_classes=3,
            ),
            name='t2w_atropos',
            n_procs=omp_nthreads,
        )

        workflow.connect([
            (inputnode, t2w_n4, [('t2w_image', 'input_image')]),
            (t2w_n4, strip_t2w_wf, [('output_image', 'inputnode.original_image')]),
            (strip_t2w_wf, t2w_atropos, [
                ('outputnode.brain_image', 'intensity_images'),
                ('outputnode.brain_mask', 'mask_image'),
            ]),
            (t2w_atropos, pepolar_report, [('classified_image', 't2w_seg')]),
        ])  # fmt:skip
    else:
        map_seg = pe.Node(
            ants.ApplyTransforms(
                dimension=3, float=True, interpolation='MultiLabel', invert_transform_flags=[True]
            ),
            name='map_seg',
            mem_gb=0.3,
        )

        sel_wm = pe.Node(ExtractWM(), name='sel_wm')

        workflow.connect([
            (inputnode, map_seg, [
                ('b0_ref', 'reference_image'),
                ('t1w_seg_transform', 'transforms'),
                ('t1w_seg', 'input_image'),
            ]),
            (map_seg, sel_wm, [('output_image', 'in_seg')]),
            (sel_wm, pepolar_report, [('out', 't1w_seg')]),
        ])  # fmt:skip

    return workflow


def _fix_hdr(in_file, newpath=None):
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix

    nii = nb.load(in_file)
    hdr = nii.header.copy()
    hdr.set_data_dtype('<f4')
    hdr.set_intent('vector', (), '')
    out_file = fname_presuffix(in_file, '_warpfield', newpath=newpath)
    nb.Nifti1Image(nii.get_fdata().astype('<f4'), nii.affine, hdr).to_filename(out_file)
    return out_file
