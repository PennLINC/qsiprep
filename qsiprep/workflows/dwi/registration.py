# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from nipype.interfaces import ants
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from pkg_resources import resource_filename as pkgrf

from ... import config
from ...interfaces.itk import ACPCReport, AffineToRigid
from ...interfaces.niworkflows import ANTSRegistrationRPT

DEFAULT_MEMORY_MIN_GB = 0.01


def init_b0_to_anat_registration_wf(
    write_report=True, transform_type='Rigid', name='b0_anat_coreg'
):
    """
    Calculates the registration between a reference b0 image and T1-space
    using `antsRegistration`

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.dwi.registration import init_b0_to_anat_registration_wf
        wf = init_b0_to_anat_registration_wf(
                              transform_type="Rigid",
                              write_report=False)

    Parameters
    ----------
    mem_gb : float
        Size of DWI file in GB
    omp_nthreads : int
        Maximum number of threads an individual process may use
    name : str
        Name of workflow (default: ``bold_reg_wf``)
    transform_type : str
        Either "Rigid" or "Affine"
    write_report : bool
        Should a reportlet be written?

    Inputs
    ------
    ref_b0_brain
        Reference image to which DWI series is aligned
        If ``fieldwarp == True``, ``ref_bold_brain`` should be unwarped
    t1_brain
        Skull-stripped ``t1_preproc``
    t1_seg
        Segmentation of preprocessed structural image, including
        gray-matter (GM), white-matter (WM) and cerebrospinal fluid (CSF)
    subjects_dir
        FreeSurfer SUBJECTS_DIR
    subject_id
        FreeSurfer subject ID

    Outputs
    -------
    itk_b0_to_t1
        Affine transform from ``ref_bold_brain`` to T1 space (ITK format)
    itk_t1_to_b0
        Affine transform from T1 space to DWI space (ITK format)
    coreg_metric
        Mattes score from the coregistration
    fallback
        Boolean indicating whether BBR was rejected (mri_coreg registration returned)
    report
        svg reportlet for the coregistration

    """
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'ref_b0_brain',
                't1_brain',
                't1_seg',
                'subjects_dir',
                'subject_id',
            ]
        ),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['itk_b0_to_t1', 'itk_t1_to_b0', 'fallback', 'coreg_metric', 'report']
        ),
        name='outputnode',
    )

    workflow = Workflow(name=name)

    # Defines a coregistration operation
    coreg = ANTSRegistrationRPT(generate_report=write_report)
    coreg.inputs.metric = ['Mattes']
    coreg.inputs.transforms = [transform_type]
    coreg.inputs.shrink_factors = [[8, 4, 2, 1]]
    coreg.inputs.smoothing_sigmas = [[7.0, 3.0, 1.0, 0.0]]
    coreg.inputs.sigma_units = ['vox']
    coreg.inputs.sampling_strategy = ['Random']
    coreg.inputs.sampling_percentage = [0.25]
    coreg.inputs.radius_or_number_of_bins = [32]
    coreg.inputs.initial_moving_transform_com = 0
    coreg.inputs.interpolation = 'HammingWindowedSinc'
    coreg.inputs.dimension = 3
    coreg.inputs.winsorize_lower_quantile = 0.025
    coreg.inputs.winsorize_upper_quantile = 0.975
    coreg.inputs.number_of_iterations = [[10000, 1000, 10000, 10000]]
    coreg.inputs.transform_parameters = [[0.2]]
    coreg.inputs.convergence_threshold = [1e-06]
    coreg.inputs.collapse_output_transforms = True
    coreg.inputs.write_composite_transform = False
    coreg.inputs.output_warped_image = True
    b0_to_anat = pe.Node(coreg, name='b0_to_anat', n_procs=config.nipype.omp_nthreads)

    workflow.connect([
        (inputnode, b0_to_anat, [
            ('t1_brain', 'fixed_image'),
            ('ref_b0_brain', 'moving_image'),
        ]),
        (b0_to_anat, outputnode, [
            ('forward_transforms', 'itk_b0_to_t1'),
            ('reverse_transforms', 'itk_t1_to_b0'),
            ('metric_value', 'coreg_metric'),
            ('out_report', 'report'),
        ]),
    ])  # fmt:skip
    return workflow


def init_direct_b0_acpc_wf(write_report=True, name='b0_anat_coreg'):
    """
    Re-orients a b=0 image directly to AC-PC. A full affine registration is run,
    but only the rigid (translation + rotation) part is included.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.dwi.registration import init_direct_b0_acpc_wf
        wf = init_direct_b0_acpc_wf(mem_gb=3,
                                    omp_nthreads=1,
                                    write_report=False)

    **Parameters**
        baby_mode : bool
            Use the infant t1w brain as the reference volume
        mem_gb : float
            Size of DWI file in GB
        omp_nthreads : int
            Maximum number of threads an individual process may use
        name : str
            Name of workflow (default: ``bold_reg_wf``)
        transform_type : str
            Either "Rigid" or "Affine"
        write_report : bool
            Should a reportlet be written?

    **Inputs**

        ref_b0_brain
            Reference image to which DWI series is aligned
            If ``fieldwarp == True``, ``ref_bold_brain`` should be unwarped
        t1_brain
            Standard space brain, either adult or infant template
        t1_seg
            Segmentation of preprocessed structural image, including
            gray-matter (GM), white-matter (WM) and cerebrospinal fluid (CSF)
        subjects_dir
            FreeSurfer SUBJECTS_DIR
        subject_id
            FreeSurfer subject ID

    **Outputs**

        itk_b0_to_t1
            Affine transform from ``ref_bold_brain`` to T1 space (ITK format)
        itk_t1_to_b0
            Affine transform from T1 space to DWI space (ITK format)
        coreg_metric
            Mattes score from the coregistration
        report
            svg reportlet for the coregistration
    """
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'ref_b0_brain',
                't1_brain',
                't1_seg',
                'subjects_dir',
                'subject_id',
            ]
        ),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['itk_b0_to_t1', 'itk_t1_to_b0', 'fallback', 'coreg_metric', 'report']
        ),
        name='outputnode',
    )

    workflow = Workflow(name=name)

    # Defines a coregistration operation
    ants_settings = pkgrf('qsiprep', 'data/intermodal_ACPC.json')
    acpc_reg = pe.Node(
        ANTSRegistrationRPT(generate_report=write_report, from_file=ants_settings),
        name='acpc_reg',
        n_procs=config.nipype.omp_nthreads,
    )

    # Extract the rigid components of the transform
    itk_to_rigid = pe.Node(AffineToRigid(), name='itk_to_rigid')

    # Apply the rigid transform to the b=0 ref
    translation_warp = pe.Node(
        ants.ApplyTransforms(dimension=3, interpolation='BSpline'), name='translation_warp'
    )
    rigid_warp = pe.Node(
        ants.ApplyTransforms(dimension=3, interpolation='BSpline'), name='rigid_warp'
    )
    acpc_report = pe.Node(ACPCReport(), name='acpc_report')

    workflow.connect([
        (inputnode, acpc_reg, [
            ('t1_brain', 'fixed_image'),
            (('t1_seg', _format_masks), 'fixed_image_masks'),
            ('ref_b0_brain', 'moving_image'),
        ]),
        (acpc_reg, itk_to_rigid, [('forward_transforms', 'affine_transform')]),
        (itk_to_rigid, rigid_warp, [('rigid_transform', 'transforms')]),
        (itk_to_rigid, translation_warp, [('translation_transform', 'transforms')]),
        (inputnode, rigid_warp, [
            ('ref_b0_brain', 'input_image'),
            ('t1_brain', 'reference_image'),
        ]),
        (inputnode, translation_warp, [
            ('ref_b0_brain', 'input_image'),
            ('t1_brain', 'reference_image'),
        ]),
        (translation_warp, acpc_report, [('output_image', 'translation_image')]),
        (rigid_warp, acpc_report, [('output_image', 'rigid_image')]),
        (itk_to_rigid, outputnode, [
            ('rigid_transform', 'itk_b0_to_t1'),
            ('rigid_transform_inverse', 'itk_t1_to_b0'),
        ]),
        (acpc_report, outputnode, [('out_report', 'report')]),
    ])  # fmt:skip

    return workflow


def _format_masks(mask_file):
    return ['NULL', mask_file]
