# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from nipype.interfaces import ants
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from niworkflows.interfaces.nibabel import RegridToZooms

from ... import config
from ...data import load as load_data
from ...interfaces.itk import ACPCReport, AffineToRigid
from ...interfaces.niworkflows import ANTSRegistrationRPT

DEFAULT_MEMORY_MIN_GB = 0.01


def init_rotation_search_wf(transform='Rigid', name='rotation_search_wf'):
    """Estimate an initial transform for a coregistration by global rotation search.

    ``antsAI`` refines a grid of candidate orientations (all combinations of
    three Euler angles at 20 degree spacing, +-90 degrees per axis) with a few
    conjugate-gradient iterations each and keeps the candidate with the best
    Mattes MI. A center-of-mass start alone leaves antsRegistration to recover
    the full rotation, and a Mattes fit reliably captures only rotations of a
    few tens of degrees; anything larger (an infant positioned differently for
    the dMRI than for the anatomical) converges to a rotated local optimum.
    The search runs on 4 mm resamples of the inputs: there are on the order of
    a thousand candidates, and the result only needs to land within the
    capture range of the full-resolution registration that follows.

    Parameters
    ----------
    transform : str
        'Rigid', 'Similarity' or 'Affine': the transform ``antsAI`` optimizes
        at each candidate orientation. Use 'Rigid' between images of the same
        subject and 'Similarity' against a template, where scale is unknown.
    name : str
        Name of workflow (default: ``rotation_search_wf``)

    Inputs
    ------
    fixed_image
        Image being registered to
    moving_image
        Image that will be transformed to fixed_image

    Outputs
    -------
    initial_transform
        ITK transform file, suitable for ``initial_moving_transform``
    """
    workflow = Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['fixed_image', 'moving_image']), name='inputnode'
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=['initial_transform']), name='outputnode')

    res_fixed = pe.Node(RegridToZooms(zooms=(4.0, 4.0, 4.0), smooth=True), name='res_fixed')
    res_moving = pe.Node(RegridToZooms(zooms=(4.0, 4.0, 4.0), smooth=True), name='res_moving')

    # In sloppy mode the arc shrinks to +-18 degrees (8 candidates vs 1000)
    arc_fraction = 0.1 if config.execution.sloppy else 0.5
    rotation_search = pe.Node(
        ants.AI(
            metric=('Mattes', 32, 'Regular', 0.25),
            transform=(transform, 0.1),
            search_factor=(20.0, arc_fraction),
            principal_axes=False,
            convergence=(10, 1e-6, 10),
            verbose=True,
        ),
        name='rotation_search',
        n_procs=config.nipype.omp_nthreads,
    )

    workflow.connect([
        (inputnode, res_fixed, [('fixed_image', 'in_file')]),
        (inputnode, res_moving, [('moving_image', 'in_file')]),
        (res_fixed, rotation_search, [('out_file', 'fixed_image')]),
        (res_moving, rotation_search, [('out_file', 'moving_image')]),
        (rotation_search, outputnode, [('output_transform', 'initial_transform')]),
    ])  # fmt:skip
    return workflow


def init_structural_to_b0_alignment_wf(name='structural_to_b0_alignment_wf'):
    """Pre-align a structural image into the frame of a distorted b=0 reference.

    TORTOISE rigidly registers its structural input to the b=0 internally
    (DRBUDDI, and EPIREG for DIFFPREP's ``--epi T2Wreg`` mode), but only from
    a center-of-mass initialization, which cannot recover a large rotation
    between the anatomical and the dMRI acquisition. Resampling the
    structural into the b=0 frame first, through an ``antsAI`` rotation
    search, hands TORTOISE a target already within its capture range. The
    search transform is deliberately coarse: TORTOISE's own rigid
    registration provides the fine alignment.

    Parameters
    ----------
    name : str
        Name of workflow (default: ``structural_to_b0_alignment_wf``)

    Inputs
    ------
    structural_image
        Anatomical image (e.g. the unfatsat T2w), in any orientation
    b0_ref
        b=0 reference image in the frame the distortion correction runs in

    Outputs
    -------
    structural_aligned
        The structural image resampled onto the b=0 reference grid
    """
    workflow = Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['structural_image', 'b0_ref']), name='inputnode'
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=['structural_aligned']), name='outputnode')

    # fixed=b0: antsAI's transform then maps b0-space points to structural
    # space, which is exactly what resampling onto the b0 grid needs
    rotation_search_wf = init_rotation_search_wf(transform='Rigid')
    resample_structural = pe.Node(
        ants.ApplyTransforms(dimension=3, interpolation='LanczosWindowedSinc'),
        name='resample_structural',
    )

    workflow.connect([
        (inputnode, rotation_search_wf, [
            ('b0_ref', 'inputnode.fixed_image'),
            ('structural_image', 'inputnode.moving_image'),
        ]),
        (inputnode, resample_structural, [
            ('structural_image', 'input_image'),
            ('b0_ref', 'reference_image'),
        ]),
        (rotation_search_wf, resample_structural, [
            ('outputnode.initial_transform', 'transforms'),
        ]),
        (resample_structural, outputnode, [('output_image', 'structural_aligned')]),
    ])  # fmt:skip
    return workflow


def init_b0_to_anat_registration_wf(
    write_report=True, transform_type='Rigid', name='b0_anat_coreg'
):
    """
    Calculates the registration between a reference b0 image and T1-space
    using `antsRegistration`, initialized by an ``antsAI`` rotation search
    so that large orientation differences between the dMRI and the
    anatomical are recovered.

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

    rotation_search_wf = init_rotation_search_wf(transform='Rigid')

    workflow.connect([
        (inputnode, rotation_search_wf, [
            ('t1_brain', 'inputnode.fixed_image'),
            ('ref_b0_brain', 'inputnode.moving_image'),
        ]),
        (rotation_search_wf, b0_to_anat, [
            ('outputnode.initial_transform', 'initial_moving_transform'),
        ]),
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
    ants_settings = str(load_data('intermodal_ACPC.json'))
    acpc_reg = pe.Node(
        ANTSRegistrationRPT(generate_report=write_report, from_file=ants_settings),
        name='acpc_reg',
        n_procs=config.nipype.omp_nthreads,
    )

    # The template is another subject at another age, so scale is unknown
    rotation_search_wf = init_rotation_search_wf(transform='Similarity')

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
        (inputnode, rotation_search_wf, [
            ('t1_brain', 'inputnode.fixed_image'),
            ('ref_b0_brain', 'inputnode.moving_image'),
        ]),
        (rotation_search_wf, acpc_reg, [
            ('outputnode.initial_transform', 'initial_moving_transform'),
        ]),
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
