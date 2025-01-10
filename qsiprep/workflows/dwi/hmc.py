"""
Head motion correction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_dwi_hmc_wf
.. autofunction:: init_dwi_model_hmc_wf

"""

import nipype.pipeline.engine as pe
from nipype.interfaces import afni, ants
from nipype.interfaces import utility as niu
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from pkg_resources import resource_filename as pkgrf

from ... import config
from ...interfaces import DerivativesDataSink
from ...interfaces.gradients import CombineMotions, GradientRotation, MatchTransforms
from ...interfaces.shoreline import (
    B0Mean,
    CalculateCNR,
    ExtractDWIsForModel,
    IterationSummary,
    ReorderOutputs,
    SHORELineReport,
    SignalPrediction,
)
from .util import init_dwi_reference_wf

DEFAULT_MEMORY_MIN_GB = 0.01


def init_dwi_hmc_wf(
    source_file,
    num_model_iterations=2,
    mem_gb=3,
    name='dwi_hmc_wf',
):
    """Perform head motion correction and susceptibility distortion correction.

    This workflow uses antsRegistration and an iteratively updated signal model to perform
    motion correction.

    **Parameters**

        hmc_transform: 'Rigid' or 'Affine'
            How many degrees of freedom to incorporate into motion correction
        hmc_model: '3dSHORE', 'none' , 'tensor' or 'SH'
            Which model to use for generating signal predictions for hmc. '3dSHORE' requires
            multiple b-values, 'none' will only use b0 images for motion correction, 'tensor'
            uses a tensor model for signal predictions for hmc, and 'SH' uses spherical harmonics
            (not implemented yet).
        hmc_align_to: 'first' or 'iterative'
            Which volume should be used to determine the motion-corrected space?
        source_file: str
            Path to one of the original dwi files (used for reportlets)
        rpe_b0: str
            Path to a reverse phase encoding image to be used for 3dQWarp's TOPUP-style
            correction
        num_model_iterations: int
            If ``hmc_model`` is ``'3dSHORE'`` or ``'SH'`` determines the number of times the
            model is updated and motion correction is estimated. Default: 2.

    **Inputs**

        dwi_files: list
            List of single-volume files across all DWI series
        b0_indices: list
            Indexes into ``dwi_files`` that correspond to b=0 volumes
        bvecs: list
            List of paths to single-line bvec files
        bvals: list
            List of paths to single-line bval files
        b0_images: list
            List of single b=0 volumes
        original_files: list
            List of the files from which each DWI volume came from.

    **Outputs**

        final_template: str
            Path to the mean of the coregistered b0 images
        forward_transforms: list
            List of ITK transforms that motion-correct the images in ``dwi_files``
        noise_free_dwis: list
            Model-predicted images reverse-transformed into alignment with ``dwi_files``
        cnr_image: str
            If hmc_model is 'none' this is the tsnr of the b=0 images. Otherwise it is the
            model fit divided by the model error in each voxel.
        optimization_data: str
            CSV file tracking the motion estimates across shoreline iterations
    """

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'dwi_files',
                'b0_indices',
                'bvecs',
                'bvals',
                'b0_images',
                'original_files',
                't1_brain',
                't1_mask',
                't1_seg',
                'original_files',
            ]
        ),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'final_template',
                'forward_transforms',
                'noise_free_dwis',
                'cnr_image',
                'optimization_data',
                'final_template_brain',
                'final_template_mask',
            ]
        ),
        name='outputnode',
    )
    hmc_model = config.workflow.hmc_model
    workflow = Workflow(name=name)
    # Unbiased align the b0s
    b0_hmc_wf = init_b0_hmc_wf()
    # Tile the transforms so each non-b0 gets the transform from the nearest b0
    match_transforms = pe.Node(MatchTransforms(), name='match_transforms')
    # Make a mask from the output template. It is bias-corrected, so good for masking
    # but bad for using as a b=0 for modeling.
    b0_template_mask = init_dwi_reference_wf(
        name='b0_template_mask',
        gen_report=False,
        source_file=source_file,
    )

    workflow.connect([
        (inputnode, match_transforms, [
            ('dwi_files', 'dwi_files'),
            ('b0_indices', 'b0_indices'),
        ]),
        (inputnode, b0_hmc_wf, [('b0_images', 'inputnode.b0_images')]),
        (b0_hmc_wf, outputnode, [('outputnode.final_template', 'final_template')]),
        (b0_hmc_wf, match_transforms, [
            (('outputnode.forward_transforms', _list_squeeze), 'transforms'),
        ]),
        (inputnode, b0_template_mask, [
            ('t1_brain', 'inputnode.t1_brain'),
            ('t1_seg', 'inputnode.t1_seg'),
            ('t1_mask', 'inputnode.t1_mask'),
        ]),
        (b0_hmc_wf, b0_template_mask, [('outputnode.final_template', 'inputnode.b0_template')]),
        (b0_template_mask, outputnode, [
            ('outputnode.ref_image_brain', 'final_template_brain'),
            ('outputnode.dwi_mask', 'final_template_mask'),
        ]),
    ])  # fmt:skip

    # If we're just aligning based on the b=0 images, compute the b=0 tsnr as the cnr
    if hmc_model.lower() == 'none':
        workflow.__postdesc__ = (
            'Each b>0 image was transformed based on the registration of the nearest b=0 image. '
        )

        concat_b0s = pe.Node(afni.TCat(outputtype='NIFTI_GZ'), name='concat_b0s')
        b0_tsnr = pe.Node(
            afni.TStat(options=' -cvarinvNOD ', outputtype='NIFTI_GZ'), name='b0_tsnr'
        )
        workflow.connect([
            (match_transforms, outputnode, [('transforms', 'forward_transforms')]),
            (b0_hmc_wf, concat_b0s, [('outputnode.aligned_images', 'in_files')]),
            (concat_b0s, b0_tsnr, [('out_file', 'in_file')]),
            (b0_tsnr, outputnode, [('out_file', 'cnr_image')]),
        ])  # fmt:skip
        return workflow

    # Do model-based motion correction
    dwi_model_hmc_wf = init_dwi_model_hmc_wf(num_iters=num_model_iterations)

    # Warp the modeled images into non-motion-corrected space
    uncorrect_model_images = pe.MapNode(
        ants.ApplyTransforms(
            invert_transform_flags=[True],
            interpolation=(
                'LanczosWindowedSinc' if not config.execution.sloppy else 'NearestNeighbor'
            ),
        ),
        iterfield=['input_image', 'reference_image', 'transforms'],
        name='uncorrect_model_images',
    )
    workflow.__postdesc__ = (
        'Model-generated images were transformed into alignment with each '
        'b>0 image. Both slicewise and whole-brain QC measures (cross '
        'correlation and R^2) were calculated.'
    )
    workflow.connect([
        (b0_hmc_wf, dwi_model_hmc_wf, [
            ('outputnode.aligned_images', 'inputnode.warped_b0_images'),
        ]),
        (b0_template_mask, dwi_model_hmc_wf, [
            ('outputnode.dwi_mask', 'inputnode.warped_b0_mask'),
        ]),
        (inputnode, dwi_model_hmc_wf, [
            ('dwi_files', 'inputnode.dwi_files'),
            ('b0_indices', 'inputnode.b0_indices'),
            ('bvecs', 'inputnode.bvec_files'),
            ('bvals', 'inputnode.bval_files'),
        ]),
        (match_transforms, dwi_model_hmc_wf, [('transforms', 'inputnode.initial_transforms')]),
        (dwi_model_hmc_wf, outputnode, [
            ('outputnode.hmc_transforms', 'forward_transforms'),
            ('outputnode.optimization_data', 'optimization_data'),
            ('outputnode.cnr_image', 'cnr_image'),
        ]),
        (dwi_model_hmc_wf, uncorrect_model_images, [
            ('outputnode.model_predicted_images', 'input_image'),
            ('outputnode.hmc_transforms', 'transforms'),
        ]),
        (inputnode, uncorrect_model_images, [('dwi_files', 'reference_image')]),
        (uncorrect_model_images, outputnode, [('output_image', 'noise_free_dwis')]),
    ])  # fmt:skip
    datasinks = [
        node for node in workflow.list_node_names() if node.split('.')[-1].startswith('ds_')
    ]

    for ds in datasinks:
        workflow.get_node(ds).inputs.source_file = source_file

    return workflow


def linear_alignment_workflow(transform='Rigid', iternum=0, omp_nthreads=1):
    """
    Takes a template image and a set of input images, does
    a linear alignment to the template and updates it with the
    inverse of the average affine transform to the new template

    Returns a workflow

    """
    iteration_wf = Workflow(name='iterative_alignment_%03d' % iternum)
    input_node_fields = ['image_paths', 'template_image', 'iteration_num']
    inputnode = pe.Node(niu.IdentityInterface(fields=input_node_fields), name='inputnode')
    inputnode.inputs.iteration_num = iternum
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['registered_image_paths', 'affine_transforms', 'updated_template']
        ),
        name='outputnode',
    )
    precision = 'sloppy' if config.execution.sloppy else 'precise'
    ants_settings = pkgrf(
        'qsiprep',
        f'data/shoreline_{precision}_{transform}.json',
    )
    reg = ants.Registration(from_file=ants_settings, num_threads=omp_nthreads)
    iter_reg = pe.MapNode(
        reg, name='reg_%03d' % iternum, iterfield=['moving_image'], n_procs=omp_nthreads
    )

    # Run the images through antsRegistration
    iteration_wf.connect(inputnode, 'image_paths', iter_reg, 'moving_image')  # fmt:skip
    iteration_wf.connect(inputnode, 'template_image', iter_reg, 'fixed_image')  # fmt:skip

    # Average the images
    averaged_images = pe.Node(
        ants.AverageImages(normalize=True, dimension=3), name='averaged_images'
    )
    iteration_wf.connect(iter_reg, 'warped_image', averaged_images, 'images')  # fmt:skip

    # Apply the inverse to the average image
    transforms_to_list = pe.Node(niu.Merge(1), name='transforms_to_list')
    transforms_to_list.inputs.ravel_inputs = True
    iteration_wf.connect(iter_reg, 'forward_transforms', transforms_to_list, 'in1')  # fmt:skip
    avg_affines = pe.Node(ants.AverageAffineTransform(), name='avg_affine')
    avg_affines.inputs.dimension = 3
    avg_affines.inputs.output_affine_transform = 'AveragedAffines.mat'
    iteration_wf.connect(transforms_to_list, 'out', avg_affines, 'transforms')  # fmt:skip

    invert_average = pe.Node(ants.ApplyTransforms(), name='invert_average')
    invert_average.inputs.interpolation = 'HammingWindowedSinc'
    invert_average.inputs.invert_transform_flags = [True]

    avg_to_list = pe.Node(niu.Merge(1), name='to_list')
    iteration_wf.connect(avg_affines, 'affine_transform', avg_to_list, 'in1')  # fmt:skip
    iteration_wf.connect(avg_to_list, 'out', invert_average, 'transforms')  # fmt:skip
    iteration_wf.connect(averaged_images, 'output_average_image',
                         invert_average, 'input_image')  # fmt:skip
    iteration_wf.connect(averaged_images, 'output_average_image',
                         invert_average, 'reference_image')  # fmt:skip
    iteration_wf.connect(invert_average, 'output_image', outputnode,
                         'updated_template')  # fmt:skip
    iteration_wf.connect(iter_reg, 'forward_transforms', outputnode,
                         'affine_transforms')  # fmt:skip
    iteration_wf.connect(iter_reg, 'warped_image', outputnode,
                         'registered_image_paths')  # fmt:skip

    return iteration_wf


def init_b0_hmc_wf(
    align_to=None,
    transform=None,
    boilerplate=True,
    name='b0_hmc_wf',
    prioritize_omp=False,
) -> Workflow:
    num_iters = 2
    if align_to is None:
        align_to = config.workflow.b0_motion_corr_to
    if transform is None:
        transform = config.workflow.hmc_transform
    if config.execution.sloppy:
        num_iters = 1
        align_to = 'first'
    omp_nthreads = config.nipype.omp_nthreads if prioritize_omp else 1
    if align_to == 'iterative' and num_iters < 2:
        raise ValueError('Must specify a positive number of iterations')

    alignment_wf = Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['b0_images']), name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'final_template',
                'forward_transforms',
                'iteration_templates',
                'motion_params',
                'aligned_images',
            ]
        ),
        name='outputnode',
    )

    desc = 'Initial motion correction was performed using only the b=0 images. '

    # Iteratively create a template
    if align_to == 'iterative':
        desc += (
            f'An unbiased b=0 template was constructed over {num_iters} iterations '
            f'of {config.workflow.hmc_model} registrations. '
        )
        initial_template = pe.Node(
            ants.AverageImages(normalize=True, dimension=3), name='initial_template'
        )
        alignment_wf.connect(inputnode, 'b0_images', initial_template, 'images')  # fmt:skip
        # Store the registration targets
        iter_templates = pe.Node(niu.Merge(num_iters), name='iteration_templates')
        alignment_wf.connect(initial_template, 'output_average_image',
                             iter_templates, 'in1')  # fmt:skip

        initial_reg = linear_alignment_workflow(iternum=0, omp_nthreads=omp_nthreads)
        alignment_wf.connect(initial_template, 'output_average_image',
                             initial_reg, 'inputnode.template_image')  # fmt:skip
        alignment_wf.connect(inputnode, 'b0_images', initial_reg,
                             'inputnode.image_paths')  # fmt:skip
        reg_iters = [initial_reg]
        for iternum in range(1, num_iters):
            reg_iters.append(linear_alignment_workflow(iternum=iternum, omp_nthreads=omp_nthreads))
            alignment_wf.connect(reg_iters[-2], 'outputnode.updated_template',
                                 reg_iters[-1], 'inputnode.template_image')  # fmt:skip
            alignment_wf.connect(inputnode, 'b0_images', reg_iters[-1],
                                 'inputnode.image_paths')  # fmt:skip
            alignment_wf.connect(reg_iters[-1], 'outputnode.updated_template',
                                 iter_templates, 'in%d' % (iternum + 1))  # fmt:skip

        # Attach to outputs
        # The last iteration aligned to the output from the second-to-last
        alignment_wf.connect(reg_iters[-2], 'outputnode.updated_template',
                             outputnode, 'final_template')  # fmt:skip
        alignment_wf.connect(reg_iters[-1], 'outputnode.affine_transforms',
                             outputnode, 'forward_transforms')  # fmt:skip
        alignment_wf.connect(reg_iters[-1], 'outputnode.registered_image_paths',
                             outputnode, 'aligned_images')  # fmt:skip
        alignment_wf.connect(iter_templates, 'out', outputnode,
                             'iteration_templates')  # fmt:skip
    elif align_to == 'first':
        desc += (
            'Each b=0 image was registered to the first b=0 image using '
            f'a {transform} registration. '
        )
        reg_to_first = linear_alignment_workflow(iternum=0, omp_nthreads=omp_nthreads)

        alignment_wf.connect([
            (inputnode, reg_to_first, [
                (('b0_images', first_image), 'inputnode.template_image'),
                ('b0_images', 'inputnode.image_paths')]),
            (reg_to_first, outputnode, [
                ('averaged_images.output_average_image', 'final_template'),
                ('outputnode.affine_transforms', 'forward_transforms'),
                ('outputnode.registered_image_paths', 'aligned_images')])
        ])  # fmt:skip
    if boilerplate:
        alignment_wf.__desc__ = desc
    return alignment_wf


def first_image(image_list):
    return image_list[0]


def _bvecs_to_list(bvec_file):
    import numpy as np

    bvec = np.loadtxt(bvec_file).T
    return list(bvec)


def _bvals_to_floats(bval_files):
    import numpy as np

    return [float(np.loadtxt(bval_file)) for bval_file in bval_files]


def init_hmc_model_iteration_wf(name='hmc_model_iter0'):
    """Create a model-based hmc registration iteration workflow.

    This workflow takes an initial set of transforms, applies them to the
    original data, and builds a signal model out of the transformed images.

    The original images are then registered to the model-generated target
    images. Included in the outputs are the transforms and optionally the
    noise-free images in original (non-aligned) space.

    .. workflow::
        :graph2use: colored
        :simple_form: yes

        from qsiprep.workflows.dwi.hmc import init_dwi_model_iteration_wf
        wf = init_dwi_model_hmc_wf(modelname='3dSHORE',
                                   transform='Affine',
                                   num_iters=2)

    Parameters
    ----------
    name : str
        name of the workflow

    Inputs

        original_dwi_files
            list of 3d dwi files, no b0's
        bvals
            list of bval files corresponding to `original_dwi_files`
        approx_aligned_dwi_files
            dwi files that have been registered through a shoreline iteration
        approx_aligned_bvecs
            list of bvec files corresponding to `approx_aligned_dwi_files`
        b0_indices
            list of which indices in `dwi_files` are b0 images
        initial_transforms
            list of transforms from a previous registration
        b0_mask
            mask of containing brain voxels
        b0_mean
            mean of the aligned b0 images

    **Outputs**

        hmc_transforms
            list of transforms, one per file in `dwi_files`
        rotated_bvecs
            rotated bvec matrix
    """

    workflow = Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'original_dwi_files',
                'bvals',
                'approx_aligned_dwi_files',
                'approx_aligned_bvecs',
                'b0_mask',
                'b0_mean',
                'original_bvecs',
            ]
        ),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'hmc_transforms',
                'aligned_dwis',
                'aligned_bvecs',
                'predicted_dwis',
                'motion_params',
            ]
        ),
        name='outputnode',
    )
    precision = 'sloppy' if config.execution.sloppy else 'precise'
    ants_settings = pkgrf(
        'qsiprep',
        f'data/shoreline_{precision}_{config.workflow.hmc_transform}.json',
    )

    predict_dwis = pe.MapNode(
        SignalPrediction(model=config.workflow.hmc_model),
        iterfield=['bval_to_predict', 'bvec_to_predict'],
        name='predict_dwis',
    )
    predict_dwis.synchronize = True

    # Register original images to the predicted images
    register_to_predicted = pe.MapNode(
        ants.Registration(from_file=ants_settings),
        iterfield=['fixed_image', 'moving_image'],
        name='register_to_predicted',
    )
    register_to_predicted.synchronize = True

    # Apply new transforms to bvecs
    post_bvec_transforms = pe.Node(GradientRotation(), name='post_bvec_transforms')

    # Summarize the motion
    calculate_motion = pe.Node(CombineMotions(), name='calculate_motion')

    workflow.connect([
        # Send inputs to DWI prediction
        (inputnode, predict_dwis, [
            ('approx_aligned_dwi_files', 'aligned_dwis'),
            ('approx_aligned_bvecs', 'aligned_bvecs'),
            ('bvals', 'bvals'),
            ('b0_mean', 'aligned_b0_mean'),
            ('b0_mask', 'aligned_mask'),
            (('approx_aligned_bvecs', _bvecs_to_list), 'bvec_to_predict'),
            (('bvals', _bvals_to_floats), 'bval_to_predict'),
        ]),
        (predict_dwis, register_to_predicted, [('predicted_image', 'fixed_image')]),
        (inputnode, register_to_predicted, [
            ('original_dwi_files', 'moving_image'),
            ('b0_mask', 'fixed_image_masks'),
        ]),
        (register_to_predicted, calculate_motion, [
            (('forward_transforms', _list_squeeze), 'transform_files'),
        ]),
        (inputnode, calculate_motion, [
            ('original_dwi_files', 'source_files'),
            ('b0_mean', 'ref_file'),
        ]),
        (calculate_motion, outputnode, [('motion_file', 'motion_params')]),
        (register_to_predicted, post_bvec_transforms, [
            (('forward_transforms', _list_squeeze), 'affine_transforms'),
        ]),
        (inputnode, post_bvec_transforms, [
            ('original_bvecs', 'bvec_files'),
            ('bvals', 'bval_files'),
        ]),
        (predict_dwis, outputnode, [('predicted_image', 'predicted_dwis')]),
        (post_bvec_transforms, outputnode, [('bvecs', 'aligned_bvecs')]),
        (register_to_predicted, outputnode, [
            ('warped_image', 'aligned_dwis'),
            ('forward_transforms', 'hmc_transforms'),
        ]),
    ])  # fmt:skip

    return workflow


def init_dwi_model_hmc_wf(
    num_iters=2,
    name='dwi_model_hmc_wf',
):
    """Create a model-based hmc workflow.

    .. workflow::
        :graph2use: colored
        :simple_form: yes

        from qsiprep.workflows.dwi.hmc import init_dwi_model_hmc_wf
        wf = init_dwi_model_hmc_wf(modelname='3dSHORE',
                                   transform='Affine',
                                   num_iters=2)

    Parameters
    ----------
    num_iters : int
        the number of times the model will be updated with transformed data

    Inputs
    ------
    dwi_files
        list of 3d dwi files
    b0_indices
        list of which indices in `dwi_files` are b0 images
    initial_transforms
        list of b0-based transforms from dwis to the b0 template
    warped_b0_images
        list of aligned b0 images
    b0_mask
        mask of containing brain voxels
    bvecs
        list of bvec files corresponding to `dwi_files`
    bvals
        list of bval files corresponding to `dwi_files`

    Outputs
    -------
    hmc_transforms
        list of transforms, one per file in `dwi_files`
    model_predicted_images: list
        Model-predicted images reverse-transformed into alignment with ``dwi_files``
    cnr_image: str
        If hmc_model is 'none' this is the tsnr of the b=0 images. Otherwise it is the
        model fit divided by the model error in each voxel.
    optimization_data: str
        CSV file tracking the motion estimates across shoreline iterations


    """
    workflow = Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'dwi_files',
                'b0_indices',
                'initial_transforms',
                'bvec_files',
                'bval_files',
                'warped_b0_images',
                'warped_b0_mask',
            ]
        ),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['hmc_transforms', 'model_predicted_images', 'cnr_image', 'optimization_data']
        ),
        name='outputnode',
    )
    workflow.__desc__ = (
        'The SHORELine method was used to estimate head motion in b>0 '
        'images. This entails leaving out each b>0 image and reconstructing '
        'the others using 3dSHORE [@merlet3dshore]. The signal for the left-'
        f'out image serves as the registration target. A total of {num_iters} '
        f'iterations were run using a {config.workflow.hmc_transform} transform. '
    )

    # Merge b0s into a single volume, put the non-b0 dwis into a list
    extract_dwis = pe.Node(ExtractDWIsForModel(), name='extract_dwis')

    # Initialize with the transforms provided
    b0_based_image_transforms = pe.MapNode(
        ants.ApplyTransforms(interpolation='BSpline'),
        iterfield=['input_image', 'transforms'],
        name='b0_based_image_transforms',
    )
    # Rotate the original bvecs as well
    b0_based_bvec_transforms = pe.Node(GradientRotation(), name='b0_based_bvec_transforms')

    # Create a mask and an average from the aligned b0 images
    b0_mean = pe.Node(B0Mean(), name='b0_mean')

    # Start building and connecting the model iterations
    initial_model_iteration = init_hmc_model_iteration_wf(name='initial_model_iteration')

    # Collect motion estimates across iterations
    collect_motion_params = pe.Node(niu.Merge(num_iters), name='collect_motion_params')

    workflow.connect([
        (inputnode, extract_dwis, [
            ('dwi_files', 'dwi_files'),
            ('bval_files', 'bval_files'),
            ('bvec_files', 'bvec_files'),
            ('initial_transforms', 'transforms'),
            ('b0_indices', 'b0_indices'),
        ]),
        (inputnode, b0_mean, [('warped_b0_images', 'b0_images')]),
        (extract_dwis, b0_based_bvec_transforms, [
            ('model_bvecs', 'bvec_files'),
            ('model_bvals', 'bval_files'),
            ('transforms', 'affine_transforms'),
        ]),
        (extract_dwis, b0_based_image_transforms, [
            ('model_dwi_files', 'input_image'),
            ('transforms', 'transforms'),
        ]),
        (b0_mean, b0_based_image_transforms, [('average_image', 'reference_image')]),

        # Connect the first iteration
        (extract_dwis, initial_model_iteration, [
            ('model_dwi_files', 'inputnode.original_dwi_files'),
            ('model_bvecs', 'inputnode.original_bvecs'),
            ('model_bvals', 'inputnode.bvals'),
        ]),
        (b0_based_image_transforms, initial_model_iteration, [
            ('output_image', 'inputnode.approx_aligned_dwi_files'),
        ]),
        (b0_based_bvec_transforms, initial_model_iteration, [
            ('bvecs', 'inputnode.approx_aligned_bvecs'),
        ]),
        (b0_mean, initial_model_iteration, [('average_image', 'inputnode.b0_mean')]),
        (inputnode, initial_model_iteration, [('warped_b0_mask', 'inputnode.b0_mask')]),
        (initial_model_iteration, collect_motion_params, [('outputnode.motion_params', 'in1')]),
    ])  # fmt:skip

    model_iterations = [initial_model_iteration]
    for iteration_num in range(num_iters - 1):
        iteration_name = 'shoreline_iteration%03d' % (iteration_num + 1)
        motion_key = 'in%d' % (iteration_num + 2)
        model_iterations.append(init_hmc_model_iteration_wf(name=iteration_name))
        workflow.connect([
            (model_iterations[-2], model_iterations[-1], [
                ('outputnode.aligned_dwis', 'inputnode.approx_aligned_dwi_files'),
                ('outputnode.aligned_bvecs', 'inputnode.approx_aligned_bvecs'),
            ]),
            (extract_dwis, model_iterations[-1], [
                ('model_dwi_files', 'inputnode.original_dwi_files'),
                ('model_bvals', 'inputnode.bvals'),
                ('model_bvecs', 'inputnode.original_bvecs'),
            ]),
            (b0_mean, model_iterations[-1], [('average_image', 'inputnode.b0_mean')]),
            (inputnode, model_iterations[-1], [('warped_b0_mask', 'inputnode.b0_mask')]),
            (model_iterations[-1], collect_motion_params, [
                ('outputnode.motion_params', motion_key),
            ]),
        ])  # fmt:skip

    # Return to the original, b0-interspersed ordering
    reorder_dwi_xforms = pe.Node(ReorderOutputs(), name='reorder_dwi_xforms')

    # Create a report:
    shoreline_report = pe.Node(SHORELineReport(), name='shoreline_report')
    ds_report_shoreline_gif = pe.Node(
        DerivativesDataSink(
            datatype='figures',
            desc='shoreline',
            suffix='animation',
        ),
        name='ds_report_shoreline_gif',
        mem_gb=1,
        run_without_submitting=True,
    )

    calculate_cnr = pe.Node(CalculateCNR(), name='calculate_cnr')

    if num_iters > 1:
        summarize_iterations = pe.Node(IterationSummary(), name='summarize_iterations')
        ds_report_iteration_plot = pe.Node(
            DerivativesDataSink(
                datatype='figures',
                desc='shorelineiters',
                suffix='dwi',
            ),
            name='ds_report_iteration_plot',
            mem_gb=0.1,
            run_without_submitting=True,
        )
        workflow.connect([
            (collect_motion_params, summarize_iterations, [('out', 'collected_motion_files')]),
            (summarize_iterations, ds_report_iteration_plot, [('plot_file', 'in_file')]),
            (summarize_iterations, outputnode, [('iteration_summary_file', 'optimization_data')]),
            (summarize_iterations, shoreline_report, [
                ('iteration_summary_file', 'iteration_summary'),
            ]),
        ])  # fmt:skip

    workflow.connect([
        (model_iterations[-1], reorder_dwi_xforms, [
            ('outputnode.hmc_transforms', 'model_based_transforms'),
            ('outputnode.predicted_dwis', 'model_predicted_images'),
            ('outputnode.aligned_dwis', 'warped_dwi_images'),
        ]),
        (b0_mean, reorder_dwi_xforms, [('average_image', 'b0_mean')]),
        (inputnode, reorder_dwi_xforms, [
            ('warped_b0_images', 'warped_b0_images'),
            ('b0_indices', 'b0_indices'),
            ('initial_transforms', 'initial_transforms'),
        ]),
        (reorder_dwi_xforms, outputnode, [
            ('hmc_warped_images', 'aligned_dwis'),
            ('full_transforms', 'hmc_transforms'),
            ('full_predicted_dwi_series', 'model_predicted_images'),
        ]),
        (inputnode, shoreline_report, [('dwi_files', 'original_images')]),
        (reorder_dwi_xforms, calculate_cnr, [
            ('hmc_warped_images', 'hmc_warped_images'),
            ('full_predicted_dwi_series', 'predicted_images'),
        ]),
        (inputnode, calculate_cnr, [('warped_b0_mask', 'mask_image')]),
        (calculate_cnr, outputnode, [('cnr_image', 'cnr_image')]),
        (reorder_dwi_xforms, shoreline_report, [
            ('full_predicted_dwi_series', 'model_predicted_images'),
            ('hmc_warped_images', 'registered_images'),
        ]),
        (shoreline_report, ds_report_shoreline_gif, [('plot_file', 'in_file')]),
    ])  # fmt:skip

    return workflow


def _list_squeeze(in_list):
    return [item[0] for item in in_list]
