"""
Head motion correction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_dwi_hmc_wf
.. autofunction:: init_dwi_model_hmc_wf

"""
import nipype.pipeline.engine as pe
from pkg_resources import resource_filename as pkgrf
from ...engine import Workflow
from nipype.interfaces import ants, afni, utility as niu
import pandas as pd
from dipy.core.geometry import decompose_matrix
import os
import numpy as np
from ...interfaces.gradients import MatchTransforms, GradientRotation, CombineMotions
from ...interfaces.shoreline import (SignalPrediction, ExtractDWIsForModel, ReorderOutputs,
                                     B0Mean, SHORELineReport, IterationSummary)
from ...interfaces import DerivativesDataSink
from .util import init_skullstrip_b0_wf

DEFAULT_MEMORY_MIN_GB = 0.01

def init_dwi_hmc_wf(hmc_transform, hmc_model, hmc_align_to, mem_gb=3, omp_nthreads=1,
                    num_model_iterations=2, write_report=True, name="dwi_hmc_wf"):
    """Perform head motion correction on a dwi series."""
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['dwi_files', 'b0_indices', 'bvecs', 'bvals', 'b0_images']),
        name='inputnode')

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=["final_template", "forward_transforms", "noise_free_dwis"]),
        name='outputnode')

    workflow = Workflow(name=name)
    # Unbiased align the b0s
    b0_hmc_wf = init_b0_hmc_wf(align_to=hmc_align_to, transform=hmc_transform)
    # Tile the transforms so each non-b0 gets the transform from the nearest b0
    match_transforms = pe.Node(MatchTransforms(), name="match_transforms")
    # Make a mask from the output template. It is bias-corrected, so good for masking
    # but bad for using as a b=0 for modeling.
    b0_template_mask = init_skullstrip_b0_wf(name="b0_template_mask")

    workflow.connect([
        (inputnode, match_transforms, [('dwi_files', 'dwi_files'),
                                       ('b0_indices', 'b0_indices')]),
        (inputnode, b0_hmc_wf, [('b0_images', 'inputnode.b0_images')]),
        (b0_hmc_wf, outputnode, [('outputnode.final_template', 'final_template')]),
        (b0_hmc_wf, match_transforms, [(('outputnode.forward_transforms', _list_squeeze),
                                        'transforms')]),
        (b0_hmc_wf, b0_template_mask, [('outputnode.final_template', 'inputnode.in_file')])
    ])

    # Do model-based motion correction
    dwi_model_hmc_wf = init_dwi_model_hmc_wf(hmc_model, hmc_transform, mem_gb, omp_nthreads,
                                             num_iters=num_model_iterations)

    workflow.connect([
        (b0_hmc_wf, dwi_model_hmc_wf, [
            ('outputnode.aligned_images', 'inputnode.warped_b0_images')]),
        (b0_template_mask, dwi_model_hmc_wf, [
            ('outputnode.mask_file', 'inputnode.warped_b0_mask')]),
        (inputnode, dwi_model_hmc_wf, [
            ('dwi_files', 'inputnode.dwi_files'),
            ('b0_indices', 'inputnode.b0_indices'),
            ('bvecs', 'inputnode.bvec_files'),
            ('bvals', 'inputnode.bval_files')]),
        (match_transforms, dwi_model_hmc_wf, [('transforms', 'inputnode.initial_transforms')])
    ])

    # Warp the modeled images into non-motion-corrected space
    uncorrect_model_images = pe.MapNode(
        ants.ApplyTransforms(invert_transform_flags=[True],
                             interpolation='LanczosWindowedSinc'),
        iterfield=['input_image', 'reference_image', 'transforms'],
        name='uncorrect_model_images')
    workflow.connect([
        (dwi_model_hmc_wf, uncorrect_model_images, [
            ('outputnode.model_predicted_images', 'input_image'),
            ('outputnode.hmc_transforms', 'transforms')]),
        (inputnode, uncorrect_model_images, [('dwi_files', 'reference_image')]),
        (uncorrect_model_images, outputnode, [('output_image', 'noise_free_dwis')])
    ])

    return workflow


def linear_alignment_workflow(transform="Rigid",
                              metric="Mattes",
                              iternum=0,
                              spatial_bias_correct=False,
                              precision="fine"):
    """
    Takes a template image and a set of input images, does
    a linear alignment to the template and updates it with the
    inverse of the average affine transform to the new template

    Returns a workflow

    """
    iteration_wf = pe.Workflow(name="iterative_alignment_%03d" % iternum)
    input_node_fields = ["image_paths", "template_image", "iteration_num"]
    inputnode = pe.Node(
        niu.IdentityInterface(fields=input_node_fields), name='inputnode')
    inputnode.inputs.iteration_num = iternum
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["registered_image_paths", "affine_transforms",
                                      "updated_template"]), name='outputnode')

    reg = ants.Registration()
    reg.inputs.metric = [metric]
    reg.inputs.transforms = [transform]
    reg.inputs.sigma_units = ["vox"]
    reg.inputs.sampling_strategy = ['Random']
    reg.inputs.sampling_percentage = [0.25]
    reg.inputs.radius_or_number_of_bins = [32]
    reg.inputs.initial_moving_transform_com = 0
    reg.inputs.interpolation = 'LanczosWindowedSinc'
    reg.inputs.dimension = 3
    reg.inputs.winsorize_lower_quantile = 0.025
    reg.inputs.winsorize_upper_quantile = 0.975
    reg.inputs.convergence_threshold = [1e-06]
    reg.inputs.collapse_output_transforms = True
    reg.inputs.write_composite_transform = False
    reg.inputs.output_warped_image = True
    if precision == "coarse":
        reg.inputs.shrink_factors = [[4, 2]]
        reg.inputs.smoothing_sigmas = [[3., 1.]]
        reg.inputs.number_of_iterations = [[1000, 10000]]
        reg.inputs.transform_parameters = [[0.3]]
    else:
        reg.inputs.shrink_factors = [[4, 2, 1]]
        reg.inputs.smoothing_sigmas = [[3., 1., 0.]]
        reg.inputs.number_of_iterations = [[1000, 10000, 10000]]
        reg.inputs.transform_parameters = [[0.1]]
    iter_reg = pe.MapNode(
        reg, name="reg_%03d" % iternum, iterfield=["moving_image"])

    # Run the images through antsRegistration
    iteration_wf.connect(inputnode, "image_paths", iter_reg, "moving_image")
    iteration_wf.connect(inputnode, "template_image", iter_reg, "fixed_image")

    # Average the images
    averaged_images = pe.Node(
        ants.AverageImages(normalize=True, dimension=3),
        name="averaged_images")
    iteration_wf.connect(iter_reg, "warped_image", averaged_images, "images")

    # Apply the inverse to the average image
    transforms_to_list = pe.Node(niu.Merge(1), name="transforms_to_list")
    transforms_to_list.inputs.ravel_inputs = True
    iteration_wf.connect(iter_reg, "forward_transforms", transforms_to_list,
                         "in1")
    avg_affines = pe.Node(ants.AverageAffineTransform(), name="avg_affine")
    avg_affines.inputs.dimension = 3
    avg_affines.inputs.output_affine_transform = "AveragedAffines.mat"
    iteration_wf.connect(transforms_to_list, "out", avg_affines, "transforms")

    invert_average = pe.Node(ants.ApplyTransforms(), name="invert_average")
    invert_average.inputs.interpolation = "HammingWindowedSinc"
    invert_average.inputs.invert_transform_flags = [True]

    avg_to_list = pe.Node(niu.Merge(1), name="to_list")
    iteration_wf.connect(avg_affines, "affine_transform", avg_to_list, "in1")
    iteration_wf.connect(avg_to_list, "out", invert_average, "transforms")
    iteration_wf.connect(averaged_images, "output_average_image",
                         invert_average, "input_image")
    iteration_wf.connect(averaged_images, "output_average_image",
                         invert_average, "reference_image")
    iteration_wf.connect(invert_average, "output_image", outputnode,
                         "updated_template")
    iteration_wf.connect(iter_reg, "forward_transforms", outputnode,
                         "affine_transforms")
    iteration_wf.connect(iter_reg, "warped_image", outputnode,
                         "registered_image_paths")

    return iteration_wf


def init_b0_hmc_wf(align_to="iterative", transform="Rigid", spatial_bias_correct=False,
                   metric="Mattes", num_iters=3, name="b0_hmc_wf"):

    if align_to == "iterative" and num_iters < 2:
        raise ValueError("Must specify a positive number of iterations")

    alignment_wf = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['b0_images']), name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(fields=[
            "final_template", "forward_transforms", "iteration_templates",
            "motion_params", "aligned_images"]),
        name='outputnode')

    # Iteratively create a template
    if align_to == "iterative":
        initial_template = pe.Node(
            ants.AverageImages(normalize=True, dimension=3),
            name="initial_template")
        alignment_wf.connect(inputnode, "b0_images", initial_template,
                             "images")
        # Store the registration targets
        iter_templates = pe.Node(
            niu.Merge(num_iters), name="iteration_templates")
        alignment_wf.connect(initial_template, "output_average_image",
                             iter_templates, "in1")

        initial_reg = linear_alignment_workflow(
            transform=transform,
            metric=metric,
            spatial_bias_correct=spatial_bias_correct,
            precision="coarse",
            iternum=0)
        alignment_wf.connect(initial_template, "output_average_image",
                             initial_reg, "inputnode.template_image")
        alignment_wf.connect(inputnode, "b0_images", initial_reg,
                             "inputnode.image_paths")
        reg_iters = [initial_reg]
        for iternum in range(1, num_iters):
            reg_iters.append(
                linear_alignment_workflow(
                    transform=transform,
                    metric=metric,
                    spatial_bias_correct=spatial_bias_correct,
                    precision="fine",
                    iternum=iternum))
            alignment_wf.connect(reg_iters[-2], "outputnode.updated_template",
                                 reg_iters[-1], "inputnode.template_image")
            alignment_wf.connect(inputnode, "b0_images", reg_iters[-1],
                                 "inputnode.image_paths")
            alignment_wf.connect(reg_iters[-1], "outputnode.updated_template",
                                 iter_templates, "in%d" % (iternum + 1))
        #
        # Attach to outputs
        # The last iteration aligned to the output from the second-to-last
        alignment_wf.connect(reg_iters[-2], "outputnode.updated_template",
                             outputnode, "final_template")
        alignment_wf.connect(reg_iters[-1], "outputnode.affine_transforms",
                             outputnode, "forward_transforms")
        alignment_wf.connect(reg_iters[-1], "outputnode.registered_image_paths",
                             outputnode, "aligned_images")
        alignment_wf.connect(iter_templates, "out", outputnode,
                             "iteration_templates")

    return alignment_wf


def _bvecs_to_list(bvec_file):
    import numpy as np
    bvec = np.loadtxt(bvec_file).T
    return list(bvec)


def _bvals_to_floats(bval_files):
    import numpy as np
    return [float(np.loadtxt(bval_file)) for bval_file in bval_files]


def init_hmc_model_iteration_wf(modelname, transform, precision="coarse", name="hmc_model_iter0"):
    """Create a model-based hmc registration iteration workflow.

    This workflow takes an initial set of transforms, applies them to the
    original data, and builds a signal model out of the transformed images.

    The original images are then registered to the model-generated target
    images. Included in the outputs are the transforms and optionally the
    noise-free images in original (non-aligned) space.

    .. workflow::
        :graph2use: colored
        :simple_form: yes

        from qsiprep.workflows.dwi.hmc import init_dwi_model_hmc_wf
        wf = init_dwi_model_hmc_wf(modelname='3dSHORE',
                                   transform='Affine',
                                   num_iters=2,
                                   mem_gb=3,
                                   omp_nthreads=1)

    **Parameters**

        modelname : str
            one of the models for reconstructing an EAP and producing
            signal estimates used for motion correction
        transform : str
            either "Rigid" or "Affine". Choosing "Affine" may help with Eddy warping
        calculate_noise_free : bool
            Should the noise-free images (in original space) be calculated and saved?
        precision : str
            Use fast "coarse" alignment or accurate "precise" registration
        name : str
            name of the workflow

    **Inputs**

        original_dwi_files
            list of 3d dwi files, no b0's
        original_bvecs
            list of bvec files corresponding to `original_dwi_files`
        original_bvals
            list of bval files corresponding to `original_dwi_files`
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
            fields=['original_dwi_files', 'original_bvecs',
                    'approx_aligned_dwi_files', 'approx_aligned_bvecs',
                    'b0_mask', 'b0_mean', 'bvals']),
        name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['hmc_transforms', 'aligned_dwis', 'aligned_bvecs', 'predicted_dwis',
                    'motion_params']),
        name='outputnode')

    ants_settings = pkgrf("qsiprep", "data/shoreline_coarse.json")
    predict_dwis = pe.MapNode(SignalPrediction(),
                              iterfield=['bval_to_predict', 'bvec_to_predict'],
                              name="predict_dwis")
    predict_dwis.synchronize = True

    # Register original images to the predicted images
    register_to_predicted = pe.MapNode(ants.Registration(from_file=ants_settings),
                                       iterfield=['fixed_image', 'moving_image'],
                                       name='register_to_predicted')
    register_to_predicted.synchronize = True

    # Apply new transforms to bvecs
    post_bvec_transforms = pe.Node(GradientRotation(), name="post_bvec_transforms")

    # Summarize the motion
    calculate_motion = pe.Node(CombineMotions(), name="calculate_motion")

    workflow.connect([
        # Send inputs to DWI prediction
        (inputnode, predict_dwis, [('approx_aligned_dwi_files', 'aligned_dwis'),
                                   ('approx_aligned_bvecs', 'aligned_bvecs'),
                                   ('bvals', 'bvals'),
                                   ('b0_mean', 'aligned_b0_mean'),
                                   ('b0_mask', 'aligned_mask'),
                                   (('approx_aligned_bvecs', _bvecs_to_list), 'bvec_to_predict'),
                                   (('bvals', _bvals_to_floats), 'bval_to_predict')]),
        (predict_dwis, register_to_predicted, [('predicted_image', 'fixed_image')]),
        (inputnode, register_to_predicted, [('original_dwi_files', 'moving_image')]),

        (register_to_predicted, calculate_motion, [
            (('forward_transforms', _list_squeeze), 'transform_files')]),
        (calculate_motion, outputnode, [('motion_file', 'motion_params')]),

        (register_to_predicted, post_bvec_transforms, [
            (('forward_transforms', _list_squeeze), 'affine_transforms')]),
        (inputnode, post_bvec_transforms, [('original_bvecs', 'bvec_files'),
                                           ('bvals', 'bval_files')]),

        (predict_dwis, outputnode, [('predicted_image', 'predicted_dwis')]),
        (post_bvec_transforms, outputnode, [('bvecs', 'aligned_bvecs')]),
        (register_to_predicted, outputnode, [('warped_image', 'aligned_dwis'),
                                             ('forward_transforms', 'hmc_transforms')])
    ])

    return workflow


def init_dwi_model_hmc_wf(modelname, transform, mem_gb, omp_nthreads,
                          num_iters=2, name='dwi_model_hmc_wf', metric="Mattes"):
    """Create a model-based hmc workflow.

    .. workflow::
        :graph2use: colored
        :simple_form: yes

        from qsiprep.workflows.dwi.hmc import init_dwi_model_hmc_wf
        wf = init_dwi_model_hmc_wf(modelname='3dSHORE',
                                   transform='Affine',
                                   calculate_noise_free=True,
                                   num_iters=2,
                                   mem_gb=3,
                                   omp_nthreads=1)

    **Parameters**

        modelname : str
            one of the models for reconstructing an EAP and producing
            signal estimates used for motion correction
        transform : str
            either "Rigid" or "Affine". Choosing "Affine" may help with Eddy warping
        num_iters : int
            the number of times the model will be updated with transformed data

    **Inputs**

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

    **Outputs**

        hmc_transforms
            list of transforms, one per file in `dwi_files`
        hmc_confounds
            file containing motion and qc parameters from hmc

    """
    workflow = Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['dwi_files', 'b0_indices', 'initial_transforms', 'bvec_files', 'bval_files',
                    'warped_b0_images', 'warped_b0_mask']),
        name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['hmc_transforms', 'model_predicted_images', 'optimization_data']),
        name='outputnode')

    # Merge b0s into a single volume, put the non-b0 dwis into a list
    extract_dwis = pe.Node(ExtractDWIsForModel(), name="extract_dwis")

    # Initialize with the transforms provided
    b0_based_image_transforms = pe.MapNode(ants.ApplyTransforms(interpolation="BSpline"),
                                           iterfield=['input_image', 'transforms'],
                                           name="b0_based_image_transforms")
    # Rotate the original bvecs as well
    b0_based_bvec_transforms = pe.Node(GradientRotation(), name="b0_based_bvec_transforms")

    # Create a mask and an average from the aligned b0 images
    b0_mean = pe.Node(B0Mean(), name='b0_mean')

    # Start building and connecting the model iterations
    initial_model_iteration = init_hmc_model_iteration_wf(
        modelname, transform, precision="coarse", name="initial_model_iteration")

    # Collect motion estimates across iterations
    collect_motion_params = pe.Node(niu.Merge(num_iters), name="collect_motion_params")

    workflow.connect([
        (inputnode, extract_dwis, [('dwi_files', 'dwi_files'),
                                   ('bval_files', 'bval_files'),
                                   ('bvec_files', 'bvec_files'),
                                   ('initial_transforms', 'transforms'),
                                   ('b0_indices', 'b0_indices')]),
        (inputnode, b0_mean, [('warped_b0_images', 'b0_images')]),
        (extract_dwis, b0_based_bvec_transforms, [('model_bvecs', 'bvec_files'),
                                                  ('model_bvals', 'bval_files'),
                                                  ('transforms', 'affine_transforms')]),
        (extract_dwis, b0_based_image_transforms, [('model_dwi_files', 'input_image'),
                                                   ('transforms', 'transforms')]),
        (b0_mean, b0_based_image_transforms, [('average_image', 'reference_image')]),

        # Connect the first iteration
        (extract_dwis, initial_model_iteration, [
            ('model_dwi_files', 'inputnode.original_dwi_files'),
            ('model_bvecs', 'inputnode.original_bvecs'),
            ('model_bvals', 'inputnode.bvals')]),
        (b0_based_image_transforms, initial_model_iteration, [
            ('output_image', 'inputnode.approx_aligned_dwi_files')]),
        (b0_based_bvec_transforms, initial_model_iteration, [
            ('bvecs', 'inputnode.approx_aligned_bvecs')]),
        (b0_mean, initial_model_iteration, [
            ('average_image', 'inputnode.b0_mean')]),
        (inputnode, initial_model_iteration, [
            ('warped_b0_mask', 'inputnode.b0_mask')]),
        (initial_model_iteration, collect_motion_params, [
            ('outputnode.motion_params', 'in1')])
    ])

    model_iterations = [initial_model_iteration]
    for iteration_num in range(num_iters-1):
        iteration_name = 'shoreline_iteration%03d' % (iteration_num + 1)
        motion_key = 'in%d' % (iteration_num + 2)
        model_iterations.append(
            init_hmc_model_iteration_wf(modelname=modelname, transform=transform,
                                        precision="precise", name=iteration_name)
        )
        workflow.connect([
            (model_iterations[-2], model_iterations[-1], [
                ('outputnode.aligned_dwis', 'inputnode.approx_aligned_dwi_files'),
                ('outputnode.aligned_bvecs', 'inputnode.approx_aligned_bvecs')]),
            (extract_dwis, model_iterations[-1], [
                ('model_dwi_files', 'inputnode.original_dwi_files'),
                ('model_bvecs', 'inputnode.original_bvecs'),
                ('model_bvals', 'inputnode.bvals')]),
            (b0_mean, model_iterations[-1], [
                ('average_image', 'inputnode.b0_mean')]),
            (inputnode, model_iterations[-1], [
                ('warped_b0_mask', 'inputnode.b0_mask')]),
            (model_iterations[-1], collect_motion_params, [
                ('outputnode.motion_params', motion_key)])
        ])

    reorder_dwi_xforms = pe.Node(ReorderOutputs(), name='reorder_dwi_xforms')
    summarize_iterations = pe.Node(IterationSummary(), name='summarize_iterations')
    ds_report_iteration_plot = pe.Node(
        DerivativesDataSink(suffix="shoreline_summary"), name='ds_report_iteration_plot',
        mem_gb=0.1, run_without_submitting=True)

    # Create a report:
    shoreline_report = pe.Node(SHORELineReport(), name='shoreline_report')
    ds_report_shoreline_gif = pe.Node(
        DerivativesDataSink(suffix="shoreline_animation"), name='ds_report_shoreline_gif',
        mem_gb=1, run_without_submitting=True)

    workflow.connect([
        (model_iterations[-1], reorder_dwi_xforms, [
            ('outputnode.hmc_transforms', 'model_based_transforms'),
            ('outputnode.predicted_dwis', 'model_predicted_images'),
            ('outputnode.aligned_dwis', 'warped_dwi_images')]),
        (b0_mean, reorder_dwi_xforms, [('average_image', 'b0_mean')]),
        (inputnode, reorder_dwi_xforms, [
            ('warped_b0_images', 'warped_b0_images'),
            ('b0_indices', 'b0_indices'),
            ('initial_transforms', 'initial_transforms')]),
        (reorder_dwi_xforms, outputnode, [
            ('full_transforms', 'hmc_transforms'),
            ('full_predicted_dwi_series', 'model_predicted_images')]),
        (collect_motion_params, summarize_iterations, [
            ('out', 'collected_motion_files')]),
        (summarize_iterations, ds_report_iteration_plot, [
            ('plot_file', 'in_file')]),
        (summarize_iterations, shoreline_report, [
            ('iteration_summary_file', 'iteration_summary')]),
        (inputnode, shoreline_report, [('dwi_files', 'original_images')]),
        (reorder_dwi_xforms, shoreline_report, [
            ('full_predicted_dwi_series', 'model_predicted_images'),
            ('hmc_warped_images', 'registered_images')]),
        (shoreline_report, ds_report_shoreline_gif, [('out_report', 'in_file')])
    ])

    return workflow


def _list_squeeze(in_list):
    return [item[0] for item in in_list]
