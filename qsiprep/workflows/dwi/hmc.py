"""
Head motion correction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_dwi_hmc_wf
.. autofunction:: init_dwi_model_hmc_wf

"""
from ...engine import Workflow
import nipype.pipeline.engine as pe
from nipype.interfaces import ants, afni, utility as niu
import pandas as pd
from dipy.core.geometry import decompose_matrix
import os
import numpy as np
from ...interfaces.gradients import MatchTransforms, GradientRotation
from ...interfaces.dipy import IdealSignalRegistration
from .util import init_skullstrip_b0_wf


def init_dwi_hmc_wf(hmc_transform, hmc_model, hmc_align_to, mem_gb=3, omp_nthreads=1,
                    write_report=True, name="dwi_hmc_wf"):
    """Perform head motion correction on a dwi series."""
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['dwi_files', 'b0_indices', 'bvecs', 'bvals', 'b0_images']),
        name='inputnode')

    outputnode = pe.Node(
        niu.IdentityInterface(fields=["final_template", "forward_transforms", "noise_free_dwis"]),
        name='outputnode')

    workflow = Workflow(name=name)
    # Unbiased align the b0s
    b0_hmc_wf = init_b0_hmc_wf(align_to=hmc_align_to, transform=hmc_transform)
    # Tile the transforms so each non-b0 gets the transform from the nearest b0
    match_transforms = pe.Node(MatchTransforms(), name="match_transforms")

    workflow.connect([
        (inputnode, match_transforms, [('dwi_files', 'dwi_files'),
                                       ('b0_indices', 'b0_indices')]),
        (inputnode, b0_hmc_wf, [('b0_images', 'inputnode.b0_images')]),
        (b0_hmc_wf, outputnode, [('outputnode.final_template', 'final_template')]),
        (b0_hmc_wf, match_transforms, [(('outputnode.forward_transforms', _list_squeeze),
                                       'transforms')]),
    ])

    if hmc_model == 'none':
        # Motion correction based only on b0's
        workflow.connect([
            (match_transforms, outputnode, [('transforms', 'forward_transforms')])
        ])
        return workflow

    # Make a mask from the output template
    b0_template_mask = init_skullstrip_b0_wf(name="b0_template_mask")
    # Do model-based motion correction
    dwi_model_hmc_wf = init_dwi_model_hmc_wf(hmc_model, hmc_transform, mem_gb, omp_nthreads)

    workflow.connect([
        (b0_hmc_wf, b0_template_mask, [('outputnode.final_template', 'inputnode.in_file')]),
        (b0_template_mask, dwi_model_hmc_wf, [('outputnode.mask_file', 'inputnode.b0_mask')]),
        (inputnode, dwi_model_hmc_wf, ([('dwi_files', 'inputnode.dwi_files'),
                                        ('b0_indices', 'inputnode.b0_indices'),
                                        ('bvecs', 'inputnode.bvecs'),
                                        ('bvals', 'inputnode.bvals')])),
        (match_transforms, dwi_model_hmc_wf, [('transforms', 'inputnode.initial_transforms')]),
        (dwi_model_hmc_wf, outputnode, [('outputnode.noise_free_dwis', 'noise_free_dwis'),
                                        ('outputnode.hmc_transforms', 'forward_transforms')])
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
    reg.inputs.interpolation = 'HammingWindowedSinc'
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
            "motion_params"
        ]),
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

        # Attach to outputs
        # The last iteration aligned to the output from the second-to-last
        alignment_wf.connect(reg_iters[-2], "outputnode.updated_template",
                             outputnode, "final_template")
        alignment_wf.connect(reg_iters[-1], "outputnode.affine_transforms",
                             outputnode, "forward_transforms")
        alignment_wf.connect(iter_templates, "out", outputnode,
                             "iteration_templates")

    return alignment_wf


def get_model_reg(model_name,
                  transform="Rigid",
                  metric="Mattes",
                  iternum=0,
                  spatial_bias_correct=False,
                  precision="fine",
                  replace_outliers=False):
    """Make a model-based hmc iteration."""
    if precision == "coarse":
        extra_opts = dict(
            shrink_factors=[[4, 2]],
            smoothing_sigmas=[[3., 1.]],
            number_of_iterations=[[1000, 10000]],
            transform_parameters=[[0.3]])
    else:
        extra_opts = dict(
            shrink_factors=[[4, 2, 1]],
            smoothing_sigmas=[[3., 1., 0.]],
            number_of_iterations=[[1000, 10000, 10000]],
            transform_parameters=[[0.1]])
    model_reg = IdealSignalRegistration(
        model_name=model_name,
        metric=[metric],
        transforms=[transform],
        sigma_units=["vox"],
        sampling_strategy=['Random'],
        sampling_percentage=[0.25],
        radius_or_number_of_bins=[32],
        initial_moving_transform_com=0,
        interpolation='LanczosWindowedSinc',
        dimension=3,
        winsorize_lower_quantile=0.025,
        winsorize_upper_quantile=0.975,
        convergence_threshold=[1e-06],
        collapse_output_transforms=True,
        write_composite_transform=False,
        output_warped_image=True,
        **extra_opts)
    return model_reg


def init_dwi_model_hmc_wf(modelname, transform, mem_gb, omp_nthreads,
                          name='dwi_model_hmc_wf', metric="Mattes",
                          num_iters=2):
    """Create a model-based hmc workflow.

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
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['dwi_files', 'b0_indices', 'initial_transforms', 'b0_mask', 'bvecs', 'bvals']),
        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['hmc_transforms', 'noise_free_dwis']), name='outputnode')

    # Initialize with transform from nearest b0
    initial_transforms = pe.MapNode(ants.ApplyTransforms(interpolation="BSpline"),
                                    iterfield=['input_image', 'transforms'],
                                    name="initial_transforms")

    # Coarse iteration
    initial_reg = pe.Node(get_model_reg(modelname, transform=transform, metric=metric,
                                        precision='coarse'),
                          name='initial_reg')
    # Fine iteration
    final_reg = pe.Node(get_model_reg(modelname, transform=transform, metric=metric,
                                      precision='fine'),
                        name='final_reg')

    workflow.connect([
        (inputnode, initial_transforms, [('dwi_files', 'input_image'),
                                         ('initial_transforms', 'transforms'),
                                         ('b0_mask', 'reference_image')]),
        (initial_transforms, initial_reg, [('output_image', 'last_iter_images')]),
        (inputnode, initial_reg, [('initial_transforms', 'initial_transforms'),
                                  ('dwi_files', 'moving_image'),
                                  ('bvals', 'bvals'), ('bvecs', 'bvecs'),
                                  ('b0_indices', 'b0_indices'),
                                  ('b0_mask', 'mask_image')]),
        (inputnode, final_reg, [('initial_transforms', 'initial_transforms'),
                                ('dwi_files', 'moving_image'),
                                ('bvals', 'bvals'),
                                ('b0_indices', 'b0_indices'),
                                ('b0_mask', 'mask_image')]),
        (initial_reg, final_reg, [('corrected_images', 'last_iter_images'),
                                  ('rotated_bvecs', 'bvecs')])
    ])

    # Inverse warp the model images
    noise_free_dwis = pe.MapNode(ants.ApplyTransforms(interpolation="LanczosWindowedSinc",
                                                      invert_transform_flags=[True]),
                                 iterfield=['input_image', 'reference_image', 'transforms'],
                                 name='noise_free_dwis')

    workflow.connect([
        (final_reg, noise_free_dwis, [('ideal_images', 'input_image'),
                                      ('transforms', 'transforms')]),
        (inputnode, noise_free_dwis, [('dwi_files', 'reference_image')]),
        (noise_free_dwis, outputnode, [('output_image', 'noise_free_dwis')]),
        (final_reg, outputnode, [('transforms', 'hmc_transforms')]),
    ])
    return workflow


def _list_squeeze(in_list):
    return [item[0] for item in in_list]
