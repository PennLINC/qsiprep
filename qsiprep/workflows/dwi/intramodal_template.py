"""
Head motion correction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_dwi_hmc_wf
.. autofunction:: init_dwi_model_hmc_wf

"""
import nipype.pipeline.engine as pe
from pkg_resources import resource_filename as pkgrf
from nipype.interfaces import ants, afni, utility as niu
from ...engine import Workflow
from ...interfaces.gradients import MatchTransforms, GradientRotation, CombineMotions
from ...interfaces.shoreline import (SignalPrediction, ExtractDWIsForModel, ReorderOutputs,
                                     B0Mean, SHORELineReport, IterationSummary, CalculateCNR)
from ...interfaces import DerivativesDataSink
from .util import init_skullstrip_b0_wf
from .hmc import init_b0_hmc_wf

DEFAULT_MEMORY_MIN_GB = 0.01


def init_intramodal_template_wf(inputs_list, transform="Rigid", num_iterations=2, mem_gb=3,
                                omp_nthreads=1, name="intramodal_template_wf"):
    """Create an unbiased intramodal template for a subject. This aligns the b=0 references
    from all the scans of a subject. Can be rigid, affine or nonlinear (BSplineSyN).

    **Parameters**
        inputs_list: list of inputs
            List if identifiers for the input b=0 images.
        transform: 'Rigid', 'Affine', 'BSplineSyN'
            Which transform to ultimately use. If 'BSplineSyN', first 2 iterations of Affine will
            be run.
        num_iterations: int
            Default: 2.

    **Inputs**

        [workflow_name]_image...
            One input for each input image. There is no input called inputs_list
        t1w_image

    **Outputs**
        [workflow_name]_transform
            transform files to the intramodal template

        intramodal_template_to_t1w_transform
            Transform from the b0

    """
    workflow = Workflow(name=name)
    input_names = [name + '_b0_template' for name in inputs_list]
    output_names = [name + '_transform' for name in inputs_list]

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=input_names + ['t1w_brain']),
        name='inputnode')
    merge_inputs = pe.Node(niu.Merge(len(input_names)), name='merge_inputs')
    for input_num, input_name in enumerate(input_names):
        workflow.connect(inputnode, input_name, merge_inputs, 'in%d' % (input_num + 1))

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=output_names + ["intramodal_template",
                                   "intramodal_template_mask",
                                   "intramodal_template_to_t1w_transform"]),
        name='outputnode')
    split_outputs = pe.Node(niu.Split(splits=[1] * len(input_names), squeeze=True),
                            name='split_outputs')
    for output_num, output_name in enumerate(output_names):
        workflow.connect(split_outputs, 'out%d' % (output_num + 1), outputnode, output_name)

    # Should we add nonlinear iterations?
    do_nonlinear = transform not in ('Rigid', 'Affine')

    # Align the b=0 images from all runs (Linear)
    initial_transform = 'Affine' if do_nonlinear else transform
    intramodal_b0_affine_template = init_b0_hmc_wf(
        align_to='iterative',
        transform=initial_transform,
        spatial_bias_correct=True,
        name='intramodal_b0_affine_template')

    intramodal_template_mask = init_skullstrip_b0_wf(name="intramodal_template_mask")

    workflow.connect([
        (merge_inputs, intramodal_b0_affine_template, [('out', 'inputnode.b0_images')]),
        (intramodal_template_mask, outputnode, [
            ('outputnode.mask_file', 'intramodal_template_mask')])
    ])
    if not do_nonlinear:
        workflow.connect([
            (intramodal_b0_affine_template, intramodal_template_mask, [
                ('outputnode.final_template', 'inputnode.in_file')]),
            (intramodal_b0_affine_template, split_outputs, [
                (('outputnode.forward_transforms', _list_squeeze), 'inlist')])
        ])
    else:
        nonlinear_alignment_wf = init_nonlinear_alignment_wf(num_iters=num_iterations)
        workflow.connect([
            (merge_inputs, nonlinear_alignment_wf, [('out', 'inputnode.images')]),
            (nonlinear_alignment_wf, intramodal_template_mask, [
                ('outputnode.final_template', 'inputnode.in_file')]),
            (intramodal_b0_affine_template, nonlinear_alignment_wf, [
                ('outputnode.final_template', 'inputnode.initial_template')]),
            (nonlinear_alignment_wf, split_outputs, [
                ('outputnode.forward_transforms', 'inlist')])
        ])

    return workflow


def nonlinear_alignment_iteration(iternum=0, gradient_step=0.1):
    """
    Takes a template image and a set of input images, does
    a linear alignment to the template and updates it with the
    inverse of the average affine transform to the new template

    Returns a workflow

    """
    iteration_wf = pe.Workflow(name="nl_iterative_alignment_%03d" % iternum)
    input_node_fields = ["image_paths", "template_image", "iteration_num"]
    inputnode = pe.Node(
        niu.IdentityInterface(fields=input_node_fields), name='inputnode')
    inputnode.inputs.iteration_num = iternum
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["registered_image_paths", "affine_transforms",
                                      "warp_transforms", "composite_transforms",
                                      "updated_template"]), name='outputnode')
    ants_settings = pkgrf("qsiprep", "data/intramodal_nonlinear.json")
    reg = ants.Registration(from_file=ants_settings)
    iter_reg = pe.MapNode(
        reg, name="reg_%03d" % iternum, iterfield=["moving_image"])

    # Average the images
    averaged_images = pe.Node(
        ants.AverageImages(normalize=True, dimension=3),
        name="averaged_images")

    # Shape update to template:
    # Average the affines so that the inverse can be applied to the template
    affines_to_list = pe.Node(niu.Merge(1), name="affines_to_list")
    warps_to_list = pe.Node(niu.Merge(1), name="warps_to_list")
    avg_affines = pe.Node(
        ants.AverageAffineTransform(dimension=3,
                                    output_affine_transform="AveragedAffines.mat"),
        name="avg_affines")

    # Average the warps:
    average_warps = pe.Node(
        ants.AverageImages(dimension=3, normalize=False), name="average_warps")
    # Scale by the gradient step
    scale_warp = pe.Node(
        ants.MultiplyImages(dimension=3, second_input=gradient_step,
                            output_product_image="scaled_warp.nii.gz"),
        name="scale_warp")
    # Align the warps to the template image
    align_warp = pe.Node(
        ants.ApplyTransforms(
            input_image_type=1, invert_transform_flags=[True]),
        name="align_warp")

    # transform the template for the shape update
    shape_update_template = pe.Node(
        ants.ApplyTransforms(interpolation="LanczosWindowedSinc",
                             invert_transform_flags=[True, False, False, False, False]),
        name="shape_update_template")
    shape_update_merge = pe.Node(niu.Merge(5), name="shape_update_merge")

    # Run the images through antsRegistration
    def get_first(input_pairs):
        return [input_pair[0] for input_pair in input_pairs]

    def get_second(input_pairs):
        return [input_pair[1] for input_pair in input_pairs]

    iteration_wf.connect([
        (inputnode, iter_reg, [
            ('image_paths', 'moving_image'),
            ('template_image', 'fixed_image')]),
        (iter_reg, affines_to_list, [(('forward_transforms', get_first), 'in1')]),
        (affines_to_list, avg_affines, [('out', 'transforms')]),
        (iter_reg, warps_to_list, [(('forward_transforms', get_second), 'in1')]),
        (iter_reg, averaged_images, [('warped_image', 'images')]),

        # Average the warps, scale them, and transform to be aligned with the template
        (warps_to_list, average_warps, [('out', 'images')]),
        (average_warps, scale_warp, [('output_average_image', 'first_input')]),
        (scale_warp, align_warp, [
            ('output_product_image', 'input_image')]),
        (avg_affines, align_warp, [('affine_transform', 'transforms')]),
        (inputnode, align_warp, [('template_image', 'reference_image')]),
        (avg_affines, shape_update_merge, [('affine_transform', 'in1')]),
        (align_warp, shape_update_merge, [
            ('output_image', 'in2'), ('output_image', 'in3'),
            ('output_image', 'in4'), ('output_image', 'in5')]),
        (shape_update_merge, shape_update_template, [('out', 'transforms')]),
        (averaged_images, shape_update_template, [
            ('output_average_image', 'input_image'),
            ('output_average_image', 'reference_image')]),
        (shape_update_template, outputnode, [('output_image', 'updated_template')]),
        (iter_reg, outputnode, [
            ('forward_transforms', 'affine_transforms'),
            ('warped_image', 'registered_image_paths')])
    ])

    return iteration_wf


def init_nonlinear_alignment_wf(transform="BSplineSyN", metric="CC",
                                num_iters=2, name="nonlinear_alignment_wf"):
    """Creates a workflow that does nonlinear template creation."""
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['images', 'initial_template']), name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(fields=[
            "final_template", "forward_transforms", "iteration_templates",
            "motion_params", "aligned_images"]),
        name='outputnode')

    # Save the iteration templates
    iter_templates = pe.Node(
        niu.Merge(num_iters), name="iteration_templates")

    initial_reg = nonlinear_alignment_iteration(iternum=0)

    workflow.connect([
        (inputnode, iter_templates, [('initial_template', 'in1')]),
        (inputnode, initial_reg, [
            ('initial_template', 'inputnode.template_image'),
            ('images', 'inputnode.image_paths')])])

    reg_iters = [initial_reg]
    for iternum in range(1, num_iters):
        reg_iters.append(nonlinear_alignment_iteration(iternum=iternum))
        workflow.connect([
            (reg_iters[-2], reg_iters[-1], [
                ('outputnode.updated_template', 'inputnode.template_image')]),
            (inputnode, reg_iters[-1], [('images', 'inputnode.image_paths')]),
            (reg_iters[-1], iter_templates, [
                ("outputnode.updated_template", "in%d" % (iternum + 1))])
        ])

    # Attach to outputs
    # The last iteration aligned to the output from the second-to-last
    workflow.connect([
        (reg_iters[-2], outputnode, [
            ('outputnode.updated_template', 'final_template')]),
        (reg_iters[-1], outputnode, [
            ('outputnode.affine_transforms', 'forward_transforms'),
            ('outputnode.registered_image_paths', 'aligned_images')]),
        (iter_templates, outputnode, [('out', 'iteration_templates')])
    ])

    return workflow


def _list_squeeze(in_list):
    return [item[0] for item in in_list]
