"""
Head motion correction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_dwi_hmc_wf
.. autofunction:: init_dwi_model_hmc_wf

"""

import nipype.pipeline.engine as pe
from nipype.interfaces import ants
from nipype.interfaces import utility as niu
from pkg_resources import resource_filename as pkgrf

from ...engine import Workflow
from ...interfaces import DerivativesDataSink
from ...interfaces.ants import MultivariateTemplateConstruction2
from .registration import init_b0_to_anat_registration_wf

DEFAULT_MEMORY_MIN_GB = 0.01


def init_intramodal_template_wf(
    inputs_list,
    t1w_source_file,
    reportlets_dir,
    transform="Rigid",
    num_iterations=2,
    mem_gb=3,
    omp_nthreads=1,
    name="intramodal_template_wf",
):
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
    input_names = [name.replace("-", "_") + "_b0_template" for name in inputs_list]
    output_names = [name.replace("-", "_") + "_transform" for name in inputs_list]

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=input_names
            + [
                "t1_brain",
                "t1_preproc",
                "t1_mask",
                "t1_seg",
                "subjects_dir",
                "subject_id",
                "t1_aseg",
                "t1_aparc",
                "t1_tpms",
                "t1_2_mni_forward_transform",
                "dwi_sampling_grid",
                "t1_2_fsnative_forward_transform",
                "t1_2_fsnative_reverse_transform",
                "t1_2_mni_reverse_transform",
            ]
        ),
        name="inputnode",
    )
    merge_inputs = pe.Node(niu.Merge(len(input_names)), name="merge_inputs")
    rename_inputs = pe.MapNode(
        niu.Rename(keep_ext=True), iterfield=["in_file", "format_string"], name="rename_inputs"
    )
    rename_inputs.inputs.format_string = input_names
    rename_inputs.synchronize = True
    for input_num, input_name in enumerate(input_names):
        workflow.connect(inputnode, input_name, merge_inputs, "in%d" % (input_num + 1))

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=output_names
            + [
                "intramodal_template",
                "intramodal_template_to_t1_affine",
                "intramodal_template_to_t1_warp",
            ]
        ),
        name="outputnode",
    )
    split_outputs = pe.Node(
        niu.Split(splits=[1] * len(input_names), squeeze=True), name="split_outputs"
    )
    for output_num, output_name in enumerate(output_names):
        workflow.connect(split_outputs, 'out%d' % (output_num + 1),
                         outputnode, output_name)  # fmt:skip

    runtime_opts = {"num_cores": 1, "parallel_control": 0}
    if omp_nthreads > 1:
        runtime_opts = {"num_cores": omp_nthreads, "parallel_control": 2}
    ants_mvtc2 = pe.Node(
        MultivariateTemplateConstruction2(
            dimension=3, iteration_limit=num_iterations, **runtime_opts
        ),
        name="ants_mvtc2",
    )

    workflow.connect([
        (merge_inputs, rename_inputs, [('out', 'in_file')]),
        (rename_inputs, ants_mvtc2, [('out_file', 'input_images')]),
        (ants_mvtc2, split_outputs, [
            ('forward_transforms', 'inlist')]),
        (ants_mvtc2, outputnode, [
            ('templates', 'intramodal_template')])
    ])  # fmt:skip

    # calculate dwi registration to T1w
    b0_coreg_wf = init_b0_to_anat_registration_wf(
        omp_nthreads=omp_nthreads, mem_gb=mem_gb, write_report=True
    )
    ds_report_imtcoreg = pe.Node(
        DerivativesDataSink(suffix="imtcoreg", source_file=t1w_source_file),
        name="ds_report_imtcoreg",
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    workflow.connect([
        (inputnode, b0_coreg_wf, [
            ('t1_brain', 'inputnode.t1_brain'),
            ('subjects_dir', 'inputnode.subjects_dir'),
            ('t1_seg', 'inputnode.t1_seg'),
            ('subject_id', 'inputnode.subject_id'),
            ('t1_2_fsnative_reverse_transform',
             'inputnode.t1_2_fsnative_reverse_transform')]),
        (ants_mvtc2, b0_coreg_wf, [
            ('templates', 'inputnode.ref_b0_brain')]),
        (b0_coreg_wf, ds_report_imtcoreg, [('outputnode.report', 'in_file')]),
        (b0_coreg_wf, outputnode, [
            ('outputnode.itk_b0_to_t1', 'intramodal_template_to_t1_affine')])
    ])  # fmt:skip

    # Fill-in datasinks of reportlets seen so far
    for node in workflow.list_node_names():
        if node.split(".")[-1].startswith("ds_report"):
            workflow.get_node(node).inputs.base_directory = reportlets_dir
            workflow.get_node(node).inputs.source_file = t1w_source_file

    return workflow


def nonlinear_alignment_iteration(iternum=0, gradient_step=0.2):
    """
    Takes a template image and a set of input images, does
    a linear alignment to the template and updates it with the
    inverse of the average affine transform to the new template

    Returns a workflow

    """
    iteration_wf = Workflow(name="nl_iterative_alignment_%03d" % iternum)
    input_node_fields = ["image_paths", "template_image", "iteration_num"]
    inputnode = pe.Node(niu.IdentityInterface(fields=input_node_fields), name="inputnode")
    inputnode.inputs.iteration_num = iternum
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "registered_image_paths",
                "affine_transforms",
                "warp_transforms",
                "composite_transforms",
                "updated_template",
            ]
        ),
        name="outputnode",
    )
    ants_settings = pkgrf("qsiprep", "data/intramodal_nonlinear.json")
    reg = ants.Registration(from_file=ants_settings)
    iter_reg = pe.MapNode(reg, name="nlreg_%03d" % iternum, iterfield=["moving_image"])

    # Average the images
    averaged_images = pe.Node(
        ants.AverageImages(normalize=True, dimension=3), name="averaged_images"
    )

    # Shape update to template:
    # Average the affines so that the inverse can be applied to the template
    affines_to_list = pe.Node(niu.Merge(1), name="affines_to_list")
    warps_to_list = pe.Node(niu.Merge(1), name="warps_to_list")
    avg_affines = pe.Node(
        ants.AverageAffineTransform(dimension=3, output_affine_transform="AveragedAffines.mat"),
        name="avg_affines",
    )

    # Average the warps:
    average_warps = pe.Node(ants.AverageImages(dimension=3, normalize=False), name="average_warps")
    # Scale by the gradient step
    scale_warp = pe.Node(
        ants.MultiplyImages(
            dimension=3, second_input=gradient_step, output_product_image="scaled_warp.nii.gz"
        ),
        name="scale_warp",
    )
    # Align the warps to the template image
    align_warp = pe.Node(
        ants.ApplyTransforms(input_image_type=1, invert_transform_flags=[True]), name="align_warp"
    )

    # transform the template for the shape update
    shape_update_template = pe.Node(
        ants.ApplyTransforms(
            interpolation="LanczosWindowedSinc",
            invert_transform_flags=[True, False, False, False, False],
        ),
        name="shape_update_template",
    )
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
    ])  # fmt:skip

    return iteration_wf


def _list_squeeze(in_list):
    return [item[0] for item in in_list]
