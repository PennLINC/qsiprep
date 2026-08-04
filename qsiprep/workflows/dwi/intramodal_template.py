"""
Head motion correction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_dwi_hmc_wf
.. autofunction:: init_dwi_model_hmc_wf

"""

import nipype.pipeline.engine as pe
from nipype.interfaces import ants
from nipype.interfaces import utility as niu
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from ... import config
from ...data import load as load_data
from ...interfaces import DerivativesDataSink
from ...interfaces.ants import MultivariateTemplateConstruction2
from ...interfaces.images import ExtractWM
from ...interfaces.template_qc import TemplateQC
from .hmc import init_b0_hmc_wf
from .registration import init_b0_to_anat_registration_wf

DEFAULT_MEMORY_MIN_GB = 0.01


def init_intramodal_template_wf(
    inputs_list,
    t1w_source_file,
    transform='BSplineSyN',
    num_iterations=2,
    mem_gb=3,
    name='intramodal_template_wf',
):
    """Create an unbiased intramodal template for a subject. This aligns the b=0 references
    from all the scans of a subject. Can be rigid, affine or nonlinear (BSplineSyN).

    Parameters
    ----------
    inputs_list: list of inputs
        List if identifiers for the input b=0 images.
    transform: 'Rigid', 'Affine', 'BSplineSyN'
        Which transform to ultimately use. If 'BSplineSyN', first 2 iterations of Affine will
        be run.
    num_iterations: int
        Default: 2.

    Inputs
    ------
    [workflow_name]_image...
        One input for each input image. There is no input called inputs_list
    t1w_image

    Outputs
    -------
    [workflow_name]_transform
        transform files to the intramodal template

    intramodal_template_to_t1w_transform
        Transform from the b0

    """
    omp_nthreads = config.nipype.omp_nthreads
    workflow = Workflow(name=name)
    input_names = [name.replace('-', '_') + '_b0_template' for name in inputs_list]
    output_names = [name.replace('-', '_') + '_transform' for name in inputs_list]

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=input_names
            + [
                't1_brain',
                't1_preproc',
                't1_mask',
                't1_seg',
                'subjects_dir',
                'subject_id',
                't1_aseg',
                't1_aparc',
                't1_tpms',
                't1_2_mni_forward_transform',
                'dwi_sampling_grid',
                't1_2_mni_reverse_transform',
            ]
        ),
        name='inputnode',
    )
    merge_inputs = pe.Node(niu.Merge(len(input_names)), name='merge_inputs')
    rename_inputs = pe.MapNode(
        niu.Rename(keep_ext=True), iterfield=['in_file', 'format_string'], name='rename_inputs'
    )
    rename_inputs.inputs.format_string = input_names
    rename_inputs.synchronize = True
    for input_num, input_name in enumerate(input_names):
        workflow.connect(inputnode, input_name, merge_inputs, 'in%d' % (input_num + 1))

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=output_names
            + [
                'intramodal_template',
                'intramodal_template_acpc',
                'intramodal_template_wm_seg',
                'template_qc_file',
                'template_agreement_map',
                'intramodal_template_to_t1_affine',
                'intramodal_template_to_t1_warp',
            ]
        ),
        name='outputnode',
    )
    split_outputs = pe.Node(
        niu.Split(splits=[1] * len(input_names), squeeze=True), name='split_outputs'
    )
    for output_num, output_name in enumerate(output_names):
        workflow.connect(split_outputs, 'out%d' % (output_num + 1), outputnode, output_name)

    # antsMultivariateTemplateConstruction2 only offers BSplineSyN/SyN/Affine, so
    # linear-only templates are built with init_b0_hmc_wf instead. That also lets
    # Rigid be offered at all -- it is not in the mvtc2 enum.
    #
    # Previously the transform was never passed here, so every intramodal template
    # was BSplineSyN regardless of --intramodal-template-transform, silently
    # nonlinearly warping genuine between-session differences into agreement.
    # Per-input agreement with the template. Numbers sort; montages do not.
    template_qc = pe.Node(TemplateQC(labels=list(inputs_list)), name='template_qc')

    linear_only = transform in ('Rigid', 'Affine')

    workflow.connect([
        (merge_inputs, rename_inputs, [('out', 'in_file')]),
    ])  # fmt:skip

    if linear_only:
        # initialize_com because sessions can differ by centimetres of table
        # position, which the shoreline settings' two resolution levels and
        # absent initialization will not recover from.
        linear_template_wf = init_b0_hmc_wf(
            align_to='iterative',
            transform=transform,
            num_iters=max(int(num_iterations), 2),
            initialize_com=True,
            boilerplate=False,
            settings='unbiased_template',
            name='intramodal_linear_template',
        )
        workflow.connect([
            (rename_inputs, linear_template_wf, [('out_file', 'inputnode.b0_images')]),
            (linear_template_wf, split_outputs, [
                (('outputnode.forward_transforms', _list_squeeze), 'inlist'),
            ]),
            (linear_template_wf, outputnode, [
                ('outputnode.final_template', 'intramodal_template'),
            ]),
            (linear_template_wf, template_qc, [
                ('outputnode.aligned_images', 'aligned_images'),
                (('outputnode.forward_transforms', _list_squeeze), 'transforms'),
            ]),
        ])  # fmt:skip
        template_node, template_field = linear_template_wf, 'outputnode.final_template'
    else:
        runtime_opts = {'num_cores': 1, 'parallel_control': 0}
        if omp_nthreads > 1:
            runtime_opts = {'num_cores': omp_nthreads, 'parallel_control': 2}
        ants_mvtc2 = pe.Node(
            MultivariateTemplateConstruction2(
                dimension=3,
                iteration_limit=num_iterations,
                transform=transform,
                **runtime_opts,
            ),
            name='ants_mvtc2',
            n_procs=omp_nthreads,
        )
        workflow.connect([
            (rename_inputs, ants_mvtc2, [('out_file', 'input_images')]),
            (ants_mvtc2, split_outputs, [('forward_transforms', 'inlist')]),
            (ants_mvtc2, outputnode, [('templates', 'intramodal_template')]),
        ])  # fmt:skip
        template_node, template_field = ants_mvtc2, 'templates'

    # calculate dwi registration to T1w
    b0_coreg_wf = init_b0_to_anat_registration_wf(write_report=True)
    ds_report_imtcoreg = pe.Node(
        DerivativesDataSink(
            datatype='figures',
            desc='intramodalcoreg',
            source_file=t1w_source_file,
        ),
        name='ds_report_imtcoreg',
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    workflow.connect([
        (inputnode, b0_coreg_wf, [
            ('t1_brain', 'inputnode.t1_brain'),
            ('subjects_dir', 'inputnode.subjects_dir'),
            ('t1_seg', 'inputnode.t1_seg'),
            ('subject_id', 'inputnode.subject_id'),
        ]),
        (b0_coreg_wf, ds_report_imtcoreg, [('outputnode.report', 'in_file')]),
        (b0_coreg_wf, outputnode, [
            ('outputnode.itk_b0_to_t1', 'intramodal_template_to_t1_affine'),
        ]),
    ])  # fmt:skip

    workflow.connect(
        template_node, template_field, b0_coreg_wf, 'inputnode.ref_b0_brain'
    )  # fmt:skip

    # The template lives in its own midpoint space, ~57mm from ACPC. Anything
    # written out has to go through the coregistration b0_coreg_wf already
    # computes, or it is unusable next to the anatomicals -- and mislabelled if
    # tagged space-ACPC.
    template_to_acpc = pe.Node(
        ants.ApplyTransforms(interpolation='LanczosWindowedSinc', float=True),
        name='template_to_acpc',
    )
    workflow.connect([
        (inputnode, template_to_acpc, [('t1_brain', 'reference_image')]),
        (b0_coreg_wf, template_to_acpc, [('outputnode.itk_b0_to_t1', 'transforms')]),
        (template_to_acpc, outputnode, [('output_image', 'intramodal_template_acpc')]),
    ])  # fmt:skip
    workflow.connect(
        template_node, template_field, template_to_acpc, 'input_image'
    )  # fmt:skip
    workflow.connect(
        template_node, template_field, template_qc, 'template'
    )  # fmt:skip
    workflow.connect([
        (template_qc, outputnode, [
            ('out_file', 'template_qc_file'),
            ('agreement_map', 'template_agreement_map'),
        ]),
    ])  # fmt:skip

    # White matter contours for the registration reportlet.
    #
    # The ordering works out: b0_coreg_wf registers the TEMPLATE to the anatomy,
    # so the template<->anat affine exists once the template is built and before
    # anything downstream needs it. Inverting it carries the anatomical
    # segmentation the other way, into template space. Same idiom as
    # init_fmap_unwarp_report_wf (MultiLabel + invert_transform_flags).
    #
    # NOTE: itk_b0_to_t1 is an affine, so inversion is exact. A nonlinear
    # intramodal transform would need its inverse warp instead -- but that
    # transform is template->anat regardless of the intramodal transform type,
    # so this stays valid for Rigid/Affine/SyN alike.
    seg_to_template = pe.Node(
        ants.ApplyTransforms(
            dimension=3,
            float=True,
            interpolation='MultiLabel',
            invert_transform_flags=[True],
        ),
        name='seg_to_template',
    )
    template_wm = pe.Node(ExtractWM(), name='template_wm')
    workflow.connect([
        (inputnode, seg_to_template, [('t1_seg', 'input_image')]),
        (b0_coreg_wf, seg_to_template, [('outputnode.itk_b0_to_t1', 'transforms')]),
        (seg_to_template, template_wm, [('output_image', 'in_seg')]),
        (template_wm, outputnode, [('out', 'intramodal_template_wm_seg')]),
    ])  # fmt:skip
    workflow.connect(
        template_node, template_field, seg_to_template, 'reference_image'
    )  # fmt:skip

    return workflow


def init_qsiprep_intramodal_template_wf(
    inputs_list,
    transform='Rigid',
    num_iterations=2,
    name='intramodal_template_wf',
):
    """Create an unbiased intramodal template for a subject.
    This aligns the b=0 references
    from all the scans of a subject. Can be rigid, affine or nonlinear (BSplineSyN).

    Parameters
    ----------
    inputs_list: list of inputs
        List if identifiers for the input b=0 images.
    transform: 'Rigid', 'Affine', 'BSplineSyN'
        Which transform to ultimately use. If 'BSplineSyN', first 2 iterations of Affine will
        be run.
    num_iterations: int
        Default: 2.

    Inputs
    ------
    [workflow_name]_image...
        One input for each input image. There is no input called inputs_list
    t1w_image

    Outputs
    -------
    [workflow_name]_transform
        transform files to the intramodal template

    intramodal_template_to_t1w_transform
        Transform from the b0

    """
    workflow = Workflow(name=name)
    input_names = [name + '_b0_template' for name in inputs_list]
    output_names = [name + '_transform' for name in inputs_list]

    inputnode = pe.Node(
        niu.IdentityInterface(fields=input_names + ['t1w_brain']), name='inputnode'
    )
    merge_inputs = pe.Node(niu.Merge(len(input_names)), name='merge_inputs')
    for input_num, input_name in enumerate(input_names):
        workflow.connect(inputnode, input_name, merge_inputs, 'in%d' % (input_num + 1))

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=output_names + ['intramodal_template', 'intramodal_template_to_t1w_transform']
        ),
        name='outputnode',
    )
    split_outputs = pe.Node(
        niu.Split(splits=[1] * len(input_names), squeeze=True), name='split_outputs'
    )
    for output_num, output_name in enumerate(output_names):
        workflow.connect(split_outputs, 'out%d' % (output_num + 1), outputnode, output_name)

    # N4 correct
    n4_correct = pe.MapNode(
        ants.N4BiasFieldCorrection(
            dimension=3,
            copy_header=True,
            n_iterations=[50, 50, 40, 30],
            shrink_factor=2,
            convergence_threshold=0.00000001,
            bspline_fitting_distance=200,
            bspline_order=3,
        ),
        name='n4_correct',
        iterfield=['input_image'],
    )

    # Should we add nonlinear iterations?
    do_nonlinear = transform not in ('Rigid', 'Affine')

    # Align the b=0 images from all runs (Linear)
    initial_transform = 'Affine' if do_nonlinear else transform
    intramodal_b0_affine_template = init_b0_hmc_wf(
        align_to='iterative',
        num_iters=2,
        transform=initial_transform,
        spatial_bias_correct=True,
        name='intramodal_b0_affine_template',
    )

    workflow.connect([
        (merge_inputs, n4_correct, [('out', 'input_image')]),
        (n4_correct, intramodal_b0_affine_template, [('output_image', 'inputnode.b0_images')]),
    ])  # fmt:skip
    if not do_nonlinear:
        workflow.connect([
            (intramodal_b0_affine_template, split_outputs, [
                (('outputnode.forward_transforms', _list_squeeze), 'inlist'),
            ]),
        ])  # fmt:skip
    else:
        nonlinear_alignment_wf = init_nonlinear_alignment_wf(num_iters=num_iterations)
        workflow.connect([
            (n4_correct, nonlinear_alignment_wf, [('output_image', 'inputnode.images')]),
            (intramodal_b0_affine_template, nonlinear_alignment_wf, [
                ('outputnode.final_template', 'inputnode.initial_template'),
            ]),
            (nonlinear_alignment_wf, split_outputs, [('outputnode.forward_transforms', 'inlist')]),
        ])  # fmt:skip

    return workflow


def nonlinear_alignment_iteration(iternum=0, gradient_step=0.2):
    """
    Takes a template image and a set of input images, does
    a linear alignment to the template and updates it with the
    inverse of the average affine transform to the new template

    Returns a workflow

    """
    iteration_wf = Workflow(name='nl_iterative_alignment_%03d' % iternum)
    input_node_fields = ['image_paths', 'template_image', 'iteration_num']
    inputnode = pe.Node(niu.IdentityInterface(fields=input_node_fields), name='inputnode')
    inputnode.inputs.iteration_num = iternum
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'registered_image_paths',
                'affine_transforms',
                'warp_transforms',
                'composite_transforms',
                'updated_template',
            ]
        ),
        name='outputnode',
    )
    ants_settings = str(load_data('intramodal_nonlinear.json'))
    reg = ants.Registration(from_file=ants_settings)
    iter_reg = pe.MapNode(reg, name='nlreg_%03d' % iternum, iterfield=['moving_image'])

    # Average the images
    averaged_images = pe.Node(
        ants.AverageImages(normalize=True, dimension=3), name='averaged_images'
    )

    # Shape update to template:
    # Average the affines so that the inverse can be applied to the template
    affines_to_list = pe.Node(niu.Merge(1), name='affines_to_list')
    warps_to_list = pe.Node(niu.Merge(1), name='warps_to_list')
    avg_affines = pe.Node(
        ants.AverageAffineTransform(dimension=3, output_affine_transform='AveragedAffines.mat'),
        name='avg_affines',
    )

    # Average the warps:
    average_warps = pe.Node(ants.AverageImages(dimension=3, normalize=False), name='average_warps')
    # Scale by the gradient step
    scale_warp = pe.Node(
        ants.MultiplyImages(
            dimension=3, second_input=gradient_step, output_product_image='scaled_warp.nii.gz'
        ),
        name='scale_warp',
    )
    # Align the warps to the template image
    align_warp = pe.Node(
        ants.ApplyTransforms(input_image_type=1, invert_transform_flags=[True]), name='align_warp'
    )

    # transform the template for the shape update
    shape_update_template = pe.Node(
        ants.ApplyTransforms(
            interpolation='LanczosWindowedSinc',
            invert_transform_flags=[True, False, False, False, False],
        ),
        name='shape_update_template',
    )
    shape_update_merge = pe.Node(niu.Merge(5), name='shape_update_merge')

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


def init_nonlinear_alignment_wf(num_iters=2, name='nonlinear_alignment_wf'):
    """Creates a workflow that does nonlinear template creation."""
    workflow = Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['images', 'initial_template']), name='inputnode'
    )
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

    # Save the iteration templates
    iter_templates = pe.Node(niu.Merge(num_iters), name='iteration_templates')

    initial_reg = nonlinear_alignment_iteration(iternum=0)

    workflow.connect([
        (inputnode, iter_templates, [('initial_template', 'in1')]),
        (inputnode, initial_reg, [
            ('initial_template', 'inputnode.template_image'),
            ('images', 'inputnode.image_paths'),
        ]),
    ])  # fmt:skip

    reg_iters = [initial_reg]
    for iternum in range(1, num_iters):
        reg_iters.append(nonlinear_alignment_iteration(iternum=iternum))
        workflow.connect([
            (reg_iters[-2], reg_iters[-1], [
                ('outputnode.updated_template', 'inputnode.template_image'),
            ]),
            (inputnode, reg_iters[-1], [('images', 'inputnode.image_paths')]),
            (reg_iters[-1], iter_templates, [
                ('outputnode.updated_template', 'in%d' % (iternum + 1)),
            ]),
        ])  # fmt:skip

    # Attach to outputs
    # The last iteration aligned to the output from the second-to-last
    workflow.connect([
        (reg_iters[-2], outputnode, [('outputnode.updated_template', 'final_template')]),
        (reg_iters[-1], outputnode, [
            ('outputnode.affine_transforms', 'forward_transforms'),
            ('outputnode.registered_image_paths', 'aligned_images'),
        ]),
        (iter_templates, outputnode, [('out', 'iteration_templates')]),
    ])  # fmt:skip

    return workflow


def _list_squeeze(in_list):
    return [item[0] for item in in_list]
