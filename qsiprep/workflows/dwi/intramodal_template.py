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
from .util import _list_squeeze

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

    merge_inputs = pe.Node(niu.Merge(len(input_names)), name='merge_inputs')
    for input_num, input_name in enumerate(input_names):
        workflow.connect(inputnode, input_name, merge_inputs, 'in%d' % (input_num + 1))

    rename_inputs = pe.MapNode(
        niu.Rename(keep_ext=True),
        iterfield=['in_file', 'format_string'],
        name='rename_inputs',
    )
    rename_inputs.inputs.format_string = input_names
    rename_inputs.synchronize = True
    workflow.connect([(merge_inputs, rename_inputs, [('out', 'in_file')])])

    split_outputs = pe.Node(
        niu.Split(splits=[1] * len(input_names), squeeze=True), name='split_outputs'
    )
    for output_num, output_name in enumerate(output_names):
        workflow.connect(split_outputs, 'out%d' % (output_num + 1), outputnode, output_name)

    # antsMultivariateTemplateConstruction2 only offers BSplineSyN/SyN/Affine, so
    # linear-only templates are built with init_b0_hmc_wf instead. That also lets
    # Rigid be offered at all -- it is not in the mvtc2 enum.
    linear_only = transform in ('Rigid', 'Affine')
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
        ])  # fmt:skip

        # Per-input agreement with the template, linear-only: mvtc2 does not
        # expose the per-input aligned images TemplateQC needs.
        template_qc = pe.Node(TemplateQC(labels=list(inputs_list)), name='template_qc')
        workflow.connect([
            (linear_template_wf, template_qc, [
                ('outputnode.final_template', 'template'),
                ('outputnode.aligned_images', 'aligned_images'),
                (('outputnode.forward_transforms', _list_squeeze), 'transforms'),
            ]),
            (template_qc, outputnode, [
                ('out_file', 'template_qc_file'),
                ('agreement_map', 'template_agreement_map'),
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
    b0_coreg_wf = init_b0_to_anat_registration_wf(
        write_report=True,
        transform_type=config.workflow.b0_to_anat_transform,
    )
    workflow.connect([
        (inputnode, b0_coreg_wf, [
            ('t1_brain', 'inputnode.t1_brain'),
            ('subjects_dir', 'inputnode.subjects_dir'),
            ('t1_seg', 'inputnode.t1_seg'),
            ('subject_id', 'inputnode.subject_id'),
        ]),
        (template_node, b0_coreg_wf, [(template_field, 'inputnode.ref_b0_brain')]),
        (b0_coreg_wf, outputnode, [
            ('outputnode.itk_b0_to_t1', 'intramodal_template_to_t1_affine'),
        ]),
    ])  # fmt:skip

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
    workflow.connect([(b0_coreg_wf, ds_report_imtcoreg, [('outputnode.report', 'in_file')])])

    # The template lives in its own midpoint space, not ACPC: anything written
    # out has to go through the coregistration b0_coreg_wf computes, or it would
    # be mislabelled if tagged space-ACPC.
    template_to_acpc = pe.Node(
        ants.ApplyTransforms(interpolation='LanczosWindowedSinc', float=True),
        name='template_to_acpc',
    )
    workflow.connect([
        (inputnode, template_to_acpc, [('t1_brain', 'reference_image')]),
        (b0_coreg_wf, template_to_acpc, [('outputnode.itk_b0_to_t1', 'transforms')]),
        (template_to_acpc, outputnode, [('output_image', 'intramodal_template_acpc')]),
        (template_node, template_to_acpc, [(template_field, 'input_image')]),
    ])  # fmt:skip

    # White matter contours for the registration reportlet: invert the
    # template->anat affine to carry the anatomical segmentation into template
    # space. itk_b0_to_t1 is an affine regardless of the intramodal transform
    # type, so the exact inversion stays valid for Rigid/Affine/SyN alike.
    seg_to_template = pe.Node(
        ants.ApplyTransforms(
            dimension=3,
            float=True,
            interpolation='MultiLabel',
            invert_transform_flags=[True],
        ),
        name='seg_to_template',
    )
    workflow.connect([
        (inputnode, seg_to_template, [('t1_seg', 'input_image')]),
        (b0_coreg_wf, seg_to_template, [('outputnode.itk_b0_to_t1', 'transforms')]),
        (template_node, seg_to_template, ([template_field, 'reference_image'])),
    ])  # fmt:skip

    template_wm = pe.Node(ExtractWM(), name='template_wm')
    workflow.connect([
        (seg_to_template, template_wm, [('output_image', 'in_seg')]),
        (template_wm, outputnode, [('out', 'intramodal_template_wm_seg')]),
    ])  # fmt:skip

    return workflow
