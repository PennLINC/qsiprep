# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
.. _sdc_drbuddi :

Correcting Susceptibility Distortion with DRBUDDI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DRBUDDI is part of the TORTOISE software that estimates and corrects
susceptibility distortion. It has multiple modes of operation

  1. Use $b=0$ images to estimate distortion.

  2. Perform a multimodal registration using $b=0$ images and FA images.
     This requires two DWI series with opposite phase encoding directions

  3. Either (1) or (2) but a t2w image is used as well

"""

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from ... import config
from ...interfaces.tortoise import (
    DRBUDDI,
    DRBUDDIAggregateOutputs,
    GatherDRBUDDIInputs,
    generate_drbuddi_boilerplate,
    sloppy_epi_working_res,
)
from ..dwi.registration import init_structural_to_b0_alignment_wf

DEFAULT_MEMORY_MIN_GB = 0.01


def _synth_shell_kwargs(bval, ndirs):
    """DRBUDDI shell-synthesis kwargs, or empty when the opt-in is off.

    Returned as kwargs rather than passed as 0 so that a stock (unpatched)
    TORTOISE, which does not know --DRBUDDI_synth_shell_bval, is unaffected
    unless the user explicitly asks for synthesis.
    """
    if not bval or bval <= 0:
        return {}
    return {'synth_shell_bval': float(bval), 'synth_shell_ndirs': int(ndirs)}


def _negate_displacement_field(in_file):
    """Negate every vector of an ITK displacement field.

    The blip-down ("-" polarity) susceptibility distortion is the opposite of
    blip-up, so DRBUDDI's initial down field is the negation of the up field.
    The vectors live in the same world frame, so a plain element-wise negation
    flips the polarity; the NIFTI_INTENT_VECTOR intent is preserved (TORTOISE
    and ANTs read it as zeros without it).
    """
    import os

    import nibabel as nb
    import numpy as np

    img = nb.load(in_file)
    neg = nb.Nifti1Image(
        (np.asanyarray(img.dataobj) * -1.0).astype('float32'), img.affine, img.header
    )
    neg.header.set_intent('vector')
    out_file = os.path.abspath('initial_moving_transform.nii.gz')
    neg.to_filename(out_file)
    return out_file


def seeds_from_gre(unit):
    """Whether DRBUDDI starts from ``unit``'s GRE candidate: for any PEPOLAR unit
    that a GRE fieldmap also lists, except when DRBUDDI only refines TOPUP's
    correction, which a full GRE warp would duplicate."""
    return (
        unit.is_pepolar
        and unit.gre_init_estimation is not None
        and unit.run.stage_with('topup') is None
    )


def connect_gre_seed(workflow, inputnode, unit, b0_source, drbuddi_wf, has_gradwarp, source_file):
    """Build the warp of ``unit``'s GRE candidate and start ``drbuddi_wf`` from it.

    ``b0_source`` is ``(node, field)`` giving the b=0 average of the volumes
    DRBUDDI corrects, before gradient unwarping; ``inputnode`` carries
    ``t1_brain``, ``t1_2_mni_reverse_transform`` and ``gradwarp_field``. The warp
    is built for DRBUDDI's up series and, with gradient unwarping, transported
    into the frame of the gradwarped up/down volumes. ``drbuddi_wf`` must have
    been built with ``initialize_from_field=True``.
    """
    from ..dwi.gradwarp import connect_gradwarp_sdc_reference
    from ..dwi.util import init_dwi_reference_wf
    from .base import gre_seed_unit, init_sdc_wf

    config.loggers.workflow.info(
        'Initializing DRBUDDI for %s with GRE fieldmap %s.',
        unit.output_name,
        unit.gre_init_estimation.b0field_id,
    )
    src_node, src_field = b0_source
    b0_ref_wf = init_dwi_reference_wf(
        source_file=source_file, name='drbuddi_gre_init_b0_ref_wf', gen_report=False
    )
    seed_sdc_wf = init_sdc_wf(gre_seed_unit(unit), gradwarp=has_gradwarp, use='drbuddi')
    seed_sdc_wf.inputs.inputnode.template = config.workflow.anatomical_template
    ref_fields = ('outputnode.ref_image', 'outputnode.ref_image_brain', 'outputnode.dwi_mask')
    workflow.connect([
        (src_node, b0_ref_wf, [(src_field, 'inputnode.b0_template')]),
        (inputnode, seed_sdc_wf, [
            ('t1_brain', 'inputnode.t1_brain'),
            ('t1_2_mni_reverse_transform', 'inputnode.t1_2_mni_reverse_transform'),
        ]),
        (seed_sdc_wf, drbuddi_wf, [('outputnode.out_warp', 'inputnode.initial_field')]),
    ])  # fmt:skip
    if has_gradwarp:
        connect_gradwarp_sdc_reference(workflow, inputnode, b0_ref_wf, ref_fields, seed_sdc_wf)
    else:
        workflow.connect([
            (b0_ref_wf, seed_sdc_wf, [
                (ref_fields[0], 'inputnode.b0_ref'),
                (ref_fields[1], 'inputnode.b0_ref_brain'),
                (ref_fields[2], 'inputnode.b0_mask'),
            ]),
        ])  # fmt:skip


def init_drbuddi_wf(
    unit,
    t2w_sdc,
    use_cuda=False,
    synth_shell_bval=None,
    synth_shell_ndirs=30,
    initialize_from_field=False,
):
    """
    This workflow implements the heuristics to choose a
    :abbr:`SDC (susceptibility distortion correction)` strategy.


    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.fieldmap import init_drbuddi_wf
        from qsiprep.tests.preproc_factory import make_preproc_unit
        from qsiplan.models import CorrectionMethod
        ap = 'data/tinytensor/sub-tinytensors/dwi/sub-tinytensors_dir-AP_dwi.nii.gz'
        pa = 'data/tinytensor/sub-tinytensors/dwi/sub-tinytensors_dir-PA_dwi.nii.gz'
        wf = init_drbuddi_wf(
            make_preproc_unit(
                [ap, pa],
                method=CorrectionMethod.PEPOLAR,
                pe_dirs={ap: 'j', pa: 'j-'},
            ),
            t2w_sdc=False,
        )

    Parameters
    ----------
    unit : :class:`~qsiplan.adapters.PreprocUnit`
        The reverse-PE DWI series (and any epi fieldmaps) to correct
    use_cuda : :obj:`bool`
        Run ``DRBUDDI_cuda`` instead of ``DRBUDDI``. The GPU must be exposed to
        the container. Results differ from the CPU build, so this is not purely
        a speed knob. Callers pass ``gpu_enabled('drbuddi')``, which is driven by
        ``--gpu`` (with ``"use_cuda"`` in ``--diffprep-config`` as a legacy
        fallback).
    t2w_sdc : bool
        Should a T2w image be included in the DRBUDDI run?
    initialize_from_field : bool
        Seed DRBUDDI's diffeomorphic search from an external displacement field
        (e.g. a GRE-fieldmap-derived warp) supplied on ``inputnode.initial_field``.
        The field becomes the initial up (blip-up) transform and its negation the
        initial down transform.


    Inputs
    ------
    initial_field
        (only when ``initialize_from_field``) an ITK displacement field in the
        pre-SDC b=0 world frame that corrects the blip-up ("+" polarity) b=0.
    dwi_file : str
        Path to a motion/eddy corrected DWI file (in LPS+)
    bval_file : str
        Corresponding bval file for dwi_file
    bvec_file : str
        Corresponding bvec file for dwi_file (in LPS+)
    original_files : list
        List of the original BIDS file for each image in dwi_file
    t1_brain
        T1w image, brain-masked
    t2_brain
        T2w image, brain masked
    b0_ref
        Pre-SDC b=0 reference in the frame ``dwi_files`` are in. The T2w is
        pre-aligned to it (antsAI rotation search) before reaching DRBUDDI,
        whose internal rigid registration cannot recover large rotations.

    Outputs
    -------
    b0_ref
        An unwarped b0 reference
    b0_mask
        The corresponding new mask after unwarping
    sdc_warps
        The deformation fields to unwarp the susceptibility distortions in each image
        in dwi_file

    """

    workflow = Workflow(name='drbuddi_sdc_wf')
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'dwi_files',
                'bval_files',
                'bvec_files',
                'original_files',
                't1_brain',
                't1_wm_seg',
                't2w_unfatsat',
                'b0_ref',
                'initial_field',
            ]
        ),
        name='inputnode',
    )

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'b0_ref',
                'b0_mask',
                'sdc_warps',
                'sdc_scaling_images',
                'report',
                'method',
                # From SDC
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
        name='outputnode',
    )

    if not unit.is_pepolar:
        raise Exception('DRBUDDI workflow requires a PEPOLAR fieldmap')

    # The interfaces still discriminate on this legacy string:
    # reverse-PE *series* vs a dedicated epi b=0.
    fieldmap_type = unit.pepolar_fieldmap_type
    epi_fmaps = list(unit.minus_files) if unit.has_bidirectional_dwi else list(unit.extra_b0)

    workflow.__desc__ = generate_drbuddi_boilerplate(
        fieldmap_type=fieldmap_type,
        t2w_sdc=t2w_sdc,
        with_topup=unit.run.stage_with('topup') is not None,
        initialized=initialize_from_field,
        keep_initialization_fixed=config.workflow.gre_init_keep_fixed,
    )

    outputnode.inputs.method = f'PEB/PEPOLAR (phase-encoding based / PE-POLARity): {fieldmap_type}'
    if initialize_from_field:
        outputnode.inputs.method += ' (GRE-initialized)'

    gather_drbuddi_inputs = pe.Node(
        GatherDRBUDDIInputs(
            dwi_series_pedir=unit.pe_dir,
            epi_fmaps=epi_fmaps,
            b0_threshold=config.workflow.b0_threshold,
            fieldmap_type=fieldmap_type,
            # Model-derived metadata so the up/down blip split skips sidecar reads.
            sidecars=unit.sidecar_overrides(),
        ),
        name='gather_drbuddi_inputs',
    )

    drbuddi = pe.Node(
        DRBUDDI(
            fieldmap_type=fieldmap_type,
            num_threads=config.nipype.omp_nthreads,
            sloppy=config.execution.sloppy,
            **sloppy_epi_working_res(),
            **_synth_shell_kwargs(synth_shell_bval, synth_shell_ndirs),
            use_cuda=use_cuda,
            # NOTE: --DRBUDDI_start_with_diffeomorphic_for_rigid_reg and
            # --DRBUDDI_disable_initial_rigid look like natural companions to
            # ``sloppy``, but both are commented out of TORTOISE's parser:
            # DRBUDDI prints "Unknown command line parameter", exits 0 (which
            # nipype reads as success), and the run dies later on missing
            # outputs. Neither flag is safe to send.
        ),
        name='drbuddi',
        n_procs=config.nipype.omp_nthreads,
    )

    if initialize_from_field:
        drbuddi.inputs.keep_initial_transform_fixed = config.workflow.gre_init_keep_fixed
        negate_initial_field = pe.Node(
            niu.Function(
                input_names=['in_file'],
                output_names=['out_file'],
                function=_negate_displacement_field,
            ),
            name='negate_initial_field',
        )
        workflow.connect([
            (inputnode, drbuddi, [('initial_field', 'initial_fixed_transform')]),
            (inputnode, negate_initial_field, [('initial_field', 'in_file')]),
            (negate_initial_field, drbuddi, [('out_file', 'initial_moving_transform')]),
        ])  # fmt:skip

    aggregate_drbuddi = pe.Node(
        DRBUDDIAggregateOutputs(fieldmap_type=fieldmap_type), name='aggregate_drbuddi'
    )

    workflow.connect([
        (inputnode, gather_drbuddi_inputs, [
            ('dwi_files', 'dwi_files'),
            ('bval_files', 'bval_files'),
            ('bvec_files', 'bvec_files'),
            ('original_files', 'original_files'),
        ]),
        (gather_drbuddi_inputs, drbuddi, [
            ('blip_assignments', 'blip_assignments'),
            ('blip_up_image', 'blip_up_image'),
            ('blip_up_json', 'blip_up_json'),
            ('blip_up_bmat', 'blip_up_bmat'),
            ('blip_down_image', 'blip_down_image'),
            ('blip_down_bmat', 'blip_down_bmat')]),
        (drbuddi, outputnode, [
            ('blip_down_b0', 'b0_down_image'),
            ('blip_up_b0', 'b0_up_image'),
            ('blip_down_b0_corrected', 'b0_down_corrected_image'),
            ('blip_up_b0_corrected', 'b0_up_corrected_image'),
            ('blip_down_FA', 'down_fa_image'),
            ('blip_up_FA', 'up_fa_image'),
            ('structural_image', 't2w_image'),
        ]),
        (drbuddi, aggregate_drbuddi, [
            ('undistorted_reference', 'undistorted_reference'),
            ('bdown_to_bup_rigid_trans_h5', 'bdown_to_bup_rigid_trans_h5'),
            ('blip_down_b0', 'blip_down_b0'),
            ('blip_down_b0_corrected', 'blip_down_b0_corrected'),
            ('blip_down_b0_corrected_jac', 'blip_down_b0_corrected_jac'),
            ('blip_down_b0_quad', 'blip_down_b0_quad'),
            ('blip_up_b0', 'blip_up_b0'),
            ('blip_up_b0_corrected', 'blip_up_b0_corrected'),
            ('blip_up_b0_corrected_jac', 'blip_up_b0_corrected_jac'),
            ('blip_up_b0_quad', 'blip_up_b0_quad'),
            ('deformation_finv', 'deformation_finv'),
            ('deformation_minv', 'deformation_minv'),
            ('blip_up_FA', 'blip_up_FA'),
            ('blip_down_FA', 'blip_down_FA'),
            ('structural_image', 'structural_image'),
        ]),
        (gather_drbuddi_inputs, aggregate_drbuddi, [('blip_assignments', 'blip_assignments')]),
        (aggregate_drbuddi, outputnode, [
            ('sdc_warps', 'sdc_warps'),
            ('sdc_scaling_images', 'sdc_scaling_images'),
            ('up_fa_corrected_image', 'up_fa_corrected_image'),
            ('down_fa_corrected_image', 'down_fa_corrected_image'),
            ('b0_ref', 'b0_ref'),
        ]),
    ])  # fmt:skip

    if t2w_sdc:
        t2w_to_b0_wf = init_structural_to_b0_alignment_wf(name='t2w_to_b0_wf')
        workflow.connect([
            (inputnode, t2w_to_b0_wf, [
                ('t2w_unfatsat', 'inputnode.structural_image'),
                ('b0_ref', 'inputnode.b0_ref'),
            ]),
            (t2w_to_b0_wf, drbuddi, [
                ('outputnode.structural_aligned', 'structural_image'),
            ]),
        ])  # fmt:skip

    return workflow
