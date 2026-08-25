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


def init_drbuddi_wf(
    unit,
    t2w_sdc,
    use_cuda=False,
    synth_shell_bval=None,
    synth_shell_ndirs=30,
):
    """
    This workflow implements the heuristics to choose a
    :abbr:`SDC (susceptibility distortion correction)` strategy.


    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.fieldmap import init_drbuddi_wf
        from qsiprep.tests.preproc_factory import make_preproc_unit
        from qsiprep.grouping.models import CorrectionMethod
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
    unit : :class:`~qsiprep.grouping.adapters.PreprocUnit`
        The reverse-PE DWI series (and any epi fieldmaps) to correct
    use_cuda : :obj:`bool`
        Run ``DRBUDDI_cuda`` instead of ``DRBUDDI``. The GPU must be exposed to
        the container. Results differ from the CPU build, so this is not purely
        a speed knob. Callers pass ``gpu_enabled('drbuddi')``, which is driven by
        ``--gpu`` (with ``"use_cuda"`` in ``--diffprep-config`` as a legacy
        fallback).
    t2w_sdc : bool
        Should a T2w image be included in the DRBUDDI run?


    Inputs
    ------
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
    )

    outputnode.inputs.method = f'PEB/PEPOLAR (phase-encoding based / PE-POLARity): {fieldmap_type}'

    gather_drbuddi_inputs = pe.Node(
        GatherDRBUDDIInputs(
            dwi_series_pedir=unit.pe_dir,
            epi_fmaps=epi_fmaps,
            b0_threshold=config.workflow.b0_threshold,
            raw_image_sdc=True,
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
        (inputnode, drbuddi, [('t2w_unfatsat', 'structural_image')]),
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

    return workflow
