"""
HMC backend that wraps TORTOISEV4 DIFFPREP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_diffprep_hmc_wf
"""

import json
from importlib.resources import files

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from ... import config
from ...interfaces.gradients import ExtractB0s
from ...interfaces.nilearn import EnhanceB0
from ...interfaces.tortoise import (
    DIFFPREP,
    DIFFPREPDecomposeTransforms,
    DIFFPREPMotionParams,
    DIFFPREPSplitOutputs,
    TORTOISEConvert,
    generate_diffprep_boilerplate,
)
from ...utils.resources import as_path
from ..fieldmap.base import init_sdc_wf
from ..fieldmap.drbuddi import init_drbuddi_wf
from .util import init_dwi_reference_wf

# BIDS PhaseEncodingDirection axes already match what TORTOISEProcess expects
# in its own JSON input — TORTOISE consumes "i", "j", "k" (with optional "-")
# straight from the BIDS sidecar.
_VALID_PE = {'i', 'i-', 'j', 'j-', 'k', 'k-'}


def _load_diffprep_config(config_path):
    """Load a --diffprep-config JSON, or return defaults."""
    if config_path is None:
        config_path = as_path(files('qsiprep.data') / 'diffprep_params.json')
    with open(config_path) as fobj:
        cfg = json.load(fobj)
    cfg.setdefault('b0_id', -1)
    cfg.setdefault('is_human_brain', True)
    cfg.setdefault('rot_eddy_center', 'isocenter')
    cfg.setdefault('extra_args', [])
    return cfg


def _resolve_phase_encoding(pe_dir):
    """Validate a BIDS PhaseEncodingDirection value. Falls back to 'j'
    (anterior-posterior) when missing or not recognised — which is what most
    clinical DWI protocols use."""
    if pe_dir in _VALID_PE:
        return pe_dir
    return 'j'


def init_diffprep_hmc_wf(
    scan_groups,
    source_file,
    t2w_sdc,
    correction_mode='quadratic',
    dwi_metadata=None,
    name='diffprep_hmc_wf',
):
    """HMC + SDC workflow that uses TORTOISEV4 DIFFPREP for motion + eddy
    correction.

    DIFFPREP fits a SHORE/MAPMRI signal model to the data and registers each
    volume to a model-predicted target with TORTOISE's 24-parameter Okan
    quadratic transform. Unlike SHORELine (rigid/affine only) it corrects
    eddy currents; unlike FSL eddy it does not require shelled data.

    The TORTOISE binary writes the corrected DWI directly, so this workflow
    follows the same "bake the correction in, emit identity per-volume
    affines" pattern that ``init_fsl_hmc_wf`` uses for FSL eddy.

    Parameters
    ----------
    scan_groups : dict
        Same scan-groups dict the other HMC backends consume.
    source_file : str
        Path to the source DWI file (used for report naming).
    t2w_sdc : bool
        Whether a T2w image is available for use in DRBUDDI's multi-modal
        registration.
    correction_mode : str
        Forwarded to TORTOISE as ``--correction_mode``. One of ``'motion'``,
        ``'quadratic'`` (recommended), or ``'cubic'``.
    dwi_metadata : dict, optional
        BIDS sidecar metadata (used for SDC).
    name : str
        Workflow name.
    """
    workflow = Workflow(name=name)
    workflow.__desc__ = generate_diffprep_boilerplate(correction_mode)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'dwi_file',
                'bvec_file',
                'bval_file',
                'json_file',
                'rpe_b0',
                't2w_unfatsat',
                'original_files',
                'rpe_b0_info',
                'hmc_optimization_data',
                't1_brain',
                't1_2_mni_reverse_transform',
                't1_mask',
                't1_seg',
                't2_brain',
            ]
        ),
        name='inputnode',
    )

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'b0_template',
                'b0_template_mask',
                'pre_sdc_template',
                'hmc_optimization_data',
                'sdc_method',
                'slice_quality',
                'motion_params',
                'cnr_map',
                'bvec_files_to_transform',
                'dwi_files_to_transform',
                'b0_indices',
                'bval_files',
                'to_dwi_ref_affines',
                'to_dwi_ref_warps',
                'sdc_scaling_images',
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
                'fieldmap_hz',
                # ----- DIFFPREP-only outputs ----------------------------------
                # Per-volume rigid (motion) affines (ITK .txt) and per-volume
                # eddy-current voxel-shift-maps (ITK 5D NIfTI), produced by
                # decomposing TORTOISE's 24-parameter Okan-quadratic transforms.
                # In v1 these are emitted for validation against TORTOISE's own
                # _moteddy.nii (and as a stepping stone to single-shot
                # resampling); the downstream pipeline still consumes the
                # baked-in corrected DWI in dwi_files_to_transform.
                'diffprep_rigid_affines',
                'diffprep_eddy_warps',
            ],
        ),
        name='outputnode',
    )

    # Load any user-supplied DIFFPREP config (or our defaults)
    diffprep_cfg = _load_diffprep_config(config.workflow.diffprep_config)

    # Convert gzipped niftis + FSL gradients into TORTOISE format (.nii + .bmtxt).
    # TORTOISEConvert already handles writing a deobliqued, float32 .nii and
    # the matching .bmtxt next to it.
    tortoise_convert = pe.Node(TORTOISEConvert(), name='tortoise_convert')

    # TORTOISE looks for a BIDS-style JSON sidecar next to the .nii. We
    # generate one with PhaseEncodingDirection so DIFFPREP can pick the
    # right phase axis for its 24-parameter transform.
    write_pe_json = pe.Node(
        niu.Function(
            input_names=['nii_file', 'phase_encoding_direction', 'working_dir'],
            output_names=['json_file'],
            function=_write_sidecar_json,
        ),
        name='write_pe_json',
    )
    pe_dir = _resolve_phase_encoding(
        (dwi_metadata or {}).get('PhaseEncodingDirection')
    )
    write_pe_json.inputs.phase_encoding_direction = pe_dir

    diffprep = pe.Node(
        DIFFPREP(
            correction_mode=correction_mode,
            b0_id=diffprep_cfg['b0_id'],
            is_human_brain=diffprep_cfg['is_human_brain'],
            rot_eddy_center=diffprep_cfg['rot_eddy_center'],
            extra_args=diffprep_cfg['extra_args'],
        ),
        name='diffprep',
    )

    split_outputs = pe.Node(
        DIFFPREPSplitOutputs(b0_threshold=config.workflow.b0_threshold),
        name='split_outputs',
    )

    motion_params = pe.Node(DIFFPREPMotionParams(), name='motion_params')

    # Decompose each 24-parameter Okan transform into a rigid affine plus a
    # dense displacement field along the phase axis. (Currently produced for
    # validation; the wiring to use them as the source of truth for
    # downstream resampling can be flipped once they're verified against
    # TORTOISE's own _moteddy.nii.)
    diffprep_decompose = pe.Node(
        DIFFPREPDecomposeTransforms(
            phase_encoding_direction=pe_dir,
            rot_eddy_center=diffprep_cfg['rot_eddy_center'],
        ),
        name='diffprep_decompose',
    )

    # Build a pre-SDC template from the corrected b=0 series for the report.
    extract_b0s = pe.Node(
        ExtractB0s(b0_threshold=config.workflow.b0_threshold),
        name='extract_b0s',
    )
    enhance_pre_sdc = pe.Node(EnhanceB0(), name='enhance_pre_sdc')

    # Build a "true" b0 reference for downstream coregistration (matches
    # what init_fsl_hmc_wf does).
    b0_ref_for_coreg = init_dwi_reference_wf(
        gen_report=False,
        desc='b0_for_coreg',
        name='b0_ref_for_coreg',
        source_file=source_file,
    )

    workflow.connect([
        (inputnode, tortoise_convert, [
            ('dwi_file', 'dwi_file'),
            ('bval_file', 'bval_file'),
            ('bvec_file', 'bvec_file'),
            ('t1_mask', 'mask_file'),
        ]),
        (tortoise_convert, write_pe_json, [('dwi_file', 'nii_file')]),
        (tortoise_convert, diffprep, [
            ('dwi_file', 'dwi_file'),
            ('bmtxt_file', 'bmtxt_file'),
        ]),
        (write_pe_json, diffprep, [('json_file', 'json_file')]),

        (diffprep, split_outputs, [
            ('corrected_dwi_file', 'corrected_dwi_file'),
            ('corrected_bmtxt_file', 'corrected_bmtxt_file'),
        ]),
        (diffprep, motion_params, [('transformations_file', 'transformations_file')]),

        # Rigid + VSM decomposition of the 24-param transforms
        (diffprep, diffprep_decompose, [
            ('transformations_file', 'transformations_file'),
            ('corrected_dwi_file', 'reference_image'),
        ]),
        (diffprep_decompose, outputnode, [
            ('affine_files', 'diffprep_rigid_affines'),
            ('warp_files', 'diffprep_eddy_warps'),
        ]),

        # Outputnode plumbing
        (split_outputs, outputnode, [
            ('dwi_files', 'dwi_files_to_transform'),
            ('bvec_files', 'bvec_files_to_transform'),
            ('bval_files', 'bval_files'),
            ('b0_indices', 'b0_indices'),
            ('forward_transforms', 'to_dwi_ref_affines'),
        ]),
        (motion_params, outputnode, [('spm_motion_file', 'motion_params')]),

        # Pre-SDC enhancement: ExtractB0s prefers b0_indices when set, so we
        # don't need a single concatenated bval file here.
        (diffprep, extract_b0s, [('corrected_dwi_file', 'dwi_series')]),
        (split_outputs, extract_b0s, [('b0_indices', 'b0_indices')]),
        (extract_b0s, enhance_pre_sdc, [('b0_average', 'b0_file')]),
        (enhance_pre_sdc, outputnode, [('enhanced_file', 'pre_sdc_template')]),

        # Final b0 template for coregistration
        (extract_b0s, b0_ref_for_coreg, [('b0_average', 'inputnode.b0_template')]),
        (inputnode, b0_ref_for_coreg, [
            ('t1_brain', 'inputnode.t1_brain'),
            ('t1_seg', 'inputnode.t1_seg'),
            ('t1_mask', 'inputnode.t1_mask'),
        ]),
        (b0_ref_for_coreg, outputnode, [
            ('outputnode.dwi_mask', 'b0_template_mask'),
        ]),
    ])  # fmt:skip

    fieldmap_info = scan_groups['fieldmap_info']
    fieldmap_type = fieldmap_info['suffix']

    # PEPOLAR (rpe_series / epi) -> DRBUDDI
    if fieldmap_type in ('epi', 'rpe_series'):
        if 'topup' in config.workflow.pepolar_method.lower():
            raise Exception(
                'TOPUP-based pepolar correction is not supported with '
                '--hmc-model diffprep_*; choose --pepolar-method DRBUDDI.'
            )
        outputnode.inputs.sdc_method = 'DRBUDDI'

        drbuddi_wf = init_drbuddi_wf(
            scan_groups=scan_groups,
            t2w_sdc=t2w_sdc,
        )

        workflow.connect([
            (split_outputs, drbuddi_wf, [
                ('dwi_files', 'inputnode.dwi_files'),
                ('bvec_files', 'inputnode.bvec_files'),
                ('bval_files', 'inputnode.bval_files'),
            ]),
            (inputnode, drbuddi_wf, [
                ('t1_brain', 'inputnode.t1_brain'),
                ('t2w_unfatsat', 'inputnode.t2w_unfatsat'),
                ('original_files', 'inputnode.original_files'),
            ]),
            (drbuddi_wf, outputnode, [
                ('outputnode.sdc_warps', 'to_dwi_ref_warps'),
                ('outputnode.sdc_scaling_images', 'sdc_scaling_images'),
                ('outputnode.method', 'sdc_method'),
                ('outputnode.fieldmap_type', 'fieldmap_type'),
                ('outputnode.b0_up_image', 'b0_up_image'),
                ('outputnode.b0_up_corrected_image', 'b0_up_corrected_image'),
                ('outputnode.b0_down_image', 'b0_down_image'),
                ('outputnode.b0_down_corrected_image', 'b0_down_corrected_image'),
                ('outputnode.up_fa_image', 'up_fa_image'),
                ('outputnode.up_fa_corrected_image', 'up_fa_corrected_image'),
                ('outputnode.down_fa_image', 'down_fa_image'),
                ('outputnode.down_fa_corrected_image', 'down_fa_corrected_image'),
                ('outputnode.t2w_image', 't2w_image'),
                ('outputnode.b0_ref', 'b0_template'),
            ]),
        ])  # fmt:skip
        return workflow

    if fieldmap_type is None:
        config.loggers.workflow.warning(
            'SDC: no fieldmaps found or they were ignored (%s).', source_file
        )
    elif fieldmap_type == 'syn':
        config.loggers.workflow.warning(
            'SDC: no fieldmaps found or they were ignored. '
            'Using EXPERIMENTAL "fieldmap-less SyN" correction for dataset %s.',
            source_file,
        )
    else:
        config.loggers.workflow.log(
            25,
            'SDC: fieldmap estimation of type "%s" intended for %s found.',
            fieldmap_type,
            source_file,
        )

    # Non-PEPOLAR fieldmaps (or no fieldmap) -> traditional SDC sub-workflow.
    b0_sdc_wf = init_sdc_wf(scan_groups['fieldmap_info'], dwi_metadata)
    b0_sdc_wf.inputs.inputnode.template = config.workflow.anatomical_template

    workflow.connect([
        (b0_ref_for_coreg, b0_sdc_wf, [
            ('outputnode.ref_image', 'inputnode.b0_ref'),
            ('outputnode.ref_image_brain', 'inputnode.b0_ref_brain'),
            ('outputnode.dwi_mask', 'inputnode.b0_mask'),
        ]),
        (inputnode, b0_sdc_wf, [
            ('t1_brain', 'inputnode.t1_brain'),
            ('t1_2_mni_reverse_transform', 'inputnode.t1_2_mni_reverse_transform'),
        ]),
        (b0_sdc_wf, outputnode, [
            ('outputnode.method', 'sdc_method'),
            ('outputnode.b0_ref', 'b0_template'),
            ('outputnode.out_warp', 'to_dwi_ref_warps'),
        ]),
    ])  # fmt:skip

    return workflow


def _write_sidecar_json(nii_file, phase_encoding_direction, working_dir=None):
    """Function-node helper. Writes a BIDS-style JSON sidecar next to a .nii
    so TORTOISEProcess can read PhaseEncodingDirection from it."""
    import json
    import os.path as op

    base = nii_file
    if base.endswith('.nii.gz'):
        base = base[: -len('.nii.gz')]
    elif base.endswith('.nii'):
        base = base[: -len('.nii')]
    json_file = base + '.json'
    if working_dir:
        json_file = op.join(working_dir, op.basename(json_file))
    with open(json_file, 'w') as fobj:
        json.dump({'PhaseEncodingDirection': phase_encoding_direction}, fobj)
    return json_file
