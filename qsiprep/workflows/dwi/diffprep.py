"""
HMC + SDC backend that wraps TORTOISEV4 DIFFPREP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_diffprep_hmc_wf

DIFFPREP fits a signal model over arbitrary q-space and corrects **head motion
and eddy currents on shelled and non-shelled data** (where FSL eddy cannot run).
Susceptibility distortion correction is performed with **TORTOISE-native** tools
where possible:

* reverse-PE (``epi`` / ``rpe_series``) -> DRBUDDI (:func:`init_drbuddi_wf`)
* GRE / phase fieldmaps -> qsiprep's fieldmap machinery (:func:`init_sdc_wf`)
* fieldmap-less with a T2w -> TORTOISE ``--epi T2Wreg`` (baked in), else SyN
* nothing -> HMC only

This mirrors the SDC coverage of :func:`~qsiprep.workflows.dwi.fsl.init_fsl_hmc_wf`.
"""

import json
from importlib.resources import files

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from ... import config
from ...interfaces.gradients import ExtractB0s, SliceQC
from ...interfaces.nilearn import EnhanceB0
from ...interfaces.tortoise import (
    DIFFPREP,
    DIFFPREPMotionParams,
    DIFFPREPSplitOutputs,
    SynthesizeDWIs,
    TORTOISEConvert,
    generate_diffprep_boilerplate,
)
from ...utils.resources import as_path
from ..fieldmap.base import init_sdc_wf
from ..fieldmap.drbuddi import init_drbuddi_wf
from .util import init_dwi_reference_wf

# BIDS PhaseEncodingDirection axes already match what TORTOISEProcess expects
# in its own JSON input -- TORTOISE consumes "i", "j", "k" (with optional "-")
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
    (anterior-posterior) when missing or not recognised -- which is what most
    clinical DWI protocols use."""
    if pe_dir in _VALID_PE:
        return pe_dir
    return 'j'


def _zeros_like_b0(b0_template, cwd=None):
    """Write a zeros placeholder CNR map on the b0-template grid.

    DIFFPREP does not emit a CNR map, but downstream resampling feeds
    ``outputnode.cnr_map`` into a mandatory ApplyTransforms input.
    """
    import os

    import nibabel as nb
    import numpy as np

    cwd = cwd or os.getcwd()
    img = nb.load(b0_template)
    out = os.path.join(cwd, 'diffprep_cnr_placeholder.nii.gz')
    nb.Nifti1Image(np.zeros(img.shape[:3], dtype='float32'), img.affine, img.header).to_filename(
        out
    )
    return out


def _write_sidecar_json(nii_file, phase_encoding_direction, working_dir=None):
    """Function-node helper. Writes a BIDS-style JSON sidecar with the same
    basename as ``nii_file`` so TORTOISEProcess can read PhaseEncodingDirection
    from it.

    The JSON is written into **this node's own working directory** (the
    function-node cwd), NOT into the directory ``nii_file`` lives in. Only the
    basename of ``nii_file`` is used (so the sidecar stem matches the ``.nii``
    TORTOISE looks up); its directory is intentionally discarded. Writing into
    the node's own cwd keeps each node's outputs inside its own directory, so
    the standard nipype ``copyfile=True`` propagation to the ``diffprep`` node
    carries a valid sidecar regardless of how the upstream node's cache was
    cleared.
    """
    import json
    import os
    import os.path as op

    base = op.basename(nii_file)
    if base.endswith('.nii.gz'):
        base = base[: -len('.nii.gz')]
    elif base.endswith('.nii'):
        base = base[: -len('.nii')]
    out_dir = working_dir if working_dir else os.getcwd()
    json_file = op.abspath(op.join(out_dir, base + '.json'))
    with open(json_file, 'w') as fobj:
        json.dump({'PhaseEncodingDirection': phase_encoding_direction}, fobj)
    return json_file


def init_diffprep_hmc_wf(
    scan_groups,
    source_file,
    t2w_sdc,
    correction_mode='quadratic',
    dwi_metadata=None,
    name='diffprep_hmc_wf',
):
    """HMC + SDC workflow that uses TORTOISEV4 DIFFPREP for motion + eddy
    correction, and TORTOISE-native SDC (DRBUDDI / T2Wreg) or qsiprep's own
    fieldmap machinery.

    Drop-in peer of :func:`~qsiprep.workflows.dwi.fsl.init_fsl_hmc_wf` with an
    identical inputnode/outputnode contract. The TORTOISE binary writes the
    corrected DWI directly, so this workflow follows the same "bake the
    correction in, emit identity per-volume affines" pattern that
    ``init_fsl_hmc_wf`` uses for FSL eddy.

    Parameters
    ----------
    scan_groups : dict
        Same scan-groups dict the other HMC backends consume.
    source_file : str
        Path to the source DWI file (used for report naming).
    t2w_sdc : bool
        Whether a T2w image is available for distortion correction (used for
        DRBUDDI's multi-modal registration and for the fieldmap-less T2Wreg
        path).
    correction_mode : str
        One of ``'motion'`` (rigid only), ``'quadratic'`` (recommended), or
        ``'cubic'``. Forwarded to TORTOISE as ``-c``.
    dwi_metadata : dict, optional
        BIDS sidecar metadata (used for the PE direction and for SDC).
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
                'original_files',
                't1_brain',
                't1_mask',
                't1_2_mni_reverse_transform',
                't2w_unfatsat',
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
            ],
        ),
        name='outputnode',
    )

    fieldmap_info = scan_groups['fieldmap_info']
    fieldmap_type = fieldmap_info['suffix']

    # TORTOISE-native T2Wreg replaces SyN for the fieldmap-less case when a T2w
    # structural is available: DIFFPREP runs the ``--epi T2Wreg`` stage in the
    # same TORTOISEProcess call and bakes the correction into its output.
    is_fieldmapless = fieldmap_type is None or fieldmap_type == 'syn'
    use_t2wreg = is_fieldmapless and bool(t2w_sdc)
    epi_mode = 'T2Wreg' if use_t2wreg else 'off'

    # Load any user-supplied DIFFPREP config (or our defaults)
    diffprep_cfg = _load_diffprep_config(config.workflow.diffprep_config)

    # Convert gzipped niftis + FSL gradients into TORTOISE format (.nii + .bmtxt).
    tortoise_convert = pe.Node(TORTOISEConvert(), name='tortoise_convert')

    # TORTOISE reads PhaseEncodingDirection from a BIDS-style JSON next to the
    # .nii. Generate one so DIFFPREP (and T2Wreg) can pick the right phase axis.
    write_pe_json = pe.Node(
        niu.Function(
            input_names=['nii_file', 'phase_encoding_direction', 'working_dir'],
            output_names=['json_file'],
            function=_write_sidecar_json,
        ),
        name='write_pe_json',
    )
    pe_dir = _resolve_phase_encoding((dwi_metadata or {}).get('PhaseEncodingDirection'))
    write_pe_json.inputs.phase_encoding_direction = pe_dir

    diffprep = pe.Node(
        DIFFPREP(
            correction_mode=correction_mode,
            b0_id=diffprep_cfg['b0_id'],
            is_human_brain=diffprep_cfg['is_human_brain'],
            rot_eddy_center=diffprep_cfg['rot_eddy_center'],
            extra_args=diffprep_cfg['extra_args'],
            epi_mode=epi_mode,
        ),
        name='diffprep',
        n_procs=config.nipype.omp_nthreads,
    )

    split_outputs = pe.Node(
        DIFFPREPSplitOutputs(b0_threshold=config.workflow.b0_threshold),
        name='split_outputs',
    )

    motion_params = pe.Node(DIFFPREPMotionParams(), name='motion_params')

    # Build a pre-SDC template from the corrected b=0 series for the report.
    extract_b0s = pe.Node(
        ExtractB0s(b0_threshold=config.workflow.b0_threshold),
        name='extract_b0s',
    )
    enhance_pre_sdc = pe.Node(EnhanceB0(), name='enhance_pre_sdc')

    # A "true" b0 reference for downstream coregistration (matches fsl path).
    b0_ref_for_coreg = init_dwi_reference_wf(
        gen_report=False,
        desc='b0_for_coreg',
        name='b0_ref_for_coreg',
        source_file=source_file,
    )

    # Slice-wise QC for the carpet plot: fit a MAPMRI model to the corrected DWI
    # and synthesize an "ideal" volume at every corrected gradient, then score
    # observed-vs-ideal per slice with the same SliceQC node SHORELine uses.
    synth_dwis = pe.Node(
        SynthesizeDWIs(num_threads=config.nipype.omp_nthreads),
        name='synth_dwis',
        n_procs=config.nipype.omp_nthreads,
    )
    slice_qc = pe.Node(SliceQC(), name='slice_qc')

    # Placeholder CNR map (DIFFPREP emits none, but it is a required downstream
    # ApplyTransforms input).
    cnr_placeholder = pe.Node(
        niu.Function(function=_zeros_like_b0, output_names=['cnr_map']),
        name='cnr_placeholder',
    )

    workflow.connect([
        (inputnode, tortoise_convert, [
            ('dwi_file', 'dwi_file'),
            ('bval_file', 'bval_file'),
            ('bvec_file', 'bvec_file'),
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

        # Outputnode plumbing (per-volume corrected data + identity affines)
        (split_outputs, outputnode, [
            ('dwi_files', 'dwi_files_to_transform'),
            ('bvec_files', 'bvec_files_to_transform'),
            ('bval_files', 'bval_files'),
            ('b0_indices', 'b0_indices'),
            ('forward_transforms', 'to_dwi_ref_affines'),
        ]),
        (motion_params, outputnode, [('spm_motion_file', 'motion_params')]),

        # Pre-SDC enhancement (report)
        (diffprep, extract_b0s, [('corrected_dwi_file', 'dwi_series')]),
        (split_outputs, extract_b0s, [('b0_indices', 'b0_indices')]),
        (extract_b0s, enhance_pre_sdc, [('b0_average', 'b0_file')]),
        (enhance_pre_sdc, outputnode, [('enhanced_file', 'pre_sdc_template')]),

        # b0 reference for coregistration
        (extract_b0s, b0_ref_for_coreg, [('b0_average', 'inputnode.b0_template')]),
        (inputnode, b0_ref_for_coreg, [
            ('t1_brain', 'inputnode.t1_brain'),
            ('t1_mask', 'inputnode.t1_mask'),
        ]),
        (b0_ref_for_coreg, outputnode, [('outputnode.dwi_mask', 'b0_template_mask')]),

        # Placeholder CNR map on the corrected b0-template grid
        (extract_b0s, cnr_placeholder, [('b0_average', 'b0_template')]),
        (cnr_placeholder, outputnode, [('cnr_map', 'cnr_map')]),

        # Carpet-plot QC
        (diffprep, synth_dwis, [
            ('corrected_dwi_file', 'dwi_file'),
            ('corrected_bmtxt_file', 'bmtxt_file'),
        ]),
        (b0_ref_for_coreg, synth_dwis, [('outputnode.dwi_mask', 'mask_file')]),
        (split_outputs, slice_qc, [('dwi_files', 'uncorrected_dwi_files')]),
        (synth_dwis, slice_qc, [
            ('per_volume_synth', 'ideal_image_files'),
            ('qc_mask', 'mask_image'),
        ]),
        (slice_qc, outputnode, [('slice_stats', 'slice_quality')]),
    ])  # fmt:skip

    # T2Wreg bakes SDC into the DIFFPREP call: feed the T2w structural.
    if use_t2wreg:
        workflow.connect([
            (inputnode, diffprep, [('t2w_unfatsat', 'structural_image')]),
        ])  # fmt:skip

    # -----------------------------------------------------------------------
    # SDC decision tree (TORTOISE-native where possible)
    # -----------------------------------------------------------------------

    # 1. PEPOLAR (reverse-PE) -> DRBUDDI
    if fieldmap_type in ('epi', 'rpe_series'):
        if 'topup' in config.workflow.pepolar_method.lower():
            raise Exception(
                'TOPUP-based pepolar correction is not supported with '
                '--hmc-model diffprep_*; choose --pepolar-method DRBUDDI.'
            )
        drbuddi_wf = init_drbuddi_wf(scan_groups=scan_groups, t2w_sdc=t2w_sdc)

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

    # 2. Fieldmap-less with a T2w -> TORTOISE T2Wreg (already baked into DIFFPREP)
    if use_t2wreg:
        outputnode.inputs.sdc_method = 'T2Wreg'
        workflow.connect([
            (b0_ref_for_coreg, outputnode, [('outputnode.ref_image', 'b0_template')]),
        ])  # fmt:skip
        return workflow

    # 3. GRE / phase fieldmaps, or SyN fallback (no T2w) -> qsiprep's init_sdc_wf.
    #    The warp is applied downstream (to_dwi_ref_warps), decoupled from HMC.
    if fieldmap_type in ('fieldmap', 'syn') or (
        fieldmap_type is not None and fieldmap_type.startswith('phase')
    ):
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

    # 4. No fieldmap, no T2w -> HMC only.
    outputnode.inputs.sdc_method = 'None'
    workflow.connect([
        (b0_ref_for_coreg, outputnode, [('outputnode.ref_image', 'b0_template')]),
    ])  # fmt:skip

    return workflow
