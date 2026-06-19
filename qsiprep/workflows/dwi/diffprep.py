"""TORTOISE DIFFPREP head-motion/eddy-current correction workflow."""

import os
import shutil

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from ... import config
from ...data import load as load_data
from ...interfaces.gradients import ExtractB0s
from ...interfaces.images import ConformDwi, SplitDWIsFSL
from ...interfaces.tortoise import (
    BmatToFSLGradients,
    DIFFPREPMotionParams,
    TORTOISEConvert,
    TORTOISEProcess,
    generate_diffprep_boilerplate,
)
from .util import init_dwi_reference_wf

DEFAULT_MEMORY_MIN_GB = 0.01


def _n_vols(dwi_files):
    """Return the number of volumes given a list of DWI files."""
    return len(dwi_files) if isinstance(dwi_files, (list, tuple)) else 1


def _identity_itk_transforms(n_volumes, cwd=None):
    """Write ``n_volumes`` identity ITK affine .mat files (bake-in: no re-move)."""

    cwd = cwd or os.getcwd()
    text = (
        '#Insight Transform File V1.0\n'
        '#Transform 0\n'
        'Transform: MatrixOffsetTransformBase_double_3_3\n'
        'Parameters: 1 0 0 0 1 0 0 0 1 0 0 0\n'
        'FixedParameters: 0 0 0\n'
    )
    out = []
    for i in range(n_volumes):
        path = os.path.join(cwd, f'identity_{i:04d}.mat')
        with open(path, 'w') as fobj:
            fobj.write(text)
        out.append(path)
    return out


def _zeros_like_b0(b0_template, cwd=None):
    """Write a zeros placeholder CNR map on the b0-template grid."""

    import nibabel as nb
    import numpy as np

    cwd = cwd or os.getcwd()
    img = nb.load(b0_template)
    out = os.path.join(cwd, 'diffprep_cnr_placeholder.nii.gz')
    nb.Nifti1Image(np.zeros(img.shape[:3], dtype='float32'), img.affine, img.header).to_filename(
        out
    )
    return out


def init_diffprep_hmc_wf(
    scan_groups,
    source_file,
    t2w_sdc,
    dwi_metadata=None,
    transformation_type='quadratic',
    name='diffprep_hmc_wf',
):
    """Motion + eddy-current correction with TORTOISE DIFFPREP (HMC-only).

    Drop-in peer of :func:`~qsiprep.workflows.dwi.fsl.init_fsl_hmc_wf`. The
    TORTOISE-resampled DWI and TORTOISE-rotated gradients are emitted directly
    ("bake-in"); per-volume affines are identity and ``sdc_method`` is ``'None'``.
    """
    if shutil.which('TORTOISEProcess') is None:
        raise RuntimeError(
            'TORTOISEProcess executable not found on PATH. The DIFFPREP HMC backend '
            'requires TORTOISE v4 (bundled in the qsiprep container).'
        )

    workflow = Workflow(name=name)
    workflow.__desc__ = generate_diffprep_boilerplate(transformation_type)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'dwi_file',
                'bvec_file',
                'bval_file',
                'json_file',
                'b0_indices',
                'b0_images',
                'original_files',
                't1_brain',
                't1_mask',
                't1_seg',
                't1_2_mni_reverse_transform',
                't2w_unfatsat',
            ]
        ),
        name='inputnode',
    )

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'pre_sdc_template',
                'bval_files',
                'hmc_optimization_data',
                'sdc_method',
                'slice_quality',
                'motion_params',
                'cnr_map',
                'bvec_files_to_transform',
                'dwi_files_to_transform',
                'b0_indices',
                'to_dwi_ref_affines',
                'to_dwi_ref_warps',
                'b0_template',
                'b0_template_mask',
            ]
        ),
        name='outputnode',
    )
    # HMC-only: no susceptibility correction here.
    outputnode.inputs.sdc_method = 'None'

    # 1. Convert denoised LPS+ DWI to TORTOISE format (.nii + .bmtxt + mask.nii)
    convert = pe.Node(TORTOISEConvert(), name='convert')

    # 2. Run DIFFPREP (motion + optional eddy)
    diffprep = pe.Node(
        TORTOISEProcess(transformation_type=transformation_type),
        name='diffprep',
        n_procs=config.nipype.omp_nthreads,
    )
    diffprep.inputs.config_file = str(
        load_data(config.workflow.diffprep_config or 'diffprep_params.json')
    )

    # 3. Bring the corrected series back to LPS+ and recover FSL gradients
    back_to_lps = pe.Node(ConformDwi(orientation='LPS'), name='back_to_lps')
    bmat_to_fsl = pe.Node(BmatToFSLGradients(), name='bmat_to_fsl')

    # 4. Split into per-volume files (bake-in outputs)
    split = pe.Node(
        SplitDWIsFSL(b0_threshold=config.workflow.b0_threshold, deoblique_bvecs=True),
        name='split',
    )

    # 4b. Pre-HMC b0 reference for DWI-space brain mask (fed to TORTOISEConvert)
    pre_hmc_extract_b0s = pe.Node(
        ExtractB0s(b0_threshold=config.workflow.b0_threshold), name='pre_hmc_extract_b0s'
    )
    pre_hmc_b0_ref = init_dwi_reference_wf(name='pre_hmc_b0_ref', gen_report=False)

    # 5. b0 template + mask
    # NOTE: ExtractB0s requires either bval_file or b0_indices to identify b0 volumes.
    # We use the corrected bval from bmat_to_fsl (consistent with back_to_lps series).
    extract_b0s = pe.Node(
        ExtractB0s(b0_threshold=config.workflow.b0_threshold), name='extract_b0s'
    )
    b0_ref = init_dwi_reference_wf(name='b0_ref', gen_report=False)

    # 6. Motion params and identity affines
    motion = pe.Node(DIFFPREPMotionParams(), name='motion')
    identity = pe.Node(
        niu.Function(function=_identity_itk_transforms, output_names=['out']),
        name='identity',
    )

    n_vols = pe.Node(niu.Function(function=_n_vols, output_names=['n_volumes']), name='n_vols')

    # Placeholder CNR map: DIFFPREP does not emit a CNR map, but downstream
    # resampling feeds outputnode.cnr_map into a mandatory ApplyTransforms input.
    cnr_placeholder = pe.Node(
        niu.Function(function=_zeros_like_b0, output_names=['cnr_map']),
        name='cnr_placeholder',
    )

    workflow.connect([
        # Pre-HMC b0 extraction and DWI-space brain mask (for TORTOISEConvert)
        (inputnode, pre_hmc_extract_b0s, [
            ('dwi_file', 'dwi_series'),
            ('bval_file', 'bval_file'),
        ]),
        (pre_hmc_extract_b0s, pre_hmc_b0_ref, [('b0_average', 'inputnode.b0_template')]),
        # Convert DWI to TORTOISE format; use DWI-space brain mask from pre-HMC b0 reference
        (inputnode, convert, [
            ('dwi_file', 'dwi_file'),
            ('bval_file', 'bval_file'),
            ('bvec_file', 'bvec_file'),
        ]),
        (pre_hmc_b0_ref, convert, [('outputnode.dwi_mask', 'mask_file')]),
        # Run DIFFPREP
        (convert, diffprep, [
            ('dwi_file', 'dwi_file'),
            ('bmtxt_file', 'bmtxt_file'),
            ('mask_file', 'mask_file'),
        ]),
        # Conform corrected DWI back to LPS+; pass gradients through ConformDwi so any
        # axis flip it applies to the image is also applied to the bvecs (keeps the
        # gradients consistent with the conformed image orientation).
        (diffprep, back_to_lps, [('corrected_dwi', 'dwi_file')]),
        # Recover FSL gradients from TORTOISE-rotated b-matrix
        (diffprep, bmat_to_fsl, [('corrected_bmtxt', 'bmtxt_file')]),
        (bmat_to_fsl, back_to_lps, [
            ('bval_file', 'bval_file'),
            ('bvec_file', 'bvec_file'),
        ]),
        # Split corrected DWI into per-volume files using conformed gradients
        (back_to_lps, split, [
            ('dwi_file', 'dwi_file'),
            ('bval_file', 'bval_file'),
            ('bvec_file', 'bvec_file'),
        ]),
        # Per-volume outputs
        (split, outputnode, [
            ('dwi_files', 'dwi_files_to_transform'),
            ('bval_files', 'bval_files'),
            ('bvec_files', 'bvec_files_to_transform'),
            ('b0_indices', 'b0_indices'),
        ]),
        # Identity affines (bake-in: TORTOISE already resampled everything)
        (split, n_vols, [('dwi_files', 'dwi_files')]),
        (n_vols, identity, [('n_volumes', 'n_volumes')]),
        (identity, outputnode, [('out', 'to_dwi_ref_affines')]),
        # Extract b0s using the conformed bval from back_to_lps so b0 detection
        # matches the conformed series. ExtractB0s requires bval_file or b0_indices.
        (back_to_lps, extract_b0s, [
            ('dwi_file', 'dwi_series'),
            ('bval_file', 'bval_file'),
        ]),
        # b0 reference workflow
        (extract_b0s, b0_ref, [('b0_average', 'inputnode.b0_template')]),
        (b0_ref, outputnode, [
            ('outputnode.ref_image', 'b0_template'),
            ('outputnode.dwi_mask', 'b0_template_mask'),
        ]),
        (extract_b0s, outputnode, [('b0_average', 'pre_sdc_template')]),
        # Placeholder CNR map on the corrected b0-template grid (required downstream)
        (extract_b0s, cnr_placeholder, [('b0_average', 'b0_template')]),
        (cnr_placeholder, outputnode, [('cnr_map', 'cnr_map')]),
        # Motion parameters
        (diffprep, motion, [('transforms_file', 'transforms_file')]),
        (motion, outputnode, [('motion_file', 'motion_params')]),
    ])  # fmt:skip

    return workflow
