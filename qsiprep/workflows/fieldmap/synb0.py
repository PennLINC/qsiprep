# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Synthetic distortion-free b=0 generation (SynB0-DISCO).

Synthetic distortion-free b=0 generation (SynB0-DISCO)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generates an undistorted synthetic b=0 from the T1w with the SynB0-DISCO
dual-channel U-Net [Schilling2019]_ [Schilling2020]_, for consumption by
distortion-correction methods that need a distortion-free target: TOPUP (as
the zero-readout volume of a "pepolar" pair) and the TORTOISE registration
backends (as the structural target).

.. [Schilling2019] Schilling et al. (2019) Synthesized b0 for diffusion
   distortion correction (Synb0-DisCo). Magnetic Resonance Imaging 64.
.. [Schilling2020] Schilling et al. (2020) Distortion correction of diffusion
   weighted MRI without reverse phase-encoding scans or field-maps.
   PLOS ONE 15(7).
"""

from nipype.interfaces import ants
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from nireports.interfaces.reporting.base import (
    SimpleBeforeAfterRPT as SimpleBeforeAfter,
)
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from ... import config
from ...interfaces.eddy import Synb0TopupInputs
from ...interfaces.fmap import (
    ModulateByWarpJacobian,
    ParallelTOPUP,
    PostHmcTopupInputs,
    TopupFieldToWarp,
)
from ...interfaces.images import ExtractWM
from ...interfaces.synb0 import (
    NormalizeForSynb0,
    Synb0FieldQC,
    Synb0Inference,
    Synb0QC,
    get_synb0_atlas,
    get_synb0_atlas_mask,
    get_synb0_dir,
)
from ..dwi.registration import init_b0_to_anat_registration_wf
from ..dwi.util import init_dwi_reference_wf


def init_synb0_wf(name='synb0_wf'):
    """Generate a synthetic distortion-free b=0 in the distorted b=0's grid.

    The U-Net consumes a FreeSurfer-style normalized T1w and the distorted
    b=0, both on the 2.5mm atlas grid it was trained on (77x91x77, MNI ICBM
    2009c asymmetric - the same physical space as qsiprep's anatomical
    template). Instead of registering to that atlas, the workflow reuses the
    full affine that AC-PC alignment already derived: composing it with the
    inverse AC-PC rigid maps AC-PC space linearly into the atlas, so the only
    registration run here is the distorted b=0 to the AC-PC T1w.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.fieldmap.synb0 import init_synb0_wf
        wf = init_synb0_wf()

    Inputs
    ------
    t1_preproc
        Bias-corrected T1w head image in AC-PC space
    t1_brain
        Skull-stripped ``t1_preproc``
    t1_seg
        Tissue segmentation of ``t1_preproc`` (white matter label 3)
    b0_ref
        Distorted b=0 reference (head), native grid
    b0_ref_brain
        Skull-stripped ``b0_ref``
    to_template_affine_transform
        Full affine to the template that AC-PC alignment was extracted from
    acpc_inv_transform
        Inverse of the AC-PC rigid transform
    atlas_image
        The 2.5mm SynB0 atlas defining the U-Net grid (prefilled in the
        containers from ``SYNB0_ATLASES``)
    output_grid
        The DWI output-space grid (``dwi_sampling_grid``); when connected,
        the synthetic b=0 is also produced in that space for the derivatives

    Outputs
    -------
    synthetic_b0
        Distortion-free synthetic b=0 on the ``b0_ref`` grid
    synthetic_b0_acpc
        The same image on ``output_grid`` (AC-PC space, aligned with the
        preprocessed DWIs through the anatomical transforms alone)
    b0_to_anat_transform
        Rigid transform from ``b0_ref`` to the AC-PC T1w (ITK format)
    anat_to_b0_transform
        Its inverse
    t1_atlas_space
        The normalized T1w on the U-Net grid (for QC)
    b0_atlas_space
        The distorted b=0 on the U-Net grid (for QC)
    synthetic_b0_atlas_space
        The U-Net output before resampling back (for QC)
    acquired_synthetic_report
        Reportlet flickering acquired vs synthetic b=0 (native grid, WM
        contours from the anatomical segmentation)
    unet_input_report
        Reportlet flickering the two U-Net input channels on the atlas grid

    """
    workflow = Workflow(name=name)
    # Callers describe the correction that uses the synthetic b=0 before this
    # text (TOPUP, unpile) or after it (SyN), so it reads either way.
    workflow.__desc__ = """\
The synthetic distortion-free b=0 image used for susceptibility distortion
correction was generated from the T1w image and the distorted b=0 reference
using the SynB0-DISCO dual-channel U-Net
[@synb0disco; @synb0discoValidation]. The T1w image was scaled to the
FreeSurfer intensity convention (white matter median at 110), the distorted
b=0 reference was rigidly registered to the T1w image, both inputs were
mapped onto the U-Net's 2.5mm atlas grid through that registration and the
AC-PC affine registration, and the mean prediction over the model folds was
resampled back onto the b=0 reference grid.
"""

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                't1_preproc',
                't1_brain',
                't1_seg',
                'b0_ref',
                'b0_ref_brain',
                'to_template_affine_transform',
                'acpc_inv_transform',
                'atlas_image',
                'atlas_mask',
                'output_grid',
            ]
        ),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'synthetic_b0',
                'synthetic_b0_acpc',
                'qc_file',
                'b0_to_anat_transform',
                'anat_to_b0_transform',
                't1_atlas_space',
                'b0_atlas_space',
                'synthetic_b0_atlas_space',
                'acquired_synthetic_report',
                'unet_input_report',
            ]
        ),
        name='outputnode',
    )

    # In the containers the atlas and model locations are known
    atlas_image = get_synb0_atlas()
    if atlas_image is not None:
        inputnode.inputs.atlas_image = atlas_image
    atlas_mask = get_synb0_atlas_mask()
    if atlas_mask is not None:
        inputnode.inputs.atlas_mask = atlas_mask

    normalize_t1 = pe.Node(NormalizeForSynb0(), name='normalize_t1')

    # The distorted b=0 has not been coregistered at this point (SDC precedes
    # the usual b=0-to-anat coregistration), so estimate the rigid here.
    b0_coreg_wf = init_b0_to_anat_registration_wf(
        write_report=False,
        name='distorted_b0_coreg_wf',
    )
    # Described in this workflow's text; its own would repeat the paragraph of
    # the final b=0-to-anatomical coregistration.
    b0_coreg_wf.__desc__ = None

    # ANTs transform lists run the last-listed transform first: images travel
    # b=0 -> AC-PC (coreg) -> pre-AC-PC (inverse rigid) -> atlas (full affine).
    merge_t1_xfms = pe.Node(niu.Merge(2), name='merge_t1_xfms')
    merge_b0_xfms = pe.Node(niu.Merge(3), name='merge_b0_xfms')
    resample_t1_to_atlas = pe.Node(
        ants.ApplyTransforms(dimension=3, interpolation='BSpline'),
        name='resample_t1_to_atlas',
    )
    resample_b0_to_atlas = pe.Node(
        ants.ApplyTransforms(dimension=3, interpolation='BSpline'),
        name='resample_b0_to_atlas',
    )

    unet = pe.Node(Synb0Inference(), name='unet')
    synb0_dir = get_synb0_dir()
    if synb0_dir is not None:
        unet.inputs.synb0_dir = synb0_dir

    # The same chain reversed (every transform inverted, order flipped);
    # all three are linear, so ANTs inverts them on the fly.
    merge_native_xfms = pe.Node(niu.Merge(3), name='merge_native_xfms')
    resample_to_native = pe.Node(
        ants.ApplyTransforms(
            dimension=3,
            interpolation='LanczosWindowedSinc',
            invert_transform_flags=[True, True, True],
        ),
        name='resample_to_native',
    )

    # A second copy on the DWI output grid, for the derivatives: drop the
    # coreg from the chain and the image lands in AC-PC space anchored by the
    # anatomical transforms alone.
    merge_acpc_xfms = pe.Node(niu.Merge(2), name='merge_acpc_xfms')
    resample_to_acpc = pe.Node(
        ants.ApplyTransforms(
            dimension=3,
            interpolation='LanczosWindowedSinc',
            invert_transform_flags=[True, True],
        ),
        name='resample_to_acpc',
    )

    # Reportlets. The acquired/synthetic flicker is deliberately NOT labeled
    # before/after: nothing was corrected - the reader judges whether the
    # synthesized target is trustworthy. WM contours give the undistorted
    # anatomical truth in both states. ``itk_t1_to_b0`` is the coregistration
    # affine listed for reverse use, so resampling the segmentation onto the
    # b=0 grid applies it inverted.
    map_dseg_to_b0 = pe.Node(
        ants.ApplyTransforms(
            dimension=3, interpolation='MultiLabel', invert_transform_flags=[True]
        ),
        name='map_dseg_to_b0',
    )
    extract_wm = pe.Node(ExtractWM(), name='extract_wm')
    acquired_synthetic_rpt = pe.Node(
        SimpleBeforeAfter(before_label='Acquired b=0', after_label='Synthetic b=0'),
        name='acquired_synthetic_rpt',
        mem_gb=0.1,
    )
    # The two U-Net input channels must be mutually aligned; flickering them
    # against each other shows misregistration one stage before the output.
    unet_input_rpt = pe.Node(
        SimpleBeforeAfter(before_label='T1w (normalized)', after_label='Distorted b=0'),
        name='unet_input_rpt',
        mem_gb=0.1,
    )

    synb0_qc = pe.Node(Synb0QC(), name='synb0_qc')

    workflow.connect([
        (inputnode, normalize_t1, [
            ('t1_preproc', 't1w_file'),
            ('t1_seg', 'dseg_file'),
        ]),
        (inputnode, b0_coreg_wf, [
            ('b0_ref_brain', 'inputnode.ref_b0_brain'),
            ('t1_brain', 'inputnode.t1_brain'),
            ('t1_seg', 'inputnode.t1_seg'),
        ]),
        (b0_coreg_wf, outputnode, [
            ('outputnode.itk_b0_to_t1', 'b0_to_anat_transform'),
            ('outputnode.itk_t1_to_b0', 'anat_to_b0_transform'),
        ]),

        # T1 (AC-PC space) onto the atlas grid
        (inputnode, merge_t1_xfms, [
            ('to_template_affine_transform', 'in1'),
            ('acpc_inv_transform', 'in2'),
        ]),
        (normalize_t1, resample_t1_to_atlas, [('out_file', 'input_image')]),
        (inputnode, resample_t1_to_atlas, [('atlas_image', 'reference_image')]),
        (merge_t1_xfms, resample_t1_to_atlas, [('out', 'transforms')]),
        (resample_t1_to_atlas, outputnode, [('output_image', 't1_atlas_space')]),

        # Distorted b=0 (native grid) onto the atlas grid
        (inputnode, merge_b0_xfms, [
            ('to_template_affine_transform', 'in1'),
            ('acpc_inv_transform', 'in2'),
        ]),
        (b0_coreg_wf, merge_b0_xfms, [('outputnode.itk_b0_to_t1', 'in3')]),
        (inputnode, resample_b0_to_atlas, [
            ('b0_ref', 'input_image'),
            ('atlas_image', 'reference_image'),
        ]),
        (merge_b0_xfms, resample_b0_to_atlas, [('out', 'transforms')]),
        (resample_b0_to_atlas, outputnode, [('output_image', 'b0_atlas_space')]),

        # The U-Net itself
        (resample_t1_to_atlas, unet, [('output_image', 't1_file')]),
        (resample_b0_to_atlas, unet, [('output_image', 'b0_file')]),
        (unet, outputnode, [('out_file', 'synthetic_b0_atlas_space')]),

        # Back onto the native b=0 grid
        (b0_coreg_wf, merge_native_xfms, [('outputnode.itk_b0_to_t1', 'in1')]),
        (inputnode, merge_native_xfms, [
            ('acpc_inv_transform', 'in2'),
            ('to_template_affine_transform', 'in3'),
        ]),
        (unet, resample_to_native, [('out_file', 'input_image')]),
        (inputnode, resample_to_native, [('b0_ref', 'reference_image')]),
        (merge_native_xfms, resample_to_native, [('out', 'transforms')]),
        (resample_to_native, outputnode, [('output_image', 'synthetic_b0')]),

        # Onto the DWI output grid
        (inputnode, merge_acpc_xfms, [
            ('acpc_inv_transform', 'in1'),
            ('to_template_affine_transform', 'in2'),
        ]),
        (unet, resample_to_acpc, [('out_file', 'input_image')]),
        (inputnode, resample_to_acpc, [('output_grid', 'reference_image')]),
        (merge_acpc_xfms, resample_to_acpc, [('out', 'transforms')]),
        (resample_to_acpc, outputnode, [('output_image', 'synthetic_b0_acpc')]),

        # Reportlets
        (inputnode, map_dseg_to_b0, [
            ('t1_seg', 'input_image'),
            ('b0_ref', 'reference_image'),
        ]),
        (b0_coreg_wf, map_dseg_to_b0, [('outputnode.itk_t1_to_b0', 'transforms')]),
        (map_dseg_to_b0, extract_wm, [('output_image', 'in_seg')]),
        (inputnode, acquired_synthetic_rpt, [('b0_ref', 'before')]),
        (resample_to_native, acquired_synthetic_rpt, [('output_image', 'after')]),
        (extract_wm, acquired_synthetic_rpt, [('out', 'wm_seg')]),
        (acquired_synthetic_rpt, outputnode, [('out_report', 'acquired_synthetic_report')]),
        (resample_t1_to_atlas, unet_input_rpt, [('output_image', 'before')]),
        (resample_b0_to_atlas, unet_input_rpt, [('output_image', 'after')]),
        (unet_input_rpt, outputnode, [('out_report', 'unet_input_report')]),

        # Scalar QC, all on the U-Net grid
        (resample_t1_to_atlas, synb0_qc, [('output_image', 't1_atlas')]),
        (resample_b0_to_atlas, synb0_qc, [('output_image', 'b0_atlas')]),
        (unet, synb0_qc, [
            ('out_file', 'synthetic_atlas'),
            ('dispersion_file', 'dispersion_atlas'),
        ]),
        (inputnode, synb0_qc, [
            ('atlas_image', 'atlas_image'),
            ('atlas_mask', 'atlas_mask'),
        ]),
        (normalize_t1, synb0_qc, [
            ('scale_factor', 'normalization_scale'),
            ('clipped_fraction', 'clipped_fraction'),
        ]),
        (b0_coreg_wf, synb0_qc, [('outputnode.coreg_metric', 'coreg_metric')]),
        (synb0_qc, outputnode, [('qc_file', 'qc_file')]),
    ])  # fmt:skip

    return workflow


def init_synb0_topup_wf(
    raw_dwi_file,
    pe_dir,
    readout_time,
    source_file,
    omp_nthreads=1,
    two_pass=True,
    name='synb0_topup_wf',
):
    """Estimate susceptibility distortion with TOPUP from a motion-corrected b=0 and its SynB0.

    The synthetic b=0 is generated from the same motion-corrected b=0 that
    TOPUP corrects, so the pair shares its noise level and contrast; the
    synthetic image joins TOPUP as a zero-readout volume. TOPUP runs in FSL's
    LAS+ orientation and its field is returned as a displacement field on the
    ``b0_template`` grid.

    The U-Net carries part of its input's distortion into the synthetic b=0, so
    TOPUP underestimates the field. With ``two_pass`` the b=0 is unwarped with
    the first field (Jacobian-modulated), a new synthetic b=0 is generated from
    it, and TOPUP runs again on the original b=0 paired with the new synthetic one.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.fieldmap.synb0 import init_synb0_topup_wf
        wf = init_synb0_topup_wf(
            raw_dwi_file='sub-01_dwi.nii.gz',
            pe_dir='j-',
            readout_time=0.05,
            source_file='sub-01_dwi.nii.gz',
        )

    Parameters
    ----------
    raw_dwi_file : str
        The DWI file as acquired; ``pe_dir`` refers to its voxel axes.
    pe_dir : str
        BIDS ``PhaseEncodingDirection``.
    readout_time : float
        BIDS ``TotalReadoutTime`` (s).
    source_file : str
        Naming source for the derivatives.
    two_pass : bool
        Regenerate the synthetic b=0 from the b=0 the first field unwarped and run
        TOPUP again (the default; ``--force one-pass-sdc`` turns it off).

    Inputs
    ------
    b0_template
        Motion-corrected, distorted b=0 average, in the frame the warp is applied in
    t1_preproc, t1_brain, t1_seg, to_template_affine_transform, acpc_inv_transform,
    dwi_sampling_grid
        As for :func:`init_synb0_wf`

    Outputs
    -------
    out_warp
        ITK displacement field on the ``b0_template`` grid
    out_hz
        TOPUP's field (Hz) on the ``b0_template`` grid
    b0_corrected
        ``b0_template`` unwarped with ``out_warp``
    field_qc_file
        :class:`~qsiprep.interfaces.synb0.Synb0FieldQC` table
    synthetic_b0, synthetic_b0_acpc, qc_file, acquired_synthetic_report, unet_input_report
        Passed through from the last pass's :func:`init_synb0_wf`
    """
    workflow = Workflow(name=name)
    workflow.__desc__ = """\
Susceptibility distortion was estimated with TOPUP [@topup] after head motion
correction: the motion-corrected b=0 average was paired with a synthetic
distortion-free b=0 generated from it, which entered TOPUP as an image with
zero readout time.{second_pass} The resulting field was converted to a displacement field
along the phase-encoding axis and applied in the final resampling.
""".format(
        second_pass=(
            ' The b=0 was then unwarped with that field (Jacobian-modulated), a new\n'
            'synthetic b=0 was generated from it, and TOPUP was run again on the original\n'
            'b=0 paired with the new synthetic b=0.'
            if two_pass
            else ''
        )
    )
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'b0_template',
                't1_preproc',
                't1_brain',
                't1_seg',
                'to_template_affine_transform',
                'acpc_inv_transform',
                'dwi_sampling_grid',
            ]
        ),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'out_warp',
                'out_hz',
                'b0_corrected',
                'field_qc_file',
                'synthetic_b0',
                'synthetic_b0_acpc',
                'qc_file',
                'acquired_synthetic_report',
                'unet_input_report',
            ]
        ),
        name='outputnode',
    )

    topup_inputs = pe.Node(
        PostHmcTopupInputs(raw_dwi_file=raw_dwi_file, pe_dir=pe_dir, readout_time=readout_time),
        name='topup_inputs',
    )
    workflow.connect([(inputnode, topup_inputs, [('b0_template', 'b0_file')])])  # fmt:skip

    passes = [
        _add_synb0_topup_pass(workflow, inputnode, topup_inputs, source_file, omp_nthreads, '')
    ]
    if two_pass:
        unwarp_b0, field_to_warp = passes[0][1], passes[0][2]
        modulate = pe.Node(ModulateByWarpJacobian(), name='modulate_unwarped_b0')
        workflow.connect([
            (unwarp_b0, modulate, [('output_image', 'in_file')]),
            (field_to_warp, modulate, [('out_warp', 'warp'), ('pe_axis', 'pe_axis')]),
        ])  # fmt:skip
        passes.append(
            _add_synb0_topup_pass(
                workflow,
                inputnode,
                topup_inputs,
                source_file,
                omp_nthreads,
                '_pass2',
                (modulate, 'out_file'),
            )
        )
    synb0_wf, unwarp_b0, field_to_warp, field_qc = passes[-1]

    workflow.connect([
        (field_to_warp, outputnode, [
            ('out_warp', 'out_warp'),
            ('out_hz', 'out_hz'),
        ]),
        (unwarp_b0, outputnode, [('output_image', 'b0_corrected')]),
        (field_qc, outputnode, [('qc_file', 'field_qc_file')]),
        (synb0_wf, outputnode, [
            ('outputnode.synthetic_b0', 'synthetic_b0'),
            ('outputnode.synthetic_b0_acpc', 'synthetic_b0_acpc'),
            ('outputnode.qc_file', 'qc_file'),
            ('outputnode.acquired_synthetic_report', 'acquired_synthetic_report'),
            ('outputnode.unet_input_report', 'unet_input_report'),
        ]),
    ])  # fmt:skip

    return workflow


def _add_synb0_topup_pass(
    workflow, inputnode, topup_inputs, source_file, omp_nthreads, suffix, unet_b0=None
):
    """One SynB0 + TOPUP pass: synthesize a b=0, run TOPUP on the original b=0 with it.

    ``unet_b0`` is the ``(node, field)`` the synthetic b=0 is generated from;
    ``inputnode.b0_template`` when None. TOPUP always pairs the synthetic b=0 with
    ``topup_inputs``' LAS+ copy of ``b0_template``. Returns the pass's
    ``(synb0_wf, unwarp_b0, field_to_warp, field_qc)``.
    """
    b0_ref_wf = init_dwi_reference_wf(
        gen_report=False,
        desc='b0_for_synb0',
        name=f'synb0_b0_ref_wf{suffix}',
        source_file=source_file,
    )
    synb0_wf = init_synb0_wf(name=f'synb0_wf{suffix}')
    synb0_topup_inputs = pe.Node(Synb0TopupInputs(), name=f'synb0_topup_inputs{suffix}')
    topup = pe.Node(
        ParallelTOPUP(out_field='fieldmap_HZ.nii.gz', scale=1, nthreads=omp_nthreads),
        name=f'topup{suffix}',
        n_procs=omp_nthreads,
    )
    field_to_warp = pe.Node(TopupFieldToWarp(), name=f'field_to_warp{suffix}')
    unwarp_b0 = pe.Node(
        ants.ApplyTransforms(dimension=3, interpolation='LanczosWindowedSinc', float=True),
        name=f'unwarp_b0{suffix}',
    )
    field_qc = pe.Node(Synb0FieldQC(), name=f'field_qc{suffix}')

    if unet_b0 is None:
        unet_b0 = (inputnode, 'b0_template')
    workflow.connect([
        (unet_b0[0], b0_ref_wf, [(unet_b0[1], 'inputnode.b0_template')]),
        (inputnode, synb0_wf, [
            ('t1_preproc', 'inputnode.t1_preproc'),
            ('t1_brain', 'inputnode.t1_brain'),
            ('t1_seg', 'inputnode.t1_seg'),
            ('to_template_affine_transform', 'inputnode.to_template_affine_transform'),
            ('acpc_inv_transform', 'inputnode.acpc_inv_transform'),
            ('dwi_sampling_grid', 'inputnode.output_grid'),
        ]),
        (b0_ref_wf, synb0_wf, [
            ('outputnode.ref_image', 'inputnode.b0_ref'),
            ('outputnode.ref_image_brain', 'inputnode.b0_ref_brain'),
        ]),
        (topup_inputs, synb0_topup_inputs, [
            ('topup_datain', 'topup_datain'),
            ('topup_imain', 'topup_imain'),
        ]),
        (synb0_wf, synb0_topup_inputs, [('outputnode.synthetic_b0', 'synthetic_b0')]),
        (synb0_topup_inputs, topup, [
            ('topup_datain', 'encoding_file'),
            ('topup_imain', 'in_file'),
            ('topup_config', 'config'),
        ]),
        (topup, field_to_warp, [('out_field', 'fieldmap')]),
        (synb0_topup_inputs, field_to_warp, [('topup_datain', 'datain')]),
        (inputnode, field_to_warp, [('b0_template', 'reference')]),
        (inputnode, unwarp_b0, [
            ('b0_template', 'input_image'),
            ('b0_template', 'reference_image'),
        ]),
        (field_to_warp, unwarp_b0, [('out_warp', 'transforms')]),
        (field_to_warp, field_qc, [('out_hz', 'fieldmap')]),
        (b0_ref_wf, field_qc, [('outputnode.dwi_mask', 'mask')]),
        (synb0_topup_inputs, field_qc, [('topup_datain', 'datain')]),
    ])  # fmt:skip
    return synb0_wf, unwarp_b0, field_to_warp, field_qc


def _as_transform_list(value):
    return [value]


def connect_post_hmc_synb0_topup(workflow, unit, inputnode, b0, source_file, has_gradwarp):
    """Add SynB0 + TOPUP after DIFFPREP or SHORELine head motion correction.

    ``b0`` is the ``(node, field)`` of the motion-corrected b=0 average. The
    warp is applied downstream of gradwarp, so with ``has_gradwarp`` it is
    estimated from a gradwarp-corrected copy. ``inputnode`` provides the
    T1w-side inputs of :func:`init_synb0_wf` and ``gradwarp_field``. Returns the
    :func:`init_synb0_topup_wf` workflow, whose ``out_warp`` and ``b0_corrected``
    the caller wires on.
    """
    from ...interfaces import DerivativesDataSink
    from ...utils.sdc import pe_readout_time
    from ..dwi.util import add_synb0_outputs

    readout_time = pe_readout_time(unit)
    if readout_time is None or not unit.pe_dir:
        raise ValueError(
            'SynB0 distortion correction needs the PhaseEncodingDirection and '
            f'TotalReadoutTime of {unit.dwi_files[0]}.'
        )
    synb0_topup_wf = init_synb0_topup_wf(
        raw_dwi_file=unit.dwi_files[0],
        pe_dir=unit.pe_dir,
        readout_time=readout_time,
        source_file=source_file,
        omp_nthreads=config.nipype.omp_nthreads,
        two_pass=config.workflow.two_pass_sdc,
    )
    add_synb0_outputs(workflow, synb0_topup_wf, source_file)
    ds_synb0_field_qc = pe.Node(
        DerivativesDataSink(
            source_file=source_file,
            desc='synb0field',
            suffix='qc',
            extension='.tsv',
        ),
        name='ds_synb0_field_qc',
        run_without_submitting=True,
        mem_gb=0.01,
    )
    workflow.connect([
        (inputnode, synb0_topup_wf, [
            ('t1_preproc', 'inputnode.t1_preproc'),
            ('t1_brain', 'inputnode.t1_brain'),
            ('t1_seg', 'inputnode.t1_seg'),
            ('to_template_affine_transform', 'inputnode.to_template_affine_transform'),
            ('acpc_inv_transform', 'inputnode.acpc_inv_transform'),
            ('dwi_sampling_grid', 'inputnode.dwi_sampling_grid'),
        ]),
        (synb0_topup_wf, ds_synb0_field_qc, [('outputnode.field_qc_file', 'in_file')]),
    ])  # fmt:skip
    if has_gradwarp:
        gradwarp_topup_b0 = pe.Node(
            ants.ApplyTransforms(interpolation='LanczosWindowedSinc', float=True),
            name='gradwarp_topup_b0',
        )
        workflow.connect([
            (b0[0], gradwarp_topup_b0, [
                (b0[1], 'input_image'),
                (b0[1], 'reference_image'),
            ]),
            (inputnode, gradwarp_topup_b0, [
                (('gradwarp_field', _as_transform_list), 'transforms'),
            ]),
            (gradwarp_topup_b0, synb0_topup_wf, [('output_image', 'inputnode.b0_template')]),
        ])  # fmt:skip
    else:
        workflow.connect([(b0[0], synb0_topup_wf, [(b0[1], 'inputnode.b0_template')])])  # fmt:skip
    return synb0_topup_wf
