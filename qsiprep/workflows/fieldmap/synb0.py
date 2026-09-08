# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
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
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from niworkflows.interfaces.reportlets.registration import SimpleBeforeAfterRPT

from ...interfaces.images import ExtractWM
from ...interfaces.synb0 import (
    NormalizeForSynb0,
    Synb0Inference,
    Synb0QC,
    get_synb0_atlas,
    get_synb0_atlas_mask,
    get_synb0_dir,
)
from ..dwi.registration import init_b0_to_anat_registration_wf


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
    workflow.__desc__ = """\
A synthetic distortion-free b=0 image was generated from the T1w image and
the distorted b=0 reference using the SynB0-DISCO dual-channel U-Net
[@synb0disco; @synb0discoValidation]. The T1w image was scaled to the
FreeSurfer intensity convention (white matter median at 110), both inputs
were mapped onto the U-Net's 2.5mm atlas grid through the AC-PC affine
registration, and the mean prediction over the model folds was resampled
back onto the b=0 reference grid.
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
    # anatomical truth in both states.
    map_dseg_to_b0 = pe.Node(
        ants.ApplyTransforms(dimension=3, interpolation='MultiLabel'),
        name='map_dseg_to_b0',
    )
    extract_wm = pe.Node(ExtractWM(), name='extract_wm')
    acquired_synthetic_rpt = pe.Node(
        SimpleBeforeAfterRPT(before_label='Acquired b=0', after_label='Synthetic b=0'),
        name='acquired_synthetic_rpt',
        mem_gb=0.1,
    )
    # The two U-Net input channels must be mutually aligned; flickering them
    # against each other shows misregistration one stage before the output.
    unet_input_rpt = pe.Node(
        SimpleBeforeAfterRPT(before_label='T1w (normalized)', after_label='Distorted b=0'),
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
