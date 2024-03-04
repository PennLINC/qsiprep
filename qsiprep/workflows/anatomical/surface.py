# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Anatomical reference preprocessing workflows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_anat_preproc_wf
.. autofunction:: init_skullstrip_ants_wf

"""
from nipype import logging
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe

from ...engine import Workflow
from ...interfaces import DerivativesDataSink as FDerivativesDataSink
from ...utils.misc import fix_multi_T1w_source_name

# from pkg_resources import resource_filename as pkgr


LOGGER = logging.getLogger("nipype.workflow")


class DerivativesDataSink(FDerivativesDataSink):
    out_path_base = "qsiprep"


TEMPLATE_MAP = {
    "MNI152NLin2009cAsym": "mni_icbm152_nlin_asym_09c",
}


def init_anat_reports_wf(
    reportlets_dir,
    output_spaces,
    force_spatial_normalization,
    template,
    freesurfer,
    name="anat_reports_wf",
):
    """
    Set up a battery of datasinks to store reports in the right location
    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "source_file",
                "t1_conform_report",
                "seg_report",
                "t1_2_mni_report",
                "recon_report",
            ]
        ),
        name="inputnode",
    )

    ds_t1_conform_report = pe.Node(
        DerivativesDataSink(base_directory=reportlets_dir, suffix="conform"),
        name="ds_t1_conform_report",
        run_without_submitting=True,
    )

    ds_t1_2_mni_report = pe.Node(
        DerivativesDataSink(base_directory=reportlets_dir, suffix="t1_2_mni"),
        name="ds_t1_2_mni_report",
        run_without_submitting=True,
    )

    ds_t1_seg_mask_report = pe.Node(
        DerivativesDataSink(base_directory=reportlets_dir, suffix="seg_brainmask"),
        name="ds_t1_seg_mask_report",
        run_without_submitting=True,
    )

    ds_recon_report = pe.Node(
        DerivativesDataSink(base_directory=reportlets_dir, suffix="reconall"),
        name="ds_recon_report",
        run_without_submitting=True,
    )

    workflow.connect([
        (inputnode, ds_t1_conform_report, [('source_file', 'source_file'),
                                           ('t1_conform_report', 'in_file')]),
        (inputnode, ds_t1_seg_mask_report, [('source_file', 'source_file'),
                                            ('seg_report', 'in_file')]),
    ])  # fmt:skip

    if freesurfer:
        workflow.connect([
            (inputnode, ds_recon_report, [('source_file', 'source_file'),
                                          ('recon_report', 'in_file')])
        ])  # fmt:skip
    if "template" in output_spaces or force_spatial_normalization:
        workflow.connect([
            (inputnode, ds_t1_2_mni_report, [('source_file', 'source_file'),
                                             ('t1_2_mni_report', 'in_file')])
        ])  # fmt:skip

    return workflow


def init_anat_derivatives_wf(
    output_dir,
    output_spaces,
    template,
    freesurfer,
    force_spatial_normalization,
    name="anat_derivatives_wf",
):
    """
    Set up a battery of datasinks to store derivatives in the right location
    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "source_files",
                "t1_template_transforms",
                "t1_preproc",
                "t1_mask",
                "t1_seg",
                "t1_tpms",
                "t1_2_mni_forward_transform",
                "t1_2_mni_reverse_transform",
                "t1_2_mni",
                "mni_mask",
                "mni_seg",
                "mni_tpms",
                "t1_2_fsnative_forward_transform",
                "surfaces",
                "t1_fs_aseg",
                "t1_fs_aparc",
            ]
        ),
        name="inputnode",
    )

    t1_name = pe.Node(niu.Function(function=fix_multi_T1w_source_name), name="t1_name")

    ds_t1_preproc = pe.Node(
        DerivativesDataSink(base_directory=output_dir, desc="preproc", keep_dtype=True),
        name="ds_t1_preproc",
        run_without_submitting=True,
    )

    ds_t1_mask = pe.Node(
        DerivativesDataSink(base_directory=output_dir, desc="brain", suffix="mask"),
        name="ds_t1_mask",
        run_without_submitting=True,
    )

    ds_t1_seg = pe.Node(
        DerivativesDataSink(base_directory=output_dir, suffix="dseg"),
        name="ds_t1_seg",
        run_without_submitting=True,
    )

    ds_t1_tpms = pe.Node(
        DerivativesDataSink(base_directory=output_dir, suffix="label-{extra_value}_probseg"),
        name="ds_t1_tpms",
        run_without_submitting=True,
    )
    ds_t1_tpms.inputs.extra_values = ["CSF", "GM", "WM"]

    ds_t1_mni = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir, space=template, desc="preproc", keep_dtype=True
        ),
        name="ds_t1_mni",
        run_without_submitting=True,
    )

    ds_mni_mask = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir, space=template, desc="brain", suffix="mask"
        ),
        name="ds_mni_mask",
        run_without_submitting=True,
    )

    ds_mni_seg = pe.Node(
        DerivativesDataSink(base_directory=output_dir, space=template, suffix="dseg"),
        name="ds_mni_seg",
        run_without_submitting=True,
    )

    ds_mni_tpms = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir, space=template, suffix="label-{extra_value}_probseg"
        ),
        name="ds_mni_tpms",
        run_without_submitting=True,
    )
    ds_mni_tpms.inputs.extra_values = ["CSF", "GM", "WM"]

    # Transforms
    suffix_fmt = "from-{}_to-{}_mode-image_xfm".format
    ds_t1_mni_inv_warp = pe.Node(
        DerivativesDataSink(base_directory=output_dir, suffix=suffix_fmt(template, "T1w")),
        name="ds_t1_mni_inv_warp",
        run_without_submitting=True,
    )

    ds_t1_template_transforms = pe.MapNode(
        DerivativesDataSink(base_directory=output_dir, suffix=suffix_fmt("orig", "T1w")),
        iterfield=["source_file", "in_file"],
        name="ds_t1_template_transforms",
        run_without_submitting=True,
    )

    ds_t1_mni_warp = pe.Node(
        DerivativesDataSink(base_directory=output_dir, suffix=suffix_fmt("T1w", template)),
        name="ds_t1_mni_warp",
        run_without_submitting=True,
    )

    workflow.connect([
        (inputnode, t1_name, [('source_files', 'in_files')]),
        (inputnode, ds_t1_template_transforms, [('source_files', 'source_file'),
                                                ('t1_template_transforms', 'in_file')]),
        (inputnode, ds_t1_preproc, [('t1_preproc', 'in_file')]),
        (inputnode, ds_t1_mask, [('t1_mask', 'in_file')]),
        (inputnode, ds_t1_seg, [('t1_seg', 'in_file')]),
        (inputnode, ds_t1_tpms, [('t1_tpms', 'in_file')]),
        (t1_name, ds_t1_preproc, [('out', 'source_file')]),
        (t1_name, ds_t1_mask, [('out', 'source_file')]),
        (t1_name, ds_t1_seg, [('out', 'source_file')]),
        (t1_name, ds_t1_tpms, [('out', 'source_file')]),
    ])  # fmt:skip

    if "template" in output_spaces or force_spatial_normalization:
        workflow.connect([
            (inputnode, ds_t1_mni_warp, [('t1_2_mni_forward_transform', 'in_file')]),
            (inputnode, ds_t1_mni_inv_warp, [('t1_2_mni_reverse_transform', 'in_file')]),
            (inputnode, ds_t1_mni, [('t1_2_mni', 'in_file')]),
            (inputnode, ds_mni_mask, [('mni_mask', 'in_file')]),
            (inputnode, ds_mni_seg, [('mni_seg', 'in_file')]),
            (inputnode, ds_mni_tpms, [('mni_tpms', 'in_file')]),
            (t1_name, ds_t1_mni_warp, [('out', 'source_file')]),
            (t1_name, ds_t1_mni_inv_warp, [('out', 'source_file')]),
            (t1_name, ds_t1_mni, [('out', 'source_file')]),
            (t1_name, ds_mni_mask, [('out', 'source_file')]),
            (t1_name, ds_mni_seg, [('out', 'source_file')]),
            (t1_name, ds_mni_tpms, [('out', 'source_file')]),
        ])  # fmt:skip

    return workflow


def _seg2msks(in_file, newpath=None):
    """Converts labels to masks"""
    import nibabel as nb
    import numpy as np
    from nipype.utils.filemanip import fname_presuffix

    nii = nb.load(in_file)
    labels = nii.get_fdata()

    out_files = []
    for i in range(1, 4):
        ldata = np.zeros_like(labels)
        ldata[labels == i] = 1
        out_files.append(fname_presuffix(in_file, suffix="_label%03d" % i, newpath=newpath))
        nii.__class__(ldata, nii.affine, nii.header).to_filename(out_files[-1])

    return out_files

    # contrast_name = anatomical_contrast.lower()
    # if not infant_mode:
    #     ref_img = pkgr("qsiprep", "data/mni_1mm_%s_lps.nii.gz" % contrast_name)
    #     ref_img_brain = pkgr("qsiprep", "data/mni_1mm_%s_lps_brain.nii.gz" % contrast_name)
    # else:
    #     ref_img = pkgr("qsiprep", "data/mni_1mm_%s_lps_infant.nii.gz" % contrast_name)
    #     ref_img_brain = pkgr("qsiprep",
    #                          "data/mni_1mm_%s_lps_brain_infant.nii.gz" % contrast_name)

    # return ref_img, ref_img_brain
