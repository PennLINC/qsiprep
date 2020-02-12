# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Utility workflows
^^^^^^^^^^^^^^^^^

.. autofunction:: init_dwi_reference_wf

"""
import os
from pathlib import Path
import nibabel as nb
import pkg_resources as pkgr

from nipype.pipeline import engine as pe
from nipype.utils.filemanip import split_filename
from nipype.interfaces import utility as niu, ants
from ...niworkflows.interfaces import SimpleBeforeAfter
from ...engine import Workflow
from ...interfaces import DerivativesDataSink
from ...interfaces.nilearn import EnhanceAndSkullstripB0


DEFAULT_MEMORY_MIN_GB = 0.01


def init_dwi_reference_wf(omp_nthreads=1, dwi_file=None, register_t1=False,
                          name='dwi_reference_wf', gen_report=False, source_file=None,
                          desc="initial"):
    """
    This workflow generates reference b=0 image and a mask.

    The raw reference image is the target of :abbr:`HMC (head motion correction)`, and a
    contrast-enhanced reference is the subject of distortion correction, as well as
    boundary-based registration to T1w and template spaces.

    A skull-stripped T1w image is downsampled to the resolution of the b0 input image
    and registered to it. The T1w mask is used as a starting point for generating
    The b=0 mask.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.dwi.util import init_dwi_reference_wf
        wf = init_dwi_reference_wf(omp_nthreads=1)

    **Parameters**

        dwi_file : str
            A b=0 image
        omp_nthreads : int
            Maximum number of threads an individual process may use
        name : str
            Name of workflow (default: ``dwi_reference_wf``)
        gen_report : bool
            Whether a mask report node should be appended in the end

    **Inputs**

        b0_template
            the b0 template used as the motion correction reference
        t1_brain
            skull-stripped T1w image from the same subject
        t1_mask
            mask image for t1_brain
        wm_seg
            white matter segmentation from the T1w image

    **Outputs**

        raw_ref_image
            Reference image to which dwi series is motion corrected
        ref_image
            Contrast-enhanced reference image
        ref_image_brain
            Skull-stripped reference image
        dwi_mask
            Skull-stripping (rough) mask of reference image
        validation_report
            HTML reportlet indicating whether ``dwi_file`` had a valid affine


    **Subworkflows**

        * :py:func:`~qsiprep.workflows.dwi.util.init_enhance_and_skullstrip_wf`

    """
    workflow = Workflow(name=name)
    workflow.__desc__ = """\
"""
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['b0_template', 't1_brain', 't1_mask', 't1_seg']),
        name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['raw_ref_image', 'ref_image', 'ref_image_brain',
                                      'dwi_mask', 'validation_report']),
        name='outputnode')

    # Simplify manually setting input image
    if dwi_file is not None:
        inputnode.inputs.b0_template = dwi_file

    # b=0 images are too diverse and tricky to reliably mask.
    # Instead register the t1w to the b=0 and use that brain mask
    if register_t1:
        affine_transform = pkgr.resource_filename('qsiprep', 'data/affine.json')
        register_t1_to_raw = pe.Node(ants.Registration(from_file=affine_transform),
                                     name='register_t1_to_raw')
        t1_mask_to_b0 = pe.Node(ants.ApplyTransforms(interpolation='MultiLabel',
                                                     invert_transform_flags=[True]),
                                name='t1_mask_to_b0')
        workflow.connect([
            (inputnode, register_t1_to_raw, [
                ('t1_brain', 'fixed_image'),
                ('t1_mask', 'fixed_image_masks'),
                ('b0_template', 'moving_image')]),
            (register_t1_to_raw, t1_mask_to_b0, [
                ('forward_transforms', 'transforms')])])
    else:
        # T1w is already aligned
        t1_mask_to_b0 = pe.Node(
            ants.ApplyTransforms(transforms='identity'),
            name='t1_mask_to_b0')

    # Do a masking of the DWI by itself
    enhance_and_mask_b0 = pe.Node(EnhanceAndSkullstripB0(), name='enhance_and_mask_b0')

    workflow.connect([
        (inputnode, t1_mask_to_b0, [
            ('t1_mask', 'input_image'),
            ('b0_template', 'reference_image')]),
        (inputnode, outputnode, [('b0_template', 'raw_ref_image')]),
        (inputnode, enhance_and_mask_b0, [('b0_template', 'b0_file')]),
        (t1_mask_to_b0, enhance_and_mask_b0, [('output_image', 't1_mask')]),
        (enhance_and_mask_b0, outputnode, [
            ('enhanced_file', 'ref_image'),
            ('skull_stripped_file', 'ref_image_brain'),
            ('mask_file', 'dwi_mask')])
    ])

    if gen_report:
        if source_file is None:
            raise Exception("Needs a source_file to write a report")
        b0ref_reportlet = pe.Node(SimpleBeforeAfter(), name='b0ref_reportlet', mem_gb=0.1)
        ds_report_b0_mask = pe.Node(
            DerivativesDataSink(desc=desc, suffix='b0ref', source_file=source_file),
            name='ds_report_b0_mask',
            mem_gb=DEFAULT_MEMORY_MIN_GB, run_without_submitting=True
        )

        workflow.connect([
            (inputnode, b0ref_reportlet, [('b0_template', 'before')]),
            (enhance_and_mask_b0, b0ref_reportlet, [
                ('enhanced_file', 'after'),
                ('plotting_mask_file', 'wm_seg')]),
            (b0ref_reportlet, outputnode, [('out_report', 'validation_report')]),
            (b0ref_reportlet, ds_report_b0_mask, [('out_report', 'in_file')])
        ])

    return workflow


def _create_mem_gb(dwi_fname):
    dwi_size_gb = os.path.getsize(dwi_fname) / (1024**3)
    try:
        dwi_nvols = nb.load(dwi_fname).shape[3]
    except IndexError:
        dwi_nvols = 1
    except nb.filebasedimages.ImageFileError:
        dwi_nvols = 1
    mem_gb = {
        'filesize': dwi_size_gb,
        'resampled': dwi_size_gb * 4,
        'largemem': dwi_size_gb * (max(dwi_nvols / 100, 1.0) + 4),
    }

    return dwi_nvols, mem_gb


def _get_concatenated_bids_name(dwi_group):
    """Derive the output name for a dwi grouping."""
    try:
        all_dwis = dwi_group['dwi_series']
        if dwi_group['fieldmap_info']['suffix'] == 'rpe_series':
            all_dwis += dwi_group['fieldmap_info']['rpe_series']
    except Exception:
        all_dwis = dwi_group

    # If a single file, use its name, otherwise use the common prefix
    if len(all_dwis) > 1:
        no_runs = []
        for dwi in all_dwis:
            no_runs.append(
                "_".join([part for part in dwi.split("_")
                          if not part.startswith("run")]))
        input_fname = os.path.commonprefix(no_runs)
        fname = split_filename(input_fname)[1]
        parts = fname.split('_')
        full_parts = [part for part in parts if not part.endswith('-')]
        fname = '_'.join(full_parts)

    else:
        input_fname = all_dwis[0]
        fname = split_filename(input_fname)[1]

    if fname.endswith("_dwi"):
        fname = fname[:-4]

    return fname.replace(".", "").replace(" ", "")


def _get_wf_name(dwi_fname):
    """Derive the workflow name based on the output file prefix."""
    spl = dwi_fname.split("_")
    nosub = "_".join(spl[1:])
    return ("dwi_preproc_" + nosub + "_wf").replace("__", "_").replace("-", "_")


def _list_squeeze(in_list):
    squeezed = []
    for item in in_list:
        if type(item) is not str:
            squeezed.append(item[0])
        else:
            squeezed.append(item)
    return squeezed


def _get_first(in_list):
    return in_list[0]


def get_source_file(dwi_files, output_prefix=None, suffix=''):
    """The reportlets need a source file. This file might not exist in the input data."""
    if output_prefix is None:
        output_prefix = _get_concatenated_bids_name(dwi_files)
    return str(Path(dwi_files[0]).parent / output_prefix) + suffix + ".nii.gz"
