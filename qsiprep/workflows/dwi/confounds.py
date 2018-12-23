# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Calculate dwi confounds
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_dwi_confs_wf
.. autofunction:: init_ica_aroma_wf

"""
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu, fsl
from nipype.interfaces.nilearn import SignalExtraction
from nipype.algorithms import confounds as nac

from niworkflows.data import get_template
from niworkflows.interfaces.segmentation import ICA_AROMARPT
from niworkflows.interfaces.masks import ROIsPlot
from niworkflows.interfaces.fixes import FixHeaderApplyTransforms as ApplyTransforms

from fmriprep.engine import Workflow
from ...interfaces import (
    TPM2ROI, AddTPMs, AddTSVHeader, GatherConfounds, ICAConfounds,
    FMRISummary, DerivativesDataSink
)
from ...interfaces.patches import (
    RobustACompCor as ACompCor,
    RobustTCompCor as TCompCor
)

from .resampling import init_dwi_mni_trans_wf

DEFAULT_MEMORY_MIN_GB = 0.01


def init_dwi_confs_wf(mem_gb, metadata, impute_slice_threshold, name="dwi_confs_wf"):
    """
    This workflow calculates confounds for a dwi series, and aggregates them
    into a :abbr:`TSV (tab-separated value)` file, for use as nuisance
    regressors in a :abbr:`GLM (general linear model)`.

    The following confounds are calculated, with column headings in parentheses:

    #. Framewise displacement, based on head-motion parameters
       (``framewise_displacement``)
    #. Estimated head-motion parameters, in mm and rad
       (``trans_x``, ``trans_y``, ``trans_z``, ``rot_x``, ``rot_y``, ``rot_z``)

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.dwi.confounds import init_dwi_confs_wf
        wf = init_dwi_confs_wf(
            mem_gb=1,
            metadata={})

    **Parameters**

        mem_gb : float
            Size of dwi file in GB - please note that this size
            should be calculated after resamplings that may extend
            the FoV
        metadata : dict
            BIDS metadata for dwi file
        name : str
            Name of workflow (default: ``dwi_confs_wf``)

    **Inputs**

        sliceqc_file
            dwi image, after the prescribed corrections (STC, HMC and SDC)
            when available.
        movpar_file
            SPM-formatted motion parameters file

    **Outputs**

        confounds_file
            TSV of all aggregated confounds
        rois_report
            Reportlet visualizing white-matter/CSF mask used for aCompCor,
            the ROI for tCompCor and the dwi brain mask.

    """
    workflow = Workflow(name=name)
    workflow.__desc__ = """\
Several confounding time-series were calculated based on the
*preprocessed dwi*: framewise displacement (FD) using the
implementation in *Nipype* [following the definitions by @power_fd_dvars].
The head-motion estimates calculated in the correction step were also
placed within the corresponding confounds file. Slicewise cross correlation
was also calculated.
"""
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['sliceqc_file', 'movpar_file']),
        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['confounds_file', 'imputed_images']),
        name='outputnode')

    # Frame displacement
    fdisp = pe.Node(nac.FramewiseDisplacement(parameter_source="SPM"),
                    name="fdisp", mem_gb=mem_gb)

    add_motion_headers = pe.Node(
        AddTSVHeader(columns=["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]),
        name="add_motion_headers", mem_gb=0.01, run_without_submitting=True)
    concat = pe.Node(GatherConfounds(), name="concat", mem_gb=0.01, run_without_submitting=True)

    workflow.connect([
        (inputnode, fdisp, [('movpar_file', 'in_file')]),

        # Collate computed confounds together
        (inputnode, add_motion_headers, [('movpar_file', 'in_file')]),
        (fdisp, concat, [('out_file', 'fd')]),
        (add_motion_headers, concat, [('out_file', 'motion')]),

        # Set outputs
        (concat, outputnode, [('confounds_file', 'confounds_file')]),

    ])

    return workflow


def init_sliceqc_wf(mem_gb, metadata, name="dwi_sliceqc_wf"):
    """

    Resamples the MNI parcellation (ad-hoc parcellation derived from the
    Harvard-Oxford template and others).

    **Parameters**

        mem_gb : float
            Size of dwi file in GB - please note that this size
            should be calculated after resamplings that may extend
            the FoV
        metadata : dict
            BIDS metadata for dwi file
        name : str
            Name of workflow (default: ``dwi_carpet_wf``)

    **Inputs**

        dwi
            dwi image, after the prescribed corrections (STC, HMC and SDC)
            when available.
        dwi_mask
            dwi series mask
        confounds_file
            TSV of all aggregated confounds
        t1_dwi_xform
            Affine matrix that maps the T1w space into alignment with
            the native dwi space
        t1_2_mni_reverse_transform
            ANTs-compatible affine-and-warp transform file

    **Outputs**

        out_carpetplot
            Path of the generated SVG file

    """
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['dwi', 'dwi_mask', 'confounds_file',
                't1_dwi_xform', 't1_2_mni_reverse_transform']),
        name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['out_carpetplot']), name='outputnode')

    # List transforms
    mrg_xfms = pe.Node(niu.Merge(2), name='mrg_xfms')

    # Carpetplot and confounds plot
    conf_plot = pe.Node(FMRISummary(
        tr=metadata['RepetitionTime'],
        confounds_list=[
            ('global_signal', None, 'GS'),
            ('csf', None, 'GSCSF'),
            ('white_matter', None, 'GSWM'),
            ('std_dvars', None, 'DVARS'),
            ('framewise_displacement', 'mm', 'FD')]),
        name='conf_plot', mem_gb=mem_gb)
    ds_report_dwi_conf = pe.Node(
        DerivativesDataSink(suffix='carpetplot'),
        name='ds_report_dwi_conf', run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB)

    workflow = Workflow(name=name)
    workflow.connect([
        (inputnode, mrg_xfms, [('t1_dwi_xform', 'in1'),
                               ('t1_2_mni_reverse_transform', 'in2')]),
        (inputnode, resample_parc, [('dwi_mask', 'reference_image')]),
        (mrg_xfms, resample_parc, [('out', 'transforms')]),
        # Carpetplot
        (inputnode, conf_plot, [
            ('dwi', 'in_func'),
            ('dwi_mask', 'in_mask'),
            ('confounds_file', 'confounds_file')]),
        (resample_parc, conf_plot, [('output_image', 'in_segm')]),
        (conf_plot, ds_report_dwi_conf, [('out_file', 'in_file')]),
        (conf_plot, outputnode, [('out_file', 'out_carpetplot')]),
    ])
    return workflow


def init_carpetplot_wf(mem_gb, metadata, name="dwi_carpet_wf"):
    """

    Resamples the MNI parcellation (ad-hoc parcellation derived from the
    Harvard-Oxford template and others).

    **Parameters**

        mem_gb : float
            Size of dwi file in GB - please note that this size
            should be calculated after resamplings that may extend
            the FoV
        metadata : dict
            BIDS metadata for dwi file
        name : str
            Name of workflow (default: ``dwi_carpet_wf``)

    **Inputs**

        dwi
            dwi image, after the prescribed corrections (STC, HMC and SDC)
            when available.
        dwi_mask
            dwi series mask
        confounds_file
            TSV of all aggregated confounds
        t1_dwi_xform
            Affine matrix that maps the T1w space into alignment with
            the native dwi space
        t1_2_mni_reverse_transform
            ANTs-compatible affine-and-warp transform file

    **Outputs**

        out_carpetplot
            Path of the generated SVG file

    """
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['dwi', 'dwi_mask', 'confounds_file',
                't1_dwi_xform', 't1_2_mni_reverse_transform']),
        name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['out_carpetplot']), name='outputnode')

    # List transforms
    mrg_xfms = pe.Node(niu.Merge(2), name='mrg_xfms')

    # Warp segmentation into EPI space
    resample_parc = pe.Node(ApplyTransforms(
        float=True,
        input_image=str(
            get_template('MNI152NLin2009cAsym') /
            'tpl-MNI152NLin2009cAsym_space-MNI_res-01_label-carpet_atlas.nii.gz'),
        dimension=3, default_value=0, interpolation='MultiLabel'),
        name='resample_parc')

    # Carpetplot and confounds plot
    conf_plot = pe.Node(FMRISummary(
        tr=metadata['RepetitionTime'],
        confounds_list=[
            ('global_signal', None, 'GS'),
            ('csf', None, 'GSCSF'),
            ('white_matter', None, 'GSWM'),
            ('std_dvars', None, 'DVARS'),
            ('framewise_displacement', 'mm', 'FD')]),
        name='conf_plot', mem_gb=mem_gb)
    ds_report_dwi_conf = pe.Node(
        DerivativesDataSink(suffix='carpetplot'),
        name='ds_report_dwi_conf', run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB)

    workflow = Workflow(name=name)
    workflow.connect([
        (inputnode, mrg_xfms, [('t1_dwi_xform', 'in1'),
                               ('t1_2_mni_reverse_transform', 'in2')]),
        (inputnode, resample_parc, [('dwi_mask', 'reference_image')]),
        (mrg_xfms, resample_parc, [('out', 'transforms')]),
        # Carpetplot
        (inputnode, conf_plot, [
            ('dwi', 'in_func'),
            ('dwi_mask', 'in_mask'),
            ('confounds_file', 'confounds_file')]),
        (resample_parc, conf_plot, [('output_image', 'in_segm')]),
        (conf_plot, ds_report_dwi_conf, [('out_file', 'in_file')]),
        (conf_plot, outputnode, [('out_file', 'out_carpetplot')]),
    ])
    return workflow
