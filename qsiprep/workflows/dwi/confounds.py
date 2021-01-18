# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Calculate dwi confounds
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_dwi_confs_wf

"""
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.algorithms import confounds as nac

from ...engine import Workflow
from ...interfaces import AddTSVHeader, GatherConfounds


DEFAULT_MEMORY_MIN_GB = 0.01


def init_dwi_confs_wf(mem_gb, metadata, impute_slice_threshold, name="dwi_confs_wf"):
    """
    This workflow calculates confounds for a dwi series, and aggregates them
    into a :abbr:`TSV (tab-separated value)` file, for use as nuisance
    regressors in a :abbr:`GLM (general linear model)`.

    The following confounds are calculated, with column headings in parentheses:

    1. Framewise displacement, based on head-motion parameters
       (``framewise_displacement``)
    2. Estimated head-motion parameters, in mm and rad
       (``trans_x``, ``trans_y``, ``trans_z``, ``rot_x``, ``rot_y``, ``rot_z``)

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.dwi.confounds import init_dwi_confs_wf
        wf = init_dwi_confs_wf(
            mem_gb=1,
            metadata={},
            impute_slice_threshold=0)

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
        motion_params
            spm motion params


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
preprocessed DWI: framewise displacement (FD) using the
implementation in *Nipype* [following the definitions by @power_fd_dvars].
The head-motion estimates calculated in the correction step were also
placed within the corresponding confounds file. Slicewise cross correlation
was also calculated.
"""
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['sliceqc_file', 'motion_params', 'bval_file', 'bvec_file', 'original_files',
                'denoising_confounds']),
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
        (inputnode, fdisp, [('motion_params', 'in_file')]),

        # Collate computed confounds together
        (inputnode, add_motion_headers, [('motion_params', 'in_file')]),
        (fdisp, concat, [('out_file', 'fd')]),
        (add_motion_headers, concat, [('out_file', 'motion')]),
        (inputnode, concat, [('sliceqc_file', 'sliceqc_file'),
                             ('bval_file', 'original_bvals'),
                             ('bvec_file', 'original_bvecs'),
                             ('original_files', 'original_files'),
                             ('denoising_confounds', 'denoising_confounds')]),
        # Set outputs
        (concat, outputnode, [('confounds_file', 'confounds_file')]),
    ])

    return workflow
