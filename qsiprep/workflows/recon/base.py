#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
qsiprep base reconstruction workflows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_qsirecon_wf
.. autofunction:: init_single_subject_wf

"""

import sys
import os
import os.path as op
from copy import deepcopy

from nipype import __version__ as nipype_ver
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.utils.filemanip import split_filename

from nilearn import __version__ as nilearn_ver

from ...engine import Workflow
from ...interfaces import (BIDSDataGrabber, BIDSInfo, BIDSFreeSurferDir,
                           SubjectSummary, AboutSummary, DerivativesDataSink)
from ...__about__ import __version__

import logging
from collections import defaultdict
import json
from ...interfaces.dsi_studio import (DSIStudioCreateSrc, DSIStudioGQIReconstruction,
                                      DSIStudioAtlasGraph, DSIStudioExport)
from ...interfaces.utils import GetConnectivityAtlases
from ...interfaces.connectivity import Controllability
from ...interfaces.anatomical import QsiprepAnatomicalIngress
from bids.layout import BIDSLayout
from .build_workflow import init_dwi_recon_workflow

LOGGER = logging.getLogger('nipype.workflow')


def init_qsirecon_wf(subject_list, run_uuid, work_dir, output_dir, recon_input,
                     recon_spec, low_mem, omp_nthreads, bids_dir, name="qsirecon_wf"):
    """
    This workflow organizes the execution of qsiprep, with a sub-workflow for
    each subject.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.recon.base import init_qsirecon_wf
        wf = init_qsirecon_wf(subject_list=['test'],
                              run_uuid='X',
                              work_dir='.',
                              recon_input='.',
                              recon_spec='doctest_spec.json',
                              output_dir='.',
                              bids_dir='.',
                              low_mem=False,
                              omp_nthreads=1)


    Parameters

        subject_list : list
            List of subject labels
        run_uuid : str
            Unique identifier for execution instance
        work_dir : str
            Directory in which to store workflow execution state and temporary
            files
        output_dir : str
            Directory in which to save derivatives
        bids_dir : str
            Root directory of BIDS dataset
        recon_input : str
            Root directory of the output from qsiprep
        recon_spec : str
            Path to a JSON file that specifies how to run reconstruction
        low_mem : bool
            Write uncompressed .nii files in some cases to reduce memory usage
    """
    qsiprep_wf = Workflow(name=name)
    qsiprep_wf.base_dir = work_dir

    reportlets_dir = os.path.join(work_dir, 'reportlets')
    for subject_id in subject_list:
        single_subject_wf = init_single_subject_wf(
            subject_id=subject_id,
            recon_input=recon_input,
            recon_spec=recon_spec,
            name="single_subject_" + subject_id + "_recon_wf",
            reportlets_dir=reportlets_dir,
            output_dir=output_dir,
            bids_dir=bids_dir,
            omp_nthreads=omp_nthreads,
            low_mem=low_mem
            )

        single_subject_wf.config['execution']['crashdump_dir'] = (os.path.join(
            output_dir, "qsirecon", "sub-" + subject_id, 'log', run_uuid))
        for node in single_subject_wf._get_all_nodes():
            node.config = deepcopy(single_subject_wf.config)

        qsiprep_wf.add_nodes([single_subject_wf])

    return qsiprep_wf


def init_single_subject_wf(
        subject_id, name, reportlets_dir, output_dir, bids_dir,
        low_mem, omp_nthreads, recon_input, recon_spec):
    """
    This workflow organizes the reconstruction pipeline for a single subject.
    Reconstruction is performed using a separate workflow for each dwi series.

    Parameters

        subject_id : str
            List of subject labels
        name : str
            Name of workflow
        low_mem : bool
            Write uncompressed .nii files in some cases to reduce memory usage
        omp_nthreads : int
            Maximum number of threads an individual process may use
        reportlets_dir : str
            Directory in which to save reportlets
        output_dir : str
            Directory in which to save derivatives
        bids_dir : str
            Root directory of BIDS dataset
        recon_input : str
            Root directory of the output from qsiprep
        recon_spec : str
            Path to a JSON file that specifies how to run reconstruction
    """
    if name in ('single_subject_wf', 'single_subject_test_recon_wf'):
        # a fake spec
        spec = {"name": "fake",
                "atlases": [],
                "space": "T1w",
                "nodes":[]}
        space = spec['space']
        # for documentation purposes
        dwi_files = ['/made/up/outputs/sub-X_dwi.nii.gz']
        layout = None
    else:
        # If recon_input is specified without qsiprep, check if we can find the subject dir
        subject_dir = 'sub-' + subject_id
        if not op.exists(op.join(recon_input, subject_dir)):
            qp_recon_input = op.join(recon_input, "qsiprep")
            LOGGER.info("%s not in %s, trying recon_input=%s",
                        subject_dir, recon_input, qp_recon_input)
            if not op.exists(op.join(qp_recon_input, subject_dir)):
                raise Exception(
                    "Unable to find subject directory in %s or %s" % (
                        recon_input, qp_recon_input))

        with open(recon_spec, "r") as f:
            spec = json.load(f)
        space = spec['space']
        layout = BIDSLayout(recon_input)
        LOGGER.info("found %s in %s", layout.get(type="dwi", extensions=['nii', 'nii.gz']),
                    recon_input)
        # Get all the output files that are in this space
        dwi_files = [f.filename for f in
                     layout.get(type="dwi", extensions=['nii', 'nii.gz'])
                     if 'space-' + space in f.filename]

    workflow = pe.Workflow(name=spec['name'])
    if len(dwi_files) == 0:
        LOGGER.info("No dwi files found for %s", subject_id)
        return workflow


    anat_src = pe.Node(
        QsiprepAnatomicalIngress(subject_id=subject_id,
                                 recon_input_dir=recon_input),
        name='anat_src')

    # create a processing pipeline for the dwis in each session
    for dwi_file in dwi_files:
        dwi_recon_wf = init_dwi_recon_workflow(dwi_file=dwi_file,
                                               workflow_spec=spec,
                                               reportlets_dir=reportlets_dir,
                                               output_dir=output_dir,
                                               omp_nthreads=omp_nthreads)
        workflow.connect([
            (
                anat_src,
                dwi_recon_wf,
                [
                    ('t1_aparc', 'inputnode.t1_aparc'),
                    ('t1_seg', 'inputnode.t1_seg'),
                    ('t1_aseg', 'inputnode.t1_aseg'),
                    ('t1_brain_mask', 'inputnode.t1_brain_mask'),
                    ('t1_preproc', 'inputnode.t1_preproc'),
                    ('t1_csf_probseg', 'inputnode.t1_csf_probseg'),
                    ('t1_gm_probseg', 'inputnode.t1_gm_probseg'),
                    ('t1_wm_probseg', 'inputnode.t1_wm_probseg'),
                    ('left_inflated_surf', 'inputnode.left_inflated_surf'),
                    ('left_midthickness_surf', 'inputnode.left_midthickness_surf'),
                    ('left_pial_surf', 'inputnode.left_pial_surf'),
                    ('left_smoothwm_surf', 'inputnode.left_smoothwm_surf'),
                    ('right_inflated_surf', 'inputnode.right_inflated_surf'),
                    ('right_midthickness_surf', 'inputnode.right_midthickness_surf'),
                    ('right_pial_surf', 'inputnode.right_pial_surf'),
                    ('right_smoothwm_surf', 'inputnode.right_smoothwm_surf'),
                    ('orig_to_t1_mode_forward_transform',
                     'inputnode.orig_to_t1_mode_forward_transform'),
                    ('t1_2_fsnative_forward_transform',
                     'inputnode.t1_2_fsnative_forward_transform'),
                    ('t1_2_mni_reverse_transform', 'inputnode.t1_2_mni_reverse_transform'),
                    ('t1_2_mni_forward_transform', 'inputnode.t1_2_mni_forward_transform'),
                    ('template_brain_mask', 'inputnode.template_brain_mask'),
                    ('template_preproc', 'inputnode.template_preproc'),
                    ('template_seg', 'inputnode.template_seg'),
                    ('template_csf_probseg', 'inputnode.template_csf_probseg'),
                    ('template_gm_probseg', 'inputnode.template_gm_probseg'),
                    ('template_wm_probseg', 'inputnode.template_wm_probseg')
                ])
        ])

    return workflow
