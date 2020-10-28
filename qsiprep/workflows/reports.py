#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
qsiprep interactive report workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_qsiprep_wf

"""
import logging
from copy import deepcopy

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu

from ..engine import Workflow
from ..interfaces import DerivativesDataSink
from ..interfaces.bids import QsiReconIngress
from ..interfaces.reports import InteractiveReport
from ..utils.bids import collect_data
from .recon.interchange import qsiprep_output_names, input_fields


LOGGER = logging.getLogger('nipype.workflow')


def init_json_preproc_report_wf(subject_list, work_dir, output_dir):
    """
    This workflow creates a json report for the dmriprep-viewer.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        import os
        from qsiprep.workflows.reports import init_json_preproc_report_wf
        wf = init_json_preproc_report_wf(
            subject_list=['qsipreptest'],
            work_dir='.',
            output_dir='.')


    Parameters:

        subject_list : list
            List of subject labels
        work_dir : str
            Directory in which to store workflow execution state and temporary
            files
        output_dir : str
            Directory in which to save derivatives

    """
    qsiprep_wf = Workflow(name='json_reports_wf')
    qsiprep_wf.base_dir = work_dir

    for subject_id in subject_list:
        single_subject_wf = init_single_subject_json_report_wf(
            subject_id=subject_id,
            name="single_subject_" + subject_id + "json_report_wf",
            output_dir=output_dir)

        for node in single_subject_wf._get_all_nodes():
            node.config = deepcopy(single_subject_wf.config)
            qsiprep_wf.add_nodes([single_subject_wf])

    return qsiprep_wf


def init_single_subject_json_report_wf(subject_id, name, output_dir):
    """
    This workflow examines the output of a qsiprep run and creates a json report for
    dmriprep-viewer. These are very useful for batch QC-ing QSIPrep runs.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.reports import init_single_subject_json_report_wf

        wf = init_single_subject_json_report_wf(
            subject_id='test',
            name='single_subject_qsipreptest_wf',
            reportlets_dir='.',
            output_dir='.')

    Parameters

        subject_id : str
            List of subject labels
        name : str
            Name of workflow
        output_dir : str
            Directory in which to read and save derivatives

    """
    if name in ('single_subject_wf', 'single_subject_qsipreptest_wf'):
        # for documentation purposes
        subject_data = {
            't1w': ['/completely/made/up/path/sub-01_T1w.nii.gz'],
            'dwi': ['/completely/made/up/path/sub-01_dwi.nii.gz']
        }
        layout = None
        LOGGER.warning("Building a test workflow")
    else:
        subject_data, layout = collect_data(output_dir, subject_id,
                                            bids_validate=False)
    dwi_files = subject_data['dwi']
    workflow = Workflow(name=name)
    scans_iter = pe.Node(niu.IdentityInterface(fields=['dwi_file']), name='scans_iter')
    scans_iter.iterables = ("dwi_file", dwi_files)
    inputnode = pe.Node(niu.IdentityInterface(fields=input_fields + ['dwi_file']),
                        name='inputnode')
    qsiprep_preprocessed_dwi_data = pe.Node(
        QsiReconIngress(), name="qsiprep_preprocessed_dwi_data")

    # For doctests
    if not name == 'fake':
        scans_iter.inputs.dwi_file = dwi_files

    interactive_report = pe.Node(
        InteractiveReport(),
        name="interactive_report")

    ds_report_json = pe.Node(
        DerivativesDataSink(base_directory=output_dir, suffix='viewer'),
        name='ds_report_json',
        run_without_submitting=True)

    # Connect the collected diffusion data (gradients, etc) to the inputnode
    workflow.connect([
        (scans_iter, qsiprep_preprocessed_dwi_data, ([('dwi_file', 'dwi_file')])),
        (qsiprep_preprocessed_dwi_data, inputnode, [
            (trait, trait) for trait in qsiprep_output_names]),
        (inputnode, interactive_report, [
            ('dwi_file', 'processed_dwi_file'),
            ('confounds_file', 'confounds_file'),
            ('qc_file', 'qc_file'),
            ('mask_file', 'mask_file')]),
        (interactive_report, ds_report_json, [('out_report', 'in_file')])
    ])

    return workflow
