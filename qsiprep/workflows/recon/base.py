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

import os
import os.path as op
from glob import glob
from copy import deepcopy
from nipype import __version__ as nipype_ver
from nilearn import __version__ as nilearn_ver
from dipy import __version__ as dipy_ver
from pkg_resources import resource_filename as pkgrf
from ...engine import Workflow
from ...utils.sloppy_recon import make_sloppy
from ...__about__ import __version__

import logging
import json
from bids.layout import BIDSLayout
from .build_workflow import init_dwi_recon_workflow
from .anatomical import init_recon_anatomical_wf
from .interchange import anatomical_input_fields

LOGGER = logging.getLogger('nipype.workflow')


def init_qsirecon_wf(subject_list, run_uuid, work_dir, output_dir, recon_input,
                     recon_spec, low_mem, omp_nthreads, sloppy,
                     name="qsirecon_wf"):
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
        recon_input : str
            Root directory of the output from qsiprep
        recon_spec : str
            Path to a JSON file that specifies how to run reconstruction
        low_mem : bool
            Write uncompressed .nii files in some cases to reduce memory usage
        sloppy : bool
            If True, replace reconstruction options with fast but bad options.
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
            omp_nthreads=omp_nthreads,
            low_mem=low_mem,
            sloppy=sloppy
            )

        single_subject_wf.config['execution']['crashdump_dir'] = (os.path.join(
            output_dir, "qsirecon", "sub-" + subject_id, 'log', run_uuid))
        for node in single_subject_wf._get_all_nodes():
            node.config = deepcopy(single_subject_wf.config)

        qsiprep_wf.add_nodes([single_subject_wf])

    return qsiprep_wf


def init_single_subject_wf(
        subject_id, name, reportlets_dir, output_dir,
        low_mem, omp_nthreads, recon_input, recon_spec, sloppy):
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
        recon_input : str
            Root directory of the output from qsiprep
        recon_spec : str
            Path to a JSON file that specifies how to run reconstruction
        sloppy : bool
            Use bad parameters for reconstruction to make the workflow faster.
    """
    if name in ('single_subject_wf', 'single_subject_test_recon_wf'):
        # a fake spec
        spec = {"name": "fake",
                "atlases": [],
                "space": "T1w",
                "anatomical": [],
                "nodes": []}
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

        spec = _load_recon_spec(recon_spec, sloppy=sloppy)
        space = spec['space']
        layout = BIDSLayout(recon_input, validate=False, absolute_paths=True)
        # Get all the output files that are in this space
        dwi_files = [f.path for f in
                     layout.get(suffix="dwi", subject=subject_id, absolute_paths=True,
                                extension=['nii', 'nii.gz'])
                     if 'space-' + space in f.filename]
        LOGGER.info("found %s in %s", dwi_files, recon_input)

        # Find the corresponding mask files

    workflow = Workflow('sub-{}_{}'.format(subject_id, spec['name']))
    workflow.__desc__ = """
Reconstruction was
performed using *QSIprep* {qsiprep_ver},
which is based on *Nipype* {nipype_ver}
(@nipype1; @nipype2; RRID:SCR_002502).

""".format(
        qsiprep_ver=__version__, nipype_ver=nipype_ver)
    workflow.__postdesc__ = """

Many internal operations of *qsiprep* use
*Nilearn* {nilearn_ver} [@nilearn, RRID:SCR_001362] and
*Dipy* {dipy_ver}[@dipy].
For more details of the pipeline, see [the section corresponding
to workflows in *qsiprep*'s documentation]\
(https://qsiprep.readthedocs.io/en/latest/workflows.html \
"qsiprep's documentation").


### References

    """.format(nilearn_ver=nilearn_ver, dipy_ver=dipy_ver)

    if len(dwi_files) == 0:
        LOGGER.info("No dwi files found for %s", subject_id)
        return workflow

    # Was there a forced normalization during preprocessing?
    template_transform = [f.path for f in
                          layout.get(subject=subject_id, suffix='xfm', extension=['.h5'])
                          if 'to-T1w' in f.path and 'from-MNI152NLin2009cAsym' in f.path]
    if template_transform:
        template_transform = template_transform[0]
        has_transform = True
    else:
        has_transform = False

    t1w_brain_mask = layout.get(subject=subject_id,
                                invalid_filters='allow',
                                desc='brain',
                                extension=['nii', 'nii.gz'],
                                modality='anat')
    has_t1w_brain_mask = bool(t1w_brain_mask)

    anat_ingress_wf = init_recon_anatomical_wf(subject_id=subject_id,
                                               recon_input_dir=recon_input,
                                               extras_to_make=spec.get('anatomical', []),
                                               name='anat_ingress_wf')

    to_connect = [('outputnode.' + name, 'inputnode.' + name) for name in anatomical_input_fields]
    # create a processing pipeline for the dwis in each session
    dwi_recon_wf = init_dwi_recon_workflow(dwi_files=dwi_files,
                                           workflow_spec=spec,
                                           reportlets_dir=reportlets_dir,
                                           output_dir=output_dir,
                                           has_t1w=has_t1w_brain_mask,
                                           has_t1w_transform=has_transform,
                                           omp_nthreads=omp_nthreads)
    workflow.connect([(anat_ingress_wf, dwi_recon_wf, to_connect)])

    return workflow


def _load_recon_spec(spec_name, sloppy=False):
    prepackaged_dir = pkgrf("qsiprep", "data/pipelines")
    prepackaged = [op.split(fname)[1][:-5] for fname in glob(prepackaged_dir+"/*.json")]
    if op.exists(spec_name):
        recon_spec = spec_name
    elif spec_name in prepackaged:
        recon_spec = op.join(prepackaged_dir + "/{}.json".format(spec_name))
    else:
        raise Exception("{} is not a file that exists or in {}".format(spec_name, prepackaged))
    with open(recon_spec, "r") as f:
        try:
            spec = json.load(f)
        except Exception:
            raise Exception("Unable to read JSON spec. Check the syntax.")
    if sloppy:
        LOGGER.warning("Forcing reconstruction to use unrealistic parameters")
        spec = make_sloppy(spec)
    return spec
