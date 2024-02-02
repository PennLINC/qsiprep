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
import nipype.pipeline.engine as pe
from nipype.utils.filemanip import split_filename
from nilearn import __version__ as nilearn_ver
from dipy import __version__ as dipy_ver
from pkg_resources import resource_filename as pkgrf
from ...utils.ingress import create_ukb_layout
from ...engine import Workflow
from ...utils.sloppy_recon import make_sloppy
from ...__about__ import __version__
from ...interfaces.ingress import QsiReconDWIIngress, UKBioBankDWIIngress
import logging
import json
from bids.layout import BIDSLayout
from .build_workflow import init_dwi_recon_workflow
from .anatomical import init_highres_recon_anatomical_wf, init_dwi_recon_anatomical_workflow
from ...interfaces.interchange import (anatomical_workflow_outputs, recon_workflow_anatomical_input_fields,
                                       ReconWorkflowInputs,
                                       qsiprep_output_names, recon_workflow_input_fields)

LOGGER = logging.getLogger('nipype.workflow')


def init_qsirecon_wf(subject_list, run_uuid, work_dir, output_dir, recon_input,
                     recon_spec, low_mem, omp_nthreads, sloppy, freesurfer_input,
                     b0_threshold, skip_odf_plots, pipeline_source, infant_mode,
                     output_resolution, name="qsirecon_wf"):
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
                              freesurfer_input="freesurfer",
                              sloppy=False,
                              omp_nthreads=1,
                              skip_odf_plots=False,
                              pipeline_source="qsiprep",
                              output_resolution=0.0,
                              infant_mode=False
                              )


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
        pipeline_source : str
            The pipeline used to process the input data. Default is "qsiprep",
            someday it can be "ukb", "hcp"
        low_mem : bool
            Write uncompressed .nii files in some cases to reduce memory usage
        freesurfer_input : Pathlib.Path
            Path to the directory containing subject freesurfer outputs ($SUBJECTS_DIR)
        sloppy : bool
            If True, replace reconstruction options with fast but bad options.
        infant_mode : bool
            Use MNI Infant templates
        output_resolution : float
            resolution to resample template space outputs. if 0, the native resolution
            will be used

    """
    qsiprep_wf = Workflow(name=name)
    qsiprep_wf.base_dir = work_dir

    reportlets_dir = os.path.join(work_dir, 'reportlets')
    for subject_id in subject_list:
        single_subject_wf = init_single_subject_wf(
            subject_id=subject_id,
            recon_input=recon_input,
            recon_spec=recon_spec,
            pipeline_source=pipeline_source,
            name="single_subject_" + subject_id + "_recon_wf",
            reportlets_dir=reportlets_dir,
            output_dir=output_dir,
            omp_nthreads=omp_nthreads,
            low_mem=low_mem,
            sloppy=sloppy,
            b0_threshold=b0_threshold,
            freesurfer_input=freesurfer_input,
            skip_odf_plots=skip_odf_plots,
            infant_mode=infant_mode,
            output_resolution=output_resolution
            )

        single_subject_wf.config['execution']['crashdump_dir'] = (os.path.join(
            output_dir, "qsirecon", "sub-" + subject_id, 'log', run_uuid))
        for node in single_subject_wf._get_all_nodes():
            node.config = deepcopy(single_subject_wf.config)

        qsiprep_wf.add_nodes([single_subject_wf])

    return qsiprep_wf


def init_single_subject_wf(
        subject_id, name, reportlets_dir, output_dir, freesurfer_input,
        skip_odf_plots, infant_mode, output_resolution, low_mem, omp_nthreads,
        recon_input, recon_spec, sloppy, b0_threshold, pipeline_source):
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
        freesurfer_input : Pathlib.Path
            Path to the directory containing subject freesurfer outputs ($SUBJECTS_DIR)
        recon_input : str
            Root directory of the output from qsiprep
        recon_spec : str
            Path to a JSON file that specifies how to run reconstruction
        sloppy : bool
            Use bad parameters for reconstruction to make the workflow faster.
        pipeline_source : str
            Which pipeline was used to process the input data
        infant_mode : bool
            Use MNI Infant templates
        output_resolution : float
            resolution to resample template space outputs. if 0, the native resolution
            will be used
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
        dwi_recon_inputs = [{"bids_dwi_file": '/made/up/outputs/sub-X_dwi.nii.gz'}]
    else:
        # TODO: Change this to handle multiple input types
        spec = _load_recon_spec(recon_spec, sloppy=sloppy)
        space = spec['space']
        dwi_recon_inputs = _get_iterable_dwi_inputs(recon_input, subject_id, pipeline_source, space)

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

    if len(dwi_recon_inputs) == 0:
        LOGGER.info("No dwi files found for %s", subject_id)
        return workflow

    # The recon spec may need additional anatomical files to be created.
    atlas_names = spec.get('atlases', [])
    needs_t1w_transform = spec_needs_to_template_transform(spec)

    # This is here because qsiprep currently only makes one anatomical result per subject
    # regardless of sessions. So process it on its
    if pipeline_source == "qsiprep":
        anat_ingress_node, available_anatomical_data = init_highres_recon_anatomical_wf(
            subject_id=subject_id,
            recon_input_dir=recon_input,
            extras_to_make=spec.get('anatomical', []),
            freesurfer_dir=freesurfer_input,
            needs_t1w_transform=needs_t1w_transform,
            pipeline_source="qsiprep",
            infant_mode=infant_mode,
            name='anat_ingress_wf')

        # Connect the anatomical-only inputs. NOTE this is not to the inputnode!
        LOGGER.info("Anatomical (T1w) available for recon: %s", available_anatomical_data)

    # create a processing pipeline for the dwis in each session
    dwi_recon_wfs = {}
    dwi_individual_anatomical_wfs = {}
    recon_full_inputs = {}
    dwi_ingress_nodes = {}
    anat_ingress_nodes = {}
    print(dwi_recon_inputs)
    for dwi_input in dwi_recon_inputs:
        dwi_file = dwi_input['bids_dwi_file']
        wf_name = _get_wf_name(dwi_file)

        # Get the preprocessed DWI and all the related preprocessed images
        if pipeline_source == "qsiprep":
            dwi_ingress_nodes[dwi_file] = pe.Node(
                QsiReconDWIIngress(dwi_file=dwi_file),
                name=wf_name + "_ingressed_dwi_data")

        elif pipeline_source == "ukb":
            dwi_ingress_nodes[dwi_file] = pe.Node(
                UKBioBankDWIIngress(dwi_file=dwi_file,
                                 data_dir=str(dwi_input['path'].absolute())),
                name=wf_name + "_ingressed_ukb_dwi_data")
            anat_ingress_nodes[dwi_file], available_anatomical_data = init_highres_recon_anatomical_wf(
                subject_id=subject_id,
                recon_input_dir=dwi_input['path'],
                extras_to_make=spec.get('anatomical', []),
                freesurfer_dir=freesurfer_input,
                pipeline_source="ukb",
                needs_t1w_transform=needs_t1w_transform,
                infant_mode=infant_mode,
                name=wf_name + "_ingressed_ukb_anat_data")

        # Create scan-specific anatomical data (mask, atlas configs, odf ROIs for reports)
        print(available_anatomical_data)
        dwi_individual_anatomical_wfs[dwi_file], dwi_available_anatomical_data = \
            init_dwi_recon_anatomical_workflow(
                atlas_names=atlas_names,
                omp_nthreads=omp_nthreads,
                infant_mode=infant_mode,
                prefer_dwi_mask=False,
                sloppy=sloppy,
                needs_t1w_transform=needs_t1w_transform,
                b0_threshold=b0_threshold,
                freesurfer_dir=freesurfer_input,
                extras_to_make=spec.get('anatomical', []),
                name=wf_name + "_dwi_specific_anat_wf",
                output_resolution=output_resolution,
                **available_anatomical_data)

        # This node holds all the inputs that will go to the recon workflow.
        # It is the definitive place to check what the input files are
        recon_full_inputs[dwi_file] = pe.Node(ReconWorkflowInputs(), name=wf_name + "_recon_inputs")

        # This is the actual recon workflow for this dwi file
        dwi_recon_wfs[dwi_file] = init_dwi_recon_workflow(
            available_anatomical_data=dwi_available_anatomical_data,
            workflow_spec=spec,
            name=wf_name + "_recon_wf",
            reportlets_dir=reportlets_dir,
            output_dir=output_dir,
            omp_nthreads=omp_nthreads,
            skip_odf_plots=skip_odf_plots)

        # Connect the collected diffusion data (gradients, etc) to the inputnode
        workflow.connect([
            # The dwi data
            (dwi_ingress_nodes[dwi_file], recon_full_inputs[dwi_file], [
                (trait, trait) for trait in qsiprep_output_names]),

            # Session-specific anatomical data
            (dwi_ingress_nodes[dwi_file], dwi_individual_anatomical_wfs[dwi_file],
             [(trait, "inputnode." + trait) for trait in qsiprep_output_names]),

            # subject dwi-specific anatomical to a special node in recon_full_inputs so
            # we have a record of what went in. Otherwise it would be lost in an IdentityInterface
            (dwi_individual_anatomical_wfs[dwi_file], recon_full_inputs[dwi_file],
             [("outputnode." + trait, trait) for trait in recon_workflow_anatomical_input_fields]),

            # send the recon_full_inputs to the dwi recon workflow
            (recon_full_inputs[dwi_file], dwi_recon_wfs[dwi_file],
             [(trait, "inputnode." + trait) for trait in recon_workflow_input_fields]),

            (anat_ingress_node if pipeline_source=="qsiprep" else anat_ingress_nodes[dwi_file],
             dwi_individual_anatomical_wfs[dwi_file],
             [("outputnode."+trait, "inputnode."+trait) for trait in anatomical_workflow_outputs])
        ])

    # Fill-in datasinks and reportlet datasinks for the anatomical workflow
    for _node in workflow.list_node_names():
        node_suffix = _node.split('.')[-1]
        if node_suffix.startswith('ds'):
            base_dir = reportlets_dir if "report" in node_suffix else output_dir
            # workflow.get_node(_node).inputs.base_directory = base_dir
            # workflow.get_node(_node).inputs.source_file = \
            #     "anat/sub-{}_desc-preproc_T1w.nii.gz".format(subject_id)
    return workflow


def spec_needs_to_template_transform(recon_spec):
    """Determine whether a recon spec needs a transform from T1wACPC to a template.
    """
    atlases = recon_spec.get("atlases", [])
    return bool(atlases)


def _get_wf_name(dwi_file):
    basedir, fname, ext = split_filename(dwi_file)
    tokens = fname.split("_")
    return "_".join(tokens[:-1]).replace("-", "_")


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


def _get_iterable_dwi_inputs(recon_input_directory, subject_id, pipeline_source, space):
    """Return inputs for the recon ingressors depending on the pipeline source.

    If qsiprep was used as the pipeline source, the iterable is going to be the
    dwi files (there can be an arbitrary number of them).

    If ukb or hcpya were used there is only one dwi file per subject, so the
    ingressors are sent the subject directory, which makes it easier to find
    the other files needed.

    """

    if pipeline_source == "qsiprep":
        # If recon_input is specified without qsiprep, check if we can find the subject dir
        subject_dir = 'sub-' + subject_id
        if not op.exists(op.join(recon_input_directory, subject_dir)):
            qp_recon_input = op.join(recon_input_directory, "qsiprep")
            LOGGER.info("%s not in %s, trying recon_input=%s",
                        subject_dir, recon_input_directory, qp_recon_input)
            if not op.exists(op.join(qp_recon_input, subject_dir)):
                raise Exception(
                    "Unable to find subject directory in %s or %s" % (
                        recon_input_directory, qp_recon_input))
            recon_input_directory = qp_recon_input


        layout = BIDSLayout(recon_input_directory, validate=False, absolute_paths=True)
        # Get all the output files that are in this space
        dwi_files = [f.path for f in
                     layout.get(suffix="dwi", subject=subject_id, absolute_paths=True,
                                extension=['nii', 'nii.gz'])
                     if 'space-' + space in f.filename]
        LOGGER.info("found %s in %s", dwi_files, recon_input_directory)
        return [{"bids_dwi_file": dwi_file} for dwi_file in dwi_files]

    if pipeline_source == "ukb":
        return create_ukb_layout(ukb_dir=recon_input_directory,
                                 participant_label=subject_id)

    raise Exception("Unknown pipeline " + pipeline_source)