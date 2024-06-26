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

import json
import os.path as op
from copy import deepcopy
from glob import glob

import nipype.pipeline.engine as pe
from bids.layout import BIDSLayout
from dipy import __version__ as dipy_ver
from nilearn import __version__ as nilearn_ver
from nipype import __version__ as nipype_ver
from nipype.utils.filemanip import split_filename
from packaging.version import Version
from pkg_resources import resource_filename as pkgrf

from ... import config
from ...engine import Workflow


def init_qsirecon_wf():
    """
    This workflow organizes the execution of qsirecon, with a sub-workflow for
    each subject.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.recon.base import init_qsirecon_wf
        wf = init_qsirecon_wf()


    Parameters


    """
    ver = Version(config.environment.version)
    qsiprep_wf = Workflow(name=f"qsirecon_{ver.major}_{ver.minor}_wf")
    qsiprep_wf.base_dir = config.execution.work_dir

    if config.workflow.recon_input_pipeline not in ("qsiprep", "ukb"):
        raise NotImplementedError(
            f"{config.workflow.recon_input_pipeline} is not supported as recon-input yet."
        )

    if config.workflow.recon_input_pipeline == "qsiprep":
        # This should work for --recon-input as long as the same dataset is in bids_dir
        # or if the call is doing preproc+recon
        to_recon_list = config.execution.participant_label
    elif config.workflow.recon_input_pipeline == "ukb":
        from ...utils.ingress import collect_ukb_participants, create_ukb_layout

        # The ukb input will always be specified as the bids input - we can't preproc it first
        ukb_layout = create_ukb_layout(config.execution.bids_dir)
        to_recon_list = collect_ukb_participants(
            ukb_layout, participant_label=config.execution.participant_label
        )

    for subject_id in to_recon_list:
        single_subject_wf = init_single_subject_recon_wf(subject_id=subject_id)

        single_subject_wf.config["execution"]["crashdump_dir"] = str(
            config.execution.qsirecon_dir / f"sub-{subject_id}" / "log" / config.execution.run_uuid
        )
        for node in single_subject_wf._get_all_nodes():
            node.config = deepcopy(single_subject_wf.config)
        qsiprep_wf.add_nodes([single_subject_wf])

        # Dump a copy of the config file into the log directory
        log_dir = (
            config.execution.qsirecon_dir / f"sub-{subject_id}" / "log" / config.execution.run_uuid
        )
        log_dir.mkdir(exist_ok=True, parents=True)
        config.to_filename(log_dir / "qsiprep.toml")

    return qsiprep_wf


def init_single_subject_recon_wf(subject_id):
    """
    This workflow organizes the reconstruction pipeline for a single subject.
    Reconstruction is performed using a separate workflow for each dwi series.

    Parameters

        subject_id : str
            Single subject label

    """
    from ...interfaces.ingress import QsiReconDWIIngress, UKBioBankDWIIngress
    from ...interfaces.interchange import (
        ReconWorkflowInputs,
        anatomical_workflow_outputs,
        qsiprep_output_names,
        recon_workflow_anatomical_input_fields,
        recon_workflow_input_fields,
    )
    from .anatomical import (
        init_dwi_recon_anatomical_workflow,
        init_highres_recon_anatomical_wf,
    )
    from .build_workflow import init_dwi_recon_workflow

    spec = _load_recon_spec()
    dwi_recon_inputs = _get_iterable_dwi_inputs(subject_id)

    workflow = Workflow(name=f"sub-{subject_id}_{spec['name']}")
    workflow.__desc__ = f"""
Reconstruction was
performed using *QSIprep* {config.__version__},
which is based on *Nipype* {nipype_ver}
(@nipype1; @nipype2; RRID:SCR_002502).

"""
    workflow.__postdesc__ = f"""

Many internal operations of *qsiprep* use
*Nilearn* {nilearn_ver} [@nilearn, RRID:SCR_001362] and
*Dipy* {dipy_ver}[@dipy].
For more details of the pipeline, see [the section corresponding
to workflows in *qsiprep*'s documentation]\
(https://qsiprep.readthedocs.io/en/latest/workflows.html \
"qsiprep's documentation").


### References

    """

    if len(dwi_recon_inputs) == 0:
        config.loggers.workflow.info("No dwi files found for %s", subject_id)
        return workflow

    # The recon spec may need additional anatomical files to be created.
    atlas_names = spec.get("atlases", [])
    needs_t1w_transform = spec_needs_to_template_transform(spec)

    # This is here because qsiprep currently only makes one anatomical result per subject
    # regardless of sessions. So process it on its
    if config.workflow.recon_input_pipeline == "qsiprep":
        anat_ingress_node, available_anatomical_data = init_highres_recon_anatomical_wf(
            subject_id=subject_id,
            extras_to_make=spec.get("anatomical", []),
            needs_t1w_transform=needs_t1w_transform,
        )

        # Connect the anatomical-only inputs. NOTE this is not to the inputnode!
        config.loggers.workflow.info(
            "Anatomical (T1w) available for recon: %s", available_anatomical_data
        )

    # create a processing pipeline for the dwis in each session
    dwi_recon_wfs = {}
    dwi_individual_anatomical_wfs = {}
    recon_full_inputs = {}
    dwi_ingress_nodes = {}
    anat_ingress_nodes = {}
    print(dwi_recon_inputs)
    for dwi_input in dwi_recon_inputs:
        dwi_file = dwi_input["bids_dwi_file"]
        wf_name = _get_wf_name(dwi_file)

        # Get the preprocessed DWI and all the related preprocessed images
        if config.workflow.recon_input_pipeline == "qsiprep":
            dwi_ingress_nodes[dwi_file] = pe.Node(
                QsiReconDWIIngress(dwi_file=dwi_file),
                name=wf_name + "_ingressed_dwi_data",
            )

        elif config.workflow.recon_input_pipeline == "ukb":
            dwi_ingress_nodes[dwi_file] = pe.Node(
                UKBioBankDWIIngress(dwi_file=dwi_file, data_dir=str(dwi_input["path"].absolute())),
                name=wf_name + "_ingressed_ukb_dwi_data",
            )
            anat_ingress_nodes[dwi_file], available_anatomical_data = (
                init_highres_recon_anatomical_wf(
                    subject_id=subject_id,
                    recon_input_dir=dwi_input["path"],
                    extras_to_make=spec.get("anatomical", []),
                    pipeline_source="ukb",
                    needs_t1w_transform=needs_t1w_transform,
                    name=wf_name + "_ingressed_ukb_anat_data",
                )
            )

        # Create scan-specific anatomical data (mask, atlas configs, odf ROIs for reports)
        print(available_anatomical_data)
        dwi_individual_anatomical_wfs[dwi_file], dwi_available_anatomical_data = (
            init_dwi_recon_anatomical_workflow(
                atlas_names=atlas_names,
                prefer_dwi_mask=False,
                needs_t1w_transform=needs_t1w_transform,
                extras_to_make=spec.get("anatomical", []),
                name=wf_name + "_dwi_specific_anat_wf",
                **available_anatomical_data,
            )
        )

        # This node holds all the inputs that will go to the recon workflow.
        # It is the definitive place to check what the input files are
        recon_full_inputs[dwi_file] = pe.Node(
            ReconWorkflowInputs(), name=wf_name + "_recon_inputs"
        )

        # This is the actual recon workflow for this dwi file
        dwi_recon_wfs[dwi_file] = init_dwi_recon_workflow(
            available_anatomical_data=dwi_available_anatomical_data,
            workflow_spec=spec,
            name=wf_name + "_recon_wf",
        )

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

            (
                anat_ingress_node if config.workflow.recon_input_pipeline == "qsiprep"
                else anat_ingress_nodes[dwi_file],
                dwi_individual_anatomical_wfs[dwi_file], [
                    (f"outputnode.{trait}", f"inputnode.{trait}")
                    for trait in anatomical_workflow_outputs
                ]
            )
        ])  # fmt:skip

    # Fill-in datasinks and reportlet datasinks for the anatomical workflow
    for _node in workflow.list_node_names():
        node_suffix = _node.split(".")[-1]
        if node_suffix.startswith("ds"):
            base_dir = (
                config.execution.reportlets_dir
                if "report" in node_suffix
                else config.execution.output_dir
            )
            workflow.get_node(_node).inputs.base_directory = base_dir
            # workflow.get_node(_node).inputs.source_file = \
            #     "anat/sub-{}_desc-preproc_T1w.nii.gz".format(subject_id)
    return workflow


def spec_needs_to_template_transform(recon_spec):
    """Determine whether a recon spec needs a transform from T1wACPC to a template."""
    atlases = recon_spec.get("atlases", [])
    return bool(atlases)


def _get_wf_name(dwi_file):
    basedir, fname, ext = split_filename(dwi_file)
    tokens = fname.split("_")
    return "_".join(tokens[:-1]).replace("-", "_")


def _load_recon_spec():
    from ...utils.sloppy_recon import make_sloppy

    spec_name = config.workflow.recon_spec
    prepackaged_dir = pkgrf("qsiprep", "data/pipelines")
    prepackaged = [op.split(fname)[1][:-5] for fname in glob(prepackaged_dir + "/*.json")]
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
    if config.execution.sloppy:
        config.loggers.workflow.warning("Forcing reconstruction to use unrealistic parameters")
        spec = make_sloppy(spec)
    return spec


def _get_iterable_dwi_inputs(subject_id):
    """Return inputs for the recon ingressors depending on the pipeline source.

    If qsiprep was used as the pipeline source, the iterable is going to be the
    dwi files (there can be an arbitrary number of them).

    If ukb or hcpya were used there is only one dwi file per subject, so the
    ingressors are sent the subject directory, which makes it easier to find
    the other files needed.

    """
    from ...utils.ingress import create_ukb_layout

    recon_input_directory = config.execution.recon_input
    if config.workflow.recon_input_pipeline == "qsiprep":
        # If recon_input is specified without qsiprep, check if we can find the subject dir
        if not (recon_input_directory / f"sub-{subject_id}").exists():
            config.loggers.workflow.info(
                "%s not in %s, trying recon_input=%s",
                subject_id,
                recon_input_directory,
                recon_input_directory / "qsiprep",
            )

            recon_input_directory = recon_input_directory / "qsiprep"
            if not (recon_input_directory / f"sub-{subject_id}").exists():
                raise Exception(
                    "Unable to find subject directory in %s or %s"
                    % (config.execution.recon_input, recon_input_directory)
                )

        layout = BIDSLayout(recon_input_directory, validate=False, absolute_paths=True)
        # Get all the output files that are in this space
        dwi_files = [
            f.path
            for f in layout.get(
                suffix="dwi",
                subject=subject_id,
                absolute_paths=True,
                extension=["nii", "nii.gz"],
            )
            if "space-T1w" in f.filename
        ]
        config.loggers.workflow.info("found %s in %s", dwi_files, recon_input_directory)
        return [{"bids_dwi_file": dwi_file} for dwi_file in dwi_files]

    if config.workflow.recon_input_pipeline == "ukb":
        return create_ukb_layout(ukb_dir=config.execution.bids_dir, participant_label=subject_id)

    raise Exception("Unknown pipeline " + config.workflow.recon_input_pipeline)
