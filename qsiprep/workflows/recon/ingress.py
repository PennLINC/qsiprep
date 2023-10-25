#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Ingress data from other
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_qsirecon_wf
.. autofunction:: init_single_subject_wf

"""
from pathlib import Path
from pkg_resources import resource_filename as pkgrf
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.interfaces import ants, afni
from ...niworkflows.interfaces.registration import RobustMNINormalizationRPT
import nipype.interfaces.io as nio
from ...engine import Workflow
import logging
from ...interfaces.bids import ReconDerivativesDataSink
from ...interfaces.anatomical import QsiprepAnatomicalIngress
from ...interfaces.mrtrix import (GenerateMasked5tt, ITKTransformConvert,
    TransformHeader)
from ...interfaces.ants import ConvertTransformFile
from ...interfaces.freesurfer import find_fs_path
from ...interfaces.gradients import ExtractB0s
from ...interfaces.interchange import (qsiprep_anatomical_ingressed_fields,
    FS_FILES_TO_REGISTER, anatomical_workflow_outputs, recon_workflow_input_fields)
from qsiprep.interfaces.utils import GetConnectivityAtlases
from .anatomical import check_hsv_inputs

LOGGER = logging.getLogger('nipype.workflow')

def init_ukb_recon_anatomical_wf(
        subject_id, recon_input_dir, extras_to_make,
        freesurfer_dir="", needs_t1w_transform=False,
        name='recon_ukb_anatomical_wf'):
    """
    This grabs anatomical outputs from qsiprep and calculates optional
    additional outputs like a dwi-resolution

    Parameters

        subject_id : str
            List of subject labels
        name : str
            Name of workflow
        recon_input_dir : str
            Root directory of the output from qsiprep
        extras_to_make : list
            list of optional derivatives that will be shared across images.
            For example ['mrtrix_5tt_hsv'].
        needs_t1w_transform :  bool
            If MNI-Space atlases need to get mapped to the DWI.
        freesurfer_dir : pathlike
            Path where the freesurfer outputs for `subject_id` go
    """

    workflow = Workflow(name=name)
    outputnode = pe.Node(
        niu.IdentityInterface(fields=anatomical_workflow_outputs),
        name="outputnode")
    desc = ""
    status = {
        "has_qsiprep_5tt_hsvs": False,
        "has_freesurfer_5tt_hsvs": False,
        "has_freesurfer": False
    }

    # Check to see if we have a T1w preprocessed by QSIPrep
    missing_qsiprep_anats = check_ukb_anatomical_outputs(
        recon_input_dir, subject_id, "T1w")
    has_qsiprep_t1w = not missing_qsiprep_anats
    status["has_qsiprep_t1w"] = has_qsiprep_t1w
    if missing_qsiprep_anats:
        LOGGER.info("Missing T1w QSIPrep outputs found: %s",
                    " ".join(missing_qsiprep_anats))
        workflow.add_nodes([outputnode])
    else:
        LOGGER.info("Found usable QSIPrep-preprocessed T1w image and mask.")
        desc += "QSIPrep-preprocessed T1w images and brain masks were used. "
        anat_ingress = pe.Node(
            QsiprepAnatomicalIngress(subject_id=subject_id,
                                     recon_input_dir=recon_input_dir),
            name='anat_ingress')

        workflow.connect([
            (anat_ingress, outputnode,
                [(name, name) for name in qsiprep_anatomical_ingressed_fields])
        ])


    # Check if the T1w-to-MNI transforms are in the QSIPrep outputs
    missing_qsiprep_transforms = check_qsiprep_anatomical_outputs(
        recon_input_dir, subject_id, "transforms")
    has_qsiprep_t1w_transforms = not missing_qsiprep_transforms
    status["has_qsiprep_t1w_transforms"] = has_qsiprep_t1w_transforms
    if missing_qsiprep_transforms:
        LOGGER.info("Missing T1w QSIPrep outputs found: %s",
                    " ".join(missing_qsiprep_transforms))
        if needs_t1w_transform:
            raise Exception("Reconstruction workflow requires transforms: " + \
                            " ".join(missing_qsiprep_transforms))
    else:
        LOGGER.info("Found T1w-to-template transforms")
        if needs_t1w_transform:
            desc += "T1w-based spatial normalization calculated during " \
                "preprocessing was used to map atlases from template space into " \
                "alignment with DWIs."

    # Are FreeSurfer Outputs available?
    subject_freesurfer_path = find_fs_path(freesurfer_dir, subject_id)
    if subject_freesurfer_path is None:
        LOGGER.info("No FreeSurfer inputs available for %s", subject_id)
        if "mrtrix_5tt_hsvs" in extras_to_make:
            raise Exception("FreeSurfer data is required to make HSVS 5tt image.")
    else:
        status['has_freesurfer'] = True
        LOGGER.info("Freesurfer directory %s exists for %s",
                    subject_freesurfer_path, subject_id)
        missing_fs_hsvs_files = check_hsv_inputs(Path(subject_freesurfer_path))

        # Check for specific files needed for HSVs.
        if missing_fs_hsvs_files:
            if "mrtrix_5tt_hsvs" in extras_to_make:
                raise Exception(" ".join(missing_fs_hsvs_files) +
                                "are missing: unable to make a HSV.")
            LOGGER.warn("HOWEVER" + " ".join(missing_fs_hsvs_files) +
                        "are missing from freesurfer.")

        # Connect the freesurfer outputs to the outputnode
        fs_source = pe.Node(
            nio.FreeSurferSource(
                subjects_dir=freesurfer_dir,
                subject_id='sub-' +subject_id),
            name="fs_source")

        # Make HSVs because we have freesurfer
        if 'mrtrix_5tt_hsvs' in extras_to_make:
            LOGGER.info("FreeSurfer data will be used to create a HSVS 5tt image.")
            status["has_freesurfer_5tt_hsvs"] = True
            create_5tt_hsvs = pe.Node(
                GenerateMasked5tt(
                    algorithm='hsvs',
                    in_file=str(subject_freesurfer_path)),
                name='create_5tt_hsvs')
            workflow.connect([
                (create_5tt_hsvs, outputnode, [('out_file', 'fs_5tt_hsvs')])])
            ds_qsiprep_5tt_hsvs = pe.Node(
                ReconDerivativesDataSink(
                    desc="hsvs",
                    suffix="5tt"),
                name='ds_qsiprep_5tt_hsvs',
                run_without_submitting=True)
            ds_fs_5tt_hsvs = pe.Node(
                ReconDerivativesDataSink(
                    desc="hsvs",
                    space="fsnative",
                    suffix="5tt"),
                name='ds_fs_5tt_hsvs',
                run_without_submitting=True)

            # Transform the 5tt image so it's registered to the QSIPrep AC-PC T1w
            if has_qsiprep_t1w:
                LOGGER.info("HSVS 5tt imaged will be registered to the "
                            "QSIPrep T1w image.")
                status["has_qsiprep_5tt_hsvs"] = True
                register_fs_to_qsiprep_wf = init_register_fs_to_qsiprep_wf(
                    use_qsiprep_reference_mask=True)
                workflow.connect([
                ])
                apply_header_to_5tt = pe.Node(
                    TransformHeader(), name="apply_header_to_5tt")
                workflow.connect([
                    (anat_ingress, register_fs_to_qsiprep_wf, [
                        ("t1_preproc", "inputnode.qsiprep_reference_image"),
                        ("t1_brain_mask", "inputnode.qsiprep_reference_mask")]),
                    (fs_source, register_fs_to_qsiprep_wf, [
                        (field, "inputnode." + field) for field in FS_FILES_TO_REGISTER]),
                    (register_fs_to_qsiprep_wf, outputnode, [
                        ("outputnode.fs_to_qsiprep_transform_mrtrix",
                         "fs_to_qsiprep_transform_mrtrix"),
                        ("outputnode.fs_to_qsiprep_transform_itk",
                         "fs_to_qsiprep_transform_itk")] + [
                        ("outputnode." + field, field) for field in FS_FILES_TO_REGISTER]),
                    (create_5tt_hsvs, apply_header_to_5tt, [("out_file", "in_image")]),
                    (create_5tt_hsvs, ds_fs_5tt_hsvs, [("out_file", "in_file")]),

                    (register_fs_to_qsiprep_wf, apply_header_to_5tt,[
                        ("outputnode.fs_to_qsiprep_transform_mrtrix", "transform_file")]),
                    (apply_header_to_5tt, outputnode, [
                        ("out_image", "qsiprep_5tt_hsvs")]),
                    (apply_header_to_5tt, ds_qsiprep_5tt_hsvs, [("out_image", "in_file")]),
                ])
                desc += "A hybrid surface/volume segmentation was created [Smith 2020]."

    workflow.__desc__ = desc
    return workflow, status


def get_t1w_registration_node(infant_mode, sloppy, omp_nthreads):

    # Gets an ants interface for t1w-based normalization
    if sloppy:
        LOGGER.info("Using QuickSyN")
        # Requires a warp file: make an inaccurate one
        settings = pkgrf('qsiprep', 'data/quick_syn.json')
        t1_2_mni = pe.Node(
            RobustMNINormalizationRPT(
                float=True,
                generate_report=True,
                settings=[settings],
            ),
            name='t1_2_mni',
            n_procs=omp_nthreads,
            mem_gb=2
        )
    else:
        t1_2_mni = pe.Node(
            RobustMNINormalizationRPT(
                float=True,
                generate_report=True,
                flavor='precise',
            ),
            name='t1_2_mni',
            n_procs=omp_nthreads,
            mem_gb=2
        )
        # Get the template image
    if not infant_mode:
        ref_img_brain = pkgrf('qsiprep', 'data/mni_1mm_t1w_lps_brain.nii.gz')
    else:
        ref_img_brain = pkgrf('qsiprep', 'data/mni_1mm_t1w_lps_brain_infant.nii.gz')

    t1_2_mni.inputs.template = 'MNI152NLin2009cAsym'
    t1_2_mni.inputs.reference_image = ref_img_brain
    t1_2_mni.inputs.orientation = "LPS"
    return t1_2_mni
