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
from pathlib import Path
from pkg_resources import resource_filename as pkgrf
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.interfaces import mrtrix3, ants
import nipype.interfaces.io as nio
from ...engine import Workflow
import logging
from ...interfaces.bids import ReconDerivativesDataSink
from ...interfaces.anatomical import QsiprepAnatomicalIngress
from ...interfaces.mrtrix import (GenerateMasked5tt, ITKTransformConvert,
    TransformHeader)
from ...interfaces.ants import ConvertTransformFile
from ...interfaces.freesurfer import find_fs_path
from .interchange import (anatomical_input_fields, CREATEABLE_ANATOMICAL_OUTPUTS,
    FS_FILES_TO_REGISTER)

LOGGER = logging.getLogger('nipype.workflow')

# Required freesurfer files for mrtrix's HSV 5tt generation
HSV_REQUIREMENTS = [
    "mri/aparc+aseg.mgz",
    "mri/brainmask.mgz",
    "mri/transforms/talairach.xfm",
    "surf/lh.white",
    "surf/lh.pial",
    "surf/rh.white",
    "surf/rh.pial"
]

# Files that must exist if QSIPrep ran the anatomical workflow
QSIPREP_ANAT_REQUIREMENTS = [
    "sub-{subject_id}/anat/sub-{subject_id}_desc-brain_mask.nii.gz",
    "sub-{subject_id}/anat/sub-{subject_id}_desc-preproc_T1w.nii.gz"
]

QSIPREP_NORMALIZED_ANAT_REQUIREMENTS = [
    "sub-{subject_id}/anat/sub-{subject_id}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5",
    "sub-{subject_id}/anat/sub-{subject_id}_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5"
]

workflow_outputs = anatomical_input_fields + \
    FS_FILES_TO_REGISTER + CREATEABLE_ANATOMICAL_OUTPUTS

def init_recon_anatomical_wf(subject_id, recon_input_dir, extras_to_make,
                             freesurfer_dir="", needs_t1w_transform=False,
                             name='recon_anatomical_wf'):
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
            For example ['mrtrix_5tt_fast', 'mrtrix_5tt_hsv'].
        needs_t1w_transform :  bool
            If MNI-Space atlases need to get mapped to the DWI.
        freesurfer_dir : pathlike
            Path where the freesurfer outputs for `subject_id` go
    """

    workflow = Workflow(name=name)
    outputnode = pe.Node(
        niu.IdentityInterface(fields=workflow_outputs),
        name="outputnode")
    desc = ""
    status = {
        "has_qsiprep_5tt_fast": False,
        "has_qsiprep_5tt_hsvs": False,
        "has_freesurfer_5tt_hsvs": False
    }

    # Check to see if we have a T1w preprocessed by QSIPrep
    missing_qsiprep_anats = check_qsiprep_anatomical_outputs(
        recon_input_dir, subject_id, "T1w")
    has_qsiprep_t1w = not missing_qsiprep_anats
    status["has_qsiprep_t1w"] = has_qsiprep_t1w
    if missing_qsiprep_anats:
        LOGGER.info("Missing T1w QSIPrep outputs found: %s",
                    " ".join(missing_qsiprep_anats))
    else:
        LOGGER.info("Found usable QSIPrep-preprocessed T1w image and mask.")
        desc += "QSIPrep-preprocessed T1w images and brain masks were used. "
        anat_ingress = pe.Node(
            QsiprepAnatomicalIngress(subject_id=subject_id,
                                     recon_input_dir=recon_input_dir),
            name='anat_ingress')

        workflow.connect([
            (anat_ingress, outputnode,
                [(name, name) for name in anatomical_input_fields])
        ])

        # If FAST 5tt is requested, make this 
        if "mrtrix_5tt_fast" in extras_to_make:
            LOGGER.info("Creating a FSL-FAST 5tt image based on the QSIPrep T1w")
            status['has_qsiprep_5tt_fast'] = True
            desc += "FSL FAST was used to segment the brain into 5 tissue types. "
            LOGGER.warn("Using FAST for ACT is strongly discouraged!!! " +
                        "Consider using HSVS and FreeSurfer.")
            create_5tt_fast = pe.Node(
                GenerateMasked5tt(algorithm='fsl'), 
                name='create_5tt_fast')
            workflow.connect([
                (anat_ingress, create_5tt_fast, [('t1_brain_mask', 'mask'),
                                                 ('t1_preproc', 'in_file')]),
                (create_5tt_fast, outputnode, [('out_file', 'qsiprep_5tt_fast')])
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
                    source_file="anat/sub-{}_desc-preproc_T1w.nii.gz".format(subject_id),
                    suffix="5tt"),
                name='ds_qsiprep_5tt_hsvs',
                run_without_submitting=True)
            ds_fs_5tt_hsvs = pe.Node(
                ReconDerivativesDataSink(
                    desc="hsvs",
                    source_file="anat/sub-{}_desc-preproc_T1w.nii.gz".format(subject_id),
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
                    (anat_ingress, register_fs_to_qsiprep_wf, [
                        ("t1_preproc", "inputnode.qsiprep_reference_image"),
                        ("t1_brain_mask", "inputnode.qsiprep_reference_mask")]),
                    (fs_source, register_fs_to_qsiprep_wf, [
                        (field, "inputnode." + field) for field in FS_FILES_TO_REGISTER]),
                    (register_fs_to_qsiprep_wf, outputnode, [
                        ("outputnode." + field, field) for field in FS_FILES_TO_REGISTER])
                ])
                apply_header_to_5tt = pe.Node(
                    TransformHeader(), name="apply_header_to_5tt")
                workflow.connect([
                    (create_5tt_hsvs, apply_header_to_5tt, [("out_file", "in_image")]),
                    (register_fs_to_qsiprep_wf, apply_header_to_5tt,[
                        ("outputnode.fs_to_qsiprep_transform_mrtrix", "transform_file")]),
                    (apply_header_to_5tt, outputnode, [
                        ("out_image", "qsiprep_5tt")]),
                    (apply_header_to_5tt, ds_qsiprep_5tt_hsvs, [("out_image", "in_file")]),
                    (create_5tt_hsvs, ds_fs_5tt_hsvs, [("out_file", "in_file")])
                ])
                desc += "A hybrid surface/volume segmentation was created [Smith 2020]."

    workflow.__desc__ = desc
    return workflow, status


def check_hsv_inputs(subj_fs_path):
    """Determine if a FreeSurfer directory has the required files for HSV."""
    missing = []
    for requirement in HSV_REQUIREMENTS:
        if not (subj_fs_path / requirement).exists():
            missing.append(requirement)
    return missing


def check_qsiprep_anatomical_outputs(recon_input_dir, subject_id, anat_type):
    """Determines whether an aligned T1w exists in a qsiprep derivatives directory.
    
    It is possible that:
      - ``--dwi-only`` was used, in which case there is NO T1w available
      - ``--skip-t1-based-spatial-normalization``, there is a T1w but no transform to a template
      - Normal mode, there is a T1w and a transform to template space.
    """
    missing = []
    recon_input_path = Path(recon_input_dir)
    to_check = QSIPREP_ANAT_REQUIREMENTS if anat_type == "T1w" else QSIPREP_NORMALIZED_ANAT_REQUIREMENTS
    for requirement in to_check:
        requirement = recon_input_path / requirement.format(subject_id=subject_id)
        if not requirement.exists():
            missing.append(str(requirement))
    return missing


def init_register_fs_to_qsiprep_wf(use_qsiprep_reference_mask=False, 
                                   name="register_fs_to_qsiprep_wf"):
    """Registers a T1w images from freesurfer to another image and transforms
    """
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=FS_FILES_TO_REGISTER + [
                "qsiprep_reference_image", "qsiprep_reference_mask"]),
        name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=FS_FILES_TO_REGISTER + [
                "fs_to_qsiprep_transform_itk", 
                "fs_to_qsiprep_transform_mrtrix"]),
        name="outputnode")
    workflow = Workflow(name=name)
    workflow.__desc__ = "FreeSurfer outputs were registered to the QSIPrep outputs."

    # Convert the freesurfer inputs so we can register them with ANTs
    convert_fs_brain = pe.Node(
        mrtrix3.MRConvert(out_file="fs_brain.nii", args="-strides -1,-2,3"), 
        name="convert_fs_brain")

    # Register the brain to the QSIPrep reference
    ants_settings = pkgrf("qsiprep", "data/freesurfer_to_qsiprep.json")
    register_to_qsiprep = pe.Node(
        ants.Registration(from_file=ants_settings),
        name="register_to_qsiprep")

    # If there is a mask for the QSIPrep reference image, use it
    if use_qsiprep_reference_mask:
        workflow.connect(inputnode, "qsiprep_reference_mask", 
                         register_to_qsiprep, "fixed_image_masks")
    
    # The more recent ANTs mat format isn't compatible with transformconvert. 
    # So convert it to ANTs text format with ConvertTransform 
    convert_ants_transform = pe.Node(
        ConvertTransformFile(dimension=3),
        name="convert_ants_transform")
    
    # Convert from ANTs text format to MRtrix3 format
    convert_ants_to_mrtrix_transform = pe.Node(
        ITKTransformConvert(), name="convert_ants_to_mrtrix_transform")
    
    # Adjust the headers of all the input images so they're aligned to the qsiprep ref
    transform_nodes = {}
    for image_name in []:
        transform_nodes[image_name] = pe.Node(
            TransformHeader(), name="transform_" + image_name)
        workflow.connect([
            (inputnode, transform_nodes[image_name], [(image_name, "in_image")]),
            (convert_ants_to_mrtrix_transform, 
             transform_nodes[image_name], [("out_transform", "transform_file")]),
            (convert_ants_to_mrtrix_transform, outputnode, [("out_image", image_name)])
        ])

    workflow.connect([
        (inputnode, convert_fs_brain, [
            ("brain", "in_file")]),
        (inputnode,register_to_qsiprep, [
            ("qsiprep_reference_image", "fixed_image")]),
        (convert_fs_brain, register_to_qsiprep, [
            ("out_file", "moving_image")]),
        (register_to_qsiprep, convert_ants_transform, [
            (("forward_transforms", _get_first), "in_transform")]),
        (register_to_qsiprep, outputnode, [
            ("composite_transform", "fs_to_qsiprep_transform_itk")]),
        (convert_ants_transform, convert_ants_to_mrtrix_transform, [
            ("out_transform", "in_transform")]),
        (convert_ants_to_mrtrix_transform, outputnode, 
         [("out_transform", "fs_to_qsiprep_transform_mrtrix")])
    ])

    return workflow


def _get_first(item):
    if isinstance(item, (list, tuple)):
        return item[0]
    return item