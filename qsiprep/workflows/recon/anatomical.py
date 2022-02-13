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
from ...interfaces.anatomical import QsiprepAnatomicalIngress
from ...interfaces.mrtrix import (GenerateMasked5tt, ITKTransformConvert,
    TransformHeader)
from ...interfaces.ants import ConvertTransformFile
from .interchange import anatomical_input_fields, freesurfer_output_names

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
    "anat/{subject_id}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5",
    "anat/{subject_id}_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5"
]

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
        niu.IdentityInterface(fields=anatomical_input_fields),
        name="outputnode")
    desc = ""

    # Check to see if we have a T1w preprocessed by QSIPrep
    missing_qsiprep_anats = check_qsiprep_anatomical_outputs(
        recon_input_dir, subject_id, "T1w")
    if missing_qsiprep_anats:
        LOGGER.info("Missing T1w QSIPrep outputs found: %s",
                    " ".join(missing_qsiprep_anats))
    has_qsiprep_t1w = not missing_qsiprep_anats
    if has_qsiprep_t1w:
        desc += "QSIPrep-preprocessed T1w images and brain masks were used. "
        anat_ingress = pe.Node(
            QsiprepAnatomicalIngress(subject_id=subject_id,
                                     recon_input_dir=recon_input_dir),
            name='anat_ingress')

        workflow.connect([
            (anat_ingress, outputnode,
                [(name, name) for name in anatomical_input_fields])
        ])

    # Check if the T1w-to-MNI transforms are in the QSIPrep outputs
    missing_qsiprep_transforms = check_qsiprep_anatomical_outputs(
        recon_input_dir, subject_id, "transforms")
    if missing_qsiprep_transforms:
        LOGGER.info("Missing T1w QSIPrep outputs found: %s",
                    " ".join(missing_qsiprep_anats))
        if needs_t1w_transform:
            raise Exception("Reconstruction workflow requires ")
    has_qsiprep_t1w_transforms = not missing_qsiprep_transforms
    if has_qsiprep_t1w_transforms and needs_t1w_transform:
        desc += "T1w-based spatial normalization calculated during preprocessing " \
            "was used to map atlases from template space into alignment with DWIs."

    # Are FreeSurfer Outputs available?
    freesurfer_path = Path(freesurfer_dir)
    subject_freesurfer_path = freesurfer_path / subject_id
    if freesurfer_dir and freesurfer_path.exists() and \
            subject_freesurfer_path.exists():
        missing_files = check_hsv_inputs(subject_freesurfer_path)

        # Check for specific files needed for HSVs. 
        if missing_files:
            if "mrtrix_5tt_hsv" in extras_to_make:
                raise Exception(" ".join(missing_files) + 
                                "are missing: unable to make a HSV.")
            LOGGER.warn(" ".join(missing_files) + "are missing from freesurfer.")
        
        # Connect the freesurfer outputs to the outputnode
        fs_source = pe.Node(nio.FreeSurferSource(subject_id=subject_id))
        workflow.connect([
            (fs_source, outputnode, 
                [(name,name) for name in freesurfer_output_names])
        ])




    # Prepare extra outputs
    if 'mrtrix_5tt_hsv' in extras_to_make:
        create_5tt = pe.Node(
            GenerateMasked5tt(
                algorithm='hsvs',
                ), 
            name='create_5tt')
        workflow.connect([
            (anat_ingress, create_5tt, [('t1_brain_mask', 'mask'),
                                        ('t1_preproc', 'in_file')]),
            (create_5tt, outputnode, [('out_file', 'mrtrix_5tt')])
        ])

    return workflow


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
        if not (recon_input_path / requirement.format(subject_id=subject_id)).exists():
            missing.append(requirement)
    return missing


def init_5tt_hsv_wf(subject_id, freesurfer_dir):
    pass


def init_register_fs_to_qsiprep_wf(freesurfer_dir, subject_id, target_file=None,
                                   use_qsiprep_reference_mask=False, 
                                   name="register_fs_to_qsiprep_wf"):
    """Registers a T1w images from freesurfer to another image and transforms
    """
    inputnode = pe.Node(
        niu.IdentityInterface(fields=["brain", "aparc_aseg", "brainmask", "aseg",
                                      "qsiprep_reference_image", "qsiprep_reference_mask"]),
        name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["brain", "aparc_aseg", "brainmask", "aseg",
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
        workflow.connect(inputnode, "qsiprep_reference_map", 
                         register_to_qsiprep, "fixed_mask")
    
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
            ("composite_transform", "in_transform")]),
        (register_to_qsiprep, outputnode, [
            ("composite_transform", "fs_to_qsiprep_transform_itk")]),
        (convert_ants_transform, convert_ants_to_mrtrix_transform, [
            ("out_transform", "in_transform")])
        (convert_ants_to_mrtrix_transform, outputnode, 
         [("out_transform", "fs_to_qsiprep_transform_mrtrix")])
    ])

    return workflow