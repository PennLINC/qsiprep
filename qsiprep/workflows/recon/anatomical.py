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
from matplotlib.pyplot import connect
from pkg_resources import resource_filename as pkgrf
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.interfaces import mrtrix3, ants, afni
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
from ...interfaces.nilearn import MaskB0Series
from .interchange import (qsiprep_anatomical_ingressed_fields,
    FS_FILES_TO_REGISTER, anatomical_workflow_outputs, recon_workflow_input_fields)
from qsiprep.interfaces.utils import GetConnectivityAtlases


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
        niu.IdentityInterface(fields=anatomical_workflow_outputs),
        name="outputnode")
    desc = ""
    status = {
        "has_qsiprep_5tt_fast": False,
        "has_qsiprep_5tt_hsvs": False,
        "has_freesurfer_5tt_hsvs": False,
        "has_freesurfer": False
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
                [(name, name) for name in qsiprep_anatomical_ingressed_fields])
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
    for image_name in FS_FILES_TO_REGISTER:
        transform_nodes[image_name] = pe.Node(
            TransformHeader(), name="transform_" + image_name)
        workflow.connect([
            (inputnode, transform_nodes[image_name], [(image_name, "in_image")]),
            (convert_ants_to_mrtrix_transform, 
             transform_nodes[image_name], [("out_transform", "transform_file")]),
            (transform_nodes[image_name], outputnode, [("out_image", image_name)])
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


def init_dwi_recon_anatomical_workflow(
    atlas_names, omp_nthreads, has_qsiprep_5tt_fast, has_qsiprep_5tt_hsvs, 
    has_freesurfer_5tt_hsvs, has_qsiprep_t1w, has_qsiprep_t1w_transforms,
    infant_mode, has_freesurfer, extras_to_make, freesurfer_dir, b0_threshold, 
    sloppy=False, prefer_dwi_mask=False, name="qsirecon_anat_wf"):
    """Ensure that anatomical data is available for the reconstruction workflows.
    
    This workflow calculates images/transforms that require a DWI spatial reference.
    Specifically, three additional features are added:

      * ``"dwi_mask"``: a brain mask in the voxel space of the DWI
      * ``"atlas_configs"``: A dictionary used by connectivity workflows to get
        brain parcellations.
      * ``"odf_rois"``: An image with some interesting ROIs for plotting ODFs

    Parameters:
    ===========
    
        has_qsiprep_5tt_fast: bool
            Is there a FAST 5tt file in qsiprep space
        has_qsiprep_5tt_hsvs: 
        has_freesurfer_5tt_hsvs: True, 
        has_qsiprep_t1w: 
        has_qsiprep_t1w_transforms: True}
    """
    # Inputnode holds data from the T1w-based anatomical workflow
    inputnode = pe.Node(
        niu.IdentityInterface(fields=recon_workflow_input_fields),
        name='inputnode')
    connect_from_inputnode = set(recon_workflow_input_fields)
    # Buffer to hold the anatomical files that are calculated here
    buffernode = pe.Node(
        niu.IdentityInterface(fields=recon_workflow_input_fields),
        name='buffernode')
    connect_from_buffernode = set()

    def _get_source_node(fieldname):
        if fieldname in connect_from_inputnode:
            return inputnode
        if fieldname in connect_from_buffernode:
            return buffernode
        raise Exception("Can't determine location of " + fieldname)

    def _exchange_fields(fields):
        connect_from_inputnode.difference_update(fields)
        connect_from_buffernode.update(fields)
    
    _exchange_fields(['dwi_mask', 'atlas_configs', 'odf_rois'])

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=recon_workflow_input_fields),
        name="outputnode")
    workflow = Workflow(name=name)
    skull_strip_method = "antsBrainExtraction"
    desc = ""
    def _get_status():
        return {"has_qsiprep_5tt_fast": has_qsiprep_5tt_fast, 
                "has_qsiprep_5tt_hsvs": has_qsiprep_5tt_hsvs, 
                "has_freesurfer_5tt_hsvs": has_freesurfer_5tt_hsvs, 
                "has_qsiprep_t1w": has_qsiprep_t1w, 
                "has_qsiprep_t1w_transforms": has_qsiprep_t1w_transforms}
    
    # Missing Freesurfer AND QSIPrep T1ws, or the user wants a DWI-based mask
    if not (has_qsiprep_t1w or has_freesurfer) or prefer_dwi_mask:
        desc += "No T1w weighted images were available for masking, so a mask " \
            "was estimated based on the b=0 images in the DWI data itself."
        extract_b0s = pe.Node(
            ExtractB0s(b0_threshold=b0_threshold),
            name='extract_b0s')
        mask_b0s = pe.Node(
            afni.Automask(outputtype="NIFTI_GZ"),
            name="mask_b0s")
        workflow.connect([
            (inputnode, extract_b0s, [
                ("dwi_file", "dwi_series"),
                ("bval_file", "bval_file")]),
            (extract_b0s, mask_b0s, [("b0_series", "in_file")]),
            (mask_b0s, outputnode, [("out_file", "dwi_mask")])
        ])
        return workflow, _get_status()

    # No data from QSIPrep was available, BUT we have freesurfer! register it and
    # get the brain, masks and possibly a to-MNI transform.
    # --> If has_freesurfer AND has qsiprep_t1w, the necessary files were created earlier
    elif has_freesurfer and not has_qsiprep_t1w:
        fs_source = pe.Node(
            nio.FreeSurferSource(
                subjects_dir=freesurfer_dir),
            name="fs_source")
        # Register the FreeSurfer brain to the DWI reference
        desc += "A brainmasked T1w image from FreeSurfer was registered to the " \
            "preprocessed DWI data. Brainmasks from FreeSurfer were used in all " \
            "subsequent reconstruction steps. "
        
        # Move these fields to the buffernode
        _exchange_fields(FS_FILES_TO_REGISTER + [
            't1_brain_mask', 't1_preproc', 'fs_to_qsiprep_transform_mrtrix', 
            'fs_to_qsiprep_transform_itk'])

        # Perform the registration and connect the outputs to buffernode
        # NOTE: using FreeSurfer "brain" image as t1_preproc and aseg as the brainmask
        has_qsiprep_t1w = True
        register_fs_to_qsiprep_wf = init_register_fs_to_qsiprep_wf(
            use_qsiprep_reference_mask=False)
        workflow.connect([
            (inputnode, fs_source, [("subject_id", "subject_id")]),
            (inputnode, register_fs_to_qsiprep_wf, [
                ("dwi_ref", "inputnode.qsiprep_reference_image")]),
            (fs_source, register_fs_to_qsiprep_wf, [
                (field, "inputnode." + field) for field in FS_FILES_TO_REGISTER]),
            (register_fs_to_qsiprep_wf, buffernode, [
                ("outputnode.brain", "t1_preproc"),
                ("outputnode.aseg", "t1_brain_mask"),
                ("outputnode.fs_to_qsiprep_transform_mrtrix", 
                    "fs_to_qsiprep_transform_mrtrix"),
                ("outputnode.fs_to_qsiprep_transform_itk",
                    "fs_to_qsiprep_transform_itk")] + [
                ("outputnode." + field, field) for field in FS_FILES_TO_REGISTER]),
        ])

    # Do we need to transform the 5tt hsvs from fsnative?
    if 'mrtrix_5tt_hsvs' in extras_to_make and not has_qsiprep_5tt_hsvs:
        # Transform the 5tt image so it's registered to the QSIPrep AC-PC T1w
        LOGGER.info("HSVS 5tt imaged will be registered to the "
                    "QSIPrep dwiref image.")
        _exchange_fields(["qsiprep_5tt_hsvs"])
        if not has_freesurfer_5tt_hsvs:
            raise Exception("The 5tt image in fsnative should have been created by now")  
        apply_header_to_5tt_hsvs = pe.Node(
            TransformHeader(), name="apply_header_to_5tt_hsvs")
        ds_qsiprep_5tt_hsvs = pe.Node(
            ReconDerivativesDataSink(
                desc="hsvs",
                suffix="5tt"),
            name='ds_qsiprep_5tt_hsvs',
            run_without_submitting=True)
        workflow.connect([
            (inputnode, apply_header_to_5tt_hsvs, [("fs_5tt_hsvs", "in_image")]),
            (apply_header_to_5tt_hsvs, buffernode, [
                ("out_image", "qsiprep_5tt_hsvs")]),
            (apply_header_to_5tt_hsvs, ds_qsiprep_5tt_hsvs, [("out_image", "in_file")]),
        ])
        desc += "A hybrid surface/volume segmentation was created [Smith 2020]."

    # If FAST 5tt is requested, make this 
    if "mrtrix_5tt_fast" in extras_to_make and not has_qsiprep_5tt_fast:
        if not has_qsiprep_t1w:
            raise Exception("Must have a T1w registered to ")
        _exchange_fields(["qsiprep_5tt_fast"])
        LOGGER.info("Creating a FSL-FAST 5tt image based on the QSIPrep T1w")
        desc += "FSL FAST was used to segment the brain into 5 tissue types. "
        LOGGER.warn("Using FAST for ACT is strongly discouraged!!! " +
                    "Consider using HSVS and FreeSurfer.")
        create_5tt_fast = pe.Node(
            GenerateMasked5tt(algorithm='fsl'), 
            name='create_5tt_fast')
        workflow.connect([
            (_get_source_node('t1_brain_mask'), create_5tt_fast, [
                ('t1_brain_mask', 'mask')]),
            (_get_source_node('t1_preproc'), create_5tt_fast, [
                ('t1_preproc', 'in_file')]),
            (create_5tt_fast, buffernode, [('out_file', 'qsiprep_5tt_fast')])
        ])

    # If we have transforms to the template space, use them to get ROIs/atlases
    if not has_qsiprep_t1w_transforms and has_qsiprep_t1w:
        if not has_qsiprep_t1w:
            raise Exception("Unable to find a T1w image for registration to the template.")
        
        desc += "In order to warp brain parcellations from template space into " \
            "alignment with the DWI data, the DWI-aligned FreeSurfer brain was " \
            "registered to template space. "
       
        # We now have qsiprep t1w and transforms!!
        has_qsiprep_t1w = has_qsiprep_t1w_transforms = True
        skull_strip_method = "FreeSurfer"

    # Simply resample the T1w mask into the DWI resolution. This was the default 
    # up to version 0.14.3
    if has_qsiprep_t1w and not prefer_dwi_mask:
        desc += "Brainmasks from {} were used in all " \
            "subsequent reconstruction steps.".format(skull_strip_method)
            # Resample anat mask
        resample_mask = pe.Node(
            afni.Resample(outputtype='NIFTI_GZ', resample_mode="NN"),
            name='resample_mask')
        
        workflow.connect([
            (inputnode, resample_mask, [
                ("t1_brain_mask", "in_file"),
                ("dwi_ref", "master")]),
            (resample_mask, buffernode, [("out_file", "dwi_mask")])
        ])

    if not has_qsiprep_t1w_transforms:
        # Calculate the transforms here:
        has_qsiprep_t1w_transforms = True
        _exchange_fields(['t1_2_mni_forward_transform', 't1_2_mni_reverse_transform'])
        t1_2_mni = pe.Node(
            get_t1w_registration_node(
                infant_mode, sloppy or not atlas_names, omp_nthreads),
            name="t1_2_mni")
        workflow.connect([
            (_get_source_node("t1_preproc"), t1_2_mni, [('t1_preproc', 'moving_image')]),
            (t1_2_mni, buffernode, [
                ("composite_transform", "t1_2_mni_forward_transform"),
                ("inverse_composite_transform", "t1_2_mni_reverse_transform")
            ])
        ])
        # TODO: add datasinks here

    LOGGER.info("Transforming ODF ROIs into DWI space for visual report.")
    # Resample ROI targets to DWI resolution for ODF plotting
    crossing_rois_file = pkgrf('qsiprep', 'data/crossing_rois.nii.gz')
    odf_rois = pe.Node(
        ants.ApplyTransforms(interpolation="MultiLabel", dimension=3),
        name="odf_rois")
    odf_rois.inputs.input_image = crossing_rois_file
    workflow.connect([
        (_get_source_node('t1_2_mni_reverse_transform'), odf_rois, [
            ('t1_2_mni_reverse_transform', 'transforms')]),
        (inputnode, odf_rois, [
            ("dwi_file", "reference_image")]),
        (odf_rois, buffernode, [("output_image", "odf_rois")])
    ])

    # Similarly, if we need atlases, transform them into DWI space
    if atlas_names:
        desc += "Cortical parcellations were mapped from template space to DWIS " \
            "using the T1w-based spatial normalization. "

        # Resample all atlases to dwi_file's resolution
        get_atlases = pe.Node(
            GetConnectivityAtlases(atlas_names=atlas_names),
            name='get_atlases',
            run_without_submitting=True)
        workflow.connect([
            (_get_source_node('t1_2_mni_reverse_transform'), get_atlases, [
                ('t1_2_mni_reverse_transform', 'forward_transform')]),
            (get_atlases, buffernode, [
                ("atlas_configs", "atlas_configs")]),
            (inputnode, get_atlases,[
                ('dwi_file', 'reference_image')])
        ])
    
    for atlas in atlas_names:
        workflow.connect([
            (get_atlases,
            pe.Node(ReconDerivativesDataSink(desc=atlas,
                                            suffix="atlas",
                                            compress=True),
                    name='dsatlas_'+atlas,
                    run_without_submitting=True),
            [(('atlas_configs', _get_resampled, atlas, 'dwi_resolution_file'), 'in_file')]),
            (get_atlases,
            pe.Node(ReconDerivativesDataSink(
                                            desc=atlas,
                                            suffix="atlas",
                                            extension=".mif.gz",
                                            compress=True),
                    name='dsatlas_mifs_'+atlas,
                    run_without_submitting=True),
            [(('atlas_configs', _get_resampled, atlas, 'dwi_resolution_mif'), 'in_file')]),
            (get_atlases,
            pe.Node(ReconDerivativesDataSink(
                                            desc=atlas,
                                            extension=".txt",
                                            suffix="mrtrixLUT"),
                    name='dsatlas_mrtrix_lut_' + atlas,
                    run_without_submitting=True),
            [(('atlas_configs', _get_resampled, atlas, 'mrtrix_lut'), 'in_file')]),
            (get_atlases,
            pe.Node(ReconDerivativesDataSink(
                                            desc=atlas,
                                            extension=".txt",
                                            suffix="origLUT"),
                    name='dsatlas_orig_lut_' + atlas,
                    run_without_submitting=True),
            [(('atlas_configs', _get_resampled, atlas, 'orig_lut'), 'in_file')]),
        ])

    # Fill in the atlas datasinks
    for node in workflow.list_node_names():
        node_suffix = node.split('.')[-1]
        if node_suffix.startswith('dsatlas_'):
            workflow.connect(
                inputnode, 'dwi_file', workflow.get_node(node), 'source_file')
    
    if "mrtrix_5tt_hsv" in extras_to_make and not has_qsiprep_5tt_hsvs:
        raise Exception("Unable to create a 5tt HSV image given input data.")
    if "mrtrix_5tt_fast" in extras_to_make and not has_qsiprep_5tt_fast:
        raise Exception("Unable to create a 5tt FAST image given input data.")

    # Directly connect anything from the inputs that we haven't created here
    workflow.connect([
        (inputnode, outputnode, [(name, name) for name in connect_from_inputnode]),
        (buffernode, outputnode, [(name, name) for name in connect_from_buffernode])
    ])
    return workflow, _get_status()


def _get_first(item):
    if isinstance(item, (list, tuple)):
        return item[0]
    return item


def _get_resampled(atlas_configs, atlas_name, to_retrieve):
    return atlas_configs[atlas_name][to_retrieve]


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
