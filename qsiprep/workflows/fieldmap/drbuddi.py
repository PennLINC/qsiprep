#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
.. _sdc_drbuddi :

Correcting Susceptibility Distortion with DRBUDDI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DRBUDDI is part of the TORTOISE software that estimates and corrects
susceptibility distortion. It has multiple modes of operation

  1. Use $b=0$ images to estimate distortion.

  2. Perform a multimodal registration using $b=0$ images and FA images.
     This requires two DWI series with opposite phase encoding directions

  3. Either (1) or (2) but a t2w image is used as well

"""

from asyncio import gather
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype import logging

from ...engine import Workflow
from ...interfaces.fmap import get_distortion_grouping
from ...interfaces.tortoise import (
    GatherDRBUDDIInputs, DRBUDDI, DRBUDDIAggregateOutputs)
from ...interfaces.reports import TopupSummary

LOGGER = logging.getLogger('nipype.workflow')
DEFAULT_MEMORY_MIN_GB = 0.01


def init_drbuddi_wf(scan_groups, b0_threshold, raw_image_sdc, t2w_sdc, omp_nthreads=1,
                    name="drbuddi_sdc_wf", sloppy=False):
    """
    This workflow implements the heuristics to choose a
    :abbr:`SDC (susceptibility distortion correction)` strategy.


    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.fieldmap import init_drbuddi_wf
        scan_groups = {
            'dwi_series': [
                'data/tinytensor/sub-tinytensors/dwi/sub-tinytensors_dir-AP_dwi.nii.gz'],
        'dwi_series_pedir': 'j',
        'fieldmap_info': {
            'suffix': 'rpe_series',
            'rpe_series': [
                'data/tinytensor/sub-tinytensors/dwi/sub-tinytensors_dir-PA_dwi.nii.gz'],
            'epi': [
                'data/tinytensor/sub-tinytensors/fmap/sub-tinytensors_dir-AP_epi.nii.gz',
                'data/tinytensor/sub-tinytensors/fmap/sub-tinytensors_dir-PA_epi.nii.gz']},
        'concatenated_bids_name': 'sub-tinytensors'}


        wf = init_drbuddi_wf(
            scan_groups=scan_groups
        )

    **Parameters**

        scan_groups : dict of distortion groupings
            Inputs configuration for distortion correction
        omp_nthreads : int
            Maximum number of threads an individual process may use
        debug : bool
            Enable debugging outputs

    **Inputs**
        dwi_file : str
            Path to a motion/eddy corrected DWI file (in LPS+)
        bval_file : str
            Corresponding bval file for dwi_file
        bvec_file : str
            Corresponding bvec file for dwi_file (in LPS+)
        original_files : list
            List of the original BIDS file for each image in dwi_file
        t1_brain
            T1w image, brain-masked
        t2_brain
            T2w image, brain masked

    **Outputs**
        b0_ref
            An unwarped b0 reference
        b0_mask
            The corresponding new mask after unwarping
        sdc_warps
            The deformation fields to unwarp the susceptibility distortions in each image
            in dwi_file

    """

    workflow = Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['dwi_files', 'bval_files', 'bvec_files', 'original_files',
                't1_brain', 't1_wm_seg', 't2w_unfatsat']),
        name='inputnode')

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['b0_ref', 'b0_mask', 'sdc_warps', 'sdc_scaling_images', 'report', 'method',
                    # From SDC
                    "fieldmap_type", "b0_up_image", "b0_up_corrected_image", "b0_down_image",
                    "b0_down_corrected_image", "up_fa_image", "up_fa_corrected_image", "down_fa_image",
                    "down_fa_corrected_image", "t2w_image"
            ]),
        name='outputnode')
    desc = ["Data was collected with reversed phase-encode blips, resulting "
            "in pairs of images with distortions going in opposite directions."]

    fieldmap_info = scan_groups['fieldmap_info']
    if fieldmap_info['suffix'] not in ('epi', 'rpe_series', 'dwi'):
        raise Exception("DRBUDDI workflow requires epi, rpe_series or dwi fieldmaps")

    # Describe what's going on
    if fieldmap_info["suffix"] == "epi":
        desc.append("Here, b=0 reference images with reversed "
                    "phase encoding directions were used to estimate ")
    else:
        desc.append("Here, multiple DWI series were acquired with opposite phase encoding "
                    "directions. A b=0 image **and** the Fractional Anisotropy "
                    "images from both phase encoding diesctions were used together in "
                    "a multi-modal registration to estimate ")
    desc.append("the susceptibility-induced off-resonance field. ")
    if t2w_sdc:
        desc.append("A T2-weighted image was included in the multimodal registration. ")
    desc.append("An updated version of DRBUDDI [@drbuddi], part of the TORTOISE [@tortoisev3] "
                "software package was used to estimate distortion. Signal intensity was adjusted"
                "in the final interpolated images using a method similar to LSR.\n" )
    workflow.__desc__ = " ".join(desc)

    outputnode.inputs.method = \
        'PEB/PEPOLAR (phase-encoding based / PE-POLARity): %s' % fieldmap_info['suffix']

    gather_drbuddi_inputs = pe.Node(
        GatherDRBUDDIInputs(
            dwi_series_pedir=scan_groups['dwi_series_pedir'],
            epi_fmaps=fieldmap_info[fieldmap_info['suffix']],
            b0_threshold=b0_threshold,
            raw_image_sdc=raw_image_sdc,
            fieldmap_type=fieldmap_info['suffix']),
        name="gather_drbuddi_inputs")

    drbuddi = pe.Node(
        DRBUDDI(
            fieldmap_type=fieldmap_info['suffix'],
            num_threads=omp_nthreads,
            sloppy=sloppy),
        name='drbuddi',
        n_procs=omp_nthreads)

    aggregate_drbuddi = pe.Node(
        DRBUDDIAggregateOutputs(
            fieldmap_type=fieldmap_info['suffix']),
        name="aggregate_drbuddi")

    workflow.connect([
        (inputnode, gather_drbuddi_inputs, [
            ("dwi_files", "dwi_files"),
            ("bval_files", "bval_files"),
            ("bvec_files", "bvec_files"),
            ("original_files", "original_files")]),
        (gather_drbuddi_inputs, drbuddi, [
            ("blip_assignments", "blip_assignments"),
            ("blip_up_image", "blip_up_image"),
            ("blip_up_json", "blip_up_json"),
            ("blip_up_bmat", "blip_up_bmat"),
            ("blip_down_image", "blip_down_image"),
            ("blip_down_bmat", "blip_down_bmat")]),
        (inputnode, drbuddi, [('t2w_unfatsat', 'structural_image')]),
        (drbuddi, outputnode, [
            ('blip_down_b0', 'b0_down_image'),
            ('blip_up_b0', 'b0_up_image'),
            ('blip_down_b0_corrected', 'b0_down_corrected_image'),
            ('blip_up_b0_corrected', 'b0_up_corrected_image'),
            ('blip_down_FA', 'down_fa_image'),
            ('blip_up_FA', 'up_fa_image'),
            ('structural_image', 't2w_image')]),
        (drbuddi, aggregate_drbuddi, [
            ("undistorted_reference", "undistorted_reference"),
            ('bdown_to_bup_rigid_trans_h5', 'bdown_to_bup_rigid_trans_h5'),
            ('blip_down_b0', 'blip_down_b0'),
            ('blip_down_b0_corrected', 'blip_down_b0_corrected'),
            ('blip_down_b0_corrected_jac', 'blip_down_b0_corrected_jac'),
            ('blip_down_b0_quad', 'blip_down_b0_quad'),
            ('blip_up_b0', 'blip_up_b0'),
            ('blip_up_b0_corrected', 'blip_up_b0_corrected'),
            ('blip_up_b0_corrected_jac', 'blip_up_b0_corrected_jac'),
            ('blip_up_b0_quad', 'blip_up_b0_quad'),
            ('deformation_finv', 'deformation_finv'),
            ('deformation_minv', 'deformation_minv'),
            ('blip_up_FA', 'blip_up_FA'),
            ('blip_down_FA', 'blip_down_FA'),
            ('structural_image', 'structural_image')]),
        (gather_drbuddi_inputs, aggregate_drbuddi, [
            ('blip_assignments', 'blip_assignments')]),
        (aggregate_drbuddi, outputnode, [
            ("sdc_warps", "sdc_warps"),
            ("sdc_scaling_images", "sdc_scaling_images"),
            ("up_fa_corrected_image", "up_fa_corrected_image"),
            ("down_fa_corrected_image", "down_fa_corrected_image"),
            ("b0_ref", "b0_ref")])
    ])

    return workflow
