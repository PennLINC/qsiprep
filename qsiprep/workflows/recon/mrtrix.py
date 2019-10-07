"""
MRTrix workflows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_mrtrix_csd_recon_wf

"""
import json
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
from nipype.interfaces import afni, mrtrix3
from nipype.utils.filemanip import copyfile, split_filename

import logging
import os
import os.path as op
from qsiprep.interfaces.bids import ReconDerivativesDataSink
from qsiprep.interfaces.utils import GetConnectivityAtlases
from qsiprep.interfaces.connectivity import Controllability
from qsiprep.interfaces.gradients import RemoveDuplicates
from qsiprep.interfaces.mrtrix import (ResponseSD, EstimateFOD, MRTrixIngress,
    Dwi2Response, GlobalTractography, MRTrixAtlasGraph, SIFT2, TckGen)
from .interchange import input_fields

LOGGER = logging.getLogger('nipype.interface')
MULTI_RESPONSE_ALGORITHMS = ('dhollander', 'msmt_5tt')

def init_mrtrix_csd_recon_wf(name="mrtrix_recon", output_suffix="", params={}):
    """Create FOD images for WM, GM and CSF.

    This workflow uses mrtrix tools to run csd on multishell data.

    Inputs

        *Default qsiprep inputs*

    Outputs

        wm_txt
            SH fiber response function for white matter
        wm_fod
            FOD SH coefficients for white matter
        gm_txt
            SH fiber response function for gray matter
        gm_fod
            FOD SH coefficients for gray matter
        csf_txt
            SH fiber response function for CSF
        csf_fod
            FOD SH coefficients for CSF
        fod_sh_mif
            The same file as wm_fod.


    Params

        response: dict
            parameters for estimating the response function. A minimal example would be
            ``{"algorighm": "dhollander"}``
        fod: dict
            parameters for dwi2fod. A minimal example would be
            ``{"algorithm": "msmt_csd", "max_sh": [6, 8, 8]}``.


    """
    inputnode = pe.Node(niu.IdentityInterface(fields=input_fields),
                        name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['fod_sh_mif', 'wm_odf', 'wm_txt', 'gm_odf', 'gm_txt', 'csf_odf',
                    'csf_txt']),
        name="outputnode")

    # Resample anat mask
    resample_mask = pe.Node(
        afni.Resample(outputtype='NIFTI_GZ', resample_mode="NN"), name='resample_mask')

    # Response estimation
    response = params.get('response', {})
    response_algorithm = response.get('algorithm', 'dhollander')
    response['algorithm'] = response_algorithm

    # FOD estimation
    fod = params.get('fod', {})
    fod_algorithm = fod.get('algorithm', 'csd')
    fod['algorithm'] = fod_algorithm

    workflow = pe.Workflow(name=name)
    create_mif = pe.Node(MRTrixIngress(), name='create_mif')
    estimate_response = pe.Node(Dwi2Response(**response), 'estimate_response')
    estimate_fod = pe.Node(EstimateFOD(**fod), 'estimate_fod')

    use_sift2 = params.get("use_sift2", False)

    if response_algorithm == 'msmt_5tt':
        workflow.connect([
            (inputnode, estimate_response, [('mrtrix_5tt', 'mtt_file')])])

    # Connect all response functions if it's multi-response
    if fod_algorithm == 'msmt_csd':
        workflow.connect([
            (estimate_response, estimate_fod, [('wm_file', 'wm_txt'),
                                               ('gm_file', 'gm_txt'),
                                               ('csf_file', 'csf_txt')])])
    else:
        workflow.connect([
            (estimate_response, estimate_fod, [('wm_file', 'wm_txt')])
        ])

    workflow.connect([
        (inputnode, resample_mask, [('t1_brain_mask', 'in_file'),
                                    ('dwi_file', 'master')]),
        (inputnode, create_mif, [('dwi_file', 'dwi_file'),
                                 ('bval_file', 'bval_file'),
                                 ('bvec_file', 'bvec_file'),
                                 ('b_file', 'b_file')]),
        (create_mif, estimate_response, [('mif_file', 'in_file')]),
        (resample_mask, estimate_response, [('out_file', 'in_mask')]),
        (estimate_response, outputnode, [('wm_file', 'wm_txt'),
                                         ('gm_file', 'gm_txt'),
                                         ('csf_file', 'csf_txt')]),

        (create_mif, estimate_fod, [('mif_file', 'in_file')]),
        (resample_mask, estimate_fod, [('out_file', 'mask_file')]),
        (estimate_fod, outputnode, [('wm_odf', 'fod_sh_mif'),
                                    ('wm_odf', 'wm_odf'),
                                    ('gm_odf', 'gm_odf'),
                                    ('csf_odf', 'csf_odf')]),
    ])

    if output_suffix:
        ds_wm_odf = pe.Node(
            ReconDerivativesDataSink(extension='.mif.gz',
                                     desc="wmFOD",
                                     suffix=output_suffix,
                                     compress=True),
            name='ds_wm_odf',
            run_without_submitting=True)
        workflow.connect(outputnode, 'wm_odf', ds_wm_odf, 'in_file')
        ds_wm_txt = pe.Node(
            ReconDerivativesDataSink(extension='.txt',
                                     desc="wmFOD",
                                     suffix=output_suffix),
            name='ds_wm_txt',
            run_without_submitting=True)
        workflow.connect(outputnode, 'wm_txt', ds_wm_txt, 'in_file')

        if fod_algorithm == 'msmt_csd':
            ds_gm_odf = pe.Node(
                ReconDerivativesDataSink(extension='.mif.gz',
                                         desc="gmFOD",
                                         suffix=output_suffix,
                                         compress=True),
                name='ds_gm_odf',
                run_without_submitting=True)
            workflow.connect(outputnode, 'gm_odf', ds_gm_odf, 'in_file')
            ds_gm_txt = pe.Node(
                ReconDerivativesDataSink(extension='.txt',
                                         desc="gmFOD",
                                         suffix=output_suffix),
                name='ds_gm_txt',
                run_without_submitting=True)
            workflow.connect(outputnode, 'gm_txt', ds_gm_txt, 'in_file')

            ds_csf_odf = pe.Node(
                ReconDerivativesDataSink(extension='.mif.gz',
                                         desc="csfFOD",
                                         suffix=output_suffix,
                                         compress=True),
                name='ds_csf_odf',
                run_without_submitting=True)
            workflow.connect(outputnode, 'csf_odf', ds_csf_odf, 'in_file')
            ds_csf_txt = pe.Node(
                ReconDerivativesDataSink(extension='.txt',
                                         desc="csfFOD",
                                         suffix=output_suffix),
                name='ds_csf_txt',
                run_without_submitting=True)
            workflow.connect(outputnode, 'csf_txt', ds_csf_txt, 'in_file')
    return workflow


def init_global_tractography_wf(name="mrtrix_recon", output_suffix="", params={}):
    """Run multi-shell, multi-tissue global tractography

    This workflow uses mrtrix tools to run csd on multishell data.

    Inputs

        dwi_file
            Preprocessed DWI series
        wm_txt
            SH fiber response function for white matter
        gm_txt
            SH fiber response function for gray matter
        csf_txt
            SH fiber response function for CSF

    Outputs

        global_wm_fod
            FOD SH image enhanced by global tractography
        global_iso_fod
            FOD SH coefficients for other tissue compartments.
        l1_penalty
             the residual data energy image, including the L1-penalty imposed
             by the particle potential


    """
    inputnode = pe.Node(
        niu.IdentityInterface(fields=input_fields + ['gm_txt', 'wm_txt', 'csf_txt']),
        name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['fod_sh_mif', 'wm_odf', 'iso_fraction', 'tck_file']),
        name="outputnode")

    workflow = pe.Workflow(name=name)
    create_mif = pe.Node(MRTrixIngress(), name='create_mif')

    # Resample anat mask
    resample_mask = pe.Node(
        afni.Resample(outputtype='NIFTI_GZ', resample_mode="NN"), name='resample_mask')
    tck_global = pe.Node(GlobalTractography(**params), name='tck_global')
    workflow.connect([
        (inputnode, resample_mask, [('t1_brain_mask', 'in_file'),
                                    ('dwi_file', 'master')]),
        (inputnode, create_mif, [('dwi_file', 'dwi_file'),
                                 ('bval_file', 'bval_file'),
                                 ('bvec_file', 'bvec_file'),
                                 ('b_file', 'b_file')]),
        (create_mif, tck_global, [('mif_file', 'dwi_file')]),
        (resample_mask, tck_global, [('out_file', 'mask')]),
        (inputnode, tck_global, [("wm_txt", "wm_txt"),
                                 ("gm_txt", "gm_txt"),
                                 ("csf_txt", "csf_txt")]),
        (tck_global, outputnode, [("wm_odf", "wm_odf"),
                                  ("isotropic_fraction", "isotropic_fraction"),
                                  ("tck_file", "tck_file"),
                                  ("residual_energy", "residual_energy"),
                                  ("wm_odf", "fod_sh_mif")])
        ])

    if output_suffix:
        ds_globalwm_odf = pe.Node(
            ReconDerivativesDataSink(extension='.mif.gz',
                                     desc="globalwmFOD",
                                     suffix=output_suffix,
                                     compress=True),
            name='ds_globalwm_odf',
            run_without_submitting=True)
        workflow.connect(outputnode, 'wm_odf', ds_globalwm_odf, 'in_file')

        ds_isotropic_fraction = pe.Node(
            ReconDerivativesDataSink(extension='.mif.gz',
                                     desc="ISOfraction",
                                     suffix=output_suffix),
            name='ds_isotropic_fraction',
            run_without_submitting=True)
        workflow.connect(outputnode, 'isotropic_fraction', ds_isotropic_fraction, 'in_file')

        ds_tck_file = pe.Node(
            ReconDerivativesDataSink(extension='.tck.gz',
                                     desc="global",
                                     suffix=output_suffix,
                                     compress=True),
            name='ds_tck_file',
            run_without_submitting=True)
        workflow.connect(outputnode, 'tck_file', ds_tck_file, 'in_file')

        ds_residual_energy = pe.Node(
            ReconDerivativesDataSink(extension='.tck.gz',
                                     desc="residualEnergy",
                                     suffix=output_suffix,
                                     compress=True),
            name='ds_residual_energy',
            run_without_submitting=True)
        workflow.connect(outputnode, 'residual_energy', ds_residual_energy, 'in_file')

    return workflow


def init_mrtrix_tractography_wf(name="mrtrix_tracking", output_suffix="", params={}):
    """Run tractography

    This workflow uses mrtrix tools to run csd on multishell data.

    Inputs

        fod_sh_mif
            mif file containing spherical harmonics for tractography

    Outputs

        global_wm_fod
            FOD SH image enhanced by global tractography
        global_iso_fod
            FOD SH coefficients for other tissue compartments.
        l1_penalty
             the residual data energy image, including the L1-penalty imposed
             by the particle potential


    """
    inputnode = pe.Node(
        niu.IdentityInterface(fields=input_fields + ['fod_sh_mif']),
        name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['tck_file', 'sift_weights']),
        name="outputnode")

    workflow = pe.Workflow(name=name)
    # Resample anat mask
    resample_mask = pe.Node(
        afni.Resample(outputtype='NIFTI_GZ', resample_mode="NN"), name='resample_mask')
    tracking_params = params.get("tckgen", {})
    use_sift2 = params.get("use_sift2", True)
    use_5tt = params.get("use_5tt", False)
    sift_params = params.get("sift2", {})
    tracking = pe.Node(TckGen(**tracking_params), name='tractography')
    workflow.connect([
        (inputnode, resample_mask, [('t1_brain_mask', 'in_file'),
                                    ('dwi_file', 'master')]),
        (inputnode, tracking, [
            ('fod_sh_mif', 'in_file'),
            ('fod_sh_mif', 'seed_dynamic')]),
        (tracking, outputnode, [("out_file", "tck_file")])])

    if use_5tt:
        workflow.connect(inputnode, 'mrtrix_5tt', tracking, 'act_file')

    if use_sift2:
        tck_sift2 = pe.Node(SIFT2(**sift_params), name="tck_sift2")
        workflow.connect([
            (inputnode, tck_sift2, [('fod_sh_mif', 'in_fod')]),
            (tracking, tck_sift2, [('out_file', 'in_tracks')]),
            (tck_sift2, outputnode, [
                ('out_mu', 'mu'),
                ('out_weights', 'sift_weights')])
        ])
        if output_suffix:
            ds_sift_weights = pe.Node(
                ReconDerivativesDataSink(extension='.csv',
                                         desc="siftweights",
                                         suffix=output_suffix),
                name='ds_sift_weights',
                run_without_submitting=True)
            workflow.connect(outputnode, 'sift_weights', ds_sift_weights, 'in_file')
        if use_5tt:
            workflow.connect(inputnode, "mrtrix_5tt", tck_sift2, "act_file")

    if output_suffix:
        ds_tck_file = pe.Node(
            ReconDerivativesDataSink(extension='.tck',
                                     desc="tracks",
                                     suffix=output_suffix),
            name='ds_tck_file',
            run_without_submitting=True)
        workflow.connect(outputnode, 'tck_file', ds_tck_file, 'in_file')

    return workflow


def init_mrtrix_connectivity_wf(name="mrtrix_connectiity", params={},
                                output_suffix="", n_procs=1):
    """Runs ``tck2connectome`` on a ``tck`` file.abs

    Inputs

        tck_file
            mrtrix3 tck file.

    Outputs

        matfile
            A MATLAB-format file with numerous connectivity matrices for each
            atlas.
    """
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=input_fields + ['tck_file', 'sift_weights', 'atlas_configs']),
        name="inputnode")
    outputnode = pe.Node(niu.IdentityInterface(fields=['matfile']),
                         name="outputnode")
    workflow = pe.Workflow(name=name)
    conmat_params = params.get("tck2connectome", {})
    use_sift_weights = params.get("use_sift_weights", False)
    calc_connectivity = pe.Node(MRTrixAtlasGraph(**conmat_params),
                                name='calc_connectivity')
    workflow.connect([
        (inputnode, calc_connectivity, [('atlas_configs', 'atlas_configs'),
                                        ('tck_file', 'in_file')]),
        (calc_connectivity, outputnode, [('connectivity_matfile', 'matfile')])
    ])

    if use_sift_weights:
        workflow.connect([
            (inputnode, calc_connectivity, [('sift_weights', 'in_weights')])
        ])

    if output_suffix:
        # Save the output in the outputs directory
        ds_connectivity = pe.Node(ReconDerivativesDataSink(suffix=output_suffix),
                                  name='ds_' + name,
                                  run_without_submitting=True)
        workflow.connect(calc_connectivity, 'connectivity_matfile', ds_connectivity, 'in_file')
    return workflow
