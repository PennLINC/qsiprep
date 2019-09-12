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
    Dwi2Response)
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
        mif_file
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
            fields=['mif_file', 'wm_odf', 'wm_txt', 'gm_odf', 'gm_txt', 'csf_odf',
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
        (estimate_fod, outputnode, [('wm_odf', 'mif_file'),
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
    pass
