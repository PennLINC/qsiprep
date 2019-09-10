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
    GenerateMasked5tt, Dwi2Response)
from .interchange import input_fields

LOGGER = logging.getLogger('nipype.interface')
MULTI_RESPONSE_ALGORITHMS = ('dhollander', 'msmt_5tt')

def init_mrtrix_csd_recon_wf(name="mrtrix_recon", output_suffix="", params={}):
    """Create FOD images for WM, GM and CSF.

    This workflow uses mrtrix tools to run csd on multishell data.

    Inputs

        *Default qsiprep inputs*

    Outputs

        wm_fod
            FOD SH coefficients for white matter.
        gm_fod
            FOD SH coefficients for gray matter.
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
            ``{"algorithm": "msmt_csd", "max_sh": 8}``.


    """
    inputnode = pe.Node(niu.IdentityInterface(fields=input_fields),
                        name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['mif_file']),
        name="outputnode")

    response = params.get('response', {})
    response_algorithm = response.get('algorithm', 'dhollander')
    response['algorithm'] = response_algorithm
    workflow = pe.Workflow(name=name)
    create_mif = pe.Node(MRTrixIngress(), name='create_mif')
    create_5tt = pe.Node(GenerateMasked5tt(algorithm='fsl'), name='create_5tt')
    estimate_response = pe.Node(Dwi2Response(**response), 'estimate_response')
    estimate_fod = pe.Node(EstimateFOD(**params['fod']), 'estimate_fod')

    if response_algorithm == 'msmt_5tt':
        workflow.connect([
            (inputnode, create_5tt, [('t1_brain_mask', 'mask'),
                                     ('t1_preproc', 'in_file')]),
            (create_5tt, estimate_response, [('out_file', 'mtt_file')])
        ])

    # Connect all response functions if it's multi-response
    if response_algorithm in ('dhollander', 'msmt_csd'):
        workflow.connect([
            (estimate_response, estimate_fod, [('wm_file', 'wm_txt'),
                                               ('gm_file', 'gm_txt'),
                                               ('csf_file', 'csf_txt')])])
    else:
        workflow.connect([
            (estimate_response, estimate_fod, [('wm_file', 'wm_txt')])
        ])

    workflow.connect([
        (inputnode, create_mif, [('dwi_file', 'dwi_file'),
                                 ('bval_file', 'bval_file'),
                                 ('bvec_file', 'bvec_file'),
                                 ('b_file', 'b_file')]),
        (create_mif, estimate_response, [('mif_file', 'in_file')]),
        (inputnode, estimate_response, [('mask_file', 'in_mask')]),

        (create_mif, estimate_fod, [('mif_file', 'in_file')]),
        (estimate_fod, outputnode, [('wm_odf', 'mif_file')])
    ])
    return workflow
