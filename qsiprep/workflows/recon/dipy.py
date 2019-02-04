
import json
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
from nipype.utils.filemanip import copyfile, split_filename

import logging
import os
import os.path as op
from qsiprep.interfaces.bids import QsiprepOutput, ReconDerivativesDataSink
from qsiprep.interfaces.utils import GetConnectivityAtlases
from qsiprep.interfaces.connectivity import Controllability
from qsiprep.interfaces.gradients import RemoveDuplicates
from qsiprep.interfaces.mrtrix import ResponseSD, EstimateFOD, MRConvert

LOGGER = logging.getLogger('nipype.interface')
qsiprep_output_names = QsiprepOutput().output_spec.class_editable_traits()
default_connections = [(trait, trait) for trait in qsiprep_output_names]
default_input_set = set(qsiprep_output_names)


def init_dipy_brainsuite_shore_recon_wf(name="dipy_3dshore_recon", output_suffix="", params={}):
    inputnode = pe.Node(niu.IdentityInterface(fields=qsiprep_output_names),
                        name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['shore_coeffs', 'rtop', 'rtap', 'rtpp']),
        name="outputnode")

    workflow = pe.Workflow(name=name)
    create_mif = pe.Node(MRConvert(), name='create_mif')
    estimate_response = pe.Node(ResponseSD(**params['response']), 'estimate_response')
    estimate_fod = pe.Node(EstimateFOD(**params['fod']), 'estimate_fod')

    workflow.connect([
        (inputnode, create_mif, [('dwi_file', 'in_file'),
                                 ('bval_file', 'in_bval'),
                                 ('bvec_file', 'in_bvec')]),
        (create_mif, estimate_response, [('out_file', 'in_file')]),
        (inputnode, estimate_response, [('dwi_mask', 'in_mask')]),
        (estimate_response, estimate_fod, [('wm_file', 'wm_txt'),
                                           ('gm_file', 'gm_txt'),
                                           ('csf_file', 'csf_txt')])

    ])
    return workflow
