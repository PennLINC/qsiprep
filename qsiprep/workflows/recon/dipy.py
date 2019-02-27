"""
Dipy Reconstruction workflows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_dipy_brainsuite_shore_recon_wf

"""
import json
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
from nipype.utils.filemanip import copyfile, split_filename

import logging
import os
import os.path as op
from qsiprep.interfaces.bids import QsiprepOutput, ReconDerivativesDataSink
from ...interfaces.dipy import BrainSuiteShoreReconstruction
from qsiprep.interfaces.converters import amplitudes_to_fibgz, amplitudes_to_sh_mif

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
    recon_shore = pe.Node(BrainSuiteShoreReconstruction(**params), name="recon_shore")

    workflow.connect([
        (inputnode, recon_shore, [('dwi_file', 'dwi_file'),
                                  ('bval_file', 'bval_file'),
                                  ('bvec_file', 'bvec_file'),
                                  ('mask_file', 'mask_file')]),

    ])
    return workflow
