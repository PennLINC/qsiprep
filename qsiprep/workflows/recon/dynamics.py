"""
Dynamics and Controllability
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_controllability_wf

"""
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
import logging
from qsiprep.interfaces.connectivity import Controllability
from qsiprep.interfaces.bids import ReconDerivativesDataSink
from .interchange import input_fields
LOGGER = logging.getLogger('nipype.workflow')


def init_controllability_wf(name="controllability", output_suffix="", params={}):
    """Calculates network controllability from connectivity matrices.

    Calculates modal and average controllability using the method of Gu et al. 2015.

    Inputs

        matfile
            MATLAB format connectivity matrices from DSI Studio connectivity, MRTrix
            connectivity or Dipy Connectivity.

    Outputs

        matfile
            MATLAB format controllability values for each node in each connectivity matrix
            in the input file.


    """
    inputnode = pe.Node(niu.IdentityInterface(fields=input_fields + ['matfile']),
                        name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['matfile']),
        name="outputnode")

    calc_control = pe.Node(Controllability(**params), name='calc_control')
    workflow = pe.Workflow(name=name)
    workflow.connect([
        (inputnode, calc_control, [('matfile', 'matfile')]),
        (calc_control, outputnode, [('controllability', 'matfile')])
    ])
    if output_suffix:
        # Save the output in the outputs directory
        ds_control = pe.Node(ReconDerivativesDataSink(suffix=output_suffix),
                             name='ds_' + name,
                             run_without_submitting=True)
        workflow.connect(calc_control, 'controllability', ds_control, 'in_file')
    return workflow
