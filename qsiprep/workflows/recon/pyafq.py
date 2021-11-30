"""
PyAFQ tractometry and visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_pyafq_wf

"""
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
import logging
from qsiprep.interfaces.pyafq import PyAFQRecon
from .interchange import input_fields
from ...interfaces.bids import ReconDerivativesDataSink
LOGGER = logging.getLogger('nipype.workflow')


def init_pyafq_wf(name="afq", output_suffix="", params={}):
    """Run PyAFQ on some qsiprep outputs

    Inputs



    Outputs

        afq_directory
            MATLAB format controllability values for each node in each connectivity matrix
            in the input file.


    """
    inputnode = pe.Node(niu.IdentityInterface(fields=input_fields + ['trk_file']),
                        name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['matfile']),
        name="outputnode")


    run_afq = pe.Node(PyAFQRecon(**params), name='run_afq')
    workflow = pe.Workflow(name=name)
    workflow.connect([
        (inputnode, run_afq, [('matfile', 'matfile')]),
        (run_afq, outputnode, [('controllability', 'matfile')])
    ])
    if output_suffix:
        # Save the output in the outputs directory
        ds_afq = pe.Node(ReconDerivativesDataSink(suffix=output_suffix),
                             name='ds_' + name,
                             run_without_submitting=True)
        workflow.connect(run_afq, 'out_dir', ds_afq, 'in_file')
    return workflow
