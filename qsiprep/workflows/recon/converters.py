"""
Converting between file formats
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_mif_to_fibgz_wf
.. autofunction:: init_fibgz_to_mif_wf

"""
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
import logging
from ...interfaces.converters import FODtoFIBGZ
from ...interfaces.bids import ReconDerivativesDataSink
from .interchange import input_fields
from ...engine import Workflow
from ...interfaces.images import ConformDwi
LOGGER = logging.getLogger('nipype.workflow')


def init_mif_to_fibgz_wf(name="mif_to_fibgz", output_suffix="", params={}):
    """Converts a MRTrix mif file to DSI Studio fib file.

    This workflow uses ``sh2amp`` to sample the FODs on the standard DSI Studio
    ODF direction set. These are then loaded and converted to the fib MATLAB v4 format
    and peak directions are detected using Dipy.

    Inputs

        mif_file
            MRTrix format mif file containing sh coefficients representing FODs.

    Outputs

        fibgz
            DSI Studio fib file containing the FODs from the input ``mif_file``.

    """
    inputnode = pe.Node(niu.IdentityInterface(fields=input_fields + ["mif_file"]),
                        name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['fib_file']), name="outputnode")
    workflow = Workflow(name=name)
    convert_to_fib = pe.Node(FODtoFIBGZ(), name="convert_to_fib")
    workflow.connect([
        (inputnode, convert_to_fib, [('mif_file', 'mif_file')]),
        (convert_to_fib, outputnode, [('fib_file', 'fib_file')])
    ])
    return workflow


def init_fibgz_to_mif_wf(name="fibgz_to_mif", output_suffix="", params={}):
    """Needs Documentation"""
    inputnode = pe.Node(niu.IdentityInterface(fields=input_fields + ["mif_file"]),
                        name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['fib_file']), name="outputnode")
    workflow = Workflow(name=name)
    convert_to_fib = pe.Node(FODtoFIBGZ(), name="convert_to_fib")
    workflow.connect([
        (inputnode, convert_to_fib, [('mif_file', 'mif_file')]),
        (convert_to_fib, outputnode, [('fib_file', 'fib_file')])
    ])
    return workflow


def init_qsiprep_to_fsl_wf(name="qsiprep_to_fsl", output_suffix="", params={}):
    """Converts QSIPrep outputs (images, bval, bvec) to fsl standard orientation"""
    inputnode = pe.Node(niu.IdentityInterface(fields=input_fields),
                        name="inputnode")
    to_reorient = ["mask_file", "dwi_file", "bval_file", "bvec_file"]
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=to_reorient),
            name="outputnode")
    workflow = Workflow(name=name)

    convert_dwi_to_fsl = pe.Node(
        ConformDwi(orientation="LAS"), name="convert_to_fsl")
    convert_mask_to_fsl = pe.Node(
        ConformDwi(orientation="LAS"),name="convert_mask_to_fsl")
    workflow.connect([
        (inputnode, convert_dwi_to_fsl, [
            ('dwi_file', 'dwi_file'),
            ('bval_file', 'bval_file'),
            ('bvec_file', 'bvec_file')]),
        (convert_dwi_to_fsl, outputnode, [
            ('dwi_file', 'dwi_file'),
            ('bval_file', 'bval_file'),
            ('bvec_file', 'bvec_file')]),
        (inputnode, convert_mask_to_fsl, [('mask_file', 'dwi_file')]),
        (convert_mask_to_fsl, outputnode, [('dwi_file', 'mask_file')])
    ])

    if output_suffix:
        # Save the output in the outputs directory
        ds_dwi_file = pe.Node(
            ReconDerivativesDataSink(
                suffix=output_suffix + "_dwi"),
                name='ds_dwi_' + name,
                run_without_submitting=True)
        ds_bval_file = pe.Node(
            ReconDerivativesDataSink(
                suffix=output_suffix + "_dwi"),
                name='ds_bval_' + name,
                run_without_submitting=True)
        ds_bvec_file = pe.Node(
            ReconDerivativesDataSink(
                suffix=output_suffix + "_dwi"),
                name='ds_bvec_' + name,
                run_without_submitting=True)
        ds_mask_file = pe.Node(
            ReconDerivativesDataSink(
                suffix=output_suffix + "_mask"),
                name='ds_mask_' + name,
                run_without_submitting=True)
        workflow.connect([
            (convert_dwi_to_fsl, ds_bval_file, [('bval_file', 'in_file')]),
            (convert_dwi_to_fsl, ds_bvec_file, [('bvec_file', 'in_file')]),
            (convert_dwi_to_fsl, ds_dwi_file, [('dwi_file', 'in_file')]),
            (convert_mask_to_fsl, ds_mask_file, [('dwi_file', 'in_file')])
        ])
    return workflow