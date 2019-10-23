"""
Converting between file formats
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_mif_to_fibgz_wf
.. autofunction:: init_fibgz_to_mif_wf

"""
import numpy as np
import os
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
import os.path as op
import nibabel as nb
from nipype.interfaces.base import (BaseInterfaceInputSpec, TraitedSpec, File, SimpleInterface,
                                    InputMultiObject, OutputMultiObject, traits, isdefined)
from nipype.utils.filemanip import fname_presuffix
import logging
from dipy.core.geometry import cart2sphere
from dipy.direction import peak_directions
from dipy.core.sphere import HemiSphere
import subprocess
from scipy.io.matlab import loadmat, savemat
from pkg_resources import resource_filename as pkgr
from qsiprep.interfaces.converters import FODtoFIBGZ
from qsiprep.interfaces.bids import ReconDerivativesDataSink
from .interchange import input_fields, default_connections
from ...engine import Workflow
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
