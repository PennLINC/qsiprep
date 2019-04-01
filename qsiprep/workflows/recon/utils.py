"""
Miscellaneous workflows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_discard_repeated_samples_wf
.. autofunction:: init_conform_dwi_wf


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
from qsiprep.interfaces.connectivity import Controllability
from qsiprep.interfaces.gradients import RemoveDuplicates
from dipy.core.geometry import cart2sphere
from dipy.direction import peak_directions
from dipy.core.sphere import HemiSphere
import subprocess
from scipy.io.matlab import loadmat, savemat
from pkg_resources import resource_filename as pkgr
from qsiprep.interfaces.converters import FODtoFIBGZ
from qsiprep.interfaces.bids import ReconDerivativesDataSink
from .interchange import input_fields, default_connections
from qsiprep.interfaces import ConformDwi
from qsiprep.interfaces.mrtrix import ResponseSD, EstimateFOD, MRConvert, MRTrixGradientTable
LOGGER = logging.getLogger('nipype.workflow')


def init_conform_dwi_wf(name="conform_dwi", output_suffix="", params={}):
    """If data were preprocessed elsewhere, ensure the gradients and images
    conform to LPS+ before running other parts of the pipeline."""
    inputnode = pe.Node(niu.IdentityInterface(fields=input_fields),
                        name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['dwi_file', 'bval_file', 'bvec_file', 'b_file']),
        name="outputnode")
    workflow = pe.Workflow(name=name)
    conform = pe.Node(ConformDwi(), name="conform_dwi")
    grad_table = pe.Node(MRTrixGradientTable(), name="grad_table")
    workflow.connect([
        (inputnode, conform, [('dwi_file', 'dwi_file')]),
        (conform, grad_table, [('bval_file', 'bval_file'),
                               ('bvec_file', 'bvec_file')]),
        (grad_table, outputnode, [('gradient_file', 'b_file')]),
        (conform, outputnode, [('bval_file', 'bval_file'),
                               ('bvec_file', 'bvec_file'),
                               ('dwi_file', 'dwi_file')])
    ])
    return workflow


def init_discard_repeated_samples_wf(name="discard_repeats", output_suffix="",
                                     space="T1w", params={}):
    """Remove a sample if a similar direction/gradient has already been sampled."""
    inputnode = pe.Node(niu.IdentityInterface(fields=input_fields),
                        name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['dwi_file', 'bval_file', 'bvec_file', 'local_bvec_file']),
        name="outputnode")
    workflow = pe.Workflow(name=name)

    discard_repeats = pe.Node(RemoveDuplicates(**params), name='discard_repeats')
    workflow.connect([
        (inputnode, discard_repeats, [('dwi_file', 'dwi_file'), ('bval_file', 'bval_file'),
                                      ('bvec_file', 'bvec_file')]),
        (discard_repeats, outputnode, [('dwi_file', 'dwi_file'), ('bval_file', 'bval_file'),
                                       ('bvec_file', 'bvec_file')])
    ])

    return workflow
