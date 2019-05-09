"""
Orchestrating the dwi-preprocessing workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_fsl_dwi_preproc_wf

"""

import os

import nibabel as nb
from nipype import logging

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.interfaces import fsl

from ...interfaces import DerivativesDataSink
from ...interfaces.eddy import GatherEddyInputs
from ...engine import Workflow

# dwi workflows
from .util import init_dwi_reference_wf, _create_mem_gb, _get_wf_name, _list_squeeze
from .registration import init_b0_to_anat_registration_wf
from .resampling import init_dwi_trans_wf
from .confounds import init_dwi_confs_wf
from .derivatives import init_dwi_derivatives_wf

DEFAULT_MEMORY_MIN_GB = 0.01
LOGGER = logging.getLogger('nipype.workflow')


def init_fsl_hmc_wf(impute_slice_threshold,
                    do_topup,
                    eddy_config,
                    mem_gb=3,
                    omp_nthreads=1,
                    name="fsl_hmc_wf"):
    """
    This workflow controls the dwi preprocessing stages using FSL tools.


    **Parameters**
        impute_slice_threshold: float
            threshold for a slice to be replaced with imputed values. Overrides the
            parameter in ``eddy_config`` if set to a number > 0.
        do_topup: bool
            Should topup be performed before eddy? requires an rpe series or an
            rpe_b0.
        eddy_config: str
            Path to a JSON file containing settings for the call to ``eddy``.


    **Inputs**

        dwi_files: list
            List of single-volume files across all DWI series
        b0_indices: list
            Indexes into ``dwi_files`` that correspond to b=0 volumes
        bvecs: list
            List of paths to single-line bvec files
        bvals: list
            List of paths to single-line bval files
        b0_images: list
            List of single b=0 volumes
        original_files: list
            List of the files from which each DWI volume came from.

    **Outputs**


    **Subworkflows**

    """

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['dwi_files', 'rpe_b0', 'b0_indices', 'bvec_files', 'bval_files', 'b0_images',
                    'original_files', 'rpe_b0_info']),
        name='inputnode')

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=["b0_template", "forward_transforms", "noise_free_dwis",
                    "optimization_data", "ideal_images", "sdc_method"]),
        name='outputnode')

    workflow = Workflow(name=name)
    workflow.add_nodes([outputnode])

    gather_inputs = pe.Node(GatherEddyInputs(), name="gather_inputs")
    eddy = pe.Node(fsl.Eddy(), name="eddy")

    workflow.connect([
        (inputnode, gather_inputs, [
            ('dwi_files', 'dwi_files'),
            ('rpe_b0', 'rpe_b0'),
            ('bval_files', 'bval_files'),
            ('bvec_files', 'bvec_files'),
            ('b0_indices', 'b0_indices'),
            ('b0_images', 'b0_images'),
            ('original_files', 'original_files')])
    ])

    if do_topup:
        inputnode.inputs.sdc_method = "TOPUP"
        topup = pe.Node(fsl.TOPUP(), name="topup")
        workflow.connect([
            (gather_inputs, topup, [
                ('topup_datain', 'encoding_file'),
                ('topup_b0s', 'in_file')]),
            (topup, eddy, [
                ('out_movpar', 'in_topup_movpar'),
                ('out_fieldcoef', 'in_topup_fieldcoef')
            ])
        ])
    else:
        inputnode.inputs.sdc_method = "None"


    return workflow
