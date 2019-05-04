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

from ...interfaces import DerivativesDataSink

from ...interfaces.reports import DiffusionSummary, GradientPlot
from ...interfaces.images import SplitDWIs, ConcatRPESplits
from ...interfaces.gradients import ExtractB0s
from ...interfaces.mrtrix import MRTrixGradientTable
from ...engine import Workflow

# dwi workflows
from .merge import init_merge_and_denoise_wf
from .hmc import init_dwi_hmc_wf
from .util import init_dwi_reference_wf, _create_mem_gb, _get_wf_name, _list_squeeze
from .registration import init_b0_to_anat_registration_wf
from .resampling import init_dwi_trans_wf
from .confounds import init_dwi_confs_wf
from .derivatives import init_dwi_derivatives_wf

DEFAULT_MEMORY_MIN_GB = 0.01
LOGGER = logging.getLogger('nipype.workflow')


def init_fsl_hmc_wf(impute_slice_threshold,
                    bidirectional_pepolar,
                    rpe_b0,
                    params_file,
                    mem_gb=3,
                    omp_nthreads=1,
                    name="fsl_hmc_wf"):
    """
    This workflow controls the dwi preprocessing stages using FSL tools.


    **Parameters**


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
            fields=['dwi_files', 'b0_indices', 'bvecs', 'bvals', 'b0_images', 'original_files']),
        name='inputnode')

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=["final_template", "forward_transforms", "noise_free_dwis",
                    "optimization_data"]),
        name='outputnode')

    workflow = Workflow(name=name)

    return workflow
