"""
Orchestrating the dwi-preprocessing workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_fsl_dwi_preproc_wf

"""

import os
import json
from pkg_resources import resource_filename as pkgr_fn

import nibabel as nb
from nipype import logging

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.interfaces import fsl, afni

from ...interfaces import DerivativesDataSink
from ...interfaces.eddy import GatherEddyInputs, ExtendedEddy
from ...interfaces.dwi_merge import MergeDWIs
from ...interfaces.images import SplitDWIs
from ...engine import Workflow

# dwi workflows
from .util import init_dwi_reference_wf, _create_mem_gb, _get_wf_name, _list_squeeze
from .registration import init_b0_to_anat_registration_wf
from .resampling import init_dwi_trans_wf
from .confounds import init_dwi_confs_wf
from .derivatives import init_dwi_derivatives_wf
from ..fieldmap.unwarp import init_fmap_unwarp_report_wf

DEFAULT_MEMORY_MIN_GB = 0.01
LOGGER = logging.getLogger('nipype.workflow')


def init_fsl_hmc_wf(impute_slice_threshold,
                    do_topup,
                    eddy_config,
                    mem_gb=3,
                    omp_nthreads=1,
                    slice_quality='outlier_n_stdev_map',
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
                    'original_files', 'rpe_b0_info', 'hmc_optimization_data']),
        name='inputnode')

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=["b0_template", "pre_sdc_template", "forward_transforms", "optimization_data",
                    "sdc_method", 'slice_quality', 'motion_params', 'cnr_map']),
        name='outputnode')

    workflow = Workflow(name=name)
    workflow.add_nodes([outputnode])

    gather_inputs = pe.Node(GatherEddyInputs(), name="gather_inputs")
    if eddy_config is None:
        # load from the defaults
        eddy_cfg_file = pkgr_fn('qsiprep.data', 'eddy_params.json')
    else:
        eddy_cfg_file = eddy_config

    with open(eddy_cfg_file, "r") as f:
        eddy_args = json.load(f)
    eddy_args['use_cuda'] = False

    eddy = pe.Node(ExtendedEddy(**eddy_args), name="eddy")
    dwi_merge = pe.Node(MergeDWIs(), name="dwi_merge")

    workflow.connect([
        (inputnode, gather_inputs, [
            ('dwi_files', 'dwi_files'),
            ('rpe_b0', 'rpe_b0'),
            ('bval_files', 'bval_files'),
            ('bvec_files', 'bvec_files'),
            ('b0_indices', 'b0_indices'),
            ('b0_images', 'b0_images'),
            ('original_files', 'original_files')]),
        (inputnode, dwi_merge, [
            ('dwi_files', 'dwi_files'),
            ('bval_files', 'bval_files'),
            ('bvec_files', 'bvec_files'),
            ('original_files', 'bids_dwi_files')]),
        (gather_inputs, eddy, [
            ('eddy_indices', 'in_index'),
            ('eddy_acqp', 'in_acqp')]),
        (dwi_merge, eddy, [
            ('out_dwi', 'in_file'),
            ('out_bval', 'in_bval'),
            ('out_bvec', 'in_bvec')]),
        (gather_inputs, outputnode, [
            ('pre_topup_image', 'pre_sdc_template')])
    ])

    if do_topup:
        inputnode.inputs.sdc_method = "TOPUP"
        topup = pe.Node(fsl.TOPUP(), name="topup")
        unwarped_mean = pe.Node(afni.TStat(), name='unwarped_mean')
        unwarped_mask = pe.Node(afni.Automask(outputtype="NIFTI_GZ"), name='unwarped_mask')

        workflow.connect([
            (gather_inputs, topup, [
                ('topup_datain', 'encoding_file'),
                ('topup_imain', 'in_file')]),
            (topup, unwarped_mean, [('out_corrected', 'in_file')]),
            (unwarped_mean, outputnode, [('out_file', 'b0_template')]),
            (topup, unwarped_mask, [('out_corrected', 'in_file')]),
            (unwarped_mask, eddy, [('out_file', 'in_mask')]),
            (topup, eddy, [
                ('out_movpar', 'in_topup_movpar'),
                ('out_fieldcoef', 'in_topup_fieldcoef')]),
        ])
    else:
        inputnode.inputs.sdc_method = "None"
        workflow.connect([
            (gather_inputs, outputnode, [('pre_topup_image', 'b0_template')])
        ])

    # Organize outputs for the rest of the pipeline
    split_eddy = pe.Node(SplitDWIs(), name="split_eddy")
    workflow.connect([
        (eddy, split_eddy, [
            ('out_bvec', 'bvec_file'),
            ('out_corrected', 'dwi_file')]),
        (dwi_merge, split_eddy, [('out_bval', 'bval_file')]),
        (dwi_merge, outputnode, [
            ('original_files', 'original_files')]),
        (split_eddy, outputnode, [
            ('dwi_files', 'corrected_dwis'),
            ('bvec_files', 'corrected_bvecs')]),
        (eddy, outputnode, [
            (slice_quality, 'slice_quality'),
            ('out_parameter', 'motion_params'),
            ('out_cnr_maps', 'cnr_map')])
    ])

    return workflow
