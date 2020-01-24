"""
Orchestrating the dwi-preprocessing workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_dwi_preproc_wf

"""

from nipype import logging

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu

from ...interfaces.images import ConcatRPESplits
from ...engine import Workflow

# dwi workflows
from .merge import init_merge_and_denoise_wf
from .util import get_source_file
from .qc import init_modelfree_qc_wf

DEFAULT_MEMORY_MIN_GB = 0.01
LOGGER = logging.getLogger('nipype.workflow')


def init_dwi_pre_hmc_wf(scan_groups,
                        b0_threshold,
                        preprocess_rpe_series,
                        dwi_denoise_window,
                        unringing_method,
                        dwi_no_biascorr,
                        no_b0_harmonization,
                        denoise_before_combining,
                        orientation,
                        omp_nthreads,
                        source_file,
                        low_mem,
                        name="pre_hmc_wf"):
    """
    This workflow merges and denoises dwi scans. The outputs from this workflow is
    a single dwi file (optionally denoised) and corresponding bvals, bvecs.

    In the general case, a single warped group will be sent to this workflow. However,
    since eddy expects a single 4D input file, two warped groups can be processed
    separately and merged into a 4D file. This happens when ``preprocess_rpe_series`` is
    ``True``. FSL's eddy also requires data in LAS+ orientation.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.dwi.pre_hmc import init_dwi_pre_hmc_wf
        wf = init_dwi_pre_hmc_wf(['/completely/made/up/path/sub-01_dwi.nii.gz'],
                                  b0_threshold=100,
                                  preprocess_rpe_series=False,
                                  dwi_denoise_window=7,
                                  unringing_method='mrdegibbs',
                                  dwi_no_biascorr=False,
                                  no_b0_harmonization=False,
                                  denoise_before_combining=True,
                                  omp_nthreads=1,
                                  low_mem=False)

    **Parameters**

        dwi_denoise_window : int
            window size in voxels for ``dwidenoise``. Must be odd. If 0, '
            '``dwidwenoise`` will not be run'
        unringing_method : str
            algorithm to use for removing Gibbs ringing. Options: none, mrdegibbs
        dwi_no_biascorr : bool
            run spatial bias correction (N4) on dwi series
        no_b0_harmonization : bool
            skip rescaling dwi scans to have matching b=0 intensities across scans
        denoise_before_combining : bool
            'run ``dwidenoise`` before combining dwis. Requires ``combine_all_dwis``'
        omp_nthreads : int
            Maximum number of threads an individual process may use
        orientation : str
            'LPS' or 'LAS'
        low_mem : bool
            Write uncompressed .nii files in some cases to reduce memory usage

    **Outputs**
        dwi_file
            a (potentially-denoised) dwi file
        bvec_file
            a bvec file
        bval_file
            a bval files
        b0_indices
            list of the positions of the b0 images in the dwi series
        b0_images
            list of paths to single-volume b0 images
        original_files
            list of paths to the original files that the single volumes came from
        original_grouping
            list of warped space group ids
        raw_concatenated
            4d image of the raw inputs concatenated (for QC and visualization)
    """
    workflow = Workflow(name=name)
    outputnode = pe.Node(
        niu.IdentityInterface(fields=[
            'dwi_file', 'bval_file', 'bvec_file', 'original_files', 'denoising_confounds',
            'noise_images', 'bias_images', 'qc_file', 'raw_concatenated', 'validation_reports']),
        name='outputnode')
    dwi_series_pedir = scan_groups['dwi_series_pedir']
    dwi_series = scan_groups['dwi_series']

    # Special case: Two reverse PE DWI series are going to get combined for eddy
    if preprocess_rpe_series:
        workflow.__desc__ = "Images were grouped into two phase encoding polarity groups. "
        rpe_series = scan_groups['fieldmap_info']['rpe_series']
        # Merge, denoise, split, hmc on the plus series
        plus_files, minus_files = (rpe_series, dwi_series) if dwi_series_pedir.endswith("-") \
            else (dwi_series, rpe_series)
        plus_source_file = get_source_file(plus_files, suffix='_PEplus')
        merge_plus = init_merge_and_denoise_wf(raw_dwi_files=plus_files,
                                               b0_threshold=b0_threshold,
                                               dwi_denoise_window=dwi_denoise_window,
                                               unringing_method=unringing_method,
                                               dwi_no_biascorr=dwi_no_biascorr,
                                               no_b0_harmonization=no_b0_harmonization,
                                               denoise_before_combining=denoise_before_combining,
                                               orientation=orientation,
                                               omp_nthreads=omp_nthreads,
                                               source_file=plus_source_file,
                                               calculate_qc=False,
                                               name="merge_plus")

        # Merge, denoise, split, hmc on the minus series
        minus_source_file = get_source_file(minus_files, suffix='_PEminus')
        merge_minus = init_merge_and_denoise_wf(raw_dwi_files=minus_files,
                                                b0_threshold=b0_threshold,
                                                dwi_denoise_window=dwi_denoise_window,
                                                unringing_method=unringing_method,
                                                dwi_no_biascorr=dwi_no_biascorr,
                                                no_b0_harmonization=no_b0_harmonization,
                                                denoise_before_combining=denoise_before_combining,
                                                orientation=orientation,
                                                omp_nthreads=omp_nthreads,
                                                source_file=minus_source_file,
                                                calculate_qc=False,
                                                name="merge_minus")

        # Combine the original images from the splits into one 4D series + bvals/bvecs
        concat_rpe_splits = pe.Node(ConcatRPESplits(), name="concat_rpe_splits")
        qc_wf = init_modelfree_qc_wf(dwi_files=plus_files + minus_files)

        workflow.connect([
            # Merge, denoise, combine
            (merge_plus, concat_rpe_splits, [
                ('outputnode.merged_image', 'dwi_plus'),
                ('outputnode.merged_bval', 'bval_plus'),
                ('outputnode.merged_bvec', 'bvec_plus'),
                ('outputnode.bias_images', 'bias_images_plus'),
                ('outputnode.noise_images', 'noise_images_plus'),
                ('outputnode.denoising_confounds', 'denoising_confounds_plus')]),
            (merge_minus, concat_rpe_splits, [
                ('outputnode.merged_image', 'dwi_minus'),
                ('outputnode.merged_bval', 'bval_minus'),
                ('outputnode.merged_bvec', 'bvec_minus'),
                ('outputnode.bias_images', 'bias_images_minus'),
                ('outputnode.noise_images', 'noise_images_minus'),
                ('outputnode.denoising_confounds', 'denoising_confounds_minus')]),
            # Connect to the outputnode
            (concat_rpe_splits, outputnode, [
                ('dwi_file', 'dwi_file'),
                ('bval_file', 'bval_file'),
                ('bvec_file', 'bvec_file'),
                ('original_files', 'original_files'),
                ('denoising_confounds', 'denoising_confounds'),
                ('validation_reports', 'validation_reports'),
                ('noise_images', 'noise_images'),
                ('bias_images', 'bias_images')]),
            (concat_rpe_splits, qc_wf, [
                ('bval_file', 'inputnode.bval_file'),
                ('bvec_file', 'inputnode.bvec_file')]),
            (qc_wf, outputnode, [('outputnode.qc_summary', 'qc_file')])])
        workflow.__postdesc__ = "Both groups were then merged into a single file, as required " \
                                "for the FSL workflows. "
        return workflow

    merge_dwis = init_merge_and_denoise_wf(
        raw_dwi_files=dwi_series,
        b0_threshold=b0_threshold,
        dwi_denoise_window=dwi_denoise_window,
        unringing_method=unringing_method,
        dwi_no_biascorr=dwi_no_biascorr,
        no_b0_harmonization=no_b0_harmonization,
        denoise_before_combining=denoise_before_combining,
        orientation=orientation,
        calculate_qc=True,
        source_file=source_file)

    qc_wf = init_modelfree_qc_wf(dwi_files=dwi_series)

    workflow.connect([
        (merge_dwis, outputnode, [
            ('outputnode.merged_image', 'dwi_file'),
            ('outputnode.merged_bval', 'bval_file'),
            ('outputnode.merged_bvec', 'bvec_file'),
            ('outputnode.bias_images', 'bias_images'),
            ('outputnode.noise_images', 'noise_images'),
            ('outputnode.validation_reports', 'validation_reports'),
            ('outputnode.denoising_confounds', 'denoising_confounds'),
            ('outputnode.original_files', 'original_files')]),
        (merge_dwis, qc_wf, [
            ('outputnode.merged_bval', 'inputnode.bval_file'),
            ('outputnode.merged_bvec', 'inputnode.bvec_file')]),
        (qc_wf, outputnode, [
            ('outputnode.qc_summary', 'qc_file'),
            ('outputnode.concatenated_data', 'raw_concatenated')])
    ])

    return workflow
