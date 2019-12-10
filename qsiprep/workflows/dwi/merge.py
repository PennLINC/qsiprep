# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Merge and denoise dwi images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_dwi_preproc_wf
.. autofunction:: init_dwi_derivatives_wf

"""
import os.path as op
from nipype import logging
from nipype.pipeline import engine as pe
from nipype.utils.filemanip import split_filename
from nipype.interfaces import utility as niu
from .util import _get_wf_name
from ...interfaces import MergeDWIs, ConformDwi
from ...interfaces.mrtrix import DWIDenoise, DWIBiasCorrect, MRDeGibbs

from ...engine import Workflow

DEFAULT_MEMORY_MIN_GB = 0.01
LOGGER = logging.getLogger('nipype.workflow')


def init_merge_and_denoise_wf(raw_dwi_files,
                              dwi_denoise_window,
                              unringing_method,
                              dwi_no_biascorr,
                              no_b0_harmonization,
                              denoise_before_combining,
                              orientation,
                              b0_threshold,
                              mem_gb=1,
                              omp_nthreads=1,
                              name="merge_and_denoise_wf"):
    """

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.dwi import init_merge_and_denoise_wf
        wf = init_merge_and_dwnoise_wf(['/path/to/dwi/sub-1_dwi.nii.gz'],
                                       dwi_denoise_window=7,
                                       unringing_method='mrdegibbs',
                                       dwi_no_biascorr=False,
                                       no_b0_harmonization=False,
                                       denoise_before_combining=True,
                                       combine_all_dwis=True)

    **Parameters**

        raw_dwi_files : list
            list of raw (in their original BIDS directory) dwi nifti files
        dwi_denoise_window : int
            window size in voxels for ``dwidenoise``. Must be odd. If 0,
            ``dwidwenoise`` will not be run
        unringing_method : str
            algorithm to use for removing Gibbs ringing. Options: none, mrdegibbs
        dwi_no_biascorr : bool
            run spatial bias correction (N4) on dwi series
        no_b0_harmonization : bool
            skip rescaling dwi scans to have matching b=0 intensities across scans
        denoise_before_combining : bool
            run ``dwidenoise`` before combining dwis. Requires ``combine_all_dwis``
            If ``dwi_denoise_window > 0`` and this is ``False``, then ``dwidenoise``
            is run on the merged dwi series.

    **Outputs**

        merged_image
            dwi series, resampled to T1w space
        merged_bval
            bvals from merged images
        merged_bvec
            bvecs from merged images
        noise_image
            image(s) created by ``dwidenoise``
        original_files
            names of the original files for each volume

    """

    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['dwi_files']), name='inputnode')

    outputnode = pe.Node(
        niu.IdentityInterface(fields=[
            'merged_image', 'merged_bval', 'merged_bvec', 'noise_images', 'bias_images',
            'denoising_confounds', 'original_files']),
        name='outputnode')

    # DWIs will be merged at some point.
    merge_dwis = pe.Node(
        MergeDWIs(bids_dwi_files=raw_dwi_files,
                  b0_threshold=b0_threshold,
                  harmonize_b0_intensities=not no_b0_harmonization),
        name='merge_dwis')
    # Create a denoising workflow for each input image
    num_dwis = len(raw_dwi_files)
    conformed_bvals = pe.Node(niu.Merge(num_dwis), name='conformed_bvals')
    conformed_bvecs = pe.Node(niu.Merge(num_dwis), name='conformed_bvecs')
    conformed_images = pe.Node(niu.Merge(num_dwis), name='conformed_images')
    # derivatives from denoising
    denoising_confounds = pe.Node(niu.Merge(num_dwis), name='denoising_confounds')
    noise_images = pe.Node(niu.Merge(num_dwis), name='noise_images')
    bias_images = pe.Node(niu.Merge(num_dwis), name='bias_images')
    # Collect conformers and denoisers
    conformers = []
    denoising_wfs = []
    for dwi_num, dwi_file in enumerate(raw_dwi_files):
        # Conform each image to the requested orientation
        conformers.append(
            pe.Node(ConformDwi(orientation=orientation),
                    name="conform_dwis%02d" % dwi_num))
        conformers[-1].inputs.dwi_file = dwi_file

        if denoise_before_combining:
            # Build the denoising workflow
            _, fname, _ = split_filename(dwi_file)
            wf_name = _get_wf_name(fname).replace('preproc', 'denoise')
            denoising_wfs.append(
                init_dwi_denoising_wf(dwi_denoise_window=dwi_denoise_window,
                                      unringing_method=unringing_method,
                                      dwi_no_biascorr=dwi_no_biascorr,
                                      no_b0_harmonization=no_b0_harmonization,
                                      mem_gb=mem_gb,
                                      omp_nthreads=omp_nthreads,
                                      name=wf_name))
            workflow.connect([
                (conformers[-1], denoising_wfs[-1], [
                    ('bval_file', 'inputnode.bval_file'),
                    ('bvec_file', 'inputnode.bvec_file'),
                    ('dwi_file', 'inputnode.dwi_file')]),
                (denoising_wfs[-1], denoising_confounds, [
                    ('outputnode.confounds', 'in%d' % dwi_num)]),
                (denoising_wfs[-1], noise_images, [
                    ('outputnode.noise_image', 'in%d' % dwi_num)]),
                (denoising_wfs[-1], bias_images, [
                    ('outputnode.bias_image', 'in%d' % dwi_num)])])
            dwi_source = denoising_wfs[-1]
            edge_prefix = 'inputnode.'
        else:
            dwi_source = conformers[-1]
            edge_prefix = ''

        workflow.connect([
            (dwi_source, conformed_images, [(edge_prefix + 'dwi_file', 'in%d' % dwi_num)]),
            (dwi_source, conformed_bvals, [(edge_prefix + 'bval_file', 'in%d' % dwi_num)]),
            (dwi_source, conformed_bvecs, [(edge_prefix + 'bvec_file', 'in%d' % dwi_num)]),
        ])

    # Merge the either conformed-only or conformed-and-denoised data
    workflow.connect([
        (conformed_images, merge_dwis, [('out', 'dwi_files')]),
        (conformed_bvals, merge_dwis, [('out', 'bval_files')]),
        (conformed_bvecs, merge_dwis, [('out', 'bval_files')]),
    ])

    if denoise_before_combining:
        workflow.connect([
            (denoising_confounds, merge_dwis, [('out', 'denoising_confounds')]),
            (merge_dwis, outputnode, [('out_bval', 'merged_bval'),
                                      ('out_bvec', 'merged_bvec'),
                                      ('original_images', 'original_files'),
                                      ('denoising_confounds', 'denoising_confounds')]),
            (noise_images, outputnode, [('out', 'noise_images')]),
            (bias_images, outputnode, [('out', 'bias_images')])
        ])

    # validate_dwis = pe.MapNode(ValidateImage(), iterfield=[], name='validate_dwis')
    conform_dwis = pe.MapNode(
        ConformDwi(orientation=orientation), iterfield=['dwi_file'], name="conform_dwis")
    merge_dwis = pe.Node(
        MergeDWIs(harmonize_b0_intensities=not no_b0_harmonization), name='merge_dwis')

    workflow.connect([
        (inputnode, conform_dwis, [('dwi_files', 'dwi_file')]),
        (inputnode, merge_dwis, [('dwi_files', 'bids_dwi_files')]),
        (conform_dwis, merge_dwis, [('bval_file', 'bval_files'),
                                    ('bvec_file', 'bvec_files')]),
        (merge_dwis, outputnode, [('out_bval', 'merged_bval'),
                                  ('out_bvec', 'merged_bvec'),
                                  ('original_images', 'original_files')])
    ])

    return workflow


def init_dwi_denoising_wf(dwi_denoise_window,
                          unringing_method,
                          dwi_no_biascorr,
                          no_b0_harmonization,
                          mem_gb=1,
                          omp_nthreads=1,
                          name="denoise_wf"):

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['dwi_file', 'bval_file', 'bvec_file']),
        name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['dwi_file', 'bval_file', 'bvec_file', 'noise_image',
                                      'bias_image', 'confounds']),
        name='outputnode')
    workflow = Workflow(name=name)

    # Get IdentityInterfaces ready to hold intermediate results
    buffernodes = []

    def get_buffernode():
        num_buffers = len(buffernodes)
        return pe.Node(niu.IdentityInterface(fields=['dwi_file']),
                       name='buffer%02d' % num_buffers)

    buffernodes.append(get_buffernode())
    workflow.connect([
        (inputnode, buffernodes[-1], [('dwi_file', 'dwi_file')]),
        (inputnode, outputnode, [
            ('bval_file', 'bval_file'),
            ('bvec_file', 'bvec_file')])])

    # Which steps to apply?
    do_denoise = dwi_denoise_window > 0
    do_unringing = not unringing_method == 'none'
    do_biascorr = not dwi_no_biascorr
    harmonize_b0s = not no_b0_harmonization
    # How many steps in the denoising pipeline
    num_steps = sum(map(int, [do_denoise, do_unringing, do_biascorr, harmonize_b0s]))
    merge_confounds = pe.Node(niu.Merge(num_steps), name='merge_confounds')

    # Add the steps
    step_num = 0
    if dwi_denoise_window > 0:
        denoiser = pe.Node(
            DWIDenoise(extent=(dwi_denoise_window, dwi_denoise_window,
                               dwi_denoise_window)),
            name='denoiser')
        buffernodes.append(get_buffernode())
        workflow.connect([
            (buffernodes[-2], denoiser, [('dwi_file', 'in_file')]),
            (denoiser, buffernodes[-1], [('out_file', 'dwi_file')]),
            (denoiser, merge_confounds, [('nmse_text', 'in%d' % step_num)])
        ])
        step_num += 1

    if do_unringing:
        if unringing_method == 'mrdegibbs':
            degibbser = pe.Node(MRDeGibbs(), name='degibbser')
        buffernodes.append(get_buffernode())
        workflow.connect([
            (buffernodes[-2], degibbser, [('dwi_file', 'in_file')]),
            (degibbser, buffernodes[-1], [('out_file', 'dwi_file')]),
            (degibbser, merge_confounds, [('nmse_text', 'in%d' % step_num)])
        ])
        step_num += 1

    if do_biascorr:
        biascorr = pe.Node(DWIBiasCorrect(), name='biascorr')
        buffernodes.append(get_buffernode())
        workflow.connect([
            (buffernodes[-2], biascorr, [('dwi_file', 'in_file')]),
            (biascorr, buffernodes[-1], [('out_file', 'dwi_file')]),
            (biascorr, merge_confounds, [('nmse_text', 'in%d' % step_num)]),
            (inputnode, biascorr, [('bval_file', 'in_bval'), ('bvec_file', 'in_bvec')])
        ])
        step_num += 1

    workflow.connect([
        (buffernodes[-1], outputnode, [('dwi_file', 'dwi_file')])])
    return workflow
