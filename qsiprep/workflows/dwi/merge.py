# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Merge and denoise dwi images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_dwi_preproc_wf
.. autofunction:: init_dwi_derivatives_wf

"""

import os
from collections import defaultdict

import nibabel as nb
from nipype import logging

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu

from ...interfaces import MergeDWIs, ConformDwi, ValidateImage
from ...interfaces.mrtrix import DWIDenoise

from ...engine import Workflow

DEFAULT_MEMORY_MIN_GB = 0.01
LOGGER = logging.getLogger('nipype.workflow')


def init_merge_and_denoise_wf(dwi_denoise_window,
                              denoise_before_combining,
                              orientation,
                              mem_gb=1,
                              omp_nthreads=1,
                              name="merge_and_denoise_wf"):
    """

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.dwi import init_merge_and_denoise_wf
        wf = init_merge_and_dwnoise_wf(dwi_denoise_window=7,
                                       denoise_before_combining=True,
                                       combine_all_dwis=True)

    **Parameters**

        dwi_denoise_window : int
            window size in voxels for ``dwidenoise``. Must be odd. If 0,
            ``dwidwenoise`` will not be run
        denoise_before_combining : bool
            run ``dwidenoise`` before combining dwis. Requires ``combine_all_dwis``
            If ``dwi_denoise_window > 0`` and this is ``False``, then ``dwidenoise``
            is run on the merged dwi series.


    **Inputs**

        dwi_files
            list of dwi files


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
            'merged_image', 'merged_bval', 'merged_bvec', 'noise_image', 'original_files']),
        name='outputnode')

    # validate_dwis = pe.MapNode(ValidateImage(), iterfield=[], name='validate_dwis')
    conform_dwis = pe.MapNode(
        ConformDwi(orientation=orientation), iterfield=['dwi_file'], name="conform_dwis")
    merge_dwis = pe.Node(MergeDWIs(), name='merge_dwis')

    workflow.connect([
        (inputnode, conform_dwis, [('dwi_files', 'dwi_file')]),
        (inputnode, merge_dwis, [('dwi_files', 'bids_dwi_files')]),
        (conform_dwis, merge_dwis, [('bval_file', 'bval_files'),
                                    ('bvec_file', 'bvec_files')]),
        (merge_dwis, outputnode, [('out_bval', 'merged_bval'),
                                  ('out_bvec', 'merged_bvec'),
                                  ('original_images', 'original_files')])
    ])

    if dwi_denoise_window > 0:
        denoiser = DWIDenoise(extent=(dwi_denoise_window, dwi_denoise_window,
                                      dwi_denoise_window))
        if denoise_before_combining:
            denoise = pe.MapNode(denoiser, iterfield='in_file', name='denoise')
            workflow.connect([
                (conform_dwis, denoise, [('dwi_file', 'in_file')]),
                (denoise, merge_dwis, [('out_file', 'dwi_files')]),
                # (denoise, outputnode, [('noise', 'noise_image')]),
                (merge_dwis, outputnode, [('out_dwi', 'merged_image')])
            ])
        else:
            denoise = pe.Node(denoiser, name='denoise')
            workflow.connect([
                (inputnode, merge_dwis, [('dwi_files', 'dwi_files')]),
                (merge_dwis, denoise, [('out_dwi', 'in_file')]),
                (denoise, outputnode, [('out_file', 'merged_image')])
            ])
    else:
        workflow.connect([
            (inputnode, merge_dwis, [('dwi_files', 'dwi_files')]),
            (merge_dwis, outputnode, [('out_dwi', 'merged_image')])
        ])

    return workflow
