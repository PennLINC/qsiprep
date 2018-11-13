#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
.. _sdc_pepolar :

Fieldmapless
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
from qsiprep.utils.bids import collect_data
from collections import defaultdict
from fmriprep.engine import Workflow
import nipype.interfaces.utility as niu
from nipype.interfaces import afni
from nipype.pipeline import engine as pe
from qsiprep.interfaces import MergeDWIs
from qsiprep.workflows.dwi.b0_alignment import (init_b0_alignment_wf,
                                                init_b0_to_anat_registration_wf)
from qsiprep.workflows.dwi.merge import init_merge_and_denoise_wf
from qsiprep.workflows.anatomical import init_anat_preproc_wf
from qsiprep.workflows.fieldmap.bidirectional_pepolar import init_bidirectional_b0_unwarping_wf
from qsiprep.interfaces.images import NiftiInfo
import pkg_resources as pkgr

from nipype.pipeline import engine as pe
from nipype.interfaces import afni, ants, utility as niu
from niworkflows.interfaces import CopyHeader

from fmriprep.engine import Workflow
from ..dwi.unbiased_rigid_alignment import get_alignment_workflow


def init_dwi_no_fieldmap_wf(use_syn,
                            pe_dir,
                            dwi_denoise_window,
                            output_spaces,
                            denoise_before_combining,
                            omp_nthreads=1,
                            name="dwi_no_fieldmap_wf"):
    """
    This workflow takes in a set of b0 files with opposite phase encoding
    direction and calculates displacement fields
    (in other words, an ANTs-compatible warp file). This is intended to be run
    in the case where there are two dwi series in the same session with reverse
    phase encoding directions.

    The warp field correcting for the distortions is estimated using AFNI's
    3dQwarp, with displacement estimation limited to the target file phase
    encoding direction.

    It also calculates a new mask for the input dataset that takes into
    account the distortions.

    .. workflow ::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.fieldmap.pepolar import init_pepolar_unwarp_wf
        wf = init_pepolar_unwarp_wf(
            bold_meta={'PhaseEncodingDirection': 'j'},
            epi_fmaps=[('/dataset/sub-01/fmap/sub-01_epi.nii.gz', 'j-')],
            omp_nthreads=8)


    Inputs

        template_plus
            b0 template in one PE


    Outputs

        out_reference
            the ``in_reference`` after unwarping
        out_reference_brain
            the ``in_reference`` after unwarping and skullstripping
        out_warp
            the corresponding :abbr:`DFM (displacements field map)` to correct
            ``template_plus``
        dwi_2_t1w_affine
            the affine from the dwi reference to the T1w
        out_mask
            mask of the unwarped input file

    """

    workflow = Workflow(name=name)
    workflow.__desc__ = """\
A deformation field to correct for susceptibility distortions was estimated
based on two b0 templates created from dwi series with opposing phase-encoding
directions, using `3dQwarp` @afni (AFNI {afni_ver}).
""".format(afni_ver=''.join(['%02d' % v for v in afni.Info().version() or []]))

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['input_dwis', 't1w_brain', 't1w_to_mni_affine', 't1w_to_mni_warp']),
        name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['out_reference', 'out_reference_brain', 'out_affine_plus', 'out_warp_plus',
                'out_affine_minus', 'out_warp_minus', 'out_mask']), name='outputnode')
    merge_dwis = init_merge_and_denoise_wf(dwi_denoise_window=dwi_denoise_window,
                                           denoise_before_combining=denoise_before_combining,
                                           name="merge_dwis")
    b0_hmc = init_b0_alignment_wf(name="b0_hmc")

    return workflow
