#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
.. _sdc_base :

Automatic selection of the appropriate SDC method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the dataset metadata indicate tha more than one field map acquisition is
``IntendedFor`` (see BIDS Specification section 8.9) the following priority
will be used:

  1. :ref:`sdc_pepolar` (or **blip-up/blip-down**)

  2. :ref:`sdc_direct_b0`

  3. :ref:`sdc_phasediff`

  4. :ref:`sdc_fieldmapless`


Table of behavior (fieldmap use-cases):

=============== =========== ============= ===============
Fieldmaps found ``use_syn`` ``force_syn``     Action
=============== =========== ============= ===============
True            *           True          Fieldmaps + SyN
True            *           False         Fieldmaps
False           *           True          SyN
False           True        False         SyN
False           False       False         HMC only
=============== =========== ============= ===============


"""

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype import logging

from ...engine import Workflow
# Fieldmap workflows
from .pepolar import init_pepolar_unwarp_wf
from .syn import init_syn_sdc_wf
from .unwarp import init_sdc_unwarp_wf

LOGGER = logging.getLogger('nipype.workflow')
DEFAULT_MEMORY_MIN_GB = 0.01


def init_sdc_wf(fieldmap_info, dwi_meta, omp_nthreads=1,
                debug=False, fmap_bspline=False, fmap_demean=True):
    """
    This workflow implements the heuristics to choose a
    :abbr:`SDC (susceptibility distortion correction)` strategy.
    When no field map information is present within the BIDS inputs,
    the EXPERIMENTAL "fieldmap-less SyN" can be performed, using
    the ``--use-syn`` argument. When ``--force-syn`` is specified,
    then the "fieldmap-less SyN" is always executed and reported
    despite of other fieldmaps available with higher priority.
    In the latter case (some sort of fieldmap(s) is available and
    ``--force-syn`` is requested), then the :abbr:`SDC (susceptibility
    distortion correction)` method applied is that with the
    highest priority.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.fieldmap import init_sdc_wf
        wf = init_sdc_wf(
            fieldmap_info={
                'suffix': 'epi',
                'epi': \
                    'sub-03/ses-2/fmap/sub-03_ses-2_run-1_epi.nii.gz',
                'metadata': {'PhaseEncodingDirection': 'j-'}
            },
            dwi_meta={
                'PhaseEncodingDirection': 'j',
            },
        )

    **Parameters**

        fmaps : list of pybids dicts
            A list of dictionaries with the available fieldmaps
            (and their metadata using the key ``'metadata'`` for the
            case of *epi* fieldmaps)
        dwi_meta : dict
            BIDS metadata dictionary corresponding to the DWI run
        omp_nthreads : int
            Maximum number of threads an individual process may use
        fmap_bspline : bool
            **Experimental**: Fit B-Spline field using least-squares
        fmap_demean : bool
            Demean voxel-shift map during unwarp
        debug : bool
            Enable debugging outputs

    **Inputs**
        b0_ref
            A b0 reference calculated at a previous stage
        b0_ref_brain
            Same as above, but brain-masked
        b0_mask
            Brain mask for the DWI run
        t1_brain
            T1w image, brain-masked, for the fieldmap-less SyN method
        t1_2_mni_reverse_transform
            MNI-to-T1w transform to map prior knowledge to the T1w
            fo the fieldmap-less SyN method
        template : str
            Name of template targeted by ``template`` output space


    **Outputs**
        b0_ref
            An unwarped b0 reference
        b0_mask
            The corresponding new mask after unwarping
        out_warp
            The deformation field to unwarp the susceptibility distortions
        syn_b0_ref
            If ``--force-syn``, an unwarped b0 reference with this
            method (for reporting purposes)
        method
            Name of the method used for SDC
        fieldmap_hz
            The fieldmap in Hz for eddy

    """

    workflow = Workflow(
        name='sdc_wf' if fieldmap_info['suffix'] is not None else 'sdc_bypass_wf')
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['b0_ref', 'b0_ref_brain', 'b0_mask',
                't1_brain', 't1_2_mni_reverse_transform', 'template']),
        name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['b0_ref', 'b0_mask', 'out_warp', 'syn_b0_ref', 'method', 'fieldmap_hz']),
        name='outputnode')

    # No fieldmaps - forward inputs to outputs
    if fieldmap_info.get('suffix') is None:
        workflow.__postdesc__ = "No susceptibility distortion correction was performed."
        outputnode.inputs.method = 'None'
        workflow.connect([
            (inputnode, outputnode, [('b0_ref', 'b0_ref'),
                                     ('b0_mask', 'b0_mask')])])
        return workflow

    workflow.__postdesc__ = """\
Based on the estimated susceptibility distortion, an
unwarped b=0 reference was calculated for a more accurate
co-registration with the anatomical reference.
"""

    # PEPOLAR path
    if fieldmap_info['suffix'] in ('epi', 'rpe_series', 'dwi'):
        outputnode.inputs.method = \
            'PEB/PEPOLAR (phase-encoding based / PE-POLARity): %s' % fieldmap_info['suffix']

        epi_fmaps = fieldmap_info[fieldmap_info['suffix']]

        # We have already sorted by compatible
        sdc_unwarp_wf = init_pepolar_unwarp_wf(
            dwi_meta=dwi_meta,
            epi_fmaps=epi_fmaps,
            omp_nthreads=omp_nthreads,
            name='pepolar_unwarp_wf')

        workflow.connect([
            (inputnode, sdc_unwarp_wf, [
                ('b0_ref', 'inputnode.in_reference'),
                ('b0_mask', 'inputnode.in_mask'),
                ('b0_ref_brain', 'inputnode.in_reference_brain')]),
        ])

    # FIELDMAP path
    if fieldmap_info['suffix'] == 'fieldmap' or fieldmap_info['suffix'].startswith('phase'):
        outputnode.inputs.method = 'FMB (%s-based)' % fieldmap_info['suffix']
        # Import specific workflows here, so we don't break everything with one
        # unused workflow.
        if fieldmap_info['suffix'] == 'fieldmap':
            from .fmap import init_fmap_wf
            fmap_estimator_wf = init_fmap_wf(
                omp_nthreads=omp_nthreads,
                fmap_bspline=fmap_bspline)
            # set inputs
            fmap_estimator_wf.inputs.inputnode.fieldmap = fieldmap_info['fieldmap']
            fmap_estimator_wf.inputs.inputnode.magnitude = fieldmap_info['magnitude']

        else:
            from .phdiff import init_phdiff_wf
            fmap_estimator_wf = init_phdiff_wf(omp_nthreads=omp_nthreads,
                                               phasetype=fieldmap_info['suffix'])
            # set inputs
            if fieldmap_info['suffix'] == 'phasediff':
                fmap_estimator_wf.inputs.inputnode.phasediff = fieldmap_info['phasediff']
            else:
                # Check that fieldmap is not bipolar
                fmap_polarity = fieldmap_info['metadata'].get('DiffusionScheme', None)
                if fmap_polarity == 'Bipolar':
                    LOGGER.warning("Bipolar fieldmaps are not supported. Ignoring")
                    workflow.__postdesc__ = ""
                    outputnode.inputs.method = 'None'
                    workflow.connect([
                        (inputnode, outputnode, [('b0_ref', 'b0_ref'),
                                                 ('b0_mask', 'b0_mask')]),
                    ])
                    return workflow
                if fmap_polarity is None:
                    LOGGER.warning("Assuming phase images are Monopolar")

                fmap_estimator_wf.inputs.inputnode.phasediff = [
                    fieldmap_info['phase1'], fieldmap_info['phase2']]
            fmap_estimator_wf.inputs.inputnode.magnitude = [
                fmap_ for key, fmap_ in sorted(fieldmap_info.items())
                if key.startswith("magnitude")
            ]

        sdc_unwarp_wf = init_sdc_unwarp_wf(
            omp_nthreads=omp_nthreads,
            fmap_demean=fmap_demean,
            debug=debug,
            name='sdc_unwarp_wf')
        sdc_unwarp_wf.inputs.inputnode.metadata = dwi_meta

        workflow.connect([
            (inputnode, sdc_unwarp_wf, [
                ('b0_ref', 'inputnode.in_reference'),
                ('b0_ref_brain', 'inputnode.in_reference_brain'),
                ('b0_mask', 'inputnode.in_mask')]),
            (fmap_estimator_wf, sdc_unwarp_wf, [
                ('outputnode.fmap', 'inputnode.fmap'),
                ('outputnode.fmap_ref', 'inputnode.fmap_ref'),
                ('outputnode.fmap_mask', 'inputnode.fmap_mask')]),
            (sdc_unwarp_wf, outputnode, [
                ('outputnode.out_hz', 'fieldmap_hz')])
        ])

    # FIELDMAP-less path
    if fieldmap_info['suffix'] == 'syn':
        syn_sdc_wf = init_syn_sdc_wf(
            bold_pe=dwi_meta.get('PhaseEncodingDirection', None),
            omp_nthreads=omp_nthreads)

        workflow.connect([
            (inputnode, syn_sdc_wf, [
                ('t1_brain', 'inputnode.t1_brain'),
                ('t1_2_mni_reverse_transform', 'inputnode.t1_2_mni_reverse_transform'),
                ('b0_ref', 'inputnode.bold_ref'),
                ('template', 'inputnode.template')]),
        ])
        outputnode.inputs.method = 'FLB ("fieldmap-less", SyN-based)'
        sdc_unwarp_wf = syn_sdc_wf

    workflow.connect([
        (sdc_unwarp_wf, outputnode, [
            ('outputnode.out_warp', 'out_warp'),
            ('outputnode.out_reference', 'b0_ref')])
    ])

    return workflow
