#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
.. _sdc_phasediff :

Phase-difference B0 estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The field inhomogeneity inside the scanner (fieldmap) is proportional to the
phase drift between two subsequent :abbr:`GRE (gradient recall echo)`
sequence.


Fieldmap preprocessing workflow for fieldmap data structure
8.9.1 in BIDS 1.0.0: one phase diff and at least one magnitude image
8.9.2 in BIDS 1.0.0: two phases and at least one magnitude image

"""
import os
from nipype.interfaces import ants, fsl, utility as niu
from nipype.pipeline import engine as pe
from .utils import cleanup_edge_pipeline, siemens2rads, demean_image
from ...niworkflows.engine.workflows import LiterateWorkflow as Workflow
from ...niworkflows.interfaces.bids import ReadSidecarJSON
from ...niworkflows.interfaces.images import IntraModalMerge
from ...niworkflows.interfaces.masks import BETRPT

from ...interfaces import Phasediff2Fieldmap, Phases2Fieldmap, DerivativesDataSink


def init_phdiff_wf(omp_nthreads, phasetype='phasediff', name='phdiff_wf'):
    """
    Estimates the fieldmap using a phase-difference image and one or more
    magnitude images corresponding to two or more :abbr:`GRE (Gradient Echo sequence)`
    acquisitions. The `original code was taken from nipype
    <https://github.com/nipy/nipype/blob/master/nipype/workflows/dmri/fsl/artifacts.py#L514>`_.

    .. workflow ::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.fieldmap.phdiff import init_phdiff_wf
        wf = init_phdiff_wf(omp_nthreads=1)


    Outputs::

      outputnode.fmap_ref - The average magnitude image, skull-stripped
      outputnode.fmap_mask - The brain mask applied to the fieldmap
      outputnode.fmap - The estimated fieldmap in Hz


    """
    
    workflow = Workflow(name=name)
    workflow.__desc__ = """\
A deformation field to correct for susceptibility distortions was estimated
based on a field map that was co-registered to the BOLD reference,
using a custom workflow of *fMRIPrep* derived from D. Greve's `epidewarp.fsl`
[script](http://www.nmr.mgh.harvard.edu/~greve/fbirn/b0/epidewarp.fsl) and
further improvements of HCP Pipelines [@hcppipelines].
"""

    # Check for FSL binary
    fsl_check = os.environ.get('FSL_BUILD')
    if fsl_check=="no_fsl":
        raise Exception(
            """Container in use does not have FSL. To use this workflow, 
            please download the qsiprep container with FSL installed.""")
    inputnode = pe.Node(niu.IdentityInterface(fields=['magnitude', 'phasediff']),
                        name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['fmap', 'fmap_ref', 'fmap_mask']), name='outputnode')

    # Merge input magnitude images
    magmrg = pe.Node(IntraModalMerge(), name='magmrg')

    # de-gradient the fields ("bias/illumination artifact")
    n4 = pe.Node(ants.N4BiasFieldCorrection(dimension=3, copy_header=True),
                 name='n4', n_procs=omp_nthreads)
    bet = pe.Node(BETRPT(generate_report=True, frac=0.6, mask=True),
                name='bet')
    ds_report_fmap_mask = pe.Node(DerivativesDataSink(
        desc='brain', suffix='mask'), name='ds_report_fmap_mask',
        mem_gb=0.01, run_without_submitting=True)
    # uses mask from bet; outputs a mask
    
    # dilate = pe.Node(fsl.maths.MathsCommand(
    #     nan2zeros=True, args='-kernel sphere 5 -dilM'), name='MskDilate')

    # FSL PRELUDE will perform phase-unwrapping
    prelude = pe.Node(fsl.PRELUDE(), name='prelude')

    denoise = pe.Node(fsl.SpatialFilter(operation='median', kernel_shape='sphere',
                                        kernel_size=5), name='denoise')

    demean = pe.Node(niu.Function(function=demean_image), name='demean')

    cleanup_wf = cleanup_edge_pipeline(name="cleanup_wf")

    compfmap = pe.Node(Phasediff2Fieldmap(), name='compfmap')

    # The phdiff2fmap interface is equivalent to:
    # rad2rsec (using rads2radsec from nipype.workflows.dmri.fsl.utils)
    # pre_fugue = pe.Node(fsl.FUGUE(save_fmap=True), name='ComputeFieldmapFUGUE')
    # rsec2hz (divide by 2pi)

    if phasetype == "phasediff":
        # Read phasediff echo times
        meta = pe.Node(ReadSidecarJSON(), name='meta', mem_gb=0.01)

        # phase diff -> radians
        pha2rads = pe.Node(niu.Function(function=siemens2rads),
                           name='pha2rads')
        # Read phasediff echo times
        meta = pe.Node(ReadSidecarJSON(), name='meta', mem_gb=0.01,
                       run_without_submitting=True)
        workflow.connect([
            (meta, compfmap, [('out_dict', 'metadata')]),
            (inputnode, pha2rads, [('phasediff', 'in_file')]),
            (pha2rads, prelude, [('out', 'phase_file')]),
            (inputnode, ds_report_fmap_mask, [('phasediff', 'source_file')]),
        ])

    else:
        workflow.__desc__ += """\
The phase difference used for unwarping was calculated using two separate phase measurements
 [@pncprocessing].
    """
        # Special case for phase1, phase2 images
        meta = pe.MapNode(ReadSidecarJSON(), name='meta', mem_gb=0.01,
                          run_without_submitting=True, iterfield=['in_file'])
        phases2fmap = pe.Node(Phases2Fieldmap(), name='phases2fmap')
        workflow.connect([
            (meta, phases2fmap, [('out_dict', 'metadatas')]),
            (inputnode, phases2fmap, [('phasediff', 'phase_files')]),
            (phases2fmap, prelude, [('out_file', 'phase_file')]),
            (phases2fmap, compfmap, [('phasediff_metadata', 'metadata')]),
            (phases2fmap, ds_report_fmap_mask, [('out_file', 'source_file')])
        ])

    workflow.connect([
        (inputnode, meta, [('phasediff', 'in_file')]),
        (inputnode, magmrg, [('magnitude', 'in_files')]),
        (magmrg, n4, [('out_avg', 'input_image')]),
        (n4, prelude, [('output_image', 'magnitude_file')]),
        (n4, bet, [('output_image', 'in_file')]),
        (bet, prelude, [('mask_file', 'mask_file')]),
        (prelude, denoise, [('unwrapped_phase_file', 'in_file')]),
        (denoise, demean, [('out_file', 'in_file')]),
        (demean, cleanup_wf, [('out', 'inputnode.in_file')]),
        (bet, cleanup_wf, [('mask_file', 'inputnode.in_mask')]),
        (cleanup_wf, compfmap, [('outputnode.out_file', 'in_file')]),
        (compfmap, outputnode, [('out_file', 'fmap')]),
        (bet, outputnode, [('mask_file', 'fmap_mask'),
                           ('out_file', 'fmap_ref')]),
        (bet, ds_report_fmap_mask, [('out_report', 'in_file')]),
    ])

    return workflow
