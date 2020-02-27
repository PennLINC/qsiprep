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

from nipype.interfaces import ants, fsl, utility as niu
from nipype.pipeline import engine as pe
from ...niworkflows.engine.workflows import LiterateWorkflow as Workflow
from ...niworkflows.interfaces.bids import ReadSidecarJSON
from ...niworkflows.interfaces.images import IntraModalMerge
from ...niworkflows.interfaces.masks import BETRPT

from ...interfaces import Phasediff2Fieldmap, Phases2Fieldmap, DerivativesDataSink


def siemens2rads(in_file, out_file=None):
    """
    Converts input phase difference map to rads
    """
    import numpy as np
    import nibabel as nb
    import os.path as op
    import math

    if out_file is None:
        fname, fext = op.splitext(op.basename(in_file))
        if fext == '.gz':
            fname, _ = op.splitext(fname)
        out_file = op.abspath('./%s_rads.nii.gz' % fname)

    in_file = np.atleast_1d(in_file).tolist()
    im = nb.load(in_file[0])
    data = im.get_data().astype(np.float32)
    hdr = im.header.copy()

    if len(in_file) == 2:
        data = nb.load(in_file[1]).get_data().astype(np.float32) - data
    elif (data.ndim == 4) and (data.shape[-1] == 2):
        data = np.squeeze(data[..., 1] - data[..., 0])
        hdr.set_data_shape(data.shape[:3])

    imin = data.min()
    imax = data.max()
    data = (2.0 * math.pi * (data - imin) / (imax - imin)) - math.pi
    hdr.set_data_dtype(np.float32)
    hdr.set_xyzt_units('mm')
    hdr['datatype'] = 16
    nb.Nifti1Image(data, im.affine, hdr).to_filename(out_file)
    return out_file


def demean_image(in_file, in_mask=None, out_file=None):
    """
    Demean image data inside mask
    """
    import numpy as np
    import nibabel as nb
    import os.path as op
    import math
    from nipype.utils import NUMPY_MMAP

    if out_file is None:
        fname, fext = op.splitext(op.basename(in_file))
        if fext == '.gz':
            fname, _ = op.splitext(fname)
        out_file = op.abspath('./%s_demean.nii.gz' % fname)

    im = nb.load(in_file, mmap=NUMPY_MMAP)
    data = im.get_data().astype(np.float32)
    msk = np.ones_like(data)

    if in_mask is not None:
        msk = nb.load(in_mask, mmap=NUMPY_MMAP).get_data().astype(np.float32)
        msk[msk > 0] = 1.0
        msk[msk < 1] = 0.0

    mean = np.median(data[msk == 1].reshape(-1))
    data[msk == 1] = data[msk == 1] - mean
    nb.Nifti1Image(data, im.affine, im.header).to_filename(out_file)
    return out_file


def cleanup_edge_pipeline(name='Cleanup'):
    """
    Perform some de-spiking filtering to clean up the edge of the fieldmap
    (copied from fsl_prepare_fieldmap)
    """
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['in_file', 'in_mask']), name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['out_file']), name='outputnode')

    fugue = pe.Node(
        fsl.FUGUE(
            save_fmap=True, despike_2dfilter=True, despike_threshold=2.1),
        name='Despike')
    erode = pe.Node(
        fsl.maths.MathsCommand(nan2zeros=True, args='-kernel 2D -ero'),
        name='MskErode')
    newmsk = pe.Node(
        fsl.MultiImageMaths(op_string='-sub %s -thr 0.5 -bin'), name='NewMask')
    applymsk = pe.Node(fsl.ApplyMask(nan2zeros=True), name='ApplyMask')
    join = pe.Node(niu.Merge(2), name='Merge')
    addedge = pe.Node(
        fsl.MultiImageMaths(op_string='-mas %s -add %s'), name='AddEdge')

    wf = pe.Workflow(name=name)
    wf.connect([(inputnode, fugue, [
        ('in_file', 'fmap_in_file'), ('in_mask', 'mask_file')
    ]), (inputnode, erode, [('in_mask', 'in_file')]), (inputnode, newmsk, [
        ('in_mask', 'in_file')
    ]), (erode, newmsk, [('out_file', 'operand_files')]), (fugue, applymsk, [
        ('fmap_out_file', 'in_file')
    ]), (newmsk, applymsk,
         [('out_file', 'mask_file')]), (erode, join, [('out_file', 'in1')]),
                (applymsk, join, [('out_file', 'in2')]), (inputnode, addedge, [
                    ('in_file', 'in_file')
                ]), (join, addedge, [('out', 'operand_files')]),
                (addedge, outputnode, [('out_file', 'out_file')])])
    return wf


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
