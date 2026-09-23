# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
.. _sdc_base :

Automatic selection of the appropriate SDC method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the dataset metadata indicate that more than one field map acquisition is
``IntendedFor`` (see BIDS Specification section 8.9) the following priority
will be used:

  1. :ref:`sdc_pepolar` (or **blip-up/blip-down**)

  2. :ref:`sdc_direct_b0`

  3. :ref:`sdc_phasediff`

  4. :ref:`sdc_fieldmapless`


Table of behavior (fieldmap use-cases):

=============== =========== ===============
Fieldmaps found ``use_syn``     Action
=============== =========== ===============
True            *           Fieldmaps
False           True        SyN
False           False       HMC only
=============== =========== ===============


"""

from nipype.interfaces import ants
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from ... import config
from ...interfaces.gradunwarp import InvertDisplacementField
from ..dwi.resampling import _listify

# Fieldmap workflows
from .pepolar import init_pepolar_unwarp_wf
from .unwarp import init_sdc_unwarp_wf

DEFAULT_MEMORY_MIN_GB = 0.01


def init_sdc_wf(unit, gradwarp=False, use='apply'):
    """
    This workflow implements the heuristics to choose a
    :abbr:`SDC (susceptibility distortion correction)` strategy for a
    scanner-measured fieldmap (PEPOLAR or GRE). Units with no measured
    fieldmap pass through unchanged; the fieldmap-less T2Wreg and SyNb0 cases
    are handled by the TORTOISE backend, not here.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.fieldmap import init_sdc_wf
        from qsiprep.tests.preproc_factory import make_preproc_unit
        from qsiplan.models import CorrectionMethod
        wf = init_sdc_wf(
            make_preproc_unit(
                ['/data/sub-03/dwi/sub-03_dwi.nii.gz'],
                method=CorrectionMethod.PEPOLAR,
                pe_dir='j',
                estimation_sources=[
                    '/data/sub-03/dwi/sub-03_dwi.nii.gz',
                    '/data/sub-03/fmap/sub-03_epi.nii.gz',
                ],
            ),
        )

    Parameters
    ----------
    unit : :class:`~qsiplan.adapters.PreprocUnit`
        The DWI series to correct and the fieldmap that corrects them
        (its lead series' sidecar metadata drives the PEPOLAR/SyN setup)
    gradwarp : bool
        Whether the caller has a gradwarp field for this unit. A GRE fieldmap's
        warp is then estimated on raw references and transported into the
        gradwarp-corrected frame through ``inputnode.gradwarp_field``; the
        workflow's ``gradwarp_mode`` (``'transport'``, else ``'reference'``)
        tells :func:`connect_gradwarp_sdc_reference` which references to feed.
    use : str
        What the caller does with a GRE fieldmap's warp, for the boilerplate:
        ``apply`` (unwarp the DWI), ``eddy`` (eddy's ``--field``), or the TORTOISE
        registration it initializes, ``t2wreg`` or ``drbuddi``.

    Inputs
    ------
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


    Outputs
    -------
    b0_ref
        An unwarped b0 reference
    b0_mask
        The corresponding new mask after unwarping
    out_warp
        The deformation field to unwarp the susceptibility distortions
    syn_b0_ref
        An unwarped b0 reference from the SyN method (for reporting purposes)
    method
        Name of the method used for SDC
    fieldmap_hz
        The fieldmap in Hz for eddy
    """
    omp_nthreads = config.nipype.omp_nthreads
    does_sdc = unit.has_scanner_measured_fieldmap or unit.is_nipreps_syn
    workflow = Workflow(name='sdc_wf' if does_sdc else 'sdc_bypass_wf')
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'b0_ref',
                'b0_ref_brain',
                'b0_mask',
                't1_brain',
                't1_2_mni_reverse_transform',
                'template',
                'gradwarp_field',
            ]
        ),
        name='inputnode',
    )
    workflow.gradwarp_mode = 'reference'

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['b0_ref', 'b0_mask', 'out_warp', 'syn_b0_ref', 'method', 'fieldmap_hz']
        ),
        name='outputnode',
    )

    # No SDC to do here - forward inputs to outputs. (The fieldmap-less T2Wreg and
    # SyNb0 cases are handled by the TORTOISE backend; classic SyN is handled below.)
    if not does_sdc:
        workflow.__postdesc__ = 'No susceptibility distortion correction was performed.'
        outputnode.inputs.method = 'None'
        workflow.connect([
            (inputnode, outputnode, [
                ('b0_ref', 'b0_ref'),
                ('b0_mask', 'b0_mask'),
            ]),
        ])  # fmt:skip
        return workflow

    workflow.__postdesc__ = """\
Based on the estimated susceptibility distortion, an
unwarped b=0 reference was calculated for a more accurate
co-registration with the anatomical reference.
"""

    # PEPOLAR path
    if unit.is_pepolar:
        outputnode.inputs.method = 'PEB/PEPOLAR (phase-encoding based / PE-POLARity)'

        # The reverse blip is the opposite-polarity DWI series when both are
        # present, otherwise the dedicated epi fieldmap(s).
        epi_fmaps = list(unit.minus_files) if unit.has_bidirectional_dwi else list(unit.extra_b0)

        # We have already sorted by compatible
        sdc_unwarp_wf = init_pepolar_unwarp_wf(
            dwi_meta=unit.dwi_metadata,
            epi_fmaps=epi_fmaps,
            omp_nthreads=omp_nthreads,
            name='pepolar_unwarp_wf',
        )

        workflow.connect([
            (inputnode, sdc_unwarp_wf, [
                ('b0_ref', 'inputnode.in_reference'),
                ('b0_mask', 'inputnode.in_mask'),
                ('b0_ref_brain', 'inputnode.in_reference_brain'),
            ]),
        ])  # fmt:skip

    # FIELDMAP path
    if unit.is_gre:
        gre = unit.gre_files()
        outputnode.inputs.method = f'FMB ({unit.gre_suffix}-based)'
        # Import specific workflows here, so we don't break everything with one
        # unused workflow.
        if unit.gre_suffix == 'fieldmap':
            from .fmap import init_fmap_wf

            fmap_estimator_wf = init_fmap_wf()
            # set inputs
            fmap_estimator_wf.inputs.inputnode.fieldmap = gre['fieldmap']
            fmap_estimator_wf.inputs.inputnode.magnitude = gre['magnitude']

        else:
            from .phdiff import init_phdiff_wf

            fmap_estimator_wf = init_phdiff_wf(phasetype=unit.gre_suffix)
            # set inputs
            if unit.gre_suffix == 'phasediff':
                fmap_estimator_wf.inputs.inputnode.phasediff = gre['phasediff']
                fmap_estimator_wf.inputs.inputnode.phase_meta = unit.metadata_for(gre['phasediff'])
            else:
                # Check that fieldmap is not bipolar
                fmap_polarity = unit.metadata_for(gre['phase1']).get('DiffusionScheme', None)
                if fmap_polarity == 'Bipolar':
                    config.loggers.workflow.warning(
                        'Bipolar fieldmaps are not supported. Ignoring'
                    )
                    workflow.__postdesc__ = ''
                    outputnode.inputs.method = 'None'
                    workflow.connect([
                        (inputnode, outputnode, [
                            ('b0_ref', 'b0_ref'),
                            ('b0_mask', 'b0_mask'),
                        ]),
                    ])  # fmt:skip
                    return workflow

                if fmap_polarity is None:
                    config.loggers.workflow.warning('Assuming phase images are Monopolar')

                fmap_estimator_wf.inputs.inputnode.phasediff = [gre['phase1'], gre['phase2']]
                fmap_estimator_wf.inputs.inputnode.phase_meta = [
                    unit.metadata_for(gre['phase1']),
                    unit.metadata_for(gre['phase2']),
                ]

            fmap_estimator_wf.inputs.inputnode.magnitude = [
                path for suffix, path in sorted(gre.items()) if suffix.startswith('magnitude')
            ]

        sdc_unwarp_wf = init_sdc_unwarp_wf(name='sdc_unwarp_wf')
        sdc_unwarp_wf.inputs.inputnode.metadata = unit.dwi_metadata

        workflow.connect([
            (inputnode, sdc_unwarp_wf, [
                ('b0_ref', 'inputnode.in_reference'),
                ('b0_ref_brain', 'inputnode.in_reference_brain'),
                ('b0_mask', 'inputnode.in_mask'),
            ]),
            (sdc_unwarp_wf, outputnode, [('outputnode.out_hz', 'fieldmap_hz')]),
        ])  # fmt:skip

        workflow.__postdesc__ = _gre_boilerplate(gradwarp, use)
        workflow.connect([
            (fmap_estimator_wf, sdc_unwarp_wf, [
                ('outputnode.fmap', 'inputnode.fmap'),
                ('outputnode.fmap_ref', 'inputnode.fmap_ref'),
                ('outputnode.fmap_mask', 'inputnode.fmap_mask'),
            ]),
        ])  # fmt:skip
        if gradwarp:
            workflow.gradwarp_mode = 'transport'
            _connect_transported_warp(workflow, inputnode, sdc_unwarp_wf, outputnode)
            return workflow

    # FIELDMAP-less classic SyN path
    if unit.is_nipreps_syn:
        from .syn import init_syn_sdc_wf

        syn_sdc_wf = init_syn_sdc_wf(bold_pe=unit.dwi_metadata.get('PhaseEncodingDirection', None))
        outputnode.inputs.method = 'FLB ("fieldmap-less", SyN-based)'
        workflow.connect([
            (inputnode, syn_sdc_wf, [
                ('t1_brain', 'inputnode.t1_brain'),
                ('t1_2_mni_reverse_transform', 'inputnode.t1_2_mni_reverse_transform'),
                ('b0_ref', 'inputnode.bold_ref'),
                ('template', 'inputnode.template'),
            ]),
        ])  # fmt:skip
        sdc_unwarp_wf = syn_sdc_wf

    workflow.connect([
        (sdc_unwarp_wf, outputnode, [
            ('outputnode.out_warp', 'out_warp'),
            ('outputnode.out_reference', 'b0_ref'),
        ]),
    ])  # fmt:skip

    return workflow


def _gre_boilerplate(gradwarp, use):
    """The sentences that follow a GRE fieldmap's estimation in the methods text."""
    desc = []
    if gradwarp:
        desc.append(
            'As the field map is subject to the same gradient nonlinearity as the DWI, '
            'the deformation was estimated against the b=0 reference before gradient '
            'nonlinearity correction, then moved into the corrected space by composing '
            'it with the gradient nonlinearity displacement field and its inverse.'
        )
    if use == 'apply':
        desc.append(
            'Based on the estimated susceptibility distortion, an unwarped b=0 '
            'reference was calculated for a more accurate co-registration with the '
            'anatomical reference.'
        )
    elif use == 'eddy':
        desc.append(
            'Rather than being applied after eddy, the field map (in Hz, rigidly aligned '
            "to eddy's first volume) was passed to eddy, which corrected susceptibility "
            'distortion within its own model.'
        )
    elif use == 't2wreg':
        desc.append(
            'Rather than being applied directly, this deformation initialized the T2Wreg '
            'registration'
            + (
                ', held fixed through its multi-resolution pyramid so that each stage '
                'estimated only a residual correction on top of it.'
                if config.workflow.gre_init_keep_fixed
                else '.'
            )
        )
    # 'drbuddi': init_drbuddi_wf's boilerplate describes the seed.
    return '\n'.join(desc) + '\n' if desc else ''


# --- GRE fieldmaps and gradient unwarping -------------------------------------
#
# The composed transform chain applies the fieldmap warp to a point in the
# gradwarp-corrected frame and only then the gradwarp field (see the note above
# ``connect_gradwarp_sdc_volumes`` in ``dwi/gradwarp.py``). A GRE fieldmap is
# measured with the same gradients as the DWI, so its content sits in the raw,
# gradient-distorted frame no matter which b=0 it is registered to. So the warp
# is estimated on the raw b=0, where it is exact, and ``gradwarp -> raw warp ->
# inverse gradwarp`` is composed into a warp on the corrected frame. This is
# exact up to interpolation, at the price of inverting the gradwarp field.


def _connect_transported_warp(workflow, inputnode, sdc_unwarp_wf, outputnode):
    invert_gradwarp = pe.Node(InvertDisplacementField(), name='invert_gradwarp')
    # antsApplyTransforms applies the first-listed transform first to a point of
    # the output grid, so this composes gradwarp, then the raw-frame warp, then
    # the inverse gradwarp -- ``phi^-1 o (id + D_raw) o phi``.
    transport_stack = pe.Node(niu.Merge(3), name='transport_stack')
    transport_warp = pe.Node(
        ants.ApplyTransforms(
            dimension=3,
            interpolation='Linear',
            float=True,
            print_out_composite_warp_file=True,
            output_image='transported_sdc_warp.nii.gz',
        ),
        name='transport_warp',
    )
    # The unwarped reference is still in the raw frame; the coregistration
    # reference must be gradwarp-corrected like every other branch's.
    smooth = 'NearestNeighbor' if config.execution.sloppy else 'LanczosWindowedSinc'
    gradwarp_unwarped_ref = pe.Node(
        ants.ApplyTransforms(dimension=3, interpolation=smooth, float=True),
        name='gradwarp_unwarped_ref',
    )
    workflow.connect([
        (inputnode, invert_gradwarp, [('gradwarp_field', 'in_file')]),
        (inputnode, transport_stack, [('gradwarp_field', 'in1')]),
        (sdc_unwarp_wf, transport_stack, [('outputnode.out_warp', 'in2')]),
        (invert_gradwarp, transport_stack, [('out_file', 'in3')]),
        (transport_stack, transport_warp, [('out', 'transforms')]),
        (inputnode, transport_warp, [
            ('b0_ref', 'input_image'),
            ('b0_ref', 'reference_image'),
        ]),
        (transport_warp, outputnode, [('output_image', 'out_warp')]),
        (inputnode, gradwarp_unwarped_ref, [(('gradwarp_field', _listify), 'transforms')]),
        (sdc_unwarp_wf, gradwarp_unwarped_ref, [
            ('outputnode.out_reference', 'input_image'),
            ('outputnode.out_reference', 'reference_image'),
        ]),
        (gradwarp_unwarped_ref, outputnode, [('output_image', 'b0_ref')]),
    ])  # fmt:skip
