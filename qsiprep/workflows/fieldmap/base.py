# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Automatic selection of the appropriate SDC method.

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

import dataclasses

from nipype.interfaces import ants
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from ... import config
from ...interfaces.gradunwarp import InvertDisplacementField
from ..dwi.resampling import _listify

# Fieldmap workflows
from .unwarp import init_sdc_unwarp_wf

DEFAULT_MEMORY_MIN_GB = 0.01


def gre_seed_unit(unit):
    """Return ``unit`` seen through its GRE candidate.

    The returned unit is led by the series ``unit.pe_dir``
    names: init_sdc_wf builds the warp for its lead series' phase encoding, and
    that series is TORTOISE's EPI/blip-up series (the plus series of a pair).
    """
    lead = unit.plus_files[0] if unit.has_bidirectional_dwi else unit.dwi_files[0]
    return dataclasses.replace(
        unit,
        estimation=unit.gre_init_estimation,
        dwi_files=(lead, *(path for path in unit.dwi_files if path != lead)),
    )


def init_gre_seed_wf(unit, has_gradwarp, source_file, use):
    """Build the warp of ``unit``'s GRE candidate that a TORTOISE registration starts from.

    The warp is built for the unit's blip-up series (see :func:`gre_seed_unit`)
    on a b=0 reference taken before HMC and SDC. With gradient unwarping it is
    transported into the gradwarp-corrected frame of the volumes the
    registration corrects.

    Parameters
    ----------
    unit : :class:`~qsiplan.adapters.PreprocUnit`
        A unit with a ``gre_init_estimation``
    has_gradwarp : bool
        Whether ``inputnode.gradwarp_field`` carries a gradwarp field
    source_file : str
        The DWI series the b=0 reference is named after
    use : str
        What starts from the field: the DRBUDDI registration (``drbuddi``) or the
        unpile fit, which takes it as a prior (``unpile``); for the boilerplate and
        the workflow name

    Inputs
    ------
    b0_template
        b=0 average of the volumes the registration corrects, before gradient
        unwarping
    t1_brain
        T1w image, brain-masked
    t1_2_mni_reverse_transform
        MNI-to-T1w transform
    gradwarp_field
        The gradwarp displacement field, used only with ``has_gradwarp``

    Outputs
    -------
    out_warp
        The GRE-derived warp, as an ITK displacement field
    fieldmap_hz
        The GRE field in Hz, registered to the b=0 reference and on its grid
    b0_ref
        The b=0 reference unwarped with that warp
    b0_mask
        Brain mask of the b=0 reference
    """
    from ..dwi.gradwarp import connect_gradwarp_sdc_reference
    from ..dwi.util import init_dwi_reference_wf

    config.loggers.workflow.info(
        'Initializing %s for %s with GRE fieldmap %s.',
        'DRBUDDI' if use == 'drbuddi' else 'unpile',
        unit.output_name,
        unit.gre_init_estimation.b0field_id,
    )
    workflow = Workflow(name=f'{use}_gre_seed_wf')
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['b0_template', 't1_brain', 't1_2_mni_reverse_transform', 'gradwarp_field']
        ),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['out_warp', 'fieldmap_hz', 'b0_ref', 'b0_mask']),
        name='outputnode',
    )

    b0_ref_wf = init_dwi_reference_wf(source_file=source_file, name='b0_ref_wf', gen_report=False)
    sdc_wf = init_sdc_wf(gre_seed_unit(unit), gradwarp=has_gradwarp, use=use)
    sdc_wf.inputs.inputnode.template = config.workflow.anatomical_template
    workflow.connect([
        (inputnode, b0_ref_wf, [('b0_template', 'inputnode.b0_template')]),
        (inputnode, sdc_wf, [
            ('t1_brain', 'inputnode.t1_brain'),
            ('t1_2_mni_reverse_transform', 'inputnode.t1_2_mni_reverse_transform'),
        ]),
        (sdc_wf, outputnode, [
            ('outputnode.out_warp', 'out_warp'),
            ('outputnode.fieldmap_hz', 'fieldmap_hz'),
            ('outputnode.b0_ref', 'b0_ref'),
        ]),
        (b0_ref_wf, outputnode, [('outputnode.dwi_mask', 'b0_mask')]),
    ])  # fmt:skip

    ref_fields = ('outputnode.ref_image', 'outputnode.ref_image_brain', 'outputnode.dwi_mask')
    if has_gradwarp:
        connect_gradwarp_sdc_reference(workflow, inputnode, b0_ref_wf, ref_fields, sdc_wf)
    else:
        workflow.connect([
            (b0_ref_wf, sdc_wf, [
                (ref_fields[0], 'inputnode.b0_ref'),
                (ref_fields[1], 'inputnode.b0_ref_brain'),
                (ref_fields[2], 'inputnode.b0_mask'),
            ]),
        ])  # fmt:skip
    return workflow


#: The anatomical inputs callers hand :func:`init_sdc_wf`. SyN uses the first two;
#: SyN or unpile against a SynB0 also generates the synthetic b0 from the next five, and
#: unpile against the T2w takes the last.
SDC_ANAT_INPUTS = (
    't1_brain',
    't1_2_mni_reverse_transform',
    't1_preproc',
    't1_seg',
    'to_template_affine_transform',
    'acpc_inv_transform',
    'dwi_sampling_grid',
    't2w_unfatsat',
)
#: ``workflow.connect`` pairs for :data:`SDC_ANAT_INPUTS`, same names on both sides.
SDC_ANAT_CONNECTIONS = [(field, f'inputnode.{field}') for field in SDC_ANAT_INPUTS]


def syn_target(unit):
    """Return the image SyN registers the b0 to: ``'synb0'``, ``'t1w'``, or None without SyN."""
    stage = unit.run.stage_with('syn') if unit.run is not None else None
    if stage is not None:
        return stage.structural_target
    if unit.is_nipreps_syn:
        return unit.estimation.structural_target
    return None


def unpile_target(unit):
    """Return the image unpile fits the b0 against: ``'synb0'``, ``'t2w'``, or None.

    A unit led by a measured fieldmap (the GRE candidate :func:`gre_seed_unit` puts
    in front of an unpile unit) is never unpile's, whatever its plan says.
    """
    if unit.has_scanner_measured_fieldmap:
        return None
    stage = unit.run.stage_with('unpile') if unit.run is not None else None
    if stage is not None:
        return stage.structural_target
    if unit.is_unpile and unit.estimation is not None:
        return unit.estimation.structural_target
    return None


def init_sdc_wf(unit, gradwarp=False, use='apply', source_file=None):
    """Build a workflow that applies the SDC strategy for a scanner-measured fieldmap.

    This workflow implements the heuristics to choose a
    :abbr:`SDC (susceptibility distortion correction)` strategy for a
    GRE fieldmap or the fieldmap-less methods that run after head motion
    correction: SyN, which registers the b0 to the inverted T1w or to a SynB0
    synthetic b0, and unpile, which fits the field against a SynB0 or the T2w.
    Units with no such fieldmap pass through unchanged; SynB0 + TOPUP is handled
    by the HMC backends.

    .. note::
        PEPOLAR fieldmaps are not supported by this workflow. They are handled
        by the eddy (TOPUP) and TORTOISE (DRBUDDI) backends instead, and passing
        a PEPOLAR unit raises a :class:`ValueError`.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.fieldmap import init_sdc_wf
        from qsiprep.tests.preproc_factory import make_preproc_unit
        from qsiplan.models import CorrectionMethod
        wf = init_sdc_wf(
            make_preproc_unit(
                ['/data/sub-03/dwi/sub-03_dwi.nii.gz'],
                method=CorrectionMethod.NIPREPS_SYN,
                pe_dir='j',
                estimation_sources=['/data/sub-03/anat/sub-03_T1w.nii.gz'],
            ),
        )

    Parameters
    ----------
    unit : :class:`~qsiplan.adapters.PreprocUnit`
        The DWI series to correct and the fieldmap that corrects them
        (its lead series' sidecar metadata drives the GRE/SyN setup)
    gradwarp : bool, optional
        Whether the caller has a gradwarp field for this unit. A GRE fieldmap's
        warp is then estimated on raw references and transported into the
        gradwarp-corrected frame through ``inputnode.gradwarp_field``; the
        workflow's ``gradwarp_mode`` (``'transport'``, else ``'reference'``)
        tells :func:`connect_gradwarp_sdc_reference` which references to feed.
    use : str, optional
        What the caller does with a GRE fieldmap's warp, for the boilerplate:
        ``apply`` (unwarp the DWI), ``eddy`` (eddy's ``--field``), the DRBUDDI
        registration it initializes (``drbuddi``), or the unpile fit it is the
        prior of (``unpile``).
    source_file : str, optional
        Naming source for the SynB0 reportlets and the unpile derivatives.

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
    t1_preproc, t1_seg, to_template_affine_transform, acpc_inv_transform, dwi_sampling_grid
        As for :func:`~qsiprep.workflows.fieldmap.synb0.init_synb0_wf`, for SyN or unpile
        against a SynB0
    t2w_unfatsat
        The subject's T2w, for unpile against the T2w


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
    target = syn_target(unit)
    unpile_to = unpile_target(unit)
    does_sdc = unit.has_scanner_measured_fieldmap or target is not None or unpile_to is not None
    workflow = Workflow(name='sdc_wf' if does_sdc else 'sdc_bypass_wf')
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'b0_ref',
                'b0_ref_brain',
                'b0_mask',
                'template',
                'gradwarp_field',
                *SDC_ANAT_INPUTS,
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

    # No SDC to do here - forward inputs to outputs. (SynB0 + TOPUP is handled by
    # the HMC backends; SyN and unpile are handled below.)
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
        raise ValueError('PEPOLAR SDC requested, but this path should be unreachable.')

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

    # FIELDMAP-less unpile path: the b=0 matched to a SynB0 or the T2w through a
    # readout model, with a GRE candidate as the prior when the unit has one.
    if unpile_to is not None:
        from ..dwi.util import add_synb0_outputs
        from .unpile import init_unpile_sdc_wf

        if unpile_to == 't2w' and config.workflow.anat_modality == 'none':
            raise ValueError(
                f'unpile with the T2w reference was selected for {unit.dwi_files[0]}, but '
                '--anat-modality none leaves no T2w to fit against.'
            )
        gre_prior = unit.gre_init_estimation is not None
        unpile_wf = init_unpile_sdc_wf(
            unit,
            source_file=source_file,
            target=unpile_to,
            two_pass=config.workflow.two_pass_sdc,
            gre_prior=gre_prior,
        )
        outputnode.inputs.method = 'unpile ({}{})'.format(
            'SynB0' if unpile_to == 'synb0' else 'T2w', ', GRE-refined' if gre_prior else ''
        )
        workflow.connect([
            (inputnode, unpile_wf, [
                ('b0_ref', 'inputnode.b0_ref'),
                ('b0_ref_brain', 'inputnode.b0_ref_brain'),
                ('b0_mask', 'inputnode.b0_mask'),
                ('t2w_unfatsat', 'inputnode.t2w_unfatsat'),
                ('t1_preproc', 'inputnode.t1_preproc'),
                ('t1_brain', 'inputnode.t1_brain'),
                ('t1_seg', 'inputnode.t1_seg'),
                ('to_template_affine_transform', 'inputnode.to_template_affine_transform'),
                ('acpc_inv_transform', 'inputnode.acpc_inv_transform'),
                ('dwi_sampling_grid', 'inputnode.dwi_sampling_grid'),
            ]),
            (unpile_wf, outputnode, [('outputnode.out_hz', 'fieldmap_hz')]),
        ])  # fmt:skip
        if gre_prior:
            # The GRE field is estimated against the same b=0 reference unpile fits,
            # registered to it and brought onto its grid in Hz.
            gre_seed_wf = init_gre_seed_wf(unit, False, source_file, use='unpile')
            workflow.connect([
                (inputnode, gre_seed_wf, [
                    ('b0_ref', 'inputnode.b0_template'),
                    ('t1_brain', 'inputnode.t1_brain'),
                    ('t1_2_mni_reverse_transform', 'inputnode.t1_2_mni_reverse_transform'),
                ]),
                (gre_seed_wf, unpile_wf, [('outputnode.fieldmap_hz', 'inputnode.fieldmap_hz')]),
            ])  # fmt:skip
        if unpile_to == 'synb0' and source_file is not None:
            add_synb0_outputs(workflow, unpile_wf, source_file)
        sdc_unwarp_wf = unpile_wf

    # FIELDMAP-less SyN path
    if target is not None:
        from .syn import init_syn_sdc_wf

        syn_sdc_wf = init_syn_sdc_wf(
            bold_pe=unit.dwi_metadata.get('PhaseEncodingDirection', None),
            target='synb0' if target == 'synb0' else 'invt1w',
        )
        workflow.connect([
            (inputnode, syn_sdc_wf, [
                ('t1_brain', 'inputnode.t1_brain'),
                ('t1_2_mni_reverse_transform', 'inputnode.t1_2_mni_reverse_transform'),
                ('b0_ref', 'inputnode.bold_ref'),
                ('template', 'inputnode.template'),
            ]),
        ])  # fmt:skip
        if target == 'synb0':
            from ..dwi.util import add_synb0_outputs
            from .synb0 import init_synb0_wf

            outputnode.inputs.method = 'FLB ("fieldmap-less", SyN against SynB0)'
            synb0_wf = init_synb0_wf()
            workflow.connect([
                (inputnode, synb0_wf, [
                    ('t1_preproc', 'inputnode.t1_preproc'),
                    ('t1_brain', 'inputnode.t1_brain'),
                    ('t1_seg', 'inputnode.t1_seg'),
                    ('to_template_affine_transform', 'inputnode.to_template_affine_transform'),
                    ('acpc_inv_transform', 'inputnode.acpc_inv_transform'),
                    ('dwi_sampling_grid', 'inputnode.output_grid'),
                    ('b0_ref', 'inputnode.b0_ref'),
                    ('b0_ref_brain', 'inputnode.b0_ref_brain'),
                ]),
                (synb0_wf, syn_sdc_wf, [('outputnode.synthetic_b0', 'inputnode.synthetic_b0')]),
            ])  # fmt:skip
            if source_file is not None:
                add_synb0_outputs(workflow, synb0_wf, source_file)
        else:
            outputnode.inputs.method = 'FLB ("fieldmap-less", SyN-based)'
        sdc_unwarp_wf = syn_sdc_wf

    workflow.connect([
        (sdc_unwarp_wf, outputnode, [
            ('outputnode.out_warp', 'out_warp'),
            ('outputnode.out_reference', 'b0_ref'),
        ]),
    ])  # fmt:skip

    return workflow


def _gre_boilerplate(gradwarp, use):
    """Return the sentences that follow a GRE fieldmap's estimation in the methods text."""
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
    elif use == 'unpile':
        desc.append(
            'Rather than being applied directly, this field map was handed to unpile as '
            'the prior of its fit and refined against the b=0.'
        )
    # 'drbuddi': init_drbuddi_wf's boilerplate describes the seed.
    return '\n'.join(desc) + '\n' if desc else ''


def _connect_transported_warp(workflow, inputnode, sdc_unwarp_wf, outputnode):
    """Carry a GRE warp estimated on the raw b=0 into the gradwarp-corrected frame.

    The composed transform chain applies the fieldmap warp to a point in the
    gradwarp-corrected frame and only then the gradwarp field (see
    :func:`~qsiprep.workflows.dwi.gradwarp.connect_gradwarp_sdc_volumes`). A GRE
    fieldmap is measured with the same gradients as the DWI, so its content sits
    in the raw, gradient-distorted frame no matter which b=0 it is registered to.
    The warp is therefore estimated on the raw b=0, where it is exact, and
    ``gradwarp -> raw warp -> inverse gradwarp`` is composed into a warp on the
    corrected frame. This is exact up to interpolation, at the price of inverting
    the gradwarp field.
    """
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
    smooth = 'Linear' if config.execution.sloppy else 'LanczosWindowedSinc'
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
