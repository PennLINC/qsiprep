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

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from ... import config

# Fieldmap workflows
from .pepolar import init_pepolar_unwarp_wf
from .unwarp import init_sdc_unwarp_wf

DEFAULT_MEMORY_MIN_GB = 0.01


def init_sdc_wf(unit):
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
            ]
        ),
        name='inputnode',
    )

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
            (fmap_estimator_wf, sdc_unwarp_wf, [
                ('outputnode.fmap', 'inputnode.fmap'),
                ('outputnode.fmap_ref', 'inputnode.fmap_ref'),
                ('outputnode.fmap_mask', 'inputnode.fmap_mask'),
            ]),
            (sdc_unwarp_wf, outputnode, [('outputnode.out_hz', 'fieldmap_hz')]),
        ])  # fmt:skip

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
