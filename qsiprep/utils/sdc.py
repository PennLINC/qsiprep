"""Susceptibility-distortion-correction gating helpers.

Pure functions over the method selection and collected subject data - no
config reads - shared by workflow construction and its tests.
"""

from qsiplan.methods import SdcTool


def t2w_sdc_enabled(selection, policy):
    """Check whether the selected methods have a stage that can consume a T2w for SDC.

    DRBUDDI's multimodal ``--structural`` is reached whenever DRBUDDI is among
    the PEPOLAR tools, with any head motion method; the fieldmap-less unpile fit
    takes the T2w when ``--sdc-anat-reference t2w`` selects it.
    """
    return SdcTool.DRBUDDI in selection.pepolar_tools or (
        policy.fieldmapless_method == 'unpile' and policy.sdc_anat_reference == 't2w'
    )


def t2w_available_for_sdc(subject_data, selection, policy, anat_modality):
    """Check whether a T2w should drive susceptibility distortion correction.

    True only when the subject has a T2w, anatomical processing runs
    (``anat_modality`` != ``'none'``), and the selected methods actually have
    a T2w-consuming stage. Every T2w consumer takes the anatomical workflow's
    ``t2w_unfatsat``, which is only produced when ``init_anat_preproc_wf`` is
    asked for additional T2ws (see ``additional_t2ws`` in
    :func:`qsiprep.workflows.base.init_single_subject_wf`, which must stay in
    sync with this). Requesting T2w-based SDC without it leaves those nodes
    with an empty input.
    """
    return (
        bool(subject_data.get('t2w'))
        and anat_modality != 'none'
        and t2w_sdc_enabled(selection, policy)
    )


def pe_readout_time(unit):
    """Get the ``TotalReadoutTime`` of a unit's lead phase-encoding series.

    The lead series is the ``+`` polarity one of a reverse-PE pair, matching
    ``unit.pe_dir``.

    Parameters
    ----------
    unit : qsiplan.adapters.PreprocUnit
        The correction unit.

    Returns
    -------
    float or None
        Readout time in seconds, or None when the metadata is missing.
    """
    lead = unit.plus_files[0] if unit.has_bidirectional_dwi else unit.dwi_files[0]
    trt = unit.sidecar_overrides().get(lead, {}).get('TotalReadoutTime')
    if trt is None:
        trt = unit.dwi_metadata.get('TotalReadoutTime')
    return float(trt) if trt is not None else None


def eddy_applies_topup(unit):
    """Whether eddy applies a TOPUP field in its own resampling.

    Under eddy, TOPUP's field is baked into eddy's output. After DIFFPREP,
    TOPUP (SynB0) produces a warp that is composed into the final resampling
    like DRBUDDI's.

    Parameters
    ----------
    unit : qsiplan.adapters.PreprocUnit
        The correction unit.

    Returns
    -------
    bool
    """
    return unit.run.stage_with('topup') is not None and unit.run.hmc_stage.tool == 'eddy'


def sdc_warp_source(unit, gre_in_eddy=False):
    """Decide where a unit's SDC displacement map comes from.

    Parameters
    ----------
    unit : qsiplan.adapters.PreprocUnit
        The correction unit.
    gre_in_eddy : bool, optional
        Whether eddy applied the unit's GRE fieldmap itself
        (:func:`~qsiprep.utils.eddy_config.eddy_applies_gre`).

    Returns
    -------
    source : str or None
        ``'fieldwarp'`` when the method wrote a standalone warp (DRBUDDI, GRE, SyN,
        unpile); ``'topup'`` when the warp is rebuilt from TOPUP's field, which
        eddy applied internally; ``'gre_in_eddy'`` when it is rebuilt the same way
        from a GRE fieldmap eddy applied; ``'topup+drbuddi'`` when DRBUDDI refined a
        TOPUP-corrected series, so its warp is only the residual; None without SDC.
    estimation_method : str or None
        The sidecar's ``EstimationMethod``.
    """
    topup = unit.run.stage_with('topup')
    readout_time = pe_readout_time(unit)
    if unit.run.stage_with('drbuddi') is not None:
        if topup is None:
            return 'fieldwarp', 'DRBUDDI'
        if readout_time is not None:
            return 'topup+drbuddi', 'TOPUP+DRBUDDI'
        return None, None
    if unit.is_gre:
        if not gre_in_eddy:
            return 'fieldwarp', 'GRE fieldmap'
        if readout_time is not None:
            return 'gre_in_eddy', 'GRE fieldmap'
        return None, None
    syn = unit.run.stage_with('syn')
    if syn is not None:
        return (
            'fieldwarp',
            'SyN (SynB0)' if syn.structural_target == 'synb0' else 'SyN (fieldmap-less)',
        )
    unpile = unit.run.stage_with('unpile')
    if unpile is not None:
        return (
            'fieldwarp',
            'unpile (SynB0)' if unpile.structural_target == 'synb0' else 'unpile (T2w)',
        )
    if topup is not None and not eddy_applies_topup(unit):
        return 'fieldwarp', 'TOPUP (SynB0)'
    if topup is not None and readout_time is not None:
        return 'topup', 'TOPUP (SynB0)' if topup.structural_target == 'synb0' else 'TOPUP'
    return None, None


def sdc_displacement_sidecar(meta, output_dir, transform_files=None):
    """Build an SDC displacement map's sidecar.

    Parameters
    ----------
    meta : dict
        Sidecar metadata.
    output_dir : str
        Root of the derivatives dataset.
    transform_files : str or list, optional
        Written transforms that carried the map into ACPC, in the order they
        apply. Left out when part of that chain is not written.

    Returns
    -------
    dict
        ``meta``, plus BEP014's ``TransformFile`` as BIDS URIs when
        ``transform_files`` is given.
    """
    import os

    meta = dict(meta)
    if transform_files:
        paths = []
        for item in [transform_files] if isinstance(transform_files, str) else transform_files:
            paths.extend([item] if isinstance(item, str) else item)
        uris = [f'bids::{os.path.relpath(path, output_dir)}' for path in paths]
        meta['TransformFile'] = uris[0] if len(uris) == 1 else uris
    return meta
