"""Susceptibility-distortion-correction gating helpers.

Pure functions over the method selection and collected subject data - no
config reads - shared by workflow construction and its tests.
"""

from qsiplan.methods import HmcMethod, SdcTool


def t2w_sdc_enabled(selection):
    """Whether the selected methods have a stage that can consume a T2w for SDC.

    DRBUDDI's multimodal ``--structural`` is reached whenever DRBUDDI is among
    the PEPOLAR tools; DIFFPREP's ``--epi T2Wreg`` covers the fieldmap-less
    case and is not gated on the PEPOLAR tool choice.
    """
    return SdcTool.DRBUDDI in selection.pepolar_tools or selection.hmc is HmcMethod.TORTOISE


def t2w_available_for_sdc(subject_data, selection, anat_modality):
    """Whether a T2w should drive susceptibility distortion correction.

    True only when the subject has a T2w, anatomical processing runs
    (``anat_modality`` != ``'none'``), and the selected methods actually have
    a T2w-consuming stage. Every T2w consumer takes the anatomical workflow's
    ``t2w_unfatsat``, which is only produced when ``init_anat_preproc_wf`` is
    asked for additional T2ws (see ``additional_t2ws`` in
    :func:`qsiprep.workflows.base.init_single_subject_wf`, which must stay in
    sync with this). Requesting T2w-based SDC without it leaves those nodes
    with an empty input.
    """
    return bool(subject_data.get('t2w')) and anat_modality != 'none' and t2w_sdc_enabled(selection)


def pe_readout_time(unit):
    """Get the ``TotalReadoutTime`` of a unit's lead phase-encoding series.

    The lead series is the ``+`` polarity one of a reverse-PE pair, matching
    ``unit.pe_dir``.

    Parameters
    ----------
    unit : qsiplan.adapters.PreprocUnit

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


def t2wreg_target(unit, t2w_sdc):
    """Get the structural target of DIFFPREP's T2Wreg stage.

    Mirrors ``use_t2wreg`` in :mod:`qsiprep.workflows.dwi.diffprep`. SynB0 units
    carry the T2Wreg stage without being T2WREG estimations, so the stage decides.

    Parameters
    ----------
    unit : qsiplan.adapters.PreprocUnit
    t2w_sdc : bool
        Whether a T2w is available for SDC (honors --anat-modality and --ignore).

    Returns
    -------
    str or None
        ``'synb0'`` or ``'t2w'``, or None when T2Wreg does not run.
    """
    stage = unit.run.stage_with('t2wreg')
    if stage is None:
        return None
    if stage.structural_target == 'synb0':
        return 'synb0'
    return 't2w' if t2w_sdc else None


def sdc_warp_source(unit, t2w_sdc, gre_in_eddy=False):
    """Decide where a unit's SDC displacement map comes from.

    Parameters
    ----------
    unit : qsiplan.adapters.PreprocUnit
    t2w_sdc : bool
        Whether a T2w is available for SDC.
    gre_in_eddy : bool
        Whether eddy applied the unit's GRE fieldmap itself
        (:func:`~qsiprep.utils.eddy_config.eddy_applies_gre`).

    Returns
    -------
    source : str or None
        ``'fieldwarp'`` when the method wrote a standalone warp (DRBUDDI, GRE, SyN,
        T2Wreg); ``'topup'`` when the warp is rebuilt from TOPUP's field, which
        eddy applied internally; ``'gre_in_eddy'`` when it is rebuilt the same way
        from a GRE fieldmap eddy applied; ``'topup+drbuddi'`` when DRBUDDI refined a
        TOPUP-corrected series, so its warp is only the residual; None without SDC.
    estimation_method : str or None
        The sidecar's ``EstimationMethod``.
    """
    target = t2wreg_target(unit, t2w_sdc)
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
    if unit.is_nipreps_syn:
        return 'fieldwarp', 'SyN (fieldmap-less)'
    if target is not None:
        return 'fieldwarp', 'TORTOISE T2Wreg (SynB0)' if target == 'synb0' else 'TORTOISE T2Wreg'
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
