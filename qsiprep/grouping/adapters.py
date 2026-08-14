"""Translate a :class:`~.models.DWIGrouping` into the legacy workflow inputs.

The workflow builders in :mod:`qsiprep.workflows` still consume the ``dict``
shapes that :func:`qsiprep.utils.grouping.group_dwi_scans` used to produce
(``dwi_series``/``fieldmap_info``/``dwi_series_pedir``/
``concatenated_bids_name``). This module is the bridge: it reads the grouping
model and emits those shapes without any of the code downstream having to
change yet.

One legacy "scan group" is emitted per *applied estimation* (the natural unit
that a single TOPUP+eddy / DIFFPREP run corrects), plus one per uncorrected
distortion group. Merging several distortion groups back into a single
concatenated output happens after HMC and is deliberately left to a later
change; here every scan group is its own output (an identity
``concatenation_scheme``).
"""

from __future__ import annotations

from collections import defaultdict

from .models import DWIGrouping, EstimationMethod, derive_output_name

#: Which sidecar suffixes name the GRE files of each estimation method.
_GRE_SUFFIX = {
    EstimationMethod.PHASEDIFF: 'phasediff',
    EstimationMethod.PHASES: 'phase1',
    EstimationMethod.DIRECT: 'fieldmap',
}


def backend_for_config(hmc_model: str, pepolar_method: str) -> str:
    """Map the CLI motion/PEPOLAR settings to a :data:`~.validation.BACKENDS` name.

    ``eddy`` runs TOPUP for PEPOLAR fieldmaps (the ``fsl`` backend) unless
    DRBUDDI refinement is requested, which makes it the two-stage ``mixed``
    backend. Everything else - ``tortoise`` and the deprecated SHORELine
    models (``3dSHORE``, ``tensor``, ``none``) - shares TORTOISE's DRBUDDI
    feasibility semantics, so it maps to ``tortoise``.
    """
    if hmc_model == 'eddy':
        return 'mixed' if 'DRBUDDI' in pepolar_method else 'fsl'
    return 'tortoise'


def to_legacy_scan_groups(grouping: DWIGrouping) -> tuple[list[dict], dict]:
    """Render a grouping as ``(scan_groups, concatenation_scheme)``.

    ``scan_groups`` matches the contract of the retired
    :func:`qsiprep.utils.grouping.group_dwi_scans`; ``concatenation_scheme``
    maps every scan group's name to itself (post-HMC concatenation of
    distinct distortion groups is left to a later change).
    """
    scan_groups = []
    for concat in sorted(grouping.concatenation_groups.values(), key=lambda c: c.output_name):
        by_estimation = defaultdict(list)
        for dgroup in grouping.distortion_groups_in(concat.multipart_id):
            by_estimation[dgroup.b0field_source].append(dgroup)

        for b0field_id, dgroups in sorted(by_estimation.items(), key=lambda item: item[0] or ''):
            if b0field_id is None:
                scan_groups.extend(_uncorrected_group(dgroup) for dgroup in dgroups)
            else:
                scan_groups.append(_corrected_group(grouping, b0field_id, dgroups))

    names = [group['concatenated_bids_name'] for group in scan_groups]
    return scan_groups, {name: name for name in names}


def _uncorrected_group(dgroup) -> dict:
    dwi_series = list(dgroup.dwi_files)
    return {
        'dwi_series': dwi_series,
        'fieldmap_info': {'suffix': None},
        'dwi_series_pedir': dgroup.signature.pe_dir or '',
        'concatenated_bids_name': derive_output_name(dwi_series),
    }


def _corrected_group(grouping: DWIGrouping, b0field_id: str, dgroups) -> dict:
    estimation = grouping.estimations[b0field_id]
    member_dwi = sorted({path for dgroup in dgroups for path in dgroup.dwi_files})

    if estimation.method is EstimationMethod.PEPOLAR:
        dwi_series, fieldmap_info, pedir = _pepolar_fieldmap(grouping, estimation, member_dwi)
    else:
        dwi_series = member_dwi
        pedir = grouping.files[member_dwi[0]].signature.pe_dir or ''
        if estimation.method in _GRE_SUFFIX:
            fieldmap_info = _gre_fieldmap(grouping, estimation)
        elif estimation.method is EstimationMethod.ANAT_CONTRAST:
            # T2Wreg is expressed downstream as fieldmap-less with t2w_sdc=True.
            fieldmap_info = {'suffix': None}
        elif estimation.method is EstimationMethod.SYNB0:
            raise NotImplementedError(
                f"SyNb0 estimation '{b0field_id}' has no legacy workflow shape; the "
                'synthesis workflow is added with the SyNb0 integration.'
            )
        else:  # pragma: no cover - every method is handled above
            raise ValueError(f'Unhandled estimation method {estimation.method!r}')

    return {
        'dwi_series': dwi_series,
        'fieldmap_info': fieldmap_info,
        'dwi_series_pedir': pedir,
        'concatenated_bids_name': derive_output_name(member_dwi),
    }


def _pepolar_fieldmap(grouping, estimation, member_dwi) -> tuple[list, dict, str]:
    """Build the ``rpe_series`` or ``epi`` shape for a PEPOLAR estimation.

    Returns ``(dwi_series, fieldmap_info, dwi_series_pedir)``. Extra b=0
    sources - epi fieldmaps and any reverse-PE DWI series borrowed from
    another output - become the ``epi`` list, which the TOPUP inputs treat as
    additional b=0 candidates.
    """
    plus = [path for path in member_dwi if grouping.files[path].signature.pe_polarity == 1]
    minus = [path for path in member_dwi if grouping.files[path].signature.pe_polarity == -1]
    extra_b0 = sorted(set(estimation.sources) - set(member_dwi))

    if plus and minus:
        fieldmap_info = {'suffix': 'rpe_series', 'rpe_series': minus}
        if extra_b0:
            fieldmap_info['epi'] = extra_b0
        return plus, fieldmap_info, grouping.files[plus[0]].signature.pe_dir

    # A single polarity in the output: the reverse blip lives in ``epi``.
    pedir = grouping.files[member_dwi[0]].signature.pe_dir or ''
    return member_dwi, {'suffix': 'epi', 'epi': extra_b0}, pedir


def _gre_fieldmap(grouping, estimation) -> dict:
    """Reconstruct a GRE (phasediff/two-phase/direct) fieldmap_info from sources."""
    fieldmap_info = {'suffix': _GRE_SUFFIX[estimation.method]}
    for path in estimation.sources:
        suffix = grouping.files[path].suffix
        fieldmap_info[suffix] = path
    primary = fieldmap_info[fieldmap_info['suffix']]
    fieldmap_info['metadata'] = dict(grouping.files[primary].metadata)
    return fieldmap_info
