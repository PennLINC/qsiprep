"""Adapt a :class:`~.models.DWIGrouping` into the shapes the workflows consume.

A :class:`PreprocUnit` is the tool-facing unit of preprocessing: the DWI series
a single HMC+SDC run corrects together, plus the fieldmap estimation (if any)
that corrects them. One unit is produced per *applied estimation* - the natural
unit a single TOPUP+eddy / DIFFPREP run handles - plus one per uncorrected
distortion group. Merging several distortion groups back into one concatenated
output happens after HMC and is deliberately left to a later change; here every
unit is its own output.

:func:`to_preproc_units` is the native entry point the workflow builders use.
:func:`to_legacy_scan_groups` renders the same units into the legacy
``dwi_series``/``fieldmap_info`` dicts, so unconverted workflow code keeps
working while the refactor proceeds.
"""

from __future__ import annotations

import dataclasses
import math
import os.path as op
from collections import defaultdict

from .models import (
    CorrectionMethod,
    DWIGrouping,
    FieldmapEstimation,
    FileRecord,
    derive_output_name,
)

#: Which sidecar suffixes name the GRE files of each estimation method.
_GRE_SUFFIX = {
    CorrectionMethod.PHASEDIFF: 'phasediff',
    CorrectionMethod.PHASES: 'phase1',
    CorrectionMethod.DIRECT: 'fieldmap',
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


@dataclasses.dataclass(frozen=True)
class PreprocUnit:
    """One backend preprocessing run over a set of DWI series.

    ``estimation`` is the fieldmap estimation that corrects these series, or
    ``None`` when they are uncorrected. All facts needed by the FSL/TORTOISE
    builders (phase-encoding split, extra b=0 sources, GRE files, metadata,
    shelled-ness) are derived from the grouping model here, so downstream code
    never re-reads sidecars or headers.
    """

    grouping: DWIGrouping = dataclasses.field(compare=False, repr=False)
    output_name: str
    dwi_files: tuple[str, ...]
    estimation: FieldmapEstimation | None

    @property
    def method(self) -> CorrectionMethod | None:
        return self.estimation.method if self.estimation else None

    @property
    def dwi_records(self) -> tuple[FileRecord, ...]:
        return tuple(self.grouping.files[path] for path in self.dwi_files)

    @property
    def is_pepolar(self) -> bool:
        return self.method is CorrectionMethod.PEPOLAR

    @property
    def is_gre(self) -> bool:
        """True for a precomputed-fieldmap estimation (phasediff/two-phase/direct)."""
        return self.method in _GRE_SUFFIX

    @property
    def using_t2w_for_sdc(self) -> bool:
        """True for fieldmap-less TORTOISE T2Wreg: register the b=0 to a T2w."""
        return self.method is CorrectionMethod.T2WREG

    @property
    def is_nipreps_syn(self) -> bool:
        """True for fieldmap-less classic ANTs SyN registration (SyN-SDC)."""
        return self.method is CorrectionMethod.NIPREPS_SYN

    @property
    def has_scanner_measured_fieldmap(self) -> bool:
        """True when a field the scanner measured (PEPOLAR or GRE) corrects these series.

        Distinguishes the real-fieldmap methods from the fieldmap-less ones
        (T2Wreg, SyNb0) and the uncorrected case.
        """
        return self.is_pepolar or self.is_gre

    @property
    def dwi_metadata(self) -> dict:
        """Sidecar metadata of the lead DWI series (already read into the model)."""
        return dict(self.dwi_records[0].metadata)

    @property
    def pe_axis(self) -> str:
        """The phase-encoding axis ('i'/'j'/'k') of the corrected series."""
        return self.pe_dir[0] if self.pe_dir else ''

    @property
    def plus_files(self) -> tuple[str, ...]:
        """Member DWI series acquired in the ``+`` phase-encoding polarity."""
        return tuple(
            path for path in self.dwi_files if self.grouping.files[path].signature.pe_polarity == 1
        )

    @property
    def minus_files(self) -> tuple[str, ...]:
        """Member DWI series acquired in the ``-`` phase-encoding polarity."""
        return tuple(
            path
            for path in self.dwi_files
            if self.grouping.files[path].signature.pe_polarity == -1
        )

    @property
    def extra_b0(self) -> tuple[str, ...]:
        """Estimation sources outside the output: epi fieldmaps and borrowed DWIs.

        These serve as additional b=0 candidates for TOPUP/DRBUDDI.
        """
        if self.estimation is None:
            return ()
        return tuple(sorted(set(self.estimation.sources) - set(self.dwi_files)))

    @property
    def has_bidirectional_dwi(self) -> bool:
        """True when both PE polarities are present among the member series."""
        return bool(self.plus_files) and bool(self.minus_files)

    @property
    def pe_dir(self) -> str:
        """The phase-encoding direction reported for the corrected series.

        For a bidirectional (rpe-series) unit this is the ``+`` polarity, to
        match the legacy convention that the plus series leads.
        """
        lead = self.plus_files[0] if self.has_bidirectional_dwi else self.dwi_files[0]
        return self.grouping.files[lead].signature.pe_dir or ''

    def gre_files(self) -> dict[str, str]:
        """GRE fieldmap file paths keyed by BIDS suffix (phasediff/magnitude*/...)."""
        return {self.grouping.files[path].suffix: path for path in self.estimation.sources}

    @property
    def gre_suffix(self) -> str:
        """The primary GRE file suffix ('phasediff'/'phase1'/'fieldmap')."""
        return _GRE_SUFFIX[self.method]

    def metadata_for(self, path: str) -> dict:
        """Sidecar metadata of a source file, already read into the model."""
        return dict(self.grouping.files[path].metadata)

    def sidecar_overrides(self) -> dict[str, dict]:
        """Per-file distortion metadata, so eddy/DRBUDDI skip re-reading sidecars.

        Keyed by file path; each value carries the ``PhaseEncodingDirection`` and
        ``TotalReadoutTime`` the model parsed (plus ``SliceTiming`` from the raw
        sidecar) for every DWI series and estimation source. Files with no known
        phase-encoding direction are omitted, so consumers fall back to disk.
        """
        paths = set(self.dwi_files)
        if self.estimation is not None:
            paths |= set(self.estimation.sources)

        overrides = {}
        for path in paths:
            record = self.grouping.files.get(path)
            if record is None or record.signature.pe_dir is None:
                continue
            overrides[path] = {
                'PhaseEncodingDirection': record.signature.pe_dir,
                'TotalReadoutTime': record.signature.readout_time,
                'SliceTiming': record.metadata.get('SliceTiming'),
            }
        return overrides

    def to_legacy_dict(self) -> dict:
        """Render this unit as a legacy ``group_dwi_scans`` scan-group dict."""
        return {
            'dwi_series': list(self._legacy_dwi_series()),
            'fieldmap_info': self._legacy_fieldmap_info(),
            'dwi_series_pedir': self.pe_dir,
            'concatenated_bids_name': self.output_name,
        }

    def _legacy_dwi_series(self) -> tuple[str, ...]:
        if self.is_pepolar and self.has_bidirectional_dwi:
            return self.plus_files
        return self.dwi_files

    def _legacy_fieldmap_info(self) -> dict:
        if self.estimation is None:
            return {'suffix': None}

        if self.is_pepolar:
            if self.has_bidirectional_dwi:
                info = {'suffix': 'rpe_series', 'rpe_series': list(self.minus_files)}
                if self.extra_b0:
                    info['epi'] = list(self.extra_b0)
                return info
            # A single polarity in the output: the reverse blip lives in ``epi``.
            return {'suffix': 'epi', 'epi': list(self.extra_b0)}

        if self.method in _GRE_SUFFIX:
            info = {'suffix': _GRE_SUFFIX[self.method]}
            info.update(self.gre_files())
            primary = info[info['suffix']]
            info['metadata'] = dict(self.grouping.files[primary].metadata)
            return info

        if self.method is CorrectionMethod.T2WREG:
            # T2Wreg is expressed downstream as fieldmap-less with t2w_sdc=True.
            return {'suffix': None}

        if self.method is CorrectionMethod.NIPREPS_SYN:
            return {'suffix': 'syn'}

        if self.method is CorrectionMethod.SYNB0:
            raise NotImplementedError(
                f"SyNb0 estimation '{self.estimation.b0field_id}' has no legacy workflow "
                'shape; the synthesis workflow is added with the SyNb0 integration.'
            )

        raise ValueError(f'Unhandled estimation method {self.method!r}')  # pragma: no cover


def to_preproc_units(grouping: DWIGrouping) -> list[PreprocUnit]:
    """Partition a grouping into the per-run :class:`PreprocUnit` list."""
    units = []
    for concat in sorted(grouping.concatenation_groups.values(), key=lambda c: c.output_name):
        by_estimation = defaultdict(list)
        for dgroup in grouping.distortion_groups_in(concat.multipart_id):
            by_estimation[dgroup.b0field_source].append(dgroup)

        for b0field_id, dgroups in sorted(by_estimation.items(), key=lambda item: item[0] or ''):
            if b0field_id is None:
                for dgroup in dgroups:
                    units.append(
                        PreprocUnit(
                            grouping=grouping,
                            output_name=derive_output_name(dgroup.dwi_files),
                            dwi_files=tuple(dgroup.dwi_files),
                            estimation=None,
                        )
                    )
            else:
                member_dwi = sorted({path for dgroup in dgroups for path in dgroup.dwi_files})
                units.append(
                    PreprocUnit(
                        grouping=grouping,
                        output_name=derive_output_name(member_dwi),
                        dwi_files=tuple(member_dwi),
                        estimation=grouping.estimations[b0field_id],
                    )
                )
    return units


def _metadata_values_agree(first, second) -> bool:
    if isinstance(first, float) and isinstance(second, float):
        return math.isclose(first, second)
    return first == second


def unit_to_sidecar(unit: PreprocUnit) -> dict:
    """Build the derivatives sidecar describing how an output was produced.

    Metadata comes from the grouping model's file records rather than being
    re-read from disk. Keys shared (and equal) across every member series are
    promoted to the top level; the per-series metadata is kept under
    ``SourceMetadata``.
    """
    scan_metadata: dict[str, dict] = {}
    common: dict | None = None
    for record in unit.dwi_records:
        name = op.basename(record.path)
        meta = dict(record.metadata)
        scan_metadata[name] = meta
        if common is None:
            common = dict(meta)
        else:
            common = {
                key: value
                for key, value in common.items()
                if key in meta and _metadata_values_agree(value, meta[key])
            }

    sidecar = dict(common or {})
    sidecar['ScanGrouping'] = {
        'output_name': unit.output_name,
        'method': unit.method.value if unit.method else None,
        'dwi_series': [op.basename(path) for path in unit.dwi_files],
        'fieldmap_sources': [op.basename(path) for path in unit.estimation.sources]
        if unit.estimation
        else [],
    }
    sidecar['SourceMetadata'] = scan_metadata
    sidecar['Sources'] = sorted(scan_metadata)
    return sidecar


def to_legacy_scan_groups(grouping: DWIGrouping) -> tuple[list[dict], dict]:
    """Render a grouping as ``(scan_groups, concatenation_scheme)``.

    ``scan_groups`` matches the contract of the retired
    :func:`qsiprep.utils.grouping.group_dwi_scans`; ``concatenation_scheme``
    maps every scan group's name to itself (post-HMC concatenation of
    distinct distortion groups is left to a later change).
    """
    scan_groups = [unit.to_legacy_dict() for unit in to_preproc_units(grouping)]
    names = [group['concatenated_bids_name'] for group in scan_groups]
    return scan_groups, {name: name for name in names}
