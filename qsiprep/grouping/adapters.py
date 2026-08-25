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
from collections import Counter, defaultdict

from .methods import selection_for_config
from .models import (
    CorrectionMethod,
    DWIGrouping,
    FieldmapEstimation,
    FileRecord,
    derive_output_name,
)
from .validation import blip_pair_polarities, blip_sort_key

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
    backend. Everything else - ``tortoise`` and the SHORELine models
    (``3dSHORE``, ``tensor``, ``none``) - shares TORTOISE's DRBUDDI
    feasibility semantics, so it maps to ``tortoise``.
    """
    return selection_for_config(hmc_model, pepolar_method).legacy_backend


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
    #: The execution-plan run this unit renders (set when the unit was derived
    #: from a compiled plan), carrying the ordered stage sequence.
    run: object | None = dataclasses.field(default=None, compare=False, repr=False)

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
    def is_single_blip_pair(self) -> bool:
        """True when the correction sources form exactly one matched blip pair.

        One axis, readout time and shim with both polarities - the shape the
        single-pass DRBUDDI stage consumes (it builds one blip-up and one
        blip-down image). Multi-axis or multi-readout PEPOLAR units are not, so
        the mixed path skips their DRBUDDI refinement and relies on TOPUP+eddy;
        TORTOISE never sees one, since the adapter has already split it per pair.
        """
        if self.estimation is None or not self.is_pepolar:
            return False
        pairs = blip_pair_polarities(self.grouping, self.estimation)
        return len(pairs) == 1 and all(len(polarities) == 2 for polarities in pairs.values())

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


def _blip_pair_key(grouping: DWIGrouping, path: str) -> tuple:
    """The blip-pair identity ``(pe_axis, readout_time, shim)`` of a source file."""
    sig = grouping.files[path].signature
    return (sig.pe_axis, sig.readout_time, sig.shim)


def _decomposes_on_tortoise(
    grouping: DWIGrouping, unit, estimation: FieldmapEstimation | None, backend: str
) -> bool:
    """True when TORTOISE must break a PEPOLAR unit into per-blip-group sub-units.

    DRBUDDI corrects one matched blip pair - same axis, readout time and shim,
    opposite polarity - at a time, so each of a unit's blip groups is routed on
    its own: a complete pair to DRBUDDI, an unpaired group to the fieldmap-less
    fallback (DIFFPREP T2Wreg with a T2w, else HMC-only). Decompose whenever the
    unit spans more than one blip group, or any of its groups is unpaired; a lone
    complete pair stays one DRBUDDI unit. Completeness is judged against the
    estimation, so a borrowed opposite blip still counts. FSL/mixed keep the
    pooled unit; backend knowledge lives here, never in the model.
    """
    if backend != 'tortoise' or estimation is None or not estimation.is_pepolar:
        return False
    pair_pols = blip_pair_polarities(grouping, estimation)
    unit_keys = {_blip_pair_key(grouping, path) for path in unit.dwi_files}
    unpaired = any(len(pair_pols.get(key, ())) < 2 for key in unit_keys)
    return len(unit_keys) > 1 or unpaired


def _estimation_on_pair(
    grouping: DWIGrouping, estimation: FieldmapEstimation, pair_key: tuple
) -> FieldmapEstimation:
    """A copy of ``estimation`` restricted to the sources in one blip pair.

    Keeps ``extra_b0``/borrowing matched: a per-pair DRBUDDI run must only see the
    b=0 sources (DWIs and epi fieldmaps) that share its axis, readout time and
    shim - not merely its axis.
    """
    sources = tuple(
        path
        for path in estimation.sources
        if path in grouping.files and _blip_pair_key(grouping, path) == pair_key
    )
    return dataclasses.replace(
        estimation,
        sources=sources,
        pe_axes=frozenset({pair_key[0]}),
        bidirectional_axes=frozenset({pair_key[0]}),
    )


def _decompose_unit(
    grouping: DWIGrouping, unit, estimation: FieldmapEstimation
) -> list[PreprocUnit]:
    """One PreprocUnit per blip group of a PEPOLAR unit.

    A complete pair keeps DRBUDDI (PEPOLAR, estimation restricted to that pair);
    an unpaired group becomes fieldmap-less (``estimation=None``) so DIFFPREP
    falls back to T2Wreg with a T2w, or leaves it uncorrected. Names come from
    each group's files, kept as-is where already distinct (e.g. ``acq-``/``run-``
    that differs by readout) and disambiguated by axis - then a per-axis index -
    only where the pooled names would collide (two groups differing only in
    ``dir-``).
    """
    pair_pols = blip_pair_polarities(grouping, estimation)
    by_pair: dict[tuple, list[str]] = defaultdict(list)
    for path in unit.dwi_files:
        by_pair[_blip_pair_key(grouping, path)].append(path)
    pair_keys = sorted(by_pair, key=blip_sort_key)
    files = {key: tuple(sorted(by_pair[key])) for key in pair_keys}
    base = {key: derive_output_name(list(files[key])) for key in pair_keys}
    base_counts = Counter(base.values())
    axis_index: dict[tuple, int] = defaultdict(int)

    subunits = []
    for key in pair_keys:
        if base_counts[base[key]] == 1:
            name = base[key]
        else:
            axis = key[0]
            same_axis = sum(1 for k in pair_keys if base[k] == base[key] and k[0] == axis)
            if same_axis > 1:
                axis_index[(base[key], axis)] += 1
                acq = f'{axis}{axis_index[(base[key], axis)]}'
            else:
                acq = axis
            name = derive_output_name(list(files[key]), acq=acq)
        complete = len(pair_pols.get(key, ())) == 2
        subunits.append(
            PreprocUnit(
                grouping=grouping,
                output_name=name,
                dwi_files=files[key],
                estimation=_estimation_on_pair(grouping, estimation, key) if complete else None,
            )
        )
    return subunits


def plan_preproc_units(grouping: DWIGrouping, plan) -> list[PreprocUnit]:
    """One :class:`PreprocUnit` per :class:`~.plan.ProcessingRun`, carrying it."""
    return [
        PreprocUnit(
            grouping=grouping,
            output_name=run.key,
            dwi_files=run.dwi_files,
            estimation=run.estimation,
            run=run,
        )
        for run in plan.runs
    ]


def plan_concatenation_scheme(plan) -> dict[str, str]:
    """Run key -> final output name, from a compiled plan's assemblies."""
    final_of = {assembly.output_group: assembly.output_name for assembly in plan.outputs}
    return {run.key: final_of.get(run.output_group, run.output_group) for run in plan.runs}


def _units_and_finals(grouping: DWIGrouping, backend: str):
    """Yield ``(PreprocUnit, final_output_name)`` for every unit, backend-aware.

    Shared by :func:`to_preproc_units` and :func:`concatenation_scheme` so the
    unit list and the concatenation scheme always agree on the (possibly split)
    unit names. Both are views over the compiled execution plan.
    """
    from .methods import canonical_selection
    from .plan import compile_plan

    plan = compile_plan(grouping, canonical_selection(backend))
    scheme = plan_concatenation_scheme(plan)
    for unit in plan_preproc_units(grouping, plan):
        yield unit, scheme[unit.output_name]


def to_preproc_units(grouping: DWIGrouping, backend: str = 'fsl') -> list[PreprocUnit]:
    """One :class:`PreprocUnit` per correction unit: each is one HMC+SDC run.

    For the ``tortoise`` backend a PEPOLAR unit is broken into one unit per blip
    group - complete pairs to DRBUDDI, unpaired groups to the fieldmap-less
    fallback (see :func:`_decomposes_on_tortoise`); every other unit is one
    PreprocUnit.
    """
    return [unit for unit, _final in _units_and_finals(grouping, backend)]


def concatenation_scheme(grouping: DWIGrouping, backend: str = 'fsl') -> dict[str, str]:
    """PreprocUnit output name -> final output name, from the model's packaging.

    Identity for outputs with a single unit; a final output spanning several
    units - including the per-axis sub-units of a TORTOISE split - maps each
    unit's preprocessed result to the shared final name, to be combined by the
    distortion-group-merge workflow.
    """
    return {unit.output_name: final for unit, final in _units_and_finals(grouping, backend)}


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


def to_legacy_scan_groups(grouping: DWIGrouping, backend: str = 'fsl') -> tuple[list[dict], dict]:
    """Render a grouping as ``(scan_groups, concatenation_scheme)``.

    ``scan_groups`` matches the contract of the retired
    :func:`qsiprep.utils.grouping.group_dwi_scans`; ``concatenation_scheme``
    maps each scan group (one per correction unit, or per axis under a TORTOISE
    split) to the final output its corrected result is combined into.
    """
    scan_groups = [unit.to_legacy_dict() for unit in to_preproc_units(grouping, backend)]
    return scan_groups, concatenation_scheme(grouping, backend)
