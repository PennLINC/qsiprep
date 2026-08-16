"""BIDS-native grouping of DWI scans for preprocessing.

This package decides, per subject, from BIDS metadata alone:

1. Which DWI files share susceptibility distortions (distortion groups).
2. Which files jointly estimate each fieldmap (``B0FieldIdentifier``).
3. Which fieldmap corrects which DWI file (``B0FieldSource``).
4. Which files are concatenated in the outputs (``MultipartID``).

Curated sidecar metadata is used verbatim; everything the user did not curate
is inferred and tagged with its provenance. The output,
:class:`~.models.DWIGrouping`, describes the *data* - how any given
processing backend arranges that data is the business of adapters and the
previews in :mod:`~.report`.

Run ``python -m qsiprep.grouping /path/to/bids`` to print the grouping and
per-backend processing previews for a dataset.

Adding a new estimation method
------------------------------
The (method x backend) behavior matrix is deliberately spelled out as prose
branches rather than a rules table, because the preview text is the product.
The cost is that a new :class:`~.models.EstimationMethod` touches five places
(SYNB0 is the worked example in each):

1. :class:`~.models.EstimationMethod` - add the member.
2. ``inference.py`` - produce it: ``_classify_method`` for curated sources
   and/or a step in ``resolve_fieldmapless``; rank it in ``_METHOD_RANK``.
3. ``validation.check_backend`` - which backends can execute it, and whether
   failing is an error or a degradation.
4. ``report.py`` - ``_METHOD_LABELS``, ``MethodGroups``/``_ids_by_kind``,
   and a narration branch in each ``_describe_*`` function.
5. Tests - a scenario skeleton, inference assertions, and regenerated golden
   reports (``QSIPREP_REGEN_GROUPING_REPORTS=1``).
6. ``adapters.py`` - the legacy ``fieldmap_info`` shape the method maps to
   (or a clear ``NotImplementedError`` when it has none yet).
"""

from .adapters import (
    PreprocUnit,
    backend_for_config,
    to_legacy_scan_groups,
    to_preproc_units,
    unit_to_sidecar,
)
from .inference import build_grouping
from .metadata import index_subject
from .models import (
    ConcatenationGroup,
    DistortionGroup,
    DistortionSignature,
    DWIGrouping,
    EstimationMethod,
    FieldmapEstimation,
    FileRecord,
    Provenance,
)
from .report import describe_processing, full_report, report_text
from .validation import BACKENDS, GroupingError, GroupingIssue, check_backend, raise_for_errors

__all__ = [
    'BACKENDS',
    'ConcatenationGroup',
    'DistortionGroup',
    'DistortionSignature',
    'DWIGrouping',
    'EstimationMethod',
    'FieldmapEstimation',
    'FileRecord',
    'GroupingError',
    'GroupingIssue',
    'PreprocUnit',
    'Provenance',
    'backend_for_config',
    'build_dwi_grouping',
    'check_backend',
    'describe_processing',
    'full_report',
    'index_subject',
    'report_text',
    'to_legacy_scan_groups',
    'to_preproc_units',
    'unit_to_sidecar',
]


def build_dwi_grouping(
    layout,
    subject_data,
    separate_all_dwis=False,
    ignore_fieldmaps=False,
    ignore_shims=False,
    ignore_fov=False,
    force_t2wreg=False,
    use_synb0=False,
    strict=True,
):
    """Group one subject's DWI scans.

    Parameters
    ----------
    layout : :class:`bids.BIDSLayout`
        Layout of the input dataset; used only for sidecar metadata reads.
    subject_data : dict
        As returned by :func:`qsiprep.utils.bids.collect_data`: must contain
        a ``'dwi'`` key listing the subject's DWI files (``'fmap'``,
        ``'t1w'``, and ``'t2w'`` are optional - they are discovered from the
        layout when absent).
    separate_all_dwis : bool
        Every DWI series becomes its own output. Fieldmap estimation still
        happens at session scope, so single series keep their SDC.
    ignore_fieldmaps : bool
        Do not index ``fmap/``. The reverse phase-encoding DWI heuristic
        still applies.
    ignore_shims : bool
        Treat all ShimSetting values as compatible. Use when data were
        re-shimmed but distortion correction across shims is wanted anyway.
    ignore_fov : bool
        Downgrade the differing-orientation field-of-view error to a warning
        and proceed, accepting misapplied distortion corrections. Grid-size
        mismatches remain errors: they cannot be stacked at all.
    force_t2wreg : bool
        Correct every DWI series by registering its b=0 to the subject's T2w
        (TORTOISE T2Wreg), overriding any fieldmaps. Errors if no T2w exists.
    use_synb0 : bool
        Give series that have no fieldmap a SyNb0 synthetic-b=0 estimation
        synthesized from the T1w. Never overrides a real fieldmap. Errors if
        no T1w exists or a target series lacks PhaseEncodingDirection.
    strict : bool
        Raise :class:`~.validation.GroupingError` if any error-severity issue
        is found. With ``strict=False`` the grouping is returned with its
        ``issues`` intact, which is what reports and previews want.

    Returns
    -------
    :class:`~.models.DWIGrouping`
    """
    from bids.layout import parse_file_entities

    records, index_issues = index_subject(layout, subject_data, ignore_fieldmaps=ignore_fieldmaps)
    subject_id = parse_file_entities(records[0].path)['subject']
    grouping = build_grouping(
        records,
        subject_id=subject_id,
        separate_all_dwis=separate_all_dwis,
        ignore_shims=ignore_shims,
        ignore_fov=ignore_fov,
        force_t2wreg=force_t2wreg,
        use_synb0=use_synb0,
        extra_issues=index_issues,
    )
    if strict:
        raise_for_errors(grouping)
    return grouping
