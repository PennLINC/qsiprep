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
"""

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
    'Provenance',
    'build_dwi_grouping',
    'check_backend',
    'describe_processing',
    'full_report',
    'index_subject',
    'report_text',
]


def build_dwi_grouping(
    layout,
    subject_data,
    separate_all_dwis=False,
    ignore_fieldmaps=False,
    ignore_shims=False,
    strict=True,
):
    """Group one subject's DWI scans.

    Parameters
    ----------
    layout : :class:`bids.BIDSLayout`
        Layout of the input dataset; used only for sidecar metadata reads.
    subject_data : dict
        As returned by :func:`qsiprep.utils.bids.collect_data`: must contain
        a ``'dwi'`` key listing the subject's DWI files (``'fmap'`` optional -
        it is discovered from the layout when absent).
    separate_all_dwis : bool
        Every DWI series becomes its own output. Fieldmap estimation still
        happens at session scope, so single series keep their SDC.
    ignore_fieldmaps : bool
        Do not index ``fmap/``. The reverse phase-encoding DWI heuristic
        still applies.
    ignore_shims : bool
        Treat all ShimSetting values as compatible. Use when data were
        re-shimmed but distortion correction across shims is wanted anyway.
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
        extra_issues=index_issues,
    )
    if strict:
        raise_for_errors(grouping)
    return grouping
