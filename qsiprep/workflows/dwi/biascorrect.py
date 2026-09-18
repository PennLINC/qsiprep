"""Deciding whether to N4-correct DWI series."""

from ... import config


def dwi_biascorrect_enabled(dwi_files=None):
    """Should N4 bias correction run on these DWI images?

    ``--dwi-biascorrect`` governs DWIs only; ``--anat-biascorrect`` governs the
    anatomicals and never reaches this path.

    ``auto`` inspects the BIDS ``ImageType`` metadata for ``NORM``, which is how
    Siemens (among others) flags that intensity normalization was already applied
    on the console. Console normalization does not remove the need for N4;
    ``auto`` and ``none`` are for deliberately skipping it anyway.

    N4 is skipped only when EVERY input image is marked normalized. A mixed set is
    concatenated into one output and so must be corrected consistently, and an
    image whose metadata is missing is treated as un-normalized -- the conservative
    direction, since running N4 unnecessarily is milder than skipping it when it
    was needed.

    Parameters
    ----------
    dwi_files : list of str or None
        Every DWI feeding one final output. Under ``--distortion-group-merge`` that
        is the union over all of the output's constituent correction units, not one
        unit's files: the constituents are concatenated, so they must share a single
        decision.
    """
    mode = config.workflow.dwi_biascorrect or 'n4'
    if mode == 'n4':
        return True
    if mode == 'none':
        return False

    if not dwi_files:
        config.loggers.workflow.warning(
            '--dwi-biascorrect auto: no DWI files to inspect; running N4.'
        )
        return True
    layout = config.execution.layout
    if layout is None:
        config.loggers.workflow.warning(
            '--dwi-biascorrect auto: no BIDS layout available; running N4.'
        )
        return True

    normalized = []
    for img in dwi_files:
        try:
            image_type = layout.get_metadata(img).get('ImageType') or []
        except (OSError, ValueError, KeyError):
            image_type = []
        normalized.append(any(str(t).upper() == 'NORM' for t in image_type))

    if all(normalized):
        config.loggers.workflow.info(
            '--dwi-biascorrect auto: all %d DWI image(s) are marked NORM in '
            'ImageType; skipping N4.',
            len(normalized),
        )
        return False
    if any(normalized):
        config.loggers.workflow.warning(
            '--dwi-biascorrect auto: %d of %d DWI images are marked NORM. '
            'Running N4 on all of them, since a concatenated set cannot be '
            'corrected consistently otherwise.',
            sum(normalized),
            len(normalized),
        )
    return True
