# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Miscellaneous utility functions."""

import logging
import math
import re

import numpy as np

LOGGER = logging.getLogger('nipype.interface')

_DWIDENOISE_ENUM_PARAMETERS = {
    'aggregator': ('exclusive', 'gaussian', 'invl0', 'rank', 'uniform'),
    'datatype': ('float32', 'float64'),
    'debias_anchor': ('sample', 'group_mean'),
    'decomposition': ('bdcsvd', 'selfadjoint'),
    'demean': ('none', 'volume_groups', 'shells', 'all'),
    'demodulate': ('none', 'linear', 'hann', 'apc'),
    'estimator': ('exp1', 'exp2', 'med', 'mrm2023', 'tbme2022'),
    'filter_method': ('optshrink', 'optthresh', 'truncate'),
    'vst_method': ('none', 'linear', 'foi', 'koay', 'mom'),
}
_DWIDENOISE_STRING_PARAMETERS = {
    'demod_axes',
    'eigenspectra',
    'lamplus',
    'max_dist',
    'noise_image',
    'patchcount',
    'preconditioned_input',
    'preconditioned_output',
    'rank_input',
    'rank_output',
    'rank_pcanonzero',
    'schedule',
    'sum_aggregation',
    'sum_optshrink',
    'variance_removed',
    'voxelcount',
    'grad_file',
    'bvec_file',
    'bval_file',
}
_DWIDENOISE_PARAMETERS = (
    set(_DWIDENOISE_ENUM_PARAMETERS)
    | _DWIDENOISE_STRING_PARAMETERS
    | {
        'fixed_rank',
        'noise_dof',
        'noise_in',
        'preserve_noise_bias',
        'residual_statistics',
    }
)


def parse_denoise_method(spec, use_phase=None):
    """Parse a denoising method and semicolon-delimited parameters.

    Parameters for dwidenoise2 use ``name:value`` syntax, for example
    ``dwidenoise2;demodulate:apc;decomposition:bdcsvd``.

    Parameters
    ----------
    spec : str
        The ``--denoise-method`` specification.
    use_phase : bool or None
        Whether phase data are available for the series being denoised. ``None`` means
        that is not known yet, as when the CLI validates the specification before any
        scan has been selected, and skips the checks that depend on it.
    """
    elements = spec.split(';')
    method = elements[0].strip()
    if method not in ('dwidenoise', 'dwidenoise2', 'patch2self', 'none'):
        raise ValueError(f'Unknown denoising method: {method!r}')
    if len(elements) > 1 and method != 'dwidenoise2':
        raise ValueError(f'{method!r} does not accept DWIDenoise2 parameters')

    parameters = {}
    for element in elements[1:]:
        name, separator, value = element.partition(':')
        name = name.strip()
        value = value.strip()
        if not separator or not name or not value:
            raise ValueError(f'Invalid DWIDenoise2 parameter: {element!r}')
        if name not in _DWIDENOISE_PARAMETERS:
            raise ValueError(f'Unknown DWIDenoise2 parameter: {name!r}')
        if name in parameters:
            raise ValueError(f'Duplicate DWIDenoise2 parameter: {name!r}')

        if name in _DWIDENOISE_ENUM_PARAMETERS:
            choices = _DWIDENOISE_ENUM_PARAMETERS[name]
            if value not in choices:
                raise ValueError(f'Invalid value for {name!r}: {value!r}; choose from {choices}')
            parsed_value = value
        elif name == 'preserve_noise_bias':
            bool_values = {'true': True, 'false': False, '1': True, '0': False}
            try:
                parsed_value = bool_values[value.lower()]
            except KeyError as exc:
                raise ValueError(f'Invalid boolean value for {name!r}: {value!r}') from exc
        elif name in ('fixed_rank', 'noise_dof'):
            parsed_value = int(value)
        elif name == 'noise_in':
            try:
                parsed_value = float(value)
            except ValueError:
                parsed_value = value
        elif name == 'residual_statistics':
            parsed_value = tuple(item.strip() for item in value.split(','))
            if len(parsed_value) != 3 or not all(parsed_value):
                raise ValueError(f'{name!r} must contain three file names')
        else:
            parsed_value = value

        parameters[name] = parsed_value

    if method == 'dwidenoise2' and use_phase is False:
        demodulation = parameters.get('demodulate', 'none')
        if demodulation != 'none':
            raise ValueError(
                f'dwidenoise2 cannot apply {demodulation!r} phase demodulation to '
                'magnitude-only data. Provide phase data or use "demodulate:none".'
            )

    return method, parameters


_DWIDENOISE2_CONFIG_KEYS = frozenset(_DWIDENOISE_ENUM_PARAMETERS) | {
    'demod_axes',
    'fixed_rank',
    'noise_dof',
    'noise_in',
    'preserve_noise_bias',
    'schedule',
}
"""Keys a ``--dwidenoise2-config`` file may set."""

# Schedule columns in the order they are written, as defined in dwidenoise2's
# cpp/core/denoise/schedule.cpp. update_noise has no fixed default: dwidenoise2 resolves an
# omitted value to true on every row but the last, and false on the last.
_SCHEDULE_COLUMNS = (
    'spatial_subsample',
    'kernel',
    'smooth_noise',
    'update_noise',
    'temporal_subsample',
    'partitions',
    'max_partition_size',
)
_SCHEDULE_DEFAULTS = {
    'spatial_subsample': 2,
    'kernel': 'aspect=2.0',
    'smooth_noise': False,
    'temporal_subsample': 1.0,
    'partitions': 1,
    'max_partition_size': 'none',
}
# An unsigned decimal or exponent-notation number. float() alone would also accept a sign,
# "nan" and "inf".
_UNSIGNED_FLOAT = re.compile(r'(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?')
# ASCII digits only: Python's \d and str.isdigit() also match digits that dwidenoise2 cannot parse
_POSITIVE_INT = re.compile(r'[0-9]*[1-9][0-9]*')


class _DuplicateKeyError(ValueError):
    """A JSON object repeats a key."""


def _reject_duplicate_keys(pairs):
    """Build a dict from JSON object pairs, raising on a repeated key."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(f'has a duplicate key {key!r}')
        result[key] = value
    return result


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value):
    if not isinstance(value, int | float) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        # A JSON integer too large to convert to a float
        return False


def _positive_float(text):
    """Parse a positive number from a kernel parameter, or return None."""
    if not _UNSIGNED_FLOAT.fullmatch(text):
        return None
    value = float(text)
    return value if value > 0 and math.isfinite(value) else None


def _kernel_type(value):
    """Return the kernel type named by a schedule ``kernel`` value, or None if it is invalid.

    The grammar follows ``parse_kernel`` in dwidenoise2's cpp/core/denoise/schedule.cpp.
    """
    if not isinstance(value, str):
        return None
    key, separator, param = value.partition('=')
    if key in ('rank', 'rank_fixed'):
        return key if not separator else None
    if key == 'cuboid':
        if not separator:
            return key
        if param.endswith('x'):
            return key if _positive_float(param[:-1]) else None
        extents = param.split(',')
        if len(extents) in (1, 3) and all(_POSITIVE_INT.fullmatch(e) for e in extents):
            return key
        return None
    if key in ('aspect', 'aspect_ratio', 'radius', 'voxels', 'rmse') and separator:
        number = _positive_float(param)
        if number is None or (key == 'rmse' and number >= 1):
            return None
        return 'aspect' if key == 'aspect_ratio' else key
    return None


def _parse_schedule_cell(column, value, where):
    """Check one schedule cell and return the value to store."""
    if column == 'spatial_subsample':
        if _is_int(value) and value >= 1:
            return value
        if (
            isinstance(value, list)
            and len(value) == 3
            and all(_is_int(v) and v >= 1 for v in value)
        ):
            return tuple(value)
        raise ValueError(
            f'{where}: "spatial_subsample" must be a positive integer or a list of three '
            f'positive integers (got {value!r}).'
        )
    if column == 'kernel':
        if _kernel_type(value) is None:
            raise ValueError(
                f'{where}: invalid "kernel" {value!r}; valid kernels are "aspect=<ratio>", '
                '"rmse=<tolerance below 1>", "rank", "radius=<mm>", "voxels=<count>", '
                '"cuboid", "cuboid=<n>", "cuboid=<x>,<y>,<z>", "cuboid=<ratio>x" and '
                '"rank_fixed".'
            )
        return value
    if column in ('smooth_noise', 'update_noise'):
        if not isinstance(value, bool):
            raise ValueError(f'{where}: "{column}" must be true or false (got {value!r}).')
        return value
    if column == 'temporal_subsample':
        if not _is_number(value) or not 0 < value <= 1:
            raise ValueError(
                f'{where}: "temporal_subsample" must be a number in (0, 1] (got {value!r}).'
            )
        return value
    if column == 'partitions':
        if not _is_int(value) or value < 1:
            raise ValueError(f'{where}: "partitions" must be a positive integer (got {value!r}).')
        return value
    # max_partition_size
    if value == 'none' or (_is_int(value) and value >= 1):
        return value
    raise ValueError(
        f'{where}: "max_partition_size" must be a positive integer or "none" (got {value!r}).'
    )


def _resolved_update_noise(schedule):
    """Return each row's update_noise as dwidenoise2 resolves it."""
    last = len(schedule) - 1
    return [row.get('update_noise', i != last) for i, row in enumerate(schedule)]


def _load_dwidenoise2_schedule(rows, source):
    """Check the rows of a ``schedule`` key and return them with triplets as tuples."""
    if not isinstance(rows, list) or not rows:
        raise ValueError(f'{source}: "schedule" must be a non-empty list of rows.')

    schedule = []
    for number, row in enumerate(rows, start=1):
        where = f'{source}: schedule row {number}'
        if not isinstance(row, dict):
            raise ValueError(f'{where} must be a JSON object (got {row!r}).')
        unknown = sorted(set(row) - set(_SCHEDULE_COLUMNS))
        if unknown:
            raise ValueError(
                f'{where} has unknown column(s) {", ".join(unknown)}; '
                f'valid columns are {", ".join(_SCHEDULE_COLUMNS)}.'
            )
        parsed = {
            column: _parse_schedule_cell(column, value, where) for column, value in row.items()
        }
        # Rule 1
        if parsed.get('partitions', 1) > 1 and parsed.get('max_partition_size', 'none') != 'none':
            raise ValueError(f'{where} sets both "partitions" and "max_partition_size".')
        schedule.append(parsed)

    last = len(schedule) - 1
    update_noise = _resolved_update_noise(schedule)
    # Rule 3
    if _kernel_type(schedule[0].get('kernel', 'aspect=2.0')) in ('rmse', 'rank'):
        raise ValueError(
            f'{source}: the first schedule row may not use the "rmse" or "rank" kernel, which '
            'need a signal-rank density from an earlier row.'
        )
    for i, row in enumerate(schedule):
        where = f'{source}: schedule row {i + 1}'
        # Rule 2
        if row.get('smooth_noise', False) and not update_noise[i]:
            raise ValueError(f'{where} sets "smooth_noise" true but "update_noise" false.')
        # Rule 4
        if i != last and not update_noise[i]:
            raise ValueError(
                f'{where} sets "update_noise" false; only the last row may skip estimating the '
                'noise level.'
            )
        # Rule 5
        if i == last and row.get('smooth_noise', False):
            raise ValueError(
                f'{where} is the last (reconstruction) row and may not set "smooth_noise" true.'
            )
    # Rule 7
    if schedule[-1].get('temporal_subsample', 1.0) < 1:
        raise ValueError(
            f'{source}: schedule row {last + 1} is the last (reconstruction) row and must use '
            'all volumes ("temporal_subsample" 1).'
        )
    return schedule


def _check_dwidenoise2_combinations(params, source):
    """Apply the rules dwidenoise2 checks after it resolves its schedule (rules 8-11)."""
    schedule = params.get('schedule')
    fixed_rank = 'fixed_rank' in params
    vst_none = params.get('vst_method') == 'none'
    exclusive = params.get('aggregator') == 'exclusive'

    if fixed_rank and 'noise_in' in params:
        raise ValueError(f'{source} sets both "fixed_rank" and "noise_in".')
    # Rule 10
    if vst_none and 'noise_in' in params:
        raise ValueError(
            f'{source} sets "noise_in" with "vst_method" "none"; the noise level only '
            'parameterizes the variance-stabilizing transform.'
        )

    if schedule is None:
        # dwidenoise2 then uses its fixedrank schedule, a single pass sized for the
        # aggregator, or its default schedule, whose last row subsamples by 2.
        if exclusive and not fixed_rank and not vst_none:
            raise ValueError(
                f'{source} sets "aggregator" "exclusive" without a schedule. The default '
                'schedule subsamples its last row by 2; provide a schedule whose last row has '
                '"spatial_subsample" 1.'
            )
        return

    kernels = [_kernel_type(row.get('kernel', 'aspect=2.0')) for row in schedule]
    # Rule 8
    if not any(_resolved_update_noise(schedule)) and 'noise_in' not in params:
        raise ValueError(
            f'{source}: no schedule row estimates the noise level and "noise_in" is not set. '
            'A single row needs "update_noise" true.'
        )
    # Rule 9
    if fixed_rank and (len(schedule) > 1 or kernels[0] != 'rank_fixed'):
        raise ValueError(
            f'{source} sets "fixed_rank", which needs a single schedule row using the '
            '"rank_fixed" kernel.'
        )
    if not fixed_rank and 'rank_fixed' in kernels:
        row_number = kernels.index('rank_fixed') + 1
        raise ValueError(
            f'{source}: schedule row {row_number} sets "kernel" "rank_fixed", which needs '
            '"fixed_rank".'
        )
    # Rule 10
    if vst_none and len(schedule) > 1:
        raise ValueError(f'{source} sets "vst_method" "none", which allows only one schedule row.')
    # Rule 11
    if exclusive:
        subsample = schedule[-1].get('spatial_subsample', _SCHEDULE_DEFAULTS['spatial_subsample'])
        factors = subsample if isinstance(subsample, tuple) else (subsample,) * 3
        if max(factors) > 1:
            raise ValueError(
                f'{source} sets "aggregator" "exclusive", which needs the last schedule row to '
                'have "spatial_subsample" 1.'
            )


def load_dwidenoise2_config(path):
    """Load and check a ``--dwidenoise2-config`` JSON file.

    Parameters
    ----------
    path : str or os.PathLike
        The configuration file.

    Returns
    -------
    dict
        DWIDenoise2 input values. ``schedule`` is present only when the file sets it, as a
        list of row dicts with ``spatial_subsample`` triplets as tuples. ``demod_axes`` is
        joined into the comma-separated string the interface takes.

    Raises
    ------
    ValueError
        If the file does not exist, cannot be read, is not a JSON object, repeats a key, has
        an unknown key or an invalid value, or breaks one of the schedule rules that
        dwidenoise2 enforces.
    """
    import json
    import os

    source = f'dwidenoise2 configuration file {path}'
    if not os.path.exists(path):
        raise ValueError(f'{source} does not exist.')
    try:
        with open(path, encoding='utf-8') as f:
            cfg = json.load(f, object_pairs_hook=_reject_duplicate_keys)
    except _DuplicateKeyError as err:
        raise ValueError(f'{source} {err}.') from err
    except OSError as err:
        raise ValueError(f'{source} could not be read: {err}') from err
    except (json.JSONDecodeError, UnicodeDecodeError) as err:
        raise ValueError(f'{source} is not valid JSON: {err}') from err
    if not isinstance(cfg, dict):
        raise ValueError(f'{source} must contain a JSON object.')

    unknown = sorted(set(cfg) - _DWIDENOISE2_CONFIG_KEYS)
    if unknown:
        raise ValueError(
            f'{source} has unknown key(s) {", ".join(unknown)}; '
            f'valid keys are {", ".join(sorted(_DWIDENOISE2_CONFIG_KEYS))}.'
        )

    params = {}
    for name, value in cfg.items():
        if name == 'schedule':
            continue
        if name in _DWIDENOISE_ENUM_PARAMETERS:
            choices = _DWIDENOISE_ENUM_PARAMETERS[name]
            if not isinstance(value, str) or value not in choices:
                raise ValueError(
                    f'{source} sets {name}={value!r}; must be one of {", ".join(choices)}.'
                )
        elif name == 'preserve_noise_bias':
            if not isinstance(value, bool):
                raise ValueError(f'{source} sets {name}={value!r}; must be true or false.')
        elif name in ('fixed_rank', 'noise_dof'):
            if not _is_int(value) or value < 1:
                raise ValueError(f'{source} sets {name}={value!r}; must be an integer >= 1.')
        elif name == 'noise_in':
            if not _is_number(value) or value < 0:
                raise ValueError(
                    f'{source} sets {name}={value!r}; must be a number >= 0. Noise-map files '
                    'are not supported.'
                )
        elif name == 'demod_axes':
            if (
                not isinstance(value, list)
                or not value
                or not all(_is_int(axis) and axis >= 0 for axis in value)
            ):
                raise ValueError(
                    f'{source} sets {name}={value!r}; must be a non-empty list of '
                    'non-negative integers.'
                )
            value = ','.join(str(axis) for axis in value)
        params[name] = value

    if 'schedule' in cfg:
        params['schedule'] = _load_dwidenoise2_schedule(cfg['schedule'], source)
    _check_dwidenoise2_combinations(params, source)
    return params


def _format_schedule_value(value):
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, tuple | list):
        return ','.join(str(v) for v in value)
    return str(value)


def format_dwidenoise2_schedule(rows):
    """Write schedule rows in the table format that ``dwidenoise2 -schedule`` reads.

    The header lists every column any row sets, plus ``update_noise``, which is always
    written so that the header is never empty. A cell a row omits gets dwidenoise2's
    default. For ``update_noise`` that is the value dwidenoise2 resolves: true on every row
    but the last, and false on the last.

    Parameters
    ----------
    rows : list of dict
        Schedule rows, as returned in the ``schedule`` key of
        :func:`load_dwidenoise2_config`.

    Returns
    -------
    str
        The schedule file text.
    """
    used = {column for row in rows for column in row} | {'update_noise'}
    columns = [column for column in _SCHEDULE_COLUMNS if column in used]
    last = len(rows) - 1

    lines = [
        '# dwidenoise2 noise estimation schedule written by QSIPrep from --dwidenoise2-config',
        ' '.join(columns),
    ]
    for i, row in enumerate(rows):
        cells = []
        for column in columns:
            if column in row:
                value = row[column]
            elif column == 'update_noise':
                value = i != last
            else:
                value = _SCHEDULE_DEFAULTS[column]
            cells.append(_format_schedule_value(value))
        lines.append(' '.join(cells))
    return '\n'.join(lines) + '\n'


# dwidenoise2's own defaults, mirrored here so the boilerplate describes what actually ran
_DWIDENOISE2_DEFAULTS = {
    'aggregator': 'gaussian',
    'decomposition': 'bdcsvd',
    'demodulate': 'apc',
    'demean': 'shells',
    'estimator': 'mrm2023',
}

_DWIDENOISE2_ESTIMATORS = {
    'exp1': 'the Marchenko-Pastur threshold search of the original `dwidenoise` [@dwidenoise1]',
    'exp2': 'a refined Marchenko-Pastur threshold search [@cordero2019complex]',
    'med': 'the median eigenvalue [@gavish2014]',
    'mrm2023': 'a Marchenko-Pastur fit generalized to multi-dimensional data [@olesen2023]',
    'tbme2022': 'a multiple-moment generalized quarter-circle estimator [@zhu2022]',
}

_DWIDENOISE2_FILTERS = {
    'optshrink': (
        'optimal shrinkage of the singular values, which minimizes the Frobenius norm '
        '[@cordero2019complex]'
    ),
    'optthresh': 'an optimal hard threshold on the singular values [@gavish2014]',
    'truncate': 'hard truncation, as in the original `dwidenoise` [@dwidenoise1]',
}

_DWIDENOISE2_DEMODULATION = {
    'apc': (
        'noise-adaptive phase correction, which re-estimates the background phase at every '
        'noise level iteration [@pizzolato2020]'
    ),
    'hann': 'a fixed nonlinear phase estimate from a Hann-windowed k-space filter [@patron2024]',
    'linear': 'a strictly linear phase term regressed from each k-space [@cordero2019complex]',
}

_DWIDENOISE2_DEMEAN = {
    'shells': 'the mean signal of each *b*-value shell was regressed out',
    'volume_groups': 'the mean signal of each volume group was regressed out',
    'all': 'the mean signal across all volumes was regressed out',
}


def _join_clauses(clauses):
    """Join clauses into a comma-separated list with a trailing 'and'."""
    if len(clauses) == 1:
        return clauses[0]

    return f'{", ".join(clauses[:-1])} and {clauses[-1]}'


def describe_dwidenoise2(parameters, complex_data):
    """Describe a ``dwidenoise2`` call for the methods boilerplate.

    ``dwidenoise2`` applies a number of methods beyond the original ``dwidenoise``, most of
    them on by default, and each carries its own citation. Describing only the parameters
    QSIPrep passed explicitly would therefore both understate what ran and omit references
    the authors ask for, so unset options are described using the defaults of the shipped
    build. The conditions attached to each citation follow the reference list that
    ``dwidenoise2`` prints in its own help.

    Parameters
    ----------
    parameters : dict
        DWIDenoise2 parameters, as returned by :func:`parse_denoise_method`.
    complex_data : bool
        Whether ``dwidenoise2`` is run on complex-valued data. Phase demodulation only
        applies to complex data, and only magnitude data need a nonlinear
        variance-stabilizing transform.

    Returns
    -------
    str
        Boilerplate text with inline ``[@citation]`` keys, beginning with 'denoised using'
        so that the caller can supply its own subject.
    """
    used = {**_DWIDENOISE2_DEFAULTS, **parameters}
    # The kernel size and the number of PCAs are set per iteration by the schedule rather
    # than by a fixed window
    schedule = parameters.get('schedule')
    if schedule is None:
        schedule_desc = 'its default schedule'
    else:
        schedule_desc = (
            f'a custom {len(schedule)}-iteration schedule provided in the QSIPrep '
            'configuration file'
        )

    sentences = [
        'denoised using the Marchenko-Pastur PCA method [@dwidenoise1; @dwidenoise2] as '
        'implemented in `dwidenoise2` [@dwidenoise2software; @cordero2019complex], which '
        'estimates the noise level over a multi-resolution series of iterations following '
        f'{schedule_desc}, sizing the sliding-window patch for noise estimation and for '
        'denoising separately.'
    ]

    preconditioning = []
    if complex_data and used['demodulate'] != 'none':
        demodulation = _DWIDENOISE2_DEMODULATION[used['demodulate']]
        preconditioning.append(
            f'the complex-valued data were phase-demodulated using {demodulation}'
        )
    if used['demean'] != 'none':
        preconditioning.append(_DWIDENOISE2_DEMEAN[used['demean']])

    # Complex data are Gaussian, so they always take the linear transform; magnitude data
    # get a nonlinear one to account for the non-central chi noise distribution
    vst_method = used.get('vst_method', 'linear' if complex_data else 'foi')
    if not complex_data and vst_method in ('foi', 'koay', 'mom'):
        vst = (
            'a nonlinear variance-stabilizing transform was applied to render the '
            'non-central chi distributed magnitude data approximately Gaussian and '
            'homoscedastic [@foi2011; @ma2020]'
        )
        if vst_method == 'koay':
            vst += ', inverted with an analytically exact correction scheme [@koay2006]'
        if 'noise_dof' in used:
            vst += f', assuming {used["noise_dof"]} receive channels'
        preconditioning.append(vst)
    elif vst_method == 'linear':
        preconditioning.append('the data were scaled by the local noise level')

    if preconditioning:
        sentences.append(f'Prior to PCA, {_join_clauses(preconditioning)}.')

    decomposition = (
        'a bidirectional divide-and-conquer SVD'
        if used['decomposition'] == 'bdcsvd'
        else 'a self-adjoint eigendecomposition'
    )
    if 'noise_in' in used:
        estimation = 'the noise level was taken from a pre-estimated noise map'
    elif 'fixed_rank' in used:
        estimation = f'the signal rank was fixed at {used["fixed_rank"]}'
    else:
        estimation = (
            'the noise level was estimated from the eigenspectrum using '
            f'{_DWIDENOISE2_ESTIMATORS[used["estimator"]]}'
        )
    sentences.append(f'Each patch was decomposed with {decomposition}, and {estimation}.')

    # dwidenoise2 truncates rather than shrinks when the rank is given rather than estimated
    default_filter = 'truncate' if 'fixed_rank' in used else 'optshrink'
    filter_method = used.get('filter_method', default_filter)
    reconstruction = (
        f'Component contributions were filtered by {_DWIDENOISE2_FILTERS[filter_method]}'
    )
    if used['aggregator'] == 'exclusive':
        reconstruction += (
            ', and each voxel was reconstructed solely from the patch centered on it.'
        )
    elif used['aggregator'] == 'gaussian':
        reconstruction += (
            ', and each voxel was reconstructed from every overlapping patch, weighted by a '
            'Gaussian function of its distance to each patch center [@manjon2013].'
        )
    else:
        reconstruction += (
            ', and each voxel was reconstructed from every overlapping patch, combined with '
            f'{used["aggregator"]} weighting [@manjon2013].'
        )
    sentences.append(reconstruction)

    if not complex_data and not used.get('preserve_noise_bias', False):
        sentences.append(
            'The inverse transform was evaluated at the exact-unbiased operating point, '
            'removing the noise-floor bias from the denoised magnitude data.'
        )

    return ' '.join(sentences) + ' '


def safe_unit_vector(vector):
    """Return the unit vector of ``vector``.

    A zero-magnitude b-vector (e.g. the magnitude-zero b-vectors Philips uses
    for b=0 volumes) cannot be normalized: dividing by a zero norm yields NaN.
    In that case ``(1, 0, 0)`` is substituted and a warning is emitted so it is
    clear the b-vector has been modified.
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        LOGGER.warning('Encountered a zero-magnitude b-vector; substituting (1, 0, 0).')
        return np.array([1.0, 0.0, 0.0])
    return vector / norm


def check_deps(workflow):
    from nipype.utils.filemanip import which

    return sorted(
        (node.interface.__class__.__name__, node.interface._cmd)
        for node in workflow._get_all_nodes()
        if (hasattr(node.interface, '_cmd') and which(node.interface._cmd.split()[0]) is None)
    )


def fix_multi_source_name(in_files, include_session, anatomical_contrast='T1w'):
    """Make up a generic source name when there are multiple source files.

    An ``anatomical_contrast`` of ``'none'`` means the anatomical reference is derived
    from the DWIs themselves, so the name is built from the DWI files instead.

    >>> fix_multi_source_name(
    ...     ['/path/to/sub-045_ses-test_T1w.nii.gz', '/path/to/sub-045_ses-retest_T1w.nii.gz'],
    ...     False,
    ...     'T1w',
    ... )
    '/path/to/sub-045_T1w.nii.gz'

    >>> fix_multi_source_name(
    ...     ['/path/to/dwi/sub-045_ses-test_dwi.nii.gz'],
    ...     False,
    ...     'none',
    ... )
    '/path/to/anat/sub-045_dwi.nii.gz'
    """
    import os

    from nipype.utils.filemanip import filename_to_list

    base, in_file = os.path.split(filename_to_list(in_files)[0])

    # Remove the session label
    base = os.path.abspath(base)
    folders = base.split(os.sep)
    if not include_session:
        folders = [f for f in folders if not f.startswith('ses-')]
    base = os.sep.join(folders)

    subject_label = in_file.split('_', 1)[0].split('-')[1]
    if anatomical_contrast == 'none':
        anatomical_contrast = 'dwi'
        base = base.replace('/dwi', '/anat')

    _session = ''
    if include_session:
        ses_entity = [f for f in folders if f.startswith('ses-')]
        if ses_entity:
            _session = f'_{ses_entity[-1]}'

    return os.path.join(base, f'sub-{subject_label}{_session}_{anatomical_contrast}.nii.gz')


def validate_eddy_config(eddy_config):
    """Validate the eddy configuration file.

    Parameters
    ----------
    eddy_config : str
        The path to the eddy configuration JSON file.

    Raises
    ------
    ValueError
        If the eddy configuration file is not valid.
    """
    import json
    import os

    if not os.path.exists(eddy_config):
        raise ValueError(f'Eddy configuration file {eddy_config} does not exist.')
    with open(eddy_config) as f:
        eddy_config = json.load(f)

    if 'cnr_maps' not in eddy_config:
        raise ValueError('Eddy configuration file must contain "cnr_maps" key.')
    if eddy_config['cnr_maps'] is not True:
        raise ValueError('Eddy configuration file must contain "cnr_maps" key with value True.')

    return


def validate_diffprep_config(diffprep_config):
    """Validate the DIFFPREP configuration file.

    Parameters
    ----------
    diffprep_config : str
        The path to the DIFFPREP configuration JSON file.

    Raises
    ------
    ValueError
        If the DIFFPREP configuration file does not exist, is not valid JSON,
        or sets an unsupported ``correction_mode``.
    """
    import json
    import os

    if not os.path.exists(diffprep_config):
        raise ValueError(f'DIFFPREP configuration file {diffprep_config} does not exist.')
    with open(diffprep_config) as f:
        cfg = json.load(f)

    # Checked here so a typo fails at parse time rather than as a KeyError
    # while the DIFFPREP workflow is being built.
    valid_modes = ('motion', 'quadratic', 'cubic')
    correction_mode = cfg.get('correction_mode', 'quadratic')
    if correction_mode not in valid_modes:
        raise ValueError(
            f'DIFFPREP configuration file {diffprep_config} sets '
            f'correction_mode={correction_mode!r}; must be one of '
            f'{", ".join(valid_modes)}.'
        )

    return


SHORELINE_MODELS = ('3dshore', 'tensor', 'none')
"""Signal models SHORELine can use to predict motion-correction targets."""

SHORELINE_TRANSFORMS = ('Affine', 'Rigid')

#: ``--dwi2anat-dof`` is user-facing; antsRegistration wants a transform name.
#: 9 is not offered: antsRegistration has no 9-DOF transform (Rigid=6,
#: Similarity=7, Affine=12), unlike the FLIRT/mri_coreg path fMRIPrep uses.
DWI2ANAT_DOF_TO_TRANSFORM = {6: 'Rigid', 12: 'Affine'}
"""Transformations SHORELine can optimize during head motion correction."""


def load_shoreline_config(path=None, model=None):
    """Load a ``--shoreline-config`` JSON file, merged over the shipped defaults.

    Parameters
    ----------
    path : str, os.PathLike or None
        The SHORELine configuration JSON file. ``None`` uses the defaults in
        ``qsiprep/data/shoreline_params.json``.
    model : str or None
        A SHORELine model from the deprecated ``--hmc-model`` alias (``3dshore``,
        ``tensor`` or ``none``). It replaces the default model, and conflicts with
        a ``model`` key in the file.

    Returns
    -------
    dict
        Exactly the keys ``model``, ``iters`` and ``transform``.

    Raises
    ------
    ValueError
        If the file does not exist, is not a JSON object, has an unknown key or an
        invalid value, or sets ``model`` while ``model`` is also given.
    """
    import json
    import os

    from ..data import load as load_data

    cfg = json.loads(load_data('shoreline_params.json').read_text())
    source = 'The default SHORELine configuration'
    if path is not None:
        source = f'SHORELine configuration file {path}'
        if not os.path.exists(path):
            raise ValueError(f'{source} does not exist.')
        try:
            with open(path, encoding='utf-8') as f:
                user_cfg = json.load(f)
        except OSError as err:
            raise ValueError(f'{source} could not be read: {err}') from err
        except (json.JSONDecodeError, UnicodeDecodeError) as err:
            raise ValueError(f'{source} is not valid JSON: {err}') from err
        if not isinstance(user_cfg, dict):
            raise ValueError(f'{source} must contain a JSON object.')
        # Unknown keys are errors so a typo cannot silently fall back to a default.
        unknown = sorted(set(user_cfg) - set(cfg))
        if unknown:
            raise ValueError(
                f'{source} has unknown key(s) {", ".join(unknown)}; '
                f'valid keys are {", ".join(sorted(cfg))}.'
            )
        if model is not None and 'model' in user_cfg:
            raise ValueError(f'--hmc-model conflicts with "model" in --shoreline-config {path}.')
        cfg.update(user_cfg)
    if model is not None:
        cfg['model'] = model

    if cfg['model'] not in SHORELINE_MODELS:
        raise ValueError(
            f'{source} sets model={cfg["model"]!r}; must be one of {", ".join(SHORELINE_MODELS)}.'
        )
    if cfg['transform'] not in SHORELINE_TRANSFORMS:
        raise ValueError(
            f'{source} sets transform={cfg["transform"]!r}; must be one of '
            f'{", ".join(SHORELINE_TRANSFORMS)}.'
        )
    iters = cfg['iters']
    # bool is a subclass of int, so "iters": true has to be rejected explicitly.
    if (
        isinstance(iters, bool)
        or not isinstance(iters, int)
        or (iters < 1 and cfg['model'] != 'none')
    ):
        raise ValueError(
            f'{source} sets iters={iters!r}; must be an integer >= 1 '
            '(any integer when model is "none").'
        )
    return cfg


def dwi_biascorrect_enabled(dwi_files=None):
    """Should N4 bias correction run on these DWI images?

    ``--dwi-biascorrect`` governs DWIs only; ``--anat-biascorrect`` governs the
    anatomicals and never reaches this path.

    ``auto`` inspects the BIDS ``ImageType`` metadata for ``NORM``, which is how
    Siemens (among others) flags that intensity normalization was already applied
    on the console. Console normalization does not necessarily remove the need
    for N4, which is why ``n4`` stays the default; ``auto`` and ``none`` are for
    deliberately skipping it.

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
    from .. import config

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


def validate_gradient_flags(gradient_file, force, ignore):
    """Validate the ``--gradient-file``/``--force``/``--ignore`` combination.

    Parameters
    ----------
    gradient_file : str, os.PathLike or None
        Path passed to ``--gradient-file``, or ``None`` if the flag was not given.
        Existence is assumed to already be checked (the CLI's ``IsFile`` argparse
        type does that); only the extension is validated here.
    force : list of str
        Values passed to ``--force``. The gradwarp-related ones are
        ``"gradwarp1D"`` and ``"gradwarp3D"``.
    ignore : list of str
        Values passed to ``--ignore``.

    Raises
    ------
    ValueError
        If both ``--force gradwarp1D`` and ``--force gradwarp3D`` are given, if a
        ``--force gradwarp{1,3}D`` is combined with ``--ignore gradwarp``, if a
        ``--force gradwarp{1,3}D`` is given without ``--gradient-file``, or if
        ``--gradient-file`` does not end in a TORTOISE-recognized extension.

    Notes
    -----
    ``--force gradwarp1D`` and ``--force gradwarp3D`` are mutually exclusive, but
    ``--force`` takes a list of unrelated values, so argparse cannot express that
    with a mutually exclusive group. It is checked here instead.

    An unrecognized ``--gradient-file`` extension is rejected outright rather than
    merely warned about. TORTOISE itself only warns on an unrecognized extension
    and then silently disables gradient nonlinearity correction; silently
    producing uncorrected output is the wrong default for a batch pipeline, so
    QSIPrep rejects it here instead.
    """
    from .. import config

    # argparse's choices constrain these to "gradwarp1D" and "gradwarp3D".
    # Deduplicated: "--force gradwarp1D gradwarp1D" names one dimensionality.
    forced_gradwarp = sorted({value for value in force if value.startswith('gradwarp')})
    ignoring_gradwarp = 'gradwarp' in ignore

    if len(forced_gradwarp) > 1:
        raise ValueError(
            f'"--force {forced_gradwarp[0]}" and "--force {forced_gradwarp[1]}" are '
            'mutually exclusive: a run is corrected in one dimension or in three, '
            'not both.'
        )

    if forced_gradwarp and ignoring_gradwarp:
        raise ValueError(
            f'"--force {forced_gradwarp[0]}" and "--ignore gradwarp" are contradictory.'
        )

    if forced_gradwarp and not gradient_file:
        raise ValueError(f'"--force {forced_gradwarp[0]}" requires --gradient-file.')

    if gradient_file:
        gradient_extensions = ('.grad', '.dat', '.gc', '.nii', '.nii.gz')
        if not str(gradient_file).endswith(gradient_extensions):
            raise ValueError(
                f'--gradient-file must end in one of {", ".join(gradient_extensions)}: '
                f'<{gradient_file}>. TORTOISE silently disables gradient nonlinearity '
                'correction for unrecognized extensions, so QSIPrep rejects it here instead.'
            )
        if ignoring_gradwarp:
            config.loggers.cli.warning(
                '--gradient-file is unused because "--ignore gradwarp" was given.'
            )

    return


if __name__ == '__main__':
    pass
