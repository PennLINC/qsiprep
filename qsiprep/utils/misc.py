# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Miscellaneous utility functions."""

import logging

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


# dwidenoise2's own defaults, mirrored here so the boilerplate describes what actually ran
_DWIDENOISE2_DEFAULTS = {
    'aggregator': 'gaussian',
    'decomposition': 'bdcsvd',
    'demodulate': 'apc',
    'demean': 'shells',
    'estimator': 'mrm2023',
    'schedule': 'default',
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
    schedule = used['schedule']
    schedule_desc = (
        'its default schedule' if schedule == 'default' else f'the {schedule!r} schedule'
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


def fix_multi_T1w_source_name(in_files):
    """Make up a generic source name when there are multiple T1s.

    >>> fix_multi_T1w_source_name([
    ...     '/path/to/sub-045_ses-test_T1w.nii.gz',
    ...     '/path/to/sub-045_ses-retest_T1w.nii.gz'])
    '/path/to/sub-045_T1w.nii.gz'
    """
    import os

    from nipype.utils.filemanip import filename_to_list

    base, in_file = os.path.split(filename_to_list(in_files)[0])
    subject_label = in_file.split('_', 1)[0].split('-')[1]
    return os.path.join(base, f'sub-{subject_label}_T1w.nii.gz')


def fix_multi_source_name(in_files, dwi_only, include_session, anatomical_contrast='T1w'):
    """Make up a generic source name when there are multiple source files.

    >>> fix_multi_source_name(
    ...     ['/path/to/sub-045_ses-test_T1w.nii.gz', '/path/to/sub-045_ses-retest_T1w.nii.gz'],
    ...     False,
    ...     False,
    ...     'T1w',
    ... )
    '/path/to/sub-045_T1w.nii.gz'
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
    if dwi_only:
        anatomical_contrast = 'dwi'
        base = base.replace('/dwi', '/anat')

    _session = ''
    if include_session:
        ses_entity = [f for f in folders if f.startswith('ses-')]
        if ses_entity:
            _session = f'_{ses_entity[-1]}'

    return os.path.join(base, f'sub-{subject_label}{_session}_{anatomical_contrast}.nii.gz')


def add_suffix(in_files, suffix):
    """Wrap nipype's fname_presuffix to conveniently just add a suffixfix.

    >>> add_suffix([
    ...     '/path/to/sub-045_ses-test_T1w.nii.gz',
    ...     '/path/to/sub-045_ses-retest_T1w.nii.gz'], '_test')
    'sub-045_ses-test_T1w_test.nii.gz'
    """
    import os.path as op

    from nipype.utils.filemanip import filename_to_list, fname_presuffix

    return op.basename(fname_presuffix(filename_to_list(in_files)[0], suffix=suffix))


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
        If the DIFFPREP configuration file does not exist or is not valid JSON.
    """
    import json
    import os

    if not os.path.exists(diffprep_config):
        raise ValueError(f'DIFFPREP configuration file {diffprep_config} does not exist.')
    with open(diffprep_config) as f:
        json.load(f)

    return


if __name__ == '__main__':
    pass
