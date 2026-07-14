# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Miscellaneous utility functions."""

import logging

import numpy as np

LOGGER = logging.getLogger('nipype.interface')

_DWIDENOISE_ENUM_PARAMETERS = {
    'datatype': ('float32', 'float64'),
    'decomposition': ('bdcsvd', 'selfadjoint'),
    'estimator': ('Exp1', 'Exp2', 'Med', 'MRM2023'),
    'shape': ('sphere', 'cuboid'),
    'demodulate': ('none', 'linear', 'nonlinear'),
    'demean': ('none', 'volume_groups', 'shells', 'all'),
    'filter_method': ('optshrink', 'optthresh', 'truncate'),
    'aggregator': ('exclusive', 'gaussian', 'invl0', 'rank', 'uniform'),
}
_DWIDENOISE_STRING_PARAMETERS = {
    'demod_axes',
    'vst',
    'preconditioned_input',
    'preconditioned_output',
    'noise_image',
    'lamplus',
    'rank_pcanonzero',
    'rank_input',
    'rank_output',
    'variance_removed',
    'eigenspectra',
    'max_dist',
    'voxelcount',
    'patchcount',
    'sum_aggregation',
    'sum_optshrink',
    'grad_file',
    'bvec_file',
    'bval_file',
}
_DWIDENOISE_PARAMETERS = (
    set(_DWIDENOISE_ENUM_PARAMETERS)
    | _DWIDENOISE_STRING_PARAMETERS
    | {
        'onepass',
        'noise_in',
        'fixed_rank',
        'radius',
        'aspect_ratio',
        'minvoxels',
        'extent',
        'subsample',
        'residual_statistics',
    }
)


def parse_denoise_method(spec):
    """Parse a denoising method and semicolon-delimited parameters.

    Parameters use ``name:value`` syntax, for example
    ``dwidenoise;demodulate:nonlinear;decomposition:bdcsvd``.
    """
    elements = spec.split(';')
    method = elements[0].strip()
    if method not in ('dwidenoise', 'patch2self', 'none'):
        raise ValueError(f'Unknown denoising method: {method!r}')
    if len(elements) > 1 and method != 'dwidenoise':
        raise ValueError(f'{method!r} does not accept DWIDenoise parameters')

    parameters = {}
    for element in elements[1:]:
        name, separator, value = element.partition(':')
        name = name.strip()
        value = value.strip()
        if not separator or not name or not value:
            raise ValueError(f'Invalid DWIDenoise parameter: {element!r}')
        if name not in _DWIDENOISE_PARAMETERS:
            raise ValueError(f'Unknown DWIDenoise parameter: {name!r}')
        if name in parameters:
            raise ValueError(f'Duplicate DWIDenoise parameter: {name!r}')

        if name in _DWIDENOISE_ENUM_PARAMETERS:
            choices = _DWIDENOISE_ENUM_PARAMETERS[name]
            if value not in choices:
                raise ValueError(f'Invalid value for {name!r}: {value!r}; choose from {choices}')
            parsed_value = value
        elif name == 'onepass':
            bool_values = {'true': True, 'false': False, '1': True, '0': False}
            try:
                parsed_value = bool_values[value.lower()]
            except KeyError as exc:
                raise ValueError(f'Invalid boolean value for {name!r}: {value!r}') from exc
        elif name in ('fixed_rank', 'minvoxels'):
            parsed_value = int(value)
        elif name in ('radius', 'aspect_ratio'):
            parsed_value = float(value)
        elif name == 'noise_in':
            try:
                parsed_value = float(value)
            except ValueError:
                parsed_value = value
        elif name in ('extent', 'subsample'):
            values = tuple(int(item.strip()) for item in value.split(','))
            if len(values) not in (1, 3):
                raise ValueError(f'{name!r} must contain one or three integers')
            parsed_value = values[0] if len(values) == 1 else values
        elif name == 'residual_statistics':
            parsed_value = tuple(item.strip() for item in value.split(','))
            if len(parsed_value) != 3 or not all(parsed_value):
                raise ValueError(f'{name!r} must contain three file names')
        else:
            parsed_value = value

        parameters[name] = parsed_value

    return method, parameters


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


if __name__ == '__main__':
    pass
