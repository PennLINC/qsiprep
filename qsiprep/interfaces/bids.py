# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Interfaces for handling BIDS-like neuroimaging structures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fetch some example data:

    >>> import os
    >>> from niworkflows import data
    >>> data_root = data.get_bids_examples(variant='BIDS-examples-1-enh-ds054')
    >>> os.chdir(data_root)

Disable warnings:

    >>> from nipype import logging
    >>> logging.getLogger('nipype.interface').setLevel('ERROR')

"""

import gzip
import os
import re
from json import dump, loads
from shutil import copyfileobj

from bids.layout import Config
from nipype import logging
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    InputMultiObject,
    OutputMultiPath,
    SimpleInterface,
    Str,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.utils.filemanip import copyfile, fname_presuffix
from niworkflows.interfaces.bids import DerivativesDataSink as BaseDerivativesDataSink
from niworkflows.interfaces.bids import _DerivativesDataSinkInputSpec

from qsiprep.data import load as load_data

LOGGER = logging.getLogger('nipype.interface')
BIDS_NAME = re.compile(
    r'^(.*\/)?(?P<subject_id>sub-[a-zA-Z0-9]+)(_(?P<session_id>ses-[a-zA-Z0-9]+))?'
    '(_(?P<task_id>task-[a-zA-Z0-9]+))?(_(?P<acq_id>acq-[a-zA-Z0-9]+))?'
    '(_(?P<space_id>space-[a-zA-Z0-9]+))?'
    '(_(?P<rec_id>rec-[a-zA-Z0-9]+))?(_(?P<run_id>run-[a-zA-Z0-9]+))?'
)

# NOTE: Modified for QSIPrep's purposes
qsiprep_spec = loads(load_data('io_spec.json').read_text())
bids_config = Config.load('bids')
deriv_config = Config.load('derivatives')

qsiprep_entities = {v['name']: v['pattern'] for v in qsiprep_spec['entities']}
merged_entities = {**bids_config.entities, **deriv_config.entities}
merged_entities = {k: v.pattern for k, v in merged_entities.items()}
merged_entities = {**merged_entities, **qsiprep_entities}
merged_entities = [{'name': k, 'pattern': v} for k, v in merged_entities.items()]
config_entities = frozenset({e['name'] for e in merged_entities})


def get_bids_params(fullpath):
    bids_patterns = [
        r'^(.*/)?(?P<subject_id>sub-[a-zA-Z0-9]+)',
        '^.*_(?P<session_id>ses-[a-zA-Z0-9]+)',
        '^.*_(?P<task_id>task-[a-zA-Z0-9]+)',
        '^.*_(?P<acq_id>acq-[a-zA-Z0-9]+)',
        '^.*_(?P<space_id>space-[a-zA-Z0-9]+)',
        '^.*_(?P<rec_id>rec-[a-zA-Z0-9]+)',
        '^.*_(?P<run_id>run-[a-zA-Z0-9]+)',
        '^.*_(?P<dir_id>dir-[a-zA-Z0-9]+)',
    ]
    matches = {
        'subject_id': None,
        'session_id': None,
        'task_id': None,
        'dir_id': None,
        'acq_id': None,
        'space_id': None,
        'rec_id': None,
        'run_id': None,
    }
    for pattern in bids_patterns:
        pat = re.compile(pattern)
        match = pat.search(fullpath)
        params = match.groupdict() if match is not None else {}
        matches.update(params)
    return matches


class FileNotFoundError(IOError):
    pass


class BIDSInfoInputSpec(BaseInterfaceInputSpec):
    in_file = File(mandatory=True, desc='input file, part of a BIDS tree')


class BIDSInfoOutputSpec(TraitedSpec):
    subject_id = traits.Str()
    session_id = traits.Str()
    task_id = traits.Str()
    acq_id = traits.Str()
    rec_id = traits.Str()
    run_id = traits.Str()


class BIDSInfo(SimpleInterface):
    """
    Extract metadata from a BIDS-conforming filename

    This interface uses only the basename, not the path, to determine the
    subject, session, task, run, acquisition or reconstruction.

    """

    input_spec = BIDSInfoInputSpec
    output_spec = BIDSInfoOutputSpec

    def _run_interface(self, runtime):
        match = BIDS_NAME.search(self.inputs.in_file)
        params = match.groupdict() if match is not None else {}
        self._results = {key: val for key, val in list(params.items()) if val is not None}
        return runtime


class BIDSDataGrabberInputSpec(BaseInterfaceInputSpec):
    subject_data = traits.Dict(Str, traits.Any)
    subject_id = Str()


class BIDSDataGrabberOutputSpec(TraitedSpec):
    out_dict = traits.Dict(desc='output data structure')
    fmap = OutputMultiPath(desc='output fieldmaps')
    t1w = OutputMultiPath(desc='output T1w images')
    roi = OutputMultiPath(desc='output ROI images')
    t2w = OutputMultiPath(desc='output T2w images')
    dwi = OutputMultiPath(desc='output DWI images')


class BIDSDataGrabber(SimpleInterface):
    """
    Collect files from a BIDS directory structure

    >>> from qsiprep.interfaces import BIDSDataGrabber
    >>> from qsiprep.utils.bids import collect_data
    >>> bids_src = BIDSDataGrabber(anat_only=False)
    >>> bids_src.inputs.subject_data = collect_data('ds114', '01')[0]
    >>> bids_src.inputs.subject_id = 'ds114'
    >>> res = bids_src.run()
    >>> res.outputs.t1w  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ['.../ds114/sub-01/ses-retest/anat/sub-01_ses-retest_T1w.nii.gz',
     '.../ds114/sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz']

    """

    input_spec = BIDSDataGrabberInputSpec
    output_spec = BIDSDataGrabberOutputSpec
    _require_funcs = True

    def __init__(self, *args, **kwargs):
        anat_only = kwargs.pop('anat_only')
        dwi_only = kwargs.pop('dwi_only')
        anatomical_contrast = kwargs.pop('anatomical_contrast')
        self._anatomical_contrast = anatomical_contrast
        super().__init__(*args, **kwargs)
        if anat_only is not None:
            self._require_funcs = not anat_only
        self._no_anat_necessary = bool(dwi_only) or anatomical_contrast == 'none'

    def _run_interface(self, runtime):
        bids_dict = self.inputs.subject_data

        self._results['out_dict'] = bids_dict
        self._results.update(bids_dict)

        if not bids_dict['t1w']:
            message = f'No T1w images found for subject sub-{self.inputs.subject_id}'
            if self._no_anat_necessary:
                LOGGER.info('%s, but no problem because --dwi-only was selected.', message)
            elif self._anatomical_contrast != 'T1w':
                LOGGER.info(
                    '%s, but no problem because --anat-modality %s was selected.',
                    message,
                    self._anatomical_contrast,
                )
            else:
                raise FileNotFoundError(message)

        if not bids_dict['t2w']:
            message = f'No T2w images found for subject sub-{self.inputs.subject_id}'
            if self._no_anat_necessary:
                LOGGER.info('%s, but no problem because --dwi-only was selected.', message)
            elif self._anatomical_contrast != 'T2w':
                LOGGER.info(
                    '%s, but no problem because --anat-modality %s was selected.',
                    message,
                    self._anatomical_contrast,
                )
            else:
                raise FileNotFoundError(message)

        if self._no_anat_necessary and not bids_dict['dwi']:
            raise FileNotFoundError(
                f'No DWI images found for subject sub-{self.inputs.subject_id}'
            )

        for imtype in ['fmap', 'roi', 'dwi']:
            if not bids_dict[imtype]:
                LOGGER.warning("No '%s' images found for sub-%s", imtype, self.inputs.subject_id)

        return runtime


class DerivativesDataSink(BaseDerivativesDataSink):
    """Store derivative files.

    A child class of the niworkflows DerivativesDataSink, using xcp_d's configuration files.
    """

    out_path_base = ''
    _allowed_entities = set(config_entities)
    _config_entities = config_entities
    _config_entities_dict = merged_entities
    _file_patterns = qsiprep_spec['default_path_patterns']


class _DerivativesMaybeDataSinkInputSpec(_DerivativesDataSinkInputSpec):
    in_file = traits.Either(
        traits.Directory(exists=True),
        InputMultiObject(File(exists=True)),
        mandatory=False,
        desc='the object to be saved',
    )


class DerivativesMaybeDataSink(DerivativesDataSink):
    input_spec = _DerivativesMaybeDataSinkInputSpec

    def _run_interface(self, runtime):
        if not isdefined(self.inputs.in_file):
            return runtime
        return super()._run_interface(runtime)


class _DerivativesSidecarInputSpec(BaseInterfaceInputSpec):
    sidecar_data = traits.Dict()
    source_file = File()


class _DerivativesSidecarOutputSpec(TraitedSpec):
    derivatives_json = File(exists=True, desc='Derivatives Sidecar')


class DerivativesSidecar(SimpleInterface):
    input_spec = _DerivativesSidecarInputSpec
    output_spec = _DerivativesSidecarOutputSpec

    def _run_interface(self, runtime):
        json_fname = fname_presuffix(
            self.inputs.source_file, use_ext=False, suffix='.json', newpath=runtime.cwd
        )
        with open(json_fname, 'w') as jsonf:
            dump(self.inputs.sidecar_data, jsonf, sort_keys=True, indent=4)
        self._results['derivatives_json'] = json_fname
        return runtime


def _splitext(fname):
    fname, ext = os.path.splitext(os.path.basename(fname))
    if ext == '.gz':
        fname, ext2 = os.path.splitext(fname)
        ext = ext2 + ext
    return fname, ext


def _copy_any(src, dst):
    src_isgz = src.endswith('.gz')
    dst_isgz = dst.endswith('.gz')
    if src_isgz == dst_isgz:
        copyfile(src, dst, copy=True, use_hardlink=True)
        return False  # Make sure we do not reuse the hardlink later

    # Unlink target (should not exist)
    if os.path.exists(dst):
        os.unlink(dst)

    src_open = gzip.open if src_isgz else open
    dst_open = gzip.open if dst_isgz else open
    with src_open(src, 'rb') as f_in:
        with dst_open(dst, 'wb') as f_out:
            copyfileobj(f_in, f_out)
    return True
