# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Interfaces for handling BIDS-like neuroimaging structures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


"""

import os
import os.path as op
from shutil import copytree, rmtree

from nipype import logging
from nipype.interfaces.base import (
    traits, isdefined, TraitedSpec, BaseInterfaceInputSpec,
    File, Directory, InputMultiPath, OutputMultiPath, Str,
    SimpleInterface
)
from ..utils.bids import BIDS_NAME, get_metadata_for_nifti
from ..utils.misc import splitext as _splitext, _copy_any

LOGGER = logging.getLogger('nipype.interface')


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
        self._results = {key: val for key, val in list(params.items())
                         if val is not None}
        return runtime


class BIDSDataGrabberInputSpec(BaseInterfaceInputSpec):
    subject_data = traits.Dict(Str, traits.Any)
    subject_id = Str()


class BIDSDataGrabberOutputSpec(TraitedSpec):
    out_dict = traits.Dict(desc='output data structure')
    fmap = OutputMultiPath(desc='output fieldmaps')
    bold = OutputMultiPath(desc='output functional images')
    sbref = OutputMultiPath(desc='output sbrefs')
    t1w = OutputMultiPath(desc='output T1w images')
    roi = OutputMultiPath(desc='output ROI images')
    t2w = OutputMultiPath(desc='output T2w images')
    flair = OutputMultiPath(desc='output FLAIR images')


class BIDSDataGrabber(SimpleInterface):
    """
    Collect files from a BIDS directory structure


    """
    input_spec = BIDSDataGrabberInputSpec
    output_spec = BIDSDataGrabberOutputSpec
    _require_funcs = True

    def __init__(self, *args, **kwargs):
        anat_only = kwargs.pop('anat_only')
        super(BIDSDataGrabber, self).__init__(*args, **kwargs)
        if anat_only is not None:
            self._require_funcs = not anat_only

    def _run_interface(self, runtime):
        bids_dict = self.inputs.subject_data

        self._results['out_dict'] = bids_dict
        self._results.update(bids_dict)

        if not bids_dict['t1w']:
            raise FileNotFoundError('No T1w images found for subject sub-{}'.format(
                self.inputs.subject_id))

        if self._require_funcs and not bids_dict['bold']:
            raise FileNotFoundError('No functional images found for subject sub-{}'.format(
                self.inputs.subject_id))

        for imtype in ['bold', 't2w', 'flair', 'fmap', 'sbref', 'roi']:
            if not bids_dict[imtype]:
                LOGGER.warning('No "%s" images found for sub-%s',
                               imtype, self.inputs.subject_id)

        return runtime


class DerivativesDataSinkInputSpec(BaseInterfaceInputSpec):
    base_directory = traits.Directory(
        desc='Path to the base directory for storing data.')
    in_file = InputMultiPath(File(exists=True), mandatory=True,
                             desc='the object to be saved')
    source_file = File(exists=False, mandatory=True, desc='the input func file')
    space = traits.Str('', usedefault=True, desc='Label for space field')
    desc = traits.Str('', usedefault=True, desc='Label for description field')
    suffix = traits.Str('', usedefault=True, desc='suffix appended to source_file')
    keep_dtype = traits.Bool(False, usedefault=True, desc='keep datatype suffix')
    extra_values = traits.List(traits.Str)
    compress = traits.Bool(desc="force compression (True) or uncompression (False)"
                                " of the output file (default: same as input)")


class DerivativesDataSinkOutputSpec(TraitedSpec):
    out_file = OutputMultiPath(File(exists=True, desc='written file path'))
    compression = OutputMultiPath(
        traits.Bool, desc='whether ``in_file`` was compressed/uncompressed '
                          'or `it was copied directly.')


class DerivativesDataSink(SimpleInterface):
    """
    Saves the `in_file` into a BIDS-Derivatives folder provided
    by `base_directory`, given the input reference `source_file`.


    """
    input_spec = DerivativesDataSinkInputSpec
    output_spec = DerivativesDataSinkOutputSpec
    out_path_base = "niworkflows"
    _always_run = True

    def __init__(self, out_path_base=None, **inputs):
        super(DerivativesDataSink, self).__init__(**inputs)
        self._results['out_file'] = []
        if out_path_base:
            self.out_path_base = out_path_base

    def _run_interface(self, runtime):
        src_fname, _ = _splitext(self.inputs.source_file)
        src_fname, dtype = src_fname.rsplit('_', 1)
        _, ext = _splitext(self.inputs.in_file[0])
        if self.inputs.compress is True and not ext.endswith('.gz'):
            ext += '.gz'
        elif self.inputs.compress is False and ext.endswith('.gz'):
            ext = ext[:-3]

        m = BIDS_NAME.search(src_fname)

        mod = op.basename(op.dirname(self.inputs.source_file))

        base_directory = runtime.cwd
        if isdefined(self.inputs.base_directory):
            base_directory = op.abspath(self.inputs.base_directory)

        out_path = '{}/{subject_id}'.format(self.out_path_base, **m.groupdict())
        if m.groupdict().get('session_id') is not None:
            out_path += '/{session_id}'.format(**m.groupdict())
        out_path += '/{}'.format(mod)

        out_path = op.join(base_directory, out_path)

        os.makedirs(out_path, exist_ok=True)

        base_fname = op.join(out_path, src_fname)

        formatstr = '{bname}{space}{desc}{suffix}{dtype}{ext}'
        if len(self.inputs.in_file) > 1 and not isdefined(self.inputs.extra_values):
            formatstr = '{bname}{space}{desc}{suffix}{i:04d}{dtype}{ext}'

        space = '_space-{}'.format(self.inputs.space) if self.inputs.space else ''
        desc = '_desc-{}'.format(self.inputs.desc) if self.inputs.desc else ''
        suffix = '_{}'.format(self.inputs.suffix) if self.inputs.suffix else ''
        dtype = '' if not self.inputs.keep_dtype else ('_%s' % dtype)

        self._results['compression'] = []
        for i, fname in enumerate(self.inputs.in_file):
            out_file = formatstr.format(
                bname=base_fname,
                space=space,
                desc=desc,
                suffix=suffix,
                i=i,
                dtype=dtype,
                ext=ext)
            if isdefined(self.inputs.extra_values):
                out_file = out_file.format(extra_value=self.inputs.extra_values[i])
            self._results['out_file'].append(out_file)
            self._results['compression'].append(_copy_any(fname, out_file))
        return runtime


class ReadSidecarJSONInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='the input nifti file')
    fields = traits.List(traits.Str, desc='get only certain fields')


class ReadSidecarJSONOutputSpec(TraitedSpec):
    subject_id = traits.Str()
    session_id = traits.Str()
    task_id = traits.Str()
    acq_id = traits.Str()
    rec_id = traits.Str()
    run_id = traits.Str()
    out_dict = traits.Dict()


class ReadSidecarJSON(SimpleInterface):
    """
    A utility to find and read JSON sidecar files of a BIDS tree
    """
    expr = BIDS_NAME
    input_spec = ReadSidecarJSONInputSpec
    output_spec = ReadSidecarJSONOutputSpec
    _always_run = True

    def _run_interface(self, runtime):
        metadata = get_metadata_for_nifti(self.inputs.in_file)
        output_keys = [key for key in list(self.output_spec().get().keys()) if key.endswith('_id')]
        outputs = self.expr.search(op.basename(self.inputs.in_file)).groupdict()

        for key in output_keys:
            id_value = outputs.get(key)
            if id_value is not None:
                self._results[key] = outputs.get(key)

        if isdefined(self.inputs.fields) and self.inputs.fields:
            for fname in self.inputs.fields:
                self._results[fname] = metadata[fname]
        else:
            self._results['out_dict'] = metadata

        return runtime


class BIDSFreeSurferDirInputSpec(BaseInterfaceInputSpec):
    derivatives = Directory(exists=True, mandatory=True,
                            desc='BIDS derivatives directory')
    freesurfer_home = Directory(exists=True, mandatory=True,
                                desc='FreeSurfer installation directory')
    subjects_dir = traits.Str('freesurfer', usedefault=True,
                              desc='Name of FreeSurfer subjects directory')
    spaces = traits.List(traits.Str, desc='Set of output spaces to prepare')
    overwrite_fsaverage = traits.Bool(False, usedefault=True,
                                      desc='Overwrite fsaverage directories, if present')


class BIDSFreeSurferDirOutputSpec(TraitedSpec):
    subjects_dir = traits.Directory(exists=True,
                                    desc='FreeSurfer subjects directory')


class BIDSFreeSurferDir(SimpleInterface):
    """Create a FreeSurfer subjects directory in a BIDS derivatives directory
    and copy fsaverage from the local FreeSurfer distribution.

    Output subjects_dir = ``{derivatives}/{subjects_dir}``, and may be passed to
    ReconAll and other FreeSurfer interfaces.
    """
    input_spec = BIDSFreeSurferDirInputSpec
    output_spec = BIDSFreeSurferDirOutputSpec
    _always_run = True

    def _run_interface(self, runtime):
        subjects_dir = os.path.join(self.inputs.derivatives,
                                    self.inputs.subjects_dir)
        os.makedirs(subjects_dir, exist_ok=True)
        self._results['subjects_dir'] = subjects_dir

        spaces = list(self.inputs.spaces)
        # Always copy fsaverage, for proper recon-all functionality
        if 'fsaverage' not in spaces:
            spaces.append('fsaverage')

        for space in spaces:
            # Skip non-freesurfer spaces and fsnative
            if not space.startswith('fsaverage'):
                continue
            source = os.path.join(self.inputs.freesurfer_home, 'subjects', space)
            dest = os.path.join(subjects_dir, space)
            # Finesse is overrated. Either leave it alone or completely clobber it.
            if os.path.exists(dest) and self.inputs.overwrite_fsaverage:
                rmtree(dest)
            if not os.path.exists(dest):
                try:
                    copytree(source, dest)
                except FileExistsError:
                    LOGGER.warning("%s exists; if multiple jobs are running in parallel"
                                   ", this can be safely ignored", dest)

        return runtime
