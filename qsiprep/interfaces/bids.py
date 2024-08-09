# -*- coding: utf-8 -*-
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
import os.path as op
import re
from shutil import copyfileobj, copytree

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
from nipype.utils.filemanip import copyfile

LOGGER = logging.getLogger("nipype.interface")
BIDS_NAME = re.compile(
    "^(.*\/)?(?P<subject_id>sub-[a-zA-Z0-9]+)(_(?P<session_id>ses-[a-zA-Z0-9]+))?"
    "(_(?P<task_id>task-[a-zA-Z0-9]+))?(_(?P<acq_id>acq-[a-zA-Z0-9]+))?"
    "(_(?P<space_id>space-[a-zA-Z0-9]+))?"
    "(_(?P<rec_id>rec-[a-zA-Z0-9]+))?(_(?P<run_id>run-[a-zA-Z0-9]+))?"
)


def get_bids_params(fullpath):
    bids_patterns = [
        r"^(.*/)?(?P<subject_id>sub-[a-zA-Z0-9]+)",
        "^.*_(?P<session_id>ses-[a-zA-Z0-9]+)",
        "^.*_(?P<task_id>task-[a-zA-Z0-9]+)",
        "^.*_(?P<acq_id>acq-[a-zA-Z0-9]+)",
        "^.*_(?P<space_id>space-[a-zA-Z0-9]+)",
        "^.*_(?P<rec_id>rec-[a-zA-Z0-9]+)",
        "^.*_(?P<run_id>run-[a-zA-Z0-9]+)",
        "^.*_(?P<dir_id>dir-[a-zA-Z0-9]+)",
    ]
    matches = {
        "subject_id": None,
        "session_id": None,
        "task_id": None,
        "dir_id": None,
        "acq_id": None,
        "space_id": None,
        "rec_id": None,
        "run_id": None,
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
    in_file = File(mandatory=True, desc="input file, part of a BIDS tree")


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
    out_dict = traits.Dict(desc="output data structure")
    fmap = OutputMultiPath(desc="output fieldmaps")
    bold = OutputMultiPath(desc="output functional images")
    sbref = OutputMultiPath(desc="output sbrefs")
    t1w = OutputMultiPath(desc="output T1w images")
    roi = OutputMultiPath(desc="output ROI images")
    t2w = OutputMultiPath(desc="output T2w images")
    flair = OutputMultiPath(desc="output FLAIR images")
    dwi = OutputMultiPath(desc="output DWI images")


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
        anat_only = kwargs.pop("anat_only")
        dwi_only = kwargs.pop("dwi_only")
        anatomical_contrast = kwargs.pop("anatomical_contrast")
        self._anatomical_contrast = anatomical_contrast
        super(BIDSDataGrabber, self).__init__(*args, **kwargs)
        if anat_only is not None:
            self._require_funcs = not anat_only
        self._no_anat_necessary = bool(dwi_only) or anatomical_contrast == "none"

    def _run_interface(self, runtime):
        bids_dict = self.inputs.subject_data

        self._results["out_dict"] = bids_dict
        self._results.update(bids_dict)

        if not bids_dict["t1w"]:
            message = "No T1w images found for subject sub-{}".format(self.inputs.subject_id)
            if self._no_anat_necessary:
                LOGGER.info("%s, but no problem because --dwi-only was selected.", message)
            elif self._anatomical_contrast != "T1w":
                LOGGER.info(
                    "%s, but no problem because --anat-modality %s was selected.",
                    message,
                    self._anatomical_contrast,
                )
            else:
                raise FileNotFoundError(message)

        if not bids_dict["t2w"]:
            message = "No T2w images found for subject sub-{}".format(self.inputs.subject_id)
            if self._no_anat_necessary:
                LOGGER.info("%s, but no problem because --dwi-only was selected.", message)
            elif self._anatomical_contrast != "T2w":
                LOGGER.info(
                    "%s, but no problem because --anat-modality %s was selected.",
                    message,
                    self._anatomical_contrast,
                )
            else:
                raise FileNotFoundError(message)

        if self._no_anat_necessary and not bids_dict["dwi"]:
            raise FileNotFoundError(
                "No DWI images found for subject sub-{}".format(self.inputs.subject_id)
            )

        for imtype in ["flair", "fmap", "sbref", "roi", "dwi"]:
            if not bids_dict[imtype]:
                LOGGER.warning("No '%s' images found for sub-%s", imtype, self.inputs.subject_id)

        return runtime


class DerivativesDataSinkInputSpec(BaseInterfaceInputSpec):
    base_directory = traits.Directory(desc="Path to the base directory for storing data.")
    in_file = traits.Either(
        traits.Directory(exists=True),
        InputMultiObject(File(exists=True)),
        mandatory=True,
        desc="the object to be saved",
    )
    source_file = File(mandatory=True, desc="the original file or name of merged files")
    space = traits.Str("", usedefault=True, desc="Label for space field")
    desc = traits.Str("", usedefault=True, desc="Label for description field")
    bundle = traits.Str("", usedefault=True, desc="Label for bundle field")
    suffix = traits.Str("", usedefault=True, desc="suffix appended to source_file")
    keep_dtype = traits.Bool(False, usedefault=True, desc="keep datatype suffix")
    extra_values = traits.List(traits.Str)
    compress = traits.Bool(
        desc="force compression (True) or uncompression (False)"
        " of the output file (default: same as input)"
    )
    extension = traits.Str()


class DerivativesDataSinkOutputSpec(TraitedSpec):
    out_file = traits.Either(
        traits.Directory(exists=True),
        InputMultiObject(File(exists=True)),
        desc="written file path",
    )
    compression = OutputMultiPath(
        traits.Bool,
        desc="whether ``in_file`` was compressed/uncompressed " "or `it was copied directly.",
    )


class DerivativesDataSink(SimpleInterface):
    """
    Saves the `in_file` into a BIDS-Derivatives folder provided
    by `base_directory`, given the input reference `source_file`.

    >>> from pathlib import Path
    >>> import tempfile
    >>> from qsiprep.utils.bids import collect_data
    >>> tmpdir = Path(tempfile.mkdtemp())
    >>> tmpfile = tmpdir / 'a_temp_file.nii.gz'
    >>> tmpfile.open('w').close()  # "touch" the file
    >>> dsink = DerivativesDataSink(base_directory=str(tmpdir))
    >>> dsink.inputs.in_file = str(tmpfile)
    >>> dsink.inputs.source_file = collect_data('ds114', '01')[0]['t1w'][0]
    >>> dsink.inputs.keep_dtype = True
    >>> dsink.inputs.suffix = 'target-mni'
    >>> res = dsink.run()
    >>> res.outputs.out_file  # doctest: +ELLIPSIS
    '.../qsiprep/sub-01/ses-retest/anat/sub-01_ses-retest_target-mni_T1w.nii.gz'

    >>> bids_dir = tmpdir / 'bidsroot' / 'sub-02' / 'ses-noanat' / 'func'
    >>> bids_dir.mkdir(parents=True, exist_ok=True)
    >>> tricky_source = bids_dir / 'sub-02_ses-noanat_task-rest_run-01_bold.nii.gz'
    >>> tricky_source.open('w').close()
    >>> dsink = DerivativesDataSink(base_directory=str(tmpdir))
    >>> dsink.inputs.in_file = str(tmpfile)
    >>> dsink.inputs.source_file = str(tricky_source)
    >>> dsink.inputs.keep_dtype = True
    >>> dsink.inputs.desc = 'preproc'
    >>> res = dsink.run()
    >>> res.outputs.out_file  # doctest: +ELLIPSIS
    '.../qsiprep/sub-02/ses-noanat/func/sub-02_ses-noanat_task-rest_run-01_desc-preproc_bold.nii.gz'

    """

    input_spec = DerivativesDataSinkInputSpec
    output_spec = DerivativesDataSinkOutputSpec
    out_path_base = "qsiprep"
    _always_run = True

    def __init__(self, out_path_base=None, **inputs):
        super(DerivativesDataSink, self).__init__(**inputs)
        self._results["out_file"] = []
        if out_path_base:
            self.out_path_base = out_path_base

    def _run_interface(self, runtime):
        src_fname, _ = _splitext(self.inputs.source_file)
        src_fname, dtype = src_fname.rsplit("_", 1)
        _, ext = _splitext(self.inputs.in_file[0])
        if self.inputs.compress is True and not ext.endswith(".gz"):
            ext += ".gz"
        elif self.inputs.compress is False and ext.endswith(".gz"):
            ext = ext[:-3]

        m = BIDS_NAME.search(src_fname)

        mod = op.basename(op.dirname(self.inputs.source_file))

        base_directory = runtime.cwd
        if isdefined(self.inputs.base_directory):
            base_directory = str(self.inputs.base_directory)

        out_path = "{}/{subject_id}".format(self.out_path_base, **m.groupdict())
        if m.groupdict().get("session_id") is not None:
            out_path += "/{session_id}".format(**m.groupdict())
        out_path += "/{}".format(mod)

        out_path = op.join(base_directory, out_path)

        os.makedirs(out_path, exist_ok=True)
        base_fname = op.join(out_path, src_fname)

        formatstr = "{bname}{space}{desc}{suffix}{dtype}{ext}"

        space = "_space-{}".format(self.inputs.space) if self.inputs.space else ""
        desc = "_desc-{}".format(self.inputs.desc) if self.inputs.desc else ""
        suffix = "_{}".format(self.inputs.suffix) if self.inputs.suffix else ""
        dtype = "" if not self.inputs.keep_dtype else ("_%s" % dtype)

        # If the derivative is a directory, copy it over
        copy_dir = op.isdir(str(self.inputs.in_file))
        if copy_dir:
            out_file = formatstr.format(
                bname=base_fname, space=space, desc=desc, suffix=suffix, dtype=dtype, ext=""
            )
            # os.makedirs(out_file, exist_ok=True)
            copytree(str(self.inputs.in_file), out_file, dirs_exist_ok=True)
            self._results["out_file"] = out_file
            return runtime

        if len(self.inputs.in_file) > 1 and not isdefined(self.inputs.extra_values):
            formatstr = "{bname}{space}{desc}{suffix}{i:04d}{dtype}{ext}"

        # Otherwise it's file(s)
        self._results["compression"] = []
        for i, fname in enumerate(self.inputs.in_file):
            out_file = formatstr.format(
                bname=base_fname, space=space, desc=desc, suffix=suffix, i=i, dtype=dtype, ext=ext
            )
            if isdefined(self.inputs.extra_values):
                out_file = out_file.format(extra_value=self.inputs.extra_values[i])
            self._results["out_file"].append(out_file)
            self._results["compression"].append(_copy_any(fname, out_file))
        return runtime


class _DerivativesMaybeDataSinkInputSpec(DerivativesDataSinkInputSpec):
    in_file = traits.Either(
        traits.Directory(exists=True),
        InputMultiObject(File(exists=True)),
        mandatory=False,
        desc="the object to be saved",
    )


class DerivativesMaybeDataSink(DerivativesDataSink):
    input_spec = _DerivativesMaybeDataSinkInputSpec

    def _run_interface(self, runtime):
        if not isdefined(self.inputs.in_file):
            return runtime
        return super(DerivativesMaybeDataSink, self)._run_interface(runtime)


def _splitext(fname):
    fname, ext = op.splitext(op.basename(fname))
    if ext == ".gz":
        fname, ext2 = op.splitext(fname)
        ext = ext2 + ext
    return fname, ext


def _copy_any(src, dst):
    src_isgz = src.endswith(".gz")
    dst_isgz = dst.endswith(".gz")
    if src_isgz == dst_isgz:
        copyfile(src, dst, copy=True, use_hardlink=True)
        return False  # Make sure we do not reuse the hardlink later

    # Unlink target (should not exist)
    if os.path.exists(dst):
        os.unlink(dst)

    src_open = gzip.open if src_isgz else open
    dst_open = gzip.open if dst_isgz else open
    with src_open(src, "rb") as f_in:
        with dst_open(dst, "wb") as f_out:
            copyfileobj(f_in, f_out)
    return True
