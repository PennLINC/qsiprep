# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Interfaces for handling BIDS-like neuroimaging structures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fetch some example data:

    >>> import os
    >>> from qsiprep.niworkflows import data
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

import simplejson as json
from bids.layout import parse_file_entities
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
from nipype.utils.filemanip import copyfile, split_filename

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


recon_entity_order = ["model", "bundles", "fit", "mdp", "mfp", "bundle", "label"]


def get_recon_output_name(
    base_dir, source_file, derivative_file, qsirecon_suffix, output_bids_entities, use_ext=True
):

    source_entities = parse_file_entities(source_file)
    out_path = op.join(base_dir, f"qsirecon-{qsirecon_suffix}")
    out_path = op.join(out_path, "sub-" + source_entities["subject"])
    if "session" in source_entities:
        out_path += "/ses-{session}".format(**source_entities)

    # Which datatype directory should this go under?
    # If it's not in the output bids entities use the source_file datatype
    datatype_dir = output_bids_entities.get("datatype", op.basename(op.dirname(source_file)))
    out_path += f"/{datatype_dir}"
    _, source_fname, _ = split_filename(source_file)
    source_fname, _ = source_fname.rsplit("_", 1)
    _, _, extension = split_filename(derivative_file)

    # It may be that the space has changed. Check if it has
    if "space" in output_bids_entities:
        source_fname = re.sub(
            "_space-[a-zA-Z0-9]+", "_space-" + output_bids_entities["space"], source_fname
        )
    base_fname = op.join(out_path, source_fname)

    # Add the new bids entities for the output file
    for entity_name in recon_entity_order:
        if entity_name in output_bids_entities:
            base_fname += "_{entity}-{value}".format(
                entity=entity_name, value=output_bids_entities[entity_name]
            )

    # Add the suffix
    suffix = output_bids_entities.get("suffix", "dwimap")
    if use_ext:
        return f"{base_fname}_{suffix}{extension}"

    return f"{base_fname}_{suffix}"


class _ReconDerivativesDataSinkInputSpec(DerivativesDataSinkInputSpec):
    mdp = traits.Str("", usedefault=True, desc="Label for model derived parameter field")
    mfp = traits.Str("", usedefault=True, desc="Label for model fit parameter field")
    model = traits.Str("", usedefault=True, desc="Label for model field")
    bundle = traits.Str("", usedefault=True, desc="Label for bundle field")
    bundles = traits.Str("", usedefault=True, desc="Label for bundles field")
    fit = traits.Str("", usedefault=True, desc="Label for fit field")
    label = traits.Str("", usedefault=True, desc="Label for label field")
    qsirecon_suffix = traits.Str(
        "", usedefault=True, desc="name appended to qsirecon- in the derivatives"
    )


class ReconDerivativesDataSink(DerivativesDataSink):
    input_spec = _ReconDerivativesDataSinkInputSpec
    out_path_base = "qsirecon"

    def _run_interface(self, runtime):

        # If there is no qsirecon suffix, then we're not saving this file
        if not self.inputs.qsirecon_suffix:
            return runtime

        # Figure out what the extension should be based on the input file and compression
        src_fname, _ = _splitext(self.inputs.source_file)
        src_fname, dtype = src_fname.rsplit("_", 1)
        _, ext = _splitext(self.inputs.in_file[0])
        if self.inputs.compress is True and not ext.endswith(".gz"):
            ext += ".gz"
        elif self.inputs.compress is False and ext.endswith(".gz"):
            ext = ext[:-3]

        # Prepare the bids entities from the inputs
        output_bids = {}
        if self.inputs.bundles:
            output_bids["bundles"] = self.inputs.bundles
        if self.inputs.bundle:
            output_bids["bundle"] = self.inputs.bundle
        if self.inputs.space:
            output_bids["space"] = self.inputs.space
        if self.inputs.model:
            output_bids["model"] = self.inputs.model
        if self.inputs.mdp:
            output_bids["mdp"] = self.inputs.mdp
        if self.inputs.mfp:
            output_bids["mfp"] = self.inputs.mfp
        if self.inputs.fit:
            output_bids["fit"] = self.inputs.fit
        if self.inputs.suffix:
            output_bids["suffix"] = self.inputs.suffix
        if self.inputs.label:
            output_bids["label"] = self.inputs.label

        # Get the output name without an extension
        bname = get_recon_output_name(
            base_dir=self.inputs.base_directory,
            source_file=self.inputs.source_file,
            derivative_file=self.inputs.in_file[0],
            qsirecon_suffix=self.inputs.qsirecon_suffix,
            output_bids_entities=output_bids,
            use_ext=False,
        )

        # Ensure the directory exists
        os.makedirs(op.dirname(bname), exist_ok=True)

        formatstr = "{bname}{ext}"
        # If the derivative is a directory, copy it over
        copy_dir = op.isdir(str(self.inputs.in_file[0]))
        if copy_dir:
            out_file = formatstr.format(bname=bname, ext="")
            copytree(str(self.inputs.in_file), out_file, dirs_exist_ok=True)
            self._results["out_file"] = out_file
            return runtime

        if len(self.inputs.in_file) > 1 and not isdefined(self.inputs.extra_values):
            formatstr = "{bname}{i:04d}{ext}"

        # Otherwise it's file(s)
        self._results["compression"] = []
        for i, fname in enumerate(self.inputs.in_file):
            out_file = formatstr.format(bname=bname, i=i, ext=ext)
            if isdefined(self.inputs.extra_values):
                out_file = out_file.format(extra_value=self.inputs.extra_values[i])
            self._results["out_file"].append(out_file)
            self._results["compression"].append(_copy_any(fname, out_file))
        return runtime


class ReadSidecarJSONInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="the input nifti file")
    fields = traits.List(traits.Str, desc="get only certain fields")


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

    expr = re.compile(
        "^sub-(?P<subject_id>[a-zA-Z0-9]+)(_ses-(?P<session_id>[a-zA-Z0-9]+))?"
        "(_task-(?P<task_id>[a-zA-Z0-9]+))?(_acq-(?P<acq_id>[a-zA-Z0-9]+))?"
        "(_rec-(?P<rec_id>[a-zA-Z0-9]+))?(_run-(?P<run_id>[a-zA-Z0-9]+))?"
    )
    input_spec = ReadSidecarJSONInputSpec
    output_spec = ReadSidecarJSONOutputSpec
    _always_run = True

    def _run_interface(self, runtime):
        metadata = get_metadata_for_nifti(self.inputs.in_file)
        output_keys = [key for key in list(self.output_spec().get().keys()) if key.endswith("_id")]
        outputs = self.expr.search(op.basename(self.inputs.in_file)).groupdict()

        for key in output_keys:
            id_value = outputs.get(key)
            if id_value is not None:
                self._results[key] = outputs.get(key)

        if isdefined(self.inputs.fields) and self.inputs.fields:
            for fname in self.inputs.fields:
                self._results[fname] = metadata[fname]
        else:
            self._results["out_dict"] = metadata

        return runtime


def get_metadata_for_nifti(in_file):
    """Fetch metadata for a given nifti file"""
    in_file = op.abspath(in_file)

    fname, ext = op.splitext(in_file)
    if ext == ".gz":
        fname, ext2 = op.splitext(fname)
        ext = ext2 + ext

    side_json = fname + ".json"
    fname_comps = op.basename(side_json).split("_")

    session_comp_list = []
    subject_comp_list = []
    top_comp_list = []
    ses = None
    sub = None

    for comp in fname_comps:
        if comp[:3] != "run":
            session_comp_list.append(comp)
            if comp[:3] == "ses":
                ses = comp
            else:
                subject_comp_list.append(comp)
                if comp[:3] == "sub":
                    sub = comp
                else:
                    top_comp_list.append(comp)

    if any([comp.startswith("ses") for comp in fname_comps]):
        bids_dir = "/".join(op.dirname(in_file).split("/")[:-3])
    else:
        bids_dir = "/".join(op.dirname(in_file).split("/")[:-2])

    top_json = op.join(bids_dir, "_".join(top_comp_list))
    potential_json = [top_json]

    subject_json = op.join(bids_dir, sub, "_".join(subject_comp_list))
    potential_json.append(subject_json)

    if ses:
        session_json = op.join(bids_dir, sub, ses, "_".join(session_comp_list))
        potential_json.append(session_json)

    potential_json.append(side_json)

    merged_param_dict = {}
    for json_file_path in potential_json:
        if op.isfile(json_file_path):
            with open(json_file_path, "r") as jsonfile:
                param_dict = json.load(jsonfile)
                merged_param_dict.update(param_dict)

    return merged_param_dict


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
