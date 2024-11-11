# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copied recent function write_bidsignore
#
# Copyright The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""
Utilities to handle BIDS inputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fetch some test data

    >>> import os
    >>> from niworkflows import data
    >>> data_root = data.get_bids_examples(variant='BIDS-examples-1-enh-ds054')
    >>> os.chdir(data_root)

"""
import json
import os
import sys
import typing as ty
import warnings
from pathlib import Path

import nibabel as nb
import numpy as np
import pandas as pd
from bids import BIDSLayout
from bids.layout import Query

from .. import config

IMPORTANT_DWI_FIELDS = [
    # From image headers:
    "Obliquity",
    "ImageOrientation",
    "NumVolumes",
    "Dim1Size",
    "Dim2Size",
    "Dim3Size",
    "VoxelSizeDim1",
    "VoxelSizeDim2",
    "VoxelSizeDim3",
    # From sidecars:
    "ParallelReductionFactorInPlane",
    "ParallelAcquisitionTechnique",
    "ParallelAcquisitionTechnique",
    "PartialFourier",
    "PhaseEncodingDirection",
    "EffectiveEchoSpacing",
    "TotalReadoutTime",
    "EchoTime",
    "SliceEncodingDirection",
    "DwellTime",
    "FlipAngle",
    "MultibandAccelerationFactor",
    "RepetitionTime",
]
SUPPORTED_AGE_UNITS = (
    "weeks",
    "months",
    "years",
)


class BIDSError(ValueError):
    def __init__(self, message, bids_root):
        indent = 10
        header = '{sep} BIDS root folder: "{bids_root}" {sep}'.format(
            bids_root=bids_root, sep="".join(["-"] * indent)
        )
        self.msg = "\n{header}\n{indent}{message}\n{footer}".format(
            header=header,
            indent="".join([" "] * (indent + 1)),
            message=message,
            footer="".join(["-"] * len(header)),
        )
        super(BIDSError, self).__init__(self.msg)
        self.bids_root = bids_root


class BIDSWarning(RuntimeWarning):
    pass


def collect_participants(bids_dir, participant_label=None, strict=False, bids_validate=True):
    """
    List the participants under the BIDS root and checks that participants
    designated with the participant_label argument exist in that folder.

    Returns the list of participants to be finally processed.

    Requesting all subjects in a BIDS directory root:

    >>> collect_participants('ds114')
    ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

    Requesting two subjects, given their IDs:

    >>> collect_participants('ds114', participant_label=['02', '04'])
    ['02', '04']

    Requesting two subjects, given their IDs (works with 'sub-' prefixes):

    >>> collect_participants('ds114', participant_label=['sub-02', 'sub-04'])
    ['02', '04']

    Requesting two subjects, but one does not exist:

    >>> collect_participants('ds114', participant_label=['02', '14'])
    ['02']

    >>> collect_participants('ds114', participant_label=['02', '14'],
    ...                      strict=True)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    qsiprep.utils.bids.BIDSError:
    ...


    """
    if isinstance(bids_dir, BIDSLayout):
        layout = bids_dir
    else:
        raise Exception("A layout is required")

    all_participants = set(layout.get_subjects())

    # Error: bids_dir does not contain subjects
    if not all_participants:
        raise BIDSError(
            "Could not find participants. Please make sure the BIDS data "
            "structure is present and correct. Datasets can be validated "
            "online using the BIDS Validator "
            "(http://incf.github.io/bids-validator/).\n"
            "If you are using Docker for Mac or Docker for Windows, you "
            'may need to adjust your "File sharing" preferences.',
            bids_dir,
        )

    # No --participant-label was set, return all
    if not participant_label:
        return sorted(all_participants)

    if isinstance(participant_label, str):
        participant_label = [participant_label]

    # Drop sub- prefixes
    participant_label = [sub[4:] if sub.startswith("sub-") else sub for sub in participant_label]
    # Remove duplicates
    participant_label = sorted(set(participant_label))
    # Remove labels not found
    found_label = sorted(set(participant_label) & set(all_participants))
    if not found_label:
        raise BIDSError(
            "Could not find participants [{}]".format(", ".join(participant_label)), bids_dir
        )

    # Warn if some IDs were not found
    notfound_label = sorted(set(participant_label) - set(all_participants))
    if notfound_label:
        exc = BIDSError(
            "Some participants were not found: {}".format(", ".join(notfound_label)), bids_dir
        )
        if strict:
            raise exc
        warnings.warn(exc.msg, BIDSWarning)

    return found_label


def collect_data(bids_dir, participant_label, session_id=None, filters=None, bids_validate=True):
    """Use pybids to retrieve the input data for a given participant."""
    if isinstance(bids_dir, BIDSLayout):
        layout = bids_dir
    else:
        layout = BIDSLayout(str(bids_dir), validate=bids_validate)

    queries = {
        "fmap": {"datatype": "fmap"},
        "sbref": {"datatype": "func", "suffix": "sbref"},
        "flair": {"datatype": "anat", "suffix": "FLAIR"},
        "t2w": {"datatype": "anat", "suffix": "T2w"},
        "t1w": {"datatype": "anat", "suffix": "T1w"},
        "roi": {"datatype": "anat", "suffix": "roi"},
        "dwi": {"datatype": "dwi", "part": ["mag", None], "suffix": "dwi"},
    }
    bids_filters = filters or {}
    for acq, entities in bids_filters.items():
        queries[acq].update(entities)

    subj_data = {
        dtype: sorted(
            layout.get(
                return_type="file",
                subject=participant_label,
                session=session_id or Query.OPTIONAL,
                extension=["nii", "nii.gz"],
                **query,
            )
        )
        for dtype, query in queries.items()
    }

    return subj_data, layout


def write_derivative_description(bids_dir, deriv_dir):
    from qsiprep import __version__

    DOWNLOAD_URL = f"https://github.com/PennLINC/qsiprep/archive/{__version__}.tar.gz"

    desc = {
        "Name": "qsiprep output",
        "BIDSVersion": "1.1.1",
        "PipelineDescription": {
            "Name": "qsiprep",
            "Version": __version__,
            "CodeURL": DOWNLOAD_URL,
        },
        "GeneratedBy": [
            {
                "Name": "qsiprep",
                "Version": __version__,
                "CodeURL": DOWNLOAD_URL,
            }
        ],
        "CodeURL": "https://github.com/pennbbl/qsiprep",
        "HowToAcknowledge": "Please cite our paper "
        "(https://www.nature.com/articles/s41592-021-01185-5#citeas), and "
        "include the generated citation boilerplate within the Methods "
        "section of the text.",
    }

    # Keys that can only be set by environment
    if "QSIPREP_DOCKER_TAG" in os.environ:
        desc["DockerHubContainerTag"] = os.environ["QSIPREP_DOCKER_TAG"]
    if "QSIPREP_SINGULARITY_URL" in os.environ:
        singularity_url = os.environ["QSIPREP_SINGULARITY_URL"]
        desc["SingularityContainerURL"] = singularity_url
        try:
            desc["SingularityContainerMD5"] = _get_shub_version(singularity_url)
        except ValueError:
            pass
    if "QSIPREP_APPTAINER_URL" in os.environ:
        apptainer_url = os.environ["QSIPREP_APPTAINER_URL"]
        desc["ApptainerContainerURL"] = apptainer_url
        try:
            desc["ApptainerContainerMD5"] = _get_ahub_version(apptainer_url)
        except ValueError:
            pass

    # Keys deriving from source dataset
    fname = os.path.join(bids_dir, "dataset_description.json")
    if os.path.exists(fname):
        with open(fname) as fobj:
            orig_desc = json.load(fobj)
    else:
        orig_desc = {}

    if "DatasetDOI" in orig_desc:
        desc["SourceDatasetsURLs"] = ["https://doi.org/{}".format(orig_desc["DatasetDOI"])]
    if "License" in orig_desc:
        desc["License"] = orig_desc["License"]

    with open(os.path.join(deriv_dir, "dataset_description.json"), "w") as fobj:
        json.dump(desc, fobj, indent=4)


def write_bidsignore(deriv_dir):
    bids_ignore = (
        "*.html",
        "logs/",
        "figures/",  # Reports
        "*_xfm.*",  # Unspecified transform files
        "*.surf.gii",  # Unspecified structural outputs
        # Unspecified functional outputs
        "*_boldref.nii.gz",
        "*_bold.func.gii",
        "*_mixing.tsv",
        "*_timeseries.tsv",
    )
    ignore_file = Path(deriv_dir) / ".bidsignore"

    ignore_file.write_text("\n".join(bids_ignore) + "\n")


def validate_input_dir(exec_env, bids_dir, participant_label):
    # Ignore issues and warnings that should not influence qsiprep
    import subprocess
    import tempfile

    validator_config_dict = {
        "ignore": [
            "EVENTS_COLUMN_ONSET",
            "EVENTS_COLUMN_DURATION",
            "TSV_EQUAL_ROWS",
            "TSV_EMPTY_CELL",
            "TSV_IMPROPER_NA",
            "VOLUME_COUNT_MISMATCH",
            "INCONSISTENT_SUBJECTS",
            "INCONSISTENT_PARAMETERS",
            "PARTICIPANT_ID_COLUMN",
            "PARTICIPANT_ID_MISMATCH",
            "TASK_NAME_MUST_DEFINE",
            "PHENOTYPE_SUBJECTS_MISSING",
            "STIMULUS_FILE_MISSING",
            "EVENTS_TSV_MISSING",
            "TSV_IMPROPER_NA",
            "ACQTIME_FMT",
            "Participants age 89 or higher",
            "DATASET_DESCRIPTION_JSON_MISSING",
            "FILENAME_COLUMN",
            "WRONG_NEW_LINE",
            "MISSING_TSV_COLUMN_CHANNELS",
            "MISSING_TSV_COLUMN_IEEG_CHANNELS",
            "MISSING_TSV_COLUMN_IEEG_ELECTRODES",
            "UNUSED_STIMULUS",
            "CHANNELS_COLUMN_SFREQ",
            "CHANNELS_COLUMN_LOWCUT",
            "CHANNELS_COLUMN_HIGHCUT",
            "CHANNELS_COLUMN_NOTCH",
            "CUSTOM_COLUMN_WITHOUT_DESCRIPTION",
            "ACQTIME_FMT",
            "SUSPICIOUSLY_LONG_EVENT_DESIGN",
            "SUSPICIOUSLY_SHORT_EVENT_DESIGN",
            "MISSING_TSV_COLUMN_EEG_ELECTRODES",
            "MISSING_SESSION",
            "NO_T1W",
        ],
        "ignoredFiles": ["/README", "/dataset_description.json", "/participants.tsv"],
    }
    # Limit validation only to data from requested participants
    if participant_label:
        all_subs = set([s.name[4:] for s in bids_dir.glob("sub-*")])
        selected_subs = set([s[4:] if s.startswith("sub-") else s for s in participant_label])
        bad_labels = selected_subs.difference(all_subs)
        if bad_labels:
            error_msg = (
                "Data for requested participant(s) label(s) not found. Could "
                "not find data for participant(s): %s. Please verify the requested "
                "participant labels."
            )
            if exec_env == "docker":
                error_msg += (
                    " This error can be caused by the input data not being "
                    "accessible inside the docker container. Please make sure all "
                    "volumes are mounted properly (see https://docs.docker.com/"
                    "engine/reference/commandline/run/#mount-volume--v---read-only)"
                )
            if exec_env == "singularity":
                error_msg += (
                    " This error can be caused by the input data not being "
                    "accessible inside the singularity container. Please make sure "
                    "all paths are mapped properly (see https://www.sylabs.io/"
                    "guides/3.0/user-guide/bind_paths_and_mounts.html)"
                )
            if exec_env == "apptainer":
                error_msg += (
                    " This error can be caused by the input data not being "
                    "accessible inside the Apptainer container. Please make sure "
                    "all paths are mapped properly (see https://apptainer.org/",
                    "docs/user/main/bind_paths_and_mounts.html)",
                )

            raise RuntimeError(error_msg % ",".join(bad_labels))

        ignored_subs = all_subs.difference(selected_subs)
        if ignored_subs:
            for sub in ignored_subs:
                validator_config_dict["ignoredFiles"].append("/sub-%s/**" % sub)
    with tempfile.NamedTemporaryFile("w+") as temp:
        temp.write(json.dumps(validator_config_dict))
        temp.flush()
        try:
            subprocess.check_call(["bids-validator", bids_dir, "-c", temp.name])
        except FileNotFoundError:
            print("bids-validator does not appear to be installed", file=sys.stderr)


def _get_shub_version(singularity_url):
    raise ValueError("Not yet implemented")


def _get_ahub_version(apptainer_url):
    raise ValueError("Not yet implemented")


def update_metadata_from_nifti_header(metadata, nifti_file):
    """Update a BIDS metadata dictionary with info from a NIfTI header.

    Code borrowed from CuBIDS.
    """
    img = nb.load(nifti_file)
    # get important info from niftis
    obliquity = np.any(nb.affines.obliquity(img.affine) > 1e-4)
    voxel_sizes = img.header.get_zooms()
    matrix_dims = img.shape
    # add nifti info to corresponding sidecars​

    metadata["Obliquity"] = str(obliquity)
    metadata["VoxelSizeDim1"] = float(voxel_sizes[0])
    metadata["VoxelSizeDim2"] = float(voxel_sizes[1])
    metadata["VoxelSizeDim3"] = float(voxel_sizes[2])
    metadata["Dim1Size"] = matrix_dims[0]
    metadata["Dim2Size"] = matrix_dims[1]
    metadata["Dim3Size"] = matrix_dims[2]
    if img.ndim == 4:
        metadata["NumVolumes"] = matrix_dims[3]
    elif img.ndim == 3:
        metadata["NumVolumes"] = 1.0
    orient = nb.orientations.aff2axcodes(img.affine)
    metadata["ImageOrientation"] = "".join(orient) + "+"


def scan_groups_to_sidecar(scan_groups):
    """Create a sidecar that reflects how the preprocessed image was created."""

    # Add the information about how the images were grouped and which fieldmaps were used
    derivatives_metadata = {"scan_grouping": scan_groups}

    # Get metadata from the individual scans that were combined to make this preprocessed image
    concatenated_dwi_files = scan_groups.get("dwi_series")
    fieldmap_info = scan_groups.get("fieldmap_info")
    if fieldmap_info.get("suffix") == "rpe_series":
        concatenated_dwi_files.extend(fieldmap_info.get("rpe_series", []))
    scan_metadata = {}
    for dwi_file in concatenated_dwi_files:
        dwi_file_name = Path(dwi_file).name
        scan_metadata[dwi_file_name] = config.execution.layout.get_metadata(dwi_file)
    derivatives_metadata["source_metadata"] = scan_metadata
    return derivatives_metadata


def parse_bids_for_age_months(
    bids_root: str | Path,
    subject_id: str,
    session_id: str | None = None,
) -> int | None:
    """
    Given a BIDS root, query the BIDS metadata files for participant age, and return in
    chronological months.

    The heuristic followed is:
    1) Check `sub-<subject_id>[/ses-<session_id>]/<sub-<subject_id>[_ses-<session-id>]_scans.tsv
    2) Check `sub-<subject_id>/sub-<subject_id>_sessions.tsv`
    3) Check `<root>/participants.tsv`

    Notes
    -----
    This function is derived from sources licensed under the Apache-2.0 terms.
    The original function this work derives from is found at:
    https://github.com/nipreps/nibabies/blob/7efc8c96d109cb755258209d83b1e164c481cf4e/
    nibabies/utils/bids.py#L218

    Copyright The NiPreps Developers <nipreps@gmail.com>

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    We support and encourage derived works from this project, please read
    about our expectations at

        https://www.nipreps.org/community/licensing/

    """
    if subject_id.startswith("sub-"):
        subject_id = subject_id[4:]
    if session_id and session_id.startswith("ses-"):
        session_id = session_id[4:]

    # Play nice with sessions
    subject = f"sub-{subject_id}"
    session = f"ses-{session_id}" if session_id else ""
    prefix = f"{subject}" + (f"_{session}" if session else "")

    subject_level = session_level = Path(bids_root) / subject
    if session_id:
        session_level = subject_level / session

    age = None

    scans_tsv = session_level / f"{prefix}_scans.tsv"
    if scans_tsv.exists():
        age = _get_age_from_tsv(
            scans_tsv,
            index_column="filename",
            index_value=r"^anat.*",
        )

    if age is not None:
        return age

    sessions_tsv = subject_level / f"{subject}_sessions.tsv"
    if sessions_tsv.exists() and session_id is not None:
        age = _get_age_from_tsv(sessions_tsv, index_column="session_id", index_value=session)

    if age is not None:
        return age

    participants_tsv = Path(bids_root) / "participants.tsv"
    if participants_tsv.exists() and age is None:
        age = _get_age_from_tsv(
            participants_tsv, index_column="participant_id", index_value=subject
        )

    return age


def _get_age_from_tsv(
    bids_tsv: Path,
    index_column: str | None = None,
    index_value: str | None = None,
) -> float | None:
    """Get age from TSV.

    Notes
    -----
    This function is derived from sources licensed under the Apache-2.0 terms.
    The original function this work derives from is found at:
    https://github.com/nipreps/nibabies/blob/7efc8c96d109cb755258209d83b1e164c481cf4e/
    nibabies/utils/bids.py#L275

    Copyright The NiPreps Developers <nipreps@gmail.com>

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    We support and encourage derived works from this project, please read
    about our expectations at

        https://www.nipreps.org/community/licensing/

    """
    df = pd.read_csv(str(bids_tsv), sep="\t")
    age_col = None

    for column in ("age_weeks", "age_months", "age_years", "age"):
        if column in df.columns:
            age_col = column
            break
    if age_col is None:
        return

    df = df[df[index_column].str.fullmatch(index_value)]

    # Multiple indices may be present after matching
    if len(df) > 1:
        warnings.warn(
            f"Multiple matches for {index_column}:{index_value} found in {bids_tsv.name}.",
            stacklevel=1,
        )

    try:
        # extract age value from row
        age = float(df.loc[df.index[0], age_col].item())
    except Exception:  # noqa: BLE001
        return

    if age_col == "age":
        # verify age is in months
        bids_json = bids_tsv.with_suffix(".json")
        age_units = _get_age_units(bids_json)
        if age_units is False:
            raise FileNotFoundError(
                f"Could not verify age unit for {bids_tsv.name} - ensure a sidecar JSON "
                "describing column `age` units is available."
            )
    else:
        age_units = age_col.split("_")[-1]

    age_months = age_to_months(age, units=age_units)
    return age_months


def _get_age_units(bids_json: Path) -> ty.Literal["weeks", "months", "years", False]:
    """Get age units.

    Notes
    -----
    This function is derived from sources licensed under the Apache-2.0 terms.
    The original function this work derives from is found at:
    https://github.com/nipreps/nibabies/blob/7efc8c96d109cb755258209d83b1e164c481cf4e/
    nibabies/utils/bids.py#L321

    Copyright The NiPreps Developers <nipreps@gmail.com>

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    We support and encourage derived works from this project, please read
    about our expectations at

        https://www.nipreps.org/community/licensing/

    """
    try:
        data = json.loads(bids_json.read_text())
    except (json.JSONDecodeError, OSError):
        return False

    units = data.get("age", {}).get("Units", "")
    if not isinstance(units, str):
        # Multiple units consfuse us
        return False

    if units.lower() in SUPPORTED_AGE_UNITS:
        return units.lower()
    return False


def age_to_months(age: int | float, units: ty.Literal["weeks", "months", "years"]) -> int:
    """Convert a given age, in either "weeks", "months", or "years", into months.

    >>> age_to_months(1, "years")
    12
    >>> age_to_months(0.5, "years")
    6
    >>> age_to_months(2, "weeks")
    0
    >>> age_to_months(3, "weeks")
    1
    >>> age_to_months(8, "months")
    8

    Notes
    -----
    This function is derived from sources licensed under the Apache-2.0 terms.
    The original function this work derives from is found at:
    https://github.com/nipreps/nibabies/blob/7efc8c96d109cb755258209d83b1e164c481cf4e/
    nibabies/utils/bids.py#L337

    Copyright The NiPreps Developers <nipreps@gmail.com>

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    We support and encourage derived works from this project, please read
    about our expectations at

        https://www.nipreps.org/community/licensing/

    """
    WEEKS_TO_MONTH = 0.230137
    YEARS_TO_MONTH = 12

    if units == "weeks":
        age *= WEEKS_TO_MONTH
    elif units == "years":
        age *= YEARS_TO_MONTH
    return int(round(age))


def cohort_by_months(template, months):
    """Produce a recommended cohort based on partipants age.

    Notes
    -----
    This function is derived from sources licensed under the Apache-2.0 terms.
    The original function this work derives from is found at:
    https://github.com/nipreps/nibabies/blob/7efc8c96d109cb755258209d83b1e164c481cf4e/
    nibabies/utils/misc.py#L50

    Copyright The NiPreps Developers <nipreps@gmail.com>

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    We support and encourage derived works from this project, please read
    about our expectations at

        https://www.nipreps.org/community/licensing/

    """
    cohort_key = {
        "MNIInfant": (
            # upper bound of template | cohort
            2,  # 1
            5,  # 2
            8,  # 3
            11,  # 4
            14,  # 5
            17,  # 6
            21,  # 7
            27,  # 8
            33,  # 9
            44,  # 10
            60,  # 11
        ),
        "UNCInfant": (
            8,  # 1
            12,  # 2
            24,  # 3
        ),
    }
    ages = cohort_key.get(template)
    if ages is None:
        raise KeyError("Template cohort information does not exist.")

    for cohort, age in enumerate(ages, 1):
        if months <= age:
            return cohort
    raise KeyError("Age exceeds all cohorts!")
