# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
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
"""Stripped out routines for Sentry"""
import os
import re
from pathlib import Path

from nibabel.optpkg import optional_package
from niworkflows.utils.misc import read_crashfile

sentry_sdk = optional_package("sentry_sdk")[0]
migas = optional_package("migas")[0]

from .. import __version__, config

CHUNK_SIZE = 16384
# Group common events with pre specified fingerprints
KNOWN_ERRORS = {
    "permission-denied": ["PermissionError: [Errno 13] Permission denied"],
    "memory-error": [
        "MemoryError",
        "Cannot allocate memory",
        "Return code: 134",
    ],
    "reconall-already-running": ["ERROR: it appears that recon-all is already running"],
    "no-disk-space": ["[Errno 28] No space left on device", "[Errno 122] Disk quota exceeded"],
    "segfault": [
        "Segmentation Fault",
        "Segfault",
        "Return code: 139",
    ],
    "potential-race-condition": [
        "[Errno 39] Directory not empty",
        "_unfinished.json",
    ],
    "keyboard-interrupt": [
        "KeyboardInterrupt",
    ],
}

# Not useful for error reports
USELESS_OPTS = [
    "bids_dir",
    "output_dir",
    "participant_label",
    "bids_database_dir",
    "bids_filter_file",
    "recon_input",
    "use_plugin",
    "fs_license_file",
    "work_dir",
]


def start_ping(run_uuid, npart):
    with sentry_sdk.configure_scope() as scope:
        if run_uuid:
            scope.set_tag("run_uuid", run_uuid)
        scope.set_tag("npart", npart)
    sentry_sdk.add_breadcrumb(message="QSIPrep started", level="info")
    sentry_sdk.capture_message("QSIPrep started", level="info")


def sentry_setup():
    release = config.environment.version or "dev"
    environment = (
        "dev"
        if (
            os.getenv("QSIPREP_DEV", "").lower in ("1", "on", "yes", "y", "true")
            or ("+" in release)
        )
        else "prod"
    )

    sentry_sdk.init(
        "https://7e85f156850d463fb77eb54045df50aa@sentry.io/1802153",
        release=release,
        environment=environment,
        before_send=before_send,
    )

    with sentry_sdk.configure_scope() as scope:
        for k, v in config.get(flat=True).items():
            if k not in USELESS_OPTS:
                scope.set_tag(k, v)


def process_crashfile(crashfile):
    """Parse the contents of a crashfile and submit sentry messages."""
    crash_info = read_crashfile(str(crashfile))
    with sentry_sdk.push_scope() as scope:
        scope.level = "fatal"

        # Extract node name
        node_name = crash_info.pop("node").split(".")[-1]
        scope.set_tag("node_name", node_name)

        # Massage the traceback, extract the gist
        traceback = crash_info.pop("traceback")
        # last line is probably most informative summary
        gist = traceback.splitlines()[-1]
        exception_text_start = 1
        for line in traceback.splitlines()[1:]:
            if not line[0].isspace():
                break
            exception_text_start += 1

        exception_text = "\n".join(traceback.splitlines()[exception_text_start:])

        # Extract inputs, if present
        inputs = crash_info.pop("inputs", None)
        if inputs:
            scope.set_extra("inputs", dict(inputs))

        # Extract any other possible metadata in the crash file
        for k, v in crash_info.items():
            strv = _chunks(str(v))
            if len(strv) == 1:
                scope.set_extra(k, strv[0])
            else:
                for i, chunk in enumerate(strv):
                    scope.set_extra("%s_%02d" % (k, i), chunk)

        fingerprint = ""
        issue_title = f"{node_name}: {gist}"
        for new_fingerprint, error_snippets in KNOWN_ERRORS.items():
            for error_snippet in error_snippets:
                if error_snippet in traceback:
                    fingerprint = new_fingerprint
                    issue_title = new_fingerprint
                    break
            if fingerprint:
                break

        message = issue_title + "\n\n"
        message += exception_text[-8192:]
        if fingerprint:
            sentry_sdk.add_breadcrumb(message=fingerprint, level="fatal")
        else:
            # remove file paths
            fingerprint = re.sub(r"(/[^/ ]*)+/?", "", message)
            # remove words containing numbers
            fingerprint = re.sub(r"([a-zA-Z]*[0-9]+[a-zA-Z]*)+", "", fingerprint)
            # adding the return code if it exists
            for line in message.splitlines():
                if line.startswith("Return code"):
                    fingerprint += line
                    break

        scope.fingerprint = [fingerprint]
        sentry_sdk.capture_message(message, "fatal")


def before_send(event, hints):
    """Filter log messages about crashed nodes."""
    if "logentry" in event and "message" in event["logentry"]:
        msg = event["logentry"]["message"]
        if msg.startswith("could not run node:"):
            return None
        if msg.startswith("Saving crash info to "):
            return None
        if re.match("Node .+ failed to run on host .+", msg):
            return None

    if "breadcrumbs" in event and isinstance(event["breadcrumbs"], list):
        fingerprints_to_propagate = [
            "no-disk-space",
            "memory-error",
            "permission-denied",
            "keyboard-interrupt",
        ]
        for bc in event["breadcrumbs"]:
            msg = bc.get("message", "empty-msg")
            if msg in fingerprints_to_propagate:
                event["fingerprint"] = [msg]
                break

    return event


def _chunks(string, length=CHUNK_SIZE):
    """
    Split a string into smaller chunks.

    >>> list(_chunks('some longer string.', length=3))
    ['som', 'e l', 'ong', 'er ', 'str', 'ing', '.']

    """
    return [string[i : i + length] for i in range(0, len(string), length)]
