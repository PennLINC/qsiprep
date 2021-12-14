# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Helpers for handling BIDS-like neuroimaging structures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


"""
from pathlib import Path
import warnings
import re
import simplejson as json

from .misc import splitext

__all__ = ['BIDS_NAME']

BIDS_NAME = re.compile(
    r'^(.*\/)?(?P<subject_id>sub-[a-zA-Z0-9]+)(_(?P<session_id>ses-[a-zA-Z0-9]+))?'
    '(_(?P<task_id>task-[a-zA-Z0-9]+))?(_(?P<acq_id>acq-[a-zA-Z0-9]+))?'
    '(_(?P<rec_id>rec-[a-zA-Z0-9]+))?(_(?P<run_id>run-[a-zA-Z0-9]+))?')


class BIDSError(ValueError):
    def __init__(self, message, bids_root):
        indent = 10
        header = '{sep} BIDS root folder: "{bids_root}" {sep}'.format(
            bids_root=bids_root, sep=''.join(['-'] * indent))
        self.msg = '\n{header}\n{indent}{message}\n{footer}'.format(
            header=header, indent=''.join([' '] * (indent + 1)),
            message=message, footer=''.join(['-'] * len(header))
        )
        super(BIDSError, self).__init__(self.msg)
        self.bids_root = bids_root


class BIDSWarning(RuntimeWarning):
    pass


def collect_participants(bids_dir, participant_label=None, strict=False):
    """
    List the participants under the BIDS root and checks that participants
    designated with the participant_label argument exist in that folder.
    Returns the list of participants to be finally processed.
    Requesting all subjects in a BIDS directory root:
    """
    bids_dir = Path(bids_dir).resolve()
    all_participants = sorted([str(subdir.name)[4:] for subdir in bids_dir.glob('sub-*')
                               if subdir.is_dir()])

    # Error: bids_dir does not contain subjects
    if not all_participants:
        raise BIDSError(
            'Could not find participants. Please make sure the BIDS data '
            'structure is present and correct. Datasets can be validated online '
            'using the BIDS Validator (http://bids-standard.github.io/bids-validator/).\n'
            'If you are using Docker for Mac or Docker for Windows, you '
            'may need to adjust your "File sharing" preferences.', bids_dir)

    # No --participant-label was set, return all
    if not participant_label:
        return all_participants

    if isinstance(participant_label, str):
        participant_label = [participant_label]

    # Drop sub- prefixes
    participant_label = [sub[4:] if sub.startswith('sub-') else sub for sub in participant_label]
    # Remove duplicates
    participant_label = sorted(set(participant_label))
    # Remove labels not found
    found_label = sorted(set(participant_label) & set(all_participants))
    if not found_label:
        raise BIDSError('Could not find participants [{}]'.format(
            ', '.join(participant_label)), bids_dir)

    # Warn if some IDs were not found
    notfound_label = sorted(set(participant_label) - set(all_participants))
    if notfound_label:
        exc = BIDSError('Some participants were not found: {}'.format(
            ', '.join(notfound_label)), bids_dir)
        if strict:
            raise exc
        warnings.warn(exc.msg, BIDSWarning)

    return found_label


def get_metadata_for_nifti(in_file, logger=None):
    """Fetch metadata for a given nifti file

    """
    in_file = Path(in_file).absolute()
    fname = splitext(in_file)[0]
    fname_comps = fname.split("_")

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

    jsonext = '{}.json'.format
    bids_dir = in_file.parent.parent.parent  # go up 3 levels
    if any([comp.startswith('ses') for comp in fname_comps]):
        bids_dir = bids_dir.parent  # one more if multisession

    top_json = bids_dir / jsonext("_".join(top_comp_list))
    potential_json = [top_json]
    
    if logger:
    
        logger.info("ABOUT TO FAIL")
        logger.info("BIDS DIR: ", bids_dir)
        logger.info("SUBID: ", sub)
        logger.info("NIFTI: ", in_file)
        logger.info("JSON: ", subject_comp_list)

    subject_json = bids_dir / sub / jsonext("_".join(subject_comp_list))
    potential_json.append(subject_json)

    if ses:
        session_json = bids_dir / sub / ses / jsonext("_".join(session_comp_list))
        potential_json.append(session_json)

    potential_json.append(in_file.parent / jsonext(fname))  # Sidecar json

    merged_param_dict = {}
    for json_file_path in potential_json:
        if json_file_path.is_file():
            merged_param_dict.update(
                json.loads(json_file_path.read_text()))

    return merged_param_dict


def group_multiecho(bold_sess):
    """
    Multiplexes multi-echo EPIs into arrays. Dual-echo is a special
    case of multi-echo, which is treated as single-echo data.
    """
    from itertools import groupby

    def _grp_echos(x):
        if '_echo-' not in x:
            return x
        echo = re.search("_echo-\\d*", x).group(0)
        return x.replace(echo, "_echo-?")

    ses_uids = []
    for _, bold in groupby(bold_sess, key=_grp_echos):
        bold = list(bold)
        # If single- or dual-echo, flatten list; keep list otherwise.
        action = getattr(ses_uids, 'append' if len(bold) > 2 else 'extend')
        action(bold)
    return ses_uids
