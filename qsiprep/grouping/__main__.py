"""Preview how qsiprep would group and process a BIDS dataset's DWI scans.

Usage::

    python -m qsiprep.grouping /path/to/bids [--participant-label 01 02] \\
        [--ignore-shims] [--separate-all-dwis] [--ignore-fieldmaps] \\
        [--backend fsl tortoise mixed]

Prints, per subject, the grouping decisions (with curated/inferred
provenance) and a plain-language preview of what each processing backend
would do with the data. Nothing is processed and nothing is written.
"""

import argparse
import sys

from . import build_dwi_grouping, describe_processing, report_text
from .validation import BACKENDS


def _build_parser():
    parser = argparse.ArgumentParser(
        prog='python -m qsiprep.grouping',
        description=__doc__.splitlines()[0],
    )
    parser.add_argument('bids_dir', help='Root of the BIDS dataset')
    parser.add_argument(
        '--participant-label',
        nargs='+',
        default=None,
        help='Subject label(s) to preview (without "sub-"). Default: all subjects.',
    )
    parser.add_argument(
        '--session-id',
        default=None,
        help='Restrict to one session label (without "ses-").',
    )
    parser.add_argument(
        '--backend',
        nargs='+',
        choices=BACKENDS,
        default=list(BACKENDS),
        help='Backend(s) to preview. Default: all three.',
    )
    parser.add_argument(
        '--ignore-shims',
        action='store_true',
        help='Treat all ShimSetting values as compatible.',
    )
    parser.add_argument(
        '--separate-all-dwis',
        action='store_true',
        help='Every DWI series becomes its own output.',
    )
    parser.add_argument(
        '--ignore-fieldmaps',
        action='store_true',
        help='Skip fmap/; only the reverse phase-encoding DWI heuristic applies.',
    )
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)

    from bids import BIDSLayout

    layout = BIDSLayout(args.bids_dir, validate=False)
    subjects = args.participant_label or layout.get_subjects()
    if not subjects:
        print(f'No subjects found in {args.bids_dir}', file=sys.stderr)
        return 1

    exit_code = 0
    for subject in subjects:
        query = {
            'subject': subject,
            'suffix': 'dwi',
            'extension': ['.nii', '.nii.gz'],
            'return_type': 'file',
        }
        if args.session_id:
            query['session'] = args.session_id
        subject_data = {'dwi': sorted(layout.get(**query))}
        if not subject_data['dwi']:
            print(f'sub-{subject}: no DWI files found, skipping.\n')
            continue

        grouping = build_dwi_grouping(
            layout,
            subject_data,
            separate_all_dwis=args.separate_all_dwis,
            ignore_fieldmaps=args.ignore_fieldmaps,
            ignore_shims=args.ignore_shims,
            strict=False,
        )
        print(report_text(grouping))
        for backend in args.backend:
            print(describe_processing(grouping, backend))
        if grouping.errors:
            exit_code = 1

    return exit_code


if __name__ == '__main__':
    sys.exit(main())
