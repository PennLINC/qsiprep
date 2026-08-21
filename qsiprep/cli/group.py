"""Preview how qsiprep would group and process a BIDS dataset's DWI scans.

Usage::

    qsiprep-group /path/to/bids [--participant-label 01 02] \\
        [--ignore-shims] [--separate-all-dwis] [--ignore-fieldmaps] \\
        [--backend fsl tortoise mixed]

Prints, per subject, the grouping decisions (with curated/inferred
provenance) and a plain-language preview of what each processing backend
would do with the data. Nothing is processed and nothing is written.
"""

import argparse
import sys

from qsiprep.grouping import build_dwi_grouping, describe_processing, render_html, report_text
from qsiprep.grouping.validation import BACKENDS


def _build_parser():
    parser = argparse.ArgumentParser(
        prog='qsiprep-group',
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
        '--ignore-fov',
        action='store_true',
        help=(
            'Concatenate series with differently-oriented fields of view anyway '
            '(distortion corrections will be misapplied). Grid-size mismatches '
            'still error.'
        ),
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
    parser.add_argument(
        '--force-t2wreg',
        action='store_true',
        help='Override all fieldmaps with T2w-registration SDC (TORTOISE T2Wreg).',
    )
    parser.add_argument(
        '--distortion-group-merge',
        choices=['concat', 'average', 'none'],
        default='concat',
        help="How the corrected results of an output's correction units are "
        'combined: concatenated (default), averaged (opposite-PE duplicate '
        'schemes), or kept as separate per-unit outputs.',
    )
    parser.add_argument(
        '--use-synb0',
        action='store_true',
        help='Give fieldmap-less series a SyNb0 synthetic-b=0 estimation from the T1w.',
    )
    parser.add_argument(
        '--html',
        metavar='PATH',
        help='Also write a self-contained explanatory HTML page for the grouping. '
        'With more than one subject, the subject label is inserted before the '
        'extension.',
    )
    return parser


def _per_subject_path(path: str, subject: str, multi: bool) -> str:
    """Insert ``sub-<label>`` before the extension when writing many subjects."""
    if not multi:
        return path
    base, dot, ext = path.rpartition('.')
    stem = base if dot else path
    suffix = f'.{ext}' if dot else ''
    return f'{stem}_sub-{subject}{suffix}'


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
            ignore_fov=args.ignore_fov,
            force_t2wreg=args.force_t2wreg,
            use_synb0=args.use_synb0,
            distortion_group_merge=args.distortion_group_merge,
            strict=False,
        )
        print(report_text(grouping))
        for backend in args.backend:
            print(describe_processing(grouping, backend))
        multi = len(subjects) > 1
        if args.html:
            path = _per_subject_path(args.html, subject, multi)
            with open(path, 'w') as fobj:
                fobj.write(render_html(grouping, backend=args.backend[0]))
            print(f'sub-{subject}: wrote {path}')
        if grouping.errors:
            exit_code = 1

    return exit_code


if __name__ == '__main__':
    sys.exit(main())
