"""Preview how qsiprep would group and process a BIDS dataset's DWI scans.

Usage::

    qsiprep-group /path/to/bids [--participant-label 01 02] \\
        [--ignore-shims] [--separate-all-dwis] [--ignore-fieldmaps] \\
        [--hmc-method eddy|shoreline|diffprep] [--sdc-method auto|topup|...]

Prints, per subject, the grouping decisions (with curated/inferred
provenance) and a plain-language preview of what the selected processing
methods would do with the data - or, with no method flags, every default
method combination. Nothing is processed and nothing is written.
"""

import argparse
import sys

from qsiprep.grouping import (
    build_dwi_grouping,
    describe_processing,
    render_html,
    report_text,
    selection_for_config,
)
from qsiprep.grouping.methods import SHORELINE_MODELS, canonical_selection
from qsiprep.grouping.report import default_preview_selections
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
        '--hmc-method',
        choices=['eddy', 'shoreline', 'diffprep'],
        default=None,
        help='Preview one head-motion-correction method instead of all defaults.',
    )
    parser.add_argument(
        '--shoreline-model',
        choices=list(SHORELINE_MODELS),
        default=None,
        help='SHORELine signal model (with --hmc-method shoreline).',
    )
    parser.add_argument(
        '--sdc-method',
        choices=['auto', 'topup', 'drbuddi', 'topup+drbuddi'],
        default='auto',
        help='PEPOLAR tool preference for the previewed method (default: auto).',
    )
    parser.add_argument(
        '--backend',
        nargs='+',
        choices=BACKENDS,
        default=None,
        help='DEPRECATED: use --hmc-method/--sdc-method. Legacy backend '
        'name(s) to preview.',
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


def _selections(args):
    """The method selections to preview, from the parsed arguments."""
    if args.shoreline_model and args.hmc_method != 'shoreline':
        raise SystemExit('--shoreline-model requires --hmc-method shoreline')
    if args.hmc_method:
        hmc = args.shoreline_model or args.hmc_method
        return [selection_for_config(hmc, args.sdc_method)]
    if args.backend:
        print(
            '--backend is deprecated; use --hmc-method/--sdc-method instead.',
            file=sys.stderr,
        )
        return [canonical_selection(backend) for backend in args.backend]
    return list(default_preview_selections())


def main(argv=None):
    args = _build_parser().parse_args(argv)
    selections = _selections(args)

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
        for selection in selections:
            print(describe_processing(grouping, selection))
        multi = len(subjects) > 1
        if args.html:
            path = _per_subject_path(args.html, subject, multi)
            with open(path, 'w') as fobj:
                fobj.write(render_html(grouping, selections=selections))
            print(f'sub-{subject}: wrote {path}')
        if grouping.errors:
            exit_code = 1

    return exit_code


if __name__ == '__main__':
    sys.exit(main())
