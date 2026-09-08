#!/usr/bin/env python3
"""Report what TORTOISE's Siemens coefficient reader would do with a .grad file.

    python3 check_grad.py /path/to/CimaX_coeff.grad
    python3 check_grad.py /path/to/in.grad --write-clean /path/to/out.grad

QSIPrep sanitizes coefficient files itself (see qsiprep.utils.gradcal), so this
is a pre-flight check for a file you have not run yet, or a way to see exactly
which lines the reader objects to. ``--write-clean`` writes a comment-free copy;
the original is never modified.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qsiprep.utils.gradcal import copy_without_comments, describe  # noqa: E402


def report(path):
    lines, findings = describe(path)
    crlf = sum(1 for line in lines if line.endswith('\r'))
    print(f'{path}: {len(lines)} lines, {crlf} with CRLF endings')

    aborts = 0
    for number, line, status, detail, commented in findings:
        kind = 'comment' if commented else 'DATA'
        if status == 'abort':
            aborts += 1
            print(f'  line {number}: ABORTS the tool -- {detail}  [{kind} line]')
            print(f'            {line!r}')
        elif detail is None:
            print(f'  line {number}: parses but has no x/y/z axis -> term DROPPED: {line!r}')
        elif commented:
            print(f'  line {number}: COMMENT silently parses as a real {detail} term: {line!r}')

    terms = sum(1 for _, _, status, _, _ in findings if status == 'term')
    print(f'\n{terms} coefficient lines parse, {aborts} would abort the tool.')
    if aborts:
        fatal_data = [f for f in findings if f[2] == 'abort' and not f[4]]
        if fatal_data:
            print('At least one is a DATA line: QSIPrep will refuse this file at startup.')
        else:
            print('All are comment lines: QSIPrep drops these automatically.')
    return aborts


def main(argv):
    if len(argv) == 2:
        report(argv[1])
    elif len(argv) == 4 and argv[2] == '--write-clean':
        report(argv[1])
        dropped = copy_without_comments(argv[1], argv[3])
        print(f'\nWrote {argv[3]} ({dropped} comment lines dropped). Re-checking:\n')
        report(argv[3])
    else:
        sys.exit(__doc__)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
