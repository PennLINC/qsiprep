"""Make a Siemens ``.grad`` coefficient file safe for TORTOISE's reader.

``GRADCAL::read_Siemens_format`` (TORTOISEV4
``src/tools/gradnonlin/gradcal.cxx:99-183``) parses every line holding a ``(``
at index 3-9 together with a ``,`` and a ``)``. It has no comment handling and
no ``try``/``catch``, so a header line documenting the normalization constants
in the same notation the data uses -- real Siemens files carry these --

    #  A(1,1) = 1.1547 (2/Sqrt[3])

is read as a coefficient. ``l`` and ``m`` parse as integers, the coefficient
substring comes out as ``" = 1.1547 (2/Sqrt[3])"``, ``std::stof`` throws
``std::invalid_argument``, and the tool dies with ``Aborted (core dumped)``
partway through a run.

A comment that *does* parse is worse than one that aborts: it is folded into
the expansion silently, so a commented-out term changes the field with nothing
in the log.

:func:`sanitize_siemens_coefficients` drops comment lines into a copy and then
checks what remains against the reader's own arithmetic, so a genuinely
malformed data line fails at startup with the line quoted rather than as a core
dump 20 minutes in. The user's file is never modified.
"""

from pathlib import Path

#: Extensions ``read_grad_file`` dispatches to ``read_Siemens_format``. The GE
#: (``.dat``) and TORTOISE (``.gc``) readers are separate functions with
#: different parsing, and ``.nii``/``.nii.gz`` is a ready-made displacement
#: field that is never parsed at all.
SIEMENS_EXTENSION = '.grad'

_WHITESPACE = ' \t\n\r\f\v'


def _stof(text):
    """``std::stof``: read a leading float, ignore trailing junk, throw if none.

    Only whether this raises is meaningful here; the returned value is not used
    and exponent suffixes are deliberately not consumed, since a string with a
    leading number never throws either way.
    """
    stripped = text.lstrip(_WHITESPACE)
    index = 0
    if index < len(stripped) and stripped[index] in '+-':
        index += 1
    digits = 0
    while index < len(stripped) and (stripped[index].isdigit() or stripped[index] == '.'):
        digits += stripped[index].isdigit()
        index += 1
    if not digits:
        raise ValueError('stof')
    return float(stripped[:index] or 0)


def _stoi(text):
    """``std::stoi``: read a leading integer, ignore trailing junk, throw if none."""
    stripped = text.lstrip(_WHITESPACE)
    index = 0
    if index < len(stripped) and stripped[index] in '+-':
        index += 1
    digits = 0
    while index < len(stripped) and stripped[index].isdigit():
        index += 1
        digits += 1
    if not digits:
        raise ValueError('stoi')
    return int(stripped[:index])


def _substr(text, pos, count):
    """``std::string::substr``, including the ``size_t`` wrap on a negative count.

    A line ending at ``)`` makes the reader's ``size() - posA3 - 2`` underflow;
    the resulting enormous count simply clamps to the end of the string, which
    is how ``stof`` ends up being handed ``""``.
    """
    if pos > len(text):
        raise IndexError('out_of_range')
    return text[pos:] if count < 0 else text[pos : pos + count]


def is_comment(line):
    """Whether the line is a comment. The reader itself does not know about these."""
    return line.lstrip().startswith('#')


def reader_verdict(line):
    """What ``read_Siemens_format`` would do with one line.

    Returns ``(status, detail)`` where status is one of:

    ``'skip'``
        The line does not look like a coefficient; the reader ignores it.
    ``'term'``
        The reader reads a coefficient off it. ``detail`` is the axis letter,
        or ``None`` when no ``x``/``y``/``z`` appears anywhere on the line and
        the term is therefore dropped without warning.
    ``'abort'``
        The reader throws and the process dies. ``detail`` says which call and
        on what text.
    """
    # find_first_of("(", 3, 3) -- the first "(" at index >= 3. The count of 3
    # reads past the one-character literal, but no std::string content matches
    # the NUL bytes it picks up, so this is just "find from index 3".
    pos_open = line.find('(', 3)
    if pos_open == -1 or pos_open >= 10:
        return 'skip', None
    # Both searched from index 0, not from the "(" -- an earlier comma or
    # parenthesis on the line is what the reader measures against.
    pos_comma = line.find(',')
    pos_close = line.find(')')
    if pos_comma == -1 or pos_close == -1:
        return 'skip', None

    degree = _substr(line, pos_open + 1, pos_comma - pos_open - 1)
    order = _substr(line, pos_comma + 1, pos_close - pos_comma - 1)
    try:
        _stoi(degree)
        _stoi(order)
    except ValueError:
        return 'abort', f'std::stoi on {degree!r}/{order!r}'

    # Everything after ")" except the final character, which is assumed to be
    # the axis letter.
    coefficient = _substr(line, pos_close + 1, len(line) - pos_close - 2)
    try:
        _stof(coefficient)
    except ValueError:
        return 'abort', f'std::stof on {coefficient!r}'

    return 'term', next((axis for axis in 'xyz' if axis in line), None)


def _lines(path):
    """Split like ``std::getline(f, s, '\\n')``, which keeps a trailing ``\\r``."""
    raw = Path(path).read_bytes().decode('latin-1')
    lines = raw.split('\n')
    if lines and lines[-1] == '':
        lines.pop()
    return lines


def sanitize_siemens_coefficients(gradient_file, dest_dir, logger=None):
    """Return a coefficient file TORTOISE's Siemens reader can survive.

    Comment lines are dropped into a copy in ``dest_dir``, keeping the original
    basename so provenance recorded as ``os.path.basename`` is unchanged. Files
    the Siemens reader never sees (GE, TORTOISE and displacement-field inputs)
    and files with nothing to drop are returned untouched.

    Parameters
    ----------
    gradient_file : str or os.PathLike
        The user's ``--gradient-file``. Never modified.
    dest_dir : str or os.PathLike
        Directory to write the sanitized copy into (the work directory).
    logger : logging.Logger, optional
        Where to report what was dropped.

    Returns
    -------
    pathlib.Path
        The file to hand to the TORTOISE tools.

    Raises
    ------
    ValueError
        If a line that is *not* a comment would still abort the reader. Such a
        file cannot be corrected by dropping comments, and letting the run start
        would only move the failure into a node hours later.
    """
    gradient_file = Path(gradient_file)
    if gradient_file.suffix != SIEMENS_EXTENSION:
        return gradient_file

    lines = _lines(gradient_file)
    # Numbered against the original file throughout, so anything reported back
    # points at a line the user can actually go and look at.
    verdicts = [
        (number, line, is_comment(line)) + reader_verdict(line)
        for number, line in enumerate(lines, start=1)
    ]

    fatal = [v for v in verdicts if v[3] == 'abort' and not v[2]]
    if fatal:
        number, line, _, _, detail = fatal[0]
        raise ValueError(
            f'{gradient_file} cannot be read by TORTOISE: line {number} makes its '
            f'Siemens coefficient reader throw ({detail}), which aborts the tool '
            f'mid-run. Offending line: {line!r}. QSIPrep drops comment lines '
            'automatically, but this one carries data, so it has to be corrected '
            'in the file itself.'
        )

    # Only comments the reader would actually act on matter; an ordinary prose
    # comment is skipped by the reader and costs nothing to leave in place.
    dropped = [v for v in verdicts if v[2] and v[3] != 'skip']
    if not dropped:
        return gradient_file

    kept = [line for line in lines if not is_comment(line)]
    out_file = Path(dest_dir) / gradient_file.name
    # Written back as latin-1 so a copy is byte-identical apart from the
    # removed lines; line endings are preserved because \r stays on the line.
    out_file.write_bytes('\n'.join(kept).encode('latin-1') + b'\n')

    if logger is not None:
        logger.warning(
            "Gradient coefficient file %s has %d comment line(s) that TORTOISE's "
            'reader would parse as coefficients, %d of them fatally. Using a '
            'sanitized copy at %s; the original is unchanged.',
            gradient_file,
            len(dropped),
            sum(1 for v in dropped if v[3] == 'abort'),
            out_file,
        )
        for number, line, _, status, _ in dropped:
            verb = 'would abort the tool' if status == 'abort' else 'would add a silent term'
            logger.warning('  dropped line %d (%s): %r', number, verb, line)

    return out_file


def describe(gradient_file):
    """Report, per line, what the reader would do. Used by ``scripts/check_grad.py``."""
    lines = _lines(gradient_file)
    report = []
    for number, line in enumerate(lines, start=1):
        status, detail = reader_verdict(line)
        if status != 'skip':
            report.append((number, line, status, detail, is_comment(line)))
    return lines, report


def copy_without_comments(gradient_file, out_file):
    """Write ``gradient_file`` to ``out_file`` with comment lines removed."""
    lines = _lines(gradient_file)
    kept = [line for line in lines if not is_comment(line)]
    Path(out_file).write_bytes('\n'.join(kept).encode('latin-1') + b'\n')
    return len(lines) - len(kept)


__all__ = [
    'copy_without_comments',
    'describe',
    'is_comment',
    'reader_verdict',
    'sanitize_siemens_coefficients',
]
