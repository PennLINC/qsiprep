"""Parsing and validation for QSIPrep's ``--output-spaces`` argument.

The grammar is::

    token   ::= space (":" key "-" value)*
    space   ::= "acpc" | <TemplateFlow template name>
    key     ::= "res" | "cohort"

``res-`` values come in three families:

* ``nativemin`` / ``nativemax`` -- the min/max zoom across the subject's DWI runs,
  made isotropic. Legal on ``acpc`` only, and resolved at node run time.
* ``<size>mm`` -- a physical size. ``p`` is the decimal point and ``x`` separates
  axes, following https://github.com/nipreps/niworkflows/issues/997.
* a bare label -- a TemplateFlow ``res`` entity, as in fMRIPrep. Standard spaces only.

Nothing here touches nipype or a BIDS layout, so it is all directly testable.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace

from qsiprep.utils.bids import COHORT_KEY

ACPC = 'acpc'
VALID_KEYS = ('res', 'cohort')

_MM_RE = re.compile(r'^(?P<sizes>[0-9p]+(?:x[0-9p]+){0,2})mm$')
_NATIVE_RE = re.compile(r'^native(?P<strategy>min|max)$')


class OutputSpacesError(ValueError):
    """An ``--output-spaces`` token could not be parsed or validated."""


@dataclass(frozen=True)
class Resolution:
    """A resolved or deferred output resolution.

    ``kind`` is ``'mm'`` (an explicit physical size), ``'label'`` (a TemplateFlow
    ``res`` entity) or ``'native'`` (deferred until the DWI headers are readable).
    ``label`` is always the spec exactly as the user wrote it, because that is what
    goes into filenames.
    """

    kind: str
    label: str
    zooms: tuple | None = None
    strategy: str | None = None


@dataclass(frozen=True)
class SpaceSpec:
    """One requested output space."""

    space: str
    resolution: Resolution | None = None
    cohort: str | None = None

    @property
    def standard(self) -> bool:
        return self.space != ACPC

    @property
    def fullname(self) -> str:
        """NiPreps-style name with the cohort folded in, for ``from-``/``to-`` labels."""
        if self.cohort in (None, 'auto'):
            return self.space
        return f'{self.space}+{self.cohort}'

    @property
    def needs_cohort_resolution(self) -> bool:
        return self.cohort == 'auto'

    @property
    def needs_native_resolution(self) -> bool:
        return self.resolution is not None and self.resolution.kind == 'native'

    def with_cohort(self, cohort: str) -> 'SpaceSpec':
        """Return a copy with ``cohort-auto`` replaced by a concrete cohort."""
        return replace(self, cohort=str(cohort))

    def __str__(self) -> str:
        parts = [self.space]
        if self.cohort is not None:
            parts.append(f'cohort-{self.cohort}')
        if self.resolution is not None:
            parts.append(f'res-{self.resolution.label}')
        return ':'.join(parts)


def _templates() -> list:
    from templateflow.conf import TF_LAYOUT

    return TF_LAYOUT.get_templates()


def _cohorts(template: str) -> list:
    from templateflow.conf import TF_LAYOUT

    return [str(c) for c in TF_LAYOUT.get_cohorts(template=template)]


def _resolutions(template: str) -> list:
    from templateflow.conf import TF_LAYOUT

    return [str(r) for r in TF_LAYOUT.get_resolutions(template=template)]


def _parse_sizes(text: str) -> tuple:
    """Turn ``2``, ``1p5`` or ``6x6x3`` into a 3-tuple of millimetres."""
    values = []
    for chunk in text.split('x'):
        try:
            values.append(float(chunk.replace('p', '.')))
        except ValueError:
            raise OutputSpacesError(f'Could not read "{chunk}" as a voxel size.') from None
    if len(values) == 1:
        values = values * 3
    if len(values) != 3:
        raise OutputSpacesError(
            f'A res- specification needs one or three sizes, got {len(values)} in "{text}".'
        )
    return tuple(values)


def _parse_resolution(value: str, space: str) -> Resolution:
    native = _NATIVE_RE.match(value)
    if native:
        if space != ACPC:
            raise OutputSpacesError(
                f'res-{value} is only valid on "acpc". "Native" means the input DWI grid, '
                f'which has no meaning in {space} space.'
            )
        return Resolution(kind='native', label=value, strategy=native.group('strategy'))

    mm = _MM_RE.match(value)
    if mm:
        zooms = _parse_sizes(mm.group('sizes'))
        if space == ACPC and len(set(zooms)) != 1:
            raise OutputSpacesError(
                f'acpc:res-{value} is anisotropic. QSIPrep writes isotropic DWI because '
                'reconstruction requires it -- use an isotropic size, res-nativemin or '
                'res-nativemax.'
            )
        return Resolution(kind='mm', label=value, zooms=zooms)

    if space == ACPC:
        raise OutputSpacesError(
            f'acpc:res-{value} is not a voxel size. Physical sizes need an "mm" suffix, '
            f'so write acpc:res-{value}mm.'
        )

    available = _resolutions(space)
    if value not in available:
        raise OutputSpacesError(
            f'{space} has no res-{value}. Available TemplateFlow resolutions are: '
            f'{", ".join(available) or "none"}. For a custom size, add an "mm" suffix '
            f'(for example res-{value}mm).'
        )
    return Resolution(kind='label', label=value)


def _validate_cohort(space: str, cohort: str | None) -> None:
    available = _cohorts(space)
    if cohort is None:
        if available:
            options = ', '.join(f'cohort-{c}' for c in available)
            raise OutputSpacesError(
                f'{space} is not fully defined: it needs a cohort. Use cohort-auto to pick '
                f"one from the participant's age, or one of: {options}."
            )
        return

    if not available:
        raise OutputSpacesError(f'{space} does not accept a cohort specification.')

    if cohort == 'auto':
        if space not in COHORT_KEY:
            options = ', '.join(f'cohort-{c}' for c in available)
            raise OutputSpacesError(
                f'cohort-auto is not supported for {space}, which has no age-to-cohort '
                f'table. Specify one explicitly: {options}.'
            )
        return

    if cohort not in available:
        options = ', '.join(f'cohort-{c}' for c in available)
        raise OutputSpacesError(
            f'{space} has no cohort-{cohort}. Available cohorts are: {options}.'
        )


def parse_space_token(token: str) -> list:
    """Parse one ``--output-spaces`` token into one or more :class:`SpaceSpec`.

    A token carrying several ``res-`` specs expands to one spec each, following
    niworkflows#997: ``MNI152NLin2009cAsym:res-1:res-3mm`` yields two.
    """
    space, *rest = token.split(':')

    if space != ACPC and space not in _templates():
        raise OutputSpacesError(
            f'"{space}" is not a known output space. Use "acpc" or a TemplateFlow '
            f'template name.'
        )

    cohort = None
    res_values = []
    for item in rest:
        key, _, value = item.partition('-')
        if not value:
            raise OutputSpacesError(f'"{item}" in "{token}" is not a key-value specification.')
        if key not in VALID_KEYS:
            raise OutputSpacesError(
                f'"{key}" in "{token}" is not a valid specifier. '
                f'QSIPrep accepts: {", ".join(VALID_KEYS)}.'
            )
        if key == 'cohort':
            if cohort is not None:
                raise OutputSpacesError(f'"{token}" specifies more than one cohort.')
            cohort = value
        else:
            res_values.append(value)

    if space == ACPC:
        if cohort is not None:
            raise OutputSpacesError('"acpc" does not accept a cohort specification.')
        if not res_values:
            raise OutputSpacesError(
                'Every "acpc" output space needs a resolution, for example '
                'acpc:res-2mm or acpc:res-nativemax.'
            )
    else:
        _validate_cohort(space, cohort)

    if not res_values:
        return [SpaceSpec(space=space, resolution=None, cohort=cohort)]

    return [
        SpaceSpec(space=space, resolution=_parse_resolution(value, space), cohort=cohort)
        for value in res_values
    ]


def parse_output_spaces(tokens: Sequence) -> list:
    """Parse every token, de-duplicate, and check the whole request makes sense."""
    specs = []
    seen = set()
    for token in tokens:
        for spec in parse_space_token(token):
            key = str(spec)
            if key not in seen:
                seen.add(key)
                specs.append(spec)

    if not any(spec.space == ACPC for spec in specs):
        raise OutputSpacesError(
            '--output-spaces must include at least one "acpc" space, because QSIPrep '
            'writes preprocessed DWI in ACPC space only. For example: acpc:res-2mm.'
        )

    return specs
