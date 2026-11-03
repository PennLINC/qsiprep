"""Parsing and validation for QSIPrep's ``--output-spaces`` argument.

The grammar is::

    token   ::= space (":" key "-" value)*
    space   ::= "acpc" | "anat" | "dwiref" | <TemplateFlow template name>
    key     ::= "res" | "cohort"

``acpc``, ``anat`` and ``dwiref`` are the DWI spaces: the preprocessed DWI is
resampled into them. Template spaces get transforms and anatomical derivatives only.

``res-`` values come in four families:

* ``nativemin`` / ``nativemax`` -- the min/max zoom of the runs that make up each
  output, made isotropic. DWI spaces only; ``nativemin`` is the default.
* ``native`` -- the voxel size of the output's runs, per axis, which may be anisotropic.
* ``<size>mm`` -- a physical size. ``p`` is the decimal point and ``x`` separates
  axes, following https://github.com/nipreps/niworkflows/issues/997.
* a bare label -- a TemplateFlow ``res`` entity, as in fMRIPrep. Template spaces only.

Native resolutions are resolved per output when the workflow is built, by
:func:`resolve_native_zooms`.

No BIDSLayout is ever constructed here and nothing needs a BIDS tree on disk, so
every function below is directly unit-testable. (Importing this module does pull
pybids in transitively, via ``qsiprep.utils.__init__``.)
"""

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace

from qsiprep.utils.bids import COHORT_KEY

ACPC = 'acpc'
ANAT = 'anat'
DWIREF = 'dwiref'
#: DWI spaces in the subject's own frames, beside AC-PC.
NATIVE_FRAMES = (ANAT, DWIREF)
#: The spaces the preprocessed DWI is resampled into.
DWI_SPACES = (ACPC, *NATIVE_FRAMES)
#: The resolution a DWI space gets when it names none.
DEFAULT_DWI_RESOLUTION = 'nativemin'
#: What --output-spaces means when it is not given.
DEFAULT_OUTPUT_SPACES = (f'{ACPC}:res-{DEFAULT_DWI_RESOLUTION}',)
VALID_KEYS = ('res', 'cohort')

_MM_RE = re.compile(r'^(?P<sizes>[0-9p]+(?:x[0-9p]+){0,2})mm$')
_NATIVE_RE = re.compile(r'^native(?P<strategy>min|max)?$')


class OutputSpacesError(ValueError):
    """An ``--output-spaces`` token could not be parsed or validated."""


@dataclass(frozen=True)
class Resolution:
    """A resolved or deferred output resolution.

    ``kind`` is ``'mm'`` (an explicit physical size), ``'label'`` (a TemplateFlow
    ``res`` entity) or ``'native'`` (taken from each output's DWI runs). A native
    resolution's ``strategy`` is ``'min'`` or ``'max'`` (made isotropic), or None for
    the runs' own voxel size per axis. ``label`` is always the spec exactly as the user
    wrote it, because that is what goes into filenames.
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
    #: The cohort came from ``cohort-auto``.
    cohort_from_age: bool = False

    @property
    def is_template(self) -> bool:
        """Whether this is a TemplateFlow space, which gets no DWI."""
        return self.space not in DWI_SPACES

    @property
    def writes_dwi(self) -> bool:
        """Whether the preprocessed DWI is resampled into this space."""
        return self.space in DWI_SPACES

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

    @property
    def age_matched(self) -> bool:
        """Whether the cohort is, or will be, the one matched to the participant's age."""
        return self.cohort == 'auto' or self.cohort_from_age

    def with_cohort(self, cohort: str) -> 'SpaceSpec':
        """Return a copy with ``cohort-auto`` replaced by a concrete cohort."""
        return replace(self, cohort=str(cohort), cohort_from_age=True)

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
        if space not in DWI_SPACES:
            raise OutputSpacesError(
                f'res-{value} is only valid on a DWI space ({", ".join(DWI_SPACES)}). '
                f'"Native" means the input DWI grid, which has no meaning in {space} space.'
            )
        return Resolution(kind='native', label=value, strategy=native.group('strategy'))

    mm = _MM_RE.match(value)
    if mm:
        return Resolution(kind='mm', label=value, zooms=_parse_sizes(mm.group('sizes')))

    if space in DWI_SPACES:
        raise OutputSpacesError(
            f'{space}:res-{value} is not a voxel size. Physical sizes need an "mm" suffix, '
            f'so write {space}:res-{value}mm.'
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


def parse_space_token(token: str) -> list[SpaceSpec]:
    """Parse one ``--output-spaces`` token into one or more :class:`SpaceSpec`.

    A token carrying several ``res-`` specs expands to one spec each, following
    niworkflows#997: ``MNI152NLin2009cAsym:res-1:res-3mm`` yields two.
    """
    space, *rest = token.split(':')

    if space not in DWI_SPACES and space not in _templates():
        names = ', '.join(f'"{name}"' for name in DWI_SPACES)
        raise OutputSpacesError(
            f'"{space}" is not a known output space. Use {names} or a TemplateFlow template name.'
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

    if space in DWI_SPACES:
        if cohort is not None:
            raise OutputSpacesError(f'"{space}" does not accept a cohort specification.')

        if not res_values:
            res_values = [DEFAULT_DWI_RESOLUTION]
    else:
        _validate_cohort(space, cohort)

    if not res_values:
        return [SpaceSpec(space=space, resolution=None, cohort=cohort)]

    return [
        SpaceSpec(space=space, resolution=_parse_resolution(value, space), cohort=cohort)
        for value in res_values
    ]


def templateflow_kwargs(spec: SpaceSpec) -> dict:
    """Turn a template-space :class:`SpaceSpec` into ``GetTemplate`` inputs.

    A ``res-<n>mm`` spec has no TemplateFlow label, so the highest-resolution grid
    (``res-1``) is fetched and resampled downstream.
    """
    if not spec.is_template:
        raise OutputSpacesError(f'{spec.space} is not a template space.')

    kwargs = {'template_name': spec.space}
    if spec.cohort not in (None, 'auto'):
        kwargs['cohort'] = spec.cohort
    if spec.resolution is not None and spec.resolution.kind == 'label':
        kwargs['resolution'] = spec.resolution.label
    return kwargs


def dedupe_specs(specs) -> list[SpaceSpec]:
    """Drop repeated specs, keeping the first occurrence of each."""
    seen = set()
    unique = []
    for spec in specs:
        key = str(spec)
        if key not in seen:
            seen.add(key)
            unique.append(spec)
    return unique


def parse_output_spaces(tokens: Sequence) -> list[SpaceSpec]:
    """Parse every token, de-duplicate, and check the whole request makes sense."""
    specs = dedupe_specs(spec for token in tokens for spec in parse_space_token(token))

    if not any(spec.writes_dwi for spec in specs):
        raise OutputSpacesError(
            f'--output-spaces must include at least one DWI space ({", ".join(DWI_SPACES)}), '
            'because template spaces get transforms and anatomical derivatives only. '
            'For example: acpc:res-2mm.'
        )

    return specs


def dwi_specs_of(specs) -> list[SpaceSpec]:
    """Return the specs the preprocessed DWI is resampled into, in request order."""
    return [spec for spec in specs if spec.writes_dwi]


def bids_space(spec, dwiref_level='distortiongroup') -> str:
    """Return the ``space-`` entity value for a DWI spec.

    ``acpc`` is written ``ACPC``. ``dwiref`` is named by the level its frame was built
    at (``distortiongroup`` or ``subject``), so the DWI lands beside the
    ``space-<level>_dwiref`` reference image it was resampled into.
    """
    if spec.space == DWIREF:
        return dwiref_level
    return {ACPC: 'ACPC'}.get(spec.space, spec.space)


def resolution_entities(resolution) -> dict:
    """Return the ``res-`` entity of the outputs on a DWI grid at ``resolution``.

    Every output on a DWI grid carries its resolution, as in fMRIPrep, except
    ``res-native``: the runs' own voxel size goes unlabeled.
    """
    if resolution is None or (resolution.kind == 'native' and resolution.strategy is None):
        return {}
    return {'res': resolution.label}


def res_entities_for(spec) -> dict:
    """Return the ``res-`` entity of a DWI spec's outputs (see :func:`resolution_entities`)."""
    return resolution_entities(spec.resolution)


def spec_node_suffix(spec, dwi_specs) -> str:
    """Return the node-name suffix that keeps one DWI spec's nodes apart from another's.

    A single spec gets none, and ACPC-only runs keep ``_res<label>``, so existing
    working directories still match. With other spaces the space is named too.
    """
    if len(dwi_specs) < 2:
        return ''
    if all(other.space == ACPC for other in dwi_specs):
        return f'_res{spec.resolution.label}'
    return f'_{spec.space}_res{spec.resolution.label}'


def canonical_zooms(path) -> tuple:
    """Return an image's voxel size along the world (RAS) axes, rounded to 3 decimals.

    The zooms are read after ``nb.as_closest_canonical``, so a sagittal or oblique run
    is compared, and gridded, axis for axis with an axial one. The rounding follows
    niworkflows' ``GenerateSamplingReference``.
    """
    import nibabel as nb

    img = nb.as_closest_canonical(nb.load(path))
    return tuple(round(float(zoom), 3) for zoom in img.header.get_zooms()[:3])


class NativeResolutionMismatch(OutputSpacesError):
    """The runs of one output do not share a voxel size, so ``res-native*`` is ambiguous."""

    def __init__(self, zooms_by_file):
        self.zooms_by_file = dict(zooms_by_file)
        super().__init__(f'The runs have different voxel sizes ({self.listing}).')

    @property
    def listing(self) -> str:
        """Every run with its voxel size."""
        return ', '.join(
            f'{os.path.basename(path)}: {format_zooms(zooms)}'
            for path, zooms in self.zooms_by_file.items()
        )

    @property
    def smallest(self) -> tuple:
        """The smallest voxel size along each axis, across the runs."""
        return tuple(min(axis) for axis in zip(*self.zooms_by_file.values(), strict=True))


def resolve_native_zooms(dwi_files) -> tuple:
    """Return the voxel size ``dwi_files`` share, on world-aligned axes.

    Raises
    ------
    NativeResolutionMismatch
        If the runs do not all have the same voxel size.
    """
    zooms = {path: canonical_zooms(path) for path in dwi_files}
    if len(set(zooms.values())) > 1:
        raise NativeResolutionMismatch(zooms)
    return next(iter(zooms.values()))


def resolved_zooms(resolution, native_zooms=None) -> tuple:
    """Return the voxel size a DWI resolution means for one output.

    ``native_zooms`` is the voxel size the output's runs share, from
    :func:`resolve_native_zooms`. Only native resolutions read it.
    """
    if resolution.kind == 'mm':
        return tuple(resolution.zooms)

    if resolution.kind != 'native':
        raise OutputSpacesError(f'res-{resolution.label} is not a DWI resolution.')

    if native_zooms is None:
        raise OutputSpacesError(f'res-{resolution.label} needs the voxel size of the DWI runs.')

    if resolution.strategy == 'min':
        return (min(native_zooms),) * 3

    if resolution.strategy == 'max':
        return (max(native_zooms),) * 3

    return tuple(native_zooms)


def zooms_label(zooms) -> str:
    """Spell a voxel size the way ``res-`` values do: ``2mm``, ``1p5x1p5x3mm``."""

    def _number(value):
        return f'{float(value):g}'.replace('.', 'p')

    if len(set(zooms)) == 1:
        return f'{_number(zooms[0])}mm'
    return 'x'.join(_number(value) for value in zooms) + 'mm'


def format_zooms(zooms) -> str:
    """Describe a voxel size for people: ``2 mm isotropic``, ``1.5 x 1.5 x 3 mm``."""
    if len(set(zooms)) == 1:
        return f'{float(zooms[0]):g} mm isotropic'
    return ' x '.join(f'{float(value):g}' for value in zooms) + ' mm'


DEFAULT_ANCHOR = 'MNI152NLin2009cAsym'
#: Templates that make ACPC alignment infant-appropriate, most specific first.
INFANT_ANCHORS = ('MNIInfant', 'UNCInfant')


def _age_in_months(bids_dir, subject_id, session_id):
    """Indirection so tests can substitute an age without a BIDS tree."""
    from qsiprep.utils.bids import parse_bids_for_age_months

    return parse_bids_for_age_months(bids_dir, subject_id, session_id)


def select_acpc_anchor(specs) -> SpaceSpec:
    """Pick the template that anchors ACPC alignment and the output grid.

    An infant template among the requested spaces anchors ACPC, otherwise
    MNI152NLin2009cAsym does. Among several cohorts of that template,
    ``cohort-auto`` wins, then the first listed.
    """
    for name in INFANT_ANCHORS:
        candidates = [spec for spec in specs if spec.space == name]
        if candidates:
            age_matched = [spec for spec in candidates if spec.age_matched]
            chosen = (age_matched or candidates)[0]
            # The anchor sets the ACPC grid, so it uses the template's default resolution
            return replace(chosen, resolution=None)
    return SpaceSpec(space=DEFAULT_ANCHOR)


def describe_acpc_anchor(specs, anchor) -> str:
    """Describe the ACPC anchor and any requested cohorts it was chosen over."""
    unused = []
    for spec in specs:
        if spec.space == anchor.space and spec.cohort not in (None, anchor.cohort):
            if spec.cohort not in unused:
                unused.append(spec.cohort)
    text = f'AC-PC alignment is anchored to {anchor.fullname}'
    if anchor.cohort is not None and (unused or anchor.cohort_from_age):
        if anchor.cohort_from_age:
            text += ", the cohort chosen from the participant's age"
        else:
            text += ', the first cohort listed in --output-spaces'
    if unused:
        many = len(unused) > 1
        text += (
            f'; {anchor.space} cohort{"s" if many else ""} {", ".join(unused)} '
            f'{"were" if many else "was"} also requested and {"get" if many else "gets"} '
            'derivatives only'
        )
    return text + '.'


def resolve_output_spaces(specs, bids_dir, subject_id, session_id) -> list:
    """Replace every ``cohort-auto`` with a cohort chosen from the participant's age.

    The age is read at most once, however many templates asked for it. The result
    is de-duplicated, since ``cohort-auto`` may resolve to a cohort also listed.
    """
    from qsiprep.utils.bids import cohort_by_months

    if not any(spec.needs_cohort_resolution for spec in specs):
        return dedupe_specs(specs)

    months = _age_in_months(bids_dir, subject_id, session_id)
    if months is None:
        wanted = ', '.join(spec.space for spec in specs if spec.needs_cohort_resolution)
        ses_str = f'_ses-{session_id}' if session_id else ''
        raise OutputSpacesError(
            f'Could not find an age for sub-{subject_id}{ses_str}, which is needed to '
            f'choose a cohort for: {wanted}. Specify the cohort explicitly, for example '
            f'MNIInfant:cohort-3.'
        )

    resolved = []
    for spec in specs:
        if not spec.needs_cohort_resolution:
            resolved.append(spec)
            continue
        try:
            cohort = cohort_by_months(spec.space, months)
        except KeyError as exc:
            raise OutputSpacesError(
                f'Could not choose a {spec.space} cohort for an age of {months} months: {exc}'
            ) from None
        resolved.append(spec.with_cohort(str(cohort)))
    return dedupe_specs(resolved)
