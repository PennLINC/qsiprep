# `--output-spaces` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `--output-resolution` and `--anatomical-template` with an fMRIPrep-style `--output-spaces` argument that supports multiple isotropic ACPC resolutions, data-derived resolutions (`res-nativemin`/`res-nativemax`), several standard spaces per run, and age-derived template cohorts (`cohort-auto`).

**Architecture:** A new QSIPrep-owned parser (`qsiprep/utils/spaces.py`) turns CLI tokens into `SpaceSpec` objects, validating template names, cohorts, and resolution labels against TemplateFlow at parse time. Two specs stay symbolic because they are unknowable at CLI time: `cohort-auto` resolves per subject at workflow-build time from the participant's age, and `res-native*` resolves at node run time from the DWI headers. The anatomical and DWI workflows then fan out over the resulting specs.

**Tech Stack:** Python 3.11, nipype, argparse, TemplateFlow (`templateflow.conf.TF_LAYOUT`), pytest, AFNI (`afni.Resample`), ANTs.

**Spec:** `docs/superpowers/specs/2026-08-26-output-spaces-design.md`

## Global Constraints

- Branch: `output-spaces`. Repo root: `/mnt/c/Users/tsalo/Documents/linc/qsiprep`.
- Run every command through micromamba: `micromamba run -n linc311 <command>`. Never `conda`, `pip venv`, or `pixi`. Do not install packages.
- QSIPrep's preferred orientation is **LPS+**. AFNI spells this `orientation='RAI'`. Every anatomical image QSIPrep writes is LPS+.
- **DWI is never resampled into a standard space.** Standard spaces produce transforms and anatomical derivatives only.
- **A single-`acpc` run must produce byte-identical derivative filenames to today.** This is what protects QSIRecon. Task 5 locks it down before any fan-out lands.
- Deprecated options are removed in `27.0.0`, matching the existing `deprecations` table in `qsiprep/cli/parser.py:57`.
- The `res-` label written into filenames is the spec **as written** (`2mm`, `1p5mm`, `nativemax`), never the resolved number.
- Existing code style: single quotes, `# fmt:skip` after `workflow.connect([...])` blocks, private input/output specs named `_XxxInputSpec`.

---

## File Structure

**Create:**
- `qsiprep/utils/spaces.py` — `SpaceSpec`, `Resolution`, the token parser, and all parse-time validation. No nipype, no BIDS layout; pure functions.
- `qsiprep/tests/test_utils_spaces.py` — grammar accept/reject table.
- `qsiprep/tests/test_output_spaces_naming.py` — derivative-naming regression guard.

**Modify:**
- `qsiprep/utils/bids.py` — hoist the cohort table to a module constant.
- `qsiprep/cli/parser.py` — the `--output-spaces` argument, its action, deprecation forwarding, post-parse enforcement.
- `qsiprep/config.py` — `workflow.output_spaces`; delete the vestigial `init_spaces()`/`workflow.spaces`.
- `qsiprep/interfaces/anatomical.py` — `GetTemplate` gains res/cohort; `VoxelSizeChooser` takes many images.
- `qsiprep/interfaces/images.py` — `ChooseInterpolator` reads the output grid.
- `qsiprep/workflows/base.py` — cohort resolution and anchor selection per subject.
- `qsiprep/workflows/anatomical/volume.py` — grid fan-out, LPS sub-workflow, normalization fan-out, derivatives, reports.
- `qsiprep/workflows/dwi/finalize.py` — DWI resampling fan-out and the `res-` entity rule.
- `qsiprep/workflows/dwi/resampling.py` — interpolator wiring and boilerplate.
- `qsiprep/data/NOTICE` — provenance for the bundled fieldmap atlas.
- `docs/*.rst`, `.circleci/*.sh`, `qsiprep/data/tests/config.toml`, and four test modules — migration.

---

## Task 1: The space grammar parser

**Files:**
- Create: `qsiprep/utils/spaces.py`
- Create: `qsiprep/tests/test_utils_spaces.py`
- Modify: `qsiprep/utils/bids.py:973-1032`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Resolution(kind: str, label: str, zooms: tuple[float, float, float] | None, strategy: str | None)` — `kind` is `'mm'`, `'label'`, or `'native'`; `strategy` is `'min'`/`'max'` when `kind == 'native'`, else `None`.
  - `SpaceSpec(space: str, resolution: Resolution | None, cohort: str | None)` with properties `.standard: bool`, `.fullname: str`, `.needs_cohort_resolution: bool`, `.needs_native_resolution: bool`, method `.with_cohort(cohort: str) -> SpaceSpec`, and `__str__` returning the canonical token.
  - `parse_space_token(token: str) -> list[SpaceSpec]`
  - `parse_output_spaces(tokens: Sequence[str]) -> list[SpaceSpec]`
  - `OutputSpacesError(ValueError)`
  - `qsiprep.utils.bids.COHORT_KEY: dict[str, tuple[int, ...]]`

- [ ] **Step 1: Hoist the cohort table so the parser and the resolver share one source of truth**

In `qsiprep/utils/bids.py`, move the `cohort_key` dict out of `cohort_by_months` and make it a module-level constant directly above that function:

```python
# Upper age bound in months for each cohort, in cohort order starting at 1.
COHORT_KEY = {
    'MNIInfant': (2, 5, 8, 11, 14, 17, 21, 27, 33, 44, 60),
    'UNCInfant': (8, 12, 24),
}
```

Then replace the body's lookup so the function reads from it:

```python
    ages = COHORT_KEY.get(template)
    if ages is None:
        raise KeyError('Template cohort information does not exist.')
```

Delete the now-unused local `cohort_key` dict. Leave the docstring and its Apache-2.0 attribution intact.

- [ ] **Step 2: Run the existing cohort tests to confirm the hoist changed nothing**

Run: `micromamba run -n linc311 pytest qsiprep/tests/test_utils_bids.py -v`
Expected: PASS, same count as before.

- [ ] **Step 3: Write the failing tests**

Create `qsiprep/tests/test_utils_spaces.py`:

```python
"""Tests for the --output-spaces grammar."""

import pytest

from qsiprep.utils.spaces import (
    OutputSpacesError,
    parse_output_spaces,
    parse_space_token,
)


def test_acpc_isotropic_mm():
    (spec,) = parse_space_token('acpc:res-2mm')
    assert spec.space == 'acpc'
    assert spec.standard is False
    assert spec.cohort is None
    assert spec.resolution.kind == 'mm'
    assert spec.resolution.zooms == (2.0, 2.0, 2.0)
    assert spec.resolution.label == '2mm'
    assert str(spec) == 'acpc:res-2mm'


def test_acpc_decimal_uses_p():
    (spec,) = parse_space_token('acpc:res-1p5mm')
    assert spec.resolution.zooms == (1.5, 1.5, 1.5)
    assert spec.resolution.label == '1p5mm'
    assert str(spec) == 'acpc:res-1p5mm'


@pytest.mark.parametrize('strategy', ['min', 'max'])
def test_acpc_native(strategy):
    (spec,) = parse_space_token(f'acpc:res-native{strategy}')
    assert spec.resolution.kind == 'native'
    assert spec.resolution.strategy == strategy
    assert spec.resolution.zooms is None
    assert spec.needs_native_resolution is True


def test_acpc_bare_number_is_rejected():
    with pytest.raises(OutputSpacesError, match='acpc:res-2mm'):
        parse_space_token('acpc:res-2')


def test_acpc_anisotropic_is_rejected():
    with pytest.raises(OutputSpacesError, match='isotropic'):
        parse_space_token('acpc:res-2x2x3mm')


def test_acpc_requires_a_resolution():
    with pytest.raises(OutputSpacesError, match='res-'):
        parse_space_token('acpc')


def test_acpc_rejects_cohort():
    with pytest.raises(OutputSpacesError, match='cohort'):
        parse_space_token('acpc:res-2mm:cohort-1')


def test_standard_space_bare():
    (spec,) = parse_space_token('MNI152NLin2009cAsym')
    assert spec.standard is True
    assert spec.resolution is None
    assert str(spec) == 'MNI152NLin2009cAsym'


def test_standard_space_templateflow_label():
    (spec,) = parse_space_token('MNI152NLin2009cAsym:res-2')
    assert spec.resolution.kind == 'label'
    assert spec.resolution.label == '2'


def test_standard_space_custom_mm():
    (spec,) = parse_space_token('MNI152NLin2009cAsym:res-1p5mm')
    assert spec.resolution.kind == 'mm'
    assert spec.resolution.zooms == (1.5, 1.5, 1.5)


def test_standard_space_anisotropic_allowed():
    (spec,) = parse_space_token('MNI152NLin2009cAsym:res-6x6x3mm')
    assert spec.resolution.zooms == (6.0, 6.0, 3.0)


def test_repeated_res_expands():
    specs = parse_space_token('MNI152NLin2009cAsym:res-1:res-3mm')
    assert [s.resolution.label for s in specs] == ['1', '3mm']


def test_unknown_resolution_label_is_rejected():
    with pytest.raises(OutputSpacesError, match='res-9'):
        parse_space_token('MNI152NLin2009cAsym:res-9')


def test_native_rejected_on_standard_space():
    with pytest.raises(OutputSpacesError, match='native'):
        parse_space_token('MNI152NLin2009cAsym:res-nativemax')


def test_unknown_space_is_rejected():
    with pytest.raises(OutputSpacesError, match='NotATemplate'):
        parse_space_token('NotATemplate')


def test_unknown_key_is_rejected():
    with pytest.raises(OutputSpacesError, match='den'):
        parse_space_token('MNI152NLin2009cAsym:den-32k')


def test_cohort_template_requires_a_cohort():
    with pytest.raises(OutputSpacesError, match='cohort'):
        parse_space_token('MNIInfant')


def test_cohort_template_accepts_a_label():
    (spec,) = parse_space_token('MNIInfant:cohort-3')
    assert spec.cohort == '3'
    assert spec.fullname == 'MNIInfant+3'
    assert spec.needs_cohort_resolution is False


def test_cohort_auto_is_deferred():
    (spec,) = parse_space_token('MNIInfant:cohort-auto')
    assert spec.cohort == 'auto'
    assert spec.needs_cohort_resolution is True
    assert spec.fullname == 'MNIInfant'
    assert str(spec) == 'MNIInfant:cohort-auto'


def test_cohort_auto_rejected_without_an_age_table():
    with pytest.raises(OutputSpacesError, match='cohort-1'):
        parse_space_token('MNIPediatricAsym:cohort-auto')


def test_invalid_cohort_is_rejected():
    with pytest.raises(OutputSpacesError, match='cohort'):
        parse_space_token('MNIInfant:cohort-99')


def test_with_cohort_replaces_auto():
    (spec,) = parse_space_token('MNIInfant:cohort-auto')
    resolved = spec.with_cohort('3')
    assert resolved.cohort == '3'
    assert resolved.fullname == 'MNIInfant+3'
    assert spec.cohort == 'auto'  # original untouched


def test_parse_output_spaces_requires_acpc():
    with pytest.raises(OutputSpacesError, match='acpc'):
        parse_output_spaces(['MNI152NLin2009cAsym'])


def test_parse_output_spaces_allows_multiple_acpc():
    specs = parse_output_spaces(['acpc:res-2mm', 'acpc:res-1p5mm'])
    assert [s.resolution.label for s in specs] == ['2mm', '1p5mm']


def test_parse_output_spaces_deduplicates_preserving_order():
    specs = parse_output_spaces(
        ['acpc:res-2mm', 'MNI152NLin2009cAsym', 'acpc:res-2mm']
    )
    assert [str(s) for s in specs] == ['acpc:res-2mm', 'MNI152NLin2009cAsym']
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `micromamba run -n linc311 pytest qsiprep/tests/test_utils_spaces.py -x -q`
Expected: collection error — `ModuleNotFoundError: No module named 'qsiprep.utils.spaces'`.

- [ ] **Step 5: Write the parser**

Create `qsiprep/utils/spaces.py`:

```python
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
from dataclasses import dataclass, replace
from collections.abc import Sequence

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
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `micromamba run -n linc311 pytest qsiprep/tests/test_utils_spaces.py -v`
Expected: PASS, 26 tests.

- [ ] **Step 7: Commit**

```bash
git add qsiprep/utils/spaces.py qsiprep/tests/test_utils_spaces.py qsiprep/utils/bids.py
git commit -m "feat: add the --output-spaces grammar parser

Parses acpc and TemplateFlow tokens, validating template names, cohorts and
resolution labels against TF_LAYOUT at parse time. cohort-auto and res-native*
stay symbolic because neither is knowable before a subject is selected.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 2: The `--output-spaces` argument

**Files:**
- Modify: `qsiprep/cli/parser.py:628-643` (replace `--anatomical-template` and `--output-resolution`), `qsiprep/cli/parser.py:930-960` (post-parse enforcement)
- Test: `qsiprep/tests/test_utils_spaces.py`

**Interfaces:**
- Consumes: `parse_output_spaces`, `OutputSpacesError` from Task 1.
- Produces: `opts.output_spaces` is a `list[str]` of canonical token strings (not `SpaceSpec` objects — the list has to survive the TOML round-trip in Task 4).

- [ ] **Step 1: Write the failing tests**

Append to `qsiprep/tests/test_utils_spaces.py`:

```python
def _min_args(tmp_path, *extra):
    bids = tmp_path / 'bids'
    (bids / 'sub-01' / 'anat').mkdir(parents=True)
    (bids / 'dataset_description.json').write_text(
        '{"Name": "t", "BIDSVersion": "1.8.0", "DatasetType": "raw"}'
    )
    return [str(bids), str(tmp_path / 'out'), 'participant', *extra]


def test_parser_accepts_output_spaces(tmp_path):
    from qsiprep.cli.parser import _build_parser

    parser = _build_parser()
    opts = parser.parse_args(
        _min_args(tmp_path, '--output-spaces', 'acpc:res-2mm', 'MNI152NLin2009cAsym')
    )
    assert opts.output_spaces == ['acpc:res-2mm', 'MNI152NLin2009cAsym']


def test_parser_expands_repeated_res(tmp_path):
    from qsiprep.cli.parser import _build_parser

    parser = _build_parser()
    opts = parser.parse_args(
        _min_args(tmp_path, '--output-spaces', 'acpc:res-2mm', 'MNI152NLin2009cAsym:res-1:res-2')
    )
    assert opts.output_spaces == [
        'acpc:res-2mm',
        'MNI152NLin2009cAsym:res-1',
        'MNI152NLin2009cAsym:res-2',
    ]


def test_parser_rejects_a_bad_token(tmp_path, capsys):
    from qsiprep.cli.parser import _build_parser

    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(_min_args(tmp_path, '--output-spaces', 'acpc:res-2'))
    assert 'acpc:res-2mm' in capsys.readouterr().err
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `micromamba run -n linc311 pytest qsiprep/tests/test_utils_spaces.py -k parser -q`
Expected: FAIL — `AttributeError: 'Namespace' object has no attribute 'output_spaces'`.

- [ ] **Step 3: Add the argparse action**

In `qsiprep/cli/parser.py`, next to the other action classes (after `ToDict`, around line 177), add:

```python
    class OutputSpacesAction(Action):
        """Parse and validate --output-spaces at parse time, storing canonical tokens."""

        def __call__(self, parser, namespace, values, option_string=None):
            from qsiprep.utils.spaces import OutputSpacesError, parse_space_token

            specs = []
            for token in values:
                try:
                    specs.extend(parse_space_token(token))
                except OutputSpacesError as exc:
                    parser.error(str(exc))

            existing = getattr(namespace, self.dest, None) or []
            setattr(namespace, self.dest, [*existing, *[str(s) for s in specs]])
```

`parse_space_token` is used rather than `parse_output_spaces` because the
"must contain an acpc space" check cannot run until deprecated flags have been
forwarded. Task 3 adds that check.

- [ ] **Step 4: Replace the two old arguments**

In `_build_parser`, delete the `--anatomical-template` and `--output-resolution` blocks at `qsiprep/cli/parser.py:628-643` and put this in their place:

```python
    g_conf.add_argument(
        '--output-spaces',
        nargs='+',
        action=OutputSpacesAction,
        default=None,
        metavar='SPACE',
        help=(
            'Standard and non-standard spaces to write outputs to, space delimited. '
            'At least one "acpc" space is required, because QSIPrep writes preprocessed '
            'DWI in ACPC space only -- for example "acpc:res-2mm". Resolutions are given '
            'as a physical size with an "mm" suffix ("res-2mm", "res-1p5mm", "res-6x6x3mm") '
            'or, for standard spaces, as a TemplateFlow resolution label ("res-1"). '
            'On "acpc", "res-nativemin" and "res-nativemax" take the smallest or largest '
            "voxel dimension of the input DWI runs (a 3x4x5 mm input gives 3x3x3 mm for "
            'nativemin and 5x5x5 mm for nativemax). Listing "acpc" more than once writes '
            'the preprocessed DWI at each resolution. Standard spaces produce transforms '
            'and anatomical derivatives; DWI is never resampled into them. Templates with '
            'cohorts accept "cohort-auto" to pick one from the participant\'s age, as in '
            '"MNIInfant:cohort-auto".'
        ),
    )
```

Note `default=None`, not `required=True`. The requirement is enforced after deprecated flags are forwarded, in Task 3.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `micromamba run -n linc311 pytest qsiprep/tests/test_utils_spaces.py -k parser -v`
Expected: PASS, 3 tests.

- [ ] **Step 6: Commit**

```bash
git add qsiprep/cli/parser.py qsiprep/tests/test_utils_spaces.py
git commit -m "feat: add the --output-spaces argument

Replaces --anatomical-template and --output-resolution in the parser. Tokens are
validated at parse time and stored as canonical strings so they survive the config
TOML round-trip.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 3: Deprecate and forward the three old flags

**Files:**
- Modify: `qsiprep/cli/parser.py:57-89` (the `deprecations` table), `qsiprep/cli/parser.py:471-505` (`--skip-anat-based-spatial-normalization`), `qsiprep/cli/parser.py:930-960` (`parse_args`)
- Test: `qsiprep/tests/test_utils_spaces.py`

**Interfaces:**
- Consumes: `opts.output_spaces` from Task 2.
- Produces: after `parse_args`, `opts.output_spaces` is always a non-empty `list[str]` containing at least one `acpc` token. `opts.output_resolution`, `opts.anatomical_template`, and `opts.skip_anat_based_spatial_normalization` no longer exist on the namespace.

The existing `forwarded_deprecations` machinery sets a scalar on a namespace attribute. These three flags need to *append* to a list and depend on `--infant`, so they are handled by a dedicated post-parse step rather than shoehorned into that table.

- [ ] **Step 1: Write the failing tests**

Append to `qsiprep/tests/test_utils_spaces.py`:

```python
def _parse(tmp_path, *extra):
    from qsiprep.cli.parser import _build_parser

    return _build_parser().parse_args(_min_args(tmp_path, *extra))


def test_output_resolution_forwards(tmp_path):
    from qsiprep.cli.parser import _apply_output_space_deprecations

    opts = _parse(tmp_path, '--output-resolution', '2')
    _apply_output_space_deprecations(opts)
    assert opts.output_spaces == ['acpc:res-2mm', 'MNI152NLin2009cAsym']


def test_output_resolution_decimal_forwards(tmp_path):
    from qsiprep.cli.parser import _apply_output_space_deprecations

    opts = _parse(tmp_path, '--output-resolution', '1.5')
    _apply_output_space_deprecations(opts)
    assert opts.output_spaces[0] == 'acpc:res-1p5mm'


def test_output_resolution_forwards_infant_template(tmp_path):
    from qsiprep.cli.parser import _apply_output_space_deprecations

    opts = _parse(tmp_path, '--output-resolution', '2', '--infant')
    _apply_output_space_deprecations(opts)
    assert opts.output_spaces == ['acpc:res-2mm', 'MNIInfant:cohort-auto']


def test_skip_normalization_drops_standard_spaces(tmp_path):
    from qsiprep.cli.parser import _apply_output_space_deprecations

    opts = _parse(
        tmp_path, '--output-resolution', '2', '--skip-anat-based-spatial-normalization'
    )
    _apply_output_space_deprecations(opts)
    assert opts.output_spaces == ['acpc:res-2mm']


def test_old_and_new_together_is_an_error(tmp_path):
    from qsiprep.cli.parser import _apply_output_space_deprecations

    opts = _parse(
        tmp_path, '--output-resolution', '2', '--output-spaces', 'acpc:res-2mm'
    )
    with pytest.raises(SystemExit):
        _apply_output_space_deprecations(opts)


def test_infant_adds_the_infant_template(tmp_path):
    from qsiprep.cli.parser import _apply_output_space_deprecations

    opts = _parse(tmp_path, '--output-spaces', 'acpc:res-2mm', '--infant')
    _apply_output_space_deprecations(opts)
    assert opts.output_spaces == ['acpc:res-2mm', 'MNIInfant:cohort-auto']


def test_infant_does_not_duplicate_an_explicit_infant_template(tmp_path):
    from qsiprep.cli.parser import _apply_output_space_deprecations

    opts = _parse(tmp_path, '--output-spaces', 'acpc:res-2mm', 'MNIInfant:cohort-3', '--infant')
    _apply_output_space_deprecations(opts)
    assert opts.output_spaces == ['acpc:res-2mm', 'MNIInfant:cohort-3']


def test_missing_acpc_is_an_error(tmp_path):
    from qsiprep.cli.parser import _apply_output_space_deprecations

    opts = _parse(tmp_path, '--output-spaces', 'MNI152NLin2009cAsym')
    with pytest.raises(SystemExit):
        _apply_output_space_deprecations(opts)


def test_nothing_given_at_all_is_an_error(tmp_path):
    from qsiprep.cli.parser import _apply_output_space_deprecations

    opts = _parse(tmp_path)
    with pytest.raises(SystemExit):
        _apply_output_space_deprecations(opts)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `micromamba run -n linc311 pytest qsiprep/tests/test_utils_spaces.py -k "forwards or skip_normalization or infant_adds or missing_acpc or nothing_given or old_and_new" -q`
Expected: FAIL — `ImportError: cannot import name '_apply_output_space_deprecations'`.

- [ ] **Step 3: Add the three deprecation entries**

In the `deprecations` dict at `qsiprep/cli/parser.py:58`, add:

```python
        '--output-resolution': (
            '27.0.0',
            'Please use `--output-spaces acpc:res-<size>mm` instead.',
        ),
        '--anatomical-template': (
            '27.0.0',
            'Please list the template in `--output-spaces` instead.',
        ),
        '--skip-anat-based-spatial-normalization': (
            '27.0.0',
            'Requesting no standard space in `--output-spaces` now skips normalization.',
        ),
```

- [ ] **Step 4: Re-declare the three flags as deprecated**

Replace the `--skip-anat-based-spatial-normalization` block at `qsiprep/cli/parser.py:500-506` with:

```python
    g_conf.add_argument(
        '--skip-anat-based-spatial-normalization',
        action=DeprecatedAction,
        default=SUPPRESS,
        help=(
            'DEPRECATED: requesting no standard space in `--output-spaces` skips '
            'normalization. This flag now drops any standard spaces from the list.'
        ),
    )
```

`DeprecatedAction` warns and keeps the dest out of the namespace, so the post-parse
step detects it via the option string. Change `DeprecatedAction.__call__` to record
what was seen, since the existing implementation only warns:

```python
        def __call__(self, parser, namespace, values, option_string=None):
            option_string = option_string or self.option_strings[0]
            _warn_deprecated(option_string)
            seen = getattr(namespace, '_deprecated_seen', [])
            namespace._deprecated_seen = [*seen, option_string]
```

Then in `g_conf`, where `--anatomical-template` and `--output-resolution` used to be, add their deprecated forms next to `--output-spaces`:

```python
    g_conf.add_argument(
        '--anatomical-template',
        action=DeprecatedStoreAction,
        default=SUPPRESS,
        choices=['MNI152NLin2009cAsym'],
        help='DEPRECATED: list the template in `--output-spaces` instead.',
    )
    g_conf.add_argument(
        '--output-resolution',
        action=DeprecatedStoreAction,
        default=SUPPRESS,
        type=float,
        help=(
            'DEPRECATED: use `--output-spaces acpc:res-<size>mm` instead. '
            'A value of 2 becomes `acpc:res-2mm`.'
        ),
    )
```

Both use `default=SUPPRESS` so "was it given?" is just `hasattr` — the same trick
`--b0-to-anat-transform` already uses and documents at `qsiprep/cli/parser.py:648-653`.

- [ ] **Step 5: Write the post-parse step**

Add this module-level function to `qsiprep/cli/parser.py`, above `parse_args`:

```python
def _format_mm(value):
    """Render a float as a res- label: 2.0 -> '2mm', 1.5 -> '1p5mm'."""
    text = f'{float(value):g}'
    return f'{text.replace(".", "p")}mm'


def _apply_output_space_deprecations(opts, parser=None):
    """Fold the deprecated output-space flags into ``opts.output_spaces``.

    Runs after the whole command line has been read, so the result does not depend
    on the order options were given in.
    """
    from qsiprep.utils.spaces import OutputSpacesError, parse_output_spaces

    def fail(message):
        if parser is not None:
            parser.error(message)
        raise SystemExit(message)

    deprecated_seen = list(getattr(opts, '_deprecated_seen', []))
    skip_normalization = '--skip-anat-based-spatial-normalization' in deprecated_seen
    legacy_resolution = getattr(opts, 'output_resolution', None)
    legacy_template = getattr(opts, 'anatomical_template', None)
    given = list(opts.output_spaces or [])

    legacy_used = [
        name
        for name, used in (
            ('--output-resolution', legacy_resolution is not None),
            ('--anatomical-template', legacy_template is not None),
            ('--skip-anat-based-spatial-normalization', skip_normalization),
        )
        if used
    ]
    if given and legacy_used:
        fail(
            f'{", ".join(legacy_used)} cannot be combined with --output-spaces. '
            'Use --output-spaces alone.'
        )

    # The infant template stands in for MNI152NLin2009cAsym, not alongside it.
    default_template = 'MNIInfant:cohort-auto' if opts.infant else 'MNI152NLin2009cAsym'

    if not given:
        if legacy_resolution is None:
            fail(
                '--output-spaces is required and must include at least one "acpc" space, '
                'for example: --output-spaces acpc:res-2mm MNI152NLin2009cAsym'
            )
        given = [f'acpc:res-{_format_mm(legacy_resolution)}']
        given.append(legacy_template or default_template)

    if opts.infant and not any(s.split(':')[0] == 'MNIInfant' for s in given):
        given.append('MNIInfant:cohort-auto')

    try:
        specs = parse_output_spaces(given)
    except OutputSpacesError as exc:
        fail(str(exc))

    if skip_normalization:
        specs = [spec for spec in specs if not spec.standard]

    opts.output_spaces = [str(spec) for spec in specs]

    for attr in ('output_resolution', 'anatomical_template', '_deprecated_seen'):
        if hasattr(opts, attr):
            delattr(opts, attr)

    return opts
```

- [ ] **Step 6: Call it from `parse_args`**

In `parse_args` (`qsiprep/cli/parser.py:930`), replace the `# Change anatomical_template based on infant parameter` block at lines 949-960 with:

```python
    _apply_output_space_deprecations(opts, parser)

    if opts.infant:
        config.loggers.cli.info(
            'Infant processing mode enabled. '
            "Inferring the subject's age and selecting the appropriate template cohort."
        )
        if opts.subject_anatomical_reference != 'sessionwise':
            config.loggers.cli.error(
                'Infant processing requires --subject-anatomical-reference sessionwise'
            )
```

Also delete the commented-out `SpatialReferences` blocks at lines 929 and 1001-1005 — they are dead scaffolding this work replaces.

- [ ] **Step 7: Run the tests to verify they pass**

Run: `micromamba run -n linc311 pytest qsiprep/tests/test_utils_spaces.py -v`
Expected: PASS, all 38 tests.

- [ ] **Step 8: Commit**

```bash
git add qsiprep/cli/parser.py qsiprep/tests/test_utils_spaces.py
git commit -m "feat: deprecate --output-resolution, --anatomical-template and --skip-anat-based-spatial-normalization

All three warn and forward into --output-spaces for one release. --output-resolution
forwards to acpc:res-<n>mm plus the anchor template, so existing invocations keep
producing the transforms QSIRecon expects. Combining old and new flags is an error.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 4: Config surface

**Files:**
- Modify: `qsiprep/config.py:255-280` (the `SpatialReferences` special case), `qsiprep/config.py:426-428` and `:558-559` and `:612-613` (the fields), `qsiprep/config.py:760-765` and `:803-823` (`init_spaces`)
- Test: `qsiprep/tests/test_utils_spaces.py`

**Interfaces:**
- Consumes: `opts.output_spaces` (`list[str]`) from Task 3.
- Produces: `config.workflow.output_spaces: list[str]`, and `config.workflow.parsed_output_spaces() -> list[SpaceSpec]`. `config.workflow.spaces` and `config.init_spaces` no longer exist.

- [ ] **Step 1: Write the failing test**

Append to `qsiprep/tests/test_utils_spaces.py`:

```python
def test_config_round_trips_output_spaces(tmp_path):
    from qsiprep import config

    config.workflow.output_spaces = ['acpc:res-2mm', 'MNIInfant:cohort-auto']
    out = tmp_path / 'config.toml'
    config.to_filename(out)
    assert 'acpc:res-2mm' in out.read_text()

    config.workflow.output_spaces = None
    config.load(out, init=False)
    assert config.workflow.output_spaces == ['acpc:res-2mm', 'MNIInfant:cohort-auto']

    specs = config.workflow.parsed_output_spaces()
    assert [s.space for s in specs] == ['acpc', 'MNIInfant']
    assert specs[1].needs_cohort_resolution is True


def test_init_spaces_is_gone():
    from qsiprep import config

    assert not hasattr(config, 'init_spaces')
    assert not hasattr(config.workflow, 'spaces')
```

- [ ] **Step 2: Run to verify failure**

Run: `micromamba run -n linc311 pytest qsiprep/tests/test_utils_spaces.py -k config_round_trips -q`
Expected: FAIL — `AttributeError: type object 'workflow' has no attribute 'parsed_output_spaces'`.

- [ ] **Step 3: Replace the config fields**

In `qsiprep/config.py`, delete the commented-out `output_spaces` block at lines 426-428 (in `execution`). In the `workflow` class, delete `anatomical_template` (lines 558-559) and `output_resolution` (lines 612-613), and add in their place:

```python
    output_spaces = None
    """Canonical ``--output-spaces`` tokens, as a list of strings."""
```

Add this classmethod to the `workflow` class, after the field declarations:

```python
    @classmethod
    def parsed_output_spaces(cls):
        """Parse :attr:`output_spaces` into :class:`~qsiprep.utils.spaces.SpaceSpec`."""
        from qsiprep.utils.spaces import parse_output_spaces

        return parse_output_spaces(cls.output_spaces or [])
```

- [ ] **Step 4: Delete the vestigial spaces machinery**

Delete `init_spaces` entirely (`qsiprep/config.py:803-823`) and its call site at line 763. In `_DeprecatedConfig`/`from_dict` (around line 258-275), delete the `SpatialReferences` import and the `isinstance(v, SpatialReferences)` branch — `output_spaces` is a plain list of strings and needs no special handling.

- [ ] **Step 5: Run to verify pass**

Run: `micromamba run -n linc311 pytest qsiprep/tests/test_utils_spaces.py -v`
Expected: PASS, all 40 tests.

- [ ] **Step 6: Check nothing else referenced the deleted names**

Run: `grep -rn "init_spaces\|workflow\.spaces\|output_resolution\|anatomical_template" --include=*.py qsiprep/ | grep -v tests/`
Expected: only `interfaces/images.py` (`ChooseInterpolator`, handled in Task 8), `workflows/anatomical/volume.py`, `workflows/dwi/*.py`, and `workflows/base.py` — all of which later tasks fix. No hits in `config.py` or `cli/`.

- [ ] **Step 7: Commit**

```bash
git add qsiprep/config.py qsiprep/tests/test_utils_spaces.py
git commit -m "refactor: store output_spaces in config and delete init_spaces

workflow.spaces was set from a niworkflows SpatialReferences and read by nothing --
leftover fMRIPrep scaffolding. Canonical token strings round-trip through TOML with
no special-casing.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 5: Derivative-naming regression guard

**Files:**
- Create: `qsiprep/tests/test_output_spaces_naming.py`

**Interfaces:**
- Consumes: `config.workflow.output_spaces` from Task 4.
- Produces: `collect_datasink_entities(workflow) -> dict[str, dict]`, mapping node name to its `DerivativesDataSink` entity settings. Later tasks reuse it.

This task lands **before** any fan-out. It is what proves a single-`acpc` run keeps producing exactly the filenames QSIRecon expects.

- [ ] **Step 1: Write the test**

Create `qsiprep/tests/test_output_spaces_naming.py`:

```python
"""Guards the derivative names a single-acpc run produces.

QSIRecon consumes these filenames, so a single `acpc` space must keep producing
exactly what QSIPrep produced before --output-spaces existed.
"""

import pytest

from qsiprep import config
from qsiprep.interfaces.bids import DerivativesDataSink

# node name -> the entities that end up in its filename
EXPECTED_ANAT_ENTITIES = {
    'ds_t1_preproc': {'space': 'ACPC', 'desc': 'preproc'},
    'ds_t1_mask': {'space': 'ACPC', 'desc': 'brain', 'suffix': 'mask'},
    'ds_t1_seg': {'space': 'ACPC', 'suffix': 'dseg'},
    'ds_t1_aseg': {'space': 'ACPC', 'desc': 'aseg', 'suffix': 'dseg'},
    'ds_t1_mni_warp': {'from': 'ACPC', 'to': 'MNI152NLin2009cAsym', 'suffix': 'xfm'},
    'ds_t1_mni_inv_warp': {'from': 'MNI152NLin2009cAsym', 'to': 'ACPC', 'suffix': 'xfm'},
    'ds_t1_template_acpc_transforms': {'from': 'anat', 'to': 'ACPC', 'suffix': 'xfm'},
    'ds_t1_template_acpc_inv_transforms': {'from': 'ACPC', 'to': 'anat', 'suffix': 'xfm'},
}


def collect_datasink_entities(workflow):
    """Map each DerivativesDataSink node name to the entities it will write."""
    found = {}
    for name in workflow.list_node_names():
        node = workflow.get_node(name)
        if node is None or not isinstance(node.interface, DerivativesDataSink):
            continue
        entities = {}
        for key in ('space', 'desc', 'suffix', 'res', 'cohort', 'from', 'to', 'mode'):
            value = node.inputs.trait_get().get(key)
            if value is not None and str(value) != '<undefined>':
                entities[key] = value
        found[name.split('.')[-1]] = entities
    return found


@pytest.fixture
def single_acpc_config():
    config.workflow.output_spaces = ['acpc:res-2mm', 'MNI152NLin2009cAsym']
    config.workflow.anat_modality = 'T1w'
    config.workflow.infant = False
    config.execution.output_dir = '/tmp/qsiprep-naming-test'
    return config


def test_single_acpc_anat_derivative_names(single_acpc_config):
    from qsiprep.workflows.anatomical.volume import init_anat_derivatives_wf
    from qsiprep.utils.spaces import parse_output_spaces

    specs = parse_output_spaces(config.workflow.output_spaces)
    wf = init_anat_derivatives_wf(output_spaces=specs)
    found = collect_datasink_entities(wf)

    for node_name, expected in EXPECTED_ANAT_ENTITIES.items():
        assert node_name in found, f'{node_name} disappeared from the derivatives workflow'
        for key, value in expected.items():
            assert found[node_name].get(key) == value, (
                f'{node_name}: {key} is {found[node_name].get(key)!r}, expected {value!r}'
            )


def test_single_acpc_writes_no_res_entity(single_acpc_config):
    from qsiprep.workflows.anatomical.volume import init_anat_derivatives_wf
    from qsiprep.utils.spaces import parse_output_spaces

    specs = parse_output_spaces(config.workflow.output_spaces)
    wf = init_anat_derivatives_wf(output_spaces=specs)
    found = collect_datasink_entities(wf)

    acpc_nodes = {n: e for n, e in found.items() if e.get('space') == 'ACPC'}
    assert acpc_nodes, 'expected some ACPC-space derivatives'
    for node_name, entities in acpc_nodes.items():
        assert 'res' not in entities, f'{node_name} gained a res- entity on a single-acpc run'
```

Note: `init_anat_derivatives_wf` currently takes `anatomical_template` and `has_t2w`.
This test calls it with the `output_spaces=` signature that Task 13 introduces. Until
then, run it with the current signature by passing
`anatomical_template='MNI152NLin2009cAsym'` and update the two call sites in Task 13.

- [ ] **Step 2: Run against the current code with the current signature**

Temporarily change both `init_anat_derivatives_wf(output_spaces=specs)` calls to
`init_anat_derivatives_wf(anatomical_template='MNI152NLin2009cAsym')`, then run:

`micromamba run -n linc311 pytest qsiprep/tests/test_output_spaces_naming.py -v`
Expected: PASS. This is the baseline — it records what today's code produces.

- [ ] **Step 3: Restore the target signature and mark it xfail until Task 13**

Change the calls back to `init_anat_derivatives_wf(output_spaces=specs)` and add at the top of both tests:

```python
@pytest.mark.xfail(reason='init_anat_derivatives_wf gains output_spaces in Task 13', strict=False)
```

- [ ] **Step 4: Run to confirm xfail**

Run: `micromamba run -n linc311 pytest qsiprep/tests/test_output_spaces_naming.py -v`
Expected: 2 xfailed.

- [ ] **Step 5: Commit**

```bash
git add qsiprep/tests/test_output_spaces_naming.py
git commit -m "test: guard the derivative names a single-acpc run produces

Records the filenames QSIRecon depends on before any output-space fan-out lands.
xfail until init_anat_derivatives_wf takes output_spaces in Task 13.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 6: `GetTemplate` learns resolutions and cohorts

**Files:**
- Modify: `qsiprep/interfaces/anatomical.py:196-252`
- Test: `qsiprep/tests/test_interfaces_images.py`

**Interfaces:**
- Consumes: `SpaceSpec` from Task 1.
- Produces: `GetTemplate` inputs become `template_name: Str`, `cohort: Str` (optional), `resolution: Str` (optional, defaults to `'1'`), `anatomical_contrast: Enum`. The old `template_spec` string with `+cohort` is gone.

`GetTemplate` currently hardcodes `resolution='1'` and splits a `template+cohort`
string. For `res-` on a standard space to select a TemplateFlow grid, it needs the
resolved spec.

- [ ] **Step 1: Write the failing test**

Append to `qsiprep/tests/test_interfaces_images.py`:

```python
def test_get_template_uses_resolution_and_cohort(tmp_path):
    from qsiprep.interfaces.anatomical import GetTemplate

    iface = GetTemplate(
        template_name='MNIInfant',
        cohort='2',
        resolution='2',
        anatomical_contrast='T1w',
    )
    result = iface.run(cwd=str(tmp_path))
    name = Path(result.outputs.template_file).name
    assert 'cohort-2' in name
    assert 'res-2' in name


def test_get_template_defaults_to_res_1(tmp_path):
    from qsiprep.interfaces.anatomical import GetTemplate

    iface = GetTemplate(template_name='MNI152NLin2009cAsym', anatomical_contrast='T1w')
    result = iface.run(cwd=str(tmp_path))
    assert 'res-01' in Path(result.outputs.template_file).name
```

Add `from pathlib import Path` to that module's imports if it is not already there.

- [ ] **Step 2: Run to verify failure**

Run: `micromamba run -n linc311 pytest qsiprep/tests/test_interfaces_images.py -k get_template -q`
Expected: FAIL — `TraitError: 'template_name' is not a trait`.

- [ ] **Step 3: Rewrite the interface**

Replace `_GetTemplateInputSpec` and `GetTemplate._run_interface` in `qsiprep/interfaces/anatomical.py`:

```python
class _GetTemplateInputSpec(BaseInterfaceInputSpec):
    template_name = traits.Str(desc='TemplateFlow template name', mandatory=True)
    cohort = traits.Str(desc='Cohort label, if the template has one')
    resolution = traits.Str('1', usedefault=True, desc='TemplateFlow resolution entity')
    anatomical_contrast = traits.Enum('T1w', 'T2w', 'none')
```

```python
    def _run_interface(self, runtime):
        from templateflow.api import get as get_template

        anatomical_contrast = self.inputs.anatomical_contrast
        if anatomical_contrast == 'none':
            LOGGER.info('Using T1w modality template for ACPC alignment')
            anatomical_contrast = 'T1w'

        cohort = self.inputs.cohort if isdefined(self.inputs.cohort) else None

        template_path = get_template(
            self.inputs.template_name,
            cohort=cohort,
            resolution=self.inputs.resolution,
            desc=None,
            suffix=anatomical_contrast,
            extension='.nii.gz',
        )
        mask_path = get_template(
            self.inputs.template_name,
            cohort=cohort,
            resolution=self.inputs.resolution,
            desc='brain',
            suffix='mask',
            extension='.nii.gz',
        )

        local_template = Path(runtime.cwd) / template_path.name
        local_mask = Path(runtime.cwd) / mask_path.name

        shutil.copy(template_path, local_template)
        shutil.copy(mask_path, local_mask)

        self._results['template_file'] = str(local_template)
        self._results['mask_file'] = str(local_mask)

        return runtime
```

- [ ] **Step 4: Add a helper that turns a spec into these inputs**

Add to `qsiprep/utils/spaces.py`:

```python
def templateflow_kwargs(spec) -> dict:
    """Turn a standard-space :class:`SpaceSpec` into ``GetTemplate`` inputs.

    A ``res-<n>mm`` spec has no TemplateFlow label, so the highest-resolution grid
    (``res-1``) is fetched and resampled downstream.
    """
    if not spec.standard:
        raise OutputSpacesError(f'{spec.space} is not a standard space.')

    kwargs = {'template_name': spec.space}
    if spec.cohort not in (None, 'auto'):
        kwargs['cohort'] = spec.cohort
    if spec.resolution is not None and spec.resolution.kind == 'label':
        kwargs['resolution'] = spec.resolution.label
    return kwargs
```

Add a matching test to `qsiprep/tests/test_utils_spaces.py`:

```python
def test_templateflow_kwargs():
    from qsiprep.utils.spaces import templateflow_kwargs

    (spec,) = parse_space_token('MNIInfant:cohort-3:res-2')
    assert templateflow_kwargs(spec) == {
        'template_name': 'MNIInfant',
        'cohort': '3',
        'resolution': '2',
    }

    (mm_spec,) = parse_space_token('MNI152NLin2009cAsym:res-1p5mm')
    assert templateflow_kwargs(mm_spec) == {'template_name': 'MNI152NLin2009cAsym'}
```

- [ ] **Step 5: Update the one existing call site**

In `qsiprep/workflows/anatomical/volume.py:189-194`, replace the `GetTemplate(template_spec=..., ...)` construction with kwargs built from the anchor spec. Task 9 supplies `anchor_spec`; for now use:

```python
    get_template = pe.Node(
        GetTemplate(
            anatomical_contrast=anat_modality,
            **templateflow_kwargs(acpc_anchor),
        ),
        name='get_template_image',
    )
```

and add `from ...utils.spaces import templateflow_kwargs` to the module imports. Delete the `# XXX: This is a temporary solution until QSIPrep supports flexible output spaces.` comment above it — it is no longer true.

`acpc_anchor` is the parameter Task 11 adds to `init_anat_preproc_wf` and Task 9 computes. Until Task 11 lands, pass `SpaceSpec(space='MNI152NLin2009cAsym')` so the module still imports.

- [ ] **Step 6: Run to verify pass**

Run: `micromamba run -n linc311 pytest qsiprep/tests/test_interfaces_images.py -k get_template qsiprep/tests/test_utils_spaces.py -k templateflow -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add qsiprep/interfaces/anatomical.py qsiprep/utils/spaces.py qsiprep/workflows/anatomical/volume.py qsiprep/tests/
git commit -m "feat: give GetTemplate resolution and cohort inputs

It hardcoded resolution='1' and parsed a template+cohort string. It now takes the
resolved spec, so res- on a standard space selects a TemplateFlow grid.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 7: `res-native*` resolution from the DWI headers

**Files:**
- Modify: `qsiprep/interfaces/anatomical.py:58-92` (`VoxelSizeChooser`), `qsiprep/workflows/anatomical/volume.py:1242-1285` (`init_output_grid_wf`)
- Test: `qsiprep/tests/test_interfaces_images.py`

**Interfaces:**
- Consumes: `Resolution` from Task 1.
- Produces: `VoxelSizeChooser` gains `input_images: InputMultiObject(File)` replacing `input_image`, and keeps `voxel_size: Float` and `anisotropic_strategy: Enum('min', 'max', 'mean')`. `init_output_grid_wf(resolution, name)` takes a `Resolution` and a workflow name.

`inputnode.input_image` on `init_output_grid_wf` has never been connected, so the
min/max path is currently dead code. This task makes it reachable and makes it consider
every DWI run rather than one.

- [ ] **Step 1: Write the failing tests**

Append to `qsiprep/tests/test_interfaces_images.py`:

```python
def _write_image(path, zooms):
    import nibabel as nb
    import numpy as np

    affine = np.diag([*zooms, 1.0])
    nb.Nifti1Image(np.zeros((4, 4, 4)), affine).to_filename(path)
    return str(path)


def test_voxel_size_chooser_max_across_runs(tmp_path):
    from qsiprep.interfaces.anatomical import VoxelSizeChooser

    a = _write_image(tmp_path / 'a.nii.gz', (3.0, 4.0, 5.0))
    b = _write_image(tmp_path / 'b.nii.gz', (2.0, 2.0, 2.0))
    result = VoxelSizeChooser(input_images=[a, b], anisotropic_strategy='max').run(
        cwd=str(tmp_path)
    )
    assert result.outputs.voxel_size == 5.0


def test_voxel_size_chooser_min_across_runs(tmp_path):
    from qsiprep.interfaces.anatomical import VoxelSizeChooser

    a = _write_image(tmp_path / 'a.nii.gz', (3.0, 4.0, 5.0))
    b = _write_image(tmp_path / 'b.nii.gz', (2.5, 2.5, 2.5))
    result = VoxelSizeChooser(input_images=[a, b], anisotropic_strategy='min').run(
        cwd=str(tmp_path)
    )
    assert result.outputs.voxel_size == 2.5


def test_voxel_size_chooser_explicit_size_wins(tmp_path):
    from qsiprep.interfaces.anatomical import VoxelSizeChooser

    a = _write_image(tmp_path / 'a.nii.gz', (3.0, 4.0, 5.0))
    result = VoxelSizeChooser(input_images=[a], voxel_size=1.7).run(cwd=str(tmp_path))
    assert result.outputs.voxel_size == 1.7
```

- [ ] **Step 2: Run to verify failure**

Run: `micromamba run -n linc311 pytest qsiprep/tests/test_interfaces_images.py -k voxel_size -q`
Expected: FAIL — `TraitError: 'input_images' is not a trait`.

- [ ] **Step 3: Rewrite `VoxelSizeChooser`**

Replace `_VoxelSizeChooserInputSpec` and the `_run_interface` body in `qsiprep/interfaces/anatomical.py`:

```python
class _VoxelSizeChooserInputSpec(BaseInterfaceInputSpec):
    voxel_size = traits.Float()
    input_images = InputMultiObject(File(exists=True))
    anisotropic_strategy = traits.Enum('min', 'max', 'mean', usedefault=True)
```

```python
    def _run_interface(self, runtime):
        if not isdefined(self.inputs.input_images) and not isdefined(self.inputs.voxel_size):
            raise Exception('Either voxel_size or input_images need to be defined')

        # An explicit size always wins; the strategies only apply to measured images.
        if isdefined(self.inputs.voxel_size):
            self._results['voxel_size'] = self.inputs.voxel_size
            return runtime

        zooms = []
        for image in self.inputs.input_images:
            zooms.extend(nb.load(image).header.get_zooms()[:3])

        if self.inputs.anisotropic_strategy == 'min':
            voxel_size = min(zooms)
        elif self.inputs.anisotropic_strategy == 'max':
            voxel_size = max(zooms)
        else:
            voxel_size = np.round(np.mean(zooms), 2)

        self._results['voxel_size'] = float(voxel_size)
        return runtime
```

Add `InputMultiObject` to the `nipype.interfaces.base` import list at the top of the module.

- [ ] **Step 4: Rewrite `init_output_grid_wf` to take a `Resolution`**

Replace `init_output_grid_wf` in `qsiprep/workflows/anatomical/volume.py`:

```python
def init_output_grid_wf(resolution, name='output_grid_wf') -> Workflow:
    """Generate a non-oblique, uniform voxel-size grid around a brain.

    Parameters
    ----------
    resolution : :class:`~qsiprep.utils.spaces.Resolution`
        The resolution to build the grid at. A ``native`` resolution leaves the size
        undefined so ``VoxelSizeChooser`` measures it from the DWI runs at run time.
    """
    workflow = Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['template_image', 'input_images']),
        name='inputnode',
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=['grid_image']), name='outputnode')

    if resolution.kind == 'native':
        voxel_size = traits.Undefined
        strategy = resolution.strategy
    else:
        # acpc resolutions are validated isotropic, so any axis is the size.
        voxel_size = resolution.zooms[0]
        strategy = 'max'

    padding = 4 if config.workflow.infant else 8

    autobox_template = pe.Node(
        afni.Autobox(outputtype='NIFTI_GZ', padding=padding), name='autobox_template'
    )
    deoblique_autobox = pe.Node(
        afni.Warp(outputtype='NIFTI_GZ', deoblique=True), name='deoblique_autobox'
    )
    voxel_size_chooser = pe.Node(
        VoxelSizeChooser(voxel_size=voxel_size, anisotropic_strategy=strategy),
        name='voxel_size_chooser',
    )
    resample_to_voxel_size = pe.Node(
        afni.Resample(outputtype='NIFTI_GZ'), name='resample_to_voxel_size'
    )

    workflow.connect([
        (inputnode, autobox_template, [('template_image', 'in_file')]),
        (autobox_template, deoblique_autobox, [('out_file', 'in_file')]),
        (deoblique_autobox, resample_to_voxel_size, [('out_file', 'in_file')]),
        (resample_to_voxel_size, outputnode, [('out_file', 'grid_image')]),
        (inputnode, voxel_size_chooser, [('input_images', 'input_images')]),
        (voxel_size_chooser, resample_to_voxel_size, [(('voxel_size', _tupleize), 'voxel_size')]),
    ])  # fmt:skip

    return workflow
```

- [ ] **Step 5: Run to verify pass**

Run: `micromamba run -n linc311 pytest qsiprep/tests/test_interfaces_images.py -k voxel_size -v`
Expected: PASS, 3 tests.

- [ ] **Step 6: Commit**

```bash
git add qsiprep/interfaces/anatomical.py qsiprep/workflows/anatomical/volume.py qsiprep/tests/test_interfaces_images.py
git commit -m "feat: resolve res-native* from every DWI run

VoxelSizeChooser's min/max strategies were unreachable because input_image was never
connected. It now takes every DWI run and init_output_grid_wf takes a Resolution.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 8: Interpolator and boilerplate read the grid

**Files:**
- Modify: `qsiprep/interfaces/images.py:530-565` (`ChooseInterpolator`), `qsiprep/workflows/dwi/resampling.py:130-200`
- Test: `qsiprep/tests/test_interfaces_images.py`

**Interfaces:**
- Consumes: `init_output_grid_wf` output from Task 7.
- Produces: `ChooseInterpolator` takes `output_grid: File` instead of `output_resolution: Float`.

`res-nativemax` has no number at workflow-build time, so the interpolator can no longer
read a config float. The grid image is already an inputnode field in `init_dwi_trans_wf`.

- [ ] **Step 1: Write the failing test**

Append to `qsiprep/tests/test_interfaces_images.py`:

```python
def test_choose_interpolator_from_grid(tmp_path):
    from qsiprep.interfaces.images import ChooseInterpolator

    dwi = _write_image(tmp_path / 'dwi.nii.gz', (2.0, 2.0, 2.0))
    coarse_grid = _write_image(tmp_path / 'coarse.nii.gz', (2.0, 2.0, 2.0))
    fine_grid = _write_image(tmp_path / 'fine.nii.gz', (1.0, 1.0, 1.0))

    same = ChooseInterpolator(dwi_files=[dwi], output_grid=coarse_grid).run(cwd=str(tmp_path))
    assert same.outputs.interpolation_method == 'LanczosWindowedSinc'

    upsampled = ChooseInterpolator(dwi_files=[dwi], output_grid=fine_grid).run(cwd=str(tmp_path))
    assert upsampled.outputs.interpolation_method == 'Linear'
```

- [ ] **Step 2: Run to verify failure**

Run: `micromamba run -n linc311 pytest qsiprep/tests/test_interfaces_images.py -k choose_interpolator -q`
Expected: FAIL — `TraitError: 'output_grid' is not a trait`.

- [ ] **Step 3: Rewrite the interface**

In `qsiprep/interfaces/images.py`, replace `_ChooseInterpolatorInputSpec` and the `_run_interface` body:

```python
class _ChooseInterpolatorInputSpec(BaseInterfaceInputSpec):
    dwi_files = InputMultiObject(File(exists=True), mandatory=True)
    output_grid = File(exists=True, mandatory=True)
    sloppy = traits.Bool(False, usedefault=True)
```

```python
    def _run_interface(self, runtime):
        if self.inputs.sloppy:
            self._results['interpolation_method'] = 'NearestNeighbor'
            LOGGER.warning('Using NN interpolation for sloppy mode')
            return runtime

        output_resolution = np.array(nb.load(self.inputs.output_grid).header.get_zooms()[:3])
        interpolator = 'LanczosWindowedSinc'
        for input_file in self.inputs.dwi_files:
            resolution_cutoff = 0.9 * np.array(nb.load(input_file).header.get_zooms()[:3])
            if np.any(output_resolution < resolution_cutoff):
                interpolator = 'Linear'
                LOGGER.warning('Using Linear interpolation for upsampling')
                break
        self._results['interpolation_method'] = interpolator
        return runtime
```

The stray `print(output_resolution, resolution_cutoff)` at line 558 goes away with it.

- [ ] **Step 4: Rewire the node and fix the boilerplate**

In `qsiprep/workflows/dwi/resampling.py`, delete the `output_resolution = config.workflow.output_resolution` line at 134 and replace the `__desc__` assignment at 135-138 with a version that names the spec instead of a number. `init_dwi_trans_wf` gains a `resolution` parameter (a `Resolution` from Task 1):

```python
    if resolution.kind == 'native':
        vox_desc = f'{resolution.strategy}imum native voxel size'
    else:
        vox_desc = f'{resolution.label.replace("p", ".").replace("mm", "")}mm isotropic voxels'

    workflow.__desc__ = f"""\
The DWI time-series were resampled to {template},
generating a *preprocessed DWI run in {template} space* with {vox_desc}.
"""
```

Then change the `get_interpolation` node at line 191-194 to drop the config read:

```python
    get_interpolation = pe.Node(
        ChooseInterpolator(sloppy=config.execution.sloppy),
        name='get_interpolation',
    )
```

and add `('output_grid', 'output_grid')` to the `inputnode -> get_interpolation` connection block.

- [ ] **Step 5: Run to verify pass**

Run: `micromamba run -n linc311 pytest qsiprep/tests/test_interfaces_images.py -k choose_interpolator -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add qsiprep/interfaces/images.py qsiprep/workflows/dwi/resampling.py qsiprep/tests/test_interfaces_images.py
git commit -m "refactor: pick the interpolator from the output grid

res-nativemax has no number at build time, so ChooseInterpolator reads zooms off the
grid image, which init_dwi_trans_wf already receives. Also makes the interpolator
correct per-resolution once ACPC fans out.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 9: Per-subject cohort resolution and anchor selection

**Files:**
- Modify: `qsiprep/workflows/base.py:207-226`
- Test: `qsiprep/tests/test_cli_run.py`

**Interfaces:**
- Consumes: `config.workflow.parsed_output_spaces()` from Task 4, `cohort_by_months` and `COHORT_KEY` from Task 1.
- Produces: `resolve_output_spaces(specs, bids_dir, subject_id, session_id) -> list[SpaceSpec]` and `select_acpc_anchor(specs) -> SpaceSpec`, both in `qsiprep/utils/spaces.py`.

- [ ] **Step 1: Write the failing tests**

Append to `qsiprep/tests/test_utils_spaces.py`:

```python
def test_select_acpc_anchor_defaults_to_2009c():
    from qsiprep.utils.spaces import select_acpc_anchor

    specs = parse_output_spaces(['acpc:res-2mm', 'MNI152NLin6Asym'])
    anchor = select_acpc_anchor(specs)
    assert anchor.space == 'MNI152NLin2009cAsym'


def test_select_acpc_anchor_prefers_an_infant_template():
    from qsiprep.utils.spaces import select_acpc_anchor

    specs = parse_output_spaces(['acpc:res-2mm', 'MNI152NLin6Asym', 'MNIInfant:cohort-3'])
    anchor = select_acpc_anchor(specs)
    assert anchor.space == 'MNIInfant'
    assert anchor.cohort == '3'


def test_resolve_output_spaces_fills_in_the_cohort(monkeypatch):
    from qsiprep.utils import spaces as spaces_mod

    monkeypatch.setattr(spaces_mod, '_age_in_months', lambda *a, **k: 7)
    specs = parse_output_spaces(['acpc:res-2mm', 'MNIInfant:cohort-auto'])
    resolved = spaces_mod.resolve_output_spaces(specs, 'bids', '01', None)
    assert resolved[1].cohort == '3'
    assert resolved[1].fullname == 'MNIInfant+3'


def test_resolve_output_spaces_errors_without_an_age(monkeypatch):
    from qsiprep.utils import spaces as spaces_mod

    monkeypatch.setattr(spaces_mod, '_age_in_months', lambda *a, **k: None)
    specs = parse_output_spaces(['acpc:res-2mm', 'MNIInfant:cohort-auto'])
    with pytest.raises(OutputSpacesError, match='MNIInfant'):
        spaces_mod.resolve_output_spaces(specs, 'bids', '01', None)


def test_resolve_output_spaces_reads_the_age_once(monkeypatch):
    from qsiprep.utils import spaces as spaces_mod

    calls = []

    def _fake(*args, **kwargs):
        calls.append(args)
        return 7

    monkeypatch.setattr(spaces_mod, '_age_in_months', _fake)
    specs = parse_output_spaces(
        ['acpc:res-2mm', 'MNIInfant:cohort-auto', 'UNCInfant:cohort-auto']
    )
    spaces_mod.resolve_output_spaces(specs, 'bids', '01', None)
    assert len(calls) == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `micromamba run -n linc311 pytest qsiprep/tests/test_utils_spaces.py -k "anchor or resolve_output" -q`
Expected: FAIL — `ImportError: cannot import name 'select_acpc_anchor'`.

- [ ] **Step 3: Add both functions**

Append to `qsiprep/utils/spaces.py`:

```python
DEFAULT_ANCHOR = 'MNI152NLin2009cAsym'
#: Templates that make ACPC alignment infant-appropriate, most specific first.
INFANT_ANCHORS = ('MNIInfant', 'UNCInfant')


def _age_in_months(bids_dir, subject_id, session_id):
    """Indirection so tests can substitute an age without a BIDS tree."""
    from qsiprep.utils.bids import parse_bids_for_age_months

    return parse_bids_for_age_months(bids_dir, subject_id, session_id)


def select_acpc_anchor(specs) -> SpaceSpec:
    """Pick the template that anchors ACPC alignment and the output grid.

    Derived rather than user-selectable: an infant template among the requested
    spaces anchors ACPC, otherwise MNI152NLin2009cAsym does. Independent of the
    order spaces were listed in.
    """
    for name in INFANT_ANCHORS:
        for spec in specs:
            if spec.space == name:
                return spec
    return SpaceSpec(space=DEFAULT_ANCHOR)


def resolve_output_spaces(specs, bids_dir, subject_id, session_id) -> list:
    """Replace every ``cohort-auto`` with a cohort chosen from the participant's age.

    The age is read at most once, however many templates asked for it.
    """
    from qsiprep.utils.bids import cohort_by_months

    if not any(spec.needs_cohort_resolution for spec in specs):
        return list(specs)

    months = _age_in_months(bids_dir, subject_id, session_id)
    if months is None:
        wanted = ', '.join(
            spec.space for spec in specs if spec.needs_cohort_resolution
        )
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
    return resolved
```

- [ ] **Step 4: Use them in `init_single_subject_wf`**

In `qsiprep/workflows/base.py`, replace lines 207-226 (the `anatomical_template` block) with:

```python
    from ..utils.spaces import resolve_output_spaces, select_acpc_anchor

    output_spaces = config.workflow.parsed_output_spaces()
    if any(spec.needs_cohort_resolution for spec in output_spaces):
        if session_ids and len(session_ids) > 1:
            raise RuntimeError(
                'Automatic cohort selection is only available for single session processing.'
            )
    output_spaces = resolve_output_spaces(
        output_spaces,
        config.execution.bids_dir,
        subject_id,
        None if not session_ids else session_ids[0],
    )
    acpc_anchor = select_acpc_anchor(output_spaces)
    acpc_specs = [spec for spec in output_spaces if not spec.standard]
    standard_specs = [spec for spec in output_spaces if spec.standard]
```

Replace every later use of `anatomical_template` in this function: `SubjectSummary(template=...)` at line 278 takes `templates=[s.fullname for s in standard_specs]` (Task 14), and the `init_anat_preproc_wf` call at line 331 and `init_dwi_finalize_wf` call at line 636 take `output_spaces=output_spaces`, `acpc_anchor=acpc_anchor` and `acpc_specs=acpc_specs` respectively (Tasks 11-13).

- [ ] **Step 5: Run to verify pass**

Run: `micromamba run -n linc311 pytest qsiprep/tests/test_utils_spaces.py -v`
Expected: PASS, all tests including the five new ones.

- [ ] **Step 6: Commit**

```bash
git add qsiprep/utils/spaces.py qsiprep/workflows/base.py qsiprep/tests/test_utils_spaces.py
git commit -m "feat: resolve cohort-auto per subject and derive the ACPC anchor

The age is read once per subject however many templates need it. The anchor is
derived from the requested spaces rather than a flag, so MNIInfant in --output-spaces
takes over --infant's template role.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 9b: Migrate the DWI-side anchor consumers

**Files:**
- Modify: `qsiprep/workflows/dwi/diffprep.py:599`, `qsiprep/workflows/dwi/base.py:57,73,253`, `qsiprep/workflows/dwi/hmc_sdc.py:30,47,250`
- Test: `qsiprep/tests/test_workflows_native.py`

**Interfaces:**
- Consumes: `select_acpc_anchor(specs) -> SpaceSpec` and `SpaceSpec.fullname` from Task 9.
- Produces: `init_dwi_preproc_wf` and `init_qsiprep_hmcsdc_wf` take `acpc_anchor` (a
  `SpaceSpec`) in place of `anatomical_template` (a string).

**Why this task exists:** it was missing from the original plan and was found by the
Task 4 review. Three DWI-side files consume the anchor template. `diffprep.py:599`
reads `config.workflow.anatomical_template` directly, which Task 4 deleted — so
TORTOISE/DIFFPREP susceptibility correction raises `AttributeError` at workflow-build
time until this lands. The other two thread it as a parameter. None is covered by any
other task, and the existing tests miss it because DIFFPREP paths need external
binaries that are absent in CI.

The value these consumers need is the ACPC anchor's `fullname` — the same string the
old `anatomical_template` carried, cohort included (`MNIInfant+3`). They feed
`b0_sdc_wf.inputs.inputnode.template`, which is the fieldmap-less SyN registration
target.

- [ ] **Step 1: Write the failing test**

Append to `qsiprep/tests/test_workflows_native.py`:

```python
def test_diffprep_sdc_uses_the_acpc_anchor(tmp_path):
    """diffprep read config.workflow.anatomical_template, which no longer exists."""
    from qsiprep import config
    from qsiprep.utils.spaces import parse_output_spaces, select_acpc_anchor

    config.workflow.output_spaces = ['acpc:res-2mm', 'MNIInfant:cohort-3']
    specs = parse_output_spaces(config.workflow.output_spaces)
    anchor = select_acpc_anchor(specs)
    assert anchor.fullname == 'MNIInfant+3'

    # The config field diffprep.py used to read must be gone, so any surviving
    # reader is a build-time AttributeError rather than a silent None.
    assert not hasattr(config.workflow, 'anatomical_template')
```

- [ ] **Step 2: Run to verify it fails or passes for the right reason**

Run: `micromamba run -n linc311 pytest qsiprep/tests/test_workflows_native.py -k diffprep_sdc -v`
Expected: PASS on both assertions (they describe the post-Task-4 world). This test
documents the contract; the real verification is Step 5's grep.

- [ ] **Step 3: Thread the anchor through the two parameter-based consumers**

In `qsiprep/workflows/dwi/base.py`, rename the `anatomical_template` parameter of
`init_dwi_preproc_wf` to `acpc_anchor`, update the docstring example at line 73 to
`acpc_anchor=SpaceSpec(space='MNI152NLin2009cAsym')`, and change the pass-through at
line 253 to `acpc_anchor=acpc_anchor`.

In `qsiprep/workflows/dwi/hmc_sdc.py`, rename the `anatomical_template` parameter of
`init_qsiprep_hmcsdc_wf` to `acpc_anchor`, update the docstring at line 47, and change
line 250 to:

```python
    b0_sdc_wf.inputs.inputnode.template = acpc_anchor.fullname
```

- [ ] **Step 4: Fix the direct config read**

In `qsiprep/workflows/dwi/diffprep.py`, replace line 599:

```python
        b0_sdc_wf.inputs.inputnode.template = config.workflow.anatomical_template
```

with a read of the resolved anchor:

```python
        from ...utils.spaces import select_acpc_anchor

        b0_sdc_wf.inputs.inputnode.template = select_acpc_anchor(
            config.workflow.parsed_output_spaces()
        ).fullname
```

Note this resolves the anchor from config rather than taking it as a parameter,
because `diffprep.py` is reached through a call chain that does not thread the anchor.
A `cohort-auto` spec is still symbolic here, so `fullname` returns the bare template
name in that case. That is harmless, but not for the reason first assumed: the
`template` inputnode field is **never read**. It is declared at
`workflows/fieldmap/syn.py:152`, threaded in from `workflows/fieldmap/base.py:130,266`,
and no node consumes it — it is write-only through the whole chain. Whatever string is
set here has no effect on any output. (Removing the dead field is a separate cleanup,
out of scope for this plan.)

- [ ] **Step 5: Confirm no consumer of the deleted field remains**

Run: `grep -rn "config\.workflow\.anatomical_template\|config\.workflow\.output_resolution" --include=*.py qsiprep/`
Expected: no hits outside `qsiprep/cli/parser.py` (which only reads them off the
argparse namespace, not off config).

- [ ] **Step 6: Update the call site**

`workflows/base.py:636` passes `anatomical_template=` into `init_dwi_preproc_wf`.
Per the controller's standing ruling that each task owns the call sites of the
signatures it changes, update it here to `acpc_anchor=acpc_anchor`.

- [ ] **Step 7: Run the suite**

Run: `micromamba run -n linc311 pytest qsiprep/tests/ -m "not integration" -q`
Expected: no NEW failures beyond the known baseline.

- [ ] **Step 8: Commit**

```bash
git add qsiprep/workflows/dwi/ qsiprep/workflows/base.py qsiprep/tests/test_workflows_native.py
git commit -m "fix: thread the ACPC anchor into the DWI-side SDC consumers

diffprep.py read config.workflow.anatomical_template directly, which no longer
exists, so TORTOISE SDC raised AttributeError at build time. dwi/base.py and
hmc_sdc.py threaded the same value as a string parameter; both now take the
resolved SpaceSpec.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 10: A reusable LPS+ reorientation sub-workflow

**Files:**
- Modify: `qsiprep/workflows/anatomical/volume.py:196-220`
- Test: `qsiprep/tests/test_workflows_native.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `init_template_lps_wf(name='template_lps_wf') -> Workflow` with `inputnode.template_file`, `inputnode.mask_file` and `outputnode.template_lps`, `outputnode.mask_lps`.

The two `reorient_tpl_*_to_lps` nodes exist only for the ACPC anchor. Task 13 needs
the same reorientation for every requested standard space.

- [ ] **Step 1: Write the failing test**

Append to `qsiprep/tests/test_workflows_native.py`:

```python
def test_template_lps_wf_reorients_to_lps():
    from qsiprep.workflows.anatomical.volume import init_template_lps_wf

    wf = init_template_lps_wf()
    # AFNI spells LPS+ as RAI.
    assert wf.get_node('reorient_brain').inputs.orientation == 'RAI'
    assert wf.get_node('reorient_mask').inputs.orientation == 'RAI'
    assert wf.get_node('outputnode') is not None
```

- [ ] **Step 2: Run to verify failure**

Run: `micromamba run -n linc311 pytest qsiprep/tests/test_workflows_native.py -k template_lps -q`
Expected: FAIL — `ImportError: cannot import name 'init_template_lps_wf'`.

- [ ] **Step 3: Add the sub-workflow**

Add to `qsiprep/workflows/anatomical/volume.py`, next to `init_output_grid_wf`:

```python
def init_template_lps_wf(name='template_lps_wf') -> Workflow:
    """Mask a template and reorient it and its mask to LPS+ (AFNI's ``RAI``).

    QSIPrep writes every anatomical image in LPS+, so every template it registers to
    or resamples into goes through here first.
    """
    workflow = Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['template_file', 'mask_file']), name='inputnode'
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['template_lps', 'mask_lps']), name='outputnode'
    )

    mask_template = pe.Node(
        afni.Calc(expr='a*b', outputtype='NIFTI_GZ'), name='mask_template'
    )
    reorient_brain = pe.Node(
        afni.Resample(orientation='RAI', outputtype='NIFTI_GZ'), name='reorient_brain'
    )
    reorient_mask = pe.Node(
        afni.Resample(orientation='RAI', outputtype='NIFTI_GZ'), name='reorient_mask'
    )

    workflow.connect([
        (inputnode, mask_template, [
            ('template_file', 'in_file_a'),
            ('mask_file', 'in_file_b'),
        ]),
        (inputnode, reorient_mask, [('mask_file', 'in_file')]),
        (mask_template, reorient_brain, [('out_file', 'in_file')]),
        (reorient_brain, outputnode, [('out_file', 'template_lps')]),
        (reorient_mask, outputnode, [('out_file', 'mask_lps')]),
    ])  # fmt:skip

    return workflow
```

- [ ] **Step 4: Use it for the anchor**

In `init_anat_preproc_wf`, delete the `mask_template`, `reorient_tpl_brain_to_lps` and `reorient_tpl_mask_to_lps` nodes (lines 196-206) and replace the connect block at lines 211-220 with:

```python
    anchor_lps_wf = init_template_lps_wf(name='anchor_lps_wf')
    reference_grid_wfs = []  # populated in Task 11

    workflow.connect([
        (get_template, anchor_lps_wf, [
            ('template_file', 'inputnode.template_file'),
            ('mask_file', 'inputnode.mask_file'),
        ]),
    ])  # fmt:skip
```

Every later reference to `reorient_tpl_brain_to_lps` becomes `anchor_lps_wf` with
output `outputnode.template_lps`, and `reorient_tpl_mask_to_lps` becomes
`anchor_lps_wf` output `outputnode.mask_lps`. Find them with:

`grep -n "reorient_tpl_brain_to_lps\|reorient_tpl_mask_to_lps\|mask_template" qsiprep/workflows/anatomical/volume.py`

- [ ] **Step 5: Run to verify pass**

Run: `micromamba run -n linc311 pytest qsiprep/tests/test_workflows_native.py -k template_lps -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add qsiprep/workflows/anatomical/volume.py qsiprep/tests/test_workflows_native.py
git commit -m "refactor: extract the template LPS+ reorientation into a sub-workflow

It was hardcoded to the ACPC anchor. Every requested standard space needs the same
treatment, because QSIPrep writes every anatomical image in LPS+.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 11: Output grid fan-out

**Files:**
- Modify: `qsiprep/workflows/anatomical/volume.py:77-220` (`init_anat_preproc_wf` signature and body)
- Test: `qsiprep/tests/test_workflows_native.py`

**Interfaces:**
- Consumes: `init_output_grid_wf(resolution, name)` from Task 7, `acpc_specs` from Task 9.
- Produces: `init_anat_preproc_wf(..., output_spaces, acpc_anchor, acpc_specs, ...)`; its `outputnode.dwi_sampling_grid` becomes a **list** of grid images, in `acpc_specs` order.

- [ ] **Step 1: Write the failing test**

Append to `qsiprep/tests/test_workflows_native.py`:

```python
def test_one_output_grid_per_acpc_resolution():
    from qsiprep.utils.spaces import parse_output_spaces, select_acpc_anchor
    from qsiprep.workflows.anatomical.volume import init_anat_preproc_wf

    config.workflow.output_spaces = ['acpc:res-2mm', 'acpc:res-1p5mm']
    config.workflow.anat_modality = 'T1w'
    config.workflow.infant = False
    specs = parse_output_spaces(config.workflow.output_spaces)
    acpc_specs = [s for s in specs if not s.standard]

    wf = init_anat_preproc_wf(
        num_anat_images=1,
        num_additional_t2ws=0,
        has_rois=False,
        output_spaces=specs,
        acpc_anchor=select_acpc_anchor(specs),
        acpc_specs=acpc_specs,
        do_biascorr=False,
        t2w_do_biascorr=False,
    )
    names = wf.list_node_names()
    assert any('output_grid_res2mm_wf' in n for n in names)
    assert any('output_grid_res1p5mm_wf' in n for n in names)
```

- [ ] **Step 2: Run to verify failure**

Run: `micromamba run -n linc311 pytest qsiprep/tests/test_workflows_native.py -k output_grid -q`
Expected: FAIL — `TypeError: init_anat_preproc_wf() got an unexpected keyword argument 'output_spaces'`.

- [ ] **Step 3: Change the signature**

In `qsiprep/workflows/anatomical/volume.py`, replace the `anatomical_template` parameter of `init_anat_preproc_wf` with `output_spaces`, `acpc_anchor` and `acpc_specs`, and update the docstring's Parameters block to describe them:

```python
    output_spaces : :obj:`list` of :class:`~qsiprep.utils.spaces.SpaceSpec`
        Every requested output space, with cohorts already resolved.
    acpc_anchor : :class:`~qsiprep.utils.spaces.SpaceSpec`
        The template that anchors ACPC alignment and the output grid's bounding box.
    acpc_specs : :obj:`list` of :class:`~qsiprep.utils.spaces.SpaceSpec`
        The requested ACPC spaces, one per output resolution, in output order.
```

- [ ] **Step 4: Build one grid per ACPC spec**

Replace the `reference_grid_wfs = []` placeholder from Task 10 with:

```python
    # One grid per requested ACPC resolution. The autobox is identical across them;
    # only the final resample differs, so this is cheap.
    for spec in acpc_specs:
        grid_wf = init_output_grid_wf(
            spec.resolution,
            name=f'output_grid_res{spec.resolution.label}_wf',
        )
        if spec.resolution.kind == 'native':
            # Measured from the DWI runs at run time. Under --anat-only there are no
            # DWI files and the grid has no consumer, so fall back to the anchor.
            grid_wf.inputs.inputnode.input_images = dwi_files or [str(anchor_reference)]
        reference_grid_wfs.append(grid_wf)
        workflow.connect([
            (anchor_lps_wf, grid_wf, [('outputnode.template_lps', 'inputnode.template_image')]),
        ])  # fmt:skip

    workflow.connect([
        (
            reference_grid_wfs[0] if len(reference_grid_wfs) == 1 else merge_grids,
            outputnode,
            [('outputnode.grid_image' if len(reference_grid_wfs) == 1 else 'out', 'dwi_sampling_grid')],
        ),
    ])  # fmt:skip
```

That conditional is hard to read. Use a `niu.Merge` unconditionally instead, so
`dwi_sampling_grid` is always a list and downstream code has one shape to handle:

```python
    merge_grids = pe.Node(niu.Merge(len(acpc_specs)), name='merge_grids')
    for index, grid_wf in enumerate(reference_grid_wfs, start=1):
        workflow.connect([
            (grid_wf, merge_grids, [('outputnode.grid_image', f'in{index}')]),
        ])  # fmt:skip

    workflow.connect([
        (merge_grids, outputnode, [('out', 'dwi_sampling_grid')]),
    ])  # fmt:skip
```

`init_anat_preproc_wf` needs `dwi_files` to wire native resolutions. Add it as a
parameter (`dwi_files=None`) and pass `subject_data['dwi']` from `init_single_subject_wf`.
`anchor_reference` is `anchor_lps_wf`'s template output; for the `--anat-only` fallback,
connect it instead of setting a static input:

```python
        if spec.resolution.kind == 'native' and not dwi_files:
            workflow.connect([
                (anchor_lps_wf, grid_wf, [('outputnode.template_lps', 'inputnode.input_images')]),
            ])  # fmt:skip
        elif spec.resolution.kind == 'native':
            grid_wf.inputs.inputnode.input_images = dwi_files
```

- [ ] **Step 5: Run to verify pass**

Run: `micromamba run -n linc311 pytest qsiprep/tests/test_workflows_native.py -k output_grid -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add qsiprep/workflows/anatomical/volume.py qsiprep/workflows/base.py qsiprep/tests/test_workflows_native.py
git commit -m "feat: build one output grid per requested ACPC resolution

dwi_sampling_grid becomes a list in spec order. Native resolutions take the DWI file
list, falling back to the anchor template under --anat-only where the grid is unused.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 12: DWI resampling fan-out and the `res-` entity rule

**Files:**
- Modify: `qsiprep/workflows/dwi/finalize.py:180-330`, `qsiprep/workflows/dwi/derivatives.py`
- Test: `qsiprep/tests/test_output_spaces_naming.py`

**Constraint added after the Task 5 review:** the ACPC DWI datasinks live in
`qsiprep/workflows/dwi/derivatives.py`, not in `finalize.py`, so the `res-` entity rule
is applied there. Task 5's guard calls `init_dwi_derivatives_wf(source_file=...)` with
a single argument and asserts no `res` entity on a single-resolution run. Any
resolution parameter added here MUST be optional with a default that preserves that
call form and that behavior — the guard is not to be edited to accommodate a signature
change.

**Interfaces:**
- Consumes: `dwi_sampling_grid` (list) from Task 11, `init_dwi_trans_wf(resolution=...)` from Task 8.
- Produces: `init_dwi_finalize_wf(..., acpc_specs)`. One `dwi_trans_wf` and one derivatives sink group per ACPC spec.

**The `res-` entity rule:** it is written **only when `len(acpc_specs) > 1`**. A single
ACPC resolution must produce the filenames QSIRecon already expects.

- [ ] **Step 1: Write the failing tests**

Append to `qsiprep/tests/test_output_spaces_naming.py`:

```python
import numpy as np


def _write_dwi(path, nvols=6):
    """Write a tiny valid 4D DWI (with .bval/.bvec) so merge nodes can build."""
    import nibabel as nb

    nb.Nifti1Image(np.zeros((4, 4, 4, nvols), dtype=np.int16), np.eye(4)).to_filename(str(path))
    stem = str(path).split('.nii')[0]
    bvals = np.array([0] + [1000] * (nvols - 1))
    np.savetxt(stem + '.bval', bvals[None, :], fmt='%d')
    np.savetxt(stem + '.bvec', np.zeros((3, nvols)), fmt='%.1f')
    return str(path)


def _build_finalize(tmp_path, output_spaces):
    """Build a finalize workflow. Mirrors the fixture style in test_workflows_native."""
    from qsiprep.tests.preproc_factory import make_preproc_unit
    from qsiprep.utils.spaces import parse_output_spaces
    from qsiprep.workflows.dwi.finalize import init_dwi_finalize_wf

    config.workflow.output_spaces = output_spaces
    config.nipype.omp_nthreads = 1
    specs = parse_output_spaces(output_spaces)
    acpc_specs = [s for s in specs if not s.standard]

    src = _write_dwi(tmp_path / 'sub-01_dwi.nii.gz')
    wf = init_dwi_finalize_wf(
        unit=make_preproc_unit([src]),
        name='dwi_finalize_wf',
        source_file=src,
        output_prefix='sub-01',
        acpc_specs=acpc_specs,
    )
    return wf, acpc_specs


def test_single_acpc_builds_one_trans_wf(tmp_path):
    wf, _ = _build_finalize(tmp_path, ['acpc:res-2mm'])
    prefixes = {n.split('.')[0] for n in wf.list_node_names() if 'dwi_trans_wf' in n}
    assert prefixes == {'dwi_trans_wf'}


def test_two_acpc_resolutions_build_two_trans_wfs(tmp_path):
    wf, _ = _build_finalize(tmp_path, ['acpc:res-2mm', 'acpc:res-1p5mm'])
    prefixes = {n.split('.')[0] for n in wf.list_node_names() if 'dwi_trans_wf' in n}
    assert prefixes == {'dwi_trans_wf_res2mm', 'dwi_trans_wf_res1p5mm'}


def test_two_acpc_resolutions_write_a_res_entity(tmp_path):
    wf, _ = _build_finalize(tmp_path, ['acpc:res-2mm', 'acpc:res-1p5mm'])
    found = collect_datasink_entities(wf)
    acpc_sinks = [e for e in found.values() if e.get('space') == 'ACPC']
    assert acpc_sinks
    assert {e.get('res') for e in acpc_sinks} == {'2mm', '1p5mm'}
```

- [ ] **Step 2: Run to verify failure**

Run: `micromamba run -n linc311 pytest qsiprep/tests/test_output_spaces_naming.py -k trans_wf -q`
Expected: FAIL — `TypeError: init_dwi_finalize_wf() got an unexpected keyword argument 'acpc_specs'`.

- [ ] **Step 3: Fan out the resampling**

In `qsiprep/workflows/dwi/finalize.py`, add `acpc_specs` to `init_dwi_finalize_wf`'s signature. Replace the single `init_dwi_trans_wf` construction and its connection at line 317 with a loop. The grid list arrives on `inputnode.dwi_sampling_grid`, so select by index with a function-connection, since the specs are known at build time:

```python
def _select_grid(grids, index):
    """Pick one output grid out of the list the anatomical workflow produced."""
    return grids[index]
```

```python
    for index, spec in enumerate(acpc_specs):
        label = spec.resolution.label
        suffix = f'_res{label}' if len(acpc_specs) > 1 else ''
        # A single ACPC resolution keeps today's node names and filenames, because
        # QSIRecon consumes them.
        # init_dwi_trans_wf(source_file, mem_gb, template='ACPC', name='dwi_trans_wf', ...)
        # -- keep every argument the existing single call site already passes and add
        # `resolution`; only `name` varies across the loop.
        trans_wf = init_dwi_trans_wf(
            source_file=source_file,
            mem_gb=mem_gb,
            resolution=spec.resolution,
            template='ACPC',
            name=f'dwi_trans_wf{suffix}',
        )
        workflow.connect([
            (inputnode, trans_wf, [
                ((('dwi_sampling_grid'), partial(_select_grid, index=index)),
                 'inputnode.output_grid'),
            ]),
        ])  # fmt:skip
```

Add `from functools import partial` to the module imports.

- [ ] **Step 4: Apply the `res-` entity rule to the sinks**

Every `DerivativesDataSink` in this workflow that writes an ACPC-space DWI output takes
`res=label` when `len(acpc_specs) > 1`, and no `res` at all otherwise. Build the kwargs
once per spec and splat them:

```python
        res_entities = {'res': label} if len(acpc_specs) > 1 else {}
```

then add `**res_entities` to each ACPC DWI sink constructed inside the loop.

- [ ] **Step 5: Record the resolved size in the sidecar**

For `res-native*`, the filename label says `nativemax` but not what it resolved to. Add
the resolved zooms to the sidecar by connecting the grid image into the sink's metadata.
Add to `qsiprep/interfaces/bids.py` usage in this workflow a small node:

```python
def _grid_metadata(grid_file):
    """Report the grid's actual voxel size, which res-native* only fixes at run time."""
    import nibabel as nb

    zooms = [round(float(z), 4) for z in nb.load(grid_file).header.get_zooms()[:3]]
    return {'Resolution': zooms}
```

Connect it into each ACPC DWI sink's `meta_dict` input.

- [ ] **Step 6: Run to verify pass**

Run: `micromamba run -n linc311 pytest qsiprep/tests/test_output_spaces_naming.py -v`
Expected: PASS on the three new tests.

- [ ] **Step 7: Commit**

```bash
git add qsiprep/workflows/dwi/finalize.py qsiprep/tests/test_output_spaces_naming.py
git commit -m "feat: resample DWI once per requested ACPC resolution

The res- entity appears only when more than one ACPC resolution was requested, so a
single-resolution run keeps the filenames QSIRecon expects. Sidecars record the
resolved voxel size, which is the only place res-nativemax reports what it became.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 13: Normalization fan-out and standard-space anatomicals

**Files:**
- Modify: `qsiprep/workflows/anatomical/volume.py:903-1050` (`init_anat_normalization_wf`), `:1387-1600` (`init_anat_derivatives_wf`)
- Test: `qsiprep/tests/test_output_spaces_naming.py`

**Interfaces:**
- Consumes: `standard_specs` from Task 9, `init_template_lps_wf` from Task 10, `templateflow_kwargs` from Task 6.
- Produces: `init_anat_normalization_wf(spec, has_rois=False, name=...)` takes a `SpaceSpec`; `init_anat_derivatives_wf(output_spaces, has_t2w=False)` replaces the `anatomical_template` parameter. This is the signature Task 5's guard expects.

- [ ] **Step 1: Remove the xfail from Task 5's guard**

Delete the two `@pytest.mark.xfail` decorators added in Task 5 Step 3.

- [ ] **Step 2: Write the additional failing tests**

Append to `qsiprep/tests/test_output_spaces_naming.py`:

```python
def test_one_normalization_per_standard_space():
    from qsiprep.utils.spaces import parse_output_spaces
    from qsiprep.workflows.anatomical.volume import init_anat_derivatives_wf

    specs = parse_output_spaces(
        ['acpc:res-2mm', 'MNI152NLin2009cAsym', 'MNI152NLin6Asym']
    )
    wf = init_anat_derivatives_wf(output_spaces=specs)
    found = collect_datasink_entities(wf)
    targets = {e.get('to') for e in found.values() if e.get('from') == 'ACPC'}
    assert 'MNI152NLin2009cAsym' in targets
    assert 'MNI152NLin6Asym' in targets


def test_standard_space_anatomicals_are_written():
    from qsiprep.utils.spaces import parse_output_spaces
    from qsiprep.workflows.anatomical.volume import init_anat_derivatives_wf

    specs = parse_output_spaces(['acpc:res-2mm', 'MNI152NLin6Asym:res-1'])
    wf = init_anat_derivatives_wf(output_spaces=specs)
    found = collect_datasink_entities(wf)
    mni = [e for e in found.values() if e.get('space') == 'MNI152NLin6Asym']
    assert any(e.get('desc') == 'preproc' for e in mni)
    assert any(e.get('suffix') == 'mask' for e in mni)
    assert all(e.get('res') == '1' for e in mni)


def test_bare_standard_space_writes_no_res_entity():
    from qsiprep.utils.spaces import parse_output_spaces
    from qsiprep.workflows.anatomical.volume import init_anat_derivatives_wf

    specs = parse_output_spaces(['acpc:res-2mm', 'MNI152NLin6Asym'])
    wf = init_anat_derivatives_wf(output_spaces=specs)
    found = collect_datasink_entities(wf)
    mni = [e for e in found.values() if e.get('space') == 'MNI152NLin6Asym']
    assert mni
    assert all('res' not in e for e in mni)


def test_cohort_is_a_separate_entity_on_space_but_inline_on_transforms():
    from qsiprep.utils.spaces import parse_output_spaces
    from qsiprep.workflows.anatomical.volume import init_anat_derivatives_wf

    specs = parse_output_spaces(['acpc:res-2mm', 'MNIInfant:cohort-3'])
    wf = init_anat_derivatives_wf(output_spaces=specs)
    found = collect_datasink_entities(wf)

    images = [e for e in found.values() if e.get('space') == 'MNIInfant']
    assert images and all(e.get('cohort') == '3' for e in images)

    transforms = [e for e in found.values() if e.get('from') == 'ACPC' and 'to' in e]
    assert any(e['to'] == 'MNIInfant+3' for e in transforms)
```

- [ ] **Step 3: Run to verify failure**

Run: `micromamba run -n linc311 pytest qsiprep/tests/test_output_spaces_naming.py -q`
Expected: FAIL — `TypeError: init_anat_derivatives_wf() got an unexpected keyword argument 'output_spaces'`.

- [ ] **Step 4: Take a `SpaceSpec` in the normalization workflow**

Change `init_anat_normalization_wf(anatomical_template, has_rois=False)` to
`init_anat_normalization_wf(spec, has_rois=False, name='anat_normalization_wf')`, pass
`name` to `Workflow(name=name)`, and replace the two uses of `anatomical_template`:
the `desc` string at line 967-971 and `anat_nlin_normalization.inputs.template` at
line 1042 both become `spec.fullname`.

- [ ] **Step 5: Build one per standard space**

In `init_anat_preproc_wf`, where the single normalization workflow is built (around line 477), loop over `standard_specs`. Each gets its own `GetTemplate` and `init_template_lps_wf`, so each registers to a correctly-oriented template at its own resolution:

```python
    for spec in standard_specs:
        label = spec.fullname.replace('+', '')
        get_std_template = pe.Node(
            GetTemplate(anatomical_contrast=anat_modality, **templateflow_kwargs(spec)),
            name=f'get_template_{label}',
        )
        std_lps_wf = init_template_lps_wf(name=f'template_lps_{label}_wf')
        norm_wf = init_anat_normalization_wf(
            spec, has_rois=has_rois, name=f'anat_normalization_{label}_wf'
        )
        workflow.connect([
            (get_std_template, std_lps_wf, [
                ('template_file', 'inputnode.template_file'),
                ('mask_file', 'inputnode.mask_file'),
            ]),
            (std_lps_wf, norm_wf, [
                ('outputnode.template_lps', 'inputnode.template_image'),
                ('outputnode.mask_lps', 'inputnode.template_mask'),
            ]),
        ])  # fmt:skip
```

Skip the loop entirely when `standard_specs` is empty — that is how "no standard space
requested" becomes "no normalization runs".

- [ ] **Step 6: Fan out the derivatives**

Change `init_anat_derivatives_wf(anatomical_template, has_t2w=False)` to
`init_anat_derivatives_wf(output_spaces, has_t2w=False)`. Keep every ACPC-space sink
exactly as it is — Task 5's guard fails if any of them change. Replace the two hardcoded
template transform sinks (`ds_t1_mni_warp`, `ds_t1_mni_inv_warp`, lines 1531-1575) with
a loop over the standard specs:

```python
    standard_specs = [spec for spec in output_spaces if spec.standard]
    for spec in standard_specs:
        label = spec.fullname.replace('+', '')
        res_entities = (
            {'res': spec.resolution.label}
            if spec.resolution is not None and spec.resolution.kind == 'label'
            else {}
        )
        cohort_entities = (
            {'cohort': spec.cohort} if spec.cohort not in (None, 'auto') else {}
        )

        # from-/to- labels carry the cohort inline because a transform label has
        # nowhere else to put it; space- pairs with a separate cohort- entity.
        ds_to_template = pe.Node(
            DerivativesDataSink(
                base_directory=config.execution.output_dir,
                to=spec.fullname,
                mode='image',
                suffix='xfm',
                **{'from': 'ACPC'},
            ),
            name=f'ds_t1_{label}_warp',
            run_without_submitting=True,
        )
        ds_from_template = pe.Node(
            DerivativesDataSink(
                base_directory=config.execution.output_dir,
                to='ACPC',
                mode='image',
                suffix='xfm',
                **{'from': spec.fullname},
            ),
            name=f'ds_t1_{label}_inv_warp',
            run_without_submitting=True,
        )
        ds_std_preproc = pe.Node(
            DerivativesDataSink(
                base_directory=config.execution.output_dir,
                compress=True,
                space=spec.space,
                desc='preproc',
                **cohort_entities,
                **res_entities,
            ),
            name=f'ds_t1_{label}_preproc',
            run_without_submitting=True,
        )
        ds_std_mask = pe.Node(
            DerivativesDataSink(
                base_directory=config.execution.output_dir,
                compress=True,
                space=spec.space,
                desc='brain',
                suffix='mask',
                **cohort_entities,
                **res_entities,
            ),
            name=f'ds_t1_{label}_mask',
            run_without_submitting=True,
        )
        ds_std_dseg = pe.Node(
            DerivativesDataSink(
                base_directory=config.execution.output_dir,
                compress=True,
                space=spec.space,
                suffix='dseg',
                **cohort_entities,
                **res_entities,
            ),
            name=f'ds_t1_{label}_dseg',
            run_without_submitting=True,
        )
```

Retain the existing node names `ds_t1_mni_warp` and `ds_t1_mni_inv_warp` for
`MNI152NLin2009cAsym` specifically, so Task 5's guard keeps matching. Do this by
special-casing the name:

```python
        warp_name = (
            'ds_t1_mni_warp' if spec.space == 'MNI152NLin2009cAsym' else f'ds_t1_{label}_warp'
        )
        inv_warp_name = (
            'ds_t1_mni_inv_warp'
            if spec.space == 'MNI152NLin2009cAsym'
            else f'ds_t1_{label}_inv_warp'
        )
```

Connect each `ds_std_*` sink to an `ants.ApplyTransforms` node that resamples the ACPC
anatomical into `spec`'s LPS+ template grid using that spec's forward transform.

Also gate the whole loop on the existing `config.execution.skip_anat_based_spatial_normalization`
checks at lines 1367 and 1606 — those checks become `if standard_specs:`.

- [ ] **Step 7: Update the two call sites**

`init_anat_derivatives_wf` and `init_anat_normalization_wf` are called from
`init_anat_preproc_wf`. Pass `output_spaces=output_spaces` and the per-spec arguments.

- [ ] **Step 8: Run to verify pass**

Run: `micromamba run -n linc311 pytest qsiprep/tests/test_output_spaces_naming.py -v`
Expected: PASS, all 9 tests — including Task 5's two guards, now un-xfailed.

- [ ] **Step 9: Commit**

```bash
git add qsiprep/workflows/anatomical/volume.py qsiprep/tests/test_output_spaces_naming.py
git commit -m "feat: normalize to every requested standard space

Each standard space gets its own TemplateFlow fetch, LPS+ reorientation, nonlinear
registration, transform pair and resampled anatomicals. space- pairs with a separate
cohort- entity while from-/to- keep +cohort inline, matching nibabies. ACPC-space
sinks are untouched, so single-acpc filenames are unchanged.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 14: Reports name every requested space

**Files:**
- Modify: `qsiprep/workflows/anatomical/volume.py:1287-1385` (`init_anat_reports_wf`, `_template_to_report_entities`), `qsiprep/interfaces/reports.py` (`SubjectSummary`)
- Test: `qsiprep/tests/test_reports.py`

**Interfaces:**
- Consumes: `standard_specs` from Task 9.
- Produces: `SubjectSummary(templates=[...])` replacing `template=`; `init_anat_reports_wf(output_spaces)` replacing `anatomical_template`.

- [ ] **Step 1: Write the failing test**

Append to `qsiprep/tests/test_reports.py`:

```python
def test_subject_summary_lists_every_template(tmp_path):
    from qsiprep.interfaces.reports import SubjectSummary

    iface = SubjectSummary(
        templates=['MNI152NLin2009cAsym', 'MNIInfant+3'],
        subject_id='01',
        subjects_dir=None,
    )
    text = iface._generate_segment()
    assert 'MNI152NLin2009cAsym' in text
    assert 'MNIInfant+3' in text
```

- [ ] **Step 2: Run to verify failure**

Run: `micromamba run -n linc311 pytest qsiprep/tests/test_reports.py -k subject_summary_lists -q`
Expected: FAIL — `TraitError: 'templates' is not a trait`.

- [ ] **Step 3: Make `SubjectSummary` take a list**

In `qsiprep/interfaces/reports.py`, change the `template` input trait to
`templates = InputMultiObject(traits.Str)` and join them with `', '` in the
`_generate_segment` output. Handle an empty list by rendering `'none (no standard space requested)'`.

- [ ] **Step 4: Generalize the report entity helper**

Replace `_template_to_report_entities(template)` in `qsiprep/workflows/anatomical/volume.py` with a version taking a `SpaceSpec`:

```python
def _spec_to_report_entities(spec):
    """Convert a SpaceSpec to reportlet filename entities."""
    entities = {'space': spec.space}
    if spec.cohort not in (None, 'auto'):
        entities['cohort'] = spec.cohort
    return entities
```

`init_anat_reports_wf(anatomical_template)` becomes `init_anat_reports_wf(output_spaces)`
and builds one `ds_report_t1_2_mni` node per standard spec, named
`ds_report_t1_2_{label}` with the `MNI152NLin2009cAsym` case keeping the original node
name for continuity.

- [ ] **Step 5: Update the call sites**

`init_anat_reports_wf` is called at `volume.py:365`; `SubjectSummary` at `base.py:278`.
Pass `output_spaces` and `templates=[s.fullname for s in standard_specs]`.

- [ ] **Step 6: Run to verify pass**

Run: `micromamba run -n linc311 pytest qsiprep/tests/test_reports.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add qsiprep/interfaces/reports.py qsiprep/workflows/anatomical/volume.py qsiprep/tests/test_reports.py
git commit -m "feat: name every requested output space in the reports

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 15: Record the bundled fieldmap atlas's provenance

**Files:**
- Modify: `qsiprep/data/NOTICE`

`qsiprep/data/mni_lps_fmap_atlas.nii.gz` is the only NIfTI QSIPrep ships. It is a
fieldmap (values -2.8 to 10.65) used by `syn_sdc_wf` as a registration mask, not a
template, and it has no TemplateFlow equivalent — the nearest candidate,
`tpl-MNI152NLin2009cAsym_res-02_desc-fMRIPrep_boldref`, correlates 0.66 and has no
negative values. `git log --follow` shows it landed in 2018 with no attribution.

- [ ] **Step 1: Add the entry**

Append to `qsiprep/data/NOTICE`:

```
mni_lps_fmap_atlas.nii.gz
-------------------------
An average B0 fieldmap in MNI152NLin2009cAsym space (1 mm, LPS+), used by the
fieldmap-less susceptibility distortion correction workflow as a registration mask
after thresholding and binarization. Derived from the average fieldmap template
described in:

    Treiber JM, White NS, Steed TC, et al. Characterization and Correction of
    Geometric Distortions in 814 Diffusion Weighted Images. PLoS One.
    2016;11(3):e0152472. doi:10.1371/journal.pone.0152472

This file has no TemplateFlow equivalent. Contributing it upstream is tracked
separately; until then it ships with QSIPrep.
```

- [ ] **Step 2: Verify the file is still referenced where the NOTICE says**

Run: `grep -rn "mni_lps_fmap_atlas" --include=*.py qsiprep/`
Expected: one hit, `qsiprep/workflows/fieldmap/syn.py:162`.

- [ ] **Step 3: Commit**

```bash
git add qsiprep/data/NOTICE
git commit -m "docs: record the provenance of the bundled fieldmap atlas

It has shipped since 2018 with no attribution. It is a fieldmap used as a SyN-SDC
registration mask, not a template, and has no TemplateFlow equivalent.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 16: Migrate docs, CI and the remaining tests

**Files:**
- Modify: `docs/usage.rst`, `docs/quickstart.rst`, `docs/installation.rst`, `docs/preprocessing.rst`, `docs/changes.md`
- Modify: `.circleci/*.sh` (12 files)
- Modify: `qsiprep/data/tests/config.toml`, `qsiprep/tests/test_cli.py`, `test_cli_run.py`, `test_gpu_cpu_ratio.py`, `test_utils_misc.py`, `test_interfaces_diffprep.py`, `test_t2w_derivatives.py`, `test_workflows_native.py`

- [ ] **Step 1: Find every remaining reference**

Run:

```bash
grep -rn "output-resolution\|output_resolution\|anatomical-template\|anatomical_template\|skip-anat-based-spatial-normalization" \
  --include=*.py --include=*.rst --include=*.md --include=*.sh --include=*.toml . \
  | grep -v '^\./\.git/' | grep -v docs/superpowers
```

Expected: hits in the files listed above. `qsiprep/cli/parser.py` keeps its deprecated
declarations; everything else migrates.

- [ ] **Step 2: Migrate the test suite**

Replace `config.workflow.anatomical_template = 'MNI152NLin2009cAsym'` with
`config.workflow.output_spaces = ['acpc:res-2mm', 'MNI152NLin2009cAsym']` in
`test_interfaces_diffprep.py:877` and `test_workflows_native.py:69`. Replace
`--output-resolution N` with `--output-spaces acpc:res-Nmm MNI152NLin2009cAsym` in
`test_cli.py` (21 occurrences), `test_cli_run.py`, `test_gpu_cpu_ratio.py`, and
`test_utils_misc.py`. Update `qsiprep/data/tests/config.toml` the same way.
`test_t2w_derivatives.py:43` passes `anatomical_template=` to a workflow builder —
change it to the `output_spaces=` signature from Task 13.

- [ ] **Step 3: Run the non-integration suite**

Run: `micromamba run -n linc311 pytest qsiprep/tests/ -m "not integration" -q`
Expected: PASS. Fix any remaining call-site mismatches.

- [ ] **Step 4: Migrate CI, keeping deprecation coverage**

Convert `--output-resolution N` to `--output-spaces acpc:res-Nmm MNI152NLin2009cAsym` in
all 12 `.circleci/*.sh` scripts, **except** leave `DSDTI_nofmap.sh` and `MultiT1w.sh` on
the old flag deliberately. Add a comment above each so nobody "fixes" them:

```bash
# Deliberately left on the deprecated --output-resolution to keep the forwarding path
# under test until it is removed in 27.0.0.
```

- [ ] **Step 5: Rewrite the docs**

In `docs/usage.rst`, replace the `--output-resolution` discussion with a section on
`--output-spaces` covering: the required `acpc` space, the three `res-` families, that
multiple `acpc` entries produce multiple resampled DWI outputs (and cost roughly N times
the resampling), that standard spaces produce transforms and anatomicals but never
resampled DWI, that N standard spaces means N nonlinear registrations, and
`cohort-auto`. Add a migration table:

| Old | New |
|---|---|
| `--output-resolution 2` | `--output-spaces acpc:res-2mm MNI152NLin2009cAsym` |
| `--output-resolution 1.5` | `--output-spaces acpc:res-1p5mm MNI152NLin2009cAsym` |
| `--output-resolution 2 --infant` | `--output-spaces acpc:res-2mm MNIInfant:cohort-auto` |
| `--output-resolution 2 --skip-anat-based-spatial-normalization` | `--output-spaces acpc:res-2mm` |

Update the example invocations in `quickstart.rst` (6), `installation.rst` (4) and
`preprocessing.rst` (3). Add a `docs/changes.md` entry describing the breaking change
and linking issue #681.

- [ ] **Step 6: Verify the docs build**

Run: `micromamba run -n linc311 python -m sphinx -b html docs docs/_build/html -q 2>&1 | tail -20`
Expected: no errors. Warnings about pre-existing issues are acceptable; anything naming
a file you touched is not.

- [ ] **Step 7: Confirm the migration is complete**

Run the Step 1 grep again. Expected: hits only in `qsiprep/cli/parser.py` (the
deprecated declarations), the two deliberately-unmigrated CI scripts, and
`docs/usage.rst`/`docs/changes.md` (the migration table and changelog).

- [ ] **Step 8: Commit**

```bash
git add docs .circleci qsiprep/data/tests/config.toml qsiprep/tests
git commit -m "docs: migrate docs, CI and tests to --output-spaces

Two CI scripts stay on --output-resolution deliberately, keeping the forwarding path
under test until it is removed in 27.0.0.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Final verification

- [ ] **Run the whole non-integration suite**

Run: `micromamba run -n linc311 pytest qsiprep/tests/ -m "not integration" -q`
Expected: PASS.

- [ ] **Confirm the single-acpc naming guard still passes**

Run: `micromamba run -n linc311 pytest qsiprep/tests/test_output_spaces_naming.py -v`
Expected: PASS. If any ACPC-space filename changed, QSIRecon breaks — stop and fix.

- [ ] **Confirm a legacy invocation still builds a workflow**

Run: `micromamba run -n linc311 pytest qsiprep/tests/test_cli_run.py -q`
Expected: PASS, with deprecation warnings on stderr for the scripts still using the old flags.

- [ ] **Check the dead references are gone**

Run: `grep -rn "init_spaces\|workflow\.spaces\|config\.workflow\.output_resolution\|config\.workflow\.anatomical_template" --include=*.py qsiprep/`
Expected: no hits.
