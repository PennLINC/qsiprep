# Selectable MRtrix3 version (`--mrtrix-version`)

Date: 2026-09-03
Status: approved
Follows: `2026-09-01-complex-mrdegibbs-design.md`

## Problem

The complex-`mrdegibbs` work (issue #1108) moved QSIPrep wholesale onto MRtrix3's
development branch. That buys complex-valued unringing, but it makes every run depend on
unreleased code, and it gives users no way back to a released MRtrix3.

This design keeps both. A new `--mrtrix-version` option selects between a released
MRtrix3 and the development branch. Complex-valued `mrdegibbs` is available only under
the development branch; every other configuration behaves as it does on `main` today.

## Decisions

| Question | Decision |
|---|---|
| Which released MRtrix3? | The existing `pennlinc/qsiprep-mrtrix3` image, reused unchanged |
| Flag values | `stable` and `dev` |
| Default | `stable` |
| Selection mechanism | `PATH` ordering, set from config |
| Version-dependent arguments | An explicit interface trait |

### Why `stable`/`dev` and not version strings

The released image is pinned to a commit SHA (`670e7b06`), not a release tag, so a
`--mrtrix-version 3.0.8` would be an assertion nobody has checked. Version-free names
also mean the CLI vocabulary does not change every time either image is rebuilt. The
concrete versions are reported in the run's metadata and in the visual report, where
they are accurate by construction.

### Why `stable` by default

`stable` reproduces `main`'s behavior for every existing user: same `dwidenoise`, same
`mrdegibbs`, same `dwibiascorrect` arguments. Complex-valued unringing becomes opt-in.

## The compatibility surface

Only four things differ between the two MRtrix3 versions in ways QSIPrep can observe:

| Item | Released (3.0.x) | Development branch |
|---|---|---|
| `dwibiascorrect` ANTs options | `-ants.b`, `-ants.c`, `-ants.s` | `-ants_b`, `-ants_c`, `-ants_s` |
| `mrdegibbs` complex I/O | not supported | supported |
| `mrdegibbs -dimensionality` | absent | present |
| `dwidenoise2` | absent | present |
| `mrinfo` stderr on non-RAS input | quiet | prints a realignment advisory |

`dwidenoise`, `mrcalc`, `mrtransform` and `transformconvert` are identical for QSIPrep's
purposes. The `mrinfo` advisory is already tolerated: `bvec_to_rasb` gates on the return
code, not on stderr being empty.

`dwibiascorrect` is the only difference that breaks *unconditionally*. Its ANTs options
are `usedefault=True`, so they are emitted on every run regardless of any other setting,
and 3.0.x rejects the underscore spelling outright.

### `dwibiascorrect` is a Python script

This is what determines the selection mechanism. `dwibiascorrect` is not a compiled
binary; it shells out to `mrcalc`, `dwiextract`, `mrmath`, `mrconvert` and
`N4BiasFieldCorrection`, resolving each through its own `PATH` lookup. Any mechanism that
controls only the commands QSIPrep invokes directly — absolute `_cmd` paths, for
instance — would leave `dwibiascorrect` pulling helpers from whichever tree happens to be
first on `PATH`, silently mixing versions inside a single node.

`PATH` ordering is therefore the only mechanism that covers the whole surface.

### `main` already ships a mixed container

On `main`, `dwidenoise2` is built from development-branch MRtrix3 in a separate Docker
stage and copied in beside the 3.0.x tree, while `dwibiascorrect` uses the 3.0.x
spelling. A two-tree image generalizes what is already there. It follows that
`--mrtrix-version stable --denoise-method dwidenoise2` stays valid and behaves exactly as
`main` does today; `dwidenoise2` keeps resolving from the development tree because it is
the only tree that has it. This is a deliberate, documented property, not an oversight.

## CLI

```
--mrtrix-version {stable,dev}     (default: stable)
```

Added to the Workflow configuration group, beside `--denoise-method` and
`--unringing-method`. The help text names what `dev` provides: complex-valued
`mrdegibbs`, at the cost of running unreleased code.

No cross-option validation is required in the parser, because `dwidenoise2` — the only
development-branch-exclusive command — resolves from the development tree under either
setting.

One advisory message is emitted at workflow-build time, not at parse time, because
`use_phase` is a per-scan property the parser cannot know. When unringing is `mrdegibbs`,
phase data are present, and the version is `stable`, log that complex-valued unringing is
available under `--mrtrix-version dev`. It fires only where it is actionable.

## Configuration

`config.workflow.mrtrix_version`, default `'stable'`.

A new `workflow.init()` performs the `PATH` reordering. `_Config.load()` already calls
`cls.init()` and swallows `AttributeError`, so this runs in every process that reloads
the config TOML, worker processes included, without touching any call site.

```python
@classmethod
def init(cls):
    roots = {
        'stable': os.getenv('MRTRIX3_STABLE_HOME'),
        'dev': os.getenv('MRTRIX3_DEV_HOME'),
    }
    if not any(roots.values()):
        return          # bare-metal install: one MRtrix3 on PATH, leave it alone
    selected = roots[cls.mrtrix_version]
    if not selected or not Path(selected, 'bin').is_dir():
        raise RuntimeError(
            f'--mrtrix-version {cls.mrtrix_version} was requested, but no MRtrix3 '
            f'installation was found for it.'
        )
    other = roots['dev' if cls.mrtrix_version == 'stable' else 'stable']
    # Drop any existing occurrence of either tree's bin, then prepend:
    # PATH becomes <selected>/bin : <other>/bin : <remaining PATH>
```

Removing the existing entries before prepending matters because the image already bakes
both trees into `PATH`. Without it, `PATH` accumulates duplicates on every reload of the
config. Resolution would still be correct, since the first match wins, but the invariant
the tests assert would not be stateable.

The roots are declared by the image through environment variables rather than hard-coded
in Python, so container paths stay in the Dockerfile where they are chosen.

The resulting order does three things at once:

1. the selected version wins for every command it provides;
2. `dwidenoise2` falls through to the development tree, because only that tree has it;
3. `/opt/3Tissue/bin`, which ships its own 3.0.x `mrdegibbs` and `dwidenoise` and is
   currently merely *appended*, can no longer shadow anything. Ordering is already
   load-bearing today and is asserted only in one Dockerfile `RUN`.

Failing hard when the selected root is declared but missing matters specifically for
`dev`: falling back silently would build the complex workflow path and then hand complex
data to a 3.0.x `mrdegibbs`. When neither variable is set, `init()` leaves `PATH`
untouched but `mrtrix_version` still drives the argument spellings and the workflow
shape, which is the right behavior for a bare-metal development checkout.

`workflow.init()` also records the resolved tree on `config.environment`, so a run's
metadata — the section already written out for issue reports — states which MRtrix3
actually ran. `workflows/base.py` reads it from there when constructing `AboutSummary`,
rather than each interface resolving it independently.

## Interfaces

### `DWIBiasCorrect`

Gains `mrtrix_version = traits.Enum('stable', 'dev', usedefault=True)`, defaulting to
`stable` so a bare `DWIBiasCorrect()` matches `main`. The three ANTs traits keep an
`argstr`, reduced to the placeholder `'%s'`, and their spelling moves into `_format_arg`.
The placeholder is load-bearing: nipype formats only those traits that declare an
`argstr`, so dropping it entirely would silently emit nothing at all.

```python
def _format_arg(self, name, spec, value):
    if name in ('ants_b', 'ants_c', 'ants_s'):
        sep = '_' if self.inputs.mrtrix_version == 'dev' else '.'
        return f'-ants{sep}{name[-1]} {value}'
    return super()._format_arg(name, spec, value)
```

`mrtrix_version` carries no `argstr`, so nipype never emits it as a flag. A trait rather
than a `config` lookup keeps the interface assertable on `.cmdline` in isolation.

### `MRDeGibbs`

No functional change. `-dimensionality` is development-branch-only, but nothing in the
workflow sets it, so there is no live incompatibility; its docstring gains a note that it
requires `--mrtrix-version dev`.

## Workflow

`qsiprep/workflows/dwi/merge.py` reads `config.workflow.*` directly, so nothing needs
threading through signatures.

```python
unring_complex = (
    denoise_complex
    and unringing_method == 'mrdegibbs'
    and config.workflow.mrtrix_version == 'dev'
)
```

and `DWIBiasCorrect(..., mrtrix_version=config.workflow.mrtrix_version)`.

Under `stable`, the existing `_ImageChain.to_magnitude()` call ahead of the unringing
block fires exactly as it does on `main`, so the fallback introduces no new code path —
it is the path that was already there.

The advisory log line described above belongs here, where `use_phase` is known.

## Visual report

Two touch points in `qsiprep/interfaces/reports.py`.

`SubjectSummary` gains `mrtrix_version`, and the subject templates gain a
`{mrtrix_warning}` slot filled only under `dev`:

> **Development-branch MRtrix3.** This run used the MRtrix3 development branch rather
> than a released version, at your request via `--mrtrix-version dev`. Development-branch
> code has not been through a release cycle and may contain bugs. Inspect these outputs
> before relying on them.

rendered as `<div class="alert alert-warning" role="alert">`. Bootstrap alert classes are
confirmed available in the assembled report: `reports-spec.yml` already uses
`class="alert alert-info" role="alert"`. Under `stable` the slot is the empty string.

`SubjectSummary` is instantiated unconditionally in `workflows/base.py`, and its reportlet
lands in the **Summary** section, second from the top of the report, so the banner appears
once per subject and is hard to miss. `DiffusionSummary` would repeat it once per DWI
series without adding information.

`AboutSummary` gains a permanent line naming the resolved MRtrix3 tree, so the report
states which version ran whether or not the warning is shown.

## Container

### `qsiprep_build`

`Dockerfile_MRtrix3` is untouched. QSIRecon builds from `pennlinc/qsiprep-mrtrix3`, so
reusing `26.1.0` as-is keeps the cross-repo blast radius at zero.

`Dockerfile_MRtrix3dev` changes in one respect: build with
`-DCMAKE_INSTALL_RPATH='$ORIGIN/../lib'` and drop its `ENV LD_LIBRARY_PATH`, so the
development tree becomes self-contained the way the 3.0.x classic build already is. With
two trees present, a global library path is a shadowing hazard. Its verification `RUN`
must execute a development-tree binary with `LD_LIBRARY_PATH` cleared, which is what
proves the RPATH took effect.

`26.9.0` was never pushed, so it can be rebuilt under the same tag. Files are staged by
name; `git add -A` is not safe in that repository.

### `Dockerfile.base`

```dockerfile
ARG TAG_MRTRIX3=26.1.0
ARG TAG_MRTRIX3DEV=26.9.0
...
COPY --from=build_mrtrix3    /opt/mrtrix3-latest /opt/mrtrix3-stable
COPY --from=build_mrtrix3dev /opt/mrtrix3        /opt/mrtrix3-dev
ENV MRTRIX3_STABLE_HOME="/opt/mrtrix3-stable" \
    MRTRIX3_DEV_HOME="/opt/mrtrix3-dev" \
    PATH="$PATH:/opt/mrtrix3-stable/bin:/opt/mrtrix3-dev/bin:/opt/3Tissue/bin"
```

Neither tree contributes to a global `LD_LIBRARY_PATH`. The baked order matches
`--mrtrix-version stable`, so the image default and the CLI default agree and
`workflow.init()` has real work to do only when `dev` is selected. New base tag
`20260903`.

### `Dockerfile`

Base image bumped. The `dwidenoise2-build` stage stays deleted, since the development
tree supplies `dwidenoise2`. The verification `RUN` becomes version-explicit:

```dockerfile
test "$(command -v mrdegibbs)"      = "/opt/mrtrix3-stable/bin/mrdegibbs" && \
test "$(command -v dwidenoise)"     = "/opt/mrtrix3-stable/bin/dwidenoise" && \
test "$(command -v dwibiascorrect)" = "/opt/mrtrix3-stable/bin/dwibiascorrect" && \
test "$(command -v dwidenoise2)"    = "/opt/mrtrix3-dev/bin/dwidenoise2" && \
/opt/mrtrix3-dev/bin/mrdegibbs -help | grep -q dimensionality && \
/opt/mrtrix3-stable/bin/dwibiascorrect ants -help > /dev/null
```

The `dwidenoise2` line pins the fall-through the design depends on. The last two prove
the development tree runs without a global library path and that the released tree's
Python wrapper still imports from its own `../lib`.

## Testing

Ordered by what each group would actually catch.

1. **Config `init()`** — parameterized over both versions, with `monkeypatch.setenv`
   pointing at two `tmp_path` roots containing `bin/`. Assert the selected tree is first
   and the other second; a no-op when neither variable is set; `RuntimeError` when the
   selected root is declared but its `bin` is absent.
2. **`DWIBiasCorrect.cmdline`** — `-ants.b`/`-ants.c` under `stable`, `-ants_b`/`-ants_c`
   under `dev`. Inputs must be real files under `tmp_path`; those traits are
   `exists=True` and a string filename raises `TraitError` at construction.
3. **Workflow graph shape** — the existing complex/magnitude tests, parameterized over
   `mrtrix_version`. With phase data, `dwidenoise` and `mrdegibbs`: under `dev` the
   magnitude split follows `mrdegibbs`; under `stable` it precedes it. The standing
   invariant — no complex data reaching `biascorr`, `get_b0s` or `outputnode.dwi_file` —
   must hold under both.
4. **Parser** — the default is `stable`; unknown values are rejected.
5. **Reports** — `SubjectSummary` emits `alert-warning` under `dev` and nothing under
   `stable`.
6. **Container** — the existing end-to-end complex `mrdegibbs` test runs under `dev`; a
   `stable` counterpart asserts `dwibiascorrect` accepts the dot-spelled options. This is
   the only test that can catch the `-ants.b`/`-ants_b` break, because it needs a real
   binary.

## Documentation

- The new option in the usage documentation.
- The complex-unringing paragraph in `docs/preprocessing.rst` qualified as
  development-branch-only.
- The boilerplate sentence about complex-valued unringing emitted only when the complex
  path is actually built, and naming the development branch.
- **No `docs/changes.md` edit.** That file is generated from PR titles at release time;
  the user-facing description belongs in the PR body.

## Open items for implementation

- Confirm that `-DCMAKE_INSTALL_RPATH='$ORIGIN/../lib'` removes the need for a global
  `LD_LIBRARY_PATH`. Docker was unavailable when this was written. If the RPATH approach
  does not hold, the fallback is a thin wrapper script per tree that sets
  `LD_LIBRARY_PATH` before exec'ing the real binary.
- Determine which released MRtrix3 version SHA `670e7b06` corresponds to, for the
  provenance line in `AboutSummary`. If it cannot be established, report the SHA.
