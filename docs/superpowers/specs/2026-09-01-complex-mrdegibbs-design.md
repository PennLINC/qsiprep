# Pass complex-valued data to `mrdegibbs`

Design for [PennLINC/qsiprep#1108](https://github.com/PennLINC/qsiprep/issues/1108).

## Problem

When phase data are available and the denoising method is `dwidenoise` or `dwidenoise2`,
QSIPrep combines magnitude and phase into a complex-valued image, denoises it, and then
immediately discards the phase. Gibbs unringing therefore runs on magnitude data alone.

`mrdegibbs` is built on the Fourier shift theorem, so it works better on complex-valued
input. The version in the MRtrix3 development branch supports this: it reads through
`Image<complex_type>` and writes `CFloat32` when the input is complex. The version QSIPrep
currently ships hardcodes `Float32` output.

The fix has three parts: build a development-branch MRtrix3, put it in the QSIPrep image,
and keep the complex data alive one step longer in the denoising workflow.

## Current state

### Two MRtrix3 builds already ship in the image

`qsiprep_build/Dockerfile_MRtrix3` builds MRtrix3 at SHA `670e7b06` (2024-01-10, the
pre-CMake `./configure && ./build` era) and publishes `pennlinc/qsiprep-mrtrix3`.
`qsiprep/Dockerfile.base` copies it to `/opt/mrtrix3-latest`.

Separately, `qsiprep/Dockerfile` carries a `dwidenoise2-build` stage that clones MRtrix3
`dev` at `b98b54e9` (2026-06-21), drops the `dwidenoise2` sources into the tree, builds
only the `dwidenoise2`, `dwi2noise` and `copy-share-data` targets, and extracts two
binaries, `libmrtrix-core.so` and the noise-estimation schedules into `/opt/dwidenoise2`.

So the image already contains a development-branch MRtrix3; it is just hidden inside the
application repo and stripped down to two commands.

### `b98b54e9` already has the complex `mrdegibbs`

Verified against the pinned commit: `cpp/cmd/mrdegibbs.cpp` at `b98b54e9` contains
`get_image<complex_type>()` and the "operates best if it can be provided with
complex-valued image data" note. **No new MRtrix3 pin is needed**, and `dwidenoise2`
compatibility is therefore unchanged.

### `pennlinc/qsiprep-mrtrix3` is shared with QSIRecon

`qsirecon/Dockerfile.base:14` and `qsirecon_build/Dockerfile:16` both consume it. QSIRecon's
MRtrix3 surface (response estimation, FOD fitting, tractography, MRtrix3Tissue interop) is
far larger than QSIPrep's and must not be moved to `dev` as a side effect of this issue.

### QSIPrep's MRtrix3 surface is small

Since reconstruction moved to QSIRecon, QSIPrep invokes only `dwidenoise`, `dwidenoise2`,
`dwibiascorrect`, `mrcalc`, `mrdegibbs`, `mrtransform` and `transformconvert`. Each was
checked against `dev`:

| Command | Status on `dev` |
| --- | --- |
| `dwidenoise` | `-extent`, `-noise`, `-mask` unchanged |
| `mrcalc` | `-polar`, `-abs` unchanged |
| `mrtransform` | `-strides` unchanged |
| `transformconvert` | unchanged |
| `mrdegibbs` | `-axes`, `-nshifts`, `-minW`, `-maxW` unchanged; adds `-dimensionality`; accepts complex |
| `dwibiascorrect ants` | **breaking**: `-ants.b/-ants.c/-ants.s` renamed to `-ants_b/-ants_c/-ants_s` |
| `BZeroThreshold` config key | unchanged |

## Design

### 1. `qsiprep_build`: new `pennlinc/qsiprep-mrtrix3dev` image

Add `Dockerfile_MRtrix3dev`, built on `nvidia/cuda:12.2.2-devel-ubuntu22.04` — the same
base as the other component images, so the binaries link against the same glibc the
runtime image provides. (The existing `dwidenoise2-build` stage builds on
`buildpack-deps:bookworm`, glibc 2.36, and copies into a jammy runtime, glibc 2.35. It
works today, but it is a forward-compatibility coin flip that should not be preserved.)

The new image absorbs the whole `dwidenoise2-build` stage:

- Two pinned ARGs: `MRTRIX3_SHA=b98b54e9ae8168eeb9af23322a07011d4754456d` and
  `DWIDENOISE2_COMMIT=cd08ec1a0f5eb1dbc9962f80c20c2bb3428c4f93`, carried over verbatim
  from `qsiprep/Dockerfile`.
- Copy `cpp/cmd/dwidenoise2.cpp`, `cpp/cmd/dwi2noise.cpp`, `cpp/core/denoise/` and
  `share/dwidenoise2/.` (into `share/mrtrix3/`) from the `dwidenoise2` checkout into the
  MRtrix3 tree, exactly as today.
- Configure with `-GNinja --preset=release -DMRTRIX_BUILD_GUI=OFF -DMRTRIX_ENABLE_GPU=OFF`,
  build **all** targets rather than a named subset, then
  `cmake --install build --prefix /opt/mrtrix3`.
- Pass **`-DCMAKE_INSTALL_LIBDIR=lib`**. MRtrix3's generated Python command wrappers do
  `sys.path.insert(0, <exe_dir>/../lib)` and then import `mrtrix3.app`, while the Python
  package installs to `${CMAKE_INSTALL_LIBDIR}/mrtrix3`. If `GNUInstallDirs` resolves
  `LIBDIR` to an architecture triplet, every Python command — including `dwibiascorrect` —
  breaks at import.
- Drop `-DCMAKE_COMPILE_WARNING_AS_ERROR=ON`. It is appropriate for developing
  `dwidenoise2` but turns any new GCC 11 warning in unrelated MRtrix3 code into a failed
  image build.

The full install replaces today's hand-copying: `install()` rules place `share/mrtrix3/`
under `${prefix}/share/mrtrix3`, which is exactly where the `dwidenoise2` schedules must
live for the binaries to find them, so the `copy-share-data` target is no longer needed.

Verification `RUN`s inside the image, so a broken build fails at build time rather than in
a user's pipeline:

```
RUN mrdegibbs -help | grep -q dimensionality && \
    dwidenoise2 -version && \
    dwibiascorrect ants -help > /dev/null && \
    test -d /opt/mrtrix3/share/mrtrix3/dwidenoise2
```

`dwibiascorrect ants -help` is the one that matters most — it is the only check that
exercises the Python wrapper path and the `LIBDIR` question above.

**Build-verify risk**: jammy ships CMake 3.22.1 against MRtrix3's
`cmake_minimum_required(VERSION 3.22)`. That is the exact floor with no headroom. If it
fails, install CMake from Kitware's apt repository in the stage.

Supporting changes in `qsiprep_build`:

- `setup_build.sh`: add `TAG_MRTRIX3DEV`, its `echo` line, and its `--build-arg`.
- `.circleci/config.yml`: add a `build_MRtrix3dev` job (a copy of `build_MRtrix3` with
  `VERSION_TAG: TAG_MRTRIX3DEV`, `IMG_NAME: qsiprep-mrtrix3dev`,
  `BUILD_FILE: Dockerfile_MRtrix3dev`) and register it in the `build_test_deploy` workflow
  and the `deployable` requires list.

`Dockerfile_MRtrix3` and `pennlinc/qsiprep-mrtrix3:26.1.0` are **left untouched**, so
QSIRecon keeps its existing pin.

### 2. `qsiprep`: base image and application Dockerfile

`Dockerfile.base`:

- Replace `ARG TAG_MRTRIX3=26.1.0` with `ARG TAG_MRTRIX3DEV=26.9.0` (the repo tags components `YY.M.patch`; `26.9.0` is the next free one).
- `FROM pennlinc/qsiprep-mrtrix3dev:${TAG_MRTRIX3DEV} AS build_mrtrix3`.
- `COPY --from=build_mrtrix3 /opt/mrtrix3 /opt/mrtrix3`.
- Change the PATH entry `/opt/mrtrix3-latest/bin` to `/opt/mrtrix3/bin`. `/opt/3Tissue/bin`
  stays **after** it, so MRtrix3Tissue's 3.0.x copies of `mrdegibbs` and `dwidenoise`
  cannot shadow the development-branch ones.

`Dockerfile`:

- Delete the entire `dwidenoise2-build` stage, the `DWIDENOISE2_COMMIT` and
  `MRTRIX3_DWIDENOISE2_COMMIT` ARGs, every `/opt/dwidenoise2` `COPY --from`, the
  `PATH`/`LD_LIBRARY_PATH` additions for it, and its verification `RUN` — roughly 40 lines.
- Bump `BASE_IMAGE` to the newly built base tag (`pennlinc/qsiprep-base:<YYYYMMDD>` of the build date, matching the existing `20260828` convention).

`/opt/3Tissue` is retained. Nothing in QSIPrep imports it, so removing it is a defensible
cleanup, but it is out of scope here.

### 3. Interfaces (`qsiprep/interfaces/mrtrix.py`)

**`DWIBiasCorrectInputSpec`** — rename the three ANTs pass-through argstrs:

```python
ants_b = traits.Str(default_value='[150,3]', argstr='-ants_b %s', usedefault=True)
ants_c = traits.Str(default_value='[200x200,1e-6]', argstr='-ants_c %s', usedefault=True)
ants_s = traits.Str(default_value='4', argstr='-ants_s %s')
```

This is the only genuinely breaking change in QSIPrep's MRtrix3 surface. Without it, bias
correction fails on every run — including the `b1_biascorrect_stage='final'` calls in
`qsiprep/workflows/dwi/finalize.py`.

**`MRDeGibbsInputSpec`** — add an optional `dimensionality` trait:

```python
dimensionality = traits.Enum(2, 3, argstr='-dimensionality %d',
                             desc='2 for slice-wise (Kellner), 3 for volume-wise (Bautista)')
```

Left unset, so the default stays 2 and behavior is unchanged. It is free to expose and the
option is new in `dev`.

The existing `out_file` name template (`%s_mrdegibbs.nii.gz`) needs no change: NIfTI
represents `CFloat32` fine, and `mrdegibbs` selects the output datatype from the input.

**`MRDeGibbs._generate_report`** — this override does not call `_to_magnitude`, unlike the
`SeriesPreprocReport._generate_report` it replaces, so it would crash on complex input at
`get_fdata()`. Convert both `input_dwi` and `denoised_nii` through
`_to_magnitude` (already defined at `qsiprep/interfaces/denoise.py:26`) immediately after
`_get_plotting_images()`, before the intensity scan, the plotting and the
`_calculate_nmse` call.

### 4. Workflow (`qsiprep/workflows/dwi/merge.py`)

`init_dwi_denoising_wf` currently chains steps through a list of `IdentityInterface`
buffer nodes wired by position (`buffernodes[-2]` → step → `buffernodes[-1]`). That
encoding cannot express what this change needs: a step's output *type* now depends on
which step ran before it.

Replace the positional indexing with a small helper local to the function that tracks both
the current image source and whether it is complex-valued — for example a
`current = (node, field, is_complex)` triple advanced by each step, with a
`to_magnitude()` that inserts a `ComplexToMagnitude` node only when the domain actually
needs to change. Keep `init_dwi_denoising_wf` as a single function; splitting each step
into its own sub-workflow is a larger change than this issue warrants and would obscure
the complex-domain review.

Resulting data flow when `use_phase` is true and `denoise_method` starts with `dwidenoise`:

```
magnitude ─┐
           ├─> PolarToComplex ─> denoiser ─> [complex] ─> degibbser ─> ComplexToMagnitude ─> biascorr ─> ...
phase ─────┘   (via PhaseToRad)
```

The `ComplexToMagnitude` node moves to the last point that still has a complex-capable
consumer:

| Denoiser | `unringing_method` | Split to magnitude |
| --- | --- | --- |
| `dwidenoise`/`dwidenoise2` + phase | `mrdegibbs` | after the degibbser |
| `dwidenoise`/`dwidenoise2` + phase | `rpg` | after the denoiser (TORTOISE is magnitude-only) |
| `dwidenoise`/`dwidenoise2` + phase | `none` | after the denoiser |
| `patch2self`, or no phase | any | never enters the complex domain |

Unchanged: `patch2self` and magnitude-only runs never build a complex image; the noise
image still comes straight off the denoiser; `outputnode.dwi_file` is always magnitude;
`dwibiascorrect` always receives magnitude.

**Boilerplate.** The complex branch of `__desc__` currently reads "After denoising, the
complex-valued data were split back into magnitude and phase, and the denoised magnitude
data were retained." That stops being true when unringing is `mrdegibbs`. Make the
sentence conditional on where the split lands, and note in the `mrdegibbs` description
that unringing was performed on complex-valued data when it was.

### 5. Tests

`qsiprep/tests/test_workflows_merge.py` sets `unringing_method='none'` in **every** test,
so the `mrdegibbs` path has no workflow coverage at all today. That gap has to close
before the rewiring can be trusted.

Graph-shape tests (fast, no container work) over
`{denoise_method} x {unringing_method} x {use_phase}`, asserting:

- `combine_complex` exists exactly when the denoiser is complex-capable and phase exists;
- `split_complex` is connected downstream of `degibbser` for
  `dwidenoise*` + `mrdegibbs` + phase;
- `split_complex` is connected downstream of `denoiser` for `rpg` and for `none`;
- neither node exists for `patch2self` or for magnitude-only input.

Executable test: extend the `_run_denoising_wf` harness (which already runs real MRtrix3
in the test container) with an `mrdegibbs` + `use_phase=True` case asserting that the
degibbser's input file is complex-valued, that its output is complex-valued, and that the
file reaching `outputnode.dwi_file` is real-valued and finite.

Interface test: pin `-ants_b` in a `DWIBiasCorrect` cmdline assertion so the rename cannot
silently regress.

### 6. Documentation

- `docs/preprocessing.rst:137` — note that unringing runs on complex-valued data when
  phase data are available and a `dwidenoise` variant is the denoising method.
- `docs/changes.md` — changelog entry.
- The generated `docs/_build/html` copies are build artifacts and are not edited.

## Rollout

The order is forced by the image dependency chain:

1. Tag `qsiprep_build` so CircleCI builds and pushes `pennlinc/qsiprep-mrtrix3dev:<tag>`.
2. Rebuild and push `pennlinc/qsiprep-base` from the updated `Dockerfile.base`.
3. Open the QSIPrep PR with the `BASE_IMAGE` bump and the code changes together.

The QSIPrep PR cannot pass CI until the new base image exists, so steps 2 and 3 land in a
single PR rather than being split.

## Out of scope

- Moving QSIRecon to a development-branch MRtrix3.
- Retaining or writing out the denoised, unringed **phase** data. The complex image is
  still reduced to magnitude before the rest of preprocessing; only the point at which
  that happens moves.
- Feeding complex data to `mrdegibbs` when the denoiser was magnitude-only
  (`patch2self`) or when no denoising ran. MRtrix3 recommends complex input generally, but
  building a complex image in branches that have never had one is a separate change with
  its own defensible-default argument.
- Removing `/opt/3Tissue` from the QSIPrep base image.
