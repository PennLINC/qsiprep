# Complex-Valued `mrdegibbs` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep magnitude+phase DWI data complex-valued through Gibbs unringing instead of reducing it to magnitude immediately after denoising, by shipping a development-branch MRtrix3 whose `mrdegibbs` reads and writes complex data.

**Architecture:** A new `pennlinc/qsiprep-mrtrix3dev` component image in the `qsiprep_build` repo builds MRtrix3 `dev` with the `dwidenoise2` sources dropped in, replacing both the 2024-era MRtrix3 in QSIPrep's base image and the ad-hoc `dwidenoise2-build` stage in QSIPrep's own Dockerfile. In `init_dwi_denoising_wf`, the positional buffer-node chain is replaced by a helper that tracks the current image *and its data domain*, so the `ComplexToMagnitude` split can be deferred to the last step that can still consume complex data.

**Tech Stack:** Docker/BuildKit multi-stage builds, CMake + Ninja (MRtrix3 `dev`), CircleCI, Python 3.11, nipype workflows, pytest.

**Spec:** `docs/superpowers/specs/2026-09-01-complex-mrdegibbs-design.md`

## Global Constraints

- Run every Python command through micromamba: `micromamba run -n linc311 <command>`. Never `pip install` or create environments.
- Two repositories are involved. `qsiprep` is at `/mnt/c/Users/tsalo/Documents/linc/qsiprep` (branch `complex-degibbs`); `qsiprep_build` is at `/mnt/c/Users/tsalo/Documents/linc/qsiprep_build`. Commit to each separately. Never run `git stash` in either.
- MRtrix3 pin: `MRTRIX3_SHA=b98b54e9ae8168eeb9af23322a07011d4754456d`. Do not bump it — it is the commit `dwidenoise2` is developed against, and it already contains the complex `mrdegibbs`.
- dwidenoise2 pin: `DWIDENOISE2_COMMIT=cd08ec1a0f5eb1dbc9962f80c20c2bb3428c4f93` from `https://github.com/tsalo/dwidenoise2.git`.
- New component image tag: `TAG_MRTRIX3DEV=26.9.0` (NiPreps CalVer, `YY.MINOR.PATCH`; the next MINOR after the current `26.8.x` series).
- `pennlinc/qsiprep-mrtrix3` and `qsiprep_build/Dockerfile_MRtrix3` must not change — QSIRecon consumes them.
- `cmake` must be invoked with `-DCMAKE_INSTALL_LIBDIR=lib`. MRtrix3's generated Python command wrappers hardcode `sys.path.insert(0, <exe_dir>/../lib)`; an architecture-triplet libdir breaks every Python command including `dwibiascorrect`.
- Existing node names (`denoiser`, `degibbser`, `biascorr`, `combine_complex`, `split_complex`, `quick_mask`, `get_b0s`, `gradient_table`, `phase_to_radians`) are asserted by tests and must be preserved.
- Tests that need real MRtrix3 binaries use the `nibs_dwi` fixture, which calls `pytest.skip` unless `--data_dir` is passed. They run in the CI container; locally they skip.

## Task order and verification reality

Tasks 1–2 change container images and cannot be verified by running QSIPrep locally; their verification is `docker build` plus in-image `RUN` checks. Tasks 3–7 are pure Python and are fully verifiable locally with `micromamba run -n linc311 python -m pytest`. Task 8 adds a container-only test that will skip locally and run in CI.

Do Tasks 1–2 first anyway: Task 3 changes `dwibiascorrect` arguments to match the new image, so the code and the image must move together.

---

### Task 1: `qsiprep_build` — new `pennlinc/qsiprep-mrtrix3dev` image

**Repository:** `/mnt/c/Users/tsalo/Documents/linc/qsiprep_build`

**Files:**
- Create: `Dockerfile_MRtrix3dev`
- Modify: `setup_build.sh` (tag block near line 12–22, echo block near line 24–41, `do_build` build-args near line 55–68)
- Modify: `.circleci/config.yml` (jobs section near line 76, workflow section near line 209 and 258)

**Interfaces:**
- Produces: Docker image `pennlinc/qsiprep-mrtrix3dev:26.9.0` containing a complete MRtrix3 `dev` install rooted at `/opt/mrtrix3` — `bin/` (C++ commands plus generated Python command wrappers), `lib/mrtrix3/` (Python API and commands), `lib/libmrtrix-core.so`, `share/mrtrix3/` (including `share/mrtrix3/dwidenoise2/` noise-estimation schedules). Task 2 consumes this via `COPY --from=build_mrtrix3 /opt/mrtrix3 /opt/mrtrix3`.

- [ ] **Step 1: Branch the build repo**

```bash
cd /mnt/c/Users/tsalo/Documents/linc/qsiprep_build
git checkout -b mrtrix3-dev
```

- [ ] **Step 2: Write `Dockerfile_MRtrix3dev`**

This absorbs the `dwidenoise2-build` stage currently in `qsiprep/Dockerfile` and extends it from "build two named targets" to "build and install everything". Note the base image is `nvidia/cuda:12.2.2-devel-ubuntu22.04` to match the other component images and the `nvidia/cuda:12.2.2-runtime-ubuntu22.04` runtime, rather than the `buildpack-deps:bookworm` the old stage used.

Create `Dockerfile_MRtrix3dev`:

```dockerfile
# MRtrix3 development branch, with the dwidenoise2 sources built into the tree.
#
# QSIPrep needs the development branch because its mrdegibbs reads and writes
# complex-valued data, which the 3.0.x mrdegibbs in pennlinc/qsiprep-mrtrix3 cannot do.
# That image is left alone because QSIRecon depends on it.
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# MRtrix3 "dev" as at 2026-06-22, the commit dwidenoise2 is developed against.
ARG MRTRIX3_SHA=b98b54e9ae8168eeb9af23322a07011d4754456d
ARG DWIDENOISE2_COMMIT=cd08ec1a0f5eb1dbc9962f80c20c2bb3428c4f93

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        curl \
        g++ \
        git \
        libfftw3-dev \
        libpng-dev \
        libtiff-dev \
        ninja-build \
        pkg-config \
        python3 \
        python-is-python3 \
        zlib1g-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /src/dwidenoise2
RUN git init . && \
    git remote add origin https://github.com/tsalo/dwidenoise2.git && \
    git fetch --depth 1 origin ${DWIDENOISE2_COMMIT} && \
    git checkout --detach FETCH_HEAD

WORKDIR /src/mrtrix3
RUN git clone --filter=blob:none --no-checkout https://github.com/MRtrix3/mrtrix3.git . && \
    git checkout --detach ${MRTRIX3_SHA}

# dwidenoise2 has no external-module build, so its sources are dropped into the MRtrix3
# tree before configuring. The per-command noise estimation schedules must land in
# share/mrtrix3/<command>/, where the built commands look for them relative to the
# executable; MRtrix3's own install() rules carry share/mrtrix3/ to the install prefix.
RUN cp /src/dwidenoise2/cpp/cmd/dwidenoise2.cpp cpp/cmd/dwidenoise2.cpp && \
    cp /src/dwidenoise2/cpp/cmd/dwi2noise.cpp cpp/cmd/dwi2noise.cpp && \
    cp -r /src/dwidenoise2/cpp/core/denoise cpp/core/denoise && \
    cp -r /src/dwidenoise2/share/dwidenoise2/. share/mrtrix3/

# CMAKE_INSTALL_LIBDIR must be plain "lib": the generated Python command wrappers do
# sys.path.insert(0, <exe_dir>/../lib) before importing mrtrix3.app, so an
# architecture-triplet libdir breaks dwibiascorrect and every other Python command.
RUN cmake -B build -GNinja \
          -DMRTRIX_BUILD_GUI=OFF \
          -DMRTRIX_ENABLE_GPU=OFF \
          -DCMAKE_INSTALL_LIBDIR=lib \
          --preset=release && \
    cmake --build build && \
    cmake --install build --prefix /opt/mrtrix3

ENV PATH="/opt/mrtrix3/bin:$PATH" \
    LD_LIBRARY_PATH="/opt/mrtrix3/lib:$LD_LIBRARY_PATH"

# Fail the build here rather than in a user's pipeline. The dwibiascorrect check is the
# important one: it is the only check that exercises the Python wrapper's ../lib import.
RUN mrdegibbs -help | grep -q dimensionality && \
    dwidenoise -version && \
    dwidenoise2 -version && \
    dwibiascorrect ants -help > /dev/null && \
    mrcalc -version && \
    mrtransform -version && \
    transformconvert -version && \
    test -d /opt/mrtrix3/share/mrtrix3/dwidenoise2
```

- [ ] **Step 3: Build the image and verify**

Run:

```bash
cd /mnt/c/Users/tsalo/Documents/linc/qsiprep_build
DOCKER_BUILDKIT=1 BUILDKIT_PROGRESS=plain \
  docker build -f Dockerfile_MRtrix3dev -t pennlinc/qsiprep-mrtrix3dev:26.9.0 .
```

Expected: build succeeds and the final verification `RUN` passes.

Two failures are anticipated and have known fixes:

1. **`CMake 3.22 or higher is required`** — jammy ships CMake 3.22.1 against MRtrix3's `cmake_minimum_required(VERSION 3.22)`, which is the exact floor. If apt's CMake is rejected or too old, replace `cmake` in the apt list with Kitware's:

```dockerfile
RUN curl -fsSL https://apt.kitware.com/keys/kitware-archive-latest.asc \
      | gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg && \
    echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main' \
      > /etc/apt/sources.list.d/kitware.list && \
    apt-get update && apt-get install -y --no-install-recommends cmake && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
```

2. **`dwibiascorrect: No module named 'mrtrix3'`** — the `CMAKE_INSTALL_LIBDIR` guard failed. Check where the Python package actually landed and fix the libdir:

```bash
docker run --rm pennlinc/qsiprep-mrtrix3dev:26.9.0 find /opt/mrtrix3 -name app.py
```

Expected: `/opt/mrtrix3/lib/mrtrix3/app.py`. Anything else means `-DCMAKE_INSTALL_LIBDIR=lib` did not take.

- [ ] **Step 4: Confirm the complex `mrdegibbs` really is in the image**

Run:

```bash
docker run --rm pennlinc/qsiprep-mrtrix3dev:26.9.0 \
  mrdegibbs -help 2>&1 | grep -i "complex-valued"
```

Expected: prints the line "it operates best if it can be provided with complex-valued image data". If this is missing, the wrong MRtrix3 commit was checked out — stop and re-check `MRTRIX3_SHA`.

- [ ] **Step 5: Add the tag to `setup_build.sh`**

Add after the `export TAG_MRTRIX3=26.1.0` line:

```bash
export TAG_MRTRIX3DEV=26.9.0
```

Add after the `echo "TAG_MRTRIX3=${TAG_MRTRIX3}"` line:

```bash
echo "TAG_MRTRIX3DEV=${TAG_MRTRIX3DEV}"
```

Do **not** add a `--build-arg` for it in `do_build`. `do_build` builds the legacy monolithic `pennlinc/qsiprep_build` image, which stays on 3.0.x MRtrix3; only the per-component CircleCI job needs the tag.

- [ ] **Step 6: Add the CircleCI job**

In `.circleci/config.yml`, add immediately after the `build_MRtrix3` job (near line 82):

```yaml
  build_MRtrix3dev:
    environment:
      VERSION_TAG: "TAG_MRTRIX3DEV"
      IMG_NAME: "qsiprep-mrtrix3dev"
      BUILD_FILE: "Dockerfile_MRtrix3dev"
    <<: *build
```

In the `build_test_deploy` workflow, add alongside the existing `- build_MRtrix3:` entry (near line 209), copying whatever `filters` block that entry uses verbatim:

```yaml
      - build_MRtrix3dev:
          filters:
            branches:
              ignore: /.*/
            tags:
              only: /.*/
```

Read the actual `build_MRtrix3` entry first and mirror its filters exactly rather than trusting the block above. Then add `- build_MRtrix3dev` to the `deployable` job's `requires` list (near line 258).

- [ ] **Step 7: Validate the CI config**

Run:

```bash
cd /mnt/c/Users/tsalo/Documents/linc/qsiprep_build
micromamba run -n linc311 python -c "import yaml,sys; d=yaml.safe_load(open('.circleci/config.yml')); assert 'build_MRtrix3dev' in d['jobs']; assert 'build_MRtrix3dev' in [list(j)[0] if isinstance(j,dict) else j for j in d['workflows']['build_test_deploy']['jobs']]; print('ok')"
```

Expected: `ok`

- [ ] **Step 8: Commit**

```bash
cd /mnt/c/Users/tsalo/Documents/linc/qsiprep_build
git add Dockerfile_MRtrix3dev setup_build.sh .circleci/config.yml
git commit -m "Add MRtrix3 development-branch image with dwidenoise2

The development branch's mrdegibbs reads and writes complex-valued data,
which QSIPrep needs to pass denoised complex data straight to unringing.
pennlinc/qsiprep-mrtrix3 is left on 3.0.x because QSIRecon depends on it."
```

---

### Task 2: `qsiprep` — swap the base image and drop the in-repo dwidenoise2 build

**Repository:** `/mnt/c/Users/tsalo/Documents/linc/qsiprep` (branch `complex-degibbs`)

**Files:**
- Modify: `Dockerfile.base:4` (the `TAG_MRTRIX3` ARG), `Dockerfile.base:15` (the `build_mrtrix3` FROM), `Dockerfile.base:39-43` (the MRtrix3 COPY and PATH)
- Modify: `Dockerfile:1-3` (ARGs), `Dockerfile:5-42` (the `dwidenoise2-build` stage), `Dockerfile:80-100` (the `/opt/dwidenoise2` COPYs, ENV and verification RUN)

**Interfaces:**
- Consumes: `pennlinc/qsiprep-mrtrix3dev:26.9.0` from Task 1.
- Produces: a base image in which `mrdegibbs`, `dwidenoise`, `dwidenoise2`, `dwibiascorrect`, `mrcalc`, `mrtransform` and `transformconvert` all resolve to `/opt/mrtrix3/bin`. Tasks 3, 4 and 8 depend on this.

- [ ] **Step 1: Point `Dockerfile.base` at the new component image**

Replace line 4:

```dockerfile
ARG TAG_MRTRIX3=26.1.0
```

with:

```dockerfile
ARG TAG_MRTRIX3DEV=26.9.0
```

Replace line 15:

```dockerfile
FROM pennlinc/qsiprep-mrtrix3:${TAG_MRTRIX3} AS build_mrtrix3
```

with:

```dockerfile
FROM pennlinc/qsiprep-mrtrix3dev:${TAG_MRTRIX3DEV} AS build_mrtrix3
```

Replace lines 39–43:

```dockerfile
COPY --from=build_mrtrix3 /opt/mrtrix3-latest /opt/mrtrix3-latest
## MRtrix3-3Tissue
COPY --from=build_3tissue /opt/3Tissue /opt/3Tissue
ENV PATH="$PATH:/opt/mrtrix3-latest/bin:/opt/3Tissue/bin" \
    MRTRIX3_DEPS="bzip2 ca-certificates curl libpng16-16 libblas3 liblapack3"
```

with:

```dockerfile
COPY --from=build_mrtrix3 /opt/mrtrix3 /opt/mrtrix3
## MRtrix3-3Tissue
COPY --from=build_3tissue /opt/3Tissue /opt/3Tissue
# /opt/3Tissue ships its own 3.0.x copies of mrdegibbs and dwidenoise, so it must stay
# after /opt/mrtrix3 on PATH or it will shadow the development-branch commands.
ENV PATH="$PATH:/opt/mrtrix3/bin:/opt/3Tissue/bin" \
    LD_LIBRARY_PATH="/opt/mrtrix3/lib:$LD_LIBRARY_PATH" \
    MRTRIX3_DEPS="bzip2 ca-certificates curl libpng16-16 libblas3 liblapack3"
```

- [ ] **Step 2: Verify no stale references to the old path remain**

Run:

```bash
cd /mnt/c/Users/tsalo/Documents/linc/qsiprep
grep -rn "mrtrix3-latest\|TAG_MRTRIX3\b" Dockerfile Dockerfile.base
```

Expected: no output.

- [ ] **Step 3: Delete the `dwidenoise2-build` stage from `Dockerfile`**

Delete lines 2–4 (the three ARGs `DWIDENOISE2_COMMIT`, the comment, and `MRTRIX3_DWIDENOISE2_COMMIT`), leaving line 1 (`ARG BASE_IMAGE=...`) in place.

Delete the whole `FROM buildpack-deps:bookworm AS dwidenoise2-build` stage — from that `FROM` line through the `cmake --build build --target dwidenoise2 dwi2noise copy-share-data` line inclusive, ending just before `FROM ghcr.io/prefix-dev/pixi:0.58.0 AS build`.

In the `FROM ${BASE_IMAGE} AS base` stage, delete every `COPY --from=dwidenoise2-build ...` block, the `ENV PATH="/opt/dwidenoise2/bin:$PATH" LD_LIBRARY_PATH="/opt/dwidenoise2/lib:$LD_LIBRARY_PATH"` line, and the `RUN dwidenoise2 -version && test -d /opt/dwidenoise2/share/mrtrix3/dwidenoise2` line. Leave `WORKDIR /home/qsiprep`, `ENV HOME=...`, `RUN chmod -R go=u $HOME` and `WORKDIR /tmp` intact.

- [ ] **Step 4: Add a base-stage verification RUN**

The deleted `RUN dwidenoise2 -version` was the only thing proving `dwidenoise2` was usable in the final image. Replace it, in the same place in the `base` stage, with a check against the new location:

```dockerfile
# Every MRtrix3 command must resolve to the development-branch build, not to the
# 3.0.x copies inside /opt/3Tissue.
RUN dwidenoise2 -version && \
    test "$(command -v mrdegibbs)" = "/opt/mrtrix3/bin/mrdegibbs" && \
    test "$(command -v dwidenoise)" = "/opt/mrtrix3/bin/dwidenoise" && \
    test -d /opt/mrtrix3/share/mrtrix3/dwidenoise2
```

- [ ] **Step 5: Verify the Dockerfile has no dangling references**

Run:

```bash
grep -rn "dwidenoise2-build\|/opt/dwidenoise2\|DWIDENOISE2_COMMIT\|MRTRIX3_DWIDENOISE2_COMMIT" Dockerfile
```

Expected: no output.

- [ ] **Step 6: Build the base image**

Run:

```bash
cd /mnt/c/Users/tsalo/Documents/linc/qsiprep
DOCKER_BUILDKIT=1 BUILDKIT_PROGRESS=plain \
  docker build -f Dockerfile.base -t pennlinc/qsiprep-base:20260901 .
```

Expected: build succeeds.

- [ ] **Step 7: Bump `BASE_IMAGE`**

In `Dockerfile` line 1, replace:

```dockerfile
ARG BASE_IMAGE=pennlinc/qsiprep-base:20260828
```

with:

```dockerfile
ARG BASE_IMAGE=pennlinc/qsiprep-base:20260901
```

CircleCI derives the base tag by grepping this exact line (`.circleci/continue_config.yml:173`), so the format must stay `name:tag` on one line.

- [ ] **Step 8: Build the test image and confirm the commands resolve**

Run:

```bash
DOCKER_BUILDKIT=1 docker build --target test -t pennlinc/qsiprep:test .
docker run --rm pennlinc/qsiprep:test bash -lc \
  'command -v mrdegibbs dwidenoise dwidenoise2 dwibiascorrect && mrdegibbs -help | grep -c complex-valued'
```

Expected: the first four paths are all under `/opt/mrtrix3/bin`, and the grep count is at least 1.

- [ ] **Step 9: Commit**

```bash
cd /mnt/c/Users/tsalo/Documents/linc/qsiprep
git add Dockerfile Dockerfile.base
git commit -m "Build on a development-branch MRtrix3 base image

Replaces the 2024-era MRtrix3 with pennlinc/qsiprep-mrtrix3dev, which also
supplies dwidenoise2, so the ad-hoc dwidenoise2 build stage can go away."
```

---

### Task 3: `dwibiascorrect` ANTs option rename

**Files:**
- Modify: `qsiprep/interfaces/mrtrix.py:447-449`
- Test: `qsiprep/tests/test_interfaces_mrtrix.py`

**Interfaces:**
- Consumes: the image from Task 2, where `dwibiascorrect` is the `dev` version.
- Produces: nothing new; `DWIBiasCorrect`'s trait names (`ants_b`, `ants_c`, `ants_s`) are unchanged, only the argstrs.

MRtrix3 `dev` renamed these options in `python/mrtrix3/commands/dwibiascorrect/ants.py` from dot-separated to underscore-separated. Without this change every bias-correction node fails, including the `b1_biascorrect_stage='final'` nodes in `qsiprep/workflows/dwi/finalize.py:585` and `:638`.

- [ ] **Step 1: Write the failing test**

Append to `qsiprep/tests/test_interfaces_mrtrix.py`:

```python
def test_dwibiascorrect_uses_underscore_ants_options():
    """Pass N4 options as -ants_b/-ants_c/-ants_s.

    MRtrix3's development branch renamed these from the dot-separated -ants.b form
    used by 3.0.x. Getting this wrong fails every bias-correction node at runtime,
    so the exact spelling is pinned here.
    """
    interface = mrtrix.DWIBiasCorrect(method='ants', in_file='dwi.nii.gz', ants_s='4')
    cmdline = interface.cmdline
    assert '-ants_b [150,3]' in cmdline
    assert '-ants_c [200x200,1e-6]' in cmdline
    assert '-ants_s 4' in cmdline
    assert '-ants.' not in cmdline
```

The file already does `from qsiprep.interfaces import mrtrix` at the top, so no new import is needed.

- [ ] **Step 2: Run the test and watch it fail**

Run:

```bash
cd /mnt/c/Users/tsalo/Documents/linc/qsiprep
micromamba run -n linc311 python -m pytest \
  qsiprep/tests/test_interfaces_mrtrix.py::test_dwibiascorrect_uses_underscore_ants_options -v
```

Expected: FAIL — the cmdline contains `-ants.b [150,3]`, so the `-ants_b` assertion fails.

- [ ] **Step 3: Rename the argstrs**

In `qsiprep/interfaces/mrtrix.py`, replace:

```python
    ants_b = traits.Str(default_value='[150,3]', argstr='-ants.b %s', usedefault=True)
    ants_c = traits.Str(default_value='[200x200,1e-6]', argstr='-ants.c %s', usedefault=True)
    ants_s = traits.Str(default_value='4', argstr='-ants.s %s')
```

with:

```python
    # MRtrix3's development branch spells these -ants_b/-ants_c/-ants_s; the
    # dot-separated form used by 3.0.x is rejected outright.
    ants_b = traits.Str(default_value='[150,3]', argstr='-ants_b %s', usedefault=True)
    ants_c = traits.Str(default_value='[200x200,1e-6]', argstr='-ants_c %s', usedefault=True)
    ants_s = traits.Str(default_value='4', argstr='-ants_s %s')
```

- [ ] **Step 4: Run the test and watch it pass**

Run:

```bash
micromamba run -n linc311 python -m pytest \
  qsiprep/tests/test_interfaces_mrtrix.py -v
```

Expected: PASS, including every pre-existing test in the file.

- [ ] **Step 5: Commit**

```bash
git add qsiprep/interfaces/mrtrix.py qsiprep/tests/test_interfaces_mrtrix.py
git commit -m "Use the development-branch spelling of dwibiascorrect ANTs options"
```

---

### Task 4: `MRDeGibbs` — complex-safe report and the `-dimensionality` option

**Files:**
- Modify: `qsiprep/interfaces/mrtrix.py:32-36` (the `from .denoise import` block), `:492-521` (`MRDeGibbsInputSpec`, whose last trait `maxw` starts at :518), `:539` (`MRDeGibbs._generate_report`)
- Test: `qsiprep/tests/test_interfaces_mrtrix.py`

**Interfaces:**
- Produces: `MRDeGibbs` accepts complex-valued `in_file` and generates its report from the magnitude data rather than silently from the real part, and exposes an optional `dimensionality` trait. Task 6 relies on the report being drawn from the magnitude when the workflow hands it complex data.

`MRDeGibbs` overrides `SeriesPreprocReport._generate_report` and, unlike the base implementation, never converts complex data to magnitude. `get_fdata()` on a complex NIfTI does not raise; it discards the imaginary part and returns the real component, so the report would silently be drawn from the real part rather than the magnitude on exactly the data this feature introduces. `_to_magnitude` already exists at `qsiprep/interfaces/denoise.py:26` and is what `SeriesPreprocReport._generate_report` uses.

- [ ] **Step 1: Write the failing tests**

Append to `qsiprep/tests/test_interfaces_mrtrix.py`:

```python
def test_mrdegibbs_dimensionality_is_optional():
    """Leave -dimensionality off unless it is set, so the default stays 2D slice-wise."""
    assert '-dimensionality' not in mrtrix.MRDeGibbs(in_file='dwi.nii.gz').cmdline
    assert '-dimensionality 3' in mrtrix.MRDeGibbs(in_file='dwi.nii.gz', dimensionality=3).cmdline


def test_mrdegibbs_report_handles_complex_input(monkeypatch, tmp_path):
    """Generate the unringing report from complex-valued data.

    mrdegibbs on MRtrix3's development branch emits complex data when it is given
    complex data. nibabel's get_fdata() does not raise on complex images; it silently
    discards the imaginary part and returns the real component, so the report has
    to reduce both images to magnitude first.
    """
    import nibabel as nb
    import numpy as np

    rng = np.random.default_rng(0)
    shape = (8, 8, 4, 3)
    affine = np.eye(4)

    def _write(name, data):
        path = tmp_path / name
        img = nb.Nifti1Image(data, affine)
        img.header.set_data_dtype(data.dtype)
        img.to_filename(path)
        return str(path)

    # A bright, structured magnitude so threshold_img(…, 50) finds a non-empty mask
    magnitude = rng.uniform(100, 400, shape)
    phase = rng.uniform(-np.pi, np.pi, shape)
    in_file = _write('in.nii.gz', (magnitude * np.exp(1j * phase)).astype(np.complex64))
    out_file = _write('out.nii.gz', (magnitude * 0.99 * np.exp(1j * phase)).astype(np.complex64))

    interface = mrtrix.MRDeGibbs(in_file=in_file)
    interface._out_report = str(tmp_path / 'report.svg')
    # Bypass nipype's name_source filename derivation and the NMSE CSV write: this test
    # is about surviving complex inputs, not about how output filenames are built.
    monkeypatch.setattr(
        mrtrix.MRDeGibbs,
        '_get_plotting_images',
        lambda self: (nb.load(in_file), nb.load(out_file), None),
    )
    monkeypatch.setattr(
        mrtrix.MRDeGibbs,
        '_calculate_nmse',
        lambda self, original_nii, corrected_nii: None,
    )

    interface._generate_report()

    assert (tmp_path / 'report.svg').is_file()
```

- [ ] **Step 2: Run the tests and watch them fail**

Run:

```bash
micromamba run -n linc311 python -m pytest \
  qsiprep/tests/test_interfaces_mrtrix.py::test_mrdegibbs_dimensionality_is_optional \
  qsiprep/tests/test_interfaces_mrtrix.py::test_mrdegibbs_report_handles_complex_input -v
```

Expected: the first FAILs with a trait error on `dimensionality`. The second does NOT
fail on its own — `get_fdata()` does not raise on complex data, it warns and silently
returns the real part — which is why that test must assert on the plotted magnitude
values rather than merely on the report file existing.

- [ ] **Step 3: Add the `dimensionality` trait**

In `MRDeGibbsInputSpec`, after the `maxw` trait, add:

```python
    dimensionality = traits.Enum(
        2,
        3,
        argstr='-dimensionality %d',
        desc=(
            'dimensionality of the operation: 2 for the slice-wise method of Kellner et al., '
            '3 for the volume-wise extension of Bautista et al. Left unset, mrdegibbs '
            'defaults to 2.'
        ),
    )
```

Do not give it `usedefault=True` — an unset trait keeps the command line identical to today's.

- [ ] **Step 4: Make the report complex-safe**

Add `_to_magnitude` to the existing `from .denoise import (...)` block at `qsiprep/interfaces/mrtrix.py:32`:

```python
from .denoise import (
    SeriesPreprocReport,
    SeriesPreprocReportInputSpec,
    SeriesPreprocReportOutputSpec,
    _to_magnitude,
)
```

In `MRDeGibbs._generate_report`, replace:

```python
        input_dwi, denoised_nii, _ = self._get_plotting_images()

        # find an image to use as the background
        image_data = input_dwi.get_fdata()
```

with:

```python
        input_dwi, denoised_nii, _ = self._get_plotting_images()

        # mrdegibbs emits complex data when it is given complex data, and the report
        # always shows magnitude. This mirrors SeriesPreprocReport._generate_report,
        # which this method overrides.
        input_dwi = _to_magnitude(input_dwi)
        denoised_nii = _to_magnitude(denoised_nii)

        # find an image to use as the background
        image_data = input_dwi.get_fdata()
```

The rest of the method, including the `self._calculate_nmse(input_dwi, denoised_nii)` call at the end, then operates on magnitude throughout.

- [ ] **Step 5: Run the tests and watch them pass**

Run:

```bash
micromamba run -n linc311 python -m pytest qsiprep/tests/test_interfaces_mrtrix.py -v
```

Expected: PASS, all tests in the file.

- [ ] **Step 6: Commit**

```bash
git add qsiprep/interfaces/mrtrix.py qsiprep/tests/test_interfaces_mrtrix.py
git commit -m "Make the mrdegibbs report complex-safe and expose -dimensionality"
```

---

### Task 5: Replace the buffer-node chain with a domain-tracking helper

This is a **pure refactor**: no behavior changes, and every existing test must still pass. It exists as its own task because the positional buffer indexing is what makes Task 6 error-prone, and a reviewer should be able to accept or reject the restructuring independently of the feature.

**Files:**
- Modify: `qsiprep/workflows/dwi/merge.py` (add `_ImageChain` at module level; rewrite the chaining inside `init_dwi_denoising_wf`, lines 406–685 — `buffernodes = []` is at :406, the final buffernode connection at :684)
- Modify: `qsiprep/tests/test_workflows_merge.py:160-167` (the `buffer01` assertion)

**Interfaces:**
- Produces: module-level class `_ImageChain(workflow, node, field, omp_nthreads)` in `qsiprep/workflows/dwi/merge.py` with attributes `source: tuple[Node, str]` and `is_complex: bool`, and methods `feed(node, field='in_file') -> None`, `advance(node, field='out_file', is_complex=False) -> None`, and `to_magnitude() -> None`. Task 6 uses `is_complex` and `to_magnitude`.

- [ ] **Step 1: Record the current test baseline**

Run:

```bash
cd /mnt/c/Users/tsalo/Documents/linc/qsiprep
micromamba run -n linc311 python -m pytest qsiprep/tests/test_workflows_merge.py -v
```

Expected: 23 tests collected; the ones needing `--data_dir` skip, the rest pass. Note the exact pass/skip counts — the same counts must hold at the end of this task.

- [ ] **Step 2: Add the `_ImageChain` helper**

Insert at module level in `qsiprep/workflows/dwi/merge.py`, immediately before `def init_dwi_denoising_wf(`:

```python
class _ImageChain:
    """Track the image flowing through the denoising steps, and its data domain.

    The denoising steps form a linear chain, each reading the previous step's output.
    Whether that image is complex-valued is not a property of the chain position but of
    which steps have run: ``dwidenoise`` handed complex data emits complex data, and so
    does ``mrdegibbs`` on MRtrix3's development branch, while ``dwibiascorrect`` and
    TORTOISE's ``rpg`` are magnitude-only. Tracking the domain alongside the current
    image lets :func:`to_magnitude` insert the split at the last step that can still
    consume complex data, instead of always splitting right after denoising.
    """

    def __init__(self, workflow, node, field, omp_nthreads):
        self.workflow = workflow
        self.source = (node, field)
        self.is_complex = False
        self._omp_nthreads = omp_nthreads

    def feed(self, node, field='in_file'):
        """Connect the current image to ``field`` on ``node``."""
        source_node, source_field = self.source
        self.workflow.connect([(source_node, node, [(source_field, field)])])

    def advance(self, node, field='out_file', is_complex=False):
        """Make ``field`` on ``node`` the current image."""
        self.source = (node, field)
        self.is_complex = is_complex

    def to_magnitude(self):
        """Reduce the current image to magnitude, if it is not already.

        A no-op on real-valued data, so callers can invoke it before any
        magnitude-only step without checking first. Only one ``split_complex``
        node is ever created, because the first call clears ``is_complex``.
        """
        if not self.is_complex:
            return

        split_complex = pe.Node(
            ComplexToMagnitude(),
            name='split_complex',
            n_procs=self._omp_nthreads,
        )
        self.feed(split_complex, 'complex_file')
        self.advance(split_complex, 'out_file', is_complex=False)
```

- [ ] **Step 3: Replace the buffer-node setup**

In `init_dwi_denoising_wf`, replace this block (currently lines 405–422):

```python
    # Get IdentityInterfaces ready to hold intermediate results
    buffernodes = []

    def get_buffernode():
        num_buffers = len(buffernodes)
        return pe.Node(
            niu.IdentityInterface(fields=['dwi_file']),
            name=f'buffer{num_buffers:02}',
        )

    buffernodes.append(get_buffernode())

    workflow.connect([
        # The first buffernode is the raw file
        (inputnode, buffernodes[0], [('dwi_file', 'dwi_file')]),
        # XXX: Why pass the bval and bvec files through unmodified?
        (inputnode, outputnode, [
            ('bval_file', 'bval_file'),
            ('bvec_file', 'bvec_file'),
        ]),
    ])  # fmt:skip
```

with:

```python
    # The chain starts at the raw input file, in the magnitude domain
    chain = _ImageChain(workflow, inputnode, 'dwi_file', omp_nthreads)

    workflow.connect([
        # XXX: Why pass the bval and bvec files through unmodified?
        (inputnode, outputnode, [
            ('bval_file', 'bval_file'),
            ('bvec_file', 'bvec_file'),
        ]),
    ])  # fmt:skip
```

- [ ] **Step 4: Rewire the denoising step**

Delete the `buffernodes.append(get_buffernode())` line inside `if do_denoise:`.

Replace the complex/magnitude branch at the end of the denoising block (currently lines 555–584):

```python
        # The denoiser's input and output are all that the complex-valued path changes
        if denoise_complex:
            phase_to_radians = pe.Node(
                PhaseToRad(),
                name='phase_to_radians',
                n_procs=omp_nthreads,
            )
            combine_complex = pe.Node(
                PolarToComplex(),
                name='combine_complex',
                n_procs=omp_nthreads,
            )
            split_complex = pe.Node(
                ComplexToMagnitude(),
                name='split_complex',
                n_procs=omp_nthreads,
            )
            workflow.connect([
                (inputnode, phase_to_radians, [('dwi_phase_file', 'phase_file')]),
                (buffernodes[-2], combine_complex, [('dwi_file', 'mag_file')]),
                (phase_to_radians, combine_complex, [('phase_file', 'phase_file')]),
                (combine_complex, denoiser, [('out_file', 'in_file')]),
                (denoiser, split_complex, [('out_file', 'complex_file')]),
                (split_complex, buffernodes[-1], [('out_file', 'dwi_file')]),
            ])  # fmt:skip
        else:
            workflow.connect([
                (buffernodes[-2], denoiser, [('dwi_file', 'in_file')]),
                (denoiser, buffernodes[-1], [('out_file', 'dwi_file')]),
            ])  # fmt:skip

        step_num += 1
```

with:

```python
        # The complex-valued path only changes what feeds the denoiser; the denoiser
        # wiring itself is the same either way.
        if denoise_complex:
            phase_to_radians = pe.Node(
                PhaseToRad(),
                name='phase_to_radians',
                n_procs=omp_nthreads,
            )
            combine_complex = pe.Node(
                PolarToComplex(),
                name='combine_complex',
                n_procs=omp_nthreads,
            )
            workflow.connect([
                (inputnode, phase_to_radians, [('dwi_phase_file', 'phase_file')]),
                (phase_to_radians, combine_complex, [('phase_file', 'phase_file')]),
            ])  # fmt:skip
            chain.feed(combine_complex, 'mag_file')
            chain.advance(combine_complex, 'out_file', is_complex=True)

        chain.feed(denoiser, 'in_file')
        chain.advance(denoiser, 'out_file', is_complex=denoise_complex)
        # Split back to magnitude immediately; Task 6 moves this later for mrdegibbs
        chain.to_magnitude()

        step_num += 1
```

- [ ] **Step 5: Rewire the unringing step**

Delete the `buffernodes.append(get_buffernode())` line inside `if do_unringing:` and replace:

```python
        workflow.connect([
            (buffernodes[-2], degibbser, [('dwi_file', 'in_file')]),
            (degibbser, ds_report_unringing, [('out_report', 'in_file')]),
            (degibbser, buffernodes[-1], [('out_file', 'dwi_file')]),
            (degibbser, merge_confounds, [('nmse_text', f'in{step_num}')]),
        ])  # fmt:skip
```

with:

```python
        workflow.connect([
            (degibbser, ds_report_unringing, [('out_report', 'in_file')]),
            (degibbser, merge_confounds, [('nmse_text', f'in{step_num}')]),
        ])  # fmt:skip
        chain.feed(degibbser, 'in_file')
        chain.advance(degibbser, 'out_file')
```

- [ ] **Step 6: Rewire the bias-correction step and the outputnode**

Delete the `buffernodes.append(get_buffernode())` line inside `if do_biascorr:` and replace:

```python
        workflow.connect([
            (buffernodes[-2], biascorr, [('dwi_file', 'in_file')]),
            (buffernodes[-2], get_b0s, [('dwi_file', 'dwi_series')]),
            (inputnode, get_b0s, [('bval_file', 'bval_file')]),
            (get_b0s, quick_mask, [('b0_series', 'in_files')]),
            (quick_mask, biascorr, [('out_mask', 'mask')]),
            (biascorr, buffernodes[-1], [('out_file', 'dwi_file')]),
```

with:

```python
        # dwibiascorrect is magnitude-only, and so is the mask built from its input
        chain.to_magnitude()
        chain.feed(biascorr, 'in_file')
        chain.feed(get_b0s, 'dwi_series')
        workflow.connect([
            (inputnode, get_b0s, [('bval_file', 'bval_file')]),
            (get_b0s, quick_mask, [('b0_series', 'in_files')]),
            (quick_mask, biascorr, [('out_mask', 'mask')]),
```

leaving the remaining connections in that `workflow.connect` list (`bias_image`, `out_report`, `nmse_text`, `in_bval`, `in_bvec`) untouched, then add after the connect block:

```python
        chain.advance(biascorr, 'out_file')
```

Replace the final chain connection:

```python
    # Connect the final buffernode (the most recent output) to the outputnode
    workflow.connect([(buffernodes[-1], outputnode, [('dwi_file', 'dwi_file')])])
```

with:

```python
    # The workflow always hands downstream steps magnitude data
    chain.to_magnitude()
    chain.feed(outputnode, 'dwi_file')
```

- [ ] **Step 7: Update the test that asserts a buffer-node name**

In `qsiprep/tests/test_workflows_merge.py`, replace:

```python
    # The mask comes from the series feeding bias correction, not the raw data
    get_b0s = workflow.get_node('get_b0s')
    assert {src.name for src, dest, _ in workflow._graph.edges(data=True) if dest is get_b0s} == {
        'inputnode',
        'buffer01',
    }
```

with:

```python
    # The mask comes from the series feeding bias correction, not the raw data
    get_b0s = workflow.get_node('get_b0s')
    assert {src.name for src, dest, _ in workflow._graph.edges(data=True) if dest is get_b0s} == {
        'inputnode',
        'denoiser',
    }
```

- [ ] **Step 8: Check that `niu` is still used**

Removing the buffer nodes may have removed the last use of `niu.IdentityInterface` — it has not, because `inputnode` and `outputnode` still use it. Confirm no unused imports were introduced:

```bash
micromamba run -n linc311 python -m ruff check qsiprep/workflows/dwi/merge.py
```

Expected: no findings. (ruff 0.15.21 is installed in `linc311`.)

- [ ] **Step 9: Run the full merge test suite**

Run:

```bash
micromamba run -n linc311 python -m pytest qsiprep/tests/test_workflows_merge.py -v
```

Expected: the same pass/skip counts recorded in Step 1. This is a refactor — any change in outcome is a bug.

- [ ] **Step 10: Commit**

```bash
git add qsiprep/workflows/dwi/merge.py qsiprep/tests/test_workflows_merge.py
git commit -m "Track the denoising chain's data domain instead of buffer positions

No behavior change. The positional buffernode indexing cannot express that a
step's output type depends on the step before it, which the next commit needs."
```

---

### Task 6: Carry complex data through `mrdegibbs`

**Files:**
- Modify: `qsiprep/workflows/dwi/merge.py` (the `denoise_complex`/`unring_complex` computation at :433-437, the denoising block's `chain.to_magnitude()` from Task 5, the unringing block)
- Test: `qsiprep/tests/test_workflows_merge.py`

**Interfaces:**
- Consumes: `_ImageChain` from Task 5.
- Produces: the graph shape the executable test in Task 8 asserts against.

- [ ] **Step 1: Write the failing tests**

Append to `qsiprep/tests/test_workflows_merge.py`:

```python
def _connections(workflow):
    """Map (source node name, destination node name) to the connected field pairs."""
    return {
        (src.name, dest.name): set(data['connect'])
        for src, dest, data in workflow._graph.edges(data=True)
    }


def _build_denoising_wf(monkeypatch, denoise_method, unringing_method, use_phase):
    """Build (without running) a denoising workflow with the given configuration."""
    monkeypatch.setattr(config.workflow, 'denoise_method', denoise_method)
    monkeypatch.setattr(config.workflow, 'dwi_denoise_window', 5)
    monkeypatch.setattr(config.workflow, 'unringing_method', unringing_method)
    monkeypatch.setattr(config.workflow, 'no_b0_harmonization', True)
    monkeypatch.setattr(config.workflow, 'b0_threshold', 100)
    monkeypatch.setattr(config.nipype, 'omp_nthreads', 1)

    return init_dwi_denoising_wf(
        source_file='sub-01_dwi.nii.gz',
        partial_fourier=1.0,
        phase_encoding_direction='j',
        n_volumes=30,
        use_phase=use_phase,
        do_biascorr=False,
    )


@pytest.mark.parametrize('denoise_method', ['dwidenoise', 'dwidenoise2'])
def test_complex_data_stay_complex_through_mrdegibbs(monkeypatch, denoise_method):
    """Hand mrdegibbs the complex-valued denoised data, and split to magnitude after it.

    mrdegibbs is built on the Fourier shift theorem, so it works better on complex
    data; MRtrix3's development branch reads and writes it.
    """
    workflow = _build_denoising_wf(monkeypatch, denoise_method, 'mrdegibbs', use_phase=True)
    connections = _connections(workflow)

    assert connections[('combine_complex', 'denoiser')] == {('out_file', 'in_file')}
    assert connections[('denoiser', 'degibbser')] == {('out_file', 'in_file')}
    assert connections[('degibbser', 'split_complex')] == {('out_file', 'complex_file')}
    assert connections[('split_complex', 'outputnode')] == {('out_file', 'dwi_file')}
    # The split happens once, after unringing, not before it
    assert ('denoiser', 'split_complex') not in connections


@pytest.mark.parametrize('denoise_method', ['dwidenoise', 'dwidenoise2'])
def test_rpg_unringing_gets_magnitude(monkeypatch, denoise_method):
    """Split to magnitude before rpg unringing, which is TORTOISE and magnitude-only."""
    workflow = _build_denoising_wf(monkeypatch, denoise_method, 'rpg', use_phase=True)
    connections = _connections(workflow)

    assert connections[('denoiser', 'split_complex')] == {('out_file', 'complex_file')}
    assert connections[('split_complex', 'degibbser')] == {('out_file', 'in_file')}
    assert ('degibbser', 'split_complex') not in connections


@pytest.mark.parametrize('unringing_method', ['mrdegibbs', 'rpg', 'none'])
def test_patch2self_never_goes_complex(monkeypatch, unringing_method):
    """Keep patch2self runs entirely in the magnitude domain, whatever the unringing."""
    workflow = _build_denoising_wf(monkeypatch, 'patch2self', unringing_method, use_phase=True)
    node_names = {node.name for node in workflow._get_all_nodes()}

    assert 'combine_complex' not in node_names
    assert 'split_complex' not in node_names


@pytest.mark.parametrize('unringing_method', ['mrdegibbs', 'rpg', 'none'])
def test_magnitude_only_input_never_goes_complex(monkeypatch, unringing_method):
    """Keep magnitude-only runs in the magnitude domain even with a complex-capable denoiser."""
    workflow = _build_denoising_wf(monkeypatch, 'dwidenoise', unringing_method, use_phase=False)
    node_names = {node.name for node in workflow._get_all_nodes()}

    assert 'combine_complex' not in node_names
    assert 'split_complex' not in node_names


@pytest.mark.parametrize('denoise_method', ['dwidenoise', 'dwidenoise2'])
def test_split_follows_the_denoiser_without_unringing(monkeypatch, denoise_method):
    """Split to magnitude right after denoising when no unringing runs."""
    workflow = _build_denoising_wf(monkeypatch, denoise_method, 'none', use_phase=True)
    connections = _connections(workflow)

    assert connections[('denoiser', 'split_complex')] == {('out_file', 'complex_file')}
    assert connections[('split_complex', 'outputnode')] == {('out_file', 'dwi_file')}
```

- [ ] **Step 2: Run the tests and watch them fail**

Run:

```bash
micromamba run -n linc311 python -m pytest \
  qsiprep/tests/test_workflows_merge.py -v -k "complex_through_mrdegibbs or rpg_unringing or never_goes_complex or split_follows"
```

Expected: `test_complex_data_stay_complex_through_mrdegibbs` FAILs with a `KeyError: ('degibbser', 'split_complex')` — today the split sits between the denoiser and the degibbser. The other four should already pass; if any of them fails, Task 5 introduced a regression — fix that before continuing.

- [ ] **Step 3: Compute both complex flags up front**

In `init_dwi_denoising_wf`, replace:

```python
    unringing_method = config.workflow.unringing_method
    do_denoise = denoise_method in ('patch2self', 'dwidenoise', 'dwidenoise2')
    do_unringing = config.workflow.unringing_method in ('mrdegibbs', 'rpg')
    harmonize_b0s = not config.workflow.no_b0_harmonization
```

with:

```python
    unringing_method = config.workflow.unringing_method
    do_denoise = denoise_method in ('patch2self', 'dwidenoise', 'dwidenoise2')
    do_unringing = config.workflow.unringing_method in ('mrdegibbs', 'rpg')
    harmonize_b0s = not config.workflow.no_b0_harmonization

    # Only the dwidenoise variants can denoise complex-valued data. Any other method
    # ignores the phase data and denoises the magnitude data alone.
    denoise_complex = do_denoise and denoise_method.startswith('dwidenoise') and use_phase
    # mrdegibbs is built on the Fourier shift theorem and reads and writes complex data
    # on MRtrix3's development branch, so complex data stay complex through unringing.
    # TORTOISE's rpg is magnitude-only.
    unring_complex = denoise_complex and unringing_method == 'mrdegibbs'
```

Then delete the now-duplicated definition inside `if do_denoise:`:

```python
        # Only the dwidenoise variants can denoise complex-valued data.
        # Any other method ignores the phase data and denoises the magnitude data alone.
        denoise_complex = denoise_method.startswith('dwidenoise') and use_phase
```

- [ ] **Step 4: Defer the split when unringing can consume complex data**

In the denoising block, replace the line added in Task 5:

```python
        # Split back to magnitude immediately; Task 6 moves this later for mrdegibbs
        chain.to_magnitude()
```

with:

```python
        # Hold the complex data if unringing can use them; otherwise split here
        if not unring_complex:
            chain.to_magnitude()
```

- [ ] **Step 5: Let the degibbser propagate the complex domain**

In the unringing block, replace the line added in Task 5:

```python
        chain.advance(degibbser, 'out_file')
```

with:

```python
        chain.advance(degibbser, 'out_file', is_complex=unring_complex)
```

The `chain.to_magnitude()` calls already in the bias-correction block and before the outputnode then insert the split in the right place with no further changes.

- [ ] **Step 6: Run the new tests and watch them pass**

Run:

```bash
micromamba run -n linc311 python -m pytest qsiprep/tests/test_workflows_merge.py -v
```

Expected: PASS (with the usual `--data_dir` skips). The five new tests pass and every pre-existing test still passes.

- [ ] **Step 7: Commit**

```bash
git add qsiprep/workflows/dwi/merge.py qsiprep/tests/test_workflows_merge.py
git commit -m "Pass complex-valued denoised data straight to mrdegibbs

Closes the magnitude round-trip between denoising and unringing when phase
data are available and a dwidenoise variant is the denoising method."
```

---

### Task 7: Methods boilerplate and documentation

**Files:**
- Modify: `qsiprep/workflows/dwi/merge.py` (the `desc` strings in the denoising and unringing blocks)
- Modify: `docs/preprocessing.rst:137`
- Modify: `docs/changes.md`
- Test: `qsiprep/tests/test_workflows_merge.py`

**Interfaces:**
- Consumes: `denoise_complex` and `unring_complex` from Task 6.

The complex branch currently claims "After denoising, the complex-valued data were split back into magnitude and phase", which stops being true when unringing is `mrdegibbs`.

- [ ] **Step 1: Write the failing test**

Append to `qsiprep/tests/test_workflows_merge.py`:

```python
def test_boilerplate_describes_where_the_split_happens(monkeypatch):
    """Say that unringing ran on complex data, and place the split after it."""
    complex_degibbs = _build_denoising_wf(monkeypatch, 'dwidenoise', 'mrdegibbs', use_phase=True)
    assert 'complex-valued' in complex_degibbs.__desc__
    assert 'After denoising, the complex-valued data were split' not in complex_degibbs.__desc__
    # The split is described after unringing is described
    assert complex_degibbs.__desc__.index('Gibbs ringing') < complex_degibbs.__desc__.index(
        'split back into magnitude'
    )

    # rpg is magnitude-only, so the split is still described right after denoising
    complex_rpg = _build_denoising_wf(monkeypatch, 'dwidenoise', 'rpg', use_phase=True)
    assert complex_rpg.__desc__.index('split back into magnitude') < complex_rpg.__desc__.index(
        'Gibbs ringing'
    )
```

- [ ] **Step 2: Run the test and watch it fail**

Run:

```bash
micromamba run -n linc311 python -m pytest \
  qsiprep/tests/test_workflows_merge.py::test_boilerplate_describes_where_the_split_happens -v
```

Expected: FAIL — the "After denoising, the complex-valued data were split" sentence is present.

- [ ] **Step 3: Make the denoising description conditional**

Replace:

```python
            if denoise_complex:
                desc += (
                    'Magnitude and phase DWI data were combined into a complex-valued file, then '
                    f'{mppca_desc}'
                    'After denoising, the complex-valued data were split back into magnitude and '
                    'phase, and the denoised magnitude data were retained. '
                )
            else:
                desc += f'DWI data were {mppca_desc}'
```

with:

```python
            if denoise_complex:
                desc += (
                    'Magnitude and phase DWI data were combined into a complex-valued file, then '
                    f'{mppca_desc}'
                )
                if not unring_complex:
                    desc += (
                        'After denoising, the complex-valued data were split back into '
                        'magnitude and phase, and the denoised magnitude data were retained. '
                    )
            else:
                desc += f'DWI data were {mppca_desc}'
```

- [ ] **Step 4: Describe complex unringing**

Replace:

```python
        if unringing_method == 'mrdegibbs':
            desc += f'{last_step}Gibbs ringing was removed using MRtrix3 [@mrtrix3; @mrdegibbs]. '
```

with:

```python
        if unringing_method == 'mrdegibbs':
            if unring_complex:
                desc += (
                    f'{last_step}Gibbs ringing was removed from the complex-valued data using '
                    'MRtrix3 [@mrtrix3; @mrdegibbs]. The complex-valued data were then split '
                    'back into magnitude and phase, and the magnitude data were retained. '
                )
            else:
                desc += (
                    f'{last_step}Gibbs ringing was removed using MRtrix3 [@mrtrix3; @mrdegibbs]. '
                )
```

- [ ] **Step 5: Run the test and watch it pass**

Run:

```bash
micromamba run -n linc311 python -m pytest qsiprep/tests/test_workflows_merge.py -v
```

Expected: PASS, all tests.

- [ ] **Step 6: Update the user documentation**

In `docs/preprocessing.rst`, after the sentence at line 137–139 about `--unringing-method mrdegibbs`, add:

```rst
When phase data are available and the denoising method is ``dwidenoise`` or
``dwidenoise2``, the complex-valued data are carried through unringing rather than
being reduced to magnitude immediately after denoising. ``mrdegibbs`` is based on the
Fourier shift theorem and operates on complex-valued data directly. The ``rpg``
unringing method works on magnitude data only.
```

- [ ] **Step 7: Add a changelog entry**

Add an entry to the unreleased section of `docs/changes.md` matching the file's existing style:

```markdown
* Pass complex-valued data to `mrdegibbs` when phase data are available (#1108)
```

Read the top of `docs/changes.md` first and match the heading and bullet conventions actually in use. Do not touch anything under `docs/_build/` — those are generated artifacts.

- [ ] **Step 8: Commit**

```bash
git add qsiprep/workflows/dwi/merge.py qsiprep/tests/test_workflows_merge.py docs/preprocessing.rst docs/changes.md
git commit -m "Describe complex-valued unringing in the boilerplate and docs"
```

---

### Task 8: End-to-end test in the container

**Files:**
- Modify: `qsiprep/tests/test_workflows_merge.py` (the `_run_denoising_wf` helper and a new test)

**Interfaces:**
- Consumes: the image from Task 2 and the workflow from Task 6.

`_run_denoising_wf` hardcodes `unringing_method='none'`, so no executable test ever runs `mrdegibbs`. This test is the only thing that proves the development-branch `mrdegibbs` really accepts and emits complex data end to end. It skips locally (no `--data_dir`) and runs in CI.

- [ ] **Step 1: Give `_run_denoising_wf` an unringing parameter**

In `_run_denoising_wf`, change the signature from:

```python
def _run_denoising_wf(
    monkeypatch,
    tmp_path,
    nibs_dwi,
    denoise_method,
    use_phase,
    dwi_denoise_window='auto',
):
```

to:

```python
def _run_denoising_wf(
    monkeypatch,
    tmp_path,
    nibs_dwi,
    denoise_method,
    use_phase,
    dwi_denoise_window='auto',
    unringing_method='none',
):
```

and change the `unringing_method` monkeypatch **inside `_run_denoising_wf` only** from:

```python
    monkeypatch.setattr(config.workflow, 'unringing_method', 'none')
```

to:

```python
    monkeypatch.setattr(config.workflow, 'unringing_method', unringing_method)
```

That exact line appears seven times in the file — six of them are in other tests that
must keep the literal `'none'`. Only the occurrence in `_run_denoising_wf`, immediately
after the `dwi_denoise_window` monkeypatch, changes. Confirm with:

```bash
grep -c "unringing_method', 'none'" qsiprep/tests/test_workflows_merge.py
```

Expected: `6` after the edit (it was 7 before).

Also update its docstring, which currently states that unringing is disabled: change "Unringing, bias correction and b=0 harmonization are all disabled so that only the denoising step is exercised." to "Bias correction and b=0 harmonization are disabled; unringing is off unless ``unringing_method`` says otherwise."

- [ ] **Step 2: Write the failing test**

Append to `qsiprep/tests/test_workflows_merge.py`:

```python
@pytest.mark.parametrize('denoise_method', ['dwidenoise', 'dwidenoise2'])
def test_denoising_wf_complex_mrdegibbs(monkeypatch, tmp_path, nibs_dwi, denoise_method):
    """Run mrdegibbs on complex-valued data and return magnitude.

    This is the only test that proves the MRtrix3 in the image really accepts and
    emits complex data; the graph-shape tests only check the wiring.
    """
    nodes, sink_dir = _run_denoising_wf(
        monkeypatch,
        tmp_path,
        nibs_dwi,
        denoise_method=denoise_method,
        use_phase=True,
        unringing_method='mrdegibbs',
    )

    degibbser = nodes['degibbser']
    degibbs_in = nb.load(degibbser.inputs.in_file)
    assert np.issubdtype(degibbs_in.header.get_data_dtype(), np.complexfloating)

    degibbs_out = nb.load(degibbser.result.outputs.out_file)
    assert np.issubdtype(degibbs_out.header.get_data_dtype(), np.complexfloating)
    assert degibbs_out.shape == degibbs_in.shape

    _assert_denoising_outputs(nodes, sink_dir, nibs_dwi['dwi_file'])
```

`_assert_denoising_outputs` already asserts that the file reaching `outputnode.dwi_file` is not complex, so the magnitude round-trip is covered without repeating the assertion here.

- [ ] **Step 3: Confirm it skips cleanly without data**

Run:

```bash
micromamba run -n linc311 python -m pytest \
  qsiprep/tests/test_workflows_merge.py -v -k complex_mrdegibbs
```

Expected: 2 SKIPPED with "--data_dir was not provided". A FAIL or ERROR here means a collection-time problem — fix it before moving on.

- [ ] **Step 4: Run it for real in the container**

Run:

```bash
cd /mnt/c/Users/tsalo/Documents/linc/qsiprep
docker run --rm -it \
  -v "$PWD:/app-src" \
  -v "$PWD/.circleci/data:/data" \
  pennlinc/qsiprep:test \
  python -m pytest /app-src/qsiprep/tests/test_workflows_merge.py -v -k complex_mrdegibbs --data_dir=/data
```

Expected: 2 PASSED.

The `nibs` dataset must be present under the mounted data directory at
`nibs/sub-22449/ses-01/dwi/`. If it is not, find how CI stages it — check `.circleci/continue_config.yml` and `.circleci/data_versions.txt` for the fetch step — and stage it the same way. If the dataset genuinely cannot be obtained locally, push the branch and let CI run this test rather than skipping the verification.

- [ ] **Step 5: Run the whole unit suite one last time**

Run:

```bash
micromamba run -n linc311 python -m pytest qsiprep/tests/ -v
```

Expected: no failures. Report the exact pass/skip/fail counts.

- [ ] **Step 6: Commit**

```bash
git add qsiprep/tests/test_workflows_merge.py
git commit -m "Run mrdegibbs on complex data end to end in the container tests"
```

---

## Final verification

- [ ] `qsiprep_build` branch `mrtrix3-dev` has one commit adding `Dockerfile_MRtrix3dev`, the `TAG_MRTRIX3DEV` tag and the `build_MRtrix3dev` CI job, with `Dockerfile_MRtrix3` untouched:

```bash
cd /mnt/c/Users/tsalo/Documents/linc/qsiprep_build
git diff --stat main...mrtrix3-dev
git diff main...mrtrix3-dev -- Dockerfile_MRtrix3
```

Expected: the second command prints nothing.

- [ ] `qsiprep` branch `complex-degibbs` contains the spec, the plan and six implementation commits:

```bash
cd /mnt/c/Users/tsalo/Documents/linc/qsiprep
git log --oneline main..complex-degibbs
```

- [ ] No reference to the removed paths survives anywhere in the repo:

```bash
grep -rnE "mrtrix3-latest|/opt/dwidenoise2|[-]ants[.][bcs] " --include="*.py" --include="Dockerfile*" . | grep -v docs/_build
```

Expected: no output.

- [ ] The full unit suite passes:

```bash
micromamba run -n linc311 python -m pytest qsiprep/tests/ -q
```

## Rollout, after the plan is executed

The image dependency chain forces this order, and it cannot be done from the branch alone:

1. Merge and tag `qsiprep_build` as `26.9.0` so CircleCI builds and pushes `pennlinc/qsiprep-mrtrix3dev:26.9.0`.
2. Push `pennlinc/qsiprep-base:20260901` built from the updated `Dockerfile.base`.
3. Open the QSIPrep PR. It cannot go green until both images are on Docker Hub, which is why the `BASE_IMAGE` bump and the code changes ship in one PR rather than being split.
