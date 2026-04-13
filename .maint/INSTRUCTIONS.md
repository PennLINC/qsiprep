# Maintenance instructions for QSIPrep

Run all commands from the repository root unless noted otherwise.

---

## Updating runtime dependencies in Dockerfile.base

`Dockerfile.base` contains non-Python runtime dependencies. These are managed
separately from pixi-managed Python/conda dependencies.

1. Edit `Dockerfile.base` to update runtime tools (FSL/AFNI/MRtrix/DSI Studio/etc.).
2. Bump the date tag in `Dockerfile`:

   ```dockerfile
   ARG BASE_IMAGE=pennlinc/qsiprep-base:<YYYYMMDD>
   ```

3. Commit and push. CircleCI `image_prep` checks if this base tag exists and builds
   `Dockerfile.base` only when the tag is missing.
4. Verify CI image jobs and confirm new base image at
   `pennlinc/qsiprep-base:<YYYYMMDD>`.

---

## Updating Python/conda dependencies

`pyproject.toml` is the source of truth:

- `[project.dependencies]` for PyPI dependencies
- `[tool.pixi.dependencies]` for conda dependencies

After editing, regenerate `pixi.lock` on Linux and commit it.

---

## Lockfile automation

`.github/workflows/pixi-lock.yml` runs on every `pull_request_target` and only
updates lockfile when the latest commit changed `pyproject.toml` or `pixi.lock`.

- Same-repo PR branches: workflow can push lock updates.
- Fork PR branches: workflow does not push lock updates.

---

## CircleCI trigger behavior

CircleCI uses `image_prep` with cache key based on:

- `Dockerfile`
- `pixi.lock`

Base image rebuild is controlled by whether `BASE_IMAGE` in `Dockerfile` exists
in Docker Hub.

- Editing `Dockerfile.base` alone does not force rebuild if base tag already exists.
- To force base rebuild, bump `ARG BASE_IMAGE=...` in `Dockerfile`.
