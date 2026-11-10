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

`.github/workflows/pixi-lock.yml` runs on every `pull_request` event. Both of its
jobs skip their pixi steps unless the PR changed `pyproject.toml` or `pixi.lock`
anywhere in its history (`git diff base...HEAD`), not only in the latest commit.

- **Fork PRs (`validate-lockfile`):** read-only permissions, no persisted
  credentials. Runs `pixi lock --check` and never pushes. If the check fails,
  regenerate `pixi.lock` on Linux and push it from the contributor branch.
- **Same-repo PRs (`update-lockfile`, the only job with `contents: write`):**
  - Checks out the PR head with `persist-credentials: false`.
  - Runs `pixi lock --check`, records `stale=true|false`, and restores the
    committed file with `git checkout -- pixi.lock`. The check rewrites the
    workspace package's hatch-vcs version (which changes with every commit) even
    when it passes, so without the reset that rewrite would be committed.
  - Only if the check failed: sets the `nipreps[bot]` identity, regenerates the
    lockfile with `uvx datalad run -i pixi.lock -i pyproject.toml -o pixi.lock --
    pixi lock`, and pushes the commit to the PR branch (non-forced). The push step
    is the only one that receives the token, as an `http.extraheader` on the
    `git push` command line.
  - Any failure of `pixi lock --check` counts as stale, including a transient
    one. A PR that deletes `pixi.lock` fails at the reset rather than being
    regenerated.
  - Keeping the token out of the pixi and DataLad steps is not full isolation.
    Code those steps run (the PR's build backend and its dependencies) can still
    tamper with the later push step, for example through a git hook,
    `$GITHUB_ENV` or `$GITHUB_PATH`. Full isolation needs a second
    `workflow_run` workflow that receives the lockfile as an artifact, as in
    nipreps/fmriprep#3722.
  - GitHub holds the checks for the bot's commit as "action required" until a
    maintainer approves them.
- **`check-lockfile`:** fails unless exactly the applicable job above succeeded and
  the other was skipped. Require this job, not the other two, as the status check.

Third-party actions in this workflow are pinned to commit SHAs with a `# vX.Y.Z`
comment.

---

## CircleCI trigger behavior

CircleCI uses `image_prep` with cache key based on:

- `Dockerfile`
- `pixi.lock`

Base image rebuild is controlled by whether `BASE_IMAGE` in `Dockerfile` exists
in Docker Hub.

- Editing `Dockerfile.base` alone does not force rebuild if base tag already exists.
- To force base rebuild, bump `ARG BASE_IMAGE=...` in `Dockerfile`.
