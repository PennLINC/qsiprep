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

Lockfile automation is split across two workflows, adapted from nipreps/fmriprep#3722,
so that the PR's code never runs in a job that can write to the repository.

**`.github/workflows/pixi-lock.yml` ("Pixi lock")** is triggered on every
`pull_request` event (not `pull_request_target`) and has one job, `check-lockfile`,
with a read-only token and no persisted credentials. It runs the PR's code (pixi and
DataLad build the workspace package).

- Always runs `Checkout pull request` and `Check pull request for dependency edits`.
  The latter looks for `pyproject.toml` or `pixi.lock` edits anywhere in the PR
  (`git diff base...HEAD`), not only in the latest commit. Editing other files alone
  still starts the workflow, but the pixi steps are skipped and the job passes.
- If the PR touched those files, runs `prefix-dev/setup-pixi` and `Check lockfile`,
  which runs `pixi lock --check`, records `stale=true|false`, and resets `pixi.lock`
  with `git checkout -- pixi.lock`. If the PR deletes `pixi.lock`, the step fails
  with an error asking the author to restore it; a deleted lockfile is never
  regenerated, because the push workflow accepts only a patch that modifies
  `pixi.lock`, not one that creates it.
- If the lockfile is stale and the PR is from a branch of this repository, runs
  `Set git identity`, `Install the latest version of uv`, `Update lockfile`
  (`uvx datalad run -i pixi.lock -i pyproject.toml -o pixi.lock -- pixi lock`),
  `Export lockfile patch` (`git format-patch -1` plus the PR number), and
  `Upload lockfile patch` (artifact `pixi-lock`, kept for one day). The export step
  fails instead if `datalad run` made no commit, which happens when the check failed
  for a reason other than a stale lockfile.
- If the lockfile is stale, `Fail on stale lockfile` fails the job, for fork and
  same-repo PRs alike. The failure keeps a stale lockfile from looking green, and it
  is what starts the push workflow. On a fork PR, regenerate `pixi.lock` on Linux
  and push it from the contributor branch.

**`.github/workflows/pixi-lock-push.yml` ("Pixi lock push")** is triggered by
`workflow_run` when a "Pixi lock" run completes. Its `push-lockfile` job runs only
when that run came from a `pull_request` event, failed, and had a head branch in this
repository. It has `contents: write`, `actions: read` and `pull-requests: read`, and
never runs the PR's code. The artifact was made by the PR's code, so it is treated as
untrusted:

- `Download lockfile patch` exits quietly if the run uploaded no `pixi-lock` artifact
  (the run failed for another reason, or the PR is from a fork). The zip is kept
  whole, and nothing is extracted by its untrusted entry names.
- `Validate pull request` requires the zip to hold exactly `pixi-lock.patch` and
  `pr-number`, the PR number to be numeric, and the PR to be open with the same head
  SHA, repository and branch as the run, and a valid branch name. A moved head means
  a newer run will handle it.
- `Fetch pull request head` fetches the head commit with plain `git`, without
  credentials (the repository is public), rather than `actions/checkout`.
- `Apply lockfile patch` requires the patch to modify `pixi.lock` and nothing else
  (no other paths, renames, deletions or mode changes), applies only its diff, and
  commits it as `nipreps[bot]` with a DataLad run record written in the workflow.
  The patch's own author and message are discarded: a forged DataLad record would
  run arbitrary commands on `datalad rerun`. It then re-checks that the result is
  one commit on the PR head that changes only `pixi.lock`.
- `Push lockfile commit` pushes non-forced, with the token passed as an
  `http.extraheader` on the `git push` command line.

`workflow_run` always runs the copy of `pixi-lock-push.yml` on `main`, so changes to
it, and to the "Pixi lock" workflow name it matches, take effect only after merge.

Checks on the bot's commit do not start on their own: GitHub may hold them as
"action required" until a maintainer approves them, or not queue them at all. Push
another commit, or close and reopen the PR, to run them.

Do not move `pixi lock --check` back under `datalad run`. The check exits 0 but still
rewrites the workspace package's hatch-vcs version in `pixi.lock`, so running it there
records that rewrite on every run, and each bot push queues another run (seen on
PennLINC/qsirecon#396).

`check-lockfile` is the job to require as a status check. Third-party actions in both
workflows are pinned to commit SHAs with a `# vX.Y.Z` comment.

---

## CircleCI trigger behavior

CircleCI uses `image_prep` with cache key based on:

- `Dockerfile`
- `pixi.lock`

Base image rebuild is controlled by whether `BASE_IMAGE` in `Dockerfile` exists
in Docker Hub.

- Editing `Dockerfile.base` alone does not force rebuild if base tag already exists.
- To force base rebuild, bump `ARG BASE_IMAGE=...` in `Dockerfile`.
