ARG BASE_IMAGE=pennlinc/qsiprep-base:20261005

FROM ghcr.io/prefix-dev/pixi:0.58.0 AS build
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    ca-certificates \
                    build-essential \
                    curl \
                    git && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN pixi config set --global run-post-link-scripts insecure

# Install dependencies before the package itself to leverage caching
RUN mkdir /app
COPY pixi.lock pyproject.toml /app
WORKDIR /app
# First install runs before COPY . so .git is missing.
# Use --skip qsiprep (lockfile name) so pixi skips building the local package.
RUN --mount=type=cache,target=/root/.cache/rattler pixi install -e qsiprep -e test --frozen --skip qsiprep
RUN --mount=type=cache,target=/root/.npm pixi run --as-is -e qsiprep npm install -g svgo@^3.2.0 bids-validator@1.14.10
RUN pixi shell-hook -e qsiprep --as-is | grep -v PATH > /shell-hook.sh
RUN pixi shell-hook -e test --as-is | grep -v PATH > /test-shell-hook.sh

# Prefetch the adult templates before COPY . so code changes do not download them again.
# Modes are set here so the chmod in the base stage leaves the files unchanged.
COPY scripts/fetch_templates.py /app/scripts/fetch_templates.py
RUN TQDM_DISABLE=1 pixi run --as-is -e qsiprep python scripts/fetch_templates.py --tf-dir /templateflow && \
    chmod -R go=u /templateflow

# Finally, install the package
COPY . /app
# Install test and production environments separately so production does not
# inherit editable-install behavior needed for test workflows.
RUN --mount=type=cache,target=/root/.cache/rattler pixi install -e test --frozen
RUN --mount=type=cache,target=/root/.cache/rattler pixi install -e qsiprep --frozen
# Ensure qsiprep is installed non-editably in the qsiprep env so the copied env is
# self-contained in the runtime image (lockfile may resolve to editable variant).
# Pixi envs do not include pip; use uv to install into the env's Python.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    /root/.local/bin/uv pip install --python /app/.pixi/envs/qsiprep/bin/python --no-deps --force-reinstall .
# pixi writes conda-meta records owner-only (0600). Non-root runs (Apptainer,
# docker --user) read eddy's record for its version, so open them up here,
# before the environments are copied into the runtime stages.
RUN chmod -R a+rX /app/.pixi/envs/qsiprep/conda-meta /app/.pixi/envs/test/conda-meta

FROM ${BASE_IMAGE} AS base
WORKDIR /home/qsiprep
ENV HOME="/home/qsiprep"

# Pin which tree each command resolves to. The default PATH order matches
# --mrtrix-version stable; config.workflow.init() reorders it for --mrtrix-version dev.
# dwidenoise2 exists only in the development tree, so it must fall through to it.
# Run compiled binaries only: MRtrix3's Python scripts (dwibiascorrect among them) have
# no interpreter here, since Python arrives with the pixi env in the stages below.
RUN test "$(command -v mrdegibbs)"      = "/opt/mrtrix3-stable/bin/mrdegibbs" && \
    test ! -e /opt/conda && \
    test "$(command -v dwidenoise)"     = "/opt/mrtrix3-stable/bin/dwidenoise" && \
    test "$(command -v dwibiascorrect)" = "/opt/mrtrix3-stable/bin/dwibiascorrect" && \
    test "$(command -v dwidenoise2)"    = "/opt/mrtrix3-dev/bin/dwidenoise2" && \
    /opt/mrtrix3-dev/bin/mrdegibbs -help | grep -q dimensionality && \
    test -d /opt/mrtrix3-dev/share/mrtrix3/dwidenoise2 && \
    test "$(/opt/mrtrix3-stable/bin/mrinfo -version | head -1 | awk '{print $3}')" = "$MRTRIX3_STABLE_VERSION" && \
    test "$(/opt/mrtrix3-dev/bin/mrinfo -version | head -1 | awk '{print $3}')" = "$MRTRIX3_DEV_VERSION"

# Not COPY --link, which would reset the owner and mode of $HOME.
COPY --from=build /templateflow $HOME/.cache/templateflow
ENV TEMPLATEFLOW_HOME="$HOME/.cache/templateflow"
RUN chmod -R go=u $HOME
WORKDIR /tmp

FROM base AS test
COPY --link --from=build /app/.pixi/envs/test /app/.pixi/envs/test
COPY --link --from=build /test-shell-hook.sh /shell-hook.sh
RUN cat /shell-hook.sh >> $HOME/.bashrc
ENV PATH="/app/.pixi/envs/test/bin:$PATH"
ENV FSLDIR="/app/.pixi/envs/test"
ENV FSL_DIR="/app/.pixi/envs/test"
ENV LD_LIBRARY_PATH="/app/.pixi/envs/test/lib:$LD_LIBRARY_PATH"
ENV QSIPREP_FREESURFER_PYTHON="/opt/freesurfer/bin/fspython"
ENV QSIPREP_TORCH_PYTHON="/opt/freesurfer-torch/bin/python"
# ANTs must come from this stage's Pixi environment, never from the base image.
RUN test "$(command -v python)" = "/app/.pixi/envs/test/bin/python" && \
    test "$(command -v antsRegistration)" = "/app/.pixi/envs/test/bin/antsRegistration" && \
    test ! -e /opt/ants && \
    /app/.pixi/envs/test/bin/python -c "import contourpy" && \
    /opt/freesurfer/bin/fspython -c "import nibabel, scipy, surfa, tensorflow" && \
    /opt/freesurfer-torch/bin/python -c "import nibabel, scipy, surfa, torch; assert torch.version.cuda"
# The prefetched templates resolve without network access.
RUN --network=none python -c "from templateflow.api import get; \
    print(get('MNI152NLin2009cAsym', resolution=1, desc=None, suffix='T1w', extension='.nii.gz'))"
ARG VCS_REF
LABEL org.opencontainers.image.revision=$VCS_REF

FROM base AS qsiprep
COPY --link --from=build /app/.pixi/envs/qsiprep /app/.pixi/envs/qsiprep
COPY --link --from=build /shell-hook.sh /shell-hook.sh
RUN cat /shell-hook.sh >> $HOME/.bashrc
ENV PATH="/app/.pixi/envs/qsiprep/bin:$PATH"
ENV FSLDIR="/app/.pixi/envs/qsiprep"
ENV FSL_DIR="/app/.pixi/envs/qsiprep"
ENV LD_LIBRARY_PATH="/app/.pixi/envs/qsiprep/lib:$LD_LIBRARY_PATH"
ENV IS_DOCKER_8395080871=1
ENV QSIPREP_FREESURFER_PYTHON="/opt/freesurfer/bin/fspython"
ENV QSIPREP_TORCH_PYTHON="/opt/freesurfer-torch/bin/python"
# Verify the runtime image can import qsiprep without source tree mounts, and
# that each ML tool's interpreter has the tool's dependencies and can compile
# its script. (FreeSurfer's mri_synthstrip exits 1 on --help/--version and
# defers "import torch" until after that check, so byte-compile it rather than
# running --help.)
# Also assert that ANTs resolves from the production Pixi environment.
RUN test "$(command -v python)" = "/app/.pixi/envs/qsiprep/bin/python" && \
    test "$(command -v antsRegistration)" = "/app/.pixi/envs/qsiprep/bin/antsRegistration" && \
    test ! -e /opt/ants && \
    /app/.pixi/envs/qsiprep/bin/python -c "import contourpy, qsiprep" && \
    /opt/freesurfer/bin/fspython -c "import nibabel, scipy, surfa, tensorflow" && \
    /opt/freesurfer/bin/fspython -m py_compile /opt/freesurfer/bin/mri_synthseg && \
    /opt/freesurfer-torch/bin/python -c "import nibabel, scipy, surfa, torch; assert torch.version.cuda" && \
    /opt/freesurfer-torch/bin/python -m py_compile /opt/freesurfer/bin/mri_synthstrip
# The prefetched templates resolve without network access.
RUN --network=none python -c "from templateflow.api import get; \
    print(get('MNI152NLin2009cAsym', resolution=1, desc=None, suffix='T1w', extension='.nii.gz'))"

ENTRYPOINT ["/app/.pixi/envs/qsiprep/bin/qsiprep"]
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="qsiprep" \
      org.label-schema.description="qsiprep - q Space Images preprocessing tool" \
      org.label-schema.url="http://qsiprep.readthedocs.io" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/pennlinc/qsiprep" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0"
