ARG BASE_IMAGE=pennlinc/qsiprep-base:20260310-3

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

FROM ${BASE_IMAGE} AS base
WORKDIR /home/qsiprep
ENV HOME="/home/qsiprep"

RUN chmod -R go=u $HOME
WORKDIR /tmp

FROM base AS test
COPY --link --from=build /app/.pixi/envs/test /app/.pixi/envs/test
COPY --link --from=build /test-shell-hook.sh /shell-hook.sh
RUN cat /shell-hook.sh >> $HOME/.bashrc
ENV PATH="/app/.pixi/envs/test/bin:$PATH"
ENV FSLDIR="/app/.pixi/envs/test"
ARG VCS_REF
LABEL org.opencontainers.image.revision=$VCS_REF

FROM base AS qsiprep
COPY --link --from=build /app/.pixi/envs/qsiprep /app/.pixi/envs/qsiprep
COPY --link --from=build /shell-hook.sh /shell-hook.sh
RUN cat /shell-hook.sh >> $HOME/.bashrc
ENV PATH="/app/.pixi/envs/qsiprep/bin:$PATH"
ENV FSLDIR="/app/.pixi/envs/qsiprep"
ENV IS_DOCKER_8395080871=1
# Verify the runtime image can import qsiprep without source tree mounts.
RUN /app/.pixi/envs/qsiprep/bin/python -c "import qsiprep"

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