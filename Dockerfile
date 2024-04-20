
# Build into a wheel in a stage that has git installed
FROM python:slim AS wheelstage
RUN pip install build
RUN apt-get update && \
    apt-get install -y --no-install-recommends git
COPY . /src/qsiprep
RUN python -m build /src/qsiprep

FROM pennbbl/qsiprep_build:24.4.24

# Install qsiprep wheel
COPY --from=wheelstage /src/qsiprep/dist/*.whl .
RUN pip install --no-cache-dir $( ls *.whl )

# Precaching fonts, set 'Agg' as default backend for matplotlib
RUN python -c "from matplotlib import font_manager" && \
    sed -i 's/\(backend *: \).*$/\1Agg/g' $( python -c "import matplotlib; print(matplotlib.matplotlib_fname())" )

RUN find $HOME -type d -exec chmod go=u {} + && \
    find $HOME -type f -exec chmod go=u {} +

RUN ldconfig
WORKDIR /tmp/
ENTRYPOINT ["/opt/conda/envs/qsiprep/bin/qsiprep"]
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/conda/envs/qsiprep/lib/python3.10/site-packages/nvidia/cudnn/lib
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="qsiprep" \
      org.label-schema.description="qsiprep - q Space Images preprocessing tool" \
      org.label-schema.url="http://qsiprep.readthedocs.io" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/pennbbl/qsiprep" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0"