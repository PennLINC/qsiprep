FROM pennbbl/qsiprep_build:23.5.5

# WORKDIR /root/
# # Installing qsiprep
COPY . /src/qsiprep
ARG VERSION

# # Force static versioning within container
RUN echo "${VERSION}" > /src/qsiprep/qsiprep/VERSION && \
    echo "include qsiprep/VERSION" >> /src/qsiprep/MANIFEST.in && \
    pip install --no-cache-dir "/src/qsiprep[all]"

# Precaching fonts, set 'Agg' as default backend for matplotlib
RUN python -c "from matplotlib import font_manager" && \
    sed -i 's/\(backend *: \).*$/\1Agg/g' $( python -c "import matplotlib; print(matplotlib.matplotlib_fname())" )

RUN find $HOME -type d -exec chmod go=u {} + && \
    find $HOME -type f -exec chmod go=u {} +

RUN ldconfig
WORKDIR /tmp/
ENTRYPOINT ["/usr/local/miniconda/bin/qsiprep"]

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
