# Use Ubuntu 16.04 LTS CUDA runtime 9.1
FROM pennbbl/qsiprep_build:20.11.0

# Installing and setting up miniconda
ARG MINICONDA_INSTALLER=Miniconda3-py38_4.10.3-Linux-x86_64.sh
RUN curl -sSLO https://repo.continuum.io/miniconda/${MINICONDA_INSTALLER}} && \
    bash ${MINICONDA_INSTALLER}} -b -p /usr/local/miniconda && \
    rm ${MINICONDA_INSTALLER}


# # Installing precomputed python packages
# RUN conda install -y  \
#                      numpy=1.18.5 \
#                      scipy=1.2.0 \
#                      mkl=2020.2 \
#                      mkl-service \
#                      scikit-learn=0.20.2 \
#                      matplotlib=2.2.3 \
#                      seaborn=0.9.0 \
#                      pandas=0.24.0 \
#                      libxml2=2.9.9 \
#                      libxslt=1.1.33 \
#                      graphviz \
#                      cython=0.29.2 \
#                      imageio=2.5.0 \
#                      olefile=0.46 \
#                      pillow=6.0.0 \
#                      scikit-image=0.14.2 \
#                      traits=4.6.0; sync &&  \
#     chmod -R a+rX /usr/local/miniconda; sync && \
#     chmod +x /usr/local/miniconda/bin/*; sync && \
#     conda build purge-all; sync && \
#     conda clean -tipsy && sync


# Unless otherwise specified each process should only use one thread - nipype
# will handle parallelization
ENV MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MRTRIX_NTHREADS=1 \
    KMP_WARNINGS=0

WORKDIR /root/

ENV QSIRECON_ATLAS /atlas/qsirecon_atlases
RUN bash -c \
    'mkdir /atlas \
    && cd  /atlas \
    && wget -nv https://upenn.box.com/shared/static/8k17yt2rfeqm3emzol5sa0j9fh3dhs0i.xz \
    && tar xvfJm 8k17yt2rfeqm3emzol5sa0j9fh3dhs0i.xz \
    && rm 8k17yt2rfeqm3emzol5sa0j9fh3dhs0i.xz \
    && echo 1'

# Installing qsiprep
COPY . /src/qsiprep
ARG VERSION

# Force static versioning within container
RUN echo "${VERSION}" > /src/qsiprep/qsiprep/VERSION && \
    echo "include qsiprep/VERSION" >> /src/qsiprep/MANIFEST.in && \
    pip install --no-cache-dir "/src/qsiprep[all]"

# Precaching fonts, set 'Agg' as default backend for matplotlib
RUN python -c "from matplotlib import font_manager" && \
    sed -i 's/\(backend *: \).*$/\1Agg/g' $( python -c "import matplotlib; print(matplotlib.matplotlib_fname())" )

RUN mkdir -p ${HOME}/.dipy

RUN python -c "import amico; amico.core.setup()"

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
