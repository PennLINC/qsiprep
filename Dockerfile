# Use Ubuntu 16.04 LTS
FROM ubuntu:xenial-20161213

# Pre-cache neurodebian key
COPY docker/files/neurodebian.gpg /root/.neurodebian.gpg

# Prepare environment
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    curl \
                    bzip2 \
                    ca-certificates \
                    xvfb \
                    cython3 \
                    build-essential \
                    autoconf \
                    libtool \
                    pkg-config && \
    curl -sSL http://neuro.debian.net/lists/xenial.us-ca.full >> /etc/apt/sources.list.d/neurodebian.sources.list && \
    apt-key add /root/.neurodebian.gpg && \
    (apt-key adv --refresh-keys --keyserver hkp://ha.pool.sks-keyservers.net 0xA5D32F012649A5A9 || true) && \
    apt-get update

# Installing freesurfer
RUN curl -sSL https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/6.0.1/freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.1.tar.gz | tar zxv --no-same-owner -C /opt \
    --exclude='freesurfer/trctrain' \
    --exclude='freesurfer/subjects/fsaverage_sym' \
    --exclude='freesurfer/subjects/fsaverage3' \
    --exclude='freesurfer/subjects/fsaverage4' \
    --exclude='freesurfer/subjects/cvs_avg35' \
    --exclude='freesurfer/subjects/cvs_avg35_inMNI152' \
    --exclude='freesurfer/subjects/bert' \
    --exclude='freesurfer/subjects/V1_average' \
    --exclude='freesurfer/average/mult-comp-cor' \
    --exclude='freesurfer/lib/cuda' \
    --exclude='freesurfer/lib/qt'

  ENV FSLDIR="/opt/fsl-5.0.11" \
      PATH="/opt/fsl-5.0.11/bin:$PATH"
  RUN apt-get update -qq \
      && apt-get install -y -q --no-install-recommends \
             bc \
             dc \
             file \
             libfontconfig1 \
             libfreetype6 \
             libgl1-mesa-dev \
             libglu1-mesa-dev \
             libgomp1 \
             libice6 \
             libxcursor1 \
             libxft2 \
             libxinerama1 \
             libxrandr2 \
             libxrender1 \
             libxt6 \
             wget \
      && apt-get clean \
      && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
      && echo "Downloading FSL ..." \
      && mkdir -p /opt/fsl-5.0.11 \
      && curl -fsSL --retry 5 https://fsl.fmrib.ox.ac.uk/fsldownloads/fsl-5.0.11-centos6_64.tar.gz \
      | tar -xz -C /opt/fsl-5.0.11 --strip-components 1 \
      && echo "Installing FSL conda environment ..." \
      && bash /opt/fsl-5.0.11/etc/fslconf/fslpython_install.sh -f /opt/fsl-5.0.11

ENV FREESURFER_HOME=/opt/freesurfer \
    SUBJECTS_DIR=/opt/freesurfer/subjects \
    FUNCTIONALS_DIR=/opt/freesurfer/sessions \
    MNI_DIR=/opt/freesurfer/mni \
    LOCAL_DIR=/opt/freesurfer/local \
    FSFAST_HOME=/opt/freesurfer/fsfast \
    MINC_BIN_DIR=/opt/freesurfer/mni/bin \
    MINC_LIB_DIR=/opt/freesurfer/mni/lib \
    MNI_DATAPATH=/opt/freesurfer/mni/data \
    FMRI_ANALYSIS_DIR=/opt/freesurfer/fsfast
ENV PERL5LIB=$MINC_LIB_DIR/perl5/5.8.5 \
    MNI_PERL5LIB=$MINC_LIB_DIR/perl5/5.8.5 \
    PATH=$FREESURFER_HOME/bin:$FSFAST_HOME/bin:$FREESURFER_HOME/tktools:$MINC_BIN_DIR:$PATH

# Installing Neurodebian packages (FSL, AFNI, git)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    afni=16.2.07~dfsg.1-5~nd16.04+1

# Install DSI Studio
ENV PATH=$PATH:/opt/dsi-studio/dsi_studio_64
RUN apt-get install -y --no-install-recommends \
                    git \
                    qt5-qmake \
                    qt5-default \
                    libboost-all-dev \
                    zlib1g \
                    zlib1g-dev \
                    libqt5opengl5-dev \
                    unzip \
                    libgl1-mesa-dev \
                    libglu1-mesa-dev \
                    freeglut3-dev \
                    mesa-utils \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
  && mkdir /opt/dsi-studio \
  && cd /opt/dsi-studio \
  && git clone -b master https://github.com/frankyeh/DSI-Studio.git src \
  && curl -sSLO https://github.com/frankyeh/TIPL/archive/master.zip > master.zip \
  && unzip master.zip \
  && mv TIPL-master src/tipl \
  && mkdir build && cd build \
  && qmake ../src && make \
  && cd /opt/dsi-studio \
  && curl -sSLO 'https://www.dropbox.com/s/ew3rv0jrqqny2dq/dsi_studio_64.zip?dl=1' > dsistudio64.zip \
  && mv 'dsi_studio_64.zip?dl=1' dsistudio64.zip \
  && unzip dsistudio64.zip && cd dsi_studio_64 \
  && find . -name '*.dll' -exec rm {} \; \
  && rm -rf iconengines imageformats platforms printsupport \
  && rm dsi_studio.exe \
  && mv ../build/dsi_studio . \
  && rm -rf /opt/dsi-studio/src /opt/dsi-studio/build

# Install mrtrix3 from source
ENV PATH="/opt/mrtrix3-latest/bin:$PATH"
RUN apt-get update -qq \
    && apt-get install -y -q --no-install-recommends \
           g++ \
           gcc \
           libeigen3-dev \
           libqt5svg5* \
           make \
           python \
           python-numpy \
           zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && mkdir -p /opt/mrtrix3-latest \
    && git clone https://github.com/MRtrix3/mrtrix3.git /opt/mrtrix3-latest \
    && cd /opt/mrtrix3-latest \
    && ./configure \
    && echo "Compiling MRtrix3 ..." \
    && ./build

# Installing ANTs latest from source
ADD https://cmake.org/files/v3.11/cmake-3.11.4-Linux-x86_64.sh /cmake-3.11.4-Linux-x86_64.sh
ENV ANTSPATH="/opt/ants-latest/bin" \
    PATH="/opt/ants-latest/bin:$PATH" \
    LD_LIBRARY_PATH="/opt/ants-latest/lib:$LD_LIBRARY_PATH"
RUN mkdir /opt/cmake \
  && sh /cmake-3.11.4-Linux-x86_64.sh --prefix=/opt/cmake --skip-license \
  && ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake \
  && apt-get update -qq \
    && apt-get install -y -q --no-install-recommends \
           g++ \
           gcc \
           make \
           zlib1g-dev \
           imagemagick \
    && mkdir -p /tmp/ants/build \
    && git config --global url."https://".insteadOf git:// \
    && git clone https://github.com/ANTsX/ANTs.git /tmp/ants/source \
    && cd /tmp/ants/build \
    && cmake -DBUILD_SHARED_LIBS=ON /tmp/ants/source \
    && make -j1 \
    && mkdir -p /opt/ants-latest \
    && mv bin lib /opt/ants-latest/ \
    && mv /tmp/ants/source/Scripts/* /opt/ants-latest/bin \
    && rm -rf /tmp/ants \
    && rm -rf /opt/cmake /usr/local/bin/cmake

ENV C3DPATH="/opt/convert3d-nightly" \
    PATH="/opt/convert3d-nightly/bin:$PATH"
RUN echo "Downloading Convert3D ..." \
    && mkdir -p /opt/convert3d-nightly \
    && curl -fsSL --retry 5 https://sourceforge.net/projects/c3d/files/c3d/Nightly/c3d-nightly-Linux-x86_64.tar.gz/download \
    | tar -xz -C /opt/convert3d-nightly --strip-components 1

# Installing SVGO
RUN curl -sL https://deb.nodesource.com/setup_10.x | bash -
RUN apt-get install -y nodejs
RUN npm install -g svgo

# Installing and setting up miniconda
RUN curl -sSLO https://repo.continuum.io/miniconda/Miniconda3-4.5.12-Linux-x86_64.sh && \
    bash Miniconda3-4.5.12-Linux-x86_64.sh -b -p /usr/local/miniconda && \
    rm Miniconda3-4.5.12-Linux-x86_64.sh

ENV PATH=/usr/local/miniconda/bin:$PATH \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONNOUSERSITE=1

# Installing precomputed python packages
RUN conda install -y mkl=2019.1 mkl-service;  sync &&\
    conda install -y numpy=1.15.4 \
                     scipy=1.2.0 \
                     scikit-learn=0.20.2 \
                     matplotlib=3.0.2 \
                     pandas=0.24.0 \
                     libxml2=2.9.9 \
                     libxslt=1.1.33 \
                     graphviz=2.40.1 \
                     cython=0.29.2 \
                     traits=4.6.0; sync &&  \
    chmod -R a+rX /usr/local/miniconda; sync && \
    chmod +x /usr/local/miniconda/bin/*; sync && \
    conda clean --all -y; sync && \
    conda clean -tipsy && sync

# Precaching fonts, set 'Agg' as default backend for matplotlib
RUN python -c "from matplotlib import font_manager" && \
    sed -i 's/\(backend *: \).*$/\1Agg/g' $( python -c "import matplotlib; print(matplotlib.matplotlib_fname())" )

# Install latest pandoc
RUN curl -o pandoc-2.2.2.1-1-amd64.deb -sSL "https://github.com/jgm/pandoc/releases/download/2.2.2.1/pandoc-2.2.2.1-1-amd64.deb" && \
    dpkg -i pandoc-2.2.2.1-1-amd64.deb && \
    rm pandoc-2.2.2.1-1-amd64.deb

# Unless otherwise specified each process should only use one thread - nipype
# will handle parallelization
ENV MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

WORKDIR /root/

ENV QSIRECON_ATLAS /atlas/qsirecon_atlases
RUN bash -c \
    'mkdir /atlas \
    && cd  /atlas \
    && wget -nv https://upenn.box.com/shared/static/pvttd0om8zhzfjr7ul6uiny24gmz7ns2.xz \
    && tar xvfJm pvttd0om8zhzfjr7ul6uiny24gmz7ns2.xz \
    && rm pvttd0om8zhzfjr7ul6uiny24gmz7ns2.xz'


# Precaching atlases
ENV CRN_SHARED_DATA /niworkflows_data
ADD docker/scripts/get_templates.sh get_templates.sh
RUN mkdir $CRN_SHARED_DATA && \
    /root/get_templates.sh && \
    chmod -R a+rX $CRN_SHARED_DATA && \
    echo "add OASIS30"

# Installing dev requirements (packages that are not in pypi)
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt && \
    rm -rf ~/.cache/pip \
    && echo 1

# Installing qsiprep
COPY . /root/src/qsiprep
ARG VERSION
# Force static versioning within container
RUN echo "${VERSION}" > /root/src/qsiprep/qsiprep/VERSION && \
    cd /root/src/qsiprep && \
    pip install .[all] && \
    rm -rf ~/.cache/pip

ENV AFNI_INSTALLDIR=/usr/lib/afni \
    PATH=${PATH}:/usr/lib/afni/bin \
    AFNI_PLUGINPATH=/usr/lib/afni/plugins \
    AFNI_MODELPATH=/usr/lib/afni/models \
    AFNI_TTATLAS_DATASET=/usr/share/afni/atlases \
    AFNI_IMSAVE_WARNINGS=NO \
    FSLOUTPUTTYPE=NIFTI_GZ


RUN ldconfig
WORKDIR /tmp/
ENTRYPOINT ["/usr/local/miniconda/bin/qsiprep"]

ARG BUILD_DATE
ARG VCS_REF
ARG VERSION
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="qsiprep" \
      org.label-schema.description="qsiprep - q Space Images preprocessing tool" \
      org.label-schema.url="http://qsiprep.org" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/pennbbl/qsiprep" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0"

# Make singularity mount directories
RUN  mkdir -p /sngl/data \
  && mkdir /sngl/qsiprep-output \
  && mkdir /sngl/out \
  && mkdir /sngl/scratch \
  && mkdir /sngl/spec
