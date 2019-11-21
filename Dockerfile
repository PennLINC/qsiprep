# Use Ubuntu 16.04 LTS
FROM ubuntu:xenial-20161213

# Pre-cache neurodebian key
COPY docker/files/neurodebian.gpg /usr/local/etc/neurodebian.gpg

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
                    pkg-config \
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
                    g++ \
                    gcc \
                    libeigen3-dev \
                    libqt5svg5* \
                    make \
                    python \
                    python-numpy \
                    zlib1g-dev \
                    imagemagick \
                    git && \
    curl -sL https://deb.nodesource.com/setup_10.x | bash - && \
    apt-get install -y --no-install-recommends \
      nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install latest pandoc
RUN curl -o pandoc-2.2.2.1-1-amd64.deb -sSL "https://github.com/jgm/pandoc/releases/download/2.2.2.1/pandoc-2.2.2.1-1-amd64.deb" && \
    dpkg -i pandoc-2.2.2.1-1-amd64.deb && \
    rm pandoc-2.2.2.1-1-amd64.deb

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

  ENV FSLDIR="/opt/fsl-6.0.1" \
      PATH="/opt/fsl-6.0.1/bin:$PATH"
  RUN echo "Downloading FSL ..." \
      && mkdir -p /opt/fsl-6.0.1 \
      && curl -fsSL --retry 5 https://fsl.fmrib.ox.ac.uk/fsldownloads/fsl-6.0.1-centos6_64.tar.gz \
      | tar -xz -C /opt/fsl-6.0.1 --strip-components 1 \
      && echo "Installing FSL conda environment ..." \
      && bash /opt/fsl-6.0.1/etc/fslconf/fslpython_install.sh -f /opt/fsl-6.0.1

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
RUN curl -sSL "http://neuro.debian.net/lists/$( lsb_release -c | cut -f2 ).us-ca.full" >> /etc/apt/sources.list.d/neurodebian.sources.list && \
    apt-key add /usr/local/etc/neurodebian.gpg && \
    (apt-key adv --refresh-keys --keyserver hkp://ha.pool.sks-keyservers.net 0xA5D32F012649A5A9 || true)

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    afni=16.2.07~dfsg.1-5~nd16.04+1 \
                    git-annex-standalone && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install DSI Studio
ENV PATH=$PATH:/opt/dsi-studio/dsi_studio_64
ARG DSI_SHA=1685bfeb4df194fa3e907f4ee032691d82aeb3ba
ARG TIPL_SHA=b01ec5a46bdec7d13506d23e97b91fd0840c2b09
RUN mkdir /opt/dsi-studio \
  && cd /opt/dsi-studio \
  && curl -sSLO https://github.com/frankyeh/DSI-Studio/archive/${DSI_SHA}.zip \
  && unzip ${DSI_SHA}.zip \
  && mv DSI-Studio-${DSI_SHA} src \
  && rm -rf ${DSI_SHA}.zip \
  && curl -sSLO https://github.com/frankyeh/TIPL/archive/${TIPL_SHA}.zip \
  && unzip ${TIPL_SHA}.zip \
  && mv TIPL-${TIPL_SHA} src/tipl \
  && rm ${TIPL_SHA}.zip \
  && mkdir build && cd build \
  && qmake ../src && make \
  && cd /opt/dsi-studio \
  && curl -sSLO 'https://upenn.box.com/shared/static/01r73d4a15utl13himv4d7cjpa6etf6z.gz' \
  && tar xvfz 01r73d4a15utl13himv4d7cjpa6etf6z.gz \
  && rm 01r73d4a15utl13himv4d7cjpa6etf6z.gz \
  && cd dsi_studio_64 \
  && mv ../build/dsi_studio . \
  && rm -rf /opt/dsi-studio/src /opt/dsi-studio/build

# Install mrtrix3 from source
ARG MRTRIX_SHA=5d6b3a6ffc6ee651151779539c8fd1e2e03fad81
ENV PATH="/opt/mrtrix3-latest/bin:$PATH"
RUN cd /opt \
    && curl -sSLO https://github.com/MRtrix3/mrtrix3/archive/${MRTRIX_SHA}.zip \
    && unzip ${MRTRIX_SHA}.zip \
    && mv mrtrix3-${MRTRIX_SHA} /opt/mrtrix3-latest \
    && rm ${MRTRIX_SHA}.zip \
    && cd /opt/mrtrix3-latest \
    && ./configure \
    && echo "Compiling MRtrix3 ..." \
    && ./build

# Installing ANTs latest from source
ARG ANTS_SHA=b8889562341471dbe18b415d2bcefb9b8f12232a
ADD https://cmake.org/files/v3.11/cmake-3.11.4-Linux-x86_64.sh /cmake-3.11.4-Linux-x86_64.sh
ENV ANTSPATH="/opt/ants-latest/bin" \
    PATH="/opt/ants-latest/bin:$PATH" \
    LD_LIBRARY_PATH="/opt/ants-latest/lib:$LD_LIBRARY_PATH"
RUN mkdir /opt/cmake \
  && sh /cmake-3.11.4-Linux-x86_64.sh --prefix=/opt/cmake --skip-license \
  && ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake \
  && apt-get update -qq \
    && mkdir /tmp/ants \
    && cd /tmp \
    && git clone https://github.com/ANTsX/ANTs.git \
    && mv ANTs /tmp/ants/source \
    && cd /tmp/ants/source \
    && git checkout ${ANTS_SHA} \
    && mkdir -p /tmp/ants/build \
    && cd /tmp/ants/build \
    && mkdir -p /opt/ants-latest \
    && git config --global url."https://".insteadOf git:// \
    && cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=/opt/ants-latest /tmp/ants/source \
    && make -j2 \
    && cd ANTS-build \
    && make install \
    && rm -rf /tmp/ants \
    && rm -rf /opt/cmake /usr/local/bin/cmake

ENV C3DPATH="/opt/convert3d-nightly" \
    PATH="/opt/convert3d-nightly/bin:$PATH"
RUN echo "Downloading Convert3D ..." \
    && mkdir -p /opt/convert3d-nightly \
    && curl -fsSL --retry 5 https://sourceforge.net/projects/c3d/files/c3d/Nightly/c3d-nightly-Linux-x86_64.tar.gz/download \
    | tar -xz -C /opt/convert3d-nightly --strip-components 1

# Create a shared $HOME directory
RUN useradd -m -s /bin/bash -G users qsiprep
WORKDIR /home/qsiprep
ENV HOME="/home/qsiprep"

# Installing SVGO
RUN curl -sL https://deb.nodesource.com/setup_10.x | bash -
RUN apt-get install -y nodejs
RUN npm install -g svgo

# Installing bids-validator
RUN npm install -g bids-validator@1.2.3

# Installing and setting up miniconda
RUN curl -sSLO https://repo.continuum.io/miniconda/Miniconda3-4.5.12-Linux-x86_64.sh && \
    bash Miniconda3-4.5.12-Linux-x86_64.sh -b -p /usr/local/miniconda && \
    rm Miniconda3-4.5.12-Linux-x86_64.sh

ENV PATH=/usr/local/miniconda/bin:$PATH \
    CPATH="/usr/local/miniconda/include/:$CPATH" \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONNOUSERSITE=1

# Installing precomputed python packages
RUN conda install -y python=3.7.1 \
                     numpy=1.15.4 \
                     scipy=1.2.0 \
                     mkl=2019.1 \
                     mkl-service \
                     scikit-learn=0.20.2 \
                     matplotlib=2.2.3 \
                     seaborn=0.9.0 \
                     pandas=0.24.0 \
                     libxml2=2.9.9 \
                     libxslt=1.1.33 \
                     graphviz=2.40.1 \
                     cython=0.29.2 \
                     imageio=2.5.0 \
                     olefile=0.46 \
                     pillow=6.0.0 \
                     scikit-image=0.14.2 \
                     traits=4.6.0; sync &&  \
    chmod -R a+rX /usr/local/miniconda; sync && \
    chmod +x /usr/local/miniconda/bin/*; sync && \
    conda build purge-all; sync && \
    conda clean -tipsy && sync


# Unless otherwise specified each process should only use one thread - nipype
# will handle parallelization
ENV MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MRTRIX_NTHREADS=1

WORKDIR /root/

ENV QSIRECON_ATLAS /atlas/qsirecon_atlases
RUN bash -c \
    'mkdir /atlas \
    && cd  /atlas \
    && wget -nv https://upenn.box.com/shared/static/8k17yt2rfeqm3emzol5sa0j9fh3dhs0i.xz \
    && tar xvfJm 8k17yt2rfeqm3emzol5sa0j9fh3dhs0i.xz \
    && rm 8k17yt2rfeqm3emzol5sa0j9fh3dhs0i.xz \
    && echo 1'


# Precaching atlases
ENV CRN_SHARED_DATA /niworkflows_data
ADD docker/scripts/get_templates.sh get_templates.sh
RUN mkdir $CRN_SHARED_DATA && \
    /root/get_templates.sh && \
    chmod -R a+rX $CRN_SHARED_DATA && \
    echo "add OASIS30"

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

RUN find $HOME -type d -exec chmod go=u {} + && \
    find $HOME -type f -exec chmod go=u {} +

ENV AFNI_INSTALLDIR=/usr/lib/afni \
    PATH=${PATH}:/usr/lib/afni/bin \
    AFNI_PLUGINPATH=/usr/lib/afni/plugins \
    AFNI_MODELPATH=/usr/lib/afni/models \
    AFNI_TTATLAS_DATASET=/usr/share/afni/atlases \
    AFNI_IMSAVE_WARNINGS=NO \
    FSLOUTPUTTYPE=NIFTI_GZ \
    MRTRIX_NTHREADS=1 \
    IS_DOCKER_8395080871=1

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

# Make singularity mount directories
RUN  mkdir -p /sngl/data \
  && mkdir /sngl/qsiprep-output \
  && mkdir /sngl/out \
  && mkdir /sngl/scratch \
  && mkdir /sngl/spec \
  && mkdir /sngl/eddy \
  && chmod a+rwx /sngl/*
