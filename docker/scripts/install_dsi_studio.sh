#!/bin/bash

# Install DSI Studio
apt-get install -y --no-install-recommends \
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
                mesa-utils

cd /opt/dsistudio
git clone -b master https://github.com/frankyeh/DSI-Studio.git src
curl -sSLO https://github.com/frankyeh/TIPL/zipball > master.zip
unzip master.zip
mv frankyeh-TIPL-* src/image
mkdir build
cd build
qmake ../src
make
cd ..
curl -sSLO https://www.dropbox.com/s/ew3rv0jrqqny2dq/dsi_studio_64.zip?dl=1 dsistudio64.zip
mv dsi_studio_64.zip?dl=1 dsi_studio_64.zip && \
unzip dsi_studio_64.zip && \
WORKDIR dsi_studio_64
RUN find . -name '*.dll' -exec rm {} \;
RUN rmdir iconengines imageformats platforms printsupport
RUN rm dsi_studio.exe
RUN cp ../build/dsi_studio
