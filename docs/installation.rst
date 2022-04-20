.. include:: links.rst

------------
Installation
------------

There are two easy ways to use qsiprep:
in a `Docker Container`_, or in a `Singularity Container`_.
Using a local container method is highly recommended.
Once you are ready to run qsiprep, see Usage_ for details.

To install::

    $ pip install --user --upgrade qsiprep-container

.. _`Docker Container`:

Docker Container
================

In order to run qsiprep in a Docker container, Docker must be `installed
<https://docs.docker.com/engine/installation/>`_.
Once Docker is installed, the recommended way to run qsiprep is to use the
``qsiprep-docker`` wrapper, which requires Python and an Internet connection.

.. note:: If running Docker Desktop on MacOS, be sure to set the memory to 6 or more GB.
    Too little memory assigned to Docker Desktop can result in a message like ``Killed.``

When run, ``qsiprep-docker`` will generate a Docker command line for you,
print it out for reporting purposes, and then run the command, e.g.::

    $ qsiprep-docker /path/to/data/dir /path/to/output/dir participant
    RUNNING: docker run --rm -it -v /path/to/data/dir:/data:ro \
        -v /path/to_output/dir:/out pennbbl/qsiprep:1.0.0 \
        /data /out participant
    ...

You may also invoke ``docker`` directly::

    $ docker run -ti --rm \
        -v filepath/to/data/dir:/data:ro \
        -v filepath/to/output/dir:/out \
        pennbbl/qsiprep:latest \
        /data /out participant

For example: ::

    $ docker run -ti --rm \
        -v $HOME/fullds005:/data:ro \
        -v $HOME/dockerout:/out \
        pennbbl/qsiprep:latest \
        /data /out participant \
        --ignore fieldmaps

If you are running Freesurfer as part of QSIPrep, 
you will need to mount your Freesurfer license.txt file when invoking ``docker`` ::

    $ docker run -ti --rm \
        -v $HOME/fullds005:/data:ro \
        -v $HOME/dockerout:/out \
        -v ${FREESURFER_HOME}/license.txt:/opt/freesurfer/license.txt \
        pennbbl/qsiprep:latest \
        /data /out participant \
        --fs-license-file /opt/freesurfer/license.txt


See `External Dependencies`_ for more information (e.g., specific versions) on
what is included in the latest Docker images.



Singularity Container
=====================

The easiest way to get a Sigularity image is to run::

    $ singularity build qsiprep-<version>.sif docker://pennbbl/qsiprep:<version>

Where ``<version>`` should be replaced with the desired version of qsiprep that you want to download.
Do not use ``latest``.

As with Docker, you will need to bind the Freesurfer license.txt when running Singularity ::

    $ singularity run --cleanenv \
        -B $HOME/fullds005:/data:ro,$HOME/dockerout:/out,${FREESURFER_HOME}/license.txt:/opt/freesurfer/license.txt \
        qsiprep-<version>.sif \
        /data /out participant \
        --fs-license-file /opt/freesurfer/license.txt


External Dependencies
---------------------

qsiprep is written using Python 3.6 (or above), and is based on
nipype_.

qsiprep requires some other non-python neuroimaging software tools:

- ANTs_ (version 2.3.9)
- AFNI_ (version Debian-16.2.07)
- FreeSurfer_ (6.0.1)
- FSL_ (6.0.3)
- `DSI Studio`_
- MRtrix_
