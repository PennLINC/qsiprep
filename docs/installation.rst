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
``qsiprep-docker`` wrapper, which requires Python and an Internet connection
and that you install the ``qsiprep-container`` package with ``pip``.

.. note:: If running Docker Desktop on MacOS (or via Docker Desktop), be sure to set
    the memory to 6 or more GB. Too little memory assigned to Docker Desktop can result
    in a message like ``Killed.``

When run, ``qsiprep-docker`` will generate a Docker command line for you,
print it out for reporting purposes, and then run the command, e.g.::

    $ qsiprep-docker /path/to/data/dir /path/to/output/dir participant
    RUNNING: docker run --rm -it -v /path/to/data/dir:/data \
        -v /path/to_output/dir:/out pennbbl/qsiprep:latest \
        /data /out participant
    ...

You may also invoke ``docker`` directly::

    $ docker run -ti --rm \
        -v /filepath/to/data/dir \
        -v /filepath/to/output/dir \
        -v ${FREESURFER_HOME}/license.txt:/opt/freesurfer/license.txt \
        pennbbl/qsiprep:latest \
        /filepath/to/data/dir /filepath/to/output/dir participant \
        --fs-license-file /opt/freesurfer/license.txt

For example: ::

    $ docker run -ti --rm \
        -v $HOME/fullds005 \
        -v $HOME/dockerout \
        -v ${FREESURFER_HOME}/license.txt:/opt/freesurfer/license.txt \
        pennbbl/qsiprep:latest \
        $HOME/fullds005 $HOME/dockerout participant \
        --ignore fieldmaps \
        --fs-license-file /opt/freesurfer/license.txt

If you are running Freesurfer as part of QSIPrep,
you will need to mount your Freesurfer license.txt file when invoking ``docker`` ::

    $ docker run -ti --rm \
        -v $HOME/fullds005 \
        -v $HOME/dockerout \
        -v ${FREESURFER_HOME}/license.txt:/opt/freesurfer/license.txt \
        pennbbl/qsiprep:latest \
        $HOME/fullds005 -v $HOME/dockerout participant \
        --fs-license-file /opt/freesurfer/license.txt


See `External Dependencies`_ for more information on what is included in the Docker image
and how it's built.



Singularity Container
=====================

The easiest way to get a Sigularity image is to run::

    $ singularity build qsiprep-<version>.sif docker://pennbbl/qsiprep:<version>

Where ``<version>`` should be replaced with the desired version of qsiprep that you want to download.
Do not use ``latest`` or ``unstable`` unless you are performing limited testing.

As with Docker, you will need to bind the Freesurfer license.txt when running Singularity ::

    $ singularity run --containall --writable-tmpfs \
        -B $HOME/fullds005,$HOME/dockerout,${FREESURFER_HOME}/license.txt:/opt/freesurfer/license.txt \
        qsiprep-<version>.sif \
        $HOME/fullds005 $HOME/dockerout participant \
        --fs-license-file /opt/freesurfer/license.txt


External Dependencies
---------------------

qsiprep is written using Python 3.10 (or above), and is based on
nipype_. The external dependencies are built in the `qsiprep_build
<https://github.com/PennLINC/qsiprep_build>`_ repository. There
you can find the URLs used to download the dependency source code
and the steps to compile each dependency.
