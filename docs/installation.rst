.. include:: links.rst

############
Installation
############

There are two easy ways to use *QSIPrep*:
as a Python library with PyPi or as a container with Docker or Singularity/Apptainer.
Using a local container method is highly recommended.

Once you are ready to run *QSIPrep*, see Usage_ for details.


**************
Python Library
**************

To install the *QSIPrep* Python library, use pip::

    $ pip install --user --upgrade qsiprep

We strongly discourage installing *QSIPrep* this way,
as *QSIPrep* relies on a number of non-Python dependencies that are difficult to install
and configure on a local system.
Pip will not install these dependencies for you.
Instead, we recommend using the Docker or Singularity/Apptainer containers,
wherein all of the necessary dependencies will come pre-installed and configured.


.. _`Docker Container`:

****************
Docker Container
****************

In order to run *QSIPrep* in a Docker container, Docker must be `installed
<https://docs.docker.com/engine/installation/>`_.

.. note::
    If running Docker Desktop on MacOS (or via Docker Desktop), be sure to set
    the memory to 6 or more GB. Too little memory assigned to Docker Desktop can result
    in a message like ``Killed.``

You may invoke ``docker`` directly::

    $ docker run -ti --rm \
        -v /filepath/to/data/dir \
        -v /filepath/to/output/dir \
        -v ${FREESURFER_HOME}/license.txt:/opt/freesurfer/license.txt \
        pennlinc/qsiprep:latest \
        /filepath/to/data/dir /filepath/to/output/dir participant \
        --fs-license-file /opt/freesurfer/license.txt

For example: ::

    $ docker run -ti --rm \
        -v $HOME/fullds005 \
        -v $HOME/dockerout \
        -v ${FREESURFER_HOME}/license.txt:/opt/freesurfer/license.txt \
        pennlinc/qsiprep:latest \
        $HOME/fullds005 $HOME/dockerout participant \
        --ignore fieldmaps \
        --fs-license-file /opt/freesurfer/license.txt

If you are running Freesurfer as part of *QSIPrep*,
you will need to mount your Freesurfer license.txt file when invoking ``docker`` ::

    $ docker run -ti --rm \
        -v $HOME/fullds005 \
        -v $HOME/dockerout \
        -v ${FREESURFER_HOME}/license.txt:/opt/freesurfer/license.txt \
        pennlinc/qsiprep:latest \
        $HOME/fullds005 -v $HOME/dockerout participant \
        --fs-license-file /opt/freesurfer/license.txt


See `External Dependencies`_ for more information on what is included in the Docker image
and how it's built.


*******************
Apptainer Container
*******************

The easiest way to get an Apptainer (formerly Singularity) image is to run::

    $ apptainer build qsiprep-<version>.sif docker://pennlinc/qsiprep:<version>

Where ``<version>`` should be replaced with the desired version of qsiprep that you want to download.
Do not use ``latest`` or ``unstable`` unless you are performing limited testing.

As with Docker, you will need to bind the Freesurfer license.txt when running Apptainer ::

    $ apptainer run --containall --writable-tmpfs \
        -B $HOME/fullds005,$HOME/dockerout,${FREESURFER_HOME}/license.txt:/opt/freesurfer/license.txt \
        qsiprep-<version>.sif \
        $HOME/fullds005 $HOME/dockerout participant \
        --fs-license-file /opt/freesurfer/license.txt


*********************
External Dependencies
*********************

*QSIPrep* is written using Python 3.10 (or above), and is based on nipype_.
The external dependencies are built in the
`qsiprep_build <https://github.com/PennLINC/qsiprep_build>`_ repository.
There you can find the URLs used to download the dependency source code
and the steps to compile each dependency.
