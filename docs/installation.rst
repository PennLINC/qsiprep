.. include:: links.rst

------------
Installation
------------

There are three ways to use qsiprep:
in a `Docker Container`_, in a `Singularity Container`_, or in a `Manually
Prepared Environment (Python 3.6+)`_.
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

See `External Dependencies`_ for more information (e.g., specific versions) on
what is included in the latest Docker images.


Singularity Container
=====================

The easiest way to get a Sigularity image is to run::

    $ singularity build qsiprep-<version>.sif docker://pennbbl/qsiprep:<version>

Where ``<version>`` should be replaced with the desired version of qsiprep that you want to download.
Do not use ``latest``.

Manually Prepared Environment (Python 3.6+)
===========================================

.. warning::

   This method is not recommended! Make sure you would rather do this than
   use a `Docker Container`_ or a `Singularity Container`_. Also, this
   method does not work at the moment because there is no package on PyPI.

Make sure all of qsiprep's `External Dependencies`_ are installed.
These tools must be installed and their binaries available in the
system's ``$PATH``.
A relatively interpretable description of how your environment can be set-up
is found in the `Dockerfile <https://github.com/pennbbl/qsiprep/blob/master/Dockerfile>`_.
As an additional installation setting, FreeSurfer requires a license file.

On a functional Python 3.5 (or above) environment with ``pip`` installed,
qsiprep can be installed using the habitual command ::

    $ pip install qsiprep

Check your installation with the ``--version`` argument ::

    $ qsiprep --version


External Dependencies
---------------------

qsiprep is written using Python 3.5 (or above), and is based on
nipype_.

qsiprep requires some other neuroimaging software tools that are
not handled by the Python's packaging system (Pypi) used to deploy
the ``qsiprep`` package:

- ANTs_ (version 2.3.9)
- AFNI_ (version Debian-16.2.07)
- Dipy_ (version 0.15.0)
- FreeSurfer_ (6.0.1)
- FSL_ (5.0.11)
