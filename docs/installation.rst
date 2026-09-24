.. include:: links.rst

############
Installation
############

There are two ways to use *QSIPrep*: as a Python package from PyPI, or as a
container with Docker or Apptainer. Use the container unless you have a
reason not to.

Once *QSIPrep* is installed, see :doc:`data` for what it expects of the input
and :doc:`running` for the command line.


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

Mount the input, output and working directories into the container and
give the container-side paths as arguments::

    $ docker run -ti --rm \
        -v /path/to/bids:/data:ro \
        -v /path/to/output:/out \
        -v /path/to/work:/work \
        pennlinc/qsiprep:<version> \
        /data /out participant \
        -w /work --output-resolution 2

Replace ``<version>`` with a release tag. ``latest`` is the last release and
``unstable`` the current ``main`` branch; do not use ``unstable`` for
anything but testing. No FreeSurfer license is needed.

To use the GPU, add ``--gpus all`` to ``docker run`` and ``--gpu`` with the
tasks to accelerate to the *QSIPrep* options (see :ref:`hmc_flags`).


*******************
Apptainer Container
*******************

The easiest way to get an Apptainer (formerly Singularity) image is to run::

    $ apptainer build qsiprep-<version>.sif docker://pennlinc/qsiprep:<version>

Where ``<version>`` should be replaced with the desired version of qsiprep that you want to download.
Do not use ``latest`` or ``unstable`` unless you are performing limited testing.

Bind the input, output and working directories::

    $ apptainer run --containall --writable-tmpfs \
        -B /path/to/bids,/path/to/output,/path/to/work \
        qsiprep-<version>.sif \
        /path/to/bids /path/to/output participant \
        -w /path/to/work --output-resolution 2

Add ``--nv`` to use the GPU.

.. note::
    **Running QSIPrep with Apptainer on Non-Internet Nodes**

    QSIPrep relies on TemplateFlow to provide standard anatomical templates. By default, it downloads
    necessary files into the ``$TEMPLATEFLOW_HOME`` directory (default: ``$HOME/.cache/templateflow``).
    However, when running with ``--containall``, the default location may not be accessible, and any
    missing templates will be tentatively downloaded into the temporary QSIPrep working directory instead.
    To avoid this, you must always set and bind ``TEMPLATEFLOW_HOME`` explicitly.

    Steps to ensure successful execution:

    1. On an **internet-enabled node**, set and bind the ``TEMPLATEFLOW_HOME`` variable to a persistent
       directory before running QSIPrep. This will ensure all necessary templates are downloaded into the
       specified location:

       .. code-block:: sh

          export TEMPLATEFLOW_HOME=/path/to/persistent/templateflow
          apptainer run --cleanenv --containall \
            -B ${TEMPLATEFLOW_HOME}:${TEMPLATEFLOW_HOME} \
            --env "TEMPLATEFLOW_HOME=$TEMPLATEFLOW_HOME" \
            /path/to/qsiprep_<VERSION>.sif <your commands>

    2. If the ``TEMPLATEFLOW_HOME`` directory is not accessible on your target HPC node, manually copy it
       to the target system after the first run.

    3. On nodes without internet access, bind the copied ``TEMPLATEFLOW_HOME`` directory and set the
       environment variable as described above before running QSIPrep.
    4. It may help to run a single subject or session on its own before running many jobs that access the templates. The single run will download the necessary templates and prevent multiple jobs from attempting to download the templates simultaneously.


    For additional troubleshooting, see `fmriprep docs <https://fmriprep.org/en/stable/faq.html#how-do-you-use-templateflow-in-the-absence-of-access-to-the-internet>`_
    or `this thread on Neurostars <https://neurostars.org/t/issue-with-qsiprep-templateflow-on-hpc-host-with-no-internet-access/31259/10?u=pierre-nedelec>`_.


*********************
External Dependencies
*********************

*QSIPrep* is written in Python 3.11 or later and is based on Nipype_. The
container image bundles the non-Python tools it calls. FSL_, MRtrix3 (a
release and the development branch), `DSI Studio`_, AFNI_, TORTOISE_
(including CUDA builds), FreeSurfer's SynthStrip and SynthSeg, SynB0-DISCO,
niimath and CUDA 12.2 come from the ``pennlinc/qsiprep-base`` image, whose versions
are pinned in ``Dockerfile.base`` in the repository. ANTs_ and the Python
environment are installed on top of it by Pixi in the main ``Dockerfile``.
Installing these tools yourself and running *QSIPrep* from PyPI is possible
but not supported.
