.. include:: links.rst

############
Installation
############

Use *QSIPrep* as a container, with Docker or Apptainer. The Python package
on PyPI is for people who maintain their own environment.

Once *QSIPrep* is installed, see :doc:`data` for what it expects of the input
and :doc:`running` for the command line.


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


.. _templateflow_cache:

**********************
TemplateFlow templates
**********************

*QSIPrep* takes its standard templates from `TemplateFlow
<https://www.templateflow.org/>`_. The container image already holds the files
it uses from the two adult templates: the 1 mm T1w and ``desc-brain`` mask of
``MNI152NLin2009cAsym`` and ``MNI152NLin6Asym``, and the 1 mm T2w of
``MNI152NLin2009cAsym`` (about 38 MB compressed). They are in
``/home/qsiprep/.cache/templateflow``, where the image's ``TEMPLATEFLOW_HOME``
points, so a run on adult data downloads no templates and needs no extra bind.

Any other template, such as the MNIInfant cohort that ``--infant`` selects, is
downloaded into ``$TEMPLATEFLOW_HOME`` the first time a run needs it. Inside
the image that download does not persist: Docker discards it with the
container, and an Apptainer image is read-only, so the download fails unless
``--writable-tmpfs`` gives the container a temporary overlay, which is
discarded as well.

To keep downloaded templates between runs, or to run on nodes without network
access, bind a TemplateFlow directory and set ``TEMPLATEFLOW_HOME`` to it. The
bound directory replaces the image's cache, so it must hold the adult
templates as well. Populate it once on a machine with network access, starting
from the image's cache::

    $ mkdir -p /path/to/templateflow
    $ apptainer exec --containall -B /path/to/templateflow qsiprep-<version>.sif \
        cp -r /home/qsiprep/.cache/templateflow/. /path/to/templateflow/

then add the templates the image leaves out. For MNIInfant cohort 1, the files
*QSIPrep* uses are::

    $ apptainer exec --containall -B /path/to/templateflow \
        --env TEMPLATEFLOW_HOME=/path/to/templateflow \
        qsiprep-<version>.sif python -c "
    from templateflow.api import get
    get('MNIInfant', cohort='1', resolution=1, desc=None, suffix=['T1w', 'T2w'], extension='.nii.gz')
    get('MNIInfant', cohort='1', resolution=1, desc='brain', suffix='mask', extension='.nii.gz')"

Copy the directory to the cluster if the compute nodes cannot see it, and bind
it in every run::

    $ apptainer run --containall --writable-tmpfs \
        -B /path/to/bids,/path/to/output,/path/to/work,/path/to/templateflow \
        --env TEMPLATEFLOW_HOME=/path/to/templateflow \
        qsiprep-<version>.sif \
        /path/to/bids /path/to/output participant \
        -w /path/to/work --output-resolution 2

With Docker, the equivalent options are
``-v /path/to/templateflow:/templateflow -e TEMPLATEFLOW_HOME=/templateflow``.
A run that needs a file missing from the bound directory tries to download it,
and fails without network access.


**************
Python Library
**************

*QSIPrep* is on PyPI, but pip only installs the Python side::

    $ pip install --user --upgrade qsiprep

Every tool listed under :ref:`external_dependencies` has to be installed and
configured separately, so this route is for people who maintain their own
environment and know why they want it. Everyone else should use a container.
Templates are downloaded into ``$TEMPLATEFLOW_HOME`` (``~/.cache/templateflow``
on Linux by default) when a run first needs them;
``python scripts/fetch_templates.py`` in a source checkout fetches the ones
the container image holds.


.. _external_dependencies:

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
