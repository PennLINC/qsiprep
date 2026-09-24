.. include:: links.rst

.. _developers:

##########
Developers
##########

*QSIPrep* follows the `NiPreps contributing guidelines
<https://www.nipreps.org/community/CONTRIBUTING/>`_. This page covers what
is specific to this repository.


*******************
Working on the code
*******************

The dependencies are hard to install outside the container, so development
happens against the ``pennlinc/qsiprep:unstable`` image. To test changes
without rebuilding it, mount your checkout over the installed package. The
package lives in the image's Pixi environment; ``PYTHONPATH`` is the simplest
way to put your copy first::

    docker run --rm -it \
        -v $HOME/projects/qsiprep:/src/qsiprep:ro \
        -e PYTHONPATH=/src/qsiprep \
        -v /path/to/bids:/data:ro -v /path/to/out:/out \
        pennlinc/qsiprep:unstable /data /out participant --output-resolution 2

With Apptainer::

    PYTHONPATH=$HOME/projects/qsiprep apptainer run qsiprep.sif \
        /path/to/bids /path/to/out participant --output-resolution 2

To rebuild the image, run ``docker build --target qsiprep -t qsiprep .`` in
the repository (``--target test`` builds the image the integration tests
run in). The Dockerfile is a Pixi-based multi-stage build whose first line
names its base, ``pennlinc/qsiprep-base:<date>``. That base carries the
non-Python tools (FSL, MRtrix3, DSI Studio, AFNI, TORTOISE, FreeSurfer's
SynthStrip and SynthSeg, SynB0) and is pulled from Docker Hub, not built
locally. Python dependencies go in ``pyproject.toml``; changing them
requires ``pixi lock`` and a rebuild of the main image. Non-Python
dependencies go in ``Dockerfile.base``; bump its date tag in both files, and
CI builds and pushes the new base when it finds the tag missing from Docker
Hub. To try a base change before that, build it yourself with
``docker build -f Dockerfile.base -t pennlinc/qsiprep-base:<date> .``.

Running tests
=============

Unit tests need no data::

    pytest qsiprep/tests

Integration tests are marked and skipped by default. They run inside the
container against datasets downloaded by ``run_local_tests.py``::

    cd qsiprep/tests
    python run_local_tests.py -m "dsdti_fmap"

They take a long time. When they pass, open the HTML reports they produce
and check that the figures look right; judging that takes some familiarity
with *QSIPrep*'s outputs.


**************************
Building the documentation
**************************

The documentation is built with Sphinx. It needs a working *QSIPrep* import
and Graphviz, because the workflow graphs are drawn by building the
workflows::

    pip install -e '.[doc]'
    make -C docs html

Read the Docs builds the same way from ``.readthedocs.yaml``.

A few conventions:

* Headings use ``#`` with overline for page titles, ``*`` with overline for
  chapters, then ``=``, ``-`` and ``^``.
* Options are written in double backticks (``--output-resolution``). Cite
  papers with ``:footcite:p:`` and a ``.. footbibliography::`` at the end
  of the page; the entries live in ``qsiprep/data/boilerplate.bib``, shared
  with the methods boilerplate.
* Workflow graphs come from ``.. workflow::`` directives, which execute
  their code block. The builders read their settings from ``qsiprep.config``
  and some read NIfTI headers, so the block should import
  ``qsiprep_docs`` from ``docs/sphinxext``: it loads the parser defaults
  into the config, writes a small fake dataset, and provides
  ``example_unit()`` for ``single``, ``pepolar``, ``epi`` and ``phasediff``
  layouts. A block that raises is reported as a build warning and renders no
  graph, so check the build log.
* Keep the pages at the level of what runs, which option changes it, and
  what is written. Method background belongs in |artifacts_book|.


*******************
Methods boilerplate
*******************

The citation boilerplate is built from two attributes of workflow objects,
``__desc__`` and ``__postdesc__``. Once the full workflow is built, the
sub-workflows are visited in topological order: each ``__desc__`` is
appended before descending into a workflow's children, and its
``__postdesc__`` after. The text is Markdown, and a citation is a Bibtex
handle with ``@``: ``[@myreference]`` renders as a parenthetical citation.
Add new entries to ``qsiprep/data/boilerplate.bib``.


********************
The qsiplan contract
********************

Scan grouping and plan compilation live in the `qsiplan`_ package.
*QSIPrep* pins a compatible range (``qsiplan >= 0.4.1, < 0.5``) and imports
its grouping, plan, report and sidecar functions from
``qsiprep/workflows/base.py``. The grouping and plan options on *QSIPrep*'s
command line (``--hmc-method``, ``--sdc-method``, ``--sdc-anat-reference``,
``--separate-all-dwis``, ``--distortion-group-merge``,
``--subject-anatomical-reference`` and the grouping values of ``--ignore``
and ``--force``) are owned by ``qsiplan.cli_spec.PLAN_OPTIONS``, and
``qsiprep/tests/test_plan_cli_spec.py`` fails if the parser drifts from that
list. The one exemption is ``--shoreline-model``, which *QSIPrep* replaced
with the ``"model"`` key of ``--shoreline-config``; qsiplan still exposes
the flag until it grows a ``--shoreline-config`` of its own. Add a grouping
option to qsiplan first, then expose it here.


***
API
***

.. toctree::
   :maxdepth: 2

   api
