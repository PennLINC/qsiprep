.. include:: links.rst

Usage
-----

Execution and the BIDS format
=============================

The ``qsiprep`` workflow takes as principal input the path of the dataset
that is to be processed.
The input dataset is required to be in valid :abbr:`BIDS (Brain Imaging Data
Structure)` format, and it must include at least one T1w structural image and
(unless disabled with a flag) a diffusion MRI series.
We highly recommend that you validate your dataset with the free, online
`BIDS Validator <http://bids-standard.github.io/bids-validator/>`_.

The exact command to run ``qsiprep`` depends on the Installation_ method.
The common parts of the command are similar to the `BIDS-Apps
<https://github.com/BIDS-Apps>`_ definition.
Example: ::

    qsiprep --bids_dir data/bids_root/ --output_dir out/ --analysis_level participant -w work/


Command-Line Arguments
======================

.. argparse::
   :ref: qsiprep.cli.run.get_parser
   :prog: qsiprep
   :nodefault:
   :nodefaultconst:


The docker wrapper CLI
======================

.. argparse::
   :ref: qsiprep_docker.get_parser
   :prog: qsiprep-docker
   :nodefault:
   :nodefaultconst:


Debugging
=========

Logs and crashfiles are outputted into the
``<output dir>/qsiprep/sub-<participant_label>/log`` directory.
Information on how to customize and understand these files can be found on the
`nipype debugging <http://nipype.readthedocs.io/en/latest/users/debug.html>`_
page.

Support and communication
=========================

The documentation of this project is found here: http://qsiprep.readthedocs.org/en/latest/.

All bugs, concerns and enhancement requests for this software can be submitted here:
https://github.com/pennbbl/qsiprep/issues.
