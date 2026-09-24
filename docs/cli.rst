.. include:: links.rst

.. _cli:

######################
Command-line reference
######################

The options are explained by decision on the :doc:`running` page. This is the
full ``qsiprep --help`` text. Options that take a list (``--ignore``,
``--force``, ``--gpu``, ``--participant-label``, ``--session-label``,
``--debug``) take space-separated values. ``--config-file`` loads the
settings of an earlier run, and any option given on the command line
overrides it.

.. argparse::
   :ref: qsiprep.cli.parser._build_parser
   :prog: qsiprep
