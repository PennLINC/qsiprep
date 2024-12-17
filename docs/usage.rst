.. include:: links.rst

#####
Usage
#####

The *QSIPrep* preprocessing workflow takes as principal input the path of
the dataset that is to be processed. The input dataset is required to be in
valid :abbr:`BIDS (Brain Imaging Data Structure)` format at least one
diffusion MRI series. The T1w image and the DWI may be in separate BIDS
<session> folders for a given subject. We highly recommend that you validate
your dataset with the free, online `BIDS Validator
<http://bids-standard.github.io/bids-validator/>`_.

The exact command to run *QSIPrep* depends on the Installation_ method.
The common parts of the command are similar to the `BIDS-Apps
<https://github.com/BIDS-Apps>`_ definition.

Example: ::

    qsiprep data/bids_root/ out/ participant -w work/


**********************
Command-Line Arguments
**********************

.. argparse::
   :ref: qsiprep.cli.parser._build_parser
   :prog: qsiprep
   :nodefault:
   :nodefaultconst:


***********
Infant mode
***********

If ``--infant`` is used, the pipeline will select an MNIInfant template with the
appropriate cohort based on the participant's age.

``--infant`` is only compatible with ``--subject-anatomical-reference sessionwise``.

.. note::

    *QSIPrep*'s cohort selection is derived from Nibabies.

Participant Ages
================

*QSIPrep* will attempt to automatically extract participant ages (in months) from the BIDS layout. Specifically, these two files will be checked:

Sessions file: <bids-root>/<subject>/subject_sessions.tsv

Participants file: <bids-root>/participants.tsv

Either file should include age (or if you wish to be more explicit: age_months) columns, and it is recommended to have an accompanying JSON file to further describe these fields, and explicitly state the values are in months.


******************
Note on using CUDA
******************

The CUDA runtime version 11.1.1 is included in the *QSIPrep* docker image.
The CUDA version of eddy is dramatically faster than the openmp version.
Information on running Docker with CUDA enabled can be found on
`dockerhub <https://github.com/NVIDIA/nvidia-docker/wiki/CUDA>`_. If running with Apptainer,
the call to Apptainer should include ``--nv``. To enable CUDA, see :ref:`configure_eddy`.


*********
Debugging
*********

Logs and crashfiles are outputted into the
``<output dir>/qsiprep/sub-<participant_label>/log`` directory.
Information on how to customize and understand these files can be found on the
`nipype debugging <http://nipype.readthedocs.io/en/latest/users/debug.html>`_
page.
