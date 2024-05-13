.. include:: links.rst

################
Developers - API
################

*****************************
Internal configuration system
*****************************

.. automodule:: qsiprep.config
   :members: from_dict, load, get, dumps, to_filename, init_spaces


***********
Library API
***********

Preprocessing Workflows
-----------------------

.. toctree::
   :glob:

   api/qsiprep.workflows.base
   api/qsiprep.workflows.anatomical
   api/qsiprep.workflows.dwi
   api/qsiprep.workflows.fieldmap


Reconstruction Workflows
------------------------

.. toctree::
   :glob:

   api/qsiprep.workflows.recon


Other Utilities
---------------

.. toctree::
   :glob:

   api/qsiprep.interfaces
   api/qsiprep.utils
   api/qsiprep.report
   api/qsiprep.viz
   api/qsiprep.qc
