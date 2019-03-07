.. _reconstruction:

---------------
Reconstruction
---------------

You can send the outputs from ``qsiprep`` to other software packages
by specifying a JSON file with the ``--recon-spec`` option. Here we use
"reconstruction" to mean reconstructing ODFs/FODs/EAPs from the preprocessed
diffusion data.

.. note::
   You can also reconstruct data preprocessed with ``dwipreproc`` and ``eddy``.
   Note **the :ref:`conform` node must be the first in your pipeline.** This
   ensures that the orientation of the preprocessed images and gradient scheme
   are correctly interpreted by downstream nodes. Skipping this step can
   result in incorrect orientation!!

``qsiprep`` supports a limited number of algorithms that are wrapped in
nipype workflows that can be configured and connected based on the
recon spec JSON file.  The output from one workflow can be the input to
another as long as the output from the upstream workflow matches the inputs to
the downstream workflow. The :ref:`recon_workflows` section lists all the
available workflows and their inputs and outputs.

Building a reconstruction pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A number of pre-configured pipelines can be found
`here <https://github.com/PennBBL/qsiprep/tree/master/qsiprep/data/pipelines>`_.
Instead of going through each possible element of a pipeline, we will go through
a simple example and describe its components.

Simple DSI Studio example
~~~~~~~~~~~~~~~~~~~~~~~~~~

The reconstruction pipeline is created by the user and specified in a JSON
file similar to the following::

  {
    "name": "dsistudio_pipeline",
    "space": "T1w",
    "atlases": ["schaefer100", "schaefer200"],
    "nodes": [
      {
        "name": "dsistudio_gqi",
        "software": "DSI Studio",
        "action": "reconstruction",
        "input": "qsiprep",
        "output_suffix": "gqi",
        "parameters": {"method": "gqi"}
      },
      {
        "name": "scalar_export",
        "software": "DSI Studio",
        "action": "export",
        "input": "dsistudio_gqi",
        "output_suffix": "gqiscalar"
      }
    ]
  }

Pipeline level metadata
^^^^^^^^^^^^^^^^^^^^^^^^^

The ``"name"`` element defines the name of the pipeline. This will ultimately
be the name of the output directory. By setting ``"space": "T1w"`` we specify
that all operations will take place in subject anatomical (``"T1w"``) space.
Many "connectomics" algorithms require a brain parcellation. A number of these
come packaged with ``qsiprep`` in the Docker image. In this case, the
atlases will be transformed from group template space to subject anatomical space
because we specified  ``"space": "T1w"`` earlier.

Pipeline nodes
^^^^^^^^^^^^^^^

The ``"nodes"`` list contains the workflows that will be run as a part of the
reconstruction pipeline. All nodes must have a ``name`` element, this serves
as an id for this node and is used to connect its outputs to a downstream
node. In this example we can see that the node with ``"name": "dsistudio_gqi"``
sends its outputs to the node with ``"name": "scalar_export"`` because
the ``"name": "scalar_export"`` node specifies ``"input": "dsistudio_gqi"``.
If no ``"input"`` is specified for a node, it is assumed that the
outputs from ``qsiprep`` will be its inputs.

By specifying ``"software": "DSI Studio"`` we will be using algorithms implemented
in `DSI Studio`_. Other options include MRTrix_ and Dipy_. Since there are many
things that `DSI Studio`_ can do, we specify that we want to reconstruct the
output from ``qsiprep`` by adding ``"action": "reconstruction"``. Additional
parameters can be sent to specify how the reconstruction should take place in
the ``"parameters"`` item. Possible options for ``"software"``, ``"action"``
and ``"parameters"`` can be found in the :ref:`recon_workflows` section.

You will have access to all the intermediate data in the pipeline's working directory,
but can specify which outputs you want to save to the output directory by setting
an ``"output_suffix"``. Looking at the outputs for a workflow in the :ref:`recon_workflows`
section you can see what is produced by each workflow. Each of these files
will be saved in your output directory for each subject with a name matching
your specified ``"output_suffix"``. In this case it will produce a file
``something_space-T1w_gqi.fib.gz``.  Since a fib file is produced by this node
and the downstream ``export_scalars`` node uses it, the scalars produced from
that node will be from this same fib file.

Executing the reconstruction pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Assuming this file is called ``qgi_scalar_export.json`` and you've installed
``qsiprep-container`` you can execute this pipeline with::

  $ qsiprep-docker \
      --bids_dir /path/to/bids \
      --recon_input /output/from/qsiprep \
      --recon_spec gqi_scalar_export.json \
      --output_dir /where/my/reconstructed/data/goes \
      --analysis_level participant \
      --fs-license-file /path/to/license.txt


.. _recon_workflows:

Reconstruction Workflows
-------------------------

Here the inputs, outputs and parameters of each workflow are listed for
each software package and action.

Dipy
~~~~~~~~~~

Dipy offers the most options for data reconstruction. For convenience, all Dipy
reconstruction workflows have the option to write outputs in MRTrix and DSI Studio
formats.

Action: ``"3dSHORE_reconstruction"``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use the 3dSHORE bases to fit diffusion signal, and estimate ODFs.


DSI Studio
~~~~~~~~~~~~~

Action: ``"reconstruction"``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Action: ``"export"``
^^^^^^^^^^^^^^^^^^^^^^^

Action: ``"connectivity"``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MRTrix
~~~~~~~~~~

Action: ``"csd"``
^^^^^^^^^^^^^^^^^^^

Action: ``"msmt_csd"``
^^^^^^^^^^^^^^^^^^^^^^^^


QSIPrep
~~~~~~~~~~~

Assorted workflows

Action: ``"conform"``
^^^^^^^^^^^^^^^^^^^^^^^^

Action: ``"controllability"``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Action: ``"discard_repeated_samples"``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: qsiprep.workflows.recon.

Action: ``"mif_to_fib"``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: qsiprep.workflows.recon.converters.init_fibgz_to_mif_wf

Action: ``"fib_to_mif"``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

..note:: Workflow not implemented yet

.. autofunction:: qsiprep.workflows.recon.converters.init_mif_to_fibgz_wf
