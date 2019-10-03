.. include:: links.rst

.. _reconstruction:


Reconstruction
================

You can send the outputs from ``qsiprep`` to other software packages
by specifying a JSON file with the ``--recon-spec`` option. Here we use
"reconstruction" to mean reconstructing ODFs/FODs/EAPs and connectivity matrices
from the preprocessed diffusion data.

The easiest way to get started is to use one of the pre-packaged workflows.
Instead of specifying a path to a file you can choose from the following:

+---------------------------+--------------+-------------+---------+-----------------+----------------+
| Option                    | Requires SDC | MultiShell  |   DSI   | DTI             |  Tractography  |
+===========================+==============+=============+=========+=================+================+
|``mrtrix_msmt_csd``        |    Yes       |  Required   |    No   |      No         | Probabilistic  |
+---------------------------+--------------+-------------+---------+-----------------+----------------+
|``mrtrix_dhollander``      |    Yes       |    Yes      |    No   |     Yes         | Probabilistic  |
+---------------------------+--------------+-------------+---------+-----------------+----------------+
|``mrtrix_dhollander_no5tt``|     No       |    Yes      |    No   |     Yes         | Probabilistic  |
+---------------------------+--------------+-------------+---------+-----------------+----------------+
|``mrtrix_tckglobal``       |    Yes       |   Required  |    No   |      No         |    Global      |
+---------------------------+--------------+-------------+---------+-----------------+----------------+
|``dsi_studio_gqi``         | Recommended  |    Yes      |   Yes   | Not Recommended | Deterministic  |
+---------------------------+--------------+-------------+---------+-----------------+----------------+
|``dipy_mapmri``            | Recommended  |    Yes      |   Yes   |      No         |   Both         |
+---------------------------+--------------+-------------+---------+-----------------+----------------+
|``dipy_3dshore``           | Recommended  |    Yes      |   Yes   |      No         |   Both         |
+---------------------------+--------------+-------------+---------+-----------------+----------------+
|``csdsi_3dshore``          | Recommended  |    Yes      |   Yes   |      No         |   Both         |
+---------------------------+--------------+-------------+---------+-----------------+----------------+

These workflows each take considerable processing time, because they output as many versions of
connectivity as possible. All included atlases and all possible weightings are included. Each of
these corresponds to a JSON file that can be found in QSIprep's
`github <https://github.com/PennBBL/qsiprep/tree/master/qsiprep/data/pipelines>`_. For extra
information about how to customize these, see :ref:`custom_reconstruction`.

``qsiprep`` supports a limited number of algorithms that are wrapped in
nipype workflows and can be configured and connected based on the
recon spec JSON file.  The output from one workflow can be the input to
another as long as the output from the upstream workflow matches the inputs to
the downstream workflow. The :ref:`recon_workflows` section lists all the
available workflows and their inputs and outputs.


Reconstruction Outputs: Connectivity matrices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instead of offering a bewildering number of options for constructing connectivity matrices,
``qsiprep`` will construct as many connectivity matrices as it can given the reconstruction
methods. It is **highly** recommended that you pick a weighting method before you run
these pipelines and only look at those numbers. If you look at more than one weighting method
be sure to adjust your statistics for these additional comparisons.



.. _custom_reconstruction:

Building a custom reconstruction pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instead of going through each possible element of a pipeline, we will go through
a simple example and describe its components.

Simple DSI Studio example
~~~~~~~~~~~~~~~~~~~~~~~~~~

The reconstruction pipeline is created by the user and specified in a JSON
file similar to the following::

  {
    "name": "dsistudio_pipeline",
    "space": "T1w",
    "anatomical": ["mrtrix_5tt"],
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
because we specified  ``"space": "T1w"`` earlier. Be sure a warp is calculated if
using these (transforms_).

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


.. _transforms:

Spaces and transforms
^^^^^^^^^^^^^^^^^^^^^^^

Transforming the a reconstruction output to template space requires that the
spatial normalization transform is calculated. This can be accomplished in
two ways

  1. During preprocessing you included ``--output-spaces template``. This will also
     result in your preprocessed DWI series being written in template space, which
     you likely don't want.
  2. You include the ``--force-spatial-normalization`` argument during preprocessing.
     This will create the warp to your template and store it in the derivatives directory
     but will not write your preprocessed DWI series in template space.

Some of the workflows require a warp to a template. For example, connectivity_ will use
this warp to transform atlases into T1w space for calculating a connectivity matrix.
