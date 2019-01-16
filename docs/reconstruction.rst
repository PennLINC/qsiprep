

.. _reconstruction:

-------------------------------
Reconstruction pipeline details
-------------------------------

You can send the outputs from ``qsiprep`` to other software packages
by specifying a JSON file with the ``--recon-spec`` option. Here we use
"reconstruction" to mean reconstructing ODFs/FODs/EAPs from the preprocessed
diffusion data.

Simple DSI Studio example
~~~~~~~~~~~~~~~~~~~~~~~~~~

The reconstruction pipeline is created by the user and specified in a JSON
file similar to the following::

  {
    "name": "dsistudio_pipeline",
    "space": "T1w",
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

This will send the preprocessed data from ``qsiprep``, only the data in ``"T1w"``
space (``"space": "T1w"``), to DSI Studio (``"software": "DSI Studio"``),
which will perform reconstruction (``"action": "reconstruction"``) and will save
its outputs with the suffix "_gqi" (``"output_suffix": "gqi"``). Additional
parameters can be sent to specify how the reconstruction should take place in
the ``"parameters"`` item. In this case it will produce a file
``something_space-T1w_gqi.fib.gz``.  This fib file is then used to get out
scalars (gfa, qa, etc) that will also be saved in the output directory.

Assuming this file is called ``qgi_scalar_export.json`` you can execute this
pipeline with::

  $ qsiprep-docker \
      --bids_dir /path/to/bids \
      --recon_input /output/from/qsiprep \
      --recon_spec gqi_scalar_export.json \
      --output_dir /where/my/reconstructed/data/goes \
      --analysis_level participant \
      --fs-license-file /path/to/license.txt


Other example reconstruction specs
-----------------------------------

Here are some other potentially useful JSON files for reconstructing ``qsiprep`` output
and a description of what they do

Reconstruct
