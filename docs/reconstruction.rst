.. include:: links.rst

.. _reconstruction:


Reconstruction
==============

You can send the outputs from ``qsiprep`` to other software packages
by specifying a JSON file with the ``--recon-spec`` option. Here we use
"reconstruction" to mean reconstructing ODFs/FODs/EAPs and connectivity matrices
from the preprocessed diffusion data.

The easiest way to get started is to use one of the :ref:`preconfigured_workflows`.
Instead of specifying a path to a file you can choose from the following:

+-------------------------------------------+-------------+---------+-----------------+----------------+
| Option                                    | MultiShell  |   DSI   | DTI             |  Tractography  |
+===========================================+=============+=========+=================+================+
|:ref:`mrtrix_multishell_msmt_ACT-fast`\*   |     Yes     |    No   |      No         | Probabilistic  |
+-------------------------------------------+-------------+---------+-----------------+----------------+
|:ref:`mrtrix_multishell_msmt_ACT-hsvs`     |     Yes     |    No   |      No         | Probabilistic  |
+-------------------------------------------+-------------+---------+-----------------+----------------+
|:ref:`mrtrix_multishell_msmt_noACT`        |     Yes     |    No   |      No         | Probabilistic  |
+-------------------------------------------+-------------+---------+-----------------+----------------+
|:ref:`mrtrix_singleshell_ss3t_noACT`       |     No      |    No   |      Yes        | Probabilistic  |
+-------------------------------------------+-------------+---------+-----------------+----------------+
|:ref:`mrtrix_singleshell_ss3t_ACT-hsvs`    |     No      |    No   |      Yes        | Probabilistic  |
+-------------------------------------------+-------------+---------+-----------------+----------------+
|:ref:`mrtrix_singleshell_ss3t_ACT-fast`\*  |     No      |    No   |      Yes        | Probabilistic  |
+-------------------------------------------+-------------+---------+-----------------+----------------+
|:ref:`pyafq`                               |     Yes     |    No   |      Yes        |   Both         |
+-------------------------------------------+-------------+---------+-----------------+----------------+
|:ref:`pyafq_input_trk`                     |     Yes     |    No   |      Yes        |   Both         |
+-------------------------------------------+-------------+---------+-----------------+----------------+
|:ref:`amico_noddi`                         |     Yes     |    No   |      No         |     None       |
+-------------------------------------------+-------------+---------+-----------------+----------------+
|:ref:`dsi_studio_gqi`                      |     Yes     |   Yes   |    Yes*         | Deterministic  |
+-------------------------------------------+-------------+---------+-----------------+----------------+
|:ref:`dipy_mapmri`                         |     Yes     |   Yes   |      No         |   Both         |
+-------------------------------------------+-------------+---------+-----------------+----------------+
|:ref:`dipy_3dshore`                        |     Yes     |   Yes   |      No         |   Both         |
+-------------------------------------------+-------------+---------+-----------------+----------------+
|:ref:`csdsi_3dshore`                       |     Yes     |   Yes   |      No         |   Both         |
+-------------------------------------------+-------------+---------+-----------------+----------------+
|:ref:`reorient_fslstd`                     |     Yes     |   Yes   |      Yes        |   None         |
+-------------------------------------------+-------------+---------+-----------------+----------------+

\* Not recommended

These workflows each take considerable processing time, because they output as many versions of
connectivity as possible. All :ref:`connectivity_atlases`  and all possible weightings are
included. Each workflow corresponds to a JSON file that can be found in QSIprep's
`github <https://github.com/PennBBL/qsiprep/tree/master/qsiprep/data/pipelines>`_. For extra
information about how to customize these, see :ref:`custom_reconstruction`.

To use a pre-packaged workflow, simply provide the name from the leftmost column above for the
``--recon-spec`` argument. For example::

  $ qsiprep-docker \
      /path/to/bids /path/for/reconstruction/outputs participant \
      --recon_input /output/from/qsiprep \
      --recon_spec dsi_studio_gqi \
      --fs-license-file /path/to/license.txt


``qsiprep`` supports a limited number of algorithms that are wrapped in
nipype workflows and can be configured and connected based on the
recon spec JSON file.  The output from one workflow can be the input to
another as long as the output from the upstream workflow matches the inputs to
the downstream workflow. The :ref:`recon_workflows` section lists all the
available workflows and their inputs and outputs.


.. _anat_reqs:

Anatomical Data for Reconstruction Workflows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some reconstruction workflows require additional anatomical data to work properly.
This table shows which reconstruction workflows depend on the availibility of 
anatomical data:


+-----------------------------------------+-------------------+-------------------+--------------+
| Option                                  |    Req. T1w       |  Req. FreeSurfer  |   Req. SDC   |
+=========================================+===================+===================+==============+
|:ref:`mrtrix_multishell_msmt_ACT-hsvs`   |       Yes         |       Yes         |    Yes       |
+-----------------------------------------+-------------------+-------------------+--------------+
|:ref:`mrtrix_multishell_msmt_ACT-fast`   |       Yes         |       No          |    Yes       |
+-----------------------------------------+-------------------+-------------------+--------------+
|:ref:`mrtrix_multishell_msmt_noACT`      |       No          |       Yes         |    No        |
+-----------------------------------------+-------------------+-------------------+--------------+
|:ref:`mrtrix_singleshell_ss3t_ACT-hsvs`  |       Yes         |       No          |    No        |
+-----------------------------------------+-------------------+-------------------+--------------+
|:ref:`mrtrix_singleshell_ss3t_ACT-fast`  |       Yes         |       No          |    No        |
+-----------------------------------------+-------------------+-------------------+--------------+
|:ref:`mrtrix_singleshell_ss3t_noACT`     |       Yes         |       No          |    No        |
+-----------------------------------------+-------------------+-------------------+--------------+
|:ref:`pyafq`                             |       No          |       No          |    No        |
+-----------------------------------------+-------------------+-------------------+--------------+
|:ref:`pyafq_input_trk`                   |       No          |       No          |    No        |
+-----------------------------------------+-------------------+-------------------+--------------+
|:ref:`amico_noddi`                       |       No          |       No          |    No        |
+-----------------------------------------+-------------------+-------------------+--------------+
|:ref:`dsi_studio_gqi`                    |       No          |       No          | Recommended  |
+-----------------------------------------+-------------------+-------------------+--------------+
|:ref:`dipy_mapmri`                       |       No          |       No          | Recommended  |
+-----------------------------------------+-------------------+-------------------+--------------+
|:ref:`dipy_3dshore`                      |       No          |       No          | Recommended  |
+-----------------------------------------+-------------------+-------------------+--------------+
|:ref:`csdsi_3dshore`                     |       No          |       No          | Recommended  |
+-----------------------------------------+-------------------+-------------------+--------------+
|:ref:`reorient_fslstd`                   |       No          |       No          |    No        |
+-----------------------------------------+-------------------+-------------------+--------------+

Data preprocessed by ``qsiprep`` may be missing a preprocessed T1w image if the ``--dwi-only`` flag
was used. This is not a problem because anatomical data can be introduced during the Reconstruction
workflows! Suppose you ran FreeSurfer on your data separately (e.g. as part of fmriprep). You can
specify the directory containing freesurfer outputs with the ``--freesurfer-input`` flag. If you 
have::
  
    derivatives/freesurfer/sub-x
    derivatives/freesurfer/sub-y
    derivatives/freesurfer/sub-z

and from ``qsiprep``::

    derivatives/qsiprep/sub-x
    derivatives/qsiprep/sub-y
    derivatives/qsiprep/sub-z

You can run::

  $ qsiprep-docker \
      derivatives/qsiprep derivatives participant \
      --recon_input derivatives/qsiprep \
      --recon_spec mrtrix_multishell_msmt_ACT-hsvs \
      --freesurfer-input derivatives/freesurfer \
      --fs-license-file /path/to/license.txt

This will read the FreeSurfer data, align it to the ``qsiprep`` results and use it
for subsequent reconstruction steps. The ``--freesurfer-input`` flag can be included
regardless even if the ``--dwi-only`` flag wasn't used. This means two possible
things can happen

If ``qsiprep`` performed anatomical preprocessing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In most cases human MRI experiments include a T1-weighted anatomical image.
By default ``qsiprep`` performs some processing steps on this image, 
including brain extraction and spatial normalization to the MNI152NLin2009cAsym
template (unless ``--infant`` was specified, then the infant template is used).

If a T1w image is available in the input BIDS data and was preprocessed by 
``qsiprep``, the ``brain.mgz`` image from freesurfer is registered to the 
AC-PC and DWI-aligned ``desc-preproc_T1w.nii`` image in the ``qsiprep`` outputs.
The transform from freesurfer native space into alignment with the ``qsiprep`` 
outputs is achieved by converting ``brain.mgz`` into NIfTI format and adjusting
the affine matrix such that the images are aligned in world coordinates. This 
prevents an extra interpolation.


If ``--dwi-only`` was used
^^^^^^^^^^^^^^^^^^^^^^^^^^

If ``--dwi-only`` was used, there will be no preprocessed T1w data in the 
``qsiprep`` results. Instead the DWI images have been aligned to AC-PC 
as closely as possibly (likely imperfectly). In this case, the FreeSurfer
skull-stripped ``brain.mgz`` is rigidly registered to ``dwiref`` of each
preprocessed DWI. The FreeSurfer brain mask is resampled to the grid of
the DWI. 

If structural connectivity is calculated during the reconstruction workflow
(or any atlases are specified in the ``"anatomical": []`` section of the 
workflow's ``.json`` file), the coregistered-to-DWI ``brain.mgz`` image will be 
normalized to the MNI152NLin2009cAsym template using ``antsRegistration``.
The reverse transform is used to get parcellations aligned to the DWI.

.. _masking:

How masking is incorporated
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A brain mask is provided as default input to all reconstruction workflows.
The source of the brain mask depends on available data and user options.

  * If ``qsiprep`` ran normally and the reconstruction workflows are run
    without ``--dwi-only``, the brain mask estimated by ``antsBrainExtraction``
    during preprocessing is used.
  * If no T1w data is available in the ``qsiprep`` outputs and the user 
    supplies FreeSurfer data with ``--freesurfer-input``, the brain mask 
    created by FreeSurfer is used.
  * If you specify ``--dwi-only`` when ``qsiprep`` performs the reconstruction
    OR if no preprocessed T1w images (either via ``qsiprep`` outputs or 
    ``--freesurfer-input``) are available, a mask is estimated based on the
    preprocessed DWI data. This is the least robust option and should be 
    avoided if at all possible.


.. _preconfigured_workflows:

Pre-configured recon_workflows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
  The MRtrix workflows are identical up to the FOD estimation. In each case the fiber response
  function is estimated using ``dwi2response dhollander`` [Dhollander2019]_ with a mask based on
  the T1w. The main differences are in

    * the CSD algorithm used in dwi2fod (msmt_csd or ss3t_csd)
    * whether a T1w-based tissue segmentation is used during tractography

  In the ``*_noACT`` versions of the pipelines, no T1w-based segmentation is used during
  tractography. Otherwise, cropping is performed at the GM/WM interface, along with backtracking.

  In all pipelines, tractography is performed using
  tckgen_, which uses the iFOD2 probabilistic tracking method to generate 1e7 streamlines with a
  maximum length of 250mm, minimum length of 30mm, FOD power of 0.33. Weights for each streamline
  were calculated using SIFT2_ [Smith2015]_ and were included for while estimating the
  structural connectivity matrix.


.. warning::
  We don't recommend using ACT with FAST segmentations. The full benefits of ACT
  require very precise tissue boundaries and FAST just doesn't do this reliably
  enough. We strongly recommend the ``hsvs`` segmentation if you're going to 
  use ACT. Note that this requires ``--freesurfer-input``

.. _mrtrix_multishell_msmt_ACT-hsvs:

``mrtrix_multishell_msmt_ACT-hsvs``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This workflow uses the ``msmt_csd`` algorithm [Jeurissen2014]_ to estimate FODs for white matter,
gray matter and cerebrospinal fluid using *multi-shell acquisitions*. The white matter FODs are
used for tractography and the T1w segmentation is used for anatomical constraints [Smith2012]_.
The T1w segmentation uses the hybrid surface volume segmentation (hsvs) [Smith2020]_ and 
requires ``--freesurfer-input``.


.. _mrtrix_multishell_msmt_ACT-fast:

``mrtrix_multishell_msmt_ACT-fast``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Identical to :ref:`mrtrix_multishell_msmt_ACT-hsvs` except FSL's FAST is used for 
tissue segmentation. This workflow is not recommended.


.. _mrtrix_multishell_msmt_noACT:

``mrtrix_multishell_msmt_noACT``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This workflow uses the ``msmt_csd`` algorithm [Jeurissen2014]_ to estimate FODs for white matter,
gray matter and cerebrospinal fluid using *multi-shell acquisitions*. The white matter FODs are
used for tractography with no T1w-based anatomical constraints.


.. _mrtrix_singleshell_ss3t_ACT-hsvs:

``mrtrix_singleshell_ss3t_ACT-hsvs``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This workflow uses the ``ss3t_csd_beta1`` algorithm [Dhollander2016]_ to estimate FODs for white
matter, and cerebrospinal fluid using *single shell (DTI) acquisitions*. The white matter FODs are
used for tractography and the T1w segmentation is used for anatomical constraints [Smith2012]_.
The T1w segmentation uses the hybrid surface volume segmentation (hsvs) [Smith2020]_ and 
requires ``--freesurfer-input``.

.. _mrtrix_singleshell_ss3t_ACT-fast:

``mrtrix_multishell_msmt_ACT-fast``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Identical to :ref:`mrtrix_singleshell_ss3t_ACT-hsvs` except FSL's FAST is used for 
tissue segmentation. This workflow is not recommended.

.. _mrtrix_singleshell_ss3t_noACT:

``mrtrix_singleshell_ss3t_noACT``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This workflow uses the ``ss3t_csd_beta1`` algorithm [Dhollander2016]_ to estimate FODs for white
matter, and cerebrospinal fluid using *single shell (DTI) acquisitions*. The white matter FODs are
used for tractography with no T1w-based anatomical constraints.

.. _pyafq:

``pyafq_tractometry``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This workflow uses the AFQ [Yeatman2012]_ implemented in Python [Kruper2021]_ to recognize
major white matter pathways within the tractography, and then extract tissue properties along
those pathways. See the `pyAFQ documentation <https://yeatmanlab.github.io/pyAFQ/>`_ .

.. _pyafq_input_trk:

``mrtrix_multishell_msmt_pyafq_tractometry``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Identical to :ref:`pyafq` except that tractography generated using IFOD2 from MRTrix3,
instead of using pyAFQ's default DIPY tractography.
This can also be used as an example for how to import tractographies from other
reconstruciton pipelines to pyAFQ.


.. _amico_noddi:

``amico_noddi``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This workflow estimates the NODDI [Zhang2012]_ model using the implementation from
AMICO [Daducci2015]_. Images with intra-cellular volume fraction (ICVF), isotropic volume
fraction (ISOVF), orientation dispersion (OD) are written to outputs. Additionally, a DSI
Studio fib file is created using the peak directions and ICVF as a stand-in for QA to be
used for tractography.

.. _dsi_studio_gqi:

``dsi_studio_gqi``
^^^^^^^^^^^^^^^^^^

Here the standard GQI plus deterministic tractography pipeline is used [Yeh2013]_.  GQI works on
almost any imaginable sampling scheme because DSI Studio will internally interpolate the q-space
data so  symmetry requirements are met. GQI models the water diffusion ODF, so ODF peaks are much
smaller  than you see with CSD. This results in a rather conservative peak detection, which greatly
benefits from having more diffusion data than a typical DTI.

5 million streamlines are created with a maximum length of 250mm, minimum length of 30mm,
random seeding, a step size of 1mm and an automatically calculated QA threshold.

Additionally, a number of anisotropy scalar images are produced such as QA, GFA and ISO.

.. _dipy_mapmri:

``dipy_mapmri``
^^^^^^^^^^^^^^^

The MAPMRI method is used to estimate EAPs from which ODFs are calculated analytically. This
method produces scalars like RTOP, RTAP, QIV, MSD, etc.

The ODFs are saved in DSI Studio format and tractography is run identically to that in
:ref:`dsi_studio_gqi`.


.. _dipy_3dshore:

``dipy_3dshore``
^^^^^^^^^^^^^^^^

This uses the BrainSuite 3dSHORE basis in a Dipy reconstruction. Much like :ref:`dipy_mapmri`,
a slew of anisotropy scalars are estimated. Here the :ref:`dsi_studio_gqi` fiber tracking is
again run on the 3dSHORE-estimated ODFs.

.. _reorient_fslstd:

``reorient_fslstd``
^^^^^^^^^^^^^^^^^^^

Reorients the ``qsiprep`` preprocessed DWI and bval/bvec to the standard FSL orientation.
This can be useful if FSL tools will be applied outside of ``qsiprep``.


.. _csdsi_3dshore:

``csdsi_3dshore``
^^^^^^^^^^^^^^^^^

**[EXPERIMENTAL]** This pipeline is for DSI or compressed-sensing DSI. The first step is a
L2-regularized 3dSHORE reconstruction of the ensemble average propagator in each voxel. These EAPs
are then used for two purposes

 1. To calculate ODFs, which are then sent to DSI Studio for tractography
 2. To estimate signal for a multishell (specifically HCP) sampling scheme, which is run
    through the :ref:`mrtrix_multishell_msmt` pipeline

All outputs, including the imputed HCP sequence are saved in the outputs directory.

.. _custom_reconstruction:

Building a custom reconstruction pipeline
==========================================


Instead of going through each possible element of a pipeline, we will go through
a simple example and describe its components.

Simple DSI Studio example
~~~~~~~~~~~~~~~~~~~~~~~~~~

The reconstruction pipeline is created by the user and specified in a JSON
file similar to the following::

  {
    "name": "dsistudio_pipeline",
    "space": "T1w",
    "anatomical": ["mrtrix_5tt_fast"],
    "atlases": ["schaefer100x7", "schaefer100x17", "schaefer200x7", "schaefer200x7", "schaefer400x7", "schaefer400x17", "brainnetome246", "aicha384", "gordon333", "aal116", "power264"],
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
      /path/to/bids /where/my/reconstructed/data/goes participant \
      --recon_input /output/from/qsiprep \
      --recon_spec gqi_scalar_export.json \
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


.. _connectivity:

Reconstruction Outputs: Connectivity matrices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instead of offering a bewildering number of options for constructing connectivity matrices,
``qsiprep`` will construct as many connectivity matrices as it can given the reconstruction
methods. It is **highly** recommended that you pick a weighting scheme before you run
these pipelines and only look at those numbers. If you look at more than one weighting method
be sure to adjust your statistics for the additional comparisons.

.. _connectivity_atlases:

Atlases
^^^^^^^

The following atlases are included in ``qsiprep`` and are used by default in the
:ref:`preconfigured_workflows`. If you use one of them please be sure to cite
the relevant publication.

 * ``schaefer100x7``, ``schaefer100x17``, ``schaefer200x7``, ``schaefer200x17``,
   ``schaefer400x7``, ``schaefer400x17``: [Schaefer2017]_, [Yeo2011]_
 * ``brainnetome246``: [Fan2016]_
 * ``aicha384``: [Joliot2015]_
 * ``gordon333``: [Gordon2014]_
 * ``aal116``: [TzourioMazoyer2002]_
 * ``power264``: [Power2011]_

.. _custom_atlases:

Using custom atlases
^^^^^^^^^^^^^^^^^^^^

It's possible to use your own atlases provided you can match the format ``qsiprep`` uses to
read atlases. The ``qsiprep`` atlas set can be downloaded directly from
`box  <https://upenn.box.com/shared/static/8k17yt2rfeqm3emzol5sa0j9fh3dhs0i.xz>`_.

In this directory there must exist a JSON file called ``atlas_config.json`` containing an
entry for each atlas you would like included. The format is::

  {
    "my_custom_atlas": {
      "file": "file_in_this_directory.nii.gz",
      "node_names": ["Region1_L", "Region1_R" ... "RegionN_R"],
      "node_ids": [1, 2, ..., N]
    }
    ...
  }

Where ``"node_names"`` are the text names of the regions in ``"my_custom_atlas"`` and
``"node_ids"`` are the numbers in the nifti file that correspond to each region. When
:ref:`custom_reconstruction` you can then inclued ``"my_custom_atlas"`` in the ``"atlases":[]``
section.

The directory containing ``atlas_config.json`` and the atlas nifti files should be mounted in
the container at ``/atlas/qsirecon_atlases``. If using ``qsiprep-docker`` or
``qsiprep-singularity`` this can be done with ``--custom-atlases /path/to/my/atlases`` or
if you're running on your own system (not recommended) you can set the environment variable
``QSIRECON_ATLAS=/path/to/my/atlases``.

The nifti images should be registered to the
`MNI152NLin2009cAsym <https://github.com/PennBBL/qsiprep/blob/master/qsiprep/data/mni_1mm_t1w_lps.nii.gz>`_
included in ``qsiprep``.
It is essential that your images are in the LPS+ orientation and have the sform zeroed-out
in the header. **Be sure to check for alignment and orientation** in your outputs.