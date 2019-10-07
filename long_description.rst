qsiprep borrows heavily from FMRIPREP to build workflows for preprocessing q-space images
such as Diffusion Spectrum Images (DSI), multi-shell HARDI and compressed sensing DSI (CS-DSI).
It utilizes Dipy and ANTs to implement a novel high-b-value head motion correction approach
using q-space methods such as 3dSHORE to iteratively generate head motion target images for each
gradient direction and strength.

Since qsiprep uses the FMRIPREP workflow-building strategy, it can also generate methods
boilerplate and quality-check figures.

Users can also reconstruct orientation distribution functions (ODFs), fiber orientation
distributions (FODs) and perform tractography, estimate anisotropy scalars and connectivity
estimation using a combination of Dipy, MRTrix and DSI Studio using a JSON-based pipeline
specification.

[Documentation `qsiprep.org <https://qsiprep.readthedocs.io>`_]
