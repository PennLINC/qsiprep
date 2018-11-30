.. _merging:

Merging multiple scans from a session
--------------------------------------

Introduction
~~~~~~~~~~~~

For q space imaging sequences it is common to have multiple separate scans to
acquire the entire sampling scheme. These scans get aligned and merged into
a single DWI series before reconstruction. It is also common to collect
a DWI scan (or scans) in the reverse phase encoding direction to use for
susceptibility distortion correction (SDC).

This creates a number of possible scenarios for preprocessing your DWIs. These
scenarios can be controlled by the `combine_all_dwis` argument. If your study
has multiple sessions, DWI scans will *never* be combined across sessions.
Merging only occurs within a session.

If `combine_all_dwis` is `False`, each dwi scan in the `dwi` directories will be processed
independently. You will have one preprocessed output per each DWI file in your input.

If `combine_all_dwis` is set to `True`, two possibilities arise. If all DWIs in a session
are in the same PE direction, they will be merged into a single series. If there are
two PE directions detected in the DWI scans and `'fieldmaps'` is not in `ignore`,
images are combined according to their PE direction, and their b0 targets are used to
perform SDC. Either way, at the end of the day, there will be one preprocessed DWI
file per session in your input.

If you have some scans you want to combine and others you want to preprocess separately,
consider creating fake sessions in your BIDS directory.


Correction methods
~~~~~~~~~~~~~~~~~~

The are two kinds of SDC available in QSIPREP:

  1. :ref:`sdc_pepolar` (also called **blip-up/blip-down**):
     acquire at least two images with varying :abbr:`PE (phase-encoding)` directions.
     Hence, the realization of distortion is different between the different
     acquisitions. The displacements map :math:`d_\text{PE}(x, y, z)` is
     estimated with an image registration process between the different
     :abbr:`PE (phase-encoding)` acquisitions, regularized by the
     readout time :math:`T_\text{ro}`.

  4. :ref:`sdc_fieldmapless`: qsiprep now experimentally supports displacement
     field estimation in the absence of fieldmaps via nonlinear registration.


In order to select the appropriate estimation workflow, the input BIDS dataset is
first queried to find the available field-mapping techniques (see :ref:`sdc_base`).
Once the field-map (or the corresponding displacement field) is estimated, the
distortion can be accounted for (see :ref:`sdc_unwarp`).
