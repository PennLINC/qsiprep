.. _merging:

Merging multiple scans from a session
--------------------------------------

Introduction
~~~~~~~~~~~~

For q-space imaging sequences it is common to have multiple separate scans to
acquire the entire sampling scheme. These scans get aligned and merged into
a single DWI series before reconstruction. It is also common to collect
a DWI scan (or scans) in the reverse phase encoding direction to use for
susceptibility distortion correction (SDC).

This creates a number of possible scenarios for preprocessing your DWIs. These
scenarios can be controlled by the ``combine_all_dwis`` argument. If your study
has multiple sessions, DWI scans will *never* be combined across sessions.
Merging only occurs within a session.

If ``combine_all_dwis`` is ``False`` (not present in the commandline call), each dwi
scan in the ``dwi`` directories will be processed independently. You will have one
preprocessed output per each DWI file in your input.

If ``combine_all_dwis`` is set to ``True``, two possibilities arise. If all DWIs in a session
are in the same PE direction, they will be merged into a single series. If there are
two PE directions detected in the DWI scans and ``'fieldmaps'`` is not in ``ignore``,
images are combined according to their PE direction, and their b0 reference images are used to
perform SDC. Either way, at the end of the day, there will be one preprocessed DWI
file per session in your input.

If you have some scans you want to combine and others you want to preprocess separately,
consider creating fake sessions in your BIDS directory.


Correction methods
~~~~~~~~~~~~~~~~~~

The are two kinds of SDC available in qsiprep:

  1. :ref:`sdc_pepolar` (also called **blip-up/blip-down**):
     This is the implementation from FMRIPREP, using 3dQwarp to
     correct a DWI series using a fieldmap in the fmaps directory.

  1. (a) This uses two separe Blip-Up/blip-Down Series (BUDS), again
      using FMRIPREP's 3dQwarp implementation, but applying the
      correct unwarping transformation depending on each scan's polarity.

  2. :ref:`sdc_fieldmapless`: The SyN-based susceptibility distortion correction
     implemented in FMRIPREP


``qsiprep`` determines if a fieldmap should be used based on the ``"IntendedFor"``
fields in the JSON sidecars in the ``fmap/`` directory. If you have two DWI
series with reverse phase encodings, but would rather use method 1 instead of
1a, include the ``--prefer-dedicated-fmaps`` argument.
