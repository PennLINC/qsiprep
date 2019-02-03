# -*- coding: utf-8 -*-
""" Utilities tests """

from __future__ import absolute_import, division, print_function, unicode_literals

import os
from qsiprep.niworkflows.interfaces.masks import SimpleShowMaskRPT


def test_compression(oasis_dir):
    """ the BET report capable test """

    uncompressed = SimpleShowMaskRPT(
        generate_report=True,
        background_file=str(oasis_dir / 'tpl-OASIS30ANTs_res-01_T1w.nii.gz'),
        mask_file=str(oasis_dir /
                      'tpl-OASIS30ANTs_res-01_label-BrainCerebellumRegistration_roi.nii.gz'),
        compress_report=False
    ).run().outputs.out_report

    compressed = SimpleShowMaskRPT(
        generate_report=True,
        background_file=str(oasis_dir / 'tpl-OASIS30ANTs_res-01_T1w.nii.gz'),
        mask_file=str(oasis_dir /
                      'tpl-OASIS30ANTs_res-01_label-BrainCerebellumRegistration_roi.nii.gz'),
        compress_report=True
    ).run().outputs.out_report

    size = int(os.stat(uncompressed).st_size)
    size_compress = int(os.stat(compressed).st_size)
    assert size >= size_compress, ('The uncompressed report is smaller (%d)'
                                   'than the compressed report (%d)' % (size, size_compress))
