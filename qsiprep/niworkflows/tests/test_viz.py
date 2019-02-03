# -*- coding: utf-8 -*-
"""Test viz module"""
import os
import nibabel as nb
from .. import viz
from .conftest import datadir


def test_carpetplot():
    """Write a carpetplot"""
    out_file = None
    save_artifacts = os.getenv('SAVE_CIRCLE_ARTIFACTS', False)
    if save_artifacts:
        out_file = os.path.join(save_artifacts, 'carpetplot.svg')
    viz.plot_carpet(
        os.path.join(datadir, 'sub-ds205s03_task-functionallocalizer_run-01_bold_volreg.nii.gz'),
        nb.load(os.path.join(
            datadir,
            'sub-ds205s03_task-functionallocalizer_run-01_bold_parc.nii.gz')).get_data(),
        output_file=out_file,
        legend=True
    )
