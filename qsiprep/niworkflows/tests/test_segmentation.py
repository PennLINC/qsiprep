# -*- coding: utf-8 -*-
""" Segmentation tests """

from __future__ import absolute_import, division, print_function, unicode_literals

import os
from shutil import copy
import pytest

from nipype.interfaces.base import Bunch
from qsiprep.niworkflows.interfaces.segmentation import FASTRPT, ReconAllRPT
from qsiprep.niworkflows.interfaces.masks import (
    BETRPT, BrainExtractionRPT, SimpleShowMaskRPT, ROIsPlot
)
from .conftest import _run_interface_mock, datadir


def _smoke_test_report(report_interface, artifact_name):
    report_interface.run()
    out_report = report_interface.inputs.out_report

    save_artifacts = os.getenv('SAVE_CIRCLE_ARTIFACTS', False)
    if save_artifacts:
        copy(out_report, os.path.join(save_artifacts, artifact_name))
    assert os.path.isfile(out_report), 'Report "%s" does not exist' % out_report


def test_BETRPT(moving):
    """ the BET report capable test """
    bet_rpt = BETRPT(generate_report=True, in_file=moving)
    _smoke_test_report(bet_rpt, 'testBET.svg')


def test_ROIsPlot(oasis_dir):
    """ the BET report capable test """
    import nibabel as nb
    import numpy as np

    im = nb.load(str(oasis_dir / 'tpl-OASIS30ANTs_res-01_variant-4_dtissue.nii.gz'))
    lookup = np.zeros(5, dtype=int)
    lookup[1] = 1
    lookup[2] = 4
    lookup[3] = 2
    lookup[4] = 3
    newdata = lookup[np.round(im.get_data()).astype(int)]
    hdr = im.header.copy()
    hdr.set_data_dtype('int16')
    hdr['scl_slope'] = 1
    hdr['scl_inter'] = 0
    out_file = os.path.abspath('segments.nii.gz')
    nb.Nifti1Image(newdata, im.affine, hdr).to_filename(out_file)
    roi_rpt = ROIsPlot(
        generate_report=True,
        in_file=str(oasis_dir / 'tpl-OASIS30ANTs_res-01_T1w.nii.gz'),
        in_mask=str(oasis_dir / 'tpl-OASIS30ANTs_res-01_brainmask.nii.gz'),
        in_rois=[out_file],
        levels=[1.5, 2.5, 3.5],
        colors=['gold', 'magenta', 'b'],
    )
    _smoke_test_report(roi_rpt, 'testROIsPlot.svg')


def test_ROIsPlot2(oasis_dir):
    """ the BET report capable test """
    import nibabel as nb
    import numpy as np

    im = nb.load(str(oasis_dir / 'tpl-OASIS30ANTs_res-01_variant-4_dtissue.nii.gz'))
    lookup = np.zeros(5, dtype=int)
    lookup[1] = 1
    lookup[2] = 4
    lookup[3] = 2
    lookup[4] = 3
    newdata = lookup[np.round(im.get_data()).astype(int)]
    hdr = im.header.copy()
    hdr.set_data_dtype('int16')
    hdr['scl_slope'] = 1
    hdr['scl_inter'] = 0

    out_files = []
    for i in range(1, 5):
        seg = np.zeros_like(newdata, dtype='uint8')
        seg[(newdata > 0) & (newdata <= i)] = 1
        out_file = os.path.abspath('segments%02d.nii.gz' % i)
        nb.Nifti1Image(seg, im.affine, hdr).to_filename(out_file)
        out_files.append(out_file)
    roi_rpt = ROIsPlot(
        generate_report=True,
        in_file=str(oasis_dir / 'tpl-OASIS30ANTs_res-01_T1w.nii.gz'),
        in_mask=str(oasis_dir / 'tpl-OASIS30ANTs_res-01_brainmask.nii.gz'),
        in_rois=out_files,
        colors=['gold', 'lightblue', 'b', 'g']
    )
    _smoke_test_report(roi_rpt, 'testROIsPlot2.svg')


def test_SimpleShowMaskRPT(oasis_dir):
    """ the BET report capable test """

    msk_rpt = SimpleShowMaskRPT(
        generate_report=True,
        background_file=str(oasis_dir / 'tpl-OASIS30ANTs_res-01_T1w.nii.gz'),
        mask_file=str(oasis_dir /
                      'tpl-OASIS30ANTs_res-01_label-BrainCerebellumRegistration_roi.nii.gz')
    )
    _smoke_test_report(msk_rpt, 'testSimpleMask.svg')


def test_BrainExtractionRPT(monkeypatch, oasis_dir, moving, nthreads):
    """ test antsBrainExtraction with reports"""

    def _agg(objekt, runtime):
        outputs = Bunch(BrainExtractionMask=os.path.join(
            datadir, 'testBrainExtractionRPTBrainExtractionMask.nii.gz')
        )
        return outputs

    # Patch the _run_interface method
    monkeypatch.setattr(BrainExtractionRPT, '_run_interface',
                        _run_interface_mock)
    monkeypatch.setattr(BrainExtractionRPT, 'aggregate_outputs',
                        _agg)

    bex_rpt = BrainExtractionRPT(
        generate_report=True,
        dimension=3,
        use_floatingpoint_precision=1,
        anatomical_image=moving,
        brain_template=str(oasis_dir / 'tpl-OASIS30ANTs_res-01_T1w.nii.gz'),
        brain_probability_mask=str(
            oasis_dir /
            'tpl-OASIS30ANTs_res-01_class-brainmask_probtissue.nii.gz'),
        extraction_registration_mask=str(
            oasis_dir /
            'tpl-OASIS30ANTs_res-01_label-BrainCerebellumRegistration_roi.nii.gz'),
        out_prefix='testBrainExtractionRPT',
        debug=True,  # run faster for testing purposes
        num_threads=nthreads
    )
    _smoke_test_report(bex_rpt, 'testANTSBrainExtraction.svg')


@pytest.mark.parametrize("segments", [True, False])
def test_FASTRPT(monkeypatch, segments, reference, reference_mask):
    """ test FAST with the two options for segments """
    from nipype.interfaces.fsl.maths import ApplyMask

    def _agg(objekt, runtime):
        outputs = Bunch(tissue_class_map=os.path.join(
            datadir, 'testFASTRPT-tissue_class_map.nii.gz'),
            tissue_class_files=[
                os.path.join(datadir, 'testFASTRPT-tissue_class_files0.nii.gz'),
                os.path.join(datadir, 'testFASTRPT-tissue_class_files1.nii.gz'),
                os.path.join(datadir, 'testFASTRPT-tissue_class_files2.nii.gz')]
        )
        return outputs

    # Patch the _run_interface method
    monkeypatch.setattr(FASTRPT, '_run_interface',
                        _run_interface_mock)
    monkeypatch.setattr(FASTRPT, 'aggregate_outputs',
                        _agg)

    brain = ApplyMask(
        in_file=reference, mask_file=reference_mask).run().outputs.out_file
    fast_rpt = FASTRPT(
        in_files=brain,
        generate_report=True,
        no_bias=True,
        probability_maps=True,
        segments=segments,
        out_basename='test'
    )
    _smoke_test_report(
        fast_rpt, 'testFAST_%ssegments.svg' % ('no' * int(not segments)))


def test_ReconAllRPT(monkeypatch):
    # Patch the _run_interface method
    monkeypatch.setattr(ReconAllRPT, '_run_interface',
                        _run_interface_mock)

    rall_rpt = ReconAllRPT(
        subject_id='fsaverage',
        directive='all',
        subjects_dir=os.getenv('SUBJECTS_DIR'),
        generate_report=True
    )

    _smoke_test_report(rall_rpt, 'testReconAll.svg')
