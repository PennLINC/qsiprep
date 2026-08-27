"""The TRXScan simulator and its truth-data kit are available in the image.

These run only where the base image provides /opt/trxscan (the in-container
unit_tests CI job); locally they skip. They pin the contract the simulation
recovery tests will build on: binaries on PATH and the sub-0001a reference
kit in a known layout.
"""

import shutil
from pathlib import Path

import pytest

KIT = Path('/opt/trxscan/data')

pytestmark = pytest.mark.skipif(
    shutil.which('trxscan') is None, reason='TRXScan not installed'
)


def test_trxscan_binaries_on_path():
    assert shutil.which('trxscan')
    assert shutil.which('trxscan-microstructure')


def test_truth_data_kit_layout():
    anat = KIT / 'sub-0001a' / 'anat'
    for label in ('WM', 'GM', 'CSF'):
        assert (anat / f'sub-0001a_space-ACPC_label-{label}_probseg.nii.gz').is_file()
    assert (anat / 'sub-0001a_space-ACPC_desc-atlas_fieldmap.nii.gz').is_file()
    tracks = KIT / 'sub-0001a' / 'tract' / 'sub-0001a_space-ACPC_desc-actsift2_tracks.trx'
    assert tracks.is_file()
    for scheme in ('hbcd_ap', 'hbcd_pa'):
        assert (KIT / 'scheme' / f'{scheme}.bval').is_file()
        assert (KIT / 'scheme' / f'{scheme}.bvec').is_file()
    assert (KIT / 'scripts' / 'prepare_acquisition_grid.py').is_file()
