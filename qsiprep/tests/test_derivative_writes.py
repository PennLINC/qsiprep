"""Tests for how DWI derivatives are written.

4D sinks run in worker processes, and derivatives that niworkflows rewrites
are compressed with ISA-L.
"""

import gzip

import nibabel as nb
import numpy as np
import pytest
from niworkflows.interfaces import bids as niworkflows_bids
from niworkflows.utils.images import unsafe_write_nifti_header_and_data as zlib_writer

from qsiprep.interfaces import bids


def _image(shape=(6, 5, 4, 3), dtype='float32'):
    rng = np.random.default_rng(0)
    data = (rng.standard_normal(shape) * 100 + 1000).astype(dtype)
    return nb.Nifti1Image(data, np.diag([1.8, 1.8, 1.8, 1.0]))


def test_niworkflows_sinks_write_with_isal():
    """Test that importing qsiprep's sinks routes niworkflows' writer to ISA-L."""
    assert niworkflows_bids.unsafe_write_nifti_header_and_data is bids.write_nifti_isal


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'int16'])
def test_isal_writer_matches_the_zlib_writer(tmp_path, dtype):
    """Test that the ISA-L file decompresses to exactly the bytes niworkflows writes."""
    img = _image(dtype=dtype)
    data = np.asanyarray(img.dataobj)
    zlib_writer(tmp_path / 'zlib.nii.gz', img.header, data)
    bids.write_nifti_isal(tmp_path / 'isal.nii.gz', img.header, data)

    with gzip.open(tmp_path / 'zlib.nii.gz') as a, gzip.open(tmp_path / 'isal.nii.gz') as b:
        assert a.read() == b.read()
    out = nb.load(tmp_path / 'isal.nii.gz')
    assert out.get_data_dtype() == np.dtype(dtype)
    assert np.array_equal(np.asanyarray(out.dataobj), data)


def test_isal_writer_is_deterministic(tmp_path):
    """Test that two writes of one image are byte-identical (no name or time in the header)."""
    img = _image()
    data = np.asanyarray(img.dataobj)
    bids.write_nifti_isal(tmp_path / 'a.nii.gz', img.header, data)
    bids.write_nifti_isal(tmp_path / 'b.nii.gz', img.header, data)
    assert (tmp_path / 'a.nii.gz').read_bytes() == (tmp_path / 'b.nii.gz').read_bytes()


def test_isal_writer_leaves_uncompressed_files_uncompressed(tmp_path):
    img = _image()
    data = np.asanyarray(img.dataobj)
    bids.write_nifti_isal(tmp_path / 'x.nii', img.header, data)
    assert (tmp_path / 'x.nii').read_bytes()[344:348] != b'\x1f\x8b\x08\x00'
    assert np.array_equal(np.asanyarray(nb.load(tmp_path / 'x.nii').dataobj), data)


def test_datasink_compresses_a_4d_series(tmp_path):
    """Test a datasink run end to end: an uncompressed input written as .nii.gz."""
    from qsiprep.interfaces import DerivativesDataSink

    img = _image()
    img.to_filename(tmp_path / 'in.nii')
    result = DerivativesDataSink(
        base_directory=str(tmp_path / 'out'),
        source_file='/data/sub-01/dwi/sub-01_dwi.nii.gz',
        space='ACPC',
        desc='preproc',
        suffix='dwi',
        extension='.nii.gz',
        compress=True,
        in_file=str(tmp_path / 'in.nii'),
    ).run()
    out = result.outputs.out_file
    out = out[0] if isinstance(out, list) else out
    assert out.endswith('.nii.gz')
    assert np.array_equal(np.asanyarray(nb.load(out).dataobj), np.asanyarray(img.dataobj))


def test_4d_dwi_sinks_run_in_workers(tmp_path):
    """Test that the 4D sinks are submitted to workers and the small ones are not."""
    from qsiprep import config
    from qsiprep.workflows.dwi.derivatives import init_dwi_derivatives_wf

    config.execution.output_dir = str(tmp_path)
    config.workflow.hmc_method = 'tortoise'
    config.workflow.write_local_bvecs = False

    wf = init_dwi_derivatives_wf('/data/sub-01/ses-1/dwi/sub-01_ses-1_dwi.nii.gz')
    nodes = {n.name: n for n in wf._get_all_nodes()}
    assert not nodes['ds_dwi_t1'].run_without_submitting
    if 'ds_jacobian' in nodes:
        assert not nodes['ds_jacobian'].run_without_submitting
    assert nodes['ds_bvals_t1'].run_without_submitting
    assert nodes['ds_dwi_mask_t1'].run_without_submitting


@pytest.mark.parametrize('flag', [False, True])
def test_jacobian_4d_flag_reaches_the_stacking_node(tmp_path, flag):
    """Test that --write-jacobian-4d is passed to StackJacobianWeights."""
    from qsiprep import config
    from qsiprep.cli.parser import _build_parser
    from qsiprep.workflows.dwi.derivatives import init_dwi_derivatives_wf

    action = next(a for a in _build_parser()._actions if a.dest == 'write_jacobian_4d')
    assert action.option_strings == ['--write-jacobian-4d']
    assert action.default is False

    config.execution.output_dir = str(tmp_path)
    config.workflow.hmc_method = 'tortoise'
    config.workflow.write_local_bvecs = False
    config.workflow.write_jacobian_4d = flag
    try:
        wf = init_dwi_derivatives_wf('/data/sub-01/ses-1/dwi/sub-01_ses-1_dwi.nii.gz')
        nodes = {n.name: n for n in wf._get_all_nodes()}
        if 'stack_jacobian' not in nodes:
            pytest.skip('Jacobian modulation is off in this configuration')
        assert nodes['stack_jacobian'].inputs.volumes is flag
    finally:
        config.workflow.write_jacobian_4d = False
