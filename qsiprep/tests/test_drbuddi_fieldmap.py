"""The DRBUDDI displacement field reaches the derivatives as a Hz fieldmap.

DRBUDDI reports distortion as an ANTs displacement field; these construction
tests check that the workflow turns it into a Hz field, exposes it through every
backend outputnode, and writes it only when a readout time (and, for the
datasink, sidecar metadata) is available.
"""

import numpy as np
import pytest
from qsiplan.models import CorrectionMethod

from qsiprep import config
from qsiprep.tests.preproc_factory import make_preproc_unit


def _cfg():
    config.nipype.omp_nthreads = 1
    config.execution.sloppy = False
    config.execution.output_dir = '/tmp/qsiprep_fieldmap_test_out'
    config.workflow.b0_threshold = 100
    return config


def _write_dwi(path, nvols=6):
    import nibabel as nb

    nb.Nifti1Image(np.zeros((4, 4, 4, nvols), dtype=np.int16), np.eye(4)).to_filename(str(path))
    stem = str(path).split('.nii')[0]
    bvals = np.array([0] + [1000] * (nvols - 1))
    np.savetxt(stem + '.bval', bvals[None, :], fmt='%d')
    np.savetxt(stem + '.bvec', np.zeros((3, nvols)), fmt='%.1f')
    return str(path)


def _rpe_unit(tmp_path, readout_time=0.0917):
    main = _write_dwi(tmp_path / 'sub-01_dir-PA_dwi.nii.gz')
    partner = _write_dwi(tmp_path / 'sub-01_dir-AP_dwi.nii.gz')
    return make_preproc_unit(
        [main, partner],
        method=CorrectionMethod.PEPOLAR,
        pe_dirs={main: 'j', partner: 'j-'},
        readout_time=readout_time,
    )


def test_drbuddi_wf_builds_field_to_hz(tmp_path):
    """The blip-up displacement field is converted to Hz with the lead PE/TRT."""
    _cfg()
    from qsiprep.workflows.fieldmap import init_drbuddi_wf

    wf = init_drbuddi_wf(_rpe_unit(tmp_path), t2w_sdc=False)
    node = wf.get_node('field_to_hz')
    assert node is not None
    # The lead (+polarity) series is PA ('j'); its readout time drives the scale.
    assert node.inputs.pe_dir == 'j'
    assert node.inputs.readout_time == pytest.approx(0.0917)
    # Both blip fields (FINV up, MINV down) feed it, and its Hz output reaches the
    # workflow outputnode.
    edges = wf._graph.edges(data=True)
    assert any(
        u.name == 'drbuddi'
        and v.name == 'field_to_hz'
        and ('deformation_finv', 'displacement_field') in d['connect']
        and ('deformation_minv', 'opposite_displacement_field') in d['connect']
        for u, v, d in edges
    )
    assert any(
        u.name == 'field_to_hz'
        and v.name == 'outputnode'
        and ('fieldmap_hz', 'fieldmap_hz') in d['connect']
        for u, v, d in edges
    )


def test_drbuddi_wf_skips_field_to_hz_without_readout_time(tmp_path, monkeypatch):
    """No readout time, no Hz field: the node is simply absent (no failure)."""
    _cfg()
    import qsiprep.workflows.fieldmap.drbuddi as drbuddi_mod

    monkeypatch.setattr(drbuddi_mod, 'pe_readout_time', lambda unit: None)
    wf = drbuddi_mod.init_drbuddi_wf(_rpe_unit(tmp_path), t2w_sdc=False)
    assert wf.get_node('field_to_hz') is None


def test_backend_outputnodes_expose_fieldmap_hz():
    """Every backend that can run DRBUDDI declares fieldmap_hz on its outputnode."""
    import inspect

    import qsiprep.workflows.dwi.diffprep as dp
    import qsiprep.workflows.dwi.fsl as fsl
    import qsiprep.workflows.dwi.hmc_sdc as hs

    for mod in (dp, fsl, hs):
        assert "'fieldmap_hz'" in inspect.getsource(mod)


def test_derivatives_wf_writes_fieldmap_only_with_meta():
    """The fieldmap datasink appears only when sidecar metadata is supplied."""
    _cfg()
    from qsiprep.workflows.dwi.derivatives import init_dwi_derivatives_wf

    without = init_dwi_derivatives_wf(source_file='/data/sub-01_dwi.nii.gz')
    assert without.get_node('ds_fieldmap_t1') is None

    meta = {'Units': 'Hz', 'EstimationMethod': 'DRBUDDI', 'PhaseEncodingDirection': 'j'}
    with_meta = init_dwi_derivatives_wf(source_file='/data/sub-01_dwi.nii.gz', fieldmap_meta=meta)
    ds = with_meta.get_node('ds_fieldmap_t1')
    assert ds is not None
    assert ds.inputs.suffix == 'fieldmap'
    assert ds.inputs.meta_dict == meta


def test_derivatives_wf_writes_component_fieldmaps():
    """component_specs adds one datasink per QC field, each with its own entities."""
    _cfg()
    from qsiprep.workflows.dwi.derivatives import init_dwi_derivatives_wf

    specs = [
        {'entities': {'direction': 'PA'}, 'meta': {'Units': 'Hz'}},
        {'entities': {'direction': 'AP'}, 'meta': {'Units': 'Hz'}},
        {'entities': {'desc': 'asymmetry'}, 'meta': {'Units': 'Hz'}},
    ]
    wf = init_dwi_derivatives_wf(
        source_file='/data/sub-01_dwi.nii.gz',
        fieldmap_meta={'Units': 'Hz'},
        component_specs=specs,
    )
    assert wf.get_node('ds_component_fieldmap_0').inputs.direction == 'PA'
    assert wf.get_node('ds_component_fieldmap_1').inputs.direction == 'AP'
    assert wf.get_node('ds_component_fieldmap_2').inputs.desc == 'asymmetry'


def test_fieldmap_datasink_builds_a_dwi_path(tmp_path):
    """A 'fieldmap' suffix in the dwi datatype must have a path template.

    Regression: io_spec.json registered 'graddev' but not 'fieldmap' for dwi
    derivatives, so the datasink built the node fine but raised "Could not build
    path" at run time. Exercise the actual path build, not just node existence.
    """
    import nibabel as nb
    import numpy as np

    from qsiprep.interfaces.bids import DerivativesDataSink

    src = tmp_path / 'sub-01_ses-1_acq-HBCD_run-01_dwi.nii.gz'
    img = nb.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4))
    img.to_filename(str(src))
    field = tmp_path / 'fieldmap_hz.nii.gz'
    img.to_filename(str(field))

    ds = DerivativesDataSink(
        base_directory=str(tmp_path / 'out'),
        source_file=str(src),
        space='ACPC',
        suffix='fieldmap',
        extension='.nii.gz',
        compress=True,
        meta_dict={'Units': 'Hz', 'EstimationMethod': 'DRBUDDI'},
    )
    ds.inputs.in_file = str(field)
    out = ds.run().outputs.out_file
    out = out[0] if isinstance(out, list) else out
    assert '/dwi/' in out
    assert out.endswith('_space-ACPC_fieldmap.nii.gz')
