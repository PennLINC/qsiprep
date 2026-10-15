"""Methods boilerplate for the configurations the TRXScan integration fixtures run.

Each test builds the DWI preprocessing workflow the way one ``trxscan_*`` run in
``test_cli.py`` does and asserts on the methods text it would write, so a wrong
statement, a sentence out of order or a missing space shows up without running
the pipeline.
"""

import json
import os
import re

import pytest
from qsiplan.models import CorrectionMethod

from qsiprep import config
from qsiprep.tests.preproc_factory import make_preproc_unit
from qsiprep.tests.test_workflows_native import _StubLayout, _write_dwi
from qsiprep.tests.utils import get_test_data_path

TEMPLATE = 'MNI152NLin2009cAsym'


@pytest.fixture
def cfg(monkeypatch):
    """Configure as ``TRXSCAN_COMMON`` does, restoring every setting afterwards."""
    monkeypatch.setenv('FSLDIR', '/tmp/fakefsl')

    def set_config(section, **settings):
        for key, value in settings.items():
            monkeypatch.setattr(getattr(config, section), key, value, raising=False)

    set_config('nipype', omp_nthreads=1)
    set_config('execution', sloppy=True, layout=_StubLayout())
    set_config(
        'workflow',
        hmc_method='eddy',
        sdc_method='topup',
        shoreline_model=None,
        b0_threshold=100,
        dwi_biascorrect='none',
        eddy_config=os.path.join(get_test_data_path(), 'eddy_config_trxscan.json'),
        no_b0_harmonization=False,
        denoise_method='none',
        unringing_method='none',
        dwidenoise_window=5,
        shoreline_iters=2,
        anatomical_template=TEMPLATE,
        anat_modality='T1w',
        dwi2anat_dof=6,
        b0_to_anat_transform='Rigid',
        hmc_transform='Affine',
        diffprep_config=None,
        tortoise_gpu_cpu_ratio=None,
        gpu=None,
        impute_slice_threshold=0,
        gradient_file=None,
        ignore=[],
        force=[],
        two_pass_sdc=True,
        output_resolution=3.0,
    )
    return set_config


def _rpe_unit(tmp_path):
    main = _write_dwi(tmp_path / 'sub-01_dir-AP_dwi.nii.gz')
    partner = _write_dwi(tmp_path / 'sub-01_dir-PA_dwi.nii.gz')
    return make_preproc_unit(
        [main, partner],
        method=CorrectionMethod.PEPOLAR,
        pe_dirs={main: 'j', partner: 'j-'},
    )


def _single_unit(tmp_path, **kwargs):
    return make_preproc_unit([_write_dwi(tmp_path / 'sub-01_dwi.nii.gz')], **kwargs)


def _boilerplate(unit, t2w_sdc=False):
    from qsiprep.workflows.dwi.base import init_dwi_preproc_wf

    wf = init_dwi_preproc_wf(
        unit,
        t2w_sdc=t2w_sdc,
        output_prefix='sub-01',
        source_file=unit.dwi_files[0],
        anatomical_template=TEMPLATE,
    )
    return wf.visit_desc()


def _assert_well_formed(text):
    """Check the typography every configuration must get right."""
    # A sentence run into the next one: "sequence.Both", "].The", "introduces.FSL".
    assert not re.search(r'[a-z\]][.][A-Z]', text), re.search(r'.{30}[a-z\]][.][A-Z].{30}', text)
    assert not re.search(r'[^\s]  +[^\s]', text), re.search(r'.{30}[^\s]  +[^\s].{30}', text)
    for wrong in ('eachseparate', 'Data was', 'version None', 'FSL workflows', '(@'):
        assert wrong not in text


def _before(text, first, second):
    assert first in text, first
    assert second in text, second
    return text.index(first) < text.index(second)


def test_rpe_topup(tmp_path, cfg):
    text = _boilerplate(_rpe_unit(tmp_path))
    _assert_well_formed(text)
    assert 'Images were grouped into two phase encoding polarity groups.' in text
    assert 'concatenated into a single series for head motion correction' in text
    assert "FSL's TOPUP" in text
    assert 'b=0 images extracted from multiple DWI series with reversed' in text
    assert 'the susceptibility distortion correction from the TOPUP field were' in text
    assert 'No susceptibility distortion correction was performed' not in text
    # eddy's output is not the final interpolation: QSIPrep resamples again.
    assert 'Final interpolation' not in text
    assert 'Eddy wrote out the corrected volumes' in text


def test_rpe_drbuddi_has_no_topup_stage(tmp_path, cfg):
    cfg('workflow', sdc_method='drbuddi')
    text = _boilerplate(_rpe_unit(tmp_path))
    _assert_well_formed(text)
    assert 'DRBUDDI [@drbuddi]' in text
    assert 'TOPUP' not in text
    assert 'initial susceptibility distortion correction' not in text
    assert 'The eddy current correction was Jacobian-modulated by eddy itself' in text


def test_epi_topup(tmp_path, cfg):
    dwi = _write_dwi(tmp_path / 'sub-01_dir-AP_dwi.nii.gz')
    epi = _write_dwi(tmp_path / 'sub-01_dir-PA_epi.nii.gz', nvols=2)
    unit = make_preproc_unit(
        [dwi],
        method=CorrectionMethod.PEPOLAR,
        estimation_sources=[dwi, epi],
        pe_dirs={dwi: 'j', epi: 'j-'},
    )
    text = _boilerplate(unit)
    _assert_well_formed(text)
    assert 'b=0 reference images with reversed phase encoding directions' in text
    assert 'Images were grouped into two phase encoding polarity groups' not in text


def test_phasediff_in_eddy(tmp_path, cfg):
    unit = _single_unit(
        tmp_path,
        method=CorrectionMethod.PHASEDIFF,
        estimation_sources=['/data/sub-01_phasediff.nii.gz', '/data/sub-01_magnitude1.nii.gz'],
        metadata={'EchoTime1': 0.004, 'EchoTime2': 0.006},
    )
    text = _boilerplate(unit)
    _assert_well_formed(text)
    assert 'was passed to eddy' in text
    # The field eddy applied is named, not TOPUP's.
    assert 'the susceptibility distortion correction from the GRE field map' in text
    assert 'TOPUP' not in text


def test_gnl_reports_gradwarp_after_eddy(tmp_path, cfg):
    from qsiprep.tests.gradient_fixtures import write_siemens_grad

    cfg('workflow', gradient_file=str(write_siemens_grad(tmp_path / 'coeff.grad')))
    main = _write_dwi(tmp_path / 'sub-01_dir-AP_dwi.nii.gz')
    partner = _write_dwi(tmp_path / 'sub-01_dir-PA_dwi.nii.gz')
    unit = make_preproc_unit(
        [main, partner],
        method=CorrectionMethod.PEPOLAR,
        pe_dirs={main: 'j', partner: 'j-'},
        metadata={'Manufacturer': 'SIEMENS'},
    )
    text = _boilerplate(unit)
    _assert_well_formed(text)
    assert _before(text, 'was used for head motion correction', 'Gradient nonlinearity was')
    assert 'gradient deviation map' in text
    assert 'TORTOISE V4 [@tortoisev4]' in text


def test_motion_states_the_missing_sdc(tmp_path, cfg):
    text = _boilerplate(_single_unit(tmp_path))
    _assert_well_formed(text)
    assert 'No susceptibility distortion correction was performed.' in text
    assert 'TOPUP' not in text


def test_diffprep_does_not_mention_fsl(tmp_path, cfg):
    cfg('workflow', hmc_method='tortoise', sdc_method='drbuddi')
    text = _boilerplate(_rpe_unit(tmp_path))
    _assert_well_formed(text)
    assert 'DIFFPREP' in text
    assert 'FSL' not in text
    assert 'eddy' not in text.lower().replace('eddy current', '')


def test_unpile_t2w(tmp_path, cfg):
    t2w = '/data/sub-01/anat/sub-01_T2w.nii.gz'
    unit = _single_unit(
        tmp_path,
        method=CorrectionMethod.UNPILE,
        estimation_sources=[t2w],
        anat_files=[t2w],
    )
    text = _boilerplate(unit, t2w_sdc=True)
    _assert_well_formed(text)
    assert 'unpile [@unpile], an experimental method' in text
    assert 'with local normalized cross-correlation as the cost function' in text
    assert 'No susceptibility distortion correction was performed' not in text


def test_synb0_generation_precedes_its_use(tmp_path, cfg):
    cfg('workflow', hmc_method='tortoise')
    t1w = '/data/sub-01/anat/sub-01_T1w.nii.gz'
    unit = _single_unit(
        tmp_path,
        method=CorrectionMethod.SYNB0,
        estimation_sources=[t1w],
        anat_files=[t1w],
    )
    text = _boilerplate(unit)
    _assert_well_formed(text)
    assert _before(text, 'SynB0-DISCO dual-channel U-Net', 'estimated with TOPUP')
    # Described once, though the second pass builds the generation workflow again.
    assert text.count('SynB0-DISCO dual-channel U-Net') == 1
    # Only the DWI's own coregistration is described, not the one SynB0 runs internally.
    assert text.count('The b=0 reference was registered to the anatomical reference') == 1


def test_synb0_in_eddy_describes_generation_where_topup_uses_it(tmp_path, cfg):
    t1w = '/data/sub-01/anat/sub-01_T1w.nii.gz'
    unit = _single_unit(
        tmp_path,
        method=CorrectionMethod.SYNB0,
        estimation_sources=[t1w],
        anat_files=[t1w],
    )
    text = _boilerplate(unit)
    _assert_well_formed(text)
    assert _before(text, 'SynB0-DISCO dual-channel U-Net', 'TOPUP was run again')
    assert _before(text, 'TOPUP was run again', 'Eddy wrote out the corrected volumes')
    assert text.count('SynB0-DISCO dual-channel U-Net') == 1


@pytest.mark.parametrize(
    ('sloppy', 'ignore', 'interpolation', 'jacobian'),
    [
        (True, [], 'using linear interpolation', 'applied in this step was Jacobian-modulated'),
        (False, ['jacobian'], 'Lanczos windowed-sinc', 'No Jacobian intensity modulation'),
    ],
)
def test_final_resampling(cfg, sloppy, ignore, interpolation, jacobian):
    from qsiprep.workflows.dwi.resampling import resampling_boilerplate

    cfg('execution', sloppy=sloppy)
    cfg('workflow', ignore=ignore)
    text = resampling_boilerplate('ACPC', 1.5)
    _assert_well_formed(text)
    assert '1.5mm isotropic voxels' in text
    assert interpolation in text
    assert jacobian in text
    assert 'gradient directions were rotated' in text


def test_gre_prior_is_described_once():
    from qsiprep.workflows.fieldmap.base import _gre_boilerplate
    from qsiprep.workflows.fieldmap.unpile import _boilerplate as unpile_boilerplate

    text = _gre_boilerplate(False, 'unpile') + unpile_boilerplate('t2w', False, True)
    assert text.count('prior') == 1
    text = _gre_boilerplate(False, 'unpile') + unpile_boilerplate('synb0', False, True)
    assert 'refined against the b=0' not in text
    assert 'applied as given' in text


def test_series_denoising_is_reported_under_the_diffusion_heading(tmp_path, cfg):
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow

    from qsiprep.workflows.base import (
        connect_series_to_unit,
        init_series_denoise_wfs,
        silence_series_descs,
    )
    from qsiprep.workflows.dwi.base import init_dwi_preproc_wf

    unit = _rpe_unit(tmp_path)
    subject_wf = Workflow(name='sub_01_wf')
    anat_wf = Workflow(name='anat_preproc_wf')
    anat_wf.__desc__ = '\n\n#### Anatomical data preprocessing\n\n'
    subject_wf.add_nodes([anat_wf])
    series_wfs = init_series_denoise_wfs([unit], 'LAS')
    dwi_preproc_wf = init_dwi_preproc_wf(
        unit,
        t2w_sdc=False,
        output_prefix='sub-01',
        source_file=unit.dwi_files[0],
        anatomical_template=TEMPLATE,
    )
    connect_series_to_unit(subject_wf, series_wfs, unit, dwi_preproc_wf, 'sub_01')
    silence_series_descs(series_wfs)

    text = subject_wf.visit_desc()
    # Both series share their (empty) denoising, which is stated once.
    assert text.count('No denoising steps were applied to the DWI data.') == 1
    assert _before(text, '#### Diffusion data preprocessing', 'No denoising steps were applied')
    assert _before(text, 'No denoising steps were applied', 'Images were grouped')


def _eddy_text(**eddy_config):
    from qsiprep.interfaces.eddy import boilerplate_from_eddy_config

    return boilerplate_from_eddy_config({'method': 'jac', **eddy_config}, '', pepolar_method='')


def test_eddy_version_comes_from_the_conda_package(tmp_path, monkeypatch):
    meta = tmp_path / 'conda-meta'
    meta.mkdir()
    (meta / 'fsl-eddy-2602.0-h2b5b6c4_0.json').write_text(
        json.dumps({'name': 'fsl-eddy', 'version': '2602.0'})
    )
    # A package whose name merely starts with fsl-eddy- is not eddy itself.
    (meta / 'fsl-eddy-cuda-2602.0-h0_0.json').write_text(
        json.dumps({'name': 'fsl-eddy-cuda', 'version': '9999'})
    )
    monkeypatch.setenv('FSLDIR', str(tmp_path))
    assert "FSL's eddy (version 2602.0)" in _eddy_text()


def test_eddy_version_is_left_out_when_unknown(tmp_path, monkeypatch):
    from nipype.interfaces.fsl.base import Info

    monkeypatch.setenv('FSLDIR', str(tmp_path))
    monkeypatch.delenv('CONDA_PREFIX', raising=False)
    monkeypatch.setattr('sys.prefix', str(tmp_path))
    monkeypatch.setattr(Info, '_version', None)
    text = _eddy_text()
    assert "FSL's eddy was used" in text
    assert 'None' not in text


def test_eddy_outlier_voxels_and_multiband_offset():
    text = _eddy_text(
        repol=True,
        outlier_nvox=300,
        outlier_nstd=5,
        outlier_type='both',
        multiband_factor=3,
        multiband_offset=-1,
    )
    assert 'at least 300 intracerebral voxels' in text
    assert 'more than 5 standard deviations' in text
    assert (
        'A multi-band acceleration factor of 3 and slices removed from the bottom of the '
        'volume were assumed.'
    ) in text
