"""Methods boilerplate text: what the generators state for a given configuration."""

import json

import pytest

from qsiprep import config
from qsiprep.tests.gradient_fixtures import write_dwi_with_gradients
from qsiprep.tests.preproc_factory import make_preproc_unit

EDDY_ARGS = {'method': 'jac', 'flm': 'quadratic', 'slm': 'linear', 'repol': True}


@pytest.fixture(autouse=True)
def _cfg(monkeypatch):
    config.nipype.omp_nthreads = 1
    config.execution.sloppy = False
    config.workflow.b0_threshold = 100
    config.workflow.no_b0_harmonization = False
    config.workflow.eddy_config = None
    config.workflow.anatomical_template = 'MNI152NLin2009cAsym'
    config.workflow.ignore = []
    config.workflow.force = []
    config.workflow.gradient_file = None
    monkeypatch.delenv('FSLDIR', raising=False)


def _eddy_text(eddy_args=None, fieldmap_type='', pepolar_method='', **kwargs):
    from qsiprep.interfaces.eddy import boilerplate_from_eddy_config

    return ' '.join(
        boilerplate_from_eddy_config(
            {**EDDY_ARGS, **(eddy_args or {})}, fieldmap_type, pepolar_method, **kwargs
        ).split()
    )


def test_eddy_version_comes_from_the_conda_records(tmp_path, monkeypatch):
    """A pixi FSL has no etc/fslversion; eddy's package record has its version."""
    from qsiprep.interfaces.eddy import eddy_version

    meta = tmp_path / 'conda-meta'
    meta.mkdir()
    for name, version in (
        ('fsl-eddy_qc-v1.4.2-py_0', 'v1.4.2'),
        ('fsl-eddy-cuda-2603.1-h1_0', '2603.1'),
        ('fsl-eddy-2602.0-hb35a746_0', '2602.0'),
    ):
        (meta / f'{name}.json').write_text(json.dumps({'version': version}))
    monkeypatch.setenv('FSLDIR', str(tmp_path))

    assert eddy_version() == '2602.0'
    assert "FSL's eddy (version 2602.0) was used" in _eddy_text()


def test_eddy_text_leaves_out_an_unknown_version():
    text = _eddy_text()
    assert 'None' not in text
    assert "FSL's eddy was used for head motion correction" in text


def test_eddy_text_reports_the_outlier_voxel_count_not_the_threshold():
    text = _eddy_text({'outlier_nvox': 500, 'outlier_nstd': 3})
    assert 'at least 500 intracerebral voxels' in text
    assert 'more than 3 standard deviations' in text


@pytest.mark.parametrize(('offset', 'end'), [(1, 'top'), (-1, 'bottom')])
def test_eddy_text_names_the_multiband_offset(offset, end):
    text = _eddy_text({'multiband_factor': 3, 'multiband_offset': offset, 'outlier_type': 'both'})
    assert (
        f'A multi-band acceleration factor of 3 and slices removed from the {end} of the '
        'volume were assumed.'
    ) in text


def test_eddy_text_without_multiband_offset():
    text = _eddy_text({'multiband_factor': 3, 'outlier_type': 'gw'})
    assert 'A multi-band acceleration factor of 3 was assumed.' in text


def test_eddy_then_drbuddi_without_topup_claims_no_initial_correction():
    text = _eddy_text(fieldmap_type='rpe_series', pepolar_method='drbuddi')
    assert 'initial susceptibility' not in text
    assert 'Interpolation after head motion correction was performed' in text
    assert 'Eddy-current correction was Jacobian-modulated' in text


def test_topup_then_drbuddi_names_the_initial_correction():
    text = _eddy_text(fieldmap_type='rpe_series', pepolar_method='topup+drbuddi')
    assert (
        'Interpolation after head motion and initial susceptibility distortion correction'
    ) in text
    assert 'Eddy-current and susceptibility distortion correction were Jacobian' in text


@pytest.mark.parametrize(
    ('pepolar_method', 'gre_in_eddy', 'expected'),
    [
        ('', False, 'Eddy-current correction was Jacobian-modulated'),
        ('topup', False, 'Eddy-current and susceptibility distortion correction were Jacobian'),
        ('', True, 'Eddy-current and susceptibility distortion correction were Jacobian'),
    ],
)
def test_eddy_jacobian_sentence_names_what_eddy_corrected(pepolar_method, gre_in_eddy, expected):
    """A GRE field passed to eddy is modulated by eddy's resampling as TOPUP's is."""
    text = _eddy_text(
        fieldmap_type='rpe_series' if pepolar_method else '',
        pepolar_method=pepolar_method,
        gre_in_eddy=gre_in_eddy,
    )
    assert expected in text
    assert 'when TOPUP was used' not in text


def test_intensity_text_without_harmonization_or_bias_correction_keeps_denoising():
    """Turning off harmonization and N4 does not mean nothing was denoised."""
    from qsiprep.workflows.dwi.merge import gen_intensity_boilerplate

    config.workflow.no_b0_harmonization = True
    text = gen_intensity_boilerplate(do_biascorr=False)
    assert 'denois' not in text
    assert text == 'Any images with a b-value less than 100 s/mm^2 were treated as a *b*=0 image.'


def test_bidirectional_pre_hmc_text_separates_its_sentences(tmp_path):
    from qsiplan.models import CorrectionMethod

    from qsiprep.workflows.dwi.pre_hmc import init_dwi_pre_hmc_wf

    config.workflow.denoise_method = 'none'
    config.workflow.unringing_method = 'none'
    main = write_dwi_with_gradients(tmp_path / 'sub-01_dir-AP_dwi.nii.gz')
    partner = write_dwi_with_gradients(tmp_path / 'sub-01_dir-PA_dwi.nii.gz')
    unit = make_preproc_unit(
        [main, partner], method=CorrectionMethod.PEPOLAR, pe_dirs={main: 'j', partner: 'j-'}
    )
    wf = init_dwi_pre_hmc_wf(unit, orientation='LPS', source_file=main, do_biascorr=True)

    text = wf.visit_desc()
    assert 'were resampled. Both phase encoding polarity groups were then merged' in text
    # The merge runs for every head motion method, not only eddy.
    assert 'FSL' not in text
    assert 'eachseparate' not in text


def test_no_denoising_text_ends_its_sentence():
    from qsiprep.workflows.dwi.merge import init_dwi_denoising_wf

    config.workflow.denoise_method = 'none'
    config.workflow.unringing_method = 'none'
    wf = init_dwi_denoising_wf(
        source_file='/data/sub-01_dwi.nii.gz',
        partial_fourier=1.0,
        phase_encoding_direction='j',
        n_volumes=10,
        use_phase=False,
    )
    assert wf.__desc__.endswith('No denoising steps were applied to the DWI data. ')


def test_synb0_text_does_not_repeat_the_coregistration_paragraph():
    from qsiprep.workflows.fieldmap.synb0 import init_synb0_wf

    text = init_synb0_wf().visit_desc()
    assert 'global search over rotations' not in text
    assert 'the distorted\nb=0 reference was rigidly registered to the T1w image' in text


def test_diffprep_without_a_fieldmap_says_no_sdc(tmp_path):
    from qsiprep.workflows.dwi.diffprep import init_diffprep_hmc_wf
    from qsiprep.workflows.fieldmap.base import NO_SDC_DESC

    config.workflow.hmc_method = 'tortoise'
    config.workflow.sdc_method = 'drbuddi'
    config.workflow.diffprep_config = None
    config.workflow.gpu = None
    dwi = write_dwi_with_gradients(tmp_path / 'sub-01_dwi.nii.gz')
    wf = init_diffprep_hmc_wf(
        make_preproc_unit([dwi]), source_file='/data/x_dwi.nii.gz', t2w_sdc=False
    )
    assert NO_SDC_DESC.strip() in wf.visit_desc()
