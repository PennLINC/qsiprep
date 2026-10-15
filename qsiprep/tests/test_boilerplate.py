"""Methods boilerplate text: what the generators state for a given configuration."""

import json
import os
import stat

import pytest

from qsiprep import config
from qsiprep.tests.gradient_fixtures import write_dwi_with_gradients
from qsiprep.tests.preproc_factory import make_preproc_unit
from qsiprep.utils import versions

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
    _clear_version_caches()
    yield
    _clear_version_caches()


def _clear_version_caches():
    for lookup in (
        versions.eddy_version,
        versions.tortoise_version,
        versions.niimath_version,
        versions.svht_denoise_version,
    ):
        lookup.cache_clear()


def _fake_tool(bin_dir, name, output):
    """Write an executable that prints ``output``, as the real tool's --version does."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    tool = bin_dir / name
    tool.write_text(f"#!/bin/sh\ncat <<'EOF'\n{output}\nEOF\n")
    tool.chmod(tool.stat().st_mode | stat.S_IEXEC)
    return tool


@pytest.fixture
def fake_tools(tmp_path, monkeypatch):
    """Put the container's tools on PATH, with the version output captured from the image."""
    bin_dir = tmp_path / 'bin'
    _fake_tool(bin_dir, 'niimath', 'v1.0.20260924 OpenMP GCC11.4.0 BSD (64-bit Linux)')
    _fake_tool(bin_dir, 'svht_denoise', 'svht_denoise 0.1.20261001')
    tortoise = tmp_path / 'TORTOISEV4'
    _fake_tool(tortoise / 'bin', 'TORTOISEProcess', '')
    (tortoise / 'settings').mkdir()
    (tortoise / 'settings' / 'version.txt').write_text('TORTOISE V4.2.0\n')
    monkeypatch.setenv(
        'PATH', os.pathsep.join([str(bin_dir), str(tortoise / 'bin'), os.environ['PATH']])
    )
    monkeypatch.setenv('MRTRIX3_DEV_VERSION', '3.0.8-2071-gb98b54e9')
    monkeypatch.setattr(config.environment, 'mrtrix3_version', '3.0.4')


def _eddy_text(eddy_args=None, fieldmap_type='', pepolar_method='', **kwargs):
    from qsiprep.interfaces.eddy import boilerplate_from_eddy_config

    return ' '.join(
        boilerplate_from_eddy_config(
            {**EDDY_ARGS, **(eddy_args or {})}, fieldmap_type, pepolar_method, **kwargs
        ).split()
    )


def test_eddy_version_comes_from_the_conda_records(tmp_path, monkeypatch):
    """A pixi FSL has no etc/fslversion; eddy's package record has its version."""
    from qsiprep.utils.versions import eddy_version

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
    assert "FSL's eddy 2602.0 was used" in _eddy_text()


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


def test_tool_versions_are_read_from_the_tools(fake_tools):
    assert versions.niimath_version() == 'v1.0.20260924'
    assert versions.svht_denoise_version() == '0.1.20261001'
    assert versions.tortoise_version() == 'V4.2.0'
    assert versions.mrtrix3_version() == '3.0.4'
    assert versions.mrtrix3_version(dev=True) == '3.0.8-2071-gb98b54e9'


def test_missing_tools_have_no_version(tmp_path, monkeypatch):
    monkeypatch.setenv('PATH', str(tmp_path))
    assert versions.niimath_version() is None
    assert versions.svht_denoise_version() is None
    assert versions.tortoise_version() is None
    assert versions.python_package_version('not-a-real-package') is None
    assert versions.versioned('niimath', None) == 'niimath'


def test_tortoise_texts_name_the_version(fake_tools):
    from qsiprep.interfaces.tortoise import (
        generate_diffprep_boilerplate,
        generate_drbuddi_boilerplate,
    )
    from qsiprep.workflows.dwi.gradwarp import gradwarp_boilerplate

    assert 'the TORTOISE V4.2.0 [@tortoisev4] software' in generate_diffprep_boilerplate(
        'quadratic'
    )
    assert 'part of the TORTOISE V4.2.0 [@tortoisev4]' in generate_drbuddi_boilerplate(
        'rpe_series', t2w_sdc=False
    )
    assert 'with TORTOISE V4.2.0' in gradwarp_boilerplate(None)
    assert 'SelectBestB0 from TORTOISE V4.2.0' in _eddy_text(
        fieldmap_type='rpe_series', pepolar_method='topup'
    )


def test_gradwarp_text_falls_back_to_v4(tmp_path, monkeypatch):
    """The gradient nonlinearity tools exist only in TORTOISE V4."""
    from qsiprep.workflows.dwi.gradwarp import gradwarp_boilerplate

    monkeypatch.setenv('PATH', str(tmp_path))
    assert 'with TORTOISE V4 to' in gradwarp_boilerplate(None)


@pytest.mark.parametrize(
    ('denoise_method', 'unringing_method', 'expected'),
    [
        ('dwidenoise', 'none', 'implemented in dwidenoise from MRtrix3 3.0.4 [@mrtrix3'),
        ('none', 'mrdegibbs', 'using MRtrix3 3.0.4 [@mrtrix3; @mrdegibbs]'),
        ('none', 'rpg', 'removed using TORTOISE V4.2.0 [@pfgibbs]'),
        ('none', 'svht', 'as implemented in `svht_denoise` 0.1.20261001 [@svht_denoise]'),
        ('svht', 'none', 'as implemented in `svht_denoise` 0.1.20261001 [@svht_denoise]'),
    ],
)
def test_denoising_text_names_the_versions(fake_tools, denoise_method, unringing_method, expected):
    from qsiprep.workflows.dwi.merge import init_dwi_denoising_wf

    config.workflow.denoise_method = denoise_method
    config.workflow.unringing_method = unringing_method
    config.workflow.denoise_config = None
    config.workflow.dwidenoise_window = 5
    config.workflow.mrtrix_version = 'stable'
    wf = init_dwi_denoising_wf(
        source_file='/data/sub-01_dwi.nii.gz',
        partial_fourier=1.0,
        phase_encoding_direction='j',
        n_volumes=30,
        use_phase=False,
    )
    assert expected in ' '.join(wf.__desc__.split())


def test_bias_correction_text_names_the_mrtrix3_version(fake_tools):
    from qsiprep.workflows.dwi.merge import gen_intensity_boilerplate

    assert '`dwibiascorrect` from MRtrix3 3.0.4 with the N4' in gen_intensity_boilerplate(True)
