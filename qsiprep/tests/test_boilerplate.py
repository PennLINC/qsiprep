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
    assert '`eddy` from *FSL* (version 2602.0) [@anderssoneddy]' in _eddy_text()


def test_eddy_text_leaves_out_an_unknown_version():
    text = _eddy_text()
    assert 'None' not in text
    assert 'corrected with `eddy` from *FSL* [@anderssoneddy]' in text


def test_eddy_text_does_not_call_its_resampling_final():
    """The eddy output is resampled again into AC-PC space."""
    text = _eddy_text(fieldmap_type='rpe_series', pepolar_method='topup')
    assert 'final' not in text.lower()
    assert '`eddy` resampled the data with its `jac` method' in text


def test_eddy_text_reports_the_outlier_voxel_count_not_the_threshold():
    text = _eddy_text({'outlier_nvox': 500, 'outlier_nstd': 3})
    assert 'at least 500 brain voxels' in text
    assert 'more than 3 standard deviations' in text


@pytest.mark.parametrize(('offset', 'end'), [(1, 'top'), (-1, 'bottom')])
def test_eddy_text_names_the_multiband_offset(offset, end):
    text = _eddy_text({'multiband_factor': 3, 'multiband_offset': offset, 'outlier_type': 'both'})
    assert (
        f'A multiband factor of 3 was assumed, with slices removed from the {end} of the volume.'
    ) in text


def test_eddy_text_without_multiband_offset():
    text = _eddy_text({'multiband_factor': 3, 'outlier_type': 'gw'})
    assert 'A multiband factor of 3 was assumed.' in text


def test_eddy_then_drbuddi_without_topup_claims_no_susceptibility_correction():
    text = _eddy_text(fieldmap_type='rpe_series', pepolar_method='drbuddi')
    assert 'susceptibility distortion correction' not in text
    assert 'TOPUP' not in text
    assert 'which Jacobian-modulates the eddy-current correction.' in text


def test_topup_then_drbuddi_names_the_first_stage():
    text = _eddy_text(fieldmap_type='rpe_series', pepolar_method='topup+drbuddi')
    assert 'In the first of two distortion-correction stages' in text
    assert 'eddy-current and susceptibility distortion corrections.' in text


@pytest.mark.parametrize(
    ('pepolar_method', 'gre_in_eddy', 'expected'),
    [
        ('', False, 'Jacobian-modulates the eddy-current correction.'),
        ('topup', False, 'Jacobian-modulates the eddy-current and susceptibility distortion'),
        ('', True, 'Jacobian-modulates the eddy-current and susceptibility distortion'),
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


@pytest.mark.parametrize('harmonize', [True, False])
def test_intensity_text_describes_harmonization_only_across_series(harmonize):
    """Harmonizing a single series across series says nothing; it is left out."""
    from qsiprep.workflows.dwi.merge import gen_intensity_boilerplate

    config.workflow.no_b0_harmonization = not harmonize
    threshold = 'Volumes with *b* < 100 s/mm² were treated as *b*=0 images.'
    assert gen_intensity_boilerplate(1) == threshold
    two = gen_intensity_boilerplate(2)
    assert two.startswith(threshold)
    assert ('matched across the series' in two) is harmonize
    assert 'denois' not in two


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
    wf = init_dwi_pre_hmc_wf(unit, orientation='LPS', source_file=main)

    text = wf.visit_desc()
    assert 'across the series. The two polarity groups were then concatenated' in text
    # The merge runs for every head motion method, not only eddy.
    assert 'FSL' not in text
    # Bias correction runs after resampling and is described there.
    assert 'dwibiascorrect' not in text


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
    assert wf.__desc__.endswith('Gibbs-ringing removal was applied to the DWI data. ')


def test_synb0_text_does_not_repeat_the_coregistration_paragraph():
    from qsiprep.workflows.fieldmap.synb0 import init_synb0_wf

    text = init_synb0_wf().visit_desc()
    assert 'global search over rotations' not in text
    assert 'the distorted\n*b*=0 reference was rigidly registered to the T1w image' in text


@pytest.mark.parametrize(
    ('sloppy', 'expected'),
    [(False, 'windowed-sinc (Lanczos) interpolation'), (True, 'with linear interpolation')],
)
def test_resampling_text_names_interpolation_and_bvec_rotation(sloppy, expected):
    from qsiprep.workflows.dwi.resampling import init_dwi_trans_wf

    config.execution.sloppy = sloppy
    config.workflow.output_resolution = 2.0
    wf = init_dwi_trans_wf(source_file='/data/sub-01_dwi.nii.gz', template='ACPC', mem_gb=1)
    text = ' '.join(wf.__desc__.split())
    assert 'resampled into AC-PC space at 2 mm isotropic resolution' in text
    assert expected in text
    assert 'The *b*-vectors were reoriented' in text


@pytest.mark.parametrize('do_biascorr', [True, False])
def test_bias_correction_is_described_after_resampling(fake_tools, do_biascorr):
    from qsiprep.workflows.dwi.finalize import init_finalize_denoising_wf

    wf = init_finalize_denoising_wf(
        source_file='/data/sub-01_dwi.nii.gz', do_biascorr=do_biascorr, num_dwi_acquisitions=1
    )
    text = wf.visit_desc()
    assert ('After resampling, B1 field inhomogeneity was corrected' in text) is do_biascorr
    if do_biascorr:
        assert '`dwibiascorrect` from *MRtrix3* 3.0.4 [@mrtrix3]' in text


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

    assert 'DIFFPREP from *TORTOISE* V4.2.0 [@tortoisev4]' in generate_diffprep_boilerplate(
        'quadratic'
    )
    assert (
        'DRBUDDI [@drbuddi] from *TORTOISE* V4.2.0 [@tortoisev4]'
        in generate_drbuddi_boilerplate('rpe_series', t2w_sdc=False)
    )
    assert 'with *TORTOISE* V4.2.0' in gradwarp_boilerplate(None)
    assert '`SelectBestB0` from *TORTOISE* V4.2.0' in _eddy_text(
        fieldmap_type='rpe_series', pepolar_method='topup'
    )


def test_gradwarp_text_falls_back_to_v4(tmp_path, monkeypatch):
    """The gradient nonlinearity tools exist only in TORTOISE V4."""
    from qsiprep.workflows.dwi.gradwarp import gradwarp_boilerplate

    monkeypatch.setenv('PATH', str(tmp_path))
    assert 'with *TORTOISE* V4 [@tortoisev4] to' in gradwarp_boilerplate(None)


@pytest.mark.parametrize(
    ('denoise_method', 'unringing_method', 'expected'),
    [
        ('dwidenoise', 'none', 'implemented in `dwidenoise` from *MRtrix3* 3.0.4 [@mrtrix3]'),
        ('none', 'mrdegibbs', '`mrdegibbs` from *MRtrix3* 3.0.4 [@mrtrix3; @mrdegibbs]'),
        ('none', 'rpg', 'RPG method [@pfgibbs] as implemented in *TORTOISE* V4.2.0'),
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


def test_option_text_names_no_operator():
    """Forced and ignored options are described as what was done, not who asked."""
    from qsiprep.workflows.dwi.gradwarp import gradwarp_boilerplate

    config.workflow.ignore = ['jacobian']
    for text in (gradwarp_boilerplate('1D', 'forced'), gradwarp_boilerplate('3D')):
        assert 'request' not in text
        assert '--' not in text
    assert 'without Jacobian intensity modulation' in gradwarp_boilerplate('3D')
