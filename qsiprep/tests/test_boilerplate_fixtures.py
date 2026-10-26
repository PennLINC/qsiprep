"""Methods boilerplate for the configurations the TRXScan integration fixtures run.

Each test builds the DWI preprocessing workflow the way one ``trxscan_*`` run in
``test_cli.py`` does and checks the methods text it would write: what it says,
the order it says it in, and its typography. ``test_boilerplate.py`` covers the
individual text builders.
"""

import os
import re

import pytest
from qsiplan.models import CorrectionMethod

from qsiprep import config
from qsiprep.tests.preproc_factory import make_preproc_unit
from qsiprep.tests.test_workflows_native import _StubLayout, _write_dwi
from qsiprep.tests.utils import get_test_data_path
from qsiprep.utils.spaces import SpaceSpec

TEMPLATE = 'MNI152NLin2009cAsym'
EDDY_TEXT = 'Head motion and eddy-current distortions were corrected with'
SYNB0_GENERATION = 'SynB0-DISCO dual-channel U-Net'
COREG = 'The *b*=0 reference was coregistered to the anatomical reference'


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
        output_spaces=['acpc:res-3mm', TEMPLATE],
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
    )
    return set_config


def _rpe_unit(tmp_path, **kwargs):
    main = _write_dwi(tmp_path / 'sub-01_dir-AP_dwi.nii.gz')
    partner = _write_dwi(tmp_path / 'sub-01_dir-PA_dwi.nii.gz')
    return make_preproc_unit(
        [main, partner],
        method=CorrectionMethod.PEPOLAR,
        pe_dirs={main: 'j', partner: 'j-'},
        **kwargs,
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
        acpc_anchor=SpaceSpec(space=TEMPLATE),
    )
    text = wf.visit_desc()
    _assert_well_formed(text)
    return text


def _assert_well_formed(text):
    """Check the typography every configuration must get right."""
    # A sentence run into the next one: "sequence.Both", "].The", "introduces.FSL".
    run_on = re.search(r'.{0,30}[a-z\]][.][A-Z].{0,30}', text)
    assert run_on is None, run_on.group()
    double = re.search(r'.{0,30}[^\s]  +[^\s].{0,30}', text)
    assert double is None, double.group()
    for wrong in ('eachseparate', 'Data was', 'version None', 'FSL workflows', '(@'):
        assert wrong not in text
    # The text is pasted into a paper's methods section, where nobody ran qsiprep: it
    # describes what was done to the data, not who asked for it or which flag did it.
    operator = re.search(r'.{0,30}(\buser|\brequest|(?<![\w-])--[a-z]).{0,30}', text, re.I)
    assert operator is None, operator.group()


def _before(text, first, second):
    assert first in text, first
    assert second in text, second
    return text.index(first) < text.index(second)


def test_rpe_topup(tmp_path, cfg):
    text = _boilerplate(_rpe_unit(tmp_path))
    assert 'The series were grouped by phase-encoding polarity.' in text
    assert 'The two polarity groups were then concatenated into a single series.' in text
    assert 'of DWI series acquired with opposite phase-encoding polarities' in text
    assert 'Jacobian-modulates the eddy-current and susceptibility distortion corrections' in text
    assert 'No susceptibility distortion correction was performed' not in text
    # eddy's output is not the final interpolation: QSIPrep resamples again.
    assert 'Final interpolation' not in text


def test_rpe_drbuddi_has_no_topup_stage(tmp_path, cfg):
    cfg('workflow', sdc_method='drbuddi')
    text = _boilerplate(_rpe_unit(tmp_path))
    assert 'DRBUDDI' in text
    assert 'TOPUP' not in text
    assert 'initial susceptibility distortion correction' not in text
    assert 'Jacobian-modulates the eddy-current correction.' in text


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
    assert '*b*=0 images of the DWI series and of the field map' in text
    assert 'grouped by phase-encoding polarity' not in text


def test_phasediff_in_eddy(tmp_path, cfg):
    unit = _single_unit(
        tmp_path,
        method=CorrectionMethod.PHASEDIFF,
        estimation_sources=['/data/sub-01_phasediff.nii.gz', '/data/sub-01_magnitude1.nii.gz'],
        metadata={'EchoTime1': 0.004, 'EchoTime2': 0.006},
    )
    text = _boilerplate(unit)
    assert 'was passed to `eddy`' in text
    # The GRE field eddy applied is modulated with the eddy-current correction.
    assert 'Jacobian-modulates the eddy-current and susceptibility distortion corrections' in text
    assert 'TOPUP' not in text


def test_gnl_reports_gradwarp_after_eddy(tmp_path, cfg):
    from qsiprep.tests.gradient_fixtures import write_siemens_grad

    cfg('workflow', gradient_file=str(write_siemens_grad(tmp_path / 'coeff.grad')))
    text = _boilerplate(_rpe_unit(tmp_path, metadata={'Manufacturer': 'SIEMENS'}))
    assert _before(text, EDDY_TEXT, 'Gradient nonlinearity was corrected')
    assert 'gradient deviation map' in text
    assert '[@tortoisev4] from the scanner gradient coefficients' in text


def test_motion_states_the_missing_sdc(tmp_path, cfg):
    text = _boilerplate(_single_unit(tmp_path))
    assert 'No susceptibility distortion correction was performed.' in text
    assert 'TOPUP' not in text


def test_diffprep_does_not_mention_fsl(tmp_path, cfg):
    cfg('workflow', hmc_method='tortoise', sdc_method='drbuddi')
    text = _boilerplate(_rpe_unit(tmp_path))
    assert 'DIFFPREP' in text
    assert 'FSL' not in text
    assert '`eddy`' not in text


def test_unpile_t2w(tmp_path, cfg):
    t2w = '/data/sub-01/anat/sub-01_T2w.nii.gz'
    unit = _single_unit(
        tmp_path,
        method=CorrectionMethod.UNPILE,
        estimation_sources=[t2w],
        anat_files=[t2w],
    )
    text = _boilerplate(unit, t2w_sdc=True)
    assert 'an experimental method' in text
    assert 'local normalized cross-correlation' in text
    assert 'No susceptibility distortion correction was performed' not in text


@pytest.mark.parametrize('hmc_method', ['tortoise', 'eddy'])
def test_synb0_describes_generation_and_coregistration_once(tmp_path, cfg, hmc_method):
    cfg('workflow', hmc_method=hmc_method)
    t1w = '/data/sub-01/anat/sub-01_T1w.nii.gz'
    unit = _single_unit(
        tmp_path,
        method=CorrectionMethod.SYNB0,
        estimation_sources=[t1w],
        anat_files=[t1w],
    )
    text = _boilerplate(unit)
    # Described once, though the second pass builds the generation workflow again.
    assert text.count(SYNB0_GENERATION) == 1
    assert 'TOPUP was run again' in text or 'TOPUP was run again'.lower() in text.lower()
    # Only the DWI's own coregistration is described, not the one SynB0 runs internally.
    assert text.count(COREG) == 1


@pytest.mark.parametrize(
    ('sloppy', 'ignore', 'interpolation', 'jacobian'),
    [
        (True, [], 'with linear interpolation.', 'applied in this step was Jacobian-modulated'),
        (False, ['jacobian'], 'or linear interpolation where', 'No Jacobian modulation'),
    ],
)
def test_final_resampling(cfg, sloppy, ignore, interpolation, jacobian):
    from qsiprep.utils.spaces import parse_output_spaces
    from qsiprep.workflows.dwi.resampling import init_dwi_trans_wf

    cfg('execution', sloppy=sloppy)
    cfg('workflow', ignore=ignore)
    (spec,) = parse_output_spaces(['acpc:res-3mm'])
    text = init_dwi_trans_wf(
        source_file='sub-01_dwi.nii.gz',
        mem_gb=1,
        resolution=spec.resolution,
        template='ACPC',
    ).__desc__
    _assert_well_formed(text)
    assert interpolation in text
    # ChooseInterpolator switches to Linear, not B-spline, when upsampling.
    assert 'B-spline' not in text
    assert jacobian in text


@pytest.mark.parametrize(('target', 'expected'), [('t2w', 'refined'), ('synb0', 'as acquired')])
def test_gre_prior_is_described_once(target, expected):
    from qsiprep.workflows.fieldmap.base import _gre_boilerplate
    from qsiprep.workflows.fieldmap.unpile import _boilerplate as unpile_boilerplate

    text = _gre_boilerplate(False, 'unpile') + unpile_boilerplate(target, False, True)
    assert text.count('given to *unpile*') == 1
    assert expected in text
    if target == 'synb0':
        # The GRE text must not contradict unpile's "applied as acquired".
        assert 'refined against' not in text


def test_template_is_cited(cfg):
    from qsiprep.workflows.anatomical.volume import init_anat_normalization_wf

    text = init_anat_normalization_wf(SpaceSpec(space=TEMPLATE)).__desc__
    assert f'{TEMPLATE} template [@mni].' in text
