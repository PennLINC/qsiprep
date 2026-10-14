r"""Unit tests for ``qsiprep.utils.jacobian_provenance``.

``jacobian_provenance_for`` replaces the old ``config.record_applied``/
``config.record_unmodulated`` invocation-global accumulation (see git history)
with a pure, per-unit computation. These tests build synthetic
:class:`~qsiplan.adapters.PreprocUnit`\ s with
:func:`qsiprep.tests.preproc_factory.make_preproc_unit` -- no BIDS layout, no
disk files, no workflow construction -- and call the function directly across
the configuration matrix that motivated the migration.

``qsiprep/tests/test_workflows_jacobian_markers.py`` additionally verifies
this function agrees with the real workflow builders (``init_fsl_hmc_wf``,
``init_qsiprep_hmcsdc_wf``, ``init_diffprep_hmc_wf``) for every CI marker; this
module is the direct, exhaustive unit-level coverage of the function itself.
"""

import json

import pytest
from qsiplan.models import CorrectionMethod

from qsiprep import config
from qsiprep.tests.preproc_factory import make_preproc_unit
from qsiprep.utils.jacobian_provenance import jacobian_provenance_for

SRC = '/data/sub-01_dwi.nii.gz'
PARTNER = '/data/sub-01_dir-PA_dwi.nii.gz'


@pytest.fixture(autouse=True)
def _reset_config():
    saved = {
        name: getattr(config.workflow, name)
        for name in (
            'hmc_method',
            'sdc_method',
            'shoreline_model',
            'eddy_config',
            'diffprep_config',
            'gradient_file',
            'ignore',
            'force',
        )
    }
    saved_sloppy = config.execution.sloppy
    yield
    for name, value in saved.items():
        setattr(config.workflow, name, value)
    config.execution.sloppy = saved_sloppy


def _cfg(hmc_method, sdc_method='auto', sloppy=False):
    config.workflow.hmc_method = hmc_method
    config.workflow.sdc_method = sdc_method
    config.workflow.shoreline_model = '3dshore' if hmc_method == 'shoreline' else None
    config.workflow.eddy_config = None
    config.workflow.diffprep_config = None
    config.workflow.gradient_file = None
    config.workflow.ignore = []
    config.workflow.force = []
    config.execution.sloppy = sloppy


def _pepolar_unit(method):
    return make_preproc_unit([SRC, PARTNER], method=method, pe_dirs={SRC: 'j', PARTNER: 'j-'})


#: A full ``--eddy-config`` override with ``method='lsr'`` -- see
#: ``qsiprep/tests/data/eddy_params.json`` for the real shipped default this
#: mirrors (only ``method`` matters for these tests).
_LSR_EDDY_ARGS = {
    'flm': 'quadratic',
    'slm': 'linear',
    'fep': False,
    'interp': 'spline',
    'nvoxhp': 1000,
    'fudge_factor': 10,
    'dont_sep_offs_move': False,
    'dont_peas': False,
    'niter': 5,
    'method': 'lsr',
    'repol': True,
    'num_threads': 1,
    'is_shelled': True,
    'use_cuda': False,
    'cnr_maps': True,
    'residuals': False,
    'output_type': 'NIFTI_GZ',
    'args': '',
}


def test_gradwarp_only(tmp_path):
    """Test the eddy backend with no fieldmap and a gradwarp coefficient file with no DIS3D tag."""
    _cfg(hmc_method='eddy', sdc_method='auto')
    config.workflow.gradient_file = str(tmp_path / 'coeff.grad')
    unit = make_preproc_unit([SRC], method=None, metadata={'Manufacturer': 'SIEMENS'})

    assert jacobian_provenance_for(unit) == (['gradwarp'], [], None)


def test_drbuddi():
    _cfg(hmc_method='eddy', sdc_method='drbuddi')
    unit = _pepolar_unit(CorrectionMethod.PEPOLAR)

    assert jacobian_provenance_for(unit) == (['sdc'], [], None)


def _gre_unit():
    return make_preproc_unit(
        [SRC],
        method=CorrectionMethod.PHASEDIFF,
        estimation_sources=['/data/sub-01_phasediff.nii.gz', '/data/sub-01_magnitude1.nii.gz'],
        metadata={'EchoTime1': 0.004, 'EchoTime2': 0.006},
    )


def test_gre():
    """Test that eddy applies a GRE fieldmap itself, like TOPUP's field: nothing external."""
    _cfg(hmc_method='eddy', sdc_method='fieldmap')

    assert jacobian_provenance_for(_gre_unit()) == ([], [], None)


def test_gre_after_eddy():
    """Test that --force gre-sdc-after-eddy applies the GRE warp downstream.

    That is where QSIPrep modulates it.
    """
    _cfg(hmc_method='eddy', sdc_method='fieldmap')
    config.workflow.force = ['gre-sdc-after-eddy']

    assert jacobian_provenance_for(_gre_unit()) == (['sdc'], [], None)


def test_eddy_lsr_with_gre(tmp_path):
    """Test that 'lsr' leaves a GRE fieldmap eddy applied unmodulated, as it does TOPUP's."""
    eddy_cfg = tmp_path / 'eddy_lsr.json'
    eddy_cfg.write_text(json.dumps(_LSR_EDDY_ARGS))
    _cfg(hmc_method='eddy', sdc_method='fieldmap')
    config.workflow.eddy_config = str(eddy_cfg)

    applied, unmodulated, reason = jacobian_provenance_for(_gre_unit())
    assert applied == []
    assert unmodulated == ['eddy-current', 'susceptibility']
    assert reason == 'FSL eddy ran with --resamp=lsr rather than jac'


def test_syn():
    _cfg(hmc_method='eddy', sdc_method='syn')
    unit = make_preproc_unit([SRC], method=CorrectionMethod.NIPREPS_SYN)

    assert jacobian_provenance_for(unit) == (['sdc'], [], None)


def test_topup_only():
    """Test that TOPUP is baked into eddy's own resampling, so nothing external is applied."""
    _cfg(hmc_method='eddy', sdc_method='topup')
    unit = _pepolar_unit(CorrectionMethod.PEPOLAR)

    assert jacobian_provenance_for(unit) == ([], [], None)


def test_tortoise_quadratic():
    """Test that eddy-current applies under the default DIFFPREP correction_mode, 'quadratic'."""
    _cfg(hmc_method='tortoise', sdc_method='auto', sloppy=False)
    unit = make_preproc_unit([SRC], method=None)

    assert jacobian_provenance_for(unit) == (['eddy-current'], [], None)


def test_tortoise_under_sloppy():
    """Test that --sloppy forces correction_mode='motion', with no eddy-current component."""
    _cfg(hmc_method='tortoise', sdc_method='auto', sloppy=True)
    unit = make_preproc_unit([SRC], method=None)

    assert jacobian_provenance_for(unit) == ([], [], None)


def test_tortoise_cubic(tmp_path):
    """Test that correction_mode='cubic' leaves eddy-current unmodulated (M4).

    M4: correction_mode='cubic' has no implemented determinant (only the
    quadratic terms are), so the eddy-current component is unmodulated --
    reviewer-verified correct; this pins it down with a direct test.
    """
    diffprep_cfg = tmp_path / 'diffprep_cubic.json'
    diffprep_cfg.write_text(json.dumps({'correction_mode': 'cubic'}))
    _cfg(hmc_method='tortoise', sdc_method='auto', sloppy=False)
    config.workflow.diffprep_config = str(diffprep_cfg)
    unit = make_preproc_unit([SRC], method=None)

    applied, unmodulated, reason = jacobian_provenance_for(unit)
    assert applied == []
    assert unmodulated == ['eddy-current']
    assert reason == (
        "DIFFPREP ran with correction_mode='cubic', whose eddy-current polynomial "
        '(cubic Okan terms) has no implemented Jacobian determinant.'
    )


def test_tortoise_unpile():
    """Test that the unpile field after DIFFPREP is applied with a weight.

    unpile's warp is applied downstream of DIFFPREP like SyN's, so QSIPrep
    Jacobian-modulates it; the unpile warp itself carries no Jacobian.
    """
    _cfg(hmc_method='tortoise', sdc_method='auto', sloppy=False)
    t2w = ['/data/sub-01_T2w.nii.gz']
    unit = make_preproc_unit(
        [SRC], method=CorrectionMethod.UNPILE, estimation_sources=t2w, anat_files=t2w
    )

    assert jacobian_provenance_for(unit) == (['eddy-current', 'sdc'], [], None)


@pytest.mark.parametrize(
    ('method', 'expect_applied'),
    [
        (CorrectionMethod.PEPOLAR, True),
        (CorrectionMethod.PHASEDIFF, True),
        (CorrectionMethod.NIPREPS_SYN, True),
        (CorrectionMethod.SYNB0, True),
        (CorrectionMethod.UNPILE, True),
        (None, False),
    ],
)
def test_shoreline_sdc_provenance(method, expect_applied):
    """Test that SHORELine's 'sdc' provenance follows the warps it actually builds.

    A scanner-measured fieldmap (PEPOLAR or GRE), SyN, unpile, and SynB0 +
    TOPUP after SHORELine each produce a warp that is weighted.
    """
    _cfg(hmc_method='shoreline', sdc_method='auto')
    if method in (CorrectionMethod.PEPOLAR,):
        unit = _pepolar_unit(method)
    else:
        unit = make_preproc_unit([SRC], method=method)

    applied, unmodulated, reason = jacobian_provenance_for(unit)
    assert ('sdc' in applied) is expect_applied
    assert unmodulated == []
    assert reason is None


def test_eddy_lsr_with_topup(tmp_path):
    """Test that 'lsr' + TOPUP-only leaves eddy-current and susceptibility unmodulated (F2)."""
    eddy_cfg = tmp_path / 'eddy_lsr.json'
    eddy_cfg.write_text(json.dumps(_LSR_EDDY_ARGS))
    _cfg(hmc_method='eddy', sdc_method='topup')
    config.workflow.eddy_config = str(eddy_cfg)
    unit = _pepolar_unit(CorrectionMethod.PEPOLAR)

    applied, unmodulated, reason = jacobian_provenance_for(unit)
    assert applied == []
    assert unmodulated == ['eddy-current', 'susceptibility']
    assert reason == 'FSL eddy ran with --resamp=lsr rather than jac'


def test_eddy_lsr_with_drbuddi(tmp_path):
    """Test that 'lsr' + DRBUDDI leaves only eddy-current unmodulated (F2 regression).

    F2 regression: 'lsr' + DRBUDDI records eddy-current unmodulated, but NOT
    susceptibility -- DRBUDDI's warp is applied downstream of eddy and is
    Jacobian-modulated by QSIPrep itself, regardless of eddy's own resampling
    method. This is the exact case the pre-migration global-state bug got
    wrong (see commit d89d0f1) and this migration must not regress.
    """
    eddy_cfg = tmp_path / 'eddy_lsr.json'
    eddy_cfg.write_text(json.dumps(_LSR_EDDY_ARGS))
    _cfg(hmc_method='eddy', sdc_method='drbuddi')
    config.workflow.eddy_config = str(eddy_cfg)
    unit = _pepolar_unit(CorrectionMethod.PEPOLAR)

    applied, unmodulated, reason = jacobian_provenance_for(unit)
    assert applied == ['sdc']
    assert unmodulated == ['eddy-current']
    assert 'susceptibility' not in unmodulated
    assert reason == 'FSL eddy ran with --resamp=lsr rather than jac'


def test_two_units_in_one_invocation_get_different_provenance():
    """Test that two units in one invocation get different provenance.

    The whole point of the migration: one invocation, two runs, two answers.

    Both units are built under the *same* global config (one gradwarp
    coefficient file, one HMC/SDC selection) -- exactly the shape of a
    multi-run/multi-subject invocation with mixed per-run metadata. Unit A's
    DWI has already been scanner-corrected in 3D (``ImageType`` carries
    ``DIS3D``) and has a PEPOLAR fieldmap; unit B has neither. Under the old
    invocation-global ``config.record_applied`` design these two runs would
    have shared one sidecar list (the union); the whole reason for this
    migration is that they must not.
    """
    _cfg(hmc_method='eddy', sdc_method='drbuddi')
    config.workflow.gradient_file = '/data/coeff.grad'

    unit_a = make_preproc_unit(
        [SRC],
        method=None,
        metadata={'Manufacturer': 'SIEMENS', 'ImageType': ['ORIGINAL', 'PRIMARY', 'DIS3D']},
    )
    unit_b = _pepolar_unit(CorrectionMethod.PEPOLAR)

    provenance_a = jacobian_provenance_for(unit_a)
    provenance_b = jacobian_provenance_for(unit_b)

    # Unit A: scanner already corrected the geometry (DIS3D) -> no gradwarp;
    # no fieldmap -> no sdc.
    assert provenance_a == ([], [], None)
    # Unit B: same global gradient_file, but this run's own DWI carries no
    # DIS3D tag -> gradwarp applies; DRBUDDI's warp is external -> sdc applies.
    assert provenance_b == (['gradwarp', 'sdc'], [], None)
    assert provenance_a != provenance_b
