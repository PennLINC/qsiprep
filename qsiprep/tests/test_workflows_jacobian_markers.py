r"""Integration-marker Jacobian gating, verified by workflow construction.

Task 12 updated 10 fixture ``_outputs.txt`` files by *inferring* from each
integration test's CLI flags whether QSIPrep would write a
``_desc-jacobian_dwimap`` derivative. That inference is reasoning, not
verification: whether the derivative is written depends on runtime data --
whether ``GatherEddyInputs.forward_warps`` or
``OkanQuadraticJacobian.ec_jacobian_images`` end up ``Undefined`` -- which is a
question workflow *construction* can answer in seconds, without downloading
any of the real datasets those markers use (they are pulled from CircleCI-only
URLs; see ``.circleci/continue_config.yml``) or running qsiprep end to end.

For every integration marker whose CLI flags could be identified in
``qsiprep/tests/test_cli.py``, this builds the actual HMC sub-workflow that
``qsiprep/workflows/dwi/base.py`` (``init_dwi_preproc_wf``, the
``hmc_tool ==`` dispatch around line 267) would build for that marker's own
``--hmc-method`` -- ``init_fsl_hmc_wf`` for ``eddy``,
``init_qsiprep_hmcsdc_wf`` for ``shoreline``, ``init_diffprep_hmc_wf`` for
``tortoise`` -- under that test's configuration, using synthetic
``PreprocUnit``\ s from ``qsiprep.tests.preproc_factory`` -- no BIDS layout,
no data download, no Docker. Matching the dispatch matters: an earlier
version of this module built ``init_fsl_hmc_wf`` (the eddy path) for
``maternal_brain_project``, which actually runs ``--hmc-method=shoreline`` and
so never touches ``init_fsl_hmc_wf`` or ``GatherEddyInputs`` at all -- fixed
below.

It then asks two structural questions:

* Is ``outputnode.to_dwi_ref_warps`` wired from a source that can produce a
  real per-run warp (an SDC/DRBUDDI/T2Wreg sub-workflow), or only from a
  provably-empty source? Two such sources exist, and they are different
  mechanisms, not one:

  - ``GatherEddyInputs.forward_warps`` (``qsiprep/interfaces/eddy.py``),
    unconditionally hardcoded to ``[]`` because eddy has already applied
    TOPUP internally ("these have already had HMC, SDC applied"). This is the
    eddy-only ``init_fsl_hmc_wf`` path.
  - ``sdc_bypass_wf`` (``qsiprep/workflows/fieldmap/base.py:116,140-149``):
    ``init_sdc_wf`` names its returned sub-workflow ``'sdc_bypass_wf'``
    instead of ``'sdc_wf'`` precisely when ``does_sdc`` is ``False`` (no
    scanner-measured fieldmap and not classic SyN), and in that branch it
    never connects anything to its own ``outputnode.out_warp`` -- confirmed
    directly by construction: building ``init_qsiprep_hmcsdc_wf`` with
    ``method=None`` yields a ``to_dwi_ref_warps`` edge from a node literally
    named ``sdc_bypass_wf``, vs. ``sdc_wf`` for a GRE/SyN unit. This is the
    SHORELine (``init_qsiprep_hmcsdc_wf``) path's equivalent; it shares
    ``init_sdc_wf`` with the eddy path's GRE/SyN branch, so the same
    "sdc_wf-vs-bypass" name distinction applies there too, it is just never
    the deciding factor for the eddy markers checked here (they either hit
    GatherEddyInputs or a real fieldmap/DRBUDDI path).

* For TORTOISE, what ``correction_mode`` is baked into the ``ec_jacobian``
  node's inputs? ``OkanQuadraticJacobian`` returns ``Undefined`` for
  ``'motion'`` and ``'cubic'`` (see ``qsiprep/interfaces/jacobian.py``), and
  ``--sloppy`` unconditionally downgrades DIFFPREP to ``'motion'``
  (``qsiprep/workflows/dwi/diffprep.py``).

Either factor being "real" means ``ComposeJacobianWeights`` produces a weight
map for that marker, which means ``ds_jacobian`` writes the derivative. This
is compared against whatever the marker's ``_outputs.txt`` actually lists.

``test_diffprep_writes_no_jacobian`` caught a real fixture bug this way:
``diffprep_outputs.txt`` listed the derivative, but ``test_diffprep`` runs
fieldmap-less (HMC only, no T2w) *and* always passes ``--sloppy``, so neither
factor is ever real for that marker. The two lines were removed from the
fixture as part of this change.

``dsdti_topup`` has no corresponding test function (``grep -n
'@pytest.mark.dsdti_topup' qsiprep/tests/test_cli.py`` returns nothing) --
it is exercised here only as a structural sanity check of the "eddy +
TOPUP-only, no DRBUDDI, no gradwarp" branch that justifies excluding every
other eddy+TOPUP-only fixture, not as a live CI gate.

Not fully covered: ``diffprep_drbuddi`` ships no ``_outputs.txt`` at all
(``test_cli.py`` passes ``check_outputs=False`` for it, and for the other
TORTOISE+DRBUDDI rpe_series markers) because DIFFPREP's per-direction
split/recombine has no expected-output manifest yet. This module still builds
its workflow and records the expected answer for when that manifest exists.
"""

import json
from pathlib import Path

import nibabel as nb
import numpy as np
import pytest
from qsiplan.models import CorrectionMethod

from qsiprep import config
from qsiprep.tests.preproc_factory import make_preproc_unit
from qsiprep.tests.utils import get_test_data_path
from qsiprep.utils.jacobian_provenance import jacobian_provenance_for

SRC = '/data/sub-01_dwi.nii.gz'

#: Node names whose ``to_dwi_ref_warps`` output is provably never a real
#: per-run warp -- each verified by reading the source, not by inference:
#:
#: * ``gather_inputs``: ``GatherEddyInputs._run_interface``
#:   (qsiprep/interfaces/eddy.py) unconditionally sets ``forward_warps = []``.
#:   Only reachable via the eddy backend (``init_fsl_hmc_wf``).
#: * ``sdc_bypass_wf``: ``init_sdc_wf`` (qsiprep/workflows/fieldmap/base.py)
#:   names its own sub-workflow this way exactly when ``does_sdc`` is False,
#:   and never connects anything to that sub-workflow's own
#:   ``outputnode.out_warp`` in that branch. Reachable from both the eddy
#:   backend's GRE/SyN branch and the SHORELine backend
#:   (``init_qsiprep_hmcsdc_wf``), which share ``init_sdc_wf``.
#:
#: Edge presence alone is not enough to conclude a warp is real -- that was
#: the bug in the first version of this check.
_EMPTY_FIELDWARP_SOURCES = {'gather_inputs', 'sdc_bypass_wf'}


class _StubLayout:
    """Minimal layout stand-in; see test_workflows_native.py's twin."""

    def get_metadata(self, path):
        raise AssertionError('layout.get_metadata should not be called')

    def get_entities(self, metadata=False):
        return {}

    def get_file(self, path):
        class _StubFile:
            def get_entities(self):
                return {}

        return _StubFile()

    def get(self, **query):
        return []


@pytest.fixture(autouse=True)
def _reset_config():
    workflow_fields = (
        'hmc_method',
        'sdc_method',
        'shoreline_model',
        'hmc_transform',
        'b0_threshold',
        'dwi_biascorrect',
        'eddy_config',
        'no_b0_harmonization',
        'denoise_method',
        'dwidenoise_window',
        'shoreline_iters',
        'anatomical_template',
        'ignore',
        'diffprep_config',
        'tortoise_gpu_cpu_ratio',
    )
    saved_workflow = {name: getattr(config.workflow, name) for name in workflow_fields}
    saved_sloppy = config.execution.sloppy
    saved_layout = config.execution.layout
    saved_omp_nthreads = config.nipype.omp_nthreads
    yield
    for name, value in saved_workflow.items():
        setattr(config.workflow, name, value)
    config.execution.sloppy = saved_sloppy
    config.execution.layout = saved_layout
    config.nipype.omp_nthreads = saved_omp_nthreads


def _cfg(hmc_method, sdc_method, sloppy):
    config.nipype.omp_nthreads = 1
    config.execution.sloppy = sloppy
    config.execution.layout = _StubLayout()
    config.workflow.hmc_method = hmc_method
    config.workflow.sdc_method = sdc_method
    # Only the shoreline backend reads these (init_dwi_model_hmc_wf); the CLI
    # parser resolves them from --shoreline-config (or the shipped defaults,
    # 3dshore/Affine -- see load_shoreline_config(None)) only when
    # --hmc-method shoreline, and forces them to None otherwise.
    config.workflow.shoreline_model = '3dshore' if hmc_method == 'shoreline' else None
    config.workflow.hmc_transform = 'Affine' if hmc_method == 'shoreline' else None
    config.workflow.b0_threshold = 100
    config.workflow.dwi_biascorrect = 'n4'
    config.workflow.eddy_config = None
    config.workflow.no_b0_harmonization = False
    config.workflow.denoise_method = 'dwidenoise'
    config.workflow.dwidenoise_window = 5
    config.workflow.shoreline_iters = 2
    config.workflow.anatomical_template = 'MNI152NLin2009cAsym'
    config.workflow.ignore = []
    config.workflow.diffprep_config = None
    config.workflow.tortoise_gpu_cpu_ratio = None


def _write_dwi(tmp_path, name, nvols=6):
    """Write a tiny valid 4D DWI (+ .bval/.bvec) so merge/split nodes can build."""
    path = tmp_path / name
    nb.Nifti1Image(np.zeros((4, 4, 4, nvols), dtype=np.int16), np.eye(4)).to_filename(str(path))
    stem = str(path).split('.nii')[0]
    bvals = np.array([0] + [1000] * (nvols - 1))
    np.savetxt(stem + '.bval', bvals[None, :], fmt='%d')
    np.savetxt(stem + '.bvec', np.zeros((3, nvols)), fmt='%.1f')
    return str(path)


def _rpe_unit(tmp_path, method):
    main = _write_dwi(tmp_path, 'sub-01_dir-AP_dwi.nii.gz')
    partner = _write_dwi(tmp_path, 'sub-01_dir-PA_dwi.nii.gz')
    return make_preproc_unit([main, partner], method=method, pe_dirs={main: 'j', partner: 'j-'})


def _has_real_fieldwarps(wf):
    """Check whether ``outputnode.to_dwi_ref_warps`` is wired from a real warp source.

    Returns ``(is_real, source_node_name)``. ``source_node_name`` is ``None``
    if nothing at all is connected to that field.
    """
    outputnode = wf.get_node('outputnode')
    for src, dst, data in wf._graph.edges(data=True):
        if dst is not outputnode:
            continue
        for _, dest in data['connect']:
            dest_name = dest[0] if isinstance(dest, tuple) else dest
            if dest_name == 'to_dwi_ref_warps':
                real = src.name not in _EMPTY_FIELDWARP_SOURCES
                return real, src.name
    return False, None


def _fixture_lists_jacobian(name):
    """Return whether ``<name>_outputs.txt`` lists the map, or None if it doesn't exist."""
    path = Path(get_test_data_path()) / f'{name}_outputs.txt'
    if not path.exists():
        return None
    return 'desc-jacobian_dwimap' in path.read_text()


def test_dsdti_synfmap_writes_jacobian(tmp_path):
    """Test that dsdti_synfmap (fieldmap-less SyN-SDC) writes the Jacobian.

    ``--ignore fieldmaps --sdc-anat-reference=invt1w`` -> fieldmap-less SyN-SDC.

    ``sdc_method='syn'`` below is config plumbing for ``_cfg``, not a mirror of
    the real marker's CLI: ``test_dsdti_synfmap`` never passes ``--sdc-method``
    at all. The branch that actually matters here
    (``fsl.py``'s ``unit.is_gre or unit.is_nipreps_syn`` check) keys off the
    synthetic unit's ``method=CorrectionMethod.NIPREPS_SYN`` below, not off
    ``config.workflow.sdc_method`` (that config field only selects between
    TOPUP/DRBUDDI for PEPOLAR units; see ``qsiplan/plan.py:280-294``).
    """
    _cfg(hmc_method='eddy', sdc_method='syn', sloppy=True)
    from qsiprep.workflows.dwi.fsl import init_fsl_hmc_wf

    unit = make_preproc_unit([SRC], method=CorrectionMethod.NIPREPS_SYN)
    wf = init_fsl_hmc_wf(unit, source_file=SRC, t2w_sdc=False)

    has_real, source = _has_real_fieldwarps(wf)
    assert has_real, f'expected a real SDC warp source, got {source!r}'
    assert _fixture_lists_jacobian('dsdti_synfmap') is True
    # jacobian_provenance_for mirrors init_fsl_hmc_wf's own GRE/SyN branch
    # (unit.is_gre or unit.is_nipreps_syn) from this same, real unit.
    applied, unmodulated, reason = jacobian_provenance_for(unit, t2w_sdc=False)
    assert (applied, unmodulated, reason) == (['sdc'], [], None)
    # I3: jacobian_provenance_for never inspects the workflow it mirrors --
    # relate it to the has_real_fieldwarps oracle so a drift between the two
    # is caught here rather than only by a full build+sidecar comparison.
    assert has_real == ('sdc' in applied)


def test_forrest_gump_writes_no_jacobian(monkeypatch):
    """Test that forrest_gump writes no Jacobian.

    ``test_forrest_gump`` passes no ``--hmc-method``, so it defaults to eddy
    with a GRE (phasediff) fieldmap -> ``init_fsl_hmc_wf``'s GRE branch, which
    hands the field to eddy (``--field``). eddy applies and Jacobian-modulates it
    internally, as it does TOPUP's field, so QSIPrep holds no weight map.
    """
    monkeypatch.setenv('FSLDIR', '/tmp/fakefsl')
    _cfg(hmc_method='eddy', sdc_method='fieldmap', sloppy=True)
    from qsiprep.workflows.dwi.fsl import init_fsl_hmc_wf

    unit = make_preproc_unit(
        [SRC],
        method=CorrectionMethod.PHASEDIFF,
        estimation_sources=['/data/sub-01_phasediff.nii.gz', '/data/sub-01_magnitude1.nii.gz'],
        metadata={'EchoTime1': 0.004, 'EchoTime2': 0.006},
    )
    wf = init_fsl_hmc_wf(unit, source_file=SRC, t2w_sdc=False)

    has_real, source = _has_real_fieldwarps(wf)
    assert not has_real
    assert source == 'gather_inputs'
    assert _fixture_lists_jacobian('forrest_gump') is False
    applied, unmodulated, reason = jacobian_provenance_for(unit, t2w_sdc=False)
    assert (applied, unmodulated, reason) == ([], [], None)
    assert has_real == ('sdc' in applied)


def test_maternal_brain_project_writes_jacobian(monkeypatch):
    """Test that maternal_brain_project writes the Jacobian.

    ``test_maternal_brain_project`` passes ``--hmc-method=shoreline``
    (``qsiprep/tests/test_cli.py:762``), which ``base.py``'s ``hmc_tool``
    dispatch (around line 267) routes to ``init_qsiprep_hmcsdc_wf`` --
    *not* ``init_fsl_hmc_wf``/``GatherEddyInputs``, which only exist on the
    eddy path. The dataset ships a GRE (phasediff) fieldmap and no
    ``--sdc-method`` override, which here means ``unit.has_scanner_measured_
    fieldmap`` is True, so ``init_sdc_wf`` builds the real ``'sdc_wf'``
    (not the ``'sdc_bypass_wf'`` no-op).

    An earlier version of this test built ``init_fsl_hmc_wf`` for this marker
    -- exactly the "workflow the real marker never runs" mistake this module
    otherwise guards against.
    """
    monkeypatch.setenv('FSLDIR', '/tmp/fakefsl')
    _cfg(hmc_method='shoreline', sdc_method='auto', sloppy=True)
    from qsiprep.workflows.dwi.hmc_sdc import init_qsiprep_hmcsdc_wf

    unit = make_preproc_unit(
        [SRC],
        method=CorrectionMethod.PHASEDIFF,
        estimation_sources=['/data/sub-01_phasediff.nii.gz', '/data/sub-01_magnitude1.nii.gz'],
        metadata={'EchoTime1': 0.004, 'EchoTime2': 0.006},
    )
    wf = init_qsiprep_hmcsdc_wf(
        unit,
        source_file=SRC,
        t2w_sdc=False,
        anatomical_template=config.workflow.anatomical_template,
    )

    has_real, source = _has_real_fieldwarps(wf)
    assert has_real, f'expected a real SDC warp source, got {source!r}'
    assert source == 'sdc_wf'
    assert _fixture_lists_jacobian('maternal_brain_project') is True
    # jacobian_provenance_for's shoreline branch has no TOPUP carve-out either
    # (see its docstring) -- any correction method reaches fieldwarps.
    applied, unmodulated, reason = jacobian_provenance_for(unit, t2w_sdc=False)
    assert (applied, unmodulated, reason) == (['sdc'], [], None)
    assert has_real == ('sdc' in applied)


def test_shoreline_no_fieldmap_has_no_real_fieldwarps(tmp_path):
    """Test that SHORELine without a fieldmap has no real fieldwarps.

    This pins the SHORELine backend's own provably-empty source.

    This is a live marker check. ``dwiref`` runs
    --hmc-method=shoreline over two sessions that each hold one phase-encoding
    direction and no fieldmap, so each session takes ``sdc_bypass_wf`` and has
    nothing to modulate. Its output fixture wrongly expected a
    desc-jacobian_dwimap until that run proved otherwise; the expectation was
    removed rather than a unity map synthesised, for the reason given in
    ``init_dwi_derivatives_wf``.
    """
    _cfg(hmc_method='shoreline', sdc_method='auto', sloppy=True)
    from qsiprep.workflows.dwi.hmc_sdc import init_qsiprep_hmcsdc_wf

    unit = make_preproc_unit([SRC], method=None)
    wf = init_qsiprep_hmcsdc_wf(
        unit,
        source_file=SRC,
        t2w_sdc=False,
        anatomical_template=config.workflow.anatomical_template,
    )

    has_real, source = _has_real_fieldwarps(wf)
    assert not has_real
    assert source == 'sdc_bypass_wf'
    # No fieldmap at all on the SHORELine backend: nothing to apply.
    applied, unmodulated, reason = jacobian_provenance_for(unit, t2w_sdc=False)
    assert (applied, unmodulated, reason) == ([], [], None)
    assert has_real == ('sdc' in applied)


@pytest.mark.parametrize('method', [CorrectionMethod.SYNB0, CorrectionMethod.T2WREG])
def test_shoreline_fieldmapless_has_no_real_fieldwarps(method):
    """Test that SHORELine with a fieldmap-less method has no real fieldwarps (C2).

    C2 regression: SHORELine + a fieldmap-less method (SYNB0/T2Wreg) is
    still a ``unit.method is not None`` case, but ``init_sdc_wf``'s own
    ``does_sdc`` gate (``qsiprep/workflows/fieldmap/base.py:114-115``,
    ``unit.has_scanner_measured_fieldmap or unit.is_nipreps_syn``) is False
    for both -- SYNB0 and T2Wreg are fieldmap-less, so neither is a scanner-
    measured fieldmap nor classic NiPreps SyN -- so it builds the
    ``sdc_bypass_wf`` no-op, same as ``method=None`` above. T2Wreg and SyNb0
    are only ever applied by the TORTOISE backend
    (``qsiprep/workflows/fieldmap/base.py``'s own ``init_sdc_wf`` docstring);
    on SHORELine, ``_stages_for_unit`` (``qsiplan/plan.py``) never attaches an
    ``ESTIMATE_AND_APPLY`` stage for either, so no warp is ever produced here.

    Before this fix, ``_shoreline_sdc_applied`` returned True for any non-None
    method, so it predicted 'sdc' applied for both -- exactly the drift this
    module's ``has_real == ('sdc' in applied)`` assertion below is built to
    catch, and exactly the gap the original enumerated cases (GRE, and
    ``method=None`` above) did not cover.
    """
    _cfg(hmc_method='shoreline', sdc_method='auto', sloppy=True)
    from qsiprep.workflows.dwi.hmc_sdc import init_qsiprep_hmcsdc_wf

    unit = make_preproc_unit([SRC], method=method)
    wf = init_qsiprep_hmcsdc_wf(
        unit,
        source_file=SRC,
        t2w_sdc=False,
        anatomical_template=config.workflow.anatomical_template,
    )

    has_real, source = _has_real_fieldwarps(wf)
    assert not has_real
    assert source == 'sdc_bypass_wf'
    applied, unmodulated, reason = jacobian_provenance_for(unit, t2w_sdc=False)
    assert (applied, unmodulated, reason) == ([], [], None)
    assert has_real == ('sdc' in applied)


def test_drbuddi_rpe_writes_jacobian(tmp_path):
    """Test that ``--sdc-method=drbuddi`` on a blip-up/blip-down series writes the Jacobian."""
    _cfg(hmc_method='eddy', sdc_method='drbuddi', sloppy=True)
    from qsiprep.workflows.dwi.fsl import init_fsl_hmc_wf

    unit = _rpe_unit(tmp_path, CorrectionMethod.PEPOLAR)
    wf = init_fsl_hmc_wf(unit, source_file=SRC, t2w_sdc=False)

    has_real, source = _has_real_fieldwarps(wf)
    assert has_real, f'expected a real DRBUDDI warp source, got {source!r}'
    assert _fixture_lists_jacobian('drbuddi_rpe') is True
    applied, unmodulated, reason = jacobian_provenance_for(unit, t2w_sdc=False)
    assert (applied, unmodulated, reason) == (['sdc'], [], None)
    assert has_real == ('sdc' in applied)


def test_diffprep_writes_no_jacobian(tmp_path):
    """Test that the diffprep marker writes no Jacobian.

    No fieldmap, no T2w, and ``--sloppy`` forces DIFFPREP's motion-only mode.

    Neither factor ``ComposeJacobianWeights`` can use is real here: there is
    no SDC warp (the fieldmap-less HMC-only branch never touches
    ``to_dwi_ref_warps``), and ``OkanQuadraticJacobian`` returns ``Undefined``
    for ``correction_mode='motion'``. ``diffprep_outputs.txt`` listed the
    derivative until this test caught the mismatch.
    """
    _cfg(hmc_method='tortoise', sdc_method='auto', sloppy=True)
    from qsiprep.workflows.dwi.diffprep import init_diffprep_hmc_wf

    unit = make_preproc_unit([SRC], method=None)
    wf = init_diffprep_hmc_wf(unit, source_file=SRC, t2w_sdc=False)

    has_real, source = _has_real_fieldwarps(wf)
    ec_mode = wf.get_node('ec_jacobian').inputs.correction_mode
    assert not has_real, f'expected no real SDC warp source, got {source!r}'
    assert ec_mode == 'motion'
    assert _fixture_lists_jacobian('diffprep') is False
    # 'motion' has no eddy-current component at all -- it does not occur, so
    # it belongs in neither AppliedCorrections nor UnmodulatedCorrections.
    applied, unmodulated, reason = jacobian_provenance_for(unit, t2w_sdc=False)
    assert (applied, unmodulated, reason) == ([], [], None)
    assert has_real == ('sdc' in applied)


def test_diffprep_drbuddi_writes_jacobian(tmp_path):
    """Test that TORTOISE DIFFPREP + DRBUDDI on an ``epi`` fieldmap writes the Jacobian.

    The SDC warp is real even though ``--sloppy`` still disables the
    eddy-current component (``correction_mode='motion'``).
    """
    _cfg(hmc_method='tortoise', sdc_method='drbuddi', sloppy=True)
    from qsiprep.workflows.dwi.diffprep import init_diffprep_hmc_wf

    unit = _rpe_unit(tmp_path, CorrectionMethod.PEPOLAR)
    wf = init_diffprep_hmc_wf(unit, source_file=SRC, t2w_sdc=False)

    has_real, source = _has_real_fieldwarps(wf)
    assert has_real, f'expected a real DRBUDDI warp source, got {source!r}'
    # No _outputs.txt ships for this marker yet (check_outputs=False in
    # test_cli.py's test_diffprep_drbuddi): nothing to compare against.
    assert _fixture_lists_jacobian('diffprep_drbuddi') is None
    # --sloppy still downgrades to correction_mode='motion', so only 'sdc' is
    # applied here; see test_diffprep_quadratic_records_eddy_current_applied
    # for the non-sloppy 'eddy-current' case.
    applied, unmodulated, reason = jacobian_provenance_for(unit, t2w_sdc=False)
    assert (applied, unmodulated, reason) == (['sdc'], [], None)
    assert has_real == ('sdc' in applied)


def test_diffprep_quadratic_records_eddy_current_applied(tmp_path):
    """Test that non-``--sloppy`` TORTOISE DRBUDDI records both 'sdc' and 'eddy-current'.

    ``diffprep_cfg``'s default ``correction_mode`` is ``'quadratic'``
    (``qsiprep/workflows/dwi/diffprep.py:85``), only downgraded to
    ``'motion'`` by ``--sloppy`` -- see ``test_diffprep_drbuddi_writes_
    jacobian`` for that case.
    """
    _cfg(hmc_method='tortoise', sdc_method='drbuddi', sloppy=False)
    from qsiprep.workflows.dwi.diffprep import init_diffprep_hmc_wf

    unit = _rpe_unit(tmp_path, CorrectionMethod.PEPOLAR)
    wf = init_diffprep_hmc_wf(unit, source_file=SRC, t2w_sdc=False)

    assert wf.get_node('ec_jacobian').inputs.correction_mode == 'quadratic'
    # jacobian_provenance_for orders eddy-current before sdc, matching the
    # order diffprep.py itself used to record them in (ec_jacobian is built
    # before the SDC branch dispatch).
    assert jacobian_provenance_for(unit, t2w_sdc=False) == (['eddy-current', 'sdc'], [], None)


def test_dsdti_topup_only_branch_has_no_jacobian(tmp_path):
    """Test that the dsdti_topup (eddy + TOPUP-only) branch has no Jacobian.

    This is a structural sanity check of the branch every other fixture excludes it for.

    No ``dsdti_topup`` test function exists (confirmed by grepping
    ``test_cli.py`` for ``@pytest.mark.dsdti_topup``), so this does not gate
    CI. It pins the "eddy + TOPUP-only, no DRBUDDI, no gradwarp" branch:
    ``GatherEddyInputs.forward_warps`` is the sole source, and it is always
    ``[]``.
    """
    _cfg(hmc_method='eddy', sdc_method='topup', sloppy=True)
    from qsiprep.workflows.dwi.fsl import init_fsl_hmc_wf

    unit = _rpe_unit(tmp_path, CorrectionMethod.PEPOLAR)
    wf = init_fsl_hmc_wf(unit, source_file=SRC, t2w_sdc=False)

    has_real, source = _has_real_fieldwarps(wf)
    assert not has_real
    assert source == 'gather_inputs'
    assert _fixture_lists_jacobian('dsdti_topup') is False
    # TOPUP is baked into eddy's own resampling on this path -- QSIPrep itself
    # applies nothing external, so 'sdc' must not appear as applied.
    applied, unmodulated, reason = jacobian_provenance_for(unit, t2w_sdc=False)
    assert (applied, unmodulated, reason) == ([], [], None)
    assert has_real == ('sdc' in applied)


#: A full ``--eddy-config`` override with ``method='lsr'`` instead of the
#: shipped default ``'jac'`` -- see ``qsiprep/tests/data/eddy_params.json``'s
#: real default (mirrored here) for the keys ``ExtendedEddy`` otherwise fills
#: in from trait defaults; only ``method`` matters for this test.
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


def test_lsr_records_susceptibility_only_when_topup_is_the_sdc_method(tmp_path):
    """Test that 'lsr' records susceptibility only when TOPUP is the SDC method (F2).

    F2 regression: ``--resamp=lsr`` always leaves 'eddy-current'
    unmodulated (eddy has baked its own resampling in and exports nothing to
    weight from it), but 'susceptibility' must be recorded as unmodulated
    only when TOPUP is actually this run's susceptibility source.
    ``init_fsl_hmc_wf`` used to record both unconditionally at eddy-config-read
    time (via ``config.record_applied``/``config.record_unmodulated``, since
    removed), before ``run_topup`` was even known -- wrong for DRBUDDI/GRE/SyN,
    whose warp is applied downstream of eddy and *is* Jacobian-modulated by
    QSIPrep regardless of eddy's own resampling method.
    """
    eddy_cfg = tmp_path / 'eddy_lsr.json'
    eddy_cfg.write_text(json.dumps(_LSR_EDDY_ARGS))
    from qsiprep.workflows.dwi.fsl import init_fsl_hmc_wf

    _cfg(hmc_method='eddy', sdc_method='topup', sloppy=True)
    config.workflow.eddy_config = str(eddy_cfg)
    unit = _rpe_unit(tmp_path, CorrectionMethod.PEPOLAR)
    init_fsl_hmc_wf(unit, source_file=SRC, t2w_sdc=False)

    applied, unmodulated, reason = jacobian_provenance_for(unit, t2w_sdc=False)
    assert unmodulated == ['eddy-current', 'susceptibility']
    assert applied == []
    assert reason == 'FSL eddy ran with --resamp=lsr rather than jac'

    _cfg(hmc_method='eddy', sdc_method='drbuddi', sloppy=True)
    config.workflow.eddy_config = str(eddy_cfg)
    unit = _rpe_unit(tmp_path, CorrectionMethod.PEPOLAR)
    init_fsl_hmc_wf(unit, source_file=SRC, t2w_sdc=False)

    # DRBUDDI's warp is applied downstream of eddy and QSIPrep modulates it
    # itself -- 'lsr' only ever affects eddy's own eddy-current component.
    applied, unmodulated, reason = jacobian_provenance_for(unit, t2w_sdc=False)
    assert unmodulated == ['eddy-current']
    assert applied == ['sdc']
    assert reason == 'FSL eddy ran with --resamp=lsr rather than jac'
