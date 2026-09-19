"""Integration-marker Jacobian gating, verified by workflow construction.

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
``qsiprep/tests/test_cli.py``, this builds the actual HMC sub-workflow
(``init_fsl_hmc_wf`` for eddy, ``init_diffprep_hmc_wf`` for TORTOISE) under
that test's configuration, using synthetic ``PreprocUnit``\\ s from
``qsiprep.tests.preproc_factory`` -- no BIDS layout, no data download, no
Docker. It then asks two structural questions:

* Is ``outputnode.to_dwi_ref_warps`` wired from a source that can produce a
  real per-run warp (an SDC/DRBUDDI/T2Wreg sub-workflow), or only from
  ``GatherEddyInputs.forward_warps``, which ``qsiprep/interfaces/eddy.py``
  hardcodes to ``[]`` because eddy has already applied TOPUP internally
  ("these have already had HMC, SDC applied")?
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

from pathlib import Path

import nibabel as nb
import numpy as np
import pytest
from qsiplan.models import CorrectionMethod

from qsiprep import config
from qsiprep.tests.preproc_factory import make_preproc_unit
from qsiprep.tests.utils import get_test_data_path

SRC = '/data/sub-01_dwi.nii.gz'

#: GatherEddyInputs._run_interface (qsiprep/interfaces/eddy.py) unconditionally
#: sets ``forward_warps = []``. An edge from ``gather_inputs`` to
#: ``to_dwi_ref_warps`` is therefore a *provably empty* source, unlike
#: ``sdc_wf``/``drbuddi_wf``/``corrected_node``. Edge presence alone is not
#: enough to conclude a warp is real.
_EMPTY_FIELDWARP_SOURCES = {'gather_inputs'}


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
        'b0_threshold',
        'b1_biascorrect_stage',
        'eddy_config',
        'no_b0_harmonization',
        'denoise_method',
        'dwi_denoise_window',
        'shoreline_iters',
        'anatomical_template',
        'jacobian_weighting',
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
    config.workflow.shoreline_model = None
    config.workflow.b0_threshold = 100
    config.workflow.b1_biascorrect_stage = 'final'
    config.workflow.eddy_config = None
    config.workflow.no_b0_harmonization = False
    config.workflow.denoise_method = 'dwidenoise'
    config.workflow.dwi_denoise_window = 5
    config.workflow.shoreline_iters = 2
    config.workflow.anatomical_template = 'MNI152NLin2009cAsym'
    config.workflow.jacobian_weighting = True
    config.workflow.diffprep_config = None
    config.workflow.tortoise_gpu_cpu_ratio = None


def _write_dwi(tmp_path, name, nvols=6):
    """A tiny valid 4D DWI (+ .bval/.bvec) so merge/split nodes can build."""
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
    """Whether ``outputnode.to_dwi_ref_warps`` is wired from a real warp source.

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
    """None if ``<name>_outputs.txt`` doesn't exist; else whether it lists the map."""
    path = Path(get_test_data_path()) / f'{name}_outputs.txt'
    if not path.exists():
        return None
    return 'desc-jacobian_dwimap' in path.read_text()


def test_dsdti_synfmap_writes_jacobian(tmp_path):
    """``--ignore fieldmaps --sdc-anat-reference=invt1w`` -> fieldmap-less SyN-SDC."""
    _cfg(hmc_method='eddy', sdc_method='syn', sloppy=True)
    from qsiprep.workflows.dwi.fsl import init_fsl_hmc_wf

    unit = make_preproc_unit([SRC], method=CorrectionMethod.NIPREPS_SYN)
    wf = init_fsl_hmc_wf(unit, source_file=SRC, t2w_sdc=False)

    has_real, source = _has_real_fieldwarps(wf)
    assert has_real, f'expected a real SDC warp source, got {source!r}'
    assert _fixture_lists_jacobian('dsdti_synfmap') is True


def test_maternal_brain_project_and_forrest_gump_write_jacobian(tmp_path, monkeypatch):
    """Both datasets ship a GRE (phasediff) fieldmap; neither passes --sdc-method."""
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
    assert has_real, f'expected a real SDC warp source, got {source!r}'
    assert _fixture_lists_jacobian('maternal_brain_project') is True
    assert _fixture_lists_jacobian('forrest_gump') is True


def test_drbuddi_rpe_writes_jacobian(tmp_path):
    """``--sdc-method=drbuddi`` on a blip-up/blip-down series."""
    _cfg(hmc_method='eddy', sdc_method='drbuddi', sloppy=True)
    from qsiprep.workflows.dwi.fsl import init_fsl_hmc_wf

    unit = _rpe_unit(tmp_path, CorrectionMethod.PEPOLAR)
    wf = init_fsl_hmc_wf(unit, source_file=SRC, t2w_sdc=False)

    has_real, source = _has_real_fieldwarps(wf)
    assert has_real, f'expected a real DRBUDDI warp source, got {source!r}'
    assert _fixture_lists_jacobian('drbuddi_rpe') is True


def test_diffprep_writes_no_jacobian(tmp_path):
    """No fieldmap, no T2w, and ``--sloppy`` forces DIFFPREP's motion-only mode.

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


def test_diffprep_drbuddi_writes_jacobian(tmp_path):
    """TORTOISE DIFFPREP + DRBUDDI on an ``epi`` fieldmap.

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


def test_dsdti_topup_only_branch_has_no_jacobian(tmp_path):
    """Structural sanity check of the branch every other fixture excludes it for.

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
