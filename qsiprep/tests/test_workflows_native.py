"""Construction smoke tests for the DWI workflow builders on ``PreprocUnit``.

These assert the FSL/eddy, SHORELine, and pre-HMC builders wire up a graph from
a :class:`~qsiprep.grouping.adapters.PreprocUnit` (the tortoise cluster is
covered in depth by ``test_interfaces_diffprep``).
"""

import numpy as np
import pytest

from qsiprep import config
from qsiprep.grouping.models import CorrectionMethod
from qsiprep.tests.preproc_factory import make_preproc_unit

SRC = '/data/sub-01_dwi.nii.gz'


def _write_dwi(path, nvols=6):
    """Write a tiny valid 4D DWI (with .bval/.bvec) so merge nodes can build."""
    import nibabel as nb

    nb.Nifti1Image(np.zeros((4, 4, 4, nvols), dtype=np.int16), np.eye(4)).to_filename(str(path))
    stem = str(path).split('.nii')[0]
    bvals = np.array([0] + [1000] * (nvols - 1))
    np.savetxt(stem + '.bval', bvals[None, :], fmt='%d')
    np.savetxt(stem + '.bvec', np.zeros((3, nvols)), fmt='%.1f')
    return str(path)


class _StubFile:
    def get_entities(self):
        return {}


class _StubLayout:
    """Minimal stand-in so denoising/merge nodes can query the layout.

    ``pre_hmc`` feeds ``init_merge_and_denoise_wf``, which still reads metadata
    and probes for phase images through the layout; the real pipeline always has
    one. The probes here find no phase files (the common case).
    """

    def get_metadata(self, path):
        return {'PhaseEncodingDirection': 'j', 'TotalReadoutTime': 0.05}

    def get_entities(self, metadata=False):
        return {}

    def get_file(self, path):
        return _StubFile()

    def get(self, **query):
        return []


def _cfg(hmc_model='eddy', pepolar_method='TOPUP', layout=None):
    config.nipype.omp_nthreads = 1
    config.execution.sloppy = False
    config.execution.layout = layout
    config.workflow.hmc_model = hmc_model
    config.workflow.pepolar_method = pepolar_method
    # The legacy keys drive these tests; clear the axis keys so a selection
    # left behind by another test cannot shadow them.
    config.workflow.hmc_method = None
    config.workflow.sdc_method = None
    config.workflow.shoreline_model = None
    config.workflow.b0_threshold = 100
    config.workflow.b1_biascorrect_stage = 'final'
    config.workflow.eddy_config = None
    config.workflow.no_b0_harmonization = False
    config.workflow.denoise_method = 'dwidenoise'
    config.workflow.dwi_denoise_window = 5
    config.workflow.shoreline_iters = 2
    config.workflow.anatomical_template = 'MNI152NLin2009cAsym'
    return config


def _rpe_unit(tmp_path):
    main = _write_dwi(tmp_path / 'sub-01_dir-AP_dwi.nii.gz')
    partner = _write_dwi(tmp_path / 'sub-01_dir-PA_dwi.nii.gz')
    return make_preproc_unit(
        [main, partner],
        method=CorrectionMethod.PEPOLAR,
        pe_dirs={main: 'j', partner: 'j-'},
    )


def test_pre_hmc_single_series_builds(tmp_path):
    _cfg(layout=_StubLayout())
    from qsiprep.workflows.dwi.pre_hmc import init_dwi_pre_hmc_wf

    src = _write_dwi(tmp_path / 'sub-01_dwi.nii.gz')
    wf = init_dwi_pre_hmc_wf(make_preproc_unit([src]), orientation='LAS', source_file=src)
    assert wf.get_node('outputnode') is not None
    # A single series is merged directly, not split into polarity groups.
    assert wf.get_node('merge_plus') is None


def test_pre_hmc_rpe_series_splits_into_polarity_groups(tmp_path):
    _cfg(layout=_StubLayout())
    from qsiprep.workflows.dwi.pre_hmc import init_dwi_pre_hmc_wf

    wf = init_dwi_pre_hmc_wf(_rpe_unit(tmp_path), orientation='LAS', source_file=SRC)
    assert wf.get_node('merge_plus') is not None
    assert wf.get_node('merge_minus') is not None


def test_fsl_hmc_topup_builds(tmp_path):
    _cfg(pepolar_method='TOPUP')
    from qsiprep.workflows.dwi.fsl import init_fsl_hmc_wf

    wf = init_fsl_hmc_wf(_rpe_unit(tmp_path), source_file=SRC, t2w_sdc=False)
    assert wf.get_node('topup') is not None


def test_fsl_hmc_no_fieldmap_builds():
    _cfg(pepolar_method='TOPUP')
    from qsiprep.workflows.dwi.fsl import init_fsl_hmc_wf

    wf = init_fsl_hmc_wf(make_preproc_unit([SRC]), source_file=SRC, t2w_sdc=False)
    assert wf.get_node('topup') is None


def test_subject_summary_renders_native_groupings():
    """The subject report renders the per-output grouping dict base.py builds.

    Regression: base.py fed SubjectSummary the grouping model instead of this
    shape, crashing the ``summary`` node on every subject.
    """
    from qsiprep.interfaces.reports import SubjectSummary

    summary = SubjectSummary(
        t1w=[],
        subject_id='01',
        template='MNI152NLin2009cAsym',
        dwi_groupings={
            'sub-01': {
                'pe_dir': 'j',
                'dwi_files': ['sub-01_dir-PA_dwi.nii.gz'],
                'fieldmap': 'pepolar',
            },
            'sub-01_dir-AP': {
                'pe_dir': 'j-',
                'dwi_files': ['sub-01_dir-AP_dwi.nii.gz'],
                'fieldmap': None,
            },
        },
    )
    segment = summary._generate_segment()
    assert 'sub-01_dir-AP' in segment
    assert 'Fieldmap type: pepolar' in segment


@pytest.mark.parametrize(
    ('pe_direction', 'expected'),
    [('j', 'Anterior-Posterior'), ('i-', 'Left-Right'), (None, 'MISSING')],
)
def test_diffusion_summary_renders_pe_direction(tmp_path, pe_direction, expected):
    """DiffusionSummary tolerates a missing PE direction (base.py maps '' -> None)."""
    from qsiprep.interfaces.reports import DiffusionSummary

    report = tmp_path / 'validation.html'
    report.write_text('<p>ok</p>\n')
    summary = DiffusionSummary(
        distortion_correction='TOPUP',
        pe_direction=pe_direction,
        hmc_transform='Affine',
        hmc_model='eddy',
        b0_to_anat_transform='Rigid',
        denoise_method='dwidenoise',
        dwi_denoise_window=5,
        validation_reports=[str(report)],
    )
    assert expected in summary._generate_segment()


def test_init_sdc_wf_phasediff_builds_without_a_layout(monkeypatch):
    """The GRE path reads phase metadata off the unit, not config.execution.layout.

    Regression guard for the removed layout.get_metadata re-reads: with no layout
    set, the old code raised; the model carries the metadata now.
    """
    monkeypatch.setenv('FSLDIR', '/tmp/fakefsl')  # phdiff only checks the env is set
    _cfg(layout=None)
    from qsiprep.workflows.fieldmap import init_sdc_wf

    unit = make_preproc_unit(
        ['/data/sub-01_dwi.nii.gz'],
        method=CorrectionMethod.PHASEDIFF,
        pe_dir='j-',
        estimation_sources=['/data/sub-01_phasediff.nii.gz', '/data/sub-01_magnitude1.nii.gz'],
        metadata={'EchoTime1': 0.004, 'EchoTime2': 0.006},
    )
    wf = init_sdc_wf(unit, dwi_meta={'PhaseEncodingDirection': 'j-'})
    assert 'FMB' in wf.get_node('outputnode').inputs.method


def test_init_sdc_wf_dispatches_classic_syn():
    """A SyN unit is not bypassed: init_sdc_wf builds the classic SyN sub-workflow."""
    _cfg(layout=None)
    from qsiprep.workflows.fieldmap import init_sdc_wf

    unit = make_preproc_unit(
        ['/data/sub-01_dwi.nii.gz'],
        method=CorrectionMethod.NIPREPS_SYN,
        pe_dir='j',
        estimation_sources=['/data/sub-01_T1w.nii.gz'],
    )
    wf = init_sdc_wf(unit, dwi_meta={'PhaseEncodingDirection': 'j'})
    assert wf.name == 'sdc_wf'  # not the bypass workflow
    assert any('syn' in node.name.lower() for node in wf._get_all_nodes())


def test_init_sdc_wf_bipolar_two_phase_bypasses(monkeypatch):
    """A two-phase GRE tagged Bipolar bypasses SDC (unsupported), off the model."""
    monkeypatch.setenv('FSLDIR', '/tmp/fakefsl')  # phdiff only checks the env is set
    _cfg(layout=None)
    from qsiprep.workflows.fieldmap import init_sdc_wf

    unit = make_preproc_unit(
        ['/data/sub-01_dwi.nii.gz'],
        method=CorrectionMethod.PHASES,
        pe_dir='j-',
        estimation_sources=[
            '/data/sub-01_phase1.nii.gz',
            '/data/sub-01_phase2.nii.gz',
            '/data/sub-01_magnitude1.nii.gz',
        ],
        metadata={'DiffusionScheme': 'Bipolar'},
    )
    wf = init_sdc_wf(unit, dwi_meta={'PhaseEncodingDirection': 'j-'})
    assert wf.get_node('outputnode').inputs.method == 'None'


def test_dwi_preproc_wf_drbuddi_without_t2w_builds(tmp_path, monkeypatch):
    """The DRBUDDI extended-report block builds when no T2w is available.

    Regression: init_dwi_preproc_wf's ``else`` branch called
    init_extended_pepolar_report_wf() with no args (segment_t2w is required),
    crashing every DRBUDDI dataset that lacked a T2w.
    """
    monkeypatch.setenv('FSLDIR', '/tmp/fakefsl')
    cfg = _cfg(hmc_model='tortoise', pepolar_method='DRBUDDI', layout=_StubLayout())
    cfg.workflow.anat_modality = 't1w'
    cfg.workflow.b0_to_anat_transform = 'Rigid'
    cfg.workflow.hmc_transform = 'Affine'
    cfg.workflow.diffprep_config = None
    cfg.workflow.tortoise_gpu_cpu_ratio = None
    cfg.workflow.gpu = None
    cfg.workflow.impute_slice_threshold = 0
    from qsiprep.workflows.dwi.base import init_dwi_preproc_wf

    wf = init_dwi_preproc_wf(
        _rpe_unit(tmp_path),
        t2w_sdc=False,
        output_prefix='sub-01',
        source_file=SRC,
        anatomical_template='MNI152NLin2009cAsym',
    )
    assert wf.get_node('extended_pepolar_report_wf') is not None


def test_drbuddi_wf_feeds_sidecar_map_and_discriminator(tmp_path):
    """The DRBUDDI builder feeds the model's sidecar map (no silent disk fallback).

    Also checks the reverse-PE-series vs epi discriminator is derived from the
    unit rather than re-read at runtime.
    """
    _cfg(hmc_model='tortoise', pepolar_method='DRBUDDI')
    from qsiprep.workflows.fieldmap import init_drbuddi_wf

    wf = init_drbuddi_wf(_rpe_unit(tmp_path), t2w_sdc=False)
    gather = wf.get_node('gather_drbuddi_inputs')
    assert wf.get_node('drbuddi') is not None
    # Both series carry their PE direction into the map, so nothing is re-read.
    assert set(gather.inputs.sidecars) == {
        str(tmp_path / 'sub-01_dir-AP_dwi.nii.gz'),
        str(tmp_path / 'sub-01_dir-PA_dwi.nii.gz'),
    }
    assert gather.inputs.fieldmap_type == 'rpe_series'


def test_unit_sidecar_round_trips_through_derivatives_sidecar(tmp_path):
    """finalize's sidecar node writes valid JSON from the model (no disk reads).

    unit_to_sidecar runs at execution via DerivativesSidecar, which json-dumps
    with sort_keys=True -- so the payload must be JSON-serializable with string
    keys. Exercises that whole contract without touching a BIDS layout.
    """
    import json

    from qsiprep.grouping.adapters import unit_to_sidecar
    from qsiprep.interfaces.bids import DerivativesSidecar

    unit = make_preproc_unit(
        ['/data/sub-01_dir-AP_dwi.nii.gz', '/data/sub-01_dir-PA_dwi.nii.gz'],
        method=CorrectionMethod.PEPOLAR,
        pe_dirs={
            '/data/sub-01_dir-AP_dwi.nii.gz': 'j',
            '/data/sub-01_dir-PA_dwi.nii.gz': 'j-',
        },
        metadata={'EchoTime': 0.1},
    )
    node = DerivativesSidecar(
        sidecar_data=unit_to_sidecar(unit),
        source_file=str(tmp_path / 'sub-01_dwi.nii.gz'),
    )
    result = node.run(cwd=str(tmp_path))
    written = json.loads(open(result.outputs.derivatives_json).read())
    # Metadata shared by both series is promoted to the top level.
    assert written['EchoTime'] == 0.1
    assert written['ScanGrouping']['method'] == 'pepolar'
    assert written['Sources'] == ['sub-01_dir-AP_dwi.nii.gz', 'sub-01_dir-PA_dwi.nii.gz']


def test_eddy_grouping_from_sidecars_needs_no_disk():
    """eddy's acqp/index build from the model's sidecar map, not from disk."""
    from qsiprep.interfaces.epi_fmap import get_distortion_grouping

    ap, pa = '/nope/sub-01_dir-AP_dwi.nii.gz', '/nope/sub-01_dir-PA_dwi.nii.gz'
    origins = [ap] * 3 + [pa] * 3
    sidecars = {
        ap: {'PhaseEncodingDirection': 'j', 'TotalReadoutTime': 0.05},
        pa: {'PhaseEncodingDirection': 'j-', 'TotalReadoutTime': 0.05},
    }
    acqps, groups = get_distortion_grouping(origins, sidecars=sidecars)
    assert acqps == ['0 1 0 0.050000', '0 -1 0 0.050000']
    assert groups == [1, 1, 1, 2, 2, 2]


def test_drbuddi_blip_assignments_from_sidecars_needs_no_disk():
    """DRBUDDI's per-volume blip labels come from the sidecar map, not from disk."""
    from qsiprep.interfaces.tortoise import split_into_up_and_down_niis

    ap, pa = '/nope/sub-01_dir-AP_dwi.nii.gz', '/nope/sub-01_dir-PA_dwi.nii.gz'
    origins = [ap, ap, pa, pa]
    sidecars = {
        ap: {'PhaseEncodingDirection': 'j', 'TotalReadoutTime': 0.05},
        pa: {'PhaseEncodingDirection': 'j-', 'TotalReadoutTime': 0.05},
    }
    # assignments_only reads no files; per-volume lists just need matching length.
    per_vol = ['v0', 'v1', 'v2', 'v3']
    assignments = split_into_up_and_down_niis(
        dwi_files=per_vol,
        bval_files=per_vol,
        bvec_files=per_vol,
        original_images=origins,
        prefix='/tmp/unused',
        make_bmat=False,
        assignments_only=True,
        sidecars=sidecars,
    )
    # First volume's group is "up"; opposite polarity is "down".
    assert assignments == ['up', 'up', 'down', 'down']


# NOTE: the SHORELine path (init_qsiprep_hmcsdc_wf) is deprecated and its
# construction depends on init_dwi_hmc_wf internals that need a fuller config
# than a smoke warrants; the CI ``drbuddi_shoreline_epi`` / ``drbuddi_tensorline_epi``
# jobs cover it. Its PreprocUnit consumption follows the same pattern validated
# here and in test_interfaces_diffprep.


def test_unknown_hmc_model_is_rejected_at_selection_time(tmp_path):
    """The subject workflow resolves the method selection before building
    anything; garbage config dies there, not deep in a builder."""
    _cfg(hmc_model='bogus', layout=_StubLayout())
    from qsiprep.grouping.methods import method_selection_from_config

    with pytest.raises(ValueError, match='hmc'):
        method_selection_from_config()


def test_distortion_group_merge_wf_rejects_unknown_strategy():
    _cfg()
    from qsiprep.workflows.dwi.distortion_group_merge import init_distortion_group_merge_wf

    with pytest.raises(ValueError, match='merging_strategy'):
        init_distortion_group_merge_wf(
            merging_strategy='bogus',
            inputs_list=['sub-01-run-1', 'sub-01-run-2'],
            source_file='sub-01_dwi.nii.gz',
            output_prefix='sub-01',
            name='bogus_merge_wf',
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


def test_legacy_method_keys_read_only_at_allowlisted_sites():
    """Routing reads the compiled plan; the back-filled legacy keys survive only
    at these sites (report gating held for SHORELine shape-compatibility, and
    display/vocabulary strings), so new reads cannot creep in unnoticed."""
    import pathlib
    import re

    root = pathlib.Path(__file__).parent.parent
    allowed = {
        # SHORELine report-shape compatibility (see the comments at the sites).
        'workflows/dwi/base.py': {'pepolar_method': 2, 'hmc_model': 1},
        'workflows/dwi/finalize.py': {'pepolar_method': 1},
        # Display strings and the SHORELine model vocabulary.
        'workflows/dwi/derivatives.py': {'hmc_model': 3},
        'workflows/dwi/hmc.py': {'hmc_model': 3},
    }
    found: dict = {}
    for path in (root / 'workflows').rglob('*.py'):
        text = path.read_text()
        for key in ('pepolar_method', 'hmc_model'):
            count = len(re.findall(rf'config\.workflow\.{key}\b', text))
            if count:
                found.setdefault(str(path.relative_to(root)), {})[key] = count
    assert found == allowed
