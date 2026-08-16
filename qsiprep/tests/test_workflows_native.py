"""Construction smoke tests for the DWI workflow builders on ``PreprocUnit``.

These assert the FSL/eddy, SHORELine, and pre-HMC builders wire up a graph from
a :class:`~qsiprep.grouping.adapters.PreprocUnit` (the tortoise cluster is
covered in depth by ``test_interfaces_diffprep``).
"""

import numpy as np
import pytest

from qsiprep import config
from qsiprep.grouping.models import EstimationMethod
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
        method=EstimationMethod.PEPOLAR,
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
