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
    config.workflow.b0_threshold = 100
    config.workflow.b1_biascorrect_stage = 'final'
    config.workflow.eddy_config = None
    config.workflow.no_b0_harmonization = False
    config.workflow.denoise_method = 'dwidenoise'
    config.workflow.dwi_denoise_window = 5
    config.workflow.shoreline_iters = 2
    config.workflow.output_spaces = ['acpc:res-2mm', 'MNI152NLin2009cAsym']
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
        templates=['MNI152NLin2009cAsym'],
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
    from qsiprep.utils.spaces import SpaceSpec
    from qsiprep.workflows.dwi.base import init_dwi_preproc_wf

    wf = init_dwi_preproc_wf(
        _rpe_unit(tmp_path),
        t2w_sdc=False,
        output_prefix='sub-01',
        source_file=SRC,
        acpc_anchor=SpaceSpec(space='MNI152NLin2009cAsym'),
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


def test_diffprep_sdc_uses_the_acpc_anchor(tmp_path):
    """diffprep reads the ACPC anchor selected from output_spaces, not a template config field."""
    from qsiprep import config
    from qsiprep.utils.spaces import parse_output_spaces, select_acpc_anchor

    config.workflow.output_spaces = ['acpc:res-2mm', 'MNIInfant:cohort-3']
    specs = parse_output_spaces(config.workflow.output_spaces)
    anchor = select_acpc_anchor(specs)
    assert anchor.fullname == 'MNIInfant+3'

    # The config field diffprep.py used to read must be gone, so any surviving
    # reader is a build-time AttributeError rather than a silent None.
    assert not hasattr(config.workflow, 'anatomical_template')


def test_template_lps_wf_reorients_to_lps():
    from qsiprep.workflows.anatomical.volume import init_template_lps_wf

    wf = init_template_lps_wf()
    # AFNI spells LPS+ as RAI.
    assert wf.get_node('reorient_brain').inputs.orientation == 'RAI'
    assert wf.get_node('reorient_mask').inputs.orientation == 'RAI'
    assert wf.get_node('outputnode') is not None

    # Node attributes alone would pass even if the two reorients' sources were
    # swapped (masked brain into reorient_mask, raw mask into reorient_brain).
    # Task 13 reuses this sub-workflow for every standard space, so pin the
    # actual wiring, not just node presence.
    assert wf.get_node('mask_template').inputs.expr == 'a*b'
    edges = {(u.name, v.name): d['connect'] for u, v, d in wf._graph.edges(data=True)}

    # reorient_brain is fed from mask_template's masked output, not the raw template.
    assert ('mask_template', 'reorient_brain') in edges
    assert edges[('mask_template', 'reorient_brain')] == [('out_file', 'in_file')]
    assert ('inputnode', 'reorient_brain') not in edges

    # reorient_mask is fed straight from inputnode.mask_file, bypassing mask_template.
    assert ('inputnode', 'reorient_mask') in edges
    assert edges[('inputnode', 'reorient_mask')] == [('mask_file', 'in_file')]
    assert ('mask_template', 'reorient_mask') not in edges


def test_one_output_grid_per_acpc_resolution(tmp_path):
    from qsiprep.utils.spaces import parse_output_spaces, select_acpc_anchor
    from qsiprep.workflows.anatomical.volume import init_anat_preproc_wf

    config.workflow.output_spaces = ['acpc:res-2mm', 'acpc:res-1p5mm']
    config.workflow.anat_modality = 'T1w'
    config.workflow.infant = False
    config.nipype.omp_nthreads = 1
    config.execution.output_dir = str(tmp_path)
    specs = parse_output_spaces(config.workflow.output_spaces)
    acpc_specs = [s for s in specs if not s.standard]

    wf = init_anat_preproc_wf(
        num_anat_images=1,
        num_additional_t2ws=0,
        has_rois=False,
        output_spaces=specs,
        acpc_anchor=select_acpc_anchor(specs),
        acpc_specs=acpc_specs,
        do_biascorr=False,
        t2w_do_biascorr=False,
    )
    names = wf.list_node_names()
    assert any('output_grid_res2mm_wf' in n for n in names)
    assert any('output_grid_res1p5mm_wf' in n for n in names)


def _build_anat_preproc_wf(tmp_path, output_spaces, use_syn_sdc=False):
    from qsiprep.utils.spaces import parse_output_spaces, select_acpc_anchor
    from qsiprep.workflows.anatomical.volume import init_anat_preproc_wf

    config.workflow.output_spaces = output_spaces
    config.workflow.anat_modality = 'T1w'
    config.workflow.infant = False
    config.workflow.use_syn_sdc = use_syn_sdc
    config.nipype.omp_nthreads = 1
    config.execution.output_dir = str(tmp_path)
    specs = parse_output_spaces(config.workflow.output_spaces)
    acpc_specs = [s for s in specs if not s.standard]

    return init_anat_preproc_wf(
        num_anat_images=1,
        num_additional_t2ws=0,
        has_rois=False,
        output_spaces=specs,
        acpc_anchor=select_acpc_anchor(specs),
        acpc_specs=acpc_specs,
        do_biascorr=False,
        t2w_do_biascorr=False,
    )


def test_anchor_normalization_is_not_duplicated(tmp_path):
    # The default request's standard space IS the ACPC anchor -- normalize once.
    wf = _build_anat_preproc_wf(tmp_path, ['acpc:res-2mm', 'MNI152NLin2009cAsym'])
    norm_wfs = {n.split('.')[0] for n in wf.list_node_names() if 'anat_normalization' in n}
    assert len(norm_wfs) == 1, f'anchor normalization duplicated: {sorted(norm_wfs)}'


def test_distinct_standard_spaces_each_normalize(tmp_path):
    # A non-anchor space gets its own registration.
    wf = _build_anat_preproc_wf(
        tmp_path, ['acpc:res-2mm', 'MNI152NLin2009cAsym', 'MNI152NLin6Asym']
    )
    norm_wfs = {n.split('.')[0] for n in wf.list_node_names() if 'anat_normalization' in n}
    assert len(norm_wfs) == 2, f'expected 2 normalizations, got {sorted(norm_wfs)}'


def test_two_resolutions_of_one_template_build(tmp_path):
    """Rule 6 of the spec: one template may be asked for at several resolutions.

    Node names used to key on spec.fullname alone, so this raised
    ``OSError: Duplicate node name``.
    """
    wf = _build_anat_preproc_wf(tmp_path, ['acpc:res-2mm', 'MNI152NLin2009cAsym:res-1:res-2'])
    norm_wfs = {n.split('.')[0] for n in wf.list_node_names() if 'anat_normalization' in n}
    assert len(norm_wfs) == 3, sorted(norm_wfs)  # the anchor plus one per resolution


def test_two_resolutions_of_one_template_report_once(tmp_path):
    """Reportlet filenames have no res- entity, so one figure per template."""
    from qsiprep.utils.spaces import parse_output_spaces
    from qsiprep.workflows.anatomical.volume import init_anat_reports_wf

    config.workflow.anat_modality = 'T1w'
    config.execution.output_dir = str(tmp_path)
    specs = parse_output_spaces(['acpc:res-2mm', 'MNI152NLin2009cAsym:res-1:res-2'])
    wf = init_anat_reports_wf(output_spaces=specs)
    reports = [n for n in wf.list_node_names() if 'ds_report_t1_2_' in n]
    assert reports == ['ds_report_t1_2_MNI152NLin2009cAsymres1'], reports


def test_two_resolutions_of_one_template_write_one_transform(tmp_path):
    from qsiprep.utils.spaces import parse_output_spaces
    from qsiprep.workflows.anatomical.volume import init_anat_derivatives_wf

    config.workflow.anat_modality = 'T1w'
    config.execution.output_dir = str(tmp_path)
    specs = parse_output_spaces(['acpc:res-2mm', 'MNI152NLin2009cAsym:res-1:res-2'])
    wf = init_anat_derivatives_wf(output_spaces=specs)
    warps = sorted(n for n in wf.list_node_names() if n.endswith('_warp'))
    assert warps == ['ds_t1_MNI152NLin2009cAsymres1_inv_warp',
                     'ds_t1_MNI152NLin2009cAsymres1_warp'], warps


def test_no_standard_space_skips_the_nonlinear_normalization(tmp_path):
    """Nothing consumes the nonlinear transform, so antsRegistration must not run.

    This is what --skip-anat-based-spatial-normalization used to do; the flag is
    deprecated and no longer sets anything, so the space list has to decide.
    """
    wf = _build_anat_preproc_wf(tmp_path, ['acpc:res-2mm'])
    names = wf.list_node_names()
    assert not any('anat_nlin_normalization' in n for n in names), (
        'a nonlinear normalization was built with no standard space requested'
    )
    # The rigid AC-PC registration still has to happen.
    assert any(n.endswith('anat_normalization_wf.acpc_reg') for n in names)


def test_syn_sdc_keeps_the_nonlinear_normalization(tmp_path):
    """SyN-SDC pulls its atlas prior through t1_2_mni_reverse_transform."""
    wf = _build_anat_preproc_wf(tmp_path, ['acpc:res-2mm'], use_syn_sdc=True)
    assert any('anat_nlin_normalization' in n for n in wf.list_node_names())


def test_standard_space_keeps_the_nonlinear_normalization(tmp_path):
    wf = _build_anat_preproc_wf(tmp_path, ['acpc:res-2mm', 'MNI152NLin2009cAsym'])
    assert any('anat_nlin_normalization' in n for n in wf.list_node_names())


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
