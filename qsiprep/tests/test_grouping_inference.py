"""Scenario tests for qsiprep.grouping inference.

Each test materializes one of the ``skeleton_grouping_*.yml`` fixtures and
asserts the grouping decisions: which estimations exist (and their
provenance), which estimation corrects which file, how distortion and
concatenation groups partition, and which issues fire.
"""

import pytest

from qsiprep.grouping import GroupingError, Provenance, check_backend
from qsiprep.grouping.models import EstimationMethod
from qsiprep.tests.grouping_scenarios import basenames, load_scenario


def issue_codes(issues):
    return {issue.code for issue in issues}


def test_hcp_style(tmp_path):
    """2xAP + 2xPA, no fmap/, no curation: fully automatic."""
    grouping = load_scenario('hcp_style', tmp_path)

    assert list(grouping.estimations) == ['auto+pepolar+j']
    estimation = grouping.estimations['auto+pepolar+j']
    assert estimation.provenance is Provenance.INFERRED
    assert estimation.method is EstimationMethod.PEPOLAR
    assert len(estimation.sources) == 4
    assert estimation.bidirectional_axes == {'j'}

    # every series is corrected by the single estimation
    assert set(grouping.application.values()) == {'auto+pepolar+j'}

    assert len(grouping.distortion_groups) == 2
    assert sorted(grouping.distortion_groups) == ['sub-01_dir-AP', 'sub-01_dir-PA']

    (concat,) = grouping.concatenation_groups.values()
    assert concat.output_name == 'sub-01'
    assert concat.provenance is Provenance.INFERRED
    assert len(concat.dwi_files) == 4
    assert not grouping.issues


def test_hcp_style_separate_all_dwis(tmp_path):
    """separate_all_dwis: four outputs, but SDC is preserved."""
    grouping = load_scenario('hcp_style', tmp_path, separate_all_dwis=True)

    assert len(grouping.concatenation_groups) == 4
    # The estimation still exists and still corrects every series
    assert set(grouping.application.values()) == {'auto+pepolar+j'}
    # Each singleton output borrows the reverse-PE series for estimation
    for multipart_id in grouping.concatenation_groups:
        assert grouping.borrowed_sources(multipart_id)


def test_abcd_style(tmp_path):
    """Single AP DWI + reverse-PE epi fmap via IntendedFor."""
    grouping = load_scenario('abcd_style', tmp_path)

    (b0field_id,) = grouping.estimations
    estimation = grouping.estimations[b0field_id]
    assert estimation.provenance is Provenance.TRANSLATED
    assert estimation.method is EstimationMethod.PEPOLAR
    # The epi fmap AND the target DWI are both sources
    assert sorted(basenames(estimation.sources)) == [
        'sub-01_dir-AP_dwi.nii.gz',
        'sub-01_dir-PA_epi.nii.gz',
    ]
    assert estimation.bidirectional_axes == {'j'}

    (concat,) = grouping.concatenation_groups.values()
    assert concat.output_name == 'sub-01_dir-AP'
    # The fmap file is never an output member
    assert basenames(concat.dwi_files) == ['sub-01_dir-AP_dwi.nii.gz']
    assert not grouping.errors


def test_abcd_style_ignore_fieldmaps(tmp_path):
    """--ignore fieldmaps: the fmap is not even indexed; no estimation."""
    grouping = load_scenario('abcd_style', tmp_path, ignore_fieldmaps=True)
    assert not grouping.estimations
    assert set(grouping.application.values()) == {None}
    assert all(rec.datatype == 'dwi' for rec in grouping.files.values())


def test_bidsuri_intendedfor(tmp_path):
    """bids:: URIs resolve identically to relative paths."""
    grouping = load_scenario('bidsuri_intendedfor', tmp_path)
    (estimation,) = grouping.estimations.values()
    assert estimation.provenance is Provenance.TRANSLATED
    assert sorted(basenames(estimation.sources)) == [
        'sub-01_dir-AP_dwi.nii.gz',
        'sub-01_dir-PA_epi.nii.gz',
    ]
    dwi_path = grouping.dwi_files[0]
    assert grouping.application[dwi_path] == estimation.b0field_id


def test_curated_b0field(tmp_path):
    """Fully curated B0FieldIdentifier/B0FieldSource: no inference at all."""
    grouping = load_scenario('curated_b0field', tmp_path)

    assert list(grouping.estimations) == ['pepolar']
    estimation = grouping.estimations['pepolar']
    assert estimation.provenance is Provenance.CURATED
    assert estimation.method is EstimationMethod.PEPOLAR

    dwi_path = grouping.dwi_files[0]
    assert grouping.application[dwi_path] == 'pepolar'
    assert grouping.application_provenance[dwi_path] is Provenance.CURATED
    assert not grouping.issues


def test_cluster_multipart(tmp_path):
    """The writeup's clusterA/B/C scenario with curated MultipartID.

    The canonical estimation != concatenation case: clusterC has no reverse
    partner of its own but borrows b=0s from the session for estimation.
    """
    grouping = load_scenario('cluster_multipart', tmp_path)

    # One inferred axis estimation spans all seven series
    assert list(grouping.estimations) == ['auto+pepolar+j']
    assert len(grouping.estimations['auto+pepolar+j'].sources) == 7

    # Three curated outputs
    outputs = {
        concat.multipart_id: concat.output_name
        for concat in grouping.concatenation_groups.values()
    }
    assert outputs == {
        'clusterA': 'sub-01_acq-A',
        'clusterB': 'sub-01_acq-B',
        'clusterC': 'sub-01_acq-C_dir-AP',
    }
    for concat in grouping.concatenation_groups.values():
        assert concat.provenance is Provenance.CURATED

    # clusterC borrows the rest of the session for its estimation
    borrowed = grouping.borrowed_sources('clusterC')
    assert len(borrowed['auto+pepolar+j']) == 6

    # 5 distortion groups: A-AP, A-PA, B-AP(x2), B-PA(x2), C-AP
    assert len(grouping.distortion_groups) == 5

    # The estimation spans all three outputs
    assert 'estimation-spans-outputs' in issue_codes(grouping.warnings)

    # fsl can use all four signatures; DRBUDDI cannot take more than two
    assert not [i for i in check_backend(grouping, 'fsl') if i.severity == 'error']
    tortoise_codes = issue_codes(check_backend(grouping, 'tortoise'))
    assert 'drbuddi-too-many-signatures' in tortoise_codes


def test_cluster_nomultipart(tmp_path):
    """Without MultipartID everything sharing the estimation concatenates."""
    grouping = load_scenario('cluster_nomultipart', tmp_path)

    (concat,) = grouping.concatenation_groups.values()
    assert concat.output_name == 'sub-01'
    assert len(concat.dwi_files) == 7
    # acq-A AP and acq-C AP share signature+estimation: one distortion group
    assert len(grouping.distortion_groups) == 4


def test_reshim_blocks_borrowing(tmp_path):
    """A re-shimmed series cannot borrow across the shim boundary."""
    grouping = load_scenario('reshim', tmp_path)

    # Only the shim-matched pair forms an estimation
    (b0field_id,) = grouping.estimations
    estimation = grouping.estimations[b0field_id]
    assert sorted(basenames(estimation.sources)) == [
        'sub-01_acq-A_dir-AP_dwi.nii.gz',
        'sub-01_acq-A_dir-PA_dwi.nii.gz',
    ]

    # acq-B is uncorrected and separate
    outputs = {
        concat.output_name: sorted(basenames(concat.dwi_files))
        for concat in grouping.concatenation_groups.values()
    }
    assert outputs == {
        'sub-01_acq-A': [
            'sub-01_acq-A_dir-AP_dwi.nii.gz',
            'sub-01_acq-A_dir-PA_dwi.nii.gz',
        ],
        'sub-01_acq-B_dir-AP': ['sub-01_acq-B_dir-AP_dwi.nii.gz'],
    }
    codes = issue_codes(grouping.warnings)
    assert 'session-multiple-shims' in codes

    # The uncorrected output draws a no-sdc warning on every backend
    for backend in ('fsl', 'tortoise', 'mixed'):
        assert 'no-sdc' in issue_codes(check_backend(grouping, backend))


def test_reshim_ignored(tmp_path):
    """ignore_shims lets the re-shimmed series join the estimation."""
    grouping = load_scenario('reshim', tmp_path, ignore_shims=True)

    (b0field_id,) = grouping.estimations
    assert len(grouping.estimations[b0field_id].sources) == 3
    assert set(grouping.application.values()) == {b0field_id}

    (concat,) = grouping.concatenation_groups.values()
    assert concat.output_name == 'sub-01'
    assert 'shims-ignored' in issue_codes(grouping.warnings)


def test_partial_curation(tmp_path):
    """Curated series keep their curation; the rest are inferred; warned."""
    grouping = load_scenario('partial_curation', tmp_path)

    assert set(grouping.estimations) == {'pepolar01', 'auto+pepolar+j'}
    provenance_by_run = {
        path: grouping.application_provenance[path] for path in grouping.dwi_files
    }
    curated = [path for path, prov in provenance_by_run.items() if prov is Provenance.CURATED]
    inferred = [path for path, prov in provenance_by_run.items() if prov is Provenance.INFERRED]
    assert len(curated) == 2
    assert len(inferred) == 2
    assert all('run-1' in path for path in curated)
    assert all('run-2' in path for path in inferred)
    assert 'mixed-application-provenance' in issue_codes(grouping.warnings)

    # Same signature, different fieldmaps: run-1 and run-2 AP stay in
    # different distortion groups but share the single output.
    assert len(grouping.distortion_groups) == 4
    assert len(grouping.concatenation_groups) == 1


def test_cross_axis_b0field(tmp_path):
    """A curated identifier spanning axes works for TOPUP, not DRBUDDI."""
    grouping = load_scenario('cross_axis_b0field', tmp_path)

    estimation = grouping.estimations['topupall']
    assert estimation.pe_axes == {'i', 'j'}
    assert estimation.bidirectional_axes == {'i', 'j'}
    assert not grouping.errors

    assert not [i for i in check_backend(grouping, 'fsl') if i.severity == 'error']
    for backend in ('tortoise', 'mixed'):
        assert 'drbuddi-cross-axis' in issue_codes(check_backend(grouping, backend))


def test_multipart_splits_estimation(tmp_path):
    """MultipartID splitting an estimation's targets is legal borrowing."""
    grouping = load_scenario('multipart_splits_estimation', tmp_path)

    assert len(grouping.concatenation_groups) == 2
    assert list(grouping.estimations) == ['auto+pepolar+j']
    assert 'estimation-spans-outputs' in issue_codes(grouping.warnings)
    # Each output borrows the other half's series for estimation
    for multipart_id in ('part1', 'part2'):
        borrowed = grouping.borrowed_sources(multipart_id)
        assert len(borrowed['auto+pepolar+j']) == 2


def test_gre_phasediff(tmp_path):
    """phasediff + magnitudes via IntendedFor: a TRANSLATED GRE estimation."""
    grouping = load_scenario('gre_phasediff', tmp_path)

    (estimation,) = grouping.estimations.values()
    assert estimation.method is EstimationMethod.PHASEDIFF
    assert estimation.provenance is Provenance.TRANSLATED
    # The DWI does not participate in a GRE estimation
    assert sorted(basenames(estimation.sources)) == [
        'sub-01_magnitude1.nii.gz',
        'sub-01_magnitude2.nii.gz',
        'sub-01_phasediff.nii.gz',
    ]
    dwi_path = grouping.dwi_files[0]
    assert grouping.application[dwi_path] == estimation.b0field_id

    # GRE routes are fine on fsl/tortoise; mixed warns that DRBUDDI adds nothing
    assert not check_backend(grouping, 'fsl')
    assert 'mixed-non-pepolar' in issue_codes(check_backend(grouping, 'mixed'))


def test_two_gre_fmaps(tmp_path):
    """Independently corrected runs concatenate (rule b), loudly."""
    grouping = load_scenario('two_gre_fmaps', tmp_path)

    assert len(grouping.estimations) == 2
    assert all(est.method is EstimationMethod.PHASEDIFF for est in grouping.estimations.values())
    (concat,) = grouping.concatenation_groups.values()
    assert concat.output_name == 'sub-01_dir-AP'
    assert len(concat.distortion_groups) == 2
    assert 'inferred-concat-merge' in issue_codes(grouping.warnings)


def test_same_ped_own_fmaps(tmp_path):
    """Same-PED runs with per-run fmaps: separate fieldmaps, one output."""
    grouping = load_scenario('same_ped_own_fmaps', tmp_path)

    assert len(grouping.estimations) == 2
    applications = {grouping.application[path] for path in grouping.dwi_files}
    assert len(applications) == 2  # each run has its own estimation

    (concat,) = grouping.concatenation_groups.values()
    assert concat.output_name == 'sub-01_dir-AP'
    # Identical signatures merge quietly - this is not a rule-b merge
    assert 'inferred-concat-merge' not in issue_codes(grouping.warnings)
    assert len(concat.distortion_groups) == 2


def test_unlinked_fmap(tmp_path):
    """An unlinked epi fmap is unused; the DWI heuristic still applies."""
    grouping = load_scenario('unlinked_fmap', tmp_path)

    assert 'unlinked-fmap' in issue_codes(grouping.warnings)
    (b0field_id,) = grouping.estimations
    estimation = grouping.estimations[b0field_id]
    assert estimation.provenance is Provenance.INFERRED
    # Sources are the two DWI series only - the fmap is not pulled in
    assert all(name.endswith('_dwi.nii.gz') for name in basenames(estimation.sources))


def test_b0only_fmap_with_bvals(tmp_path):
    """A dwi-like epi fmap (with bval/bvec) is a source, never an output."""
    grouping = load_scenario('b0only_fmap_with_bvals', tmp_path)

    (estimation,) = grouping.estimations.values()
    assert 'sub-01_dir-PA_epi.nii.gz' in basenames(estimation.sources)
    for concat in grouping.concatenation_groups.values():
        assert 'sub-01_dir-PA_epi.nii.gz' not in basenames(concat.dwi_files)


def test_missing_pedir(tmp_path):
    """A series without PhaseEncodingDirection is an uncorrected singleton."""
    grouping = load_scenario('missing_pedir', tmp_path)

    assert 'missing-pedir' in issue_codes(grouping.warnings)
    assert not grouping.estimations
    assert set(grouping.application.values()) == {None}
    assert len(grouping.concatenation_groups) == 2


def test_mixed_trt(tmp_path):
    """Opposing PEDs with different readout times share one estimation."""
    grouping = load_scenario('mixed_trt', tmp_path)

    (estimation,) = grouping.estimations.values()
    assert estimation.bidirectional_axes == {'j'}
    assert len(grouping.distortion_groups) == 2
    (concat,) = grouping.concatenation_groups.values()
    assert concat.output_name == 'sub-01'
    # Exactly two signatures: fine for TOPUP (two acqp rows) and DRBUDDI
    for backend in ('fsl', 'tortoise', 'mixed'):
        assert not [i for i in check_backend(grouping, backend) if i.severity == 'error']


def test_multi_session(tmp_path):
    """Grouping never crosses sessions."""
    grouping = load_scenario('multi_session', tmp_path)

    assert sorted(grouping.estimations) == [
        'auto+pepolar+ses-1+j',
        'auto+pepolar+ses-2+j',
    ]
    outputs = sorted(concat.output_name for concat in grouping.concatenation_groups.values())
    assert outputs == ['sub-01_ses-1', 'sub-01_ses-2']
    for concat in grouping.concatenation_groups.values():
        sessions = {grouping.files[path].session for path in concat.dwi_files}
        assert len(sessions) == 1


def test_name_collision(tmp_path):
    """Two outputs deriving the same BIDS name is a hard error."""
    with pytest.raises(GroupingError, match='output-name-collision'):
        load_scenario('name_collision', tmp_path)

    grouping = load_scenario('name_collision', tmp_path / 'nonstrict', strict=False)
    assert 'output-name-collision' in issue_codes(grouping.errors)
