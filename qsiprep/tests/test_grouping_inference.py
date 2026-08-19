"""Scenario tests for qsiprep.grouping inference.

Each test materializes one of the ``skeleton_grouping_*.yml`` fixtures and
asserts the grouping decisions: which estimations exist (and their
provenance), which estimation corrects which file, how distortion and
concatenation groups partition, and which issues fire.
"""

import dataclasses
import os.path as op

import pytest

from qsiprep.grouping import GroupingError, Provenance, check_backend
from qsiprep.grouping.inference import build_grouping
from qsiprep.grouping.models import CorrectionMethod
from qsiprep.tests.grouping_scenarios import basenames, load_scenario


def issue_codes(issues):
    return {issue.code for issue in issues}


def test_hcp_style(tmp_path):
    """2xAP + 2xPA, no fmap/, no curation: fully automatic."""
    grouping = load_scenario('hcp_style', tmp_path)

    assert list(grouping.estimations) == ['auto+pepolar+j']
    estimation = grouping.estimations['auto+pepolar+j']
    assert estimation.provenance is Provenance.INFERRED
    assert estimation.method is CorrectionMethod.PEPOLAR
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
    assert estimation.method is CorrectionMethod.PEPOLAR
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
    assert not any(rec.datatype == 'fmap' for rec in grouping.files.values())


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
    assert estimation.method is CorrectionMethod.PEPOLAR

    dwi_path = grouping.dwi_files[0]
    assert grouping.application[dwi_path] == 'pepolar'
    assert grouping.application_provenance[dwi_path] is Provenance.CURATED
    assert not grouping.issues


def test_intendedfor_superseded(tmp_path):
    """A fmap with both B0FieldIdentifier and IntendedFor uses B0Field only."""
    grouping = load_scenario('intendedfor_superseded', tmp_path)

    assert 'intendedfor-superseded' in issue_codes(grouping.warnings)

    by_name = {op.basename(path): path for path in grouping.dwi_files}
    run1 = by_name['sub-01_dir-AP_run-1_dwi.nii.gz']
    run2 = by_name['sub-01_dir-AP_run-2_dwi.nii.gz']
    # run-1 is corrected via B0FieldSource; run-2 (only the ignored IntendedFor
    # pointed at it) gets no fieldmap.
    assert grouping.application[run1] == 'pepolar01'
    assert grouping.application[run2] is None


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

    # fsl pools all four signatures; DRBUDDI takes each matched blip pair (one per
    # readout time, 0.05 and 0.08) on its own, so this is feasible on both paths.
    for backend in ('fsl', 'tortoise'):
        assert not [i for i in check_backend(grouping, backend) if i.severity == 'error']


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


def test_cross_axis_unpaired(tmp_path):
    """i- and j series with no same-axis partner still pool: any two
    differing phase encodings jointly determine the field. TOPUP consumes
    the estimation; DRBUDDI errors."""
    grouping = load_scenario('cross_axis_unpaired', tmp_path)

    (b0field_id,) = grouping.estimations
    assert b0field_id == 'auto+pepolar+ij'
    estimation = grouping.estimations[b0field_id]
    assert estimation.provenance is Provenance.INFERRED
    assert estimation.method is CorrectionMethod.PEPOLAR
    assert estimation.pe_axes == {'i', 'j'}
    assert estimation.bidirectional_axes == frozenset()

    # Both series are corrected by the pooled estimation, in one output.
    assert set(grouping.application.values()) == {b0field_id}
    (concat,) = grouping.concatenation_groups.values()
    assert concat.output_name == 'sub-01'

    # Backend feasibility is check_backend's call, not the model's.
    assert not [i for i in check_backend(grouping, 'fsl') if i.severity == 'error']
    assert 'drbuddi-no-opposing-pair' in issue_codes(check_backend(grouping, 'tortoise'))


def test_partial_curation(tmp_path):
    """Curation anywhere in a session disables the heuristic for the rest of
    it: the uncurated run-2 pair is NOT paired with itself."""
    grouping = load_scenario('partial_curation', tmp_path)

    assert set(grouping.estimations) == {'pepolar01'}
    for path in grouping.dwi_files:
        if 'run-1' in path:
            assert grouping.application[path] == 'pepolar01'
            assert grouping.application_provenance[path] is Provenance.CURATED
        else:
            assert grouping.application[path] is None
    assert 'reverse-pe-not-inferred' in issue_codes(grouping.warnings)

    # Corrected and uncorrected series never share an output: the curated
    # pair concatenates; each uncorrected run-2 series stands alone.
    outputs = sorted(concat.output_name for concat in grouping.concatenation_groups.values())
    assert outputs == ['sub-01_dir-AP_run-2', 'sub-01_dir-PA_run-2', 'sub-01_run-1']
    assert len(grouping.distortion_groups) == 4


def test_partial_curation_stranded(tmp_path):
    """An uncurated series in a curated session gets no inferred PEPOLAR;
    the fieldmap-less ladder (user-controllable at the CLI) still applies."""
    grouping = load_scenario('partial_curation_stranded', tmp_path)

    assert not [
        estimation
        for estimation in grouping.estimations.values()
        if estimation.provenance is Provenance.INFERRED and estimation.is_pepolar
    ]
    assert 'reverse-pe-not-inferred' in issue_codes(grouping.warnings)

    # The unlinked series is corrected by the inferred T2Wreg fallback, in
    # its own correction unit; both units are corrected, so their corrected
    # results are concatenated into ONE final output.
    (run2,) = [path for path in grouping.dwi_files if 'run-2' in path]
    applied = grouping.application[run2]
    assert grouping.estimations[applied].method is CorrectionMethod.T2WREG
    (concat,) = grouping.concatenation_groups.values()
    assert concat.output_name == 'sub-01'
    assert len(concat.correction_units) == 2


def test_partial_intendedfor(tmp_path):
    """IntendedFor counts as curation: it disables the heuristic for the
    session's remaining series exactly like B0Field* metadata does."""
    grouping = load_scenario('partial_intendedfor', tmp_path)

    # The linked AP series is corrected by the translated estimation; no
    # inferred PEPOLAR estimation exists for the unlinked PA series.
    assert not [
        estimation
        for estimation in grouping.estimations.values()
        if estimation.provenance is Provenance.INFERRED and estimation.is_pepolar
    ]
    (ap,) = [path for path in grouping.dwi_files if 'dir-AP' in path]
    (pa,) = [path for path in grouping.dwi_files if 'dir-PA' in path]
    applied = grouping.application[ap]
    assert grouping.estimations[applied].provenance is Provenance.TRANSLATED
    assert grouping.application[pa] is None
    assert 'reverse-pe-not-inferred' in issue_codes(grouping.warnings)


def test_partial_multipart(tmp_path):
    """Series without a MultipartID are not combined when other series have
    one: each becomes its own output, with a warning."""
    grouping = load_scenario('partial_multipart', tmp_path)

    outputs = {
        concat.output_name: basenames(concat.dwi_files)
        for concat in grouping.concatenation_groups.values()
    }
    # The uncurated run-2 pair shares one correction unit (one estimation
    # corrects both), so that unit becomes one standalone output.
    assert outputs == {
        'sub-01_run-1': [
            'sub-01_dir-AP_run-1_dwi.nii.gz',
            'sub-01_dir-PA_run-1_dwi.nii.gz',
        ],
        'sub-01_run-2': [
            'sub-01_dir-AP_run-2_dwi.nii.gz',
            'sub-01_dir-PA_run-2_dwi.nii.gz',
        ],
    }
    assert 'partial-multipart' in issue_codes(grouping.warnings)

    # With no B0Field curation anywhere, all four series still share the
    # single inferred estimation (concatenation and estimation membership
    # are independent).
    assert set(grouping.application.values()) == {'auto+pepolar+j'}


def test_cross_axis_b0field(tmp_path):
    """A curated identifier spanning axes works for every backend when each
    axis is its own opposing pair: TOPUP pools all four directions, and DRBUDDI
    (tortoise / mixed) corrects one axis at a time and recombines."""
    grouping = load_scenario('cross_axis_b0field', tmp_path)

    estimation = grouping.estimations['topupall']
    assert estimation.pe_axes == {'i', 'j'}
    assert estimation.bidirectional_axes == {'i', 'j'}
    assert not grouping.errors

    for backend in ('fsl', 'tortoise', 'mixed'):
        issues = check_backend(grouping, backend)
        assert 'drbuddi-cross-axis' not in issue_codes(issues)
        assert not [i for i in issues if i.severity == 'error']


def test_partial_pair_fallback(tmp_path):
    """A matched blip pair plus an unmatched singleton, pooled in one estimation.

    DRBUDDI corrects the pair; the singleton has no opposing blip, so on the
    TORTOISE path it falls back to T2Wreg (a T2w is present) - a warning, not an
    error, so nothing aborts. The mixed path corrects the singleton with
    TOPUP+eddy and does not flag it."""
    grouping = load_scenario('partial_pair', tmp_path)

    for backend in ('fsl', 'tortoise', 'mixed'):
        assert not [i for i in check_backend(grouping, backend) if i.severity == 'error']

    tortoise = check_backend(grouping, 'tortoise')
    unpaired = [i for i in tortoise if i.code == 'drbuddi-no-opposing-pair']
    assert unpaired
    assert all(i.severity == 'warning' for i in unpaired)
    # On the mixed path the singleton is corrected by TOPUP+eddy; the multi-group
    # unit just gets single-stage (the single-pass DRBUDDI refinement is skipped).
    mixed = check_backend(grouping, 'mixed')
    assert 'drbuddi-no-opposing-pair' not in issue_codes(mixed)
    assert 'drbuddi-refinement-multigroup' in issue_codes(mixed)


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
    assert estimation.method is CorrectionMethod.PHASEDIFF
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
    """Runs with their own fieldmaps are separate correction units; their
    corrected results are concatenated into one final output."""
    grouping = load_scenario('two_gre_fmaps', tmp_path)

    assert len(grouping.estimations) == 2
    assert all(est.method is CorrectionMethod.PHASEDIFF for est in grouping.estimations.values())
    (concat,) = grouping.concatenation_groups.values()
    assert concat.output_name == 'sub-01_dir-AP'
    units = grouping.correction_units_in(concat.multipart_id)
    assert len(units) == 2
    assert {unit.b0field_source for unit in units} == set(grouping.estimations)


def test_same_ped_own_fmaps(tmp_path):
    """Same-PED runs with per-run fmaps: separate correction units (never
    blend the fields), one final output of the corrected results."""
    grouping = load_scenario('same_ped_own_fmaps', tmp_path)

    assert len(grouping.estimations) == 2
    applications = {grouping.application[path] for path in grouping.dwi_files}
    assert len(applications) == 2  # each run has its own estimation

    (concat,) = grouping.concatenation_groups.values()
    assert concat.output_name == 'sub-01_dir-AP'
    assert len(concat.correction_units) == 2


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
    # TOPUP pools the two readout times (two acqp rows). DRBUDDI needs the readout
    # time matched within a blip pair, which these opposing PEs lack - but nothing
    # aborts: the TORTOISE path falls the series back to T2Wreg/HMC-only with a
    # warning, and the mixed path corrects via TOPUP+eddy.
    for backend in ('fsl', 'tortoise', 'mixed'):
        assert not [i for i in check_backend(grouping, backend) if i.severity == 'error']
    assert 'drbuddi-no-opposing-pair' in issue_codes(check_backend(grouping, 'tortoise'))


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


def test_acq_multipartid_names_output(tmp_path):
    """A MultipartID beginning with 'acq-' renames the output's acq- entity,
    so the same layout that collides in name_collision is fine here."""
    grouping = load_scenario('acq_multipart', tmp_path)

    outputs = {
        concat.multipart_id: concat.output_name
        for concat in grouping.concatenation_groups.values()
    }
    assert outputs == {
        'acq-partA': 'sub-01_acq-partA_dir-AP',
        'acq-partB': 'sub-01_acq-partB_dir-AP',
    }
    assert 'output-name-collision' not in issue_codes(grouping.errors)


def test_acq_multipartid_invalid_label(tmp_path):
    """An 'acq-' MultipartID whose label is not a valid BIDS label errors."""
    grouping = load_scenario('acq_multipart', tmp_path, strict=False)
    records = [record for record in grouping.files.values() if record.is_dwi]
    bad = dataclasses.replace(records[0], multipart_id='acq-part_A')
    regrouped = build_grouping([bad, *records[1:]], subject_id='01')
    assert 'multipartid-acq-invalid' in issue_codes(regrouped.errors)


def test_name_collision(tmp_path):
    """Colliding output names are a hard error; the fix is 'acq-' MultipartIDs."""
    with pytest.raises(GroupingError, match='output-name-collision'):
        load_scenario('name_collision', tmp_path)

    grouping = load_scenario('name_collision', tmp_path / 'nonstrict', strict=False)
    assert 'output-name-collision' in issue_codes(grouping.errors)


def test_per_axis_curated_fmaps_merge_after_correction(tmp_path):
    """Per-axis curated estimations form separate units whose corrected
    results package into one final output (formerly a name collision)."""
    grouping = load_scenario('name_collision_inferred', tmp_path)

    (concat,) = grouping.concatenation_groups.values()
    assert concat.output_name == 'sub-01'
    assert len(concat.correction_units) == 2
    assert 'output-name-collision' not in issue_codes(grouping.errors)


def test_fieldmapless_t2w(tmp_path):
    """No fieldmap + a T2w: the inferred T2Wreg fallback applies."""
    grouping = load_scenario('fieldmapless_t2w', tmp_path)

    assert list(grouping.estimations) == ['auto+t2wreg']
    estimation = grouping.estimations['auto+t2wreg']
    assert estimation.method is CorrectionMethod.T2WREG
    assert estimation.provenance is Provenance.INFERRED
    assert basenames(estimation.sources) == ['sub-01_T2w.nii.gz']

    dwi_path = grouping.dwi_files[0]
    assert grouping.application[dwi_path] == 'auto+t2wreg'
    assert grouping.application_provenance[dwi_path] is Provenance.INFERRED

    # tortoise executes T2Wreg; fsl/mixed only warn (nobody demanded it)
    assert not check_backend(grouping, 'tortoise')
    for backend in ('fsl', 'mixed'):
        anat_issues = [
            issue
            for issue in check_backend(grouping, backend)
            if issue.code == 'anat-sdc-unsupported'
        ]
        assert anat_issues
        assert anat_issues[0].severity == 'warning'


def test_fieldmapless_t1w_only(tmp_path):
    """No fieldmap and no T2w: genuinely uncorrectable by default."""
    grouping = load_scenario('fieldmapless_t1w_only', tmp_path)

    assert not grouping.estimations
    assert set(grouping.application.values()) == {None}
    for backend in ('fsl', 'tortoise', 'mixed'):
        assert 'no-sdc' in issue_codes(check_backend(grouping, backend))


def test_fieldmapless_t1w_only_synb0(tmp_path):
    """use_synb0 corrects the same data via a synthetic b=0 from the T1w."""
    grouping = load_scenario('fieldmapless_t1w_only', tmp_path, use_synb0=True)

    assert list(grouping.estimations) == ['auto+synb0']
    estimation = grouping.estimations['auto+synb0']
    assert estimation.method is CorrectionMethod.SYNB0
    assert estimation.provenance is Provenance.FORCED
    assert basenames(estimation.sources) == ['sub-01_T1w.nii.gz']

    dwi_path = grouping.dwi_files[0]
    assert grouping.application[dwi_path] == 'auto+synb0'

    # The synthetic b=0 is a target every backend can consume: TOPUP's missing
    # blip (fsl/mixed) or the T2Wreg registration target (tortoise/mixed)
    for backend in ('fsl', 'tortoise', 'mixed'):
        assert not check_backend(grouping, backend)


def test_fieldmapless_t1w_only_syn(tmp_path):
    """use_nipreps_syn_sdc corrects the same data via a classic ANTs SyN registration."""
    grouping = load_scenario('fieldmapless_t1w_only', tmp_path, use_nipreps_syn_sdc=True)

    assert list(grouping.estimations) == ['auto+syn']
    estimation = grouping.estimations['auto+syn']
    assert estimation.method is CorrectionMethod.NIPREPS_SYN
    assert estimation.provenance is Provenance.FORCED
    assert basenames(estimation.sources) == ['sub-01_T1w.nii.gz']

    dwi_path = grouping.dwi_files[0]
    assert grouping.application[dwi_path] == 'auto+syn'

    # SyN routes through init_sdc_wf on every backend: feasible everywhere (the
    # mixed path only warns that DRBUDDI has nothing to refine).
    for backend in ('fsl', 'tortoise', 'mixed'):
        assert not [i for i in check_backend(grouping, backend) if i.severity == 'error']


def test_syn_never_overrides_a_real_fieldmap(tmp_path):
    """use_nipreps_syn_sdc is a fallback: a series with a fieldmap keeps it."""
    grouping = load_scenario('gre_phasediff', tmp_path, use_nipreps_syn_sdc=True)

    methods = {est.method for est in grouping.estimations.values()}
    assert CorrectionMethod.NIPREPS_SYN not in methods
    assert CorrectionMethod.PHASEDIFF in methods


def test_ignore_sdc_disables_all_correction(tmp_path):
    """``ignore_sdc`` leaves every series uncorrected -- fieldmaps AND reverse-PE.

    ``mixed_trt`` is an opposite-PE pair that normally gets a PEPOLAR estimation
    from the reverse-PE heuristic, so a lack of estimations proves the heuristic
    (not just fmap/ indexing) is off. Opposite-PE series stay separate outputs,
    since merging them without SDC would stack opposing distortions.
    """
    grouping = load_scenario('mixed_trt', tmp_path, ignore_sdc=True)

    assert not grouping.estimations
    assert set(grouping.application.values()) == {None}
    output_names = sorted(c.output_name for c in grouping.concatenation_groups.values())
    assert output_names == ['sub-01_dir-AP', 'sub-01_dir-PA']


def test_syn_missing_pedir(tmp_path):
    """SyN on a series without PhaseEncodingDirection is a hard error."""
    with pytest.raises(GroupingError, match='syn-missing-pedir'):
        load_scenario('missing_pedir', tmp_path, use_nipreps_syn_sdc=True)

    grouping = load_scenario(
        'missing_pedir', tmp_path / 'nonstrict', use_nipreps_syn_sdc=True, strict=False
    )
    assert 'syn-missing-pedir' in issue_codes(grouping.errors)
    # The series that does have PE info still gets its SyN estimation.
    assert 'auto+syn' in grouping.estimations


def test_t2w_hcp_pepolar_wins(tmp_path):
    """A real PEPOLAR pair always beats the fieldmap-less fallback."""
    grouping = load_scenario('t2w_hcp', tmp_path)

    assert list(grouping.estimations) == ['auto+pepolar+j']
    assert set(grouping.application.values()) == {'auto+pepolar+j'}
    # The T2w is still indexed (DRBUDDI can use it as a structural target)
    assert grouping.anat_files('T2w')


def test_t2w_hcp_force_t2wreg(tmp_path):
    """force_t2wreg overrides the PEPOLAR pairing for every series."""
    grouping = load_scenario('t2w_hcp', tmp_path, force_t2wreg=True)

    estimation = grouping.estimations['auto+t2wreg']
    assert estimation.provenance is Provenance.FORCED
    assert set(grouping.application.values()) == {'auto+t2wreg'}
    assert all(prov is Provenance.FORCED for prov in grouping.application_provenance.values())
    # The losing PEPOLAR estimation stays visible (unapplied) so the report's
    # "(also eligible: ...)" line resolves to something the reader can look up
    assert 'auto+pepolar+j' in grouping.estimations

    # Demanded-but-unsupported is an error on fsl/mixed
    for backend in ('fsl', 'mixed'):
        anat_issues = [
            issue
            for issue in check_backend(grouping, backend)
            if issue.code == 'anat-sdc-unsupported'
        ]
        assert anat_issues
        assert anat_issues[0].severity == 'error'
    assert not check_backend(grouping, 'tortoise')


def test_force_t2wreg_requires_t2w(tmp_path):
    """Forcing T2Wreg without a T2w is a hard error."""
    with pytest.raises(GroupingError, match='t2wreg-requires-t2w'):
        load_scenario('hcp_style', tmp_path, force_t2wreg=True)


def test_curated_t2wreg(tmp_path):
    """A T2w B0FieldIdentifier named by a DWI's B0FieldSource: curated T2Wreg."""
    grouping = load_scenario('curated_t2wreg', tmp_path)

    assert list(grouping.estimations) == ['anatreg']
    estimation = grouping.estimations['anatreg']
    assert estimation.method is CorrectionMethod.T2WREG
    assert estimation.provenance is Provenance.CURATED

    dwi_path = grouping.dwi_files[0]
    assert grouping.application[dwi_path] == 'anatreg'
    assert grouping.application_provenance[dwi_path] is Provenance.CURATED

    assert not check_backend(grouping, 'tortoise')
    anat_issues = [
        issue for issue in check_backend(grouping, 'fsl') if issue.code == 'anat-sdc-unsupported'
    ]
    assert anat_issues
    assert anat_issues[0].severity == 'error'


def test_synb0_missing_pedir(tmp_path):
    """SyNb0 on a series without PhaseEncodingDirection is a hard error."""
    with pytest.raises(GroupingError, match='synb0-missing-pedir'):
        load_scenario('missing_pedir', tmp_path, use_synb0=True)

    grouping = load_scenario('missing_pedir', tmp_path / 'nonstrict', use_synb0=True, strict=False)
    assert 'synb0-missing-pedir' in issue_codes(grouping.errors)
    # The series that does have PE info still gets its SyNb0 estimation
    assert 'auto+synb0' in grouping.estimations


def test_shell_mix(tmp_path):
    """Shelled + non-shelled series in one output: eddy errors, TORTOISE warns."""
    grouping = load_scenario('shell_mix', tmp_path)

    assert set(basenames(grouping.dwi_files)) == {
        'sub-01_dir-AP_dwi.nii.gz',
        'sub-01_dir-PA_dwi.nii.gz',
    }
    shelled_states = {
        grouping.files[path].filename: grouping.files[path].shelled for path in grouping.dwi_files
    }
    assert shelled_states == {
        'sub-01_dir-AP_dwi.nii.gz': True,
        'sub-01_dir-PA_dwi.nii.gz': False,
    }

    # One PEPOLAR estimation, one output holding both series
    (concat,) = grouping.concatenation_groups.values()
    assert len(concat.dwi_files) == 2

    for backend in ('fsl', 'mixed'):
        codes = issue_codes(check_backend(grouping, backend))
        assert 'eddy-requires-shelled' in codes
    tortoise_issues = check_backend(grouping, 'tortoise')
    assert 'mixed-shelled-nonshelled' in issue_codes(tortoise_issues)
    assert all(issue.severity == 'warning' for issue in tortoise_issues)


def test_nonshelled_pair(tmp_path):
    """All-non-shelled data: eddy errors; TORTOISE is clean (no mixture)."""
    grouping = load_scenario('nonshelled_pair', tmp_path)

    assert all(grouping.files[path].shelled is False for path in grouping.dwi_files)
    for backend in ('fsl', 'mixed'):
        assert 'eddy-requires-shelled' in issue_codes(check_backend(grouping, backend))
    tortoise_codes = issue_codes(check_backend(grouping, 'tortoise'))
    assert 'mixed-shelled-nonshelled' not in tortoise_codes
    assert 'eddy-requires-shelled' not in tortoise_codes


def test_shelling_undetermined_skips_checks(tmp_path):
    """Fixtures without bval files leave shelled undetermined: no data checks."""
    grouping = load_scenario('hcp_style', tmp_path)
    assert all(grouping.files[path].shelled is None for path in grouping.dwi_files)
    for backend in ('fsl', 'tortoise', 'mixed'):
        codes = issue_codes(check_backend(grouping, backend))
        assert 'eddy-requires-shelled' not in codes
        assert 'mixed-shelled-nonshelled' not in codes


def test_maxb_mismatch(tmp_path):
    """Very different maximum b-values in one output draw a warning."""
    grouping = load_scenario('maxb_mismatch', tmp_path)

    max_bvals = {
        grouping.files[p].filename: grouping.files[p].max_bval for p in grouping.dwi_files
    }
    assert max_bvals == {
        'sub-01_dir-AP_dwi.nii.gz': 1000.0,
        'sub-01_dir-PA_dwi.nii.gz': 3000.0,
    }
    assert 'maxb-mismatch' in issue_codes(grouping.warnings)


def test_fov_shift(tmp_path):
    """A rigid FoV offset warns (with unverifiable shim evidence here)."""
    grouping = load_scenario('fov_shift', tmp_path)

    (issue,) = [i for i in grouping.issues if i.code == 'fov-shifted']
    assert issue.severity == 'warning'
    assert 'cannot be verified' in issue.message
    assert '11.2 mm' in issue.message


def test_fov_oblique(tmp_path):
    """Differently-oriented FoVs error by default; ignore_fov downgrades."""
    with pytest.raises(GroupingError, match='fov-oblique'):
        load_scenario('fov_oblique', tmp_path)

    grouping = load_scenario('fov_oblique', tmp_path / 'nonstrict', strict=False)
    (issue,) = [i for i in grouping.issues if i.code == 'fov-oblique']
    assert issue.severity == 'error'
    assert '5.0 degrees' in issue.message

    grouping = load_scenario('fov_oblique', tmp_path / 'ignored', ignore_fov=True)
    (issue,) = [i for i in grouping.issues if i.code == 'fov-oblique']
    assert issue.severity == 'warning'


def test_fov_grid_mismatch_is_not_ignorable(tmp_path):
    """Different voxel grids are a hard error even with ignore_fov."""
    with pytest.raises(GroupingError, match='fov-grid-mismatch'):
        load_scenario('fov_grid', tmp_path)

    with pytest.raises(GroupingError, match='fov-grid-mismatch'):
        load_scenario('fov_grid', tmp_path / 'still-strict', ignore_fov=True)


def test_mixed_refinement_needs_rpe_series(tmp_path):
    """A second DRBUDDI stage without reverse-PE dMRI series draws a warning.

    abcd_style (epi fmap only, no T2w): the warning says correction is
    single-stage. abcd_t2w (same + T2w): the warning says the second stage is
    T2Wreg instead, and the preview narrates it.
    """
    from qsiprep.grouping import describe_processing

    grouping = load_scenario('abcd_style', tmp_path)
    (issue,) = [
        i for i in check_backend(grouping, 'mixed') if i.code == 'drbuddi-refinement-not-useful'
    ]
    assert 'probably not useful' in issue.message
    assert 'single-stage' in issue.message

    grouping = load_scenario('abcd_t2w', tmp_path)
    (issue,) = [
        i for i in check_backend(grouping, 'mixed') if i.code == 'drbuddi-refinement-not-useful'
    ]
    assert 'T2Wreg against a structural image' in issue.message
    assert 'T2Wreg registers the eddy-corrected b=0 to the T2w image' in describe_processing(
        grouping, 'mixed'
    )


def test_mixed_refinement_with_rpe_series(tmp_path):
    """With reverse-PE dMRI series the DRBUDDI refinement stage is legitimate."""
    from qsiprep.grouping import describe_processing

    grouping = load_scenario('t2w_hcp', tmp_path)
    assert 'drbuddi-refinement-not-useful' not in issue_codes(check_backend(grouping, 'mixed'))
    preview = describe_processing(grouping, 'mixed')
    assert 'DRBUDDI re-estimates distortion along the j axis' in preview
    # The T2w still rides along as DRBUDDI's structural target
    assert 'structural registration target' in preview


def test_synb0_overrides_t2w_as_structural_target(tmp_path):
    """With use_synb0, the synthetic b=0 replaces the T2w as DRBUDDI's
    structural target, even though the PEPOLAR estimation is unchanged."""
    from qsiprep.grouping import describe_processing

    grouping = load_scenario('t2w_hcp', tmp_path, use_synb0=True)
    # The real fieldmap still wins the application contest
    assert set(grouping.application.values()) == {'auto+pepolar+j'}
    preview = describe_processing(grouping, 'tortoise')
    assert 'a SyNb0 synthetic b=0 (from sub-01_T1w.nii.gz, in place of the T2w image)' in preview
