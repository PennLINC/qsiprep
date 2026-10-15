"""SynB0 + TOPUP after head motion correction, its second pass, and SyN against a SynB0."""

import nibabel as nb
import numpy as np
import pytest

from qsiprep.interfaces.fmap import PostHmcTopupInputs, TopupFieldToWarp


def _image(path, axcodes, shape=(6, 7, 5), zooms=(2.0, 2.5, 3.0), data=None):
    """Write an image on a fixed world grid, stored in ``axcodes`` orientation."""
    ras = nb.Nifti1Image(
        np.arange(np.prod(shape), dtype=np.float32).reshape(shape) if data is None else data,
        np.diag([*zooms, 1.0]) + np.array([[0, 0, 0, -6], [0, 0, 0, -9], [0, 0, 0, 4], [0] * 4]),
    )
    current = nb.orientations.io_orientation(ras.affine)
    target = nb.orientations.axcodes2ornt(axcodes)
    img = ras.as_reoriented(nb.orientations.ornt_transform(current, target))
    img.to_filename(path)
    return str(path)


@pytest.mark.parametrize(
    ('raw_axcodes', 'pe_dir', 'vector'),
    [
        # dcm2niix's usual LAS+: its j axis is LAS+'s j axis.
        (('L', 'A', 'S'), 'j', '0 1 0'),
        (('L', 'A', 'S'), 'j-', '0 -1 0'),
        # An LPS raw file runs j posterior-ward, against LAS+'s j.
        (('L', 'P', 'S'), 'j', '0 -1 0'),
        (('R', 'A', 'S'), 'i', '-1 0 0'),
        (('L', 'P', 'S'), 'i', '1 0 0'),
        # A permuted raw file: its i axis is anterior-posterior.
        (('P', 'L', 'S'), 'i-', '0 1 0'),
    ],
)
def test_topup_inputs_express_the_raw_pe_direction_in_las(tmp_path, raw_axcodes, pe_dir, vector):
    raw = _image(tmp_path / 'raw_dwi.nii.gz', raw_axcodes)
    b0 = _image(tmp_path / 'b0.nii.gz', ('L', 'P', 'S'))

    result = PostHmcTopupInputs(
        b0_file=b0, raw_dwi_file=raw, pe_dir=pe_dir, readout_time=0.05
    ).run(cwd=str(tmp_path))

    datain = open(result.outputs.topup_datain).read().split()
    assert ' '.join(datain[:3]) == vector
    assert float(datain[3]) == pytest.approx(0.05)
    imain = nb.load(result.outputs.topup_imain)
    assert nb.aff2axcodes(imain.affine) == ('L', 'A', 'S')
    assert imain.shape == (6, 7, 5, 1)
    # Only the storage order changed: the same world point holds the same value.
    lps = nb.load(b0)
    world = lps.affine @ [1, 2, 3, 1]
    ijk = np.round(np.linalg.solve(imain.affine, world)[:3]).astype(int)
    assert imain.get_fdata()[(*ijk, 0)] == lps.get_fdata()[1, 2, 3]


def test_topup_inputs_refuse_an_oblique_pe_direction(tmp_path):
    raw_img = nb.load(_image(tmp_path / 'raw.nii.gz', ('L', 'A', 'S')))
    rotation = np.eye(4)
    rotation[:2, :2] = [[np.cos(0.5), -np.sin(0.5)], [np.sin(0.5), np.cos(0.5)]]
    oblique = nb.Nifti1Image(raw_img.get_fdata(), rotation @ raw_img.affine)
    oblique.to_filename(tmp_path / 'oblique.nii.gz')
    b0 = _image(tmp_path / 'b0.nii.gz', ('L', 'P', 'S'))

    with pytest.raises(ValueError, match='not along a voxel axis'):
        PostHmcTopupInputs(
            b0_file=b0,
            raw_dwi_file=str(tmp_path / 'oblique.nii.gz'),
            pe_dir='j',
            readout_time=0.05,
        ).run(cwd=str(tmp_path))


def test_topup_field_warp_matches_the_eddy_path_conversion(tmp_path):
    """The warp equals what ``_hz_to_warp`` makes of the same field, at every world point."""
    from qsiprep.workflows.dwi.resampling import _hz_to_warp

    rng = np.random.default_rng(0)
    hz = rng.normal(0, 40, (6, 7, 5)).astype(np.float32)
    field_las = _image(tmp_path / 'field_las.nii.gz', ('R', 'A', 'S'), data=hz)
    # Stored as LAS+, as TOPUP sees it.
    field_img = nb.load(field_las)
    las = field_img.as_reoriented(
        nb.orientations.ornt_transform(
            nb.orientations.io_orientation(field_img.affine),
            nb.orientations.axcodes2ornt(('L', 'A', 'S')),
        )
    )
    las.to_filename(tmp_path / 'fieldmap_HZ.nii.gz')
    datain = tmp_path / 'datain.txt'
    datain.write_text('0 -1 0 0.062\n0 -1 0 0.000000\n')
    reference = _image(tmp_path / 'b0_lps.nii.gz', ('L', 'P', 'S'))

    result = TopupFieldToWarp(
        fieldmap=str(tmp_path / 'fieldmap_HZ.nii.gz'),
        datain=str(datain),
        reference=reference,
    ).run(cwd=str(tmp_path))
    warp = nb.load(result.outputs.out_warp)
    ref = nb.load(reference)
    assert warp.shape == (*ref.shape, 1, 3)
    np.testing.assert_allclose(warp.affine, ref.affine)

    # Both name their output <field>_warp.nii.gz.
    (tmp_path / 'eddy_path').mkdir()
    expected = nb.load(
        _hz_to_warp(
            str(tmp_path / 'fieldmap_HZ.nii.gz'), 0.062, 'j-', newpath=str(tmp_path / 'eddy_path')
        )
    )
    # Compare at the same world points: map each reference voxel into the LAS+ grid.
    ijk = np.indices(ref.shape).reshape(3, -1)
    world = ref.affine @ np.vstack([ijk, np.ones(ijk.shape[1])])
    las_ijk = np.round(np.linalg.solve(expected.affine, world)[:3]).astype(int)
    got = warp.get_fdata()[ijk[0], ijk[1], ijk[2], 0]
    want = expected.get_fdata()[las_ijk[0], las_ijk[1], las_ijk[2], 0]
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)
    # Only the phase-encoding (A-P) component is nonzero.
    assert np.abs(got[:, [0, 2]]).max() == 0
    assert np.abs(got[:, 1]).max() > 0

    hz_ref = nb.load(result.outputs.out_hz)
    np.testing.assert_allclose(hz_ref.affine, ref.affine)


def test_topup_field_warp_refuses_another_grid(tmp_path):
    field = _image(tmp_path / 'fieldmap_HZ.nii.gz', ('L', 'A', 'S'))
    other = _image(tmp_path / 'other.nii.gz', ('L', 'P', 'S'), shape=(6, 7, 6))
    datain = tmp_path / 'datain.txt'
    datain.write_text('0 1 0 0.05\n')
    with pytest.raises(ValueError, match='not on the grid'):
        TopupFieldToWarp(fieldmap=field, datain=str(datain), reference=other).run(
            cwd=str(tmp_path)
        )


@pytest.mark.parametrize(('ref_axcodes', 'axis'), [(('L', 'P', 'S'), 1), (('P', 'L', 'S'), 0)])
def test_topup_field_warp_reports_the_pe_axis_of_the_reference(tmp_path, ref_axcodes, axis):
    field = _image(tmp_path / 'fieldmap_HZ.nii.gz', ('L', 'A', 'S'))
    reference = _image(tmp_path / 'b0.nii.gz', ref_axcodes)
    datain = tmp_path / 'datain.txt'
    datain.write_text('0 1 0 0.05\n')
    result = TopupFieldToWarp(fieldmap=field, datain=str(datain), reference=reference).run(
        cwd=str(tmp_path)
    )
    assert result.outputs.pe_axis == axis


def test_modulation_scales_by_the_phase_encoding_jacobian(tmp_path):
    """A displacement growing linearly along PE stretches every interior voxel by 1 + slope."""
    from qsiprep.interfaces.fmap import ModulateByWarpJacobian

    shape, slope = (6, 7, 5), 0.3
    b0 = _image(tmp_path / 'b0.nii.gz', ('L', 'P', 'S'), data=np.ones(shape, np.float32))
    img = nb.load(b0)
    displacement = np.zeros((*img.shape, 1, 3), np.float32)
    # LPS storage: voxel axis 1 runs posterior-ward, the ITK warp's y.
    displacement[..., 0, 1] = (
        slope * np.arange(img.shape[1])[None, :, None] * img.header.get_zooms()[1]
    )
    warp = nb.Nifti1Image(displacement, img.affine)
    warp.header.set_intent('vector')
    warp.to_filename(tmp_path / 'warp.nii.gz')

    result = ModulateByWarpJacobian(in_file=b0, warp=str(tmp_path / 'warp.nii.gz'), pe_axis=1).run(
        cwd=str(tmp_path)
    )
    out = nb.load(result.outputs.out_file).get_fdata()
    np.testing.assert_allclose(out[:, 1:-1, :], 1 + slope, rtol=1e-5)
    # TORTOISE leaves the edge slices along PE at unity.
    np.testing.assert_allclose(out[:, [0, -1], :], 1.0)


def _connections(wf, src, dst):
    return [
        conn
        for u, v, data in wf._graph.edges(data=True)
        if u.name == src and v.name == dst
        for conn in data['connect']
    ]


def test_synb0_topup_second_pass_regenerates_synb0_from_the_unwarped_b0():
    from qsiprep.workflows.fieldmap.synb0 import init_synb0_topup_wf

    single = init_synb0_topup_wf(
        'sub-01_dwi.nii.gz', 'j-', 0.05, 'sub-01_dwi.nii.gz', two_pass=False
    )
    assert single.get_node('topup_pass2') is None
    assert 'again' not in single.__postdesc__

    wf = init_synb0_topup_wf('sub-01_dwi.nii.gz', 'j-', 0.05, 'sub-01_dwi.nii.gz')
    assert ('output_image', 'in_file') in _connections(wf, 'unwarp_b0', 'modulate_unwarped_b0')
    assert {('out_warp', 'warp'), ('pe_axis', 'pe_axis')} <= set(
        _connections(wf, 'field_to_warp', 'modulate_unwarped_b0')
    )
    assert ('out_file', 'inputnode.b0_template') in _connections(
        wf, 'modulate_unwarped_b0', 'synb0_b0_ref_wf_pass2'
    )
    # TOPUP's second run pairs the new synthetic b=0 with the ORIGINAL b=0.
    assert ('topup_imain', 'topup_imain') in _connections(
        wf, 'topup_inputs', 'synb0_topup_inputs_pass2'
    )
    assert ('outputnode.synthetic_b0', 'synthetic_b0') in _connections(
        wf, 'synb0_wf_pass2', 'synb0_topup_inputs_pass2'
    )
    # The pass that is applied, reported and checked is the second.
    assert ('out_warp', 'out_warp') in _connections(wf, 'field_to_warp_pass2', 'outputnode')
    assert ('output_image', 'b0_corrected') in _connections(wf, 'unwarp_b0_pass2', 'outputnode')
    assert ('qc_file', 'field_qc_file') in _connections(wf, 'field_qc_pass2', 'outputnode')
    assert ('outputnode.synthetic_b0', 'synthetic_b0') in _connections(
        wf, 'synb0_wf_pass2', 'outputnode'
    )
    assert 'again' in wf.__postdesc__


def _syn_unit(reference):
    from qsiplan.models import CorrectionMethod

    from qsiprep.tests.preproc_factory import make_preproc_unit

    t1w = ['/data/sub-01_T1w.nii.gz']
    return make_preproc_unit(
        ['/data/sub-01_dwi.nii.gz'],
        method=CorrectionMethod.NIPREPS_SYN,
        estimation_sources=t1w,
        anat_files=t1w,
        reference=reference,
    )


@pytest.mark.parametrize(
    ('reference', 'fixed_source', 'label'),
    [
        ('invt1w', ('t1_2_ref', 'output_image'), 'SyN (fieldmap-less)'),
        ('synb0', ('inputnode', 'synthetic_b0'), 'SyN (SynB0)'),
    ],
)
def test_syn_registers_to_its_reference(reference, fixed_source, label):
    from qsiprep import config
    from qsiprep.utils.sdc import sdc_warp_source
    from qsiprep.workflows.fieldmap.base import init_sdc_wf

    config.nipype.omp_nthreads = 1
    unit = _syn_unit(reference)
    wf = init_sdc_wf(unit, source_file='/data/sub-01_dwi.nii.gz')
    syn_wf = wf.get_node('syn_sdc_wf')
    assert (fixed_source[1], 'fixed_image') in _connections(syn_wf, fixed_source[0], 'syn')
    assert (wf.get_node('synb0_wf') is not None) is (reference == 'synb0')
    if reference == 'synb0':
        assert ('outputnode.synthetic_b0', 'inputnode.synthetic_b0') in _connections(
            wf, 'synb0_wf', 'syn_sdc_wf'
        )
        assert 'SynB0-DISCO' in syn_wf.__desc__
    assert sdc_warp_source(unit) == ('fieldwarp', label)
