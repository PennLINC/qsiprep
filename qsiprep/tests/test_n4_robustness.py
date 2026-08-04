"""N4's log-domain fit must not be steered by near-zero voxels.

Covers both halves of the fix: the DWI weight image handed to
``dwibiascorrect``, and the intensity truncation ahead of the anatomical N4.
"""

import numpy as np
import pytest


def _config(**overrides):
    from qsiprep import config

    config.execution.sloppy = False
    config.nipype.omp_nthreads = 1
    config.workflow.anat_biascorrect = 'n4'
    config.workflow.subject_anatomical_reference = 'unbiased'
    config.workflow.b0_threshold = 100
    for key, value in overrides.items():
        setattr(config.workflow, key, value)
    return config


def _write(path, data, affine=None):
    import nibabel as nb

    nb.Nifti1Image(
        data.astype('float32'), affine if affine is not None else np.eye(4)
    ).to_filename(str(path))
    return str(path)


@pytest.fixture
def dwi_with_dropout(tmp_path):
    """A series whose mask contains a patch of susceptibility dropout."""
    rng = np.random.default_rng(0)
    shape = (12, 12, 12)

    mask = np.zeros(shape, dtype=bool)
    mask[2:10, 2:10, 2:10] = True

    b0 = rng.normal(300, 20, shape).astype('float32')  # background
    b0[mask] = rng.normal(4000, 200, int(mask.sum()))
    # dropout inside the brain: near-background, the log-domain outliers
    dropout = np.zeros(shape, dtype=bool)
    dropout[3:5, 3:5, 3:5] = True
    b0[dropout] = rng.normal(40, 10, int(dropout.sum()))

    vols, bvals = [], []
    for _ in range(4):
        vols.append(b0.copy())
        bvals.append(0)
    for _ in range(6):
        vols.append(b0 * 0.3)
        bvals.append(2000)

    dwi = _write(tmp_path / 'dwi.nii.gz', np.stack(vols, -1))
    bval = tmp_path / 'dwi.bval'
    bval.write_text(' '.join(str(b) for b in bvals) + '\n')
    mask_f = _write(tmp_path / 'mask.nii.gz', mask.astype('float32'))
    return dwi, str(bval), mask_f, mask, dropout


def test_dropout_voxels_are_removed_from_the_weights(dwi_with_dropout):
    import nibabel as nb

    from qsiprep.interfaces.bias import N4WeightMask

    dwi, bval, mask_f, mask, dropout = dwi_with_dropout
    res = N4WeightMask(dwi_file=dwi, bval_file=bval, mask_file=mask_f).run()
    w = np.asanyarray(nb.load(res.outputs.out_file).dataobj) > 0

    assert not w[dropout].any(), 'dropout voxels still carry weight'
    # everything else in the mask survives
    assert w[mask & ~dropout].all()
    assert not w[~mask].any(), 'weight leaked outside the mask'
    assert res.outputs.n_dropped == int(dropout.sum())


def test_geometry_is_preserved(dwi_with_dropout):
    import nibabel as nb

    from qsiprep.interfaces.bias import N4WeightMask

    dwi, bval, mask_f, _, _ = dwi_with_dropout
    res = N4WeightMask(dwi_file=dwi, bval_file=bval, mask_file=mask_f).run()
    out, ref = nb.load(res.outputs.out_file), nb.load(mask_f)
    assert out.shape == ref.shape
    assert np.allclose(out.affine, ref.affine)


def test_a_mostly_dark_mask_is_passed_through_untouched(tmp_path):
    """If most of the mask is dim, something upstream is already broken.

    Shrinking it further would hide that, so the mask goes through unchanged and
    the caller is warned.
    """
    import nibabel as nb

    from qsiprep.interfaces.bias import N4WeightMask

    rng = np.random.default_rng(2)
    shape = (10, 10, 10)
    mask = np.zeros(shape, dtype=bool)
    mask[1:9, 1:9, 1:9] = True

    b0 = rng.normal(300, 10, shape).astype('float32')
    b0[mask] = rng.normal(310, 10, int(mask.sum()))  # barely above background
    bright = np.zeros(shape, dtype=bool)
    bright[2:4, 2:4, 2:4] = True
    b0[bright] = 5000

    dwi = _write(tmp_path / 'd.nii.gz', np.stack([b0, b0 * 0.3], -1))
    bval = tmp_path / 'd.bval'
    bval.write_text('0 2000\n')
    mask_f = _write(tmp_path / 'm.nii.gz', mask.astype('float32'))

    res = N4WeightMask(dwi_file=dwi, bval_file=bval, mask_file=mask_f).run()
    w = np.asanyarray(nb.load(res.outputs.out_file).dataobj) > 0
    assert res.outputs.n_dropped == 0
    assert res.outputs.fraction_dropped == 0.0
    assert np.array_equal(w, mask)


def test_rejects_a_series_with_no_b0(tmp_path):
    from qsiprep.interfaces.bias import N4WeightMask

    data = np.ones((6, 6, 6, 3), dtype='float32')
    dwi = _write(tmp_path / 'd.nii.gz', data)
    bval = tmp_path / 'd.bval'
    bval.write_text('1000 2000 3000\n')
    mask_f = _write(tmp_path / 'm.nii.gz', np.ones((6, 6, 6), dtype='float32'))

    with pytest.raises(ValueError, match='No b=0 volumes'):
        N4WeightMask(dwi_file=dwi, bval_file=bval, mask_file=mask_f).run()


def test_biascorr_receives_the_conditioned_weights_not_the_raw_mask(tmp_path):
    """The whole point: N4's -w must be the damped mask.

    dwi_mask_t1 must still reach downstream consumers unchanged -- the fix is
    about what N4 fits, not about shrinking the brain mask.
    """
    from qsiprep.workflows.dwi.finalize import init_finalize_denoising_wf

    _config().execution.output_dir = str(tmp_path)

    wf = init_finalize_denoising_wf(
        source_file='/data/sub-01/ses-1/dwi/sub-01_ses-1_dwi.nii.gz',
        do_biascorr=True,
        num_dwi_acquisitions=1,
    )
    edges = {(u.name, v.name): d['connect'] for u, v, d in wf._graph.edges(data=True)}

    assert ('n4_weights', 'biascorr') in edges, 'biascorr does not use the conditioned weights'
    assert edges[('n4_weights', 'biascorr')] == [('out_file', 'mask')]

    # the raw mask feeds the conditioner, and no longer feeds biascorr directly
    assert ('dwi_mask_t1', 'mask_file') in edges[('inputnode', 'n4_weights')]
    assert ('dwi_mask_t1', 'mask') not in edges.get(('inputnode', 'biascorr'), [])


def test_anatomical_n4_truncates_first(tmp_path):
    """Both the single-image and multi-image paths must clip before N4."""
    from qsiprep.workflows.anatomical.volume import init_anat_template_wf

    _config().execution.output_dir = str(tmp_path)

    for num_images in (1, 3):
        wf = init_anat_template_wf(num_images=num_images, do_biascorr=True)
        nodes = {n.name: n for n in wf._get_all_nodes()}
        assert 'truncate_intensity' in nodes, f'no truncation for num_images={num_images}'

        edges = {(u.name, v.name): d['connect'] for u, v, d in wf._graph.edges(data=True)}
        assert edges[('truncate_intensity', 'n4_correct')] == [('out_file', 'input_image')], (
            f'N4 is not fed by the truncation for num_images={num_images}'
        )
        # nothing else may reach N4's input
        into_n4 = [k for k in edges if k[1] == 'n4_correct']
        assert into_n4 == [('truncate_intensity', 'n4_correct')], into_n4

        trunc = nodes['truncate_intensity'].interface.inputs
        assert trunc.operation == 'TruncateImageIntensity'
        assert trunc.secondary_arg == '0.01 0.999 256'


def test_anatomical_truncation_absent_when_biascorr_is_off(tmp_path):
    from qsiprep.workflows.anatomical.volume import init_anat_template_wf

    _config().execution.output_dir = str(tmp_path)
    wf = init_anat_template_wf(num_images=3, do_biascorr=False)
    names = {n.name for n in wf._get_all_nodes()}
    assert 'truncate_intensity' not in names
    assert 'n4_correct' not in names
