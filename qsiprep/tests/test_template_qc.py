"""Per-input agreement metrics for the subject-level template."""

import numpy as np
import pytest


def _write(path, data, affine=None):
    import nibabel as nb

    nb.Nifti1Image(data.astype('float32'), affine if affine is not None else np.eye(4)).to_filename(
        str(path)
    )
    return str(path)


def _structured(shape=(28, 28, 28), shift=0):
    """A block with internal structure, so correlation inside it is meaningful.

    A uniform block will not do: the QC mask covers the block interior, and
    inside a flat region every image is constant-plus-noise, so correlation is
    ~0 for good and bad inputs alike.
    """
    vol = np.zeros(shape, dtype='float32')
    gx, gy, gz = np.indices(shape)
    body = (
        (gx >= 6 + shift) & (gx < 22 + shift)
        & (gy >= 6) & (gy < 22)
        & (gz >= 6) & (gz < 22)
    )
    vol[body] = 60.0
    # internal features -- the thing a correlation can actually latch onto
    vol[(gx >= 10 + shift) & (gx < 14 + shift) & (gy >= 10) & (gy < 18) & (gz >= 10) & (gz < 18)] = 140.0
    vol[(gx >= 16 + shift) & (gx < 19 + shift) & (gy >= 12) & (gy < 16) & (gz >= 12) & (gz < 16)] = 20.0
    return vol


@pytest.fixture
def template_set(tmp_path):
    """Three inputs matching a template, plus one deliberate outlier."""
    rng = np.random.default_rng(0)
    base = _structured()
    template = _write(tmp_path / 'template.nii.gz', base + rng.normal(0, 1, base.shape))

    good = [
        _write(tmp_path / f'good{i}.nii.gz', base + rng.normal(0, 2, base.shape))
        for i in range(3)
    ]
    # same anatomy, displaced: the internal features no longer line up
    outlier = _write(
        tmp_path / 'outlier.nii.gz',
        _structured(shift=5) + rng.normal(0, 2, base.shape),
    )
    return template, good + [outlier]


def test_outlier_is_flagged(template_set):
    import pandas as pd

    from qsiprep.interfaces.template_qc import TemplateQC

    template, images = template_set
    res = TemplateQC(
        aligned_images=images,
        template=template,
        labels=['a', 'b', 'c', 'bad'],
    ).run()
    frame = pd.read_csv(res.outputs.out_file, sep='\t')

    assert list(frame['label']) == ['a', 'b', 'c', 'bad']
    bad = frame.set_index('label').loc['bad']
    assert bad['corr_to_template'] < frame['corr_to_template'].max()
    assert bool(bad['outlier']) is True
    assert frame['outlier'].sum() == 1, 'only the planted outlier should be flagged'


def test_agreement_map_is_written_on_the_template_grid(template_set):
    import nibabel as nb

    from qsiprep.interfaces.template_qc import TemplateQC

    template, images = template_set
    res = TemplateQC(aligned_images=images, template=template).run()
    out = nb.load(res.outputs.agreement_map)
    ref = nb.load(template)
    assert out.shape == ref.shape
    assert np.allclose(out.affine, ref.affine)


def test_transform_columns_are_populated_for_float_matrices(tmp_path, template_set):
    """ANTs writes AffineTransform_float_3_3 when registration runs float=True.

    Hardcoding the double-precision key left these columns silently NaN.
    """
    import pandas as pd
    from scipy import io as sio

    from qsiprep.interfaces.template_qc import TemplateQC

    template, images = template_set
    mats = []
    for i in range(len(images)):
        path = tmp_path / f'xfm{i}.mat'
        params = np.concatenate([np.eye(3).ravel(), [float(i), 0.0, 0.0]])
        sio.savemat(
            str(path),
            {'AffineTransform_float_3_3': params.reshape(-1, 1),
             'fixed': np.zeros((3, 1))},
        )
        mats.append(str(path))

    res = TemplateQC(aligned_images=images, template=template, transforms=mats).run()
    frame = pd.read_csv(res.outputs.out_file, sep='\t')
    assert frame['translation_mm'].notna().all(), 'float-precision matrices were not read'
    assert frame['translation_mm'].iloc[2] == pytest.approx(2.0, abs=1e-6)


def test_unreadable_transform_does_not_break_the_run(tmp_path, template_set):
    """QC must never fail a run -- but it must not fail silently either."""
    import pandas as pd

    from qsiprep.interfaces.template_qc import TemplateQC

    template, images = template_set
    junk = tmp_path / 'junk.mat'
    junk.write_bytes(b'not a matlab file')

    res = TemplateQC(
        aligned_images=images, template=template, transforms=[str(junk)] * len(images)
    ).run()
    frame = pd.read_csv(res.outputs.out_file, sep='\t')
    assert frame['corr_to_template'].notna().all(), 'correlation should still be computed'
