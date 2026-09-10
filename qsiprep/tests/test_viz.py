"""Tests for the reportlet plotting functions.

Reportlets that show a "before" and an "after" image must display both with the
same field of view, otherwise the two frames of the flicker animation are not
comparable.
"""

import matplotlib as mpl
import nibabel as nb
import numpy as np
import pytest

mpl.use('Agg')

from qsiprep.interfaces import fmap  # noqa: E402
from qsiprep.viz import utils as viz_utils  # noqa: E402


def _grid_img(data, resolution, origin=None):
    """Build a Nifti1Image on an isotropic RAS-ish grid."""
    affine = np.diag([resolution, resolution, resolution, 1.0])
    if origin is None:
        origin = [-(n * resolution) / 2 for n in data.shape]
    affine[:3, 3] = origin
    return nb.Nifti1Image(data.astype('float32'), affine)


@pytest.fixture
def fov_mismatch():
    """A segmentation plus "before"/"after" images sampled on different grids.

    This mimics DRBUDDI, which writes its corrected images onto a finer grid with
    a larger field of view than the images it was given.
    """
    seg_data = np.zeros((60, 60, 60))
    seg_data[20:40, 20:40, 20:40] = 1
    seg = _grid_img(seg_data, 2.0)

    rng = np.random.default_rng(42)
    # "Before": coarse grid, smaller FOV
    before = _grid_img(rng.uniform(1, 100, size=(24, 24, 24)), 5.0)
    # "After": fine grid, larger FOV
    after = _grid_img(rng.uniform(1, 100, size=(70, 70, 70)), 2.0)
    return seg, before, after


def _capture_axis_limits(monkeypatch, module):
    """Record the axis limits of every display right before it is rasterized."""
    captured = []
    real_extract_svg = module.extract_svg

    def spy(display, compress='auto'):
        captured.append(
            {
                name: (tuple(ax.ax.get_xlim()), tuple(ax.ax.get_ylim()))
                for name, ax in display.axes.items()
            }
        )
        return real_extract_svg(display, compress=compress)

    monkeypatch.setattr(module, 'extract_svg', spy)
    return captured


def test_plot_pepolar_fov_is_independent_of_the_plotted_image(monkeypatch, fov_mismatch):
    """The before and after panels share a field of view despite differing grids."""
    seg, before, after = fov_mismatch
    from nilearn.image import crop_img

    _, crop_offset = crop_img(seg, return_offset=True)
    cuts = {'z': [-10.0, 0.0, 10.0], 'x': [-10.0, 0.0, 10.0], 'y': [-10.0, 0.0, 10.0]}
    captured = _capture_axis_limits(monkeypatch, fmap)

    for img, div_id in ((before, 'moving-image'), (after, 'fixed-image')):
        fmap.plot_pepolar(
            img,
            img,
            seg,
            div_id,
            estimate_brightness=True,
            cuts=cuts,
            crop_offset=crop_offset,
            label='Test',
            compress=False,
        )

    # Six panels per call (three orientations x blip up/down), twelve in total.
    assert len(captured) == 12
    assert all(limits == captured[0] for limits in captured)


def test_plot_fa_reg_fov_is_independent_of_the_plotted_image(monkeypatch, fov_mismatch):
    """The FA before and after panels share a field of view despite differing grids."""
    seg, before, after = fov_mismatch
    from nilearn.image import crop_img

    _, crop_offset = crop_img(seg, return_offset=True)
    cuts = {'z': [-10.0, 0.0, 10.0], 'x': [-10.0, 0.0, 10.0], 'y': [-10.0, 0.0, 10.0]}
    captured = _capture_axis_limits(monkeypatch, fmap)

    for img, div_id in ((before, 'moving-image'), (after, 'fixed-image')):
        fmap.plot_fa_reg(
            img,
            seg,
            div_id,
            cuts=cuts,
            crop_offset=crop_offset,
            label='Test',
            compress=False,
        )

    assert len(captured) == 6
    assert all(limits == captured[0] for limits in captured)


def test_plot_denoise_honors_the_crop(monkeypatch, fov_mismatch):
    """Contours drawn from an uncropped image do not undo the requested crop."""
    seg, _, _ = fov_mismatch
    from nilearn.image import crop_img

    cropped_seg, crop_offset = crop_img(seg, return_offset=True)
    rng = np.random.default_rng(0)
    # A difference image that is non-zero over the whole, uncropped field of view.
    contour = _grid_img(rng.normal(size=seg.shape), 2.0)
    cuts = {'z': [-10.0, 0.0, 10.0], 'x': [-10.0, 0.0, 10.0], 'y': [-10.0, 0.0, 10.0]}
    captured = _capture_axis_limits(monkeypatch, viz_utils)

    viz_utils.plot_denoise(
        seg,
        seg,
        'moving-image',
        estimate_brightness=True,
        cuts=cuts,
        crop_offset=crop_offset,
        label='Test',
        lowb_contour=contour,
        highb_contour=contour,
        compress=False,
    )

    # The plotted extent must match the cropped image, not the full field of view.
    span = max(cropped_seg.shape) * 2.0
    for limits in captured:
        for xlim, ylim in limits.values():
            assert abs(xlim[1] - xlim[0]) <= span
            assert abs(ylim[1] - ylim[0]) <= span
