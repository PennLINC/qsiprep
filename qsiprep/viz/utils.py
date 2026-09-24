"""Visualization utilities."""

from contextlib import contextmanager

import nibabel as nb
import numpy as np
from lxml import etree
from nilearn import plotting
from niworkflows.viz.utils import SVGNS, extract_svg, robust_set_limits, uuid4
from svgutils.transform import SVGFigure

from ..utils.misc import invert_displacement_field


@contextmanager
def fixed_field_of_view(display):
    """Keep a nilearn display's field of view fixed while objects are added to it.

    Nilearn grows a display's axes to fit the union of everything drawn on them, so an
    overlay or contour taken from an image with a larger field of view than the
    background silently undoes any cropping of the background. When the "before" and
    "after" images of a reportlet sit on different grids, that leaves the two frames
    zoomed relative to one another.

    Restoring the limits that the background image established keeps both frames on the
    same field of view.
    """
    limits = {name: (ax.ax.get_xlim(), ax.ax.get_ylim()) for name, ax in display.axes.items()}
    try:
        yield display
    finally:
        for name, ax in display.axes.items():
            xlim, ylim = limits[name]
            ax.ax.set_xlim(*xlim)
            ax.ax.set_ylim(*ylim)


def plot_denoise(
    lowb_nii,
    highb_nii,
    div_id,
    plot_params=None,
    highb_plot_params=None,
    order=('z', 'x', 'y'),
    cuts=None,
    crop_offset=None,
    estimate_brightness=False,
    label=None,
    lowb_contour=None,
    highb_contour=None,
    upper_label_suffix=': low-b',
    lower_label_suffix=': high-b',
    compress='auto',
    overlay=None,
    overlay_params=None,
):
    """
    Plot the foreground and background views.
    Default order is: axial, coronal, sagittal

    Updated version from sdcflows
    """
    plot_params = plot_params or {}
    highb_plot_params = highb_plot_params or {}

    # Use default MNI cuts if none defined
    if cuts is None:
        raise NotImplementedError

    # Do the low-b image first
    out_files = []

    # Plot each cut axis for low-b
    if estimate_brightness:
        plot_params = robust_set_limits(
            lowb_nii.get_fdata(dtype='float32').reshape(-1), plot_params
        )

    lowb_nii_cropped = lowb_nii if crop_offset is None else lowb_nii.slicer[crop_offset]
    for i, mode in enumerate(list(order)):
        plot_params['display_mode'] = mode
        plot_params['cut_coords'] = cuts[mode]
        if i == 0:
            plot_params['title'] = label + upper_label_suffix
        else:
            plot_params['title'] = None

        # Generate nilearn figure
        display = plotting.plot_anat(lowb_nii_cropped, **plot_params)
        with fixed_field_of_view(display):
            if lowb_contour is not None:
                display.add_contours(lowb_contour, linewidths=1)

        svg = extract_svg(display, compress=compress)
        display.close()

        # Find and replace the figure_1 id.
        xml_data = etree.fromstring(svg)  # noqa: S320
        find_text = etree.ETXPath(f"//{{{SVGNS}}}g[@id='figure_1']")
        find_text(xml_data)[0].set('id', f'{div_id}-{mode}-{uuid4()}')

        svg_fig = SVGFigure()
        svg_fig.root = xml_data
        out_files.append(svg_fig)

    # Plot each cut axis for high-b
    if estimate_brightness:
        highb_plot_params = robust_set_limits(
            highb_nii.get_fdata(dtype='float32').reshape(-1), highb_plot_params
        )

    highb_nii_cropped = highb_nii if crop_offset is None else highb_nii.slicer[crop_offset]
    for i, mode in enumerate(list(order)):
        highb_plot_params['display_mode'] = mode
        highb_plot_params['cut_coords'] = cuts[mode]
        if i == 0:
            highb_plot_params['title'] = label + lower_label_suffix
        else:
            highb_plot_params['title'] = None

        # Generate nilearn figure
        display = plotting.plot_anat(highb_nii_cropped, **highb_plot_params)
        with fixed_field_of_view(display):
            if highb_contour is not None:
                display.add_contours(highb_contour, linewidths=1)

        svg = extract_svg(display, compress=compress)
        display.close()

        # Find and replace the figure_1 id.
        xml_data = etree.fromstring(svg)  # noqa: S320
        find_text = etree.ETXPath(f"//{{{SVGNS}}}g[@id='figure_1']")
        find_text(xml_data)[0].set('id', f'{div_id}-{mode}-{uuid4()}')

        svg_fig = SVGFigure()
        svg_fig.root = xml_data
        out_files.append(svg_fig)

    return out_files


def plot_acpc(
    acpc_registered_img,
    div_id,
    plot_params=None,
    order=('z', 'x', 'y'),
    crop_offset=None,
    estimate_brightness=False,
    label=None,
    compress='auto',
):
    """
    Plot the results of an AC-PC transformation.
    """
    plot_params = plot_params or {}

    # Do the low-b image first
    out_files = []
    if estimate_brightness:
        plot_params = robust_set_limits(
            acpc_registered_img.get_fdata(dtype='float32').reshape(-1), plot_params
        )

    # Plot each cut axis for low-b
    acpc_registered_img_cropped = (
        acpc_registered_img if crop_offset is None else acpc_registered_img.slicer[crop_offset]
    )
    for i, mode in enumerate(list(order)):
        plot_params['display_mode'] = mode
        plot_params['cut_coords'] = [-20.0, 0.0, 20.0]
        if i == 0:
            plot_params['title'] = label
        else:
            plot_params['title'] = None

        # Generate nilearn figure
        display = plotting.plot_anat(acpc_registered_img_cropped, **plot_params)
        for _coord, axis in display.axes.items():
            axis.ax.axvline(0, lw=1)
            axis.ax.axhline(0, lw=1)
        svg = extract_svg(display, compress=compress)
        display.close()

        # Find and replace the figure_1 id.
        xml_data = etree.fromstring(svg)  # noqa: S320
        find_text = etree.ETXPath(f"//{{{SVGNS}}}g[@id='figure_1']")
        find_text(xml_data)[0].set('id', f'{div_id}-{mode}-{uuid4()}')

        svg_fig = SVGFigure()
        svg_fig.root = xml_data
        out_files.append(svg_fig)

    return out_files


def sdc_warp_glyph_field(warp_file):
    """Load an SDC displacement field as the vectors 3D Slicer draws.

    Slicer draws where points move, the inverse of an image-resampling field,
    so the field is inverted and its ITK (LPS) vectors converted to RAS.

    Parameters
    ----------
    warp_file : str
        ITK displacement field.

    Returns
    -------
    disp_ras : ndarray
        Displacements in RAS mm, shape (X, Y, Z, 3).
    magnitude : ndarray
        Displacement length per voxel, in mm.
    affine : ndarray
        Voxel-to-RAS affine of the field.
    """
    import SimpleITK as sitk

    affine = nb.load(warp_file).affine
    inverse = invert_displacement_field(warp_file)
    disp = np.moveaxis(sitk.GetArrayFromImage(inverse), [0, 1, 2], [2, 1, 0])  # (X,Y,Z,3) LPS
    disp_ras = disp * np.array([-1.0, -1.0, 1.0])
    return disp_ras, np.linalg.norm(disp, axis=-1), affine


def sdc_warp_display_planes(disp_ras, affine):
    """Find the phase-encoding axis and the two planes that contain it.

    Parameters
    ----------
    disp_ras : ndarray
        Displacements in RAS mm, shape (X, Y, Z, 3).
    affine : ndarray
        Voxel-to-RAS affine.

    Returns
    -------
    ped_ras : int
        RAS axis with the largest mean displacement.
    slice_axes : list of int
        The two voxel axes to slice along, least-displaced normal first.
    vox_to_ras : ndarray
        The RAS axis closest to each voxel axis.
    """
    vox_to_ras = np.argmax(np.abs(affine[:3, :3]), axis=0)
    ras_disp = np.array([np.abs(disp_ras[..., a]).mean() for a in range(3)])
    ped_ras = int(np.argmax(ras_disp))
    normal_ras = sorted((a for a in range(3) if a != ped_ras), key=lambda a: ras_disp[a])
    slice_axes = [int(np.where(vox_to_ras == n)[0][0]) for n in normal_ras]
    return ped_ras, slice_axes, vox_to_ras


def _glyph_slice(disp_ras, mag, b0, affine, slice_axis, sl, vox_to_ras):
    """Background, physical mesh, in-plane displacement components for one slice."""
    inplane = sorted((a for a in range(3) if a != slice_axis), key=lambda a: vox_to_ras[a])
    h_ax, v_ax = inplane  # voxel axes -> horizontal, vertical
    h_ras, v_ras = vox_to_ras[h_ax], vox_to_ras[v_ax]
    gvv, ghh = np.meshgrid(
        np.arange(disp_ras.shape[v_ax]), np.arange(disp_ras.shape[h_ax]), indexing='ij'
    )
    h = affine[h_ras, slice_axis] * sl + affine[h_ras, h_ax] * ghh + affine[h_ras, v_ax] * gvv
    v = affine[v_ras, slice_axis] * sl + affine[v_ras, h_ax] * ghh + affine[v_ras, v_ax] * gvv
    take = [slice(None)] * 3
    take[slice_axis] = sl

    def grid(arr):
        plane = arr[tuple(take)]
        return plane.T if h_ax < v_ax else plane

    return {
        'bg': grid(b0),
        'H': h + affine[h_ras, 3],
        'V': v + affine[v_ras, 3],
        'dh': grid(disp_ras[..., h_ras]),
        'dv': grid(disp_ras[..., v_ras]),
        'm2': grid(mag),
        'h_ras': h_ras,
        'v_ras': v_ras,
    }


def _glyph_slice_positions(b0, mag, slice_axis, n):
    """``n`` slice indices spread over the part of the brain that carries displacement."""
    others = tuple(a for a in range(3) if a != slice_axis)
    brain = (b0 > 0.1 * b0.max()).sum(axis=others)
    valid = np.where((brain > brain.max() * 0.15) & (mag.sum(axis=others) > 0))[0]
    if valid.size == 0:
        valid = np.where(brain > 0)[0]
    if valid.size == 0:
        valid = np.array([b0.shape[slice_axis] // 2])
    picks = sorted({int(np.quantile(valid, q)) for q in np.linspace(0.5 / n, 1 - 0.5 / n, n)})
    while len(picks) < n:  # small brains can collapse the quantiles
        picks.append(picks[-1])
    return picks[:n]


def sdc_warp_glyph_scale(mag, affine, step):
    """Choose how to draw a field's arrows so small and large fields both read.

    Low-signal voxels are not masked: dropout is where displacement matters most.

    Parameters
    ----------
    mag : ndarray
        Displacement length per voxel, in mm.
    affine : ndarray
        Voxel-to-RAS affine.
    step : int
        Voxels between arrows.

    Returns
    -------
    min_mag : float
        Shortest arrow drawn: 0.5 mm, or a tenth of the 99th percentile if smaller.
    vmax : float
        Upper color limit, the 99th percentile.
    arrow_scale : float
        Arrow length multiplier. Above 1 only when the largest arrows would span
        less than 40% of the arrow spacing.
    """
    moving = mag[mag > 0.01]
    if not moving.size:
        return 0.5, 1.0, 1.0
    p99 = float(np.percentile(moving, 99))
    spacing = step * float(np.mean(np.linalg.norm(affine[:3, :3], axis=0)))
    stretch = 0.8 * spacing / p99
    return min(0.5, 0.1 * p99), p99, stretch if stretch >= 2 else 1.0


def _draw_glyph(ax, panel, clim, step, min_mag, arrow_scale):
    """Draw one glyph panel; anterior/superior/left to screen-left/top."""
    xd, ud, yd, vd = -panel['H'], -panel['dh'], panel['V'], panel['dv']
    ax.pcolormesh(xd, yd, panel['bg'], cmap='gray', shading='nearest', rasterized=True)
    keep = panel['m2'][::step, ::step] > min_mag
    q = ax.quiver(
        xd[::step, ::step][keep],
        yd[::step, ::step][keep],
        ud[::step, ::step][keep],
        vd[::step, ::step][keep],
        panel['m2'][::step, ::step][keep],
        cmap='turbo',
        clim=clim,
        angles='xy',
        scale_units='xy',
        scale=1.0 / arrow_scale,
        width=0.005,
        headwidth=4,
        pivot='tail',
    )
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ends = {0: ('R', 'L'), 1: ('A', 'P'), 2: ('S', 'I')}  # (positive end, negative end)
    for frac, txt in [
        ((0.03, 0.5), ends[panel['h_ras']][0]),
        ((0.95, 0.5), ends[panel['h_ras']][1]),
        ((0.5, 0.95), ends[panel['v_ras']][0]),
        ((0.5, 0.05), ends[panel['v_ras']][1]),
    ]:
        ax.text(
            *frac,
            txt,
            transform=ax.transAxes,
            color='yellow',
            fontsize=8,
            ha='center',
            va='center',
            weight='bold',
        )
    for sp in ax.spines.values():
        sp.set_color('0.4')
    return q


def plot_sdc_warp(warp_file, b0_ref, out_file, n_slices=3, step=4, title=None):
    """Draw an SDC displacement field over a b=0, as 3D Slicer's glyphs would.

    Several slices are drawn in each of the two planes that contain the
    phase-encoding axis; the perpendicular plane would hide the displacement.

    Parameters
    ----------
    warp_file : str
        ITK displacement field.
    b0_ref : str
        b=0 image on the same grid, drawn underneath.
    out_file : str
        Output figure path.
    n_slices : int
        Slices per plane.
    step : int
        Voxels between arrows.
    title : str, optional
        Figure title.

    Returns
    -------
    str
        ``out_file``.
    """
    import matplotlib as mpl

    mpl.use('Agg')
    import matplotlib.pyplot as plt

    disp_ras, mag, affine = sdc_warp_glyph_field(warp_file)
    b0 = np.asarray(nb.load(b0_ref).dataobj, dtype=float)
    _ped, slice_axes, vox_to_ras = sdc_warp_display_planes(disp_ras, affine)
    min_mag, vmax, arrow_scale = sdc_warp_glyph_scale(mag, affine, step)
    plane_name = {0: 'sagittal', 1: 'coronal', 2: 'axial'}

    fig, axes = plt.subplots(
        len(slice_axes),
        n_slices,
        figsize=(3.6 * n_slices, 3.4 * len(slice_axes)),
        facecolor='black',
        squeeze=False,
    )
    q = None
    for row, slice_axis in enumerate(slice_axes):
        for col, sl in enumerate(_glyph_slice_positions(b0, mag, slice_axis, n_slices)):
            panel = _glyph_slice(disp_ras, mag, b0, affine, slice_axis, sl, vox_to_ras)
            q = _draw_glyph(axes[row][col], panel, (0.0, vmax), step, min_mag, arrow_scale)
        axes[row][0].set_ylabel(plane_name[int(vox_to_ras[slice_axis])], color='white')
    title = title or ''
    if arrow_scale > 1:
        title += (
            f'\narrows drawn {arrow_scale:.{0 if arrow_scale >= 10 else 1}f}x longer; '
            'color shows the true size'
        )
    fig.suptitle(title, color='white')
    cb = fig.colorbar(q, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label('|displacement| (mm)', color='white')
    cb.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cb.ax, 'yticklabels'), color='white')

    fig.savefig(out_file, bbox_inches='tight', facecolor='black')
    plt.close(fig)
    return out_file
