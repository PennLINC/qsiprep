"""Visualization utilities."""

from contextlib import contextmanager

from lxml import etree
from nilearn import plotting
from niworkflows.viz.utils import SVGNS, extract_svg, robust_set_limits, uuid4
from svgutils.transform import SVGFigure


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
