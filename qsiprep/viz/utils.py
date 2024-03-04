"""Visualization utilities."""

import numpy as np
from lxml import etree
from nilearn.plotting import plot_anat
from niworkflows.viz.utils import SVGNS, extract_svg, robust_set_limits, uuid4
from svgutils.transform import SVGFigure


def plot_denoise(
    lowb_nii,
    highb_nii,
    div_id,
    plot_params=None,
    highb_plot_params=None,
    order=("z", "x", "y"),
    cuts=None,
    estimate_brightness=False,
    label=None,
    lowb_contour=None,
    highb_contour=None,
    upper_label_suffix=": low-b",
    lower_label_suffix=": high-b",
    compress="auto",
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
    if estimate_brightness:
        plot_params = robust_set_limits(
            lowb_nii.get_fdata(dtype="float32").reshape(-1), plot_params
        )
    # Plot each cut axis for low-b
    for i, mode in enumerate(list(order)):
        plot_params["display_mode"] = mode
        plot_params["cut_coords"] = cuts[mode]
        if i == 0:
            plot_params["title"] = label + upper_label_suffix
        else:
            plot_params["title"] = None

        # Generate nilearn figure
        display = plot_anat(lowb_nii, **plot_params)
        if lowb_contour is not None:
            display.add_contours(lowb_contour, linewidths=1)

        svg = extract_svg(display, compress=compress)
        display.close()

        # Find and replace the figure_1 id.
        xml_data = etree.fromstring(svg)
        find_text = etree.ETXPath("//{%s}g[@id='figure_1']" % SVGNS)
        find_text(xml_data)[0].set("id", "%s-%s-%s" % (div_id, mode, uuid4()))

        svg_fig = SVGFigure()
        svg_fig.root = xml_data
        out_files.append(svg_fig)

    # Plot each cut axis for high-b
    if estimate_brightness:
        highb_plot_params = robust_set_limits(
            highb_nii.get_fdata(dtype="float32").reshape(-1), highb_plot_params
        )
    for i, mode in enumerate(list(order)):
        highb_plot_params["display_mode"] = mode
        highb_plot_params["cut_coords"] = cuts[mode]
        if i == 0:
            highb_plot_params["title"] = label + lower_label_suffix
        else:
            highb_plot_params["title"] = None

        # Generate nilearn figure
        display = plot_anat(highb_nii, **highb_plot_params)
        if highb_contour is not None:
            display.add_contours(highb_contour, linewidths=1)

        svg = extract_svg(display, compress=compress)
        display.close()

        # Find and replace the figure_1 id.
        xml_data = etree.fromstring(svg)
        find_text = etree.ETXPath("//{%s}g[@id='figure_1']" % SVGNS)
        find_text(xml_data)[0].set("id", "%s-%s-%s" % (div_id, mode, uuid4()))

        svg_fig = SVGFigure()
        svg_fig.root = xml_data
        out_files.append(svg_fig)

    return out_files


def plot_acpc(
    acpc_registered_img,
    div_id,
    plot_params=None,
    order=("z", "x", "y"),
    cuts=None,
    estimate_brightness=False,
    label=None,
    compress="auto",
):
    """
    Plot the results of an AC-PC transformation.
    """
    plot_params = plot_params or {}

    # Do the low-b image first
    out_files = []
    if estimate_brightness:
        plot_params = robust_set_limits(
            acpc_registered_img.get_fdata(dtype="float32").reshape(-1), plot_params
        )

    # Plot each cut axis for low-b
    for i, mode in enumerate(list(order)):
        plot_params["display_mode"] = mode
        plot_params["cut_coords"] = [-20.0, 0.0, 20.0]
        if i == 0:
            plot_params["title"] = label
        else:
            plot_params["title"] = None

        # Generate nilearn figure
        display = plot_anat(acpc_registered_img, **plot_params)
        for coord, axis in display.axes.items():
            axis.ax.axvline(0, lw=1)
            axis.ax.axhline(0, lw=1)
        svg = extract_svg(display, compress=compress)
        display.close()

        # Find and replace the figure_1 id.
        xml_data = etree.fromstring(svg)
        find_text = etree.ETXPath("//{%s}g[@id='figure_1']" % SVGNS)
        find_text(xml_data)[0].set("id", "%s-%s-%s" % (div_id, mode, uuid4()))

        svg_fig = SVGFigure()
        svg_fig.root = xml_data
        out_files.append(svg_fig)

    return out_files


def slices_from_bbox(mask_data, cuts=3, padding=0):
    """Finds equi-spaced cuts for presenting images"""
    B = np.argwhere(mask_data > 0)
    start_coords = B.min(0)
    stop_coords = B.max(0) + 1

    vox_coords = []
    for start, stop in zip(start_coords, stop_coords):
        inc = abs(stop - start) / (cuts + 2 * padding + 1)
        slices = [start + (i + 1) * inc for i in range(cuts + 2 * padding)]
        vox_coords.append(slices[padding:-padding])
    return {k: [int(_v) for _v in v] for k, v in zip(["x", "y", "z"], vox_coords)}
