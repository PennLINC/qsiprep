# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Helper tools for visualization purposes"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import os.path as op
import subprocess
import base64
import re
from sys import version_info
from uuid import uuid4
from io import open, StringIO

import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt

from lxml import etree
from nilearn import image as nlimage
from nilearn.plotting import plot_anat
from svgutils.transform import SVGFigure
from seaborn import color_palette

from .. import NIWORKFLOWS_LOG
from nipype.utils import filemanip

try:
    from shutil import which
except ImportError:

    def which(cmd):
        """
        A homemade which command

        >>> from qsiprep.niworkflows.viz.utils import which
        >>> which('ls')
        True
        >>> which('madeoutcommand')
        False

        """

        try:
            subprocess.run([cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           close_fds=True)
        except OSError as e:
            from errno import ENOENT
            if e.errno == ENOENT:
                return False
            raise e
        return True


SVGNS = "http://www.w3.org/2000/svg"
PY3 = version_info[0] > 2

# Patch subprocess in python 2
if not hasattr(subprocess, 'DEVNULL'):
    setattr(subprocess, 'DEVNULL', -3)

if not hasattr(subprocess, 'run'):
    def _run(args, input=None, stdout=None, stderr=None, shell=False, check=False,
             close_fds=False):
        from collections import namedtuple

        devnull = open(os.devnull, 'r+')
        stdin = subprocess.PIPE if input is not None else None

        if stdout == subprocess.DEVNULL:
            stdout = devnull

        if stderr == subprocess.DEVNULL:
            stderr = devnull

        proc = subprocess.Popen(args, stdout=stdout, shell=shell, stdin=stdin,
                                close_fds=close_fds)
        result = namedtuple('CompletedProcess', 'stdout stderr')
        res = result(*proc.communicate(input=input))

        devnull.close()

        if check and proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, args)

        return res
    setattr(subprocess, 'run', _run)


def robust_set_limits(data, plot_params):
    plot_params['vmin'] = plot_params.get(
        'vmin', np.percentile(data, 15))
    plot_params['vmax'] = plot_params.get(
        'vmax', np.percentile(data, 99.8))
    return plot_params


def svg_compress(image, compress='auto'):
    ''' takes an image as created by nilearn.plotting and returns a blob svg.
    Performs compression (can be disabled). A bit hacky. '''

    # Check availability of svgo and cwebp
    has_compress = all((which('svgo'), which('cwebp')))
    if compress is True and not has_compress:
        raise RuntimeError('Compression is required, but svgo or cwebp are not installed')
    else:
        compress = (compress is True or compress == 'auto') and has_compress

    # Compress the SVG file using SVGO
    if compress:
        cmd = 'svgo -i - -o - -q -p 3 --pretty --disable=cleanupNumericValues'
        try:
            pout = subprocess.run(cmd, input=image.encode('utf-8'), stdout=subprocess.PIPE,
                                  shell=True, check=True, close_fds=True).stdout
        except OSError as e:
            from errno import ENOENT
            if compress is True and e.errno == ENOENT:
                raise e
        else:
            image = pout.decode('utf-8')

    # Convert all of the rasters inside the SVG file with 80% compressed WEBP
    if compress:
        new_lines = []
        with StringIO(image) as fp:
            for line in fp:
                if "image/png" in line:
                    tmp_lines = [line]
                    while "/>" not in line:
                        line = fp.readline()
                        tmp_lines.append(line)
                    content = ''.join(tmp_lines).replace('\n', '').replace(
                        ',  ', ',')

                    left = content.split('base64,')[0] + 'base64,'
                    left = left.replace("image/png", "image/webp")
                    right = content.split('base64,')[1]
                    png_b64 = right.split('"')[0]
                    right = '"' + '"'.join(right.split('"')[1:])

                    cmd = "cwebp -quiet -noalpha -q 80 -o - -- -"
                    pout = subprocess.run(
                        cmd, input=base64.b64decode(png_b64), shell=True,
                        stdout=subprocess.PIPE, check=True, close_fds=True).stdout
                    webpimg = base64.b64encode(pout).decode('utf-8')
                    new_lines.append(left + webpimg + right)
                else:
                    new_lines.append(line)
        lines = new_lines
    else:
        lines = image.splitlines()

    svg_start = 0
    for i, line in enumerate(lines):
        if '<svg ' in line:
            svg_start = i
            continue

    image_svg = lines[svg_start:]  # strip out extra DOCTYPE, etc headers
    return ''.join(image_svg)  # straight up giant string


def svg2str(display_object, dpi=300):
    """
    Serializes a nilearn display object as a string
    """
    from io import StringIO
    image_buf = StringIO()
    display_object.frame_axes.figure.savefig(
        image_buf, dpi=dpi, format='svg',
        facecolor='k', edgecolor='k')
    return image_buf.getvalue()


def extract_svg(display_object, dpi=300, compress='auto'):
    """
    Removes the preamble of the svg files generated with nilearn
    """
    image_svg = svg2str(display_object, dpi)
    if compress is True or compress == 'auto':
        image_svg = svg_compress(image_svg, compress)
    image_svg = re.sub(' height="[0-9]+[a-z]*"', '', image_svg, count=1)
    image_svg = re.sub(' width="[0-9]+[a-z]*"', '', image_svg, count=1)
    image_svg = re.sub(' viewBox',
                       ' preseveAspectRation="xMidYMid meet" viewBox',
                       image_svg, count=1)
    start_tag = '<svg '
    start_idx = image_svg.find(start_tag)
    end_tag = '</svg>'
    end_idx = image_svg.rfind(end_tag)
    if start_idx is -1 or end_idx is -1:
        NIWORKFLOWS_LOG.info('svg tags not found in extract_svg')
    # rfind gives the start index of the substr. We want this substr
    # included in our return value so we add its length to the index.
    end_idx += len(end_tag)
    return image_svg[start_idx:end_idx]


def cuts_from_bbox(mask_nii, cuts=3):
    """Finds equi-spaced cuts for presenting images"""
    from nibabel.affines import apply_affine
    mask_data = mask_nii.get_data()
    B = np.argwhere(mask_data > 0)
    start_coords = B.min(0)
    stop_coords = B.max(0) + 1

    vox_coords = []
    for start, stop in zip(start_coords, stop_coords):
        inc = abs(stop - start) / (cuts + 1)
        vox_coords.append([start + (i + 1) * inc for i in range(cuts)])

    ras_coords = []
    for cross in np.array(vox_coords).T:
        ras_coords.append(apply_affine(mask_nii.affine, cross).tolist())

    ras_cuts = [list(coords) for coords in np.transpose(ras_coords)]
    return {k: v for k, v in zip(['x', 'y', 'z'], ras_cuts)}


def _3d_in_file(in_file):
    ''' if self.inputs.in_file is 3d, return it.
    if 4d, pick an arbitrary volume and return that.

    if in_file is a list of files, return an arbitrary file from
    the list, and an arbitrary volume from that file
    '''

    in_file = filemanip.filename_to_list(in_file)[0]

    try:
        in_file = nb.load(in_file)
    except AttributeError:
        in_file = in_file

    if in_file.get_data().ndim == 3:
        return in_file

    return nlimage.index_img(in_file, 0)


def plot_segs(image_nii, seg_niis, out_file, bbox_nii=None, masked=False,
              colors=None, compress='auto', **plot_params):
    """ plot segmentation as contours over the image (e.g. anatomical).
    seg_niis should be a list of files. mask_nii helps determine the cut
    coordinates. plot_params will be passed on to nilearn plot_* functions. If
    seg_niis is a list of size one, it behaves as if it was plotting the mask.
    """
    plot_params = {} if plot_params is None else plot_params

    image_nii = _3d_in_file(image_nii)
    data = image_nii.get_data()

    plot_params = robust_set_limits(data, plot_params)

    bbox_nii = nb.load(image_nii if bbox_nii is None else bbox_nii)
    if masked:
        bbox_nii = nlimage.threshold_img(bbox_nii, 1e-3)

    cuts = cuts_from_bbox(bbox_nii, cuts=7)
    plot_params['colors'] = colors or plot_params.get('colors', None)
    out_files = []
    for d in plot_params.pop('dimensions', ('z', 'x', 'y')):
        plot_params['display_mode'] = d
        plot_params['cut_coords'] = cuts[d]
        svg = _plot_anat_with_contours(image_nii, segs=seg_niis, compress=compress,
                                       **plot_params)

        # Find and replace the figure_1 id.
        try:
            xml_data = etree.fromstring(svg)
        except etree.XMLSyntaxError as e:
            NIWORKFLOWS_LOG.info(e)
            return
        find_text = etree.ETXPath("//{%s}g[@id='figure_1']" % SVGNS)
        find_text(xml_data)[0].set('id', 'segmentation-%s-%s' % (d, uuid4()))

        svg_fig = SVGFigure()
        svg_fig.root = xml_data
        out_files.append(svg_fig)

    return out_files


def _plot_anat_with_contours(image, segs=None, compress='auto',
                             **plot_params):
    nsegs = len(segs or [])
    plot_params = plot_params or {}
    # plot_params' values can be None, however they MUST NOT
    # be None for colors and levels from this point on.
    colors = plot_params.pop('colors', None) or []
    levels = plot_params.pop('levels', None) or []
    missing = nsegs - len(colors)
    if missing > 0:  # missing may be negative
        colors = colors + color_palette("husl", missing)

    colors = [[c] if not isinstance(c, list) else c
              for c in colors]

    if not levels:
        levels = [[0.5]] * nsegs

    # anatomical
    display = plot_anat(image, **plot_params)

    # remove plot_anat -specific parameters
    plot_params.pop('display_mode')
    plot_params.pop('cut_coords')

    plot_params['linewidths'] = 0.5
    for i in reversed(range(nsegs)):
        plot_params['colors'] = colors[i]
        display.add_contours(segs[i], levels=levels[i],
                             **plot_params)

    svg = extract_svg(display, compress=compress)
    display.close()
    return svg


def plot_registration(anat_nii, div_id, plot_params=None,
                      order=('z', 'x', 'y'), cuts=None,
                      estimate_brightness=False, label=None, contour=None,
                      compress='auto'):
    """
    Plots the foreground and background views
    Default order is: axial, coronal, sagittal
    """
    plot_params = {} if plot_params is None else plot_params

    # Use default MNI cuts if none defined
    if cuts is None:
        raise NotImplementedError  # TODO

    out_files = []
    if estimate_brightness:
        plot_params = robust_set_limits(anat_nii.get_data().reshape(-1),
                                        plot_params)

    # FreeSurfer ribbon.mgz
    ribbon = contour is not None and np.array_equal(
        np.unique(contour.get_data()), [0, 2, 3, 41, 42])

    if ribbon:
        contour_data = contour.get_data() % 39
        white = nlimage.new_img_like(contour, contour_data == 2)
        pial = nlimage.new_img_like(contour, contour_data >= 2)

    # dual mask
    dual_mask = contour is not None and np.array_equal(
        np.unique(contour.get_data().astype(np.uint8)), [0, 1, 2])

    if dual_mask:
        contour_data = contour.get_data()
        outer_mask = nlimage.new_img_like(contour, contour_data == 1)
        inner_mask = nlimage.new_img_like(contour, contour_data == 2)
        all_mask = nlimage.new_img_like(contour, contour_data > 0)

    # Plot each cut axis
    for i, mode in enumerate(list(order)):
        plot_params['display_mode'] = mode
        plot_params['cut_coords'] = cuts[mode]
        if i == 0:
            plot_params['title'] = label
        else:
            plot_params['title'] = None

        # Generate nilearn figure
        display = plot_anat(anat_nii, **plot_params)
        if ribbon:
            kwargs = {'levels': [0.5], 'linewidths': 0.5}
            display.add_contours(white, colors='b', **kwargs)
            display.add_contours(pial, colors='r', **kwargs)
        elif dual_mask:
            kwargs = {'levels': [0.5], 'linewidths': 0.75}
            display.add_contours(inner_mask, colors='b', **kwargs)
            display.add_contours(outer_mask, colors='r', **kwargs)
            display.add_contours(all_mask, colors='c', **kwargs)
        elif contour is not None:
            display.add_contours(contour, colors='b', levels=[0.5],
                                 linewidths=0.5)

        svg = extract_svg(display, compress=compress)
        display.close()

        # Find and replace the figure_1 id.
        try:
            xml_data = etree.fromstring(svg)
        except etree.XMLSyntaxError as e:
            NIWORKFLOWS_LOG.info(e)
            return
        find_text = etree.ETXPath("//{%s}g[@id='figure_1']" % SVGNS)
        find_text(xml_data)[0].set('id', '%s-%s-%s' % (div_id, mode, uuid4()))

        svg_fig = SVGFigure()
        svg_fig.root = xml_data
        out_files.append(svg_fig)

    return out_files


def plot_denoise(lowb_nii, highb_nii, div_id, plot_params=None, highb_plot_params=None,
                 order=('z', 'x', 'y'), cuts=None,
                 estimate_brightness=False, label=None, lowb_contour=None,
                 highb_contour=None,
                 compress='auto', overlay=None, overlay_params=None):
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
            lowb_nii.get_fdata(dtype='float32').reshape(-1),
            plot_params)
    # Plot each cut axis for low-b
    for i, mode in enumerate(list(order)):
        plot_params['display_mode'] = mode
        plot_params['cut_coords'] = cuts[mode]
        if i == 0:
            plot_params['title'] = label + ": low-b"
        else:
            plot_params['title'] = None

        # Generate nilearn figure
        display = plot_anat(lowb_nii, **plot_params)
        if lowb_contour is not None:
            display.add_contours(lowb_contour, linewidths=1)

        svg = extract_svg(display, compress=compress)
        display.close()

        # Find and replace the figure_1 id.
        xml_data = etree.fromstring(svg)
        find_text = etree.ETXPath("//{%s}g[@id='figure_1']" % SVGNS)
        find_text(xml_data)[0].set('id', '%s-%s-%s' % (div_id, mode, uuid4()))

        svg_fig = SVGFigure()
        svg_fig.root = xml_data
        out_files.append(svg_fig)

    # Plot each cut axis for high-b
    if estimate_brightness:
        highb_plot_params = robust_set_limits(
            highb_nii.get_fdata(dtype='float32').reshape(-1),
            highb_plot_params)
    for i, mode in enumerate(list(order)):
        highb_plot_params['display_mode'] = mode
        highb_plot_params['cut_coords'] = cuts[mode]
        if i == 0:
            highb_plot_params['title'] = label + ': high-b'
        else:
            highb_plot_params['title'] = None

        # Generate nilearn figure
        display = plot_anat(highb_nii, **highb_plot_params)
        if highb_contour is not None:
            display.add_contours(highb_contour, linewidths=1)

        svg = extract_svg(display, compress=compress)
        display.close()

        # Find and replace the figure_1 id.
        xml_data = etree.fromstring(svg)
        find_text = etree.ETXPath("//{%s}g[@id='figure_1']" % SVGNS)
        find_text(xml_data)[0].set('id', '%s-%s-%s' % (div_id, mode, uuid4()))

        svg_fig = SVGFigure()
        svg_fig.root = xml_data
        out_files.append(svg_fig)

    return out_files


def plot_acpc(acpc_registered_img, div_id, plot_params=None,
              order=('z', 'x', 'y'), cuts=None,
              estimate_brightness=False, label=None,
              compress='auto'):
    """
    Plot the results of an AC-PC transformation.
    """
    plot_params = plot_params or {}

    # Do the low-b image first
    out_files = []
    if estimate_brightness:
        plot_params = robust_set_limits(
            acpc_registered_img.get_fdata(dtype='float32').reshape(-1),
            plot_params)

    # Plot each cut axis for low-b
    for i, mode in enumerate(list(order)):
        plot_params['display_mode'] = mode
        plot_params['cut_coords'] = [-20.0, 0.0, 20.0]
        if i == 0:
            plot_params['title'] = label
        else:
            plot_params['title'] = None

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
        find_text(xml_data)[0].set('id', '%s-%s-%s' % (div_id, mode, uuid4()))

        svg_fig = SVGFigure()
        svg_fig.root = xml_data
        out_files.append(svg_fig)

    return out_files


def compose_view(bg_svgs, fg_svgs, ref=0, out_file='report.svg'):
    """
    Composes the input svgs into one standalone svg and inserts
    the CSS code for the flickering animation
    """
    import svgutils.transform as svgt

    if fg_svgs is None:
        fg_svgs = []

    # Merge SVGs and get roots
    svgs = bg_svgs + fg_svgs
    roots = [f.getroot() for f in svgs]

    # Query the size of each
    sizes = []
    for f in svgs:
        viewbox = [float(v) for v in f.root.get("viewBox").split(" ")]
        width = int(viewbox[2])
        height = int(viewbox[3])
        sizes.append((width, height))
    nsvgs = len(bg_svgs)

    sizes = np.array(sizes)

    # Calculate the scale to fit all widths
    width = sizes[ref, 0]
    scales = width / sizes[:, 0]
    heights = sizes[:, 1] * scales

    # Compose the views panel: total size is the width of
    # any element (used the first here) and the sum of heights
    fig = svgt.SVGFigure(width, heights[:nsvgs].sum())

    yoffset = 0
    for i, r in enumerate(roots):
        r.moveto(0, yoffset, scale=scales[i])
        if i == (nsvgs - 1):
            yoffset = 0
        else:
            yoffset += heights[i]

    # Group background and foreground panels in two groups
    if fg_svgs:
        newroots = [
            svgt.GroupElement(roots[:nsvgs], {'class': 'background-svg'}),
            svgt.GroupElement(roots[nsvgs:], {'class': 'foreground-svg'})
        ]
    else:
        newroots = roots
    fig.append(newroots)
    fig.root.attrib.pop("width")
    fig.root.attrib.pop("height")
    fig.root.set("preserveAspectRatio", "xMidYMid meet")
    out_file = op.abspath(out_file)
    fig.save(out_file)

    # Post processing
    with open(out_file, 'r' if PY3 else 'rb') as f:
        svg = f.read().split('\n')

    # Remove <?xml... line
    if svg[0].startswith("<?xml"):
        svg = svg[1:]

    # Add styles for the flicker animation
    if fg_svgs:
        svg.insert(2, """\
<style type="text/css">
@keyframes flickerAnimation%s { 0%% {opacity: 1;} 100%% { opacity: 0; }}
.foreground-svg { animation: 1s ease-in-out 0s alternate none infinite paused flickerAnimation%s;}
.foreground-svg:hover { animation-play-state: running;}
</style>""" % tuple([uuid4()] * 2))

    with open(out_file, 'w' if PY3 else 'wb') as f:
        f.write('\n'.join(svg))
    return out_file


def transform_to_2d(data, max_axis):
    """
    Projects 3d data cube along one axis using maximum intensity with
    preservation of the signs. Adapted from nilearn.
    """
    import numpy as np
    # get the shape of the array we are projecting to
    new_shape = list(data.shape)
    del new_shape[max_axis]

    # generate a 3D indexing array that points to max abs value in the
    # current projection
    a1, a2 = np.indices(new_shape)
    inds = [a1, a2]
    inds.insert(max_axis, np.abs(data).argmax(axis=max_axis))

    # take the values where the absolute value of the projection
    # is the highest
    maximum_intensity_data = data[inds]

    return np.rot90(maximum_intensity_data)


def plot_melodic_components(melodic_dir, in_file, tr=None,
                            out_file='melodic_reportlet.svg',
                            compress='auto', report_mask=None,
                            noise_components_file=None):
    from nilearn.image import index_img, iter_img
    import nibabel as nb
    import numpy as np
    import seaborn as sns
    from matplotlib.gridspec import GridSpec
    import os
    import re
    from io import StringIO
    sns.set_style("white")
    current_palette = sns.color_palette()
    in_nii = nb.load(in_file)
    if not tr:
        tr = in_nii.header.get_zooms()[3]
        units = in_nii.header.get_xyzt_units()
        if units:
            if units[-1] == 'msec':
                tr = tr / 1000.0
            elif units[-1] == 'usec':
                tr = tr / 1000000.0
            elif units[-1] != 'sec':
                NIWORKFLOWS_LOG.warning('Unknown repetition time units '
                                        'specified - assuming seconds')
        else:
            NIWORKFLOWS_LOG.warning(
                'Repetition time units not specified - assuming seconds')

    from nilearn.input_data import NiftiMasker
    from nilearn.plotting import cm

    if not report_mask:
        nifti_masker = NiftiMasker(mask_strategy='epi')
        nifti_masker.fit(index_img(in_nii, range(2)))
        mask_img = nifti_masker.mask_img_
    else:
        mask_img = nb.load(report_mask)

    mask_sl = []
    for j in range(3):
        mask_sl.append(transform_to_2d(mask_img.get_data(), j))

    timeseries = np.loadtxt(os.path.join(melodic_dir, "melodic_mix"))
    power = np.loadtxt(os.path.join(melodic_dir, "melodic_FTmix"))
    stats = np.loadtxt(os.path.join(melodic_dir, "melodic_ICstats"))
    n_components = stats.shape[0]
    Fs = 1.0 / tr
    Ny = Fs / 2
    f = Ny * (np.array(list(range(1, power.shape[0] + 1)))) / (power.shape[0])

    n_rows = int((n_components + (n_components % 2)) / 2)
    fig = plt.figure(figsize=(6.5 * 1.5, n_rows * 0.85))
    gs = GridSpec(n_rows * 2, 9,
                  width_ratios=[1, 1, 1, 4, 0.001, 1, 1, 1, 4, ],
                  height_ratios=[1.1, 1] * n_rows)

    noise_components = None
    if noise_components_file:
        noise_components = np.loadtxt(noise_components_file,
                                      dtype=int, delimiter=',', ndmin=1)

    for i, img in enumerate(
            iter_img(os.path.join(melodic_dir, "melodic_IC.nii.gz"))):

        col = i % 2
        row = int(i / 2)
        l_row = row * 2

        # Set default colors
        color_title = 'k'
        color_time = current_palette[0]
        color_power = current_palette[1]

        if noise_components is not None and noise_components.size > 0:
            # If a noise components list is provided, assign red/green
            color_title = color_time = color_power = (
                'r' if (i + 1) in noise_components else 'g')

        data = img.get_data()
        for j in range(3):
            ax1 = fig.add_subplot(gs[l_row:l_row + 2, j + col * 5])
            sl = transform_to_2d(data, j)
            m = np.abs(sl).max()
            ax1.imshow(sl, vmin=-m, vmax=+m, cmap=cm.cold_white_hot,
                       interpolation="nearest")
            ax1.contour(mask_sl[j], levels=[0.5], colors='k', linewidths=0.5)
            plt.axis("off")
            ax1.autoscale_view('tight')
            if j == 0:
                ax1.set_title(
                    "C%d: Tot. var. expl. %.2g%%" % (i + 1, stats[i, 1]), x=0,
                    y=1.18, fontsize=7,
                    horizontalalignment='left',
                    verticalalignment='top',
                    color=color_title)

        ax2 = fig.add_subplot(gs[l_row, 3 + col * 5])
        ax3 = fig.add_subplot(gs[l_row + 1, 3 + col * 5])

        ax2.plot(np.arange(len(timeseries[:, i])) * tr, timeseries[:, i],
                 linewidth=min(200 / len(timeseries[:, i]), 1.0),
                 color=color_time)
        ax2.set_xlim([0, len(timeseries[:, i]) * tr])
        ax2.axes.get_yaxis().set_visible(False)
        ax2.autoscale_view('tight')
        ax2.tick_params(axis='both', which='major', pad=0)
        sns.despine(left=True, bottom=True)
        for tick in ax2.xaxis.get_major_ticks():
            tick.label.set_fontsize(6)
            tick.label.set_color(color_time)

        ax3.plot(f[0:], power[0:, i], color=color_power,
                 linewidth=min(100 / len(power[0:, i]), 1.0))
        ax3.set_xlim([f[0], f.max()])
        ax3.axes.get_yaxis().set_visible(False)
        ax3.autoscale_view('tight')
        ax3.tick_params(axis='both', which='major', pad=0)
        for tick in ax3.xaxis.get_major_ticks():
            tick.label.set_fontsize(6)
            tick.label.set_color(color_power)
        sns.despine(left=True, bottom=True)

    plt.subplots_adjust(hspace=0.5)

    image_buf = StringIO()
    fig.savefig(image_buf, dpi=300, format='svg', transparent=True,
                bbox_inches='tight', pad_inches=0.01)
    fig.clf()
    image_svg = image_buf.getvalue()

    if compress is True or compress == 'auto':
        image_svg = svg_compress(image_svg, compress)
    image_svg = re.sub(' height="[0-9]+[a-z]*"', '', image_svg, count=1)
    image_svg = re.sub(' width="[0-9]+[a-z]*"', '', image_svg, count=1)
    image_svg = re.sub(' viewBox',
                       ' preseveAspectRation="xMidYMid meet" viewBox',
                       image_svg, count=1)

    with open(out_file, 'w' if PY3 else 'wb') as f:
        f.write(image_svg)


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
    return {k: [int(_v) for _v in v] for k, v in zip(['x', 'y', 'z'], vox_coords)}
