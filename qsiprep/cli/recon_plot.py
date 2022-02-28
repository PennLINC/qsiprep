#!/usr/bin/env python
import warnings
import os
import sys
import os.path as op
from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
import nibabel as nb
import numpy as np
from qsiprep.niworkflows.viz.utils import slices_from_bbox
from qsiprep.interfaces.converters import fib2amps, mif2amps
from dipy.core.sphere import HemiSphere
from dipy.core.ndindex import ndindex
from dipy.reconst.odf import gfa
from dipy.direction import peak_directions
from PIL import Image
from fury import actor, window
from nipype import logging

LOGGER = logging.getLogger('nipype.interface')

warnings.filterwarnings("ignore", category=ImportWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)


def sink_mask_file(in_file, orig_file, out_dir):
    import os
    from nipype.utils.filemanip import fname_presuffix, copyfile
    os.makedirs(out_dir, exist_ok=True)
    out_file = fname_presuffix(orig_file, suffix='_mask', newpath=out_dir)
    copyfile(in_file, out_file, copy=True, use_hardlink=True)
    return out_file


def recon_plot():
    """Convert fib to mif."""
    parser = ArgumentParser(
        description='qsiprep: Convert DSI Studio fib file to MRtrix mif file.',
        formatter_class=RawTextHelpFormatter)

    parser.add_argument('--fib',
                        action='store',
                        type=os.path.abspath,
                        help='DSI Studio fib file to convert')
    parser.add_argument('--mif',
                        type=os.path.abspath,
                        action='store',
                        help='path to a MRtrix mif file')
    parser.add_argument('--amplitudes',
                        type=os.path.abspath,
                        action='store',
                        help='4D ampliudes corresponding to --directions')
    parser.add_argument('--directions',
                        type=os.path.abspath,
                        action='store',
                        help='text file of directions corresponding to --amplitudes')
    parser.add_argument('--mask_file',
                        action='store',
                        type=os.path.abspath,
                        help='a NIfTI-1 format file defining a brain mask.')
    parser.add_argument('--odf_rois',
                        action='store',
                        type=os.path.abspath,
                        help='a NIfTI-1 format file with ROIs for plotting ODFs')
    parser.add_argument('--peaks_image',
                        action='store',
                        default="peaks_mosiac.png",
                        type=os.path.abspath,
                        help='png file for odf peaks image')
    parser.add_argument('--odfs_image',
                        action='store',
                        default="odfs_mosaic.png",
                        type=os.path.abspath,
                        help='png file for odf results')
    parser.add_argument('--background_image',
                        action='store',
                        type=os.path.abspath,
                        help='a NIfTI-1 format file with a valid q/sform.')
    parser.add_argument('--subtract-iso',
                        action='store_true',
                        help='subtract ODF min so visualization looks similar in mrview')
    parser.add_argument('--peaks_only',
                        action='store_true',
                        help='only plot the peaks')
    parser.add_argument('--ncuts', type=int, default=3, 
                        help="number of slices to plot")
    parser.add_argument('--padding', type=int, default=10, 
                        help="number of slices to plot")
    opts = parser.parse_args()

    if opts.mif:
        odf_img, directions = mif2amps(opts.mif, os.getcwd())
        LOGGER.info("converting %s to plot ODF/peaks", opts.mif)
    elif opts.fib:
        odf_img, directions = fib2amps(opts.fib,
                                        opts.background_image,
                                        os.getcwd())
        LOGGER.info("converting %s to plot ODF/peaks", opts.fib)
    elif opts.amplitudes and opts.directions:
        LOGGER.info("loading amplitudes=%s, directions=%s "
                    "to plot ODF/peaks", opts.amplitudes, opts.directions)
        odf_img = nb.load(opts.amplitudes)
        directions = np.load(opts.directions)
    else:
        raise Exception('Requires either a mif file or fib file')

    odf_4d = odf_img.get_fdata()
    sphere = HemiSphere(xyz=directions.astype(np.float))
    if not opts.background_image:
        background_data = odf_4d.mean(3)
    else:
        background_data = nb.load(opts.background_image).get_fdata()

    LOGGER.info("saving peaks image to %s", opts.peaks_image)
    peak_slice_series(odf_4d, sphere, background_data, opts.peaks_image,
                        n_cuts=opts.ncuts, mask_image=opts.mask_file,
                        padding=opts.padding)

    # Plot ODFs in interesting regions
    if opts.odf_rois and not opts.peaks_only:
        LOGGER.info("saving odfs image to %s", opts.odfs_image)
        odf_roi_plot(odf_4d, sphere, background_data, opts.odfs_image, 
                     opts.odf_rois,
                     subtract_iso=opts.subtract_iso,
                     mask=opts.mask_file)
    sys.exit(0)


def plot_peak_slice(odf_4d, sphere, background_data, out_file, axis, slicenum, mask_data,
                    tile_size=1200, normalize_peaks=True):
    view_up = [(0., 0., 1.), (0., 0., 1.), (0., -1., 0.)]

    # Make a slice mask to reduce memory
    new_shape = list(odf_4d.shape)
    new_shape[axis] = 1
    image_shape = new_shape[:3]
    midpoint = (new_shape[0] / 2., new_shape[1] / 2., new_shape[2] / 2.)

    if axis == 0:
        odf_slice = odf_4d[slicenum, :, :, :].reshape(new_shape)
        image_slice = background_data[slicenum, :, :].reshape(image_shape)
        mask_slice = mask_data[slicenum, :, :].reshape(image_shape)
        camera_dist = max(midpoint[1], midpoint[2]) * np.pi
    elif axis == 1:
        odf_slice = odf_4d[:, slicenum, :, :].reshape(new_shape)
        image_slice = background_data[:, slicenum, :].reshape(image_shape)
        mask_slice = mask_data[:, slicenum, :].reshape(image_shape)
        camera_dist = max(midpoint[0], midpoint[2]) * np.pi
    elif axis == 2:
        odf_slice = odf_4d[:, :, slicenum, :].reshape(new_shape)
        image_slice = background_data[:, :, slicenum].reshape(image_shape)
        mask_slice = mask_data[:, :, slicenum].reshape(image_shape)
        camera_dist = max(midpoint[0], midpoint[1]) * np.pi
    position = list(midpoint)
    position[axis] += camera_dist

    # Find the actual peaks
    peak_dirs, peak_values = peaks_from_odfs(odf_slice, sphere,
                                             relative_peak_threshold=.1,
                                             min_separation_angle=15,
                                             mask=mask_slice,
                                             normalize_peaks=normalize_peaks,
                                             npeaks=3)
    if normalize_peaks:
        peak_values = peak_values / peak_values.max() * np.pi
    peak_actor = actor.peak_slicer(peak_dirs, peak_values, colors=None)
    image_actor = actor.slicer(image_slice, opacity=0.6, interpolation='nearest')
    image_size = (tile_size, tile_size)
    scene = window.Scene()
    scene.add(image_actor)
    scene.add(peak_actor)

    xfov_min, xfov_max = 0, new_shape[0] - 1
    yfov_min, yfov_max = 0, new_shape[1] - 1
    zfov_min, zfov_max = 0, new_shape[2] - 1
    peak_actor.display_extent(xfov_min, xfov_max, yfov_min, yfov_max, zfov_min, zfov_max)
    image_actor.display_extent(xfov_min, xfov_max, yfov_min, yfov_max, zfov_min, zfov_max)
    scene.set_camera(focal_point=tuple(midpoint),
                     position=tuple(position),
                     view_up=view_up[axis])
    window.record(scene, out_path=out_file, reset_camera=False, size=image_size)


def peak_slice_series(odf_4d, sphere, background_data, out_file, mask_image=None,
                      prefix='odf', tile_size=1200, n_cuts=3, padding=4,
                      normalize_peaks=True):

    # Make a slice mask to reduce memory
    if mask_image is None:
        LOGGER.info("No mask image for plotting peaks")
        image_mask = np.ones(background_data.shape)
    else:
        image_mask = nb.load(mask_image).get_fdata()

    slice_indices = slices_from_bbox(background_data, cuts=n_cuts, padding=padding)
    LOGGER.info("Plotting slice indices %s", slice_indices)
    # Render the axial slices
    z_image = Image.new('RGB', (tile_size, tile_size * n_cuts))
    for slicenum, z_slice in enumerate(slice_indices['z']):
        png_file = '{}_tra_{:03d}.png'.format(prefix, z_slice)
        plot_peak_slice(odf_4d, sphere, background_data, png_file, 2, z_slice, image_mask,
                        tile_size, normalize_peaks)
        z_image.paste(Image.open(png_file), (0, slicenum * tile_size))

    # Render the sagittal slices
    x_image = Image.new('RGB', (tile_size, tile_size * n_cuts))
    for slicenum, x_slice in enumerate(slice_indices['x']):
        png_file = '{}_sag_{:03d}.png'.format(prefix, x_slice)
        plot_peak_slice(odf_4d, sphere, background_data, png_file, 0, x_slice, image_mask,
                        tile_size, normalize_peaks)
        x_image.paste(Image.open(png_file), (0, slicenum * tile_size))

    # Render the coronal slices
    y_image = Image.new('RGB', (tile_size, tile_size * n_cuts))
    for slicenum, y_slice in enumerate(slice_indices['y']):
        png_file = '{}_cor_{:03d}.png'.format(prefix, y_slice)
        plot_peak_slice(odf_4d, sphere, background_data, png_file, 1, y_slice, image_mask,
                        tile_size, normalize_peaks)
        y_image.paste(Image.open(png_file), (0, slicenum * tile_size))

    final_image = Image.new('RGB', (tile_size * 3, tile_size * n_cuts))
    final_image.paste(z_image, (0, 0))
    final_image.paste(x_image, (tile_size, 0))
    final_image.paste(y_image, (tile_size * 2, 0))
    final_image.save(out_file)


def peaks_from_odfs(odf4d, sphere, relative_peak_threshold,
                    min_separation_angle, mask=None,
                    gfa_thr=0, normalize_peaks=False,
                    npeaks=5):

    shape = odf4d.shape[:-1]
    if mask is None:
        mask = np.ones(shape, dtype='bool')
    else:
        if mask.shape != shape:
            raise ValueError("Mask is not the same shape as data.")

    gfa_array = np.zeros(shape)
    qa_array = np.zeros((shape + (npeaks,)))

    peak_dirs = np.zeros((shape + (npeaks, 3)))
    peak_values = np.zeros((shape + (npeaks,)))
    peak_indices = np.zeros((shape + (npeaks,)), dtype='int')
    peak_indices.fill(-1)

    global_max = -np.inf
    for idx in ndindex(shape):
        if not mask[idx]:
            continue
        odf = odf4d[idx]
        gfa_array[idx] = gfa(odf)
        if gfa_array[idx] < gfa_thr:
            global_max = max(global_max, odf.max())
            continue
        # Get peaks of odf
        direction, pk, ind = peak_directions(odf, sphere,
                                             relative_peak_threshold,
                                             min_separation_angle)
        # Calculate peak metrics
        if pk.shape[0] != 0:
            global_max = max(global_max, pk[0])
            n = min(npeaks, pk.shape[0])
            qa_array[idx][:n] = pk[:n] - odf.min()
            peak_dirs[idx][:n] = direction[:n]
            peak_indices[idx][:n] = ind[:n]
            peak_values[idx][:n] = pk[:n]

            if normalize_peaks:
                peak_values[idx][:n] /= pk[0]
                peak_dirs[idx] *= peak_values[idx][:, None]

    qa_array /= global_max

    return peak_dirs, peak_values


def get_camera_for_roi(roi_data, roi_id, view_axis):
    voxel_coords = np.row_stack(np.nonzero(roi_data == roi_id))
    centroid = voxel_coords.mean(1)
    other_axes = [[1, 2], [0, 2], [0, 1]][view_axis]
    projected_x = voxel_coords[other_axes[0]]
    projected_y = voxel_coords[other_axes[1]]
    xspan = projected_x.max() - projected_x.min()
    yspan = projected_y.max() - projected_y.min()
    camera_distance = max(xspan, yspan) * np.pi
    return centroid, camera_distance


def plot_an_odf_slice(odf_4d, full_sphere, background_data, tile_size, filename,
                      centroid, axis, camera_distance, subtract_iso, mask_image):
    view_up = [(0., 0., 1.), (0., 0., 1.), (0., -1., 0.)]

    # Adjust the centroid so it's only a single slice
    slicenum = int(np.round(centroid)[axis])
    centroid[axis] = 0
    position = centroid.copy()
    position[axis] = position[axis] + camera_distance

    # Roll if viewing an axial slice
    roll = 3 if axis == 2 else 0
    position[1] = position[1] - roll

    # Ensure the dimensions reflect that there is only one slice
    new_shape = list(odf_4d.shape)
    new_shape[axis] = 1
    image_shape = new_shape[:3]
    if axis == 0:
        odf_slice = odf_4d[slicenum, :, :, :].reshape(new_shape)
        image_slice = background_data[slicenum, :, :].reshape(image_shape)
    elif axis == 1:
        odf_slice = odf_4d[:, slicenum, :, :].reshape(new_shape)
        image_slice = background_data[:, slicenum, :].reshape(image_shape)
    elif axis == 2:
        odf_slice = odf_4d[:, :, slicenum, :].reshape(new_shape)
        image_slice = background_data[:, :, slicenum].reshape(image_shape)

    # Tile to get the whole ODF
    odf_slice = np.tile(odf_slice, (1, 1, 1, 2))
    if subtract_iso:
        odf_slice = odf_slice - odf_slice.min(3, keepdims=True)
    # Make graphics objects
    odf_actor = actor.odf_slicer(odf_slice, sphere=full_sphere,
                                 colormap=None, scale=0.6, mask=image_slice)
    image_actor = actor.slicer(image_slice, opacity=0.6, interpolation='nearest')
    image_size = (tile_size, tile_size)
    scene = window.Scene()
    scene.add(image_actor)
    scene.add(odf_actor)
    xfov_min, xfov_max = 0, new_shape[0] - 1
    yfov_min, yfov_max = 0, new_shape[1] - 1
    zfov_min, zfov_max = 0, new_shape[2] - 1
    odf_actor.display_extent(xfov_min, xfov_max, yfov_min, yfov_max, zfov_min, zfov_max)
    image_actor.display_extent(xfov_min, xfov_max, yfov_min, yfov_max, zfov_min, zfov_max)
    scene.set_camera(focal_point=tuple(centroid),
                     position=tuple(position),
                     view_up=view_up[axis])
    window.record(scene, out_path=filename, reset_camera=False, size=image_size)
    scene.clear()


def odf_roi_plot(odf_4d, halfsphere, background_data, out_file, roi_file,
                 prefix='odf', tile_size=1200, subtract_iso=False, mask=None):

    roi_data = nb.load(roi_file).get_fdata()
    roi_image = Image.new('RGB', (tile_size * 3, tile_size))
    roi1_centroid, roi1_distance = get_camera_for_roi(roi_data, 1, 2)
    roi2_centroid, roi2_distance = get_camera_for_roi(roi_data, 2, 1)
    roi3_centroid, roi3_distance = get_camera_for_roi(roi_data, 3, 1)

    camera_distance = max(roi1_distance, roi2_distance, roi3_distance)
    # Make a slice mask to reduce memory
    if mask is None:
        image_mask = np.ones(roi_data.shape)
    else:
        image_mask = nb.load(mask).get_fdata()

    # Fill out the other half of the sphere
    odf_sphere = halfsphere.mirror()
    semiovale_axial_file = '{}_semoivale_axial.png'.format(prefix)
    plot_an_odf_slice(odf_4d, odf_sphere, background_data, tile_size,
                      semiovale_axial_file, centroid=roi1_centroid, axis=2,
                      camera_distance=camera_distance, subtract_iso=subtract_iso,
                      mask_image=image_mask)
    roi_image.paste(Image.open(semiovale_axial_file), (0, 0))

    # Render the coronal slice with a double-crossing
    cst_x_cc_file = '{}_CSTxCC.png'.format(prefix)
    plot_an_odf_slice(odf_4d, odf_sphere, background_data, tile_size,
                      cst_x_cc_file, centroid=roi2_centroid, axis=1,
                      camera_distance=camera_distance, subtract_iso=subtract_iso,
                      mask_image=image_mask)
    roi_image.paste(Image.open(cst_x_cc_file), (tile_size, 0))

    # Render the corpus callosum
    cc_file = '{}_CC.png'.format(prefix)
    plot_an_odf_slice(odf_4d, odf_sphere, background_data, tile_size,
                      cc_file, centroid=roi3_centroid, axis=1,
                      camera_distance=camera_distance, subtract_iso=subtract_iso,
                      mask_image=image_mask)
    roi_image.paste(Image.open(cc_file), (2 * tile_size, 0))

    roi_image.save(out_file)