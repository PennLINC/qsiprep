import base64
import os.path as op
from io import BytesIO
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from dipy.segment.mask import median_otsu
from nipype.utils.filemanip import save_json, load_json


def reorient_array(data, aff):
    # rearrange the matrix to RAS orientation
    orientation = nib.orientations.io_orientation(aff)
    data_RAS = nib.orientations.apply_orientation(data, orientation)
    # In RAS
    return nib.orientations.apply_orientation(
        data_RAS,
        nib.orientations.axcodes2ornt("IPL")
    )


def mplfig(data, outfile=None, as_bytes=False):
    fig = plt.figure(frameon=False, dpi=data.shape[0])
    fig.set_size_inches(float(data.shape[1])/data.shape[0], 1)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(data, aspect=1, cmap=plt.cm.Greys_r)  # previous aspect="normal"
    if outfile:
        fig.savefig(outfile, dpi=data.shape[0], transparent=True)
        plt.close()
        return outfile
    if as_bytes:
        IObytes = BytesIO()
        plt.savefig(IObytes, format='png', dpi=data.shape[0], transparent=True)
        IObytes.seek(0)
        base64_jpgData = base64.b64encode(IObytes.read())
        return base64_jpgData.decode("ascii")


def mplfigcontour(data, outfile=None, as_bytes=False):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(float(data.shape[1])/data.shape[0], 1)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    bg = np.zeros(data.shape)
    bg[:] = np.nan
    ax.imshow(bg, aspect=1, cmap=plt.cm.Greys_r)  # used to be aspect="normal"
    ax.contour(data, colors="red", linewidths=0.1)
    if outfile:
        fig.savefig(outfile, dpi=data.shape[0], transparent=True)
        plt.close()
        return outfile
    if as_bytes:
        IObytes = BytesIO()
        plt.savefig(IObytes, format='png', dpi=data.shape[0], transparent=True)
        IObytes.seek(0)
        base64_jpgData = base64.b64encode(IObytes.read())
        return base64_jpgData.decode("ascii")


def load_and_reorient(filename):
    img = nib.load(filename)
    data, aff = img.get_data(), img.affine
    data = reorient_array(data, aff)
    return data


def make_a_square(data_mat, include_last_dim=True):
    """Applies zero padding to make a 2d matrix a square.

    Examples:
    ---------
    >>> too_long = np.arange(4 * 7).reshape((4, 7))
    >>> long_squared = make_a_square(too_long)
    >>> long_squared.shape
    (7, 7)
    >>> long_squared.sum(1)
    array([  0,  21,  70, 119, 168,   0,   0])
    >>> too_tall = np.arange(6 * 5 * 3).reshape((6, 5, 3))
    >>> tall_squared = make_a_square(too_tall)
    >>> tall_squared.shape
    (6, 6, 6)
    >>> tall_2squared = make_a_square(too_tall, include_last_dim=False)
    >>> tall_2squared.shape
    (6, 6, 3)
    """
    shapes = data_mat.shape if include_last_dim else data_mat.shape[:-1]

    # Is it already square?
    if all([shape == shapes[0] for shape in shapes]):
        return data_mat
    n_dims_to_pad = len(shapes)
    largest_side = np.argmax(shapes)
    sides_to_pad = np.arange(n_dims_to_pad).tolist()
    sides_to_pad.pop(largest_side)

    # Must specify padding for all dims
    padding = [(0, 0)] * data_mat.ndim
    for side_to_pad in sides_to_pad:
        needed_padding = shapes[largest_side] - shapes[side_to_pad]
        left_pad = int(needed_padding // 2)
        right_pad = needed_padding - left_pad
        padding[side_to_pad] = (left_pad, right_pad)
    return np.pad(data_mat, padding, "constant", constant_values=(0, 0))


def nearest_square(limit):
    answer = 0
    while (answer + 1) ** 2 < limit:
        answer += 1
    if (answer ** 2) == limit:
        return answer
    else:
        return answer + 1


def create_sprite_from_tiles(tile, out_file=None, as_bytes=False):
    num_slices = tile.shape[-1]
    N = nearest_square(num_slices)
    M = int(np.ceil(num_slices/N))
    # tile is square, so just make a big arr
    pix = tile.shape[0]

    if len(tile.shape) == 3:
        mosaic = np.zeros((N*tile.shape[0], M*tile.shape[0]))
    else:
        mosaic = np.zeros((N*tile.shape[0], M*tile.shape[0], tile.shape[-2]))

    mosaic[:] = np.nan
    helper = np.arange(N*M).reshape((N, M))

    for t in range(num_slices):
        x, y = np.nonzero(helper == t)
        xmin = x[0] * pix
        xmax = (x[0] + 1) * pix
        ymin = y[0] * pix
        ymax = (y[0] + 1) * pix
        x_span = xmax - xmin
        y_span = ymax - ymin

        if len(tile.shape) == 3:
            mosaic[xmin:xmax, ymin:ymax] = tile[:x_span, :y_span, t]
        else:
            mosaic[xmin:xmax, ymin:ymax, :] = tile[:x_span, :y_span, :, t]

    if as_bytes:
        img = mplfig(mosaic, out_file, as_bytes=as_bytes)
        return dict(img=img, N=N, M=M, pix=pix, num_slices=num_slices)

    if out_file:
        img = mplfig(mosaic, out_file), N, M, pix, num_slices

    return dict(mosaic=mosaic, N=N, M=M, pix=pix, num_slices=num_slices)


def createSprite4D(dwi_file):

    # initialize output dict
    output = []

    # load the file
    dwi = load_and_reorient(dwi_file)[:, :, :, 1:]

    # create tiles from center slice on each orientation
    for orient in ['sag', 'ax', 'cor']:
        axis_tiles = get_middle_slice_tiles(dwi, orient)
        # create sprite images for the axis
        results = embed_tiles_in_json_sprite(axis_tiles, as_bytes=True)
        results['img_type'] = '4dsprite'
        results['orientation'] = orient
        output.append(results)

    return output


def square_and_normalize_slice(slice2d):
    tile_data = make_a_square(slice2d)
    max_value = np.percentile(tile_data, 98)
    tile_data[tile_data > max_value] = max_value
    return tile_data / max_value


def embed_tiles_in_json_sprite(tile_list, as_bytes=True, out_file=None):
    """Make a big rectangle containing the images for a brainsprite.

    Parameters:
    -----------

        tile_list : list
          List of 2d square numpy arrays to stick in a mosaic

    Returns:
    --------

        mosaic : np.ndarray
            Mosaic of tile images

    """
    # Tiles are squares
    tile_size = tile_list[0].shape[0]
    num_tiles = len(tile_list)
    num_tile_rows = nearest_square(num_tiles)
    num_tile_cols = int(np.ceil(num_tiles/num_tile_rows))
    mosaic = np.zeros((num_tile_rows * tile_size,
                       num_tile_cols * tile_size))

    i_indices, j_indices = np.unravel_index(np.arange(num_tiles),
                                            (num_tile_rows, num_tile_cols))
    i_tile_offsets = tile_size * i_indices
    j_tile_offsets = tile_size * j_indices

    for tile, i_offset, j_offset in zip(tile_list, i_tile_offsets,
                                        j_tile_offsets):
        mosaic[i_offset:(i_offset + tile_size),
               j_offset:(j_offset + tile_size)] = tile

    if as_bytes:
        img = mplfig(mosaic, out_file, as_bytes=as_bytes)
        return dict(img=img, N=num_tile_rows, M=num_tile_cols,
                    pix=tile_size, num_slices=num_tiles)

    return dict(mosaic=mosaic, N=num_tile_rows, M=num_tile_cols,
                pix=tile_size, num_slices=num_tiles)


def get_middle_slice_tiles(data, slice_direction):
    """Create a strip of intensity-normalized, square middle slices.

    """
    slicer = {"ax": 0, "cor": 1, "sag": 2}
    all_data_slicer = [slice(None), slice(None), slice(None)]
    num_slices = data.shape[slicer[slice_direction]]
    slice_num = int(num_slices / 2)
    all_data_slicer[slicer[slice_direction]] = slice_num
    middle_slices = data[tuple(all_data_slicer)]
    num_slices = middle_slices.shape[2]
    slice_tiles = [square_and_normalize_slice(middle_slices[..., mid_slice])
                   for mid_slice in range(num_slices)]

    return slice_tiles


def createB0_ColorFA_Mask_Sprites(b0_file, colorFA_file, mask_file):
    colorfa = make_a_square(load_and_reorient(colorFA_file), include_last_dim=False)
    b0 = make_a_square(load_and_reorient(b0_file)[:, :, :, 0])
    anat_mask = make_a_square(load_and_reorient(mask_file))

    # make a b0 sprite
    _, mask = median_otsu(b0)
    outb0 = create_sprite_from_tiles(b0, as_bytes=True)
    outb0['img_type'] = 'brainsprite'

    # make a colorFA sprite, masked by b0
    Q = make_a_square(colorfa, include_last_dim=False)
    Q[np.logical_not(mask)] = np.nan
    Q = np.moveaxis(Q,  -2, -1)
    outcolorFA = create_sprite_from_tiles(Q, as_bytes=True)
    outcolorFA['img_type'] = 'brainsprite'

    # make an anat mask contour sprite
    outmask = create_sprite_from_tiles(
        make_a_square(anat_mask, include_last_dim=False))
    img = mplfigcontour(outmask.pop("mosaic"), as_bytes=True)
    outmask['img'] = img

    return outb0, outcolorFA, outmask


def create_report_json(dwi_corrected_file, eddy_rms, eddy_report,
                       color_fa_file, anat_mask_file,
                       outlier_indices,
                       eddy_qc_file,
                       outpath=op.abspath('./report.json')):

    report = {}
    report['dwi_corrected'] = createSprite4D(dwi_corrected_file)

    b0, colorFA, mask = createB0_ColorFA_Mask_Sprites(dwi_corrected_file,
                                                      color_fa_file,
                                                      anat_mask_file)
    report['b0'] = b0
    report['colorFA'] = colorFA
    report['anat_mask'] = mask
    report['outlier_volumes'] = outlier_indices.tolist()

    with open(eddy_report, 'r') as f:
        report['eddy_report'] = f.readlines()

    report['eddy_params'] = np.genfromtxt(eddy_rms).tolist()
    eddy_qc = load_json(eddy_qc_file)
    report['eddy_quad'] = eddy_qc
    save_json(outpath, report)
    return outpath
