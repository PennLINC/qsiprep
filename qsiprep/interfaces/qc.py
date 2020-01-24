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


def reshape3D(data, n=256):
    return np.pad(data, (
        (
            (n-data.shape[0]) // 2,
            ((n-data.shape[0]) + (data.shape[0] % 2 > 0)) // 2
        ),
        (
            (n-data.shape[1]) // 2,
            ((n-data.shape[1]) + (data.shape[1] % 2 > 0)) // 2
        ),
        (0, 0)
    ), "constant", constant_values=(0, 0))


def reshape4D(data, n=256):
    return np.pad(data, (
        (
            (n-data.shape[0]) // 2,
            ((n-data.shape[0]) + (data.shape[0] % 2 > 0)) // 2
        ),
        (
            (n-data.shape[1]) // 2,
            ((n-data.shape[1]) + (data.shape[1] % 2 > 0)) // 2
        ),
        (0, 0), (0, 0)
    ), "constant", constant_values=(0, 0))


def get_middle_slices(data, slice_direction):
    slicer = {"ax": 0, "cor": 1, "sag": 2}
    all_data_slicer = [slice(None), slice(None), slice(None)]
    num_slices = data.shape[slicer[slice_direction]]
    slice_num = int(num_slices / 2)
    all_data_slicer[slicer[slice_direction]] = slice_num
    tile = data[tuple(all_data_slicer)]

    # make it a square
    N = max(tile.shape[:2])
    tile = reshape3D(tile, N)

    return tile


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

        if len(tile.shape) == 3:
            mosaic[xmin:xmax, ymin:ymax] = tile[:, :, t]
        else:
            mosaic[xmin:xmax, ymin:ymax, :] = tile[:, :, :, t]

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
        tile = get_middle_slices(dwi, orient)

        # create sprite images for each
        results = create_sprite_from_tiles(tile, as_bytes=True)
        results['img_type'] = '4dsprite'
        results['orientation'] = orient
        output.append(results)

    return output


def createB0_ColorFA_Mask_Sprites(b0_file, colorFA_file, mask_file):
    colorfa = load_and_reorient(colorFA_file)
    b0 = load_and_reorient(b0_file)[:, :, :, 0]
    anat_mask = load_and_reorient(mask_file)

    N = max(*b0.shape[:2])

    # make a b0 sprite
    b0 = reshape3D(b0, N)
    _, mask = median_otsu(b0)
    outb0 = create_sprite_from_tiles(b0, as_bytes=True)
    outb0['img_type'] = 'brainsprite'

    # make a colorFA sprite, masked by b0
    Q = reshape4D(colorfa, N)
    Q[np.logical_not(mask)] = np.nan
    Q = np.moveaxis(Q,  -2, -1)
    outcolorFA = create_sprite_from_tiles(Q, as_bytes=True)
    outcolorFA['img_type'] = 'brainsprite'

    # make an anat mask contour sprite
    outmask = create_sprite_from_tiles(reshape3D(anat_mask, N))
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
