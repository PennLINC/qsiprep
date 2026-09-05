"""Run every fold of the SynB0-DISCO dual-channel U-Net and average the result.

This is a standalone script, not a qsiprep module: it is executed with the
isolated Torch interpreter (see ``QSIPREP_TORCH_PYTHON`` and the
``Synb0Inference`` interface), where qsiprep itself is not installed. It may
import only the standard library, numpy, nibabel, torch, and the ``model.py``/
``util.py`` modules shipped next to the SynB0 weights.

The preprocessing (padding, normalization, channel order, unpadding) must stay
identical to SynB0-DISCO's ``inference.py``; only the fold loop and averaging
are additions (the reference pipeline averages the fold outputs with fslmaths).
"""

import argparse
import os
import sys
from glob import glob


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--t1', required=True, help='normalized T1w on the 2.5mm atlas grid')
    parser.add_argument('--b0', required=True, help='distorted b=0 on the 2.5mm atlas grid')
    parser.add_argument(
        '--synb0-dir',
        required=True,
        help='SynB0 distribution containing dual_channel_unet/, model.py and util.py',
    )
    parser.add_argument('--out', required=True, help='output synthetic b=0 image')
    parser.add_argument(
        '--dispersion-out',
        help='also write the across-fold standard deviation image (the '
        "ensemble's disagreement, a model-uncertainty map for QC)",
    )
    args = parser.parse_args()

    import nibabel as nb
    import numpy as np
    import torch

    sys.path.insert(0, args.synb0_dir)
    import util
    from model import UNet3D

    weight_files = sorted(glob(os.path.join(args.synb0_dir, 'dual_channel_unet', '*.pth')))
    if not weight_files:
        sys.exit(f'No model weights found in {args.synb0_dir}/dual_channel_unet')

    device = torch.device('cpu')

    img_t1 = np.expand_dims(util.get_nii_img(args.t1), axis=3)
    img_b0 = np.expand_dims(util.get_nii_img(args.b0), axis=3)

    # Pad (77, 91, 77) to (80, 96, 80): the U-Net needs dims divisible by 8
    img_t1 = np.pad(img_t1, ((2, 1), (3, 2), (2, 1), (0, 0)), 'constant')
    img_b0 = np.pad(img_b0, ((2, 1), (3, 2), (2, 1), (0, 0)), 'constant')

    img_t1 = util.nii2torch(img_t1)
    img_b0 = util.nii2torch(img_b0)

    # The T1 scale is a fixed 0-150 (FreeSurfer-normalized intensities); the
    # b=0 is scaled by its own 99th percentile and restored afterwards.
    img_t1 = util.normalize_img(img_t1, 150, 0, 1, -1)
    max_b0 = np.percentile(img_b0, 99)
    img_b0 = util.normalize_img(img_b0, max_b0, 0, 1, -1)

    img_data = np.concatenate((img_b0, img_t1), axis=1)
    img_data = torch.from_numpy(img_data).float().to(device)

    folds = []
    for weights in weight_files:
        model = UNet3D(2, 1).to(device)
        model.load_state_dict(torch.load(weights, map_location=device))
        model.eval()
        with torch.no_grad():
            fold_out = model(img_data)
        fold_out = util.unnormalize_img(fold_out, max_b0, 0, 1, -1)
        fold_out = fold_out[:, :, 2:-1, 2:-1, 3:-2]  # undo the padding
        folds.append(np.squeeze(util.torch2nii(fold_out.detach().cpu())))
        print(f'finished fold {os.path.basename(weights)}')
    folds = np.stack(folds, axis=-1)

    template = nb.load(args.b0)
    nb.Nifti1Image(folds.mean(-1), template.affine, template.header).to_filename(args.out)
    if args.dispersion_out:
        nb.Nifti1Image(folds.std(-1), template.affine, template.header).to_filename(
            args.dispersion_out
        )


if __name__ == '__main__':
    main()
