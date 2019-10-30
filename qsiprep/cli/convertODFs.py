#!/usr/bin/env python
import warnings
import os
from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
from qsiprep.interfaces.converters import FODtoFIBGZ, FIBGZtoFOD

warnings.filterwarnings("ignore", category=ImportWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)


def sink_mask_file(in_file, orig_file, out_dir):
    import os
    from nipype.utils.filemanip import fname_presuffix, copyfile
    os.makedirs(out_dir, exist_ok=True)
    out_file = fname_presuffix(orig_file, suffix='_mask', newpath=out_dir)
    copyfile(in_file, out_file, copy=True, use_hardlink=True)
    return out_file


def fib_to_mif():
    """Convert fib to mif."""
    parser = ArgumentParser(
        description='qsiprep: Convert DSI Studio fib file to MRtrix mif file.',
        formatter_class=RawTextHelpFormatter)

    parser.add_argument('--fib',
                        required=True,
                        action='store',
                        type=os.path.abspath,
                        default='',
                        help='DSI Studio fib file to convert')
    parser.add_argument('--mif',
                        type=os.path.abspath,
                        required=False,
                        action='store',
                        default='',
                        help='output path for a MRtrix mif file')
    parser.add_argument('--ref_image',
                        required=True,
                        action='store',
                        type=os.path.abspath,
                        help='a NIfTI-1 format file with a valid q/sform.')
    parser.add_argument('--subtract-iso',
                        required=False,
                        action='store_true',
                        help='subtract ODF min so visualization looks similar in mrview')
    opts = parser.parse_args()
    converter = FIBGZtoFOD(mif_file=opts.mif,
                           fib_file=opts.fib,
                           ref_image=opts.ref_image,
                           subtract_iso=opts.subtract_iso)
    converter.run()


def mif_to_fib():
    """Convert mif to fib."""
    parser = ArgumentParser(
        description='qsiprep: Convert MRtrix mif file to DSI Studio fib file',
        formatter_class=RawTextHelpFormatter)

    parser.add_argument('--mif',
                        type=os.path.abspath,
                        required=True,
                        action='store',
                        default='',
                        help='MRtrix mif file to convert')
    parser.add_argument('--fib',
                        required=True,
                        action='store',
                        type=os.path.abspath,
                        default='',
                        help='the output path for the DSI Studio fib file')
    parser.add_argument('--mask',
                        required=False,
                        action='store',
                        type=os.path.abspath,
                        help='a NIfTI-1 format mask file.')
    parser.add_argument('--num_fibers',
                        required=False,
                        action='store',
                        type=int,
                        default=5,
                        help='maximum number of fixels per voxel.')
    parser.add_argument('--unit-odf',
                        required=False,
                        action='store_true',
                        help='force ODFs to sum to 1.')
    opts = parser.parse_args()
    if opts.mask is not None:
        converter = FODtoFIBGZ(mif_file=opts.mif,
                               fib_file=opts.fib,
                               num_fibers=opts.num_fibers,
                               unit_odf=opts.unit_odf,
                               mask_file=opts.mask)
    else:
        converter = FODtoFIBGZ(mif_file=opts.mif,
                               fib_file=opts.fib,
                               num_fibers=opts.num_fibers,
                               unit_odf=opts.unit_odf)
    converter.run()


if __name__ == "__main__":
    mif_to_fib()
