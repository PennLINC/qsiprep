#!/usr/bin/env python
"""Flatten a niworkflows "flicker" report SVG into a static side-by-side PNG.

*QSIPrep*'s denoising and phase-correction reportlets are animated SVGs: a single
``<svg>`` holds two complete mosaics -- a ``<g class="background-svg">`` (the
"before" frame) and a ``<g class="foreground-svg">`` (the "after" frame) -- that
a CSS opacity animation flickers between. That animation cannot be shown in the
static HTML/PDF documentation, so this script renders each frame on its own and
stacks them side by side into a single PNG (background on the left, foreground on
the right), letting readers compare both frames at a glance.

Rasterization is delegated to ``rsvg-convert`` (librsvg), which is found on
``PATH`` or supplied with ``--rsvg``. The two frames are recombined with Pillow.

Examples
--------
Regenerate the two figures bundled in the docs (writes to ``docs/_static``)::

    python docs/tools/svg_frames_to_png.py

Convert an arbitrary report SVG::

    python docs/tools/svg_frames_to_png.py --pair INPUT.svg OUTPUT.png
"""

import argparse
import copy
import os
import shutil
import subprocess
import sys
import tempfile

from lxml import etree
from PIL import Image

SVG_NS = 'http://www.w3.org/2000/svg'
NSMAP = {'s': SVG_NS}

# Default conversions: the example figures referenced from docs/preprocessing.rst.
# Paths are relative to the repository's ``docs`` directory.
DEFAULT_PAIRS = [
    (
        '/mnt/c/Users/tsalo/Documents/datasets/qsiprep-development/dwidenoise-out/'
        'sub-20188/ses-1/figures/'
        'sub-20188_ses-1_dir-AP_run-01_part-mag_desc-denoising_dwi.svg',
        '_static/denoising_example.png',
    ),
    (
        '/mnt/c/Users/tsalo/Documents/datasets/qsiprep-development/tvc-out-pi/'
        'sub-20188/ses-1/figures/'
        'sub-20188_ses-1_dir-AP_run-01_part-mag_desc-phasecorrection_dwi.svg',
        '_static/phasecorrection_example.png',
    ),
]


def _find_rsvg(explicit=None):
    """Locate the ``rsvg-convert`` executable."""
    if explicit:
        if not os.path.isfile(explicit):
            raise FileNotFoundError(f'rsvg-convert not found at {explicit}')
        return explicit
    found = shutil.which('rsvg-convert')
    if found:
        return found
    # Fall back to common micromamba/conda environments that ship librsvg.
    for env in ('fsl', 'mapenv'):
        candidate = os.path.expanduser(f'~/micromamba/envs/{env}/bin/rsvg-convert')
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        'Could not find "rsvg-convert". Install librsvg or pass --rsvg /path/to/rsvg-convert.'
    )


def _frame_svg(root, keep_class):
    """Return a deep copy of ``root`` containing only the requested frame group.

    The sibling frame is removed, and the kept group is forced to full opacity
    with its animation class stripped, so ``rsvg-convert`` renders it as an
    opaque still frame.
    """
    clone = copy.deepcopy(root)
    drop_class = 'foreground-svg' if keep_class == 'background-svg' else 'background-svg'

    for el in clone.xpath(f".//s:g[@class='{drop_class}']", namespaces=NSMAP):
        el.getparent().remove(el)

    for el in clone.xpath(f".//s:g[@class='{keep_class}']", namespaces=NSMAP):
        # Remove the class (it triggers the CSS flicker animation) and pin opacity.
        el.attrib.pop('class', None)
        el.set('style', 'opacity:1')

    return clone


def _rasterize(svg_root, rsvg, zoom):
    """Rasterize an SVG element tree to a Pillow image via ``rsvg-convert``."""
    with tempfile.TemporaryDirectory() as tmp:
        svg_path = os.path.join(tmp, 'frame.svg')
        png_path = os.path.join(tmp, 'frame.png')
        etree.ElementTree(svg_root).write(svg_path, xml_declaration=True, encoding='utf-8')
        subprocess.run(
            [
                rsvg,
                '--zoom',
                str(zoom),
                '--background-color',
                'white',
                '--output',
                png_path,
                svg_path,
            ],
            check=True,
        )
        return Image.open(png_path).convert('RGB')


def _stack_side_by_side(left, right, gap, bg):
    """Place two images side by side on a single canvas, vertically centered."""
    height = max(left.height, right.height)
    width = left.width + gap + right.width
    canvas = Image.new('RGB', (width, height), bg)
    canvas.paste(left, (0, (height - left.height) // 2))
    canvas.paste(right, (left.width + gap, (height - right.height) // 2))
    return canvas


def convert(in_svg, out_png, rsvg, zoom=1.0, gap=16, bg=(255, 255, 255)):
    """Flatten one flicker SVG into a side-by-side PNG."""
    root = etree.parse(in_svg).getroot()

    found = {
        c
        for c in ('background-svg', 'foreground-svg')
        if root.xpath(f".//s:g[@class='{c}']", namespaces=NSMAP)
    }
    if found != {'background-svg', 'foreground-svg'}:
        raise ValueError(
            f'{in_svg}: expected both background-svg and foreground-svg frames, found {found}'
        )

    left = _rasterize(_frame_svg(root, 'background-svg'), rsvg, zoom)
    right = _rasterize(_frame_svg(root, 'foreground-svg'), rsvg, zoom)

    combined = _stack_side_by_side(left, right, gap, bg)
    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or '.', exist_ok=True)
    combined.save(out_png, optimize=True)
    print(f'{in_svg}\n  -> {out_png} ({combined.width}x{combined.height})')


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--pair',
        nargs=2,
        metavar=('IN_SVG', 'OUT_PNG'),
        action='append',
        help='An input flicker SVG and the output PNG path. May be repeated. '
        'If omitted, the example denoising and phase-correction figures are regenerated.',
    )
    parser.add_argument(
        '--rsvg', help='Path to the rsvg-convert executable (default: search PATH).'
    )
    parser.add_argument(
        '--zoom', type=float, default=1.0, help='Rasterization zoom factor (default: 1.0).'
    )
    parser.add_argument(
        '--gap', type=int, default=16, help='Pixel gap between the two frames (default: 16).'
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    rsvg = _find_rsvg(args.rsvg)

    if args.pair:
        pairs = args.pair
    else:
        # Resolve default output paths rel. to docs directory (this file's parent's parent)
        docs_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pairs = [(src, os.path.join(docs_dir, dst)) for src, dst in DEFAULT_PAIRS]

    for in_svg, out_png in pairs:
        convert(in_svg, out_png, rsvg, zoom=args.zoom, gap=args.gap)


if __name__ == '__main__':
    main()
