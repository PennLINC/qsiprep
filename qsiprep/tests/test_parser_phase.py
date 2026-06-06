"""Tests for the --dwi-phase-correction CLI flag."""

import os
import tempfile

import pytest

from qsiprep.cli.parser import _build_parser


def test_phase_correction_flag_default():
    with tempfile.TemporaryDirectory() as tmpdir:
        bids_dir = os.path.join(tmpdir, 'bids')
        out_dir = os.path.join(tmpdir, 'out')
        os.makedirs(bids_dir)

        parser = _build_parser()
        opts = parser.parse_args([bids_dir, out_dir, 'participant', '--output-resolution', '1.2'])
        assert opts.dwi_phase_correction == 'none'


def test_phase_correction_flag_set():
    with tempfile.TemporaryDirectory() as tmpdir:
        bids_dir = os.path.join(tmpdir, 'bids')
        out_dir = os.path.join(tmpdir, 'out')
        os.makedirs(bids_dir)

        parser = _build_parser()
        opts = parser.parse_args(
            [
                bids_dir,
                out_dir,
                'participant',
                '--output-resolution',
                '1.2',
                '--dwi-phase-correction',
                'tv',
            ]
        )
        assert opts.dwi_phase_correction == 'tv'


def test_phase_correction_flag_rejects_bad_value():
    with tempfile.TemporaryDirectory() as tmpdir:
        bids_dir = os.path.join(tmpdir, 'bids')
        out_dir = os.path.join(tmpdir, 'out')
        os.makedirs(bids_dir)

        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    bids_dir,
                    out_dir,
                    'participant',
                    '--output-resolution',
                    '1.2',
                    '--dwi-phase-correction',
                    'bogus',
                ]
            )
