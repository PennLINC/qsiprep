import os
import os.path as op
import argparse
import shutil
from unittest import mock
import pytest
from pkg_resources import resource_filename as pkgrf
from qsiprep.cli.run import main as cli_main
from get_data import (RECON_INPUT, FS_LICENSE, WORKING_DIR, bids_singlescan_data,
                      WORKING_SINGLE_DIR, preprocessed_data, URL_PREFIX, get_default_cli_args)

"""
def test_controllability_recon_only(preprocessed_data, tmpdir):
    controllability_recon_only = get_default_cli_args()
    controllability_recon_only.work_dir = tmpdir
    controllability_recon_only.recon_only = True
    controllability_recon_only.output_dir = op.join(tmpdir, "control_recon_only")
    controllability_recon_only.recon_spec = pkgrf("qsiprep",
                                                  "data/pipelines/controllability.json")

    with mock.patch.object(
            argparse.ArgumentParser, 'parse_args', return_value=controllability_recon_only):

        with pytest.raises(SystemExit):
            cli_main()
"""

def test_preproc_plus_controllability(bids_singlescan_data, tmpdir):
    preproc_plus_controllability = get_default_cli_args()
    preproc_plus_controllability.bids_dir = WORKING_SINGLE_DIR + "/DSCSDSI"
    preproc_plus_controllability.work_dir = tmpdir
    preproc_plus_controllability.recon_only = False
    preproc_plus_controllability.output_dir = op.join(tmpdir, "preproc_control")
    preproc_plus_controllability.recon_spec = pkgrf("qsiprep",
                                                    "data/pipelines/controllability.json")

    with mock.patch.object(
            argparse.ArgumentParser, 'parse_args', return_value=preproc_plus_controllability):

        with pytest.raises(SystemExit):
            cli_main()
