import os
import os.path as op
import argparse
from unittest import mock
import pytest
from qsiprep.cli.run import main as cli_main
from get_data import (bids_data, WORKING_DIR, ANAT_URL, URL_PREFIX, get_default_cli_args,
                      link_anatomicals)


RUN_ANAT = True


@pytest.fixture
def anatomical_pipeline(bids_data):
    """Download data and run the anatomical preprocessing."""
    anat_only_opts = get_default_cli_args()
    anat_only_opts.anat_only = True
    anat_only_opts.output_dir = op.join(WORKING_DIR, "anat_only_output")

    if RUN_ANAT:
        with mock.patch.object(
                argparse.ArgumentParser, 'parse_args', return_value=anat_only_opts):

            with pytest.raises(SystemExit):
                cli_main()

    else:
        status = os.system('curl -sSL {} | tar xvfJ - -C {}'.format(
            URL_PREFIX + ANAT_URL, WORKING_DIR))
        assert status == 0
        wd_files = os.listdir(WORKING_DIR)
        assert 'qsiprep_wf' in wd_files


def test_anat(anatomical_pipeline):
    assert True

"""
def test_buds(anatomical_pipeline, tmpdir):
    buds_opts = get_default_cli_args()
    buds_opts.work_dir = tmpdir
    buds_opts.ignore = []
    buds_opts.output_dir = op.join(tmpdir, "buds")
    buds_opts.combine_all_dwis = True
    buds_opts.anat_only = False
    buds_opts.prefer_dedicated_fmaps = False

    assert link_anatomicals(tmpdir)
    with mock.patch.object(
            argparse.ArgumentParser, 'parse_args', return_value=buds_opts):

        with pytest.raises(SystemExit):
            cli_main()
"""

def test_all_separate_no_sdc(anatomical_pipeline, tmpdir):
    all_separate_no_sdc_opts = get_default_cli_args()
    all_separate_no_sdc_opts.work_dir = tmpdir
    all_separate_no_sdc_opts.ignore = []
    all_separate_no_sdc_opts.output_dir = op.join(tmpdir, "all_separate_no_sdc")
    all_separate_no_sdc_opts.combine_all_dwis = False
    all_separate_no_sdc_opts.anat_only = False
    all_separate_no_sdc_opts.prefer_dedicated_fmaps = True

    #assert link_anatomicals(tmpdir)
    with mock.patch.object(
            argparse.ArgumentParser, 'parse_args', return_value=all_separate_no_sdc_opts):

        with pytest.raises(SystemExit):
            cli_main()

"""
def test_all_separate_syn_sdc(anatomical_pipeline, tmpdir):
    all_separate_syn_opts = get_default_cli_args()
    all_separate_syn_opts.work_dir = tmpdir
    all_separate_syn_opts.ignore = ["fieldmaps"]
    all_separate_syn_opts.output_dir = op.join(tmpdir, "all_separate_syn")
    all_separate_syn_opts.combine_all_dwis = False
    all_separate_syn_opts.prefer_dedicated_fmaps = False
    all_separate_syn_opts.force_syn = True
    all_separate_syn_opts.use_syn_sdc = True

    assert link_anatomicals(tmpdir)
    with mock.patch.object(
            argparse.ArgumentParser, 'parse_args', return_value=all_separate_syn_opts):

        # with pytest.raises(SystemExit):
        cli_main()
"""
