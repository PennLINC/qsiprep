"""Command-line interface tests."""

import os
import sys
from unittest.mock import patch

import pytest
from nipype import config as nipype_config

from qsiprep.cli import run
from qsiprep.cli.parser import parse_args
from qsiprep.cli.workflow import build_boilerplate, build_workflow
from qsiprep.tests.utils import (
    check_generated_files,
    download_test_data,
    get_test_data_path,
)
from qsiprep.utils.bids import write_derivative_description
from qsiprep.viz.reports import generate_reports

nipype_config.enable_debug_mode()


@pytest.mark.mrtrix_singleshell_ss3t
def test_mrtrix_singleshell_ss3t(data_dir, output_dir, working_dir):
    """Run reconstruction workflow tests.

    All supported reconstruction workflows get tested

    This tests the following features:
    - Blip-up + Blip-down DWI series for TOPUP/Eddy
    - Eddy is run on a CPU
    - Denoising is skipped
    - A follow-up reconstruction using the dsi_studio_gqi workflow

    Inputs
    ------
    - qsiprep single shell results (data/DSDTI_fmap)
    """
    TEST_NAME = "mrtrix_singleshell_ss3t"

    dataset_dir = download_test_data("singleshell_output", data_dir)

    out_dir = os.path.join(output_dir, TEST_NAME)
    work_dir = os.path.join(working_dir, TEST_NAME)

    parameters = [
        dataset_dir,
        out_dir,
        "participant",
        f"-w={work_dir}",
        f"--recon-input={dataset_dir}",
        "--sloppy",
        "--recon-spec mrtrix_singleshell_ss3t_ACT-fast",
        "--recon-only",
    ]

    _run_and_generate(TEST_NAME, parameters, test_main=False)


@pytest.mark.mrtrix_multishell_ss3t
def test_mrtrix_multishell_ss3t(data_dir, output_dir, working_dir):
    """Run reconstruction workflow tests.

    All supported reconstruction workflows get tested

    This tests the following features:
    - Blip-up + Blip-down DWI series for TOPUP/Eddy
    - Eddy is run on a CPU
    - Denoising is skipped
    - A follow-up reconstruction using the dsi_studio_gqi workflow

    Inputs
    ------
    - qsiprep multi shell results (data/DSDTI_fmap)
    """
    TEST_NAME = "mrtrix_multishell_ss3t"

    dataset_dir = download_test_data("multishell_output", data_dir)

    out_dir = os.path.join(output_dir, TEST_NAME)
    work_dir = os.path.join(working_dir, TEST_NAME)

    parameters = [
        dataset_dir,
        out_dir,
        "participant",
        f"-w={work_dir}",
        f"--recon-input={dataset_dir}",
        "--sloppy",
        "--recon-spec mrtrix_multishell_ss3t_ACT-fast",
        "--recon-only",
    ]

    _run_and_generate(TEST_NAME, parameters, test_main=False)


@pytest.mark.dsdti_fmap
def test_dsdti_fmap(data_dir, output_dir, working_dir):
    """Run AllFieldmaps test.

    Instead of running full workflows, this test checks that workflows can
    be built for all sorts of fieldmap configurations.

    This tests the following features:
    - Blip-up + Blip-down DWI series for TOPUP/Eddy
    - Eddy is run on a CPU
    - Denoising is skipped
    - A follow-up reconstruction using the dsi_studio_gqi workflow

    Inputs
    ------
    - DSDTI BIDS data (data/DSDTI_fmap)
    """
    TEST_NAME = "dsdti_fmap"

    dataset_dir = download_test_data("DSDTI_fmap", data_dir)
    out_dir = os.path.join(output_dir, TEST_NAME)
    work_dir = os.path.join(working_dir, TEST_NAME)

    parameters = [
        dataset_dir,
        out_dir,
        "participant",
        f"-w={work_dir}",
        "--boilerplate",
        "--sloppy",
        "--write-graph",
        "--mem_mb=4096",
        "--output-resolution=5",
    ]

    _run_and_generate(TEST_NAME, parameters, test_main=False)


@pytest.mark.dscsdsi_fmap
def test_dscsdsi_fmap(data_dir, output_dir, working_dir):
    """Run AllFieldmaps test.

    Instead of running full workflows, this test checks that workflows can
    be built for all sorts of fieldmap configurations.

    This tests the following features:
    - Blip-up + Blip-down DWI series for TOPUP/Eddy
    - Eddy is run on a CPU
    - Denoising is skipped
    - A follow-up reconstruction using the dsi_studio_gqi workflow

    Inputs
    ------
    - DSDTI BIDS data (data/DSCSDSI_fmap)
    """
    TEST_NAME = "dscsdsi_fmap"

    dataset_dir = download_test_data("DSCSDSI_fmap", data_dir)
    out_dir = os.path.join(output_dir, TEST_NAME)
    work_dir = os.path.join(working_dir, TEST_NAME)

    parameters = [
        dataset_dir,
        out_dir,
        "participant",
        f"-w={work_dir}",
        "--boilerplate",
        "--sloppy",
        "--write-graph",
        "--mem_mb=4096",
        "--output-resolution=5",
    ]

    _run_and_generate(TEST_NAME, parameters, test_main=False)


@pytest.mark.amico_noddi
def test_amico_noddi(data_dir, output_dir, working_dir):
    """Run reconstruction workflow test.

    All supported reconstruction workflows get tested.

    This tests the following features:
    - Blip-up + Blip-down DWI series for TOPUP/Eddy
    - Eddy is run on a CPU
    - Denoising is skipped
    - A follow-up reconstruction using the dsi_studio_gqi workflow

    Inputs
    ------
    - DSDTI BIDS data (data/singleshell_output)
    """
    TEST_NAME = "amico_noddi"

    dataset_dir = download_test_data("singleshell_output", data_dir)
    # XXX: Having to modify dataset_dirs is suboptimal.
    dataset_dir = os.path.join(dataset_dir, "qsiprep")
    out_dir = os.path.join(output_dir, TEST_NAME)
    work_dir = os.path.join(working_dir, TEST_NAME)

    parameters = [
        dataset_dir,
        out_dir,
        "participant",
        f"-w={work_dir}",
        f"--recon-input={dataset_dir}",
        "--sloppy",
        "--recon-spec=amico_noddi",
        "--recon-only",
        "--output-resolution=5",
    ]

    _run_and_generate(TEST_NAME, parameters, test_main=True)


def _run_and_generate(test_name, parameters, test_main=False):
    from qsiprep import config

    # TODO: Add this param
    # parameters.append("--clean-workdir")
    parameters.append("--stop-on-first-crash")
    parameters.append("--notrack")
    parameters.append("-vv")

    if test_main:
        # This runs, but for some reason doesn't count toward coverage.
        argv = ["qsiprep"] + parameters
        with patch.object(sys, "argv", argv):
            with pytest.raises(SystemExit) as e:
                run.main()

            assert e.value.code == 0
    else:
        # XXX: This isn't working because config.execution.fs_license_file is None.
        parse_args(parameters)
        config_file = config.execution.work_dir / f"config-{config.execution.run_uuid}.toml"
        config.loggers.cli.warning(f"Saving config file to {config_file}")
        config.to_filename(config_file)

        retval = build_workflow(config_file, exec_mode="auto", retval={})
        qsiprep_wf = retval["workflow"]
        qsiprep_wf.run()
        write_derivative_description(config.execution.fmri_dir, config.execution.qsiprep_dir)

        build_boilerplate(str(config_file), qsiprep_wf)
        session_list = (
            config.execution.bids_filters.get("bold", {}).get("session")
            if config.execution.bids_filters
            else None
        )
        generate_reports(
            subject_list=config.execution.participant_label,
            output_dir=config.execution.qsiprep_dir,
            run_uuid=config.execution.run_uuid,
            session_list=session_list,
        )

    output_list_file = os.path.join(get_test_data_path(), f"{test_name}_outputs.txt")
    check_generated_files(config.execution.qsiprep_dir, output_list_file)
