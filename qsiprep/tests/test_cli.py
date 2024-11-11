"""Command-line interface tests."""

import os
import sys
from unittest.mock import patch

import pytest
from nipype import config as nipype_config

from qsiprep.cli import run
from qsiprep.cli.parser import parse_args
from qsiprep.cli.workflow import build_boilerplate, build_workflow
from qsiprep.reports.core import generate_reports
from qsiprep.tests.utils import (
    check_generated_files,
    download_test_data,
    get_test_data_path,
)
from qsiprep.utils.bids import write_derivative_description

nipype_config.enable_debug_mode()

DEFAULT_NUM_CPUS = 4


@pytest.mark.integration
@pytest.mark.dsdti_fmap
def test_dsdti_fmap(data_dir, output_dir, working_dir):
    """Run AllFieldmaps test on DSDTI data.

    Was in AllFieldmapsTests.sh. I split it between this and the DSCSDSI test.
    XXX: Not called in CircleCI.

    Instead of running full workflows, this test checks that workflows can
    be built for all sorts of fieldmap configurations.

    This tests the following features:
    - Blip-up + Blip-down DWI series for TOPUP/Eddy
    - Eddy is run on a CPU
    - Denoising is skipped

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
        "--mem-mb=4096",
        "--output-resolution=5",
    ]

    _run_and_generate(TEST_NAME, parameters, test_main=True)


@pytest.mark.integration
@pytest.mark.dscsdsi_fmap
def test_dscsdsi_fmap(data_dir, output_dir, working_dir):
    """Run AllFieldmaps test on DSCSDSI data.

    Was in AllFieldmapsTests.sh. I split it between this and the DSDTI test.
    XXX: Not called in CircleCI.

    Instead of running full workflows, this test checks that workflows can
    be built for all sorts of fieldmap configurations.

    This tests the following features:
    - Blip-up + Blip-down DWI series for TOPUP/Eddy
    - Eddy is run on a CPU
    - Denoising is skipped

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
        "--b0-motion-corr-to=first",
        "--write-graph",
        "--mem-mb=4096",
        "--output-resolution=5",
    ]

    _run_and_generate(TEST_NAME, parameters, test_main=True)


@pytest.mark.integration
@pytest.mark.cuda
def test_cuda(data_dir, output_dir, working_dir):
    """

    Was in CUDATest.sh.
    XXX: Not called in CircleCI.

    This tests the following features:
    - Blip-up + Blip-down DWI series for TOPUP/Eddy
    - Eddy is run on a CPU
    - Denoising is skipped

    Inputs
    ------
    - DSDTI BIDS data (data/drbuddi_rpe_series)
    """
    TEST_NAME = "cuda"

    dataset_dir = download_test_data("drbuddi_rpe_series", data_dir)
    # XXX: Having to modify dataset_dirs is suboptimal.
    dataset_dir = os.path.join(dataset_dir, "qsiprep")
    out_dir = os.path.join(output_dir, TEST_NAME)
    work_dir = os.path.join(working_dir, TEST_NAME)
    test_data_path = get_test_data_path()
    eddy_config = os.path.join(test_data_path, "eddy_config.json")

    parameters = [
        dataset_dir,
        out_dir,
        "participant",
        f"-w={work_dir}",
        "--sloppy",
        "--anat-modality=none",
        "--denoise-method=none",
        "--b1-biascorrect-stage=none",
        "--pepolar-method=DRBUDDI",
        f"--eddy-config={eddy_config}",
        "--output-resolution=5",
    ]

    _run_and_generate(TEST_NAME, parameters, test_main=True)


@pytest.mark.integration
@pytest.mark.drbuddi_rpe
def test_drbuddi_rpe(data_dir, output_dir, working_dir):
    """

    Was in DRBUDDI_eddy_rpe_series.sh.

    This tests the following features:
    - Blip-up + Blip-down DWI series for TOPUP/Eddy
    - Eddy is run on a CPU
    - Denoising is skipped

    Inputs:
    -------

    - qsiprep single shell results (data/DSDTI_fmap)
    - qsiprep multi shell results (data/DSDTI_fmap)
    """
    TEST_NAME = "drbuddi_rpe"

    dataset_dir = download_test_data("drbuddi_rpe_series", data_dir)
    # XXX: Having to modify dataset_dirs is suboptimal.
    dataset_dir = os.path.join(dataset_dir, "tinytensor_rpe_series")
    out_dir = os.path.join(output_dir, TEST_NAME)
    work_dir = os.path.join(working_dir, TEST_NAME)
    test_data_path = get_test_data_path()
    eddy_config = os.path.join(test_data_path, "eddy_config.json")

    parameters = [
        dataset_dir,
        out_dir,
        "participant",
        f"-w={work_dir}",
        "--sloppy",
        "--anat-modality=none",
        "--denoise-method=none",
        "--b0-motion-corr-to=first",
        "--b1-biascorrect-stage=none",
        "--pepolar-method=DRBUDDI",
        f"--eddy-config={eddy_config}",
        "--output-resolution=5",
    ]

    _run_and_generate(TEST_NAME, parameters, test_main=True)


@pytest.mark.integration
@pytest.mark.drbuddi_shoreline_epi
def test_drbuddi_shoreline_epi(data_dir, output_dir, working_dir):
    """Test EPI fieldmap correction with SHORELine + DRBUDDI.

    Was in DRBUDDI_SHORELine_epi.sh.

    This tests the following features:
    - SHORELine (here, just b=0 registration) motion correction
    """
    TEST_NAME = "drbuddi_shoreline_epi"

    dataset_dir = download_test_data("drbuddi_epi", data_dir)
    # XXX: Having to modify dataset_dirs is suboptimal.
    dataset_dir = os.path.join(dataset_dir, "tinytensor_epi")
    out_dir = os.path.join(output_dir, TEST_NAME)
    work_dir = os.path.join(working_dir, TEST_NAME)

    parameters = [
        dataset_dir,
        out_dir,
        "participant",
        f"-w={work_dir}",
        "--sloppy",
        "--anat-modality=none",
        "--denoise-method=none",
        "--b0-motion-corr-to=first",
        "--b1-biascorrect-stage=none",
        "--pepolar-method=DRBUDDI",
        "--hmc-model=none",
        "--output-resolution=2",
        "--shoreline-iters=1",
    ]

    _run_and_generate(TEST_NAME, parameters, test_main=True)


@pytest.mark.integration
@pytest.mark.drbuddi_tensorline_epi
def test_drbuddi_tensorline_epi(data_dir, output_dir, working_dir):
    """Test EPI fieldmap correction with TENSORLine + DRBUDDI.

    Was in DRBUDDI_TENSORLine_epi.sh.

    This tests the following features:
    - TENSORLine (tensor-based) motion correction
    """
    TEST_NAME = "drbuddi_tensorline_epi"

    dataset_dir = download_test_data("DSDTI", data_dir)
    # XXX: Having to modify dataset_dirs is suboptimal.
    dataset_dir = os.path.join(dataset_dir, "DSDTI")
    out_dir = os.path.join(output_dir, TEST_NAME)
    work_dir = os.path.join(working_dir, TEST_NAME)

    parameters = [
        dataset_dir,
        out_dir,
        "participant",
        f"-w={work_dir}",
        "--sloppy",
        "--anat-modality=none",
        "--denoise-method=none",
        "--b0-motion-corr-to=first",
        "--b1-biascorrect-stage=none",
        "--pepolar-method=DRBUDDI",
        "--hmc-model=tensor",
        "--output-resolution=5",
        "--shoreline-iters=1",
    ]

    _run_and_generate(TEST_NAME, parameters, test_main=True)


@pytest.mark.integration
@pytest.mark.dscsdsi
def test_dscsdsi(data_dir, output_dir, working_dir):
    """DSCSDSI test

    Was in DSCSDSI.sh.

    This tests the following features:
    - Whether the --anat-only workflow is successful
    - Whether the regular qsiprep workflow can resume using the working directory from --anat-only
    - The SHORELine motion correction workflow
    - Skipping B1 biascorrection
    - Using the SyN-SDC distortion correction method

    Inputs
    ------
    - DSCSDSI BIDS data (data/DSCSDSI_nofmap)
    """
    TEST_NAME = "dscsdsi"

    dataset_dir = download_test_data("DSCSDSI", data_dir)
    # XXX: Having to modify dataset_dirs is suboptimal.
    dataset_dir = os.path.join(dataset_dir, "DSCSDSI_nofmap")
    out_dir = os.path.join(output_dir, TEST_NAME)
    work_dir = os.path.join(working_dir, TEST_NAME)

    parameters = [
        dataset_dir,
        out_dir,
        "participant",
        f"-w={work_dir}",
        "--sloppy",
        "--write-graph",
        "--use-syn-sdc",
        "--force-syn",
        "--b1-biascorrect-stage=none",
        "--hmc-model=3dSHORE",
        "--hmc-transform=Rigid",
        "--output-resolution=5",
        "--shoreline-iters=1",
    ]

    _run_and_generate(TEST_NAME, parameters, test_main=True)


@pytest.mark.integration
@pytest.mark.dsdti_nofmap
def test_dsdti_nofmap(data_dir, output_dir, working_dir):
    """DSCDTI_nofmap test.

    Was in DSDTI_nofmap.sh.

    This tests the following features:
    - A workflow with no distortion correction followed by eddy
    - Eddy is run on a CPU
    - Denoising is skipped

    Inputs
    ------
    - DSDTI BIDS data (data/DSDTI)
    """
    TEST_NAME = "dsdti_nofmap"

    dataset_dir = download_test_data("DSDTI", data_dir)
    # XXX: Having to modify dataset_dirs is suboptimal.
    dataset_dir = os.path.join(dataset_dir, "DSDTI")
    out_dir = os.path.join(output_dir, TEST_NAME)
    work_dir = os.path.join(working_dir, TEST_NAME)
    test_data_path = get_test_data_path()
    eddy_config = os.path.join(test_data_path, "eddy_config.json")

    parameters = [
        dataset_dir,
        out_dir,
        "participant",
        f"-w={work_dir}",
        "--sloppy",
        f"--eddy-config={eddy_config}",
        "--denoise-method=none",
        "--unringing-method=rpg",
        "--b1-biascorrect-stage=none",
        "--output-resolution=5",
    ]

    _run_and_generate(TEST_NAME, parameters, test_main=True)


@pytest.mark.integration
@pytest.mark.dsdti_synfmap
def test_dsdti_synfmap(data_dir, output_dir, working_dir):
    """DSCDTI_synfmap test

    Was in DSDTI_synfmap.sh.

    This tests the following features:
    - A workflow with no distortion correction followed by eddy
    - Eddy is run on a CPU
    - Denoising is skipped

    Inputs
    ------
    - DSDTI BIDS data (data/DSDTI)
    """
    TEST_NAME = "dsdti_synfmap"

    dataset_dir = download_test_data("DSDTI", data_dir)
    # XXX: Having to modify dataset_dirs is suboptimal.
    dataset_dir = os.path.join(dataset_dir, "DSDTI")
    out_dir = os.path.join(output_dir, TEST_NAME)
    work_dir = os.path.join(working_dir, TEST_NAME)
    test_data_path = get_test_data_path()
    eddy_config = os.path.join(test_data_path, "eddy_config.json")

    parameters = [
        dataset_dir,
        out_dir,
        "participant",
        f"-w={work_dir}",
        "--sloppy",
        f"--eddy-config={eddy_config}",
        "--denoise-method=none",
        "--force-syn",
        "--b1-biascorrect-stage=final",
        "--output-resolution=5",
    ]

    _run_and_generate(TEST_NAME, parameters, test_main=True)


@pytest.mark.integration
@pytest.mark.intramodal_template
def test_intramodal_template(data_dir, output_dir, working_dir):
    """IntramodalTemplate test

    A two-session dataset is used to create an intramodal template.

    This tests the following features:
    - Blip-up + Blip-down DWI series for TOPUP/Eddy
    - Eddy is run on a CPU
    - Denoising is skipped

    Inputs
    ------
    - twoses BIDS data (data/DSDTI_fmap)
    """
    TEST_NAME = "intramodal_template"

    dataset_dir = download_test_data("twoses", data_dir)
    # XXX: Having to modify dataset_dirs is suboptimal.
    dataset_dir = os.path.join(dataset_dir, "twoses")
    out_dir = os.path.join(output_dir, TEST_NAME)
    work_dir = os.path.join(working_dir, TEST_NAME)

    parameters = [
        dataset_dir,
        out_dir,
        "participant",
        f"-w={work_dir}",
        "--b1-biascorrect-stage=none",
        "--hmc-model=none",
        "--b0-motion-corr-to=first",
        "--output-resolution=5",
        "--intramodal-template-transform=BSplineSyN",
        "--intramodal-template-iters=2",
    ]

    _run_and_generate(TEST_NAME, parameters, test_main=True)


@pytest.mark.integration
@pytest.mark.multi_t1w
def test_multi_t1w(data_dir, output_dir, working_dir):
    """MultiT1w test

    This tests the following features:
    - freesurfer's robust template

    Inputs
    ------
    - DSDTI BIDS data (data/DSDTI)
    """
    TEST_NAME = "multi_t1w"

    dataset_dir = download_test_data("twoses", data_dir)
    # XXX: Having to modify dataset_dirs is suboptimal.
    dataset_dir = os.path.join(dataset_dir, "DSDTI")
    out_dir = os.path.join(output_dir, TEST_NAME)
    work_dir = os.path.join(working_dir, TEST_NAME)

    parameters = [
        dataset_dir,
        out_dir,
        "participant",
        f"-w={work_dir}",
        "--b1-biascorrect-stage=none",
        "--hmc-model=none",
        "--b0-motion-corr-to=first",
        "--output-resolution=5",
        "--intramodal-template-transform=BSplineSyN",
        "--intramodal-template-iters=2",
    ]

    _run_and_generate(TEST_NAME, parameters, test_main=True)


@pytest.mark.integration
@pytest.mark.maternal_brain_project
def test_maternal_brain_project(data_dir, output_dir, working_dir):
    """Run QSIPrep on Maternal Brain Project data.

    The dataset was built from the Maternal Brain Project dataset:
    https://openneuro.org/datasets/ds005299/versions/1.0.0

    The first subject's first session DWI data were downsampled to 5 mm isotropic voxels.
    The dataset contains multi-shell DWI data with a GRE field map.
    """
    TEST_NAME = "maternal_brain_project"

    dataset_dir = download_test_data("maternal_brain_project", data_dir)
    out_dir = os.path.join(output_dir, TEST_NAME)
    work_dir = os.path.join(working_dir, TEST_NAME)

    test_data_path = get_test_data_path()
    bids_filter = os.path.join(test_data_path, f"{TEST_NAME}_filter.json")

    parameters = [
        dataset_dir,
        out_dir,
        "participant",
        f"-w={work_dir}",
        "--sloppy",
        "--denoise-method=none",
        "--b1-biascorrect-stage=none",
        "--write-graph",
        "--output-resolution=5",
        "--hmc-model=3dSHORE",
        f"--bids-filter-file={bids_filter}",
    ]

    _run_and_generate(TEST_NAME, parameters, test_main=True)


@pytest.mark.integration
@pytest.mark.forrest_gump
def test_forrest_gump(data_dir, output_dir, working_dir):
    """Run QSIPrep on Forrest Gump data.

    The dataset was built from the Forrest Gump dataset:
    https://openneuro.org/datasets/ds000113/versions/1.3.0

    The first subject's first session DWI data were downsampled to 5 mm isotropic voxels.
    The dataset contains single-shell DWI data with a GRE field map.
    """
    TEST_NAME = "forrest_gump"

    dataset_dir = download_test_data("forrest_gump", data_dir)
    out_dir = os.path.join(output_dir, TEST_NAME)
    work_dir = os.path.join(working_dir, TEST_NAME)

    test_data_path = get_test_data_path()
    bids_filter = os.path.join(test_data_path, f"{TEST_NAME}_filter.json")

    parameters = [
        dataset_dir,
        out_dir,
        "participant",
        f"-w={work_dir}",
        "--sloppy",
        "--denoise-method=none",
        "--b1-biascorrect-stage=none",
        "--write-graph",
        "--output-resolution=5",
        f"--bids-filter-file={bids_filter}",
    ]

    _run_and_generate(TEST_NAME, parameters, test_main=True)


def _check_arg_specified(argname, arglist):
    for arg in arglist:
        if arg.startswith(argname):
            return True
    return False


def _update_resources(parameters):
    """We should use all the available CPUs for testing.

    Sometimes a test will set a specific amount of cpus. In that
    case, the number should be kept. Otherwise, try to read the
    env variable (specified in each job in config.yml). If
    this variable doesn't work, just set it to 4.
    """
    nthreads = int(os.environ.get("CIRCLECPUS", DEFAULT_NUM_CPUS))
    if not _check_arg_specified("--nthreads", parameters):
        parameters.append(f"--nthreads={nthreads}")
    if not _check_arg_specified("--omp-nthreads", parameters):
        parameters.append(f"--omp-nthreads={nthreads}")
    return parameters


def _run_and_generate(test_name, parameters, test_main=True):
    from qsiprep import config

    # TODO: Add --clean-workdir param to CLI
    parameters.append("--stop-on-first-crash")
    parameters.append("--notrack")
    parameters.append("-vv")

    # Update resource parameters
    parameters = _update_resources(parameters)

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

        retval = build_workflow(config_file, retval={})
        qsiprep_wf = retval["workflow"]
        qsiprep_wf.run()
        write_derivative_description(config.execution.bids_dir, config.execution.output_dir)

        build_boilerplate(str(config_file), qsiprep_wf)
        session_list = (
            config.execution.bids_filters.get("bold", {}).get("session")
            if config.execution.bids_filters
            else None
        )
        generate_reports(
            subject_list=config.execution.participant_label,
            output_dir=config.execution.output_dir,
            run_uuid=config.execution.run_uuid,
            session_list=session_list,
        )

    output_list_file = os.path.join(get_test_data_path(), f"{test_name}_outputs.txt")
    optional_outputs_list = os.path.join(get_test_data_path(), f"{test_name}_optional_outputs.txt")
    if not os.path.isfile(optional_outputs_list):
        optional_outputs_list = None

    check_generated_files(config.execution.output_dir, output_list_file, optional_outputs_list)
