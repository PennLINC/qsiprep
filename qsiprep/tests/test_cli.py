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


@pytest.mark.integration
@pytest.mark.mrtrix_singleshell_ss3t_act
def test_mrtrix_singleshell_ss3t_act(data_dir, output_dir, working_dir):
    """Run reconstruction workflow tests.

    Was in 3TissueReconTests.sh. I split it between this and the multi-shell test.

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
    TEST_NAME = "mrtrix_singleshell_ss3t_act"

    dataset_dir = download_test_data("singleshell_output", data_dir)
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
        "--recon-spec=mrtrix_singleshell_ss3t_ACT-fast",
        "--recon-only",
        "--output-resolution=5",
    ]

    _run_and_generate(TEST_NAME, parameters, test_main=True)


@pytest.mark.integration
@pytest.mark.mrtrix_singleshell_ss3t_noact
def test_mrtrix_singleshell_ss3t_noact(data_dir, output_dir, working_dir):
    """Run reconstruction workflow tests.

    Was in 3TissueReconTests.sh. I split it between this and the single-shell test.

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
    TEST_NAME = "mrtrix_singleshell_ss3t_noact"

    dataset_dir = download_test_data("singleshell_output", data_dir)
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
        "--recon-spec=mrtrix_singleshell_ss3t_noACT",
        "--recon-only",
        "--output-resolution=5",
    ]

    _run_and_generate(TEST_NAME, parameters, test_main=True)


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

    _run_and_generate(TEST_NAME, parameters, test_main=True)


@pytest.mark.integration
@pytest.mark.amico_noddi
def test_amico_noddi(data_dir, output_dir, working_dir):
    """Run reconstruction workflow test.

    Was in AMICOReconTests.sh.
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


@pytest.mark.integration
@pytest.mark.autotrack
def test_autotrack(data_dir, output_dir, working_dir):
    """Run reconstruction workflow test.

    Was in AutoTrackTest.sh.

    All supported reconstruction workflows get tested.

    This tests the following features:
    - Blip-up + Blip-down DWI series for TOPUP/Eddy
    - Eddy is run on a CPU
    - Denoising is skipped
    - A follow-up reconstruction using the dsi_studio_gqi workflow

    Inputs
    ------
    - DSDTI BIDS data (data/multishell_output)
    """
    TEST_NAME = "autotrack"

    dataset_dir = download_test_data("multishell_output", data_dir)
    # XXX: Having to modify dataset_dirs is suboptimal.
    dataset_dir = os.path.join(dataset_dir, "multishell_output", "qsiprep")
    out_dir = os.path.join(output_dir, TEST_NAME)
    work_dir = os.path.join(working_dir, TEST_NAME)

    parameters = [
        dataset_dir,
        out_dir,
        "participant",
        f"-w={work_dir}",
        f"--recon-input={dataset_dir}",
        "--sloppy",
        "--recon-spec=dsi_studio_autotrack",
        "--recon-only",
        "--output-resolution=5",
    ]

    _run_and_generate(TEST_NAME, parameters, test_main=True)


@pytest.mark.integration
@pytest.mark.cuda
def test_cuda(data_dir, output_dir, working_dir):
    """Run reconstruction workflow test.

    Was in CUDATest.sh.
    XXX: Not called in CircleCI.

    All supported reconstruction workflows get tested.

    This tests the following features:
    - Blip-up + Blip-down DWI series for TOPUP/Eddy
    - Eddy is run on a CPU
    - Denoising is skipped
    - A follow-up reconstruction using the dsi_studio_gqi workflow

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
        "--b1_biascorrect_stage=none",
        "--pepolar-method=DRBUDDI",
        f"--eddy_config={eddy_config}",
        "--output-resolution=5",
    ]

    _run_and_generate(TEST_NAME, parameters, test_main=True)


@pytest.mark.integration
@pytest.mark.dipy_mapmri
def test_dipy_mapmri(data_dir, output_dir, working_dir):
    """Run reconstruction workflow test.

    Was in DipyReconTests.sh. I split it between this and the dipy_dki test.

    All supported reconstruction workflows get tested

    This tests the following features:
    - Blip-up + Blip-down DWI series for TOPUP/Eddy
    - Eddy is run on a CPU
    - Denoising is skipped
    - A follow-up reconstruction using the dsi_studio_gqi workflow

    Inputs:
    -------

    - qsiprep single shell results (data/DSDTI_fmap)
    - qsiprep multi shell results (data/DSDTI_fmap)
    """
    TEST_NAME = "dipy_mapmri"

    dataset_dir = download_test_data("multishell_output", data_dir)
    # XXX: Having to modify dataset_dirs is suboptimal.
    dataset_dir = os.path.join(dataset_dir, "multishell_output", "qsiprep")
    out_dir = os.path.join(output_dir, TEST_NAME)
    work_dir = os.path.join(working_dir, TEST_NAME)

    parameters = [
        dataset_dir,
        out_dir,
        "participant",
        f"-w={work_dir}",
        "--sloppy",
        f"--recon-input={dataset_dir}",
        "--recon-spec=dipy_mapmri",
        "--recon-only",
        "--output-resolution=5",
    ]

    _run_and_generate(TEST_NAME, parameters, test_main=True)


@pytest.mark.integration
@pytest.mark.dipy_dki
def test_dipy_dki(data_dir, output_dir, working_dir):
    """Run reconstruction workflow test.

    Was in DipyReconTests.sh. I split it between this and the dipy_mapmri test.

    All supported reconstruction workflows get tested

    This tests the following features:
    - Blip-up + Blip-down DWI series for TOPUP/Eddy
    - Eddy is run on a CPU
    - Denoising is skipped
    - A follow-up reconstruction using the dsi_studio_gqi workflow

    Inputs:
    -------

    - qsiprep single shell results (data/DSDTI_fmap)
    - qsiprep multi shell results (data/DSDTI_fmap)
    """
    TEST_NAME = "dipy_dki"

    dataset_dir = download_test_data("multishell_output", data_dir)
    # XXX: Having to modify dataset_dirs is suboptimal.
    dataset_dir = os.path.join(dataset_dir, "multishell_output", "qsiprep")
    out_dir = os.path.join(output_dir, TEST_NAME)
    work_dir = os.path.join(working_dir, TEST_NAME)

    parameters = [
        dataset_dir,
        out_dir,
        "participant",
        f"-w={work_dir}",
        "--sloppy",
        f"--recon-input={dataset_dir}",
        "--recon-spec=dipy_dki",
        "--recon-only",
        "--output-resolution=5",
    ]

    _run_and_generate(TEST_NAME, parameters, test_main=True)


@pytest.mark.integration
@pytest.mark.drbuddi_rpe
def test_drbuddi_rpe(data_dir, output_dir, working_dir):
    """Run reconstruction workflow test.

    Was in DRBUDDI_eddy_rpe_series.sh.

    All supported reconstruction workflows get tested

    This tests the following features:
    - Blip-up + Blip-down DWI series for TOPUP/Eddy
    - Eddy is run on a CPU
    - Denoising is skipped
    - A follow-up reconstruction using the dsi_studio_gqi workflow

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
        "--nthreads=4",
        "--anat-modality=none",
        "--denoise-method=none",
        "--b1_biascorrect_stage=none",
        "--pepolar-method=DRBUDDI",
        f"--eddy_config={eddy_config}",
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
        "--b1-biascorrect-stage=none",
        "--pepolar-method=DRBUDDI",
        "--hmc-model=none",
        "--output-resolution=2",
        "--shoreline-iters=1",
        "--nthreads=1",
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
        "--b1-biascorrect-stage=none",
        "--pepolar-method=DRBUDDI",
        "--hmc-model=tensor",
        "--output-resolution=2",
        "--shoreline-iters=1",
        "--nthreads=1",
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
        "--nthreads=1",
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
    - A follow-up reconstruction using the dsi_studio_gqi workflow

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
        "--nthreads=1",
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
    - A follow-up reconstruction using the dsi_studio_gqi workflow

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
        "--nthreads=1",
    ]

    _run_and_generate(TEST_NAME, parameters, test_main=True)


@pytest.mark.integration
@pytest.mark.dsdti_topup
def test_dsdti_topup(data_dir, output_dir, working_dir):
    """DSCDTI_TOPUP test

    This tests the following features:
    - TOPUP on a single-shell sequence
    - Eddy is run on a CPU
    - mrdegibbs is run
    - A follow-up reconstruction using the dsi_studio_gqi workflow

    Inputs
    ------
    - DSDTI BIDS data (data/DSDTI)
    """
    TEST_NAME = "dsdti_topup"

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
        "--unringing-method=mrdegibbs",
        "--b1-biascorrect-stage=legacy",
        "--output-resolution=5",
        "--recon-spec=dsi_studio_gqi",
        "--nthreads=1",
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
    - A follow-up reconstruction using the dsi_studio_gqi workflow

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
        "--hmc_model=none",
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
        "--hmc_model=none",
        "--b0-motion-corr-to=first",
        "--output-resolution=5",
        "--intramodal-template-transform=BSplineSyN",
        "--intramodal-template-iters=2",
    ]

    _run_and_generate(TEST_NAME, parameters, test_main=True)


@pytest.mark.integration
@pytest.mark.scalar_mapper
def test_scalar_mapper(data_dir, output_dir, working_dir):
    """Test the TORTOISE recon workflow.

    All supported reconstruction workflows get tested

    Inputs
    ------
    - qsiprep multi shell results (data/DSDTI_fmap)
    """
    TEST_NAME = "scalar_mapper"

    dataset_dir = download_test_data("multishell_output", data_dir)
    # XXX: Having to modify dataset_dirs is suboptimal.
    dataset_dir = os.path.join(dataset_dir, "multishell_output", "qsiprep")
    out_dir = os.path.join(output_dir, TEST_NAME)
    work_dir = os.path.join(working_dir, TEST_NAME)

    parameters = [
        dataset_dir,
        out_dir,
        "participant",
        f"-w={work_dir}",
        f"--recon-input={dataset_dir}",
        "--sloppy",
        "--recon-spec=test_scalar_maps",
        "--recon-only",
        "--output-resolution=3.5",
        "--nthreads=1",
    ]

    _run_and_generate(TEST_NAME, parameters, test_main=True)


@pytest.mark.integration
@pytest.mark.pyafq_recon_external_trk
def test_pyafq_recon_external_trk(data_dir, output_dir, working_dir):
    """Reconstruction workflow tests

    All supported reconstruction workflows get tested

    This tests the following features:
    - pyAFQ pipeline with tractography done in mrtrix

    Inputs
    ------
    - qsiprep multi shell results (data/DSDTI_fmap)
    """
    TEST_NAME = "pyafq_recon_external_trk"

    dataset_dir = download_test_data("multishell_output", data_dir)
    # XXX: Having to modify dataset_dirs is suboptimal.
    dataset_dir = os.path.join(dataset_dir, "multishell_output", "qsiprep")
    out_dir = os.path.join(output_dir, TEST_NAME)
    work_dir = os.path.join(working_dir, TEST_NAME)

    parameters = [
        dataset_dir,
        out_dir,
        "participant",
        f"-w={work_dir}",
        f"--recon-input={dataset_dir}",
        "--sloppy",
        "--recon-spec=mrtrix_multishell_msmt_pyafq_tractometry",
        "--recon-only",
        "--output-resolution=5",
    ]

    _run_and_generate(TEST_NAME, parameters, test_main=True)


@pytest.mark.integration
@pytest.mark.pyafq_recon_full
def test_pyafq_recon_full(data_dir, output_dir, working_dir):
    """Reconstruction workflow tests

    All supported reconstruction workflows get tested

    This tests the following features:
    - Full pyAFQ pipeline

    Inputs
    ------
    - qsiprep multi shell results (data/DSDTI_fmap)
    """
    TEST_NAME = "pyafq_recon_full"

    dataset_dir = download_test_data("multishell_output", data_dir)
    # XXX: Having to modify dataset_dirs is suboptimal.
    dataset_dir = os.path.join(dataset_dir, "multishell_output", "qsiprep")
    out_dir = os.path.join(output_dir, TEST_NAME)
    work_dir = os.path.join(working_dir, TEST_NAME)

    parameters = [
        dataset_dir,
        out_dir,
        "participant",
        f"-w={work_dir}",
        f"--recon-input={dataset_dir}",
        "--sloppy",
        "--recon-spec=pyafq_tractometry",
        "--recon-only",
        "--output-resolution=5",
    ]

    _run_and_generate(TEST_NAME, parameters, test_main=True)


@pytest.mark.integration
@pytest.mark.mrtrix3_recon
def test_mrtrix3_recon(data_dir, output_dir, working_dir):
    """Reconstruction workflow tests

    All supported reconstruction workflows get tested

    This tests the following features:
    - Blip-up + Blip-down DWI series for TOPUP/Eddy
    - Eddy is run on a CPU
    - Denoising is skipped
    - A follow-up reconstruction using the dsi_studio_gqi workflow

    Inputs
    ------
    - qsiprep single shell results (data/DSDTI_fmap)
    - qsiprep multi shell results (data/DSDTI_fmap)
    """
    TEST_NAME = "mrtrix3_recon"

    dataset_dir = download_test_data("multishell_output", data_dir)
    # XXX: Having to modify dataset_dirs is suboptimal.
    dataset_dir = os.path.join(dataset_dir, "multishell_output", "qsiprep")
    out_dir = os.path.join(output_dir, TEST_NAME)
    work_dir = os.path.join(working_dir, TEST_NAME)

    parameters = [
        dataset_dir,
        out_dir,
        "participant",
        f"-w={work_dir}",
        f"--recon-input={dataset_dir}",
        "--sloppy",
        "--recon-spec=mrtrix_multishell_msmt_ACT-fast",
        "--recon-only",
        "--output-resolution=5",
    ]

    _run_and_generate(TEST_NAME, parameters, test_main=True)


@pytest.mark.integration
@pytest.mark.tortoise_recon
def test_tortoise_recon(data_dir, output_dir, working_dir):
    """Test the TORTOISE recon workflow

    All supported reconstruction workflows get tested

    Inputs
    ------
    - qsiprep multi shell results (data/DSDTI_fmap)
    """
    TEST_NAME = "tortoise_recon"

    dataset_dir = download_test_data("multishell_output", data_dir)
    # XXX: Having to modify dataset_dirs is suboptimal.
    dataset_dir = os.path.join(dataset_dir, "multishell_output", "qsiprep")
    out_dir = os.path.join(output_dir, TEST_NAME)
    work_dir = os.path.join(working_dir, TEST_NAME)

    parameters = [
        dataset_dir,
        out_dir,
        "participant",
        f"-w={work_dir}",
        f"--recon-input={dataset_dir}",
        "--sloppy",
        "--recon-spec=TORTOISE",
        "--recon-only",
        "--output-resolution=5",
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

    parameters = [
        dataset_dir,
        out_dir,
        "participant",
        f"-w={work_dir}",
        "--sloppy",
        "--write-graph",
        "--output-resolution=5",
        "--hmc-model=3dSHORE",
        "--nthreads=1",
        "--omp-nthreads=1",
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

    parameters = [
        dataset_dir,
        out_dir,
        "participant",
        f"-w={work_dir}",
        "--sloppy",
        "--write-graph",
        "--output-resolution=5",
        "--nthreads=1",
        "--omp-nthreads=1",
    ]

    _run_and_generate(TEST_NAME, parameters, test_main=True)


def _run_and_generate(test_name, parameters, test_main=True):
    from qsiprep import config

    # TODO: Add --clean-workdir param to CLI
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
    optional_outputs_list = os.path.join(get_test_data_path(), f"{test_name}_optional_outputs.txt")
    if not os.path.isfile(optional_outputs_list):
        optional_outputs_list = None

    check_generated_files(config.execution.output_dir, output_list_file, optional_outputs_list)
