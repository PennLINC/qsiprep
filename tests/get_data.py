import sys
sys.path.append("..")
import os
import argparse
import pytest
from urllib.request import urlretrieve
from unittest import mock  # python 3.3+
from qsiprep.cli.run import main as cli_main
from pkg_resources import resource_filename as pkgrf
from qsiprep.workflows.base import init_qsiprep_wf
import tempfile
import os.path as op
from copy import deepcopy
import base64
import shutil
from glob import glob

URL_PREFIX = "https://upenn.box.com/shared/static/"
DATA_LOCATIONS = {
    "realistic_highmotion": "qdk3o0mdlkqczoo7tsz1zwe2c10je2fr.xz",
    "realistic_nomotion": "wxyrjjtmku3ttb6jim4aeja7j5uo84zg.xz",
    "realistic_lowmotion": "66k7abc3zr5ihigkmelg88r2m26erzez.xz",
    "noisefree_lowmotion": "l99ualupkgkgc8ql3tllad6v7nxu2fsd.xz",
    "noisefree_highmotion": "e9ekewjiel0xefkz8aa827pit7uso857.xz",
    "noisefree_nomotion": "g42654lj0luw88pzuaepvwfsk2t5144g.xz"
}
RUN_ANAT = False
DSCSDSI_URL = "yes38tqb0a1vcem5su5ye2nyncddetkb.xz"
ANAT_URL = "9aan3r9xdzpbc1t4gchn67jwnixeb8ff.xz"
PREPROCESSED = "3qbln4gcqz2rxbnf7ras1gp7wyqdiray.xz"

# Can't use pytest's temp_dir because input directories have to be mocked
WORKING_DIR = tempfile.mkdtemp()
WORKING_SINGLE_DIR = tempfile.mkdtemp()
BIDS_DIR = op.join(WORKING_DIR, "DSCSDSI")
RECON_INPUT = op.join(WORKING_DIR, "output/qsiprep")
FS_LICENSE = op.join(WORKING_DIR, "license.txt")
os.environ['FS_LICENSE'] = FS_LICENSE
LICENSE_CODE = "bWF0dGhldy5jaWVzbGFrQHBzeWNoLnVjc2IuZWR1C" \
               "jIwNzA2CipDZmVWZEg1VVQ4clkKRlNCWVouVWtlVElDdwo="
with open(FS_LICENSE, "w") as f:
    f.write(base64.b64decode(LICENSE_CODE).decode())


def get_default_cli_args():
    return argparse.Namespace(
        bids_dir=BIDS_DIR,
        analysis_level="participant",
        participant_label=["tester"],
        run_uuid="test_run",
        work_dir='',
        ignore=[],
        recon_only=False,
        hires=False,
        freesurfer=False,
        do_reconall=False,
        debug=True,
        low_mem=False,
        anat_only=False,
        longitudinal=False,
        combine_all_dwis=True,
        dwi_denoise_window=0,
        denoise_before_combining=True,
        write_local_bvecs=False,
        omp_nthreads=1,
        skull_strip_template="OASIS",
        skull_strip_fixed_seed=False,
        force_spatial_normalization=False,
        output_resolution=5,
        template="MNI152NLin2009cAsym",
        b0_motion_corr_to="iterative",
        fs_license_file=FS_LICENSE,
        hmc_transform="Rigid",
        hmc_model="3dSHORE",
        impute_slice_threshold=0,
        b0_to_t1w_transform="Rigid",
        prefer_dedicated_fmaps=False,
        fmap_bspline=False,
        fmap_no_demean=False,
        use_syn_sdc=False,
        force_syn=False,
        verbose_count=2,
        recon_input=None,
        recon_spec=None,
        use_plugin=None,
        nthreads=1,
        mem_mb=None,
        stop_on_first_crash=True,
        resource_monitor=False,
        reports_only=False,
        sloppy=True,
        write_graph=True,
        boilerplate=False
    )


@pytest.fixture(scope="session")
def bids_data():
    """Download downsampled CS-DSI data."""
    status = os.system('curl -sSL {} | tar xvfJ - -C {}'.format(
        URL_PREFIX + DSCSDSI_URL, WORKING_DIR))
    assert status == 0
    assert 'DSCSDSI' in os.listdir(WORKING_DIR)


@pytest.fixture(scope="session")
def bids_singlescan_data():
    """Download downsampled CS-DSI data."""
    status = os.system('curl -sSL {} | tar xvfJ - -C {}'.format(
        URL_PREFIX + DSCSDSI_URL, WORKING_SINGLE_DIR))
    assert status == 0
    assert 'DSCSDSI' in os.listdir(WORKING_SINGLE_DIR)
    to_delete = glob(WORKING_SINGLE_DIR + '/DSCSDSI/sub-tester/dwi/*HASC55PA*')
    for delete_me in to_delete:
        os.remove(delete_me)


@pytest.fixture(scope="session")
def preprocessed_data():
    """Download downsampled CS-DSI data."""
    status = os.system('curl -sSL {} | tar xvfJ - -C {}'.format(
        URL_PREFIX + PREPROCESSED, WORKING_DIR))
    assert status == 0
    assert 'output' in os.listdir(WORKING_DIR)


def link_anatomicals(new_dir):
    old_wd = op.join(WORKING_DIR, "qsiprep_wf")
    new_wd = op.join(new_dir, "qsiprep_wf")
    old_anat_dir = op.join(old_wd, "single_subject_tester_wf/anat_preproc_wf")
    new_anat_dir = op.join(new_wd, "single_subject_tester_wf/anat_preproc_wf")

    # Make new pipeline directory
    shutil.copytree(old_anat_dir, new_anat_dir)

    return True


def pytest_sessionfinish(session, exitstatus):
    """ whole test run finishes. """
    shutil.rmtree(WORKING_DIR)
    shutil.rmtree(WORKING_SINGLE_DIR)
