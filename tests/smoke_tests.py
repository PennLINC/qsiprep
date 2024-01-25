import pytest
from pkg_resources import resource_filename as pkgrf
from qsiprep.workflows.base import init_qsiprep_wf


@pytest.mark.smoke
def test_preproc_pepolar_sdc(tmp_path):
    # Get the empty bids data
    bids_root = pkgrf("qsiprep", "data/abcd")
    work_dir = str(tmp_path.absolute() / "preproc_pepolar_sdc_work")
    output_dir = str(tmp_path.absolute() / "preproc_pepolar_sdc_output")
    bids_dir = bids_root
    subject_list = ["abcd"]
    wf = init_qsiprep_wf(
        subject_list=subject_list,
        run_uuid="test",
        work_dir=work_dir,
        output_dir=output_dir,
        bids_dir=bids_dir,
        ignore=[],
        debug=False,
        low_mem=False,
        anat_only=False,
        longitudinal=False,
        force_spatial_normalization=False,
        denoise_before_combining=True,
        dwi_denoise_window=7,
        combine_all_dwis=True,
        omp_nthreads=1,
        output_spaces=["T1w", "template"],
        template="MNI152NLin2009cAsym",
        output_resolution=1.25,
        motion_corr_to="iterative",
        b0_to_t1w_transform="Rigid",
        hmc_transform="Affine",
        hmc_model="3dSHORE",
        impute_slice_threshold=0,
        write_local_bvecs=False,
        fmap_bspline=False,
        fmap_demean=True,
        force_syn=False,
    )

    assert len(wf.list_node_names())


@pytest.mark.smoke
def test_preproc_syn_sdc(tmp_path):
    # Get the empty bids data
    bids_root = pkgrf("qsiprep", "data/abcd")
    work_dir = str(tmp_path.absolute() / "preproc_syn_sdc_work")
    output_dir = str(tmp_path.absolute() / "preproc_syn_sdc_output")
    bids_dir = bids_root
    subject_list = ["abcd"]
    wf = init_qsiprep_wf(
        subject_list=subject_list,
        run_uuid="test",
        work_dir=work_dir,
        output_dir=output_dir,
        bids_dir=bids_dir,
        ignore=[],
        debug=False,
        low_mem=False,
        anat_only=False,
        longitudinal=False,
        denoise_before_combining=True,
        dwi_denoise_window=7,
        combine_all_dwis=True,
        omp_nthreads=1,
        force_spatial_normalization=False,
        output_spaces=["T1w", "template"],
        template="MNI152NLin2009cAsym",
        output_resolution=1.25,
        motion_corr_to="iterative",
        b0_to_t1w_transform="Rigid",
        hmc_transform="Affine",
        hmc_model="3dSHORE",
        impute_slice_threshold=0,
        write_local_bvecs=False,
        fmap_bspline=False,
        fmap_demean=True,
        force_syn=True,
    )
    assert len(wf.list_node_names())


@pytest.mark.smoke
def test_preproc_nonehmc_sdc(tmp_path):
    # Get the empty bids data
    bids_root = pkgrf("qsiprep", "data/abcd")
    work_dir = str(tmp_path.absolute() / "preproc_nonehmc_work")
    output_dir = str(tmp_path.absolute() / "preproc_nonehmc_output")
    bids_dir = bids_root
    subject_list = ["abcd"]
    wf = init_qsiprep_wf(
        subject_list=subject_list,
        run_uuid="test",
        work_dir=work_dir,
        output_dir=output_dir,
        bids_dir=bids_dir,
        ignore=[],
        debug=False,
        low_mem=False,
        anat_only=False,
        longitudinal=False,
        denoise_before_combining=True,
        dwi_denoise_window=7,
        combine_all_dwis=True,
        omp_nthreads=1,
        output_spaces=["T1w", "template"],
        force_spatial_normalization=False,
        template="MNI152NLin2009cAsym",
        output_resolution=1.25,
        motion_corr_to="iterative",
        b0_to_t1w_transform="Rigid",
        hmc_transform="Affine",
        hmc_model="none",
        impute_slice_threshold=0,
        write_local_bvecs=False,
        fmap_bspline=False,
        fmap_demean=True,
        force_syn=False,
    )
    assert len(wf.list_node_names())


@pytest.mark.smoke
def test_preproc_buds(tmp_path):
    # Get the empty bids data
    bids_root = pkgrf("qsiprep", "data/buds")
    work_dir = str(tmp_path.absolute() / "preproc_buds_work")
    output_dir = str(tmp_path.absolute() / "preproc_buds_output")
    bids_dir = bids_root
    subject_list = ["1"]

    wf = init_qsiprep_wf(
        subject_list=subject_list,
        run_uuid="test",
        work_dir=work_dir,
        output_dir=output_dir,
        bids_dir=bids_dir,
        ignore=[],
        debug=False,
        low_mem=False,
        anat_only=False,
        longitudinal=False,
        denoise_before_combining=True,
        force_spatial_normalization=False,
        dwi_denoise_window=7,
        combine_all_dwis=True,
        omp_nthreads=1,
        output_spaces=["T1w", "template"],
        template="MNI152NLin2009cAsym",
        output_resolution=1.25,
        motion_corr_to="iterative",
        b0_to_t1w_transform="Rigid",
        hmc_transform="Affine",
        hmc_model="none",
        impute_slice_threshold=0,
        write_local_bvecs=False,
        fmap_bspline=False,
        fmap_demean=True,
        force_syn=False,
    )
    assert len(wf.list_node_names())
