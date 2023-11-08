import pkg_resources

from qsiprep.workflows import base


def test_roi_found(tmp_path):
    # ensure that a dataset with an image matching
    # sub*/anat/*roi.nii.gz
    # is picked up by the bidssrc
    bids_dir = pkg_resources.resource_filename("qsiprep", "data/lesion")
    output_dir = str(tmp_path.absolute() / "lesion_output")
    wf = base.init_single_subject_wf(
        subject_id="1",
        name="test_roi",
        reportlets_dir=tmp_path,
        output_dir=output_dir,
        bids_dir=bids_dir,
        ignore=[],
        debug=False,
        write_local_bvecs=False,
        low_mem=False,
        dwi_only=False,
        anat_only=True,
        longitudinal=False,
        b0_threshold=100,
        denoise_before_combining=True,
        bids_filters=None,
        anatomical_contrast="T1w",
        dwi_denoise_window=7,
        denoise_method="patch2self",
        unringing_method="mrdegibbs",
        b1_biascorrect_stage=False,
        no_b0_harmonization=False,
        infant_mode=False,
        combine_all_dwis=True,
        raw_image_sdc=False,
        distortion_group_merge="none",
        pepolar_method="TOPUP",
        omp_nthreads=1,
        skull_strip_template="OASIS",
        force_spatial_normalization=True,
        skull_strip_fixed_seed=False,
        freesurfer=False,
        hires=False,
        template="MNI152NLin2009cAsym",
        output_resolution=2.0,
        prefer_dedicated_fmaps=True,
        motion_corr_to="iterative",
        b0_to_t1w_transform="Rigid",
        intramodal_template_iters=0,
        intramodal_template_transform="Rigid",
        hmc_model="3dSHORE",
        hmc_transform="Affine",
        shoreline_iters=2,
        eddy_config=None,
        impute_slice_threshold=0.0,
        fmap_bspline=False,
        fmap_demean=True,
        use_syn=False,
        force_syn=False,
    )
    subject_data = wf.get_node("bidssrc").inputs.subject_data
    subject_data_has_roi = len(subject_data.get("roi")) == 1

    assert all([subject_data_has_roi])
