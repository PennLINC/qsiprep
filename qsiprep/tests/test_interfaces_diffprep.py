import numpy as np


def _write_dummy_nii(path):
    import nibabel as nb

    img = nb.Nifti1Image(np.zeros((4, 4, 4, 6), dtype='float32'), np.eye(4))
    img.to_filename(path)


def test_tortoiseprocess_cmdline(tmp_path):
    from qsiprep.interfaces.tortoise import TORTOISEProcess

    dwi = tmp_path / 'dwi.nii'
    _write_dummy_nii(str(dwi))
    bmtxt = tmp_path / 'dwi.bmtxt'
    bmtxt.write_text('0 0 0 0 0 0\n')
    mask = tmp_path / 'mask.nii'
    _write_dummy_nii(str(mask))

    iface = TORTOISEProcess(
        dwi_file=str(dwi),
        bmtxt_file=str(bmtxt),
        mask_file=str(mask),
        transformation_type='quadratic',
    )
    cmd = iface.cmdline
    assert cmd.startswith('TORTOISEProcess')
    assert 'dwi.nii' in cmd
    assert 'quadratic' in cmd
    assert 'dwi.bmtxt' in cmd
    assert 'mask.nii' in cmd


def test_diffprep_motion_params(tmp_path):
    from qsiprep.interfaces.tortoise import DIFFPREPMotionParams

    # Two volumes, 6 rigid params each (tx ty tz rx ry rz), space-separated.
    transforms = tmp_path / 'xfms.txt'
    transforms.write_text('0 0 0 0 0 0\n1.5 -2.0 0.5 0.01 -0.02 0.0\n')

    iface = DIFFPREPMotionParams(transforms_file=str(transforms))
    res = iface.run(cwd=str(tmp_path))
    out = np.loadtxt(res.outputs.motion_file)

    assert out.shape == (2, 6)
    assert np.allclose(out[1], [1.5, -2.0, 0.5, 0.01, -0.02, 0.0])


def test_bmat_to_fsl_roundtrip(tmp_path):
    from qsiprep.interfaces.tortoise import BmatToFSLGradients, make_bmat_file  # noqa: F401  # make_bmat_file imported per brief; its FSLBVecsToTORTOISEBmatrix binary is unavailable offline, so the .bmtxt is hand-built below

    # Build a .bmtxt directly (FSLBVecsToTORTOISEBmatrix binary unavailable in CI).
    # TORTOISE b-matrix format: one row per volume, 6 entries [Bxx Bxy Bxz Byy Byz Bzz]
    # where B = b * g * g^T for unit gradient g.
    #   vol 0: b=0,    g=[0,0,0]          -> B=0
    #   vol 1: b=1000, g=[1,0,0]          -> Bxx=1000, rest=0
    #   vol 2: b=1000, g=[0,1,0]          -> Byy=1000, rest=0
    #   vol 3: b=2000, g=[0.7071,0.7071,0]-> Bxx=Bxy=Byy=1000, rest=0
    bmtxt_path = tmp_path / 'in.bmtxt'
    bmtxt_path.write_text(
        '0 0 0 0 0 0\n'
        '1000 0 0 0 0 0\n'
        '0 0 0 1000 0 0\n'
        '1000 1000 0 1000 0 0\n'
    )

    iface = BmatToFSLGradients(bmtxt_file=str(bmtxt_path))
    res = iface.run(cwd=str(tmp_path))

    out_bvals = np.loadtxt(res.outputs.bval_file, ndmin=1)
    out_bvecs = np.loadtxt(res.outputs.bvec_file, ndmin=2)

    assert np.allclose(out_bvals, [0, 1000, 1000, 2000], atol=1.0)
    # First weighted dir aligns with x; sign is arbitrary, compare abs.
    assert np.allclose(np.abs(out_bvecs[:, 1]), [1, 0, 0], atol=1e-3)
    # Off-diagonal volume (index 3) is the only one that non-degenerately
    # exercises principal-eigenvector selection; an argmin/argmax swap would
    # pass the diagonal volumes but fail here. Sign is arbitrary -> compare abs.
    assert np.allclose(np.abs(out_bvecs[:, 3]), [np.sqrt(0.5), np.sqrt(0.5), 0], atol=1e-3)


def test_init_diffprep_hmc_wf_contract():
    from qsiprep import config
    from qsiprep.workflows.dwi.diffprep import init_diffprep_hmc_wf

    config.workflow.b0_threshold = 100
    scan_groups = {
        'dwi_series': ['/data/sub-01_dwi.nii.gz'],
        'fieldmap_info': {'suffix': None},
        'dwi_series_pedir': 'j',
    }
    wf = init_diffprep_hmc_wf(
        scan_groups=scan_groups,
        source_file='/data/sub-01_dwi.nii.gz',
        t2w_sdc='',
        transformation_type='quadratic',
    )
    outputnode = wf.get_node('outputnode')
    required = {
        'dwi_files_to_transform',
        'bvec_files_to_transform',
        'bval_files',
        'b0_indices',
        'to_dwi_ref_affines',
        'b0_template',
        'b0_template_mask',
        'sdc_method',
        'motion_params',
    }
    assert required.issubset(set(outputnode.inputs.copyable_trait_names()))
    # Fix 1: _n_vols should be at module level (picklable under MultiProc)
    from qsiprep.workflows.dwi import diffprep as diffprep_mod

    assert callable(getattr(diffprep_mod, '_n_vols', None)), '_n_vols must be module-level'
    # Fix 2: convert mask comes from DWI-space pre-HMC b0 reference, not t1_mask
    assert wf.get_node('pre_hmc_extract_b0s') is not None
    assert wf.get_node('pre_hmc_b0_ref') is not None
    assert wf.get_node('convert') is not None


def test_diffprep_order_helper():
    from qsiprep.workflows.dwi.base import _diffprep_order

    assert _diffprep_order('diffprep_motion') == 'motion'
    assert _diffprep_order('diffprep_quadratic') == 'quadratic'
    assert _diffprep_order('diffprep_cubic') == 'cubic'
