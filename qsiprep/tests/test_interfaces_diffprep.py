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
