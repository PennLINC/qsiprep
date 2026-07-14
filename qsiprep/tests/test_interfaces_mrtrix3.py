"""Tests for the qsiprep.interfaces.dipy module."""

import os

import nibabel as nb
import pytest

from qsiprep import config
from qsiprep.interfaces import mrtrix
from qsiprep.workflows.dwi.merge import init_dwi_denoising_wf


def test_dwidenoise(datasets, tmp_path_factory):
    """Test qsiprep.interfaces.mrtrix.DWIDenoise."""
    tmpdir = tmp_path_factory.mktemp('test_dwidenoise')

    in_dir = datasets['forrest_gump']
    in_file = os.path.join(in_dir, 'sub-01/ses-forrestgump/dwi/sub-01_ses-forrestgump_dwi.nii.gz')
    in_img = nb.load(in_file)

    interface = mrtrix.DWIDenoise(
        shape='cuboid',
        extent=(5, 5, 5),
        in_file=in_file,
        nthreads=1,
    )
    results = interface.run(cwd=tmpdir)

    assert os.path.isfile(results.outputs.out_file)
    denoised_img = nb.load(results.outputs.out_file)
    assert denoised_img.shape == in_img.shape

    assert os.path.isfile(results.outputs.noise_image)
    noise_img = nb.load(results.outputs.noise_image)
    assert noise_img.shape == in_img.shape[:3]
    assert noise_img.ndim == 3

    assert os.path.isfile(results.outputs.out_report)
    assert os.path.isfile(results.outputs.nmse_text)


@pytest.mark.parametrize(
    ('shape', 'kernel_option', 'error'),
    [
        ('sphere', {'extent': (5, 5, 5)}, "'extent' cannot be used"),
        ('cuboid', {'radius': 2.5}, "'radius' cannot be used"),
    ],
)
def test_dwidenoise_kernel_shape_validation(tmp_path, shape, kernel_option, error):
    """Reject kernel options that do not apply to the selected shape."""
    in_file = tmp_path / 'dwi.nii.gz'
    in_file.touch()
    interface = mrtrix.DWIDenoise(in_file=in_file, shape=shape, **kernel_option)

    with pytest.raises(ValueError, match=error):
        _ = interface.cmdline


def test_dwidenoise_kernel_options_are_mutually_exclusive(tmp_path):
    """Reject simultaneous spherical and cuboid kernel size options."""
    in_file = tmp_path / 'dwi.nii.gz'
    in_file.touch()

    with pytest.raises(OSError, match='mutually exclusive'):
        mrtrix.DWIDenoise(
            in_file=in_file,
            shape='sphere',
            radius=2.5,
            extent=(5, 5, 5),
        )


def test_dwidenoise_cli_parameters_reach_workflow(monkeypatch):
    """Forward parsed DWIDenoise parameters to the workflow node."""
    monkeypatch.setattr(
        config.workflow,
        'denoise_method',
        'dwidenoise;demodulate:nonlinear;decomposition:bdcsvd',
    )
    monkeypatch.setattr(config.workflow, 'dwi_denoise_window', 5)
    monkeypatch.setattr(config.workflow, 'unringing_method', 'none')
    monkeypatch.setattr(config.workflow, 'no_b0_harmonization', True)
    monkeypatch.setattr(config.nipype, 'omp_nthreads', 1)

    workflow = init_dwi_denoising_wf(
        source_file='sub-01_dwi.nii.gz',
        partial_fourier=1.0,
        phase_encoding_direction='j',
        n_volumes=30,
        use_phase=False,
        do_biascorr=False,
    )
    denoiser = workflow.get_node('denoiser')

    assert denoiser.inputs.demodulate == 'nonlinear'
    assert denoiser.inputs.decomposition == 'bdcsvd'
