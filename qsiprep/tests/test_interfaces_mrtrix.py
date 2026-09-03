"""Tests for the qsiprep.interfaces.mrtrix module."""

import os

import nibabel as nb
import numpy as np
import pytest
from traits.trait_errors import TraitError

from qsiprep.interfaces import mrtrix
from qsiprep.tests.utils import field_of_view


def test_dwidenoise(datasets, tmp_path_factory):
    """Test qsiprep.interfaces.mrtrix.DWIDenoise."""
    tmpdir = tmp_path_factory.mktemp('test_dwidenoise')

    in_dir = datasets['forrest_gump']
    in_file = os.path.join(in_dir, 'sub-01/ses-forrestgump/dwi/sub-01_ses-forrestgump_dwi.nii.gz')
    in_img = nb.load(in_file)

    interface = mrtrix.DWIDenoise(
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


def test_dwidenoise2(datasets, tmp_path_factory):
    """Test qsiprep.interfaces.mrtrix.DWIDenoise2."""
    tmpdir = tmp_path_factory.mktemp('test_dwidenoise2')

    in_dir = datasets['forrest_gump']
    in_file = os.path.join(in_dir, 'sub-01/ses-forrestgump/dwi/sub-01_ses-forrestgump_dwi.nii.gz')
    in_img = nb.load(in_file)

    interface = mrtrix.DWIDenoise2(
        in_file=in_file,
        nthreads=1,
    )
    results = interface.run(cwd=tmpdir)

    assert os.path.isfile(results.outputs.out_file)
    denoised_img = nb.load(results.outputs.out_file)
    assert denoised_img.shape == in_img.shape

    assert os.path.isfile(results.outputs.noise_image)
    noise_img = nb.load(results.outputs.noise_image)
    assert noise_img.ndim == 3
    # The schedule estimates noise on a subsampled grid, so the noise map is coarser
    # than the input but must cover the input's field of view.
    assert all(n <= i for n, i in zip(noise_img.shape, in_img.shape[:3], strict=True))
    overshoot = field_of_view(noise_img) - field_of_view(in_img)
    assert np.all(overshoot > -1e-4)
    assert np.all(overshoot < np.array(noise_img.header.get_zooms()[:3]))

    assert os.path.isfile(results.outputs.out_report)
    assert os.path.isfile(results.outputs.nmse_text)


@pytest.mark.parametrize(
    'kernel_option',
    ['shape', 'radius', 'extent', 'aspect_ratio', 'minvoxels', 'subsample', 'onepass'],
)
def test_dwidenoise2_has_no_kernel_options(tmp_path, kernel_option):
    """The kernel and subsampling come from the schedule, not from command-line options."""
    in_file = tmp_path / 'dwi.nii.gz'
    in_file.touch()

    with pytest.raises(TraitError, match='undefined'):
        mrtrix.DWIDenoise2(in_file=in_file, **{kernel_option: 1})


def test_dwidenoise2_passes_schedule(tmp_path):
    """Select a bundled noise estimation schedule by name."""
    in_file = tmp_path / 'dwi.nii.gz'
    in_file.touch()

    interface = mrtrix.DWIDenoise2(in_file=in_file, schedule='vlarge')

    assert '-schedule vlarge' in interface.cmdline


def test_dwidenoise2_formats_fslgrad(tmp_path):
    """Pass the bvec and bval files to dwidenoise2 as a single -fslgrad option."""
    in_file = tmp_path / 'dwi.nii.gz'
    bvec_file = tmp_path / 'dwi.bvec'
    bval_file = tmp_path / 'dwi.bval'
    for path in (in_file, bvec_file, bval_file):
        path.touch()

    interface = mrtrix.DWIDenoise2(
        in_file=in_file,
        bvec_file=bvec_file,
        bval_file=bval_file,
    )

    assert f'-fslgrad {bvec_file} {bval_file}' in interface.cmdline


@pytest.mark.parametrize(
    ('mrtrix_version', 'separator', 'rejected'),
    [('stable', '.', '-ants_'), ('dev', '_', '-ants.')],
)
def test_dwibiascorrect_ants_option_spelling(tmp_path, mrtrix_version, separator, rejected):
    """Spell the N4 options the way the selected MRtrix3 expects.

    3.0.x uses -ants.b and rejects the underscore form; the development branch
    renamed them to -ants_b and rejects the dot form. These options are emitted on
    every run, so getting this wrong fails every bias-correction node at runtime.
    """
    in_file = tmp_path / 'dwi.nii.gz'
    in_file.touch()

    interface = mrtrix.DWIBiasCorrect(
        method='ants',
        in_file=in_file,
        ants_s='4',
        mrtrix_version=mrtrix_version,
    )
    cmdline = interface.cmdline

    assert f'-ants{separator}b [150,3]' in cmdline
    assert f'-ants{separator}c [200x200,1e-6]' in cmdline
    assert f'-ants{separator}s 4' in cmdline
    assert rejected not in cmdline
    # mrtrix_version selects a spelling; it is not itself an mrtrix option
    assert 'mrtrix_version' not in cmdline
    assert '--mrtrix' not in cmdline


def test_dwibiascorrect_defaults_to_stable_spelling(tmp_path):
    """A bare DWIBiasCorrect() matches the released MRtrix3, like the CLI default."""
    in_file = tmp_path / 'dwi.nii.gz'
    in_file.touch()

    cmdline = mrtrix.DWIBiasCorrect(method='ants', in_file=in_file).cmdline
    assert '-ants.b [150,3]' in cmdline
    assert '-ants_' not in cmdline


def test_mrdegibbs_dimensionality_is_optional(tmp_path):
    """Leave -dimensionality off unless it is set, so the default stays 2D slice-wise."""
    in_file = tmp_path / 'dwi.nii.gz'
    in_file.touch()

    assert '-dimensionality' not in mrtrix.MRDeGibbs(in_file=in_file).cmdline
    assert '-dimensionality 3' in mrtrix.MRDeGibbs(in_file=in_file, dimensionality=3).cmdline


def test_mrdegibbs_report_handles_complex_input(monkeypatch, tmp_path):
    """Generate the unringing report from complex-valued data, using the magnitude.

    mrdegibbs on MRtrix3's development branch emits complex data when it is given
    complex data. nibabel's get_fdata() does not raise on a complex image: it emits a
    ComplexWarning and silently returns the real component, discarding the imaginary
    part. Without an explicit magnitude conversion the report would be drawn from that
    real part instead of the true magnitude (sqrt(real**2 + imag**2)) -- a wrong
    picture rather than a crash. This test picks a phase that makes the two
    unmistakably different values, then inspects what actually reaches the plotting
    layer (rather than only checking that a report file got written, which happens
    either way) to confirm it is the magnitude.
    """
    import nibabel as nb
    import numpy as np

    rng = np.random.default_rng(0)
    shape = (8, 8, 4, 3)
    affine = np.eye(4)

    # A fixed, non-zero phase whose cosine (~0.54) is unmistakably different from 1:
    # using the real part in place of the magnitude scales every voxel down by that
    # factor, easily distinguished at float32 precision. It stays positive enough
    # that threshold_img(…, 50) still finds a non-empty mask even if the fix is
    # reverted and the real part leaks through.
    phase = 1.0
    magnitude = rng.uniform(150, 400, shape)
    complex_factor = np.exp(1j * phase)

    def _write(name, mag):
        path = tmp_path / name
        img = nb.Nifti1Image((mag * complex_factor).astype(np.complex64), affine)
        img.header.set_data_dtype(np.complex64)
        img.to_filename(path)
        return str(path)

    in_file = _write('in.nii.gz', magnitude)
    denoised_magnitude = magnitude * 0.99
    out_file = _write('out.nii.gz', denoised_magnitude)

    interface = mrtrix.MRDeGibbs(in_file=in_file)
    interface._out_report = str(tmp_path / 'report.svg')
    # Bypass nipype's name_source filename derivation and the NMSE CSV write: this test
    # is about what reaches the plotting layer, not about how output filenames are built.
    monkeypatch.setattr(
        mrtrix.MRDeGibbs,
        '_get_plotting_images',
        lambda self: (nb.load(in_file), nb.load(out_file), None),
    )
    monkeypatch.setattr(
        mrtrix.MRDeGibbs,
        '_calculate_nmse',
        lambda self, original_nii, corrected_nii: None,
    )

    # Record the images that actually reach plot_denoise, instead of only trusting
    # that a report file being written proves anything about its content.
    recorded_calls = []

    def _record_plot_denoise(lowb_nii, highb_nii, *args, **kwargs):
        recorded_calls.append((lowb_nii.get_fdata().copy(), highb_nii.get_fdata().copy()))
        return object()

    def _fake_compose_view(*args, **kwargs):
        with open(kwargs['out_file'], 'w') as fobj:
            fobj.write('<svg/>')

    monkeypatch.setattr(mrtrix, 'plot_denoise', _record_plot_denoise)
    monkeypatch.setattr(mrtrix, 'compose_view', _fake_compose_view)

    interface._generate_report()

    assert (tmp_path / 'report.svg').is_file()
    assert recorded_calls, 'plot_denoise was never called'

    # The first plot_denoise call is the moving-image (denoised low-b/high-b) pair --
    # values taken straight from denoised_nii with no subtraction involved -- so they
    # must equal the true magnitude of the denoised data, not the phase-scaled real
    # part that leaks through if the report skips _to_magnitude().
    mean_per_volume = magnitude.mean(axis=tuple(range(magnitude.ndim - 1)))
    lowb_index = int(np.argmax(mean_per_volume))
    highb_index = int(np.argmin(mean_per_volume))
    expected_lowb = denoised_magnitude[..., lowb_index].astype(np.float32)
    expected_highb = denoised_magnitude[..., highb_index].astype(np.float32)

    recorded_lowb, recorded_highb = recorded_calls[0]
    np.testing.assert_allclose(recorded_lowb, expected_lowb, rtol=1e-3, atol=1e-2)
    np.testing.assert_allclose(recorded_highb, expected_highb, rtol=1e-3, atol=1e-2)

    # The second plot_denoise call is the fixed-image ("Estimated Ringing") pair:
    # input_dwi minus denoised_nii. It is the only thing that guards the input_dwi
    # conversion -- the first call touches denoised_nii alone, so dropping
    # `input_dwi = _to_magnitude(input_dwi)` leaves it passing while the ringing panel
    # (and the NMSE that _calculate_nmse writes to a user-facing confounds CSV) is
    # computed from the phase-scaled real part.
    expected_diff_lowb = (magnitude - denoised_magnitude)[..., lowb_index].astype(np.float32)
    expected_diff_highb = (magnitude - denoised_magnitude)[..., highb_index].astype(np.float32)

    recorded_diff_lowb, recorded_diff_highb = recorded_calls[1]
    np.testing.assert_allclose(recorded_diff_lowb, expected_diff_lowb, rtol=1e-3, atol=1e-2)
    np.testing.assert_allclose(recorded_diff_highb, expected_diff_highb, rtol=1e-3, atol=1e-2)
