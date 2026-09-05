"""Tests for the qsiprep.interfaces.mrtrix module."""

import os
import subprocess

import nibabel as nb
import numpy as np
import pytest
from traits.trait_errors import TraitError

from qsiprep import config
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


def _dwibiascorrect_option_is_accepted(tmp_path, flag):
    """Ask the ``dwibiascorrect`` on ``PATH`` whether it recognizes ``flag``.

    ``-help`` short-circuits MRtrix3's argument parsing -- even a nonsense option
    like ``-ants.TOTALLYBOGUS`` "succeeds" under ``-help`` -- so it cannot be used to
    probe whether an option is real. Instead this points the command at an input file
    that does not exist and lets real argument parsing run. A rejected option makes
    MRtrix3's App-level parser fail immediately with "unrecognized arguments" and a
    usage message; an accepted option makes dwibiascorrect proceed past parsing and
    fail later, when it actually tries (and fails) to open the missing input.

    ``-scratch`` keeps that attempt's scratch directory inside ``tmp_path`` --
    dwibiascorrect retains it on error for debugging, and without an explicit
    ``-scratch`` it defaults to the current working directory, leaking a
    ``dwibiascorrect-tmp-*`` directory into the repo when tests run with cwd there.
    The development branch additionally requires the ``-scratch`` directory to
    already exist (the released version creates it), so it is made ahead of time;
    a shared, pre-existing directory is safe to reuse across calls in the same test
    because dwibiascorrect nests each attempt in its own randomly-named subdirectory.
    The subprocess's own cwd is also pinned to ``tmp_path``, as a second guard against
    any other file dwibiascorrect might drop next to itself rather than under
    ``-scratch``.
    """
    scratch_dir = tmp_path / 'scratch'
    scratch_dir.mkdir(exist_ok=True)
    result = subprocess.run(
        [
            'dwibiascorrect',
            'ants',
            flag,
            '[150,3]',
            str(tmp_path / 'nonexistent_input.mif'),
            str(tmp_path / 'out.mif'),
            '-scratch',
            str(scratch_dir),
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )
    combined_output = result.stdout + result.stderr
    rejected = 'unrecognized arguments' in combined_output
    return not rejected, combined_output


@pytest.mark.parametrize('mrtrix_version', ['stable', 'dev'])
def test_dwibiascorrect_options_are_accepted_by_the_real_binary(
    monkeypatch, tmp_path, mrtrix_version
):
    """Ask the selected dwibiascorrect whether it knows the options QSIPrep passes.

    This is the only test that can catch the -ants.b/-ants_b break, because it needs
    a real MRtrix3 to parse the option. A bogus-option control proves the probe can
    tell rejection from acceptance in the first place, and the opposite spelling is
    checked too, so a regression in either direction shows up here.
    """
    if not (os.getenv('MRTRIX3_STABLE_HOME') and os.getenv('MRTRIX3_DEV_HOME')):
        pytest.skip('both MRtrix3 trees are only available inside the container')

    monkeypatch.setattr(config.workflow, 'mrtrix_version', mrtrix_version)
    # config.workflow.init() mutates os.environ['PATH'] directly, which monkeypatch
    # would not undo. Setting PATH through monkeypatch first registers it for
    # restoration at teardown.
    monkeypatch.setenv('PATH', os.environ.get('PATH', ''))
    config.workflow.init()

    own_separator = '_' if mrtrix_version == 'dev' else '.'
    other_separator = '.' if mrtrix_version == 'dev' else '_'

    accepted, output = _dwibiascorrect_option_is_accepted(tmp_path, f'-ants{own_separator}b')
    assert accepted, output

    # The other tree's spelling must be genuinely rejected by this tree's binary --
    # this is the -ants.b/-ants_b break itself, caught in whichever direction the
    # PATH reordering got wrong.
    other_accepted, other_output = _dwibiascorrect_option_is_accepted(
        tmp_path, f'-ants{other_separator}b'
    )
    assert not other_accepted, other_output

    # Control: a genuinely bogus option must also be rejected, or this probe would
    # report "accepted" no matter what dwibiascorrect actually did.
    bogus_accepted, bogus_output = _dwibiascorrect_option_is_accepted(
        tmp_path, '-ants.TOTALLYBOGUS'
    )
    assert not bogus_accepted, bogus_output
