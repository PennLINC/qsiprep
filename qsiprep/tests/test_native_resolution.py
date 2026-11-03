"""Tests for per-output native resolutions and the axis-aligned output grids."""

import nibabel as nb
import numpy as np
import pytest

from qsiprep import config
from qsiprep.utils.spaces import (
    NativeResolutionMismatch,
    Resolution,
    bids_space,
    canonical_zooms,
    parse_output_spaces,
    parse_space_token,
    res_entities_for,
    resolve_native_zooms,
    resolved_zooms,
    spec_node_suffix,
    zooms_label,
)


def _write(path, zooms, affine=None, shape=(4, 4, 4)):
    if affine is None:
        affine = np.diag([*zooms, 1.0])
    nb.Nifti1Image(np.zeros(shape, dtype='float32'), affine).to_filename(str(path))
    return str(path)


def _sagittal_affine(zooms):
    """Return an affine whose voxel axes are (A, S, R) with ``zooms`` along them."""
    affine = np.zeros((4, 4))
    affine[1, 0] = zooms[0]  # i -> anterior
    affine[2, 1] = zooms[1]  # j -> superior
    affine[0, 2] = zooms[2]  # k -> right
    affine[3, 3] = 1
    return affine


def test_runs_with_one_voxel_size_resolve_to_it(tmp_path):
    files = [_write(tmp_path / f'run{i}.nii.gz', (2, 2, 2)) for i in range(2)]
    assert resolve_native_zooms(files) == (2.0, 2.0, 2.0)


def test_runs_with_different_voxel_sizes_are_refused(tmp_path):
    """Test that the mismatch names every run with its voxel size."""
    small = _write(tmp_path / 'sub-01_acq-hi_dwi.nii.gz', (1.5, 1.5, 1.5))
    large = _write(tmp_path / 'sub-01_acq-lo_dwi.nii.gz', (2, 2, 2))
    with pytest.raises(NativeResolutionMismatch) as excinfo:
        resolve_native_zooms([small, large])
    message = str(excinfo.value)
    assert 'sub-01_acq-hi_dwi.nii.gz: 1.5 mm isotropic' in message
    assert 'sub-01_acq-lo_dwi.nii.gz: 2 mm isotropic' in message
    assert excinfo.value.smallest == (1.5, 1.5, 1.5)


def test_a_sagittal_run_is_compared_axis_for_axis(tmp_path):
    """Test that zooms are read on the world axes, not the voxel axes.

    A sagittal run's slice axis is left-right, so its 3 mm slice spacing is the
    world x spacing even though it is the third voxel axis.
    """
    sagittal = _write(tmp_path / 'sag.nii.gz', None, affine=_sagittal_affine((1.5, 1.5, 3.0)))
    assert canonical_zooms(sagittal) == (3.0, 1.5, 1.5)
    axial = _write(tmp_path / 'ax.nii.gz', (3.0, 1.5, 1.5))
    assert resolve_native_zooms([sagittal, axial]) == (3.0, 1.5, 1.5)


@pytest.mark.parametrize(
    ('token', 'expected'),
    [
        ('acpc:res-nativemin', (1.5, 1.5, 1.5)),
        ('acpc:res-nativemax', (3.0, 3.0, 3.0)),
        ('acpc:res-native', (1.5, 2.0, 3.0)),
        ('acpc:res-2mm', (2.0, 2.0, 2.0)),
        ('anat:res-1p5x1p5x3mm', (1.5, 1.5, 3.0)),
    ],
)
def test_resolved_zooms(token, expected):
    (spec,) = parse_space_token(token)
    assert resolved_zooms(spec.resolution, (1.5, 2.0, 3.0)) == expected


def test_a_native_resolution_needs_the_runs():
    (spec,) = parse_space_token('acpc:res-nativemin')
    with pytest.raises(ValueError, match='voxel size of the DWI runs'):
        resolved_zooms(spec.resolution)


@pytest.mark.parametrize(
    ('zooms', 'label'),
    [((2, 2, 2), '2mm'), ((1.5, 1.5, 1.5), '1p5mm'), ((1.5, 1.5, 3), '1p5x1p5x3mm')],
)
def test_zooms_label(zooms, label):
    assert zooms_label(zooms) == label


@pytest.mark.parametrize(
    ('token', 'entities'),
    [
        ('acpc:res-native', {}),
        ('acpc:res-nativemin', {'res': 'nativemin'}),
        ('anat:res-2mm', {'res': '2mm'}),
    ],
)
def test_res_native_writes_no_res_entity(token, entities):
    """Test that only res-native, the runs' own grid, goes unlabeled."""
    (spec,) = parse_space_token(token)
    assert res_entities_for(spec) == entities


@pytest.mark.parametrize(
    ('token', 'level', 'expected'),
    [
        ('acpc', 'distortiongroup', 'ACPC'),
        ('anat', 'distortiongroup', 'anat'),
        ('dwiref', 'distortiongroup', 'distortiongroup'),
        ('dwiref', 'subject', 'subject'),
    ],
)
def test_bids_space(token, level, expected):
    """Test that dwiref is named by its resolved level, never space-dwiref."""
    (spec,) = parse_space_token(token)
    assert bids_space(spec, level) == expected


def test_node_suffixes_stay_historical_for_acpc_only_runs():
    single = parse_output_spaces(['acpc:res-2mm'])
    assert spec_node_suffix(single[0], single) == ''
    two = parse_output_spaces(['acpc:res-2mm', 'acpc:res-1p5mm'])
    assert [spec_node_suffix(spec, two) for spec in two] == ['_res2mm', '_res1p5mm']
    mixed = parse_output_spaces(['acpc:res-2mm', 'anat:res-2mm'])
    assert [spec_node_suffix(spec, mixed) for spec in mixed] == ['_acpc_res2mm', '_anat_res2mm']


def _oblique_affine(zooms, degrees=20):
    theta = np.deg2rad(degrees)
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    affine = np.eye(4)
    affine[:3, :3] = rotation @ np.diag(zooms)
    affine[:3, 3] = [-20, -30, -10]
    return affine


def test_grid_is_axis_aligned_lps_at_the_requested_size():
    """Test that an oblique reference still gives an axis-aligned LPS grid."""
    from qsiprep.interfaces.images import axis_aligned_grid

    fixed = nb.Nifti1Image(np.zeros((20, 20, 10), dtype='float32'), _oblique_affine((1, 1, 1)))
    grid = axis_aligned_grid(fixed, (1.5, 1.5, 3.0))
    assert grid.header.get_zooms()[:3] == pytest.approx((1.5, 1.5, 3.0))
    assert nb.aff2axcodes(grid.affine) == ('L', 'P', 'S')
    assert np.allclose(grid.affine[:3, :3], np.diag(np.diag(grid.affine[:3, :3])))


def test_grid_covers_the_mask_with_a_margin():
    """Test that every masked voxel center falls inside the grid, plus two voxels."""
    from qsiprep.interfaces.images import axis_aligned_grid

    affine = _oblique_affine((2, 2, 2))
    mask = np.zeros((20, 20, 10), dtype='uint8')
    mask[5:12, 6:14, 2:7] = 1
    mask_img = nb.Nifti1Image(mask, affine)
    fixed = nb.Nifti1Image(np.zeros((20, 20, 10), dtype='float32'), affine)
    grid = axis_aligned_grid(fixed, (2, 2, 2), mask_img=mask_img, margin=2)

    centers = (affine @ np.c_[np.argwhere(mask), np.ones(mask.sum())].T).T[:, :3]
    inverse = np.linalg.inv(grid.affine)
    voxels = (inverse @ np.c_[centers, np.ones(len(centers))].T).T[:, :3]
    shape = np.array(grid.shape)
    assert (voxels >= 2 - 1e-6).all()
    assert (voxels <= shape - 3 + 1e-6).all()
    # Cropped: much smaller than the whole field of view
    assert np.prod(shape) < np.prod(fixed.shape)


def test_sampling_grid_takes_the_zooms_of_an_image_on_its_world_axes(tmp_path):
    """Test that a sagittal image's slice spacing lands on the grid's x axis."""
    from qsiprep.interfaces.images import SamplingGrid

    fixed = _write(tmp_path / 'fixed.nii.gz', (1, 1, 1), shape=(30, 30, 30))
    moving = _write(tmp_path / 'sag.nii.gz', None, affine=_sagittal_affine((1.5, 1.5, 3.0)))
    result = SamplingGrid(fixed_image=fixed, zooms_image=moving).run(cwd=str(tmp_path))
    grid = nb.load(result.outputs.out_file)
    assert grid.header.get_zooms()[:3] == pytest.approx((3.0, 1.5, 1.5))
    assert nb.aff2axcodes(grid.affine) == ('L', 'P', 'S')


def test_interpolation_pairs_axes_by_world_direction(tmp_path):
    """Test that a sagittal run's upsampled axis is found by direction, not index.

    The run is 3 mm left-right (its third voxel axis) and 1.5 mm otherwise. A grid
    at 1.5 mm isotropic upsamples left-right, which index pairing misses.
    """
    from qsiprep.interfaces.images import ChooseInterpolator

    sagittal = _write(tmp_path / 'sag.nii.gz', None, affine=_sagittal_affine((1.5, 1.5, 3.0)))
    grid = _write(tmp_path / 'grid.nii.gz', (1.5, 1.5, 1.5))
    result = ChooseInterpolator(dwi_files=[sagittal], output_grid=grid).run(cwd=str(tmp_path))
    assert result.outputs.interpolation_method == 'Linear'

    same = _write(tmp_path / 'grid3.nii.gz', (3.0, 1.5, 1.5))
    result = ChooseInterpolator(dwi_files=[sagittal], output_grid=same).run(cwd=str(tmp_path))
    assert result.outputs.interpolation_method == 'LanczosWindowedSinc'


def test_acpc_grid_passes_every_axis_to_afni():
    """Test that an anisotropic ACPC size reaches 3dresample axis by axis."""
    from qsiprep.workflows.anatomical.volume import init_output_grid_wf

    wf = init_output_grid_wf((1.5, 1.5, 3.0))
    assert wf.get_node('resample_to_voxel_size').inputs.voxel_size == (1.5, 1.5, 3.0)


def test_methods_text_names_an_anisotropic_size():
    from qsiprep.workflows.dwi.resampling import init_dwi_trans_wf

    config.nipype.omp_nthreads = 1
    resolution = Resolution(kind='mm', label='1p5x1p5x3mm', zooms=(1.5, 1.5, 3.0))
    wf = init_dwi_trans_wf(source_file='sub-01_dwi.nii.gz', mem_gb=1, resolution=resolution)
    assert '1.5×1.5×3 mm voxels' in wf.__desc__
    assert 'isotropic' not in wf.__desc__.split('interpolation')[0]


def test_transport_stack_drops_what_the_resampling_drops():
    """Test that the Jacobian weights follow the DWI's chain, coreg and trailing included."""
    from qsiprep.interfaces.fmap import _transport_stack

    assert _transport_stack('affine.mat', None, 'coreg.mat') == ['coreg.mat', 'affine.mat']
    assert _transport_stack(
        'affine.mat',
        None,
        'coreg.mat',
        drop_stages=['b=0 to T1w'],
    ) == ['affine.mat']
    assert _transport_stack(None, None, 'coreg.mat', drop_stages=['b=0 to T1w']) == ['identity']
    assert _transport_stack(None, None, 'coreg.mat', trailing_transform='inv.mat') == [
        'inv.mat',
        'coreg.mat',
    ]


@pytest.mark.parametrize(
    ('module', 'name'),
    [
        ('qsiprep.interfaces.mrtrix', 'MRTrixGradientTable'),
        ('qsiprep.interfaces.dsi_studio', 'DSIStudioBTable'),
    ],
)
def test_gradient_writers_never_read_the_grid(module, name):
    """Test that the gradient tables cannot depend on an anisotropic grid.

    They are written from the rotated bvals and bvecs alone. That is safe because
    every grid is axis-aligned LPS, so the b-vectors' frame is the grid's.
    """
    import importlib

    interface = getattr(importlib.import_module(module), name)
    assert {'bval_file', 'bvec_file'} <= set(interface.input_spec().copyable_trait_names())
    image_inputs = {
        trait
        for trait in interface.input_spec().copyable_trait_names()
        if trait.endswith(('image', 'grid', 'nifti', 'dwi_file'))
    }
    assert not image_inputs
