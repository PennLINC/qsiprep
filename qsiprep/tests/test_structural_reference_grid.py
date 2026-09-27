"""The structural handed to TORTOISE keeps its own voxel size.

DRBUDDI derives its working grid from the structural image's spacing and
refines it by 1.3x until it is under 1 mm. Resampling the T2w onto the DWI
grid (1.7 mm) therefore made DRBUDDI work at 0.77 mm instead of 1 mm, more
than doubling its memory, which exhausted an 8 GB GPU. The pre-alignment
workflow now resamples the structural onto the b=0 field of view at the
structural's own spacing.
"""

import nibabel as nb
import numpy as np
import pytest


def _image(shape, zooms, rotation_deg=0.0, origin=(0.0, 0.0, 0.0)):
    theta = np.radians(rotation_deg)
    rot = np.array(
        [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    )
    affine = np.eye(4)
    affine[:3, :3] = rot @ np.diag(zooms)
    affine[:3, 3] = origin
    img = nb.Nifti1Image(np.zeros(shape, dtype=np.float32), affine)
    img.header.set_zooms(zooms)
    return img


@pytest.mark.parametrize('rotation_deg', [0.0, 30.0])
def test_reference_grid_keeps_fov_and_takes_spacing(rotation_deg):
    from qsiprep.interfaces.images import reference_grid_at_spacing

    b0 = _image((10, 12, 8), (1.7, 1.7, 1.7), rotation_deg, origin=(5.0, -3.0, 2.0))
    grid = reference_grid_at_spacing(b0, (1.0, 1.0, 1.0))

    # voxel size comes from the requested spacing
    assert np.allclose(grid.header.get_zooms()[:3], (1.0, 1.0, 1.0))
    # the grid covers the b0 extent: ceil(10*1.7), ceil(12*1.7), ceil(8*1.7)
    assert grid.shape == (17, 21, 14)
    # same direction cosines as the b0
    b0_dirs = b0.affine[:3, :3] / np.array(b0.header.get_zooms()[:3])
    grid_dirs = grid.affine[:3, :3] / np.array(grid.header.get_zooms()[:3])
    assert np.allclose(b0_dirs, grid_dirs)
    # the first-voxel corner (index -0.5) is shared
    corner = np.array([-0.5, -0.5, -0.5, 1.0])
    assert np.allclose(b0.affine @ corner, grid.affine @ corner)


def test_reference_grid_at_native_spacing_is_identity():
    from qsiprep.interfaces.images import reference_grid_at_spacing

    b0 = _image((10, 12, 8), (1.7, 1.7, 1.7), 20.0, origin=(1.0, 2.0, 3.0))
    grid = reference_grid_at_spacing(b0, (1.7, 1.7, 1.7))
    assert grid.shape == b0.shape
    assert np.allclose(grid.affine, b0.affine)


def test_reference_grid_interface_writes_file(tmp_path):
    from qsiprep.interfaces.images import ReferenceGridAtSpacing

    b0_file = tmp_path / 'b0.nii.gz'
    t2w_file = tmp_path / 't2w.nii.gz'
    _image((10, 12, 8), (1.7, 1.7, 1.7)).to_filename(b0_file)
    _image((30, 30, 30), (1.0, 1.0, 1.0)).to_filename(t2w_file)

    result = ReferenceGridAtSpacing(fov_image=str(b0_file), spacing_image=str(t2w_file)).run(
        cwd=str(tmp_path)
    )
    grid = nb.load(result.outputs.out_file)
    assert grid.shape == (17, 21, 14)
    assert np.allclose(grid.header.get_zooms()[:3], (1.0, 1.0, 1.0))


def test_structural_alignment_resamples_onto_grid_at_structural_spacing():
    """The resampling reference is the grid node's output, not the b=0 itself."""
    from qsiprep import config
    from qsiprep.workflows.dwi.registration import init_structural_to_b0_alignment_wf

    config.nipype.omp_nthreads = 1
    wf = init_structural_to_b0_alignment_wf(name='t2w_to_b0_test')
    names = {n.name for n in wf._get_all_nodes()}
    assert 'reference_grid' in names

    reference_sources = {
        src.name
        for src, dest, meta in wf._graph.edges(data=True)
        if dest.name == 'resample_structural'
        and any(dest_field == 'reference_image' for _, dest_field in meta['connect'])
    }
    assert reference_sources == {'reference_grid'}
