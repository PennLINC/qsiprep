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


def test_structural_alignment_refines_the_rotation_search_with_a_masked_rigid():
    """The rotation search gives the capture range; a rigid registration from it, the accuracy."""
    from nipype.interfaces.ants import Registration

    from qsiprep import config
    from qsiprep.workflows.dwi.registration import init_structural_to_b0_alignment_wf

    config.nipype.omp_nthreads = 1
    wf = init_structural_to_b0_alignment_wf(name='t2w_to_b0_test')
    refine = wf.get_node('refine_alignment')
    assert type(refine.interface) is Registration
    sources = {
        dest_field: (src.name, src_field)
        for src, dest, meta in wf._graph.edges(data=True)
        if dest.name == 'refine_alignment'
        for src_field, dest_field in meta['connect']
    }
    assert sources['initial_moving_transform'] == (
        'rotation_search_wf',
        'outputnode.initial_transform',
    )
    assert sources['fixed_image'] == ('inputnode', 'b0_ref')
    assert sources['moving_image'] == ('inputnode', 'structural_image')
    # Unmasked unless a caller supplies a b=0 brain mask
    assert 'fixed_image_masks' not in sources
    assert 'Affine' not in refine.inputs.transforms
    transform_sources = {
        (src.name, src_field)
        for src, dest, meta in wf._graph.edges(data=True)
        if dest.name == 'resample_structural'
        for src_field, dest_field in meta['connect']
        if dest_field == 'transforms'
    }
    assert transform_sources == {('refine_alignment', 'composite_transform')}


def test_structural_alignment_dilates_a_supplied_brain_mask():
    """Test that a brain mask restricts the rigid only after dilation.

    The head boundary has to stay in the metric.
    """
    from qsiprep import config
    from qsiprep.workflows.dwi.registration import (
        FIXED_MASK_MARGIN_MM,
        init_structural_to_b0_alignment_wf,
    )

    config.nipype.omp_nthreads = 1
    wf = init_structural_to_b0_alignment_wf(name='t2w_to_b0_test', masked=True)
    edges = {
        (src.name, src_field, dest.name, dest_field)
        for src, dest, meta in wf._graph.edges(data=True)
        for src_field, dest_field in meta['connect']
    }
    assert ('inputnode', 'b0_mask', 'dilate_mask', 'in_file') in edges
    assert ('dilate_mask', 'out_file', 'refine_alignment', 'fixed_image_masks') in edges
    assert not any(
        e[2:] == ('refine_alignment', 'fixed_image_masks') and e[0] == 'inputnode' for e in edges
    )
    assert wf.get_node('dilate_mask').inputs.margin_mm == FIXED_MASK_MARGIN_MM


def test_dilate_mask_mm_uses_millimetres_on_the_coarsest_needed_axis(tmp_path, monkeypatch):
    """The margin is in millimetres: a 10 mm margin on a 2 mm grid grows the mask by 5 voxels."""
    import nibabel as nb
    import numpy as np

    from qsiprep.workflows.dwi.registration import _dilate_mask_mm

    monkeypatch.chdir(tmp_path)
    data = np.zeros((41, 41, 41), dtype='uint8')
    data[20, 20, 20] = 1
    for zooms, expected in (((2.0, 2.0, 2.0), 5), ((1.0, 1.0, 2.5), 10)):
        img = nb.Nifti1Image(data, np.diag([*zooms, 1.0]))
        img.header.set_zooms(zooms)
        in_file = tmp_path / f'mask_{zooms[0]}.nii.gz'
        img.to_filename(in_file)
        out = np.asanyarray(nb.load(_dilate_mask_mm(str(in_file), 10.0)).dataobj) > 0
        # binary_dilation's default cross structure reaches `iterations` voxels along each axis
        assert out[20 + expected, 20, 20]
        assert not out[20 + expected + 1, 20, 20]
