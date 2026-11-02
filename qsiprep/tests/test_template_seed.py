"""The iterative template starts from the inputs registered to the first and moved to their mean.

Starting from the plain average of misaligned inputs leaves a copy of every input in the
target, and registration aligns each input to its own copy (#917, #1199). The first round now
registers to the first input instead and resamples every input once, through its own
transform and the inverse average, to their mean position. Later rounds are unchanged, and
the anatomical merge mask moves with the template.
"""

import shutil

import nibabel as nb
import numpy as np
import pytest
from scipy import ndimage
from scipy.linalg import expm, logm

LPS = np.diag([-1.0, -1.0, 1.0, 1.0])


def _config():
    from qsiprep import config

    config.execution.sloppy = False
    config.nipype.omp_nthreads = 1
    config.workflow.hmc_transform = 'Rigid'
    return config


def _connections(wf, src, dst):
    return wf._graph.get_edge_data(wf.get_node(src), wf.get_node(dst))['connect']


def test_midpoint_stack_applies_the_inverse_average_first():
    """ANTs applies the first-listed transform to the point first (test_ants_transform_order)."""
    from qsiprep.workflows.dwi.hmc import _midpoint_transforms

    transforms, flags = _midpoint_transforms('avg.mat', [['a.mat'], ['b.mat']])
    assert transforms == [['avg.mat', 'a.mat'], ['avg.mat', 'b.mat']]
    assert flags == [[True, False], [True, False]]


def test_seed_round_registers_to_the_first_input():
    from qsiprep.workflows.dwi.hmc import linear_alignment_workflow

    _config()
    seed = linear_alignment_workflow(seed_from_first=True, settings='unbiased_template')
    fixed = [c[0] for c in _connections(seed, 'inputnode', 'reg_000') if c[1] == 'fixed_image']
    assert len(fixed) == 1
    # nipype stores a connection function as its source
    assert fixed[0][0] == 'image_paths'
    assert 'def first_image' in fixed[0][1]
    names = seed.list_node_names()
    assert 'to_midpoint' in names
    assert 'invert_average' not in names


@pytest.mark.parametrize('use_masks', [True, False])
def test_later_rounds_take_the_previous_template_and_mask(use_masks):
    from qsiprep.workflows.dwi.hmc import init_b0_hmc_wf

    _config()
    wf = init_b0_hmc_wf(
        align_to='iterative',
        transform='Rigid',
        use_masks=use_masks,
        num_iters=3,
        settings='unbiased_template',
    )
    for prev, this in [(0, 1), (1, 2)]:
        carried = _connections(
            wf, f'iterative_alignment_{prev:03d}', f'iterative_alignment_{this:03d}'
        )
        assert ('outputnode.updated_template', 'inputnode.template_image') in carried
        assert (('outputnode.updated_mask', 'inputnode.template_mask') in carried) is use_masks
        from_input = _connections(wf, 'inputnode', f'iterative_alignment_{this:03d}')
        assert from_input == [('b0_images', 'inputnode.image_paths')]


def test_final_template_is_the_target_of_the_last_round():
    from qsiprep.workflows.dwi.hmc import init_b0_hmc_wf

    _config()
    wf = init_b0_hmc_wf(align_to='iterative', transform='Rigid', num_iters=2)
    assert ('outputnode.updated_template', 'final_template') in _connections(
        wf, 'iterative_alignment_000', 'outputnode'
    )
    assert ('outputnode.affine_transforms', 'forward_transforms') in _connections(
        wf, 'iterative_alignment_001', 'outputnode'
    )


def test_nonlinear_dwiref_starts_mvtc2_from_the_seed():
    from qsiprep.workflows.dwi.dwiref import init_dwiref_wf

    _config().workflow.dwi2anat_dof = 6
    wf = init_dwiref_wf(
        inputs_list=['group_a', 'group_b'],
        t1w_source_file='/data/sub-01_T1w.nii.gz',
        transform='BSplineSyN',
    )
    assert _connections(wf, 'iterative_alignment_000', 'ants_mvtc2') == [
        ('outputnode.updated_template', 'initial_template')
    ]


def test_mvtc2_passes_the_initial_template(tmp_path):
    from qsiprep.interfaces.ants import MultivariateTemplateConstruction2

    seed = tmp_path / 'seed.nii.gz'
    seed.touch()
    mvtc2 = MultivariateTemplateConstruction2(initial_template=str(seed))
    assert f'-z {seed}' in mvtc2.cmdline


def _rigid(rotvec_deg, shift_mm):
    from scipy.spatial.transform import Rotation

    out = np.eye(4)
    out[:3, :3] = Rotation.from_rotvec(np.radians(rotvec_deg)).as_matrix()
    out[:3, 3] = shift_mm
    return out


def _write_moved(src, affine, moved_to_src, path):
    """Write ``src(moved_to_src(y))``: the object moved by the inverse of ``moved_to_src``."""
    vox = np.linalg.inv(affine) @ moved_to_src @ affine
    data = ndimage.affine_transform(src, vox[:3, :3], offset=vox[:3, 3], order=3)
    nb.Nifti1Image(data.astype('float32'), affine).to_filename(path)
    return str(path)


@pytest.mark.skipif(
    shutil.which('antsRegistration') is None, reason='antsRegistration not installed'
)
def test_template_sits_at_the_mean_position(tmp_path):
    """Three moved copies of one object: the template lands at their mean position.

    Rotations about different axes make the transform stack's order matter, so the
    midpoint resamples only line up when the inverse average is applied first.
    """
    from qsiprep.interfaces.itk import linear_transform_matrix
    from qsiprep.workflows.dwi.hmc import init_b0_hmc_wf

    _config()
    shape, voxel = (48, 48, 48), 2.5
    affine = np.diag([voxel, voxel, voxel, 1.0])
    affine[:3, 3] = -voxel * (np.array(shape) - 1) / 2
    grid = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing='ij'), -1)
    xyz = grid * voxel + affine[:3, 3]
    envelope = (((xyz / [42.0, 36.0, 32.0]) ** 2).sum(-1) < 1).astype(float)
    texture = ndimage.gaussian_filter(np.random.default_rng(0).standard_normal(shape), 2)
    src = ndimage.gaussian_filter(envelope, 1) * (1 + 0.5 * texture / texture.std())
    src = np.clip(src, 0, None)

    # RAS point maps from each image to the source
    to_src = [
        _rigid([0, 0, 9], [5, -4, 2]),
        _rigid([10, 0, 0], [-6, 5, -3]),
        _rigid([0, -8, 0], [2, 6, 5]),
    ]
    images = [
        _write_moved(src, affine, m, tmp_path / f'in{i}.nii.gz') for i, m in enumerate(to_src)
    ]

    wf = init_b0_hmc_wf(
        align_to='iterative',
        transform='Rigid',
        num_iters=2,
        settings='unbiased_template',
        name='seed_check',
    )
    wf.base_dir = str(tmp_path / 'work')
    wf.inputs.inputnode.b0_images = images
    results = {n.fullname: n.result.outputs for n in wf.run().nodes()}

    midpoint = [
        nb.load(f).get_fdata()
        for f in results['seed_check.iterative_alignment_000.to_midpoint'].output_image
    ]
    inside = np.min(midpoint, axis=0) > 0.3 * src.max()
    for i in range(3):
        for j in range(i + 1, 3):
            corr = np.corrcoef(midpoint[i][inside], midpoint[j][inside])[0, 1]
            assert corr > 0.98, f'midpoint resamples {i} and {j} correlate at {corr:.3f}'

    # Template -> image maps, RAS
    forward = [
        LPS @ linear_transform_matrix(x[0]) @ LPS
        for x in results['seed_check.iterative_alignment_001.reg_001'].forward_transforms
    ]
    pts = xyz[envelope > 0][::20]

    def _displacement(mat):
        return np.linalg.norm(pts @ mat[:3, :3].T + mat[:3, 3] - pts, axis=1).mean()

    # template -> source through each image: the same map when the images are aligned
    through = [m @ f for m, f in zip(to_src, forward, strict=True)]
    for i in range(1, 3):
        err = _displacement(np.linalg.inv(through[0]) @ through[i])
        assert err < 0.75, f'image {i} is {err:.2f} mm from image 0 in the template'

    mean_log = np.mean([logm(f).real for f in forward], axis=0)
    offset = _displacement(expm(mean_log))
    assert offset < 0.75, f'template is {offset:.2f} mm from the mean position'
