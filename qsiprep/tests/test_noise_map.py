"""The denoisers' noise maps are carried to the ACPC grid and scaled like the signal.

The transport tests run ``antsApplyTransforms`` and are skipped when it is absent,
so the file is still collectable outside the container.
"""

import json
import math
import shutil

import nibabel as nb
import numpy as np
import pandas as pd
import pytest
from nipype.interfaces.base import isdefined

from qsiprep import config
from qsiprep.data import load as load_data
from qsiprep.interfaces.dwi_merge import _total_b0_correction, harmonize_b0s
from qsiprep.interfaces.noise_map import (
    ConcatNoiseMaps,
    ResampleNoiseMaps,
    scale_noise_map,
    series_reference_volumes,
)

requires_ants = pytest.mark.skipif(
    shutil.which('antsApplyTransforms') is None, reason='antsApplyTransforms not installed'
)
IDENTITY = str(load_data('itkIdentityTransform.txt'))
AFFINE = np.diag([-2.0, -2.0, 2.0, 1.0])
SHAPE = (12, 12, 12)
# Away from the edges, where linear interpolation of the zero background reaches in
INTERIOR = (slice(2, -2),) * 3


def _write_confounds(path, series, bvals, corrections=None, sep='\t'):
    frame = pd.DataFrame({'original_file': series, 'bval': bvals})
    if corrections is not None:
        frame['series_b0_correction'] = corrections
    frame.to_csv(path, sep=sep, index=False)
    return str(path)


def _write_image(path, data, affine=AFFINE):
    nb.Nifti1Image(np.asarray(data, dtype=np.float32), affine).to_filename(str(path))
    return str(path)


def _ramp(offset=1.0):
    """Build a smooth map, linear along each axis so linear interpolation is exact."""
    ii, jj, kk = np.meshgrid(*(np.arange(n) for n in SHAPE), indexing='ij')
    return offset + 0.1 * ii + 0.2 * jj + 0.05 * kk


def test_reference_volume_is_each_series_first_b0(tmp_path):
    """Test that each series follows its own first b=0 and keeps its harmonization."""
    confounds = _write_confounds(
        tmp_path / 'confounds.tsv',
        ['a_dwi.nii.gz'] * 3 + ['b_dwi.nii.gz'] * 3,
        [1000, 0, 1000, 1000, 1000, 5],
        [1.1, 1.1, 1.1, np.nan, np.nan, np.nan],
    )
    index, references, scales = series_reference_volumes(
        confounds, ['/bids/sub-01/dwi/a_dwi.nii.gz', '/bids/sub-01/dwi/b_dwi.nii.gz'], 100
    )
    assert index == [0, 0, 0, 1, 1, 1]
    assert references == [1, 5]
    # A series left unscaled (no b=0 at that merge) records NaN; it was scaled by 1
    assert scales == [1.1, 1.0]


def test_series_without_b0_follows_its_first_volume(tmp_path):
    """Test that a series with no b=0 falls back to its first volume, and no column to 1."""
    confounds = _write_confounds(tmp_path / 'c.tsv', ['a_dwi.nii.gz'] * 2, [1000, 2000])
    _, references, scales = series_reference_volumes(confounds, ['a_dwi.nii.gz'], 100)
    assert references == [0]
    assert scales == [1.0]


def test_confounds_naming_an_unknown_series_raise(tmp_path):
    """Test that a volume from a series with no noise map is an error, not a silent skip."""
    confounds = _write_confounds(tmp_path / 'c.tsv', ['a_dwi.nii.gz', 'c_dwi.nii.gz'], [0, 0])
    with pytest.raises(ValueError, match='c_dwi.nii.gz'):
        series_reference_volumes(confounds, ['a_dwi.nii.gz'], 100)


def test_noise_scales_linearly_with_the_signal():
    """Test that sigma takes the signal's factors, weights and bias division unsquared."""
    noise = np.full((2, 2), 4.0)
    weight = np.array([[0.5, 1.0], [2.0, 1.0]])
    bias = np.array([[2.0, 0.0], [4.0, -1.0]])
    np.testing.assert_allclose(
        scale_noise_map(noise, factor=1.5, weight=weight, bias=bias),
        [[1.5, 0.0], [3.0, 0.0]],
    )


def test_b0_corrections_compound_across_merges():
    """Test that a second harmonization multiplies the first; NaN means unscaled."""
    np.testing.assert_allclose(
        _total_b0_correction([1.2, 1.2, np.nan], [0.9, 1.0, 1.1]), [1.08, 1.2, 1.1]
    )
    with pytest.raises(ValueError, match='volumes'):
        _total_b0_correction([1.0, 1.0], [1.0])


def test_harmonization_skips_only_the_series_without_b0(tmp_path):
    """Test that a b=0-less last series does not stop the others being harmonized."""
    images, bvals = [], []
    for num, (value, series_bvals) in enumerate([(100.0, [0, 1000]), (200.0, [1000, 1000])]):
        images.append(_write_image(tmp_path / f's{num}.nii.gz', np.full((2, 2, 2, 2), value)))
        np.savetxt(tmp_path / f's{num}.bval', [series_bvals], fmt='%d')
        bvals.append(str(tmp_path / f's{num}.bval'))
    harmonized, _, corrections = harmonize_b0s(images, bvals, 100, True)
    assert corrections[0] == pytest.approx(1.0)
    assert np.isnan(corrections[1])
    # The image was left alone, as its NaN correction says
    assert harmonized[1].get_fdata().max() == pytest.approx(200.0)


def _resample_inputs(tmp_path, **overrides):
    """Two series of three volumes each, on one grid, all moved by identity transforms."""
    native = [
        _write_image(tmp_path / 'noise_a.nii.gz', _ramp(1.0)),
        _write_image(tmp_path / 'noise_b.nii.gz', _ramp(3.0)),
    ]
    grid = _write_image(tmp_path / 'grid.nii.gz', np.zeros(SHAPE))
    confounds = _write_confounds(
        tmp_path / 'confounds.tsv',
        ['sub-01_dir-AP_dwi.nii.gz'] * 3 + ['sub-01_dir-PA_dwi.nii.gz'] * 3,
        [0, 1000, 1000, 1000, 0, 1000],
        [1.1] * 3 + [0.9] * 3,
    )
    inputs = {
        'noise_images': native,
        'series_files': ['/bids/sub-01/dwi/sub-01_dir-AP_dwi.nii.gz',
                         '/bids/sub-01/dwi/sub-01_dir-PA_dwi.nii.gz'],
        'series_sources': ['sub-01/dwi/sub-01_dir-AP_dwi.nii.gz',
                           'sub-01/dwi/sub-01_dir-PA_dwi.nii.gz'],
        'channel_scales': [1.0, 1 / math.sqrt(2)],
        'pe_directions': ['j-', 'j'],
        'readout_times': [0.05, 0.05],
        'denoise_method': 'dwidenoise',
        'confounds_file': confounds,
        'volume_transforms': [IDENTITY] * 6,
        'reference_image': grid,
    }  # fmt:skip
    inputs.update(overrides)
    return inputs


@requires_ants
def test_noise_maps_take_the_factors_of_their_reference_volume(tmp_path):
    """Test each series' map: harmonization, channel scaling, its b=0's weight, the bias."""
    weights = [
        _write_image(tmp_path / 'w0.nii.gz', np.full(SHAPE, 0.5)),
        _write_image(tmp_path / 'w1.nii.gz', np.full(SHAPE, 2.0)),
    ]
    bias = _write_image(tmp_path / 'bias.nii.gz', np.full(SHAPE, 2.0))
    result = ResampleNoiseMaps(
        **_resample_inputs(
            tmp_path,
            # Volume 0 (series a's b=0) uses weight 0, volume 4 (series b's) weight 1
            weight_images=weights,
            weight_index=[0, 1, 1, 0, 1, 0],
            weight_method='Jacobian',
            bias_field=bias,
            native_noise_maps=[
                '/out/sub-01/dwi/sub-01_dir-AP_stat-noise_dwimap.nii.gz',
                '/out/sub-01/dwi/sub-01_dir-PA_stat-noise_dwimap.nii.gz',
            ],
            output_dir='/out',
            approximations=['a note'],
        )  # fmt:skip
    ).run(cwd=str(tmp_path))

    data = nb.load(result.outputs.out_file).get_fdata()
    assert data.shape == SHAPE + (2,)
    np.testing.assert_allclose(
        data[INTERIOR + (0,)], (_ramp(1.0) * 1.1 * 0.5 / 2.0)[INTERIOR], rtol=1e-5
    )
    np.testing.assert_allclose(
        data[INTERIOR + (1,)],
        (_ramp(3.0) * 0.9 / math.sqrt(2) * 2.0 / 2.0)[INTERIOR],
        rtol=1e-5,
    )

    meta = result.outputs.meta_dict
    assert meta['Quantity'] == 'sigma'
    assert meta['NoiseMapIndex'] == [0, 0, 0, 1, 1, 1]
    assert meta['AppliedScaling'] == [
        'b0 harmonization', 'complex channel scaling', 'Jacobian', 'bias field'
    ]  # fmt:skip
    assert meta['SignalRedistributionMethod'] == 'Jacobian'
    assert meta['Approximations'] == ['a note']
    series = meta['NoiseMapSeries']
    assert [entry['ReferenceVolume'] for entry in series] == [0, 4]
    assert [entry['HarmonizationScale'] for entry in series] == [1.1, 0.9]
    assert series[1]['ChannelScaling'] == pytest.approx(1 / math.sqrt(2))
    assert series[0]['NativeNoiseMap'] == 'bids::sub-01/dwi/sub-01_dir-AP_stat-noise_dwimap.nii.gz'
    assert series[1]['Source'] == 'sub-01/dwi/sub-01_dir-PA_dwi.nii.gz'
    assert series[1]['PhaseEncodingDirection'] == 'j'
    json.dumps(meta)  # the sink writes it as JSON


@requires_ants
def test_a_single_series_gives_a_3d_map(tmp_path):
    """Test that one series gives a 3D map, and patch2self is labeled a residual."""
    inputs = _resample_inputs(tmp_path, denoise_method='patch2self')
    for key in (
        'noise_images',
        'series_files',
        'series_sources',
        'channel_scales',
        'pe_directions',
        'readout_times',
    ):
        inputs[key] = inputs[key][:1]  # fmt:skip
    inputs['confounds_file'] = _write_confounds(
        tmp_path / 'one.tsv', ['sub-01_dir-AP_dwi.nii.gz'] * 2, [0, 1000]
    )
    inputs['volume_transforms'] = [IDENTITY] * 2
    inputs['resolution_label'] = 'nativemin'
    result = ResampleNoiseMaps(**inputs).run(cwd=str(tmp_path))

    assert nb.load(result.outputs.out_file).shape == SHAPE
    meta = result.outputs.meta_dict
    assert meta['Quantity'] == 'residual_rms'
    assert 'not a noise standard deviation' in meta['Description']
    assert meta['AppliedScaling'] == []
    # The voxel size a res-native* grid turned out to be
    assert meta['Resolution'] == {'nativemin': '2 mm isotropic'}


@requires_ants
@pytest.mark.parametrize('modulated', [True, False])
def test_the_field_eddy_applied_is_undone_with_its_jacobian(tmp_path, modulated):
    """Test that a field eddy applied itself modulates the map as eddy modulated the signal.

    A field rising linearly along j shifts voxels by an amount that grows by
    ``slope * readout`` voxels per voxel, so its Jacobian is ``1 + slope * readout``
    (opposite sign for ``j-``). A uniform noise map stays uniform under the shift.
    """
    slope, readout = 0.4, 0.05
    jj = np.broadcast_to(np.arange(SHAPE[1])[None, :, None], SHAPE)
    fieldmap = _write_image(tmp_path / 'hz.nii.gz', slope * jj)
    inputs = _resample_inputs(
        tmp_path,
        fieldmap_hz=fieldmap,
        fieldmap_modulated=modulated,
        readout_times=[readout, readout],
    )
    inputs['noise_images'] = [
        _write_image(tmp_path / 'flat_a.nii.gz', np.full(SHAPE, 3.0)),
        _write_image(tmp_path / 'flat_b.nii.gz', np.full(SHAPE, 3.0)),
    ]
    inputs['confounds_file'] = _write_confounds(
        tmp_path / 'unscaled.tsv',
        ['sub-01_dir-AP_dwi.nii.gz'] * 3 + ['sub-01_dir-PA_dwi.nii.gz'] * 3,
        [0, 1000, 1000, 1000, 0, 1000],
    )
    inputs['channel_scales'] = [1.0, 1.0]
    result = ResampleNoiseMaps(**inputs).run(cwd=str(tmp_path))

    data = nb.load(result.outputs.out_file).get_fdata()
    # Keep clear of the edges the shift pulls the background in from
    interior = (slice(3, -3),) * 3
    jacobian = {'j-': 1 - slope * readout, 'j': 1 + slope * readout}
    for series_num, pe_dir in enumerate(['j-', 'j']):
        expected = 3.0 * (jacobian[pe_dir] if modulated else 1.0)
        np.testing.assert_allclose(data[interior + (series_num,)], expected, rtol=1e-4)
    assert ('eddy Jacobian' in result.outputs.meta_dict['AppliedScaling']) == modulated


def test_concatenated_units_take_the_merge_harmonization(tmp_path):
    """Test that concatenating units rescales each by its merge factor and offsets indices."""
    unit_a = _write_image(
        tmp_path / 'a.nii.gz', np.stack([np.ones(SHAPE), 2 * np.ones(SHAPE)], -1)
    )
    unit_b = _write_image(tmp_path / 'b.nii.gz', 3 * np.ones(SHAPE))

    def _entry(name, reference, scale):
        return {'Source': name, 'PhaseEncodingDirection': 'j', 'ReferenceVolume': reference,
                'HarmonizationScale': scale, 'ChannelScaling': 1.0}  # fmt:skip

    common = {'Description': 'd', 'DenoisingMethod': 'dwidenoise', 'Quantity': 'sigma',
              'Interpolation': 'Linear'}  # fmt:skip
    meta_a = {**common, 'AppliedScaling': ['b0 harmonization'], 'NoiseMapIndex': [0, 0, 1],
              'NoiseMapSeries': [_entry('a1', 0, 1.0), _entry('a2', 2, 1.25)],
              'Approximations': ['eddy note']}  # fmt:skip
    meta_b = {**common, 'AppliedScaling': ['bias field'], 'NoiseMapIndex': [0, 0],
              'NoiseMapSeries': [_entry('b1', 1, 1.0)]}  # fmt:skip
    # The merged confounds hold the total: unit a was scaled by 2 at the merge, b by 0.5
    confounds = _write_confounds(
        tmp_path / 'merged_confounds.csv',
        ['a1', 'a1', 'a2', 'b1', 'b1'],
        [0, 1000, 0, 1000, 0],
        [2.0, 2.0, 2.5, 0.5, 0.5],
        sep=',',
    )
    result = ConcatNoiseMaps(
        noise_maps=[unit_a, unit_b], meta_dicts=[meta_a, meta_b], confounds_file=confounds
    ).run(cwd=str(tmp_path))

    data = nb.load(result.outputs.out_file).get_fdata()
    assert data.shape == SHAPE + (3,)
    np.testing.assert_allclose(data[0, 0, 0], [2.0, 4.0, 1.5])
    meta = result.outputs.meta_dict
    assert meta['NoiseMapIndex'] == [0, 0, 1, 2, 2]
    assert [entry['ReferenceVolume'] for entry in meta['NoiseMapSeries']] == [0, 2, 4]
    assert [entry['HarmonizationScale'] for entry in meta['NoiseMapSeries']] == [2.0, 2.5, 0.5]
    assert meta['AppliedScaling'] == ['b0 harmonization', 'bias field']
    assert meta['Approximations'] == ['eddy note']


# ---------------------------------------------------------------------------
# Workflow construction


def _finalize_wf(
    tmp_path,
    monkeypatch,
    hmc_method,
    sdc_method,
    denoise_method,
    output_spaces=('acpc:res-1p2mm',),
    write_derivatives=True,
):
    from qsiprep.tests.gradient_fixtures import write_dwi_with_gradients
    from qsiprep.tests.test_sdc_warp_derivative import _reverse_pe_unit
    from qsiprep.utils.spaces import parse_output_spaces
    from qsiprep.workflows.dwi.finalize import init_dwi_finalize_wf

    for section, key, value in [
        (config.execution, 'output_dir', str(tmp_path / 'out')),
        (config.execution, 'bids_dir', str(tmp_path)),
        (config.execution, 'sloppy', False),
        (config.workflow, 'hmc_method', hmc_method),
        (config.workflow, 'sdc_method', sdc_method),
        (config.workflow, 'denoise_method', denoise_method),
        (config.workflow, 'output_spaces', list(output_spaces)),
        (config.workflow, 'dwiref_definition', 'distortion-group'),
        (config.workflow, 'b0_threshold', 100),
        (config.nipype, 'omp_nthreads', 1),
    ]:
        monkeypatch.setattr(section, key, value)
    unit, source = _reverse_pe_unit(tmp_path, dwi_writer=write_dwi_with_gradients)
    return init_dwi_finalize_wf(
        unit=unit,
        name='dwi_finalize_wf',
        source_file=source,
        output_prefix='sub-01',
        acpc_specs=parse_output_spaces(config.workflow.output_spaces),
        do_biascorr=True,
        write_derivatives=write_derivatives,
    )


def _edges(workflow):
    return {(u.name, v.name): set(d['connect']) for u, v, d in workflow._graph.edges(data=True)}


@pytest.mark.parametrize(
    ('hmc_method', 'sdc_method', 'from_eddy_field'),
    [('eddy', 'topup', True), ('eddy', 'drbuddi', False), ('shoreline', 'drbuddi', False)],
)
def test_finalize_carries_the_noise_maps_to_acpc(
    tmp_path, monkeypatch, hmc_method, sdc_method, from_eddy_field
):
    """Test the noise map node's wiring, and that only a field eddy applied is undone."""
    wf = _finalize_wf(tmp_path, monkeypatch, hmc_method, sdc_method, 'dwidenoise')
    node = wf.get_node('resample_noise')
    edges = _edges(wf)

    assert edges[('dwi_trans_wf', 'resample_noise')] >= {
        ('outputnode.volume_transforms', 'volume_transforms'),
        ('outputnode.jacobian_weights', 'weight_images'),
        ('outputnode.jacobian_weight_index', 'weight_index'),
    }
    assert edges[('final_denoise_wf', 'resample_noise')] == {
        ('outputnode.bias_field', 'bias_field')
    }
    # Only a field eddy applied itself is undone here
    from_inputnode = edges[('inputnode', 'resample_noise')]
    assert (('fieldmap_hz', 'fieldmap_hz') in from_inputnode) == from_eddy_field
    assert node.inputs.fieldmap_modulated == from_eddy_field
    assert node.inputs.pe_directions == ['j-', 'j']
    assert node.inputs.series_sources == ['sub-01_dir-AP_dwi.nii.gz', 'sub-01_dir-PA_dwi.nii.gz']
    assert bool(node.inputs.approximations) == (hmc_method == 'eddy')

    sink = wf.get_node('dwi_derivatives_wf').get_node('ds_noise_map_t1')
    assert sink.inputs.space == 'ACPC'
    assert sink.inputs.statistic == 'noise'
    assert sink.inputs.res == '1p2mm'
    # Only res-native* or several resolutions report the grid
    assert not isdefined(node.inputs.resolution_label)


def test_each_acpc_resolution_gets_its_own_noise_map(tmp_path, monkeypatch):
    """Test that every resolution's map follows its own transforms, weights and bias field."""
    wf = _finalize_wf(
        tmp_path, monkeypatch, 'shoreline', 'drbuddi', 'dwidenoise',
        output_spaces=('acpc:res-2mm', 'acpc:res-1p5mm'),
    )  # fmt:skip
    edges = _edges(wf)
    for slot, label in enumerate(['2mm', '1p5mm'], start=1):
        node = wf.get_node(f'resample_noise_res{label}')
        assert node.inputs.resolution_label == label
        assert ('outputnode.volume_transforms', 'volume_transforms') in edges[
            (f'dwi_trans_wf_res{label}', f'resample_noise_res{label}')
        ]
        assert edges[(f'final_denoise_wf_res{label}', f'resample_noise_res{label}')] == {
            ('outputnode.bias_field', 'bias_field')
        }
        assert edges[(f'resample_noise_res{label}', f'dwi_derivatives_wf_res{label}')] == {
            ('out_file', 'inputnode.noise_map_t1'),
            ('meta_dict', 'inputnode.noise_map_meta'),
        }
        sink = wf.get_node(f'dwi_derivatives_wf_res{label}').get_node('ds_noise_map_t1')
        assert sink.inputs.res == label
        # The merge path reads one list slot per resolution, in acpc_specs order
        assert edges[(f'resample_noise_res{label}', 'merge_out_noise_map_t1')] == {
            ('out_file', f'in{slot}')
        }


def test_finalize_skips_the_noise_maps_without_a_denoiser(tmp_path, monkeypatch):
    """Test that no noise map node or sink is built when denoising is off."""
    wf = _finalize_wf(tmp_path, monkeypatch, 'eddy', 'topup', 'none')

    assert wf.get_node('resample_noise') is None
    assert wf.get_node('dwi_derivatives_wf').get_node('ds_noise_map_t1') is None


@pytest.mark.parametrize('strategy', ['concat', 'average'])
def test_merge_concatenates_noise_maps_only_when_concatenating(tmp_path, monkeypatch, strategy):
    """Test that concatenated units join their noise maps and averaged pairs write none."""
    from qsiprep.workflows.dwi.distortion_group_merge import init_distortion_group_merge_wf

    monkeypatch.setattr(config.execution, 'output_dir', str(tmp_path / 'out'))
    monkeypatch.setattr(config.workflow, 'denoise_method', 'dwidenoise')
    monkeypatch.setattr(config.nipype, 'omp_nthreads', 1)
    from qsiprep.utils.spaces import parse_output_spaces

    wf = init_distortion_group_merge_wf(
        merging_strategy=strategy,
        inputs_list=['dir-AP', 'dir-PA'],
        source_file='sub-01_dwi.nii.gz',
        output_prefix='sub-01',
        name='merge_wf',
        resolution=parse_output_spaces(['acpc:res-2mm'])[0].resolution,
    )
    concat = wf.get_node('concat_noise_maps')
    if strategy == 'average':
        assert concat is None
        return
    edges = _edges(wf)
    assert edges[('distortion_merger', 'concat_noise_maps')] == {
        ('merged_denoising_confounds', 'confounds_file')
    }
    assert edges[('inputnode', 'merge_noise_maps')] == {
        ('dir_AP_noise_map', 'in1'),
        ('dir_PA_noise_map', 'in2'),
    }
    assert edges[('concat_noise_maps', 'dwi_derivatives_wf')] == {
        ('out_file', 'inputnode.noise_map_t1'),
        ('meta_dict', 'inputnode.noise_map_meta'),
    }
    assert wf.get_node('dwi_derivatives_wf').get_node('ds_noise_map_t1').inputs.res == '2mm'
