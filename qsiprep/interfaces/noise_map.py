"""Carry the denoisers' noise, rank and rank density maps onto the output grid.

Noise maps are scaled like the signal. Rank and rank density maps are not intensities, so they
are only resampled.
"""

import os

import nibabel as nb
import numpy as np
import pandas as pd
from nipype import logging
from nipype.interfaces import ants
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    InputMultiObject,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)

LOGGER = logging.getLogger('nipype.interface')

#: What each denoiser's map holds
NOISE_QUANTITIES = {
    'dwidenoise': 'sigma',
    'dwidenoise2': 'sigma',
    'svht': 'sigma',
    'patch2self': 'residual_rms',
}

#: The stat- entity of each quantity's map
NOISE_STATISTICS = {
    'sigma': 'std',
    'residual_rms': 'rms',
}

#: The sidecar Units of each kind of map. A noise map is in the intensity units of its
#: series, a rank map counts signal components, and a rank density is signal rank per mm
#: of kernel radius.
MAP_UNITS = {
    'noise': 'arbitrary',
    'rank': 'count',
    'rankdensity': '1/mm',
}

#: The prefix of the sidecar keys that index each kind of map's volumes and series
_MAP_KEYS = {
    'noise': 'NoiseMap',
    'rank': 'RankMap',
    'rankdensity': 'RankDensityMap',
}

#: What each kind of map holds, for the ACPC sidecar
_MAP_CONTENTS = {
    'rank': 'Signal rank (the number of principal components kept as signal)',
    'rankdensity': (
        'Signal-rank density (signal rank per mm of kernel radius) that sized the '
        'reconstruction kernel'
    ),
}


def noise_map_statistic(denoise_method):
    """Return the stat- entity of the noise map ``denoise_method`` writes."""
    return NOISE_STATISTICS[NOISE_QUANTITIES[denoise_method]]


def read_confounds(path):
    """Read a confounds table: tab-separated for ``.tsv``, comma-separated otherwise."""
    return pd.read_csv(path, sep='\t' if str(path).endswith('.tsv') else ',')


def series_reference_volumes(confounds_file, series_files, b0_threshold):
    """Find each volume's series and each series' reference volume.

    Parameters
    ----------
    confounds_file : str
        The output's confounds TSV, one row per output volume, with the
        ``original_file`` and ``bval`` columns and, when the series were
        harmonized, ``series_b0_correction``.
    series_files : list of str
        The member series, in the order of the noise maps.
    b0_threshold : float
        Largest b-value counted as b=0.

    Returns
    -------
    noise_map_index : list of int
        For each output volume, the index of its series in ``series_files``.
    reference_volumes : list of int
        For each series, its first b=0 volume in the output, or its first
        volume when it has no b=0.
    harmonization_scales : list of float
        For each series, the b=0 harmonization scale its volumes received.
    """
    confounds = read_confounds(confounds_file)
    names = [os.path.basename(str(path)) for path in series_files]
    originals = confounds['original_file'].astype(str).map(os.path.basename)
    missing = sorted(set(originals) - set(names))
    if missing:
        raise ValueError(f'The confounds name series with no noise map: {", ".join(missing)}')
    noise_map_index = [names.index(name) for name in originals]

    bvals = confounds['bval'].to_numpy(dtype=float)
    if 'series_b0_correction' in confounds:
        corrections = np.nan_to_num(
            confounds['series_b0_correction'].to_numpy(dtype=float), nan=1.0
        )
    else:
        corrections = np.ones(len(confounds))

    index = np.asarray(noise_map_index)
    reference_volumes = []
    harmonization_scales = []
    for series_num, name in enumerate(names):
        volumes = np.flatnonzero(index == series_num)
        if volumes.size == 0:
            raise ValueError(f'No output volume comes from {name}')
        b0_volumes = volumes[bvals[volumes] <= b0_threshold]
        reference = int(b0_volumes[0] if b0_volumes.size else volumes[0])
        reference_volumes.append(reference)
        harmonization_scales.append(float(corrections[reference]))
    return noise_map_index, reference_volumes, harmonization_scales


def apply_transforms(image, reference, transforms, out_file, num_threads=1):
    """Resample a scalar map with linear interpolation, zero outside its footprint."""
    xfm = ants.ApplyTransforms(
        input_image=image,
        reference_image=reference,
        transforms=list(transforms),
        output_image=out_file,
        interpolation='Linear',
        default_value=0.0,
        dimension=3,
        float=True,
        num_threads=num_threads,
    )
    xfm.terminal_output = 'allatonce'
    xfm.resource_monitor = False
    LOGGER.info('Resampling noise map: %s', xfm.cmdline)
    xfm.run()
    return out_file


def fill_nonfinite_median(data):
    """Replace non-finite voxels with the median of their finite neighbors.

    dwidenoise2 writes NaN into its noise map for each patch whose PCA failed. ANTs would
    read those as zeros and pull the interpolated values around them down, so each such
    voxel takes the median of the finite values among its 26 neighbors before resampling.
    A voxel with no finite neighbor is filled in a later pass, from the values filled
    before it.

    Parameters
    ----------
    data : :class:`numpy.ndarray`
        3D image data.

    Returns
    -------
    filled : :class:`numpy.ndarray`
        A copy of ``data`` with the non-finite voxels filled. Voxels stay non-finite only if
        the image has no finite voxel at all.
    count : int
        The number of non-finite voxels in ``data``.
    """
    filled = np.array(data, copy=True)
    bad = ~np.isfinite(filled)
    count = int(bad.sum())
    while bad.any() and not bad.all():
        values = {}
        for voxel in map(tuple, np.argwhere(bad)):
            box = tuple(slice(max(i - 1, 0), i + 2) for i in voxel)
            neighbors = filled[box][~bad[box]]
            if neighbors.size:
                values[voxel] = np.median(neighbors)
        for voxel, value in values.items():
            filled[voxel] = value
            bad[voxel] = False
    return filled, count


def scale_noise_map(noise, factor=1.0, weight=None, bias=None):
    """Apply the signal's intensity factors to a noise standard deviation map.

    The signal is multiplied by ``factor`` and ``weight`` and divided by
    ``bias``, so its noise standard deviation is too. Voxels where the bias
    field is not positive are set to zero, as no corrected signal exists there.
    """
    scaled = np.asarray(noise, dtype=np.float64) * factor
    if weight is not None:
        scaled = scaled * weight
    if bias is not None:
        bias = np.asarray(bias, dtype=np.float64)
        valid = bias > 0
        scaled = np.divide(scaled, bias, out=np.zeros_like(scaled), where=valid)
    return scaled.astype(np.float32)


class _ResampleNoiseMapsInputSpec(BaseInterfaceInputSpec):
    kind = traits.Enum(
        'noise',
        'rank',
        'rankdensity',
        usedefault=True,
        desc='which map noise_images holds. Rank and rank density maps take none of the '
        "signal's intensity factors.",
    )
    noise_images = InputMultiObject(
        File(exists=True),
        mandatory=True,
        desc="each member series' native map, in the order of series_files",
    )
    series_meta = traits.List(
        traits.Dict(),
        desc="each member series' native sidecar, which names the denoiser's software "
        'and parameters',
    )
    series_files = traits.List(
        traits.Str(), mandatory=True, desc='the member series (raw BIDS paths)'
    )
    series_sources = traits.List(
        traits.Str(), mandatory=True, desc='BIDS URI of each member series'
    )
    native_noise_maps = traits.List(
        traits.Str(), desc='the written native map derivative of each series'
    )
    output_dir = traits.Str(desc='root of the derivatives dataset, to name native_noise_maps')
    channel_scales = traits.List(
        traits.Float(),
        mandatory=True,
        desc='factor that turns each map into a per-channel sigma (1/sqrt(2) for a '
        'series dwidenoise or dwidenoise2 denoised as complex data)',
    )
    pe_directions = traits.List(
        traits.Str(), mandatory=True, desc='PhaseEncodingDirection of each series'
    )
    readout_times = traits.List(
        traits.Either(None, traits.Float()),
        desc='TotalReadoutTime of each series; needed with fieldmap_hz',
    )
    denoise_method = traits.Enum(*NOISE_QUANTITIES, mandatory=True)
    confounds_file = File(
        exists=True,
        mandatory=True,
        desc="the output's confounds: each volume's series, b-value and b=0 harmonization",
    )
    b0_threshold = traits.Float(100, usedefault=True)
    volume_transforms = InputMultiObject(
        File(exists=True),
        mandatory=True,
        desc='the composed transform of each output volume to the output grid',
    )
    reference_image = File(exists=True, mandatory=True, desc='the output grid')
    weight_images = InputMultiObject(
        File(exists=True), desc='the intensity weight maps, on the output grid'
    )
    weight_index = traits.List(traits.Int(), desc='for each volume, its weight map')
    weight_method = traits.Str(desc="'Jacobian' or 'LSR'")
    fieldmap_hz = File(
        exists=True,
        desc='the off-resonance field eddy applied in its own resampling (TOPUP, or a '
        'GRE fieldmap), in the corrected frame',
    )
    fieldmap_modulated = traits.Bool(
        False, usedefault=True, desc="whether eddy Jacobian-modulated fieldmap_hz's correction"
    )
    bias_field = File(exists=True, desc='the B1 bias field the output was divided by')
    approximations = traits.List(traits.Str(), desc='what the transport leaves out')
    resolution_label = traits.Str(
        desc="the output grid's res- label; when given, the sidecar reports the grid's "
        'voxel size, as for the other outputs of a res-native* or multi-resolution run'
    )
    num_threads = traits.Int(1, usedefault=True, nohash=True)


class _ResampleNoiseMapsOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='the map on the output grid, one volume per series')
    meta_dict = traits.Dict(desc='the sidecar of out_file')


class ResampleNoiseMaps(SimpleInterface):
    """Carry each member series' noise, rank or rank density map to the output grid.

    Each series' map follows the transform of the series' first b=0 volume. A noise
    map is also scaled as that volume's signal was: b=0 harmonization, the channel
    scaling that makes every map a per-channel sigma, the Jacobian or LSR weight,
    eddy's own modulation of the field it applied, and division by the bias field.
    Rank and rank density maps are not intensities, so they are only resampled.
    """

    input_spec = _ResampleNoiseMapsInputSpec
    output_spec = _ResampleNoiseMapsOutputSpec

    def _run_interface(self, runtime):
        inputs = self.inputs
        num_series = len(inputs.series_files)
        counted = ['noise_images', 'series_sources', 'channel_scales', 'pe_directions']
        if isdefined(inputs.series_meta) and inputs.series_meta:
            counted.append('series_meta')

        for name in counted:
            count = len(getattr(inputs, name))
            if count != num_series:
                raise ValueError(f'{name} has {count} entries, not {num_series}')

        noise_map_index, reference_volumes, harmonization_scales = series_reference_volumes(
            inputs.confounds_file, inputs.series_files, inputs.b0_threshold
        )
        if len(inputs.volume_transforms) != len(noise_map_index):
            raise ValueError(
                f'{len(inputs.volume_transforms)} volume transforms for '
                f'{len(noise_map_index)} output volumes'
            )
        scaled = inputs.kind == 'noise'
        weights = None
        if scaled and isdefined(inputs.weight_images) and inputs.weight_images:
            weights = [inputs.weight_images[i] for i in inputs.weight_index]

        bias = None
        if scaled and isdefined(inputs.bias_field):
            bias = np.asanyarray(nb.load(inputs.bias_field).dataobj)

        fieldmap_hz = inputs.fieldmap_hz if isdefined(inputs.fieldmap_hz) else None

        volumes = []
        self._filled = []
        for series_num in range(num_series):
            native = self._fill_native(runtime, series_num, inputs.noise_images[series_num])
            if fieldmap_hz is not None:
                native = self._apply_eddy_field(runtime, series_num, native, fieldmap_hz)
            reference = reference_volumes[series_num]
            resampled = apply_transforms(
                native,
                inputs.reference_image,
                [inputs.volume_transforms[reference]],
                os.path.join(runtime.cwd, f'{inputs.kind}_series{series_num:02d}.nii.gz'),
                num_threads=inputs.num_threads,
            )
            resampled = np.asanyarray(nb.load(resampled).dataobj)
            if not scaled:
                volumes.append(resampled.astype(np.float32))
                continue

            weight = None
            if weights is not None:
                weight = np.asanyarray(nb.load(weights[reference]).dataobj)
            volumes.append(
                scale_noise_map(
                    resampled,
                    factor=harmonization_scales[series_num] * inputs.channel_scales[series_num],
                    weight=weight,
                    bias=bias,
                )
            )

        grid = nb.load(inputs.reference_image)
        data = volumes[0] if num_series == 1 else np.stack(volumes, axis=-1)
        out_img = nb.Nifti1Image(data, grid.affine)
        out_img.header.set_xyzt_units('mm')
        out_file = os.path.join(runtime.cwd, f'{inputs.kind}_map.nii.gz')
        out_img.to_filename(out_file)
        self._results['out_file'] = out_file
        self._results['meta_dict'] = self._sidecar(
            noise_map_index, reference_volumes, harmonization_scales, weights, bias
        )
        return runtime

    def _fill_native(self, runtime, series_num, native):
        """Fill the native map's non-finite voxels on its own grid, before any resampling."""
        img = nb.load(native)
        filled, count = fill_nonfinite_median(np.asanyarray(img.dataobj))
        self._filled.append(count)
        if not count:
            return native
        if not np.isfinite(filled).all():
            raise ValueError(f'{native} has no finite voxel to fill its {count} others from')
        LOGGER.warning(
            '%d voxels of %s are not finite; they take the median of their neighbors.',
            count,
            native,
        )
        header = img.header.copy()
        header.set_data_dtype(np.float32)
        header.set_slope_inter(1, 0)
        out_file = os.path.join(
            runtime.cwd, f'{self.inputs.kind}_series{series_num:02d}_filled.nii.gz'
        )
        nb.Nifti1Image(filled.astype(np.float32), img.affine, header).to_filename(out_file)
        return out_file

    def _apply_eddy_field(self, runtime, series_num, native, fieldmap_hz):
        """Undo the susceptibility distortion eddy corrected internally, as eddy did."""
        from ..utils.jacobian import jacobian_determinant
        from ..workflows.dwi.resampling import _hz_to_warp

        pe_dir = self.inputs.pe_directions[series_num]
        readout_times = self.inputs.readout_times if isdefined(self.inputs.readout_times) else []
        readout_time = readout_times[series_num] if series_num < len(readout_times) else None
        if readout_time is None:
            raise ValueError(
                f'{self.inputs.series_files[series_num]} has no TotalReadoutTime, which is '
                'needed to rebuild the distortion eddy corrected'
            )
        series_dir = os.path.join(runtime.cwd, f'series{series_num:02d}')
        os.makedirs(series_dir, exist_ok=True)
        warp = _hz_to_warp(fieldmap_hz, readout_time, pe_dir, newpath=series_dir)
        corrected = apply_transforms(
            native,
            fieldmap_hz,
            [warp],
            os.path.join(series_dir, f'{self.inputs.kind}_corrected_frame.nii.gz'),
            num_threads=self.inputs.num_threads,
        )
        if self.inputs.kind != 'noise' or not self.inputs.fieldmap_modulated:
            return corrected

        jacobian = jacobian_determinant(
            warp,
            os.path.join(series_dir, 'eddy_field_jacobian.nii.gz'),
            pe_axis='ijk'.index(pe_dir[0]),
        )
        img = nb.load(corrected)
        modulated = scale_noise_map(
            np.asanyarray(img.dataobj), weight=np.asanyarray(nb.load(jacobian).dataobj)
        )
        out_file = os.path.join(series_dir, 'noise_corrected_frame_modulated.nii.gz')
        nb.Nifti1Image(modulated, img.affine).to_filename(out_file)
        return out_file

    def _sidecar(self, noise_map_index, reference_volumes, harmonization_scales, weights, bias):
        inputs = self.inputs
        key = _MAP_KEYS[inputs.kind]
        quantity = NOISE_QUANTITIES[inputs.denoise_method]
        if inputs.kind != 'noise':
            description = (
                f'{_MAP_CONTENTS[inputs.kind]} estimated by {inputs.denoise_method} for each '
                "member series, carried to the output grid with the transform of that series' "
                'reference b=0 volume. It is not an intensity, so it is not scaled, and '
                'linear interpolation makes it fractional between voxels. Volume k of the '
                f'map describes the output volumes whose {key}Index is k.'
            )
        else:
            if quantity == 'sigma':
                what = (
                    'Noise standard deviation per channel of each member series, estimated by '
                    f'{inputs.denoise_method} on the raw series'
                )
            else:
                what = (
                    'Root mean square over volumes of the residual Patch2Self removed from '
                    'each member series; it mixes noise with fit error and is not a noise '
                    'standard deviation'
                )
            description = (
                f"{what}, carried to the output grid with the transform of that series' "
                'reference b=0 volume and scaled as its signal was, in the intensity units '
                'of the preprocessed series. Volume k of the noise map describes the output '
                f'volumes whose {key}Index is k. It describes the noise in the source data: '
                'interpolation lowers the noise of the preprocessed series and correlates it '
                'spatially, so its per-voxel noise is somewhat smaller.'
            )
        description += (
            ' Non-finite voxels of a native map (dwidenoise2 marks each patch whose PCA '
            'failed with NaN) took the median of their finite neighbors before resampling; '
            'FilledNativeVoxels counts them for each series.'
        )

        applied = []
        if inputs.kind == 'noise':
            if any(abs(scale - 1.0) > 1e-6 for scale in harmonization_scales):
                applied.append('b0 harmonization')

            if any(abs(scale - 1.0) > 1e-6 for scale in inputs.channel_scales):
                applied.append('complex channel scaling')

            if isdefined(inputs.fieldmap_hz) and inputs.fieldmap_modulated:
                applied.append('eddy Jacobian')

            if weights is not None:
                method = inputs.weight_method if isdefined(inputs.weight_method) else 'Jacobian'
                applied.append(method)

            if bias is not None:
                applied.append('bias field')

        series_meta = inputs.series_meta if isdefined(inputs.series_meta) else []
        series = []
        for series_num in range(len(inputs.series_files)):
            entry = {
                'Source': inputs.series_sources[series_num],
                'PhaseEncodingDirection': inputs.pe_directions[series_num],
                'ReferenceVolume': reference_volumes[series_num],
            }
            if inputs.kind == 'noise':
                entry['HarmonizationScale'] = harmonization_scales[series_num]
                entry['ChannelScaling'] = inputs.channel_scales[series_num]

            if isdefined(inputs.native_noise_maps) and inputs.native_noise_maps:
                native = inputs.native_noise_maps[series_num]
                if isdefined(inputs.output_dir):
                    native = f'bids::{os.path.relpath(native, inputs.output_dir)}'
                entry[f'Native{key}'] = native

            entry['FilledNativeVoxels'] = self._filled[series_num]
            if series_meta:
                # Parameters can differ between series, e.g. dwidenoise's automatic window
                for field in ('UsedPhase', 'DenoisingParameters'):
                    if field in series_meta[series_num]:
                        entry[field] = series_meta[series_num][field]
            series.append(entry)

        sidecar = {
            'Description': description,
            'Units': MAP_UNITS[inputs.kind],
            'DenoisingMethod': inputs.denoise_method,
        }
        if series_meta and 'Software' in series_meta[0]:
            sidecar['Software'] = series_meta[0]['Software']

        if inputs.kind == 'noise':
            sidecar['Quantity'] = quantity
            sidecar['AppliedScaling'] = applied
        sidecar.update(
            {
                'Interpolation': 'Linear',
                f'{key}Index': list(noise_map_index),
                f'{key}Series': series,
            }
        )
        if weights is not None and isdefined(inputs.weight_method):
            sidecar['SignalRedistributionMethod'] = inputs.weight_method
        if isdefined(inputs.approximations) and inputs.approximations:
            sidecar['Approximations'] = list(inputs.approximations)
        if isdefined(inputs.resolution_label):
            from ..workflows.dwi.finalize import _grid_metadata

            sidecar.update(_grid_metadata(inputs.reference_image, inputs.resolution_label))
        return sidecar


class _ConcatNoiseMapsInputSpec(BaseInterfaceInputSpec):
    kind = traits.Enum(
        'noise',
        'rank',
        'rankdensity',
        usedefault=True,
        desc='which map noise_maps holds. Only noise maps take the harmonization.',
    )
    noise_maps = InputMultiObject(File(exists=True), mandatory=True, desc="each unit's map")
    meta_dicts = traits.List(traits.Dict(), mandatory=True, desc='their sidecars')
    confounds_file = File(
        exists=True,
        mandatory=True,
        desc="the merged output's confounds, whose series_b0_correction is the total "
        'b=0 harmonization of each volume',
    )


class _ConcatNoiseMapsOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='the maps of all units, one volume per series')
    meta_dict = traits.Dict(desc='the sidecar of out_file')


class ConcatNoiseMaps(SimpleInterface):
    """Join the denoiser maps of correction units concatenated into one output.

    The merge harmonizes the b=0 intensity of the units once more, so each unit's
    noise maps are scaled by the factor its volumes received there. The volume and
    series indices are offset to the merged output.
    """

    input_spec = _ConcatNoiseMapsInputSpec
    output_spec = _ConcatNoiseMapsOutputSpec

    def _run_interface(self, runtime):
        kind = self.inputs.kind
        key = _MAP_KEYS[kind]
        if len(self.inputs.noise_maps) != len(self.inputs.meta_dicts):
            raise ValueError('Each map needs its sidecar')

        confounds = read_confounds(self.inputs.confounds_file)
        if 'series_b0_correction' in confounds:
            totals = np.nan_to_num(
                confounds['series_b0_correction'].to_numpy(dtype=float), nan=1.0
            )
        else:
            totals = np.ones(len(confounds))

        volumes = []
        index = []
        series = []
        affine = None
        volume_offset = 0
        for noise_map, meta in zip(self.inputs.noise_maps, self.inputs.meta_dicts, strict=True):
            img = nb.load(noise_map)
            affine = img.affine if affine is None else affine
            data = np.asanyarray(img.dataobj)
            if data.ndim == 3:
                data = data[..., np.newaxis]

            for series_num, entry in enumerate(meta[f'{key}Series']):
                reference = entry['ReferenceVolume'] + volume_offset
                entry = {**entry, 'ReferenceVolume': int(reference)}
                if kind == 'noise':
                    merge_scale = totals[reference] / entry['HarmonizationScale']
                    volumes.append(data[..., series_num] * merge_scale)
                    entry['HarmonizationScale'] = float(totals[reference])
                else:
                    volumes.append(data[..., series_num])
                series.append(entry)
            index += [len(series) - data.shape[-1] + i for i in meta[f'{key}Index']]
            volume_offset += len(meta[f'{key}Index'])

        if volume_offset != len(confounds):
            raise ValueError(
                f'The maps describe {volume_offset} volumes but the merged output '
                f'has {len(confounds)}'
            )
        merged = volumes[0] if len(volumes) == 1 else np.stack(volumes, axis=-1)
        out_file = os.path.join(runtime.cwd, f'{kind}_map.nii.gz')
        out_img = nb.Nifti1Image(merged.astype(np.float32), affine)
        out_img.header.set_xyzt_units('mm')
        out_img.to_filename(out_file)

        first = self.inputs.meta_dicts[0]
        sidecar = {
            field: first[field]
            for field in ('Description', 'Units', 'DenoisingMethod', 'Software', 'Quantity')
            if field in first
        }
        if kind == 'noise':
            applied = []
            for meta in self.inputs.meta_dicts:
                applied += [item for item in meta['AppliedScaling'] if item not in applied]

            if 'b0 harmonization' not in applied and any(
                abs(entry['HarmonizationScale'] - 1.0) > 1e-6 for entry in series
            ):
                applied.insert(0, 'b0 harmonization')
            sidecar['AppliedScaling'] = applied

        approximations = []
        for meta in self.inputs.meta_dicts:
            approximations += [
                item for item in meta.get('Approximations', []) if item not in approximations
            ]

        sidecar.update(
            {
                'Interpolation': first['Interpolation'],
                f'{key}Index': index,
                f'{key}Series': series,
            }
        )
        for field in ('SignalRedistributionMethod', 'Resolution'):
            if field in first:
                sidecar[field] = first[field]

        if approximations:
            sidecar['Approximations'] = approximations
        self._results['out_file'] = out_file
        self._results['meta_dict'] = sidecar
        return runtime
