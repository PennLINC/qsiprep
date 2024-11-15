# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Image tools interfaces
~~~~~~~~~~~~~~~~~~~~~~


"""

from mimetypes import guess_type

import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
import seaborn as sns
from matplotlib import gridspec as mgs
from nipype import logging
from nipype.interfaces import ants
from nipype.interfaces.ants import Registration
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)
from nipype.interfaces.mixins import reporting
from nipype.utils.filemanip import fname_presuffix
from niworkflows.interfaces.norm import SpatialNormalization, _SpatialNormalizationInputSpec
from niworkflows.interfaces.reportlets.base import RegistrationRC, _SVGReportCapableInputSpec
from niworkflows.interfaces.reportlets.registration import (
    _ANTSRegistrationInputSpecRPT,
    _ANTSRegistrationOutputSpecRPT,
)
from seaborn import color_palette

LOGGER = logging.getLogger('nipype.interface')


class ANTSRegistrationRPT(RegistrationRC, Registration):
    input_spec = _ANTSRegistrationInputSpecRPT
    output_spec = _ANTSRegistrationOutputSpecRPT

    def _post_run_hook(self, runtime):
        self._fixed_image = self.inputs.fixed_image[0]
        self._moving_image = self.aggregate_outputs(runtime=runtime).warped_image
        LOGGER.info(
            'Report - setting fixed (%s) and moving (%s) images',
            self._fixed_image,
            self._moving_image,
        )

        return super()._post_run_hook(runtime)


class dMRIPlot:
    """
    Generates the dMRI Summary Plot
    """

    def __init__(
        self,
        sliceqc_file,
        mask_file,
        confounds,
        usecols=None,
        units=None,
        vlines=None,
        spikes_files=None,
        min_slice_size_percentile=10.0,
    ):
        if sliceqc_file.endswith('.npz') or sliceqc_file.endswith('.npy'):
            self.qc_data = np.load(sliceqc_file)
        else:
            # Load the info from eddy
            slice_scores = np.loadtxt(sliceqc_file, skiprows=1)
            # Get the slice counts
            mask_img = nb.load(mask_file)
            mask = mask_img.get_fdata() > 0
            masked_slices = (
                mask * np.arange(mask_img.shape[2])[np.newaxis, np.newaxis, :]
            ).astype(int)
            slice_nums, slice_counts = np.unique(masked_slices[mask], return_counts=True)
            self.qc_data = {'slice_scores': slice_scores, 'slice_counts': slice_counts}

        self.confounds = confounds

    def plot(self, figure=None):
        """Main plotter"""
        sns.set_style('whitegrid')
        sns.set_context('paper', font_scale=0.8)

        if figure is None:
            figure = plt.gcf()

        to_plot = ['bval', 'hmc_xcorr', 'framewise_displacement']
        confound_names = [p for p in to_plot if p in self.confounds.columns]
        nconfounds = len(confound_names)
        nrows = 1 + nconfounds

        # Create grid
        grid = mgs.GridSpec(
            nrows, 1, wspace=0.0, hspace=0.05, height_ratios=[1] * (nrows - 1) + [5]
        )

        grid_id = 0
        palette = color_palette('husl', nconfounds)

        for i, name in enumerate(confound_names):
            tseries = self.confounds[name]
            confoundplot(tseries, grid[grid_id], color=palette[i], name=name)
            grid_id += 1

        plot_sliceqc(
            self.qc_data['slice_scores'].T,
            self.qc_data['slice_counts'],
            subplot=grid[-1],
        )
        return figure


def plot_sliceqc(
    slice_data,
    nperslice,
    size=(950, 800),
    subplot=None,
    title=None,
    output_file=None,
    lut=None,
    tr=None,
):
    """
    Plot an image representation of voxel intensities across time also know
    as the "carpet plot" or "Power plot". See Jonathan Power Neuroimage
    2017 Jul 1; 154:150-158.

    Parameters
    ----------
        slice_data: 2d array
            errors in each slice for each volume
        nperslice: 1d array
            number of voxels included in each slice
        axes : matplotlib axes, optional
            The axes used to display the plot. If None, the complete
            figure is used.
        title : string, optional
            The title displayed on the figure.
        output_file : string, or None, optional
            The name of an image file to export the plot to. Valid extensions
            are .png, .pdf, .svg. If output_file is not None, the plot
            is saved to a file, and the display is closed.
        tr : float , optional
            Specify the TR, if specified it uses this value. If left as None,
            # Frames is plotted instead of time.
    """

    # Define TR and number of frames
    notr = False
    if tr is None:
        notr = True
        tr = 1.0

    # If subplot is not defined
    if subplot is None:
        subplot = mgs.GridSpec(1, 1)[0]

    # Define nested GridSpec
    wratios = [1, 100]
    gs = mgs.GridSpecFromSubplotSpec(1, 2, subplot_spec=subplot, width_ratios=wratios, wspace=0.0)

    # Segmentation colorbar
    ax0 = plt.subplot(gs[0])
    ax0.set_yticks([])
    ax0.set_xticks([])
    ax0.imshow(nperslice[:, np.newaxis], interpolation='nearest', aspect='auto', cmap='plasma')
    ax0.grid(False)
    ax0.spines['left'].set_visible(False)
    ax0.spines['bottom'].set_color('none')
    ax0.spines['bottom'].set_visible(False)

    # Carpet plot
    ax1 = plt.subplot(gs[1])
    ax1.imshow(slice_data, interpolation='nearest', aspect='auto', cmap='viridis')
    ax1.grid(False)
    ax1.set_yticks([])
    ax1.set_yticklabels([])

    # Set 10 frame markers in X axis
    interval = max((int(slice_data.shape[1] + 1) // 10, int(slice_data.shape[1] + 1) // 5, 1))
    xticks = list(range(0, slice_data.shape[1])[::interval])
    ax1.set_xticks(xticks)
    if notr:
        ax1.set_xlabel('time (frame #)')
    else:
        ax1.set_xlabel('time (s)')
    labels = tr * (np.array(xticks))
    ax1.set_xticklabels([f'{t:.2f}' for t in labels.tolist()], fontsize=5)

    # Remove and redefine spines
    for side in ['top', 'right']:
        # Toggle the spine objects
        ax0.spines[side].set_color('none')
        ax0.spines[side].set_visible(False)
        ax1.spines[side].set_color('none')
        ax1.spines[side].set_visible(False)

    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_color('none')
    ax1.spines['left'].set_visible(False)

    if output_file is not None:
        figure = plt.gcf()
        figure.savefig(output_file, bbox_inches='tight')
        plt.close(figure)
        figure = None
        return output_file

    return [ax0, ax1], gs


def confoundplot(
    tseries,
    gs_ts,
    gs_dist=None,
    name=None,
    units=None,
    tr=None,
    hide_x=True,
    color='b',
    nskip=0,
    cutoff=None,
    ylims=None,
):
    # Define TR and number of frames
    notr = False
    if tr is None:
        notr = True
        tr = 1.0
    ntsteps = len(tseries)
    tseries = np.array(tseries)

    # Define nested GridSpec
    gs = mgs.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_ts, width_ratios=[1, 100], wspace=0.0)

    ax_ts = plt.subplot(gs[1])
    ax_ts.grid(False)

    # Set 10 frame markers in X axis
    interval = max((ntsteps // 10, ntsteps // 5, 1))
    xticks = list(range(0, ntsteps)[::interval])
    ax_ts.set_xticks(xticks)

    if not hide_x:
        if notr:
            ax_ts.set_xlabel('time (frame #)')
        else:
            ax_ts.set_xlabel('time (s)')
            labels = tr * np.array(xticks)
            ax_ts.set_xticklabels([f'{t:.2f}' for t in labels.tolist()])
    else:
        ax_ts.set_xticklabels([])

    if name is not None:
        if units is not None:
            name += f' [{units}]'

        ax_ts.annotate(
            name,
            xy=(0.0, 0.7),
            xytext=(0, 0),
            xycoords='axes fraction',
            textcoords='offset points',
            va='center',
            ha='left',
            color=color,
            size=8,
            bbox={
                'boxstyle': 'round',
                'fc': 'w',
                'ec': 'none',
                'color': 'none',
                'lw': 0,
                'alpha': 0.8,
            },
        )

    for side in ['top', 'right']:
        ax_ts.spines[side].set_color('none')
        ax_ts.spines[side].set_visible(False)

    if not hide_x:
        ax_ts.spines['bottom'].set_position(('outward', 20))
        ax_ts.xaxis.set_ticks_position('bottom')
    else:
        ax_ts.spines['bottom'].set_color('none')
        ax_ts.spines['bottom'].set_visible(False)

    # ax_ts.spines["left"].set_position(('outward', 30))
    ax_ts.spines['left'].set_color('none')
    ax_ts.spines['left'].set_visible(False)
    # ax_ts.yaxis.set_ticks_position('left')

    ax_ts.set_yticks([])
    ax_ts.set_yticklabels([])

    nonnan = tseries[~np.isnan(tseries)]
    if nonnan.size > 0:
        # Calculate Y limits
        def_ylims = [nonnan.min() - 0.1 * abs(nonnan.min()), 1.1 * nonnan.max()]
        if ylims is not None:
            if ylims[0] is not None:
                def_ylims[0] = min([def_ylims[0], ylims[0]])
            if ylims[1] is not None:
                def_ylims[1] = max([def_ylims[1], ylims[1]])

        # Add space for plot title and mean/SD annotation
        def_ylims[0] -= 0.1 * (def_ylims[1] - def_ylims[0])

        ax_ts.set_ylim(def_ylims)

        # Annotate stats
        maxv = nonnan.max()
        mean = nonnan.mean()
        stdv = nonnan.std()
        p95 = np.percentile(nonnan, 95.0)
    else:
        maxv = 0
        mean = 0
        stdv = 0
        p95 = 0

    stats_label = (
        r'max: {max:.3f}{units} $\bullet$ mean: {mean:.3f}{units} '
        r'$\bullet$ $\sigma$: {sigma:.3f}'
    ).format(max=maxv, mean=mean, units=units or '', sigma=stdv)
    ax_ts.annotate(
        stats_label,
        xy=(0.98, 0.7),
        xycoords='axes fraction',
        xytext=(0, 0),
        textcoords='offset points',
        va='center',
        ha='right',
        color=color,
        size=4,
        bbox={
            'boxstyle': 'round',
            'fc': 'w',
            'ec': 'none',
            'color': 'none',
            'lw': 0,
            'alpha': 0.8,
        },
    )

    # Annotate percentile 95
    ax_ts.plot((0, ntsteps - 1), [p95] * 2, linewidth=0.1, color='lightgray')
    ax_ts.annotate(
        f'{p95:.2f}',
        xy=(0, p95),
        xytext=(-1, 0),
        textcoords='offset points',
        va='center',
        ha='right',
        color='lightgray',
        size=3,
    )

    if cutoff is None:
        cutoff = []

    for thr in cutoff:
        ax_ts.plot((0, ntsteps - 1), [thr] * 2, linewidth=0.2, color='dimgray')

        ax_ts.annotate(
            f'{thr:.2f}',
            xy=(0, thr),
            xytext=(-1, 0),
            textcoords='offset points',
            va='center',
            ha='right',
            color='dimgray',
            size=3,
        )

    ax_ts.plot(tseries, color=color, linewidth=0.8)
    ax_ts.set_xlim((0, ntsteps - 1))

    if gs_dist is not None:
        ax_dist = plt.subplot(gs_dist)
        sns.distplot(tseries, vertical=True, ax=ax_dist)
        ax_dist.set_xlabel('Timesteps')
        ax_dist.set_ylim(ax_ts.get_ylim())
        ax_dist.set_yticklabels([])

        return [ax_ts, ax_dist], gs
    return ax_ts, gs


class RobustMNINormalizationInputSpecRPT(
    _SVGReportCapableInputSpec,
    _SpatialNormalizationInputSpec,
):
    # Template orientation.
    orientation = traits.Enum(
        'LPS',
        mandatory=True,
        usedefault=True,
        desc='modify template orientation (should match input image)',
    )


class RobustMNINormalizationOutputSpecRPT(
    reporting.ReportCapableOutputSpec,
    ants.registration.RegistrationOutputSpec,
):
    # Try to work around TraitError of "undefined 'reference_image' attribute"
    reference_image = traits.File(desc='the output reference image')


class RobustMNINormalizationRPT(RegistrationRC, SpatialNormalization):
    input_spec = RobustMNINormalizationInputSpecRPT
    output_spec = RobustMNINormalizationOutputSpecRPT

    def _post_run_hook(self, runtime):
        # We need to dig into the internal ants.Registration interface
        self._fixed_image = self._get_ants_args()['fixed_image']
        if isinstance(self._fixed_image, list | tuple):
            self._fixed_image = self._fixed_image[0]  # get first item if list

        if self._get_ants_args().get('fixed_image_mask') is not None:
            self._fixed_image_mask = self._get_ants_args().get('fixed_image_mask')
        self._moving_image = self.aggregate_outputs(runtime=runtime).warped_image
        LOGGER.info(
            'Report - setting fixed (%s) and moving (%s) images',
            self._fixed_image,
            self._moving_image,
        )

        return super()._post_run_hook(runtime)


class FUGUEvsm2ANTSwarpInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input displacements field map')
    pe_dir = traits.Enum('i', 'i-', 'j', 'j-', 'k', 'k-', desc='phase-encoding axis')


class FUGUEvsm2ANTSwarpOutputSpec(TraitedSpec):
    out_file = File(desc='the output warp field')


class FUGUEvsm2ANTSwarp(SimpleInterface):
    """Convert a voxel-shift-map to ants warp."""

    input_spec = FUGUEvsm2ANTSwarpInputSpec
    output_spec = FUGUEvsm2ANTSwarpOutputSpec

    def _run_interface(self, runtime):
        nii = nb.load(self.inputs.in_file)

        phaseEncDim = {'i': 0, 'j': 1, 'k': 2}[self.inputs.pe_dir[0]]

        if len(self.inputs.pe_dir) == 2:
            phaseEncSign = 1.0
        else:
            phaseEncSign = -1.0

        # Fix header
        hdr = nii.header.copy()
        hdr.set_data_dtype(np.dtype('<f4'))
        hdr.set_intent('vector', (), '')

        # Get data, convert to mm
        data = nii.get_fdata()

        aff = np.diag([1.0, 1.0, -1.0])
        if np.linalg.det(aff) < 0 and phaseEncDim != 0:
            # Reverse direction since ITK is LPS
            aff *= -1.0

        aff = aff.dot(nii.affine[:3, :3])

        data *= phaseEncSign * nii.header.get_zooms()[phaseEncDim]

        # Add missing dimensions
        zeros = np.zeros_like(data)
        field = [zeros, zeros]
        field.insert(phaseEncDim, data)
        field = np.stack(field, -1)
        # Add empty axis
        field = field[:, :, :, np.newaxis, :]

        # Write out
        self._results['out_file'] = fname_presuffix(
            self.inputs.in_file, suffix='_antswarp', newpath=runtime.cwd
        )
        nb.Nifti1Image(field.astype(np.dtype('<f4')), nii.affine, hdr).to_filename(
            self._results['out_file']
        )

        return runtime


def _mat2itk(args):
    from nipype.interfaces.c3 import C3dAffineTool
    from nipype.utils.filemanip import fname_presuffix

    in_file, in_ref, in_src, index, newpath = args
    # Generate a temporal file name
    out_file = fname_presuffix(in_file, suffix='_itk-%05d.txt' % index, newpath=newpath)

    # Run c3d_affine_tool
    C3dAffineTool(
        transform_file=in_file,
        reference_file=in_ref,
        source_file=in_src,
        fsl2ras=True,
        itk_transform=out_file,
        resource_monitor=False,
    ).run()
    transform = '#Transform %d\n' % index
    with open(out_file) as itkfh:
        transform += ''.join(itkfh.readlines()[2:])

    return (index, transform)


def _applytfms(args):
    """
    Applies ANTs' antsApplyTransforms to the input image.
    All inputs are zipped in one tuple to make it digestible by
    multiprocessing's map
    """
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix
    from niworkflows.interfaces.fixes import FixHeaderApplyTransforms as ApplyTransforms

    in_file, in_xform, ifargs, index, newpath = args
    out_file = fname_presuffix(
        in_file, suffix='_xform-%05d' % index, newpath=newpath, use_ext=True
    )

    copy_dtype = ifargs.pop('copy_dtype', False)
    xfm = ApplyTransforms(
        input_image=in_file, transforms=in_xform, output_image=out_file, **ifargs
    )
    xfm.terminal_output = 'allatonce'
    xfm.resource_monitor = False
    runtime = xfm.run().runtime

    if copy_dtype:
        nii = nb.load(out_file)
        in_dtype = nb.load(in_file).get_data_dtype()

        # Overwrite only iff dtypes don't match
        if in_dtype != nii.get_data_dtype():
            nii.set_data_dtype(in_dtype)
            nii.to_filename(out_file)

    return (out_file, runtime.cmdline)


def _arrange_xfms(transforms, num_files, tmp_folder):
    """
    Convenience method to arrange the list of transforms that should be applied
    to each input file
    """
    base_xform = ['#Insight Transform File V1.0', '#Transform 0']
    # Initialize the transforms matrix
    xfms_T = []
    for i, tf_file in enumerate(transforms):
        # If it is a deformation field, copy to the tfs_matrix directly
        if guess_type(tf_file)[0] != 'text/plain':
            xfms_T.append([tf_file] * num_files)
            continue

        with open(tf_file) as tf_fh:
            tfdata = tf_fh.read().strip()

        # If it is not an ITK transform file, copy to the tfs_matrix directly
        if not tfdata.startswith('#Insight Transform File'):
            xfms_T.append([tf_file] * num_files)
            continue

        # Count number of transforms in ITK transform file
        nxforms = tfdata.count('#Transform')

        # Remove first line
        tfdata = tfdata.split('\n')[1:]

        # If it is a ITK transform file with only 1 xform, copy to the tfs_matrix directly
        if nxforms == 1:
            xfms_T.append([tf_file] * num_files)
            continue

        if nxforms != num_files:
            raise RuntimeError(
                'Number of transforms (%d) found in the ITK file does not match'
                ' the number of input image files (%d).' % (nxforms, num_files)
            )

        # At this point splitting transforms will be necessary, generate a base name
        out_base = fname_presuffix(
            tf_file, suffix='_pos-%03d_xfm-{:05d}' % i, newpath=tmp_folder.name
        ).format
        # Split combined ITK transforms file
        split_xfms = []
        for xform_i in range(nxforms):
            # Find start token to extract
            startidx = tfdata.index('#Transform %d' % xform_i)
            next_xform = base_xform + tfdata[startidx + 1 : startidx + 4] + ['']
            xfm_file = out_base(xform_i)
            with open(xfm_file, 'w') as out_xfm:
                out_xfm.write('\n'.join(next_xform))
            split_xfms.append(xfm_file)
        xfms_T.append(split_xfms)

    # Transpose back (only Python 3)
    return list(map(list, zip(*xfms_T, strict=False)))
