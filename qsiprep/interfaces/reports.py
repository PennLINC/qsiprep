#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Interfaces to generate reportlets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

import os
import os.path as op
import time
import json
import re
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import pandas as pd
import numpy as np
from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec,
    File, Directory, InputMultiPath, InputMultiObject, Str, isdefined,
    SimpleInterface)
from nipype.interfaces import freesurfer as fs
from .gradients import concatenate_bvals, concatenate_bvecs
from .qc import createB0_ColorFA_Mask_Sprites, createSprite4D
from .bids import get_bids_params

SUBJECT_TEMPLATE = """\t<ul class="elem-desc">
\t\t<li>Subject ID: {subject_id}</li>
\t\t<li>Structural images: {n_t1s:d} T1-weighted {t2w}</li>
\t\t<li>Diffusion-weighted series: inputs {n_dwis:d}, outputs {n_outputs:d}</li>
{groupings}
\t\t<li>Resampling targets: {output_spaces}
\t\t<li>FreeSurfer reconstruction: {freesurfer_status}</li>
\t</ul>
"""

DIFFUSION_TEMPLATE = """\t\t<h3 class="elem-title">Summary</h3>
\t\t<ul class="elem-desc">
\t\t\t<li>Phase-encoding (PE) direction: {pedir}</li>
\t\t\t<li>Susceptibility distortion correction: {sdc}</li>
\t\t\t<li>Coregistration Transform: {coregistration}</li>
\t\t\t<li>Denoising Window: {denoise_window}</li>
\t\t\t<li>HMC Transform: {hmc_transform}</li>
\t\t\t<li>HMC Model: {hmc_model}</li>
\t\t\t<li>DWI series resampled to spaces: {output_spaces}</li>
\t\t\t<li>Confounds collected: {confounds}</li>
\t\t\t<li>Impute slice threshold: {impute_slice_threshold}</li>
\t\t</ul>
{validation_reports}
"""

ABOUT_TEMPLATE = """\t<ul>
\t\t<li>qsiprep version: {version}</li>
\t\t<li>qsiprep command: <code>{command}</code></li>
\t\t<li>Date preprocessed: {date}</li>
\t</ul>
</div>
"""

TOPUP_TEMPLATE = """\
\t\t<p class="elem-desc">
\t\t{summary}</p>
"""

GROUPING_TEMPLATE = """\t<ul>
\t\t<li>Output Name: {output_name}</li>
{input_files}
</ul>
"""

INTERACTIVE_TEMPLATE = """
<script src="https://unpkg.com/vue"></script>
<script src="https://nipreps.github.io/dmriprep-viewer/dmriprepReport.umd.min.js"></script>
<link rel="stylesheet" href="https://nipreps.github.io/dmriprep-viewer/dmriprepReport.css">

<div id="app">
  <demo :report="report"></demo>
</div>

<script>
var report = REPORT
  new Vue({
    components: {
      demo: dmriprepReport
    },
    data () {
      return {
        report
      }
    }
  }).$mount('#app')

</script>
"""


class SummaryOutputSpec(TraitedSpec):
    out_report = File(exists=True, desc='HTML segment containing summary')


class SummaryInterface(SimpleInterface):
    output_spec = SummaryOutputSpec

    def _generate_segment(self):
        raise NotImplementedError()

    def _run_interface(self, runtime):
        segment = self._generate_segment()
        fname = os.path.join(runtime.cwd, 'report.html')
        with open(fname, 'w') as fobj:
            fobj.write(segment)
        self._results['out_report'] = fname
        return runtime


class SubjectSummaryInputSpec(BaseInterfaceInputSpec):
    t1w = InputMultiPath(File(exists=True), desc='T1w structural images')
    t2w = InputMultiPath(File(exists=True), desc='T2w structural images')
    subjects_dir = Directory(desc='FreeSurfer subjects directory')
    subject_id = Str(desc='Subject ID')
    dwi_groupings = traits.Dict(desc='groupings of DWI files and their output names')
    output_spaces = traits.List(desc='Target spaces')
    template = traits.Enum('MNI152NLin2009cAsym', desc='Template space')


class SubjectSummaryOutputSpec(SummaryOutputSpec):
    # This exists to ensure that the summary is run prior to the first ReconAll
    # call, allowing a determination whether there is a pre-existing directory
    subject_id = Str(desc='FreeSurfer subject ID')


class SubjectSummary(SummaryInterface):
    input_spec = SubjectSummaryInputSpec
    output_spec = SubjectSummaryOutputSpec

    def _run_interface(self, runtime):
        if isdefined(self.inputs.subject_id):
            self._results['subject_id'] = self.inputs.subject_id
        return super(SubjectSummary, self)._run_interface(runtime)

    def _generate_segment(self):
        if not isdefined(self.inputs.subjects_dir):
            freesurfer_status = 'Not run'
        else:
            recon = fs.ReconAll(subjects_dir=self.inputs.subjects_dir,
                                subject_id=self.inputs.subject_id,
                                T1_files=self.inputs.t1w,
                                flags='-noskullstrip')
            if recon.cmdline.startswith('echo'):
                freesurfer_status = 'Pre-existing directory'
            else:
                freesurfer_status = 'Run by qsiprep'

        output_spaces = [self.inputs.template if space == 'template' else space
                         for space in self.inputs.output_spaces]

        t2w_seg = ''
        if self.inputs.t2w:
            t2w_seg = '(+ {:d} T2-weighted)'.format(len(self.inputs.t2w))

        # Add text for how the dwis are grouped
        n_dwis = 0
        n_outputs = 0
        groupings = ''
        if isdefined(self.inputs.dwi_groupings):
            for output_fname, group_info in self.inputs.dwi_groupings.items():
                n_outputs += 1
                files_desc = []
                files_desc.append(
                    '\t\t\t<li>Scan group: %s (PE Dir %s)</li><ul>' % (
                        output_fname, group_info['dwi_series_pedir']))
                files_desc.append('\t\t\t\t<li>DWI Files: </li>')
                for dwi_file in group_info['dwi_series']:
                    files_desc.append("\t\t\t\t\t<li> %s </li>" % dwi_file)
                    n_dwis += 1
                fieldmap_type = group_info['fieldmap_info']['suffix']
                if fieldmap_type is not None:
                    files_desc.append('\t\t\t\t<li>Fieldmap type: %s </li>' % fieldmap_type)

                    for key, value in group_info['fieldmap_info'].items():
                        files_desc.append("\t\t\t\t\t<li> %s: %s </li>" % (key, str(value)))
                        n_dwis += 1
                files_desc.append("</ul>")
                groupings += GROUPING_TEMPLATE.format(output_name=output_fname,
                                                      input_files='\n'.join(files_desc))

        return SUBJECT_TEMPLATE.format(subject_id=self.inputs.subject_id,
                                       n_t1s=len(self.inputs.t1w),
                                       t2w=t2w_seg,
                                       n_dwis=n_dwis,
                                       n_outputs=n_outputs,
                                       groupings=groupings,
                                       output_spaces=', '.join(output_spaces),
                                       freesurfer_status=freesurfer_status)


class DiffusionSummaryInputSpec(BaseInterfaceInputSpec):
    distortion_correction = traits.Str(desc='Susceptibility distortion correction method',
                                       mandatory=True)
    pe_direction = traits.Enum(None, 'i', 'i-', 'j', 'j-', mandatory=True,
                               desc='Phase-encoding direction detected')
    distortion_correction = traits.Str(mandatory=True, desc='Method used for SDC')
    impute_slice_threshold = traits.CFloat(desc='threshold for imputing a slice')
    hmc_transform = traits.Str(mandatory=True, desc='transform used during HMC')
    hmc_model = traits.Str(desc='model used for hmc')
    b0_to_t1w_transform = traits.Enum("Rigid", "Affine", desc='Transform type for coregistration')
    dwi_denoise_window = traits.Int(desc='window size for dwidenoise')
    output_spaces = traits.List(desc='Target spaces')
    confounds_file = File(exists=True, desc='Confounds file')
    validation_reports = InputMultiObject(File(exists=True))


class DiffusionSummary(SummaryInterface):
    input_spec = DiffusionSummaryInputSpec

    def _generate_segment(self):
        if self.inputs.pe_direction is None:
            pedir = 'MISSING - Assuming Anterior-Posterior'
        else:
            pedir = {'i': 'Left-Right', 'j': 'Anterior-Posterior'}[self.inputs.pe_direction[0]]

        if isdefined(self.inputs.confounds_file):
            with open(self.inputs.confounds_file) as cfh:
                conflist = cfh.readline().strip('\n').strip()
        else:
            conflist = ''

        validation_summaries = []
        for summary in self.inputs.validation_reports:
            with open(summary, 'r') as summary_f:
                validation_summaries.extend(summary_f.readlines())
        validation_summary = '\n'.join(validation_summaries)

        return DIFFUSION_TEMPLATE.format(
            pedir=pedir,
            sdc=self.inputs.distortion_correction,
            coregistration=self.inputs.b0_to_t1w_transform,
            hmc_transform=self.inputs.hmc_transform,
            hmc_model=self.inputs.hmc_model,
            denoise_window=self.inputs.dwi_denoise_window,
            output_spaces=', '.join(self.inputs.output_spaces),
            confounds=re.sub(r'[\t ]+', ', ', conflist),
            impute_slice_threshold=self.inputs.impute_slice_threshold,
            validation_reports=validation_summary
            )


class AboutSummaryInputSpec(BaseInterfaceInputSpec):
    version = Str(desc='qsiprep version')
    command = Str(desc='qsiprep command')
    # Date not included - update timestamp only if version or command changes


class AboutSummary(SummaryInterface):
    input_spec = AboutSummaryInputSpec

    def _generate_segment(self):
        return ABOUT_TEMPLATE.format(version=self.inputs.version,
                                     command=self.inputs.command,
                                     date=time.strftime("%Y-%m-%d %H:%M:%S %z"))


class TopupSummaryInputSpec(BaseInterfaceInputSpec):
    summary = Str(desc='Summary of TOPUP inputs')


class TopupSummary(SummaryInterface):
    input_spec = TopupSummaryInputSpec

    def _generate_segment(self):
        return TOPUP_TEMPLATE.format(summary=self.inputs.summary)


class GradientPlotInputSpec(BaseInterfaceInputSpec):
    orig_bvec_files = InputMultiObject(File(exists=True), mandatory=True,
                                       desc='bvecs from DWISplit')
    orig_bval_files = InputMultiObject(File(exists=True), mandatory=True,
                                       desc='bvals from DWISplit')
    source_files = traits.List(desc='source file for each gradient')
    final_bvec_file = File(exists=True, desc='bval file')


class GradientPlotOutputSpec(SummaryOutputSpec):
    plot_file = File(exists=True)


class GradientPlot(SummaryInterface):
    input_spec = GradientPlotInputSpec
    output_spec = GradientPlotOutputSpec

    def _run_interface(self, runtime):
        outfile = os.path.join(runtime.cwd, "bvec_plot.gif")
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=0.8)

        orig_bvecs = concatenate_bvecs(self.inputs.orig_bvec_files)
        bvals = concatenate_bvals(self.inputs.orig_bval_files, None)
        if isdefined(self.inputs.source_files):
            file_array = np.array(self.inputs.source_files)
            unique_files, filenums = np.unique(file_array, return_inverse=True)
        else:
            filenums = np.ones_like(bvals)

        # Plot the final bvecs if provided
        final_bvecs = None
        if isdefined(self.inputs.final_bvec_file):
            final_bvecs = np.loadtxt(self.inputs.final_bvec_file).T

        plot_gradients(bvals, orig_bvecs, filenums, outfile, final_bvecs)
        self._results['plot_file'] = outfile
        return runtime


def plot_gradients(bvals, orig_bvecs, source_filenums, output_fname, final_bvecs=None,
                   frames=60):
    qrads = np.sqrt(bvals)
    qvecs = (qrads[:, np.newaxis] * orig_bvecs)
    qx, qy, qz = qvecs.T
    maxvals = qvecs.max(0)
    minvals = qvecs.min(0)

    def add_lines(ax):
        labels = ['L', 'P', 'S']
        for axnum in range(3):
            minvec = np.zeros(3)
            maxvec = np.zeros(3)
            minvec[axnum] = minvals[axnum]
            maxvec[axnum] = maxvals[axnum]
            x, y, z = np.column_stack([minvec, maxvec])
            ax.plot(x, y, z, color="k")
            txt_pos = maxvec + 5
            ax.text(txt_pos[0], txt_pos[1], txt_pos[2], labels[axnum], size=8,
                    zorder=1, color='k')

    if final_bvecs is not None:
        fqx, fqy, fqz = (qrads[:, np.newaxis] * final_bvecs).T
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5),
                                 subplot_kw={"aspect": "equal", "projection": "3d"})
        orig_ax = axes[0]
        final_ax = axes[1]
        axes_list = [orig_ax, final_ax]
        final_ax.scatter(fqx, fqy, fqz, c=source_filenums, marker="+")
        orig_ax.scatter(qx, qy, qz, c=source_filenums, marker="+")
        final_ax.axis('off')
        add_lines(final_ax)
        final_ax.set_title('After Preprocessing')
    else:
        fig, orig_ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5),
                                    subplot_kw={"aspect": "equal", "projection": "3d"})
        axes_list = [orig_ax]
        orig_ax.scatter(qx, qy, qz, c=source_filenums, marker="+")
    orig_ax.axis('off')
    orig_ax.set_title("Original Scheme")
    add_lines(orig_ax)
    # Animate rotating the axes
    rotate_amount = np.ones(frames) * 180 / frames
    stay_put = np.zeros_like(rotate_amount)
    rotate_azim = np.concatenate([rotate_amount, stay_put, -rotate_amount, stay_put])
    rotate_elev = np.concatenate([stay_put, rotate_amount, stay_put, -rotate_amount])
    plt.tight_layout()

    def rotate(i):
        for ax in axes_list:
            ax.azim += rotate_azim[i]
            ax.elev += rotate_elev[i]
        return tuple(axes_list)

    anim = animation.FuncAnimation(fig, rotate, frames=frames*4,
                                   interval=20, blit=False)
    anim.save(output_fname, writer='imagemagick', fps=32)

    plt.close(fig)
    fig = None


def topup_selection_to_report(selected_indices, original_files, spec_lookup,
                              image_source='combined DWI series'):
    """Write a description of how the images were selected for TOPUP.

    >>> selected_indices = [0, 15, 30, 45]
    >>> original_files = ["sub-1_dir-AP_dwi.nii.gz"] * 30 + ["sub-1_dir-PA_dwi.nii.gz"] * 30
    >>> spec_lookup = {"sub-1_dir-AP_dwi.nii.gz": "0 1 0 0.087",
    ...                "sub-1_dir-PA_dwi.nii.gz": "0 -1 0 0.087"}
    >>> print(topup_selection_to_report(selected_indices, original_files, spec_lookup))
    A total of 2 distortion groups was included in the combined dwi data. Distortion \
group '0 1 0 0.087' was represented by images 0, 15 from sub-1_dir-AP_dwi.nii.gz. \
Distortion group '0 -1 0 0.087' was represented by images 0, 15 from sub-1_dir-PA_dwi.nii.gz. "

    Or

    >>> selected_indices = [0, 15, 30, 45]
    >>> original_files = ["sub-1_dir-AP_run-1_dwi.nii.gz"] * 15 + [
    ...                   "sub-1_dir-AP_run-2_dwi.nii.gz"] * 15 + [
    ...                   "sub-1_dir-PA_dwi.nii.gz"] * 30
    >>> spec_lookup = {"sub-1_dir-AP_run-1_dwi.nii.gz": "0 1 0 0.087",
    ...                "sub-1_dir-AP_run-2_dwi.nii.gz": "0 1 0 0.087",
    ...                "sub-1_dir-PA_dwi.nii.gz": "0 -1 0 0.087"}
    >>> print(topup_selection_to_report(selected_indices, original_files, spec_lookup))
    A total of 2 distortion groups was included in the combined dwi data. Distortion \
group '0 1 0 0.087' was represented by image 0 from sub-1_dir-AP_run-1_dwi.nii.gz and \
image 0 from sub-1_dir-AP_run-2_dwi.nii.gz. Distortion group '0 -1 0 0.087' was represented \
by images 0, 15 from sub-1_dir-PA_dwi.nii.gz.

    >>> selected_indices = [0, 15, 30, 45, 60]
    >>> original_files = ["sub-1_dir-AP_run-1_dwi.nii.gz"] * 15 + [
    ...                   "sub-1_dir-AP_run-2_dwi.nii.gz"] * 15 + [
    ...                   "sub-1_dir-AP_run-3_dwi.nii.gz"] * 15 + [
    ...                   "sub-1_dir-PA_dwi.nii.gz"] * 30
    >>> spec_lookup = {"sub-1_dir-AP_run-1_dwi.nii.gz": "0 1 0 0.087",
    ...                "sub-1_dir-AP_run-2_dwi.nii.gz": "0 1 0 0.087",
    ...                "sub-1_dir-AP_run-3_dwi.nii.gz": "0 1 0 0.087",
    ...                "sub-1_dir-PA_dwi.nii.gz": "0 -1 0 0.087"}
    >>> print(topup_selection_to_report(selected_indices, original_files, spec_lookup))
    A total of 2 distortion groups was included in the combined dwi data. Distortion \
group '0 1 0 0.087' was represented by image 0 from sub-1_dir-AP_run-1_dwi.nii.gz, \
image 0 from sub-1_dir-AP_run-2_dwi.nii.gz and image 0 from sub-1_dir-AP_run-3_dwi.nii.gz. \
Distortion group '0 -1 0 0.087' was represented by images 0, 15 from sub-1_dir-PA_dwi.nii.gz.

    >>> selected_indices = [0, 15, 30, 45]
    >>> original_files = ["sub-1_dir-PA_dwi.nii.gz"] * 60
    >>> spec_lookup = {"sub-1_dir-PA_dwi.nii.gz": "0 -1 0 0.087"}
    >>> print(topup_selection_to_report(selected_indices, original_files, spec_lookup))
    A total of 1 distortion group was included in the combined dwi data. \
Distortion group '0 -1 0 0.087' was represented by images 0, 15, 30, 45 \
from sub-1_dir-PA_dwi.nii.gz.

    """
    image_indices = defaultdict(list)
    for imgnum, image in enumerate(original_files):
        image_indices[image].append(imgnum)

    # Collect the original volume number within each source image
    selected_per_image = defaultdict(list)
    for b0_index in selected_indices:
        b0_image = original_files[b0_index]
        first_index = min(image_indices[b0_image])
        within_image_index = b0_index - first_index
        selected_per_image[b0_image].append(within_image_index)

    # Collect the images and indices within each warp group
    selected_per_warp_group = defaultdict(list)
    for original_image, selection in selected_per_image.items():
        warp_group = spec_lookup[original_image]
        selected_per_warp_group[warp_group].append((original_image, selection))

    # Make the description
    num_groups = len(selected_per_warp_group)
    plural = 's' if num_groups > 1 else ''
    plural2 = 'were' if plural == 's' else 'was'
    desc = ["A total of {num_groups} distortion group{plural} {plural2} included in the "
            "{image_source} data. ".format(num_groups=num_groups, plural=plural,
                                           plural2=plural2, image_source=image_source)]
    for distortion_group, image_list in selected_per_warp_group.items():
        group_desc = [
            "Distortion group '{spec}' was represented by ".format(spec=distortion_group)]
        for image_name, image_indices in image_list:
            formatted_indices = ", ".join(map(str, image_indices))
            plural = 's' if len(image_indices) > 1 else ''
            group_desc += [
                "image{plural} {imgnums} from {img_name}".format(plural=plural,
                                                                 imgnums=formatted_indices,
                                                                 img_name=image_name),
                ", "]
        group_desc[-1] = ". "
        if len(image_list) > 1:
            group_desc[-3] = " and "
        desc += group_desc

    return ''.join(desc)


class _SeriesQCInputSpec(BaseInterfaceInputSpec):
    pre_qc = File(exists=True, desc='qc file from the raw data')
    t1_qc = File(exists=True, desc='qc file from preprocessed image in t1 space')
    mni_qc = File(exists=True, desc='qc file from preprocessed image in template space')
    confounds_file = File(exists=True, desc='confounds file')
    t1_dice_score = traits.Float()
    mni_dice_score = traits.Float()
    output_file_name = traits.File()


class _SeriesQCOutputSpec(TraitedSpec):
    series_qc_file = File(exists=True)


class SeriesQC(SimpleInterface):
    input_spec = _SeriesQCInputSpec
    output_spec = _SeriesQCOutputSpec

    def _run_interface(self, runtime):
        image_qc = _load_qc_file(self.inputs.pre_qc, prefix="raw_")
        if isdefined(self.inputs.t1_qc):
            image_qc.update(_load_qc_file(self.inputs.t1_qc, prefix="t1_"))
        if isdefined(self.inputs.mni_qc):
            image_qc.update(_load_qc_file(self.inputs.mni_qc, prefix="mni_"))
        motion_summary = calculate_motion_summary(self.inputs.confounds_file)
        image_qc.update(motion_summary)

        # Add in Dice scores if available
        if isdefined(self.inputs.t1_dice_score):
            image_qc['t1_dice_distance'] = [self.inputs.t1_dice_score]
        if isdefined(self.inputs.mni_dice_score):
            image_qc['mni_dice_distance'] = [self.inputs.mni_dice_score]

        # Get the metadata
        output_file = self.inputs.output_file_name
        image_qc['file_name'] = output_file
        bids_info = get_bids_params(output_file)
        image_qc.update(bids_info)
        output = op.join(runtime.cwd, "dwi_qc.csv")
        pd.DataFrame(image_qc).to_csv(output, index=False)
        self._results['series_qc_file'] = output
        return runtime


def _load_qc_file(fname, prefix=""):
    with open(fname, "r") as qc_file:
        qc_data = qc_file.readlines()
    data = qc_data[2]
    parts = data.split('\t')
    _, dims, voxel_size, dirs, max_b, _, ndc, bad_slices, _ = parts
    voxelsx, voxelsy, voxelsz = map(float, voxel_size.strip().split())
    dimx, dimy, dimz = map(float, dims.strip().split())
    n_dirs = float(dirs)
    max_b = float(max_b)
    dwi_corr = float(ndc)
    n_bad_slices = float(bad_slices)
    data = {
        prefix + 'dimension_x': [dimx],
        prefix + 'dimension_y': [dimy],
        prefix + 'dimension_z': [dimz],
        prefix + 'voxel_size_x': [voxelsx],
        prefix + 'voxel_size_y': [voxelsy],
        prefix + 'voxel_size_z': [voxelsz],
        prefix + 'max_b': [max_b],
        prefix + 'neighbor_corr': [dwi_corr],
        prefix + 'num_bad_slices': [n_bad_slices],
        prefix + 'num_directions': [n_dirs]
    }
    return data


def calculate_motion_summary(confounds_tsv):
    if not isdefined(confounds_tsv) or confounds_tsv is None:
        return {
            "mean_fd": [np.nan],
            "max_fd": [np.nan],
            "max_rotation": [np.nan],
            "max_translation": [np.nan],
            "max_rel_rotation": [np.nan],
            "max_rel_translation": [np.nan]
        }
    df = pd.read_csv(confounds_tsv, delimiter="\t")
    translations = df[['trans_x', 'trans_y', 'trans_z']].to_numpy()
    rotations = df[['rot_x', 'rot_y', 'rot_z']].to_numpy()

    def padded_diff(data):
        out = np.zeros_like(data)
        out[1:] = np.diff(data, axis=0)
        return out

    drotations = padded_diff(rotations)
    dtranslations = padded_diff(translations)

    # We don't want the relative values across the boundaries of runs.
    # Determine which values should be ignored
    file_labels, _ = pd.factorize(df['original_file'])
    new_files = padded_diff(file_labels)

    def file_masked(data):
        masked_data = data.copy()
        masked_data[new_files > 0] = 0
        return masked_data

    framewise_disp = file_masked(df['framewise_displacement'].abs())
    motion_summary = {
        "mean_fd": [framewise_disp.mean()],
        "max_fd": [framewise_disp.max()],
        "max_rotation": [file_masked(np.abs(rotations)).max()],
        "max_translation": [file_masked(np.abs(translations)).max()],
        "max_rel_rotation": [file_masked(np.abs(drotations)).max()],
        "max_rel_translation": [file_masked(np.abs(dtranslations)).max()]
    }
    return motion_summary


class _InteractiveReportInputSpec(TraitedSpec):
    raw_dwi_file = File(exists=True, mandatory=True)
    processed_dwi_file = File(exists=True, mandatory=True)
    confounds_file = File(exists=True, mandatory=True)
    mask_file = File(exists=True, mandatory=True)
    color_fa = File(exists=True, mandatory=True)
    carpetplot_data = File(exists=True, mandatory=True)


class InteractiveReport(SimpleInterface):
    input_spec = _InteractiveReportInputSpec
    output_spec = SummaryOutputSpec

    def _run_interface(self, runtime):
        report = {}
        report['dwi_corrected'] = createSprite4D(self.inputs.processed_dwi_file)

        b0, colorFA, mask = createB0_ColorFA_Mask_Sprites(self.inputs.processed_dwi_file,
                                                          self.inputs.color_fa,
                                                          self.inputs.mask_file)
        report['carpetplot'] = []
        if isdefined(self.inputs.carpetplot_data):
            with open(self.inputs.carpetplot_data, 'r') as carpet_f:
                carpet_data = json.load(carpet_f)
            report.update(carpet_data)

        report['b0'] = b0
        report['colorFA'] = colorFA
        report['anat_mask'] = mask
        report['outlier_volumes'] = []
        report['eddy_params'] = [[i, i] for i in range(30)]
        eddy_qc = {}
        report['eddy_quad'] = eddy_qc

        df = pd.read_csv(self.inputs.confounds_file, delimiter="\t")
        translations = df[['trans_x', 'trans_y', 'trans_z']].to_numpy()
        rms = np.sqrt((translations ** 2).sum(1))
        fdisp = df['framewise_displacement'].tolist()
        fdisp[0] = None
        report['eddy_params'] = [[fd_, rms_] for fd_, rms_ in zip(fdisp, rms)]

        # Get the sampling scheme
        xyz = df[["grad_x", "grad_y", "grad_z"]].to_numpy()
        bval = df['bval'].to_numpy()
        qxyz = np.sqrt(bval)[:, None] * xyz
        report['q_coords'] = qxyz.tolist()
        report['color'] = _filename_to_colors(df['original_file'])

        safe_json = json.dumps(report)
        out_file = op.join(runtime.cwd, "interactive_report.html")
        with open(out_file, "w") as out_html:
            out_html.write(INTERACTIVE_TEMPLATE.replace("REPORT", safe_json))
        self._results['out_report'] = out_file
        return runtime


def _filename_to_colors(labels_column, colormap="rainbow"):
    cmap = matplotlib.cm.get_cmap(colormap)
    labels, _ = pd.factorize(labels_column)
    n_samples = labels.shape[0]
    max_label = labels.max()
    if max_label == 0:
        return [(1.0, 0.0, 0.0)] * n_samples
    labels = labels / max_label
    colors = np.array([cmap(label) for label in labels])
    return colors.tolist()
