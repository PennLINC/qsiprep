#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Interfaces to generate reportlets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

import os
import time
import re
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec,
    File, Directory, InputMultiPath, InputMultiObject, Str, isdefined,
    SimpleInterface)
from nipype.interfaces import freesurfer as fs
from .gradients import concatenate_bvals, concatenate_bvecs
from matplotlib import animation

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
"""

ABOUT_TEMPLATE = """\t<ul>
\t\t<li>qsiprep version: {version}</li>
\t\t<li>qsiprep command: <code>{command}</code></li>
\t\t<li>Date preprocessed: {date}</li>
\t</ul>
</div>
"""

GROUPING_TEMPLATE = """\t<ul>
\t\t<li>Output Name: {output_name}</li>
{input_files}
</ul>
"""


class SummaryOutputSpec(TraitedSpec):
    out_report = File(exists=True, desc='HTML segment containing summary')


class SummaryInterface(SimpleInterface):
    output_spec = SummaryOutputSpec

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

        return DIFFUSION_TEMPLATE.format(
            pedir=pedir,
            sdc=self.inputs.distortion_correction,
            coregistration=self.inputs.b0_to_t1w_transform,
            hmc_transform=self.inputs.hmc_transform,
            hmc_model=self.inputs.hmc_model,
            denoise_window=self.inputs.dwi_denoise_window,
            output_spaces=', '.join(self.inputs.output_spaces),
            confounds=re.sub(r'[\t ]+', ', ', conflist),
            impute_slice_threshold=self.inputs.impute_slice_threshold
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
