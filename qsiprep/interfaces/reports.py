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

from collections import Counter
from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec,
    File, Directory, InputMultiPath, Str, isdefined,
    SimpleInterface)
from nipype.interfaces import freesurfer as fs
from fmriprep.interfaces import FunctionalSummary
from .bids import BIDS_NAME

SUBJECT_TEMPLATE = """\t<ul class="elem-desc">
\t\t<li>Subject ID: {subject_id}</li>
\t\t<li>Structural images: {n_t1s:d} T1-weighted {t2w}</li>
\t\t<li>Diffusion-weighted series: inputs {n_dwis:d}, outputs {n_outputs:d}</li>
{groupings}
\t\t<li>Resampling targets: {output_spaces}
\t\t<li>FreeSurfer reconstruction: {freesurfer_status}</li>
\t</ul>
"""

FUNCTIONAL_TEMPLATE = """\t\t<h3 class="elem-title">Summary</h3>
\t\t<ul class="elem-desc">
\t\t\t<li>Phase-encoding (PE) direction: {pedir}</li>
\t\t\t<li>Slice timing correction: {stc}</li>
\t\t\t<li>Susceptibility distortion correction: {sdc}</li>
\t\t\t<li>Registration: {registration}</li>
\t\t\t<li>Functional series resampled to spaces: {output_spaces}</li>
\t\t\t<li>Confounds collected: {confounds}</li>
\t\t</ul>
"""

DIFFUSION_TEMPLATE = """\t\t<h3 class="elem-title">Summary</h3>
\t\t<ul class="elem-desc">
\t\t\t<li>Phase-encoding (PE) direction: {pedir}</li>
\t\t\t<li>Susceptibility distortion correction: {sdc}</li>
\t\t\t<li>Registration: {registration}</li>
\t\t\t<li>DWI series resampled to spaces: {output_spaces}</li>
\t\t\t<li>Confounds collected: {confounds}</li>
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
        for output_fname, dwi_files in self.inputs.dwi_groupings:
            n_outputs += 1
            files_desc = []
            if isinstance(dwi_files, dict):
                for pe_dir, dwi_group in dwi_files.items():
                    files_desc.append('\t\t\t<li>PE dir group: %s </li>\n' + pe_dir)
                    for dwi_file in dwi_group:
                        files_desc.append("\t\t\t\t<li> %s </li>" % dwi_file)
                        n_dwis += 1
            else:
                for dwi_file in dwi_files:
                    files_desc.append("\t\t\t<li> %s </li>" % dwi_file)
                    n_dwis += 1
            groupings += GROUPING_TEMPLATE.format(output_name=output_fname,
                                                  input_files='\n'.join(files_desc))

        return SUBJECT_TEMPLATE.format(subject_id=self.inputs.subject_id,
                                       n_t1s=len(self.inputs.t1w),
                                       t2w=t2w_seg,
                                       n_dwi=n_dwis,
                                       n_outputs=n_outputs,
                                       groupings=groupings,
                                       output_spaces=', '.join(output_spaces),
                                       freesurfer_status=freesurfer_status)


class DiffusionSummaryInputSpec(BaseInterfaceInputSpec):
    distortion_correction = traits.Str(desc='Susceptibility distortion correction method',
                                       mandatory=True)
    pe_direction = traits.Enum(None, 'i', 'i-', 'j', 'j-', mandatory=True,
                               desc='Phase-encoding direction detected')
    registration_dof = traits.Enum(6, 9, 12, desc='Registration degrees of freedom',
                                   mandatory=True)
    output_spaces = traits.List(desc='Target spaces')
    confounds_file = File(exists=True, desc='Confounds file')


class DiffusionSummary(SummaryInterface):
    input_spec = DiffusionSummaryInputSpec

    def _generate_segment(self):
        dof = self.inputs.registration_dof
        stc = {True: 'Applied',
               False: 'Not applied',
               'TooShort': 'Skipped (too few volumes)'}[self.inputs.slice_timing]
        reg = {
            'FSL': [
                'FSL <code>flirt</code> with boundary-based registration'
                ' (BBR) metric - %d dof' % dof,
                'FSL <code>flirt</code> rigid registration - 6 dof'],
            'FreeSurfer': [
                'FreeSurfer <code>bbregister</code> '
                '(boundary-based registration, BBR) - %d dof' % dof,
                'FreeSurfer <code>mri_coreg</code> - %d dof' % dof],
        }[self.inputs.registration][self.inputs.fallback]
        if self.inputs.pe_direction is None:
            pedir = 'MISSING - Assuming Anterior-Posterior'
        else:
            pedir = {'i': 'Left-Right', 'j': 'Anterior-Posterior'}[self.inputs.pe_direction[0]]

        if isdefined(self.inputs.confounds_file):
            with open(self.inputs.confounds_file) as cfh:
                conflist = cfh.readline().strip('\n').strip()
        return FUNCTIONAL_TEMPLATE.format(
            pedir=pedir, stc=stc, sdc=self.inputs.distortion_correction, registration=reg,
            output_spaces=', '.join(self.inputs.output_spaces))
            #confounds=re.sub(r'[\t ]+', ', ', conflist))


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
