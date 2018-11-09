#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
qsiprep base processing workflows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_qsiprep_wf
.. autofunction:: init_single_subject_wf

"""

import sys
import os
from copy import deepcopy

from nipype import __version__ as nipype_ver
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nilearn import __version__ as nilearn_ver

from fmriprep.engine import Workflow
from ..interfaces import (BIDSDataGrabber, BIDSInfo, BIDSFreeSurferDir,
                          SubjectSummary, AboutSummary, DerivativesDataSink)
from ..utils.bids import collect_data
from ..utils.misc import fix_multi_T1w_source_name
from ..__about__ import __version__

from fmriprep.workflows.anatomical import init_anat_preproc_wf
from .dwi import init_dwi_preproc_wf


def init_qsiprep_wf(subject_list, run_uuid, work_dir, output_dir, bids_dir,
                    ignore, debug, low_mem, anat_only, longitudinal, hires,
                    denoise_before_combining, dwi_denoise_window,
                    combine_all_dwis, discard_repeated_samples, omp_nthreads,
                    skull_strip_template, skull_strip_fixed_seed, freesurfer,
                    output_spaces, template, b0_motion_corr_to, b0_to_t1w_dof,
                    fmap_bspline, fmap_demean, use_syn, force_syn):
    """
    This workflow organizes the execution of qsiprep, with a sub-workflow for
    each subject.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        import os
        os.environ['FREESURFER_HOME'] = os.getcwd()
        from qsiprep.workflows.base import init_qsiprep_wf
        wf = init_qsiprep_wf(subject_list=['qsipreptest'],
                              task_id='',
                              run_uuid='X',
                              work_dir='.',
                              output_dir='.',
                              bids_dir='.',
                              ignore=[],
                              debug=False,
                              low_mem=False,
                              anat_only=False,
                              longitudinal=False,
                              t2s_coreg=False,
                              freesurfer=False,
                              hires=False,
                              denoise_before_combining=True,
                              dwi_denoise_window=7,
                              combine_all_dwis=True,
                              discard_repeated_samples=True,
                              omp_nthreads=1,
                              skull_strip_template='OASIS',
                              skull_strip_fixed_seed=False,
                              output_spaces=['T1w', 'template'],
                              template='MNI152NLin2009cAsym',
                              b0_motion_corr_to='iterative',
                              b0_to_t1w_dof=9,
                              fmap_bspline=False,
                              fmap_demean=True,
                              use_syn=True,
                              force_syn=True)


    Parameters

        subject_list : list
            List of subject labels
        run_uuid : str
            Unique identifier for execution instance
        work_dir : str
            Directory in which to store workflow execution state and temporary
            files
        output_dir : str
            Directory in which to save derivatives
        bids_dir : str
            Root directory of BIDS dataset
        ignore : list
            Preprocessing steps to skip (may include "slicetiming",
            "fieldmaps")
        low_mem : bool
            Write uncompressed .nii files in some cases to reduce memory usage
        anat_only : bool
            Disable diffusion workflows
        longitudinal : bool
            Treat multiple sessions as longitudinal (may increase runtime)
            See sub-workflows for specific differences
        dwi_denoise_window : int
            window size in voxels for ``dwidenoise``. Must be odd. If 0, '
            '``dwidwenoise`` will not be run'
        denoise_before_combining : bool
            'run ``dwidenoise`` before combining dwis. Requires ``combine_all_dwis``'
        combine_all_dwis : bool
            Combine all dwi sequences within a session into a single data set
        discard_repeated_samples : Bool
            Ignore images if their q space coordinate has already been sampled
        omp_nthreads : int
            Maximum number of threads an individual process may use
        skull_strip_template : str
            Name of ANTs skull-stripping template ('OASIS' or 'NKI')
        skull_strip_fixed_seed : bool
            Do not use a random seed for skull-stripping - will ensure
            run-to-run replicability when used with --omp-nthreads 1
        freesurfer : bool
            Enable FreeSurfer surface reconstruction (may increase runtime)
        hires : bool
            Enable sub-millimeter preprocessing in FreeSurfer
        output_spaces : list
            List of output spaces functional images are to be resampled to.
            Some parts of pipeline will only be instantiated for some output
            spaces.

            Valid spaces:

             - T1w
             - template

        template : str
            Name of template targeted by ``template`` output space
        b0_motion_corr_to : str
            Motion correct using the 'first' b0 image or use an 'iterative'
            method to motion correct to the midpoint of the b0 images
        b0_to_t1w_dof : 6, 9 or 12
            Degrees-of-freedom for b0-T1w registration
        fmap_bspline : bool
            **Experimental**: Fit B-Spline field using least-squares
        fmap_demean : bool
            Demean voxel-shift map during unwarp
        use_syn : bool
            **Experimental**: Enable ANTs SyN-based susceptibility distortion
            correction (SDC). If fieldmaps are present and enabled, this is not
            run, by default.
        force_syn : bool
            **Temporary**: Always run SyN-based SDC
    """
    qsiprep_wf = Workflow(name='qsiprep_wf')
    qsiprep_wf.base_dir = work_dir

    if freesurfer:
        fsdir = pe.Node(
            BIDSFreeSurferDir(
                derivatives=output_dir,
                freesurfer_home=os.getenv('FREESURFER_HOME'),
                spaces=output_spaces),
            name='fsdir',
            run_without_submitting=True)

    reportlets_dir = os.path.join(work_dir, 'reportlets')
    for subject_id in subject_list:
        single_subject_wf = init_single_subject_wf(
            subject_id=subject_id,
            name="single_subject_" + subject_id + "_wf",
            reportlets_dir=reportlets_dir,
            output_dir=output_dir,
            bids_dir=bids_dir,
            ignore=ignore,
            debug=debug,
            low_mem=low_mem,
            anat_only=anat_only,
            longitudinal=longitudinal,
            freesurfer=freesurfer,
            hires=hires,
            combine_all_dwis=combine_all_dwis,
            discard_repeated_samples=discard_repeated_samples,
            omp_nthreads=omp_nthreads,
            skull_strip_template=skull_strip_template,
            skull_strip_fixed_seed=skull_strip_fixed_seed,
            output_spaces=output_spaces,
            template=template,
            b0_motion_corr_to=b0_motion_corr_to,
            b0_to_t1w_dof=b0_to_t1w_dof,
            fmap_bspline=fmap_bspline,
            fmap_demean=fmap_demean,
            use_syn=use_syn,
            force_syn=force_syn)

        single_subject_wf.config['execution']['crashdump_dir'] = (os.path.join(
            output_dir, "qsiprep", "sub-" + subject_id, 'log', run_uuid))
        for node in single_subject_wf._get_all_nodes():
            node.config = deepcopy(single_subject_wf.config)
        if freesurfer:
            qsiprep_wf.connect(fsdir, 'subjects_dir', single_subject_wf,
                               'inputnode.subjects_dir')
        else:
            qsiprep_wf.add_nodes([single_subject_wf])

    return qsiprep_wf


def init_single_subject_wf(
        subject_id, name, reportlets_dir, output_dir, bids_dir, ignore, debug,
        low_mem, anat_only, longitudinal, denoise_before_combining, dwi_denoise_window,
        combine_all_dwis, discard_repeated_samples, omp_nthreads, skull_strip_template,
        skull_strip_fixed_seed, freesurfer, hires, output_spaces, template,
        b0_motion_corr_to, b0_to_t1w_dof, fmap_bspline, fmap_demean, use_syn,
        force_syn):
    """
    This workflow organizes the preprocessing pipeline for a single subject.
    It collects and reports information about the subject, and prepares
    sub-workflows to perform anatomical and diffusion preprocessing.

    Anatomical preprocessing is performed in a single workflow, regardless of
    the number of sessions.
    Diffusion preprocessing is performed using a separate workflow for each
    session's dwi series.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.base import init_single_subject_wf
        wf = init_single_subject_wf(subject_id='test',
                                    task_id='',
                                    name='single_subject_wf',
                                    reportlets_dir='.',
                                    output_dir='.',
                                    bids_dir='.',
                                    ignore=[],
                                    debug=False,
                                    low_mem=False,
                                    anat_only=False,
                                    longitudinal=False,
                                    freesurfer=False,
                                    hires=False,
                                    dwi_denoise_window=7,
                                    denoise_before_combining=True,
                                    combine_all_dwis=True,
                                    discard_repeated_samples=True,
                                    omp_nthreads=1,
                                    skull_strip_template='OASIS',
                                    skull_strip_fixed_seed=False,
                                    template='MNI152NLin2009cAsym',
                                    output_spaces=['T1w', 'template'],
                                    b0_motion_corr_to='iterative',
                                    b0_to_t1w_dof=9,
                                    fmap_bspline=False,
                                    fmap_demean=True,
                                    use_syn=True,
                                    force_syn=True)

    Parameters

        subject_id : str
            List of subject labels
        task_id : str or None
            Task ID of BOLD series to preprocess, or ``None`` to preprocess all
        name : str
            Name of workflow
        ignore : list
            Preprocessing steps to skip (may include "sbref", "fieldmaps")
        debug : bool
            Enable debugging outputs
        low_mem : bool
            Write uncompressed .nii files in some cases to reduce memory usage
        anat_only : bool
            Disable functional workflows
        longitudinal : bool
            Treat multiple sessions as longitudinal (may increase runtime)
            See sub-workflows for specific differences
        dwi_denoise_window : int
            window size in voxels for ``dwidenoise``. Must be odd. If 0, '
            '``dwidwenoise`` will not be run'
        denoise_before_combining : bool
            'run ``dwidenoise`` before combining dwis. Requires ``combine_all_dwis``'
        combine_all_dwis : Bool
            Combine all dwi sequences within a session into a single data set
        discard_repeated_samples : Bool
            Ignore images if their q space coordinate has already been sampled
        omp_nthreads : int
            Maximum number of threads an individual process may use
        skull_strip_template : str
            Name of ANTs skull-stripping template ('OASIS' or 'NKI')
        skull_strip_fixed_seed : bool
            Do not use a random seed for skull-stripping - will ensure
            run-to-run replicability when used with --omp-nthreads 1
        freesurfer : bool
            Enable FreeSurfer surface reconstruction (may increase runtime)
        hires : bool
            Enable sub-millimeter preprocessing in FreeSurfer
        reportlets_dir : str
            Directory in which to save reportlets
        output_dir : str
            Directory in which to save derivatives
        bids_dir : str
            Root directory of BIDS dataset
        output_spaces : list
            List of output spaces functional images are to be resampled to.
            Some parts of pipeline will only be instantiated for some output
            spaces.

            Valid spaces:

             - T1w
             - template

        template : str
            Name of template targeted by ``template`` output space
        b0_motion_corr_to : str
            Motion correct using the 'first' b0 image or use an 'iterative'
            method to motion correct to the midpoint of the b0 images
        b0_to_t1w_dof : 6, 9 or 12
            Degrees-of-freedom for b0-T1w registration
        fmap_bspline : bool
            **Experimental**: Fit B-Spline field using least-squares
        fmap_demean : bool
            Demean voxel-shift map during unwarp
        use_syn : bool
            **Experimental**: Enable ANTs SyN-based susceptibility distortion
            correction (SDC). If fieldmaps are present and enabled, this is not
            run, by default.
        force_syn : bool
            **Temporary**: Always run SyN-based SDC


    Inputs

        subjects_dir
            FreeSurfer SUBJECTS_DIR

    """
    if name in ('single_subject_wf', 'single_subject_qsipreptest_wf'):
        # for documentation purposes
        subject_data = {
            't1w': ['/completely/made/up/path/sub-01_T1w.nii.gz'],
            'dwi': ['/completely/made/up/path/sub-01_dwi.nii.gz']
        }
        layout = None
    else:
        subject_data, layout = collect_data(bids_dir, subject_id)

    # Make sure we always go through these two checks
    if not anat_only and subject_data['dwi'] == []:
        raise Exception("No dwi images found for participant {}. "
                        "All workflows require dwi images.".format(subject_id))

    if not subject_data['t1w']:
        raise Exception("No T1w images found for participant {}. "
                        "All workflows require T1w images.".format(subject_id))

    workflow = Workflow(name=name)
    workflow.__desc__ = """
Results included in this manuscript come from preprocessing
performed using *QSIprep* {qsiprep_ver}
(@qsiprep1; @qsiprep2; RRID:SCR_016216),
which is based on *Nipype* {nipype_ver}
(@nipype1; @nipype2; RRID:SCR_002502).

""".format(
        qsiprep_ver=__version__, nipype_ver=nipype_ver)
    workflow.__postdesc__ = """

Many internal operations of *qsiprep* use
*Nilearn* {nilearn_ver} [@nilearn, RRID:SCR_001362],
mostly within the functional processing workflow.
For more details of the pipeline, see [the section corresponding
to workflows in *qsiprep*'s documentation]\
(https://qsiprep.readthedocs.io/en/latest/workflows.html \
"qsiprep's documentation").


### References

""".format(nilearn_ver=nilearn_ver)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['subjects_dir']), name='inputnode')

    bidssrc = pe.Node(
        BIDSDataGrabber(subject_data=subject_data, anat_only=anat_only),
        name='bidssrc')

    bids_info = pe.Node(
        BIDSInfo(), name='bids_info', run_without_submitting=True)

    summary = pe.Node(
        SubjectSummary(output_spaces=output_spaces, template=template),
        name='summary',
        run_without_submitting=True)

    about = pe.Node(
        AboutSummary(version=__version__, command=' '.join(sys.argv)),
        name='about',
        run_without_submitting=True)

    ds_report_summary = pe.Node(
        DerivativesDataSink(base_directory=reportlets_dir, suffix='summary'),
        name='ds_report_summary',
        run_without_submitting=True)

    ds_report_about = pe.Node(
        DerivativesDataSink(base_directory=reportlets_dir, suffix='about'),
        name='ds_report_about',
        run_without_submitting=True)

    # Preprocessing of T1w (includes registration to MNI)
    anat_preproc_wf = init_anat_preproc_wf(
        name="anat_preproc_wf",
        skull_strip_template=skull_strip_template,
        skull_strip_fixed_seed=skull_strip_fixed_seed,
        output_spaces=output_spaces,
        template=template,
        debug=debug,
        longitudinal=longitudinal,
        omp_nthreads=omp_nthreads,
        freesurfer=freesurfer,
        hires=hires,
        reportlets_dir=reportlets_dir,
        output_dir=output_dir,
        num_t1w=len(subject_data['t1w']))

    workflow.connect([
        (inputnode, anat_preproc_wf, [('subjects_dir',
                                       'inputnode.subjects_dir')]),
        (bidssrc, bids_info, [(('t1w', fix_multi_T1w_source_name),
                               'in_file')]),
        (inputnode, summary, [('subjects_dir', 'subjects_dir')]),
        (bidssrc, summary, [('t1w', 't1w'), ('t2w', 't2w'), ('bold', 'bold')]),
        (bids_info, summary, [('subject_id', 'subject_id')]),
        (bidssrc, anat_preproc_wf, [('t1w', 'inputnode.t1w'),
                                    ('t2w', 'inputnode.t2w'),
                                    ('roi', 'inputnode.roi'),
                                    ('flair', 'inputnode.flair')]),
        (summary, anat_preproc_wf, [('subject_id', 'inputnode.subject_id')]),
        (bidssrc, ds_report_summary, [(('t1w', fix_multi_T1w_source_name),
                                       'source_file')]),
        (summary, ds_report_summary, [('out_report', 'in_file')]),
        (bidssrc, ds_report_about, [(('t1w', fix_multi_T1w_source_name),
                                     'source_file')]),
        (about, ds_report_about, [('out_report', 'in_file')]),
    ])

    if anat_only:
        return workflow

    # Handle the grouping of multiple dwi files within a session
    sessions = layout.get_sessions()
    all_dwis = subject_data['dwi']
    dwi_groups = []
    if sessions:
        for session in sessions:
            dwi_groups.append([img for img in all_dwis if 'ses-'+session in img])
    else:
        dwi_groups = [all_dwis]

    for dwi_files in dwi_groups:
        dwi_preproc_wf = init_dwi_preproc_wf(
            dwi_files=dwi_files,
            layout=layout,
            ignore=ignore,
            freesurfer=freesurfer,
            dwi_denoise_window=dwi_denoise_window,
            denoise_before_combining=denoise_before_combining,
            b0_motion_corr_to=b0_motion_corr_to,
            b0_to_t1w_dof=b0_to_t1w_dof,
            reportlets_dir=reportlets_dir,
            output_spaces=output_spaces,
            template=template,
            output_dir=output_dir,
            omp_nthreads=omp_nthreads,
            low_mem=low_mem,
            fmap_bspline=fmap_bspline,
            fmap_demean=fmap_demean,
            use_syn=use_syn,
            force_syn=force_syn,
            num_dwi=len(all_dwis))

        workflow.connect([
            (
                anat_preproc_wf,
                dwi_preproc_wf,
                [
                    ('outputnode.t1_preproc', 'inputnode.t1_preproc'),
                    ('outputnode.t1_brain', 'inputnode.t1_brain'),
                    ('outputnode.t1_mask', 'inputnode.t1_mask'),
                    ('outputnode.t1_seg', 'inputnode.t1_seg'),
                    ('outputnode.t1_aseg', 'inputnode.t1_aseg'),
                    ('outputnode.t1_aparc', 'inputnode.t1_aparc'),
                    ('outputnode.t1_tpms', 'inputnode.t1_tpms'),
                    ('outputnode.t1_2_mni_forward_transform',
                     'inputnode.t1_2_mni_forward_transform'),
                    ('outputnode.t1_2_mni_reverse_transform',
                     'inputnode.t1_2_mni_reverse_transform'),
                    # Undefined if --no-freesurfer, but this is safe
                    ('outputnode.subjects_dir', 'inputnode.subjects_dir'),
                    ('outputnode.subject_id', 'inputnode.subject_id'),
                    ('outputnode.t1_2_fsnative_forward_transform',
                     'inputnode.t1_2_fsnative_forward_transform'),
                    ('outputnode.t1_2_fsnative_reverse_transform',
                     'inputnode.t1_2_fsnative_reverse_transform')
                ]),
        ])

    return workflow
