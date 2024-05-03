#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright The NiPreps Developers <nipreps@gmail.com>
#
# Changed to run qsiprep workflows
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""
qsiprep base processing workflows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_qsiprep_wf
.. autofunction:: init_single_subject_wf

"""
import logging
import sys
from collections import defaultdict
from copy import deepcopy

from nilearn import __version__ as nilearn_ver
from nipype import __version__ as nipype_ver
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from packaging.version import Version

from .. import config
from ..engine import Workflow
from ..interfaces import (
    AboutSummary,
    BIDSDataGrabber,
    BIDSInfo,
    DerivativesDataSink,
    SubjectSummary,
)
from ..utils.bids import collect_data
from ..utils.grouping import group_dwi_scans
from ..utils.misc import fix_multi_source_name
from .anatomical.volume import init_anat_preproc_wf
from .dwi.base import init_dwi_preproc_wf
from .dwi.distortion_group_merge import init_distortion_group_merge_wf
from .dwi.finalize import init_dwi_finalize_wf
from .dwi.intramodal_template import init_intramodal_template_wf
from .dwi.util import get_source_file

LOGGER = logging.getLogger("nipype.workflow")


def init_qsiprep_wf():
    """Organize the execution of qsiprep, with a sub-workflow for each subject.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        import os
        os.environ['FREESURFER_HOME'] = os.getcwd()
        from qsiprep.workflows.base import init_qsiprep_wf

        wf = init_qsiprep_wf()
    """
    ver = Version(config.environment.version)
    qsiprep_wf = Workflow(name=f'qsiprep_{ver.major}_{ver.minor}_wf')
    qsiprep_wf.base_dir = config.execution.work_dir

    for subject_id in config.execution.participant_label:
        single_subject_wf = init_single_subject_wf(subject_id)

        single_subject_wf.config["execution"]["crashdump_dir"] = str(
            config.execution.qsiprep_dir / f"sub-{subject_id}" / "log", config.execution.run_uuid
        )
        for node in single_subject_wf._get_all_nodes():
            node.config = deepcopy(single_subject_wf.config)
        qsiprep_wf.add_nodes([single_subject_wf])

        # Dump a copy of the config file into the log directory
        log_dir = (
            config.execution.qsiprep_dir / f'sub-{subject_id}' / 'log' / config.execution.run_uuid
        )
        log_dir.mkdir(exist_ok=True, parents=True)
        config.to_filename(log_dir / 'qsiprep.toml')
    return qsiprep_wf


def init_single_subject_wf(subject_id: str):
    """Organize the preprocessing pipeline for a single subject.

    This workflow collects and reports information about the subject, and prepares
    sub-workflows to perform anatomical and diffusion preprocessing.

    Anatomical preprocessing is performed in a single workflow, regardless of
    the number of sessions.
    Diffusion preprocessing is performed using a separate workflow for each
    session's dwi series.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.base import init_single_subject_wf

        wf = init_single_subject_wf('qsiprepXtest')

    Parameters
    ----------
    subject_id : str
        Single subject label
   
    Inputs
    ------
    subjects_dir
        FreeSurfer SUBJECTS_DIR
    """
    if subject_id == "qsiprepXtest":
        # for documentation purposes
        subject_data = {
            "t1w": ["/completely/made/up/path/sub-01_T1w.nii.gz"],
            "dwi": ["/completely/made/up/path/sub-01_dwi.nii.gz"],
            "t2w": ["/completely/made/up/path/sub-01_T2w.nii.gz"],
            "roi": [],
        }
        layout = None
        LOGGER.warning("Building a test workflow")
    else:
        subject_data = collect_data(
            config.execution.layout,
            subject_id,
            filters=config.execution.bids_filters,
            bids_validate=False,
        )[0]

    # Warn about --dwi-only and non-none --anat-modality
    if config.workflow.dwi_only and config.workflow.anatomical_contrast != "none":
        anatomical_contrast = "none"
    if anatomical_contrast == "none":
        dwi_only = True

    if anat_only and dwi_only:
        raise Exception("--anat-only and --dwi-only are mutually exclusive.")

    # Make sure we always go through these two checks
    if not anat_only and subject_data["dwi"] == []:
        raise Exception(
            "No dwi images found for participant {}. "
            "All workflows require dwi images unless "
            "--anat-only is specified.".format(subject_id)
        )

    if not dwi_only and not subject_data.get(anatomical_contrast.lower()):
        raise Exception(
            "No {} images found for participant {}. "
            "To bypass anatomical processing choose "
            "--anat-modality none".format(anatomical_contrast, subject_id)
        )

    additional_t2ws = 0
    if "drbuddi" in pepolar_method.lower() and subject_data["t2w"]:
        additional_t2ws = len(subject_data["t2w"])

    # Inspect the dwi data and provide advice on pipeline choices
    # provide_processing_advice(subject_data, layout, unringing_method)

    workflow = Workflow(name=name)
    workflow.__desc__ = """
Preprocessing was performed using *QSIPrep* {qsiprep_ver},
which is based on *Nipype* {nipype_ver}
(@nipype1; @nipype2; RRID:SCR_002502).

""".format(
        qsiprep_ver=__version__, nipype_ver=nipype_ver
    )
    workflow.__postdesc__ = """

Many internal operations of *QSIPrep* use
*Nilearn* {nilearn_ver} [@nilearn, RRID:SCR_001362] and
*Dipy* [@dipy].
For more details of the pipeline, see [the section corresponding
to workflows in *QSIPrep*'s documentation]\
(https://qsiprep.readthedocs.io/en/latest/workflows.html \
"QSIPrep's documentation").


### References

""".format(
        nilearn_ver=nilearn_ver
    )

    merging_distortion_groups = not distortion_group_merge.lower() == "none"

    inputnode = pe.Node(niu.IdentityInterface(fields=["subjects_dir"]), name="inputnode")

    bidssrc = pe.Node(
        BIDSDataGrabber(
            subject_data=subject_data,
            dwi_only=dwi_only,
            anat_only=anat_only,
            anatomical_contrast=anatomical_contrast,
        ),
        name="bidssrc",
    )

    bids_info = pe.Node(BIDSInfo(), name="bids_info", run_without_submitting=True)

    summary = pe.Node(
        SubjectSummary(template=template), name="summary", run_without_submitting=True
    )

    about = pe.Node(
        AboutSummary(version=__version__, command=" ".join(sys.argv)),
        name="about",
        run_without_submitting=True,
    )

    ds_report_summary = pe.Node(
        DerivativesDataSink(base_directory=reportlets_dir, suffix="summary"),
        name="ds_report_summary",
        run_without_submitting=True,
    )

    ds_report_about = pe.Node(
        DerivativesDataSink(base_directory=reportlets_dir, suffix="about"),
        name="ds_report_about",
        run_without_submitting=True,
    )

    num_anat_images = 0 if dwi_only else len(subject_data[anatomical_contrast.lower()])
    # Preprocessing of anatomical data (includes possible registration template)
    info_modality = "dwi" if dwi_only else anatomical_contrast.lower()
    anat_preproc_wf = init_anat_preproc_wf(
        template=template,
        debug=debug,
        dwi_only=dwi_only,
        infant_mode=infant_mode,
        longitudinal=longitudinal,
        omp_nthreads=omp_nthreads,
        output_dir=output_dir,
        num_anat_images=num_anat_images,
        output_resolution=output_resolution,
        nonlinear_register_to_template=force_spatial_normalization,
        reportlets_dir=reportlets_dir,
        anatomical_contrast=anatomical_contrast,
        num_additional_t2ws=additional_t2ws,
        has_rois=bool(subject_data["roi"]),
        name="anat_preproc_wf",
    )

    workflow.connect([
        (inputnode, anat_preproc_wf, [('subjects_dir', 'inputnode.subjects_dir')]),
        (bidssrc, bids_info, [
            ((info_modality, fix_multi_source_name, dwi_only, anatomical_contrast), 'in_file'),
        ]),
        (inputnode, summary, [('subjects_dir', 'subjects_dir')]),
        (bidssrc, summary, [('t1w', 't1w'), ('t2w', 't2w')]),
        (bids_info, summary, [('subject_id', 'subject_id')]),
        (bidssrc, anat_preproc_wf, [
            ('t1w', 'inputnode.t1w'),
            ('t2w', 'inputnode.t2w'),
            ('roi', 'inputnode.roi'),
            ('flair', 'inputnode.flair'),
        ]),
        (summary, anat_preproc_wf, [('subject_id', 'inputnode.subject_id')]),
        (bidssrc, ds_report_summary, [
            ((info_modality, fix_multi_source_name, dwi_only, anatomical_contrast), 'source_file'),
        ]),
        (summary, ds_report_summary, [('out_report', 'in_file')]),
        (bidssrc, ds_report_about, [
            ((info_modality, fix_multi_source_name, dwi_only, anatomical_contrast), 'source_file'),
        ]),
        (about, ds_report_about, [('out_report', 'in_file')])
    ])  # fmt:skip

    if anat_only:
        return workflow

    if impute_slice_threshold > 0 and hmc_model == "none":
        LOGGER.warning(
            "hmc_model must not be 'none' if slices are to be imputed. "
            "setting `impute_slice_threshold=0`"
        )
        impute_slice_threshold = 0

    # Handle the grouping of multiple dwi files within a session
    # concatenation_scheme maps the outputs to their final concatenation group
    dwi_fmap_groups, concatenation_scheme = group_dwi_scans(
        layout,
        subject_data,
        using_fsl=True,
        combine_scans=combine_all_dwis,
        ignore_fieldmaps="fieldmaps" in ignore,
        concatenate_distortion_groups=merging_distortion_groups,
    )
    LOGGER.info(dwi_fmap_groups)

    # If a merge is happening at the end, make sure
    if merging_distortion_groups:
        # create a mapping of which across-distortion-groups are contained in each merge
        merged_group_names = sorted(set(concatenation_scheme.values()))
        merged_to_subgroups = defaultdict(list)
        for subgroup_name, destination_name in concatenation_scheme.items():
            merged_to_subgroups[destination_name].append(subgroup_name)

        merging_group_workflows = {}
        for merged_group in merged_group_names:
            merging_group_workflows[merged_group] = init_distortion_group_merge_wf(
                merging_strategy=distortion_group_merge,
                harmonize_b0_intensities=not no_b0_harmonization,
                b0_threshold=b0_threshold,
                template=template,
                output_dir=output_dir,
                output_prefix=merged_group,
                source_file=merged_group + "_dwi.nii.gz",
                shoreline_iters=shoreline_iters,
                hmc_model=hmc_model,
                inputs_list=merged_to_subgroups[merged_group],
                omp_nthreads=omp_nthreads,
                reportlets_dir=reportlets_dir,
                name=merged_group.replace("-", "_") + "_final_merge_wf",
            )

            workflow.connect([
                (anat_preproc_wf, merging_group_workflows[merged_group], [
                    ('outputnode.t1_brain', 'inputnode.t1_brain'),
                    ('outputnode.t1_seg', 'inputnode.t1_seg'),
                    ('outputnode.t1_mask', 'inputnode.t1_mask'),
                ]),
            ])  # fmt:skip

    outputs_to_files = {
        dwi_group["concatenated_bids_name"]: dwi_group for dwi_group in dwi_fmap_groups
    }
    if force_syn:
        for group_name in outputs_to_files:
            outputs_to_files[group_name]["fieldmap_info"] = {"suffix": "syn"}
    summary.inputs.dwi_groupings = outputs_to_files

    make_intramodal_template = False
    if intramodal_template_iters > 0:
        if len(outputs_to_files) < 2:
            raise Exception("Cannot make an intramodal with less than 2 groups.")
        make_intramodal_template = True

    intramodal_template_wf = init_intramodal_template_wf(
        omp_nthreads=omp_nthreads,
        t1w_source_file=fix_multi_source_name(subject_data[info_modality], dwi_only),
        reportlets_dir=reportlets_dir,
        num_iterations=intramodal_template_iters,
        transform=intramodal_template_transform,
        inputs_list=sorted(outputs_to_files.keys()),
        name="intramodal_template_wf",
    )

    if make_intramodal_template:
        workflow.connect([
            (anat_preproc_wf, intramodal_template_wf, [
                ('outputnode.t1_preproc', 'inputnode.t1_preproc'),
                ('outputnode.t1_brain', 'inputnode.t1_brain'),
                ('outputnode.t1_mask', 'inputnode.t1_mask'),
                ('outputnode.t1_seg', 'inputnode.t1_seg'),
                ('outputnode.t1_aseg', 'inputnode.t1_aseg'),
                ('outputnode.t1_aparc', 'inputnode.t1_aparc'),
                ('outputnode.t1_2_mni_forward_transform', 'inputnode.t1_2_mni_forward_transform'),
                ('outputnode.t1_2_mni_reverse_transform', 'inputnode.t1_2_mni_reverse_transform'),
                ('outputnode.dwi_sampling_grid', 'inputnode.dwi_sampling_grid'),
            ]),
        ])  # fmt:skip

    # create a processing pipeline for the dwis in each session
    for output_fname, dwi_info in outputs_to_files.items():
        source_file = get_source_file(dwi_info["dwi_series"], output_fname, suffix="_dwi")
        output_wfname = output_fname.replace("-", "_")
        dwi_preproc_wf = init_dwi_preproc_wf(
            scan_groups=dwi_info,
            dwi_only=dwi_only,
            output_prefix=output_fname,
            layout=layout,
            ignore=ignore,
            b0_threshold=b0_threshold,
            dwi_denoise_window=dwi_denoise_window,
            denoise_method=denoise_method,
            unringing_method=unringing_method,
            b1_biascorrect_stage=b1_biascorrect_stage,
            no_b0_harmonization=no_b0_harmonization,
            denoise_before_combining=denoise_before_combining,
            motion_corr_to=motion_corr_to,
            b0_to_t1w_transform=b0_to_t1w_transform,
            hmc_model=hmc_model,
            hmc_transform=hmc_transform,
            shoreline_iters=shoreline_iters,
            eddy_config=eddy_config,
            raw_image_sdc=raw_image_sdc,
            pepolar_method=pepolar_method,
            impute_slice_threshold=impute_slice_threshold,
            reportlets_dir=reportlets_dir,
            template=template,
            output_dir=output_dir,
            omp_nthreads=omp_nthreads,
            low_mem=low_mem,
            fmap_bspline=fmap_bspline,
            fmap_demean=fmap_demean,
            t2w_sdc=bool(subject_data.get("t2w")),
            sloppy=debug,
            source_file=source_file,
        )
        dwi_finalize_wf = init_dwi_finalize_wf(
            scan_groups=dwi_info,
            name=dwi_preproc_wf.name.replace("dwi_preproc", "dwi_finalize"),
            output_prefix=output_fname,
            layout=layout,
            hmc_model=hmc_model,
            shoreline_iters=shoreline_iters,
            write_local_bvecs=write_local_bvecs,
            reportlets_dir=reportlets_dir,
            template=template,
            output_resolution=output_resolution,
            output_dir=output_dir,
            omp_nthreads=omp_nthreads,
            do_biascorr=b1_biascorrect_stage == "final",
            b0_threshold=b0_threshold,
            make_intramodal_template=make_intramodal_template,
            source_file=source_file,
            write_derivatives=not merging_distortion_groups,
        )

        workflow.connect([
            (anat_preproc_wf, dwi_preproc_wf, [
                ('outputnode.t1_preproc', 'inputnode.t1_preproc'),
                ('outputnode.t1_brain', 'inputnode.t1_brain'),
                ('outputnode.t1_mask', 'inputnode.t1_mask'),
                ('outputnode.t1_seg', 'inputnode.t1_seg'),
                ('outputnode.t1_aseg', 'inputnode.t1_aseg'),
                ('outputnode.t1_aparc', 'inputnode.t1_aparc'),
                ('outputnode.t1_2_mni_forward_transform', 'inputnode.t1_2_mni_forward_transform'),
                ('outputnode.t1_2_mni_reverse_transform', 'inputnode.t1_2_mni_reverse_transform'),
                ('outputnode.dwi_sampling_grid', 'inputnode.dwi_sampling_grid'),
                ('outputnode.t2w_unfatsat', 'inputnode.t2w_unfatsat'),
            ]),
            (anat_preproc_wf, dwi_finalize_wf, [
                ('outputnode.t1_preproc', 'inputnode.t1_preproc'),
                ('outputnode.t1_brain', 'inputnode.t1_brain'),
                ('outputnode.t1_mask', 'inputnode.t1_mask'),
                ('outputnode.t1_seg', 'inputnode.t1_seg'),
                ('outputnode.t1_aseg', 'inputnode.t1_aseg'),
                ('outputnode.t1_aparc', 'inputnode.t1_aparc'),
                ('outputnode.t1_2_mni_forward_transform', 'inputnode.t1_2_mni_forward_transform'),
                ('outputnode.t1_2_mni_reverse_transform', 'inputnode.t1_2_mni_reverse_transform'),
                ('outputnode.dwi_sampling_grid', 'inputnode.dwi_sampling_grid'),
            ]),
            (dwi_preproc_wf, dwi_finalize_wf, [
                ('outputnode.dwi_files', 'inputnode.dwi_files'),
                ('outputnode.cnr_map', 'inputnode.cnr_map'),
                ('outputnode.bval_files', 'inputnode.bval_files'),
                ('outputnode.bvec_files', 'inputnode.bvec_files'),
                ('outputnode.b0_ref_image', 'inputnode.b0_ref_image'),
                ('outputnode.b0_indices', 'inputnode.b0_indices'),
                ('outputnode.hmc_xforms', 'inputnode.hmc_xforms'),
                ('outputnode.fieldwarps', 'inputnode.fieldwarps'),
                ('outputnode.itk_b0_to_t1', 'inputnode.itk_b0_to_t1'),
                ('outputnode.hmc_optimization_data', 'inputnode.hmc_optimization_data'),
                ('outputnode.raw_qc_file', 'inputnode.raw_qc_file'),
                ('outputnode.coreg_score', 'inputnode.coreg_score'),
                ('outputnode.raw_concatenated', 'inputnode.raw_concatenated'),
                ('outputnode.confounds', 'inputnode.confounds'),
                ('outputnode.carpetplot_data', 'inputnode.carpetplot_data'),
                ('outputnode.sdc_scaling_images', 'inputnode.sdc_scaling_images'),
            ]),
        ])  # fmt:skip

        if make_intramodal_template:
            input_name = "inputnode.{name}_b0_template".format(name=output_wfname)
            output_name = "outputnode.{name}_transform".format(name=output_wfname)
            workflow.connect([
                (dwi_preproc_wf, intramodal_template_wf, [
                    ('outputnode.b0_ref_image', input_name),
                ]),
                (intramodal_template_wf, dwi_finalize_wf, [
                    (output_name, 'inputnode.b0_to_intramodal_template_transforms'),
                    (
                        'outputnode.intramodal_template_to_t1_affine',
                        'inputnode.intramodal_template_to_t1_affine',
                    ),
                    (
                        'outputnode.intramodal_template_to_t1_warp',
                        'inputnode.intramodal_template_to_t1_warp',
                    ),
                    ('outputnode.intramodal_template', 'inputnode.intramodal_template'),
                ]),
            ])  # fmt:skip

        if merging_distortion_groups:
            image_name = "inputnode.{name}_image".format(name=output_wfname)
            bval_name = "inputnode.{name}_bval".format(name=output_wfname)
            bvec_name = "inputnode.{name}_bvec".format(name=output_wfname)
            original_bvec_name = "inputnode.{name}_original_bvec".format(name=output_wfname)
            original_bids_name = "inputnode.{name}_original_image".format(name=output_wfname)
            raw_concatenated_image_name = "inputnode.{name}_raw_concatenated_image".format(
                name=output_wfname
            )
            confounds_name = "inputnode.{name}_confounds".format(name=output_wfname)
            b0_ref_name = "inputnode.{name}_b0_ref".format(name=output_wfname)
            cnr_name = "inputnode.{name}_cnr".format(name=output_wfname)
            carpetplot_name = "inputnode.{name}_carpetplot_data".format(name=output_wfname)
            final_merge_wf = merging_group_workflows[concatenation_scheme[output_fname]]
            workflow.connect([
                (dwi_finalize_wf, final_merge_wf, [
                    ('outputnode.bvals_t1', bval_name),
                    ('outputnode.bvecs_t1', bvec_name),
                    ('outputnode.dwi_t1', image_name),
                    ('outputnode.t1_b0_ref', b0_ref_name),
                    ('outputnode.cnr_map_t1', cnr_name),
                ]),
                (dwi_preproc_wf, final_merge_wf, [
                    ('outputnode.raw_concatenated', raw_concatenated_image_name),
                    ('outputnode.original_bvecs', original_bvec_name),
                    ('outputnode.original_files', original_bids_name),
                    ('outputnode.carpetplot_data', carpetplot_name),
                    ('outputnode.confounds', confounds_name),
                ])
            ])  # fmt:skip

    return workflow


def provide_processing_advice(subject_data, layout, unringing_method):
    """Provide advice on preprocessing options based on the data provided."""
    # metadata = {dwi_file: layout.get_metadata(dwi_file) for dwi_file in subject_data["dwi"]}
    LOGGER.warn("Partial Fourier acquisitions found for %s. Consider using --unringing-method rpg")
