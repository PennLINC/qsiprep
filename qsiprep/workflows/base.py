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

import sys
from collections import defaultdict
from copy import deepcopy

from nilearn import __version__ as nilearn_ver
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from packaging.version import Version

from .. import config
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

    for subject_id, session_ids in config.execution.processing_list:
        # We may need to select a session or multiple sessions to consider together
        single_subject_wf = init_single_subject_wf(subject_id, session_ids)

        # Should we put these in session specific directories? The uuid is unique but opaque
        single_subject_wf.config['execution']['crashdump_dir'] = str(
            config.execution.output_dir / f'sub-{subject_id}' / 'log' / config.execution.run_uuid
        )
        for node in single_subject_wf._get_all_nodes():
            node.config = deepcopy(single_subject_wf.config)
        qsiprep_wf.add_nodes([single_subject_wf])

        # Dump a copy of the config file into the log directory
        log_dir = (
            config.execution.output_dir / f'sub-{subject_id}' / 'log' / config.execution.run_uuid
        )
        log_dir.mkdir(exist_ok=True, parents=True)
        config.to_filename(log_dir / 'qsiprep.toml')
    return qsiprep_wf


def init_single_subject_wf(subject_id: str, session_ids: list):
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
    subject_id : :obj:`str`
        Single subject label

    """
    if subject_id == 'qsiprepXtest':
        # for documentation purposes
        subject_data = {
            't1w': ['/completely/made/up/path/sub-01_T1w.nii.gz'],
            'dwi': ['/completely/made/up/path/sub-01_dwi.nii.gz'],
            't2w': ['/completely/made/up/path/sub-01_T2w.nii.gz'],
            'roi': [],
        }
        config.loggers.workflow.warning('Building a test workflow')
    else:
        subject_data = collect_data(
            config.execution.layout,
            subject_id,
            session_id=session_ids,
            filters=config.execution.bids_filters,
            bids_validate=False,
        )[0]

    # Make sure we always go through these two checks
    if not config.workflow.anat_only and subject_data['dwi'] == []:
        raise Exception(
            f'No dwi images found for participant {subject_id}. '
            'All workflows require dwi images unless '
            '--anat-only is specified.'
        )

    if not config.workflow.anat_modality == 'none' and not subject_data.get(
        config.workflow.anat_modality.lower()
    ):
        raise Exception(
            f'No {config.workflow.anat_modality} images found for participant {subject_id}. '
            'To bypass anatomical processing choose '
            '--anat-modality none'
        )

    anatomical_template = config.workflow.anatomical_template
    if config.workflow.infant:
        from ..utils.bids import cohort_by_months, parse_bids_for_age_months

        if session_ids and len(session_ids) > 1:
            raise RuntimeError('Infant template is only available for single session processing.')

        # Calculate the age and age-specific spaces
        session_id = None if not session_ids else session_ids[0]
        age = parse_bids_for_age_months(
            config.execution.bids_dir,
            subject_id,
            session_id,
        )
        if age is None:
            ses_str = f'_ses-{session_id}' if session_id else ''
            raise RuntimeError(f'Could not find age for sub-{subject_id}{ses_str}')

        cohort = cohort_by_months(anatomical_template, age)
        anatomical_template = f'{anatomical_template}+{cohort}'

    additional_t2ws = 0
    if 'drbuddi' in config.workflow.pepolar_method.lower() and subject_data['t2w']:
        additional_t2ws = len(subject_data['t2w'])

    # Inspect the dwi data and provide advice on pipeline choices
    # provide_processing_advice(subject_data, layout, unringing_method)

    _ses_name = '_ses_' + '_'.join(map(str, session_ids)) if session_ids else ''
    workflow = Workflow(name=f'sub_{subject_id}{_ses_name}_wf')
    workflow.__desc__ = f"""
Preprocessing was performed using *QSIPrep* {config.environment.version} [@cieslak2021qsiprep],
which is based on *Nipype* {config.environment.nipype_version}
[@nipype1; @nipype2; RRID:SCR_002502].

"""
    workflow.__postdesc__ = f"""

Many internal operations of *QSIPrep* use
*Nilearn* {nilearn_ver} [@nilearn, RRID:SCR_001362] and
*Dipy* [@dipy].
For more details of the pipeline, see [the section corresponding
to workflows in *QSIPrep*'s documentation]\
(https://qsiprep.readthedocs.io/en/latest/workflows.html \
"QSIPrep's documentation").


### References

"""

    merging_distortion_groups = not config.workflow.distortion_group_merge.lower() == 'none'

    inputnode = pe.Node(niu.IdentityInterface(fields=['subjects_dir']), name='inputnode')

    bidssrc = pe.Node(
        BIDSDataGrabber(
            subject_data=subject_data,  # Data has already been selected with sub/ses filters
            dwi_only=config.workflow.anat_modality == 'none',
            anat_only=config.workflow.anat_only,
            anatomical_contrast=config.workflow.anat_modality,
        ),
        name='bidssrc',
    )

    bids_info = pe.Node(BIDSInfo(), name='bids_info', run_without_submitting=True)

    summary = pe.Node(
        SubjectSummary(template=anatomical_template),
        name='summary',
        run_without_submitting=True,
    )

    about = pe.Node(
        AboutSummary(version=config.environment.version, command=' '.join(sys.argv)),
        name='about',
        run_without_submitting=True,
    )

    ds_report_summary = pe.Node(
        DerivativesDataSink(
            base_directory=config.execution.output_dir,
            datatype='figures',
            suffix='summary',
        ),
        name='ds_report_summary',
        run_without_submitting=True,
    )

    ds_report_about = pe.Node(
        DerivativesDataSink(
            base_directory=config.execution.output_dir,
            datatype='figures',
            suffix='about',
        ),
        name='ds_report_about',
        run_without_submitting=True,
    )

    num_anat_images = (
        0
        if config.workflow.anat_modality == 'none'
        else len(subject_data[config.workflow.anat_modality.lower()])
    )
    # Preprocessing of anatomical data (includes possible registration template)
    info_modality = (
        'dwi' if config.workflow.anat_modality == 'none' else config.workflow.anat_modality.lower()
    )
    anat_preproc_wf = init_anat_preproc_wf(
        num_anat_images=num_anat_images,
        num_additional_t2ws=additional_t2ws,
        has_rois=bool(subject_data['roi']),
        anatomical_template=anatomical_template,
    )

    workflow.connect([
        (inputnode, anat_preproc_wf, [('subjects_dir', 'inputnode.subjects_dir')]),
        (bidssrc, bids_info, [
            ((info_modality,
              fix_multi_source_name,
              config.workflow.anat_modality == 'none',
              config.workflow.subject_anatomical_reference == 'sessionwise',
              config.workflow.anat_modality),
             'in_file'),
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
            ((info_modality,
              fix_multi_source_name,
              config.workflow.anat_modality == 'none',
              config.workflow.subject_anatomical_reference == 'sessionwise',
              config.workflow.anat_modality),
             'source_file'),
        ]),
        (summary, ds_report_summary, [('out_report', 'in_file')]),
        (bidssrc, ds_report_about, [
            ((info_modality,
              fix_multi_source_name,
              config.workflow.anat_modality == 'none',
              config.workflow.subject_anatomical_reference == 'sessionwise',
              config.workflow.anat_modality),
             'source_file'),
        ]),
        (about, ds_report_about, [('out_report', 'in_file')]),
    ])  # fmt:skip

    if config.workflow.anat_only:
        return workflow

    # Handle the grouping of multiple dwi files within a session
    # concatenation_scheme maps the outputs to their final concatenation group
    dwi_fmap_groups, concatenation_scheme = group_dwi_scans(
        subject_data,
        using_fsl=True,
        combine_scans=not config.workflow.separate_all_dwis,
        ignore_fieldmaps='fieldmaps' in config.workflow.ignore,
        concatenate_distortion_groups=merging_distortion_groups,
    )
    config.loggers.workflow.info(dwi_fmap_groups)

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
                merging_strategy=config.workflow.distortion_group_merge,
                source_file=merged_group + '_dwi.nii.gz',
                inputs_list=merged_to_subgroups[merged_group],
                output_prefix=merged_group,
                name=merged_group.replace('-', '_') + '_final_merge_wf',
            )

            workflow.connect([
                (anat_preproc_wf, merging_group_workflows[merged_group], [
                    ('outputnode.t1_brain', 'inputnode.t1_brain'),
                    ('outputnode.t1_seg', 'inputnode.t1_seg'),
                    ('outputnode.t1_mask', 'inputnode.t1_mask'),
                ]),
            ])  # fmt:skip

    outputs_to_files = {
        dwi_group['concatenated_bids_name']: dwi_group for dwi_group in dwi_fmap_groups
    }
    if config.workflow.force_syn:
        for group_name in outputs_to_files:
            outputs_to_files[group_name]['fieldmap_info'] = {'suffix': 'syn'}
    summary.inputs.dwi_groupings = outputs_to_files

    make_intramodal_template = False
    if config.workflow.intramodal_template_iters > 0:
        if len(outputs_to_files) < 2:
            raise Exception('Cannot make an intramodal with less than 2 groups.')
        make_intramodal_template = True

    intramodal_template_wf = init_intramodal_template_wf(
        t1w_source_file=fix_multi_source_name(
            subject_data[info_modality],
            dwi_only=config.workflow.anat_modality == 'none',
            include_session=config.workflow.subject_anatomical_reference == 'sessionwise',
            anatomical_contrast=config.workflow.anat_modality,
        ),
        inputs_list=sorted(outputs_to_files.keys()),
        name='intramodal_template_wf',
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
        source_file = get_source_file(dwi_info['dwi_series'], output_fname, suffix='_dwi')
        output_wfname = output_fname.replace('-', '_')
        dwi_preproc_wf = init_dwi_preproc_wf(
            scan_groups=dwi_info,
            output_prefix=output_fname,
            source_file=source_file,
            t2w_sdc=bool(subject_data.get('t2w')),
            anatomical_template=anatomical_template,
        )
        dwi_finalize_wf = init_dwi_finalize_wf(
            scan_groups=dwi_info,
            name=dwi_preproc_wf.name.replace('dwi_preproc', 'dwi_finalize'),
            output_prefix=output_fname,
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
            input_name = f'inputnode.{output_wfname}_b0_template'
            output_name = f'outputnode.{output_wfname}_transform'
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
            image_name = f'inputnode.{output_wfname}_image'
            bval_name = f'inputnode.{output_wfname}_bval'
            bvec_name = f'inputnode.{output_wfname}_bvec'
            original_bvec_name = f'inputnode.{output_wfname}_original_bvec'
            original_bids_name = f'inputnode.{output_wfname}_original_image'
            raw_concatenated_image_name = f'inputnode.{output_wfname}_raw_concatenated_image'
            confounds_name = f'inputnode.{output_wfname}_confounds'
            b0_ref_name = f'inputnode.{output_wfname}_b0_ref'
            cnr_name = f'inputnode.{output_wfname}_cnr'
            carpetplot_name = f'inputnode.{output_wfname}_carpetplot_data'
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
                ]),
            ])  # fmt:skip

    for node in workflow.list_node_names():
        node_name = node.split('.')[-1]
        if node_name.startswith('ds_'):
            workflow.get_node(node).interface.out_path_base = ''
            workflow.get_node(node).interface.inputs.base_directory = config.execution.output_dir

        if node_name.startswith('ds_report_'):
            workflow.get_node(node).interface.inputs.datatype = 'figures'

    return workflow


def provide_processing_advice(subject_data, layout, unringing_method):
    """Provide advice on preprocessing options based on the data provided."""
    # metadata = {dwi_file: layout.get_metadata(dwi_file) for dwi_file in subject_data["dwi"]}
    config.loggers.utils.warning(
        'Partial Fourier acquisitions found for %s. Consider using --unringing-method rpg'
    )
