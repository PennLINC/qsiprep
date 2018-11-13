from qsiprep.utils.bids import collect_data
from collections import defaultdict
from fmriprep.engine import Workflow
import nipype.interfaces.utility as niu
from nipype.pipeline import engine as pe
from qsiprep.interfaces import MergeDWIs
from qsiprep.workflows.dwi.b0_alignment import (init_b0_alignment_wf,
                                                init_b0_to_anat_registration_wf)
from qsiprep.workflows.dwi.merge import init_merge_and_denoise_wf
from qsiprep.workflows.anatomical import init_anat_preproc_wf
from qsiprep.workflows.fieldmap.bidirectional_pepolar import init_bidirectional_b0_unwarping_wf
workflow = pe.Workflow(name="test_merge")

subject_data, layout = collect_data('/Users/mcieslak/projects/test_bids_data/pepolar', '1')
# Handle the grouping of multiple dwi files within a session
sessions = layout.get_sessions()
all_dwis = subject_data['dwi']
dwi_groups = []
if sessions:
    for session in sessions:
        dwi_groups.append([img for img in all_dwis if 'ses-'+session in img])
else:
    dwi_groups = [all_dwis]

dwi_files = dwi_groups[0]
ignore = []
combine_all_dwis = True


# Create a list of (dwi image, PE dir, fieldmaps)
parsed_dwis = []
for ref_file in dwi_files:
    # Find associated sbref, if possible
    entities = layout.parse_file_entities(ref_file)
    entities['type'] = 'sbref'
    files = layout.get(**entities, extensions=['nii', 'nii.gz'])
    if len(files):
        print("Ignoring sbref for %s", ref_file)
    metadata = layout.get_metadata(ref_file)

    # Find fieldmaps. Options: (epi|syn)
    fmaps = []
    if 'fieldmaps' not in ignore:
        fmaps = layout.get_fieldmap(ref_file, return_list=True)
        fmap_files = []
        for fmap in fmaps:
            if fmap['type'] == 'epi':
                fmap_files.append(fmap['epi'])
        if len(fmap_files) > 1:
            print("Multiple field maps found for %s", ref_file)
        fmap_file = fmap_files[0]
    parsed_dwis.append(
        (ref_file, metadata['PhaseEncodingDirection'], fmap_file))

# Depending on how the scans are organized, either use the fieldmap or the RPE series
scans_and_maps = []
if combine_all_dwis:
    groups = defaultdict(list)
    for ref_file, pe_dir, fmap in parsed_dwis:
        groups[(pe_dir, fmap)].append(ref_file)
    if len(groups) > 2:
        print("Unable to match fieldmaps to DWIs.")
    for k, v in groups.items():
        scans_and_maps.append((v, k))

    if len(groups) == 2:
        print("Using b0 templates from opposite PEs to unwarp instead of fieldmaps")
        pe_keys = sorted(list(groups.keys()))
        pe0_scans = groups[pe_keys[0]]
        pe1_scans = groups[pe_keys[1]]
        merge0 = init_merge_and_denoise_wf(dwi_files=pe0_scans, dwi_denoise_window=7,
                                           denoise_before_combining=True, combine_all_dwis=True,
                                           name="merge0")
        align0 = init_b0_alignment_wf(name="align0")
        merge1 = init_merge_and_denoise_wf(dwi_files=pe1_scans, dwi_denoise_window=7,
                                           denoise_before_combining=True, combine_all_dwis=True,
                                           name="merge1")
        align1 = init_b0_alignment_wf(name="align1")

        workflow.connect([
            (merge0, align0, [('outputnode.merged_image', 'input_node.dwi_nifti'),
                              ('outputnode.merged_bval', 'input_node.bvals'),
                              ('outputnode.merged_bvec', 'input_node.bvecs')]),
            (merge1, align1, [('outputnode.merged_image', 'input_node.dwi_nifti'),
                              ('outputnode.merged_bval', 'input_node.bvals'),
                              ('outputnode.merged_bvec', 'input_node.bvecs')]),
        ])

        register_b0_refs = init_bidirectional_b0_unwarping_wf(template_plus_pe='j')
        workflow.connect([
            (align0, register_b0_refs, [('output_node.b0_template', 'inputnode.template_plus')]),
            (align1, register_b0_refs, [('output_node.b0_template', 'inputnode.template_minus')]),

        ])

# Do some anatomical
anat_wf = init_anat_preproc_wf('OASIS', ['T1w'], 'MNI152NLin2009cAsym', False,
                               False, False, 1, False,
                               '/Users/mcieslak/projects/qsiprep/scratch/test_merge/reports',
                               '/Users/mcieslak/projects/qsiprep/scratch/test_merge/anat_outputs',
                               1)
anat_node = pe.Node(niu.IdentityInterface(fields=["t1w", "subjects_dir"]),
                    name="anat_identity")
anat_node.inputs.t1w = subject_data['t1w']
anat_node.inputs.subjects_dir = '/Users/mcieslak/projects/qsiprep/scratch/freesurfer'
workflow.connect([(anat_node, anat_wf, [('t1w', 'inputnode.t1w'),
                                        ('subjects_dir', 'inputnode.subjects_dir')
                                        ])])

# Register the unwarped image to the t1 template
b0_coreg_wf = init_b0_to_anat_registration_wf()
workflow.connect([
    (anat_wf, b0_coreg_wf, [('outputnode.t1_brain', 'input_node.anat_image')]),
    (register_b0_refs, b0_coreg_wf, [('outputnode.out_reference', 'input_node.b0_image')])
])

# Create an output grid for the dwis
autobox_t1 = pe.Node(afni.Autobox(), name="autobox_t1")
autobox_t1.inputs.outputtype = "NIFTI_GZ"
autobox_t1.inputs.padding = 5
deoblique_autobox = pe.Node(afni.Warp(), name="deoblique_autobox")
deoblique_autobox.inputs.deoblique = True
deoblique_autobox.inputs.outputtype = "NIFTI_GZ"
resample_to_dwi = pe.Node(afni.Resample(), name="resample_to_dwi")
resample_to_dwi.inputs.voxel_size = (2.0, 2.0, 2.0)
resample_to_dwi.inputs.outputtype = "NIFTI_GZ"

workflow.connect([
    (anat_wf, autobox_t1, [('outputnode.t1_brain', 'in_file')]),
    (autobox_t1, deoblique_autobox, [('out_file', 'in_file')]),
    (deoblique_autobox, resample_to_dwi, [('out_file', 'in_file')])
])

# Put the to-anat and to-b0 transforms into a transform list
# there will be many transforms this time: motioncorr -> pe template -> unwarp -> anat
concat_image_transforms = pe.MapNode(
    util.Merge(4), name='concat_image_transforms', iterfield=['in4'])
mc_reg_wf.connect([(motion_corr_wf, concat_image_transforms, [
                    ('output_node.to_b0_affines', 'in4')]),
                  [()]
])

# Put the to-anat and to-b0 transforms into a transform list
concat_image_transforms = pe.MapNode(
    util.Merge(2), name="concat_image_transforms", iterfield=["in2"])
mc_reg_wf.connect(motion_corr_wf, "output_node.to_b0_affines",
                  concat_image_transforms, "in2")
mc_reg_wf.connect(coreg_wf, "output_node.b0_to_anat_transform",
                  concat_image_transforms, "in1")
# Reverse order for bvec transform
concat_bvec_transforms = pe.MapNode(
    util.Merge(2), name="concat_bvec_transforms", iterfield=["in1"])
mc_reg_wf.connect(motion_corr_wf, "output_node.to_b0_affines",
                  concat_bvec_transforms, "in1")
mc_reg_wf.connect(coreg_wf, "output_node.b0_to_anat_transform",
                  concat_bvec_transforms, "in2")


workflow.base_dir = "/Users/mcieslak/projects/qsiprep/scratch"
workflow.config['execution']['stop_on_first_crash'] = 'true'
workflow.run()
