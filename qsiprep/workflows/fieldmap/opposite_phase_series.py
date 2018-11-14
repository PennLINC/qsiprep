from qsiprep.utils.bids import collect_data
from collections import defaultdict
from fmriprep.engine import Workflow
import nipype.interfaces.utility as niu
from nipype.interfaces import afni
from nipype.pipeline import engine as pe
from qsiprep.interfaces import MergeDWIs
from qsiprep.workflows.dwi.hmc import (init_hmc_wf,
                                                init_b0_to_anat_registration_wf)
from qsiprep.workflows.dwi.merge import init_merge_and_denoise_wf
from qsiprep.workflows.anatomical import init_anat_preproc_wf
from qsiprep.workflows.fieldmap.bidirectional_pepolar import init_bidirectional_b0_unwarping_wf
from qsiprep.interfaces.images import NiftiInfo

def init_opposite_phase_series_wf(template_plus_pe,
                                  dwi_denoise_window,
                                  output_spaces,
                                  denoise_before_combining,
                                  combine_all_dwis,
                                  name='opposite_phase_series_wf'):

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['pe_plus_dwis', 'pe_minus_dwis', 't1w_brain',
                              't1w_2_mni_affine']),
        name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=[]),
        name='outputnode')

    merge_plus = init_merge_and_denoise_wf(dwi_denoise_window=dwi_denoise_window,
                                           denoise_before_combining=denoise_before_combining,
                                           combine_all_dwis=combine_all_dwis,
                                           name="merge_plus")
    b0_hmc_plus = init_hmc_wf(name="b0_hmc_plus")
    merge_minus = init_merge_and_denoise_wf(dwi_denoise_window=dwi_denoise_window,
                                            denoise_before_combining=denoise_before_combining,
                                            combine_all_dwis=combine_all_dwis,
                                            name="merge_minus")
    b0_hmc_minus = init_hmc_wf(name="b0_hmc_minus")

    # Create unwarped template and get warps for plus and minus
    register_b0_refs = init_bidirectional_b0_unwarping_wf(template_plus_pe=template_plus_pe)

    # Register the unwarped image to the t1 template
    b0_coreg_wf = init_b0_to_anat_registration_wf()

    # Create an output grid for the dwis
    autobox_t1 = pe.Node(afni.Autobox(), name="autobox_t1")
    autobox_t1.inputs.outputtype = "NIFTI_GZ"
    autobox_t1.inputs.padding = 5
    deoblique_autobox = pe.Node(afni.Warp(), name="deoblique_autobox")
    deoblique_autobox.inputs.deoblique = True
    deoblique_autobox.inputs.outputtype = "NIFTI_GZ"
    resample_to_dwi = pe.Node(afni.Resample(), name="resample_to_dwi")
    resample_to_dwi.inputs.outputtype = "NIFTI_GZ"
    dwi_info = pe.Node(NiftiInfo(), name='dwi_info')

    workflow = Workflow(name=name)
    workflow.connect([
        (inputnode, merge_plus, [('pe_plus_dwis', 'dwi_files')]),
        (inputnode, merge_minus, [('pe_minus_dwis', 'dwi_files')]),
        (merge_plus, b0_hmc_plus, [('outputnode.merged_image', 'inputnode.dwi_nifti'),
                                   ('outputnode.merged_bval', 'inputnode.bvals'),
                                   ('outputnode.merged_bvec', 'inputnode.bvecs')]),
        (merge_minus, b0_hmc_minus, [('outputnode.merged_image', 'inputnode.dwi_nifti'),
                                     ('outputnode.merged_bval', 'inputnode.bvals'),
                                     ('outputnode.merged_bvec', 'inputnode.bvecs')]),
        (b0_hmc_plus, register_b0_refs, [('outputnode.b0_template', 'inputnode.template_plus')]),
        (b0_hmc_minus, register_b0_refs, [('outputnode.b0_template', 'inputnode.template_minus')]),
        (inputnode, b0_coreg_wf, [('t1w_brain', 'inputnode.anat_image')]),
        (register_b0_refs, b0_coreg_wf, [('outputnode.out_reference', 'inputnode.b0_image')]),
        (inputnode, autobox_t1, [('t1w_brain', 'in_file')]),
        (autobox_t1, deoblique_autobox, [('out_file', 'in_file')]),
        (deoblique_autobox, resample_to_dwi, [('out_file', 'in_file')]),
        (register_b0_refs, dwi_info, [('outputnode.out_reference', 'in_file')]),
        (dwi_info, resample_to_dwi, [('voxel_size', 'voxel_size')])
    ])

    if "T1w" in output_spaces:
        # Put the to-anat and to-b0 transforms into a transform list
        # there will be many transforms this time: motioncorr -> pe template -> unwarp -> anat
        concat_plus_image_transforms = pe.MapNode(
            niu.Merge(4), name='concat_plus_image_transforms', iterfield=['in4'])

        workflow.connect(
            [(b0_hmc_plus, concat_plus_image_transforms, [('outputnode.to_b0_affines', 'in4')])])
        """
        # Put the to-anat and to-b0 transforms into a transform list
        concat_image_transforms = pe.MapNode(
            niu.Merge(2), name="concat_image_transforms", iterfield=["in2"])
        mc_reg_wf.connect(motion_corr_wf, "outputnode.to_b0_affines",
                          concat_image_transforms, "in2")
        mc_reg_wf.connect(coreg_wf, "outputnode.b0_to_anat_transform",
                          concat_image_transforms, "in1")
        # Reverse order for bvec transform
        concat_bvec_transforms = pe.MapNode(
            niu.Merge(2), name="concat_bvec_transforms", iterfield=["in1"])
        mc_reg_wf.connect(motion_corr_wf, "outputnode.to_b0_affines",
                          concat_bvec_transforms, "in1")
        mc_reg_wf.connect(coreg_wf, "outputnode.b0_to_anat_transform",
                          concat_bvec_transforms, "in2")
        """

    return workflow
