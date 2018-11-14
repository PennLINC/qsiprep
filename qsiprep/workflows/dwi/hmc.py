from .unbiased_rigid_alignment import get_alignment_workflow
from fmriprep.engine import Workflow
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as util
from copy import deepcopy
from nipype.interfaces import ants, afni
from .force_orientation import force_orientation


def multi_transform_bvecs(bvec_file, transform_list):
    """Use antsApplyTransformsToPoints to warp two points (a vector)
    and uses the relationship between the warped points to rotate the
    b vector. Only works if absolutely everything is in LPS+.
    transform_list must be in the order they would be specified on the
    commandline to antsApplyTransformsToPoints. They will be inverted
    in this function.
    """
    import os
    import numpy as np
    orig_bvec = np.loadtxt(bvec_file)
    bvec_txt = open("pre_rotation.csv", "w")
    bvec_txt.write("x,y,z,t\n0.0,0.0,0.0,0.0\n")
    # In case there is only one bvec
    if orig_bvec.ndim == 1:
        orig_bvec = orig_bvec[None, :]
    for vec in orig_bvec:
        bvec_txt.write(",".join(map(str, 5 * vec)) + ",0.0\n")
    bvec_txt.close()  # Save it for ants

    def unit_vector(vector):
        """ Returns the unit vector of the vector.  """
        if np.abs(vector).sum() == 0: return vector
        return vector / np.linalg.norm(vector)

    transforms = " ".join(
        ["--transform [%s, 1]" % trf for trf in transform_list])
    os.system("antsApplyTransformsToPoints "
        "--dimensionality 3 "
        "--input pre_rotation.csv "
        "--output rotated_vecs.csv " +
        transforms)

    rotated_vecs = np.loadtxt(
        "rotated_vecs.csv", skiprows=1, delimiter=",")[:, :3]
    rotated_vecs = np.row_stack(
        [unit_vector(v) for v in rotated_vecs - rotated_vecs[0]])
    np.savetxt("rotated_aattp.bvec", rotated_vecs[1:].T, fmt=str("%.8f"))
    return os.path.abspath("rotated_aattp.bvec")


def recombine_dwis(dwi_chunks,
                   bval_chunks,
                   bvec_chunks,
                   original_bvals,
                   original_bvecs,
                   b0_images,
                   b0_threshold=10):
    from dipy.io import read_bvals_bvecs
    import nibabel as nib
    import numpy as np
    import os
    from os.path import abspath

    bvals, bvecs = read_bvals_bvecs(original_bvals, original_bvecs)
    template_img = nib.load(b0_images[0])
    output_dwi_data = np.zeros(template_img.shape + (len(bvals), ))

    # Fill in the b0_images
    b0_indices = np.flatnonzero(bvals < b0_threshold)
    output_b0_data = np.zeros(template_img.shape + (len(b0_indices), ))
    assert len(b0_indices) == len(b0_images)
    for b0_num, (b0_index, b0_image) in enumerate(zip(b0_indices, b0_images)):
        b0_data = nib.load(b0_image).get_data()
        output_dwi_data[..., b0_index] = b0_data
        output_b0_data[..., b0_num] = b0_data
    nib.Nifti1Image(output_b0_data,
                    template_img.affine).to_filename("concatenated_b0s.nii.gz")
    # Fill in the non-b0 images
    non_b0_indices = np.flatnonzero(bvals >= b0_threshold)
    reversed_non_b0_indices = non_b0_indices[::-1].tolist()
    for dir_dwi_image in dwi_chunks:
        dwi_img = nib.load(dir_dwi_image)
        if len(dwi_img.shape) == 3:
            idx = reversed_non_b0_indices.pop()
            output_dwi_data[..., idx] = dwi_img.get_data()
        elif len(dwi_img.shape) == 4:
            chunk_data = dwi_img.get_data()
            for within_chunk_index in range(dwi_img.shape[3]):
                idx = reversed_non_b0_indices.pop()
                output_dwi_data[..., idx] = chunk_data[..., within_chunk_index]
    assert len(reversed_non_b0_indices) == 0
    # dwi signal cannot be negative. Sometimes interpolation causes it to be.
    output_dwi_data[output_dwi_data < 0] = 0
    nib.Nifti1Image(output_dwi_data,
                    template_img.affine).to_filename("concatenated_dwi.nii.gz")

    # Recombine the bvals
    output_bvals = np.zeros_like(bvals)
    chunks = [np.loadtxt(fname) for fname in bval_chunks]

    concat_bvals = np.concatenate(
        [np.loadtxt(fname, ndmin=1) for fname in bval_chunks])
    output_bvals[non_b0_indices] = concat_bvals
    np.savetxt("concatenated.bvals", output_bvals[:, None].T, fmt=str("%d"))
    # Recombine the bvals
    output_bvecs = np.zeros_like(bvecs)
    concat_bvecs = np.column_stack(
        [np.loadtxt(fname, ndmin=2) for fname in bvec_chunks]).T
    output_bvecs[non_b0_indices] = concat_bvecs
    np.savetxt("concatenated.bvecs", output_bvecs.T, fmt=str("%.8f"))

    return [
        abspath("concatenated_dwi.nii.gz"),
        abspath("concatenated.bvals"),
        abspath("concatenated.bvecs"),
        abspath("concatenated_b0s.nii.gz")
    ]

def linear_alignment_workflow(transform="Rigid",
                              metric="Mattes",
                              iternum=0,
                              spatial_bias_correct=False,
                              precision="fine"):
    """
    Takes a template image and a set of input images, does
    a linear alignment to the template and updates it with the
    inverse of the average affine transform to the new template

    Returns a workflow

    """
    iteration_wf = pe.Workflow(name="iterative_alignment_%03d" % iternum)
    input_node_fields = ["image_paths", "template_image", "iteration_num"]
    inputnode = pe.Node(
        util.IdentityInterface(fields=input_node_fields), name='inputnode')
    inputnode.inputs.iteration_num = iternum
    outputnode = pe.Node(
        util.IdentityInterface(fields=[
            "registered_image_paths", "affine_transforms", "updated_template"
        ]),
        name='outputnode')

    reg = ants.Registration()
    reg.inputs.metric = [metric]
    reg.inputs.transforms = [transform]
    reg.inputs.sigma_units = ["vox"]
    reg.inputs.sampling_strategy = ['Random']
    reg.inputs.sampling_percentage = [0.25]
    reg.inputs.radius_or_number_of_bins = [32]
    reg.inputs.initial_moving_transform_com = 0
    reg.inputs.interpolation = 'HammingWindowedSinc'
    reg.inputs.dimension = 3
    reg.inputs.winsorize_lower_quantile = 0.025
    reg.inputs.winsorize_upper_quantile = 0.975
    reg.inputs.convergence_threshold = [1e-06]
    reg.inputs.collapse_output_transforms = True
    reg.inputs.write_composite_transform = False
    reg.inputs.output_warped_image = True
    if precision == "coarse":
        reg.inputs.shrink_factors = [[4, 2]]
        reg.inputs.smoothing_sigmas = [[3., 1.]]
        reg.inputs.number_of_iterations = [[1000, 10000]]
        reg.inputs.transform_parameters = [[0.3]]
    else:
        reg.inputs.shrink_factors = [[4, 2, 1]]
        reg.inputs.smoothing_sigmas = [[3., 1., 0.]]
        reg.inputs.number_of_iterations = [[1000, 10000, 10000]]
        reg.inputs.transform_parameters = [[0.1]]
    iter_reg = pe.MapNode(
        reg, name="reg_%03d" % iternum, iterfield=["moving_image"])

    # Run the images through antsRegistration
    iteration_wf.connect(inputnode, "image_paths", iter_reg, "moving_image")
    iteration_wf.connect(inputnode, "template_image", iter_reg, "fixed_image")

    # Average the images
    averaged_images = pe.Node(
        ants.AverageImages(normalize=True, dimension=3),
        name="averaged_images")
    iteration_wf.connect(iter_reg, "warped_image", averaged_images, "images")

    # Apply the inverse to the average image
    transforms_to_list = pe.Node(util.Merge(1), name="transforms_to_list")
    transforms_to_list.inputs.ravel_inputs = True
    iteration_wf.connect(iter_reg, "forward_transforms", transforms_to_list,
                         "in1")
    avg_affines = pe.Node(ants.AverageAffineTransform(), name="avg_affine")
    avg_affines.inputs.dimension = 3
    avg_affines.inputs.output_affine_transform = "AveragedAffines.mat"
    iteration_wf.connect(transforms_to_list, "out", avg_affines, "transforms")

    invert_average = pe.Node(ants.ApplyTransforms(), name="invert_average")
    invert_average.inputs.interpolation = "HammingWindowedSinc"
    invert_average.inputs.invert_transform_flags = [True]

    avg_to_list = pe.Node(util.Merge(1), name="to_list")
    iteration_wf.connect(avg_affines, "affine_transform", avg_to_list, "in1")
    iteration_wf.connect(avg_to_list, "out", invert_average, "transforms")
    iteration_wf.connect(averaged_images, "output_average_image",
                         invert_average, "input_image")
    iteration_wf.connect(averaged_images, "output_average_image",
                         invert_average, "reference_image")
    iteration_wf.connect(invert_average, "output_image", outputnode,
                         "updated_template")
    iteration_wf.connect(iter_reg, "forward_transforms", outputnode,
                         "affine_transforms")
    iteration_wf.connect(iter_reg, "warped_image", outputnode,
                         "registered_image_paths")

    return iteration_wf

def init_b0_hmc_wf(align_to="iterative", transform="Rigid", spatial_bias_correct=False,
                   metric="Mattes", num_iters=3, name="b0_hmc_wf"):

    if align_to == "iterative" and num_iters < 2:
        raise ValueError("Must specify a positive number of iterations")

    alignment_wf = pe.Workflow(name=name)
    inputnode = pe.Node(
        util.IdentityInterface(fields=['dwi_series']), name='inputnode')
    outputnode = pe.Node(
        util.IdentityInterface(fields=[
            "final_template", "forward_transforms", "iteration_templates",
            "motion_params"
        ]),
        name='outputnode')

    # Iteratively create a template
    if align_to == "iterative":
        initial_template = pe.Node(
            ants.AverageImages(normalize=True, dimension=3),
            name="initial_template")
        alignment_wf.connect(inputnode, "input_images", initial_template,
                             "images")
        # Store the registration targets
        iter_templates = pe.Node(
            util.Merge(num_iters), name="iteration_templates")
        alignment_wf.connect(initial_template, "output_average_image",
                             iter_templates, "in1")

        initial_reg = linear_alignment_workflow(
            transform=transform,
            metric=metric,
            spatial_bias_correct=spatial_bias_correct,
            precision="coarse",
            iternum=0)
        alignment_wf.connect(initial_template, "output_average_image",
                             initial_reg, "inputnode.template_image")
        alignment_wf.connect(inputnode, "input_images", initial_reg,
                             "inputnode.image_paths")
        reg_iters = [initial_reg]
        for iternum in range(1, num_iters):
            reg_iters.append(
                linear_alignment_workflow(
                    transform=transform,
                    metric=metric,
                    spatial_bias_correct=spatial_bias_correct,
                    precision="fine",
                    iternum=iternum))
            alignment_wf.connect(reg_iters[-2], "outputnode.updated_template",
                                 reg_iters[-1], "inputnode.template_image")
            alignment_wf.connect(inputnode, "input_images", reg_iters[-1],
                                 "inputnode.image_paths")
            alignment_wf.connect(reg_iters[-1], "outputnode.updated_template",
                                 iter_templates, "in%d" % (iternum + 1))

        # Compute distance travelled to the template
        summarize_motion = pe.Node(
            interface=util.Function(
                input_names=["motions"],
                output_names=["stacked_motion"],
                function=combine_motion),
            name="summarize_motion")
        alignment_wf.connect(reg_iters[-1], "outputnode.affine_transforms",
                             summarize_motion, "motions")

        # Attach to outputs
        # The last iteration aligned to the output from the second-to-last
        alignment_wf.connect(reg_iters[-2], "outputnode.updated_template",
                             outputnode, "final_template")
        alignment_wf.connect(reg_iters[-1], "outputnode.affine_transforms",
                             outputnode, "forward_transforms")
        alignment_wf.connect(iter_templates, "out", outputnode,
                             "iteration_templates")
        alignment_wf.connect(summarize_motion,"stacked_motion", outputnode,
                             "motion_params")
    return alignment_wf


"""
def (b0_motion_corr_to="iterative",
                                             b0_motion_corr_transform="Rigid",
                                             b0_motion_corr_metric="Mattes",
                                             b0_motion_corr_num_iters=3,
                                             coregister_to="T1w"):
    inputnode = pe.Node(
        util.IdentityInterface(
            fields=["dwi_nifti", "bvals", "bvecs", "anat_image"]),
        name='inputnode')
    outputnode = pe.Node(
        util.IdentityInterface(fields=[
            "b0_template", "motion_params", "iteration_templates",
            "registered_b0_images",
            "registered_dwi", "to_b0_affines"
        ]),
        name='outputnode')

    # Force the dwi and bvecs into LPS+
    itk_orientation = ('L', 'P', 'S')
    force_lps = util.Function(
        input_names=["input_image", "new_axcodes", "bvecs"],
        output_names=["reoriented_nifti", "reoriented_bvecs"],
        function=force_orientation)
    dwi_to_lps = pe.Node(deepcopy(force_lps), name="dwi_to_lps")

    # Apply motion correction to the dwis
    mc_reg_wf = Workflow(name="b0_motion_corr_and_coreg")
    mc_reg_wf.connect(inputnode, "dwi_nifti", dwi_to_lps, "input_image")
    mc_reg_wf.connect(inputnode, "bvecs", dwi_to_lps, "bvecs")
    dwi_to_lps.inputs.new_axcodes = itk_orientation

    motion_corr_wf = get_hmc_workflow()
    mc_reg_wf.connect(dwi_to_lps, "reoriented_nifti", motion_corr_wf,
                      "inputnode.dwi_nifti")
    mc_reg_wf.connect(dwi_to_lps, "reoriented_bvecs", motion_corr_wf,
                      "inputnode.bvecs")
    mc_reg_wf.connect(inputnode, "bvals", motion_corr_wf, "inputnode.bvals")

    bvec_transform = pe.MapNode(
        util.Function(
            input_names=["bvec_file", "transform_list"],
            output_names=["rotated_bvec_file"],
            function=multi_transform_bvecs),
        name="bvec_transform",
        iterfield=["bvec_file", "transform_list"])
    bvec_transform.synchronize = True

    recombine_dwi_chunks = pe.Node(
        interface=util.Function(
            input_names=[
                "dwi_chunks", "bval_chunks", "bvec_chunks", "original_bvals",
                "original_bvecs", "b0_images"
            ],
            output_names=[
                "recombined_dwis", "recombined_bvals", "recombined_bvecs",
                "recombined_b0s"
            ],
            function=recombine_dwis),
        name="recombine_dwi_chunks")

    # Nodes for resampling after motion corr and coreg
    transform_multivol = ants.ApplyTransforms()
    transform_multivol.inputs.float = True
    transform_multivol.inputs.dimension = 3
    transform_multivol.inputs.interpolation = "HammingWindowedSinc"
    transform_multivol.inputs.input_image_type = 3
    transform_singlevol = deepcopy(transform_multivol)
    transform_singlevol.inputs.input_image_type = 0
    warp_dwi_chunks = pe.MapNode(
        transform_multivol,
        name="warp_dwi_chunks",
        iterfield=["input_image", "transforms"])
    warp_dwi_chunks.synchronize = True
    warp_b0s = pe.MapNode(
        transform_singlevol,
        name="warp_b0s",
        iterfield=["input_image", "transforms"])
    warp_b0s.synchronize = True

    # Coregister to the T1w. This requires a lot of extra stuff
    if coregister_to == "T1w":
        t1_to_lps = pe.Node(deepcopy(force_lps), name="t1_to_lps")
        t1_to_lps.inputs.new_axcodes = itk_orientation
        mc_reg_wf.connect(inputnode, "anat_image", t1_to_lps, "input_image")
        coreg_wf = init_b0_to_anat_registration_wf()
        mc_reg_wf.connect(t1_to_lps, "reoriented_nifti", coreg_wf,
                          "inputnode.anat_image")
        mc_reg_wf.connect(motion_corr_wf, "outputnode.b0_template", coreg_wf,
                          "inputnode.b0_image")
        # Put the to-anat and to-b0 transforms into a transform list
        concat_image_transforms = pe.MapNode(
            util.Merge(2), name="concat_image_transforms", iterfield=["in2"])
        mc_reg_wf.connect(motion_corr_wf, "outputnode.to_b0_affines",
                          concat_image_transforms, "in2")
        mc_reg_wf.connect(coreg_wf, "outputnode.b0_to_anat_transform",
                          concat_image_transforms, "in1")
        # Reverse order for bvec transform
        concat_bvec_transforms = pe.MapNode(
            util.Merge(2), name="concat_bvec_transforms", iterfield=["in1"])
        mc_reg_wf.connect(motion_corr_wf, "outputnode.to_b0_affines",
                          concat_bvec_transforms, "in1")
        mc_reg_wf.connect(coreg_wf, "outputnode.b0_to_anat_transform",
                          concat_bvec_transforms, "in2")

        # Create a volume that the DWIs should get warped into.
        # If the anat is in a very different space, this will
        # ensure that a reasonable grid is created
        autobox_t1 = pe.Node(afni.Autobox(), name="autobox_t1")
        autobox_t1.inputs.outputtype = "NIFTI_GZ"
        autobox_t1.inputs.padding = 5
        deoblique_autobox = pe.Node(afni.Warp(), name="deoblique_autobox")
        deoblique_autobox.inputs.deoblique = True
        deoblique_autobox.inputs.outputtype = "NIFTI_GZ"
        resample_to_dwi = pe.Node(afni.Resample(), name="resample_to_dwi")
        resample_to_dwi.inputs.voxel_size = (2.0, 2.0, 2.0)
        resample_to_dwi.inputs.outputtype = "NIFTI_GZ"
        warp_b0_template = pe.Node(
            deepcopy(transform_singlevol), name="warp_b0_template")
        mc_reg_wf.connect(t1_to_lps, "reoriented_nifti", autobox_t1, "in_file")
        mc_reg_wf.connect(autobox_t1, "out_file", deoblique_autobox, "in_file")
        mc_reg_wf.connect(deoblique_autobox, "out_file", resample_to_dwi,
                          "in_file")
        # Warp the dwis
        mc_reg_wf.connect(concat_image_transforms, "out", warp_dwi_chunks,
                          "transforms")
        mc_reg_wf.connect(resample_to_dwi, "out_file", warp_dwi_chunks,
                          "reference_image")
        # Warp the b0s
        mc_reg_wf.connect(concat_image_transforms, "out", warp_b0s,
                          "transforms")
        mc_reg_wf.connect(resample_to_dwi, "out_file", warp_b0s,
                          "reference_image")
        # Correct the bvecs
        mc_reg_wf.connect(concat_bvec_transforms, "out", bvec_transform,
                          "transform_list")

    # Register everything to the b0 template
    elif coregister_to == "b0":
        # Warp the dwi's
        mc_reg_wf.connect(motion_corr_wf, "outputnode.to_b0_affines",
                          warp_dwi_chunks, "transforms")
        mc_reg_wf.connect(motion_corr_wf, "outputnode.b0_template",
                          warp_dwi_chunks, "reference_image")
        # Warp the b0s
        mc_reg_wf.connect(motion_corr_wf, "outputnode.to_b0_affines",
                          warp_b0s, "transforms")
        mc_reg_wf.connect(motion_corr_wf, "outputnode.b0_template", warp_b0s,
                          "reference_image")
        # Correct the bvecs
        mc_reg_wf.connect(motion_corr_wf, "outputnode.to_b0_affines",
                          bvec_transform, "transform_list")

    # Common to both
    mc_reg_wf.connect(motion_corr_wf, "outputnode.dwi_chunks",
                      warp_dwi_chunks, "input_image")
    mc_reg_wf.connect(motion_corr_wf, "outputnode.b0_images", warp_b0s,
                      "input_image")
    mc_reg_wf.connect(motion_corr_wf, "outputnode.bvec_chunks",
                      bvec_transform, "bvec_file")
    mc_reg_wf.connect(warp_dwi_chunks, "output_image", recombine_dwi_chunks,
                      "dwi_chunks")
    mc_reg_wf.connect(bvec_transform, "rotated_bvec_file",
                      recombine_dwi_chunks, "bvec_chunks")
    mc_reg_wf.connect(motion_corr_wf, "outputnode.bval_chunks",
                      recombine_dwi_chunks, "bval_chunks")
    mc_reg_wf.connect(inputnode, "bvals", recombine_dwi_chunks,
                      "original_bvals")
    mc_reg_wf.connect(inputnode, "bvecs", recombine_dwi_chunks,
                      "original_bvecs")
    mc_reg_wf.connect(warp_b0s, "output_image", recombine_dwi_chunks,
                      "b0_images")
    return mc_reg_wf
"""
