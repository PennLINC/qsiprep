from .unbiased_rigid_alignment import get_alignment_workflow
import fmriprep.engine as pe
import nipype.interfaces.utility as util
from copy import deepcopy
from nipype.interfaces import ants, afni
from ..orientation.force_orientation import force_orientation

def combine_bvals(bvals):
    import numpy as np
    import os
    collected_vals = []
    for bval_file in bvals:
        collected_vals.append(np.loadtxt(bval_file))
    final_bvals = np.concatenate(collected_vals)
    final_bvals = final_bvals[np.logical_not(final_bvals) == 0]
    np.savetxt(
        "restacked.bval",
        np.concatenate([np.array([0]), final_bvals]),
        fmt=str("%i"))
    return os.path.abspath("restacked.bval")


def combine_bvecs(bvecs):
    """ Assumes no b0's are included in the bvecs"""
    import numpy as np
    import os
    collected_vecs = [np.zeros((3, 1))]
    for bvec_file in bvecs:
        collected_vecs.append(np.loadtxt(bvec_file))
    final_bvecs = np.column_stack(collected_vecs)
    np.savetxt("restacked.bvec", final_bvecs, fmt=str("%.8f"))
    return os.path.abspath("restacked.bvec")


def split_dwi(dwi_nifti="", bval_file="", bvec_file="", b0_threshold=10):
    """
    splits DWI into a series of b0 images and their following dwi
    """
    from dipy.io import read_bvals_bvecs
    import nibabel as nib
    import numpy as np
    import os

    bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)
    img = nib.load(dwi_nifti)
    data = img.get_data()
    aff = img.get_affine()

    b0s = np.flatnonzero(bvals < b0_threshold)
    b0_paths = []
    dwi_paths = []
    bval_paths = []
    bvec_paths = []

    # Which b0 is each diffusion-weighted volume nearest to?
    from collections import defaultdict
    nearest_b0 = defaultdict(list)
    for n in np.arange(data.shape[-1]):
        if n in b0s:
            continue
        nearest = np.argmin(np.abs(n - b0s))
        nearest_b0[b0s[nearest]].append(n)

    # Save the b0s and their corresponding nearest images
    for n_b0, b0_index in enumerate(sorted(nearest_b0.keys())):
        #output paths
        out_path = "b0_%03d.nii.gz" % n_b0
        dwi_out_path = "dwi_%03d.nii.gz" % n_b0
        bval_out_path = "dwi_%03d.bval" % n_b0
        bvec_out_path = "dwi_%03d.bvec" % n_b0

        indices = np.array(nearest_b0[b0_index])
        dwi = data[..., indices]
        _bval = bvals[indices]
        _bvec = bvecs[indices]

        nib.Nifti1Image(dwi, aff).to_filename(dwi_out_path)
        nib.Nifti1Image(data[..., b0_index], aff).to_filename(out_path)
        np.savetxt(bvec_out_path, _bvec)
        np.savetxt(bval_out_path, _bval)

        dwi_paths.append(os.path.abspath(dwi_out_path))
        b0_paths.append(os.path.abspath(out_path))
        bval_paths.append(os.path.abspath(bval_out_path))
        bvec_paths.append(os.path.abspath(bvec_out_path))
    return b0_paths, dwi_paths, bval_paths, bvec_paths


def multi_transform_bvecs(bvec_file, transform_list):
    """
    Uses antsApplyTransformsToPoints to warp two points (a vector)
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


def init_b0_alignment_wf(align_to="iterative",
                         transform="Rigid",
                         metric="Mattes",
                         num_iters=3,
                         name="b0_alignment"):

    if align_to == "iterative" and num_iters < 2:
        raise ValueError("Must specify > 2 iterations")

    b0_alignment_wf = pe.Workflow(name=name)
    input_node = pe.Node(
        util.IdentityInterface(fields=["dwi_nifti", "bvals", "bvecs"]),
        name='input_node')
    output_node = pe.Node(
        util.IdentityInterface(fields=[
            "b0_template",
            "motion_params",
            "iteration_templates",
            "b0_images",
            "bval_chunks",
            "dwi_chunks",
            "bvec_chunks",
            "to_b0_affines",
        ]),
        name='output_node')

    dwi_splitter = pe.Node(
        util.Function(
            input_names=["dwi_nifti", "bval_file", "bvec_file"],
            output_names=["b0_paths", "dwi_paths", "bval_paths", "bvec_paths"],
            function=split_dwi),
        name="dwi_splitter")

    b0_alignment_wf.connect(input_node, "dwi_nifti", dwi_splitter, "dwi_nifti")
    b0_alignment_wf.connect(input_node, "bvals", dwi_splitter, "bval_file")
    b0_alignment_wf.connect(input_node, "bvecs", dwi_splitter, "bvec_file")

    # Align the b0s, get the transforms and the template
    iterative_alignment_wf = get_alignment_workflow(
        align_to=align_to,
        transform=transform,
        spatial_bias_correct=False,
        metric=metric,
        num_iters=num_iters)
    b0_alignment_wf.connect(dwi_splitter, "b0_paths", iterative_alignment_wf,
                            "input_node.input_images")
    b0_alignment_wf.connect(iterative_alignment_wf,
                            "output_node.final_template", output_node,
                            "b0_template")
    b0_alignment_wf.connect(iterative_alignment_wf,
                            "output_node.motion_params", output_node,
                            "motion_params")
    b0_alignment_wf.connect(iterative_alignment_wf,
                            "output_node.iteration_templates", output_node,
                            "iteration_templates")
    b0_alignment_wf.connect(iterative_alignment_wf,
                            "output_node.forward_transforms", output_node,
                            "to_b0_affines")
    b0_alignment_wf.connect([(dwi_splitter, output_node, [
        ("b0_paths", "b0_images"),
        ("bval_paths", "bval_chunks"),
        ("dwi_paths", "dwi_chunks"),
        ("bvec_paths", "bvec_chunks"),
    ])])

    return b0_alignment_wf


def get_b0_to_anat_registration_workflow(biascorrect_anat=False,
                                         biascorrect_b0=False):
    """
    registers a b0 to an anatomical scan. Bias corrects each
    (if requested) and coregisters the b0 to the anat. Returns
    the transform and Mattes score
    """
    input_node = pe.Node(
        util.IdentityInterface(fields=["b0_image", "anat_image"]),
        name='input_node')
    output_node = pe.Node(
        util.IdentityInterface(
            fields=["b0_to_anat_transform", "coreg_metric"]),
        name='output_node')
    b0_anat_coreg_wf = pe.Workflow(name="b0_anat_coreg")

    # Defines a coregistration operation
    coreg = ants.Registration()
    coreg.inputs.metric = ["Mattes"]
    coreg.inputs.transforms = ["Rigid"]
    coreg.inputs.shrink_factors = [[8, 4, 2, 1]]
    coreg.inputs.smoothing_sigmas = [[7., 3., 1., 0.]]
    coreg.inputs.sigma_units = ["vox"]
    coreg.inputs.sampling_strategy = ['Random']
    coreg.inputs.sampling_percentage = [0.25]
    coreg.inputs.radius_or_number_of_bins = [32]
    coreg.inputs.initial_moving_transform_com = 0
    coreg.inputs.interpolation = 'HammingWindowedSinc'
    coreg.inputs.dimension = 3
    coreg.inputs.winsorize_lower_quantile = 0.025
    coreg.inputs.winsorize_upper_quantile = 0.975
    coreg.inputs.number_of_iterations = [[10000, 1000, 10000, 10000]]
    coreg.inputs.transform_parameters = [[0.2]]
    coreg.inputs.convergence_threshold = [1e-06]
    coreg.inputs.collapse_output_transforms = True
    coreg.inputs.write_composite_transform = False
    coreg.inputs.output_warped_image = True
    b0_to_anat = pe.Node(coreg, name="b0_to_anat")

    # N4 bias correction on the t2 and b0 before coregistration
    biascorr = ants.N4BiasFieldCorrection()
    biascorr.inputs.n_iterations = [50, 50, 50, 50, 50]
    if biascorrect_anat:
        biascorr_anat = pe.Node(deepcopy(biascorr), name="biascorr_anat")
        b0_anat_coreg_wf.connect(input_node, "anat_image", biascorr_anat,
                                 "input_image")
        b0_anat_coreg_wf.connect(biascorr_anat, "output_image", b0_to_anat,
                                 "fixed_image")
    else:
        b0_anat_coreg_wf.connect(input_node, "anat_image", b0_to_anat,
                                 "fixed_image")

    if biascorrect_b0:
        biascorr_b0 = pe.Node(deepcopy(biascorr), name="biascorr_b0")
        b0_anat_coreg_wf.connect(input_node, "b0_image", biascorr_b0,
                                 "input_image")
        b0_anat_coreg_wf.connect(biascorr_b0, "output_image", b0_to_anat,
                                 "moving_image")
    else:
        b0_anat_coreg_wf.connect(input_node, "b0_image", b0_to_anat,
                                 "moving_image")

    b0_anat_coreg_wf.connect(b0_to_anat, "forward_transforms", output_node,
                             "b0_to_anat_transform")
    b0_anat_coreg_wf.connect(b0_to_anat, "metric_value", output_node,
                             "coreg_metric")

    return b0_anat_coreg_wf


def init_b0_motioncorr_registration_pipeline(b0_motion_corr_to="iterative",
                                             b0_motion_corr_transform="Rigid",
                                             b0_motion_corr_metric="Mattes",
                                             b0_motion_corr_num_iters=3,
                                             coregister_to="T1w"):
    """Applies a simple b0-based
    """
    input_node = pe.Node(
        util.IdentityInterface(
            fields=["dwi_nifti", "bvals", "bvecs", "anat_image"]),
        name='input_node')
    output_node = pe.Node(
        util.IdentityInterface(fields=[
            "b0_template", "motion_params", "iteration_templates",
            "registered_b0_images", "bvals", "corrected_bvecs",
            "registered_dwi", "to_b0_affines", "to_T1w_affines"
        ]),
        name='output_node')

    # Force the dwi and bvecs into LPS+
    itk_orientation = ('L', 'P', 'S')
    force_lps = util.Function(
        input_names=["input_image", "new_axcodes", "bvecs"],
        output_names=["reoriented_nifti", "reoriented_bvecs"],
        function=force_orientation)
    dwi_to_lps = pe.Node(deepcopy(force_lps), name="dwi_to_lps")

    # Apply motion correction to the dwis
    mc_reg_wf = pe.Workflow(name="b0_motion_corr_and_coreg")
    mc_reg_wf.connect(input_node, "dwi_nifti", dwi_to_lps, "input_image")
    mc_reg_wf.connect(input_node, "bvecs", dwi_to_lps, "bvecs")
    dwi_to_lps.inputs.new_axcodes = itk_orientation

    motion_corr_wf = get_b0_alignment_workflow()
    mc_reg_wf.connect(dwi_to_lps, "reoriented_nifti", motion_corr_wf,
                      "input_node.dwi_nifti")
    mc_reg_wf.connect(dwi_to_lps, "reoriented_bvecs", motion_corr_wf,
                      "input_node.bvecs")
    mc_reg_wf.connect(input_node, "bvals", motion_corr_wf, "input_node.bvals")

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
        mc_reg_wf.connect(input_node, "anat_image", t1_to_lps, "input_image")
        coreg_wf = get_b0_to_anat_registration_workflow()
        mc_reg_wf.connect(t1_to_lps, "reoriented_nifti", coreg_wf,
                          "input_node.anat_image")
        mc_reg_wf.connect(motion_corr_wf, "output_node.b0_template", coreg_wf,
                          "input_node.b0_image")
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
        mc_reg_wf.connect(motion_corr_wf, "output_node.to_b0_affines",
                          warp_dwi_chunks, "transforms")
        mc_reg_wf.connect(motion_corr_wf, "output_node.b0_template",
                          warp_dwi_chunks, "reference_image")
        # Warp the b0s
        mc_reg_wf.connect(motion_corr_wf, "output_node.to_b0_affines",
                          warp_b0s, "transforms")
        mc_reg_wf.connect(motion_corr_wf, "output_node.b0_template", warp_b0s,
                          "reference_image")
        # Correct the bvecs
        mc_reg_wf.connect(motion_corr_wf, "output_node.to_b0_affines",
                          bvec_transform, "transform_list")

    # Common to both
    mc_reg_wf.connect(motion_corr_wf, "output_node.dwi_chunks",
                      warp_dwi_chunks, "input_image")
    mc_reg_wf.connect(motion_corr_wf, "output_node.b0_images", warp_b0s,
                      "input_image")
    mc_reg_wf.connect(motion_corr_wf, "output_node.bvec_chunks",
                      bvec_transform, "bvec_file")
    mc_reg_wf.connect(warp_dwi_chunks, "output_image", recombine_dwi_chunks,
                      "dwi_chunks")
    mc_reg_wf.connect(bvec_transform, "rotated_bvec_file",
                      recombine_dwi_chunks, "bvec_chunks")
    mc_reg_wf.connect(motion_corr_wf, "output_node.bval_chunks",
                      recombine_dwi_chunks, "bval_chunks")
    mc_reg_wf.connect(input_node, "bvals", recombine_dwi_chunks,
                      "original_bvals")
    mc_reg_wf.connect(input_node, "bvecs", recombine_dwi_chunks,
                      "original_bvecs")
    mc_reg_wf.connect(warp_b0s, "output_image", recombine_dwi_chunks,
                      "b0_images")
    return mc_reg_wf
