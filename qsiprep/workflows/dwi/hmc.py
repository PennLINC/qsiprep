from fmriprep.engine import Workflow
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as util
from nipype.interfaces import ants, afni
import pandas as pd
from dipy.core.geometry import decompose_matrix


def combine_motion(motions):
    import numpy as np
    import os
    from dipy.core.geometry import decompose_matrix
    import pandas as pd
    collected_motion = []
    for motion_file in motions:
        if os.path.exists("output.txt"):
            os.remove("output.txt")
        # Convert to homogenous matrix
        os.system("ConvertTransformFile 3 %s output.txt --RAS --hm" % (motion_file[0]))
        affine = np.loadtxt("output.txt")
        scale, shear, angles, translate, persp = decompose_matrix(affine)
        collected_motion.append(np.concatenate([scale, shear,
                                np.array(angles)*180/np.pi, translate]))

    final_motion = np.row_stack(collected_motion)
    cols = ["scaleX", "scaleY", "scaleZ", "shearXY", "shearXZ",
            "shearYZ", "rotateX", "rotateY", "rotateZ", "shiftX", "shiftY",
            "shiftZ"]
    motion_df = pd.DataFrame(data=final_motion, columns=cols)
    motion_df.to_csv("motion_params.csv", index=False)
    return os.path.abspath("motion_params.csv")


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
        util.IdentityInterface(fields=["registered_image_paths", "affine_transforms",
                                       "updated_template"]), name='outputnode')

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
        util.IdentityInterface(fields=['b0_images']), name='inputnode')
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
        alignment_wf.connect(inputnode, "b0_images", initial_template,
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
        alignment_wf.connect(inputnode, "b0_images", initial_reg,
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
            alignment_wf.connect(inputnode, "b0_images", reg_iters[-1],
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
        alignment_wf.connect(summarize_motion, "stacked_motion", outputnode,
                             "motion_params")
    return alignment_wf
