# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Anatomical reference preprocessing workflows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_anat_preproc_wf
.. autofunction:: init_skullstrip_ants_wf

"""
from pkg_resources import resource_filename as pkgr

from nipype.pipeline import engine as pe
from nipype.interfaces import (
    utility as niu,
    ants,
    afni,
    mrtrix3
)
from nipype.interfaces.ants import BrainExtraction, N4BiasFieldCorrection

from ...niworkflows.interfaces.registration import RobustMNINormalizationRPT
from ...niworkflows.interfaces.masks import ROIsPlot

from ...engine import Workflow
from ...interfaces import (TemplateDimensions, DerivativesDataSink as FDerivativesDataSink
)

from qsiprep.interfaces import Conform
from ...utils.misc import fix_multi_source_name
from ...interfaces.freesurfer import (
        PrepareSynthStripGrid, FixHeaderSynthStrip, SynthSeg)
from ...interfaces.anatomical import DesaturateSkull, GetTemplate
from ...interfaces.itk import DisassembleTransform, AffineToRigid

from nipype import logging
LOGGER = logging.getLogger('nipype.workflow')


class DerivativesDataSink(FDerivativesDataSink):
    out_path_base = "qsiprep"

TEMPLATE_MAP = {
    'MNI152NLin2009cAsym': 'mni_icbm152_nlin_asym_09c',
    }


#  pylint: disable=R0914
def init_anat_preproc_wf(template, debug, dwi_only,
                         infant_mode, longitudinal, omp_nthreads,
                         output_dir, num_anat_images, output_resolution,
                         nonlinear_register_to_template,
                         reportlets_dir, anatomical_contrast,
                         num_additional_t2ws, has_rois,
                         name='anat_preproc_wf'):
    r"""
    This workflow controls the anatomical preprocessing stages of qsiprep.

    This includes:

     - Creation of a structural template (AC-PC aligned)
     - Skull-stripping and bias correction
     - Tissue segmentation
     - Normalization

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.anatomical import init_anat_preproc_wf
        wf = init_anat_preproc_wf(omp_nthreads=1,
                                  reportlets_dir='.',
                                  output_dir='.',
                                  anatomical_contrast="T1w",
                                  template='MNI152NLin2009cAsym',
                                  output_resolution=1.25,
                                  dwi_only=False,
                                  infant_mode=False,
                                  nonlinear_register_to_template=True,
                                  longitudinal=False,
                                  debug=False,
                                  num_anat_images=1)

    **Parameters**
        dwi_only : bool
            Do not process any anatomical data. Outputs will simply be the template
            and all transforms will be 'identity'
        infant_mode : bool
            Use infant templates
        nonlinear_register_to_template : bool
            Run spatial normalization to template anatomical reference
        output_resolution : float
            A float describing the isotropic voxel size of the output data.
            Sometimes it can be nice to upsample DWIs. If you choose to upsample, be
            sure to choose a robust option for ``interpolation`` to avoid ringing
            artifacts. One option is 'BSpline', which matches mrtrix.
        template : str
            Name of template targeted by ``template`` output space
        debug : bool
            Enable debugging outputs
        longitudinal : bool
            Create unbiased structural template, regardless of number of inputs
            (may increase runtime)
        omp_nthreads : int
            Maximum number of threads an individual process may use
        reportlets_dir : str
            Directory in which to save reportlets
        output_dir : str
            Directory in which to save derivatives
        name : str, optional
            Workflow name (default: anat_preproc_wf)

    **Inputs**

        t1w
            List of T1-weighted structural images
        t2w
            List of T2-weighted structural images
        flair
            List of FLAIR images
        subjects_dir
            FreeSurfer SUBJECTS_DIR


    **Outputs**

        t1_preproc
            Bias-corrected structural template, defining T1w space
        t1_brain
            Skull-stripped ``t1_preproc``
        t1_mask
            Mask of the skull-stripped template image
        t1_seg
            Segmentation of preprocessed structural image, including
            gray-matter (GM), white-matter (WM) and cerebrospinal fluid (CSF)
        t1_tpms
            List of tissue probability maps in T1w space
        t2_preproc
            List of preprocessed t2w files
        t1_2_mni
            T1w template, normalized to MNI space
        t1_2_mni_forward_transform
            ANTs-compatible affine-and-warp transform file
        t1_2_mni_reverse_transform
            ANTs-compatible affine-and-warp transform file (inverse)
        t1_resampling_grid
            Image of the preprocessed t1 to be used as the reference output for dwis

    """

    workflow = Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['t1w', 't2w', 'roi', 'flair', 'subjects_dir', 'subject_id']),
        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['t1_preproc', 't2_preproc', 't1_brain', 't1_mask', 't1_seg', 't1_aseg', 't1_aparc',
                't1_2_mni', 't1_2_mni_forward_transform', 't1_2_mni_reverse_transform',
                't2w_unfatsat', 'segmentation_qc',
                'template_transforms', 'acpc_transform', 'acpc_inv_transform',
                'dwi_sampling_grid']),
        name='outputnode')

    # Make sure we have usable anatomical reference images/masks
    desc = """\
A template {contrast}w image in {template} space was selected as a standard
reference image. """
    get_template_image = pe.Node(
        GetTemplate(template_name=template,
                    infant_mode=infant_mode,
                    anatomical_contrast=anatomical_contrast),
        name="get_template_image")

    # Create the output reference grid_image
    reference_grid_wf = init_output_grid_wf(voxel_size=output_resolution,
                                            padding=4 if infant_mode else 8)
    workflow.connect([
        (get_template_image, reference_grid_wf, [
            ('template_mask_file', 'inputnode.template_image')]),
        (reference_grid_wf, outputnode, [('outputnode.grid_image', 'dwi_sampling_grid')])])

    if dwi_only:
        LOGGER.info("No anatomical scans available! Visual reports will show template masks.")
        workflow.connect([
            (get_template_image, outputnode, [
                ('template_file', 't1_preproc'),
                ('template_brain_file', 't1_brain'),
                ('template_mask_file', 't1_mask'),
                ('template_mask_file', 't1_seg')])])
        workflow.add_nodes([inputnode])
        return workflow

    workflow.__postdesc__ = """\
Brain tissue segmentation of cerebrospinal fluid (CSF),
white-matter (WM) and gray-matter (GM) was performed on
the {contrast} using `SynthSeg` [FreeSurfer, @synthseg].
Brain extraction was performed on the {contrast} image
using `SynthStrip` [FreeSurfer, @synthstrip]
""".format(
        ants_ver=BrainExtraction().version or '<ver>',
        contrast=anatomical_contrast
    )
    desc = """Anatomical data preprocessing

: """
    desc += """\
A total of {num_anats} {contrast}-weighted ({contrast}w) images were found within the input
BIDS dataset.
All of them were corrected for intensity non-uniformity (INU)
using `N4BiasFieldCorrection` [@n4, ANTs {ants_ver}].
""" if num_anat_images > 1 else """\
The {contrast}-weighted ({contrast}w) image was corrected for intensity non-uniformity (INU)
using `N4BiasFieldCorrection` [@n4, ANTs {ants_ver}],
and used as an anatomical reference throughout the workflow.
"""
    workflow.__desc__ = desc.format(
        num_anats=num_anat_images,
        ants_ver=BrainExtraction().version or '<ver>',
        contrast=anatomical_contrast[:-1],  # remove the "w"
        template=template
    )

    # Ensure there is 1 and only 1 anatomical reference
    anat_reference_wf = init_anat_template_wf(longitudinal=longitudinal,
                                              omp_nthreads=omp_nthreads,
                                              num_images=num_anat_images,
                                              sloppy=debug,
                                              anatomical_contrast=anatomical_contrast)

    # Do some padding to prevent memory issues in the synth workflows
    pad_anat_reference_wf = init_dl_prep_wf(name="pad_anat_reference_wf")

    # Skull strip the anatomical reference
    synthstrip_anat_wf = init_synthstrip_wf(
        omp_nthreads=omp_nthreads,
        unfatsat=anatomical_contrast=="T2w",
        name="synthstrip_anat_wf")

    # Segment the anatomical reference
    synthseg_anat_wf = init_synthseg_wf(
        omp_nthreads=omp_nthreads,
        sloppy=debug,
        name="synthseg_anat_wf")

    # Perform registrations
    anat_normalization_wf = init_anat_normalization_wf(
        sloppy=debug,
        template_name=template,
        omp_nthreads=omp_nthreads,
        do_nonlinear=nonlinear_register_to_template,
        has_rois=has_rois,
        name='anat_normalization_wf')

    # Resampling
    rigid_acpc_resample_brain = pe.Node(
        ants.ApplyTransforms(input_image_type=0,
                             interpolation='LanczosWindowedSinc'),
        name='rigid_acpc_resample_brain')
    rigid_acpc_resample_head = pe.Node(
        ants.ApplyTransforms(input_image_type=0,
                             interpolation='LanczosWindowedSinc'),
        name='rigid_acpc_resample_head')
    rigid_acpc_resample_unfatsat = pe.Node(
        ants.ApplyTransforms(input_image_type=0,
                             interpolation='LanczosWindowedSinc'),
        name='rigid_acpc_resample_unfatsat')
    rigid_acpc_resample_aseg = pe.Node(
        ants.ApplyTransforms(input_image_type=0,
                             interpolation='MultiLabel'),
        name='rigid_acpc_resample_aseg')
    rigid_acpc_resample_mask = pe.Node(
        ants.ApplyTransforms(input_image_type=0,
                             interpolation='MultiLabel'),
        name='rigid_acpc_resample_mask')

    acpc_aseg_to_dseg = pe.Node(
        mrtrix3.LabelConvert(
            in_lut=pkgr("qsiprep", "data/FreeSurferColorLUT.txt"),
        in_config=pkgr("qsiprep", "data/FreeSurfer2dseg.txt"),
            out_file="acpc_dseg.nii.gz"),
        name='acpc_aseg_to_dseg')

    # What to do about T2w's?
    if anatomical_contrast == "T2w":
        workflow.connect([
            (synthstrip_anat_wf, rigid_acpc_resample_unfatsat, [
                ('outputnode.unfatsat', 'input_image')]),
            (anat_normalization_wf, rigid_acpc_resample_unfatsat, [
                ('outputnode.to_template_rigid_transform', 'transforms')]),
            (get_template_image, rigid_acpc_resample_unfatsat, [
                ('template_file', 'reference_image')]),
            (rigid_acpc_resample_unfatsat, outputnode, [('output_image', 't2w_unfatsat')]),
            (rigid_acpc_resample_head, outputnode, [('output_image', 't2_preproc')])])
    else:
        if num_additional_t2ws > 0:
            t2w_preproc_wf = init_t2w_preproc_wf(
                omp_nthreads=omp_nthreads,
                num_t2ws=num_additional_t2ws,
                longitudinal=longitudinal,
                sloppy=debug,
                name="t2w_preproc_wf")
            workflow.connect([
                (rigid_acpc_resample_brain, t2w_preproc_wf, [
                    ('output_image', 'inputnode.t1_brain')]),
                (inputnode, t2w_preproc_wf, [('t2w', 'inputnode.t2w_images')]),
                (t2w_preproc_wf, outputnode, [
                    ('outputnode.t2_preproc', 't2_preproc'),
                    ('outputnode.t2w_unfatsat', 't2w_unfatsat')])
            ])

    seg2msks = pe.Node(niu.Function(function=_seg2msks), name='seg2msks')
    seg_rpt = pe.Node(ROIsPlot(colors=['r', 'magenta', 'b', 'g']), name='seg_rpt')
    anat_reports_wf = init_anat_reports_wf(
       reportlets_dir=reportlets_dir,
       nonlinear_register_to_template=nonlinear_register_to_template,
       name='anat_reports_wf')

    workflow.connect([
        (inputnode, anat_reference_wf, [
            (anatomical_contrast.lower(), 'inputnode.images')]),

        # Make a single anatomical reference. Pad it.
        (anat_reference_wf, pad_anat_reference_wf, [
            ('outputnode.template', 'inputnode.image')]),
        (anat_reference_wf, outputnode, [
            ('outputnode.template_transforms', 'anat_template_transforms')]),

        # SynthStrip
        (pad_anat_reference_wf, synthstrip_anat_wf, [
            ('outputnode.padded_image', 'inputnode.padded_image')]),
        (anat_reference_wf, synthstrip_anat_wf, [
            ('outputnode.template', 'inputnode.original_image')]),

        # SynthSeg
        (pad_anat_reference_wf, synthseg_anat_wf,[
            ('outputnode.padded_image', 'inputnode.padded_image')]),
        (anat_reference_wf, synthseg_anat_wf, [
            ('outputnode.template', 'inputnode.original_image')]),
        (synthseg_anat_wf, outputnode, [
            ('outputnode.qc_file', 'segmentation_qc')]),

        # Make AC-PC transform, do nonlinear registration if requested
        (synthstrip_anat_wf, anat_normalization_wf, [
            ('outputnode.brain_mask', 'inputnode.brain_mask')]),
        (anat_reference_wf, anat_normalization_wf, [
            ('outputnode.bias_corrected', 'inputnode.anatomical_reference')]),
        (get_template_image, anat_normalization_wf, [
            ('template_file', 'inputnode.template_image'),
            ('template_mask_file', 'inputnode.template_mask')]),
        (anat_normalization_wf, outputnode, [
            ('outputnode.to_template_rigid_transform', 'acpc_transform'),
            ('outputnode.from_template_rigid_transform', 'acpc_inv_transform'),
            ('outputnode.to_template_nonlinear_transform', 't1_2_mni_forward_transform'),
            ('outputnode.from_template_nonlinear_transform', 't1_2_mni_reverse_transform')]),

        # Resampling
        (synthstrip_anat_wf, rigid_acpc_resample_brain, [
            ('outputnode.brain_image', 'input_image')]),
        (synthstrip_anat_wf, rigid_acpc_resample_mask, [
            ('outputnode.brain_mask', 'input_image')]),

        (anat_reference_wf, rigid_acpc_resample_head, [
            ('outputnode.bias_corrected', 'input_image')]),
        (synthseg_anat_wf, rigid_acpc_resample_aseg, [
            ('outputnode.aparc_image', 'input_image')]),

        (get_template_image, rigid_acpc_resample_brain, [
            ('template_file', 'reference_image')]),
        (get_template_image, rigid_acpc_resample_mask, [
            ('template_file', 'reference_image')]),
        (get_template_image, rigid_acpc_resample_head, [
            ('template_file', 'reference_image')]),
        (get_template_image, rigid_acpc_resample_aseg, [
            ('template_file', 'reference_image')]),
        (anat_normalization_wf, rigid_acpc_resample_brain, [
            ('outputnode.to_template_rigid_transform', 'transforms')]),
        (anat_normalization_wf, rigid_acpc_resample_mask, [
            ('outputnode.to_template_rigid_transform', 'transforms')]),
        (anat_normalization_wf, rigid_acpc_resample_head, [
            ('outputnode.to_template_rigid_transform', 'transforms')]),
        (anat_normalization_wf, rigid_acpc_resample_aseg, [
            ('outputnode.to_template_rigid_transform', 'transforms')]),
        (rigid_acpc_resample_brain, outputnode, [('output_image', 't1_brain')]),
        (rigid_acpc_resample_mask, outputnode, [('output_image', 't1_mask')]),
        (rigid_acpc_resample_head, outputnode, [('output_image', 't1_preproc')]),
        (rigid_acpc_resample_aseg, outputnode, [('output_image', 't1_aseg')]),
        (rigid_acpc_resample_aseg, acpc_aseg_to_dseg, [('output_image', 'in_file')]),
        (acpc_aseg_to_dseg, outputnode, [('out_file', 't1_seg')]),

        # Reports
        (outputnode, seg2msks, [('t1_seg', 'in_file')]),
        (seg2msks, seg_rpt, [('out', 'in_rois')]),
        (outputnode, seg_rpt, [
            ('t1_mask', 'in_mask'),
            ('t1_preproc', 'in_file')]),
        (inputnode, anat_reports_wf, [
            ((anatomical_contrast.lower(), fix_multi_source_name, False, anatomical_contrast),
             'inputnode.source_file')]),
        (anat_reference_wf, anat_reports_wf, [
            ('outputnode.out_report', 'inputnode.t1_conform_report')]),
        (seg_rpt, anat_reports_wf, [('out_report', 'inputnode.seg_report')]),
        (anat_normalization_wf, anat_reports_wf, [
                ('outputnode.out_report', 'inputnode.t1_2_mni_report')])
    ])

    anat_derivatives_wf = init_anat_derivatives_wf(
        output_dir=output_dir,
        template=template,
        anatomical_contrast=anatomical_contrast,
        nonlinear_register_to_template=nonlinear_register_to_template,
        name="anat_derivatives_wf")

    workflow.connect([
        (anat_reference_wf, anat_derivatives_wf, [
            ('outputnode.valid_list', 'inputnode.source_files')]),
        (outputnode, anat_derivatives_wf, [
            ('anat_template_transforms', 'inputnode.t1_template_transforms'),
            ('acpc_transform', 'inputnode.t1_acpc_transform'),
            ('acpc_inv_transform', 'inputnode.t1_acpc_inv_transform'),
            ('t1_preproc', 'inputnode.t1_preproc'),
            ('t1_mask', 'inputnode.t1_mask'),
            ('t1_seg', 'inputnode.t1_seg'),
            ('t1_aseg', 'inputnode.t1_aseg'),
            ('t1_2_mni_forward_transform', 'inputnode.t1_2_mni_forward_transform'),
            ('t1_2_mni_reverse_transform', 'inputnode.t1_2_mni_reverse_transform'),
            ('t1_2_mni', 'inputnode.t1_2_mni')
        ]),
    ])

    return workflow


def init_t2w_preproc_wf(omp_nthreads, num_t2ws, longitudinal, sloppy,
                        name="t2w_preproc_wf"):
    """If T1w is the anatomical contrast, you may also want to process the T2ws for
    worlflows that can use them (ie DRBUDDI). This """
    workflow = Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['t2w_images', 't1_brain']),
        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['t2_preproc', 't2w_unfatsat']),
        name='outputnode')

#     desc = """\
# Additionally, a total of {num_t2ws} T2-weighted (T2w) images were found within the input
# BIDS dataset.
# All of them were corrected for intensity non-uniformity (INU)
# using `N4BiasFieldCorrection` [@n4, ANTs {ants_ver}].
# """ if num_t2ws > 1 else """\
# The T2-weighted (T2w) image was corrected for intensity non-uniformity (INU)
# using `N4BiasFieldCorrection` [@n4, ANTs {ants_ver}],
# and used as an anatomical reference throughout the workflow.
# """
#     workflow.__desc__ = desc.format(
#         num_anats=num_t2ws,
#         ants_ver=BrainExtraction().version or '<ver>'
#     )

    # Ensure there is 1 and only 1 anatomical reference
    anat_reference_wf = init_anat_template_wf(longitudinal=longitudinal,
                                              omp_nthreads=omp_nthreads,
                                              num_images=num_t2ws,
                                              sloppy=sloppy,
                                              anatomical_contrast="T2w")

    # Skull strip the anatomical reference
    synthstrip_anat_wf = init_synthstrip_wf(
        do_padding=True,
        omp_nthreads=omp_nthreads,
        unfatsat=True,
        name="synthstrip_anat_wf")

    # Perform registrations
    settings = pkgr("qsiprep", "data/affine.json")
    t2_brain_to_t1_brain = pe.Node(
        ants.Registration(),
        name="t2_brain_to_t1_brain",
        n_procs=omp_nthreads)

    # Resampling
    rigid_resample_t2w = pe.Node(
        ants.ApplyTransforms(input_image_type=0,
                             interpolation='LanczosWindowedSinc'),
        name='rigid_resample_t2w')
    rigid_resample_unfatsat = pe.Node(
        ants.ApplyTransforms(input_image_type=0,
                             interpolation='LanczosWindowedSinc'),
        name='rigid_resample_unfatsat')

    workflow.connect([
        (inputnode, anat_reference_wf, [
            ('t2w_images', 'inputnode.images')]),

        # Make a single anatomical reference. Pad it.
        (anat_reference_wf, synthstrip_anat_wf, [
            ('outputnode.template', 'inputnode.original_image')]),
        # (anat_reference_wf, outputnode, [
        #    ('outputnode.template_transforms', 'anat_template_transforms')]),

        # Register the skull-stripped T2w to the skull-stripped T2w
        (synthstrip_anat_wf, t2_brain_to_t1_brain, [
            ('outputnode.brain_image', 'moving_image')]),
        (inputnode, t2_brain_to_t1_brain, [
            ('t1_brain', 'fixed_image')]),

        # Resampling
        (synthstrip_anat_wf, rigid_resample_unfatsat, [
            ('outputnode.unfatsat', 'input_image')]),
        (anat_reference_wf, rigid_resample_t2w, [
            ('outputnode.bias_corrected', 'input_image')]),
        (inputnode, rigid_resample_unfatsat, [
            ('t1_brain', 'reference_image')]),
        (inputnode, rigid_resample_t2w, [
            ('t1_brain', 'reference_image')]),
        (t2_brain_to_t1_brain, rigid_resample_unfatsat, [
            ('forward_transforms', 'transforms')]),
        (t2_brain_to_t1_brain, rigid_resample_t2w, [
            ('forward_transforms', 'transforms')]),
        (rigid_resample_unfatsat, outputnode, [('output_image', 't2w_unfatsat')]),
        (rigid_resample_t2w, outputnode, [('output_image', 't2_preproc')])
    ])

    return workflow


def init_anat_template_wf(longitudinal, omp_nthreads, num_images,
                          anatomical_contrast, sloppy, name='anat_template_wf'):
    r"""
    This workflow generates a canonically oriented structural template from
    input anatomical images.


    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.anatomical import init_anat_template_wf
        wf = init_anat_template_wf(longitudinal=False, omp_nthreads=1,
                                   anatomical_contrast="T1w", num_images=1)

    **Parameters**

        longitudinal : bool
            Create unbiased structural template, regardless of number of inputs
            (may increase runtime)
        omp_nthreads : int
            Maximum number of threads an individual process may use
        anatomical_contrast : str
            Contrast to use for anatomical images
        num_images : int
            Number of anatomical images
        name : str, optional
            Workflow name (default: anat_template_wf)


    **Inputs**

        anatomical_images
            List of structural images


    **Outputs**

        template
            Structural template, defining T1w space
        template_transforms
            List of affine transforms from ``template`` to original images
        out_report
            Conformation report
    """

    from ..dwi.hmc import init_b0_hmc_wf
    workflow = Workflow(name=name)

    if num_images > 1:
        workflow.__desc__ = """\
A {contrast}-reference map was computed after registration of
{num_images} {contrast} images (after INU-correction) using
`antsRegistration` [ANTs {ants_ver}].
""".format(contrast=anatomical_contrast, num_images=num_images,
           ants_ver=BrainExtraction().version or '<ver>',)

    inputnode = pe.Node(niu.IdentityInterface(fields=['images']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['template', 'bias_corrected', 'valid_list',
                'template_transforms', 'out_report']),
        name='outputnode')

    # 0. Reorient anatomical image(s) to LPS and resample to common voxel space
    template_dimensions = pe.Node(TemplateDimensions(), name='template_dimensions')
    anat_conform = pe.MapNode(Conform(deoblique_header=True), iterfield='in_file',
                              name='anat_conform')

    workflow.connect([
        (inputnode, template_dimensions, [('images', 't1w_list')]),
        (template_dimensions, anat_conform, [
            ('t1w_valid_list', 'in_file'),
            ('target_zooms', 'target_zooms'),
            ('target_shape', 'target_shape')]),
        (template_dimensions, outputnode, [('out_report', 'out_report'),
                                           ('t1w_valid_list', 'valid_list')]),
    ])

    # To match what was done in antsBrainExtraction.sh
    # -c "[ 50x50x50x50,0.0000001 ]"
    # -s 4
    # -b [ 200 ]
    n4_interface = N4BiasFieldCorrection(
        dimension=3,
        copy_header=True,
        n_iterations=[50,50,50,50],
        convergence_threshold=0.0000001,
        shrink_factor=4,
        bspline_fitting_distance=200.)

    if num_images == 1:
        def _get_first(in_list):
            if isinstance(in_list, (list, tuple)):
                return in_list[0]
            return in_list
        n4_correct = pe.Node(
            n4_interface,
            name='n4_correct',
            n_procs=omp_nthreads)

        outputnode.inputs.template_transforms = [pkgr('qsiprep', 'data/itkIdentityTransform.txt')]

        workflow.connect([
            (anat_conform, outputnode, [
                (('out_file', _get_first), 'template')]),
            (anat_conform, n4_correct, [
                (('out_file', _get_first), 'input_image')]),
            (n4_correct, outputnode, [('output_image', 'bias_corrected')])
        ])

        return workflow

    # 1. Template (only if several images)
    # 1a. Correct for bias field: the bias field is an additive factor
    #     in log-transformed intensity units. Therefore, it is not a linear
    #     combination of fields and N4 fails with merged images.
    # 1b. Align and merge if several T1w images are provided
    n4_correct = pe.MapNode(
        n4_interface,
        iterfield='input_image', name='n4_correct',
        n_procs=omp_nthreads)


    # Make an unbiased template, same as used for b=0 registration
    anat_merge_wf = init_b0_hmc_wf(
        align_to="first" if not longitudinal else "iterative",
        transform="Rigid",
        sloppy=sloppy,
        name="anat_merge_wf",
        num_iters=2,
        omp_nthreads=omp_nthreads,
        boilerplate=False
    )

    workflow.connect([
        (anat_conform, n4_correct, [('out_file', 'input_image')]),
        (n4_correct, anat_merge_wf, [('output_image', 'inputnode.b0_images')]),
        (anat_merge_wf, outputnode, [
            ('outputnode.final_template', 'template'),
            ('outputnode.final_template', 'bias_corrected'),
            ('outputnode.forward_transforms', 'template_transforms')])])

    return workflow


def init_anat_normalization_wf(sloppy, template_name, omp_nthreads,
                               do_nonlinear, has_rois=False,
                               name='anat_normalization_wf'):
    r"""
    This workflow performs registration from the original anatomical reference to the
    template anatomical reference.


    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.anatomical import init_skullstrip_ants_wf
        wf = init_anat_registration_wf(debug=False,
                                       omp_nthreads=1,
                                       acpc_template='test')

    Parameters
        template_image: str
            Path to an image that will be used for Rigid ACPC align
        skull_strip_template : str
            Name of ANTs skull-stripping template ('OASIS' or 'NKI')
        debug : bool
            Enable debugging outputs
        omp_nthreads : int
            Maximum number of threads an individual process may use

    Inputs

        in_file
            T1-weighted structural image to skull-strip

    Outputs

        to_template_nonlinear_transform
            Bias-corrected ``in_file``, before skull-stripping
        to_template_rigid_transform
            Skull-stripped ``in_file``
        out_mask
            Binary mask of the skull-stripped ``in_file``
        out_report
            Reportlet visualizing quality of skull-stripping
    """

    workflow = Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['template_image', 'template_mask',
                    'anatomical_reference', 'brain_mask', 'roi']),
        name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(fields=[
            'to_template_nonlinear_transform', 'to_template_rigid_transform',
            'from_template_nonlinear_transform', 'from_template_rigid_transform',
            'out_report']),
        name='outputnode')


    # get a good ACPC transform
    acpc_settings = pkgr(
        "qsiprep",
        "data/intramodal_ACPC.json" if not sloppy else "data/intramodal_ACPC_sloppy.json")
    acpc_reg = pe.Node(
        RobustMNINormalizationRPT(
            float=True,
            generate_report=False,
            settings=[acpc_settings]),
        name="acpc_reg",
        n_procs=omp_nthreads)
    disassemble_transform = pe.Node(
        DisassembleTransform(),
        name="disassemble_transform")
    extract_rigid_transform = pe.Node(
        AffineToRigid(),
        name="extract_rigid_transform")

    workflow.connect([
        (inputnode, acpc_reg, [
            ('template_image', 'reference_image'),
            ('template_mask', 'reference_mask'),
            ('anatomical_reference', 'moving_image'),
            ('roi', 'lesion_mask'),
            ('brain_mask', 'moving_mask')]),
        (acpc_reg, disassemble_transform, [
            ('composite_transform', 'in_file')]),
        (disassemble_transform, extract_rigid_transform, [
            (('out_transforms', _get_affine_component), 'affine_transform')]),
        (extract_rigid_transform, outputnode, [
            ('rigid_transform', 'to_template_rigid_transform'),
            ('rigid_transform_inverse', 'from_template_rigid_transform')]),
        ])

    if not do_nonlinear:
        return workflow

    rigid_acpc_resample_anat = pe.Node(
        ants.ApplyTransforms(input_image_type=0,
                             interpolation='LanczosWindowedSinc'),
        name='rigid_acpc_resample_anat')
    LOGGER.info("Running nonlinear normalization to template")
    rigid_acpc_resample_mask = pe.Node(
        ants.ApplyTransforms(input_image_type=0,
                             interpolation='MultiLabel'),
        name='rigid_acpc_resample_mask')

    if sloppy:
        LOGGER.info("Using QuickSyN")
        # Requires a warp file: make an inaccurate one
        settings = pkgr('qsiprep', 'data/quick_syn.json')
        anat_norm_interface = RobustMNINormalizationRPT(
            float=True,
            generate_report=True,
            settings=[settings])
    else:
        anat_norm_interface = RobustMNINormalizationRPT(
                float=True,
                generate_report=True,
                flavor='precise')

    anat_nlin_normalization = pe.Node(
        anat_norm_interface,
        name='anat_nlin_normalization',
        n_procs=omp_nthreads)
    anat_nlin_normalization.inputs.template = template_name
    anat_nlin_normalization.inputs.orientation = "LPS"

    workflow.connect([
        (inputnode, anat_nlin_normalization, [
            ('template_image', 'reference_image'),
            ('template_mask', 'reference_mask')]),
        (inputnode, rigid_acpc_resample_mask, [
            ('template_image', 'reference_image'),
            ('brain_mask', 'input_image')]),
        (inputnode, rigid_acpc_resample_anat, [
            ('template_image', 'reference_image'),
            ('anatomical_reference', 'input_image')]),
        (extract_rigid_transform, rigid_acpc_resample_anat, [
            ('rigid_transform', 'transforms')]),
        (extract_rigid_transform, rigid_acpc_resample_mask, [
            ('rigid_transform', 'transforms')]),
        (rigid_acpc_resample_anat, anat_nlin_normalization, [
            ('output_image', 'moving_image')]),
        (rigid_acpc_resample_mask, anat_nlin_normalization, [
            ('output_image', 'moving_mask')]),
        (anat_nlin_normalization, outputnode, [
            ('composite_transform', 'to_template_nonlinear_transform'),
            ('inverse_composite_transform', 'from_template_nonlinear_transform'),
            ('out_report', 'out_report')])
    ])

    if has_rois:
        rigid_acpc_resample_roi = pe.Node(
            ants.ApplyTransforms(input_image_type=0,
                                interpolation='MultiLabel'),
            name='rigid_acpc_resample_roi')
        workflow.connect([
            (rigid_acpc_resample_roi, anat_nlin_normalization, [
                ('output_image', 'lesion_mask')]),
            (extract_rigid_transform, rigid_acpc_resample_roi, [
                ('rigid_transform', 'transforms')]),
            (inputnode, rigid_acpc_resample_roi, [
                ('template_image', 'reference_image'),
                ('roi', 'input_image')]),
        ])


    return workflow


def init_dl_prep_wf(name='dl_prep_wf'):
    """Prepare images for use in the FreeSurfer deep learning functions"""
    workflow = Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['image']), name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['padded_image']),
        name="outputnode")
    skulled_1mm_resample = pe.Node(
        afni.Resample(
            outputtype="NIFTI_GZ",
            voxel_size=(1.0, 1.0, 1.0)),
        name="skulled_1mm_resample")
    skulled_autobox = pe.Node(
        afni.Autobox(outputtype="NIFTI_GZ", padding=3),
        name='skulled_autobox')
    prepare_synthstrip_reference = pe.Node(
        PrepareSynthStripGrid(),
        name="prepare_synthstrip_reference")
    resample_skulled_to_reference = pe.Node(
        ants.ApplyTransforms(
            dimension=3,
            interpolation="BSpline",
            transforms=['identity']),
        name="resample_skulled_to_reference")

    workflow.connect([
        (inputnode, skulled_1mm_resample, [('image', 'in_file')]),
        (skulled_1mm_resample, skulled_autobox, [('out_file', 'in_file')]),
        (skulled_autobox, prepare_synthstrip_reference, [('out_file', 'input_image')]),
        (prepare_synthstrip_reference, resample_skulled_to_reference, [
            ('prepared_image', 'reference_image')]),
        (inputnode, resample_skulled_to_reference, [('image', 'input_image')]),
        (resample_skulled_to_reference, outputnode, [('output_image', 'padded_image')])
    ])
    return workflow


def init_synthstrip_wf(omp_nthreads, do_padding=False,
                       unfatsat=False, name="synthstrip_wf"):
    workflow = Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['padded_image', 'original_image']),
        name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['brain_image', 'brain_mask', 'unfatsat']),
        name='outputnode')

    synthstrip = pe.Node(
        FixHeaderSynthStrip(),
        name="synthstrip",
        n_procs=omp_nthreads)
    mask_to_original_grid = pe.Node(
        ants.ApplyTransforms(
            dimension=3,
            transforms=['identity'],
            interpolation="NearestNeighbor"),
        name="mask_to_original_grid")
    mask_brain = pe.Node(
        ants.MultiplyImages(
            dimension=3,
            output_product_image="masked_brain.nii"),
        name="mask_brain")

    # For T2w images, create an artificially skull-downweighted image
    if unfatsat:
        desaturate_skull = pe.Node(DesaturateSkull(), name='desaturate_skull')
        workflow.connect([
            (mask_brain, desaturate_skull, [('output_product_image', 'brain_mask_image')]),
            (inputnode, desaturate_skull, [('original_image', 'skulled_t2w_image')]),
            (desaturate_skull, outputnode, [('desaturated_t2w', 'unfatsat')])
        ])

    # If the input image isn't already padded, do it here
    if do_padding:
        padding_wf = init_dl_prep_wf(name="pad_before_"+name)
        workflow.connect([
            (inputnode, padding_wf, [('original_image', 'inputnode.image')]),
            (padding_wf, synthstrip, [('outputnode.padded_image', 'input_image')])])
    else:
        workflow.connect([(inputnode, synthstrip, [('padded_image', 'input_image')])])

    workflow.connect([
        (synthstrip, mask_to_original_grid, [('out_brain_mask', 'input_image')]),
        (inputnode, mask_to_original_grid, [('original_image', 'reference_image')]),
        (mask_to_original_grid, outputnode, [('output_image', 'brain_mask')]),
        (inputnode, mask_brain, [('original_image', 'first_input')]),
        (mask_to_original_grid, mask_brain, [("output_image", "second_input")]),
        (mask_brain, outputnode, [('output_product_image', 'brain_image')])
    ])

    return workflow


def init_synthseg_wf(omp_nthreads, sloppy, name="synthseg_wf"):
    workflow = Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['padded_image', 'original_image']),
        name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['aparc_image', 'posterior_image', 'qc_file']),
        name='outputnode')

    synthseg = pe.Node(
        SynthSeg(
            fast=sloppy,
            num_threads=omp_nthreads),
        n_procs=omp_nthreads,
        name='synthseg')

    workflow.connect([
        (inputnode, synthseg, [('padded_image', 'input_image')]),
        (synthseg, outputnode, [
            ('out_seg', 'aparc_image'),
            ('out_post', 'posterior_image'),
            ('out_qc', 'qc_file')
        ])
    ])
    return workflow


def init_output_grid_wf(voxel_size, padding, name='output_grid_wf'):
    """Generate a non-oblique, uniform voxel-size grid around a brain."""
    workflow = Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['template_image']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['grid_image']), name='outputnode')
    autobox_template = pe.Node(afni.Autobox(outputtype="NIFTI_GZ", padding=padding),
                               name='autobox_template')
    deoblique_autobox = pe.Node(afni.Warp(outputtype="NIFTI_GZ", deoblique=True),
                                name="deoblique_autobox")
    resample_to_voxel_size = pe.Node(afni.Resample(outputtype="NIFTI_GZ"),
                                     name="resample_to_voxel_size")
    resample_to_voxel_size.inputs.voxel_size = (voxel_size, voxel_size, voxel_size)

    workflow.connect([
        (inputnode, autobox_template, [('template_image', 'in_file')]),
        (autobox_template, deoblique_autobox, [('out_file', 'in_file')]),
        (deoblique_autobox, resample_to_voxel_size, [('out_file', 'in_file')]),
        (resample_to_voxel_size, outputnode, [('out_file', 'grid_image')])
    ])

    return workflow


def init_anat_reports_wf(reportlets_dir, nonlinear_register_to_template,
                         name='anat_reports_wf'):
    """
    Set up a battery of datasinks to store reports in the right location
    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['source_file', 't1_conform_report', 'seg_report',
                    't1_2_mni_report', 'recon_report']),
        name='inputnode')

    ds_t1_conform_report = pe.Node(
        DerivativesDataSink(base_directory=reportlets_dir, suffix='conform'),
        name='ds_t1_conform_report', run_without_submitting=True)

    ds_t1_2_mni_report = pe.Node(
        DerivativesDataSink(base_directory=reportlets_dir, suffix='t1_2_mni'),
        name='ds_t1_2_mni_report', run_without_submitting=True)

    ds_t1_seg_mask_report = pe.Node(
        DerivativesDataSink(base_directory=reportlets_dir, suffix='seg_brainmask'),
        name='ds_t1_seg_mask_report', run_without_submitting=True)

    workflow.connect([
        (inputnode, ds_t1_conform_report, [('source_file', 'source_file'),
                                           ('t1_conform_report', 'in_file')]),
        (inputnode, ds_t1_seg_mask_report, [('source_file', 'source_file'),
                                            ('seg_report', 'in_file')]),
    ])

    if nonlinear_register_to_template:
        workflow.connect([
            (inputnode, ds_t1_2_mni_report, [('source_file', 'source_file'),
                                             ('t1_2_mni_report', 'in_file')])
        ])

    return workflow


def init_anat_derivatives_wf(output_dir, template, anatomical_contrast,
                             nonlinear_register_to_template, name='anat_derivatives_wf'):
    """
    Set up a battery of datasinks to store derivatives in the right location
    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['source_files', 't1_template_transforms',
                    't1_acpc_transform', 't1_acpc_inv_transform',
                    't1_preproc', 't1_mask', 't1_seg',
                    't1_2_mni_forward_transform', 't1_2_mni_reverse_transform',
                    't1_2_mni', 't1_aseg']),
        name='inputnode')

    t1_name = pe.Node(
        niu.Function(
            function=fix_multi_source_name,
            input_names=["in_files", "dwi_only", "anatomical_contrast"]),
        name='t1_name')
    t1_name.inputs.anatomical_contrast = anatomical_contrast
    t1_name.inputs.dwi_only = False

    ds_t1_preproc = pe.Node(
        DerivativesDataSink(base_directory=output_dir, desc='preproc', keep_dtype=True),
        name='ds_t1_preproc', run_without_submitting=True)

    ds_t1_mask = pe.Node(
        DerivativesDataSink(base_directory=output_dir, desc='brain', suffix='mask'),
        name='ds_t1_mask', run_without_submitting=True)

    ds_t1_seg = pe.Node(
        DerivativesDataSink(base_directory=output_dir, suffix='dseg'),
        name='ds_t1_seg', run_without_submitting=True)

    ds_t1_aseg = pe.Node(
        DerivativesDataSink(base_directory=output_dir, desc='aseg', suffix='dseg'),
        name='ds_t1_aseg', run_without_submitting=True)

    # Transforms
    suffix_fmt = 'from-{}_to-{}_mode-image_xfm'.format
    ds_t1_template_transforms = pe.MapNode(
        DerivativesDataSink(base_directory=output_dir, suffix=suffix_fmt('orig', 'T1w')),
        iterfield=['source_file', 'in_file'],
        name='ds_t1_template_transforms', run_without_submitting=True)

    ds_t1_mni_inv_warp = pe.Node(
        DerivativesDataSink(base_directory=output_dir,
                            suffix=suffix_fmt(template, 'T1w')),
        name='ds_t1_mni_inv_warp', run_without_submitting=True)

    ds_t1_template_acpc_transform = pe.Node(
        DerivativesDataSink(base_directory=output_dir, suffix=suffix_fmt('T1wNative', 'T1wACPC')),
        name='ds_t1_template_acpc_transforms', run_without_submitting=True)

    ds_t1_template_acpc_inv_transform = pe.Node(
        DerivativesDataSink(base_directory=output_dir, suffix=suffix_fmt('T1wACPC', 'T1wNative')),
        name='ds_t1_template_acpc_inv_transforms', run_without_submitting=True)

    ds_t1_mni_warp = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            suffix=suffix_fmt('T1w', template)),
        name='ds_t1_mni_warp', run_without_submitting=True)

    workflow.connect([
        (inputnode, t1_name, [('source_files', 'in_files')]),
        (inputnode, ds_t1_template_transforms, [('source_files', 'source_file'),
                                                ('t1_template_transforms', 'in_file')]),
        (inputnode, ds_t1_template_acpc_transform, [('t1_acpc_transform', 'in_file')]),
        (inputnode, ds_t1_template_acpc_inv_transform, [('t1_acpc_inv_transform', 'in_file')]),
        (inputnode, ds_t1_preproc, [('t1_preproc', 'in_file')]),
        (inputnode, ds_t1_mask, [('t1_mask', 'in_file')]),
        (inputnode, ds_t1_seg, [('t1_seg', 'in_file')]),
        (inputnode, ds_t1_aseg, [('t1_aseg', 'in_file')]),
        (t1_name, ds_t1_preproc, [('out', 'source_file')]),
        (t1_name, ds_t1_template_acpc_transform, [('out', 'source_file')]),
        (t1_name, ds_t1_template_acpc_inv_transform, [('out', 'source_file')]),
        (t1_name, ds_t1_aseg, [('out', 'source_file')]),
        (t1_name, ds_t1_mask, [('out', 'source_file')]),
        (t1_name, ds_t1_seg, [('out', 'source_file')]),
    ])

    if nonlinear_register_to_template:
        workflow.connect([
            (inputnode, ds_t1_mni_warp, [('t1_2_mni_forward_transform', 'in_file')]),
            (inputnode, ds_t1_mni_inv_warp, [('t1_2_mni_reverse_transform', 'in_file')]),
            (t1_name, ds_t1_mni_warp, [('out', 'source_file')]),
            (t1_name, ds_t1_mni_inv_warp, [('out', 'source_file')]),
        ])

    return workflow


def _seg2msks(in_file, newpath=None):
    """Converts labels to masks"""
    import nibabel as nb
    import numpy as np
    from nipype.utils.filemanip import fname_presuffix

    nii = nb.load(in_file)
    labels = nii.get_fdata()

    out_files = []
    for i in range(1, 4):
        ldata = np.zeros_like(labels)
        ldata[labels == i] = 1
        out_files.append(fname_presuffix(
            in_file, suffix='_label%03d' % i, newpath=newpath))
        nii.__class__(ldata, nii.affine, nii.header).to_filename(out_files[-1])

    return out_files


def _get_affine_component(transform_list):
    # The disassembled transform will have the affine transform as the first element
    if isinstance(transform_list, str):
        return transform_list
    return transform_list[0]
