# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Merge and denoise dwi images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_dwi_preproc_wf
.. autofunction:: init_dwi_derivatives_wf

"""
from nipype import logging
from nipype.pipeline import engine as pe
from nipype.utils.filemanip import split_filename
from nipype.interfaces import utility as niu
from .util import _get_wf_name
from .qc import init_modelfree_qc_wf
from ...interfaces import ConformDwi, DerivativesDataSink
from ...interfaces.dipy import Patch2Self
from ...interfaces.mrtrix import DWIDenoise, DWIBiasCorrect, MRDeGibbs
from ...interfaces.gradients import ExtractB0s
from ...interfaces.nilearn import MaskEPI, Merge
from ...interfaces.dwi_merge import MergeDWIs, StackConfounds
from ...engine import Workflow

DEFAULT_MEMORY_MIN_GB = 0.01
LOGGER = logging.getLogger('nipype.workflow')


def init_merge_and_denoise_wf(raw_dwi_files,
                              dwi_denoise_window,
                              unringing_method,
                              dwi_no_biascorr,
                              denoise_method,
                              no_b0_harmonization,
                              denoise_before_combining,
                              orientation,
                              b0_threshold,
                              source_file,
                              mem_gb=1,
                              omp_nthreads=1,
                              calculate_qc=False,
                              phase_id="same",
                              name="merge_and_denoise_wf"):
    """

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.dwi import init_merge_and_denoise_wf
        wf = init_merge_and_dwnoise_wf(['/path/to/dwi/sub-1_dwi.nii.gz'],
                                       source_file='/data/sub-1/dwi/sub-1_dwi.nii.gz',
                                       dwi_denoise_window=7,
                                       denoise_method='patch2self',
                                       b0_threshold=100,
                                       unringing_method='mrdegibbs',
                                       dwi_no_biascorr=False,
                                       no_b0_harmonization=False,
                                       denoise_before_combining=True,
                                       combine_all_dwis=True)

    **Parameters**

        raw_dwi_files : list
            list of raw (in their original BIDS directory) dwi nifti files
        b0_threshold : int
            Maximum b value for an image to be considered a b=0
        dwi_denoise_window : int
            window size in voxels for image-based denoising. Must be odd. If 0, '
            'denoising will not be run'
        denoise_method : str
            Either 'dwidenoise', 'patch2self' or 'none'
        unringing_method : str
            algorithm to use for removing Gibbs ringing. Options: none, mrdegibbs
        dwi_no_biascorr : bool
            run spatial bias correction (N4) on dwi series
        no_b0_harmonization : bool
            skip rescaling dwi scans to have matching b=0 intensities across scans
        denoise_before_combining : bool
            run ``dwidenoise`` before combining dwis. Requires ``combine_all_dwis``
            If ``dwi_denoise_window > 0`` and this is ``False``, then ``dwidenoise``
            is run on the merged dwi series.
        calculate_qc : bool
            Should DSI Studio's QC be calculated on the merged raw data?

    **Outputs**

        merged_image
            dwi series, conformed, denoised if requested
        merged_raw_image
            dwi series, conformed, raw
        merged_bval
            bvals from merged images
        merged_bvec
            bvecs from merged images
        noise_image
            image(s) created by ``dwidenoise``
        original_files
            names of the original files for each volume
        qc_summary
            DSI Studio QC text file

    """

    workflow = Workflow(name=name)
    outputnode = pe.Node(
        niu.IdentityInterface(fields=[
            'merged_image', 'merged_raw_image', 'merged_bval', 'merged_bvec', 'noise_images',
            'bias_images', 'denoising_confounds', 'original_files', 'qc_summary',
            'validation_reports']),
        name='outputnode')
    desc = []

    # DWIs will be merged at some point.
    merge_dwis = pe.Node(
        MergeDWIs(bids_dwi_files=raw_dwi_files,
                  b0_threshold=b0_threshold,
                  harmonize_b0_intensities=not no_b0_harmonization),
        name='merge_dwis')
    # Create a denoising workflow for each input image
    num_dwis = len(raw_dwi_files)
    if num_dwis > 1:
        if denoise_before_combining:
            order = "on individual DWI series before concatenation"
        else:
            order = "on the concatenated DWI series"
        desc.append("A total of %d DWI series in the %s distortion group were "
                    "concatenated, with preprocessing operations performed %s."
                    % (num_dwis, phase_id, order))
    workflow.__desc__ = " ".join(desc)
    conformed_bvals = pe.Node(niu.Merge(num_dwis), name='conformed_bvals')
    conformed_bvecs = pe.Node(niu.Merge(num_dwis), name='conformed_bvecs')
    conformed_images = pe.Node(niu.Merge(num_dwis), name='conformed_images')
    conformed_raw_images = pe.Node(niu.Merge(num_dwis), name='conformed_raw_images')
    conformation_reports = pe.Node(niu.Merge(num_dwis), name='conformation_reports')
    # derivatives from denoising
    denoising_confounds = pe.Node(niu.Merge(num_dwis), name='denoising_confounds')
    noise_images = pe.Node(niu.Merge(num_dwis), name='noise_images')
    bias_images = pe.Node(niu.Merge(num_dwis), name='bias_images')
    # Collect conformers and denoisers
    conformers = []
    denoising_wfs = []
    for dwi_num, dwi_file in enumerate(raw_dwi_files, start=1):
        # Conform each image to the requested orientation
        conformers.append(
            pe.Node(ConformDwi(orientation=orientation),
                    name="conform_dwis%02d" % dwi_num))
        conformers[-1].inputs.dwi_file = dwi_file

        if denoise_before_combining:
            # Build the denoising workflow
            _, fname, _ = split_filename(dwi_file)
            wf_name = _get_wf_name(fname).replace('preproc', 'denoise')
            denoising_wfs.append(
                init_dwi_denoising_wf(dwi_denoise_window=dwi_denoise_window,
                                      denoise_method=denoise_method,
                                      unringing_method=unringing_method,
                                      dwi_no_biascorr=dwi_no_biascorr,
                                      no_b0_harmonization=no_b0_harmonization,
                                      b0_threshold=b0_threshold,
                                      mem_gb=mem_gb,
                                      omp_nthreads=omp_nthreads,
                                      source_file=dwi_file,
                                      name=wf_name))
            workflow.connect([
                (conformers[-1], denoising_wfs[-1], [
                    ('bval_file', 'inputnode.bval_file'),
                    ('bvec_file', 'inputnode.bvec_file'),
                    ('dwi_file', 'inputnode.dwi_file')]),
                (denoising_wfs[-1], denoising_confounds, [
                    ('outputnode.confounds', 'in%d' % dwi_num)]),
                (denoising_wfs[-1], noise_images, [
                    ('outputnode.noise_image', 'in%d' % dwi_num)]),
                (denoising_wfs[-1], bias_images, [
                    ('outputnode.bias_image', 'in%d' % dwi_num)])])
            dwi_source = denoising_wfs[-1]
            edge_prefix = 'outputnode.'
        else:
            dwi_source = conformers[-1]
            edge_prefix = ''

        workflow.connect([
            (dwi_source, conformed_images, [(edge_prefix + 'dwi_file', 'in%d' % dwi_num)]),
            (conformers[-1], conformed_raw_images, [('dwi_file', 'in%d' % dwi_num)]),
            (dwi_source, conformed_bvals, [(edge_prefix + 'bval_file', 'in%d' % dwi_num)]),
            (dwi_source, conformed_bvecs, [(edge_prefix + 'bvec_file', 'in%d' % dwi_num)]),
            (conformers[-1], conformation_reports, [('out_report', 'in%d' % dwi_num)])
        ])

    # Get an orientation-conformed version of the raw inputs and their gradients
    raw_merge = pe.Node(Merge(is_dwi=True), name='raw_merge')

    # Merge the either conformed-only or conformed-and-denoised data
    workflow.connect([
        (conformed_images, merge_dwis, [('out', 'dwi_files')]),
        (conformed_raw_images, raw_merge, [('out', 'in_files')]),
        (raw_merge, outputnode, [('out_file', 'merged_raw_image')]),
        (conformed_bvals, merge_dwis, [('out', 'bval_files')]),
        (conformed_bvecs, merge_dwis, [('out', 'bvec_files')]),
        (conformation_reports, outputnode, [('out', 'validation_reports')]),
        (merge_dwis, outputnode, [
            ('original_images', 'original_files'),
            ('out_bval', 'merged_bval'),
            ('out_bvec', 'merged_bvec')])])

    # Get a QC score for the raw data
    if calculate_qc:
        qc_wf = init_modelfree_qc_wf()
        workflow.connect([
            (qc_wf, outputnode, [('outputnode.qc_summary', 'qc_summary')]),
            (raw_merge, qc_wf, [('out_file', 'inputnode.dwi_file')]),
            (merge_dwis, qc_wf, [('out_bval', 'inputnode.bval_file'),
                                 ('out_bvec', 'inputnode.bvec_file')])])

    # We have denoised and combined, therefore we are done
    if denoise_before_combining:
        workflow.connect([
            (denoising_confounds, merge_dwis, [('out', 'denoising_confounds')]),
            (merge_dwis, outputnode, [
                ('out_dwi', 'merged_image'),
                ('merged_denoising_confounds', 'denoising_confounds')]),
            (noise_images, outputnode, [('out', 'noise_images')]),
            (bias_images, outputnode, [('out', 'bias_images')])
        ])
        return workflow

    # Send the merged series for denoising
    merge_confounds = pe.Node(niu.Merge(2), name="merge_confounds")
    hstack_confounds = pe.Node(StackConfounds(axis=1), name='hstack_confounds')
    denoising_wf = init_dwi_denoising_wf(
        dwi_denoise_window=dwi_denoise_window,
        denoise_method=denoise_method,
        unringing_method=unringing_method,
        dwi_no_biascorr=dwi_no_biascorr,
        no_b0_harmonization=no_b0_harmonization,
        b0_threshold=b0_threshold,
        mem_gb=mem_gb,
        omp_nthreads=omp_nthreads,
        source_file=source_file,
        name='merged_denoise')
    workflow.connect([
        (merge_dwis, denoising_wf, [
            ('out_bval', 'inputnode.bval_file'),
            ('out_dwi', 'inputnode.dwi_file'),
            ('out_bvec', 'inputnode.bvec_file')]),
        (merge_dwis, merge_confounds, [('merged_denoising_confounds', 'in1')]),
        (denoising_wf, merge_confounds, [('outputnode.confounds', 'in2')]),
        (merge_confounds, hstack_confounds, [('out', 'in_files')]),
        (hstack_confounds, outputnode, [('confounds_file', 'denoising_confounds')]),
        (denoising_wf, outputnode, [
            ('outputnode.dwi_file', 'merged_image'),
            (('outputnode.noise_image', _as_list), 'noise_images'),
            (('outputnode.bias_image', _as_list), 'bias_images')])
    ])

    return workflow


def init_dwi_denoising_wf(dwi_denoise_window,
                          denoise_method,
                          unringing_method,
                          dwi_no_biascorr,
                          no_b0_harmonization,
                          b0_threshold,
                          source_file,
                          mem_gb=1,
                          omp_nthreads=1,
                          name="denoise_wf"):

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['dwi_file', 'bval_file', 'bvec_file']),
        name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['dwi_file', 'bval_file', 'bvec_file', 'noise_image',
                                      'bias_image', 'confounds']),
        name='outputnode')
    workflow = Workflow(name=name)

    # Get IdentityInterfaces ready to hold intermediate results
    buffernodes = []

    def get_buffernode():
        num_buffers = len(buffernodes)
        return pe.Node(niu.IdentityInterface(fields=['dwi_file']),
                       name='buffer%02d' % num_buffers)

    buffernodes.append(get_buffernode())
    workflow.connect([
        (inputnode, buffernodes[-1], [('dwi_file', 'dwi_file')]),
        (inputnode, outputnode, [
            ('bval_file', 'bval_file'),
            ('bvec_file', 'bvec_file')])])

    # Which steps to apply?
    do_denoise = denoise_method in ('patch2self', 'dwidenoise')
    do_unringing = unringing_method == 'mrdegibbs'
    do_biascorr = not dwi_no_biascorr
    harmonize_b0s = not no_b0_harmonization
    # How many steps in the denoising pipeline
    num_steps = sum(map(int, [do_denoise, do_unringing, do_biascorr, harmonize_b0s]))
    merge_confounds = pe.Node(niu.Merge(num_steps), name='merge_confounds')

    # Add the steps
    step_num = 1  # Merge inputs start at 1
    if do_denoise:
        if denoise_method == 'dwidenoise':
            denoiser = pe.Node(
                DWIDenoise(extent=(dwi_denoise_window, dwi_denoise_window,
                                   dwi_denoise_window)),
                name='denoiser')
        else:
            denoiser = pe.Node(
                Patch2Self(patch_radius=dwi_denoise_window),
                name='denoiser')
            workflow.connect([
                (inputnode, denoiser, [('bval_file', 'bval_file')])])
        ds_report_denoising = pe.Node(
            DerivativesDataSink(suffix=name + '_denoising',
                                source_file=source_file),
            name='ds_report_' + name + '_denoising',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        buffernodes.append(get_buffernode())
        workflow.connect([
            (buffernodes[-2], denoiser, [('dwi_file', 'in_file')]),
            (denoiser, ds_report_denoising, [('out_report', 'in_file')]),
            (denoiser, buffernodes[-1], [('out_file', 'dwi_file')]),
            (denoiser, merge_confounds, [('nmse_text', 'in%d' % step_num)])
        ])
        step_num += 1

    if do_unringing:
        if unringing_method == 'mrdegibbs':
            degibbser = pe.Node(MRDeGibbs(), name='degibbser')
        ds_report_unringing = pe.Node(
            DerivativesDataSink(suffix=name + '_unringing',
                                source_file=source_file),
            name='ds_report_' + name + '_unringing',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        buffernodes.append(get_buffernode())
        workflow.connect([
            (buffernodes[-2], degibbser, [('dwi_file', 'in_file')]),
            (degibbser, ds_report_unringing, [('out_report', 'in_file')]),
            (degibbser, buffernodes[-1], [('out_file', 'dwi_file')]),
            (degibbser, merge_confounds, [('nmse_text', 'in%d' % step_num)])
        ])
        step_num += 1

    if do_biascorr:
        biascorr = pe.Node(DWIBiasCorrect(use_ants=True), name='biascorr')
        ds_report_biascorr = pe.Node(
            DerivativesDataSink(suffix=name + '_biascorr',
                                source_file=source_file),
            name='ds_report_' + name + '_biascorr',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB)
        get_b0s = pe.Node(ExtractB0s(b0_threshold=b0_threshold), name='get_b0s')
        quick_mask = pe.Node(MaskEPI(lower_cutoff=0.02), name='quick_mask')
        buffernodes.append(get_buffernode())
        workflow.connect([
            (buffernodes[-2], biascorr, [('dwi_file', 'in_file')]),
            (buffernodes[-2], get_b0s, [('dwi_file', 'dwi_series')]),
            (inputnode, get_b0s, [('bval_file', 'bval_file')]),
            (get_b0s, quick_mask, [('b0_series', 'in_files')]),
            (quick_mask, biascorr, [('out_mask', 'mask')]),
            (biascorr, buffernodes[-1], [('out_file', 'dwi_file')]),
            (biascorr, ds_report_biascorr, [('out_report', 'in_file')]),
            (biascorr, merge_confounds, [('nmse_text', 'in%d' % step_num)]),
            (inputnode, biascorr, [('bval_file', 'in_bval'), ('bvec_file', 'in_bvec')])
        ])
        step_num += 1

    workflow.connect([
        (buffernodes[-1], outputnode, [('dwi_file', 'dwi_file')])])

    # If any denoising operations were run, collect their confounds
    if step_num > 1:
        hstack_confounds = pe.Node(StackConfounds(axis=1), name='hstack_confounds')
        workflow.connect([
            (merge_confounds, hstack_confounds, [('out', 'in_files')]),
            (hstack_confounds, outputnode, [('confounds_file', 'confounds')])])

    return workflow


def _as_list(item):
    return [item]


def gen_denoising_boilerplate(denoise_method,
                              dwi_denoise_window,
                              unringing_method,
                              dwi_no_biascorr,
                              no_b0_harmonization,
                              b0_threshold):
    """Generates methods boilerplate for the denoising workflow."""
    desc = ["Any images with a b-value less than %d s/mm^2 were treated as a "
            "*b*=0 image." % b0_threshold]
    do_denoise = denoise_method in ('dwidenoise', 'patch2self')
    do_unringing = unringing_method == 'mrdegibbs'
    do_biascorr = not dwi_no_biascorr
    harmonize_b0s = not no_b0_harmonization
    last_step = ""
    if do_denoise:
        if denoise_method == 'dwidenoise':
            desc.append("MP-PCA denoising as implemented in MRtrix3's `dwidenoise`"
                        "[@dwidenoise1] was applied with "
                        "a %d-voxel window." % dwi_denoise_window)
            last_step = "After MP-PCA, "
        if denoise_method == 'patch2self':
            desc.append("Denoising using `patch2self` "
                        "[@patch2self] was applied")
            if dwi_denoise_window == "auto":
                desc.append("with settings based on developer recommendations.")
            else:
                desc.append("with a %d-voxel window." % dwi_denoise_window)
            last_step = "After `patch2self`, "
    if do_unringing:
        unringing_txt = {
            "mrdegibbs": "MRtrix3's `mrdegibbs` [@mrdegibbs].",
            "dipy": "Dipy [@dipy]."
        }[unringing_method]

        desc.append(last_step + "Gibbs unringing was performed using "
                    + unringing_txt)
        last_step = "Following unringing, "

    if do_biascorr:
        desc.append(last_step + "B1 field inhomogeneity was corrected using "
                    "`dwibiascorrect` from MRtrix3 with the N4 algorithm "
                    "[@n4].")
        last_step = "After B1 bias correction, "

    if harmonize_b0s:
        desc.append(last_step + "the mean intensity of the DWI series was adjusted "
                    "so all the mean intensity of the b=0 images matched across each"
                    "separate DWI scanning sequence.")
        last_step = True

    if not last_step:
        return "No denoising steps were applied to the DWI data."

    return " ".join(desc)
