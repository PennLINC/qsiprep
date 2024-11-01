# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Merge and denoise dwi images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_dwi_preproc_wf
.. autofunction:: init_dwi_derivatives_wf

"""
import pandas as pd
from bids.layout import Query
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from nipype.utils.filemanip import split_filename

from ... import config
from ...engine import Workflow
from ...interfaces import ConformDwi, DerivativesDataSink
from ...interfaces.dipy import Patch2Self
from ...interfaces.dwi_merge import MergeDWIs, PhaseToRad, StackConfounds
from ...interfaces.gradients import ExtractB0s
from ...interfaces.mrtrix import (
    ComplexToMagnitude,
    DWIBiasCorrect,
    DWIDenoise,
    MRDeGibbs,
    PolarToComplex,
)
from ...interfaces.nilearn import MaskEPI, Merge
from ...interfaces.tortoise import Gibbs
from ...utils.bids import IMPORTANT_DWI_FIELDS, update_metadata_from_nifti_header
from .qc import init_modelfree_qc_wf
from .util import _get_wf_name

DEFAULT_MEMORY_MIN_GB = 0.01


def init_merge_and_denoise_wf(
    raw_dwi_files,
    orientation,
    source_file,
    do_biascorr,
    calculate_qc=False,
    phase_id="same",
    name="merge_and_denoise_wf",
):
    """

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from qsiprep.workflows.dwi import init_merge_and_denoise_wf
        wf = init_merge_and_dwnoise_wf(
            ['/path/to/dwi/sub-1_dwi.nii.gz'],
            source_file='/data/sub-1/dwi/sub-1_dwi.nii.gz',
        )

    Parameters
    ----------
    raw_dwi_files : list
        list of raw (in their original BIDS directory) dwi nifti files


    Outputs
    -------
    merged_image
        dwi series, conformed, denoised if requested
    merged_raw_image
        dwi series, conformed, raw
    merged_bval
        bvals from merged images
    merged_bvec
        bvecs from merged images
    merged_json
        JSON file containing slice timings for slice2vol
    noise_image
        image(s) created by ``dwidenoise``
    original_files
        names of the original files for each volume
    qc_summary
        DSI Studio QC text file

    """
    workflow = Workflow(name=name)
    omp_nthreads = config.nipype.omp_nthreads
    denoise_before_combining = not config.workflow.denoise_after_combining
    layout = config.execution.layout
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "merged_image",
                "merged_raw_image",
                "merged_bval",
                "merged_bvec",
                "merged_json",
                "noise_images",
                "bias_images",
                "denoising_confounds",
                "original_files",
                "qc_summary",
                "validation_reports",
            ]
        ),
        name="outputnode",
    )
    desc = []

    # DWIs will be merged at some point.
    merge_dwis = pe.Node(
        MergeDWIs(
            bids_dwi_files=raw_dwi_files,
            b0_threshold=config.workflow.b0_threshold,
            harmonize_b0_intensities=not config.workflow.no_b0_harmonization,
            scan_metadata={
                scan: config.execution.layout.get_metadata(scan) for scan in raw_dwi_files
            },
        ),
        name="merge_dwis",
        n_procs=omp_nthreads,
    )
    # Create a denoising workflow for each input image
    num_dwis = len(raw_dwi_files)
    if num_dwis > 1:
        if denoise_before_combining:
            order = "on individual DWI series before concatenation"
        else:
            order = "on the concatenated DWI series"
        desc.append(
            "A total of %d DWI series in the %s distortion group were "
            "concatenated, with preprocessing operations performed %s."
            % (num_dwis, phase_id, order)
        )
    workflow.__desc__ = " ".join(desc)
    conformed_bvals = pe.Node(niu.Merge(num_dwis), name="conformed_bvals")
    conformed_bvecs = pe.Node(niu.Merge(num_dwis), name="conformed_bvecs")
    conformed_images = pe.Node(niu.Merge(num_dwis), name="conformed_images")
    conformed_raw_images = pe.Node(niu.Merge(num_dwis), name="conformed_raw_images")
    conformation_reports = pe.Node(niu.Merge(num_dwis), name="conformation_reports")
    # derivatives from denoising
    denoising_confounds = pe.Node(niu.Merge(num_dwis), name="denoising_confounds")
    noise_images = pe.Node(niu.Merge(num_dwis), name="noise_images")
    bias_images = pe.Node(niu.Merge(num_dwis), name="bias_images")
    # Collect conformers and denoisers
    conformers = []
    denoising_wfs = []

    # Get a data frame of the raw_dwi_files and their imaging parameters:
    dwi_df = get_acq_parameters_df(raw_dwi_files, layout=layout)
    for dwi_num, row in dwi_df.iterrows():
        dwi_num += 1  # start at 1
        dwi_file = row.BIDSFile

        # Conform each image to the requested orientation
        conformers.append(
            pe.Node(ConformDwi(orientation=orientation), name="conform_dwis%02d" % dwi_num),
        )
        conformers[-1].inputs.dwi_file = dwi_file

        if denoise_before_combining:
            # Build the denoising workflow
            _, fname, _ = split_filename(dwi_file)
            wf_name = _get_wf_name(fname).replace("preproc", "denoise")

            # Set up a strict query for a phase file based on the magnitude file.
            all_entities = layout.get_entities(metadata=False)
            # No other non-matching entities allowed
            query = {k: Query.NONE for k in all_entities.keys()}
            query.update(layout.get_file(dwi_file).get_entities())
            query["part"] = "phase"
            phase_files = layout.get(**query)
            phase_available = False
            if len(phase_files) == 1:
                phase_available = True
                config.loggers.workflow.info("Phase file found for %s", dwi_file)
                phase_file = phase_files[0].path

            use_phase = phase_available and "phase" not in config.workflow.ignore
            if use_phase:
                conform_phase = pe.Node(
                    ConformDwi(
                        orientation=orientation,
                        dwi_file=phase_file,
                    ),
                    name=f"conform_phase{dwi_num}",
                )

            denoising_wfs.append(
                init_dwi_denoising_wf(
                    partial_fourier=row.PartialFourier,
                    phase_encoding_direction=row.PhaseEncodingAxis,
                    source_file=dwi_file,
                    use_phase=use_phase,
                    do_biascorr=do_biascorr,
                    name=wf_name,
                ),
            )
            workflow.connect([
                (conformers[-1], denoising_wfs[-1], [
                    ('bval_file', 'inputnode.bval_file'),
                    ('bvec_file', 'inputnode.bvec_file'),
                    ('dwi_file', 'inputnode.dwi_file'),
                ]),
                (denoising_wfs[-1], denoising_confounds, [
                    ('outputnode.confounds', f'in{dwi_num}'),
                ]),
                (denoising_wfs[-1], noise_images, [('outputnode.noise_image', f'in{dwi_num}')]),
                (denoising_wfs[-1], bias_images, [('outputnode.bias_image', f'in{dwi_num}')]),
            ])  # fmt:skip

            if use_phase:
                workflow.connect([
                    (conform_phase, denoising_wfs[-1], [('dwi_file', 'inputnode.dwi_phase_file')]),
                ])  # fmt:skip

            dwi_source = denoising_wfs[-1]
            edge_prefix = "outputnode."
        else:
            dwi_source = conformers[-1]
            edge_prefix = ""

        workflow.connect([
            (dwi_source, conformed_images, [(f'{edge_prefix}dwi_file', f'in{dwi_num}')]),
            (conformers[-1], conformed_raw_images, [('dwi_file', f'in{dwi_num}')]),
            (dwi_source, conformed_bvals, [(f'{edge_prefix}bval_file', f'in{dwi_num}')]),
            (dwi_source, conformed_bvecs, [(f'{edge_prefix}bvec_file', f'in{dwi_num}')]),
            (conformers[-1], conformation_reports, [('out_report', f'in{dwi_num}')]),
        ])  # fmt:skip

    # Get an orientation-conformed version of the raw inputs and their gradients
    raw_merge = pe.Node(Merge(is_dwi=True), name="raw_merge", n_procs=omp_nthreads)

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
            ('out_bvec', 'merged_bvec'),
            ('merged_metadata', 'merged_json')
        ]),
    ])  # fmt:skip

    # Get a QC score for the raw data
    if calculate_qc:
        qc_wf = init_modelfree_qc_wf(
            bvec_convention="DIPY" if orientation == "LPS" else "FSL",
        )
        workflow.connect([
            (qc_wf, outputnode, [('outputnode.qc_summary', 'qc_summary')]),
            (raw_merge, qc_wf, [('out_file', 'inputnode.dwi_file')]),
            (merge_dwis, qc_wf, [
                ('out_bval', 'inputnode.bval_file'),
                ('out_bvec', 'inputnode.bvec_file'),
            ]),
        ])  # fmt:skip

    # We have denoised and combined, therefore we are done
    if denoise_before_combining:
        workflow.connect([
            (denoising_confounds, merge_dwis, [('out', 'denoising_confounds')]),
            (merge_dwis, outputnode, [
                ('out_dwi', 'merged_image'),
                ('merged_denoising_confounds', 'denoising_confounds'),
            ]),
            (noise_images, outputnode, [('out', 'noise_images')]),
            (bias_images, outputnode, [('out', 'bias_images')])
        ])  # fmt:skip

        return workflow

    # Send the merged series for denoising
    merge_confounds = pe.Node(niu.Merge(2), name="merge_confounds")
    hstack_confounds = pe.Node(StackConfounds(axis=1), name="hstack_confounds")
    denoising_wf = init_dwi_denoising_wf(
        partial_fourier=get_merged_parameter(dwi_df, "PartialFourier", "all"),
        phase_encoding_direction=get_merged_parameter(dwi_df, "PhaseEncodingAxis", "all"),
        source_file=source_file,
        use_phase=False,  # can't use phase with concatenated data
        do_biascorr=do_biascorr,
        name="merged_denoise",
    )

    workflow.connect([
        (merge_dwis, denoising_wf, [
            ('out_bval', 'inputnode.bval_file'),
            ('out_dwi', 'inputnode.dwi_file'),
            ('out_bvec', 'inputnode.bvec_file'),
        ]),
        (merge_dwis, merge_confounds, [('merged_denoising_confounds', 'in1')]),
        (denoising_wf, merge_confounds, [('outputnode.confounds', 'in2')]),
        (merge_confounds, hstack_confounds, [('out', 'in_files')]),
        (hstack_confounds, outputnode, [('confounds_file', 'denoising_confounds')]),
        (denoising_wf, outputnode, [
            ('outputnode.dwi_file', 'merged_image'),
            (('outputnode.noise_image', _as_list), 'noise_images'),
            (('outputnode.bias_image', _as_list), 'bias_images'),
        ])
    ])  # fmt:skip

    return workflow


def init_dwi_denoising_wf(
    source_file,
    partial_fourier,
    phase_encoding_direction,
    use_phase,
    do_biascorr,
    name="denoise_wf",
):
    """Build a workflow to denoise a DWI series.

    Parameters
    ----------
    source_file : str
        path to the original dwi file
    partial_fourier : float
        fraction of k-space acquired
    phase_encoding_direction : str
        direction of phase encoding
    use_phase : bool
        True if phase data are available for the DWI scan.
        If True, and ``denoise_method`` is ``dwidenoise``, then ``dwidenoise``
        will be run on the complex-valued data.
    do_biascorr : bool
        If True run dwi_biascorrect
    name : str
        name of the workflow

    Inputs
    ------
    dwi_file
        path to the dwi file
    bval_file
        path to the bval file
    bvec_file
        path to the bvec file
    dwi_phase_file
        path to the dwi phase file (optional)

    Outputs
    -------
    dwi_file
        path to the denoised dwi file
    bval_file
        path to the denoised bval file
    bvec_file
        path to the denoised bvec file
    noise_image
        path to the noise image
    bias_image
        path to the bias image
    confounds
        path to the confounds file
    """

    inputnode = pe.Node(
        niu.IdentityInterface(fields=["dwi_file", "bval_file", "bvec_file", "dwi_phase_file"]),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "dwi_file",
                "bval_file",
                "bvec_file",
                "noise_image",
                "bias_image",
                "confounds",
            ],
        ),
        name="outputnode",
    )
    workflow = Workflow(name=name)
    omp_nthreads = config.nipype.omp_nthreads

    # Get IdentityInterfaces ready to hold intermediate results
    buffernodes = []

    def get_buffernode():
        num_buffers = len(buffernodes)
        return pe.Node(
            niu.IdentityInterface(fields=["dwi_file"]),
            name=f"buffer{num_buffers:02}",
        )

    buffernodes.append(get_buffernode())

    workflow.connect([
        # The first buffernode is the raw file
        (inputnode, buffernodes[0], [('dwi_file', 'dwi_file')]),
        # XXX: Why pass the bval and bvec files through unmodified?
        (inputnode, outputnode, [
            ('bval_file', 'bval_file'),
            ('bvec_file', 'bvec_file'),
        ]),
    ])  # fmt:skip

    # Which steps to apply?
    denoise_method = config.workflow.denoise_method
    unringing_method = config.workflow.unringing_method
    do_denoise = denoise_method in ("patch2self", "dwidenoise")
    do_unringing = config.workflow.unringing_method in ("mrdegibbs", "rpg")
    harmonize_b0s = not config.workflow.no_b0_harmonization
    # Configure the denoising window
    if (
        config.workflow.denoise_method == "dwidenoise"
    ) and config.workflow.dwi_denoise_window == "auto":
        dwi_denoise_window = 5
        config.loggers.workflow.info("Automatically using 5, 5, 5 window for dwidenoise")

    # How many steps in the denoising pipeline
    num_steps = sum(map(int, [do_denoise, do_unringing, do_biascorr, harmonize_b0s]))
    merge_confounds = pe.Node(niu.Merge(num_steps), name="merge_confounds")

    # Add the steps
    step_num = 1  # Merge inputs start at 1
    if do_denoise:
        # Add buffernode for denoised DWI
        buffernodes.append(get_buffernode())

        ds_report_denoising = pe.Node(
            DerivativesDataSink(
                datatype="figures",
                desc="denoising",
                source_file=source_file,
            ),
            name=f"ds_report_{name}_denoising",
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB,
        )

        if (denoise_method == "dwidenoise") and use_phase:
            # If there are phase files available, then we can use dwidenoise
            # on the complex-valued data.
            phase_to_radians = pe.Node(
                PhaseToRad(),
                name="phase_to_radians",
                n_procs=omp_nthreads,
            )
            workflow.connect([(inputnode, phase_to_radians, [("dwi_phase_file", "phase_file")])])

            combine_complex = pe.Node(
                PolarToComplex(),
                name="combine_complex",
                n_procs=omp_nthreads,
            )
            workflow.connect([
                (buffernodes[-2], combine_complex, [('dwi_file', 'mag_file')]),
                (phase_to_radians, combine_complex, [('phase_file', 'phase_file')]),
            ])  # fmt:skip

            denoiser = pe.Node(
                DWIDenoise(
                    extent=(dwi_denoise_window, dwi_denoise_window, dwi_denoise_window),
                    nthreads=omp_nthreads,
                ),
                name="denoiser",
                n_procs=omp_nthreads,
            )

            workflow.connect([
                (combine_complex, denoiser, [('out_file', 'in_file')]),
                (denoiser, ds_report_denoising, [('out_report', 'in_file')]),
                (denoiser, merge_confounds, [('nmse_text', f'in{step_num}')]),
            ])  # fmt:skip

            split_complex = pe.Node(
                ComplexToMagnitude(),
                name="split_complex",
                n_procs=omp_nthreads,
            )

            workflow.connect([
                (denoiser, split_complex, [('out_file', 'complex_file')]),
                (split_complex, buffernodes[-1], [('out_file', 'dwi_file')]),
            ])  # fmt:skip

        elif denoise_method == "dwidenoise":
            denoiser = pe.Node(
                DWIDenoise(
                    extent=(dwi_denoise_window, dwi_denoise_window, dwi_denoise_window),
                    nthreads=omp_nthreads,
                ),
                name="denoiser",
                n_procs=omp_nthreads,
            )
        else:
            denoiser = pe.Node(
                Patch2Self(patch_radius=dwi_denoise_window),
                name="denoiser",
                n_procs=omp_nthreads,
            )
            workflow.connect([(inputnode, denoiser, [("bval_file", "bval_file")])])

        if (denoise_method in ("dwidenoise", "patch2self")) and not use_phase:
            workflow.connect([
                (buffernodes[-2], denoiser, [('dwi_file', 'in_file')]),
                (denoiser, ds_report_denoising, [('out_report', 'in_file')]),
                (denoiser, buffernodes[-1], [('out_file', 'dwi_file')]),
                (denoiser, merge_confounds, [('nmse_text', f'in{step_num}')]),
            ])  # fmt:skip

        step_num += 1

    if do_unringing:
        if unringing_method == "mrdegibbs":
            degibbser = pe.Node(
                MRDeGibbs(nthreads=omp_nthreads),
                name="degibbser",
                n_procs=omp_nthreads,
            )
        elif unringing_method == "rpg":
            pe_code = {
                "i": 0,
                "i-": 0,
                "j": 1,
                "j-": 1,
                "k": 2,
            }.get(phase_encoding_direction)
            if pe_code is None:
                raise Exception("rpg requires an i[-],j[-] or k[-] PhaseEncodingDirection")

            degibbser = pe.Node(
                Gibbs(
                    kspace_coverage=partial_fourier,
                    phase_encoding_dir=pe_code,
                ),
                name="degibbser",
                n_procs=omp_nthreads,
            )

        ds_report_unringing = pe.Node(
            DerivativesDataSink(
                datatype="figures",
                desc="unringing",
                extension=".svg",
                source_file=source_file,
            ),
            name=f"ds_report_{name}_unringing",
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB,
        )
        # Add buffernode for unringed DWI
        buffernodes.append(get_buffernode())

        workflow.connect([
            (buffernodes[-2], degibbser, [('dwi_file', 'in_file')]),
            (degibbser, ds_report_unringing, [('out_report', 'in_file')]),
            (degibbser, buffernodes[-1], [('out_file', 'dwi_file')]),
            (degibbser, merge_confounds, [('nmse_text', f'in{step_num}')]),
        ])  # fmt:skip
        step_num += 1

    if do_biascorr:
        biascorr = pe.Node(DWIBiasCorrect(method="ants"), name="biascorr", n_procs=omp_nthreads)
        ds_report_biascorr = pe.Node(
            DerivativesDataSink(
                datatype="figures",
                desc="biascorr",
                source_file=source_file,
            ),
            name=f"ds_report_{name}_biascorr",
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB,
        )
        get_b0s = pe.Node(ExtractB0s(b0_threshold=config.workflow.b0_threshold), name="get_b0s")
        quick_mask = pe.Node(MaskEPI(lower_cutoff=0.02), name="quick_mask")

        # Add buffernode for bias-corrected DWI
        buffernodes.append(get_buffernode())

        workflow.connect([
            (buffernodes[-2], biascorr, [('dwi_file', 'in_file')]),
            (buffernodes[-2], get_b0s, [('dwi_file', 'dwi_series')]),
            (inputnode, get_b0s, [('bval_file', 'bval_file')]),
            (get_b0s, quick_mask, [('b0_series', 'in_files')]),
            (quick_mask, biascorr, [('out_mask', 'mask')]),
            (biascorr, buffernodes[-1], [('out_file', 'dwi_file')]),
            (biascorr, ds_report_biascorr, [('out_report', 'in_file')]),
            (biascorr, merge_confounds, [('nmse_text', f'in{step_num}')]),
            (inputnode, biascorr, [
                ('bval_file', 'in_bval'),
                ('bvec_file', 'in_bvec'),
            ]),
        ])  # fmt:skip
        step_num += 1

    # Connect the final buffernode (the most recent output) to the outputnode
    workflow.connect([(buffernodes[-1], outputnode, [("dwi_file", "dwi_file")])])

    # If any denoising operations were run, collect their confounds
    if step_num > 1:
        hstack_confounds = pe.Node(StackConfounds(axis=1), name="hstack_confounds")
        workflow.connect([
            (merge_confounds, hstack_confounds, [('out', 'in_files')]),
            (hstack_confounds, outputnode, [('confounds_file', 'confounds')]),
        ])  # fmt:skip

    return workflow


def _as_list(item):
    return [item]


def gen_denoising_boilerplate():
    """Generate a methods boilerplate for the denoising workflow."""

    denoise_method = config.workflow.denoise_method
    dwi_denoise_window = config.workflow.dwi_denoise_window
    unringing_method = config.workflow.unringing_method
    b1_biascorrect_stage = config.workflow.b1_biascorrect_stage
    no_b0_harmonization = config.workflow.no_b0_harmonization
    b0_threshold = config.workflow.b0_threshold
    use_phase = "phase" not in config.workflow.ignore
    desc = [
        f"Any images with a b-value less than {b0_threshold} s/mm^2 were treated as a "
        "*b*=0 image."
    ]
    do_denoise = denoise_method in ("dwidenoise", "patch2self")
    do_unringing = unringing_method in ("rpg", "mrdegibbs")
    harmonize_b0s = not no_b0_harmonization
    last_step = ""
    if do_denoise:
        if denoise_method == "dwidenoise":
            desc.append(
                "MP-PCA denoising as implemented in MRtrix3's `dwidenoise`"
                f"[@dwidenoise1] was applied with a {dwi_denoise_window}-voxel window."
            )
            last_step = "After MP-PCA, "
            if use_phase:
                desc.append(
                    "When phase data were available, this was done on complex-valued data."
                )

        elif denoise_method == "patch2self":
            desc.append("Denoising using `patch2self` [@patch2self] was applied")
            if dwi_denoise_window == "auto":
                desc.append("with settings based on developer recommendations.")
            else:
                desc.append(f"with a {dwi_denoise_window}-voxel window.")
            last_step = "After `patch2self`, "

    if do_unringing:
        unringing_txt = {
            "mrdegibbs": "MRtrix3's `mrdegibbs` [@mrdegibbs].",
            "dipy": "Dipy [@dipy].",
            "rpg": "TORTOISE's `Gibbs` [@pfgibbs].",
        }[unringing_method]

        desc.append(f"{last_step}Gibbs unringing was performed using {unringing_txt}")
        last_step = "Following unringing, "

    if b1_biascorrect_stage == "legacy":
        desc.append(
            f"{last_step}B1 field inhomogeneity was corrected using "
            "`dwibiascorrect` from MRtrix3 with the N4 algorithm [@n4]."
        )
        last_step = "After B1 bias correction, "

    if harmonize_b0s:
        desc.append(
            f"{last_step}the mean intensity of the DWI series was adjusted "
            "so all the mean intensity of the b=0 images matched across each"
            "separate DWI scanning sequence."
        )
        last_step = True

    if b1_biascorrect_stage == "final":
        desc.append(
            "B1 field inhomogeneity was corrected using "
            "`dwibiascorrect` from MRtrix3 with the N4 algorithm "
            "[@n4] after corrected images were resampled."
        )
        last_step = True

    if not last_step:
        return "No denoising steps were applied to the DWI data."

    return " ".join(desc)


def get_acq_parameters_df(dwi_file_list, layout):
    """Figure out what the"""
    file_rows = []
    for dwi_file in dwi_file_list:
        metadata = layout.get_metadata(dwi_file)
        update_metadata_from_nifti_header(metadata, dwi_file)
        metadata["BIDSFile"] = dwi_file
        file_rows.append(metadata)

    merged_acq_params = pd.DataFrame(
        file_rows,
        columns=["BIDSFile"] + IMPORTANT_DWI_FIELDS,
    )
    merged_acq_params["PhaseEncodingAxis"] = merged_acq_params[
        "PhaseEncodingDirection"
    ].str.replace("-", "")
    return merged_acq_params


def get_merged_parameter(parameter_df, parameter_name, selection_mode="all"):
    """Return a single parameter from a parameter dataframe."""
    col = parameter_df[parameter_name]
    unique_values = col.unique()
    if len(unique_values) > 1:
        config.loggers.workflow.warn(
            "Found %d unique values for %s",
            parameter_name,
            len(unique_values),
        )

    # Require that all the values are the same
    if selection_mode == "all":
        if len(unique_values) > 1:
            raise Exception(
                "More than one value for %s was found (%s): exiting!",
                parameter_name,
                str(unique_values),
            )

        return unique_values[0]

    if selection_mode == "mode":
        return col.mode()[0]

    raise Exception("selection_mode must be 'all' or 'mode'")
