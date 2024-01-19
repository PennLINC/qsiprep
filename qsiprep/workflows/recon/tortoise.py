"""
MRTrix workflows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Note that in nipype interfaces the threading-controlling attribute is
``nthreads``, not the typical ``num_threads`` expected by nipype.
To keep threading consistent between nipype and mrtrix, the
``nthreads`` attribute needs to be set in the interface and the
``n_procs`` attribute needs to be set on the Node.


.. autofunction:: init_mrtrix_csd_recon_wf

"""
import logging
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
from ...interfaces.bids import ReconDerivativesDataSink
from qsiprep.interfaces.tortoise import (
    EstimateMAPMRI, EstimateTensor)
from ...interfaces.interchange import recon_workflow_input_fields
from ...engine import Workflow

LOGGER = logging.getLogger('nipype.interface')

CITATIONS = {
    "dhollander": "(@dhollander2019response, @dhollander2016unsupervised)",
    "msmt_5tt": "(@msmt5tt)",
    "csd": "(@originalcsd, @tournier2007robust)",
    "msmt_csd": "(@originalcsd, @msmt5tt)"
}


def init_tortoise_mapmri_wf(omp_nthreads,
                            available_anatomical_data,
                            name="tortoise_mapmri_recon",
                            output_suffix="", params={}):
    """Estimate MAPMRI using TORTOISE.

    This workflow uses mrtrix tools to run csd on multishell data. At the end,
    mtnormalise is run.

    Inputs

        *Default qsiprep inputs*

        qsiprep_5tt_hsvs
            A hybrid surface volume segmentation 5tt image aligned with the
            QSIPrep T1w


    Outputs

        wm_txt
            SH fiber response function for white matter
        wm_fod
            FOD SH coefficients for white matter
        gm_txt
            SH fiber response function for gray matter
        gm_fod
            FOD SH coefficients for gray matter
        csf_txt
            SH fiber response function for CSF
        csf_fod
            FOD SH coefficients for CSF
        fod_sh_mif
            The same file as wm_fod.


    Params

        response: dict
            parameters for estimating the response function. A minimal example would be
            ``{"algorighm": "dhollander"}``
        fod: dict
            parameters for dwi2fod. A minimal example would be
            ``{"algorithm": "msmt_csd", "max_sh": [6, 8, 8]}``.
        mtnormalize: bool
            Should the FODs be mtnormalized?


    """
    inputnode = pe.Node(
        niu.IdentityInterface(fields=recon_workflow_input_fields),
        name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['fod_sh_mif', 'wm_odf', 'wm_txt', 'gm_odf', 'gm_txt', 'csf_odf',
                    'csf_txt']),
        name="outputnode")
    workflow = Workflow(name=name)
    plot_reports = params.pop("plot_reports", True)
    desc = """MRtrix3 Reconstruction

: """

    # Response estimation
    response = params.get('response', {})
    response_algorithm = response.get('algorithm', 'dhollander')

    if response_algorithm == "fast":
        response_algorithm = "dhollander"

    response['algorithm'] = response_algorithm
    response['nthreads'] = omp_nthreads
    if response_algorithm == 'csd':
        desc += 'Single-tissue '
    else:
        desc += 'Multi-tissue '
    LOGGER.info("Response configuration: %s", response)

    desc += """\
fiber response functions were estimated using the {} algorithm.
FODs were estimated via constrained spherical deconvolution
(CSD, @originalcsd, @tournier2008csd) \
""".format(response_algorithm)
    if response_algorithm == 'msmt_5tt':
        desc += 'using a T1w-based segmentation {}.'.format(CITATIONS[response_algorithm])
    else:
        desc += 'using an unsupervised multi-tissue method {}.'.format(
            CITATIONS[response_algorithm])

    # FOD estimation
    fod = params.get('fod', {})
    fod_algorithm = fod.get('algorithm', 'msmt_csd')
    fod['algorithm'] = fod_algorithm
    fod['nthreads'] = omp_nthreads
    LOGGER.info("Using %d threads in MRtrix3", omp_nthreads)
    using_multitissue = fod_algorithm in ('ss3t', 'msmt_csd')

    # Intensity normalize?
    run_mtnormalize = params.get('mtnormalize', True) and using_multitissue

    create_mif = pe.Node(MRTrixIngress(), name='create_mif')
    method_5tt = response.pop("method_5tt","dhollander")
    # Use dwi2response from 3Tissue for updated dhollander
    estimate_response = pe.Node(
        SS3TDwi2Response(**response),
        name='estimate_response',
        n_procs=omp_nthreads)

    if response_algorithm == 'msmt_5tt':
        if method_5tt == "hsvs":
            workflow.connect([
                (inputnode, estimate_response, [('qsiprep_5tt_hsvs', 'mtt_file')])])
        else:
            raise Exception("Unrecognized 5tt method: " + method_5tt)

    if fod_algorithm in ('msmt_csd', 'csd'):
        estimate_fod = pe.Node(
            EstimateFOD(**fod),
            name='estimate_fod',
            n_procs=omp_nthreads)
        desc += ' Reconstruction was done using MRtrix3 (@mrtrix3).'
    elif fod_algorithm == 'ss3t':
        estimate_fod = pe.Node(
            SS3TEstimateFOD(**fod),
            name='estimate_fod',
            n_procs=omp_nthreads)
        desc += """ \
A single-shell-optimized multi-tissue CSD was performed using MRtrix3Tissue
(https://3Tissue.github.io), a fork of MRtrix3 (@mrtrix3)"""

    workflow.connect([
        (estimate_response, estimate_fod, [('wm_file', 'wm_txt'),
                                           ('gm_file', 'gm_txt'),
                                           ('csf_file', 'csf_txt')]),
        (inputnode, create_mif, [('dwi_file', 'dwi_file'),
                                 ('bval_file', 'bval_file'),
                                 ('bvec_file', 'bvec_file'),
                                 ('b_file', 'b_file')]),
        (create_mif, estimate_fod, [('mif_file', 'in_file')]),
        (inputnode, estimate_fod, [('dwi_mask', 'mask_file')]),
        (create_mif, estimate_response, [('mif_file', 'in_file')]),
        (estimate_response, outputnode, [('wm_file', 'wm_txt'),
                                         ('gm_file', 'gm_txt'),
                                         ('csf_file', 'csf_txt')]),
        (inputnode, estimate_response, [('dwi_mask', 'in_mask')])])


    if not run_mtnormalize:
        workflow.connect([
            (estimate_fod, plot_peaks, [('wm_odf', 'mif_file')]),
            (estimate_fod, outputnode, [('wm_odf', 'fod_sh_mif'),
                                        ('wm_odf', 'wm_odf'),
                                        ('gm_odf', 'gm_odf'),
                                        ('csf_odf', 'csf_odf')])])
    else:
        intensity_norm = pe.Node(
            MTNormalize(
                nthreads=omp_nthreads,
                inlier_mask='inliers.nii.gz',
                norm_image='norm.nii.gz'),
            name='intensity_norm',
            n_procs=omp_nthreads)
        workflow.connect([
            (inputnode, intensity_norm, [('dwi_mask', 'mask_file')]),
            (estimate_fod, intensity_norm, [('wm_odf', 'wm_odf'),
                                            ('gm_odf', 'gm_odf'),
                                            ('csf_odf', 'csf_odf')]),
            (intensity_norm, outputnode, [('wm_normed_odf', 'fod_sh_mif'),
                                          ('wm_normed_odf', 'wm_odf'),
                                          ('gm_normed_odf', 'gm_odf'),
                                          ('csf_normed_odf', 'csf_odf')])])
        desc += " FODs were intensity-normalized using mtnormalize (@mtnormalize)."

    if plot_reports:
        # Make a visual report of the model
        plot_peaks = pe.Node(
            CLIReconPeaksReport(),
            name='plot_peaks',
            n_procs=omp_nthreads)
        ds_report_peaks = pe.Node(
            ReconDerivativesDataSink(extension='.png',
                                    desc="wmFOD",
                                    suffix='peaks'),
            name='ds_report_peaks',
            run_without_submitting=True)
        workflow.connect([
            (inputnode, plot_peaks, [('dwi_ref', 'background_image'),
                                    ('odf_rois', 'odf_rois'),
                                    ('dwi_mask', 'mask_file')]),
            (plot_peaks, ds_report_peaks, [('peak_report', 'in_file')])])

        # Plot targeted regions
        if available_anatomical_data['has_qsiprep_t1w_transforms']:
            ds_report_odfs = pe.Node(
                ReconDerivativesDataSink(extension='.png',
                                        desc="wmFOD",
                                        suffix='odfs'),
                name='ds_report_odfs',
                run_without_submitting=True)
            workflow.connect(plot_peaks, 'odf_report', ds_report_odfs, 'in_file')

        fod_source, fod_key = (estimate_fod, "wm_odf") if not run_mtnormalize \
            else (intensity_norm, "wm_normed_odf")
        workflow.connect(fod_source, fod_key, plot_peaks, "mif_file")

    if output_suffix:
        normed = '' if not run_mtnormalize else 'mtnormed'
        ds_wm_odf = pe.Node(
            ReconDerivativesDataSink(extension='.mif.gz',
                                     desc="wmFOD" + normed,
                                     suffix=output_suffix,
                                     compress=True),
            name='ds_wm_odf',
            run_without_submitting=True)
        workflow.connect(outputnode, 'wm_odf', ds_wm_odf, 'in_file')
        ds_wm_txt = pe.Node(
            ReconDerivativesDataSink(extension='.txt',
                                     desc="wmFOD",
                                     suffix=output_suffix),
            name='ds_wm_txt',
            run_without_submitting=True)
        workflow.connect(outputnode, 'wm_txt', ds_wm_txt, 'in_file')

        # If multitissue write out FODs for csf, gm
        if using_multitissue:
            ds_gm_odf = pe.Node(
                ReconDerivativesDataSink(extension='.mif.gz',
                                         desc="gmFOD" + normed,
                                         suffix=output_suffix,
                                         compress=True),
                name='ds_gm_odf',
                run_without_submitting=True)
            workflow.connect(outputnode, 'gm_odf', ds_gm_odf, 'in_file')
            ds_gm_txt = pe.Node(
                ReconDerivativesDataSink(extension='.txt',
                                         desc="gmFOD",
                                         suffix=output_suffix),
                name='ds_gm_txt',
                run_without_submitting=True)
            workflow.connect(outputnode, 'gm_txt', ds_gm_txt, 'in_file')

            ds_csf_odf = pe.Node(
                ReconDerivativesDataSink(extension='.mif.gz',
                                         desc="csfFOD" + normed,
                                         suffix=output_suffix,
                                         compress=True),
                name='ds_csf_odf',
                run_without_submitting=True)
            workflow.connect(outputnode, 'csf_odf', ds_csf_odf, 'in_file')
            ds_csf_txt = pe.Node(
                ReconDerivativesDataSink(extension='.txt',
                                         desc="csfFOD",
                                         suffix=output_suffix),
                name='ds_csf_txt',
                run_without_submitting=True)
            workflow.connect(outputnode, 'csf_txt', ds_csf_txt, 'in_file')

            if run_mtnormalize:
                ds_mt_norm = pe.Node(
                    ReconDerivativesDataSink(extension='.mif.gz',
                                             desc="mtnorm",
                                             suffix=output_suffix,
                                             compress=True),
                    name='ds_mt_norm',
                    run_without_submitting=True)
                workflow.connect(intensity_norm, 'norm_image', ds_mt_norm, 'in_file')
                ds_inlier_mask = pe.Node(
                    ReconDerivativesDataSink(extension='.mif.gz',
                                             desc="mtinliermask",
                                             suffix=output_suffix,
                                             compress=True),
                    name='ds_inlier_mask',
                    run_without_submitting=True)
                workflow.connect(intensity_norm, 'inlier_mask', ds_inlier_mask, 'in_file')

    workflow.__desc__ = desc
    return workflow