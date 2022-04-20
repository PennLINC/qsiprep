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
from nipype.interfaces import afni
from ...interfaces.bids import ReconDerivativesDataSink
from ...interfaces.reports import CLIReconPeaksReport, ConnectivityReport
from qsiprep.interfaces.mrtrix import (
    EstimateFOD, SS3TEstimateFOD, MRTrixIngress, SS3TDwi2Response, GlobalTractography,
    MRTrixAtlasGraph, SIFT2, TckGen, MTNormalize)
from .interchange import recon_workflow_input_fields
from ...engine import Workflow

LOGGER = logging.getLogger('nipype.interface')
MULTI_RESPONSE_ALGORITHMS = ('dhollander', 'msmt_5tt')

CITATIONS = {
    "dhollander": "(@dhollander2019response, @dhollander2016unsupervised)",
    "msmt_5tt": "(@msmt5tt)",
    "csd": "(@originalcsd, @tournier2007robust)",
    "msmt_csd": "(@originalcsd, @msmt5tt)"
}


def init_mrtrix_csd_recon_wf(omp_nthreads, available_anatomical_data, name="mrtrix_recon",
                             output_suffix="", params={}):
    """Create FOD images for WM, GM and CSF.

    This workflow uses mrtrix tools to run csd on multishell data. At the end,
    mtnormalise is run.

    Inputs

        *Default qsiprep inputs*

        qsiprep_5tt_hsvs
            A hybrid surface volume segmentation 5tt image aligned with the 
            QSIPrep T1w
        
        qsiprep_5tt_fast
            A FSL-FAST-based 5tt image aligned with the QSIPrep T1w image

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
    method_5tt = response.pop("method_5tt", "fast")
    # Use dwi2response from 3Tissue for updated dhollander
    estimate_response = pe.Node(
        SS3TDwi2Response(**response), 
        name='estimate_response',
        n_procs=omp_nthreads)

    if response_algorithm == 'msmt_5tt':
        if method_5tt == "hsvs":
            workflow.connect([
                (inputnode, estimate_response, [('qsiprep_5tt_hsvs', 'mtt_file')])])
        elif method_5tt == "fast":
            workflow.connect([
                (inputnode, estimate_response, [('qsiprep_5tt_fast', 'mtt_file')])])
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


def init_global_tractography_wf(omp_nthreads, available_anatomical_data, name="mrtrix_recon",
                                output_suffix="", params={}):
    """Run multi-shell, multi-tissue global tractography

    This workflow uses mrtrix tools to run csd on multishell data.

    Inputs

        dwi_file
            Preprocessed DWI series
        wm_txt
            SH fiber response function for white matter
        gm_txt
            SH fiber response function for gray matter
        csf_txt
            SH fiber response function for CSF

    Outputs

        global_wm_fod
            FOD SH image enhanced by global tractography
        global_iso_fod
            FOD SH coefficients for other tissue compartments.
        l1_penalty
             the residual data energy image, including the L1-penalty imposed
             by the particle potential


    """
    inputnode = pe.Node(
        niu.IdentityInterface(fields=recon_workflow_input_fields + ['gm_txt', 'wm_txt', 'csf_txt']),
        name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['fod_sh_mif', 'wm_odf', 'iso_fraction', 'tck_file']),
        name="outputnode")

    workflow = pe.Workflow(name=name)
    plot_reports = params.pop("plot_reports", True)

    create_mif = pe.Node(MRTrixIngress(), name='create_mif')

    # Resample anat mask
    tck_global = pe.Node(GlobalTractography(**params), name='tck_global')
    workflow.connect([
        (inputnode, create_mif, [('dwi_file', 'dwi_file'),
                                 ('bval_file', 'bval_file'),
                                 ('bvec_file', 'bvec_file'),
                                 ('b_file', 'b_file')]),
        (create_mif, tck_global, [('mif_file', 'dwi_file')]),
        (inputnode, tck_global, [('dwi_mask', 'mask')]),
        (inputnode, tck_global, [("wm_txt", "wm_txt"),
                                 ("gm_txt", "gm_txt"),
                                 ("csf_txt", "csf_txt")]),
        (tck_global, outputnode, [("wm_odf", "wm_odf"),
                                  ("isotropic_fraction", "isotropic_fraction"),
                                  ("tck_file", "tck_file"),
                                  ("residual_energy", "residual_energy"),
                                  ("wm_odf", "fod_sh_mif")])
        ])

    if output_suffix:
        ds_globalwm_odf = pe.Node(
            ReconDerivativesDataSink(extension='.mif.gz',
                                     desc="globalwmFOD",
                                     suffix=output_suffix,
                                     compress=True),
            name='ds_globalwm_odf',
            run_without_submitting=True)
        workflow.connect(outputnode, 'wm_odf', ds_globalwm_odf, 'in_file')

        ds_isotropic_fraction = pe.Node(
            ReconDerivativesDataSink(extension='.mif.gz',
                                     desc="ISOfraction",
                                     suffix=output_suffix),
            name='ds_isotropic_fraction',
            run_without_submitting=True)
        workflow.connect(outputnode, 'isotropic_fraction', ds_isotropic_fraction, 'in_file')

        ds_tck_file = pe.Node(
            ReconDerivativesDataSink(extension='.tck.gz',
                                     desc="global",
                                     suffix=output_suffix,
                                     compress=True),
            name='ds_tck_file',
            run_without_submitting=True)
        workflow.connect(outputnode, 'tck_file', ds_tck_file, 'in_file')

        ds_residual_energy = pe.Node(
            ReconDerivativesDataSink(extension='.tck.gz',
                                     desc="residualEnergy",
                                     suffix=output_suffix,
                                     compress=True),
            name='ds_residual_energy',
            run_without_submitting=True)
        workflow.connect(outputnode, 'residual_energy', ds_residual_energy, 'in_file')

    return workflow


def init_mrtrix_tractography_wf(omp_nthreads, available_anatomical_data, name="mrtrix_tracking",
                                output_suffix="", params={}):
    """Run tractography

    This workflow uses mrtrix tools to run csd on multishell data.

    Inputs

        fod_sh_mif
            mif file containing spherical harmonics for tractography

    Outputs

        global_wm_fod
            FOD SH image enhanced by global tractography
        global_iso_fod
            FOD SH coefficients for other tissue compartments.
        l1_penalty
             the residual data energy image, including the L1-penalty imposed
             by the particle potential


    """
    inputnode = pe.Node(
        niu.IdentityInterface(fields=recon_workflow_input_fields + ['fod_sh_mif']),
        name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['tck_file', 'sift_weights']),
        name="outputnode")

    workflow = pe.Workflow(name=name)
    plot_reports = params.pop("plot_reports", True)
    # Resample anat mask
    tracking_params = params.get("tckgen", {})
    tracking_params['nthreads'] = omp_nthreads
    use_sift2 = params.get("use_sift2", True)
    use_5tt = params.get("use_5tt", False)
    sift_params = params.get("sift2", {})
    sift_params['nthreads'] = omp_nthreads
    tracking = pe.Node(
        TckGen(**tracking_params), 
        name='tractography',
        n_procs=omp_nthreads)
    workflow.connect([
        (inputnode, tracking, [
            ('fod_sh_mif', 'in_file'),
            ('fod_sh_mif', 'seed_dynamic')]),
        (tracking, outputnode, [("out_file", "tck_file")])])

    # Which 5tt image should be used?
    method_5tt = params.get("method_5tt", "hsvs")
    if use_5tt:
        if method_5tt == "hsvs":
            connect_5tt = "qsiprep_5tt_hsvs"
        elif method_5tt == "fast":
            connect_5tt = 'qsiprep_5tt_fast'
        else:
            raise Exception("Unrecognized 5tt method: " + method_5tt)
        workflow.connect(inputnode, connect_5tt, tracking, 'act_file')

    if use_sift2:
        tck_sift2 = pe.Node(
            SIFT2(**sift_params), 
            name="tck_sift2",
            n_procs=omp_nthreads)
        workflow.connect([
            (inputnode, tck_sift2, [('fod_sh_mif', 'in_fod')]),
            (tracking, tck_sift2, [('out_file', 'in_tracks')]),
            (tck_sift2, outputnode, [
                ('out_mu', 'mu'),
                ('out_weights', 'sift_weights')])
        ])
        if output_suffix:
            ds_sift_weights = pe.Node(
                ReconDerivativesDataSink(extension='.csv',
                                         desc="siftweights",
                                         suffix=output_suffix),
                name='ds_sift_weights',
                run_without_submitting=True)
            workflow.connect(outputnode, 'sift_weights', ds_sift_weights, 'in_file')
        if use_5tt:
            workflow.connect(inputnode, connect_5tt, tck_sift2, "act_file")

    if output_suffix:
        ds_tck_file = pe.Node(
            ReconDerivativesDataSink(extension='.tck',
                                     desc="tracks",
                                     suffix=output_suffix),
            name='ds_tck_file',
            run_without_submitting=True)
        workflow.connect(outputnode, 'tck_file', ds_tck_file, 'in_file')

    return workflow


def init_mrtrix_connectivity_wf(omp_nthreads, available_anatomical_data, name="mrtrix_connectiity",
                                params={}, output_suffix=""):
    """Runs ``tck2connectome`` on a ``tck`` file.

    Inputs

        tck_file
            mrtrix3 tck file.

    Outputs

        matfile
            A MATLAB-format file with numerous connectivity matrices for each
            atlas.
    """
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=recon_workflow_input_fields + ['tck_file', 'sift_weights', 'atlas_configs']),
        name="inputnode")
    outputnode = pe.Node(niu.IdentityInterface(fields=['matfile']),
                         name="outputnode")
    plot_reports = params.pop("plot_reports", True)
    workflow = pe.Workflow(name=name)
    conmat_params = params.get("tck2connectome", {})
    calc_connectivity = pe.Node(
        MRTrixAtlasGraph(tracking_params=conmat_params),
        name='calc_connectivity',
        n_procs=omp_nthreads)
    workflow.connect([
        (inputnode, calc_connectivity, [('atlas_configs', 'atlas_configs'),
                                        ('tck_file', 'in_file'),
                                        ('sift_weights', 'in_weights')]),
        
        (calc_connectivity, outputnode, [('connectivity_matfile', 'matfile')])
    ])

    if plot_reports:
        plot_connectivity = pe.Node(
            ConnectivityReport(), 
            name='plot_connectivity',
            n_procs=omp_nthreads)
        ds_report_connectivity = pe.Node(
            ReconDerivativesDataSink(extension='.svg',
                                    desc="MRtrix3Connectivity",
                                    suffix='matrices'),
            name='ds_report_connectivity',
            run_without_submitting=True)
        workflow.connect([
            (calc_connectivity, plot_connectivity, [
                ('connectivity_matfile', 'connectivity_matfile')]),
            (plot_connectivity, ds_report_connectivity, [('out_report', 'in_file')])])

    if output_suffix:
        # Save the output in the outputs directory
        ds_connectivity = pe.Node(ReconDerivativesDataSink(suffix=output_suffix),
                                  name='ds_' + name,
                                  run_without_submitting=True)
        workflow.connect(calc_connectivity, 'connectivity_matfile', ds_connectivity, 'in_file')

    return workflow
