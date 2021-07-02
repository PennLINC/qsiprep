"""
DSI Studio workflows
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_dsi_studio_recon_wf
.. autofunction:: init_dsi_studio_connectivity_wf
.. autofunction:: init_dsi_studio_export_wf

"""
import nipype.pipeline.engine as pe
from nipype.interfaces import afni, utility as niu
from qsiprep.interfaces.dsi_studio import (DSIStudioCreateSrc, DSIStudioGQIReconstruction,
                                           DSIStudioAtlasGraph, DSIStudioExport,
                                           DSIStudioTracking,
                                           FixDSIStudioExportHeader)

import logging
from qsiprep.interfaces.bids import ReconDerivativesDataSink
from .interchange import input_fields
from ...engine import Workflow
from ...interfaces.reports import ReconPeaksReport, ConnectivityReport

LOGGER = logging.getLogger('nipype.interface')


def init_dsi_studio_recon_wf(omp_nthreads, has_transform, name="dsi_studio_recon",
                             output_suffix="", params={}):
    """Reconstructs diffusion data using DSI Studio.

    This workflow creates a ``.src.gz`` file from the input dwi, bvals and bvecs,
    then reconstructs ODFs using GQI.

    Inputs

        *Default qsiprep inputs*

    Outputs

        fibgz
            A DSI Studio fib file containing GQI ODFs, peaks and scalar values.

    Params

        ratio_of_mean_diffusion_distance: float
            Default 1.25. Distance to sample EAP at.

    """
    inputnode = pe.Node(niu.IdentityInterface(fields=input_fields + ['odf_rois']),
                        name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['fibgz']),
        name="outputnode")
    workflow = Workflow(name=name)
    desc = """DSI Studio Reconstruction

: """
    create_src = pe.Node(DSIStudioCreateSrc(), name="create_src")
    romdd = params.get("ratio_of_mean_diffusion_distance", 1.25)
    gqi_recon = pe.Node(
        DSIStudioGQIReconstruction(ratio_of_mean_diffusion_distance=romdd),
        name="gqi_recon")
    desc += """\
Diffusion orientation distribution functions (ODFs) were reconstructed using
generalized q-sampling imaging (GQI, @yeh2010gqi) with a ratio of mean diffusion
distance of %02f.""" % romdd

    # Resample anat mask
    resample_mask = pe.Node(
        afni.Resample(outputtype='NIFTI_GZ', resample_mode="NN"), name='resample_mask')

    # Make a visual report of the model
    plot_peaks = pe.Node(ReconPeaksReport(subtract_iso=True), name='plot_peaks')
    ds_report_peaks = pe.Node(
        ReconDerivativesDataSink(extension='.png',
                                 desc="GQIODF",
                                 suffix='peaks'),
        name='ds_report_peaks',
        run_without_submitting=True)

    # Plot targeted regions
    if has_transform:
        ds_report_odfs = pe.Node(
            ReconDerivativesDataSink(extension='.png',
                                     desc="GQIODF",
                                     suffix='odfs'),
            name='ds_report_odfs',
            run_without_submitting=True)
        workflow.connect(plot_peaks, 'odf_report', ds_report_odfs, 'in_file')

    workflow.connect([
        (inputnode, create_src, [('dwi_file', 'input_nifti_file'),
                                 ('bval_file', 'input_bvals_file'),
                                 ('bvec_file', 'input_bvecs_file')]),
        (inputnode, resample_mask, [('t1_brain_mask', 'in_file'),
                                    ('dwi_file', 'master')]),
        (create_src, gqi_recon, [('output_src', 'input_src_file')]),
        (resample_mask, gqi_recon, [('out_file', 'mask')]),
        (gqi_recon, outputnode, [('output_fib', 'fibgz')]),
        (gqi_recon, plot_peaks, [('output_fib', 'fib_file')]),
        (inputnode, plot_peaks, [('dwi_ref', 'background_image'),
                                 ('odf_rois', 'odf_rois')]),
        (resample_mask, plot_peaks, [('out_file', 'mask_file')]),
        (plot_peaks, ds_report_peaks, [('out_report', 'in_file')])
    ])

    if output_suffix:
        # Save the output in the outputs directory
        ds_gqi_fibgz = pe.Node(
            ReconDerivativesDataSink(
                extension='.fib.gz',
                suffix=output_suffix,
                compress=True),
            name='ds_gqi_fibgz',
            run_without_submitting=True)
        workflow.connect(gqi_recon, 'output_fib', ds_gqi_fibgz, 'in_file')
    workflow.__desc__ = desc
    return workflow


def init_dsi_studio_tractography_wf(omp_nthreads, has_transform, name="dsi_studio_tractography",
                                    params={}, output_suffix=""):
    """Calculate streamline-based connectivity matrices using DSI Studio.

    DSI Studio has a deterministic tractography algorithm that can be used to
    estimate pairwise regional connectivity. It calculates multiple connectivity
    measures.

    Inputs

        fibgz
            A DSI Studio fib file produced by DSI Studio reconstruction.
        trk_file
            a DSI Studio trk.gz file

    Outputs

        trk_file
            A DSI-Studio format trk file
        fibgz
            The input fib file, as it is needed by downstream nodes in addition to
            the trk file.

    Params

        fiber_count
            number of streamlines to generate. Cannot also specify seed_count
        seed_count
            Number of seeds to track from. Does not guarantee a fixed number of
            streamlines and cannot be used with the fiber_count option.
        method
            0: streamline (Euler) 4: Runge Kutta
        seed_plan
            0: = traits.Enum((0, 1), argstr="--seed_plan=%d")
        initial_dir
            Seeds begin oriented as 0: the primary orientation of the ODF 1: a random orientation
            or 2: all orientations
        connectivity_type
            "pass" to count streamlines passing through a region. "end" to force
            streamlines to terminate in regions they count as connecting.
        connectivity_value
            "count", "ncount", "fa" used to quantify connection strength.
        random_seed
            Setting to True generates truly random (not-reproducible) seeding.
        fa_threshold
            If not specified, will use the DSI Studio Otsu threshold. Otherwise
            specigies the minimum qa value per fixed to be used for tracking.
        step_size
            Streamline propagation step size in millimeters.
        turning_angle
            Maximum turning angle in degrees for steamline propagation.
        smoothing
            DSI Studio smoothing factor
        min_length
            Minimum streamline length in millimeters.
        max_length
            Maximum streamline length in millimeters.

    """
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=input_fields + ['fibgz']),
        name="inputnode")
    outputnode = pe.Node(niu.IdentityInterface(fields=['trk_file', 'fibgz']),
                         name="outputnode")
    workflow = Workflow(name=name)
    tracking = pe.Node(DSIStudioTracking(nthreads=omp_nthreads, **params),
                       name='tracking')
    workflow.connect([
        (inputnode, tracking, [('fibgz', 'input_fib')]),
        (tracking, outputnode, [('output_trk', 'trk_file')]),
        (inputnode, outputnode, [('fibgz', 'fibgz')])
    ])
    if output_suffix:
        # Save the output in the outputs directory
        ds_tracking = pe.Node(ReconDerivativesDataSink(suffix=output_suffix),
                              name='ds_' + name,
                              run_without_submitting=True)
        workflow.connect(tracking, 'output_trk', ds_tracking, 'in_file')
    return workflow


def init_dsi_studio_connectivity_wf(omp_nthreads, has_transform, name="dsi_studio_connectivity",
                                    params={}, output_suffix=""):
    """Calculate streamline-based connectivity matrices using DSI Studio.

    DSI Studio has a deterministic tractography algorithm that can be used to
    estimate pairwise regional connectivity. It calculates multiple connectivity
    measures.

    Inputs

        fibgz
            A DSI Studio fib file produced by DSI Studio reconstruction.
        trk_file
            a DSI Studio trk.gz file

    Outputs

        matfile
            A MATLAB-format file with numerous connectivity matrices for each
            atlas.

    Params

        fiber_count
            number of streamlines to generate. Cannot also specify seed_count
        seed_count
            Number of seeds to track from. Does not guarantee a fixed number of
            streamlines and cannot be used with the fiber_count option.
        method
            0: streamline (Euler) 4: Runge Kutta
        seed_plan
            0: = traits.Enum((0, 1), argstr="--seed_plan=%d")
        initial_dir
            Seeds begin oriented as 0: the primary orientation of the ODF 1: a random orientation
            or 2: all orientations
        connectivity_type
            "pass" to count streamlines passing through a region. "end" to force
            streamlines to terminate in regions they count as connecting.
        connectivity_value
            "count", "ncount", "fa" used to quantify connection strength.
        random_seed
            Setting to True generates truly random (not-reproducible) seeding.
        fa_threshold
            If not specified, will use the DSI Studio Otsu threshold. Otherwise
            specigies the minimum qa value per fixed to be used for tracking.
        step_size
            Streamline propagation step size in millimeters.
        turning_angle
            Maximum turning angle in degrees for steamline propagation.
        smoothing
            DSI Studio smoothing factor
        min_length
            Minimum streamline length in millimeters.
        max_length
            Maximum streamline length in millimeters.

    """
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=input_fields + ['fibgz', 'trk_file', 'atlas_configs']),
        name="inputnode")
    outputnode = pe.Node(niu.IdentityInterface(fields=['matfile']),
                         name="outputnode")
    workflow = pe.Workflow(name=name)
    calc_connectivity = pe.Node(DSIStudioAtlasGraph(nthreads=omp_nthreads, **params),
                                name='calc_connectivity')
    plot_connectivity = pe.Node(ConnectivityReport(), name='plot_connectivity')
    ds_report_connectivity = pe.Node(
        ReconDerivativesDataSink(extension='.svg',
                                 desc="DSIStudioConnectivity",
                                 suffix='matrices'),
        name='ds_report_connectivity',
        run_without_submitting=True)

    workflow.connect([
        (inputnode, calc_connectivity, [('atlas_configs', 'atlas_configs'),
                                        ('fibgz', 'input_fib'),
                                        ('trk_file', 'trk_file')]),
        (calc_connectivity, plot_connectivity, [
            ('connectivity_matfile', 'connectivity_matfile')]),
        (plot_connectivity, ds_report_connectivity, [('out_report', 'in_file')]),
        (calc_connectivity, outputnode, [('connectivity_matfile', 'matfile')])
    ])
    if output_suffix:
        # Save the output in the outputs directory
        ds_connectivity = pe.Node(ReconDerivativesDataSink(suffix=output_suffix),
                                  name='ds_' + name,
                                  run_without_submitting=True)
        workflow.connect(calc_connectivity, 'connectivity_matfile', ds_connectivity, 'in_file')
    return workflow


def init_dsi_studio_export_wf(omp_nthreads, has_transform, name="dsi_studio_export",
                              params={}, output_suffix=""):
    """Export scalar maps from a DSI Studio fib file into NIfTI files with correct headers.

    This workflow exports gfa, fa0, fa1, fa2 and iso.

    Inputs

        fibgz
            A DSI Studio fib file

    Outputs

        gfa
            NIfTI file containing generalized fractional anisotropy (GFA).
        fa0
            Quantitative Anisotropy for the largest fixel in each voxel.
        fa1
            Quantitative Anisotropy for the second-largest fixel in each voxel.
        fa2
            Quantitative Anisotropy for the third-largest fixel in each voxel.
        iso
            Isotropic component of the ODF in each voxel.

    """
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=input_fields + ['fibgz']),
        name="inputnode")
    scalar_names = ['gfa', 'fa0', 'fa1', 'fa2', 'iso', 'dti_fa', 'md', 'rd', 'ad']
    outputnode = pe.Node(
        niu.IdentityInterface(fields=[name + "_file" for name in scalar_names]),
        name="outputnode")
    workflow = pe.Workflow(name=name)
    export = pe.Node(DSIStudioExport(to_export=",".join(scalar_names)), name='export')
    fixhdr_nodes = {}
    for scalar_name in scalar_names:
        output_name = scalar_name + '_file'
        fixhdr_nodes[scalar_name] = pe.Node(FixDSIStudioExportHeader(), name='fix_'+scalar_name)
        connections = [(export, fixhdr_nodes[scalar_name], [(output_name, 'dsi_studio_nifti')]),
                       (inputnode, fixhdr_nodes[scalar_name], [('dwi_file',
                                                                'correct_header_nifti')]),
                       (fixhdr_nodes[scalar_name], outputnode, [('out_file', scalar_name)])]
        if output_suffix:
            connections += [(fixhdr_nodes[scalar_name],
                             pe.Node(
                                 ReconDerivativesDataSink(desc=scalar_name,
                                                          suffix=output_suffix),
                                 name='ds_%s_%s' % (name, scalar_name)),
                             [('out_file', 'in_file')])]
        workflow.connect(connections)

    workflow.connect([(inputnode, export, [('fibgz', 'input_file')])])

    return workflow
