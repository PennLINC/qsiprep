import logging
import nipype.pipeline.engine as pe
from nipype.interfaces import utility as niu

from ... import config
from ...engine import Workflow
from ...interfaces.interchange import recon_workflow_input_fields

from .dsi_studio import (
    init_dsi_studio_autotrack_wf,
    init_dsi_studio_connectivity_wf,
    init_dsi_studio_export_wf,
    init_dsi_studio_recon_wf,
    init_dsi_studio_tractography_wf,
)
from .mrtrix import (
    init_global_tractography_wf,
    init_mrtrix_connectivity_wf,
    init_mrtrix_csd_recon_wf,
    init_mrtrix_tractography_wf,
)

LOGGER = logging.getLogger("nipype.interface")


def init_singleshell_benchmarking_wf(
    available_anatomical_data, name="_recon", qsirecon_suffix="SingleShellBenchmark", params={}
):
    inputnode = pe.Node(
        niu.IdentityInterface(fields=recon_workflow_input_fields + ["odf_rois"]), name="inputnode"
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["tck_files", "bundle_names", "recon_scalars"]),
        name="outputnode",
    )
    outputnode.inputs.recon_scalars = []
    workflow = Workflow(name=name)
    omp_nthreads = config.nipype.omp_nthreads

    autotrack_params = params.get("autotrack", {})
    bundle_names = _get_dsi_studio_bundles(autotrack_params.get("track_id", ""))
    bundle_desc = (
        "AutoTrack attempted to reconstruct the following bundles:\n  * "
        + "\n  * ".join(bundle_names)
        + "\n\n"
    )
    LOGGER.info(bundle_desc)

    # First do a standard GQI reconstruction
    gqi_params = params.get("gqi_recon", {})
    initial_gqi_wf = init_dsi_studio_recon_wf(
        available_anatomical_data=available_anatomical_data,
        name="initial_gqi",
        qsirecon_suffix=f"{qsirecon_suffix}_part-GQI",
        params=gqi_params,
    )

    # Do an SS3T recon to feed to autotrack
    ss3t_params = params.get("ss3t_recon")
    ss3t_wf = init_mrtrix_csd_recon_wf(
        available_anatomical_data=available_anatomical_data,
        name="ss3t_recon",
        qsirecon_suffix=f"{qsirecon_suffix}_part-SS3T",
        params=ss3t_params,
    )
    ss3t_to_fib = pe.Node(FODtoFIBGZ(), name="ss3t_to_fib")

    # For comparison, also do a regular CSD
    csd_params = params.get("csd_recon")
    csd_wf = init_mrtrix_csd_recon_wf(
        available_anatomical_data=available_anatomical_data,
        name="csd_recon",
        qsirecon_suffix=f"{qsirecon_suffix}_part-CSD",
        params=csd_params,
    )
    csd_to_fib = pe.Node(FODtoFIBGZ(), name="csd_to_fib")

    # Run autotrack!
    gqi_autotrack = pe.Node(
        AutoTrack(num_threads=omp_nthreads, **params), name="gqi_autotrack", n_procs=omp_nthreads
    )
    ss3t_autotrack = pe.Node(
        AutoTrack(num_threads=omp_nthreads, **params), name="ss3t_autotrack", n_procs=omp_nthreads
    )
    csd_autotrack = pe.Node(
        AutoTrack(num_threads=omp_nthreads, **params), name="csd_autotrack", n_procs=omp_nthreads
    )

    # Create a single output
    aggregate_gqi_atk_results = pe.Node(
        AggregateAutoTrackResults(expected_bundles=bundle_names), name="aggregate_gqi_atk_results"
    )
    aggregate_ss3t_atk_results = pe.Node(
        AggregateAutoTrackResults(expected_bundles=bundle_names), name="aggregate_ss3t_atk_results"
    )
    aggregate_csd_atk_results = pe.Node(
        AggregateAutoTrackResults(expected_bundles=bundle_names), name="aggregate_csd_atk_results"
    )

    convert_gqi_to_tck = pe.MapNode(DSIStudioTrkToTck(), name="convert_gqi_to_tck", iterfield="trk_file")
    convert_ss3t_to_tck = pe.MapNode(DSIStudioTrkToTck(), name="convert_ss3t_to_tck", iterfield="trk_file")
    convert_csd_to_tck = pe.MapNode(DSIStudioTrkToTck(), name="convert_csd_to_tck", iterfield="trk_file")

    # Save the bundle csv
    ds_gqi_bundle_csv = pe.Node(
        ReconDerivativesDataSink(suffix="bundlestats", qsirecon_suffix=f"{qsirecon_suffix}_part-GQI"),
        name="ds_gqi_bundle_csv",
        run_without_submitting=True,
    )
    ds_ss3t_bundle_csv = pe.Node(
        ReconDerivativesDataSink(suffix="bundlestats", qsirecon_suffix=f"{qsirecon_suffix}_part-SS3T"),
        name="ds_ss3t_bundle_csv",
        run_without_submitting=True,
    )
    ds_csd_bundle_csv = pe.Node(
        ReconDerivativesDataSink(suffix="bundlestats", qsirecon_suffix=f"{qsirecon_suffix}_part-CSD"),
        name="ds_csd_bundle_csv",
        run_without_submitting=True,
    )

    # Save the mapping file. We're only using the mapping from GQI
    ds_mapping = pe.Node(
        ReconDerivativesDataSink(suffix="mapping", qsirecon_suffix=qsirecon_suffix),
        name="ds_mapping",
        run_without_submitting=True,
    )

    workflow.connect([
        # Connect the qsiprep inputs to the recon workflows we're creating here
        # (normally this is done in build_workflow())
        (inputnode, initial_gqi_wf,
            _as_connections(recon_workflow_input_fields, dest_prefix='inputnode.')),
        (inputnode, ss3t_wf,
            _as_connections(recon_workflow_input_fields, dest_prefix='inputnode.')),
        (inputnode, csd_wf,
            _as_connections(recon_workflow_input_fields, dest_prefix='inputnode.')),

        # Convert the sh mifs from csd and ss3t into fib files
        (csd_wf, ss3t_to_fib, [("outputnode.mif_file", "mif_file")]),
        (initial_gqi_wf, csd_to_fib, [("outputnode.fibgz", "fib_file")]),
        (ss3t_wf, ss3t_to_fib, [("outputnode.mif_file", "mif_file")]),
        (initial_gqi_wf, ss3t_to_fib, [("outputnode.fibgz", "fib_file")]),

        # Send the fib files to autotrack. Use the map file from gqi in ss3t and csd
        (initial_gqi_wf, gqi_autotrack, [("outputnode.fibgz", "fib_file")]),
        (ss3t_to_fib, ss3t_autotrack, [("fib_file", "fib_file")]),
        (gqi_autotrack, ss3t_autotrack, [("map_file", "map_file")]),
        (csd_to_fib, csd_autotrack, [("fib_file", "fib_file")]),
        (gqi_autotrack, csd_autotrack, [("map_file", "map_file")]),
    ])  # fmt:skip

    return workflow


def _as_connections(attr_list, src_prefix="", dest_prefix=""):
    return [(src_prefix + item, dest_prefix + item) for item in attr_list]