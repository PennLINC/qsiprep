import nipype.pipeline.engine as pe
from nipype.interfaces import utility as niu

from ... import config
from ...engine import Workflow
from ...interfaces.interchange import default_input_set, recon_workflow_input_fields
from .amico import init_amico_noddi_fit_wf
from .converters import init_mif_to_fibgz_wf, init_qsiprep_to_fsl_wf
from .dipy import (
    init_dipy_brainsuite_shore_recon_wf,
    init_dipy_dki_recon_wf,
    init_dipy_mapmri_recon_wf,
)
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
from .scalar_mapping import init_scalar_to_bundle_wf, init_scalar_to_template_wf
from .steinhardt import init_steinhardt_order_param_wf
from .tortoise import init_tortoise_estimator_wf
from .utils import init_conform_dwi_wf, init_discard_repeated_samples_wf


def _check_repeats(nodelist):
    total_len = len(nodelist)
    unique_len = len(set(nodelist))
    if not total_len == unique_len:
        raise Exception

def init_singleshell_benchmarking_wf(
    available_anatomical_data, name="_recon", qsirecon_suffix="SingleShellBenchmark", params={}
):
    pass
    gqi_params = params.get("gqi_recon", {})
    initial_gqi_wf = init_dsi_studio_recon_wf(
        available_anatomical_data=available_anatomical_data,
        name="initial_gqi",
        qsirecon_suffix=f"{qsirecon_suffix}_part-GQI",
        params=gqi_params
    )

    ss3t_params = params.get("ss3t_recon")
    ss3t_wf = init_dsi_studio_recon_wf(
        available_anatomical_data=available_anatomical_data,
        name="ss3t_recon",
        qsirecon_suffix=f"{qsirecon_suffix}_part-SS3T",
        params=ss3t_params
    )

    csd_params = params.get("csd_recon")
    csd_wf = init_dsi_studio_recon_wf(
        available_anatomical_data=available_anatomical_data,
        name="csd_recon",
        qsirecon_suffix=f"{qsirecon_suffix}_part-CSD",
        params=csd_params
    )


def init_dwi_recon_workflow(
    workflow_spec,
    available_anatomical_data,
    name="recon_wf",
):
    """Convert a workflow spec into a nipype workflow."""

    workflow = Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=recon_workflow_input_fields), name="inputnode"
    )
    # Read nodes from workflow spec, make sure we can implement them
    nodes_to_add = []
    workflow_metadata_nodes = {}
    for node_spec in workflow_spec["nodes"]:
        if not node_spec["name"]:
            raise Exception("Node has no name [{}]".format(node_spec))
        new_node = workflow_from_spec(
            available_anatomical_data=available_anatomical_data,
            node_spec=node_spec,
        )
        if new_node is None:
            raise Exception("Unable to create a node for %s" % node_spec)
        nodes_to_add.append(new_node)

        # Make an identity interface that just has the info of this node
        workflow_metadata_nodes[node_spec["name"]] = pe.Node(
            niu.IdentityInterface(fields=["input_metadata"]), name=node_spec["name"] + "_spec"
        )
        workflow_metadata_nodes[node_spec["name"]].inputs.input_metadata = node_spec
        nodes_to_add.append(workflow_metadata_nodes[node_spec["name"]])

    workflow.add_nodes(nodes_to_add)
    _check_repeats(workflow.list_node_names())

    # Create a node that gathers scalar outputs from those that produce them
    scalar_gatherer = pe.Node(niu.Merge(len(nodes_to_add)), name="scalar_gatherer")

    # Now that all nodes are in the workflow, connect them
    for node_num, node_spec in enumerate(workflow_spec["nodes"], start=1):

        # get the nipype node object
        node_name = node_spec["name"]
        node = workflow.get_node(node_name)

        consuming_scalars = node_spec.get("scalars_from", [])
        if consuming_scalars:
            workflow.connect(scalar_gatherer, "out",
                             node, "inputnode.collected_scalars")  # fmt:skip
        else:
            workflow.connect(node, "outputnode.recon_scalars",
                             scalar_gatherer, f"in{node_num}")  # fmt:skip
        if node_spec.get("input", "qsiprep") == "qsiprep":
            # directly connect all the qsiprep outputs to every node
            workflow.connect([
                (inputnode, node,
                 _as_connections(recon_workflow_input_fields, dest_prefix='inputnode.'))
            ])  # fmt:skip

        # connect the outputs from the upstream node to this node
        else:
            upstream_node = workflow.get_node(node_spec["input"])
            upstream_outputnode_name = node_spec["input"] + ".outputnode"
            upstream_outputnode = workflow.get_node(upstream_outputnode_name)
            upstream_outputs = set(upstream_outputnode.outputs.get().keys())
            downstream_inputnode_name = node_name + ".inputnode"
            downstream_inputnode = workflow.get_node(downstream_inputnode_name)
            downstream_inputs = set(downstream_inputnode.outputs.get().keys())

            connect_from_upstream = upstream_outputs.intersection(downstream_inputs)
            connect_from_qsiprep = default_input_set - connect_from_upstream

            config.loggers.workflow.debug(
                "connecting %s from %s to %s", connect_from_qsiprep, inputnode, node
            )
            workflow.connect([
                (
                    inputnode,
                    node,
                    _as_connections(
                        connect_from_qsiprep - set(("mapping_metadata",)),
                        dest_prefix='inputnode.'))
            ])  # fmt:skip
            _check_repeats(workflow.list_node_names())

            config.loggers.workflow.debug(
                "connecting %s from %s to %s",
                connect_from_upstream,
                upstream_outputnode_name,
                downstream_inputnode_name,
            )
            workflow.connect([
                (
                    upstream_node,
                    node,
                    _as_connections(
                        connect_from_upstream - set(("mapping_metadata",)),
                        src_prefix='outputnode.',
                        dest_prefix='inputnode.'))
            ])  # fmt:skip
            _check_repeats(workflow.list_node_names())

            # Send metadata about the upstream node into the downstream node
            workflow.connect(
                workflow_metadata_nodes[node_spec['input']],
                "input_metadata",
                node,
                "inputnode.mapping_metadata")  # fmt:skip

    # Fill-in datasinks and reportlet datasinks seen so far
    for node in workflow.list_node_names():
        node_suffix = node.split(".")[-1]
        if node_suffix.startswith("ds_") or node_suffix.startswith("recon_scalars"):
            base_dir = (
                config.execution.reportlets_dir
                if "report" in node_suffix
                else config.execution.output_dir
            )
            workflow.connect(inputnode, 'dwi_file',
                             workflow.get_node(node), 'source_file')  # fmt:skip
            # config.loggers.workflow.info("setting %s base dir to %s", node_suffix, base_dir )
            if node_suffix.startswith("ds"):
                workflow.get_node(node).inputs.base_directory = base_dir

    return workflow


def workflow_from_spec(available_anatomical_data, node_spec):
    """Build a nipype workflow based on a json file."""
    software = node_spec.get("software", "qsiprep")
    qsirecon_suffix = node_spec.get("qsirecon_suffix", "")
    node_name = node_spec.get("name", None)
    parameters = node_spec.get("parameters", {})

    # It makes more sense intuitively to have scalars_from in the
    # root of a recon spec "node". But to pass it to the workflow
    # it needs to go in parameters
    if "scalars_from" in node_spec and node_spec["scalars_from"]:
        if parameters.get("scalars_from"):
            config.loggers.workflow.warning("overwriting scalars_from in parameters")
        parameters["scalars_from"] = node_spec["scalars_from"]

    if config.execution.skip_odf_reports:
        config.loggers.workflow.info("skipping ODF plots for %s", node_name)
        parameters["plot_reports"] = False

    if node_name is None:
        raise Exception('Node %s must have a "name" attribute' % node_spec)
    kwargs = {
        "available_anatomical_data": available_anatomical_data,
        "name": node_name,
        "qsirecon_suffix": qsirecon_suffix,
        "params": parameters,
    }

    # DSI Studio operations
    if software == "DSI Studio":
        if node_spec["action"] == "reconstruction":
            return init_dsi_studio_recon_wf(**kwargs)
        if node_spec["action"] == "export":
            return init_dsi_studio_export_wf(**kwargs)
        if node_spec["action"] == "tractography":
            return init_dsi_studio_tractography_wf(**kwargs)
        if node_spec["action"] == "connectivity":
            return init_dsi_studio_connectivity_wf(**kwargs)
        if node_spec["action"] == "autotrack":
            return init_dsi_studio_autotrack_wf(**kwargs)

    # MRTrix3 operations
    elif software == "MRTrix3":
        if node_spec["action"] == "csd":
            return init_mrtrix_csd_recon_wf(**kwargs)
        if node_spec["action"] == "global_tractography":
            return init_global_tractography_wf(**kwargs)
        if node_spec["action"] == "tractography":
            return init_mrtrix_tractography_wf(**kwargs)
        if node_spec["action"] == "connectivity":
            return init_mrtrix_connectivity_wf(**kwargs)

    # Dipy operations
    elif software == "Dipy":
        if node_spec["action"] == "3dSHORE_reconstruction":
            return init_dipy_brainsuite_shore_recon_wf(**kwargs)
        if node_spec["action"] == "MAPMRI_reconstruction":
            return init_dipy_mapmri_recon_wf(**kwargs)
        if node_spec["action"] == "DKI_reconstruction":
            return init_dipy_dki_recon_wf(**kwargs)

    # AMICO operations
    elif software == "AMICO":
        if node_spec["action"] == "fit_noddi":
            return init_amico_noddi_fit_wf(**kwargs)

    elif software == "pyAFQ":
        from .pyafq import init_pyafq_wf

        if node_spec["action"] == "pyafq_tractometry":
            return init_pyafq_wf(**kwargs)

    elif software == "TORTOISE":
        if node_spec["action"] == "estimate":
            return init_tortoise_estimator_wf(**kwargs)

    # qsiprep operations
    else:
        if node_spec["action"] == "discard_repeated_samples":
            return init_discard_repeated_samples_wf(**kwargs)
        if node_spec["action"] == "conform":
            return init_conform_dwi_wf(**kwargs)
        if node_spec["action"] == "mif_to_fib":
            return init_mif_to_fibgz_wf(**kwargs)
        if node_spec["action"] == "reorient_fslstd":
            return init_qsiprep_to_fsl_wf(**kwargs)
        if node_spec["action"] == "steinhardt_order_parameters":
            return init_steinhardt_order_param_wf(**kwargs)
        if node_spec["action"] == "bundle_map":
            return init_scalar_to_bundle_wf(**kwargs)
        if node_spec["action"] == "template_map":
            return init_scalar_to_template_wf(**kwargs)

    raise Exception("Unknown node %s" % node_spec)


def _as_connections(attr_list, src_prefix="", dest_prefix=""):
    return [(src_prefix + item, dest_prefix + item) for item in attr_list]
