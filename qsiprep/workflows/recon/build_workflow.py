import logging
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
from nipype.utils.filemanip import split_filename
from qsiprep.interfaces.bids import QsiReconIngress, ReconDerivativesDataSink
from qsiprep.interfaces.utils import GetConnectivityAtlases
from .dsi_studio import (init_dsi_studio_recon_wf, init_dsi_studio_export_wf,
                         init_dsi_studio_connectivity_wf)
from .dipy import init_dipy_brainsuite_shore_recon_wf, init_dipy_mapmri_recon_wf
from .mrtrix import init_mrtrix_vanilla_csd_recon_wf
from .converters import init_mif_to_fibgz_wf
from .dynamics import init_controllability_wf
from .utils import init_conform_dwi_wf, init_discard_repeated_samples_wf
from ...engine import Workflow
from .interchange import (qsiprep_output_names, input_fields, default_connections,
                          default_input_set)

LOGGER = logging.getLogger('nipype.interface')


def _get_resampled(atlas_configs, atlas_name):
    return atlas_configs[atlas_name]['dwi_resolution_file']


def _check_repeats(nodelist):
    total_len = len(nodelist)
    unique_len = len(set(nodelist))
    if not total_len == unique_len:
        raise Exception


def init_dwi_recon_workflow(dwi_file, workflow_spec, output_dir, reportlets_dir,
                            omp_nthreads, name="recon_wf"):
    atlas_names = workflow_spec.get('atlases', [])
    space = workflow_spec['space']
    workflow = Workflow(name=_get_wf_name(dwi_file))
    inputnode = pe.Node(niu.IdentityInterface(fields=input_fields), name='inputnode')
    preprocessed_data = pe.Node(QsiReconIngress(), name="preprocessed_data")

    # For doctests
    if not workflow_spec['name'] == 'fake':
        inputnode.inputs.dwi_file = dwi_file
        preprocessed_data.inputs.dwi_file = dwi_file

    # Connect the collected diffusion data (gradients, etc) to the inputnode
    workflow.connect([
        (preprocessed_data, inputnode, [
            (trait, trait) for trait in qsiprep_output_names])
    ])
    # Resample all atlases to dwi_file's resolution
    get_atlases = pe.Node(
        GetConnectivityAtlases(atlas_names=atlas_names, space=space),
        name='get_atlases',
        run_without_submitting=True)

    # Save the atlases
    if len(atlas_names) > 0:
        if space == "T1w":
            workflow.connect([
                (inputnode, get_atlases,
                 [('t1_2_mni_reverse_transform', 'forward_transform')])])
        for atlas in workflow_spec['atlases']:
            workflow.connect([
                (get_atlases,
                 pe.Node(ReconDerivativesDataSink(space=space,
                                                  desc=atlas,
                                                  suffix="atlas",
                                                  compress=True),
                         name='ds_atlases_'+atlas,
                         run_without_submitting=True),
                 [(('atlas_configs', _get_resampled, atlas), 'in_file')])
            ])
        workflow.connect(inputnode, "dwi_file", get_atlases, "reference_image")
    # Read nodes from workflow spec, make sure we can implement them
    nodes_to_add = []
    for node_spec in workflow_spec['nodes']:
        if not node_spec['name']:
            raise Exception("Node has no name [{}]".format(node_spec))
        new_node = workflow_from_spec(node_spec)
        if new_node is None:
            raise Exception("Unable to create a node for %s" % node_spec)
        nodes_to_add.append(new_node)
    workflow.add_nodes(nodes_to_add)
    _check_repeats(workflow.list_node_names())
    # Now that all nodes are in the workflow, connect them
    for node_spec in workflow_spec['nodes']:

        # get the nipype node object
        node_name = node_spec['name']
        node = workflow.get_node(node_name)

        if node_spec.get('input', 'qsiprep') == 'qsiprep':
            # directly connect all the qsiprep outputs to every node
            for from_conn, to_conn in default_connections:
                workflow.connect(inputnode, from_conn, node, 'inputnode.' + to_conn)
                _check_repeats(workflow.list_node_names())

        # connect the outputs from the upstream node to this node
        else:
            upstream_node = workflow.get_node(node_spec['input'])
            upstream_outputnode_name = node_spec['input'] + '.outputnode'
            upstream_outputnode = workflow.get_node(upstream_outputnode_name)
            upstream_outputs = set(upstream_outputnode.outputs.get().keys())
            downstream_inputnode_name = node_name + ".inputnode"
            downstream_inputnode = workflow.get_node(downstream_inputnode_name)
            downstream_inputs = set(downstream_inputnode.outputs.get().keys())

            connect_from_upstream = upstream_outputs.intersection(downstream_inputs)
            connect_from_qsiprep = default_input_set - connect_from_upstream

            LOGGER.info("connecting %s from %s to %s", connect_from_qsiprep,
                        inputnode, node)
            # workflow.connect([(inputnode, node, _as_connections(connect_from_qsiprep))])
            for qp_connection in connect_from_qsiprep:
                workflow.connect(inputnode, qp_connection, node, 'inputnode.' + qp_connection)
                _check_repeats(workflow.list_node_names())

            LOGGER.info("connecting %s from %s to %s", connect_from_upstream,
                        upstream_outputnode_name, downstream_inputnode_name)
            # workflow.connect([(upstream_node, node, _as_connections(connect_from_upstream))])
            for upstream_connection in connect_from_upstream:
                workflow.connect(upstream_node, "outputnode." + upstream_connection,
                                 node, 'inputnode.' + upstream_connection)
                _check_repeats(workflow.list_node_names())

        # If it's a connectivity calculation, send it the atlas configs
        if node_spec['action'] == 'connectivity':
            workflow.connect([(get_atlases, node,
                               [('atlas_configs', 'inputnode.atlas_configs')])])
        _check_repeats(workflow.list_node_names())

    # Fill-in datasinks and reportlet datasinks seen so far
    for node in workflow.list_node_names():
        node_suffix = node.split('.')[-1]
        if node_suffix.startswith('ds_'):
            workflow.get_node(node).inputs.source_file = dwi_file
            workflow.get_node(node).inputs.space = space
            if "report" in node_suffix:
                workflow.get_node(node).inputs.base_directory = reportlets_dir
            else:
                workflow.get_node(node).inputs.base_directory = output_dir

    return workflow


def workflow_from_spec(node_spec):
    """Build a nipype workflow based on a json file."""
    software = node_spec.get("software", "qsiprep")
    output_suffix = node_spec.get("output_suffix", "")
    node_name = node_spec.get("name", None)
    parameters = node_spec.get("parameters", {})

    if node_name is None:
        raise Exception('Node %s must have a "name" attribute' % node_spec)
    kwargs = {"name": node_name,
              "output_suffix": output_suffix,
              "params": parameters}

    # DSI Studio operations
    if software == "DSI Studio":
        if node_spec["action"] == "reconstruction":
            return init_dsi_studio_recon_wf(**kwargs)
        if node_spec["action"] == "export":
            return init_dsi_studio_export_wf(**kwargs)
        if node_spec["action"] == "connectivity":
            return init_dsi_studio_connectivity_wf(**kwargs)

    # MRTrix3 operations
    elif software == "MRTrix3":
        if node_spec["action"] == "csd":
            return init_mrtrix_vanilla_csd_recon_wf(**kwargs)

    # Dipy operations
    elif software == "Dipy":
        if node_spec["action"] == "3dSHORE_reconstruction":
            return init_dipy_brainsuite_shore_recon_wf(**kwargs)
        if node_spec["action"] == "MAPMRI_reconstruction":
            return init_dipy_mapmri_recon_wf(**kwargs)

    # qsiprep operations
    else:
        if node_spec['action'] == "controllability":
            return init_controllability_wf(**kwargs)
        if node_spec['action'] == 'discard_repeated_samples':
            return init_discard_repeated_samples_wf(**kwargs)
        if node_spec['action'] == 'conform':
            return init_conform_dwi_wf(**kwargs)
        if node_spec['action'] == 'mif_to_fib':
            return init_mif_to_fibgz_wf(**kwargs)
    raise Exception("Unknown node %s", node_spec)


def _as_connections(attr_list):
    return [(item, item) for item in attr_list]


def _get_wf_name(dwi_file):
    basedir, fname, ext = split_filename(dwi_file)
    tokens = fname.split("_")
    return "_".join(tokens[:-1]).replace("-", "_")
