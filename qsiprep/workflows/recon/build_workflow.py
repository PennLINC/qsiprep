import logging
import nipype.pipeline.engine as pe
from pkg_resources import resource_filename as pkgr
from nipype.interfaces import ants, utility as niu
from nipype.utils.filemanip import split_filename
from qsiprep.interfaces import anatomical
from qsiprep.interfaces.bids import QsiReconIngress, ReconDerivativesDataSink
from .dsi_studio import (init_dsi_studio_recon_wf, init_dsi_studio_export_wf,
                         init_dsi_studio_connectivity_wf, init_dsi_studio_tractography_wf)
from .dipy import (init_dipy_brainsuite_shore_recon_wf, init_dipy_mapmri_recon_wf,
    init_dipy_dki_recon_wf)
from .mrtrix import (init_mrtrix_csd_recon_wf, init_global_tractography_wf,
                     init_mrtrix_tractography_wf, init_mrtrix_connectivity_wf)
from .amico import init_amico_noddi_fit_wf
from .converters import init_mif_to_fibgz_wf, init_qsiprep_to_fsl_wf
from .dynamics import init_controllability_wf
from .utils import init_conform_dwi_wf, init_discard_repeated_samples_wf
from .steinhardt import init_steinhardt_order_param_wf
from ...engine import Workflow
from .interchange import (qsiprep_output_names, default_input_set,
    recon_workflow_input_fields, recon_workflow_anatomical_input_fields,
    ReconWorkflowInputs)
from .anatomical import init_dwi_recon_anatomical_workflow

LOGGER = logging.getLogger('nipype.interface')

def _check_repeats(nodelist):
    total_len = len(nodelist)
    unique_len = len(set(nodelist))
    if not total_len == unique_len:
        raise Exception


def init_dwi_recon_workflow(dwi_file, workflow_spec, output_dir, prefer_dwi_mask,
                            reportlets_dir, available_anatomical_data, omp_nthreads, b0_threshold,
                            infant_mode, skip_odf_plots, freesurfer_dir=None, 
                            sloppy=False, name="recon_wf"):
    """Convert a workflow spec into a nipype workflow.

    """
    # Get the preprocessed DWI and all the related preprocessed images
    qsiprep_preprocessed_dwi_data = pe.Node(
        QsiReconIngress(dwi_file=dwi_file), 
        name="qsiprep_preprocessed_dwi_data")

    # Get the anatomical data (masks, atlases, etc)
    atlas_names = workflow_spec.get('atlases', [])
    registered_anat_wf, available_anatomical_data = init_dwi_recon_anatomical_workflow(
        atlas_names=atlas_names,
        omp_nthreads=omp_nthreads,
        infant_mode=infant_mode,
        prefer_dwi_mask=prefer_dwi_mask,
        sloppy=sloppy,
        b0_threshold=b0_threshold,
        extras_to_make=workflow_spec.get('anatomical', []),
        freesurfer_dir=freesurfer_dir,
        name="qsirecon_anat_wf",
        **available_anatomical_data)
    
    # For doctests
    # if not workflow_spec['name'] == 'fake':
    #     inputnode.inputs.dwi_file = dwi_file

    workflow = Workflow(name=_get_wf_name(dwi_file) + "_" + name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=recon_workflow_input_fields),
        name='inputnode')
    
    # Connect the collected diffusion data (gradients, etc) to the inputnode
    workflow.connect([
        (qsiprep_preprocessed_dwi_data, registered_anat_wf, [
            (trait, 'inputnode.' + trait) for trait in qsiprep_output_names]),
        (registered_anat_wf, inputnode, [
            ('outputnode.'+trait, trait) for trait in 
            recon_workflow_input_fields])
    ])

    # Read nodes from workflow spec, make sure we can implement them
    nodes_to_add = []
    for node_spec in workflow_spec['nodes']:
        if not node_spec['name']:
            raise Exception("Node has no name [{}]".format(node_spec))
        new_node = workflow_from_spec(
            omp_nthreads=omp_nthreads,
            available_anatomical_data=available_anatomical_data,
            node_spec=node_spec,
            skip_odf_plots=skip_odf_plots)
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
            workflow.connect([
                (inputnode, node,
                 _as_connections(recon_workflow_input_fields, dest_prefix='inputnode.'))])
            # for from_conn, to_conn in default_connections:
            #     workflow.connect(inputnode, from_conn, node, 'inputnode.' + to_conn)
            #     _check_repeats(workflow.list_node_names())

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

            # LOGGER.info("connecting %s from %s to %s", connect_from_qsiprep,
            #             inputnode, node)
            workflow.connect([
                (inputnode, node,
                 _as_connections(connect_from_qsiprep, dest_prefix='inputnode.'))])
            # for qp_connection in connect_from_qsiprep:
            #    workflow.connect(inputnode, qp_connection, node, 'inputnode.' + qp_connection)
            _check_repeats(workflow.list_node_names())

            # LOGGER.info("connecting %s from %s to %s", connect_from_upstream,
            #             upstream_outputnode_name, downstream_inputnode_name)
            workflow.connect([
                (upstream_node, node,
                 _as_connections(
                    connect_from_upstream, src_prefix='outputnode.', dest_prefix='inputnode.'))])
            # for upstream_connection in connect_from_upstream:
            #     workflow.connect(upstream_node, "outputnode." + upstream_connection,
            #                      node, 'inputnode.' + upstream_connection)
            _check_repeats(workflow.list_node_names())

    # Fill-in datasinks and reportlet datasinks seen so far
    for node in workflow.list_node_names():
        node_suffix = node.split('.')[-1]
        if node_suffix.startswith('ds_'):
            workflow.connect(inputnode, 'dwi_file', workflow.get_node(node), 'source_file')
            if "report" in node_suffix:
                workflow.get_node(node).inputs.base_directory = reportlets_dir
            else:
                workflow.get_node(node).inputs.base_directory = output_dir

    return workflow


def workflow_from_spec(omp_nthreads, available_anatomical_data, node_spec,
                       skip_odf_plots):
    """Build a nipype workflow based on a json file."""
    software = node_spec.get("software", "qsiprep")
    output_suffix = node_spec.get("output_suffix", "")
    node_name = node_spec.get("name", None)
    parameters = node_spec.get("parameters", {})
    if skip_odf_plots:
        LOGGER.info("skipping ODF plots for %s", node_name)
        parameters['plot_reports'] = False

    if node_name is None:
        raise Exception('Node %s must have a "name" attribute' % node_spec)
    kwargs = {
        "omp_nthreads": omp_nthreads,
        "available_anatomical_data": available_anatomical_data,
        "name": node_name,
        "output_suffix": output_suffix,
        "params": parameters}


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
        if node_spec['action'] == 'reorient_fslstd':
            return init_qsiprep_to_fsl_wf(**kwargs)
        if node_spec['action'] == 'steinhardt_order_parameters':
            return init_steinhardt_order_param_wf(**kwargs)

    raise Exception("Unknown node %s" % node_spec)


def _as_connections(attr_list, src_prefix='', dest_prefix=''):
    return [(src_prefix + item, dest_prefix + item) for item in attr_list]


def _get_wf_name(dwi_file):
    basedir, fname, ext = split_filename(dwi_file)
    tokens = fname.split("_")
    return "_".join(tokens[:-1]).replace("-", "_")
