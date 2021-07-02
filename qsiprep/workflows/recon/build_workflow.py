import logging
import nipype.pipeline.engine as pe
from pkg_resources import resource_filename as pkgr
from nipype.interfaces import ants, utility as niu
from nipype.utils.filemanip import split_filename
from qsiprep.interfaces.bids import QsiReconIngress, ReconDerivativesDataSink
from qsiprep.interfaces.utils import GetConnectivityAtlases
from .dsi_studio import (init_dsi_studio_recon_wf, init_dsi_studio_export_wf,
                         init_dsi_studio_connectivity_wf, init_dsi_studio_tractography_wf)
from .dipy import init_dipy_brainsuite_shore_recon_wf, init_dipy_mapmri_recon_wf
from .mrtrix import (init_mrtrix_csd_recon_wf, init_global_tractography_wf,
                     init_mrtrix_tractography_wf, init_mrtrix_connectivity_wf)
from .amico import init_amico_noddi_fit_wf
from .converters import init_mif_to_fibgz_wf, init_qsiprep_to_fsl_wf
from .dynamics import init_controllability_wf
from .utils import init_conform_dwi_wf, init_discard_repeated_samples_wf
from ...engine import Workflow
from .interchange import (qsiprep_output_names, input_fields, default_input_set)

LOGGER = logging.getLogger('nipype.interface')


def _get_resampled(atlas_configs, atlas_name, to_retrieve):
    return atlas_configs[atlas_name][to_retrieve]


def _check_repeats(nodelist):
    total_len = len(nodelist)
    unique_len = len(set(nodelist))
    if not total_len == unique_len:
        raise Exception


def init_dwi_recon_workflow(dwi_files, workflow_spec, output_dir, reportlets_dir, has_t1w,
                            has_t1w_transform, omp_nthreads, name="recon_wf"):
    """Convert a workflow spec into a nipype workflow.

    """
    atlas_names = workflow_spec.get('atlases', [])
    space = workflow_spec['space']
    workflow = Workflow(name=name)
    scans_iter = pe.Node(niu.IdentityInterface(fields=['dwi_file']), name='scans_iter')
    scans_iter.iterables = ("dwi_file", dwi_files)
    inputnode = pe.Node(niu.IdentityInterface(fields=input_fields + ['dwi_file']),
                        name='inputnode')
    qsiprep_preprocessed_dwi_data = pe.Node(
        QsiReconIngress(), name="qsiprep_preprocessed_dwi_data")

    # For doctests
    if not workflow_spec['name'] == 'fake':
        scans_iter.inputs.dwi_file = dwi_files

    # Connect the collected diffusion data (gradients, etc) to the inputnode
    workflow.connect([
        (scans_iter, qsiprep_preprocessed_dwi_data, ([('dwi_file', 'dwi_file')])),
        (qsiprep_preprocessed_dwi_data, inputnode, [
            (trait, trait) for trait in qsiprep_output_names])])

    # Resample all atlases to dwi_file's resolution
    get_atlases = pe.Node(
        GetConnectivityAtlases(atlas_names=atlas_names, space=space),
        name='get_atlases',
        run_without_submitting=True)

    # Resample ROI targets to DWI resolution for ODF plotting
    crossing_rois_file = pkgr('qsiprep', 'data/crossing_rois.nii.gz')
    odf_rois = pe.Node(
        ants.ApplyTransforms(interpolation="MultiLabel", dimension=3),
        name="odf_rois")
    odf_rois.inputs.input_image = crossing_rois_file
    if has_t1w_transform and space == "T1w":
        workflow.connect(inputnode, 't1_2_mni_reverse_transform', odf_rois, 'transforms')
    elif space == 'template':
        odf_rois.inputs.transforms = ['identity']
    else:
        LOGGER.warning("Unable to transform ODF ROIs to dwi data. "
                       "No ODF reports will be created.")
        odf_rois = pe.Node(niu.IdentityInterface(fields=['output_image']), name='odf_rois')
    workflow.connect(scans_iter, 'dwi_file', odf_rois, 'reference_image')

    # Save the atlases
    if len(atlas_names) > 0:
        if space == "T1w":
            if not has_t1w_transform:
                LOGGER.critical("No reverse transform found, unable to move atlases"
                                " into DWI space")
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
                 [(('atlas_configs', _get_resampled, atlas, 'dwi_resolution_file'), 'in_file')]),
                (get_atlases,
                 pe.Node(ReconDerivativesDataSink(space=space,
                                                  desc=atlas,
                                                  suffix="atlas",
                                                  extension=".mif.gz",
                                                  compress=True),
                         name='ds_atlas_mifs_'+atlas,
                         run_without_submitting=True),
                 [(('atlas_configs', _get_resampled, atlas, 'dwi_resolution_mif'), 'in_file')]),
                (get_atlases,
                 pe.Node(ReconDerivativesDataSink(space=space,
                                                  desc=atlas,
                                                  extension=".txt",
                                                  suffix="mrtrixLUT"),
                         name='ds_atlas_mrtrix_lut_' + atlas,
                         run_without_submitting=True),
                 [(('atlas_configs', _get_resampled, atlas, 'mrtrix_lut'), 'in_file')]),
                (get_atlases,
                 pe.Node(ReconDerivativesDataSink(space=space,
                                                  desc=atlas,
                                                  extension=".txt",
                                                  suffix="origLUT"),
                         name='ds_atlas_orig_lut_' + atlas,
                         run_without_submitting=True),
                 [(('atlas_configs', _get_resampled, atlas, 'orig_lut'), 'in_file')]),
            ])
        workflow.connect(inputnode, "dwi_file", get_atlases, "reference_image")

    # Read nodes from workflow spec, make sure we can implement them
    nodes_to_add = []
    for node_spec in workflow_spec['nodes']:
        if not node_spec['name']:
            raise Exception("Node has no name [{}]".format(node_spec))
        new_node = workflow_from_spec(omp_nthreads, has_t1w,
                                      has_t1w_transform or space == 'template',
                                      node_spec)
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
                 _as_connections(input_fields, dest_prefix='inputnode.'))])
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

        # If it's a connectivity calculation, send it the atlas configs
        if node_spec['action'] == 'connectivity':
            workflow.connect([(get_atlases, node,
                               [('atlas_configs', 'inputnode.atlas_configs')])])
        _check_repeats(workflow.list_node_names())

        # Send the ODF rois to reconstruction nodes
        if node_spec['action'] == 'csd' or 'reconstruction' in node_spec['action']:
            workflow.connect([(odf_rois, node,
                               [('output_image', 'inputnode.odf_rois')])])
        _check_repeats(workflow.list_node_names())

    # Fill-in datasinks and reportlet datasinks seen so far
    for node in workflow.list_node_names():
        node_suffix = node.split('.')[-1]
        if node_suffix.startswith('ds_'):
            workflow.connect(scans_iter, 'dwi_file', workflow.get_node(node), 'source_file')
            workflow.get_node(node).inputs.space = space
            if "report" in node_suffix:
                workflow.get_node(node).inputs.base_directory = reportlets_dir
            else:
                workflow.get_node(node).inputs.base_directory = output_dir

    return workflow


def workflow_from_spec(omp_nthreads, has_t1w, has_t1w_transform, node_spec):
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
            return init_dsi_studio_recon_wf(omp_nthreads, has_t1w_transform, **kwargs)
        if node_spec["action"] == "export":
            return init_dsi_studio_export_wf(omp_nthreads, has_t1w_transform, **kwargs)
        if node_spec["action"] == "tractography":
            return init_dsi_studio_tractography_wf(omp_nthreads, has_t1w_transform, **kwargs)
        if node_spec["action"] == "connectivity":
            return init_dsi_studio_connectivity_wf(omp_nthreads, has_t1w_transform, **kwargs)

    # MRTrix3 operations
    elif software == "MRTrix3":
        if node_spec["action"] == "csd":
            return init_mrtrix_csd_recon_wf(omp_nthreads, has_t1w_transform, **kwargs)
        if node_spec["action"] == "global_tractography":
            return init_global_tractography_wf(omp_nthreads, has_t1w_transform, **kwargs)
        if node_spec["action"] == "tractography":
            return init_mrtrix_tractography_wf(omp_nthreads, has_t1w_transform, **kwargs)
        if node_spec["action"] == "connectivity":
            return init_mrtrix_connectivity_wf(omp_nthreads, has_t1w_transform, **kwargs)

    # Dipy operations
    elif software == "Dipy":
        if node_spec["action"] == "3dSHORE_reconstruction":
            return init_dipy_brainsuite_shore_recon_wf(omp_nthreads, has_t1w_transform, **kwargs)
        if node_spec["action"] == "MAPMRI_reconstruction":
            return init_dipy_mapmri_recon_wf(omp_nthreads, has_t1w_transform, **kwargs)

    # AMICO operations
    elif software == "AMICO":
        if node_spec["action"] == "fit_noddi":
            return init_amico_noddi_fit_wf(omp_nthreads, has_t1w_transform, **kwargs)
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

    raise Exception("Unknown node %s" % node_spec)


def _as_connections(attr_list, src_prefix='', dest_prefix=''):
    return [(src_prefix + item, dest_prefix + item) for item in attr_list]


def _get_wf_name(dwi_file):
    basedir, fname, ext = split_filename(dwi_file)
    tokens = fname.split("_")
    return "_".join(tokens[:-1]).replace("-", "_")
