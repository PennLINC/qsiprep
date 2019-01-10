import json
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
from nipype.utils.filemanip import copyfile, split_filename

import logging
import os
import os.path as op
from qsiprep.interfaces.dsi_studio import (DSIStudioCreateSrc, DSIStudioGQIReconstruction,
                                           DSIStudioAtlasGraph, DSIStudioExport,
                                           FixDSIStudioExportHeader)
from qsiprep.interfaces.bids import QsiprepOutput, ReconDerivativesDataSink
from qsiprep.interfaces.utils import GetConnectivityAtlases
from qsiprep.interfaces.connectivity import Controllability
from qsiprep.interfaces.gradients import RemoveDuplicates
from qsiprep.interfaces.mrtrix import ResponseSD, EstimateFOD, MRConvert

LOGGER = logging.getLogger('nipype.interface')
qsiprep_output_names = QsiprepOutput().output_spec.class_editable_traits()
default_connections = [(trait, trait) for trait in qsiprep_output_names]
default_input_set = set(qsiprep_output_names)


def _get_resampled(atlas_configs, atlas_name):
    return atlas_configs[atlas_name]['dwi_resolution_file']


def init_dwi_recon_workflow(dwi_file, workflow_spec, output_dir, reportlets_dir, name="recon_wf"):
    atlas_names = workflow_spec['atlases']
    space = workflow_spec['space']

    # Collect all relevant qsiprep outputs for dwi_file
    preprocessed_data = pe.Node(QsiprepOutput(), name='preprocessed_data')
    preprocessed_data.inputs.in_file = dwi_file

    # Resample all atlases to dwi_file's resolution
    get_atlases = pe.Node(GetConnectivityAtlases(atlas_names=atlas_names), name='get_atlases')
    get_atlases.inputs.reference_image = dwi_file

    # Go time.
    workflow = pe.Workflow(name=_get_wf_name(dwi_file))

    # Transform from MNI to T1w if analysis should happen in T1w space
    workflow.add_nodes([preprocessed_data, get_atlases])
    if space == "T1w":
        workflow.connect([
            (preprocessed_data, get_atlases,
             [('t1_2_mni_reverse_transform', 'forward_transform')])])

    # Save the atlases
    if len(workflow_spec['atlases']):
        for atlas in workflow_spec['atlases']:
            workflow.connect([
                (get_atlases,
                 pe.Node(ReconDerivativesDataSink(
                                             space=space,
                                             desc=atlas,
                                             suffix="atlas",
                                             compress=True),
                         name='ds_atlases_'+atlas,
                         run_without_submitting=True),
                 [(('atlas_configs', _get_resampled, atlas), 'in_file')])
            ])

    # Read nodes from workflow spec, make sure we can implement them
    nodes_to_add = []
    for node_spec in workflow_spec['nodes']:
        new_node = workflow_from_spec(node_spec)
        if new_node is None:
            raise Exception("Unable to create a node for %s", node_spec)
        nodes_to_add.append(new_node)
    workflow.add_nodes(nodes_to_add)

    # Now that all nodes are in the workflow, connect them
    for node_spec in workflow_spec['nodes']:

        # get the nipype node object
        node_name = node_spec['name'] + ".inputnode"
        node = workflow.get_node(node_name)

        if node_spec.get('input', 'qsiprep') == 'qsiprep':
            # directly connect all the qsiprep outputs to every node
            workflow.connect([(preprocessed_data, node, default_connections)])

        # connect the outputs from the upstream node to this node
        else:
            upstream_node_name = node_spec['input'] + '.outputnode'
            upstream_node = workflow.get_node(upstream_node_name)
            upstream_outputs = set(upstream_node.outputs.get().keys())

            node_inputs = set(node.inputs.get().keys())
            connect_from_upstream = upstream_outputs.intersection(node_inputs)
            connect_from_qsiprep = default_input_set - upstream_outputs

            LOGGER.info("connecting %s from %s to %s", connect_from_qsiprep,
                        preprocessed_data, node)
            LOGGER.info("connecting %s from %s to %s", connect_from_upstream,
                        upstream_node, node)
            workflow.connect([(preprocessed_data, node, _as_connections(connect_from_qsiprep))])
            workflow.connect([(upstream_node, node, _as_connections(connect_from_upstream))])

        # If it's a connectivity calculation, send it the atlas configs
        if node_spec['action'] == 'connectivity':
            workflow.connect([(get_atlases, node,
                               [('atlas_configs', 'atlas_configs')])])

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


def init_dsi_studio_recon_wf(name="dsi_studio_recon", output_suffix="", params={}):
    inputnode = pe.Node(niu.IdentityInterface(fields=qsiprep_output_names),
                        name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['fibgz']),
        name="outputnode")
    workflow = pe.Workflow(name=name)
    create_src = pe.Node(DSIStudioCreateSrc(), name="create_src")
    gqi_recon = pe.Node(DSIStudioGQIReconstruction(), name="gqi_recon")

    workflow.connect([
        (inputnode, create_src, [('dwi_file', 'input_nifti_file'),
                                 ('bval_file', 'input_bvals_file'),
                                 ('bvec_file', 'input_bvecs_file')]),
        (create_src, gqi_recon, [('output_src', 'input_src_file')]),
        (gqi_recon, outputnode, [('output_fib', 'fibgz')])
    ])

    if output_suffix:
        # Save the output in the outputs directory
        ds_gqi_fibgz = pe.Node(ReconDerivativesDataSink(
                                    extension='.fib.gz',
                                    suffix=output_suffix,
                                    compress=True),
                               name='ds_gqi_fibgz',
                               run_without_submitting=True)
        workflow.connect(gqi_recon, 'output_fib', ds_gqi_fibgz, 'in_file')
    return workflow


def init_mrtrix_vanilla_csd_recon_workflow(name="mrtrix_recon", output_suffix="", params={}):
    inputnode = pe.Node(niu.IdentityInterface(fields=qsiprep_output_names),
                        name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['fibgz']),
        name="outputnode")

    workflow = pe.Workflow(name=name)
    create_mif = pe.Node(MRConvert(), name='create_mif')
    estimate_response = pe.Node(ResponseSD(**params['response']), 'estimate_response')
    estimate_fod = pe.Node(EstimateFOD(**params['fod']), 'estimate_fod')

    workflow.connect([
        (inputnode, create_mif, [('dwi_file', 'in_file'),
                                 ('bval_file', 'in_bval'),
                                 ('bvec_file', 'in_bvec')]),
        (create_mif, estimate_response, [('out_file', 'in_file')]),
        (inputnode, estimate_response, [('dwi_mask', 'in_mask')]),
        (estimate_response, estimate_fod, [('wm_file', 'wm_txt'),
                                           ('gm_file', 'gm_txt'),
                                           ('csf_file', 'csf_txt')])

    ])
    return workflow

def init_dsi_studio_connectivity_workflow(name="dsi_studio_connectivity", n_procs=1,
                                          params={}, output_suffix=""):
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=qsiprep_output_names + ['fibgz', 'atlas_configs']),
        name="inputnode")
    outputnode = pe.Node(niu.IdentityInterface(fields=['matfile']),
                         name="outputnode")
    workflow = pe.Workflow(name=name)
    calc_connectivity = pe.Node(DSIStudioAtlasGraph(n_procs=n_procs, **params),
                                name='calc_connectivity')
    workflow.connect([
        (inputnode, calc_connectivity, [('atlas_configs', 'atlas_configs'),
                                        ('fibgz', 'input_fib')]),
        (calc_connectivity, outputnode, [('connectivity_matfile', 'matfile')])
    ])
    if output_suffix:
        # Save the output in the outputs directory
        ds_connectivity = pe.Node(ReconDerivativesDataSink(suffix=output_suffix),
                                  name='ds_' + name,
                                  run_without_submitting=True)
        workflow.connect(calc_connectivity, 'connectivity_matfile', ds_connectivity, 'in_file')
    return workflow


def init_dsi_studio_export_workflow(name="dsi_studio_export", params={}, output_suffix=""):
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=qsiprep_output_names + ['fibgz']),
        name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['gfa', 'fa0', 'fa1', 'fa2', 'iso']),
        name="outputnode")
    workflow = pe.Workflow(name=name)
    export = pe.Node(DSIStudioExport(to_export="gfa,fa0,fa1,fa2,fa3,iso"), name='export')
    fixhdr_nodes = {}
    for scalar_name in ['gfa', 'fa0', 'fa1', 'fa2', 'iso']:
        output_name = scalar_name + '_file'
        fixhdr_nodes[scalar_name] = pe.Node(FixDSIStudioExportHeader(), name='fix_'+scalar_name)
        connections = [(export, fixhdr_nodes[scalar_name], [(output_name, 'dsi_studio_nifti')]),
                       (inputnode, fixhdr_nodes[scalar_name], [('dwi_file',
                                                                'correct_header_nifti')]),
                       (fixhdr_nodes[scalar_name], outputnode, [('out_file', scalar_name)])]
        if output_suffix:
            connections += [(fixhdr_nodes[scalar_name],
                             pe.Node(ReconDerivativesDataSink(desc=scalar_name, suffix=output_suffix),
                                     name='ds_%s_%s' % (name, scalar_name)),
                            [('out_file', 'in_file')])]
        workflow.connect(connections)

    workflow.connect([(inputnode, export, [('fibgz', 'input_file')])])

    return workflow


def init_controllability_workflow(name="controllability", output_suffix="", params={}):
    inputnode = pe.Node(niu.IdentityInterface(fields=qsiprep_output_names + ['matfile']),
                        name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['matfile']),
        name="outputnode")

    calc_control = pe.Node(Controllability(**params), name='calc_control')
    workflow = pe.Workflow(name=name)
    workflow.connect([
        (inputnode, calc_control, [('matfile', 'matfile')]),
        (calc_control, outputnode, [('controllability', 'matfile')])
    ])
    if output_suffix:
        # Save the output in the outputs directory
        ds_control = pe.Node(ReconDerivativesDataSink(suffix=output_suffix),
                             name='ds_' + name,
                             run_without_submitting=True)
        workflow.connect(calc_control, 'controllability', ds_control, 'in_file')
    return workflow


def init_discard_repeated_samples_workflow(name="discard_repeats", output_suffix="",
                                           space="T1w", params={}):
    """Remove a sample if a similar direction/gradient has already been sampled."""
    inputnode = pe.Node(niu.IdentityInterface(fields=qsiprep_output_names),
                        name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['dwi_file', 'bval_file', 'bvec_file', 'local_bvec_file']),
        name="outputnode")
    workflow = pe.Workflow(name=name)

    discard_repeats = pe.Node(RemoveDuplicates(**params), name='discard_repeats')
    workflow.connect([
        (inputnode, discard_repeats, [('dwi_file', 'dwi_file'), ('bval_file', 'bval_file'),
                                      ('bvec_file', 'bvec_file')]),
        (discard_repeats, outputnode, [('dwi_file', 'dwi_file'), ('bval_file', 'bval_file'),
                                       ('bvec_file', 'bvec_file')])
    ])

    return workflow


def workflow_from_spec(node_spec):
    software = node_spec.get("software", "qsiprep")
    output_suffix = node_spec.get("output_suffix", "")
    node_name = node_spec.get("name", None)
    parameters = node_spec.get("parameters", {})

    if node_name is None:
        raise Exception('Node %s must have a "name" attribute' % node_spec)
    kwargs = {"name": node_name,
              "output_suffix": output_suffix,
              "params": parameters}
    if software == "DSI Studio":
        if node_spec["action"] == "reconstruction":
            return init_dsi_studio_recon_wf(**kwargs)
        if node_spec["action"] == "export":
            return init_dsi_studio_export_workflow(**kwargs)
        if node_spec["action"] == "connectivity":
            return init_dsi_studio_connectivity_workflow(**kwargs)
    elif software == "MRTrix3":
        if node_spec["action"] == "csd":
            return init_mrtrix_vanilla_csd_recon_workflow(**kwargs)
    else:
        if node_spec['action'] == "controllability":
            return init_controllability_workflow(**kwargs)
        if node_spec['action'] == 'discard_repeated_samples':
            return init_discard_repeated_samples_workflow(**kwargs)
    raise Exception("Unknown node %s", node_spec)


def _as_connections(attr_list):
    return [(item, item) for item in attr_list]


def _get_wf_name(dwi_file):
    basedir, fname, ext = split_filename(dwi_file)
    tokens = fname.split("_")
    return "_".join(tokens[:-1])
