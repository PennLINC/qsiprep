import json
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
from nipype.utils.filemanip import copyfile, split_filename

import logging
import os
import os.path as op
from qsiprep.interfaces.dsi_studio import (DSIStudioCreateSrc, DSIStudioGQIReconstruction,
                                           DSIStudioAtlasGraph, DSIStudioExport)
from qsiprep.interfaces.bids import QsiprepOutput, DerivativesDataSink
from qsiprep.interfaces.utils import GetConnectivityAtlases
from qsiprep.interfaces.connectivity import Controllability

LOGGER = logging.getLogger('nipype.interface')
qsiprep_output_names = QsiprepOutput().output_spec.class_editable_traits()
default_connections = [(trait, trait) for trait in qsiprep_output_names]
default_input_set = set(qsiprep_output_names)


def init_dwi_recon_workflow(dwi_file, workflow_spec, name="recon_wf"):
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

    # Read nodes from workflow spec, make sure we can implement them
    nodes_to_add = []
    for node_spec in workflow_spec['nodes']:
        new_node = workflow_from_spec(node_spec, space=space)
        if new_node is None:
            raise Exception("Unable to create a node for %s", node_spec)
        nodes_to_add.append(new_node)
    workflow.add_nodes(nodes_to_add)

    # Now that all nodes are in the workflow, connect them
    for node_spec in workflow_spec['nodes']:

        # get the nipype node object
        node_name = node_spec['name'] + ".inputnode"
        node = workflow.get_node(node_name)

        # directly connect all the qsiprep outputs to every node
        workflow.connect([(preprocessed_data, node, default_connections)])

        # connect the outputs from the upstream node to this node
        if not node_spec['input'] == 'qsiprep':
            upstream_node_name = node_spec['input'] + '.outputnode'
            upstream_node = workflow.get_node(upstream_node_name)

            # Connect outputs from the inputnode to spec of this node
            connections, overwrite = get_connections(upstream_node, node)
            LOGGER.info("connecting %s", (upstream_node, node, connections))
            workflow.connect([(upstream_node, node, connections)])

            # if the upstream node provides an updated version of the qsiprep output, use it
            for to_overwrite in overwrite:
                workflow.disconnect([(preprocessed_data, node, [(to_overwrite, to_overwrite)])])
                workflow.connect([(upstream_node, node, [(to_overwrite, to_overwrite)])])

        # If it's a connectivity calculation, send it the atlas configs
        if node_spec['action'] == 'connectivity':
            workflow.connect([(get_atlases, node,
                               [('atlas_configs', 'atlas_configs')])])
    return workflow


def get_connections(src, dest):
    src_outputs = set(src.outputs.get().keys())
    dest_inputs = set(dest.inputs.get().keys())
    overlap = src_outputs.intersection(dest_inputs)
    new_connections = [(trait, trait) for trait in overlap - default_input_set]
    overwritten = overlap.intersection(default_input_set)
    return new_connections, overwritten


def init_dsi_studio_recon_wf(space, method="gqi", name="dsi_studio_recon"):
    inputnode = pe.Node(niu.IdentityInterface(fields=qsiprep_output_names),
                        name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['fibgz']),
        name="outputnode")
    workflow = pe.Workflow(name=name)
    create_src = pe.Node(DSIStudioCreateSrc(), name="create_src")
    gqi_recon = pe.Node(DSIStudioGQIReconstruction(), name="gqi_recon")

    # Save the output in the outputs directory
    ds_gqi_fibgz = pe.Node(DerivativesDataSink(
                                space=space,
                                extension='.fib.gz',
                                compress=True),
                           name='ds_gqi_fibgz',
                           run_without_submitting=True)
    workflow.connect([
        (inputnode, create_src, [('dwi_file', 'input_nifti_file'),
                                 ('bval_file', 'input_bvals_file'),
                                 ('bvec_file', 'input_bvecs_file')]),
        (create_src, gqi_recon, [('output_src', 'input_src_file')]),
        (gqi_recon, outputnode, [('output_fib', 'fibgz')]),
        (inputnode, ds_gqi_fibgz, [('dwi_file', 'source_file')]),
        (gqi_recon, ds_gqi_fibgz, [('output_fib', 'in_file')])
    ])
    return workflow


def init_dsi_studio_connectivity_workflow(name="dsi_studio_connectivity", n_procs=1,
                                          **kwargs):
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=qsiprep_output_names + ['fibgz', 'atlas_configs']),
        name="inputnode")
    outputnode = pe.Node(niu.IdentityInterface(fields=['matfile']),
                         name="outputnode")
    workflow = pe.Workflow(name=name)
    calc_connectivity = pe.Node(DSIStudioAtlasGraph(n_procs=n_procs, **kwargs),
                                name='calc_connectivity')
    workflow.connect([
        (inputnode, calc_connectivity, [('atlas_configs', 'atlas_configs'),
                                        ('fibgz', 'input_fib')]),
        (calc_connectivity, outputnode, [('connectivity_matfile', 'matfile')])
    ])
    return workflow


def init_dsi_studio_export_workflow(space, name="dsi_studio_export"):
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=qsiprep_output_names + ['fibgz']),
        name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['gfa', 'fa0', 'fa1', 'fa2', 'fa3', 'iso']),
        name="outputnode")
    workflow = pe.Workflow(name=name)
    export = pe.Node(DSIStudioExport(to_export="gfa,fa0,fa1,fa2,fa3,iso"), name='export')

    workflow.connect([
        (inputnode, export, [('fibgz', 'input_file')]),
        (export, outputnode, [('gfa_file', 'gfa'), ('fa0_file', 'fa0'), ('fa1_file', 'fa1'),
                              ('fa2_file', 'fa2'), ('fa3_file', 'fa3'), ('iso_file', 'iso')])
    ])

    return workflow


def init_controllability_workflow(name="controllability"):
    inputnode = pe.Node(niu.IdentityInterface(fields=qsiprep_output_names + ['matfile']),
                        name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['matfile']),
        name="outputnode")

    calc_control = pe.Node(Controllability(), name='calc_control')
    workflow = pe.Workflow(name=name)
    workflow.connect([
        (inputnode, calc_control, [('matfile', 'matfile')]),
        (calc_control, outputnode, [('controllability', 'matfile')])
    ])
    return workflow


def workflow_from_spec(node_spec, space):
    software = node_spec.get("software", "qsiprep")
    if software == "DSI Studio":
        if node_spec["action"] == "reconstruction":
            return init_dsi_studio_recon_wf(name=node_spec["name"], space=space,
                                            **node_spec["parameters"])
        if node_spec["action"] == "export":
            return init_dsi_studio_export_workflow(name=node_spec["name"], space=space)

        if node_spec["action"] == "connectivity":
            return init_dsi_studio_connectivity_workflow(name=node_spec["name"], n_procs=1,
                                                         **node_spec["parameters"])
    else:
        if node_spec['action'] == "controllability":
            return init_controllability_workflow(name=node_spec["name"])


def _get_wf_name(dwi_file):
    basedir, fname, ext = split_filename(dwi_file)
    tokens = fname.split("_")
    return "_".join(tokens[1:-1])
