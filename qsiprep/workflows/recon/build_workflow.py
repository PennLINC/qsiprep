import json
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
from nipype.utils.filemanip import copyfile, split_filename

import logging
import os
import os.path as op
from qsiprep.interfaces.bids import QsiprepOutput, ReconDerivativesDataSink
from qsiprep.interfaces.utils import GetConnectivityAtlases
from qsiprep.interfaces.connectivity import Controllability
from qsiprep.interfaces.gradients import RemoveDuplicates
from qsiprep.interfaces.mrtrix import ResponseSD, EstimateFOD, MRConvert, MRTrixGradientTable
from qsiprep.interfaces import ConformDwi
from .dsi_studio import (init_dsi_studio_recon_wf, init_dsi_studio_export_wf,
                         init_dsi_studio_connectivity_wf)
from .dipy import init_dipy_brainsuite_shore_recon_wf
from .mrtrix import init_mrtrix_vanilla_csd_recon_wf
from .converters import init_mif_to_fibgz_wf

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
    workflow.add_nodes([preprocessed_data])

    # Save the atlases
    if len(workflow_spec['atlases']):
        workflow.add_nodes([get_atlases])
        if space == "T1w":
            workflow.connect([
                (preprocessed_data, get_atlases,
                 [('t1_2_mni_reverse_transform', 'forward_transform')])])
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


def init_controllability_workflow(name="controllability", output_suffix="", params={}):
    """Calculates network controllability from connectivity matrices.

    Calculates modal and average controllability using the method of Gu et al. 2015.

    Inputs:
    -------

    matfile
        MATLAB format connectivity matrices from DSI Studio connectivity, MRTrix
        connectivity or Dipy Connectivity.

    Outputs:
    --------

    matfile
        MATLAB format controllability values for each node in each connectivity matrix
        in the input file.


    """
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


def init_discard_repeated_samples_wf(name="discard_repeats", output_suffix="",
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


def init_conform_dwi_wf(name="conform_dwi", output_suffix="", params={}):
    """If data were preprocessed elsewhere, ensure the gradients and images
    conform to LPS+ before running other parts of the pipeline."""
    inputnode = pe.Node(niu.IdentityInterface(fields=qsiprep_output_names),
                        name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['dwi_file', 'bval_file', 'bvec_file', 'b_file']),
        name="outputnode")
    workflow = pe.Workflow(name=name)
    conform = pe.Node(ConformDwi(), name="conform_dwi")
    grad_table = pe.Node(MRTrixGradientTable(), name="grad_table")
    workflow.connect([
        (inputnode, conform, [('dwi_file', 'dwi_file')]),
        (conform, grad_table, [('bval_file', 'bval_file'),
                               ('bvec_file', 'bvec_file')]),
        (grad_table, outputnode, [('gradient_file', 'b_file')]),
        (conform, outputnode, [('bval_file', 'bval_file'),
                               ('bvec_file', 'bvec_file'),
                               ('dwi_file', 'dwi_file')])
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

    # qsiprep operations
    else:
        if node_spec['action'] == "controllability":
            return init_controllability_workflow(**kwargs)
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
    return "_".join(tokens[:-1])
