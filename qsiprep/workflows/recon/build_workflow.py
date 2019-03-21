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
from .dynamics import init_controllability_workflow
from .utils import init_conform_dwi_wf, init_discard_repeated_samples_wf
from ...interfaces.images import ConformDwi
from ...interfaces.gradients import ExtractB0s
from ...engine import Workflow
from ..dwi.registration import init_b0_to_anat_registration_wf

LOGGER = logging.getLogger('nipype.interface')
qsiprep_output_names = QsiReconIngress().output_spec.class_editable_traits()
default_connections = [(trait, trait) for trait in qsiprep_output_names]
default_input_set = set(qsiprep_output_names)


def _get_resampled(atlas_configs, atlas_name):
    return atlas_configs[atlas_name]['dwi_resolution_file']


def init_dwi_recon_workflow(dwi_file, do_coreg, workflow_spec, output_dir, reportlets_dir,
                            omp_nthreads, name="recon_wf"):
    atlas_names = workflow_spec.get('atlases', [])
    space = workflow_spec['space']

    input_fields = [
        'dwi_file',
        't1_preproc',
        't1_aparc'
        't1_seg',
        't1_aseg',
        't1_brain_mask',
        't1_preproc',
        't1_csf_probseg',
        't1_gm_probseg',
        't1_wm_probseg',
        'left_inflated_surf',
        'left_midthickness_surf',
        'left_pial_surf',
        'left_smoothwm_surf',
        'right_inflated_surf',
        'right_midthickness_surf',
        'right_pial_surf',
        'right_smoothwm_surf',
        'orig_to_t1_mode_forward_transform',
        't1_2_fsnative_forward_transform',
        't1_2_mni_reverse_transform',
        't1_2_mni_forward_transform',
        'template_brain_mask',
        'template_preproc',
        'template_seg',
        'template_csf_probseg',
        'template_gm_probseg',
        'template_wm_probseg',
    ]

    workflow = Workflow(name=_get_wf_name(dwi_file))
    inputnode = pe.Node(niu.IdentityInterface(fields=input_fields), name='inputnode')
    conform_dwi = pe.Node(ConformDwi(), name="conform_dwi")
    # For doctests
    if not workflow_spec['name'] == 'fake':
        inputnode.inputs.dwi_file = dwi_file

    # Create a b0 template for registration/Resampling
    b0_series = pe.Node(ExtractB0s(), name='b0_series')

    workflow.connect([
        (inputnode, conform_dwi, [('dwi_file', 'dwi_file')]),
        (conform_dwi, b0_series, [('dwi_file', 'dwi_file'),
                                  ('bval_file', 'bval_file'),
                                  ('bvec_file', 'bvec_file')])
    ])

    # If the anatomical pipeline is run outside of qsiprep, coregistration needs to occur
    if do_coreg:
        b0_t1_coreg = init_b0_to_anat_registration_wf(name=b0_t1_coreg)
        resampling_wf = init_recon_trans_wf(working_space=)

    else:
        resampling_wf = init_recon_trans_wf(working_space="T1w")
    """
    # Resample all atlases to dwi_file's resolution
    get_atlases = pe.Node(GetConnectivityAtlases(atlas_names=atlas_names), name='get_atlases')
    # Save the atlases
    if len(atlas_names) > 0:
        workflow.add_nodes([get_atlases])
        if space == "T1w":
            workflow.connect([
                (preprocessed_data, get_atlases,
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
    """












def add_user_nodes():
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


def workflow_from_spec(node_spec):
    """Build a nipype workflow based on a json file.
    """
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
