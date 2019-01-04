
# coding: utf-8

# # Design of `qsirecon`
# 
# This document describes how data is read by and processed by `qsirecon`, which takes the output from `qsiprep` and performs reconstruction/tracking/etc on these data. 
# 
# ## How does pybids deal with derivatives?
# 
# The output from `qsiprep` goes into a directory that mirrors the input data. We need to get the masks, bvals, bvecs dwi data, anatomical data, transforms and tissue probability maps from the `qsiprep` output. Here is a typical output from a `qsiprep` run (omitting figures):
# 
# ```
# qsiprep/sub-abcd.html
# qsiprep/sub-abcd
# qsiprep/sub-abcd/dwi
# qsiprep/sub-abcd/dwi/sub-abcd_space-T1w_desc-brain_mask.nii.gz
# qsiprep/sub-abcd/dwi/sub-abcd_space-T1w_desc-preproc.bvec
# qsiprep/sub-abcd/dwi/sub-abcd_space-MNI152NLin2009cAsym_desc-preproc_dwi.nii.gz
# qsiprep/sub-abcd/dwi/sub-abcd_space-T1w_bvec.nii.gz
# qsiprep/sub-abcd/dwi/sub-abcd_space-MNI152NLin2009cAsym_dwiref.nii.gz
# qsiprep/sub-abcd/dwi/sub-abcd_confounds.tsv
# qsiprep/sub-abcd/dwi/sub-abcd_space-MNI152NLin2009cAsym_b0series.nii.gz
# qsiprep/sub-abcd/dwi/sub-abcd_space-T1w_desc-preproc.bval
# qsiprep/sub-abcd/dwi/sub-abcd_space-T1w_b0series.nii.gz
# qsiprep/sub-abcd/dwi/sub-abcd_space-MNI152NLin2009cAsym_desc-preproc.bvec
# qsiprep/sub-abcd/dwi/sub-abcd_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz
# qsiprep/sub-abcd/dwi/sub-abcd_space-MNI152NLin2009cAsym_bvec.nii.gz
# qsiprep/sub-abcd/dwi/sub-abcd_space-T1w_dwiref.nii.gz
# qsiprep/sub-abcd/dwi/sub-abcd_space-T1w_desc-preproc.nii.gz
# qsiprep/sub-abcd/dwi/sub-abcd_space-MNI152NLin2009cAsym_desc-preproc.bval
# qsiprep/sub-abcd/anat
# qsiprep/sub-abcd/anat/sub-abcd_dseg.nii.gz
# qsiprep/sub-abcd/anat/sub-abcd_space-MNI152NLin2009cAsym_dseg.nii.gz
# qsiprep/sub-abcd/anat/sub-abcd_space-MNI152NLin2009cAsym_label-CSF_probseg.nii.gz
# qsiprep/sub-abcd/anat/sub-abcd_label-WM_probseg.nii.gz
# qsiprep/sub-abcd/anat/sub-abcd_space-MNI152NLin2009cAsym_label-WM_probseg.nii.gz
# qsiprep/sub-abcd/anat/sub-abcd_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5
# qsiprep/sub-abcd/anat/sub-abcd_space-MNI152NLin2009cAsym_label-GM_probseg.nii.gz
# qsiprep/sub-abcd/anat/sub-abcd_label-GM_probseg.nii.gz
# qsiprep/sub-abcd/anat/sub-abcd_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz
# qsiprep/sub-abcd/anat/sub-abcd_desc-brain_mask.nii.gz
# qsiprep/sub-abcd/anat/sub-abcd_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5
# qsiprep/sub-abcd/anat/sub-abcd_label-CSF_probseg.nii.gz
# qsiprep/sub-abcd/anat/sub-abcd_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz
# qsiprep/sub-abcd/anat/sub-abcd_from-orig_to-T1w_mode-image_xfm.txt
# qsiprep/sub-abcd/anat/sub-abcd_desc-preproc_T1w.nii.gz
# qsiprep/dataset_description.json
# ```

# In[1]:


from bids.layout import BIDSLayout

qsiprep_output = "/Users/mcieslak/projects/qsiprep/scratch/abcd_test/test_output/qsiprep/"
orig_bids = "/Users/mcieslak/projects/test_bids_data/downsampled/abcd"

qp_layout = BIDSLayout(qsiprep_output)

# Get the preprocessed dwi niftis
dwi_files = [f.filename for f in qp_layout.get(type='dwi', extensions=['nii', 'nii.gz'])]

print(dwi_files)


# We see there is one output in MNI space and another in T1w space. We only want to work on the one in T1w for this pipeline

# In[2]:


get_ipython().run_cell_magic('writefile', 'pipeline.json', '{\n  "name": "dsistudio_pipeline",\n  "space": "T1w",\n  "nodes": [\n    {\n      "name": "dsistudio_gqi",\n      "software": "DSI Studio",\n      "action": "reconstruction",\n      "input": "qsiprep",\n      "output": [\n        "fibgz",\n        "gfa"\n      ],\n      "parameters": {\n        "method": "gqi"\n      }\n    },\n    {\n      "name": "deterministic_tracking",\n      "software": "DSI Studio",\n      "action": "tractography",\n      "input": "dsistudio_gqi",\n      "output": [\n        "connectivity"\n      ],\n      "parameters": {\n        "atlas": [\n          "schaeffer200"\n        ]\n      }\n    },\n    {\n      "name": "controlability",\n      "input": "deterministic_tracking",\n      "output": [\n        "summary"\n      ]\n    }\n  ]\n}')


# In[4]:


get_ipython().run_cell_magic('writefile', 'pipeline.json', '{\n  "name": "dsistudio_pipeline",\n  "space": "T1w",\n  "nodes": [\n    {\n      "name": "dsistudio_gqi",\n      "software": "DSI Studio",\n      "action": "reconstruction",\n      "input": "qsiprep",\n      "output": ["fibgz"],\n      "parameters": {\n        "method": "gqi"\n      }\n    },\n    {\n      "name": "scalar_export",\n      "software": "DSI Studio",\n      "action": "export",\n      "input": "dsistudio_gqi",\n      "output": ["gfa"],\n      "parameters": {}\n    }\n  ]\n}')


# In[5]:


import json
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
import logging
import os
from qsiprep.interfaces.dsi_studio import (DSIStudioCreateSrc, DSIStudioGQIReconstruction,
                                           DSIStudioTracking, DSIStudioExport)
from qsiprep.interfaces.bids import QsiprepOutput

LOGGER = logging.getLogger('nipype.interface')
json_file = "pipeline.json"
with open(json_file, "r") as f:
    spec = json.load(f)

work_space = spec["space"]
dwi_files = [f for f in dwi_files if 'space-'+work_space in f]
print(dwi_files)


# In[6]:


inputnode = pe.Node(niu.IdentityInterface(fields=['dwi_file']),
                    name="inputnode")
inputnode.iterables = [('dwi_file', dwi_files)]

# Get bvals, bvecs
grab_schemes = pe.Node(QsiprepOutput(), name="grab_schemes")


# Here are some functions that create the remaining nodes and connect them to the scheme grabber node.

# In[41]:


qsiprep_outputs = QsiprepOutput().output_spec.class_editable_traits()

def init_dsi_studio_recon_wf(method="gqi", name="dsi_studio_recon"):
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=qsiprep_outputs),
        name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['fibgz']),
        name="outputnode")
    workflow = pe.Workflow(name=name)
    create_src = pe.Node(DSIStudioCreateSrc(), name="create_src")
    gqi_recon = pe.Node(DSIStudioGQIReconstruction(), name="gqi_recon")
    # ds_gqi_fibgz = 
    workflow.connect([
        (inputnode, create_src, [('dwi_file', 'input_nifti_file'),
                                 ('bval_file', 'input_bvals_file'),
                                 ('bvec_file', 'input_bvecs_file')]),
        (create_src, gqi_recon, [('output_src', 'input_src_file')]),
        (gqi_recon, outputnode, [('output_fib', 'fibgz')]),
    ])
    return workflow


def init_dsi_studio_connectivity_workflow(space, name="dsi_studio_connectivity"):
    inputnode = pe.Node(
    niu.IdentityInterface(
            fields=qsiprep_outputs + ['fibgz']),
        name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['matfiles']),
        name="outputnode")
    workflow = pe.Workflow(name=name)
    return workflow


def init_dsi_studio_export_workflow(space, name="dsi_studio_export"):
    inputnode = pe.Node(
    niu.IdentityInterface(
            fields=qsiprep_outputs + ['fibgz']),
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


def workflow_from_spec(node_spec, space):
    software = node_spec.get("software", "qsiprep")
    if software == "DSI Studio":
        if node_spec["action"] == "reconstruction":
            return init_dsi_studio_recon_wf(name=node_spec["name"], **node_spec["parameters"])
        if node_spec["action"] == "export":
            return init_dsi_studio_export_workflow(name=node_spec["name"], space=space)

        
# Create a workflow
workflow_name = spec['name']
workflow = pe.Workflow(name=workflow_name)
workflow.base_dir = os.getcwd()
workflow.connect([(inputnode, grab_schemes, [('dwi_file', 'in_file')])])

# First pass: add the workflows
for node_spec in spec["nodes"]:
    workflow.add_nodes([workflow_from_spec(node_spec, space=spec['space'])])


# Now we need to connect the inputs and outputs from each of the nodes

# In[42]:


default_connections = [(trait, trait) for trait in qsiprep_outputs]
default_input_set = set(qsiprep_outputs)

def get_connections(src, dest):
    src_outputs = set(src.outputs.get().keys())
    dest_inputs = set(dest.inputs.get().keys())
    overlap = src_outputs.intersection(dest_inputs) - default_input_set
    return [(trait, trait) for trait in overlap]


for node_spec in spec['nodes']:
    node_name = node_spec['name'] + ".inputnode"
    node = workflow.get_node(node_name)
    # directly connect all the qsiprep outputs to every node
    workflow.connect([(grab_schemes, node, default_connections)])
    if not node_spec['input'] == 'qsiprep':
        upstream_node_name = node_spec['input'] + '.outputnode'
        upstream_node = workflow.get_node(upstream_node_name)
        # Connect outputs from the inputnode to spec of this node
        connections = get_connections(upstream_node, node)
        print("connecting",(upstream_node, node, connections))
        workflow.connect([(upstream_node, node, connections)])
    
workflow.config['execution']['stop_on_first_crash'] = 'true'
workflow.run()


# In[10]:


get_ipython().run_line_magic('qtconsole', '')

