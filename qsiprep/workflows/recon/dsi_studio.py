"""
DSI Studio workflows
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_dsi_studio_recon_wf
.. autofunction:: init_dsi_studio_connectivity_workflow
.. autofunction:: init_dsi_studio_export_workflow

"""
import json
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
from nipype.utils.filemanip import copyfile, split_filename
from qsiprep.interfaces.dsi_studio import (DSIStudioCreateSrc, DSIStudioGQIReconstruction,
                                           DSIStudioAtlasGraph, DSIStudioExport,
                                           FixDSIStudioExportHeader)

import logging
import os
import os.path as op
from qsiprep.interfaces.bids import QsiprepOutput, ReconDerivativesDataSink
from qsiprep.interfaces.utils import GetConnectivityAtlases
from qsiprep.interfaces.connectivity import Controllability
from qsiprep.interfaces.gradients import RemoveDuplicates
from qsiprep.interfaces.mrtrix import ResponseSD, EstimateFOD, MRConvert

LOGGER = logging.getLogger('nipype.interface')
qsiprep_output_names = QsiprepOutput().output_spec.class_editable_traits()
default_connections = [(trait, trait) for trait in qsiprep_output_names]
default_input_set = set(qsiprep_output_names)


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
