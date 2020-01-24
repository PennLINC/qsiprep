#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Class and utilities for testing the workflows module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

"""

from pathlib import Path
import json
import tempfile
import unittest
import logging
from networkx.exception import NetworkXUnfeasible

from nipype.pipeline import engine as pe
from nipype.interfaces.base import isdefined
from nipype.interfaces import utility as niu

logging.disable(logging.INFO)  # <- do we really want to do this?


class TestWorkflow(unittest.TestCase):
    ''' Subclass for test within the workflow module.
    invoke tests with ``python -m unittest discover test'''

    def assertIsAlmostExpectedWorkflow(self, expected_name, expected_interfaces,
                                       expected_inputs, expected_outputs,
                                       actual):
        ''' somewhat hacky way to confirm workflows are as expected, but with low confidence '''
        self.assertIsInstance(actual, pe.Workflow)
        self.assertEqual(expected_name, actual.name)

        # assert it has the same nodes
        actual_nodes = [actual.get_node(name)
                        for name in actual.list_node_names()]
        actual_interfaces = [node.interface.__class__.__name__
                             for node in actual_nodes]

        # assert lists equal
        self.assertIsSubsetOfList(expected_interfaces, actual_interfaces)
        self.assertIsSubsetOfList(actual_interfaces, expected_interfaces)

        # assert expected inputs, outputs exist
        actual_inputs, actual_outputs = self.get_inputs_outputs(actual_nodes)

        self.assertIsSubsetOfList(expected_outputs, actual_outputs)
        self.assertIsSubsetOfList(expected_inputs, actual_inputs)

    def assertIsSubsetOfList(self, expecteds, actuals):
        for expected in expecteds:
            self.assertIn(expected, actuals)

    def get_inputs_outputs(self, nodes):
        def get_io_names(pre, ios):
            return [pre + str(io[0]) for io in ios]

        actual_inputs = []
        actual_outputs = []
        node_tuples = [(node.name, node.inputs.items(), node.outputs.items())
                       for node in nodes]
        for name, inputs, outputs in node_tuples:
            pre = str(name) + "."
            actual_inputs += get_io_names(pre, inputs)

            pre = pre if pre[0:-1] != 'inputnode' else ""
            actual_outputs += get_io_names(pre, outputs)

        return actual_inputs, actual_outputs

    def assert_circular(self, workflow, circular_connections):
        ''' check key paths in workflow by specifying some connections that should induce
        circular paths, which trips a NetworkX error.
        circular_connections is a list of tuples:
            [('from_node_name', 'to_node_name', ('from_node.output_field','to_node.input_field'))]
        '''

        for from_node, to_node, fields in circular_connections:
            from_node = workflow.get_node(from_node)
            to_node = workflow.get_node(to_node)
            workflow.connect([(from_node, to_node, fields)])

            self.assertRaises(NetworkXUnfeasible, workflow.write_graph)

            workflow.disconnect([(from_node, to_node, fields)])

    def assert_inputs_set(self, workflow, additional_inputs={}):
        ''' Check that all mandatory inputs of nodes in the workflow (at the first level) are
        already set. Additionally, check that inputs in additional_inputs are set. An input is
        "set" if it is
            a) defined explicitly (e.g. in the Interface declaration)
            OR
            b) connected to another node's output (e.g. using the workflow.connect method)
        additional_inputs is a dict:
            {'node_name': ['mandatory', 'input', 'fields']}'''
        dummy_node = pe.Node(niu.IdentityInterface(fields=['dummy']), name='DummyNode')
        node_names = [name for name in workflow.list_node_names() if name.count('.') == 0]
        for node_name in set(node_names + list(additional_inputs.keys())):
            node = workflow.get_node(node_name)
            mandatory_inputs = list(node.inputs.traits(mandatory=True).keys())
            other_inputs = additional_inputs[node_name] if node_name in additional_inputs else []
            for field in set(mandatory_inputs + other_inputs):
                if isdefined(getattr(node.inputs, field)):
                    pass
                else:  # not explicitly defined
                    # maybe it is connected to an output
                    with self.assertRaises(Exception):
                        # throws an error if the input is already connected
                        workflow.connect([(dummy_node, node, [('dummy', field)])])


def get_grouping_test_data():
    """Write a number of grouping test datasets to base_path."""

    dataset_desctiption = {
        "Acknowledgements": "",
        "Authors": [],
        "BIDSVersion": "1.0.2",
        "DatasetDOI": "",
        "Funding": "",
        "HowToAcknowledge": "",
        "License": "",
        "Name": "test_data",
        "ReferencesAndLinks": [],
        "template": "project"
    }

    base_dir = tempfile.mkdtemp()
    empty_bids_dir = Path(base_dir) / 'empty_bids'
    empty_bids_dir.mkdir(parents=True, exist_ok=True)

    def write_json(pth, content):
        with pth.open('w') as f:
            json.dump(content, f)

    def make_empty_bids(root, project_name):
        project_root = root / project_name
        project_root.mkdir(parents=True, exist_ok=True)
        (project_root / 'README').touch()
        write_json(project_root / 'dataset_description.json', dataset_desctiption)
        (project_root / 'sub-1' / 'dwi').mkdir(parents=True, exist_ok=True)
        (project_root / 'sub-1' / 'fmap').mkdir(parents=True, exist_ok=True)
        (project_root / 'sub-1' / 'anat').mkdir(parents=True, exist_ok=True)
        return project_root / 'sub-1'

    def write_test_bids(name, files_and_metas):
        test_bids = make_empty_bids(empty_bids_dir, name)
        for fname, meta in files_and_metas:
            _nifti = fname + ".nii.gz"
            _json = fname + '.json'
            (test_bids / _nifti).touch()
            write_json(test_bids / _json, meta)
        return test_bids.parent

    # One dwi, no fmaps
    write_test_bids(
        'easy',
        [('dwi/sub-1_dwi', {'PhaseEncodingDirection': 'j'})])

    write_test_bids(
        'concat1',
        [("dwi/sub-1_run-01_dwi", {'PhaseEncodingDirection': 'j'}),
         ("dwi/sub-1_run-02_dwi", {'PhaseEncodingDirection': 'j'})]
    )

    write_test_bids(
        'opposite',
        [("dwi/sub-1_dir-AP_dwi", {'PhaseEncodingDirection': 'j'}),
         ("dwi/sub-1_dir-PA_dwi", {'PhaseEncodingDirection': 'j-'})])

    write_test_bids(
        'opposite_concat',
        [("dwi/sub-1_dir-AP_run-1_dwi", {'PhaseEncodingDirection': 'j'}),
         ("dwi/sub-1_dir-AP_run-2_dwi", {'PhaseEncodingDirection': 'j'}),
         ("dwi/sub-1_dir-PA_run-1_dwi", {'PhaseEncodingDirection': 'j-'}),
         ("dwi/sub-1_dir-PA_run-2_dwi", {'PhaseEncodingDirection': 'j-'})])

    write_test_bids(
        'phasediff',
        [("dwi/sub-1_dir-AP_run-1_dwi", {'PhaseEncodingDirection': 'j'}),
         ("dwi/sub-1_dir-AP_run-2_dwi", {'PhaseEncodingDirection': 'j'}),
         ("fmap/sub-1_magnitude1", {'PhaseEncodingDirection': 'j'}),
         ("fmap/sub-1_magnitude2", {'PhaseEncodingDirection': 'j'}),
         ("fmap/sub-1_phasediff", {
             'PhaseEncodingDirection': 'j',
             'IntendedFor': ['dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
                             'dwi/sub-1_dir-AP_run-2_dwi.nii.gz']})])

    write_test_bids(
        'epi',
        [("dwi/sub-1_dir-AP_run-1_dwi", {'PhaseEncodingDirection': 'j'}),
         ("dwi/sub-1_dir-AP_run-2_dwi", {'PhaseEncodingDirection': 'j'}),
         ("fmap/sub-1_dir-PA_epi", {
             'PhaseEncodingDirection': 'j-',
             'IntendedFor': ['dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
                             'dwi/sub-1_dir-AP_run-2_dwi.nii.gz']})])

    write_test_bids(
        'separate_fmaps',
        [("dwi/sub-1_dir-AP_run-1_dwi", {'PhaseEncodingDirection': 'j'}),
         ("dwi/sub-1_dir-AP_run-2_dwi", {'PhaseEncodingDirection': 'j'}),
         ("fmap/sub-1_dir-PA_run-1_epi", {
             'PhaseEncodingDirection': 'j-',
             'IntendedFor': ['dwi/sub-1_dir-AP_run-1_dwi.nii.gz']}),
         ("fmap/sub-1_dir-PA_run-2_epi", {
             'PhaseEncodingDirection': 'j-',
             'IntendedFor': ['dwi/sub-1_dir-AP_run-2_dwi.nii.gz']}),
         ])

    write_test_bids(
        'mixed_fmaps',
        [("dwi/sub-1_dir-AP_run-1_dwi", {'PhaseEncodingDirection': 'j'}),
         ("dwi/sub-1_dir-PA_run-2_dwi", {'PhaseEncodingDirection': 'j-'}),
         ("fmap/sub-1_dir-PA_run-1_epi", {
             'PhaseEncodingDirection': 'j-',
             'IntendedFor': ['dwi/sub-1_dir-AP_run-1_dwi.nii.gz']}),
         ("fmap/sub-1_dir-AP_run-2_epi", {
             'PhaseEncodingDirection': 'j',
             'IntendedFor': ['dwi/sub-1_dir-PA_run-2_dwi.nii.gz']}),
         ])

    write_test_bids(
        'missing_info',
        [("dwi/sub-1_dir-AP_run-1_dwi", {}),
         ("dwi/sub-1_dir-PA_run-2_dwi", {})]
    )

    write_test_bids(
        'wtf',
        [("dwi/sub-1_run-1_dwi", {}),
         ("dwi/sub-1_run-2_dwi", {}),
         ("dwi/sub-1_dir-AP_run-1_dwi", {'PhaseEncodingDirection': 'j'}),
         ("dwi/sub-1_dir-AP_run-2_dwi", {'PhaseEncodingDirection': 'j'}),
         ("dwi/sub-1_dir-PA_run-1_dwi", {'PhaseEncodingDirection': 'j-'}),
         ("dwi/sub-1_dir-PA_run-2_dwi", {'PhaseEncodingDirection': 'j-'}),
         ("dwi/sub-1_dir-IS_dwi", {'PhaseEncodingDirection': 'k-'}),
         ])

    write_test_bids(
        'appa_fmaps',
        [("dwi/sub-1_dir-AP_run-1_dwi", {'PhaseEncodingDirection': 'j'}),
         ("dwi/sub-1_dir-AP_run-2_dwi", {'PhaseEncodingDirection': 'j'}),
         ("fmap/sub-1_dir-PA_run-1_epi", {
             'PhaseEncodingDirection': 'j-',
             'IntendedFor': ['dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
                             'dwi/sub-1_dir-AP_run-2_dwi.nii.gz']}),
         ("fmap/sub-1_dir-AP_run-2_epi", {
             'PhaseEncodingDirection': 'j',
             'IntendedFor': ['dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
                             'dwi/sub-1_dir-AP_run-2_dwi.nii.gz']}),
         ])

    return empty_bids_dir
