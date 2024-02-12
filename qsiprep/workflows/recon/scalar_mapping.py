"""
Converting between file formats
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_mif_to_fibgz_wf
.. autofunction:: init_fibgz_to_mif_wf

"""
import logging

import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe

from ...engine import Workflow
from ...interfaces.bids import ReconDerivativesDataSink
from ...interfaces.interchange import recon_workflow_input_fields
from ...interfaces.scalar_mapping import BundleMapper

LOGGER = logging.getLogger('nipype.workflow')


def init_scalar_to_bundle_wf(omp_nthreads, available_anatomical_data,
                             name="scalar_to_bundle", output_suffix="", params={}):
    """Map scalar images to bundles



    Inputs
        tck_files
            MRtrix3 format tck files for each bundle
        bundle_names
            Names that describe which bundles are present in `tck_files`
        recon_scalars
            List of dictionaries containing scalar info

    Outputs

        bundle_summaries
            summary statistics in tsv formar

    """
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=recon_workflow_input_fields +
            ["tck_files", "bundle_names", "recon_scalars", "collected_scalars"]),
        name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['bundle_summary']), name="outputnode")
    workflow = Workflow(name=name)
    bundle_mapper = pe.Node(
        BundleMapper(**params),
        name="bundle_mapper")
    workflow.connect([
        (inputnode, bundle_mapper, [
            ("collected_scalars", "recon_scalars"),
            ("tck_files", "tck_files"),
            ("dwi_ref", "dwiref_image"),
            ("bundle_names", "bundle_names")]),
        (bundle_mapper, outputnode, [
            ("bundle_summary", "bundle_summary")])
    ])
    if output_suffix:

        ds_bundle_summaries = pe.Node(
            ReconDerivativesDataSink(desc="bundlemap",
                                     suffix=output_suffix),
            name='ds_bundle_summaries',
            run_without_submitting=True)
        workflow.connect([
            (bundle_mapper, ds_bundle_summaries, [("bundle_summary", "in_file")])
        ])

    return workflow


def init_scalar_to_atlas_wf(omp_nthreads, available_anatomical_data,
                            name="scalar_to_template", output_suffix="", params={}):
    """Map scalar images to atlas regions

    Inputs
        tck_files
            MRtrix3 format tck files for each bundle
        bundle_names
            Names that describe which bundles are present in `tck_files`
        recon_scalars
            List of dictionaries containing scalar info

    Outputs
        bundle_summaries
            summary statistics in tsv format

    """
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=recon_workflow_input_fields +
            ["tck_files", "bundle_names", "recon_scalars"]),
        name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['bundle_summaries']), name="outputnode")
    workflow = Workflow(name=name)
    bundle_mapper = pe.Node(
        BundleMapper(**params),
        name="bundle_mapper")
    workflow.connect([
        (inputnode, bundle_mapper, [
            ("recon_scalars", "recon_scalars"),
            ("tck_files", "tck_files"),
            ("dwi_ref", "dwiref_image")])
    ])
    if output_suffix:

        ds_bundle_summaries = pe.Node(
            ReconDerivativesDataSink(desc="bundlemap",
                                     suffix=output_suffix),
            name='ds_bundle_summaries',
            run_without_submitting=True)
        workflow.connect([
            (bundle_mapper, ds_bundle_summaries, [("bundle_summaries", "in_file")])
        ])
    return workflow


def init_scalar_to_template_wf(omp_nthreads, available_anatomical_data,
                               name="scalar_to_template", output_suffix="", params={}):
    """Maps scalar data to a volumetric template"""

    return workflow


def init_scalar_to_surface_wf(omp_nthreads, available_anatomical_data,
                              name="scalar_to_surface", output_suffix="", params={}):
    """Maps scalar data to a surface."""
    raise NotImplementedError()
    return workflow
