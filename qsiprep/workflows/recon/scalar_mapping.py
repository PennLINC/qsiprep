"""
Summarize and Transform recon outputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



"""

import logging

import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe

from ...engine import Workflow
from ...interfaces.bids import ReconDerivativesDataSink
from ...interfaces.interchange import recon_workflow_input_fields
from ...interfaces.recon_scalars import (
    ReconScalarsDataSink,
    ReconScalarsTableSplitterDataSink,
)
from ...interfaces.scalar_mapping import BundleMapper, TemplateMapper

LOGGER = logging.getLogger("nipype.workflow")


def init_scalar_to_bundle_wf(
    available_anatomical_data, name="scalar_to_bundle", qsirecon_suffix="", params={}
):
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
            summary statistics in tsv format

    """
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=recon_workflow_input_fields
            + ["tck_files", "bundle_names", "recon_scalars", "collected_scalars"]
        ),
        name="inputnode",
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=["bundle_summary"]), name="outputnode")
    workflow = Workflow(name=name)
    bundle_mapper = pe.Node(BundleMapper(**params), name="bundle_mapper")
    ds_bundle_mapper = pe.Node(
        ReconScalarsTableSplitterDataSink(suffix="scalarstats"),
        name="ds_bundle_mapper",
        run_without_submitting=True,
    )
    ds_tdi_summary = pe.Node(
        ReconScalarsTableSplitterDataSink(suffix="tdistats"),
        name="ds_tdi_summary",
        run_without_submitting=True,
    )
    workflow.connect([
        (inputnode, bundle_mapper, [
            ("collected_scalars", "recon_scalars"),
            ("tck_files", "tck_files"),
            ("dwi_ref", "dwiref_image"),
            ("mapping_metadata", "mapping_metadata"),
            ("bundle_names", "bundle_names")]),
        (bundle_mapper, ds_bundle_mapper, [
            ("bundle_summary", "summary_tsv")]),
        (bundle_mapper, outputnode, [
            ("bundle_summary", "bundle_summary")]),
        (bundle_mapper, ds_tdi_summary, [
            ("tdi_stats", "summary_tsv")])
    ])  # fmt:skip

    return workflow


def init_scalar_to_atlas_wf(
    available_anatomical_data,
    name="scalar_to_template",
    qsirecon_suffix="",
    params={},
):
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
            fields=recon_workflow_input_fields + ["recon_scalars", "collected_scalars"]
        ),
        name="inputnode",
    )
    # outputnode = pe.Node(niu.IdentityInterface(fields=["atlas_summaries"]), name="outputnode")
    workflow = Workflow(name=name)
    bundle_mapper = pe.Node(BundleMapper(**params), name="bundle_mapper")
    workflow.connect([
        (inputnode, bundle_mapper, [
            ("recon_scalars", "recon_scalars"),
            ("tck_files", "tck_files"),
            ("dwi_ref", "dwiref_image")])
    ])  # fmt:skip
    if qsirecon_suffix:

        ds_bundle_summaries = pe.Node(
            ReconDerivativesDataSink(desc="bundlemap", qsirecon_suffix=qsirecon_suffix),
            name="ds_bundle_summaries",
            run_without_submitting=True,
        )
        workflow.connect([
            (bundle_mapper, ds_bundle_summaries, [("bundle_summaries", "in_file")])
        ])  # fmt:skip
    return workflow


def init_scalar_to_template_wf(
    available_anatomical_data,
    name="scalar_to_template",
    qsirecon_suffix="",
    params={},
):
    """Maps scalar data to a volumetric template


    Inputs
        recon_scalars
            List of dictionaries containing scalar info

    Outputs

        template_scalars
            List of transformed files

    """
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=recon_workflow_input_fields + ["recon_scalars", "collected_scalars"]
        ),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["template_scalars", "template_scalar_sidecars"]),
        name="outputnode",
    )
    workflow = Workflow(name=name)
    template_mapper = pe.Node(TemplateMapper(**params), name="template_mapper")
    # Datasink will always be used, and the qsirecon-suffix is determined when
    # the scalars are originally calculated
    ds_template_scalars = pe.Node(
        ReconScalarsDataSink(), name="ds_template_scalars", run_without_submitting=True
    )
    workflow.connect([
        (inputnode, template_mapper, [
            ("collected_scalars", "recon_scalars"),
            ("t1_2_mni_forward_transform", "to_template_transform"),
            ("resampling_template", "template_reference_image")]),
        (template_mapper, outputnode, [
            ("template_space_scalars", "template_scalars")]),
        (template_mapper, ds_template_scalars, [
            # Send the resampled files (without metadata) so they don't get cleaned up
            ("template_space_scalars", "resampled_files"),
            ("template_space_scalar_info", "recon_scalars")])
    ])  # fmt:skip

    return workflow


def init_scalar_to_surface_wf(
    available_anatomical_data,
    name="scalar_to_surface",
    qsirecon_suffix="",
    params={},
):
    """Maps scalar data to a surface."""
    raise NotImplementedError()
