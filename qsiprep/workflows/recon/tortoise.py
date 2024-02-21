"""
TORTOISE recon workflows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. autofunction:: init_tortoise_estimate_wf

"""
import logging

import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
from nipype.interfaces.base import traits

from ...engine import Workflow
from ...interfaces.bids import ReconDerivativesDataSink
from ...interfaces.interchange import recon_workflow_input_fields
from ...interfaces.recon_scalars import TORTOISEReconScalars, ReconScalarsDataSink
from ...engine import Workflow

from ...interfaces.recon_scalars import TORTOISEReconScalars
from qsiprep.interfaces.tortoise import (
    ComputeADMap,
    ComputeFAMap,
    ComputeLIMap,
    ComputeMAPMRI_NG,
    ComputeMAPMRI_PA,
    ComputeMAPMRI_RTOP,
    ComputeRDMap,
    EstimateMAPMRI,
    EstimateTensor,
    TORTOISEConvert
)


LOGGER = logging.getLogger('nipype.interface')

CITATIONS = {
    "dhollander": "(@dhollander2019response, @dhollander2016unsupervised)",
    "msmt_5tt": "(@msmt5tt)",
    "csd": "(@originalcsd, @tournier2007robust)",
    "msmt_csd": "(@originalcsd, @msmt5tt)"
}


def init_tortoise_estimator_wf(
    omp_nthreads,
    available_anatomical_data,
    name="tortoise_recon",
    qsirecon_suffix="", params={}):
    """Run estimators from TORTOISE.

    This workflow may run ``EstimateTensor`` and/or ``EstimateMAPMRI``
    depending on the configuration.


    Inputs

        *Default qsiprep inputs*

    Outputs



    Params

        estimate_tensor: dict
            parameters for estimating a tensor fit. A minimal example would be
            ``{"bval_cutoff": 2400, "reg_mode": "WLLS"}``
        estimate_mapmri: dict
            parameters for EstimateMAMPRI. A minimal example would be
            ``{"map_order": 4}``.
        estimate_tensor_separately: bool
            If you're estimating MAPMRI, should the tensor estimation occur
            first outside of the call to ``EstimateMAPMRI``? Setting to
            ``True`` would require entries for both ``"estimate_tensor"``
            and ``"estimate_mapmri"``.

    """
    inputnode = pe.Node(
        niu.IdentityInterface(fields=recon_workflow_input_fields),
        name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                # Tensor fit and derivatives
                'dt_image', 'fa_image', 'ad_image', 'eigvec_image',
                'gm_odf', 'gm_txt', 'csf_odf',
                'csf_txt', 'scalar_image_info', 'recon_scalars']),
        name="outputnode")
    workflow = Workflow(name=name)
    recon_scalars = pe.Node(
        TORTOISEReconScalars(qsirecon_suffix=qsirecon_suffix),
        name="recon_scalars")
    plot_reports = params.pop("plot_reports", True)
    desc = "TORTOISE Reconstruction\n\n: " + \
        "Methods implemented in TORTOISE (@tortoisev3) were used for reconstruction. "

    tensor_opts = params.get("estimate_tensor", {})
    estimate_tensor_separately = params.get("estimate_tensor_separately", False)
    if estimate_tensor_separately and not tensor_opts:
        raise Exception('Setting "estimate_tensor_separately": true requires options'
                        'for "estimate_tensor". Please update your pipeline config.' )

    # Do we have deltas?
    deltas = (params.get("big_delta", None), params.get("small_delta", None))
    approximate_deltas = None in deltas

    # TORTOISE requires unzipped float32 nifti files and a bmtxt file.
    tortoise_convert = pe.Node(TORTOISEConvert(), name="tortoise_convert")
    workflow.connect([
        (inputnode, tortoise_convert, [
            ('dwi_file', 'dwi_file'),
            ('bval_file', 'bval_file'),
            ('bvec_file', 'bvec_file'),
            ('dwi_mask', 'mask_file')]),
        (recon_scalars, outputnode, [("scalar_info", "recon_scalars")])])

    # EstimateTensor
    if tensor_opts:
        tensor_opts['num_threads'] = omp_nthreads
        estimate_tensor = pe.Node(
            EstimateTensor(**tensor_opts),
            name="estimate_tensor",
            n_procs=omp_nthreads)
        desc += "A diffusion tensor model was fit using ``EstimateTensor`` " \
            "with {} regularization. ".format(tensor_opts.get("reg_mode", "WLLS"))

        # Set up datasinks
        compute_dt_fa = pe.Node(ComputeFAMap(), name="compute_dt_fa")
        compute_dt_rd = pe.Node(ComputeRDMap(), name="compute_dt_rd")
        compute_dt_ad = pe.Node(ComputeADMap(), name="compute_dt_ad")
        compute_dt_li = pe.Node(ComputeLIMap(), name="compute_dt_li")
        workflow.connect([
            (tortoise_convert, estimate_tensor, [
                ("dwi_file", "in_file"),
                ("mask_file", "mask"),
                ("bmtxt_file", "bmtxt_file")]),
            (estimate_tensor, compute_dt_fa, [
                ("dt_file", "in_file"),
                ("am_file", "am_file")]),
            (estimate_tensor, compute_dt_rd, [
                ("dt_file", "in_file"),
                ("am_file", "am_file")]),
            (estimate_tensor, compute_dt_ad, [
                ("dt_file", "in_file"),
                ("am_file", "am_file")]),
            (estimate_tensor, compute_dt_li, [
                ("dt_file", "in_file"),
                ("am_file", "am_file")]),
            (estimate_tensor, recon_scalars, [("am_file", "am_file")]),
            (compute_dt_fa, recon_scalars, [("fa_file", "fa_file")]),
            (compute_dt_rd, recon_scalars, [("rd_file", "rd_file")]),
            (compute_dt_ad, recon_scalars, [("ad_file", "ad_file")]),
            (compute_dt_li, recon_scalars, [("li_file", "li_file")])
        ])

    # EstimateMAPMRI
    mapmri_opts = params.get("estimate_mapmri", {})
    if not mapmri_opts:
        return workflow

    # Set deltas if we have them. Prevent only one from being defined
    if approximate_deltas:
        LOGGER.warning('Both "big_delta" and "small_delta" are required for precise MAPMRI')
        big_delta = little_delta = traits.Undefined
    else:
        mapmri_opts["big_delta"], mapmri_opts["small_delta"] = deltas
    mapmri_opts["num_threads"] = omp_nthreads

    estimate_mapmri = pe.Node(
        EstimateMAPMRI(**mapmri_opts),
        name="estimate_mapmri",
        n_procs=omp_nthreads)

    compute_mapmri_pa = pe.Node(
        ComputeMAPMRI_PA(num_threads=1),
        name="compute_mapmri_pa",
        n_procs=1)

    compute_mapmri_rtop = pe.Node(
        ComputeMAPMRI_RTOP(num_threads=1),
        name="compute_mapmri_rtop",
        n_procs=1)

    compute_mapmri_ng = pe.Node(
        ComputeMAPMRI_NG(num_threads=1),
        name="compute_mapmri_ng",
        n_procs=1)

    if estimate_tensor_separately:
        workflow.connect([
            (estimate_tensor, estimate_mapmri, [
                ("dt_file", "dt_file"),
                ("am_file", "a0_file")])])

    workflow.connect([
        (tortoise_convert, estimate_mapmri,[
            ("bmtxt_file", "bmtxt_file"),
            ("dwi_file", "in_file"),
            ("mask_file", "mask")]),
        (estimate_mapmri, compute_mapmri_pa, [
            ("coeffs_file", "in_file"),
            ("uvec_file", "uvec_file")]),
        (compute_mapmri_pa, recon_scalars,[
            ("pa_file", "pa_file"),
            ("path_file", "path_file")]),
        (estimate_mapmri, compute_mapmri_rtop, [
            ("coeffs_file", "in_file"),
            ("uvec_file", "uvec_file")]),
        (compute_mapmri_rtop, recon_scalars,[
            ("rtop_file", "rtop_file"),
            ("rtap_file", "rtap_file"),
            ("rtpp_file", "rtpp_file")]),
        (estimate_mapmri, compute_mapmri_ng, [
            ("coeffs_file", "in_file"),
            ("uvec_file", "uvec_file")]),
        (compute_mapmri_ng, recon_scalars,[
            ("ng_file", "ng_file"),
            ("ngpar_file", "ngpar_file"),
            ("ngperp_file", "ngperp_file")]),
    ])
    if qsirecon_suffix:
        ds_recon_scalars = pe.Node(
            ReconScalarsDataSink(),
            name="ds_recon_scalars",
            run_without_submitting=True)
        workflow.connect(
            recon_scalars,
            "scalar_info",
            ds_recon_scalars,
            "recon_scalars")
    workflow.__desc__ = desc
    return workflow