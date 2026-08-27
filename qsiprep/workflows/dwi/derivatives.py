"""
Writing outputs from a dwi preproc workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_dwi_derivatives_wf

"""

from nipype import logging
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from ... import config
from ...interfaces import DerivativesDataSink
from ...interfaces.tsnr import DWITSNR

DEFAULT_MEMORY_MIN_GB = 0.01

# The BIDS ``model`` entity on the CNR derivative names the *signal model* the
# CNR was computed from, not the HMC backend -- for the older backends the two
# strings simply coincide (3dSHORE, tensor, eddy). The ``tortoise`` backend
# breaks that: DIFFPREP itself emits no CNR, so qsiprep derives one from the
# MAPMRI fit it already runs for slice QC, and the entity must name MAPMRI.
_CNR_MODEL_LABELS = {'tortoise': 'MAPMRI'}


def _cnr_model_label(hmc_model):
    """BIDS-safe ``model`` entity naming the signal model behind the CNR map."""
    return _CNR_MODEL_LABELS.get(hmc_model, hmc_model)


def _cnr_description(hmc_model):
    """Sidecar description for the CNR map, flagging DIFFPREP's in-sample fit."""
    desc = 'Contrast-to-noise ratio map for the HMC step.'
    if hmc_model == 'tortoise':
        desc += (
            ' DIFFPREP does not emit a CNR map, so this was computed from the '
            'MAPMRI model qsiprep fits to the corrected data for slice-wise QC. '
            'Unlike the SHORELine CNR -- whose predictions exclude q-space '
            'neighbours of the volume being predicted -- these predictions are '
            'in-sample (the model is fit to all volumes and synthesized at the '
            'measured gradients), so values are optimistically biased and are '
            'not quantitatively comparable to the 3dSHORE or eddy CNR. The '
            'spatial pattern remains informative.'
        )
    return desc


def _tsnr_meta(n_b0, median_tsnr):
    """Sidecar metadata for the TSNR map."""
    return {
        'Description': (
            'Temporal SNR (mean/SD across the b=0 volumes) of the final '
            'resampled series. With few b=0 volumes the estimate is noisy.'
        ),
        'NumberOfB0Volumes': n_b0,
        'MedianTSNR': median_tsnr,
    }


LOGGER = logging.getLogger('nipype.workflow')


def init_dwi_derivatives_wf(
    source_file, resolution=None, write_hmc_optimization=True, name='dwi_derivatives_wf'
) -> Workflow:
    """Set up a battery of datasinks to store derivatives in the right location.

    QSIRecon's primary input is the preprocessed ACPC-space DWI this workflow
    writes, so the single-argument call form (``resolution=None``) must keep
    producing exactly the filenames it always has: no ``res-`` entity.

    Parameters
    ----------
    resolution : Resolution or None
        Set only when more than one ACPC resolution was requested. Adds a
        ``res-<label>`` entity to every ACPC DWI sink below, and (since the
        filename alone does not say what ``res-native*`` resolved to) expects
        ``inputnode.resolution_meta`` to be wired with the resolved voxel size.
    write_hmc_optimization : bool
        The hmcOptimization sidecar is produced before resampling and does not vary
        by output resolution. When ``init_dwi_derivatives_wf`` is instantiated once
        per ACPC resolution, every instance would otherwise write the exact same
        path -- a same-path collision under nipype's MultiProc plugin. Callers doing
        that fan-out should pass this as ``True`` for exactly one instance (its first
        spec) and ``False`` for the rest.
    """
    output_dir = str(config.execution.output_dir)
    workflow = Workflow(name=name)
    res_entities = {'res': resolution.label} if resolution is not None else {}
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'source_file',
                'dwi_t1',
                'dwi_mask_t1',
                'cnr_map_t1',
                'bvals_t1',
                'bvecs_t1',
                'local_bvecs_t1',
                't1_b0_ref',
                'gradient_table_t1',
                'btable_t1',
                'hmc_optimization_data',
                'series_qc',
                'resolution_meta',
            ]
        ),
        name='inputnode',
    )

    if (
        write_hmc_optimization
        and config.workflow.hmc_model == '3dSHORE'
        and config.workflow.shoreline_iters > 1
    ):
        ds_optimization = pe.Node(
            DerivativesDataSink(
                source_file=source_file,
                base_directory=output_dir,
                suffix='hmcOptimization',
            ),
            name='ds_optimization',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB,
        )
        workflow.connect([(inputnode, ds_optimization, [('hmc_optimization_data', 'in_file')])])

    # Temporal SNR over the b=0 volumes, computed on the final resampled series
    # so it reflects what the user actually gets. See interfaces/tsnr.py for why
    # b=0 only.
    tsnr = pe.Node(DWITSNR(), name='tsnr', mem_gb=DEFAULT_MEMORY_MIN_GB)
    tsnr_meta = pe.Node(
        niu.Function(
            input_names=['n_b0', 'median_tsnr'],
            output_names=['meta_dict'],
            function=_tsnr_meta,
        ),
        name='tsnr_meta',
        run_without_submitting=True,
    )
    ds_tsnr = pe.Node(
        DerivativesDataSink(
            source_file=source_file,
            base_directory=output_dir,
            space='ACPC',
            statistic='tsnr',
            suffix='dwimap',
            extension='.nii.gz',
            compress=True,
            **res_entities,
        ),
        name='ds_tsnr',
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    # 4D DWI in ACPC space

    ds_dwi_t1 = pe.Node(
        DerivativesDataSink(
            source_file=source_file,
            base_directory=output_dir,
            space='ACPC',
            desc='preproc',
            suffix='dwi',
            extension='.nii.gz',
            compress=True,
            **res_entities,
        ),
        name='ds_dwi_t1',
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )
    ds_bvals_t1 = pe.Node(
        DerivativesDataSink(
            source_file=source_file,
            base_directory=output_dir,
            space='ACPC',
            suffix='dwi',
            extension='.bval',
            desc='preproc',
            **res_entities,
        ),
        name='ds_bvals_t1',
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )
    ds_bvecs_t1 = pe.Node(
        DerivativesDataSink(
            source_file=source_file,
            base_directory=output_dir,
            space='ACPC',
            suffix='dwi',
            extension='.bvec',
            desc='preproc',
            **res_entities,
        ),
        name='ds_bvecs_t1',
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )
    ds_t1_b0_ref = pe.Node(
        DerivativesDataSink(
            source_file=source_file,
            base_directory=output_dir,
            space='ACPC',
            suffix='dwiref',
            extension='.nii.gz',
            compress=True,
            **res_entities,
        ),
        name='ds_t1_b0_ref',
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )
    ds_dwi_mask_t1 = pe.Node(
        DerivativesDataSink(
            source_file=source_file,
            base_directory=output_dir,
            space='ACPC',
            desc='brain',
            suffix='mask',
            extension='.nii.gz',
            compress=True,
            **res_entities,
        ),
        name='ds_dwi_mask_t1',
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )
    ds_cnr_map_t1 = pe.Node(
        DerivativesDataSink(
            source_file=source_file,
            base_directory=output_dir,
            space='ACPC',
            model=_cnr_model_label(config.workflow.hmc_model),
            statistic='cnr',
            suffix='dwimap',
            extension='.nii.gz',
            compress=True,
            meta_dict={
                'Description': _cnr_description(config.workflow.hmc_model),
            },
            **res_entities,
        ),
        name='ds_cnr_map_t1',
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )
    ds_gradient_table_t1 = pe.Node(
        DerivativesDataSink(
            source_file=source_file,
            base_directory=output_dir,
            space='ACPC',
            desc='preproc',
            suffix='dwi',
            extension='.b',
            **res_entities,
        ),
        name='ds_gradient_table_t1',
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )
    ds_btable_t1 = pe.Node(
        DerivativesDataSink(
            source_file=source_file,
            base_directory=output_dir,
            space='ACPC',
            desc='preproc',
            suffix='dwi',
            extension='.b_table.txt',
            **res_entities,
        ),
        name='ds_btable_t1',
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    workflow.connect([
        (inputnode, tsnr, [
            ('dwi_t1', 'dwi_file'),
            ('bvals_t1', 'bval_file'),
            ('dwi_mask_t1', 'mask_file'),
        ]),
        (tsnr, tsnr_meta, [
            ('n_b0', 'n_b0'),
            ('median_tsnr', 'median_tsnr'),
        ]),
        (tsnr, ds_tsnr, [('out_file', 'in_file')]),
        (tsnr_meta, ds_tsnr, [('meta_dict', 'meta_dict')]),
        (inputnode, ds_dwi_t1, [('dwi_t1', 'in_file')]),
        (inputnode, ds_bvals_t1, [('bvals_t1', 'in_file')]),
        (inputnode, ds_bvecs_t1, [('bvecs_t1', 'in_file')]),
        (inputnode, ds_t1_b0_ref, [('t1_b0_ref', 'in_file')]),
        (inputnode, ds_dwi_mask_t1, [('dwi_mask_t1', 'in_file')]),
        (inputnode, ds_cnr_map_t1, [('cnr_map_t1', 'in_file')]),
        (inputnode, ds_gradient_table_t1, [('gradient_table_t1', 'in_file')]),
        (inputnode, ds_btable_t1, [('btable_t1', 'in_file')]),
    ])  # fmt:skip

    if resolution is not None:
        # The filename alone doesn't say what res-native* resolved to -- only
        # the sidecar does. ds_cnr_map_t1 and ds_tsnr keep their own descriptive
        # meta_dict as-is; the rest get the resolved voxel size here.
        workflow.connect([
            (inputnode, ds_dwi_t1, [('resolution_meta', 'meta_dict')]),
            (inputnode, ds_bvals_t1, [('resolution_meta', 'meta_dict')]),
            (inputnode, ds_bvecs_t1, [('resolution_meta', 'meta_dict')]),
            (inputnode, ds_t1_b0_ref, [('resolution_meta', 'meta_dict')]),
            (inputnode, ds_dwi_mask_t1, [('resolution_meta', 'meta_dict')]),
            (inputnode, ds_gradient_table_t1, [('resolution_meta', 'meta_dict')]),
            (inputnode, ds_btable_t1, [('resolution_meta', 'meta_dict')]),
        ])  # fmt:skip
    # If requested, write local bvecs
    # if config.workflow.write_local_bvecs:
    #     ds_local_bvecs_t1 = pe.Node(
    #         DerivativesDataSink(
    #             base_directory=output_dir,
    #             source_file=source_file,
    #             space="ACPC",
    #             suffix="bvec",
    #             compress=True,
    #         ),
    #         name="ds_local_bvecs_t1",
    #         run_without_submitting=True,
    #         mem_gb=DEFAULT_MEMORY_MIN_GB,
    #     )
    #     workflow.connect([(inputnode, ds_local_bvecs_t1, [('local_bvecs_t1', 'in_file')])])
    return workflow
