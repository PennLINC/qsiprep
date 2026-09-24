# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Resampling workflows
++++++++++++++++++++

.. autofunction:: init_dwi_trans_wf

"""

from nipype.interfaces import ants
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from ... import config
from ...interfaces.ants import GetImageType
from ...interfaces.fmap import ApplyJacobianWeights
from ...interfaces.gradients import (  # LocalGradientRotation,
    ComposeSDCWarp,
    ComposeTransforms,
    ExtractB0s,
    GradientRotation,
)
from ...interfaces.images import ChooseInterpolator
from ...interfaces.jacobian import ComposeJacobianWeights
from ...interfaces.nilearn import Merge
from .qc import init_modelfree_qc_wf
from .util import init_dwi_reference_wf

DEFAULT_MEMORY_MIN_GB = 0.01


def init_dwi_trans_wf(
    source_file,
    mem_gb,
    template='ACPC',
    name='dwi_trans_wf',
    use_compression=True,
    write_local_bvecs=False,
    write_reports=True,
    concatenate=True,
    doing_topup=False,
    pe_axis=None,
    weight_fieldwarps=True,
    sdc_warp_source=None,
    sdc_pe_dir=None,
    sdc_readout_time=None,
):
    """
    This workflow samples dwi images to the ``output_grid`` in a "single shot"
    from the original DWI series.

    .. workflow::
        :graph2use: colored
        :simple_form: yes

        from qsiprep.workflows.dwi.resampling import init_dwi_trans_wf
        wf = init_dwi_trans_wf(source_file='sub-1_dwi.nii.gz',
                               template='MNI152NLin2009cAsym',
                               output_resolution=1.2,
                               mem_gb=3,
                               omp_nthreads=1)

    **Parameters**

        template : str
            Name of template targeted by ``template`` output space
        mem_gb : float
            Size of DWI file in GB
        omp_nthreads : int
            Maximum number of threads an individual process may use
        name : str
            Name of workflow (default: ``dwi_trans_wf``)
        use_compression : bool
            Save registered DWI series as ``.nii.gz``
        use_fieldwarp : bool
            Include SDC warp in single-shot transform from DWI to MNI
        output_resolution : float
            Voxel size in mm for the output data
        to_mni : bool
            Include warps to MNI
        write_local_bvecs : bool
            if true, local bvec niftis are written

    **Inputs**

        itk_b0_to_t1
            Affine transform from ``ref_bold_brain`` to T1 space (ITK format)
        t1_2_mni_forward_transform
            ANTs-compatible affine-and-warp transform file
        dwi_files
            Individual 3D volumes, not motion corrected
        cnr_map
            Contrast to noise map from model-based hmc
        fieldmap_hz
            Fieldmap in Hz. Only written out if TOPUP was used.
        bval_files
            individual bval files
        bvec_files
            one-lined bvec files
        b0_ref_image
            b0 template for the dwi series
        b0_indices
            List of indices that contain a b0 image
        dwi_mask
            Skull-stripping mask of reference image
        name_source
            DWI series NIfTI file
            Used to recover original information lost during processing
        hmc_xforms
            List of affine transforms aligning each volume to ``ref_image`` in ITK format
        fieldwarps
            a :abbr:`DFM (displacements field map)` in ITK format
        gradwarp_field
            a gradient nonlinearity displacement field in native DWI space, or
            undefined if no gradwarp correction is being applied
        output_grid
            File defining the output space
        t1_mask
            Brain mask from the t1w

    **Outputs**

        dwi_resampled
            DWI series, resampled to template space. One file if ``concatenate``, otherwise a
            list of files
        dwi_ref_resampled
            Reference, contrast-enhanced summary of the DWI series, resampled to template space
        dwi_mask_resampled
            DWI series mask in template space
        cnr_map_resampled
            Contrast to noise map resampled
        bvals
            bvals file for the DWI series
        rotated_bvecs
            bvecs rotated for transforms to ``output_grid``
        local_bvecs
            NIfTI file containing the bvec rotation matrix (due to transforms) in each voxel.
            Includes rotations introduced by warpingdenoisin

    """
    workflow = Workflow(name=name)
    output_resolution = config.workflow.output_resolution
    workflow.__desc__ = """\
The DWI time-series were resampled to {tpl},
generating a *preprocessed DWI run in {tpl} space* with {vox}mm isotropic voxels.
""".format(tpl=template, vox=str(output_resolution).rstrip('0').rstrip('.'))

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'itk_b0_to_t1',
                't1_mask',
                't1_brain',
                'b0_to_dwiref_transforms',
                'dwiref_to_t1_warp',
                'dwiref_to_t1_affine',
                't1_2_mni_forward_transform',
                'name_source',
                'dwi_files',
                'cnr_map',
                'bval_files',
                'bvec_files',
                'b0_ref_image',
                'b0_indices',
                'dwi_mask',
                'hmc_xforms',
                'fieldwarps',
                'gradwarp_field',
                'output_grid',
                'ec_jacobian_images',
                # Only set when DRBUDDI ran: TORTOISE's LSR ratios, which
                # replace the Jacobian weight.
                'sdc_scaling_images',
                # Only written out if TOPUP was used
                'fieldmap_hz',
            ]
        ),
        name='inputnode',
    )

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'dwi_resampled',
                'dwi_ref_resampled',
                'dwi_mask_resampled',
                'cnr_map_resampled',
                'bvals',
                'resampled_dwi_mask',
                'rotated_bvecs',
                'local_bvecs',
                'b0_series',
                'resampled_qc',
                # Only defined when Jacobian weighting applied (no --ignore jacobian)
                # weights: the unique output-grid weight maps and the
                # per-volume index into them.
                'jacobian_weights',
                'jacobian_weight_index',
                'jacobian_method',
                # Only written out if TOPUP was used
                'fieldmap_hz_resampled',
                # The SDC displacement field on the output grid
                'sdc_warp_to_template',
                # TOPUP+DRBUDDI only: DRBUDDI's refinement of the TOPUP field
                'sdc_refinement_to_template',
            ]
        ),
        name='outputnode',
    )

    # get composite warps and composed affines for warping and rotating
    compose_transforms = pe.Node(ComposeTransforms(), name='compose_transforms')
    get_interpolation = pe.Node(
        ChooseInterpolator(sloppy=config.execution.sloppy, output_resolution=output_resolution),
        name='get_interpolation',
    )
    dwi_transform = pe.MapNode(
        ants.ApplyTransforms(float=True),
        name='dwi_transform',
        iterfield=['input_image', 'transforms'],
    )
    # num_threads parallelizes the per-unique-map antsApplyTransports calls
    # inside the interface's own ThreadPoolExecutor -- see ApplyJacobianWeights'
    # docstring. n_procs must be paired with it (as every other in-node
    # fan-out in this repo does -- e.g. diffprep.py's synth_dwis, fsl.py's
    # gather_inputs, hmc.py's iter_reg, anatomical/volume.py's n4_correct) so
    # nipype's MultiProc scheduler reserves that many slots for this node
    # instead of co-scheduling other work against a node that is itself
    # running up to omp_nthreads concurrent antsApplyTransforms processes.
    scale_dwis = pe.Node(
        ApplyJacobianWeights(num_threads=config.nipype.omp_nthreads),
        name='scale_dwis',
        n_procs=config.nipype.omp_nthreads,
    )
    rotate_gradients = pe.Node(GradientRotation(), name='rotate_gradients')
    cnr_image_type = pe.Node(GetImageType(), name='cnr_image_type')
    cnr_tfm = pe.Node(
        ants.ApplyTransforms(interpolation='LanczosWindowedSinc', float=True),
        name='cnr_tfm',
        mem_gb=1,
    )

    workflow.connect([
        (inputnode, compose_transforms, [
            ('output_grid', 'reference_image'),
            ('dwi_files', 'dwi_files'),
            ('hmc_xforms', 'hmc_affines'),
            ('itk_b0_to_t1', 'hmcsdc_dwi_ref_to_t1w_affine'),
            ('fieldwarps', 'fieldwarps'),
            (('gradwarp_field', _listify), 'gradwarp'),
            ('b0_to_dwiref_transforms', 'b0_to_dwiref_transforms'),
            (('dwiref_to_t1_affine', _get_first),
             'dwiref_to_t1_affine'),
            ('dwiref_to_t1_warp', 'dwiref_to_t1_warp'),
        ]),
        # TODO: check that the cnr_tfm is also appropriately warped for shoreline
        (compose_transforms, cnr_tfm, [(('out_warps', _get_first), 'transforms')]),
        (inputnode, rotate_gradients, [
            ('bvec_files', 'bvec_files'),
            ('bval_files', 'bval_files'),
            ('dwi_files', 'original_images'),
        ]),
        (compose_transforms, rotate_gradients, [('out_affines', 'affine_transforms')]),
        (rotate_gradients, outputnode, [
            ('bvals', 'bvals'),
            ('bvecs', 'rotated_bvecs'),
        ]),
        (inputnode, cnr_image_type, [('cnr_map', 'image')]),
        (cnr_image_type, cnr_tfm, [('image_type', 'input_image_type')]),
        (inputnode, cnr_tfm, [
            ('cnr_map', 'input_image'),
            ('output_grid', 'reference_image'),
        ]),
        (cnr_tfm, outputnode, [('output_image', 'cnr_map_resampled')]),
        (compose_transforms, dwi_transform, [('out_warps', 'transforms')]),
        (inputnode, dwi_transform, [
            ('dwi_files', 'input_image'),
            ('output_grid', 'reference_image'),
        ]),
        (inputnode, get_interpolation, [('dwi_files', 'dwi_files')]),
        (get_interpolation, dwi_transform, [('interpolation_method', 'interpolation')]),
        (dwi_transform, scale_dwis, [('output_image', 'dwi_files')]),
        (inputnode, scale_dwis, [
            ('output_grid', 'reference_image'),
            ('itk_b0_to_t1', 'hmcsdc_dwi_ref_to_t1w_affine'),
            ('b0_to_dwiref_transforms', 'b0_to_dwiref_transforms'),
            (('dwiref_to_t1_affine', _get_first), 'dwiref_to_t1_affine'),
            ('dwiref_to_t1_warp', 'dwiref_to_t1_warp'),
        ]),
    ])  # fmt:skip

    # The weight covers gradwarp and SDC only. HMC is excluded by policy and is
    # coordinate-safe to exclude because it is the outermost transform in the
    # pull-back (see the design spec); coregistration and the dwiref and
    # template warps are excluded because modulating by a spatial-normalization
    # warp is VBM-style volume modulation, wrong for DWI signal.
    if 'jacobian' not in (config.workflow.ignore or []):
        # num_threads/n_procs paired as for scale_dwis above: the node shells
        # out to antsApplyTransforms and CreateJacobianDeterminantImage, which
        # run single-threaded on nipype's default.
        compose_jacobian = pe.Node(
            ComposeJacobianWeights(
                num_threads=config.nipype.omp_nthreads,
                pe_axis=pe_axis,
                weight_fieldwarps=weight_fieldwarps,
            ),
            name='compose_jacobian',
            n_procs=config.nipype.omp_nthreads,
        )
        workflow.connect([
            (inputnode, compose_jacobian, [
                ('dwi_files', 'dwi_files'),
                ('b0_ref_image', 'b0_ref_image'),
                ('dwi_mask', 'mask'),
                (('gradwarp_field', _listify), 'gradwarp_field'),
                ('fieldwarps', 'fieldwarps'),
                ('ec_jacobian_images', 'ec_jacobian_images'),
                ('sdc_scaling_images', 'sdc_scaling_images'),
            ]),
            (compose_jacobian, scale_dwis, [
                ('jacobian_weight_images', 'jacobian_weight_images'),
            ]),
            (compose_jacobian, outputnode, [('method', 'jacobian_method')]),
            (scale_dwis, outputnode, [
                ('resampled_weight_images', 'jacobian_weights'),
                ('weight_index', 'jacobian_weight_index'),
            ]),
        ])  # fmt:skip

    if doing_topup:
        fieldmap_hz_tfm = pe.Node(
            ants.ApplyTransforms(interpolation='NearestNeighbor', float=True),
            name='fieldmap_hz_tfm',
            mem_gb=1,
        )
        workflow.connect([
            (inputnode, fieldmap_hz_tfm, [
                ('fieldmap_hz', 'input_image'),
                ('output_grid', 'reference_image'),
            ]),
            (compose_transforms, fieldmap_hz_tfm, [(('out_warps', _get_first), 'transforms')]),
            (fieldmap_hz_tfm, outputnode, [('output_image', 'fieldmap_hz_resampled')]),
        ])  # fmt:skip

    if sdc_warp_source is not None:
        # Re-express the SDC (susceptibility) displacement field on the output
        # grid as a transform, so its vectors are rotated into ACPC world
        # coordinates. It rides only the stages that carry the corrected DWI
        # frame to the output grid (compose_transforms.sdc_warp_transforms) --
        # see ComposeSDCWarp.
        compose_sdc_warp = pe.Node(ComposeSDCWarp(), name='compose_sdc_warp', mem_gb=1)

        def _first_sdc_warp_node():
            # Volume 0's susceptibility warp, as a node rather than an inline
            # connection function: the DIFFPREP T2Wreg path already reaches
            # ``fieldwarps`` through one (``_as_transform_list`` in diffprep.py),
            # and nipype refuses two inline functions in series across an
            # IdentityInterface. Built only on the branches that read a
            # standalone warp; on the TOPUP branch ``fieldwarps`` is empty.
            node = pe.Node(
                niu.Function(function=_first_warp, output_names=['out']),
                name='first_sdc_warp',
                run_without_submitting=True,
            )
            workflow.connect([(inputnode, node, [('fieldwarps', 'fieldwarps')])])
            return node

        workflow.connect([
            (inputnode, compose_sdc_warp, [('output_grid', 'reference_image')]),
            (compose_transforms, compose_sdc_warp, [
                ('sdc_warp_transforms', 'to_template_transforms'),
            ]),
            (compose_sdc_warp, outputnode, [('sdc_warp_to_template', 'sdc_warp_to_template')]),
        ])  # fmt:skip

        if sdc_warp_source == 'fieldwarp':
            # DRBUDDI, GRE, SyN and T2Wreg all write the susceptibility warp
            # directly (fieldwarps); conjugate volume 0's onto the output grid.
            first_sdc_warp = _first_sdc_warp_node()
            workflow.connect([
                (first_sdc_warp, compose_sdc_warp, [('out', 'sdc_warps')]),
            ])  # fmt:skip
        else:
            # TOPUP only estimates an off-resonance field (eddy applies it and
            # leaves no standalone warp -- its fieldwarps carry eddy's *combined*
            # motion/eddy-current/SDC correction, not a pure susceptibility warp),
            # so the displacement field is rebuilt from the field.
            hz_to_warp = pe.Node(
                niu.Function(function=_hz_to_warp, output_names=['out_file']),
                name='hz_to_warp',
            )
            hz_to_warp.inputs.readout_time = sdc_readout_time
            hz_to_warp.inputs.pe_dir = sdc_pe_dir
            workflow.connect([(inputnode, hz_to_warp, [('fieldmap_hz', 'in_file')])])

            if sdc_warp_source == 'topup':
                workflow.connect([(hz_to_warp, compose_sdc_warp, [('out_file', 'sdc_warps')])])
            else:
                # TOPUP+DRBUDDI: DRBUDDI refined the series eddy had already
                # corrected with TOPUP's field, so its fieldwarp is only the
                # residual. The total field runs a corrected point through
                # DRBUDDI's refinement, then TOPUP's field; the refinement is also
                # conjugated on its own, to show where DRBUDDI changed TOPUP's answer.
                sdc_warp_chain = pe.Node(niu.Merge(2), name='sdc_warp_chain')
                compose_sdc_refinement = pe.Node(
                    ComposeSDCWarp(), name='compose_sdc_refinement', mem_gb=1
                )
                first_sdc_warp = _first_sdc_warp_node()
                workflow.connect([
                    (first_sdc_warp, sdc_warp_chain, [('out', 'in1')]),
                    (hz_to_warp, sdc_warp_chain, [('out_file', 'in2')]),
                    (sdc_warp_chain, compose_sdc_warp, [('out', 'sdc_warps')]),
                    (inputnode, compose_sdc_refinement, [('output_grid', 'reference_image')]),
                    (first_sdc_warp, compose_sdc_refinement, [('out', 'sdc_warps')]),
                    (compose_transforms, compose_sdc_refinement, [
                        ('sdc_warp_transforms', 'to_template_transforms'),
                    ]),
                    (compose_sdc_refinement, outputnode, [
                        ('sdc_warp_to_template', 'sdc_refinement_to_template'),
                    ]),
                ])  # fmt:skip

    # If concatenation is not happening here, send the still-split images to outputs
    if not concatenate:
        workflow.connect([(scale_dwis, outputnode, [('scaled_images', 'dwi_resampled')])])
        return workflow

    merge = pe.Node(Merge(compress=use_compression), name='merge', mem_gb=mem_gb * 3)
    extract_b0_series = pe.Node(ExtractB0s(), name='extract_b0_series')
    final_b0_ref = init_dwi_reference_wf(
        gen_report=write_reports,
        desc='resampled',
        name='final_b0_ref',
        source_file=source_file,
    )

    workflow.connect([
        (inputnode, merge, [('name_source', 'header_source')]),
        (scale_dwis, merge, [('scaled_images', 'in_files')]),
        (merge, outputnode, [('out_file', 'dwi_resampled')]),
        (merge, extract_b0_series, [('out_file', 'dwi_series')]),
        (inputnode, extract_b0_series, [('b0_indices', 'b0_indices')]),
        (extract_b0_series, final_b0_ref, [('b0_average', 'inputnode.b0_template')]),
        (inputnode, final_b0_ref, [('t1_mask', 'inputnode.t1_mask')]),
        (final_b0_ref, outputnode, [
            ('outputnode.ref_image', 'dwi_ref_resampled'),
            ('outputnode.dwi_mask', 'resampled_dwi_mask'),
        ]),
    ])  # fmt:skip

    # Calculate QC metrics on the resampled data
    calculate_qc = init_modelfree_qc_wf(
        bvec_convention='DIPY',  # Resampled is always LPS+
        name='calculate_qc',
    )
    workflow.connect([
        (rotate_gradients, calculate_qc, [
            ('bvals', 'inputnode.bval_file'),
            ('bvecs', 'inputnode.bvec_file'),
        ]),
        (merge, calculate_qc, [('out_file', 'inputnode.dwi_file')]),
        (calculate_qc, outputnode, [('outputnode.qc_summary', 'resampled_qc')]),
    ])  # fmt:skip
    # if write_local_bvecs:
    #     local_grad_rotation = pe.Node(LocalGradientRotation(), name="local_grad_rotation")
    #     workflow.connect([
    #         (compose_transforms, local_grad_rotation, [('out_warps', 'warp_transforms')]),
    #         (inputnode, local_grad_rotation, [('bvec_files', 'bvec_files')]),
    #         (local_grad_rotation, outputnode, [('local_bvecs', 'local_bvecs')])
    #     ])  # fmt:skip

    return workflow


def _hz_to_warp(in_file, readout_time, pe_dir, newpath=None):
    """TOPUP off-resonance field (Hz) -> ITK displacement field along the PE axis.

    TOPUP shifts each voxel by ``field_Hz * TotalReadoutTime`` voxels along its
    acquisition-parameter vector, which qsiprep writes from the raw BIDS
    ``PhaseEncodingDirection`` in the voxel axes of the grid TOPUP ran on (LAS+,
    not the input's own orientation). The shift is therefore taken along that
    voxel axis of this image and carried to world space by its affine, so the
    vector is right on any grid; ``FUGUEvsm2ANTSwarp`` instead hard-codes
    +i=R, +j=A, +k=I, which is wrong for the i and k axes of an LAS+ grid.
    """
    import os

    import nibabel as nb
    import numpy as np
    from nipype.utils.filemanip import fname_presuffix

    img = nb.load(in_file)
    axis = 'ijk'.index(pe_dir[0])
    sign = -1.0 if pe_dir.endswith('-') else 1.0
    shift = np.asanyarray(img.dataobj, dtype='float32') * float(readout_time) * sign
    # One voxel step along the PE axis in world mm, RAS -> ITK's LPS.
    step_lps = img.affine[:3, axis] * np.array([-1.0, -1.0, 1.0])
    field = (shift[..., np.newaxis] * step_lps)[:, :, :, np.newaxis, :].astype('float32')

    out = nb.Nifti1Image(field, img.affine)
    out.header.set_intent('vector')
    out_file = fname_presuffix(in_file, suffix='_warp', newpath=newpath or os.getcwd())
    out.to_filename(out_file)
    return out_file


def _first_warp(fieldwarps):
    """Volume 0's SDC warp: GRE hands over a single path, the others a list."""
    return fieldwarps if isinstance(fieldwarps, str) else fieldwarps[0]


def _first(inlist):
    return inlist[0]


def _get_first(lll):
    from nipype.interfaces.base import isdefined

    if isdefined(lll):
        return lll[0]
    return lll


def _listify(value):
    """Wrap a single gradwarp field in a one-element list for ``ComposeTransforms``.

    ``ComposeTransforms.gradwarp`` silently drops a list whose length matches
    neither 1 nor the DWI count -- unlike ``fieldwarps``, it does not warn on a
    mismatch. Asserting single-element-ness here means a future mis-wire that
    feeds this something other than one field fails loudly instead of vanishing
    into ``ComposeTransforms``.

    ``Undefined`` (an unconnected ``gradwarp_field``, e.g. no gradwarp plan, or
    a DIS3D plan that intentionally is not wired into resampling) passes
    through unchanged rather than becoming a spurious single-item list.
    """
    from nipype.interfaces.base import isdefined

    if not isdefined(value):
        return value
    assert not isinstance(value, list), (
        f'_listify expects a single gradwarp field, got a list: {value!r}'
    )
    return [value]
