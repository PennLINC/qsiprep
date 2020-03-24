"""
Dipy Reconstruction workflows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_dipy_brainsuite_shore_recon_wf

"""
import logging
import nipype.pipeline.engine as pe
from nipype.interfaces import afni, utility as niu
from qsiprep.interfaces.bids import ReconDerivativesDataSink
from ...interfaces.dipy import BrainSuiteShoreReconstruction, MAPMRIReconstruction
from .interchange import input_fields
from ...engine import Workflow
from ...interfaces.reports import ReconPeaksReport

LOGGER = logging.getLogger('nipype.interface')


def external_format_datasinks(output_suffix, params, wf):
    """Add datasinks for Dipy Reconstructions in other formats."""
    outputnode = wf.get_node("outputnode")
    if params["write_fibgz"]:
        ds_fibgz = pe.Node(
            ReconDerivativesDataSink(extension='.fib.gz',
                                     suffix=output_suffix,
                                     compress=True),
            name='ds_{}_fibgz'.format(output_suffix),
            run_without_submitting=True)
        wf.connect(outputnode, 'fibgz', ds_fibgz, 'in_file')
    if params["write_mif"]:
        ds_mif = pe.Node(
            ReconDerivativesDataSink(extension='.mif',
                                     suffix=output_suffix,
                                     compress=False),
            name='ds_{}_mif'.format(output_suffix),
            run_without_submitting=True)
        wf.connect(outputnode, 'fod_sh_mif', ds_mif, 'in_file')


def init_dipy_brainsuite_shore_recon_wf(omp_nthreads, has_transform, name="dipy_3dshore_recon",
                                        output_suffix="", params={}):
    """Reconstruct EAPs, ODFs, using 3dSHORE (brainsuite-style basis set).

    Inputs

        *qsiprep outputs*

    Outputs

        shore_coeffs
            3dSHORE coefficients
        rtop
            Voxelwise Return-to-origin probability.
        rtap
            Voxelwise Return-to-axis probability.
        rtpp
            Voxelwise Return-to-plane probability.


    Params

        write_fibgz: bool
            True writes out a DSI Studio fib file
        write_mif: bool
            True writes out a MRTrix mif file with sh coefficients
        convert_to_multishell: str
            either "HCP", "ABCD", "lifespan" will resample the data with this scheme
        radial_order: int
            Radial order for spherical harmonics (even)
        zeta: float
            Zeta parameter for basis set.
        tau:float
            Diffusion parameter (default= 4 * np.pi**2)
        regularization
            "L2" or "L1". Default is "L2"
        lambdaN
            LambdaN parameter for L2 regularization. (default=1e-8)
        lambdaL
            LambdaL parameter for L2 regularization. (default=1e-8)
        regularization_weighting: int or "CV"
            L1 regualrization weighting. Default "CV" (use cross-validation).
            Can specify a static value to use in all voxels.
        l1_positive_constraint: bool
            Use positivity constraint.
        l1_maxiter
            Maximum number of iterations for L1 optization. (Default=1000)
        l1_alpha
            Alpha parameter for L1 optimization. (default=1.0)
        pos_grid: int
            Grid points for estimating EAP(default=11)
        pos_radius
            Radius for EAP estimation (default=20e-03)

    """

    inputnode = pe.Node(niu.IdentityInterface(fields=input_fields + ['odf_rois']),
                        name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['shore_coeffs_image', 'rtop_image', 'alpha_image', 'r2_image',
                    'cnr_image', 'regularization_image', 'fibgz', 'fod_sh_mif',
                    'dwi_file', 'bval_file', 'bvec_file', 'b_file']),
        name="outputnode")

    workflow = Workflow(name=name)
    desc = """Dipy Reconstruction

: """
    resample_mask = pe.Node(
        afni.Resample(outputtype='NIFTI_GZ', resample_mode="NN"), name='resample_mask')
    recon_shore = pe.Node(BrainSuiteShoreReconstruction(**params), name="recon_shore")
    doing_extrapolation = params.get("extrapolate_scheme") in ("HCP", "ABCD")

    plot_peaks = pe.Node(ReconPeaksReport(), name='plot_peaks')
    ds_report_peaks = pe.Node(
        ReconDerivativesDataSink(extension='.png',
                                 desc="3dSHOREODF",
                                 suffix='peaks'),
        name='ds_report_peaks',
        run_without_submitting=True)

    workflow.connect([
        (inputnode, recon_shore, [('dwi_file', 'dwi_file'),
                                  ('bval_file', 'bval_file'),
                                  ('bvec_file', 'bvec_file')]),
        (inputnode, resample_mask, [('t1_brain_mask', 'in_file'),
                                    ('dwi_file', 'master')]),
        (resample_mask, recon_shore, [('out_file', 'mask_file')]),
        (recon_shore, outputnode, [('shore_coeffs_image', 'shore_coeffs_image'),
                                   ('rtop_image', 'rtop_image'),
                                   ('alpha_image', 'alpha_image'),
                                   ('r2_image', 'r2_image'),
                                   ('cnr_image', 'cnr_image'),
                                   ('regularization_image', 'regularization_image'),
                                   ('fibgz', 'fibgz'),
                                   ('fod_sh_mif', 'fod_sh_mif'),
                                   ('extrapolated_dwi', 'dwi_file'),
                                   ('extrapolated_bvals', 'bval_file'),
                                   ('extrapolated_bvecs', 'bvec_file'),
                                   ('extrapolated_b', 'b_file')]),
        (inputnode, plot_peaks, [('dwi_ref', 'background_image'),
                                 ('odf_rois', 'odf_rois')]),
        (resample_mask, plot_peaks, [('out_file', 'mask_file')]),
        (recon_shore, plot_peaks, [('odf_directions', 'directions_file'),
                                   ('odf_amplitudes', 'odf_file')]),
        (plot_peaks, ds_report_peaks, [('out_report', 'in_file')])])

    # Plot targeted regions
    if has_transform:
        ds_report_odfs = pe.Node(
            ReconDerivativesDataSink(extension='.png',
                                     desc="3dSHOREODF",
                                     suffix='odfs'),
            name='ds_report_odfs',
            run_without_submitting=True)
        workflow.connect(plot_peaks, 'odf_report', ds_report_odfs, 'in_file')

    if output_suffix:
        external_format_datasinks(output_suffix, params, workflow)

        ds_rtop = pe.Node(
            ReconDerivativesDataSink(extension='.nii.gz',
                                     desc="rtop",
                                     suffix=output_suffix,
                                     compress=True),
            name='ds_bsshore_rtop',
            run_without_submitting=True)
        workflow.connect(outputnode, 'rtop_image', ds_rtop, 'in_file')

        ds_coeff = pe.Node(
            ReconDerivativesDataSink(extension='.nii.gz',
                                     desc="SHOREcoeff",
                                     suffix=output_suffix,
                                     compress=True),
            name='ds_bsshore_coeff',
            run_without_submitting=True)
        workflow.connect(outputnode, 'shore_coeffs_image', ds_coeff, 'in_file')

        ds_alpha = pe.Node(
            ReconDerivativesDataSink(extension='.nii.gz',
                                     desc="L1alpha",
                                     suffix=output_suffix,
                                     compress=True),
            name='ds_bsshore_alpha',
            run_without_submitting=True)
        workflow.connect(outputnode, 'alpha_image', ds_alpha, 'in_file')

        ds_r2 = pe.Node(
            ReconDerivativesDataSink(extension='.nii.gz',
                                     desc="r2",
                                     suffix=output_suffix,
                                     compress=True),
            name='ds_bsshore_r2',
            run_without_submitting=True)
        workflow.connect(outputnode, 'r2_image', ds_r2, 'in_file')

        ds_cnr = pe.Node(
            ReconDerivativesDataSink(extension='.nii.gz',
                                     desc="CNR",
                                     suffix=output_suffix,
                                     compress=True),
            name='ds_bsshore_cnr',
            run_without_submitting=True)
        workflow.connect(outputnode, 'cnr_image', ds_cnr, 'in_file')

        ds_regl = pe.Node(
            ReconDerivativesDataSink(extension='.nii.gz',
                                     desc="regularization",
                                     suffix=output_suffix,
                                     compress=True),
            name='ds_bsshore_regl',
            run_without_submitting=True)
        workflow.connect(outputnode, 'regularization_image', ds_regl, 'in_file')
        if doing_extrapolation:
            ds_extrap_dwi = pe.Node(
                ReconDerivativesDataSink(extension='.nii.gz',
                                         desc="extrapolated",
                                         suffix=output_suffix,
                                         compress=True),
                name='ds_extrap_dwi',
                run_without_submitting=True)
            workflow.connect(outputnode, 'dwi_file', ds_extrap_dwi, 'in_file')
            ds_extrap_bval = pe.Node(
                ReconDerivativesDataSink(extension='.bval',
                                         desc="extrapolated",
                                         suffix=output_suffix),
                name='ds_extrap_bval',
                run_without_submitting=True)
            workflow.connect(outputnode, 'bval_file', ds_extrap_bval, 'in_file')
            ds_extrap_bvec = pe.Node(
                ReconDerivativesDataSink(extension='.bvec',
                                         desc="extrapolated",
                                         suffix=output_suffix),
                name='ds_extrap_bvec',
                run_without_submitting=True)
            workflow.connect(outputnode, 'bvec_file', ds_extrap_bvec, 'in_file')
            ds_extrap_b = pe.Node(
                ReconDerivativesDataSink(extension='.b',
                                         desc="extrapolated",
                                         suffix=output_suffix),
                name='ds_extrap_b',
                run_without_submitting=True)
            workflow.connect(outputnode, 'b_file', ds_extrap_b, 'in_file')
    workflow.__desc__ = desc
    return workflow


def init_dipy_mapmri_recon_wf(omp_nthreads, has_transform, name="dipy_mapmri_recon",
                              output_suffix="", params={}):
    """Reconstruct EAPs, ODFs, using 3dSHORE (brainsuite-style basis set).

    Inputs

        *qsiprep outputs*

    Outputs

        shore_coeffs
            3dSHORE coefficients
        rtop
            Voxelwise Return-to-origin probability.
        rtap
            Voxelwise Return-to-axis probability.
        rtpp
            Voxelwise Return-to-plane probability.
        msd
            Voxelwise MSD
        qiv
            q-space inverse variance
        lapnorm
            Voxelwise norm of the Laplacian

    Params

        write_fibgz: bool
            True writes out a DSI Studio fib file
        write_mif: bool
            True writes out a MRTrix mif file with sh coefficients
        radial_order: int
            An even integer that represent the order of the basis
        laplacian_regularization: bool
            Regularize using the Laplacian of the MAP-MRI basis.
        laplacian_weighting: str or scalar
            The string 'GCV' makes it use generalized cross-validation to find
            the regularization weight. A scalar sets the regularization
            weight to that value and an array will make it selected the
            optimal weight from the values in the array.
        positivity_constraint: bool
            Constrain the propagator to be positive.
        pos_grid: int
            Grid points for estimating EAP(default=15)
        pos_radius
            Radius for EAP estimation (default=20e-03) or "adaptive"
        anisotropic_scaling : bool,
            If True, uses the standard anisotropic MAP-MRI basis. If False,
            uses the isotropic MAP-MRI basis (equal to 3D-SHORE).
        eigenvalue_threshold : float,
            Sets the minimum of the tensor eigenvalues in order to avoid
            stability problem.
        bval_threshold : float,
            Sets the b-value threshold to be used in the scale factor
            estimation. In order for the estimated non-Gaussianity to have
            meaning this value should set to a lower value (b<2000 s/mm^2)
            such that the scale factors are estimated on signal points that
            reasonably represent the spins at Gaussian diffusion.
        dti_scale_estimation : bool,
            Whether or not DTI fitting is used to estimate the isotropic scale
            factor for isotropic MAP-MRI.
            When set to False the algorithm presets the isotropic tissue
            diffusivity to static_diffusivity. This vastly increases fitting
            speed but at the cost of slightly reduced fitting quality. Can
            still be used in combination with regularization and constraints.
        static_diffusivity : float,
            the tissue diffusivity that is used when dti_scale_estimation is
            set to False. The default is that of typical white matter
            D=0.7e-3 _[5].
        cvxpy_solver : str, optional
            cvxpy solver name. Optionally optimize the positivity constraint
            with a particular cvxpy solver. See http://www.cvxpy.org/ for
            details.
            Default: None (cvxpy chooses its own solver)
    """

    inputnode = pe.Node(niu.IdentityInterface(fields=input_fields + ['odf_rois']),
                        name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['mapmri_coeffs', 'rtop', 'rtap', 'rtpp', 'fibgz', 'fod_sh_mif',
                    'parng', 'perng', 'ng', 'qiv', 'lapnorm', 'msd']),
        name="outputnode")

    workflow = Workflow(name=name)
    desc = """Dipy Reconstruction

: """
    recon_map = pe.Node(MAPMRIReconstruction(**params), name="recon_map")
    resample_mask = pe.Node(
        afni.Resample(outputtype='NIFTI_GZ', resample_mode="NN"), name='resample_mask')
    plot_peaks = pe.Node(ReconPeaksReport(), name='plot_peaks')
    ds_report_peaks = pe.Node(
        ReconDerivativesDataSink(extension='.png',
                                 desc="MAPLMRIODF",
                                 suffix='peaks'),
        name='ds_report_peaks',
        run_without_submitting=True)

    workflow.connect([
        (inputnode, recon_map, [('dwi_file', 'dwi_file'),
                                ('bval_file', 'bval_file'),
                                ('bvec_file', 'bvec_file')]),
        (inputnode, resample_mask, [('t1_brain_mask', 'in_file'),
                                    ('dwi_file', 'master')]),
        (resample_mask, recon_map, [('out_file', 'mask_file')]),
        (recon_map, outputnode, [('mapmri_coeffs', 'mapmri_coeffs'),
                                 ('rtop', 'rtop'),
                                 ('rtap', 'rtap'),
                                 ('rtpp', 'rtpp'),
                                 ('parng', 'parng'),
                                 ('perng', 'perng'),
                                 ('msd', 'msd'),
                                 ('ng', 'ng'),
                                 ('qiv', 'qiv'),
                                 ('lapnorm', 'lapnorm'),
                                 ('fibgz', 'fibgz'),
                                 ('fod_sh_mif', 'fod_sh_mif')]),
        (resample_mask, plot_peaks, [('out_file', 'mask_file')]),
        (inputnode, plot_peaks, [('dwi_ref', 'background_image'),
                                 ('odf_rois', 'odf_rois')]),
        (recon_map, plot_peaks, [('odf_directions', 'directions_file'),
                                 ('odf_amplitudes', 'odf_file')]),
        (plot_peaks, ds_report_peaks, [('out_report', 'in_file')])])

    # Plot targeted regions
    if has_transform:
        ds_report_odfs = pe.Node(
            ReconDerivativesDataSink(extension='.png',
                                     desc="MAPLMRIODF",
                                     suffix='odfs'),
            name='ds_report_odfs',
            run_without_submitting=True)
        workflow.connect(plot_peaks, 'odf_report', ds_report_odfs, 'in_file')

    if output_suffix:
        external_format_datasinks(output_suffix, params, workflow)
        connections = []
        for scalar_name in ['rtop', 'rtap', 'rtpp', 'qiv', 'msd', 'lapnorm']:
            connections += [(outputnode,
                             pe.Node(
                                 ReconDerivativesDataSink(desc=scalar_name,
                                                          suffix=output_suffix),
                                 name='ds_%s_%s' % (name, scalar_name)),
                             [(scalar_name, 'in_file')])]
        workflow.connect(connections)
    workflow.__desc__ = desc
    return workflow
