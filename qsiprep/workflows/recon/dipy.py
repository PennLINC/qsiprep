"""
Dipy Reconstruction workflows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_dipy_brainsuite_shore_recon_wf

"""
import json
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
from nipype.utils.filemanip import copyfile, split_filename

import logging
import os
import os.path as op
from qsiprep.interfaces.bids import ReconDerivativesDataSink
from ...interfaces.dipy import BrainSuiteShoreReconstruction, MAPMRIReconstruction
from .interchange import input_fields

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


def init_dipy_brainsuite_shore_recon_wf(name="dipy_3dshore_recon", output_suffix="", params={}):
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

    inputnode = pe.Node(niu.IdentityInterface(fields=input_fields),
                        name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['shore_coeffs', 'rtop', 'rtap', 'rtpp', 'fibgz', 'fod_sh_mif']),
        name="outputnode")

    workflow = pe.Workflow(name=name)
    recon_shore = pe.Node(BrainSuiteShoreReconstruction(**params), name="recon_shore")

    workflow.connect([
        (inputnode, recon_shore, [('dwi_file', 'dwi_file'),
                                  ('bval_file', 'bval_file'),
                                  ('bvec_file', 'bvec_file'),
                                  ('mask_file', 'mask_file')]),
        (recon_shore, outputnode, [('shore_coeffs', 'shore_coeffs'),
                                   ('rtop', 'rtop'),
                                   ('fibgz', 'fibgz'),
                                   ('fod_sh_mif', 'fod_sh_mif')])

    ])
    if output_suffix:
        external_format_datasinks(output_suffix, params, workflow)
        ds_rtop = pe.Node(
            ReconDerivativesDataSink(extension='.nii.gz',
                                     desc="rtop",
                                     suffix=output_suffix,
                                     compress=True),
            name='ds_bsshore_rtop',
            run_without_submitting=True)
        workflow.connect(outputnode, 'rtop', ds_rtop, 'in_file')
        ds_coeff = pe.Node(
            ReconDerivativesDataSink(extension='.nii.gz',
                                     desc="SHOREcoeff",
                                     suffix=output_suffix,
                                     compress=True),
            name='ds_bsshore_coeff',
            run_without_submitting=True)
        workflow.connect(outputnode, 'shore_coeffs', ds_coeff, 'in_file')

    return workflow


def init_dipy_mapmri_recon_wf(name="dipy_mapmri_recon", output_suffix="", params={}):
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
        lapnorm
            Voxelwise norm of the Laplacian

    Params

        write_fibgz: bool
            True writes out a DSI Studio fib file
        write_mif: bool
            True writes out a MRTrix mif file with sh coefficients
        radial_order: int
            An even integer that represent the order of the basis
        lablacian_regularization: bool
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

    inputnode = pe.Node(niu.IdentityInterface(fields=input_fields),
                        name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['mapmri_coeffs', 'rtop', 'rtap', 'rtpp', 'fibgz', 'fod_sh_mif',
                    'parng', 'perng', 'ng', 'qiv', 'lapnorm']),
        name="outputnode")

    workflow = pe.Workflow(name=name)
    recon_map = pe.Node(MAPMRIReconstruction(**params), name="recon_map")

    workflow.connect([
        (inputnode, recon_map, [('dwi_file', 'dwi_file'),
                                ('bval_file', 'bval_file'),
                                ('bvec_file', 'bvec_file'),
                                ('mask_file', 'mask_file')]),
        (recon_map, outputnode, [('mapmri_coeffs', 'mapmri_coeffs'),
                                 ('rtop', 'rtop'),
                                 ('rtap', 'rtap'),
                                 ('rtpp', 'rtpp'),
                                 ('parng', 'parng'),
                                 ('perng', 'perng'),
                                 ('ng', 'ng'),
                                 ('qiv', 'qiv'),
                                 ('lapnorm', 'lapnorm'),
                                 ('fibgz', 'fibgz'),
                                 ('fod_sh_mif', 'fod_sh_mif')])

    ])
    if output_suffix:
        external_format_datasinks(output_suffix, params, workflow)
        connections = []
        for scalar_name in ['rtop', 'rtap', 'rtpp']:
            connections += [(outputnode,
                             pe.Node(
                                 ReconDerivativesDataSink(desc=scalar_name,
                                                          suffix=output_suffix),
                                 name='ds_%s_%s' % (name, scalar_name)),
                             [(scalar_name, 'in_file')])]
        workflow.connect(connections)

    return workflow
