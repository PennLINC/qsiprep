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
from qsiprep.interfaces.bids import QsiprepOutput, ReconDerivativesDataSink
from ...interfaces.dipy import BrainSuiteShoreReconstruction
from qsiprep.interfaces.converters import amplitudes_to_fibgz, amplitudes_to_sh_mif

LOGGER = logging.getLogger('nipype.interface')
qsiprep_output_names = QsiprepOutput().output_spec.class_editable_traits()
default_connections = [(trait, trait) for trait in qsiprep_output_names]
default_input_set = set(qsiprep_output_names)


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

    inputnode = pe.Node(niu.IdentityInterface(fields=qsiprep_output_names),
                        name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['shore_coeffs', 'rtop', 'rtap', 'rtpp']),
        name="outputnode")

    workflow = pe.Workflow(name=name)
    recon_shore = pe.Node(BrainSuiteShoreReconstruction(**params), name="recon_shore")

    workflow.connect([
        (inputnode, recon_shore, [('dwi_file', 'dwi_file'),
                                  ('bval_file', 'bval_file'),
                                  ('bvec_file', 'bvec_file'),
                                  ('mask_file', 'mask_file')]),

    ])
    return workflow
