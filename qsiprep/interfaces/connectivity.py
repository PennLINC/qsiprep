from __future__ import print_function
import logging
import numpy as np
from scipy.io.matlab import loadmat, savemat
from scipy.linalg import schur, svd
from nipype.interfaces.base import TraitedSpec, BaseInterfaceInputSpec, File, SimpleInterface
from nipype.utils.filemanip import fname_presuffix

LOGGER = logging.getLogger('nipype.interface')


class ControllabilityInputSpec(BaseInterfaceInputSpec):
    matfile = File(exists=True, desc='connectivity matrices in matlab format')


class ControllabilityOutputSpec(TraitedSpec):
    controllability = File(exists=True, desc='input connectivity data and controllability')


class Controllability(SimpleInterface):
    input_spec = ControllabilityInputSpec
    output_spec = ControllabilityOutputSpec

    def _run_interface(self, runtime):
        mat = loadmat(self.inputs.matfile, squeeze_me=True)
        outfile = fname_presuffix(self.inputs.matfile, suffix="_controllability",
                                  newpath=runtime.cwd)
        connectivity_info = _calculate_controllability(mat)
        LOGGER.info("writing %s", outfile)
        savemat(outfile, connectivity_info, do_compression=True)
        self._results['controllability'] = outfile
        return runtime


def ave_control(A):
    Anormed = A / (1 + svd(A)[1][0])   # Matrix normalization
    T, U = schur(Anormed, 'real')    # Schur stability

    midMat = (U**2).T
    v = np.diag(T)
    P = np.column_stack([1 - v*v.T] * A.shape[0])
    return np.sum(midMat/P, axis=0)


def modal_control(A):
    Anormed = A / (1 + svd(A)[1][0])   # Matrix normalization
    T, U = schur(Anormed, 'real')    # Schur stability
    eigVals = np.diag(T)
    N = A.shape[0]
    phi = np.zeros(N)

    b = 1-eigVals**2
    U2 = U**2
    for i in range(N):
        phi[i] = np.dot(U2[i], b)
    return phi


def _calculate_controllability(mat):
    connectivity_keys = [k for k in mat.keys() if k.endswith("connectivity")]
    for key in connectivity_keys:
        adjmat = mat[key]
        mat[key + "_modal_ctl"] = modal_control(adjmat)
        mat[key + "_ave_ctl"] = ave_control(adjmat)
    return mat
