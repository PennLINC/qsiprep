
"""Handle merging and spliting of DSI files."""
import numpy as np
import os
import os.path as op
import nibabel as nb
from nipype.interfaces.base import (BaseInterfaceInputSpec, TraitedSpec, File, SimpleInterface,
                                    InputMultiObject, OutputMultiObject, traits, isdefined)
from nipype.utils.filemanip import fname_presuffix
import logging
from dipy.core.geometry import cart2sphere
from dipy.direction import peak_directions
from dipy.core.sphere import HemiSphere
import subprocess
from scipy.io.matlab import loadmat, savemat
from pkg_resources import resource_filename as pkgr


LOGGER = logging.getLogger('nipype.workflow')
ODF_COLS = 20000  # Number of columns in DSI Studio odf split


class FODtoFIBGZInputSpec(BaseInterfaceInputSpec):
    fod_file = File(exists=True, mandatory=True)
    mask_file = File(exists=True, mandatory=True)
    num_fibers = traits.Int(3, usedefault=True)


class FODtoFIBGZOutputSpec(TraitedSpec):
    fibgz = File(exists=True)


class FODtoFIBGZ(SimpleInterface):
    input_spec = FODtoFIBGZInputSpec
    output_spec = FODtoFIBGZOutputSpec

    def _run_interface(self, runtime):
        mif_file = self.inputs.fod_file
        mask_file = self.inputs.mask_file
        output_fib_file = fname_presuffix(mif_file, newpath=runtime.cwd, extension=".fib")

        verts, faces = get_dsi_studio_ODF_geometry("odf8")
        num_dirs, _ = verts.shape
        hemisphere = num_dirs // 2
        x, y, z = verts[:hemisphere].T
        _, theta, phi = cart2sphere(x, y, z)
        hs = HemiSphere(x=x, y=y, z=z)
        dirs_txt = op.join(runtime.cwd, "directions.txt")
        np.savetxt(dirs_txt, np.column_stack([phi, theta]))

        odf_amplitudes_nii = op.join(runtime.cwd, "amplitudes.nii.gz")
        popen_run(["sh2amp", "-nonnegative", mif_file, dirs_txt, odf_amplitudes_nii])

        if not op.exists(odf_amplitudes_nii):
            raise FileNotFoundError("Unable to create %s", odf_amplitudes_nii)
        amplitudes_img = nb.load(odf_amplitudes_nii)
        ampl_data = amplitudes_img.get_data()

        if isdefined(mask_file):
            mask_img = nb.load(mask_file)
        else:
            ampl_mask = ampl_data.sum(3) > 1e-6
            nb.Nifti1Image(ampl_mask.astype(np.float),
                           amplitudes_img.affine)

        if not np.allclose(mask_img.affine, amplitudes_img.affine):
            raise ValueError("Differing orientation between mask and amplitudes")
        if not mask_img.shape == amplitudes_img.shape[:3]:
            raise ValueError("Differing grid between mask and amplitudes")

        # Get the flat mask
        flat_mask = mask_img.get_data().flatten(order="F") > 0
        odf_array = ampl_data.reshape(-1, ampl_data.shape[3], order="F")
        masked_odfs = odf_array[flat_mask, :]
        n_fibers = self.inputs.num_fibers
        n_odfs = masked_odfs.shape[0]
        peak_indices = np.zeros((n_odfs, n_fibers))
        peak_vals = np.zeros((n_odfs, n_fibers))

        dsi_mat = {}
        # Create matfile that can be read by dsi Studio
        dsi_mat['dimension'] = np.array(amplitudes_img.shape[:3])
        dsi_mat['voxel_size'] = np.array(amplitudes_img.header.get_zooms()[:3])
        n_voxels = int(np.prod(dsi_mat['dimension']))
        LOGGER.info("Detecting Peaks")
        for odfnum in range(masked_odfs.shape[0]):
            dirs, vals, indices = peak_directions(masked_odfs[odfnum], hs)
            for dirnum, (val, idx) in enumerate(zip(vals, indices)):
                if dirnum == n_fibers:
                    break
                peak_indices[odfnum, dirnum] = idx
                peak_vals[odfnum, dirnum] = val

        for nfib in range(n_fibers):
            # fill in the "fa" values
            fa_n = np.zeros(n_voxels)
            fa_n[flat_mask] = peak_vals[:, nfib]
            dsi_mat['fa%d' % nfib] = fa_n.astype(np.float32)

            # Fill in the index values
            index_n = np.zeros(n_voxels)
            index_n[flat_mask] = peak_indices[:, nfib]
            dsi_mat['index%d' % nfib] = index_n.astype(np.int16)

        # Add in the ODFs
        num_odf_matrices = n_odfs // ODF_COLS
        split_indices = (np.arange(num_odf_matrices) + 1) * ODF_COLS
        odf_splits = np.array_split(masked_odfs, split_indices, axis=0)
        for splitnum, odfs in enumerate(odf_splits):
            dsi_mat['odf%d' % splitnum] = odfs.T.astype(np.float32)

        dsi_mat['odf_vertices'] = verts.T
        dsi_mat['odf_faces'] = faces.T
        dsi_mat['z0'] = np.array([1.])
        savemat(output_fib_file, dsi_mat, format='4', appendmat=False)
        self._results['fibgz'] = output_fib_file

        return runtime


def get_dsi_studio_ODF_geometry(odf_key):
    mat_path = pkgr('qsiprep', 'data/odfs.mat')
    m = loadmat(mat_path)
    odf_vertices = m[odf_key + "_vertices"].T
    odf_faces = m[odf_key + "_faces"].T
    return odf_vertices, odf_faces


def popen_run(arg_list):
    cmd = subprocess.Popen(arg_list, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
    out, err = cmd.communicate()
    LOGGER.info(out)
    LOGGER.info(err)
