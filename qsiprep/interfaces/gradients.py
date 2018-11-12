"""Handle merging and spliting of DSI files."""
import numpy as np
import os.path as op
from nipype.interfaces.base import (BaseInterfaceInputSpec, TraitedSpec, File, SimpleInterface,
                                    InputMultiObject, traits)


class RecombineDWIsInputSpec(BaseInterfaceInputSpec):
    dwi_chunks = InputMultiObject(
        File(exists=True), mandatory=True, desc='list of dwi chunks')
    bval_chunks = InputMultiObject(
        File(exists=True), mandatory=True, desc='list of bval chunks')
    bvec_chunks = InputMultiObject(
        File(exists=True), mandatory=True, desc='list of bvec chunks')
    original_bval = File(exists=True, mandatory=True,
                         desc='bval file from before splitting')
    original_bvec = File(exists=True, mandatory=True,
                         desc='bvec file from before splitting')
    b0_images = InputMultiObject(
        File(exists=True), mandatory=True, desc='list of b0 images')
    b0_threshold = traits.Int(10, desc='maximum b value for a image to be considered b=0')


class RecombineDWIsOutputSpec(TraitedSpec):
    out_dwi = File(desc='the merged dwi image')
    out_bval = File(desc='the merged bvec file')
    out_bvec = File(desc='the merged bval file')
    out_b0s = File(desc='4d series of b0 images')


class RecombineDWIs(SimpleInterface):
    input_spec = RecombineDWIsInputSpec
    output_spec = RecombineDWIsOutputSpec

    def _run_interface(self, runtime):
        concat_dwi, concat_bvals, concat_bvecs, concat_b0 = recombine_dwis(
            self.inputs.dwi_chunks, self.inputs.bval_chunks, self.inputs.bvec_chunks,
            self.inputs.original_bval, self.inputs.original_bvec, self.inputs.b0_images,
            self.inputs.b0_threshold)

        self._results['out_dwi'] = concat_dwi
        self._results['out_bval'] = concat_bvals
        self._results['out_bvec'] = concat_bvecs
        self._results['out_b0s'] = concat_b0

        return runtime


def split_dwi(dwi_nifti="", bval_file="", bvec_file="", b0_threshold=10):
    """
    split DWI into a series of b0 images and their following dwi
    """
    from dipy.io import read_bvals_bvecs
    import nibabel as nib
    import numpy as np
    import os

    bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)
    img = nib.load(dwi_nifti)
    data = img.get_data()
    aff = img.get_affine()

    b0s = np.flatnonzero(bvals < b0_threshold)
    b0_paths = []
    dwi_paths = []
    bval_paths = []
    bvec_paths = []

    # Which b0 is each diffusion-weighted volume nearest to?
    from collections import defaultdict
    nearest_b0 = defaultdict(list)
    for n in np.arange(data.shape[-1]):
        if n in b0s:
            continue
        nearest = np.argmin(np.abs(n - b0s))
        nearest_b0[b0s[nearest]].append(n)

    # Save the b0s and their corresponding nearest images
    for n_b0, b0_index in enumerate(sorted(nearest_b0.keys())):
        # output paths
        out_path = "b0_%03d.nii.gz" % n_b0
        dwi_out_path = "dwi_%03d.nii.gz" % n_b0
        bval_out_path = "dwi_%03d.bval" % n_b0
        bvec_out_path = "dwi_%03d.bvec" % n_b0

        indices = np.array(nearest_b0[b0_index])
        dwi = data[..., indices]
        _bval = bvals[indices]
        _bvec = bvecs[indices]

        nib.Nifti1Image(dwi, aff).to_filename(dwi_out_path)
        nib.Nifti1Image(data[..., b0_index], aff).to_filename(out_path)
        np.savetxt(bvec_out_path, _bvec)
        np.savetxt(bval_out_path, _bval)

        dwi_paths.append(os.path.abspath(dwi_out_path))
        b0_paths.append(os.path.abspath(out_path))
        bval_paths.append(os.path.abspath(bval_out_path))
        bvec_paths.append(os.path.abspath(bvec_out_path))
    return b0_paths, dwi_paths, bval_paths, bvec_paths


def multi_transform_bvecs(bvec_file, transform_list):
    """
    Use antsApplyTransformsToPoints to warp two points (a vector)
    and uses the relationship between the warped points to rotate the
    b vector. Only works if absolutely everything is in LPS+.
    transform_list must be in the order they would be specified on the
    commandline to antsApplyTransformsToPoints. They will be inverted
    in this function.
    """
    import os
    import numpy as np
    orig_bvec = np.loadtxt(bvec_file)
    bvec_txt = open("pre_rotation.csv", "w")
    bvec_txt.write("x,y,z,t\n0.0,0.0,0.0,0.0\n")
    # In case there is only one bvec
    if orig_bvec.ndim == 1:
        orig_bvec = orig_bvec[None, :]
    for vec in orig_bvec:
        bvec_txt.write(",".join(map(str, 5 * vec)) + ",0.0\n")
    bvec_txt.close()  # Save it for ants

    def unit_vector(vector):
        """ The unit vector of the vector."""
        if np.abs(vector).sum() == 0:
            return vector
        return vector / np.linalg.norm(vector)

    transforms = " ".join(
        ["--transform [%s, 1]" % trf for trf in transform_list])
    os.system("antsApplyTransformsToPoints "
              "--dimensionality 3 "
              "--input pre_rotation.csv "
              "--output rotated_vecs.csv " +
              transforms)

    rotated_vecs = np.loadtxt(
        "rotated_vecs.csv", skiprows=1, delimiter=",")[:, :3]
    rotated_vecs = np.row_stack(
        [unit_vector(v) for v in rotated_vecs - rotated_vecs[0]])
    np.savetxt("rotated_aattp.bvec", rotated_vecs[1:].T, fmt=str("%.8f"))
    return os.path.abspath("rotated_aattp.bvec")


def recombine_dwis(dwi_chunks,
                   bval_chunks,
                   bvec_chunks,
                   original_bvals,
                   original_bvecs,
                   b0_images,
                   b0_threshold=10):
    from dipy.io import read_bvals_bvecs
    import nibabel as nib
    import numpy as np
    import os
    from os.path import abspath

    bvals, bvecs = read_bvals_bvecs(original_bvals, original_bvecs)
    template_img = nib.load(b0_images[0])
    output_dwi_data = np.zeros(template_img.shape + (len(bvals), ))

    # Fill in the b0_images
    b0_indices = np.flatnonzero(bvals < b0_threshold)
    output_b0_data = np.zeros(template_img.shape + (len(b0_indices), ))
    assert len(b0_indices) == len(b0_images)
    for b0_num, (b0_index, b0_image) in enumerate(zip(b0_indices, b0_images)):
        b0_data = nib.load(b0_image).get_data()
        output_dwi_data[..., b0_index] = b0_data
        output_b0_data[..., b0_num] = b0_data
    nib.Nifti1Image(output_b0_data,
                    template_img.affine).to_filename("concatenated_b0s.nii.gz")
    # Fill in the non-b0 images
    non_b0_indices = np.flatnonzero(bvals >= b0_threshold)
    reversed_non_b0_indices = non_b0_indices[::-1].tolist()
    for dir_dwi_image in dwi_chunks:
        dwi_img = nib.load(dir_dwi_image)
        if len(dwi_img.shape) == 3:
            idx = reversed_non_b0_indices.pop()
            output_dwi_data[..., idx] = dwi_img.get_data()
        elif len(dwi_img.shape) == 4:
            chunk_data = dwi_img.get_data()
            for within_chunk_index in range(dwi_img.shape[3]):
                idx = reversed_non_b0_indices.pop()
                output_dwi_data[..., idx] = chunk_data[..., within_chunk_index]
    assert len(reversed_non_b0_indices) == 0
    # dwi signal cannot be negative. Sometimes interpolation causes it to be.
    output_dwi_data[output_dwi_data < 0] = 0
    nib.Nifti1Image(output_dwi_data,
                    template_img.affine).to_filename("concatenated_dwi.nii.gz")

    # Recombine the bvals
    output_bvals = np.zeros_like(bvals)
    chunks = [np.loadtxt(fname) for fname in bval_chunks]

    concat_bvals = np.concatenate(
        [np.loadtxt(fname, ndmin=1) for fname in bval_chunks])
    output_bvals[non_b0_indices] = concat_bvals
    np.savetxt("concatenated.bvals", output_bvals[:, None].T, fmt=str("%d"))
    # Recombine the bvals
    output_bvecs = np.zeros_like(bvecs)
    concat_bvecs = np.column_stack(
        [np.loadtxt(fname, ndmin=2) for fname in bvec_chunks]).T
    output_bvecs[non_b0_indices] = concat_bvecs
    np.savetxt("concatenated.bvecs", output_bvecs.T, fmt=str("%.8f"))

    return [
        abspath("concatenated_dwi.nii.gz"),
        abspath("concatenated.bval"),
        abspath("concatenated.bvec"),
        abspath("concatenated_b0s.nii.gz")
    ]


def combine_bvals(bvals):
    """Load, merge and save fsl-style bvals files."""
    collected_vals = []
    for bval_file in bvals:
        collected_vals.append(np.loadtxt(bval_file))
    final_bvals = np.concatenate(collected_vals)
    np.savetxt("restacked.bval", final_bvals, fmt=str("%i"))
    return op.abspath("restacked.bval")


def combine_bvecs(bvecs):
    """Load, merge and save fsl-style bvecs files."""
    collected_vecs = []
    for bvec_file in bvecs:
        collected_vecs.append(np.loadtxt(bvec_file))
    final_bvecs = np.column_stack(collected_vecs)
    np.savetxt("restacked.bvec", final_bvecs, fmt=str("%.8f"))
    return op.abspath("restacked.bvec")
