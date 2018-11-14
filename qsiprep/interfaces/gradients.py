"""Handle merging and spliting of DSI files."""
import numpy as np
import os.path as op
import nibabel as nb
from nipype.interfaces.base import (BaseInterfaceInputSpec, TraitedSpec, File, SimpleInterface,
                                    InputMultiObject, OutputMultiObject, traits, isdefined)
from nipype.interfaces import afni

class WarpAndRecombineDWIsInputSpec(BaseInterfaceInputSpec):
    dwi_files = InputMultiObject(
        File(exists=True), mandatory=True, desc='list of dwi files')
    bval_files = InputMultiObject(
        File(exists=True), mandatory=True, desc='list of bval files')
    bvec_files = InputMultiObject(
        File(exists=True), mandatory=True, desc='list of bvec files')
    original_b0_indices = traits.List(mandatory=True)
    reference_image = File(exists=True, mandatory=True,
                           desc='grid into which outputs are written')
    # Transforms to apply
    b0_hmc_affines = InputMultiObject(File(exists=True), mandatory=True,
                                      desc='affines registering b0s to motioncorr target')
    motion_corr_to_dwi_ref_affine = File(exists=True, mandatory=True)
    dwi_ref_to_unwarped_warp = File(exists=True, mandtory=False,
                                    desc='SDC unwarping transform')
    dwi_ref_to_t1w_affine = File(exists=True, mandatory=True, desc='affine from dwi ref to t1w')
    t1w_to_mni_affine = File(exists=True, mandatory=False, desc='affine from t1w to MNI')
    t1w_to_mni_warp = File(exists=True, mandatory=False, desc='t1w to MNI warp')


class WarpAndRecombineDWIsOutputSpec(TraitedSpec):
    out_dwi = File(desc='the merged dwi image')
    out_bval = File(desc='the merged bvec file')
    out_bvec = File(desc='the merged bval file')
    out_b0s = File(desc='4d series of b0 images')


class WarpAndRecombineDWIs(SimpleInterface):
    input_spec = WarpAndRecombineDWIsInputSpec
    output_spec = WarpAndRecombineDWIsOutputSpec

    def _run_interface(self, runtime):
        dwi_files = self.inputs.dwi_files
        bval_files = self.inputs.bval_files
        bvec_files = self.inputs.bvec_files
        b0_hmc_affines = self.inputs.b0_hmc_affines
        original_b0_indices = np.array(self.inputs.original_b0_indices).astype(np.int)
        num_dwis = len(dwi_files)

        # Do sanity checks
        if not len(dwi_files) == len(bval_files) == len(bvec_files):
            raise Exception("bvals, bvecs and dwis do not match")
        if not len(b0_hmc_affines) == len(original_b0_indices):
            raise Exception('number of hmc affines do not match number of b0 images')

        # Create a list of which hmc affines go with each of the split images
        dwi_hmc_affines = []
        for index in range(num_dwis):
            # There is an direct transform for each b0
            if index in original_b0_indices:
                this_transform = b0_hmc_affines[index]
            else:
                nearest_b0_num = np.argmin(np.abs(index - original_b0_indices))
                this_transform = b0_hmc_affines[nearest_b0_num]
            dwi_hmc_affines.append(this_transform)

        if not len(dwi_hmc_affines) == num_dwis:
            raise Exception("Shouldn't happen")

        image_transforms = [self.inputs.motion_corr_to_dwi_ref_affine,
                            self.inputs.dwi_ref_to_unwarped_warp,
                            self.inputs.dwi_ref_to_t1w_affine,
                            self.inputs.t1w_to_mni_affine,
                            self.inputs.t1w_to_mni_warp]
        final_image_transforms = [trf for trf in image_transforms if isdefined(trf)]

        rotated_bvecs = []
        warped_images = []
        for dwi_file, bvec, hmc_transform in zip(dwi_files, bvec_files, dwi_hmc_affines):
            (warped_image, rotated_bvec_file,
                rotated_bvec_image) = warp_dwi(dwi_file, bvec,
                                               [hmc_transform] + final_image_transforms)
            rotated_bvecs.append(rotated_bvec_file)
            warped_images.append(warped_image)

        # recombine the bvalues (these shouldn't change)
        combined_bval = np.concatenate(
            [np.loadtxt(bval_file) for bval_file in bval_files]).squeeze()

        # recombine the rotated bvecs
        combined_bvec = np.row_stack([np.loadtxt(fname, ndmin=2) for fname in rotated_bvecs])

        combined_dwis = afni.TCat(in_files=warped_images, outputtype="NIFTI_GZ"
                                  ).run().outputs.out_file

        self._results['out_dwi'] = combined_dwis
        self._results['out_bval'] = combined_bval
        self._results['out_bvec'] = combined_bvec

        return runtime


def warp_dwi(image, bvec_file, transform_list):
    """Requires that bvec_file contains only 3 numbers on a single line.

    Use antsApplyTransformsToPoints to warp two points (a vector)
    and uses the relationship between the warped points to rotate the
    b vector. Only works if absolutely everything is in LPS+.
    transform_list must be in the order they would be specified on the
    commandline to antsApplyTransformsToPoints. They will be inverted
    in this function.
    """
    import os
    import numpy as np
    orig_vec = np.loadtxt(bvec_file)
    # Save it for ants
    with open("pre_rotation.csv", "w") as bvec_txt:
        bvec_txt.write("x,y,z,t\n0.0,0.0,0.0,0.0\n")
        bvec_txt.write(",".join(map(str, 5 * orig_vec)) + ",0.0\n")

    def unit_vector(vector):
        """The unit vector of the vector."""
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

    rotated_vec = np.loadtxt(
        "rotated_vecs.csv", skiprows=1, delimiter=",")[:, :3]
    rotated_unit_vec = unit_vector(rotated_vec[1] - rotated_vec[0])

    return warped_image,


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
