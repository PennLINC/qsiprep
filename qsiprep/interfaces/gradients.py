"""Handle merging and spliting of DSI files."""
import numpy as np
import os.path as op
import nibabel as nb
from nipype.interfaces.base import (BaseInterfaceInputSpec, TraitedSpec, File, SimpleInterface,
                                    InputMultiObject, OutputMultiObject, traits, isdefined)
from nipype.interfaces import afni, ants
from nipype.utils.filemanip import fname_presuffix


class WarpAndRecombineDWIsInputSpec(BaseInterfaceInputSpec):
    output_grid = File(exists=True, mandatory=True,
                       desc='grid into which outputs are written')
    b0_ref_image = File(exists=True, desc='')
    dwi_files = InputMultiObject(
        File(exists=True), mandatory=True, desc='list of dwi files')
    bval_files = InputMultiObject(
        File(exists=True), mandatory=True, desc='list of bval files')
    bvec_files = InputMultiObject(
        File(exists=True), mandatory=True, desc='list of bvec files')
    dwi_mask = File(exists=True, desc='dwi mask image')
    original_b0_indices = traits.List(mandatory=True)
    # Transforms to apply
    hmc_to_ref_affines = InputMultiObject(File(exists=True), mandatory=True,
                                          desc='affines registering b0s to b0 reference')
    dwi_ref_to_unwarped_warp = InputMultiObject(File(exists=True), mandtory=False,
                                                desc='SDC unwarping transform')

    unwarped_dwi_ref_to_t1w_affine = File(exists=True, mandatory=True,
                                          desc='affine from dwi ref to t1w')
    t1_2_mni_forward_transform = InputMultiObject(File(exists=True), mandatory=False,
                                                  desc='affine and warp to mni')


class WarpAndRecombineDWIsOutputSpec(TraitedSpec):
    out_dwi = File(desc='the merged dwi image')
    out_bval = File(desc='the merged bvec file')
    out_bvec = File(desc='the merged bval file')
    out_b0s = File(desc='4d series of b0 images')
    out_b0_ref = File(desc='reference b0 image')

class WarpAndRecombineDWIs(SimpleInterface):
    input_spec = WarpAndRecombineDWIsInputSpec
    output_spec = WarpAndRecombineDWIsOutputSpec

    def _run_interface(self, runtime):
        dwi_files = self.inputs.dwi_files
        bval_files = self.inputs.bval_files
        bvec_files = self.inputs.bvec_files
        b0_hmc_affines = self.inputs.hmc_to_ref_affines
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
                this_transform = b0_hmc_affines[np.flatnonzero(original_b0_indices == index)[0]]
            else:
                nearest_b0_num = np.argmin(np.abs(index - original_b0_indices))
                this_transform = b0_hmc_affines[nearest_b0_num]
            dwi_hmc_affines.append(this_transform)

        if not len(dwi_hmc_affines) == num_dwis:
            raise Exception("Shouldn't happen")

        image_transforms = [dwi_hmc_affines,
                            self.inputs.dwi_ref_to_unwarped_warp,
                            [self.inputs.unwarped_dwi_ref_to_t1w_affine] * num_dwis]
        if isdefined(self.inputs.t1_2_mni_forward_transform):
            image_transforms.append([self.inputs.t1_2_mni_forward_transform * num_dwis])

        rotated_bvecs = []
        warped_images = []
        transform_lists = [list(its) for its in zip(*tuple(image_transforms))]
        for dwi_file, bvec, transforms in zip(dwi_files, bvec_files, transform_lists):
            (warped_image, rotated_bvec_file,
                warped_bvec_image) = _warp_dwi(dwi_file, bvec, transforms[::-1],
                                               self.inputs.output_grid, runtime)
            rotated_bvecs.append(rotated_bvec_file)
            warped_images.append(warped_image)

        # recombine the bvalues (these shouldn't change)
        combined_bval = np.concatenate(
            [np.loadtxt(bval_file) for bval_file in bval_files]).squeeze()

        # recombine the rotated bvecs
        combined_bvec = np.row_stack([np.loadtxt(fname, ndmin=2) for fname in rotated_bvecs])

        # recombine the images
        combined_dwis = afni.TCat(in_files=warped_images, outputtype="NIFTI_GZ"
                                  ).run().outputs.out_file

        # Make just a b0 series
        combined_b0s = afni.TCat(in_files=[warped_images[n] for n in
                                           self.inputs.original_b0_indices],
                                 outputtype="NIFTI_GZ").run().outputs.out_file

        self._results['out_dwi'] = combined_dwis
        self._results['out_bval'] = combined_bval
        self._results['out_bvec'] = combined_bvec
        self._results['out_b0s'] = combined_b0s

        return runtime


def _warp_dwi(image, bvec_file, transform_list, ref_image, runtime):
    """Requires that bvec_file contains only 3 numbers on a single line.

    Use antsApplyTransformsToPoints to warp two points (a vector)
    and uses the relationship between the warped points to rotate the
    b vector. Only works if absolutely everything is in LPS+.
    transform_list must be in the order they would be specified on the
    commandline to antsApplyTransforms. They will be inverted
    in this function (ie last transform is applied first).
    """
    import os
    import numpy as np
    orig_vec = np.loadtxt(bvec_file)
    if (orig_vec**2).sum() == 0:
        rotated_unit_vec = orig_vec
    else:
        orig_txt = fname_presuffix(bvec_file, suffix='_pre_rotation.csv', newpath=runtime.cwd,
                                   use_ext=False)
        rotated_txt = fname_presuffix(bvec_file, suffix='_post_rotation.csv', newpath=None,
                                      use_ext=False)

        # Save it for ants
        with open(orig_txt, "w") as bvec_txt:
            bvec_txt.write("x,y,z,t\n0.0,0.0,0.0,0.0\n")
            bvec_txt.write(",".join(map(str, 5 * orig_vec)) + ",0.0\n")

        def unit_vector(vector):
            """The unit vector of the vector."""
            if np.abs(vector).sum() == 0:
                return vector
            return vector / np.linalg.norm(vector)

        # Only use the affine transforms for global bvecs
        bvec_transforms = [trf for trf in transform_list if ".nii" not in trf]
        # Reverse order and inverse to antsApplyTransformsToPoints
        transforms = " ".join(
            ["--transform [%s, 1]" % trf for trf in bvec_transforms[::-1]])
        os.system("antsApplyTransformsToPoints --dimensionality 3 --input " + orig_txt +
                  " --output " + rotated_txt + " " + transforms)
        rotated_vec = np.loadtxt(rotated_txt, skiprows=1, delimiter=",")[:, :3]
        rotated_unit_vec = unit_vector(rotated_vec[1] - rotated_vec[0])

    # Apply all transforms to the image
    warped_image = fname_presuffix(image, suffix='_warped', newpath=runtime.cwd, use_ext=True)
    xfm = ants.ApplyTransforms(
        input_image=image, transforms=transform_list, output_image=warped_image,
        interpolation="LanczosWindowedSinc", dimension=3, reference_image=ref_image)
    xfm.run()

    return warped_image, rotated_unit_vec, None


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
