
"""Handle merging and spliting of DSI files."""
import os
import logging
from subprocess import Popen, PIPE
import nibabel as nb
import numpy as np
import pandas as pd
from nilearn import image as nim
from nipype.interfaces.base import (BaseInterfaceInputSpec, TraitedSpec, File, SimpleInterface,
                                    InputMultiObject, OutputMultiObject, traits, isdefined)
from nipype.interfaces import ants
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.ants.resampling import ApplyTransformsInputSpec
from dipy.sims.voxel import all_tensor_evecs
from dipy.reconst.dti import decompose_tensor
from dipy.core.geometry import normalized_vector
from sklearn.metrics import r2_score
from .itk import disassemble_transform

LOGGER = logging.getLogger('nipype.interface')
tensor_index = {
    "xx": (0, 0),
    "xy": (0, 1),
    "xz": (0, 2),
    "yy": (1, 1),
    "yz": (1, 2),
    "zz": (2, 2)
}


class RemoveDuplicatesInputSpec(BaseInterfaceInputSpec):
    dwi_file = File(exists=True, mandatory=True)
    bval_file = File(exists=True, mandatory=True)
    bvec_file = File(exists=True, mandatory=True)
    local_bvec_file = File(exists=True)
    distance_cutoff = traits.Float(5.0, usedefault=True)
    expected_directions = traits.Int()


class RemoveDuplicatesOutputSpec(TraitedSpec):
    dwi_file = File(exists=True)
    bval_file = File(exists=True)
    bvec_file = File(exists=True)
    local_bvec_file = File(exists=True)


class RemoveDuplicates(SimpleInterface):
    input_spec = RemoveDuplicatesInputSpec
    output_spec = RemoveDuplicatesOutputSpec

    def _run_interface(self, runtime):
        bvecs = np.loadtxt(self.inputs.bvec_file).T
        bvals = np.loadtxt(self.inputs.bval_file).squeeze()
        orig_bvals = bvals.copy()
        bvals = np.sqrt(bvals-bvals.min())
        bvals = bvals/bvals.max() * 100
        original_image = nb.load(self.inputs.dwi_file)
        cutoff = self.inputs.distance_cutoff

        scaled_bvecs = bvals[:, np.newaxis] * bvecs
        ok_vecs = []
        seen_vecs = []

        def is_unique_sample(vec):
            if len(seen_vecs) == 0:
                return True
            vec_array = np.row_stack(seen_vecs)
            distances = np.linalg.norm(vec_array - vec, axis=1)
            distances_flip = np.linalg.norm(vec_array + vec, axis=1)
            return np.all(distances > cutoff) and np.all(distances_flip > cutoff)

        for vec_num, vec in enumerate(scaled_bvecs):
            magnitude = np.linalg.norm(vec)
            # Is it a b0?
            if magnitude < cutoff:
                ok_vecs.append(vec_num)
            else:
                if is_unique_sample(vec):
                    ok_vecs.append(vec_num)
                    seen_vecs.append(vec)

        # If an expected number of directions was specified, check that it's there
        expected = self.inputs.expected_directions
        if isdefined(expected):
            if not len(seen_vecs) == expected:
                raise Exception("Expected %d unique samples but found %d",
                                expected, len(seen_vecs))

        # Are all the directions unique?
        if len(ok_vecs) == len(bvals):
            self._results['dwi_file'] = self.inputs.dwi_file
            self._results['bval_file'] = self.inputs.bval_file
            self._results['bvec_file'] = self.inputs.bvec_file
            self._results['local_bvec_file'] = self.inputs.local_bvec_file

            return runtime

        # Extract the unique samples
        output_bval = fname_presuffix(self.inputs.bval_file, newpath=runtime.cwd,
                                      suffix="_unique")
        output_bvec = fname_presuffix(self.inputs.bvec_file, newpath=runtime.cwd,
                                      suffix="_unique")
        output_nii = fname_presuffix(self.inputs.dwi_file, newpath=runtime.cwd,
                                     suffix="_unique")
        unique_indices = np.array(ok_vecs)
        unique_bvals = orig_bvals[unique_indices]
        np.savetxt(output_bval, unique_bvals, fmt='%d', newline=' ')
        unique_bvecs = bvecs[unique_indices]
        np.savetxt(output_bvec, unique_bvecs.T, fmt='%.8f')
        unique_data = original_image.get_data()[..., unique_indices]
        nb.Nifti1Image(unique_data, original_image.affine, original_image.header
                       ).to_filename(output_nii)
        self._results['bval_file'] = output_bval
        self._results['bvec_file'] = output_bvec
        self._results['dwi_file'] = output_nii
        # TODO: support local bvecs
        return runtime


class SliceQCInputSpec(BaseInterfaceInputSpec):
    uncorrected_dwi_files = InputMultiObject(File(exists=True), desc='uncorrected dwi files')
    ideal_image_files = InputMultiObject(File(exists=True), desc='model-based images')
    mask_image = File(exists=True, desc='brain mask')
    impute_slice_threshold = traits.Float(0., desc='threshold for using imputed data in a slice')
    min_slice_size_percentile = traits.CFloat(10.0, usedefault=True, desc='slices bigger than '
                                              'this percentile are candidates for imputation.')


class SliceQCOutputSpec(TraitedSpec):
    imputed_images = OutputMultiObject(File(exists=True), desc='dwi files with imputed slices')
    slice_stats = File(exists=True, desc='npy file with the slice-by-TR error matrix')


class SliceQC(SimpleInterface):
    input_spec = SliceQCInputSpec
    output_spec = SliceQCOutputSpec

    def _run_interface(self, runtime):
        threshold = self.inputs.impute_slice_threshold
        ideal_image_files = self.inputs.ideal_image_files
        uncorrected_image_files = self.inputs.uncorrected_dwi_files

        self._results['imputed_images'] = self.inputs.uncorrected_dwi_files
        if threshold > 0:
            LOGGER.warning("Imputation is not implemented yet")

        output_npz = os.path.join(runtime.cwd, "slice_stats.npz")
        mask_img = nb.load(self.inputs.mask_image)
        mask = mask_img.get_data() > 0
        masked_slices = (mask * np.arange(mask_img.shape[2])[np.newaxis, np.newaxis, :]
                         ).astype(np.int)
        slice_nums, slice_counts = np.unique(masked_slices[mask], return_counts=True)
        min_size = np.percentile(slice_counts, self.inputs.min_slice_size_percentile)
        too_small = slice_nums[slice_counts < min_size]
        for small_slice in too_small:
            masked_slices[masked_slices == small_slice] = 0
        valid_slices = slice_nums[slice_counts > min_size]
        valid_slices = valid_slices[valid_slices > 0]
        slice_scores = []
        wb_xcorrs = []
        wb_r2s = []
        # If impute slice threshold==0 or hmc_model=="none"
        if isdefined(ideal_image_files):
            for ideal_image, input_image in zip(ideal_image_files,
                                                uncorrected_image_files):
                slices, wb_xcorr, wb_r2 = _score_slices(ideal_image, input_image,
                                                        masked_slices, valid_slices)
                slice_scores.append(slices)
                wb_xcorrs.append(wb_xcorr)
                wb_r2s.append(wb_r2)
        else:
            num_trs = len(uncorrected_image_files)
            num_slices = mask_img.shape[2]
            wb_xcorrs = np.zeros(num_trs)
            wb_r2s = np.zeros(num_trs)
            slice_scores = np.zeros((num_slices, num_trs))

        np.savez(output_npz, slice_scores=slice_scores, wb_r2s=np.array(wb_r2s),
                 wb_xcorrs=np.array(wb_xcorrs), valid_slices=valid_slices,
                 masked_slices=masked_slices, slice_nums=slice_nums, slice_counts=slice_counts)
        self._results['slice_stats'] = output_npz
        return runtime


def _score_slices(ideal_image, input_image, masked_slices, valid_slices):
    """Compute similarity metrics on a pair of images."""
    def crosscor(vec1, vec2):
        v1bar = vec1 - vec1.mean()
        v2bar = vec2 - vec2.mean()
        return np.inner(v1bar, v2bar)**2 / (np.inner(v1bar, v1bar) * np.inner(v2bar, v2bar))

    slice_scores = np.zeros(valid_slices.shape)
    ideal_data = nb.load(ideal_image).get_fdata()
    input_data = nb.load(input_image).get_fdata()
    for nslice, slicenum in enumerate(valid_slices):
        slice_mask = masked_slices == slicenum
        ideal_slice = ideal_data[slice_mask]
        data_slice = input_data[slice_mask]
        slice_scores[nslice] = crosscor(ideal_slice, data_slice)

    global_mask = masked_slices > 0
    wb_ideal = ideal_data[global_mask]
    wb_input = input_data[global_mask]
    global_xcorr = crosscor(wb_input, wb_ideal)
    global_r2 = r2_score(wb_input, wb_ideal)
    return slice_scores, global_xcorr, global_r2


class CombineMotionsInputSpec(BaseInterfaceInputSpec):
    transform_files = InputMultiObject(File(exists=True), mandatory=True,
                                       desc='transform files from hmc')
    source_files = InputMultiObject(File(exists=True), mandatory=True,
                                    desc='Moving images')
    ref_file = File(exists=True, mandatory=True, desc='Fixed Image')


class CombineMotionsOututSpec(TraitedSpec):
    motion_file = File(exists=True)
    spm_motion_file = File(exists=True)


class CombineMotions(SimpleInterface):
    input_spec = CombineMotionsInputSpec
    output_spec = CombineMotionsOututSpec

    def _run_interface(self, runtime):
        collected_motion = []
        output_fname = os.path.join(runtime.cwd, "motion_params.csv")
        output_spm_fname = os.path.join(runtime.cwd, "spm_movpar.txt")
        ref_file = self.inputs.ref_file
        for motion_file, src_file in zip(self.inputs.transform_files,
                                         self.inputs.source_files):
            collected_motion.append(
                get_fsl_motion_params(motion_file, src_file, ref_file, runtime.cwd))

        final_motion = np.row_stack(collected_motion)
        cols = ["scaleX", "scaleY", "scaleZ", "shearXY", "shearXZ",
                "shearYZ", "rotateX", "rotateY", "rotateZ", "shiftX", "shiftY",
                "shiftZ"]
        motion_df = pd.DataFrame(data=final_motion, columns=cols)
        motion_df.to_csv(output_fname, index=False)
        self._results['motion_file'] = output_fname

        spmcols = motion_df[['shiftX', 'shiftY', 'shiftZ', 'rotateX', 'rotateY', 'rotateZ']]
        self._results['spm_motion_file'] = output_spm_fname
        np.savetxt(output_spm_fname, spmcols.values)

        return runtime


class MatchTransformsInputSpec(BaseInterfaceInputSpec):
    b0_indices = traits.List(mandatory=True)
    dwi_files = InputMultiObject(File(exists=True), mandatory=True)
    transforms = InputMultiObject(File(exists=True), mandatory=True)


class MatchTransformsOutputSpec(TraitedSpec):
    transforms = OutputMultiObject(File(exists=True), mandatory=True)


class MatchTransforms(SimpleInterface):
    input_spec = MatchTransformsInputSpec
    output_spec = MatchTransformsOutputSpec

    def _run_interface(self, runtime):
        self._results['transforms'] = match_transforms(self.inputs.dwi_files,
                                                       self.inputs.transforms,
                                                       self.inputs.b0_indices)
        return runtime


class ExtractB0sInputSpec(BaseInterfaceInputSpec):
    b0_indices = traits.List()
    bval_file = File(exists=True)
    b0_threshold = traits.Int(50, usedefault=True)
    dwi_series = File(exists=True, mandatory=True)


class ExtractB0sOutputSpec(TraitedSpec):
    b0_series = File(exists=True)
    b0_average = File(exists=True)


class ExtractB0s(SimpleInterface):
    """Extract a b0 series and a mean b0 from a dwi series."""

    input_spec = ExtractB0sInputSpec
    output_spec = ExtractB0sOutputSpec

    def _run_interface(self, runtime):
        output_fname = fname_presuffix(self.inputs.dwi_series, suffix='_b0_series',
                                       use_ext=True, newpath=runtime.cwd)
        output_mean_fname = fname_presuffix(output_fname, suffix='_mean',
                                            use_ext=True, newpath=runtime.cwd)
        if isdefined(self.inputs.b0_indices):
            indices = np.array(self.inputs.b0_indices).astype(np.int)
        elif isdefined(self.inputs.bval_file):
            bvals = np.loadtxt(self.inputs.bval_file)
            indices = np.flatnonzero(bvals < self.inputs.b0_threshold)
            if indices.size == 0:
                raise ValueError("No b<%d images found" % self.inputs.b0_threshold)
        else:
            raise ValueError("No gradient information available")
        new_data = nim.index_img(self.inputs.dwi_series, indices)
        new_data.to_filename(output_fname)
        self._results['b0_series'] = output_fname
        if new_data.ndim == 3:
            self._results['b0_average'] = output_fname
        else:
            mean_image = nim.math_img('img.mean(3)', img=new_data)
            mean_image.to_filename(output_mean_fname)
            self._results['b0_average'] = output_mean_fname

        return runtime


class SplitIntramodalTransformInputSpec(BaseInterfaceInputSpec):
    transform_file = File(exists=True)


class SplitIntramodalTransformOutputSpec(TraitedSpec):
    transform_files = OutputMultiObject(File(exists=True))


class SplitIntramodalTransform(SimpleInterface):
    input_spec = SplitIntramodalTransformInputSpec
    output_spec = SplitIntramodalTransformOutputSpec

    def _run_interface(self, runtime):
        transform_file = self.inputs.transform_file
        if isdefined(transform_file) and transform_file:
            if not transform_file.endswith(".h5"):
                self._results["transform_files"] = [transform_file]
            # If it's an h5 file, split it into affine and warp
            affine_part, warp_part = disassemble_transform(transform_file, runtime.cwd)
            self._results["transform_files"] = [affine_part, warp_part]
        else:
            self._results["transform_files"] = traits.Undefined
        return runtime


class ComposeTransformsInputSpec(ApplyTransformsInputSpec):
    input_image = File(mandatory=False)
    dwi_files = InputMultiObject(
        File(exists=True), mandatory=True, desc='list of dwi files')
    reference_image = File(exists=True, mandatory=True, desc='output grid')
    # Transforms to apply
    hmc_affines = InputMultiObject(File(exists=True),
                                   desc='head motion correction affines')
    fieldwarps = InputMultiObject(File(exists=True),
                                  mandtory=False,
                                  desc='SDC unwarping transform')
    b0_to_intramodal_template_transforms = InputMultiObject(
        File(exists=True), mandtory=False, desc='list of transforms to register the b=0 to '
        'the intramodal template.')
    intramodal_template_to_t1_affine = File(exists=True,
                                            desc='affine from the intramodal template to t1')
    intramodal_template_to_t1_warp = File(exists=True,
                                          desc='warp from the intramodal template to t1')
    hmcsdc_dwi_ref_to_t1w_affine = File(exists=True,
                                        desc='affine from dwi ref to t1w')
    t1_2_mni_forward_transform = InputMultiObject(File(exists=True), mandatory=False,
                                                  desc='composite (h5) transform to mni')
    save_cmd = traits.Bool(True, usedefault=True,
                           desc='write a log of command lines that were applied')
    copy_dtype = traits.Bool(False, usedefault=True,
                             desc='copy dtype from inputs to outputs')
    num_threads = traits.Int(1, usedefault=True, nohash=True,
                             desc='number of parallel processes')
    transforms = File(mandatory=False)


class ComposeTransformsOutputSpec(TraitedSpec):
    out_warps = OutputMultiObject(File(exists=True),
                                  desc='composed all transforms to output_grid')
    out_affines = OutputMultiObject(File(exists=True),
                                    desc='composed affine-only transforms to output_grid')
    transform_lists = OutputMultiObject(traits.List(File(exists=True)),
                                        desc='lists of transforms for each image')
    log_cmdline = File(desc='a list of command lines used to apply transforms')


class ComposeTransforms(SimpleInterface):
    input_spec = ComposeTransformsInputSpec
    output_spec = ComposeTransformsOutputSpec

    def _run_interface(self, runtime):
        dwi_files = self.inputs.dwi_files
        num_dwis = len(dwi_files)

        # concatenate transformations to get into the desired output space
        image_transforms = []
        image_transform_names = []

        def include_transform(transform):
            if not isdefined(transform):
                return False
            LOGGER.info('Including %s', transform)
            return len(transform) == num_dwis

        hmc_affines = self.inputs.hmc_affines
        fieldwarps = self.inputs.fieldwarps
        if isdefined(fieldwarps):
            fieldwarps = fieldwarps * num_dwis

        # The affine transform to the t1 can come from hmcsdc or the intramodal template
        coreg_to_t1 = traits.Undefined
        if isdefined(self.inputs.intramodal_template_to_t1_affine):
            if isdefined(self.inputs.hmcsdc_dwi_ref_to_t1w_affine):
                LOGGER.warning('Two b0 to t1 transforms are provided: using intramodal')
            coreg_to_t1 = self.inputs.intramodal_template_to_t1_affine
        else:
            coreg_to_t1 = self.inputs.hmcsdc_dwi_ref_to_t1w_affine
        if isdefined(coreg_to_t1):
            coreg_to_t1 = [coreg_to_t1] * num_dwis

        # Handle transforms to intramodal transforms
        intramodal_transforms = self.inputs.b0_to_intramodal_template_transforms
        intramodal_affine = traits.Undefined
        intramodal_warp = traits.Undefined
        if isdefined(intramodal_transforms):
            intramodal_affine = [intramodal_transforms[0]] * num_dwis
            if len(intramodal_transforms) == 2:
                intramodal_warp = [intramodal_transforms[1]] * num_dwis
            elif len(intramodal_transforms) > 2:
                raise Exception("Unsupported intramodal template transform")

        # If an intramodal template to t1 affine is present, copy for each dwi
        intramodal_template_to_t1_affine = self.inputs.intramodal_template_to_t1_affine
        if isdefined(intramodal_template_to_t1_affine):
            intramodal_template_to_t1_affine = [intramodal_template_to_t1_affine] * num_dwis

        # If an intramodal template to t1 warp is present, copy for each dwi
        intramodal_template_to_t1_warp = self.inputs.intramodal_template_to_t1_warp
        if isdefined(intramodal_template_to_t1_warp):
            intramodal_template_to_t1_affine = [intramodal_template_to_t1_warp] * num_dwis

        transform_order = [
            (hmc_affines, 'hmc'),
            (fieldwarps, 'fieldwarp'),
            (intramodal_affine, 'to b=0 affine'),
            (intramodal_warp, 'to b=0 warp'),
            (coreg_to_t1, 'b=0 to T1w'),
        ]

        for transform_list, transform_name in transform_order:
            LOGGER.info(transform_name)
            if include_transform(transform_list):
                image_transforms.append(transform_list)
                image_transform_names.append(transform_name)

        # Same t1-to-mni transform for every image
        mni_xform = self.inputs.t1_2_mni_forward_transform
        if isdefined(mni_xform):
            assert len(mni_xform) == 2
            image_transforms.append([mni_xform[0]] * num_dwis)
            image_transforms.append([mni_xform[1]] * num_dwis)
            image_transform_names += ['mni affine', 'mni warp']

        # Check that all the transform lists have the same numbers of transforms
        assert all([len(xform_list) == len(image_transforms[0]) for
                    xform_list in image_transforms])

        # If there is just a coreg transform, then we have everything
        if image_transform_names == ['b=0 to T1w']:
            self._results['out_warps'] = image_transforms[0]
            self._results['out_affines'] = image_transforms[0]
            self._results['transform_lists'] = image_transforms
            return runtime

        # Reverse the order for ANTs
        image_transforms = image_transforms[::-1]

        # List of lists, one list per input file
        xfms_list = []
        for image_num in range(num_dwis):
            xfms_list.append([xfm[image_num] for xfm in image_transforms])

        LOGGER.info("Composing %s transforms", ' -> '.join(image_transform_names))

        # Get all inputs from the ApplyTransforms object
        ifargs = self.inputs.get()

        # Extract number of input images and transforms
        # Get number of parallel jobs
        num_threads = ifargs.pop('num_threads')
        save_cmd = ifargs.pop('save_cmd')

        # Remove certain keys
        for key in ['environ', 'ignore_exception', 'print_out_composite_warp_file',
                    'terminal_output', 'output_image', 'input_image', 'transforms',
                    'dwi_files', 'original_b0_indices', 'hmc_affines',
                    'b0_to_intramodal_template_transforms',
                    'intramodal_template_to_t1_affine', 'intramodal_template_to_t1_warp',
                    'fieldwarps', 'hmcsdc_dwi_ref_to_t1w_affine', 'interpolation',
                    't1_2_mni_forward_transform', 'copy_dtype']:
            ifargs.pop(key, None)

        # In qsiprep the transforms have already been merged
        assert len(xfms_list) == num_dwis
        self._results['transform_lists'] = xfms_list

        # Inputs are ready to run in parallel
        if num_threads < 1:
            num_threads = None

        if num_threads == 1:
            out_files = [_compose_tfms((in_file, in_xfm, ifargs, i, runtime.cwd))
                         for i, (in_file, in_xfm) in enumerate(zip(dwi_files, xfms_list))]
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=num_threads) as pool:
                mapper = pool.map(_compose_tfms, [
                    (in_file, in_xfm, ifargs, i, runtime.cwd)
                    for i, (in_file, in_xfm) in enumerate(zip(dwi_files, xfms_list))])
                out_files = list(mapper)

        # Collect output file names, after sorting by index
        self._results['out_warps'] = [el[0] for el in out_files]
        self._results['out_affines'] = [el[2] for el in out_files]

        if save_cmd:
            self._results['log_cmdline'] = os.path.join(runtime.cwd, 'command.txt')
            with open(self._results['log_cmdline'], 'w') as cmdfile:
                print('\n-------\n'.join(
                      ['\n-------\n'.join([el[1], el[3]]) for el in out_files]),
                      file=cmdfile)
        return runtime


class GradientRotationInputSpec(BaseInterfaceInputSpec):
    affine_transforms = InputMultiObject(File(exists=True), desc='ITK affine transforms')
    original_images = InputMultiObject(
        File(exists=True), desc='NIfTI images corresponding to bvals, bvecs')
    bvec_files = InputMultiObject(File(exists=True),
                                  desc='list of split bvec files, must correspond to a '
                                       'non-oblique image/reference frame.')
    bval_files = InputMultiObject(File(exists=True), desc='list of split bval files')


class GradientRotationOutputSpec(TraitedSpec):
    bvals = File(exists=True)
    bvecs = File(exists=True)
    log_cmdline = File(exists=True)


class GradientRotation(SimpleInterface):
    """Reorient gradients accordint to transorms."""

    input_spec = GradientRotationInputSpec
    output_spec = GradientRotationOutputSpec

    def _run_interface(self, runtime):
        out_root = os.path.join(runtime.cwd, "rotated")

        # Simple concatenation of bvals
        bval_fname = out_root + ".bval"
        concatenate_bvals(self.inputs.bval_files, bval_fname)
        self._results['bvals'] = bval_fname

        bvec_fname = out_root + ".bvec"
        bvecs = concatenate_bvecs(self.inputs.bvec_files)
        commands = bvec_rotation(bvecs, self.inputs.affine_transforms, bvec_fname, runtime)
        self._results['bvecs'] = bvec_fname

        self._results['log_cmdline'] = os.path.join(runtime.cwd, 'command.txt')
        with open(self._results['log_cmdline'], 'w') as cmdfile:
            print('\n-------\n'.join(commands), file=cmdfile)
        return runtime


class LocalGradientRotationInputSpec(GradientRotationInputSpec):
    warp_transforms = InputMultiObject(File(exists=True), desc='Warps')
    mask_image = File(exists=True, desc='brain mask in the output space')
    bvec_files = InputMultiObject(File(exists=True), desc='list of split bvec files')


class LocalGradientRotationOutputSpec(TraitedSpec):
    local_bvecs = File(exists=True)
    log_cmdline = File(exists=True)


class LocalGradientRotation(SimpleInterface):
    input_spec = LocalGradientRotationInputSpec
    output_spec = LocalGradientRotationOutputSpec

    def _run_interface(self, runtime):
        out_root = os.path.join(runtime.cwd, "rotated")
        # Create the local bvecs
        local_bvec_fname = out_root + "_local_bvecs.nii.gz"
        self._results['local_bvecs'] = local_bvec_fname
        original_bvecs = concatenate_bvecs(self.inputs.bvec_files)
        commands = local_bvec_rotation(original_bvecs, self.inputs.warp_transforms,
                                       self.inputs.mask_image, runtime, local_bvec_fname)
        self._results['log_cmdline'] = os.path.join(runtime.cwd, 'command.txt')
        with open(self._results['log_cmdline'], 'w') as cmdfile:
            print('\n-------\n'.join(commands[1]), file=cmdfile)
        return runtime


def get_fsl_motion_params(itk_file, src_file, ref_file, working_dir):
    tmp_fsl_file = fname_presuffix(itk_file, newpath=working_dir,
                                   suffix='_FSL.xfm', use_ext=False)
    fsl_convert_cmd = "c3d_affine_tool " \
        "-ref {ref_file} " \
        "-src {src_file} " \
        "-itk {itk_file} " \
        "-ras2fsl -o {fsl_file}".format(
            src_file=src_file, ref_file=ref_file, itk_file=itk_file,
            fsl_file=tmp_fsl_file)
    os.system(fsl_convert_cmd)
    proc = Popen(['avscale', '--allparams', tmp_fsl_file, src_file], stdout=PIPE,
                 stderr=PIPE)
    stdout, _ = proc.communicate()

    def get_measures(line):
        line = line.strip().split()
        return np.array([float(num) for num in line[-3:]])

    lines = stdout.decode("utf-8").split("\n")
    flip = np.array([1, -1, -1])
    rotation = get_measures(lines[6]) * flip
    translation = get_measures(lines[8]) * flip
    scale = get_measures(lines[10])
    shear = get_measures(lines[12])

    return np.concatenate([scale, shear, rotation, translation])


def match_transforms(dwi_files, transforms, b0_indices):
    original_b0_indices = np.array(b0_indices)
    num_dwis = len(dwi_files)
    num_transforms = len(transforms)

    if num_dwis == num_transforms:
        return transforms

    # Do sanity checks
    if not len(transforms) == len(b0_indices):
        raise Exception('number of transforms does not match number of b0 images')

    # Create a list of which hmc affines go with each of the split images
    nearest_affines = []
    for index in range(num_dwis):
        nearest_b0_num = np.argmin(np.abs(index - original_b0_indices))
        this_transform = transforms[nearest_b0_num]
        nearest_affines.append(this_transform)

    return nearest_affines


def concatenate_bvals(bval_list, out_file):
    """Create an FSL-style bvals file from split bval files."""
    collected_vals = []
    for bval_file in bval_list:
        collected_vals.append(np.loadtxt(bval_file, ndmin=1))
    final_bvals = np.concatenate(collected_vals).squeeze()
    if out_file is not None:
        np.savetxt(out_file, final_bvals, fmt=str("%i"))
    return final_bvals


def concatenate_bvecs(input_files):
    """Create Dipy-style gradient array (3-columns) from bvec files."""
    if len(input_files) == 1:
        stacked = np.loadtxt(input_files[0])
    else:
        collected_vecs = []
        for bvec_file in input_files:
            collected_vecs.append(np.loadtxt(bvec_file))
            stacked = np.row_stack(collected_vecs)
    if not stacked.shape[1] == 3:
        stacked = stacked.T
    return stacked


def bvec_rotation(ortho_bvecs, transforms, output_file, runtime):
    """Rotate bvecs using antsApplyTransformsToPoints and antsTransformInfo.

    Parameters:
    -----------

    ortho_bvecs: np.ndarray (n, 3)
        bvecs relative to a non-oblique output volume

    transforms: list
        List of transform files that will be applied to the vectors

    original_images: list
        List of images that correspond to the original bvecs. Used to
        rotate the bvecs to world coordinates reference frame.

    output_file: str
        Path to write the new bvec file

    runtime: runtime object
        Nipype node runtime object

    """
    aattp_rotated = []
    commands = []
    for bvec, transform in zip(ortho_bvecs, transforms):
        vec, cmd = aattp_rotate_vec(bvec, transform, runtime)
        aattp_rotated.append(vec)
        commands.append(cmd)
    rotated_vecs = np.row_stack(aattp_rotated)
    np.savetxt(output_file, rotated_vecs.T, fmt=str("%.8f"))
    return commands


def aattp_rotate_vec(orig_vec, transform, runtime):
    if (orig_vec**2).sum() == 0:
        return orig_vec, "b0: No rotation"

    orig_txt = fname_presuffix(transform, suffix='_pre_rotation.csv', newpath=runtime.cwd,
                               use_ext=False)
    rotated_txt = fname_presuffix(transform, suffix='_post_rotation.csv', newpath=runtime.cwd,
                                  use_ext=False)

    # Save it for ants
    with open(orig_txt, "w") as bvec_txt:
        bvec_txt.write("x,y,z,t\n0.0,0.0,0.0,0.0\n")
        bvec_txt.write(",".join(map(str, 5 * orig_vec)) + ",0.0\n")

    def unit_vector(vector):
        """The unit vector of the vector."""
        return vector / np.linalg.norm(vector)

    # Only use the affine transforms for global bvecs
    # Reverse order and inverse to antsApplyTransformsToPoints
    transforms = "--transform [%s, 1]" % transform
    cmd = "antsApplyTransformsToPoints --dimensionality 3 --input " + orig_txt + \
          " --output " + rotated_txt + " " + transforms
    LOGGER.info(cmd)
    os.system(cmd)
    rotated_vec = np.loadtxt(rotated_txt, skiprows=1, delimiter=",")[:, :3]
    rotated_unit_vec = unit_vector(rotated_vec[1] - rotated_vec[0])

    return rotated_unit_vec, cmd


def _compose_tfms(args):
    """Create a composite transform from inputs."""
    in_file, in_xform, ifargs, index, newpath = args
    out_file = fname_presuffix(in_file, suffix='_xform-%05d' % index,
                               newpath=newpath, use_ext=True)

    xfm = ants.ApplyTransforms(
        input_image=in_file, transforms=in_xform, output_image=out_file,
        print_out_composite_warp_file=True, interpolation='LanczosWindowedSinc', **ifargs)
    xfm.terminal_output = 'allatonce'
    xfm.resource_monitor = False
    runtime = xfm.run().runtime
    LOGGER.info(runtime.cmdline)

    # Force floating point precision
    nii = nb.load(out_file, mmap=False)
    nii.set_data_dtype(np.dtype('float32'))
    nii.to_filename(out_file)

    # Get just the affine Transforms
    affines = [transform for transform in in_xform if '.nii' not in transform]
    out_affine = fname_presuffix(in_file, suffix='_affine_xform-%05d.mat' % index,
                                 newpath=newpath, use_ext=False)
    affine_file, affine_cmd = compose_affines(ifargs['reference_image'], affines, out_affine)

    return (out_file, runtime.cmdline, affine_file, affine_cmd)


def compose_affines(reference_image, affine_list, output_file):
    """Use antsApplyTransforms to get a single affine from multiple affines."""
    cmd = "antsApplyTransforms -d 3 -r %s -o Linear[%s] " % (
        reference_image, output_file)
    cmd += " ".join(["--transform %s" % trf for trf in affine_list])
    LOGGER.info(cmd)
    os.system(cmd)
    if not os.path.exists(output_file):
        LOGGER.critical(cmd)
        assert False
    return output_file, cmd


def create_tensor_image(mask_img, direction, prefix):
    """set intent as NIFTI_INTENT_SYMMATRIX (1005),
    [dxx, dxy, dyy, dxz, dyz, dzz] are the components
    info from here
    https://github.com/ANTsX/ANTs/wiki/Importing-diffusion-tensor-data-from-other-software
    """
    out_fname = prefix + "_tensor.nii"
    evecs = all_tensor_evecs(direction)
    evals = np.diag([1.0, 0.5, 0.05])
    tensor = np.linalg.multi_dot([evecs, evals, evecs.T])

    temp_components = []
    for direction in ['xx', 'xy', 'xz', 'yy', 'yz', 'zz']:
        this_component = prefix + '_temp_dtiComp_%s.nii.gz' % direction
        LOGGER.info("writing %s", this_component)
        nb.Nifti1Image(mask_img.get_data()*tensor[tensor_index[direction]], mask_img.affine,
                       mask_img.header).to_filename(this_component)
        temp_components.append(this_component)

    compose_cmd = 'ImageMath 3 %s ComponentTo3DTensor %s' % (
        out_fname, prefix + '_temp_dtiComp_')
    LOGGER.info(compose_cmd)
    os.system(compose_cmd)
    for temp_component in temp_components:
        os.remove(temp_component)

    return out_fname


def reorient_tensor_image(tensor_image, warp_file, mask_img, prefix, output_fname):
    cmds = []
    to_remove = []
    reoriented_tensor_fname = prefix + "reoriented_tensor.nii"
    reorient_cmd = "ReorientTensorImage 3 %s %s %s" % (tensor_image,
                                                       reoriented_tensor_fname,
                                                       warp_file)
    LOGGER.info(reorient_cmd)
    os.system(reorient_cmd)
    cmds.append(reorient_cmd)
    to_remove.append(reoriented_tensor_fname)

    # Load the reoriented tensor and get the principal directions out
    reoriented_dt_img = nb.load(reoriented_tensor_fname)
    reoriented_tensor_data = reoriented_dt_img.get_data().squeeze()

    mask_data = mask_img.get_data() > 0
    output_data = np.zeros(mask_img.shape + (3,))

    reoriented_tensors = reoriented_tensor_data[mask_data]
    reoriented_vectors = np.zeros((reoriented_tensors.shape[0], 3))

    def tensor_from_vec(vec):
        """[dxx, dxy, dyy, dxz, dyz, dzz]."""
        return np.array([
            [vec[0], vec[1], vec[3]],
            [vec[1], vec[2], vec[4]],
            [vec[3], vec[4], vec[5]]
        ])

    for nrow, row in enumerate(reoriented_tensors):
        row_tensor = tensor_from_vec(row)
        evals, evecs = decompose_tensor(row_tensor)
        reoriented_vectors[nrow] = evecs[:, 0]

    output_data[mask_data] = normalized_vector(reoriented_vectors)
    vector_data = get_vector_nii(output_data, mask_img.affine, mask_img.header)
    vector_data.to_filename(output_fname)
    os.remove(reoriented_tensor_fname)
    os.remove(tensor_image)
    return output_fname, reorient_cmd


def get_vector_nii(data, affine, header):
    hdr = header.copy()
    hdr.set_data_dtype(np.dtype('<f4'))
    hdr.set_intent('vector', (), '')
    return nb.Nifti1Image(data[:, :, :, np.newaxis, :].astype(
                          np.dtype('<f4')), affine, hdr)


def local_bvec_rotation(original_bvecs, warp_transforms, mask_image, runtime, output_fname):
    """Create a vector in each voxel that accounts for nonlinear warps."""
    prefix = os.path.join(runtime.cwd, "local_bvec_")
    mask_img = nb.load(mask_image)
    mask_data = mask_img.get_data()
    b0_image = get_vector_nii(np.stack([np.zeros_like(mask_data)] * 3, -1), mask_img.affine,
                              mask_img.header)
    commands = []
    rotated_vec_files = []
    for vecnum, (original_bvec, warp_file) in enumerate(zip(original_bvecs, warp_transforms)):
        num_prefix = prefix + "%03d_" % vecnum
        out_fname = num_prefix + "rotated.nii.gz"
        # if it's a b0, no rotation needed
        if np.sum(original_bvec**2) == 0:
            b0_image.to_filename(out_fname)
            rotated_vec_files.append(out_fname)
            commands.append("B0: No rotation")
            continue

        # otherwise, rotate it
        tensor_fname = create_tensor_image(mask_img, original_bvec, num_prefix)
        rotated_vector_fname, rotate_cmd = reorient_tensor_image(tensor_fname,
                                                                 warp_file,
                                                                 mask_img,
                                                                 num_prefix,
                                                                 out_fname)
        commands.append(rotate_cmd)
        rotated_vec_files.append(out_fname)
    concatenated = np.stack(
        [nb.load(img, mmap=False).get_data().astype("<f4") for img in rotated_vec_files], -1)
    nb.Nifti1Image(concatenated, mask_img.affine, mask_img.header).to_filename(output_fname)
    for temp_file in rotated_vec_files:
        os.remove(temp_file)
    return rotated_vec_files, commands
