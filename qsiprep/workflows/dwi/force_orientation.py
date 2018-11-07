


def force_orientation(input_image, new_axcodes, bvecs=None):
    import nibabel as nib
    import os.path as op
    import numpy as np
    suffix = ''.join(new_axcodes)
    output_nii = "reoriented_%s.nii.gz" % suffix
    output_bvec = "reoriented_%s.bvec" % suffix
    input_img = nib.load(input_image)
    input_axcodes = nib.aff2axcodes(input_img.affine)
    # Is the input image oriented how we want?
    if not input_axcodes == new_axcodes:
        # Re-orient
        input_orientation = nib.orientations.axcodes2ornt(input_axcodes)
        desired_orientation = nib.orientations.axcodes2ornt(new_axcodes)
        transform_orientation = nib.orientations.ornt_transform(
                    input_orientation,desired_orientation)
        reoriented_img = input_img.as_reoriented(transform_orientation)
        reoriented_img.to_filename(output_nii)
        reoriented_bvec = None
        if bvecs is not None:

            bvec_array = np.loadtxt(bvecs)
            if not bvec_array.shape[0] == transform_orientation.shape[0]:
                raise ValueError("Unrecognized bvec format")
            output_array = np.zeros_like(bvec_array)
            for this_axnum, (axnum, flip) in enumerate(transform_orientation):
                output_array[this_axnum] = bvec_array[int(axnum)] * flip
            np.savetxt(output_bvec, output_array, fmt="%.8f ",)
            return op.abspath(output_nii), op.abspath(output_bvec)
        return op.abspath(output_nii), None

    else:
        return input_image, bvecs
