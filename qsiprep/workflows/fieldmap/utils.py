"""
Functions copied from nipype

"""

from nipype.interfaces import fsl, utility as niu
from nipype.pipeline import engine as pe


def siemens2rads(in_file, out_file=None):
    """
    Converts input phase difference map to rads
    """
    import numpy as np
    import nibabel as nb
    import os.path as op
    import math

    if out_file is None:
        fname, fext = op.splitext(op.basename(in_file))
        if fext == '.gz':
            fname, _ = op.splitext(fname)
        out_file = op.abspath('./%s_rads.nii.gz' % fname)

    in_file = np.atleast_1d(in_file).tolist()
    im = nb.load(in_file[0])
    data = im.get_data().astype(np.float32)
    hdr = im.header.copy()

    if len(in_file) == 2:
        data = nb.load(in_file[1]).get_data().astype(np.float32) - data
    elif (data.ndim == 4) and (data.shape[-1] == 2):
        data = np.squeeze(data[..., 1] - data[..., 0])
        hdr.set_data_shape(data.shape[:3])

    imin = data.min()
    imax = data.max()
    data = (2.0 * math.pi * (data - imin) / (imax - imin)) - math.pi
    hdr.set_data_dtype(np.float32)
    hdr.set_xyzt_units('mm')
    hdr['datatype'] = 16
    nb.Nifti1Image(data, im.affine, hdr).to_filename(out_file)
    return out_file


def demean_image(in_file, in_mask=None, out_file=None):
    """
    Demean image data inside mask
    """
    import numpy as np
    import nibabel as nb
    import os.path as op

    if out_file is None:
        fname, fext = op.splitext(op.basename(in_file))
        if fext == '.gz':
            fname, _ = op.splitext(fname)
        out_file = op.abspath('./%s_demean.nii.gz' % fname)

    im = nb.load(in_file)
    data = im.get_data().astype(np.float32)
    msk = np.ones_like(data)

    if in_mask is not None:
        msk = nb.load(in_mask).get_data().astype(np.float32)
        msk[msk > 0] = 1.0
        msk[msk < 1] = 0.0

    mean = np.median(data[msk == 1].reshape(-1))
    data[msk == 1] = data[msk == 1] - mean
    nb.Nifti1Image(data, im.affine, im.header).to_filename(out_file)
    return out_file


def cleanup_edge_pipeline(name='Cleanup'):
    """
    Perform some de-spiking filtering to clean up the edge of the fieldmap
    (copied from fsl_prepare_fieldmap)
    """
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['in_file', 'in_mask']), name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['out_file']), name='outputnode')

    fugue = pe.Node(
        fsl.FUGUE(
            save_fmap=True, despike_2dfilter=True, despike_threshold=2.1),
        name='Despike')
    erode = pe.Node(
        fsl.maths.MathsCommand(nan2zeros=True, args='-kernel 2D -ero'),
        name='MskErode')
    newmsk = pe.Node(
        fsl.MultiImageMaths(op_string='-sub %s -thr 0.5 -bin'), name='NewMask')
    applymsk = pe.Node(fsl.ApplyMask(nan2zeros=True), name='ApplyMask')
    join = pe.Node(niu.Merge(2), name='Merge')
    addedge = pe.Node(
        fsl.MultiImageMaths(op_string='-mas %s -add %s'), name='AddEdge')

    wf = pe.Workflow(name=name)
    wf.connect([(inputnode, fugue, [
        ('in_file', 'fmap_in_file'), ('in_mask', 'mask_file')
    ]), (inputnode, erode, [('in_mask', 'in_file')]), (inputnode, newmsk, [
        ('in_mask', 'in_file')
    ]), (erode, newmsk, [('out_file', 'operand_files')]), (fugue, applymsk, [
        ('fmap_out_file', 'in_file')
    ]), (newmsk, applymsk,
         [('out_file', 'mask_file')]), (erode, join, [('out_file', 'in1')]),
                (applymsk, join, [('out_file', 'in2')]), (inputnode, addedge, [
                    ('in_file', 'in_file')
                ]), (join, addedge, [('out', 'operand_files')]),
                (addedge, outputnode, [('out_file', 'out_file')])])
    return wf
