"""
Functions copied from nipype

"""

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe

from ...interfaces.fmap import CleanupEdgeFilter, DespikeFilter


def siemens2rads(in_file, out_file=None):
    """
    Converts input phase difference map to rads
    """
    import math
    import os.path as op

    import nibabel as nb
    import numpy as np

    if out_file is None:
        fname, fext = op.splitext(op.basename(in_file))
        if fext == '.gz':
            fname, _ = op.splitext(fname)
        out_file = op.abspath(f'./{fname}_rads.nii.gz')

    in_file = np.atleast_1d(in_file).tolist()
    im = nb.load(in_file[0])
    data = im.get_fdata().astype(np.float32)
    hdr = im.header.copy()

    if len(in_file) == 2:
        data = nb.load(in_file[1]).get_fdata().astype(np.float32) - data
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
    import os.path as op

    import nibabel as nb
    import numpy as np

    if out_file is None:
        fname, fext = op.splitext(op.basename(in_file))
        if fext == '.gz':
            fname, _ = op.splitext(fname)
        out_file = op.abspath(f'./{fname}_demean.nii.gz')

    im = nb.load(in_file)
    data = im.get_fdata().astype(np.float32)
    msk = np.ones_like(data)

    if in_mask is not None:
        msk = nb.load(in_mask).get_fdata().astype(np.float32)
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
    inputnode = pe.Node(niu.IdentityInterface(fields=['in_file', 'in_mask']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out_file']), name='outputnode')

    # Despiking and the erode/subtract/mask/add edge chain are both implemented in
    # nibabel/numpy, so this pipeline needs no external tools.
    despike = pe.Node(DespikeFilter(threshold=2.1), name='Despike')
    edge_cleanup = pe.Node(CleanupEdgeFilter(), name='EdgeCleanup')

    wf = pe.Workflow(name=name)
    wf.connect([
        (inputnode, despike, [
            ('in_file', 'in_file'),
            ('in_mask', 'in_mask')]),
        (inputnode, edge_cleanup, [
            ('in_file', 'in_file'),
            ('in_mask', 'in_mask')]),
        (despike, edge_cleanup, [('out_file', 'despiked_file')]),
        (edge_cleanup, outputnode, [('out_file', 'out_file')])
    ])  # fmt:skip
    return wf
