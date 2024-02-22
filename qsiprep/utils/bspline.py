#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import numpy as np
from nipype import logging

LOGGER = logging.getLogger("nipype.interfaces")


def get_ijk(data, offset=0):
    """
    Calculates voxel coordinates from data
    """
    from numpy import mgrid

    if not isinstance(offset, (list, tuple)):
        offset = [offset] * 3

    grid = mgrid[
        offset[0] : (offset[0] + data.shape[0]),
        offset[1] : (offset[1] + data.shape[1]),
        offset[2] : (offset[2] + data.shape[2]),
    ]
    return grid.reshape(3, -1).T


def compute_affine(data, zooms):
    """
    Compose a RAS affine mat, since the affine of the image might not be RAS
    """
    aff = np.eye(4) * (list(zooms) + [1])
    aff[:3, 3] -= aff[:3, :3].dot(np.array(data.shape[:3], dtype=float) - 1.0) * 0.5
    return aff


def tbspl_eval(points, knots, zooms, njobs=None):
    """
    Evaluate tensor product BSpline
    """
    raise Exception("Removed BSpline")


def _evalp(args):
    import numpy as np
    from scipy.sparse import csr_matrix

    point, knots, vbspl, zooms = args
    u_vec = (knots - point[np.newaxis, ...]) / zooms[np.newaxis, ...]
    c = vbspl(u_vec.reshape(-1)).reshape((knots.shape[0], 3)).prod(axis=1)
    return csr_matrix(c)
