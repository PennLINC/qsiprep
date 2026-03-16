#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Interfaces for using SynB0-DISCO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
import os
import os.path as op


def get_synb0_atlas(masked=True, res="low"):

    atlas_dir = os.getenv("SYNB0_ATLASES")
    if not atlas_dir:
        raise Exception("Unable to locate SynB0 atlases. Define a SYNB0_ATLASES variable.")

    res_str = "" if res == "high" else "_2_5"
    mask_str = "_mask" if masked else ""

    atlas_file = f"mni_icbm152_t1_tal_nlin_asym_09c{mask_str}{res_str}.nii.gz"
    return op.join(atlas_dir, atlas_file)
