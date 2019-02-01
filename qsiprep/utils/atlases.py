#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Loading atlases
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


"""
import os
import os.path as op
import json


def get_atlases(atlas_names):
    atlas_names = list(set(atlas_names))
    atlas_dir = os.getenv("QSIRECON_ATLAS")
    if atlas_dir is None:
        raise Exception("No environment variable found for QSIRECON_ATLAS")
    with open(op.join(atlas_dir, 'atlas_config.json')) as f:
        atlas_config = json.load(f)
    outputs = {}
    for atlas_name in atlas_names:
        if atlas_name not in atlas_config:
            raise Exception("Atlas %s not found in atlas_config.json" % atlas_name)
        atlas_data = atlas_config[atlas_name]
        atlas_data['file'] = op.join(atlas_dir, atlas_data['file'])
        outputs[atlas_name] = atlas_data
    return outputs
