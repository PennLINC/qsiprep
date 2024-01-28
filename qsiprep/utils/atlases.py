#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Loading atlases
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

"""
import os
import json


def get_atlases(atlas_names):
    """Build dictionary of atlases to use to parcellate QSIPrep outputs.

    Parameters
    ----------
    atlas_names : :obj:`list` of :obj:`str`
        List of atlas names to load.

    Returns
    -------
    outputs : :obj:`dict`
        Dictionary of atlases to use for parcellation.
        Keys are atlas names, values are dictionaries with the following keys:

        - ``file`` : :obj:`str`
            Path to atlas file.
        - ``node_names`` : :obj:`list` of :obj:`str`
            List of node names in atlas.
        - ``node_ids`` : :obj:`list` of :obj:`int`
            List of node IDs in atlas. Each element corresponds to an entry in ``node_names``.

    Notes
    -----
    The location of the atlas_config.json file is determined by the ``QSIRECON_ATLAS`` environment
    variable.
    """
    atlas_names = list(set(atlas_names))
    atlas_dir = os.getenv("QSIRECON_ATLAS")
    if atlas_dir is None:
        raise Exception("No environment variable found for QSIRECON_ATLAS")

    with open(os.path.join(atlas_dir, "atlas_config.json")) as f:
        atlas_config = json.load(f)

    outputs = {}
    for atlas_name in atlas_names:
        if atlas_name not in atlas_config:
            raise Exception("Atlas %s not found in atlas_config.json", atlas_name)

        atlas_data = atlas_config[atlas_name]
        atlas_data["file"] = os.path.join(atlas_dir, atlas_data["file"])
        outputs[atlas_name] = atlas_data

    return outputs
