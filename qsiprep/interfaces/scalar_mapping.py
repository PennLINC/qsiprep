#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import os
import os.path as op
from pkg_resources import resource_filename as pkgr
import nibabel as nb
import numpy as np
from nipype import logging
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec, File, SimpleInterface, isdefined,
    InputMultiObject
)


class ScalarMapperInputSpec(BaseInterfaceInputSpec):
    scalars_from = InputMultiObject(traits.Str())
    recon_scalars = InputMultiObject(traits.Any())
    dwiref_image = File(exists=True)


class ScalarMapperOutputSpec(TraitedSpec):
    mapped_scalars = traits.List(File(exists=True))


class ScalarMapper(SimpleInterface):
    input_spec = ScalarMapperInputSpec
    output_spec = ScalarMapperOutputSpec


# For mapping to bundles
class _BundleMapperInputSpec(ScalarMapperInputSpec):
    tck_files = InputMultiObject(
        File(exists=True),
        desc="Paths to tck files")
    bundle_names = InputMultiObject(traits.Str())


class _BundleMapperOutputSpec(ScalarMapperOutputSpec):
    bundle_summary = File(exists=True)
    bundle_profiles = File(exists=True)


class BundleMapper(ScalarMapper):
    input_spec = _BundleMapperInputSpec
    output_spec = _BundleMapperOutputSpec


# For mapping to atlases
class _AtlasMapperInputSpec(ScalarMapperInputSpec):
    atlas_configs = traits.Any()


class _AtlasMapperOutputSpec(ScalarMapperOutputSpec):
    region_stats = File(exists=True, mandatory=True)


class AtlasMapper(ScalarMapper):
    input_spec = _AtlasMapperInputSpec
    output_spec = _AtlasMapperOutputSpec
