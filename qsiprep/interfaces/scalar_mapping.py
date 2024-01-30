#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Workflows for AMICO
~~~~~~~~~~~~~~~~~~~


"""
import os
import os.path as op
from pkg_resources import resource_filename as pkgr
import nibabel as nb
import numpy as np
from nipype import logging
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec, File, SimpleInterface, isdefined
)


class ScalarMapperInputSpec(BaseInterfaceInputSpec):
    scalar_filter = traits.InputMultiObject(traits.Str())
    recon_scalars = traits.InputMultiObject(traits.Any())


class ScalarMapperOutputSpec(TraitedSpec):
    mapped_scalars = traits.List(File(exists=True))


class ScalarMapper(SimpleInterface):
    input_spec = ScalarMapperInputSpec
    output_spec = ScalarMapperOutputSpec


class _BundleMapperInputSpec(ScalarMapperInputSpec):
    bundles = traits.InputMultiObject(
        File(exists=True),
        desc="Paths to tck files")
    bundle_names = traits.InputMultiObject(traits.Str())


class _BundleMapperOutputSpec(ScalarMapper):
    bundle_summary = File(exists=True)
    bundle_profiles = File(exists=True)


class BundleMapper(ScalarMapper):
    input_spec = _BundleMapperInputSpec
    output_spec = _BundleMapperOutputSpec

