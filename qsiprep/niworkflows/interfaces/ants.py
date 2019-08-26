#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Nipype interfaces for ANTs commands
"""

import os
from nipype.interfaces import base
from nipype.interfaces.ants.base import ANTSCommandInputSpec, ANTSCommand
from nipype.interfaces.base import traits, isdefined


class ImageMathInputSpec(ANTSCommandInputSpec):
    dimension = traits.Int(3, usedefault=True, position=1, argstr='%d',
                           desc='dimension of output image')
    output_image = base.File(position=2, argstr='%s', name_source=['op1'],
                             name_template='%s_maths', desc='output image file',
                             keep_extension=True)
    operation = base.Str(mandatory=True, position=3, argstr='%s',
                         desc='operations and intputs')
    op1 = base.File(exists=True, mandatory=True, position=-2, argstr='%s',
                    desc='first operator')
    op2 = traits.Either(base.File(exists=True), base.Str, position=-1,
                        argstr='%s', desc='second operator')


class ImageMathOuputSpec(base.TraitedSpec):
    output_image = base.File(exists=True, desc='output image file')


class ImageMath(ANTSCommand):
    """
    Operations over images

    Example:
    --------

    """

    _cmd = 'ImageMath'
    input_spec = ImageMathInputSpec
    output_spec = ImageMathOuputSpec


class ResampleImageBySpacingInputSpec(ANTSCommandInputSpec):
    dimension = traits.Int(3, usedefault=True, position=1, argstr='%d',
                           desc='dimension of output image')
    input_image = base.File(exists=True, mandatory=True, position=2, argstr='%s',
                            desc='input image file')
    output_image = base.File(position=3, argstr='%s', name_source=['input_image'],
                             name_template='%s_resampled', desc='output image file',
                             keep_extension=True)
    out_spacing = traits.Either(
        traits.List(traits.Float, minlen=2, maxlen=3),
        traits.Tuple(traits.Float, traits.Float, traits.Float),
        traits.Tuple(traits.Float, traits.Float),
        position=4, argstr='%s', mandatory=True, desc='output spacing'
    )
    apply_smoothing = traits.Bool(False, argstr='%d', position=5,
                                  desc='smooth before resampling')
    addvox = traits.Int(argstr='%d', position=6, requires=['apply_smoothing'],
                        desc='addvox pads each dimension by addvox')
    nn_interp = traits.Bool(argstr='%d', desc='nn interpolation',
                            position=-1, requires=['addvox'])


class ResampleImageBySpacingOutputSpec(base.TraitedSpec):
    output_image = traits.File(exists=True, desc='resampled file')


class ResampleImageBySpacing(ANTSCommand):
    """
    Resamples an image with a given spacing


    """
    _cmd = 'ResampleImageBySpacing'
    input_spec = ResampleImageBySpacingInputSpec
    output_spec = ResampleImageBySpacingOutputSpec

    def _format_arg(self, name, trait_spec, value):
        if name == 'out_spacing':
            if len(value) != self.inputs.dimension:
                raise ValueError('out_spacing dimensions should match dimension')

            value = ' '.join(['%d' % d for d in value])

        return super(ResampleImageBySpacing, self)._format_arg(
            name, trait_spec, value)


class ThresholdImageInputSpec(ANTSCommandInputSpec):
    dimension = traits.Int(3, usedefault=True, position=1, argstr='%d',
                           desc='dimension of output image')
    input_image = base.File(exists=True, mandatory=True, position=2, argstr='%s',
                            desc='input image file')
    output_image = base.File(position=3, argstr='%s', name_source=['input_image'],
                             name_template='%s_resampled', desc='output image file',
                             keep_extension=True)

    mode = traits.Enum('Otsu', 'Kmeans', argstr='%s', position=4,
                       requires=['num_thresholds'], xor=['th_low', 'th_high'],
                       desc='whether to run Otsu / Kmeans thresholding')
    num_thresholds = traits.Int(position=5, argstr='%d',
                                desc='number of thresholds')
    input_mask = base.File(exists=True, requires=['num_thresholds'], argstr='%s',
                           desc='input mask for Otsu, Kmeans')

    th_low = traits.Float(position=4, argstr='%f', xor=['mode'],
                          desc='lower threshold')
    th_high = traits.Float(position=5, argstr='%f', xor=['mode'],
                           desc='upper threshold')
    inside_value = traits.Float(1, position=6, argstr='%f', requires=['th_low'],
                                desc='inside value')
    outside_value = traits.Float(0, position=7, argstr='%f', requires=['th_low'],
                                 desc='outside value')


class ThresholdImageOutputSpec(base.TraitedSpec):
    output_image = traits.File(exists=True, desc='resampled file')


class ThresholdImage(ANTSCommand):
    """
    Apply thresholds on images


    """
    _cmd = 'ThresholdImage'
    input_spec = ThresholdImageInputSpec
    output_spec = ThresholdImageOutputSpec


class AIInputSpec(ANTSCommandInputSpec):
    dimension = traits.Int(3, usedefault=True, argstr='-d %d',
                           desc='dimension of output image')
    verbose = traits.Bool(False, usedefault=True, argstr='-v %d',
                          desc='enable verbosity')

    fixed_image = traits.File(
        exists=True, mandatory=True,
        desc='Image to which the moving_image should be transformed')
    moving_image = traits.File(
        exists=True, mandatory=True,
        desc='Image that will be transformed to fixed_image')

    fixed_image_mask = traits.File(
        exists=True, argstr='-x %s', desc='fixed mage mask')
    moving_image_mask = traits.File(
        exists=True, requires=['fixed_image_mask'],
        desc='moving mage mask')

    metric_trait = (
        traits.Enum("Mattes", "GC", "MI"),
        traits.Int(32),
        traits.Enum('Regular', 'Random', 'None'),
        traits.Range(value=0.2, low=0.0, high=1.0)
    )
    metric = traits.Tuple(*metric_trait, argstr='-m %s', mandatory=True,
                          desc='the metric(s) to use.')

    transform = traits.Tuple(
        traits.Enum('Affine', 'Rigid', 'Similarity'),
        traits.Range(value=0.1, low=0.0, exclude_low=True),
        argstr='-t %s[%f]', usedefault=True,
        desc='Several transform options are available')

    principal_axes = traits.Bool(False, usedefault=True, argstr='-p %d', xor=['blobs'],
                                 desc='align using principal axes')
    search_factor = traits.Tuple(
        traits.Float(20), traits.Range(value=0.12, low=0.0, high=1.0),
        usedefault=True, argstr='-s [%f,%f]', desc='search factor')

    search_grid = traits.Either(
        traits.Tuple(traits.Float, traits.Tuple(traits.Float, traits.Float, traits.Float)),
        traits.Tuple(traits.Float, traits.Tuple(traits.Float, traits.Float)),
        argstr='-g %s', desc='Translation search grid in mm')

    convergence = traits.Tuple(
        traits.Range(low=1, high=10000, value=10),
        traits.Float(1e-6),
        traits.Range(low=1, high=100, value=10),
        usedefault=True, argstr='-c [%d,%f,%d]', desc='convergence')

    output_transform = traits.File(
        'initialization.mat', usedefault=True, argstr='-o %s',
        desc='output file name')


class AIOuputSpec(base.TraitedSpec):
    output_transform = traits.File(exists=True, desc='output file name')


class AI(ANTSCommand):
    """
    The replacement for ``AffineInitializer``.

    Example:
    --------

    """

    _cmd = 'antsAI'
    input_spec = AIInputSpec
    output_spec = AIOuputSpec

    def _run_interface(self, runtime, correct_return_codes=(0, )):
        runtime = super(AI, self)._run_interface(
            runtime, correct_return_codes)

        setattr(self, '_output', {
            'output_transform': os.path.join(
                runtime.cwd,
                os.path.basename(self.inputs.output_transform))
        })
        return runtime

    def _format_arg(self, opt, spec, val):
        if opt == 'metric':
            val = '%s[{fixed_image},{moving_image},%d,%s,%f]' % val
            val = val.format(
                fixed_image=self.inputs.fixed_image,
                moving_image=self.inputs.moving_image)
            return spec.argstr % val

        if opt == 'search_grid':
            val1 = 'x'.join(['%f' % v for v in val[1]])
            fmtval = '[%s]' % ','.join([str(val[0]), val1])
            return spec.argstr % fmtval

        if opt == 'fixed_image_mask':
            if isdefined(self.inputs.moving_image_mask):
                return spec.argstr % ('[%s,%s]' % (
                    val, self.inputs.moving_image_mask))

        return super(AI, self)._format_arg(opt, spec, val)

    def _list_outputs(self):
        return getattr(self, '_output')
