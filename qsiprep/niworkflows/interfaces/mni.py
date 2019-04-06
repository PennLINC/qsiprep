# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" A robust ANTs T1-to-MNI registration workflow with fallback retry """

from __future__ import print_function, division, absolute_import, unicode_literals
from os import path as op

import pkg_resources as pkgr
from multiprocessing import cpu_count
from packaging.version import Version
import nibabel as nb
import numpy as np

from nipype.interfaces.ants.registration import RegistrationOutputSpec
from nipype.interfaces.ants import AffineInitializer
from nipype.interfaces.base import (
    traits, isdefined, BaseInterface, BaseInterfaceInputSpec, File)

from ..data import getters
from .. import NIWORKFLOWS_LOG, __version__
from .fixes import (
    FixHeaderApplyTransforms as ApplyTransforms,
    FixHeaderRegistration as Registration
)


niworkflows_version = Version(__version__)


class RobustMNINormalizationInputSpec(BaseInterfaceInputSpec):
    """
    Set inputs to RobustMNINormalization
    """
    # Enable deprecation
    package_version = niworkflows_version

    # Moving image.
    moving_image = File(exists=True, mandatory=True, desc='image to apply transformation to')
    # Reference image (optional).
    reference_image = File(exists=True, desc='override the reference image')
    # Moving mask (optional).
    moving_mask = File(exists=True, desc='moving image mask')
    # Reference mask (optional).
    reference_mask = File(exists=True, desc='reference image mask')
    # Lesion mask (optional).
    lesion_mask = File(exists=True, desc='lesion mask image')
    # Number of threads to use for ANTs/ITK processes.
    num_threads = traits.Int(cpu_count(), usedefault=True, nohash=True,
                             desc="Number of ITK threads to use")
    # ANTs parameter set to use.
    flavor = traits.Enum('precise', 'testing', 'fast', usedefault=True,
                         desc='registration settings parameter set')
    # Template orientation.
    orientation = traits.Enum('LPS', mandatory=True, usedefault=True,
                              desc='modify template orientation (should match input image)')
    # Modality of the reference image.
    reference = traits.Enum('T1w', 'T2w', 'boldref', 'PDw', mandatory=True, usedefault=True,
                            desc='set the reference modality for registration')
    # T1 or EPI registration?
    moving = traits.Enum('T1w', 'bold', usedefault=True, mandatory=True,
                         desc='registration type')
    # Template to use as the default reference image.
    template = traits.Enum(
        'MNI152NLin2009cAsym',
        'OASIS',
        'NKI',
        'mni_icbm152_linear',
        usedefault=True, desc='define the template to be used')
    # Load other settings from file.
    settings = traits.List(File(exists=True), desc='pass on the list of settings files')
    # Resolution of the default template.
    template_resolution = traits.Enum(1, 2, mandatory=True, usedefault=True,
                                      desc='template resolution')
    # Use explicit masking?
    explicit_masking = traits.Bool(True, usedefault=True,
                                   desc="""\
Set voxels outside the masks to zero thus creating an artificial border
that can drive the registration. Requires reliable and accurate masks.
See https://sourceforge.net/p/advants/discussion/840261/thread/27216e69/#c7ba\
""")
    initial_moving_transform = File(exists=True, desc='transform for initialization')
    float = traits.Bool(False, usedefault=True, desc='use single precision calculations')


class RobustMNINormalization(BaseInterface):
    """
    An interface to robustly run T1-to-MNI spatial normalization.
    Several settings are sequentially tried until some work.
    """
    input_spec = RobustMNINormalizationInputSpec
    output_spec = RegistrationOutputSpec

    def _list_outputs(self):
        return self._results

    def __init__(self, **inputs):
        self.norm = None
        self.retry = 1
        self._results = {}
        self.terminal_output = 'file'
        super(RobustMNINormalization, self).__init__(**inputs)

    def _get_settings(self):
        """
        Return any settings defined by the user, as well as any pre-defined
        settings files that exist for the image modalities to be registered.
        """
        # If user-defined settings exist...
        if isdefined(self.inputs.settings):
            # Note this in the log and return those settings.
            NIWORKFLOWS_LOG.info('User-defined settings, overriding defaults')
            return self.inputs.settings

        # Define a prefix for output files based on the modality of the moving image.
        filestart = '{}-mni_registration_{}_'.format(
            self.inputs.moving.lower(), self.inputs.flavor)

        # Get a list of settings files that match the flavor.
        filenames = [i for i in pkgr.resource_listdir('qsiprep.niworkflows', 'data')
                     if i.startswith(filestart) and i.endswith('.json')]
        # Return the settings files.
        return [pkgr.resource_filename('qsiprep.niworkflows.data', f)
                for f in sorted(filenames)]

    def _run_interface(self, runtime):
        # Get a list of settings files.
        settings_files = self._get_settings()
        ants_args = self._get_ants_args()
        if not isdefined(self.inputs.initial_moving_transform):
            NIWORKFLOWS_LOG.info('Estimating initial transform using AffineInitializer')
            init = AffineInitializer(
                fixed_image=ants_args['fixed_image'],
                moving_image=ants_args['moving_image'],
                num_threads=self.inputs.num_threads)
            init.resource_monitor = False
            init.terminal_output = 'allatonce'
            init_result = init.run()
            # Save outputs (if available)
            init_out = _write_outputs(init_result.runtime, '.nipype-init')
            if init_out:
                NIWORKFLOWS_LOG.info(
                    'Terminal outputs of initialization saved (%s).',
                    ', '.join(init_out))

            ants_args['initial_moving_transform'] = init_result.outputs.out_file

        # For each settings file...
        for ants_settings in settings_files:

            NIWORKFLOWS_LOG.info('Loading settings from file %s.',
                                 ants_settings)
            # Configure an ANTs run based on these settings.
            self.norm = Registration(from_file=ants_settings,
                                     **ants_args)
            self.norm.resource_monitor = False
            self.norm.terminal_output = self.terminal_output

            # Print the retry number and command line call to the log.
            NIWORKFLOWS_LOG.info(
                'Retry #%d, commandline: \n%s', self.retry, self.norm.cmdline)
            self.norm.ignore_exception = True
            # Try running registration.
            interface_result = self.norm.run()

            if interface_result.runtime.returncode != 0:
                NIWORKFLOWS_LOG.warning('Retry #%d failed.', self.retry)
                # Save outputs (if available)
                term_out = _write_outputs(interface_result.runtime,
                                          '.nipype-%04d' % self.retry)
                if term_out:
                    NIWORKFLOWS_LOG.warning(
                        'Log of failed retry saved (%s).', ', '.join(term_out))
            else:
                runtime.returncode = 0
                # Grab the outputs.
                self._results.update(interface_result.outputs.get())
                if isdefined(self.inputs.moving_mask):
                    self._validate_results()

                # Note this in the log.
                NIWORKFLOWS_LOG.info(
                    'Successful spatial normalization (retry #%d).', self.retry)
                # Break out of the retry loop.
                return runtime

            self.retry += 1

        # If all tries fail, raise an error.
        raise RuntimeError(
            'Robust spatial normalization failed after %d retries.' % (self.retry - 1))

    def _get_ants_args(self):
        args = {'moving_image': self.inputs.moving_image,
                'num_threads': self.inputs.num_threads,
                'float': self.inputs.float,
                'terminal_output': 'file',
                'write_composite_transform': True,
                'initial_moving_transform': self.inputs.initial_moving_transform}

        """
        Moving image handling - The following truth table maps out the intended action
        sequence. Future refactoring may more directly encode this.

        moving_mask and lesion_mask are files
        True = file
        False = None

        | moving_mask | explicit_masking | lesion_mask | action
        |-------------|------------------|-------------|-------------------------------------------
        | True        | True             | True        | Update `moving_image` after applying
        |             |                  |             | mask.
        |             |                  |             | Set `moving_image_masks` applying
        |             |                  |             | `create_cfm` with `global_mask=True`.
        |-------------|------------------|-------------|-------------------------------------------
        | True        | True             | False       | Update `moving_image` after applying
        |             |                  |             | mask.
        |-------------|------------------|-------------|-------------------------------------------
        | True        | False            | True        | Set `moving_image_masks` applying
        |             |                  |             | `create_cfm` with `global_mask=False`
        |-------------|------------------|-------------|-------------------------------------------
        | True        | False            | False       | args['moving_image_masks'] = moving_mask
        |-------------|------------------|-------------|-------------------------------------------
        | False       | *                | True        | Set `moving_image_masks` applying
        |             |                  |             | `create_cfm` with `global_mask=True`
        |-------------|------------------|-------------|-------------------------------------------
        | False       | *                | False       | No action
        """
        # If a moving mask is provided...
        if isdefined(self.inputs.moving_mask):
            # If explicit masking is enabled...
            if self.inputs.explicit_masking:
                # Mask the moving image.
                # Do not use a moving mask during registration.
                args['moving_image'] = mask(
                    self.inputs.moving_image,
                    self.inputs.moving_mask,
                    "moving_masked.nii.gz")

            # If explicit masking is disabled...
            else:
                # Use the moving mask during registration.
                # Do not mask the moving image.
                args['moving_image_masks'] = self.inputs.moving_mask

            # If a lesion mask is also provided...
            if isdefined(self.inputs.lesion_mask):
                # Create a cost function mask with the form:
                # [global mask - lesion mask] (if explicit masking is enabled)
                # [moving mask - lesion mask] (if explicit masking is disabled)
                # Use this as the moving mask.
                args['moving_image_masks'] = create_cfm(
                    self.inputs.moving_mask,
                    lesion_mask=self.inputs.lesion_mask,
                    global_mask=self.inputs.explicit_masking)

        # If no moving mask is provided...
        # But a lesion mask *IS* provided...
        elif isdefined(self.inputs.lesion_mask):
            # Create a cost function mask with the form: [global mask - lesion mask]
            # Use this as the moving mask.
            args['moving_image_masks'] = create_cfm(
                self.inputs.moving_image,
                lesion_mask=self.inputs.lesion_mask,
                global_mask=True)

        """
        Reference image handling - The following truth table maps out the intended action
        sequence. Future refactoring may more directly encode this.

        reference_mask and lesion_mask are files
        True = file
        False = None

        | reference_mask | explicit_masking | lesion_mask | action
        |----------------|------------------|-------------|----------------------------------------
        | True           | True             | True        | Update `fixed_image` after applying
        |                |                  |             | mask.
        |                |                  |             | Set `fixed_image_masks` applying
        |                |                  |             | `create_cfm` with `global_mask=True`.
        |----------------|------------------|-------------|----------------------------------------
        | True           | True             | False       | Update `fixed_image` after applying
        |                |                  |             | mask.
        |----------------|------------------|-------------|----------------------------------------
        | True           | False            | True        | Set `fixed_image_masks` applying
        |                |                  |             | `create_cfm` with `global_mask=False`
        |----------------|------------------|-------------|----------------------------------------
        | True           | False            | False       | args['fixed_image_masks'] = fixed_mask
        |----------------|------------------|-------------|----------------------------------------
        | False          | *                | True        | Set `fixed_image_masks` applying
        |                |                  |             | `create_cfm` with `global_mask=True`
        |----------------|------------------|-------------|----------------------------------------
        | False          | *                | False       | No action
        """
        # If a reference image is provided...
        if isdefined(self.inputs.reference_image):
            # Use the reference image as the fixed image.
            args['fixed_image'] = self.inputs.reference_image

            # If a reference mask is provided...
            if isdefined(self.inputs.reference_mask):
                # If explicit masking is enabled...
                if self.inputs.explicit_masking:
                    # Mask the reference image.
                    # Do not use a fixed mask during registration.
                    args['fixed_image'] = mask(
                        self.inputs.reference_image,
                        self.inputs.reference_mask,
                        "fixed_masked.nii.gz")

                    # If a lesion mask is also provided...
                    if isdefined(self.inputs.lesion_mask):
                        # Create a cost function mask with the form: [global mask]
                        # Use this as the fixed mask.
                        args['fixed_image_masks'] = create_cfm(
                            self.inputs.reference_mask,
                            lesion_mask=None,
                            global_mask=True)

                # If a reference mask is provided...
                # But explicit masking is disabled...
                else:
                    # Use the reference mask as the fixed mask during registration.
                    # Do not mask the fixed image.
                    args['fixed_image_masks'] = self.inputs.reference_mask

            # If no reference mask is provided...
            # But a lesion mask *IS* provided ...
            elif isdefined(self.inputs.lesion_mask):
                # Create a cost function mask with the form: [global mask]
                # Use this as the fixed mask
                args['fixed_image_masks'] = create_cfm(
                    self.inputs.reference_image,
                    lesion_mask=None,
                    global_mask=True)

        # If no reference image is provided, fall back to the default template.
        else:
            # Raise an error if the user specifies an unsupported image orientation.
            if self.inputs.orientation == 'LAS':
                raise NotImplementedError

            # Get the template specified by the user.
            template = getters.get_template(self.inputs.template)
            # Set the template resolution.
            resolution = self.inputs.template_resolution
            _tpl_fmt = 'tpl-{}*_res-%02d_%s.nii.gz'.format(self.inputs.template)

            # Find actual files
            ref_template = str(
                list(template.glob(_tpl_fmt % (resolution, self.inputs.reference)))[0])
            ref_mask = str(
                list(template.glob(_tpl_fmt % (resolution, 'brainmask')))[0])

            # Default is explicit masking disabled
            args['fixed_image'] = ref_template
            # Use the template mask as the fixed mask.
            args['fixed_image_masks'] = ref_mask

            # Overwrite defaults if explicit masking
            if self.inputs.explicit_masking:
                # Mask the template image with the template mask.
                args['fixed_image'] = mask(ref_template, ref_mask,
                                           "fixed_masked.nii.gz")
                # Do not use a fixed mask during registration.
                args.pop('fixed_image_masks', None)

                # If a lesion mask is provided...
                if isdefined(self.inputs.lesion_mask):
                    # Create a cost function mask with the form: [global mask]
                    # Use this as the fixed mask.
                    args['fixed_image_masks'] = create_cfm(
                        ref_mask, lesion_mask=None, global_mask=True)

        return args

    def _validate_results(self):
        forward_transform = self._results['composite_transform']
        input_mask = self.inputs.moving_mask
        if isdefined(self.inputs.reference_mask):
            target_mask = self.inputs.reference_mask
        else:
            template = getters.get_template(self.inputs.template)
            resolution = self.inputs.template_resolution
            target_mask = str(
                list(template.glob('tpl-%s*_res-%02d_brainmask.nii.gz' % (
                     self.inputs.template, resolution)))[0])

        res = ApplyTransforms(dimension=3,
                              input_image=input_mask,
                              reference_image=target_mask,
                              transforms=forward_transform,
                              interpolation='NearestNeighbor',
                              resource_monitor=False).run()
        input_mask_data = (nb.load(res.outputs.output_image).get_data() != 0)
        target_mask_data = (nb.load(target_mask).get_data() != 0)

        overlap_voxel_count = np.logical_and(input_mask_data, target_mask_data)

        overlap_perc = float(overlap_voxel_count.sum()) / float(input_mask_data.sum()) * 100

        assert overlap_perc > 50, \
            "Normalization failed: only %d%% of the normalized moving image " \
            "mask overlaps with the reference image mask." % overlap_perc


def mask(in_file, mask_file, new_name):
    """
    Apply a binary mask to an image.

    Parameters
    ----------
    in_file : str
        Path to a NIfTI file to mask
    mask_file : str
        Path to a binary mask
    new_name : str
        Path/filename for the masked output image.

    Returns
    -------
    str
        Absolute path of the masked output image.

    Notes
    -----
    in_file and mask_file must be in the same
    image space and have the same dimensions.
    """
    import nibabel as nb
    import os
    # Load the input image
    in_nii = nb.load(in_file)
    # Load the mask image
    mask_nii = nb.load(mask_file)
    # Set all non-mask voxels in the input file to zero.
    data = in_nii.get_data()
    data[mask_nii.get_data() == 0] = 0
    # Save the new masked image.
    new_nii = nb.Nifti1Image(data, in_nii.affine, in_nii.header)
    new_nii.to_filename(new_name)
    return os.path.abspath(new_name)


def create_cfm(in_file, lesion_mask=None, global_mask=True, out_path=None):
    """
    Create a mask to constrain registration.

    Parameters
    ----------
    in_file : str
        Path to an existing image (usually a mask).
        If global_mask = True, this is used as a size/dimension reference.
    out_path : str
        Path/filename for the new cost function mask.
    lesion_mask : str, optional
        Path to an existing binary lesion mask.
    global_mask : bool
        Create a whole-image mask (True) or limit to reference mask (False)
        A whole image-mask is 1 everywhere

    Returns
    -------
    str
        Absolute path of the new cost function mask.

    Notes
    -----
    in_file and lesion_mask must be in the same
    image space and have the same dimensions
    """
    import os
    import numpy as np
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix

    if out_path is None:
        out_path = fname_presuffix(in_file, suffix='_cfm', newpath=os.getcwd())
    else:
        out_path = os.path.abspath(out_path)

    if not global_mask and not lesion_mask:
        NIWORKFLOWS_LOG.warning(
            'No lesion mask was provided and global_mask not requested, '
            'therefore the original mask will not be modified.')

    # Load the input image
    in_img = nb.load(in_file)

    # If we want a global mask, create one based on the input image.
    data = np.ones(in_img.shape, dtype=np.uint8) if global_mask else in_img.get_data()
    if set(np.unique(data)) - {0, 1}:
        raise ValueError("`global_mask` must be true if `in_file` is not a binary mask")

    # If a lesion mask was provided, combine it with the secondary mask.
    if lesion_mask is not None:
        # Reorient the lesion mask and get the data.
        lm_img = nb.as_closest_canonical(nb.load(lesion_mask))

        # Subtract lesion mask from secondary mask, set negatives to 0
        data = np.fmax(data - lm_img.get_data(), 0)
        # Cost function mask will be created from subtraction
    # Otherwise, CFM will be created from global mask

    cfm_img = nb.Nifti1Image(data, in_img.affine, in_img.header)

    # Save the cost function mask.
    cfm_img.set_data_dtype(np.uint8)
    cfm_img.to_filename(out_path)

    return out_path


def _write_outputs(runtime, out_fname=None):
    if out_fname is None:
        out_fname = '.nipype'

    out_files = []
    for name in ['stdout', 'stderr', 'merged']:
        stream = getattr(runtime, name, '')
        if stream:
            out_file = op.join(runtime.cwd, name + out_fname)
            with open(out_file, 'w') as outf:
                print(stream, file=outf)
            out_files.append(out_file)
    return out_files
