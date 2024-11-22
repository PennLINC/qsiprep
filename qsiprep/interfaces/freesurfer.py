# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
FreeSurfer tools interfaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fetch some example data:

    >>> import os
    >>> from niworkflows import data
    >>> data_root = data.get_bids_examples(variant='BIDS-examples-1-enh-ds054')
    >>> os.chdir(data_root)

Disable warnings:

    >>> from nipype import logging
    >>> logging.getLogger('nipype.interface').setLevel('ERROR')

"""

import os.path as op

import nibabel as nb
from nipype.interfaces import freesurfer as fs
from nipype.interfaces.afni import Zeropad
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.interfaces.freesurfer.base import FSCommandOpenMP, FSTraitedSpec
from nipype.utils.filemanip import fname_presuffix
from niworkflows.utils.images import _copyxform
from scipy import ndimage


class FSTraitedSpecOpenMP(FSTraitedSpec):
    num_threads = traits.Int(desc='allows for specifying more threads', nohash=True)


class StructuralReference(fs.RobustTemplate):
    """Variation on RobustTemplate that simply copies the source if a single
    volume is provided.

    >>> from qsiprep.utils.bids import collect_data
    >>> t1w = collect_data('ds114', '01')[0]['t1w']
    >>> template = StructuralReference()
    >>> template.inputs.in_files = t1w
    >>> template.inputs.auto_detect_sensitivity = True
    >>> template.cmdline  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    'mri_robust_template --satit --mov .../sub-01_ses-retest_T1w.nii.gz
        .../sub-01_ses-test_T1w.nii.gz --template mri_robust_template_out.mgz'

    """

    def _num_vols(self):
        n_files = len(self.inputs.in_files)
        if n_files != 1:
            return n_files

        img = nb.load(self.inputs.in_files[0])
        if len(img.shape) == 3:
            return 1

        return img.shape[3]

    @property
    def cmdline(self):
        if self._num_vols() == 1:
            return 'echo Only one time point!'
        return super().cmdline

    def _list_outputs(self):
        outputs = super()._list_outputs()
        if self._num_vols() == 1:
            in_file = self.inputs.in_files[0]
            outputs['out_file'] = in_file
            if isdefined(outputs['transform_outputs']):
                transform_file = outputs['transform_outputs'][0]
                fs.utils.LTAConvert(
                    in_lta='identity.nofile',
                    source_file=in_file,
                    target_file=in_file,
                    out_lta=transform_file,
                ).run()
        return outputs


class _PrepareSynthStripGridInputSpec(BaseInterfaceInputSpec):
    input_image = File(exists=True, mandatory=True)


class _PrepareSynthStripGridOutputSpec(TraitedSpec):
    prepared_image = File(exists=True)


class PrepareSynthStripGrid(SimpleInterface):
    input_spec = _PrepareSynthStripGridInputSpec
    output_spec = _PrepareSynthStripGridOutputSpec

    def _run_interface(self, runtime):
        out_fname = fname_presuffix(
            self.inputs.input_image,
            newpath=runtime.cwd,
            suffix='_SynthStripGrid.nii',
            use_ext=False,
        )
        self._results['prepared_image'] = out_fname

        # possibly downsample the image for sloppy mode. Always ensure float32
        img = nb.load(self.inputs.input_image)
        if not img.ndim == 3:
            raise Exception('3D inputs are required for Synthstrip')
        xvoxels, yvoxels, zvoxels = img.shape

        def get_padding(nvoxels):
            extra_slices = nvoxels % 64
            if extra_slices == 0:
                return 0
            complete_64s = nvoxels // 64
            return 64 * (complete_64s + 1) - nvoxels

        def split_padding(padding):
            halfpad = padding // 2
            return halfpad, halfpad + halfpad % 2

        spad = get_padding(zvoxels)
        rpad, lpad = split_padding(get_padding(xvoxels))
        apad, ppad = split_padding(get_padding(yvoxels))

        zeropad = Zeropad(
            S=spad,
            R=rpad,
            L=lpad,
            A=apad,
            P=ppad,
            in_files=self.inputs.input_image,
            out_file=out_fname,
        )

        _ = zeropad.run()
        assert op.exists(out_fname)
        return runtime


class _SynthStripInputSpec(FSTraitedSpecOpenMP):
    input_image = File(argstr='-i %s', exists=True, mandatory=True)
    no_csf = traits.Bool(argstr='--no-csf', desc='Exclude CSF from brain border.')
    border = traits.Int(argstr='-b %d', desc='Mask border threshold in mm. Default is 1.')
    gpu = traits.Bool(argstr='-g')
    out_brain = File(
        argstr='-o %s',
        name_template='%s_brain.nii.gz',
        name_source=['input_image'],
        keep_extension=False,
        desc='skull stripped image with corrupt sform',
    )
    out_brain_mask = File(
        argstr='-m %s',
        name_template='%s_mask.nii.gz',
        name_source=['input_image'],
        keep_extension=False,
        desc='mask image with corrupt sform',
    )


class _SynthStripOutputSpec(TraitedSpec):
    out_brain = File(exists=True)
    out_brain_mask = File(exists=True)


class SynthStrip(FSCommandOpenMP):
    input_spec = _SynthStripInputSpec
    output_spec = _SynthStripOutputSpec
    _cmd = 'mri_synthstrip'

    def _num_threads_update(self):
        if self.inputs.num_threads:
            self.inputs.environ.update({'OMP_NUM_THREADS': '1'})


class FixHeaderSynthStrip(SynthStrip):
    def _run_interface(self, runtime, correct_return_codes=(0,)):
        # Run normally
        runtime = super()._run_interface(runtime, correct_return_codes)

        outputs = self._list_outputs()
        if not op.exists(outputs['out_brain']):
            raise Exception('mri_synthstrip failed!')

        if outputs.get('out_brain_mask'):
            _copyxform(self.inputs.input_image, outputs['out_brain_mask'])

        _copyxform(self.inputs.input_image, outputs['out_brain'])

        return runtime


class MockSynthStrip(SimpleInterface):
    input_spec = _SynthStripInputSpec
    output_spec = _SynthStripOutputSpec

    def _run_interface(self, runtime):
        from nipype.interfaces.fsl import BET

        this_bet = BET(
            mask=True,
            in_file=self.inputs.input_image,
            output_type='NIFTI_GZ',
        )
        result = this_bet.run()
        self._results['out_brain'] = result.outputs.out_file
        self._results['out_brain_mask'] = result.outputs.mask_file

        return runtime


class _SynthSegInputSpec(FSTraitedSpecOpenMP):
    input_image = File(argstr='--i %s', exists=True, mandatory=True)
    num_threads = traits.Int(
        default=1, argstr='--threads %d', usedefault=True, desc='Number of threads to use'
    )
    fast = traits.Bool(argstr='--fast', desc='fast predictions (lower quality).')
    robust = traits.Bool(argstr='--robust', desc='use robust predictions (slower).')
    out_seg = File(
        argstr='--o %s',
        name_template='%s_aseg.nii.gz',
        name_source=['input_image'],
        keep_extension=False,
        desc='segmentation image',
    )
    out_post = File(
        argstr='--post %s',
        name_template='%s_post.nii.gz',
        name_source=['input_image'],
        keep_extension=False,
        desc='posteriors image',
    )
    out_qc = File(
        argstr='--qc %s',
        name_template='%s_qc.csv',
        name_source=['input_image'],
        keep_extension=False,
        desc='qc csv',
    )
    cpu = traits.Bool(
        True, argstr='--cpu', usedefault=True, desc='Enforce running with CPU rather than GPU.'
    )


class _SynthSegOutputSpec(TraitedSpec):
    out_seg = File(exists=True)
    out_post = File(exists=True)
    out_qc = File(exists=True)


class SynthSeg(FSCommandOpenMP):
    input_spec = _SynthSegInputSpec
    output_spec = _SynthSegOutputSpec
    _cmd = 'mri_synthseg'

    def _format_arg(self, name, trait_spec, value):
        # Hardcode threads to be 1
        if name == 'num_threads':
            return '--threads 1'
        return super()._format_arg(name, trait_spec, value)

    def _num_threads_update(self):
        if self.inputs.num_threads:
            self.inputs.environ.update({'OMP_NUM_THREADS': '1'})


class MockSynthSeg(SimpleInterface):
    """A fake version of synthseg for testing."""

    input_spec = _SynthSegInputSpec
    output_spec = _SynthSegOutputSpec

    def _run_interface(self, runtime):
        from nipype.interfaces.fsl import BET

        output_qc = op.join(runtime.cwd, 'fake_synthseg_qc.csv')
        with open(output_qc, 'w') as qcf:
            qcf.write('Test QC file\n')

        # Get a brain mask
        this_bet = BET(
            mask=True,
            in_file=self.inputs.input_image,
            output_type='NIFTI_GZ',
        )
        result = this_bet.run()
        self._results['out_post'] = result.outputs.out_file

        # Make a fake segmentation
        img = nb.load(result.outputs.mask_file)
        orig_mask = img.get_fdata() > 0
        eroded1 = ndimage.binary_erosion(orig_mask, iterations=3)
        eroded2 = ndimage.binary_erosion(eroded1, iterations=3)
        final = orig_mask.astype(int) + eroded1 + eroded2
        out_img = nb.Nifti1Image(final, img.affine, header=img.header)
        out_fname = fname_presuffix(self.inputs.input_image, suffix='_dseg', newpath=runtime.cwd)
        out_img.to_filename(out_fname)
        self._results['out_seg'] = out_fname
        return runtime
