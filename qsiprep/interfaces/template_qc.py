"""Quality metrics for a subject-level template built from several images.

Answers "how do the individual images stack up against each other?" with numbers
rather than figures. Numbers sort, so an outlier announces itself instead of
waiting for someone to notice it in a montage.

This was written after a session (sub-0001a ses-8) that looked unremarkable in
every per-session figure but sat at 0.889 correlation to the template while every
other session was 0.959-0.975. That gap was obvious the moment it was tabulated
and invisible before.
"""

import numpy as np
import pandas as pd
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    InputMultiObject,
    SimpleInterface,
    TraitedSpec,
    traits,
)

#: Below this, the median absolute deviation is treated as degenerate.
_MIN_SPREAD = 1e-4
#: Absolute correlation drop below the median that counts as an outlier when the
#: inputs are too consistent for a scale-based rule to work.
_ABS_DROP = 0.05


class _TemplateQCInputSpec(BaseInterfaceInputSpec):
    aligned_images = InputMultiObject(File(exists=True), mandatory=True)
    template = File(exists=True, mandatory=True)
    transforms = InputMultiObject(File(exists=True))
    labels = traits.List(traits.Str)
    out_file = File('template_qc.tsv', usedefault=True)
    agreement_map = File('template_agreement.nii.gz', usedefault=True)


class _TemplateQCOutputSpec(TraitedSpec):
    out_file = File(exists=True)
    agreement_map = File(exists=True)


class TemplateQC(SimpleInterface):
    """Per-input agreement with the template, plus a voxelwise agreement map.

    Columns
    -------
    corr_to_template
        Correlation with the template inside the template's brain mask. The
        single most useful number: it is what identifies a session that does not
        belong.
    resid_cv
        SD/mean of the input relative to the template, over the same mask.
    translation_mm, rotation_deg
        How far this image had to move to reach the template. Large values mean
        unusual head placement, which is worth knowing even when the
        registration succeeded.

    The agreement map is voxelwise SD/mean across the aligned inputs. Note this
    is NOT a pure noise measure -- it mixes scan-to-scan noise with residual
    misregistration and genuine change. High values ringing the cortex indicate
    alignment error; a hot spot in one region usually means one input differs.
    """

    input_spec = _TemplateQCInputSpec
    output_spec = _TemplateQCOutputSpec

    def _run_interface(self, runtime):
        import os

        import nibabel as nb
        from scipy import io as sio
        from scipy import ndimage

        template_img = nb.load(self.inputs.template)
        template = np.asanyarray(template_img.dataobj, dtype='float32')

        positive = template[template > 0]
        if positive.size:
            mask = template > np.percentile(positive, 60)
            mask = ndimage.binary_erosion(ndimage.binary_fill_holes(mask), iterations=2)
        else:
            mask = np.ones(template.shape, dtype=bool)
        if mask.sum() < 100:  # degenerate template; fall back to any signal
            mask = template > 0

        labels = list(self.inputs.labels) if self.inputs.labels else []
        transforms = list(self.inputs.transforms) if self.inputs.transforms else []

        stack, rows = [], []
        for i, path in enumerate(self.inputs.aligned_images):
            data = np.asanyarray(nb.load(path).dataobj, dtype='float32')
            if data.shape != template.shape:
                continue
            stack.append(data)

            a, b = template[mask], data[mask]
            corr = float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else float('nan')
            scaled = b / b.mean() if b.mean() else b
            ref = a / a.mean() if a.mean() else a
            resid_cv = float((scaled - ref).std())

            translation = rotation = float('nan')
            if i < len(transforms):
                try:
                    mat = sio.loadmat(transforms[i])
                    # ITK writes AffineTransform_<float|double>_3_3 depending on
                    # the precision the registration ran at -- do not hardcode it.
                    key = next(
                        k
                        for k in mat
                        if k.startswith('AffineTransform') or k.startswith('MatrixOffset')
                    )
                    params = np.asarray(mat[key]).ravel()
                    rot = params[:9].reshape(3, 3)
                    translation = float(np.linalg.norm(params[9:12]))
                    # rotation angle of the matrix, in degrees
                    cos = (np.trace(rot) - 1.0) / 2.0
                    rotation = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
                except (KeyError, StopIteration, ValueError, IndexError) as exc:
                    # QC must never break a run, but a silent NaN column is
                    # useless -- say why it is empty.
                    from nipype import logging

                    logging.getLogger('nipype.interface').warning(
                        'TemplateQC could not read %s: %s', transforms[i], exc
                    )

            rows.append(
                {
                    'label': labels[i] if i < len(labels) else f'input{i:02d}',
                    'corr_to_template': corr,
                    'resid_cv': resid_cv,
                    'translation_mm': translation,
                    'rotation_deg': rotation,
                }
            )

        frame = pd.DataFrame(rows)
        if not frame.empty and frame['corr_to_template'].notna().sum() > 2:
            corr = frame['corr_to_template']
            median = corr.median()
            # Distance from the median, scaled by the median absolute deviation:
            # robust to the outlier itself, unlike a mean-based z-score, which an
            # outlier drags toward itself.
            spread = (corr - median).abs().median()
            if spread > _MIN_SPREAD:
                frame['corr_dev'] = (corr - median) / spread
                frame['outlier'] = frame['corr_dev'] < -3
            else:
                # MAD collapses to ~0 whenever the inputs agree closely, which is
                # the common case for a healthy subject. Scaling by it would make
                # every deviation infinite or undefined and flag nothing at all,
                # so fall back to an absolute drop below the median.
                frame['corr_dev'] = corr - median
                frame['outlier'] = frame['corr_dev'] < -_ABS_DROP
        out_file = os.path.join(runtime.cwd, self.inputs.out_file)
        frame.to_csv(out_file, sep='\t', index=False, na_rep='n/a')
        self._results['out_file'] = out_file

        agreement = os.path.join(runtime.cwd, self.inputs.agreement_map)
        if len(stack) > 1:
            arr = np.stack(stack)
            mean = arr.mean(0)
            sd = arr.std(0)
            with np.errstate(divide='ignore', invalid='ignore'):
                cov = np.where(mean > 0, sd / mean, 0.0)
            cov[~np.isfinite(cov)] = 0.0
        else:
            cov = np.zeros(template.shape, dtype='float32')
        nb.Nifti1Image(
            cov.astype('float32'), template_img.affine, template_img.header
        ).to_filename(agreement)
        self._results['agreement_map'] = agreement
        return runtime
