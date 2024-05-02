#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import logging
import os.path as op
import subprocess

import nilearn.image as nim
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker
from nipype.interfaces import ants
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    InputMultiObject,
    OutputMultiObject,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.utils.filemanip import fname_presuffix

LOGGER = logging.getLogger("nipype.interface")
from .bids import get_bids_params


class ScalarMapperInputSpec(BaseInterfaceInputSpec):
    scalars_from = InputMultiObject(traits.Str())
    recon_scalars = InputMultiObject(traits.Any())
    dwiref_image = File(exists=True)
    mapping_metadata = traits.Dict(
        desc="Info about the upstream workflow that created the anatomical mapping units"
    )


class ScalarMapperOutputSpec(TraitedSpec):
    mapped_scalars = traits.List(File(exists=True))


class ScalarMapper(SimpleInterface):
    input_spec = ScalarMapperInputSpec
    output_spec = ScalarMapperOutputSpec

    def _load_scalars(self):
        self.recon_scalars = self.inputs.recon_scalars
        for scalar_obj in self.recon_scalars:
            scalar_obj["image"] = nim.load_img(scalar_obj["path"])

    def _update_with_bids_info(self, summary_row_list):
        # Add BIDS info to the summarized scalars
        bids_info = get_bids_params(self.inputs.dwiref_image)
        for summary_row in summary_row_list:
            summary_row.update(bids_info)

    def _unload_scalars(self):
        for recon_scalar in self._results["mapped_scalars"]:
            if "image" in recon_scalar:
                del recon_scalar["image"]

    def _run_interface(self, runtime):
        self._do_mapping(runtime)
        return runtime


# For mapping to bundles
class _BundleMapperInputSpec(ScalarMapperInputSpec):
    tck_files = InputMultiObject(File(exists=True), desc="Paths to tck files")
    bundle_names = InputMultiObject(traits.Str())


class _BundleMapperOutputSpec(ScalarMapperOutputSpec):
    bundle_summary = File(exists=True)
    tdi_stats = File(exists=True)


def _get_tdi_img(dwiref_image, tck_file, output_tdi_file):
    subprocess.run(
        [
            "tckmap",
            "-template",
            dwiref_image,
            "-contrast",
            "tdi",
            "-force",
            tck_file,
            output_tdi_file,
        ],
        check=True,
    )
    return nim.load_img(output_tdi_file)


class BundleMapper(ScalarMapper):
    input_spec = _BundleMapperInputSpec
    output_spec = _BundleMapperOutputSpec

    def _do_mapping(self, runtime):
        self._load_scalars()
        bundle_dfs = []
        tdi_dfs = []
        source_suffix = self.inputs.mapping_metadata.get("qsirecon_suffix", "QSIPrep")
        for tck_name, tck_file in zip(self.inputs.bundle_names, self.inputs.tck_files):
            output_tdi_file = fname_presuffix(
                tck_file, suffix="_tdi.nii", newpath=runtime.cwd, use_ext=False
            )

            # Create a TDI, where streamline count is mapped to voxels
            tdi_img = _get_tdi_img(self.inputs.dwiref_image, tck_file, output_tdi_file)

            # Create a Masker from all voxels containing streamlines
            msk_img = nim.math_img("a>0", a=tdi_img)
            bundle_masker = NiftiMasker(msk_img)

            # Get a weighting vector from the TDI
            tdi_weights = bundle_masker.fit_transform(tdi_img).squeeze()
            tdi_weights = tdi_weights / tdi_weights.sum()

            # Start gathering stats with the TDI first
            tdi_dfs.append(
                calculate_mask_stats(
                    bundle_masker,
                    tck_name,
                    "bundle",
                    {
                        "image": tdi_img,
                        "variable_name": "tdi",
                        # Check that this is ok:
                        "source_file": self.inputs.recon_scalars[0]["source_file"],
                        "qsirecon_suffix": source_suffix,
                        "desc": "Streamline counts per voxel",
                    },
                )
            )

            # Then get the same stats for the scalars
            for recon_scalar in self.inputs.recon_scalars:
                bundle_dfs.append(
                    calculate_mask_stats(
                        bundle_masker, tck_name, "bundle", recon_scalar, tdi_weights
                    )
                )

        # Write the scalar summary df
        self._update_with_bids_info(bundle_dfs)
        summary_file = op.join(runtime.cwd, "bundle_stats.tsv")
        summary_df = pd.DataFrame(bundle_dfs)
        # Add information about which bundle workflow created these summaries
        if isdefined(self.inputs.mapping_metadata):
            summary_df["bundle_source"] = self.inputs.mapping_metadata.get("qsirecon_suffix")
            summary_df["bundle_params_id"] = self.inputs.mapping_metadata.get("name")
        summary_df.to_csv(summary_file, index=False, sep="\t")
        self._results["bundle_summary"] = summary_file

        # Write the TDI df
        self._update_with_bids_info(tdi_dfs)
        tdi_file = op.join(runtime.cwd, "tdi_stats.tsv")
        tdi_summary_df = pd.DataFrame(tdi_dfs)
        # Add information about which bundle workflow created these summaries
        if isdefined(self.inputs.mapping_metadata):
            tdi_summary_df["bundle_source"] = self.inputs.mapping_metadata.get("qsirecon_suffix")
            tdi_summary_df["bundle_params_id"] = self.inputs.mapping_metadata.get("name")
        tdi_summary_df.to_csv(tdi_file, index=False, sep="\t")
        self._results["tdi_stats"] = tdi_file


# For mapping to atlases


def calculate_mask_stats(
    masker, mask_name, mask_variable_name, recon_scalar, weighting_vector=None
):

    # Get the scalar data in the masked region
    voxel_data = masker.fit_transform(recon_scalar["image"]).squeeze()
    # Find out how much of this scalar is finite
    nz_voxel_data = voxel_data.copy()
    nz_voxel_data[nz_voxel_data == 0] = np.nan
    nz_voxel_data[~np.isfinite(voxel_data)] = np.nan

    # Make a prettier variable name
    variable_name = recon_scalar["variable_name"].replace("_image", "").replace("_file", "")

    results = {
        mask_variable_name: mask_name,
        "variable_name": variable_name,
        "qsirecon_suffix": recon_scalar["qsirecon_suffix"],
        "source_file": recon_scalar["source_file"],
        "zero_proportion": np.sum(np.isnan(nz_voxel_data)) / voxel_data.shape[0],
        "mean": np.mean(voxel_data),
        "stdev": np.std(voxel_data),
        "median": np.median(voxel_data),
        "masked_mean": np.nanmean(nz_voxel_data),
        "masked_median": np.nanmedian(nz_voxel_data),
        "masked_stdev": np.nanstd(nz_voxel_data),
    }

    if weighting_vector is not None:
        results["weighted_mean"] = np.sum(voxel_data * weighting_vector)
        nz_weighting_vector = weighting_vector.copy()
        try:
            nz_weighting_vector[np.isnan(nz_voxel_data)] = np.nan
            nz_weighting_vector = nz_weighting_vector / np.nansum(nz_weighting_vector)
            results["masked_weighted_mean"] = np.nansum(nz_voxel_data * nz_weighting_vector)
        except Exception as exc:
            LOGGER.warn(
                f"Error calculating weighted mean of {variable_name} in {mask_name}\n{exc}"
            )
            results["masked_weighted_mean"] = np.nan

    return results


class _TemplateMapperInputSpec(ScalarMapperInputSpec):
    template_reference_image = File(exists=True, mandatory=True)
    to_template_transform = File(exists=True, mandatory=True)
    interpolation = traits.Str("NearestNeighbor", usedefault=True)


class _TemplateMapperOutputSpec(ScalarMapperOutputSpec):
    template_space_scalars = OutputMultiObject(traits.Any())
    template_space_scalar_info = OutputMultiObject(traits.Any())


class TemplateMapper(ScalarMapper):
    input_spec = _TemplateMapperInputSpec
    output_spec = _TemplateMapperOutputSpec

    def _do_mapping(self, runtime):
        resampled_images = []
        resampled_image_metadata = []
        # Then get the same stats for the scalars
        for recon_scalar in self.inputs.recon_scalars:
            if recon_scalar.get("reorient_on_resample", False):
                # LOGGER.info(f"Skipping {recon_scalar}")
                continue
            new_metadata = recon_scalar.copy()
            output_fname = op.split(recon_scalar["path"])[1]
            output_fname = output_fname.replace("_space-T1w_", "_transformed_")
            output_fname = op.join(runtime.cwd, output_fname)
            transform = ants.ApplyTransforms(
                input_image=recon_scalar["path"],
                dimension=3,
                transforms=[self.inputs.to_template_transform],
                reference_image=self.inputs.template_reference_image,
                output_image=output_fname,
                interpolation=self.inputs.interpolation,
            )
            transform.terminal_output = "allatonce"
            transform.resource_monitor = False
            transform.run()
            resampled_images.append(output_fname)

            # Create new metadata for the resampled image
            new_metadata["path"] = output_fname
            if "bids" not in new_metadata:
                raise Exception(f"incomplete metadata spec {new_metadata}")
            new_metadata["bids"]["space"] = "MNI152NLin2009cAsym"
            resampled_image_metadata.append(new_metadata)

        self._results["template_space_scalars"] = resampled_images
        self._results["template_space_scalar_info"] = resampled_image_metadata
