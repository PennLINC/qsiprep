#!python
import os
from pathlib import Path
import os.path as op
import logging
from glob import glob
from copy import deepcopy
from subprocess import Popen, PIPE
import numpy as np
from nipype.interfaces.base import (TraitedSpec, CommandLineInputSpec, BaseInterfaceInputSpec,
                                    CommandLine, File, traits, isdefined, SimpleInterface,
                                    InputMultiObject, OutputMultiObject)
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
from nipype.utils.filemanip import fname_presuffix, split_filename, which
from scipy.io.matlab import loadmat, savemat
import nibabel as nb
import pandas as pd
from .bids import get_bids_params
LOGGER = logging.getLogger('nipype.interface')


class DSIStudioCommandLineInputSpec(CommandLineInputSpec):
    num_threads = traits.Int(
        1,
        usedefault=True,
        argstr="--thread_count=%d",
        nohash=True)


class DSIStudioCreateSrcInputSpec(DSIStudioCommandLineInputSpec):
    test_trait = traits.Bool()
    input_nifti_file = File(
        desc="DWI Nifti file",
        argstr="--source=%s")
    input_dicom_dir = File(
        desc="Directory with DICOM data from only the dwi",
        exists=True,
        argstr="--source=%s")
    bvec_convention = traits.Enum(
        ("DIPY", "FSL"),
        usedefault=True,
        desc="Convention used for bvecs. FSL assumes LAS+ no matter image orientation")
    input_bvals_file = File(
        desc="Text file containing b values", exists=True, argstr="--bval=%s")
    input_bvecs_file = File(
        desc="Text file containing b vectors (FSL format)",
        exists=True,
        argstr="--bvec=%s")
    input_b_table_file = File(
        desc="Text file containing q-space sampling (DSI Studio format)",
        exists=True,
        argstr="--b_table=%s")
    recursive = traits.Bool(
        False,
        desc="Search for DICOM files recursively",
        argstr="--recursive=1")
    subject_id = traits.Str("data")
    output_src = File(
        desc="Output file (.src.gz)",
        argstr="--output=%s",
        genfile=True)
    grad_dev = File(
        desc="Gradient deviation file",
        exists=True,
        copyfile=True,
        position=-1,
        argstr="#%s")


class DSIStudioCreateSrcOutputSpec(TraitedSpec):
    output_src = File(
        desc="Output file (.src.gz)",
        name_source="subject_id")


class DSIStudioCreateSrc(CommandLine):
    input_spec = DSIStudioCreateSrcInputSpec
    output_spec = DSIStudioCreateSrcOutputSpec
    _cmd = "dsi_studio --action=src "

    def _pre_run_hook(self, runtime):
        """As of QSIPrep > 0.17 DSI Studio changed from DIPY bvecs to FSL bvecs."""

        # b_table files and dicom directories are ok
        if isdefined(self.inputs.input_b_table_file) or \
                isdefined(self.inputs.input_dicom_dir):
            return runtime

        if not (isdefined(self.inputs.input_bvals_file) and \
                isdefined(self.inputs.input_bvecs_file)):
            raise Exception("without a b_table or dicom directory, both bvals and bvecs "
                            "must be specified")

        # If the bvecs are in DIPY format, convert them to a b_table.txt
        if self.inputs.bvec_convention == "DIPY":
            btable_file = self._gen_filename('output_src').replace(".src.gz", ".b_table.txt")
            btable_from_bvals_bvecs(self.inputs.input_bvals_file,
                                    self.inputs.input_bvecs_file,
                                    btable_file)
            self.inputs.input_b_table_file = btable_file
            self.inputs.input_bvals_file = traits.Undefined
            self.inputs.input_bvecs_file = traits.Undefined
            LOGGER.info("Converted DIPY LPS+ bval/bvec to DSI Studio b_table")
        return runtime


    def _gen_filename(self, name):
        if not name == 'output_src':
            return None
        if isdefined(self.inputs.input_nifti_file):
            _, fname, ext = split_filename(self.inputs.input_nifti_file)
        elif isdefined(self.inputs.input_dicom_dir):
            fname = op.split(self.inputs.dicom_dir)[1]
        else:
            raise Exception('Need either an input dicom director or nifti')

        output = op.abspath(fname) + ".src.gz"
        return output

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['output_src'] = self._gen_filename('output_src')
        return outputs


class _DSIStudioQCOutputSpec(TraitedSpec):
    qc_txt = File(exists=True, desc="Text file with QC measures")


class DSIStudioQC(SimpleInterface):
    output_spec = _DSIStudioQCOutputSpec

    def _run_interface(self, runtime):
        # DSI Studio (0.12.2) action=qc has two modes, depending on wether the
        # input is a file (src.gz|nii.gz)|(fib.gz) or a directory. For
        # directories, the action will be run on a number of detected files
        # (which *cannot* be symbolic links for some reason).
        src_file = fname_presuffix(self.inputs.src_file, newpath=runtime.cwd)
        cmd = ['dsi_studio', '--action=qc', '--source='+src_file]
        proc = Popen(cmd, cwd=runtime.cwd, stdout=PIPE, stderr=PIPE)
        out, err = proc.communicate()
        if out:
            LOGGER.info(out.decode())
        if err:
            LOGGER.critical(err.decode())
        self._results['qc_txt'] = src_file.replace(self.ext, '.qc.txt')
        return runtime


class _DSIStudioSrcQCInputSpec(DSIStudioCommandLineInputSpec):
    src_file = File(exists=True, copyfile=False, argstr="%s",
                    desc='DSI Studio src[.gz] file')


class DSIStudioSrcQC(DSIStudioQC):
    input_spec = _DSIStudioSrcQCInputSpec
    ext = '.src.gz'


class _DSIStudioFibQCInputSpec(DSIStudioCommandLineInputSpec):
    src_file = File(exists=True, copyfile=False, argstr="%s",
                    desc='DSI Studio fib[.gz] file')


class DSIStudioFibQC(DSIStudioQC):
    input_spec = _DSIStudioFibQCInputSpec
    ext = '.fib.gz'


# Step 2 reonstruct ODFs
class DSIStudioReconstructionInputSpec(DSIStudioCommandLineInputSpec):
    input_src_file = File(
        desc="DSI Studio src file",
        mandatory=True,
        exists=True,
        copyfile=False,
        argstr="--source=%s")
    mask = File(
        desc="Volume to mask brain voxels",
        exists=True,
        copyfile=False,
        argstr="--mask=%s")
    grad_dev = File(
        desc="Gradient deviation file",
        exists=True,
        copyfile=True,
        position=-1,
        argstr="#%s")
    thread_count = traits.Int(1, usedefault=True, argstr="--thread_count=%d")

    dti_no_high_b = traits.Bool(
        True,
        usedefault=True,
        argstr="--dti_no_high_b=%d",
        desc="specify whether the construction of DTI should ignore high b-value (b>1500)")
    r2_weighted = traits.Bool(
        False,
        usedefault=True,
        argstr="--r2_weighted=%d",
        desc="specify whether GQI and QSDR uses r2-weighted to calculate SDF")

    # Outputs
    output_odf = traits.Bool(
        True,
        usedefault=True,
        desc="Include full ODF's in output",
        argstr="--record_odf=1")
    odf_order = traits.Enum((8, 4, 5, 6, 10, 12, 16, 20),
                            usedefault=True,
                            desc="ODF tesselation order")
    # Which scalars to include
    other_output = traits.Str(
        "all",
        argstr="--other_output=%s",
        desc="additional diffusion metrics to calculate",
        usedefault=True
    )
    align_acpc = traits.Bool(
        False,
        usedefault=True,
        argstr="--align_acpc=%d",
        desc="rotate image volume to align ap-pc"
    )
    check_btable = traits.Enum(
        (0, 1),
        usedefault=True,
        argstr="--check_btable=%d",
        desc="Check if btable matches nifti orientation (not foolproof)")

    num_fibers = traits.Int(
        3,
        usedefault=True,
        argstr="--num_fiber=%d",
        desc="number of fiber populations estimated at each voxel")


class DSIStudioReconstructionOutputSpec(TraitedSpec):
    output_fib = File(desc="Output File", exists=True)


class DSIStudioGQIReconstructionInputSpec(DSIStudioReconstructionInputSpec):
    ratio_of_mean_diffusion_distance = traits.Float(
        1.25, usedefault=True, argstr="--param0=%.4f")


class DSIStudioDSIReconstructionInputSpec(DSIStudioReconstructionInputSpec):
    hamming_window_len = traits.Int(16, argstr="--param0=%d")


class DSIStudioReconstruction(CommandLine):
    input_spec = DSIStudioReconstructionInputSpec
    output_spec = DSIStudioReconstructionOutputSpec
    _cmd = "dsi_studio --action=rec "


class DSIStudioDTIReconstruction(DSIStudioReconstruction):
    _cmd = "dsi_studio --action=rec --method=1"
    input_spec = DSIStudioReconstructionInputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        print("current dir", os.getcwd())
        srcname = os.path.split(self.inputs.input_src_file)[-1]
        print("input src", self.inputs.input_src_file)
        print("split src name", srcname)
        target = os.path.join(os.getcwd(), srcname) + "*dti.fib.gz"
        print("search target", target)
        results = glob(target)
        assert len(results) == 1
        outputs["output_fib"] = results[0]

        return outputs


class DSIStudioGQIReconstruction(DSIStudioReconstruction):
    _cmd = "dsi_studio --action=rec --method=4"
    input_spec = DSIStudioGQIReconstructionInputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        print("current dir", os.getcwd())
        srcname = os.path.split(self.inputs.input_src_file)[-1]
        print("input src", self.inputs.input_src_file)
        print("split src name", srcname)
        target = os.path.join(os.getcwd(), srcname) + "*gqi*.fib.gz"
        print("search target", target)
        results = glob(target)
        assert len(results) == 1
        outputs["output_fib"] = results[0]

        return outputs


class DSIStudioExportInputSpec(DSIStudioCommandLineInputSpec):
    input_file = File(
        exists=True, argstr="--source=%s", mandatory=True, copyfile=False)
    to_export = traits.Str(mandatory=True, argstr="--export=%s")


class DSIStudioExportOutputSpec(DSIStudioCommandLineInputSpec):
    qa_file = File(desc="Exported scalar nifti")
    color_file = File(desc="Exported scalar nifti")
    dti_fa_file = File(desc="Exported scalar nifti")
    txx_file = File(desc="Exported scalar nifti")
    txy_file = File(desc="Exported scalar nifti")
    txz_file = File(desc="Exported scalar nifti")
    tyy_file = File(desc="Exported scalar nifti")
    tyz_file = File(desc="Exported scalar nifti")
    tzz_file = File(desc="Exported scalar nifti")
    rd1_file = File(desc="Exported scalar nifti")
    rd2_file = File(desc="Exported scalar nifti")
    ha_file = File(desc="Exported scalar nifti")
    md_file = File(desc="Exported scalar nifti")
    ad_file = File(desc="Exported scalar nifti")
    rd_file = File(desc="Exported scalar nifti")
    gfa_file = File(desc="Exported scalar nifti")
    iso_file = File(desc="Exported scalar nifti")
    rdi_file = File(desc="Exported scalar nifti")
    nrdi02L_file = File(desc="Exported scalar nifti")
    nrdi04L_file = File(desc="Exported scalar nifti")
    nrdi06L_file = File(desc="Exported scalar nifti")
    image0_file = File(desc="Exported files")


class DSIStudioExport(CommandLine):
    input_spec = DSIStudioExportInputSpec
    output_spec = DSIStudioExportOutputSpec
    _cmd = "dsi_studio --action=exp"

    def _list_outputs(self):
        outputs = self.output_spec().get()
        to_expect = self.inputs.to_export.split(",")
        cwd = Path()
        results = list(cwd.glob("*.nii.gz"))
        for expected in to_expect:
            matches = [fname.absolute().name for fname in results if
                       fname.name.endswith("." + expected + ".nii.gz")]
            if len(matches) == 1:
                outputs[expected + "_file"] = matches[0]
            elif len(matches) == 0:
                raise Exception("No exported scalar found matching " + expected)
            else:
                raise Exception("Found too mand scalar files matching " + expected)

        return outputs


class DSIStudioConnectivityMatrixInputSpec(DSIStudioCommandLineInputSpec):
    trk_file = File(exists=True, argstr="--tract=%s", copyfile=False)
    atlas_config = traits.Dict(desc='atlas configs for atlases to run connectivity for')
    atlas_name = traits.Str()
    input_fib = File(
        exists=True, argstr="--source=%s", mandatory=True, copyfile=False)
    fiber_count = traits.Int(xor=["seed_count"], argstr="--fiber_count=%d")
    seed_count = traits.Int(xor=["fiber_count"], argstr="--seed_count=%d")
    seed_plan = traits.Enum((0, 1), argstr="--seed_plan=%d")
    initial_dir = traits.Enum((0, 1, 2), argstr="--initial_dir=%d")
    interpolation = traits.Enum((0, 1, 2), argstr="--interpolation=%d")
    # ROI related options
    seed_file = File(
        exists=True,
        desc="specify the seeding file. "
        "Supported file format includes text, Analyze, and "
        "nifti files.",
        argstr="--seed=%s")
    to_export = traits.Str(argstr="--export=%s")
    connectivity = traits.Str(argstr="--connectivity=%s")
    connectivity_type = traits.Str(argstr="--connectivity_type=%s")
    connectivity_value = traits.Str(argstr="--connectivity_value=%s")
    random_seed = traits.Bool(argstr="--random_seed=1")
    # Tracking options
    fa_threshold = traits.Float(argstr="--fa_threshold=%.2f")
    step_size = traits.CFloat(argstr="--step_size=%.2f")
    turning_angle = traits.CFloat(argstr="--turning_angle=%.2f")
    interpo_angle = traits.CFloat(argstr="--interpo_angle=%.2f")
    smoothing = traits.CFloat(argstr="--smoothing=%.2f")
    min_length = traits.CInt(argstr="--min_length=%d")
    max_length = traits.CInt(argstr="--max_length=%d")
    thread_count = traits.Int(1, argstr="--thread_count=%d", usedefault=True)


class DSIStudioConnectivityMatrixOutputSpec(TraitedSpec):
    # What to write out
    connectivity_matfile = traits.File(exists=True)


class DSIStudioConnectivityMatrix(CommandLine):
    input_spec = DSIStudioConnectivityMatrixInputSpec
    output_spec = DSIStudioConnectivityMatrixOutputSpec
    _cmd = "dsi_studio --action=ana "

    def _post_run_hook(self, runtime):
        atlas_config = self.inputs.atlas_config
        atlas_name = self.inputs.atlas_name

        # Aggregate the connectivity/network data from DSI Studio
        official_labels = np.array(atlas_config['node_ids']).astype(int)
        connectivity_data = {
            atlas_name + "_region_ids": official_labels,
            atlas_name + "_region_labels": np.array(atlas_config['node_names'])
        }

        # Gather the connectivity matrices
        matfiles = glob(runtime.cwd + "/*.connectivity.mat")
        for matfile in matfiles:
            measure = "_".join(matfile.split(".")[-4:-2])
            connectivity_data[atlas_name + "_" + measure + "_connectivity"] = \
                _sanitized_connectivity_matrix(matfile, official_labels)

        # Gather the network measure files
        network_results = glob(runtime.cwd + "/*network*txt")
        for network_result in network_results:
            measure = "_".join(network_result.split(".")[-4:-2])
            connectivity_data.update(
                _sanitized_network_measures(network_result, official_labels,
                                            atlas_name, measure))
        merged_matfile = op.join(runtime.cwd, atlas_name + "_connectivity.mat")
        savemat(merged_matfile, connectivity_data, long_field_names=True)
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['connectivity_matfile'] = op.abspath(
            self.inputs.atlas_name + "_connectivity.mat")
        return outputs


class DSIStudioAtlasGraphInputSpec(DSIStudioConnectivityMatrixInputSpec):
    atlas_configs = traits.Dict(desc='atlas configs for atlases to run connectivity for')


class DSIStudioAtlasGraphOutputSpec(TraitedSpec):
    connectivity_matfile = File(exists=True)
    commands = File()


class DSIStudioAtlasGraph(SimpleInterface):
    """Produce one connectivity matrix per atlas based on DSI Studio tractography"""
    input_spec = DSIStudioAtlasGraphInputSpec
    output_spec = DSIStudioAtlasGraphOutputSpec

    def _run_interface(self, runtime):
        # Get all inputs from the ApplyTransforms object
        ifargs = self.inputs.get()
        ifargs.pop('connectivity')
        ifargs['thread_count'] = 1

        # Get number of parallel jobs
        num_threads = ifargs.pop('num_threads')
        atlas_configs = ifargs.pop('atlas_configs')
        workflow = pe.Workflow(name='dsistudio_atlasgraph')
        nodes = []
        merge_mats = pe.Node(niu.Merge(len(atlas_configs)), name='merge_mats')
        outputnode = pe.Node(niu.IdentityInterface(fields=['matfiles']), name='outputnode')
        workflow.connect(merge_mats, 'out', outputnode, 'matfiles')
        for atlasnum, (atlas_name, atlas_config) in enumerate(atlas_configs.items(), start=1):
            node_args = deepcopy(ifargs)
            # Symlink in the fib file
            node_args.pop('atlas_config')
            node_args.pop('atlas_name')
            nodes.append(
                pe.Node(
                    DSIStudioConnectivityMatrix(
                        atlas_config=atlas_config,
                        atlas_name=atlas_name,
                        connectivity=atlas_config['dwi_resolution_file'],
                        **node_args),
                    name=atlas_name)
            )
            workflow.connect(nodes[-1], 'connectivity_matfile',
                             merge_mats, 'in%d' % atlasnum)

        workflow.config['execution']['stop_on_first_crash'] = 'true'
        workflow.config['execution']['remove_unnecessary_outputs'] = 'false'
        workflow.base_dir = runtime.cwd
        if num_threads > 1:
            plugin_settings = {
                'plugin': 'MultiProc',
                'plugin_args': {
                    'raise_insufficient': False,
                    'maxtasksperchild': 1,
                    'n_procs': num_threads
                }
            }
            wf_result = workflow.run(**plugin_settings)
        else:
            wf_result = workflow.run()
        merge_node, = [node for node in list(wf_result.nodes) if node.name.endswith('merge_mats')]
        merged_connectivity_file = op.join(runtime.cwd, "combined_connectivity.mat")
        _merge_conmats(merge_node.result.outputs.out, merged_connectivity_file)
        self._results['connectivity_matfile'] = merged_connectivity_file

        return runtime


def _parse_network_file(txtfile):
    with open(txtfile, "r") as f:
        lines = f.readlines()
    network_data = {}
    for line in lines:
        sanitized_line = line.strip().replace("(", "_").replace(")", "")
        tokens = sanitized_line.split("\t")
        measure_name = tokens[0]
        if measure_name == 'network_measures':
            network_data['region_ids'] = [token.split('_')[-1] for token in tokens[1:]]
            continue

        values = list(map(float, tokens[1:]))
        if len(values) == 1:
            network_data[measure_name] = values[0]
        else:
            network_data[measure_name] = np.array(values)

    return network_data


def _merge_conmats(matfile_lists, outfile):
    """Merge the many matfiles output by dsi studio and ensure they conform"""
    connectivity_values = {}
    for matfile in matfile_lists:
        connectivity_values.update(loadmat(matfile))
    savemat(outfile, connectivity_values, long_field_names=True)


def _sanitized_connectivity_matrix(conmat, official_labels):
    """Load a matfile from DSI studio and re-format the connectivity matrix.

    Parameters:
    -----------

        conmat : str
            Path to a connectivity matfile from DSI Studio
        official_labels : ndarray (M,)
            Array of official ROI labels. The matrix in conmat will be reordered to
            match the ROI labels in this array

    Returns:
    --------
        connectivity_matrix : ndarray (M, M)
            The DSI Studio data reordered to match official_labels
    """
    m = loadmat(conmat)
    n_atlas_labels = len(official_labels)
    # Column names are binary strings. Very confusing.
    column_names = "".join(
        [s.decode('UTF-8') for s in m["name"].squeeze().view("S1")]).split("\n")[:-1]
    matfile_region_ids = np.array([int(name.split("_")[-1]) for name in column_names])

    # Where does each column go? Make an index array
    connectivity = m['connectivity']
    in_this_mask = np.isin(official_labels, matfile_region_ids)
    truncated_labels = official_labels[in_this_mask]
    assert np.all(truncated_labels == matfile_region_ids)
    output = np.zeros((n_atlas_labels, n_atlas_labels))
    new_row = np.searchsorted(official_labels, matfile_region_ids)

    for row_index, conn in zip(new_row, connectivity):
        tmp = np.zeros(n_atlas_labels)
        tmp[in_this_mask] = conn
        output[row_index] = tmp

    return output


def _sanitized_network_measures(network_txt, official_labels, atlas_name, measure):
    """Load a network text file from DSI studio and re-format it.

    Parameters:
    -----------

        network_txt : str
            Path to a network text file from DSI Studio
        official_labels : ndarray (M,)
            Array of official ROI labels. The matrix in conmat will be reordered to
            match the ROI labels in this array
        atlas_name : str
            Name of the atlas used
        measure : the name of the connectivity measure

    Returns:
    --------
        connectivity_matrix : ndarray (M, M)
            The DSI Studio data reordered to match official_labels
    """
    network_values = {}
    n_atlas_labels = len(official_labels)
    network_data = _parse_network_file(network_txt)
    # Make sure to get the full atlas
    network_region_ids = np.array(network_data['region_ids']).astype(int)
    # If all the regions are found
    in_this_mask = np.isin(official_labels, network_region_ids)
    if np.all(in_this_mask):
        truncated_labels = official_labels
    else:
        truncated_labels = official_labels[in_this_mask]
    assert np.all(truncated_labels == network_region_ids)

    for net_measure_name, net_measure_data in network_data.items():
        variable_name = atlas_name + "_" + measure + "_" + net_measure_name
        if type(net_measure_data) is np.ndarray:
            tmp = np.zeros(n_atlas_labels)
            tmp[in_this_mask] = net_measure_data
            network_values[variable_name] = tmp
        else:
            network_values[variable_name] = net_measure_data

    return network_values


class DSIStudioTrackingInputSpec(DSIStudioConnectivityMatrixInputSpec):
    roi = File(exists=True, argstr="--roi=%s")
    roi2 = File(exists=True, argstr="--roi2=%s")
    roa = File(exists=True, argstr="--roa=%s")
    end = File(exists=True, argstr="--end=%s")
    end2 = File(exists=True, argstr="--end2=%s")
    ter = File(exists=True, argstr="--ter=%s")
    output_trk = traits.Str(
        name_template="%s.trk.gz",
        desc="Output file (trk.gz)",
        argstr="--output=%s",
        name_source="input_fib")


class DSIStudioTrackingOutputSpec(TraitedSpec):
    output_trk = traits.Str(
        name_template="%s.trk.gz",
        desc="Output file (trk.gz)",
        argstr="--output=%s",
        name_source="input_fib")
    output_qa = File()
    output_gfa = File()
    connectivity_matrices = traits.List()


class DSIStudioTracking(CommandLine):
    input_spec = DSIStudioTrackingInputSpec
    output_spec = DSIStudioTrackingOutputSpec
    _cmd = "dsi_studio --action=trk"

    def _list_outputs(self):
        outputs = self.output_spec().get()
        results = glob("*trk.gz")
        if len(results) == 1:
            trk_out = os.path.abspath(results[0])
            outputs["output_trk"] = trk_out
        else:
            raise Exception("DSI Studio did not produce a trk.gz file")
        conmat_results = glob("*.connectivity.mat")
        outputs["connectivity_matrices"] = [
            os.path.abspath(c) for c in conmat_results
        ]
        if isdefined(self.inputs.to_export):
            if "gfa" in self.inputs.to_export:
                outputs["output_gfa"] = trk_out + ".gfa.txt"
            if "qa" in self.inputs.to_export:
                outputs["output_qa"] = trk_out + ".qa.txt"
        return outputs


class FixDSIStudioExportHeaderInputSpec(BaseInterfaceInputSpec):
    dsi_studio_nifti = File(exists=True, mandatory=True)
    correct_header_nifti = File(exists=True, mandatory=True)


class FixDSIStudioExportHeaderOutputSpec(TraitedSpec):
    out_file = File(exists=True)


class FixDSIStudioExportHeader(SimpleInterface):
    input_spec = FixDSIStudioExportHeaderInputSpec
    output_spec = FixDSIStudioExportHeaderOutputSpec

    def _run_interface(self, runtime):
        dsi_studio_file = self.inputs.dsi_studio_nifti
        new_file = fname_presuffix(dsi_studio_file, suffix="fixhdr", newpath=runtime.cwd)
        dsi_img = nb.load(dsi_studio_file)
        correct_img = nb.load(self.inputs.correct_header_nifti)

        new_axcodes = nb.aff2axcodes(correct_img.affine)
        input_axcodes = nb.aff2axcodes(dsi_img.affine)

        # Is the input image oriented how we want?
        if not input_axcodes == new_axcodes:
            # Re-orient
            input_orientation = nb.orientations.axcodes2ornt(input_axcodes)
            desired_orientation = nb.orientations.axcodes2ornt(new_axcodes)
            transform_orientation = nb.orientations.ornt_transform(input_orientation,
                                                                   desired_orientation)
            reoriented_img = dsi_img.as_reoriented(transform_orientation)

        else:
            reoriented_img = dsi_img

        # No matter what, still use the correct affine
        nb.Nifti1Image(
            reoriented_img.get_fdata(),
            correct_img.affine).to_filename(new_file)
        self._results['out_file'] = new_file

        return runtime


class _DSIStudioQCMergeInputSpec(BaseInterfaceInputSpec):
    src_qc = File(exists=True, mandatory=True)
    fib_qc = File(exists=True, mandatory=True)


class _DSIStudioQCMergeOutputSpec(TraitedSpec):
    qc_file = File(exists=True)


class DSIStudioMergeQC(SimpleInterface):
    input_spec = _DSIStudioQCMergeInputSpec
    output_spec = _DSIStudioQCMergeOutputSpec

    def _run_interface(self, runtime):
        output_csv = runtime.cwd + "/merged_qc.csv"
        src_qc = load_src_qc_file(self.inputs.src_qc)
        fib_qc = load_fib_qc_file(self.inputs.fib_qc)
        src_qc.update(fib_qc)
        qc_df = pd.DataFrame(src_qc)
        qc_df.to_csv(output_csv, index=False)
        self._results['qc_file'] = output_csv
        return runtime


class _DSIStudioBTableInputSpec(BaseInterfaceInputSpec):
    bval_file = File(exists=True, mandatory=True)
    bvec_file = File(exists=True, mandatory=True)
    bvec_convention = traits.Enum(
        ("DIPY", "FSL"),
        usedefault=True,
        desc="Convention used for bvecs. FSL assumes LAS+ no matter image orientation")


class _DSIStudioBTableOutputSpec(TraitedSpec):
    btable_file = File(exists=True)


class DSIStudioBTable(SimpleInterface):
    input_spec = _DSIStudioBTableInputSpec
    output_spec = _DSIStudioBTableOutputSpec

    def _run_interface(self, runtime):
        if self.inputs.bvec_convention != "DIPY":
            raise NotImplementedError("Only DIPY Bvecs supported for now")
        btab_file = op.join(runtime.cwd, "btable.txt")
        btable_from_bvals_bvecs(self.inputs.bval_file, self.inputs.bvec_file, btab_file)
        self._results['btable_file'] = btab_file
        return runtime


class _AutoTrackInputSpec(DSIStudioCommandLineInputSpec):
    fib_file = File(exists=True,
                    mandatory=True,
                    copyfile=False,
                    argstr="--source=%s")
    map_file = File(exists=True,
                    mandatory=True,
                    copyfile=False)
    track_id = traits.Str(
        "Fasciculus,Cingulum,Aslant,Corticos,Thalamic_R,Reticular,Optic,Fornix,Corpus",
        usedefault=True,
        argstr="--track_id=%s",
        desc="""specify the id number or the name of the bundle. The id can be found in
            /atlas/ICBM152/HCP1065.tt.gz.txt . This text file is included in DSI
            Studio package (For Mac, right-click on dsi_studio_64.app to find
            content). You can specify partial name of the bundle:

            example:
            for tracking left and right arcuate fasciculus, assign
            --track_id=0,1 or --track_id=arcuate (DSI Studio will find bundles
            with names containing "arcuate", case insensitive)

            example:
            for tracking left and right arcuate and cingulum, assign
            -track_id=0,1,2,3 or -track_id=arcuate,cingulum""")
    track_voxel_ratio = traits.CFloat(
        2.0,
        usedefault=True,
        argstr="--track_voxel_ratio=%.2f",
        desc="the track-voxel ratio for the total number of streamline count. A larger " \
            "value gives better mapping with the expense of computation time.")
    tolerance = traits.Str(
        "22,26,30",
        argstr="--tolerance=%s",
        desc="""the tolerance for the bundle recognition. The unit is in mm. Multiple values
            can be assigned using comma separator. A larger value may include larger track
            variation but also subject to more false results.""")
    yield_rate = traits.CFloat(
        0.00001,
        argstr="--yield_rate=%.10f",
        desc="This rate will be used to terminate tracking early if DSI Studio find the " \
            "fiber trackings is not generating results")
    export_trk = traits.Bool(
        True,
        usedefault=True,
        argstr="--export_trk=%d")
    trk_format = traits.Enum(
        ("trk.gz", "tt.gz"),
        default="trk.gz",
        usedefault=True,
        argstr="--trk_format=%s")
    output_dir = traits.Str(
        "cwd",
        argstr="%s",
        usedefault=True,
        desc="Forces DSI Studio to write results in cwd")


class _AutoTrackOutputSpec(DSIStudioCommandLineInputSpec):
    native_trk_files = OutputMultiObject(File(exists=True))
    stat_files = OutputMultiObject(File(exists=True))


class AutoTrack(CommandLine):
    input_spec = _AutoTrackInputSpec
    output_spec = _AutoTrackOutputSpec
    _cmd = "dsi_studio --action=atk"

    def _format_arg(self, name, trait_spec, value):
        if name == "output_dir":
            return "--output=" + str(Path(".").absolute())
        return super()._format_arg(name, trait_spec, value)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        cwd = Path(".")
        trk_files = [str(fname.absolute()) for fname in cwd.rglob("*."+self.inputs.trk_format)]
        stat_files = [str(fname.absolute()) for fname in cwd.rglob("*.stat.txt")]
        outputs["native_trk_files"] = trk_files
        outputs["stat_files"] = stat_files
        return outputs


class _AutoTrackInitInputSpec(_AutoTrackInputSpec):
    num_threads = 1  # Force this so it doesn't use a ton of threads
    map_file = File(mandatory=False)
    track_id = "0"


class _AutoTrackInitOutputSpec(_AutoTrackOutputSpec):
    map_file = File(exists=True)


class AutoTrackInit(AutoTrack):
    input_spec = _AutoTrackInitInputSpec
    output_spec = _AutoTrackInitOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        cwd = Path(".")
        map_files = list(cwd.glob("*.map.gz"))
        if len(map_files) > 1:
            raise Exception("Too many map files generated")
        if not map_files:
            raise Exception("No map files found in " + str(cwd.absolute()))
        outputs["map_file"] = str(map_files[0].absolute())
        return outputs


class _AggregateAutoTrackResultsInputSpec(BaseInterfaceInputSpec):
    expected_bundles = InputMultiObject(traits.Str())
    trk_files = InputMultiObject(File(exists=True))
    stat_files = InputMultiObject(File(exists=True))
    source_file = File(exists=True)


class _AggregateAutoTrackResultsOutputSpec(TraitedSpec):
    bundle_csv = File(exists=True)
    found_bundle_names = OutputMultiObject(traits.Str())
    found_bundle_files = OutputMultiObject(File(exists=True))


class AggregateAutoTrackResults(SimpleInterface):
    input_spec = _AggregateAutoTrackResultsInputSpec
    output_spec = _AggregateAutoTrackResultsOutputSpec

    def _run_interface(self, runtime):

        def bundle_from_file(file_name):
            return Path(file_name).parts[-2]

        trk_map = {bundle_from_file(fname): fname for fname in self.inputs.trk_files}
        stat_map ={bundle_from_file(fname): fname for fname in self.inputs.stat_files}

        stats_rows = []
        found_bundle_files = []
        found_bundle_names = []
        for bundle_name in self.inputs.expected_bundles:
            if bundle_name in trk_map:
                found_bundle_files.append(trk_map[bundle_name])
                found_bundle_names.append(bundle_name)
            stats_rows.append(
                stat_txt_to_df(stat_map.get(bundle_name, "NA"), bundle_name))

        stats_df = pd.DataFrame(stats_rows)
        bids_info = get_bids_params(self.inputs.source_file)
        for name, value in bids_info.items():
            stats_df[name] = value
        stats_df["source_file"] = op.split(self.inputs.source_file)[1]
        csv_file = fname_presuffix(self.inputs.source_file, newpath=runtime.cwd,
                                   suffix="_bundlestats.csv", use_ext=False)
        stats_df.to_csv(csv_file, index=False)
        self._results["found_bundle_names"] = found_bundle_names
        self._results["found_bundle_files"] = found_bundle_files
        self._results["bundle_csv"] = csv_file
        return runtime


def stat_txt_to_df(stat_txt_file, bundle_name):
    bundle_stats = {'bundle_name': bundle_name}
    if stat_txt_file == "NA":
        return bundle_stats
    with open(stat_txt_file, "r") as statf:
        lines = [
            line.strip().replace(" ", "_").replace("^", "").replace("(", "_").replace(")", "")
            for line in statf]

    for line in lines:
        name, value = line.split("\t")
        bundle_stats[name] = float(value)

    return bundle_stats


def load_src_qc_file(fname, prefix=""):
    with open(fname, "r") as qc_file:
        qc_data = qc_file.readlines()
    data = qc_data[1]
    parts = data.strip().split('\t')
    if len(parts) == 7:
        _, dims, voxel_size, dirs, max_b, ndc, bad_slices = parts
    elif len(parts) == 8:
        _, dims, voxel_size, dirs, max_b, _, ndc, bad_slices = parts
    else:
        raise Exception("Unknown QC File format")

    voxelsx, voxelsy, voxelsz = map(float, voxel_size.strip().split())
    dimx, dimy, dimz = map(float, dims.strip().split())
    n_dirs = float(dirs)
    max_b = float(max_b)
    dwi_corr = float(ndc)
    n_bad_slices = float(bad_slices)
    data = {
        prefix + 'dimension_x': [dimx],
        prefix + 'dimension_y': [dimy],
        prefix + 'dimension_z': [dimz],
        prefix + 'voxel_size_x': [voxelsx],
        prefix + 'voxel_size_y': [voxelsy],
        prefix + 'voxel_size_z': [voxelsz],
        prefix + 'max_b': [max_b],
        prefix + 'neighbor_corr': [dwi_corr],
        prefix + 'num_bad_slices': [n_bad_slices],
        prefix + 'num_directions': [n_dirs]
    }
    return data


def load_fib_qc_file(fname):
    with open(fname, "r") as fibqc_f:
        lines = [line.strip().split() for line in fibqc_f]
    return {'coherence_index': [float(lines[0][-1])],
            'incoherence_index': [float(lines[1][-1])]}


def btable_from_bvals_bvecs(bval_file, bvec_file, output_file):
    """Create a b-table from DIPY-style bvals/bvecs.

    Assuming these come from qsiprep they will be in LPS+, which
    is the same convention as DSI Studio's btable.
    """
    bvals = np.loadtxt(bval_file).squeeze()
    bvecs = np.loadtxt(bvec_file).squeeze()
    if not 3 in bvecs.shape:
        raise Exception(
            "uninterpretable bval/bvec files\n\t{}\n\t{}".format(bval_file, bvec_file))
    if not bvecs.shape[1] == 3:
        bvecs = bvecs.T

    if not bvecs.shape[0] == bvals.shape[0]:
        raise Exception("Bval/Bvec mismatch")

    rows = []
    for row in map(tuple, np.column_stack([bvals, bvecs])):
        rows.append("%d %.6f %.6f %.6f" % row)

    # Write the actual file:
    with open(output_file, "w") as btablef:
        btablef.write("\n".join(rows) + "\n")


def _get_dsi_studio_bundles(desired_bundles=""):

    dsi_studio_exe = which('dsi_studio')
    if not dsi_studio_exe:
        raise Exception("No dsi_studio executable found in $PATH")
    bundle_dir = op.split(dsi_studio_exe)[0]
    bundle_file = os.getenv(
        "DSI_STUDIO_BUNDLES",
        op.join(bundle_dir, "atlas/ICBM152_adult/ICBM152_adult.tt.gz.txt"))
    if not op.exists(bundle_file):
        raise Exception("No such file {} for loading bundles".format(bundle_file))

    with open(bundle_file, "r") as bundlef:
        all_bundles = [line.strip() for line in bundlef]

    if not desired_bundles:
        LOGGER.info("Using all {} bundles from {}".format(len(all_bundles), bundle_file))
        return all_bundles

    def get_bundles(search_string):
        return [bundle for bundle in all_bundles if search_string.lower() in bundle.lower()]

    # Figure out which bundles we'll be tracking
    bundles_to_track = []
    for bundle in desired_bundles.split(","):
        if bundle.isdigit():
            bundle_index = int(bundle)
            if bundle_index < 0 or bundle_index > len(all_bundles):
                raise Exception("{} is not a valid bundle index, check {}".format(
                    bundle_index, bundle_file))
            bundles_to_track.append(all_bundles[bundle_index])
        else:
            matching_bundles = get_bundles(bundle)
            if not matching_bundles:
                LOGGER.warn("No matching bundles found for " + bundle)
            bundles_to_track.extend(matching_bundles)
    return bundles_to_track
