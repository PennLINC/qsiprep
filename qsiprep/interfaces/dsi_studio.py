#!python
import os
import os.path as op
import logging
from glob import glob
from copy import deepcopy
from subprocess import Popen, PIPE
import numpy as np
from nipype.interfaces.base import (TraitedSpec, CommandLineInputSpec, BaseInterfaceInputSpec,
                                    CommandLine, File, traits, isdefined, SimpleInterface)
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
from nipype.utils.filemanip import fname_presuffix, split_filename
from scipy.io.matlab import loadmat, savemat
import nibabel as nb
import pandas as pd
LOGGER = logging.getLogger('nipype.interface')


class DSIStudioCommandLineInputSpec(CommandLineInputSpec):
    nthreads = traits.Int(1, usedefault=True, argstr="--threads=%d")


class DSIStudioCreateSrcInputSpec(DSIStudioCommandLineInputSpec):
    test_trait = traits.Bool()
    input_nifti_file = File(
        desc="DWI Nifti file",
        argstr="--source=%s")
    input_dicom_dir = File(
        desc="Directory with DICOM data from only the dwi",
        exists=True,
        argstr="--source=%s")
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
    output_odf = traits.Bool(
        True,
        usedefault=True,
        desc="Include full ODF's in output",
        argstr="--record_odf=1")
    odf_order = traits.Enum((8, 4, 5, 6, 10, 12, 16, 20),
                            usedefault=True,
                            desc="ODF tesselation order")
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

    # Decomposition traits
    decomposition = traits.Bool(
        False, argstr="--decomposition=1", desc="Apply ODF Decomposition")
    decomp_fraction = traits.Float(
        0.05,
        desc="Decomposition Fraction",
        argstr="--param3=%.2f",
        requires=["decomposition"])
    decomp_m_value = traits.Int(
        10,
        desc="Decomposition m value",
        argstr="--param4=%d",
        requires=["decomposition"])

    # Deconvolution traits
    deconvolution = traits.Bool(
        False, argstr="--deconvolution=1", desc="Apply ODF Deconvolution")
    deconv_regularization = traits.Float(
        0.5,
        requires=["deconvolution"],
        desc="Deconvolution regularization parameter",
        argstr="--param2=%.2f")


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
    gfa_file = File(desc="Exported files")
    fa0_file = File(desc="Exported files")
    fa1_file = File(desc="Exported files")
    fa2_file = File(desc="Exported files")
    fa3_file = File(desc="Exported files")
    fa4_file = File(desc="Exported files")
    iso_file = File(desc="Exported files")
    dti_fa_file = File(desc="Exported files")
    md_file = File(desc="Exported files")
    rd_file = File(desc="Exported files")
    ad_file = File(desc="Exported files")
    image0_file = File(desc="Exported files")


class DSIStudioExport(CommandLine):
    input_spec = DSIStudioExportInputSpec
    output_spec = DSIStudioExportOutputSpec
    _cmd = "dsi_studio --action=exp"

    def _list_outputs(self):
        outputs = self.output_spec().get()
        to_expect = self.inputs.to_export.split(",")
        for expected in to_expect:
            matches = glob("*" + expected + "*.nii.gz")
            if len(matches) == 1:
                outputs[expected + "_file"] = os.path.abspath(matches[0])
        return outputs


class DSIStudioConnectivityMatrixInputSpec(DSIStudioCommandLineInputSpec):
    trk_file = File(exists=True, argstr="--tract=%s")
    atlas_config = traits.Dict(desc='atlas configs for atlases to run connectivity for')
    atlas_name = traits.Str()
    input_fib = File(
        exists=True, argstr="--source=%s", mandatory=True, copyfile=False)
    fiber_count = traits.Int(xor=["seed_count"], argstr="--fiber_count=%d")
    seed_count = traits.Int(xor=["fiber_count"], argstr="--seed_count=%d")
    method = traits.Enum((0, 4), argstr="--method=%d", usedefault=True)
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
        official_labels = np.array(atlas_config['node_ids']).astype(np.int)
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
        num_threads = ifargs.pop('nthreads')
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
            wf_result = workflow.run(plugin='MultiProc', plugin_args={'n_procs': num_threads})
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
    network_region_ids = np.array(network_data['region_ids']).astype(np.int)
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
            reoriented_img.get_data()[::-1, ::-1, :],
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
