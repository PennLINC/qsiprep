#!python
from __future__ import print_function

from nipype.interfaces.base import (TraitedSpec, CommandLineInputSpec, BaseInterfaceInputSpec,
                                    CommandLine, File, traits, isdefined, SimpleInterface)

import os
import os.path as op
from glob import glob
from nipype.utils.filemanip import fname_presuffix
import logging
from copy import deepcopy
import numpy as np
from scipy.io.matlab import loadmat, savemat
import nibabel as nb
LOGGER = logging.getLogger('nipype.interface')


# Step 1 from DSI Studio, importing DICOM files or nifti
class DSIStudioCreateSrcInputSpec(CommandLineInputSpec):
    test_trait = traits.Bool()
    input_nifti_file = File(
        desc="DWI Nifti file",
        argstr="--source=%s",
        exists=True,
        copyfile=False)
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
        name_template="%s.src.gz",
        default='',
        usedefault=True,
        desc="Output file (.src.gz)",
        argstr="--output=%s",
        name_source="subject_id",
        genfile=True)
    grad_dev = File(
        desc="Gradient deviation file",
        exists=True,
        copyfile=True,
        position=-1,
        argstr="#%s")


class DSIStudioCreateSrcOutputSpec(TraitedSpec):
    output_src = File(
        name_template="%s.src.gz",
        desc="Output file (.src.gz)",
        argstr="--output=%s",
        name_source="subject_id")


class DSIStudioCreateSrc(CommandLine):
    input_spec = DSIStudioCreateSrcInputSpec
    output_spec = DSIStudioCreateSrcOutputSpec
    _cmd = "dsi_studio --action=src "

    def _format_arg(self, name, spec, value):
        if name == "output_src":
            if not self.inputs.output_src:
                return '--output=' + self._gen_filename()
            return '--output=' + self.inputs.output_src
        return super(DSIStudioCreateSrc, self)._format_arg(name, spec, value)

    def _gen_filename(self):
        cwd = os.getcwd()
        input_fname = self.inputs.input_dicom_dir if isdefined(self.inputs.input_dicom_dir) \
            else self.inputs.input_nifti_file
        if not isdefined(input_fname):
            raise Exception()
        return fname_presuffix(input_fname, newpath=cwd, suffix=".src.gz", use_ext=False)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["output_src"] = self._gen_filename()
        return outputs


# Step 2 reonstruct ODFs
class DSIStudioReconstructionInputSpec(CommandLineInputSpec):
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
        (1, 0),
        usedefault=True,
        argstr="--check_btable=%d",
        desc="Check if btable matches nifti orientation (not foolproof)")
    num_fibers = traits.Int(
        3,
        usedefault=True,
        argstr="--num_fiber=%d",
        desc="number of fiber populations estimated at each voxel")
    grad_dev = File(
        desc="Gradient deviation file",
        exists=True,
        copyfile=True,
        position=-1,
        argstr="#%s")

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
        target = os.path.join(os.getcwd(), srcname) + \
            "*odf%d*f%d*gqi.%.2f.fib.gz" % (self.inputs.odf_order,
            self.inputs.num_fibers,
            self.inputs.ratio_of_mean_diffusion_distance)
        print("search target", target)
        results = glob(target)
        assert len(results) == 1
        outputs["output_fib"] = results[0]

        return outputs


class DSIStudioExportInputSpec(CommandLineInputSpec):
    input_file = File(
        exists=True, argstr="--source=%s", mandatory=True, copyfile=False)
    to_export = traits.Str(mandatory=True, argstr="--export=%s")


class DSIStudioExportOutputSpec(CommandLineInputSpec):
    gfa_file = File(desc="Exported files")
    fa0_file = File(desc="Exported files")
    fa1_file = File(desc="Exported files")
    fa2_file = File(desc="Exported files")
    fa3_file = File(desc="Exported files")
    fa4_file = File(desc="Exported files")
    iso_file = File(desc="Exported files")
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


class DSIStudioConnectivityMatrixInputSpec(CommandLineInputSpec):
    tract_file = File(exists=True, argstr="--tract=%s")
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
    output_trk = traits.Str(
        name_template="%s.trk.gz",
        desc="Output file (trk.gz)",
        argstr="--output=%s",
        name_source="input_fib")


class DSIStudioConnectivityMatrixOutputSpec(TraitedSpec):
    # What to write out
    connectivity_matrices = traits.List()


class DSIStudioConnectivityMatrix(CommandLine):
    input_spec = DSIStudioConnectivityMatrixInputSpec
    output_spec = DSIStudioConnectivityMatrixOutputSpec
    _cmd = "dsi_studio --action=ana "

    def _list_outputs(self):
        outputs = self.output_spec().get()
        results = glob("*.connectivity.mat")
        outputs["connectivity_matrices"] = [
            os.path.abspath(c) for c in results
        ]
        network_results = glob("*network*txt")
        outputs["connectivity_matrices"] += [op.abspath(c) for c in network_results]
        return outputs


class DSIStudioAtlasGraphInputSpec(DSIStudioConnectivityMatrixInputSpec):
    atlas_configs = traits.Dict(desc='atlas configs for atlases to run connectivity for')
    n_procs = traits.Int(1, usedefault=True)


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
        num_threads = ifargs.pop('n_procs')
        atlas_configs = ifargs.pop('atlas_configs')

        # flatten the atlas_configs
        args = [(atlas_name, atlas_config, ifargs) for atlas_name, atlas_config
                in atlas_configs.items()]

        if num_threads == 1:
            outputs = [_dsi_studio_connectivity(arg) for arg in args]
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=num_threads) as pool:
                outputs = list(pool.map(_dsi_studio_connectivity, args))

        commands = [out[0] for out in outputs]
        commands_file = op.join(runtime.cwd, "dsi_studio_commands.txt")
        with open(commands_file, "w") as f:
            f.write("\n----------\n".join(commands))
        self._results['commands'] = commands_file

        matfile_lists = [out[1] for out in outputs]
        merged_connectivity_file = op.join(runtime.cwd, "combined_connectivity.mat")
        _merge_conmats(matfile_lists, args, merged_connectivity_file)
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
            network_data['region_ids'] = tokens[1:]
            continue

        values = list(map(float, tokens[1:]))
        if len(values) == 1:
            network_data[measure_name] = values[0]
        else:
            network_data[measure_name] = np.array(values)

    return network_data


def _merge_conmats(matfile_lists, recon_args, outfile):
    """Merge the many matfiles output by dsi studio and ensure they conform"""
    connectivity_values = {}

    for matfile_list, (atlas_name, atlas_config, ifargs) in zip(matfile_lists, recon_args):
        matfiles = [f for f in matfile_list if f.endswith('.mat')]
        txtfiles = [f for f in matfile_list if f.endswith('.txt')]

        labels = np.array(atlas_config['node_ids']).astype(np.int)
        connectivity_values[atlas_name + "_region_ids"] = labels
        connectivity_values[atlas_name + "_region_labels"] = np.array(atlas_config['node_names'])
        n_atlas_labels = len(labels)

        for conmat in matfiles:
            m = loadmat(conmat)
            measure = "_".join(conmat.split(".")[-4:-2])
            # Column names are binary strings. Very confusing.
            column_names = "".join(
                [s.decode('UTF-8') for s in m["name"].squeeze().view("S1")]).split("\n")[:-1]
            region_ids = np.array([int(name[6:]) for name in column_names])

            # Where does each column go? Make an index array
            connectivity = m['connectivity']
            in_this_mask = np.in1d(labels, region_ids)
            truncated_labels = labels[in_this_mask]
            assert np.all(truncated_labels == region_ids)
            output = np.zeros((n_atlas_labels, n_atlas_labels))
            new_row = np.searchsorted(labels, region_ids)

            for row_index, conn in zip(new_row, connectivity):
                tmp = np.zeros(n_atlas_labels)
                tmp[in_this_mask] = conn
                output[row_index] = tmp
            connectivity_values[atlas_name + "_" + measure + "_connectivity"] = output

        for network_txt in txtfiles:
            measure = "_".join(network_txt.split(".")[-4:-2])
            network_data = _parse_network_file(network_txt)

            # Make sure to get the full atlas
            region_ids = np.array(network_data.pop('region_ids')).astype(np.int)
            in_this_mask = np.in1d(labels, region_ids)
            truncated_labels = labels[in_this_mask]
            assert np.all(truncated_labels == region_ids)
            new_row = np.searchsorted(labels, region_ids)

            for net_measure_name, net_measure_data in network_data.items():
                variable_name = atlas_name + "_" + measure + "_" + net_measure_name
                if type(net_measure_data) is np.ndarray:
                    tmp = np.zeros_like(net_measure_data)
                    tmp[in_this_mask] = net_measure_data
                    connectivity_values[variable_name] = tmp
                else:
                    connectivity_values[variable_name] = net_measure_data

    savemat(outfile, connectivity_values)


def _dsi_studio_connectivity(args):
    atlas_name, atlas_config, ifargs = args
    rundir = op.abspath(atlas_name)
    # Make a temporary directory for this node
    os.makedirs(rundir, exist_ok=True)
    ifargs = deepcopy(ifargs)
    input_fib = ifargs.pop('input_fib')
    new_fib = op.join(rundir, op.split(input_fib)[1])
    os.symlink(input_fib, new_fib)
    con = DSIStudioTracking(input_fib=new_fib, connectivity=atlas_config['dwi_resolution_file'],
                            **ifargs)
    con.terminal_output = 'allatonce'
    con.resource_monitor = False
    LOGGER.info(con.cmdline)
    run = con.run(cwd=rundir)
    runtime = run.runtime

    return runtime.cmdline, run.outputs.connectivity_matrices


class DSIStudioTrackingInputSpec(DSIStudioConnectivityMatrixInputSpec):
    roi = File(exists=True, argstr="--roi=%s")
    roi2 = File(exists=True, argstr="--roi2=%s")
    roa = File(exists=True, argstr="--roa=%s")
    end = File(exists=True, argstr="--end=%s")
    end2 = File(exists=True, argstr="--end2=%s")
    ter = File(exists=True, argstr="--ter=%s")


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
        results = glob(self.inputs.output_trk + "*src*fib*trk.gz")
        if len(results) == 1:
            trk_out = os.path.abspath(results[0])
            outputs["output_trk"] = trk_out
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
        nb.Nifti1Image(reoriented_img.get_data(), correct_img.affine, correct_img.header
                       ).to_filename(new_file)
        self._results['out_file'] = new_file

        return runtime
