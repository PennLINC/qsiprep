import logging
import os
import os.path as op
import signal
from glob import glob
from subprocess import PIPE, Popen

import numpy as np
import pandas as pd
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    CommandLine,
    CommandLineInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.utils.filemanip import split_filename

from .. import config

LOGGER = logging.getLogger('nipype.interface')
DSI_STUDIO_VERSION = '94b9c79'
_APPTAINER_LIBRARY_PATH = '/.singularity.d/libs'


def _get_dsi_studio_environment(environ=None):
    # Exclude host libraries injected by Apptainer's --nv.
    environment = dict(os.environ if environ is None else environ)
    library_path = environment.get('LD_LIBRARY_PATH')
    if library_path is None:
        return environment
    apptainer_path = op.normpath(_APPTAINER_LIBRARY_PATH)
    library_paths = library_path.split(os.pathsep)
    library_paths = [path for path in library_paths if op.normpath(path) != apptainer_path]
    environment['LD_LIBRARY_PATH'] = os.pathsep.join(library_paths)
    return environment


class DSIStudioCommandLineInputSpec(CommandLineInputSpec):
    num_threads = traits.Int(1, usedefault=True, argstr='--thread_count=%d', nohash=True)


class DSIStudioCommandLine(CommandLine):
    """Run DSI Studio without libraries injected by Apptainer."""

    def _run_interface(self, runtime, correct_return_codes=(0,)):
        environment = {**os.environ, **self.inputs.environ}
        environment = _get_dsi_studio_environment(environment)
        if 'LD_LIBRARY_PATH' in environment:
            value = environment['LD_LIBRARY_PATH']
            self.inputs.environ['LD_LIBRARY_PATH'] = value
        return super()._run_interface(runtime, correct_return_codes)


class DSIStudioCreateSrcInputSpec(DSIStudioCommandLineInputSpec):
    test_trait = traits.Bool()
    input_nifti_file = File(desc='DWI Nifti file', argstr='--source=%s')
    input_dicom_dir = File(
        desc='Directory with DICOM data from only the dwi', exists=True, argstr='--source=%s'
    )
    bvec_convention = traits.Enum(
        ('DIPY', 'FSL'),
        usedefault=True,
        desc='Convention used for bvecs. FSL assumes LAS+ no matter image orientation',
    )
    input_bvals_file = File(desc='Text file containing b values', exists=True, argstr='--bval=%s')
    input_bvecs_file = File(
        desc='Text file containing b vectors (FSL format)', exists=True, argstr='--bvec=%s'
    )
    input_b_table_file = File(
        desc='Text file containing q-space sampling (DSI Studio format)',
        exists=True,
        argstr='--b_table=%s',
    )
    recursive = traits.Bool(
        False, desc='Search for DICOM files recursively', argstr='--recursive=1'
    )
    subject_id = traits.Str('data')
    output_src = File(desc='Output file (.src.gz)', argstr='--output=%s', genfile=True)
    grad_dev = File(
        desc='Gradient deviation file', exists=True, copyfile=True, position=-1, argstr='#%s'
    )


class DSIStudioCreateSrcOutputSpec(TraitedSpec):
    output_src = File(desc='Output file (.src.gz)', name_source='subject_id')


class DSIStudioCreateSrc(DSIStudioCommandLine):
    input_spec = DSIStudioCreateSrcInputSpec
    output_spec = DSIStudioCreateSrcOutputSpec
    _cmd = 'dsi_studio --action=src '

    def _pre_run_hook(self, runtime):
        """Convert DIPY-style bvals/bvecs to a DSI Studio b-table when needed.

        As of QSIPrep > 0.17 DSI Studio changed from DIPY bvecs to FSL bvecs.
        """
        # b_table files and dicom directories are ok
        if isdefined(self.inputs.input_b_table_file) or isdefined(self.inputs.input_dicom_dir):
            return runtime

        if not (
            isdefined(self.inputs.input_bvals_file) and isdefined(self.inputs.input_bvecs_file)
        ):
            raise Exception(
                'without a b_table or dicom directory, both bvals and bvecs must be specified'
            )

        # If the bvecs are in DIPY format, convert them to a b_table.txt
        if self.inputs.bvec_convention == 'DIPY':
            btable_file = self._gen_filename('output_src').replace('.src.gz', '.b_table.txt')
            btable_from_bvals_bvecs(
                self.inputs.input_bvals_file, self.inputs.input_bvecs_file, btable_file
            )
            self.inputs.input_b_table_file = btable_file
            self.inputs.input_bvals_file = traits.Undefined
            self.inputs.input_bvecs_file = traits.Undefined
            LOGGER.info('Converted DIPY LPS+ bval/bvec to DSI Studio b_table')
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

        output = op.abspath(fname) + '.src.gz'
        return output

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['output_src'] = self._gen_filename('output_src')
        return outputs


def _describe_exit(returncode):
    """Render a subprocess status, naming the signal when there was one."""
    if returncode is None:
        return 'did not report an exit status'
    if returncode < 0:
        signum = -returncode
        try:
            name = signal.Signals(signum).name
        except ValueError:
            name = f'signal {signum}'
        return f'was killed by {name}'
    return f'exited with status {returncode}'


def _qc_output_problem(returncode, qc_txt):
    """Say why DSI Studio produced no QC measurements, or return None.

    DSI Studio can fail without saying so. It exits non-zero, or crashes on a
    signal, or returns 0 having written an empty qc.txt. None of these was
    checked, and the only symptom used to appear much later as ``IndexError:
    list index out of range`` in ``load_src_qc_file``. A segfault on a DWI
    series with extreme intensity outliers is one way to get here.

    QC must not fail a run, so the caller logs this reason and the QC values
    become n/a. The reason is also carried to SeriesQC, which puts it in the
    HTML report.
    """
    status = _describe_exit(returncode)
    if returncode != 0:
        return f'DSI Studio {status}.'
    if not op.exists(qc_txt):
        return f'DSI Studio {status} but wrote no QC file.'
    if op.getsize(qc_txt) == 0:
        return (
            f'DSI Studio {status} but wrote an empty QC file. It does this when it '
            'cannot process the image at all, for example when the image has extreme '
            'intensity outliers.'
        )
    return None


def _run_qc_action(source_file, cwd, qc_txt):
    """Run DSI Studio's QC on one src or fib file and move its table to ``qc_txt``.

    Returns why there are no measurements, or None. ``qc_txt`` exists either
    way: an empty file reads as "no measurements" downstream.
    """
    cmd = ['dsi_studio', '--action=qc', '--source=' + source_file]
    proc = Popen(
        cmd,
        cwd=cwd,
        env=_get_dsi_studio_environment(),
        stdout=PIPE,
        stderr=PIPE,
    )
    out, err = proc.communicate()
    if out:
        LOGGER.info(out.decode())
    if err:
        LOGGER.critical(err.decode())

    # DSI Studio names the table qc.txt for src and fib files alike
    written = op.join(cwd, 'qc.txt')
    problem = _qc_output_problem(proc.returncode, written)
    if problem is not None:
        LOGGER.warning(
            'No DSI Studio QC measurements for %s; QC values will be n/a. %s Command: %s',
            source_file,
            problem,
            ' '.join(cmd),
        )
    if op.exists(written):
        os.replace(written, qc_txt)
    else:
        open(qc_txt, 'w').close()
    return problem


# Step 2 reonstruct ODFs: This is needed still in preprocessing because it's used for QC
class DSIStudioReconstructionInputSpec(DSIStudioCommandLineInputSpec):
    input_src_file = File(
        desc='DSI Studio src file',
        mandatory=True,
        exists=True,
        copyfile=False,
        argstr='--source=%s',
    )
    mask = File(
        desc='Volume to mask brain voxels', exists=True, copyfile=False, argstr='--mask=%s'
    )
    grad_dev = File(
        desc='Gradient deviation file', exists=True, copyfile=True, position=-1, argstr='#%s'
    )
    thread_count = traits.Int(1, usedefault=True, argstr='--thread_count=%d', nohash=True)

    dti_no_high_b = traits.Bool(
        True,
        usedefault=True,
        argstr='--dti_no_high_b=%d',
        desc='specify whether the construction of DTI should ignore high b-value (b>1500)',
    )
    r2_weighted = traits.Bool(
        False,
        usedefault=True,
        argstr='--r2_weighted=%d',
        desc='specify whether GQI and QSDR uses r2-weighted to calculate SDF',
    )

    # Outputs
    output_odf = traits.Bool(
        True, usedefault=True, desc="Include full ODF's in output", argstr='--record_odf=1'
    )
    odf_order = traits.Enum(
        (8, 4, 5, 6, 10, 12, 16, 20), usedefault=True, desc='ODF tessellation order'
    )
    # Which scalars to include
    other_output = traits.Str(
        'all',
        argstr='--other_output=%s',
        desc='additional diffusion metrics to calculate',
        usedefault=True,
    )
    align_acpc = traits.Bool(
        False, usedefault=True, argstr='--align_acpc=%d', desc='rotate image volume to align ap-pc'
    )
    check_btable = traits.Enum(
        (0, 1),
        usedefault=True,
        argstr='--check_btable=%d',
        desc='Check if btable matches nifti orientation (not foolproof)',
    )

    num_fibers = traits.Int(
        3,
        usedefault=True,
        argstr='--num_fiber=%d',
        desc='number of fiber populations estimated at each voxel',
    )


class DSIStudioReconstructionOutputSpec(TraitedSpec):
    output_fib = File(desc='Output File', exists=True)


class DSIStudioGQIReconstructionInputSpec(DSIStudioReconstructionInputSpec):
    ratio_of_mean_diffusion_distance = traits.Float(1.25, usedefault=True, argstr='--param0=%.4f')


class DSIStudioGQIReconstruction(DSIStudioCommandLine):
    input_spec = DSIStudioGQIReconstructionInputSpec
    output_spec = DSIStudioReconstructionOutputSpec
    _cmd = 'dsi_studio --action=rec --method=4'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        config.loggers.interface.info(f'current dir: {os.getcwd()}')
        srcname = os.path.split(self.inputs.input_src_file)[-1]
        config.loggers.interface.info(f'input src {self.inputs.input_src_file}')
        config.loggers.interface.info(f'split src name {srcname}')
        target = os.path.join(os.getcwd(), srcname) + '*gqi*.fib.gz'
        config.loggers.interface.info(f'search target: {target}')
        results = glob(target)
        assert len(results) == 1
        outputs['output_fib'] = results[0]

        return outputs


class _DSIStudioQCInputSpec(BaseInterfaceInputSpec):
    dwi_file = File(exists=True, mandatory=True, desc='a single 4D DWI series')
    bval_file = File(exists=True, mandatory=True, desc='b-values of dwi_file')
    bvec_file = File(exists=True, mandatory=True, desc='b-vectors of dwi_file')
    bvec_convention = traits.Enum(
        ('DIPY', 'FSL'),
        usedefault=True,
        desc='Convention used for bvecs. FSL assumes LAS+ no matter image orientation',
    )
    check_btable = traits.Enum(
        (0, 1),
        usedefault=True,
        desc='Check if btable matches nifti orientation (not foolproof)',
    )
    num_threads = traits.Int(1, usedefault=True, nohash=True, desc='threads for the GQI fit')


class _DSIStudioQCOutputSpec(TraitedSpec):
    src_qc = File(exists=True, desc='QC measures of the src file')
    fib_qc = File(exists=True, desc='QC measures of the GQI fit')
    src_qc_warning = traits.Str(
        desc='Why DSI Studio produced no src QC measurements. Undefined when it succeeded.'
    )
    fib_qc_warning = traits.Str(
        desc='Why DSI Studio produced no fib QC measurements. Undefined when it succeeded.'
    )


class DSIStudioQC(SimpleInterface):
    """Run DSI Studio's src and fib QC on a DWI series and keep only the measurements.

    The src file and the GQI fit are made here and removed once their QC tables
    are written. Nothing else reads them, and the fit with its ODFs is larger
    than the series it was made from.
    """

    input_spec = _DSIStudioQCInputSpec
    output_spec = _DSIStudioQCOutputSpec

    def _run_interface(self, runtime):
        create_src = DSIStudioCreateSrc(
            input_nifti_file=self.inputs.dwi_file,
            input_bvals_file=self.inputs.bval_file,
            input_bvecs_file=self.inputs.bvec_file,
            bvec_convention=self.inputs.bvec_convention,
        )
        create_src.resource_monitor = False
        src_file = create_src.run(cwd=runtime.cwd).outputs.output_src
        self._results['src_qc'] = op.join(runtime.cwd, 'src_qc.txt')
        problem = _run_qc_action(src_file, runtime.cwd, self._results['src_qc'])
        if problem is not None:
            self._results['src_qc_warning'] = problem

        fit_gqi = DSIStudioGQIReconstruction(
            input_src_file=src_file,
            thread_count=self.inputs.num_threads,
            check_btable=self.inputs.check_btable,
        )
        fit_gqi.resource_monitor = False
        fib_file = fit_gqi.run(cwd=runtime.cwd).outputs.output_fib
        self._results['fib_qc'] = op.join(runtime.cwd, 'fib_qc.txt')
        problem = _run_qc_action(fib_file, runtime.cwd, self._results['fib_qc'])
        if problem is not None:
            self._results['fib_qc_warning'] = problem

        os.remove(fib_file)
        os.remove(src_file)
        return runtime


class _DSIStudioQCMergeInputSpec(BaseInterfaceInputSpec):
    src_qc = File(exists=True, mandatory=True)
    fib_qc = File(exists=True, mandatory=True)
    src_qc_warning = traits.Str(desc='why DSIStudioQC produced no src measurements')
    fib_qc_warning = traits.Str(desc='why DSIStudioQC produced no fib measurements')
    cs_dmri_qc = File(exists=True, desc="cs_dmri's one-row QC table (CsDmriQC)")
    cs_dmri_qc_warning = traits.Str(desc='why CsDmriQC produced no measurements')


class _DSIStudioQCMergeOutputSpec(TraitedSpec):
    qc_file = File(exists=True)


class DSIStudioMergeQC(SimpleInterface):
    input_spec = _DSIStudioQCMergeInputSpec
    output_spec = _DSIStudioQCMergeOutputSpec

    def _run_interface(self, runtime):
        output_csv = runtime.cwd + '/merged_qc.csv'

        src_warning = _stage_warning(self.inputs.src_qc_warning, self.inputs.src_qc)
        fib_warning = _stage_warning(self.inputs.fib_qc_warning, self.inputs.fib_qc)

        # A stage with a warning is n/a even if its file holds something: after
        # a crash, whatever DSI Studio wrote cannot be trusted to be complete.
        merged = (
            _missing_qc(SRC_QC_MEASURES) if src_warning else load_src_qc_file(self.inputs.src_qc)
        )
        merged.update(
            _missing_qc(FIB_QC_MEASURES) if fib_warning else load_fib_qc_file(self.inputs.fib_qc)
        )

        messages = [
            f'{kind} QC: {warning}'
            for kind, warning in (('SRC', src_warning), ('FIB', fib_warning))
            if warning
        ]
        for message in messages:
            LOGGER.warning('DSI Studio QC values will be n/a. %s', message)

        if isdefined(self.inputs.cs_dmri_qc):
            _add_cs_dmri_qc(merged, self.inputs.cs_dmri_qc)
            if isdefined(self.inputs.cs_dmri_qc_warning) and self.inputs.cs_dmri_qc_warning:
                messages.append(self.inputs.cs_dmri_qc_warning)
        # Always present, empty when QC succeeded, so the column set never
        # varies. SeriesQC removes it from image_qc.tsv and reports it instead.
        merged[QC_WARNINGS_COLUMN] = [' '.join(messages)]

        pd.DataFrame(merged).to_csv(output_csv, index=False)
        self._results['qc_file'] = output_csv
        return runtime


class _DSIStudioBTableInputSpec(BaseInterfaceInputSpec):
    bval_file = File(exists=True, mandatory=True)
    bvec_file = File(exists=True, mandatory=True)
    bvec_convention = traits.Enum(
        ('DIPY', 'FSL'),
        usedefault=True,
        desc='Convention used for bvecs. FSL assumes LAS+ no matter image orientation',
    )


class _DSIStudioBTableOutputSpec(TraitedSpec):
    btable_file = File(exists=True)


class DSIStudioBTable(SimpleInterface):
    input_spec = _DSIStudioBTableInputSpec
    output_spec = _DSIStudioBTableOutputSpec

    def _run_interface(self, runtime):
        if self.inputs.bvec_convention != 'DIPY':
            raise NotImplementedError('Only DIPY Bvecs supported for now')
        btab_file = op.join(runtime.cwd, 'btable.txt')
        btable_from_bvals_bvecs(self.inputs.bval_file, self.inputs.bvec_file, btab_file)
        self._results['btable_file'] = btab_file
        return runtime


#: Column of merged_qc.csv that carries why a QC stage has no measurements.
#: Empty when QC succeeded. SeriesQC turns it into an HTML report warning.
QC_WARNINGS_COLUMN = 'qc_warnings'

#: The measures load_src_qc_file returns, in order. A QC stage that failed
#: returns the same keys with NaN values, so every run's image_qc.tsv has the
#: same columns and the tables still line up across subjects.
SRC_QC_MEASURES = (
    'dimension_x',
    'dimension_y',
    'dimension_z',
    'voxel_size_x',
    'voxel_size_y',
    'voxel_size_z',
    'max_b',
    'neighbor_corr',
    'masked_neighbor_corr',
    'dwi_contrast',
    'num_bad_slices',
    'num_directions',
)
FIB_QC_MEASURES = ('coherence_index',)


def _add_cs_dmri_qc(merged, cs_dmri_qc):
    """Append cs_dmri's measures to a merged DSI Studio QC row.

    Both report the image dimensions, voxel size and maximum b-value under the
    same names. DSI Studio's values are kept; cs_dmri's fill them only where
    DSI Studio produced none.
    """
    row = pd.read_csv(cs_dmri_qc).to_dict(orient='records')[0]
    for name, value in row.items():
        if name in merged and not pd.isna(merged[name][0]):
            continue
        merged[name] = [value]


def _qc_problem(lines):
    """Say why a DSI Studio qc.txt holds no measurements, or return None.

    A well-formed file has a header and one data row. Anything shorter used
    to surface as ``IndexError: list index out of range``.
    """
    if len(lines) >= 2 and lines[1].strip():
        return None
    if not lines:
        return 'The QC file is empty.'
    return 'The QC file holds only a header.'


def _missing_qc(measures, prefix=''):
    """Return the n/a row for a QC stage that produced no measurements."""
    return {prefix + name: [np.nan] for name in measures}


def _stage_warning(upstream, qc_file):
    """Explain why a QC stage has no measurements, preferring the reason DSIStudioQC gave."""
    if isdefined(upstream) and upstream:
        return upstream
    with open(qc_file) as fobj:
        return _qc_problem(fobj.readlines())


def load_src_qc_file(fname, prefix=''):
    with open(fname) as qc_file:
        qc_data = qc_file.readlines()
    if _qc_problem(qc_data) is not None:
        return _missing_qc(SRC_QC_MEASURES, prefix)
    data = qc_data[1]
    parts = data.strip().split('\t')
    dwi_contrast = np.nan
    ndc_masked = np.nan
    if len(parts) == 7:
        _, dims, voxel_size, dirs, max_b, ndc, bad_slices = parts
    elif len(parts) == 8:
        _, dims, voxel_size, dirs, max_b, _, ndc, bad_slices = parts
    elif len(parts) == 9:
        _, dims, voxel_size, dirs, max_b, dwi_contrast, ndc, ndc_masked, bad_slices = parts
    else:
        raise ValueError(
            f'Unrecognized DSI Studio QC row in {fname}: expected 7, 8 or 9 '
            f'tab-separated fields, got {len(parts)}: {data.strip()!r}'
        )

    voxelsx, voxelsy, voxelsz = map(float, voxel_size.strip().split())
    dimx, dimy, dimz = map(float, dims.strip().split())
    n_dirs = float(dirs.split('/')[1])
    max_b = float(max_b)
    dwi_corr = float(ndc)
    n_bad_slices = float(bad_slices)
    ndc_masked = float(ndc_masked)
    dwi_contrast = float(dwi_contrast)
    values = (
        dimx,
        dimy,
        dimz,
        voxelsx,
        voxelsy,
        voxelsz,
        max_b,
        dwi_corr,
        ndc_masked,
        dwi_contrast,
        n_bad_slices,
        n_dirs,
    )
    return {prefix + name: [value] for name, value in zip(SRC_QC_MEASURES, values, strict=True)}


def load_fib_qc_file(fname):
    with open(fname) as fibqc_f:
        lines = fibqc_f.readlines()
    if _qc_problem(lines) is not None:
        return _missing_qc(FIB_QC_MEASURES)
    return {'coherence_index': [float(lines[1].strip().split()[-1])]}


def btable_from_bvals_bvecs(bval_file, bvec_file, output_file):
    """Create a b-table from DIPY-style bvals/bvecs.

    Assuming these come from qsiprep they will be in LPS+, which
    is the same convention as DSI Studio's btable.
    """
    bvals = np.loadtxt(bval_file).squeeze()
    bvecs = np.loadtxt(bvec_file).squeeze()
    if 3 not in bvecs.shape:
        raise Exception(f'uninterpretable bval/bvec files\n\t{bval_file}\n\t{bvec_file}')
    if not bvecs.shape[1] == 3:
        bvecs = bvecs.T

    if not bvecs.shape[0] == bvals.shape[0]:
        raise Exception('Bval/Bvec mismatch')

    rows = []
    for row in map(tuple, np.column_stack([bvals, bvecs])):
        rows.append('%d %.6f %.6f %.6f' % row)

    # Write the actual file:
    with open(output_file, 'w') as btablef:
        btablef.write('\n'.join(rows) + '\n')
