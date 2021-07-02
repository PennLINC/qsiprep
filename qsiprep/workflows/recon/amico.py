"""
AMICO Reconstruction workflows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_amico_noddi_fit_wf

"""
import logging
import nipype.pipeline.engine as pe
from nipype.interfaces import afni, utility as niu
from qsiprep.interfaces.bids import ReconDerivativesDataSink
from .interchange import input_fields
from ...engine import Workflow
from ...interfaces.amico import NODDI
from ...interfaces.reports import ReconPeaksReport
from ...interfaces.converters import NODDItoFIBGZ

LOGGER = logging.getLogger('nipype.interface')


def init_amico_noddi_fit_wf(omp_nthreads, has_transform,
                            name="amico_noddi_recon",
                            output_suffix="", params={}):
    """Reconstruct EAPs, ODFs, using 3dSHORE (brainsuite-style basis set).

    Inputs

        *qsiprep outputs*

    Outputs

        directions_image
            Image of directions
        icvf_image
            Voxelwise ICVF.
        od_image
            Voxelwise Orientation Dispersion
        isovf_image
            Voxelwise ISOVF
        config_file
            Pickle file with model configurations in it
        fibgz

    """

    inputnode = pe.Node(niu.IdentityInterface(fields=input_fields + ['odf_rois']),
                        name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['directions_image', 'icvf_image', 'od_image',
                    'isovf_image', 'config_file', 'fibgz']),
        name="outputnode")

    workflow = Workflow(name=name)
    desc = """NODDI Reconstruction

: """
    resample_mask = pe.Node(
        afni.Resample(outputtype='NIFTI_GZ', resample_mode="NN"), name='resample_mask')
    noddi_fit = pe.Node(NODDI(**params), name="recon_noddi")
    desc += """\
The NODDI model (@noddi) was fit using the AMICO implementation (@amico).
A value of %.1E was used for parallel diffusivity and %.1E for isotropic
diffusivity.""" % (params['dPar'], params['dIso'])
    if params.get('is_exvivo'):
        desc += " An additional component was added to the model foe ex-vivo data."

    convert_to_fibgz = pe.Node(NODDItoFIBGZ(), name='convert_to_fibgz')

    plot_peaks = pe.Node(ReconPeaksReport(), name='plot_peaks')
    ds_report_peaks = pe.Node(
        ReconDerivativesDataSink(extension='.png',
                                 desc="NODDI",
                                 suffix='peaks'),
        name='ds_report_peaks',
        run_without_submitting=True)

    workflow.connect([
        (inputnode, noddi_fit, [('dwi_file', 'dwi_file'),
                                ('bval_file', 'bval_file'),
                                ('bvec_file', 'bvec_file')]),
        (inputnode, resample_mask, [('t1_brain_mask', 'in_file'),
                                    ('dwi_file', 'master')]),
        (resample_mask, noddi_fit, [('out_file', 'mask_file')]),
        (noddi_fit, outputnode, [
            ('directions_image', 'directions_image'),
            ('icvf_image', 'icvf_image'),
            ('od_image', 'od_image'),
            ('isovf_image', 'isovf_image'),
            ('config_file', 'config_file'),
            ]),
        (noddi_fit, convert_to_fibgz, [
            ('directions_image', 'directions_file'),
            ('icvf_image', 'icvf_file'),
            ('od_image', 'od_file'),
            ('isovf_image', 'isovf_file'),
            ]),
        (resample_mask, convert_to_fibgz, [('out_file', 'mask_file')]),
        (convert_to_fibgz, plot_peaks, [('fibgz_file', 'fib_file')]),
        (convert_to_fibgz, outputnode, [('fibgz_file', 'fibgz')]),
        (resample_mask, plot_peaks, [('out_file', 'mask_file')]),
        (noddi_fit, plot_peaks, [('icvf_image', 'background_image')]),
        (plot_peaks, ds_report_peaks, [('out_report', 'in_file')]),
        ])

    if output_suffix:
        ds_fibgz = pe.Node(
        ReconDerivativesDataSink(extension='.fib.gz',
                                    suffix=output_suffix,
                                    compress=True),
        name='ds_{}_fibgz'.format(output_suffix),
        run_without_submitting=True)
        workflow.connect(outputnode, 'fibgz', ds_fibgz, 'in_file')

        # Niftis from AMICO
        ds_directions = pe.Node(
            ReconDerivativesDataSink(extension='.nii.gz',
                                     desc="directions",
                                     suffix=output_suffix,
                                     compress=True),
            name='ds_noddi_directions',
            run_without_submitting=True)
        workflow.connect(outputnode, 'directions_image', ds_directions, 'in_file')

        ds_icvf = pe.Node(
            ReconDerivativesDataSink(extension='.nii.gz',
                                     desc="ICVF",
                                     suffix=output_suffix,
                                     compress=True),
            name='ds_noddi_icvf',
            run_without_submitting=True)
        workflow.connect(outputnode, 'icvf_image', ds_icvf, 'in_file')

        ds_isovf = pe.Node(
            ReconDerivativesDataSink(extension='.nii.gz',
                                     desc="ISOVF",
                                     suffix=output_suffix,
                                     compress=True),
            name='ds_noddi_isovf',
            run_without_submitting=True)
        workflow.connect(outputnode, 'isovf_image', ds_isovf, 'in_file')

        ds_od = pe.Node(
            ReconDerivativesDataSink(extension='.nii.gz',
                                     desc="OD",
                                     suffix=output_suffix,
                                     compress=True),
            name='ds_noddi_od',
            run_without_submitting=True)
        workflow.connect(outputnode, 'od_image', ds_od, 'in_file')

        ds_config = pe.Node(
            ReconDerivativesDataSink(extension='.nii.gz',
                                     desc="AMICOconfig",
                                     suffix=output_suffix,
                                     compress=True),
            name='ds_noddi_config',
            run_without_submitting=True)
        workflow.connect(outputnode, 'config_file', ds_config, 'in_file')

    workflow.__desc__ = desc
    return workflow
