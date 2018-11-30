from collections import defaultdict
from fmriprep.engine import Workflow
import nipype.interfaces.utility as niu
from nipype.interfaces import afni
from nipype.pipeline import engine as pe

from qsiprep.workflows.fieldmap.bidirectional_pepolar import init_bidirectional_b0_unwarping_wf
from qsiprep.interfaces.images import NiftiInfo
from qsiprep.interfaces.images import SplitDWIs



def init_opposite_phase_series_wf(template_plus_pe,
                                  dwi_denoise_window,
                                  output_spaces,
                                  denoise_before_combining,
                                  combine_all_dwis,
                                  name='opposite_phase_series_wf'):
    """Perform HMC and SDC on two DWI series, using b0's from opposite series for SDC.

    sdfg
    """
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['pe_plus_template', 'pe_minus_template', 't1_brain',
                              't1_2_mni_forward_transform']),
        name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['to_unwarp_affines', 'unwarped_b0_ref', ]),
        name='outputnode')

    # Create unwarped template and get warps for plus and minus
    register_b0_refs = init_bidirectional_b0_unwarping_wf(template_plus_pe=template_plus_pe)

    workflow = Workflow(name=name)
    workflow.connect([
        (inputnode, register_b0_refs, [('pe_plus_template', 'inputnode.template_plus')]),
        (inputnode, register_b0_refs, [('pe_minus_template', 'inputnode.template_minus')]),
    ])

    return workflow
