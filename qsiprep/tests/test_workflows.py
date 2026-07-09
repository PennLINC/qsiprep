from pathlib import Path
from qsiprep.workflows.anatomical.volume import init_synb0_anat_wf

input_nii = "/data/qsiprep_testing_data/anat_preprocessing/" \
    "rigid_acpc_resample_brain_t1w/masked_brain_trans.nii"

wf = init_synb0_anat_wf()
wf.inputs.inputnode.t1w_brain_acpc = input_nii
wf.base_dir = "/data/qsiprep_testing/synb0"
wf.run()

