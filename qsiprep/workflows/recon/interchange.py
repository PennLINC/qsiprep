from qsiprep.interfaces.bids import QsiReconIngress
from nipype.interfaces.io import FreeSurferSource
from qsiprep.interfaces.anatomical import QsiprepAnatomicalIngress

# Anatomical (t1w/t2w) slots
FS_FILES_TO_REGISTER = ["brain", "aparc_aseg", "brainmask", "aseg"]
CREATEABLE_ANATOMICAL_OUTPUTS = [
    "fs_5tt_hsvs", "qsiprep_5tt_hsvs", "qsiprep_5tt_fast"]
freesurfer_output_names = FreeSurferSource.output_spec.class_editable_traits()
anatomical_input_fields = QsiprepAnatomicalIngress.output_spec.class_editable_traits()

# dMRI slots
qsiprep_output_names = QsiReconIngress().output_spec.class_editable_traits()

default_input_set = set(
    qsiprep_output_names + anatomical_input_fields +
    FS_FILES_TO_REGISTER + CREATEABLE_ANATOMICAL_OUTPUTS )
input_fields = sorted(default_input_set)
default_connections = [(trait, trait) for trait in input_fields]