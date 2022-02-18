from http.server import SimpleHTTPRequestHandler
from qsiprep.interfaces.bids import QsiReconIngress
from nipype.interfaces.io import FreeSurferSource
from qsiprep.interfaces.anatomical import QsiprepAnatomicalIngress
from nipype.interfaces.base import (TraitedSpec, BaseInterfaceInputSpec,
                                    traits, SimpleInterface)

# Anatomical (t1w/t2w) slots
FS_FILES_TO_REGISTER = ["brain", "aseg"]
CREATEABLE_ANATOMICAL_OUTPUTS = [
    "fs_5tt_hsvs", "qsiprep_5tt_hsvs", "qsiprep_5tt_fast",
    "fs_to_qsiprep_transform_itk", "fs_to_qsiprep_transform_mrtrix"]

# These come directly from QSIPrep outputs. They're aligned to the DWIs in AC-PC
qsiprep_anatomical_ingressed_fields = \
    QsiprepAnatomicalIngress.output_spec.class_editable_traits()

# The init_recon_anatomical anatomical workflow can create additional
# anatomical files (segmentations/masks/etc) that can be used downstream.
# These are **independent** of the DWI data and handled separately
anatomical_workflow_outputs = qsiprep_anatomical_ingressed_fields + \
    FS_FILES_TO_REGISTER + CREATEABLE_ANATOMICAL_OUTPUTS

# These are read directly from QSIPrep's dwi results.
qsiprep_output_names = QsiReconIngress().output_spec.class_editable_traits()

# dMRI + registered anatomical fields
recon_workflow_anatomical_input_fields = anatomical_workflow_outputs + [
    "dwi_mask", "atlas_configs", "odf_rois"]

# Check that no conflicts have been introduced
overlapping_names = set(qsiprep_output_names
    ).intersection(recon_workflow_anatomical_input_fields)
if overlapping_names:
    raise Exception("Someone has added overlapping outputs between the anatomical "
        "and dwi inputs: " + " ".join(overlapping_names))
    
recon_workflow_input_fields = qsiprep_output_names + \
    recon_workflow_anatomical_input_fields
default_input_set = set(recon_workflow_input_fields)
default_connections = [(trait, trait) for trait in recon_workflow_input_fields]


class _ReconWorkflowInputsInputSpec(BaseInterfaceInputSpec):
    pass


class _ReconWorkflowInputsOutputSpec(TraitedSpec):
    pass

class ReconWorkflowInputs(SimpleInterface):
    input_spec = _ReconWorkflowInputsInputSpec
    output_spec = _ReconWorkflowInputsOutputSpec

    def _run_interface(self, runtime):
        for name in recon_workflow_input_fields:
            self._results[name] = self.inputs.get(name)
        return runtime

for name in recon_workflow_input_fields:
    _ReconWorkflowInputsInputSpec.add_class_trait(name, traits.Any)
    _ReconWorkflowInputsOutputSpec.add_class_trait(name, traits.Any)