from qsiprep.interfaces.bids import QsiReconIngress
qsiprep_output_names = QsiReconIngress().output_spec.class_editable_traits()
anatomical_input_fields = [
    't1_aparc',
    't1_seg',
    't1_aseg',
    't1_brain_mask',
    't1_preproc',
    't1_csf_probseg',
    't1_gm_probseg',
    't1_wm_probseg',
    'left_inflated_surf',
    'left_midthickness_surf',
    'left_pial_surf',
    'left_smoothwm_surf',
    'mrtrix_5tt',
    'right_inflated_surf',
    'right_midthickness_surf',
    'right_pial_surf',
    'right_smoothwm_surf',
    'orig_to_t1_mode_forward_transform',
    't1_2_fsnative_forward_transform',
    't1_2_mni_reverse_transform',
    't1_2_mni_forward_transform',
    'template_brain_mask',
    'template_preproc',
    'template_seg',
    'template_csf_probseg',
    'template_gm_probseg',
    'template_wm_probseg',
]
default_input_set = set(qsiprep_output_names + anatomical_input_fields)
input_fields = list(default_input_set)
default_connections = [(trait, trait) for trait in input_fields]
