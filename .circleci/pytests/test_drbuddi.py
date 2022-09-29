from qsiprep.workflows.base import (
    collect_data, group_dwi_scans, defaultdict,
    get_source_file)
from qsiprep.workflows.dwi.pre_hmc import init_dwi_pre_hmc_wf
from pprint import pprint

subject_data, layout = collect_data(
    "data/tinytensor",
    "tinytensors",
    filters=None)

pprint(subject_data)

hmc_model = "eddy"
combine_all_dwis = True
ignore = []
merging_distortion_groups = True

dwi_fmap_groups, concatenation_scheme = group_dwi_scans(
    layout, subject_data,
    using_fsl=hmc_model == 'eddy',
    combine_scans=combine_all_dwis,
    ignore_fieldmaps="fieldmaps" in ignore,
    concatenate_distortion_groups=merging_distortion_groups)
print("\n\n\n\n")
pprint(dwi_fmap_groups)
print("\n\n\n\n")
pprint(concatenation_scheme)

# create a mapping of which across-distortion-groups are contained in each merge
merged_group_names = sorted(set(concatenation_scheme.values()))
merged_to_subgroups = defaultdict(list)
for subgroup_name, destination_name in concatenation_scheme.items():
    merged_to_subgroups[destination_name].append(subgroup_name)

print(merged_group_names, "\n\n\n")
pprint(merged_to_subgroups)

merged_to_subgroups['sub-tinytensors']
outputs_to_files = {dwi_group['concatenated_bids_name']: dwi_group
                    for dwi_group in dwi_fmap_groups}
pprint(outputs_to_files)

(output_fname, dwi_info), = outputs_to_files.items()

# This file doesn't exist! It is only used for naming the merged output.
source_file = get_source_file(dwi_info['dwi_series'], output_fname, suffix="_dwi")
print(source_file, "\n\n")
output_wfname = output_fname.replace('-', '_')
print(output_wfname, "\n\n")
print(output_fname, "\n\n")
pprint(dwi_info)


scan_groups=dwi_info
output_prefix=output_fname
layout=layout
ignore=ignore
b0_threshold=100
dwi_denoise_window=3
denoise_method="none"
unringing_method="none"
dwi_no_biascorr=False
no_b0_harmonization=False
denoise_before_combining=True
hmc_model="eddy"
eddy_config=None
raw_image_sdc=False
reportlets_dir='reports'
omp_nthreads=2
low_mem=False
sloppy=False
source_file=source_file


all_dwis = scan_groups['dwi_series']
fieldmap_info = scan_groups['fieldmap_info']
dwi_metadata = layout.get_metadata(all_dwis[0])
fieldmap_type = fieldmap_info['suffix']
doing_bidirectional_pepolar = fieldmap_type == 'rpe_series'
preprocess_rpe_series = doing_bidirectional_pepolar and hmc_model == 'eddy'

pre_hmc_wf = init_dwi_pre_hmc_wf(scan_groups=scan_groups,
                                 b0_threshold=b0_threshold,
                                 preprocess_rpe_series=preprocess_rpe_series,
                                 dwi_denoise_window=dwi_denoise_window,
                                 denoise_method=denoise_method,
                                 unringing_method=unringing_method,
                                 dwi_no_biascorr=dwi_no_biascorr,
                                 no_b0_harmonization=no_b0_harmonization,
                                 orientation='LAS' if hmc_model == 'eddy' else 'LPS',
                                 source_file=source_file,
                                 low_mem=low_mem,
                                 denoise_before_combining=denoise_before_combining,
                                 omp_nthreads=omp_nthreads)

