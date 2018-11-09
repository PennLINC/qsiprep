from qsiprep.utils.bids import collect_data
subject_data, layout = collect_data('/Users/mcieslak/projects/test_bids_data/crash', '0001a')
# Handle the grouping of multiple dwi files within a session
sessions = layout.get_sessions()
all_dwis = subject_data['dwi']
dwi_groups = []
if sessions:
    for session in sessions:
        dwi_groups.append([img for img in all_dwis if 'ses-'+session in img])
else:
    dwi_groups = [all_dwis]
