# Walk through grouping logic

The things we need to account for:

1. Curator-defined metadata to determine which files should be concatenated together *at some point* (i.e., MultipartID).
    - If files share a MultipartID, they should be concatenated together, but not with other files.
    - Should this affect field map grouping?
2. Curator-defined metadata to determine how field maps should be constructed (e.g., IntendedFor, B0FieldSource/B0FieldIdentifier).
    - If files have IntendedFor/B0FieldSource/B0FieldIdentifier, those should be used to define the field maps rather than QSIPrep's heuristic.
3. Entities in filenames indicating that files should not be concatenated.
    - Currently, the only entities we account for are subject and session.
4. Whether the files can be concatenated or not.
    - The ShimSetting metadata field.
    - TotalReadoutTime.
        - If metadata (MultipartID, B0FieldSource, B0FieldIdentifier, IntendedFor) conflict with ShimSetting or TotalReadoutTime, raise an exception.
5. User-provided settings that control how files are concatenated and what distortion correction method is used.
    - separate_all_dwis: If True, do not concatenate files, but do use combinations of files to create field maps.
        - If separate_all_dwis and MultipartID conflict, defer to separate_all_dwis but raise a warning.
    - ignore: If "fieldmaps" provided, do not use files in `fmap` directory to define field maps. DWI-based field maps will still be used.
        - If "fieldmaps" in ignore and fmap files have IntendedFor/B0FieldIdentifier, defer to ignore.
    - pepolar_method: If "drbuddi" is used, only use reverse phase-encoded files to define field maps.
        - If pepolar_method and distortion correction-related metadata (e.g., B0FieldSource/B0FieldIdentifier specify a B0Field using multiple PEDs) conflict, raise an exception for now, but plan to support if someone asks for it.


## List of Relevant Things

- Subject entity: Impacts all groups
- Session entity: Impacts all groups
- ShimSetting: Impacts "Distortion Groups" and "Field Map * Groups"
- TotalReadoutTime: Impacts "Distortion Groups" and "Field Map * Groups" (maybe, ask Okan)
- B0FieldSource: Impacts "Field Map Estimation Groups"
- B0FieldIdentifier: Impacts "Field Map Application Groups"
- IntendedFor: Impacts both "Field Map Estimation Groups" and "Field Map Application Groups"
- PhaseEncodingDirection: Impacts "Distortion Groups"
- MultipartID: Impacts "Concatenation Groups"
- `--separate-all-dwis`: Impacts "Distortion Groups" and "Concatenation Groups"
- `--denoise-before-combining`: Hopefully doesn't impact grouping- just order of steps in workflow.
- `--pepolar-method`: Impacts "Field Map Estimation Groups" and "Field Map Application Groups". Could be changed to `--estimate-per-axis True|False` internally.
- `--ignore`: Impacts whether fmap files are used for field map estimation groups.

## Outputs

1. Field Map Estimation Groups: List of valid field maps and the files used to create them.
2. Field Map Application Groups: Links from valid field maps to target DWI files or vice versa.
3. Distortion Groups: Groups of DWI files to be concatenated prior to denoising.
3. Concatenation Groups: Groups of DWI files to be concatenated or averaged in the output directory.

## Steps

1. Build a list of valid field maps from available data.
    - The goal here is to mimic SDCFlows' output.
      Something like this:
      ```
      FieldmapEstimation(sources=<2 files>, method=<EstimatorType.PEPOLAR: 2>, bids_id='auto0001')
        j-      fmap/sub-01_dir-AP_epi.nii.gz
        j       fmap/sub-01_dir-PA_epi.nii.gz
      FieldmapEstimation(sources=<2 files>, method=<EstimatorType.PEPOLAR: 2>, bids_id='auto0002')
        j-      dwi/sub-01_dir-AP_run-2_dwi.nii.gz
        j       dwi/sub-01_dir-PA_run-2_dwi.nii.gz
      ```

    - Problem: There could be many possible field map combinations. For example, with a multi-run, multi-PED dataset (two DWI runs per PED, AP/PA/RL/LR PEDs), we'd have AP run-01/PA run-01, AP run-01/PA run-02, AP run-01/PA run-01/RL run-01, etc. We need to just group everything that can be used to estimate a field map together.

2.

## Questions

1. Distortion correction always happens before concatenation, right?
2. Should MultipartID impact Distortion Groups?


## Scenarios

### Scenario 1: Multi-run, multi-PED dataset

```
multirun_multiped/
├── dataset_description.json
└── sub-01
    ├── anat
    │   └── sub-01_T1w.nii.gz
    └── dwi
        ├── sub-01_dir-AP_run-1_dwi.json
        ├── sub-01_dir-AP_run-1_dwi.nii.gz
        ├── sub-01_dir-AP_run-2_dwi.json
        ├── sub-01_dir-AP_run-2_dwi.nii.gz
        ├── sub-01_dir-LR_run-1_dwi.json
        ├── sub-01_dir-LR_run-1_dwi.nii.gz
        ├── sub-01_dir-LR_run-2_dwi.json
        ├── sub-01_dir-LR_run-2_dwi.nii.gz
        ├── sub-01_dir-PA_run-1_dwi.json
        ├── sub-01_dir-PA_run-1_dwi.nii.gz
        ├── sub-01_dir-PA_run-2_dwi.json
        ├── sub-01_dir-PA_run-2_dwi.nii.gz
        ├── sub-01_dir-RL_run-1_dwi.json
        ├── sub-01_dir-RL_run-1_dwi.nii.gz
        ├── sub-01_dir-RL_run-2_dwi.json
        └── sub-01_dir-RL_run-2_dwi.nii.gz
```


#### Scenario 1A: Everything plays nicely

There is no metadata specifying field map groups or concatenation groups, so QSIPrep uses its default heuristics.

Four distortion groups:

- sub-01_dir-AP
    - sub-01_dir-AP_run-1_dwi.nii.gz
    - sub-01_dir-AP_run-2_dwi.nii.gz
- sub-01_dir-PA
    - sub-01_dir-PA_run-1_dwi.nii.gz
    - sub-01_dir-PA_run-2_dwi.nii.gz
- sub-01_dir-LR
    - sub-01_dir-LR_run-1_dwi.nii.gz
    - sub-01_dir-LR_run-2_dwi.nii.gz
- sub-01_dir-RL
    - sub-01_dir-RL_run-1_dwi.nii.gz
    - sub-01_dir-RL_run-2_dwi.nii.gz

One field map estimation group:

- TOPUP
    - sub-01_dir-AP_run-1_dwi.nii.gz
    - sub-01_dir-AP_run-2_dwi.nii.gz
    - sub-01_dir-PA_run-1_dwi.nii.gz
    - sub-01_dir-PA_run-2_dwi.nii.gz
    - sub-01_dir-LR_run-1_dwi.nii.gz
    - sub-01_dir-LR_run-2_dwi.nii.gz
    - sub-01_dir-RL_run-1_dwi.nii.gz
    - sub-01_dir-RL_run-2_dwi.nii.gz

One concatenation group:

- sub-01
    - sub-01_dir-AP_dwi.nii.gz
    - sub-01_dir-PA_dwi.nii.gz
    - sub-01_dir-LR_dwi.nii.gz
    - sub-01_dir-RL_dwi.nii.gz


#### Scenario 1B: Distortion groups don't match field map estimation groups

We have the same data, but the curator specified that the run-1 files should be used to create one field map and the run-2 files should be used to create another.
We should allow this without raising warnings.

QSIPrep will need to account for the field map estimation groups when designing the distortion groups.

Eight distortion groups:

- sub-01_dir-AP_run-1_dwi.nii.gz
- sub-01_dir-AP_run-2_dwi.nii.gz
- sub-01_dir-PA_run-1_dwi.nii.gz
- sub-01_dir-PA_run-2_dwi.nii.gz
- sub-01_dir-LR_run-1_dwi.nii.gz
- sub-01_dir-LR_run-2_dwi.nii.gz
- sub-01_dir-RL_run-1_dwi.nii.gz
- sub-01_dir-RL_run-2_dwi.nii.gz

Two field map estimation groups:

- sub-01_run-1
    - sub-01_dir-AP_run-1_dwi.nii.gz
    - sub-01_dir-PA_run-1_dwi.nii.gz
    - sub-01_dir-LR_run-1_dwi.nii.gz
    - sub-01_dir-RL_run-1_dwi.nii.gz
- sub-01_run-2
    - sub-01_dir-AP_run-2_dwi.nii.gz
    - sub-01_dir-PA_run-2_dwi.nii.gz
    - sub-01_dir-LR_run-2_dwi.nii.gz
    - sub-01_dir-RL_run-2_dwi.nii.gz

One concatenation group

- sub-01
    - sub-01_dir-AP_run-1_dwi.nii.gz
    - sub-01_dir-AP_run-2_dwi.nii.gz
    - sub-01_dir-PA_run-1_dwi.nii.gz
    - sub-01_dir-PA_run-2_dwi.nii.gz
    - sub-01_dir-LR_run-1_dwi.nii.gz
    - sub-01_dir-LR_run-2_dwi.nii.gz
    - sub-01_dir-RL_run-1_dwi.nii.gz
    - sub-01_dir-RL_run-2_dwi.nii.gz


#### Scenario 1C: Concatenation groups split up field map estimation groups

The curator specified that all eight runs should be used to create one field map, but MultiPartID specifies run-1 files are one group and run-2 files are another.

Let's just raise an exception here, since field map estimation groups are not subsets of concatenation groups.

Four distortion groups:

- sub-01_dir-AP
    - sub-01_dir-AP_run-1_dwi.nii.gz
    - sub-01_dir-AP_run-2_dwi.nii.gz
- sub-01_dir-PA
    - sub-01_dir-PA_run-1_dwi.nii.gz
    - sub-01_dir-PA_run-2_dwi.nii.gz
- sub-01_dir-LR
    - sub-01_dir-LR_run-1_dwi.nii.gz
    - sub-01_dir-LR_run-2_dwi.nii.gz
- sub-01_dir-RL
    - sub-01_dir-RL_run-1_dwi.nii.gz
    - sub-01_dir-RL_run-2_dwi.nii.gz

One field map estimation group:

- TOPUP
    - sub-01_dir-AP_run-1_dwi.nii.gz
    - sub-01_dir-AP_run-2_dwi.nii.gz
    - sub-01_dir-PA_run-1_dwi.nii.gz
    - sub-01_dir-PA_run-2_dwi.nii.gz
    - sub-01_dir-LR_run-1_dwi.nii.gz
    - sub-01_dir-LR_run-2_dwi.nii.gz
    - sub-01_dir-RL_run-1_dwi.nii.gz
    - sub-01_dir-RL_run-2_dwi.nii.gz

Two concatenation groups (raise an error)

- sub-01_run-1
    - sub-01_dir-AP_run-1_dwi.nii.gz
    - sub-01_dir-PA_run-1_dwi.nii.gz
    - sub-01_dir-LR_run-1_dwi.nii.gz
    - sub-01_dir-RL_run-1_dwi.nii.gz
- sub-01_run-2
    - sub-01_dir-AP_run-2_dwi.nii.gz
    - sub-01_dir-PA_run-2_dwi.nii.gz
    - sub-01_dir-LR_run-2_dwi.nii.gz
    - sub-01_dir-RL_run-2_dwi.nii.gz


#### Scenario 1D: Field map groups split up distortion groups

The automatic distortion group collection conflicts with the field map groups defined by the curator.

For example, let's say that the B0Field* specify that run-1 files should be used for one field map and run-2 ones should be used for another.

I think we want the distortion groups to reflect the field map estimation groups, but that's difficult.

Four distortion groups:

- sub-01_dir-AP
    - sub-01_dir-AP_run-1_dwi.nii.gz
    - sub-01_dir-AP_run-2_dwi.nii.gz
- sub-01_dir-PA
    - sub-01_dir-PA_run-1_dwi.nii.gz
    - sub-01_dir-PA_run-2_dwi.nii.gz
- sub-01_dir-LR
    - sub-01_dir-LR_run-1_dwi.nii.gz
    - sub-01_dir-LR_run-2_dwi.nii.gz
- sub-01_dir-RL
    - sub-01_dir-RL_run-1_dwi.nii.gz
    - sub-01_dir-RL_run-2_dwi.nii.gz

Two field map estimation groups:

- TOPUP_run1
    - sub-01_dir-AP_run-1_dwi.nii.gz
    - sub-01_dir-PA_run-1_dwi.nii.gz
    - sub-01_dir-LR_run-1_dwi.nii.gz
    - sub-01_dir-RL_run-1_dwi.nii.gz
- TOPUP_run2
    - sub-01_dir-AP_run-2_dwi.nii.gz
    - sub-01_dir-PA_run-2_dwi.nii.gz
    - sub-01_dir-LR_run-2_dwi.nii.gz
    - sub-01_dir-RL_run-2_dwi.nii.gz

Two concatenation groups (raise an error)

- sub-01_run-1
    - sub-01_dir-AP_run-1_dwi.nii.gz
    - sub-01_dir-PA_run-1_dwi.nii.gz
    - sub-01_dir-LR_run-1_dwi.nii.gz
    - sub-01_dir-RL_run-1_dwi.nii.gz
- sub-01_run-2
    - sub-01_dir-AP_run-2_dwi.nii.gz
    - sub-01_dir-PA_run-2_dwi.nii.gz
    - sub-01_dir-LR_run-2_dwi.nii.gz
    - sub-01_dir-RL_run-2_dwi.nii.gz


