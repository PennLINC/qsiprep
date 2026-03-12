# Walk through grouping logic

NOTE: The specific concatenated filenames in this document are not necessarily correct. They are just attempts to follow the logic.

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
    - pepolar_method: If "drbuddi" is used, only use reverse phase-encoded files to define field maps (i.e., `estimate_per_axis=True`).
        - If `estimate_per_axis=True` and B0FieldSource/B0FieldIdentifier specify a B0Field using multiple PEDs, raise an exception for now, but plan to support (by splitting up the B0FieldIdentifier into per-axis versions) if someone asks for it.


## List of Relevant Things

- Subject entity: Impacts all groups. Files should not be combined in a distortion group, field map estimation group, field map application group, or concatenation group if they are from different subjects.
- Session entity: Impacts all groups. Files should not be combined in a distortion group, field map estimation group, field map application group, or concatenation group if they are from different sessions.
- ShimSetting: Impacts "Distortion Groups" and "Field Map * Groups"
- TotalReadoutTime: Impacts "Distortion Groups" and "Field Map * Groups"
- B0FieldSource: Impacts "Field Map Estimation Groups"
- B0FieldIdentifier: Impacts "Field Map Application Groups"
- IntendedFor: Impacts both "Field Map Estimation Groups" and "Field Map Application Groups"
- PhaseEncodingDirection: Impacts "Distortion Groups"
- MultipartID: Impacts "Concatenation Groups"
- `--separate-all-dwis`: Impacts "Distortion Groups" and "Concatenation Groups"
- `--denoise-before-combining`: Hopefully doesn't impact grouping- just order of steps in workflow.
- `--pepolar-method`: Impacts "Field Map Estimation Groups" and "Field Map Application Groups". Changed to `estimate_per_axis True|False` parameter internally. `True` if DRBUDDI in `pepolar-method` and `False` otherwise.
- `--ignore`: Impacts whether fmap files are used for field map estimation groups.

## Outputs

1.  Distortion Groups: Groups of DWI files to be concatenated prior to denoising.
    A dictionary of unique concatenated filenames and the files used to create them.
    For example:
    ```
    {
        'sub-01_dir-AP': [
            '/path/to/dwi/sub-01_dir-AP_run-1_dwi.nii.gz',
            '/path/to/dwi/sub-01_dir-AP_run-2_dwi.nii.gz',
        ],
        'sub-01_dir-PA': [
            '/path/to/dwi/sub-01_dir-PA_run-1_dwi.nii.gz',
            '/path/to/dwi/sub-01_dir-PA_run-2_dwi.nii.gz',
        ],
        'sub-01_dir-LR': [
            '/path/to/dwi/sub-01_dir-LR_run-1_dwi.nii.gz',
            '/path/to/dwi/sub-01_dir-LR_run-2_dwi.nii.gz',
        ],
        'sub-01_dir-RL': [
            '/path/to/dwi/sub-01_dir-RL_run-1_dwi.nii.gz',
            '/path/to/dwi/sub-01_dir-RL_run-2_dwi.nii.gz',
        ],
        # One run doesn't match other files and is its own distortion group.
        'sub-01_acq-unique_dir-AP_run-3': [
            '/path/to/dwi/sub-01_acq-unique_dir-AP_run-3_dwi.nii.gz',
        ],
    }
    ```
2.  Field Map Estimation Groups: List of valid field maps and the files used to create them.
    A dictionary of valid field map estimators from available data.
    Keys are field map identifiers (B0FieldIdentifier when available, auto_000X when not).
    Values are lists of files used to create the field map.
    Since field map estimation happens after distortion groups are created and denoised,
    the DWI files in the values should be distortion group IDs.

    For example:
    ```
    {
        # The files in the AP, PA, LR, and RL distortion groups all have B0FieldIdentifier/B0FieldSource
        # indicating that they belong together.
        'b0identifier01': ['sub-01_dir-AP', 'sub-01_dir-PA', 'sub-01_dir-RL', 'sub-01_dir-LR'],
        # The unique run has a B0FieldIdentifier/B0FieldSource pairing it with an EPI file in fmap/
        'b0identifier02': ['/path/to/fmap/sub-01_dir-AP_epi.nii.gz', 'sub-01_acq-unique_dir-AP_run-3'],
    }
    ```

    If IntendedFor is used to link fmaps to DWI files instead of B0FieldSource/B0FieldIdentifier,
    the key will be auto000X:
    ```
    {
        'auto0001': ['/path/to/fmap/sub-01_dir-AP_epi.nii.gz', 'sub-01_acq-unique_dir-AP_run-3'],
    }
    ```

    Alternatively, if no curator-generated metadata fields are present to define the field maps,
    QSIPrep will automatically group them based on whether they have appropriate metadata, like ShimSetting
    and TotalReadoutTime:
    ```
    # The files in the LR and RL distortion groups do not have metadata linking them,
    # so they're grouped automatically by
    'auto_00001': ['sub-01_dir-AP', 'sub-01_dir-PA', 'sub-01_dir-RL', 'sub-01_dir-LR']
    # sub-01_acq-unique_dir-AP_run-3 is not linked to any field map because fmap-dwi field maps required
    # some curator-generated metadata. Instead, Syn-SDC may be applied, depending on how QSIPrep is called.
    ```
3.  Field Map Application Groups: Links from valid field maps to target DWI files or vice versa.
    A dictionary of field map identifiers and the DWI files that will use them for distortion correction.
    Keys are field map identifiers (B0FieldIdentifier when available, auto_000X when not).
    Values are lists of distortion group IDs that will be distortion-corrected
    using the field map.
    Since field map estimation happens after distortion groups are created and denoised,
    the DWI files in the values should be distortion group IDs.
    For example:
    ```
    {
        # The files in the AP, PA, LR, and RL distortion groups all have B0FieldIdentifier/B0FieldSource
        # indicating that they belong together.
        'b0identifier01': ['sub-01_dir-AP', 'sub-01_dir-PA', 'sub-01_dir-RL', 'sub-01_dir-LR'],
        # The fmap file is not a target for distortion correction.
        'b0identifier02': ['sub-01_acq-unique_dir-AP_run-3'],
    }
    ```
4.  Concatenation Groups: Groups of DWI files to be concatenated or averaged in the output directory.
    A dictionary of unique concatenated filenames and the distortion groups used to create them.
    For example:
    ```
    {
        'sub-01': ['sub-01_dir-AP', 'sub-01_dir-PA', 'sub-01_dir-LR', 'sub-01_dir-RL'],
        'sub-01_acq-unique_dir-AP_run-3': ['sub-01_acq-unique_dir-AP_run-3'],
    }
    ```

## Steps

I'm not sure what order the outputs should be generated in.
Distortion groups are applied first (concatenated, then denoised), then field map groups, then concatenation groups.
However, distortion groups should be subsets of field map groups, so we need to know the field map groups before we can finalize the distortion groups, and field map groups should be subsets of concatenation groups, so we need to know the concatenation groups before we can finalize the distortion groups.


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
    - sub-01_dir-AP
    - sub-01_dir-PA
    - sub-01_dir-LR
    - sub-01_dir-RL

One concatenation group:

- sub-01
    - sub-01_dir-AP
    - sub-01_dir-PA
    - sub-01_dir-LR
    - sub-01_dir-RL


#### Scenario 1B: Distortion groups don't match field map estimation groups

We have the same data, but the curator specified (via B0FieldIdentifier/B0FieldSource) that the run-1 files should be used to create one field map and the run-2 files should be used to create another.
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

- topuprun01 (B0FieldIdentifier key)
    - sub-01_dir-AP_run-1_dwi.nii.gz
    - sub-01_dir-PA_run-1_dwi.nii.gz
    - sub-01_dir-LR_run-1_dwi.nii.gz
    - sub-01_dir-RL_run-1_dwi.nii.gz
- topuprun02 (B0FieldIdentifier key)
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
    - sub-01_dir-AP
    - sub-01_dir-PA
    - sub-01_dir-LR
    - sub-01_dir-RL

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


## Gaps and Inconsistencies

The following issues were identified during review of the logic above.
They should be resolved before or during implementation.

1. **No algorithm for computation order.**
   The document acknowledges (line 137) uncertainty about the order. The proposed order is:
   distortion groups -> field map estimation groups -> field map application groups ->
   concatenation groups -> validate consistency -> refine distortion groups (split any
   distortion group that spans multiple field map estimation groups).

2. **Missing scenarios.** The document does not cover:
   - `separate_all_dwis=True` (each file stays separate).
   - `ignore_fieldmaps=True` when fmap/ files with IntendedFor/B0FieldIdentifier exist.
   - Missing PhaseEncodingDirection metadata.
   - Non-DWI fieldmaps in fmap/ (phasediff, phase1/phase2).
   - Interaction between `estimate_per_axis=True` and B0FieldIdentifier-based grouping.

3.  **`denoise_before_combining` interaction is not discussed.**
    Line 39 correctly notes it "hopefully doesn't impact grouping," but if denoising
    happens before distortion-group concatenation, the operational meaning of
    "distortion group" changes. This should be clarified or confirmed as out of scope.
