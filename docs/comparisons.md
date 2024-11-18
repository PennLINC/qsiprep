# Comparisons to other pipelines

Other pipelines for preprocessing DWI data are currently being developed. Below are tables comparing
their current feature sets. These other

 * [Tractoflow](https://doi.org/10.1016/j.neuroimage.2020.116889)
 * [PreQual](https://doi.org/10.1101/2020.09.14.260240)
 * [MRtrix3_connectome](https://github.com/BIDS-Apps/MRtrix3_connectome)
 * [dMRIPrep](https://github.com/nipreps/dmriprep)


## Supported Sampling Schemes

|                             | QSIPrep | Tractoflow | PreQual | MRtrix3_connectome | dMRIPrep |
| --------------------------- | :-----: | :--------: | :-----: | :----------------: | :------: |
| Single Shell                |    ✔    |     ✔      |    ✔    |         ✔          |    ✔     |
| Multi Shell                 |    ✔    |     ✔      |    ✔    |         ✔          |    ✔     |
| Cartesian                   |    ✔    |     ✘      |    ✘    |         ✘          |    ✘     |
| Random (Compressed Sensing) |    ✔    |     ✘      |    ✘    |         ✘          |    ✘     |

## Preprocessing

|                                                     | QSIPrep                  | Tractoflow             | PreQual                 | MRtrix3_connectome | dMRIPrep         |
| --------------------------------------------------- | :----------------------: | :--------------------: | :---------------------: | :----------------: | :--------------: |
| BIDS App                                            |            ✔             |           ✔            |            ✘            |         ✔          |        ✔         |
| Gradient direction sanity check                     |     Q-form matching      |           ✘            |     `dwigradcheck`      |         ✘          |        ✘         |
| Workflow management                                 |          NiPyPe          |        NextFlow        |         Custom          |       Custom       |      NiPyPe      |
| MP-PCA denoising                                    |            ✔             |           ✔            |            ✔            |         ✔          |        ✘         |
| Patch2self denoising                                |            ✔             |           ✘            |            ✘            |         ✘          |        ✘         |
| Gibbs unringing                                     |       `mrdegibbs`        | `mrdegibbs` (disabled) | `mrdegibbs` (disabled)  |    `mrdegibbs`     |        ✘         |
| B1 bias field correction                            |            N4            |           N4           |           N4            |         N4         |        ✘         |
| Automatic distortion group concatenation            |            ✔             |           ✘            |            ✘            |         ✔          |        ✘         |
| T1w brain extraction                                |           ANTs           |          ANTs          |            ✘            |         ✘          |       ANTs       |
| Intensity normalization                             |  scaled by *b*=0 means   | *b*=0 mean set to 1000 |            ✘            |   `mtnormalize`    |        ✘         |
| b=0 to T1w coregistration                           | ANTs linear registration |  ANTs Non-Linear SyN   |            ✘            |    `mrregister`    |     FSL BBR      |
| Head Motion Correction (shelled schemes)            |          `eddy`          |         `eddy`         |         `eddy`          |       `eddy`       |        ✘         |
| Head Motion Correction (Cartesian / Random Schemes) |        SHORELine         |           ✘            |            ✘            |         ✘          |        ✘         |
| PEPOLAR EPI distortion correction                   |         `TOPUP`          |        `TOPUP`         |         `TOPUP`         |      `TOPUP`       |     `TOPUP`      |
| GRE Fieldmap EPI distortion correction              |            ✔             |           ✘            |            ✘            |         ✘          |        ✔         |
| Fieldmapless Distortion Correction                  |     PE-Direction SyN     |           ✘            | Non-Linear registration |    SyN b0-DISCO    | PE-Direction SyN |
| T1w-based Normalization                             |        ANTs (SyN)        |           ✘            |            ✘            |         ✘          |    ANTs (SyN)    |
| HTML Report                                         |            ✔             |           ✘            |            ✔            |         ✘          |        ✔         |
| Containerized                                       |            ✔             |           ✔            |            ✔            |         ✔          |        ✔         |

## QC

|                               | QSIPrep           | Tractoflow | PreQual | MRtrix3_connectome |
| ----------------------------- | :---------------: | :--------: | :-----: | :----------------: |
| Automated methods boilerplate |         ✔         |     ✘      |    ✘    |         ✘          |
| HTML Preprocessing Report     | [NiWorkflows-based](preprocessing.html#visual-reports) |     ✘      | Custom  |      EddyQuad      |
