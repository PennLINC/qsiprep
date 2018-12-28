#!/usr/bin/env bash
SCRIPT=${HOME}/projects/qsiprep/wrapper/qsiprep_docker.py
INPUT_DIR=${HOME}/projects/test_bids_data/downsampled/abcd

DEV_DIR=${HOME}/projects/qsiprep
OUTPUT_DIR=${DEV_DIR}/scratch/downsampled_abcd
WORK_DIR=${OUTPUT_DIR}/work


python3 ${HOME}/projects/qsiprep/wrapper/qsiprep_docker.py \
    --patch-qsiprep ${DEV_DIR}/qsiprep \
    --stop-on-first-crash \
    --b0-to-t1w-transform Rigid \
    --dwi_denoise_window 7 \
    --template MNI152NLin2009cAsym \
    --output-resolution 1.7 \
    --b0-motion-corr-to iterative \
    --hmc-model 3dSHORE \
    --impute-slice-threshold 0.0 \
    --work-dir ${WORK_DIR} \
    --n_cpus 1 \
    -v -v \
    --shell \
    --omp-nthreads 1 \
    --sloppy \
    --bids-dir ${INPUT_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --analysis-level participant \
    --fs-license-file /Applications/freesurfer/license.txt \
    --output-space T1w template
