#!/usr/bin/env bash
SCRIPT=${HOME}/projects/qsiprep/wrapper/qsiprep_docker.py
INPUT_DIR=${HOME}/projects/test_bids_data/downsampled/abcd

DEV_DIR=${HOME}/projects/qsiprep
QSIPREP_OUTPUT_DIR=${DEV_DIR}/scratch/abcd_test/test_output/qsiprep
WORK_DIR=${OUTPUT_DIR}/recon_work
SPEC_FILE=${DEV_DIR}/scratch/dsistudio_gqi_ctl.json
OUTPUT_DIR=${DEV_DIR}/scratch/downsampled_abcd_recon

python3 ${HOME}/projects/qsiprep/wrapper/qsiprep_docker.py \
    --patch-qsiprep ${DEV_DIR}/qsiprep \
    --stop-on-first-crash \
    --omp-nthreads 1 \
    --bids-dir ${INPUT_DIR} \
    --recon_input ${QSIPREP_OUTPUT_DIR} \
    --recon_spec ${SPEC_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --analysis-level participant \
    --fs-license-file /Applications/freesurfer/license.txt
