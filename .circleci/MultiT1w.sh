#!/bin/bash

cat << DOC

MultiT1w test
==================

This tests the following features:
 - freesurfer's robust template

Inputs:
-------

 - DSDTI BIDS data (data/DSDTI)

DOC
set +e

source ./get_data.sh
TESTDIR=${PWD}
TESTNAME=MultiT1w
get_config_data ${TESTDIR}
get_bids_data ${TESTDIR} DSDTI
CFG=${TESTDIR}/data/nipype.cfg
EDDY_CFG=${TESTDIR}/data/eddy_config.json

# For the run
setup_dir ${TESTDIR}/${TESTNAME}
TEMPDIR=${TESTDIR}/${TESTNAME}/work
OUTPUT_DIR=${TESTDIR}/${TESTNAME}/derivatives
BIDS_INPUT_DIR=${TESTDIR}/data/DSDTI/DSDTI
export FS_LICENSE=${TESTDIR}/data/license.txt
QSIPREP_CMD=$(run_qsiprep_cmd ${BIDS_INPUT_DIR} ${OUTPUT_DIR})

# CRITICAL: delete the fieldmap data
rm -rf data/DSDTI/DSDTI/sub-PNC/fmap

# Create a shifted version of the t1w
if [[ "${IN_CI}" = 'true' ]]; then
    3dWarp \
        -matvec_in2out 'MATRIX(1,0,0,2,0,1,0,4,0,0,1,1)' \
        -gridset ${BIDS_INPUT_DIR}/sub-PNC/anat/sub-PNC_T1w.nii.gz \
        -prefix ${BIDS_INPUT_DIR}/sub-PNC/anat/sub-PNC_run-02_T1w.nii.gz \
        ${BIDS_INPUT_DIR}/sub-PNC/anat/sub-PNC_T1w.nii.gz
else
    docker run -u $(id -u) \
        -v ${BIDS_INPUT_DIR}:/BIDS \
        --rm -ti --entrypoint 3dWarp \
        ${IMAGE} \
        -matvec_in2out 'MATRIX(1,0,0,2,0,1,0,4,0,0,1,1)' \
        -gridset /BIDS/sub-PNC/anat/sub-PNC_T1w.nii.gz \
        -prefix /BIDS/sub-PNC/anat/sub-PNC_run-02_T1w.nii.gz \
        /BIDS/sub-PNC/anat/sub-PNC_T1w.nii.gz

fi

cp ${BIDS_INPUT_DIR}/sub-PNC/anat/sub-PNC_T1w.json \
   ${BIDS_INPUT_DIR}/sub-PNC/anat/sub-PNC_run-02_T1w.json

# Do the anatomical run on its own
${QSIPREP_CMD} \
	 -w ${TEMPDIR} \
     --eddy-config ${EDDY_CFG} \
     --denoise-method none \
     --sloppy \
     --output-resolution 5 \
     --anat-only \
     -vv


# Explicitly test --subject-anatomical-reference unbiased
TESTNAME=Longitudinal
setup_dir ${TESTDIR}/${TESTNAME}
TEMPDIR=${TESTDIR}/${TESTNAME}/work
OUTPUT_DIR=${TESTDIR}/${TESTNAME}/derivatives

${QSIPREP_CMD} \
	 -w ${TEMPDIR} \
     --eddy-config ${EDDY_CFG} \
     --denoise-method none \
     --sloppy \
     --output-resolution 5 \
     --anat-only \
     --subject-anatomical-reference unbiased \
     -vv


