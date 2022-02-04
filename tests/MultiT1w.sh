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
BIDS_INPUT_DIR=${TESTDIR}/data/DSDTI
export FS_LICENSE=${TESTDIR}/data/license.txt

# CRITICAL: delete the fieldmap data
rm -rf data/DSDTI/sub-PNC/fmap

# Create a shifted version of the t1w
docker run -u $(id -u) \
    -v ${BIDS_INPUT_DIR}:/BIDS \
    --rm -ti --entrypoint 3dWarp \
    ${IMAGE} \
    -matvec_in2out 'MATRIX(1,0,0,2,0,1,0,4,0,0,1,1)' \
    -gridset /BIDS/sub-PNC/anat/sub-PNC_T1w.nii.gz \
    -prefix /BIDS/sub-PNC/anat/sub-PNC_run-02_T1w.nii.gz \
    /BIDS/sub-PNC/anat/sub-PNC_T1w.nii.gz

cp ${BIDS_INPUT_DIR}/sub-PNC/anat/sub-PNC_T1w.json \
   ${BIDS_INPUT_DIR}/sub-PNC/anat/sub-PNC_run-02_T1w.json

# Do the anatomical run on its own
qsiprep-docker -i ${IMAGE} \
	-e qsiprep_DEV 1 -u $(id -u) \
	--config ${CFG} ${PATCH} -w ${TEMPDIR} \
	 ${BIDS_INPUT_DIR} ${OUTPUT_DIR} \
	 participant \
     --eddy-config ${EDDY_CFG} \
     --denoise-method none \
     --sloppy --mem_mb 4096 \
     --output-space T1w \
     --output-resolution 5 \
     --anat-only \
     --nthreads ${NTHREADS} -vv


# For the run
TESTNAME=Longitudinal
setup_dir ${TESTDIR}/${TESTNAME}
TEMPDIR=${TESTDIR}/${TESTNAME}/work
OUTPUT_DIR=${TESTDIR}/${TESTNAME}/derivatives
qsiprep-docker -i ${IMAGE} \
	-e qsiprep_DEV 1 -u $(id -u) \
	--config ${CFG} ${PATCH} -w ${TEMPDIR} \
	 ${BIDS_INPUT_DIR} ${OUTPUT_DIR} \
	 participant \
     --eddy-config ${EDDY_CFG} \
     --denoise-method none \
     --sloppy --mem_mb 4096 \
     --output-space T1w \
     --output-resolution 5 \
     --anat-only \
     --longitudinal \
     --nthreads ${NTHREADS} -vv


