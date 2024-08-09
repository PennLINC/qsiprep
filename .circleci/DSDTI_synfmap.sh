#!/bin/bash

cat << DOC

DSCDTI_nofmap test
==================

This tests the following features:
 - A workflow with no distortion correction followed by eddy
 - Eddy is run on a CPU
 - Denoising is skipped

Inputs:
-------

 - DSDTI BIDS data (data/DSDTI)

DOC
set +e

source ./get_data.sh
TESTDIR=${PWD}
TESTNAME=DSDTI_nofmap
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
QSIPREP_CMD=$(run_qsiprep_cmd ${BIDS_INPUT_DIR} ${OUTPUT_DIR})

# CRITICAL: delete the fieldmap data
rm -rf data/DSDTI/sub-PNC/fmap


${QSIPREP_CMD} \
	-w ${TEMPDIR} \
     --eddy-config ${EDDY_CFG} \
     --sloppy \
	--force-syn \
	--b1-biascorrect-stage final \
     --denoise-method none \
	--output-resolution 5 \
	-vv



