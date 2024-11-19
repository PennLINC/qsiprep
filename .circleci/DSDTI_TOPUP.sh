#!/bin/bash

cat << DOC

DSCDTI_TOPUP test
=================

This tests the following features:
 - TOPUP on a single-shell sequence
 - Eddy is run on a CPU
 - mrdegibbs is run

Inputs:
-------

 - DSDTI BIDS data (data/DSDTI)

DOC

set +e
source ./get_data.sh
TESTDIR=${PWD}
TESTNAME=DSDTI_TOPUP
get_config_data ${TESTDIR}
#get_bids_data ${TESTDIR} DSDTI
CFG=${TESTDIR}/data/nipype.cfg
EDDY_CFG=${TESTDIR}/data/eddy_config.json

# For the run
setup_dir ${TESTDIR}/${TESTNAME}
TEMPDIR=${TESTDIR}/${TESTNAME}/work
OUTPUT_DIR=${TESTDIR}/${TESTNAME}/derivatives
BIDS_INPUT_DIR=${TESTDIR}/data/DSDTI
export FS_LICENSE=${TESTDIR}/data/license.txt
QSIPREP_CMD=$(run_qsiprep_cmd ${BIDS_INPUT_DIR} ${OUTPUT_DIR})

# Do the anatomical run on its own
${QSIPREP_CMD} \
	-w ${TEMPDIR} \
	--sloppy \
	--unringing-method mrdegibbs \
	--b1-biascorrect-stage legacy \
	--eddy-config ${EDDY_CFG} \
	--output-resolution 5 \
    -vv


