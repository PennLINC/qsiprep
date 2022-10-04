#!/bin/bash

cat << DOC

Test paired DWI series
======================

This tests the following features:
 - Eddy is run on a CPU
 - A follow-up reconstruction using the dsi_studio_gqi workflow

Inputs:
-------

 - DSDTI BIDS data (data/DSDTI)

DOC

set +e
source ./get_data.sh
TESTDIR=${PWD}
TESTNAME=HBCD_DRBUDDI
get_config_data ${TESTDIR}
get_bids_data ${TESTDIR} tinytensors
CFG=${TESTDIR}/data/nipype.cfg
EDDY_CFG=${TESTDIR}/data/eddy_config.json

# For the run
setup_dir ${TESTDIR}/${TESTNAME}
TEMPDIR=${TESTDIR}/${TESTNAME}/work
OUTPUT_DIR=${TESTDIR}/${TESTNAME}/derivatives
BIDS_INPUT_DIR=${TESTDIR}/data/tinytensor
export FS_LICENSE=${TESTDIR}/data/license.txt
QSIPREP_CMD=$(run_qsiprep_cmd ${BIDS_INPUT_DIR} ${OUTPUT_DIR})

# Do the anatomical run on its own
${QSIPREP_CMD} \
	-w ${TEMPDIR} \
	--sloppy \
	--dwi-only \
	--output-space T1w \
	--pepolar-method DRBUDDI \
	--eddy_config ${EDDY_CFG} \
	--output-resolution 2 \
    -vv --stop-on-first-crash


