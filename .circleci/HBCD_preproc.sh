#!/bin/bash

cat << DOC

Test paired DWI series with DRBUDDI
===================================

This tests the following features:
 - Eddy is run on a CPU
 - DRBUDDI is run with two DWI series

DOC

set +e
source ./get_data.sh
TESTDIR=${PWD}
TESTNAME=HBCD
get_config_data ${TESTDIR}
get_bids_data ${TESTDIR} HBCD
CFG=${TESTDIR}/data/nipype.cfg
EDDY_CFG=${TESTDIR}/data/eddy_config.json

# For the run
setup_dir ${TESTDIR}/${TESTNAME}
TEMPDIR=${TESTDIR}/${TESTNAME}/work
OUTPUT_DIR=${TESTDIR}/${TESTNAME}/derivatives
BIDS_INPUT_DIR=${TESTDIR}/data/hbcd_sim
export FS_LICENSE=${TESTDIR}/data/license.txt
QSIPREP_CMD=$(run_qsiprep_cmd ${BIDS_INPUT_DIR} ${OUTPUT_DIR})

# Do the HBCD-style run
${QSIPREP_CMD} \
	-w ${TEMPDIR} \
	--sloppy \
	--anat-modality T2w \
	--infant \
	--denoise-method dwidenoise \
	--b1-biascorrect-stage none \
	--pepolar-method DRBUDDI \
	--eddy-config ${EDDY_CFG} \
	--output-resolution 5 \
    -vv --stop-on-first-crash


