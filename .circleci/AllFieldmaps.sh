#!/bin/bash

cat << DOC

AllFieldmaps test
=================

Instead of running full workflows, this test checks that workflows can
be built for all sorts of fieldmap configurations.

This tests the following features:
 - Blip-up + Blip-down DWI series for TOPUP/Eddy
 - Eddy is run on a CPU
 - Denoising is skipped

Inputs:
-------

 - DSDTI BIDS data (data/DSDTI_fmap)

DOC
set +e
source ./get_data.sh
TESTDIR=${PWD}
get_config_data ${TESTDIR}
CFG=${TESTDIR}/data/nipype.cfg
EDDY_CFG=${TESTDIR}/data/eddy_config.json
export FS_LICENSE=${TESTDIR}/data/license.txt
get_bids_data ${TESTDIR} fmaps

# Test blip-up blip-down shelled series (TOPUP/eddy)
TESTNAME=DTI_SDC
setup_dir ${TESTDIR}/${TESTNAME}
TEMPDIR=${TESTDIR}/${TESTNAME}/work
OUTPUT_DIR=${TESTDIR}/${TESTNAME}/derivatives
BIDS_INPUT_DIR=${TESTDIR}/data/fmaptests/DSDTI_fmap
QSIPREP_CMD=$(run_qsiprep_cmd ${BIDS_INPUT_DIR} ${OUTPUT_DIR})

${QSIPREP_CMD} \
	 -w ${TEMPDIR} \
	 --boilerplate \
	 --sloppy --write-graph --mem-mb 4096 \
	 -vv --output-resolution 5

# Test blip-up blip-down non-shelled series (SHORELine/sdcflows)
TESTNAME=DSI_SDC
setup_dir ${TESTDIR}/${TESTNAME}
TEMPDIR=${TESTDIR}/${TESTNAME}/work
OUTPUT_DIR=${TESTDIR}/${TESTNAME}/derivatives
BIDS_INPUT_DIR=${TESTDIR}/data/fmaptests/DSCSDSI_fmap
QSIPREP_CMD=$(run_qsiprep_cmd ${BIDS_INPUT_DIR} ${OUTPUT_DIR})

# Test blip-up blip-down shelled series (TOPUP/eddy)
${QSIPREP_CMD} \
	 -w ${TEMPDIR} \
	 --boilerplate \
     --hmc-model 3dSHORE \
	 --sloppy --write-graph --mem-mb 4096 \
	 -vv --output-resolution 5




