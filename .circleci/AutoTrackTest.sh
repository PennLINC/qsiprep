#!/bin/bash

cat << DOC

Reconstruction workflow tests
=============================

All supported reconstruction workflows get tested

This tests the following features:

Inputs:
-------

 - qsiprep single shell results (data/DSDTI_fmap)
 - qsiprep multi shell results (data/DSDTI_fmap)

DOC
set +e
source ./get_data.sh
TESTDIR=${PWD}
get_config_data ${TESTDIR}
get_bids_data ${TESTDIR} multishell_output
CFG=${TESTDIR}/data/nipype.cfg
EDDY_CFG=${TESTDIR}/data/eddy_config.json
export FS_LICENSE=${TESTDIR}/data/license.txt

# Test MRtrix3 multishell msmt with ACT
TESTNAME=autotrack
setup_dir ${TESTDIR}/${TESTNAME}
TEMPDIR=${TESTDIR}/${TESTNAME}/work
OUTPUT_DIR=${TESTDIR}/${TESTNAME}/derivatives
BIDS_INPUT_DIR=${TESTDIR}/data/multishell_output/qsiprep
QSIPREP_CMD=$(run_qsiprep_cmd ${BIDS_INPUT_DIR} ${OUTPUT_DIR})

${QSIPREP_CMD} \
	 -w ${TEMPDIR} \
	 --recon-input ${BIDS_INPUT_DIR} \
	 --sloppy \
     --stop-on-first-crash \
	 --recon-spec dsi_studio_autotrack \
	 --recon-only \
	 -vv
