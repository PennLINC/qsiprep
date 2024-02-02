#!/bin/bash

cat << DOC

Test the TORTOISE recon workflow
=================================

All supported reconstruction workflows get tested


Inputs:
-------
 - qsiprep multi shell results (data/DSDTI_fmap)

DOC
set +e
source ./get_data.sh
TESTDIR=${PWD}
TESTNAME=scalar_mapper_test
get_config_data ${TESTDIR}
get_bids_data ${TESTDIR} multishell_output
CFG=${TESTDIR}/data/nipype.cfg

# Test MRtrix3 multishell msmt with ACT
setup_dir ${TESTDIR}/${TESTNAME}
TEMPDIR=${TESTDIR}/${TESTNAME}/work
OUTPUT_DIR=${TESTDIR}/${TESTNAME}/derivatives
BIDS_INPUT_DIR=${TESTDIR}/data/multishell_output/qsiprep
export FS_LICENSE=${TESTDIR}/data/license.txt
QSIPREP_CMD=$(run_qsiprep_cmd ${BIDS_INPUT_DIR} ${OUTPUT_DIR})

${QSIPREP_CMD} \
   -w ${TEMPDIR} \
	--recon-input ${BIDS_INPUT_DIR} \
	--sloppy \
    --stop-on-first-crash \
	--recon-spec test_scalar_maps \
	--recon-only \
	-vv