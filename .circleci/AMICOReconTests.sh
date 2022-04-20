#!/bin/bash

cat << DOC

Reconstruction workflow tests
=============================

All supported reconstruction workflows get tested

This tests the following features:
 - Blip-up + Blip-down DWI series for TOPUP/Eddy
 - Eddy is run on a CPU
 - Denoising is skipped
 - A follow-up reconstruction using the dsi_studio_gqi workflow

Inputs:
-------

 - qsiprep single shell results (data/DSDTI_fmap)
 - qsiprep multi shell results (data/DSDTI_fmap)

DOC
set +e
source ./get_data.sh
TESTDIR=${PWD}
get_config_data ${TESTDIR}
get_bids_data ${TESTDIR} singleshell_output
get_bids_data ${TESTDIR} multishell_output
CFG=${TESTDIR}/data/nipype.cfg
EDDY_CFG=${TESTDIR}/data/eddy_config.json
export FS_LICENSE=${TESTDIR}/data/license.txt


# Test amico_noddi
TESTNAME=amico_noddi_test
setup_dir ${TESTDIR}/${TESTNAME}
TEMPDIR=${TESTDIR}/${TESTNAME}/work
OUTPUT_DIR=${TESTDIR}/${TESTNAME}/derivatives
BIDS_INPUT_DIR=${TESTDIR}/data/multishell_output/qsiprep
QSIPREP_CMD=$(run_qsiprep_cmd ${BIDS_INPUT_DIR} ${OUTPUT_DIR})

${QSIPREP_CMD}  \
	 -w ${TEMPDIR} \
	 --recon-input ${BIDS_INPUT_DIR} \
	 --sloppy \
	 --recon-spec amico_noddi \
	 --recon-only \
	 -vv
