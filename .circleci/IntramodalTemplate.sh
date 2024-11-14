#!/bin/bash

cat << DOC

IntramodalTemplate test
=======================

A two-session dataset is used to create an intramodal template.

This tests the following features:
 - Blip-up + Blip-down DWI series for TOPUP/Eddy
 - Eddy is run on a CPU
 - Denoising is skipped
 - A follow-up reconstruction using the dsi_studio_gqi workflow

Inputs:
-------

 - twoses BIDS data (data/DSDTI_fmap)

DOC
set +e

source ./get_data.sh
TESTDIR=${PWD}
get_config_data ${TESTDIR}
get_bids_data ${TESTDIR} twoses
CFG=${TESTDIR}/data/nipype.cfg
EDDY_CFG=${TESTDIR}/data/eddy_config.json
export FS_LICENSE=${TESTDIR}/data/license.txt

# Test blip-up blip-down shelled series (TOPUP/eddy)
TESTNAME=imtemplate
setup_dir ${TESTDIR}/${TESTNAME}
TEMPDIR=${TESTDIR}/${TESTNAME}/work
OUTPUT_DIR=${TESTDIR}/${TESTNAME}/derivatives
BIDS_INPUT_DIR=${TESTDIR}/data/twoses/twoses
QSIPREP_CMD=$(run_qsiprep_cmd ${BIDS_INPUT_DIR} ${OUTPUT_DIR})

${QSIPREP_CMD} \
	 -w ${TEMPDIR} \
	 --sloppy \
	 --b1-biascorrect-stage none \
	 --hmc-model none \
	 --b0-motion-corr-to first \
	 --output-resolution 5 \
	 --intramodal-template-transform BSplineSyN \
	 --intramodal-template-iters 2 \
	 -vv
