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
 - A follow-up reconstruction using the dsi_studio_gqi workflow

Inputs:
-------

 - DSDTI BIDS data (data/DSDTI_fmap)

DOC

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

qsiprep-docker -i pennbbl/qsiprep:latest \
	-e qsiprep_DEV 1 -u $(id -u) \
	--config ${CFG} ${PATCH} -w ${TEMPDIR} \
	 ${BIDS_INPUT_DIR} ${OUTPUT_DIR} \
	 participant \
	 --boilerplate \
	 --sloppy --write-graph --mem_mb 4096 \
	 --nthreads 2 -vv --output-resolution 5

# Test blip-up blip-down non-shelled series (SHORELine/sdcflows)
TESTNAME=DSI_SDC
setup_dir ${TESTDIR}/${TESTNAME}
TEMPDIR=${TESTDIR}/${TESTNAME}/work
OUTPUT_DIR=${TESTDIR}/${TESTNAME}/derivatives
BIDS_INPUT_DIR=${TESTDIR}/data/fmaptests/DSCSDSI_fmap

# Test blip-up blip-down shelled series (TOPUP/eddy)
qsiprep-docker -i pennbbl/qsiprep:latest \
	-e qsiprep_DEV 1 -u $(id -u) \
	--config ${CFG} ${PATCH} -w ${TEMPDIR} \
	 ${BIDS_INPUT_DIR} ${OUTPUT_DIR} \
	 participant \
	 --boilerplate \
     --hmc-model 3dSHORE \
	 --sloppy --write-graph --mem_mb 4096 \
	 --nthreads 2 -vv --output-resolution 5




