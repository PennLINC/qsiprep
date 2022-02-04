#!/bin/bash

cat << DOC

DSCDTI_nofmap test
==================

This tests the following features:
 - A workflow with no distortion correction followed by eddy
 - Eddy is run on a CPU
 - Denoising is skipped
 - A follow-up reconstruction using the dsi_studio_gqi workflow

Inputs:
-------

 - DSDTI BIDS data (data/DSDTI)

DOC

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

# CRITICAL: delete the fieldmap data
rm -rf data/DSDTI/sub-PNC/fmap

# Do the anatomical run on its own
qsiprep-docker -i pennbbl/qsiprep:latest \
	-e qsiprep_DEV 1 -u $(id -u) \
	--config ${CFG} ${PATCH} -w ${TEMPDIR} \
	 ${BIDS_INPUT_DIR} ${OUTPUT_DIR} \
	 participant \
     --eddy-config ${EDDY_CFG} \
     --denoise-method none \
     --sloppy --mem_mb 4096 \
     --output-space T1w \
     --output-resolution 5 \
     --nthreads ${NTHREADS} -vv

rm -rf ${TESTDIR}/${TESTNAME}
setup_dir ${TESTDIR}/${TESTNAME}

qsiprep-docker -i pennbbl/qsiprep:latest \
	-e qsiprep_DEV 1 -u $(id -u) \
	--config ${CFG} ${PATCH} -w ${TEMPDIR} \
	 ${BIDS_INPUT_DIR} ${OUTPUT_DIR} \
	 participant \
     --eddy-config ${EDDY_CFG} \
     --sloppy --mem_mb 4096 \
	 --force-syn \
     --denoise-method none \
	 --output-space T1w \
	 --output-resolution 5 \
	 --nthreads ${NTHREADS} -vv



