#!/bin/bash

cat << DOC

Reconstruction workflow tests for ingressed data
==================================================

Process a UKB dataset through a recon workflow

Inputs:
-------

 - ukb processed data (data/ukb)

DOC
set +e
source ./get_data.sh
TESTDIR=${PWD}
get_config_data ${TESTDIR}
# get_bids_data ${TESTDIR} ukb_output
CFG=${TESTDIR}/data/nipype.cfg
export FS_LICENSE=${TESTDIR}/data/license.txt

# Test MRtrix3 multishell msmt with ACT
TESTNAME=ukb_ingress_recon
setup_dir ${TESTDIR}/${TESTNAME}
TEMPDIR=${TESTDIR}/${TESTNAME}/work
OUTPUT_DIR=${TESTDIR}/${TESTNAME}/derivatives
BIDS_INPUT_DIR=${TESTDIR}/data/ukb

qsiprep-docker -i pennbbl/qsiprep:unstable \
	-e qsiprep_DEV 1 -u $(id -u) \
	--patch-qsiprep /Users/mcieslak/projects/qsiprep/qsiprep \
	--config ${CFG} ${PATCH} -w ${TEMPDIR} \
	 ${BIDS_INPUT_DIR} ${OUTPUT_DIR} \
	 participant \
	 --recon-input ${BIDS_INPUT_DIR} \
	 --recon-input-pipeline ukb \
	 --participant-label 1234567 \
	 --sloppy \
     --stop-on-first-crash \
	 --recon-spec dsi_studio_autotrack \
	 --recon-only \
	 --mem_mb 4096 \
	 --nthreads 1 -vv

