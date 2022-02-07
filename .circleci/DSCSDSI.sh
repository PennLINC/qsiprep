#!/bin/bash

cat << DOC

DSCSDSI test
============

This tests the following features:
 - Whether the --anat-only workflow is successful
 - Whether the regular qsiprep workflow can resume using the
   working directory from --anat-only
 - The SHORELine motion correction workflow
 - Skipping B1 biascorrection
 - Using the SyN-SDC distortion correction method

Inputs:
-------

 - DSCSDSI BIDS data (data/DSCSDSI_nofmap)

DOC

# Setup environment and get data
source ./get_data.sh
TESTDIR=${PWD}
TESTNAME=DSCSDSI
get_config_data ${TESTDIR}
get_bids_data ${TESTDIR} DSCSDSI
CFG=${TESTDIR}/data/nipype.cfg
QSIPREP_CMD=$(run_qsiprep_cmd ${CFG})

# For the run
setup_dir ${TESTDIR}/${TESTNAME}
TEMPDIR=${TESTDIR}/${TESTNAME}/work
OUTPUT_DIR=${TESTDIR}/${TESTNAME}/derivatives
BIDS_INPUT_DIR=${TESTDIR}/data/DSCSDSI_nofmap
export FS_LICENSE=${TESTDIR}/data/license.txt

# name: Run full qsiprep on DSCSDSI
${QSIPREP_CMD} \
   ${BIDS_INPUT_DIR} ${OUTPUT_DIR} participant \
   -w ${TEMPDIR} \
   --sloppy --write-graph --use-syn-sdc \
   --force-syn \
   --output-space T1w \
   --dwi-no-biascorr \
   --hmc_model 3dSHORE \
   --hmc-transform Rigid \
   --shoreline_iters 1 \
   --output-resolution 5 \
   --stop-on-first-crash \
   -vv

