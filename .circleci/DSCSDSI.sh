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
set +e
# Setup environment and get data
source ./get_data.sh
TESTDIR=${PWD}
TESTNAME=DSCSDSI
get_config_data ${TESTDIR}
get_bids_data ${TESTDIR} DSCSDSI
CFG=${TESTDIR}/data/nipype.cfg

# For the run
setup_dir ${TESTDIR}/${TESTNAME}
TEMPDIR=${TESTDIR}/${TESTNAME}/work
OUTPUT_DIR=${TESTDIR}/${TESTNAME}/derivatives
BIDS_INPUT_DIR=${TESTDIR}/data/DSCSDSI_nofmap
export FS_LICENSE=${TESTDIR}/data/license.txt
QSIPREP_CMD=$(run_qsiprep_cmd ${BIDS_INPUT_DIR} ${OUTPUT_DIR})

# name: Run full qsiprep on DSCSDSI
${QSIPREP_CMD} \
   -w ${TEMPDIR} \
   --sloppy --write-graph --use-syn-sdc \
   --force-syn \
   --b1-biascorrect-stage none \
   --hmc-model 3dSHORE \
   --hmc-transform Rigid \
   --shoreline-iters 1 \
   --output-resolution 5 \
   --stop-on-first-crash \
   -vv

