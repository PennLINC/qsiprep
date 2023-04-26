#!/bin/bash

cat << DOC

Test EPI fieldmap correction with SHORELine + DRBUDDI
=====================================================

This tests the following features:
 - SHORELine (here, just b=0 registration) motion correction
 -

DOC

set +e
source ./get_data.sh
TESTDIR=${PWD}
TESTNAME=DRBUDDI_SHORELINE_EPI
get_config_data ${TESTDIR}
get_bids_data ${TESTDIR} drbuddi_epi
CFG=${TESTDIR}/data/nipype.cfg

# For the run
setup_dir ${TESTDIR}/${TESTNAME}
TEMPDIR=${TESTDIR}/${TESTNAME}/work
OUTPUT_DIR=${TESTDIR}/${TESTNAME}/derivatives
BIDS_INPUT_DIR=${TESTDIR}/data/tinytensor_epi
export FS_LICENSE=${TESTDIR}/data/license.txt
QSIPREP_CMD=$(run_qsiprep_cmd ${BIDS_INPUT_DIR} ${OUTPUT_DIR})

# Do the anatomical run on its own
${QSIPREP_CMD} \
	-w ${TEMPDIR} \
	--sloppy \
	--anat-modality none \
	--denoise-method none \
	--b1-biascorrect-stage none \
	--pepolar-method DRBUDDI \
	--hmc-model none \
	--output-resolution 2 \
	--shoreline-iters 1 \
    -vv --stop-on-first-crash


