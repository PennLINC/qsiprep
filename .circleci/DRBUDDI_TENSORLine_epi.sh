#!/bin/bash

cat << DOC

Test EPI fieldmap correction with TENSORLine + DRBUDDI
======================================================

This tests the following features:
 - TENSORLine (tensor-based) motion correction

DOC

set +e
source ./get_data.sh
TESTDIR=${PWD}
TESTNAME=DRBUDDI_TENSORLINE_EPI
get_config_data ${TESTDIR}
get_bids_data ${TESTDIR} DSDTI
CFG=${TESTDIR}/data/nipype.cfg

# For the run
setup_dir ${TESTDIR}/${TESTNAME}
TEMPDIR=${TESTDIR}/${TESTNAME}/work
OUTPUT_DIR=${TESTDIR}/${TESTNAME}/derivatives
BIDS_INPUT_DIR=${TESTDIR}/data/DSDTI
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
	--hmc-model tensor \
	--output-resolution 2 \
	--shoreline-iters 1 \
    -vv --stop-on-first-crash