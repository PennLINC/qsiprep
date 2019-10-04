WORKDIR=$(mktemp -d "${TMPDIR:-${WORKDIR}}/qsiprep.XXXXXXXXX")
mkdir -p ${WORKDIR}/data
cd ${WORKDIR}/data
if [[ ! -d ${WORKDIR}/data/DSCSDSI ]]; then
  wget --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 0 -q \
    -O DSCSDSI_nofmap.tar.xz "https://upenn.box.com/shared/static/eq6nvnyazi2zlt63uowqd0zhnlh6z4yv.xz"
  tar xvfJ DSCSDSI_nofmap.tar.xz -C ${WORKDIR}/data/
else
  echo "Dataset DSCSDSI was cached"
fi
# name: Get BUDS scans from downsampled CS-DSI
# command: |
  if [[ ! -d ${WORKDIR}/data/DSCSDSI_BUDS ]]; then
    wget --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 0 -q \
      -O dscsdsi_buds.tar.xz "https://upenn.box.com/shared/static/bvhs3sw2swdkdyekpjhnrhvz89x3k87t.xz"
    tar xvfJ dscsdsi_buds.tar.xz -C ${WORKDIR}/data/
  else
    echo "Dataset DSCSDSI_BUDS was cached"
  fi

# name: Get downsampled DTI
# command: |
  if [[ ! -d ${WORKDIR}/data/DSDTI ]]; then
    wget --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 0 -q \
      -O DSDTI.tar.xz "https://upenn.box.com/shared/static/iefjtvfez0c2oug0g1a9ulozqe5il5xy.xz"
    tar xvfJ DSDTI.tar.xz -C ${WORKDIR}/data/
    wget --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 0 -q \
      -O ${WORKDIR}/data/eddy_config.json \
      "https://upenn.box.com/shared/static/93g89mug6cejzn6jtq0v7wkupz86iafh.json"
    chmod a+r ${WORKDIR}/data/eddy_config.json
  else
    echo "Dataset DSDTI was cached"
  fi

# name: Get multisession CS-DSI
# command: |
  if [[ ! -d ${WORKDIR}/data/twoses ]]; then
    wget --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 0 -q \
      -O twoses.tar.xz "https://upenn.box.com/shared/static/c949fjjhhen3ihgnzhkdw5jympm327pp.xz"
    tar xvfJ twoses.tar.xz -C ${WORKDIR}/data/
  else
    echo "Dataset twoses was cached"
  fi

# name: Store FreeSurfer license file
# command: |
  mkdir -p ${WORKDIR}/fslicense
  cd ${WORKDIR}/fslicense
  echo "cHJpbnRmICJtYXR0aGV3LmNpZXNsYWtAcHN5Y2gudWNzYi5lZHVcbjIwNzA2XG4gKkNmZVZkSDVVVDhyWVxuIEZTQllaLlVrZVRJQ3dcbiIgPiBsaWNlbnNlLnR4dAo=" | base64 -d | sh

# name: Create Nipype config files
# command: |
  mkdir -p ${WORKDIR}/DSCSDSI_BUDS ${WORKDIR}/DSCSDSI ${WORKDIR}/DSDTI ${WORKDIR}/twoses
  printf "[execution]\nstop_on_first_crash = true\n" > ${WORKDIR}/DSCSDSI/nipype.cfg
  echo "poll_sleep_duration = 0.01" >> ${WORKDIR}/DSCSDSI/nipype.cfg
  echo "hash_method = content" >> ${WORKDIR}/DSCSDSI/nipype.cfg
  cp ${WORKDIR}/DSCSDSI/nipype.cfg ${WORKDIR}/DSCSDSI_BUDS/nipype.cfg
  cp ${WORKDIR}/DSCSDSI/nipype.cfg ${WORKDIR}/DSDTI/nipype.cfg
  cp ${WORKDIR}/DSCSDSI/nipype.cfg ${WORKDIR}/twoses/nipype.cfg


#  DSCSDSI:

cd ${WORKDIR}/DSCSDSI

export FS_LICENSE=${WORKDIR}/fslicense/license.txt

# name: Run anatomical workflow on DSCSDSI
# command: |
  mkdir -p ${WORKDIR}/DSCSDSI/work ${WORKDIR}/DSCSDSI/derivatives
  qsiprep \
      --stop-on-first-crash \
      -w ${WORKDIR}/DSCSDSI/work \
      --bids-dir ${WORKDIR}/data/DSCSDSI_nofmap --output-dir ${WORKDIR}/DSCSDSI/derivatives \
      --analysis-level participant \
      --force-spatial-normalization \
      --sloppy --write-graph --mem_mb 4096 \
      --fs-license-file $FREESURFER_HOME/license.txt \
      --anat-only -vv --output-resolution 5

# name: Run full qsiprep on DSCSDSI
# command: |
  qsiprep \
      -w ${WORKDIR}/DSCSDSI/work \
      --bids-dir ${WORKDIR}/data/DSCSDSI_nofmap --output-dir ${WORKDIR}/DSCSDSI/derivatives \
      --analysis-level participant \
      --sloppy --write-graph --use-syn-sdc --force-syn --mem_mb 4096 \
      --output-space T1w \
      --hmc_model 3dSHORE \
      --hmc-transform Rigid \
      --shoreline_iters 1 \
      --force-spatial-normalization \
      --output-resolution 5 \
      --fs-license-file $FREESURFER_HOME/license.txt \
      -vv

# name: Run DIPY 3dSHORE recon on outputs
# command: |
  qsiprep -w ${WORKDIR}/DSCSDSI/work \
      --bids-dir ${WORKDIR}/data/DSCSDSI_nofmap \
      --recon-input ${WORKDIR}/DSCSDSI/derivatives/qsiprep \
      --output-dir ${WORKDIR}/DSCSDSI/derivatives \
      --analysis-level participant \
      --recon-spec ${HOME}/projects/qsiprep/.circleci/3dshore_dsistudio_mrtrix.json \
      --recon-only \
      --mem_mb 4096 \
      --output-resolution 5 \
      --fs-license-file $FREESURFER_HOME/license.txt \
      --nthreads 2 -vv


#  DSDTI:

# name: Run anatomical workflow on DSDTI
# command: |
  mkdir -p ${WORKDIR}/DSDTI/work ${WORKDIR}/DSDTI/derivatives
  qsiprep \
      -w ${WORKDIR}/DSDTI/work \
      --bids-dir ${WORKDIR}/data/DSDTI \
      --output-dir ${WORKDIR}/DSDTI/derivatives \
      --analysis-level participant \
      --sloppy --write-graph --mem_mb 4096 \
      --fs-license-file $FREESURFER_HOME/license.txt \
      --anat-only -vv --output-resolution 5

# name: Run full qsiprep (eddy) on DSDTI
# command: |
  qsiprep \
      -w ${WORKDIR}/DSDTI/work \
      --bids-dir ${WORKDIR}/data/DSDTI --output-dir ${WORKDIR}/DSDTI/derivatives \
      --analysis-level participant \
      --sloppy --mem_mb 4096 \
      --output-space T1w \
      --hmc_model eddy \
      --force-spatial-normalization \
      --eddy_config ${WORKDIR}/data/eddy_config.json \
      --fs-license-file $FREESURFER_HOME/license.txt \
      --output-resolution 5 \
      --nthreads 2 -vv

# name: run mrtrix3 connectome workflow on DSDTI
# command: |
  qsiprep -w ${WORKDIR}/DSDTI/work \
      --bids-dir ${WORKDIR}/data/DSDTI \
      --recon-input ${WORKDIR}/DSDTI/derivatives/qsiprep \
      --output-dir ${WORKDIR}/DSDTI/derivatives \
      --analysis-level participant \
      --recon-spec ${HOME}/projects/qsiprep/.circleci/mrtrix_msmt_csd_test.json \
      --recon-only \
      --mem_mb 4096 \
      --output-resolution 5 \
      --fs-license-file $FREESURFER_HOME/license.txt \
      --nthreads 2 -vv

# IntramodalTemplate:
# name: Run full qsiprep with an intramodal template
# command: |
  mkdir -p ${WORKDIR}/twoses/work ${WORKDIR}/twoses/derivatives
  qsiprep \
      -w ${WORKDIR}/twoses/work \
      --bids-dir ${WORKDIR}/data/twoses --output-dir ${WORKDIR}/twoses/derivatives \
      --analysis-level participant \
      --sloppy --mem_mb 4096 \
      --output-space T1w \
      --hmc_model none \
      --b0-motion-corr-to first \
      --output-resolution 5 \
      --intramodal-template-transform BSplineSyN \
      --fs-license-file $FREESURFER_HOME/license.txt \
      --intramodal-template-iters 2 \
      -vv

#BlipUpDownSeries:
#name: Run full qsiprep on DSCSDSI_BUDS
#command: |
mkdir -p ${WORKDIR}/DSCSDSI_BUDS/work ${WORKDIR}/DSCSDSI_BUDS/derivatives
  qsiprep \
      -w ${WORKDIR}/DSCSDSI_BUDS/work \
      --bids-dir ${WORKDIR}/data/DSCSDSI_BUDS --output-dir ${WORKDIR}/DSCSDSI_BUDS/derivatives \
      --analysis-level participant \
      --sloppy --write-graph --mem_mb 4096 \
      --output-space T1w \
      --hmc_model none \
      --hmc-transform Rigid \
      --output-resolution 5 \
      --fs-license-file $FREESURFER_HOME/license.txt \
      --nthreads 2 -vv
