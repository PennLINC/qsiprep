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

# name: Get all kinds of fieldmaps for DTI data
# command: |
if [[ ! -d ${WORKDIR}/data/fmaptests/DSDTI_fmap ]]; then
  mkdir -p ${WORKDIR}/data/fmaptests/
  wget --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 0 -q \
    -O DSDTI_fmap.tar.xz "https://upenn.box.com/shared/static/rxr6qbi6ezku9gw3esfpnvqlcxaw7n5n.gz"
  tar xvfJ DSDTI_fmap.tar.xz -C ${WORKDIR}/data/fmaptests
else
  echo "Dataset DSDTI_fmap was cached"
fi

# name: Get all kinds of fieldmaps for DSI data
# command: |
if [[ ! -d ${WORKDIR}/data/fmaptests/DSCSDSI_fmap ]]; then
  mkdir -p ${WORKDIR}/data/fmaptests/
  wget --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 0 -q \
    -O DSCSDSI_fmap.tar.xz "https://upenn.box.com/shared/static/l561psez1ojzi4p3a12eidaw9vbizwdc.gz"
  tar xvfJ DSCSDSI_fmap.tar.xz -C ${WORKDIR}/data/fmaptests
else
  echo "Dataset DSCSDSI_fmap was cached"
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

#  name: Get MultiShell outputs
#  command: |
    if [[ ! -d ${WORKDIR}/data/multishell_output ]]; then
      wget --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 0 -q \
        -O multishell_output.tar.gz "https://upenn.box.com/shared/static/nwxdn4ale8dkebvpjmxbx99dqjzwvlmh.gz"
      tar xvfz multishell_output.tar.gz -C ${WORKDIR}/data/
    else
      echo "Dataset multishell_output was cached"
    fi

# get singleshell output
if [[ ! -d ${WORKDIR}/data/singleshell_output ]]; then
  mkdir -p ${WORKDIR}/data/singleshell_output
  wget --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 0 -q \
    -O singleshell_output.tar.gz "https://upenn.box.com/shared/static/9jhf0eo3ml6ojrlxlz6lej09ny12efgg.gz"
  tar xvfz singleshell_output.tar.gz -C ${WORKDIR}/data/singleshell_output
else
  echo "Dataset singleshell_output was cached"
fi

# get multishell output

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
      ${WORKDIR}/data/DSCSDSI_nofmap ${WORKDIR}/DSCSDSI/derivatives \
      participant \
      --force-spatial-normalization \
      --sloppy --write-graph --mem_mb 4096 \
      --fs-license-file $FREESURFER_HOME/license.txt \
      --anat-only -vv --output-resolution 5

# name: Run full qsiprep on DSCSDSI
# command: |
  qsiprep \
      -w ${WORKDIR}/DSCSDSI/work \
       ${WORKDIR}/data/DSCSDSI_nofmap  ${WORKDIR}/DSCSDSI/derivatives \
       participant \
      --sloppy --write-graph --use-syn-sdc --force-syn --mem_mb 4096 \
      --output-space T1w \
      --hmc_model none \
      --hmc-transform Rigid \
      --shoreline_iters 1 \
      --force-spatial-normalization \
      --output-resolution 5 \
      --fs-license-file $FREESURFER_HOME/license.txt \
      -vv

# name: Run DIPY 3dSHORE recon on outputs
# command: |
  qsiprep -w ${WORKDIR}/DSCSDSI/work \
       ${WORKDIR}/data/DSCSDSI_nofmap \
      --recon-input ${WORKDIR}/DSCSDSI/derivatives/qsiprep \
       ${WORKDIR}/DSCSDSI/derivatives \
       participant \
      --recon-spec dipy_3dshore \
      --sloppy \
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
       ${WORKDIR}/data/DSDTI \
       ${WORKDIR}/DSDTI/derivatives \
       participant \
      --sloppy --write-graph --mem_mb 4096 \
      --fs-license-file $FREESURFER_HOME/license.txt \
      --anat-only -vv --output-resolution 5

# name: Run full qsiprep (eddy) on DSDTI
# command: |
  qsiprep \
      -w ${WORKDIR}/DSDTI/work \
       ${WORKDIR}/data/DSDTI  ${WORKDIR}/DSDTI/derivatives \
       participant \
      --sloppy --mem_mb 4096 \
      --output-space T1w \
      --hmc_model eddy \
      --stop_on_first_crash \
      --force-spatial-normalization \
      --eddy_config ${WORKDIR}/data/eddy_config.json \
      --fs-license-file $FREESURFER_HOME/license.txt \
      --output-resolution 5 \
      --nthreads 2 -vv

      # Run eddy without fieldmaps
      qsiprep \
          -w ${WORKDIR}/DSDTI/work \
           ${WORKDIR}/data/DSDTI  ${WORKDIR}/DSDTI/derivatives \
           participant \
          --sloppy --mem_mb 4096 \
          --output-space T1w \
          --ignore fieldmaps \
          --hmc_model eddy \
          --eddy_config ${WORKDIR}/data/eddy_config.json \
          --fs-license-file $FREESURFER_HOME/license.txt \
          --output-resolution 5 \
          --nthreads 2 -vv

          # Run eddy with syn
          qsiprep \
              -w ${WORKDIR}/DSDTI/work \
               ${WORKDIR}/data/DSDTI  ${WORKDIR}/DSDTI/derivatives \
               participant \
              --sloppy --mem_mb 4096 \
              --output-space T1w \
              --ignore fieldmaps \
              --force-syn \
              --hmc_model eddy \
              --eddy_config ${WORKDIR}/data/eddy_config.json \
              --fs-license-file $FREESURFER_HOME/license.txt \
              --output-resolution 5 \
              --nthreads 2 -vv

# name: run mrtrix3 connectome workflow on DSDTI
# command: |
  qsiprep -w ${WORKDIR}/DSDTI/work \
       ${WORKDIR}/data/DSDTI \
      --recon-input ${WORKDIR}/DSDTI/derivatives/qsiprep \
       ${WORKDIR}/DSDTI/derivatives \
       participant \
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
       ${WORKDIR}/data/twoses  ${WORKDIR}/twoses/derivatives \
       participant \
      --sloppy --mem_mb 4096 \
      --output-space T1w \
      --hmc_model none \
      --b0-motion-corr-to first \
      --output-resolution 5 \
      --intramodal-template-transform BSplineSyN \
      --fs-license-file $FREESURFER_HOME/license.txt \
      --intramodal-template-iters 2 \
      -vv

# AllFieldmaps:
mkdir -p ${WORKDIR}/DTI_SDC/work ${WORKDIR}/DTI_SDC/derivatives
qsiprep \
    -w ${WORKDIR}/DTI_SDC/work \
     ${WORKDIR}/data/fmaptests/DSDTI_fmap  ${WORKDIR}/DTI_SDC/derivatives \
     participant \
    --boilerplate \
    --sloppy --write-graph --mem_mb 4096 \
      --fs-license-file $FREESURFER_HOME/license.txt \
    --nthreads 2 -vv --output-resolution 5

qsiprep \
    -w ${WORKDIR}/DTI_SDC/work \
     ${WORKDIR}/data/fmaptests/DSDTI_fmap  ${WORKDIR}/DTI_SDC/derivatives \
     participant \
    --boilerplate \
    --combine-all-dwis \
      --fs-license-file $FREESURFER_HOME/license.txt \
    --sloppy --write-graph --mem_mb 4096 \
    --nthreads 2 -vv --output-resolution 5

# name: Test Fieldmap setups for DSI (using BUDS)
# command: |
mkdir -p ${WORKDIR}/DSI_SDC/work ${WORKDIR}/DSI_SDC/derivatives
qsiprep \
    -w ${WORKDIR}/DSI_SDC/work \
     ${WORKDIR}/data/fmaptests/DSCSDSI_fmap  ${WORKDIR}/DSI_SDC/derivatives \
     participant \
    --boilerplate \
      --fs-license-file $FREESURFER_HOME/license.txt \
    --sloppy --write-graph --mem_mb 4096 \
    --nthreads 2 -vv --output-resolution 5

qsiprep \
    -w ${WORKDIR}/DSI_SDC/work \
     ${WORKDIR}/data/fmaptests/DSCSDSI_fmap  ${WORKDIR}/DSI_SDC/derivatives \
     participant \
    --combine-all-dwis \
    --boilerplate \
      --fs-license-file $FREESURFER_HOME/license.txt \
    --sloppy --write-graph --mem_mb 4096 \
    --nthreads 2 -vv --output-resolution 5

# name: Run mrtrix on downsampled abcd
# command: |
  qsiprep -w /Users/mcieslak/Desktop/multishell_output/DSCSDSI/work \
       /Users/mcieslak/Desktop/multishell_output/qsiprep \
      --recon-input /Users/mcieslak/Desktop/multishell_output/qsiprep \
       /Users/mcieslak/Desktop/multishell_output/multishell_output/derivatives \
       participant \
      --recon-spec mrtrix_msmt_csd \
      --sloppy \
      --recon-only \
      --fs-license-file $FREESURFER_HOME/license.txt \
      --mem_mb 4096 \
      --output-resolution 5 \
      --fs-license-file $FREESURFER_HOME/license.txt \
      --nthreads 2 -vv

#- run:
#name: Run mrtrix_multishell_msmt
#no_output_timeout: 2h
#command: |
  mkdir -p ${WORKDIR}/multishell_output/work ${WORKDIR}/multishell_output/derivatives/mrtrix_multishell_msmt
  qsiprep -w ${WORKDIR}/multishell_output/work \
       ${WORKDIR}/data/multishell_output/qsiprep  \
       ${WORKDIR}/multishell_output/derivatives/mrtrix_multishell_msmt \
       participant \
      --recon-input ${WORKDIR}/data/multishell_output/qsiprep \
      --sloppy \
      --recon-spec mrtrix_multishell_msmt \
      --recon-only \
      --mem_mb 4096 \
      --fs-license-file $FREESURFER_HOME/license.txt \
      --nthreads 1 -vv

#- run:
#name: Run mrtrix_multishell_msmt_noACT
#no_output_timeout: 2h
#command: |
  mkdir -p ${WORKDIR}/multishell_output/work ${WORKDIR}/multishell_output/derivatives/mrtrix_multishell_msmt_noACT
  qsiprep -w ${WORKDIR}/multishell_output/work \
       ${WORKDIR}/data/multishell_output/qsiprep \
       ${WORKDIR}/multishell_output/derivatives/mrtrix_multishell_msmt_noACT \
       participant \
      --sloppy \
      --recon-input ${WORKDIR}/data/multishell_output/qsiprep \
      --recon-spec mrtrix_multishell_msmt_noACT \
      --recon-only \
      --fs-license-file $FREESURFER_HOME/license.txt \
      --mem_mb 4096 \
      --nthreads 1 -vv

#- run:
#name: Run mrtrix_singleshell_ss3t
#no_output_timeout: 2h
#command: |
  mkdir -p ${WORKDIR}/singleshell_output/work ${WORKDIR}/singleshell_output/derivatives/mrtrix_singleshell_ss3t
  qsiprep -w ${WORKDIR}/singleshell_output/work \
       ${WORKDIR}/data/singleshell_output/qsiprep \
       ${WORKDIR}/singleshell_output/derivatives/mrtrix_singleshell_ss3t \
       participant \
      --sloppy \
      --recon-input ${WORKDIR}/data/singleshell_output/qsiprep \
      --recon-spec mrtrix_singleshell_ss3t \
      --recon-only \
      --mem_mb 4096 \
      --fs-license-file $FREESURFER_HOME/license.txt \
      --nthreads 1 -vv

#- run:
#name: Run mrtrix_singleshell_ss3t_noACT
#no_output_timeout: 2h
#command: |
  mkdir -p ${WORKDIR}/singleshell_output/work ${WORKDIR}/singleshell_output/derivatives/mrtrix_singleshell_ss3t_noACT
  qsiprep -w ${WORKDIR}/singleshell_output/work \
       ${WORKDIR}/data/singleshell_output/qsiprep \
       ${WORKDIR}/singleshell_output/derivatives/mrtrix_singleshell_ss3t_noACT \
       participant \
      --sloppy \
      --recon-input ${WORKDIR}/data/singleshell_output/qsiprep \
      --recon-spec mrtrix_singleshell_ss3t_noACT \
      --fs-license-file $FREESURFER_HOME/license.txt \
      --recon-only \
      --mem_mb 4096 \
      --nthreads 1 -vv
